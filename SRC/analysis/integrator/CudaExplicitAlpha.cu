/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
**                                                                    **
** (C) Copyright 1999, The Regents of the University of California    **
** All Rights Reserved.                                               **
**                                                                    **
** Commercial use of this program without express permission of the   **
** University of California, Berkeley, is strictly prohibited.  See   **
** file 'COPYRIGHT'  in main directory for information on usage and   **
** redistribution,  and for a DISCLAIMER OF ALL WARRANTIES.           **
**                                                                    **
** Developed by:                                                      **
**   Frank McKenna (fmckenna@ce.berkeley.edu)                         **
**   Gregory L. Fenves (fenves@ce.berkeley.edu)                       **
**   Filip C. Filippou (filippou@ce.berkeley.edu)                     **
**                                                                    **
** ****************************************************************** */

#include <CudaExplicitAlpha.h>
#include <CudaGenBcsrLinSOE.h>
#include <AnalysisModel.h>
#include <Channel.h>
#include <DOF_Group.h>
#include <DOF_GrpIter.h>
#include <FE_EleIter.h>
#include <FE_Element.h>
#include <FEM_ObjectBroker.h>
#include <Integrator.h>
#include <Matrix.h>
#include <OPS_Globals.h>
#include <LinearSOE.h>
#include <Vector.h>
#include <classTags.h>
#include <elementAPI.h>
#include <cmath>
#include <Domain.h>
#include <Element.h>
#include <Node.h>
#include <cstring>

#include "CudaCsrMatrix.h"
#include "CudaUtils.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/memory.h>
#include <vector>

using thrust::raw_pointer_cast;

using namespace CudaUtils;

// Kolay-Ricles explicit integrator on GPU (CudaGenBcsrLinSOE + cuDSS).
// Device scalar type T follows SOE precision (dDDI=double, dFFI=float); host stays double.

namespace {

// --- Templated device kernels ---

template<typename T>
__global__ void kernelAxpy(int n, T alpha, const T *x, T *y)
{
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        y[i] += alpha * x[i];
    }
}

template<typename T>
__global__ void kernelNewStep(int n, T dt, T gamma, T alphaF, T alphaM, const T *alphaSol2, T *U, T *Udot,
                              const T *Uddot, T *alphaSol1, T *Ualpha, T *Ualphadot, T *Ualphadotdot,
                              int incrementalAccel, int alphaClose)
{
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const T ut = U[i];
        const T utdot = Udot[i];
        const T utddot = Uddot[i];
        const T uhat = dt * alphaSol1[i];
        alphaSol1[i] = uhat;
        const T un1 = ut + dt * utdot + (static_cast<T>(0.5) + gamma) * dt * uhat;
        const T udotn1 = utdot + uhat;
        U[i] = un1;
        Udot[i] = udotn1;
        Ualpha[i] = (static_cast<T>(1.0) - alphaF) * ut + alphaF * un1;
        Ualphadot[i] = (static_cast<T>(1.0) - alphaF) * utdot + alphaF * udotn1;
        if (incrementalAccel) {
            Ualphadotdot[i] = utddot;
        } else if (alphaClose) {
            Ualphadotdot[i] = (static_cast<T>(1.0) - alphaM) * utddot;
        } else {
            Ualphadotdot[i] = utddot - alphaSol2[i];
        }
    }
}

static int gridBlocks(int n, int blockSize = 256)
{
    return (n + blockSize - 1) / blockSize;
}

} // namespace

static constexpr int kMotionFields = 3;

struct CudaExplicitAlpha::ImplBase {
    struct HostMotionPtrs {
        double *U;
        double *Udot;
        double *Uddot;
        double *Ut;
        double *Utdot;
        double *Utdotdot;
        double *Ualpha;
        double *Ualphadot;
        double *Ualphadotdot;
    };

    virtual ~ImplBase() = default;

    virtual void allocate(int n, int alphaNumRhs) = 0;
    virtual void destroySolvers() = 0;
    virtual void shutdown() = 0;
    virtual void ensureAlphaBuffers(int alphaNumRhs) = 0;
    virtual int formOperators(CudaExplicitAlpha *integrator, CudaGenBcsrLinSOE *cudaSOE) = 0;
    virtual HostMotionPtrs ensureHostMotionBuffers(int n) = 0;
    virtual int newStepPredictor(CudaExplicitAlpha *integrator) = 0;
    virtual int formUnbalance(CudaGenBcsrLinSOE *cudaSOE) = 0;
    virtual int useSolverStream(CudaGenBcsrLinSOE *cudaSOE) = 0;
    virtual int updateState(CudaGenBcsrLinSOE *cudaSOE, bool incrementalAccel) = 0;
    virtual void zeroState() = 0;
    virtual int getSize() const = 0;
};

template<typename T>
struct ImplT : CudaExplicitAlpha::ImplBase {
    int size = 0;
    int alphaNumRhs = 2;

    thrust::device_vector<T> d_state_cur;    // [U | Udot | Uddot]
    thrust::device_vector<T> d_state_alpha;  // [Ualpha | Ualphadot | Ualphadotdot]
    thrust::device_vector<T> d_w2;
    thrust::device_vector<T> d_alphaRhs;
    thrust::device_vector<T> d_alphaSol;
    pinned_host_vector<double> h_state_cur;    // [U | Udot | Uddot]
    pinned_host_vector<double> h_state_prev;   // [Ut | Utdot | Utdotdot] step-start backup
    pinned_host_vector<double> h_state_alpha;  // [Ualpha | Ualphadot | Ualphadotdot]

    cusparseHandle_t cusparseHandle = nullptr;
    cudaStream_t stream = nullptr;
    bool ownsStream = false;
    CudaCsrMatrix *matM = nullptr;
    CudaCsrMatrix *matAlpha = nullptr;
    CudaCsrMatrix *matA = nullptr;
    int boundStructureRows = 0;
    int boundStructureNnz = 0;
    const int *boundRowPtr = nullptr;
    const int *boundColIdx = nullptr;

    int getSize() const override { return size; }

    T *devU() { return raw_pointer_cast(d_state_cur.data()); }
    T *devUdot() { return devU() + size; }
    T *devUddot() { return devU() + 2 * size; }
    const T *devU() const { return raw_pointer_cast(d_state_cur.data()); }
    const T *devUdot() const { return devU() + size; }
    const T *devUddot() const { return devU() + 2 * size; }
    T *devUalpha() { return raw_pointer_cast(d_state_alpha.data()); }
    T *devUalphadot() { return devUalpha() + size; }
    T *devUalphadotdot() { return devUalpha() + 2 * size; }

    void allocate(int n, int alphaNumRhsIn) override
    {
        size = n;
        alphaNumRhs = alphaNumRhsIn;
        const T zero = static_cast<T>(0.0);
        const std::size_t motionSize = static_cast<std::size_t>(kMotionFields * n);
        d_state_cur.assign(motionSize, zero);
        d_state_alpha.assign(motionSize, zero);
        d_w2.assign(n, zero);
        d_alphaRhs.assign(alphaNumRhs * n, zero);
        d_alphaSol.assign(alphaNumRhs * n, zero);
    }

    HostMotionPtrs ensureHostMotionBuffers(int n) override
    {
        HostMotionPtrs buf{};
        if (n <= 0) {
            return buf;
        }
        const std::size_t un = static_cast<std::size_t>(n);
        const std::size_t motionSize = static_cast<std::size_t>(kMotionFields) * un;
        h_state_cur.resize(motionSize);
        h_state_prev.resize(motionSize);
        h_state_alpha.resize(motionSize);
        double *cur = raw_pointer_cast(h_state_cur.data());
        double *prev = raw_pointer_cast(h_state_prev.data());
        double *alpha = raw_pointer_cast(h_state_alpha.data());
        buf.U = cur;
        buf.Udot = cur + n;
        buf.Uddot = cur + 2 * n;
        buf.Ut = prev;
        buf.Utdot = prev + n;
        buf.Utdotdot = prev + 2 * n;
        buf.Ualpha = alpha;
        buf.Ualphadot = alpha + n;
        buf.Ualphadotdot = alpha + 2 * n;
        return buf;
    }

    void ensureAlphaBuffers(int alphaNumRhsIn) override
    {
        if (alphaNumRhsIn == alphaNumRhs && static_cast<int>(d_alphaSol.size()) == alphaNumRhs * size) {
            return;
        }
        alphaNumRhs = alphaNumRhsIn;
        const T zero = static_cast<T>(0.0);
        d_alphaRhs.assign(alphaNumRhs * size, zero);
        d_alphaSol.assign(alphaNumRhs * size, zero);
    }

    void destroySolvers() override
    {
        if (matAlpha != nullptr) {
            matAlpha->detachSharedSpmvPattern();
        }
        if (matA != nullptr) {
            matA->detachSharedSpmvPattern();
        }
        delete matAlpha;
        delete matA;
        delete matM;
        matM = nullptr;
        matAlpha = nullptr;
        matA = nullptr;
        boundStructureRows = 0;
        boundStructureNnz = 0;
        boundRowPtr = nullptr;
        boundColIdx = nullptr;
    }

    void shutdown() override
    {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        }
        cudaDeviceSynchronize();
        destroySolvers();
        if (cusparseHandle != nullptr) {
            cusparseDestroy(cusparseHandle);
            cusparseHandle = nullptr;
        }
        if (ownsStream && stream != nullptr) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
        stream = nullptr;
        ownsStream = false;
    }

    void makeMatrixConfigs(CudaPrecision prec, CudaCsrMatrix::SolverConfig &solver,
                           CudaCsrMatrix::SpmvConfig &spmv, CudaCsrMatrix::ExecutionContext &exec,
                           const CuSparseBackend *sharedPattern = nullptr) const
    {
        solver = CudaCsrMatrix::SolverConfig{};
        solver.precision = prec;
        solver.syncAfterSolve = false;
        exec.stream = stream;
        spmv.externalHandle = cusparseHandle;
        spmv.sharedPattern = sharedPattern;
    }

    int useSolverStream(CudaGenBcsrLinSOE *cudaSOE) override
    {
        if (cudaSOE == nullptr) {
            return -1;
        }
        void *streamPtr = cudaSOE->getSolverStream();
        if (streamPtr == nullptr) {
            if (stream == nullptr) {
                cudaCheckError(cudaStreamCreate(&stream), "create integrator fallback stream");
                ownsStream = true;
            }
            return 0;
        }
        cudaStream_t solverStream = static_cast<cudaStream_t>(streamPtr);
        if (ownsStream && stream != nullptr && stream != solverStream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
        stream = solverStream;
        ownsStream = false;
        if (cusparseHandle != nullptr) {
            cuSparseCheckError(cusparseSetStream(cusparseHandle, stream), "cusparseSetStream");
        }
        return 0;
    }

    void initCusparse()
    {
        if (cusparseHandle == nullptr) {
            cuSparseCheckError(cusparseCreate(&cusparseHandle), "cusparseCreate");
            cuSparseCheckError(cusparseSetStream(cusparseHandle, stream), "cusparseSetStream");
        }
    }

    void ensureMatrices(CudaPrecision prec)
    {
        initCusparse();
        if (matM == nullptr) {
            CudaCsrMatrix::SolverConfig solver;
            CudaCsrMatrix::SpmvConfig spmv;
            CudaCsrMatrix::ExecutionContext exec;
            makeMatrixConfigs(prec, solver, spmv, exec);
            matM = new CudaCsrMatrix(solver, spmv, exec);

            makeMatrixConfigs(prec, solver, spmv, exec, matM->getSpmvBackend());
            matAlpha = new CudaCsrMatrix(solver, spmv, exec);

            makeMatrixConfigs(prec, solver, spmv, exec, matM->getSpmvBackend());
            matA = new CudaCsrMatrix(solver, spmv, exec);
        }
    }

    int bindSharedStructure(CudaGenBcsrLinSOE *soe)
    {
        const int nnz = soe->getNumNonZeroValues();
        const int *rowPtr = soe->getDeviceRowPtrs();
        const int *colIdx = soe->getDeviceColIndices();
        if (rowPtr == nullptr || colIdx == nullptr) {
            return -1;
        }

        const bool samePattern = boundStructureRows == size && boundStructureNnz == nnz && boundRowPtr == rowPtr &&
                                 boundColIdx == colIdx && matM != nullptr && matM->isStructureBound();

        if (!samePattern) {
            if (matM != nullptr && matM->isStructureBound()) {
                matM->reset();
                matAlpha->reset();
                matA->reset();
            }
            if (matM->bindStructure(size, nnz, rowPtr, colIdx) != 0) {
                return -1;
            }
            if (matAlpha->bindStructure(size, nnz, rowPtr, colIdx) != 0) {
                return -1;
            }
            if (matA->bindStructure(size, nnz, rowPtr, colIdx) != 0) {
                return -1;
            }
            boundStructureRows = size;
            boundStructureNnz = nnz;
            boundRowPtr = rowPtr;
            boundColIdx = colIdx;
        }
        return 0;
    }

    int applyM(const T *x, T *y) { return matM->spmv(x, y); }

    int formOperators(CudaExplicitAlpha *integrator, CudaGenBcsrLinSOE *cudaSOE) override
    {
        // Build/refactor GPU operators used by newStepPredictor() and formUnbalance().
        const double bdt2 = integrator->beta * integrator->deltaT * integrator->deltaT;
        const double gdt = integrator->gamma * integrator->deltaT;
        const bool areClose = integrator->areAlphaMFClose();
        const int numRhs = areClose ? 1 : 2;
        ensureAlphaBuffers(numRhs);

        if (useSolverStream(cudaSOE) != 0) {
            return -1;
        }
        cudaSOE->syncIndicesToDevice();
        cudaSOE->ensureDeviceVectorSizes();
        ensureMatrices(cudaSOE->getPrecision());
        if (bindSharedStructure(cudaSOE) != 0) {
            return -1;
        }

        void *rhsM = thrust::raw_pointer_cast(d_alphaRhs.data());
        void *solM = thrust::raw_pointer_cast(d_w2.data());
        void *rhsAlpha = rhsM;
        void *solAlpha1 = thrust::raw_pointer_cast(d_alphaSol.data());

        // --- Mass operator M (applyM / M^{-1} in predictor and formUnbalance) ---
        cudaSOE->zeroA();
        if (integrator->formTangentIntoSOE(INITIAL_TANGENT, 0.0, 0.0, 1.0) != 0) {
            return -1;
        }
        cudaSOE->syncAValuesToDevice();
        if (matM->copyValues(cudaSOE->getDeviceAValues()) != 0) {
            return -2;
        }

        // --- A_alpha for predictor solve (Newmark effective tangent) ---
        //     A_alpha = beta*dt^2*K + gamma*dt*C + M
        cudaSOE->zeroA();
        if (integrator->formTangentIntoSOE(INITIAL_TANGENT, bdt2, gdt, 1.0) != 0) {
            return -3;
        }
        cudaSOE->syncAValuesToDevice();
        if (matAlpha->copyValues(cudaSOE->getDeviceAValues()) != 0) {
            return -4;
        }

        if (!matAlpha->isFactored()) {
            if (matAlpha->factorize(rhsAlpha, solAlpha1, numRhs, nullptr) != 0) {
                return -4;
            }
        } else if (matAlpha->refactorize(rhsAlpha, solAlpha1, numRhs) != 0) {
            return -4;
        }
        if (!matM->isFactored()) {
            if (matM->factorize(rhsM, solM, 1, matAlpha) != 0) {
                return -2;
            }
        } else if (matM->refactorize(rhsM, solM) != 0) {
            return -2;
        }

        // --- A for generalized-alpha effective tangent (SpMV in predictor / formUnbalance) ---
        //     alphaM ~= alphaF:  A = alphaM*M
        //     else:             A = alphaF*beta*dt^2*K + alphaF*gamma*dt*C + alphaM*M
        cudaSOE->zeroA();
        if (areClose) {
            if (integrator->formTangentIntoSOE(INITIAL_TANGENT, 0.0, 0.0, integrator->alphaM) != 0) {
                return -5;
            }
        } else if (integrator->formTangentIntoSOE(INITIAL_TANGENT, integrator->alphaF * bdt2,
                                                  integrator->alphaF * gdt, integrator->alphaM) != 0) {
            return -5;
        }
        cudaSOE->syncAValuesToDevice();
        matA->bindValues(cudaSOE->getDeviceAValues());
        return 0;
    }

    int newStepPredictor(CudaExplicitAlpha *integrator) override
    {
        if (size > 0) {
            d_state_cur = h_state_cur;
        }
        T *dU = devU();
        T *dUdot = devUdot();
        T *dUddot = devUddot();
        T *dAlphaSol1 = thrust::raw_pointer_cast(d_alphaSol.data());
        T *dAlphaSol2 = alphaNumRhs > 1 ? dAlphaSol1 + size : nullptr;
        T *dUalpha = devUalpha();
        T *dUalphadot = devUalphadot();
        T *dUalphadotdot = devUalphadotdot();
        T *dAlphaRhs = thrust::raw_pointer_cast(d_alphaRhs.data());

        const T dt = static_cast<T>(integrator->deltaT);
        const T gammaT = static_cast<T>(integrator->gamma);
        const T alphaFT = static_cast<T>(integrator->alphaF);
        const T alphaMT = static_cast<T>(integrator->alphaM);
        const bool alphaClose = integrator->areAlphaMFClose();
        const int incAccel = integrator->incrementalAccel ? 1 : 0;

        if (alphaClose) {
            applyM(dUddot, dAlphaRhs);
            if (matAlpha->solve(dAlphaRhs, dAlphaSol1, 1) != 0) {
                return -5;
            }
            kernelNewStep<<<gridBlocks(size), 256, 0, stream>>>(
                size, dt, gammaT, alphaFT, alphaMT, static_cast<const T *>(nullptr), dU, dUdot, dUddot, dAlphaSol1,
                dUalpha, dUalphadot, dUalphadotdot, incAccel, 1);
        } else {
            applyM(dUddot, dAlphaRhs);
            matA->spmv(dUddot, dAlphaRhs + size);
            if (matAlpha->solve(dAlphaRhs, dAlphaSol1, alphaNumRhs) != 0) {
                return -5;
            }
            kernelNewStep<<<gridBlocks(size), 256, 0, stream>>>(size, dt, gammaT, alphaFT, alphaMT, dAlphaSol2, dU,
                                                                dUdot, dUddot, dAlphaSol1, dUalpha, dUalphadot,
                                                                dUalphadotdot, incAccel, 0);
        }
        // All predictor ops share integrator stream; sync once before Thrust D2H host mirror update.
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync before host read");
        h_state_alpha = d_state_alpha;
        return 0;
    }

    int formUnbalance(CudaGenBcsrLinSOE *cudaSOE) override
    {
        cudaSOE->syncBToDevice();
        void *b = cudaSOE->getDeviceB();
        T *w2 = thrust::raw_pointer_cast(d_w2.data());
        if (matM == nullptr || !matM->isFactored()) {
            return -3;
        }
        if (matM->solve(b, w2) != 0) {
            return -3;
        }
        matAlpha->spmv(w2, b);
        cudaSOE->setBPrimaryLocation(CudaGenBcsrLinSOE::DataLocation::Device);
        return 0;
    }

    int updateState(CudaGenBcsrLinSOE *cudaSOE, bool incrementalAccel) override
    {
        T *dUddot = devUddot();
        const T *dX = static_cast<const T *>(cudaSOE->getDeviceX());
        if (dX == nullptr) {
            return -4;
        }
        if (incrementalAccel) {
            const T one = static_cast<T>(1.0);
            kernelAxpy<<<gridBlocks(size), 256, 0, stream>>>(size, one, dX, dUddot);
        } else {
            cudaMemcpyAsync(dUddot, dX, static_cast<std::size_t>(size) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        }
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync before host read");
        h_state_cur = d_state_cur;
        return 0;
    }

    void zeroState() override
    {
        if (size <= 0) {
            return;
        }
        const T zero = static_cast<T>(0.0);
        d_state_cur.assign(static_cast<std::size_t>(kMotionFields * size), zero);
    }
};

template struct ImplT<double>;
template struct ImplT<float>;

void CudaExplicitAlpha::destroyDeviceImpl()
{
    if (m_impl != nullptr) {
        m_impl->shutdown();
        delete m_impl;
        m_impl = nullptr;
    }
}

void CudaExplicitAlpha::ensureDeviceImpl(CudaGenBcsrLinSOE *cudaSOE)
{
    const CudaPrecision prec = cudaSOE->getPrecision();
    const bool useFloat = (prec == CudaPrecision::dFFI);
    if (m_impl != nullptr && m_deviceUsesFloat == useFloat) {
        return;
    }
    destroyDeviceImpl();
    if (useFloat) {
        m_impl = new ImplT<float>();
    } else {
        m_impl = new ImplT<double>();
    }
    m_deviceUsesFloat = useFloat;
}

int CudaExplicitAlpha::formOperators(CudaGenBcsrLinSOE *cudaSOE)
{
    return m_impl->formOperators(this, cudaSOE);
}

namespace {

bool parseCudaExplicitAlphaOptions(CudaExplicitAlpha::Options &opts)
{
    opts = CudaExplicitAlpha::Options{};
    while (OPS_GetNumRemainingInputArgs() > 0) {
        const char *tok = OPS_GetString();
        if (tok == nullptr) {
            break;
        }
        if (strcmp(tok, "-updateElemDisp") == 0) {
            opts.updElemDisp = true;
        } else if (strcmp(tok, "-incrementalAccel") == 0) {
            opts.incrementalAccel = true;
        } else if (strcmp(tok, "-alphaCloseCheck") == 0) {
            opts.useAlphaCloseCheck = true;
        } else {
            opserr << "WARNING CudaExplicitAlpha family - unknown flag " << tok
                   << "; want <-updateElemDisp> <-incrementalAccel> <-alphaCloseCheck>\n";
            return false;
        }
    }
    return true;
}

} // namespace

void *OPS_CudaExplicitAlpha(void)
{
    if (OPS_GetNumRemainingInputArgs() < 4) {
        opserr << "WARNING integrator CudaExplicitAlpha alphaF alphaM gamma beta "
                  "<-updateElemDisp> <-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }
    double alphaF = 0.0;
    double alphaM = 0.0;
    double gamma = 0.0;
    double beta = 0.0;
    int numData = 1;
    if (OPS_GetDoubleInput(&numData, &alphaF) != 0 || OPS_GetDoubleInput(&numData, &alphaM) != 0 ||
        OPS_GetDoubleInput(&numData, &gamma) != 0 || OPS_GetDoubleInput(&numData, &beta) != 0) {
        opserr << "WARNING CudaExplicitAlpha - invalid alphaF/alphaM/gamma/beta\n";
        return nullptr;
    }
    CudaExplicitAlpha::Options opts;
    if (!parseCudaExplicitAlphaOptions(opts)) {
        return nullptr;
    }
    return new CudaExplicitAlpha(alphaF, alphaM, gamma, beta, opts);
}

void *OPS_CudaKRAlpha(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING integrator CudaKRAlpha rhoInf "
                  "<-updateElemDisp> <-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }
    double rhoInf = 0.0;
    int numData = 1;
    if (OPS_GetDoubleInput(&numData, &rhoInf) != 0) {
        opserr << "WARNING CudaKRAlpha - invalid rhoInf\n";
        return nullptr;
    }
    const double den = 1.0 + rhoInf;
    const double alphaF = 1.0 / den;
    const double alphaM = (2.0 - rhoInf) / den;
    const double gamma = 0.5 * (3.0 - rhoInf) / den;
    const double beta = 1.0 / (den * den);
    CudaExplicitAlpha::Options opts;
    if (!parseCudaExplicitAlphaOptions(opts)) {
        return nullptr;
    }
    return new CudaExplicitAlpha(alphaF, alphaM, gamma, beta, opts);
}

void *OPS_CudaMKRAlpha(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING integrator CudaMKRAlpha rhoInfEquivalent "
                  "<-updateElemDisp> <-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }
    double rhoInfEquivalent = 0.0;
    int numData = 1;
    if (OPS_GetDoubleInput(&numData, &rhoInfEquivalent) != 0) {
        opserr << "WARNING CudaMKRAlpha - invalid rhoInfEquivalent\n";
        return nullptr;
    }
    const double r = (rhoInfEquivalent >= 0.0) ? std::sqrt(rhoInfEquivalent) : rhoInfEquivalent;
    const double alphaF = 1.0 / (r + 1.0);
    const double r2 = r * r;
    const double r3 = r2 * r;
    const double num = (2.0 * r3 + r2 - 1.0);
    const double den = (r3 + r2 + r + 1.0);
    const double alphaM = 1.0 - num / den;
    const double gamma = alphaM - alphaF + 0.5;
    const double beta = 0.5 * (gamma + 0.5);
    CudaExplicitAlpha::Options opts;
    if (!parseCudaExplicitAlphaOptions(opts)) {
        return nullptr;
    }
    return new CudaExplicitAlpha(alphaF, alphaM, gamma, beta, opts);
}

CudaExplicitAlpha::CudaExplicitAlpha()
    : CudaExplicitAlpha(INTEGRATOR_TAGS_CudaExplicitAlpha, 0.5, 0.5, 0.5, 0.25, Options{})
{
}

CudaExplicitAlpha::CudaExplicitAlpha(double _alphaF, double _alphaM, double _gamma, double _beta)
    : CudaExplicitAlpha(_alphaF, _alphaM, _gamma, _beta, Options{})
{
}

CudaExplicitAlpha::CudaExplicitAlpha(double _alphaF, double _alphaM, double _gamma, double _beta, const Options &opts)
    : CudaExplicitAlpha(INTEGRATOR_TAGS_CudaExplicitAlpha, _alphaF, _alphaM, _gamma, _beta, opts)
{
}

CudaExplicitAlpha::CudaExplicitAlpha(int classTag, double _alphaF, double _alphaM, double _gamma, double _beta,
                                     const Options &opts)
    : TransientIntegrator(classTag),
      alphaM(_alphaM),
      alphaF(_alphaF),
      beta(_beta),
      gamma(_gamma),
      updElemDisp(opts.updElemDisp),
      incrementalAccel(opts.incrementalAccel),
      useAlphaCloseCheck(opts.useAlphaCloseCheck),
      updateCount(0),
      c1(0.0),
      c2(0.0),
      c3(0.0),
      deltaT(0.0),
      Ut(new Vector(0)),
      Utdot(new Vector(0)),
      Utdotdot(new Vector(0)),
      U(new Vector(0)),
      Udot(new Vector(0)),
      Udotdot(new Vector(0)),
      Ualpha(new Vector(0)),
      Ualphadot(new Vector(0)),
      Ualphadotdot(new Vector(0)),
      operatorsBuilt(false),
      m_impl(nullptr)
{
}

CudaExplicitAlpha::~CudaExplicitAlpha()
{
    CudaGenBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) == 0) {
        cudaSOE->setXSyncMode(true);
    }
    destroyDeviceImpl();
    delete Ut;
    delete Utdot;
    delete Utdotdot;
    delete U;
    delete Udot;
    delete Udotdot;
    delete Ualpha;
    delete Ualphadot;
    delete Ualphadotdot;
}

bool CudaExplicitAlpha::areAlphaMFClose() const
{
    return useAlphaCloseCheck && std::fabs(alphaM - alphaF) <= toleranceAlphaMF;
}

int CudaExplicitAlpha::validateCudaSOE(CudaGenBcsrLinSOE *&cudaSOE) const
{
    LinearSOE *soe = this->getLinearSOE();
    if (soe == nullptr) {
        return -1;
    }
    cudaSOE = dynamic_cast<CudaGenBcsrLinSOE *>(soe);
    if (cudaSOE == nullptr || cudaSOE->getBlockSize() != 1 || cudaSOE->getCudaGenBcsrLinSolver() == nullptr) {
        return -2;
    }
    if (!isUniformPrecision(cudaSOE->getPrecision())) {
        opserr << "ERROR CudaExplicitAlpha::validateCudaSOE() - unsupported SOE precision "
               << cudaPrecisionToString(cudaSOE->getPrecision())
               << "; only uniform dDDI and dFFI are supported\n";
        return -3;
    }
    return 0;
}

int CudaExplicitAlpha::formTangentIntoSOE(int statFlag, double c1v, double c2v, double c3v)
{
    if (this->getAnalysisModel() == nullptr) {
        return -1;
    }
    statusFlag = statFlag;
    c1 = c1v;
    c2 = c2v;
    c3 = c3v;
    return this->TransientIntegrator::formTangent(statFlag);
}

int CudaExplicitAlpha::formTangent(int statFlag)
{
    (void)statFlag;
    return 0;
}

int CudaExplicitAlpha::formEleTangent(FE_Element *theEle)
{
    if (operatorsBuilt) {
        return 0;
    }
    theEle->zeroTangent();
    if (statusFlag == CURRENT_TANGENT) {
        theEle->addKtToTang(c1);
    } else if (statusFlag == INITIAL_TANGENT) {
        theEle->addKiToTang(c1);
    }
    theEle->addCtoTang(c2);
    theEle->addMtoTang(c3);
    return 0;
}

int CudaExplicitAlpha::formNodTangent(DOF_Group *theDof)
{
    if (operatorsBuilt) {
        return 0;
    }
    theDof->zeroTangent();
    theDof->addCtoTang(c2);
    theDof->addMtoTang(c3);
    return 0;
}

int CudaExplicitAlpha::domainChanged()
{
    CudaGenBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) != 0) {
        operatorsBuilt = false;
        return -1;
    }
    const int size = cudaSOE->getNumEqn();
    if (size <= 0) {
        operatorsBuilt = false;
        return 0;
    }

    ensureDeviceImpl(cudaSOE);
    cudaSOE->setXSyncMode(false);

    m_impl->destroySolvers();
    if (m_impl->useSolverStream(cudaSOE) != 0) {
        operatorsBuilt = false;
        return -1;
    }
    m_impl->allocate(size, areAlphaMFClose() ? 1 : 2);
    const ImplBase::HostMotionPtrs buf = m_impl->ensureHostMotionBuffers(size);
    Ut->setData(buf.Ut, size);
    Utdot->setData(buf.Utdot, size);
    Utdotdot->setData(buf.Utdotdot, size);
    U->setData(buf.U, size);
    Udot->setData(buf.Udot, size);
    Udotdot->setData(buf.Uddot, size);
    Ualpha->setData(buf.Ualpha, size);
    Ualphadot->setData(buf.Ualphadot, size);
    Ualphadotdot->setData(buf.Ualphadotdot, size);
    operatorsBuilt = false;

    AnalysisModel *theModel = this->getAnalysisModel();
    DOF_GrpIter &theDOFs = theModel->getDOFs();
    DOF_Group *dofPtr;
    while ((dofPtr = theDOFs()) != nullptr) {
        const ID &id = dofPtr->getID();
        const Vector &disp = dofPtr->getCommittedDisp();
        const Vector &vel = dofPtr->getCommittedVel();
        const Vector &accel = dofPtr->getCommittedAccel();
        for (int i = 0; i < id.Size(); ++i) {
            const int loc = id(i);
            if (loc >= 0) {
                (*U)(loc) = disp(i);
                (*Udot)(loc) = vel(i);
                (*Udotdot)(loc) = accel(i);
            }
        }
    }
    *Ut = *U;
    *Utdot = *Udot;
    *Utdotdot = *Udotdot;
    return 0;
}

int CudaExplicitAlpha::newStep(double _deltaT)
{
    updateCount = 0;
    if (alphaF < 0.5 || alphaF > 1.0 || beta <= 0.0 || gamma <= 0.0 || _deltaT <= 0.0) {
        return -1;
    }
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr || m_impl == nullptr) {
        return -2;
    }
    CudaGenBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) != 0) {
        return -3;
    }
    ensureDeviceImpl(cudaSOE);
    cudaSOE->setXSyncMode(false);

    *Ut = *U;
    *Utdot = *Udot;
    *Utdotdot = *Udotdot;
    if (_deltaT != deltaT) {
        deltaT = _deltaT;
        operatorsBuilt = false;
    }
    if (!operatorsBuilt) {
        if (formOperators(cudaSOE) != 0) {
            return -4;
        }
        operatorsBuilt = true;
    }

    if (m_impl->newStepPredictor(this) != 0) {
        return -5;
    }
    theModel->setResponse(*Ualpha, *Ualphadot, *Ualphadotdot);
    if (theModel->updateDomain(theModel->getCurrentDomainTime() + alphaF * deltaT, deltaT) < 0) {
        return -6;
    }
    return 0;
}

int CudaExplicitAlpha::formUnbalance()
{
    if (m_impl == nullptr || TransientIntegrator::formUnbalance() < 0) {
        return -1;
    }
    if (areAlphaMFClose()) {
        return 0;
    }
    CudaGenBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) != 0) {
        return -2;
    }
    const int rc = m_impl->formUnbalance(cudaSOE);
    return rc;
}

int CudaExplicitAlpha::update(const Vector &aiPlusOne)
{
    updateCount++;
    if (updateCount > 1) {
        return -1;
    }
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr || U->Size() <= 0 || aiPlusOne.Size() != U->Size() || m_impl == nullptr) {
        return -2;
    }
    CudaGenBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) != 0) {
        return -4;
    }
    if (m_impl->updateState(cudaSOE, incrementalAccel) != 0) {
        return -4;
    }
    theModel->setVel(*Udot);
    theModel->setAccel(*Udotdot);
    if (theModel->updateDomain() < 0) {
        return -3;
    }
    theModel->setDisp(*U);
    return 0;
}

int CudaExplicitAlpha::commit()
{
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr) {
        return -1;
    }
    theModel->setCurrentDomainTime(theModel->getCurrentDomainTime() + (1.0 - alphaF) * deltaT);
    if (updElemDisp) {
        theModel->updateDomain();
    }
    return theModel->commitDomain();
}

int CudaExplicitAlpha::revertToLastStep()
{
    CudaGenBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) == 0) {
        cudaSOE->setXSyncMode(true);
    }
    if (U->Size() > 0) {
        *U = *Ut;
        *Udot = *Utdot;
        *Udotdot = *Utdotdot;
    }
    return 0;
}

double CudaExplicitAlpha::getCFactor() { return c2; }

const Vector &CudaExplicitAlpha::getVel() { return *Ualphadot; }

int CudaExplicitAlpha::sendSelf(int cTag, Channel &theChannel)
{
    Vector data(8);
    data(0) = alphaM;
    data(1) = alphaF;
    data(2) = beta;
    data(3) = gamma;
    data(4) = updElemDisp ? 1.0 : 0.0;
    data(5) = incrementalAccel ? 1.0 : 0.0;
    data(6) = useAlphaCloseCheck ? 1.0 : 0.0;
    data(7) = 0.0;
    return theChannel.sendVector(this->getDbTag(), cTag, data);
}

int CudaExplicitAlpha::recvSelf(int cTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    Vector data(8);
    if (theChannel.recvVector(this->getDbTag(), cTag, data) < 0) {
        return -1;
    }
    alphaM = data(0);
    alphaF = data(1);
    beta = data(2);
    gamma = data(3);
    updElemDisp = data(4) > 0.5;
    incrementalAccel = data(5) > 0.5;
    useAlphaCloseCheck = data(6) > 0.5;
    operatorsBuilt = false;
    return 0;
}

void CudaExplicitAlpha::Print(OPS_Stream &s, int flag)
{
    (void)flag;
    s << "CudaExplicitAlpha alphaF=" << alphaF << " alphaM=" << alphaM;
    s << endln;
}

int CudaExplicitAlpha::revertToStart()
{
    CudaGenBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) == 0) {
        cudaSOE->setXSyncMode(true);
    }
    if (U->Size() > 0) {
        Ut->Zero();
        Utdot->Zero();
        Utdotdot->Zero();
        U->Zero();
        Udot->Zero();
        Udotdot->Zero();
        if (m_impl != nullptr && m_impl->getSize() > 0) {
            m_impl->zeroState();
        }
    }
    return 0;
}
