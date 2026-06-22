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

// Developed: Chinmoy Kolay (chk311@lehigh.edu)
// Implemented: Andreas Schellenberg (andreas.schellenberg@gmail.com)
// CUDA implementation: Gustavo A. Araujo R. (garaujor@stanford.edu)
//
// Description: GPU explicit Kolay-Ricles trapezoidal-rule (TP) integrator for
// CudaBcsrLinSOE + cuDSS, based on the CPU ExplicitAlpha_TP family.
//
// Reference: Kolay, C. and J. Ricles (2014). "Development of a family of
// unconditionally stable explicit direct integration algorithms with
// controllable numerical energy dissipation." Earthquake Engineering and
// Structural Dynamics, 43(9):1361-1380.

#include <CudaExplicitAlpha_TP.h>
#include <CudaBcsrLinSOE.h>
#include <AnalysisModel.h>
#include <Channel.h>
#include <DOF_Group.h>
#include <DOF_GrpIter.h>
#include <FE_Element.h>
#include <FEM_ObjectBroker.h>
#include <LinearSOE.h>
#include <Vector.h>
#include <classTags.h>
#include <elementAPI.h>
#include <cmath>
#include <cstring>

#include "CudaCsrMatrix.h"
#include "CudaUtils.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/memory.h>

using thrust::raw_pointer_cast;

using namespace CudaUtils;

// Kolay-Ricles trapezoidal-rule (TP) integrator on GPU (CudaBcsrLinSOE + cuDSS).
// Unlike the midpoint CUDA path, TP uses two residual passes per step and a blended Put vector.
// Device scalar type T follows SOE precision (dDDI=double, dFFI=float); host stays double.

namespace {

// --- Templated device kernels ---

template<typename T>
__global__ void kernelAxpby(int n, T alpha, const T *x, T beta, T *y)
{
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

// TP displacement/velocity predictor + trial acceleration (full-step kinematics, not alpha-interpolated).
// alphaSol1 = alpha^{-1} M Uddot_n; alphaSol3 = alpha_3 Uddot_n when alphaM != alphaF.
template<typename T>
__global__ void kernelNewStepTP(int n, T dt, T gamma, T alphaF, T alphaM, const T *alphaSol1,
                                const T *alphaSol3, const T *Ut, const T *Utdot, const T *Utdotdot, T *U, T *Udot,
                                T *Uddot, int incrementalAccel, int alphaClose)
{
    const int stride = blockDim.x * gridDim.x;
    const T invAlphaF = static_cast<T>(1.0) / alphaF;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const T uhat = dt * alphaSol1[i];  // Udot_hat = dt * alpha_1 * Uddot_n
        U[i] = Ut[i] + dt * Utdot[i] + (static_cast<T>(0.5) + gamma) * dt * uhat;
        Udot[i] = Utdot[i] + uhat;
        // Trial accel for the second residual pass at t_{n+1}
        if (incrementalAccel) {
            Uddot[i] = Utdotdot[i] * invAlphaF;
        } else if (alphaClose) {
            Uddot[i] = (static_cast<T>(1.0) - alphaM) * Utdotdot[i] * invAlphaF;
        } else {
            Uddot[i] = (Utdotdot[i] - alphaSol3[i]) * invAlphaF;
        }
    }
}

static int gridBlocks(int n, int blockSize = 256)
{
    return (n + blockSize - 1) / blockSize;
}

} // namespace

static constexpr int kMotionFields = 3;  // U, Udot, Uddot packed per node block

// Pimpl interface: double (host) / float|double (device) selected from SOE precision.
struct CudaExplicitAlpha_TP::ImplBase {
    struct HostMotionPtrs {
        double *U;
        double *Udot;
        double *Uddot;
        double *Ut;       // step-start backup (maps to h_state_prev)
        double *Utdot;
        double *Utdotdot;
    };

    virtual ~ImplBase() = default;

    virtual void allocate(int n, int alphaNumRhs) = 0;
    virtual void destroySolvers() = 0;
    virtual void shutdown() = 0;
    virtual void ensureAlphaBuffers(int alphaNumRhs) = 0;
    virtual int formOperators(CudaExplicitAlpha_TP *integrator, CudaBcsrLinSOE *cudaSOE) = 0;
    virtual HostMotionPtrs ensureHostMotionBuffers(int n) = 0;
    virtual int newStepPredictor(CudaExplicitAlpha_TP *integrator) = 0;
    virtual int formUnbalanceFromPut(CudaBcsrLinSOE *cudaSOE, bool alphaClose) = 0;
    virtual int updateState(CudaBcsrLinSOE *cudaSOE, bool incrementalAccel, double alphaF) = 0;
    virtual void zeroState() = 0;
    virtual int getSize() const = 0;
    virtual double *getPutHostPtr() = 0;
    virtual int capturePutFromDeviceB(CudaBcsrLinSOE *cudaSOE) = 0;
    virtual void blendPutFromHostB(CudaBcsrLinSOE *cudaSOE, double oneMinusAlphaF, double alphaF) = 0;
};

template<typename T>
struct ImplT_TP : CudaExplicitAlpha_TP::ImplBase {
    int size = 0;
    int alphaNumRhs = 2;  // 1 when alphaM ~= alphaF shortcut applies

    thrust::device_vector<T> d_state_cur;    // [U | Udot | Uddot]
    thrust::device_vector<T> d_state_prev;   // [Ut | Utdot | Utdotdot] step-start backup
    thrust::device_vector<T> d_w2;           // workspace for M^{-1} and solves
    thrust::device_vector<T> d_put;          // blended TP unbalance on device
    thrust::device_vector<T> d_alphaRhs;     // RHS for alpha-operator solve(s)
    thrust::device_vector<T> d_alphaSol;     // alphaSol1 [| alphaSol3]
    pinned_host_vector<double> h_state_cur;
    pinned_host_vector<double> h_state_prev;
    pinned_host_vector<double> h_put;          // mirrored by OpenSees Vector *Put

    cusparseHandle_t cusparseHandle = nullptr;
    cudaStream_t stream = nullptr;           // cached SOE stream from getCudaStream()
    CudaCsrMatrix *matM = nullptr;             // mass operator M
    CudaCsrMatrix *matAlpha = nullptr;         // alpha = M + gamma*dt*C + beta*dt^2*K
    CudaCsrMatrix *matA = nullptr;             // A = alphaM*M + alphaF*gamma*dt*C + alphaF*beta*dt^2*K
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
    void allocate(int n, int alphaNumRhsIn) override
    {
        size = n;
        alphaNumRhs = alphaNumRhsIn;
        const T zero = static_cast<T>(0.0);
        const std::size_t motionSize = static_cast<std::size_t>(kMotionFields * n);
        d_state_cur.assign(motionSize, zero);
        d_state_prev.assign(motionSize, zero);
        d_w2.assign(n, zero);
        d_put.assign(n, zero);
        d_alphaRhs.assign(alphaNumRhs * n, zero);
        d_alphaSol.assign(alphaNumRhs * n, zero);
        h_put.assign(static_cast<std::size_t>(n), 0.0);
    }

    HostMotionPtrs ensureHostMotionBuffers(int n) override
    {
        HostMotionPtrs buf{};
        if (n <= 0) {
            return buf;
        }
        const std::size_t motionSize = static_cast<std::size_t>(kMotionFields) * static_cast<std::size_t>(n);
        h_state_cur.resize(motionSize);
        h_state_prev.resize(motionSize);
        h_put.resize(static_cast<std::size_t>(n));
        double *cur = raw_pointer_cast(h_state_cur.data());
        double *prev = raw_pointer_cast(h_state_prev.data());
        buf.U = cur;
        buf.Udot = cur + n;
        buf.Uddot = cur + 2 * n;
        buf.Ut = prev;
        buf.Utdot = prev + n;
        buf.Utdotdot = prev + 2 * n;
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
        stream = nullptr;
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

    int bindStream(CudaBcsrLinSOE *cudaSOE)
    {
        if (cudaSOE == nullptr) {
            return -1;
        }
        stream = static_cast<cudaStream_t>(cudaSOE->getCudaStream());
        if (stream == nullptr) {
            return -1;
        }
        if (cusparseHandle != nullptr) {
            cuSparseCheckError(cusparseSetStream(cusparseHandle, stream), "cusparseSetStream");
        }
        return 0;
    }

    void initCusparse()
    {
        if (cusparseHandle == nullptr) {
            cuSparseCheckError(cusparseCreate(&cusparseHandle), "cusparseCreate");
        }
        if (stream != nullptr) {
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

    int bindSharedStructure(CudaBcsrLinSOE *soe)
    {
        // Reuse one CSR pattern for matM, matAlpha, and matA (values differ).
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

    // Build/refactor GPU operators used by newStepPredictor() and formUnbalanceFromPut().
    int formOperators(CudaExplicitAlpha_TP *integrator, CudaBcsrLinSOE *cudaSOE) override
    {
        const double bdt2 = integrator->beta * integrator->deltaT * integrator->deltaT;
        const double gdt = integrator->gamma * integrator->deltaT;
        const bool areClose = integrator->areAlphaMFClose();
        const int numRhs = areClose ? 1 : 2;
        ensureAlphaBuffers(numRhs);

        if (bindStream(cudaSOE) != 0) {
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

        // --- Mass operator M ---
        cudaSOE->zeroA();
        if (integrator->formTangentIntoSOE(INITIAL_TANGENT, 0.0, 0.0, 1.0) != 0) {
            return -1;
        }
        cudaSOE->syncAValuesToDevice();
        if (matM->copyValues(cudaSOE->getDeviceAValues()) != 0) {
            return -2;
        }

        // --- alpha = beta*dt^2*K + gamma*dt*C + M (predictor + RHS transform) ---
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

        // --- A on primary SOE (SpMV for alpha_3 when alphaM != alphaF) ---
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

    // GPU predictor: alpha_1*Uddot_n solve, then full-step trial (U, Udot, Uddot).
    int newStepPredictor(CudaExplicitAlpha_TP *integrator) override
    {
        if (size > 0) {
            d_state_cur = h_state_cur;
            d_state_prev = h_state_prev;
        }

        const T *dUt = raw_pointer_cast(d_state_prev.data());
        const T *dUtdot = dUt + size;
        const T *dUtdotdot = dUt + 2 * size;

        T *dU = devU();
        T *dUdot = devUdot();
        T *dUddot = devUddot();
        T *dAlphaSol1 = thrust::raw_pointer_cast(d_alphaSol.data());
        T *dAlphaSol3 = alphaNumRhs > 1 ? dAlphaSol1 + size : nullptr;
        T *dAlphaRhs = thrust::raw_pointer_cast(d_alphaRhs.data());

        const T dt = static_cast<T>(integrator->deltaT);
        const T gammaT = static_cast<T>(integrator->gamma);
        const T alphaFT = static_cast<T>(integrator->alphaF);
        const T alphaMT = static_cast<T>(integrator->alphaM);
        const bool alphaClose = integrator->areAlphaMFClose();
        const int incAccel = integrator->incrementalAccel ? 1 : 0;

        if (alphaClose) {
            // alpha_1 = alpha^{-1} M Uddot_n
            applyM(dUtdotdot, dAlphaRhs);
            if (matAlpha->solve(dAlphaRhs, dAlphaSol1, 1) != 0) {
                return -5;
            }
            kernelNewStepTP<<<gridBlocks(size), 256, 0, stream>>>(
                size, dt, gammaT, alphaFT, alphaMT, dAlphaSol1, static_cast<const T *>(nullptr), dUt, dUtdot,
                dUtdotdot, dU, dUdot, dUddot, incAccel, 1);
        } else {
            // Second RHS column: A * Uddot_n for alpha_3 when alphaM != alphaF
            applyM(dUtdotdot, dAlphaRhs);
            matA->spmv(dUtdotdot, dAlphaRhs + size);
            if (matAlpha->solve(dAlphaRhs, dAlphaSol1, alphaNumRhs) != 0) {
                return -5;
            }
            kernelNewStepTP<<<gridBlocks(size), 256, 0, stream>>>(size, dt, gammaT, alphaFT, alphaMT, dAlphaSol1,
                                                                  dAlphaSol3, dUt, dUtdot, dUtdotdot, dU, dUdot, dUddot,
                                                                  incAccel, 0);
        }
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync before host read");
        h_state_cur = d_state_cur;
        return 0;
    }

    double *getPutHostPtr() override { return raw_pointer_cast(h_put.data()); }

    // Put <- B^(1): sync SOE B to device and copy into d_put (B is overwritten in pass 2).
    int capturePutFromDeviceB(CudaBcsrLinSOE *cudaSOE) override
    {
        if (bindStream(cudaSOE) != 0) {
            return -1;
        }
        cudaSOE->syncBToDevice();
        const T *dB = static_cast<const T *>(cudaSOE->getDeviceB());
        if (dB == nullptr) {
            return -2;
        }
        T *dPut = thrust::raw_pointer_cast(d_put.data());
        cudaCheckError(cudaMemcpyAsync(dPut, dB, static_cast<std::size_t>(size) * sizeof(T),
                                       cudaMemcpyDeviceToDevice, stream),
                       "capture Put from device B");
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync after Put capture");
        return 0;
    }

    void blendPutFromHostB(CudaBcsrLinSOE *cudaSOE, double oneMinusAlphaF, double alphaF) override
    {
        if (bindStream(cudaSOE) != 0) {
            return;
        }
        cudaSOE->syncBToDevice();
        const T *dB = static_cast<const T *>(cudaSOE->getDeviceB());
        T *dPut = thrust::raw_pointer_cast(d_put.data());
        // Put <- (1 - alphaF) * Put + alphaF * B^(2)
        kernelAxpby<<<gridBlocks(size), 256, 0, stream>>>(size, static_cast<T>(alphaF), dB,
                                                            static_cast<T>(oneMinusAlphaF), dPut);
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync after Put blend");
        h_put = d_put;
    }

    // CudaExplicitAlpha_TP::formUnbalance() (from Linear::solveCurrentStep): B <- Put or B <- alpha * M^{-1} * Put.
    int formUnbalanceFromPut(CudaBcsrLinSOE *cudaSOE, bool alphaClose) override
    {
        if (bindStream(cudaSOE) != 0) {
            return -1;
        }
        if (alphaClose) {
            // Proportional shortcut: primary SOE holds A = alphaM*M, so B = Put directly.
            return cudaSOE->setDeviceB(thrust::raw_pointer_cast(d_put.data()), size);
        }

        T *w2 = thrust::raw_pointer_cast(d_w2.data());
        void *put = thrust::raw_pointer_cast(d_put.data());
        if (matM == nullptr || !matM->isFactored()) {
            return -3;
        }
        if (matM->solve(put, w2) != 0) {
            return -3;
        }
        void *b = cudaSOE->getDeviceB();
        matAlpha->spmv(w2, b);  // B = alpha * M^{-1} * Put
        cudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Device);
        return 0;
    }

    // TP update only refreshes acceleration; U and Udot were set in newStep().
    int updateState(CudaBcsrLinSOE *cudaSOE, bool incrementalAccel, double alphaF) override
    {
        if (bindStream(cudaSOE) != 0) {
            return -1;
        }
        T *dUddot = devUddot();
        const T *dX = static_cast<const T *>(cudaSOE->getDeviceX());
        if (dX == nullptr) {
            return -4;
        }
        if (incrementalAccel) {
            // Match ExplicitAlpha_TP::update addVector(alphaF, aiPlusOne, 1.0):
            // Uddot <- alphaF * Uddot_trial + X  (trial Uddot was set in newStepPredictor).
            const T alphaFT = static_cast<T>(alphaF);
            const T one = static_cast<T>(1.0);
            kernelAxpby<<<gridBlocks(size), 256, 0, stream>>>(size, one, dX, alphaFT, dUddot);
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
        d_put.assign(static_cast<std::size_t>(size), zero);
        std::fill(h_put.begin(), h_put.end(), 0.0);
    }
};

template struct ImplT_TP<double>;
template struct ImplT_TP<float>;

void CudaExplicitAlpha_TP::destroyDeviceImpl()
{
    if (m_impl != nullptr) {
        m_impl->shutdown();
        delete m_impl;
        m_impl = nullptr;
    }
}

void CudaExplicitAlpha_TP::ensureDeviceImpl(CudaBcsrLinSOE *cudaSOE)
{
    // Pick ImplT_TP<float> or ImplT_TP<double> to match uniform SOE precision.
    const CudaPrecision prec = cudaSOE->getPrecision();
    const bool useFloat = (prec == CudaPrecision::dFFI);
    if (m_impl != nullptr && m_deviceUsesFloat == useFloat) {
        return;
    }
    destroyDeviceImpl();
    if (useFloat) {
        m_impl = new ImplT_TP<float>();
    } else {
        m_impl = new ImplT_TP<double>();
    }
    m_deviceUsesFloat = useFloat;
}

int CudaExplicitAlpha_TP::formOperators(CudaBcsrLinSOE *cudaSOE)
{
    return m_impl->formOperators(this, cudaSOE);
}

namespace {

bool validateCudaExplicitAlphaTPParams(double alphaF, double alphaM, double gamma, double beta)
{
    if (alphaF < 0.5 || alphaF > 1.0) {
        opserr << "WARNING - invalid alphaF for CudaExplicitAlpha_TP, want 0.5 <= alphaF <= 1.0\n";
        return false;
    }
    if (gamma <= 0.0 || beta <= 0.0) {
        opserr << "WARNING - invalid gamma/beta for CudaExplicitAlpha_TP, want gamma > 0 and beta > 0\n";
        return false;
    }
    if (alphaM < 0.5 || alphaM > 2.0) {
        opserr << "WARNING - recommended for unconditional stability (linear): 0.5 <= alphaM <= 2.0\n";
    }
    if (gamma < 0.5) {
        opserr << "WARNING - recommended for unconditional stability (linear): gamma >= 0.5\n";
    }
    if (beta < 0.5 * gamma) {
        opserr << "WARNING - recommended for unconditional stability (linear): beta >= gamma/2\n";
    }
    return true;
}

bool parseCudaExplicitAlphaTPOptions(CudaExplicitAlpha_TP::Options &opts)
{
    opts = CudaExplicitAlpha_TP::Options{};
    while (OPS_GetNumRemainingInputArgs() > 0) {
        const char *tok = OPS_GetString();
        if (tok == nullptr) {
            break;
        }
        if (strcmp(tok, "-incrementalAccel") == 0) {
            opts.incrementalAccel = true;
        } else if (strcmp(tok, "-alphaCloseCheck") == 0) {
            opts.useAlphaCloseCheck = true;
        } else {
            opserr << "WARNING CudaExplicitAlpha_TP family - unknown flag " << tok
                   << "; want <-incrementalAccel> <-alphaCloseCheck>\n";
            return false;
        }
    }
    return true;
}

} // namespace

void *OPS_CudaExplicitAlpha_TP(void)
{
    if (OPS_GetNumRemainingInputArgs() < 4) {
        opserr << "WARNING integrator CudaExplicitAlpha_TP alphaF alphaM gamma beta "
                  "<-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }
    double alphaF = 0.0;
    double alphaM = 0.0;
    double gamma = 0.0;
    double beta = 0.0;
    int numData = 1;
    if (OPS_GetDoubleInput(&numData, &alphaF) != 0 || OPS_GetDoubleInput(&numData, &alphaM) != 0 ||
        OPS_GetDoubleInput(&numData, &gamma) != 0 || OPS_GetDoubleInput(&numData, &beta) != 0) {
        opserr << "WARNING CudaExplicitAlpha_TP - invalid alphaF/alphaM/gamma/beta\n";
        return nullptr;
    }
    if (!validateCudaExplicitAlphaTPParams(alphaF, alphaM, gamma, beta)) {
        return nullptr;
    }
    CudaExplicitAlpha_TP::Options opts;
    if (!parseCudaExplicitAlphaTPOptions(opts)) {
        return nullptr;
    }
    return new CudaExplicitAlpha_TP(alphaF, alphaM, gamma, beta, opts);
}

void *OPS_CudaKRAlpha_TP(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING integrator CudaKRAlpha_TP rhoInf "
                  "<-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }
    double rhoInf = 0.0;
    int numData = 1;
    if (OPS_GetDoubleInput(&numData, &rhoInf) != 0) {
        opserr << "WARNING CudaKRAlpha_TP - invalid rhoInf\n";
        return nullptr;
    }
    const double den = 1.0 + rhoInf;
    const double alphaF = 1.0 / den;
    const double alphaM = (2.0 - rhoInf) / den;
    const double gamma = 0.5 * (3.0 - rhoInf) / den;
    const double beta = 1.0 / (den * den);
    CudaExplicitAlpha_TP::Options opts;
    if (!parseCudaExplicitAlphaTPOptions(opts)) {
        return nullptr;
    }
    return new CudaExplicitAlpha_TP(alphaF, alphaM, gamma, beta, opts);
}

void *OPS_CudaMKRAlpha_TP(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING integrator CudaMKRAlpha_TP rhoInfEquivalent "
                  "<-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }
    double rhoInfEquivalent = 0.0;
    int numData = 1;
    if (OPS_GetDoubleInput(&numData, &rhoInfEquivalent) != 0) {
        opserr << "WARNING CudaMKRAlpha_TP - invalid rhoInfEquivalent\n";
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
    CudaExplicitAlpha_TP::Options opts;
    if (!parseCudaExplicitAlphaTPOptions(opts)) {
        return nullptr;
    }
    return new CudaExplicitAlpha_TP(alphaF, alphaM, gamma, beta, opts);
}

CudaExplicitAlpha_TP::CudaExplicitAlpha_TP()
    : CudaExplicitAlpha_TP(INTEGRATOR_TAGS_CudaExplicitAlpha_TP, 0.5, 0.5, 0.5, 0.25, Options{})
{
}

CudaExplicitAlpha_TP::CudaExplicitAlpha_TP(double _alphaF, double _alphaM, double _gamma, double _beta)
    : CudaExplicitAlpha_TP(_alphaF, _alphaM, _gamma, _beta, Options{})
{
}

CudaExplicitAlpha_TP::CudaExplicitAlpha_TP(double _alphaF, double _alphaM, double _gamma, double _beta,
                                           const Options &opts)
    : CudaExplicitAlpha_TP(INTEGRATOR_TAGS_CudaExplicitAlpha_TP, _alphaF, _alphaM, _gamma, _beta, opts)
{
}

CudaExplicitAlpha_TP::CudaExplicitAlpha_TP(int classTag, double _alphaF, double _alphaM, double _gamma, double _beta,
                                             const Options &opts)
    : TransientIntegrator(classTag),
      alphaM(_alphaM),
      alphaF(_alphaF),
      beta(_beta),
      gamma(_gamma),
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
      Put(new Vector(0)),
      residM(0.0),
      residD(_alphaF),
      residR(_alphaF),
      residP(_alphaF),
      operatorsBuilt(false),
      m_impl(nullptr)
{
}

CudaExplicitAlpha_TP::~CudaExplicitAlpha_TP()
{
    CudaBcsrLinSOE *cudaSOE = nullptr;
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
    delete Put;
}

bool CudaExplicitAlpha_TP::areAlphaMFClose() const
{
    return useAlphaCloseCheck && std::fabs(alphaM - alphaF) <= toleranceAlphaMF;
}

int CudaExplicitAlpha_TP::validateCudaSOE(CudaBcsrLinSOE *&cudaSOE) const
{
    LinearSOE *soe = this->getLinearSOE();
    if (soe == nullptr) {
        return -1;
    }
    cudaSOE = dynamic_cast<CudaBcsrLinSOE *>(soe);
    if (cudaSOE == nullptr || cudaSOE->getBlockSize() != 1 || cudaSOE->getCudaBcsrLinSolver() == nullptr) {
        return -2;
    }
    if (!isUniformPrecision(cudaSOE->getPrecision())) {
        opserr << "ERROR CudaExplicitAlpha_TP::validateCudaSOE() - unsupported SOE precision "
               << cudaPrecisionToString(cudaSOE->getPrecision())
               << "; only uniform dDDI and dFFI are supported\n";
        return -3;
    }
    return 0;
}

int CudaExplicitAlpha_TP::formTangentIntoSOE(int statFlag, double c1v, double c2v, double c3v)
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

int CudaExplicitAlpha_TP::formTangent(int statFlag)
{
    (void)statFlag;
    return 0;  // operators assembled in formOperators(); Linear::formTangent is a no-op
}

int CudaExplicitAlpha_TP::formEleTangent(FE_Element *theEle)
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

int CudaExplicitAlpha_TP::formNodTangent(DOF_Group *theDof)
{
    if (operatorsBuilt) {
        return 0;
    }
    theDof->zeroTangent();
    theDof->addCtoTang(c2);
    theDof->addMtoTang(c3);
    return 0;
}

// TP residual weights (residM/D/R/P) switch between the two passes in newStep().
int CudaExplicitAlpha_TP::formEleResidual(FE_Element *theEle)
{
    theEle->zeroResidual();
    theEle->addRIncInertiaToResidual(residR);
    theEle->addM_Force(*Udotdot, residR - residM);
    return 0;
}

int CudaExplicitAlpha_TP::formNodUnbalance(DOF_Group *theDof)
{
    theDof->zeroUnbalance();
    theDof->addPtoUnbalance(residP);
    theDof->addD_Force(*Udot, -residD);
    theDof->addM_Force(*Udotdot, -residM);
    return 0;
}

int CudaExplicitAlpha_TP::domainChanged()
{
    CudaBcsrLinSOE *cudaSOE = nullptr;
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
    cudaSOE->setXSyncMode(false);  // keep X on device between solve and updateState

    m_impl->destroySolvers();
    m_impl->allocate(size, areAlphaMFClose() ? 1 : 2);
    const ImplBase::HostMotionPtrs buf = m_impl->ensureHostMotionBuffers(size);
    Ut->setData(buf.Ut, size);
    Utdot->setData(buf.Utdot, size);
    Utdotdot->setData(buf.Utdotdot, size);
    U->setData(buf.U, size);
    Udot->setData(buf.Udot, size);
    Udotdot->setData(buf.Uddot, size);
    Put->setData(m_impl->getPutHostPtr(), size);  // *Put aliases pinned h_put
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

// Per-step flow (host residual assembly + GPU predictor/RHS):
//   1) newStep: TransientIntegrator::formUnbalance at t_n   -> d_put <- device B^(1)
//   2) GPU predictor                                      -> trial (U, Udot, Uddot) at t_{n+1}
//   3) newStep: TransientIntegrator::formUnbalance at t_{n+1} -> blend Put with B^(2)
// Linear::solveCurrentStep then calls CudaExplicitAlpha_TP::formUnbalance() (Put -> B) and update() (accel only).
int CudaExplicitAlpha_TP::newStep(double _deltaT)
{
    updateCount = 0;
    if (alphaF < 0.5 || alphaF > 1.0 || beta <= 0.0 || gamma <= 0.0 || _deltaT <= 0.0) {
        return -1;
    }
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr || m_impl == nullptr) {
        return -2;
    }
    CudaBcsrLinSOE *cudaSOE = nullptr;
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

    // --- First pass at t_n: no explicit nodal M*a term (residM = 0) ---
    residD = residR = residP = 1.0;
    residM = 0.0;

    double time = theModel->getCurrentDomainTime();
    if (theModel->updateDomain(time, deltaT) < 0) {
        return -5;
    }
    if (TransientIntegrator::formUnbalance() < 0) {
        return -6;
    }
    if (m_impl->capturePutFromDeviceB(cudaSOE) != 0) {
        return -6;
    }

    if (m_impl->newStepPredictor(this) != 0) {
        return -7;
    }

    // Full-step trial kinematics for the second pass (not alpha-interpolated).
    theModel->setResponse(*U, *Udot, *Udotdot);

    // --- Second pass at t_{n+1}: include nodal M*a (residM = 1) ---
    time += deltaT;
    if (theModel->updateDomain(time, deltaT) < 0) {
        return -8;
    }

    residM = 1.0;
    if (TransientIntegrator::formUnbalance() < 0) {
        return -9;
    }

    m_impl->blendPutFromHostB(cudaSOE, 1.0 - alphaF, alphaF);
    return 0;
}

int CudaExplicitAlpha_TP::formUnbalance()
{
    if (m_impl == nullptr) {
        return -1;
    }
    CudaBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) != 0) {
        return -2;
    }
    // Uses blended Put from newStep(); does not re-assemble equilibrium residual.
    return m_impl->formUnbalanceFromPut(cudaSOE, areAlphaMFClose());
}

int CudaExplicitAlpha_TP::update(const Vector &aiPlusOne)
{
    updateCount++;
    if (updateCount > 1) {
        return -1;  // linear algorithm only
    }
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr || U->Size() <= 0 || aiPlusOne.Size() != U->Size() || m_impl == nullptr) {
        return -2;
    }
    CudaBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) != 0) {
        return -4;
    }
    if (m_impl->updateState(cudaSOE, incrementalAccel, alphaF) != 0) {
        return -4;
    }
    theModel->setAccel(*Udotdot);
    if (theModel->updateDomain() < 0) {
        return -3;
    }
    return 0;
}

int CudaExplicitAlpha_TP::commit()
{
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr) {
        return -1;
    }
    // Domain time was advanced to t_{n+1} during the second updateDomain in newStep().
    return theModel->commitDomain();
}

int CudaExplicitAlpha_TP::revertToLastStep()
{
    CudaBcsrLinSOE *cudaSOE = nullptr;
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

double CudaExplicitAlpha_TP::getCFactor() { return c2; }

const Vector &CudaExplicitAlpha_TP::getVel() { return *Udot; }  // end-of-step velocity, not alpha-level

int CudaExplicitAlpha_TP::sendSelf(int cTag, Channel &theChannel)
{
    Vector data(6);
    data(0) = alphaM;
    data(1) = alphaF;
    data(2) = beta;
    data(3) = gamma;
    data(4) = incrementalAccel ? 1.0 : 0.0;
    data(5) = useAlphaCloseCheck ? 1.0 : 0.0;
    return theChannel.sendVector(this->getDbTag(), cTag, data);
}

int CudaExplicitAlpha_TP::recvSelf(int cTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    Vector data(6);
    if (theChannel.recvVector(this->getDbTag(), cTag, data) < 0) {
        return -1;
    }
    alphaM = data(0);
    alphaF = data(1);
    beta = data(2);
    gamma = data(3);
    incrementalAccel = data(4) > 0.5;
    useAlphaCloseCheck = data(5) > 0.5;
    residM = 0.0;
    residD = alphaF;
    residR = alphaF;
    residP = alphaF;
    operatorsBuilt = false;
    return 0;
}

void CudaExplicitAlpha_TP::Print(OPS_Stream &s, int flag)
{
    (void)flag;
    s << "CudaExplicitAlpha_TP alphaF=" << alphaF << " alphaM=" << alphaM;
    s << endln;
}

int CudaExplicitAlpha_TP::revertToStart()
{
    CudaBcsrLinSOE *cudaSOE = nullptr;
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
        Put->Zero();
        if (m_impl != nullptr && m_impl->getSize() > 0) {
            m_impl->zeroState();
        }
    }
    return 0;
}
