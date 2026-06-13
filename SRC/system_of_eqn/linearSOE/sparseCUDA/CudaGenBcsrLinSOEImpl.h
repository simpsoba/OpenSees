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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCuda/CudaGenBcsrLinSOEImpl.h
                                                                        
// Written: gaaraujo 
// Created: 08/2025
//
// Description: This file contains the internal template implementation for 
// CudaGenBcsrLinSOE. This is an internal implementation detail and should
// not be included by users of the CudaGenBcsrLinSOE interface.
//

#ifndef CudaGenBcsrLinSOEImpl_h
#define CudaGenBcsrLinSOEImpl_h

#ifndef _CUDA
#error "CudaGenBcsrLinSOEImpl.h requires a CUDA build"
#endif

// OpenSees includes
#include <CudaGenBcsrLinSOE.h>
#include <CudaGenBcsrLinSolver.h>

// CUDA includes
#include <cuda_runtime.h>

// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/memory.h>
#include <thrust/system/cuda/execution_policy.h>

using thrust::raw_pointer_cast;

namespace CudaGenBcsrLinSOEDetail {
int addScalarDiagonalToA(int numEqn, int blockSize, const int *rowPtr, const int *colIdx, void *values,
                           CudaPrecision prec, const void *deviceScalarDiag, double scale, cudaStream_t stream);
int addBlockDiagonalToA(int numBlockRows, int blockSize, const int *rowPtr, const int *colIdx, void *values,
                        CudaPrecision prec, const double *deviceBlockDiag, double scale, cudaStream_t stream);
}

// C++ includes
#include <type_traits>

// Forward declarations
class CudaGenBcsrLinSolver;

namespace {
template<typename VectorType>
struct AddDoubleToB {
    __device__ __host__ VectorType operator()(double a, VectorType b) const { return b + static_cast<VectorType>(a); }
};
}

// This template class provides the actual implementation for different data types.
// It inherits from CudaGenBcsrLinSOE and implements all the pure virtual methods.
// The template parameters allow us to specialize matrix and vector types independently:
//   - MatrixType: Type for matrix values (A)
//   - VectorType: Type for vectors (x, b)
// 
// Currently instantiated combinations:
//   - CudaGenBcsrLinSOEImpl<double, double> (dDDI) - uniform double precision
//   - CudaGenBcsrLinSOEImpl<float, float>   (dFFI) - uniform float precision
// 
// Additional combinations can be instantiated when needed:
//   - CudaGenBcsrLinSOEImpl<double, float>  (dDFI) - double matrix, float vectors
//   - CudaGenBcsrLinSOEImpl<float, double>  (dFDI) - float matrix, double vectors
// 
// Note: Whether a precision mode is "supported" depends on the solver, not the SOE.
//       The SOE simply provides the data in the requested format.
template<typename MatrixType, typename VectorType = MatrixType>
class CudaGenBcsrLinSOEImpl : public CudaGenBcsrLinSOE
{
public:
    // Constructor with solver
    explicit CudaGenBcsrLinSOEImpl(
        CudaGenBcsrLinSolver &theSolver, 
        int blockSize = DEFAULT_BLOCK_SIZE, 
        bool paddingEnabled = true,
        bool verbose = false,
        bool symmetricStorage = false
    )
    : CudaGenBcsrLinSOE(CudaGenBcsrLinSOEImpl<MatrixType, VectorType>::getClassTagForType(), theSolver, blockSize, paddingEnabled, verbose, symmetricStorage),
      m_deviceAValues(), m_deviceX(), m_deviceB(), m_deviceCsrIndices()
    {
        // Now that the derived class is fully constructed, we can safely call setLinearSOE
        theSolver.setLinearSOE(*this);
    }
    
    // Default constructor
    CudaGenBcsrLinSOEImpl()
    : CudaGenBcsrLinSOE(CudaGenBcsrLinSOEImpl<MatrixType, VectorType>::getClassTagForType()),
      m_deviceAValues(), m_deviceX(), m_deviceB(), m_deviceCsrIndices()
    {
        /* Nothing to do here */
    }

    // Destructor
    ~CudaGenBcsrLinSOEImpl() = default;

    // Disable copy constructor and assignment
    CudaGenBcsrLinSOEImpl(const CudaGenBcsrLinSOEImpl&) = delete;
    CudaGenBcsrLinSOEImpl& operator=(const CudaGenBcsrLinSOEImpl&) = delete;

    // Move constructor and assignment
    CudaGenBcsrLinSOEImpl(CudaGenBcsrLinSOEImpl&&) = default;
    CudaGenBcsrLinSOEImpl& operator=(CudaGenBcsrLinSOEImpl&&) = default;

    // Required methods for CudaGenBcsrLinSOE subclasses
    // These methods provide type-erased access to device data.
    // The void* return type allows the solver to work with any data type without knowing the specifics.
    const void* getDeviceAValues(void) const noexcept override {
        return m_deviceAValues.empty() ? nullptr : m_deviceAValues.data().get();
    }
    
    void* getDeviceAValues(void) noexcept override {
        return m_deviceAValues.empty() ? nullptr : m_deviceAValues.data().get();
    }
    
    const void* getDeviceX(void) const noexcept override {
        return m_deviceX.empty() ? nullptr : m_deviceX.data().get();
    }
    
    void* getDeviceX(void) noexcept override {
        return m_deviceX.empty() ? nullptr : m_deviceX.data().get();
    }
    
    const void* getDeviceB(void) const noexcept override {
        return m_deviceB.empty() ? nullptr : m_deviceB.data().get();
    }
    
    void* getDeviceB(void) noexcept override {
        return m_deviceB.empty() ? nullptr : m_deviceB.data().get();
    }
    
    // Precision query method
    CudaPrecision getPrecision(void) const noexcept override {
        // Determine precision from template types
        constexpr bool matrixDouble = std::is_same<MatrixType, double>::value;
        constexpr bool vectorDouble = std::is_same<VectorType, double>::value;
        
        if constexpr (matrixDouble && vectorDouble) {
            return CudaPrecision::dDDI;  // Double matrix, Double vectors
        } else if constexpr (!matrixDouble && !vectorDouble) {
            return CudaPrecision::dFFI;  // Float matrix, Float vectors
        } else if constexpr (matrixDouble && !vectorDouble) {
            return CudaPrecision::dDFI;  // Double matrix, Float vectors
        } else {  // !matrixDouble && vectorDouble
            return CudaPrecision::dFDI;  // Float matrix, Double vectors
        }
    }
    
    // Host (double)-device (VectorType) data transfer methods
    inline void uploadVectorsToDevice(void) override {
        m_deviceB = this->CudaGenBcsrLinSOE::m_hostB;
        m_deviceX.resize(this->CudaGenBcsrLinSOE::m_hostX.size());
    }

    inline void downloadRhsFromDevice(void) override {
        this->CudaGenBcsrLinSOE::m_hostB = m_deviceB;
    }
    
    inline void downloadSolutionFromDevice(void) override {
        this->CudaGenBcsrLinSOE::m_hostX = m_deviceX;
        this->CudaGenBcsrLinSOE::m_X.setData(
            raw_pointer_cast(this->CudaGenBcsrLinSOE::m_hostX.data()), 
            this->CudaGenBcsrLinSOE::m_X.Size()
        );
    }
    
    inline void uploadAValuesToDevice(void) override {
        m_deviceAValues = this->CudaGenBcsrLinSOE::m_hostAValues;
    }

    inline void downloadAValuesFromDevice(void) override {
        this->CudaGenBcsrLinSOE::m_hostAValues = m_deviceAValues;
    }

    inline void ensureDeviceVectorSizes(void) override {
        const size_t bSize = static_cast<size_t>(this->CudaGenBcsrLinSOE::m_hostB.size());
        const size_t xSize = static_cast<size_t>(this->CudaGenBcsrLinSOE::m_hostX.size());
        const size_t aSize = this->CudaGenBcsrLinSOE::m_hostAValues.size();
        m_deviceB.resize(bSize);
        m_deviceX.resize(xSize);
        m_deviceAValues.resize(aSize);
    }

    inline void uploadAIndicesToDevice(void) override {
        m_deviceCsrIndices = this->CudaGenBcsrLinSOE::m_hostCsrIndices;
    }

    const int* getDeviceRowPtrs(void) const override {
        return m_deviceCsrIndices.empty() ? nullptr : m_deviceCsrIndices.data().get();
    }

    int* getDeviceRowPtrs(void) override {
        return m_deviceCsrIndices.empty() ? nullptr : m_deviceCsrIndices.data().get();
    }

    const int* getDeviceColIndices(void) const override {
        return m_deviceCsrIndices.empty() ? nullptr
            : m_deviceCsrIndices.data().get() + this->getNumRowBlocks() + 1;
    }

    int* getDeviceColIndices(void) override {
        return m_deviceCsrIndices.empty() ? nullptr
            : m_deviceCsrIndices.data().get() + this->getNumRowBlocks() + 1;
    }

    int addScalarDiagonalToA(const void *deviceScalarDiag, double scale, void *stream) override
    {
        const int n = this->getNumEqn();
        if (scale == 0.0 || n <= 0 || deviceScalarDiag == nullptr) {
            return 0;
        }
        const int *rowPtr = getDeviceRowPtrs();
        const int *colIdx = getDeviceColIndices();
        void *values = getDeviceAValues();
        if (rowPtr == nullptr || colIdx == nullptr || values == nullptr) {
            return -1;
        }
        const int rc = CudaGenBcsrLinSOEDetail::addScalarDiagonalToA(
            n, this->getBlockSize(), rowPtr, colIdx, values, this->getPrecision(), deviceScalarDiag, scale,
            stream != nullptr ? static_cast<cudaStream_t>(stream) : cudaStream_t{0});
        if (rc == 0 && this->m_matrixStatus == MatrixStatus::UNCHANGED) {
            this->m_matrixStatus = MatrixStatus::COEFFICIENTS_CHANGED;
        }
        return rc;
    }

    int addBlockDiagonalToA(const double *deviceBlockDiag, double scale, void *stream) override
    {
        const int bs = this->getBlockSize();
        if (bs <= 1) {
            return -1;
        }
        const int numBlockRows = this->getNumRowBlocks();
        if (scale == 0.0 || numBlockRows <= 0 || bs <= 0 || deviceBlockDiag == nullptr) {
            return 0;
        }
        const int *rowPtr = getDeviceRowPtrs();
        const int *colIdx = getDeviceColIndices();
        void *values = getDeviceAValues();
        if (rowPtr == nullptr || colIdx == nullptr || values == nullptr) {
            return -1;
        }
        const int rc = CudaGenBcsrLinSOEDetail::addBlockDiagonalToA(
            numBlockRows, bs, rowPtr, colIdx, values, this->getPrecision(), deviceBlockDiag, scale,
            stream != nullptr ? static_cast<cudaStream_t>(stream) : cudaStream_t{0});
        if (rc == 0 && this->m_matrixStatus == MatrixStatus::UNCHANGED) {
            this->m_matrixStatus = MatrixStatus::COEFFICIENTS_CHANGED;
        }
        return rc;
    }

    inline void addToDeviceBFromHost(int n, const double* hostData, void* stream) override {
        if (n <= 0 || hostData == nullptr) return;
        const size_t un = static_cast<size_t>(n);
        if (m_deviceB.size() < un) return;
        if (m_deviceStagingB.size() < un)
            m_deviceStagingB.resize(un);
        cudaStream_t s = (stream != nullptr) ? static_cast<cudaStream_t>(stream) : 0;
        cudaMemcpyAsync(raw_pointer_cast(m_deviceStagingB.data()), hostData, un * sizeof(double), cudaMemcpyHostToDevice, s);
        if (s != 0)
            thrust::transform(thrust::cuda::par.on(s),
                m_deviceStagingB.begin(), m_deviceStagingB.begin() + n,
                m_deviceB.begin(), m_deviceB.begin(), AddDoubleToB<VectorType>());
        else
            thrust::transform(thrust::cuda::par,
                m_deviceStagingB.begin(), m_deviceStagingB.begin() + n,
                m_deviceB.begin(), m_deviceB.begin(), AddDoubleToB<VectorType>());
    }

private:    
    // Device memory storage using Thrust vectors
    thrust::device_vector<VectorType> m_deviceX, m_deviceB;
    thrust::device_vector<MatrixType> m_deviceAValues;
    thrust::device_vector<double> m_deviceStagingB;
    thrust::device_vector<int> m_deviceCsrIndices;

public:
    // Helper function to get class tag for type
    static int getClassTagForType() {
        constexpr bool matrixDouble = std::is_same_v<MatrixType, double>;
        constexpr bool vectorDouble = std::is_same_v<VectorType, double>;
        
        if constexpr (matrixDouble && vectorDouble) {
            return LinSOE_TAGS_CudaBcsrLinSOE_DOUBLE;        // dDDI
        } else if constexpr (!matrixDouble && !vectorDouble) {
            return LinSOE_TAGS_CudaBcsrLinSOE_FLOAT;         // dFFI
        } else if constexpr (matrixDouble && !vectorDouble) {
            return LinSOE_TAGS_CudaBcsrLinSOE_DOUBLE_FLOAT;  // dDFI
        } else {  // !matrixDouble && vectorDouble
            return LinSOE_TAGS_CudaBcsrLinSOE_FLOAT_DOUBLE;  // dFDI
        }
    }

private:
};

// Explicit template instantiations
// All four precision modes are instantiated and ready to use.
// Solvers validate which modes they support at construction/connection time.
template class CudaGenBcsrLinSOEImpl<double, double>;  // dDDI - uniform double precision
template class CudaGenBcsrLinSOEImpl<float, float>;    // dFFI - uniform float precision
template class CudaGenBcsrLinSOEImpl<double, float>;   // dDFI - double matrix, float vectors (mixed)
template class CudaGenBcsrLinSOEImpl<float, double>;   // dFDI - float matrix, double vectors (mixed)

#endif
