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

// OpenSees includes
#include <CudaGenBcsrLinSOE.h>
#include <CudaGenBcsrLinSolver.h>
#ifdef _CUDA
// CUDA includes
#include <cuda_runtime.h>

// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/memory.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <iterator>

// Bring thrust::raw_pointer_cast into scope for CUDA builds
using thrust::raw_pointer_cast;
#else
#include <vector>
#include <algorithm>

// Define a passthrough raw_pointer_cast for non-CUDA builds
template<typename T>
inline T* raw_pointer_cast(T* ptr) { return ptr; }
#endif

// C++ includes
#include <type_traits>

// Forward declarations
class CudaGenBcsrLinSolver;

#ifdef _CUDA
namespace {
template<typename VectorType>
struct AddDoubleToB {
    __device__ __host__ VectorType operator()(double a, VectorType b) const { return b + static_cast<VectorType>(a); }
};
struct CompareTupleByFirst {
    __device__ __host__ bool operator()(const thrust::tuple<int,int,int,double>& a, const thrust::tuple<int,int,int,double>& b) const {
        return thrust::get<0>(a) < thrust::get<0>(b);
    }
};
struct MakeKey {
    int n;
    explicit MakeKey(int n_) : n(n_) {}
    __device__ __host__ int operator()(int r, int c) const { return r * n + c; }
};
struct MakeBlockKey {
    int numBlockCols;
    int blockSize;
    MakeBlockKey(int nbc, int bs) : numBlockCols(nbc), blockSize(bs) {}
    __device__ __host__ int operator()(int row, int col) const {
        int blockRow = row / blockSize;
        int blockCol = col / blockSize;
        int localRow = row % blockSize;
        int localCol = col % blockSize;
        int bs2 = blockSize * blockSize;
        return (blockRow * numBlockCols + blockCol) * bs2 + (localRow * blockSize + localCol);
    }
};
struct LowerTriangleOnly {
    __device__ __host__ bool operator()(const thrust::tuple<int, int, double>& t) const {
        return thrust::get<0>(t) >= thrust::get<1>(t);
    }
};
// Scatter reduced (unique key, summed value) COO into block CSR. Keys encode (blockRow, blockCol, localRow, localCol).
template<typename MatrixType>
__global__ void scatterCOOToCSRKernel(
    const int* __restrict__ rowPtr,
    const int* __restrict__ colInd,
    MatrixType* __restrict__ Avalues,
    const int* __restrict__ keys,
    const double* __restrict__ vals,
    int numBlockCols,
    int blockSize,
    int nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    // Decode key: key = (blockRow * numBlockCols + blockCol) * bs2 + (localRow * blockSize + localCol)
    int key = keys[i];
    int bs2 = blockSize * blockSize;
    int localCol = key % blockSize;
    key /= blockSize;
    int localRow = key % blockSize;
    key /= blockSize;
    int blockCol = key % numBlockCols;
    int blockRow = key / numBlockCols;
    double v = vals[i];
    // Find block index k in block CSR for this (blockRow, blockCol)
    int start = rowPtr[blockRow];
    int end = rowPtr[blockRow + 1];
    for (int k = start; k < end; k++) {
        if (colInd[k] == blockCol) {
            // Write into block at entry (localRow, localCol)
            int offset = k * bs2 + localRow * blockSize + localCol;
            Avalues[offset] = static_cast<MatrixType>(v);
            return;
        }
    }
}
}
#endif

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
      m_deviceAValues(), m_deviceX(), m_deviceB()
    {
        // Now that the derived class is fully constructed, we can safely call setLinearSOE
        theSolver.setLinearSOE(*this);
    }
    
    // Default constructor
    CudaGenBcsrLinSOEImpl()
    : CudaGenBcsrLinSOE(CudaGenBcsrLinSOEImpl<MatrixType, VectorType>::getClassTagForType()),
      m_deviceAValues(), m_deviceX(), m_deviceB()
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
        #ifdef _CUDA
        return m_deviceAValues.empty() ? nullptr : m_deviceAValues.data().get(); 
        #else
        return m_deviceAValues.empty() ? nullptr : m_deviceAValues.data(); 
        #endif
    }
    
    void* getDeviceAValues(void) noexcept override { 
        #ifdef _CUDA
        return m_deviceAValues.empty() ? nullptr : m_deviceAValues.data().get(); 
        #else
        return m_deviceAValues.empty() ? nullptr : m_deviceAValues.data(); 
        #endif
    }
    
    const void* getDeviceX(void) const noexcept override { 
        #ifdef _CUDA
        return m_deviceX.empty() ? nullptr : m_deviceX.data().get(); 
        #else
        return m_deviceX.empty() ? nullptr : m_deviceX.data(); 
        #endif
    }
    
    void* getDeviceX(void) noexcept override { 
        #ifdef _CUDA
        return m_deviceX.empty() ? nullptr : m_deviceX.data().get(); 
        #else
        return m_deviceX.empty() ? nullptr : m_deviceX.data(); 
        #endif
    }
    
    const void* getDeviceB(void) const noexcept override { 
        #ifdef _CUDA
        return m_deviceB.empty() ? nullptr : m_deviceB.data().get(); 
        #else
        return m_deviceB.empty() ? nullptr : m_deviceB.data(); 
        #endif
    }
    
    void* getDeviceB(void) noexcept override { 
        #ifdef _CUDA
        return m_deviceB.empty() ? nullptr : m_deviceB.data().get(); 
        #else
        return m_deviceB.empty() ? nullptr : m_deviceB.data(); 
        #endif
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
        #ifdef _CUDA
        // Use thrust for transfer (handles type conversion)
        m_deviceB = this->CudaGenBcsrLinSOE::m_hostB;
        m_deviceX.resize(this->CudaGenBcsrLinSOE::m_hostX.size());
        #else
        m_deviceB.resize(this->CudaGenBcsrLinSOE::m_hostB.size());
        std::transform(
            this->CudaGenBcsrLinSOE::m_hostB.begin(),
            this->CudaGenBcsrLinSOE::m_hostB.end(),
            m_deviceB.begin(),
            [](double val){ return static_cast<VectorType>(val); } // convert to device type
        );
        m_deviceX.resize(this->CudaGenBcsrLinSOE::m_hostX.size());
        #endif
    }
    
    inline void downloadSolutionFromDevice(void) override {
        #ifdef _CUDA
        // Use thrust for transfer (handles type conversion)
        this->CudaGenBcsrLinSOE::m_hostX = m_deviceX;
        #else
        this->CudaGenBcsrLinSOE::m_hostX.resize(m_deviceX.size());
        std::transform(
            m_deviceX.begin(),
            m_deviceX.end(),
            this->CudaGenBcsrLinSOE::m_hostX.begin(),
            [](VectorType val){ return static_cast<double>(val); } // convert to host type
        );
        #endif
        this->CudaGenBcsrLinSOE::m_X.setData(
            raw_pointer_cast(this->CudaGenBcsrLinSOE::m_hostX.data()), 
            this->CudaGenBcsrLinSOE::m_X.Size()
        );
    }
    
    inline void uploadAValuesToDevice(void) override {
        #ifdef _CUDA
        // Use thrust for transfer (handles type conversion)
        m_deviceAValues = this->CudaGenBcsrLinSOE::m_hostAValues;
        #else
        m_deviceAValues.resize(this->CudaGenBcsrLinSOE::m_hostAValues.size());
        std::transform(
            this->CudaGenBcsrLinSOE::m_hostAValues.begin(),
            this->CudaGenBcsrLinSOE::m_hostAValues.end(),
            m_deviceAValues.begin(),
            [](double val){ return static_cast<MatrixType>(val); } // convert to device type
        );
        #endif
    }

    inline void downloadAValuesFromDevice(void) override {
        #ifdef _CUDA
        this->CudaGenBcsrLinSOE::m_hostAValues = m_deviceAValues;
        #else
        this->CudaGenBcsrLinSOE::m_hostAValues.resize(m_deviceAValues.size());
        std::transform(
            m_deviceAValues.begin(),
            m_deviceAValues.end(),
            this->CudaGenBcsrLinSOE::m_hostAValues.begin(),
            [](MatrixType val){ return static_cast<double>(val); }
        );
        #endif
    }

    inline void ensureDeviceVectorSizes(void) override {
        const size_t bSize = static_cast<size_t>(this->CudaGenBcsrLinSOE::m_hostB.size());
        const size_t xSize = static_cast<size_t>(this->CudaGenBcsrLinSOE::m_hostX.size());
        const size_t aSize = this->CudaGenBcsrLinSOE::m_hostAValues.size();
        m_deviceB.resize(bSize);
        m_deviceX.resize(xSize);
        m_deviceAValues.resize(aSize);
    }

    inline void addToDeviceBFromHost(int n, const double* hostData, void* stream) override {
        if (n <= 0 || hostData == nullptr) return;
        #ifdef _CUDA
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
        #else
        (void)stream;
        for (int i = 0; i < n && i < static_cast<int>(m_deviceB.size()); i++)
            m_deviceB[static_cast<size_t>(i)] += static_cast<VectorType>(hostData[i]);
        #endif
    }

    int assembleAFromCOO(int nnz, const int* rows, const int* cols, const double* vals) override {
        const int numBlockRows = this->getNumRowBlocks();
        const int blockSize = this->getBlockSize();
        if (numBlockRows <= 0 || blockSize <= 0 || this->m_hostAValues.empty())
            return -1;
        #ifdef _CUDA
        // Ensure device has structure (rowPtr, colInd) and vector sizes (B, X, Avalues)
        this->uploadAIndicesToDevice();
        this->ensureDeviceVectorSizes();
        if (nnz <= 0) {
            thrust::fill(m_deviceAValues.begin(), m_deviceAValues.end(), static_cast<MatrixType>(0));
            return 0;
        }
        // Copy host COO to device
        const size_t un = static_cast<size_t>(nnz);
        thrust::device_vector<int> d_row(rows, rows + un);
        thrust::device_vector<int> d_col(cols, cols + un);
        thrust::device_vector<double> d_val(vals, vals + un);
        size_t work_nnz = un;
        // Symmetric lower: keep only row >= col to avoid double-counting; replace working buffers with filtered
        if (this->getMatrixStorageMode() == CudaGenBcsrLinSOE::MatrixStorageMode::SYMMETRIC_LOWER) {
            thrust::device_vector<int> d_row_filt(un), d_col_filt(un);
            thrust::device_vector<double> d_val_filt(un);
            auto zip_in = thrust::make_zip_iterator(thrust::make_tuple(d_row.begin(), d_col.begin(), d_val.begin()));
            auto zip_out = thrust::make_zip_iterator(thrust::make_tuple(d_row_filt.begin(), d_col_filt.begin(), d_val_filt.begin()));
            auto end_out = thrust::copy_if(thrust::device, zip_in, zip_in + nnz, zip_out, LowerTriangleOnly());
            work_nnz = static_cast<size_t>(std::distance(zip_out, end_out));
            if (work_nnz == 0) {
                thrust::fill(m_deviceAValues.begin(), m_deviceAValues.end(), static_cast<MatrixType>(0));
                this->m_matrixStatus = CudaGenBcsrLinSOE::MatrixStatus::COEFFICIENTS_CHANGED;
                return 0;
            }
            d_row = std::move(d_row_filt);
            d_col = std::move(d_col_filt);
            d_val = std::move(d_val_filt);
        }
        // Assign a unique key per (blockRow, blockCol, localRow, localCol) for block CSR ordering
        const int numBlockCols = numBlockRows;
        thrust::device_vector<int> d_keys(work_nnz);
        thrust::transform(d_row.begin(), d_row.begin() + work_nnz, d_col.begin(),
            d_keys.begin(), MakeBlockKey(numBlockCols, blockSize));
        // Sort by key so duplicate keys are adjacent
        auto zip = thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_row.begin(), d_col.begin(), d_val.begin()));
        thrust::sort(thrust::device, zip, zip + work_nnz, CompareTupleByFirst());
        // Sum values for identical keys; output is one (key, sum) per unique key
        thrust::device_vector<int> d_keys_unique(work_nnz);
        thrust::device_vector<double> d_vals_reduced(work_nnz);
        auto end = thrust::reduce_by_key(thrust::device,
            d_keys.begin(), d_keys.begin() + work_nnz, d_val.begin(),
            d_keys_unique.begin(), d_vals_reduced.begin(),
            thrust::equal_to<int>(), thrust::plus<double>());
        int reduced_nnz = static_cast<int>(end.first - d_keys_unique.begin());
        // Zero A then scatter each (key, val) into block CSR
        thrust::fill(m_deviceAValues.begin(), m_deviceAValues.end(), static_cast<MatrixType>(0));
        int* rowPtr = this->getDeviceRowPtrs();
        int* colInd = this->getDeviceColIndices();
        MatrixType* Avalues = raw_pointer_cast(m_deviceAValues.data());
        if (rowPtr && colInd && Avalues && reduced_nnz > 0) {
            int block = 256;
            int grid = (reduced_nnz + block - 1) / block;
            scatterCOOToCSRKernel<MatrixType><<<grid, block>>>(
                rowPtr, colInd, Avalues,
                raw_pointer_cast(d_keys_unique.data()),
                raw_pointer_cast(d_vals_reduced.data()),
                numBlockCols, blockSize, reduced_nnz);
            if (cudaGetLastError() != cudaSuccess)
                return -1;
        }
        this->m_matrixStatus = CudaGenBcsrLinSOE::MatrixStatus::COEFFICIENTS_CHANGED;
        return 0;
        #else
        (void)nnz; (void)rows; (void)cols; (void)vals;
        return -1;
        #endif
    }

private:    
    // Device memory storage using Thrust vectors
    #ifdef _CUDA
    thrust::device_vector<VectorType> m_deviceX, m_deviceB;
    thrust::device_vector<MatrixType> m_deviceAValues;
    thrust::device_vector<double> m_deviceStagingB;
    #else
    std::vector<VectorType> m_deviceX, m_deviceB;
    std::vector<MatrixType> m_deviceAValues;
    #endif

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
