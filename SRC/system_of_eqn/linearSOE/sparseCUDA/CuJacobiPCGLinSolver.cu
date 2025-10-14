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

// Written: gaaraujo
// Created: 10/2025
//
// Description: Conjugate Gradient solver with diagonal Jacobi preconditioning

// OpenSees includes
#include <classTags.h>
#include <OPS_Globals.h>

// Solver core classes
#include <CudaGenBcsrLinSOE.h>
#include <CuJacobiPCGLinSolver.h>

// CUDA utilities
#include "CudaUtils.h"

// For parsing command line arguments
#include <elementAPI.h>
#include <unordered_map>
#include <functional>
#include "ParameterUtils.h"

// C++ includes
#include <vector>
#include <string>
#include <cstring>
#include <cmath>

#ifdef _CUDA
// Use CudaUtils namespace for error checking and helpers
using namespace CudaUtils;
#endif // _CUDA

CuJacobiPCGLinSolver::CuJacobiPCGLinSolver(
    int maxIterations, double relativeTolerance, double absoluteTolerance, bool verbose)
    : CuPCGLinSolver(
        SOLVER_TAGS_CuJacobiPCGLinSolver,  // Our own class tag
        nullptr,  // No external preconditioner - we implement our own
        maxIterations, 
        relativeTolerance, 
        absoluteTolerance,
        0,        // updateFrequency = 0 (diagonal extracted lazily on first use)
        false,    // updateOnFailure = false (not applicable for Jacobi)
        verbose)
{
    #ifdef _CUDA
    m_d_diagInv = nullptr;
    m_diagAllocatedSize = 0;
    m_diagonalExtracted = false;
    #endif // _CUDA
}

CuJacobiPCGLinSolver::~CuJacobiPCGLinSolver()
{
    #ifdef _CUDA
    if (m_d_diagInv != nullptr) {
        cudaCheckError(cudaFree(m_d_diagInv), "free diagonal inverse", false);
        m_d_diagInv = nullptr;
    }
    #endif // _CUDA
}

#ifdef _CUDA

// ============================================================================
// CUDA kernels for extracting diagonal from CSR/BSR matrix
// ============================================================================

// Kernel: Extract diagonal from CSR matrix (blockSize = 1)
// For each row i, find the diagonal element A[i,i] and compute 1/A[i,i]
template<typename T>
__global__ void extractDiagonalCSR_kernel(
    const int* __restrict__ rowPtrs,
    const int* __restrict__ colIndices,
    const T* __restrict__ values,
    T* __restrict__ diagInv,
    int numRows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int row = i; row < numRows; row += stride) {
        int rowStart = rowPtrs[row];
        int rowEnd = rowPtrs[row + 1];
        
        // Find diagonal element in this row
        T diagValue = 0.0;
        bool found = false;
        for (int j = rowStart; j < rowEnd; j++) {
            if (colIndices[j] == row) {
                diagValue = values[j];
                found = true;
                break;
            }
        }
        
        // Store inverse (with safeguard for zero/small diagonals)
        if (found && fabs(diagValue) > 1e-15) {
            diagInv[row] = 1.0 / diagValue;
        } else {
            diagInv[row] = 1.0;  // Identity for zero/missing diagonal
        }
    }
}

// Kernel: Extract diagonal from BSR matrix (blockSize > 1)
// For each block row i, extract the diagonal block and compute inverse of diagonal elements
template<typename T>
__global__ void extractDiagonalBSR_kernel(
    const int* __restrict__ rowPtrs,
    const int* __restrict__ colIndices,
    const T* __restrict__ values,
    T* __restrict__ diagInv,
    int numBlockRows,
    int blockSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int blockRow = idx; blockRow < numBlockRows; blockRow += stride) {
        int rowStart = rowPtrs[blockRow];
        int rowEnd = rowPtrs[blockRow + 1];
        
        // Find diagonal block in this block row
        bool found = false;
        for (int j = rowStart; j < rowEnd; j++) {
            if (colIndices[j] == blockRow) {
                // Found diagonal block
                const T* blockValues = &values[j * blockSize * blockSize];
                
                // Extract diagonal elements from the block (row-major order)
                for (int k = 0; k < blockSize; k++) {
                    T diagValue = blockValues[k * blockSize + k];
                    int scalarIdx = blockRow * blockSize + k;
                    
                    if (fabs(diagValue) > 1e-15) {
                        diagInv[scalarIdx] = 1.0 / diagValue;
                    } else {
                        diagInv[scalarIdx] = 1.0;  // Identity for zero/small diagonal
                    }
                }
                found = true;
                break;
            }
        }
        
        // If diagonal block not found, use identity
        if (!found) {
            for (int k = 0; k < blockSize; k++) {
                int scalarIdx = blockRow * blockSize + k;
                diagInv[scalarIdx] = 1.0;
            }
        }
    }
}

// ============================================================================
// CUDA kernels for applying Jacobi preconditioner: z = diagInv .* r
// ============================================================================

template<typename T>
__global__ void applyJacobi_kernel(
    T* __restrict__ z,
    const T* __restrict__ r,
    const T* __restrict__ diagInv,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        z[i] = diagInv[i] * r[i];
    }
}

// Vectorized version for float (4-way)
__global__ void applyJacobi_kernel_float4(
    float* __restrict__ z,
    const float* __restrict__ r,
    const float* __restrict__ diagInv,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n_vec = n / 4;
    
    float4* z_vec = reinterpret_cast<float4*>(z);
    const float4* r_vec = reinterpret_cast<const float4*>(r);
    const float4* diagInv_vec = reinterpret_cast<const float4*>(diagInv);
    
    for (int i = idx; i < n_vec; i += stride) {
        float4 r_val = __ldg(&r_vec[i]);
        float4 d_val = __ldg(&diagInv_vec[i]);
        
        float4 z_val;
        z_val.x = d_val.x * r_val.x;
        z_val.y = d_val.y * r_val.y;
        z_val.z = d_val.z * r_val.z;
        z_val.w = d_val.w * r_val.w;
        
        z_vec[i] = z_val;
    }
    
    // Handle remainder
    if (idx == 0) {
        for (int i = n_vec * 4; i < n; i++) {
            z[i] = diagInv[i] * r[i];
        }
    }
}

// Vectorized version for double (2-way)
__global__ void applyJacobi_kernel_double2(
    double* __restrict__ z,
    const double* __restrict__ r,
    const double* __restrict__ diagInv,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n_vec = n / 2;
    
    double2* z_vec = reinterpret_cast<double2*>(z);
    const double2* r_vec = reinterpret_cast<const double2*>(r);
    const double2* diagInv_vec = reinterpret_cast<const double2*>(diagInv);
    
    for (int i = idx; i < n_vec; i += stride) {
        double2 r_val = __ldg(&r_vec[i]);
        double2 d_val = __ldg(&diagInv_vec[i]);
        
        double2 z_val;
        z_val.x = d_val.x * r_val.x;
        z_val.y = d_val.y * r_val.y;
        
        z_vec[i] = z_val;
    }
    
    // Handle remainder
    if (idx == 0) {
        for (int i = n_vec * 2; i < n; i++) {
            z[i] = diagInv[i] * r[i];
        }
    }
}

// ============================================================================
// Implementation methods
// ============================================================================

int CuJacobiPCGLinSolver::setSize(void)
{
    // Call parent setSize to allocate PCG workspace
    int result = CuPCGLinSolver::setSize();
    if (result != 0) return result;
    
    // Allocate memory for diagonal inverse (but don't extract values yet)
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "ERROR: CuJacobiPCGLinSolver::setSize() - LinearSOE not set" << endln;
        return -1;
    }
    
    int blockSize = theSOE->getBlockSize();
    int numBlockRows = theSOE->getNumRowBlocks();
    int numScalarRows = numBlockRows * blockSize;
    
    // Determine element size based on vector precision
    CudaPrecision precision = theSOE->getPrecision();
    size_t elementSize;
    if (precision == CudaPrecision::dDDI || precision == CudaPrecision::dFDI) {
        elementSize = sizeof(double);  // Vector type is double
    } else {
        elementSize = sizeof(float);   // Vector type is float
    }
    
    size_t requiredSize = numScalarRows * elementSize;
    
    // Allocate or reallocate if size changed
    if (requiredSize > m_diagAllocatedSize || m_d_diagInv == nullptr) {
        bool isFirstAllocation = (m_d_diagInv == nullptr);
        
        if (m_d_diagInv != nullptr) {
            cudaCheckError(cudaFree(m_d_diagInv), "free old diagonal inverse", false);
            m_d_diagInv = nullptr;
        }
        
        cudaCheckError(cudaMalloc(&m_d_diagInv, requiredSize), "allocate diagonal inverse");
        m_diagAllocatedSize = requiredSize;
        
        // Only print on first allocation and only if verbose mode is enabled
        if (isFirstAllocation && m_verbose) {
            double memoryMB = requiredSize / (1024.0 * 1024.0);
            opserr << "INFO: CuJacobiPCGLinSolver::setSize() - Allocated " 
                   << memoryMB << " MB for diagonal inverse (size=" << numScalarRows << ")" << endln;
        }
    }
    
    // Mark diagonal as not extracted (values will be extracted on first solve when matrix is ready)
    m_diagonalExtracted = false;
    
    return 0;
}

template<typename T>
int CuJacobiPCGLinSolver::extractDiagonalImpl()
{
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "ERROR: CuJacobiPCGLinSolver::extractDiagonalImpl() - LinearSOE not set" << endln;
        return -1;
    }
    
    // Verify memory has been allocated
    if (m_d_diagInv == nullptr) {
        opserr << "ERROR: CuJacobiPCGLinSolver::extractDiagonalImpl() - Diagonal memory not allocated. Call setSize() first." << endln;
        return -1;
    }
    
    int blockSize = theSOE->getBlockSize();
    int numBlockRows = theSOE->getNumRowBlocks();
    int numScalarRows = numBlockRows * blockSize;
    int* rowPtrs = theSOE->getDeviceRowPtrs();
    int* colIndices = theSOE->getDeviceColIndices();
    T* AValues = static_cast<T*>(theSOE->getDeviceAValues());
    
    // Validation
    if (rowPtrs == nullptr || colIndices == nullptr || AValues == nullptr) {
        opserr << "ERROR: CuJacobiPCGLinSolver::extractDiagonalImpl() - Null device pointers" << endln;
        opserr << "  rowPtrs: " << rowPtrs << ", colIndices: " << colIndices << ", AValues: " << AValues << endln;
        return -1;
    }
    
    if (numBlockRows <= 0 || numScalarRows <= 0) {
        opserr << "ERROR: CuJacobiPCGLinSolver::extractDiagonalImpl() - Invalid dimensions" << endln;
        opserr << "  numBlockRows: " << numBlockRows << ", blockSize: " << blockSize << endln;
        return -1;
    }
    
    T* diagInv = static_cast<T*>(m_d_diagInv);
    
    // Additional validation: check the number of non-zeros
    int numNonZeroBlocks = theSOE->getNumNonZeroBlocks();
    if (numNonZeroBlocks <= 0) {
        opserr << "WARNING: CuJacobiPCGLinSolver::extractDiagonalImpl() - Matrix has no non-zero blocks" << endln;
        opserr << "  Initializing diagonal to identity" << endln;
        // Initialize diagonal to identity
        if (sizeof(T) == sizeof(double)) {
            cudaCheckError(cudaMemset(diagInv, 0, numScalarRows * sizeof(T)), "zero diagonal");
            // Set to 1.0 using a kernel would be better, but for now just warn
        }
        return 0;
    }
    
    // Launch kernel to extract diagonal
    const int blockDim = 256;
    
    if (blockSize == 1) {
        // CSR format - extract scalar diagonal
        int numBlocks = (numScalarRows + blockDim - 1) / blockDim;
        extractDiagonalCSR_kernel<T><<<numBlocks, blockDim>>>(
            rowPtrs, colIndices, AValues, diagInv, numScalarRows);
    } else {
        // BSR format - extract diagonal from blocks
        int numBlocks = (numBlockRows + blockDim - 1) / blockDim;
        extractDiagonalBSR_kernel<T><<<numBlocks, blockDim>>>(
            rowPtrs, colIndices, AValues, diagInv, numBlockRows, blockSize);
    }
    
    cudaCheckError(cudaGetLastError(), "extract diagonal kernel");
    cudaCheckError(cudaDeviceSynchronize(), "synchronize after extract diagonal");
    
    m_diagonalExtracted = true;
    return 0;
}

int CuJacobiPCGLinSolver::extractDiagonal(bool force_update)
{
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "ERROR: CuJacobiPCGLinSolver::extractDiagonal() - LinearSOE not set" << endln;
        return -1;
    }
    
    // Check if diagonal has already been extracted and matrix hasn't changed
    if (m_diagonalExtracted && !force_update) {
        CudaGenBcsrLinSOE::MatrixStatus matrixStatus = theSOE->getMatrixStatus();
        if (matrixStatus == CudaGenBcsrLinSOE::MatrixStatus::UNCHANGED) {
            // Matrix unchanged, no need to re-extract diagonal
            return 0;
        }
    }
    
    // Dispatch to typed implementation based on MATRIX precision (not vector!)
    // The diagonal is extracted from the matrix, so we need to match the matrix data type
    CudaPrecision precision = theSOE->getPrecision();
    
    // Precision modes: dDDI (double matrix), dFFI (float matrix), dFDI (float matrix), dDFI (double matrix)
    if (precision == CudaPrecision::dDDI || precision == CudaPrecision::dDFI) {
        // Matrix is double (dDDI: double/double, dDFI: double/float)
        return extractDiagonalImpl<double>();
    } else {
        // Matrix is float (dFFI: float/float, dFDI: float/double)
        return extractDiagonalImpl<float>();
    }
}

template<typename T>
int CuJacobiPCGLinSolver::applyJacobiPreconditionerImpl(T* z, T* r, int n)
{
    if (m_d_diagInv == nullptr) {
        opserr << "ERROR: CuJacobiPCGLinSolver::applyJacobiPreconditionerImpl() - "
               << "Diagonal not extracted. Call setSize() first." << endln;
        return -1;
    }
    
    T* diagInv = static_cast<T*>(m_d_diagInv);
    
    // Launch vectorized kernel
    const int blockSize = 256;
    
    if (sizeof(T) == sizeof(float)) {
        int numBlocks = (n / 4 + blockSize - 1) / blockSize;
        applyJacobi_kernel_float4<<<numBlocks, blockSize>>>(
            reinterpret_cast<float*>(z),
            reinterpret_cast<const float*>(r),
            reinterpret_cast<const float*>(diagInv),
            n);
    } else {
        int numBlocks = (n / 2 + blockSize - 1) / blockSize;
        applyJacobi_kernel_double2<<<numBlocks, blockSize>>>(
            reinterpret_cast<double*>(z),
            reinterpret_cast<const double*>(r),
            reinterpret_cast<const double*>(diagInv),
            n);
    }
    
    cudaCheckError(cudaGetLastError(), "apply Jacobi preconditioner kernel");
    
    return 0;
}

int CuJacobiPCGLinSolver::applyPreconditioner(void* z, void* r, int n, bool updatePreconditioner)
{
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "ERROR: CuJacobiPCGLinSolver::applyPreconditioner() - LinearSOE not set" << endln;
        return -1;
    }
    
    // Extract diagonal on first use or if update requested
    if (!m_diagonalExtracted || updatePreconditioner) {
        int result = extractDiagonal(true);
        if (result != 0) return result;
    }
    
    // Dispatch to typed implementation based on vector precision
    CudaPrecision precision = theSOE->getPrecision();
    
    if (precision == CudaPrecision::dDDI || precision == CudaPrecision::dFDI) {
        // Vector type is double
        return applyJacobiPreconditionerImpl<double>((double*)z, (double*)r, n);
    } else {
        // Vector type is float
        return applyJacobiPreconditionerImpl<float>((float*)z, (float*)r, n);
    }
}

#endif // _CUDA

// Note: CuJacobiPCGLinSolver is invoked via:
//   system CuPCG -preconditioner Jacobi [options]
// See OPS_CuPCGLinSolver() in CuPCGLinSolver.cu for the parser implementation.

