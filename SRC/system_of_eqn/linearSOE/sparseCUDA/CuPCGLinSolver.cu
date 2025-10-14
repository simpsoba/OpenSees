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
// Description: General PCG solver with arbitrary preconditioner

// OpenSees includes
#include <classTags.h>
#include <OPS_Globals.h>

// Solver core classes
#include <CudaGenBcsrLinSOE.h>
#include <CuPCGLinSolver.h>
#include <CuDSSLinSolver.h>

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

// Forward declarations of factory functions from other solver files
#ifdef _CUDSS
CudaGenBcsrLinSolver* createCuDSSSolverFromParser();
#endif

#ifdef _AMGX
CudaGenBcsrLinSolver* createAmgXSolverFromParser();
#endif

#ifdef _CUDA
// Static member initialization
bool CuPCGLinSolver::m_CuSparseInitialized = false;
int CuPCGLinSolver::m_ActiveSolverInstances = 0;
cusparseHandle_t CuPCGLinSolver::m_cuSparseHandle = nullptr;
cublasHandle_t CuPCGLinSolver::m_cublasHandle = nullptr;
cudaStream_t CuPCGLinSolver::m_cudaStream = nullptr;

// Use CudaUtils namespace for error checking and helpers
using namespace CudaUtils;
#endif // _CUDA

CuPCGLinSolver::CuPCGLinSolver(
    CudaGenBcsrLinSolver* preconditioner,
    int maxIterations, double relativeTolerance, double absoluteTolerance,
    int updateFrequency, bool updateOnFailure, bool verbose)
    :CudaGenBcsrLinSolver(SOLVER_TAGS_CuPCGLinSolver, CudaPrecision::dDDI),  // Default to double, updated in setLinearSOE
    m_verbose(verbose),
    m_maxIterations(maxIterations),
    m_relativeTolerance(relativeTolerance),
    m_absoluteTolerance(absoluteTolerance),
    m_updateFrequency(updateFrequency),
    m_updateOnFailure(updateOnFailure),
    m_lastIterationCount(0),
    m_numRefactorizations(0),
    m_solvesSinceUpdate(0),
    m_isFirstSolve(true),
    m_preconditioner(preconditioner)
{
    #ifdef _CUDA
    m_dBuffer = nullptr;
    m_bufferSize = 0;
    m_d_workspaceBlock = nullptr;
    m_d_x = nullptr;
    m_d_r = nullptr;
    m_d_z = nullptr;
    m_d_p = nullptr;
    m_d_Ap = nullptr;
    m_d_temp = nullptr;
    m_allocatedSize = 0;
    m_spMatDescr = nullptr;
    m_vecX = nullptr;
    m_vecY = nullptr;
    
    // Preconditioner is optional (nullptr = identity preconditioner)
    // Precision will be determined from SOE when attached via setLinearSOE()
    m_MatrixValueType = CUDA_R_64F; // Default, will be set properly when SOE is attached
    m_VectorValueType = CUDA_R_64F; // Default, will be set properly when SOE is attached
    
    init("");
    #endif // _CUDA
}

#ifdef _CUDA
void CuPCGLinSolver::init(const char* precision)
{
    if (!m_CuSparseInitialized) {
        /* Create a CUDA stream */
        cudaCheckError(cudaStreamCreate(&m_cudaStream), "create CUDA stream");

        /* Create the cuSPARSE handle */
        cuSparseCheckError(cusparseCreate(&m_cuSparseHandle), "create cuSPARSE handle");
        
        /* Create the cuBLAS handle */
        cublasCheckError(cublasCreate(&m_cublasHandle), "create cuBLAS handle");
        
        /* Set the CUDA stream */
        cuSparseCheckError(cusparseSetStream(m_cuSparseHandle, m_cudaStream), "set cuSPARSE stream");
        cublasCheckError(cublasSetStream(m_cublasHandle, m_cudaStream), "set cuBLAS stream");

        m_CuSparseInitialized = true;
    }

    m_ActiveSolverInstances++;
}
#endif // _CUDA

CuPCGLinSolver::~CuPCGLinSolver()
{
    #ifdef _CUDA
    cleanup();
    
    if (m_ActiveSolverInstances == 1) {
        cuSparseCheckError(cusparseDestroy(m_cuSparseHandle), "destroy cuSPARSE handle", false);
        cublasCheckError(cublasDestroy(m_cublasHandle), "destroy cuBLAS handle", false);
        cudaCheckError(cudaStreamSynchronize(m_cudaStream), "synchronize CUDA stream", false);
        cudaCheckError(cudaStreamDestroy(m_cudaStream), "destroy CUDA stream", false);
        m_CuSparseInitialized = false;
    }
    
    if (m_ActiveSolverInstances > 0) {
        m_ActiveSolverInstances--;
    }
    #endif // _CUDA
}

#ifdef _CUDA
void CuPCGLinSolver::cleanup()
{
    /* Destroy cuSPARSE objects */
    if (m_spMatDescr != nullptr) {
        cuSparseCheckError(cusparseDestroySpMat(m_spMatDescr), "destroy cuSPARSE matrix", false);
        m_spMatDescr = nullptr;
    }
    if (m_vecX != nullptr) {
        cuSparseCheckError(cusparseDestroyDnVec(m_vecX), "destroy cuSPARSE vector X", false);
        m_vecX = nullptr;
    }
    if (m_vecY != nullptr) {
        cuSparseCheckError(cusparseDestroyDnVec(m_vecY), "destroy cuSPARSE vector Y", false);
        m_vecY = nullptr;
    }
    if (m_dBuffer != nullptr) {
        cudaCheckError(cudaFree(m_dBuffer), "free cuSPARSE buffer", false);
        m_dBuffer = nullptr;
    }
    
    /* Free PCG workspace vectors (single block allocation) */
    if (m_d_workspaceBlock != nullptr) {
        cudaCheckError(cudaFree(m_d_workspaceBlock), "free PCG workspace block", false);
        m_d_workspaceBlock = nullptr;
        // Reset all vector pointers (they pointed into the workspace block)
        m_d_x = nullptr;
        m_d_r = nullptr;
        m_d_z = nullptr;
        m_d_p = nullptr;
        m_d_Ap = nullptr;
        m_d_temp = nullptr;
    }
}
#endif // _CUDA

int CuPCGLinSolver::setLinearSOE(CudaGenBcsrLinSOE &theSOE)
{
    #ifdef _CUDA
    // Support both scalar CSR (blockSize = 1) and BSR (blockSize > 1)
    int blockSize = theSOE.getBlockSize();
    if (blockSize < 1 || blockSize > 32) {
        opserr << "WARNING: CuPCGLinSolver::setLinearSOE() - "
               << "blockSize must be between 1 and 32, got " << blockSize << endln;
        return -1;
    }
    
    // Adopt precision from SOE
    m_precision = theSOE.getPrecision();
    
    // CuPCG has a specific limitation from cuSPARSE SpMV:
    // Supports: dDDI, dFFI, dFDI
    // Does NOT support: dDFI - double matrix with float vectors
    if (m_precision == CudaPrecision::dDFI) {
        opserr << "ERROR: CuPCGLinSolver::setLinearSOE() - "
               << "Precision dDFI (double matrix, float vectors) is not supported by cuSPARSE SpMV. "
               << "Supported modes: dDDI, dFFI, dFDI" << endln;
        return -1;
    }
    
    // Set CUDA data types for matrix and vectors separately
    if (m_precision == CudaPrecision::dDDI) {
        m_MatrixValueType = CUDA_R_64F;  // double matrix
        m_VectorValueType = CUDA_R_64F;  // double vectors
    } else if (m_precision == CudaPrecision::dFFI) {
        m_MatrixValueType = CUDA_R_32F;  // float matrix
        m_VectorValueType = CUDA_R_32F;  // float vectors
    } else if (m_precision == CudaPrecision::dFDI) {
        m_MatrixValueType = CUDA_R_32F;  // float matrix
        m_VectorValueType = CUDA_R_64F;  // double vectors
    }
    
    // Set for this solver
    int result = this->CudaGenBcsrLinSolver::setLinearSOE(theSOE);
    if (result != 0) return result;
    
    // Set for the preconditioner (if provided)
    if (m_preconditioner != nullptr) {
        return m_preconditioner->setLinearSOE(theSOE);
    }
    
    return 0;
    #endif // _CUDA

    return 0;
}

void CuPCGLinSolver::setPreconditioner(CudaGenBcsrLinSolver* preconditioner)
{
    #ifdef _CUDA
    // Transfer ownership of the new preconditioner
    m_preconditioner.reset(preconditioner);
    
    // Reset solve state since we have a new preconditioner
    m_isFirstSolve = true;
    m_solvesSinceUpdate = 0;
    
    // Attach the LinearSOE to the new preconditioner if we already have one
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE != nullptr && preconditioner != nullptr) {
        preconditioner->setLinearSOE(*theSOE);
    }
    
    // Note: Workspace will be reallocated in next setSize() call if needed
    // (number of vectors changes depending on whether preconditioner is present)
    
    if (m_verbose) {
        opserr << "INFO: CuPCGLinSolver::setPreconditioner() - "
               << (preconditioner != nullptr ? "New preconditioner set" : "Preconditioner removed")
               << endln;
    }
    #endif // _CUDA
}

int CuPCGLinSolver::setSize()
{
    #ifdef _CUDA
    // Update preconditioner size if needed
    if (m_preconditioner != nullptr && m_isFirstSolve) {
        int result = m_preconditioner->setSize();
        if (result != 0) return result;
    }
    
    // Allocate/reallocate PCG workspace if size changed
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "ERROR: CuPCGLinSolver::setSize() - LinearSOE not set" << endln;
        return -1;
    }
    
    int blockSize = theSOE->getBlockSize();
    int numBlockRows = theSOE->getNumRowBlocks();
    int numScalarRows = numBlockRows * blockSize;
    
    // Calculate required workspace size
    size_t vectorSize = numScalarRows * (m_VectorValueType == CUDA_R_64F ? sizeof(double) : sizeof(float));
    
    // Determine number of vectors needed
    // Always need: x, r, p, Ap (4 vectors - x is separate because preconditioner can modify the SOE)
    // With preconditioner: also need z, temp (6 vectors total)
    // Without preconditioner: only 4 vectors needed
    int numVectors = (m_preconditioner != nullptr) ? 6 : 4;
    size_t requiredSize = numVectors * vectorSize;
    
    // Only reallocate if we need more space than currently allocated
    if (requiredSize > m_allocatedSize) {
        // Free existing workspace if allocated
        if (m_d_workspaceBlock != nullptr) {
            cudaCheckError(cudaFree(m_d_workspaceBlock), "free old PCG workspace block", false);
            m_d_workspaceBlock = nullptr;
            m_d_x = nullptr;
            m_d_r = nullptr;
            m_d_z = nullptr;
            m_d_p = nullptr;
            m_d_Ap = nullptr;
            m_d_temp = nullptr;
        }
        
        // Allocate new workspace
        cudaCheckError(cudaMalloc(&m_d_workspaceBlock, requiredSize), "allocate PCG workspace block");
        m_allocatedSize = requiredSize;
        
        if (m_verbose) {
            double memoryMB = requiredSize / (1024.0 * 1024.0);
            opserr << "INFO: CuPCGLinSolver::setSize() - Allocated " << memoryMB << " MB for " 
                   << numVectors << " PCG workspace vectors (size=" << numScalarRows 
                   << ", " << (m_preconditioner != nullptr ? "preconditioned" : "unpreconditioned") 
                   << ")" << endln;
        }
    }
    
    // Set up pointers into the workspace block (always update, even if not reallocating)
    char* basePtr = static_cast<char*>(m_d_workspaceBlock);
    m_d_x = basePtr + 0 * vectorSize;
    m_d_r = basePtr + 1 * vectorSize;
    m_d_p = basePtr + 2 * vectorSize;
    m_d_Ap = basePtr + 3 * vectorSize;
    
    if (m_preconditioner != nullptr) {
        m_d_z = basePtr + 4 * vectorSize;
        m_d_temp = basePtr + 5 * vectorSize;
    } else {
        m_d_z = nullptr;
        m_d_temp = nullptr;
    }
    
    #endif
    return 0;
}

int CuPCGLinSolver::solve(void)
{
    #ifdef _CUDA
    if (m_isFirstSolve && m_preconditioner != nullptr) {
        // First solve with preconditioner: use it directly (performs factorization)
        if (m_verbose) {
            opserr << "INFO: CuPCGLinSolver - First solve using preconditioner directly" << endln;
        }
        int result = m_preconditioner->solve();
        if (result == 0) {
            m_isFirstSolve = false;
            m_lastIterationCount = 0;
            m_numRefactorizations++;
            m_solvesSinceUpdate = 0; // Reset counter after update
        }
        return result;
    } else {
        // No preconditioner, or subsequent solves: use PCG
        if (m_isFirstSolve && m_verbose) {
            opserr << "INFO: CuPCGLinSolver - No preconditioner, using unpreconditioned CG" << endln;
        }
        m_isFirstSolve = false;
        
        // Increment solve counter
        m_solvesSinceUpdate++;
        
        // Attempt PCG solve
        return solvePCG();
    }
    #endif // _CUDA

    return 0;
}

#ifdef _CUDA

// ============================================================================
// Custom CUDA kernels - combine operations to reduce kernel launches
// Grid-stride loop pattern: each thread can process multiple elements
// ============================================================================

// Kernel: x = x + alpha*p  AND  r = r - alpha*Ap (combined update)
template<typename T>
__global__ void updateXandR_kernel(
    T* __restrict__ x,
    T* __restrict__ r, 
    const T* __restrict__ p,
    const T* __restrict__ Ap,
    T alpha,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        x[i] += alpha * p[i];
        r[i] -= alpha * Ap[i];
    }
}

// Kernel: p = beta*p + vec (update search direction)
// vec can be z (preconditioned) or r (unpreconditioned)
template<typename T>
__global__ void updateP_kernel(
    T* __restrict__ p,
    const T* __restrict__ vec,
    T beta,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        p[i] = beta * p[i] + vec[i];
    }
}

// Kernel: copy vector (replaces cudaMemcpy for device-to-device)
template<typename T>
__global__ void copy_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

// Kernel: initialize x=0 and r=b (combined initialization)
template<typename T>
__global__ void initXandR_kernel(
    T* __restrict__ x,
    T* __restrict__ r,
    const T* __restrict__ b,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        x[i] = 0;
        r[i] = b[i];
    }
}

// Helpers to launch kernels
template<typename T>
void launchUpdateXandR(T* x, T* r, const T* p, const T* Ap, T alpha, int n, cudaStream_t stream)
{
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    updateXandR_kernel<<<numBlocks, blockSize, 0, stream>>>(x, r, p, Ap, alpha, n);
}

template<typename T>
void launchUpdateP(T* p, const T* vec, T beta, int n, cudaStream_t stream)
{
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    updateP_kernel<<<numBlocks, blockSize, 0, stream>>>(p, vec, beta, n);
}

template<typename T>
void launchCopy(T* dst, const T* src, int n, cudaStream_t stream)
{
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    copy_kernel<<<numBlocks, blockSize, 0, stream>>>(dst, src, n);
}

template<typename T>
void launchInitXandR(T* x, T* r, const T* b, int n, cudaStream_t stream)
{
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    initXandR_kernel<<<numBlocks, blockSize, 0, stream>>>(x, r, b, n);
}

int CuPCGLinSolver::solvePCG()
{
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "ERROR: CuPCGLinSolver::solvePCG() - LinearSOE not set" << endln;
        return -1;
    }

    int blockSize = theSOE->getBlockSize();
    int numBlockRows = theSOE->getNumRowBlocks();
    int numNonZeroBlocks = theSOE->getNumNonZeroBlocks();
    int* rowPtrs = theSOE->getDeviceRowPtrs();
    int* colIndices = theSOE->getDeviceColIndices();
    void* AValues = theSOE->getDeviceAValues();
    void* xValues = theSOE->getDeviceX();
    void* bValues = theSOE->getDeviceB();

    // Total number of scalar rows (for vector operations)
    int numScalarRows = numBlockRows * blockSize;
    
    // Workspace is allocated in setSize()
    // Verify allocation is correct
    if (m_d_workspaceBlock == nullptr) {
        opserr << "ERROR: CuPCGLinSolver::solvePCG() - Workspace not allocated. "
               << "This should have been allocated in setSize()." << endln;
        return -1;
    }

    // Setup cuSPARSE matrix descriptor if needed
    CudaGenBcsrLinSOE::MatrixStatus matrixStatus = theSOE->getMatrixStatus();
    
    if (m_spMatDescr == nullptr || matrixStatus == CudaGenBcsrLinSOE::MatrixStatus::STRUCTURE_CHANGED) {
        // Need to create/recreate matrix descriptor (structure changed or first time)
        if (m_spMatDescr != nullptr) {
            cuSparseCheckError(cusparseDestroySpMat(m_spMatDescr), "destroy old matrix descriptor");
            m_spMatDescr = nullptr;
        }
        
        cusparseIndexType_t indexType = CUSPARSE_INDEX_32I;
        
        if (blockSize == 1) {
            // Scalar CSR format
            cuSparseCheckError(cusparseCreateCsr(
                &m_spMatDescr, numBlockRows, numBlockRows, numNonZeroBlocks,
                rowPtrs, colIndices, AValues,
                indexType, indexType,
                CUSPARSE_INDEX_BASE_ZERO,
                m_MatrixValueType  // Matrix data type
            ), "create cuSPARSE CSR matrix");
        } else {
            // Block CSR (BSR) format
            cuSparseCheckError(cusparseCreateBsr(
                &m_spMatDescr,
                numBlockRows,              // brows
                numBlockRows,              // bcols
                numNonZeroBlocks,          // bnnz
                blockSize,                 // rowBlockSize
                blockSize,                 // colBlockSize
                rowPtrs,                   // bsrRowOffsets
                colIndices,                // bsrColInd
                AValues,                   // bsrValues
                indexType,                 // bsrRowOffsetsType
                indexType,                 // bsrColIndType
                CUSPARSE_INDEX_BASE_ZERO,  // idxBase
                m_MatrixValueType,         // valueType - Matrix data type
                CUSPARSE_ORDER_ROW         // order (row-major within blocks)
            ), "create cuSPARSE BSR matrix");
        }
    } else if (matrixStatus == CudaGenBcsrLinSOE::MatrixStatus::COEFFICIENTS_CHANGED) {
        // Only coefficients changed - update values (works for both CSR and BSR)
        cuSparseCheckError(cusparseSpMatSetValues(
            m_spMatDescr, AValues
        ), "update matrix values");
    }
    // else: UNCHANGED - no update needed

    // Setup dense vector descriptors
    if (m_vecX == nullptr) {
        cuSparseCheckError(cusparseCreateDnVec(&m_vecX, numScalarRows, m_d_p, m_VectorValueType), "create vecX");
        cuSparseCheckError(cusparseCreateDnVec(&m_vecY, numScalarRows, m_d_Ap, m_VectorValueType), "create vecY");
    }

    // Get SpMV buffer size, allocate, and preprocess if needed
    if (m_dBuffer == nullptr) {
        double alpha = 1.0, beta = 0.0;
        cuSparseCheckError(cusparseSpMV_bufferSize(
            m_cuSparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, m_spMatDescr, m_vecX, &beta, m_vecY,
            m_VectorValueType,  // Compute type (use vector precision for accumulation)
            CUSPARSE_SPMV_ALG_DEFAULT, &m_bufferSize
        ), "get SpMV buffer size");
        
        if (m_bufferSize > 0) {
            cudaCheckError(cudaMalloc(&m_dBuffer, m_bufferSize), "allocate SpMV buffer");
        }
        
        // Preprocess SpMV for better performance (done once per matrix structure)
        cuSparseCheckError(cusparseSpMV_preprocess(
            m_cuSparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, m_spMatDescr, m_vecX, &beta, m_vecY,
            m_VectorValueType,  // Compute type (use vector precision for accumulation)
            CUSPARSE_SPMV_ALG_DEFAULT, m_dBuffer
        ), "preprocess SpMV");
    }

    // Determine if we need to update the preconditioner
    bool updatePreconditioner = false;
    if (m_updateFrequency >= 1 && m_solvesSinceUpdate >= m_updateFrequency) {
        m_solvesSinceUpdate = 0;
        updatePreconditioner = true;
    }

    // Call the appropriate PCG implementation based on vector precision
    int result;
    if (m_VectorValueType == CUDA_R_64F) {
        result = solvePCG_impl<double>((double*)m_d_x, (double*)bValues, numScalarRows, updatePreconditioner);
    } else {
        result = solvePCG_impl<float>((float*)m_d_x, (float*)bValues, numScalarRows, updatePreconditioner);
    }
    
    if (result != 0 && m_preconditioner != nullptr && m_updateOnFailure && m_updateFrequency != 1) {
        if (m_verbose) {
            opserr << "INFO: CuPCGLinSolver - PCG failed, refactorizing preconditioner and retrying (updateOnFailure)" << endln;
        }
        m_solvesSinceUpdate = 0;
        updatePreconditioner = true;

        if (m_VectorValueType == CUDA_R_64F) {
            result = solvePCG_impl<double>((double*)m_d_x, (double*)bValues, numScalarRows, updatePreconditioner);
        } else {
            result = solvePCG_impl<float>((float*)m_d_x, (float*)bValues, numScalarRows, updatePreconditioner);
        }
    }

    // Copy solution from workspace to SOE
    if (result == 0) {
        size_t vectorSize = numScalarRows * (m_VectorValueType == CUDA_R_64F ? sizeof(double) : sizeof(float));
        cudaCheckError(cudaMemcpy(xValues, m_d_x, vectorSize, cudaMemcpyDeviceToDevice), "copy solution to SOE");
    }
    
    return result;
}

// Template helper for preconditioner application (used by virtual method)
template<typename T>
int CuPCGLinSolver::applyPreconditionerImpl(T* z, T* r, int n, bool updatePreconditioner)
{
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    
    if (m_preconditioner != nullptr) {
        T* temp = (T*)m_d_temp;
        // Apply preconditioner: save current RHS, set r as RHS, solve, restore
        cudaCheckError(cudaMemcpy(temp, theSOE->getDeviceB(), n * sizeof(T), cudaMemcpyDeviceToDevice), "save b");
        cudaCheckError(cudaMemcpy(theSOE->getDeviceB(), r, n * sizeof(T), cudaMemcpyDeviceToDevice), "set RHS to r");
        
        // Initialize z to zero for the preconditioner solve
        cudaCheckError(cudaMemset(z, 0, n * sizeof(T)), "zero z");
        cudaCheckError(cudaMemcpy(theSOE->getDeviceX(), z, n * sizeof(T), cudaMemcpyDeviceToDevice), "init precond solution");
        
        // Apply preconditioner: choose solve() or solveNoRefact() based on flag
        int precond_result;
        if (updatePreconditioner) {
            // Update preconditioner (calls setSize() + solve())
            precond_result = m_preconditioner->setSize();
            if (precond_result != 0) {
                opserr << "ERROR: CuPCGLinSolver - Preconditioner setSize failed" << endln;
                cudaCheckError(cudaMemcpy(theSOE->getDeviceB(), temp, n * sizeof(T), cudaMemcpyDeviceToDevice), "restore b");
                return -1;
            }
            m_numRefactorizations++; // Count each forced preconditioner update
            precond_result = m_preconditioner->solve();
        } else {
            // Reuse existing preconditioner setup
            precond_result = m_preconditioner->solveNoRefact();
        }
        
        if (precond_result != 0) {
            opserr << "ERROR: CuPCGLinSolver - Preconditioner application failed" << endln;
            cudaCheckError(cudaMemcpy(theSOE->getDeviceB(), temp, n * sizeof(T), cudaMemcpyDeviceToDevice), "restore b");
            return -1;
        }
        
        cudaCheckError(cudaMemcpy(z, theSOE->getDeviceX(), n * sizeof(T), cudaMemcpyDeviceToDevice), "get z from precond");
        cudaCheckError(cudaMemcpy(theSOE->getDeviceB(), temp, n * sizeof(T), cudaMemcpyDeviceToDevice), "restore b");
    } else {
        // No preconditioner: M = I, so z = r
        cudaCheckError(cudaMemcpy(z, r, n * sizeof(T), cudaMemcpyDeviceToDevice), "copy r to z (identity preconditioner)");
    }

    return 0;
}

// Virtual method that dispatches to template based on precision
int CuPCGLinSolver::applyPreconditioner(void* z, void* r, int n, bool updatePreconditioner)
{
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "ERROR: CuPCGLinSolver::applyPreconditioner() - LinearSOE not set" << endln;
        return -1;
    }
    
    // Dispatch to typed implementation based on vector precision
    CudaPrecision precision = theSOE->getPrecision();
    if (precision == CudaPrecision::dDDI || precision == CudaPrecision::dFDI) {
        // Vector type is double
        return applyPreconditionerImpl<double>((double*)z, (double*)r, n, updatePreconditioner);
    } else {
        // Vector type is float
        return applyPreconditionerImpl<float>((float*)z, (float*)r, n, updatePreconditioner);
    }
}

// PCG implementation templated on vector type T
template<typename T>
int CuPCGLinSolver::solvePCG_impl(T* x, T* b, int n, bool updatePreconditioner)
{
    T* r = (T*)m_d_r;
    T* z = (T*)m_d_z;
    T* p = (T*)m_d_p;
    T* Ap = (T*)m_d_Ap;

    // Get SOE for preconditioner application
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();

    // Check if we're using an unpreconditioned solver (identity preconditioner)
    const bool isUnpreconditioned = (m_preconditioner == nullptr);

    // Initial calcs (iter = 0)
    int iter = 0;

    // Compute norm of RHS for relative tolerance
    T normB = 0.0;
    cublasCheckError(cublasNrm2(m_cublasHandle, n, b, 1, &normB), "norm of b");
    
    // Early exit for zero RHS
    if (normB == 0.0) {
        cudaCheckError(cudaMemset(x, 0, n * sizeof(T)), "zero solution for zero RHS");
        m_lastIterationCount = 0;
        if (m_verbose) {
            opserr << "INFO: CuPCGLinSolver - Zero RHS, returning zero solution" << endln;
        }
        return 0;
    }
    
    // Initialize x=0 and r=b
    launchInitXandR(x, r, b, n, m_cudaStream);
    
    // Compute dynamic tolerance: tol = max(relTol * ||b||, absTol)
    T tol = std::max(static_cast<T>(m_relativeTolerance * normB), static_cast<T>(m_absoluteTolerance));
    
    if (m_verbose) {
        opserr << "  PCG tolerance = max(" << m_relativeTolerance << " * " << normB 
               << ", " << m_absoluteTolerance << ") = " << tol << endln;
    }

    T rho = 0.0, rho_old = 0.0;
    T residualNorm = 0.0;
    
    // Initialize: for unpreconditioned CG, z = r, so p = r and rho = ||r||^2
    if (isUnpreconditioned) {
        // p = r (no need for z)
        launchCopy(p, r, n, m_cudaStream);
        
        // Compute initial residual norm
        cublasCheckError(cublasNrm2(m_cublasHandle, n, r, 1, &residualNorm), "norm of r");
        
        // Check cuda errors from initializing x, r, and p
        cudaCheckError(cudaGetLastError(), "init kernels");
        
        // For unpreconditioned: rho = r^T * r = ||r||^2
        rho_old = residualNorm * residualNorm;
    } else {
        // z = M^{-1} * r (apply preconditioner)
        int precond_result = applyPreconditioner((void*)z, (void*)r, n, updatePreconditioner);
        if (precond_result != 0) {
            opserr << "ERROR: CuPCGLinSolver - Preconditioner application failed" << endln;
            return -1;
        }
        
        // p = z
        launchCopy(p, z, n, m_cudaStream);
        
        // rho_old = r^T * z
        cublasCheckError(cublasDot(m_cublasHandle, n, r, 1, z, 1, &rho_old), "dot for rho_old");
        
        // Check cuda errors from initializing x, r, z, p, and rho_old
        cudaCheckError(cudaGetLastError(), "init kernels");
        
        // Compute initial residual norm
        cublasCheckError(cublasNrm2(m_cublasHandle, n, r, 1, &residualNorm), "norm of r");
    }
    
    // Check if already converged (initial guess was good)
    if (residualNorm < tol) {
        m_lastIterationCount = 0;
        if (m_verbose) {
            opserr << "INFO: CuPCGLinSolver - Already converged (initial residual = " 
                   << residualNorm << ")" << endln;
        }
        return 0;
    }
    
    const T one = 1.0, zero = 0.0;
    
    for (iter = 1; iter <= m_maxIterations; iter++) {
        // Ap = A * p
        cusparseDnVecSetValues(m_vecX, p);
        cusparseDnVecSetValues(m_vecY, Ap);
        
        cuSparseCheckError(cusparseSpMV(
            m_cuSparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, m_spMatDescr, m_vecX, &zero, m_vecY,
            m_VectorValueType,  // Compute type (use vector precision for accumulation)
            CUSPARSE_SPMV_ALG_DEFAULT, m_dBuffer
        ), "SpMV in PCG");

        // pAp = p^T * Ap
        T pAp = 0.0;
        cublasCheckError(cublasDot(m_cublasHandle, n, p, 1, Ap, 1, &pAp), "dot for pAp");
        
        // alpha = rho_old / pAp
        T alpha = rho_old / pAp;

        // Combined update: x = x + alpha*p AND r = r - alpha*Ap (single kernel)
        launchUpdateXandR(x, r, p, Ap, alpha, n, m_cudaStream);

        // Check convergence: ||r|| < tol (syncs - check errors here)
        cublasCheckError(cublasNrm2(m_cublasHandle, n, r, 1, &residualNorm), "norm of r");
        cudaCheckError(cudaGetLastError(), "PCG iteration kernels");
        
        if (m_verbose && (iter % 10 == 0 || iter <= 5)) {
            opserr << "  PCG iteration " << iter << ", residual = " << residualNorm << " (tol = " << tol << ")" << endln;
        }

        if (residualNorm < tol) {
            m_lastIterationCount = iter;
            if (m_verbose) {
                opserr << "  PCG iteration " << iter << ", residual = " << residualNorm << " (tol = " << tol << ")" << endln;
                opserr << "INFO: CuPCGLinSolver - PCG converged in " << m_lastIterationCount 
                       << " iterations, residual = " << residualNorm << endln;
            }
            
            // Solution will be copied back to SOE in solvePCG()
            return 0;
        }

        // Update search direction
        if (isUnpreconditioned) {
            // Optimized unpreconditioned path: z = r, so rho = r^T * r = ||r||^2
            // We already computed ||r|| for convergence check, so reuse it!
            rho = residualNorm * residualNorm;
            
            // beta = rho / rho_old
            T beta = rho / rho_old;
            
            // p = beta*p + r (pass r directly, no z vector needed)
            launchUpdateP(p, r, beta, n, m_cudaStream);
            
            rho_old = rho;
        } else {
            // Preconditioned path: z = M^{-1} * r
            int precond_result = applyPreconditioner((void*)z, (void*)r, n, false);
            if (precond_result != 0) {
                opserr << "ERROR: CuPCGLinSolver - Preconditioner application failed" << endln;
                return -1;
            }

            // rho = r^T * z (syncs - check errors here)
            cublasCheckError(cublasDot(m_cublasHandle, n, r, 1, z, 1, &rho), "dot for rho");
            cudaCheckError(cudaGetLastError(), "update p kernel");
            
            // beta = rho / rho_old
            T beta = rho / rho_old;

            // p = beta*p + z (pass z from preconditioner)
            launchUpdateP(p, z, beta, n, m_cudaStream);
            
            rho_old = rho;
        }
    }

    // Loop exited without convergence - we completed all maxIterations
    m_lastIterationCount = m_maxIterations;
    
    // Final error check
    cudaCheckError(cudaStreamSynchronize(m_cudaStream), "final sync");
    cudaCheckError(cudaGetLastError(), "PCG iteration kernels");
    
    opserr << "WARNING: CuPCGLinSolver - PCG did not converge in " << m_maxIterations << " iterations" << endln;
    return -1;
}

#endif // _CUDA

// OpenSees API for creating solver
struct CuPCGConfig {
    std::string preconditioner = "None";  // Preconditioner type (default: unpreconditioned CG)
    std::string precision = "dDDI";
    int maxIterations = 100;
    double relativeTolerance = 1e-6;
    double absoluteTolerance = 1e-12;
    int updateFrequency = 1;      // Update preconditioner every N solves (1 = always, default)
    bool updateOnFailure = true;  // Update when PCG fails
    bool verbose = false;
};

class CuPCGParameterParser {
private:
    static std::unordered_map<std::string, std::function<void(CuPCGConfig&)>> const configParsers;

public:
    static bool parseParameters(CuPCGConfig& config);
    static void printUsageInfo();
};

const std::unordered_map<std::string, std::function<void(CuPCGConfig&)>> 
CuPCGParameterParser::configParsers = {
    {"precision", [](CuPCGConfig& config) { 
        const char* value = OPS_GetString();
        if (value && (strcmp(value, "dDDI") == 0 || strcmp(value, "dFFI") == 0)) {
            config.precision = value;
        } else if (value) {
            throw std::invalid_argument("Invalid precision. Only dDDI and dFFI are supported.");
        }
    }},
    {"maxIter", [](CuPCGConfig& config) { 
        int numData = 1;
        int val = 0;
        if (OPS_GetIntInput(&numData, &val) == 0) {
            if (val <= 0) throw std::invalid_argument("maxIter must be positive");
            config.maxIterations = val;
        }
    }},
    {"relTol", [](CuPCGConfig& config) { 
        int numData = 1;
        double val = 0.0;
        if (OPS_GetDoubleInput(&numData, &val) == 0) {
            if (val <= 0.0) throw std::invalid_argument("relTol must be positive");
            config.relativeTolerance = val;
        }
    }},
    {"absTol", [](CuPCGConfig& config) { 
        int numData = 1;
        double val = 0.0;
        if (OPS_GetDoubleInput(&numData, &val) == 0) {
            if (val <= 0.0) throw std::invalid_argument("absTol must be positive");
            config.absoluteTolerance = val;
        }
    }},
    {"updateFrequency", [](CuPCGConfig& config) { 
        int numData = 1;
        int val = 0;
        if (OPS_GetIntInput(&numData, &val) == 0) {
            if (val < 0) throw std::invalid_argument("updateFrequency cannot be negative");
            config.updateFrequency = val;
        }
    }},
    {"updateOnFailure", [](CuPCGConfig& config) { 
        int numData = 1;
        int val = 0;
        if (OPS_GetIntInput(&numData, &val) == 0) {
            if (val != 0 && val != 1) throw std::invalid_argument("updateOnFailure must be 0 or 1");
            config.updateOnFailure = (val == 1);
        }
    }},
    {"verbose", [](CuPCGConfig& config) { 
        int numData = 1;
        int flag = 0;
        if (OPS_GetIntInput(&numData, &flag) == 0) {
            if (flag != 0 && flag != 1) throw std::invalid_argument("verbose must be 0 or 1");
            config.verbose = (flag == 1);
        }
    }}
};

bool CuPCGParameterParser::parseParameters(CuPCGConfig& config) {
    try {
        while (OPS_GetNumRemainingInputArgs() > 0) {
            const char* key = OPS_GetString();
            if (!key) {
                opserr << "WARNING: CuPCGParameterParser::parseParameters() - "
                       << "Invalid input argument" << endln;
                return false;
            }
            
            // Special handling for -preconditioner: it terminates CuPCG parsing
            // and all remaining args are for the preconditioner
            if (strcmp(key, "-preconditioner") == 0 || strcmp(key, "preconditioner") == 0) {
                // Read preconditioner type
                const char* precondType = OPS_GetString();
                if (precondType) {
                    config.preconditioner = precondType;
                }
                // Stop parsing - remaining arguments are for the preconditioner
                return true;
            }
            
            auto it = ParameterUtils::findParameter(key, configParsers);
            if (it != configParsers.end()) {
                it->second(config);
            } else {
                continue;
            }
        }
        return true;
    } catch (const std::exception& e) {
        opserr << "WARNING: CuPCGParameterParser::parseParameters() - "
               << e.what() << endln;
        return false;
    }
}

void CuPCGParameterParser::printUsageInfo() {
    opserr << "CuPCGParameterParser::printUsageInfo() - " << endln;
    opserr << "Usage: system CuPCG [options] -preconditioner <type> [preconditioner-options]" << endln;
    opserr << "CuPCG Options:" << endln;
    opserr << "  -precision <dDDI|dFFI>          Precision mode (default: dDDI)" << endln;
    opserr << "  -maxIter <int>                  Max PCG iterations (default: 100)" << endln;
    opserr << "  -relTol <double>                Relative convergence tolerance (default: 1e-6)" << endln;
    opserr << "  -absTol <double>                Absolute convergence tolerance (default: 1e-12)" << endln;
    opserr << "                                  Convergence: ||r|| < max(relTol*||b||, absTol)" << endln;
    opserr << "  -updateFrequency <N>            Update preconditioner every N solves (default: 1)" << endln;
    opserr << "                                  0 = never update (after first solve)" << endln;
    opserr << "                                  1 = always update (every solve, most robust)" << endln;
    opserr << "                                  N > 1 = periodic update (every N solves)" << endln;
    opserr << "  -updateOnFailure <0|1>          Update when PCG fails (default: 1)" << endln;
    opserr << "                                  Provides adaptive failsafe regardless of frequency" << endln;
    opserr << "  -verbose <0|1>                  Enable verbose output (default: 0)" << endln;
    opserr << "" << endln;
    opserr << "Preconditioner Types:" << endln;
    opserr << "  -preconditioner CuDSS [options] Direct solver preconditioner (uses cuDSS)" << endln;
    opserr << "  -preconditioner AmgX [options]  Iterative solver preconditioner (uses AmgX)" << endln;
    opserr << "  -preconditioner None            No preconditioner (unpreconditioned CG)" << endln;
    opserr << "" << endln;
    opserr << "Update Strategy Examples:" << endln;
    opserr << "  (1, *) = ALWAYS:               Update every solve (default, most robust)" << endln;
    opserr << "  (0, 1) = NEVER + FAILSAFE:     Efficient, refactorizes only on failure" << endln;
    opserr << "  (0, 0) = NEVER:                One factorization, always PCG (may fail if matrix changes)" << endln;
    opserr << "  (5, 0) = PERIODIC:             Update every 5 solves (strict schedule)" << endln;
    opserr << "  (5, 1) = PERIODIC + FAILSAFE:  Update every 5 solves, or on failure (adaptive)" << endln;
    opserr << "" << endln;
    opserr << "Examples:" << endln;
    opserr << "  # Default: always update (most robust)" << endln;
    opserr << "  system CuPCG -maxIter 200 -relTol 1e-8 -preconditioner CuDSS -verbose 1" << endln;
    opserr << "" << endln;
    opserr << "  # Efficient: never update, but refactorize if PCG fails" << endln;
    opserr << "  system CuPCG -updateFrequency 0 -updateOnFailure 1 -preconditioner CuDSS" << endln;
    opserr << "" << endln;
    opserr << "  # Never update (strict - may fail if matrix changes)" << endln;
    opserr << "  system CuPCG -updateFrequency 0 -updateOnFailure 0 -preconditioner CuDSS" << endln;
    opserr << "" << endln;
    opserr << "  # Always update (maximum robustness, expensive)" << endln;
    opserr << "  system CuPCG -updateFrequency 1 -preconditioner AmgX -solver PCG" << endln;
    opserr << "" << endln;
    opserr << "  # Update every 10 solves with failsafe" << endln;
    opserr << "  system CuPCG -updateFrequency 10 -updateOnFailure 1 -preconditioner CuDSS" << endln;
    opserr << "" << endln;
    opserr << "  # Unpreconditioned CG" << endln;
    opserr << "  system CuPCG -maxIter 500 -preconditioner None" << endln;
    opserr << "" << endln;
    opserr << "Description:" << endln;
    opserr << "  General PCG solver with any CudaGenBcsrLinSolver as preconditioner." << endln;
    opserr << "  Supports both scalar CSR (blockSize=1) and BSR (blockSize>1) formats." << endln;
    opserr << "  First solve uses preconditioner directly, subsequent solves use PCG." << endln;
    opserr << "  All arguments after '-preconditioner <type>' are passed to that solver's parser." << endln;
}

void* OPS_CuPCGLinSolver()
{
    #ifndef _CUDA
    opserr << "WARNING: OPS_CuPCGLinSolver() - "
           << "CUDA not available" << endln;
    return nullptr;
    #else
    
    CuPCGConfig config;
    
    // Parse CuPCG-specific parameters (stops at -preconditioner)
    if (!CuPCGParameterParser::parseParameters(config)) {
        opserr << "WARNING: OPS_CuPCGLinSolver() - "
               << "Failed to parse parameters, using defaults" << endln;
        CuPCGParameterParser::printUsageInfo();
    }
    
    // Create the preconditioner based on config.preconditioner
    // Remaining OPS arguments are passed to the preconditioner factory
    CudaGenBcsrLinSolver* precond = nullptr;
    
    if (config.preconditioner == "CuDSS" || config.preconditioner == "cuDSS" || 
        config.preconditioner == "CUDSS" || config.preconditioner == "cudss") {
        #ifdef _CUDSS
        precond = createCuDSSSolverFromParser();
        if (precond == nullptr) {
            opserr << "ERROR: OPS_CuPCGLinSolver() - "
                   << "Failed to create CuDSS preconditioner" << endln;
            return nullptr;
        }
        #else
        opserr << "WARNING: OPS_CuPCGLinSolver() - "
               << "cuDSS not available, falling back to unpreconditioned CG" << endln;
        precond = nullptr;
        config.preconditioner = "None";
        #endif
    } else if (config.preconditioner == "AmgX" || config.preconditioner == "amgx" || 
               config.preconditioner == "AMGX" || config.preconditioner == "Amgx") {
        #ifdef _AMGX
        precond = createAmgXSolverFromParser();
        if (precond == nullptr) {
            opserr << "ERROR: OPS_CuPCGLinSolver() - "
                   << "Failed to create AmgX preconditioner" << endln;
            return nullptr;
        }
        #else
        opserr << "WARNING: OPS_CuPCGLinSolver() - "
               << "AmgX not available, falling back to unpreconditioned CG" << endln;
        precond = nullptr;
        config.preconditioner = "None";
        #endif
    } else if (config.preconditioner == "None" || config.preconditioner == "none" || 
               config.preconditioner == "NULL" || config.preconditioner == "null") {
        // No preconditioner (identity preconditioner)
        precond = nullptr;
    } else {
        opserr << "ERROR: OPS_CuPCGLinSolver() - "
               << "Unknown preconditioner type: " << config.preconditioner.c_str() << endln;
        opserr << "Currently supported: CuDSS, AmgX, None" << endln;
        return nullptr;
    }
    
    // Create the PCG solver with the preconditioner
    CuPCGLinSolver* solver = new CuPCGLinSolver(
        precond,
        config.maxIterations,
        config.relativeTolerance,
        config.absoluteTolerance,
        config.updateFrequency,
        config.updateOnFailure,
        config.verbose
    );
    
    // Parse precision from config string
    CudaPrecision precision;
    if (!cudaPrecisionFromString(config.precision.c_str(), precision)) {
        opserr << "WARNING: OPS_CuPCGLinSolver() - "
               << "Invalid precision '" << config.precision.c_str() << "', defaulting to dDDI" << endln;
        precision = CudaPrecision::dDDI;
    }
    
    // CuPCG supports: dDDI, dFFI, dFDI (via cuSPARSE SpMV)
    // Does NOT support: dDFI (double matrix, float vectors - cuSPARSE limitation)
    if (precision == CudaPrecision::dDFI) {
        opserr << "ERROR: OPS_CuPCGLinSolver() - "
               << "Precision dDFI (double matrix, float vectors) is not supported by cuSPARSE SpMV. "
               << "Supported modes: dDDI, dFFI, dFDI" << endln;
        delete solver;
        return nullptr;
    }
    
    // CuPCG supports both scalar CSR (blockSize = 1) and BSR (blockSize > 1)
    // Note, however, certain preconditioners only support blockSize = 1
    const int blockSize = 1;
    const bool paddingEnabled = false;
    
    // Create SOE based on precision mode
    switch(precision) {
        case CudaPrecision::dDDI:
            return CudaGenBcsrLinSOE::createDouble(*solver, blockSize, paddingEnabled, config.verbose);
        case CudaPrecision::dFFI:
            return CudaGenBcsrLinSOE::createFloat(*solver, blockSize, paddingEnabled, config.verbose);
        case CudaPrecision::dFDI:
            return CudaGenBcsrLinSOE::createFloatDouble(*solver, blockSize, paddingEnabled, config.verbose);
        default:
            // Should never reach here due to validation above
            opserr << "ERROR: OPS_CuPCGLinSolver() - Unexpected precision mode" << endln;
            delete solver;
            return nullptr;
    }
    
    #endif // _CUDA
}

