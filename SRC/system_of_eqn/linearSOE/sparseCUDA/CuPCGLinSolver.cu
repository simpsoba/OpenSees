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
    bool refactorOnNonConvergence, bool verbose)
    :CudaGenBcsrLinSolver(SOLVER_TAGS_CuPCGLinSolver), 
    m_verbose(verbose),
    m_maxIterations(maxIterations),
    m_relativeTolerance(relativeTolerance),
    m_absoluteTolerance(absoluteTolerance),
    m_refactorOnNonConvergence(refactorOnNonConvergence),
    m_lastIterationCount(0),
    m_numRefactorizations(0),
    m_isFirstSolve(true),
    m_preconditioner(preconditioner)
{
    #ifdef _CUDA
    m_dBuffer = nullptr;
    m_bufferSize = 0;
    m_d_x = nullptr;
    m_d_r = nullptr;
    m_d_z = nullptr;
    m_d_p = nullptr;
    m_d_Ap = nullptr;
    m_d_temp = nullptr;
    m_spMatDescr = nullptr;
    m_vecX = nullptr;
    m_vecY = nullptr;
    
    // Preconditioner is optional (nullptr = identity preconditioner)
    // Determine precision from preconditioner (assume it matches)
    // We'll validate this in setLinearSOE
    m_ValueType = CUDA_R_64F; // Default, will be set properly when SOE is attached
    
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
    
    /* Free PCG workspace vectors */
    if (m_d_x != nullptr) {
        cudaCheckError(cudaFree(m_d_x), "free PCG x vector", false);
        m_d_x = nullptr;
    }
    if (m_d_r != nullptr) {
        cudaCheckError(cudaFree(m_d_r), "free PCG r vector", false);
        m_d_r = nullptr;
    }
    if (m_d_z != nullptr) {
        cudaCheckError(cudaFree(m_d_z), "free PCG z vector", false);
        m_d_z = nullptr;
    }
    if (m_d_p != nullptr) {
        cudaCheckError(cudaFree(m_d_p), "free PCG p vector", false);
        m_d_p = nullptr;
    }
    if (m_d_Ap != nullptr) {
        cudaCheckError(cudaFree(m_d_Ap), "free PCG Ap vector", false);
        m_d_Ap = nullptr;
    }
    if (m_d_temp != nullptr) {
        cudaCheckError(cudaFree(m_d_temp), "free PCG temp vector", false);
        m_d_temp = nullptr;
    }
}
#endif // _CUDA

int CuPCGLinSolver::setLinearSOE(CudaGenBcsrLinSOE &theSOE)
{
    #ifdef _CUDA
    // Only support scalar CSR (blockSize = 1)
    if (theSOE.getBlockSize() != 1) {
        opserr << "WARNING: CuPCGLinSolver::setLinearSOE() - "
               << "Only blockSize = 1 is supported" << endln;
        return -1;
    }
    
    // Determine precision from SOE
    m_ValueType = theSOE.isDoublePrecision() ? CUDA_R_64F : CUDA_R_32F;
    
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

int CuPCGLinSolver::setSize()
{
    #ifdef _CUDA
    // Let preconditioner handle size setting (if provided)
    if (m_preconditioner != nullptr) {
        return m_preconditioner->setSize();
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
        }
        return result;
    } else {
        // No preconditioner, or subsequent solves: use PCG
        if (m_isFirstSolve && m_verbose) {
            opserr << "INFO: CuPCGLinSolver - No preconditioner, using unpreconditioned CG" << endln;
        }
        m_isFirstSolve = false;
        return solvePCG();
    }
    #endif // _CUDA

    return 0;
}

#ifdef _CUDA

int CuPCGLinSolver::solvePCG()
{
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "ERROR: CuPCGLinSolver::solvePCG() - LinearSOE not set" << endln;
        return -1;
    }

    int numRows = theSOE->getNumRowBlocks();
    int numNonZero = theSOE->getNumNonZeroBlocks();
    int* rowPtrs = theSOE->getDeviceRowPtrs();
    int* colIndices = theSOE->getDeviceColIndices();
    void* AValues = theSOE->getDeviceAValues();
    void* xValues = theSOE->getDeviceX();
    void* bValues = theSOE->getDeviceB();

    // Allocate PCG workspace vectors if needed
    size_t vectorSize = numRows * (m_ValueType == CUDA_R_64F ? sizeof(double) : sizeof(float));
    if (m_d_r == nullptr) {
        cudaCheckError(cudaMalloc(&m_d_x, vectorSize), "allocate PCG x vector");
        cudaCheckError(cudaMalloc(&m_d_r, vectorSize), "allocate PCG r vector");
        cudaCheckError(cudaMalloc(&m_d_z, vectorSize), "allocate PCG z vector");
        cudaCheckError(cudaMalloc(&m_d_p, vectorSize), "allocate PCG p vector");
        cudaCheckError(cudaMalloc(&m_d_Ap, vectorSize), "allocate PCG Ap vector");
        cudaCheckError(cudaMalloc(&m_d_temp, vectorSize), "allocate PCG temp vector");
    }

    // Setup cuSPARSE matrix descriptor if needed
    if (m_spMatDescr == nullptr) {
        cusparseIndexType_t indexType = CUSPARSE_INDEX_32I;
        
        cuSparseCheckError(cusparseCreateCsr(
            &m_spMatDescr, numRows, numRows, numNonZero,
            rowPtrs, colIndices, AValues,
            indexType, indexType,
            CUSPARSE_INDEX_BASE_ZERO,
            m_ValueType
        ), "create cuSPARSE CSR matrix");
    } else {
        // Update matrix values
        cuSparseCheckError(cusparseCsrSetPointers(
            m_spMatDescr, rowPtrs, colIndices, AValues
        ), "update cuSPARSE CSR pointers");
    }

    // Setup dense vector descriptors
    if (m_vecX == nullptr) {
        cuSparseCheckError(cusparseCreateDnVec(&m_vecX, numRows, m_d_p, m_ValueType), "create vecX");
        cuSparseCheckError(cusparseCreateDnVec(&m_vecY, numRows, m_d_Ap, m_ValueType), "create vecY");
    }

    // Get SpMV buffer size, allocate, and preprocess if needed
    if (m_dBuffer == nullptr) {
        double alpha = 1.0, beta = 0.0;
        cuSparseCheckError(cusparseSpMV_bufferSize(
            m_cuSparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, m_spMatDescr, m_vecX, &beta, m_vecY,
            m_ValueType, CUSPARSE_SPMV_ALG_DEFAULT, &m_bufferSize
        ), "get SpMV buffer size");
        
        if (m_bufferSize > 0) {
            cudaCheckError(cudaMalloc(&m_dBuffer, m_bufferSize), "allocate SpMV buffer");
        }
        
        // Preprocess SpMV for better performance (done once per matrix structure)
        cuSparseCheckError(cusparseSpMV_preprocess(
            m_cuSparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, m_spMatDescr, m_vecX, &beta, m_vecY,
            m_ValueType, CUSPARSE_SPMV_ALG_DEFAULT, m_dBuffer
        ), "preprocess SpMV");
    }

    // Call the appropriate PCG implementation based on precision
    int result;
    if (m_ValueType == CUDA_R_64F) {
        result = solvePCG_impl<double>((double*)m_d_x, (double*)bValues, numRows);
    } else {
        result = solvePCG_impl<float>((float*)m_d_x, (float*)bValues, numRows);
    }
    
    // Copy solution from workspace to SOE
    if (result == 0) {
        cudaCheckError(cudaMemcpy(xValues, m_d_x, vectorSize, cudaMemcpyDeviceToDevice), "copy solution to SOE");
    }
    
    return result;
}

// Helper function to apply preconditioner
template<typename T>
int CuPCGLinSolver::applyPreconditioner(T* z, T* r, int n)
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
        
        // Apply preconditioner (use existing factorization)
        int precond_result = m_preconditioner->solveNoRefact();
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

template<typename T>
int CuPCGLinSolver::solvePCG_impl(T* x, T* b, int n)
{
    T* r = (T*)m_d_r;
    T* z = (T*)m_d_z;
    T* p = (T*)m_d_p;
    T* Ap = (T*)m_d_Ap;

    // Get SOE for preconditioner application
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();

    // Initial calcs (iter = 0)
    int iter = 0;

    // Start with zero initial guess
    cudaCheckError(cudaMemset(x, 0, n * sizeof(T)), "zero initial guess");

    // r = b - A*x = b (since x = 0, no need to compute A*x)
    cudaCheckError(cudaMemcpy(r, b, n * sizeof(T), cudaMemcpyDeviceToDevice), "copy b to r");

    // z = M^{-1} * r (apply preconditioner)
    int precond_result = applyPreconditioner(z, r, n);
    if (precond_result != 0) {
        opserr << "ERROR: CuPCGLinSolver - Preconditioner application failed" << endln;
        return -1;
    }
    
    // p = z
    cudaCheckError(cudaMemcpy(p, z, n * sizeof(T), cudaMemcpyDeviceToDevice), "copy z to p");

    // Compute norm of RHS for relative tolerance
    T normB = 0.0;
    cublasCheckError(cublasNrm2(m_cublasHandle, n, b, 1, &normB), "norm of b");
    
    // Compute dynamic tolerance: tol = max(relTol * ||b||, absTol)
    T tol = std::max(m_relativeTolerance * normB, m_absoluteTolerance);
    
    if (m_verbose) {
        opserr << "  PCG tolerance = max(" << m_relativeTolerance << " * " << normB 
               << ", " << m_absoluteTolerance << ") = " << tol << endln;
    }

    T rho = 0.0, rho_old = 0.0;
    
    for (iter = 1; iter <= m_maxIterations; iter++) {
        // rho_old = r^T * z
        cublasCheckError(cublasDot(m_cublasHandle, n, r, 1, z, 1, &rho_old), "dot for rho_old");
        
        // Ap = A * p
        cusparseDnVecSetValues(m_vecX, p);
        cusparseDnVecSetValues(m_vecY, Ap);
        
        const T one = 1.0, zero = 0.0;
        cuSparseCheckError(cusparseSpMV(
            m_cuSparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, m_spMatDescr, m_vecX, &zero, m_vecY,
            m_ValueType, CUSPARSE_SPMV_ALG_DEFAULT, m_dBuffer
        ), "SpMV in PCG");

        // pAp = p^T * Ap
        T pAp = 0.0;
        cublasCheckError(cublasDot(m_cublasHandle, n, p, 1, Ap, 1, &pAp), "dot for pAp");
        
        // alpha = rho_old / pAp
        T alpha = rho_old / pAp;

        // x = x + alpha * p
        cublasCheckError(cublasAxpy(m_cublasHandle, n, &alpha, p, 1, x, 1), "update x");

        // r = r - alpha * Ap
        T neg_alpha = -alpha;
        cublasCheckError(cublasAxpy(m_cublasHandle, n, &neg_alpha, Ap, 1, r, 1), "update r");

        // Check convergence: ||r|| < tol
        T residualNorm = 0.0;
        cublasCheckError(cublasNrm2(m_cublasHandle, n, r, 1, &residualNorm), "norm of r");
        
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

        // z = M^{-1} * r (apply preconditioner)
        precond_result = applyPreconditioner(z, r, n);
        if (precond_result != 0) {
            opserr << "ERROR: CuPCGLinSolver - Preconditioner application failed" << endln;
            return -1;
        }

        // rho = r^T * z
        cublasCheckError(cublasDot(m_cublasHandle, n, r, 1, z, 1, &rho), "dot for rho");
        
        // beta = rho / rho_old
        T beta = rho / rho_old;

        // p = z + beta * p (need to do in two steps)
        // Step 1: p = beta * p
        cublasCheckError(cublasScal(m_cublasHandle, n, &beta, p, 1), "scale p by beta");
        // Step 2: p = z + p
        cublasCheckError(cublasAxpy(m_cublasHandle, n, &one, z, 1, p, 1), "add z to p");
    }

    // Loop exited without convergence - we completed all maxIterations
    m_lastIterationCount = m_maxIterations;
    
    // PCG did not converge - refactorize if we have a preconditioner and refactorOnNonConvergence is enabled
    if (m_preconditioner != nullptr && m_refactorOnNonConvergence) {
        if (m_verbose) {
            opserr << "INFO: CuPCGLinSolver - PCG did not converge in " << m_maxIterations 
                   << " iterations, refactorizing and solving directly" << endln;
        }
        
        // Trigger refactorization by solving with preconditioner
        // (b is still in theSOE->getDeviceB(), solution will go to theSOE->getDeviceX())
        int result = m_preconditioner->solve();
        m_numRefactorizations++;
        return result;
    }

    opserr << "WARNING: CuPCGLinSolver - PCG did not converge in " << m_maxIterations << " iterations" << endln;
    return -1;
}

#endif // _CUDA

// OpenSees API for creating solver
struct CuPCGConfig {
    std::string preconditioner = "CuDSS";  // Preconditioner type
    std::string precision = "dDDI";
    int maxIterations = 100;
    double relativeTolerance = 1e-6;
    double absoluteTolerance = 1e-12;
    bool refactorOnNonConvergence = true;
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
    {"preconditioner", [](CuPCGConfig& config) { 
        const char* value = OPS_GetString();
        if (value) config.preconditioner = value;
    }},
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
    {"refactorOnNonConvergence", [](CuPCGConfig& config) { 
        int numData = 1;
        int val = 0;
        if (OPS_GetIntInput(&numData, &val) == 0) {
            if (val != 0 && val != 1) throw std::invalid_argument("refactorOnNonConvergence must be 0 or 1");
            config.refactorOnNonConvergence = (val == 1);
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
    opserr << "Usage: system CuPCG [options]" << endln;
    opserr << "Options:" << endln;
    opserr << "  -preconditioner <CuDSS|None>    Preconditioner type (default: CuDSS)" << endln;
    opserr << "  -precision <dDDI|dFFI>          Precision mode (default: dDDI)" << endln;
    opserr << "  -maxIter <int>                  Max PCG iterations (default: 100)" << endln;
    opserr << "  -relTol <double>                Relative convergence tolerance (default: 1e-6)" << endln;
    opserr << "  -absTol <double>                Absolute convergence tolerance (default: 1e-12)" << endln;
    opserr << "                                  Convergence: ||r|| < max(relTol*||b||, absTol)" << endln;
    opserr << "  -refactorOnNonConvergence <0|1> Refactorize if PCG fails to converge (default: 1)" << endln;
    opserr << "                                  (only applies when preconditioner != None)" << endln;
    opserr << "  -verbose <0|1>                  Enable verbose output (default: 0)" << endln;
    opserr << "Description:" << endln;
    opserr << "  General PCG solver with optional preconditioning. With preconditioner," << endln;
    opserr << "  first solve uses it directly, subsequent solves use PCG. Without" << endln;
    opserr << "  preconditioner (None), always uses unpreconditioned CG." << endln;
}

void* OPS_CuPCGLinSolver()
{
    #ifndef _CUDA
    opserr << "WARNING: OPS_CuPCGLinSolver() - "
           << "CUDA not available" << endln;
    return nullptr;
    #else
    
    CuPCGConfig config;
    
    if (!CuPCGParameterParser::parseParameters(config)) {
        opserr << "WARNING: OPS_CuPCGLinSolver() - "
               << "Failed to parse parameters, using defaults" << endln;
        CuPCGParameterParser::printUsageInfo();
    }
    
    // Create the preconditioner based on config
    CudaGenBcsrLinSolver* precond = nullptr;
    if (config.preconditioner == "CuDSS" || config.preconditioner == "cuDSS" || 
        config.preconditioner == "CUDSS" || config.preconditioner == "cudss") {
        #ifdef _CUDSS
        precond = new CuDSSLinSolver(config.precision.c_str(), config.verbose);
        #else
        opserr << "WARNING: OPS_CuPCGLinSolver() - "
               << "cuDSS not available, falling back to unpreconditioned CG" << endln;
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
        opserr << "Currently supported: CuDSS, None" << endln;
        return nullptr;
    }
    
    // Create the PCG solver with the preconditioner
    CuPCGLinSolver* solver = new CuPCGLinSolver(
        precond,
        config.maxIterations,
        config.relativeTolerance,
        config.absoluteTolerance,
        config.refactorOnNonConvergence,
        config.verbose
    );
    
    // CuPCG only supports scalar CSR (blockSize = 1, no padding)
    const int blockSize = 1;
    const bool paddingEnabled = false;
    
    if (config.precision == "dFFI") {
        return CudaGenBcsrLinSOE::createFloat(*solver, blockSize, paddingEnabled, config.verbose);
    } else {
        return CudaGenBcsrLinSOE::createDouble(*solver, blockSize, paddingEnabled, config.verbose);
    }
    
    #endif // _CUDA
}

