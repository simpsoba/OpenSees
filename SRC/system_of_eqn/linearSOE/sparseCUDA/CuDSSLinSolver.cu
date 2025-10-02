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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCUDA/CuDSSLinSolver.cpp,v $
                                                                        
                                                                        
// Written: gaaraujo 
// Created: 10/2025
//
// Description: This file contains the class definition for 
// CuDSSLinSolver. It solves the CudaGenBcsrLinSOE object by calling
// CuDSS routines.
//

// OpenSees includes
#include <classTags.h>
#include <OPS_Globals.h>

// Solve core classes
#include <CudaGenBcsrLinSOE.h>
#include <CuDSSLinSolver.h>

// for parsing command line arguments
#ifdef _CUDSS
#include <elementAPI.h>
#include <FileStream.h>
#include <unordered_map>
#endif // _CUDSS

// C++ includes
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

#ifdef _CUDSS
// Static member initialization
bool CuDSSLinSolver::m_CuDSSInitialized = false;
int CuDSSLinSolver::m_ActiveSolverInstances = 0;
#endif // _CUDSS

/* Anonymous namespace for helper functions */
namespace {
    #ifdef _CUDSS
    void cudaCheckError(cudaError_t error, const char* message, bool throwError = true)
    {
        if (error != cudaSuccess) {
            char* errorString = cudaGetErrorString(error);
            if (throwError) {
                throw std::runtime_error(
                "CUDA API returned error " + 
                std::string(errorString) + 
                " for " + std::string(message)
                );
            } else {
                opserr << "CUDA API returned error " << errorString << " for " << message << endln;
            }
        }
    }
    void cuDSSCheckError(cudssStatus_t error, const char* message, bool throwError = true)
    {
        if (error != CUDSS_STATUS_SUCCESS) {
            if (throwError) {
                throw std::runtime_error(
                "cuDSS API returned error " + 
                std::to_string(error) + 
                " for " + std::string(message)
                );
            } else {
                opserr << "cuDSS API returned error " << error << " for " << message << endln;
            }
        }
    }
    #endif // _CUDSS
}
CuDSSLinSolver::CuDSSLinSolver(std::string dataType, bool verbose)
    :CudaGenBcsrLinSolver(SOLVER_TAGS_CuDSSLinSolver), 
    m_verbose(verbose)
{
    #ifdef _CUDSS
    _init(dataType);
    #endif // _CUDSS
}

void CuDSSLinSolver::_init(std::string dataType)
{
    #ifdef _CUDSS
    if (!m_CuDSSInitialized) {
        
        /* Create a CUDA stream */
        cudaCheckError(cudaStreamCreate(&m_cudaStream), "create CUDA stream");

        /* Create the cuDSS handle */
        cuDSSCheckError(cudssCreateHandle(&m_Handle), "create cuDSS handle");
        
        /* Set the CUDA stream */
        cuDSSCheckError(cudssSetStream(m_Handle, m_cudaStream), "set CUDA stream");

        /* Initialize cuDSS */
        m_CuDSSInitialized = true;
    }

    /* Create cuDSS solver configuration and data objects */
    cuDSSCheckError(cudssConfigCreate(&m_Config), "create cuDSS solver configuration");
    cuDSSCheckError(cudssDataCreate(m_Handle, &m_Data), "create cuDSS solver data");
    
    // (optional) Modifying solver settings, e.g., reordering algorithm
    cudssAlgType_t reorderAlgorithm = CUDSS_ALG_DEFAULT;
    cudssConfigSet(m_Config, CUDSS_REORDERING_ALG, &reorderAlgorithm, sizeof(cudssAlgType_t));

    m_Matrix = nullptr;
    m_RHS = nullptr;
    m_Solution = nullptr;
    
    m_IndexType = CUDA_R_32I;
    if (dataType == "float") {
        m_ValueType = CUDA_R_32F;
    } else {
        m_ValueType = CUDA_R_64F;
    }

    /* Increment counter */
    m_ActiveSolverInstances++;
    #endif // _CUDSS

    return;
}

CuDSSLinSolver::~CuDSSLinSolver()
{
    #ifdef _CUDSS
    /* Destroy opaque objects, matrix wrappers and the cuDSS handle */
    if (m_Matrix != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_Matrix), "destroy cuDSS matrix", false);
    }
    if (m_RHS != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_RHS), "destroy cuDSS right-hand side", false);
    }
    if (m_Solution != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_Solution), "destroy cuDSS solution", false);
    }
    cuDSSCheckError(cudssDataDestroy(m_Handle, m_Data), "destroy cuDSS solver configuration", false);
    cuDSSCheckError(cudssConfigDestroy(m_Config), "destroy cuDSS solver data", false);
    
    if (m_ActiveSolverInstances == 1) {
        cuDSSCheckError(cudssDestroy(m_Handle), "destroy cuDSS handle", false);
        cudaCheckError(cudaStreamSynchronize(m_cudaStream), "synchronize CUDA stream", false);
        m_CuDSSInitialized = false;
    }
    /* Decrement counter */
    if (m_ActiveSolverInstances > 0) {
        m_ActiveSolverInstances--;
    }
    #endif // _CUDSS

    return;
}

int CuDSSLinSolver::setLinearSOE(CudaGenBcsrLinSOE &theSOE) {
    #ifdef _CUDSS
    bool bothDouble = theSOE.isDoublePrecision() && m_DataType == CUDA_R_64F;
    bool bothFloat = !theSOE.isDoublePrecision() && m_DataType == CUDA_R_32F;
    if (bothDouble || bothFloat) {
        return this->CudaGenBcsrLinSolver::setLinearSOE(theSOE);
    } else {
        opserr << "WARNING: CuDSSLinSolver::setLinearSOE() - "
                << "precision mismatch between LinearSOE and CuDSSLinSolver" << endln;
        return -1;
    }
    #endif // _CUDSS

    return 0;
}

int CuDSSLinSolver::solve(void) {
    #ifdef _CUDSS
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "WARNING: CuDSSLinSolver::solve() - "
               << "LinearSOE not set" << endln;
        return -1;
    }

    // Extract info from the SOE
    CudaGenBcsrLinSOE::MatrixStatus matrixStatus = theSOE->getMatrixStatus();
    int numRows = theSOE->getNumRowBlocks();
    int numCols = numRows;
    int numNZ = theSOE->getNumNonZeroValues();
    int blockSize = theSOE->getBlockSize();
    int* rowPtrs = theSOE->getDeviceRowPtrs();
    int* colIndices = theSOE->getDeviceColIndices();

    // Check that system is not scalar CSR (block CSR not supported)
    if (blockSize != 1) {
        opserr << "ERROR: CuDSSLinSolver::solve() - "
               << "Only blockSize = 1 is supported" << endln;
        return -1;
    }
    
    // Check if device pointers are valid
    if (!rowPtrs) {
        opserr << "ERROR: CuDSSLinSolver::solve() - getDeviceRowPtrs() returned nullptr" << endln;
        return -1;
    }
    if (!colIndices) {
        opserr << "ERROR: CuDSSLinSolver::solve() - getDeviceColIndices() returned nullptr" << endln;
        return -1;
    }
    void* values = theSOE->getDeviceAValues();
    void* x = theSOE->getDeviceX();
    void* b = theSOE->getDeviceB();

    if (!values) {
        opserr << "ERROR: CuDSSLinSolver::solve() - getDeviceAValues() returned nullptr" << endln;
        return -1;
    }
    if (!x) {
        opserr << "ERROR: CuDSSLinSolver::solve() - getDeviceX() returned nullptr" << endln;
        return -1;
    }
    if (!b) {
        opserr << "ERROR: CuDSSLinSolver::solve() - getDeviceB() returned nullptr" << endln;
        return -1;
    }

    // Upload the matrix data to the GPU and factorize if necessary
    if (matrixStatus == CudaGenBcsrLinSOE::MatrixStatus::STRUCTURE_CHANGED) {
        /* Create the cuDSS matrix */
        cudssMatrixType_t matrixType = CUDSS_MTYPE_GENERAL;
        cudssMatrixViewType_t matrixView = CUDSS_MVIEW_FULL;
        cudssIndexBase_t indexBase = CUDSS_BASE_ZERO;
        cuDSSCheckError(cudssMatrixCreateCsr(
            m_Handle, numRows, numCols, numNZ, 
            rowPtrs, nullptr, colIndices, values, 
            m_IndexType, m_ValueType, 
            matrixType, matrixView, indexBase
        ), "create cuDSS matrix");
        
        /* Symbolic factorization */
        cuDSSCheckError(cudssExecute(
            m_Handle, CUDSS_PHASE_ANALYSIS, m_Config, m_Data,
            m_Matrix, m_RHS, m_Solution
        ), "cuDSS symbolic factorization");
        
        /* Numerical factorization */
        cuDSSCheckError(cudssExecute(
            m_Handle, CUDSS_PHASE_FACTORIZATION, m_Config, m_Data,
            m_Matrix, m_RHS, m_Solution
        ), "cuDSS numerical factorization");
    } else if (matrixStatus == CudaGenBcsrLinSOE::MatrixStatus::COEFFICIENTS_CHANGED) {
        /* Update coefficients */
        cuDSSCheckError(cudssMatrixSetValues(m_Matrix, values), "set cuDSS matrix values");

        /* Numerical factorization */
        cuDSSCheckError(cudssExecute(
            m_Handle, CUDSS_PHASE_REFACTORIZATION, m_Config, m_Data,
            m_Matrix, m_RHS, m_Solution
        ), "cuDSS numerical factorization");
    } else { /* pass */ }

    // Upload the rhs and solution data to the GPU
    cuDSSCheckError(cudssMatrixCreateDn(
        m_RHS, (int64_t)numRows, 1, (int64_t)numRows, b, m_ValueType, CUDSS_LAYOUT_COL_MAJOR
    ), "create cuDSS right-hand side");
    cuDSSCheckError(cudssMatrixCreateDn(
        m_Solution, (int64_t)numRows, 1, (int64_t)numRows, x, m_ValueType, CUDSS_LAYOUT_COL_MAJOR
    ), "create cuDSS solution");

    // Solve the system of equations
    cudssStatus_t status;
    status = cudssExecute(
        m_Handle, CUDSS_PHASE_SOLVE, m_Config, m_Data,
        m_Matrix, m_RHS, m_Solution
    );
    
    if (status != CUDSS_STATUS_SUCCESS) {
        opserr << "WARNING: CuDSSLinSolver::solve() - "
               << "cuDSS solve failed with status " << status << endln;
        return -1;
    }

    if (m_verbose) {
        opserr << "INFO: CuDSSLinSolver::solve() - "
               << "cuDSS solve successful" << endln;
    }

    #endif // _CUDSS

    return 0;
}

int CuDSSLinSolver::setSize() {
    return 0;
}

// OpenSees API for parsing command line arguments
#ifdef _CUDSS

struct CuDSSConfig {
    std::string dataType = "double";
    bool verbose = false;
    int blockSize = 1;
};

CudaGenBcsrLinSolver* createCuDSSSolver(const CuDSSConfig& config) {
    return new CuDSSLinSolver(
        config.dataType, config.verbose
    );
}

void* OPS_CuDSSLinSolver()
{
    // Handle case with no arguments - use default configuration
    if (OPS_GetNumRemainingInputArgs() == 0) {
        CuDSSConfig defaultConfig; // Use default values
        auto solver = createCuDSSSolver(defaultConfig);
        return CudaGenBcsrLinSOE::createDouble(
            *solver, defaultConfig.blockSize, 
            false, defaultConfig.verbose
        );
    }

    // If we get here, parsing failed
    opserr << "WARNING: OPS_CuDSSLinSolver() - "
           << "Failed to parse CuDSSLinSolver parameters" << endln;
    return nullptr;
}
#else // _CUDSS
void* OPS_CuDSSLinSolver() {
    opserr << "WARNING: OPS_CuDSSLinSolver() - "
           << "cuDSS not available" << endln;
    return nullptr;
}
#endif // _CUDSS

