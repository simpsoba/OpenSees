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
#include <cstring>
#include <cmath>

#ifdef _CUDSS
// Static member initialization
bool CuDSSLinSolver::m_CuDSSInitialized = false;
int CuDSSLinSolver::m_ActiveSolverInstances = 0;
cudssHandle_t CuDSSLinSolver::m_Handle = nullptr;
cudaStream_t CuDSSLinSolver::m_cudaStream = nullptr;
#endif // _CUDSS

/* Anonymous namespace for helper functions */
namespace {
    #ifdef _CUDSS
    void cudaCheckError(cudaError_t error, const char* message, bool throwError = true)
    {
        if (error != cudaSuccess) {
            const char* errorString = cudaGetErrorString(error);
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
    init(dataType);
    #endif // _CUDSS
}

void CuDSSLinSolver::init(std::string dataType)
{
    #ifdef _CUDSS
    if (!m_CuDSSInitialized) {
        
        /* Create a CUDA stream */
        cudaCheckError(cudaStreamCreate(&m_cudaStream), "create CUDA stream");

        /* Create the cuDSS handle */
        cuDSSCheckError(cudssCreate(&m_Handle), "create cuDSS handle");
        
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
    cudssConfigSet(m_Config, CUDSS_CONFIG_REORDERING_ALG, &reorderAlgorithm, sizeof(cudssAlgType_t));

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
    bool bothDouble = theSOE.isDoublePrecision() && m_ValueType == CUDA_R_64F;
    bool bothFloat = !theSOE.isDoublePrecision() && m_ValueType == CUDA_R_32F;
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
    void* AValues = theSOE->getDeviceAValues();
    void* xValues = theSOE->getDeviceX();
    void* bValues = theSOE->getDeviceB();

    // Check if device pointers are valid
    if (!AValues) {
        opserr << "ERROR: CuDSSLinSolver::solve() - getDeviceAValues() returned nullptr" << endln;
        return -1;
    }
    if (!xValues) {
        opserr << "ERROR: CuDSSLinSolver::solve() - getDeviceX() returned nullptr" << endln;
        return -1;
    }
    if (!bValues) {
        opserr << "ERROR: CuDSSLinSolver::solve() - getDeviceB() returned nullptr" << endln;
        return -1;
    }

    // Setup matrices if structure has changed (setupMatrices checks internally)
    if (setupMatrices() != 0) {
        opserr << "ERROR: CuDSSLinSolver::solve() - setupMatrices() failed" << endln;
        return -1;
    }

    // Handle matrix updates and factorization
    if (matrixStatus == CudaGenBcsrLinSOE::MatrixStatus::STRUCTURE_CHANGED) {
        /* Update the RHS and solution pointers */
        cuDSSCheckError(cudssMatrixSetValues(m_RHS, bValues), "update cuDSS RHS values");
        cuDSSCheckError(cudssMatrixSetValues(m_Solution, xValues), "update cuDSS solution values");

        /* Numerical factorization (first time) */
        cuDSSCheckError(cudssExecute(
            m_Handle, CUDSS_PHASE_FACTORIZATION, m_Config, m_Data,
            m_Matrix, m_Solution, m_RHS
        ), "cuDSS numerical factorization");
    } else if (matrixStatus == CudaGenBcsrLinSOE::MatrixStatus::COEFFICIENTS_CHANGED) {
        /* Update coefficients */
        cuDSSCheckError(cudssMatrixSetValues(m_Matrix, AValues), "set cuDSS matrix values");
        
        /* Update the RHS and solution pointers */
        cuDSSCheckError(cudssMatrixSetValues(m_RHS, bValues), "update cuDSS RHS values");
        cuDSSCheckError(cudssMatrixSetValues(m_Solution, xValues), "update cuDSS solution values");

        /* Numerical factorization (refactorize with updated coefficients) */
        cuDSSCheckError(cudssExecute(
            m_Handle, CUDSS_PHASE_REFACTORIZATION, m_Config, m_Data,
            m_Matrix, m_Solution, m_RHS
        ), "cuDSS numerical factorization");
    } else {
        /* No changes, just update RHS and solution pointers */
        cuDSSCheckError(cudssMatrixSetValues(m_RHS, bValues), "update cuDSS RHS values");
        cuDSSCheckError(cudssMatrixSetValues(m_Solution, xValues), "update cuDSS solution values");
    }

    // Solve the system of equations
    cudssStatus_t status;
    status = cudssExecute(
        m_Handle, CUDSS_PHASE_SOLVE, m_Config, m_Data,
        m_Matrix, m_Solution, m_RHS
    );
    
    if (status != CUDSS_STATUS_SUCCESS) {
        opserr << "WARNING: CuDSSLinSolver::solve() - "
               << "cuDSS solve failed with status " << status << endln;
        return -1;
    }

    // Synchronize the CUDA stream to ensure solve is complete before returning
    cudaCheckError(cudaStreamSynchronize(m_cudaStream), "synchronize CUDA stream after solve");

    if (m_verbose) {
        opserr << "INFO: CuDSSLinSolver::solve() - "
               << "cuDSS solve successful" << endln;
    }

    #endif // _CUDSS

    return 0;
}

int CuDSSLinSolver::setSize() {
    // In OpenSees, setSize() is called before data is ready on the GPU
    // Matrix initialization is done in solve() via setupMatrices()
    return 0;
}

int CuDSSLinSolver::setupMatrices() {
    #ifdef _CUDSS
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "WARNING: CuDSSLinSolver::setupMatrices() - "
               << "LinearSOE not set" << endln;
        return -1;
    }

    // Check matrix status - only setup if structure has changed
    CudaGenBcsrLinSOE::MatrixStatus matrixStatus = theSOE->getMatrixStatus();
    if (matrixStatus != CudaGenBcsrLinSOE::MatrixStatus::STRUCTURE_CHANGED) {
        // Matrices already set up, nothing to do
        return 0;
    }

    // Extract info from the SOE
    int numRows = theSOE->getNumRowBlocks();
    int numCols = numRows;
    int numNZ = theSOE->getNumNonZeroValues();
    int blockSize = theSOE->getBlockSize();
    int* rowPtrs = theSOE->getDeviceRowPtrs();
    int* colIndices = theSOE->getDeviceColIndices();

    // Check that system is scalar CSR (block CSR not supported)
    if (blockSize != 1) {
        opserr << "ERROR: CuDSSLinSolver::setupMatrices() - "
               << "Only blockSize = 1 is supported" << endln;
        return -1;
    }
    
    // Check if device pointers are valid
    if (!rowPtrs) {
        opserr << "ERROR: CuDSSLinSolver::setupMatrices() - getDeviceRowPtrs() returned nullptr" << endln;
        return -1;
    }
    if (!colIndices) {
        opserr << "ERROR: CuDSSLinSolver::setupMatrices() - getDeviceColIndices() returned nullptr" << endln;
        return -1;
    }
    void* AValues = theSOE->getDeviceAValues();
    if (!AValues) {
        opserr << "ERROR: CuDSSLinSolver::setupMatrices() - getDeviceAValues() returned nullptr" << endln;
        return -1;
    }

    // Get x and b pointers for dense matrix creation
    void* xValues = theSOE->getDeviceX();
    void* bValues = theSOE->getDeviceB();
    if (!xValues) {
        opserr << "ERROR: CuDSSLinSolver::setupMatrices() - getDeviceX() returned nullptr" << endln;
        return -1;
    }
    if (!bValues) {
        opserr << "ERROR: CuDSSLinSolver::setupMatrices() - getDeviceB() returned nullptr" << endln;
        return -1;
    }

    /* Destroy existing matrices if they exist */
    if (m_Matrix != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_Matrix), "destroy existing cuDSS matrix");
        m_Matrix = nullptr;
    }
    if (m_RHS != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_RHS), "destroy existing cuDSS RHS");
        m_RHS = nullptr;
    }
    if (m_Solution != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_Solution), "destroy existing cuDSS solution");
        m_Solution = nullptr;
    }
    
    /* Create the cuDSS CSR matrix */
    cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
    cudssIndexBase_t ibase = CUDSS_BASE_ZERO;
    cuDSSCheckError(cudssMatrixCreateCsr(
        &m_Matrix, numRows, numCols, numNZ, 
        rowPtrs, nullptr, colIndices, AValues, 
        m_IndexType, m_ValueType, 
        mtype, mview, ibase
    ), "create cuDSS matrix");
    
    /* Create the cuDSS dense matrices for RHS and solution */
    cuDSSCheckError(cudssMatrixCreateDn(
        &m_RHS, (int64_t)numRows, 1, (int64_t)numRows, bValues, m_ValueType, CUDSS_LAYOUT_COL_MAJOR
    ), "create cuDSS right-hand side");
    cuDSSCheckError(cudssMatrixCreateDn(
        &m_Solution, (int64_t)numRows, 1, (int64_t)numRows, xValues, m_ValueType, CUDSS_LAYOUT_COL_MAJOR
    ), "create cuDSS solution");
    
    /* Symbolic factorization */
    cuDSSCheckError(cudssExecute(
        m_Handle, CUDSS_PHASE_ANALYSIS, m_Config, m_Data,
        m_Matrix, m_Solution, m_RHS
    ), "cuDSS symbolic factorization");

    if (m_verbose) {
        opserr << "INFO: CuDSSLinSolver::setupMatrices() - "
               << "All matrices created and symbolic factorization complete" << endln;
    }
    #endif // _CUDSS

    return 0;
}

// OpenSees API for creating CuDSS solver
#ifdef _CUDSS
void* OPS_CuDSSLinSolver()
{
    // Use default configuration for now
    std::string dataType = "double";
    bool verbose = false;
    int blockSize = 1;
    
    // Create solver with default settings
    CuDSSLinSolver* solver = new CuDSSLinSolver(dataType, verbose);
    
    // Create and return SOE
    return CudaGenBcsrLinSOE::createDouble(*solver, blockSize, false, verbose);
}
#else // _CUDSS
void* OPS_CuDSSLinSolver() {
    opserr << "WARNING: OPS_CuDSSLinSolver() - "
           << "cuDSS not available" << endln;
    return nullptr;
}
#endif // _CUDSS

