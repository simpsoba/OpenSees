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

// CUDA utilities
#include "CudaUtils.h"

// for parsing command line arguments
#ifdef _CUDSS
#include <elementAPI.h>
#include <FileStream.h>
#include <unordered_map>
#include "ParameterUtils.h"
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

// Use CudaUtils namespace for error checking
using namespace CudaUtils;
#endif // _CUDSS

CuDSSLinSolver::CuDSSLinSolver(const char* precision, bool verbose, 
                               bool hybridMemoryMode, size_t hybridDeviceMemoryLimit, 
                               bool hybridExecuteMode, bool multiThreadingMode,
                               const char* threadingLibPath)
    :CudaGenBcsrLinSolver(SOLVER_TAGS_CuDSSLinSolver), 
    m_verbose(verbose),
    m_hybridMemoryMode(hybridMemoryMode),
    m_hybridDeviceMemoryLimit(hybridDeviceMemoryLimit),
    m_hybridExecuteMode(hybridExecuteMode),
    m_multiThreadingMode(multiThreadingMode),
    m_threadingLibPath(threadingLibPath ? threadingLibPath : "")
{
    #ifdef _CUDSS
    init(precision);
    #endif // _CUDSS
}

void CuDSSLinSolver::init(const char* precision)
{
    #ifdef _CUDSS
    if (!m_CuDSSInitialized) {
        
        /* Create a CUDA stream */
        cudaCheckError(cudaStreamCreate(&m_cudaStream), "create CUDA stream");

        /* Create the cuDSS handle */
        cuDSSCheckError(cudssCreate(&m_Handle), "create cuDSS handle");
        
        /* Set the CUDA stream */
        cuDSSCheckError(cudssSetStream(m_Handle, m_cudaStream), "set CUDA stream");

        /* Setup OpenMP multi-threading */
        #ifdef CUDSS_USE_OPENMP
        if (m_multiThreadingMode) {
            // Determine threading library path
            // If "NULL" -> pass NULL to cudssSetThreadingLayer (let cuDSS use CUDSS_THREADING_LIB env var)
            // Otherwise use provided path (default is set in config struct)
            const char* threadingLib = (m_threadingLibPath == "NULL") ? nullptr : m_threadingLibPath.c_str();
            
            // Set the threading layer in cuDSS
            cudssStatus_t status = cudssSetThreadingLayer(m_Handle, threadingLib);
            if (status != CUDSS_STATUS_SUCCESS) {
                opserr << "WARNING: CuDSSLinSolver::init() - "
                       << "cudssSetThreadingLayer failed with status " << status << endln;
                opserr << "Continuing without multi-threading support" << endln;
            } else if (m_verbose) {
                opserr << "INFO: CuDSSLinSolver::init() - "
                       << "OpenMP multi-threading mode enabled" << endln;
                if (threadingLib) {
                    opserr << "Threading library: " << threadingLib << endln;
                } else {
                    opserr << "Threading library: NULL ";
                    opserr << "(cuDSS will use the threading layer library name ";
                    opserr << "from the environment variable 'CUDSS_THREADING_LIB')" << endln;
                }
            }
        }
        #else
        if (m_multiThreadingMode && m_verbose) {
            opserr << "INFO: CuDSSLinSolver::init() - "
                   << "OpenMP multi-threading support not available (OpenMP not found at build time)" << endln;
        }
        #endif // CUDSS_USE_OPENMP

        /* Initialize cuDSS */
        m_CuDSSInitialized = true;
    }

    /* Create cuDSS solver configuration and data objects */
    cuDSSCheckError(cudssConfigCreate(&m_Config), "create cuDSS solver configuration");
    cuDSSCheckError(cudssDataCreate(m_Handle, &m_Data), "create cuDSS solver data");
    
    // (optional) Modifying solver settings, e.g., reordering algorithm
    cudssAlgType_t reorderAlgorithm = CUDSS_ALG_DEFAULT;
    cudssConfigSet(m_Config, CUDSS_CONFIG_REORDERING_ALG, &reorderAlgorithm, sizeof(cudssAlgType_t));
    
    /* Configure hybrid modes (must be set before analysis phase) */
    if (m_hybridMemoryMode) {
        int hybridMemoryModeEnabled = 1;
        cuDSSCheckError(cudssConfigSet(m_Config, CUDSS_CONFIG_HYBRID_MODE, &hybridMemoryModeEnabled, sizeof(int)), 
                       "enable hybrid memory mode");
        
        // Optionally set device memory limit
        if (m_hybridDeviceMemoryLimit > 0) {
            cuDSSCheckError(cudssConfigSet(m_Config, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT, 
                                          &m_hybridDeviceMemoryLimit, sizeof(size_t)), 
                           "set hybrid device memory limit");
        }
        
        if (m_verbose) {
            opserr << "INFO: CuDSSLinSolver::init() - "
                   << "Hybrid memory mode enabled";
            if (m_hybridDeviceMemoryLimit > 0) {
                opserr << " with device memory limit = " << m_hybridDeviceMemoryLimit << " bytes";
            }
            opserr << endln;
        }
    }
    
    if (m_hybridExecuteMode) {
        int hybridExecuteModeEnabled = 1;
        cuDSSCheckError(cudssConfigSet(m_Config, CUDSS_CONFIG_HYBRID_EXECUTE_MODE, &hybridExecuteModeEnabled, sizeof(int)), 
                       "enable hybrid execute mode");
        
        if (m_verbose) {
            opserr << "INFO: CuDSSLinSolver::init() - "
                   << "Hybrid execute mode enabled" << endln;
        }
    }

    m_Matrix = nullptr;
    m_RHS = nullptr;
    m_Solution = nullptr;
    
    /* Parse precision string (format: dXYI where X=matrix type, Y=vector type, I=index type)
     * d = device (always required for GPU)
     * X,Y = D (double) or F (float)
     * I = I (int32) - currently only int32 is supported
     */
    m_IndexType = CUDA_R_32I; // Currently only 32-bit integers supported
    
    if (strcmp(precision, "dFFI") == 0) {
        m_ValueType = CUDA_R_32F; // Float precision
    } else if (strcmp(precision, "dDDI") == 0) {
        m_ValueType = CUDA_R_64F; // Double precision
    } else {
        opserr << "WARNING: CuDSSLinSolver::init() - "
               << "Invalid precision '" << precision << "'. Only dDDI and dFFI are supported. "
               << "Setting precision to dDDI (double)" << endln;
        m_ValueType = CUDA_R_64F; // Default to double precision
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

int CuDSSLinSolver::solveNoRefact(void) {
    #ifdef _CUDSS
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "WARNING: CuDSSLinSolver::solveNoRefact() - "
               << "LinearSOE not set" << endln;
        return -1;
    }
    
    void* xValues = theSOE->getDeviceX();
    void* bValues = theSOE->getDeviceB();

    // Check if device pointers are valid
    if (!xValues) {
        opserr << "ERROR: CuDSSLinSolver::solveNoRefact() - getDeviceX() returned nullptr" << endln;
        return -1;
    }
    if (!bValues) {
        opserr << "ERROR: CuDSSLinSolver::solveNoRefact() - getDeviceB() returned nullptr" << endln;
        return -1;
    }
    
    // Ensure matrices are set up
    if (m_Matrix == nullptr || m_RHS == nullptr || m_Solution == nullptr) {
        opserr << "ERROR: CuDSSLinSolver::solveNoRefact() - "
               << "Matrices not initialized. Call solve() first to perform factorization." << endln;
        return -1;
    }

    /* Update only RHS and solution pointers (no factorization) */
    cuDSSCheckError(cudssMatrixSetValues(m_RHS, bValues), "update cuDSS RHS values");
    cuDSSCheckError(cudssMatrixSetValues(m_Solution, xValues), "update cuDSS solution values");

    // Solve using existing factorization
    cudssStatus_t status = cudssExecute(
        m_Handle, CUDSS_PHASE_SOLVE, m_Config, m_Data,
        m_Matrix, m_Solution, m_RHS
    );
    
    if (status != CUDSS_STATUS_SUCCESS) {
        opserr << "WARNING: CuDSSLinSolver::solveNoRefact() - "
               << "cuDSS solve failed with status " << status << endln;
        return -1;
    }

    // Synchronize the CUDA stream
    cudaCheckError(cudaStreamSynchronize(m_cudaStream), "synchronize CUDA stream after solve");

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

struct CuDSSConfig {
    std::string precision = "dDDI";
    bool verbose = false;
    bool hybridMemoryMode = false;        // Hybrid host/device memory mode
    size_t hybridDeviceMemoryLimit = 0;   // Device memory limit for hybrid memory mode (0 = use internal heuristic)
    bool hybridExecuteMode = false;       // Hybrid host/device execute mode
    bool multiThreadingMode = false;      // OpenMP multi-threading mode (requires OpenMP at build time)
    std::string threadingLibPath = "/usr/lib/x86_64-linux-gnu/libcudss_mtlayer_gomp.so";  // Threading layer library path ("NULL" = pass NULL to cuDSS)
};

class CuDSSParameterParser {
private:
    static std::unordered_map<std::string, std::function<void(CuDSSConfig&)>> const configParsers;

public:
    static bool parseParameters(CuDSSConfig& config);
    static void printUsageInfo();
};

const std::unordered_map<std::string, std::function<void(CuDSSConfig&)>> 
CuDSSParameterParser::configParsers = {
    {"precision", [](CuDSSConfig& config) { 
        const char* value = OPS_GetString();
        if (value && (strcmp(value, "dDDI") == 0 || strcmp(value, "dFFI") == 0)) {
            config.precision = value;
        } else if (value) {
            throw std::invalid_argument("Invalid precision. Only dDDI and dFFI are supported.");
        }
    }},
    {"verbose", [](CuDSSConfig& config) { 
        int numData = 1;
        int flag = 0;
        if (OPS_GetIntInput(&numData, &flag) == 0) {
            if (flag != 0 && flag != 1) throw std::invalid_argument("verbose must be 0 or 1");
            config.verbose = (flag == 1);
        }
    }},
    {"hybridMemoryMode", [](CuDSSConfig& config) { 
        int numData = 1;
        int flag = 0;
        if (OPS_GetIntInput(&numData, &flag) == 0) {
            if (flag != 0 && flag != 1) throw std::invalid_argument("hybridMemoryMode must be 0 or 1");
            config.hybridMemoryMode = (flag == 1);
        }
    }},
    {"hybridDeviceMemoryLimit", [](CuDSSConfig& config) { 
        int numData = 1;
        double limit = 0.0;
        if (OPS_GetDoubleInput(&numData, &limit) == 0) {
            if (limit < 0.0) throw std::invalid_argument("hybridDeviceMemoryLimit cannot be negative");
            config.hybridDeviceMemoryLimit = static_cast<size_t>(limit);
        }
    }},
    {"hybridExecuteMode", [](CuDSSConfig& config) { 
        int numData = 1;
        int flag = 0;
        if (OPS_GetIntInput(&numData, &flag) == 0) {
            if (flag != 0 && flag != 1) throw std::invalid_argument("hybridExecuteMode must be 0 or 1");
            config.hybridExecuteMode = (flag == 1);
        }
    }},
    {"multiThreadingMode", [](CuDSSConfig& config) { 
        int numData = 1;
        int flag = 0;
        if (OPS_GetIntInput(&numData, &flag) == 0) {
            if (flag != 0 && flag != 1) throw std::invalid_argument("multiThreadingMode must be 0 or 1");
            config.multiThreadingMode = (flag == 1);
        }
    }},
    {"threadingLibPath", [](CuDSSConfig& config) { 
        const char* value = OPS_GetString();
        if (value) config.threadingLibPath = value;
    }}
};

bool CuDSSParameterParser::parseParameters(CuDSSConfig& config) {
    try {
        while (OPS_GetNumRemainingInputArgs() > 0) {
            const char* key = OPS_GetString();
            if (!key) {
                opserr << "WARNING: CuDSSParameterParser::parseParameters() - "
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
        opserr << "WARNING: CuDSSParameterParser::parseParameters() - "
               << e.what() << endln;
        return false;
    }
}

void CuDSSParameterParser::printUsageInfo() {
    opserr << "CuDSSParameterParser::printUsageInfo() - " << endln;
    opserr << "Usage: system CuDSS [options]" << endln;
    opserr << "Options:" << endln;
    opserr << "  -precision <dDDI|dFFI>          Precision mode (default: dDDI)" << endln;
    opserr << "  -verbose <0|1>                  Enable verbose output (default: 0)" << endln;
    opserr << "  -hybridMemoryMode <0|1>         Hybrid host/device memory mode (default: 0)" << endln;
    opserr << "  -hybridDeviceMemoryLimit <bytes> Device memory limit for hybrid mode (default: 0=auto)" << endln;
    opserr << "  -hybridExecuteMode <0|1>        Hybrid host/device execute mode (default: 0)" << endln;
    opserr << "  -multiThreadingMode <0|1>       OpenMP multi-threading mode (default: 0)" << endln;
    opserr << "                                  (requires OpenMP at build time)" << endln;
    opserr << "  -threadingLibPath <path|NULL>   Path to threading layer library" << endln;
    opserr << "                                  (default: /usr/lib/x86_64-linux-gnu/libcudss_mtlayer_gomp.so," << endln;
    opserr << "                                   use 'NULL' to let cuDSS choose via CUDSS_THREADING_LIB env var)" << endln;
    opserr << "Notes:" << endln;
    opserr << "  - hybridMemoryMode and hybridExecuteMode are mutually exclusive" << endln;
    opserr << "  - To control thread count: export OMP_NUM_THREADS=<n> before running OpenSees" << endln;
}

void* OPS_CuDSSLinSolver()
{
    CuDSSConfig config;
    
    // Parse command-line arguments
    if (!CuDSSParameterParser::parseParameters(config)) {
        opserr << "WARNING: OPS_CuDSSLinSolver() - "
               << "Failed to parse parameters, using defaults" << endln;
        opserr << "For valid parameters, use:" << endln;
        CuDSSParameterParser::printUsageInfo();
    }
    
    // Validate that hybrid modes are mutually exclusive
    if (config.hybridMemoryMode && config.hybridExecuteMode) {
        opserr << "ERROR: OPS_CuDSSLinSolver() - "
               << "hybridMemoryMode and hybridExecuteMode are mutually exclusive. "
               << "Only one can be enabled at a time." << endln;
        return nullptr;
    }
    
    // Validate that hybridDeviceMemoryLimit is only used with hybridMemoryMode
    if (config.hybridDeviceMemoryLimit > 0 && !config.hybridMemoryMode) {
        opserr << "WARNING: OPS_CuDSSLinSolver() - "
               << "hybridDeviceMemoryLimit is only valid with hybridMemoryMode enabled. "
               << "Ignoring hybridDeviceMemoryLimit." << endln;
    }
    
    // Create solver with all configuration options
    CuDSSLinSolver* solver = new CuDSSLinSolver(
        config.precision.c_str(), 
        config.verbose,
        config.hybridMemoryMode,
        config.hybridDeviceMemoryLimit,
        config.hybridExecuteMode,
        config.multiThreadingMode,
        config.threadingLibPath.c_str()
    );
    
    // CuDSS only supports scalar CSR (blockSize = 1, no padding)
    const int blockSize = 1;
    const bool paddingEnabled = false;
    
    // Create and return SOE based on precision
    if (config.precision == "dFFI") {
        return CudaGenBcsrLinSOE::createFloat(*solver, blockSize, paddingEnabled, config.verbose);
    } else {
        return CudaGenBcsrLinSOE::createDouble(*solver, blockSize, paddingEnabled, config.verbose);
    }
}
#else // _CUDSS
void* OPS_CuDSSLinSolver() {
    opserr << "WARNING: OPS_CuDSSLinSolver() - "
           << "cuDSS not available" << endln;
    return nullptr;
}
#endif // _CUDSS

