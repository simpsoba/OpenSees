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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCUDA/CudaDirectSparseSolver.cpp,v $
                                                                        
                                                                        
// Written: gaaraujo 
// Created: 10/2025
//
// Description: This file contains the class definition for 
// CudaDirectSparseSolver. It solves the CudaGenBcsrLinSOE object by calling
// CuDSS routines.
//

// OpenSees includes
#include <classTags.h>
#include <OPS_Globals.h>

// Solve core classes
#include <CudaGenBcsrLinSOE.h>
#include <CudaDirectSparseSolver.h>
#include "ParameterUtils.h"

// CUDA utilities
#include "CudaUtils.h"

// for parsing command line arguments
#include <elementAPI.h>
#include <FileStream.h>
#include <unordered_map>
#include "ParameterUtils.h"

// C++ includes
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <stdexcept>

using namespace CudaUtils;

CudaDirectSparseSolver::CudaDirectSparseSolver(CudaPrecision precision, bool verbose, 
                               bool hybridMemoryMode, const std::vector<size_t>& hybridDeviceMemoryLimits, 
                               bool hybridExecuteMode, bool multiThreadingMode,
                               const char* threadingLibPath,
                               CuDSSMatrixType cudssMatType,
                               bool useMultiGPU,
                               const std::vector<int>& deviceIndices)
    :CudaGenBcsrLinSolver(SOLVER_TAGS_CudaDirectSparseSolver, precision), 
    m_verbose(verbose),
    m_hybridMemoryMode(hybridMemoryMode),
    m_hybridDeviceMemoryLimits(hybridDeviceMemoryLimits),
    m_hybridExecuteMode(hybridExecuteMode),
    m_multiThreadingMode(multiThreadingMode),
    m_threadingLibPath(threadingLibPath ? threadingLibPath : ""),
    m_cudssMatType(cudssMatType),
    m_useMultiGPU(useMultiGPU),
    m_deviceIndices(deviceIndices),
    m_matrix(nullptr)
{
    if (!isUniformPrecision(precision)) {
        opserr << "ERROR: CudaDirectSparseSolver::CudaDirectSparseSolver() - "
               << "Precision " << cudaPrecisionToString(precision) << " is not supported by cuDSS. "
               << "cuDSS only supports uniform precision: dDDI (double) and dFFI (float)." << endln;
        throw std::invalid_argument("CuDSS does not support mixed precision");
    }
    init(precision);
}

void CudaDirectSparseSolver::init(CudaPrecision precision)
{
    CudaCsrMatrix::Options opts;
    opts.precision = precision;
    opts.syncAfterSolve = true;
    opts.verbose = m_verbose;
    opts.hybridMemoryMode = m_hybridMemoryMode;
    opts.hybridDeviceMemoryLimits = m_hybridDeviceMemoryLimits;
    opts.hybridExecuteMode = m_hybridExecuteMode;
    opts.multiThreadingMode = m_multiThreadingMode;
    opts.threadingLibPath = m_threadingLibPath;
    opts.matKind = toCuDssMatrixKind(m_cudssMatType);
    opts.useMultiGPU = m_useMultiGPU;
    opts.deviceIndices = m_deviceIndices;
    m_matrix = new CudaCsrMatrix(opts);
    return;
}

CudaDirectSparseSolver::~CudaDirectSparseSolver()
{
    delete m_matrix;
    m_matrix = nullptr;
    return;
}

CuDssMatrixKind CudaDirectSparseSolver::toCuDssMatrixKind(CuDSSMatrixType type)
{
    switch (type) {
        case CuDSSMatrixType::SYMMETRIC:
            return CuDssMatrixKind::SYMMETRIC;
        case CuDSSMatrixType::SPD:
            return CuDssMatrixKind::SPD;
        default:
            return CuDssMatrixKind::FULL;
    }
}

int CudaDirectSparseSolver::setLinearSOE(CudaGenBcsrLinSOE &theSOE) {
    // cuDSS only supports scalar CSR (blockSize = 1), not BSR
    if (theSOE.getBlockSize() != 1) {
        opserr << "WARNING: CudaDirectSparseSolver::setLinearSOE() - "
               << "cuDSS only supports scalar CSR (blockSize = 1), got blockSize = " 
               << theSOE.getBlockSize() << endln;
        return -1;
    }
    
    // Check precision match using unified enum
    if (theSOE.getPrecision() != this->getPrecision()) {
        opserr << "WARNING: CudaDirectSparseSolver::setLinearSOE() - "
               << "precision mismatch: SOE is " << cudaPrecisionToString(theSOE.getPrecision())
               << ", solver is " << cudaPrecisionToString(this->getPrecision()) << endln;
        return -1;
    }
    
    return this->CudaGenBcsrLinSolver::setLinearSOE(theSOE);
}

int CudaDirectSparseSolver::solve(void) {
    if (m_matrix == nullptr) {
        opserr << "ERROR: CudaDirectSparseSolver::solve() - direct solver not initialized\n";
        return -1;
    }
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "WARNING: CudaDirectSparseSolver::solve() - LinearSOE not set" << endln;
        return -1;
    }

    CudaGenBcsrLinSOE::MatrixStatus matrixStatus = theSOE->getMatrixStatus();
    void* AValues = theSOE->getDeviceAValues();
    void* xValues = theSOE->getDeviceX();
    void* bValues = theSOE->getDeviceB();

    if (!AValues || !xValues || !bValues) {
        opserr << "ERROR: CudaDirectSparseSolver::solve() - null device pointer(s)" << endln;
        return -1;
    }

    if (setupMatrices() != 0) {
        opserr << "ERROR: CudaDirectSparseSolver::solve() - setupMatrices() failed" << endln;
        return -1;
    }

    m_matrix->bindValues(AValues);

    if (matrixStatus == CudaGenBcsrLinSOE::MatrixStatus::STRUCTURE_CHANGED) {
        if (m_matrix->factorize(bValues, xValues) != 0) {
            return -1;
        }
    } else if (matrixStatus == CudaGenBcsrLinSOE::MatrixStatus::COEFFICIENTS_CHANGED) {
        if (m_matrix->refactorize(bValues, xValues) != 0) {
            return -1;
        }
    }

    if (m_matrix->solve(bValues, xValues) != 0) {
        opserr << "WARNING: CudaDirectSparseSolver::solve() - cuDSS solve failed" << endln;
        return -1;
    }

    if (m_verbose) {
        opserr << "INFO: CudaDirectSparseSolver::solve() - cuDSS solve successful" << endln;
    }

    return 0;
}

int CudaDirectSparseSolver::setSize() {
    // In OpenSees, setSize() is called before data is ready on the GPU
    // Matrix initialization is done in solve() via setupMatrices()
    return 0;
}

int CudaDirectSparseSolver::setupMatrices() {
    if (m_matrix == nullptr) {
        opserr << "ERROR: CudaDirectSparseSolver::setupMatrices() - direct solver not initialized\n";
        return -1;
    }
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "WARNING: CudaDirectSparseSolver::setupMatrices() - LinearSOE not set" << endln;
        return -1;
    }

    CudaGenBcsrLinSOE::MatrixStatus matrixStatus = theSOE->getMatrixStatus();
    if (matrixStatus != CudaGenBcsrLinSOE::MatrixStatus::STRUCTURE_CHANGED) {
        return 0;
    }

    int numRows = theSOE->getNumRowBlocks();
    int numNZ = theSOE->getNumNonZeroValues();
    if (theSOE->getBlockSize() != 1) {
        opserr << "ERROR: CudaDirectSparseSolver::setupMatrices() - Only blockSize = 1 is supported" << endln;
        return -1;
    }

    int* rowPtrs = theSOE->getDeviceRowPtrs();
    int* colIndices = theSOE->getDeviceColIndices();
    void* AValues = theSOE->getDeviceAValues();
    if (!rowPtrs || !colIndices || !AValues) {
        opserr << "ERROR: CudaDirectSparseSolver::setupMatrices() - null device pointer(s)" << endln;
        return -1;
    }

    if (m_matrix->bindStructure(numRows, numNZ, rowPtrs, colIndices) != 0) {
        return -1;
    }
    if (m_matrix->bindValues(AValues) != 0) {
        return -1;
    }

    if (m_verbose) {
        opserr << "INFO: CudaDirectSparseSolver::setupMatrices() - CSR structure bound" << endln;
    }

    return 0;
}

// OpenSees API for creating CuDSS solver

struct CuDSSConfig {
    std::string precision = "dDDI";
    bool verbose = false;
    bool hybridMemoryMode = false;        // Hybrid host/device memory mode
    std::vector<size_t> hybridDeviceMemoryLimits;   // Per-device limit for hybrid memory (empty = use heuristic; one value = same for all)
    bool hybridExecuteMode = false;       // Hybrid host/device execute mode
    bool multiThreadingMode = false;      // OpenMP multi-threading mode (requires OpenMP at build time)
    std::string threadingLibPath = "/usr/lib/x86_64-linux-gnu/libcudss_mtlayer_gomp.so";  // Threading layer library path ("NULL" = pass NULL to cuDSS)
    std::string cudssMatTypeStr = "full"; // full | symmetric | spd (symmetric/spd use lower storage in SOE)
    std::string parallelMode = "single";  // single | multiGPU | MGMN (parallelism across processes/GPUs)
    int distributed = 0;                  // For MGMN: 0 = root-only (gather-scatter), 1 = row-wise distributed
    std::vector<int> deviceIndices;       // For multiGPU: empty = use all devices; else list of device IDs
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
        config.hybridDeviceMemoryLimits.clear();
        int numData = 1;
        double limit = 0.0;
        if (OPS_GetDoubleInput(&numData, &limit) != 0) return;
        if (limit < 0.0) throw std::invalid_argument("hybridDeviceMemoryLimit cannot be negative");
        config.hybridDeviceMemoryLimits.push_back(static_cast<size_t>(limit));
        while (OPS_GetNumRemainingInputArgs() > 0) {
            if (OPS_GetDoubleInput(&numData, &limit) != 0) break;
            if (limit < 0.0) throw std::invalid_argument("hybridDeviceMemoryLimit value cannot be negative");
            config.hybridDeviceMemoryLimits.push_back(static_cast<size_t>(limit));
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
    }},
    {"matrixType", [](CuDSSConfig& config) { 
        const char* value = OPS_GetString();
        if (value) {
            std::string s(value);
            if (s == "full" || s == "symmetric" || s == "spd") {
                config.cudssMatTypeStr = s;
            } else {
                throw std::invalid_argument("matrixType must be full, symmetric, or spd");
            }
        }
    }},
    {"parallelMode", [](CuDSSConfig& config) { 
        const char* value = OPS_GetString();
        if (value) {
            std::string s(value);
            if (s == "single" || s == "multiGPU" || s == "MGMN") {
                config.parallelMode = s;
            } else {
                throw std::invalid_argument("parallelMode must be single, multiGPU, or MGMN");
            }
        }
    }},
    {"distributed", [](CuDSSConfig& config) { 
        int numData = 1;
        int flag = 0;
        if (OPS_GetIntInput(&numData, &flag) == 0) {
            if (flag != 0 && flag != 1) throw std::invalid_argument("distributed must be 0 or 1");
            config.distributed = flag;
        }
    }},
    {"devices", [](CuDSSConfig& config) { 
        const char* value = OPS_GetString();
        if (!value) return;
        config.deviceIndices.clear();
        if (strcmp(value, "all") == 0) return;  // empty = use all devices
        int id = atoi(value);
        config.deviceIndices.push_back(id);
        int numData = 1;
        while (OPS_GetNumRemainingInputArgs() > 0) {
            int next = 0;
            if (OPS_GetIntInput(&numData, &next) != 0) break;
            config.deviceIndices.push_back(next);
        }
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

        // Validation: parallelMode vs single-process vs OpenSeesMP (works for Tcl and Python)
        int np = getNumProcesses();
        bool isParallel = (np > 1);
        if (config.parallelMode == "MGMN" && !isParallel) {
            opserr << "ERROR: CuDSSParameterParser::parseParameters() - "
                   << "parallelMode MGMN requires OpenSeesMP (multiple processes). Use single or multiGPU for single-process runs." << endln;
            return false;
        }
        if (config.parallelMode == "multiGPU" && isParallel) {
            opserr << "ERROR: CuDSSParameterParser::parseParameters() - "
                   << "parallelMode multiGPU is for single-process multi-GPU only. In OpenSeesMP use parallelMode MGMN." << endln;
            return false;
        }
        if (config.parallelMode == "MGMN" && (config.hybridMemoryMode || config.hybridExecuteMode)) {
            opserr << "ERROR: CuDSSParameterParser::parseParameters() - "
                   << "hybridMemoryMode and hybridExecuteMode are not allowed with parallelMode MGMN." << endln;
            return false;
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
    opserr << "  -precision <dDDI|dFFI>          Precision (default: dDDI)" << endln;
    opserr << "  -verbose <0|1>                  Enable verbose output (default: 0)" << endln;
    opserr << "  -parallelMode <single|multiGPU|MGMN>  Parallelism across processes/GPUs (default: single)" << endln;
    opserr << "                                  single: one process, one GPU" << endln;
    opserr << "                                  multiGPU: one process, multiple GPUs" << endln;
    opserr << "                                  MGMN: OpenSeesMP, multi-GPU multi-node (requires getNP > 1)" << endln;
    opserr << "  -distributed <0|1>               For MGMN only: 0 = root-only gather-scatter (default), 1 = row-wise distributed" << endln;
    opserr << "  -devices <all|id1 [id2 ...]>     For multiGPU only: GPU IDs to use (default: all)" << endln;
    opserr << "  -hybridMemoryMode <0|1>         Hybrid host/device memory mode (default: 0)" << endln;
    opserr << "  -hybridDeviceMemoryLimit <bytes1 [bytes2 ...]> Per-device limit for hybrid memory (one value=all devices; 0=min)" << endln;
    opserr << "  -hybridExecuteMode <0|1>        Hybrid host/device execute mode (default: 0)" << endln;
    opserr << "  -multiThreadingMode <0|1>       OpenMP multi-threading mode (default: 0; requires OpenMP at build time)" << endln;
    opserr << "  -threadingLibPath <path|NULL>   Path to threading layer (when multiThreadingMode is enabled)" << endln;
    opserr << "                                  (default: /usr/lib/x86_64-linux-gnu/libcudss_mtlayer_gomp.so," << endln;
    opserr << "                                   use 'NULL' to let cuDSS use CUDSS_THREADING_LIB env var)" << endln;
    opserr << "  -matrixType <full|symmetric|spd> Matrix type: full (default), symmetric, or spd" << endln;
    opserr << "                                  (symmetric and spd use lower storage; halves matrix memory)" << endln;
    opserr << "Notes:" << endln;
    opserr << "  - hybridMemoryMode and hybridExecuteMode are mutually exclusive; hybridExecute mode is not allowed with parallelMode MGMN" << endln;
    opserr << "  - MGMN is only valid in OpenSeesMP (getNP > 1); multiGPU is only valid for single process" << endln;
    opserr << "  - When multiThreadingMode is enabled: export OMP_NUM_THREADS=<n> to control thread count" << endln;
}

// Factory function to create CuDSS solver from parsed config
CudaGenBcsrLinSolver* createCuDSSSolverFromConfig(const CuDSSConfig& config) {
    // Convert string precision to enum
    CudaPrecision precision;
    if (!cudaPrecisionFromString(config.precision.c_str(), precision)) {
        opserr << "WARNING: createCuDSSSolverFromConfig() - "
               << "Invalid precision '" << config.precision.c_str() << "', defaulting to dDDI" << endln;
        precision = CudaPrecision::dDDI;
    }
    
    CuDSSMatrixType cudssMatType = CuDSSMatrixType::FULL;
    if (config.cudssMatTypeStr == "symmetric") cudssMatType = CuDSSMatrixType::SYMMETRIC;
    else if (config.cudssMatTypeStr == "spd") cudssMatType = CuDSSMatrixType::SPD;

    bool useMultiGPU = (config.parallelMode == "multiGPU");
    return new CudaDirectSparseSolver(
        precision, 
        config.verbose,
        config.hybridMemoryMode,
        config.hybridDeviceMemoryLimits,
        config.hybridExecuteMode,
        config.multiThreadingMode,
        config.threadingLibPath.c_str(),
        cudssMatType,
        useMultiGPU,
        config.deviceIndices
    );
}

// Factory function that parses OPS arguments and creates solver
CudaGenBcsrLinSolver* createCuDSSSolverFromParser() {
    CuDSSConfig config;
    
    // Parse command-line arguments
    if (!CuDSSParameterParser::parseParameters(config)) {
        opserr << "WARNING: createCuDSSSolverFromParser() - "
               << "Failed to parse parameters, using defaults" << endln;
        opserr << "For valid parameters, use:" << endln;
        CuDSSParameterParser::printUsageInfo();
    }
    
    // Validate that hybrid modes are mutually exclusive
    if (config.hybridMemoryMode && config.hybridExecuteMode) {
        opserr << "ERROR: createCuDSSSolverFromParser() - "
               << "hybridMemoryMode and hybridExecuteMode are mutually exclusive. "
               << "Only one can be enabled at a time." << endln;
        return nullptr;
    }
    
    // Validate that hybridDeviceMemoryLimits is only used with hybridMemoryMode
    if (!config.hybridDeviceMemoryLimits.empty() && !config.hybridMemoryMode) {
        opserr << "WARNING: createCuDSSSolverFromParser() - "
               << "hybridDeviceMemoryLimit is only valid with hybridMemoryMode enabled. "
               << "Ignoring hybridDeviceMemoryLimit." << endln;
    }
    
    return createCuDSSSolverFromConfig(config);
}

void* OPS_CudaDirectSparseSolver()
{
    CuDSSConfig config;

    // Expand dict to CLI args if present ({"key": val} -> "-key", val)
    if (OPS_GetNumRemainingInputArgs() == 1) {
        (void)OPS_ExpandDictArgs();
    }

    // Parse CLI-style parameters (after any normalization).
    if (!CuDSSParameterParser::parseParameters(config)) {
        opserr << "WARNING: OPS_CudaDirectSparseSolver() - "
               << "Failed to parse parameters, using defaults" << endln;
        opserr << "For valid parameters, use:" << endln;
        CuDSSParameterParser::printUsageInfo();
    }

    // Validate that hybrid modes are mutually exclusive
    if (config.hybridMemoryMode && config.hybridExecuteMode) {
        opserr << "ERROR: OPS_CudaDirectSparseSolver() - "
               << "hybridMemoryMode and hybridExecuteMode are mutually exclusive. "
               << "Only one can be enabled at a time." << endln;
        return nullptr;
    }

    // Validate that hybridDeviceMemoryLimits is only used with hybridMemoryMode
    if (!config.hybridDeviceMemoryLimits.empty() && !config.hybridMemoryMode) {
        opserr << "WARNING: OPS_CudaDirectSparseSolver() - "
               << "hybridDeviceMemoryLimit is only valid with hybridMemoryMode enabled. "
               << "Ignoring hybridDeviceMemoryLimit." << endln;
    }

    CudaGenBcsrLinSolver* solver = nullptr;
    try {
        solver = createCuDSSSolverFromConfig(config);
    } catch (const std::exception& e) {
        opserr << "ERROR: OPS_CudaDirectSparseSolver() - "
               << "Failed to create solver: " << e.what() << endln;
        return nullptr;
    }
    if (solver == nullptr) return nullptr;

    const int blockSize = 1;
    const bool paddingEnabled = false;
    CudaPrecision precision = solver->getPrecision();
    const bool symmetricStorage = (config.cudssMatTypeStr == "symmetric" || config.cudssMatTypeStr == "spd");
    switch (precision) {
        case CudaPrecision::dDDI:
            return CudaGenBcsrLinSOE::createDouble(*solver, blockSize, paddingEnabled, config.verbose, symmetricStorage);
        case CudaPrecision::dFFI:
            return CudaGenBcsrLinSOE::createFloat(*solver, blockSize, paddingEnabled, config.verbose, symmetricStorage);
        default:
            opserr << "ERROR: OPS_CudaDirectSparseSolver() - Unexpected precision mode" << endln;
            delete solver;
            return nullptr;
    }
}

