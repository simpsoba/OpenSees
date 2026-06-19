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
                                                                       
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCUDA/CudaDirectSparseSolver.h
//
// Written: gaaraujo
// Created: 10/2025
//
// Description: OpenSees LinearSOESolver for GPU direct sparse solve via cuDSS.
// A CudaDirectSparseSolver solves a CudaGenBcsrLinSOE by delegating to CudaCsrMatrix.
//
// The CuDSS library provides a high-performance GPU-accelerated direct solver
// for sparse linear systems of the form AX = B. The cuDSS functionality allows 
// flexibility in matrix properties and solver configuration, as well as 
// execution parameters like CUDA streams.
//
// This solver is only built when cuDSS is available (OPS_Cuda_CuDSS).
//

#ifndef CudaDirectSparseSolver_h
#define CudaDirectSparseSolver_h

// OpenSees includes
#include <CudaGenBcsrLinSolver.h>
#include <OPS_Stream.h>
#include "CudaCsrMatrix.h"

// C++ includes
#include <cudss.h>
#include <string>
#include <vector>

// Matrix type for cuDSS: full, symmetric, or SPD (symmetric positive definite).
// symmetric and spd both use symmetric lower storage in CudaGenBcsrLinSOE.
// CuDSSMatrixType is defined in CuDSSBackend.h.

class CudaDirectSparseSolver : public CudaGenBcsrLinSolver
{
public:
    // Constructor with default parameters
    // useMultiGPU: use cuDSS multi-GPU (MG) mode; deviceIndices: GPU IDs (empty = use all)
    CudaDirectSparseSolver(
        CudaPrecision precision = CudaPrecision::dDDI,
        bool verbose = false,
        bool hybridMemoryMode = false,
        const std::vector<size_t>& hybridDeviceMemoryLimits = {},
        bool hybridExecuteMode = false,
        bool multiThreadingMode = false,
        const char* threadingLibPath = nullptr,
        CuDSSMatrixType cudssMatType = CuDSSMatrixType::FULL,
        bool useMultiGPU = false,
        const std::vector<int>& deviceIndices = {},
        int irNSteps = 0,
        double irTol = 0.0
    );
    
    // Destructor
    ~CudaDirectSparseSolver();
    
    // Solver methods
    int solve(void) override;
    int setSize(void) override;
    int setLinearSOE(CudaGenBcsrLinSOE &theSOE) override;
    LinearSOESolver *getCopy(void) const override;

protected:

private:
    // Verbosity flag
    bool m_verbose;
    
    // Hybrid mode settings
    bool m_hybridMemoryMode;
    std::vector<size_t> m_hybridDeviceMemoryLimits;  // Per-device limit (empty = use heuristic)
    bool m_hybridExecuteMode;
    
    // Multi-threading settings
    bool m_multiThreadingMode;
    std::string m_threadingLibPath;

    // Matrix type: full, symmetric, or SPD (affects cuDSS mtype; symmetric/SPD use lower storage)
    CuDSSMatrixType m_cudssMatType;

    // Multi-GPU (MG) mode: when true, use cudssCreateMg; otherwise cudssCreate
    bool m_useMultiGPU;
    std::vector<int> m_deviceIndices;

    // Iterative refinement (CUDSS_CONFIG_IR_N_STEPS / CUDSS_CONFIG_IR_TOL); 0 steps = disabled
    int m_irNSteps;
    double m_irTol;

    // Helper function to initialize cuDSS matrices when structure changes
    int setupMatrices();

    // cuDSS initializer (to be used by constructors only)
    void init(CudaPrecision precision);

    CudaCsrMatrix::SolverConfig makeSolverConfig(CudaPrecision precision) const;
    int ensureMatrix(CudaGenBcsrLinSOE *theSOE);

    CudaCsrMatrix *m_matrix = nullptr;
};
#endif
