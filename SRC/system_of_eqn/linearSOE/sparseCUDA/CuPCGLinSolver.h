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
// Description: Preconditioned Conjugate Gradient solver using cuSPARSE for SpMV
// and any CudaGenBcsrLinSolver as preconditioner. First solve uses the preconditioner
// directly, subsequent solves use PCG. Refactorization is triggered when PCG
// iterations exceed a user-specified threshold.

#ifndef CuPCGLinSolver_h
#define CuPCGLinSolver_h

#include <CudaGenBcsrLinSolver.h>
#include <OPS_Stream.h>

// C++ includes
#include <string>
#include <memory>

// CUDA includes
#ifdef _CUDA
#include <cusparse.h>
#include <cublas_v2.h>
#endif // _CUDA

class CuPCGLinSolver : public CudaGenBcsrLinSolver
{
public:
    // Constructor - takes ownership of the preconditioner (can be nullptr for identity)
    CuPCGLinSolver(
        CudaGenBcsrLinSolver* preconditioner,
        int maxIterations = 100,
        double relativeTolerance = 1e-6,
        double absoluteTolerance = 1e-12,
        bool refactorOnNonConvergence = true,
        bool verbose = false
    );
    
    // Destructor
    ~CuPCGLinSolver();
    
    // Solver methods
    int solve(void) override;
    int setSize(void) override;
    int setLinearSOE(CudaGenBcsrLinSOE &theSOE) override;
    
    // Query methods
    int getNumIterations() const { return m_lastIterationCount; }
    int getNumRefactorizations() const { return m_numRefactorizations; }
    
protected:

private:
    // Configuration
    bool m_verbose;
    int m_maxIterations;
    double m_relativeTolerance;
    double m_absoluteTolerance;
    bool m_refactorOnNonConvergence;
    
    // Solve statistics
    int m_lastIterationCount;
    int m_numRefactorizations;
    bool m_isFirstSolve;
    
    // Preconditioner (takes ownership)
    std::unique_ptr<CudaGenBcsrLinSolver> m_preconditioner;
    
    // Helper functions
    int solvePCG();        // PCG solve with preconditioner
    
    #ifdef _CUDA
    // Static members (shared across instances)
    static bool m_CuSparseInitialized;
    static int m_ActiveSolverInstances;
    static cusparseHandle_t m_cuSparseHandle;
    static cublasHandle_t m_cublasHandle;
    static cudaStream_t m_cudaStream;
    
    // cuSPARSE objects (for PCG SpMV)
    cusparseSpMatDescr_t m_spMatDescr;
    cusparseDnVecDescr_t m_vecX;
    cusparseDnVecDescr_t m_vecY;
    void* m_dBuffer;       // cuSPARSE workspace buffer
    size_t m_bufferSize;
    
    // PCG workspace vectors (device memory)
    void* m_d_x;           // solution workspace
    void* m_d_r;           // residual
    void* m_d_z;           // preconditioned residual
    void* m_d_p;           // search direction
    void* m_d_Ap;          // A * p
    void* m_d_temp;        // temporary vector for RHS storage
    
    // Precision
    cudaDataType_t m_ValueType;
    
    // Helper methods for PCG
    template<typename T>
    int solvePCG_impl(T* x, T* b, int n);
    
    template<typename T>
    int applyPreconditioner(T* z, T* r, int n);

    void init(const char* precision);
    void cleanup();
    #endif // _CUDA
};
#endif

