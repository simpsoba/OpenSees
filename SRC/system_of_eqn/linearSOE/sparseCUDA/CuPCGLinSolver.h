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
    // updateFrequency: Update preconditioner every N solves (0 = never, 1 = always, N > 1 = periodic)
    // updateOnFailure: If true, also update when PCG fails regardless of frequency
    CuPCGLinSolver(
        CudaGenBcsrLinSolver* preconditioner,
        int maxIterations = 100,
        double relativeTolerance = 1e-6,
        double absoluteTolerance = 1e-12,
        int updateFrequency = 1,
        bool updateOnFailure = true,
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
    // Virtual method for applying preconditioner (can be overridden by derived classes)
    // Uses void* pointers with runtime type dispatch via getPrecision()
    virtual int applyPreconditioner(void* z, void* r, int n, bool updatePreconditioner);

private:
    // Configuration
    bool m_verbose;
    int m_maxIterations;
    double m_relativeTolerance;
    double m_absoluteTolerance;
    int m_updateFrequency;     // Update preconditioner every N solves (0 = never)
    bool m_updateOnFailure;    // Update preconditioner when PCG fails
    
    // Solve statistics
    int m_lastIterationCount;
    int m_numRefactorizations;
    int m_solvesSinceUpdate;   // Counter for periodic updates
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
    
    // PCG workspace vectors (device memory) - allocated in one contiguous block
    void* m_d_workspaceBlock; // Single allocation for all vectors
    void* m_d_x;           // solution workspace (points into workspace block)
    void* m_d_r;           // residual (points into workspace block)
    void* m_d_z;           // preconditioned residual (points into workspace block)
    void* m_d_p;           // search direction (points into workspace block)
    void* m_d_Ap;          // A * p (points into workspace block)
    void* m_d_temp;        // temporary vector for RHS storage (points into workspace block)
    
    // Precision (matrix and vector types can differ)
    cudaDataType_t m_MatrixValueType;
    cudaDataType_t m_VectorValueType;
    
    // Helper methods for PCG
    template<typename T>
    int solvePCG_impl(T* x, T* b, int n, bool updatePreconditioner);
    
    // Template helper for preconditioner application (used by virtual method)
    template<typename T>
    int applyPreconditionerImpl(T* z, T* r, int n, bool updatePreconditioner);

    void init(const char* precision);
    void cleanup();
    #endif // _CUDA
};
#endif

