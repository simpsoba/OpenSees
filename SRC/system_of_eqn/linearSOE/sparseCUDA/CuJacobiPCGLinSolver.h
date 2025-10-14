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
// This is a subclass of CuPCGLinSolver that overrides the applyPreconditioner
// method to use a diagonal (Jacobi) preconditioner: M = diag(A)

#ifndef CuJacobiPCGLinSolver_h
#define CuJacobiPCGLinSolver_h

#include <CuPCGLinSolver.h>
#include <OPS_Stream.h>

// C++ includes
#include <string>

class CuJacobiPCGLinSolver : public CuPCGLinSolver
{
public:
    // Constructor - does not use an external preconditioner
    // (passes nullptr to CuPCGLinSolver constructor)
    CuJacobiPCGLinSolver(
        int maxIterations = 100,
        double relativeTolerance = 1e-6,
        double absoluteTolerance = 1e-12,
        bool verbose = false
    );
    
    // Destructor
    ~CuJacobiPCGLinSolver();
    
    // Override setSize to extract and store diagonal
    int setSize(void) override;
    
protected:
    // Override applyPreconditioner to implement Jacobi preconditioning
    // z = M^{-1} * r = (1 / diag(A)) .* r (element-wise)
    int applyPreconditioner(void* z, void* r, int n, bool updatePreconditioner) override;

private:
    #ifdef _CUDA
    // Device memory for diagonal inverse (1 / diag(A))
    void* m_d_diagInv;
    size_t m_diagAllocatedSize;
    bool m_diagonalExtracted;  // Track if diagonal has been extracted
    
    // Helper method to extract diagonal from matrix
    int extractDiagonal(bool force_update);
    
    // Template helper for applying Jacobi preconditioner
    template<typename T>
    int applyJacobiPreconditionerImpl(T* z, T* r, int n);
    
    // Template helper for extracting diagonal
    template<typename T>
    int extractDiagonalImpl();
    #endif // _CUDA
};
#endif

