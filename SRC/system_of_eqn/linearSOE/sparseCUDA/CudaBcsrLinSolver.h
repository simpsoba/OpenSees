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
** ****************************************************************** */                                                                        
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCuda/CudaBcsrLinSolver.h
//
// Written: gaaraujo 
// Created: 08/2025
//
// Description: This file contains the class definition for 
// CudaBcsrLinSolver. It solves a CudaBcsrLinSOE object.

#ifndef CudaBcsrLinSolver_h
#define CudaBcsrLinSolver_h

// OpenSees includes
#include <LinearSOESolver.h>
#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include "CudaUtils.h"

// Forward declarations
class CudaBcsrLinSOE;

class CudaBcsrLinSolver : public LinearSOESolver
{
public:
    CudaBcsrLinSolver(int classTag, CudaPrecision precision = CudaPrecision::dDDI);    
    virtual ~CudaBcsrLinSolver();

    // Set the associated CudaBcsrLinSOE object
    virtual int setLinearSOE(CudaBcsrLinSOE &theSOE);
    
    // Get the associated SOE
    CudaBcsrLinSOE* getLinearSOE(void) const;
    
    // Get solver precision
    CudaPrecision getPrecision(void) const { return m_precision; }

    // Abstract methods that must be implemented by subclasses
    int solve(void) override = 0;
    virtual int setSize(void) override;
    LinearSOESolver *getCopy(void) const override;

    // Parallel communication methods
    int sendSelf(int commitTag, Channel &theChannel) override;   
    int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker) override;  

protected:
    CudaBcsrLinSOE* theSOE;
    CudaPrecision m_precision;

private:
};
#endif