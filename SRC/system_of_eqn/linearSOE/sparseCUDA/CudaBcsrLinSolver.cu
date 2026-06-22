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
                                                                        
// $Revision: 1.1 $
// $Date: 2005-04-08 02:38:18 $
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCuda/CudaBcsrLinSolver.cpp,v $
                                                                        
// Written: gaaraujo 
// Created: 08/2025
//
// Description: This file contains the implementation for CudaBcsrLinSolver
//
// What: "@(#) CudaBcsrLinSolver.C, revA"

// OpenSees includes
#include <CudaBcsrLinSolver.h>
#include <CudaBcsrLinSOE.h>

CudaBcsrLinSolver::CudaBcsrLinSolver(int theClassTag, CudaPrecision precision)    
:LinearSOESolver(theClassTag),
 theSOE(nullptr),
 m_precision(precision)
{

}    

CudaBcsrLinSolver::~CudaBcsrLinSolver()    
{

}    

int CudaBcsrLinSolver::setLinearSOE(CudaBcsrLinSOE &theCudaBcsrLinSOE)
{
    theSOE = &theCudaBcsrLinSOE;
    return 0;
}

CudaBcsrLinSOE* CudaBcsrLinSolver::getLinearSOE(void) const
{
    return theSOE;
}

int CudaBcsrLinSolver::setSize(void)
{
    return 0;
}

int CudaBcsrLinSolver::sendSelf(int commitTag, Channel &theChannel)
{
    return 0;
}

int CudaBcsrLinSolver::recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    return 0;
}

LinearSOESolver *
CudaBcsrLinSolver::getCopy(void) const
{
    return nullptr;
}
