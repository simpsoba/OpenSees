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

// $Source: OpenSees/SRC/actor/objectBroker/FEM_ObjectBrokerCuda.cu,v $

// Written: gaaraujo 
// Created: 09/2025
//
// Description: This file contains CUDA-specific object creation for 
// FEM_ObjectBrokerAllClasses. It's separated to avoid Thrust compilation
// issues in non-CUDA compilation units.
//

#include "FEM_ObjectBrokerAllClasses.h"
#include "LinearSOE.h"

#ifdef _CUDA
#include "CudaGenBcsrLinSOE.h"
#endif // _CUDA

// Helper function for CUDA LinearSOE creation
LinearSOE* createCudaLinearSOE(int classTagSOE)
{
#ifdef _CUDA
    return CudaGenBcsrLinSOE::createFromClassTag(classTagSOE);
#else
    return nullptr;
#endif
}
