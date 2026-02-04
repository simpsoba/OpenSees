/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
** (C) Copyright 1999, The Regents of the University of California    **
** All Rights Reserved.                                               **
**                                                                    **
** Commercial use of this program without express permission of the   **
** University of California, Berkeley, is strictly prohibited.  See   **
** file 'COPYRIGHT'  in main directory for information on usage and   **
** redistribution,  and for a DISCLAIMER OF ALL WARRANTIES.           **
**                                                                    **
** ****************************************************************** */

// ParameterUtils.cpp -- getNumProcesses() implementation (sparseCUDA scope only)

#include "ParameterUtils.h"

#ifdef _PARALLEL_INTERPRETERS
#include <mpi.h>
#endif

int getNumProcesses()
{
#ifdef _PARALLEL_INTERPRETERS
    int np = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    return np;
#else
    return 1;
#endif
}
