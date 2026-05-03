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

// ParameterUtils.cpp -- getNumProcesses() for CuDSS (compiled into OPS_Cuda_Serial and/or
// OPS_Cuda_Parallel with different macros; see CMakeLists.txt).

#include "ParameterUtils.h"

#if defined(OPS_CUDA_PARALLEL_MPI_SIZE)
#include <mpi.h>
#endif

int getNumProcesses()
{
#if defined(OPS_CUDA_PARALLEL_MPI_SIZE)
    int np = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    return np;
#else
    return 1;
#endif
}
