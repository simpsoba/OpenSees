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
** Description: Device CSR matrix operator (Matrix.h-style ownership).**
** Supports cuSPARSE SpMV and optional cuDSS direct factorization/solve.**
** ****************************************************************** */

#ifndef CudaCsrMatrix_h
#define CudaCsrMatrix_h

#include "CuDSSBackend.h"
#include "CuSparseBackend.h"
#include "CudaUtils.h"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_vector.h>

#include <cstddef>

class CudaCsrMatrix
{
public:
    using SolverConfig = CuDSSBackend::Config;

    struct SpmvConfig {
        cusparseHandle_t externalHandle;
        const CuSparseBackend *sharedPattern;

        SpmvConfig() : externalHandle(nullptr), sharedPattern(nullptr) {}
    };

    struct ExecutionContext {
        cudaStream_t stream;

        ExecutionContext() : stream(nullptr) {}
    };

    CudaCsrMatrix(const SolverConfig &solver = SolverConfig{},
                  SpmvConfig spmv = SpmvConfig{},
                  ExecutionContext exec = ExecutionContext{});
    ~CudaCsrMatrix();

    CudaCsrMatrix(const CudaCsrMatrix &) = delete;
    CudaCsrMatrix &operator=(const CudaCsrMatrix &) = delete;

    void reset();
    void detachSharedSpmvPattern() { m_spmv.sharedPattern = nullptr; }

    int bindStructure(int numRows, int numNnz, const int *rowPtr, const int *colIdx);
    int copyStructure(int numRows, int numNnz, const int *rowPtr, const int *colIdx);

    int bindValues(void *values);
    int copyValues(const void *deviceValues);

    int spmv(const void *x, void *y, double alpha = 1.0, double beta = 0.0,
             const void *values = nullptr) const;

    int factorize(void *rhs, void *solution, int numRhs = 1, const CudaCsrMatrix *symbolicSource = nullptr);
    int refactorize(void *rhs, void *solution, int numRhs = 1);
    int solve(void *rhs, void *solution, int numRhs = 1);

    CuDSSBackend *getCuDSSBackend();
    const CuDSSBackend *getCuDSSBackend() const;

    CuSparseBackend *getSpmvBackend();
    const CuSparseBackend *getSpmvBackend() const;

    bool isStructureBound() const { return m_structureBound; }
    bool isFactored() const;
    bool ownsValues() const { return m_ownsValues; }
    bool ownsStructure() const { return m_ownsStructure; }

    int getNumRows() const { return m_numRows; }
    int getNumNnz() const { return m_numNnz; }
    void *getValues();
    const void *getValues() const;

    cudaStream_t getStream() const { return m_exec.stream; }
    cusparseHandle_t getCusparseHandle() const;

    const SolverConfig &getSolverConfig() const { return m_solver; }
    const SpmvConfig &getSpmvConfig() const { return m_spmv; }

private:
    SolverConfig m_solver;
    SpmvConfig m_spmv;
    ExecutionContext m_exec;

    int m_numRows = 0;
    int m_numNnz = 0;
    bool m_structureBound = false;
    bool m_ownsStructure = false;
    bool m_ownsValues = false;

    const int *m_rowPtr = nullptr;
    const int *m_colIdx = nullptr;
    void *m_values = nullptr;

    thrust::device_vector<int> m_ownedRowPtr;
    thrust::device_vector<int> m_ownedColIdx;
    thrust::device_vector<char> m_ownedValues;

    CuSparseBackend *m_spmvBackend = nullptr;

    CuDSSBackend *m_cuDSS = nullptr;

    void destroySpmvBackend();
    void destroyCuDSS();
    int syncSpmvBackend() const;
    CuDSSBackend::Config makeCuDSSConfig() const;
    int *rowPtr() const;
    int *colIdx() const;
    std::size_t valueSize() const;
};

#endif
