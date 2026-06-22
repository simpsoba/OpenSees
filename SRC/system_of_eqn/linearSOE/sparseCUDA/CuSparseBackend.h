/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
** (C) Copyright 1999, The Regents of the University of California    **
** All Rights Reserved.                                               **
**                                                                    **
** Description: cuSPARSE SpMV backend for CSR and BSR device matrices.**
** Composed by CudaCsrMatrix and CudaBcsrLinSOE; independent of cuDSS.**
** ****************************************************************** */

#ifndef CuSparseBackend_h
#define CuSparseBackend_h

#include "CudaUtils.h"

#include <cuda_runtime.h>
#include <cusparse.h>

class CuSparseBackend
{
public:
    struct Config {
        CudaPrecision precision;
        cusparseHandle_t externalHandle;
        cudaStream_t stream;
        // When set, reuse cuSPARSE descriptors/work buffer from this backend.
        const CuSparseBackend *sharedPattern;

        Config();
    };

    explicit CuSparseBackend(const Config &config = Config());
    ~CuSparseBackend();

    CuSparseBackend(const CuSparseBackend &) = delete;
    CuSparseBackend &operator=(const CuSparseBackend &) = delete;

    void reset();

    cusparseHandle_t getHandle() const;
    bool isReady() const { return m_ready; }
    int getBlockSize() const { return m_blockSize; }

    /**
     * Bind sparsity pattern (borrowed device pointers).
     * blockSize==1: numRows = DOF rows, numNnz = scalar nnz.
     * blockSize>1: numRows = block rows, numNnz = nnzb.
     */
    int bindStructure(int blockSize, int numRows, int numNnz, int *rowPtr, int *colIdx);

    /** Bind coefficient buffer (borrowed device pointer). */
    int bindValues(void *values);

    /** y = alpha * A * x + beta * y */
    int spmv(const void *x, void *y, double alpha = 1.0, double beta = 0.0, const void *values = nullptr) const;

private:
    Config m_config;

    int m_blockSize = 1;
    int m_numRows = 0;
    int m_numNnz = 0;
    int m_numDofs = 0;
    int *m_rowPtr = nullptr;
    int *m_colIdx = nullptr;
    void *m_values = nullptr;

    mutable cusparseHandle_t m_localHandle = nullptr;

    mutable cusparseSpMatDescr_t m_spmat = nullptr;
    mutable cusparseDnVecDescr_t m_vecX = nullptr;
    mutable cusparseDnVecDescr_t m_vecY = nullptr;
    mutable void *m_buffer = nullptr;
    mutable size_t m_bufferSize = 0;
    mutable bool m_ready = false;

    cudaDataType_t valueType() const;
    void destroyDescriptors() const;
    int ensureHandle() const;
    int rebuild() const;
    int multiply(const void *coeffs, const void *x, void *y, double alpha, double beta) const;
};

#endif
