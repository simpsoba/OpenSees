/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
** (C) Copyright 1999, The Regents of the University of California    **
** All Rights Reserved.                                               **
**                                                                    **
** Description: cuSPARSE SpMV backend for CSR and BSR device matrices.**
** ****************************************************************** */

#include "CuSparseBackend.h"
#include <OPS_Globals.h>

using namespace CudaUtils;

CuSparseBackend::Config::Config()
    : precision(CudaPrecision::dDDI), externalHandle(nullptr), stream(nullptr), sharedPattern(nullptr)
{
}

CuSparseBackend::CuSparseBackend(const Config &config) : m_config(config) {}

CuSparseBackend::~CuSparseBackend() { reset(); }

cudaDataType_t CuSparseBackend::valueType() const
{
    return (m_config.precision == CudaPrecision::dFFI) ? CUDA_R_32F : CUDA_R_64F;
}

void CuSparseBackend::destroyDescriptors() const
{
    if (m_buffer != nullptr) {
        cudaFree(m_buffer);
        m_buffer = nullptr;
        m_bufferSize = 0;
    }
    if (m_vecX != nullptr) {
        cusparseDestroyDnVec(m_vecX);
        m_vecX = nullptr;
    }
    if (m_vecY != nullptr) {
        cusparseDestroyDnVec(m_vecY);
        m_vecY = nullptr;
    }
    if (m_spmat != nullptr) {
        cusparseDestroySpMat(m_spmat);
        m_spmat = nullptr;
    }
    m_ready = false;
}

void CuSparseBackend::reset()
{
    destroyDescriptors();
    if (m_config.externalHandle == nullptr && m_localHandle != nullptr) {
        cusparseDestroy(m_localHandle);
    }
    m_localHandle = nullptr;

    m_blockSize = 1;
    m_numRows = 0;
    m_numNnz = 0;
    m_numDofs = 0;
    m_rowPtr = nullptr;
    m_colIdx = nullptr;
    m_values = nullptr;
}

cusparseHandle_t CuSparseBackend::getHandle() const
{
    if (m_config.externalHandle != nullptr) {
        return m_config.externalHandle;
    }
    return m_localHandle;
}

int CuSparseBackend::ensureHandle() const
{
    if (m_config.externalHandle != nullptr) {
        return 0;
    }
    if (m_localHandle == nullptr) {
        cuSparseCheckError(cusparseCreate(&m_localHandle), "cusparseCreate");
    }
    if (m_config.stream != nullptr) {
        cuSparseCheckError(cusparseSetStream(m_localHandle, m_config.stream), "cusparseSetStream");
    }
    return 0;
}

int CuSparseBackend::bindStructure(int blockSize, int numRows, int numNnz, int *rowPtr, int *colIdx)
{
    if (blockSize <= 0 || numRows <= 0 || numNnz < 0 || rowPtr == nullptr || colIdx == nullptr) {
        opserr << "ERROR CuSparseBackend::bindStructure() - invalid arguments\n";
        return -1;
    }
    if (m_blockSize == blockSize && m_numRows == numRows && m_numNnz == numNnz && m_rowPtr == rowPtr &&
        m_colIdx == colIdx && m_ready) {
        return 0;
    }

    destroyDescriptors();
    m_blockSize = blockSize;
    m_numRows = numRows;
    m_numNnz = numNnz;
    m_numDofs = numRows * blockSize;
    m_rowPtr = rowPtr;
    m_colIdx = colIdx;
    return 0;
}

int CuSparseBackend::bindValues(void *values)
{
    if (values == nullptr) {
        opserr << "ERROR CuSparseBackend::bindValues() - null values\n";
        return -1;
    }
    m_values = values;
    if (m_ready) {
        return rebuild();
    }
    return 0;
}

int CuSparseBackend::rebuild() const
{
    destroyDescriptors();
    if (m_numRows <= 0 || m_numNnz < 0 || m_rowPtr == nullptr || m_colIdx == nullptr || m_values == nullptr) {
        return -1;
    }
    if (ensureHandle() != 0) {
        return -1;
    }

    const cudaDataType_t dtype = valueType();
    cusparseHandle_t handle = getHandle();

    if (m_blockSize == 1) {
        cuSparseCheckError(cusparseCreateCsr(&m_spmat, m_numRows, m_numRows, m_numNnz, m_rowPtr, m_colIdx, m_values,
                                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                             dtype),
                           "cusparseCreateCsr");
    } else {
        cuSparseCheckError(cusparseCreateBsr(&m_spmat, m_numRows, m_numRows, m_numNnz, m_blockSize, m_blockSize,
                                              m_rowPtr, m_colIdx, m_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_BASE_ZERO, dtype, CUSPARSE_ORDER_ROW),
                           "cusparseCreateBsr");
    }

    cuSparseCheckError(cusparseCreateDnVec(&m_vecX, m_numDofs, m_values, dtype), "cusparseCreateDnVec x");
    cuSparseCheckError(cusparseCreateDnVec(&m_vecY, m_numDofs, m_values, dtype), "cusparseCreateDnVec y");

    size_t bufSize = 0;
    if (dtype == CUDA_R_32F) {
        float one = 1.0f;
        float zero = 0.0f;
        cuSparseCheckError(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, m_spmat, m_vecX,
                                                     &zero, m_vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize),
                           "cusparseSpMV_bufferSize");
    } else {
        double one = 1.0;
        double zero = 0.0;
        cuSparseCheckError(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, m_spmat, m_vecX,
                                                     &zero, m_vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize),
                           "cusparseSpMV_bufferSize");
    }
    if (bufSize > 0) {
        cudaCheckError(cudaMalloc(&m_buffer, bufSize), "allocate SpMV buffer");
        m_bufferSize = bufSize;
    }
    m_ready = true;
    return 0;
}

int CuSparseBackend::multiply(const void *coeffs, const void *x, void *y, double alpha, double beta) const
{
    if (!m_ready) {
        return -1;
    }

    if (coeffs != m_values) {
        cuSparseCheckError(cusparseSpMatSetValues(m_spmat, const_cast<void *>(coeffs)), "cusparseSpMatSetValues");
    }

    cusparseHandle_t handle = getHandle();
    cuSparseCheckError(cusparseDnVecSetValues(m_vecX, const_cast<void *>(x)), "cusparseDnVecSetValues x");
    cuSparseCheckError(cusparseDnVecSetValues(m_vecY, y), "cusparseDnVecSetValues y");

    int rc = 0;
    const cudaDataType_t dtype = valueType();
    if (dtype == CUDA_R_32F) {
        const float alphaF = static_cast<float>(alpha);
        const float betaF = static_cast<float>(beta);
        if (cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alphaF, m_spmat, m_vecX, &betaF, m_vecY,
                         CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, m_buffer) != CUSPARSE_STATUS_SUCCESS) {
            rc = -1;
        }
    } else if (cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, m_spmat, m_vecX, &beta, m_vecY,
                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, m_buffer) != CUSPARSE_STATUS_SUCCESS) {
        rc = -1;
    }

    if (coeffs != m_values) {
        cuSparseCheckError(cusparseSpMatSetValues(m_spmat, m_values), "cusparseSpMatSetValues restore");
    }
    if (rc != 0) {
        opserr << "ERROR CuSparseBackend::multiply() - cusparseSpMV failed\n";
        return -1;
    }
    return 0;
}

int CuSparseBackend::spmv(const void *x, void *y, double alpha, double beta, const void *values) const
{
    const void *coeffs = (values != nullptr) ? values : m_values;
    if (m_config.sharedPattern != nullptr) {
        return m_config.sharedPattern->spmv(x, y, alpha, beta, coeffs);
    }

    if (m_rowPtr == nullptr || m_colIdx == nullptr || coeffs == nullptr) {
        opserr << "ERROR CuSparseBackend::spmv() - backend not ready\n";
        return -1;
    }
    if (!m_ready && rebuild() != 0) {
        return -1;
    }
    return multiply(coeffs, x, y, alpha, beta);
}
