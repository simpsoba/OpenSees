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

#include "CudaCsrMatrix.h"
#include <OPS_Globals.h>
#include <thrust/memory.h>

using namespace CudaUtils;
using thrust::raw_pointer_cast;

CudaCsrMatrix::CudaCsrMatrix() = default;

CudaCsrMatrix::CudaCsrMatrix(const Options &options) : m_options(options) {}

CudaCsrMatrix::~CudaCsrMatrix() { reset(); }

std::size_t CudaCsrMatrix::valueSize() const
{
    return (m_options.precision == CudaPrecision::dFFI) ? sizeof(float) : sizeof(double);
}

cudaDataType_t CudaCsrMatrix::valueCudaType() const
{
    return (m_options.precision == CudaPrecision::dFFI) ? CUDA_R_32F : CUDA_R_64F;
}

void CudaCsrMatrix::reset()
{
    destroySpMV();
    destroyCuDSS();
    if (m_ownsCusparseHandle && m_localCusparseHandle != nullptr) {
        cusparseDestroy(m_localCusparseHandle);
    }
    m_localCusparseHandle = nullptr;
    m_ownsCusparseHandle = false;

    m_ownedRowPtr.clear();
    m_ownedColIdx.clear();
    m_ownedValues.clear();

    m_rowPtr = nullptr;
    m_colIdx = nullptr;
    m_values = nullptr;
    m_numRows = 0;
    m_numNnz = 0;
    m_structureBound = false;
    m_ownsStructure = false;
    m_ownsValues = false;
}

void CudaCsrMatrix::destroySpMV()
{
    if (m_spmvBuffer != nullptr) {
        cudaFree(m_spmvBuffer);
        m_spmvBuffer = nullptr;
        m_spmvBufferSize = 0;
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
    m_spmvReady = false;
}

void CudaCsrMatrix::destroyCuDSS()
{
    delete m_cuDSS;
    m_cuDSS = nullptr;
}

CuDSSBackend::Config CudaCsrMatrix::makeCuDSSConfig() const
{
    CuDSSBackend::Config cfg;
    cfg.precision = m_options.precision;
    cfg.verbose = m_options.verbose;
    cfg.hybridMemoryMode = m_options.hybridMemoryMode;
    cfg.hybridDeviceMemoryLimits = m_options.hybridDeviceMemoryLimits;
    cfg.hybridExecuteMode = m_options.hybridExecuteMode;
    cfg.multiThreadingMode = m_options.multiThreadingMode;
    cfg.threadingLibPath = m_options.threadingLibPath;
    cfg.matKind = m_options.matKind;
    cfg.useMultiGPU = m_options.useMultiGPU;
    cfg.deviceIndices = m_options.deviceIndices;
    cfg.externalStream = m_options.stream;
    cfg.syncAfterSolve = m_options.syncAfterSolve;
    return cfg;
}

int CudaCsrMatrix::ensureCusparseHandle()
{
    if (m_options.cusparseHandle != nullptr) {
        return 0;
    }
    if (m_localCusparseHandle == nullptr) {
        cuSparseCheckError(cusparseCreate(&m_localCusparseHandle), "cusparseCreate");
        m_ownsCusparseHandle = true;
    }
    if (m_options.stream != nullptr) {
        cuSparseCheckError(cusparseSetStream(m_localCusparseHandle, m_options.stream), "cusparseSetStream");
    }
    return 0;
}

cusparseHandle_t CudaCsrMatrix::getCusparseHandle() const
{
    if (m_options.cusparseHandle != nullptr) {
        return m_options.cusparseHandle;
    }
    return m_localCusparseHandle;
}

int *CudaCsrMatrix::rowPtr()
{
    if (m_ownsStructure) {
        return m_ownedRowPtr.empty() ? nullptr : raw_pointer_cast(m_ownedRowPtr.data());
    }
    return const_cast<int *>(m_rowPtr);
}

int *CudaCsrMatrix::colIdx()
{
    if (m_ownsStructure) {
        return m_ownedColIdx.empty() ? nullptr : raw_pointer_cast(m_ownedColIdx.data());
    }
    return const_cast<int *>(m_colIdx);
}

int CudaCsrMatrix::bindStructure(int numRows, int numNnz, const int *rowPtr, const int *colIdx)
{
    if (numRows <= 0 || numNnz < 0 || rowPtr == nullptr || colIdx == nullptr) {
        opserr << "ERROR CudaCsrMatrix::bindStructure() - invalid arguments\n";
        return -1;
    }
    if (m_structureBound && m_numRows == numRows && m_numNnz == numNnz && !m_ownsStructure &&
        m_rowPtr == rowPtr && m_colIdx == colIdx) {
        return 0;
    }

    destroySpMV();
    destroyCuDSS();

    m_ownedRowPtr.clear();
    m_ownedColIdx.clear();
    m_ownsStructure = false;
    m_numRows = numRows;
    m_numNnz = numNnz;
    m_rowPtr = rowPtr;
    m_colIdx = colIdx;
    m_structureBound = true;
    return 0;
}

int CudaCsrMatrix::copyStructure(int numRows, int numNnz, const int *rowPtr, const int *colIdx)
{
    if (numRows <= 0 || numNnz < 0 || rowPtr == nullptr || colIdx == nullptr) {
        opserr << "ERROR CudaCsrMatrix::copyStructure() - invalid arguments\n";
        return -1;
    }

    destroySpMV();
    destroyCuDSS();

    const std::size_t rowPtrCount = static_cast<std::size_t>(numRows) + 1u;
    const std::size_t colCount = static_cast<std::size_t>(numNnz);
    m_ownedRowPtr.resize(rowPtrCount);
    m_ownedColIdx.resize(colCount);
    cudaCheckError(cudaMemcpy(raw_pointer_cast(m_ownedRowPtr.data()), rowPtr, rowPtrCount * sizeof(int),
                              cudaMemcpyDeviceToDevice),
                   "copyStructure rowPtr");
    cudaCheckError(cudaMemcpy(raw_pointer_cast(m_ownedColIdx.data()), colIdx, colCount * sizeof(int),
                              cudaMemcpyDeviceToDevice),
                   "copyStructure colIdx");

    m_numRows = numRows;
    m_numNnz = numNnz;
    m_rowPtr = raw_pointer_cast(m_ownedRowPtr.data());
    m_colIdx = raw_pointer_cast(m_ownedColIdx.data());
    m_ownsStructure = true;
    m_structureBound = true;
    return 0;
}

int CudaCsrMatrix::bindValues(void *values)
{
    if (values == nullptr) {
        opserr << "ERROR CudaCsrMatrix::bindValues() - null values\n";
        return -1;
    }
    m_ownedValues.clear();
    m_ownsValues = false;
    m_values = values;
    if (m_spmvReady) {
        return rebuildSpMV();
    }
    return 0;
}

int CudaCsrMatrix::copyValues(const void *deviceValues)
{
    if (deviceValues == nullptr) {
        opserr << "ERROR CudaCsrMatrix::copyValues() - null source\n";
        return -1;
    }
    if (!m_structureBound) {
        opserr << "ERROR CudaCsrMatrix::copyValues() - structure not bound\n";
        return -1;
    }
    const std::size_t bytes = static_cast<std::size_t>(m_numNnz) * valueSize();
    if (m_ownedValues.size() != bytes) {
        m_ownedValues.resize(bytes);
    }
    cudaCheckError(cudaMemcpy(raw_pointer_cast(m_ownedValues.data()), deviceValues, bytes, cudaMemcpyDeviceToDevice),
                   "copyValues");
    m_values = raw_pointer_cast(m_ownedValues.data());
    m_ownsValues = true;
    if (m_spmvReady) {
        return rebuildSpMV();
    }
    return 0;
}

int CudaCsrMatrix::rebuildSpMV()
{
    destroySpMV();
    if (!m_structureBound || m_values == nullptr) {
        return -1;
    }
    if (ensureCusparseHandle() != 0) {
        return -1;
    }

    const cudaDataType_t valueType = valueCudaType();
    cusparseHandle_t handle = getCusparseHandle();
    cuSparseCheckError(cusparseCreateCsr(&m_spmat, m_numRows, m_numRows, m_numNnz, rowPtr(), colIdx(), m_values,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                         valueType),
                       "cusparseCreateCsr");

    // Placeholder pointers; actual x/y are set in spmv() via cusparseDnVecSetValues.
    cuSparseCheckError(cusparseCreateDnVec(&m_vecX, m_numRows, m_values, valueType), "cusparseCreateDnVec x");
    cuSparseCheckError(cusparseCreateDnVec(&m_vecY, m_numRows, m_values, valueType), "cusparseCreateDnVec y");

    size_t bufSize = 0;
    if (valueType == CUDA_R_32F) {
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
        cudaCheckError(cudaMalloc(&m_spmvBuffer, bufSize), "allocate SpMV buffer");
        m_spmvBufferSize = bufSize;
    }
    m_spmvReady = true;
    return 0;
}

int CudaCsrMatrix::spmv(const void *x, void *y, double alpha, double beta)
{
    if (!m_structureBound || m_values == nullptr) {
        opserr << "ERROR CudaCsrMatrix::spmv() - matrix not ready\n";
        return -1;
    }
    if (!m_spmvReady && rebuildSpMV() != 0) {
        return -1;
    }

    cusparseHandle_t handle = getCusparseHandle();
    cuSparseCheckError(cusparseDnVecSetValues(m_vecX, const_cast<void *>(x)), "cusparseDnVecSetValues x");
    cuSparseCheckError(cusparseDnVecSetValues(m_vecY, y), "cusparseDnVecSetValues y");

    if (valueCudaType() == CUDA_R_32F) {
        const float alphaF = static_cast<float>(alpha);
        const float betaF = static_cast<float>(beta);
        cuSparseCheckError(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alphaF, m_spmat, m_vecX, &betaF,
                                        m_vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, m_spmvBuffer),
                           "cusparseSpMV");
    } else {
        cuSparseCheckError(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, m_spmat, m_vecX, &beta, m_vecY,
                                        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, m_spmvBuffer),
                           "cusparseSpMV");
    }
    return 0;
}

bool CudaCsrMatrix::isFactored() const
{
    return m_cuDSS != nullptr && m_cuDSS->isFactored();
}

void *CudaCsrMatrix::getValues() { return m_values; }

const void *CudaCsrMatrix::getValues() const { return m_values; }

int CudaCsrMatrix::factorize(void *rhs, void *solution, int numRhs)
{
    if (!m_structureBound || m_values == nullptr) {
        opserr << "ERROR CudaCsrMatrix::factorize() - structure/values not bound\n";
        return -1;
    }
    if (m_cuDSS == nullptr) {
        m_cuDSS = new CuDSSBackend(makeCuDSSConfig());
    }
    if (!m_cuDSS->isStructureBound()) {
        if (m_cuDSS->bindStructure(m_numRows, m_numNnz, rowPtr(), colIdx(), m_values, rhs, solution, numRhs) != 0) {
            return -1;
        }
    }
    return m_cuDSS->factorize(m_values, rhs, solution, numRhs);
}

int CudaCsrMatrix::refactorize(void *rhs, void *solution, int numRhs)
{
    if (m_cuDSS == nullptr || !m_cuDSS->isFactored()) {
        opserr << "ERROR CudaCsrMatrix::refactorize() - not factored\n";
        return -1;
    }
    if (m_values == nullptr) {
        opserr << "ERROR CudaCsrMatrix::refactorize() - values not bound\n";
        return -1;
    }
    return m_cuDSS->refactorize(m_values, rhs, solution, numRhs);
}

int CudaCsrMatrix::solve(void *rhs, void *solution, int numRhs)
{
    if (m_cuDSS == nullptr || !m_cuDSS->isFactored()) {
        opserr << "ERROR CudaCsrMatrix::solve() - not factored\n";
        return -1;
    }
    return m_cuDSS->solve(rhs, solution, numRhs);
}
