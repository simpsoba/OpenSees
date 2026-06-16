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

CudaCsrMatrix::CudaCsrMatrix(const SolverConfig &solver, SpmvConfig spmv, ExecutionContext exec)
    : m_solver(solver), m_spmv(spmv), m_exec(exec)
{
    if (m_spmv.sharedPattern == nullptr) {
        CuSparseBackend::Config cfg;
        cfg.precision = m_solver.precision;
        cfg.externalHandle = m_spmv.externalHandle;
        cfg.stream = m_exec.stream;
        m_spmvBackend = new CuSparseBackend(cfg);
    }
}

CudaCsrMatrix::~CudaCsrMatrix() { reset(); }

std::size_t CudaCsrMatrix::valueSize() const
{
    return (m_solver.precision == CudaPrecision::dFFI) ? sizeof(float) : sizeof(double);
}

void CudaCsrMatrix::destroySpmvBackend()
{
    delete m_spmvBackend;
    m_spmvBackend = nullptr;
}

void CudaCsrMatrix::reset()
{
    destroySpmvBackend();
    destroyCuDSS();

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

    if (m_spmv.sharedPattern == nullptr) {
        CuSparseBackend::Config cfg;
        cfg.precision = m_solver.precision;
        cfg.externalHandle = m_spmv.externalHandle;
        cfg.stream = m_exec.stream;
        m_spmvBackend = new CuSparseBackend(cfg);
    }
}

void CudaCsrMatrix::destroyCuDSS()
{
    delete m_cuDSS;
    m_cuDSS = nullptr;
}

CuDSSBackend::Config CudaCsrMatrix::makeCuDSSConfig() const
{
    CuDSSBackend::Config cfg = m_solver;
    cfg.externalStream = m_exec.stream;
    return cfg;
}

cusparseHandle_t CudaCsrMatrix::getCusparseHandle() const
{
    if (m_spmv.sharedPattern != nullptr) {
        return m_spmv.sharedPattern->getHandle();
    }
    if (m_spmvBackend != nullptr) {
        return m_spmvBackend->getHandle();
    }
    return m_spmv.externalHandle;
}

CuSparseBackend *CudaCsrMatrix::getSpmvBackend() { return m_spmvBackend; }

const CuSparseBackend *CudaCsrMatrix::getSpmvBackend() const { return m_spmvBackend; }

int CudaCsrMatrix::syncSpmvBackend() const
{
    if (m_spmvBackend == nullptr) {
        return 0;
    }
    if (!m_structureBound) {
        return -1;
    }
    if (m_spmvBackend->bindStructure(1, m_numRows, m_numNnz, rowPtr(), colIdx()) != 0) {
        return -1;
    }
    if (m_values != nullptr && m_spmvBackend->bindValues(m_values) != 0) {
        return -1;
    }
    return 0;
}

int *CudaCsrMatrix::rowPtr() const
{
    if (m_ownsStructure) {
        return m_ownedRowPtr.empty()
                   ? nullptr
                   : const_cast<int *>(raw_pointer_cast(m_ownedRowPtr.data()));
    }
    return const_cast<int *>(m_rowPtr);
}

int *CudaCsrMatrix::colIdx() const
{
    if (m_ownsStructure) {
        return m_ownedColIdx.empty()
                   ? nullptr
                   : const_cast<int *>(raw_pointer_cast(m_ownedColIdx.data()));
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

    if (m_spmvBackend != nullptr) {
        m_spmvBackend->reset();
    }
    destroyCuDSS();

    m_ownedRowPtr.clear();
    m_ownedColIdx.clear();
    m_ownsStructure = false;
    m_numRows = numRows;
    m_numNnz = numNnz;
    m_rowPtr = rowPtr;
    m_colIdx = colIdx;
    m_structureBound = true;
    return syncSpmvBackend();
}

int CudaCsrMatrix::copyStructure(int numRows, int numNnz, const int *rowPtr, const int *colIdx)
{
    if (numRows <= 0 || numNnz < 0 || rowPtr == nullptr || colIdx == nullptr) {
        opserr << "ERROR CudaCsrMatrix::copyStructure() - invalid arguments\n";
        return -1;
    }

    if (m_spmvBackend != nullptr) {
        m_spmvBackend->reset();
    }
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
    return syncSpmvBackend();
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
    if (m_spmvBackend != nullptr) {
        return m_spmvBackend->bindValues(m_values);
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
    if (m_spmvBackend != nullptr) {
        return m_spmvBackend->bindValues(m_values);
    }
    return 0;
}

int CudaCsrMatrix::spmv(const void *x, void *y, double alpha, double beta, const void *values) const
{
    const void *coeffs = (values != nullptr) ? values : m_values;
    if (m_spmv.sharedPattern != nullptr) {
        return m_spmv.sharedPattern->spmv(x, y, alpha, beta, coeffs);
    }

    if (!m_structureBound || coeffs == nullptr || m_spmvBackend == nullptr) {
        opserr << "ERROR CudaCsrMatrix::spmv() - matrix not ready\n";
        return -1;
    }
    if (syncSpmvBackend() != 0) {
        return -1;
    }
    return m_spmvBackend->spmv(x, y, alpha, beta, coeffs);
}

bool CudaCsrMatrix::isFactored() const
{
    return m_cuDSS != nullptr &&
           m_cuDSS->getPhaseStatus() >= CuDSSBackend::PhaseStatus::FactorizationComplete;
}

void *CudaCsrMatrix::getValues() { return m_values; }

const void *CudaCsrMatrix::getValues() const { return m_values; }

int CudaCsrMatrix::factorize(void *rhs, void *solution, int numRhs, const CudaCsrMatrix *symbolicSource)
{
    if (!m_structureBound || m_values == nullptr) {
        opserr << "ERROR CudaCsrMatrix::factorize() - structure/values not bound\n";
        return -1;
    }
    if (m_cuDSS == nullptr) {
        m_cuDSS = new CuDSSBackend(makeCuDSSConfig());
    }
    if (m_cuDSS->getPhaseStatus() < CuDSSBackend::PhaseStatus::PatternBound) {
        const CuDSSBackend *sharedBackend =
            (symbolicSource != nullptr) ? symbolicSource->getCuDSSBackend() : nullptr;
        if (m_cuDSS->bindStructure(m_numRows, m_numNnz, rowPtr(), colIdx(), m_values, rhs, solution, numRhs,
                                   sharedBackend) != 0) {
            return -1;
        }
    }
    return m_cuDSS->factorize(m_values, rhs, solution, numRhs);
}

CuDSSBackend *CudaCsrMatrix::getCuDSSBackend() { return m_cuDSS; }

const CuDSSBackend *CudaCsrMatrix::getCuDSSBackend() const { return m_cuDSS; }

int CudaCsrMatrix::refactorize(void *rhs, void *solution, int numRhs)
{
    if (m_cuDSS == nullptr || m_cuDSS->getPhaseStatus() < CuDSSBackend::PhaseStatus::FactorizationComplete) {
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
    if (m_cuDSS == nullptr || m_cuDSS->getPhaseStatus() < CuDSSBackend::PhaseStatus::FactorizationComplete) {
        opserr << "ERROR CudaCsrMatrix::solve() - not factored\n";
        return -1;
    }
    return m_cuDSS->solve(rhs, solution, numRhs);
}
