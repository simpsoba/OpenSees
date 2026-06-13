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
#include "CudaUtils.h"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_vector.h>

#include <cstddef>
#include <string>
#include <vector>

class CudaCsrMatrix
{
public:
    struct Options {
        CudaPrecision precision = CudaPrecision::dDDI;
        cudaStream_t stream = nullptr;
        // When null, a private handle is created lazily on first spmv().
        cusparseHandle_t cusparseHandle = nullptr;
        bool syncAfterSolve = true;
        CuDssMatrixKind matKind = CuDssMatrixKind::FULL;
        bool verbose = false;
        bool hybridMemoryMode = false;
        std::vector<std::size_t> hybridDeviceMemoryLimits;
        bool hybridExecuteMode = false;
        bool multiThreadingMode = false;
        std::string threadingLibPath = "/usr/lib/x86_64-linux-gnu/libcudss_mtlayer_gomp.so";
        bool useMultiGPU = false;
        std::vector<int> deviceIndices;
    };

    CudaCsrMatrix();
    explicit CudaCsrMatrix(const Options &options);
    ~CudaCsrMatrix();

    CudaCsrMatrix(const CudaCsrMatrix &) = delete;
    CudaCsrMatrix &operator=(const CudaCsrMatrix &) = delete;

    void reset();

    // --- Structure: bind = borrow device pointers; copy = own a device copy ---
    int bindStructure(int numRows, int numNnz, const int *rowPtr, const int *colIdx);
    int copyStructure(int numRows, int numNnz, const int *rowPtr, const int *colIdx);

    // --- Values: bind = borrow; copy = own (D2D into internal buffer) ---
    int bindValues(void *values);
    int copyValues(const void *deviceValues);

    // y = alpha * A * x + beta * y  (lazy cuSPARSE SpMV setup)
    int spmv(const void *x, void *y, double alpha = 1.0, double beta = 0.0);

    // Direct solve (lazy cuDSS setup); values must be bound before factorize/refactorize.
    int factorize(void *rhs, void *solution, int numRhs = 1);
    int refactorize(void *rhs, void *solution, int numRhs = 1);
    int solve(void *rhs, void *solution, int numRhs = 1);

    bool isStructureBound() const { return m_structureBound; }
    bool isFactored() const;
    bool ownsValues() const { return m_ownsValues; }
    bool ownsStructure() const { return m_ownsStructure; }

    int getNumRows() const { return m_numRows; }
    int getNumNnz() const { return m_numNnz; }
    void *getValues();
    const void *getValues() const;

    cudaStream_t getStream() const { return m_options.stream; }
    cusparseHandle_t getCusparseHandle() const;

private:
    Options m_options;

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

    cusparseHandle_t m_localCusparseHandle = nullptr;
    bool m_ownsCusparseHandle = false;

    cusparseSpMatDescr_t m_spmat = nullptr;
    cusparseDnVecDescr_t m_vecX = nullptr;
    cusparseDnVecDescr_t m_vecY = nullptr;
    void *m_spmvBuffer = nullptr;
    size_t m_spmvBufferSize = 0;
    bool m_spmvReady = false;

    CuDSSBackend *m_cuDSS = nullptr;

    void destroySpMV();
    void destroyCuDSS();
    int ensureCusparseHandle();
    int rebuildSpMV();
    CuDSSBackend::Config makeCuDSSConfig() const;
    int *rowPtr();
    int *colIdx();
    std::size_t valueSize() const;
    cudaDataType_t valueCudaType() const;
};

#endif
