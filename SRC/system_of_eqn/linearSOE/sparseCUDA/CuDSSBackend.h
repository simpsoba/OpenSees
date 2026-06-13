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
** Description: cuDSS backend for device-CSR direct sparse            **
** factorization/solve. Composed by CudaCsrMatrix; not an OpenSees      **
** LinearSOESolver.                                                   **
** ****************************************************************** */

#ifndef CuDSSBackend_h
#define CuDSSBackend_h

#include "CudaUtils.h"

#include <cuda_runtime.h>
#include <cudss.h>

#include <cstddef>
#include <string>
#include <vector>

enum class CuDssMatrixKind { FULL, SYMMETRIC, SPD };

class CuDSSBackend
{
public:
    struct Config {
        CudaPrecision precision;
        bool verbose;
        bool hybridMemoryMode;
        std::vector<std::size_t> hybridDeviceMemoryLimits;
        bool hybridExecuteMode;
        bool multiThreadingMode;
        std::string threadingLibPath;
        CuDssMatrixKind matKind;
        bool useMultiGPU;
        std::vector<int> deviceIndices;
        // When set, cuDSS work is enqueued on this caller-owned stream and the
        // caller is responsible for synchronization (pair with syncAfterSolve=false).
        cudaStream_t externalStream;
        // If false, solve() returns without synchronizing the stream.
        bool syncAfterSolve;

        Config()
            : precision(CudaPrecision::dDDI),
              verbose(false),
              hybridMemoryMode(false),
              hybridExecuteMode(false),
              multiThreadingMode(false),
              threadingLibPath("/usr/lib/x86_64-linux-gnu/libcudss_mtlayer_gomp.so"),
              matKind(CuDssMatrixKind::FULL),
              useMultiGPU(false),
              externalStream(nullptr),
              syncAfterSolve(true)
        {
        }

        // Caller-owned stream; caller synchronizes when syncAfterSolve is false.
        Config(CudaPrecision prec, cudaStream_t stream, bool syncAfterSolveIn)
            : Config()
        {
            precision = prec;
            externalStream = stream;
            syncAfterSolve = syncAfterSolveIn;
        }
    };

    explicit CuDSSBackend(const Config &config = Config());
    ~CuDSSBackend();

    CuDSSBackend(const CuDSSBackend &) = delete;
    CuDSSBackend &operator=(const CuDSSBackend &) = delete;

    cudaStream_t stream() const { return m_cudaStream; }

    /** Bind CSR structure (device pointers, not owned). Runs symbolic analysis. */
    int bindStructure(int numRows, int numNnz, int *rowPtr, int *colIdx, void *values, void *rhs,
                      void *solution, int numRhs = 1);

    /** Update sparse values pointer (same structure). */
    int setMatrixValues(void *values);

    /** First numeric factorization after bindStructure. */
    int factorize(void *values, void *rhs, void *solution, int numRhs = 1);

    /** Refactorize with updated coefficients. */
    int refactorize(void *values, void *rhs, void *solution, int numRhs = 1);

    /** Solve using existing factorization; updates RHS/solution pointers. */
    int solve(void *rhs, void *solution, int numRhs = 1);

    bool isStructureBound() const { return m_structureBound; }
    bool isFactored() const { return m_factored; }

private:
    Config m_config;

    void initHandle();
    void destroyHandle();
    int applyHybridMemoryLimits();
    int recreateDenseDescriptors(int numRows, void *rhs, void *solution, int numRhs);
    void destroyMatrixObjects();

    cudssHandle_t m_Handle;
    cudaStream_t m_cudaStream;
    cudssConfig_t m_Config;
    cudssData_t m_Data;
    cudssMatrix_t m_Matrix;
    cudssMatrix_t m_RHS;
    cudssMatrix_t m_Solution;

    cudaDataType_t m_ValueType;
    cudaDataType_t m_IndexType;

    int m_numRows;
    int m_numRhs;
    bool m_structureBound;
    bool m_factored;
    bool m_ownsStream;
    std::vector<int> m_deviceIndices;
};

#endif
