/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
** (C) Copyright 1999, The Regents of the University of California   **
** All Rights Reserved.                                               **
**                                                                    **
** Commercial use of this program without express permission of the    **
** University of California, Berkeley, is strictly prohibited.       **
** ****************************************************************** */

// Written: OpenSees
// Description: DistributedCudaGenBcsrLinSOE - gather-scatter parallel SOE
// for OpenSeesMP: subprocesses send A and B to rank 0; rank 0 assembles,
// solves with CudaGenBcsrLinSOE + CudaGenBcsrLinSolver (e.g. CuDSS), sends X back.

#ifndef DistributedCudaGenBcsrLinSOE_h
#define DistributedCudaGenBcsrLinSOE_h

#include <LinearSOE.h>
#include <Vector.h>
#include <ID.h>
#include <vector>

class Channel;
class FEM_ObjectBroker;
class Graph;
class Matrix;
class CudaGenBcsrLinSolver;
class CudaGenBcsrLinSOE;

class DistributedCudaGenBcsrLinSOE : public LinearSOE
{
public:
    // blockSize, paddingEnabled, symmetricStorage: used when creating the inner CudaGenBcsrLinSOE on rank 0.
    // CuDSS only supports blockSize 1; AmgX and others support blockSize > 1.
    // symmetricStorage: lower-triangle-only storage (same as CudaGenBcsrLinSOE / -matrixType symmetric|spd).
    DistributedCudaGenBcsrLinSOE(CudaGenBcsrLinSolver& theSolver, int blockSize = 1, bool paddingEnabled = false, bool symmetricStorage = false);
    DistributedCudaGenBcsrLinSOE();  // for broker recvSelf
    ~DistributedCudaGenBcsrLinSOE();

    int setProcessID(int processTag);
    int setChannels(int numChannels, Channel** theChannels);

    void setBlockSize(int blockSize);
    void setPaddingEnabled(bool paddingEnabled);
    void setSymmetricStorage(bool symmetricStorage);
    int getBlockSize() const { return m_blockSize; }
    bool getPaddingEnabled() const { return m_paddingEnabled; }
    bool getSymmetricStorage() const { return m_symmetricStorage; }

    int getNumEqn(void) const override;
    int setSize(Graph& theGraph) override;
    int addA(const Matrix& m, const ID& id, double fact = 1.0) override;
    int addA(const Matrix& m) override;
    int addB(const Vector& v, const ID& id, double fact = 1.0) override;
    int setB(const Vector& v, double fact = 1.0) override;
    void zeroA(void) override;
    void zeroB(void) override;
    const Vector& getX(void) override;
    const Vector& getB(void) override;
    double normRHS(void) override;
    void setX(int loc, double value) override;
    void setX(const Vector& x) override;
    int solve(void) override;

    int sendSelf(int commitTag, Channel& theChannel) override;
    int recvSelf(int commitTag, Channel& theChannel, FEM_ObjectBroker& theBroker) override;

private:
    int processID;
    int numChannels;
    Channel** theChannels;

    // Rank 0 only: the actual SOE and solver
    CudaGenBcsrLinSOE* theSOE;
    CudaGenBcsrLinSolver* theSolver;
    bool solverOwned;  // true if we created theSOE (rank 0)

    // Subprocess: local B and triplets for A (global indices)
    int size;           // global size (same on all ranks)
    double* myB;
    Vector* myVectB;
    Vector* vectX;
    Vector* vectB;

    // Triplet storage for subprocess A contributions (row, col, value).
    // Storage in vectors (push_back/clear); ID/Vector used only as wrappers for Channel send.
    std::vector<int> m_tripletRows;
    std::vector<int> m_tripletCols;
    std::vector<double> m_tripletVals;
    ID* tripletRows;
    ID* tripletCols;
    Vector* tripletVals;

    // Rank 0: receive buffer for remote B (length size)
    double* workArea;  // used for Vector remoteB(workArea, size)
    bool isAfactored;

    // Rank 0 with channels: if true, getB() or normRHS() just gathered B into device; solve() can skip B recv once. Cleared after solve() and on addB/setB/zeroB.
    bool m_BGatheredFromChannels;

    // Rank 0 with channels: persistent CUDA stream for B accumulation (avoids create/destroy per solve/getB/normRHS). void* to avoid CUDA in header; cast to cudaStream_t in .cu.
    void* m_cudaStream;

    // Block size, padding, and symmetric storage for the inner CudaGenBcsrLinSOE (rank 0). Defaults match CuDSS (blockSize 1, full storage).
    int m_blockSize;
    bool m_paddingEnabled;
    bool m_symmetricStorage;
};

#endif
