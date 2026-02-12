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

// DistributedCudaGenBcsrLinSOE - gather-scatter parallel SOE for OpenSeesMP.
// Protocol follows DistributedProfileSPDLinSOE / DistributedBandSPDLinSOE.
// Solver-independent: works with any CudaGenBcsrLinSolver (CuDSS, AmgX, etc.).

#ifdef _CUDA

#include <classTags.h>
#include <DistributedCudaGenBcsrLinSOE.h>
#include <CudaGenBcsrLinSOE.h>
#include <CudaGenBcsrLinSolver.h>
#include <CudaUtils.h>
#include <Matrix.h>
#include <Graph.h>
#include <Vertex.h>
#include <VertexIter.h>
#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include <OPS_Globals.h>
#include <cstring>
#include <cuda_runtime.h>
#include <thrust/memory.h>

DistributedCudaGenBcsrLinSOE::DistributedCudaGenBcsrLinSOE(CudaGenBcsrLinSolver& theSolvr, int blockSize, bool paddingEnabled, bool symmetricStorage)
    : LinearSOE(LinSOE_TAGS_DistributedCudaGenBcsrLinSOE),
      processID(0), numChannels(0), theChannels(nullptr),
      theSOE(nullptr), theSolver(&theSolvr), solverOwned(false),
      size(0), myB(nullptr), myVectB(nullptr), vectX(nullptr), vectB(nullptr),
      tripletRows(nullptr), tripletCols(nullptr), tripletVals(nullptr),
      workArea(nullptr), isAfactored(false), m_BGatheredFromChannels(false), m_cudaStream(nullptr),
      m_blockSize(blockSize), m_paddingEnabled(paddingEnabled), m_symmetricStorage(symmetricStorage)
{
    // Solver's SOE is set on rank 0 in setSize() to the real CudaGenBcsrLinSOE (theSOE).
    // On other ranks the solver's solve() is never called; only this wrapper's solve() runs (channel send/recv).
}

DistributedCudaGenBcsrLinSOE::DistributedCudaGenBcsrLinSOE()
    : LinearSOE(LinSOE_TAGS_DistributedCudaGenBcsrLinSOE),
      processID(0), numChannels(0), theChannels(nullptr),
      theSOE(nullptr), theSolver(nullptr), solverOwned(false),
      size(0), myB(nullptr), myVectB(nullptr), vectX(nullptr), vectB(nullptr),
      tripletRows(nullptr), tripletCols(nullptr), tripletVals(nullptr),
      workArea(nullptr), isAfactored(false), m_BGatheredFromChannels(false), m_cudaStream(nullptr),
      m_blockSize(1), m_paddingEnabled(false), m_symmetricStorage(false)
{
}

DistributedCudaGenBcsrLinSOE::~DistributedCudaGenBcsrLinSOE()
{
    if (theChannels != nullptr)
        delete[] theChannels;
    theChannels = nullptr;
    numChannels = 0;

    if (solverOwned && theSOE != nullptr)
        delete theSOE;
    theSOE = nullptr;
    theSolver = nullptr;

    if (myB != nullptr)
        delete[] myB;
    myB = nullptr;
    if (myVectB != nullptr)
        delete myVectB;
    myVectB = nullptr;
    if (vectX != nullptr)
        delete vectX;
    vectX = nullptr;
    if (vectB != nullptr)
        delete vectB;
    vectB = nullptr;
    if (tripletRows != nullptr)
        delete tripletRows;
    tripletRows = nullptr;
    if (tripletCols != nullptr)
        delete tripletCols;
    tripletCols = nullptr;
    if (tripletVals != nullptr)
        delete tripletVals;
    tripletVals = nullptr;
    if (workArea != nullptr)
        delete[] workArea;
    workArea = nullptr;
    if (m_cudaStream != nullptr) {
        cudaStreamDestroy(static_cast<cudaStream_t>(m_cudaStream));
        m_cudaStream = nullptr;
    }
}

int DistributedCudaGenBcsrLinSOE::setProcessID(int dTag)
{
    processID = dTag;
    return 0;
}

int DistributedCudaGenBcsrLinSOE::setChannels(int nChannels, Channel** theC)
{
    numChannels = nChannels;
    theChannels = theC;
    return 0;
}

void DistributedCudaGenBcsrLinSOE::setBlockSize(int blockSize)
{
    m_blockSize = blockSize;
}

void DistributedCudaGenBcsrLinSOE::setPaddingEnabled(bool paddingEnabled)
{
    m_paddingEnabled = paddingEnabled;
}

void DistributedCudaGenBcsrLinSOE::setSymmetricStorage(bool symmetricStorage)
{
    m_symmetricStorage = symmetricStorage;
}

int DistributedCudaGenBcsrLinSOE::getNumEqn(void) const
{
    return size;
}

int DistributedCudaGenBcsrLinSOE::setSize(Graph& theGraph)
{
    if (processID != 0) {
        Channel* theChannel = theChannels[0];
        theGraph.sendSelf(0, *theChannel);

        static ID data(1);
        theChannel->recvID(0, 0, data);
        size = data(0);

        if (size <= 0) {
            opserr << "WARNING DistributedCudaGenBcsrLinSOE::setSize() - invalid size " << size << endln;
            return -1;
        }

        if (myB != nullptr) delete[] myB;
        if (myVectB != nullptr) delete myVectB;
        if (vectX != nullptr) delete vectX;
        if (vectB != nullptr) delete vectB;
        myB = new double[size](); // value-initialize to 0.0
        myVectB = new Vector(myB, size);
        vectX = new Vector(size);
        vectB = new Vector(size);

        if (tripletRows != nullptr) delete tripletRows;
        if (tripletCols != nullptr) delete tripletCols;
        if (tripletVals != nullptr) delete tripletVals;
        tripletRows = new ID(0);
        tripletCols = new ID(0);
        tripletVals = new Vector(0);
        return 0;
    }

    // Rank 0: merge graphs from subprocesses
    FEM_ObjectBroker theBroker;
    for (int j = 0; j < numChannels; j++) {
        Channel* theChannel = theChannels[j];
        Graph theSubGraph;
        if (theSubGraph.recvSelf(0, *theChannel, theBroker) < 0) {
            opserr << "WARNING DistributedCudaGenBcsrLinSOE::setSize() - failed to recv graph from channel " << j << endln;
            return -1;
        }
        theGraph.merge(theSubGraph);
    }

    size = theGraph.getNumVertex();
    if (size <= 0) {
        opserr << "WARNING DistributedCudaGenBcsrLinSOE::setSize() - merged size " << size << endln;
        return -1;
    }

    if (theSOE != nullptr && solverOwned) {
        delete theSOE;
        theSOE = nullptr;
    }
    solverOwned = true;
    const CudaPrecision precision = theSolver->getPrecision();
    switch (precision) {
        case CudaPrecision::dDDI:
            theSOE = CudaGenBcsrLinSOE::createDouble(*theSolver, m_blockSize, m_paddingEnabled, false, m_symmetricStorage);
            break;
        case CudaPrecision::dFFI:
            theSOE = CudaGenBcsrLinSOE::createFloat(*theSolver, m_blockSize, m_paddingEnabled, false, m_symmetricStorage);
            break;
        case CudaPrecision::dDFI:
            theSOE = CudaGenBcsrLinSOE::createDoubleFloat(*theSolver, m_blockSize, m_paddingEnabled, false, m_symmetricStorage);
            break;
        case CudaPrecision::dFDI:
            theSOE = CudaGenBcsrLinSOE::createFloatDouble(*theSolver, m_blockSize, m_paddingEnabled, false, m_symmetricStorage);
            break;
        default:
            opserr << "WARNING DistributedCudaGenBcsrLinSOE::setSize() - unexpected precision, using createDouble" << endln;
            theSOE = CudaGenBcsrLinSOE::createDouble(*theSolver, m_blockSize, m_paddingEnabled, false, m_symmetricStorage);
            break;
    }
    if (theSOE == nullptr) {
        opserr << "WARNING DistributedCudaGenBcsrLinSOE::setSize() - failed to create CudaGenBcsrLinSOE" << endln;
        return -1;
    }
    theSolver->setLinearSOE(*theSOE);

    int result = theSOE->setSize(theGraph);
    if (result != 0) return result;

    if (workArea != nullptr) delete[] workArea;
    workArea = new double[size]();
    if (tripletRows != nullptr) delete tripletRows;
    if (tripletCols != nullptr) delete tripletCols;
    if (tripletVals != nullptr) delete tripletVals;
    tripletRows = new ID(0);
    tripletCols = new ID(0);
    tripletVals = new Vector(0);
    if (vectB != nullptr) delete vectB;
    if (vectX != nullptr) delete vectX;
    vectB = new Vector(size);
    vectX = new Vector(size);
    isAfactored = false;
    m_BGatheredFromChannels = false;
    // Create stream once when we have channels; reuse across setSize() calls (stream is not tied to problem size).
    if (numChannels > 0 && m_cudaStream == nullptr) {
        cudaStream_t stream = nullptr;
        (void)cudaStreamCreate(&stream);
        m_cudaStream = stream;
    }
    return 0;
}

int DistributedCudaGenBcsrLinSOE::addA(const Matrix& m, const ID& id, double fact)
{
    if (fact == 0.0) return 0;
    int idSize = id.Size();
    if (idSize != m.noRows() && idSize != m.noCols()) {
        opserr << "DistributedCudaGenBcsrLinSOE::addA() - Matrix and ID not of similar sizes" << endln;
        return -1;
    }
    if (processID == 0)
        return theSOE->addA(m, id, fact);
    for (int i = 0; i < idSize; i++)
        for (int j = 0; j < idSize; j++) {
            int row = id(i), col = id(j);
            if (row >= 0 && col >= 0 && row < size && col < size) {
                m_tripletRows.push_back(row);
                m_tripletCols.push_back(col);
                m_tripletVals.push_back(m(i, j) * fact);
            }
        }
    return 0;
}

int DistributedCudaGenBcsrLinSOE::addA(const Matrix& m)
{
    if (processID == 0)
        return theSOE->addA(m);
    opserr << "DistributedCudaGenBcsrLinSOE::addA(Matrix) - not supported on subprocess" << endln;
    return -1;
}

int DistributedCudaGenBcsrLinSOE::addB(const Vector& v, const ID& id, double fact)
{
    if (fact == 0.0) return 0;
    if (id.Size() != v.Size()) {
        opserr << "DistributedCudaGenBcsrLinSOE::addB() - Vector and ID not of similar sizes" << endln;
        return -1;
    }
    if (processID == 0) {
        m_BGatheredFromChannels = false;
        return theSOE->addB(v, id, fact);
    }
    for (int i = 0; i < id.Size(); i++) {
        int pos = id(i);
        if (pos >= 0 && pos < size)
            myB[pos] += v(i) * fact;
    }
    return 0;
}

int DistributedCudaGenBcsrLinSOE::setB(const Vector& v, double fact)
{
    if (fact == 0.0) return 0;
    if (v.Size() != size) {
        opserr << "DistributedCudaGenBcsrLinSOE::setB() - incompatible sizes" << endln;
        return -1;
    }
    if (processID == 0) {
        m_BGatheredFromChannels = false;
        return theSOE->setB(v, fact);
    }
    for (int i = 0; i < size; i++)
        myB[i] = v(i) * fact;
    return 0;
}

void DistributedCudaGenBcsrLinSOE::zeroA(void)
{
    if (processID == 0)
        theSOE->zeroA();
    else {
        m_tripletRows.clear();
        m_tripletCols.clear();
        m_tripletVals.clear();
    }
}

void DistributedCudaGenBcsrLinSOE::zeroB(void)
{
    if (processID == 0) {
        m_BGatheredFromChannels = false;
        theSOE->zeroB();
    } else if (myB != nullptr && size > 0)
        std::memset(myB, 0, static_cast<size_t>(size) * sizeof(double));
}

const Vector& DistributedCudaGenBcsrLinSOE::getX(void)
{
    if (processID == 0)
        return theSOE->getX();
    return *vectX;
}

const Vector& DistributedCudaGenBcsrLinSOE::getB(void)
{
    if (processID != 0) {
        Channel* theChannel = theChannels[0];
        theChannel->sendVector(0, 0, *myVectB);
        theChannel->recvVector(0, 0, *vectB);
        return *vectB;
    }
    if (numChannels == 0)
        return theSOE->getB();
    // Rank 0 with channels: accumulate B on device (same pattern as solve()), then sync to host and return.
    theSOE->setSyncSource(CudaGenBcsrLinSOE::SyncSource::DEVICE);
    theSOE->uploadVectorsToDevice();
    Vector remoteB(workArea, size);
    for (int j = 0; j < numChannels; j++) {
        Channel* theChannel = theChannels[j];
        theChannel->recvVector(0, 0, remoteB);
        theSOE->addToDeviceBFromHost(size, workArea,
            m_cudaStream != nullptr ? m_cudaStream : nullptr);
    }
    if (m_cudaStream != nullptr)
        cudaStreamSynchronize(static_cast<cudaStream_t>(m_cudaStream));
    theSOE->syncHostFromDevice();
    if (vectB != nullptr)
        *vectB = theSOE->getB();
    m_BGatheredFromChannels = true;
    for (int j = 0; j < numChannels; j++) {
        Channel* theChannel = theChannels[j];
        theChannel->sendVector(0, 0, *vectB);
    }
    return *vectB;
}

double DistributedCudaGenBcsrLinSOE::normRHS(void)
{
    if (processID != 0) {
        getB();
        return (vectB != nullptr) ? vectB->Norm() : 0.0;
    }
    if (numChannels == 0)
        return theSOE->normRHS();
    // Rank 0 with channels: accumulate B on device, then compute norm on GPU (no B download).
    theSOE->setSyncSource(CudaGenBcsrLinSOE::SyncSource::DEVICE);
    theSOE->uploadVectorsToDevice();
    Vector remoteB(workArea, size);
    for (int j = 0; j < numChannels; j++) {
        Channel* theChannel = theChannels[j];
        theChannel->recvVector(0, 0, remoteB);
        theSOE->addToDeviceBFromHost(size, workArea,
            m_cudaStream != nullptr ? m_cudaStream : nullptr);
    }
    if (m_cudaStream != nullptr)
        cudaStreamSynchronize(static_cast<cudaStream_t>(m_cudaStream));
    m_BGatheredFromChannels = true;
    return theSOE->normRHS();
}

void DistributedCudaGenBcsrLinSOE::setX(int loc, double value)
{
    if (processID == 0)
        theSOE->setX(loc, value);
    else if (loc >= 0 && loc < size)
        (*vectX)(loc) = value;
}

void DistributedCudaGenBcsrLinSOE::setX(const Vector& x)
{
    if (processID == 0)
        theSOE->setX(x);
    else if (x.Size() == size)
        *vectX = x;
}

int DistributedCudaGenBcsrLinSOE::solve(void)
{
    static ID result(1);

    if (processID != 0) {
        Channel* theChannel = theChannels[0];
        theChannel->sendVector(0, 0, *myVectB);

        if (!isAfactored) {
            const int nnz = static_cast<int>(m_tripletRows.size());
            if (nnz > 0) {
                tripletRows->setData(m_tripletRows.data(), nnz, false);
                tripletCols->setData(m_tripletCols.data(), nnz, false);
                tripletVals->setData(m_tripletVals.data(), nnz);
            }
            theChannel->sendID(0, 0, *tripletRows);
            theChannel->sendID(0, 0, *tripletCols);
            theChannel->sendVector(0, 0, *tripletVals);
        }
        theChannel->recvVector(0, 0, *vectX);
        theChannel->recvVector(0, 0, *vectB);
        theChannel->recvID(0, 0, result);
        isAfactored = true;
        return result(0);
    }

    // Rank 0: if no subprocesses, solve locally
    if (numChannels == 0) {
        result(0) = theSOE->solve();
        return result(0);
    }
    // Rank 0: accumulate B on device (skip recv if getB() or normRHS() just gathered).
    theSOE->setSyncSource(CudaGenBcsrLinSOE::SyncSource::DEVICE);
    if (!m_BGatheredFromChannels) {
        theSOE->uploadVectorsToDevice();
        Vector remoteB(workArea, size);
        for (int j = 0; j < numChannels; j++) {
            Channel* theChannel = theChannels[j];
            theChannel->recvVector(0, 0, remoteB);
            theSOE->addToDeviceBFromHost(size, workArea,
                m_cudaStream != nullptr ? m_cudaStream : nullptr);
        }
        if (m_cudaStream != nullptr)
            cudaStreamSynchronize(static_cast<cudaStream_t>(m_cudaStream));
    }
    m_BGatheredFromChannels = false;

    if (!isAfactored) {
        std::vector<int> cooRows, cooCols;
        std::vector<double> cooVals;
        const int soeSize = theSOE->getNumEqn();
        const int* ArowPtr = thrust::raw_pointer_cast(theSOE->m_hostCsrIndices.data());
        const int* AcolIdx = thrust::raw_pointer_cast(theSOE->m_hostCsrIndices.data()) + soeSize + 1;
        const double* Avals = thrust::raw_pointer_cast(theSOE->m_hostAValues.data());
        const int localNnz = ArowPtr[soeSize];
        cooRows.resize(static_cast<size_t>(localNnz));
        cooCols.resize(static_cast<size_t>(localNnz));
        cooVals.resize(static_cast<size_t>(localNnz));
        for (int i = 0, idx = 0; i < soeSize; i++) {
            for (int k = ArowPtr[i]; k < ArowPtr[i + 1]; k++, idx++) {
                cooRows[idx] = i;
                cooCols[idx] = AcolIdx[k];
                cooVals[idx] = Avals[k];
            }
        }
        for (int j = 0; j < numChannels; j++) {
            Channel* theChannel = theChannels[j];
            theChannel->recvID(0, 0, *tripletRows);
            theChannel->recvID(0, 0, *tripletCols);
            theChannel->recvVector(0, 0, *tripletVals);
            const int nnz = tripletRows->Size();
            if (nnz <= 0) continue;
            const int* r = &(*tripletRows)(0);
            const int* c = &(*tripletCols)(0);
            const double* v = &(*tripletVals)(0);
            cooRows.insert(cooRows.end(), r, r + nnz);
            cooCols.insert(cooCols.end(), c, c + nnz);
            cooVals.insert(cooVals.end(), v, v + nnz);
        }
        const int totalNnz = static_cast<int>(cooRows.size());
        if (theSOE->assembleAFromCOO(totalNnz, cooRows.data(), cooCols.data(), cooVals.data()) != 0) {
            opserr << "WARNING DistributedCudaGenBcsrLinSOE::solve() - assembleAFromCOO failed" << endln;
        }
    }

    result(0) = theSOE->solve();

    for (int j = 0; j < numChannels; j++) {
        Channel* theChannel = theChannels[j];
        theChannel->sendVector(0, 0, theSOE->getX());
        theChannel->sendVector(0, 0, theSOE->getB());
        theChannel->sendID(0, 0, result);
    }
    isAfactored = true;
    m_BGatheredFromChannels = false;
    return result(0);
}

int DistributedCudaGenBcsrLinSOE::sendSelf(int commitTag, Channel& theChannel)
{
    ID idData(1);
    idData(0) = processID;
    int res = theChannel.sendID(0, commitTag, idData);
    if (res < 0) {
        opserr << "WARNING DistributedCudaGenBcsrLinSOE::sendSelf() - failed to send" << endln;
        return -1;
    }
    return 0;
}

int DistributedCudaGenBcsrLinSOE::recvSelf(int commitTag, Channel& theChannel, FEM_ObjectBroker& theBroker)
{
    ID idData(1);
    int res = theChannel.recvID(0, commitTag, idData);
    if (res < 0) {
        opserr << "WARNING DistributedCudaGenBcsrLinSOE::recvSelf() - failed to recv" << endln;
        return -1;
    }
    processID = idData(0);
    numChannels = 1;
    theChannels = new Channel*[1];
    theChannels[0] = &theChannel;
    return 0;
}

#else
// Stub when _CUDA not defined (build without CUDA)
#include <DistributedCudaGenBcsrLinSOE.h>
#include <OPS_Globals.h>
DistributedCudaGenBcsrLinSOE::DistributedCudaGenBcsrLinSOE(CudaGenBcsrLinSolver& /*theSolver*/, int /*blockSize*/, bool /*paddingEnabled*/, bool /*symmetricStorage*/) : LinearSOE(0), processID(0), numChannels(0), theChannels(nullptr), theSOE(nullptr), theSolver(nullptr), solverOwned(false), size(0), myB(nullptr), myVectB(nullptr), vectX(nullptr), vectB(nullptr), tripletRows(nullptr), tripletCols(nullptr), tripletVals(nullptr), workArea(nullptr), isAfactored(false), m_BGatheredFromChannels(false), m_cudaStream(nullptr), m_blockSize(1), m_paddingEnabled(false), m_symmetricStorage(false) {}
DistributedCudaGenBcsrLinSOE::DistributedCudaGenBcsrLinSOE() : LinearSOE(0), processID(0), numChannels(0), theChannels(nullptr), theSOE(nullptr), theSolver(nullptr), solverOwned(false), size(0), myB(nullptr), myVectB(nullptr), vectX(nullptr), vectB(nullptr), tripletRows(nullptr), tripletCols(nullptr), tripletVals(nullptr), workArea(nullptr), isAfactored(false), m_BGatheredFromChannels(false), m_cudaStream(nullptr), m_blockSize(1), m_paddingEnabled(false), m_symmetricStorage(false) {}
DistributedCudaGenBcsrLinSOE::~DistributedCudaGenBcsrLinSOE() {}
int DistributedCudaGenBcsrLinSOE::setProcessID(int) { return 0; }
int DistributedCudaGenBcsrLinSOE::setChannels(int, Channel**) { return 0; }
void DistributedCudaGenBcsrLinSOE::setBlockSize(int) {}
void DistributedCudaGenBcsrLinSOE::setPaddingEnabled(bool) {}
void DistributedCudaGenBcsrLinSOE::setSymmetricStorage(bool) {}
int DistributedCudaGenBcsrLinSOE::getNumEqn(void) const { return 0; }
int DistributedCudaGenBcsrLinSOE::setSize(Graph&) { opserr << "DistributedCudaGenBcsrLinSOE requires CUDA\n"; return -1; }
int DistributedCudaGenBcsrLinSOE::addA(const Matrix&, const ID&, double) { return -1; }
int DistributedCudaGenBcsrLinSOE::addA(const Matrix&) { return -1; }
int DistributedCudaGenBcsrLinSOE::addB(const Vector&, const ID&, double) { return -1; }
int DistributedCudaGenBcsrLinSOE::setB(const Vector&, double) { return -1; }
void DistributedCudaGenBcsrLinSOE::zeroA(void) {}
void DistributedCudaGenBcsrLinSOE::zeroB(void) {}
const Vector& DistributedCudaGenBcsrLinSOE::getX(void) { static Vector v; return v; }
const Vector& DistributedCudaGenBcsrLinSOE::getB(void) { static Vector v; return v; }
double DistributedCudaGenBcsrLinSOE::normRHS(void) { return 0.0; }
void DistributedCudaGenBcsrLinSOE::setX(int, double) {}
void DistributedCudaGenBcsrLinSOE::setX(const Vector&) {}
int DistributedCudaGenBcsrLinSOE::solve(void) { return -1; }
int DistributedCudaGenBcsrLinSOE::sendSelf(int, Channel&) { return -1; }
int DistributedCudaGenBcsrLinSOE::recvSelf(int, Channel&, FEM_ObjectBroker&) { return -1; }
#endif
