/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
** ****************************************************************** */

// Written: gaaraujo
// Created: 07/2026
//
// DistributedCudaBcsrLinSOE: SparseGen-style gather-to-root around CudaBcsrLinSOE.

#include <DistributedCudaBcsrLinSOE.h>
#include <CudaBcsrLinSOE.h>
#include <CudaBcsrLinSolver.h>
#include <Matrix.h>
#include <Graph.h>
#include <Vertex.h>
#include <VertexIter.h>
#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include <ID.h>
#include <classTags.h>
#include <OPS_Globals.h>

#include <algorithm>
#include <cstring>

DistributedCudaBcsrLinSOE::DistributedCudaBcsrLinSOE(CudaBcsrLinSOE *theSOE)
    : LinearSOE(LinSOE_TAGS_DistributedCudaBcsrLinSOE),
      theCudaSOE(theSOE),
      processID(0), numChannels(0), theChannels(0),
      workArea(0), sizeWork(0),
      myB(0), myVectB(0), myBsize(0)
{
    if (theCudaSOE == nullptr) {
        opserr << "FATAL DistributedCudaBcsrLinSOE - null CudaBcsrLinSOE\n";
    }
}

DistributedCudaBcsrLinSOE::DistributedCudaBcsrLinSOE()
    : LinearSOE(LinSOE_TAGS_DistributedCudaBcsrLinSOE),
      theCudaSOE(0),
      processID(0), numChannels(0), theChannels(0),
      workArea(0), sizeWork(0),
      myB(0), myVectB(0), myBsize(0)
{
}

DistributedCudaBcsrLinSOE::~DistributedCudaBcsrLinSOE()
{
    if (theChannels != 0)
        delete[] theChannels;

    if (workArea != 0)
        delete[] workArea;

    if (myB != 0)
        delete[] myB;

    if (myVectB != 0)
        delete myVectB;

    delete theCudaSOE;
}

bool
DistributedCudaBcsrLinSOE::needsMatrixTransfer(void) const
{
    if (theCudaSOE == nullptr)
        return true;
    return theCudaSOE->getMatrixStatus() != CudaBcsrLinSOE::MatrixStatus::UNCHANGED;
}

void
DistributedCudaBcsrLinSOE::ensureMyBSize(int size)
{
    if (size <= myBsize && myB != 0 && myVectB != 0)
        return;

    if (myB != 0)
        delete[] myB;
    if (myVectB != 0)
        delete myVectB;

    myB = new double[size];
    myVectB = new Vector(myB, size);
    myBsize = size;
    for (int i = 0; i < size; ++i)
        myB[i] = 0.0;
}

void
DistributedCudaBcsrLinSOE::ensureWorkArea(int size)
{
    if (size <= sizeWork && workArea != 0)
        return;

    if (workArea != 0)
        delete[] workArea;
    workArea = new double[size];
    sizeWork = size;
}

int
DistributedCudaBcsrLinSOE::getNumEqn(void) const
{
    return theCudaSOE != nullptr ? theCudaSOE->getNumEqn() : 0;
}

int
DistributedCudaBcsrLinSOE::setSize(Graph &theGraph)
{
    if (theCudaSOE == nullptr) {
        opserr << "WARNING DistributedCudaBcsrLinSOE::setSize() - no CudaBcsrLinSOE\n";
        return -1;
    }

    int result = 0;

    // Worker: send local graph, receive global CSR structure
    if (processID != 0) {
        if (numChannels < 1 || theChannels == 0 || theChannels[0] == 0) {
            opserr << "WARNING DistributedCudaBcsrLinSOE::setSize() - worker has no channel\n";
            return -1;
        }

        Channel *theChannel = theChannels[0];
        theGraph.sendSelf(0, *theChannel);

        static ID data(6);
        theChannel->recvID(0, 0, data);
        const int numEqn = data(0);
        const int numValues = data(1);
        const int numIndices = data(2);
        const int blockSize = data(3);
        const int paddedSize = data(4);
        result = data(5);

        if (result < 0) {
            opserr << "WARNING DistributedCudaBcsrLinSOE::setSize() - rank 0 reported failure\n";
            return result;
        }

        ID indexData(numIndices);
        theChannel->recvID(0, 0, indexData);

        result = theCudaSOE->installHostStructure(
            numEqn, blockSize, paddedSize,
            &indexData(0), numIndices, numValues,
            /*invokeSolverSetSize=*/false);
        if (result < 0)
            return result;

        ensureMyBSize(numEqn);
        return 0;
    }

    // Rank 0: merge remote graphs, build structure, broadcast to workers
    FEM_ObjectBroker theBroker;
    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        Graph theSubGraph;
        theSubGraph.recvSelf(0, *theChannel, theBroker);
        theGraph.merge(theSubGraph);
    }

    result = theCudaSOE->setSize(theGraph);

    const int numEqn = theCudaSOE->getNumEqn();
    const int numValues = theCudaSOE->getNumNonZeroValues();
    const int numIndices = theCudaSOE->getNumCsrIndices();
    const int blockSize = theCudaSOE->getBlockSize();
    const int paddedSize = theCudaSOE->getPaddedVectorSize();

    static ID data(6);
    data(0) = numEqn;
    data(1) = numValues;
    data(2) = numIndices;
    data(3) = blockSize;
    data(4) = paddedSize;
    data(5) = result;

    ID indexData(numIndices);
    if (numIndices > 0 && theCudaSOE->getHostCsrIndicesData() != nullptr) {
        std::copy_n(theCudaSOE->getHostCsrIndicesData(), numIndices, &indexData(0));
    }

    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        theChannel->sendID(0, 0, data);
        if (result >= 0)
            theChannel->sendID(0, 0, indexData);
    }

    if (result < 0)
        return result;

    ensureMyBSize(numEqn);
    ensureWorkArea(std::max(numValues, numEqn));
    return 0;
}

int
DistributedCudaBcsrLinSOE::addA(const Matrix &m, const ID &id, double fact)
{
    if (theCudaSOE == nullptr)
        return -1;
    return theCudaSOE->addA(m, id, fact);
}

int
DistributedCudaBcsrLinSOE::addA(const Matrix &m)
{
    if (theCudaSOE == nullptr)
        return -1;
    return theCudaSOE->addA(m);
}

int
DistributedCudaBcsrLinSOE::addB(const Vector &v, const ID &id, double fact)
{
    if (fact == 0.0)
        return 0;

    const int idSize = id.Size();
    if (idSize != v.Size()) {
        opserr << "WARNING DistributedCudaBcsrLinSOE::addB() - Vector and ID size mismatch\n";
        return -1;
    }

    const int size = myBsize;
    if (fact == 1.0) {
        for (int i = 0; i < idSize; ++i) {
            const int pos = id(i);
            if (pos >= 0 && pos < size)
                myB[pos] += v(i);
        }
    } else if (fact == -1.0) {
        for (int i = 0; i < idSize; ++i) {
            const int pos = id(i);
            if (pos >= 0 && pos < size)
                myB[pos] -= v(i);
        }
    } else {
        for (int i = 0; i < idSize; ++i) {
            const int pos = id(i);
            if (pos >= 0 && pos < size)
                myB[pos] += v(i) * fact;
        }
    }
    return 0;
}

int
DistributedCudaBcsrLinSOE::setB(const Vector &v, double fact)
{
    if (fact == 0.0)
        return 0;

    if (v.Size() != myBsize) {
        // Gather-to-root SOEs keep a *global* RHS after graph merge. Callers that
        // pass AnalysisModel/Arpack local size (processID unset, ParallelNumberer
        // without setNumEqn) hit this path — see ArpackSOE::setProcessID wiring.
        opserr << "WARNING DistributedCudaBcsrLinSOE::setB() - incompatible sizes "
               << "SOE(global)=" << myBsize << " vector=" << v.Size()
               << " (expected global eqn count; check ParallelNumberer setNumEqn "
               << "and ArpackSOE setProcessID for OpenSeesMP)\n";
        return -1;
    }

    if (fact == 1.0) {
        for (int i = 0; i < myBsize; ++i)
            myB[i] = v(i);
    } else if (fact == -1.0) {
        for (int i = 0; i < myBsize; ++i)
            myB[i] = -v(i);
    } else {
        for (int i = 0; i < myBsize; ++i)
            myB[i] = v(i) * fact;
    }
    return 0;
}

void
DistributedCudaBcsrLinSOE::zeroA(void)
{
    if (theCudaSOE != nullptr)
        theCudaSOE->zeroA();
}

void
DistributedCudaBcsrLinSOE::zeroB(void)
{
    for (int i = 0; i < myBsize; ++i)
        myB[i] = 0.0;
}

const Vector &
DistributedCudaBcsrLinSOE::getX(void)
{
    return theCudaSOE->getX();
}

const Vector &
DistributedCudaBcsrLinSOE::getB(void)
{
    Vector &vectB = theCudaSOE->getHostBVector();

    if (processID != 0) {
        Channel *theChannel = theChannels[0];
        theChannel->sendVector(0, 0, *myVectB);
        theChannel->recvVector(0, 0, vectB);
        theCudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
        return vectB;
    }

    vectB = *myVectB;
    ensureWorkArea(myBsize);
    Vector remoteB(workArea, myBsize);
    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        theChannel->recvVector(0, 0, remoteB);
        vectB += remoteB;
    }
    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        theChannel->sendVector(0, 0, vectB);
    }
    theCudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
    return vectB;
}

double
DistributedCudaBcsrLinSOE::normRHS(void)
{
    return this->getB().Norm();
}

void
DistributedCudaBcsrLinSOE::setX(int loc, double value)
{
    if (theCudaSOE != nullptr)
        theCudaSOE->setX(loc, value);
}

void
DistributedCudaBcsrLinSOE::setX(const Vector &x)
{
    if (theCudaSOE != nullptr)
        theCudaSOE->setX(x);
}

int
DistributedCudaBcsrLinSOE::solve(void)
{
    static ID result(1);

    if (theCudaSOE == nullptr) {
        opserr << "WARNING DistributedCudaBcsrLinSOE::solve() - no CudaBcsrLinSOE\n";
        return -1;
    }

    const bool sendA = needsMatrixTransfer();
    const int numValues = theCudaSOE->getNumNonZeroValues();
    Vector &vectX = theCudaSOE->getHostXVector();
    Vector &vectB = theCudaSOE->getHostBVector();

    if (processID != 0) {
        Channel *theChannel = theChannels[0];
        theChannel->sendVector(0, 0, *myVectB);

        if (sendA) {
            Vector vectA(theCudaSOE->getHostAValuesData(), numValues);
            theChannel->sendVector(0, 0, vectA);
        }

        theChannel->recvVector(0, 0, vectX);
        theChannel->recvVector(0, 0, vectB);
        theChannel->recvID(0, 0, result);

        theCudaSOE->setXPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
        theCudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
        if (result(0) == 0)
            theCudaSOE->setMatrixStatus(CudaBcsrLinSOE::MatrixStatus::UNCHANGED);

        return result(0);
    }

    // Rank 0: merge RHS and (if needed) A, then GPU solve
    vectB = *myVectB;
    theCudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);

    ensureWorkArea(std::max(numValues, myBsize));
    Vector remoteB(myBsize);

    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        theChannel->recvVector(0, 0, remoteB);
        vectB += remoteB;

        if (sendA) {
            Vector remoteA(numValues);
            theChannel->recvVector(0, 0, remoteA);
            double *A = theCudaSOE->getHostAValuesData();
            for (int i = 0; i < numValues; ++i)
                A[i] += remoteA(i);
            theCudaSOE->setAValuesPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
        }
    }

    result(0) = theCudaSOE->solve();

    // Ensure host X is current before broadcast
    (void)theCudaSOE->getX();

    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        theChannel->sendVector(0, 0, vectX);
        theChannel->sendVector(0, 0, vectB);
        theChannel->sendID(0, 0, result);
    }

    return result(0);
}

int
DistributedCudaBcsrLinSOE::formAp(const Vector &p, Vector &Ap)
{
    // Match DistributedProfileSPD / MumpsParallelSOE: no distributed SpMV yet.
    // A correct gather-to-root formAp (merge A on rank 0, SpMV, broadcast Ap)
    // can be added later; static/transient analyze does not need it.
    (void)p;
    (void)Ap;
    opserr << "WARNING DistributedCudaBcsrLinSOE::formAp() - not supported "
           << "(gather-to-root SpMV not implemented); returning -1\n";
    return -1;
}

LinearSOE *
DistributedCudaBcsrLinSOE::getCopy(void) const
{
    if (theCudaSOE == nullptr)
        return nullptr;

    LinearSOE *cudaCopy = theCudaSOE->getCopy();
    if (cudaCopy == nullptr)
        return nullptr;

    CudaBcsrLinSOE *cudaSOE = dynamic_cast<CudaBcsrLinSOE *>(cudaCopy);
    if (cudaSOE == nullptr) {
        delete cudaCopy;
        return nullptr;
    }

    DistributedCudaBcsrLinSOE *out = new DistributedCudaBcsrLinSOE(cudaSOE);
    out->setProcessID(processID);
    out->setChannels(numChannels, theChannels);
    return out;
}

int
DistributedCudaBcsrLinSOE::saveSparseA(OPS_Stream &output, int baseIndex)
{
    if (theCudaSOE == nullptr)
        return -1;
    return theCudaSOE->saveSparseA(output, baseIndex);
}

int
DistributedCudaBcsrLinSOE::sendSelf(int commitTag, Channel &theChannel)
{
    int sendID = 0;

    if (processID == 0) {
        bool found = false;
        for (int i = 0; i < numChannels; ++i) {
            if (theChannels[i] == &theChannel) {
                sendID = i + 1;
                found = true;
            }
        }

        if (!found) {
            int nextNumChannels = numChannels + 1;
            Channel **nextChannels = new Channel *[nextNumChannels];
            for (int i = 0; i < numChannels; ++i)
                nextChannels[i] = theChannels[i];
            nextChannels[numChannels] = &theChannel;

            if (theChannels != 0)
                delete[] theChannels;
            theChannels = nextChannels;
            numChannels = nextNumChannels;
            sendID = numChannels;
        }
    } else {
        sendID = processID;
    }

    ID idData(1);
    idData(0) = sendID;
    int res = theChannel.sendID(0, commitTag, idData);
    if (res < 0) {
        opserr << "WARNING DistributedCudaBcsrLinSOE::sendSelf() - failed to send data\n";
        return -1;
    }
    return 0;
}

int
DistributedCudaBcsrLinSOE::recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    ID idData(1);
    int res = theChannel.recvID(0, commitTag, idData);
    if (res < 0) {
        opserr << "WARNING DistributedCudaBcsrLinSOE::recvSelf() - failed to recv data\n";
        return -1;
    }
    processID = idData(0);

    numChannels = 1;
    if (theChannels != 0)
        delete[] theChannels;
    theChannels = new Channel *[1];
    theChannels[0] = &theChannel;

    // Broker path does not reconstruct the CUDA SOE/solver; OpenSeesMP
    // constructs DistributedCudaBcsrLinSOE via the Distributed* commands.
    (void)theBroker;
    return 0;
}

int
DistributedCudaBcsrLinSOE::setProcessID(int dTag)
{
    processID = dTag;
    return 0;
}

int
DistributedCudaBcsrLinSOE::setChannels(int nChannels, Channel **theC)
{
    numChannels = nChannels;

    if (theChannels != 0)
        delete[] theChannels;

    if (numChannels > 0 && theC != 0) {
        theChannels = new Channel *[numChannels];
        for (int i = 0; i < numChannels; ++i)
            theChannels[i] = theC[i];
    } else {
        theChannels = 0;
        numChannels = 0;
    }
    return 0;
}
