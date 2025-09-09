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
** ****************************************************************** */
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCuda/CudaBcsrLinSOE.h
                                                                        
// Written: gaaraujo 
// Created: 08/2025
//
// Description: This file contains the class definition for 
// CudaBcsrLinSOE. It stores the sparse matrix A in a fashion
// required by the CudaBcsrLinSolver object.
//

#ifndef CudaBcsrLinSOE_h
#define CudaBcsrLinSOE_h

// OpenSees includes
#include <CudaGenBcsrLinSOE.h>
#include <CudaGenBcsrLinSolver.h>

#include <memory>
#ifdef _CUDA
// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
// #include <thrust/mr/allocator.h>
// #include <thrust/mr/device_memory_resource.h>
#else
#include <vector>
#include <algorithm>
#endif

// C++ includes
#include <type_traits>

// Forward declarations
class CudaGenBcsrLinSolver;

template<typename DataType>
class CudaBcsrLinSOETag;

template<>
class CudaBcsrLinSOETag<double> {
    public:
        static constexpr int value = LinSOE_TAGS_CudaBcsrLinSOE_DOUBLE;
};

template<>
class CudaBcsrLinSOETag<float> {
    public:
        static constexpr int value = LinSOE_TAGS_CudaBcsrLinSOE_FLOAT;
};

template<typename DataType>
class CudaBcsrLinSOE : public CudaGenBcsrLinSOE
{
    public:
        // Constructor with solver
        explicit CudaBcsrLinSOE(
            CudaGenBcsrLinSolver &theSolver, 
            int blockSize = DEFAULT_BLOCK_SIZE, 
            bool paddingEnabled = true,
            bool verbose = false
        )
        : CudaGenBcsrLinSOE(CudaBcsrLinSOETag<DataType>::value, theSolver, blockSize, paddingEnabled, verbose),
          m_deviceAValues(), m_deviceX(), m_deviceB()
        {
            // Now that the derived class is fully constructed, we can safely call setLinearSOE
            theSolver.setLinearSOE(*this);
        }
        
        // Default constructor
        CudaBcsrLinSOE()
        : CudaGenBcsrLinSOE(CudaBcsrLinSOETag<DataType>::value),
          m_deviceAValues(), m_deviceX(), m_deviceB()
        {
            /* Nothing to do here */
        }

        // Destructor
        ~CudaBcsrLinSOE() = default;

        // Disable copy constructor and assignment
        CudaBcsrLinSOE(const CudaBcsrLinSOE&) = delete;
        CudaBcsrLinSOE& operator=(const CudaBcsrLinSOE&) = delete;

        // Move constructor and assignment
        CudaBcsrLinSOE(CudaBcsrLinSOE&&) = default;
        CudaBcsrLinSOE& operator=(CudaBcsrLinSOE&&) = default;

        // Required methods for CudaGenBcsrLinSOE subclasses
        const void* getDeviceAValues(void) const noexcept override { 
            #ifdef _CUDA
            return m_deviceAValues.empty() ? nullptr : m_deviceAValues.data().get(); 
            #else
            return m_deviceAValues.empty() ? nullptr : m_deviceAValues.data(); 
            #endif
        }
        
        void* getDeviceAValues(void) noexcept override { 
            #ifdef _CUDA
            return m_deviceAValues.empty() ? nullptr : m_deviceAValues.data().get(); 
            #else
            return m_deviceAValues.empty() ? nullptr : m_deviceAValues.data(); 
            #endif
        }
        
        const void* getDeviceX(void) const noexcept override { 
            #ifdef _CUDA
            return m_deviceX.empty() ? nullptr : m_deviceX.data().get(); 
            #else
            return m_deviceX.empty() ? nullptr : m_deviceX.data(); 
            #endif
        }
        
        void* getDeviceX(void) noexcept override { 
            #ifdef _CUDA
            return m_deviceX.empty() ? nullptr : m_deviceX.data().get(); 
            #else
            return m_deviceX.empty() ? nullptr : m_deviceX.data(); 
            #endif
        }
        
        const void* getDeviceB(void) const noexcept override { 
            #ifdef _CUDA
            return m_deviceB.empty() ? nullptr : m_deviceB.data().get(); 
            #else
            return m_deviceB.empty() ? nullptr : m_deviceB.data(); 
            #endif
        }
        
        void* getDeviceB(void) noexcept override { 
            #ifdef _CUDA
            return m_deviceB.empty() ? nullptr : m_deviceB.data().get(); 
            #else
            return m_deviceB.empty() ? nullptr : m_deviceB.data(); 
            #endif
        }
        
        bool isDoublePrecision(void) const noexcept override { 
            return std::is_same<DataType, double>::value; 
        }
        
        // Host-device data transfer methods
        inline void uploadVectorsToDevice(void) override {
            #ifdef _CUDA
            m_deviceB = this->CudaGenBcsrLinSOE::m_hostB;
            m_deviceX.resize(this->CudaGenBcsrLinSOE::m_hostX.size());
            #else
            m_deviceB.resize(this->CudaGenBcsrLinSOE::m_hostB.size());
            std::transform(
                this->CudaGenBcsrLinSOE::m_hostB.begin(),
                this->CudaGenBcsrLinSOE::m_hostB.end(),
                m_deviceB.begin(),
                [](double val){ return static_cast<DataType>(val); } // convert to device type
            );
            m_deviceX.resize(this->CudaGenBcsrLinSOE::m_hostX.size());
            #endif
        }
        
        inline void downloadSolutionFromDevice(void) override {
            #ifdef _CUDA
            this->CudaGenBcsrLinSOE::m_hostX = m_deviceX;
            #else
            this->CudaGenBcsrLinSOE::m_hostX.resize(m_deviceX.size());
            std::transform(
                m_deviceX.begin(),
                m_deviceX.end(),
                this->CudaGenBcsrLinSOE::m_hostX.begin(),
                [](DataType val){ return static_cast<double>(val); } // convert to host type
            );
            #endif
            this->CudaGenBcsrLinSOE::m_X.setData(
                this->CudaGenBcsrLinSOE::m_hostX.data(), 
                this->CudaGenBcsrLinSOE::m_X.Size()
            );
        }
        
        inline void uploadAValuesToDevice(void) override {
            #ifdef _CUDA
            m_deviceAValues = this->CudaGenBcsrLinSOE::m_hostAValues;
            #else
            m_deviceAValues.resize(this->CudaGenBcsrLinSOE::m_hostAValues.size());
            std::transform(
                this->CudaGenBcsrLinSOE::m_hostAValues.begin(),
                this->CudaGenBcsrLinSOE::m_hostAValues.end(),
                m_deviceAValues.begin(),
                [](double val){ return static_cast<DataType>(val); } // convert to device type
            );
            #endif
        }

    private:    
        // Device memory storage using Thrust vectors
        #ifdef _CUDA
        thrust::device_vector<DataType> m_deviceX, m_deviceB, m_deviceAValues;
        #else
        std::vector<DataType> m_deviceX, m_deviceB, m_deviceAValues;
        #endif

        
};


// Explicit template instantiations for common data types
// This ensures the compiler generates optimized code for these types
template class CudaBcsrLinSOE<double>;
template class CudaBcsrLinSOE<float>;

#endif