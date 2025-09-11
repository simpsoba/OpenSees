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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCuda/CudaGenBcsrLinSOEImpl.h
                                                                        
// Written: gaaraujo 
// Created: 08/2025
//
// Description: This file contains the internal template implementation for 
// CudaGenBcsrLinSOE. This is an internal implementation detail and should
// not be included by users of the CudaGenBcsrLinSOE interface.
//

#ifndef CudaGenBcsrLinSOEImpl_h
#define CudaGenBcsrLinSOEImpl_h

// OpenSees includes
#include <CudaGenBcsrLinSOE.h>
#include <CudaGenBcsrLinSolver.h>
#ifdef _CUDA
// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#else
#include <vector>
#include <algorithm>
#endif

// C++ includes
#include <type_traits>

// Forward declarations
class CudaGenBcsrLinSolver;

// This template class provides the actual implementation for different data types.
// It inherits from CudaGenBcsrLinSOE and implements all the pure virtual methods.
// The template parameter DataType allows us to specialize for double and float without code duplication.
template<typename DataType>
class CudaGenBcsrLinSOEImpl : public CudaGenBcsrLinSOE
{
public:
    // Constructor with solver
    explicit CudaGenBcsrLinSOEImpl(
        CudaGenBcsrLinSolver &theSolver, 
        int blockSize = DEFAULT_BLOCK_SIZE, 
        bool paddingEnabled = true,
        bool verbose = false
    )
    : CudaGenBcsrLinSOE(CudaGenBcsrLinSOEImpl<DataType>::getClassTagForType(), theSolver, blockSize, paddingEnabled, verbose),
      m_deviceAValues(), m_deviceX(), m_deviceB()
    {
        // Now that the derived class is fully constructed, we can safely call setLinearSOE
        theSolver.setLinearSOE(*this);
    }
    
    // Default constructor
    CudaGenBcsrLinSOEImpl()
    : CudaGenBcsrLinSOE(CudaGenBcsrLinSOEImpl<DataType>::getClassTagForType()),
      m_deviceAValues(), m_deviceX(), m_deviceB()
    {
        /* Nothing to do here */
    }

    // Destructor
    ~CudaGenBcsrLinSOEImpl() = default;

    // Disable copy constructor and assignment
    CudaGenBcsrLinSOEImpl(const CudaGenBcsrLinSOEImpl&) = delete;
    CudaGenBcsrLinSOEImpl& operator=(const CudaGenBcsrLinSOEImpl&) = delete;

    // Move constructor and assignment
    CudaGenBcsrLinSOEImpl(CudaGenBcsrLinSOEImpl&&) = default;
    CudaGenBcsrLinSOEImpl& operator=(CudaGenBcsrLinSOEImpl&&) = default;

    // Required methods for CudaGenBcsrLinSOE subclasses
    // These methods provide type-erased access to device data.
    // The void* return type allows the solver to work with any data type without knowing the specifics.
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
    
    // This method allows the solver to know the data type of the SOE at runtime.
    bool isDoublePrecision(void) const noexcept override { 
        return std::is_same<DataType, double>::value; 
    }
    
    // Host (double)-device (DataType) data transfer methods
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

public:
    // Helper function to get class tag for type
    static int getClassTagForType() {
        if constexpr (std::is_same_v<DataType, double>) {
            return LinSOE_TAGS_CudaBcsrLinSOE_DOUBLE;
        } else if constexpr (std::is_same_v<DataType, float>) {
            return LinSOE_TAGS_CudaBcsrLinSOE_FLOAT;
        } else {
            static_assert(std::is_same_v<DataType, double> || std::is_same_v<DataType, float>, 
                         "Only double and float types are supported");
            return -1; // This should never be reached due to static_assert
        }
    }

private:
};

// Explicit template instantiations for common data types
// This ensures the compiler generates optimized code for these types
template class CudaGenBcsrLinSOEImpl<double>;
template class CudaGenBcsrLinSOEImpl<float>;

#endif
