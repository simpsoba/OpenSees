#==============================================================================
# 
#        OpenSees -- Open System For Earthquake Engineering Simulation
#                Pacific Earthquake Engineering Research Center
#
#==============================================================================

target_sources(OPS_SysOfEqn
    PRIVATE
        CudaGenBcsrLinSOE.cpp
        CudaGenBcsrLinSolver.cpp
        AmgXLinSolver.cpp
    PUBLIC
        CudaGenBcsrLinSOE.h
        CudaGenBcsrLinSolver.h
        CudaBcsrLinSOE.h
        AmgXLinSolver.h
)

target_include_directories(OPS_SysOfEqn PUBLIC ${CMAKE_CURRENT_LIST_DIR})

if(CUDAToolkit_FOUND)
    # CUDA available - force NVCC compilation for sparseCuda files
    set_source_files_properties(
        CudaGenBcsrLinSOE.cpp
        CudaGenBcsrLinSolver.cpp
        AmgXLinSolver.cpp
        PROPERTIES
        LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    # Add compile definitions
    if(DEFINED AMGX_NO_MPI_DIR)
        # AMGX also available
        add_compile_definitions(_CUDA _AMGX)
        message(STATUS "CUDA and AMGX available - compiling full sparseCuda implementation with NVCC")
    else()
        # CUDA available but no AMGX
        add_compile_definitions(_CUDA)
        message(STATUS "CUDA available but AMGX not available - using sparseCuda with dummy AMGX (compiled with NVCC)")
    endif()
else()
    # No CUDA available - files will use dummy implementations (compiled with GCC)
    add_compile_definitions(_NO_CUDA _NO_AMGX)
    message(STATUS "CUDA not available - sparseCuda files will use dummy implementations (compiled with GCC)")
endif()