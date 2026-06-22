#==============================================================================
# 
#        OpenSees -- Open System For Earthquake Engineering Simulation
#                Pacific Earthquake Engineering Research Center
#
# GPU sparse SOE layout (see root CMakeLists.txt):
#   OPS_Cuda          - CudaBcsrLinSOE / CudaBcsrLinSolver (base SOE framework)
#   OPS_Cuda_AmgX     - AmgXLinSolver (optional, when AMGX is found)
#   OPS_Cuda_CuDSS    - CuDSSBackend, CudaDirectSparseSolver, CudaExplicitAlpha (optional, when cuDSS is found)
#   OPS_Cuda_Parallel - distributed assembly SOEs for OpenSeesSP/MP (future)
#
# Headers and include directory belong to OPS_Cuda so that GPU types are only
# visible to targets that explicitly link OPS_Cuda, not to all of OPS_SysOfEqn.
#==============================================================================

if(TARGET OPS_Cuda)
  target_sources(OPS_Cuda
    PUBLIC
      CudaBcsrLinSOE.h
      CudaBcsrLinSolver.h
      CuSparseBackend.h
  )

  target_include_directories(OPS_Cuda PUBLIC ${CMAKE_CURRENT_LIST_DIR})
endif()

if(TARGET OPS_Cuda_AmgX)
  target_sources(OPS_Cuda_AmgX
    PUBLIC
      AmgXLinSolver.h
  )
endif()

if(TARGET OPS_Cuda_CuDSS)
  target_sources(OPS_Cuda_CuDSS
    PUBLIC
      CudaCsrMatrix.h
      CudaDirectSparseSolver.h
      CuDSSBackend.h
  )
endif()
