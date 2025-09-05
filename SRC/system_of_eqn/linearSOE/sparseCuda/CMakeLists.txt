#==============================================================================
# 
#        OpenSees -- Open System For Earthquake Engineering Simulation
#                Pacific Earthquake Engineering Research Center
#
#==============================================================================

target_sources(OPS_SysOfEqn
    PUBLIC
        CudaGenBcsrLinSOE.h
        CudaGenBcsrLinSolver.h
        CudaBcsrLinSOE.h
        AmgXLinSolver.h
)

target_include_directories(OPS_SysOfEqn PUBLIC ${CMAKE_CURRENT_LIST_DIR})