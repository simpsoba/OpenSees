#==============================================================================
# 
#        OpenSees -- Open System For Earthquake Engineering Simulation
#                Pacific Earthquake Engineering Research Center
#
#==============================================================================

target_sources(OPS_InterpTcl
    PRIVATE
        myCommands.cpp
)

target_sources(OPS_INTERPRETER
    PRIVATE
        Block2D.cpp
        Block3D.cpp
    PUBLIC
        Block2D.h
        Block3D.h
)

# Parallel OpenSeesPy uses OPS_INTERPRETER_PI (same interpreter .cpp as OPS_INTERPRETER);
# modelbuilder Block2D/3D provide ~Block2D/~Block3D and must be linked there too.
if(MPI_FOUND)
  target_sources(OPS_INTERPRETER_PI
      PRIVATE
          Block2D.cpp
          Block3D.cpp
      PUBLIC
          Block2D.h
          Block3D.h
  )
  target_include_directories(OPS_INTERPRETER_PI PUBLIC ${CMAKE_CURRENT_LIST_DIR})
endif()

target_include_directories(OPS_InterpTcl PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(OPS_INTERPRETER PUBLIC ${CMAKE_CURRENT_LIST_DIR})

