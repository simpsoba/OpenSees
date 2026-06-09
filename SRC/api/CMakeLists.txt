#==============================================================================
# 
#        OpenSees -- Open System For Earthquake Engineering Simulation
#                Pacific Earthquake Engineering Research Center
#
#==============================================================================
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  add_library(OPS_Api OBJECT)
  target_sources(OPS_Api PRIVATE win32Functions.cpp)
  target_include_directories(OPS_Api PUBLIC ${CMAKE_CURRENT_LIST_DIR})
else()
  # Header-only on Unix; OBJECT needs at least one source file.
  add_library(OPS_Api INTERFACE)
  target_include_directories(OPS_Api INTERFACE ${CMAKE_CURRENT_LIST_DIR})
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # Headers-only object library: no TU on Linux/macOS without explicit linker language.
  set_target_properties(OPS_Api PROPERTIES LINKER_LANGUAGE CXX)
endif()

target_sources(OPS_InterpTcl
  PRIVATE
    elementAPI_TCL.cpp
    packages.cpp
)

target_sources(OPS_Api
  INTERFACE
    elementAPI.h
    GenericDict.h
)

