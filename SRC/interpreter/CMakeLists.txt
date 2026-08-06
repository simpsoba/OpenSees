set(
  OPS_INTERPRETER_CPP
  DL_Interpreter.cpp
  OpenSeesCommands.cpp
  OpenSeesUniaxialMaterialCommands.cpp
  OpenSeesElementCommands.cpp
  OpenSeesTimeSeriesCommands.cpp
  OpenSeesPatternCommands.cpp
  OpenSeesSectionCommands.cpp
  OpenSeesOutputCommands.cpp
  OpenSeesCrdTransfCommands.cpp
  OpenSeesDampingCommands.cpp
  OpenSeesBeamIntegrationCommands.cpp
  OpenSeesNDMaterialCommands.cpp
  OpenSeesMiscCommands.cpp
  OpenSeesParameterCommands.cpp
  OpenSeesFrictionModelCommands.cpp
  OpenSeesReliabilityCommands.cpp
  OpenSeesNDTestCommands.cpp
  OpenSeesIGACommands.cpp
)

target_sources(OPS_INTERPRETER PRIVATE ${OPS_INTERPRETER_CPP})

if(MPI_FOUND)
  add_library(OPS_INTERPRETER_PI OBJECT EXCLUDE_FROM_ALL)
  target_sources(OPS_INTERPRETER_PI PRIVATE ${OPS_INTERPRETER_CPP})
  # Prefer explicit prefix from find_path (Homebrew, custom METIS5_ROOT); /usr/include needs no extra -I.
  if(DEFINED OPENSEES_METIS5_INCLUDE_DIR AND OPENSEES_METIS5_INCLUDE_DIR AND NOT OPENSEES_METIS5_INCLUDE_DIR STREQUAL "OPENSEES_METIS5_INCLUDE_DIR-NOTFOUND")
    target_include_directories(
      OPS_INTERPRETER_PI BEFORE PRIVATE "${OPENSEES_METIS5_INCLUDE_DIR}"
    )
  endif()
  if(DEFINED OPENMPI)
    target_compile_definitions(
      OPS_INTERPRETER_PI
      PRIVATE
      _PARALLEL_INTERPRETERS
      ${MUMPS_FLAG}
      _OPENMPI
    )
  else()
    target_compile_definitions(
      OPS_INTERPRETER_PI
      PRIVATE
      _PARALLEL_INTERPRETERS
      ${MUMPS_FLAG}
    )
  endif()
endif()


target_sources(
  OpenSeesPy
  PRIVATE
  PythonModule.cpp
  PythonWrapper.cpp
  PythonStream.cpp
)
