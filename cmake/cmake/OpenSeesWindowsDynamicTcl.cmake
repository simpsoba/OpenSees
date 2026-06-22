# Dynamic Tcl (tcl86t.dll) for OpenSeesFresco / OpenSeesSPFresco / OpenSeesMPFresco on Windows.
# Conan links static tcl86ts into the default OpenSees target; OpenFrescoTcl
# plugins require the host to share tcl86t.dll like the Win64 sln build.

function(ops_configure_windows_dynamic_tcl)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
    return()
  endif()

  set(_ops_tcl_root "C:/Program Files/Tcl")
  if(NOT EXISTS "${_ops_tcl_root}/lib/tcl86t.lib")
    message(WARNING
      "tcl86t.lib not found at ${_ops_tcl_root}; "
      "OpenSeesFresco, OpenSeesSPFresco, and OpenSeesMPFresco will not be configured.")
    return()
  endif()

  set(OPS_FRESCO_TCL_ROOT "${_ops_tcl_root}" PARENT_SCOPE)
  set(OPS_FRESCO_TCL_INCLUDE "${_ops_tcl_root}/include" PARENT_SCOPE)
  set(OPS_FRESCO_TCL_LIBRARY "${_ops_tcl_root}/lib/tcl86t.lib" PARENT_SCOPE)
  set(OPS_FRESCO_TCL_STUB_LIBRARY "${_ops_tcl_root}/lib/tclstub86.lib" PARENT_SCOPE)
  set(OPS_FRESCO_TCL_SCRIPT_DIR "${_ops_tcl_root}/lib/tcl8.6" PARENT_SCOPE)
  set(OPS_FRESCO_TCL_DLL "${_ops_tcl_root}/bin/tcl86t.dll" PARENT_SCOPE)
endfunction()


function(ops_fresco_conan_libs out_var)
  set(_libs ${CONAN_LIBS})
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND DEFINED OPS_FRESCO_TCL_LIBRARY)
    set(_filtered)
    foreach(_lib ${_libs})
      if(NOT _lib MATCHES "^tcl")
        list(APPEND _filtered ${_lib})
      endif()
    endforeach()
    list(APPEND _filtered
      "${OPS_FRESCO_TCL_LIBRARY}"
      ws2_32 netapi32 userenv)
    set(_libs ${_filtered})
  endif()
  set(${out_var} ${_libs} PARENT_SCOPE)
endfunction()


function(ops_add_fresco_tcl_runtime target_name)
  if(NOT DEFINED OPS_FRESCO_TCL_DLL OR NOT EXISTS "${OPS_FRESCO_TCL_DLL}")
    return()
  endif()

  add_custom_command(
    TARGET ${target_name} POST_BUILD
    COMMENT "Copying tcl86t.dll next to ${target_name}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${OPS_FRESCO_TCL_DLL}"
      $<TARGET_FILE_DIR:${target_name}>
  )

  if(DEFINED OPS_FRESCO_TCL_SCRIPT_DIR AND EXISTS "${OPS_FRESCO_TCL_SCRIPT_DIR}")
    file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/lib/tcl8.6)
    file(GLOB _ops_tcl_scripts "${OPS_FRESCO_TCL_SCRIPT_DIR}/*.tcl")
    if(_ops_tcl_scripts)
      file(COPY ${_ops_tcl_scripts}
           DESTINATION ${PROJECT_BINARY_DIR}/lib/tcl8.6)
    endif()
  endif()
endfunction()
