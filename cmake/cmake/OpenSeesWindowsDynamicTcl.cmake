# Dynamic Tcl (tcl86t.dll) for OpenSees / OpenSeesSP / OpenSeesMP on Windows.
# Conan often links static tcl86ts; OpenFresco loadPackage plugins require the
# host to share tcl86t.dll like the Win64 Visual Studio build.

function(ops_configure_windows_dynamic_tcl)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
    return()
  endif()

  set(_ops_tcl_root "C:/Program Files/Tcl")
  if(NOT EXISTS "${_ops_tcl_root}/lib/tcl86t.lib")
    message(WARNING
      "tcl86t.lib not found at ${_ops_tcl_root}; "
      "falling back to find_package/Conan Tcl (OpenFresco loadPackage may fail).")
    return()
  endif()

  set(OPS_WINDOWS_DYNAMIC_TCL_ROOT "${_ops_tcl_root}" PARENT_SCOPE)
  set(OPS_WINDOWS_DYNAMIC_TCL_INCLUDE "${_ops_tcl_root}/include" PARENT_SCOPE)
  set(OPS_WINDOWS_DYNAMIC_TCL_LIBRARY "${_ops_tcl_root}/lib/tcl86t.lib" PARENT_SCOPE)
  set(OPS_WINDOWS_DYNAMIC_TCL_STUB_LIBRARY "${_ops_tcl_root}/lib/tclstub86.lib" PARENT_SCOPE)
  set(OPS_WINDOWS_DYNAMIC_TCL_SCRIPT_DIR "${_ops_tcl_root}/lib/tcl8.6" PARENT_SCOPE)
  set(OPS_WINDOWS_DYNAMIC_TCL_DLL "${_ops_tcl_root}/bin/tcl86t.dll" PARENT_SCOPE)
endfunction()


function(ops_add_windows_dynamic_tcl_runtime target_name)
  if(NOT DEFINED OPS_WINDOWS_DYNAMIC_TCL_DLL OR NOT EXISTS "${OPS_WINDOWS_DYNAMIC_TCL_DLL}")
    return()
  endif()

  add_custom_command(
    TARGET ${target_name} POST_BUILD
    COMMENT "Copying tcl86t.dll next to ${target_name}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${OPS_WINDOWS_DYNAMIC_TCL_DLL}"
      $<TARGET_FILE_DIR:${target_name}>
  )

  if(DEFINED OPS_WINDOWS_DYNAMIC_TCL_SCRIPT_DIR AND EXISTS "${OPS_WINDOWS_DYNAMIC_TCL_SCRIPT_DIR}")
    file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/lib/tcl8.6)
    file(GLOB _ops_tcl_scripts "${OPS_WINDOWS_DYNAMIC_TCL_SCRIPT_DIR}/*.tcl")
    if(_ops_tcl_scripts)
      file(COPY ${_ops_tcl_scripts}
           DESTINATION ${PROJECT_BINARY_DIR}/lib/tcl8.6)
    endif()
  endif()
endfunction()
