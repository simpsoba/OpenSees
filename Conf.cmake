#==============================================================================
# 
#        OpenSees -- Open System For Earthquake Engineering Simulation
#                Pacific Earthquake Engineering Research Center
#
#     (c) Copyright 1999-2021 The Regents of the University of California
#                             All Rights Reserved
# (Copyright and Disclaimer @ http://www.berkeley.edu/OpenSees/copyright.html)
#
#==============================================================================


#==============================================================================
#                           Select Default Executable
#==============================================================================
set(OPS_FINAL_TARGET "OpenSees" CACHE STRING "OpenSees final target")


#==============================================================================
#                            Basic Switches
#==============================================================================

option(OPS_Use_Dev_Directories
  "Include files in DEVELOPER directory"                   OFF)

option(FMK
  "Special FMK Code"                                       OFF)

option(PARALLEL_PROCESSING
  "OpenSeesSP build mode: apply _PARALLEL_PROCESSING to object libs and OPS_InterpTcl (mutually exclusive with OpenSeesMP _PARALLEL_INTERPRETERS in one configure)" OFF)

set(OPS_Use_Graphics_Option
  None
  # Base
  # OpenGL
)

