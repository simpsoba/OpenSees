# Smoke test: cuDSS multithreaded (MT) mode (-multiThreadingMode 1)
# Matches NVIDIA simple_multithreaded_mode: cudssSetThreadingLayer + mtlayer shim.
#
# Run:
#   opensees-cuda-env.bat
#   OpenSees.exe tests/cudss_multithreading_smoke.tcl
#
# Optional NVIDIA-style env layer (threadingLibPath NULL + CUDSS_THREADING_LIB):
#   set CUDSS_THREADING_LIB=cudss_mtlayer_vcomp14064_0.dll
#   OpenSees.exe tests/cudss_multithreading_smoke.tcl env

set useEnvLayer 0
if {$argc >= 1 && [string equal -nocase [lindex $argv 0] env]} {
  set useEnvLayer 1
}

wipe
model BasicBuilder -ndm 1 -ndf 1
node 1 0.0
node 2 1.0
mass 2 1.0
fix 1 1
uniaxialMaterial Elastic 1 100.0
element truss 1 1 2 1.0 1

if {$useEnvLayer} {
  system CuDSS -multiThreadingMode 1 -threadingLibPath NULL -verbose 1
} else {
  system CuDSS -multiThreadingMode 1 -verbose 1
}

numberer RCM
constraints Plain
algorithm Linear
integrator CudaExplicitAlpha_TP 0.5 0.5 0.5 0.25
analysis Transient

timeSeries Constant 1
pattern Plain 1 1 {
  load 2 1.0
}

set dt 0.01
set finalDisp 0.0
for {set i 0} {$i < 20} {incr i} {
  set ok [analyze 1 $dt]
  if {$ok != 0} {
    puts "ERROR analyze failed at step $i"
    exit 1
  }
  set finalDisp [nodeDisp 2]
}

# Reference from cuda_explicit_alpha_tp_smoke.tcl (system CuDSS, MT off)
set ref 0.01418169617060478609
set tol 1.0e-10
if {[expr {abs($finalDisp - $ref) > $tol}]} {
  puts "ERROR MT result differs from ST reference"
  puts "  expected = $ref"
  puts "  got      = $finalDisp"
  exit 1
}

if {$useEnvLayer} {
  puts "cuDSS MT smoke test passed (CUDSS_THREADING_LIB / NULL path)."
} else {
  puts "cuDSS MT smoke test passed (default mtlayer path)."
}
puts "final disp = $finalDisp"
exit 0
