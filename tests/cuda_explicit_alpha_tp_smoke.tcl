# Minimal Tcl smoke test for CudaExplicitAlpha_TP + CuDSS
wipe
model BasicBuilder -ndm 1 -ndf 1
node 1 0.0
node 2 1.0
mass 2 1.0
fix 1 1
uniaxialMaterial Elastic 1 100.0
element truss 1 1 2 1.0 1

system CuDSS
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

if {[expr {abs($finalDisp) < 1.0e-12}]} {
  puts "ERROR response remained zero"
  exit 1
}

puts "CudaExplicitAlpha_TP smoke test passed."
puts "final disp = $finalDisp"
exit 0
