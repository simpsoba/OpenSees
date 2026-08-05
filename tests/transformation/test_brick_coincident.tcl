# Coincident equalDOF with a partially fixed retained node.
#
# 3D truss chain 18--22--30 with coincident node 26 glued to retained node 22
# by equalDOF on all three DOFs. Node 22 has UY and UZ fixed, UX free. Five
# static LoadControl steps under constraints Transformation. Checks that
# node 26 tracks node 22 at every step (including free UX) and that the fixed
# DOFs stay zero.
wipe
model basic -ndm 3 -ndf 3

node 18 0.0 0.0 0.0
node 22 5.0 0.0 0.0
node 26 5.0 0.0 0.0
node 30 10.0 0.0 0.0

fix 18 1 1 1
fix 22 0 1 1
fix 30 0 1 1

equalDOF 22 26 1 2 3

uniaxialMaterial Elastic 1 1000.0
element truss 1 18 22 1.0 1
element truss 2 22 30 1.0 1

timeSeries Linear 1
pattern Plain 1 1 {
    load 30 10.0 0.0 0.0
}

constraints Transformation
numberer Plain
system UmfPack
test NormUnbalance 1.0e-8 2 0
algorithm Newton
integrator LoadControl 0.2
analysis Static

set tol 1.0e-8
set fail 0
for {set step 1} {$step <= 5} {incr step} {
    if {[analyze 1] != 0} {
        puts "FAIL: analysis did not converge at step $step"
        exit 1
    }
    set ux22 [nodeDisp 22 1]
    set ux26 [nodeDisp 26 1]
    set uy26 [nodeDisp 26 2]
    set uz26 [nodeDisp 26 3]
    set diff [expr {abs($ux22 - $ux26)}]
    puts [format "step %d: ux22=% .8e ux26=% .8e |diff|=%.3e uy26=% .1e uz26=% .1e" \
        $step $ux22 $ux26 $diff $uy26 $uz26]
    if {$diff > $tol} { set fail 1 }
    if {[expr {abs($uy26)}] > $tol || [expr {abs($uz26)}] > $tol} { set fail 1 }
}

if {$fail} {
    puts "FAIL: coincident equalDOF displacements inconsistent across steps"
    exit 1
}
if {[expr {abs([nodeDisp 22 1])}] < 1.0e-12} {
    puts "FAIL: expected nonzero UX on retained node"
    exit 1
}
puts "PASS"
exit 0
