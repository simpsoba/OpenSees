# Partial SP on retained node with equalDOF (multi-step Transformation).
#
# Two parallel 2D trusses 1--2 and 3--4. equalDOF ties node 3 to retained
# node 2 on both DOFs; node 2 has UY fixed and UX free. Five static
# LoadControl steps (total load factor 1) under constraints Transformation.
# Checks that node 3 tracks node 2 at every step, that UY stays zero, and that
# final UX matches the analytical parallel-spring result
# u = P / (2 * EA/L) = 5e-3.
#
# A single step cannot expose the bug (trial DOFs start at zero), and a fully
# fixed retained node cannot either (no free retained DOF to drop).
wipe
model basic -ndm 2 -ndf 2

node 1 0.0 0.0
node 2 1.0 0.0
node 3 1.0 0.0
node 4 2.0 0.0

fix 1 1 1
fix 4 1 1
# partial fix on the RETAINED node: Uy fixed, Ux free
fix 2 0 1

equalDOF 2 3   1 2

uniaxialMaterial Elastic 1 1000.0
element truss 1 1 2 10.0 1
element truss 2 3 4 10.0 1

timeSeries Linear 1
pattern Plain 1 1 {
    load 2 100.0 0.0
}

constraints Transformation
numberer Plain
system UmfPack
test NormUnbalance 1.0e-8 2 0
algorithm Newton
integrator LoadControl 0.2
analysis Static

set tol 1.0e-10
set fail 0
for {set step 1} {$step <= 5} {incr step} {
    if {[analyze 1] != 0} {
        puts "FAIL: analysis did not converge at step $step"
        exit 1
    }
    set ux2 [nodeDisp 2 1]
    set ux3 [nodeDisp 3 1]
    set uy3 [nodeDisp 3 2]
    set diff [expr {abs($ux2 - $ux3)}]
    puts [format "step %d: retained ux2=% .8e constrained ux3=% .8e |diff|=%.3e uy3=% .1e" \
        $step $ux2 $ux3 $diff $uy3]
    if {$diff > $tol} { set fail 1 }
    if {[expr {abs($uy3)}] > $tol} { set fail 1 }
}

if {$fail} {
    puts "FAIL: constrained node does not track partially fixed retained node"
    exit 1
}
# Analytical: two trusses EA/L = 10000 in parallel under P = 100 => u = 5e-3
set ux2 [nodeDisp 2 1]
if {[expr {abs($ux2 - 5.0e-3)}] > 1.0e-9} {
    puts [format "FAIL: wrong global solution ux2=%.6e expected 5e-3" $ux2]
    exit 1
}
puts "PASS"
exit 0
