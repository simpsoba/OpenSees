# Fully fixed retained node with Transformation.
#
# 2D truss 1--2--3. Node 1 is fully fixed; equalDOF ties node 2 UX to node 1;
# unused UY DOFs are restrained. Single static LoadControl step under
# constraints Transformation. Checks that node 2 UX stays at the retained SP
# value (0) while free node 3 moves under axial load.
wipe
model basic -ndm 2 -ndf 2

node 1 0.0 0.0
node 2 10.0 0.0
node 3 20.0 0.0

fix 1 1 1
equalDOF 1 2 1
fix 2 0 1
fix 3 0 1

uniaxialMaterial Elastic 1 100.0
element truss 1 2 3 1.0 1

timeSeries Linear 1
pattern Plain 1 1 {
    load 3 10.0 0.0
}

constraints Transformation
numberer Plain
system UmfPack
test NormUnbalance 1.0e-8 2 0
algorithm Newton
integrator LoadControl 1.0
analysis Static

if {[analyze 1] != 0} {
    puts "FAIL: analysis did not converge"
    exit 1
}

set ux1 [nodeDisp 1 1]
set ux2 [nodeDisp 2 1]
set ux3 [nodeDisp 3 1]
puts [format "ux1=%.6e ux2=%.6e ux3=%.6e" $ux1 $ux2 $ux3]

if {abs($ux1 - $ux2) > 1.0e-8} {
    puts "FAIL: constrained node does not match retained SP"
    exit 1
}
if {abs($ux1) > 1.0e-10 || abs($ux2) > 1.0e-10} {
    puts "FAIL: retained SP not enforced"
    exit 1
}
if {abs($ux3) < 1.0e-12} {
    puts "FAIL: expected motion at free node 3"
    exit 1
}
puts "PASS"
exit 0
