# Shared LargeMP brick column mesh.
# Sourced by Example.tcl (serial OpenSees or OpenSeesMP).
#
# Units: metre, kilonewton, second  => mass unit is tonne (t = kN·s²/m)
# Geometry: 2 m × 2 m cross-section, height 10 m (solid concrete pier)

model basic -ndm 3 -ndf 3

# Concrete-like isotropic solid
set E    2.5e7   ;# 25 GPa
set nu   0.20
set rho  2.4     ;# 2.4 t/m³
nDMaterial ElasticIsotropic 1 $E $nu $rho

set eleArgs "1"
set element stdBrick

set nn [expr {($nz+1)*($nx+1)*($ny+1)}]

block3D $nx $ny $nz   1 1  $element  $eleArgs {
    1   -1     -1      0
    2    1     -1      0
    3    1      1      0
    4   -1      1      0
    5   -1     -1     10
    6    1     -1     10
    7    1      1     10
    8   -1      1     10
}

# Extra lumped mass = 50% of solid self-weight, uniform on all nodes
# (nonstructural / finishes; also exercises nodal-mass partition ownership)
set Lx 2.0
set Ly 2.0
set Lz 10.0
set Mstruct [expr {$rho*$Lx*$Ly*$Lz}]
set Madd    [expr {0.5*$Mstruct}]
set mNode   [expr {$Madd/double($nn)}]
for {set n 1} {$n <= $nn} {incr n} {
    mass $n $mNode $mNode $mNode
}

# Gravity via uniform acceleration (OpenSees: UniformExcitation + Constant series).
# Direction 3 = global Z. Excitation applies F = -M*ugddot, so fact = +g
# yields downward body force when +Z is upward.
set g 9.81
timeSeries Constant 1
pattern UniformExcitation 1 3 -accel 1 -fact $g

fixZ 0.0   1 1 1
