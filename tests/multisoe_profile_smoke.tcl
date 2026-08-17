# Serial smoke: ExplicitAlphaMultiSOE with ProfileSPD vs BandGeneral.
# Run: OpenSees tests/multisoe_profile_smoke.tcl
#
# Checks getCopy + unfactored formAp (second analyze step after soeM was factored).
# Uses rhoInf=0.5 so |alphaM-alphaF| is above the MultiSOE alpha-close shortcut.

wipe

proc buildModel {} {
    wipe
    model BasicBuilder -ndm 1 -ndf 1
    node 1 0.0
    node 2 1.0
    node 3 2.0
    mass 2 1.0
    mass 3 1.0
    fix 1 1
    uniaxialMaterial Elastic 1 100.0
    element truss 1 1 2 1.0 1
    element truss 2 2 3 1.0 1
    timeSeries Constant 1
    pattern Plain 1 1 {
        load 3 1.0
    }
}

proc runTransient {sysName} {
    buildModel
    constraints Plain
    numberer RCM
    system $sysName
    algorithm Linear
    integrator MKRAlphaExplicitMultiSOE 0.5
    analysis Transient

    set dt 0.01
    for {set i 0} {$i < 12} {incr i} {
        set ok [analyze 1 $dt]
        if {$ok != 0} {
            puts "ERROR: analyze failed for $sysName at step $i"
            exit 1
        }
    }
    return [nodeDisp 3 1]
}

set dProf [runTransient ProfileSPD]
set dRef  [runTransient UmfPack]

puts "MultiSOE+ProfileSPD tip=$dProf"
puts "MultiSOE+UmfPack tip=$dRef"

set tol 1.0e-6
if {abs($dProf) < 1.0e-12 || abs($dRef) < 1.0e-12} {
    puts "ERROR: response remained near zero"
    exit 1
}
if {abs($dProf - $dRef) > $tol} {
    puts "ERROR: ProfileSPD vs UmfPack disagree ($dProf vs $dRef)"
    exit 1
}

puts "MultiSOE/ProfileSPD serial smoke test passed."
wipe
