# OpenSeesMP smoke: ExplicitAlphaMultiSOE + CudaKRAlpha with gather-add SOEs.
# Run: mpirun -np 2 OpenSeesMP tests/multisoe_mp_smoke.tcl
#
# Builds a 1D two-truss series (same topology as distributed_cudss_mp_smoke.tcl),
# applies a tip load, and runs a short explicit transient with:
#   - KRAlphaExplicitMultiSOE + Mumps
#   - KRAlphaExplicitMultiSOE + ParallelProfileSPD
#   - KRAlphaExplicitMultiSOE + DistributedCuDSS
#   - CudaKRAlpha + DistributedCuDSS
# Tip displacements are compared across backends.

wipe

set pid [getPID]
set np  [getNP]

if {$np < 2} {
    puts "multisoe_mp_smoke requires at least 2 MPI ranks (got $np)"
    exit 1
}

proc buildModel {} {
    wipe
    model BasicBuilder -ndm 1 -ndf 1
    uniaxialMaterial Elastic 1 100.0

    set pid [getPID]
    if {$pid == 0} {
        node 1 0.0
        node 2 1.0
        mass 2 1.0
        fix 1 1
        element Truss 1 1 2 1.0 1
    } else {
        node 2 1.0
        node 3 2.0
        mass 2 1.0
        mass 3 1.0
        element Truss 2 2 3 1.0 1
        timeSeries Constant 1
        pattern Plain 1 1 {
            load 3 1.0
        }
    }
}

proc tipDisp {} {
    if {[getPID] == [expr {[getNP] - 1}]} {
        return [nodeDisp 3 1]
    }
    return 0.0
}

proc runTransient {sysName integName {extraArgs {}}} {
    buildModel
    constraints Plain
    numberer ParallelPlain
    system $sysName
    algorithm Linear
    if {$extraArgs eq ""} {
        integrator $integName 1.0
    } else {
        eval integrator $integName 1.0 $extraArgs
    }
    analysis Transient

    set dt 0.01
    set ok 0
    for {set i 0} {$i < 10} {incr i} {
        set ok [analyze 1 $dt]
        if {$ok != 0} {
            puts "ERROR: analyze failed for $sysName + $integName on pid [getPID] at step $i"
            exit 1
        }
    }
    return [tipDisp]
}

set dMumps [runTransient Mumps KRAlphaExplicitMultiSOE]
set dProf  [runTransient ParallelProfileSPD KRAlphaExplicitMultiSOE]
set dCuda  [runTransient DistributedCuDSS KRAlphaExplicitMultiSOE]
set dCudaI [runTransient DistributedCuDSS CudaKRAlpha]

set pid [getPID]
puts "pid=$pid MultiSOE+Mumps tip=$dMumps"
puts "pid=$pid MultiSOE+ParallelProfileSPD tip=$dProf"
puts "pid=$pid MultiSOE+DistCuDSS tip=$dCuda"
puts "pid=$pid CudaKRAlpha+DistCuDSS tip=$dCudaI"

set tol 1.0e-6
if {$pid == [expr {$np - 1}]} {
    if {abs($dMumps) < 1.0e-12 || abs($dProf) < 1.0e-12 || abs($dCuda) < 1.0e-12 || abs($dCudaI) < 1.0e-12} {
        puts "ERROR: response remained near zero"
        exit 1
    }
    if {abs($dMumps - $dProf) > $tol} {
        puts "ERROR: MultiSOE Mumps vs ParallelProfileSPD disagree ($dMumps vs $dProf)"
        exit 1
    }
    if {abs($dMumps - $dCuda) > $tol} {
        puts "ERROR: MultiSOE Mumps vs DistCuDSS disagree ($dMumps vs $dCuda)"
        exit 1
    }
    if {abs($dMumps - $dCudaI) > $tol} {
        puts "ERROR: MultiSOE Mumps vs CudaKRAlpha disagree ($dMumps vs $dCudaI)"
        exit 1
    }
    puts "MultiSOE/CudaKRAlpha OpenSeesMP smoke test passed."
}

wipe
