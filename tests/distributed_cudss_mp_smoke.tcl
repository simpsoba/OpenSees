# OpenSeesMP smoke: gather-to-root DistributedCuDSS vs Mumps on a 2-subdomain truss.
# Run: mpirun -np 2 OpenSeesMP tests/distributed_cudss_mp_smoke.tcl

wipe

set pid [getPID]
set np  [getNP]

if {$np < 2} {
    puts "distributed_cudss_mp_smoke requires at least 2 MPI ranks (got $np)"
    exit 1
}

proc buildModel {} {
    wipe
    model BasicBuilder -ndm 1 -ndf 1
    uniaxialMaterial Elastic 1 100.0

    set pid [getPID]

    # Shared interface node 2 exists on both ranks.
    if {$pid == 0} {
        node 1 0.0
        node 2 1.0
        fix 1 1
        element Truss 1 1 2 1.0 1
    } else {
        node 2 1.0
        node 3 2.0
        element Truss 2 2 3 1.0 1
        timeSeries Linear 1
        pattern Plain 1 1 {
            load 3 1.0
        }
    }
}

proc runStatic {sysName} {
    buildModel
    constraints Plain
    numberer ParallelPlain
    system $sysName
    algorithm Linear
    integrator LoadControl 1.0
    analysis Static

    set ok [analyze 1]
    if {$ok != 0} {
        puts "ERROR: analyze failed for system $sysName on pid [getPID]"
        exit 1
    }

    set d2 0.0
    set d3 0.0
    if {[getPID] == 0} {
        set d2 [nodeDisp 2 1]
    } else {
        set d2 [nodeDisp 2 1]
        set d3 [nodeDisp 3 1]
    }
    return [list $d2 $d3]
}

# Reference: Mumps (collective distributed)
set mumpsDisp [runStatic Mumps]
set m2 [lindex $mumpsDisp 0]
set m3 [lindex $mumpsDisp 1]

# Candidate: DistributedCuDSS (gather-to-root GPU)
set cudaDisp [runStatic DistributedCuDSS]
set c2 [lindex $cudaDisp 0]
set c3 [lindex $cudaDisp 1]

set pid [getPID]
puts "pid=$pid Mumps:    u2=$m2 u3=$m3"
puts "pid=$pid DistCuDSS: u2=$c2 u3=$c3"

# Exact 1D two-spring series, tip load 1, k=100 each -> u2=0.01, u3=0.02
set tol 1.0e-8
if {$pid == [expr {$np - 1}]} {
    if {abs($c3 - 0.02) > $tol || abs($m3 - 0.02) > $tol} {
        puts "ERROR: unexpected tip displacement (expected 0.02)"
        exit 1
    }
    if {abs($c3 - $m3) > $tol || abs($c2 - $m2) > $tol} {
        puts "ERROR: DistributedCuDSS and Mumps disagree"
        exit 1
    }
    puts "DistributedCuDSS OpenSeesMP smoke test passed."
}

wipe
