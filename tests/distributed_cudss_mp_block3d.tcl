# OpenSeesMP regression: block3D cantilever sliced by getPID.
# Compares DistributedCuDSS vs Mumps and ParallelProfileSPD for static + transient.
#
# Run:
#   mpirun -np 2 OpenSeesMP tests/distributed_cudss_mp_block3d.tcl
#   mpirun -np 4 OpenSeesMP tests/distributed_cudss_mp_block3d.tcl

wipe

set pid [getPID]
set np  [getNP]

# Modest mesh so the smoke stays fast; nz must divide np.
set nx 2
set ny 2
set nz 4

if {$np < 2} {
    if {$pid == 0} {
        puts "ERROR: distributed_cudss_mp_block3d requires np >= 2 (got $np)"
    }
    exit 1
}
if {$nz % $np != 0} {
    if {$pid == 0} {
        puts "ERROR: nz=$nz must be divisible by np=$np"
    }
    exit 1
}

set localNz    [expr {$nz / $np}]
set firstLayer [expr {$pid * $localNz}]
set z0         [expr {10.0 * $firstLayer / $nz}]
set z1         [expr {10.0 * ($firstLayer + $localNz) / $nz}]

set nodesPerLayer [expr {($nx + 1) * ($ny + 1)}]
set elesPerLayer  [expr {$nx * $ny}]
set startNode     [expr {$firstLayer * $nodesPerLayer + 1}]
set startEle      [expr {$firstLayer * $elesPerLayer + 1}]
set tipNode       [expr {($nz + 1) * $nodesPerLayer}]

proc buildModel {} {
    global pid np nx ny nz localNz z0 z1 startNode startEle nodesPerLayer tipNode

    wipe
    model BasicBuilder -ndm 3 -ndf 3
    nDMaterial ElasticIsotropic 1 2.5e7 0.20 2.4

    # Contiguous vertical slab per rank; shared interface node tags match.
    block3D $nx $ny $localNz $startNode $startEle stdBrick 1 [subst {
        1 -1.0 -0.75 $z0   2  1.0 -0.75 $z0
        3  1.0  0.75 $z0   4 -1.0  0.75 $z0
        5 -1.0 -0.75 $z1   6  1.0 -0.75 $z1
        7  1.0  0.75 $z1   8 -1.0  0.75 $z1
    }]

    if {$pid == 0} {
        for {set nodeTag 1} {$nodeTag <= $nodesPerLayer} {incr nodeTag} {
            fix $nodeTag 1 1 1
        }
    }

    if {$pid == [expr {$np - 1}]} {
        timeSeries Constant 1
        pattern Plain 1 1 {
            load $tipNode 10.0 10.0 0.0
        }
    }
}

# Returns tip displacement list {ux uy uz} on the owning rank; zeros elsewhere.
proc tipDisp {} {
    global pid np tipNode
    if {$pid == [expr {$np - 1}]} {
        return [list [nodeDisp $tipNode 1] [nodeDisp $tipNode 2] [nodeDisp $tipNode 3]]
    }
    return [list 0.0 0.0 0.0]
}

proc runStatic {sysName numbererName} {
    buildModel
    constraints Plain
    numberer $numbererName
    system $sysName
    algorithm Linear
    integrator LoadControl 1.0
    analysis Static

    set ok [analyze 1]
    if {$ok != 0} {
        puts "ERROR: static analyze failed system=$sysName numberer=$numbererName pid=[getPID]"
        exit 1
    }
    return [tipDisp]
}

proc runTransient {sysName numbererName} {
    buildModel
    constraints Plain
    numberer $numbererName
    system $sysName
    algorithm Linear
    integrator Newmark 0.5 0.25
    analysis Transient

    set dt 0.001
    set nsteps 10
    set ok [analyze $nsteps $dt]
    if {$ok != 0} {
        puts "ERROR: transient analyze failed system=$sysName numberer=$numbererName pid=[getPID]"
        exit 1
    }
    return [tipDisp]
}

proc maxAbsDiff {a b} {
    set dmax 0.0
    for {set i 0} {$i < [llength $a]} {incr i} {
        set d [expr {abs([lindex $a $i] - [lindex $b $i])}]
        if {$d > $dmax} {
            set dmax $d
        }
    }
    return $dmax
}

proc checkAgree {label refName refDisp candName candDisp tol} {
    global pid np
    if {$pid != [expr {$np - 1}]} {
        return
    }
    set diff [maxAbsDiff $refDisp $candDisp]
    puts [format "COMPARE %s %s=%s  %s=%s  max|du|=%.6e" \
        $label $refName $refDisp $candName $candDisp $diff]
    if {$diff > $tol} {
        puts "ERROR: $label disagreement exceeds tol=$tol"
        exit 1
    }
}

set tol 1.0e-6

# --- Static ---
set mumpsStatic [runStatic Mumps ParallelPlain]
set spdStatic   [runStatic ParallelProfileSPD ParallelRCM]
set cudaStatic  [runStatic DistributedCuDSS ParallelPlain]

checkAgree static Mumps $mumpsStatic DistributedCuDSS $cudaStatic $tol
checkAgree static ParallelProfileSPD $spdStatic DistributedCuDSS $cudaStatic $tol

if {$pid == [expr {$np - 1}]} {
    set tipNorm [expr {sqrt( \
        [lindex $cudaStatic 0]*[lindex $cudaStatic 0] + \
        [lindex $cudaStatic 1]*[lindex $cudaStatic 1] + \
        [lindex $cudaStatic 2]*[lindex $cudaStatic 2])}]
    if {$tipNorm < 1.0e-12} {
        puts "ERROR: static tip displacement is ~0 (model/load likely wrong)"
        exit 1
    }
}

# --- Transient ---
set mumpsTrans [runTransient Mumps ParallelPlain]
set spdTrans   [runTransient ParallelProfileSPD ParallelRCM]
set cudaTrans  [runTransient DistributedCuDSS ParallelPlain]

checkAgree transient Mumps $mumpsTrans DistributedCuDSS $cudaTrans $tol
checkAgree transient ParallelProfileSPD $spdTrans DistributedCuDSS $cudaTrans $tol

if {$pid == [expr {$np - 1}]} {
    set tipNorm [expr {sqrt( \
        [lindex $cudaTrans 0]*[lindex $cudaTrans 0] + \
        [lindex $cudaTrans 1]*[lindex $cudaTrans 1] + \
        [lindex $cudaTrans 2]*[lindex $cudaTrans 2])}]
    if {$tipNorm < 1.0e-12} {
        puts "ERROR: transient tip displacement is ~0 (model/load likely wrong)"
        exit 1
    }
    puts [format \
        "DistributedCuDSS block3D MP regression passed (np=%d static tip=%s transient tip=%s)" \
        $np $cudaStatic $cudaTrans]
}

wipe
exit 0
