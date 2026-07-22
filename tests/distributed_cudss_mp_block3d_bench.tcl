# Parameterized block3D cantilever bench for serial OpenSees and OpenSeesMP.
#
# Loading: Linear timeSeries + UniformExcitation in global Z (10×g ramp).
# Static:  10 LoadControl steps of 0.1 (t: 0 -> 1)
# Transient: short Newmark run (0.5 s) so tip can start oscillating about the ramp
# Algorithm: Newton + NormDispIncr
# Solvers: serial UmfPack/CuDSS (Plain); MP Mumps/DistributedCuDSS (ParallelPlain)
#
# Usage:
#   OpenSees tests/distributed_cudss_mp_block3d_bench.tcl nx ny nz
#   mpirun -np N OpenSeesMP tests/distributed_cudss_mp_block3d_bench.tcl nx ny nz
#
# Optional 4th arg: static | transient | both  (default both)
#
# Emits BENCH / HISTORY / COMPARE / PASS lines.

wipe

set pid [getPID]
set np  [getNP]

if {$argc < 3} {
    if {$pid == 0} {
        puts "Usage: ... block3d_bench.tcl nx ny nz \[static|transient|both\]"
    }
    exit 1
}

set nx [lindex $argv 0]
set ny [lindex $argv 1]
set nz [lindex $argv 2]
set mode both
if {$argc >= 4} {
    set mode [string tolower [lindex $argv 3]]
}

if {![string is integer -strict $nx] || ![string is integer -strict $ny] || ![string is integer -strict $nz]} {
    if {$pid == 0} { puts "ERROR: nx ny nz must be integers" }
    exit 1
}
if {$nx < 1 || $ny < 1 || $nz < 1} {
    if {$pid == 0} { puts "ERROR: nx ny nz must be >= 1" }
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
set z0         [expr {10.0 * $firstLayer / double($nz)}]
set z1         [expr {10.0 * ($firstLayer + $localNz) / double($nz)}]

set nodesPerLayer [expr {($nx + 1) * ($ny + 1)}]
set elesPerLayer  [expr {$nx * $ny}]
set startNode     [expr {$firstLayer * $nodesPerLayer + 1}]
set startEle      [expr {$firstLayer * $elesPerLayer + 1}]
set tipNode       [expr {($nz + 1) * $nodesPerLayer}]
set nEleGlobal    [expr {$nx * $ny * $nz}]
set nNodeGlobal   [expr {($nx + 1) * ($ny + 1) * ($nz + 1)}]
# Estimate used only before the first analyze (UmfPack gate / START banner).
# Reported ndof comes from systemSize after the first analyze.
set nDofEst       [expr {($nNodeGlobal - $nodesPerLayer) * 3}]
set nDof          0

set gAccel 9.81
# Apply inertial loading equivalent to 10x self-weight via UniformExcitation.
set gravityLoadFactor 10.0
set nStepsStatic 10
set staticDLambda 0.1
# Short transient: 0.5 s is enough to see motion about the Linear ramp while
# keeping wall time low enough for a denser DOF sweep up to ~10^4.
set nStepsTransient 50
set transientDt 0.01
set newtonTol 1.0e-8
set newtonIter 25

proc buildModel {} {
    global pid np nx ny localNz z0 z1 startNode startEle nodesPerLayer gAccel gravityLoadFactor

    wipe
    model BasicBuilder -ndm 3 -ndf 3
    # Density required so UniformExcitation can form inertial gravity loads.
    # Softer modulus so 10×g gravity produces clearly visible tip motion.
    nDMaterial ElasticIsotropic 1 2.5e5 0.20 2.4

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

    # Linear ramp of vertical ground acceleration => body force -M*a_g.
    # fact = 10*g so the applied load is 10x the structure weight.
    timeSeries Linear 1
    pattern UniformExcitation 1 3 -accel 1 -fact [expr {$gravityLoadFactor * $gAccel}]
}

proc tipDisp {} {
    global pid np tipNode
    if {$pid == [expr {$np - 1}]} {
        return [list [nodeDisp $tipNode 1] [nodeDisp $tipNode 2] [nodeDisp $tipNode 3]]
    }
    return [list 0.0 0.0 0.0]
}

proc tipNorm {disp} {
    set ux [lindex $disp 0]
    set uy [lindex $disp 1]
    set uz [lindex $disp 2]
    return [expr {sqrt($ux*$ux + $uy*$uy + $uz*$uz)}]
}

proc maxAbsDiff {a b} {
    set dmax 0.0
    for {set i 0} {$i < [llength $a]} {incr i} {
        set d [expr {abs([lindex $a $i] - [lindex $b $i])}]
        if {$d > $dmax} { set dmax $d }
    }
    return $dmax
}

proc emitHistory {analysisType sysName numbererName step tSec disp} {
    global pid np nx ny nz nDof
    if {$pid != [expr {$np - 1}]} {
        return
    }
    set ux [lindex $disp 0]
    set uy [lindex $disp 1]
    set uz [lindex $disp 2]
    set tn [tipNorm $disp]
    puts [format \
        "HISTORY type=%s np=%d nx=%d ny=%d nz=%d ndof=%d system=%s numberer=%s step=%d time=%.8e ux=%.16e uy=%.16e uz=%.16e tip_norm=%.16e" \
        $analysisType $np $nx $ny $nz $nDof $sysName $numbererName $step $tSec $ux $uy $uz $tn]
}

proc emitBench {analysisType sysName numbererName analyzeMs disp} {
    global pid np nx ny nz nEleGlobal nNodeGlobal nDof
    if {$pid != [expr {$np - 1}]} {
        return
    }
    set ux [lindex $disp 0]
    set uy [lindex $disp 1]
    set uz [lindex $disp 2]
    set tn [tipNorm $disp]
    puts [format \
        "BENCH type=%s np=%d nx=%d ny=%d nz=%d nodes=%d eles=%d ndof=%d system=%s numberer=%s analyze_ms=%d tip=%.16e,%.16e,%.16e tip_norm=%.16e" \
        $analysisType $np $nx $ny $nz $nNodeGlobal $nEleGlobal $nDof \
        $sysName $numbererName $analyzeMs $ux $uy $uz $tn]
}

# Returns: analyze_ms finalTipDisp
proc runCase {analysisType sysName numbererName} {
    global pid nStepsStatic nStepsTransient staticDLambda transientDt newtonTol newtonIter nDof

    buildModel
    constraints Plain
    numberer $numbererName
    system $sysName
    test NormDispIncr $newtonTol $newtonIter 0
    # Linear elastic + constant Newmark/LoadControl: form+factor once, reuse A.
    algorithm Newton -factorOnce

    if {$analysisType eq "static"} {
        integrator LoadControl $staticDLambda
        analysis Static
        set t0 [clock milliseconds]
        for {set s 1} {$s <= $nStepsStatic} {incr s} {
            set ok [analyze 1]
            if {$ok != 0} {
                puts "ERROR: static step $s failed system=$sysName numberer=$numbererName pid=$pid"
                exit 1
            }
            if {$s == 1} {
                set nDof [systemSize]
            }
            set tSec [expr {$s * $staticDLambda}]
            emitHistory static $sysName $numbererName $s $tSec [tipDisp]
        }
        set t1 [clock milliseconds]
    } elseif {$analysisType eq "transient"} {
        integrator Newmark 0.5 0.25
        analysis Transient
        set t0 [clock milliseconds]
        for {set s 1} {$s <= $nStepsTransient} {incr s} {
            set ok [analyze 1 $transientDt]
            if {$ok != 0} {
                puts "ERROR: transient step $s failed system=$sysName numberer=$numbererName pid=$pid"
                exit 1
            }
            if {$s == 1} {
                set nDof [systemSize]
            }
            set tSec [expr {$s * $transientDt}]
            emitHistory transient $sysName $numbererName $s $tSec [tipDisp]
        }
        set t1 [clock milliseconds]
    } else {
        puts "ERROR: unknown analysisType=$analysisType"
        exit 1
    }

    set ms [expr {$t1 - $t0}]
    return [list $ms [tipDisp]]
}

proc solverCases {} {
    global np nDofEst
    if {$np == 1} {
        # UmfPack is only for small correctness checks; it gets slow at ~10k DOFs.
        if {$nDofEst <= 2000} {
            return {
                {UmfPack Plain}
                {CuDSS Plain}
            }
        }
        return {
            {CuDSS Plain}
        }
    }
    # Skip ProfileSPD here; Mumps + DistributedCuDSS are the MP comparison set.
    return {
        {Mumps ParallelPlain}
        {DistributedCuDSS ParallelPlain}
    }
}

proc runAnalysisType {analysisType tol} {
    global pid np

    set cases [solverCases]
    set refName ""
    set refDisp {}
    set first 1

    foreach case $cases {
        set sysName [lindex $case 0]
        set numbererName [lindex $case 1]
        set result [runCase $analysisType $sysName $numbererName]
        set ms [lindex $result 0]
        set disp [lindex $result 1]
        emitBench $analysisType $sysName $numbererName $ms $disp

        if {$pid == [expr {$np - 1}]} {
            if {[tipNorm $disp] < 1.0e-16} {
                puts "ERROR: near-zero tip for type=$analysisType system=$sysName"
                exit 1
            }
            if {$first} {
                set refName $sysName
                set refDisp $disp
                set first 0
            } else {
                set diff [maxAbsDiff $refDisp $disp]
                puts [format "COMPARE type=%s ref=%s cand=%s max|du|=%.6e" \
                    $analysisType $refName $sysName $diff]
                if {$diff > $tol} {
                    puts "ERROR: disagreement exceeds tol=$tol"
                    exit 1
                }
            }
        }
    }
}

set tol 1.0e-6

if {$pid == 0} {
    puts [format "START block3d_bench np=%d nx=%d ny=%d nz=%d mode=%s nodes=%d eles=%d ndof_est=%d" \
        $np $nx $ny $nz $mode $nNodeGlobal $nEleGlobal $nDofEst]
}

if {$mode eq "static" || $mode eq "both"} {
    runAnalysisType static $tol
}
if {$mode eq "transient" || $mode eq "both"} {
    runAnalysisType transient $tol
}

if {$pid == 0} {
    puts [format "PASS block3d_bench np=%d nx=%d ny=%d nz=%d mode=%s" \
        $np $nx $ny $nz $mode]
}

wipe
exit 0
