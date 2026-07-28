# 40-story 1-D shear building via OpenSeesMP + partition
#
# Unlike ShearBuilding40MP.tcl (manual per-rank mesh build), every rank builds
# the full model, then partition removes foreign elements/nodes.
#
# Run:
#   mpiexec -n 4 OpenSeesMP ShearBuilding40Partition.tcl
#   SHEAR40_PARTITION=metis mpiexec -n 4 OpenSeesMP ShearBuilding40Partition.tcl
#
# SHEAR40_PARTITION:
#   custom (default) - even story bands via partition -customPartition
#   metis            - METIS_PartMeshNodal via partition

# =============================================================================
# Example mode
# =============================================================================
# analytical only here (no OpenFresco). Use ShearBuilding40MP.tcl for local.
set expElementMode analytical

if {[info exists ::env(SHEAR40_MODE)] && [string length [string trim $::env(SHEAR40_MODE)]] > 0} {
    set expElementMode [string trim $::env(SHEAR40_MODE)]
}
if {$expElementMode ne "analytical"} {
    puts "ShearBuilding40Partition.tcl supports analytical mode only"
    puts "(use ShearBuilding40MP.tcl for OpenFresco local)"
    exit 1
}

set partitionMode custom
if {[info exists ::env(SHEAR40_PARTITION)] && [string length [string trim $::env(SHEAR40_PARTITION)]] > 0} {
    set partitionMode [string tolower [string trim $::env(SHEAR40_PARTITION)]]
}
if {$partitionMode ne "custom" && $partitionMode ne "metis"} {
    puts "ERROR SHEAR40_PARTITION must be custom or metis (got '$partitionMode')"
    exit 1
}

set outputDir "output-partition-$partitionMode"

# =============================================================================
# Model settings
# =============================================================================
set Nstories 40
set m 1.0
set kbottom 900
set ktop 600
set ExpEleTag 20

# =============================================================================
# Rayleigh damping  C = alphaM*M + betaK*K
# =============================================================================
set zeta 0.05
set w1 2.0
set w2 20.0

# =============================================================================
# Analysis scheme
# =============================================================================
set analysisScheme NewmarkExplicit

set nSteps 3120
set dt 0.02

source [file join [pwd] shear40_print.tcl]

# =============================================================================

proc puts0 {msg} {
    global pid
    if {$pid == 0} {
        puts $msg
    }
}

proc storyStiff {i Nstories kbottom ktop} {
    return [expr {$kbottom + ($ktop-$kbottom)*double($i-1)/double($Nstories-1)}]
}

proc applyAnalysisScheme {scheme} {
    switch -exact $scheme {
        NewmarkExplicit {
            integrator NewmarkExplicit 0.5
            algorithm Linear
        }
        AlphaOSGeneralized {
            integrator AlphaOSGeneralized 0.9
            algorithm Linear
        }
        Newmark {
            integrator Newmark 0.5 0.25
            test EnergyIncr 1.0e-10 20 0
            algorithm KrylovNewton
        }
        default {
            puts "ERROR unknown analysisScheme: $scheme (use NewmarkExplicit, AlphaOSGeneralized, or Newmark)"
            exit 1
        }
    }
}

proc addNumericalSpring {i Nstories kbottom ktop} {
    set k [storyStiff $i $Nstories $kbottom $ktop]
    uniaxialMaterial Elastic $i $k
    element zeroLength $i [expr {$i - 1}] $i -mat $i -dir 1 -doRayleigh
}

# Even contiguous story bands -> element-tag/part flat list for -customPartition
proc evenStoryCustomPartition {Nstories numP} {
    set base [expr {$Nstories / $numP}]
    set rem [expr {$Nstories % $numP}]
    set custom [dict create]
    set story 1
    for {set p 0} {$p < $numP} {incr p} {
        set cnt [expr {$p < $rem ? $base + 1 : $base}]
        if {$cnt <= 0} {
            continue
        }
        set first $story
        set last [expr {$story + $cnt - 1}]
        for {set i $first} {$i <= $last} {incr i} {
            dict set custom $i $p
        }
        set story [expr {$last + 1}]
    }
    return $custom
}

# --- model generation ---

set numP [getNP]
set pid [getPID]

if {$pid == 0} {
    resetOutputDir $outputDir
}

wipe
if {$pid == 0} {
    logFile [file join $outputDir ShearBuilding40Partition_${partitionMode}.log]
}

model BasicBuilder -ndm 1 -ndf 1

puts0 "=== ShearBuilding40 / OpenSeesMP+partition ==="
puts0 "mode=$expElementMode  partitionMode=$partitionMode  numP=$numP"
puts0 "ExpEleTag=$ExpEleTag  Nstories=$Nstories  outputDir=$outputDir"
puts0 "analysis: $nSteps steps, dt=$dt s"

# Full mesh on every rank (required before partition)
node 0 0
fix 0 1
for {set i 1} {$i <= $Nstories} {incr i} {
    node $i 0 -mass $m
}
for {set i 1} {$i <= $Nstories} {incr i} {
    addNumericalSpring $i $Nstories $kbottom $ktop
}
puts0 "element: analytical - $Nstories zeroLength springs (no OpenFresco)"

set nBefore [llength [getNodeTags]]
set eBefore [llength [getEleTags]]
puts0 "before partition: nodes=$nBefore eles=$eBefore"

if {$numP > 1} {
    if {$partitionMode eq "custom"} {
        set customParts [evenStoryCustomPartition $Nstories $numP]
        if {[dict size $customParts] != $Nstories} {
            puts "ERROR custom partition size [dict size $customParts] != Nstories $Nstories"
            exit 1
        }
        puts0 "partition -customPartition ([dict size $customParts] elements)"
        if {$pid == 0} {
            for {set p 0} {$p < $numP} {incr p} {
                set tags {}
                dict for {ele part} $customParts {
                    if {$part == $p} {
                        lappend tags $ele
                    }
                }
                puts "  part $p stories/eles: [lsort -integer $tags]"
            }
        }
        partition -customPartition [dict size $customParts] {*}$customParts
    } else {
        puts0 "partition (METIS)"
        partition -ncuts 1 -niter 10 -ufactor 30
    }

    set localNodes [lsort -integer [getNodeTags]]
    set localEles  [lsort -integer [getEleTags]]
    puts "rank $pid after partition: nodes=[llength $localNodes] eles=[llength $localEles]  eles=$localEles"
} else {
    set localNodes [lsort -integer [getNodeTags]]
    set localEles  [lsort -integer [getEleTags]]
    puts0 "serial: skipping partition"
}

timeSeries Path 1 -filePath elcentro.txt -dt $dt -factor 1.0
pattern UniformExcitation 2 1 -accel 1

set alphaM [expr {$zeta * 2.0 * $w1 * $w2 / ($w1 + $w2)}]
set betaK [expr {$zeta * 2.0 / ($w1 + $w2)}]
rayleigh $alphaM 0.0 $betaK 0.0
puts0 "Rayleigh damping: zeta=$zeta at w1=$w1 w2=$w2 rad/s (alphaM=$alphaM betaK=$betaK)"

applyAnalysisScheme $analysisScheme
if {$numP > 1} {
    numberer ParallelPlain
    set solverLabel "Mumps+ParallelPlain"
} else {
    numberer Plain
    set solverLabel "UmfPack+Plain"
}
constraints Plain
if {$numP > 1} {
    system Mumps
} else {
    system UmfPack
}
analysis Transient

puts0 "Model built: target=OpenSeesMP+partition mode=$expElementMode partitionMode=$partitionMode solver=$solverLabel"

foreach i $localNodes {
    recorder Node -file [file join $outputDir node_${i}_disp.out] -time -node $i -dof 1 disp
}
if {[lsearch -exact $localEles $ExpEleTag] >= 0} {
    recorder Element -file [file join $outputDir Elmt_Frc.out] -time -ele $ExpEleTag forces
}

record
puts0 "Analysis starting..."

set ok [analyze $nSteps $dt]
puts0 "Analysis done: ok=$ok"

if {$ok == 0 && [lsearch -exact $localNodes $Nstories] >= 0} {
    set tip [nodeDisp $Nstories 1]
    puts "rank $pid tip node $Nstories disp=$tip"
    set fid [open [file join $outputDir tip_rank$pid.out] w]
    puts $fid $tip
    close $fid
}

wipe
exit
