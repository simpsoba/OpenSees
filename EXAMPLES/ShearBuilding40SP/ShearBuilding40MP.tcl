# 40-story 1-D shear building with a local experimental element (OpenSeesMPFresco)
#
# Run with MPI, e.g.:
#   mpiexec -n 4 bin\OpenSeesMPFresco.exe ShearBuilding40MP.tcl
#
# References:
#   local -> OpenFresco EXAMPLES/OneBayFrame/OpenSees/OneBayFrame_Local.tcl
#   MP pattern  -> EXAMPLES/ParallelModelMP/exampleMP.tcl

# =============================================================================
# Example mode
# =============================================================================
# analytical  - all stories are numerical zeroLength springs (no OpenFresco)
# local       - one story replaced by expElement twoNodeLink
set expElementMode analytical

if {[info exists ::env(SHEAR40_MODE)] && [string length [string trim $::env(SHEAR40_MODE)]] > 0} {
    set expElementMode [string trim $::env(SHEAR40_MODE)]
}
set outputDir "output-mp-$expElementMode"

# =============================================================================
# Model settings
# =============================================================================
set Nstories 40
set m 1.0
set kbottom 900
set ktop 600
set ExpEleTag 20

# MPI rank that owns the experimental element (OpenSeesSP main domain = rank 0)
set expRank 0

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

proc defineAnalyticalModel {Nstories kbottom ktop} {
    for {set i 1} {$i <= $Nstories} {incr i} {
        addNumericalSpring $i $Nstories $kbottom $ktop
    }
    printElementAnalytical $Nstories
}

proc setupLocalSite {ExpEleTag Nstories kbottom ktop} {
    global expSiteTag

    set k [storyStiff $ExpEleTag $Nstories $kbottom $ktop]
    set expMatTag 9001
    set expCtrlTag 9000
    set expSetupTag 9000
    set expSiteTag 9000

    uniaxialMaterial Elastic $expMatTag $k
    expControlPoint 9001  $ExpEleTag disp
    expControlPoint 9002  $ExpEleTag disp 1 force
    expControl SimUniaxialMaterials $expCtrlTag $expMatTag
    expSetup OneActuator $expSetupTag -control $expCtrlTag 1 -sizeTrialOut 1 1
    expSite LocalSite $expSiteTag $expSetupTag

    return $k
}

proc defineLocalExperiment {ExpEleTag Nstories kbottom ktop} {
    global expSiteTag

    set k [setupLocalSite $ExpEleTag $Nstories $kbottom $ktop]

    for {set i 1} {$i <= $Nstories} {incr i} {
        if {$i == $ExpEleTag} {
            expElement twoNodeLink $i [expr {$i - 1}] $i -dir 1 -site $expSiteTag -initStif $k
            printElementLocal $i $k
        } else {
            addNumericalSpring $i $Nstories $kbottom $ktop
        }
    }
}

# --- OpenSeesMP manual partitioning (instead of partition command) ---

proc evenStoryPartitions {Nstories numP} {
    set base [expr {$Nstories / $numP}]
    set rem [expr {$Nstories % $numP}]
    set parts {}
    set story 1
    for {set p 0} {$p < $numP} {incr p} {
        set cnt [expr {$p < $rem ? $base + 1 : $base}]
        if {$cnt <= 0} {
            lappend parts [list $p 0 0]
            continue
        }
        set first $story
        set last [expr {$story + $cnt - 1}]
        set story [expr {$last + 1}]
        lappend parts [list $p $first $last]
    }
    return $parts
}

proc swapExpBandToRank {parts ExpEleTag expRank} {
    set expIdx -1
    set targetIdx -1
    set idx 0
    foreach part $parts {
        lassign $part p first last
        if {$last > 0 && $first <= $ExpEleTag && $ExpEleTag <= $last} {
            set expIdx $idx
        }
        if {$p == $expRank} {
            set targetIdx $idx
        }
        incr idx
    }
    if {$expIdx < 0} {
        puts "ERROR ExpEleTag $ExpEleTag is not covered by the even story split"
        exit 1
    }
    if {$targetIdx < 0 || $expIdx == $targetIdx} {
        return $parts
    }

    set expPart [lindex $parts $expIdx]
    set targetPart [lindex $parts $targetIdx]
    lassign $expPart ep ef el
    lassign $targetPart tp tf tl
    set parts [lreplace $parts $expIdx $expIdx [list $ep $tf $tl]]
    set parts [lreplace $parts $targetIdx $targetIdx [list $tp $ef $el]]
    return $parts
}

proc computeStoryPartitions {Nstories numP expElementMode ExpEleTag expRank} {
    set parts [evenStoryPartitions $Nstories $numP]
    if {$expElementMode eq "analytical"} {
        return $parts
    }
    return [swapExpBandToRank $parts $ExpEleTag $expRank]
}

proc rankForStory {parts story} {
    foreach part $parts {
        lassign $part p first last
        if {$last > 0 && $first <= $story && $story <= $last} {
            return $p
        }
    }
    return -1
}

proc partitionForRank {pid parts} {
    foreach part $parts {
        lassign $part p first last
        if {$p == $pid} {
            return [list $first $last]
        }
    }
    return [list 0 0]
}

proc skipNumericalSpring {expElementMode ExpEleTag i pid expRank} {
    if {$pid != $expRank} {
        return 0
    }
    if {$expElementMode eq "local" && $i == $ExpEleTag} {
        return 1
    }
    return 0
}

# --- model generation ---

set numP [getNP]
set pid [getPID]

if {$pid == 0} {
    resetOutputDir $outputDir
}

wipe
if {$pid == 0} {
    logFile [file join $outputDir ShearBuilding40MP_${expElementMode}.log]
}

model BasicBuilder -ndm 1 -ndf 1

set parts [computeStoryPartitions $Nstories $numP $expElementMode $ExpEleTag $expRank]
lassign [partitionForRank $pid $parts] firstStory lastStory

if {$pid == 0} {
    printRunHeader "OpenSeesMPFresco"
    if {$expElementMode ne "analytical"} {
        printPartitionExp $ExpEleTag
    }
    puts "partition: MP manual map (numP=$numP expRank=$expRank) $parts"
}

if {$lastStory == 0} {
    puts "partition: rank $pid (no local stories)"
} else {
    set firstNode [expr {$firstStory - 1}]
    set lastNode $lastStory
    puts "partition: rank $pid stories $firstStory-$lastStory nodes $firstNode-$lastNode"
}

if {$firstStory == 1} {
    node 0 0
    fix 0 1
}

if {$lastStory > 0} {
    # Tie nodes include firstStory-1; mass only on owned stories firstStory..lastStory
    for {set i [expr {$firstStory - 1}]} {$i <= $lastStory} {incr i} {
        if {$i == 0} {
            continue
        }
        if {$i >= $firstStory} {
            node $i 0 -mass $m
        } else {
            node $i 0
        }
    }
}

if {$pid == $expRank && $expElementMode ne "analytical"} {
    global expSiteTag
    loadPackage OpenFrescoTcl
    printOpenFrescoVersion
    if {$expElementMode eq "local"} {
        set k [setupLocalSite $ExpEleTag $Nstories $kbottom $ktop]
        expElement twoNodeLink $ExpEleTag [expr {$ExpEleTag - 1}] $ExpEleTag \
            -dir 1 -site $expSiteTag -initStif $k
        printElementLocal $ExpEleTag $k
    }
}

if {$lastStory > 0} {
    for {set i $firstStory} {$i <= $lastStory} {incr i} {
        if {[skipNumericalSpring $expElementMode $ExpEleTag $i $pid $expRank]} {
            continue
        }
        addNumericalSpring $i $Nstories $kbottom $ktop
    }
}

if {$pid == 0 && $expElementMode eq "analytical"} {
    printElementAnalytical $Nstories
}

timeSeries Path 1 -filePath elcentro.txt -dt $dt -factor 1.0
pattern UniformExcitation 2 1 -accel 1

set alphaM [expr {$zeta * 2.0 * $w1 * $w2 / ($w1 + $w2)}]
set betaK [expr {$zeta * 2.0 / ($w1 + $w2)}]
rayleigh $alphaM 0.0 $betaK 0.0
if {$pid == 0} {
    printRayleighDamping $alphaM $betaK
}

# integrator -> test -> algorithm -> numberer -> constraints -> system -> analysis
applyAnalysisScheme $analysisScheme
numberer ParallelRCM
constraints Plain
system Mumps
set solverLabel "Mumps+ParallelRCM"
analysis Transient

if {$pid == 0} {
    printModelBuilt "OpenSeesMPFresco" $solverLabel "expRank=$expRank"
}

if {$lastStory > 0} {
    for {set i [expr {$firstStory - 1}]} {$i <= $lastStory} {incr i} {
        recorder Node -file [file join $outputDir node_${i}_disp.out] -time -node $i -dof 1 disp
    }
}

set recorderRank $expRank
if {$expElementMode eq "analytical"} {
    set recorderRank [rankForStory $parts $ExpEleTag]
}

if {$pid == $recorderRank} {
    recorder Element -file [file join $outputDir Elmt_Frc.out] -time -ele $ExpEleTag forces
    if {$expElementMode ne "analytical"} {
        recorder Element -file [file join $outputDir Elmt_ctrlDsp.out] -time -ele $ExpEleTag ctrlDisp
    }
}

record
if {$pid == 0} {
    printAnalysisStart
}

set ok [analyze $nSteps $dt]
if {$pid == 0} {
    printAnalysisDone $ok
}

if {$pid == $expRank && $expElementMode ne "analytical"} {
    wipeExp
}
wipe
exit
