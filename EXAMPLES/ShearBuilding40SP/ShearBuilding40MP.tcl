# 40-story 1-D shear building with a local experimental element (OpenSeesMPFresco)
#
# OpenSeesMP manual partitioning analogue of ShearBuilding40SP.tcl:
#   - all MPI ranks run the Tcl interpreter on a partial domain
#   - interface story nodes are replicated on adjacent ranks (same node tag)
#   - use ParallelRCM + Mumps (not OpenSeesSP partition / RCM)
#   - experimental element forced onto expRank (default 0): even story split, then
#     swap the band containing ExpEleTag onto expRank (OpenSeesSP pins only the
#     exp element to main-domain rank 0, not the whole lower building)
#
# Run with MPI, e.g.:
#   mpiexec -n 4 OpenSeesMPFresco.exe ShearBuilding40MP.tcl
#
# References:
#   twoNodeLink -> OpenFresco EXAMPLES/OneBayFrame/OpenSees/OneBayFrame_Local.tcl
#   generic     -> OpenFresco EXAMPLES/ThreeStoryBuilding/ThreeStoryBuilding_Master.tcl
#   MP pattern  -> EXAMPLES/ParallelModelMP/exampleMP.tcl

# =============================================================================
# Example mode
# =============================================================================
set expElementMode twoNodeLink

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
set analysisScheme Newmark

set nSteps 3120
set dt 0.02

set outputDir output-mp

# =============================================================================

proc storyStiff {i Nstories kbottom ktop} {
    return [expr {$kbottom + ($ktop-$kbottom)*double($i-1)/double($Nstories-1)}]
}

proc ensureOutputDir {dir} {
    if {[file exists $dir]} {
        foreach f [glob -nocomplain [file join $dir *]] {
            catch { file delete -force $f }
        }
    } else {
        file mkdir $dir
    }
}

proc cleanupStaleOutput {outputDir} {
    set outNorm [file normalize [file join [pwd] $outputDir]]
    set outPrefix "${outNorm}[file separator]"
    foreach pat {
        Node_Dsp.out*
        Elmt_Frc.out*
        Elmt_ctrlDsp.out*
        node_*_disp.out*
        ShearBuilding40MP_*.log
        ShearBuilding40SP_*.log
        run_output.txt
        run_HHTHybridSimulation_Linear.txt
    } {
        foreach f [glob -nocomplain $pat] {
            set fNorm [file normalize $f]
            if {[string first $outPrefix $fNorm] == 0} {
                continue
            }
            catch { file delete -force $f }
        }
    }
}

proc applyAnalysisScheme {scheme} {
    switch -exact $scheme {
        NewmarkExplicit {
            algorithm Linear
            integrator NewmarkExplicit 0.5
        }
        AlphaOSGeneralized {
            algorithm Linear
            integrator AlphaOSGeneralized 0.9
        }
        Newmark {
            test EnergyIncr 1.0e-10 20 0
            algorithm KrylovNewton
            integrator Newmark 0.5 0.25
        }
        default {
            puts "ERROR unknown analysisScheme: $scheme (use NewmarkExplicit, AlphaOSGeneralized, or Newmark)"
            exit 1
        }
    }
}

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

# Swap story bands so the partition containing ExpEleTag moves onto expRank.
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

# Extend expRank through ExpEleTag+1 in generic mode (do not split the 3-node element).
proc extendGenericExpPartition {parts expRank ExpEleTag Nstories} {
    set idx 0
    foreach part $parts {
        lassign $part p first last
        if {$p == $expRank && $last > 0} {
            if {$last < [expr {$ExpEleTag + 1}]} {
                if {[expr {$ExpEleTag + 1}] > $Nstories} {
                    puts "ERROR generic ExpEleTag $ExpEleTag needs node [expr {$ExpEleTag + 1}] but Nstories=$Nstories"
                    exit 1
                }
                set parts [lreplace $parts $idx $idx [list $p $first [expr {$ExpEleTag + 1}]]]
            }
            break
        }
        incr idx
    }

    set idx 0
    foreach part $parts {
        lassign $part p first last
        if {$p != $expRank && $last > 0 && $first <= [expr {$ExpEleTag + 1}] && $last >= [expr {$ExpEleTag + 1}]} {
            set newFirst [expr {$ExpEleTag + 2}]
            if {$newFirst > $last} {
                set parts [lreplace $parts $idx $idx [list $p 0 0]]
            } else {
                set parts [lreplace $parts $idx $idx [list $p $newFirst $last]]
            }
        }
        incr idx
    }
    return $parts
}

proc computeStoryPartitions {Nstories numP expElementMode ExpEleTag expRank} {
    set parts [evenStoryPartitions $Nstories $numP]
    set parts [swapExpBandToRank $parts $ExpEleTag $expRank]
    if {$expElementMode eq "generic"} {
        set parts [extendGenericExpPartition $parts $expRank $ExpEleTag $Nstories]
    }
    return $parts
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

proc addNumericalSpring {i Nstories kbottom ktop} {
    set k [storyStiff $i $Nstories $kbottom $ktop]
    uniaxialMaterial Elastic $i $k
    element zeroLength $i [expr {$i - 1}] $i -mat $i -dir 1 -doRayleigh
}

proc defineTwoNodeLinkExperiment {ExpEleTag Nstories kbottom ktop expSiteTag} {
    set k [storyStiff $ExpEleTag $Nstories $kbottom $ktop]
    set expMatTag 9001
    set expCtrlTag 9000
    set expSetupTag 9000

    uniaxialMaterial Elastic $expMatTag $k
    expControlPoint 9001  $ExpEleTag disp
    expControlPoint 9002  $ExpEleTag disp 1 force
    expControl SimUniaxialMaterials $expCtrlTag $expMatTag
    expSetup OneActuator $expSetupTag -control $expCtrlTag 1 -sizeTrialOut 1 1
    expSite LocalSite $expSiteTag $expSetupTag

    expElement twoNodeLink $ExpEleTag [expr {$ExpEleTag - 1}] $ExpEleTag -dir 1 -site $expSiteTag -initStif $k
    puts "twoNodeLink element $ExpEleTag (k=$k) between nodes [expr {$ExpEleTag - 1}] and $ExpEleTag"
}

proc defineGenericExperiment {ExpEleTag Nstories kbottom ktop expSiteTag} {
    if {$ExpEleTag < 2 || $ExpEleTag >= $Nstories} {
        puts "ERROR ExpEleTag must be between 2 and [expr {$Nstories - 1}] for generic element"
        exit 1
    }

    set nLo [expr {$ExpEleTag - 1}]
    set nMid $ExpEleTag
    set nHi [expr {$ExpEleTag + 1}]

    set kA [storyStiff $ExpEleTag $Nstories $kbottom $ktop]
    set kB [storyStiff [expr {$ExpEleTag + 1}] $Nstories $kbottom $ktop]
    set k22 [expr {$kA + $kB}]

    set kExp [list \
        $kA              [expr {-$kA}]  0.0 \
        [expr {-$kA}]    $k22           [expr {-$kB}] \
        0.0              [expr {-$kB}]  $kB]

    set expMatTag1 9001
    set expMatTag2 9002
    set expMatTag3 9003
    set expCtrlTag 9000
    set expSetupTag 9000

    uniaxialMaterial Elastic $expMatTag1 $kA
    uniaxialMaterial Elastic $expMatTag2 $k22
    uniaxialMaterial Elastic $expMatTag3 $kB

    expControlPoint 9001  1 disp 2 disp 3 disp
    expControlPoint 9002  1 disp 2 disp 3 disp  1 force 2 force 3 force
    expControl SimUniaxialMaterials $expCtrlTag $expMatTag1 $expMatTag2 $expMatTag3
    expSetup NoTransformation $expSetupTag -control $expCtrlTag -dof 1 2 3 -sizeTrialOut 3 3
    expSite LocalSite $expSiteTag $expSetupTag

    expElement generic $ExpEleTag -node $nLo $nMid $nHi -dof 1 -dof 1 -dof 1 -site $expSiteTag -initStif {*}$kExp
    puts "generic element $ExpEleTag on nodes $nLo $nMid $nHi (replaces springs $ExpEleTag and [expr {$ExpEleTag + 1}])"
}

# --- model generation ---

set numP [getNP]
set pid [getPID]

if {$pid == 0} {
    ensureOutputDir $outputDir
    cleanupStaleOutput $outputDir
}

wipe
if {$pid == 0} {
    logFile [file join $outputDir ShearBuilding40MP_${expElementMode}.log]
}

model BasicBuilder -ndm 1 -ndf 1

set parts [computeStoryPartitions $Nstories $numP $expElementMode $ExpEleTag $expRank]
lassign [partitionForRank $pid $parts] firstStory lastStory

if {$pid == 0} {
    puts "MP partition map (numP=$numP, expRank=$expRank): $parts"
}

if {$lastStory == 0} {
    puts "rank $pid: no local stories"
} else {
    set firstNode [expr {$firstStory - 1}]
    set lastNode $lastStory
    puts "rank $pid: stories $firstStory-$lastStory, nodes $firstNode-$lastNode"
}

if {$firstStory == 1} {
    node 0 0
    fix 0 1
}

if {$lastStory > 0} {
    for {set i [expr {$firstStory - 1}]} {$i <= $lastStory} {incr i} {
        if {$i == 0} {
            continue
        }
        node $i 0 -mass $m
    }
}

if {$pid == $expRank} {
    loadPackage OpenFrescoTcl
    puts "rank $pid: OpenFresco package version = [packageVersion]"
    set expSiteTag 9000
    switch -exact $expElementMode {
        twoNodeLink {
            defineTwoNodeLinkExperiment $ExpEleTag $Nstories $kbottom $ktop $expSiteTag
        }
        generic {
            defineGenericExperiment $ExpEleTag $Nstories $kbottom $ktop $expSiteTag
        }
        default {
            puts "ERROR unknown expElementMode: $expElementMode (use twoNodeLink or generic)"
            exit 1
        }
    }
}

if {$lastStory > 0} {
    for {set i $firstStory} {$i <= $lastStory} {incr i} {
        if {$pid == $expRank} {
            if {$expElementMode eq "twoNodeLink" && $i == $ExpEleTag} {
                continue
            }
            if {$expElementMode eq "generic" && ($i == $ExpEleTag || $i == [expr {$ExpEleTag + 1}])} {
                continue
            }
        }
        addNumericalSpring $i $Nstories $kbottom $ktop
    }
}

timeSeries Path 1 -filePath elcentro.txt -dt $dt -factor 1.0
pattern UniformExcitation 1 1 -accel 1

set alphaM [expr {$zeta * 2.0 * $w1 * $w2 / ($w1 + $w2)}]
set betaK [expr {$zeta * 2.0 / ($w1 + $w2)}]
rayleigh $alphaM $betaK 0.0 0.0
if {$pid == 0} {
    puts "Rayleigh damping: zeta=$zeta at w1=$w1 w2=$w2 rad/s (alphaM=$alphaM betaK=$betaK)"
}

system Mumps
numberer ParallelRCM
constraints Plain
applyAnalysisScheme $analysisScheme
analysis Transient

if {$pid == 0} {
    puts "Model built (mode=$expElementMode ExpEleTag=$ExpEleTag expRank=$expRank analysisScheme=$analysisScheme)"
}

if {$lastStory > 0} {
    for {set i [expr {$firstStory - 1}]} {$i <= $lastStory} {incr i} {
        recorder Node -file [file join $outputDir node_${i}_disp.out] -time -node $i -dof 1 disp
    }
}

if {$pid == $expRank} {
    recorder Element -file [file join $outputDir Elmt_Frc.out] -time -ele $ExpEleTag forces
    recorder Element -file [file join $outputDir Elmt_ctrlDsp.out] -time -ele $ExpEleTag ctrlDisp
}

record
if {$pid == 0} {
    puts "Starting transient analysis ($nSteps steps, dt=$dt s)"
}

set ok [analyze $nSteps $dt]
if {$pid == 0} {
    if {$ok != 0} {
        puts "WARNING analyze failed at time [getTime] (return code $ok)"
    } else {
        puts "Finished transient analysis ($nSteps steps)"
    }
}

if {$pid == $expRank} {
    wipeExp
}
wipe
exit
