# 40-story 1-D shear building with a local experimental element (OpenSeesFresco)
#
# Sequential reference case — no MPI partitioning.
# Run:
#   OpenSeesFresco.exe ShearBuilding40.tcl
#
# References:
#   twoNodeLink -> OpenFresco EXAMPLES/OneBayFrame/OpenSees/OneBayFrame_Local.tcl
#   generic     -> OpenFresco EXAMPLES/ThreeStoryBuilding/ThreeStoryBuilding_Master.tcl

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

set outputDir output

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
        ShearBuilding40_*.log
        ShearBuilding40SP_*.log
        ShearBuilding40MP_*.log
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

proc addNumericalSpring {i Nstories kbottom ktop} {
    set k [storyStiff $i $Nstories $kbottom $ktop]
    uniaxialMaterial Elastic $i $k
    element zeroLength $i [expr {$i - 1}] $i -mat $i -dir 1 -doRayleigh
}

proc defineTwoNodeLinkExperiment {ExpEleTag Nstories kbottom ktop} {
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

    for {set i 1} {$i <= $Nstories} {incr i} {
        if {$i == $ExpEleTag} {
            expElement twoNodeLink $i [expr {$i - 1}] $i -dir 1 -site $expSiteTag -initStif $k
            puts "twoNodeLink element $i (k=$k) between nodes [expr {$i - 1}] and $i"
        } else {
            addNumericalSpring $i $Nstories $kbottom $ktop
        }
    }
}

proc defineGenericExperiment {ExpEleTag Nstories kbottom ktop} {
    global expSiteTag

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
    set expSiteTag 9000

    uniaxialMaterial Elastic $expMatTag1 $kA
    uniaxialMaterial Elastic $expMatTag2 $k22
    uniaxialMaterial Elastic $expMatTag3 $kB

    expControlPoint 9001  1 disp 2 disp 3 disp
    expControlPoint 9002  1 disp 2 disp 3 disp  1 force 2 force 3 force
    expControl SimUniaxialMaterials $expCtrlTag $expMatTag1 $expMatTag2 $expMatTag3
    expSetup NoTransformation $expSetupTag -control $expCtrlTag -dof 1 2 3 -sizeTrialOut 3 3
    expSite LocalSite $expSiteTag $expSetupTag

    for {set i 1} {$i <= $Nstories} {incr i} {
        if {$i == $ExpEleTag || $i == [expr {$ExpEleTag + 1}]} {
            continue
        }
        addNumericalSpring $i $Nstories $kbottom $ktop
    }

    expElement generic $ExpEleTag -node $nLo $nMid $nHi -dof 1 -dof 1 -dof 1 -site $expSiteTag -initStif {*}$kExp
    puts "generic element $ExpEleTag on nodes $nLo $nMid $nHi (replaces springs $ExpEleTag and [expr {$ExpEleTag + 1}])"
}

# --- model generation ---

ensureOutputDir $outputDir
cleanupStaleOutput $outputDir

wipe
logFile [file join $outputDir ShearBuilding40_${expElementMode}.log]
model BasicBuilder -ndm 1 -ndf 1

loadPackage OpenFrescoTcl
puts "OpenFresco package version = [packageVersion]"

node 0 0
fix 0 1
for {set i 1} {$i <= $Nstories} {incr i} {
    node $i 0 -mass $m
}

switch -exact $expElementMode {
    twoNodeLink {
        defineTwoNodeLinkExperiment $ExpEleTag $Nstories $kbottom $ktop
    }
    generic {
        defineGenericExperiment $ExpEleTag $Nstories $kbottom $ktop
    }
    default {
        puts "ERROR unknown expElementMode: $expElementMode (use twoNodeLink or generic)"
        exit 1
    }
}

timeSeries Path 1 -filePath elcentro.txt -dt $dt -factor 1.0
pattern UniformExcitation 1 1 -accel 1

set alphaM [expr {$zeta * 2.0 * $w1 * $w2 / ($w1 + $w2)}]
set betaK [expr {$zeta * 2.0 / ($w1 + $w2)}]
rayleigh $alphaM $betaK 0.0 0.0
puts "Rayleigh damping: zeta=$zeta at w1=$w1 w2=$w2 rad/s (alphaM=$alphaM betaK=$betaK)"

system BandGeneral
numberer RCM
constraints Plain
applyAnalysisScheme $analysisScheme
analysis Transient

puts "Model built (mode=$expElementMode ExpEleTag=$ExpEleTag analysisScheme=$analysisScheme solver=BandGeneral)"

for {set i 0} {$i <= $Nstories} {incr i} {
    recorder Node -file [file join $outputDir node_${i}_disp.out] -time -node $i -dof 1 disp
}
recorder Element -file [file join $outputDir Elmt_Frc.out] -time -ele $ExpEleTag forces
recorder Element -file [file join $outputDir Elmt_ctrlDsp.out] -time -ele $ExpEleTag ctrlDisp

record
puts "Starting transient analysis ($nSteps steps, dt=$dt s)"

set ok [analyze $nSteps $dt]
if {$ok != 0} {
    puts "WARNING analyze failed at time [getTime] (return code $ok)"
} else {
    puts "Finished transient analysis ($nSteps steps)"
}

wipeExp
wipe
exit
