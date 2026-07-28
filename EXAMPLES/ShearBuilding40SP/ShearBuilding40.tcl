# 40-story 1-D shear building with a local experimental element (OpenSeesFresco)
#
# Sequential reference case — no MPI partitioning.
# Run:
#   OpenSeesFresco.exe ShearBuilding40.tcl
#
# References:
#   local -> OpenFresco EXAMPLES/OneBayFrame/OpenSees/OneBayFrame_Local.tcl

# =============================================================================
# Example mode
# =============================================================================
# analytical  - all stories are numerical zeroLength springs (no OpenFresco)
# local       - one story replaced by expElement twoNodeLink
set expElementMode analytical

if {[info exists ::env(SHEAR40_MODE)] && [string length [string trim $::env(SHEAR40_MODE)]] > 0} {
    set expElementMode [string trim $::env(SHEAR40_MODE)]
}
set outputDir "output-$expElementMode"

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

# --- model generation ---

resetOutputDir $outputDir

wipe
logFile [file join $outputDir ShearBuilding40_${expElementMode}.log]
model BasicBuilder -ndm 1 -ndf 1
printRunHeader "OpenSeesFresco"

if {$expElementMode ne "analytical"} {
    loadPackage OpenFrescoTcl
    printOpenFrescoVersion
}

node 0 0
fix 0 1
for {set i 1} {$i <= $Nstories} {incr i} {
    node $i 0 -mass $m
}

switch -exact $expElementMode {
    analytical {
        defineAnalyticalModel $Nstories $kbottom $ktop
    }
    local {
        defineLocalExperiment $ExpEleTag $Nstories $kbottom $ktop
    }
    default {
        puts "ERROR unknown expElementMode: $expElementMode (use analytical or local)"
        exit 1
    }
}

timeSeries Path 1 -filePath elcentro.txt -dt $dt -factor 1.0
pattern UniformExcitation 2 1 -accel 1

set alphaM [expr {$zeta * 2.0 * $w1 * $w2 / ($w1 + $w2)}]
set betaK [expr {$zeta * 2.0 / ($w1 + $w2)}]
rayleigh $alphaM 0.0 $betaK 0.0
printRayleighDamping $alphaM $betaK

# integrator -> test -> algorithm -> numberer -> constraints -> system -> analysis
applyAnalysisScheme $analysisScheme
numberer RCM
constraints Plain
system BandGeneral
analysis Transient

printModelBuilt "OpenSeesFresco" "BandGeneral+RCM"

for {set i 0} {$i <= $Nstories} {incr i} {
    recorder Node -file [file join $outputDir node_${i}_disp.out] -time -node $i -dof 1 disp
}
recorder Element -file [file join $outputDir Elmt_Frc.out] -time -ele $ExpEleTag forces
if {$expElementMode ne "analytical"} {
    recorder Element -file [file join $outputDir Elmt_ctrlDsp.out] -time -ele $ExpEleTag ctrlDisp
}

record
printAnalysisStart

set ok [analyze $nSteps $dt]
printAnalysisDone $ok

if {$expElementMode ne "analytical"} {
    wipeExp
}
wipe
exit
