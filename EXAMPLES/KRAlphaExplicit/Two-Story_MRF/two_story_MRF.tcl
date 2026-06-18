# two_story_MRF.tcl
# Two-Story MRF (Kolay & Ricles) — Tcl counterpart to two_story_MRF.py
#
# Usage (from this directory):
#   /path/to/OpenSees two_story_MRF.tcl
#   /path/to/OpenSees two_story_MRF.tcl KRAlphaExplicit 0.5
#   /path/to/OpenSees two_story_MRF.tcl CudaKRAlpha 1.0 3.0 -incrementalAccel
#   /path/to/OpenSees two_story_MRF.tcl CudaKRAlpha 0.5 3.0 -massMode 1  # element lumped M
#   /path/to/OpenSees two_story_MRF.tcl CudaKRAlpha 0.5 3.0 -massMode 2  # nodal lumped M
#   /path/to/OpenSees two_story_MRF.tcl Newmark
#   /path/to/OpenSees two_story_MRF.tcl NewmarkCPU
#
# Recorder order matches KRAlphaSparse / two_story_MRF.py:
#   analysis Transient -> mkdir -> results.txt + logFile -> recorders -> analyze -> remove recorders

set scriptDir [file dirname [info script]]
cd $scriptDir

# --- analysis controls (override via argv: method [rho] [scale]) ---
set integratorMethod KRAlphaExplicit
set rho 0.5
set scaleFactor 3.0
set dtAnalysis 0.005
set freeVibrationSeconds 5.0
set gmFile [file join $scriptDir ground_motions RSN960_NORTHR_LOS270.AT2]

if {$argc >= 1} {
    set integratorMethod [lindex $argv 0]
}

proc _argvIsNumeric {s} {
    return [string is double -strict $s]
}

set flagStart 1
if {$argc >= 2 && [_argvIsNumeric [lindex $argv 1]]} {
    set rho [lindex $argv 1]
    set flagStart 2
    if {$argc >= 3 && [_argvIsNumeric [lindex $argv 2]]} {
        set scaleFactor [lindex $argv 2]
        set flagStart 3
    }
}

# Optional flags after [rho] [scaleFactor] (case-insensitive).
# massMode: 0 = consistent (-cMass), 1 = element lumped, 2 = nodal lumped.
set massMode 0
set numbererType RCM
set useIncrementalAccel 0
set useAlphaCloseCheck 0
set systemOverride ""
set cudssPrecision ""
set cudssIrNSteps 0
for {set i $flagStart} {$i < $argc} {incr i} {
    set flag [lindex $argv $i]
    set flagLower [string tolower $flag]
    switch -exact $flagLower {
        -massmode {
            incr i
            if {$i >= $argc} {
                puts stderr "ERROR: -massMode requires 0, 1, or 2"
                exit 1
            }
            set massMode [lindex $argv $i]
            if {![string is integer -strict $massMode] || $massMode < 0 || $massMode > 2} {
                puts stderr "ERROR: -massMode must be 0 (consistent), 1 (element lumped), or 2 (nodal lumped)"
                exit 1
            }
        }
        -numberer {
            incr i
            if {$i >= $argc} {
                puts stderr "ERROR: -numberer requires Plain, RCM, or AMD"
                exit 1
            }
            set numbererType [lindex $argv $i]
            if {$numbererType ni {Plain RCM AMD}} {
                puts stderr "ERROR: -numberer must be Plain, RCM, or AMD"
                exit 1
            }
        }
        -incrementalaccel { set useIncrementalAccel 1 }
        -alphaclosecheck { set useAlphaCloseCheck 1 }
        -system {
            incr i
            if {$i >= $argc} {
                puts stderr "ERROR: -system requires an argument (e.g. UmfPack, SuperLU, CuDSS)"
                exit 1
            }
            set systemOverride [lindex $argv $i]
        }
        -cudssprecision {
            incr i
            if {$i >= $argc} {
                puts stderr "ERROR: -cudssPrecision requires an argument (dFFI)"
                exit 1
            }
            set cudssPrecision [lindex $argv $i]
        }
        -cudssirnsteps {
            incr i
            if {$i >= $argc} {
                puts stderr "ERROR: -cudssIrNSteps requires an integer argument"
                exit 1
            }
            set cudssIrNSteps [lindex $argv $i]
            if {![string is integer -strict $cudssIrNSteps] || $cudssIrNSteps < 0} {
                puts stderr "ERROR: -cudssIrNSteps must be a non-negative integer"
                exit 1
            }
        }
        default {
            puts stderr "Unknown flag: [lindex $argv $i]"
            puts stderr "Optional flags: -massMode 0|1|2 -numberer Plain|RCM|AMD -incrementalAccel -alphaCloseCheck -system SOE -cudssPrecision dFFI -cudssIrNSteps N"
            exit 1
        }
    }
}

proc formatParamsLabel {paramsList} {
    set parts {}
    foreach p $paramsList {
        if {[string is double -strict $p]} {
            lappend parts $p
        } else {
            lappend parts "'$p'"
        }
    }
    return [format "\[%s\]" [join $parts {, }]]
}

proc appendIntegratorFlags {paramsList} {
    global useIncrementalAccel useAlphaCloseCheck
    if {$useIncrementalAccel} { lappend paramsList -incrementalAccel }
    if {$useAlphaCloseCheck} { lappend paramsList -alphaCloseCheck }
    return $paramsList
}

switch -exact $integratorMethod {
    Newmark {
        set integratorParams [list 0.5 0.25]
        set maxIter 25
        set pFlag 0
        set algo Newton
        set linearSOE CuDSS
        set newmarkCpu 0
    }
    NewmarkCPU {
        set integratorMethod Newmark
        set integratorParams [list 0.5 0.25]
        set maxIter 25
        set pFlag 0
        set algo Newton
        set linearSOE FullGeneral
        set newmarkCpu 1
    }
    KRAlphaExplicit {
        set integratorParams [list $rho]
        set maxIter 1
        set pFlag 5
        set algo Linear
        set linearSOE FullGeneral
        set newmarkCpu 0
    }
    MKRAlphaExplicit {
        set integratorParams [list $rho]
        set maxIter 1
        set pFlag 5
        set algo Linear
        set linearSOE FullGeneral
        set newmarkCpu 0
    }
    CudaKRAlpha {
        set integratorParams [appendIntegratorFlags [list $rho]]
        set maxIter 1
        set pFlag 5
        set algo Linear
        set linearSOE CuDSS
        set newmarkCpu 0
    }
    CudaMKRAlpha {
        set integratorParams [appendIntegratorFlags [list $rho]]
        set maxIter 1
        set pFlag 5
        set algo Linear
        set linearSOE CuDSS
        set newmarkCpu 0
    }
    KRAlphaExplicitMultiSOE {
        set integratorParams [appendIntegratorFlags [list $rho]]
        set maxIter 1
        set pFlag 5
        set algo Linear
        set linearSOE CuDSS
        set newmarkCpu 0
    }
    MKRAlphaExplicitMultiSOE {
        set integratorParams [appendIntegratorFlags [list $rho]]
        set maxIter 1
        set pFlag 5
        set algo Linear
        set linearSOE CuDSS
        set newmarkCpu 0
    }
    default {
        puts stderr "Unknown integrator: $integratorMethod"
        puts stderr "Use: Newmark | NewmarkCPU | KRAlphaExplicit | MKRAlphaExplicit | KRAlphaExplicitMultiSOE | MKRAlphaExplicitMultiSOE | CudaKRAlpha | CudaMKRAlpha"
        exit 1
    }
}

if {$systemOverride ne ""} {
    if {$integratorMethod in {Newmark KRAlphaExplicitMultiSOE MKRAlphaExplicitMultiSOE CudaKRAlpha CudaMKRAlpha}} {
        set linearSOE $systemOverride
    } else {
        puts stderr "WARNING two_story_MRF.tcl: -system $systemOverride ignored for $integratorMethod"
    }
}

if {$integratorMethod in {KRAlphaExplicit MKRAlphaExplicit} && ($useIncrementalAccel || $useAlphaCloseCheck)} {
    puts stderr "WARNING two_story_MRF.tcl: -incrementalAccel/-alphaCloseCheck ignored for dense KRAlphaExplicit/MKRAlphaExplicit"
}

# Output tree follows massMode: results / results_1 / results_2 (figures_* from run_integrators).
if {$massMode == 0} {
    set resultsSubdir results
} else {
    set resultsSubdir "results_$massMode"
}

set folderSystemLabel ""
if {$linearSOE eq "CuDSS" && $cudssPrecision eq "dFFI"} {
    if {$cudssIrNSteps > 0} {
        set folderSystemLabel CuDSS_dFFI_ir$cudssIrNSteps
    } else {
        set folderSystemLabel CuDSS_dFFI
    }
} elseif {$systemOverride ne "" && $systemOverride ne "CuDSS"} {
    set folderSystemLabel $systemOverride
}

proc buildOutputFolderName {integratorMethod integratorParams folderSystemLabel newmarkCpu numbererType} {
    set parts [list $integratorMethod]
    if {$numbererType ne "RCM"} {
        lappend parts $numbererType
    }
    if {$newmarkCpu} {
        lappend parts FullGeneral
    } elseif {$folderSystemLabel ne ""} {
        lappend parts $folderSystemLabel
    }
    set paramsLabel [formatParamsLabel $integratorParams]
    return [format "%s_params-%s" [join $parts _] $paramsLabel]
}

set outputFolder [file join $scriptDir $resultsSubdir [buildOutputFolderName $integratorMethod $integratorParams $folderSystemLabel $newmarkCpu $numbererType]]

# --- units (match two_story_MRF.py) ---
set kN 1.0
set meter 1.0
set sec 1.0
set kip 4.44822
set inch 0.0254
set foot [expr 12.0 * $inch]
set lbf [expr $kip / 1000.0]
set gravity 9.80665
set kg [expr ($kN / 1000.0) * ($sec * $sec) / $meter]
set kPa [expr $kN / ($meter * $meter)]
set MPa [expr 1000.0 * $kPa]
set GPa [expr 1000.0 * $MPa]

set MONITOR_NODES {17 10 2}
set MONITOR_DOFS {1 2 3}

proc nodeLength {n1 n2} {
    set x1 [nodeCoord $n1 1]
    set y1 [nodeCoord $n1 2]
    set x2 [nodeCoord $n2 1]
    set y2 [nodeCoord $n2 2]
    return [expr hypot($x2 - $x1, $y2 - $y1)]
}

# Nodal lumped: half-length mass to each end node (same rule as lumped dispBeamColumn).
proc addLineLumpedMass {n1 n2 rho mxVar myVar} {
    upvar 1 $mxVar mx
    upvar 1 $myVar my
    set mHalf [expr 0.5 * $rho * [nodeLength $n1 $n2]]
    foreach n [list $n1 $n2] {
        if {![info exists mx($n)]} { set mx($n) 0.0 }
        if {![info exists my($n)]} { set my($n) 0.0 }
        set mx($n) [expr $mx($n) + $mHalf]
        set my($n) [expr $my($n) + $mHalf]
    }
}

proc assignNodalLumpedMasses {beamRho colRho leanRho mEps IrEps floorMass} {
    array set mx {}
    array set my {}
    foreach {n1 n2 rho} [list \
        1 2 $beamRho 2 3 $beamRho 3 4 $beamRho 4 5 $beamRho 5 6 $beamRho 6 7 $beamRho \
        9 10 $beamRho 10 11 $beamRho 11 12 $beamRho 12 13 $beamRho 13 14 $beamRho 14 15 $beamRho \
        15 17 $colRho 16 20 $colRho 17 21 $colRho 18 22 $colRho 19 9 $colRho 20 23 $colRho \
        21 24 $colRho 22 25 $colRho 23 26 $colRho 24 18 $colRho 25 27 $colRho 26 28 $colRho \
        27 29 $colRho 28 15 $colRho 29 30 $colRho 30 31 $colRho 31 32 $colRho 32 33 $colRho \
        33 7 $colRho 19 16 $leanRho 16 8 $leanRho] {
        addLineLumpedMass $n1 $n2 $rho mx my
    }
    foreach n {1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33} {
        if {![info exists mx($n)]} { set mx($n) 0.0 }
        if {![info exists my($n)]} { set my($n) 0.0 }
        set ux [expr $mx($n) + $mEps]
        set uy [expr $my($n) + $mEps]
        if {$n == 8 || $n == 16} { set ux [expr $ux + $floorMass] }
        mass $n $ux $uy $IrEps
    }
}

proc elementLineMass {rho} {
    global massMode
    if {$massMode == 2} {
        return [list]
    }
    return [list -mass $rho]
}

proc buildModel {} {
    global meter sec kg GPa MPa inch foot lbf gravity kN massMode numbererType
    if {$massMode == 0} {
        set elementMassOpts [list -cMass]
    } else {
        set elementMassOpts [list]
    }

    wipe
    model basic -ndm 2 -ndf 3

    set L [expr 6.0 * $meter]
    set H [expr 3.0 * $meter]
    set dLeaning [expr 0.2 * $L]
    set Ljoint_beam [expr $L / 12.0]
    set Hjoint_col [expr $H / 6.0]

    # Floor 2
    node 1  0.0           [expr 2 * $H]
    node 2  $Ljoint_beam  [expr 2 * $H]
    node 3  [expr 2 * $Ljoint_beam] [expr 2 * $H]
    node 4  [expr $L / 2.0]         [expr 2 * $H]
    node 5  [expr $L - 2 * $Ljoint_beam] [expr 2 * $H]
    node 6  [expr $L - $Ljoint_beam]     [expr 2 * $H]
    node 7  $L            [expr 2 * $H]
    node 8  [expr $L + $dLeaning] [expr 2 * $H]

    # Floor 1
    node 9  0.0           $H
    node 10 $Ljoint_beam  $H
    node 11 [expr 2 * $Ljoint_beam] $H
    node 12 [expr $L / 2.0]         $H
    node 13 [expr $L - 2 * $Ljoint_beam] $H
    node 14 [expr $L - $Ljoint_beam]     $H
    node 15 $L            $H
    node 16 [expr $L + $dLeaning] $H

    # Base
    node 17 0.0 0.0
    node 18 $L  0.0
    node 19 [expr $L + $dLeaning] 0.0

    # Left column interior
    node 20 0.0 [expr ($H - 2 * $Hjoint_col) / 2.0]
    node 21 0.0 [expr $H - 2 * $Hjoint_col]
    node 22 0.0 [expr $H - $Hjoint_col]
    node 23 0.0 [expr $H + $Hjoint_col]
    node 24 0.0 [expr $H + 2 * $Hjoint_col]
    node 25 0.0 [expr 2 * $H - 2 * $Hjoint_col]
    node 26 0.0 [expr 2 * $H - $Hjoint_col]

    # Right column interior
    node 27 $L [expr ($H - 2 * $Hjoint_col) / 2.0]
    node 28 $L [expr $H - 2 * $Hjoint_col]
    node 29 $L [expr $H - $Hjoint_col]
    node 30 $L [expr $H + $Hjoint_col]
    node 31 $L [expr $H + 2 * $Hjoint_col]
    node 32 $L [expr 2 * $H - 2 * $Hjoint_col]
    node 33 $L [expr 2 * $H - $Hjoint_col]

    fix 17 1 1 0
    fix 18 1 1 0
    fix 19 1 1 0

    rigidDiaphragm 1 8 4
    rigidDiaphragm 1 16 12

    set Lref [expr 0.1 * $meter]
    set mEps [expr 1.0 * $kg]
    set IrEps [expr $mEps * $Lref * $Lref]
    set floorMass [expr 50.97e3 * $kg]

    set Es [expr 200.0 * $GPa]
    set Fy [expr 345.0 * $MPa]
    set eta 0.01
    uniaxialMaterial Steel01 1 $Fy $Es $eta

    # W24x55 beams
    set d  [expr 23.6 * $inch]
    set tw [expr 0.395 * $inch]
    set bf [expr 7.01 * $inch]
    set tf [expr 0.505 * $inch]
    set beam_linear_density [expr (55.0 * $lbf / $foot) / $gravity]
    section WFSection2d 1 1 $d $tw $bf $tf 10 3

    # W14x120 columns
    set d  [expr 14.5 * $inch]
    set tw [expr 0.59 * $inch]
    set bf [expr 14.7 * $inch]
    set tf [expr 0.94 * $inch]
    set col_linear_density [expr (120.0 * $lbf / $foot) / $gravity]
    section WFSection2d 2 1 $d $tw $bf $tf 10 3

    set leanMass [expr 1.0e-3 * $kg / $meter]

    if {$massMode == 2} {
        assignNodalLumpedMasses $beam_linear_density $col_linear_density $leanMass $mEps $IrEps $floorMass
    } else {
        foreach n {1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33} {
            if {$n in {8 16}} { continue }
            mass $n $mEps $mEps $IrEps
        }
        mass 8  [expr $floorMass] $mEps $IrEps
        mass 16 [expr $floorMass] $mEps $IrEps
    }

    geomTransf Linear 1
    beamIntegration Legendre 1 1 2
    set beamMass [elementLineMass $beam_linear_density]
    element dispBeamColumn 1  1  2  1 1 {*}$elementMassOpts {*}$beamMass
    element dispBeamColumn 2  2  3  1 1 {*}$elementMassOpts {*}$beamMass
    element dispBeamColumn 3  3  4  1 1 {*}$elementMassOpts {*}$beamMass
    element dispBeamColumn 4  4  5  1 1 {*}$elementMassOpts {*}$beamMass
    element dispBeamColumn 5  5  6  1 1 {*}$elementMassOpts {*}$beamMass
    element dispBeamColumn 6  6  7  1 1 {*}$elementMassOpts {*}$beamMass
    element dispBeamColumn 9  9  10 1 1 {*}$elementMassOpts {*}$beamMass
    element dispBeamColumn 10 10 11 1 1 {*}$elementMassOpts {*}$beamMass
    element dispBeamColumn 11 11 12 1 1 {*}$elementMassOpts {*}$beamMass
    element dispBeamColumn 12 12 13 1 1 {*}$elementMassOpts {*}$beamMass
    element dispBeamColumn 13 13 14 1 1 {*}$elementMassOpts {*}$beamMass
    element dispBeamColumn 14 14 15 1 1 {*}$elementMassOpts {*}$beamMass

    geomTransf Linear 2
    beamIntegration Legendre 2 2 2
    set colMass [elementLineMass $col_linear_density]
    element dispBeamColumn 15 17 20 2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 16 20 21 2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 17 21 22 2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 18 22 9  2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 19 9  23 2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 20 23 24 2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 21 24 25 2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 22 25 26 2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 23 26 1  2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 24 18 27 2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 25 27 28 2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 26 28 29 2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 27 29 15 2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 28 15 30 2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 29 30 31 2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 30 31 32 2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 31 32 33 2 2 {*}$elementMassOpts {*}$colMass
    element dispBeamColumn 32 33 7  2 2 {*}$elementMassOpts {*}$colMass

    set A_leaning [expr 9.76e-2 * $meter * $meter]
    set I_leaning [expr 7.125e-4 * $meter * $meter * $meter * $meter]
    geomTransf PDelta 3
    set leanMassArg [elementLineMass $leanMass]
    element elasticBeamColumn 33 19 16 $A_leaning $Es $I_leaning 3 {*}$elementMassOpts {*}$leanMassArg
    element elasticBeamColumn 34 16 8  $A_leaning $Es $I_leaning 3 {*}$elementMassOpts {*}$leanMassArg

    # Gravity
    set Pfloor1 [expr 500.0 * $kN]
    set Pfloor2 [expr 500.0 * $kN]
    timeSeries Linear 1
    pattern Plain 1 1 {
        load 8  0.0 [expr -$Pfloor2] 0.0
        load 16 0.0 [expr -$Pfloor1] 0.0
    }

    wipeAnalysis
    constraints Transformation
    numberer $numbererType
    system FullGeneral
    test NormDispIncr 1.0e-6 10
    algorithm KrylovNewton
    integrator LoadControl [expr 1.0 / 10.0]
    analysis Static
    analyze 10
    loadConst -time 0.0
}

proc finalNorm {iters norms} {
    set n [llength $norms]
    if {$iters > 0 && $n >= $iters} {
        return [lindex $norms [expr {$iters - 1}]]
    }
    if {$n > 0} {
        return [lindex $norms end]
    }
    return 0.0
}

proc setupRayleighDamping {} {
    set evalues [eigen 2]
    set omega_i [expr sqrt([lindex $evalues 0])]
    set omega_j [expr sqrt([lindex $evalues 1])]
    set zeta_i 0.02
    set zeta_j 0.02
    set denom [expr $omega_i * $omega_i - $omega_j * $omega_j]
    set a0 [expr 2.0 * ($zeta_j * $omega_i - $zeta_i * $omega_j) * ($omega_j * $omega_i) / $denom]
    set a1 [expr 2.0 * ($zeta_i * $omega_i - $zeta_j * $omega_j) / $denom]
    rayleigh $a0 0.0 $a1 0.0
    foreach ele {1 2 5 6 9 10 13 14} {
        setElementRayleighDampingFactors $ele $a0 0.0 0.0 0.0
    }
}

proc runDynamicAnalysis {} {
    global scriptDir gmFile gravity scaleFactor dtAnalysis freeVibrationSeconds
    global integratorMethod integratorParams maxIter pFlag algo linearSOE cudssPrecision cudssIrNSteps numbererType
    global outputFolder MONITOR_NODES MONITOR_DOFS sec

    file mkdir $outputFolder
    set gmDat [file join $outputFolder gm.dat]
    source [file join $scriptDir .. .. verification ReadRecord.tcl]
    ReadRecord $gmFile $gmDat dtFile nPts

    timeSeries Path 2 -filePath $gmDat -dt $dtFile -factor $gravity
    pattern UniformExcitation 2 1 -accel 2 -fact $scaleFactor

    wipeAnalysis
    if {$linearSOE eq "CuDSS" && $cudssPrecision eq "dFFI"} {
        if {$cudssIrNSteps > 0} {
            eval system CuDSS -precision dFFI -irNSteps $cudssIrNSteps
            set systemLog "CuDSS -precision dFFI -irNSteps $cudssIrNSteps"
        } else {
            eval system CuDSS -precision dFFI
            set systemLog "CuDSS -precision dFFI"
        }
    } else {
        system $linearSOE
        set systemLog $linearSOE
    }
    constraints Transformation
    numberer $numbererType
    test NormUnbalance 1.0e-8 $maxIter $pFlag
    algorithm $algo
    eval integrator $integratorMethod {*}$integratorParams
    analysis Transient

    set resultsPath [file join $outputFolder results.txt]
    set resultsFd [open $resultsPath a+]
    set logPath [file join $outputFolder OpenSees.log]
    logFile $logPath -noEcho

    set currentTime [clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}]
    puts $resultsFd "$currentTime - Analysis STARTED."

    set freeVibration [expr $freeVibrationSeconds * $sec]
    set motionSteps [expr int(round($nPts * $dtFile / $dtAnalysis))]
    set freeSteps [expr int(round($freeVibration / $dtAnalysis))]
    set nSteps [expr $motionSteps + $freeSteps]
    set tFinal [expr $nSteps * $dtAnalysis]
    set gmLogName [file tail $gmDat]

    puts $resultsFd "$currentTime - Running $gmLogName (dt = $dtFile s, npts = $nPts) with dt_analysis = $dtAnalysis s, n_steps = $nSteps (t_end = $tFinal s), scale factor = $scaleFactor."
    puts $resultsFd "$currentTime - test NormUnbalance 1.0e-8 $maxIter $pFlag; algorithm $algo; system $systemLog; integrator $integratorMethod $integratorParams."
    close $resultsFd

    foreach resp {disp vel accel} {
        set outFile [file join $outputFolder ${resp}.out]
        recorder Node -file $outFile -time -node 17 10 2 -dof 1 2 3 $resp
    }
    set rayleighFile [file join $outputFolder rayleigh-forces.out]
    recorder Node -file $rayleighFile -time -node 17 10 2 -dof 1 2 3 rayleighForces
    set elemFile [file join $outputFolder element-1-forces.txt]
    recorder Element -file $elemFile -time -ele 1 localForce

    set ok 0
    set step 0
    set tWall0 [clock milliseconds]
    set convPath [file join $outputFolder convergence.dat]
    set convFd [open $convPath w]
    puts $convFd "# time iters final_norm"
    while {$ok == 0 && $step < $nSteps} {
        set ok [analyze 1 $dtAnalysis]
        if {$ok != 0 && $maxIter > 1} {
            puts "$algo failed .. trying ModifiedNewton -initial"
            test NormUnbalance 1.0e-8 [expr $maxIter * 100] $pFlag
            algorithm ModifiedNewton -initial
            set ok [analyze 1 $dtAnalysis]
            if {$ok == 0} {
                puts "that worked .. back to $algo"
                test NormUnbalance 1.0e-8 $maxIter $pFlag
                algorithm $algo
            }
        }
        if {$ok == 0} {
            set iters [testIter]
            set norms [testNorms]
            set tStep [getTime]
            puts $convFd "$tStep $iters [finalNorm $iters $norms]"
        }
        incr step
    }
    close $convFd

    set wallTime [expr {([clock milliseconds] - $tWall0) / 1000.0}]
    set timingPath [file join $outputFolder timing.txt]
    set timingFd [open $timingPath w]
    set currentTime [clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}]
    puts $timingFd "# $currentTime"
    puts $timingFd "label transient"
    puts $timingFd "wall_time_s $wallTime"
    close $timingFd

    set resultsFd [open $resultsPath a+]
    set currentTime [clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}]
    if {$ok == 0} {
        puts $resultsFd "$currentTime - Analysis COMPLETED successfully."
        puts $resultsFd "$currentTime - transient wall time: $wallTime s"
        puts "Passed!"
    } else {
        puts $resultsFd "$currentTime - Analysis FAILED at time t = [getTime] s"
        puts $resultsFd "$currentTime - transient wall time: $wallTime s"
        puts "Failed!"
    }
    close $resultsFd

    remove recorders
}

buildModel
setupRayleighDamping
runDynamicAnalysis

puts "Output folder: $outputFolder"
foreach f {disp.out vel.out accel.out rayleigh-forces.out element-1-forces.txt} {
    set path [file join $outputFolder $f]
    if {[file exists $path]} {
        puts "  $f: [file size $path] bytes"
    } else {
        puts "  $f: missing"
    }
}
