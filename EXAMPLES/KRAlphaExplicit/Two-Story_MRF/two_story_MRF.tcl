# two_story_MRF.tcl
# Two-Story MRF (Kolay & Ricles) — Tcl counterpart to two_story_MRF.py
#
# Usage (from this directory):
#   /path/to/OpenSees two_story_MRF.tcl
#   /path/to/OpenSees two_story_MRF.tcl KRAlphaExplicit 0.5
#   /path/to/OpenSees two_story_MRF.tcl CudaKRAlpha 1.0 3.0 -incrementalAccel
#   /path/to/OpenSees two_story_MRF.tcl CudaKRAlpha 1.0 3.0 -incrementalAccel -alphaCloseCheck
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

if {$argc >= 1} { set integratorMethod [lindex $argv 0] }
if {$argc >= 2} { set rho [lindex $argv 1] }
if {$argc >= 3} { set scaleFactor [lindex $argv 2] }

set useDiagonalMass 0
set useIncrementalAccel 0
set useAlphaCloseCheck 0
for {set i 3} {$i < $argc} {incr i} {
    set flag [string tolower [lindex $argv $i]]
    switch -glob $flag {
        diagonalmass -
        lumped { set useDiagonalMass 1 }
        -incrementalaccel { set useIncrementalAccel 1 }
        -alphaclosecheck { set useAlphaCloseCheck 1 }
        default {
            puts stderr "Unknown flag: [lindex $argv $i]"
            puts stderr "Optional flags: -incrementalAccel -alphaCloseCheck diagonalMass"
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
    global useDiagonalMass useIncrementalAccel useAlphaCloseCheck
    if {$useDiagonalMass} { lappend paramsList -diagonalMass }
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

if {$integratorMethod in {KRAlphaExplicit MKRAlphaExplicit} && ($useIncrementalAccel || $useAlphaCloseCheck || $useDiagonalMass)} {
    puts stderr "WARNING two_story_MRF.tcl: -incrementalAccel/-alphaCloseCheck/-diagonalMass ignored for dense KRAlphaExplicit/MKRAlphaExplicit"
}

# Match Python result folder names
if {$integratorMethod eq "Newmark" && [info exists newmarkCpu] && $newmarkCpu} {
    set paramsLabel {[0.5, 0.25]}
    set outputFolder [file join $scriptDir results Newmark_FullGeneral_params-${paramsLabel}]
} elseif {$integratorMethod eq "Newmark"} {
    set paramsLabel {[0.5, 0.25]}
    set outputFolder [file join $scriptDir results Newmark_params-${paramsLabel}]
} else {
    set paramsLabel [formatParamsLabel $integratorParams]
    set outputFolder [file join $scriptDir results ${integratorMethod}_params-${paramsLabel}]
}

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

proc buildModel {} {
    global meter sec kg GPa MPa inch foot lbf gravity kN

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

    mass 8  [expr 50.97e3 * $kg] [expr 1.0 * $kg] [expr 1.0 * $kg * $meter]
    mass 16 [expr 50.97e3 * $kg] [expr 1.0 * $kg] [expr 1.0 * $kg * $meter]

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

    geomTransf Linear 1
    beamIntegration Legendre 1 1 2
    element dispBeamColumn 1  1  2  1 1 -cMass -mass $beam_linear_density
    element dispBeamColumn 2  2  3  1 1 -cMass -mass $beam_linear_density
    element dispBeamColumn 3  3  4  1 1 -cMass -mass $beam_linear_density
    element dispBeamColumn 4  4  5  1 1 -cMass -mass $beam_linear_density
    element dispBeamColumn 5  5  6  1 1 -cMass -mass $beam_linear_density
    element dispBeamColumn 6  6  7  1 1 -cMass -mass $beam_linear_density
    element dispBeamColumn 9  9  10 1 1 -cMass -mass $beam_linear_density
    element dispBeamColumn 10 10 11 1 1 -cMass -mass $beam_linear_density
    element dispBeamColumn 11 11 12 1 1 -cMass -mass $beam_linear_density
    element dispBeamColumn 12 12 13 1 1 -cMass -mass $beam_linear_density
    element dispBeamColumn 13 13 14 1 1 -cMass -mass $beam_linear_density
    element dispBeamColumn 14 14 15 1 1 -cMass -mass $beam_linear_density

    geomTransf Linear 2
    beamIntegration Legendre 2 2 2
    element dispBeamColumn 15 17 20 2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 16 20 21 2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 17 21 22 2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 18 22 9  2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 19 9  23 2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 20 23 24 2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 21 24 25 2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 22 25 26 2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 23 26 1  2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 24 18 27 2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 25 27 28 2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 26 28 29 2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 27 29 15 2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 28 15 30 2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 29 30 31 2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 30 31 32 2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 31 32 33 2 2 -cMass -mass $col_linear_density
    element dispBeamColumn 32 33 7  2 2 -cMass -mass $col_linear_density

    set A_leaning [expr 9.76e-2 * $meter * $meter]
    set I_leaning [expr 7.125e-4 * $meter * $meter * $meter * $meter]
    geomTransf PDelta 3
    set leanMass [expr 1.0e-3 * $kg / $meter]
    element elasticBeamColumn 33 19 16 $A_leaning $Es $I_leaning 3 -mass $leanMass -cMass
    element elasticBeamColumn 34 16 8  $A_leaning $Es $I_leaning 3 -mass $leanMass -cMass

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
    numberer RCM
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
    global integratorMethod integratorParams maxIter pFlag algo linearSOE
    global outputFolder MONITOR_NODES MONITOR_DOFS sec

    file mkdir $outputFolder
    set gmDat [file join $outputFolder gm.dat]
    source [file join $scriptDir .. .. verification ReadRecord.tcl]
    ReadRecord $gmFile $gmDat dtFile nPts

    timeSeries Path 2 -filePath $gmDat -dt $dtFile -factor $gravity
    pattern UniformExcitation 2 1 -accel 2 -fact $scaleFactor

    wipeAnalysis
    system $linearSOE
    constraints Transformation
    numberer RCM
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
    puts $resultsFd "$currentTime - test NormUnbalance 1.0e-8 $maxIter $pFlag; algorithm $algo; system $linearSOE; integrator $integratorMethod $integratorParams."
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

    set resultsFd [open $resultsPath a+]
    set currentTime [clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}]
    if {$ok == 0} {
        puts $resultsFd "$currentTime - Analysis COMPLETED successfully."
        puts "Passed!"
    } else {
        puts $resultsFd "$currentTime - Analysis FAILED at time t = [getTime] s"
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
