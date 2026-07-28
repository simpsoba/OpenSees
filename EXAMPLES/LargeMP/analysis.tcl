# Shared LargeMP analysis after the model (and optional partition) exist.
# Callers set: pid, np, parallel, ownsTip, nn, soe, num, tipKind
#   tipKind = "serial" | "mp" | "custom"  (output file naming)

# Rank 0 removes stale compare_tip inputs so a new run cannot mix old rank files.
proc cleanLargeMPOutputs {patterns} {
    foreach pat $patterns {
        foreach f [glob -nocomplain $pat] {
            file delete -force $f
        }
    }
}

proc writeTip {prefix ownsTip nn pid tipKind} {
    if {!$ownsTip} { return }
    set ux [nodeDisp $nn 1]
    set uy [nodeDisp $nn 2]
    set uz [nodeDisp $nn 3]
    if {$tipKind eq "serial"} {
        set label "serial"
        set file "$prefix.serial.txt"
    } else {
        set label "rank $pid"
        set file "$prefix.$tipKind.rank$pid.txt"
    }
    puts "$label $prefix tip disp: $ux $uy $uz"
    set fid [open $file w]
    puts $fid "$ux $uy $uz"
    close $fid
}

proc writeEigen {lambdas pid tipKind} {
    if {$tipKind eq "serial"} {
        set label "serial"
        set file "tip_eigen.serial.txt"
    } else {
        set label "rank $pid"
        set file "tip_eigen.$tipKind.rank$pid.txt"
    }
    puts "$label eigenvalues: $lambdas"
    set fid [open $file w]
    puts $fid [string trim $lambdas]
    close $fid
}

# Classical Rayleigh: same zeta at omega1 and omega3
proc rayleighFromModes {lam1 lam3 zeta} {
    set w1 [expr {sqrt($lam1)}]
    set w3 [expr {sqrt($lam3)}]
    set T1 [expr {2.0*acos(-1.0)/$w1}]
    set T3 [expr {2.0*acos(-1.0)/$w3}]
    set alphaM [expr {2.0*$zeta*$w1*$w3/($w1+$w3)}]
    set betaK  [expr {2.0*$zeta/($w1+$w3)}]
    return [list $T1 $T3 $alphaM $betaK]
}

proc runLargeMPAnalysis {pid parallel ownsTip nn soe num tipKind} {
    integrator LoadControl  1.0  1
    test NormUnbalance     1.0e-10    20     0
    algorithm Newton
    numberer $num
    constraints Plain
    system $soe
    analysis Static

    set ok [analyze 2]
    if {$ok != 0} {
        puts "ERROR: rank $pid static analyze failed ($ok)"
        exit 1
    }
    writeTip tip_disp $ownsTip $nn $pid $tipKind

    set lambdas [eigen 3]
    writeEigen $lambdas $pid $tipKind

    set zeta 0.05
    lassign [rayleighFromModes [lindex $lambdas 0] [lindex $lambdas 2] $zeta] \
        T1 T3 alphaM betaK
    if {$parallel} {
        puts "rank $pid Rayleigh zeta=$zeta T1=$T1 T3=$T3 alphaM=$alphaM betaK=$betaK"
    } else {
        puts "serial Rayleigh zeta=$zeta T1=$T1 T3=$T3 alphaM=$alphaM betaK=$betaK"
    }

    wipeAnalysis
    setTime 0.0
    remove loadPattern 1
    rayleigh $alphaM 0.0 0.0 $betaK

    test EnergyIncr     1.0e-10    20   0
    algorithm Newton
    numberer $num
    system $soe
    constraints Plain
    integrator Newmark 0.5 0.25
    analysis Transient

    set dt [expr {$T1/20.0}]
    set nSteps 40
    set ok [analyze $nSteps $dt]
    if {$ok != 0} {
        puts "ERROR: rank $pid transient analyze failed ($ok)"
        exit 1
    }
    writeTip tip_dyn $ownsTip $nn $pid $tipKind

    if {$tipKind eq "serial"} {
        puts "wrote tip_eigen.serial.txt tip_disp.serial.txt tip_dyn.serial.txt"
    } else {
        puts "rank $pid wrote tip_eigen.$tipKind.rank$pid.txt"
        if {$ownsTip} {
            puts "rank $pid wrote tip_disp.$tipKind.rank$pid.txt tip_dyn.$tipKind.rank$pid.txt"
        }
    }
}
