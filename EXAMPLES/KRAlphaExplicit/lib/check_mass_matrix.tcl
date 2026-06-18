# check_mass_matrix.tcl
# Reusable mass-matrix extraction / diagonal check (GimmeMCK + printA).
#
# Reference: https://openseesdigital.com/2020/05/17/gimme-all-your-damping-all-your-mass-and-stiffness-too/
#
# Procs:
#   assembleMassMatrixFile <outDenseFile> ?outSparseFile?
#   massMatrixDiagonalReport <denseFile> <N> ?tol?
#   runMassMatrixCheck <label> <outReportFile> ?tol?

proc assembleMassMatrixFile {outDenseFile {outSparseFile ""}} {
    wipeAnalysis
    constraints Transformation
    numberer Plain
    system FullGeneral
    test NormUnbalance 1.0e-12 1 0
    algorithm Linear
    integrator GimmeMCK 1.0 0.0 0.0
    analysis Transient
    set ok [analyze 1 0.0]
    if {$ok != 0} {
        puts stderr "WARNING: GimmeMCK analyze returned $ok (expected for singular M blocks)"
    }
    set N [systemSize]
    if {$N <= 0} {
        error "assembleMassMatrixFile: systemSize returned $N"
    }
    printA -file $outDenseFile
    if {$outSparseFile ne ""} {
        printA -sparse 0 -file $outSparseFile
    }
    return $N
}

proc _readDenseMatrix {filename N} {
    set fd [open $filename r]
    set rows {}
    while {[gets $fd line] >= 0} {
        set line [string trim $line]
        if {$line eq ""} { continue }
        set rows [lappend rows $line]
    }
    close $fd
    if {[llength $rows] != $N} {
        error "dense matrix file has [llength $rows] rows, expected $N"
    }
    set M {}
    foreach line $rows {
        set vals [regexp -all -inline {\S+} $line]
        if {[llength $vals] != $N} {
            error "row has [llength $vals] entries, expected $N"
        }
        lappend M $vals
    }
    return $M
}

proc massMatrixDiagonalReport {denseFile N {tol 1.0e-12}} {
    set M [_readDenseMatrix $denseFile $N]
    set diagSum 0.0
    set offSum 0.0
    set offMax 0.0
    set offCount 0
    set nnz 0
    for {set i 0} {$i < $N} {incr i} {
        set row [lindex $M $i]
        for {set j 0} {$j < $N} {incr j} {
            set v [lindex $row $j]
            if {abs($v) <= $tol} { continue }
            incr nnz
            if {$i == $j} {
                set diagSum [expr {$diagSum + abs($v)}]
            } else {
                incr offCount
                set a [expr {abs($v)}]
                set offSum [expr {$offSum + $a}]
                if {$a > $offMax} { set offMax $a }
            }
        }
    }
    set isDiag [expr {$offCount == 0}]
  return [list $isDiag $offCount $offMax $offSum $diagSum $nnz]
}

proc runMassMatrixCheck {label outReportFile {tol 1.0e-12}} {
    set denseFile [file join [file dirname $outReportFile] "_mass_dense_tmp.txt"]
    set sparseFile [file join [file dirname $outReportFile] "_mass_sparse_tmp.mtx"]
    set N [assembleMassMatrixFile $denseFile $sparseFile]

    lassign [massMatrixDiagonalReport $denseFile $N $tol] isDiag offCount offMax offSum diagSum nnz

    set fd [open $outReportFile w]
    puts $fd "Mass matrix check: $label"
    puts $fd "  equations N     = $N"
    puts $fd "  tolerance       = $tol"
    puts $fd "  nonzeros (|v|>tol) = $nnz"
    puts $fd "  diagonal |.| sum = $diagSum"
    puts $fd "  off-diagonal count = $offCount"
    puts $fd "  off-diagonal max |.| = $offMax"
    puts $fd "  off-diagonal |.| sum = $offSum"
    if {$isDiag} {
        puts $fd "  RESULT: DIAGONAL (within tolerance)"
    } else {
        puts $fd "  RESULT: NOT DIAGONAL"
    }
    puts $fd ""
    puts $fd "Dense M written to: $denseFile"
    puts $fd "Sparse M written to: $sparseFile"
    close $fd

    puts "Mass matrix check: $label"
    puts "  N=$N  off-diagonal nnz=$offCount  max|off|=$offMax"
    if {$isDiag} {
        puts "  => DIAGONAL"
    } else {
        puts "  => NOT DIAGONAL (see $outReportFile)"
    }
    return $isDiag
}
