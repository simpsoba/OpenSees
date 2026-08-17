# LargeMP — LargeSP brick column, serial or OpenSeesMP (METIS partition).
#
# Serial (UmfPack reference):
#   OpenSees Example.tcl
#   OpenSees Example.tcl 4 4 20
#
# Parallel (METIS + Mumps); also writes ele_part.np${np}.map for
# ExampleCustomPartition.tcl:
#   mpiexec -n 4 OpenSeesMP Example.tcl
#   mpiexec -n 4 OpenSeesMP Example.tcl 4 4 20
#
# Compare: python3 compare_tip.py

wipe

set outDir [file dirname [info script]]
if {$outDir eq ""} { set outDir "." }
cd $outDir

set pid [getPID]
set np  [getNP]
set parallel [expr {$np > 1}]

set nx 10
set ny 10
set nz 100
if {$argc == 3} {
    foreach {a b c} $argv break
    if {![string is integer -strict $a] || ![string is integer -strict $b] ||
        ![string is integer -strict $c]} {
        puts "usage: Example.tcl \[nx ny nz\]  (positive integers)"
        exit 1
    }
    set nx $a
    set ny $b
    set nz $c
} elseif {$argc != 0} {
    puts "usage: Example.tcl \[nx ny nz\]"
    exit 1
}

source analysis.tcl

# Drop stale tip / rank-map files before this run writes new ones.
if {$pid == 0} {
    if {$parallel} {
        cleanLargeMPOutputs [list \
            tip_eigen.mp.rank*.txt tip_disp.mp.rank*.txt tip_dyn.mp.rank*.txt \
            ele_part.rank*.txt]
    } else {
        cleanLargeMPOutputs [list \
            tip_eigen.serial.txt tip_disp.serial.txt tip_dyn.serial.txt]
    }
}
if {$parallel} {
    barrier
}

source common.tcl

set nBefore [llength [getNodeTags]]
set eBefore [llength [getEleTags]]
puts "rank $pid / $np before partition: nodes=$nBefore eles=$eBefore tipNode=$nn"

if {$nBefore != $nn} {
    puts "WARNING: expected $nn nodes before partition, got $nBefore"
}

if {$parallel} {
    partition -ncuts 1 -niter 10 -ufactor 30

    set nodeTags [getNodeTags]
    set eleTags  [getEleTags]
    set nAfter [llength $nodeTags]
    set eAfter [llength $eleTags]
    puts "rank $pid / $np after partition: nodes=$nAfter eles=$eAfter"

    if {$eAfter >= $eBefore} {
        puts "ERROR: rank $pid expected fewer elements after partition ($eAfter >= $eBefore)"
        exit 1
    }
    if {$nAfter >= $nBefore} {
        puts "ERROR: rank $pid expected fewer nodes after partition ($nAfter >= $nBefore)"
        exit 1
    }

    # Local METIS ownership map (element tag -> this rank). Merged on rank 0.
    set partFile "ele_part.rank$pid.txt"
    set fid [open $partFile w]
    foreach e [lsort -integer $eleTags] {
        puts $fid "$e $pid"
    }
    close $fid

    barrier
    if {$pid == 0} {
        set mapFile "ele_part.np${np}.map"
        set mfid [open $mapFile w]
        puts $mfid "# LargeMP METIS element partition map"
        puts $mfid "# np=$np nx=$nx ny=$ny nz=$nz"
        puts $mfid "# eleTag part"
        set nAssigned 0
        for {set r 0} {$r < $np} {incr r} {
            set rf "ele_part.rank$r.txt"
            if {![file exists $rf]} {
                puts "ERROR: missing $rf after METIS partition"
                exit 1
            }
            set rfid [open $rf r]
            while {[gets $rfid line] >= 0} {
                set line [string trim $line]
                if {$line eq ""} { continue }
                puts $mfid $line
                incr nAssigned
            }
            close $rfid
        }
        close $mfid
        puts "rank 0 wrote $mapFile ($nAssigned element assignments)"
        if {$nAssigned != $eBefore} {
            puts "WARNING: map has $nAssigned assignments, expected $eBefore elements"
        }
    }
    barrier

    set ownsTip [expr {[lsearch -exact $nodeTags $nn] >= 0}]
    set soe "Mumps"
    set num "ParallelPlain"
    set tipKind "mp"
} else {
    set ownsTip 1
    set soe "UmfPack"
    set num "Plain"
    set tipKind "serial"
    puts "serial mode: system $soe numberer $num"
}

runLargeMPAnalysis $pid $parallel $ownsTip $nn $soe $num $tipKind

wipe
exit 0
