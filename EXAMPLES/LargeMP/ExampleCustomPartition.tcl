# LargeMP — same mesh/analysis as Example.tcl, but partition from a saved
# METIS map (ele_part.np${np}.map) via -customPartition.
#
# Workflow:
#   mpiexec -n 4 OpenSeesMP Example.tcl 4 4 20          # writes ele_part.np4.map
#   mpiexec -n 4 OpenSeesMP ExampleCustomPartition.tcl 4 4 20
#   python3 compare_tip.py --against mp --kind custom
#
# Requires the same np (and preferably same nx ny nz) as the map header.

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
        puts "usage: ExampleCustomPartition.tcl \[nx ny nz\]"
        exit 1
    }
    set nx $a
    set ny $b
    set nz $c
} elseif {$argc != 0} {
    puts "usage: ExampleCustomPartition.tcl \[nx ny nz\]"
    exit 1
}

if {!$parallel} {
    puts "ERROR: ExampleCustomPartition.tcl requires np>1 (use Example.tcl for serial)"
    exit 1
}

source analysis.tcl

# Stale custom tip files confuse compare_tip globs across np / mesh sizes.
if {$pid == 0} {
    cleanLargeMPOutputs [list \
        tip_eigen.custom.rank*.txt tip_disp.custom.rank*.txt tip_dyn.custom.rank*.txt]
}
barrier

set mapFile "ele_part.np${np}.map"
if {![file exists $mapFile]} {
    puts "ERROR: missing $mapFile — run Example.tcl with the same np first"
    exit 1
}

# Parse map into flat tag/part list for -customPartition
set customPairs {}
set mapNp -1
set mapNx -1
set mapNy -1
set mapNz -1
set fid [open $mapFile r]
while {[gets $fid line] >= 0} {
    set line [string trim $line]
    if {$line eq ""} { continue }
    if {[string match "# np=*" $line]} {
        regexp {np=([0-9]+)} $line -> mapNp
        regexp {nx=([0-9]+)} $line -> mapNx
        regexp {ny=([0-9]+)} $line -> mapNy
        regexp {nz=([0-9]+)} $line -> mapNz
        if {$mapNp != $np} {
            puts "ERROR: $mapFile is for np=$mapNp but this run has np=$np"
            exit 1
        }
        if {$mapNx != $nx || $mapNy != $ny || $mapNz != $nz} {
            puts "ERROR: $mapFile mesh nx/ny/nz=$mapNx/$mapNy/$mapNz != $nx/$ny/$nz"
            exit 1
        }
        continue
    }
    if {[string index $line 0] eq "#"} { continue }
    foreach {etag part} $line break
    if {![string is integer -strict $etag] || ![string is integer -strict $part]} {
        puts "ERROR: bad map line: $line"
        exit 1
    }
    lappend customPairs $etag $part
}
close $fid

set nPairs [expr {[llength $customPairs] / 2}]
if {$nPairs < 1} {
    puts "ERROR: no element assignments in $mapFile"
    exit 1
}
puts "rank $pid / $np reading $mapFile ($nPairs elements)"

source common.tcl

set nBefore [llength [getNodeTags]]
set eBefore [llength [getEleTags]]
puts "rank $pid / $np before partition: nodes=$nBefore eles=$eBefore tipNode=$nn"

if {$nPairs != $eBefore} {
    puts "WARNING: map has $nPairs assignments for $eBefore mesh elements"
}

partition -customPartition $nPairs {*}$customPairs

set nodeTags [getNodeTags]
set eleTags  [getEleTags]
set nAfter [llength $nodeTags]
set eAfter [llength $eleTags]
puts "rank $pid / $np after custom partition: nodes=$nAfter eles=$eAfter"

if {$eAfter >= $eBefore} {
    puts "ERROR: rank $pid expected fewer elements after partition"
    exit 1
}

set ownsTip [expr {[lsearch -exact $nodeTags $nn] >= 0}]
set soe "Mumps"
set num "ParallelPlain"
set tipKind "custom"

runLargeMPAnalysis $pid $parallel $ownsTip $nn $soe $num $tipKind

wipe
exit 0
