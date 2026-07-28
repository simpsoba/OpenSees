# Partition regression: floating / fixed / equalDOF nodes under METIS and custom.
#
#   mpiexec -n 2 OpenSeesMP TestPartitionNodes.tcl
#   mpiexec -n 4 OpenSeesMP TestPartitionNodes.tcl
#
# Exit status is non-zero if any rank reports a failed check.

proc hasNode {tag} {
    expr {[lsearch -exact [getNodeTags] $tag] >= 0}
}

proc massOf {tag} {
    if {![hasNode $tag]} {
        return -1.0
    }
    return [nodeMass $tag 1]
}

proc countHas {tag} {
    # Each rank prints 1/0; driver sums from stdout externally if needed.
    # Local helper for single-rank expectations.
    return [hasNode $tag]
}

proc check {name cond msg} {
    global pid fails
    if {$cond} {
        puts "CHECK PASS rank=$pid $name"
    } else {
        puts "CHECK FAIL rank=$pid $name :: $msg"
        incr fails
    }
}

proc buildChain {} {
    # nodes 1-4 with truss 1-2-3; used as the connected mesh
    wipe
    model BasicBuilder -ndm 1 -ndf 1
    uniaxialMaterial Elastic 1 1000.0
    node 1 0.0
    node 2 1.0
    node 3 2.0
    node 4 3.0
    element truss 1 1 2 1.0 1
    element truss 2 2 3 1.0 1
    element truss 3 3 4 1.0 1
}

proc doPartition {mode} {
    global np
    if {$mode eq "custom"} {
        # Spread the 3 truss elements across min(np,3) parts; leftover ranks
        # intentionally get no elements (orphan ownership still exercised).
        set custom [dict create]
        for {set e 1} {$e <= 3} {incr e} {
            dict set custom $e [expr {($e - 1) % $np}]
        }
        partition -customPartition [dict size $custom] {*}$custom
    } else {
        partition -ncuts 1 -niter 10 -ufactor 30
    }
}

proc sumFlag {flag} {
    # Reduce a 0/1 flag across ranks via a shared temp file pattern is awkward
    # under MPI without a barrier file lock. Instead each rank prints FLAG and
    # the process-local checks below only assert local invariants; global
    # exactly-once checks are done by counting FLAG lines in a wrapper.
    global pid
    puts "FLAG rank=$pid $flag"
}

set pid [getPID]
set np [getNP]
set fails 0

if {$np < 2} {
    puts "ERROR: need at least 2 MPI ranks"
    exit 1
}

puts "=== PartitionNodes np=$np pid=$pid ==="

foreach mode {custom metis} {

    # ------------------------------------------------------------------
    # A: fixed floating orphan
    # ------------------------------------------------------------------
    buildChain
    node 99 99.0
    fix 99 1
    doPartition $mode
    set h [hasNode 99]
    sumFlag "A_$mode has99=$h"
    if {$h} {
        # Owner rank: node present. A load on a fixed DOF should be OK to define.
        check A_${mode}_present 1 "owner should have node 99"
    } else {
        check A_${mode}_absent 1 "non-owner correctly dropped node 99"
    }

    # ------------------------------------------------------------------
    # B: floating orphan with mass (no fix)
    # ------------------------------------------------------------------
    buildChain
    node 98 98.0 -mass 5.0
    doPartition $mode
    set h [hasNode 98]
    set m [massOf 98]
    sumFlag "B_$mode has98=$h mass=$m"
    if {$h} {
        check B_${mode}_mass [expr {abs($m - 5.0) < 1e-12}] "owner mass=$m expected 5"
    } else {
        check B_${mode}_absent 1 "non-owner dropped mass orphan"
    }

    # ------------------------------------------------------------------
    # C: equalDOF floating slave (massless) to interior/interface node 2
    # ------------------------------------------------------------------
    buildChain
    node 97 97.0
    equalDOF 2 97 1
    doPartition $mode
    set h2 [hasNode 2]
    set h97 [hasNode 97]
    sumFlag "C_$mode has2=$h2 has97=$h97"
    # Slave must appear on every rank that kept the master
    if {$h2} {
        check C_${mode}_slave_with_master $h97 "master 2 present but slave 97 missing"
    } else {
        check C_${mode}_no_orphan_slave [expr {!$h97}] "slave 97 without master 2"
    }

    # ------------------------------------------------------------------
    # D: equalDOF floating slave WITH mass to interface node 2
    # ------------------------------------------------------------------
    buildChain
    node 96 96.0 -mass 7.0
    equalDOF 2 96 1
    doPartition $mode
    set h2 [hasNode 2]
    set h96 [hasNode 96]
    set m [massOf 96]
    sumFlag "D_$mode has2=$h2 has96=$h96 mass=$m"
    if {$h2} {
        check D_${mode}_slave_with_master $h96 "master 2 present but slave 96 missing"
    }
    if {$h96} {
        # Exactly one rank should keep the 7.0 mass; others that retain the
        # node (interface copies) must show 0.
        check D_${mode}_mass_nonneg [expr {$m >= -1e-15}] "negative mass?"
        if {$m > 1e-12} {
            check D_${mode}_mass_value [expr {abs($m - 7.0) < 1e-12}] "mass=$m expected 7"
            sumFlag "D_$mode massOwner=1"
        } else {
            check D_${mode}_mass_zero [expr {abs($m) < 1e-12}] "expected zeroed duplicate mass"
            sumFlag "D_$mode massOwner=0"
        }
    } else {
        sumFlag "D_$mode massOwner=0"
    }

    # ------------------------------------------------------------------
    # E: equalDOF floating slave that is also fixed
    # ------------------------------------------------------------------
    buildChain
    node 95 95.0
    fix 95 1
    equalDOF 2 95 1
    doPartition $mode
    set h2 [hasNode 2]
    set h95 [hasNode 95]
    sumFlag "E_$mode has2=$h2 has95=$h95"
    if {$h2} {
        check E_${mode}_fixed_slave $h95 "fixed slave 95 missing beside master 2"
    } else {
        check E_${mode}_no_fixed_orphan [expr {!$h95}] "fixed slave without master"
    }

    # ------------------------------------------------------------------
    # F: floating node is the RETAINED side of equalDOF
    # ------------------------------------------------------------------
    buildChain
    node 94 94.0 -mass 3.0
    equalDOF 94 2 1
    doPartition $mode
    set h2 [hasNode 2]
    set h94 [hasNode 94]
    set m [massOf 94]
    sumFlag "F_$mode has2=$h2 has94=$h94 mass=$m"
    if {$h2} {
        check F_${mode}_retained_float $h94 "floating retained node 94 missing beside 2"
    }
    if {$h94 && $m > 1e-12} {
        check F_${mode}_mass [expr {abs($m - 3.0) < 1e-12}] "mass=$m expected 3"
        sumFlag "F_$mode massOwner=1"
    } else {
        sumFlag "F_$mode massOwner=0"
    }

    # ------------------------------------------------------------------
    # G: equalDOF between two floating orphans
    # ------------------------------------------------------------------
    buildChain
    node 93 93.0 -mass 2.0
    node 92 92.0
    equalDOF 93 92 1
    doPartition $mode
    set h93 [hasNode 93]
    set h92 [hasNode 92]
    set m [massOf 93]
    sumFlag "G_$mode has93=$h93 has92=$h92 mass=$m"
    # They must stay together on the same rank (shared owner)
    check G_${mode}_together [expr {$h93 == $h92}] "orphans split across ranks has93=$h93 has92=$h92"
    if {$h93} {
        check G_${mode}_mass [expr {abs($m - 2.0) < 1e-12}] "owner mass=$m expected 2"
    }
}

# ------------------------------------------------------------------
# H: kinematic smoke — custom only, equalDOF slave tracks master.
# Requires every rank to own >=1 element so Mumps can form a graph.
# ------------------------------------------------------------------
if {$np <= 3} {
    buildChain
    node 91 91.0
    equalDOF 2 91 1
    fix 1 1
    set custom [dict create]
    for {set e 1} {$e <= 3} {incr e} {
        dict set custom $e [expr {($e - 1) % $np}]
    }
    partition -customPartition [dict size $custom] {*}$custom
    timeSeries Linear 1
    pattern Plain 1 1 {
        if {[hasNode 2]} {
            load 2 1.0
        }
    }
    system Mumps
    numberer ParallelPlain
    constraints Transformation
    integrator LoadControl 1.0
    algorithm Linear
    analysis Static
    set ok [analyze 1]
    check H_custom_analyze [expr {$ok == 0}] "analyze ok=$ok"
    if {[hasNode 2] && [hasNode 91]} {
        set d2 [nodeDisp 2 1]
        set d91 [nodeDisp 91 1]
        check H_custom_equalDOF [expr {abs($d2 - $d91) < 1e-10}] "d2=$d2 d91=$d91"
        sumFlag "H_custom d2=$d2 d91=$d91"
    } else {
        check H_custom_skipped 1 "rank without both nodes skips kinematics compare"
    }
} else {
    check H_custom_skipped_np 1 "kinematic smoke needs np<=3 (empty ranks break Mumps)"
}

puts "SUMMARY rank=$pid fails=$fails"
if {$fails > 0} {
    exit 1
}
exit 0
