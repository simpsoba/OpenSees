# Shared console output for ShearBuilding40.tcl / ShearBuilding40SP.tcl / ShearBuilding40MP.tcl

proc resetOutputDir {dir} {
    catch { file delete -force $dir }
    file mkdir $dir
}

proc printRunHeader {target} {
    global expElementMode outputDir Nstories ExpEleTag nSteps dt
    puts "=== ShearBuilding40 / $target ==="
    puts "mode=$expElementMode  ExpEleTag=$ExpEleTag  Nstories=$Nstories  outputDir=$outputDir"
    puts "analysis: $nSteps steps, dt=$dt s"
}

proc printOpenFrescoVersion {} {
    puts "OpenFresco version = [packageVersion]"
}

proc printElementAnalytical {Nstories} {
    puts "element: analytical - $Nstories zeroLength springs (no OpenFresco)"
}

proc printElementLocal {ExpEleTag k} {
    set nLo [expr {$ExpEleTag - 1}]
    puts "element: local twoNodeLink $ExpEleTag (k=$k) between nodes $nLo $ExpEleTag"
}

proc printRayleighDamping {alphaM betaK} {
    global zeta w1 w2
    puts "Rayleigh damping: zeta=$zeta at w1=$w1 w2=$w2 rad/s (alphaM=$alphaM betaK=$betaK)"
}

proc printModelBuilt {target solver {extra ""}} {
    global expElementMode ExpEleTag outputDir analysisScheme
    set msg "Model built: target=$target mode=$expElementMode ExpEleTag=$ExpEleTag outputDir=$outputDir analysisScheme=$analysisScheme solver=$solver"
    if {$extra ne ""} {
        append msg " $extra"
    }
    puts $msg
}

proc printPartitionExp {ExpEleTag} {
    puts "partition: element $ExpEleTag assigned to main domain"
}

proc printAnalysisStart {} {
    global nSteps dt
    puts "Starting transient analysis ($nSteps steps, dt=$dt s)"
}

proc printAnalysisDone {ok} {
    global nSteps
    if {$ok != 0} {
        puts "WARNING analyze failed at time [getTime] (return code $ok)"
    } else {
        puts "Finished transient analysis ($nSteps steps)"
    }
}
