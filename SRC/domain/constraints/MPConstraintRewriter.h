/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
**                                                                    **
** (C) Copyright 1999, The Regents of the University of California    **
** All Rights Reserved.                                               **
**                                                                    **
** Commercial use of this program without express permission of the   **
** University of California, Berkeley, is strictly prohibited.  See   **
** file 'COPYRIGHT'  in main directory for information on usage and   **
** redistribution,  and for a DISCLAIMER OF ALL WARRANTIES.           **
**                                                                    **
** Developed by:                                                      **
**   Frank McKenna (fmckenna@ce.berkeley.edu)                         **
**   Gregory L. Fenves (fenves@ce.berkeley.edu)                       **
**   Filip C. Filippou (filippou@ce.berkeley.edu)                     **
**                                                                    **
** ****************************************************************** */

// Written: gaaraujo
// Created: 08/26
//
// Purpose: Rewrite Domain MP_Constraints into a Transformation-ready form
// (equalDOF stars, invertible-MP flips when an SP is on the constrained
// side, chain composition). Call after all fix/sp/pattern definitions, before
// partition / OpenSeesMP, and before creating Transformation / analyze.
// SP-dependent decisions use the Domain snapshot at call time.

#ifndef MPConstraintRewriter_h
#define MPConstraintRewriter_h

class Domain;

// Options for rewriteMPConstraints().
struct RewriteMPOptions {
    bool checkOnly = false; // inspect only; Domain unchanged
    bool verbose = false;   // print detailed report to opserr
    const char *fileName = nullptr; // write detailed report here (does not imply verbose)
};

// Default: rewrite MPs in the Domain. New MPs are built first; on analysis
// failure the Domain is left unchanged. On the rare mid-commit addMP failure
// after deletes (-30), the Domain may be partially updated.
//
// Return values (also set as the Tcl/Python result):
//   apply (default):  0 = already ready / nothing changed;
//                     >0 = MP definitions changed;
//                     <0 = failed (Domain unchanged, except rare -30).
//   -checkOnly:       0 = already Transformation-ready;
//                     >0 = not ready, but a rewrite is possible
//                          (count = MP definitions that differ);
//                     <0 = cannot be made Transformation-ready.
//
// If -checkOnly returns >0, still check the subsequent apply return (or catch)
// before selecting Transformation; do not assume apply cannot fail.
//
// Console stays to a one-line summary unless -verbose. -file only selects the
// report destination; it does not change analysis or mutation semantics.
int rewriteMPConstraints(Domain &theDomain, const RewriteMPOptions &opts = {});

// OPS / interpreter entry (parses -checkOnly, -verbose, -file).
int OPS_rewriteMPConstraints();

#endif
