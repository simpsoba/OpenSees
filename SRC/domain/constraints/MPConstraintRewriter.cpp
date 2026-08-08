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
// Purpose: Rewrite Domain MP_Constraints into Transformation-ready form
// (equalDOF stars, MP flip, chain composition). See rewriteMPConstraints().

#include "MPConstraintRewriter.h"

#include <Domain.h>
#include <ID.h>
#include <MP_Constraint.h>
#include <MP_ConstraintIter.h>
#include <Matrix.h>
#include <OPS_Globals.h>
#include <SP_Constraint.h>
#include <SP_ConstraintIter.h>
#include <elementAPI.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

// Pack (nodeTag, dofIndex) into one 64-bit key for maps/union-find:
// high 32 bits = node tag, low 32 bits = zero-based DOF index.
inline std::int64_t packDofKey(int nodeTag, int dof) {
    return (static_cast<std::int64_t>(nodeTag) << 32) | static_cast<std::uint32_t>(dof);
}

inline int unpackNode(std::int64_t key) { return static_cast<int>(key >> 32); }

inline int unpackDof(std::int64_t key) { return static_cast<int>(key & 0xffffffffu); }

bool isIdentityEqualDOF(const Matrix &C, const ID &cDOF, const ID &rDOF) {
    if (cDOF.Size() != rDOF.Size() || C.noRows() != cDOF.Size() || C.noCols() != rDOF.Size())
        return false;
    for (int i = 0; i < cDOF.Size(); ++i) {
        if (cDOF(i) != rDOF(i))
            return false;
        for (int j = 0; j < rDOF.Size(); ++j) {
            const double expected = (i == j) ? 1.0 : 0.0;
            if (std::fabs(C(i, j) - expected) > 1.0e-12)
                return false;
        }
    }
    return true;
}

class UnionFind {
public:
    std::int64_t find(std::int64_t x) {
        auto it = parent.find(x);
        if (it == parent.end()) {
            parent.emplace(x, x);
            rank.emplace(x, 0);
            return x;
        }
        if (it->second != x)
            it->second = find(it->second);
        return it->second;
    }

    void unite(std::int64_t a, std::int64_t b) {
        a = find(a);
        b = find(b);
        if (a == b)
            return;
        if (rank[a] < rank[b])
            std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b])
            ++rank[a];
    }

    void ensure(std::int64_t x) {
        if (parent.find(x) == parent.end()) {
            parent.emplace(x, x);
            rank.emplace(x, 0);
        }
    }

    bool contains(std::int64_t x) const { return parent.find(x) != parent.end(); }

private:
    std::unordered_map<std::int64_t, std::int64_t> parent;
    std::unordered_map<std::int64_t, int> rank;
};

struct WorkingMP {
    int retained = 0;
    int constrained = 0;
    ID cDOF;
    ID rDOF;
    Matrix C; // owns a copy of the MP constraint matrix (Uc = C Ur)
    bool identity = false;
};

std::unique_ptr<MP_Constraint> makeMP(WorkingMP &w) {
    return std::make_unique<MP_Constraint>(w.retained, w.constrained, w.C, w.cDOF, w.rDOF);
}

// Format one MP line for the before/after report (1-based DOF indices).
std::string formatMPLine(const WorkingMP &w) {
    std::ostringstream ss;
    ss << (w.identity ? "equalDOF" : "MP     ") << "  retained " << w.retained
       << "  constrained " << w.constrained << "  DOFs [";
    for (int i = 0; i < w.cDOF.Size(); ++i) {
        if (i)
            ss << ' ';
        ss << (w.cDOF(i) + 1);
    }
    ss << ']';
    return ss.str();
}

// Accumulates a human-readable rewrite report for -verbose / -file.
struct RewriteReport {
    int nIn = 0;
    int nTimeVarying = 0;
    int nIdentityIn = 0;
    int nGeneralIn = 0;
    int nFlipped = 0;
    int nComposed = 0;
    int nOut = 0;
    int nDifferingMPs = 0; // |symmetric difference| of MP definitions
    int nRemovedMPs = 0;   // definitions only in current
    int nAddedMPs = 0;     // definitions only in proposed/after
    std::ostringstream before;
    std::ostringstream after;
    std::ostringstream changes;
    std::ostringstream warnings;

    void beforeLine(const std::string &s) { before << "  " << s << '\n'; }
    void afterLine(const std::string &s) { after << "  " << s << '\n'; }
    void change(const std::string &s) { changes << "  - " << s << '\n'; }
    void warn(const std::string &s) { warnings << "  ! " << s << '\n'; }

    // proposed=true for -checkOnly (Domain unchanged; show what would happen).
    std::string str(const char *label, bool proposed) const {
        std::ostringstream ss;
        ss << label << " report:\n";
        if (proposed)
            ss << "Mode: inspection only — Domain was NOT modified\n";
        else
            ss << "Mode: applied — Domain multipoint constraints were updated\n";
        ss << "Summary: " << nIn << " multipoint constraint(s) in -> " << nOut << " out";
        ss << " (equalDOF=" << nIdentityIn << ", general=" << nGeneralIn
           << ", time-varying kept=" << nTimeVarying << ", flipped=" << nFlipped
           << ", chains merged=" << nComposed << ")\n";
        if (nDifferingMPs == 0) {
            ss << "MP definitions: all unchanged (already Transformation-ready)\n";
            if (proposed)
                ss << "WOULD CHANGE:\n  - (none)\n";
            else
                ss << "WHAT CHANGED:\n  - (none)\n";
            return ss.str();
        }
        ss << "MP definitions that differ: " << nDifferingMPs << " (" << nRemovedMPs
           << " only in " << (proposed ? "CURRENT" : "BEFORE") << ", " << nAddedMPs
           << " only in " << (proposed ? "PROPOSED" : "AFTER") << ")\n";
        if (!before.str().empty()) {
            ss << (proposed ? "CURRENT:\n" : "BEFORE:\n");
            ss << before.str();
        }
        if (!after.str().empty()) {
            ss << (proposed ? "PROPOSED:\n" : "AFTER:\n");
            ss << after.str();
        }
        if (!changes.str().empty()) {
            ss << (proposed ? "WOULD CHANGE:\n" : "WHAT CHANGED:\n");
            ss << changes.str();
        }
        if (!warnings.str().empty()) {
            ss << "WARNINGS:\n";
            ss << warnings.str();
        }
        return ss.str();
    }
};

std::string mpSignature(const WorkingMP &w) {
    std::ostringstream ss;
    ss << w.retained << '>' << w.constrained << (w.identity ? ":I:" : ":G:");
    for (int i = 0; i < w.cDOF.Size(); ++i)
        ss << w.cDOF(i) << '/' << w.rDOF(i) << ',';
    ss << '|';
    for (int i = 0; i < w.C.noRows(); ++i)
        for (int j = 0; j < w.C.noCols(); ++j)
            ss << w.C(i, j) << ',';
    return ss.str();
}

// Compare MP-definition multisets. Fills removed/added counts and returns
// their sum (size of the symmetric difference). 0 => identical layouts.
int mpDefinitionDifference(const std::vector<std::string> &before,
                           const std::vector<std::string> &after, int &nRemoved,
                           int &nAdded) {
    std::unordered_map<std::string, int> counts;
    for (const auto &s : before)
        ++counts[s];
    for (const auto &s : after)
        --counts[s];
    nRemoved = 0;
    nAdded = 0;
    for (const auto &kv : counts) {
        if (kv.second > 0)
            nRemoved += kv.second;
        else if (kv.second < 0)
            nAdded += -kv.second;
    }
    return nRemoved + nAdded;
}

int rewriteMPs(Domain &theDomain, std::vector<std::unique_ptr<MP_Constraint>> &ownedMPs,
               const char *handlerLabel, RewriteReport &report) {
    ownedMPs.clear();
    if (handlerLabel == nullptr)
        handlerLabel = "Constraint";

    // --- collect SPs ---
    std::unordered_map<std::int64_t, SP_Constraint *> spMap;
    SP_ConstraintIter &spIter = theDomain.getDomainAndLoadPatternSPs();
    SP_Constraint *sp = nullptr;
    while ((sp = spIter()) != nullptr)
        spMap[packDofKey(sp->getNodeTag(), sp->getDOF_Number())] = sp;

    auto hasSP = [&spMap](int node, int dof) -> bool {
        return spMap.find(packDofKey(node, dof)) != spMap.end();
    };

    // --- collect MPs into working list (domain order) ---
    std::vector<WorkingMP> work;
    std::vector<std::string> beforeSigs;
    work.reserve(static_cast<std::size_t>(theDomain.getNumMPs()));
    beforeSigs.reserve(static_cast<std::size_t>(theDomain.getNumMPs()));
    MP_ConstraintIter &mpIter = theDomain.getMPs();
    MP_Constraint *mp = nullptr;
    while ((mp = mpIter()) != nullptr) {
        WorkingMP w;
        w.retained = mp->getNodeRetained();
        w.constrained = mp->getNodeConstrained();
        w.cDOF = mp->getConstrainedDOFs();
        w.rDOF = mp->getRetainedDOFs();
        w.C = mp->getConstraint();
        w.identity = isIdentityEqualDOF(w.C, w.cDOF, w.rDOF);
        ++report.nIn;
        beforeSigs.push_back(mpSignature(w));
        report.beforeLine(formatMPLine(w) + (mp->isTimeVarying() ? "  (time-varying)" : ""));
        if (mp->isTimeVarying()) {
            // time-varying MPs are kept as-is and must not participate in chains
            ++report.nTimeVarying;
            ownedMPs.push_back(makeMP(w));
            continue;
        }
        if (w.identity)
            ++report.nIdentityIn;
        else
            ++report.nGeneralIn;
        work.push_back(std::move(w));
    }

    // Split identity vs general
    std::vector<WorkingMP> identityMPs;
    std::vector<WorkingMP> generalMPs;
    identityMPs.reserve(work.size());
    generalMPs.reserve(work.size());
    for (auto &w : work) {
        if (w.identity)
            identityMPs.push_back(std::move(w));
        else
            generalMPs.push_back(std::move(w));
    }

    // --- equalDOF clique rewrite via DOF-level union-find ---
    UnionFind uf;
    for (const auto &w : identityMPs) {
        for (int i = 0; i < w.cDOF.Size(); ++i) {
            const std::int64_t ck = packDofKey(w.constrained, w.cDOF(i));
            const std::int64_t rk = packDofKey(w.retained, w.rDOF(i));
            uf.ensure(ck);
            uf.ensure(rk);
            uf.unite(ck, rk);
        }
    }

    // Collect members of each component. Also record which (node,dof) keys
    // were written as retained vs constrained in the original equalDOF MPs
    // so we can honor user intent (equalDOF Rnode Cnode ...) when rewriting.
    std::unordered_map<std::int64_t, std::vector<std::int64_t>> components;
    std::unordered_set<std::int64_t> seenKeys;
    std::unordered_set<std::int64_t> writtenRetained;
    std::unordered_set<std::int64_t> writtenConstrained;
    for (const auto &w : identityMPs) {
        for (int i = 0; i < w.cDOF.Size(); ++i) {
            const std::int64_t ck = packDofKey(w.constrained, w.cDOF(i));
            const std::int64_t rk = packDofKey(w.retained, w.rDOF(i));
            writtenConstrained.insert(ck);
            writtenRetained.insert(rk);
            for (std::int64_t key : {ck, rk}) {
                if (seenKeys.insert(key).second)
                    components[uf.find(key)].push_back(key);
            }
        }
    }

    // Choose deterministic root per component:
    //   1) SP-bearing key (smallest if several) — physics wins
    //   2) else key written only as retained, never as constrained
    //      (preserves equalDOF Rnode Cnode intent / clear retained roots)
    //   3) else smallest (node,dof)
    std::unordered_map<std::int64_t, std::int64_t> rootOf; // component -> root key
    for (auto &kv : components) {
        auto &members = kv.second;
        std::sort(members.begin(), members.end());
        std::int64_t chosen = members.front();
        int bestRank = 3; // 0=SP, 1=retained-only, 2=fallback
        for (std::int64_t key : members) {
            const bool spHere = hasSP(unpackNode(key), unpackDof(key));
            const bool retainedOnly = writtenRetained.count(key) && !writtenConstrained.count(key);
            const int rank = spHere ? 0 : (retainedOnly ? 1 : 2);
            if (rank < bestRank || (rank == bestRank && key < chosen)) {
                chosen = key;
                bestRank = rank;
            }
        }
        // Conflicting SPs with different values: hard error
        if (bestRank == 0) {
            SP_Constraint *ref = spMap[packDofKey(unpackNode(chosen), unpackDof(chosen))];
            const double refVal = ref->getValue();
            for (std::int64_t key : members) {
                auto it = spMap.find(key);
                if (it == spMap.end())
                    continue;
                if (std::fabs(it->second->getValue() - refVal) > 1.0e-12) {
                    opserr << "ERROR " << handlerLabel << ": "
                           << "conflicting SP values in equalDOF group involving nodes "
                           << unpackNode(chosen) << " and " << unpackNode(key) << "\n";
                    return -10;
                }
            }
        }
        rootOf[kv.first] = chosen;
    }

    // Emit star equalDOFs. Each DOF-component may prefer a different root
    // (e.g. SP on UY/UZ of a roller retained node vs free UX clique root).
    //
    // Root choice (above) already honors user retained/constrained intent when
    // possible (SP, then written-retained-only). Transformation attaches at
    // most one MP -- hence one retained node -- per constrained node, so a
    // constrained node with several preferred roots must be resolved by
    // actually moving clique roots (not merely relabeling the star edge):
    //   1) coalesce: move every affected component onto one preferred retained
    //      node when that node is equalDOF-equivalent on each DOF (preserves
    //      user orientation and leaves that retained node free);
    //   2) else re-root onto the shared constrained node (identity equalDOF is
    //      symmetric), keeping SP-bearing roots when possible;
    //   3) never place an SP on a newly constrained DOF.
    // Passes run to a fixed point in sorted node order (deterministic).
    // stars[constrainedNode][retainedNode] = list of (cDof, rDof) pairs
    using RootDofMap = std::unordered_map<int, std::vector<std::pair<int, int>>>;
    using StarMap = std::unordered_map<int, RootDofMap>;

    int starBuildError = 0;
    auto buildStarsFromRoots = [&]() -> StarMap {
        StarMap out;
        starBuildError = 0;
        out.reserve(components.size());
        for (auto &kv : components) {
            auto rit = rootOf.find(kv.first);
            if (rit == rootOf.end()) {
                starBuildError = -18;
                out.clear();
                return out;
            }
            const std::int64_t rootKey = rit->second;
            const int rootNode = unpackNode(rootKey);
            const int rootDof = unpackDof(rootKey);
            for (std::int64_t key : kv.second) {
                if (key == rootKey)
                    continue;
                const int cNode = unpackNode(key);
                const int cDof = unpackDof(key);
                if (cDof != rootDof) {
                    // Defensive: identity equalDOF union-find is single-DOF per component.
                    opserr << "ERROR " << handlerLabel << ": "
                           << "equalDOF clique couples different DOF indices (" << rootDof + 1
                           << " and " << cDof + 1 << ") -- not representable as Transformation MP\n";
                    starBuildError = -11;
                    out.clear();
                    return out;
                }
                out[cNode][rootNode].emplace_back(cDof, rootDof);
            }
        }
        return out;
    };

    // Do not call uf.find on absent keys -- find() would insert phantoms.
    auto dofSameComponent = [&](int nodeA, int nodeB, int dof) -> bool {
        const std::int64_t ka = packDofKey(nodeA, dof);
        const std::int64_t kb = packDofKey(nodeB, dof);
        if (!uf.contains(ka) || !uf.contains(kb))
            return false;
        return uf.find(ka) == uf.find(kb);
    };

    auto pickUnifiedRoot = [&](const RootDofMap &byRoot) -> int {
        int unified = byRoot.begin()->first;
        bool foundSP = false;
        for (const auto &byR : byRoot) {
            const int rNode = byR.first;
            bool spHere = false;
            for (const auto &pr : byR.second) {
                if (hasSP(rNode, pr.second)) {
                    spHere = true;
                    break;
                }
            }
            if (spHere && (!foundSP || rNode < unified)) {
                unified = rNode;
                foundSP = true;
            } else if (!foundSP && rNode < unified) {
                unified = rNode;
            }
        }
        return unified;
    };

    auto rootsAreUnifiable = [&](const RootDofMap &byRoot, int unified, int &conflictR,
                                 int &conflictDof) -> bool {
        for (const auto &byR : byRoot) {
            const int rNode = byR.first;
            if (rNode == unified)
                continue;
            for (const auto &pr : byR.second) {
                const int dof = pr.first;
                if (!dofSameComponent(unified, rNode, dof)) {
                    conflictR = rNode;
                    conflictDof = dof;
                    return false;
                }
            }
        }
        return true;
    };

    auto formatNodeList = [](const RootDofMap &byRoot) -> std::string {
        std::vector<int> tags;
        tags.reserve(byRoot.size());
        for (const auto &byR : byRoot)
            tags.push_back(byR.first);
        std::sort(tags.begin(), tags.end());
        std::ostringstream os;
        for (std::size_t i = 0; i < tags.size(); ++i) {
            if (i)
                os << ", ";
            os << tags[i];
        }
        return os.str();
    };

    // Move listed (cNode,dof) components onto newRootNode. skipSPRoots keeps
    // components whose current root already carries an SP on that DOF.
    // Returns: 1 = rootOf changed, 0 = already there, -19 = SP would be constrained.
    auto moveComponentsToNode = [&](int cNode, const RootDofMap &byRoot, int newRootNode,
                                    bool skipSPRoots) -> int {
        bool changed = false;
        for (const auto &byR : byRoot) {
            const int curRoot = byR.first;
            for (const auto &pr : byR.second) {
                const int dof = pr.first;
                if (skipSPRoots && hasSP(curRoot, dof))
                    continue;
                const std::int64_t memberKey = packDofKey(cNode, dof);
                const std::int64_t newRoot = packDofKey(newRootNode, dof);
                if (!uf.contains(memberKey) || !uf.contains(newRoot))
                    continue;
                if (uf.find(memberKey) != uf.find(newRoot))
                    continue;
                const std::int64_t comp = uf.find(newRoot);
                auto cit = components.find(comp);
                if (cit == components.end())
                    continue;
                for (std::int64_t key : cit->second) {
                    if (key == newRoot)
                        continue;
                    if (hasSP(unpackNode(key), unpackDof(key)))
                        return -19;
                }
                auto rit = rootOf.find(comp);
                if (rit == rootOf.end())
                    return -19;
                if (rit->second != newRoot) {
                    rit->second = newRoot;
                    changed = true;
                }
            }
        }
        return changed ? 1 : 0;
    };

    StarMap stars = buildStarsFromRoots();
    if (starBuildError < 0)
        return starBuildError;

    // Resolve multi-retained constrained nodes to a fixed point.
    {
        const int maxPasses = static_cast<int>(components.size()) + 2;
        for (int pass = 0; pass < maxPasses; ++pass) {
            std::vector<int> multiNodes;
            multiNodes.reserve(stars.size());
            for (const auto &byConstrained : stars) {
                if (byConstrained.second.size() > 1)
                    multiNodes.push_back(byConstrained.first);
            }
            if (multiNodes.empty())
                break;
            std::sort(multiNodes.begin(), multiNodes.end());

            bool changed = false;
            for (int cNode : multiNodes) {
                auto sit = stars.find(cNode);
                if (sit == stars.end() || sit->second.size() <= 1)
                    continue;
                const RootDofMap &byRoot = sit->second;

                const int unified = pickUnifiedRoot(byRoot);
                int conflictR = -1;
                int conflictDof = -1;
                const bool unifiable =
                    rootsAreUnifiable(byRoot, unified, conflictR, conflictDof);

                const char *strategy = nullptr;
                int spBlocked = 0;
                auto tryMove = [&](int newRootNode, bool skipSPRoots, const char *name) -> bool {
                    const int rc = moveComponentsToNode(cNode, byRoot, newRootNode, skipSPRoots);
                    if (rc > 0) {
                        strategy = name;
                        return true;
                    }
                    if (rc < 0)
                        spBlocked = rc;
                    return false;
                };

                // Prefer real coalesce (keeps user retained) before re-rooting.
                if (!(unifiable && tryMove(unified, false, "coalesce")) &&
                    !tryMove(cNode, true, "re-root") && !tryMove(cNode, false, "re-root")) {
                    if (spBlocked < 0) {
                        opserr << "ERROR " << handlerLabel << ": "
                               << "node " << cNode
                               << " is constrained by equalDOF to multiple retained nodes ("
                               << formatNodeList(byRoot).c_str()
                               << ") on different DOFs. Transformation allows only one "
                               << "retained node per constrained node. Resolving that layout "
                               << "is blocked because an SP would end up on a constrained DOF. "
                               << "Reorient equalDOF so the shared node is retained, "
                               << "or use Penalty/Lagrange\n";
                        return -19;
                    }
                    opserr << "ERROR " << handlerLabel << ": "
                           << "node " << cNode
                           << " is constrained by equalDOF to multiple retained nodes ("
                           << formatNodeList(byRoot).c_str()
                           << "). Transformation allows only one retained node per "
                           << "constrained node, and those retained nodes are not "
                           << "equalDOF-equivalent";
                    if (conflictDof >= 0) {
                        opserr << " on DOF " << conflictDof + 1 << " (e.g. nodes " << unified
                               << " and " << conflictR << ")";
                    }
                    opserr << ". Reorient equalDOF so the shared node is retained, "
                           << "or use Penalty/Lagrange\n";
                    return -25;
                }

                changed = true;
                if (std::strcmp(strategy, "re-root") == 0) {
                    const std::string msg =
                        "re-oriented equalDOF around node " + std::to_string(cNode) +
                        " (was multi-retained via nodes " + formatNodeList(byRoot) +
                        "; Transformation allows one retained node per constrained node)";
                    report.warn(msg);
                    report.change("equalDOF: " + msg);
                    opserr << "WARNING rewriteMPConstraints: " << msg.c_str() << "\n";
                } else {
                    report.change("equalDOF: coalesce multi-retained node " +
                                  std::to_string(cNode) + " onto retained node " +
                                  std::to_string(unified) + " (nodes " +
                                  formatNodeList(byRoot) + ")");
                }
            }

            if (!changed) {
                opserr << "ERROR " << handlerLabel << ": "
                       << "could not resolve multi-retained equalDOF layout\n";
                return -25;
            }
            stars = buildStarsFromRoots();
            if (starBuildError < 0)
                return starBuildError;
        }
        for (const auto &byConstrained : stars) {
            if (byConstrained.second.size() > 1) {
                opserr << "ERROR " << handlerLabel << ": "
                       << "could not resolve multi-retained equalDOF layout for node "
                       << byConstrained.first << "\n";
                return -25;
            }
        }
    }

    std::vector<WorkingMP> resolvedIdentity;
    resolvedIdentity.reserve(stars.size());
    std::vector<int> constrainedTags;
    constrainedTags.reserve(stars.size());
    for (const auto &byConstrained : stars)
        constrainedTags.push_back(byConstrained.first);
    std::sort(constrainedTags.begin(), constrainedTags.end());

    for (int cNode : constrainedTags) {
        const RootDofMap &byRoot = stars[cNode];
        if (byRoot.empty())
            continue;
        // After the fixed-point pass each constrained node has one retained.
        const int unified = byRoot.begin()->first;

        std::size_t nPairs = 0;
        for (const auto &byR : byRoot)
            nPairs += byR.second.size();
        std::vector<std::pair<int, int>> pairs;
        pairs.reserve(nPairs);
        for (const auto &byR : byRoot) {
            for (const auto &pr : byR.second)
                pairs.emplace_back(pr.first, pr.first);
        }
        std::sort(pairs.begin(), pairs.end());
        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

        WorkingMP w;
        w.retained = unified;
        w.constrained = cNode;
        w.cDOF = ID(static_cast<int>(pairs.size()));
        w.rDOF = ID(static_cast<int>(pairs.size()));
        w.C = Matrix(static_cast<int>(pairs.size()), static_cast<int>(pairs.size()));
        w.C.Zero();
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            w.cDOF(static_cast<int>(i)) = pairs[i].first;
            w.rDOF(static_cast<int>(i)) = pairs[i].second;
            w.C(static_cast<int>(i), static_cast<int>(i)) = 1.0;
        }
        w.identity = true;
        resolvedIdentity.push_back(std::move(w));
    }

    // Record equalDOF rewrite notes only when the identity-MP set actually changes.
    {
        std::vector<std::string> beforeId;
        std::vector<std::string> afterId;
        beforeId.reserve(identityMPs.size());
        afterId.reserve(resolvedIdentity.size());
        for (const auto &w : identityMPs)
            beforeId.push_back(mpSignature(w));
        for (const auto &w : resolvedIdentity)
            afterId.push_back(mpSignature(w));
        int nRem = 0, nAdd = 0;
        if (mpDefinitionDifference(beforeId, afterId, nRem, nAdd) > 0) {
            report.change("equalDOF groups -> Transformation-ready star layout");
            for (const auto &w : resolvedIdentity) {
                std::ostringstream dofSs;
                for (int i = 0; i < w.cDOF.Size(); ++i) {
                    if (i)
                        dofSs << ' ';
                    dofSs << (w.cDOF(i) + 1);
                }
                report.change("equalDOF: retained " + std::to_string(w.retained) +
                              "  constrained " + std::to_string(w.constrained) + "  DOFs [" +
                              dofSs.str() + "]");
            }
        }
    }

    // --- flip general MPs that have SPs on constrained DOFs ---
    for (auto &w : generalMPs) {
        bool spOnConstrained = false;
        for (int i = 0; i < w.cDOF.Size(); ++i) {
            if (hasSP(w.constrained, w.cDOF(i))) {
                spOnConstrained = true;
                break;
            }
        }
        if (!spOnConstrained)
            continue;

        bool spOnRetained = false;
        for (int i = 0; i < w.rDOF.Size(); ++i) {
            if (hasSP(w.retained, w.rDOF(i))) {
                spOnRetained = true;
                break;
            }
        }

        if (spOnRetained) {
            opserr << "ERROR " << handlerLabel << ": "
                   << "MP has fix/sp constraints on both sides (retained node " << w.retained
                   << ", constrained node " << w.constrained << "). Flipping would leave "
                   << "fix/sp constraints on the new constrained side, so the MP cannot be "
                   << "made Transformation-ready\n";
            return -24;
        }

        if (w.C.noRows() != w.C.noCols()) {
            opserr << "ERROR " << handlerLabel << ": "
                   << "SP on constrained DOFs of MP (retained " << w.retained << ", constrained "
                   << w.constrained << ") but constraint matrix is not square — cannot flip\n";
            return -12;
        }

        Matrix Cinv(w.C.noRows(), w.C.noCols());
        if (w.C.Invert(Cinv) < 0) {
            opserr << "ERROR " << handlerLabel << ": "
                   << "SP on constrained DOFs of MP (retained " << w.retained << ", constrained "
                   << w.constrained << ") but constraint matrix is not invertible — cannot flip\n";
            return -13;
        }

        const int oldR = w.retained;
        const int oldC = w.constrained;
        // Swap roles: old retained becomes constrained
        std::swap(w.retained, w.constrained);
        std::swap(w.cDOF, w.rDOF);
        w.C = Cinv;
        w.identity = isIdentityEqualDOF(w.C, w.cDOF, w.rDOF);
        ++report.nFlipped;
        report.change("flip MP (SP on constrained side, C invertible): retained " +
                      std::to_string(oldR) + " constrained " + std::to_string(oldC) +
                      "  ->  retained " + std::to_string(w.retained) + " constrained " +
                      std::to_string(w.constrained) + " (C inverted)");
    }

    // Combine for chain composition: identity stars + general
    std::vector<WorkingMP> all = std::move(resolvedIdentity);
    all.insert(all.end(), std::make_move_iterator(generalMPs.begin()),
               std::make_move_iterator(generalMPs.end()));

    // constrained (node,dof) -> (mpIndex, row). Stable under composition
    // (only retained sides change; constrained nodes/DOFs do not).
    // Time-varying MPs are kept aside but still occupy constrained DOFs.
    std::unordered_set<std::int64_t> takenConstrainedDofs;
    for (const auto &p : ownedMPs) {
        const ID &cd = p->getConstrainedDOFs();
        for (int i = 0; i < cd.Size(); ++i)
            takenConstrainedDofs.insert(packDofKey(p->getNodeConstrained(), cd(i)));
    }

    std::unordered_map<std::int64_t, std::pair<int, int>> cIndex;
    for (int m = 0; m < static_cast<int>(all.size()); ++m) {
        for (int i = 0; i < all[m].cDOF.Size(); ++i) {
            const std::int64_t key = packDofKey(all[m].constrained, all[m].cDOF(i));
            if (cIndex.find(key) != cIndex.end() || takenConstrainedDofs.count(key)) {
                opserr << "ERROR " << handlerLabel << ": "
                       << "duplicate constrained DOF " << all[m].cDOF(i) + 1 << " on node "
                       << all[m].constrained << " after rewrite -- not representable as a "
                       << "single Transformation MP\n";
                return -22;
            }
            cIndex[key] = {m, i};
        }
    }

    // Resolve each MP to a free retained root once: resolve inners first, then
    // compose outer with the already-collapsed inner (one matmul jumps to the
    // free root). color: 0 = unseen, 1 = on DFS stack, 2 = fully resolved.
    const int nMP = static_cast<int>(all.size());
    std::vector<char> color(static_cast<std::size_t>(nMP), 0);

    std::function<int(int)> resolve = [&](int m) -> int {
        if (color[static_cast<std::size_t>(m)] == 2)
            return 0;
        if (color[static_cast<std::size_t>(m)] == 1) {
            opserr << "ERROR " << handlerLabel << ": "
                   << "constraint cycle involving node " << all[m].retained << "\n";
            return -15;
        }
        color[static_cast<std::size_t>(m)] = 1;

        WorkingMP &outer = all[m];
        std::vector<int> hitRows;
        int innerMP = -1;
        for (int j = 0; j < outer.rDOF.Size(); ++j) {
            const std::int64_t key = packDofKey(outer.retained, outer.rDOF(j));
            auto it = cIndex.find(key);
            if (it == cIndex.end())
                continue;
            if (it->second.first == m) {
                opserr << "ERROR " << handlerLabel << ": "
                       << "constraint cycle involving node " << outer.retained << "\n";
                return -15;
            }
            if (innerMP < 0)
                innerMP = it->second.first;
            else if (innerMP != it->second.first) {
                opserr << "ERROR " << handlerLabel << ": "
                       << "retained DOFs of MP (node " << outer.constrained
                       << ") depend on multiple constrained nodes — "
                       << "not representable with Transformation "
                       << "(use Penalty/Lagrange or EQ_Constraint)\n";
                return -16;
            }
            hitRows.push_back(j);
        }

        if (innerMP < 0) {
            // Retained node is free — already Transformation-ready.
            color[static_cast<std::size_t>(m)] = 2;
            return 0;
        }

        if (static_cast<int>(hitRows.size()) != outer.rDOF.Size()) {
            opserr << "ERROR " << handlerLabel << ": "
                   << "partial chain composition for node " << outer.constrained
                   << " is not supported — retained DOFs only partly "
                   << "constrained\n";
            return -17;
        }

        const int rcInner = resolve(innerMP);
        if (rcInner < 0)
            return rcInner;

        WorkingMP &inner = all[innerMP];
        // Uc_outer = C_outer * Ur_outer, Ur_outer = C_inner * Ur_inner
        // (inner already points at the free root).
        if (inner.constrained != outer.retained) {
            opserr << "ERROR " << handlerLabel << ": "
                   << "internal chain bookkeeping error\n";
            return -18;
        }

        Matrix CinnerRows(outer.rDOF.Size(), inner.rDOF.Size());
        CinnerRows.Zero();
        for (int j = 0; j < outer.rDOF.Size(); ++j) {
            const int dof = outer.rDOF(j);
            int row = -1;
            for (int k = 0; k < inner.cDOF.Size(); ++k) {
                if (inner.cDOF(k) == dof) {
                    row = k;
                    break;
                }
            }
            if (row < 0) {
                opserr << "ERROR " << handlerLabel << ": "
                       << "DOF " << dof + 1 << " of node " << outer.retained
                       << " missing from inner constraint\n";
                return -23;
            }
            for (int col = 0; col < inner.rDOF.Size(); ++col)
                CinnerRows(j, col) = inner.C(row, col);
        }

        Matrix Cnew(outer.C.noRows(), inner.rDOF.Size());
        Cnew.addMatrixProduct(0.0, outer.C, CinnerRows, 1.0);

        const int oldRetained = outer.retained;
        const int newRetained = inner.retained;
        outer.retained = inner.retained;
        outer.rDOF = inner.rDOF;
        outer.C = Cnew;
        outer.identity = isIdentityEqualDOF(outer.C, outer.cDOF, outer.rDOF);
        ++report.nComposed;
        report.change("merge chain for constrained node " + std::to_string(outer.constrained) +
                      ": retained " + std::to_string(oldRetained) + " -> " +
                      std::to_string(newRetained) + " (C_new = C_out * C_in)");

        color[static_cast<std::size_t>(m)] = 2;
        return 0;
    };

    for (int m = 0; m < nMP; ++m) {
        const int rc = resolve(m);
        if (rc < 0)
            return rc;
    }

    // All entries in `all` remain valid MPs (outers were rewritten in place).
    // Remove duplicates: same (retained,constrained,dofs).
    std::vector<WorkingMP> unique;
    unique.reserve(all.size());
    std::unordered_set<std::string> sigs;
    for (auto &w : all) {
        std::string sig = std::to_string(w.retained) + ">" + std::to_string(w.constrained) + ":";
        for (int i = 0; i < w.cDOF.Size(); ++i)
            sig += std::to_string(w.cDOF(i)) + "/" + std::to_string(w.rDOF(i)) + ",";
        if (sigs.insert(sig).second)
            unique.push_back(std::move(w));
    }

    // Final sanity: no constrained node appears in two MPs (including any
    // time-varying MPs already stashed in ownedMPs).
    std::unordered_set<int> constrainedNodes;
    for (const auto &p : ownedMPs)
        constrainedNodes.insert(p->getNodeConstrained());

    for (const auto &w : unique) {
        if (!constrainedNodes.insert(w.constrained).second) {
            opserr << "ERROR " << handlerLabel << ": "
                   << "multiple MPs remain on constrained node " << w.constrained
                   << " after normalization\n";
            return -20;
        }
    }

    // A retained DOF used by an MP must not itself be a constrained DOF.
    // (Same node may be constrained on other DOFs -- Transformation handles
    // that; a node-level ban would reject legal disjoint-DOF layouts.)
    std::unordered_set<std::int64_t> constrainedDofs;
    for (const auto &p : ownedMPs) {
        const ID &cd = p->getConstrainedDOFs();
        for (int i = 0; i < cd.Size(); ++i)
            constrainedDofs.insert(packDofKey(p->getNodeConstrained(), cd(i)));
    }
    for (const auto &w : unique) {
        for (int i = 0; i < w.cDOF.Size(); ++i)
            constrainedDofs.insert(packDofKey(w.constrained, w.cDOF(i)));
    }
    for (const auto &w : unique) {
        for (int i = 0; i < w.rDOF.Size(); ++i) {
            const std::int64_t key = packDofKey(w.retained, w.rDOF(i));
            if (constrainedDofs.find(key) == constrainedDofs.end())
                continue;
            opserr << "ERROR " << handlerLabel << ": "
                   << "retained node " << w.retained << " DOF " << w.rDOF(i) + 1
                   << " is still constrained after composition (unsupported topology)\n";
            return -21;
        }
    }

    for (auto &w : unique)
        ownedMPs.push_back(makeMP(w));

    report.nOut = static_cast<int>(ownedMPs.size());
    std::vector<std::string> afterSigs;
    afterSigs.reserve(ownedMPs.size());
    for (const auto &p : ownedMPs) {
        WorkingMP tmp;
        tmp.retained = p->getNodeRetained();
        tmp.constrained = p->getNodeConstrained();
        tmp.cDOF = p->getConstrainedDOFs();
        tmp.rDOF = p->getRetainedDOFs();
        tmp.C = p->getConstraint();
        tmp.identity = isIdentityEqualDOF(tmp.C, tmp.cDOF, tmp.rDOF);
        report.afterLine(formatMPLine(tmp) + (p->isTimeVarying() ? "  (time-varying)" : ""));
        afterSigs.push_back(mpSignature(tmp));
    }
    report.nDifferingMPs =
        mpDefinitionDifference(beforeSigs, afterSigs, report.nRemovedMPs, report.nAddedMPs);

    return 0;
}

void emitRewriteReport(const RewriteReport &report, bool verbose, const char *fileName,
                       bool proposed) {
    const bool wantFile = fileName != nullptr && fileName[0] != '\0';
    if (!verbose && !wantFile)
        return;
    const std::string text = report.str("rewriteMPConstraints", proposed);
    if (verbose)
        opserr << text.c_str();
    if (wantFile) {
        std::ofstream out(fileName);
        if (!out) {
            opserr << "WARNING rewriteMPConstraints: could not open report file '" << fileName
                   << "'\n";
            return;
        }
        out << text;
    }
}

void emitCheckOnlyOneLiner(const RewriteReport &report) {
    if (report.nDifferingMPs == 0) {
        opserr << "rewriteMPConstraints -checkOnly: yes — already Transformation-ready\n";
    } else {
        opserr << "rewriteMPConstraints -checkOnly: no — not Transformation-ready; "
                  "rewrite is possible\n";
    }
}

void emitApplyOneLiner(const RewriteReport &report) {
    if (report.nDifferingMPs == 0) {
        opserr << "rewriteMPConstraints: nothing to do — already Transformation-ready\n";
    } else {
        opserr << "rewriteMPConstraints: done — multipoint constraints rewritten into "
                  "Transformation-ready form\n";
    }
}

int runRewrite(Domain &theDomain, const RewriteMPOptions &opts) {
    RewriteReport report;
    std::vector<std::unique_ptr<MP_Constraint>> newMPs;
    const int rc = rewriteMPs(theDomain, newMPs, "rewriteMPConstraints", report);
    if (rc < 0) {
        // Impossible to produce a Transformation-ready layout.
        if (opts.checkOnly) {
            opserr << "rewriteMPConstraints -checkOnly: no — not Transformation-ready; "
                      "rewrite is NOT possible\n";
        }
        return rc;
    }

    if (opts.checkOnly) {
        emitCheckOnlyOneLiner(report);
        emitRewriteReport(report, opts.verbose, opts.fileName, /*proposed=*/true);
        return report.nDifferingMPs; // 0 ready, >0 needs rewrite but possible
    }

    // Apply: replace Domain MPs only when something actually changes.
    if (report.nDifferingMPs > 0) {
        std::vector<int> tags;
        tags.reserve(static_cast<std::size_t>(theDomain.getNumMPs()));
        {
            MP_ConstraintIter &it = theDomain.getMPs();
            MP_Constraint *mp = nullptr;
            while ((mp = it()) != nullptr)
                tags.push_back(mp->getTag());
        }
        for (int tag : tags) {
            MP_Constraint *old = theDomain.removeMP_Constraint(tag);
            if (old != nullptr)
                delete old;
        }

        for (auto &p : newMPs) {
            MP_Constraint *mp = p.release();
            if (theDomain.addMP_Constraint(mp) == false) {
                opserr << "ERROR rewriteMPConstraints: failed to add MP tag " << mp->getTag()
                       << "\n";
                delete mp;
                return -30;
            }
        }
    }

    emitApplyOneLiner(report);
    emitRewriteReport(report, opts.verbose, opts.fileName, /*proposed=*/false);
    return report.nDifferingMPs;
}

} // namespace

int rewriteMPConstraints(Domain &theDomain, const RewriteMPOptions &opts) {
    return runRewrite(theDomain, opts);
}

int OPS_rewriteMPConstraints() {
    Domain *theDomain = OPS_GetDomain();
    if (theDomain == nullptr) {
        opserr << "WARNING rewriteMPConstraints: no Domain\n";
        return -1;
    }

    RewriteMPOptions opts;
    while (OPS_GetNumRemainingInputArgs() > 0) {
        const char *opt = OPS_GetString();
        if (opt == nullptr)
            break;
        if (strcmp(opt, "-verbose") == 0 || strcmp(opt, "-Verbose") == 0) {
            opts.verbose = true;
        } else if (strcmp(opt, "-checkOnly") == 0 || strcmp(opt, "-CheckOnly") == 0 ||
                   strcmp(opt, "-check") == 0 || strcmp(opt, "-Check") == 0) {
            // -check kept as a synonym for -checkOnly
            opts.checkOnly = true;
        } else if (strcmp(opt, "-file") == 0 || strcmp(opt, "-File") == 0) {
            if (OPS_GetNumRemainingInputArgs() < 1) {
                opserr << "WARNING rewriteMPConstraints -file requires a file name\n";
                return -1;
            }
            opts.fileName = OPS_GetString();
        } else {
            opserr << "WARNING rewriteMPConstraints unknown option '" << opt
                   << "' — expected -checkOnly, -verbose, or -file name\n";
            return -1;
        }
    }

    const int rc = rewriteMPConstraints(*theDomain, opts);
    int numData = 1;
    int data = rc;
    OPS_SetIntOutput(&numData, &data, true);
    return rc;
}
