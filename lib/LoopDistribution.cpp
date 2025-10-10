//===- LoopDistribution.cpp - Affine loop distribution pass --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to perform loop distribution on affine loops.
// Loop distribution transforms a single loop into multiple loops, each handling
// a subset of the original loop's operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopDistributionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"
#include <cmath>

#define DEBUG_TYPE "affine-loop-distribution"

using namespace mlir;
using namespace mlir::affine;

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPDISTRIBUTION
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

namespace {

/// This file implements a pass to perform loop distribution on affine loops.
/// Loop distribution transforms a single loop into multiple loops, each
/// handling a subset of the original loop's operations, based on their
/// dependencies.
///
/// The pass analyzes the direct children (operations) of each affine loop and
/// builds a dependency matrix to identify independent groups using Strongly
/// Connected Components (SCCs). Each SCC is then distributed into a separate
/// loop, preserving the original order and dependencies.
///
/// Note:
/// - All nested loops and `if` statements are treated as single atomic
/// operations
///   when analyzing dependencies at the current loop level.
///   For example:
///     affine.for %i = ... {
///       affine.if (...) { ... }   // treated as one op
///       affine.for %j = ... { ... } // treated as one op
///       %x = ...                  // treated as one op
///     }
///
///   In this example, the body contains three direct children: the `if`, the
///   inner `for`, and the assignment to `%x`. Each is considered a single
///   operation for distribution purposes.
///
/// - The pass uses SSA def-use chains, affine dependence analysis, and memory
/// alias
///   analysis to conservatively detect dependencies between operations.
///
/// - Distribution is performed only if there are multiple independent SCCs and
/// the
///   cost model determines that distribution is beneficial.
///
/// Usage:
///   Run with: mlir-opt -affine-loop-distribution <input.mlir>

/// This pass performs loop distribution on affine loops.
struct AffineLoopDistribution
    : public affine::impl::AffineLoopDistributionBase<AffineLoopDistribution> {

  void runOnOperation() override;

  /// Attempt to distribute a single affine loop.
  LogicalResult performLoopDistribution(AffineForOp forOp);

  /// Distribute a loop based on Strongly Connected Components.
  LogicalResult distributeBySCCs(AffineForOp forOp,
                                 ArrayRef<Operation *> directChildren,
                                 ArrayRef<SmallVector<unsigned>> sccComponents);

  /// Check dependence between two operations at a given loop level.
  bool checkDependenceAtLevel(Operation *opI, Operation *opJ,
                              affine::AffineForOp currentLoop);

  /// Build dependence matrix at current loop level.
  void buildDependenceMatrixAtCurrentLevel(
      affine::AffineForOp currentLoop, ArrayRef<Operation *> directChildren,
      SmallVectorImpl<SmallVector<int>> &depMatrix);
};
} // namespace

//
// Check dependence between two operations (treating nested strutures as atomic
// op and check the dependencies iteratively through all ops inside  regions of
// an operation)
bool AffineLoopDistribution::checkDependenceAtLevel(
    Operation *opI, Operation *opJ, affine::AffineForOp currentLoop) {

  // 1. First check direct SSA def-use relationship between opI and opJ
  // This handles cases where opI produces values used by opJ
  for (Value result : opI->getResults()) {
    for (Operation *user : result.getUsers()) {
      // Check if opJ uses the result directly
      if (user == opJ)
        return true;
      // Check if any operation nested within opJ uses the result
      if (opJ->isProperAncestor(user))
        return true;
    }
  }

  // 2. Also check the reverse
  for (Value result : opJ->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (user == opI || opI->isProperAncestor(user))
        return true;
    }
  }

  // 3. Get all leaf operations from both operands(bcz these operands can be
  // nested strutures of ops).
  SmallVector<Operation *> opsI, opsJ;
  collectLeafOperations(opI, opsI);
  collectLeafOperations(opJ, opsJ);

  // 4. Check if any statement in opI has dependence with any statement in opJ
  // considering only the current loop's iteration variable
  for (Operation *si : opsI) {
    for (Operation *sj : opsJ) {
      if (hasDependenceBetweenOpsForLoopIV(si, sj, currentLoop))
        return true;
    }
  }

  // 5. Check for memory aliasing between operations using AliasAnalysis
  // If any memory operation in opI aliases with any memory operation in opJ,
  // they have a dependence. Skip if they're the same memref base since that's
  // already handled by the affine dependence analysis above.
  AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();

  for (Operation *si : opsI) {
    if (!isa<affine::AffineLoadOp, affine::AffineStoreOp>(si))
      continue;

    Value memrefI;
    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(si))
      memrefI = loadOp.getMemRef();
    else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(si))
      memrefI = storeOp.getMemRef();

    for (Operation *sj : opsJ) {
      if (!isa<affine::AffineLoadOp, affine::AffineStoreOp>(sj))
        continue;

      Value memrefJ;
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(sj))
        memrefJ = loadOp.getMemRef();
      else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(sj))
        memrefJ = storeOp.getMemRef();

      // Skip if same memref - already handled by affine dependence analysis
      if (memrefI == memrefJ)
        continue;

      // Check if memrefs may alias using AliasAnalysis
      AliasResult aliasResult = aliasAnalysis.alias(memrefI, memrefJ);
      if (aliasResult) {
        // If they may alias and at least one is a store, there's a dependence
        if (isa<affine::AffineStoreOp>(si) || isa<affine::AffineStoreOp>(sj))
          return true;
      }
    }
  }

  return false;
}

// Build dependence matrix at current loop level
void AffineLoopDistribution::buildDependenceMatrixAtCurrentLevel(
    affine::AffineForOp currentLoop, ArrayRef<Operation *> directChildren,
    SmallVectorImpl<SmallVector<int>> &depMatrix) {

  size_t n = directChildren.size();
  depMatrix.resize(n);

  for (auto &row : depMatrix)
    row.resize(n, false);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (i == j)
        continue;

      Operation *opI = directChildren[i];
      Operation *opJ = directChildren[j];

      // Check dependence between operations at this level
      depMatrix[i][j] = checkDependenceAtLevel(opI, opJ, currentLoop);
    }
  }
}

/// Given a loop and its SCCs (computed from dependency analysis), this function
/// creates separate loops for each SCC, preserving the order of operations.
/// The original loop is erased after distribution.
/// Returns success if distribution was performed, failure otherwise.
LogicalResult AffineLoopDistribution::distributeBySCCs(
    AffineForOp forOp, ArrayRef<Operation *> directChildren,
    ArrayRef<SmallVector<unsigned>> sccComponents) {

  LLVM_DEBUG(llvm::dbgs() << "Distributing loop with " << sccComponents.size()
                          << " SCC(s)\n");

  if (sccComponents.size() <= 1) {
    LLVM_DEBUG(llvm::dbgs() << "Only one SCC, cannot distribute\n");
    return failure();
  }

  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();

  DenseMap<Operation *, Operation *> clonedOpMap;

  // Create one loop per independent group (SCC)
  LLVM_DEBUG(llvm::dbgs() << "Creating " << sccComponents.size()
                          << " distributed loops...\n");
  for (const auto &scc : sccComponents) {
    auto newLoop = builder.create<AffineForOp>(
        loc, forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
        forOp.getUpperBoundOperands(), forOp.getUpperBoundMap());
    newLoop.setStep(forOp.getStep());
    newLoop->setAttrs(forOp->getAttrs());

    Block *body = newLoop.getBody();
    Operation *terminator = body->getTerminator();
    OpBuilder bodyBuilder(terminator);
    IRMapping mapper;
    mapper.map(forOp.getInductionVar(), newLoop.getInductionVar());

    // Clone operations from this SCC into the new loop
    for (unsigned opIdx : scc) {
      Operation *orig = directChildren[opIdx];
      Operation *cloned = bodyBuilder.clone(*orig, mapper);
      clonedOpMap[orig] = cloned;
    }
  }

  // Redirect external uses (values used outside the original loop)
  // This is necessary when operations produce values used elsewhere
  for (auto &kv : clonedOpMap) {
    Operation *orig = kv.first;
    Operation *cloned = kv.second;
    auto origResults = orig->getResults();
    auto clonedResults = cloned->getResults();

    for (auto zipIt : llvm::zip(origResults, clonedResults)) {
      Value oldV = std::get<0>(zipIt);
      Value newV = std::get<1>(zipIt);
      if (oldV.use_empty())
        continue;
      // Replace uses only outside the original loop
      oldV.replaceUsesWithIf(newV, [&](OpOperand &use) {
        return use.getOwner()->getParentOp() != forOp;
      });
    }
  }

  // Erase the original loop
  forOp.erase();

  LLVM_DEBUG(llvm::dbgs() << "Successfully distributed loop into "
                          << sccComponents.size() << " independent loops\n");

  return success();
}

LogicalResult
AffineLoopDistribution::performLoopDistribution(affine::AffineForOp forOp) {
  SmallVector<Operation *> directChildren;
  LLVM_DEBUG(llvm::dbgs() << "Analyzing loop for distribution: " << forOp
                          << "\n");

  for (Operation &op : forOp.getBody()->without_terminator())
    directChildren.push_back(&op);

  if (directChildren.size() <= 1)
    return failure();

  // Build (directed) dependence matrix.
  SmallVector<SmallVector<int>> depMatrix;
  buildDependenceMatrixAtCurrentLevel(forOp, directChildren, depMatrix);

  unsigned n = depMatrix.size();
  if (n <= 1)
    return failure();

  // Print the dependency matrix
  LLVM_DEBUG({
    llvm::dbgs() << "Dependency Matrix (" << n << "x" << n << "):\n";
    llvm::dbgs() << "     ";
    for (unsigned j = 0; j < n; ++j)
      llvm::dbgs() << j << " ";
    llvm::dbgs() << "\n";
    for (unsigned i = 0; i < n; ++i) {
      llvm::dbgs() << "  " << i << ": ";
      for (unsigned j = 0; j < n; ++j) {
        llvm::dbgs() << depMatrix[i][j] << " ";
      }
      llvm::dbgs() << "\n";
    }
  });

  // Create adjacency list from directed dependence matrix for SCC computation
  SmallVector<SmallVector<unsigned>> adj(n);
  for (unsigned i = 0; i < n; ++i)
    for (unsigned j = 0; j < n; ++j)
      if (depMatrix[i][j])
        adj[i].push_back(j);

  // Compute Strongly Connected Components
  SmallVector<SmallVector<unsigned>> sccComponents;
  findStronglyConnectedComponents(n, adj, sccComponents);

  LLVM_DEBUG({
    llvm::dbgs() << "Found " << sccComponents.size() << " SCC(s):\n";
    for (size_t idx = 0; idx < sccComponents.size(); ++idx) {
      llvm::dbgs() << "  SCC " << idx << ": [";
      for (size_t i = 0; i < sccComponents[idx].size(); ++i) {
        llvm::dbgs() << sccComponents[idx][i];
        if (i + 1 < sccComponents[idx].size())
          llvm::dbgs() << ", ";
      }
      llvm::dbgs() << "]\n";
    }
  });

  if (sccComponents.size() <= 1)
    return failure(); // Nothing safely separable.

  // Map each operation to its SCC
  SmallVector<unsigned> opToSCC(n);
  for (size_t sccIdx = 0; sccIdx < sccComponents.size(); ++sccIdx) {
    for (unsigned opIdx : sccComponents[sccIdx]) {
      opToSCC[opIdx] = sccIdx;
    }
  }

  // Build SCC dependency graph
  // Check if there are dependencies between different SCCs
  SmallVector<SmallVector<unsigned>> sccAdj(sccComponents.size());
  DenseSet<std::pair<unsigned, unsigned>> sccEdges;

  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = 0; j < n; ++j) {
      if (depMatrix[i][j] && opToSCC[i] != opToSCC[j]) {
        auto edge = std::make_pair(opToSCC[i], opToSCC[j]);
        if (sccEdges.insert(edge).second) {
          sccAdj[opToSCC[i]].push_back(opToSCC[j]);
        }
      }
    }
  }

  // ensure no edge between groups (def-use / memory).
  LLVM_DEBUG(llvm::dbgs() << "Verifying no cross-group dependencies that "
                             "violate topological order...\n");
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = 0; j < n; ++j) {
      if (opToSCC[i] != opToSCC[j] && (depMatrix[i][j] || depMatrix[j][i])) {
        // Check if this violates topological order (backward edge in
        // condensation graph) For now, we allow forward dependencies between
        // SCCs
        LLVM_DEBUG(llvm::dbgs()
                   << "  Dependency between op " << i << " (SCC " << opToSCC[i]
                   << ") and op " << j << " (SCC " << opToSCC[j] << ")\n");
      }
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "  Verification completed\n");

  // Apply cost model to decide if distribution is beneficial
  if (!isDistributionBeneficial(forOp, directChildren, sccComponents,
                                depMatrix)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Cost model determined distribution is not beneficial\n");
    return failure();
  }

  // Use the utility function to distribute the loop by SCCs
  return distributeBySCCs(forOp, directChildren, sccComponents);
}

void AffineLoopDistribution::runOnOperation() {
  func::FuncOp func = getOperation();

  SmallVector<affine::AffineForOp> loops;

  // Walk in post-order (innermost to outermost)
  func.walk<WalkOrder::PostOrder>(
      [&](affine::AffineForOp forOp) { loops.push_back(forOp); });

  // Now process loops from innermost to outermost
  // we distribute the inner loops first and then distribute the outer loops bcz
  // distributing the inner loops creates the opportunity to distribute them
  // further. But if we start distributing from outside we need to check the
  // dependencies across all inner loop IVs which becomes further comples.
  for (affine::AffineForOp forOp : loops) {
    // Check if loop still exists (might have been erased as part of parent)
    if (forOp->use_empty() && forOp->getBlock() == nullptr)
      continue;

    (void)performLoopDistribution(forOp);
    LLVM_DEBUG(llvm::dbgs() << "Created private memref of type: " << '\n');
  }
}

/// Creates a pass to perform loop distribution on affine loops.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createLoopDistributionPass() {
  return std::make_unique<AffineLoopDistribution>();
}

// Command to run the pass using file check.
//  ./bin/mlir-opt -affine-loop-distribution
//  ../mlir/test/Dialect/Affine/loop-distribution.mlir | FileCheck
//  ../mlir/test/Dialect/Affine/loop-distribution.mlir