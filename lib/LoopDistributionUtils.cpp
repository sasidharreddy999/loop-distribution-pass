//===- LoopDistributionUtils.cpp - Utilities for loop distribution -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop distribution transformation utility functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/LoopDistributionUtils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-distribution-utils"

using namespace mlir;
using namespace mlir::affine;

// Check if memory access uses a specific value in its indices
static bool accessUsesValue(MemRefAccess &access, Value val) {
  for (Value index : access.indices) {
    if (index == val)
      return true;

    // Check if index is computed from val
    if (Operation *defOp = index.getDefiningOp()) {
      SmallVector<Operation *> worklist = {defOp};
      DenseSet<Operation *> visited;

      while (!worklist.empty()) {
        Operation *curr = worklist.pop_back_val();

        if (!visited.insert(curr).second)
          continue;

        for (Value operand : curr->getOperands()) {
          if (operand == val)
            return true;
          if (Operation *op = operand.getDefiningOp())
            worklist.push_back(op);
        }
      }
    }
  }

  return false;
}

// Iterative DFS
static void dfsIterative(unsigned startNode, SmallVectorImpl<bool> &vis,
                         ArrayRef<SmallVector<unsigned>> adj,
                         SmallVectorImpl<unsigned> &st) {
  // Stack entry: (node, nextChildIndex)
  struct DFSState {
    unsigned node;
    size_t nextChild;
  };

  SmallVector<DFSState> stack;
  stack.push_back({startNode, 0});
  vis[startNode] = true;

  while (!stack.empty()) {
    DFSState &current = stack.back();
    const SmallVector<unsigned> &neighbors = adj[current.node];
    bool foundUnvisited = false;

    // Try to find an unvisited neighbor
    for (size_t i = current.nextChild; i < neighbors.size(); ++i) {
      unsigned neighbor = neighbors[i];
      if (!vis[neighbor]) {
        vis[neighbor] = true;
        current.nextChild = i + 1; // Resume from next neighbor
        stack.push_back({neighbor, 0});
        foundUnvisited = true;
        break;
      }
    }

    // No unvisited neighbors - done with this node
    if (!foundUnvisited) {
      st.push_back(current.node);
      stack.pop_back();
    }
  }
}

/// Find Strongly Connected Components using Kosaraju's algorithm.
void mlir::affine::findStronglyConnectedComponents(
    unsigned numNodes,

    ArrayRef<SmallVector<unsigned>> adj,
    SmallVectorImpl<SmallVector<unsigned>> &sccComponents) {
  // Step 1: Perform DFS on original graph to get finishing times
  SmallVector<bool> vis(numNodes, false);
  SmallVector<unsigned> finishStack;

  for (unsigned i = 0; i < numNodes; ++i) {
    if (!vis[i]) {
      dfsIterative(i, vis, adj, finishStack);
    }
  }

  // Step 2: Create transpose graph
  SmallVector<SmallVector<unsigned>> adjTranspose(numNodes);
  for (unsigned i = 0; i < numNodes; ++i) {
    for (unsigned neighbor : adj[i]) {
      adjTranspose[neighbor].push_back(i);
    }
  }

  // Step 3: Perform DFS on transpose graph in reverse finishing order
  // Each DFS traversal identifies one SCC
  std::fill(vis.begin(), vis.end(), false);

  // Process nodes in reverse order of finishing times
  while (!finishStack.empty()) {
    unsigned node = finishStack.pop_back_val();

    if (!vis[node]) {
      // Found a new SCC - collect all nodes reachable from this node
      SmallVector<unsigned> scc;
      SmallVector<unsigned> stack;
      stack.push_back(node);
      vis[node] = true;

      while (!stack.empty()) {
        unsigned curr = stack.pop_back_val();
        scc.push_back(curr);

        for (unsigned neighbor : adjTranspose[curr]) {
          if (!vis[neighbor]) {
            vis[neighbor] = true;
            stack.push_back(neighbor);
          }
        }
      }

      // Sort SCC for deterministic output
      llvm::sort(scc);
      sccComponents.push_back(std::move(scc));
    }
  }
}

bool mlir::affine::hasDependenceBetweenOpsForLoopIV(Operation *srcOp,
                                                    Operation *dstOp,
                                                    AffineForOp currLoop) {

  // 1. SSA def-use dependence (always matters)
  for (Value operand : dstOp->getOperands()) {
    if (operand.getDefiningOp() == srcOp)
      return true;
  }

  // 2. Memory dependence with respect to current loop
  if (!isa<affine::AffineLoadOp, affine::AffineStoreOp>(srcOp) ||
      !isa<affine::AffineLoadOp, affine::AffineStoreOp>(dstOp))
    return false;

  MemRefAccess srcAccess(srcOp);
  MemRefAccess dstAccess(dstOp);

  // Different memrefs - no dependence
  if (srcAccess.memref != dstAccess.memref)
    return false;

  // Check if accesses depend on current loop variable
  Value currLoopIV = currLoop.getInductionVar();
  bool srcUsesIV = accessUsesValue(srcAccess, currLoopIV);
  bool dstUsesIV = accessUsesValue(dstAccess, currLoopIV);

  // If neither uses current loop IV, they're loop-independent at this level
  // but still may have dependence (same iteration)
  if (!srcUsesIV && !dstUsesIV) {
    // Check for loop-independent (same-iteration) dependence
    FlatAffineValueConstraints constraints;
    DependenceResult result = checkMemrefAccessDependence(
        srcAccess, dstAccess, 0, &constraints, nullptr);
    return result.value != DependenceResult::NoDependence;
  }

  // At least one uses current loop IV - check loop-carried dependence
  // Get the depth of current loop
  unsigned loopDepth = getLoopNestingDepth(currLoop);

  FlatAffineValueConstraints dependenceConstraints;
  DependenceResult result = checkMemrefAccessDependence(
      srcAccess, dstAccess, loopDepth, &dependenceConstraints, nullptr);

  return result.value != DependenceResult::NoDependence;
}

/// Compute the nesting depth of a loop.
unsigned mlir::affine::getLoopNestingDepth(AffineForOp loop) {
  unsigned depth = 1;
  Operation *parent = loop->getParentOp();

  while (parent) {
    if (isa<AffineForOp>(parent)) {
      depth++;
    }
    parent = parent->getParentOp();
  }

  return depth;
}

/// Collect all leaf operations starting from 'root'.
/// A leaf op is one that contains no nested non-terminator operations
/// in any of its regions.
void mlir::affine::collectLeafOperations(
    Operation *root, SmallVectorImpl<Operation *> &leafOps) {
  if (!root)
    return;

  SmallVector<Operation *> stack;
  stack.push_back(root);

  while (!stack.empty()) {
    Operation *op = stack.pop_back_val();

    bool hasNested = false;
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (Operation &nested : block.without_terminator()) {
          hasNested = true;
          stack.push_back(&nested);
        }
      }
    }

    if (!hasNested)
      leafOps.push_back(op);
  }
}

/// Estimate parallelism gain from distribution.
/// Higher score means more parallelism potential.
/// Independent SCCs can potentially be parallelized.
double mlir::affine::estimateParallelismGain(
    ArrayRef<SmallVector<unsigned>> sccComponents,
    const SmallVector<SmallVector<int>> &depMatrix) {

  if (sccComponents.size() <= 1)
    return 0.0;

  unsigned n = depMatrix.size();

  // Map each operation to its SCC
  SmallVector<unsigned> opToSCC(n);
  for (size_t sccIdx = 0; sccIdx < sccComponents.size(); ++sccIdx) {
    for (unsigned opIdx : sccComponents[sccIdx]) {
      opToSCC[opIdx] = sccIdx;
    }
  }

  // Build SCC dependency graph
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

  // Count independent SCCs (no dependencies between them)
  unsigned independentSCCs = 0;
  for (size_t i = 0; i < sccComponents.size(); ++i) {
    if (sccAdj[i].empty()) {
      independentSCCs++;
    }
  }

  // Parallelism gain: ratio of independent SCCs to total SCCs
  // More independent SCCs = more potential for parallel execution
  double parallelismScore =
      static_cast<double>(independentSCCs) / sccComponents.size();

  LLVM_DEBUG(llvm::dbgs() << "  Parallelism score: " << parallelismScore << " ("
                          << independentSCCs << " independent SCCs out of "
                          << sccComponents.size() << ")\n");

  return parallelismScore;
}

/// Estimate locality cost from distribution.
/// Lower score means better locality (less overhead from distribution).
/// Distribution can hurt temporal locality if the same data is accessed
/// in different distributed loops.
double mlir::affine::estimateLocalityCost(
    AffineForOp forOp, ArrayRef<Operation *> directChildren,
    ArrayRef<SmallVector<unsigned>> sccComponents) {

  if (sccComponents.size() <= 1)
    return 0.0;

  // Track which memrefs are accessed in each SCC
  SmallVector<DenseSet<Value>> sccMemrefs(sccComponents.size());

  for (size_t sccIdx = 0; sccIdx < sccComponents.size(); ++sccIdx) {
    for (unsigned opIdx : sccComponents[sccIdx]) {
      Operation *op = directChildren[opIdx];

      // Collect all memory operations
      SmallVector<Operation *> memOps;
      collectLeafOperations(op, memOps);

      for (Operation *memOp : memOps) {
        Value memref;
        if (auto loadOp = dyn_cast<affine::AffineLoadOp>(memOp))
          memref = loadOp.getMemRef();
        else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(memOp))
          memref = storeOp.getMemRef();

        if (memref)
          sccMemrefs[sccIdx].insert(memref);
      }
    }
  }

  // Count shared memrefs between SCCs (indicates locality loss)
  unsigned sharedMemrefs = 0;
  for (size_t i = 0; i < sccComponents.size(); ++i) {
    for (size_t j = i + 1; j < sccComponents.size(); ++j) {
      for (Value memref : sccMemrefs[i]) {
        if (sccMemrefs[j].contains(memref)) {
          sharedMemrefs++;
          break; // Count each pair only once
        }
      }
    }
  }

  // Locality cost: ratio of shared memrefs to total possible pairs
  unsigned totalPairs = (sccComponents.size() * (sccComponents.size() - 1)) / 2;
  double localityCost =
      totalPairs > 0 ? static_cast<double>(sharedMemrefs) / totalPairs : 0.0;

  LLVM_DEBUG(llvm::dbgs() << "  Locality cost: " << localityCost << " ("
                          << sharedMemrefs << " shared memrefs across SCCs)\n");

  return localityCost;
}

/// Cost model to determine if distribution is beneficial.
/// Balances parallelism opportunities against locality costs.
bool mlir::affine::isDistributionBeneficial(
    AffineForOp forOp, ArrayRef<Operation *> directChildren,
    ArrayRef<SmallVector<unsigned>> sccComponents,
    const SmallVector<SmallVector<int>> &depMatrix) {

  LLVM_DEBUG(llvm::dbgs() << "Evaluating cost model for distribution...\n");

  // Don't distribute if only one SCC
  if (sccComponents.size() <= 1) {
    LLVM_DEBUG(llvm::dbgs() << "  Decision: SKIP (only 1 SCC)\n");
    return false;
  }

  // Estimate parallelism gain
  double parallelismGain = estimateParallelismGain(sccComponents, depMatrix);

  // Estimate locality cost
  double localityCost =
      estimateLocalityCost(forOp, directChildren, sccComponents);

  // Simple cost function: benefit = parallelism_gain - locality_cost
  // Weights can be adjusted based on target architecture
  const double parallelismWeight = 1.0; // Favor parallelism
  const double localityWeight = 0.5;    // Penalize locality loss, but less

  double benefit =
      (parallelismWeight * parallelismGain) - (localityWeight * localityCost);

  // Decision threshold: distribute if benefit >= 0
  // When benefit is neutral, favor distribution to expose parallelism
  bool shouldDistribute = benefit >= 0.0;

  LLVM_DEBUG({
    llvm::dbgs() << "  Parallelism gain: " << parallelismGain << "\n";
    llvm::dbgs() << "  Locality cost: " << localityCost << "\n";
    llvm::dbgs() << "  Net benefit: " << benefit << "\n";
    llvm::dbgs() << "  Decision: " << (shouldDistribute ? "DISTRIBUTE" : "SKIP")
                 << "\n";
  });

  return shouldDistribute;
}