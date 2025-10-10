//===- LoopDistributionUtils.h - Loop distribution utilities ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various loop distribution utility
// methods: these are not passes by themselves but are used either by passes,
// optimization sequences, or in turn by other transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_LOOPDISTRIBUTIONUTILS_H
#define MLIR_DIALECT_AFFINE_LOOPDISTRIBUTIONUTILS_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Operation;

namespace affine {
class AffineForOp;

/// Checks if two operations have a dependence with respect to curretn loop
/// induction variable. Returns true if dependence exists.
bool hasDependenceBetweenOpsForLoopIV(Operation *srcOp, Operation *dstOp,
                                      AffineForOp currLoop);

/// Computes the nesting depth of a loop within its parent loop nest. Returns 0
/// for outermost loops.
unsigned getLoopNestingDepth(AffineForOp loop);

/// Collects all leaf (non-loop) operations from the given operation or nested
/// loop structure. This is useful for identifying actual computation statements
/// in nested loop bodies.
void collectLeafOperations(Operation *root,
                           SmallVectorImpl<Operation *> &leafOps);

/// Find Strongly Connected Components using Kosaraju's algorithm.
/// Returns each SCC as a set of node indices.
void findStronglyConnectedComponents(
    unsigned numNodes, ArrayRef<SmallVector<unsigned>> adj,
    SmallVectorImpl<SmallVector<unsigned>> &sccComponents);

/// Estimate parallelism gain from distribution.
/// Higher score means more parallelism potential.
/// Independent SCCs can potentially be parallelized.
double estimateParallelismGain(ArrayRef<SmallVector<unsigned>> sccComponents,
                               const SmallVector<SmallVector<int>> &depMatrix);

/// Estimate locality cost from distribution.
/// Lower score means better locality (less overhead from distribution).
/// Distribution can hurt temporal locality if the same data is accessed
/// in different distributed loops.
double estimateLocalityCost(AffineForOp forOp,
                            ArrayRef<Operation *> directChildren,
                            ArrayRef<SmallVector<unsigned>> sccComponents);

/// Cost model to determine if distribution is beneficial.
/// Balances parallelism opportunities against locality costs.
bool isDistributionBeneficial(AffineForOp forOp,
                              ArrayRef<Operation *> directChildren,
                              ArrayRef<SmallVector<unsigned>> sccComponents,
                              const SmallVector<SmallVector<int>> &depMatrix);

} // namespace affine
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_LOOPDISTRIBUTIONUTILS_H
