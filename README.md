# Affine Loop Distribution Pass

## Overview

The **Affine Loop Distribution Pass** is an MLIR transformation pass that splits a single affine loop into multiple independent loops, each containing a subset of the original loop's operations. This transformation exposes parallelism opportunities and can improve cache locality by grouping memory accesses to the same data structures.

### What is Loop Distribution?

Loop distribution (also known as loop fission) is a compiler optimization that transforms:

```mlir
// BEFORE: Single loop with multiple independent operations
affine.for %i = 0 to N {
  %a = affine.load %A[%i]
  affine.store %a, %B[%i]
  %c = affine.load %C[%i]
  affine.store %c, %D[%i]
}
```

Into:

```mlir
// AFTER: Multiple loops, each with independent operations
affine.for %i = 0 to N {
  %a = affine.load %A[%i]
  affine.store %a, %B[%i]
}
affine.for %i = 0 to N {
  %c = affine.load %C[%i]
  affine.store %c, %D[%i]
}
```

## Benefits

1. **Parallelism Exposure**: Independent loops can be executed in parallel on different cores or SIMD units.
2. **Cache Locality**: Grouping operations that access the same memory can improve temporal locality.
3. **Vectorization Opportunities**: Simpler loop bodies are easier for vectorizers to optimize.
4. **Register Pressure Reduction**: Smaller loop bodies require fewer live variables.

## Algorithm

### High-Level Steps

1. **Loop Traversal**: Process loops from innermost to outermost (post-order walk).
2. **Dependency Analysis**: Build a dependency matrix for direct children operations.
3. **SCC Computation**: Find Strongly Connected Components to identify independent groups.
4. **Cost Model Evaluation**: Determine if distribution is beneficial.
5. **Loop Distribution**: Create separate loops for each SCC and clone operations.

### Detailed Algorithm

#### 1. Loop Traversal (Post-Order)

The pass processes loops from innermost to outermost because:
- Inner loop distribution creates opportunities for outer loop distribution
- Avoids complex multi-level dependency analysis
- Maintains transformation composability

```cpp
func.walk<WalkOrder::PostOrder>([&](affine::AffineForOp forOp) {
  loops.push_back(forOp);
});
```

#### 2. Dependency Analysis

For each loop, we analyze dependencies between **direct children** operations:

**Direct Children**: Operations immediately nested in the loop body (not recursively nested).

Example:
```mlir
affine.for %i = ... {
  affine.if (...) { ... }   // Direct child 1 (treated as atomic)
  affine.for %j = ... { ... } // Direct child 2 (treated as atomic)
  %x = ...                  // Direct child 3
}
```

**Dependency Types Checked**:

a) **SSA Def-Use Dependencies**:
   - If operation A produces a value used by operation B
   - Checked both forward and backward (A→B and B→A)

b) **Memory Dependencies** (Affine Dependence Analysis):
   - Loop-carried dependencies: Dependencies across different iterations
   - Loop-independent dependencies: Dependencies within the same iteration
   - Only considers the current loop's induction variable
   - Uses `checkMemrefAccessDependence` for precise analysis

c) **Memory Aliasing** (Alias Analysis):
   - Conservative check for operations accessing different memrefs
   - If two memrefs may alias and at least one is a write, there's a dependency
   - Skips same-memref checks (handled by affine dependence analysis)

**Key Implementation Detail**: Nested structures (loops, conditionals) are treated as atomic operations at the current level, but their leaf operations are analyzed for memory dependencies.

#### 3. Dependency Matrix Construction

Build an N×N boolean matrix where N is the number of direct children:
- `depMatrix[i][j] = true` means operation i depends on operation j
- This is a **directed graph** representation

```
Example Dependency Matrix (3 operations):
     0 1 2
  0: 0 0 0   (op 0 has no dependencies)
  1: 0 0 0   (op 1 has no dependencies)
  2: 0 1 0   (op 2 depends on op 1)
```

#### 4. Strongly Connected Components (SCCs)

**Why SCCs?**
- Operations in the same SCC have cyclic dependencies (must stay together)
- Operations in different SCCs with no edges between them are independent
- SCCs preserve correctness while maximizing distribution opportunities

**Algorithm**: Kosaraju's Algorithm (Two-pass DFS)

**Step 1**: First DFS on original graph
- Visit all nodes in DFS order
- Record finish times (when a node completes)
- Use iterative DFS to avoid stack overflow

**Step 2**: Create transpose graph
- Reverse all edge directions
- If there was an edge i→j, create edge j→i

**Step 3**: Second DFS on transpose graph
- Process nodes in reverse finish time order
- Each DFS tree in this pass identifies one SCC
- Nodes reachable in one DFS traversal form an SCC

**Why This Works**:
- Kosaraju's algorithm correctly identifies strongly connected components
- Time complexity: O(V + E) where V is nodes and E is edges
- Space efficient with iterative implementation

**Example with Cyclic Dependency**:
```
Original Graph:          Transpose Graph:         SCCs:
    0 → 1                    0 ← 1                SCC1: {0, 1, 2}
    ↑   ↓                    ↑   ↓                SCC2: {3}
    └───2 → 3                └───2 ← 3

Edges: 0→1, 1→2, 2→0,    Reversed: 1→0, 2→1,    Nodes 0,1,2 form a cycle
       2→3                           0→2, 3→2    Node 3 is independent
```

**Step-by-step for this example**:
1. First DFS: Visit order might be 0→1→2→3, finish stack: [3, 2, 1, 0]
2. Transpose graph reverses all edges
3. Second DFS from finish stack (reverse order):
   - Start from 0: reaches 1, 2 → SCC {0, 1, 2}
   - Start from 3: only 3 itself → SCC {3}

#### 5. Cost Model Evaluation

The pass uses a cost model to decide whether distribution is beneficial:

**Parallelism Gain** (Higher is better):
```
parallelismScore = (number of independent SCCs) / (total SCCs)
```
- Independent SCCs have no dependencies between them
- Can potentially be parallelized

**Locality Cost** (Lower is better):
```
localityCost = (number of shared memrefs between SCCs) / (total SCC pairs)
```
- Shared memrefs indicate potential cache misses after distribution
- Distribution may hurt temporal locality

**Decision Function**:
```
benefit = (parallelismWeight × parallelismGain) - (localityWeight × localityCost)
shouldDistribute = (benefit >= 0.0)
```

**Default Weights**:
- `parallelismWeight = 1.0` (favor exposing parallelism)
- `localityWeight = 0.5` (penalize locality loss, but less)

**Tunable Parameters**: Weights can be adjusted based on target architecture (e.g., more cores → higher parallelism weight).

#### 6. Loop Distribution

If distribution is beneficial, create new loops:

**For each SCC**:
1. Create a new `affine.for` loop with same bounds and step
2. Clone operations from the SCC into the new loop body
3. Use `IRMapping` to remap induction variable and SSA values

**External Use Handling**:
- If an operation produces values used outside the original loop
- Replace external uses with values from the cloned operation

**Original Loop Erasure**:
- After all distributed loops are created, erase the original loop

```cpp
for (const auto &scc : sccComponents) {
  auto newLoop = builder.create<AffineForOp>(...);
  // Clone operations from this SCC
  for (unsigned opIdx : scc) {
    Operation *cloned = bodyBuilder.clone(*directChildren[opIdx], mapper);
  }
}
forOp.erase(); // Remove original loop
```

## Implementation Files

### Core Files

1. **`LoopDistribution.cpp`** (`mlir/lib/Dialect/Affine/Transforms/`)
   - Main pass implementation
   - Loop traversal and orchestration
   - Dependency checking and distribution logic

2. **`LoopDistributionUtils.cpp`** (`mlir/lib/Dialect/Affine/Utils/`)
   - Utility functions for distribution
   - SCC computation (Kosaraju's algorithm)
   - Cost model evaluation
   - Helper functions for dependency analysis

3. **`LoopDistributionUtils.h`** (`mlir/include/mlir/Dialect/Affine/`)
   - Public API declarations
   - Function signatures and documentation

### Key Classes and Functions

#### `AffineLoopDistribution` Pass

**Main Methods**:
- `runOnOperation()`: Entry point, walks all loops in post-order
- `performLoopDistribution()`: Main logic for distributing a single loop
- `checkDependenceAtLevel()`: Checks if two operations have dependencies
- `buildDependenceMatrixAtCurrentLevel()`: Constructs dependency matrix
- `distributeBySCCs()`: Creates distributed loops from SCCs

#### Utility Functions

**Dependency Analysis**:
- `hasDependenceBetweenOpsForLoopIV()`: Checks memory dependencies for a specific loop IV
- `collectLeafOperations()`: Recursively collects leaf operations from nested structures
- `getLoopNestingDepth()`: Computes loop nesting depth

**SCC Computation**:
- `findStronglyConnectedComponents()`: Kosaraju's algorithm implementation

**Cost Model**:
- `estimateParallelismGain()`: Computes parallelism benefit score
- `estimateLocalityCost()`: Computes locality cost score
- `isDistributionBeneficial()`: Final distribution decision

## Usage

### Running the Pass

```bash
mlir-opt -affine-loop-distribution input.mlir -o output.mlir
```

### With Debug Output

```bash
mlir-opt -affine-loop-distribution -debug-only=affine-loop-distribution input.mlir
```

Debug output includes:
- Dependency matrix
- SCCs found
- Parallelism and locality scores
- Distribution decisions

### Programmatic Usage

```cpp
#include "mlir/Dialect/Affine/Passes.h"

// Create the pass
auto pass = mlir::affine::createLoopDistributionPass();

// Add to pass manager
pm.addPass(std::move(pass));
```

## Examples

### Example 1: Simple Independence

**Input**:
```mlir
func.func @example1(%A: memref<100xf32>, %C: memref<100xf32>) {
  %B = memref.alloc() : memref<100xf32>
  %D = memref.alloc() : memref<100xf32>
  affine.for %i = 0 to 100 {
    %u = affine.load %A[%i]
    affine.store %u, %B[%i]
    %v = affine.load %C[%i]
    affine.store %v, %D[%i]
  }
  return
}
```

**Output** (2 independent loops):
```mlir
func.func @example1(%A: memref<100xf32>, %C: memref<100xf32>) {
  %B = memref.alloc() : memref<100xf32>
  %D = memref.alloc() : memref<100xf32>
  affine.for %i = 0 to 100 {
    %u = affine.load %A[%i]
    affine.store %u, %B[%i]
  }
  affine.for %i = 0 to 100 {
    %v = affine.load %C[%i]
    affine.store %v, %D[%i]
  }
  return
}
```

**Analysis**:
- Operations on A/B and C/D have no dependencies
- Creates 2 SCCs: {ops 0-1} and {ops 2-3}
- Both loops can be executed in parallel or vectorized independently

### Example 2: Loop-Carried Dependency

**Input**:
```mlir
func.func @example2(%A: memref<1024xf32>, %B: memref<1024xf32>) {
  %C = memref.alloc() : memref<1024xf32>
  affine.for %i = 1 to 1024 {
    %v = affine.load %A[%i]
    %vd = arith.addf %v, %v
    affine.store %vd, %B[%i]
    %x = affine.load %B[%i - 1]  // Reads previous iteration's write
    affine.store %x, %C[%i]
  }
  return
}
```

**Output** (2 loops with preserved dependency):
```mlir
func.func @example2(%A: memref<1024xf32>, %B: memref<1024xf32>) {
  %C = memref.alloc() : memref<1024xf32>
  affine.for %i = 1 to 1024 {
    %v = affine.load %A[%i]
    %vd = arith.addf %v, %v
    affine.store %vd, %B[%i]
  }
  affine.for %i = 1 to 1024 {
    %x = affine.load %B[%i - 1]
    affine.store %x, %C[%i]
  }
  return
}
```

**Analysis**:
- Second part reads `B[%i-1]` which depends on previous iteration of first part
- However, operations are in different SCCs because dependency is properly ordered
- First loop must complete before second loop starts
- First loop can be vectorized/parallelized independently

### Example 3: No Distribution (Aliasing)

**Input**:
```mlir
func.func @example3(%A: memref<1024xf32>, %B: memref<1024xf32>) {
  // C aliases with A through reinterpret_cast
  %C = memref.reinterpret_cast %A to offset: [0], sizes: [32, 32], 
                                strides: [32, 1] : memref<1024xf32> to memref<32x32xf32>
  affine.for %i = 1 to 1024 {
    %v = affine.load %A[%i]
    %vd = arith.addf %v, %v
    affine.store %vd, %B[%i]
    %x = affine.load %B[%i - 1]
    affine.store %x, %C[%i mod 32, %i floordiv 32]
  }
  return
}
```

**Output** (No change - loop not distributed):
```mlir
// Same as input - aliasing detected between A and C
```

**Analysis**:
- Alias analysis detects that C may alias with A
- Store to C may affect load from A
- Operations must remain together
- Distribution would be incorrect

### Example 4: Nested Loop Distribution

**Input**:
```mlir
func.func @example4(%A: memref<100xf32>, %C: memref<100xf32>) {
  affine.for %t = 0 to 10 {
    %B = memref.alloc() : memref<100xf32>
    %D = memref.alloc() : memref<100xf32>
    affine.for %i = 0 to 100 {
      %u = affine.load %A[%i]
      affine.store %u, %B[%i]
      %v = affine.load %C[%i]
      affine.store %v, %D[%i]
    }
  }
  return
}
```

**Output** (Inner loop distributed):
```mlir
func.func @example4(%A: memref<100xf32>, %C: memref<100xf32>) {
  affine.for %t = 0 to 10 {
    %B = memref.alloc() : memref<100xf32>
    %D = memref.alloc() : memref<100xf32>
    affine.for %i = 0 to 100 {
      %u = affine.load %A[%i]
      affine.store %u, %B[%i]
    }
    affine.for %i = 0 to 100 {
      %v = affine.load %C[%i]
      affine.store %v, %D[%i]
    }
  }
  return
}
```

**Analysis**:
- Post-order traversal processes inner loop first
- Inner loop is distributed into 2 independent loops
- Outer loop remains unchanged
- Each inner distributed loop can be optimized independently

### Example 5: Three-Way Distribution

**Input**:
```mlir
func.func @example5(%A: memref<100xf32>, %C: memref<100xf32>, %E: memref<100xf32>) {
  %B = memref.alloc() : memref<100xf32>
  %D = memref.alloc() : memref<100xf32>
  %F = memref.alloc() : memref<100xf32>
  affine.for %i = 0 to 100 {
    %u = affine.load %A[%i]
    affine.store %u, %B[%i]
    %v = affine.load %C[%i]
    affine.store %v, %D[%i]
    %w = affine.load %E[%i]
    affine.store %w, %F[%i]
  }
  return
}
```

**Output** (3 independent loops):
```mlir
func.func @example5(%A: memref<100xf32>, %C: memref<100xf32>, %E: memref<100xf32>) {
  %B = memref.alloc() : memref<100xf32>
  %D = memref.alloc() : memref<100xf32>
  %F = memref.alloc() : memref<100xf32>
  affine.for %i = 0 to 100 {
    %u = affine.load %A[%i]
    affine.store %u, %B[%i]
  }
  affine.for %i = 0 to 100 {
    %v = affine.load %C[%i]
    affine.store %v, %D[%i]
  }
  affine.for %i = 0 to 100 {
    %w = affine.load %E[%i]
    affine.store %w, %F[%i]
  }
  return
}
```

**Analysis**:
- Three completely independent operation groups
- Creates 3 SCCs, each with 2 operations
- All three loops can be executed in parallel
- Maximum parallelism exposure

## Technical Details

### Atomic Operation Treatment

At each loop level, the pass treats nested structures atomically:

- **Nested Loops**: Entire inner loop is one operation
- **Conditionals (`affine.if`)**: Entire conditional with its body is one operation
- **Other Regions**: Any operation with regions is treated as single unit

**Rationale**: This simplifies dependency analysis and prevents incorrect distribution that would violate nested loop semantics.

**Leaf Operation Analysis**: While nested structures are atomic at the current level, their leaf operations are analyzed for memory dependencies to ensure correctness.

### Memory Dependency Checking

Two-level check for memory operations:

1. **Affine Dependence Analysis**:
   - Uses `checkMemrefAccessDependence()` from MLIR
   - Analyzes affine expressions in memory access indices
   - Determines if accesses to the same memref conflict
   - Considers loop-carried dependencies at specific depth

2. **Alias Analysis**:
   - Uses MLIR's `AliasAnalysis` framework
   - Checks if different memrefs may alias
   - Conservative: assumes alias means potential dependency
   - Only applied if at least one operation is a store

### SSA Value Mapping

During cloning operations into distributed loops:

- `IRMapping` tracks old→new value mappings
- Induction variable is mapped: old loop IV → new loop IV
- SSA values produced within loop are automatically remapped
- External uses (values used outside loop) are handled specially

### Edge Cases Handled

1. **Empty Loops**: Skipped (nothing to distribute)
2. **Single Operation**: No distribution possible
3. **Single SCC**: All operations have cyclic dependencies
4. **Alias Detection**: Conservatively prevents incorrect distribution
5. **Nested Region Operations**: Treated atomically at current level

## Performance Considerations

### When Distribution Helps

1. **Independent Computations**: Operations with no data dependencies
2. **Different Memory Regions**: Accesses to separate arrays
3. **Vectorization**: Simpler loop bodies are easier to vectorize
4. **Cache Optimization**: Grouping similar memory accesses

### When Distribution May Hurt

1. **Shared Data**: Multiple loops accessing same data may increase cache misses
2. **Loop Overhead**: Creating more loops adds control overhead
3. **Register Spilling**: May increase live ranges if loops are later merged
4. **Instruction Cache**: More code may reduce I-cache efficiency

### Cost Model Tuning

The cost model can be tuned for specific architectures:

```cpp
// In isDistributionBeneficial():
const double parallelismWeight = 1.5;  // Higher for many-core systems
const double localityWeight = 0.3;     // Lower if bandwidth is plentiful
```

**Recommended Tuning**:
- **Many-Core Systems**: Increase `parallelismWeight` (favor distribution)
- **Memory-Bound Systems**: Increase `localityWeight` (favor locality)
- **SIMD-Heavy Systems**: Increase `parallelismWeight` (favor vectorization)

## Testing

### Test File Location

`mlir/test/Dialect/Affine/loop-distribution.mlir`

### Running Tests

```bash
# Run with lit (LLVM Integrated Tester)
cd build
ninja check-mlir-dialect-affine

# Run specific test
./bin/mlir-opt -affine-loop-distribution \
  ../mlir/test/Dialect/Affine/loop-distribution.mlir | \
  FileCheck ../mlir/test/Dialect/Affine/loop-distribution.mlir
```

### Test Coverage

The test suite covers:
- ✅ Basic independence (2-way distribution)
- ✅ Loop-carried dependencies
- ✅ Alias detection (preventing incorrect distribution)
- ✅ Nested loop distribution
- ✅ Three-way distribution
- ✅ Nested control flow (if statements with loops)

## Debugging

### Debug Output

Enable detailed debug output:

```bash
mlir-opt -affine-loop-distribution -debug-only=affine-loop-distribution input.mlir
```

**Debug Information Includes**:
1. Loop being analyzed
2. Dependency matrix (NxN boolean matrix)
3. SCCs found (list of operations in each component)
4. Parallelism score and locality cost
5. Distribution decision (DISTRIBUTE or SKIP)
6. Number of distributed loops created

### Example Debug Output

```
Analyzing loop for distribution: ...
Dependency Matrix (4x4):
     0 1 2 3
  0: 0 0 0 0
  1: 0 0 0 0
  2: 0 0 0 0
  3: 0 0 0 0

Found 4 SCC(s):
  SCC 0: [0]
  SCC 1: [1]
  SCC 2: [2]
  SCC 3: [3]

Evaluating cost model for distribution...
  Parallelism score: 1.0 (4 independent SCCs out of 4)
  Locality cost: 0.0 (0 shared memrefs across SCCs)
  Net benefit: 1.0
  Decision: DISTRIBUTE

Creating 4 distributed loops...
Successfully distributed loop into 4 independent loops
```

## Limitations and Future Work

### Current Limitations

1. **Affine Loops Only**: Only works with `affine.for` loops
2. **Direct Children**: Only distributes operations at current level
3. **Conservative Aliasing**: May miss optimization opportunities with complex aliasing
4. **No Inter-Procedural Analysis**: Function calls are treated conservatively
5. **Fixed Cost Model**: Weights are compile-time constants

### Future Enhancements

1. **Dynamic Cost Model**: Profile-guided optimization weights
2. **Multi-Level Distribution**: Simultaneously distribute multiple loop levels
3. **Partial Distribution**: Distribute subsets when full distribution isn't beneficial
4. **Integration with Loop Fusion**: Combined fission-fusion optimization
5. **Target-Specific Tuning**: Architecture-aware cost models
6. **Polyhedral Model Integration**: Use more sophisticated dependency analysis

## References

### Academic Papers

1. **Loop Distribution**:
   - Kennedy, K., & McKinley, K. S. (1993). "Maximizing loop parallelism and improving data locality via loop fusion and distribution."

2. **Strongly Connected Components**:
   - Kosaraju, S. R. (1978). "Strong connectivity algorithm."

3. **Affine Dependence Analysis**:
   - Feautrier, P. (1988). "Parametric integer programming."
   - Pugh, W., & Wonnacott, D. (1992). "Eliminating false data dependences using the Omega test."

### MLIR Documentation

- [Affine Dialect Documentation](https://mlir.llvm.org/docs/Dialects/Affine/)
- [Affine Analysis and Utilities](https://mlir.llvm.org/docs/Dialects/Affine/#analysis-and-utilities)
- [Writing a Pass](https://mlir.llvm.org/docs/PassManagement/)

## Contributing

### Adding New Features

To extend the pass:

1. **Add utility functions** to `LoopDistributionUtils.cpp/h`
2. **Update cost model** in `isDistributionBeneficial()`
3. **Add tests** in `loop-distribution.mlir`
4. **Update this README** with new functionality

### Code Style

Follow LLVM coding standards:
- 80-character line limit
- 2-space indentation
- CamelCase for classes, camelCase for functions
- Comprehensive comments for complex logic

## Authors and Maintenance

This pass was developed as part of the MLIR Affine dialect transformation infrastructure.

**Maintainers**: See `mlir/lib/Dialect/Affine/Maintainers.md`

**Contact**: LLVM/MLIR mailing lists and GitHub issues

## License

Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
