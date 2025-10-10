// RUN: mlir-opt -affine-loop-distribution %s | FileCheck %s
// Do not update anything in this test case.
// Last updated: Sep 18, 2025.

// CHECK-LABEL: func @distribute_for_locality
func.func @distribute_for_locality(%A: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %c0 = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %A, %c0 : memref<?x?xf32>
  %N = memref.dim %A, %c1 : memref<?x?xf32>
  %B = memref.alloc(%M, %N) : memref<?x?xf32>
  %D = memref.alloc(%M, %N) : memref<?x?xf32>
  affine.for %i = 0 to %M {
    affine.for %j = 0 to %N {
      %v = affine.load %A[%i, %j] : memref<?x?xf32>
      %vd = arith.addf %v, %v : f32
      affine.store %vd, %B[%i, %j] : memref<?x?xf32>
      %w = affine.load %C[%i, %j] : memref<?x?xf32>
      %wd = arith.mulf %w, %w : f32
      affine.store %wd, %D[%i, %j] : memref<?x?xf32>
    }
  }
  // CHECK:      affine.for %{{.*}} = 0 to %{{.*}}
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to %{{.*}}
  // CHECK-NEXT:     affine.load
  // CHECK-NEXT:     arith.addf
  // CHECK-NEXT:     affine.store
  // CHECK:      affine.for %{{.*}} = 0 to %{{.*}}
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to %{{.*}}
  // CHECK-NEXT:     affine.load
  // CHECK-NEXT:     arith.mulf
  // CHECK-NEXT:     affine.store
  return
}

// CHECK-LABEL: func @distribute_for_parallelism
func.func @distribute_for_parallelism(%A: memref<1024xf32>, %B: memref<1024xf32>) {
  %C = memref.alloc() : memref<1024xf32>
  affine.for %i = 1 to 1024 {
    %v = affine.load %A[%i] : memref<1024xf32>
    %vd = arith.addf %v, %v : f32
    affine.store %vd, %B[%i] : memref<1024xf32>
    %x = affine.load %B[%i - 1] : memref<1024xf32>
    affine.store %x, %C[%i] : memref<1024xf32>
  }
  // CHECK:      affine.for %{{.*}} = 1 to 1024
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   arith.addf
  // CHECK-NEXT:   affine.store
  // CHECK:      affine.for %{{.*}} = 1 to 1024
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   affine.store
  return
}

// CHECK-LABEL: func @do_not_distribute_alias
func.func @do_not_distribute_alias(%A: memref<1024xf32>, %B: memref<1024xf32>) {
  // %C aliases with %A.
  %C = memref.reinterpret_cast %A to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<1024xf32> to memref<32x32xf32>
  affine.for %i = 1 to 1024 {
    %v = affine.load %A[%i] : memref<1024xf32>
    %vd = arith.addf %v, %v : f32
    affine.store %vd, %B[%i] : memref<1024xf32>
    %x = affine.load %B[%i - 1] : memref<1024xf32>
    affine.store %x, %C[%i mod 32, %i floordiv 32] : memref<32x32xf32>
  }
  // CHECK:      affine.for %{{.*}} = 1 to 1024
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   arith.addf
  // CHECK-NEXT:   affine.store
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   affine.store
  return
}

// Test with an affine.if nested within.
// Test with imperfect nests.

// Test inner distribution. Outer loop remains intact, distribution in the inner
// body.

// CHECK-LABEL: func @inner_distribute
func.func @inner_distribute(%A: memref<100xf32>, %C: memref<100xf32>) {
  // CHECK: affine.for
  affine.for %t = 0 to 10 {
    %B = memref.alloc() : memref<100xf32>
    %D = memref.alloc() : memref<100xf32>
    affine.for %i = 0 to 100 {
      %u = affine.load %A[%i] : memref<100xf32>
      affine.store %u, %B[%i] : memref<100xf32>
      %v = affine.load %C[%i] : memref<100xf32>
      affine.store %v, %D[%i] : memref<100xf32>
    }
    // Distributed loops.
    // CHECK: affine.for %{{.*}} = 0 to 100
    // CHECK-NEXT: affine.load
    // CHECK-NEXT: affine.store
    // CHECK: affine.for %{{.*}} = 0 to 100
    // CHECK-NEXT: affine.load
    // CHECK-NEXT: affine.store
  }
  // CHECK: return
  return
}

// CHECK-LABEL: func @three_way_distribute
func.func @three_way_distribute(%A: memref<100xf32>, %C: memref<100xf32>, %E: memref<100xf32>) {
  %B = memref.alloc() : memref<100xf32>
  %D = memref.alloc() : memref<100xf32>
  %F = memref.alloc() : memref<100xf32>
  affine.for %i = 0 to 100 {
    %u = affine.load %A[%i] : memref<100xf32>
    affine.store %u, %B[%i] : memref<100xf32>
    %v = affine.load %C[%i] : memref<100xf32>
    affine.store %v, %D[%i] : memref<100xf32>
    %w = affine.load %E[%i] : memref<100xf32>
    affine.store %w, %F[%i] : memref<100xf32>
  }
  // CHECK: affine.for %{{.*}} = 0 to 100
  // CHECK-NEXT: affine.load
  // CHECK-NEXT: affine.store
  // CHECK: affine.for %{{.*}} = 0 to 100
  // CHECK-NEXT: affine.load
  // CHECK-NEXT: affine.store
  // CHECK: affine.for %{{.*}} = 0 to 100
  // CHECK-NEXT: affine.load
  // CHECK-NEXT: affine.store
  return
}

// CHECK-LABEL: func @nested_control_flow_distribute
func.func @nested_control_flow_distribute(%A: memref<100xf32>, %B: memref<100xf32>, %C: memref<100xf32>) {
  %D = memref.alloc() : memref<100xf32>
  %E = memref.alloc() : memref<100xf32>
  %F = memref.alloc() : memref<100xf32>
  
  affine.for %t = 0 to 10 {
    // First sequential loop - should be distributed
    affine.for %i = 0 to 100 {
      %u = affine.load %A[%i] : memref<100xf32>
      affine.store %u, %D[%i] : memref<100xf32>
    }
    
    // Second sequential loop - should be distributed
    affine.for %j = 0 to 100 {
      %v = affine.load %B[%j] : memref<100xf32>
      affine.store %v, %E[%j] : memref<100xf32>
    }
    
    // Conditional with nested loop - should remain grouped
    affine.if affine_set<(d0) : (d0 mod 2 == 0)>(%t) {
      affine.for %k = 0 to 100 {
        %w = affine.load %C[%k] : memref<100xf32>
        affine.store %w, %F[%k] : memref<100xf32>
      }
    }
  }
  
  // The outer loop is distributed into 3 separate loops (in reverse order)
  // First distributed loop contains the affine.if with nested loop
  // CHECK: affine.for %{{.*}} = 0 to 10
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 100
  // CHECK-NEXT: affine.load
  // CHECK-NEXT: affine.store
  
  // Second distributed loop contains the second sequential loop
  // CHECK: affine.for %{{.*}} = 0 to 10
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 100
  // CHECK-NEXT: affine.load
  // CHECK-NEXT: affine.store
  
  // Third distributed loop contains the first sequential loop
  // CHECK: affine.for %{{.*}} = 0 to 10
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 100
  // CHECK-NEXT: affine.load
  // CHECK-NEXT: affine.store
  return
}