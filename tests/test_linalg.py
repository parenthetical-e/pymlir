import pytest
import mlir

# All source strings have been taken from MLIR's codebase.
# See llvm-project/mlir/test/Dialect/Linalg


def assert_roundtrip_equivalence(source):
    assert source == mlir.parse_string(source).dump()


def test_batch_matmul():
    assert_roundtrip_equivalence(
        """module {
  func.func @named_ops(%a3: memref<?x?x?xf32>, %b3: memref<?x?x?xf32>, %c3: memref<?x?x?xf32>, %ta3: tensor<?x?x?xf32>, %tb3: tensor<?x?x?xf32>, %tc3: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
    linalg.batch_matmul ins ( %a3 , %b3 : memref<?x?x?xf32> , memref<?x?x?xf32> ) outs ( %c3 : memref<?x?x?xf32> )
    linalg.batch_matmul ins ( %ta3 , %tb3 : tensor<?x?x?xf32> , tensor<?x?x?xf32> ) outs ( %c3 : memref<?x?x?xf32> )
    %res1 = linalg.batch_matmul ins ( %ta3 , %tb3 : tensor<?x?x?xf32> , tensor<?x?x?xf32> ) init ( %tc3 : tensor<?x?x?xf32> ) -> tensor<?x?x?xf32>
    %res2 = linalg.batch_matmul ins ( %ta3 , %b3 : tensor<?x?x?xf32> , memref<?x?x?xf32> ) init ( %tc3 : tensor<?x?x?xf32> ) -> tensor<?x?x?xf32>
    return %res1, %res2 : tensor<?x?x?xf32>, tensor<?x?x?xf32>
  }
}"""
    )


def test_conv():
    assert_roundtrip_equivalence(
        """module {
  func.func @conv1d_no_symbols(%in: memref<?xf32>, %filter: memref<?xf32>, %out: memref<?xf32>) {
    linalg.conv_1d ins ( %in , %filter : memref<?xf32> , memref<?xf32> ) outs ( %out : memref<?xf32> )
    return
  }
  func.func @conv2d_no_symbols(%in: memref<?x?xf32>, %filter: memref<?x?xf32>, %out: memref<?x?xf32>) {
    linalg.conv_2d ins ( %in , %filter : memref<?x?xf32> , memref<?x?xf32> ) outs ( %out : memref<?x?xf32> )
    return
  }
  func.func @conv3d_no_symbols(%in: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>, %out: memref<?x?x?xf32>) {
    linalg.conv_3d ins ( %in , %filter : memref<?x?x?xf32> , memref<?x?x?xf32> ) outs ( %out : memref<?x?x?xf32> )
    return
  }
}"""
    )


def test_copy():
    assert_roundtrip_equivalence(
        """module {
  func.func @copy_view(%arg0: memref<?xf32, strided<[1], offset: ?>>, %arg1: memref<?xf32, strided<[1], offset: ?>>) {
    linalg.copy( %arg0 , %arg1 )  : memref<?xf32, strided<[1], offset: ?>> , memref<?xf32, strided<[1], offset: ?>>
    return
  }
  func.func @copy_view3(%arg0: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>, %arg1: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>) {
    linalg.copy( %arg0 , %arg1 ) {inputPermutation = affine_map<(i, j, k) -> (i, k, j)>, outputPermutation = affine_map<(i, j, k) -> (k, j, i)>} : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>> , memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>
    return
  }
}"""
    )


def test_dot():
    assert_roundtrip_equivalence(
        """module {
  func.func @dot(%arg0: memref<?xi8>, %M: index) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %1 = view %arg0 [ %c0 ] [ %M ] : memref<?xi8> to memref<?xf32>
    %2 = view %arg0 [ %c0 ] [ %M ] : memref<?xi8> to memref<?xf32>
    %3 = view %arg0 [ %c0 ] [  ] : memref<?xi8> to memref<f32>
    linalg.dot ins ( %1 , %2 : memref<?xf32> , memref<?xf32> ) outs ( %3 : memref<f32> )
    return
  }
}"""
    )


def test_fill():
    assert_roundtrip_equivalence(
        """module {
  func.func @fill_view(%arg0: f32, %arg1: tensor<?x?xf32>) {
    linalg.fill ins ( %arg0 : f32 ) outs ( %arg1 : tensor<?x?xf32> ) -> tensor<?x?xf32>
    linalg.fill ins ( %arg0 : f32 ) outs ( %arg1 : tensor<?x?xf32> )
    return
  }
}"""
    )


def test_fill_rng_2d():
    assert_roundtrip_equivalence(
        """module {
  func.func @fill_rng_2d(%min: f64, %max: f64, %seed: i32, %O: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = linalg.fill_rng_2d ins ( %min , %max , %seed : f64 , f64 , i32 ) outs ( %O : tensor<16x32xf32> ) -> tensor<16x32xf32>
    %1 = linalg.fill_rng_2d ins ( %min , %max , %seed : f64 , f64 , i32 ) outs ( %O : tensor<16x32xf32> )
    return %1 : tensor<16x32xf32>
  }
}"""
    )


def test_generic():
    assert_roundtrip_equivalence(
        """module {
  func.func @example(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>) {
    linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"]}  ins ( %A, %B : memref<?x?xf64>, memref<?x?xf64> ) outs ( %C : memref<?x?xf64> ) {
      ^bb0 (%a: f64, %b: f64, %c: f64):
        %c0 = constant 3.14 : f64
        %d = addf %a , %b : f64
        linalg.yield %d : f64
    }
    return
  }
}"""
    )


def test_indexed_generic():
    assert_roundtrip_equivalence(
        """module {
  func.func @indexed_generic_region(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg1: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>, %arg2: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>) {
    linalg.indexed_generic {args_in = 1, args_out = 2, iterator_types = ["parallel", "parallel", "parallel"], indexing_maps = [affine_map<(i, j, k) -> (i, j)>, affine_map<(i, j, k) -> (i, j, k)>, affine_map<(i, j, k) -> (i, k, j)>], library_call = "some_external_function_name_2", doc = "B(i,j,k), C(i,k,j) = foo(A(i, j) * B(i,j,k), i * j * k + C(i,k,j))"}  ins ( %arg0 : memref<?x?xf32, strided<[?, 1], offset: ?>> ) outs ( %arg1, %arg2 : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>, memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>> ) {
      ^bb0 (%i: index, %j: index, %k: index, %a: f32, %b: f32, %c: f32):
        %result_1 = mulf %a , %b : f32
        %ij = addi %i , %j : index
        %ijk = addi %ij , %k : index
        %ijk_int = index_cast %ijk : index to i32
        %ijk_float = sitofp %ijk_int : (i32) -> f32
        %result_2 = addf %c , %ijk_float : f32
        linalg.yield %result_1, %result_2 : f32, f32
    }
    return
  }
}"""
    )


def test_reduce():
    assert_roundtrip_equivalence(
        """module {
  func.func @reduce(%arg0: tensor<16x32x64xf32>, %arg1: tensor<16x64xf32>) {
    %reduce = linalg.reduce ins ( %arg0 : tensor<16x32x64xf32> ) outs ( %arg1 : tensor<16x64xf32> ) dimensions = [ 1 ] ( %in: f32, %out: f32 ) {
      %0 = arith.addf %out, %in : f32
      linalg.yield %0 : f32
    }
    return
  }
}"""
    )


def test_view():
    assert_roundtrip_equivalence(
        """module {
  func.func @views(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
    %c0 = constant 0 : index
    %0 = muli %arg0 , %arg0 : index
    %1 = alloc (%0) : memref<?xi8>
    %2 = linalg.range %arg0 : %arg1 : %arg 2 : !linalg.range
    %3 = view %1 [ %c0 ] [ %arg0, %arg0 ] : memref<?xi8> to memref<?x?xf32>
    %4 = linalg.slice %3 [ %2, %2 ] : memref<?x?xf32> , !linalg.range, !linalg.range  , memref<?x?xf32>
    %5 = linalg.slice %3 [ %2, %arg2 ] : memref<?x?xf32> , !linalg.range, index  , memref<?xf32, strided<[1], offset: ?>>
    %6 = linalg.slice %3 [ %arg2, %2 ] : memref<?x?xf32> , index, !linalg.range  , memref<?xf32, strided<[1], offset: ?>>
    %7 = linalg.slice %3 [ %arg2, %arg3 ] : memref<?x?xf32> , index, index  , memref<f32>
    %8 = view %1 [ %c0 ] [ %arg0, %arg0 ] : memref<?xi8> to memref<?x?xvector<4x4xf32>>
    dealloc %1 : memref<?xi8>
    return
  }
}"""
    )


def test_matmul():
    assert_roundtrip_equivalence(
        """module {
  func.func @matmul(%arg0: memref<?xi8>, %M: index, %N: index, %K: index) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %A = view %arg0 [ %c0 ] [ %M, %K ] : memref<?xi8> to memref<?x?xf32>
    %B = view %arg0 [ %c0 ] [ %K, %N ] : memref<?xi8> to memref<?x?xf32>
    %C = view %arg0 [ %c0 ] [ %M, %N ] : memref<?xi8> to memref<?x?xf32>
    linalg.matmul ins ( %A , %B : memref<?x?xf32> , memref<?x?xf32> ) outs ( %C : memref<?x?xf32> )
    linalg.matmul ins ( %A , %B : memref<?x?xf32> , memref<?x?xf32> ) outs ( %C : memref<?x?xf32> ) -> memref<?x?xf32>
    return
  }
}"""
    )


def test_matvec():
    assert_roundtrip_equivalence(
        """module {
  func.func @matvec(%arg0: memref<?xi8>, %M: index, %N: index) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %2 = view %arg0 [ %c0 ] [ %M, %N ] : memref<?xi8> to memref<?x?xf32>
    %3 = view %arg0 [ %c0 ] [ %M ] : memref<?xi8> to memref<?xf32>
    %4 = view %arg0 [ %c0 ] [ %N ] : memref<?xi8> to memref<?xf32>
    linalg.matvec ins ( %2 , %3 : memref<?x?xf32> , memref<?xf32> ) outs ( %4 : memref<?xf32> )
    return
  }
}"""
    )


def test_transpose():
    assert_roundtrip_equivalence(
        """module {
  func.func @transpose(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>) {
    %transpose = linalg.transpose ins ( %arg0 : memref<?x?xf16> ) outs ( %arg1 : memref<?x?xf16> ) permutation = [ 1, 0 ]
    return
  }
}"""
    )
