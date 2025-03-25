# MLIR (Multi-Level Intermediate Representation)

## Introduction

MLIR (Multi-Level Intermediate Representation) is a compiler infrastructure designed by Chris Lattner while at Google to unify the multiple compiler IRs used in machine learning and other domains. It provides a flexible and extensible framework that enables representation, transformation, and optimization at different levels of abstraction. While normally used to optimize code and target its compilation to unique chips, at Atomic Machines we use the MLIR to move from all levels of design intent to all levels of the process graph to finally generate a complete set of code needed to create devices on the Matter Compiler.

General features of MLIR include:

- **Multi-level abstraction**: MLIR can represent code at different levels from high-level domain-specific representations down to low-level machine code
- **Dialect ecosystem**: Functionality is organized into dialects, which are namespaces containing operations, types, and attributes for specific domains
- **Extensibility**: Users can create custom dialects to represent domain-specific concepts
- **Progressive lowering/raising**: Code can be progressively transformed from high-level to lower-level dialects, and from low-level to high-level dialects. Both are useful in the MC device design process.
- **Strong type system**: MLIR has a rich type system that captures semantic information

This README provides examples of MLIR syntax to help you understand how programs are represented in this IR. It uses examples mostly from the compiler world... But note that this syntax is not designed to be edited directly. 

We use this libary, `pymlir` to generate, manipulate, and parse MLIR code. This is library distinct from the official MLIR python bindings which are not as well suited to direct and extensive MLIR syntax manipulation.

This readme exists to show you how MLIR looks. But see the last example at the bottom of the page for an example of how to manipluate MLIR using this `pymlir` package.


## Basic MLIR Structure

MLIR programs consist of operations, types, attributes, and dialects organized in a hierarchical structure.

### Operations

Operations are the basic computational units in MLIR. They have a name, operands, results, attributes, and regions.

```mlir
// Basic operation syntax
%result = "dialect.operation"(%operand1, %operand2) : (type1, type2) -> result_type

// Example of addition operation
%sum = arith.addi %a, %b : i32

// Operation with attributes
%conv = "mhlo.convolution"(%input, %filter) {
  dilations = [1, 1],
  strides = [1, 1],
  padding = "SAME"
} : (tensor<1x28x28x1xf32>, tensor<3x3x1x16xf32>) -> tensor<1x28x28x16xf32>
```

### Types

MLIR has built-in primitive types and allows dialects to define their own types.

```mlir
// Integer types
%a = arith.constant 42 : i32  // 32-bit integer
%b = arith.constant 7 : i8    // 8-bit integer

// Floating point types
%c = arith.constant 3.14 : f32  // 32-bit float
%d = arith.constant 2.71 : f64  // 64-bit float

// Tensor type
%t1 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>

// Memref type (memory reference with layout information)
%buf = memref.alloc() : memref<10x10xf32>

// Vector type
%vec = vector.broadcast %c : f32 to vector<4xf32>

// Function type
func.func @add(%x: i32, %y: i32) -> i32 { ... }
```

### Attributes

Attributes are compile-time known values attached to operations:

```mlir
// String attribute
"module"() {sym_name = "my_module"} : () -> ()

// Dictionary attribute
%2 = "dialect.op"() {
  string_attr = "value",
  int_attr = 42,
  array_attr = [1, 2, 3],
  dict_attr = {a = "b", c = 10}
} : () -> i32

// Type attribute
"func.func"() {type = (i32, i32) -> i32} : () -> ()
```

## Dialect Examples

Dialects are namespaces that group related operations and types. Here are examples from common dialects:

### Standard Dialect

```mlir
// Control flow
func.func @conditional(%cond: i1, %a: f32, %b: f32) -> f32 {
  %result = arith.select %cond, %a, %b : f32
  return %result : f32
}

// Arithmetic operations
%1 = arith.addi %a, %b : i32
%2 = arith.muli %1, %c : i32
%3 = arith.divf %d, %e : f32
%4 = arith.remf %d, %e : f32
```

### LLVM Dialect

```mlir
// LLVM pointer type
%ptr = llvm.alloca %size x !llvm.i32 : (i32) -> !llvm.ptr<i32>

// Memory operations
%val = llvm.load %ptr : !llvm.ptr<i32>
llvm.store %val, %ptr : !llvm.ptr<i32>

// Function call
%result = llvm.call @function(%arg1, %arg2) : (i32, i32) -> i32
```

### Tensor Dialect

```mlir
// Tensor operations
%sliced = tensor.extract_slice %tensor[0, 0] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
%expanded = tensor.pad %source low[1, 1] high[1, 1] {
^bb0(%arg0: index, %arg1: index):
  tensor.yield %pad_value : f32
} : tensor<2x3xf32> to tensor<4x5xf32>
```

### Linalg Dialect (Linear Algebra)

```mlir
// Matrix multiplication
linalg.matmul 
  ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>)
  outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>

// Generic operation
linalg.generic {
  indexing_maps = [
    affine_map<(i, j) -> (i, j)>,
    affine_map<(i, j) -> (i, j)> 
  ],
  iterator_types = ["parallel", "parallel"]
}
ins(%input : tensor<10x20xf32>)
outs(%output : tensor<10x20xf32>) {
  ^bb0(%in: f32, %out: f32):
    %result = arith.addf %in, %in : f32
    linalg.yield %result : f32
} -> tensor<10x20xf32>
```

### Vector Dialect

```mlir
// Vector operations
%broadcast = vector.broadcast %scalar : f32 to vector<4xf32>
%add = vector.add %v1, %v2 : vector<4xf32>
%dot = vector.dot %v1, %v2 : vector<4xf32>, vector<4xf32> -> f32
```

## Structured Control Flow

```mlir
// If-else construct
scf.if %condition {
  // True branch
  scf.yield %true_value : f32
} else {
  // False branch
  scf.yield %false_value : f32
}

// For loop
%sum = scf.for %i = %lb to %ub step %step 
  iter_args(%iter = %init) -> (f32) {
    %new_iter = arith.addf %iter, %value : f32
    scf.yield %new_iter : f32
}

// While loop
%result = scf.while (%arg0 = %init) : (f32) -> f32 {
  %condition = arith.cmpf "ult", %arg0, %limit : f32
  scf.condition(%condition) %arg0 : f32
} do {
^bb0(%arg0: f32):
  %next = arith.addf %arg0, %step : f32
  scf.yield %next : f32
}
```

## Regions and Blocks

```mlir
// A region with multiple blocks
"test.region"() ({
  ^bb0(%arg0: i32):
    %cond = arith.cmpi "eq", %arg0, %c0 : i32
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %1 = arith.addi %arg0, %c1 : i32
    cf.br ^bb3(%1: i32)
  ^bb2:
    %2 = arith.subi %arg0, %c1 : i32
    cf.br ^bb3(%2: i32)
  ^bb3(%3: i32):
    "test.return"(%3) : (i32) -> ()
}) : () -> i32
```

## Module Structure

```mlir
module {
  func.func @main() -> i32 {
    %0 = arith.constant 0 : i32
    return %0 : i32
  }
  
  // Global values
  memref.global "private" @global_var : memref<10xi32> = dense<0>
  
  // Nested modules are possible
  module @submodule {
    func.func @helper() -> i32 {
      %c1 = arith.constant 1 : i32
      return %c1 : i32
    }
  }
}
```

## Creating Custom Dialects

MLIR allows you to define custom dialects for your domain-specific needs:

```python
from mlir.dialect import Dialect, DialectOp, DialectType

# Define operations
class Add(DialectOp):
    _syntax_ = "add {operand_a.ssa_use}, {operand_b.ssa_use} : {type.type}"

class Custom1DType(DialectType):
    _syntax_ = "custom.1d<{dim.integer}>"

# Create the dialect
custom_dialect = Dialect(
    name="custom",
    ops=[Add],
    types=[Custom1DType],
    preamble="// Custom dialect preamble in Lark syntax"
)
```

This would enable MLIR code like:

```mlir
%result = custom.add %a, %b : !custom.1d<5>
```

## Additional Resources

- [MLIR Official Documentation](https://mlir.llvm.org/docs/)
- [MLIR GitHub Repository](https://github.com/llvm/llvm-project/tree/main/mlir)
- [MLIR Specification](https://mlir.llvm.org/docs/LangRef/)
