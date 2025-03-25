"""Tests pyMLIR on examples that use the Toy dialect."""

import pytest
import os
from mlir import parse_string, parse_path
from mlir.dialects.func import func


def test_toy_simple():
    code = """
module {
  func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
    %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %t_tensor : tensor<3x2xf64>
  }
}
    """

    module = parse_string(code)
    assert module is not None


def test_toy_advanced():
    module = parse_path(os.path.join(os.path.dirname(__file__), "toy.mlir"))
    assert module is not None


if __name__ == "__main__":
    test_toy_simple()
    test_toy_advanced()
