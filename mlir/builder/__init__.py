from .builder import IRBuilder, DialectBuilder, AffineBuilder
from .match import Reads, Writes, Isa, All, And, Or, Not


__doc__ = """
.. currentmodule:: mlir.builder

.. automodule:: mlir.builder.builder

.. automodule:: mlir.builder.match
"""


__all__ = [
    "IRBuilder",
    "DialectBuilder",
    "AffineBuilder",
    "Reads",
    "Writes",
    "Isa",
    "All",
    "And",
    "Or",
    "Not",
]
