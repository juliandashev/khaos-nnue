"""KhaosChess NNUE training pipeline.

The modules in this package split along a dependency line on purpose:

    quant, features, format, refeval   pure standard library
    model, dataset                     require torch (and numpy)

Everything that defines the contract with the C++ engine lives on the
dependency-free side, so it can be tested and cross-checked against the engine
without a training environment installed.
"""
