# Project Requirements

This document outlines the dependencies required to run the Quantum vs. Classical European Option Pricing toolkit.

## Analytical Engine
* `numpy`
* `scipy`
* `pandas`

## Plotting
Used in `complexity_comparison.py` and `real_world_validation.py`.
* `matplotlib`

## Live Market Data
Used in `real_world_validation.py`. 
* `yfinance` *(Note: requires ordinary internet access to Yahoo Finance; will not work in a network-sandboxed environment.)*

## Quantum Engine
Used in `qae_pricer.py`.

**Version Pinning Notice:** `qiskit-finance==0.4.1` (its last release) is incompatible with `qiskit>=1.0`. Aer cannot transpile a custom instruction its circuits emit (`AerError: unknown instruction: P(X)`). The following combination is verified working end-to-end and must be used exactly as specified:
* `qiskit==0.45.3`
* `qiskit-aer==0.13.3`
* `qiskit-algorithms==0.2.2`
* `qiskit-finance==0.4.1`


