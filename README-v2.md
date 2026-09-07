# Quantum vs. Classical European Option Pricing

A dual-engine European option pricing toolkit built to validate and benchmark quantum computing methods in financial engineering. 

This repository contains **two independent pricing engines** for a vanilla European option: an exact closed-form analytical engine and a quantum Iterative Amplitude Estimation (IAE) engine. It also provides comprehensive benchmarking scripts that compare them against each other and against live real-world market data.

---

## 1. Executive Summary

This suite bridges classical quantitative finance with quantum computing. It serves a dual purpose:
1. **Classical Ground Truth:** Establishes an exact theoretical baseline for pricing options using classical mathematics (Black-Scholes-Merton).
2. **Quantum Demonstration:** Implements a quantum-simulated pricer to demonstrate theoretical quantum advantage—specifically showing how quantum algorithms achieve quadratic speedups in query complexity compared to classical random sampling.

The project culminates in a real-world validation module that tests these algorithms against live, streaming market data to ensure practical relevance.

---

## 2. Requirements & Dependencies

The framework leverages a modern Python ecosystem (Python 3.12 is recommended). Below is the comprehensive matrix of requirements:

### Standard Python Library Modules
No external installation is required for these core modules:
* `math`, `time`: Used for analytical calculations and wall-clock benchmarking.
* `typing`, `dataclasses`, `enum`: Provides strict type-hinting, structural data modeling, and parameter enforcement.
* `argparse`: Facilitates command-line interfaces.

### Classical Third-Party Libraries
Standard quantitative and data science packages required:
* `numpy`: Used heavily for matrix operations, logarithmic returns, and Classical Monte Carlo array generations.
* `pandas`: Handles DataFrame construction for time-series walk-forward results and statistical aggregations.
* `scipy`: Specifically `scipy.stats` (for `norm` distribution handling and `jarque_bera` normality testing).
* `matplotlib`: Configured with the non-interactive `Agg` backend to generate and save analytical plots.
* `yfinance`: Critical for fetching live market spot prices and risk-free rates. Requires an active internet connection.

### Quantum Third-Party Libraries (Strict Version Constraints)
Due to the rapidly evolving nature of the Qiskit ecosystem, you **must** use the following pinned versions to ensure API compatibility. A plain `pip install qiskit-finance` will resolve to a broken combination with newer Qiskit versions.
* `qiskit==0.45.3`: Core quantum circuit framework.
* `qiskit-aer==0.13.3`: High-performance classical simulator for quantum circuits.
* `qiskit-algorithms==0.2.2`: Houses the Iterative Amplitude Estimation (IAE) algorithm.
* `qiskit-finance==0.4.1`: Provides specialized financial applications (`LogNormalDistribution`, `EuropeanCallPricing`).

### Installation Quickstart

**⚠️ Important Note for GitHub Users:** 
Please ensure that you download the pre-packaged uploaded interpreter and the required models contained within the `.venv` and `qenv` files uploaded to this GitHub repository. Utilizing these provided environment files ensures that all complex quantum dependencies and model structures are correctly aligned.

```bash
git clone <this-repo-url>
cd <repo-folder>

# Option 1: Use the uploaded environments (Recommended)
# Ensure you have downloaded the .venv and qenv folders from the repository and activate:
source .venv/bin/activate      # Linux/Mac
# or
.venv\Scripts\activate         # Windows

# Option 2: Build from scratch (if not using the provided .venv / qenv)
python3.12 -m venv venv
source venv/bin/activate
pip install numpy scipy pandas matplotlib yfinance qiskit==0.45.3 qiskit-aer==0.13.3 qiskit-algorithms==0.2.2 qiskit-finance==0.4.1
```

---

## 3. File Breakdown & Functionality

### 3.1. `option_engine.py` (The Classical Baseline)
**Purpose:** The mathematical bedrock of the suite. Implements the classical Black-Scholes-Merton exact analytical model for pricing European options.
**How it works:** It defines a `EuropeanOption` class taking standard financial inputs. Using the `scipy.stats.norm` cumulative distribution function, it computes the exact, closed-form price of the contract and exact partial derivatives ("Greeks": Delta, Gamma, Vega, Theta, Rho). It features strict boundary checks and an interactive Command Line Interface (REPL) for dynamic market parameter tweaking.

### 3.2. `qae_pricer.py` (The Quantum Engine)
**Purpose:** Translates the financial pricing problem into a quantum circuit and solves it using Iterative Amplitude Estimation (IAE).
**How it works:** Initializes a log-normal probability distribution across qubits (via Qiskit Finance) to represent future asset prices. An objective function evaluates the option payoff, mapping it to a quantum operator. The IAE algorithm estimates the expected payoff through quantum phase rotations rather than random sampling. Includes a Classical Monte Carlo (CMC) baseline for cross-validation.

### 3.3. `complexity_comparison.py` (The Benchmarker)
**Purpose:** Rigorous benchmarking study comparing the algorithmic scaling properties of Classical Monte Carlo vs. Iterative Amplitude Estimation.
**How it works:** Prompts for market inputs, computes the exact price, and systematically runs CMC and IAE across a range of precision targets. It isolates *theoretical query complexity* from *wall-clock simulation time*. The generated plot (`complexity_comparison.png`) proves the quantum algorithm scales at $O(1/N)$, a quadratic advantage over classical Monte Carlo's $O(1/\sqrt{N})$.

### 3.4. `real_world_validation.py` (The Live Market Test)
**Purpose:** Removes algorithms from sterile simulations and forces them to confront live, unpredictable market conditions.
**How it works:** Utilizes `yfinance` to pull real-time spot prices and historical data for Apple (AAPL) and the 13-week T-Bill yield. It evaluates the fundamental Black-Scholes assumption (log-returns normality) using the Jarque-Bera statistical test. It then executes a "walk-forward" analysis, calculating the option price using both the classical and quantum engines on real daily data. Results are output to a CSV and a comparison line plot.

---

## 4. Usage Examples

**1. Run the Interactive Analytical Engine:**
```bash
python option_engine.py --repl
```

**2. Run the Quantum Option Pricer:**
```bash
python qae_pricer.py
```

**3. Generate the Algorithm Complexity Benchmark Plot:**
```bash
python complexity_comparison.py
```

**4. Run the Live Real-Market Validation (Requires Internet):**
```bash
python real_world_validation.py
```
