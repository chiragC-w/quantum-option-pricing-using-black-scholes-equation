# Quantum vs. Classical European Option Pricing

A dual-engine European option pricing toolkit built to validate and benchmark the method proposed in:

> Chirag M, Suryaansh S. **"Option Pricing with Qiskit Simulations."** Department of Computer Science and Engineering, BMSIT&M / MITB.
> See [`Option Pricing via Quantum estimation with Qiskit Simulations.pdf`] for the original submission (as authored).

It contains **two independent pricing engines** for a vanilla European option which is an exact closed-form analytical engine and a quantum Iterative Amplitude Estimation (IAE) engine and scripts that benchmark them against each other and against real market data.

## Contents

| File | Purpose |
|---|---|
| `option_engine.py` | Analytical Black-Scholes-Merton engine: closed-form price and Greek letters (Δ, Γ, ν, Θ, ρ), used as ground truth. |
| `qae_pricer.py` | Quantum pricer using Qiskit's Iterative Amplitude Estimation (IAE) which is the paper's actual core method and a classical Monte Carlo (CMC) baseline for comparison. |
| `complexity_comparison.py` | Runs both engines across a range of precision targets and plots [CMC vs. IAE] on two separate panels: theoretical query/sample complexity, and measured wall-clock time. |
| `real_world_validation.py` | Feeds both engines genuine AAPL market data and checks the Black-Scholes/GBM normality assumption and cross-engine agreement against it (not a synthetic test case). |
| `requirements.txt` | Pinned, verified-working dependency versions. |
| `Option Pricing via Quantum estimation with Qiskit Simulations.pdf` | The original submitted paper, **"Option Pricing with Qiskit Simulations."** |

## Why need Two Engines?

Black-Scholes has a closed-form solution, so there's no real need to price a vanilla European option quantumly and the value here is in **validating the paper's quantum method against a known-correct answer** and **honestly measuring what it costs today**, not in using it for actual pricing. `option_engine.py` supplies that known-correct answer so `qae_pricer.py` is the thing being tested.

## Theoretical Background

- The paper's **Section 7** ("Derivation and Integration of stochastic differential") solves the GBM SDE via Itô's Lemma to `ln(S_t)`, giving the boxed closed form `S_t = S0 * exp[(μ − σ²/2)t + σW_t]`. Under the risk-neutral measure (μ → r, and here extended with a continuous dividend yield q → μ = r − q), this yields the log-normal law used for pricing.
- The paper's **Section 4, Step 3** gives the standard closed-form Black-Scholes call price built on this log-normal distribution (`d1`, `d2`, `N(·)`); that code block is kept untouched as read-only reference. `option_engine.py` implements the **put-call-symmetric, dividend-adjusted** generalization of that same formula:

  ```
  d1 = [ln(S0/K) + (r − q + σ²/2)T] / (σ√T)
  d2 = d1 − σ√T
  Call = S0·e^(−qT)·N(d1) − K·e^(−rT)·N(d2)
  Put  = K·e^(−rT)·N(−d2) − S0·e^(−qT)·N(−d1)
  ```

- The paper's **Section 4, Step 7** (Classical Monte Carlo) independently confirms the same log-normal terminal-price sampler; `qae_pricer.py`'s `classical_monte_carlo_price()` reimplements it as the CMC baseline used throughout.
- Greeks (Δ, Γ, ν, Θ, ρ) are the exact partial derivatives of the closed-form price w.r.t. S0, σ, T, and r respectively. See `option_engine.py` docstrings for the closed forms.

## Installation

Please ensure that you download the pre-packaged uploaded interpreter and the required models contained within the `.venv` and `qenv` files uploaded to this GitHub repository which is inside the zip file. Utilizing these provided environment files ensures that all complex quantum dependencies and model structures are correctly aligned.

**Recommended python version: Python 3.12.** With no real quantum computer in the loop, every "quantum" circuit here is simulated via classical matrix algebra (Qiskit Aer's statevector/shot simulator) and this is the classical way of CPU-bound linear algebra, not lightweight scripting, so a modern, fast Python build matters more than it would for typical Python code. Python 3.12 is the best-supported choice for this stack: it's fast enough for Aer's simulation workload, and it's the version this repo's dependency set (below) was verified against.

```bash
git clone <https://github.com/chiragC-w/quantum-option-pricing-using-black-scholes-equation>
cd <repo-folder>

# Option 1: Use the uploaded environments (Recommended)
# Ensure you have downloaded the .venv and qenv folders from the repository and activate:
.venv\Scripts\activate         # Windows

# Option 2: Build from scratch (if not using the provided .venv / qenv)
python3.12 -m venv venv
source venv/bin/activate       #it will be installed in the bin
pip install -r requirements.txt
```

**Dependency note:** `qiskit-finance==0.4.1` (its last release) is **not** compatible with `qiskit>=1.0` which is a a plain `pip install qiskit qiskit-finance` will resolve to a broken combination (`AerError: unknown instruction: P(X)`). `requirements.txt` pins the exact stack that works:
```
qiskit==0.45.3
qiskit-aer==0.13.3
qiskit-algorithms==0.2.2
qiskit-finance==0.4.1
```

## Quickstart

**Analytical engine (Python API):**
```python
from option_engine import EuropeanOption, OptionType, OptionPricingEngine

opt = EuropeanOption(S0=100, K=105, r=0.05, sigma=0.20, T=40/365, q=0.0, option_type=OptionType.CALL)
print(opt.price())              # 1.0437
print(opt.greeks())             # Greeks(delta=..., gamma=..., vega=..., theta=..., rho=...)

opt.update(S0=102, sigma=0.25)  # live market update -> instant recalculation
print(opt.summary())            # pandas Series: inputs + price + all Greeks

# Batch / portfolio pricing
df = OptionPricingEngine.price_portfolio([
    {"S0": 100, "K": 105, "r": 0.05, "sigma": 0.20, "T": 40/365, "option_type": "call"},
    {"S0": 100, "K": 95,  "r": 0.05, "sigma": 0.20, "T": 40/365, "option_type": "put"},
])
```

**Analytical engine (CLI):**
```bash
python option_engine.py --S0 100 --K 105 --r 0.05 --sigma 0.2 --T 0.1096 --type call
python option_engine.py --repl   # interactive mode: prompts + live field updates
```

**Quantum engine (IAE):**
```python
from qae_pricer import QAEOptionPricer, convergence_comparison

qpricer = QAEOptionPricer(S0=100, K=105, r=0.05, sigma=0.20, T=40/365, num_uncertainty_qubits=3)
print(qpricer.cross_validate())        # qae_price vs. analytical_price, abs/pct error, oracle_queries

qpricer.update(S0=102, sigma=0.25)     # live re-price
print(qpricer.put_price_via_parity())  # put via put-call parity (EuropeanCallPricing is call-only)

# CMC-vs-IAE convergence sweep (error ~ 1/sqrt(N) vs error ~ 1/N)
df = convergence_comparison(100, 105, 0.05, 0.20, 40/365)
```

Run the full complexity benchmark and generate its comparison plot:
```bash
python complexity_comparison.py
```

Run the real-market validation:
```bash
python real_world_validation.py
```

## API Reference

### `EuropeanOption(S0, K, r, sigma, T, q=0.0, option_type=OptionType.CALL)`
| Input | Meaning | Constraint |
|---|---|---|
| `S0` | Spot price | > 0 |
| `K` | Strike | > 0 |
| `r` | Risk-free rate | finite, \|r\| ≤ 2.0 |
| `sigma` | Volatility | ≥ 0, ≤ 5.0 |
| `T` | Time to maturity (years) | ≥ 0 |
| `q` | Continuous dividend yield | any real |

- `.update(**kwargs)` — mutate any field, re-validates, returns `self` (chainable).
- `.price()` — analytical price; falls back to discounted intrinsic value when `T≈0` or `sigma≈0` (no divide-by-zero).
- `.greeks()` — returns a `Greeks` dataclass:

  | Greek | Meaning | Sign convention |
  |---|---|---|
  | `delta` | ∂price/∂S0 | Call ∈ [0,1]; Put ∈ [−1,0] |
  | `gamma` | ∂delta/∂S0 | Always ≥ 0 |
  | `vega` | ∂price/∂σ (per 1.00 vol) | Always ≥ 0 |
  | `theta` | ∂price/∂T (per year; ÷365 for daily decay) | Usually ≤ 0 |
  | `rho` | ∂price/∂r (per 1.00 rate) | Call ≥ 0; Put ≤ 0 |

- `.summary()` — one-row `pandas.Series` (inputs + price + Greeks).

### `OptionPricingEngine.price_portfolio(rows: list[dict]) -> pandas.DataFrame`
Vectorized batch pricing; one row of output per input contract dict.

### `QAEOptionPricer(S0, K, r, sigma, T, q=0.0, num_uncertainty_qubits=3, rescaling_factor=0.25)`
| Input | Meaning | Constraint |
|---|---|---|
| `S0`, `K` | Spot / strike | > 0 |
| `sigma`, `T` | Volatility / maturity | **> 0, strictly** (IAE needs a non-degenerate distribution to load; use `EuropeanOption` for the `T=0`/`sigma=0` limits) |
| `num_uncertainty_qubits` | Qubits for the asset-price grid (2^n points) | integer in `[1, 8]` |
| `rescaling_factor` | Slope of the linear payoff approximation | float |

- `.update(**kwargs)` — same chainable contract as `EuropeanOption`, explicit field whitelist (prevents accidentally clobbering a method like `.price`).
- `.price(epsilon_target=0.01, alpha=0.05)` — runs IAE once, returns a `QAEResult(price, oracle_queries, epsilon_target, num_uncertainty_qubits)`. Only **calls** are supported directly (Qiskit Finance's `EuropeanCallPricing` has no put variant).
- `.put_price_via_parity(...)` — put price via put-call parity applied to the QAE call estimate.
- `.cross_validate(epsilon_target=0.01)` — `pandas.Series` with `qae_price`, `analytical_price`, `abs_error`, `pct_error`, `oracle_queries`.

### Module-level functions (`qae_pricer.py`)
- `classical_monte_carlo_price(S0, K, r, sigma, T, N, rng=None, q=0.0) -> (price_estimate, std_error)` which are CMC baseline.
- `convergence_comparison(S0, K, r, sigma, T, epsilon_targets=None, sample_sizes=None, num_uncertainty_qubits=3, seed=42) -> pandas.DataFrame` this sweeps both methods and reports `[method, queries, error]` per run.

## Relationship to the Paper

- `qae_pricer.py` implements the paper's Section 4 ("Implementation in Qiskit") pipeline gives log-normal distribution loading, linear payoff encoding, and Iterative Amplitude Estimation — as a reusable, dynamically-updatable class, rather than a one-off script.
- `option_engine.py` supplies the "exact price" the paper's own results section compares against, extended with a continuous dividend yield and Greeks not in the original snippets.
- `complexity_comparison.py` reproduces the paper's Section 5/6 CMC-vs-IAE convergence claim empirically, and additionally separates **theoretical query complexity** from **measured wall-clock time**. the paper's own claimed advantage is in query complexity (O(1/N) vs. O(1/√N)), and this repo's benchmark shows that does **not** currently translate into a wall-clock speedup on a classical simulator (IAE measured meaningfully slower in real time across runs here, despite using far fewer oracle queries).
- `real_world_validation.py` goes one step further: instead of one synthetic test case, it feeds both engines **real AAPL market data** (see below) and checks the paper's core assumptions against it.

## Real-World Validation (`real_world_validation.py`)

Both engines price under a Black-Scholes / Geometric Brownian Motion assumption: log-returns are normally distributed. This script checks that assumption, and the two engines' agreement, against **genuine market data** rather than a synthetic test case:

- **Data:** 50 real daily AAPL closes, 2026-06-23 to 2026-08-31 ([TipRanks](https://www.tipranks.com/stocks/aapl/historical-prices), pulled 2026-09-01), and the real US 3-month T-bill yield, 3.86% ([TradingEconomics](https://tradingeconomics.com/united-states/3-month-bill-yield), 2026-09-01). Hardcoded in the script for reproducibility but neither source has a free public API.
- **Check 1 — GBM assumption:** tests AAPL's real daily log-returns for skewness, excess kurtosis, and normality (Jarque-Bera). Over this window, the data **rejects normality at 5% significance** (skew −1.18, excess kurtosis +3.83). Real returns have a fatter left tail than Black-Scholes assumes, a real (if small-sample) illustration of a well-known limitation of the GBM model both engines rely on.
- **Check 2 — walk-forward model agreement:** for 30 real trading days, prices a 10-day at-the-money call using that day's *real* spot price and *real* trailing 20-day realized volatility, with both engines. Result: **mean 7.95% price disagreement, correlation 0.99** between the quantum and analytical prices. the two engines track each other closely under real, time-varying conditions, with the quantum pricer's `n=3`-qubit discretization producing a consistent small upward bias (matching the pattern already documented in `complexity_comparison.py`'s results).
- **Deliberately not tested:** a single option price against a single path's realized payoff. That comparison is statistically misleading (an option price is an expectation over many possible paths), so this script avoids it.
- NOTE: this comparison was done around the time this repository was created.

Outputs: `real_world_validation.png` (price series + walk-forward pricing comparison) and `real_world_validation_results.csv` (full walk-forward data, for independent verification).

## Known Limitations & Guardrails

- **Analytical engine:** near-zero volatility / at expiration degenerates cleanly to discounted intrinsic value and boundary Greeks (no NaN/inf); non-finite or extreme `r`/`sigma` are rejected with a `ValueError` at construction or update time; `S0`/`K` must be strictly positive, `T`/`sigma` non-negative.
- **Quantum engine:** `QAEOptionPricer` only supports **calls** directly; puts are obtained via put-call parity. `T`/`sigma` must be strictly positive (no degenerate-limit support — use `EuropeanOption` instead for those cases).
- IAE at `num_uncertainty_qubits=3` has visible discretization along with payoff-approximation error (paper Sec. 6.4); this can dominate over amplitude-estimation shot noise at tight `epsilon_target` values. Use `num_uncertainty_qubits=5` for a more accurate (but deeper, slower) circuit.
- (IMPORTANT) All quantum results here are from Qiskit Aer's classical simulator, not real quantum hardware.
