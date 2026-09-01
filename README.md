# European Option Pricing Engine

Production-ready Black-Scholes-Merton engine with dynamic market inputs and exact analytical Greeks.

## 1. Theoretical Summary

Scope per the source paper's Section-3-onward content:

- **Sec. 7 (Derivation and Integration of stochastic differential)** solves the GBM SDE via Itô's Lemma to `ln(S_t)`, giving the boxed closed form `S_t = S0 * exp[(μ − σ²/2)t + σW_t]`. Under the risk-neutral measure (μ → r, and here extended with a continuous dividend yield q → μ = r − q), this yields the log-normal law used for pricing.
- **Sec. 4, Step 3** references the standard closed-form Black-Scholes call price built on this log-normal distribution (`d1`, `d2`, `N(·)`); that code block is kept untouched as read-only reference. This engine implements the **put-call-symmetric, dividend-adjusted** generalization of that same formula:

```
d1 = [ln(S0/K) + (r − q + σ²/2)T] / (σ√T)
d2 = d1 − σ√T
Call = S0·e^(−qT)·N(d1) − K·e^(−rT)·N(d2)
Put  = K·e^(−rT)·N(−d2) − S0·e^(−qT)·N(−d1)
```

- **Sec. 4, Step 7 (Classical Monte Carlo)** independently confirms the same log-normal terminal-price sampler; this engine uses the closed-form analytical route instead of simulation, since Sec. 2/3 establish that CMC/QAE are only needed when no closed form exists — the vanilla European option always has one.

Greeks (Δ, Γ, ν, Θ, ρ) are the exact partial derivatives of this price w.r.t. S0, σ, T, and r respectively — see `option_engine.py` docstrings for closed forms.

## 2. Installation

```bash
pip install numpy scipy pandas --break-system-packages
```

Python ≥ 3.10 (uses `tuple[float, float]` built-in generics).

## 3. Quickstart

```python
from option_engine import EuropeanOption, OptionType, OptionPricingEngine

# Single contract, dynamic inputs
opt = EuropeanOption(S0=100, K=105, r=0.05, sigma=0.20, T=40/365, q=0.0,
                      option_type=OptionType.CALL)
print(opt.price())          # 1.0437
print(opt.greeks())         # Greeks(delta=..., gamma=..., vega=..., theta=..., rho=...)

# Real-time market tick -> instant recalculation
opt.update(S0=102, sigma=0.25)
print(opt.summary())        # pandas Series: inputs + price + all Greeks

# Batch / portfolio pricing
df = OptionPricingEngine.price_portfolio([
    {"S0": 100, "K": 105, "r": 0.05, "sigma": 0.20, "T": 40/365, "option_type": "call"},
    {"S0": 100, "K": 95,  "r": 0.05, "sigma": 0.20, "T": 40/365, "option_type": "put"},
])
```

CLI:
```bash
python option_engine.py --S0 100 --K 105 --r 0.05 --sigma 0.2 --T 0.1096 --type call
python option_engine.py --repl   # interactive, prompts + live field updates
```

## 4. API Reference

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

## 5. Edge-Case Guardrails
- Near-zero volatility / at expiration → closed-form degenerates to discounted intrinsic value and boundary Greeks (no NaN/inf).
- Non-finite or extreme `r`/`sigma` are rejected with a `ValueError` at construction or update time.
- `S0`, `K` must be strictly positive; `T`, `sigma` must be non-negative.

## 6. Quantum Companion: `qae_pricer.py`

The paper's actual core contribution is pricing via **Iterative Amplitude Estimation** (Sec. 3–4), not the analytical formula above. `qae_pricer.py` implements that pipeline — log-normal state loading, linear payoff encoding, IAE — as a `QAEOptionPricer` class with the same dynamic-update contract as `EuropeanOption`, plus a `convergence_comparison()` helper reproducing the paper's CMC-vs-IAE convergence study (Sec. 5–6) against this engine's analytical price as ground truth.

**⚠️ Dependency warning:** `pip install qiskit qiskit-aer qiskit-algorithms qiskit-finance` currently pulls `qiskit-finance==0.4.1` (its last release) alongside `qiskit>=2.0`, which **breaks**: `qiskit-finance`'s circuits use a custom instruction current Aer can't transpile (`AerError: unknown instruction: P(X)`). Use this pinned, tested-working stack instead, in a clean virtualenv:

```bash
python3 -m venv qenv && source qenv/bin/activate
pip install qiskit==0.45.3 qiskit-aer==0.13.3 qiskit-algorithms==0.2.2 qiskit-finance==0.4.1 numpy scipy pandas
```

```python
from qae_pricer import QAEOptionPricer, convergence_comparison

pricer = QAEOptionPricer(S0=100, K=105, r=0.05, sigma=0.20, T=40/365, num_uncertainty_qubits=3)
print(pricer.cross_validate())   # qae_price vs. analytical_price, abs/pct error, oracle_queries

pricer.update(S0=102, sigma=0.25)          # live re-price
print(pricer.put_price_via_parity())       # put via put-call parity (EuropeanCallPricing is call-only)

df = convergence_comparison(100, 105, 0.05, 0.20, 40/365)   # CMC error ~1/sqrt(N) vs IAE error ~1/N
```

Measured results from this module are now embedded in the updated paper (`rpsf_updated.tex`, §"Reproduced Empirical Data" and §"Companion Software Implementation and Reproducibility").
