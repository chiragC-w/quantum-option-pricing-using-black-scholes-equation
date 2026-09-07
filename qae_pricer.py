"""Quantum Amplitude Estimation (IAE) European call pricer — Qiskit.

This is the companion module implementing the paper's actual core
contribution: pricing via the operator A = F o P_X run through 
Iterative Amplitude Estimation.

It also reproduces the empirical claim: CMC error ~ O(1/sqrt(N))
vs IAE error ~ O(1/N), benchmarked against the closed-form Black-Scholes
price from ``option_engine.EuropeanOption``.

Dependency note: requires the pinned stack:
    qiskit==0.45.3  qiskit-aer==0.13.3
    qiskit-algorithms==0.2.2  qiskit-finance==0.4.1
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import IterativeAmplitudeEstimation
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit_finance.applications.estimation import EuropeanCallPricing

from option_engine import EuropeanOption, OptionType


@dataclass
class QAEResult:
    """Outcome of one Iterative Amplitude Estimation run."""
    price: float
    oracle_queries: int
    epsilon_target: float
    num_uncertainty_qubits: int


class QAEOptionPricer:
    """European **call** pricer via Iterative Amplitude Estimation."""

    def __init__(
        self,
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        q: float = 0.0,
        num_uncertainty_qubits: int = 3,
        rescaling_factor: float = 0.25,
    ) -> None:
        self.S0, self.K, self.r, self.sigma, self.T, self.q = S0, K, r, sigma, T, q
        self.num_uncertainty_qubits = num_uncertainty_qubits
        self.rescaling_factor = rescaling_factor
        self._validate()

    def update(self, **kwargs) -> "QAEOptionPricer":
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown field '{key}'")
            setattr(self, key, value)
        self._validate()
        return self

    def _validate(self) -> None:
        if self.S0 <= 0 or self.K <= 0:
            raise ValueError("S0 and K must be > 0")
        if self.sigma <= 0 or self.T <= 0:
            raise ValueError(
                "QAE requires sigma > 0 and T > 0 to load a non-degenerate "
                "log-normal distribution; use EuropeanOption for T=0/sigma=0."
            )
        if not (1 <= self.num_uncertainty_qubits <= 8):
            raise ValueError("num_uncertainty_qubits must be in [1, 8] for simulator feasibility")

    def _build_problem(self):
        mu = (self.r - self.q - 0.5 * self.sigma ** 2) * self.T + np.log(self.S0)
        sigma_bs = self.sigma * np.sqrt(self.T)
        mean = np.exp(mu + sigma_bs ** 2 / 2)
        variance = (np.exp(sigma_bs ** 2) - 1) * np.exp(2 * mu + sigma_bs ** 2)
        stddev = np.sqrt(variance)
        low, high = max(0.0, mean - 3 * stddev), mean + 3 * stddev

        uncertainty_model = LogNormalDistribution(
            self.num_uncertainty_qubits, mu=mu, sigma=sigma_bs ** 2, bounds=(low, high),
        )
        pricer = EuropeanCallPricing(
            num_state_qubits=self.num_uncertainty_qubits,
            strike_price=self.K,
            rescaling_factor=self.rescaling_factor,
            bounds=(low, high),
            uncertainty_model=uncertainty_model,
        )
        return pricer, pricer.to_estimation_problem()

    def price(self, epsilon_target: float = 0.01, alpha: float = 0.05) -> QAEResult:
        pricer, problem = self._build_problem()
        iae = IterativeAmplitudeEstimation(epsilon_target=epsilon_target, alpha=alpha, sampler=Sampler())
        result = iae.estimate(problem)
        return QAEResult(
            price=pricer.interpret(result),
            oracle_queries=result.num_oracle_queries,
            epsilon_target=epsilon_target,
            num_uncertainty_qubits=self.num_uncertainty_qubits,
        )

    def put_price_via_parity(self, epsilon_target: float = 0.01, alpha: float = 0.05) -> float:
        call = self.price(epsilon_target, alpha).price
        return call - self.S0 * np.exp(-self.q * self.T) + self.K * np.exp(-self.r * self.T)

    def cross_validate(self, epsilon_target: float = 0.01) -> pd.Series:
        qae = self.price(epsilon_target)
        analytical = EuropeanOption(self.S0, self.K, self.r, self.sigma, self.T, self.q, OptionType.CALL).price()
        return pd.Series({
            "qae_price": qae.price,
            "analytical_price": analytical,
            "abs_error": abs(qae.price - analytical),
            "pct_error": abs(qae.price - analytical) / analytical * 100,
            "oracle_queries": qae.oracle_queries,
        })


def classical_monte_carlo_price(
    S0: float, K: float, r: float, sigma: float, T: float, N: int,
    rng: Optional[np.random.Generator] = None, q: float = 0.0,
) -> tuple[float, float]:
    rng = rng or np.random.default_rng()
    Z = rng.standard_normal(N)
    S_T = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    discounted_payoffs = np.exp(-r * T) * np.maximum(S_T - K, 0.0)
    return discounted_payoffs.mean(), discounted_payoffs.std(ddof=1) / np.sqrt(N)


def convergence_comparison(
    S0: float, K: float, r: float, sigma: float, T: float,
    epsilon_targets: Optional[list[float]] = None,
    sample_sizes: Optional[list[int]] = None,
    num_uncertainty_qubits: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    epsilon_targets = epsilon_targets or list(np.logspace(-1, -2, num=5))
    sample_sizes = sample_sizes or list(np.logspace(2, 6, num=10, dtype=int))

    exact_price = EuropeanOption(S0, K, r, sigma, T, option_type=OptionType.CALL).price()
    rng = np.random.default_rng(seed)
    rows = []

    for N in sample_sizes:
        est, _ = classical_monte_carlo_price(S0, K, r, sigma, T, int(N), rng)
        rows.append({"method": "CMC", "queries": int(N), "error": abs(est - exact_price)})

    qae_pricer = QAEOptionPricer(S0, K, r, sigma, T, num_uncertainty_qubits=num_uncertainty_qubits)
    for eps in epsilon_targets:
        result = qae_pricer.price(epsilon_target=float(eps))
        rows.append({"method": "IAE", "queries": result.oracle_queries, "error": abs(result.price - exact_price)})

    return pd.DataFrame(rows)


def _prompt_float(prompt_text: str, min_value: Optional[float] = None) -> float:
    """Robust CLI helper that retries until a valid float is entered."""
    while True:
        raw_input = input(prompt_text).strip()
        if not raw_input:
            print("  [!] Input required. Please enter a numerical value.")
            continue
        try:
            val = float(raw_input)
            if min_value is not None and val < min_value:
                print(f"  [!] Value must be >= {min_value}. Try again.")
                continue
            return val
        except ValueError:
            print("  [!] Invalid input. Please enter a valid number (e.g. 100 or 0.05).")


if __name__ == "__main__":
    print("==========================================================")
    print(" Quantum Amplitude Estimation (IAE) Option Pricer")
    print("==========================================================")
    print("Enter the required stock & market inputs:\n")

    S0 = _prompt_float("1. Stock Price (S0, e.g. 100.0): ", min_value=0.0001)
    K = _prompt_float("2. Strike Price (K, e.g. 105.0): ", min_value=0.0001)
    r = _prompt_float("3. Risk-Free Rate (r, e.g. 0.05 for 5%): ")
    sigma = _prompt_float("4. Volatility (sigma, e.g. 0.20 for 20%): ", min_value=0.0001)
    T = _prompt_float("5. Time to Maturity in Years (T, e.g. 0.1095 for 40 days): ", min_value=0.0001)

    print("\n--- Single QAE call price + cross-validation vs analytical ---")
    pricer = QAEOptionPricer(S0, K, r, sigma, T, num_uncertainty_qubits=3)
    print(pricer.cross_validate().to_string())

    print("\n--- CMC vs IAE convergence comparison (Sec. 5/6) ---")
    print("Simulating across multiple target precisions and sample sizes...")
    t0 = time.time()
    df = convergence_comparison(S0, K, r, sigma, T)
    print(df.to_string())
    print(f"(elapsed {time.time() - t0:.1f}s)")
