"""Quantum Amplitude Estimation (IAE) European call pricer — Qiskit.

This is the companion module implementing the paper's actual core
contribution: pricing via the operator A = F o P_X (Sec. 3, "Circuit Design
and Methodology") run through Iterative Amplitude Estimation, following the
Sec. 4 ("Implementation in Qiskit") Steps 2 and 4-9 exactly as given (those
code blocks are kept read-only elsewhere; this module re-derives the same
math independently inside a validated, dynamic-input OOP wrapper).

It also reproduces the Sec. 5/6 empirical claim: CMC error ~ O(1/sqrt(N))
vs IAE error ~ O(1/N), benchmarked against the closed-form Black-Scholes
price from ``option_engine.EuropeanOption`` (the same role ``exact_price``
plays in the paper's Step 3).

Dependency note (see README): qiskit-finance 0.4.1 (its last release) is
NOT compatible with qiskit>=1.0. This module requires the pinned stack:
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
    """Outcome of one Iterative Amplitude Estimation run.

    Attributes:
        price: QAE-estimated option price.
        oracle_queries: Number of Grover-oracle calls IAE used (its
            "sample count" analog to classical N).
        epsilon_target: Target error on the rescaled amplitude that was
            requested.
        num_uncertainty_qubits: Qubits used for the asset-price grid.
    """
    price: float
    oracle_queries: int
    epsilon_target: float
    num_uncertainty_qubits: int


class QAEOptionPricer:
    """European **call** pricer via Iterative Amplitude Estimation.

    Mirrors ``EuropeanOption``'s dynamic market-input contract (S0, K, r,
    sigma, T, q), but prices by loading the risk-neutral log-normal
    terminal-price distribution into a quantum state (``LogNormalDistribution``,
    Sec. 4 Step 4), encoding the payoff via controlled rotations
    (``EuropeanCallPricing``, Step 5), and estimating the resulting
    amplitude with IAE (Step 6) instead of a closed form.

    Only vanilla calls are supported: Qiskit Finance's ``EuropeanCallPricing``
    has no put variant. Price a put via put-call parity using
    :meth:`put_price_via_parity`.

    Args:
        S0: Spot price. Must be > 0.
        K: Strike price. Must be > 0.
        r: Risk-free rate.
        sigma: Volatility. Must be > 0 (IAE needs a non-degenerate
            distribution to load; use ``EuropeanOption`` for the sigma=0
            or T=0 analytical limits).
        T: Time to maturity in years. Must be > 0.
        q: Continuous dividend yield. Default 0.
        num_uncertainty_qubits: Qubits for the asset-price grid (2^n grid
            points). Sec. 6.4: n=3 is shallow but has visible
            discretization error; n=5 reduces that error at the cost of a
            deeper circuit.
        rescaling_factor: Slope of the linear payoff approximation
            (Step 5); controls the payoff-encoding approximation error.
    """

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
        """Apply real-time market updates. Returns self (chainable)."""
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
        """Constructs the distribution-loading + payoff operator (Sec. 3/4 Steps 2, 4, 5)."""
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
        """Runs IAE once (Sec. 4 Step 6) and returns the price estimate.

        Args:
            epsilon_target: Target error on the rescaled amplitude.
            alpha: 1 - confidence level (default: 95% CI).

        Returns:
            A ``QAEResult``.
        """
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
        """European put price via put-call parity applied to the QAE call estimate.

        Put = Call - S0*e^(-qT) + K*e^(-rT). Not part of the paper's circuit;
        a standard analytical bridge since EuropeanCallPricing is call-only.
        """
        call = self.price(epsilon_target, alpha).price
        return call - self.S0 * np.exp(-self.q * self.T) + self.K * np.exp(-self.r * self.T)

    def cross_validate(self, epsilon_target: float = 0.01) -> pd.Series:
        """Compares this QAE call estimate against the analytical Black-Scholes price.

        Returns:
            Series with qae_price, analytical_price, abs_error, pct_error.
        """
        qae = self.price(epsilon_target)
        analytical = EuropeanOption(self.S0, self.K, self.r, self.sigma, self.T, self.q, OptionType.CALL).price()
        return pd.Series({
            "qae_price": qae.price,
            "analytical_price": analytical,
            "abs_error": abs(qae.price - analytical),
            "pct_error": abs(qae.price - analytical) / analytical * 100,
            "oracle_queries": qae.oracle_queries,
        })


# ---------------------------------------------------------------------- #
# Classical Monte Carlo baseline (Sec. 4 Step 7 formula, reimplemented)
# ---------------------------------------------------------------------- #
def classical_monte_carlo_price(
    S0: float, K: float, r: float, sigma: float, T: float, N: int,
    rng: Optional[np.random.Generator] = None, q: float = 0.0,
) -> tuple[float, float]:
    """Classical Monte Carlo European call estimate (CMC baseline, Sec 4 Step 7).

    Args:
        S0, K, r, sigma, T, q: market inputs.
        N: number of sample paths.
        rng: optional numpy Generator for reproducibility.

    Returns:
        (price_estimate, standard_error) tuple.
    """
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
    """Reproduces the Sec. 5/6 CMC-vs-IAE convergence comparison (Step 8/9).

    Sweeps IAE over a range of target precisions and CMC over a range of
    sample sizes, measuring each method's absolute error against the exact
    Black-Scholes price (the same role ``exact_price`` plays in the paper).

    Args:
        S0, K, r, sigma, T: market inputs.
        epsilon_targets: IAE target errors to sweep (default: logspace -1..-2, 5 pts).
        sample_sizes: CMC sample sizes to sweep (default: logspace 2..6, 10 pts).
        num_uncertainty_qubits: qubits for the QAE distribution grid.
        seed: RNG seed for CMC reproducibility.

    Returns:
        DataFrame with columns [method, queries, error], one row per run.
    """
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


if __name__ == "__main__":
    S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.20, 40 / 365

    print("--- Single QAE call price + cross-validation vs analytical ---")
    pricer = QAEOptionPricer(S0, K, r, sigma, T, num_uncertainty_qubits=3)
    print(pricer.cross_validate())

    print("\n--- Dynamic market update -> instant re-price ---")
    pricer.update(S0=102, sigma=0.25)
    print(pricer.cross_validate())

    print("\n--- CMC vs IAE convergence comparison (Sec. 5/6) ---")
    t0 = time.time()
    df = convergence_comparison(100.0, 105.0, 0.05, 0.20, 40 / 365)
    print(df.to_string())
    print(f"(elapsed {time.time() - t0:.1f}s)")
