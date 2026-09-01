"""European option pricing engine (Black-Scholes-Merton analytics).

Implements the closed-form pricing model and exact analytical Greeks that
follow from the Geometric Brownian Motion solution derived in the source
paper (Sec. 7, "Derivation and Integration of stochastic differential"):

    S_T = S_0 * exp[(r - q - sigma**2/2) * T + sigma * W_T]   (risk-neutral drift)

which gives ln(S_T) ~ N(ln(S0) + (r - q - sigma**2/2) T, sigma**2 T), the
same log-normal law used in Sec. 4's ``black_scholes_call`` reference
snippet (extended here with a continuous dividend yield q).

Author: engine built for Chirag M.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

_MIN_T = 1e-8          # below this, treat as expired (avoid div-by-zero)
_MIN_SIGMA = 1e-8      # below this, treat as zero-volatility (deterministic) limit


class OptionType(str, Enum):
    """Contract type for a vanilla European option."""
    CALL = "call"
    PUT = "put"


@dataclass
class Greeks:
    """Container for the five analytical Greeks.

    Attributes:
        delta: Sensitivity of price to a $1 move in spot.
        gamma: Sensitivity of delta to a $1 move in spot.
        vega: Sensitivity of price to a 1.00 (100%) change in volatility
            (divide by 100 for a per-1%-point figure).
        theta: Sensitivity of price to the passage of one year of time
            (negative = value decays as T shrinks); divide by 365 for
            per-calendar-day decay.
        rho: Sensitivity of price to a 1.00 (100%) change in the risk-free
            rate (divide by 100 for a per-1%-point figure).
    """
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self) -> dict:
        return asdict(self)


class EuropeanOption:
    """A European vanilla option priced under Black-Scholes-Merton.

    Holds a live set of market/contract inputs (S0, K, r, sigma, T, q) that
    can be mutated at any time via ``update()``; every read (price/greeks)
    recomputes from the current state, so the engine behaves correctly
    under continuous, real-time market data feeds.

    Args:
        S0: Spot price of the underlying. Must be > 0.
        K: Strike price. Must be > 0.
        r: Continuously-compounded risk-free rate (e.g. 0.05 for 5%).
        sigma: Annualized volatility (e.g. 0.20 for 20%). Must be >= 0.
        T: Time to maturity in years. Must be >= 0.
        q: Continuous dividend yield (e.g. 0.02 for 2%). Default 0.
        option_type: ``OptionType.CALL`` or ``OptionType.PUT``.

    Raises:
        ValueError: If any input violates its domain constraint.
    """

    def __init__(
        self,
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        q: float = 0.0,
        option_type: OptionType = OptionType.CALL,
    ) -> None:
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.q = q
        self.option_type = OptionType(option_type)
        self._validate()

    # ------------------------------------------------------------------ #
    # Dynamic input handling
    # ------------------------------------------------------------------ #
    def update(self, **kwargs) -> "EuropeanOption":
        """Apply one or more real-time market updates and re-validate.

        Args:
            **kwargs: Any of S0, K, r, sigma, T, q, option_type.

        Returns:
            self, so calls can be chained: ``opt.update(S0=101).price()``.

        Raises:
            ValueError: On an unknown field or an invalid resulting state.
        """
        valid_fields = {"S0", "K", "r", "sigma", "T", "q", "option_type"}
        for key, value in kwargs.items():
            if key not in valid_fields:
                raise ValueError(f"Unknown field '{key}'. Valid: {valid_fields}")
            setattr(self, key, OptionType(value) if key == "option_type" else value)
        self._validate()
        return self

    def _validate(self) -> None:
        if self.S0 <= 0:
            raise ValueError(f"Spot price S0 must be > 0, got {self.S0}")
        if self.K <= 0:
            raise ValueError(f"Strike price K must be > 0, got {self.K}")
        if self.sigma < 0:
            raise ValueError(f"Volatility sigma must be >= 0, got {self.sigma}")
        if self.T < 0:
            raise ValueError(f"Time to maturity T must be >= 0, got {self.T}")
        if not math.isfinite(self.r):
            raise ValueError(f"Risk-free rate r must be finite, got {self.r}")
        if abs(self.r) > 2.0:
            raise ValueError(f"Risk-free rate r={self.r} is outside a sane [-200%, 200%] range")
        if self.sigma > 5.0:
            raise ValueError(f"Volatility sigma={self.sigma} is outside a sane [0, 500%] range")

    # ------------------------------------------------------------------ #
    # Core math
    # ------------------------------------------------------------------ #
    def _is_degenerate(self) -> bool:
        """True at/near expiration or zero volatility, where d1/d2 blow up."""
        return self.T <= _MIN_T or self.sigma <= _MIN_SIGMA

    def _d1_d2(self) -> tuple[float, float]:
        vol_sqrt_t = self.sigma * math.sqrt(self.T)
        d1 = (math.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        return d1, d2

    def _intrinsic(self) -> float:
        """Discounted intrinsic value: the degenerate-case (T->0 or sigma->0) price."""
        fwd = self.S0 * math.exp((self.r - self.q) * self.T)
        payoff = max(fwd - self.K, 0.0) if self.option_type == OptionType.CALL else max(self.K - fwd, 0.0)
        return math.exp(-self.r * self.T) * payoff

    def price(self) -> float:
        """Analytical Black-Scholes-Merton price.

        Returns:
            The fair value of the option under the risk-neutral measure.
        """
        if self._is_degenerate():
            return self._intrinsic()
        d1, d2 = self._d1_d2()
        disc_q = math.exp(-self.q * self.T)
        disc_r = math.exp(-self.r * self.T)
        if self.option_type == OptionType.CALL:
            return self.S0 * disc_q * norm.cdf(d1) - self.K * disc_r * norm.cdf(d2)
        return self.K * disc_r * norm.cdf(-d2) - self.S0 * disc_q * norm.cdf(-d1)

    def greeks(self) -> Greeks:
        """Exact analytical Greeks (Delta, Gamma, Vega, Theta, Rho).

        In the degenerate limits (T -> 0 or sigma -> 0) these collapse to
        their well-defined boundary values (e.g. Delta -> 0/1 indicator,
        Gamma/Vega -> 0) rather than raising a divide-by-zero error.

        Returns:
            A ``Greeks`` dataclass instance.
        """
        disc_q = math.exp(-self.q * self.T)
        disc_r = math.exp(-self.r * self.T)
        is_call = self.option_type == OptionType.CALL

        if self._is_degenerate():
            fwd = self.S0 * math.exp((self.r - self.q) * self.T)
            itm = (fwd > self.K) if is_call else (fwd < self.K)
            delta = disc_q * (1.0 if itm else 0.0) if is_call else -disc_q * (1.0 if itm else 0.0)
            return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

        d1, d2 = self._d1_d2()
        pdf_d1 = norm.pdf(d1)
        sqrt_t = math.sqrt(self.T)

        gamma = disc_q * pdf_d1 / (self.S0 * self.sigma * sqrt_t)
        vega = self.S0 * disc_q * pdf_d1 * sqrt_t

        if is_call:
            delta = disc_q * norm.cdf(d1)
            theta = (
                -(self.S0 * disc_q * pdf_d1 * self.sigma) / (2 * sqrt_t)
                - self.r * self.K * disc_r * norm.cdf(d2)
                + self.q * self.S0 * disc_q * norm.cdf(d1)
            )
            rho = self.K * self.T * disc_r * norm.cdf(d2)
        else:
            delta = -disc_q * norm.cdf(-d1)
            theta = (
                -(self.S0 * disc_q * pdf_d1 * self.sigma) / (2 * sqrt_t)
                + self.r * self.K * disc_r * norm.cdf(-d2)
                - self.q * self.S0 * disc_q * norm.cdf(-d1)
            )
            rho = -self.K * self.T * disc_r * norm.cdf(-d2)

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

    def summary(self) -> pd.Series:
        """Single-row summary of inputs, price, and Greeks.

        Returns:
            A pandas Series suitable for logging, display, or concatenation
            into a batch DataFrame.
        """
        g = self.greeks()
        return pd.Series({
            "option_type": self.option_type.value,
            "S0": self.S0, "K": self.K, "r": self.r,
            "sigma": self.sigma, "T": self.T, "q": self.q,
            "price": self.price(),
            **g.as_dict(),
        })


class OptionPricingEngine:
    """Batch/portfolio-level wrapper around :class:`EuropeanOption`."""

    @staticmethod
    def price_portfolio(rows: list[dict]) -> pd.DataFrame:
        """Price a batch of contracts in one call.

        Args:
            rows: List of dicts, each with keys S0, K, r, sigma, T and
                optionally q (default 0.0) and option_type (default "call").

        Returns:
            A DataFrame with one row per contract: inputs, price, Greeks.
        """
        records = []
        for row in rows:
            opt = EuropeanOption(
                S0=row["S0"], K=row["K"], r=row["r"], sigma=row["sigma"], T=row["T"],
                q=row.get("q", 0.0), option_type=row.get("option_type", OptionType.CALL),
            )
            records.append(opt.summary())
        return pd.DataFrame(records)


# ---------------------------------------------------------------------- #
# CLI: interactive real-time adjustment loop
# ---------------------------------------------------------------------- #
def _repl() -> None:
    print("European Option Engine — interactive mode. Ctrl+C to exit.")
    S0 = float(input("S0 (spot): "))
    K = float(input("K (strike): "))
    r = float(input("r (risk-free, e.g. 0.05): "))
    sigma = float(input("sigma (vol, e.g. 0.2): "))
    T = float(input("T (years, e.g. 40/365): "))
    q = float(input("q (dividend yield, default 0): ") or 0.0)
    opt_type = input("type (call/put) [call]: ").strip().lower() or "call"

    opt = EuropeanOption(S0, K, r, sigma, T, q, OptionType(opt_type))
    while True:
        print(opt.summary().to_string())
        cmd = input("\nUpdate a field as name=value (e.g. sigma=0.25), or 'quit': ").strip()
        if cmd.lower() in ("quit", "exit", "q"):
            break
        try:
            field, value = cmd.split("=")
            field = field.strip()
            value = value.strip()
            opt.update(**{field: OptionType(value) if field == "option_type" else float(value)})
        except Exception as exc:  # noqa: BLE001 - surfaced directly to the CLI user
            print(f"Invalid input: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="European option Black-Scholes pricing engine.")
    parser.add_argument("--S0", type=float, help="Spot price")
    parser.add_argument("--K", type=float, help="Strike price")
    parser.add_argument("--r", type=float, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, help="Volatility")
    parser.add_argument("--T", type=float, help="Time to maturity (years)")
    parser.add_argument("--q", type=float, default=0.0, help="Dividend yield")
    parser.add_argument("--type", choices=["call", "put"], default="call")
    parser.add_argument("--repl", action="store_true", help="Launch interactive mode")
    args = parser.parse_args()

    if args.repl or args.S0 is None:
        _repl()
        return

    opt = EuropeanOption(args.S0, args.K, args.r, args.sigma, args.T, args.q, OptionType(args.type))
    print(opt.summary().to_string())


if __name__ == "__main__":
    main()
