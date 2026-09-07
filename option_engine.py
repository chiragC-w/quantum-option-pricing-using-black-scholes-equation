"""European option pricing engine (Black-Scholes-Merton analytics).

Implements the closed-form pricing model and exact analytical Greeks that
follow from the Geometric Brownian Motion solution:

    S_T = S_0 * exp[(r - q - sigma**2/2) * T + sigma * W_T]
!letters are case sensitive!
Author: Chirag M.
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
_MIN_SIGMA = 1e-8      # below this, treat as zero-volatility limit


class OptionType(str, Enum):
    """Contract type for a vanilla European option."""
    CALL = "call"
    PUT = "put"


@dataclass
class Greeks:
    """Container for the five analytical Greeks."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self) -> dict:
        return asdict(self)


class EuropeanOption:
    """A European vanilla option priced under Black-Scholes-Merton."""

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

    def update(self, **kwargs) -> "EuropeanOption":
        """Apply market updates dynamically and re-validate."""
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
            raise ValueError(f"Risk-free rate r={self.r} is outside [-200%, 200%] range")
        if self.sigma > 5.0:
            raise ValueError(f"Volatility sigma={self.sigma} is outside [0, 500%] range")

    def _is_degenerate(self) -> bool:
        return self.T <= _MIN_T or self.sigma <= _MIN_SIGMA

    def _d1_d2(self) -> tuple[float, float]:
        vol_sqrt_t = self.sigma * math.sqrt(self.T)
        d1 = (math.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        return d1, d2

    def _intrinsic(self) -> float:
        fwd = self.S0 * math.exp((self.r - self.q) * self.T)
        payoff = max(fwd - self.K, 0.0) if self.option_type == OptionType.CALL else max(self.K - fwd, 0.0)
        return math.exp(-self.r * self.T) * payoff

    def price(self) -> float:
        if self._is_degenerate():
            return self._intrinsic()
        d1, d2 = self._d1_d2()
        disc_q = math.exp(-self.q * self.T)
        disc_r = math.exp(-self.r * self.T)
        if self.option_type == OptionType.CALL:
            return self.S0 * disc_q * norm.cdf(d1) - self.K * disc_r * norm.cdf(d2)
        return self.K * disc_r * norm.cdf(-d2) - self.S0 * disc_q * norm.cdf(-d1)

    def greeks(self) -> Greeks:
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
        g = self.greeks()
        return pd.Series({
            "Contract Type": self.option_type.value.upper(),
            "Stock Price (Spot)": self.S0,
            "Strike Price": self.K,
            "Risk-Free Rate (Annual)": self.r,
            "Volatility (Annual)": self.sigma,
            "Time to Maturity (Years)": self.T,
            "Dividend Yield": self.q,
            "Theoretical Option Price": self.price(),
            "Price Change per $1 Stock Move": g.delta,
            "Rate of Change for $1 Stock Move": g.gamma,
            "Price Change per 1% Volatility Move": g.vega / 100,
            "Value Decay per 1 Day": g.theta / 365,
            "Price Change per 1% Rate Move": g.rho / 100,
        })


class OptionPricingEngine:
    @staticmethod
    def price_portfolio(rows: list[dict]) -> pd.DataFrame:
        records = []
        for row in rows:
            opt = EuropeanOption(
                S0=row["S0"], K=row["K"], r=row["r"], sigma=row["sigma"], T=row["T"],
                q=row.get("q", 0.0), option_type=row.get("option_type", OptionType.CALL),
            )
            records.append(opt.summary())
        return pd.DataFrame(records)


def _prompt_float(
    prompt_text: str, 
    default: Optional[float] = None, 
    min_value: Optional[float] = None
) -> float:
    """Robust CLI helper that retries until a valid float is entered."""
    while True:
        raw_input = input(prompt_text).strip()
        if not raw_input:
            if default is not None:
                return default
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


def _repl() -> None:
    print("==========================================================")
    print(" European Option Pricing Engine — Interactive Market Mode")
    print("==========================================================")
    print("Enter the required stock & market inputs:\n")

    S0 = _prompt_float("1. Stock Price (S0, e.g. 100.0): ", min_value=0.0001)
    K = _prompt_float("2. Strike Price (K, e.g. 100.0): ", min_value=0.0001)
    r = _prompt_float("3. Risk-Free Rate (r, e.g. 0.05 for 5%): ")
    sigma = _prompt_float("4. Volatility (sigma, e.g. 0.20 for 20%): ", min_value=0.0)
    T = _prompt_float("5. Time to Maturity in Years (T, e.g. 0.5 for 6 months): ", min_value=0.0)
    q = _prompt_float("6. Dividend Yield (q, default 0.0): ", default=0.0)

    while True:
        opt_type_raw = input("7. Option Type (call/put) [default: call]: ").strip().lower() or "call"
        if opt_type_raw in ("call", "put"):
            opt_type = OptionType(opt_type_raw)
            break
        print("  [!] Please type 'call' or 'put'.")

    opt = EuropeanOption(S0=S0, K=K, r=r, sigma=sigma, T=T, q=q, option_type=opt_type)

    print("\n---------------- Initial Calculated Output ----------------")
    print(opt.summary().to_string())
    print("-----------------------------------------------------------\n")

    while True:
        cmd = input("Update parameters (e.g. S0=105 or sigma=0.25), or 'quit': ").strip()
        if cmd.lower() in ("quit", "exit", "q"):
            break
        if not cmd:
            continue
            
        try:
            if "=" not in cmd:
                raise ValueError("Format must include an '=' sign (e.g., S0=102)")
                
            field, value = cmd.split("=", 1)
            field = field.strip()
            value = value.strip()
            parsed_val = OptionType(value.lower()) if field == "option_type" else float(value)
            
            opt.update(**{field: parsed_val})
            print("\n---------------- Updated Calculated Output ----------------")
            print(opt.summary().to_string())
            print("-----------------------------------------------------------\n")
        except ValueError as exc:
            print(f"  [!] Failed to update: {exc}\n")


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
