"""Price one down-and-out call and write results/results.json."""

import json
from pathlib import Path

import numpy as np

from src.down_out import down_out_call, hedge_errors, monte_carlo_price, vanilla_call

SPOT = 100.0
STRIKE = 100.0
BARRIER = 90.0
T = 1.0
R = 0.02
Q = 0.0
SIGMA = 0.20
SIGMA_WRONG = 0.25


def summarise(errors):
    return {
        "mean": float(errors.mean()),
        "std": float(errors.std(ddof=1)),
        "n_paths": int(errors.size),
    }


def main():
    closed = float(down_out_call(SPOT, STRIKE, BARRIER, T, R, Q, SIGMA))
    closed_wrong = float(down_out_call(SPOT, STRIKE, BARRIER, T, R, Q, SIGMA_WRONG))
    vanilla = float(vanilla_call(SPOT, STRIKE, T, R, Q, SIGMA))
    mc = monte_carlo_price(SPOT, STRIKE, BARRIER, T, R, Q, SIGMA, n_steps=252, n_paths=200_000, seed=0)
    right = hedge_errors(SPOT, STRIKE, BARRIER, T, R, Q, SIGMA, SIGMA, n_paths=20_000, seed=1)
    wrong = hedge_errors(SPOT, STRIKE, BARRIER, T, R, Q, SIGMA, SIGMA_WRONG, n_paths=20_000, seed=1)
    payload = {
        "contract": {
            "spot": SPOT,
            "strike": STRIKE,
            "barrier": BARRIER,
            "expiry_years": T,
            "rate": R,
            "dividend_yield": Q,
            "true_volatility": SIGMA,
            "wrong_hedge_volatility": SIGMA_WRONG,
        },
        "vanilla_call": vanilla,
        "closed_form": closed,
        "closed_form_at_hedge_vol": closed_wrong,
        "monte_carlo": mc,
        "hedge_error_correct_vol": summarise(right),
        "hedge_error_wrong_vol": summarise(wrong),
    }
    out = Path(__file__).resolve().parent / "results" / "results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
