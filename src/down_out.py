"""European down-and-out call under geometric Brownian motion.

The barrier H is below the spot and at or below the strike, so the closed
form is the vanilla call minus the down-and-in call. Monte Carlo paths are
daily, and a Brownian bridge fills in the chance of touching H between two
prints. The hedge sells or buys the closed-form delta once a day. The
underlying is always simulated at the true volatility. The delta can be
taken at a different volatility.
"""

import numpy as np
from scipy.stats import norm


def _d1(spot, strike, t, r, q, sigma):
    return (np.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))


def vanilla_call(spot, strike, t, r, q, sigma):
    spot = np.asarray(spot, dtype=float)
    if np.any(t <= 0):
        return np.maximum(spot - strike, 0.0)
    d1 = _d1(spot, strike, t, r, q, sigma)
    d2 = d1 - sigma * np.sqrt(t)
    return spot * np.exp(-q * t) * norm.cdf(d1) - strike * np.exp(-r * t) * norm.cdf(d2)


def down_out_call(spot, strike, barrier, t, r, q, sigma):
    """Closed form. Barrier must be at or below the strike. Price is 0 at or below the barrier."""
    if barrier > strike:
        raise ValueError("barrier must be at or below the strike")
    spot = np.asarray(spot, dtype=float)
    out = np.zeros(np.shape(spot), dtype=float)
    alive = spot > barrier
    if not np.any(alive):
        return out.reshape(np.shape(spot))
    s = spot[alive] if spot.ndim else spot
    if t <= 0:
        value = np.where(s > barrier, np.maximum(s - strike, 0.0), 0.0)
    else:
        lam = (r - q + 0.5 * sigma ** 2) / sigma ** 2
        y = (np.log(barrier ** 2 / (s * strike)) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        vanilla = vanilla_call(s, strike, t, r, q, sigma)
        reflected = (
            s * np.exp(-q * t) * (barrier / s) ** (2 * lam) * norm.cdf(y)
            - strike * np.exp(-r * t) * (barrier / s) ** (2 * lam - 2) * norm.cdf(y - sigma * np.sqrt(t))
        )
        value = vanilla - reflected
    if spot.ndim == 0:
        return float(value)
    out[alive] = value
    return out


def call_delta(spot, strike, barrier, t, r, q, sigma, bump=1e-4):
    """Central difference of the closed form. Zero once the barrier has been touched."""
    spot = np.asarray(spot, dtype=float)
    h = np.maximum(bump * spot, 1e-6)
    up = down_out_call(spot + h, strike, barrier, t, r, q, sigma)
    down = down_out_call(np.maximum(spot - h, barrier * (1 + 1e-8)), strike, barrier, t, r, q, sigma)
    return (up - down) / (2 * h)


def bridge_hit_probability(s0, s1, barrier, sigma, dt):
    """Probability a GBM touches the barrier between two observations, both above it."""
    return np.exp(-2.0 * np.log(s0 / barrier) * np.log(s1 / barrier) / (sigma ** 2 * dt))


def monte_carlo_price(spot, strike, barrier, t, r, q, sigma, n_steps=252, n_paths=100_000, seed=0):
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    s = np.full(n_paths, spot)
    alive = np.ones(n_paths, dtype=bool)
    drift = (r - q - 0.5 * sigma ** 2) * dt
    scale = sigma * np.sqrt(dt)
    for _ in range(n_steps):
        z = rng.standard_normal(n_paths)
        s_new = s * np.exp(drift + scale * z)
        both_above = alive & (s > barrier) & (s_new > barrier)
        hit = alive & (s_new <= barrier)
        if np.any(both_above):
            p_hit = bridge_hit_probability(s[both_above], s_new[both_above], barrier, sigma, dt)
            hit[both_above] = rng.random(both_above.sum()) < p_hit
        alive &= ~hit
        s = np.where(alive, s_new, s)
    payoff = np.where(alive, np.maximum(s - strike, 0.0), 0.0)
    return float(np.exp(-r * t) * payoff.mean())


def hedge_errors(
    spot, strike, barrier, t, r, q, sigma_true, sigma_hedge,
    n_days=252, n_sub=4, n_paths=20_000, seed=1,
):
    """Terminal hedge portfolio minus the option payoff, one number per path.

    The portfolio is started at the closed-form price computed with sigma_true.
    Deltas are taken from sigma_hedge. The path is simulated at sigma_true.
    """
    rng = np.random.default_rng(seed)
    n_steps = n_days * n_sub
    dt = t / n_steps
    s = np.full(n_paths, spot)
    alive = np.ones(n_paths, dtype=bool)
    price0 = down_out_call(spot, strike, barrier, t, r, q, sigma_true)
    shares = np.full(n_paths, call_delta(spot, strike, barrier, t, r, q, sigma_hedge))
    cash = np.full(n_paths, price0 - shares[0] * spot)
    drift = (r - q - 0.5 * sigma_true ** 2) * dt
    scale = sigma_true * np.sqrt(dt)
    for step in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        s_new = s * np.exp(drift + scale * z)
        both_above = alive & (s > barrier) & (s_new > barrier)
        hit = alive & (s_new <= barrier)
        if np.any(both_above):
            p_hit = bridge_hit_probability(s[both_above], s_new[both_above], barrier, sigma_true, dt)
            hit[both_above] = rng.random(both_above.sum()) < p_hit
        cash *= np.exp(r * dt)
        unwind = np.where(s_new <= barrier, s_new, barrier)
        cash = np.where(hit, cash + shares * unwind, cash)
        shares = np.where(hit, 0.0, shares)
        alive &= ~hit
        s = np.where(alive, s_new, s)
        if step % n_sub == 0 and step < n_steps and np.any(alive):
            t_left = t - step * dt
            new_delta = np.zeros(n_paths)
            new_delta[alive] = call_delta(s[alive], strike, barrier, t_left, r, q, sigma_hedge)
            cash = np.where(alive, cash - (new_delta - shares) * s, cash)
            shares = np.where(alive, new_delta, shares)
    payoff = np.where(alive, np.maximum(s - strike, 0.0), 0.0)
    return cash + shares * s - payoff
