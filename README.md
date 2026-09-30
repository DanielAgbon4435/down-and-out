# A down-and-out call, and what the hedge does when the volatility is wrong

One European call, strike 100, barrier 90, one year, rate 2%, no dividends. The underlying is geometric Brownian motion with volatility 20%. Nothing here is a market price. The underlying is simulated.

The barrier sits below the strike, so the price is the vanilla call minus the down-and-in call. A daily Monte Carlo uses a Brownian bridge for the chance of touching 90 between two prints. The hedge holds the closed-form delta, rebalanced once a day, and is funded at the true price of 7.30. The delta can be computed at 20% or at 25%.

## What comes out

| | |
|---|---|
| Vanilla call, same strike, no barrier | 8.92 |
| Down-and-out, closed form | 7.30 |
| Down-and-out, 200,000 paths | 7.30 |
| Hedge at 20% vol, 20,000 paths | mean error −0.05, standard deviation 0.28 |
| Hedge at 25% vol, same paths | mean error −0.05, standard deviation 0.78 |

The Monte Carlo matches the closed form. Selling the option at the true price and hedging at the true delta leaves a small residual, because the hedge is daily rather than continuous. Hedging at 25% does not move the average. The option was still sold at 7.30, so the book is not biased. The hedge stops replicating: the standard deviation of the error goes from 0.28 to 0.78.

The same contract priced at 25% volatility is 7.92. That 0.62 gap is a different question. It is the amount you would have overcharged, not a hedging profit.

## Run it

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=. python3 run.py
PYTHONPATH=. python3 -m pytest
```

`run.py` rewrites `results/results.json`.
