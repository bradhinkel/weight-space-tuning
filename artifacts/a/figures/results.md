| run | arm | test irrelevance | mean call acc | recovery | retention (pts) | weights touched | method wall-clock | GPU $ | driver $ (notional, Max) |
|---|---|---|---|---|---|---|---|---|---|
| base | — | 78.75 | 68.40 | 1.000 | -16.22 | — | — | — | — |
| broken | — | 0.83 | 84.62 | 0.000 | +0.00 | 57.95% vs base | 239 min | $2.35 | — |
| Fable b10 | 1 | 82.08 | 69.32 | 1.043 | -15.31 | 4,096 (0.0002%) | 23 min | $0.23 | $8.99 |
| Fable b25 | 1 | 64.58 | 83.49 | 0.818 | -1.13 | 2,041 (0.0001%) | 33 min | $0.33 | $12.81 |
| Fable b50 | 1 | 67.92 | 74.59 | 0.861 | -10.04 | 0.90% | 41 min | $0.40 | $20.93 |
| steering ablation | 2 | 0.83 | 83.46 | 0.000 | -1.16 | 33.09% | 16 min | $0.16 | — |
| corrective LoRA | 3 | 82.50 | 82.73 | 1.048 | -1.89 | 52.82% | 46 min | $0.45 | — |
| Fable b10, unmetered CPU | 1 (separate point) | 55.42 | 82.37 | 0.701 | -2.25 | 0.70% | 141 min | $1.39 | $10.42 |
