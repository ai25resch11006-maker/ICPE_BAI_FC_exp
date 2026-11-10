Certified training (Prop. B.12, eta = 0.1, K = 1000 test rollouts every 10 steps from step 10,000) and inference on the fixed instance (500 seeded trajectories).

| δ′ (train δ) | N | Certified | T / stop step | Training wall-clock | s / training step | Training avg τ (last 500 steps) | Training accuracy p̂ | Final c | Test avg τ (last epoch) | Inference: ICPE error | Inference: ICPE avg τ (max) | Inference: ICPE ms / trajectory | Inference: TaS avg τ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 (0.09) | 100 | yes | 162 / 11,610 | 8.2 h | 2.5 | 8.3 | 0.9095 | 0.0715 | 8.8 | 0.0980 | 10.31 (92) | 20.3 | 27.54 |
| 0.05 (0.04) | 100 | yes | 194 / 11,930 | 8.4 h | 2.5 | 12.0 | 0.9606 | 0.0370 | 12.1 | 0.0080 | 15.32 (99) | 28.7 | 32.07 |
| 0.01 (0.005) | 300 | no | — / 60,000 | 171.3 h | 10.3 | 264.7 | 0.9866 | 0.0000 | 267.1 | 0.0080 | 273.81 (300) | 492.8 | 41.15 |
