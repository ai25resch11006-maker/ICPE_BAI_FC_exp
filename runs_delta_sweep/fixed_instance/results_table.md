Fixed instance μ = [1.0411, 0.218, 1.4861, 0.6195] (best arm 2), env_seed 0; trajectory j uses seed j for every algorithm and δ.

| δ | log(1/δ) | K | Algorithm | Trajectories | Avg stopping time (± 95% CI) | Avg wall-clock (ms/traj) | Error probability | Ended by a_stop |
|---|---|---|---|---|---|---|---|---|
| e^0 = 1 | 0 | 4 | ICPE | 500 | 1.00 ± 0.00 | 11.85 | 1.0000 | 500 / 500 |
| e^0 = 1 | 0 | 4 | TaS | 500 | 10.62 ± 0.51 | 2.12 | 0.0960 | 500 / 500 |
| e^0 = 1 | 0 | 4 | TTPS | 500 | 11.82 ± 0.69 | 1.57 | 0.0940 | 500 / 500 |
| e^0 = 1 | 0 | 4 | Uniform | 500 | 16.98 ± 0.84 | 1.89 | 0.0640 | 500 / 500 |
| e^0 = 1 | 0 | 4 | Lower bound T*(μ)·kl(δ,1−δ) | — | 0.00 | — | — | — |
