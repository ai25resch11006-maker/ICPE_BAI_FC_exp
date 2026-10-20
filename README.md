# ICPE — Algorithm 1 (In-Context Pure Explorer), stochastic bandits

This folder trains and evaluates **ICPE** (Algorithm 1 of *In-Context Learning for Pure Exploration*, Russo, Welch, Pacchiano, ICLR 2026) on the paper's **stochastic Gaussian bandit** experiment in the **fixed-confidence** setting: the agent pulls arms of a K-armed bandit until it decides to stop, then names the arm it thinks is best, aiming for accuracy 1 − δ with as few pulls as possible.

The algorithm code is the authors' (branch `neurips_v2` of [rssalessio/icpe](https://github.com/rssalessio/icpe)); by default it trains with the replay loop of branch `neurips_submission`, the code behind the paper's figure 2 (`--update_scheme v2` selects the `neurips_v2` loop). A few changes, marked **CHANGED** in the code, are listed in [Differences from the repository](#differences-from-the-repository). Every file is explained line by line in [CODE_EXPLAINED.md](CODE_EXPLAINED.md).

---

## Quick start

```bash
conda activate bai
cd /DATA1/rf1365/BAI_Erasures/algo
```

(On another machine, create an environment from `requirements.txt` first: see [Requirements](#requirements).)

1. **Train** a model (K = 4 arms, δ = 0.1, seed 1, N = 100, `neurips_submission` replay loop; about 10–13 hours on one GPU):

```bash
mkdir -p runs_submission/K4 && cd runs_submission/K4
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline PYTHONUNBUFFERED=1 nohup setsid python ../../main.py --num_actions 4 --seed 1 --delta 0.1 > train.log 2>&1 < /dev/null &
cd ../..
```

(For N = 200, add `--horizon 200`: about 27 hours, and the model folder ends in `_200_4`.)

2. **Run inference** on one fixed bandit, 5000 seeded trajectories, when training has finished:

```bash
python inference_fixed_env.py --model runs_submission/K4/models/sequential/StochasticGaussianBanditFixedConfidenceSequential_100_4/icpe_30000_1.pkl
```

3. **Read the results** in `runs_submission/K4/models/sequential/StochasticGaussianBanditFixedConfidenceSequential_100_4/inference_env0/trajectories.csv`.

The sections below explain each step and its options.

---

## Requirements

| | |
|---|---|
| Python environment | on this server: the `bai` conda environment, which has everything installed. Install anything new only into `bai`. Elsewhere: Python 3.11 and the packages in `requirements.txt` (below). |
| GPU | one NVIDIA GPU per training run (A6000 here). A run uses about 13 GB of GPU memory with N = 200 (7 GB with N = 100) and keeps the GPU busy. Saved models contain GPU tensors, so loading them also needs a GPU. |
| RAM | the replay buffer needs about 0.04 · N · (K+1) GB: **about 40 GB for K = 4, N = 200** (20 GB for N = 100; 120 GB for K = 14, N = 200). It fills up during the first training steps. |
| Disk | each checkpoint is about 120 MB; a 30,000-step run saves 30 of them (about 3.5 GB). |
| Time (one A6000) | about 3.3 s per training step with N = 200 (**about 27 hours** for 30,000 steps); about 1.6 s with N = 100 (10–13 hours). Inference: at most about 35 minutes for 5000 trajectories with N = 200. |
| wandb | logging is part of the code. Always set `WANDB_MODE=offline` (no account needed; logs stay in a local `wandb/` folder). |

### Compute used (server `ai104`)

All runs and timings in this README come from this machine:

| | |
|---|---|
| Server | `ai104`, Ubuntu 24.04.4 LTS (Linux kernel 6.8.0) |
| CPU | 2 × Intel Xeon Gold 6230 @ 2.10 GHz: 40 cores, 80 threads |
| RAM | 251 GB, plus 255 GB swap |
| GPUs | 3 × NVIDIA RTX A6000, 48 GB each (49,140 MiB) |
| NVIDIA driver | 580.178.04 (supports CUDA up to 13.0; the PyTorch CUDA 12.4 build runs on it) |
| Disk | `/DATA1`: 22 TB volume (about 13 TB free at the time of writing) |
| Shared | the server is shared with other users: check free GPUs with `nvidia-smi` and free RAM with `free -g` before starting a run, and never stop other users' processes |

Measured on this server:

| Run | Time per training step | 30,000 steps | GPU memory | GPU use | RAM (replay buffer full) |
|---|---|---|---|---|---|
| N = 100, K = 4, one run alone on a GPU | about 1.6 s | 10–13 h | about 7 GB | about 95–100% | about 20 GB |
| N = 100, K = 4…14 (K sweep), one run per GPU | about 1.6 s | 10–12.5 h | about 7 GB | about 95–100% | about 4(K+1) GB, up to 60 GB for K = 14 |
| N = 100, two runs on one GPU without MPS | about 3.0 s each | about 25 h | | | |
| N = 100, two runs on one GPU with CUDA MPS | about 2.34 s each | about 19.5 h | | | |
| N = 100, three runs on one GPU with CUDA MPS | about 3.36 s each | about 28 h | | | |
| N = 200, K = 4, one run alone on a GPU | about 3.3 s | about 27 h | about 13 GB | about 60% | about 40 GB |

With N = 200, one training step (after warm-up) splits into: Q_θ updates 44%, Q_θ forward passes while playing 20%, I_ϕ updates 15%, copying transitions into the replay buffer 8%, the rest about 6%. `torch.compile` did not speed up the updates.

### Python packages (`requirements.txt`)

The exact versions the code was run with (the `bai` environment):

| Package | Version | Used for |
|---|---|---|
| `torch` | 2.5.1 (CUDA 12.4 build) | networks, training, GPU |
| `numpy` | 1.26.4 | bandits, replay buffer |
| `pandas` | 2.2.3 | evaluation tables, `trajectories.csv` |
| `transformers` | 4.46.3 | the GPT-2 model used by both networks |
| `omegaconf` | 2.3.0 | the configuration object |
| `wandb` | 0.18.7 | training logs (run offline) |
| `gymnasium` | 1.0.0 | base class of the bandit environments |
| `matplotlib` | 3.11.2 | only `plot_tau_vs_log_delta.py` |

To create a new environment on another machine:

```bash
conda create -n bai python=3.11
```

```bash
conda activate bai
```

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

The extra index gives the CUDA 12.4 build of PyTorch; use the index matching your machine's CUDA driver if it differs. To check that everything is installed:

```bash
python -c "import torch, transformers, omegaconf, wandb, gymnasium, pandas, matplotlib; print('OK, CUDA:', torch.cuda.is_available())"
```

---

## Step 1 — Train

### Command

Run `main.py` from the folder where the outputs should go (all output paths are relative to the current folder):

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python /DATA1/rf1365/BAI_Erasures/algo/main.py --num_actions 4 --seed 1 --delta 0.1
```

`CUDA_VISIBLE_DEVICES` chooses the GPU (0, 1 or 2 on this server; check which are free with `nvidia-smi`).

### Options

| Option | Meaning | Default |
|---|---|---|
| `--num_actions K` | number of arms K; also sets the arm-mean range to [0, 0.4K] and the minimum gap between neighbouring means to 0.4, as the authors' script does | 4 |
| `--horizon N` | maximum pulls per episode: an episode ends when the agent takes STOP, or after N pulls if it never does | 100 (as the authors) |
| `--update_scheme` | replay loop: `submission` = 5 Guesser updates then 5 Chooser updates, each on a fresh batch (branch `neurips_submission`); `v2` = 5 rounds, one shared batch per round (branch `neurips_v2`) | `submission` |
| `--delta δ` | target error probability; the target accuracy is 1 − δ | 0.1 |
| `--seed X` | random seed of the run | 1 |
| `--max_training_steps S` | number of training steps | 30000 |

Every other setting is the authors' and is in `CONFIG` at the top of `icpe_setup.py` (reward noise std 0.5, 100 bandits per training step, batch 1024, 5 update rounds per step, learning rate 3·10⁻⁶ → 10⁻⁶ over the first half of training, λ learning rate 3·10⁻⁵, replay buffer 5,000,000 transitions, transformers 3 layers × 256 × 2 heads).

### Running in the background

Use `nohup setsid` so the run continues after you close the terminal or VS Code:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline PYTHONUNBUFFERED=1 nohup setsid python ../../main.py --num_actions 4 --seed 1 --delta 0.1 > train.log 2>&1 < /dev/null &
```

### Checking progress

Number of training steps completed (out of 30,000):

```bash
grep -c "Step .* completed" train.log
```

Last lines of the log (any error appears here as a `Traceback`):

```bash
tail -n 3 train.log
```

Is it still running, and how busy are the GPUs:

```bash
ps -u $USER -o pid,etime,cmd | grep "[m]ain.py"
```

```bash
nvidia-smi
```

### Stopping

Stop one run by its process id (from the `ps` command above):

```bash
kill <PID>
```

or all of your training runs:

```bash
pkill -u $USER -f "main.py"
```

### Training outputs

In the folder you started from:

| Output | Contents |
|---|---|
| `train.log` | the configuration, then after every training step a debug block for one bandit and `Step <n> completed. Final Action Correct: <mean>/<std>`; `Training complete.` at the end |
| `models/sequential/StochasticGaussianBanditFixedConfidenceSequential_<N>_<K>/icpe_<step>_<seed>.pkl` | the trained networks (Q_θ and I_ϕ) and the configuration, every 1,000 steps and at the end (`icpe_30000_1.pkl`) |
| same folder, `episodes_<seed>.npz` | **CHANGED:** the seed (`seeds`) and arm means (`means`, episodes × K) of every training episode so far, rewritten at every checkpoint. Episode e was played in training step e // 100; its seed is `seed × 100,000,000 + e` |
| `wandb/` | training curves (offline): Guesser accuracy `performance/est_rewards_argmax`, λ `loss/cost`, average episode length `performance/avg_horizon`, losses, learning rates |

### Running inference automatically after training

`runs_N200/K4/inference_when_done.sh` waits until the K = 4 run in `runs_N200/K4/` has finished (final model saved, process exited, `Training complete.` in its log) and then runs Step 2 on it. Start it from `runs_N200/K4/` after starting the training:

```bash
nohup setsid ./inference_when_done.sh > inference_when_done.log 2>&1 < /dev/null &
```

(It is written for that run; for another run, copy it and change the paths and the `main.py` arguments inside.)

---

## Step 2 — Inference on one fixed bandit (`inference_fixed_env.py`)

Plays one bandit many times with a trained model and records every trajectory.

### Command

From the `algo` folder:

```bash
python inference_fixed_env.py --model <path to icpe_<step>_<seed>.pkl>
```

| Option | Meaning | Default |
|---|---|---|
| `--model` | the trained model to use (any checkpoint; the model's own K, N and δ are used) | required |
| `--env_seed` | seed used to draw the fixed bandit from the training distribution | 0 |
| `--num_traj` | number of trajectories | 5000 |
| `--seed_start` | trajectory i uses seed `seed_start + i − 1` | 1 (seeds 1 … 5000) |
| `--out` | output folder | `inference_env<env_seed>/` next to the model |

### What it does

1. Loads the model.
2. Draws **one fixed bandit** from the training distribution with `env_seed` (for K = 4 and env_seed 0: means 1.0411, 0.2180, 1.4861, 0.6195; best arm 2).
3. For each trajectory i = 1 … 5000, with its own seed (`np.random.seed`, `torch.manual_seed`, and the bandit's reward noise all seeded with it): the trained Chooser Q_θ picks the action with the highest value (an arm or STOP) round after round, until it takes STOP or reaches N pulls; then the Guesser I_ϕ names the best arm. The same seed always gives the same trajectory.
4. Measures the wall-clock time of each trajectory (GPU synchronised before and after; one untimed warm-up pass first).

### Outputs

`environment.json`: the fixed bandit (means, stds, best arm, env_seed), K, N, confidence level, model path, seed range, device.

`trajectories.csv`, one row per trajectory:

| Column | Meaning |
|---|---|
| `traj_id` | 1 … 5000 |
| `seed` | seed of the trajectory |
| `stop_step` | round (1-based) in which STOP was taken = `num_pulls` + 1; −1 if it reached N pulls without STOP |
| `num_pulls` | number of arm pulls |
| `stopped` | 1 if STOP was taken, 0 if it reached N pulls |
| `wall_ms` | wall-clock time of the trajectory in milliseconds |
| `correct` | 1 if the predicted arm is the best arm, 0 if not |

A quick summary of a results file (run it inside the output folder):

```bash
python -c "import pandas as pd; d=pd.read_csv('trajectories.csv'); print(d[['num_pulls','stopped','wall_ms','correct']].describe())"
```

---

## Step 3 (optional) — Many new bandits, and E[τ] vs log(1/δ)

The repository's own evaluation, `icpe_inference.sample_trajectories(agent, B, N)`, plays B new bandits × N trajectories each; the authors' notebook uses B = 300, N = 15. `plot_tau_vs_log_delta.py` uses it to plot the expected stopping time and the accuracy against log(1/δ), for models trained with δ = e^(−x) in folders `runs_stochastic_delta/log<x>/`:

```bash
python plot_tau_vs_log_delta.py --logs 2 4 6 8 10
```

Options: `--K` (4), `--seed` (1), `--steps` (checkpoint, 30000), `--B` (300), `--N` (15), `--eval_seed` (0), `--path` (model path template with `{x}`, `{K}`, `{steps}`, `{seed}`), `--out` (`runs_stochastic_delta/tau_vs_log_delta`). It writes `.png`, `.csv` and `.npz`.

---

## Fixed-instance δ sweep: ICPE vs TaS, TTPS, Uniform

One instance μ is drawn once from the training prior (K = 4, `--env_seed 0`) and kept for every δ and every algorithm. For δ = e⁻ˣ, x = 0, 2, 4, 6, 8, 10, each algorithm plays 500 trajectories on it; trajectory j uses seed j for every algorithm and every δ, with common random numbers (the k-th pull of arm a returns the same reward in every run with that seed). Recorded per trajectory: stopping time τ (pulls + 1), correctness, wall-clock time.

- **Baselines** (`baselines.py`, NEW): NumPy ports of the authors' TaS, TTPS (β = 0.65, ε = 0.3) and Uniform with their GLR stopping rule (σ = 0.5, minimum gap 0.4). **No horizon cap**: a trajectory runs until the stopping rule fires.
- **ICPE**: one model per δ, trained with `runs_delta_sweep/train.sh`. Its transformers have exactly N input positions, so a trajectory cannot go past N pulls; choose N so that it is never reached (default N = 300; the baselines need up to about 270 pulls at δ = e⁻¹⁰ on the default instance). Any trajectory that reaches N without STOP is counted in `not_stopped`.
- **λ update for small δ** (CHANGED, `--p_hat_window W`): p̂ is the accuracy over the last W training steps (100 episodes each). With the authors' W = 1, p̂ moves in steps of 0.01 and λ cannot react to targets above 0.99. `train.sh` uses W = ⌈0.1/δ⌉ (1, 1, 6, 41, 299, 2203 for x = 0…10), about 10/δ episodes; `WINDOW=1` restores the authors' update.

```bash
# 1. train the six ICPE models (one queue per GPU; ~60 GB RAM per run at N = 300)
HORIZON=300 GPUS="0 1 2" nohup bash runs_delta_sweep/train.sh > runs_delta_sweep/train_all.log 2>&1 &
# 2. baselines can run right away (CPU, no models needed)
python eval_fixed_instance.py --algos TaS TTPS Uniform
# 3. ICPE, once the models exist (GPU)
python eval_fixed_instance.py --algos ICPE --icpe_N 300
# 4. plots
python plot_fixed_instance.py
```

Outputs in `runs_delta_sweep/fixed_instance/`: `environment.json` (the instance), `log<x>/<algo>.csv` (one row per trajectory), `summary.csv` (per δ and algorithm: average τ ± 95% CI, empirical error probability ± 95% CI and its rule-of-three bound 3/500 when no error is seen, average ms per trajectory), `tau_vs_log_delta.png` and `fixed_instance_panels.png` (τ, error probability, time). With 500 trajectories an error probability below about 0.006 cannot be distinguished from 0.

## Reproducing the authors' runs (N = 100)

The authors' script `experiment_stochastic_fc.sh` trains seeds 1, 2, 3 and K = 14, 12, 10, 8, 6, 4 with N = 100:

```bash
for seed in 1 2 3; do for K in 14 12 10 8 6 4; do WANDB_MODE=offline python main.py --num_actions $K --seed $seed --horizon 100; done; done
```

Their notebook evaluates K = 4, 6, 8 at the step-5,000 checkpoint, K = 10 at step 10,000 and K = 12, 14 at step 30,000.

### Several runs on one GPU (CUDA MPS)

Two ordinary runs on one GPU take turns (each runs at half speed). Under CUDA MPS they run at the same time: measured with N = 100, 2.34 s per step each instead of 3.0 s. Start a private MPS daemon (it is used only by processes that set the same pipe folder):

```bash
mkdir -p /DATA1/rf1365/BAI_Erasures/.mps/pipe /DATA1/rf1365/BAI_Erasures/.mps/log
CUDA_VISIBLE_DEVICES=0,1,2 CUDA_MPS_PIPE_DIRECTORY=/DATA1/rf1365/BAI_Erasures/.mps/pipe CUDA_MPS_LOG_DIRECTORY=/DATA1/rf1365/BAI_Erasures/.mps/log nvidia-cuda-mps-control -d
```

then add `CUDA_MPS_PIPE_DIRECTORY=/DATA1/rf1365/BAI_Erasures/.mps/pipe` in front of each training command. Stop the daemon when done:

```bash
echo quit | CUDA_MPS_PIPE_DIRECTORY=/DATA1/rf1365/BAI_Erasures/.mps/pipe nvidia-cuda-mps-control
```

Check that all runs fit in RAM together (see [Requirements](#requirements)).

---

## How it works (short version)

**The bandits.** K arms; each pull of arm a returns μ_a + 0.5·𝒩(0, 1). The means are drawn in [0, 0.4K] with neighbouring means at least 0.4 apart (`minimum_gap_sampling`), in random order. Every episode uses a new bandit.

**Two networks** (GPT-2 transformers reading the history of pulls):
- **Q_θ, the Chooser**: scores each action (K arms and STOP); the agent takes the highest.
- **I_ϕ, the Guesser**: gives the probability that each arm is the best; the answer is its most likely arm.

**One training step** (repeated 30,000 times):
1. 100 new bandits (100 episodes); **CHANGED:** each episode's reward noise is seeded and recorded.
2. Play them: in each round every unfinished bandit takes one action from Q_θ (during warm-up: random arms, never STOP); every transition goes into the replay buffer. An episode ends at STOP or after N pulls.
3. Learn: 5 rounds; each takes 1,024 random transitions and updates I_ϕ (predict the true best arm) and, after warm-up, Q_θ (value = −λ per pull + log-confidence of the target Guesser in the true arm at the end).
4. Update λ, the price per pull: if the Guesser's accuracy on this step's 100 episodes is below 1 − δ, λ goes down (explore longer); if above, λ goes up (stop sooner).
5. Log; checkpoint every 1,000 steps.

**Target networks** (slow copies used to build Q_θ's targets):

| | ϕ̄, target Guesser | θ̄, target Chooser |
|---|---|---|
| Update | full copy ϕ̄ ← ϕ every **200** Guesser updates = every **40** training steps, for any N (**CHANGED**: the repository uses 2 × N updates) | θ̄ ← 0.01·θ + 0.99·θ̄ after every Chooser update (5 per training step) |
| N = 200 | first copy at about step 46; about 750 copies in 30,000 steps | from step 13; about 150,000 small updates |

**Inference** (Algorithm 1, lines 13–15): no training; Q_θ plays greedily until STOP (or N pulls), then I_ϕ names the best arm.

Details, with line numbers: [CODE_EXPLAINED.md](CODE_EXPLAINED.md).

---

## Files

| File | Contents |
|---|---|
| `main.py` | entry point: options → configuration → training → final save |
| `icpe_setup.py` | `CONFIG` (all settings), the networks and their target copies, replay buffer, `ICPEAgent` |
| `icpe_sampling.py` | bandits (with the CHANGED per-episode seed), arm-mean sampling, history, choosing actions |
| `icpe_training.py` | training loop, Guesser update and ϕ̄ copies, λ update, checkpoints and episode file |
| `icpe_q_learning.py` | Chooser update and θ̄ update |
| `icpe_inference.py` | the repository's evaluation (`sample_trajectories`), `save`, `load` |
| `inference_fixed_env.py` | **new**: Step 2 |
| `plot_tau_vs_log_delta.py` | **new**: Step 3 |
| `CODE_EXPLAINED.md` | line-by-line explanation |
| `requirements.txt` | Python packages and exact versions |
| `plot_tau_vs_delta.py`, `runs_delta/`, `ICPE_overview.*` | from an earlier deterministic version; do not work with this code |

---

## Runs and results in this folder

| Folder | N | K | δ | Status |
|---|---|---|---|---|
| `runs_stochastic_delta/log<x>/` | 100 | 4 | e^(−x), x = 2, 4, 6, 8, 10 | trained (Oct 15–16) and evaluated, results below |
| `runs_stochastic_K/K<K>/` | 100 | 14, 12, 10, 8, 6, 4 | 0.1 | trained (Oct 16–17, 10–12.5 h each), not yet evaluated |
| `runs_N200/K4_stopped_at_step450/` | 200 | 4 | 0.1 | 1st attempt, stopped on request at step 450 (no checkpoint) |
| `runs_N200/K4/` | 200 | 4 | 0.1 | 2nd attempt, stopped on request at step 38 (no checkpoint); no inference run |

**There is no N = 200 model yet.** Starting Step 1 in `runs_N200/K4/` again replaces its `train.log`.

The two N = 100 sweeps were trained before the per-episode seeds and the fixed T_ϕ were added: their reward noise used the global random generator, and they have no `episodes_<seed>.npz`. With N = 100 their T_ϕ was 200 updates, the same as now. Their models load and run with the current code (each model carries its own N).

### δ sweep: E[τ] vs log(1/δ) (K = 4, N = 100, seed 1, 30,000 steps)

Evaluated with `plot_tau_vs_log_delta.py` (300 bandits × 15 trajectories per model):

![E[τ] vs log(1/δ)](runs_stochastic_delta/tau_vs_log_delta.png)

| log(1/δ) | E[τ] ± 95% CI | Accuracy ± 95% CI | Target 1 − δ | λ at the end of training | Steps with λ > 0 |
|---|---|---|---|---|---|
| 2 | 6.57 ± 0.11 | 0.8758 ± 0.0114 | 0.86466 | 0.096 | 29,623 / 30,000 |
| 4 | 13.59 ± 0.32 | 0.9816 ± 0.0039 | 0.98168 | 0.023 | 25,831 / 30,000 |
| 6 | 61.80 ± 1.84 | 0.9949 ± 0.0026 | 0.99752 | 0.0000067 | 4,136 / 30,000 |
| 8 | 57.73 ± 1.50 | 0.9984 ± 0.0013 | 0.99966 | 0 | 0 / 30,000 |
| 10 | 57.73 ± 1.50 | 0.9984 ± 0.0013 | 0.99995 | 0 | 0 / 30,000 |

- For log(1/δ) = 2 and 4 the stopping time grows with log(1/δ) and the accuracy meets the target.
- For log(1/δ) = 8 and 10 the trained models are identical: λ moves against the recent average of (1 − δ) − accuracy on each step's 100 episodes, starts at 0 and cannot go below 0. With targets ≥ 0.99966 and an accuracy of about 0.998 that average stayed positive (all 100 correct gives −0.0003, one mistake gives +0.0097), so λ stayed 0 and δ had no effect on training.
- For log(1/δ) = 6, λ was near 0 most of the time, so it is almost the same policy; 61.8 vs 57.7 is the difference between two training runs.
- So with these settings the stopping time follows δ up to log(1/δ) ≈ 4–5; beyond that the accuracy levels off at about 0.995–0.998.

---

## Differences from the repository

The algorithm code is copied from the `neurips_v2` branch (commit `7833a2f`; local clone in `../ICPE`). Not from the repository:

- **CHANGED: replay loop of `neurips_submission` by default** (`agent.update_scheme = 'submission'`, `icpe_training.train_networks_submission`): 5 Guesser updates on fresh batches, then 5 Chooser updates on other fresh batches, as in commit `0260c4f`, the code that produced the paper's figure 2 (its results file is identical in both branches and dated 21 May 2025, before the `neurips_v2` commits). `--update_scheme v2` restores the `neurips_v2` loop (one shared batch per round) and its `np.random.rand() < 5e-4` attention-dump call. The other differences between the two branches (the unused `timestep` argument, renames, buffer dtypes) do not change training.
- Horizon N = 100 as in the authors' script (it was 200 in an earlier version of this folder; `runs_N200/` used `--horizon 200`).
- **CHANGED: T_ϕ fixed at 200 Guesser updates** (every 40 training steps) for any N; the repository uses `frequency_update_e_network × N`, which would give 400 with N = 200 (`icpe_training.py` L77).
- **CHANGED: per-episode seeds in training.** Each training episode's reward noise comes from its own generator (`GaussianBanditEnv.seed`), seeded with `experiment.seed × 100,000,000 + episode number`; all seeds and means are saved in `episodes_<seed>.npz`. In the repository the noise uses the global random generator, so runs are no longer bit-identical to the repository's. `sample_trajectories` does not seed its bandits, as in the repository.
- Hydra configuration replaced by the `CONFIG` dict and `argparse` options; models are saved under `models/sequential/` in the current folder.
- Fixed confidence only (the stop action, the per-pull cost λ and its update always on). Other bandit types, magic actions, baselines and other agents are left out; commented-out repository code is not copied.
- The code is split into six files whose functions take the agent as an argument.
- New files: `inference_fixed_env.py`, `plot_tau_vs_log_delta.py`, `runs_*/` helper scripts (`launch.sh`, `plot_when_done.sh`, `inference_when_done.sh`).
- Before the CHANGED parts were added, 28 training steps of this code and of the authors' code (same settings and seed) gave identical network weights and identical printed output.
