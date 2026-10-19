# ICPE, stochastic bandits (fixed confidence): line-by-line explanation of `algo/`

This document explains every part of the code in `algo/`, in the order the program runs, with line numbers (`L…`). The algorithm code is the authors' implementation from the `neurips_v2` branch of [rssalessio/icpe](https://github.com/rssalessio/icpe), split into six files. Everything that is **not** from the repository is marked **CHANGED** in the code and in this document.

| File | What it contains | Algorithm 1 |
|---|---|---|
| [main.py](main.py) | reads options, builds the config, trains, saves | all |
| [icpe_setup.py](icpe_setup.py) | settings, the two GPT-2 networks and their target copies, replay buffer, agent | lines 1–2 |
| [icpe_sampling.py](icpe_sampling.py) | history, bandits, gap sampling, choosing actions | lines 4, 6 |
| [icpe_training.py](icpe_training.py) | training loop, I_ϕ update, ϕ̄ copies, λ update, logging | lines 3–12 |
| [icpe_q_learning.py](icpe_q_learning.py) | Q_θ update, θ̄ update | lines 10–11 |
| [icpe_inference.py](icpe_inference.py) | test-time play, answer, stopping time, save/load | lines 13–15 |
| [inference_fixed_env.py](inference_fixed_env.py) | **new**: inference on one fixed bandit, one seeded trajectory at a time | lines 14–15 |
| [plot_tau_vs_log_delta.py](plot_tau_vs_log_delta.py) | **new**: E[τ] vs log(1/δ) plot from `sample_trajectories` | lines 13–15 |

**Running example** (current defaults): **K = 4** arms, **N = 200** maximum pulls (**CHANGED**: the authors' script uses N = 100), **B = 100** bandits per training step, batch size **1024**. Shapes are written like `(100, 200, 5)` = (bandits, history rows, numbers per row).

**Words used below:**
- **Chooser** = Q_θ (`critic`): scores each action (4 arms + STOP).
- **Guesser** = I_ϕ (`i_net`): gives the probability that each arm is the best.
- **Target copies** θ̄ (`critic_tgt`) and ϕ̄ (`i_tgt_net`): lagging versions of the two networks, used only to compute training targets (section 7).
- **λ** (`_cost_multiplier`): the price paid per pull.
- **Training step**: one pass of the outer loop (100 new bandits, play, learn, update λ). **Round**: one pass of the inner play loop (each unfinished bandit takes one action). **Episode**: one bandit played from an empty history until STOP or N pulls.

---

## 1. `main.py`, the entry point

### L1–17: docstring
Says what the file is: one run of the authors' launch script `experiment_stochastic_fc.sh`, for stochastic bandits, fixed confidence, with the horizon default changed to N = 200.

### L18–27: imports
`argparse` (command line), `os` (folders), `numpy`/`torch` (seeding), `wandb` (logging), `datetime`, `OmegaConf`, and from the other files `create_config`, `ICPEAgent`, `train`, `save`.

### L74–97: what runs first (`if __name__ == '__main__':`)
| Line | Code | Meaning |
|---|---|---|
| L75 | `argparse.ArgumentParser(...)` | sets up command-line options |
| L76–82 | `--num_actions`, `--horizon`, `--max_training_steps`, `--seed`, `--delta` | all default to `None` = "not given, keep the value in `CONFIG`" (N = 200, 30,000 steps, seed 1, δ = 0.1) |
| L83 | `args = parser.parse_args()` | reads what you typed |
| L85 | `K = args.num_actions if ... else create_config().environment.num_actions` | the K this run uses: yours, or the default 4 |
| L86–96 | `overrides = {...}` | the changes to apply to `CONFIG`, as dotted paths |
| L91 | `'agent.confidence_level': None if args.delta is None else 1 - args.delta` | `--delta 0.1` → target accuracy 0.9 |
| L94 | `'environment.bandit_env.max_value': float('%.6g' % (0.4 * K))` | upper end of the arm-mean range = 0.4K, written with 6 significant digits exactly as the authors' script's `awk` prints it (K = 4 → 1.6) |
| L95 | `'environment.bandit_env.minimum_gap': float('%.6g' % (1 / K))` | gap setting on the 0-to-1 ruler = 1/K (K = 4 → 0.25; the real gap is 0.25 × 1.6 = 0.4) |
| L97 | `main(create_config(**{k: v ... if v is not None}))` | drops the "not given" entries, builds the config, runs `main` |

### L30–71: `main(cfg)`
| Line | Code | Meaning |
|---|---|---|
| L31 | `date = ...strftime(...)` | timestamp for the wandb note |
| L32 | `print(f'Configuration {cfg}')` | prints the full config into the log |
| L34–35 | `np.random.seed(...)`, `torch.manual_seed(...)` | fixes all randomness: same seed → identical run (bandits, initial weights; the reward noise is seeded per episode, section 4) |
| L37–50 | `wandb.init(...)` | starts a log run (offline with `WANDB_MODE=offline`). Project = experiment name; run name like `ICPE - (200/4/0)` = (N/K/oracle) |
| L52 | `device = 'cuda' if ... else 'cpu'` | GPU if available |
| L56 | `agent = ICPEAgent(cfg, device)` | **Algorithm 1, lines 1–2**: buffer, 4 networks (Q_θ, θ̄, I_ϕ, ϕ̄), optimizers, λ, schedulers |
| L58–62 | `PATH = f'models/sequential/{name}_{N}_{K}/'`, `os.makedirs` | output folder, relative to where you run the command (e.g. `..._200_4/`) |
| L65 | `train(agent, PATH)` | **Algorithm 1, lines 3–12**: 30,000 training steps, with a checkpoint every 1,000 |
| L67 | `save(agent, PATH)` | final model `icpe_<steps>_<seed>.pkl` |
| L69–71 | `print(...)`, `wandb.finish()` | done |

No evaluation (Algorithm 1, lines 13–15) is run here, as in the authors' training script; it is done afterwards on a saved model (section 6).

---

## 2. `icpe_setup.py`, settings, networks, buffer, agent

### L1–34: docstring and imports
Which repository files each part comes from. Imports include `GPT2Config`/`GPT2Model` (Hugging Face), `LinearLR` (learning-rate decay), `OmegaConf`, `NamedTuple`.

### L46–125: `CONFIG`, Algorithm 1 line 1 (the inputs)
The authors' experiment settings, merged the way Hydra merges their YAML files, with their launch script's values, except **N = 200 (CHANGED, L79)**. `max_value` and `minimum_gap` are those of K = 4; `main.py` resets them for other K. Meaning of each key, and where it is used:

| Key | Value | Meaning | Used in |
|---|---|---|---|
| `experiment.name` | `StochasticGaussianBanditFixedConfidenceSequential` | wandb project, model folder | main.py L38, L58 |
| `experiment.description` | text | wandb note only | main.py L39 |
| `experiment.seed` | 1 | random seed (authors: 1, 2, 3); also the base of the episode seeds | main.py L34–35; training.py L199; file names |
| `experiment.num_processes` | 5 | TaS baseline only | unused |
| `training.save_frequency` | 1000 | checkpoint (and `episodes_<seed>.npz`) every 1,000 steps | training.py L310–314 |
| `training.max_training_steps` | 30000 | number of training steps (× 100 bandits = 3 million episodes); also the decay length | training.py L194; setup L579–580 |
| `training.use_oracle_reward` | False | Chooser's reward comes from the learned Guesser | training.py L89; q_learning.py L28 |
| `training.cost_per_timestep` | 0 | old fixed-price setting | unused |
| `training.eval.*`, `eval.*` | 100/10/1, 20/20 | evaluation sizes of other experiments' scripts (left over from the shared base config) | unused |
| `environment.bandit_env.type` | `GaussianBandit` | reward = mean + noise | sampling.py L393 |
| `min_value`, `max_value` | 0, 1.6 | arm means in [0, 0.4K] | sampling.py L398 |
| `std` | 0.5 | noise of each pull | sampling.py L399 |
| `minimum_gap` | 0.25 | gap on the 0-to-1 ruler (real gap 0.4) | sampling.py L394–398 |
| `num_magic_actions`, `std_magic_action` | 0 | magic-arm variant (not included) | unused |
| `environment.num_actions` | 4 | K | many |
| `environment.horizon` | **200 (CHANGED from 100)** | N: max pulls per episode (an episode ends at STOP or after N pulls), history length, GPT-2 context | many |
| `agent.type` | `ICPE` | label | main.py |
| `learning_rate` → `final_learning_rate` | 3e-6 → 1e-6 | step size of both networks, decaying linearly over the first 15,000 steps | setup L559–584 |
| `grad_clipping` | 1 | max gradient size | training.py L70; q_learning.py L59 |
| `buffer_size` | 5,000,000 | replay buffer capacity | setup L516 |
| `gamma` | 1 | no discounting | q_learning.py L39 |
| `batch_size` | 1024 | memories per update | training.py L88 |
| `num_env_instances`, `num_trajectories` | 100, 1 | 100 different bandits per step, each played once | training.py L188 |
| `polyak_update_q_network` | 0.01 | **T_θ**: θ̄ moves 1% toward θ after every Q_θ update | q_learning.py L66 |
| `frequency_update_e_network` | 2 | **T_ϕ**: ϕ̄ ← ϕ every 2 × 100 = 200 Guesser updates, for any N (CHANGED: the repository uses 2 × N) | training.py L77 |
| `replay_ratio` | 5 | update rounds per training step | training.py L86–87 |
| `e_network_bootstrap_num_batches` | 50 | Q_θ starts after 50 × 1024 × 5 = 256,000 memories | training.py L94; setup L530–531 |
| `initial_buffer_bootstrap_num_batches` | 20 | I_ϕ starts after 20 × 1024 × 5 = 102,400 memories | training.py L86 |
| `confidence_level` | 0.9 | target accuracy 1 − δ | training.py L106 |
| `cost_learning_rate` | 3e-5 | step size of λ | setup L562–570 |
| `critic_network`, `e_network` | 256 wide, 3 layers, 2 heads, dropout 0, gelu | size of Q_θ and I_ϕ (`dropout`, `layer_norm_eps` unused) | setup L550–557 |

### L128–136: `create_config(**overrides)`
- `OmegaConf.create(CONFIG)` (L133) turns the dict into a config object (`cfg.agent.batch_size` style).
- `OmegaConf.update(cfg, key, value)` (L134–135) applies each dotted override, e.g. `'environment.num_actions'`.

### L142–145: `TransformerState`
A pair `(state, timestep)`: the network input tensor and the number of real pulls of each history. The networks accept `timestep` but do not use it (the authors' code that used it is commented out).

### L147–155: `TransformerParameters`
A dataclass listing the network settings. Only documentation: the code passes the config block in its place.

### L158–174: `BaseTransformer`, the parent of both networks
This is where the two ways of updating a target copy are defined (used in section 7):
- **`hard_update(source)` (L162–164):** `self.load_state_dict(source.state_dict())`: every weight of this network becomes an **exact copy** of `source`. Used for ϕ̄ ← ϕ, and for θ̄ ← θ, ϕ̄ ← ϕ at the start.
- **`soft_update(source, tau)` (L166–168) → `polyak_update` (L170–173):** for every weight, `target = tau·source + (1−tau)·target` (L173). With tau = 0.01: θ̄ ← 0.01·θ + 0.99·θ̄, a 1% step toward θ.

### L176–191: `create_gpt2_config(horizon, params)`
Builds a **small, untrained** GPT-2: `n_positions = N = 200` (L182: rows it can read; a longer history would not fit), `n_embd = 256`, `n_layer = 3`, `n_head = 2`, dropouts 0, `use_cache = False`, activation `gelu`.

### L200–250: `CriticTransformer` = Q_θ, the Chooser
- **`__init__` (L201–221):** stores sizes; creates `self._gpt2 = GPT2Model(config)`; `self._embed = Linear(K+1 → 256)` (L219, embeds one history row); `self._output = Linear(256 → K+1)` (L221, one score per action: 4 arms + STOP).
- **`forward(state, timestep, ...)` (L223–250):**

| Code | Shape | Meaning |
|---|---|---|
| `embeds = state` | (B, 200, 5) | the histories |
| `embeds = F.gelu(self._embed(embeds))` | (B, 200, 256) | each row → a 256-number vector |
| `h = self._gpt2.forward(inputs_embeds=embeds)` | | GPT-2 adds position information and lets rows attend to earlier rows (3 layers) |
| `h = h.last_hidden_state` | (B, 200, 256) | output for every row |
| `h = h[:,-1,:]` (L246) | (B, 256) | keep the **last row**, which has attended to all rows |
| `out = self._output(h)` | (B, 5) | **Q_θ(history, a)** for a = arm 0–3, STOP |

`timestep`, `_save_attention`, `step`, `traj_t` are accepted but unused (their code is commented out in the repository and not copied).

### L257–307: `InferenceTransformer` = I_ϕ, the Guesser
The same structure, except `self._output = Linear(256 → K)` (L276) and at the end `F.log_softmax(out, dim=-1)` (L305). The output (B, 4) is **log P(arm k is the best | history)**.

### L313–340: weight initialization and builders
- `init_weights(m)` (L313–317): for every `nn.Linear` (the input and output layers), weights ~ truncated normal with std 1/√(inputs), biases 0. GPT-2's inner layers keep Hugging Face's own initialization.
- `create_critic_network(...)` (L320–329), `create_inference_network(...)` (L332–340): build the network and apply `init_weights`.

### L346–355: `ReplayBufferSamples`
One sampled batch: `states`, `actions`, `rewards`, `env_rewards`, `next_states`, `dones`, `optimal_actions`, `timesteps`.

### L358–482: `ReplayBuffer`, the memory box ℬ
- **`__init__` (L359–377):** stores sizes; `enable_last_transitions_sampling = False` (so `get_last_inds` is never used); calls `reset()`.
- **`reset` (L379–397):** pre-allocates numpy arrays: `states` and `next_states` (5M, 200, 5) float32 (L385–386), about 20 GB each, taking RAM only as they fill; `actions`, `rewards`, `env_rewards`, `dones`, `optimal_actions`, `timesteps` (5M,). `pos` = next free slot, `full` = wrapped around yet.
- **`add(...)` (L399–444):** stores the transitions of one round for all 100 bandits. `not_stopped = ~stopped` (L413) skips bandits that stopped in an earlier round. It writes from `pos`, and when it reaches the end, writes the rest at the beginning (overwriting the oldest memories, `full = True`).
- **`sample(batch_size, ...)` (L458–479):** `np.random.randint(0, filled size, 1024)` random positions (L461), shuffled, returned as GPU tensors.
- **`__len__` (L481–482):** number of memories stored.

### L488–584: `ICPEAgent`, Algorithm 1 line 2
A container for everything; the other files' functions take it as `agent`.

| Line | Code | Meaning |
|---|---|---|
| L500–505 | `self.cfg`, `self.agent_cfg`, `self.device`, counters `current_step`, `gradient_step`, `e_gradient_step` | settings and counters: training steps, Q_θ updates, I_ϕ updates (`e_gradient_step` drives the ϕ̄ copies) |
| L507 | `self.num_actions = K + 1` | 5 actions: 4 arms + STOP |
| L508 | `self.horizon = N` | 200 |
| L511 | `self.use_amp = (self.device != "cpu")` | mixed precision (float16 where safe) on the GPU only |
| L512–513 | `q_scaler`, `i_scaler = GradScaler(...)` | scale the losses so float16 gradients don't underflow (does nothing on CPU) |
| L516–524 | `self.buffer = ReplayBuffer(5M, state_dim=5, ..., horizon=200)` | the memory box |
| L526 | `self.reset_networks()` | builds networks, target copies, optimizers, λ, schedulers |
| L527 | `self.running_mean_est = 0` | unused (as in the repository) |
| L529–531 | `has_training_started` | True once the buffer has > 256,000 memories: then actions are argmax Q_θ instead of random |
| L533–539 | `base_num_actions` = K = 4; `state_dim` = K + 1 = 5 | sizes |
| L541–543 | `cost_per_timestep` → `self._cost_multiplier.item()` | **λ, the current price per pull**, as a plain number |

**`reset_networks` (L546–584):**
| Line | Code | Meaning |
|---|---|---|
| L547 | `bins = 1` | no distributional variant |
| L550–552 | `critic`, `critic_tgt`, `critic_tgt.hard_update(critic)` | Q_θ, and its target θ̄ created as an exact copy: **θ̄ ← θ** (Algorithm 1 line 2) |
| L555–557 | `i_net`, `i_tgt_net`, `i_tgt_net.hard_update(i_net)` | I_ϕ, and **ϕ̄ ← ϕ** |
| L559–560 | `AdamW(..., lr=3e-6)` for Q_θ and I_ϕ | the optimizers of the two learning networks (the target copies have no optimizer: they only change by copying) |
| L562–567 | `cost_lr = cost_learning_rate` (3e-5) | step size of λ |
| L568 | `self.min_cost = 0` | lower limit of λ |
| L569 | `self._cost_multiplier = torch.tensor(0, requires_grad=True)` | **λ starts at 0**, as a learnable number |
| L570 | `self.cost_optimizer = AdamW([λ], lr=3e-5)` | the optimizer that adjusts λ |
| L573–578 | `end_factor = 1e-6 / 3e-6 = 1/3` | the learning rate will end at one third |
| L579–580 | `LinearLR(..., total_iters=15000)` for both optimizers | learning rate falls linearly over 15,000 steps, then stays at 1e-6 |
| L582–584 | `self.scheduler_step = lambda: (...)` | one call steps both schedules (called once per training step) |

---

## 3. `icpe_sampling.py`, history, bandits, actions

### L1–35: docstring and imports
Includes `from icpe_setup import TransformerState`.

### L41–45: `Experience`
A named tuple `(step, reward, action, done)` for one pull: `step` is the row it goes into (0 … N−1). Only `step`, `action` and `reward` are written into the history.

### L50–197: `State`, the history 𝒟_t
Stored as an array (…, N, 2): row t = `[arm index, reward]` of pull t, unused rows `[0, 0]`. (`num_actions = 1` here means "one action column", not K. The docstring, copied from the repository, says "one-hot" and `2*reward_dim`; the array actually holds the arm number and has 1 + 1 columns.)

| Lines | Method | What it does |
|---|---|---|
| L51–63 | `__init__` | stores the array and the column counts |
| L65–78 | `zeros_numpy(200)` | empty (200, 2) history (float64) |
| L80–95 | `zeros_torch` | same in torch (unused) |
| L97–103 | `actions` | column 0 as integers → (…, 200) |
| L105–107 | `rewards` | last column → (…, 200) |
| L109–112 | `add(exp)` | writes `[action, reward]` into row `exp.step` |
| L114–115 | `to_tensor(device)` | numpy → float32 torch on the device |
| L117–128 | `stack(states)` | 100 histories → one (100, 200, 2) array |
| L130–165 | `compress(K, t)` | per arm: pull counts / N and running mean, (100, 8). Used only for the debug print |
| L167–197 | `to_transformer_input(K, t)` | the network input, below |

**`to_transformer_input`** (L167–197):
- builds a mask `row r is real if r < t` for each bandit (t = rows to show);
- one-hot encodes the arm of each row (4 numbers), zeroes the unreal rows, appends the reward;
- returns `TransformerState(state=(100, 200, 5), timestep=t)`.
- Example row: pulled arm 2, got 1.04 → `[0, 0, 1, 0, 1.04]`. Without the mask, empty rows would read "arm 0, reward 0".

### L199–202: `BanditEnvType`
Allowed bandit names; a string Enum, so `'GaussianBandit'` from the config equals `BanditEnvType.GAUSSIAN_BANDIT`.

### L205–268: `BanditEnv`, one bandit (template class)
- **`__init__` (L213–230):** `means`, `horizon` = 200, `dim` = K, **`num_actions = K + 1`** (STOP = action K), gymnasium spaces (formalities), `current_step = 0`, **`optimal_action = argmax(means)` = H\***, empty history, `stopped = False`.
- **`reset` (L232–235):** pulls back to 0, new empty history.
- **`step(action)` (L237–252)**, Algorithm 1 line 6 (observe x_{t+1}):

| Line | Code | Meaning |
|---|---|---|
| L238 | `assert 0 <= action <= K` | valid action |
| L240–241 | `if self.stopped: return ..., True, True` | already stopped in an earlier round → nothing happens, report done and "stopped" |
| L243 | `self.stopped = action == K` | is this action STOP? |
| L245 | `done = (current_step + 1 >= N) or self.stopped` | the episode ends at the N-th pull (`step` = N−1) or at STOP |
| L246–250 | if not STOP: `reward = self._step(action)`, `Experience(...)`, write it, `current_step += 1` | a real pull: noisy reward into the history |
| L252 | `return state, done, False` | on the STOP round: done = True, stopped = False |

- **L254–268:** `_step` and `parameters` are left to the subclass.

### L270–300: `GaussianBanditEnv`
- **`__init__` (L271–286):** checks one std per arm, none negative; stores `self.std`; **CHANGED:** `self.rng = None` (L286).
- **`seed(seed)` (L288–291), CHANGED (not in the repository):** `self.rng = np.random.RandomState(seed)`: this bandit gets its own random-number generator for its reward noise.
- **`_step(action)` (L293–296):** if `self.rng` is set (CHANGED, L294–295), `means[action] + std[action] * self.rng.normal()`; otherwise, as in the repository, `means[action] + std[action] * np.random.normal()` (L296). Mean + 0.5 × fresh standard-normal noise: **this is what makes the bandit stochastic**.
- **`parameters()` (L298–299):** everything needed to rebuild the bandit (used to make copies).

### L302–346: `VecBanditEnv`, 100 bandits together
- **`__init__` (L306–321):** the list of bandits; `optimal_actions` (100,); `means` (100, 4).
- **`reset` (L324–328):** resets all; returns stacked empty histories and all-False `dones`/`stopped`.
- **`step(actions)` (L331–343):** calls each bandit's `step` with its own action, then regroups the results into states, dones, stopped.

### L352–360: `minimum_gap_sampling(K, min_val, max_val, min_gap)`, the arm means
| Line | Code | Meaning (K = 4, range [0, 1.6], gap 0.25) |
|---|---|---|
| L353 | `assert (K-1)*min_gap <= 1` | 3 gaps × 0.25 = 0.75 must fit on a 0-to-1 ruler |
| L354 | `L = 1 - (K-1)*min_gap` | 0.25 of free room for randomness |
| L355 | `u = np.random.rand(K) * L` | 4 random numbers in [0, 0.25) |
| L358 | `s = np.sort(u)` | sorted, so later gaps can't shrink |
| L359 | `x = s + np.arange(K)*min_gap` | add guaranteed steps 0, 0.25, 0.5, 0.75 → neighbours ≥ 0.25 apart |
| L360 | `np.random.permutation(x) * (max-min) + min` | shuffle which arm is which, stretch to [0, 1.6] → **neighbouring means ≥ 0.4 apart** (each gap between 0.4 and 0.8) |

### L363–378: `create_bandit_env(type, **kwargs)`
Builds a `GaussianBanditEnv` from a parameter dict (used for copies); errors for unknown types.

### L381–404: `sample_env(cfg)`, Algorithm 1 line 4 (M ∼ 𝒫)
| Line | Code | Meaning |
|---|---|---|
| L393 | `if cfg.bandit_env.type == GAUSSIAN_BANDIT:` | the only type here |
| L394–397 | `minimum_gap = cfg.bandit_env.minimum_gap` (or 0 if missing) | gap setting |
| L398 | `means = minimum_gap_sampling(K, 0, 1.6, 0.25)` | the hidden arm means (drawn with the global `np.random`) |
| L399 | `stds = 0.5 * np.ones(K)` | same noise for every arm |
| L401–403 | `return GaussianBanditEnv(means, horizon=200, stds)` | the new bandit |
| L404 | `raise Exception(...)` | unknown type |

### L407–423: `sample_envs(agent, B, N)`
`B` new bandits from `sample_env` (L419), each copied `N` times (L422; training: B = 100, N = 1), wrapped in a `VecBanditEnv`. Number of episodes = B × N.

### L429–441: `sample_actions(agent, state, greedy)`, Algorithm 1 line 6
| Line | Code | Meaning |
|---|---|---|
| L429 | `@torch.inference_mode()` | no gradients: acting, not learning |
| L431–432 | `qvalues = agent.critic.forward(state.state, state.timestep)` (under autocast) | (100, 5) scores |
| L434–435 | `if agent.has_training_started or greedy: actions = qvalues.argmax(-1)` | best-scoring action (an arm or STOP) |
| L436–438 | `else: np.random.choice(K, size=100)` | **warm-up**: random arms, never STOP, so warm-up episodes last N = 200 pulls |
| L441 | `return actions, qvalues, None` | |

### L447–462: `mle(actions, rewards, K)`
Running average reward of each arm after each pull (one-hot × reward, cumulative sums, divided by cumulative counts). Used for logging only.

---

## 4. `icpe_training.py`, the training phase (Algorithm 1, lines 3–12)

### L1–39: docstring and imports
Maps each Algorithm 1 line to its function, and describes the CHANGED episode seeding. `import os` is used to save the episode file.

### L42–53: `compute_mingaps_and_probs(means)` (debug print only)
- `min_gap` = gap between the two largest means of each bandit;
- `gaps_probs` ∝ 1/gap² per arm: a "how hard to tell apart from the best" weight. Printed, not used for learning.

### L185–314: `train(agent, save_path)`, one loop pass = one training step
**L188:** `B, N = 100, 1`.
**L191–192 (CHANGED):** `episode_seeds, episode_means = [], []`, `num_episodes = 0`: lists that collect the seed and the means of every episode.
**L194:** `for step in range(30000):`, **Algorithm 1 line 3**.

**Phase A: new bandits (line 4)**
| Line | Code | Meaning |
|---|---|---|
| L196 | `envs = sample_envs(agent, B=100, N=1)` | 100 new random bandits = 100 episodes |
| L199 (CHANGED) | `seeds = experiment.seed × 100,000,000 + num_episodes + [0, 1, …, 99]` | one seed per episode, unique over the whole run: seed 1 gives 100000000, 100000001, …, 102999999 |
| L200–201 (CHANGED) | `env.seed(int(seed))` for each bandit | each episode's reward noise comes from its own generator seeded with its seed |
| L202–204 (CHANGED) | append `seeds` and `envs.means`; `num_episodes += 100` | record seed and means of each episode |
| L206–207 | `states, dones, stopped = envs.reset()`, `.to_tensor(device)` | 100 empty histories on the device |
| L208 | `true_optimal = means.argmax(-1)` | H\* of each bandit, the secret answers (labels only) |
| L209 | `traj_current_timestep = zeros(100)` | rows to show, per bandit |

**Phase B: play (lines 5–8): `for t in range(200):` (L212)**, one pass = one round
| Line | Code | Meaning |
|---|---|---|
| L214–215 | `seq_state = states.to_transformer_input(K, t)`; `actions, qvalues, probs = sample_actions(agent, seq_state)` | the Chooser picks one action per bandit (**line 6**) |
| L221–226 | `if np.random.rand() < 5e-4:` two extra forward passes | originally saved attention images (code commented out in the repository); kept because the random draw is part of the random-number sequence |
| L229–232 | `next_states, dones, stopped = envs.step(actions.tolist())` → tensors | the bandits respond: noisy rewards, STOP, end at 200 pulls |
| L235–238 | `mle_rewards = (best average == H*) * dones` | logging only: was "the arm with the best average" right, at the end |
| L240–248 | `agent.buffer.add(states=D_t, actions, rewards=mle_rewards, env_rewards, dones, next_states=D_{t+1}, optimal_actions=H*, stopped, timesteps=t)` | **line 7**: one memory per bandit still playing. `next_states` uses mask t + 1, so it includes the new pull. On a STOP transition no pull was added, so its extra unmasked row is an empty row, read as `[1, 0, 0, 0, 0]` (as in the repository) |
| L250–251 | `states = next_states`; `t += 1 - stopped` | move on; bandits that stopped in an earlier round stop counting |
| L253–254 | `if torch.all(dones): break` | all 100 finished |

**Phase C: learn (lines 10–11)**
| Line | Code | Meaning |
|---|---|---|
| L256 | `eloss, qloss = train_networks(agent)` | 5 rounds of updates, including the ϕ̄ and θ̄ updates (section 7) |

**Phase D: λ (line 9)**
| Line | Code | Meaning |
|---|---|---|
| L258–262 | `log_probs = agent.i_net.forward(final histories)`, and the same for ϕ̄ (`op`, `op_tgt` for printing) | the just-updated Guesser's answers on the 100 finished episodes |
| L265 | `cost_updates = update_cost(agent, log_probs, true_optimal)` | adjust λ (below) |

**Phase E: report, decay, save**
| Line | Code | Meaning |
|---|---|---|
| L268–288 | `compress(...)`, `compute_mingaps_and_probs(...)`, prints | debug block for bandit 0: counts, probs, gaps, observed and true means, Q-values, I_ϕ and ϕ̄ probabilities |
| L291–294 | `learning_rates = {...}` | current learning rates, for logging |
| L297–306 | `log_statistics(...)` | prints `Step n completed …`, logs everything to wandb |
| L307 | `agent.current_step += 1` | one more training step |
| L309 | `agent.scheduler_step()` | learning rates decay a little |
| L310–311 | `if current_step % 1000 == 0: save(agent, save_path)` | checkpoint |
| L312–314 (CHANGED) | `np.savez(save_path/episodes_<seed>.npz, seeds=..., means=...)` | the seed (int64) and K means (float64) of **every** episode so far, in episode order (episode e was played in training step e // 100) |

### L82–98: `train_networks(agent)`, Algorithm 1 line 10
| Line | Code | Meaning |
|---|---|---|
| L86 | `if len(buffer) > 20 × 1024 × 5:` | learning only after 102,400 memories |
| L87 | `for _ in range(5):` | 5 update rounds |
| L88 | `batch = agent.buffer.sample(1024, ...)` | 1,024 random memories |
| L89–91 | `eloss = train_estimation_net(agent, batch)` | Guesser update (and possibly a ϕ̄ copy) |
| L94–96 | `if len(buffer) > 50 × 1024 × 5: rlloss = train_rl_nets(agent, batch)`, `gradient_step += 1` | Chooser update (and θ̄ update) on the **same** batch, once there are 256,000 memories |
| L98 | `return eloss, rlloss` | for logging |

When this starts with N = 200: warm-up episodes are all 200 pulls (random arms, never STOP), so each warm-up step stores 100 × 200 = 20,000 memories. The Guesser starts training in about the 6th training step (> 102,400) and the Chooser, and greedy actions, in about the 13th (> 256,000). With N = 100 these were about the 11th and 26th.

### L56–79: `train_estimation_net(agent, batch)`, eq. (7) and the ϕ̄ copies
| Line | Code | Meaning |
|---|---|---|
| L58 | `agent.i_net.train()` | training mode (no dropout anyway) |
| L60–63 | `log_probs = i_net(batch.next_states, ...).gather(-1, H*)` | log I_ϕ(H\* \| 𝒟_{t+1}) for each of the 1,024 memories |
| L64 | `est_loss = -log_probs.mean()` | **eq. 7**: cross-entropy, the average "surprise" at the true answer |
| L66 | `zero_grad()` | clear old gradients |
| L68 | `i_scaler.scale(est_loss).backward()` | gradients (scaled for mixed precision on GPU) |
| L69 | `i_scaler.unscale_(i_optimizer)` | undo the scaling before clipping |
| L70–71 | `clip_grad_norm_(..., 1)` | cap the gradient size at 1 |
| L72–73 | `i_scaler.step(...)`, `i_scaler.update()` | AdamW step (skipped if the gradients overflowed), adjust the scale |
| L76 | `e_gradient_step += 1` | count I_ϕ updates |
| L77–78 | `if e_gradient_step % (2 × 100) == 0: i_tgt_net.hard_update(i_net)` | **line 11**: ϕ̄ ← ϕ every 200 I_ϕ updates (section 7). **CHANGED:** the repository multiplies by `environment.horizon` (2 × N); here it is fixed at 2 × 100 = 200, the value of the authors' N = 100 setting, so that N = 200 only changes the episode limit |
| L79 | `return {'e_loss': ...}` | |

### L101–115: `update_cost(agent, log_probs, true_optimal)`, λ (Algorithm 1 line 9)
| Line | Code | Meaning |
|---|---|---|
| L103 | `estm_rewards_argmax = (argmax I_ϕ == H*)` | right or wrong, for each of this step's **100** final histories |
| L106 | `err = 0.9 - accuracy` | > 0: not accurate enough; < 0: more accurate than needed |
| L108 | `cost_loss = λ * err.detach()` | its gradient with respect to λ is `err` |
| L109–111 | `zero_grad()`, `backward()`, `cost_optimizer.step()` | AdamW moves λ **against** the recent average of `err`: too inaccurate → λ down (cheaper pulls), too accurate → λ up |
| L112–113 | `λ.clamp_(0, 1)` | keep λ in [0, 1] |
| L115 | `return {'cost': λ, 'target_confidence_error': |err|}` | logged |

This is the paper's implementation form (arXiv app. C.1): λ here is the cost c = 1/λ_paper, updated as c ← c − β(1 − δ − p̂).

### L118–182: `log_statistics(...)`
Computes and logs: Guesser accuracy and confidence on the true arm, best-average accuracy (printed as `Final Action Correct`), Q-value statistics, the losses, λ, learning rates, average and median episode length. `exploration_probs` is `None`, so the `policy/*` block is skipped.

---

## 5. `icpe_q_learning.py`, the Chooser update (eqs. 9–10) and θ̄

### L19–68: `train_rl_nets(agent, batch)`
**Target, without gradients (L22–39):**
| Line | Code | Meaning |
|---|---|---|
| L24 | `next_timesteps = t + (1 - done)` | timestep of the next history (passed to the networks, unused by them) |
| L25 | `next_act = critic(next_states).argmax(-1)` | the **online** Q_θ picks the best next action a′ |
| L26 | `next_q = critic_tgt(next_states).gather(-1, next_act)` | the **target** θ̄ scores it: Q_θ̄(𝒟_{t+1}, a′) (Double DQN) |
| L28–29 | oracle branch | not used (False) |
| L32–33 | `rewards_next = i_tgt_net(next_states).gather(-1, H*)` | **log I_ϕ̄(H\* \| 𝒟_{t+1})**: the target Guesser's log-confidence in the true arm (the `.exp()` is commented out in the repository, so it stays a log) |
| L35 | `cost = -(1 - done) * λ` | **−λ for every transition that doesn't end the episode** |
| L37–39 | `tgt_val = cost + done·rewards_next + 1·(1 - done)·next_q` | the target: if the episode ends, the log-confidence; otherwise pay λ and add the next move's value |

**Prediction and loss (L43–54):**
| Line | Code | Meaning |
|---|---|---|
| L45–46 | `qvalues = critic(states)`; `.gather(-1, actions)` | Q_θ(𝒟_t, a_t) for the action actually taken |
| L48 | `qloss = mse(Q_θ(𝒟_t, a_t), target)` | **eq. 9**: make the prediction match the target |
| L51 | `idxs = actions != STOP` | memories where the agent did not stop |
| L52 | `qnext = critic(next_states[idxs])[..., -1]` | Q_θ(𝒟_{t+1}, STOP) |
| L53–54 | `qloss += mse(qnext, rewards_next[idxs])` | **eq. 10, "pretend to stop"**: stopping now would be worth the target Guesser's log-confidence now |

**Update (L56–68):**
| Line | Code | Meaning |
|---|---|---|
| L56–62 | zero_grad → scaled backward → unscale → clip to 1 → step → update scaler | one AdamW step on Q_θ |
| L66 | `critic_tgt.soft_update(critic, 0.01)` | **line 11**: θ̄ ← 0.01·θ + 0.99·θ̄ after every update (section 7) |
| L68 | `return {'value_loss': ...}` | |

---

## 6. Inference (Algorithm 1, lines 13–15)

At inference nothing is trained: no buffer, no λ update, no target copies. The trained Chooser Q_θ plays **greedily** (always argmax, no random actions), and the trained Guesser I_ϕ gives the answer.

```
line 13  draw a bandit M (never seen in training)
line 14  t = 0, empty history
         repeat:  a_t = argmax_a Q_θ(𝒟_t, a)          (4 arms + STOP)
                  if a_t = STOP → stop;  else pull arm a_t, observe reward, 𝒟_{t+1} = 𝒟_t + (a_t, x)
         until STOP or N pulls
line 15  answer Ĥ = argmax_k I_ϕ(k | final history);  correct if Ĥ = H*
```

### 6.1 `icpe_inference.py`, the repository's version (many bandits at once)

#### L33–93: `HistoryData`
The results of a test: arms pulled, rewards, I_ϕ's guesses and probabilities at each round, correctness, exploration distribution, **`stopping_time`**, all shaped (B, copies, N[, K]).
- **`to_numpy` (L50–67):** converts everything to numpy.
- **`statistics` (L69–93):** accuracy per test bandit and overall (from the last time slot), and an exploration table.

#### L96–167: `HistoryBuilder`
- **`__init__` (L97–123):** pre-allocates the result tensors.
- **`record_exploration_dist(h, dist)` (L126–129):** stores the softmax of the arm Q-values at round h (statistics only).
- **`build(actions, rewards, probs, means, stopping_time)` (L131–167):**
  - **L147:** `opt_actions = probs.argmax(-1)`, the Guesser's answer Ĥ at each round (**line 15**);
  - **L152:** `rew_opt_actions = (Ĥ == true best)`, i.e. correct or not;
  - **L154:** stores the stopping times;
  - **L155–167:** reshapes into a `HistoryData`.

#### L173–233: `sample_trajectories(agent, B, N)`, lines 13–15
| Line | Code | Meaning |
|---|---|---|
| L185 | `envs = sample_envs(agent, B, N)` | **line 13**: B new bandits, each copied N times = B × N episodes (the reward noise uses the global `np.random`; no per-episode seed) |
| L187–197 | builder, empty histories, H\*, `t = 0`, space for I_ϕ outputs | setup |
| L199 | `for h in range(N):` | **line 14**, one round per pass |
| L200–201 | `sample_actions(agent, seq_state, greedy=True)` | always argmax, no random actions |
| L202–204 | `log_probabilities[:, h] = i_net(history)` | the Guesser's belief before this round's action |
| L207–208 | `softmax(qvalues)[..., :K]` → `record_exploration_dist` | statistics only |
| L211–219 | `envs.step(...)`, `t += 1 - stopped` | play; `t` also counts the round on which STOP is chosen |
| L220–221 | `break` when all done | |
| L224–228 | `log_probabilities[:, -1] = i_net(final history)` | the Guesser's belief on the final history, stored in the last slot |
| L231 | `traj_builder.build(..., stopping_time=t)` | **line 15**: Ĥ = argmax I_ϕ, correctness and stopping time |

The stopping time here is `traj_current_timestep`: the number of pulls, **plus one when the episode ended with STOP** (the STOP round is counted), or N if it never stopped. The authors' notebook (and `plot_tau_vs_log_delta.py`) takes correctness at index `stopping_time − 1`, with B = 300 bandits × N = 15 episodes each.

#### L236–262: `evaluate(agent, B, N)`
Runs `sample_trajectories`, computes `statistics()`, prints one random test bandit's means, best-average estimate and exploration table. Not called by `main.py`.

#### L269–287: `save` / `load`
- `save` (L269–276): pickles `{'cfg', 'critic', 'Enet', 'step'}` to `icpe_<step>_<seed>.pkl`. Only Q_θ and I_ϕ are saved: no buffer, no optimizers, no λ, no target copies.
- `load` (L279–287): builds a new `ICPEAgent` from the saved config (so the model's own N and K), loads the weights into both networks **and** their target copies (θ̄ = θ, ϕ̄ = ϕ; the targets are not used at inference).

### 6.2 `inference_fixed_env.py`, **new**: one fixed bandit, one seeded trajectory at a time

Not from the repository. It runs exactly the rollout and answer of `sample_trajectories` (same calls: `to_transformer_input`, `sample_actions(..., greedy=True)`, `envs.step`, `i_net.forward` on the final history), but for **one fixed bandit** and **one trajectory at a time**, so that every trajectory has its own seed and its own wall-clock time.

#### L101–155: `main()`
| Line | Code | Meaning |
|---|---|---|
| L103–107 | options `--model`, `--env_seed` (0), `--num_traj` (5000), `--seed_start` (1), `--out` | which model, which fixed bandit, how many trajectories, first seed, output folder |
| L110–112 | `device`, `agent = load(args.model, device)`, `cfg = agent.cfg` | the trained model with its own config (K, N, δ) |
| L115–117 | `np.random.seed(env_seed)`; `env0 = sample_env(cfg.environment)`; `means, stds = ...` | **the fixed bandit**: drawn once from the training distribution 𝒫 (`minimum_gap_sampling`), reproducible from `env_seed`. K = 4, `env_seed` 0 → means (1.0411, 0.2180, 1.4861, 0.6195), best arm 2 |
| L119–133 | output folder (default `inference_env<env_seed>/` next to the model) and `environment.json` | means, stds, best arm, `env_seed`, K, N, confidence level, model path, number of trajectories, seed range, device |
| L136–141 | GPU warm-up: one Q_θ and one I_ϕ forward pass on zeros | CUDA initialisation, so that it is not counted in the first trajectory's time; uses no random numbers |
| L143–150 | `for i in 1..num_traj: run_trajectory(agent, means, stds, seed_start + i - 1, device)` | trajectory i uses seed i (1 … 5000); progress printed every 500 |
| L152–154 | `trajectories.csv` | one row per trajectory (columns below) |

#### L49–98: `run_trajectory(agent, means, stds, seed, device)`, lines 14–15 for one trajectory
| Line | Code | Meaning |
|---|---|---|
| L52–53 | `np.random.seed(seed)`, `torch.manual_seed(seed)` | fix the global seeds for this trajectory |
| L57–58 | `sync(device)`; `t0 = time.perf_counter()` | start the clock (GPU synchronised first) |
| L60–62 | `env = GaussianBanditEnv(horizon=N, means, stds)`; `env.seed(seed)`; `VecBanditEnv([env])` | a fresh copy of the fixed bandit whose reward noise comes from a generator seeded with `seed`, so the same seed gives the same rewards |
| L63–65 | `envs.reset()`, `t = 0` | empty history |
| L67 | `stop_step = -1` | "has not stopped" |
| L68 | `for h in range(N):` | **line 14**, one round per pass |
| L69–70 | `to_transformer_input`; `sample_actions(agent, seq_state, greedy=True)` | Q_θ picks argmax action |
| L72–73 | `if action == STOP: stop_step = h + 1` | record the round (1-based) in which STOP was taken |
| L75–81 | `envs.step(...)`, `t += 1 - stopped` | pull the arm (or stop), as in `sample_trajectories` |
| L82–83 | `if torch.all(dones): break` | STOP taken or N pulls reached |
| L86–88 | `log_probs = i_net(final history)`; `predicted = argmax` | **line 15**: Ĥ = argmax I_ϕ (same final history as `sample_trajectories`, including the extra empty row after a STOP) |
| L90–91 | `sync(device)`; `wall_ms = (perf_counter() − t0) × 1000` | stop the clock: the time covers creating the bandit, the whole rollout and the final guess |
| L93–98 | return the row | see below |

**`trajectories.csv` columns:**

| Column | Meaning |
|---|---|
| `traj_id` | 1 … 5000 |
| `seed` | seed of the trajectory (= `traj_id` with the defaults) |
| `stop_step` | round (1-based) in which STOP was taken = `num_pulls` + 1; −1 if the trajectory reached N pulls without STOP. Equals the repository's `stopping_time` when STOP was taken |
| `num_pulls` | number of arm pulls (= N if it never stopped) |
| `stopped` | 1 if STOP was taken, 0 if it reached N pulls |
| `wall_ms` | wall-clock time of the trajectory in milliseconds |
| `correct` | 1 if Ĥ = argmax I_ϕ is the best arm, else 0 |

Reproducibility: running the script twice with the same model, `env_seed` and seeds gives identical rows except `wall_ms` (checked on a test model).

### 6.3 `plot_tau_vs_log_delta.py`, **new**
Loads one model per δ, runs `sample_trajectories(agent, 300, 15)` (the notebook's sizes), takes τ = `stopping_time` and correctness at index τ − 1, and plots E[τ] and accuracy against log(1/δ) with 95% intervals over the 300 per-bandit averages.

---

## 7. The target copies θ̄ and ϕ̄ in detail (Algorithm 1, lines 2 and 11)

**Why they exist.** The Chooser learns from targets that are built from network outputs: the Guesser's log-confidence (reward) and the Chooser's own value of the next history. If those came from the networks being trained, the target would change at every update and training would chase a moving target. So each network has a **target copy** that changes slowly, and only the copies are used to build targets. In the paper's Algorithm 1 these are "every T_ϕ steps set ϕ̄ ← ϕ (similarly, every T_θ steps set θ̄ ← θ)"; the paper names T_ϕ and T_θ but gives no values for this experiment; the values below are the authors' code.

**Where each copy is created, used and updated:**

| | ϕ̄ (`i_tgt_net`), target Guesser | θ̄ (`critic_tgt`), target Chooser |
|---|---|---|
| Created | setup L555–557: new `InferenceTransformer`, then `hard_update(i_net)`: ϕ̄ = ϕ | setup L550–552: new `CriticTransformer`, then `hard_update(critic)`: θ̄ = θ |
| Has an optimizer? | No: changes only by copying | No: changes only by averaging |
| Used for | the reward: `rewards_next = log I_ϕ̄(H* \| 𝒟_{t+1})` (q_learning L32–33), in the target of eq. 9 (L39) and the "pretend to stop" target of eq. 10 (L53); also printed as `OP tgt` (training L262) | the next value in the target: `next_q = Q_θ̄(𝒟_{t+1}, a′)` with a′ chosen by Q_θ (q_learning L25–26, Double DQN) |
| Not used for | the Guesser's own loss (eq. 7 uses I_ϕ) and the λ update (uses I_ϕ) | choosing actions (uses Q_θ) |
| Update rule | **hard copy**: ϕ̄ ← ϕ (setup L162–164) | **soft (Polyak) update**: θ̄ ← 0.01·θ + 0.99·θ̄ (setup L170–173) |
| When | training L76–78: after every I_ϕ update, `e_gradient_step += 1`; when it is a multiple of `frequency_update_e_network × 100` = **200** (CHANGED: the repository uses × N) | q_learning L66: after **every** Q_θ update |
| Config key | `frequency_update_e_network` = 2 (× 100) | `polyak_update_q_network` = 0.01 |

**Counting.** Each training step has 5 update rounds (`replay_ratio`), each with one I_ϕ update and (after its warm-up) one Q_θ update. So:

| | N = 100 (the runs in `runs_stochastic_delta/`, `runs_stochastic_K/`) | N = 200 (current default, `runs_N200/`) |
|---|---|---|
| T_ϕ in I_ϕ updates | **200** | **200** (fixed, same as N = 100) |
| T_ϕ in training steps | 200 / 5 = **every 40 steps** | 200 / 5 = **every 40 steps** |
| I_ϕ updates start | about step 11 (buffer > 102,400) | about step 6 |
| First ϕ̄ copy | about step 11 + 40 = 51 | about step 6 + 40 = 46 |
| ϕ̄ copies in 30,000 steps | about 750 | about 750 |
| θ̄ updates per step | 5 (Polyak 0.01 each) | 5 (Polyak 0.01 each) |
| Q_θ updates start | about step 26 (buffer > 256,000) | about step 13 |
| θ̄ updates in 30,000 steps | about 150,000 | about 150,000 |

**What "soft" means for θ̄.** After n Polyak updates, the weight of a value of θ from n updates ago in θ̄ is 0.01 × 0.99ⁿ: θ̄ is an exponentially weighted average of past θ, mostly the last ~100 updates (~20 training steps). There is no exact T_θ; θ̄ moves a little after every Q_θ update.

**At inference** the target copies play no role: `load` sets them equal to the trained networks, and only Q_θ (actions) and I_ϕ (answer) are used.

---

## One training step, end to end (N = 200)

```
A  100 new bandits (means ≥ 0.4 apart on [0, 0.4K], noise 0.5); episode seeds recorded, reward noise seeded
B  for t = 0..199:  history → Q_θ → action (random arms during warm-up) → bandits respond → store memory
C  if buffer > 102,400:  5 × [ batch of 1,024 → I_ϕ update (eq. 7); if buffer > 256,000 → Q_θ update (eqs. 9–10) ]
       ϕ̄ ← ϕ every 200 I_ϕ updates (40 steps);  θ̄ ← 1% toward θ after every Q_θ update
D  Guesser's accuracy on the 100 episodes vs 0.9 → λ up or down (lr 3·10⁻⁵, λ ∈ [0, 1])
E  print, log, decay learning rate (3·10⁻⁶ → 1·10⁻⁶ over 15,000 steps),
   checkpoint + episodes_<seed>.npz every 1,000 steps
```
Repeated 30,000 times. The algorithm code (before the CHANGED parts) was checked against the authors' own code: 28 training steps with the same settings and seed gave identical weights and identical printed output.
