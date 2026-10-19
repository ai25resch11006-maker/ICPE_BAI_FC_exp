"""
ICPE - sampling (Algorithm 1, lines 4 and 6).

Line 4: sample environment M ~ P with hypothesis H* (sample_env, sample_envs).
Line 6: execute action a_t = argmax_a Q_theta(D_t, a) in M and observe x_{t+1}
        (sample_actions, BanditEnv.step, VecBanditEnv.step).
The history D_t is the State class; State.to_transformer_input is the network input.
Fixed-confidence setting only: every bandit has K arms plus the stop action a_stop (index K).
Stochastic Gaussian bandits: rewards N(mu_a, std^2), means drawn with minimum_gap_sampling.

All code is taken from the neurips_v2 branch of the ICPE repository
(https://github.com/rssalessio/icpe, branch neurips_v2, commit 7833a2f):
  icpe/envs/bandit/state.py                             -> Experience, State
  icpe/envs/bandit/config.py                            -> BanditEnvType
  icpe/envs/bandit/base.py                              -> BanditEnv, VecBanditEnv
  icpe/envs/bandit/gaussian_bandit.py                   -> GaussianBanditEnv
  icpe/envs/bandit/builder.py                           -> create_bandit_env, sample_env
  icpe/envs/bandit/utils.py                             -> minimum_gap_sampling
  icpe/agents/bandit/sequential/base_offpolicy_agent.py -> sample_envs  (BaseOffPolicyAgent.sample_envs)
  icpe/agents/bandit/sequential/icpe.py                 -> sample_actions (ICPE._sample_actions)
  icpe/agents/bandit/utils.py                           -> mle
Only the Gaussian bandit used by the ICPE bandit experiments is included. The characteristic-time
methods (used by the TaS/TTPS baselines, not by ICPE) and the magic-action branch of sample_env
(num_magic_actions > 0, not used by the stochastic experiment) are left out.
"""
from __future__ import annotations
import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from enum import Enum
from numpy.typing import NDArray
from typing import Generic, TypeVar, Tuple, NamedTuple, List
from icpe_setup import TransformerState


# ---------------------------------------------------------------------------
# History D_t                                  [icpe/envs/bandit/state.py]
# ---------------------------------------------------------------------------
class Experience(NamedTuple):
    step: int
    reward: float
    action: int
    done: bool


T = TypeVar("T", np.ndarray, torch.Tensor)

class State(Generic[T]):
    def __init__(self, state: T, num_actions: int = 1, num_rewards: int = 1):
        """Initialize network state that tracks action vectors and reward vectors

        Args:
            state: Tensor of shape (..., H, 2*reward_dim) where:
                  - First num_actions channels contain one-hot action vectors
                  - Last num_actions channels contain reward vectors
            num_actions: Number of actions taken per round
            num_rewards: Number of rewards observed per round
        """
        self.state = state
        self.num_actions = num_actions
        self.num_rewards = num_rewards

    @staticmethod
    def zeros_numpy(shape: tuple[int], num_actions: int = 1, num_rewards: int = 1) -> State[np.ndarray]:
        """Create a state with zeros

        Args:
            shape: Shape of the state (typically just horizon H)
            num_actions (int): Number of actions taken per round
            num_rewards (int): number of rewards observed per round
        """
        if isinstance(shape, int):
            shape = (shape, )
        return State(
            np.zeros(tuple(shape) + (num_actions + num_rewards,), dtype=np.float64, order='C'),
            num_actions, num_rewards)

    @staticmethod
    def zeros_torch(shape: Tuple[int] | int, num_actions: int = 1, num_rewards: int = 1, device: torch.device = torch.device('cpu')) -> State[torch.Tensor]:
        """Create a state with zeros

        Args:
            shape: Shape of the state (typically just horizon H)
            num_actions (int): Number of actions taken per round
            num_rewards (int): umber of rewards observed per round
            device: Torch device to place tensor on
        """
        if isinstance(shape, int):
            shape = (shape, )
        return State(
            torch.zeros(tuple(shape) + (num_actions + num_rewards,), dtype=torch.float32, device=device),
            num_actions, num_rewards)


    @property
    def actions(self) -> T:
        if isinstance(self.state, np.ndarray):
            return self.state[..., :self.num_actions].astype(np.ulong).squeeze(-1)
        elif isinstance(self.state, torch.Tensor):
            return self.state[..., :self.num_actions].type(torch.long).squeeze(-1)
        raise Exception('Type not found')

    @property
    def rewards(self) -> T:
        return self.state[..., -self.num_rewards:].squeeze(-1)

    def add(self, step: Experience) -> State:
        self.state[..., step.step, :self.num_actions] = step.action
        self.state[..., step.step, -self.num_rewards:] = step.reward
        return self

    def to_tensor(self, device: torch.device = torch.device('cpu')) -> State[torch.Tensor]:
        return State(torch.from_numpy(self.state).to(device).type(torch.float32), self.num_actions, self.num_rewards)

    @staticmethod
    def stack(states: List[State[T]]) -> State[T]:
        if isinstance(states[0].state, np.ndarray):
            return State(np.stack([states[i].state for i in range(len(states))]),
                         num_actions=states[0].num_actions,
                         num_rewards=states[0].num_rewards)
        elif isinstance(states[0].state, torch.Tensor):
            return State(torch.stack([states[i].state for i in range(len(states))]),
                         num_actions=states[0].num_actions,
                         num_rewards=states[0].num_rewards)
        raise Exception('Type not found')


    @torch.no_grad()
    def compress(self, num_actions: int, timestep: torch.Tensor | int):
        if not isinstance(self.state, torch.Tensor):
            raise Exception('Not implemented')


        batch_size, horizon = self.rewards.shape[0], self.rewards.shape[1]

        timestep_mask = torch.arange(horizon, device=self.rewards.device).unsqueeze(0).expand(batch_size,-1)

        if isinstance(timestep, torch.Tensor):
            timestep_mask = timestep_mask < timestep.unsqueeze(-1)
        else:
            timestep_mask = timestep_mask < timestep
        actions = torch.nn.functional.one_hot(self.actions, num_actions)

        # Mask actions and rewards
        masked_actions = actions * timestep_mask.unsqueeze(-1)  # [B, H, num_actions]

        if self.num_rewards == 1:
            masked_rewards = self.rewards * timestep_mask  # [B, H]
            rewards_expanded = masked_rewards.unsqueeze(-1) * masked_actions  # [B, num_actions]
        else:
            rewards_expanded = self.rewards * timestep_mask.unsqueeze(-1)  # [B, H]

        # Compute cumulative statistics for all timesteps
        counts = torch.sum(masked_actions, dim=1)  # [B, num_actions]

        total_rewards = torch.sum(rewards_expanded, dim=1)  # [B, num_actions]
        means = total_rewards / torch.clamp(counts, min=1.0)  # [B, num_actions]

        # For each timestep, concatenate its statistics
        return torch.cat([
            counts / horizon,  # Running counts up to each timestep
            means,   # Running means up to each timestep
        ], dim=-1)  # [B, 2*num_actions]

    @torch.no_grad()
    def to_transformer_input(self, num_actions: int, timestep: torch.Tensor | int) -> TransformerState:


        batch_size, horizon = self.rewards.shape[0], self.rewards.shape[1]

        timestep_mask = torch.arange(horizon, device=self.rewards.device).unsqueeze(0).expand(batch_size,-1)

        if isinstance(timestep, torch.Tensor):
            timestep_mask = timestep_mask < timestep.unsqueeze(-1)
        else:
            timestep_mask = timestep_mask < timestep
        actions = torch.nn.functional.one_hot(self.actions, num_actions)


        masked_actions = actions * timestep_mask.unsqueeze(-1)

        if self.num_rewards == 1:
            masked_rewards = self.rewards * timestep_mask
            return TransformerState(
                state=torch.cat([masked_actions, masked_rewards.unsqueeze(-1)], dim=-1),
                timestep=timestep)
        else:
            masked_rewards = self.rewards * timestep_mask.unsqueeze(-1)
            return TransformerState(
                state=torch.cat([masked_actions, masked_rewards], dim=-1),
                timestep=timestep)


# ---------------------------------------------------------------------------
# Environment M                                [icpe/envs/bandit/config.py, base.py, gaussian_bandit.py]
# ---------------------------------------------------------------------------
class BanditEnvType(str, Enum):
    GAUSSIAN_BANDIT = 'GaussianBandit'
    BINARY_BANDIT = 'BinaryBandit'
    BERNOULLI_BANDIT = 'BernoulliBandit'


class BanditEnv(gymnasium.Env, ABC):
    optimal_action: int
    means: NDArray[np.float64]
    horizon: int
    current_state: State
    env_type: str = None
    num_actions: int

    def __init__(self, horizon: int, means: NDArray[np.float64]):
        """Create an abstract bandit environment

        Args:
            means (NDArray[np.float64]): vector of average rewards
            horizon (int): maximum episode length
        """
        super().__init__()
        self.means = means
        self.horizon = horizon
        self.dim = len(means)
        self.num_actions = self.dim + 1    # K arms + a_stop
        self.observation_space = gymnasium.spaces.Box(low=1, high=1, shape=(1,))
        self.action_space = gymnasium.spaces.Discrete(n = self.dim, start = 0)
        self.current_step = 0
        self.optimal_action = np.argmax(means).astype(np.longlong)
        self.current_state = State.zeros_numpy(self.horizon)
        self.stopped = False

    def reset(self) -> Tuple[State, bool, bool]:
        self.current_step = 0
        self.current_state = State.zeros_numpy(self.horizon)
        return self.current_state, False, False

    def step(self, action: int) -> Tuple[State, bool, bool]:
        assert 0 <= action <= self.num_actions - 1, f"Action {action} needs to be in 0,1,..,{self.num_actions-1}"

        if self.stopped:
            return self.current_state, True, True

        self.stopped = action == self.num_actions - 1

        done = (self.current_step + 1 >= self.horizon) or self.stopped
        if not self.stopped:
            reward = self._step(action)
            exp = Experience(self.current_step, reward, action, done)
            self.current_state.add(exp)
            self.current_step += 1

        return self.current_state, done, False

    @abstractmethod
    def _step(self, action: int) -> float:
        """Computes the reward given an action

        Args:
            action (int): action taken
        Returns:
            float: reward
        """
        raise Exception('Method _step not implemented!')

    @abstractmethod
    def parameters(self) -> dict[int | float| np.ndarray]:
        raise Exception('Not implemeneted')


class GaussianBanditEnv(BanditEnv):
    def __init__(self, horizon: int, means: NDArray[np.float64],  stds: NDArray[np.float64]):
        """Creates a gaussian bandit model

        Args:
            horizon (int): maximum episode length
            means (NDArray[np.float64]): average rewards of the arms
        """
        assert len(stds) == len(means), "Size of the standard deviation vector is not the same as the number of arms"


        super().__init__(horizon=horizon, means=means)

        self.std = stds
        assert np.any(self.std < 0) == False, 'Standard deviation needs to be non-negative'
        self.env_type = BanditEnvType.GAUSSIAN_BANDIT
        self.rng = None     # CHANGED (not in the repository): see seed()

    def seed(self, seed: int):
        """CHANGED (not in the repository): draw this environment's reward noise from its own
        generator, seeded with `seed`, instead of the global np.random."""
        self.rng = np.random.RandomState(seed)

    def _step(self, action: int) -> float:
        if self.rng is not None:
            return self.means[action] + self.std[action] * self.rng.normal()
        return self.means[action] + self.std[action] * np.random.normal()

    def parameters(self) -> dict[int | np.ndarray]:
        return {'horizon': self.horizon, 'means': self.means, 'stds': self.std}


class VecBanditEnv(object):
    num_actions: int
    horizon: int

    def __init__(self, envs: List[BanditEnv]):
        """
        Create a batched bandit environment.

        Args:
            envs (List[BanditEnv]): List of individual bandit environments.
        """

        super().__init__()
        self.envs = envs
        self.batch_size = len(envs)
        self.optimal_actions = np.stack([env.optimal_action for env in envs]).astype(np.longlong)
        self.means = np.stack([env.means for env in envs]).astype(np.float64)

        ## We Assume the environments are homogeneous!!
        self.num_actions = envs[0].num_actions
        self.horizon = envs[0].horizon

    def reset(self) -> Tuple[State, bool, bool]:
        """
        Reset all sub-environments and return the batched initial observations.
        """
        obs = State.stack([env.reset()[0] for env in self.envs])
        return obs, np.zeros(self.batch_size, dtype=np.bool_),  np.zeros(self.batch_size, dtype=np.bool_)

    def step(self, actions: List[int]) -> Tuple[State, np.ndarray, np.ndarray]:
        """
        Take a list of actions, one for each sub-environment, and perform a step in each.

        Args:
            actions (List[int]): List of actions, one for each sub-environment.

        Returns:
            Tuple containing batched states and termination flags
        """
        new_states, dones, stopped = zip(*map(lambda env, act: env.step(act), self.envs, actions))
        new_states = State.stack(new_states)
        return new_states, np.array(dones), np.array(stopped)

    def parameters(self) -> List[dict[int | float| np.ndarray]]:
        return [env.parameters() for env in self.envs]


# ---------------------------------------------------------------------------
# Line 4: task distribution P                  [icpe/envs/bandit/utils.py, icpe/envs/bandit/builder.py]
# ---------------------------------------------------------------------------
def minimum_gap_sampling(num_actions: int, min_val: float, max_val: float, min_gap: float):
    assert (num_actions - 1) * min_gap <= 1, 'Minimum gap is not smaller than 1/(N-1)'
    L = (1 - (num_actions-1) * min_gap)
    u = np.random.rand(num_actions) * L


    s = np.sort(u)
    x = s+np.arange(num_actions) * min_gap
    return np.random.permutation(x) * (max_val - min_val) + min_val


def create_bandit_env(type: BanditEnvType, **kwargs) -> BanditEnv:
    """Sample a random environment according to the configuration

    Args:
        type (BanditEnvType): bandit env type
        params (dict): parameters of the environment

    Raises:
        Exception: If the type of environment is not found

    Returns:
        BanditEnv: The random bandit environment
    """
    if type == BanditEnvType.GAUSSIAN_BANDIT:
        return GaussianBanditEnv(**kwargs)
    raise Exception(f'Environment type {type} not implemented!')


def sample_env(cfg) -> BanditEnv:
    """Sample a random environment according to the configuration

    Args:
        cfg (BanditEnvConfig): bandit env configuration

    Raises:
        Exception: If the type of environment is not found

    Returns:
        BanditEnv: The random bandit environment
    """
    if cfg.bandit_env.type == BanditEnvType.GAUSSIAN_BANDIT:
        if 'minimum_gap' in cfg.bandit_env:
            minimum_gap = cfg.bandit_env.minimum_gap
        else:
            minimum_gap= 0
        means = minimum_gap_sampling(cfg.num_actions, cfg.bandit_env.min_value, cfg.bandit_env.max_value, minimum_gap)
        stds = cfg.bandit_env.std * np.ones(cfg.num_actions)

        return GaussianBanditEnv(means=means,
                                 horizon=cfg.horizon,
                                 stds=stds)
    raise Exception(f'Environment type {cfg.bandit_env.type} not implemented!')


def sample_envs(agent, B: int, N: int) -> VecBanditEnv:
    """Generate B*N environments. B environments are sampled at random, and then
       each env is repeated N times.

    Args:
        B (int): Number of environments to sample
        N (int): Number of repetitions per environment

    Returns:
        BatchedBanditEnv: A vectorized environment
    """
    # Sample B environment
    envs = [sample_env(agent.cfg.environment) for _ in range(B)]

    # Repeat each env N times
    repeated_envs = [create_bandit_env(env.env_type, **env.parameters()) for env in envs for _ in range(N)]
    return VecBanditEnv(repeated_envs)


# ---------------------------------------------------------------------------
# Line 6: a_t = argmax_a Q_theta(D_t, a)       [icpe/agents/bandit/sequential/icpe.py]
# ---------------------------------------------------------------------------
@torch.inference_mode()
def sample_actions(agent, state: TransformerState, greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.amp.autocast(agent.device, enabled=agent.use_amp):
        qvalues = agent.critic.forward(state.state, state.timestep)

    if agent.has_training_started or greedy:
        actions = qvalues.argmax(-1)
    else:
        actions = np.random.choice(agent.cfg.environment.num_actions, size=state.state.shape[0])
        actions = torch.from_numpy(actions).to(agent.device)


    return actions, qvalues, None


# ---------------------------------------------------------------------------
# Empirical means of the arms                  [icpe/agents/bandit/utils.py]
# ---------------------------------------------------------------------------
@torch.no_grad()
def mle(actions: torch.Tensor, rewards: torch.Tensor, K: int) -> torch.Tensor:
    """Compute the MLE of the average rewards

    Args:
        actions (torch.Tensor): size (B, N, H)
        rewards (torch.Tensor): size (B, N, H)
        K (int): number of actions

    Returns:
        torch.Tensor: Vector of size size (B, N, H, K)
    """

    one_hot_act = F.one_hot(actions, num_classes=K)  # (B,N,H,K)
    sum_rews = (one_hot_act * rewards.unsqueeze(-1)).cumsum(dim=-2)
    counts = one_hot_act.cumsum(dim=-2)
    return sum_rews / torch.clamp(counts, min=1)