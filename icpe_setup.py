"""
ICPE - initial setup (Algorithm 1, lines 1-2). Stochastic Gaussian bandits, fixed-confidence
setting only: the stop action a_stop is always available and lambda is learned with eq. (11).

Line 1 (inputs): the configuration below.
Line 2: initialize buffer B, networks Q_theta (critic) and I_phi (inference network), and set
        theta_bar <- theta, phi_bar <- phi.

All code is taken from the neurips_v2 branch of the ICPE repository
(https://github.com/rssalessio/icpe, branch neurips_v2, commit 7833a2f), which contains the
stochastic-bandit experiment of the paper (sec. 4.1, fig. 2; arXiv app. D.1.2):
  configs/config_sequential.yaml, configs/agent/sequential/icpe.yaml,
  configs/environment/bandit.yaml, configs/environment/bandit_env/gaussian_bandit.yaml,
  configs/experiments/sequential/stochastic_gaussian_fixed_confidence/icpe.yaml,
  experiment_stochastic_fc.sh (overrides)                    -> CONFIG
  icpe/models/sequence_net/utils.py                          -> TransformerState, TransformerParameters,
                                                                BaseTransformer, create_gpt2_config
  icpe/models/sequence_net/critic_transformer.py             -> CriticTransformer
  icpe/models/sequence_net/inference_transformer.py          -> InferenceTransformer
  icpe/models/sequence_net/builder.py                        -> init_weights, create_critic_network,
                                                                create_inference_network
  icpe/agents/bandit/buffers.py                              -> ReplayBufferSamples, ReplayBuffer
  icpe/agents/bandit/sequential/base_offpolicy_agent.py      -> ICPEAgent (__init__, properties, reset_networks)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, NamedTuple
from omegaconf import OmegaConf, DictConfig
from torch.optim.lr_scheduler import LinearLR
from transformers import GPT2Config, GPT2Model


# ---------------------------------------------------------------------------
# Line 1: inputs. Values of configs/experiments/sequential/stochastic_gaussian_fixed_confidence/icpe.yaml
# merged with its defaults (config_sequential.yaml, agent/sequential/icpe.yaml, environment/bandit.yaml,
# environment/bandit_env/gaussian_bandit.yaml), as Hydra composes them, with the overrides of
# experiment_stochastic_fc.sh (horizon 100, std 0.5, max_training_steps 30000, and for K arms
# max_value = 0.4*K, minimum_gap = 1/K; seeds 1, 2, 3; K in 14, 12, 10, 8, 6, 4).
# max_value and minimum_gap below are those of K = 4; main.py sets them from K as the script does.
# CHANGED (not in the repository): horizon N = 200 instead of the script's 100.
# ---------------------------------------------------------------------------
CONFIG = {
    'experiment': {
        'name': 'StochasticGaussianBanditFixedConfidenceSequential',
        'description': 'Stochastic Gaussian Bandit with fixed confidence-ICPE Agent',
        'seed': 1,
        'num_processes': 5,
    },
    'training': {
        'save_frequency': 1000,
        'max_training_steps': 30000,
        'use_oracle_reward': False,
        'cost_per_timestep': 0,
        'eval': {
            'eval_period': 100,
            'eval_num_env_instances': 10,
            'eval_num_trajectories': 1,
        },
    },
    'eval': {
        'num_env_instances': 20,
        'num_trajectories': 20,
    },
    'environment': {
        'bandit_env': {
            'type': 'GaussianBandit',
            'min_value': 0,
            'max_value': 1.6,
            'std': 0.5,
            'num_magic_actions': 0,
            'std_magic_action': 0,
            'minimum_gap': 0.25,
        },
        'num_actions': 4,
        'horizon': 200,                       # CHANGED: the repository's script uses 100
    },
    'agent': {
        'type': 'ICPE',
        'learning_rate': 3e-6,
        'final_learning_rate': 1e-6,
        'grad_clipping': 1,
        'buffer_size': 5000000,
        'gamma': 1,
        'batch_size': 1024,
        'num_env_instances': 100,
        'num_trajectories': 1,
        'polyak_update_q_network': 0.01,
        'frequency_update_e_network': 2,
        'replay_ratio': 5,
        'e_network_bootstrap_num_batches': 50,
        'initial_buffer_bootstrap_num_batches': 20,
        'confidence_level': 0.9,              # 1 - delta (delta = 0.1)
        'cost_learning_rate': 3e-5,
        
        
        # Critic network settings
        'critic_network': {
            'embed_dim': 256,
            'num_layers': 3,
            'num_heads': 2,
            'dropout': 0,
            'resid_dropout': 0,
            'embed_dropout': 0,
            'activation_type': 'gelu',
            'layer_norm_eps': 1e-5,
            'attn_dropout': 0,
        },
        # E network settings
        'e_network': {
            'embed_dim': 256,
            'num_layers': 3,
            'num_heads': 2,
            'dropout': 0,
            'resid_dropout': 0,
            'embed_dropout': 0,
            'activation_type': 'gelu',
            'layer_norm_eps': 1e-5,
            'attn_dropout': 0,
        },
    },
}


def create_config(**overrides) -> DictConfig:
    """Build the configuration as an OmegaConf DictConfig (what Hydra passes in the repo).

    Overrides use dotted keys, e.g. create_config(**{'environment.num_actions': 6}).
    """
    cfg = OmegaConf.create(CONFIG)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


# ---------------------------------------------------------------------------
# Transformers                                 [icpe/models/sequence_net/utils.py]
# ---------------------------------------------------------------------------
class TransformerState(NamedTuple):
    state: torch.Tensor
    timestep: torch.Tensor


@dataclass
class TransformerParameters:
    embed_dim: int
    num_heads: int
    num_layers: int
    attn_dropout: float
    resid_dropout: float
    embed_dropout: float
    activation_type: str = "gelu"


class BaseTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def hard_update(self, source: BaseTransformer) -> None:
        """ Hard update of the network from a source network """
        self.load_state_dict(source.state_dict())

    def soft_update(self, source: BaseTransformer, tau: float) -> None:
        """ Polyak update (soft update) of the network using a time constant tau """
        return self.polyak_update(source, tau)

    def polyak_update(self, source: BaseTransformer, tau: float) -> None:
        """ Polyak update (soft update) of the network using a time constant tau """
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def create_gpt2_config(
    horizon: int,
    params: TransformerParameters
) -> GPT2Config:
    """Helper function to create a GPT2Config with standard parameters"""
    return GPT2Config(
        n_positions=horizon,
        n_embd=params.embed_dim,
        n_layer=params.num_layers,
        n_head=params.num_heads,
        resid_pdrop=params.resid_dropout,
        embd_pdrop=params.embed_dropout,
        attn_pdrop=params.attn_dropout,
        use_cache=False,
        activation_function=params.activation_type,
    )


# ---------------------------------------------------------------------------
# Q_theta network                              [icpe/models/sequence_net/critic_transformer.py]
# The repository's forward contains commented-out code (left padding, attention masks, position ids,
# attention saving) that is not executed; it is not copied. timestep, _save_attention, step and traj_t
# are accepted but unused, as in the repository.
# ---------------------------------------------------------------------------
class CriticTransformer(BaseTransformer):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        horizon: int,
        config: TransformerParameters,
        bins: int = 1
    ):
        super().__init__()
        self._num_actions = num_actions
        self._cfg = config
        self.bins = bins
        self._state_dim = state_dim
        self._horizon = horizon

        self._gpt2_config = create_gpt2_config(horizon, config)
        self._gpt2 = GPT2Model(self._gpt2_config)

        self._embed = nn.Linear(state_dim, config.embed_dim)

        self._output = nn.Linear(config.embed_dim, num_actions * bins)

    def forward(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        _save_attention: bool = False,
        step: int = 0,
        traj_t: int  = 0
    ) -> torch.Tensor:
        """Forward pass

        Args:
            actions: [B, T, action_size] one-hot encoded actions
            rewards: [B, T] rewards

        Returns:
            values: [B, T, action_size * bins] value estimates
        """
        embeds = state

        embeds = F.gelu(self._embed(embeds))

        h = self._gpt2.forward(inputs_embeds=embeds)
        h = h.last_hidden_state
        h = h[:,-1,:]

        out = self._output(h)

        return out


# ---------------------------------------------------------------------------
# I_phi network                                [icpe/models/sequence_net/inference_transformer.py]
# Same note on the commented-out code as for CriticTransformer.
# ---------------------------------------------------------------------------
class InferenceTransformer(BaseTransformer):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        horizon: int,
        config: TransformerParameters
    ):
        super().__init__()
        self._num_actions = num_actions
        self._state_dim = state_dim
        self._cfg = config
        self._horizon = horizon

        self._gpt2_config = create_gpt2_config(horizon, config)
        self._gpt2 = GPT2Model(self._gpt2_config)

        self._embed = nn.Linear(state_dim, config.embed_dim)

        self._output = nn.Linear(config.embed_dim, num_actions)

    def forward(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        _save_attention: bool = False,
        step: int = 0 ,
        traj_t: int=0
    ) -> torch.Tensor:
        """Forward pass

        Args:
            actions: [B, T, action_size] one-hot encoded actions
            rewards: [B, T] rewards

        Returns:
            log_probs: [B, T, action_size] log probabilities
        """
        # states  [B, H, dim_state+dim_action]
        embeds = state
        embeds = F.gelu(self._embed(embeds))

        h = self._gpt2.forward(inputs_embeds=embeds)
        h = h.last_hidden_state

        h = h[:, -1, :]

        out = self._output(h)
        log_probs = F.log_softmax(out, dim=-1)

        return log_probs


# ---------------------------------------------------------------------------
# Network builders                             [icpe/models/sequence_net/builder.py]
# ---------------------------------------------------------------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        stddev = 1 / np.sqrt(m.weight.shape[1])
        torch.nn.init.trunc_normal_(m.weight, mean=0, std=stddev, a=-2*stddev, b=2*stddev)
        torch.nn.init.zeros_(m.bias.data)


def create_critic_network(
    state_dim: int,
    num_actions: int,
    horizon: int,
    config: TransformerParameters,
    bins: int = 1
) -> CriticTransformer:
    # Create network and initialize weights
    network = CriticTransformer(state_dim, num_actions, horizon, config, bins).apply(init_weights)
    return network


def create_inference_network(
    state_dim: int,
    num_actions: int,
    horizon: int,
    config: TransformerParameters
) -> InferenceTransformer:
    # Create network and initialize weights
    network = InferenceTransformer(state_dim, num_actions, horizon, config).apply(init_weights)
    return network


# ---------------------------------------------------------------------------
# Replay buffer B                              [icpe/agents/bandit/buffers.py]
# ---------------------------------------------------------------------------
@dataclass
class ReplayBufferSamples:
    states: torch.Tensor      # Shape: [batch_size, horizon]
    actions: torch.Tensor      # Shape: [batch_size, horizon]
    rewards: torch.Tensor # Shape: [batch_size, horizon]
    env_rewards: torch.Tensor   # Shape: [batch_size]
    next_states: torch.Tensor # Shape: [batch_size, horizon]
    dones: torch.Tensor     # Shape: [batch_size]
    optimal_actions: torch.Tensor # Shape: [batch_size]
    timesteps: torch.Tensor # Shape: [batch-size]


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        num_actions: int,
        num_traj: int,
        device: torch.device,
        horizon: Optional[int] = None,
        enable_last_transitions_sampling: Optional[bool] = False
    ):
        self.buffer_size = buffer_size + (buffer_size % num_traj)
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.num_traj = num_traj
        self.device = device
        self.horizon = horizon
        self.enable_last_transitions_sampling = enable_last_transitions_sampling

        self.reset()

    def reset(self):
        # History buffers
        if self.horizon is None:
            self.states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
            self.next_states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        else:
            self.states = np.zeros((self.buffer_size, self.horizon, self.state_dim), dtype=np.float32)
            self.next_states = np.zeros((self.buffer_size, self.horizon, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.buffer_size, dtype=np.int32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.env_rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.bool_)
        self.optimal_actions = np.zeros((self.buffer_size), dtype=np.int32)
        self.timesteps = np.zeros((self.buffer_size), dtype=np.int16)
        self.new_transitions_idxs = []

        # Position and full flags
        self.pos = 0
        self.full = False

    def add(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        env_rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
        optimal_actions: torch.Tensor,
        stopped: torch.Tensor,
        timesteps: torch.Tensor
    ) -> None:
        """Add a new transition to the buffer"""

        not_stopped = ~stopped
        n_transitions = int(not_stopped.sum())

        space_left = min(n_transitions, self.buffer_size - self.pos)
        remaining = n_transitions - space_left

        self.states[self.pos:self.pos+space_left] = states[not_stopped][:space_left].detach().cpu().numpy()
        self.actions[self.pos:self.pos+space_left] = actions[not_stopped][:space_left].detach().cpu().numpy()
        self.rewards[self.pos:self.pos+space_left] = rewards[not_stopped][:space_left].detach().cpu().numpy()
        self.env_rewards[self.pos:self.pos+space_left] = env_rewards[not_stopped][:space_left].detach().cpu().numpy()
        self.dones[self.pos:self.pos+space_left] = dones[not_stopped][:space_left].detach().cpu().numpy()
        self.next_states[self.pos:self.pos+space_left] = next_states[not_stopped][:space_left].detach().cpu().numpy()
        self.optimal_actions[self.pos:self.pos+space_left] = optimal_actions[not_stopped][:space_left].detach().cpu().numpy()
        self.timesteps[self.pos:self.pos+space_left] = timesteps[not_stopped][:space_left].detach().cpu().numpy()
        if self.enable_last_transitions_sampling:
            self.new_transitions_idxs.extend([x for x in range(self.pos, self.pos+space_left)])
        self.pos += space_left

        if remaining > 0:
            self.full = True
            self.states[:remaining] = states[not_stopped][space_left:].detach().cpu().numpy()
            self.actions[:remaining] = actions[not_stopped][space_left:].detach().cpu().numpy()
            self.rewards[:remaining] = rewards[not_stopped][space_left:].detach().cpu().numpy()
            self.env_rewards[:remaining] = env_rewards[not_stopped][space_left:].detach().cpu().numpy()
            self.dones[:remaining] = dones[not_stopped][space_left:].detach().cpu().numpy()
            self.next_states[:remaining] = next_states[not_stopped][space_left:].detach().cpu().numpy()
            self.optimal_actions[:remaining] = optimal_actions[not_stopped][space_left:].detach().cpu().numpy()
            self.timesteps[:remaining] = timesteps[not_stopped][space_left:].detach().cpu().numpy()
            if self.enable_last_transitions_sampling:
                self.new_transitions_idxs.extend([x for x in range(remaining)])
            self.pos = remaining



    def get_last_inds(self, size: int):
        L = len(self.new_transitions_idxs)
        if size < L:
            idxs = np.array(self.new_transitions_idxs[:size])
            self.new_transitions_idxs = self.new_transitions_idxs[size:]
        else:
            idxs = np.array(self.new_transitions_idxs)
            self.new_transitions_idxs = []
        return idxs


    def sample(self, batch_size: int, add_last_transtions: bool = True) -> ReplayBufferSamples:
        """Sample a batch of transitions"""
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        if add_last_transtions and self.enable_last_transitions_sampling:  # only if we have transitions
            last_inds = self.get_last_inds(batch_size)
            n_last = min(batch_size,len(last_inds))
            batch_inds[:n_last] = last_inds  # force inclusion of the last transitions
        np.random.shuffle(batch_inds)

        return ReplayBufferSamples(
            states=torch.tensor(self.states[batch_inds], device=self.device, dtype=torch.float32),
            actions=torch.tensor(self.actions[batch_inds], device=self.device, dtype=torch.long),
            rewards=torch.tensor(self.rewards[batch_inds], device=self.device, dtype=torch.float32),
            env_rewards=torch.tensor(self.env_rewards[batch_inds], device=self.device, dtype=torch.float32),
            next_states=torch.tensor(self.next_states[batch_inds], device=self.device, dtype=torch.float32),
            dones=torch.tensor(self.dones[batch_inds], device=self.device, dtype=torch.float32),
            optimal_actions=torch.tensor(self.optimal_actions[batch_inds], device=self.device, dtype=torch.long),
            timesteps=torch.tensor(self.timesteps[batch_inds], device=self.device, dtype=torch.long)
        )


    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos


# ---------------------------------------------------------------------------
# Agent state: line 2 of Algorithm 1          [icpe/agents/bandit/sequential/base_offpolicy_agent.py]
# ---------------------------------------------------------------------------
class ICPEAgent(object):
    """Holds the configuration, buffer, networks, optimizers and schedulers of ICPE.

    __init__, the properties and reset_networks are those of BaseOffPolicyAgent in the repo
    (neurips_v2). The other methods of the repo agent are functions in icpe_sampling.py,
    icpe_q_learning.py, icpe_training.py and icpe_inference.py that take this object as first argument.
    """
    cfg: DictConfig
    device: torch.device
    current_step: int

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.agent_cfg = cfg.agent
        self.device = device
        self.current_step = 0
        self.gradient_step = 0
        self.e_gradient_step = 0

        self.num_actions = self.cfg.environment.num_actions + 1    # K arms + a_stop
        self.horizon = self.cfg.environment.horizon
        self.total_batch_size = self.agent_cfg.num_env_instances * self.agent_cfg.num_trajectories

        self.use_amp = (self.device != "cpu")
        self.q_scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.i_scaler = torch.amp.GradScaler(enabled=self.use_amp)


        self.buffer = ReplayBuffer(
            buffer_size=self.agent_cfg.buffer_size,
            state_dim=self.state_dim,
            num_actions=self.num_actions,
            num_traj=self.agent_cfg.num_env_instances * self.agent_cfg.num_trajectories,
            device=device,
            horizon=self.horizon,
            enable_last_transitions_sampling=False
        )

        self.reset_networks()
        self.running_mean_est = 0

    @property
    def has_training_started(self) -> bool:
        return len(self.buffer) > self.agent_cfg.e_network_bootstrap_num_batches * self.agent_cfg.batch_size * self.agent_cfg.replay_ratio

    @property
    def base_num_actions(self) -> int:
        return self.cfg.environment.num_actions

    @property
    def state_dim(self) -> int:
        return self.base_num_actions + 1 # one hot encoding of the actions plus one reward

    @property
    def cost_per_timestep(self) -> float:
        return self._cost_multiplier.item()


    def reset_networks(self):
        bins = 1 if 'bins' not in self.agent_cfg else self.agent_cfg.bins

        # Q_theta and theta_bar <- theta
        self.critic = create_critic_network(self.state_dim, self.num_actions, self.horizon, self.agent_cfg.critic_network, bins=bins).to(self.device)
        self.critic_tgt = create_critic_network(self.state_dim, self.num_actions, self.horizon, self.agent_cfg.critic_network,bins=bins).to(self.device)
        self.critic_tgt.hard_update(self.critic)

        # I_phi and phi_bar <- phi
        self.i_net = create_inference_network(self.state_dim, self.base_num_actions, self.horizon, self.agent_cfg.e_network).to(self.device)
        self.i_tgt_net = create_inference_network(self.state_dim, self.base_num_actions, self.horizon, self.agent_cfg.e_network).to(self.device)
        self.i_tgt_net.hard_update(self.i_net)

        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr = self.agent_cfg.learning_rate)
        self.i_optimizer = torch.optim.AdamW(self.i_net.parameters(), lr = self.agent_cfg.learning_rate)

        # Initial lambda for eq. (11) and its optimizer
        if 'cost_learning_rate' not in self.agent_cfg:
            print('cost_learning_rate missing from config. Setting to 0')
            cost_lr = 0
        else:
            cost_lr = self.agent_cfg.cost_learning_rate
        self.min_cost = 0#1 / (self.num_actions * np.sqrt(self.horizon))
        self._cost_multiplier = torch.tensor(self.min_cost, requires_grad=True, device=self.device, dtype=torch.float32)
        self.cost_optimizer = torch.optim.AdamW([self._cost_multiplier], lr=cost_lr)

        # Learning rate decay from learning_rate to final_learning_rate over half of the training steps
        if 'final_learning_rate' not in self.agent_cfg:
            final_learning_rate = self.agent_cfg.learning_rate
        else:
            final_learning_rate = self.agent_cfg.final_learning_rate

        end_factor = final_learning_rate/self.agent_cfg.learning_rate
        self.scheduler_exp = LinearLR(self.critic_optimizer, start_factor=1.0, end_factor=end_factor, total_iters=int(0.5 * self.cfg.training.max_training_steps) )
        self.scheduler_inf = LinearLR(self.i_optimizer, start_factor=1.0, end_factor=end_factor, total_iters=int(0.5 * self.cfg.training.max_training_steps))

        self.scheduler_step = lambda: (
                self.scheduler_exp.step(),
                self.scheduler_inf.step()
            )