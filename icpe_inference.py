"""
ICPE - inference phase (Algorithm 1, lines 13-15).

Line 13: sample unknown environment M ~ P             -> sample_envs
Line 14: collect D_tau with pi_t(D_t) = argmax_a Q_theta, until a_stop or N
                                                      -> sample_actions(..., greedy=True)
Line 15: return H_hat = argmax_H I_phi(H | D_tau)     -> HistoryBuilder.build (opt_actions)
HistoryData.stopping_time is the stopping time of each trajectory (traj_current_timestep).

All code is taken from the neurips_v2 branch of the ICPE repository
(https://github.com/rssalessio/icpe, branch neurips_v2, commit 7833a2f):
  icpe/agents/bandit/history.py                         -> HistoryData, HistoryBuilder
  icpe/agents/bandit/sequential/base_offpolicy_agent.py -> sample_trajectories, evaluate (eval)
  icpe/agents/bandit/sequential/icpe.py                 -> save, load (ICPE.save, ICPE.load)
HistoryBuilder methods not used by ICPE (record_optimal_dist, record_old_log_probs,
compute_info_gain) are left out.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple
from icpe_setup import ICPEAgent
from icpe_sampling import sample_envs, sample_actions, mle


# ---------------------------------------------------------------------------
# Trajectories collected at inference          [icpe/agents/bandit/history.py]
# ---------------------------------------------------------------------------
@dataclass
class HistoryData:
    means: torch.Tensor                     # (B, ...)
    exp_actions: torch.Tensor               # (B, N, H)
    opt_actions: torch.Tensor               # (B, N, H)
    soft_opt_actions: torch.Tensor          # (B, N, H, K)
    rew_exp_actions: torch.Tensor           # (B, N, H)
    rew_opt_actions: torch.Tensor           # (B, N, H)
    exp_distribution: torch.Tensor          # (B, N, H, K)
    rew_soft_opt_actions: torch.Tensor      # (B, N, H)
    stopping_time: torch.Tensor             # (B, N)
    old_log_probs: torch.Tensor = None      # (B, N, H), optional if PPO is used
    returns: torch.Tensor = None            # (B, N, H)
    info_gain_rewards: torch.Tensor = None  # (B, N, H)



    def to_numpy(self) -> HistoryData:
        """Convert all tensors to numpy arrays."""
        def t2np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy()

        return HistoryData(
            means=t2np(self.means),
            exp_actions=t2np(self.exp_actions),
            opt_actions=t2np(self.opt_actions),
            rew_exp_actions=t2np(self.rew_exp_actions),
            rew_opt_actions=t2np(self.rew_opt_actions),
            exp_distribution=t2np(self.exp_distribution),
            soft_opt_actions=t2np(self.soft_opt_actions),
            rew_soft_opt_actions=t2np(self.rew_soft_opt_actions),
            old_log_probs=None if self.old_log_probs is None else t2np(self.old_log_probs),
            info_gain_rewards=None if self.info_gain_rewards is None else t2np(self.info_gain_rewards),
            stopping_time=t2np(self.stopping_time)
        )

    def statistics(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute statistics on the trajectories and return as DataFrames."""
        # @TODO This method should not be here
        traj = self.to_numpy()
        rews: np.ndarray = traj.rew_opt_actions[...,-1]  # Reward at the last timestep, shape (B,N)
        B, N, H, K = traj.exp_distribution.shape

        # Compute avg per env + total env
        avgs = [rews[b].mean() for b in range(B)] + [np.mean(rews)]
        # Compute std per env + total std
        stds = [np.nan_to_num(rews[b].std(ddof=1)) for b in range(B)] + [np.sqrt(np.nan_to_num(rews.var(-1, ddof=1)).mean() + np.nan_to_num(rews.mean(-1).var(ddof=1)))]
        indexes = [f'Env_{b}' for b in range(B)] + ['Overall']

        estimation_df =  pd.DataFrame(
            {"avg": avgs,
             "std": stds}, index=indexes)

        # Exploration statistics
        idxs = pd.MultiIndex.from_tuples(
            [(b,h) for b in range(B) for h in range(H)],
            names=["Env", "Stage"])
        avg_exp_dist = traj.exp_distribution.mean(1) # average over trajectories
        exp_df = pd.DataFrame({f'a{i}': avg_exp_dist[..., i].flatten() for i in range(K)},
                              index=idxs)
        return estimation_df, exp_df


class HistoryBuilder:
    def __init__(self, B: int, N: int, horizon: int, num_actions: int, device: torch.device = torch.device('cpu')):
        self.B = B
        self.N = N
        self.horizon = horizon
        self.num_actions = num_actions
        self.device = device

        # Pre-allocate tensors
        # We'll store actions as long, distributions and rewards as float
        shape_2d = (B*N, horizon)
        shape_3d = (B*N, horizon, num_actions)

        self.exp_actions_one_hot = torch.zeros(shape_3d, dtype=torch.long, device=device)
        self.exp_dist = torch.zeros(shape_3d, dtype=torch.float32, device=device)
        self.opt_dist = torch.zeros(shape_3d, dtype=torch.float32, device=device)
        self.rew_exp_actions = torch.zeros(shape_2d, dtype=torch.float32, device=device)
        self.rew_opt_actions = torch.zeros(shape_2d, dtype=torch.float32, device=device)
        self.exp_actions = torch.zeros(shape_2d, dtype=torch.long, device=device)
        self.soft_opt_actions = torch.zeros(shape_3d, dtype=torch.float32, device=device)
        self.opt_actions = torch.zeros(shape_2d, dtype=torch.long, device=device)
        self.rew_soft_opt_actions = torch.zeros(shape_2d, dtype=torch.float32, device=device)
        self.info_gain_rewards = torch.zeros(shape_2d, dtype=torch.float32, device=device)
        self.stopping_time = torch.zeros(B*N, dtype=torch.float32, device=device)

        # Pre-allocate space for old log probabilities (if PPO is used)
        self.old_log_probs = torch.zeros(shape_2d, dtype=torch.float32, device=device)

        self._idxs = torch.arange(B*N, device=device)

    def record_exploration_dist(self, h: int, dist: torch.Tensor):
        """Record the exploration distribution at time step h."""
        # dist shape: (B*N, num_actions)
        self.exp_dist[:, h] = dist

    def build(self, actions: torch.Tensor, rewards: torch.Tensor, probs: torch.Tensor, means: torch.Tensor, stopping_time: torch.Tensor, use_mle: bool = False) -> HistoryData:
        """Finalize and return a HistoryData object."""
        B, N, H, K = self.B, self.N, self.horizon, self.num_actions

        last_h = self.horizon - 1
        self.rew_exp_actions = rewards
        self.exp_actions = actions
        self.exp_actions_one_hot = F.one_hot(self.exp_actions, num_classes=K)
        if use_mle:
            _mle = mle(actions, rewards, K)
            self.opt_actions = _mle.argmax(-1)
            self.soft_opt_actions = F.softmax(_mle, dim=-1)
            self.rew_soft_opt_actions = self.soft_opt_actions.gather(-1, means.argmax(-1,keepdim=True).unsqueeze(1).repeat(1,H,1)).squeeze(-1)

        else:
            # Line 15: H_hat = argmax_H I_phi(H | D)
            self.opt_actions = probs.argmax(-1) #mle.argmax(-1)
            self.soft_opt_actions =probs.exp() #F.softmax(_mle, dim=-1)
            self.rew_soft_opt_actions = probs.exp().gather(-1, means.argmax(-1,keepdim=True).unsqueeze(1).repeat(1,H,1)).squeeze(-1)

        optimal_actions = means.argmax(-1).to(self.device)
        self.rew_opt_actions = (self.opt_actions == optimal_actions.unsqueeze(-1).repeat(1, H)).float()

        self.stopping_time = stopping_time
        return HistoryData(
            means=means[::N],  # shape (B, K)
            exp_actions=self.exp_actions.reshape(B, N, H),
            opt_actions=self.opt_actions.reshape(B, N, H),
            rew_exp_actions=self.rew_exp_actions.reshape(B, N, H),
            rew_opt_actions=self.rew_opt_actions.reshape(B, N, H),
            exp_distribution=self.exp_dist.reshape(B, N, H, K),
            rew_soft_opt_actions=self.rew_soft_opt_actions.reshape(B, N, H),
            soft_opt_actions=self.soft_opt_actions.reshape(B, N, H, K),
            old_log_probs=self.old_log_probs.reshape(B, N, H),
            info_gain_rewards=self.info_gain_rewards.reshape(B, N, H),
            stopping_time=self.stopping_time.reshape(B,N)
        )


# ---------------------------------------------------------------------------
# Lines 13-15                                  [icpe/agents/bandit/sequential/base_offpolicy_agent.py]
# ---------------------------------------------------------------------------
@torch.inference_mode()
def sample_trajectories(agent, B: int, N: int) -> HistoryData:
    """Sample trajectories according to the current policy.

    Args:
        B (int): number of environments to sample
        N (int): number of trajectories per environment

    Returns:
        HistoryData: Sampled trajectories
    """
    # Sample environments                                                      # line 13
    envs = sample_envs(agent, B, N)
    horizon = agent.cfg.environment.horizon
    num_actions = agent.base_num_actions

    traj_builder = HistoryBuilder(B, N, horizon, num_actions, device=agent.device)
    states, dones, stopped = envs.reset()
    states = states.to_tensor(agent.device)


    true_optimal = torch.tensor(envs.means, device=agent.device).argmax(-1)
    traj_current_timestep = torch.zeros(B*N, device=agent.device, dtype=torch.float32)

    log_probabilities = torch.zeros(B*N, horizon, agent.base_num_actions, device=agent.device)

    for h in range(agent.horizon):                                             # line 14
        seq_state = states.to_transformer_input(agent.base_num_actions, traj_current_timestep)
        actions, qvalues, probs = sample_actions(agent, seq_state, greedy=True)
        log_probs = agent.i_net.forward(seq_state.state, seq_state.timestep)

        log_probabilities[:, h] = log_probs


        informative_probs: torch.Tensor = torch.nn.functional.softmax(qvalues)
        traj_builder.record_exploration_dist(h, informative_probs[...,:agent.base_num_actions])

        # Execute action and get rewards
        next_states, dones, stopped= envs.step(actions.tolist())
        next_states = next_states.to_tensor(agent.device)
        dones = torch.from_numpy(dones).to(agent.device)
        stopped = torch.from_numpy(stopped).to(agent.device)



        states = next_states
        traj_current_timestep = traj_current_timestep + (1-stopped.float())
        if torch.all(dones):
            break



    means = torch.from_numpy(envs.means).to(dtype=torch.float32, device=agent.device)
    seq_state = states.to_transformer_input(agent.base_num_actions, traj_current_timestep)
    log_probs = agent.i_net.forward(seq_state.state, seq_state.timestep)
    log_probabilities[:, -1] = log_probs


    traj = traj_builder.build(states.actions, states.rewards, log_probabilities.exp(), means, traj_current_timestep)  # line 15

    return traj


def evaluate(agent, B: int, N: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate method over N trajectories per B environments

    Args:
        B (int): number of environments to sample
        N (int): number of trajectories per environment

    Returns:
        df (pandas.DataFrame): Pandas dataframe with the evaluation statistics
    """
    traj = sample_trajectories(agent, B, N)
    df_est, df_exp = traj.statistics()
    K = traj.exp_distribution.shape[-1]
    H = agent.cfg.environment.horizon
    idenv = np.random.choice(B)

    eval_dict = {
        x: [float(y) for y in np.round(df_exp.loc[(idenv, x)].to_numpy(),3)] for x in [0, H//4, H//2, 3*H//4,H-1]
    }

    print(f'----------------------------------')
    print(f'[Eval] Agent: {agent.agent_cfg.type} - Env: {agent.cfg.environment.bandit_env.type} - Horizon: {agent.cfg.environment.horizon}')
    print(f'[Eval] Env means: {traj.means[idenv]}')
    print(f'[Eval] MLE: { mle(traj.exp_actions, traj.rew_exp_actions, K)[idenv].mean(0)[-1]}')


    print(df_exp.loc[idenv].round(2))
    return df_est, df_exp


# ---------------------------------------------------------------------------
# Saving / loading the trained networks        [icpe/agents/bandit/sequential/icpe.py]
# ---------------------------------------------------------------------------
def save(agent, path: str):
    model = {'cfg': agent.cfg,
             'critic': agent.critic.state_dict(),
             'Enet': agent.i_net.state_dict(),
             'step': agent.current_step}
    filename = f'{path}/icpe_{agent.current_step}_{agent.cfg.experiment.seed}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load(path: str, device: torch.device) -> ICPEAgent:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    agent = ICPEAgent(data['cfg'], device)
    agent.critic.load_state_dict(data['critic'])
    agent.i_net.load_state_dict(data['Enet'])
    agent.critic_tgt.load_state_dict(data['critic'])
    agent.i_tgt_net.load_state_dict(data['Enet'])
    return agent
