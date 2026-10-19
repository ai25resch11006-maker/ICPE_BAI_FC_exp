"""
ICPE - Q-learning of Q_theta (Algorithm 1, lines 10-11 for theta).

Line 10: update theta using L_policy(B; theta), fixed confidence: eq. (9)-(10). The reward of a terminal
         transition is log I_phi_bar(H* | D_{t+1}) (the repository's .exp() is commented out).
Line 11: theta_bar <- theta (the repository uses a Polyak/soft update after every step,
         with rate agent.polyak_update_q_network).

All code is taken from the neurips_v2 branch of the ICPE repository
(https://github.com/rssalessio/icpe, branch neurips_v2, commit 7833a2f):
  icpe/agents/bandit/sequential/icpe.py -> train_rl_nets (ICPE._train_rl_nets)
"""
from __future__ import annotations
import torch
from typing import Dict, List
from icpe_setup import ReplayBufferSamples


def train_rl_nets(agent, batch: ReplayBufferSamples) -> Dict[str, List[float]]:
    """Perform DQN  update."""

    with torch.no_grad(), torch.amp.autocast(agent.device, enabled=agent.use_amp):
        # Don't enable noise for target values
        next_timesteps = batch.timesteps + (1 - batch.dones)
        next_act = agent.critic.forward(batch.next_states, next_timesteps).argmax(-1, keepdim=True)
        next_q = agent.critic_tgt.forward(batch.next_states, next_timesteps).gather(-1, next_act)

        if agent.cfg.training.use_oracle_reward:
            rewards = batch.rewards
        else:
            # Reward from the target inference network I_phi_bar
            log_probs_next = agent.i_tgt_net.forward(batch.next_states, next_timesteps).gather(-1, batch.optimal_actions.unsqueeze(-1)).flatten()
            rewards_next =  log_probs_next.flatten().detach()#.exp()

        cost = -(1-batch.dones.float()) * agent.cost_per_timestep

        next_q = next_q.flatten()
        d = batch.dones.float()
        tgt_val = cost +  d * rewards_next + agent.agent_cfg.gamma * (1 - d) * next_q



    with torch.amp.autocast(agent.device, enabled=agent.use_amp):

        qvalues = agent.critic.forward(batch.states, batch.timesteps)
        qvalues_act = qvalues.gather(-1, batch.actions.unsqueeze(-1)).flatten()

        qloss = torch.nn.functional.mse_loss(qvalues_act, tgt_val.detach())

        # Prevent double accounting
        idxs = batch.actions != agent.num_actions - 1
        qnext = agent.critic.forward(batch.next_states[idxs], batch.timesteps[idxs])[...,-1]
        qloss_stopping_current = torch.nn.functional.mse_loss(qnext, rewards_next[idxs])
        qloss = qloss + qloss_stopping_current

    agent.critic_optimizer.zero_grad()
    agent.q_scaler.scale(qloss).backward()
    agent.q_scaler.unscale_(agent.critic_optimizer)
    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(),
                                agent.agent_cfg.grad_clipping)
    agent.q_scaler.step(agent.critic_optimizer)
    agent.q_scaler.update()


    # theta_bar <- theta (soft update)
    agent.critic_tgt.soft_update(agent.critic, agent.agent_cfg.polyak_update_q_network)

    return {'value_loss': qloss.item()}
