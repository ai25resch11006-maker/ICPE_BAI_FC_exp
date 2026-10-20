"""
ICPE - training phase (Algorithm 1, lines 3-12), stochastic Gaussian bandits, fixed confidence.

Line 3:    while training is not over                    -> for step in range(max_training_steps)
Line 4:    sample M ~ P with H*                          -> sample_envs, true_optimal
Lines 5-8: repeat ... until a_t = a_stop or t > N        -> for t in range(horizon)
Line 6:    a_t = argmax_a Q_teta(D_t, a), observe x_t+1 -> sample_actions, envs.step
Line 7:    add (D_t, a_t, x_t+1, H*) to B                -> agent.buffer.add
Line 9:    lambda update (eq. 11)                        -> update_cost, after the rollout
Line 10:   update phi with L_inf (eq. 7)                 -> train_estimation_net
           update theta with L_policy (eq. 9-10)         -> train_rl_nets (icpe_q_learning.py)
Line 11:   phi_bar <- phi every T_phi steps              -> train_estimation_net
           theta_bar <- theta                            -> train_rl_nets (icpe_q_learning.py)
The networks are trained once per training step, after the rollout, and the learning rates decay
linearly (agent.scheduler_step). A checkpoint is saved every training.save_frequency steps.
CHANGED: agent.update_scheme selects the replay loop of neurips_submission (default: replay_ratio
I-steps, then replay_ratio Q-steps, each on a fresh batch) or of neurips_v2 (replay_ratio rounds, one
shared batch per round).

CHANGED (not in the repository): every training episode gets its own seed,
    seed = experiment.seed * 100,000,000 + episode number (0, 1, 2, ...),
which seeds that episode's reward noise (GaussianBanditEnv.seed). The seed and the arm means of every
episode are saved with each checkpoint in episodes_<experiment.seed>.npz (arrays `seeds`, `means`).

All code is taken from the neurips_v2 branch of the ICPE repository
(https://github.com/rssalessio/icpe, branch neurips_v2, commit 7833a2f):
  icpe/agents/bandit/sequential/base_offpolicy_agent.py -> train_estimation_net (_train_estimation_net),
      train_networks (_train_networks), update_cost (_update_cost), log_statistics (_log_statistics),
      train (train)
  icpe/agents/bandit/utils.py                           -> compute_mingaps_and_probs (debug print)
"""
from __future__ import annotations
import os
import numpy as np
import torch
import wandb
from typing import Tuple, List, Dict
from icpe_setup import ReplayBufferSamples
from icpe_sampling import sample_envs, sample_actions, mle
from icpe_q_learning import train_rl_nets
from icpe_inference import save


def compute_mingaps_and_probs(means: np.ndarray, minimum_gap: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    min_gap = np.diff(np.sort(means, axis=-1), axis=-1)[:,-1]
    min_gap_repeated = np.repeat(min_gap[:,None], means.shape[-1], axis=-1)

    min_gap_repeated[np.isclose(min_gap_repeated,0)]=1e-6

    gaps = means.max(-1,keepdims=True)- means + minimum_gap
    idxs = np.isclose(gaps,0)
    gaps[idxs] = min_gap_repeated[idxs]
    gaps_inv_squared = 1/gaps ** 2
    gaps_probs = gaps_inv_squared / gaps_inv_squared.sum(-1, keepdims=True)
    return min_gap, gaps_probs


def train_estimation_net(agent, batch: ReplayBufferSamples) -> Dict[str, List[float]]:
    """Line 10: update phi with L_inf (eq. 7); line 11: phi_bar <- phi every T_phi steps."""
    agent.i_net.train()

    with torch.amp.autocast(agent.device, enabled=agent.use_amp):
        log_probs = agent.i_net.forward(
            batch.next_states,
            batch.timesteps).gather(-1, batch.optimal_actions.unsqueeze(-1))
        est_loss = -log_probs.mean()

    agent.i_optimizer.zero_grad()
    # scale → backward → unscale → clip → step → update
    agent.i_scaler.scale(est_loss).backward()
    agent.i_scaler.unscale_(agent.i_optimizer)
    torch.nn.utils.clip_grad_norm_(agent.i_net.parameters(),
                                   agent.agent_cfg.grad_clipping)
    agent.i_scaler.step(agent.i_optimizer)
    agent.i_scaler.update()


    agent.e_gradient_step += 1
    if agent.e_gradient_step % int(agent.agent_cfg.frequency_update_e_network * 100) == 0:   # CHANGED: repository uses * environment.horizon; fixed at 2 * 100 = 200 updates (its N = 100 value) for any N
        agent.i_tgt_net.hard_update(agent.i_net)
    return {'e_loss': [est_loss.item()]}


def train_networks(agent) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Line 10: update phi (eq. 7) and theta (eq. 9-10) from the buffer.

    CHANGED (not in the repository): agent.update_scheme selects the replay loop.
      'submission' (default) - neurips_submission (commit 0260c4f), BaseOffPolicyAgent._train_networks:
                               replay_ratio phi-steps on fresh batches, then replay_ratio theta-steps on
                               other fresh batches.
      'v2'                   - neurips_v2 (commit 7833a2f): replay_ratio rounds; in each, one batch
                               updates phi, then theta.
    """
    if agent.agent_cfg.get('update_scheme', 'submission') == 'submission':
        return train_networks_submission(agent)
    return train_networks_v2(agent)


def train_networks_submission(agent) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """neurips_submission: all phi-steps first (fresh batches), then all theta-steps (fresh batches)."""
    eloss, rlloss = {}, {}
    cfg = agent.agent_cfg

    if len(agent.buffer) > cfg.initial_buffer_bootstrap_num_batches * cfg.batch_size * cfg.replay_ratio:
        if not agent.cfg.training.use_oracle_reward:
            # Don't add last transitions to avoid leakage! if the capacity of the E network is sufficiently large it will overfit new samples
            for _ in range(cfg.replay_ratio):
                eloss = train_estimation_net(agent, agent.buffer.sample(cfg.batch_size, add_last_transtions=False))

        # 30 so that the enetwork can bootstrap int he meantime.
        if len(agent.buffer) > cfg.e_network_bootstrap_num_batches * cfg.batch_size * cfg.replay_ratio:
            for _ in range(cfg.replay_ratio):
                rlloss = train_rl_nets(agent, agent.buffer.sample(cfg.batch_size, add_last_transtions=False))
                agent.gradient_step += 1

    return eloss, rlloss


def train_networks_v2(agent) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """neurips_v2: replay_ratio rounds; in each, one batch B ~ buffer updates phi, then theta."""
    eloss, rlloss = {}, {}

    if len(agent.buffer) >  agent.agent_cfg.initial_buffer_bootstrap_num_batches * agent.agent_cfg.batch_size * agent.agent_cfg.replay_ratio:
        for _ in range(agent.agent_cfg.replay_ratio):
            batch = agent.buffer.sample(agent.agent_cfg.batch_size, add_last_transtions=False)
            if not agent.cfg.training.use_oracle_reward:
                # Don't add last transitions to avoid leakage! if the capacity of the E network is sufficiently large it will overfit new samples
                eloss = train_estimation_net(agent, batch)

            # 30 so that the enetwork can bootstrap int he meantime.
            if len(agent.buffer) > agent.agent_cfg.e_network_bootstrap_num_batches * agent.agent_cfg.batch_size * agent.agent_cfg.replay_ratio:
                rlloss = train_rl_nets(agent, batch)
                agent.gradient_step += 1

    return eloss, rlloss


def update_cost(agent, log_probs: torch.Tensor, true_optimal: torch.Tensor):
    """Line 9: lambda update, eq. (11). confidence_level = 1 - delta."""
    estm_rewards_argmax = (log_probs.argmax(-1) == true_optimal).float()


    err = agent.agent_cfg.confidence_level-estm_rewards_argmax.mean()

    cost_loss = agent._cost_multiplier * err.detach()
    agent.cost_optimizer.zero_grad()
    cost_loss.backward()
    agent.cost_optimizer.step()
    with torch.no_grad():
        agent._cost_multiplier.data.clamp_(min=agent.min_cost, max=1)

    return {'cost': agent._cost_multiplier.item(), 'target_confidence_error': err.abs().item()}


@torch.no_grad()
def log_statistics(agent, step: int, training_statistics: Dict[str, List[float]], mle_rewards: torch.Tensor,
                    log_probs_e_network: torch.Tensor, true_optimal: torch.Tensor,
                    qvalues: torch.Tensor, exploration_probs: torch.Tensor = None, horizons: torch.Tensor = None):
        """Logs statistics to Wandb

        Args:
            step (int): current training step
            training_statistics (Dict[str, List[float]]): A dictonary containing training statistics
            mle_rewards (torch.Tensor): The estimated maximum likelihood reward vector
            log_probs_e_network (torch.Tensor): log probabilities of the estimated optimal arm from the e network
            true_optimal (torch.Tensor): vector with the true optimal arms
            qvalues (torch.Tensor): qvalues of the exploration policy
            exploration_probs (Optional[torch.Tensor]): exploration probabilities of the true policy. Defaults to None
            horizon (Optional[torch.Tensor]): array of horizons at which the trajectory stopped
        """
        estm_rewards_argmax = (log_probs_e_network.argmax(-1) == true_optimal).float()
        estm_rewards_probs = log_probs_e_network.exp().gather(-1, true_optimal.unsqueeze(-1)).flatten()

        final_action_correct = mle_rewards
        mu = np.round(final_action_correct.mean().item(), 3)
        std = np.round(final_action_correct.std().item(), 3)
        estm_rewards_argmax = estm_rewards_argmax.detach().cpu().numpy()
        estm_rewards_probs = estm_rewards_probs.detach().cpu().numpy()

        print(f"Step {agent.current_step + 1} completed. Final Action Correct: {mu}/{std} ")

        values = {
            "values/avg_qvalues": qvalues.detach().cpu().numpy().mean(),
            "values/std_avg_qvalues": qvalues.detach().cpu().numpy().mean(-1).std(),
            "values/max_qvalues": qvalues.max(-1)[0].mean().detach().cpu().numpy(),
            "values/min_qvalues": qvalues.min(-1)[0].mean().detach().cpu().numpy(),
        }

        performance = {
            "performance/mle_rewards": mle_rewards.mean().item(),
            "performance/est_rewards_argmax": estm_rewards_argmax.mean(),
            "performance/est_rewards_probs": estm_rewards_probs.mean(),
        }
        training_stats = {
            f"loss/{x}": np.mean(y) for x,y in training_statistics.items()
        }

        policy = {}
        if exploration_probs is not None:
            exploration_probs = exploration_probs.detach().cpu().numpy()
            # Geometric mean for probabilities is better suited
            avg_inf_prob = np.prod(exploration_probs, -1) ** (1 / exploration_probs.shape[-1])
            entropy = -(exploration_probs * np.log(exploration_probs + 1e-9)).sum(-1) /np.log(avg_inf_prob.shape[-1])

            policy = {
                'policy/max_probability': exploration_probs.max(-1).mean(),
                'policy/min_probability': exploration_probs.min(-1).mean(),
                'policy/avg_probability': avg_inf_prob.mean(),
                'policy/std_avg_probability': avg_inf_prob.std(),
                'policy/entropy': entropy.mean()
                }

        if horizons is not None:
            _performance_horizons = {'performance/avg_horizon': horizons.mean().item(), 'performance/median_horizon': horizons.median().item()}
            performance = {**performance, **_performance_horizons}

        wandb.log({
            **performance, **training_stats, **values, **policy
        }, step=step)


def train(agent, save_path: str):
    """Training phase of Algorithm 1 (lines 3-12)."""
    # Collect rollouts
    B, N = agent.agent_cfg.num_env_instances, agent.agent_cfg.num_trajectories

    # CHANGED (not in the repository): seed and means of every training episode
    episode_seeds, episode_means = [], []
    num_episodes = 0

    for step in range(agent.cfg.training.max_training_steps):                 # line 3
        # Sample environments                                                  # line 4
        envs = sample_envs(agent, B=B, N=N)

        # CHANGED: seed each episode's reward noise and record it with the episode's means
        seeds = agent.cfg.experiment.seed * 100_000_000 + num_episodes + np.arange(B*N)
        for env, seed in zip(envs.envs, seeds):
            env.seed(int(seed))
        episode_seeds.append(seeds)
        episode_means.append(envs.means.copy())
        num_episodes += B*N

        states, dones, stopped = envs.reset()
        states = states.to_tensor(agent.device)
        true_optimal = torch.tensor(envs.means, device=agent.device).argmax(-1)
        traj_current_timestep = torch.zeros(B*N, device=agent.device, dtype=torch.float32)

        # Collect transitions                                                  # lines 5-8
        for t in range(agent.horizon):
            with torch.no_grad(), torch.amp.autocast(agent.device, enabled=agent.use_amp):
                seq_state = states.to_transformer_input(agent.base_num_actions, traj_current_timestep)
                actions, qvalues, probs = sample_actions(agent, seq_state)    # line 6


            # neurips_v2 only: these two calls were used to save attention maps; that code is
            # commented out inside the networks, so they only compute a forward pass. They are kept
            # for update_scheme 'v2' because np.random.rand() is part of its random number sequence;
            # neurips_submission has no such call.
            if agent.agent_cfg.get('update_scheme', 'submission') == 'v2' and np.random.rand() < 5e-4:
                with torch.no_grad(), torch.amp.autocast(agent.device, enabled=agent.use_amp):

                    _ = agent.i_net.forward(seq_state.state, seq_state.timestep, _save_attention=True, step=step, traj_t=t)

                    _ = agent.critic.forward(seq_state.state, seq_state.timestep, _save_attention=True, step=step, traj_t=t)

            # Execute action and get rewards
            next_states, dones, stopped= envs.step(actions.tolist())
            next_states = next_states.to_tensor(agent.device)
            dones = torch.from_numpy(dones).to(agent.device)
            stopped = torch.from_numpy(stopped).to(agent.device)


            _mle = mle(next_states.actions, next_states.rewards, agent.base_num_actions)
            mle_estimates = _mle.gather(1, traj_current_timestep.long().view(-1,1,1).expand(-1, 1, _mle.size(2))).squeeze(1)
            mle_predicted_optimal = mle_estimates.argmax(-1)
            mle_rewards = (mle_predicted_optimal == true_optimal).float() * dones

            agent.buffer.add(states=seq_state.state,                           # line 7
                            actions=actions,
                            rewards=mle_rewards,
                            env_rewards=next_states.rewards.gather(-1, traj_current_timestep.unsqueeze(-1).long()).flatten(),
                            dones=dones,
                            next_states=next_states.to_transformer_input(agent.base_num_actions, traj_current_timestep + 1).state,
                            optimal_actions=true_optimal,
                            stopped=stopped,
                            timesteps=traj_current_timestep)

            states = next_states
            traj_current_timestep = traj_current_timestep + (1-stopped.float())

            if torch.all(dones):
                break

        eloss, qloss = train_networks(agent)                                   # lines 10-11
        with torch.no_grad(), torch.amp.autocast(agent.device, enabled=agent.use_amp):

            seq_state = states.to_transformer_input(agent.base_num_actions, traj_current_timestep)
            log_probs = agent.i_net.forward(seq_state.state, seq_state.timestep)
            op = log_probs.exp().detach().cpu().numpy().round(2)
            op_tgt = agent.i_tgt_net.forward(seq_state.state, seq_state.timestep).exp().detach().cpu().numpy().round(2)

        # Line 9: update lambda according to eq. (11)
        cost_updates = update_cost(agent, log_probs, true_optimal)


        qv = qvalues.detach().cpu().numpy().round(2)[0]

        compressed_state = states.compress(agent.base_num_actions, traj_current_timestep)
        counts = (compressed_state[:, :agent.base_num_actions][0]*agent.horizon).detach().cpu().numpy().round(3)
        running_means = compressed_state[:, agent.base_num_actions:].detach().cpu().numpy()

        true_min_gap, true_gaps_probs = compute_mingaps_and_probs(envs.means)
        est_min_gap, est_gaps_probs = compute_mingaps_and_probs(running_means)
        print('Counts: ', counts)
        print('Probs: ', counts/agent.horizon)
        print(f'Est Gaps prob: {est_gaps_probs[0]}')
        print(f'True Gaps prob: {true_gaps_probs[0]}')
        print('Means: ', running_means[0])
        print('True Means: ', envs.means[0].round(3))
        print(f'Min gap: {true_min_gap.min().round(3)}')
        print(f'Max gap: {true_min_gap.max().round(3)}')
        print(f'Avg gap(std): {true_min_gap.mean().round(3)}/ {true_min_gap.std().round(3)}')
        print('QV: ', qv)
        print(f'OP     : {op[0]} - {op.mean()} -{op.std()} ')
        print(f'OP  tgt: {op_tgt[0]} - {op_tgt.mean()} -{op_tgt.std()} ')
        print()


        learning_rates = {
            "lr_exploration_net": agent.critic_optimizer.param_groups[0]['lr'],
            "lr_inference_net": agent.i_optimizer.param_groups[0]['lr'],
        }


        log_statistics(
            agent,
            step=agent.current_step,
            training_statistics={**eloss, **qloss, **cost_updates, **learning_rates},
            mle_rewards=mle_rewards,
            log_probs_e_network=log_probs,
            true_optimal=true_optimal,
            qvalues=qvalues,
            exploration_probs=probs,
            horizons=traj_current_timestep)
        agent.current_step += 1

        agent.scheduler_step()
        if agent.current_step % agent.cfg.training.save_frequency == 0:
            save(agent, save_path)
            # CHANGED: save the seeds and means of all episodes so far
            np.savez(os.path.join(save_path, f'episodes_{agent.cfg.experiment.seed}.npz'),
                     seeds=np.concatenate(episode_seeds), means=np.concatenate(episode_means))
