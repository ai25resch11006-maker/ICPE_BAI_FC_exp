"""
ICPE - main file: runs Algorithm 1 (setup -> training phase) for stochastic Gaussian bandits in the
fixed-confidence setting, and saves the trained networks.

Adapted from train_sequential.py and experiment_stochastic_fc.sh of the neurips_v2 branch of the ICPE
repository (https://github.com/rssalessio/icpe, branch neurips_v2, commit 7833a2f), restricted to the
ICPE agent. Hydra is replaced by icpe_setup.create_config. One call of this file is one run of the
script's loop: for K arms it sets environment.bandit_env.max_value = 0.4 * K and
environment.bandit_env.minimum_gap = 1 / K, as the script does (std 0.5 and 30,000 training steps are
the defaults in CONFIG; the horizon default is N = 200, CHANGED from the script's 100). As in the repository, no evaluation is run here; checkpoints
are saved every training.save_frequency steps and at the end. The inference phase (Algorithm 1,
lines 13-15) is icpe_inference.sample_trajectories on a loaded model.

Usage (inside the `bai` conda environment, from this folder), the paper's runs:
    for seed in 1 2 3; for K in 14 12 10 8 6 4:
        python main.py --num_actions K --seed seed
"""
import argparse
import os
import numpy as np
import torch
import wandb
from datetime import datetime
from omegaconf import OmegaConf
from icpe_setup import create_config, ICPEAgent
from icpe_training import train
from icpe_inference import save


def main(cfg):
    date = datetime.today().strftime("%Y_%m_%d-%H_%M_%S")
    print(f'Configuration {cfg}')

    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)

    wandb.init(
        project=cfg.experiment.name,
        notes=f'[{date}] - {cfg.experiment.description}',
        name=f'{cfg.agent.type} - ({cfg.environment.horizon}/{cfg.environment.num_actions}/{int(cfg.training.use_oracle_reward)})',
        tags=[
            cfg.agent.type,
            f'{cfg.environment.bandit_env.type}',
            f'H={cfg.environment.horizon}',
            f'K={cfg.environment.num_actions}',
            f'Oracle={cfg.training.use_oracle_reward}',
            'transformer'
        ],
        config=OmegaConf.to_container(cfg)
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    # Algorithm 1, lines 1-2: initial setup
    agent = ICPEAgent(cfg, device)

    env_name = f'{cfg.experiment.name}_{cfg.environment.horizon}_{cfg.environment.num_actions}'
    PATH = f'models/sequential/{env_name}/'
    if not os.path.exists(PATH):
        # Create a new directory because it does not exist
        os.makedirs(PATH)

    # Algorithm 1, lines 3-12: training phase
    train(agent, PATH)

    save(agent, PATH)

    print("Training complete.")

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ICPE (Algorithm 1) - stochastic Gaussian bandits, fixed confidence')
    parser.add_argument('--num_actions', type=int, default=None,
                        help='environment.num_actions (K); also sets max_value = 0.4*K and minimum_gap = 1/K')
    parser.add_argument('--horizon', type=int, default=None, help='environment.horizon (N, default 200)')
    parser.add_argument('--max_training_steps', type=int, default=None, help='training.max_training_steps (default 30000)')
    parser.add_argument('--seed', type=int, default=None, help='experiment.seed (default 1)')
    parser.add_argument('--delta', type=float, default=None,
                        help='target accuracy 1 - delta: sets agent.confidence_level = 1 - delta (default 0.9)')
    args = parser.parse_args()

    K = args.num_actions if args.num_actions is not None else create_config().environment.num_actions
    overrides = {
        'environment.num_actions': args.num_actions,
        'environment.horizon': args.horizon,
        'training.max_training_steps': args.max_training_steps,
        'experiment.seed': args.seed,
        'agent.confidence_level': None if args.delta is None else 1 - args.delta,
        # experiment_stochastic_fc.sh: max_value=$(awk "BEGIN{print 0.4 * $num}"), minimum_gap=$(awk "BEGIN{print 1/($num)}")
        # awk prints numbers with 6 significant digits (%.6g), e.g. 1/6 -> 0.166667, 0.4*6 -> 2.4
        'environment.bandit_env.max_value': float('%.6g' % (0.4 * K)),
        'environment.bandit_env.minimum_gap': float('%.6g' % (1 / K)),
    }
    main(create_config(**{k: v for k, v in overrides.items() if v is not None}))