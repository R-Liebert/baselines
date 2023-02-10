#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
import baselines.common.tf_util as U
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
import os.path as osp
import numpy as np


def main():
    parser = mujoco_arg_parser()
    add_other_args(parser)
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure(osp.join(args.workspace, args.log_dir))
    else:
        logger.configure(osp.join(args.workspace, args.log_dir), format_strs=[])
        logger.set_level(logger.DISABLED)

    workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         hid_size=32, num_hid_layers=2)
    env = make_mujoco_env(args.env, workerseed)
    expert_dir = None if args.expert_dir is None else osp.join(args.workspace, args.expert_dir)
    pretrain_dir = None if args.pretrain_dir is None else osp.join(args.workspace, args.pretrain_dir)
    if args.random_sample_switch_iter:
        T = args.hard_switch_iter
        assert T is not None
        low = T // 2
        high = T + 1
        prob_mass = np.arange(low, high) ** 3
        prob_mass = prob_mass / np.sum(prob_mass)
        hard_switch_iter = np.random.choice(np.arange(low, high), p=prob_mass)
    else:
        hard_switch_iter = args.hard_switch_iter
    trpo_mpi.learn(policy_fn, env, cg_iters=10, cg_damping=0.1,
                   gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3,
                   timesteps_per_batch=args.timesteps_per_batch, max_iters=args.max_iters,
                   mode=args.mode,
                   policy_save_freq=args.policy_save_freq,
                   expert_dir=expert_dir,
                   ilrate_multi=args.ilrate_multi, ilrate_decay=args.ilrate_decay,
                   hard_switch_iter=hard_switch_iter,
                   save_no_policy=args.save_no_policy,
                   pretrain_dir=pretrain_dir,
                   il_gae=args.il_gae,
                   initialize_with_expert_logstd=args.initialize_with_expert_logstd)
    env.close()


def add_other_args(parser):
    parser.add_argument('--timesteps_per_batch', type=int, default=1000)  # number of timesteps per batch per worker
    parser.add_argument('--mode', type=str, choices=['train_expert', 'pretrain_il', 'online_il', 'batch_il', 'thor', 'lols'])
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--workspace', type=str, default='')
    parser.add_argument('--log_dir', default='', type=str, help='this is where the policy and results will be saved')
    parser.add_argument('--policy_save_freq', default=50, type=int,
                        help='learner policy will be saved per these iterations')
    parser.add_argument('--expert_dir', type=str, help='the directory of the expert policy to be loaded')
    parser.add_argument('--expert_file_prefix', type=str, help='the prefix for the policy checkpoint', default='policy')
    parser.add_argument('--pretrain_dir', type=str, help='the directory of a pretrained policy for learner initialization')
    parser.add_argument('--pretrain_file_prefix', type=str, help='the prefix for the policy checkpoint', default='policy')
    parser.add_argument('--save_no_policy', action='store_true',
                        help='whether the learner policies will be saved')
    parser.add_argument('--ilrate_multi', type=float, default=0.0,
                        help='the multiplier of il objective [online_il, batch_il]')
    parser.add_argument('--ilrate_decay', default=1.0, type=float, help='the decay of il objective [online_il, batch_il]')
    parser.add_argument('--hard_switch_iter', type=int, help='switch from il to rl at this iteration [online_il, batch_il]')
    parser.add_argument('--random_sample_switch_iter', action='store_true',
                        help='random sample the iteration to switch from il to rl, with a distribution parameterized by ' +
                        '--hard_switch_iter')
    parser.add_argument('--truncated_horizon', type=int, help='horizon [thor]')
    parser.add_argument('--deterministic_expert', action='store_true',
                        help='whether to use deterministic expert for BC')
    parser.add_argument('--il_gae', action='store_true')
    parser.add_argument('--initialize_with_expert_logstd', action='store_true')


if __name__ == '__main__':
    main()
