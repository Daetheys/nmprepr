import os
import click
import pickle

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import set_seed, setup_logger

from nmp.launcher.sac import *
from nmp import settings
import gtimer

from curriculum.utils import SaveReplayBufferEnv

class MazeTrainer:
    def __init__(self,env_name, exp_dir, seed,resume, mode, archi, epochs,
                 reward_scale, hidden_dim, batch_size, learning_rate, n_layers,
                 soft_target_tau, auto_alpha, alpha, frac_goal_replay, horizon,
                 replay_buffer_size, snapshot_mode, snapshot_gap, cpu,
                 num_expl_steps_per_train_loop, num_eval_steps_per_epoch,
                 min_num_steps_before_training, num_trains_per_train_loop,
                 replay_buffer_file
                 ):
        machine_log_dir = settings.log_dir()
        self.base_dir = exp_dir
        exp_dir = os.path.join(machine_log_dir, exp_dir, f"seed{seed}")

        # multi-gpu and batch size scaling
        # replay_buffer_size = replay_buffer_size
        # num_expl_steps_per_train_loop = 3333
        # num_eval_steps_per_epoch = 500
        # min_num_steps_before_training = 10000
        # num_trains_per_train_loop = 500
        # learning rate and soft update linear scaling
        policy_lr = learning_rate
        qf_lr = learning_rate
        self.variant = dict(
            env_name=env_name,
            algorithm="sac",
            version="normal",
            seed=seed,
            resume=resume,
            mode=mode,
            archi=archi,
            replay_buffer_kwargs=dict(max_replay_buffer_size=replay_buffer_size,),
            algorithm_kwargs=dict(
                batch_size=batch_size,
                num_epochs=epochs,
                num_eval_steps_per_epoch=num_eval_steps_per_epoch,
                num_expl_steps_per_train_loop=num_expl_steps_per_train_loop,
                num_trains_per_train_loop=num_trains_per_train_loop,
                min_num_steps_before_training=min_num_steps_before_training,
                max_path_length=horizon,
            ),
            trainer_kwargs=dict(
                discount=0.99,
                soft_target_tau=soft_target_tau,
                target_update_period=1,
                policy_lr=policy_lr,
                qf_lr=qf_lr,
                reward_scale=reward_scale,
                use_automatic_entropy_tuning=auto_alpha,
                alpha=alpha,
            ),
            qf_kwargs=dict(hidden_dim=hidden_dim, n_layers=n_layers),
            policy_kwargs=dict(hidden_dim=hidden_dim, n_layers=n_layers),
            log_dir=exp_dir,
        )
        if mode == "her":
            self.variant["replay_buffer_kwargs"].update(
                dict(
                    fraction_goals_rollout_goals=1
                    - frac_goal_replay,  # equal to k = 4 in HER paper
                    fraction_goals_env_goals=0,
                )
            )
        set_seed(seed)

        self.setup_logger_kwargs = {
            "exp_prefix": exp_dir,
            "variant": self.variant,
            "log_dir": exp_dir,
            "snapshot_mode": snapshot_mode,
            "snapshot_gap": snapshot_gap,
        }
        setup_logger(**self.setup_logger_kwargs)
        ptu.set_gpu_mode(not cpu, distributed_mode=False)
        print(f"Start training...")

        self.expl_env = gym.make(self.variant["env_name"])
        self.eval_env = gym.make(self.variant["env_name"])
        self.expl_env.seed(self.variant["seed"])
        self.eval_env.set_eval()

        mode = self.variant["mode"]
        archi = self.variant["archi"]
        if mode == "her":
            self.variant["her"] = dict(
                observation_key="observation",
                desired_goal_key="desired_goal",
                achieved_goal_key="achieved_goal",
                representation_goal_key="representation_goal",
            )

        if replay_buffer_file:
            with open(replay_buffer_file, 'rb') as f1:
                self.replay_buffer = pickle.load(f1)
                self.replay_buffer.env = self.expl_env
        else:
            self.replay_buffer = get_replay_buffer(self.variant, self.expl_env)

        qf1, qf2, target_qf1, target_qf2, policy, shared_base = get_networks(self.variant,self.expl_env)
        expl_policy = policy
        eval_policy = MakeDeterministic(policy)

        self.expl_path_collector, self.eval_path_collector = get_path_collector(
            self.variant, self.expl_env, self.eval_env, expl_policy, eval_policy
        )

        mode = self.variant["mode"]
        self.trainer = SACTrainer(
            env=self.eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **self.variant["trainer_kwargs"],
        )
        if mode == "her":
            self.trainer = HERTrainer(self.trainer)

    def change_env(self,env_name):
        self.expl_env = gym.make(env_name)
        self.eval_env = gym.make(env_name)
        self.expl_env.seed(self.variant["seed"])
        self.eval_env.set_eval()

        self.expl_path_collector._env = self.expl_env
        self.eval_path_collector._env = self.eval_env

        self.trainer.env = self.eval_env

        self.replay_buffer.env = self.expl_env
        # self.replay_buffer = get_replay_buffer(self.variant, self.expl_env)

    def save_replay_buffer(self):
        with SaveReplayBufferEnv(self.replay_buffer):
            with open('/root/' + self.exp_dir + '/replay_buffer', 'wb') as f1:
                pickle.dump(self.replay_buffer, f1)

    def set_dir(self, m, epoch):
        machine_log_dir = settings.log_dir()
        exp_dir = os.path.join(machine_log_dir, self.base_dir, m, f"epoch_{epoch}", f"seed{self.variant['seed']}")
        print(f'Directory is {exp_dir}')
        self.setup_logger_kwargs['exp_dir'] = exp_dir
        self.setup_logger_kwargs['log_dir'] = exp_dir
        setup_logger(**self.setup_logger_kwargs)

    def train(self,nb_epochs):
        gtimer.reset_root()
        self.variant["algorithm_kwargs"]["num_epochs"] = nb_epochs
        self.algorithm = TorchBatchRLAlgorithm(
            trainer=self.trainer,
            exploration_env=self.expl_env,
            evaluation_env=self.eval_env,
            exploration_data_collector=self.expl_path_collector,
            evaluation_data_collector=self.eval_path_collector,
            replay_buffer=self.replay_buffer,
            **self.variant["algorithm_kwargs"],
        )

        self.algorithm.to(ptu.device)
        self.algorithm.train()
