import csv
import sys
import os
import copy
import gc
import pickle

from torch import cuda
import numpy as np
from colabgymrender.recorder import Recorder
from curriculum.maze_trainer import MazeTrainer
import gym

class HideOut:
    def __enter__(self,*args,**kwargs):
        self.out = sys.stdout
        sys.stdout = open('/dev/null','w')
    def __exit__(self,*args,**kwargs):
        sys.stdout.close()
        sys.stdout = self.out

class CurriculumTrainer:
    def __init__(self,
                 mazes,
                 batch_size=128,
                 hidden_dim=256,
                 nb_layers=12,
                 lr=3e-4,
                 horizon=150,
                 threshold=0.90,
                 count_next_threshold=1,
                 num_expl_steps_per_train_loop=3333,
                 num_eval_steps_per_epoch = 500,
                 min_num_steps_before_training=10000,
                 num_trains_per_train_loop=500,
                 replay_buffer_size=int(1e6),
                 frac_goal_replay=0.8,
                 n_viz_path=None,
                 filename_net=None,
                 alpha=None,
                 reward_scale=1.,
                 load_replay_buffer_file = None,
                 save_replay= True,
                 archi = 'pointnet',
                 seed = 0,
                 timeout_threshold=50
                 ):
        self.mazes = mazes

        self.alpha=alpha

        cpu = not cuda.is_available()

        self.exp_dir = 'maze_baseline'

        self.args = dict(env_name=self.mazes[0],
                    exp_dir='maze_baseline',
                    seed=seed,
                    resume=False if filename_net is None else filename_net,
                    mode="her",
                    archi=archi,
                    epochs=0,
                    reward_scale=reward_scale,
                    hidden_dim=hidden_dim,
                    batch_size=batch_size,
                    learning_rate=lr,
                    n_layers=nb_layers,
                    soft_target_tau=5e-3,
                    auto_alpha=alpha is None,
                    alpha=0.1 if alpha is None else alpha,
                    frac_goal_replay=frac_goal_replay,
                    horizon=horizon,
                    replay_buffer_size=replay_buffer_size,
                    snapshot_mode="last",
                    snapshot_gap=10,
                    cpu=cpu,
                    num_expl_steps_per_train_loop=num_expl_steps_per_train_loop,
                    num_eval_steps_per_epoch=num_eval_steps_per_epoch,
                    min_num_steps_before_training=min_num_steps_before_training,
                    num_trains_per_train_loop=num_trains_per_train_loop,
                    replay_buffer_file=load_replay_buffer_file
                    )

        self.mazetrainer = MazeTrainer(**self.args)

        self.threshold = threshold

        self.count_next_threshold = count_next_threshold

        self.save_replay = save_replay

        self.n_viz_path = n_viz_path
        
        self.timeout_threshold = timeout_threshold

    def load(self, file):
        self.args['resume'] = file
        self.mazetrainer = MazeTrainer(**self.args)

    def train(self):
        for m in self.mazes:
            print('----------------------------')
            print('           ',m)
            print('----------------------------')
            self.mazetrainer.change_env(m)
            c = 0
            count_next = 0
            
            timeout = 0
            
            while True:
                with HideOut():
                    self.mazetrainer.set_dir(m, c + 1)
                    self.mazetrainer.train(1)
                    if self.save_replay:
                        self.mazetrainer.save_replay_buffer()
                #Get score
                with open('/root/maze_baseline/seed0/progress.csv', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)

                    l = []
                    for row in reader:
                        l.append(row['evaluation/SuccessRate'])
                    score = eval(l[-1])[0]
                print(c,"{:.2f}%".format(100 * score))
                c += 1

                # save some paths
                if self.n_viz_path is not None:
                    os.makedirs('/content/videos/'+m, exist_ok =True)
                    visualization_env = gym.make(m)
                    visualization_env = Recorder(visualization_env, '/content/videos/'+m+f'/epoch_{c}', fps=30)
                    for i in range(self.n_viz_path):
                        o = visualization_env.reset()
                        done = False
                        path_max=75
                        j = 0
                        while not done and j < path_max:
                            a = self.mazetrainer.trainer._base_trainer.policy.get_action(np.hstack((o['observation'], o['representation_goal'])),deterministic=True)
                            o,r,d,_ = visualization_env.step(copy.deepcopy(a[0]))
                            j += 1
                    print('Videos saved')
                    visualization_env.close()
                    del visualization_env
                    gc.collect()
                timeout += 1
                if score >= self.threshold:
                    count_next += 1
                    if count_next >= self.count_next_threshold:
                        break
                else:
                    count_next = 0
                if timeout > self.timeout_threshold:
                    print('Timed Out')
                    break
