import csv
import sys
from curriculum.maze_trainer import MazeTrainer
from torch import cuda
from colabgymrender.recorder import Recorder
import gym
import copy

class CurriculumTrainer:
    def __init__(self,
                 mazes,
                 batch_size=128,
                 hidden_dim=256,
                 nb_layers=12,
                 lr=3e-4,
                 threshold=0.90,
                 count_next_threshold=1,
                 num_expl_steps_per_train_loop=3333,
                 num_eval_steps_per_epoch = 500,
                 min_num_steps_before_training=10000,
                 num_trains_per_train_loop=500,
                 replay_buffer_size=int(1e6),
                 frac_goal_replay=0.8,
                 n_viz_path=None
                 ):
        self.mazes = mazes

        cpu = not cuda.is_available()

        args = dict(env_name=self.mazes[0],
                    exp_dir='maze_baseline',
                    seed=0,
                    resume=False,
                    mode="her",
                    archi="pointnet",
                    epochs=0,
                    reward_scale=1.,
                    hidden_dim=hidden_dim,
                    batch_size=batch_size,
                    learning_rate=lr,
                    n_layers=nb_layers,
                    soft_target_tau=5e-3,
                    auto_alpha=True,
                    alpha=0.1,
                    frac_goal_replay=frac_goal_replay,
                    horizon=75,
                    replay_buffer_size=replay_buffer_size,
                    snapshot_mode="last",
                    snapshot_gap=10,
                    cpu=cpu,
                    num_expl_steps_per_train_loop=num_expl_steps_per_train_loop,
                    num_eval_steps_per_epoch=num_eval_steps_per_epoch,
                    min_num_steps_before_training=min_num_steps_before_training,
                    num_trains_per_train_loop=num_trains_per_train_loop
                    )

        self.mazetrainer = MazeTrainer(**args)

        self.threshold = threshold

        self.count_next_threshold = count_next_threshold

        self.n_viz_path = n_viz_path

    def train(self):
        for m in self.mazes:
            print('----------------------------')
            print('           ',m)
            print('----------------------------')
            self.mazetrainer.change_env(m)
            c = 0
            count_next = 0

            if self.n_viz_path is not None:
                visualization_env = gym.make(m)
                visualization_env = Recorder(visualization_env, '/content/videos/'+m, fps=30)

            while True:
                out = sys.stdout
                try:
                    sys.stdout = open('/dev/null','w')
                    self.mazetrainer.train(1)
                    sys.stdout.close()
                    sys.stdout = out
                except:
                    sys.stdout = out
                    raise
                #Get score
                with open('/root/maze_baseline/seed0/progress.csv', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    l = []
                    for row in reader:
                        l.append(row['evaluation/SuccessRate'])
                    score = eval(l[-1])[0]
                print(c,"{:.2f}%".format(100 *score))
                c += 1

                # save some paths
                if self.n_viz_path is not None:
                    for i in range(self.n_viz_path):
                        o = visualization_env.reset()
                        done = False
                        path_max=75
                        for i in range(path_max):
                            a = policy.get_action(o['observation'],deterministic=True)
                            o,r,d,_ = visualization_env.step(copy.deepcopy(a[0]))

                if score >= self.threshold:
                    count_next += 1
                    if count_next >= self.count_next_threshold:
                        break
                else:
                    count_next = 0
