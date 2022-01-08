import csv
import sys
from curriculum.maze_trainer import MazeTrainer
from torch import cuda

class CurriculumTrainer:
    def __init__(self,
                 mazes,
                 batch_size=128,
                 hidden_dim=256,
                 nb_layers=12,
                 lr=3e-4,
                 threshold=0.90):
        self.mazes = mazes

        cpu = not cuda.is_available()

        arguments = dict(env_name=self.mazes[0],
                    exp_dir='maze_baseline',
                    seed=0,
                    resume=False,
                    mode="her",
                    archi="pointnet",
                    epochs=0,
                    reward_scale=1.,
                    hidden_dim=hidden_size,
                    batch_size=batch_size,
                    learning_rate=lr,
                    n_layers=nb_layers,
                    soft_target_tau=5e-3,
                    auto_alpha=True,
                    alpha=0.1,
                    frac_goal_replay=0.8,
                    horizon=75,
                    replay_buffer_size=int(1e6),
                    snapshot_mode="last",
                    snapshot_gap=10,
                    cpu=cpu
                    )

        self.mazetrainer = MazeTrainer(**args)

        self.threshold = threshold

        self.count_next_threshold = 3

    def train(self):
        for m in self.mazes:
            print('----------------------------')
            print('           ',m)
            print('----------------------------')
            self.mazetrainer.change_env(m)
            c = 0
            count_next = 0
            while True:
                out = sys.stdout
                try:
                    sys.stdout = open('/dev/null','w')
                    self.mazetrainer.train(1)
                    sys.stdout.close()
                    sys.stdout = out
                except:
                    sys.stdout = out
                #Get score
                with open('/root/maze_baseline/seed0/progress.csv', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    l = []
                    for row in reader:
                        l.append(row['evaluation/SuccessRate'])
                    score = eval(l[-1])[0]
                print(c,"{:.2f}%".format(100 *score))
                c += 1
                if score >= self.threshold:
                    count_next += 1
                    if count_next >= self.count_next_threshold:
                        break
                else:
                    count_next = 0
