import csv
import sys
from curriculum.maze_trainer import MazeTrainer
from torch import cuda

class CurriculumTrainer:
    def __init__(self,mazes,threshold=0.95):
        self.mazes = mazes

        batch_size = 256
        hidden_size = 256
        nb_layers = 3
        lr = 3e-4

        cpu = not cuda.is_available()
        
        self.mazetrainer = MazeTrainer(self.mazes[0],'maze_baseline',0,False,"her","pointnet",0,1.,hidden_size,batch_size,lr,nb_layers,5e-3,True,0.1,0.8,75,int(1e6),"last",10,cpu)
    
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
                sys.stdout = open('/dev/null','w')
                self.mazetrainer.train(1)
                sys.stdout.close()
                sys.stdout = out
                #Get score
                with open('/root/maze_baseline/seed0/progress.csv', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    l = []
                    for row in reader:
                        l.append(row['evaluation/SuccessRate'])
                    score = eval(l[-1])[0]
                print(c,score)
                c += 1
                if score >= self.threshold:
                    count_next += 1
                    if count_next >= self.count_next_threshold:
                        break
                else:
                    count_next = 0
