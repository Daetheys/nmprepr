from ast import NodeTransformer
import os
import numpy as np
from gym import spaces
import hppfcl
from numpy.lib.function_base import _delete_dispatcher
import pinocchio as pin
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections

from mpenv.core.mesh import Mesh
from mpenv.envs.base import Base
from torch._C import Value
from mpenv.envs.maze_generator import Maze
from mpenv.envs import utils as envs_utils
from mpenv.envs.utils import ROBOTS_PROPS, random
from mpenv.core import utils
from mpenv.core.geometry import Geometries

from mpenv.observers.robot_links import RobotLinksObserver
from mpenv.observers.point_cloud import PointCloudObserver
from mpenv.observers.ray_tracing import RayTracingObserver
from mpenv.observers.maze import MazeObserver

from mpenv.envs.cst import SPHERE_2D_DIAMETER, SPHERE_2D_RADIUS

class MazeGoal(Base):
    def __init__(self, grid_size, easy=False, coordinate_jitter=False,
                 min_gap=3*SPHERE_2D_DIAMETER, depth_specified=False, depth_min=0, depth_max=0, init_distance=None,
                 min_maze_size = None, max_maze_size=None,collision_reward=-4):
        super().__init__(robot_name="sphere")

        self.thickness = 0.02
        self.grid_size = grid_size

        self.easy = easy
        self.coordinate_jitter = coordinate_jitter
        self.min_gap = min_gap
        self.distance = init_distance # distance from the gate
        self.depth_specified = depth_specified # depth is 0 if we fix the distance
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.min_maze_size = min_maze_size
        self.max_maze_size = max_maze_size

        self.dict_reward['collision'] = collision_reward

        self.robot_name = "sphere"
        self.freeflyer_bounds = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )
        self.robot_props = ROBOTS_PROPS["sphere2d"]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.robot_props["action_dim"],), dtype=np.float32
        )

        self.fig, self.ax, self.pos = None, None, None

        self.maze = None
        self.goal_cell = None
        self.init_cell = None
        self.gate = None

    def _reset(self, idx_env=None, start=None, goal=None):
        model_wrapper = self.model_wrapper
        self.robot = self.add_robot("sphere2d", self.freeflyer_bounds)

        if self.max_maze_size is not None:
          self.grid_size = np.random.randint(self.min_maze_size,
                                             self.max_maze_size + 1)

        depth = None if not self.depth_specified else np.random.randint(self.depth_min, self.depth_max+1)
        self.maze, self.goal_cell, self.init_cell, self.gate = self.make_maze(depth)

        self.geoms, self.idx_env, self.subdiv_x, self.subdiv_y = self.get_obstacles_geoms(idx_env)
        for geom_obj in self.geoms.geom_objs:
            self.add_obstacle(geom_obj, static=True)
        model_wrapper.create_data()

        if not self.depth_specified:
            straight_path = True
            while straight_path:
                self.goal_state = self.random_configuration()
                self.state = self.random_configuration()

                straight_path = self.is_straight_path(self.state, self.goal_state)
        else:
            colliding = True
            straight_path = True
            condition = colliding or straight_path
            i = 0
            while condition:
                q_goal = self.get_random_state_cell(self.goal_cell)
                self.set_goal_state(q_goal)

                if self.distance is not None:
                    q_state = self.get_random_state_near_goal(self.init_cell, self.distance, self.gate)
                else:
                    q_state = self.get_random_state_cell(self.init_cell)

                self.set_state(q_state)

                condition = self.collision_somewhere() or self.is_straight_path(self.state, self.goal_state)
                i += 1
                if i % 25 == 0:
                    return self.observation()
        if start is not None:
            self.set_state(start)
        if goal is not None:
            self.set_goal_state(goal)

        if self.fig:
            plt.close()
        self.fig, self.ax, self.pos = None, None, None

        return self.observation()

    def collision_somewhere(self):
        return self.model_wrapper.collision(self.state.q) or self.model_wrapper.collision(self.goal_state.q)

    def is_straight_path(self, state, goal_state):
        "Filter start and goal with straight path solution"
        if self.easy:
            return False
        straight_path = self.model_wrapper.arange(
            state, goal_state, self.delta_collision_check
        )
        _, collide = self.stopping_configuration(straight_path)
        return not(collide.any())

    def make_maze(self, depth):
        maze = Maze(self.grid_size, self.grid_size)
        maze.make_maze()

        if depth is not None:
          x0, y0 = np.random.randint(maze.nx), np.random.randint(maze.ny)
          bfs, depth_list, d_max, gates = maze.depth_bfs(x0,y0)

          while d_max < depth:
            maze = Maze(self.grid_size, self.grid_size)
            maze.make_maze()
            x0, y0 = np.random.randint(maze.nx), np.random.randint(maze.ny)
            bfs, depth_list, d_max, gates = maze.depth_bfs(x0,y0)

          depth_list = np.array(depth_list) # to make it easier
          idx = np.where(depth_list == depth)[0]
          i = np.random.choice(idx)

          goal_cell = bfs[0]
          init_cell = bfs[i]
          gate = gates[i]

          return maze, goal_cell, init_cell, gate
        else:
            return maze, None, None, None

    def get_obstacles_geoms(self, idx_env):
        np_random = self._np_random
        geom_objs, subdiv_x, subdiv_y = extract_obstacles(self.maze, self.thickness, self.coordinate_jitter, self.min_gap)
        geoms = Geometries(geom_objs)
        return geoms, idx_env, subdiv_x, subdiv_y

    def is_in_cell(self, configuration, cell):
        x, y = cell.x, cell.y
        qx, qy = configuration.q[0], configuration.q[1]
        return self.subdiv_x[x] <= qx <= self.subdiv_x[x+1] and \
               self.subdiv_y[y] <= qy <= self.subdiv_y[y+1]

    def get_random_state_cell(self, cell):
      delta = self.thickness + SPHERE_2D_RADIUS + .005
      q = np.zeros(7)
      q[-1] = 1.

      min_x = self.subdiv_x[cell.x  ] + delta
      max_x = self.subdiv_x[cell.x+1] - delta

      min_y = self.subdiv_y[cell.y  ] + delta
      max_y = self.subdiv_y[cell.y+1] - delta

      q[0] = random(min_x, max_x)
      q[1] = random(min_y, max_y)
      return q

    def get_random_state_near_goal(self, cell, distance, gate):
        q = np.zeros(7)
        q[-1] = 1.
        if gate == 'N':
            min_x = self.subdiv_x[cell.x  ]
            max_x = self.subdiv_x[cell.x+1]

            min_y = self.subdiv_y[cell.y  ]
            max_y = min(self.subdiv_y[cell.y+1],
                        self.subdiv_y[cell.y  ] + distance)
        elif gate == 'S':
            min_x = self.subdiv_x[cell.x  ]
            max_x = self.subdiv_x[cell.x+1]

            min_y = max(self.subdiv_y[cell.y  ],
                        self.subdiv_y[cell.y+1] - distance)
            max_y = self.subdiv_y[cell.y+1]
        elif gate == 'W':
            min_x = self.subdiv_x[cell.x  ]
            max_x = min(self.subdiv_x[cell.x+1],
                        self.subdiv_x[cell.x  ] + distance)

            min_y = self.subdiv_y[cell.y  ]
            max_y = self.subdiv_y[cell.y+1]
        elif gate == 'E':
            min_x = max(self.subdiv_x[cell.x  ],
                        self.subdiv_x[cell.x+1] - distance)
            max_x = self.subdiv_x[cell.x+1]

            min_y = self.subdiv_y[cell.y  ]
            max_y = self.subdiv_y[cell.y+1]
        else:
            min_x = self.subdiv_x[cell.x  ]
            max_x = self.subdiv_x[cell.x+1]

            min_y = self.subdiv_y[cell.y  ]
            max_y = self.subdiv_y[cell.y+1]


        q[0] = random(min_x, max_x)
        q[1] = random(min_y, max_y)
        return q

    def set_eval(self):
        pass

    def render(self, *unused_args, **unused_kwargs):
        if self.fig is None:
            self.init_matplotlib()
            self.pos = self.ax.scatter(
                self.state.q[0], self.state.q[1], color="orange", s=200
            )
        else:
            prev_pos = self.pos.get_offsets().data
            new_pos = self.state.q[:2][None, :]
            x = np.vstack((prev_pos, new_pos))
            self.init_matplotlib()
            self.ax.plot(x[:, 0], x[:, 1], c=(0, 0, 0, 0.5))
            self.pos.set_offsets(new_pos)
        # plt.draw()
        # plt.pause(0.01)
        self.fig.canvas.draw()
        data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        size = int(np.sqrt(data.size // 3))
        data = data.reshape((size, size, 3))

        return data

    def init_matplotlib(self):
        if self.fig is not None:
            self.fig.clf()
            plt.close()

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, aspect="equal")
        ax.set_xlim(0.0 - self.thickness, 1.0 + self.thickness)
        ax.set_ylim(0.0 - self.thickness, 1.0 + self.thickness)
        ax.set_xticks([])
        ax.set_yticks([])

        obstacles = self.geoms.geom_objs
        rects = []
        for i, obst in enumerate(obstacles):
            x, y = obst.placement.translation[:2]
            half_side = obst.geometry.halfSide
            w, h = 2 * half_side[:2]
            rects.append(
                patches.Rectangle(
                    (x - w / 2, y - h / 2), w, h  # (x,y)  # width  # height
                )
            )
        coll = collections.PatchCollection(rects, zorder=1)
        coll.set_alpha(0.6)
        ax.add_collection(coll)

        size = self.robot_props["dist_goal"]
        offsets = np.stack((self.state.q, self.goal_state.q), 0)[:, :2]
        sg = collections.EllipseCollection(
            widths=size,
            heights=size,
            facecolors=[(1, 0, 0, 0.8), (0, 1, 0, 0.8)],
            angles=0,
            units="xy",
            offsets=offsets,
            transOffset=ax.transData,
        )
        ax.add_collection(sg)

        plt.tight_layout()
        self.fig = fig
        self.ax = ax


def generate_subdivision(nb_gaps, min_gap):
    subdiv = np.random.rand(nb_gaps-1)
    subdiv.sort()
    subdiv = np.hstack(([0.], subdiv, [1.]))

    while np.min(np.diff(subdiv)) < min_gap:
      subdiv = np.random.rand(nb_gaps-1)
      subdiv.sort()
      subdiv = np.hstack(([0.], subdiv, [1.]))
    return subdiv

def extract_obstacles(maze, thickness, coordinate_jitter=False, min_gap=3*SPHERE_2D_DIAMETER):
    if coordinate_jitter:
        if min_gap > .9 * 1/maze.nx or min_gap > .9 * 1/maze.ny:
          raise ValueError
        subdivision_x = generate_subdivision(maze.nx, min_gap)
        subdivision_y = generate_subdivision(maze.ny, min_gap)
    else:
        scx = 1/maze.nx
        scy = 1/maze.ny
        subdivision_x, subdivision_y = np.array([i*scx for i in range(maze.nx)]+[1.]), np.array([i*scy for i in range(maze.ny)]+[1.])

    obstacles_coord = []
    for x in range(maze.nx):
        obstacles_coord.append((subdivision_x[x], 0, subdivision_x[x+1], 0))
    for y in range(maze.ny):
        obstacles_coord.append((0, subdivision_y[y], 0, subdivision_y[y+1]))
    # Draw the "South" and "East" walls of each cell, if present (these
    # are the "North" and "West" walls of a neighbouring cell in
    # general, of course).
    for x in range(maze.nx):
        for y in range(maze.ny):
            if maze.cell_at(x, y).walls["S"]:
                x1, y1, x2, y2 = (
                    subdivision_x[x],
                    subdivision_y[y+1],
                    subdivision_x[x+1],
                    subdivision_y[y+1],
                )
                obstacles_coord.append((x1, y1, x2, y2))
            if maze.cell_at(x, y).walls["E"]:
                x1, y1, x2, y2 = (
                    subdivision_x[x+1],
                    subdivision_y[y],
                    subdivision_x[x+1],
                    subdivision_y[y+1],
                )

                obstacles_coord.append((x1, y1, x2, y2))
    obstacles = []
    for i, obst_coord in enumerate(obstacles_coord):
        x1, y1, x2, y2 = obst_coord[0], obst_coord[1], obst_coord[2], obst_coord[3]
        x1 -= thickness / 2
        x2 += thickness / 2
        y1 -= thickness / 2
        y2 += thickness / 2
        box_size = [x2 - x1, y2 - y1, 0.1]
        pos = [(x1 + x2) / 2, (y1 + y2) / 2, 0]
        placement = pin.SE3(np.eye(3), np.array(pos))
        mesh = Mesh(
            name=f"obstacle{i}",
            geometry=hppfcl.Box(*box_size),
            placement=placement,
            color=(0, 0, 1, 0.8),
        )
        obstacles.append(mesh.geom_obj())
    return obstacles, subdivision_x, subdivision_y


def maze_edges(grid_size, easy=True, coordinate_jitter=False,
               min_gap=3*SPHERE_2D_DIAMETER, depth=None, distance=None,
               min_maze_size=None, max_maze_size=None,collision_reward=-4,
               depth_specified=False, depth_min=0, depth_max=0):
    env = MazeGoal(grid_size=grid_size,
                  easy=easy,
                  coordinate_jitter=coordinate_jitter,
                  min_gap=min_gap,
                  init_distance=distance,
                  min_maze_size = min_maze_size,
                  max_maze_size=max_maze_size,
                  collision_reward=collision_reward,
                  depth_specified=depth_specified,
                  depth_min=depth_min,
                  depth_max=depth_max)
    env = MazeObserver(env)
    coordinate_frame = "local"
    env = RobotLinksObserver(env, coordinate_frame)
    return env


def maze_raytracing(n_samples, n_rays):
    env = MazeGoal(grid_size=3)
    visibility_radius = 0.7
    memory_distance = 0.06
    env = RayTracingObserver(env, n_samples, n_rays, visibility_radius, memory_distance)
    coordinate_frame = "local"
    env = RobotLinksObserver(env, coordinate_frame)
    return env
