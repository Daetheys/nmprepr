# df_maze.py
import random

# Create a maze using the depth-first algorithm described at
# https://scipython.com/blog/making-a-maze/
# Christian Hill, April 2017.


class Cell:
    """A cell in the maze.

    A maze "Cell" is a point in the grid which may be surrounded by walls to
    the north, east, south or west.

    """

    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {"N": "S", "S": "N", "E": "W", "W": "E"}

    def __init__(self, x, y):
        """Initialize the cell at (x,y). At first it is surrounded by walls."""

        self.x, self.y = x, y
        self.walls = {"N": True, "S": True, "E": True, "W": True}

    def has_all_walls(self):
        """Does this cell still have all its walls?"""

        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        """Knock down the wall between cells self and other."""

        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False

    def __repr__(self):
        return f"({self.x}, {self.y})"


class Maze:
    """A Maze, represented as a grid of cells."""
    wall_pairs = {"N": "S", "S": "N", "E": "W", "W": "E"}

    def __init__(self, nx, ny, i0=None):
        """Initialize the maze grid.
        The maze consists of nx x ny cells and will be constructed starting
        at the cell indexed at (ix, iy).

        """

        self.nx, self.ny = nx, ny
        if i0 is None:
          self.ix, self.iy = random.randrange(0, nx), random.randrange(0, ny)
        else:
          self.ix, self.iy = i0
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]

    def cell_at(self, x, y):
        """Return the Cell object at (x,y)."""

        return self.maze_map[x][y]

    def __str__(self):
        """Return a (crude) string representation of the maze."""

        maze_rows = ["-" * self.nx * 2]
        for y in range(self.ny):
            maze_row = ["|"]
            for x in range(self.nx):
                if self.maze_map[x][y].walls["E"]:
                    maze_row.append(" |")
                else:
                    maze_row.append("  ")
            maze_rows.append("".join(maze_row))
            maze_row = ["|"]
            for x in range(self.nx):
                if self.maze_map[x][y].walls["S"]:
                    maze_row.append("-+")
                else:
                    maze_row.append(" +")
            maze_rows.append("".join(maze_row))
        return "\n".join(maze_rows)

    def write_svg(self, filename):
        """Write an SVG image of the maze to filename."""

        aspect_ratio = self.nx / self.ny
        # Pad the maze all around by this amount.
        padding = 10
        # Height and width of the maze image (excluding padding), in pixels
        height = 500
        width = int(height * aspect_ratio)
        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = height / self.ny, width / self.nx

        def write_wall(ww_f, ww_x1, ww_y1, ww_x2, ww_y2):
            """Write a single wall to the SVG image file handle f."""

            print(
                '<line x1="{}" y1="{}" x2="{}" y2="{}"/>'.format(
                    ww_x1, ww_y1, ww_x2, ww_y2
                ),
                file=ww_f,
            )

        # Write the SVG image file for maze
        with open(filename, "w") as f:
            # SVG preamble and styles.
            print('<?xml version="1.0" encoding="utf-8"?>', file=f)
            print('<svg xmlns="http://www.w3.org/2000/svg"', file=f)
            print('    xmlns:xlink="http://www.w3.org/1999/xlink"', file=f)
            print(
                '    width="{:d}" height="{:d}" viewBox="{} {} {} {}">'.format(
                    width + 2 * padding,
                    height + 2 * padding,
                    -padding,
                    -padding,
                    width + 2 * padding,
                    height + 2 * padding,
                ),
                file=f,
            )
            print('<defs>\n<style type="text/css"><![CDATA[', file=f)
            print("line {", file=f)
            print("    stroke: #000000;\n    stroke-linecap: square;", file=f)
            print("    stroke-width: 5;\n}", file=f)
            print("]]></style>\n</defs>", file=f)
            # Draw the "South" and "East" walls of each cell, if present (these
            # are the "North" and "West" walls of a neighbouring cell in
            # general, of course).
            for x in range(self.nx):
                for y in range(self.ny):
                    if self.cell_at(x, y).walls["S"]:
                        x1, y1, x2, y2 = (
                            x * scx,
                            (y + 1) * scy,
                            (x + 1) * scx,
                            (y + 1) * scy,
                        )
                        write_wall(f, x1, y1, x2, y2)
                    if self.cell_at(x, y).walls["E"]:
                        x1, y1, x2, y2 = (
                            (x + 1) * scx,
                            y * scy,
                            (x + 1) * scx,
                            (y + 1) * scy,
                        )
                        write_wall(f, x1, y1, x2, y2)
            # Draw the North and West maze border, which won't have been drawn
            # by the procedure above.
            print('<line x1="0" y1="0" x2="{}" y2="0"/>'.format(width), file=f)
            print('<line x1="0" y1="0" x2="0" y2="{}"/>'.format(height), file=f)
            print("</svg>", file=f)

    def find_valid_neighbours(self, cell):
        """Return a list of unvisited neighbours to cell."""

        delta = [("W", (-1, 0)), ("E", (1, 0)), ("S", (0, 1)), ("N", (0, -1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def find_neighbours(self, cell):
        """Return a list of unvisited neighbours to cell."""

        delta = [("W", (-1, 0)), ("E", (1, 0)), ("S", (0, 1)), ("N", (0, -1))]
        neighbours = []
        gates = []
        for direction, (dx, dy) in delta:
            if not cell.walls[direction]:
                x2, y2 = cell.x + dx, cell.y + dy
                neighbour = self.cell_at(x2, y2)
                neighbours.append(neighbour)
                gates.append(Maze.wall_pairs[direction])
        return neighbours, gates

    def make_maze(self):
        # Total number of cells.
        n = self.nx * self.ny
        cell_stack = []
        current_cell = self.cell_at(self.ix, self.iy)
        # Total number of visited cells during maze construction.
        nv = 1

        while nv < n:
            neighbours = self.find_valid_neighbours(current_cell)

            if not neighbours:
                # We've reached a dead end: backtrack.
                current_cell = cell_stack.pop()
                continue

            # Choose a random neighbouring cell and move to it.
            direction, next_cell = random.choice(neighbours)
            current_cell.knock_down_wall(next_cell, direction)
            cell_stack.append(current_cell)
            current_cell = next_cell
            nv += 1

    def bfs(self, x_0, y_0):
        BFS = [self.cell_at(x_0, y_0)]
        i = 0
        n = self.nx * self.ny
        visited = [[False] * self.nx for _ in range(self.ny)]
        visited[y_0][x_0] = True
        while i < len(BFS):
            cur_cell = BFS[i]
            neighbours = self.find_neighbours(cur_cell)
            neighbours2 = [n for n in neighbours if not visited[n.y][n.x]]
            BFS = BFS + neighbours2
            for c in neighbours:
                visited[c.y][c.x] = True
            i += 1
        return BFS

    def depth_bfs(self, x_0, y_0):
        BFS = [self.cell_at(x_0, y_0)]
        depth = [0]
        gates = [None]
        d_max = 0

        i = 0

        visited = [[False] * self.nx for _ in range(self.ny)]
        visited[y_0][x_0] = True

        while i < len(BFS):
            cur_cell = BFS[i]
            d = depth[i]

            neighbours, neighbours_gates = self.find_neighbours(cur_cell)
            neighbours2 = [n for n in neighbours if not visited[n.y][n.x]]
            neighbours_gates = [neighbours_gates[i] for i in range(len(neighbours))
                                                    if not visited[neighbours[i].y][neighbours[i].x]]

            BFS = BFS + neighbours2
            gates = gates + neighbours_gates
            depth = depth + [d+1]*len(neighbours2)

            for c in neighbours2:
                visited[c.y][c.x] = True

            i += 1
            if len(neighbours2) != 0:
                d_max = max(d+1, d_max)
        return BFS, depth, d_max, gates

if __name__=='__main__':
    m = Maze(10,10)
    m.make_maze()
    print(m)
    b, d, d_max = m.depth_bfs(5,5)
    print(b)
    print(d)
    print(d_max)
