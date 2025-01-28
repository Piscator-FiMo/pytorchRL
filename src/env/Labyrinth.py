from enum import Enum
import random
import sys
from typing import List, Optional, Self, Tuple

import numpy as np
import matplotlib.pyplot as plt

# (0,0) is the top left corner


class Orientation(Enum):
    # North / Up
    NORTH = (0, -1)
    # East / Right
    EAST = (1, 0)
    # South / Down
    SOUTH = (0, 1)
    # West / Left
    WEST = (-1, 0)

    def opposite(self):
        opposites = {
            Orientation.NORTH: Orientation.SOUTH,
            Orientation.EAST: Orientation.WEST,
            Orientation.SOUTH: Orientation.NORTH,
            Orientation.WEST: Orientation.EAST
        }
        return opposites[self]

    @staticmethod
    def get_shuffled(rnd: random.Random):
        values = list(Orientation)
        rnd.shuffle(values)
        return values


class Tile:

    def __init__(self, x, y, value='#') -> None:
        self.x = x
        self.y = y
        self.position = (x, y)
        self.value = value

    def get_neighbour(self, orientation: Orientation) -> Tuple[int, int]:
        x, y = self.position
        dx, dy = orientation.value
        return x + dx, y + dy

    def get_diagonal_neighbour(self, a: Orientation, b: Orientation) -> Tuple[int, int]:
        x, y = self.position
        if a == b or a == b.opposite():
            return x, y
        dx_a, dy_a = a.value
        dx_b, dy_b = b.value
        return x + dx_a + dx_b, y + dy_a + dy_b

    def get_all_neighbours_oriented(self, orientation: Orientation) -> List[Tuple[int, int]]:
        neighbours = []
        for orient in Orientation:
            if orient != orientation.opposite():
                neighbours.append(self.get_neighbour(orient))
                if orient != orientation:
                    neighbours.append(
                        self.get_diagonal_neighbour(orientation, orient))

        return neighbours

    def __str__(self) -> str:
        return f"{self.position} {self.value}"

    def __repr__(self) -> str:
        return self.__str__()


class Labyrinth:

    def __init__(self, columns: int = 10, rows: int = 10, seed: Optional[int] = None) -> None:
        self.rnd = random.Random(seed)
        self.columns = columns
        self.rows = rows
        self.tiles = [[Tile(x, y, '#') for x in range(columns)] for y in range(rows)]
        self.end = None
        self.start = None
        self._generate()

    def get_tile_at(self, x, y) -> Tile:
        if self._is_out_of_bounds(x, y):
            return Tile(x, y, '#')
        return self.tiles[y][x]

    def is_wall_at(self, x, y) -> bool:
        return self._is_out_of_bounds(x, y) or self.get_tile_at(x, y).value == '#'

    def _is_out_of_bounds(self, x, y) -> bool:
        return not (x in range(0, self.columns) and y in range(0, self.rows))

    def get_array(self):
        return [[cell.value for cell in row] for row in self.tiles]

    def regenerate_start(self):
        if self.start is not None:
            self.start.value = '.'

        x = self.rnd.randint(0, self.columns)
        y = self.rnd.randint(0, self.rows)
        while self.is_wall_at(x, y) or self.get_tile_at(x, y) == self.end:
            x = self.rnd.randint(0, self.columns)
            y = self.rnd.randint(0, self.rows)
        self.start = self.get_tile_at(x, y)
        self.start.value = 'S'

    def _generate(self):
        # Define the exit
        x = self.rnd.randint(1, self.columns - 1)
        y = self.rnd.randint(1, self.rows - 1)
        end = self.get_tile_at(x, y)

        # Generate the labyrinth from the exit
        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)
        self._recursion(end)
        sys.setrecursionlimit(limit)

        # mark the exit tile
        end.value = 'E'
        self.end = end

        # Generate a random start
        self.regenerate_start()

    def _recursion(self, tile: Tile):
        tile.value = '.'
        for orientation in Orientation.get_shuffled(self.rnd):
            next_tile = tile.get_neighbour(orientation)
            if self._is_out_of_bounds(next_tile[0], next_tile[1]):
                break
            next_tile = self.tiles[next_tile[1]][next_tile[0]]
            if next_tile is not None:
                neighbours = next_tile.get_all_neighbours_oriented(orientation)
                usable = True
                for neighbour in neighbours:
                    neighbour = self.get_tile_at(neighbour[0], neighbour[1])
                    if neighbour.value != '#':
                        usable = False
                        break

                if usable:
                    self._recursion(next_tile)


def convert(symbol):
    match symbol:
        case '#':
            return 1
        case '.':
            return 0
        case 'S':
            return 2
        case 'E':
            return 3


if __name__ == '__main__':
    def render(rows):
        rows = [[convert(cell) for cell in row] for row in rows]
        plt.imshow(np.array(rows), interpolation="nearest", origin="upper")
        plt.axis('off')
        plt.show()
    l = Labyrinth(15, 25)
    [print(row) for row in l.tiles]
    render(l.get_array())
    l.regenerate_start()
    render(l.get_array())
