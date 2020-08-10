# Based on tutorial by Tech With Tim
# URL: https://www.youtube.com/watch?v=JtiK0DOeI4A

import pygame
import math
from queue import PriorityQueue

WIDTH = 600
WIN = pygame.display.set_mode((600, 600))
pygame.display.set_caption("A* Path Finding Visualization")

RED = (255, 0, 0)  # Node out of the open set
GREEN = (0, 255, 0)  # Node in the open set
WHITE = (255, 255, 255)  # Blank node
BLACK = (0, 0, 0)  # Barrier node
PURPLE = (128, 0, 128)  # Path node
ORANGE = (255, 165, 0)  # Start node
TURQUOISE = (64, 224, 208)  # End node
GREY = (128, 128, 128)  # For drawing grid lines


class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.width = width
        self.total_rows = total_rows

        self.x = row * width
        self.y = col * width

        self.color = WHITE
        self.neighbors = []

    def get_pos(self):
        return self.row, self.col

    def is_close(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_close(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_start(self):
        self.color = ORANGE

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):  # This function updates the neighbors (white nodes) of the node
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():  # Checks down neighbor
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # Checks up neighbor
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # Checks left neighbor
            self.neighbors.append(grid[self.row][self.col - 1])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():  # Checks right neighbor
            self.neighbors.append(grid[self.row][self.col + 1])

    def __lt__(self):
        return False


def a_star(draw, grid, start, end):
    count = 0  # Counts how many times a node is added to the queue to break ties (Same F Score but favor most recent)
    open_set = PriorityQueue()
    open_set.put((0, count, start))  # f score, count, node
    came_from = {}

    # G Score is the current shortest distance to get from the start node to the current node
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0

    # F Score is the addition of g score and f score
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = heur(start.get_pos(), end.get_pos())

    # Helps to check what is in or not in the PriorityQueue
    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # This allows the user to exit this while loop
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)  # Synchronize with the PriorityQueue

        if current == end:
            reconstruct_path(came_from, current, draw)
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1  # Add one because our nodes are 1 unit apart

            if temp_g_score < g_score[neighbor]:  # We check if going to the neighbor results in a shorter path
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = heur(neighbor.get_pos(), end.get_pos()) + temp_g_score

                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()  # We placed it in the open set

        draw()

        if current != start:  # If the node we just looked at (not the start node) will not be added to the open set
            current.make_close()

    return False  # Did not find path


def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()


def heur(p1, p2):
    """
    Heuristic function for the algorithm using manhattan distance.
    Used to estimate the distance from a node to the end (H Score).
    :param p1:
    :param p2:
    :return:
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def make_grid(rows, width):
    """
    Creates the grid data structure with nodes in the grid.
    :param rows:
    :param width:
    :return:
    """
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)
    return grid


def draw_grid(win, rows, width):
    """
    Draws the gridlines using pygame.draw.line() function.
    :param win:
    :param rows:
    :param width:
    :return:
    """
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
    """
    Draws the nodes (with their color and width) and the grid.
    :param win:
    :param grid:
    :param rows:
    :param width:
    :return:
    """
    win.fill(WHITE)
    for row in grid:
        for node in row:
            node.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    """
    Helper function to give us the cursor position.
    :param pos:
    :param rows:
    :param width:
    :return:
    """
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col


def main(win, width):
    ROWS = 50
    grid = make_grid(ROWS, width)  # Creating the grid

    # These refer to the start and end nodes
    start = None
    end = None

    run = True
    started = False

    # Main game loop
    while run:
        draw(win, grid, ROWS, width)  # Draw the nodes and grid for every iteration of the while loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if started:  # Once the algorithm has begun, we do not want the user to be able to change the nodes
                continue
            if pygame.mouse.get_pressed()[0]:  # Left mouse button action - Place start, end, and barrier nodes
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]
                if not start and node != end:  # Checks if the start node has been created; if not, it is created
                    start = node
                    start.make_start()
                elif not end and node != start:  # Checks if the end node has been created; if not, it is created
                    end = node
                    end.make_end()
                elif node != start and node != end:
                    node.make_barrier()
            elif pygame.mouse.get_pressed()[2]:  # Right mouse button action - Remove nodes
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]
                node.reset()  # Removes the barriers and start/end nodes
                if node == start:
                    start = None
                elif node == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)

                    a_star(lambda: draw(win, grid, ROWS, width), grid, start, end)  # lambda is an anonymous function

                if event.key == pygame.K_c:  # Resets the entire grid
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

    pygame.quit()


main(WIN, WIDTH)
