class Agent:
    def __init__(self, x, y, environment):
        self.x = x  # Current x-coordinate
        self.y = y  # Current y-coordinate
        self.environment = environment  # The grid representing the environment

    def sense(self):
        """Return the surrounding cells the agent can sense (adjacent cells)."""
        surrounding = []
        # Check up, down, left, right
        if self.x > 0: surrounding.append((self.x - 1, self.y))  # Up
        if self.x < len(self.environment) - 1: surrounding.append((self.x + 1, self.y))  # Down
        if self.y > 0: surrounding.append((self.x, self.y - 1))  # Left
        if self.y < len(self.environment[0]) - 1: surrounding.append((self.x, self.y + 1))  # Right
        return surrounding

    def move(self, new_x, new_y):
        """Move the agent to a new position."""
        if self.environment[new_x][new_y] != 1:  # 1 is an obstacle, so the agent can't move here
            self.x = new_x
            self.y = new_y
        else:
            print("Cannot move to an obstacle!")
def simulate(agent):
    target_found = False
    while not target_found:
        # Sense the environment
        neighbors = agent.sense()

        # Simple decision: just move to the first available neighboring cell
        for neighbor in neighbors:
            x, y = neighbor
            if agent.environment[x][y] == 'T':  # Target found
                target_found = True
                print(f"Target found at position ({x}, {y})!")
                break
            agent.move(x, y)  # Move to an open space
            print(f"Agent moved to ({x}, {y})")

# Initialize the environment (grid) and agent
environment = [
    [0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 1, 'T'],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

agent = Agent(0, 0, environment)
simulate(agent)

from collections import deque


def bfs(environment, start, target):
    """Breadth-First Search to find the shortest path from start to target."""
    rows, cols = len(environment), len(environment[0])
    queue = deque([(start, [])])  # Queue of (position, path_taken)
    visited = set([start])

    while queue:
        (x, y), path = queue.popleft()

        # Check if we've reached the target
        if environment[x][y] == target:
            return path + [(x, y)]

        # Explore neighboring cells (up, down, left, right)
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in neighbors:
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited and environment[nx][ny] != 1:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(x, y)]))

    return []  # No path found


# Example usage
start = (0, 0)
target = 'T'
path = bfs(environment, start, target)
print(f"Path to target: {path}")
