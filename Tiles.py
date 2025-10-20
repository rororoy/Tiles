""" Roy Lavrov - 322492059 """
""" This is an implementation of the Tiles game in Python where the player can move tiles ordered from 1 to 8
around the 3x3 board where one tile is missing which allows the player to move around the tiles by sliding 
them up, down, left and right to reach the state of a solved puzzle where where the tiles are ordered from
1 to 8 where the empty tile appears in the top left corner. 
This code splits into 4 parts:

1. An implementation of all the components of the problem solving process using a class:
    - State space (Using numpy array ndarray with dtype = uint8)
    - Initial state
    - Target state
    - Actions (Left, Right, Up, Down)
    - Transition model
    - Cost Function

2. A heuristic function to guide the search for an informed solution using the A* algorithm

3. An implementation of the BFS and A* algorithms to solve the puzzle

4. A main function to test the implementation of the algorithms

INPUT: e.g Tiles.py 1 4 0 5 8 2 3 6 7
0 represents the empty tile

OUTPUT (Example for BFS): 
Algorithm: BFS
Path: 2 8 5 3 6 7 8 5 4 1 (The path here represnts the sequence of moves to solve the puzzle, notice we dont need to specify the direction the tile moved)
Length: 10
Expanded: 357
"""

"""Part 1: Implementation of the problem solving process using a class"""

import numpy as np
import sys
from typing import List, Tuple, Optional, Set
from collections import deque
import heapq
from collections import deque

class TilesProblem:
    """
    A class to represent the Tiles puzzle problem with all necessary components
    for problem solving including state space, actions, transition model, and cost function.
    
    The state space uses numpy arrays with dtype=uint8 for efficient memory usage and
    fast operations. The internal representation is optimized for computational efficiency
    rather than human-readable display.
    """
    
    def __init__(self, initial_state: List[int]):
        """
        Initialize the Tiles problem with the given initial state.
        
        Args:
            initial_state: A list of 9 integers representing the initial board state
                          where 0 represents the empty tile
        """
        # Convert initial state to numpy array with dtype uint8 for efficient storage
        self.initial_state = np.array(initial_state, dtype=np.uint8).reshape(3, 3)
        
        # Define the target state (goal state) as numpy array with uint8 dtype
        # Target: tiles 1-8 in order with empty tile (0) in top-left corner
        self.target_state = np.array([
            [0, 1, 2],
            [3, 4, 5], 
            [6, 7, 8]
        ], dtype=np.uint8)
        
        # Define possible actions (directions the empty tile can move)
        self.actions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        
        # Action to direction mapping
        self.action_directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
    
    def get_empty_position(self, state: np.ndarray) -> Tuple[int, int]:
        """
        Find the position of the empty tile (0) in the current state.
        
        Args:
            state: Current board state as 3x3 numpy array
            
        Returns:
            Tuple of (row, col) coordinates of the empty tile
        """
        empty_pos = np.where(state == 0)
        return (empty_pos[0][0], empty_pos[1][0])
    
    def is_valid_position(self, row: int, col: int) -> bool:
        """
        Check if the given position is within the 3x3 board bounds.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            True if position is valid, False otherwise
        """
        return 0 <= row < 3 and 0 <= col < 3
    
    def get_possible_actions(self, state: np.ndarray) -> List[str]:
        """
        Get all possible actions (moves) from the current state.
        
        Args:
            state: Current board state as 3x3 numpy array
            
        Returns:
            List of valid action strings
        """
        empty_row, empty_col = self.get_empty_position(state)
        possible_actions = []
        
        for action, (dr, dc) in self.action_directions.items():
            new_row, new_col = empty_row + dr, empty_col + dc
            if self.is_valid_position(new_row, new_col):
                possible_actions.append(action)
        
        return possible_actions
    
    def apply_action(self, state: np.ndarray, action: str) -> np.ndarray:
        """
        Apply the given action to the current state and return the new state.
        
        Args:
            state: Current board state as 3x3 numpy array with dtype uint8
            action: Action to apply ('UP', 'DOWN', 'LEFT', 'RIGHT')
            
        Returns:
            New state after applying the action (maintains uint8 dtype)
        """
        # Create a copy of the state to avoid modifying the original
        # Ensure the copy maintains the uint8 dtype for efficiency
        new_state = state.copy().astype(np.uint8)
        empty_row, empty_col = self.get_empty_position(state)
        
        # Get the direction for this action
        dr, dc = self.action_directions[action]
        new_row, new_col = empty_row + dr, empty_col + dc
        
        # Swap the empty tile with the tile in the target position
        new_state[empty_row, empty_col], new_state[new_row, new_col] = \
            new_state[new_row, new_col], new_state[empty_row, empty_col]
        
        return new_state
    
    def is_goal_state(self, state: np.ndarray) -> bool:
        """
        Check if the given state is the goal state.
        
        Args:
            state: Current board state as 3x3 numpy array
            
        Returns:
            True if state matches the target state, False otherwise
        """
        return np.array_equal(state, self.target_state)
    
    def state_to_string(self, state: np.ndarray) -> str:
        """
        Convert a state to a string representation for hashing and comparison.
        
        Args:
            state: Board state as 3x3 numpy array
            
        Returns:
            String representation of the state
        """
        return ''.join(map(str, state.flatten()))
    
    def string_to_state(self, state_str: str) -> np.ndarray:
        """
        Convert a string representation back to a numpy array state.
        
        Args:
            state_str: String representation of the state
            
        Returns:
            Board state as 3x3 numpy array with dtype uint8
        """
        return np.array(list(map(int, state_str)), dtype=np.uint8).reshape(3, 3)
    
    def cost(self) -> int:
        """Cost of any action in this problem (always 1)."""
        return 1
    
    def print_state(self, state: np.ndarray):
        """
        Print the current state in a readable format.
        
        Args:
            state: Board state as 3x3 numpy array
        """
        for row in state:
            print(" ".join(str(cell) if cell != 0 else " " for cell in row))
        print()

    def get_moved_tile(self, state: np.ndarray, action: str) -> int:
        """
        Get the tile number that moves when applying the given action.
        
        The tile that moves is the one that swaps positions with the blank.
        Since the action describes where the blank moves, the tile is at
        that target position BEFORE the move happens.
        
        Args:
            state: Current board state
            action: Action to apply (direction blank will move)
            
        Returns:
            The number of the tile that will move (integer)
        
        Example:
            State: [[1,4,0], [5,8,2], [3,6,7]]
            Action: 'LEFT'
            Returns: 4 (the tile at position (0,1) that will swap with blank)
        """
        # Find where the blank currently is
        empty_row, empty_col = self.get_empty_position(state)
        
        # Get the direction the blank will move
        dr, dc = self.action_directions[action]
        
        # Calculate where the blank will move TO
        # The tile at that position is the one that's moving
        tile_row, tile_col = empty_row + dr, empty_col + dc
        
        # Return the tile number at that position
        return int(state[tile_row, tile_col])



class Heuristic:
    """Represents the heuristic function for the Tiles puzzle problem."""
    
    def __init__(self, problem: TilesProblem):
        """
        Initialize the heuristic with the problem instance.
        
        Args:
            problem: TilesProblem instance containing the target state
        """
        self.problem = problem
        self.target_state = problem.target_state 

    def heuristic_manhatten_distance(self, state: np.ndarray) -> int:
        """
        This function calculates the Manhattan distance between the current state and the goal state.
        """
        distance = 0
        for i in range(3):
            for j in range(3):
                tile = state[i, j]
                if tile != 0:
                    # Find where this tile should be in the target state
                    goal_pos = np.where(self.target_state == tile)
                    goal_row, goal_col = goal_pos[0][0], goal_pos[1][0]
                    distance += abs(i - goal_row) + abs(j - goal_col)
        return distance

    def heuristic_linear_conflicts(self, state: np.ndarray) -> int:
        """
        Count linear conflicts: pairs of tiles in the same row/column that are
        in their goal row/column but in reversed relative order.
        """
        conflicts = 0
        
        # ROW conflicts
        for row in range(3):
            row_tiles = []  # Store (tile_value, current_col, goal_col)
            
            for col in range(3):
                tile = state[row, col]
                if tile != 0:
                    # Find where this tile should be in the goal
                    goal_pos = np.where(self.target_state == tile)
                    goal_row = goal_pos[0][0]
                    goal_col = goal_pos[1][0]
                    
                    # Only consider tiles that belong to this row in the goal
                    if goal_row == row:
                        row_tiles.append((tile, col, goal_col))
            
            # Check all pairs for conflicts
            for i in range(len(row_tiles)):
                for j in range(i + 1, len(row_tiles)):
                    tile_i, col_i, goal_col_i = row_tiles[i]
                    tile_j, col_j, goal_col_j = row_tiles[j]
                    
                    # Conflict: i is LEFT of j, but i should be RIGHT of j
                    if col_i < col_j and goal_col_i > goal_col_j:
                        conflicts += 1
        
        # COLUMN conflicts
        for col in range(3):
            col_tiles = []  # Store (tile_value, current_row, goal_row)
            
            for row in range(3):
                tile = state[row, col]
                if tile != 0:
                    goal_pos = np.where(self.target_state == tile)
                    goal_row = goal_pos[0][0]
                    goal_col = goal_pos[1][0]
                    
                    # Only consider tiles that belong to this column in the goal
                    if goal_col == col:
                        col_tiles.append((tile, row, goal_row))
            
            # Check all pairs for conflicts
            for i in range(len(col_tiles)):
                for j in range(i + 1, len(col_tiles)):
                    tile_i, row_i, goal_row_i = col_tiles[i]
                    tile_j, row_j, goal_row_j = col_tiles[j]
                    
                    # Conflict: i is ABOVE j, but i should be BELOW j
                    if row_i < row_j and goal_row_i > goal_row_j:
                        conflicts += 1
        
        return conflicts

    def heuristic(self, state: np.ndarray) -> int:
        """
        The full heuristic implementation for the problem: 
        h(n) = Manhattan Distance + 2 × Linear Conflicts
        """
        return self.heuristic_manhatten_distance(state) + (2 * self.heuristic_linear_conflicts(state))

class SearchAlgorithms:
    def __init__(self, problem: TilesProblem):
        """
        Initialize the SearchAlgorithms class with a TilesProblem instance.
        
        Args:
            problem: TilesProblem instance containing the puzzle state and methods
        """
        self.problem = problem
        self.heuristic = Heuristic(problem)
    
    def bfs(self):
        # Initialize BFS frontier with tuples: (state, path, cost)
        initial_state = self.problem.initial_state
        frontier = deque([(initial_state, [], 0)])
        
        # Track visited states using their string keys for efficiency
        reached = {self.problem.state_to_string(initial_state)}

        expanded = 0
        
        # Start BFS loop
        while frontier:
            current, path, cost = frontier.popleft()

            if self.problem.is_goal_state(current):
                return path, len(path), expanded
            
            expanded += 1

            # Explore neighbors
            for action in self.problem.get_possible_actions(current):
                child_state = self.problem.apply_action(current, action)
                child_key = self.problem.state_to_string(child_state)

                if child_key not in reached:
                    reached.add(child_key)

                    # Record the moved tile for path tracking
                    moved_tile = self.problem.get_moved_tile(current, action)
                    frontier.append((
                        child_state, 
                        path + [moved_tile],
                        cost + self.problem.cost()
                    ))

        return None, 0, expanded

            

def main():
    """
    Main function to run the Tiles puzzle with command line arguments.
    Usage: python Tiles.py 1 4 0 5 8 2 3 6 7
    """
    # Check if correct number of arguments provided
    if len(sys.argv) != 10:
        print("Error: Please provide exactly 9 numbers (including 0 for empty tile)")
        print("Usage: python Tiles.py 1 4 0 5 8 2 3 6 7")
        print("Example: python Tiles.py 1 4 0 5 8 2 3 6 7")
        sys.exit(1)
    
    try:
        # Parse command line arguments (skip script name)
        initial_state = [int(x) for x in sys.argv[1:]]
        
        # Validate that we have exactly 9 numbers
        if len(initial_state) != 9:
            print("Error: Must provide exactly 9 numbers")
            sys.exit(1)
        
        # Validate that numbers are in range 0-8 and each appears exactly once
        if set(initial_state) != set(range(9)):
            print("Error: Must include each number from 0 to 8 exactly once")
            sys.exit(1)
        
        # Create the TilesProblem instance
        problem = TilesProblem(initial_state)
        
        # Create heuristic instance
        heuristic = Heuristic(problem)
        
        # Check if already solved
        if problem.is_goal_state(problem.initial_state):
            print("\nThe puzzle is already solved!")
            return
        
    except ValueError as e:
        print(f"Error: Invalid input - all arguments must be integers")
        print("Usage: python Tiles.py 1 4 0 5 8 2 3 6 7")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)



    algorithm = SearchAlgorithms(problem)
    path, path_length, expanded_nodes = algorithm.bfs()

    print("")
    print("Algorithm: BFS")
    if path is None:
        print("No solution")
        print("Expanded: " + str(expanded_nodes))
        return
    print("Path: " + ' '.join(map(str, path)))
    print("Length: " + str(path_length))
    print("Expanded: " + str(expanded_nodes))


if __name__ == "__main__":
    main()
