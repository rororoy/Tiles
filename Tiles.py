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

class TilesProblem:
    """
    A class to represent the Tiles puzzle problem with all necessary components
    for problem solving including state space, actions, transition model, and cost function.
    """
    
    def __init__(self, initial_state: List[int]):
        """
        Initialize the Tiles problem with the given initial state.
        
        Args:
            initial_state: A list of 9 integers representing the initial board state
                          where 0 represents the empty tile
        """
        # Convert initial state to numpy array with dtype uint8
        self.initial_state = np.array(initial_state, dtype=np.uint8).reshape(3, 3)
        
        # Define the target state (goal state)
        # Target: tiles 1-8 in order with empty tile (0) in top-left corner
        self.target_state = np.array([
            [0, 1, 2],
            [3, 4, 5], 
            [6, 7, 8]
        ], dtype=np.uint8)
        
        # Define possible actions (directions the empty tile can move)
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
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
            state: Current board state as 3x3 numpy array
            action: Action to apply ('UP', 'DOWN', 'LEFT', 'RIGHT')
            
        Returns:
            New state after applying the action
        """
        # Create a copy of the state to avoid modifying the original
        new_state = state.copy()
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
            Board state as 3x3 numpy array
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
        print("Current state:")
        for row in state:
            print(" ".join(str(cell) if cell != 0 else " " for cell in row))
        print()



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
        