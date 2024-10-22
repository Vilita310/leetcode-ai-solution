# 1730. Shortest Path to Get Food

### Problem Description 
You are starving and you want to eat food as quickly as possible. You want to find the shortest path to arrive at any food cell.

You are given an `m x n` character matrix, `grid`, of these different types of cells:
`'*'` is your location. There is exactly one `'*'` cell.

`'#'` is a food cell. There may be multiple food cells.

`'O'` is free space, and you can travel through these cells.

`'X'` is an obstacle, and you cannot travel through these cells.

You can travel to any adjacent cell north, east, south, or west of your current location if there is not an obstacle.

Return the length of the shortest path for you to reach any food cell. If there is no path for you to reach food, return `-1`.


Example 1:
Input: grid = [["X","X","X","X","X","X"],["X","*","O","O","O","X"],["X","O","O","#","O","X"],["X","X","X","X","X","X"]]
Output: 3
Explanation: It takes 3 steps to reach the food.


Example 2:
Input: grid = [["X","X","X","X","X"],["X","*","X","O","X"],["X","O","X","#","X"],["X","X","X","X","X"]]
Output: -1
Explanation: It is not possible to reach the food.


Example 3:
Input: grid = [["X","X","X","X","X","X","X","X"],["X","*","O","X","O","#","O","X"],["X","O","O","X","O","O","X","X"],["X","O","O","O","O","#","O","X"],["X","X","X","X","X","X","X","X"]]
Output: 6
Explanation: There can be multiple food cells. It only takes 6 steps to reach the bottom food.


Example 4:
Input: grid = [["O","*"],["#","O"]]
Output: 2

Example 5:
Input: grid = [["X","*"],["#","X"]]
Output: -1

Constraints:
`m == grid.length`
`n == grid[i].length`
`1 <= m, n <= 200`
`grid[row][col]` is `'*'`, `'X'`, `'O'`, or `'#'`.

The `grid` contains exactly one `'*'`.

### Solution 
 To solve the problem of finding the shortest path to any food cell in a given grid, we need to implement a breadth-first search (BFS) algorithm, as it is well-suited for unweighted grid traversal problems like this one.

## Problem Breakdown:
1. **Grid Elements**:
    - `'*'`: Your starting location.
    - `'#'`: Food cell(s).
    - `'O'`: Free space (you can move through these cells).
    - `'X'`: Obstacles (you cannot move through these cells).

2. **Constraints**:
    - You can move up, down, left, or right from your current cell.
    - You can only traverse through `'O'` or `'#'`.
    - The goal is to find the shortest path (in terms of the number of moves) from `'*'` to any `'#'`.

3. **Return Values**:
    - If a path exists to a food cell, return the length of the path.
    - If no path exists, return `-1`.

## Implementation Steps:
1. First, we locate the starting position (`'*'`) in the grid to begin our BFS.
2. We will use a queue to manage our BFS frontier and a set to keep track of visited positions.
3. From the current position, we will explore all four possible directions (up, down, left, right).
4. Each time we encounter a food cell (`'#'`), we record the distance traveled and return it.
5. If we exhaust all possibilities without finding food, we return `-1`.

Here's the code implementation based on the above explanation:



```python
from collections import deque

class Solution:
    def getFood(self, grid):
        # Determine the dimensions of the grid
        m, n = len(grid), len(grid[0])
        
        # Find the starting position (the cell with '*')
        start_x, start_y = -1, -1
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '*':
                    start_x, start_y = i, j
                    break
            if start_x != -1:
                break
        
        if start_x == -1: 
            return -1 # '*' not found which should not happen as per constraints
        
        # BFS setup
        queue = deque([(start_x, start_y, 0)])  # (x, y, distance)
        visited = set((start_x, start_y))
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        
        while queue:
            x, y, distance = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check boundaries and state of the cell
                if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited:
                    if grid[nx][ny] == '#':
                        return distance + 1  # Found a food cell
                    elif grid[nx][ny] == 'O':
                        queue.append((nx, ny, distance + 1))
                        visited.add((nx, ny))
        
        return -1  # No food cell was reachable

# Example usage
solution = Solution()
print(solution.getFood([["X","X","X","X","X","X"],
                         ["X","*","O","O","O","X"],
                         ["X","O","O","#","O","X"],
                         ["X","X","X","X","X","X"]]))  # Output: 3

```

### Explanation of Code:
- We locate the `'*'` in the grid.
- We use a queue to track our position and the distance traveled. The queue keeps tuples of the form `(x, y, distance)`.
- We explore each direction one step at a time, checking if the next cell is within boundary limits and not visited.
- If a food cell (`'#'`) is found, we return the distance incremented by 1, which represents the number of moves taken to reach it.
- If the queue is exhausted without finding food, we return `-1`, indicating that no food cell is reachable.

# 1740. Find Distance in a Binary Tree

### Problem Description 
Given the root of a binary tree and two integers `p` and `q`, return the distance between the nodes of value `p` and value `q` in the tree.

The distance between two nodes is the number of edges on the path from one to the other.


Example 1:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 0
Output: 3
Explanation: There are 3 edges between 5 and 0: 5-3-1-0.


Example 2:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 7
Output: 2
Explanation: There are 2 edges between 5 and 7: 5-2-7.


Example 3:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 5
Output: 0
Explanation: The distance between a node and itself is 0.


Constraints:
The number of nodes in the tree is in the range `[1, 104]`.

`0 <= Node.val <= 109`
All `Node.val` are unique.

`p` and `q` are values in the tree.

### Solution 
 To solve the problem of finding the distance between two nodes in a binary tree, we can break down the solution into several steps:

1. **Find the Lowest Common Ancestor (LCA)** of the two nodes `p` and `q`. The LCA is defined as the deepest node that is an ancestor to both `p` and `q`.
2. **Calculate the distance** from the LCA to each of the target nodes (`p` and `q`) separately.
3. **Sum the distances** from the LCA to `p` and `q` to get the total distance between them.

### Solution Outline:
- We need to traverse the binary tree to find both the LCA and the respective distances to `p` and `q`.
- We will implement a Depth First Search (DFS) strategy to navigate through the tree.

### Python Code:
Here’s how we can implement this in Python using the LeetCode format. We will define a `TreeNode` class to represent each node of the tree and then create our main solution class with the necessary methods.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def findDistance(self, root: TreeNode, p: int, q: int) -> int:
        # Function to find the LCA of p and q
        def findLCA(node):
            if node is None:
                return None
            if node.val == p or node.val == q:
                return node
            
            left = findLCA(node.left)
            right = findLCA(node.right)
            
            if left and right:
                return node
            return left if left else right
        
        # Function to measure the distance from the node to the target value
        def findDistanceFromNode(node, target):
            if node is None:
                return -1
            if node.val == target:
                return 0
            
            leftDistance = findDistanceFromNode(node.left, target)
            rightDistance = findDistanceFromNode(node.right, target)
            
            if leftDistance != -1:
                return leftDistance + 1
            if rightDistance != -1:
                return rightDistance + 1
            return -1
        
        # Step 1: Find the LCA of p and q
        lca = findLCA(root)
        
        # Step 2: Find distances from LCA to p and q
        distance_to_p = findDistanceFromNode(lca, p)
        distance_to_q = findDistanceFromNode(lca, q)
        
        # Step 3: Return the total distance
        return distance_to_p + distance_to_q

# Example Usage:
# Given a binary tree:
#         3
#        / \
#       5   1
#      / \ / \
#     6  2 0  8
#       / \
#      7   4
# The binary tree can be created using TreeNode as shown below

# Create the binary tree
root = TreeNode(3)
root.left = TreeNode(5)
root.right = TreeNode(1)
root.left.left = TreeNode(6)
root.left.right = TreeNode(2)
root.right.left = TreeNode(0)
root.right.right = TreeNode(8)
root.left.right.left = TreeNode(7)
root.left.right.right = TreeNode(4)

# Initialize the solution
solution = Solution()

# Example: the distance between nodes with values 5 and 0
print(solution.findDistance(root, 5, 0))  # Output: 3

# Example: the distance between nodes with values 5 and 7
print(solution.findDistance(root, 5, 7))  # Output: 2

# Example: the distance between the same node
print(solution.findDistance(root, 5, 5))  # Output: 0

```

### Explanation:
1. **TreeNode Class**: This class is used to create nodes of the binary tree.
2. **Solution Class**: Contains methods to find the distance between the two nodes.
   - **findLCA**: This method recursively finds the LCA of nodes with values `p` and `q`.
   - **findDistanceFromNode**: This method calculates the distance from a given node to a target value.
3. **findDistance**: The main method to find the distance between `p` and `q`.
4. We create an instance of the binary tree, then call `findDistance` with different sets of `p` and `q` values to demonstrate the solution.

This code will work on the LeetCode platform as it adheres to the required input and output formats.

# 1766. Tree of Coprimes

### Problem Description 
There is a tree (i.e., a connected, undirected graph that has no cycles) consisting of `n` nodes numbered from `0` to `n - 1` and exactly `n - 1` edges. Each node has a value associated with it, and the root of the tree is node `0`.

To represent this tree, you are given an integer array `nums` and a 2D array `edges`. Each `nums[i]` represents the `ith` node's value, and each `edges[j] = [uj, vj]` represents an edge between nodes `uj` and `vj` in the tree.

Two values `x` and `y` are coprime if `gcd(x, y) == 1` where `gcd(x, y)` is the greatest common divisor of `x` and `y`.

An ancestor of a node `i` is any other node on the shortest path from node `i` to the root. A node is not considered an ancestor of itself.

Return an array `ans` of size `n`, where `ans[i]` is the closest ancestor to node `i` such that `nums[i]` and `nums[ans[i]]` are coprime, or `-1` if there is no such ancestor.


Example 1:
Input: nums = [2,3,3,2], edges = [[0,1],[1,2],[1,3]]
Output: [-1,0,0,1]
Explanation: In the above figure, each node's value is in parentheses.

- Node 0 has no coprime ancestors.

- Node 1 has only one ancestor, node 0. Their values are coprime (gcd(2,3) == 1).

- Node 2 has two ancestors, nodes 1 and 0. Node 1's value is not coprime (gcd(3,3) == 3), but node 0's
  value is (gcd(2,3) == 1), so node 0 is the closest valid ancestor.

- Node 3 has two ancestors, nodes 1 and 0. It is coprime with node 1 (gcd(3,2) == 1), so node 1 is its
  closest valid ancestor.


Example 2:
Input: nums = [5,6,10,2,3,6,15], edges = [[0,1],[0,2],[1,3],[1,4],[2,5],[2,6]]
Output: [-1,0,-1,0,0,0,-1]

Constraints:
`nums.length == n`
`1 <= nums[i] <= 50`
`1 <= n <= 105`
`edges.length == n - 1`
`edges[j].length == 2`
`0 <= uj, vj < n`
`uj != vj`

### Solution 
 To solve the problem, we need to navigate through a tree structure represented by nodes and edges and determine the closest ancestor for each node such that the values at those nodes are coprime.

### Step-by-step Explanation:

1. **Understanding the Tree Structure**: The tree is represented as an undirected graph with nodes (each having a value) and edges. Each node can have multiple ancestors, and we need to check these ancestors to find one that is coprime with the current node's value.

2. **Coprime Check**: Two values are considered coprime if their greatest common divisor (GCD) is 1. We will use the `gcd` function from the `math` module for this.

3. **Graph Representation**: We will represent the tree using an adjacency list since it efficiently stores edges and allows easy traversal during our DFS (Depth First Search).

4. **DFS Traversal**: The algorithm will perform a DFS starting from the root (node 0). During this traversal, we will keep track of the ancestors and their values. For each node, we will check the most recent ancestor that is coprime with the node's value.

5. **Handling Ancestors**: As we traverse down the tree, we need to maintain a stack of ancestors. When moving back up the tree (when recursive DFS returns), we will remove the node from the ancestors list to ensure we are tracking only the relevant ancestors.

### Implementation:

Here's the Python code structured to fit LeetCode's requirements:



```python
from math import gcd
from collections import defaultdict
from typing import List

class Solution:
    def coprimeAncestors(self, nums: List[int], edges: List[List[int]]) -> List[int]:
        n = len(nums)

        # Building the adjacency list for the tree
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        # Initialize the answer array with -1
        ans = [-1] * n

        # DFS function to explore the tree
        def dfs(node: int, parent: int, ancestors: List[int]):
            # Check ancestors for coprimality
            for ancestor in reversed(ancestors):
                if gcd(nums[node], nums[ancestor]) == 1:
                    ans[node] = ancestor
                    break
            
            # Include current node in ancestors and explore children
            ancestors.append(node)
            for neighbor in graph[node]:
                if neighbor != parent:  # Avoid going back to parent
                    dfs(neighbor, node, ancestors)
            ancestors.pop()  # Backtrack

        # Start DFS from the root node 0
        dfs(0, -1, [])

        return ans

```

### Explanation of the Code:
1. **Imports**: We import necessary functions and types from `math` and `collections`.
   
2. **Graph Construction**: The adjacency list is constructed for the edges.

3. **DFS Function**: The recursive function `dfs` does the following:
   - **Ancestor Coprime Check**: Checks each ancestor in reverse order (closest first) for coprimality with the current node's value.
   - **Backtracking Ancestors**: Before diving deeper into the tree, the current node is added to the ancestors. After finishing all children, it is removed from the ancestors list.
4. **Initial Call**: The DFS traversal begins at the root node (0).

5. **Return Result**: Finally, the computed answer array is returned containing the closest coprime ancestors for each node.

This implementation is efficient with a time complexity of O(n) since each node and edge is visited at most twice. It should work well within the problem constraints provided.

# 1778. Shortest Path in a Hidden Grid

### Problem Description 
This is an interactive problem.

There is a robot in a hidden grid, and you are trying to get it from its starting cell to the target cell in this grid. The grid is of size `m x n`, and each cell in the grid is either empty or blocked. It is guaranteed that the starting cell and the target cell are different, and neither of them is blocked.

You want to find the minimum distance to the target cell. However, you do not know the grid's dimensions, the starting cell, nor the target cell. You are only allowed to ask queries to the `GridMaster` object.

Thr `GridMaster` class has the following functions:
`boolean canMove(char direction)` Returns `true` if the robot can move in that direction. Otherwise, it returns `false`.

`void move(char direction)` Moves the robot in that direction. If this move would move the robot to a blocked cell or off the grid, the move will be ignored, and the robot will remain in the same position.

`boolean isTarget()` Returns `true` if the robot is currently on the target cell. Otherwise, it returns `false`.

Note that `direction` in the above functions should be a character from `{'U','D','L','R'}`, representing the directions up, down, left, and right, respectively.

Return the minimum distance between the robot's initial starting cell and the target cell. If there is no valid path between the cells, return `-1`.

Custom testing:
The test input is read as a 2D matrix `grid` of size `m x n` where:
`grid[i][j] == -1` indicates that the robot is in cell `(i, j)` (the starting cell).

`grid[i][j] == 0` indicates that the cell `(i, j)` is blocked.

`grid[i][j] == 1` indicates that the cell `(i, j)` is empty.

`grid[i][j] == 2` indicates that the cell `(i, j)` is the target cell.

There is exactly one `-1` and `2` in `grid`. Remember that you will not have this information in your code.


Example 1:
Input: grid = [[1,2],[-1,0]]
Output: 2
Explanation: One possible interaction is described below:
The robot is initially standing on cell (1, 0), denoted by the -1.

- master.canMove('U') returns true.

- master.canMove('D') returns false.

- master.canMove('L') returns false.

- master.canMove('R') returns false.

- master.move('U') moves the robot to the cell (0, 0).

- master.isTarget() returns false.

- master.canMove('U') returns false.

- master.canMove('D') returns true.

- master.canMove('L') returns false.

- master.canMove('R') returns true.

- master.move('R') moves the robot to the cell (0, 1).

- master.isTarget() returns true. 
We now know that the target is the cell (0, 1), and the shortest path to the target cell is 2.


Example 2:
Input: grid = [[0,0,-1],[1,1,1],[2,0,0]]
Output: 4
Explanation: The minimum distance between the robot and the target cell is 4.


Example 3:
Input: grid = [[-1,0],[0,2]]
Output: -1
Explanation: There is no path from the robot to the target cell.


Constraints:
`1 <= n, m <= 500`
`m == grid.length`
`n == grid[i].length`
`grid[i][j]` is either `-1`, `0`, `1`, or `2`.

There is exactly one `-1` in `grid`.

There is exactly one `2` in `grid`.

### Solution 
 To solve this interactive problem, we will implement a depth-first search (DFS) algorithm to navigate through the grid, starting from the robot's initial position (denoted by `-1`) and trying to find the target position (denoted by `2`). The goal is to find the minimum distance between the robot's initial position and the target position. 

Before we jump into the Python code, let’s break down the approach into steps:

### Step-by-step Approach

1. **Initialize Directions**: Create a list of movement directions that correspond to `Up`, `Down`, `Left`, and `Right`.

2. **Implement DFS**: Use a recursive function to explore all path options from the current position. Each time we make a move to a new cell, we increment the distance.

3. **Check for Target**: After moving to a new cell, we check if the current cell is the target cell (`isTarget` returns `True`). If it is, we can update the minimum distance and backtrack.

4. **Boundary Conditions**: Before moving, we will check:
   - If the move is valid using `canMove`.
   - If the cell has been visited or is blocked, so we do not revisit it.
   
5. **Backtrack**: If we find that a cell does not lead to the target, we will backtrack and mark it as unvisited.

6. **Return the Result**: If the target is found, return the minimum distance; otherwise, return `-1`.

### Python Code

Here is the implementation following the LeetCode solution format:



```python
class Solution:
    def findShortestPath(self, master) -> int:
        # Directions: Up, Down, Left, Right
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        direction_chars = ['R', 'L', 'U', 'D']
        
        # Initialization
        visited = set()  # To keep track of visited positions
        min_distance = float('inf')  # Initialize minimum distance
        current_distance = 0  # Distance traveled so far
        
        # DFS function to explore the grid
        def dfs(r, c):
            nonlocal min_distance, current_distance
            visited.add((r, c))

            if master.isTarget():
                min_distance = min(min_distance, current_distance)
                # Terminate search early if we found the target
                visited.remove((r, c))
                return True
            
            for i in range(4):  # Explore all four directions
                new_r = r + directions[i][0]
                new_c = c + directions[i][1]
                move_dir = direction_chars[i]

                if (new_r, new_c) not in visited and master.canMove(move_dir):
                    master.move(move_dir)  # Move to the new cell
                    current_distance += 1  # Increment path distance

                    # Recursive DFS call
                    if dfs(new_r, new_c):
                        break  # If target is found, break out

                    # Backtrack
                    current_distance -= 1
                    visited.remove((new_r, new_c))  # Unmark the position
                    # Revert the move
                    reverse_dir = {'R': 'L', 'L': 'R', 'U': 'D', 'D': 'U'}
                    master.move(reverse_dir[move_dir])
            
            return False  # If we did not reach the target here

        # Start the DFS from initial position (0, 0), as we do not know the start position
        dfs(0, 0)

        return min_distance if min_distance != float('inf') else -1

```

### Explanation of the Code

- **Directions**: We initialize movement directions `(row_change, column_change)` and their corresponding character representations.

- **Visited Set**: It tracks the cells we've already explored to prevent looping back.

- **DFS Function**: This recursive function explores the grid:
  - If `isTarget()` is true, it updates the minimum distance found.
  - For each direction, it checks if the robot can move, then it moves, increases the current distance, and recursively calls `dfs()`.
  - If the cell does not lead to finding the target, it backtracks (reverts the move and decreases the distance).
  
- **Returning the Result**: At the end of the traversal, we check if we found the target or not and return the appropriate result.

### Conclusion

This code implements a depth-first search approach that effectively explores the grid, respects the movement constraints, and returns the shortest path to the target if reachable, complying with the interactive nature of the problem constraints.

# 1810. Minimum Path Cost in a Hidden Grid

### Problem Description 
This is an interactive problem.

There is a robot in a hidden grid, and you are trying to get it from its starting cell to the target cell in this grid. The grid is of size `m x n`, and each cell in the grid is either empty or blocked. It is guaranteed that the starting cell and the target cell are different, and neither of them is blocked.

Each cell has a cost that you need to pay each time you move to the cell. The starting cell's cost is not applied before the robot moves.

You want to find the minimum total cost to move the robot to the target cell. However, you do not know the grid's dimensions, the starting cell, nor the target cell. You are only allowed to ask queries to the `GridMaster` object.

The `GridMaster` class has the following functions:
`boolean canMove(char direction)` Returns `true` if the robot can move in that direction. Otherwise, it returns `false`.

`int move(char direction)` Moves the robot in that direction and returns the cost of moving to that cell. If this move would move the robot to a blocked cell or off the grid, the move will be ignored, the robot will remain in the same position, and the function will return `-1`.

`boolean isTarget()` Returns `true` if the robot is currently on the target cell. Otherwise, it returns `false`.

Note that `direction` in the above functions should be a character from `{'U','D','L','R'}`, representing the directions up, down, left, and right, respectively.

Return the minimum total cost to get the robot from its initial starting cell to the target cell. If there is no valid path between the cells, return `-1`.

Custom testing:
The test input is read as a 2D matrix `grid` of size `m x n` and four integers `r1`, `c1`, `r2`, and `c2` where:
`grid[i][j] == 0` indicates that the cell `(i, j)` is blocked.

`grid[i][j] >= 1` indicates that the cell `(i, j)` is empty and `grid[i][j]` is the cost to move to that cell.

`(r1, c1)` is the starting cell of the robot.

`(r2, c2)` is the target cell of the robot.

Remember that you will not have this information in your code.


Example 1:
Input: grid = [[2,3],[1,1]], r1 = 0, c1 = 1, r2 = 1, c2 = 0
Output: 2
Explanation: One possible interaction is described below:
The robot is initially standing on cell (0, 1), denoted by the 3.

- master.canMove('U') returns false.

- master.canMove('D') returns true.

- master.canMove('L') returns true.

- master.canMove('R') returns false.

- master.move('L') moves the robot to the cell (0, 0) and returns 2.

- master.isTarget() returns false.

- master.canMove('U') returns false.

- master.canMove('D') returns true.

- master.canMove('L') returns false.

- master.canMove('R') returns true.

- master.move('D') moves the robot to the cell (1, 0) and returns 1.

- master.isTarget() returns true.

- master.move('L') doesn't move the robot and returns -1.

- master.move('R') moves the robot to the cell (1, 1) and returns 1.

We now know that the target is the cell (0, 1), and the minimum total cost to reach it is 2. 

Example 2:
Input: grid = [[0,3,1],[3,4,2],[1,2,0]], r1 = 2, c1 = 0, r2 = 0, c2 = 2
Output: 9
Explanation: The minimum cost path is (2,0) -> (2,1) -> (1,1) -> (1,2) -> (0,2).


Example 3:
Input: grid = [[1,0],[0,1]], r1 = 0, c1 = 0, r2 = 1, c2 = 1
Output: -1
Explanation: There is no path from the robot to the target cell.


Constraints:
`1 <= n, m <= 100`
`m == grid.length`
`n == grid[i].length`
`0 <= grid[i][j] <= 100`

### Solution 
 To solve the problem of navigating a robot in a hidden grid to reach a target cell with the minimum cost, we will utilize a breadth-first search (BFS) strategy. This allows us to explore all possible paths in a systematic way and find the least costly route to the target cell, while adhering to the rules provided by the `GridMaster` class methods.

### Solution Overview

1. **Initialization**: Start by storing the initial costs to reach the neighboring cells and maintaining a `visited` set to avoid revisiting cells.

2. **BFS Exploration**: Use a queue to implement BFS. For each position, explore all possible directions (up, down, left, right) and make queries to see if a move is possible. If a move is possible, record the cost and add that position to the queue.

3. **Target Check**: After moving to a cell, check if the robot has reached the target cell. If it has, return the accumulated cost.

4. **Handle Blockages**: If a move is not possible (blocked or out of bounds), simply skip that move and continue exploring other directions.

5. **Termination**: If the queue is exhausted and the target has not been reached, return `-1`, indicating no valid path exists.

### Python Code

Here’s how the solution can be implemented in Python, following the format suitable for LeetCode:



```python
class Solution:
    def minimalCost(self, master: 'GridMaster') -> int:
        from collections import deque

        # Directions mapping
        directions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        direction_keys = list(directions.keys())
        
        # BFS setup
        queue = deque()
        visited = set()
        
        # Starting from initial position (0, 0)
        total_cost = 0
        queue.append((0, 0, total_cost))
        visited.add((0, 0))

        while queue:
            x, y, total_cost = queue.popleft()
            
            # Check if the current position is the target
            if master.isTarget():
                return total_cost
            
            # Explore all four directions
            for direction in direction_keys:
                dx, dy = directions[direction]
                new_x, new_y = x + dx, y + dy
                
                # Check if we can move in that direction
                if master.canMove(direction):
                    # Attempt to move
                    cost = master.move(direction)

                    if cost != -1:  # If move was successful
                        if (new_x, new_y) not in visited:  
                            visited.add((new_x, new_y))
                            queue.append((new_x, new_y, total_cost + cost))
                            
                            # If we've reached the target, we can return early
                            if master.isTarget():
                                return total_cost + cost
                    
        # If reached here, it means we can't find a path to the target
        return -1

```

### Explanation

1. **Directions Mapping**: We create a dictionary that maps direction letters ('U', 'D', 'L', 'R') to coordinate changes.

2. **BFS Queue**: We use a deque (double-ended queue) for efficiently popping from the front, along with a set to keep track of visited positions to prevent cycles.

3. **Processing Each Cell**: For each cell, we check if it's the target. If not, we check possible moves. After a successful move, we record the cost and add the new position to the BFS queue.

4. **Cost Accumulation**: We maintain and update the `total_cost` for each step taken, allowing us to return it once we reach the target cell.

5. **Early Termination**: We make additional checks to return as soon as we find the target without needing to traverse the entire graph.

### Runtime Complexity
The complexity is approximately O(V + E) where V is the number of cells visited and E is the number of edges explored (each cell has at most 4 edges). Given constraints, this approach is efficient and practical for the problem size.

