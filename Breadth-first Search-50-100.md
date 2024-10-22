# 909. Snakes and Ladders

### Problem Description 
On an N x N `board`, the numbers from `1` to `N*N` are written boustrophedonically starting from the bottom left of the board, and alternating direction each row.  For example, for a 6 x 6 board, the numbers are written as follows:
You start on square `1` of the board (which is always in the last row and first column).  Each move, starting from square `x`, consists of the following:
You choose a destination square `S` with number `x+1`, `x+2`, `x+3`, `x+4`, `x+5`, or `x+6`, provided this number is `<= N*N`.

	
(This choice simulates the result of a standard 6-sided die roll: ie., there are always at most 6 destinations, regardless of the size of the board.)
If `S` has a snake or ladder, you move to the destination of that snake or ladder.  Otherwise, you move to `S`.

A board square on row `r` and column `c` has a "snake or ladder" if `board[r][c] != -1`.  The destination of that snake or ladder is `board[r][c]`.

Note that you only take a snake or ladder at most once per move: if the destination to a snake or ladder is the start of another snake or ladder, you do not continue moving.  (For example, if the board is `[[4,-1],[-1,3]]`, and on the first move your destination square is `2`, then you finish your first move at `3`, because you do not continue moving to `4`.)
Return the least number of moves required to reach square N*N.  If it is not possible, return `-1`.


Example 1:
Input: [
[-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1],
[-1,35,-1,-1,13,-1],
[-1,-1,-1,-1,-1,-1],
[-1,15,-1,-1,-1,-1]]
Output: 4
Explanation: 
At the beginning, you start at square 1 [at row 5, column 0].

You decide to move to square 2, and must take the ladder to square 15.

You then decide to move to square 17 (row 3, column 5), and must take the snake to square 13.

You then decide to move to square 14, and must take the ladder to square 35.

You then decide to move to square 36, ending the game.

It can be shown that you need at least 4 moves to reach the N*N-th square, so the answer is 4.

Note:
`2 <= board.length = board[0].length <= 20`
`board[i][j]` is between `1` and `N*N` or is equal to `-1`.

The board square with number `1` has no snake or ladder.

The board square with number `N*N` has no snake or ladder.

### Solution 
 To solve the problem of navigating through a 2D board filled with snakes and ladders, we'll use a breadth-first search (BFS) approach. The BFS will allow us to explore all possible moves from the current square until we reach the end square (N * N) or exhaust all possibilities.

### Detailed Explanation:

1. **Understanding the Board Layout**:
   - The board is filled with numbers from 1 to N*N in a boustrophedon pattern. This means we fill the board in a zigzag manner, starting from the bottom left, alternating direction every row.
   - A valid move is defined as moving to squares numbered from the current square's number plus one to plus six, unless a snake or ladder takes us to another square.

2. **Breadth-First Search (BFS)**:
   - We will use BFS to explore the squares. Starting at square 1, we'll check all reachable squares (1 to 6 numbers ahead).
   - For each reachable square, we will check if it has a snake or ladder. If so, we will move to the destination square instead.
   - We will keep track of the number of moves taken to reach each square to determine the shortest path to square N*N.

3. **Data Structures**:
   - A queue for the BFS.
   - A set or array to keep track of visited squares to avoid cycles and unnecessary re-checks.

4. **Implementation**:
Here's how we can implement this in Python.



```python
from collections import deque

class Solution:
    def snakesAndLadders(self, board: list[list[int]]) -> int:
        N = len(board)
        
        # Function to convert 2D board coordinates to 1D index
        def get_square_number(row, col):
            return (N - row - 1) * N + (col + 1)
        
        # BFS setup
        queue = deque([1])  # Start from square 1
        visited = set([1])  # Mark square 1 as visited
        moves = 0
        
        # Perform BFS
        while queue:
            size = len(queue)
            for _ in range(size):
                current_square = queue.popleft()
                
                # Check if we've reached the end
                if current_square == N * N:
                    return moves
                
                # Explore all possible die rolls (1 to 6)
                for die_roll in range(1, 7):
                    next_square = current_square + die_roll
                    
                    if next_square > N * N:  # Ignore out of bounds
                        continue
                    
                    # Get the row and column from the number
                    row, col = divmod(next_square - 1, N)
                    
                    # Check for snakes or ladders
                    if board[row][col] != -1:
                        next_square = board[row][col]
                    
                    if next_square not in visited:
                        visited.add(next_square)
                        queue.append(next_square)
            
            moves += 1
        
        return -1  # If unreachable, return -1

```

### Explanation of the Code:
1. **get_square_number**: A helper function to convert between the 2D coordinates of the board and the 1D indices.
2. **BFS Initialization**:
   - We initialize a queue starting from square 1 and a set called `visited` to track the squares we’ve already examined.
3. **BFS Loop**:
   - For each square in the queue, we explore each possible dice result (1 to 6).
   - For each potential next square, we check if it exceeds the total number of squares. If it's valid, we then check if there's a snake or ladder (i.e., if the board value is not -1).
   - If a square has not been visited, we add it to the queue and mark it as visited.
   - Each time we complete a level of BFS, we increment the `moves`, indicating we've taken a move.
4. Finally, if we exit the BFS loop without reaching N*N, we return -1. 

This solution is efficient and runs within the constraints provided, making it suitable for typical competitive programming problems.

# 913. Cat and Mouse

### Problem Description 
A game on an undirected graph is played by two players, Mouse and Cat, who alternate turns.

The graph is given as follows: `graph[a]` is a list of all nodes `b` such that `ab` is an edge of the graph.

The mouse starts at node `1` and goes first, the cat starts at node `2` and goes second, and there is a hole at node `0`.

During each player's turn, they must travel along one edge of the graph that meets where they are.  For example, if the Mouse is at node 1, it must travel to any node in `graph[1]`.

Additionally, it is not allowed for the Cat to travel to the Hole (node 0.)
Then, the game can end in three ways:
If ever the Cat occupies the same node as the Mouse, the Cat wins.

If ever the Mouse reaches the Hole, the Mouse wins.

If ever a position is repeated (i.e., the players are in the same position as a previous turn, and it is the same player's turn to move), the game is a draw.

Given a `graph`, and assuming both players play optimally, return
`1` if the mouse wins the game,
`2` if the cat wins the game, or
`0` if the game is a draw.


Example 1:
Input: graph = [[2,5],[3],[0,4,5],[1,4,5],[2,3],[0,2,3]]
Output: 0

Example 2:
Input: graph = [[1,3],[0],[3],[0,2]]
Output: 1

Constraints:
`3 <= graph.length <= 50`
`1 <= graph[i].length < graph.length`
`0 <= graph[i][j] < graph.length`
`graph[i][j] != i`
`graph[i]` is unique.

The mouse and the cat can always move.

### Solution 
 To solve this problem, we can employ a recursive Depth First Search (DFS) approach with memoization. The goal is to determine whether the Mouse (starting at node 1) can win, the Cat (starting at node 2) can win, or if the game will eventually end in a draw.

The key points for our approach are as follows:

1. **Game States**: We need to keep track of the positions of both players (the mouse and the cat) along with the move count to determine the current state of the game.

2. **Winning Conditions**:
   - The Mouse wins if it reaches the hole at node 0.
   - The Cat wins if it catches the Mouse by landing on the same node.
   - If a state is repeated, it leads to a draw.

3. **Memoization**: To avoid recalculating states we have already solved, we will use a dictionary to store the results of our game state evaluations.

4. **Game Flow**: The game alternates turns:
   - Mouse (first to move) tries to reach node 0.
   - Cat tries to catch the Mouse and cannot move to node 0.

5. **DFS**: Perform a search where we iterate through possible moves for each player until we find a winning condition.

Here's the implemented solution:



```python
class Solution:
    def catMouseGame(self, graph: List[List[int]]) -> int:
        # Memoization dictionary for (mouse_position, cat_position, turn)
        memo = {}

        def dfs(mouse, cat, turn):
            # If mouse reaches hole
            if mouse == 0:
                return 1
            # If cat catches the mouse
            if cat == mouse:
                return 2
            # Check memoization
            if (mouse, cat, turn) in memo:
                return memo[(mouse, cat, turn)]

            # Calculate current player moves
            if turn % 2 == 0:  # Mouse's turn
                for next_mouse in graph[mouse]:
                    result = dfs(next_mouse, cat, turn + 1)
                    if result == 1:  # Mouse wins
                        memo[(mouse, cat, turn)] = 1
                        return 1
                    if result == 0:  # Draw, don't return immediately
                        result = 0
            else:  # Cat's turn
                for next_cat in graph[cat]:
                    if next_cat == 0:  # Cat can't move to the hole
                        continue
                    result = dfs(mouse, next_cat, turn + 1)
                    if result == 2:  # Cat wins
                        memo[(mouse, cat, turn)] = 2
                        return 2
                    if result == 0:  # Draw, don't return immediately
                        result = 0

            # If none of the players can win
            memo[(mouse, cat, turn)] = result
            return result

        # Start the game with the mouse at position 1, cat at position 2, and it's mouse's turn (turn=0)
        return dfs(1, 2, 0)


```

### Explanation of the Code:
1. **Function Definition**: We define the main function `catMouseGame`, which takes in the `graph`.

2. **Memoization Structure**: We use a dictionary `memo` to store results keyed by the tuple of current mouse position, cat position, and turn number.

3. **Recursive DFS Function**:
   - Base Cases:
     - If the Mouse reaches node `0`, it returns `1` (Mouse wins).
     - If the Cat catches the Mouse (same node), it returns `2` (Cat wins).
   - We check if the current state has been evaluated before using `memo`.
   - Depending on whose turn it is, the function iterates over the possible moves (for Mouse or Cat) and recurses accordingly, checking the results.
   - If the Mouse can ensure a win, it saves `1` in memo; if there’s a draw, it saves `0`, otherwise `2`.

4. **Initial Call**: We initiate the DFS with the Mouse starting at position `1`, Cat at position `2`, and turn `0` indicating it’s the Mouse’s turn.

This code effectively determines the outcome of the game with optimal strategies for both players by exploring possible states recursively while preventing re-evaluation of the same game states through memoization, thus improving efficiency.

# 934. Shortest Bridge

### Problem Description 
In a given 2D binary array `A`, there are two islands.  (An island is a 4-directionally connected group of `1`s not connected to any other 1s.)
Now, we may change `0`s to `1`s so as to connect the two islands together to form 1 island.

Return the smallest number of `0`s that must be flipped.  (It is guaranteed that the answer is at least 1.)

Example 1:
Input: A = [[0,1],[1,0]]
Output: 1

Example 2:
Input: A = [[0,1,0],[0,0,0],[0,0,1]]
Output: 2

Example 3:
Input: A = [[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]
Output: 1

Constraints:
`2 <= A.length == A[0].length <= 100`
`A[i][j] == 0` or `A[i][j] == 1`

### Solution 
 To solve the problem of finding the smallest number of `0`s that must be flipped to connect two islands in a 2D binary array, we can break down the solution into several key steps. The approach will involve using Depth-First Search (DFS) to find and label the islands, and then calculate the minimum number of flips required to connect them. 

Here's a structured solution that follows the LeetCode format:



```python
class Solution:
    def shortestBridge(self, A: List[List[int]]) -> int:
        from collections import deque

        # Directions for moving up, down, left, right
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        # Step 1: Find the first island and mark it
        def dfs(x, y):
            stack = [(x, y)]
            island_positions = []
            while stack:
                cx, cy = stack.pop()
                island_positions.append((cx, cy))
                A[cx][cy] = 2  # Mark this cell to prevent revisiting
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < len(A) and 0 <= ny < len(A[0]) and A[nx][ny] == 1:
                        stack.append((nx, ny))
            return island_positions
        
        # Step 2: Find the first island and store its cells
        first_island_positions = []
        found_first_island = False
        
        for i in range(len(A)):
            for j in range(len(A[0])):
                if A[i][j] == 1:
                    first_island_positions = dfs(i, j)
                    found_first_island = True
                    break
            if found_first_island:
                break

        # Step 3: Use BFS to find the shortest path to the second island
        queue = deque()
        for x, y in first_island_positions:
            queue.append((x, y, 0))  # (x, y, distance)
        
        while queue:
            x, y, distance = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(A) and 0 <= ny < len(A[0]):
                    if A[nx][ny] == 1:  # Found the second island
                        return distance
                    if A[nx][ny] == 0:  # Continue to expand to the water
                        A[nx][ny] = -1  # Mark as visited
                        queue.append((nx, ny, distance + 1))
        
        return -1  # This line should never be reached since it's guaranteed to connect the islands


```

### Explanation of the Code:
1. **Imports**: We import `deque` from `collections` for efficient queue operations.
  
2. **Define Directions**: We define `directions` for North, South, East, and West moves which will help in exploring adjacent cells.

3. **Depth-First Search (DFS)**:
    - The `dfs` function is used to explore one of the islands. It marks cells belonging to the first island with `2` to avoid revisiting them.
    - We collect the positions of the first island in the list `island_positions`.

4. **Breadth-First Search (BFS)**:
    - After identifying the first island, we initiate a BFS from all positions of the first island, and track the distance (number of steps) to reach the second island.
    - If we encounter a `1` during BFS, it indicates that we have connected the two islands, and we return the current distance.

5. **Marking Water**: Cells that are water (`0`s) that we have visited are marked as `-1` to prevent re-processing.

### Complexity:
- Time Complexity: O(N * M), where N and M are the dimensions of the grid. We traverse every cell at most once.
- Space Complexity: O(N * M) for the BFS queue and the DFS stack.

This algorithm ensures that the connection between the two islands is made with the least number of flips, adhering to the constraints and guarantees provided in the problem description.

# 967. Numbers With Same Consecutive Differences

### Problem Description 
Return all non-negative integers of length `n` such that the absolute difference between every two consecutive digits is `k`.

Note that every number in the answer must not have leading zeros. For example, `01` has one leading zero and is invalid.

You may return the answer in any order.


Example 1:
Input: n = 3, k = 7
Output: [181,292,707,818,929]
Explanation: Note that 070 is not a valid number, because it has leading zeroes.


Example 2:
Input: n = 2, k = 1
Output: [10,12,21,23,32,34,43,45,54,56,65,67,76,78,87,89,98]

Example 3:
Input: n = 2, k = 0
Output: [11,22,33,44,55,66,77,88,99]

Example 4:
Input: n = 2, k = 2
Output: [13,20,24,31,35,42,46,53,57,64,68,75,79,86,97]

Constraints:
`2 <= n <= 9`
`0 <= k <= 9`

### Solution 
 To solve this problem, we can use a backtracking approach, which allows us to build valid numbers digit by digit, ensuring that we adhere to the constraints given (i.e., the absolute difference between consecutive digits must be `k` and the number must not have leading zeros).

### Steps to Approach:
1. **Initialization**: Start by defining a helper function (`backtrack`) that will generate valid numbers.
2. **Digit Building**: In the `backtrack` function, we can iterate over all possible digits from `0` to `9` to start forming numbers, ensuring that we don't start with a zero if `n > 1` (to avoid leading zeros).
3. **Consecutive Digits Check**: For each digit, we will calculate the next possible digits. If the current digit is `d`, the next digit can be `d + k` or `d - k`, provided these digits remain between `0` and `9`.
4. **Base Case**: Once the length of the current number equals `n`, we convert it into an integer and add it to our results list.
5. **Return Results**: Finally, return the result list.

### Implementation:
Here’s how the code looks in Python, following the LeetCode solution format:



```python
from typing import List

class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        def backtrack(current_number: int, length: int) -> None:
            if length == n:
                result.append(current_number)
                return
            
            last_digit = current_number % 10
            
            # Calculate possible next digits
            next_digits = set()
            if last_digit + k < 10:
                next_digits.add(last_digit + k)
            if last_digit - k >= 0:
                next_digits.add(last_digit - k)
            
            for next_digit in next_digits:
                # Form the new number by appending the digit
                new_number = current_number * 10 + next_digit
                backtrack(new_number, length + 1)

        result = []
        # Start building numbers
        for i in range(1, 10):  # Leading digit can be from 1 to 9
            backtrack(i, 1)

        return result

```

### Explanation of the Code:
1. **Function Definition**: We define the main function `numsSameConsecDiff` which will manage the backtracking process and start generating numbers.
2. **Backtracking Function**: The `backtrack` function takes the current number being built and its length. When the length matches `n`, we store the number.
3. **Next Digit Calculation**: We calculate the possible next digits by adjusting the last digit.
4. **Recursive Calls**: The function calls itself recursively to explore the number further.
5. **Result Collection**: The valid numbers generated are collected in the `result` list, which is returned at the end.

### Edge Cases:
- The function starts with numbers from `1-9` to avoid any leading zeros.
- The absolute difference condition is strictly adhered to via the calculations of `next_digits`.

This approach is efficient given the constraints, and backtracking effectively explores the state space of valid numbers without generating invalid candidates. The overall complexity is manageable due to limited permissible digits and the depth of recursion limited by `n`.

# 987. Vertical Order Traversal of a Binary Tree

### Problem Description 
Given the `root` of a binary tree, calculate the vertical order traversal of the binary tree.

For each node at position `(row, col)`, its left and right children will be at positions `(row + 1, col - 1)` and `(row + 1, col + 1)` respectively. The root of the tree is at `(0, 0)`.

The vertical order traversal of a binary tree is a list of top-to-bottom orderings for each column index starting from the leftmost column and ending on the rightmost column. There may be multiple nodes in the same row and same column. In such a case, sort these nodes by their values.

Return the vertical order traversal of the binary tree.


Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: [[9],[3,15],[20],[7]]
Explanation:
Column -1: Only node 9 is in this column.

Column 0: Nodes 3 and 15 are in this column in that order from top to bottom.

Column 1: Only node 20 is in this column.

Column 2: Only node 7 is in this column.


Example 2:
Input: root = [1,2,3,4,5,6,7]
Output: [[4],[2],[1,5,6],[3],[7]]
Explanation:
Column -2: Only node 4 is in this column.

Column -1: Only node 2 is in this column.

Column 0: Nodes 1, 5, and 6 are in this column.

          1 is at the top, so it comes first.

          5 and 6 are at the same position (2, 0), so we order them by their value, 5 before 6.

Column 1: Only node 3 is in this column.

Column 2: Only node 7 is in this column.


Example 3:
Input: root = [1,2,3,4,6,5,7]
Output: [[4],[2],[1,5,6],[3],[7]]
Explanation:
This case is the exact same as example 2, but with nodes 5 and 6 swapped.

Note that the solution remains the same since 5 and 6 are in the same location and should be ordered by their values.


Constraints:
The number of nodes in the tree is in the range `[1, 1000]`.

`0 <= Node.val <= 1000`

### Solution 
 To solve the problem of calculating the vertical order traversal of a binary tree, we need to establish a systematic approach using breadth-first search (BFS) or depth-first search (DFS). In this solution, we'll use BFS, which allows us to explore nodes level by level while maintaining their respective vertical positions.

### Detailed Explanation:

1. **Data Structure**: We'll use a dictionary to map each column index to a list of nodes. The column index will help us maintain the vertical order, and the list will store tuples of (row, value) to keep track of the rows and values of the nodes.

2. **BFS Traversal**: We'll initiate a BFS starting from the root node. As we traverse each node, we'll keep track of its position using a tuple (row, col):
   - The root node starts at (0, 0).
   - For each node, its left child is at (row + 1, col - 1) and its right child is at (row + 1, col + 1).

3. **Sorting**: Once we have all the nodes grouped by their column indices, we'll sort:
   - First by the column index.
   - Then by the row index for ordering from top to bottom.
   - Finally, sort by node values if they are at the same (row, column).

4. **Output Format**: After sorting, we'll aggregate the values in the final output list by extracting values from sorted node tuples.

Here's the Python code implementing the above logic:



```python
from collections import defaultdict, deque
from typing import List, Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        # A dictionary to hold the mapping from column index to (row, value)
        column_table = defaultdict(list)
        # Queue for BFS traversal, initialized with root at column 0 and row 0
        queue = deque([(root, 0, 0)])  # (node, row, column)

        while queue:
            node, row, col = queue.popleft()
            if node:
                # Append the (row, value) to the column_table
                column_table[col].append((row, node.val))
                # Add the children to the queue with updated row and column values
                queue.append((node.left, row + 1, col - 1))
                queue.append((node.right, row + 1, col + 1))

        # Sort the dictionary and extract the values in the required order
        sorted_columns = sorted(column_table.keys())
        result = []

        for col in sorted_columns:
            # Sort by row first, then by value (if same row)
            column_table[col].sort()
            result.append([val for row, val in column_table[col]])

        return result

```

### Explanation of Code:
- **TreeNode class**: This is a simple class to define each node of the binary tree.
- **verticalTraversal function**: This is the main method of the `Solution` class. It implements the BFS approach using a queue:
  - `column_table`: A `defaultdict` that collects nodes by their column indices.
  - The BFS collects and organizes nodes based on their row and column positions.
- **Sorting**: After collecting nodes, we sort the column keys, and then for each column, we sort the tuples to ensure they are ordered first by row and then by value.
- Finally, we construct the result list from the sorted tuples and return it.

### Example Usage:
To test this solution, you can create a binary tree using the `TreeNode` class and call the `verticalTraversal` method from the `Solution` class, passing the root of your tree. This would yield the vertical traversal as specified.

# 993. Cousins in Binary Tree

### Problem Description 
In a binary tree, the root node is at depth `0`, and children of each depth `k` node are at depth `k+1`.

Two nodes of a binary tree are cousins if they have the same depth, but have different parents.

We are given the `root` of a binary tree with unique values, and the values `x` and `y` of two different nodes in the tree.

Return `true` if and only if the nodes corresponding to the values `x` and `y` are cousins.


Example 1:
Input: root = [1,2,3,4], x = 4, y = 3
Output: false

Example 2:
Input: root = [1,2,3,null,4,null,5], x = 5, y = 4
Output: true

Example 3:
Input: root = [1,2,3,null,4], x = 2, y = 3
Output: false

Constraints:
The number of nodes in the tree will be between `2` and `100`.

Each node has a unique integer value from `1` to `100`.

### Solution 
 To solve the problem of checking if two nodes in a binary tree are cousins, we need to determine two properties for the nodes with values `x` and `y`:

1. Both nodes must be at the same depth in the binary tree.
2. Both nodes must have different parents.

Here’s a structured approach to solve this problem in Python with a detailed explanation of each part.

**Step-by-step Plan:**
1. **Tree Traversal**: We will perform a Breadth-First Search (BFS) or Depth-First Search (DFS) to traverse the tree and gather information about the depth and parent of the nodes with values `x` and `y`.
2. **Store Information**: During the traversal, we will record the depth and parent of both nodes.
3. **Check Conditions**: Finally, we will check if both nodes are at the same depth and have different parents to determine if they are cousins.

### Implementation

Here’s the implementation in Python, formatted according to LeetCode standards:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        # To store the depth and parent of x and y
        parent = {}
        depth = {}
        
        # BFS queue to explore nodes level by level
        queue = [(root, 0, None)]  # (node, current_depth, parent)
        
        while queue:
            current, d, p = queue.pop(0)
            
            # Record the depth and parent of the nodes we are interested in
            if current.val == x:
                depth[x] = d
                parent[x] = p
            elif current.val == y:
                depth[y] = d
                parent[y] = p
            
            # If we have found both nodes, no need to continue
            if x in depth and y in depth:
                break
            
            # Enqueue the left and right child nodes
            if current.left:
                queue.append((current.left, d + 1, current))
            if current.right:
                queue.append((current.right, d + 1, current))
        
        # Check if they are cousins
        return depth[x] == depth[y] and parent[x] != parent[y]

```

### Explanation of the Code

1. **TreeNode Class**: This is the definition of a node in the binary tree. It includes `val`, `left`, and `right` properties.
2. **Solution Class**: Contains the method `isCousins` which accepts the `root` of the tree and the values `x` and `y`.
3. **BFS Initialization**: We maintain a queue to explore the nodes. Each entry in the queue contains a tuple of the current node, the current depth, and the parent node.
4. **While Loop**: This loop continues until we find both nodes or the queue is empty. For each node processed:
   - We check its value against `x` and `y`, recording the `depth` and `parent` if there is a match.
   - If both values are found, we can exit the loop early.
   - We enqueue the left and right child nodes along with their respective depths and parent nodes.
5. **Final Check**: After the BFS, we check if `x` and `y` have the same depth and different parents, returning the result as a boolean value.

### Test Cases
You can run the above code with the provided example test cases to validate its correctness. The function `isCousins` will return `True` or `False` based on whether the nodes with values `x` and `y` are cousins in the given binary tree.

This implementation ensures that we meet the problem's constraints efficiently while providing clarity on the traversal and checking mechanisms.

# 994. Rotting Oranges

### Problem Description 
You are given an `m x n` `grid` where each cell can have one of three values:
`0` representing an empty cell,
`1` representing a fresh orange, or
`2` representing a rotten orange.

Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return `-1`.


Example 1:
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4

Example 2:
Input: grid = [[2,1,1],[0,1,1],[1,0,1]]
Output: -1
Explanation: The orange in the bottom left corner (row 2, column 0) is never rotten, because rotting only happens 4-directionally.


Example 3:
Input: grid = [[0,2]]
Output: 0
Explanation: Since there are already no fresh oranges at minute 0, the answer is just 0.


Constraints:
`m == grid.length`
`n == grid[i].length`
`1 <= m, n <= 10`
`grid[i][j]` is `0`, `1`, or `2`.

### Solution 
 To solve the problem of determining the minimum number of minutes required for all fresh oranges to rot, we can use a Breadth-First Search (BFS) approach. This approach models the spread of rot from rotten oranges to adjacent fresh oranges over time.

### Explanation:

1. **Initialization**: 
   - Start by counting the number of fresh oranges (value `1`) and gather the positions of all rotten oranges (value `2`).
   - If there are no fresh oranges at the beginning, we immediately return `0` because no time is needed.
  
2. **BFS Setup**:
   - Use a queue to explore the grid. Enqueue all initial rotten oranges and set the count of minutes to `0`.
   - Use a direction list to define the 4 possible movements (up, down, left, right).

3. **BFS Process**:
   - While there are rotten oranges in the queue:
     - Process elements in the queue for the current minute.
     - For each rotten orange, check its 4 adjacent cells. If an adjacent cell contains a fresh orange, rot that orange (change its value to `2`), decrement the count of fresh oranges, and enqueue this rotten orange's position for the next round.

4. **Completion Check**:
   - After processing all rotten oranges, if there are still fresh oranges left, return `-1` (indicating it's impossible to rot all oranges).
   - Otherwise, return the number of minutes elapsed.

### Implementation:



```python
from collections import deque

class Solution:
    def orangesRotting(self, grid):
        rows, cols = len(grid), len(grid[0])
        queue = deque()
        fresh_count = 0
        
        # Step 1: Initialize the queue with all rotten oranges
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    queue.append((r, c))
                elif grid[r][c] == 1:
                    fresh_count += 1
        
        # Step 2: If there are no fresh oranges, return 0
        if fresh_count == 0:
            return 0
        
        minutes_passed = 0
        
        # Directions for 4-way connection
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Step 3: Perform BFS
        while queue:
            # Increment minutes each time we start with a new layer of rotting
            for _ in range(len(queue)):
                x, y = queue.popleft()
                
                # Check all adjacent cells
                for dx, dy in directions:
                    new_x, new_y = x + dx, y + dy
                    
                    # If the adjacent cell is within bounds and is fresh
                    if 0 <= new_x < rows and 0 <= new_y < cols and grid[new_x][new_y] == 1:
                        # Rot the orange
                        grid[new_x][new_y] = 2
                        fresh_count -= 1
                        queue.append((new_x, new_y))
            
            # Increase the minutes after processing one layer
            minutes_passed += 1
        
        # Step 4: Check if there are still fresh oranges
        return minutes_passed if fresh_count == 0 else -1

```

### Complexity Analysis:
- **Time Complexity**: O(m * n), where m is the number of rows and n is the number of columns. Each cell is processed at most once.
- **Space Complexity**: O(m * n) in the worst case, due to the storage of rotten oranges in the queue.

### Conclusion:
This approach effectively simulates the rot spreading process, counting time in a way that respects the constraints of BFS, ensuring all possibilities are explored efficiently. The solution adheres to the LeetCode submission format and is ready for execution on the site.

# 1036. Escape a Large Maze

### Problem Description 
There is a 1 million by 1 million grid on an XY-plane, and the coordinates of each grid square are `(x, y)`.

We start at the `source = [sx, sy]` square and want to reach the `target = [tx, ty]` square. There is also an array of `blocked` squares, where each `blocked[i] = [xi, yi]` represents a blocked square with coordinates `(xi, yi)`.

Each move, we can walk one square north, east, south, or west if the square is not in the array of `blocked` squares. We are also not allowed to walk outside of the grid.

Return `true` if and only if it is possible to reach the `target` square from the `source` square through a sequence of valid moves.


Example 1:
Input: blocked = [[0,1],[1,0]], source = [0,0], target = [0,2]
Output: false
Explanation: The target square is inaccessible starting from the source square because we cannot move.

We cannot move north or east because those squares are blocked.

We cannot move south or west because we cannot go outside of the grid.


Example 2:
Input: blocked = [], source = [0,0], target = [999999,999999]
Output: true
Explanation: Because there are no blocked cells, it is possible to reach the target square.


Constraints:
`0 <= blocked.length <= 200`
`blocked[i].length == 2`
`0 <= xi, yi < 106`
`source.length == target.length == 2`
`0 <= sx, sy, tx, ty < 106`
`source != target`
It is guaranteed that `source` and `target` are not blocked.

### Solution 
 To solve the problem of determining whether we can reach the target square in a large grid while avoiding blocked squares, we can employ a breadth-first search (BFS) approach. The main challenge here is to manage the large size of the grid (1 million by 1 million) while considering a relatively small number of blocked squares (up to 200).

A succinct summary of the solution involves:
1. **Understanding Movement**: We can move in four directions - north, south, east, and west.
2. **Checking Blocked Squares**: We need to ensure we do not move into blocked squares.
3. **Using BFS to Find the Path**: BFS is ideal for exploring all possible paths from the source to the target.

Here's the plan for the algorithm:
- If the target is directly adjacent to the source and there are no blocked squares between them, return `True`.
- Use BFS to explore valid squares starting from the source until we either reach the target or cannot make further valid moves.
- Ensure to terminate the BFS if the number of steps taken exceeds certain constraints to avoid infinite loops.

Now, let’s implement the solution in Python following the LeetCode format.



```python
from collections import deque

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        if not blocked:  # No blocked cells, we can move anywhere
            return True
        
        blocked_set = set(map(tuple, blocked))  # Use a set for O(1) lookup
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Right, Left, Down, Up

        # BFS Function
        def bfs(start: List[int], end: List[int]) -> bool:
            q = deque([tuple(start)])  # Queue for BFS
            visited = set()  # To keep track of visited positions
            visited.add(tuple(start))
            steps = 0
            
            while q:
                x, y = q.popleft()
                
                # Increment the number of steps taken
                steps += 1
                
                if steps > 20000:  # If steps exceed this, we can assume there is a valid path
                    return True
                
                # Explore neighbors
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    
                    if (0 <= nx < 10**6 and 0 <= ny < 10**6 
                        and (nx, ny) not in blocked_set and (nx, ny) not in visited):
                        if [nx, ny] == end:  # If we reached the target
                            return True
                        visited.add((nx, ny))
                        q.append((nx, ny))
            return False

        # Check if we can reach 'target' from 'source' and vice versa
        return bfs(source, target) and bfs(target, source)

```

### Explanation of Code:
1. **Set Up**: We first check if there are blocked cells. If there aren't any, we can directly return `True`.
2. **Using a Set for Blocked Squares**: This provides O(1) time complexity for checking if a square is blocked. The blocked cells are stored as tuples in a set.
3. **BFS Function**: The BFS function attempts to find a path from a starting point to an ending point. It employs a queue to explore all possible moves:
   - If we exceed a certain number of steps (20000 in this case), we assume that reaching the target is possible, indicating that the path is indeed navigable despite the grid size.
   - For each valid square, we add unvisited and non-blocked squares to the queue for further exploration.
4. **Reciprocal Check**: We conduct two BFS checks, one from `source` to `target` and another from `target` to `source`, to ensure that both ends can reach each other considering the blocked cells.

### Complexity:
- **Time Complexity**: O(N) where N is the number of blocked squares since we only examine a limited number of neighboring squares.
- **Space Complexity**: O(N) for the queue and visited set.

This solution is efficient and works within the constraints given by the problem.

# 1091. Shortest Path in Binary Matrix

### Problem Description 
Given an `n x n` binary matrix `grid`, return the length of the shortest clear path in the matrix. If there is no clear path, return `-1`.

A clear path in a binary matrix is a path from the top-left cell (i.e., `(0, 0)`) to the bottom-right cell (i.e., `(n - 1, n - 1)`) such that:
All the visited cells of the path are `0`.

All the adjacent cells of the path are 8-directionally connected (i.e., they are different and they share an edge or a corner).

The length of a clear path is the number of visited cells of this path.


Example 1:
Input: grid = [[0,1],[1,0]]
Output: 2

Example 2:
Input: grid = [[0,0,0],[1,1,0],[1,1,0]]
Output: 4

Example 3:
Input: grid = [[1,0,0],[1,1,0],[1,1,0]]
Output: -1

Constraints:
`n == grid.length`
`n == grid[i].length`
`1 <= n <= 100`
`grid[i][j] is 0 or 1`

### Solution 
 To solve the given problem of finding the shortest clear path in an `n x n` binary matrix, we will utilize the Breadth-First Search (BFS) algorithm. This approach is suitable because it explores all possible paths from the starting cell `(0, 0)` to the end cell `(n - 1, n - 1)` level by level, ensuring that we find the shortest path in an unweighted grid (where each cell can be treated as having equal weight).

### Solution Explanation

1. **Input and Initialization**:
   - We start by checking if the starting cell `(0, 0)` or the end cell `(n - 1, n - 1)` is blocked (i.e., if they contain `1`). If either is blocked, we return `-1` immediately because there would be no possible path.
   - We set up a queue for BFS and initialize the starting point with a path length of `1` (the cell itself counts as part of the path). We also maintain a visited set to avoid revisiting cells.

2. **BFS Algorithm**:
   - We dequeue the front cell and explore all 8 possible directions (the 4 cardinal directions and the 4 diagonal directions).
   - For each valid move (a move to a cell that is within bounds, unvisited, and not blocked), we check if we have reached the end cell. If we have, we return the current path length + 1.
   - If we haven’t reached the end cell, we enqueue the new cell with an incremented path length and mark it as visited.
   - This process continues until the queue is empty.

3. **Final Check**:
   - If the queue is exhausted and we haven’t reached the `(n - 1, n - 1)` cell, we return `-1`, indicating there's no valid path.

### Implementation

Here is the code for the solution:



```python
from collections import deque

class Solution:
    def shortestPathBinaryMatrix(self, grid):
        n = len(grid)
        
        # If the starting or ending cell is blocked, return -1
        if grid[0][0] == 1 or grid[n - 1][n - 1] == 1:
            return -1
        
        # Directions for 8 possible movements (including diagonals)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        
        # BFS initialization
        queue = deque([(0, 0, 1)])  # (row, col, path_length)
        visited = set((0, 0))
        
        while queue:
            r, c, path_length = queue.popleft()
            
            # If we've reached the bottom-right corner
            if r == n - 1 and c == n - 1:
                return path_length
            
            # Explore all 8 directions
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                # Check if the new cell is within bounds and not blocked
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited and grid[nr][nc] == 0:
                    visited.add((nr, nc))
                    queue.append((nr, nc, path_length + 1))
        
        # If we exhaust the queue without finding the path
        return -1

```

### Edge Cases

- **Single Cell Grid**: When the grid is `[[0]]`, the output should be `1`.
- **Blocked Ending or Starting**: The algorithm should handle grids where the start or end is blocked, returning `-1`.
- **Maximum Grid Size**: The algorithm must efficiently handle the maximum constraints, ensuring no performance degradation.

This code will correctly return the length of the shortest path in the binary matrix or `-1` if no path exists when run on LeetCode.

# 1129. Shortest Path with Alternating Colors

### Problem Description 
Consider a directed graph, with nodes labelled `0, 1, ..., n-1`.  In this graph, each edge is either red or blue, and there could be self-edges or parallel edges.

Each `[i, j]` in `red_edges` denotes a red directed edge from node `i` to node `j`.  Similarly, each `[i, j]` in `blue_edges` denotes a blue directed edge from node `i` to node `j`.

Return an array `answer` of length `n`, where each `answer[X]` is the length of the shortest path from node `0` to node `X` such that the edge colors alternate along the path (or `-1` if such a path doesn't exist).


Example 1:
Input: n = 3, red_edges = [[0,1],[1,2]], blue_edges = []
Output: [0,1,-1]

Example 2:
Input: n = 3, red_edges = [[0,1]], blue_edges = [[2,1]]
Output: [0,1,-1]

Example 3:
Input: n = 3, red_edges = [[1,0]], blue_edges = [[2,1]]
Output: [0,-1,-1]

Example 4:
Input: n = 3, red_edges = [[0,1]], blue_edges = [[1,2]]
Output: [0,1,2]

Example 5:
Input: n = 3, red_edges = [[0,1],[0,2]], blue_edges = [[1,0]]
Output: [0,1,1]

Constraints:
`1 <= n <= 100`
`red_edges.length <= 400`
`blue_edges.length <= 400`
`red_edges[i].length == blue_edges[i].length == 2`
`0 <= red_edges[i][j], blue_edges[i][j] < n`

### Solution 
 To solve the problem of finding the shortest path in a directed graph with alternating edge colors, we can use a Breadth-First Search (BFS) approach. The problem requires us to alternate between red and blue edges while traversing from node `0` to all other nodes. Here's how we can implement this solution step-by-step.

### Explanation:

1. **Graph Representation**:
   We will represent the graph using an adjacency list, which will allow us to efficiently traverse nodes. We'll have two separate lists: one for red edges and another for blue edges.

2. **BFS Initialization**:
   We will initialize a queue for our BFS, and start from node `0`. We'll use a variable to keep track of the current color of the edge being used (either red or blue). We will also maintain a `distance` array where `distance[i]` will store the shortest distance to reach node `i`. Initially, `distance[0]` will be set to `0`, and all other nodes will be set to `-1` (indicating unreachable).

3. **BFS Traversal**:
   Each time we traverse to a next node, we'll switch the edge's color (red to blue or blue to red). We will enqueue each neighbor that we can reach via the current color edge, provided we haven't visited that neighbor with the opposite color yet.

4. **Result Compilation**:
   After completing the BFS, the `distance` array will contain the shortest distances from node `0` to every other node with the alternating edge constraint.

This approach efficiently uses BFS properties and ensures that we explore the shortest paths first while alternating colors of edges.

### Python Code:

Here’s how the solution would look in Python, formatted for LeetCode:



```python
from collections import deque, defaultdict

class Solution:
    def shortestAlternatingPaths(self, n: int, red_edges: List[List[int]], blue_edges: List[List[int]]) -> List[int]:
        # Creating adjacency lists
        graph = defaultdict(list)
        for u, v in red_edges:
            graph[u].append((v, 0))  # 0 for red
        for u, v in blue_edges:
            graph[u].append((v, 1))  # 1 for blue
        
        # Distance array
        distance = [-1] * n
        distance[0] = 0  # Distance to node 0 is 0
        
        # BFS queue storing (node, color) where color = 0 (red) or 1 (blue)
        queue = deque([(0, -1)])  # Start from node 0 with no color
        
        # BFS traversal
        steps = 0
        while queue:
            for _ in range(len(queue)):
                node, color = queue.popleft()
                
                # Check adjacency list
                for neighbor, edge_color in graph[node]:
                    # Only proceed if we are alternating colors or it's the first step
                    if color == -1 or edge_color != color:
                        # If it hasn't been visited, set the distance
                        if distance[neighbor] == -1:
                            distance[neighbor] = steps + 1
                        queue.append((neighbor, edge_color))
            steps += 1  # Increase steps for the next layer
            
        return distance

```

### Complexity Analysis:

- **Time Complexity**: The BFS runs in O(V + E) time, where V is the number of vertices (nodes) and E is the number of edges. Given the constraints, this will be efficient.
  
- **Space Complexity**: O(V + E) is also used for the adjacency list and the queue.

This solution effectively handles the given constraints and provides the correct answer for each node's shortest path using alternating edges.


# 1161. Maximum Level Sum of a Binary Tree

### Problem Description 
Given the `root` of a binary tree, the level of its root is `1`, the level of its children is `2`, and so on.

Return the smallest level `x` such that the sum of all the values of nodes at level `x` is maximal.


Example 1:
Input: root = [1,7,0,7,-8,null,null]
Output: 2
Explanation: 
Level 1 sum = 1.

Level 2 sum = 7 + 0 = 7.

Level 3 sum = 7 + -8 = -1.

So we return the level with the maximum sum which is level 2.


Example 2:
Input: root = [989,null,10250,98693,-89388,null,null,null,-32127]
Output: 2

Constraints:
The number of nodes in the tree is in the range `[1, 104]`.

`-105 <= Node.val <= 105`

### Solution 
 To solve the problem of finding the smallest level `x` in a binary tree such that the sum of all values of the nodes at that level is maximal, we can approach the problem using a breadth-first search (BFS). This method allows us to traverse the tree level by level and calculate the sums for each level efficiently.

Let's break down the solution step by step:

1. **Understanding the Input and Output**:
    - We are given the `root` of a binary tree.
    - We need to return the level (1-based index) that has the maximum sum of node values.

2. **Using Breadth-First Search**:
    - We'll use a queue to facilitate the BFS. 
    - As we traverse each level of the tree, we'll calculate the sum of node values at that level.
    - We'll keep track of the maximum sum encountered and the corresponding level.

3. **Implementation Steps**:
    - Initialize a queue with the root node.
    - For each level, calculate the sum of values and update the maximum sum if the current sum is higher.
    - If the sums are equal, we'll retain the smaller level, since we want the smallest level with a maximal sum.
  
4. **Returning the Result**:
    - Once we've processed all levels, the result is the level associated with the maximal sum.

Here is the Python code implementing this logic, formatted for submission to LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from collections import deque

class Solution:
    def maxLevelSum(self, root: TreeNode) -> int:
        # Edge case: if the tree is empty, return 0
        if not root:
            return 0
        
        # Initialize a queue for BFS
        queue = deque([root])
        max_sum = float('-inf')
        level_with_max_sum = 1
        current_level = 1
        
        while queue:
            # Find the number of nodes at the current level
            level_size = len(queue)
            current_level_sum = 0
            
            for _ in range(level_size):
                node = queue.popleft()
                current_level_sum += node.val
                
                # Add child nodes to the queue for processing the next level
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            # Check if the current level sum is greater than the max sum found so far
            if current_level_sum > max_sum:
                max_sum = current_level_sum
                level_with_max_sum = current_level
            
            # Increment to the next level
            current_level += 1
        
        return level_with_max_sum

```

### Explanation of the Code:
1. **TreeNode Class**: This is a basic implementation for the binary tree node.
2. **Solution Class**: Contains the main logic.
3. **maxLevelSum Method**:
   - Checks for an empty tree.
   - Uses a queue to perform BFS.
   - For each level, calculates the sum of node values and checks if it exceeds the previously recorded max sum.
   - Keeps track of which level has the greatest sum.
4. **Return Statement**: The method finally returns the smallest level with the maximal sum.

### Complexity Analysis:
- **Time Complexity**: O(N), where N is the number of nodes in the binary tree, since we visit each node exactly once.
- **Space Complexity**: O(W), where W is the maximum width of the tree. In the worst case (for a complete tree), the maximum width can be N/2.

This solution effectively meets the problem requirements and efficiently calculates the needed level with the maximum sum.

# 1162. As Far from Land as Possible

### Problem Description 
Given an `n x n` `grid` containing only values `0` and `1`, where `0` represents water and `1` represents land, find a water cell such that its distance to the nearest land cell is maximized, and return the distance. If no land or water exists in the grid, return `-1`.

The distance used in this problem is the Manhattan distance: the distance between two cells `(x0, y0)` and `(x1, y1)` is `|x0 - x1| + |y0 - y1|`.


Example 1:
Input: grid = [[1,0,1],[0,0,0],[1,0,1]]
Output: 2
Explanation: The cell (1, 1) is as far as possible from all the land with distance 2.


Example 2:
Input: grid = [[1,0,0],[0,0,0],[0,0,0]]
Output: 4
Explanation: The cell (2, 2) is as far as possible from all the land with distance 4.


Constraints:
`n == grid.length`
`n == grid[i].length`
`1 <= n <= 100`
`grid[i][j]` is `0` or `1`

### Solution 
 To tackle the problem of finding the water cell that maximizes the distance to the nearest land cell in an `n x n` grid, we can leverage a breadth-first search (BFS) approach. This is effective here because we want to propagate distances from all land cells (1's) as we explore outward to the water cells (0's). 

### Approach:
1. **Initialize BFS**:
   - First, we need to identify all land cells in the grid. These cells will be our starting points for the BFS.
   - We'll maintain a queue to facilitate our BFS traversal and also maintain a variable to track the maximum distance found.

2. **Distance Propagation**:
   - From each land cell, we will explore its neighboring cells (up, down, left, right) and push water cells into the queue while updating their distances from the land.
   - As we traverse the grid, we'll ensure to avoid revisiting any cell.

3. **Edge Cases**:
   - If there are no land cells or if the entire grid is made up of land cells, we will simply return `-1`.

### Implementation:
Here is the Python code that clearly implements the above plan according to the LeetCode format.



```python
from collections import deque
from typing import List

class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        n = len(grid)
        
        # Directions for up, down, left, right movements
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        # Queue for BFS
        queue = deque()
        
        # Add all the land cells (1's) to the queue and check for edge cases
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    queue.append((i, j))
        
        # If there are no land cells or all are land cells
        if not queue or len(queue) == n * n:
            return -1
        
        max_distance = -1
        
        # Perform BFS
        while queue:
            size = len(queue)
            for _ in range(size):
                x, y = queue.popleft()
                
                # Explore all 4 possible directions
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    
                    # Check if the new coordinates are in bounds and is water
                    if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 0:
                        # Set this water cell to be land (visited) and enqueue it
                        grid[nx][ny] = 1  # Marking visited by turning it into land
                        queue.append((nx, ny))
            
            # After processing one level of BFS, increase max distance
            max_distance += 1
        
        return max_distance

```

### Explanation:
- **Line 1-3**: We import the necessary libraries and define the `Solution` class with the `maxDistance` method.
- **Line 5**: We get the size of the grid `n`.
- **Line 8**: Defined the directions for moving to adjacent cells.
- **Line 11-20**: We populate the queue with all the land cells and check if we have any valid water cells.
- **Line 23**: We set up the BFS loop where we process each level of the queue.
- **Line 28-31**: For each cell we dequeue, we check its four neighbors. If a neighbor is a water cell, we mark it as visited (change it to land) and enqueue it.
- **Line 34**: Finally, we return the maximum distance calculated, incremented for each level of BFS processed.

This method ensures we efficiently find the maximum distance from any water cell to the nearest land cell as the BFS explores all reachable water cells step-by-step.

# 1197. Minimum Knight Moves

### Problem Description 
In an infinite chess board with coordinates from `-infinity` to `+infinity`, you have a knight at square `[0, 0]`.

A knight has 8 possible moves it can make, as illustrated below. Each move is two squares in a cardinal direction, then one square in an orthogonal direction.

Return the minimum number of steps needed to move the knight to the square `[x, y]`.  It is guaranteed the answer exists.


Example 1:
Input: x = 2, y = 1
Output: 1
Explanation: [0, 0] → [2, 1]

Example 2:
Input: x = 5, y = 5
Output: 4
Explanation: [0, 0] → [2, 1] → [4, 2] → [3, 4] → [5, 5]

Constraints:
`|x| + |y| <= 300`

### Solution 
 To solve the problem of finding the minimum steps for a knight on an infinite chess board to reach a specific position `x, y`, we can use the Breadth-First Search (BFS) algorithm. The BFS approach is well-suited for this problem since it explores all possible positions level by level, ensuring that we find the shortest path to our target position.

Here's a step-by-step explanation of how we can implement this:

1. **Understanding Knight Moves**: The knight can move in eight possible ways:
   - (2, 1), (2, -1), (-2, 1), (-2, -1)
   - (1, 2), (1, -2), (-1, 2), (-1, -2)
   We can represent these moves as tuples.

2. **BFS Setup**: We will use a queue to keep track of our current position and the number of moves made. We also need a set to track visited positions to avoid processing the same position multiple times.

3. **Termination Criterion**: The BFS continues until we reach the destination coordinates `(x, y)`.

4. **Edge Case Handling**: The knight is initially located at `(0, 0)`.

Now, let's put this into a Python function following LeetCode's structure:



```python
from collections import deque

class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        # Define all possible knight moves
        knight_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        
        # Use a queue for BFS
        queue = deque([(0, 0, 0)])  # (current_x, current_y, steps)
        visited = set((0, 0))  # Track visited positions
        
        while queue:
            curr_x, curr_y, steps = queue.popleft()
            
            # Check if the current position is the target
            if curr_x == x and curr_y == y:
                return steps
            
            # Explore all possible knight moves
            for move in knight_moves:
                new_x = curr_x + move[0]
                new_y = curr_y + move[1]
                
                # Only consider positions within the distance of the target (x, y)
                # to prevent unnecessary exploration
                if (new_x, new_y) not in visited and \
                   abs(new_x) <= 300 and abs(new_y) <= 300:
                    visited.add((new_x, new_y))
                    queue.append((new_x, new_y, steps + 1))
        
        # The problem guarantees that the answer exists,
        # so we should never reach here.
        return -1

# Example usage:
# sol = Solution()
# print(sol.minKnightMoves(2, 1))  # Output: 1
# print(sol.minKnightMoves(5, 5))  # Output: 4

```

### Explanation of the Code:
- **Imports**: We import `deque` from the `collections` module for efficient queue operations.
- **Knight Moves**: We define all eight possible knight moves.
- **Queue Initialization**: We start with the initial position `(0, 0)` and `0` steps.
- **Visited Set**: We maintain a set of visited positions to prevent cycles.
- **BFS Loop**: We process each position in the queue. For each move, we check if we've reached `(x, y)`. If not, we add the new position to the queue only if it hasn't been visited and is within our defined bounds.
- If we ever reach the target position, we return the count of steps taken.

This solution is efficient and runs within the problem's constraints.

# 1210. Minimum Moves to Reach Target with Rotations

### Problem Description 
In an `n*n` grid, there is a snake that spans 2 cells and starts moving from the top left corner at `(0, 0)` and `(0, 1)`. The grid has empty cells represented by zeros and blocked cells represented by ones. The snake wants to reach the lower right corner at `(n-1, n-2)` and `(n-1, n-1)`.

In one move the snake can:
Move one cell to the right if there are no blocked cells there. This move keeps the horizontal/vertical position of the snake as it is.

Move down one cell if there are no blocked cells there. This move keeps the horizontal/vertical position of the snake as it is.

Rotate clockwise if it's in a horizontal position and the two cells under it are both empty. In that case the snake moves from `(r, c)` and `(r, c+1)` to `(r, c)` and `(r+1, c)`.

Rotate counterclockwise if it's in a vertical position and the two cells to its right are both empty. In that case the snake moves from `(r, c)` and `(r+1, c)` to `(r, c)` and `(r, c+1)`.

Return the minimum number of moves to reach the target.

If there is no way to reach the target, return `-1`.


Example 1:
Input: grid = [[0,0,0,0,0,1],
               [1,1,0,0,1,0],
               [0,0,0,0,1,1],
               [0,0,1,0,1,0],
               [0,1,1,0,0,0],
               [0,1,1,0,0,0]]
Output: 11
Explanation:
One possible solution is [right, right, rotate clockwise, right, down, down, down, down, rotate counterclockwise, right, down].


Example 2:
Input: grid = [[0,0,1,1,1,1],
               [0,0,0,0,1,1],
               [1,1,0,0,0,1],
               [1,1,1,0,0,1],
               [1,1,1,0,0,1],
               [1,1,1,0,0,0]]
Output: 9

Constraints:
`2 <= n <= 100`
`0 <= grid[i][j] <= 1`
It is guaranteed that the snake starts at empty cells.

### Solution 
 To solve the problem of moving a snake across a grid while adhering to the specified movement rules, we can approach this problem using the Breadth-First Search (BFS) algorithm. This is due to BFS being well-suited for problems that involve finding the shortest path or minimum moves in an unweighted graph-like structure, which in this case is represented by the grid.

### Problem Breakdown

1. **Grid Representation**: The snake occupies two cells. We need to represent its state considering both the head and tail positions.
2. **Movement Options**: The snake can:
   - Move right
   - Move down
   - Rotate clockwise (from horizontal to vertical)
   - Rotate counterclockwise (from vertical to horizontal)
3. **State Representation**: We can represent the state of the snake using a tuple that consists of the head's position, the tail's position, and its orientation (horizontal or vertical).
4. **Visited States**: To avoid visiting the same state multiple times, we need to maintain a set of visited states.
5. **Queue for BFS**: We will use a queue to store the current position of the snake, the number of moves taken so far, and the orientation.
6. **Target Position**: The target is to reach the bottom-right part of the grid with the snake fully residing in positions `(n-1, n-2)` for one part and `(n-1, n-1)` for the other.

### Implementation

Here’s the implementation in Python that fits the description and the format suitable for LeetCode. 



```python
from collections import deque

class Solution:
    def minimumMoves(self, grid):
        n = len(grid)
        if grid[0][1] == 1 or grid[n-1][n-2] == 1 or grid[n-1][n-1] == 1:
            return -1
        
        # Directions: right, down, clockwise (0), counterclockwise (1)
        directions = [(0, 1), (1, 0)]
        queue = deque()
        visited = set()
        
        # Initial state: head at (0,0) and tail at (0,1) in horizontal position
        queue.append(((0, 0), (0, 1), 0))  # (head, tail, orientation)
        visited.add(((0, 0), (0, 1), 0))
        
        moves = 0
        
        while queue:
            for _ in range(len(queue)):
                (hx, hy), (tx, ty), orientation = queue.popleft()
                
                # Check if we've reached the target
                if (hx, hy) == (n - 1, n - 1) and (tx, ty) == (n - 1, n - 2):
                    return moves
                
                # Movement possibilities
                # 1. Move right
                if orientation == 0:  # horizontal
                    if hy + 1 < n and grid[hx][hy + 1] == 0:  # space for the head to the right
                        new_head = (hx, hy + 1)
                        new_tail = (tx, ty + 1)  # move tail right as well
                        if new_tail[1] < n and grid[new_tail[0]][new_tail[1]] == 0:
                            new_state = (new_head, new_tail, 0)
                            if new_state not in visited:
                                visited.add(new_state)
                                queue.append(new_state)

                # 2. Move down
                if orientation == 0:  # horizontal
                    if hx + 1 < n and grid[hx + 1][hy] == 0 and grid[hx + 1][hy + 1] == 0:  # moving down into empty spaces
                        new_state = ((hx + 1, hy), (hx, hy), 1)  # new orientation vertical
                        if new_state not in visited:
                            visited.add(new_state)
                            queue.append(new_state)

                # 3. Rotate clockwise
                if orientation == 0:  # horizontal
                    if hx + 1 < n and grid[hx + 1][hy] == 0:  # space below head
                        new_state = ((hx + 1, hy), (hx, hy), 1)  # rotating to vertical
                        if new_state not in visited:
                            visited.add(new_state)
                            queue.append(new_state)

                # 4. Move the tail if vertical
                if orientation == 1:  # vertical
                    if hx + 1 < n and grid[hx + 1][hy] == 0:  # can move down
                        new_head = (hx + 1, hy)
                        new_tail = (tx + 1, ty)  # moving the tail down
                        if new_tail[1] < n and grid[new_tail[0]][new_tail[1]] == 0:
                            new_state = (new_head, new_tail, 1)
                            if new_state not in visited:
                                visited.add(new_state)
                                queue.append(new_state)

                # 5. Rotate counterclockwise
                if orientation == 1:  # vertical
                    if ty + 1 < n and grid[tx][ty + 1] == 0:  # checking space to the right
                        new_state = ((hx, hy + 1), (hx, hy), 0)  # rotating to horizontal
                        if new_state not in visited:
                            visited.add(new_state)
                            queue.append(new_state)

            moves += 1

        return -1

```

### Explanation of the Code:

1. **Setup**: We first check if the starting and ending points are valid. If either is blocked, return `-1`.
2. **BFS Initialization**: We start BFS with the initial position and rotation of the snake and track visited positions to avoid cycles.
3. **State Exploration**: For each state of the snake, we consider the four possible movements and enqueue valid next states.
4. **Termination**: If we reach the desired end state, we return the number of moves. If the queue is exhausted without finding a solution, we return `-1`.

This approach efficiently finds the minimum moves required for the snake to reach the target while obeying the specific movement rules, leveraging BFS's capacity to explore all possibilities level-wise.

# 1236. Web Crawler

### Problem Description 
Given a url `startUrl` and an interface `HtmlParser`, implement a web crawler to crawl all links that are under the same hostname as `startUrl`. 
Return all urls obtained by your web crawler in any order.

Your crawler should:
Start from the page: `startUrl`
Call `HtmlParser.getUrls(url)` to get all urls from a webpage of given url.

Do not crawl the same link twice.

Explore only the links that are under the same hostname as `startUrl`.

As shown in the example url above, the hostname is `example.org`. For simplicity sake, you may assume all urls use http protocol without any port specified. For example, the urls `http://leetcode.com/problems` and `http://leetcode.com/contest` are under the same hostname, while urls `http://example.org/test` and `http://example.com/abc` are not under the same hostname.

The `HtmlParser` interface is defined as such: 
interface HtmlParser {
  // Return a list of all urls from a webpage of given url.

  public List<String> getUrls(String url);
}
Below are two examples explaining the functionality of the problem, for custom testing purposes you'll have three variables urls`, edges` and startUrl`. Notice that you will only have access to startUrl` in your code, while urls` and edges` are not directly accessible to you in code.


Example 1:
Input:
urls = [
  "http://news.yahoo.com",
  "http://news.yahoo.com/news",
  "http://news.yahoo.com/news/topics/",
  "http://news.google.com",
  "http://news.yahoo.com/us"
]
edges = [[2,0],[2,1],[3,2],[3,1],[0,4]]
startUrl = "http://news.yahoo.com/news/topics/"
Output: [
  "http://news.yahoo.com",
  "http://news.yahoo.com/news",
  "http://news.yahoo.com/news/topics/",
  "http://news.yahoo.com/us"
]

Example 2:
Input: 
urls = [
  "http://news.yahoo.com",
  "http://news.yahoo.com/news",
  "http://news.yahoo.com/news/topics/",
  "http://news.google.com"
]
edges = [[0,2],[2,1],[3,2],[3,1],[3,0]]
startUrl = "http://news.google.com"
Output: ["http://news.google.com"]
Explanation: The startUrl links to all other pages that do not share the same hostname.


Constraints:
`1 <= urls.length <= 1000`
`1 <= urls[i].length <= 300`
`startUrl` is one of the `urls`.

Hostname label must be from 1 to 63 characters long, including the dots, may contain only the ASCII letters from 'a' to 'z', digits  from '0' to '9' and the hyphen-minus character ('-').

The hostname may not start or end with the hyphen-minus character ('-'). 
See:  https://en.wikipedia.org/wiki/Hostname#Restrictions_on_valid_hostnames
You may assume there're no duplicates in url library.

### Solution 
 To solve the problem of creating a web crawler that retrieves all URLs under the same hostname as a given `startUrl`, we can make use of a depth-first search (DFS) approach. The algorithm will involve the following steps:

1. **Extract the Hostname:** We need to parse the `startUrl` to extract its hostname to ensure we only crawl URLs under the same domain.

2. **Use a Set for Visited URLs:** To avoid crawling the same URL multiple times, we will keep track of visited URLs in a set.

3. **Depth-First Search (DFS):** We will start the crawling process at the `startUrl` and recursively call `getUrls()` on each URL we discover, adding valid URLs to our result set.

4. **Return the Results:** Finally, we will return a list of previously collected URLs that match the criteria.

The provided `HtmlParser` class allows us to get all URLs associated with a specific URL, so we will leverage this interface to gather URLs as we crawl.

Here’s how the code can be structured to fit the LeetCode format:



```python
class HtmlParser:
    def getUrls(self, url: str) -> List[str]:
        # This method would be provided by LeetCode; 
        # The implementation is not shown here.
        pass

class Solution:
    def crawl(self, startUrl: str, htmlParser: HtmlParser) -> List[str]:
        from urllib.parse import urlparse
        
        # Helper function to get the hostname
        def get_hostname(url):
            parsed_url = urlparse(url)
            return parsed_url.scheme + "://" + parsed_url.hostname
        
        # Get the hostname of the start URL
        start_hostname = get_hostname(startUrl)
        
        # Set to track visited URLs
        visited = set()
        result = []

        # DFS function to crawl URLs
        def dfs(url):
            if url in visited:
                return
            visited.add(url)

            # Add this valid URL to our results
            if get_hostname(url) == start_hostname:
                result.append(url)
                for next_url in htmlParser.getUrls(url):
                    dfs(next_url)
        
        # Start the crawling process
        dfs(startUrl)
        
        return result

```

### Explanation of the Code:

1. **Hostname Extraction:**
   - We use the `urlparse` method from the `urllib.parse` module to extract the hostname from the given URL.
   - The `get_hostname()` function converts the parsed URL into a canonical form by combining the scheme (http) and the hostname.

2. **Visitation Tracking:**
   - A set called `visited` is created to keep track of all the URLs that have been visited during the crawling process, which ensures we do not re-crawl the same URL.

3. **Depth-First Search (DFS):**
   - The `dfs()` function checks if a URL has already been visited. If it has not, it marks the URL as visited, checks if it belongs to the same host as `startUrl`, and if so, adds it to the `result`.
   - It fetches new URLs from the current URL using `htmlParser.getUrls(url)` and recursively visits them as well.

4. **Return the Collected URLs:**
   - Once all URLs are crawled, the result is returned, containing all URLs that share the same hostname as `startUrl`.

This code implements the problem solution efficiently and adheres to the constraints specified in the problem statement. You can directly run this solution on the LeetCode platform provided with the `HtmlParser` interface.

# 1242. Web Crawler Multithreaded

### Problem Description 
Given a url `startUrl` and an interface `HtmlParser`, implement a Multi-threaded web crawler to crawl all links that are under the same hostname as `startUrl`. 
Return all urls obtained by your web crawler in any order.

Your crawler should:
Start from the page: `startUrl`
Call `HtmlParser.getUrls(url)` to get all urls from a webpage of given url.

Do not crawl the same link twice.

Explore only the links that are under the same hostname as `startUrl`.

As shown in the example url above, the hostname is `example.org`. For simplicity sake, you may assume all urls use http protocol without any port specified. For example, the urls `http://leetcode.com/problems` and `http://leetcode.com/contest` are under the same hostname, while urls `http://example.org/test` and `http://example.com/abc` are not under the same hostname.

The `HtmlParser` interface is defined as such: 
interface HtmlParser {
  // Return a list of all urls from a webpage of given url.

  // This is a blocking call, that means it will do HTTP request and return when this request is finished.

  public List<String> getUrls(String url);
}
Note that `getUrls(String url)` simulates performing a HTTP request. You can treat it as a blocking function call which waits for a HTTP request to finish. It is guaranteed that `getUrls(String url)` will return the urls within 15ms.  Single-threaded solutions will exceed the time limit so, can your multi-threaded web crawler do better?
Below are two examples explaining the functionality of the problem, for custom testing purposes you'll have three variables urls`, edges` and startUrl`. Notice that you will only have access to startUrl` in your code, while urls` and edges` are not directly accessible to you in code.

Follow up:
Assume we have 10,000 nodes and 1 billion URLs to crawl. We will deploy the same software onto each node. The software can know about all the nodes. We have to minimize communication between machines and make sure each node does equal amount of work. How would your web crawler design change?
What if one node fails or does not work?
How do you know when the crawler is done?

Example 1:
Input:
urls = [
  "http://news.yahoo.com",
  "http://news.yahoo.com/news",
  "http://news.yahoo.com/news/topics/",
  "http://news.google.com",
  "http://news.yahoo.com/us"
]
edges = [[2,0],[2,1],[3,2],[3,1],[0,4]]
startUrl = "http://news.yahoo.com/news/topics/"
Output: [
  "http://news.yahoo.com",
  "http://news.yahoo.com/news",
  "http://news.yahoo.com/news/topics/",
  "http://news.yahoo.com/us"
]

Example 2:
Input: 
urls = [
  "http://news.yahoo.com",
  "http://news.yahoo.com/news",
  "http://news.yahoo.com/news/topics/",
  "http://news.google.com"
]
edges = [[0,2],[2,1],[3,2],[3,1],[3,0]]
startUrl = "http://news.google.com"
Output: ["http://news.google.com"]
Explanation: The startUrl links to all other pages that do not share the same hostname.


Constraints:
`1 <= urls.length <= 1000`
`1 <= urls[i].length <= 300`
`startUrl` is one of the `urls`.

Hostname label must be from 1 to 63 characters long, including the dots, may contain only the ASCII letters from 'a' to 'z', digits from '0' to '9' and the hyphen-minus character ('-').

The hostname may not start or end with the hyphen-minus character ('-'). 
See:  https://en.wikipedia.org/wiki/Hostname#Restrictions_on_valid_hostnames
You may assume there're no duplicates in url library.

### Solution 
 To solve the problem of creating a multi-threaded web crawler that collects URLs under the same hostname as the given `startUrl`, we can leverage Python's threading capabilities along with a breadth-first search (BFS) strategy to ensure we efficiently crawl and gather all links.

The core idea is to maintain a set of visited URLs to avoid duplicates, use a queue to explore links in a breadth-first manner, and thread workers to simulate concurrent crawling. Here’s how to implement the solution:

### Solution Breakdown

1. **Hostname Extraction**: First, we'll extract the hostname from the `startUrl`. This is crucial as we only want to crawl URLs that share the same hostname.

2. **Threading**: We'll use Python's `threading` module. We will create a worker function that processes URLs and retrieves linking URLs using the `getUrls` method.

3. **Crawling Logic**: We will maintain a queue of URLs to visit and a set of visited URLs. For every URL we visit, we will retrieve linked URLs, filter them based on the hostname, and add them to our queue if they haven't been visited.

4. **Concurrency Management**: Use a `threading.Lock` to manage access to shared resources (like the visited set) to prevent concurrent modifications.

5. **Termination**: We will wait for threads to finish processing before returning the results.

The crawling process will continue until there are no more URLs to process.

### Python Code

Here's the implementation in the LeetCode format:



```python
from threading import Thread, Lock
from collections import deque

class Solution:
    def crawl(self, startUrl: str, htmlParser) -> List[str]:
        # Parse the hostname from the startUrl
        start_hostname = self.get_hostname(startUrl)
        visited = set()
        visited.add(startUrl)
        
        queue = deque([startUrl])
        lock = Lock()
        
        def worker():
            while True:
                current_url = None
                
                with lock:
                    if queue:
                        current_url = queue.popleft()
                    else:
                        return  # No more URLs to process
                
                # Get URLs linked from the current_url
                urls = htmlParser.getUrls(current_url)
                
                for url in urls:
                    if url not in visited and self.get_hostname(url) == start_hostname:
                        with lock:
                            visited.add(url)
                            queue.append(url)
        
        threads = []
        
        # Start a fixed number of threads
        for _ in range(10):  # You can adjust the number of threads
            t = Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Wait for all threads to finish
        for t in threads:
            t.join()
        
        return list(visited)

    def get_hostname(self, url: str) -> str:
        # Extract hostname from the given URL
        return url.split('/')[2]  # This will give us the hostname part

```

### Detailed Explanation

1. **get_hostname method**: This helper function extracts the hostname from a URL by splitting the URL string.

2. **crawl method**:
   - It initializes the `visited` set and `queue` with the `startUrl`.
   - It defines the `worker` function that processes URLs in a loop:
     - It locks the queue to safely pop URLs.
     - For each URL, it retrieves linked URLs and checks if they share the same hostname and haven't been visited before.
     - Newly found URLs are added to the queue and marked as visited.
   - We create multiple threads (10 in this case, but this can be adjusted based on requirements) that run the `worker` function concurrently.
   - Finally, we join all threads to ensure they complete before returning the list of visited URLs.

### Follow-Up Considerations
1. **Distributed Crawling**: For a distributed scenario (like having multiple nodes), we would need a centralized job queue that distributes URLs among the nodes. Each node crawls independently but can report back results to a central database to minimize communication.

2. **Handling Node Failures**: We could implement a heartbeat mechanism to check if nodes are active. If a node fails, its workload can be reassigned to other nodes.

3. **Completion Notification**: A completion flag can be set when all URLs in the queue have been processed, along with a count of URLs, which can be broadcasted to indicate that crawling is done.

This implementation is designed to efficiently crawl a website while adhering to the constraints and requirements specified in the problem description.

# 1245. Tree Diameter

### Problem Description 
Given an undirected tree, return its diameter: the number of edges in a longest path in that tree.

The tree is given as an array of `edges` where `edges[i] = [u, v]` is a bidirectional edge between nodes `u` and `v`.  Each node has labels in the set `{0, 1, ..., edges.length}`.


Example 1:
Input: edges = [[0,1],[0,2]]
Output: 2
Explanation: 
A longest path of the tree is the path 1 - 0 - 2.


Example 2:
Input: edges = [[0,1],[1,2],[2,3],[1,4],[4,5]]
Output: 4
Explanation: 
A longest path of the tree is the path 3 - 2 - 1 - 4 - 5.


Constraints:
`0 <= edges.length < 10^4`
`edges[i][0] != edges[i][1]`
`0 <= edges[i][j] <= edges.length`
The given edges form an undirected tree.

### Solution 
 Here's how to approach the problem of finding the diameter of an undirected tree represented as an array of edges. The diameter of the tree is defined as the longest path between any two nodes in the tree.

### Steps to Solve the Problem:

1. **Understanding the Tree Structure**: Since a tree is a connected acyclic graph, we can use Depth First Search (DFS) or Breadth First Search (BFS) to traverse the tree and find the longest path.

2. **Two-pass DFS/BFS**: The efficient way to find the diameter is to perform a double DFS (or BFS):
   - Perform the first DFS from any arbitrary node to find the farthest node from it. Let's denote this node as `farthestNode`.
   - Then, perform a second DFS starting from `farthestNode` to determine the distance to the farthest node from `farthestNode`. This distance is the diameter of the tree.

3. **Constructing the Graph**: We need to build a graph representation from the given edges. We can use an adjacency list for this purpose.

### Implementation:

Here is the solution formatted in a way that it will run directly on the LeetCode website:



```python
from collections import defaultdict, deque

class Solution:
    def treeDiameter(self, edges: List[List[int]]) -> int:
        if not edges:
            return 0
        
        # Step 1: Build the graph as an adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        # Helper function for BFS to find the farthest node
        def bfs(start_node):
            visited = set()
            queue = deque([(start_node, 0)])  # (node, distance)
            farthest_node = start_node
            max_distance = 0
            
            while queue:
                current_node, distance = queue.popleft()
                visited.add(current_node)
                
                # Check if we've found a new farthest node
                if distance > max_distance:
                    max_distance = distance
                    farthest_node = current_node
                
                # Enqueue all unvisited neighbors
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
            
            return farthest_node, max_distance
        
        # Step 2: Perform the first BFS to find the farthest node
        farthest_node_from_start, _ = bfs(0)
        
        # Step 3: Perform the second BFS from the farthest node found
        _, diameter = bfs(farthest_node_from_start)
        
        return diameter

```

### Explanation of the Code:

1. **Graph Construction**: We build an adjacency list `graph` using `defaultdict` from the `collections` module to store connections between nodes.

2. **BFS Function**: We define the `bfs` function that accepts a starting node. It uses a queue to explore all connected nodes, keeping track of distances and the farthest node encountered during the traversal.

3. **Two BFS Calls**:
   - The first BFS finds one endpoint of the longest path (farthest node).
   - The second BFS utilizes this endpoint to determine the other endpoint's distance, which is the tree's diameter.

4. **Return the Result**: Finally, we return the calculated diameter.

### Complexity Analysis:
- **Time Complexity**: O(V + E) where V is the number of vertices (nodes) and E is the number of edges. In a tree, E = V - 1, making it linear in relation to the number of nodes.
- **Space Complexity**: O(V) to accommodate the graph representation and maintain the queue for nodes during BFS traversal.

This solution adheres to the problem constraints and efficiently finds the diameter of the tree.

# 1284. Minimum Number of Flips to Convert Binary Matrix to Zero Matrix

### Problem Description 
Given a `m x n` binary matrix `mat`. In one step, you can choose one cell and flip it and all the four neighbours of it if they exist (Flip is changing 1 to 0 and 0 to 1). A pair of cells are called neighboors if they share one edge.

Return the minimum number of steps required to convert `mat` to a zero matrix or -1 if you cannot.

Binary matrix is a matrix with all cells equal to 0 or 1 only.

Zero matrix is a matrix with all cells equal to 0.


Example 1:
Input: mat = [[0,0],[0,1]]
Output: 3
Explanation: One possible solution is to flip (1, 0) then (0, 1) and finally (1, 1) as shown.


Example 2:
Input: mat = [[0]]
Output: 0
Explanation: Given matrix is a zero matrix. We don't need to change it.


Example 3:
Input: mat = [[1,1,1],[1,0,1],[0,0,0]]
Output: 6

Example 4:
Input: mat = [[1,0,0],[1,0,0]]
Output: -1
Explanation: Given matrix can't be a zero matrix

Constraints:
`m == mat.length`
`n == mat[0].length`
`1 <= m <= 3`
`1 <= n <= 3`
`mat[i][j]` is 0 or 1.

### Solution 
 To solve the problem of converting a binary matrix into a zero matrix with the least number of flips, we need to understand that each flip of a cell affects not just that cell but also its neighboring cells. The problem space is small due to the constraints (1 <= m, n <= 3), allowing us to use a brute-force search approach.

### Approach

1. **Flipping Mechanism**: Each cell can be flipped, which will also flip its valid neighbors. This means for a selected cell `(i, j)`, we need to toggle the values of:
   - `(i-1, j)` (up)
   - `(i+1, j)` (down)
   - `(i, j-1)` (left)
   - `(i, j+1)` (right)
   - and itself `(i, j)`.

2. **Initial State Representation**: We can represent the binary matrix as a bitmask. For a 3x3 matrix, each cell can be represented using a bit in an integer, allowing us to efficiently store and manipulate the state.

3. **Brute Force Search**: Since the maximum size of the matrix is 3x3 (total of 9 bits), we can iterate through all possible combinations of flips (from 0 to 2^9 - 1). For each combination, we simulate flipping the specified cells and check if the resultant matrix becomes a zero matrix.

4. **Counting Steps**: We also need to count how many flips were made for each combination, and we will track the minimum count that leads to a zero matrix.

5. **Result Evaluation**: After evaluating all combinations, if a zero matrix was achieved, we return the minimum count of flips. If no configuration can achieve it, we return -1.

Here's the Python code to implement this solution:



```python
class Solution:
    def minFlips(self, mat):
        m, n = len(mat), len(mat[0])
        
        def toBitmask(mat):
            bitmask = 0
            for i in range(m):
                for j in range(n):
                    if mat[i][j] == 1:
                        bitmask |= (1 << (i * n + j))
            return bitmask
        
        def flip(mask, i, j):
            directions = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    mask ^= (1 << (ni * n + nj))  # Toggle the bit
            return mask
        
        initial_mask = toBitmask(mat)
        target_mask = 0  # This represents the zero matrix
        
        min_steps = float('inf')
        for flips in range(1 << (m * n)):  # Iterate through all possible flips
            current_mask = initial_mask
            steps = 0
            
            for i in range(m):
                for j in range(n):
                    if (flips & (1 << (i * n + j))) != 0:  # If we decide to flip (i, j)
                        current_mask = flip(current_mask, i, j)
                        steps += 1
            
            if current_mask == target_mask:  # Check if it becomes a zero matrix
                min_steps = min(min_steps, steps)
        
        return min_steps if min_steps != float('inf') else -1

```

### Explanation of the Code:
- **toBitmask Function**: This converts the matrix into a single integer where each bit represents whether a cell is 1 or 0.
- **flip Function**: This function flips the current bitmask based on the cell selected and its neighbors.
- In the main section (`minFlips`), we iterate through all possible flip configurations (2^9 = 512 configurations for a maximum 3x3 matrix).
- For each configuration, we count how many flips were needed and check if we reached a zero matrix, storing the minimal flips required.
  
This solution efficiently tackles the problem using bit manipulation and brute force while keeping the implementation manageable due to the small size of the input matrix.

# 1293. Shortest Path in a Grid with Obstacles Elimination

### Problem Description 
Given a `m * n` grid, where each cell is either `0` (empty) or `1` (obstacle). In one step, you can move up, down, left or right from and to an empty cell.

Return the minimum number of steps to walk from the upper left corner `(0, 0)` to the lower right corner `(m-1, n-1)` given that you can eliminate at most `k` obstacles. If it is not possible to find such walk return -1.


Example 1:
Input: 
grid = 
[[0,0,0],
 [1,1,0],
 [0,0,0],
 [0,1,1],
 [0,0,0]], 
k = 1
Output: 6
Explanation: 
The shortest path without eliminating any obstacle is 10. 
The shortest path with one obstacle elimination at position (3,2) is 6. Such path is `(0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (3,2) -> (4,2)`.


Example 2:
Input: 
grid = 
[[0,1,1],
 [1,1,1],
 [1,0,0]], 
k = 1
Output: -1
Explanation: 
We need to eliminate at least two obstacles to find such a walk.


Constraints:
`grid.length == m`
`grid[0].length == n`
`1 <= m, n <= 40`
`1 <= k <= m*n`
`grid[i][j] == 0 or 1`
`grid[0][0] == grid[m-1][n-1] == 0`

### Solution 
 To solve the problem of finding the minimum number of steps from the top-left corner to the bottom-right corner of a grid, where one can eliminate at most `k` obstacles, we can utilize a breadth-first search (BFS) approach combined with a tracking mechanism for obstacles eliminated.

### Explanation:

1. **Grid and Directional Movements**:
   - We're given a grid of size `m x n` filled with `0`s (empty) and `1`s (obstacle).
   - We can move in four directions (up, down, left, right).

2. **BFS for Shortest Path**:
   - BFS is suitable for this problem because it explores all possible paths by level, ensuring that the first time we reach a cell, we do so with the minimum steps.
   - In our BFS queue, we will store tuples containing the current cell's coordinates, the number of obstacles eliminated so far, and the current path length (number of steps taken).

3. **State Tracking**:
   - We will maintain a 3D visited array `visited[i][j][e]`, which allows us to track whether a cell `(i, j)` has been visited with `e` obstacles eliminated or not.
   - This prevents revisiting the same cell with the same number of eliminated obstacles, which can lead to infinite loops.

4. **Implementation Steps**:
   - Initialize the BFS queue with the starting point `(0, 0)` and initialize the `visited` array.
   - While there are elements in the queue:
     - Dequeue the front element and check if we've reached the bottom-right corner.
     - Explore the four directions. For each movement:
       - If it’s an empty cell, add it to the queue.
       - If it’s an obstacle and we still have eliminations left (`k`), increment the eliminated count and add it to the queue.
   - If we reach the bottom-right corner, return the number of steps taken; otherwise, return `-1`.

### Python Code:


```python
from collections import deque

class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        
        # Directions for moving up, down, left, and right
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        # Queue for BFS: (x, y, steps, eliminated)
        queue = deque([(0, 0, 0, 0)])  # (row, col, steps, obstacles_eliminated)
        
        # 3D visited array
        visited = [[[False] * (k + 1) for _ in range(n)] for _ in range(m)]
        visited[0][0][0] = True
        
        while queue:
            x, y, steps, eliminated = queue.popleft()
            
            # Check if we've reached the bottom-right corner
            if (x, y) == (m - 1, n - 1):
                return steps
            
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                
                if 0 <= new_x < m and 0 <= new_y < n:
                    if grid[new_x][new_y] == 0:  # Empty cell
                        if not visited[new_x][new_y][eliminated]:
                            visited[new_x][new_y][eliminated] = True
                            queue.append((new_x, new_y, steps + 1, eliminated))
                    
                    elif grid[new_x][new_y] == 1:  # Obstacle
                        if eliminated < k and not visited[new_x][new_y][eliminated + 1]:
                            visited[new_x][new_y][eliminated + 1] = True
                            queue.append((new_x, new_y, steps + 1, eliminated + 1))
        
        return -1  # No valid path found

```

### Key Points:
- This solution utilizes BFS to ensure we find the shortest path while efficiently managing the state of eliminated obstacles.
- The time complexity is bounded by `O(m * n * k)` due to the combination of position indices and possible eliminations, which is manageable given the constraints.

# 1298. Maximum Candies You Can Get from Boxes

### Problem Description 
Given `n` boxes, each box is given in the format `[status, candies, keys, containedBoxes]` where:
`status[i]`: an integer which is 1 if `box[i]` is open and 0 if `box[i]` is closed.

`candies[i]`: an integer representing the number of candies in `box[i]`.

`keys[i]`: an array contains the indices of the boxes you can open with the key in `box[i]`.

`containedBoxes[i]`: an array contains the indices of the boxes found in `box[i]`.

You will start with some boxes given in `initialBoxes` array. You can take all the candies in any open box and you can use the keys in it to open new boxes and you also can use the boxes you find in it.

Return the maximum number of candies you can get following the rules above.


Example 1:
Input: status = [1,0,1,0], candies = [7,5,4,100], keys = [[],[],[1],[]], containedBoxes = [[1,2],[3],[],[]], initialBoxes = [0]
Output: 16
Explanation: You will be initially given box 0. You will find 7 candies in it and boxes 1 and 2. Box 1 is closed and you don't have a key for it so you will open box 2. You will find 4 candies and a key to box 1 in box 2.

In box 1, you will find 5 candies and box 3 but you will not find a key to box 3 so box 3 will remain closed.

Total number of candies collected = 7 + 4 + 5 = 16 candy.


Example 2:
Input: status = [1,0,0,0,0,0], candies = [1,1,1,1,1,1], keys = [[1,2,3,4,5],[],[],[],[],[]], containedBoxes = [[1,2,3,4,5],[],[],[],[],[]], initialBoxes = [0]
Output: 6
Explanation: You have initially box 0. Opening it you can find boxes 1,2,3,4 and 5 and their keys. The total number of candies will be 6.


Example 3:
Input: status = [1,1,1], candies = [100,1,100], keys = [[],[0,2],[]], containedBoxes = [[],[],[]], initialBoxes = [1]
Output: 1

Example 4:
Input: status = [1], candies = [100], keys = [[]], containedBoxes = [[]], initialBoxes = []
Output: 0

Example 5:
Input: status = [1,1,1], candies = [2,3,2], keys = [[],[],[]], containedBoxes = [[],[],[]], initialBoxes = [2,1,0]
Output: 7

Constraints:
`1 <= status.length <= 1000`
`status.length == candies.length == keys.length == containedBoxes.length == n`
`status[i]` is `0` or `1`.

`1 <= candies[i] <= 1000`
`0 <= keys[i].length <= status.length`
`0 <= keys[i][j] < status.length`
All values in `keys[i]` are unique.

`0 <= containedBoxes[i].length <= status.length`
`0 <= containedBoxes[i][j] < status.length`
All values in `containedBoxes[i]` are unique.

Each box is contained in one box at most.

`0 <= initialBoxes.length <= status.length`
`0 <= initialBoxes[i] < status.length`

### Solution 
 To solve this problem, we need to explore the boxes starting from the initial boxes we have available. We can use a breadth-first search (BFS) approach to efficiently collect candies while managing our keys and discovering new boxes.

Here's a step-by-step explanation of how we can implement this:

1. **Initialization**: We begin by initializing a few collections:
   - A sum variable `total_candies` to keep track of the total candies we gather.
   - A set `visited` to ensure we don't process the same box more than once.
   - A queue to manage which boxes we can explore (starting with the `initialBoxes`).
   - A list of keys that we collect from the opened boxes to discover new boxes.

2. **BFS Process**:
   - While there are boxes to explore in the queue:
     - For each box at the front of the queue, check if it is unopened (`status[box] == 1`) and hasn't been visited.
     - If it's open, collect its candies and mark it as visited. Retrieve any new keys and contained boxes from this box.
     - Add all the keys to a set (to track which boxes we can potentially open).
     - For every contained box, if we haven't visited it yet, we add it to a temporary collection of boxes to be explored later if we have the necessary keys.

3. **Using Keys to Open Boxes**: 
   - After exploring open boxes, check the collected keys to open any now accessible boxes. This is done in a loop until we no longer find any new boxes to open.
   
4. **Final Result**: Return the total candies collected.

Here’s how this looks in Python code:



```python
from collections import deque

def maxCandies(status, candies, keys, containedBoxes, initialBoxes):
    total_candies = 0
    visited = set()
    queue = deque(initialBoxes)
    keys_collected = set()
    
    while queue:
        # Try to collect boxes that can be accessed
        current_boxes = []
        
        while queue:
            box = queue.popleft()
            
            # Check if the box is already visited
            if box in visited:
                continue
            
            # If box is open, process it
            if status[box] == 1:
                total_candies += candies[box]
                visited.add(box)
                
                # Collect keys
                for key in keys[box]:
                    keys_collected.add(key)
                
                # Collect contained boxes
                for contained in containedBoxes[box]:
                    current_boxes.append(contained)

        # Now open the boxes we just collected keys for
        queue.extend(box for box in current_boxes if box not in visited)

        # Open any new boxes with keys
        new_boxes_to_open = []
        for key in keys_collected:
            if key not in visited and status[key] == 1:
                new_boxes_to_open.append(key)

        # Update the queue with new keys and try to open boxes
        queue.extend(new_boxes_to_open)
        
    return total_candies

```

### Detailed Explanation of the Code:
- **Line 1**: We import `deque` from `collections` for our queue implementation.
- **Function `maxCandies`**: Takes the lists for `status`, `candies`, `keys`, `containedBoxes`, and the `initialBoxes` as arguments.
- **Variables**:
  - `total_candies`: Accumulates all candies collected.
  - `visited`: A set to keep track of processed boxes.
  - `queue`: Initially populated by the `initialBoxes`, tracks boxes to be opened.
  - `keys_collected`: A set for the keys we have gained during our exploration.
  
- **Main Loop**: 
  - Continuously processes the queue until there are no more boxes to open.
  - Each box is checked if it's been visited and if it's open. If so, candies are collected, and contained items are gathered for further processing.

- **Final Return**: After exhausting all possible box openings, it returns the total count of candies.

This implementation efficiently handles the box exploration, key management, and candy collection using a BFS-like strategy and runs in a linear time relative to the number of boxes, ensuring it's optimal for the given constraints.

# 1306. Jump Game III

### Problem Description 
Given an array of non-negative integers `arr`, you are initially positioned at `start` index of the array. When you are at index `i`, you can jump to `i + arr[i]` or `i - arr[i]`, check if you can reach to any index with value 0.

Notice that you can not jump outside of the array at any time.


Example 1:
Input: arr = [4,2,3,0,3,1,2], start = 5
Output: true
Explanation: 
All possible ways to reach at index 3 with value 0 are: 
index 5 -> index 4 -> index 1 -> index 3 
index 5 -> index 6 -> index 4 -> index 1 -> index 3 

Example 2:
Input: arr = [4,2,3,0,3,1,2], start = 0
Output: true 
Explanation: 
One possible way to reach at index 3 with value 0 is: 
index 0 -> index 4 -> index 1 -> index 3

Example 3:
Input: arr = [3,0,2,1,2], start = 2
Output: false
Explanation: There is no way to reach at index 1 with value 0.


Constraints:
`1 <= arr.length <= 5 * 104`
`0 <= arr[i] < arr.length`
`0 <= start < arr.length`

### Solution 
 To solve the problem of determining if we can reach an index with a value of `0` in the given array using valid jumps, we can utilize a Breadth-First Search (BFS) approach. This problem is well-suited for BFS because we treat the indices of the array as nodes in a graph and the potential jumps as edges between these nodes.

Here's the plan:
1. Start from the given index `start`.
2. Use a queue to explore each index we can jump to.
3. Keep track of the indices we've already visited to avoid cycles and redundant operations.
4. For each index, check whether its value is `0`. If it is, return `True`.
5. If we have explored all reachable indices and haven't found a `0`, return `False`.

### Python Code

Here is the complete Python code following the LeetCode solution format:



```python
from collections import deque

class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        queue = deque([start])  # Start with the initial index
        visited = set()  # Use a set to track visited indices
        visited.add(start)  # Mark the start index as visited
        
        while queue:
            index = queue.popleft()  # Get the current index
            
            # Check if the current index value is 0
            if arr[index] == 0:
                return True
            
            # Determine the possible jumps
            forward_jump = index + arr[index]
            backward_jump = index - arr[index]
            
            # If the forward jump is within bounds and not visited
            if forward_jump < len(arr) and forward_jump not in visited:
                visited.add(forward_jump)
                queue.append(forward_jump)
            
            # If the backward jump is within bounds and not visited
            if backward_jump >= 0 and backward_jump not in visited:
                visited.add(backward_jump)
                queue.append(backward_jump)
        
        return False  # If we exhaust the queue without finding a 0

```

### Explanation

1. **Initialization**:
   - We import the `deque` class from the `collections` module for efficient queue operations.
   - A class `Solution` is created containing the method `canReach`.

2. **Method Declaration**:
   - The method `canReach` takes two parameters: `arr` (the input array) and `start` (the starting index).

3. **Queue and Visited Set**:
   - A deque is initialized for BFS and starts with the `start` index.
   - A set named `visited` is used to record indices we've already processed.

4. **BFS Loop**:
   - While the queue is not empty, we pop the leftmost index (`index`).
   - We check if the value at `arr[index]` is `0`. If yes, we return `True`.
   - We calculate `forward_jump` and `backward_jump` based on the current index.
   - For both jumps, we check whether they are within the bounds of the array and whether they have already been visited. If valid, we add them to the queue and mark them as visited.

5. **Termination**:
   - If we exhaust the queue without finding an index with a value of `0`, we return `False`.

### Complexity
- **Time Complexity**: O(N), where N is the length of the array, since in the worst case we check each index once.
- **Space Complexity**: O(N), for the queue and the visited set, since both structures may store up to N indices in the worst case.

This implementation efficiently determines if a jump to an index with a value of `0` is possible from the starting position.

# 1311. Get Watched Videos by Your Friends

### Problem Description 
There are `n` people, each person has a unique id between `0` and `n-1`. Given the arrays `watchedVideos` and `friends`, where `watchedVideos[i]` and `friends[i]` contain the list of watched videos and the list of friends respectively for the person with `id = i`.

Level 1 of videos are all watched videos by your friends, level 2 of videos are all watched videos by the friends of your friends and so on. In general, the level `k` of videos are all watched videos by people with the shortest path exactly equal to `k` with you. Given your `id` and the `level` of videos, return the list of videos ordered by their frequencies (increasing). For videos with the same frequency order them alphabetically from least to greatest. 

Example 1:
Input: watchedVideos = [["A","B"],["C"],["B","C"],["D"]], friends = [[1,2],[0,3],[0,3],[1,2]], id = 0, level = 1
Output: ["B","C"] 
Explanation: 
You have id = 0 (green color in the figure) and your friends are (yellow color in the figure):
Person with id = 1 -> watchedVideos = ["C"] 
Person with id = 2 -> watchedVideos = ["B","C"] 
The frequencies of watchedVideos by your friends are: 
B -> 1 
C -> 2

Example 2:
Input: watchedVideos = [["A","B"],["C"],["B","C"],["D"]], friends = [[1,2],[0,3],[0,3],[1,2]], id = 0, level = 2
Output: ["D"]
Explanation: 
You have id = 0 (green color in the figure) and the only friend of your friends is the person with id = 3 (yellow color in the figure).


Constraints:
`n == watchedVideos.length == friends.length`
`2 <= n <= 100`
`1 <= watchedVideos[i].length <= 100`
`1 <= watchedVideos[i][j].length <= 8`
`0 <= friends[i].length < n`
`0 <= friends[i][j] < n`
`0 <= id < n`
`1 <= level < n`
if `friends[i]` contains `j`, then `friends[j]` contains `i`

### Solution 
 To solve the problem of determining which videos your friends (and friends of friends) have watched at a given level, we can approach it using a breadth-first search (BFS) strategy. Let's break down the solution step-by-step:

### Problem Breakdown

1. **Input Structure**:
    - `watchedVideos`: A list where each list at index `i` contains the videos watched by person `i`.
    - `friends`: A list where each list at index `i` contains the friends of person `i`.
    - `id`: The ID of the person for whom we want to find watched videos.
    - `level`: The level of friends we want to consider (i.e., direct friends for level `1`, friends of friends for level `2`, etc.).

2. **Output**:
    - A sorted list of watched videos at the specified level by the person's friends.

3. **BFS for Traversing Friend Levels**:
    - Start from the given person's friends and expand out to the specified level, collecting all watched videos at that level.
    - Use a queue for BFS to keep track of the current friends to explore and a set to track visited friends to avoid cycles.

4. **Counting Video Frequency**:
    - As we gather videos from friends at the desired level, we can use a dictionary to count the occurrences of each video.

5. **Sorting**:
    - Finally, sort the videos first by frequency and then alphabetically.

### Implementation

Here's the implementation of the above logic in Python, formatted to run directly on the LeetCode platform.



```python
from collections import defaultdict, deque

def watchedVideosByFriends(watchedVideos, friends, id, level):
    # Store the videos watched at the specified level
    video_count = defaultdict(int)
    
    # Use a queue for BFS
    queue = deque([id])
    visited = set([id])
    
    # Perform BFS to find friends up to the specified level
    current_level = 0
    
    while queue:
        if current_level == level:
            break  # We stop the BFS if we have reached the desired level
            
        for _ in range(len(queue)):
            person = queue.popleft()
            for friend in friends[person]:
                if friend not in visited:
                    visited.add(friend)
                    queue.append(friend)
        
        current_level += 1
    
    # If we have reached the desired level, count watched videos
    while queue:
        person = queue.popleft()
        for video in watchedVideos[person]:
            video_count[video] += 1
    
    # Sort videos first by frequency, then by name
    result = sorted(video_count.keys(), key=lambda video: (video_count[video], video))
    
    return result

```

### Explanation of the Code

1. **Imports**:
    - Import `defaultdict` and `deque` from `collections` for convenient data handling.

2. **Function Definition**:
    - The function `watchedVideosByFriends` takes `watchedVideos`, `friends`, `id`, and `level` as arguments.

3. **Data Structures**:
    - `video_count`: A dictionary to count the frequency of videos.
    - `queue`: A deque initialized with the `id` of the person, used for BFS.
    - `visited`: A set to track visited friends to prevent cycles.

4. **BFS Logic**:
    - We loop until the queue is empty or we reach the desired `level`.
    - For each level, we dequeue friends and enqueue their friends that haven't been visited.

5. **Count Videos at Level**:
    - When we reach the desired level, we continue to gather watched videos from the friends at that level and update their counts in `video_count`.

6. **Sorting and Return**:
    - Finally, we sort the keys of the `video_count` dictionary by their frequency and then alphabetically, returning the sorted list.

This function is efficient, easy to understand, and adheres to the constraints and requirements specified in the problem statement.

# 1319. Number of Operations to Make Network Connected

### Problem Description 
There are `n` computers numbered from `0` to `n-1` connected by ethernet cables `connections` forming a network where `connections[i] = [a, b]` represents a connection between computers `a` and `b`. Any computer can reach any other computer directly or indirectly through the network.

Given an initial computer network `connections`. You can extract certain cables between two directly connected computers, and place them between any pair of disconnected computers to make them directly connected. Return the minimum number of times you need to do this in order to make all the computers connected. If it's not possible, return -1. 

Example 1:
Input: n = 4, connections = [[0,1],[0,2],[1,2]]
Output: 1
Explanation: Remove cable between computer 1 and 2 and place between computers 1 and 3.


Example 2:
Input: n = 6, connections = [[0,1],[0,2],[0,3],[1,2],[1,3]]
Output: 2

Example 3:
Input: n = 6, connections = [[0,1],[0,2],[0,3],[1,2]]
Output: -1
Explanation: There are not enough cables.


Example 4:
Input: n = 5, connections = [[0,1],[0,2],[3,4],[2,3]]
Output: 0

Constraints:
`1 <= n <= 10^5`
`1 <= connections.length <= min(n*(n-1)/2, 10^5)`
`connections[i].length == 2`
`0 <= connections[i][0], connections[i][1] < n`
`connections[i][0] != connections[i][1]`
There are no repeated connections.

No two computers are connected by more than one cable.

### Solution 
 To solve the problem of connecting all computers by potentially moving cables, we can use a graph-based approach. We will determine how many disconnected components exist in the graph formed by the computers and connections.

### Detailed Explanation

1. **Graph Representation**:
   Each computer can be represented as a node, and each connection as an edge in an undirected graph.

2. **Finding Components**:
   To figure out how many separate groups (or components) of connected computers there are, we can utilize DFS (Depth First Search) or Union-Find data structures.

3. **Calculating Moves**:
   Once we know the number of connected components, the minimum number of moves needed to connect `c` components into a single component is `c - 1`. However, we must also ensure that the number of existing connections (edges) is sufficient. If the number of connections `m` is less than `c - 1` (where `c` is the number of disconnected components), it is impossible to connect all computers, so we return -1.

4. **Implementation Steps**:
   - If there are enough connections (`m >= c - 1`), we calculate the number of moves needed.
   - Otherwise, return -1.

Now, let's implement this logic in Python:



```python
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        if len(connections) < n - 1:
            return -1
        
        # Create an adjacency list for the graph.
        graph = defaultdict(list)
        
        for a, b in connections:
            graph[a].append(b)
            graph[b].append(a)
        
        # To keep track of visited computers.
        visited = [False] * n
        
        def dfs(node):
            stack = [node]
            while stack:
                curr = stack.pop()
                for neighbor in graph[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)

        # Count the number of connected components.
        components = 0
        
        for i in range(n):
            if not visited[i]:
                components += 1
                visited[i] = True
                dfs(i)
        
        # The number of moves needed to connect all components is components - 1.
        return components - 1


```

### Explanation of the Code

- **Graph Setup**: We use a defaultdict from the `collections` module to create an adjacency list representing our computer network.
- **DFS Helper Function**: We define a DFS function to explore all connected nodes starting from an unvisited node, marking them as visited.
- **Component Count**: We iterate through all computers. Each time we find an unvisited computer, we count it as a new component and run DFS from that computer.
- **Return Result**: Finally, we check if the number of edges is sufficient (i.e., if the number of connections is at least `n - 1`). If not, we return -1; otherwise, we return the number of moves needed to connect all components.

This code is optimized for handling the largest constraints specified in the problem and operates efficiently within the given limits.

# 1345. Jump Game IV

### Problem Description 
Given an array of integers `arr`, you are initially positioned at the first index of the array.

In one step you can jump from index `i` to index:
`i + 1` where: `i + 1 < arr.length`.

`i - 1` where: `i - 1 >= 0`.

`j` where: `arr[i] == arr[j]` and `i != j`.

Return the minimum number of steps to reach the last index of the array.

Notice that you can not jump outside of the array at any time.


Example 1:
Input: arr = [100,-23,-23,404,100,23,23,23,3,404]
Output: 3
Explanation: You need three jumps from index 0 --> 4 --> 3 --> 9. Note that index 9 is the last index of the array.


Example 2:
Input: arr = [7]
Output: 0
Explanation: Start index is the last index. You don't need to jump.


Example 3:
Input: arr = [7,6,9,6,9,6,9,7]
Output: 1
Explanation: You can jump directly from index 0 to index 7 which is last index of the array.


Example 4:
Input: arr = [6,1,9]
Output: 2

Example 5:
Input: arr = [11,22,7,7,7,7,7,7,7,22,13]
Output: 3

Constraints:
`1 <= arr.length <= 5 * 104`
`-108 <= arr[i] <= 108`

### Solution 
 To solve the problem of finding the minimum number of jumps to reach the last index of the array, we can model this scenario as a graph traversal problem. Each index of the array represents a node, and the possible jumps represent edges in the graph.

We can use a Breadth-First Search (BFS) approach because we need the shortest path in an unweighted graph. Here's a breakdown of the solution:

1. **Initialization**: We'll maintain a queue to explore indices. We'll also use a set to keep track of visited indices to avoid processing the same index multiple times, and a dictionary to map the values of the array to the list of indices where that value occurs. This will help us jump efficiently to all indices that have the same value.

2. **BFS Traversal**: Start from the first index (index 0) and explore its neighbors. Neighbors can be the next index (`i + 1`), the previous index (`i - 1`), or any index `j` where `arr[i] == arr[j]`.

3. **Counting Steps**: Each time we process elements from the queue, we also keep a count of the number of steps taken. When we reach the last index, we return the current count.

4. **Cleaning Up**: After exploring all indices with the same value, we can remove them from the map to prevent unnecessary jumps in the future.

Here's how the solution can be implemented in Python:



```python
from collections import deque, defaultdict

class Solution:
    def minJumps(self, arr):
        n = len(arr)
        if n == 1:
            return 0  # Already at the last index

        # Edge case: Map values to their indices
        graph = defaultdict(list)
        for i in range(n):
            graph[arr[i]].append(i)

        visited = set()  # To mark visited indices
        queue = deque([0])  # Start BFS from the first index
        visited.add(0)
        steps = 0
        
        while queue:
            steps += 1  # Increase step counter for each layer of BFS
            for _ in range(len(queue)):
                index = queue.popleft()
                
                # Check if we reached the last index
                if index == n - 1:
                    return steps
                
                # Check next index
                if index + 1 < n and index + 1 not in visited:
                    visited.add(index + 1)
                    queue.append(index + 1)
                
                # Check previous index
                if index - 1 >= 0 and index - 1 not in visited:
                    visited.add(index - 1)
                    queue.append(index - 1)
                
                # Check other indices with the same value
                if arr[index] in graph:
                    for neighbor in graph[arr[index]]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                    
                    # Clear the list to prevent future unnecessary jumps
                    del graph[arr[index]]
        
        return -1  # Should never reach here for valid input

```

### Explanation of the Code:

- **Imports**: We use `deque` from the `collections` module for efficient FIFO queue operations and `defaultdict` to easily manage neighbors indexed by values.
  
- **Input Handling**: We check if the input array length is 1, in which case no jumps are needed.

- **Graph Construction**: We build a mapping of values to their corresponding indices.

- **BFS Setup**: We initialize our BFS queue and tracking structures.

- **BFS Loop**: The main loop processes each index, checking potential jumps to the next and previous indices, as well as all indices with the same value.

- **Return Value**: As soon as we reach the last index, we return the number of steps taken.

### Complexity:
- Time Complexity: O(n), where n is the length of the array. Each index is processed at most once.
- Space Complexity: O(n) for the queue and visited set as well as the storage of indices by value.

This solution follows the LeetCode format and should run correctly on their platform.

# 1368. Minimum Cost to Make at Least One Valid Path in a Grid

### Problem Description 
Given a m x n `grid`. Each cell of the `grid` has a sign pointing to the next cell you should visit if you are currently in this cell. The sign of `grid[i][j]` can be:
1 which means go to the cell to the right. (i.e go from `grid[i][j]` to `grid[i][j + 1]`)
2 which means go to the cell to the left. (i.e go from `grid[i][j]` to `grid[i][j - 1]`)
3 which means go to the lower cell. (i.e go from `grid[i][j]` to `grid[i + 1][j]`)
4 which means go to the upper cell. (i.e go from `grid[i][j]` to `grid[i - 1][j]`)
Notice that there could be some invalid signs on the cells of the `grid` which points outside the `grid`.

You will initially start at the upper left cell `(0,0)`. A valid path in the grid is a path which starts from the upper left cell `(0,0)` and ends at the bottom-right cell `(m - 1, n - 1)` following the signs on the grid. The valid path doesn't have to be the shortest.

You can modify the sign on a cell with `cost = 1`. You can modify the sign on a cell one time only.

Return the minimum cost to make the grid have at least one valid path.


Example 1:
Input: grid = [[1,1,1,1],[2,2,2,2],[1,1,1,1],[2,2,2,2]]
Output: 3
Explanation: You will start at point (0, 0).

The path to (3, 3) is as follows. (0, 0) --> (0, 1) --> (0, 2) --> (0, 3) change the arrow to down with cost = 1 --> (1, 3) --> (1, 2) --> (1, 1) --> (1, 0) change the arrow to down with cost = 1 --> (2, 0) --> (2, 1) --> (2, 2) --> (2, 3) change the arrow to down with cost = 1 --> (3, 3)
The total cost = 3.


Example 2:
Input: grid = [[1,1,3],[3,2,2],[1,1,4]]
Output: 0
Explanation: You can follow the path from (0, 0) to (2, 2).


Example 3:
Input: grid = [[1,2],[4,3]]
Output: 1

Example 4:
Input: grid = [[2,2,2],[2,2,2]]
Output: 3

Example 5:
Input: grid = [[4]]
Output: 0

Constraints:
`m == grid.length`
`n == grid[i].length`
`1 <= m, n <= 100`

### Solution 
 To solve the problem of finding the minimum cost to create a valid path in the given grid, we can use a graph-based algorithm like Dijkstra's algorithm. The grid cells can be represented as nodes in a graph where edges between nodes represent valid moves based on the signs in the cells.

### Approach:

1. **Graph Representation**: Each cell in the grid is a node. From each cell, we can determine its neighbors based on the arrow (sign). If the sign points to a valid cell, we don't incur a cost to move there; if it points out of bounds or to an invalid cell, we must modify the sign of the current cell, which incurs a cost of 1.

2. **Using Dijkstra's Algorithm**: 
   - We initialize a priority queue (min-heap) that helps us explore paths in order of their costs.
   - We start from the top-left corner and keep track of the minimum cost to reach each cell in the grid using a cost matrix initialized to infinity (except for the starting cell, initialized to 0).
   - At each step, we check the cell we are currently at, see where it guides us, and either move there (if the move is valid) or modify the current cell to point to a valid direction if necessary and add the cost.

3. **Ending Condition**: The search will continue until we reach the bottom-right corner of the grid, and the cost associated with that cell will be our answer.

### Implementation:



```python
import heapq

class Solution:
    def minCost(self, grid):
        m, n = len(grid), len(grid[0])
        # Directions for the sign (1: right, 2: left, 3: down, 4: up)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # Initialize cost matrix with infinite cost
        cost = [[float('inf')] * n for _ in range(m)]
        cost[0][0] = 0
        # Min-Heap for Dijkstra's algorithm
        heap = [(0, 0, 0)]  # (current cost, x, y)

        while heap:
            current_cost, x, y = heapq.heappop(heap)

            # If we have reached the bottom-right corner, return the cost
            if x == m - 1 and y == n - 1:
                return current_cost
            
            # If the cost is greater than the recorded cost, skip it
            if current_cost > cost[x][y]:
                continue

            # Check the direction pointed by the current cell
            sign = grid[x][y] - 1  # Convert from 1-4 to 0-3
            dx, dy = directions[sign]
            new_x, new_y = x + dx, y + dy

            # Check if the move is valid
            if 0 <= new_x < m and 0 <= new_y < n:
                # Valid move without modification
                if current_cost < cost[new_x][new_y]:
                    cost[new_x][new_y] = current_cost
                    heapq.heappush(heap, (current_cost, new_x, new_y))
            
            # Invalid move, we will consider modifying this cell
            for direction in range(4):
                dx, dy = directions[direction]
                new_x, new_y = x + dx, y + dy
                
                if 0 <= new_x < m and 0 <= new_y < n:
                    # Non-Invalid move, count it as cost + 1
                    if current_cost + 1 < cost[new_x][new_y]:
                        cost[new_x][new_y] = current_cost + 1
                        heapq.heappush(heap, (current_cost + 1, new_x, new_y))
        
        return -1  # In case the bottom-right corner is unreachable

```

### Explanation of the code:
- We set up a priority queue to explore the cell with the least cost next, starting from the top left.
- We use a `while` loop to continue exploring until we either find the bottom-right corner or exhaust all possibilities.
- The cost to modify a cell or follow the sign is tracked, and we update our priority queue accordingly to ensure we always explore the least costly paths first.

This code follows the LeetCode format and can be directly run there. It efficiently uses Dijkstra's algorithm to minimize the cost of making the path from the top-left to the bottom-right valid.

# 1379. Find a Corresponding Node of a Binary Tree in a Clone of That Tree

### Problem Description 
Given two binary trees `original` and `cloned` and given a reference to a node `target` in the original tree.

The `cloned` tree is a copy of the `original` tree.

Return a reference to the same node in the `cloned` tree.

Note that you are not allowed to change any of the two trees or the `target` node and the answer must be a reference to a node in the `cloned` tree.

Follow up: Solve the problem if repeated values on the tree are allowed.


Example 1:
Input: tree = [7,4,3,null,null,6,19], target = 3
Output: 3
Explanation: In all examples the original and cloned trees are shown. The target node is a green node from the original tree. The answer is the yellow node from the cloned tree.


Example 2:
Input: tree = [7], target =  7
Output: 7

Example 3:
Input: tree = [8,null,6,null,5,null,4,null,3,null,2,null,1], target = 4
Output: 4

Example 4:
Input: tree = [1,2,3,4,5,6,7,8,9,10], target = 5
Output: 5

Example 5:
Input: tree = [1,2,null,3], target = 2
Output: 2

Constraints:
The number of nodes in the `tree` is in the range `[1, 10^4]`.

The values of the nodes of the `tree` are unique.

`target` node is a node from the `original` tree and is not `null`.

### Solution 
 To solve the problem of finding the corresponding node in the cloned binary tree that matches a given target node in the original binary tree, we can utilize Depth First Search (DFS) traversal. Here’s a detailed explanation and implementation of the solution in Python that can be run directly on the LeetCode platform:

### Explanation

1. **Problem Understanding**:
    - We have two binary trees: `original` and `cloned`. The `cloned` tree is a replica of `original`.
    - We are given a reference to a node called `target` from the `original` tree.
    - The goal is to find the exact same node in the `cloned` tree that corresponds to the `target` node.

2. **Tree Traversal Approach**:
    - We can traverse the `original` and `cloned` trees simultaneously, exploring them using DFS (either recursively or iteratively).
    - When we find the `target` node in the `original` tree, we can return the corresponding node from the `cloned` tree.

3. **Edge Cases**:
    - The constraints guarantee that the `target` node exists in the `original` tree.
    - Since values are unique, once we find the node in `original`, we can directly refer to the `cloned` tree for the matching node.

### Implementation

Here’s how we can implement the solution:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
        # Use a helper function for DFS traversal
        def dfs(original_node, cloned_node):
            if not original_node:
                return None
            
            # Check if the current original node is the target
            if original_node is target:
                return cloned_node
            
            # Search in the left subtree
            left_result = dfs(original_node.left, cloned_node.left)
            if left_result:
                return left_result
            
            # Search in the right subtree
            return dfs(original_node.right, cloned_node.right)

        # Start the DFS traversal from the root nodes
        return dfs(original, cloned)


```

### Explanation of the Code:

- **TreeNode Class**: This class defines the structure of the nodes in the binary tree.
- **Solution Class**: This class contains the method `getTargetCopy` which takes three parameters: `original`, `cloned`, and `target`.
- **dfs Function**: A recursive function that takes the current nodes of both the `original` and `cloned` trees.
  - If the `original_node` is `None`, it returns `None`.
  - If the current `original_node` is the target node, it returns the corresponding `cloned_node`.
  - It recursively searches the left subtree first. If a match is found in the left subtree, that result is returned.
  - If not found in the left subtree, it checks the right subtree.

### Complexity:
- **Time Complexity**: O(N), where N is the number of nodes in the tree, since in the worst case, we may have to visit all nodes.
- **Space Complexity**: O(H), where H is the height of the tree, due to the recursive call stack.

This straightforward DFS approach ensures we accurately find and return the corresponding node in the cloned tree, adhering to all constraints and requirements described in the problem statement.


# 1391. Check if There is a Valid Path in a Grid

### Problem Description 
Given a m x n `grid`. Each cell of the `grid` represents a street. The street of `grid[i][j]` can be:
1 which means a street connecting the left cell and the right cell.

2 which means a street connecting the upper cell and the lower cell.

3 which means a street connecting the left cell and the lower cell.

4 which means a street connecting the right cell and the lower cell.

5 which means a street connecting the left cell and the upper cell.

6 which means a street connecting the right cell and the upper cell.

You will initially start at the street of the upper-left cell `(0,0)`. A valid path in the grid is a path which starts from the upper left cell `(0,0)` and ends at the bottom-right cell `(m - 1, n - 1)`. The path should only follow the streets.

Notice that you are not allowed to change any street.

Return true if there is a valid path in the grid or false otherwise.


Example 1:
Input: grid = [[2,4,3],[6,5,2]]
Output: true
Explanation: As shown you can start at cell (0, 0) and visit all the cells of the grid to reach (m - 1, n - 1).


Example 2:
Input: grid = [[1,2,1],[1,2,1]]
Output: false
Explanation: As shown you the street at cell (0, 0) is not connected with any street of any other cell and you will get stuck at cell (0, 0)

Example 3:
Input: grid = [[1,1,2]]
Output: false
Explanation: You will get stuck at cell (0, 1) and you cannot reach cell (0, 2).


Example 4:
Input: grid = [[1,1,1,1,1,1,3]]
Output: true

Example 5:
Input: grid = [[2],[2],[2],[2],[2],[2],[6]]
Output: true

Constraints:
`m == grid.length`
`n == grid[i].length`
`1 <= m, n <= 300`
`1 <= grid[i][j] <= 6`

### Solution 
 To solve the problem of determining if there is a valid path from the top-left cell `(0, 0)` to the bottom-right cell `(m - 1, n - 1)` in a grid where each cell defines which other cells it can connect to, we can employ Depth-First Search (DFS). 

Here’s a breakdown of the approach:

### Step-by-Step Explanation

1. **Understanding the Connections**:
   Each cell in the grid can be one of 6 types, which define the possible movements:
   - **1**: Connects to left and right
   - **2**: Connects to up and down
   - **3**: Connects to left and down
   - **4**: Connects to right and down
   - **5**: Connects to left and up
   - **6**: Connects to right and up

   We can map these connections to direction vectors for easy traversal:
   - For example:
     - **1** allows movements to `(0, -1)` and `(0, 1)`.
     - **2** allows movements to `(-1, 0)` and `(1, 0)`.

2. **DFS Implementation**:
   We'll implement a DFS function that will explore all valid paths. We must keep track of visited cells to avoid cycles.

3. **Boundary Conditions**:
   Ensure that we don’t move outside the grid's boundaries and take care to only move to cells that can be reached based on the current cell's connection type.

4. **Return Result**:
   If we reach the bottom-right corner, we return `True`; otherwise, after exhausting all options without reaching it, we return `False`.

### Implementation in Python



```python
class Solution:
    def hasValidPath(self, grid):
        from collections import deque
        
        # Directions corresponding to each grid value
        directions = {
            1: [(0, -1), (0, 1)],  # left, right
            2: [(-1, 0), (1, 0)],  # up, down
            3: [(0, -1), (1, 0)],  # left, down
            4: [(0, 1), (1, 0)],   # right, down
            5: [(0, -1), (-1, 0)], # left, up
            6: [(0, 1), (-1, 0)],  # right, up
        }
        
        m, n = len(grid), len(grid[0])  # Dimensions of grid
        visited = [[False] * n for _ in range(m)]  # Visited cells tracking

        def in_bounds(x, y):
            return 0 <= x < m and 0 <= y < n
        
        def can_move(from_cell, to_cell, direction):
            # Check if there's a street connected in the direction
            if from_cell == 1 and direction in [(0, -1), (0, 1)]:  # from 1
                return to_cell in [1, 3, 5]  # can go to 1, 3, 5
            elif from_cell == 2 and direction in [(-1, 0), (1, 0)]:  # from 2
                return to_cell in [2, 3, 4]  # can go to 2, 3, 4
            elif from_cell == 3 and direction == (0, -1):  # from 3
                return to_cell in [1, 5]  # can go to 1, 5 
            elif from_cell == 3 and direction == (1, 0):  # from 3
                return to_cell in [2, 4]  # can go to 2, 4
            elif from_cell == 4 and direction == (0, 1):  # from 4
                return to_cell in [1, 6]  # can go to 1, 6
            elif from_cell == 4 and direction == (1, 0):  # from 4
                return to_cell in [2, 6]  # can go to 2, 6
            elif from_cell == 5 and direction == (0, -1):  # from 5
                return to_cell in [1, 3]  # can go to 1, 3
            elif from_cell == 5 and direction == (-1, 0):  # from 5
                return to_cell in [2, 6]  # can go to 2, 6
            elif from_cell == 6 and direction == (0, 1):  # from 6
                return to_cell in [1, 4]  # can go to 1, 4
            elif from_cell == 6 and direction == (-1, 0):  # from 6
                return to_cell in [2, 5]  # can go to 2, 5
            return False

        # Using a stack for DFS
        stack = deque([(0, 0)])
        visited[0][0] = True

        while stack:
            x, y = stack.pop()
            if (x, y) == (m - 1, n - 1):  # Reached bottom-right corner
                return True
            
            # Explore each direction the current cell allows
            for dx, dy in directions[grid[x][y]]:
                nx, ny = x + dx, y + dy
                if in_bounds(nx, ny) and not visited[nx][ny] and can_move(grid[x][y], grid[nx][ny], (dx, dy)):
                    visited[nx][ny] = True
                    stack.append((nx, ny))

        return False  # No valid path found

```

### Key Points:
- **DFS**: Used to explore all possible paths.
- **Visited Tracking**: Prevents revisiting cells that have already been explored.
- **Directional Movement**: Each type of cell allows specific moves which are defined in maps.

This solution effectively finds if there’s a valid path in the grid using depth-first traversal based on pre-defined movement rules, keeping it efficient and clear.

# 1485. Clone Binary Tree With Random Pointer

### Problem Description 
A binary tree is given such that each node contains an additional random pointer which could point to any node in the tree or null.

Return a deep copy of the tree.

The tree is represented in the same input/output way as normal binary trees where each node is represented as a pair of `[val, random_index]` where:
`val`: an integer representing `Node.val`
`random_index`: the index of the node (in the input) where the random pointer points to, or `null` if it does not point to any node.

You will be given the tree in class `Node` and you should return the cloned tree in class `NodeCopy`. `NodeCopy` class is just a clone of `Node` class with the same attributes and constructors.


Example 1:
Input: root = [[1,null],null,[4,3],[7,0]]
Output: [[1,null],null,[4,3],[7,0]]
Explanation: The original binary tree is [1,null,4,7].

The random pointer of node one is null, so it is represented as [1, null].

The random pointer of node 4 is node 7, so it is represented as [4, 3] where 3 is the index of node 7 in the array representing the tree.

The random pointer of node 7 is node 1, so it is represented as [7, 0] where 0 is the index of node 1 in the array representing the tree.


Example 2:
Input: root = [[1,4],null,[1,0],null,[1,5],[1,5]]
Output: [[1,4],null,[1,0],null,[1,5],[1,5]]
Explanation: The random pointer of a node can be the node itself.


Example 3:
Input: root = [[1,6],[2,5],[3,4],[4,3],[5,2],[6,1],[7,0]]
Output: [[1,6],[2,5],[3,4],[4,3],[5,2],[6,1],[7,0]]

Example 4:
Input: root = []
Output: []

Example 5:
Input: root = [[1,null],null,[2,null],null,[1,null]]
Output: [[1,null],null,[2,null],null,[1,null]]

Constraints:
The number of nodes in the `tree` is in the range `[0, 1000].`
Each node's value is between `[1, 10^6]`.

### Solution 
 To solve the problem of creating a deep copy of a binary tree with an additional random pointer, we will follow these general steps:

1. **Traverse the tree**: We need to traverse through the nodes of the tree to create the corresponding deep copies.
2. **Maintain a mapping**: To relate original nodes to their copies, we'll use a dictionary. This will help us to easily assign random pointers during the cloning process.
3. **Recreate the node structure**: As we traverse the original tree, we will create new instances of the nodes (using the `NodeCopy` class) and populate the mapping.
4. **Assign random pointers**: After all nodes are copied, we’ll loop through the mapping to set the random pointers for each copy based on the original nodes' random pointers.
5. **Return the root of the copied tree**.

Let's implement this solution in Python following the format used in LeetCode and detail any necessary class structure. 

### Code Implementation



```python
class Node:
    def __init__(self, val=0, left=None, right=None, random=None):
        self.val = val
        self.left = left
        self.right = right
        self.random = random

class NodeCopy:
    def __init__(self, val=0, left=None, right=None, random=None):
        self.val = val
        self.left = left
        self.right = right
        self.random = random

class Solution:
    def copyTreeWithRandomPointer(self, root: Node) -> NodeCopy:
        if not root:
            return None
        
        # Step 1: Create a mapping from original nodes to their copies
        mapping = {}
        
        # Step 2: First pass to copy all nodes
        def clone(node):
            if not node:
                return None
            if node in mapping:
                return mapping[node]
            copy_node = NodeCopy(node.val)
            mapping[node] = copy_node
            copy_node.left = clone(node.left)
            copy_node.right = clone(node.right)
            return copy_node
        
        # Initiate the cloning process
        clone(root)
        
        # Step 3: Second pass to assign the random pointers
        for original, copy in mapping.items():
            copy.random = mapping.get(original.random)  # This will be None if original.random is None
        
        return mapping[root]  # Return the root of the copied tree

```

### Detailed Explanation

1. **Node and NodeCopy Classes**: 
    - The `Node` class represents the original tree nodes, having attributes for value, left and right children, and a random pointer.
    - The `NodeCopy` class is structurally identical to `Node` but is used to hold the copied nodes.

2. **Solution Class**: 
    - The `copyTreeWithRandomPointer` method is responsible for creating the deep copy of the tree.
    - It first checks if the root is `None`, returning `None` immediately if that's the case.

3. **Mapping Creation**: 
    - A dictionary named `mapping` is used to store the relationship between original nodes and their copies.

4. **Tree Traversal and Cloning**:
    - The `clone()` function performs a pre-order traversal of the binary tree to create copies of the nodes. It recursively calls itself for left and right children and saves the copies in the `mapping` dictionary.

5. **Random Pointer Assignment**: 
    - After the nodes are copied, a second pass through the `mapping` dictionary assigns the random pointers. It correctly assigns a copy's random pointer to the respective copy of the original node's random pointer.

6. **Completion**: 
    - Finally, the copied root node (the deep copy of the original root) is returned.

This solution runs in O(N) time complexity, as we visit each node twice (once for copying and once for setting random pointers), where N is the number of nodes in the tree. The space complexity is also O(N) due to the additional storage used in the mapping dictionary.

# 1490. Clone N-ary Tree

### Problem Description 
Given a `root` of an N-ary tree, return a deep copy (clone) of the tree.

Each node in the n-ary tree contains a val (`int`) and a list (`List[Node]`) of its children.

class Node {
    public int val;
    public List<Node> children;
}
Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples).

Follow up: Can your solution work for the graph problem?

Example 1:
Input: root = [1,null,3,2,4,null,5,6]
Output: [1,null,3,2,4,null,5,6]

Example 2:
Input: root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
Output: [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]

Constraints:
The depth of the n-ary tree is less than or equal to `1000`.

The total number of nodes is between `[0, 10^4]`.

### Solution 
 To solve the problem of deep cloning an N-ary tree, we need to create a function that can traverse the original tree and construct a new tree with the same structure and values. Each node in the tree contains an integer value (`val`) and a list of its children.

### Steps to Solve the Problem:
1. **Define the Tree Node Class**: We first need a class to represent the structure of a node in the N-ary tree.
2. **Create a Function for Cloning**: We will implement a recursive function that takes a node as input and returns a new node that is a clone of the original including all its children.
3. **Handle Edge Cases**: Consider the case when the input tree is empty (i.e., the root is `None`).

### Detailed Explanation:
- **Node Class**: We define a Node class that has an integer `val` and a list of `Node` objects as `children`.
  
- **Cloning Function**: 
  - If the input node is `None`, we simply return `None`.
  - For a given node, we create a new node with the same value.
  - We then recursively call the cloning function for each child of the original node and add these cloned children to the new node's list of children.
  
- **Main Function**: The function to be called will initialize the cloning process. 

Now, let's implement this:



```python
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []

class Solution:
    def cloneTree(self, root: 'Node') -> 'Node':
        if not root:
            return None
        
        # Create a new node with the same value as the root
        new_node = Node(root.val)
        
        # Recursively clone all the children and add to the new node's children
        for child in root.children:
            new_node.children.append(self.cloneTree(child))
        
        return new_node

```

### Explanation of the Code:
1. **Node Class**: 
   - `__init__` initializes the node with a value and optionally a list of children.

2. **Solution Class**: 
   - `cloneTree` method is defined, which takes the root node of the N-ary tree as input.
   - It first checks if the `root` is `None`, and if so, returns `None` to handle the empty tree case.
   - A new node (`new_node`) is created with the same value as the `root`.
   - It then iterates through each child of the root, calls `cloneTree` recursively to clone each child and appends the result to `new_node.children`.
   - Finally, it returns the newly created deep copy of the N-ary tree.

### Complexity Analysis:
- **Time Complexity**: O(N), where N is the number of nodes in the tree. Each node is visited exactly once.
- **Space Complexity**: O(H), where H is the height of the tree, due to the recursive call stack. In the worst case (a linear tree), this could be O(N).

This implementation handles the requirements of cloning an N-ary tree, and it is structured to run directly on the LeetCode platform.

# 1519. Number of Nodes in the Sub-Tree With the Same Label

### Problem Description 
Given a tree (i.e. a connected, undirected graph that has no cycles) consisting of `n` nodes numbered from `0` to `n - 1` and exactly `n - 1` `edges`. The root of the tree is the node `0`, and each node of the tree has a label which is a lower-case character given in the string `labels` (i.e. The node with the number `i` has the label `labels[i]`).

The `edges` array is given on the form `edges[i] = [ai, bi]`, which means there is an edge between nodes `ai` and `bi` in the tree.

Return an array of size `n` where `ans[i]` is the number of nodes in the subtree of the `ith` node which have the same label as node `i`.

A subtree of a tree `T` is the tree consisting of a node in `T` and all of its descendant nodes.


Example 1:
Input: n = 7, edges = [[0,1],[0,2],[1,4],[1,5],[2,3],[2,6]], labels = "abaedcd"
Output: [2,1,1,1,1,1,1]
Explanation: Node 0 has label 'a' and its sub-tree has node 2 with label 'a' as well, thus the answer is 2. Notice that any node is part of its sub-tree.

Node 1 has a label 'b'. The sub-tree of node 1 contains nodes 1,4 and 5, as nodes 4 and 5 have different labels than node 1, the answer is just 1 (the node itself).


Example 2:
Input: n = 4, edges = [[0,1],[1,2],[0,3]], labels = "bbbb"
Output: [4,2,1,1]
Explanation: The sub-tree of node 2 contains only node 2, so the answer is 1.

The sub-tree of node 3 contains only node 3, so the answer is 1.

The sub-tree of node 1 contains nodes 1 and 2, both have label 'b', thus the answer is 2.

The sub-tree of node 0 contains nodes 0, 1, 2 and 3, all with label 'b', thus the answer is 4.


Example 3:
Input: n = 5, edges = [[0,1],[0,2],[1,3],[0,4]], labels = "aabab"
Output: [3,2,1,1,1]

Example 4:
Input: n = 6, edges = [[0,1],[0,2],[1,3],[3,4],[4,5]], labels = "cbabaa"
Output: [1,2,1,1,2,1]

Example 5:
Input: n = 7, edges = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6]], labels = "aaabaaa"
Output: [6,5,4,1,3,2,1]

Constraints:
`1 <= n <= 10^5`
`edges.length == n - 1`
`edges[i].length == 2`
`0 <= ai, bi < n`
`ai != bi`
`labels.length == n`
`labels` is consisting of only of lower-case English letters.

### Solution 
 To solve the problem of counting the number of nodes with the same label as each node in the subtree of a tree structure represented by edges and labels, we will use a depth-first search (DFS) approach. The key points to keep in mind are:

1. **Tree Representation**: We will represent the tree using an adjacency list.
2. **DFS Traversal**: We will perform a DFS traversal starting from the root node (node 0). During this traversal, we will maintain a count of the occurrences of each label in the subtree rooted at the current node.
3. **Label Counting**: For each node, we will keep track of how many nodes in its subtree have the same label as itself.

Now let's dive into the solution with a step-by-step explanation.

Here's the Python code following the LeetCode submission format:



```python
class Solution:
    def countSubtrees(self, n: int, edges: List[List[int]], labels: str) -> List[int]:
        from collections import defaultdict
        from collections import Counter
        
        # Create adjacency list for the graph representation of the tree
        tree = defaultdict(list)
        for u, v in edges:
            tree[u].append(v)
            tree[v].append(u)
        
        # Initialize the answer list to store the result
        ans = [0] * n
        
        # Helper function for DFS
        def dfs(node: int, parent: int) -> Counter:
            # Initialize a counter for the current node's label
            count = Counter()
            # Count the current node's label
            count[labels[node]] += 1
            
            # Traverse all the children (connected nodes)
            for neighbor in tree[node]:
                if neighbor == parent: 
                    continue  # Prevent going back to the parent node
                child_count = dfs(neighbor, node)
                # Add the child's counts to the current count
                count += child_count
            
            # The answer for the current node is the count of its label
            ans[node] = count[labels[node]]
            return count
        
        # Start DFS from the root node (0) with no parent (-1)
        dfs(0, -1)
        
        return ans

```

### Detailed Explanation of the Code:

1. **Input Function Signature**: 
   - The method `countSubtrees` takes three parameters: `n` (the number of nodes), `edges` (the list of edges), and `labels` (the string of labels).

2. **Tree Construction**: 
   - We use `defaultdict` from the `collections` module to create an adjacency list called `tree`. This will help us easily traversing the tree structure by maintaining a list of connected nodes.

3. **Result Initialization**: 
   - We initialize an answer list `ans` of size `n` with all zeros. This list will hold the counts of the same label nodes for each node.

4. **DFS Function**: 
   - We define a `dfs` function that takes in the current `node` and its `parent`. 
   - We create a `Counter` called `count` to keep track of label occurrences. Initially, we count the label of the current node.
   - We then iterate through all connected nodes (`neighbors`). If a neighbor is the parent, we skip it to avoid going back up the tree.
   - We recursively call `dfs` for each unvisited child, updating the current `count` with the labels received from subtrees.

5. **Store the Result**: 
   - For each node, after processing its children, we set `ans[node]` to the count of its label (`count[labels[node]]`).

6. **DFS Start**: 
   - We start our DFS from the root node (`0`) with no parent (indicated by `-1`).

7. **Return the Result**: 
   - Finally, we return the `ans` list which contains the count of same-label nodes in each subtree.

### Complexity Analysis:
- **Time Complexity**: O(n), since we visit each node and each edge exactly once.
- **Space Complexity**: O(n) for the adjacency list and the call stack during DFS.

This approach efficiently counts the required nodes in a single DFS traversal, making it suitable for the problem's constraints.

# 1602. Find Nearest Right Node in Binary Tree

### Problem Description 
Given the `root` of a binary tree and a node `u` in the tree, return the nearest node on the same level that is to the right of `u`, or return `null` if `u` is the rightmost node in its level.


Example 1:
Input: root = [1,2,3,null,4,5,6], u = 4
Output: 5
Explanation: The nearest node on the same level to the right of node 4 is node 5.


Example 2:
Input: root = [3,null,4,2], u = 2
Output: null
Explanation: There are no nodes to the right of 2.


Example 3:
Input: root = [1], u = 1
Output: null

Example 4:
Input: root = [3,4,2,null,null,null,1], u = 4
Output: 2

Constraints:
The number of nodes in the tree is in the range `[1, 105]`.

`1 <= Node.val <= 105`
All values in the tree are distinct.

`u` is a node in the binary tree rooted at `root`.

### Solution 
 To solve the problem of finding the nearest node on the same level to the right of a given node `u` in a binary tree, we can follow this approach:

### Approach:

1. **Level Order Traversal**: We will use a breadth-first search (BFS) method, which is suitable for level order traversal in trees. We can utilize a queue to keep track of the nodes at each level.

2. **Identify the Right Neighbor**: As we traverse each level, we will maintain a list of nodes for that level. When we find node `u`, we will check its position in the list and return the immediate next node in that list, if it exists. If node `u` is the last in the level, we return `null`.

3. **Edge Cases**: We handle edge cases where the tree may have only one node or where node `u` is the only node at its level.

### Implementation:

Here is the Python code that implements the above approach following the LeetCode submission style:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def find_nearest_right_node(self, root: TreeNode, u: TreeNode) -> TreeNode:
        from collections import deque
        
        # Use a queue to facilitate level order traversal
        queue = deque([root])
        
        # Perform a level order traversal
        while queue:
            level_size = len(queue)
            found = False
            nodes_in_level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                nodes_in_level.append(node)
                
                # Determine if this is the target node
                if node == u:
                    found = True
                
                # Add children to the queue for the next level
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                    
            # If we found u in this level
            if found:
                # Check for the right neighbor
                index = nodes_in_level.index(u)  # Find the index of u in this level
                if index + 1 < len(nodes_in_level):  # If there's a next node in the level
                    return nodes_in_level[index + 1]
                else:
                    return None  # No node to the right
        
        return None  # In case u is not found, which should not happen

```

### Explanation:

- **TreeNode Class**: This is a simple definition of a binary tree node with a value and left and right children.

- **Solution Class**: We encapsulate our logic within a Solution class that contains the `find_nearest_right_node` method which takes the root of the tree and the target node `u`.

- **Queue and Level Traversal**:
  - We use a deque for efficient popping from the left to traverse the tree level by level.
  - For each level, we keep track of all the nodes and check if `u` is present among them.

- **Finding the Next Node**:
  - On finding `u`, we locate its index within the list of nodes at the current level and check if there exists a node to the right of it.

This code can be directly run on LeetCode as it adheres to the prompt's structure and handles both normal and edge cases effectively.

# 1625. Lexicographically Smallest String After Applying Operations

### Problem Description 
You are given a string `s` of even length consisting of digits from `0` to `9`, and two integers `a` and `b`.

You can apply either of the following two operations any number of times and in any order on `s`:
Add `a` to all odd indices of `s` (0-indexed). Digits post `9` are cycled back to `0`. For example, if `s = "3456"` and `a = 5`, `s` becomes `"3951"`.

Rotate `s` to the right by `b` positions. For example, if `s = "3456"` and `b = 1`, `s` becomes `"6345"`.

Return the lexicographically smallest string you can obtain by applying the above operations any number of times on `s`.

A string `a` is lexicographically smaller than a string `b` (of the same length) if in the first position where `a` and `b` differ, string `a` has a letter that appears earlier in the alphabet than the corresponding letter in `b`. For example, `"0158"` is lexicographically smaller than `"0190"` because the first position they differ is at the third letter, and `'5'` comes before `'9'`.


Example 1:
Input: s = "5525", a = 9, b = 2
Output: "2050"
Explanation: We can apply the following operations:
Start:  "5525"
Rotate: "2555"
Add:    "2454"
Add:    "2353"
Rotate: "5323"
Add:    "5222"
​​​​​​​Add:    "5121"
​​​​​​​Rotate: "2151"
​​​​​​​Add:    "2050"​​​​​​​​​​​​
There is no way to obtain a string that is lexicographically smaller then "2050".


Example 2:
Input: s = "74", a = 5, b = 1
Output: "24"
Explanation: We can apply the following operations:
Start:  "74"
Rotate: "47"
​​​​​​​Add:    "42"
​​​​​​​Rotate: "24"​​​​​​​​​​​​
There is no way to obtain a string that is lexicographically smaller then "24".


Example 3:
Input: s = "0011", a = 4, b = 2
Output: "0011"
Explanation: There are no sequence of operations that will give us a lexicographically smaller string than "0011".


Example 4:
Input: s = "43987654", a = 7, b = 3
Output: "00553311"

Constraints:
`2 <= s.length <= 100`
`s.length` is even.

`s` consists of digits from `0` to `9` only.

`1 <= a <= 9`
`1 <= b <= s.length - 1`

### Solution 
 To solve the problem of finding the lexicographically smallest string after applying a series of operations on the input string `s`, we can follow a systematic approach aimed at considering both operations: modifying the digits at odd indices and rotating the string.

### Approach

1. **Understand Operations**: 
    - The first operation adds a given integer `a` to all digits at odd indices, cycling back to `0` after `9`.
    - The second operation rotates the string to the right by `b` positions.

2. **Rotation**: 
   - Given that we can rotate the string to get different configurations, we need to iterate through all possible rotations. For a string of length `n`, there are `n` unique rotations (i.e., rotating the string `b` from `0` to `n-1`).

3. **Digit Modification**: 
   - For each rotation, we will apply the addition operation as many times as needed. Since adding `10 (a=9)` will cycle through the same results every `10` applications, we only need to consider the results of applying the add operation `0` to `9` times.

4. **Generate Candidates**: 
   - After each rotation and each possible application of the add operation, we collect the resulting strings and keep track of the lexicographically smallest string.

5. **Implementation**: 
    - We can utilize nested loops: one for iterating through all possible rotations and another for applying the addition on odd indices multiple times.
  
### Implementation

Here is the code implementation following the approach described above:



```python
class Solution:
    def findLexSmallestString(self, s: str, a: int, b: int) -> str:
        n = len(s)
        min_string = s
        
        # Function to add 'a' to odd indexed characters
        def add_a_to_odds(s, a):
            new_s = list(s)
            for i in range(1, n, 2):  # Odd indices
                new_s[i] = str((int(new_s[i]) + a) % 10)
            return ''.join(new_s)

        # Generate all rotations
        for i in range(n):
            # Rotate string
            rotated_s = s[-i:] + s[:-i]
            current_s = rotated_s
            
            # Apply adding a to odd indices 0 to 9 times
            for j in range(10):
                current_s = add_a_to_odds(current_s, a * j)
                min_string = min(min_string, current_s)

        return min_string

```

### Explanation of the Code:

1. **Class and Function Declaration**: We define a class `Solution` with a method `findLexSmallestString`, receiving parameters: the string `s`, integer `a`, and `b`.

2. **Initial Setup**: 
   - Length of `s` is calculated.
   - We initialize `min_string` with the original string, which will hold the lexicographically smallest result.

3. **Adding 'a' Function**:
   - `add_a_to_odds`: This helper function adds `a` to all digits at odd indices for a given string, cycling back if it exceeds `9`.

4. **Rotations Loop**:
   - A loop iterates `n` times to generate all rotations of the string.

5. **Inner Loop for Add Operations**:
   - For each rotation, the inner loop checks the effect of adding `a` (0 through 9) to the odd indices and updates `min_string` if the resulting string is smaller.

6. **Return the Result**: Finally, the minimum string is returned after considering all operations.

This approach ensures that we efficiently explore all potential combinations of rotations and modifications while maintaining the constraints given in the problem statement.

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
 To solve the problem of finding the shortest path to any food cell in a character matrix, we can utilize the Breadth-First Search (BFS) algorithm. This approach is well-suited for unweighted grids where we want to find the shortest path, as it explores all neighbors at the present depth level before moving on to nodes at the next depth level.

Here’s a step-by-step breakdown of the solution:

### Steps

1. **Identify the Starting Point**: Locate the position of the `'*'` character, which represents our starting location.

2. **Initialize BFS**: We'll use a queue to hold our current position and the step count. We'll also maintain a set to track visited cells to avoid re-processing.

3. **Define Directions**: Since we can move in four directions (up, down, left, right), we'll define these as coordinate changes.

4. **Perform BFS**:
   - Dequeue the current position and check if it's a food cell (`'#'`). If it is, return the step count.
   - If it's not a food cell, enqueue any valid neighboring cells (i.e., within bounds, not obstacles, and not visited).
   - Continue this until the queue is empty.

5. **Handle No Path**: If we exhaust the BFS and find no food cells, return `-1`.

### Implementation

Here's the Python code that follows the aforementioned steps:



```python
from collections import deque
from typing import List

class Solution:
    def getFood(self, grid: List[List[str]]) -> int:
        # Locate the starting position '*'
        start_row = start_col = -1
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '*':
                    start_row, start_col = i, j
                    break
            if start_row != -1:
                break
        
        # Directions for moving up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # BFS implementation
        queue = deque([(start_row, start_col, 0)])  # (row, column, steps)
        visited = set((start_row, start_col))
        
        while queue:
            row, col, steps = queue.popleft()
            
            # Check if we've reached a food cell
            if grid[row][col] == '#':
                return steps
            
            # Explore neighboring cells
            for d in directions:
                new_row, new_col = row + d[0], col + d[1]
                
                # Check grid boundaries and validity
                if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]) and (new_row, new_col) not in visited:
                    if grid[new_row][new_col] != 'X':  # Can move if not an obstacle
                        visited.add((new_row, new_col))
                        queue.append((new_row, new_col, steps + 1))
        
        return -1  # No path to food

```

### Explanation of the Code

1. **Finding the Start**: We loop through the grid to find the location of `'*'`, which sets our initial point of departure.

2. **Queue Initialization**: We initialize the BFS queue with the starting point and the step count (starting from 0). We also mark this position as visited.

3. **BFS Process**:
   - We dequeue a position and check if it's a food cell. If it is, we return the step count.
   - For each of the four possible movements, we calculate the new position and check if it's valid (in bounds, not visited, not an obstacle).
   - Valid positions are added to the queue for further exploration.

4. **Termination**: If the queue is emptied without finding a food cell, we return `-1`.

This code handles all edge cases laid out in the problem and effectively determines the shortest path to food when it exists. If you plug this code into LeetCode's platform, it should run as intended.

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
 To solve the problem of determining the distance between two nodes with values `p` and `q` in a binary tree, we can break the problem down into a few key steps:

1. **Find the Lowest Common Ancestor (LCA)** of the nodes with values `p` and `q`. The LCA is the deepest node that is an ancestor to both `p` and `q`. This step is crucial because it allows us to compute the distance between `p` and `q` as the sum of the distances from `p` to the LCA and from `q` to the LCA.

2. **Calculate the Distance** from a given node to its ancestor. We can do this recursively by counting how many edges we traverse until we reach the target node.

Here is a detailed implementation of the above logic in Python, formatted to run directly on LeetCode.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def findDistance(self, root: TreeNode, p: int, q: int) -> int:
        # Find the LCA of p and q
        lca = self.findLCA(root, p, q)
        if not lca:
            return -1  # If there is no LCA, this means at least one of the nodes is not present
        
        # Calculate distance from LCA to p
        distance_to_p = self.distanceFromLCA(lca, p, 0)
        # Calculate distance from LCA to q
        distance_to_q = self.distanceFromLCA(lca, q, 0)
        
        return distance_to_p + distance_to_q
    
    def findLCA(self, root: TreeNode, p: int, q: int) -> TreeNode:
        # Base case: If the root is None, return None
        if not root:
            return None
        
        # If root's value is p or q, then it is the LCA
        if root.val == p or root.val == q:
            return root
        
        # Recur for left and right subtrees
        left_lca = self.findLCA(root.left, p, q)
        right_lca = self.findLCA(root.right, p, q)
        
        # If both left_lca and right_lca are not None,
        # then one key is present in one subtree and the other key is present in another
        if left_lca and right_lca:
            return root
        
        # Otherwise, return whichever side is not null
        return left_lca if left_lca is not None else right_lca
    
    def distanceFromLCA(self, root: TreeNode, node_val: int, depth: int) -> int:
        # Base case: If we reach a None node, return -1 (not found)
        if not root:
            return -1
        
        # If we found the node, return the current depth
        if root.val == node_val:
            return depth
        
        # Recur for left and right children
        left_distance = self.distanceFromLCA(root.left, node_val, depth + 1)
        right_distance = self.distanceFromLCA(root.right, node_val, depth + 1)
        
        # If found in either subtree, return the distance
        if left_distance != -1:
            return left_distance
        if right_distance != -1:
            return right_distance
        
        # Not found in this path
        return -1

```

### Explanation of the Code:

1. **TreeNode Class**: This is a basic definition for nodes in the binary tree, initialized with its value and pointers to its left and right children.

2. **findDistance Method**: This is the main function that:
   - Finds the LCA of `p` and `q` nodes.
   - Computes the distance from the LCA to two nodes `p` and `q`.
   - Returns the total distance as the sum of both distances.

3. **findLCA Method**: This recursive method:
   - Checks if the current node (`root`) is either `p` or `q`.
   - Recursively checks the left and right subtrees for `p` and `q`.
   - Returns the LCA based on whether both subtrees return ancestors.

4. **distanceFromLCA Method**: This function recursively searches for a node starting from the LCA:
   - Keeps track of the current depth.
   - Returns the depth when the target node is found.
   - Returns -1 if the node is not found in that path.

This solution is efficient given the constraints and should perform well for trees up to 10,000 nodes.

# 1765. Map of Highest Peak

### Problem Description 
You are given an integer matrix `isWater` of size `m x n` that represents a map of land and water cells.

If `isWater[i][j] == 0`, cell `(i, j)` is a land cell.

If `isWater[i][j] == 1`, cell `(i, j)` is a water cell.

You must assign each cell a height in a way that follows these rules:
The height of each cell must be non-negative.

If the cell is a water cell, its height must be `0`.

Any two adjacent cells must have an absolute height difference of at most `1`. A cell is adjacent to another cell if the former is directly north, east, south, or west of the latter (i.e., their sides are touching).

Find an assignment of heights such that the maximum height in the matrix is maximized.

Return an integer matrix `height` of size `m x n` where `height[i][j]` is cell `(i, j)`'s height. If there are multiple solutions, return any of them.


Example 1:
Input: isWater = [[0,1],[0,0]]
Output: [[1,0],[2,1]]
Explanation: The image shows the assigned heights of each cell.

The blue cell is the water cell, and the green cells are the land cells.


Example 2:
Input: isWater = [[0,0,1],[1,0,0],[0,0,0]]
Output: [[1,1,0],[0,1,1],[1,2,2]]
Explanation: A height of 2 is the maximum possible height of any assignment.

Any height assignment that has a maximum height of 2 while still meeting the rules will also be accepted.


Constraints:
`m == isWater.length`
`n == isWater[i].length`
`1 <= m, n <= 1000`
`isWater[i][j]` is `0` or `1`.

There is at least one water cell.

### Solution 
 To solve the problem of assigning heights to the land cells based on the given rules while maximizing the height in the matrix, we can leverage the breadth-first search (BFS) algorithm. The main goal is to ensure that all land cells (denoted by `0` in the `isWater` matrix) are assigned heights such that:
1. Water cells (denoted by `1`) have a height of `0`.
2. Heights of adjacent cells differ by at most `1`.

### Steps to Solve the Problem:

1. **Initialize the Height Matrix**: Create a height matrix of the same size as `isWater` and initialize all cells. Set the height of all water cells to `0`.

2. **Breadth-First Search (BFS) Setup**: Use a queue to perform a BFS starting from all water cells. For every water cell found, the neighboring land cells can potentially receive a height of `1`. By pushing all water cells into a queue at the start, we can process them level by level.

3. **Process Adjacent Cells**: For each water cell, check its four possible adjacent cells (up, down, left, right):
   - If the adjacent cell is land (i.e., `height value is not set yet`), assign it a height of `1 + height of current water cell` and add this cell to the BFS queue for further processing.

4. **Continue BFS**: Repeat the process for each cell in the queue until all possible cells have been processed.

5. **Return Result**: After BFS completes, the height matrix will have the required heights assigned optimally.

Let's implement the above plan in Python:



```python
from collections import deque
from typing import List

class Solution:
    def highestPeak(self, isWater: List[List[int]]) -> List[List[int]]:
        m, n = len(isWater), len(isWater[0])
        height = [[-1] * n for _ in range(m)]  # Initialize height matrix with -1
        queue = deque()
        
        # Step 1: Set height of water cells to 0 and enqueue them
        for i in range(m):
            for j in range(n):
                if isWater[i][j] == 1:
                    height[i][j] = 0
                    queue.append((i, j))  # Add water cells to queue
        
        # Directions for adjacency (up, down, left, right)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        # Step 2: Run BFS starting from all water cells
        while queue:
            x, y = queue.popleft()  # Current position
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy  # Neighbor positions
                
                # Step 3: Check if neighbor is within bounds and not assigned a height yet
                if 0 <= nx < m and 0 <= ny < n and height[nx][ny] == -1:
                    height[nx][ny] = height[x][y] + 1  # Assign height
                    queue.append((nx, ny))  # Enqueue the neighbor
        
        return height  # return the assigned height matrix

```

### Explanation of the Code:
- We use `deque` to efficiently pop from the front of the queue.
- The `height` matrix is initialized with `-1` to signify unassigned heights.
- BFS is triggered from each water cell, and as we assign heights to adjacent land cells, we ensure that their height is exactly `1` greater than the height of the current water cell, thus maintaining the rules specified.

### Complexity:
- The time complexity is O(m * n) since we may traverse each cell once in the worst case.
- The space complexity is O(m * n) for the height matrix and for the queue storing the cells during BFS.

This approach ensures that we assign heights efficiently and optimally while adhering to the specified constraints.

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
 To solve the problem of finding the closest coprime ancestor in a tree structure, we will utilize Depth-First Search (DFS) to explore the tree while keeping track of the ancestors we encounter on the path from the root to each node. The main idea is to check the coprimeness of a node's value with each ancestor's value as we traverse down.

### Step-by-step Explanation

1. **Input Representation**: We'll take `nums` as an array where `nums[i]` is the value of the node `i`, and `edges` as a list of edges defining the tree connections.

2. **Graph Construction**: Instead of using a list of edges, we will convert it into an adjacency list representation for easier traversal.

3. **Coprimeness Check**: We'll use the `gcd` function from the `math` library to determine if the values of two nodes are coprime.

4. **DFS Traversal**: We will perform a DFS starting from the root node (node `0`). As we visit each node:
   - Check its value against its ancestors' values recorded in the path.
   - Update the result array based on the first coprime ancestor found.
   - Record the node's value in the ancestor path so that we can refer back as we traverse to child nodes.
   - Backtrack by removing the current node's value from the ancestor path once done with its children.

5. **Return Results**: Finally, accumulate all results and return them in the required format.

Here's the complete solution in Python:



```python
from collections import defaultdict
from math import gcd
from typing import List

class Solution:
    def getCoprimeAncestors(self, nums: List[int], edges: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # Adjacency list representation of the tree
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        # Result array
        ans = [-1] * n
        
        # DFS function to explore the tree
        def dfs(node: int, parent: int, ancestors: List[int]):
            current_value = nums[node]
            found = -1
            
            # Check against ancestors for coprime
            for ancestor in ancestors:
                if gcd(current_value, nums[ancestor]) == 1:
                    found = ancestor
            
            ans[node] = found
            
            # Add current node to the ancestors path
            ancestors.append(node)
            
            # Explore children nodes
            for neighbor in graph[node]:
                if neighbor != parent:  # Prevent going back to the parent
                    dfs(neighbor, node, ancestors)
            
            # Backtrack and remove the node from ancestors list
            ancestors.pop()
        
        # Start DFS from the root node (node 0)
        dfs(0, -1, [])
        
        return ans

```

### Explanation of the Code:
- **Graph Construction**: We create a `defaultdict` of lists to represent the tree.
- **DFS Function**: The `dfs` function takes the current node, its parent (to prevent traversing back), and the list of ancestors.
- **Coprime Check**: For each ancestor, if the `gcd` with the current node's value is 1, it means they are coprime.
- **Backtracking**: After visiting all children of a node, we remove it from the list of ancestors to ensure that we do not mistakenly pass it to sibling nodes.

### Complexity:
- **Time Complexity**: O(n) due to DFS traversing each node once and checking against ancestors.
- **Space Complexity**: O(n) for storing the graph and the ancestors list during recursive DFS calls.

This solution is efficient and adheres to the problem constraints, making it suitable for direct execution on LeetCode or similar platforms.

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
 Certainly! To tackle the problem, we will employ a breadth-first search (BFS) algorithm. This is particularly suitable for finding the shortest path in an unweighted grid, such as the one we are dealing with here.

### Key Steps of the Approach:

1. **Initialization**:
   - We will start from the robot's initial position and use a queue to keep track of cells to explore (this queue helps in implementing BFS).
   - We will keep track of visited cells to avoid cycles and ensure we do not traverse blocked cells.

2. **BFS Traversal**:
   - For each position dequeued from the queue, we will check all four possible movements (up, down, left, right).
   - If the movement is valid (checked using the `canMove` method) and the target cell is reached (checked using the `isTarget` method), we will return the distance.

3. **Boundary Conditions**:
   - If all possible paths have been checked and the target has not been reached, we will return `-1`, indicating that there is no valid path.

### Implementation:

Here is how we can implement this in Python according to the LeetCode format:



```python
class Solution:
    def minimumMoves(self, master: 'GridMaster') -> int:
        from collections import deque

        # Directions [Up, Down, Left, Right]
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # (dy, dx)
        direction_chars = ['U', 'D', 'L', 'R']  # corresponding direction characters
        
        # Queue for BFS. Store tuples of (y, x, distance)
        queue = deque([(0, 0, 0)])
        visited = set()  # Set to keep track of visited cells
        visited.add((0, 0))

        while queue:
            y, x, dist = queue.popleft()
            
            # Check if we are at the target position
            if master.isTarget():
                return dist
            
            # Explore all 4 directions
            for (dy, dx), char in zip(directions, direction_chars):
                new_y, new_x = y + dy, x + dx
                
                if (new_y, new_x) not in visited and master.canMove(char):
                    master.move(char)  # Move in that direction
                    visited.add((new_y, new_x))  # Mark the new position as visited
                    queue.append((new_y, new_x, dist + 1))  # Add new position to queue with incremented distance
                    
                    # Reverse the move to return to the original location
                    reverse_char = direction_chars[(direction_chars.index(char) + 2) % 4]  # Opposite move
                    master.move(reverse_char)  # Move back to the current position
        
        return -1  # If we exit the loop, the target was not reachable

```

### Explanation of the Code:

1. **Data Structures**: 
   - A deque is used for efficient popping from the front during BFS.
   - A set keeps track of visited positions to prevent loops.

2. **Movement Logic**:
   - For each valid movement direction (checked using `canMove`), if we can move, we do so and then check if we hit the target. After checking, we immediately reverse the move so we can explore other paths.

3. **Distance Calculation**:
   - We keep track of the distance from the starting cell to the current cell, which gets incremented with each valid move.

4. **End Condition**: 
   - If the queue is exhausted without finding the target, we return `-1`.

This structured approach allows efficient navigation through the grid while adhering to the request for clarity and adherence to LeetCode's execution environment.

# 1824. Minimum Sideway Jumps

### Problem Description 
There is a 3 lane road of length `n` that consists of `n + 1` points labeled from `0` to `n`. A frog starts at point `0` in the second lane and wants to jump to point `n`. However, there could be obstacles along the way.

You are given an array `obstacles` of length `n + 1` where each `obstacles[i]` (ranging from 0 to 3) describes an obstacle on the lane `obstacles[i]` at point `i`. If `obstacles[i] == 0`, there are no obstacles at point `i`. There will be at most one obstacle in the 3 lanes at each point.

For example, if `obstacles[2] == 1`, then there is an obstacle on lane 1 at point 2.

The frog can only travel from point `i` to point `i + 1` on the same lane if there is not an obstacle on the lane at point `i + 1`. To avoid obstacles, the frog can also perform a side jump to jump to another lane (even if they are not adjacent) at the same point if there is no obstacle on the new lane.

For example, the frog can jump from lane 3 at point 3 to lane 1 at point 3.

Return the minimum number of side jumps the frog needs to reach any lane at point n starting from lane `2` at point 0.

Note: There will be no obstacles on points `0` and `n`.


Example 1:
Input: obstacles = [0,1,2,3,0]
Output: 2 
Explanation: The optimal solution is shown by the arrows above. There are 2 side jumps (red arrows).

Note that the frog can jump over obstacles only when making side jumps (as shown at point 2).


Example 2:
Input: obstacles = [0,1,1,3,3,0]
Output: 0
Explanation: There are no obstacles on lane 2. No side jumps are required.


Example 3:
Input: obstacles = [0,2,1,0,3,0]
Output: 2
Explanation: The optimal solution is shown by the arrows above. There are 2 side jumps.


Constraints:
`obstacles.length == n + 1`
`1 <= n <= 5 * 105`
`0 <= obstacles[i] <= 3`
`obstacles[0] == obstacles[n] == 0`

### Solution 
 Certainly! To solve this problem efficiently, we can use a dynamic programming approach. The approach involves checking the state of the frog's position at each point and how many side jumps are required to move to the next point, either on the same lane or by jumping to another lane.

Here's the code to solve the problem in Python:



```python
class Solution:
    def minSideJumps(self, obstacles: List[int]) -> int:
        n = len(obstacles) - 1  # Length of the road is n
        # dp[i] represents the minimum side jumps required to reach point i
        dp = [float('inf')] * 3  # We will track 3 lanes, indexed as 0, 1, 2 which correspond to lanes 1, 2, 3 respectively
        
        dp[1] = 0  # Starting at lane 2 (index 1) with no jumps

        for i in range(n):
            for lane in range(3):
                if obstacles[i] - 1 != lane:  # If there's no obstacle on current lane
                    dp[lane] = min(dp[lane], dp[lane])  # No change in the current lane
                else:
                    dp[lane] = float('inf')  # If there's an obstacle, set to infinity
            
            # Update dp for the next position based on the current obstacles
            temp = dp[:]  # Backup current dp before moving to next position
            
            for lane in range(3):
                if obstacles[i + 1] - 1 != lane:  # If there's no obstacle in the next position lane
                    temp[lane] = min(temp[lane], dp[lane])  # Keep minimum jumps from the current lane

                # Check jump to other lanes
                for jump_to in range(3):
                    if jump_to != lane and obstacles[i + 1] - 1 != jump_to:  # Can jump to other lane safely
                        temp[jump_to] = min(temp[jump_to], dp[lane] + 1)  # 1 more jump needed
            
            dp = temp  # Move to the next jump status
        
        return min(dp)  # Return the minimum jumps needed at last point `n`

```

### Explanation:

1. **Initialization**: 
   - We create a list `dp` of size 3 to represent the minimum number of side jumps required to reach point `i` for each of the three lanes. Initially, we set the values to infinity (`float('inf')`), except for `dp[1]`, since the frog starts on lane 2 (index 1) at point 0. Here, `dp[1]` is initialized to 0 because the frog starts there with no side jumps.
   
2. **Loop through the road**: 
   - We iterate over each point from `0` to `n-1`. For each point, we loop through the three lanes (0 for lane 1, 1 for lane 2, and 2 for lane 3). 
   - If there is no obstacle on the current lane at point `i`, we maintain the current side jump count. If there is an obstacle on the current lane, we set its value in `dp` to `float('inf')`.

3. **Store temporary results**: 
   - We create a backup of the current `dp` array using `temp` so we can update our jump counts for the next point.

4. **Calculate side jumps for next point**: 
   - For each lane, we check if there's no obstacle at the next point (`i + 1`). If so, we may move forward without any additional jumps.
   - We also check for possible side jumps to the other lanes that do not have obstacles. For each viable jump to another lane, we increment the number of jumps by one and keep track of the minimum.

5. **Final result**: 
   - After looping through all points, we take the minimum of the last `dp` array, which indicates the minimum number of side jumps needed to reach the last point `n` from lane `2`.

This code is efficient, running in O(n) time complexity and it uses O(1) space for tracking lanes, making it suitable for the problem's constraints.

