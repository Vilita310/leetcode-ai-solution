# 199. Binary Tree Right Side View

### Problem Description 
Given the `root` of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.


Example 1:
Input: root = [1,2,3,null,5,null,4]
Output: [1,3,4]

Example 2:
Input: root = [1,null,3]
Output: [1,3]

Example 3:
Input: root = []
Output: []

Constraints:
The number of nodes in the tree is in the range `[0, 100]`.

`-100 <= Node.val <= 100`

### Solution 
 To solve the problem of returning the values of the nodes visible from the right side of a binary tree, we can utilize a breadth-first search (BFS) approach, specifically using a queue. This method allows us to traverse each level of the tree, while always keeping track of the last node visited at each level. The last node of each level will be visible when viewed from the right side.

Here’s how the algorithm works step-by-step:

1. **Input Validation**: First, check if the `root` is `None`. If it is, return an empty list, since there are no nodes to view.

2. **Initialization**: Use a queue (from the `collections` module) to facilitate level-order traversal. Initialize the queue with the `root` node and prepare an empty list to store the values of the visible nodes.

3. **Level-order Traversal**: We will process nodes level by level. For each level, we'll:
   - Record the number of nodes at the current level.
   - Iterate through each node in this level:
     - Dequeue the node and store its value.
     - Enqueue its right child first (if it exists), followed by its left child (if it exists).
   - After processing all nodes at the current level, the last value recorded will be the rightmost node of that level, which we append to our result list.

4. **Return the Result**: Finally, once the queue is empty, return the list containing the values of the nodes that are visible from the right side of the tree.

Here’s the implementation in Python that follows the LeetCode requirements:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from collections import deque

class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:
            return []

        result = []
        queue = deque([root])  # Initialize the queue with the root node

        while queue:
            level_length = len(queue)  # Number of nodes at the current level
            for i in range(level_length):
                node = queue.popleft()  # Dequeue the front node

                # If it's the last node in the current level, add it to result
                if i == level_length - 1:
                    result.append(node.val)

                # Enqueue the right child first, then the left child
                if node.right:
                    queue.append(node.right)
                if node.left:
                    queue.append(node.left)

        return result

```

### Explanation of the Code:
- **TreeNode Class**: This is the definition of the binary tree node, with attributes for value (`val`), left child (`left`), and right child (`right`).
- **Solution Class**: This contains the method `rightSideView` which implements the BFS approach.
  
  1. **Initialization**: We check if `root` is `None` and return an empty list if so. The `result` list is initialized to store the visible nodes, and we start our queue with the `root` node.

  2. **Processing Each Level**: We enter a while loop that continues as long as there are nodes in the queue. Inside this loop:
     - The `level_length` variable holds the count of nodes currently in the queue.
     - A for loop processes each node at the current level. We keep popping nodes from the front of the queue.
     - If the node is the last one in the current level (`i == level_length - 1`), its value is appended to the `result` list.
     - Nodes are enqueued with the right child first (if it exists), followed by the left child (if it exists). This ensures that when the next level is processed, the rightmost nodes are encountered first.

3. **Return Statement**: Finally, the method returns the `result` list containing the values of the nodes visible from the right side of the tree.

This implementation is efficient with a time complexity of O(N), where N is the number of nodes in the tree, and it uses O(W) additional space, where W is the maximum width of the tree, due to the queue.

# 346. Moving Average from Data Stream

### Problem Description 
Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

Implement the `MovingAverage` class:
`MovingAverage(int size)` Initializes the object with the size of the window `size`.

`double next(int val)` Returns the moving average of the last `size` values of the stream.


Example 1:
Input
["MovingAverage", "next", "next", "next", "next"]
[[3], [1], [10], [3], [5]]
Output
[null, 1.0, 5.5, 4.66667, 6.0]
Explanation
MovingAverage movingAverage = new MovingAverage(3);
movingAverage.next(1); // return 1.0 = 1 / 1
movingAverage.next(10); // return 5.5 = (1 + 10) / 2
movingAverage.next(3); // return 4.66667 = (1 + 10 + 3) / 3
movingAverage.next(5); // return 6.0 = (10 + 3 + 5) / 3

Constraints:
`1 <= size <= 1000`
`-105 <= val <= 105`
At most `104` calls will be made to `next`.

### Solution 
 To solve the problem of calculating the moving average for a stream of integers with a specified window size, we can implement a class called `MovingAverage`. The key idea is to maintain a fixed-size queue (for the latest values) and a sum of the values currently in the window. This will allow for efficient calculation of the moving average.

Here is the step-by-step breakdown:

1. **Initialization**: The `MovingAverage` class will be initialized with a size for the sliding window. It will store the maximum size, a collection for the last values (we can use a list), and an integer to keep track of the sum of the current window.

2. **Adding New Values**: When a new integer is added using the `next` method:
   - If the length of stored values is equal to the specified window size, we need to remove the oldest value to make room for the new one. This involves subtracting the oldest value from the sum.
   - Next, we add the new value to both the list of values and the sum.
   - Finally, compute the moving average by dividing the sum by the number of values stored (which will either be the window size or less if not yet filled).

3. **Calculating the Moving Average**: For the moving average, we simply return the sum divided by the count of numbers in the current window.

The code implementation is as follows:



```python
class MovingAverage:
    def __init__(self, size: int):
        """
        Initialize the object with the size of the window 'size'.
        """
        self.size = size          # Store the size of the sliding window
        self.values = []         # List to store the values within the window
        self.current_sum = 0.0   # Running sum of the current window values

    def next(self, val: int) -> float:
        """
        Returns the moving average of the last 'size' values of the stream.
        """
        if len(self.values) == self.size:
            # Remove the oldest value from the sum and the list if we are at capacity
            oldest_value = self.values.pop(0)
            self.current_sum -= oldest_value

        # Add the new value to the current window
        self.values.append(val)
        self.current_sum += val
        
        # Calculate and return the moving average
        return self.current_sum / len(self.values)

# Example Usage:
# movingAverage = MovingAverage(3)
# print(movingAverage.next(1))  # Output: 1.0
# print(movingAverage.next(10)) # Output: 5.5
# print(movingAverage.next(3))  # Output: 4.66667
# print(movingAverage.next(5))  # Output: 6.0

```

### Explanation of the Example:
1. **Initialization**:
   - The `MovingAverage` is created with a window size of 3. The initial state is an empty values list and a sum of 0.

2. **Calling `next`**:
   - `next(1)`: Adds `1` to the window -> [1]. The average is `1.0`.
   - `next(10)`: Adds `10` -> [1, 10]. The average is `(1 + 10) / 2` which equals `5.5`.
   - `next(3)`: Adds `3` -> [1, 10, 3]. The average is `(1 + 10 + 3) / 3` which gives `4.66667`.
   - `next(5)`: Adds `5`. The window is now full. It removes `1` (the oldest value) and adds `5`, resulting in [10, 3, 5]. The average now is `(10 + 3 + 5) / 3` which equals `6.0`.

### Additional Considerations:
- This solution efficiently manages the moving average by ensuring that each `next` call performs an average of constant time operations, leading to a time complexity of O(1) per call.
- Space complexity is O(n) to maintain the window of values where n is the size of the sliding window, which is manageable given the constraints.

# 353. Design Snake Game

### Problem Description 
Design a Snake game that is played on a device with screen size `height x width`. Play the game online if you are not familiar with the game.

The snake is initially positioned at the top left corner `(0, 0)` with a length of `1` unit.

You are given an array `food` where `food[i] = (ri, ci)` is the row and column position of a piece of food that the snake can eat. When a snake eats a piece of food, its length and the game's score both increase by `1`.

Each piece of food appears one by one on the screen, meaning the second piece of food will not appear until the snake eats the first piece of food.

When a piece of food appears on the screen, it is guaranteed that it will not appear on a block occupied by the snake.

The game is over if the snake goes out of bounds (hits a wall) or if its head occupies a space that its body occupies after moving (i.e. a snake of length 4 cannot run into itself).

Implement the `SnakeGame` class:
`SnakeGame(int width, int height, int[][] food)` Initializes the object with a screen of size `height x width` and the positions of the `food`.

`int move(String direction)` Returns the score of the game after applying one `direction` move by the snake. If the game is over, return `-1`.


Example 1:
Input
["SnakeGame", "move", "move", "move", "move", "move", "move"]
[[3, 2, [[1, 2], [0, 1]]], ["R"], ["D"], ["R"], ["U"], ["L"], ["U"]]
Output
[null, 0, 0, 1, 1, 2, -1]
Explanation
SnakeGame snakeGame = new SnakeGame(3, 2, [[1, 2], [0, 1]]);
snakeGame.move("R"); // return 0
snakeGame.move("D"); // return 0
snakeGame.move("R"); // return 1, snake eats the first piece of food. The second piece of food appears
                     // at (0, 1).

snakeGame.move("U"); // return 1
snakeGame.move("L"); // return 2, snake eats the second food. No more food appears.

snakeGame.move("U"); // return -1, game over because snake collides with border

Constraints:
`1 <= width, height <= 104`
`1 <= food.length <= 50`
`food[i].length == 2`
`0 <= ri < height`
`0 <= ci < width`
`direction.length == 1`
`direction` is `'U'`, `'D'`, `'L'`, or `'R'`.

At most `104` calls will be made to `move`.

### Solution 
 To solve the Snake Game problem, we'll implement the `SnakeGame` class that will interact with a 2D grid representing the game's space. The snake will be represented by a list of coordinates indicating its current position, and we will track the food pieces and the score. The key points to implement are tracking the snake's position, consuming food, maintaining the length of the snake, and checking for game-over conditions.

Here’s the solution formatted like a LeetCode submission:



```python
class SnakeGame:

    def __init__(self, width: int, height: int, food: List[List[int]]):
        self.width = width
        self.height = height
        self.food = food
        self.score = 0
        self.snake = [(0, 0)]  # Start with the snake at (0, 0)
        self.food_index = 0  # To track which food to eat next
        self.directions = {
            'U': (-1, 0),  # Up
            'D': (1, 0),   # Down
            'L': (0, -1),  # Left
            'R': (0, 1)    # Right
        }

    def move(self, direction: str) -> int:
        # Get the current head position of the snake
        head_x, head_y = self.snake[0]
        # Determine the new head position based on the direction
        move_x, move_y = self.directions[direction]
        new_head = (head_x + move_x, head_y + move_y)

        # Check game over conditions
        if (new_head[0] < 0 or new_head[0] >= self.height or
                new_head[1] < 0 or new_head[1] >= self.width or
                new_head in self.snake):
            return -1  # Game over

        # If we have food to eat and the new head position is on food
        if self.food_index < len(self.food) and new_head == tuple(self.food[self.food_index]):
            self.score += 1  # Increase score
            self.food_index += 1  # Move to the next food
        else:
            # If not eating, remove the tail of the snake
            self.snake.pop()  # Remove the end of the snake

        # Add the new head position to the snake
        self.snake.insert(0, new_head)

        return self.score

```

### Explanation
1. **Initialization (`__init__` method)**:
   - We initialize the board dimensions with `width` and `height`.
   - `food` holds the positions of food items.
   - `score` keeps track of the snake's score.
   - `snake` initializes with the starting position of the snake at `(0, 0)`.
   - `food_index` keeps track of how many food items have been consumed.
   - `directions` is a dictionary mapping directions (`'U'`, `'D'`, `'L'`, `'R'`) to their respective coordinate changes.

2. **Move Method (`move` function)**:
   - We get the current head position of the snake.
   - Based on the provided direction, we compute the new head position.
   - We check for game-over conditions:
     - If the new head is out of bounds (either negative or greater than or equal to dimensions).
     - If the new head collides with any part of the snake's body (i.e., if it's already in the `snake` list).
   - For eating food:
     - If the new head position matches the current food position, we increment the score and the `food_index`.
     - If not eating food, we remove the tail of the snake to keep its size constant unless it is growing.
   - Finally, we add the new head position to our snake and return the current score.

### Constraints Handling
- The code efficiently maintains the state of the game ensuring that the snake can grow when it eats food, while also checking for boundary and self-collision conditions.

This solution should work effectively given the problem's constraints, and you can run it directly in the LeetCode environment after importing required libraries like List from typing. Make sure to use the appropriate environment to test this code.

# 363. Max Sum of Rectangle No Larger Than K

### Problem Description 
Given an `m x n` matrix `matrix` and an integer `k`, return the max sum of a rectangle in the matrix such that its sum is no larger than `k`.

It is guaranteed that there will be a rectangle with a sum no larger than `k`.


Example 1:
Input: matrix = [[1,0,1],[0,-2,3]], k = 2
Output: 2
Explanation: Because the sum of the blue rectangle [[0, 1], [-2, 3]] is 2, and 2 is the max number no larger than k (k = 2).


Example 2:
Input: matrix = [[2,2,-1]], k = 3
Output: 3

Constraints:
`m == matrix.length`
`n == matrix[i].length`
`1 <= m, n <= 100`
`-100 <= matrix[i][j] <= 100`
`-105 <= k <= 105`
Follow up: What if the number of rows is much larger than the number of columns?

### Solution 
 To solve the problem of finding the maximum sum of a rectangle in a matrix that is less than or equal to a given integer `k`, we can adopt a method involving the concept of a cumulative sum along with an efficient data structure to track sums.

### Explanation

1. **Cumulative Sum**: We'll utilize the concept of a cumulative or prefix sum where we calculate the sum of elements in the matrix efficiently in a way that allows us to extract the sum of any subrectangle in constant time.

2. **Iterative Approach**: We iterate over pairs of rows. For each pair of rows, we consider the submatrix that can be formed between these two rows. We then collapse the selected rows into a single array representing column sums.

3. **Set for Tracking Sums**: To efficiently find the best possible sum that doesn’t exceed `k`, we will store the prefix sums in a sorted set. For each new column sum, we will use a binary search to find the largest possible sum that can be formed by adding it to an existing prefix sum that does not exceed `k`.

4. **Complexity Consideration**: The overall complexity will be O(m^2 * n * log(n)), which is efficient for the given constraints (`m, n <= 100`).

Here is the code that implements the above logic:



```python
class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        from sortedcontainers import SortedList
        
        max_sum = float('-inf')
        rows, cols = len(matrix), len(matrix[0])
        
        # Iterate over all pairs of rows
        for start_row in range(rows):
            # Initialize a list to store the column sums for the row range
            col_sum = [0] * cols
            
            for end_row in range(start_row, rows):
                # Update the column sums for the current end_row
                for c in range(cols):
                    col_sum[c] += matrix[end_row][c]
                
                # Now we need to find the maximum subarray sum <= k in col_sum
                # Using a sorted list to maintain the prefix sums
                prefix_sums = SortedList([0])
                current_sum = 0
                
                for sum_value in col_sum:
                    current_sum += sum_value
                    
                    # We want to find the largest prefix_sum in prefix_sums
                    # such that current_sum - prefix_sum <= k
                    target = current_sum - k
                    
                    # Using binary search to find the best prefix
                    idx = prefix_sums.bisect_left(target)
                    
                    if idx < len(prefix_sums):
                        max_sum = max(max_sum, current_sum - prefix_sums[idx])
                    
                    # Add the current prefix sum to set
                    prefix_sums.add(current_sum)

        return max_sum

```

### Key Parts of the Code:
1. **Initialization**: We define `max_sum` to track the maximum found so far and set up our dimensions `rows` and `cols`.

2. **Outer Loop for Row Pairs**: The first outer loop picks the starting row, and the inner loop iterates through the end rows to define the rectangle.

3. **Column Sums Calculation**: For each pair of rows selected, we compute the corresponding column sums in `col_sum`.

4. **Maintaining Prefix Sums**: By using a `SortedList`, we maintain and query prefix sums efficiently.

5. **Binary Search for Best Fit**: The binary search helps us to find the most appropriate prefix sum, ensuring we are getting the maximum sum that is still less than or equal to `k`.

### Conclusion
This approach efficiently finds the maximum sum of a rectangle in the matrix that is no larger than `k` using smart data structures and algorithms. It adheres to all the constraints and thus is optimal for the problem at hand.

# 582. Kill Process

### Problem Description 
You have `n` processes forming a rooted tree structure. You are given two integer arrays `pid` and `ppid`, where `pid[i]` is the ID of the `ith` process and `ppid[i]` is the ID of the `ith` process's parent process.

Each process has only one parent process but may have multiple children processes. Only one process has `ppid[i] = 0`, which means this process has no parent process (the root of the tree).

When a process is killed, all of its children processes will also be killed.

Given an integer `kill` representing the ID of a process you want to kill, return a list of the IDs of the processes that will be killed. You may return the answer in any order.


Example 1:
Input: pid = [1,3,10,5], ppid = [3,0,5,3], kill = 5
Output: [5,10]
Explanation: The processes colored in red are the processes that should be killed.


Example 2:
Input: pid = [1], ppid = [0], kill = 1
Output: [1]

Constraints:
`n == pid.length`
`n == ppid.length`
`1 <= n <= 5 * 104`
`1 <= pid[i] <= 5 * 104`
`0 <= ppid[i] <= 5 * 104`
Only one process has no parent.

All the values of `pid` are unique.

`kill` is guaranteed to be in `pid`.

### Solution 
 To solve the problem of killing a process and all of its children in a tree structure represented by arrays `pid` and `ppid`, we can follow these steps:

1. **Build the Tree Structure**: We need to use a dictionary to represent the tree, where each key is the process ID (`pid`), and the value is a list of child process IDs that correspond to that parent ID (`ppid`).

2. **Depth-First Search (DFS)**: Once the tree is constructed, we can use a DFS approach starting from the `kill` process to collect all processes that will be killed, including the initial process and all its descendants.

3. **Return the Result**: Finally, we return the collected process IDs as a list.

Let's implement this logic in Python:



```python
from collections import defaultdict

class Solution:
    def killProcess(self, pid: List[int], ppid: List[int], kill: int) -> List[int]:
        # Step 1: Build the tree
        tree = defaultdict(list)
        for p, parent in zip(pid, ppid):
            tree[parent].append(p)
        
        # Step 2: Use DFS to collect all processes to be killed
        def dfs(process_id):
            killed_processes = [process_id]  # Start with the current process
            for child_process in tree[process_id]:  # Check all children
                killed_processes.extend(dfs(child_process))  # Recursively kill children
            return killed_processes
        
        # Step 3: Call DFS on the process to be killed
        return dfs(kill)


```

### Explanation of the Code:

1. **Importing Required Libraries**: We import `defaultdict` from `collections` for easy management of tree structures.

2. **Class Definition**: We define a class `Solution` containing the `killProcess` method.

3. **Building the Tree**:
   - We create a `defaultdict` called `tree`, where each key will be a parent process and its value will be a list of child processes.
   - We iterate over the provided `pid` and `ppid` arrays simultaneously using `zip`, filling the tree accordingly.

4. **DFS Function**:
   - The `dfs` function takes a `process_id` as input and initializes a list `killed_processes` with the current process ID.
   - It then iterates over each child process of the current `process_id` (retrieved from the `tree`).
   - For each child, it calls `dfs` recursively and appends the results to `killed_processes`.
   - Finally, it returns the full list of killed processes.

5. **Invoke DFS on the Kill Process**:
   - We call the `dfs` function starting from the `kill` process to get all of the processes that will be killed.

### Complexity:
- **Time Complexity**: O(n) where n is the number of processes. In the worst case, we visit each process once.
- **Space Complexity**: O(n) for storing the tree and the list of killed processes, in the worst case scenario.

This approach ensures that we traverse the entire affected subtree efficiently and collect all necessary process IDs, as demonstrated in the provided examples.

# 621. Task Scheduler

### Problem Description 
Given a characters array `tasks`, representing the tasks a CPU needs to do, where each letter represents a different task. Tasks could be done in any order. Each task is done in one unit of time. For each unit of time, the CPU could complete either one task or just be idle.

However, there is a non-negative integer `n` that represents the cooldown period between two same tasks (the same letter in the array), that is that there must be at least `n` units of time between any two same tasks.

Return the least number of units of times that the CPU will take to finish all the given tasks.


Example 1:
Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
Explanation: 
A -> B -> idle -> A -> B -> idle -> A -> B
There is at least 2 units of time between any two same tasks.


Example 2:
Input: tasks = ["A","A","A","B","B","B"], n = 0
Output: 6
Explanation: On this case any permutation of size 6 would work since n = 0.

["A","A","A","B","B","B"]
["A","B","A","B","A","B"]
["B","B","B","A","A","A"]
...

And so on.


Example 3:
Input: tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2
Output: 16
Explanation: 
One possible solution is
A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> idle -> idle -> A -> idle -> idle -> A

Constraints:
`1 <= task.length <= 104`
`tasks[i]` is upper-case English letter.

The integer `n` is in the range `[0, 100]`.

### Solution 
 To solve the problem of scheduling tasks with cooldowns in a way that minimizes the total time taken, we can follow these steps:

1. **Understand the Problem**: We have tasks represented by letters in an array. We need to ensure that the same letter (task) does not occur within `n` time units of itself. If we run out of available tasks to schedule, we must stay idle, which also counts as time.

2. **Plan**: We first count the frequency of each task. The key insight is that we need to schedule the most frequent tasks first and fill in the gaps with the less frequent tasks or idle time as needed. We can think of constructing "blocks" of tasks spaced out with idle time.

3. **Execution**:
   - Count the frequency of each task using a counter.
   - Identify the task with the highest frequency. 
   - Calculate the required time slots for these highest frequency tasks.
   - Use idle time to fill the gaps based on the cooldown requirement `n`.

4. **Implementation**:
   - Create an array to count the occurrences of each task.
   - Count the maximum frequency and how many tasks have this maximum frequency.
   - Calculate the initial time required based on these maximum frequency tasks and how many additional idle slots are needed.

Here is the Python code implementing this logic in a format suitable for LeetCode:



```python
from collections import Counter

class Solution:
    def leastInterval(self, tasks: list[str], n: int) -> int:
        # Count the frequency of each task
        task_count = Counter(tasks)
        
        # Get the maximum frequency of any task
        max_freq = max(task_count.values())
        
        # Count how many tasks have the maximum frequency
        max_freq_tasks_count = sum(1 for freq in task_count.values() if freq == max_freq)
        
        # Calculate the minimum units of time required
        # Use the formula:
        # Total time = (Max frequency - 1) * (n + 1) + max_freq_tasks_count
        total_time = (max_freq - 1) * (n + 1) + max_freq_tasks_count
        
        # Return the maximum of calculated time and length of tasks,
        # because if we have very few tasks, we need at least len(tasks) time.
        return max(total_time, len(tasks))

# Example usage:
sol = Solution()
print(sol.leastInterval(["A","A","A","B","B","B"], 2))  # Output: 8
print(sol.leastInterval(["A","A","A","B","B","B"], 0))  # Output: 6
print(sol.leastInterval(["A","A","A","A","A","A","B","C","D","E","F","G"], 2))  # Output: 16

```

### Detailed Explanation:
- **Counter**: We use `Counter` to tally occurrences of each task efficiently.
- **Max Frequency Calculation**: By identifying the maximum frequency task, we can establish how many 'blocks' we need to schedule and how they will be arranged, taking into account the necessary idle time due to cooldown requirements.
- **Formula for Total Time**: The formula used accounts for the required idle time slots that need to be created between the occurrences of the max frequency tasks. The final time is adjusted to ensure we cover all tasks by comparing with the total number of tasks given.

This solution ensures that the tasks are scheduled optimally, fulfilling the cooldown constraint while minimizing the total time required to complete all tasks.

# 622. Design Circular Queue

### Problem Description 
Design your implementation of the circular queue. The circular queue is a linear data structure in which the operations are performed based on FIFO (First In First Out) principle and the last position is connected back to the first position to make a circle. It is also called "Ring Buffer".

One of the benefits of the circular queue is that we can make use of the spaces in front of the queue. In a normal queue, once the queue becomes full, we cannot insert the next element even if there is a space in front of the queue. But using the circular queue, we can use the space to store new values.

Implementation the `MyCircularQueue` class:
`MyCircularQueue(k)` Initializes the object with the size of the queue to be `k`.

`int Front()` Gets the front item from the queue. If the queue is empty, return `-1`.

`int Rear()` Gets the last item from the queue. If the queue is empty, return `-1`.

`boolean enQueue(int value)` Inserts an element into the circular queue. Return `true` if the operation is successful.

`boolean deQueue()` Deletes an element from the circular queue. Return `true` if the operation is successful.

`boolean isEmpty()` Checks whether the circular queue is empty or not.

`boolean isFull()` Checks whether the circular queue is full or not.


Example 1:
Input
["MyCircularQueue", "enQueue", "enQueue", "enQueue", "enQueue", "Rear", "isFull", "deQueue", "enQueue", "Rear"]
[[3], [1], [2], [3], [4], [], [], [], [4], []]
Output
[null, true, true, true, false, 3, true, true, true, 4]
Explanation
MyCircularQueue myCircularQueue = new MyCircularQueue(3);
myCircularQueue.enQueue(1); // return True
myCircularQueue.enQueue(2); // return True
myCircularQueue.enQueue(3); // return True
myCircularQueue.enQueue(4); // return False
myCircularQueue.Rear();     // return 3
myCircularQueue.isFull();   // return True
myCircularQueue.deQueue();  // return True
myCircularQueue.enQueue(4); // return True
myCircularQueue.Rear();     // return 4

Constraints:
`1 <= k <= 1000`
`0 <= value <= 1000`
At most `3000` calls will be made to `enQueue`, `deQueue`, `Front`, `Rear`, `isEmpty`, and `isFull`.

Follow up: Could you solve the problem without using the built-in queue?

### Solution 
 To solve the problem of designing a circular queue as described, we can create a class `MyCircularQueue` that implements the necessary methods to manage the circular queue. Let's break down the implementation step-by-step.

### Key Concepts:
1. **Circular Array**: We'll use an array to represent the circular queue. When we reach the end of the array, we wrap around to the beginning.
2. **Pointers:** 
   - `head`: Points to the front of the queue.
   - `tail`: Points to the next position to insert a new element.
   - `size`: Keeps track of the current number of elements in the queue.
3. **Capacity Management**: We have a fixed capacity `k` which denotes the maximum number of elements the queue can hold.

### Methods:
- **Constructor**: Initializes an empty circular queue with a given size.
- **enQueue(value)**: Adds an element to the queue if it's not full.
- **deQueue()**: Removes an element from the queue if it's not empty.
- **Front()**: Returns the front element or -1 if the queue is empty.
- **Rear()**: Returns the last element or -1 if the queue is empty.
- **isEmpty()**: Checks if the queue is empty.
- **isFull()**: Checks if the queue is full.

### Implementation:
Now, let's write the complete implementation of the class:



```python
class MyCircularQueue:
    def __init__(self, k: int):
        self.k = k  # Size of the queue
        self.queue = [0] * k  # Initialize the queue with fixed size
        self.head = 0  # Pointer to the front of the queue
        self.tail = 0  # Pointer to the next insert position
        self.size = 0  # Current number of elements

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        self.queue[self.tail] = value  # Insert the item at the tail
        self.tail = (self.tail + 1) % self.k  # Move tail to the next position
        self.size += 1  # Increase size
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        self.head = (self.head + 1) % self.k  # Move head to the next position
        self.size -= 1  # Decrease size
        return True

    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.head]  # Return the front element

    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        # Calculate the last element index as (tail - 1) % k
        return self.queue[(self.tail - 1 + self.k) % self.k]

    def isEmpty(self) -> bool:
        return self.size == 0  # If size is 0, the queue is empty

    def isFull(self) -> bool:
        return self.size == self.k  # If size is equal to k, the queue is full

# Example Usage
myCircularQueue = MyCircularQueue(3)
print(myCircularQueue.enQueue(1))  # Returns True
print(myCircularQueue.enQueue(2))  # Returns True
print(myCircularQueue.enQueue(3))  # Returns True
print(myCircularQueue.enQueue(4))  # Returns False
print(myCircularQueue.Rear())       # Returns 3
print(myCircularQueue.isFull())     # Returns True
print(myCircularQueue.deQueue())    # Returns True
print(myCircularQueue.enQueue(4))   # Returns True
print(myCircularQueue.Rear())       # Returns 4

```

### Explanation of the Code:
- In the constructor `__init__`, we initialize the queue with a specified size using an array and set pointers for the head and tail.
- The `enQueue` method checks if the queue is full before inserting an element at the tail and updates the array and pointers accordingly.
- The `deQueue` method checks if the queue is empty before removing an element from the front (head) and updates the pointers.
- The `Front` and `Rear` methods return the respective elements from the queue or -1 if empty.
- The `isEmpty` and `isFull` methods provide a simple check on the size of the queue.

This implementation efficiently manages the circular queue with O(1) time complexity for each operation.

# 641. Design Circular Deque

### Problem Description 
Design your implementation of the circular double-ended queue (deque).

Your implementation should support following operations:
`MyCircularDeque(k)`: Constructor, set the size of the deque to be k.

`insertFront()`: Adds an item at the front of Deque. Return true if the operation is successful.

`insertLast()`: Adds an item at the rear of Deque. Return true if the operation is successful.

`deleteFront()`: Deletes an item from the front of Deque. Return true if the operation is successful.

`deleteLast()`: Deletes an item from the rear of Deque. Return true if the operation is successful.

`getFront()`: Gets the front item from the Deque. If the deque is empty, return -1.

`getRear()`: Gets the last item from Deque. If the deque is empty, return -1.

`isEmpty()`: Checks whether Deque is empty or not. 
`isFull()`: Checks whether Deque is full or not.


Example:
MyCircularDeque circularDeque = new MycircularDeque(3); // set the size to be 3
circularDeque.insertLast(1);			// return true
circularDeque.insertLast(2);			// return true
circularDeque.insertFront(3);			// return true
circularDeque.insertFront(4);			// return false, the queue is full
circularDeque.getRear();  			// return 2
circularDeque.isFull();				// return true
circularDeque.deleteLast();			// return true
circularDeque.insertFront(4);			// return true
circularDeque.getFront();			// return 4
Note:
All values will be in the range of [0, 1000].

The number of operations will be in the range of [1, 1000].

Please do not use the built-in Deque library.

### Solution 
 To solve the problem of implementing a circular double-ended queue (deque) in Python, you can use an array-based approach along with two pointers to manage the front and rear of the deque. The circular nature of the deque can be achieved by wrapping the indices around when they reach the boundaries of the underlying array.

Here’s a detailed explanation of the implementation:

1. **Data Structures**:
   - We'll maintain an array (list) to hold the elements of the deque.
   - Two pointers, `front` and `rear`, will track the positions where operations will be performed.
   - A variable `size` will keep track of the current number of elements in the deque.
   - A variable `capacity` will store the maximum size of the deque.

2. **Circular Index Handling**:
   - To efficiently handle the circular nature, we can use modulo operations. For example, the next position of `front` can be calculated as `(front - 1 + capacity) % capacity`, and for `rear`, it can be calculated as `(rear + 1) % capacity`.

3. **Operations**:
   - **Insertions (Front and Rear)**:
     - Ensure that the deque is not full before inserting; if it's full, return `False`.
   - **Deletions (Front and Rear)**:
     - Ensure that the deque is not empty before deleting; if it's empty, return `False`.
   - **Front and Rear Access**:
     - Retrieve the elements at the `front` and `rear` indices, return `-1` if the deque is empty.
   - **Checks (Is Empty and Is Full)**:
     - Use the `size` variable to determine if the deque is empty (size = 0) or full (size = capacity).

Here is the implementation based on the explanation:



```python
class MyCircularDeque:

    def __init__(self, k: int):
        self.capacity = k
        self.deque = [0] * k  # Initialize deque with a fixed size of k
        self.front = -1  # Points to the front of the deque
        self.rear = 0    # Points to the rear of the deque
        self.size = 0    # Current number of elements in the deque

    def insertFront(self, value: int) -> bool:
        if self.isFull():
            return False
        if self.isEmpty():
            self.front = 0 # When inserting into an empty deque
        else:
            self.front = (self.front - 1) % self.capacity
        
        self.deque[self.front] = value
        self.size += 1
        return True

    def insertLast(self, value: int) -> bool:
        if self.isFull():
            return False
        if self.isEmpty():
            self.rear = 0 # When inserting into an empty deque
        else:
            self.rear = (self.rear + 1) % self.capacity
        
        self.deque[self.rear] = value
        self.size += 1
        return True

    def deleteFront(self) -> bool:
        if self.isEmpty():
            return False
        if self.size == 1:
            self.front = -1  # Reset front when deque becomes empty
            self.rear = -1
        else:
            self.front = (self.front + 1) % self.capacity
        
        self.size -= 1
        return True

    def deleteLast(self) -> bool:
        if self.isEmpty():
            return False
        if self.size == 1:
            self.front = -1  # Reset front when deque becomes empty
            self.rear = -1
        else:
            self.rear = (self.rear - 1 + self.capacity) % self.capacity
        
        self.size -= 1
        return True

    def getFront(self) -> int:
        if self.isEmpty():
            return -1
        return self.deque[self.front]

    def getRear(self) -> int:
        if self.isEmpty():
            return -1
        return self.deque[self.rear]

    def isEmpty(self) -> bool:
        return self.size == 0

    def isFull(self) -> bool:
        return self.size == self.capacity

```

### Explanation of Each Method:
- **Constructor (`__init__`)**: Initializes the deque with a specific capacity and other required variables.
- **`insertFront(value)`**: Inserts a value at the front if the deque is not full.
- **`insertLast(value)`**: Inserts a value at the rear if the deque is not full.
- **`deleteFront()`**: Deletes the front element if the deque is not empty.
- **`deleteLast()`**: Deletes the rear element if the deque is not empty.
- **`getFront()`**: Returns the front element or `-1` if the deque is empty.
- **`getRear()`**: Returns the rear element or `-1` if the deque is empty.
- **`isEmpty()`**: Checks if the deque is empty.
- **`isFull()`**: Checks if the deque is full.

This implementation adheres to the constraints and requirements specified in the problem statement and should work efficiently within the given limits.

# 862. Shortest Subarray with Sum at Least K

### Problem Description 
Return the length of the shortest, non-empty, contiguous subarray of `A` with sum at least `K`.

If there is no non-empty subarray with sum at least `K`, return `-1`.


Example 1:
Input: A = [1], K = 1
Output: 1

Example 2:
Input: A = [1,2], K = 4
Output: -1

Example 3:
Input: A = [2,-1,2], K = 3
Output: 3
Note:
`1 <= A.length <= 50000`
`-10 ^ 5 <= A[i] <= 10 ^ 5`
`1 <= K <= 10 ^ 9`

### Solution 
 To solve the problem of finding the length of the shortest, non-empty, contiguous subarray of `A` with a sum that is at least `K`, we can use a combination of prefix sums and a deque (double-ended queue) to efficiently track the minimum length of qualifying subarrays.

### Explanation:

1. **Prefix Sums**:
   - We will compute the prefix sums of the array `A`. The prefix sum at index `i` represents the sum of all elements from the start of the array up to index `i`.
   - We can use these prefix sums to determine the sum of any subarray `A[l..r]` by subtracting the prefix sum at `l-1` from the prefix sum at `r`.

2. **Using a Deque**:
   - We'll maintain a deque (a double-ended queue) that will store indices of the prefix sums. This will allow us to efficiently find the shortest valid subarray.
   - As we iterate through the array, for each prefix sum, we will check if there exists a previous prefix sum such that the difference is at least `K` (i.e., `prefix[j] <= prefix[i] - K`). This indicates that the subarray from `j+1` to `i` has a sum that meets or exceeds `K`.
   - Whenever we find such an index, we can calculate the length of the subarray and update our minimum length variable.

3. **Optimization**:
   - We will ensure that the prefix sums in the deque are always sorted. This helps in maintaining efficient operations for both adding new sums and removing old sums from consideration.

### Implementation:

Below is the Python code that implements the above approach, formatted according to LeetCode specifications.



```python
from collections import deque
import sys

class Solution:
    def shortestSubarray(self, A: List[int], K: int) -> int:
        n = len(A)
        # Create prefix sum array with an initial 0 for prefix[0]
        prefix = [0] * (n + 1)
        
        for i in range(n):
            prefix[i + 1] = prefix[i] + A[i]
        
        # Initialize the deque (to store indices of prefix sums)
        d = deque()
        min_length = sys.maxsize  # Use a large value as initial minimum length
        
        for i in range(n + 1):
            # Check if we can form a valid subarray ending at i
            while d and prefix[i] - prefix[d[0]] >= K:
                min_length = min(min_length, i - d.popleft())
            
            # Maintain the order in deque
            while d and prefix[i] < prefix[d[-1]]:
                d.pop()
            
            d.append(i)  # Append the current index to the deque
        
        return min_length if min_length != sys.maxsize else -1

```

### Key Points:
- **Time Complexity**: The overall time complexity of this algorithm is O(n), where n is the length of `A`. Each index is processed at most twice (once added and once removed from the deque).
- **Space Complexity**: The space complexity is also O(n) because we are storing the prefix sums and using the deque.
- The solution efficiently finds the shortest subarray with the required properties by leveraging prefix sums and a deque to manage candidate indices.

# 933. Number of Recent Calls

### Problem Description 
You have a `RecentCounter` class which counts the number of recent requests within a certain time frame.

Implement the `RecentCounter` class:
`RecentCounter()` Initializes the counter with zero recent requests.

`int ping(int t)` Adds a new request at time `t`, where `t` represents some time in milliseconds, and returns the number of requests that has happened in the past `3000` milliseconds (including the new request). Specifically, return the number of requests that have happened in the inclusive range `[t - 3000, t]`.

It is guaranteed that every call to `ping` uses a strictly larger value of `t` than the previous call.


Example 1:
Input
["RecentCounter", "ping", "ping", "ping", "ping"]
[[], [1], [100], [3001], [3002]]
Output
[null, 1, 2, 3, 3]
Explanation
RecentCounter recentCounter = new RecentCounter();
recentCounter.ping(1);     // requests = [1], range is [-2999,1], return 1
recentCounter.ping(100);   // requests = [1, 100], range is [-2900,100], return 2
recentCounter.ping(3001);  // requests = [1, 100, 3001], range is [1,3001], return 3
recentCounter.ping(3002);  // requests = [1, 100, 3001, 3002], range is [2,3002], return 3

Constraints:
`1 <= t <= 109`
Each test case will call `ping` with strictly increasing values of `t`.

At most `104` calls will be made to `ping`.

### Solution 
 To solve the problem of counting recent requests within a specific time frame, we can utilize a list to keep track of the timestamps when each request occurs. The core idea is to maintain a window of valid request times that fall within the last 3000 milliseconds from the given ping time.

Here's a step-by-step breakdown of how we can implement the `RecentCounter` class:

1. **Initialization**: We will initialize an empty list (or deque) to store the requests. This list will be populated with the timestamps each time the `ping` method is called.

2. **Ping Method**: When the `ping(t)` method is called:
   - We first append the current timestamp `t` to our list of requests.
   - Next, we'll discard any timestamps that are older than `t - 3000` milliseconds. This ensures that we only keep the relevant timestamps within the desired range.
   - Finally, the method returns the count of the requests that remain in the list.

3. **Performance considerations**: Since we are guaranteed that timestamps will always be in increasing order, the list will contain timestamps in sorted order, allowing us to efficiently filter out the old timestamps.

Here's the implementation of the `RecentCounter` class:



```python
class RecentCounter:

    def __init__(self):
        """Initialize the counter with zero recent requests."""
        self.requests = []

    def ping(self, t: int) -> int:
        """Add a new request at time t, where t is the time in milliseconds, and return the number of requests
        that occurred in the last 3000 milliseconds (including the current request).
        """
        # Add the current request time
        self.requests.append(t)
        
        # Eliminate requests that are older than t - 3000
        while self.requests[0] < t - 3000:
            self.requests.pop(0)
        
        # Return the count of requests within the range
        return len(self.requests)

```

### Explanation of the Code:
- **Constructor (`__init__`)**: Initializes the `RecentCounter` instance and creates an empty list `requests` to store the request timestamps.
  
- **Ping Method**: 
  - Appends the new timestamp `t` to the `requests` list.
  - Removes timestamps from the front of the list (`requests[0]`) that are less than `t - 3000`.
  - Returns the length of the `requests` list, which reflects the number of valid requests within the last 3000 milliseconds.

### Example Usage:
To see how this class works, you can create an instance and invoke the `ping` method as shown below:



```python
recentCounter = RecentCounter()
print(recentCounter.ping(1))    # Output: 1
print(recentCounter.ping(100))  # Output: 2
print(recentCounter.ping(3001)) # Output: 3
print(recentCounter.ping(3002)) # Output: 3

```

This Python code will work correctly on LeetCode as per the problem statement. The implementation handles the constraints efficiently, ensuring we can process up to 10,000 requests without degradation in performance due to the linear nature of list operations.

# 1673. Find the Most Competitive Subsequence

### Problem Description 
Given an integer array `nums` and a positive integer `k`, return the most competitive subsequence of `nums` of size `k`.

An array's subsequence is a resulting sequence obtained by erasing some (possibly zero) elements from the array.

We define that a subsequence `a` is more competitive than a subsequence `b` (of the same length) if in the first position where `a` and `b` differ, subsequence `a` has a number less than the corresponding number in `b`. For example, `[1,3,4]` is more competitive than `[1,3,5]` because the first position they differ is at the final number, and `4` is less than `5`.


Example 1:
Input: nums = [3,5,2,6], k = 2
Output: [2,6]
Explanation: Among the set of every possible subsequence: {[3,5], [3,2], [3,6], [5,2], [5,6], [2,6]}, [2,6] is the most competitive.


Example 2:
Input: nums = [2,4,3,3,5,4,9,6], k = 4
Output: [2,3,3,4]

Constraints:
`1 <= nums.length <= 105`
`0 <= nums[i] <= 109`
`1 <= k <= nums.length`

### Solution 
 To solve the problem of finding the most competitive subsequence of size `k` from the given integer array `nums`, we can use a greedy approach with a stack. The objective is to maintain the smallest elements in the subsequence while ensuring that we have enough elements left in the array to fill the subsequence of size `k`.

Here’s a step-by-step breakdown of the solution:

### Approach:

1. **Stack for the Result**: We will use a stack to build our most competitive subsequence. The stack allows us to easily manage and retrieve the smallest elements while keeping track of our current choice.

2. **Iterate Through Elements**: We iterate through the elements of `nums`. For each element, we consider whether to include it in our subsequence.

3. **Maintain the Size Limit**: If the stack is not empty and the current element is smaller than the top element of the stack, we can pop elements from the stack (i.e., discard them from our current potential subsequence), as long as:
   - Removing the top of the stack still leaves enough remaining elements in `nums` to complete the sequence of size `k`.

4. **Push Current Element**: After checking for popping, we push the current element onto the stack if we haven't yet filled our subsequence of size `k`.

5. **Construct the Result**: Finally, the first `k` elements from the stack will be our answer since we have ensured that they are the most competitive.

### Python Code:

Here’s how the implementation looks in Python:



```python
class Solution:
    def mostCompetitive(self, nums: List[int], k: int) -> List[int]:
        stack = []
        n = len(nums)
        
        for i in range(n):
            # While stack is not empty, and the current number is less than the last
            # number in the stack, and we can still complete a sequence of length k
            while (stack and stack[-1] > nums[i] and 
                   len(stack) + (n - i) > k):
                stack.pop()
            # If we still need more elements in the subsequence, push current element
            if len(stack) < k:
                stack.append(nums[i])
        
        # Return the first k numbers of the stack
        return stack[:k]

```

### Explanation of the Code:

1. **Initialization**: We create an empty `stack` to store the most competitive subsequence elements.

2. **Loop through `nums`**: For each element in `nums`, we check the conditions for popping from the stack:
   - `stack` should not be empty.
   - The top element of the stack should be greater than the current element `nums[i]` (to maintain competitiveness).
   - `(len(stack) + (n - i) > k)` ensures that if we pop the stack, there are enough remaining elements to still reach size `k`.

3. **Push Current Element**: If we haven't yet filled our required size `k`, we push `nums[i]` to the stack.

4. **Return Result**: Finally, we return the first `k` elements of the stack as the most competitive subsequence.

### Complexity Analysis:
- **Time Complexity**: O(n), where n is the length of `nums`. Each element is pushed and popped from the stack at most once.
- **Space Complexity**: O(k) for the stack that holds our result.

This solution efficiently finds the most competitive subsequence as required by the problem statement.

# 1825. Finding MK Average

### Problem Description 
You are given two integers, `m` and `k`, and a stream of integers. You are tasked to implement a data structure that calculates the MKAverage for the stream.

The MKAverage can be calculated using these steps:
If the number of the elements in the stream is less than `m` you should consider the MKAverage to be `-1`. Otherwise, copy the last `m` elements of the stream to a separate container.

Remove the smallest `k` elements and the largest `k` elements from the container.

Calculate the average value for the rest of the elements rounded down to the nearest integer.

Implement the `MKAverage` class:
`MKAverage(int m, int k)` Initializes the MKAverage object with an empty stream and the two integers `m` and `k`.

`void addElement(int num)` Inserts a new element `num` into the stream.

`int calculateMKAverage()` Calculates and returns the MKAverage for the current stream rounded down to the nearest integer.


Example 1:
Input
["MKAverage", "addElement", "addElement", "calculateMKAverage", "addElement", "calculateMKAverage", "addElement", "addElement", "addElement", "calculateMKAverage"]
[[3, 1], [3], [1], [], [10], [], [5], [5], [5], []]
Output
[null, null, null, -1, null, 3, null, null, null, 5]
Explanation
MKAverage obj = new MKAverage(3, 1); 
obj.addElement(3);        // current elements are [3]
obj.addElement(1);        // current elements are [3,1]
obj.calculateMKAverage(); // return -1, because m = 3 and only 2 elements exist.

obj.addElement(10);       // current elements are [3,1,10]
obj.calculateMKAverage(); // The last 3 elements are [3,1,10].

                          // After removing smallest and largest 1 element the container will be `[3].

                          // The average of [3] equals 3/1 = 3, return 3
obj.addElement(5);        // current elements are [3,1,10,5]
obj.addElement(5);        // current elements are [3,1,10,5,5]
obj.addElement(5);        // current elements are [3,1,10,5,5,5]
obj.calculateMKAverage(); // The last 3 elements are [5,5,5].

                          // After removing smallest and largest 1 element the container will be `[5].

                          // The average of [5] equals 5/1 = 5, return 5
``

Constraints:
`3 <= m <= 105`
`1 <= k*2 < m`
`1 <= num <= 105`
At most `105` calls will be made to `addElement` and `calculateMKAverage`.

### Solution 
 To solve the MKAverage problem, we need to implement a class that processes elements in a stream and calculates the MKAverage based on specified rules. Let's break down the solution and then present the code which is suitable for running in an online coding platform like LeetCode.

### Problem Breakdown

1. **Initialization (`__init__` method)**:
   - The class `MKAverage` is initialized with two integers, `m` and `k`.
   - We will use a list `stream` to store the incoming elements.
   - We can also use a deque for efficient addition and removal of elements if necessary.

2. **Adding Elements (`addElement` method)**:
   - We append new elements to `stream`.
   - This simulates the stream in which data can be added.

3. **Calculating MKAverage (`calculateMKAverage` method)**:
   - If the number of elements in the `stream` is less than `m`, return `-1`.
   - Otherwise, we will take the last `m` elements of the stream.
   - Remove the smallest `k` and largest `k` elements from this subset.
   - Compute the average of the remaining elements, and return the average rounded down to the nearest integer.

### Complexity Consideration

- We need to sort the last `m` elements to remove the smallest and largest `k` elements. Sorting has a time complexity of O(m log m).
- Therefore, every operation of adding an element followed by calculating the MKAverage can potentially be O(m log m) in the worst case, but given the constraints (1 ≤ num ≤ 10^5), this should be manageable.

### Implementation

Here is the complete code:



```python
from collections import deque

class MKAverage:

    def __init__(self, m: int, k: int):
        self.m = m
        self.k = k
        self.stream = []
        self.last_m_elements = deque(maxlen=m)  # store the last 'm' elements

    def addElement(self, num: int) -> None:
        self.stream.append(num)
        self.last_m_elements.append(num)

    def calculateMKAverage(self) -> int:
        if len(self.stream) < self.m:
            return -1
        
        # Get the last 'm' elements
        recent_m_elements = list(self.last_m_elements)
        
        # Sort the elements to remove the smallest k and largest k
        recent_m_elements.sort()
        
        # Remove k smallest and k largest
        truncated_list = recent_m_elements[self.k:self.m - self.k]
        
        # Calculate the average of the remaining elements
        if not truncated_list:  # If no elements left after truncation
            return 0
        
        average = sum(truncated_list) // len(truncated_list)  # floor division
        return average

```

### Explanation of the Code:

1. **`__init__`:**
    - Initializes the `MKAverage` object with parameters `m` and `k`.
    - Initializes the `stream` list to store all elements and a `deque` with a maximum length of `m` to keep track of the most recent `m` elements.

2. **`addElement`:**
    - Adds an element to both the `stream` list and the `deque`. The `deque` ensures we only keep the last `m` elements.

3. **`calculateMKAverage`:**
    - Checks if the total number of elements is less than `m`. If yes, it returns `-1`.
    - Fetches the last `m` elements and sorts them to easily remove the smallest `k` and largest `k` elements.
    - After truncating, it calculates the average of the remaining elements. If there are no elements left after truncation, it returns `0`.

This structure ensures that we can efficiently maintain the stream and calculate the MKAverage according to the specified requirements of the problem while adherently processing each function with care to performance limitations.

