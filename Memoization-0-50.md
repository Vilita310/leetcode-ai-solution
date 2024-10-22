# 329. Longest Increasing Path in a Matrix

### Problem Description 
Given an `m x n` integers `matrix`, return the length of the longest increasing path in `matrix`.

From each cell, you can either move in four directions: left, right, up, or down. You may not move diagonally or move outside the boundary (i.e., wrap-around is not allowed).


Example 1:
Input: matrix = [[9,9,4],[6,6,8],[2,1,1]]
Output: 4
Explanation: The longest increasing path is `[1, 2, 6, 9]`.


Example 2:
Input: matrix = [[3,4,5],[3,2,6],[2,2,1]]
Output: 4
Explanation: The longest increasing path is `[3, 4, 5, 6]`. Moving diagonally is not allowed.


Example 3:
Input: matrix = [[1]]
Output: 1

Constraints:
`m == matrix.length`
`n == matrix[i].length`
`1 <= m, n <= 200`
`0 <= matrix[i][j] <= 231 - 1`

### Solution 
 To solve the problem of finding the longest increasing path in a matrix, we can utilize Depth-First Search (DFS) along with memoization to optimize our solution.

### Detailed Explanation:

1. **Understanding the Problem**: We need to explore each cell in the matrix and find paths that consist of strictly increasing values. From each cell, we can move to its four adjacent cells (up, down, left, right). The goal is to determine the length of the longest such path.

2. **Constraints**: Given the constraints of the matrix size (up to 200x200), a brute force search would be inefficient. Instead, an optimized approach using memorization will help us avoid redundant calculations.

3. **Approach**:
   - Use DFS to explore the possible paths starting from each cell.
   - Use a memoization table to store the results of already computed paths for each cell.
   - For each cell, explore its adjacent cells. If an adjacent cell has a value greater than the current cell, recursively compute the path from that cell.
   - The length of the longest increasing path from a cell is `1 + maximum path length from its valid neighboring cells`.

4. **Initialization**: Before starting DFS, ensure we have a memoization table initialized where each cell is set to -1 initially, indicating that it has not yet been computed.

5. **Final Result**: Loop through each cell in the matrix, calculate the longest increasing path using DFS, and track the maximum length found.

Now, let's implement the solution in Python following the LeetCode solution format.



```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        memo = [[-1 for _ in range(n)] for _ in range(m)]
        
        def dfs(x: int, y: int) -> int:
            # If already computed, return the stored result
            if memo[x][y] != -1:
                return memo[x][y]
            
            # Directions for left, right, up, down
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            max_length = 1  # The length of the path starting from (x, y)

            # Explore all four possible directions
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # Check bounds and increasing condition
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] > matrix[x][y]:
                    # Compute the length of increasing path from the neighbor
                    length = 1 + dfs(nx, ny)
                    max_length = max(max_length, length)
            
            # Memoize the result
            memo[x][y] = max_length
            return max_length
        
        longest_path = 0
        # Iterate through each cell in the matrix
        for i in range(m):
            for j in range(n):
                # Compute the longest increasing path starting from (i, j)
                longest_path = max(longest_path, dfs(i, j))

        return longest_path

```

### Explanation of the Code:
- We create a `Solution` class containing the method `longestIncreasingPath`.
- `memo` is used to store the longest increasing path length for each cell.
- The nested `dfs` function recursively calculates the longest increasing path for each cell.
- We explore each potential movement direction and ensure we only proceed to cells with a higher value.
- Finally, we iterate over each cell in the matrix to find the overall longest path and return that value.

This approach ensures that we efficiently compute the longest increasing paths without excessive recomputation, making it suitable even for the maximum constraints.

