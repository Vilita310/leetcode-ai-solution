# 207. Course Schedule

### Problem Description 
There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [ai, bi]` indicates that you must take course `bi` first if you want to take course `ai`.

For example, the pair `[0, 1]`, indicates that to take course `0` you have to first take course `1`.

Return `true` if you can finish all courses. Otherwise, return `false`.


Example 1:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0. So it is possible.


Example 2:
Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.


Constraints:
`1 <= numCourses <= 105`
`0 <= prerequisites.length <= 5000`
`prerequisites[i].length == 2`
`0 <= ai, bi < numCourses`
All the pairs prerequisites[i] are unique.

### Solution 
 To solve the given problem, we can use a graph representation of the courses and their prerequisites. The problem can be conceptualized as detecting a cycle in a directed graph. Here's how we'll approach it:

1. **Graph Representation**: We can represent the courses and their dependencies using a directed graph. Each course is a node, and a directed edge from course `bi` to course `ai` indicates that `bi` must be completed before `ai`.

2. **Cycle Detection**: If there is a cycle in this directed graph, it means that there is no way to complete all courses (since at least one course is dependent on itself indirectly). We can use Depth-First Search (DFS) to detect cycles in the graph.

3. **DFS Marking**: We will use three markers for DFS:
   - `0` (unvisited)
   - `1` (visiting) - indicates that the node is currently in the recursion stack
   - `2` (visited) - indicates that we've completely processed the node

4. **Implementation Steps**:
   - Build an adjacency list for the graph from the prerequisites.
   - Use a DFS function to explore the courses:
     - If we reach a node that is currently being visited (`1`), we found a cycle.
     - After fully processing a node, we mark it as `2`.

5. **Return Result**: If we find a cycle during our DFS traversal, we return `false`. Otherwise, if we can complete the DFS for all courses without finding a cycle, we return `true`.

Here’s the Python code implementing the above logic, formatted to run directly on LeetCode:



```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # Create a graph as an adjacency list
        from collections import defaultdict
        
        graph = defaultdict(list)
        for ai, bi in prerequisites:
            graph[bi].append(ai)
        
        # Visitation states: 0 = unvisited, 1 = visiting, 2 = visited
        visit = [0] * numCourses
        
        def dfs(course: int) -> bool:
            if visit[course] == 1:  # Currently visiting, cycle found
                return False
            if visit[course] == 2:  # Already fully processed
                return True
            
            # Mark the course as visiting
            visit[course] = 1
            
            # Visit all the neighbors (next courses)
            for next_course in graph[course]:
                if not dfs(next_course):  # If a cycle is found
                    return False
            
            # Mark the course as visited
            visit[course] = 2
            return True
        
        # Check all courses
        for course in range(numCourses):
            if visit[course] == 0:  # Not visited yet
                if not dfs(course):  # If a cycle is detected, return false
                    return False
        
        # If no cycles detected, return true
        return True

```

### Explanation of the Code:
- **Graph Construction**: We construct an adjacency list using a dictionary where each course points to the courses that depend on it.
- **DFS Function**: Our `dfs` function checks for cycles through marking each course:
  - It first marks the current course as `visiting`.
  - Then, it recursively visits all courses that depend on the current course.
  - If we encounter a course that has already been marked as `visiting`, we found a cycle, and we return `false`.
  - After exploring all dependents, we mark the current course as `visited`.
- **Final Traversal**: Finally, we check each course; if any DFS call detects a cycle, we return `false`.

This algorithm has a complexity of \(O(V + E)\) where \(V\) is the number of courses and \(E\) is the number of prerequisite relations.

# 210. Course Schedule II

### Problem Description 
There are a total of `n` courses you have to take labelled from `0` to `n - 1`.

Some courses may have `prerequisites`, for example, if `prerequisites[i] = [ai, bi]` this means you must take the course `bi` before the course `ai`.

Given the total number of courses `numCourses` and a list of the `prerequisite` pairs, return the ordering of courses you should take to finish all courses.

If there are many valid answers, return any of them. If it is impossible to finish all courses, return an empty array.


Example 1:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1].


Example 2:
Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0.

So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3].


Example 3:
Input: numCourses = 1, prerequisites = []
Output: [0]

Constraints:
`1 <= numCourses <= 2000`
`0 <= prerequisites.length <= numCourses * (numCourses - 1)`
`prerequisites[i].length == 2`
`0 <= ai, bi < numCourses`
`ai != bi`
All the pairs `[ai, bi]` are distinct.

### Solution 
 To solve the problem of finding an order to take the courses given their prerequisites, we can model the situation using graph theory. Each course can be represented as a node and each prerequisite pair as a directed edge. Our objective is to perform a topological sort on this directed graph.

### Detailed Explanation

1. **Graph Representation**: 
   We represent the directed graph using an adjacency list. We also maintain an array to count the in-degrees (the number of prerequisites) for each course.

2. **Topological Sort**: 
   We can use Kahn's algorithm for topological sorting, which processes nodes with zero in-degrees (courses that have no prerequisites) and reduces the in-degrees of their neighbors. If we can process all nodes, we can construct a valid ordering of the courses.

3. **Cycle Detection**: 
   If at the end of processing, we haven't been able to process all courses, it means there is a cycle in the graph, and thus it's impossible to complete all courses. In such a case, we return an empty list.

4. **Implementation**:
   The steps involve initializing our graph, setting up the in-degrees, and using a queue to hold the courses that can be taken next (zero in-degree). We will continue until we either process all courses or find itself stuck.

### Python Code
Here is the Python solution that follows the LeetCode format:



```python
from collections import deque, defaultdict

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # Initialize the graph and in-degree count
        graph = defaultdict(list)
        in_degree = [0] * numCourses
        
        # Build the graph
        for pair in prerequisites:
            a, b = pair
            graph[b].append(a)  # b -> a means to take a, b must be taken first
            in_degree[a] += 1   # Increase the in-degree of a

        # Queue for courses with zero in-degree
        zero_in_degree_queue = deque()
        for i in range(numCourses):
            if in_degree[i] == 0:
                zero_in_degree_queue.append(i)

        order = []
        
        # Process the queue
        while zero_in_degree_queue:
            course = zero_in_degree_queue.popleft()
            order.append(course)  # Take the course
            
            # Decrease in-degree of neighbors
            for neighbor in graph[course]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree_queue.append(neighbor)

        # Check if we were able to take all courses
        if len(order) == numCourses:
            return order
        else:
            return []

```

### Explanation of Code

- **Graph Construction**: 
  In the loop where we process the prerequisites, we set up the graph, mapping each course to its neighbors and counting in-degrees.

- **Initializing the Queue**: 
  We add all courses with zero in-degrees to the queue since they can be taken immediately.

- **Processing the Queue**: 
  We dequeue a course, add it to the result list, and then reduce the in-degrees for all its neighboring courses. If any neighbor's in-degree drops to zero, we add it to the queue.

- **Final Check**: 
  Finally, we check if we have been able to schedule all courses by comparing the length of the result list with the total number of courses.

This approach runs efficiently with a time complexity of O(V + E), where V is the number of courses and E is the number of prerequisites, allowing it to handle the upper limits specified in the problem statement.

# 269. Alien Dictionary

### Problem Description 
There is a new alien language that uses the English alphabet. However, the order among the letters is unknown to you.

You are given a list of strings `words` from the alien language's dictionary, where the strings in `words` are sorted lexicographically by the rules of this new language.

Return a string of the unique letters in the new alien language sorted in lexicographically increasing order by the new language's rules. If there is no solution, return `""`. If there are multiple solutions, return any of them.

A string `s` is lexicographically smaller than a string `t` if at the first letter where they differ, the letter in `s` comes before the letter in `t` in the alien language. If the first `min(s.length, t.length)` letters are the same, then `s` is smaller if and only if `s.length < t.length`.


Example 1:
Input: words = ["wrt","wrf","er","ett","rftt"]
Output: "wertf"

Example 2:
Input: words = ["z","x"]
Output: "zx"

Example 3:
Input: words = ["z","x","z"]
Output: ""
Explanation: The order is invalid, so return `""`.


Constraints:
`1 <= words.length <= 100`
`1 <= words[i].length <= 100`
`words[i]` consists of only lowercase English letters.

### Solution 
 To solve the problem of determining the order of the letters in an unknown alien language, we can model the problem as a graph. Each letter can be represented as a node, and a directed edge from node A to node B indicates that letter A appears before letter B in the alien language.

### Steps to Solve the Problem:

1. **Build the Graph**:
   - Compare each pair of adjacent words in the list.
   - For each pair, figure out the first character that differs, and establish a directed edge from the first character to the differing character.

2. **Topological Sort**:
   - Since we are constructing a directed graph, we need to perform a topological sort to get the letters in the correct order.
   - This can be done using Kahn's algorithm which uses in-degree counts and a queue for processing nodes.

3. **Handle Edge Cases**:
   - If there's a cycle in the graph, it's impossible to determine a valid order, thus returning an empty string.
   - If we successfully perform the topological sort, we should return the collected letters in the determined order.

### Python Implementation:

Here’s how we can implement the above steps in Python:



```python
from collections import defaultdict, deque

class Solution:
    def alienOrder(self, words: list[str]) -> str:
        # Step 1: Build the graph and compute in-degrees
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        
        # Initialize the in_degree for each unique letter
        for word in words:
            for char in word:
                in_degree[char] = 0

        # Step 2: Determine the order of characters
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            min_length = min(len(w1), len(w2))
            found_diff = False
            
            for j in range(min_length):
                if w1[j] != w2[j]:  # Found the first different character
                    if w2[j] not in graph[w1[j]]:
                        graph[w1[j]].add(w2[j])
                        in_degree[w2[j]] += 1
                    found_diff = True
                    break
            
            # If we reach the end of w1 but w1 is longer than w2, it's invalid
            if not found_diff and len(w1) > len(w2):
                return ""

        # Step 3: Topological Sort using Kahn's algorithm
        # Nodes with no incoming edges (in-degree of 0)
        queue = deque([char for char in in_degree if in_degree[char] == 0])
        result = []
        
        while queue:
            curr = queue.popleft()
            result.append(curr)
            for neighbor in graph[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If the result contains all the letters of the alien language, return it.
        if len(result) == len(in_degree):
            return ''.join(result)

        return ""  # There was a cycle, or not all nodes were processed

# Example usage
solution = Solution()
print(solution.alienOrder(["wrt","wrf","er","ett","rftt"]))  # Output: "wertf" or any valid order
print(solution.alienOrder(["z","x"]))  # Output: "zx"
print(solution.alienOrder(["z","x","z"]))  # Output: ""

```

### Explanation of the Code:
- **Graph Construction**: We use a defaultdict of sets to represent the graph and a defaultdict of integers to maintain the in-degree of each character.
- **Finding the First Difference**: We compare adjacent words to build the graph and update in-degrees accordingly.
- **Topological Sort**: We utilize a queue to manage the characters with in-degree zero. As we process each character, we reduce the in-degrees of its neighbors and add them to the queue when their in-degrees drop to zero.
- Finally, we check if we processed all characters to confirm that a valid order exists.

This solution adheres to the constraints given and runs efficiently within the provided limits.

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
 To solve the problem of finding the longest increasing path in a given `m x n` matrix, we can use Depth-First Search (DFS) in combination with memoization. This approach allows us to explore all possible paths in the matrix while storing the length of the longest path we can achieve starting from each cell, avoiding recomputation and improving efficiency.

### Detailed Explanation

1. **Initialization**: We start by checking if the input matrix is empty. If it is, we return `0`. We then initialize variables to hold the dimensions of the matrix (`m` and `n`), a memoization table (`memo`) to store the computed results for each cell, and a variable to track the maximum path length found.

2. **Depth-First Search**: We define a recursive DFS function that will:
   - Check if we can move to adjacent cells (up, down, left, right) from the current cell.
   - Ensure that we only move to cells with values greater than the current cell's value (to maintain the increasing condition).
   - Utilize the memoization table: if we have already computed the longest path starting from a cell, we can return that value immediately to save time.

3. **Iterate Over Each Cell**: We will call our DFS function for each cell in the matrix to ensure that we have explored all possible starting points.

4. **Return the Result**: Finally, we will return the maximum length found.

### Python Code

Here’s the complete implementation following the LeetCode solution format:



```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        memo = [[-1] * n for _ in range(m)]
        max_length = 0
        
        def dfs(x: int, y: int) -> int:
            if memo[x][y] != -1:  # Return already computed result
                return memo[x][y]
            
            # Directions: down, up, right, left
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            max_path = 1  # Include the current cell
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # Check bounds and increasing condition
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] > matrix[x][y]:
                    max_path = max(max_path, 1 + dfs(nx, ny))
            
            memo[x][y] = max_path  # Memoize the result
            return max_path
        
        for i in range(m):
            for j in range(n):
                max_length = max(max_length, dfs(i, j))  # Explore from each cell
        
        return max_length

```

### How It Works
- The `longestIncreasingPath` method checks if the matrix is empty and initializes our variables.
- The `dfs` function explores the matrix recursively and calculates the longest increasing path starting from each cell.
- For each cell, it checks its four neighbors and computes the maximum path length found, storing results in the `memo` array to avoid duplicate calculations.
- The outer loop calls the `dfs` function for every cell in the matrix, and the maximum path length is updated accordingly.

### Complexity
- **Time Complexity**: O(m * n) since each cell is processed once.
- **Space Complexity**: O(m * n) for the memoization table.

This implementation is efficient for the given problem constraints and follows the required solution format for LeetCode.

# 444. Sequence Reconstruction

### Problem Description 
Check whether the original sequence `org` can be uniquely reconstructed from the sequences in `seqs`. The `org` sequence is a permutation of the integers from 1 to n, with 1 ≤ n ≤ 104. Reconstruction means building a shortest common supersequence of the sequences in `seqs` (i.e., a shortest sequence so that all sequences in `seqs` are subsequences of it). Determine whether there is only one sequence that can be reconstructed from `seqs` and it is the `org` sequence.


Example 1:
Input: org = [1,2,3], seqs = [[1,2],[1,3]]
Output: false
Explanation: [1,2,3] is not the only one sequence that can be reconstructed, because [1,3,2] is also a valid sequence that can be reconstructed.


Example 2:
Input: org = [1,2,3], seqs = [[1,2]]
Output: false
Explanation: The reconstructed sequence can only be [1,2].


Example 3:
Input: org = [1,2,3], seqs = [[1,2],[1,3],[2,3]]
Output: true
Explanation: The sequences [1,2], [1,3], and [2,3] can uniquely reconstruct the original sequence [1,2,3].


Example 4:
Input: org = [4,1,5,2,6,3], seqs = [[5,2,6,3],[4,1,5,2]]
Output: true

Constraints:
`1 <= n <= 10^4`
`org` is a permutation of {1,2,...,n}.

`1 <= segs[i].length <= 10^5`
`seqs[i][j]` fits in a 32-bit signed integer.

UPDATE (2017/1/8):
The seqs parameter had been changed to a list of list of strings (instead of a 2d array of strings). Please reload the code definition to get the latest changes.

### Solution 
 To solve the problem of determining whether the original sequence `org` can be uniquely reconstructed from the sequences in `seqs`, we can approach it by ensuring two key conditions are met:

1. All numbers from 1 to n must appear in `seqs`.
2. The order of numbers in `seqs` must match the order in `org` in such a way that `org` is the only possible reconstruction.

Here's how we can structure the solution:

1. **Check if all numbers from 1 to n are present:** We can do this by adding all numbers from sequences in `seqs` into a set and confirming that it matches the set {1, 2, ..., n}.

2. **Check the order consistency:** We can create an index mapping for `org` for quick look-up. Then, as we traverse through each sequence in `seqs`, we will validate their order. If at any point we find two different numbers that should be adjacent in `org` but appear in the wrong order in `seqs`, then `org` cannot be uniquely reconstructed.

3. **Count the number of valid sequences:** If we find any ambiguity (like multiple valid predecessors for the same number), it means the reconstruction is not unique.

Here's the Python implementation based on the outlined plan:



```python
class Solution:
    def sequenceReconstruction(self, org: List[int], seqs: List[List[int]]) -> bool:
        n = len(org)
        org_set = set(org)
        
        # Step 1: Check if we can create a complete set from seqs
        created_set = set()
        for seq in seqs:
            created_set.update(seq)
        
        # Check if all numbers from 1 to n are covered
        if created_set != org_set:
            return False
        
        # Step 2: Create an index map for org
        index_map = {num: i for i, num in enumerate(org)}
        
        # Step 3: Check the order of appearances in seqs
        # We also need to make sure that we do not create any ambiguity
        prev_index = -1  # There is no previous number at the start
        for seq in seqs:
            for num in seq:
                if num not in index_map:
                    return False  # Number not in org

                current_index = index_map[num]

                # If current_index > prev_index then it's in the correct order
                if current_index < prev_index:
                    return False  # Not correct order, returning False
                
                # This allows us to check for ambiguity
                prev_index = current_index
            
        # Step 4: Count how many times we can go to the next number
        # We need to make sure there is exactly one way to reconstruct the org
        # Reset prev_index for a second pass
        prev_index = -1
        for seq in seqs:
            for num in seq:
                current_index = index_map[num]
                if current_index > prev_index + 1:
                    # There are missing numbers in between which means ambiguity
                    return False
                prev_index = current_index

        return True

```

### Explanation of Code:
- **Lines 1-4:** We start by defining our function and importing the required types (assuming List is imported).
- **Step 1:** We ensure the entirety of `org` is present in `seqs` by comparing sets.
- **Step 2:** We create a mapping from the values in `org` to their indices to facilitate the checking of order.
- **Step 3:** As we iterate through each sequence in `seqs`, we check the order of numbers against their indices in `org`.
- **Step 4:** We confirm that for any number in `org`, all preceding numbers must exist in `seqs`—otherwise it confirms ambiguity in reconstruction.

This code follows the structure suitable for submission to LeetCode and handles the problem's constraints adequately.

# 1203. Sort Items by Groups Respecting Dependencies

### Problem Description 
There are `n` items each belonging to zero or one of `m` groups where `group[i]` is the group that the `i`-th item belongs to and it's equal to `-1` if the `i`-th item belongs to no group. The items and the groups are zero indexed. A group can have no item belonging to it.

Return a sorted list of the items such that:
The items that belong to the same group are next to each other in the sorted list.

There are some relations between these items where `beforeItems[i]` is a list containing all the items that should come before the `i`-th item in the sorted array (to the left of the `i`-th item).

Return any solution if there is more than one solution and return an empty list if there is no solution.


Example 1:
Input: n = 8, m = 2, group = [-1,-1,1,0,0,1,0,-1], beforeItems = [[],[6],[5],[6],[3,6],[],[],[]]
Output: [6,3,4,1,5,2,0,7]

Example 2:
Input: n = 8, m = 2, group = [-1,-1,1,0,0,1,0,-1], beforeItems = [[],[6],[5],[6],[3],[],[4],[]]
Output: []
Explanation: This is the same as example 1 except that 4 needs to be before 6 in the sorted list.


Constraints:
`1 <= m <= n <= 3 * 104`
`group.length == beforeItems.length == n`
`-1 <= group[i] <= m - 1`
`0 <= beforeItems[i].length <= n - 1`
`0 <= beforeItems[i][j] <= n - 1`
`i != beforeItems[i][j]`
`beforeItems[i] `does not contain duplicates elements.

### Solution 
 To solve the problem of sorting the items while respecting group constraints and the "before" dependencies, we can utilize graph theory, specifically the concept of topological sorting in a directed acyclic graph (DAG). Here’s how we can approach the problem step-by-step:

### Problem Breakdown

1. **Group Items**: We need to first categorize the items into their respective groups. Items that belong to the same group should be placed next to each other in the sorted output.

2. **Build Dependency Graph**: For each item `i`, if there are items in `beforeItems[i]`, this indicates that item `j` should come before `i`. We'll construct a directed graph where edges point from each element in `beforeItems[i]` to `i`.

3. **Topological Sort**: We will then need to perform a topological sort on this graph to ensure that the order of items respects the dependencies. This requires using Kahn's algorithm or DFS to find a valid order.

4. **Combine Groups**: Finally, we will sort the items within their groups and concatenate them to produce the final output list.

5. **Check for Cycles**: If we can't perform a topological sort (i.e., if we detect a cycle), we return an empty list.

### Implementation

Here's the code implementing the above logic in Python. This format is suitable for direct use in LeetCode.



```python
from collections import defaultdict, deque
from typing import List

class Solution:
    def sortItems(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        # Step 1: Use an adjustment approach for groups
        next_group_id = max(group) + 1  # Next available group id
        item_group = [0] * n  # A new list for items' group ids
        for i in range(n):
            if group[i] == -1:
                group[i] = next_group_id
                next_group_id += 1
            item_group[i] = group[i]
        
        # Step 2: Create adjacency lists and in-degree count
        group_items = defaultdict(list)
        group_graph = defaultdict(list)
        item_graph = defaultdict(list)
        item_in_degree = [0] * n
        group_in_degree = [0] * next_group_id
        
        # Create item graph and group graph
        for i in range(n):
            for pre in beforeItems[i]:
                item_graph[pre].append(i)
                item_in_degree[i] += 1
                if item_group[pre] != item_group[i]:
                    group_graph[item_group[pre]].append(item_group[i])
                    group_in_degree[item_group[i]] += 1

        # Step 3: Topological sort for items
        def topological_sort(graph, in_degree):
            zero_in_degree = deque()
            for i in range(len(in_degree)):
                if in_degree[i] == 0:
                    zero_in_degree.append(i)
            
            sorted_order = []
            while zero_in_degree:
                node = zero_in_degree.popleft()
                sorted_order.append(node)
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        zero_in_degree.append(neighbor)
                        
            return sorted_order if len(sorted_order) == len(in_degree) else []

        sorted_items = topological_sort(item_graph, item_in_degree)
        sorted_groups = topological_sort(group_graph, group_in_degree)

        if not sorted_items or not sorted_groups:
            return []

        # Step 4: Sort items within each group
        group_to_items = defaultdict(list)
        for item in sorted_items:
            group_id = item_group[item]
            group_to_items[group_id].append(item)

        # Step 5: Prepare the final sorted list
        result = []
        for group_id in sorted_groups:
            result.extend(group_to_items[group_id])
        
        return result

```

### Explanation of the Code

1. **Initialization**: We initialize data structures for maintaining adjacency lists for both items and groups, as well as their in-degrees.

2. **Group Adjustments**: We create a new unique group for items that do not belong to any group (group ID is -1).

3. **Graph Construction**: We populate the graphs based on the input `beforeItems`.

4. **Topological Sorting**: We define a function `topological_sort` that performs Kahn's algorithm to find the sorted order of nodes based on their dependencies.

5. **Final Output**: We collect the items based on the sorted order of groups and return the final result.

Overall, this approach ensures that items are sorted according to their groups and dependencies, and it returns an empty list when no valid sorting is possible due to cycles in the dependencies.

