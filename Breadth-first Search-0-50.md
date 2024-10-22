# 101. Symmetric Tree

### Problem Description 
Given the `root` of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).


Example 1:
Input: root = [1,2,2,3,4,4,3]
Output: true

Example 2:
Input: root = [1,2,2,null,3,null,3]
Output: false

Constraints:
The number of nodes in the tree is in the range `[1, 1000]`.

`-100 <= Node.val <= 100`
Follow up: Could you solve it both recursively and iteratively?

### Solution 
 To solve the problem of checking if a binary tree is symmetric, we can approach it both recursively and iteratively. We'll first implement the recursive method, then the iterative method, and both will follow the structure that LeetCode requires.

### Problem Explanation:
A binary tree is symmetric if the left subtree is a mirror reflection of the right subtree. This means that for every node in the left subtree, there must be a corresponding node in the right subtree with the same value, and their children must also be mirror images.

### Recursive Approach:
In the recursive approach, we define a helper function that takes two nodes and checks if they are mirrors of each other. The conditions to check for mirroring are:
1. Both nodes are `None`: return `True`.
2. Only one of them is `None`: return `False`.
3. The values of the nodes are different: return `False`.
4. Recursively check the children:
   - Check the left child of the first node against the right child of the second node.
   - Check the right child of the first node against the left child of the second node.

### Iterative Approach:
In the iterative approach, we can use a queue to perform a breadth-first traversal of the tree. At each level, we compare the relevant nodes from both sides of the tree.

### LeetCode Solution Code:
Here's the Python code that implements both the recursive and iterative approaches.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    # Recursive approach
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        def isMirror(left: TreeNode, right: TreeNode) -> bool:
            if not left and not right:
                return True
            if not left or not right:
                return False
            return (left.val == right.val) and isMirror(left.right, right.left) and isMirror(left.left, right.right)
        
        return isMirror(root.left, root.right)

    # Iterative approach
    def isSymmetricIterative(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        queue = [(root.left, root.right)]
        
        while queue:
            left, right = queue.pop(0)
            
            if not left and not right:
                continue
            if not left or not right:
                return False
            if left.val != right.val:
                return False
            
            queue.append((left.left, right.right))
            queue.append((left.right, right.left))
        
        return True

```

### Explanation:
1. **TreeNode Class**: This is a standard class for tree nodes with a value, left child, and right child.
2. **isSymmetric Method**:
   - This checks if the tree is symmetric using the recursive `isMirror` helper function. 
   - If the `root` is `None`, it's symmetric (an empty tree).
3. **isMirror Helper Function**:
   - Compares two nodes recursively based on the mirror condition.
4. **isSymmetricIterative Method**:
   - Uses a queue to traverse the tree in an iterative manner to check for symmetry.
   - Pairs of nodes from the left and right subtrees are enqueued for comparison.

### Complexity:
Both approaches run in O(n) time complexity, where n is the number of nodes in the tree, since we need to visit each node. The space complexity is O(h) for the recursive approach due to the call stack, where h is the height of the tree, and O(n) for the iterative approach in the worst case when all nodes are in the queue. 

Feel free to run this code directly in the LeetCode environment, ensuring it meets the input/output format as required.

# 102. Binary Tree Level Order Traversal

### Problem Description 
Given the `root` of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).


Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]

Example 2:
Input: root = [1]
Output: [[1]]

Example 3:
Input: root = []
Output: []

Constraints:
The number of nodes in the tree is in the range `[0, 2000]`.

`-1000 <= Node.val <= 1000`

### Solution 
 To solve the problem of performing a level order traversal on a binary tree, we can use a breadth-first search (BFS) approach with the help of a queue. The idea is to traverse each level of the tree starting from the root and move down each level from left to right. 

Here's a step-by-step breakdown of the solution, followed by the Python code formatted for LeetCode:

### Detailed Explanation

1. **Data Structure**: 
   - We can utilize a queue to keep track of the nodes that we need to visit.
   - A list will be used to store the final result, which will contain values for each level of the tree.

2. **Initialization**:
   - If the `root` is `None`, we can immediately return an empty list since there are no nodes to traverse.
   - Initialize a queue with the root node and prepare our result list.

3. **BFS Traversal**:
   - We will loop until the queue is empty. 
   - For each level, we can determine how many nodes are present in that level (the length of the queue).
   - We will then process each node in that level, adding its value to the current level list and enqueue its children (left and right) for the next level.

4. **Completion**:
   - After processing each level and gathering the results, we will append the current level list to the results and repeat for the next level until the queue is empty.

5. **Return the Result**: After all levels are processed, we will return the results containing the values at each level.

### Python Code



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        from collections import deque
        
        result = []
        queue = deque([root])  # Initialize the queue with the root node

        while queue:
            level_size = len(queue)  # Number of nodes at the current level
            current_level = []  # List to store nodes at the current level
            
            for _ in range(level_size):
                node = queue.popleft()  # Remove the front node from the queue
                current_level.append(node.val)  # Add node's value to current level list
                
                if node.left:
                    queue.append(node.left)  # Add left child to the queue
                if node.right:
                    queue.append(node.right)  # Add right child to the queue

            result.append(current_level)  # Add the current level list to the result

        return result  # Return the final result containing all levels


```

### Explanation of the Code:
- The `TreeNode` class defines the structure of each node in the binary tree.
- The `Solution` class has the `levelOrder` method which implements the BFS traversal.
- We check for the case where `root` is None and return an empty list.
- We use a `deque` from the `collections` module as our queue for efficient popping from the left.
- The outer loop continues as long as there are nodes to visit. The inner loop handles each node at the current level.
- We collect values and enqueue the children before finally storing the result for that level.
- At the end, the list of lists containing level-wise node values is returned.

This solution effectively handles the constraints provided and works within the limits specified in the problem description.

# 103. Binary Tree Zigzag Level Order Traversal

### Problem Description 
Given the `root` of a binary tree, return the zigzag level order traversal of its nodes' values. (i.e., from left to right, then right to left for the next level and alternate between).


Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[20,9],[15,7]]

Example 2:
Input: root = [1]
Output: [[1]]

Example 3:
Input: root = []
Output: []

Constraints:
The number of nodes in the tree is in the range `[0, 2000]`.

`-100 <= Node.val <= 100`

### Solution 
 To solve the problem of zigzag level order traversal of a binary tree, we can implement a breadth-first search (BFS) using a queue. The idea is to traverse each level of the tree while keeping track of whether we should traverse from left to right or right to left.

Here's how we can achieve this step by step:

1. **Use a Queue for BFS**: We use a queue to facilitate level order traversal of the binary tree. Python's `collections.deque` provides an efficient way to implement a queue.

2. **Determine Level Direction**: We maintain a boolean variable that indicates whether the current level should be traversed in left-to-right or right-to-left order.

3. **Collect Results**: We store the values of nodes at each level in the results list. Depending on the current direction, we either append the node values normally or reverse them before appending.

4. **Switch Level Direction**: At the end of each level, we toggle the direction for the next level.

5. **Handle Edge Cases**: If the tree is empty (i.e., the root is `None`), we should return an empty list.

Here's the complete Python code implementing the above logic:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from collections import deque

class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> list[list[int]]:
        result = []
        if not root:  # If the root is None, return empty result
            return result

        queue = deque([root])  # Initialize the queue with the root
        left_to_right = True  # Direction flag

        while queue:
            level_size = len(queue)
            current_level = deque()  # Use deque to append values based on the direction

            for _ in range(level_size):
                node = queue.popleft()  # Get the current node from the queue
                # Append to current level based on the direction
                if left_to_right:
                    current_level.append(node.val)  # Left to right
                else:
                    current_level.appendleft(node.val)  # Right to left

                # Add the child nodes to the queue for the next level
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(list(current_level))  # Append the current level to results
            left_to_right = not left_to_right  # Toggle the direction

        return result

```

### Detailed Explanation:
- **TreeNode Class**: This is a simple class definition for a binary tree node, containing a value and pointers to left and right children.
  
- **Initialization**: We initialize an empty list `result` to store the final zigzag traversal. If the `root` is `None`, we immediately return an empty list.

- **Queue Initialization**: We use a deque to hold the nodes at the current level that we will process. We start with the root node in the queue.

- **BFS Loop**: While there are nodes in the queue, we calculate the size of the current level (`level_size`).

- **Current Level Handling**: For each node in the current level, we pop it from the queue and determine where to place its value (either at the end or the front of `current_level` based on `left_to_right`).

- **Child Nodes**: We enqueue the left and right children of the current node, if they exist.

- **Finalizing Level**: After processing all nodes at the current level, we convert `current_level` deque to a list and append it to `result`.

- **Direction Toggle**: Finally, we toggle `left_to_right` for the next level.

This code efficiently provides the zigzag level order traversal for any binary tree as required by the problem statement.

# 107. Binary Tree Level Order Traversal II

### Problem Description 
Given the `root` of a binary tree, return the bottom-up level order traversal of its nodes' values. (i.e., from left to right, level by level from leaf to root).


Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: [[15,7],[9,20],[3]]

Example 2:
Input: root = [1]
Output: [[1]]

Example 3:
Input: root = []
Output: []

Constraints:
The number of nodes in the tree is in the range `[0, 2000]`.

`-1000 <= Node.val <= 1000`

### Solution 
 To solve the problem of obtaining the bottom-up level order traversal of a binary tree, we can utilize a breadth-first search (BFS) approach. Here’s a detailed explanation of how this can be accomplished, followed by the complete code structured to run directly on LeetCode.

### Approach

1. **Understanding Level Order Traversal**: Normally, when we perform a level order traversal, we visit nodes level by level from top to bottom. In this case, we need to reverse that order.

2. **Using a Queue**: We will use a queue to facilitate the BFS traversal. The queue will help us explore each node and subsequently their children level by level.

3. **Storing Levels**: Instead of directly storing levels as we visit them, we will collect them in a list and reverse that list at the end of our traversal.

4. **Edge Case**: If the tree is empty (`root` is `None`), we should return an empty list.

### Code Implementation

Here’s the complete code using the aforementioned approach:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from collections import deque

class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        
        # We will use a queue to perform the level order traversal
        queue = deque([root])
        result = []
        
        while queue:
            level_size = len(queue)
            current_level = []
            
            for _ in range(level_size):
                node = queue.popleft()  # Get the front of the queue
                
                # Store the value of the current node
                current_level.append(node.val)
                
                # Add the left and right children to the queue if they exist
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            # Insert the current level at the beginning (reverse order)
            result.insert(0, current_level)
        
        return result

```

### Explanation of the Code

1. **TreeNode Class**: A class definition for a binary tree node is provided. Each node contains a `val`, `left`, and `right`.

2. **Solution Class and Method**: The `Solution` class contains a single method named `levelOrderBottom`. This method takes the `root` of a binary tree as a parameter.

3. **Checking for Empty Tree**: The first `if not root:` statement checks if the tree is empty. If it is, we return an empty list.

4. **Queue Initialization**: We initialize a queue using `deque` (from the `collections` module) and add the root node to it.

5. **Level Order Traversal Loop**:
   - We run a loop while our queue is not empty.
   - For each level of the tree, we determine its size (`level_size`), create a list `current_level`, and iterate through the nodes at that level.
   - Each node's value is appended to `current_level`, and its children (if they exist) are added to the queue for the next iteration.
   - After processing nodes at the current level, we insert `current_level` at the start of the result list to ensure bottom-up order.

6. **Return Result**: Finally, we return the collected `result`, which now contains the desired bottom-up level order traversal.

This implementation is efficient and adheres to the constraints provided in the problem statement, effectively solving the task using BFS.

# 111. Minimum Depth of Binary Tree

### Problem Description 
Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

Note: A leaf is a node with no children.


Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: 2

Example 2:
Input: root = [2,null,3,null,4,null,5,null,6]
Output: 5

Constraints:
The number of nodes in the tree is in the range `[0, 105]`.

`-1000 <= Node.val <= 1000`

### Solution 
 To solve the problem of finding the minimum depth of a binary tree, we can utilize a depth-first search (DFS) or breadth-first search (BFS). In this solution, I will employ the BFS approach, as it is particularly well-suited for finding the shortest path in unweighted trees.

### Explanation:

1. **Minimum Depth Definition**: The minimum depth of a binary tree is the length of the shortest path from the root node down to the nearest leaf node. A leaf is defined as a node that has no children (both left and right children are null).

2. **Breadth-First Search (BFS)**:
   - We'll use a queue to traverse the tree level by level.
   - For each node, we'll check if it is a leaf node (both left and right children are null).
   - If we encounter a leaf node, we return the current depth, as it guarantees that this is the shortest path to a leaf.
   - If the queue is exhausted without finding a leaf, it means our tree is empty, and we should return 0.

### Code Implementation:

Here’s how you can implement this in Python according to the LeetCode solution format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from collections import deque

class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        queue = deque([(root, 1)])  # Store tuple of node and its depth
        
        while queue:
            node, depth = queue.popleft()
            
            # Check if it's a leaf node
            if not node.left and not node.right:
                return depth
            
            # If left child exists, add it to the queue
            if node.left:
                queue.append((node.left, depth + 1))
            # If right child exists, add it to the queue
            if node.right:
                queue.append((node.right, depth + 1))
        
        return 0  # This return is mainly for safety, should not be reached

```

### How the Code Works:

1. **TreeNode Class**: This class defines the structure of each node in the binary tree.

2. **Solution Class**: Contains the `minDepth` method that takes the root of the binary tree as input.

3. **Queue Initialization**: A deque is used for efficient popping from the front of the queue. Each entry in the queue is a tuple containing the current node and its depth.

4. **Loop Through the Queue**: 
   - We pop the first node from the queue and check if it is a leaf. If it is, return the current depth.
   - If the node has children, we push its non-null children onto the queue with an incremented depth.

5. **Return Value**: The method returns the minimum depth once a leaf node is found. If the input tree is empty, it immediately returns 0.

### Test Cases:

- To test this implementation, create instances of `TreeNode` to form a binary tree and create an instance of `Solution` to call the `minDepth` method.

This BFS approach ensures we find the shortest path to a leaf node effectively.

# 116. Populating Next Right Pointers in Each Node

### Problem Description 
You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to `NULL`.

Initially, all next pointers are set to `NULL`.

Follow up:
You may only use constant extra space.

Recursive approach is fine, you may assume implicit stack space does not count as extra space for this problem.


Example 1:
Input: root = [1,2,3,4,5,6,7]
Output: [1,#,2,3,#,4,5,6,7,#]
Explanation: Given the above perfect binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.


Constraints:
The number of nodes in the given tree is less than `4096`.

`-1000 <= node.val <= 1000`

### Solution 
 To solve the problem of populating the `next` pointers in a perfect binary tree, we can use a recursive approach that leverages the structure of the tree itself. The key is to connect the nodes on the same level without using extra space, so we will utilize the properties of perfect binary trees.

### Explanation

In a perfect binary tree:
1. Each node has two children.
2. All the leaves are at the same level.

Given a node `root`, we can easily find its left and right children (`root.left` and `root.right`). For each node, we can connect its left child to its right child, and then connect the right child of the current node to the left child of the next node (if it exists).

Here's how we can systematically achieve this:

1. Start from the root of the tree.
2. For each node, connect its left child to its right child.
3. Next, connect the right child of the current node to the left child of the next node, which can be found by accessing `next` pointer of the current node.
4. Recursively process the left and right children.

### Implementation

The following Python code implements the above logic. The `Node` class is defined as per the problem statement, and the function `connect` populates the `next` pointers.

Here's the complete implementation:



```python
# Definition for a Node.
class Node:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return root
        
        # Connect left and right children
        if root.left and root.right:
            root.left.next = root.right
            
            # Connect right child to the left child of the next node (if exists)
            if root.next:
                root.right.next = root.next.left
            
            # Recursive calls for the next level
            self.connect(root.left)
            self.connect(root.right)
        
        return root

```

### Key Points:
1. The function `connect` takes the root node of the binary tree as input and modifies it in place.
2. This function uses recursion to traverse the tree, so we make use of the implicit call stack as our extra space.
3. The conditions inside the function ensure that we only attempt to connect nodes when both left and right children are available, making the process streamlined and efficient.

### Complexity:
- **Time Complexity**: O(N) where N is the number of nodes in the tree since we visit each node exactly once.
- **Space Complexity**: O(1) in terms of extra space, as the recursion stack does not count against the constant space requirement due to the problem's constraints.

This implementation should work correctly when submitted on LeetCode, adhering to the problem's requirements and constraints.

# 126. Word Ladder II

### Problem Description 
A transformation sequence from word `beginWord` to word `endWord` using a dictionary `wordList` is a sequence of words `beginWord -> s1 -> s2 -> ... -> sk` such that:
Every adjacent pair of words differs by a single letter.

Every `si` for `1 <= i <= k` is in `wordList`. Note that `beginWord` does not need to be in `wordList`.

`sk == endWord`
Given two words, `beginWord` and `endWord`, and a dictionary `wordList`, return all the shortest transformation sequences from `beginWord` to `endWord`, or an empty list if no such sequence exists. Each sequence should be returned as a list of the words `[beginWord, s1, s2, ..., sk]`.


Example 1:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
Explanation: There are 2 shortest transformation sequences:
"hit" -> "hot" -> "dot" -> "dog" -> "cog"
"hit" -> "hot" -> "lot" -> "log" -> "cog"

Example 2:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
Output: []
Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.


Constraints:
`1 <= beginWord.length <= 10`
`endWord.length == beginWord.length`
`1 <= wordList.length <= 5000`
`wordList[i].length == beginWord.length`
`beginWord`, `endWord`, and `wordList[i]` consist of lowercase English letters.

`beginWord != endWord`
All the words in `wordList` are unique.

### Solution 
 To solve the problem of finding all the shortest transformation sequences from `beginWord` to `endWord` using a dictionary `wordList`, we can break down the solution into the following key steps:

1. **Breadth-First Search (BFS)**: We utilize BFS to find the shortest path in terms of word transformations, as it explores all possibilities level by level. This will help determine the minimum length of the transformation sequence.

2. **Graph Representation**: Each word can be considered as a node in a graph, where an edge exists between two nodes if they differ by exactly one letter. We will create this connectivity during the BFS.

3. **Tracking Paths**: We need to keep track of all possible paths that lead to each word during the BFS. For this, we will maintain a parent dictionary that records which words could lead to the current word.

4. **Backtrack to Find All Paths**: After BFS is complete, we will have a parent mapping of how to reach the `endWord`. Using this mapping, we can backtrack from `endWord` to reconstruct all possible shortest paths back to `beginWord`.

5. **Return the Result**: Finally, we format the output as required.

Here's the complete implementation in Python:



```python
from collections import deque, defaultdict

class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: list[str]) -> list[list[str]]:
        wordSet = set(wordList)
        if endWord not in wordSet:
            return []
        
        # BFS setup
        queue = deque([beginWord])
        visited = {beginWord}
        parent_map = defaultdict(list)
        found = False
        
        # Perform BFS
        while queue and not found:
            level_visited = set()  # To keep track of words visited in this level
            for _ in range(len(queue)):
                current_word = queue.popleft()
                for i in range(len(current_word)):
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        if c != current_word[i]:
                            next_word = current_word[:i] + c + current_word[i + 1:]
                            if next_word in wordSet:
                                if next_word == endWord:
                                    found = True
                                if next_word not in visited:
                                    level_visited.add(next_word)
                                    queue.append(next_word)
                                parent_map[next_word].append(current_word)
            visited.update(level_visited)

        # Backtrack to find all paths to endWord
        def backtrack(word):
            if word == beginWord:
                return [[beginWord]]
            return [[beginWord] + path for parent in parent_map[word] for path in backtrack(parent)]

        # If we found the endWord, backtrack to find all paths
        if found:
            return backtrack(endWord)
        else:
            return []

# Example Usage
solution = Solution()
print(solution.findLadders("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]))

```

### Detailed Explanation:

1. **Initialization**:
   - We create a set from `wordList` for quick lookups. If `endWord` is not in `wordSet`, we return an empty list right away.
   - We initialize a BFS queue starting with `beginWord`, a `visited` set to track visited words, and a `parent_map` to keep track of the possible predecessors for constructing paths.

2. **BFS Process**:
   - While the queue is not empty and we haven't found `endWord`, we iterate through each word in the queue.
   - For each character in the current word, we replace it with each letter from 'a' to 'z' to generate all potential one-letter transformations.
   - If a transformation (next_word) is in `wordSet` and hasn't been visited yet, we add it to the queue for the next BFS level and record `current_word` as a potential predecessor.

3. **Backtracking**:
   - The `backtrack` function is recursive; it constructs all paths from `endWord` back to `beginWord` using the `parent_map`.
   - If we reach `beginWord`, we return a list containing just `beginWord`. Otherwise, we build paths by concatenating the current word with paths leading from its predecessors.

4. **Output**:
   - Finally, if `endWord` was found, we get and return all possible paths; otherwise, we return an empty list.

This solution follows the BFS technique, ensuring that we find the shortest paths efficiently, while the backtrack function allows us to explore all paths stemming from the shortest transformation sequences.

# 127. Word Ladder

### Problem Description 
A transformation sequence from word `beginWord` to word `endWord` using a dictionary `wordList` is a sequence of words `beginWord -> s1 -> s2 -> ... -> sk` such that:
Every adjacent pair of words differs by a single letter.

Every `si` for `1 <= i <= k` is in `wordList`. Note that `beginWord` does not need to be in `wordList`.

`sk == endWord`
Given two words, `beginWord` and `endWord`, and a dictionary `wordList`, return the number of words in the shortest transformation sequence from `beginWord` to `endWord`, or `0` if no such sequence exists.


Example 1:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.


Example 2:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
Output: 0
Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.


Constraints:
`1 <= beginWord.length <= 10`
`endWord.length == beginWord.length`
`1 <= wordList.length <= 5000`
`wordList[i].length == beginWord.length`
`beginWord`, `endWord`, and `wordList[i]` consist of lowercase English letters.

`beginWord != endWord`
All the words in `wordList` are unique.

### Solution 
 To solve the problem of finding the number of words in the shortest transformation sequence from `beginWord` to `endWord`, we can use the Breadth-First Search (BFS) algorithm. This approach is well-suited for problems requiring the shortest path in an unweighted graph, where each node represents a word, and edges exist between nodes (words) that are one letter apart.

Here's a step-by-step explanation of how the algorithm works:

1. **Input Validation**: First, we need to check if the `endWord` exists in the `wordList`. If it does not, we immediately return 0, as there's no possible transformation sequence.

2. **BFS Initialization**: We initialize a queue for BFS that starts with a tuple containing the `beginWord` and the depth (or length) of the path, initialized to 1 (since the path includes the `beginWord` itself). We also use a set to keep track of visited words to prevent cycles.

3. **Word Transformation Generation**: For each word, we can generate valid transformations by changing one letter at a time. For instance, changing 'h' in "hit" can produce "ait", "bit", "cit", etc. We can approach this systematically by iterating through each character position and replacing it with all letters from 'a' to 'z'.

4. **BFS Exploration**: In each step of the BFS:
   - Dequeue the front item.
   - For each valid transformation generated, check if it matches the `endWord`. If it does, return the current path length + 1.
   - If it’s in the `wordList` and hasn't been visited yet, append it to the queue and mark it as visited.

5. **Termination**: If we exhaust the queue without finding the `endWord`, it means no transformation sequence exists, and we return 0.

Here is the implementation of the solution in Python:



```python
from collections import deque

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # Convert wordList to a set for O(1) lookups
        wordSet = set(wordList)
        
        # If endWord not in wordList, return 0
        if endWord not in wordSet:
            return 0
        
        # Initialize the BFS queue
        queue = deque([(beginWord, 1)])  # (current_word, current_length)
        visited = set()  # Set to track visited words
        visited.add(beginWord)
        
        # Start BFS
        while queue:
            current_word, current_length = queue.popleft()
            
            # Check all possible one-letter transformations
            for i in range(len(current_word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    # Generate a new word by changing one letter
                    new_word = current_word[:i] + c + current_word[i+1:]
                    
                    # Check if new_word is endWord
                    if new_word == endWord:
                        return current_length + 1
                    
                    # If new_word is in wordSet and not visited yet
                    if new_word in wordSet and new_word not in visited:
                        visited.add(new_word)  # Mark as visited
                        queue.append((new_word, current_length + 1))  # Enqueue new word with updated length
        
        # If we exhaust the queue, there is no transformation sequence
        return 0

```

### Explanation of Key Parts:
- **Queue and Visited Set**: We maintain a queue for BFS and a set for tracking visited words to avoid processing the same word multiple times.
- **Transformation Generation**: For each letter in the word, we replace it with every letter from 'a' to 'z', generating all possible transformations.
- **Efficiency**: The BFS guarantees that we're exploring the shortest paths first, ensuring that the first time we reach the `endWord`, it's via the shortest route.

This solution is efficient enough given the constraints and adheres to the problem requirements. It should work seamlessly when submitted to the LeetCode platform.

# 130. Surrounded Regions

### Problem Description 
Given an `m x n` matrix `board` containing `'X'` and `'O'`, capture all regions surrounded by `'X'`.

A region is captured by flipping all `'O'`s into `'X'`s in that surrounded region.


Example 1:
Input: board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
Output: [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
Explanation: Surrounded regions should not be on the border, which means that any 'O' on the border of the board are not flipped to 'X'. Any 'O' that is not on the border and it is not connected to an 'O' on the border will be flipped to 'X'. Two cells are connected if they are adjacent cells connected horizontally or vertically.


Example 2:
Input: board = [["X"]]
Output: [["X"]]

Constraints:
`m == board.length`
`n == board[i].length`
`1 <= m, n <= 200`
`board[i][j]` is `'X'` or `'O'`.

### Solution 
 To solve the problem of capturing all regions surrounded by 'X' in a given `m x n` matrix `board`, we can use a Depth-First Search (DFS) or Breadth-First Search (BFS) approach. The idea is to identify the 'O's that are not surrounded by 'X's by processing the 'O's on the borders, and marking all connected 'O's as not surrounded. Anywhere that remains 'O' after this process is considered surrounded and should be flipped to 'X'.

Here’s a step-by-step explanation of the algorithm:

1. **Identify Border 'O's**: The first step is to traverse the edges (borders) of the matrix. If we find an 'O', we need to mark this as a non-capturable 'O' and recursively mark all connected 'O's to this one using DFS or BFS.

2. **DFS/BFS Traversal**: From any 'O' found on the border, we will use DFS or BFS to visit all connected 'O's (up, down, left, right). We'll mark these cells as 'T' (for Temporary) to signify they should not be flipped to 'X'.

3. **Flip Remaining 'O's**: After marking the 'T's, traverse the board again. Change all 'O's (that are not marked) to 'X's as they are surrounded. Convert all 'T's back to 'O's, as they are not surrounded.

4. **Edge Cases**: Handle cases like a single cell matrix, or matrices entirely filled with 'X's or 'O's.

Here’s the Python code structured according to LeetCode's submission format:



```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board or not board[0]:
            return
        
        rows, cols = len(board), len(board[0])
        
        def dfs(r, c):
            # Mark the current cell as 'T'
            board[r][c] = 'T'
            # Directions for moving up, down, left, right
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                # Continue the DFS if within bounds and 'O'
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == 'O':
                    dfs(nr, nc)
        
        # Step 1: Mark all 'O's connected to the border
        for i in range(rows):
            if board[i][0] == 'O':
                dfs(i, 0)  # Left border
            if board[i][cols - 1] == 'O':
                dfs(i, cols - 1)  # Right border
        
        for j in range(cols):
            if board[0][j] == 'O':
                dfs(0, j)  # Top border
            if board[rows - 1][j] == 'O':
                dfs(rows - 1, j)  # Bottom border

        # Step 2: Flip 'O's to 'X's and 'T's back to 'O's
        for i in range(rows):
            for j in range(cols):
                if board[i][j] == 'O':
                    board[i][j] = 'X'  # Captured region
                elif board[i][j] == 'T':
                    board[i][j] = 'O'  # Non-captured region

# Example usage:
# sol = Solution()
# board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
# sol.solve(board)
# print(board)  # Output: [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]

```

### Explanation of the Code:
- We define a DFS function to mark all connected 'O's from any border 'O'.
- We scan the borders of the board to start the marking process.
- Finally, we revise the board to flip the 'O's to 'X's where appropriate and restore the marked 'T's back to 'O's.
- The implementation modifies the board in place as specified in the problem. 

By running the code as described, it adheres to the LeetCode problem statement and effectively solves the requirements of the challenge.

# 133. Clone Graph

### Problem Description 
Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a val (`int`) and a list (`List[Node]`) of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}
Test case format:
For simplicity sake, each node's value is the same as the node's index (1-indexed). For example, the first node with `val = 1`, the second node with `val = 2`, and so on. The graph is represented in the test case using an adjacency list.

Adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.

The given node will always be the first node with `val = 1`. You must return the copy of the given node as a reference to the cloned graph.


Example 1:
Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
Explanation: There are 4 nodes in the graph.

1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).

2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).

3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).

4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).


Example 2:
Input: adjList = [[]]
Output: [[]]
Explanation: Note that the input contains one empty list. The graph consists of only one node with val = 1 and it does not have any neighbors.


Example 3:
Input: adjList = []
Output: []
Explanation: This an empty graph, it does not have any nodes.


Example 4:
Input: adjList = [[2],[1]]
Output: [[2],[1]]

Constraints:
`1 <= Node.val <= 100`
`Node.val` is unique for each node.

Number of Nodes will not exceed 100.

There is no repeated edges and no self-loops in the graph.

The Graph is connected and all nodes can be visited starting from the given node.

### Solution 
 To solve the problem of cloning a connected undirected graph represented using nodes with a value and a list of neighbors, we'll follow a depth-first search (DFS) approach. This will allow us to create a deep copy of the graph efficiently while ensuring all connections between nodes are preserved.

### Key Components
- **Node Class**: The graph consists of nodes which have a value and a list of neighbors.
- **Deep Copying**: We need to create a new instance of each node while replicating the connections (neighbors).
- **Visited Tracking**: To prevent processing the same node multiple times and to keep track of already copied nodes, a dictionary can be used.

### Algorithm Steps
1. **Base Condition**: If the input node is `None`, return `None`.
2. **DFS Function**:
   - If the node has already been visited, return the corresponding cloned node to avoid duplicates.
   - Create a new node with the same value as the original node.
   - Mark it as visited.
   - Recursively call the DFS for each neighbor and collect the results into the new node's neighbors list.
3. **Initiate DFS** from the starting node and return the cloned node.

### Implementation

Here’s how you can implement this in Python, formatted according to LeetCode's solution style:



```python
# Definition for a Node.
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return None
        
        visited = {}
        
        def dfs(node):
            if node in visited:
                return visited[node]
            
            # Clone the node
            clone_node = Node(node.val)
            visited[node] = clone_node
            
            # Recursively clone neighbors
            for neighbor in node.neighbors:
                clone_node.neighbors.append(dfs(neighbor))
            
            return clone_node
        
        return dfs(node)

```

### Explanation of the Code
1. **Node Class**: This defines `Node` with a `val` for its value and a `neighbors` list for its connections.
2. **Solution Class**: Contains the `cloneGraph` method which is called with the original graph's starting node.
3. **DFS Function**:
   - Checks if the node has already been visited. If so, it returns the corresponding cloned node to prevent cycles.
   - Creates a new `Node` for each unique node in the original graph.
   - For each neighbor of the current node, it invokes `dfs` recursively and appends the result to the cloned node's neighbors.
4. **Return Value**: Finally, the cloned graph's starting node is returned.

### Test Cases
This implementation will work for the provided sample test cases. You can test it with LeetCode's platform by defining the nodes and their connections as per the adjacency list format.

### Complexity
- **Time Complexity**: O(V + E), where V is the number of nodes (vertices) and E is the number of edges since each node and edge will be visited once.
- **Space Complexity**: O(V) due to the recursive stack and the `visited` dictionary for tracking the cloned nodes. 

This efficient approach ensures that we can handle the cloning of graphs with a maximum of 100 nodes while keeping the implementation clear and concise.

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
 To tackle the problem of obtaining the right side view of a binary tree, we can utilize a breadth-first search (BFS) strategy. The idea is to traverse the tree level by level and collect the last node of each level, which represents the rightmost node visible from that perspective.

Here's how we can implement this in Python, following the LeetCode solution format to ensure it's suitable for direct execution on their platform.

### Python Code



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from collections import deque

class Solution:
    def rightSideView(self, root: TreeNode) -> list[int]:
        if not root:
            return []

        # Initialize a queue for BFS
        queue = deque([root])
        right_view = []

        while queue:
            # Determine the number of nodes at the current level
            level_length = len(queue)
            # Iterate through the current level
            for i in range(level_length):
                node = queue.popleft()
                # If it's the rightmost element of the level, add it to the result
                if i == level_length - 1:
                    right_view.append(node.val)

                # Add the left and right children to the queue for next level
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

        return right_view

```

### Detailed Explanation

1. **TreeNode Definition**: We define a `TreeNode` class, which represents each node of the binary tree. It has a value (`val`), a left child (`left`), and a right child (`right`).

2. **Solution Class**: We create a `Solution` class that contains the `rightSideView` method, which takes the root of the binary tree as input.

3. **Check for Empty Tree**: We first check if the `root` is `None`. If it is, we return an empty list since there are no nodes to view.

4. **BFS Initialization**: A queue (using `collections.deque` for efficiency) is initialized with the root node. We also create a list, `right_view`, to store the values of the nodes that we can see from the right side.

5. **Level Traversal Using BFS**:
   - We enter a loop that continues until the queue is empty.
   - For each level of the tree, we retrieve the current length of the queue to determine how many nodes are on that level.
   - We traverse through each node at that level:
     - We pop nodes from the left of the queue.
     - We check if the current node is the last node at this level (i.e., `i == level_length - 1`). If it is, we append its value to the `right_view`.
     - We add the left child of the node to the queue if it exists, followed by the right child.

6. **Return the Result**: After traversing all levels of the binary tree, we return the `right_view`, which contains the values of the nodes visible from the right side.

### Complexity Analysis
- **Time Complexity**: O(N), where N is the number of nodes in the tree, since we visit each node exactly once.
- **Space Complexity**: O(W), where W is the maximum width of the tree, which occurs at the last level of the tree.

This method is efficient and directly adheres to the requirements of providing the right side view of a binary tree. You can use this code in your LeetCode submissions, and it should work seamlessly for the given problem statement.

# 200. Number of Islands

### Problem Description 
Given an `m x n` 2D binary grid `grid` which represents a map of `'1'`s (land) and `'0'`s (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.


Example 1:
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1

Example 2:
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3

Constraints:
`m == grid.length`
`n == grid[i].length`
`1 <= m, n <= 300`
`grid[i][j]` is `'0'` or `'1'`.

### Solution 
 To solve the problem of counting the number of islands in a 2D binary grid, we can use a Depth-First Search (DFS) approach. An island is defined as a group of adjacent (horizontally or vertically) '1's (land) that are surrounded by '0's (water). We'll traverse the grid, and each time we find a '1', we'll initiate a DFS to mark all the connected lands as visited (by turning them into '0's). This way, we can count how many distinct islands we have encountered.

Here is the detailed solution, structured in the format expected for a LeetCode submission:



```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        
        # Initialize variables
        num_islands = 0
        rows = len(grid)
        cols = len(grid[0])
        
        def dfs(r: int, c: int) -> None:
            # Check for out-of-bounds or water
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
                return
            
            # Mark the land as visited by turning it into water
            grid[r][c] = '0'
            
            # Explore all four directions (up, down, left, right)
            dfs(r - 1, c)  # Up
            dfs(r + 1, c)  # Down
            dfs(r, c - 1)  # Left
            dfs(r, c + 1)  # Right
        
        # Go through each cell in the grid
        for r in range(rows):
            for c in range(cols):
                # If a land cell is found, we've found an island
                if grid[r][c] == '1':
                    num_islands += 1  # Increase island count
                    dfs(r, c)  # Run DFS to mark the entire island as visited
        
        return num_islands

```

### Explanation of the Code:

1. **Grid Check**: First, we check if the grid is empty. If it is, we return 0, since there are no islands.

2. **Variable Initialization**: 
   - We initialize `num_islands` to keep track of the number of islands found.
   - We also determine the number of rows and columns in the grid.

3. **Depth-First Search (DFS) Function**:
   - We define a helper function `dfs(r, c)`, which marks all connected '1's as visited (by setting them to '0').
   - The function checks boundary conditions to avoid going out of grid bounds or revisiting water cells.

4. **Main Loop**:
   - We loop through each cell in the grid.
   - When we encounter a '1', it indicates a new island. We increment the `num_islands` counter and call the `dfs()` function to mark the entire island.

5. **Return Result**: Finally, we return the count of islands found.

### Complexity Analysis:
- **Time Complexity**: O(m * n), where m is the number of rows and n is the number of columns in the grid. Each cell is visited at most once.
- **Space Complexity**: O(m * n), in the worst case if the entire grid is land (all '1's), as the recursion stack could go as deep as the grid.

This code can be directly executed on the LeetCode platform, where `List` is assumed to be imported from `typing`. The function can handle grids of lengths as described in the constraints.

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
 To solve the problem of determining whether all courses can be finished given a list of prerequisites, we can model the problem using graph theory. Specifically, we can represent courses as nodes and prerequisites as directed edges. The task then comes down to detecting cycles in the directed graph.

If there is a cycle, it means that there are courses that depend on each other in a circular manner, making it impossible to finish all courses. Conversely, if we can traverse the graph without encountering cycles, it is possible to finish all courses.

We'll use Kahn's algorithm for topological sorting, which utilizes an in-degree array to track the number of prerequisites (incoming edges) for each course. The algorithm proceeds as follows:

1. Create an adjacency list of courses based on the prerequisites.
2. Create an in-degree array to count the prerequisites for each course.
3. Use a queue to perform the topological sorting, starting from courses with no prerequisites (in-degree of 0).
4. Keep track of the number of courses that have been processed.
5. If the count of processed courses equals `numCourses`, return `true`; otherwise, return `false`.

Here is the Python code that implements the above logic:



```python
from collections import deque, defaultdict

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # Step 1: Create adjacency list and in-degree count
        adj_list = defaultdict(list)
        in_degree = [0] * numCourses
        
        # Build the graph
        for ai, bi in prerequisites:
            adj_list[bi].append(ai)  # bi -> ai indicates bi must be taken before ai
            in_degree[ai] += 1       # Increase in-degree count for ai
        
        # Step 2: Initialize the queue with courses that have no prerequisites
        queue = deque()
        for course in range(numCourses):
            if in_degree[course] == 0:
                queue.append(course)
        
        # Step 3: Process the courses
        processed_courses = 0
        while queue:
            current_course = queue.popleft()
            processed_courses += 1
            
            # Decrease the in-degree of the adjacent courses
            for neighbor in adj_list[current_course]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Step 4: Check if all courses were processed
        return processed_courses == numCourses

```

### Explanation of the Code

1. **Data Structures**:
   - `adj_list`: A dictionary mapping each course to a list of courses that depend on it.
   - `in_degree`: A list that keeps track of how many prerequisites each course has.

2. **Building the Graph**:
   - We loop through the `prerequisites` list to populate the `adj_list` and `in_degree`. For each prerequisite pair `[ai, bi]`, we note that `bi` must be completed before `ai`.

3. **Queue Initialization**:
   - Courses with zero prerequisites are added to a queue to start the processing.

4. **Topological Sorting**:
   - We process each course in the queue, incrementing the count of `processed_courses`.
   - For each processed course, we reduce the in-degree of its neighbors. If any neighbor's in-degree drops to zero, we add it to the queue.

5. **Final Check**:
   - After processing, we check if the number of processed courses equals `numCourses`. If it does, all courses can be finished; otherwise, they cannot.

### Edge Cases
- If there are no prerequisites (`prerequisites.length` is 0), the function should return `true` directly since all courses can be finished.

This solution has a time complexity of O(V + E), where V is the number of courses and E is the number of prerequisites, which is efficient given the problem constraints.

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
 To solve the problem of finding a valid ordering of courses, we can model the courses and prerequisites as a directed graph and then use topological sorting to find a valid order for the courses. Topological sorting is suitable for Directed Acyclic Graphs (DAG), where we can provide an ordering of vertices such that for every directed edge from vertex `u` to vertex `v`, `u` comes before `v` in the ordering.

Here's how to approach the solution step-by-step:

1. **Graph Representation**: We'll represent the courses schedule as a directed graph using an adjacency list. Each course will be represented as a node, and a directed edge from course `b` to course `a` means you must complete course `b` before course `a`.

2. **In-Degree Calculation**: We need to maintain an array to count the number of prerequisites (in-degree) for each course. This will help us identify which courses can be taken without any prerequisites.

3. **Queue Initialization**: Using a queue, we will start with all courses that have zero in-degree (no prerequisites). As we process each course, we will decrease the in-degree of its neighboring courses (those that depend on it).

4. **Topological Sort**: We will perform a BFS-like process where we continually process courses from the queue, adding them to our result as they can be completed, and enqueue any new courses that reach zero in-degree.

5. **Check for Cycles**: If the total courses processed is less than `numCourses`, it indicates that there is a cycle in the graph, and we cannot complete all courses, hence we should return an empty list.

Here is a Python implementation of the above approach:



```python
from collections import deque, defaultdict
from typing import List

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # Step 1: Create an adjacency list and in-degree count
        graph = defaultdict(list)
        in_degree = [0] * numCourses
        
        # Step 2: Build the graph and in-degree count
        for a, b in prerequisites:
            graph[b].append(a)  # b -> a means b is a prerequisite of a
            in_degree[a] += 1   # Increase in-degree for course a

        # Step 3: Initialize the queue with courses that have no prerequisites
        queue = deque()
        for i in range(numCourses):
            if in_degree[i] == 0:
                queue.append(i)

        # Step 4: Process the graph
        order = []
        while queue:
            # Take the course from the front of the queue
            course = queue.popleft()
            order.append(course)  # Add it to the order
            
            # Decrease the in-degree of its neighbors
            for neighbor in graph[course]:
                in_degree[neighbor] -= 1
                # If in-degree becomes zero, add it to the queue
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Step 5: Check if we processed all courses
        if len(order) == numCourses:
            return order  # Valid order found
        else:
            return []  # Cycle detected; no valid order

```

### Explanation of the code:
- We begin by initializing a graph and in-degree array.
- We populate the graph using the provided `prerequisites`, and simultaneously calculate in-degrees.
- A queue is populated with courses that have no prerequisites (in-degree of zero).
- We perform a BFS, processing each course and updating the in-degrees of its dependent courses. Whenever a course's in-degree reaches zero, it is added to the queue to be processed.
- Finally, if the length of the order array matches `numCourses`, we return the order; otherwise, we return an empty list indicating that it's impossible to finish all courses due to a cycle. 

This implementation runs in O(V + E) time complexity, where V is the number of courses and E is the number of prerequisite pairs, and it efficiently handles the constraints specified.

# 261. Graph Valid Tree

### Problem Description 
You have a graph of `n` nodes labeled from `0` to `n - 1`. You are given an integer n and a list of `edges` where `edges[i] = [ai, bi]` indicates that there is an undirected edge between nodes `ai` and `bi` in the graph.

Return `true` if the edges of the given graph make up a valid tree, and `false` otherwise.


Example 1:
Input: n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
Output: true

Example 2:
Input: n = 5, edges = [[0,1],[1,2],[2,3],[1,3],[1,4]]
Output: false

Constraints:
`1 <= 2000 <= n`
`0 <= edges.length <= 5000`
`edges[i].length == 2`
`0 <= ai, bi < n`
`ai != bi`
There are no self-loops or repeated edges.

### Solution 
 To determine whether a given set of edges forms a valid tree, we need to verify two key properties of a tree:

1. **A tree must be connected**: There should be a single component that includes all the nodes.
2. **A tree must not have cycles**: There should be no circular paths.

Given this, we can summarize the conditions that must hold true for a valid tree:
- The number of edges must be exactly `n - 1`, where `n` is the number of nodes. If there are more or fewer edges, it cannot be a valid tree.
- We can use Depth First Search (DFS) or Breadth First Search (BFS) to traverse the graph starting from any node and ensure we can visit all `n` nodes without encountering a cycle.

Here is the step-by-step Python coding solution:



```python
class Solution:
    def validTree(self, n: int, edges: list[list[int]]) -> bool:
        # A tree with 'n' nodes must have exactly 'n-1' edges
        if len(edges) != n - 1:
            return False
        
        # Build the graph using an adjacency list
        graph = {i: [] for i in range(n)}
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        # Set to keep track of visited nodes during DFS
        visited = set()
        
        # Define DFS function
        def dfs(node, parent):
            # Add current node to the visited set
            visited.add(node)
            # Visit all the neighbors
            for neighbor in graph[node]:
                # If the neighbor is already visited and it's not the parent, we've found a cycle
                if neighbor in visited and neighbor != parent:
                    return False
                # If neighbor hasn't been visited, recursively visit it
                if neighbor not in visited:
                    if not dfs(neighbor, node):
                        return False
            return True
        
        # Start DFS from node 0 and None as parent
        if not dfs(0, None):
            return False
        
        # After DFS, check if all nodes were visited
        return len(visited) == n

# Example usage:
# sol = Solution()
# print(sol.validTree(5, [[0,1],[0,2],[0,3],[1,4]]))  # Output: True
# print(sol.validTree(5, [[0,1],[1,2],[2,3],[1,3],[1,4]]))  # Output: False

```

### Explanation:

1. **Edge Count Check**:
   - We check if the number of edges is equal to `n - 1`. If it's not, we can immediately return `False` because it violates the tree property of having exactly `n - 1` edges.

2. **Graph Representation**:
   - We create an adjacency list to represent the graph, where `graph[i]` contains a list of nodes directly connected to node `i`.

3. **Depth-First Search (DFS)**:
   - We perform a DFS starting from node `0` (assuming nodes are labeled from `0` to `n-1`).
   - The `dfs` function checks each node, adds it to the visited set, and explores its neighbors.
   - If we visit a previously visited neighbor that is not the parent of the current node, it indicates a cycle, and we return `False`.

4. **Final Check**:
   - After completing the DFS, we check if the size of the `visited` set equals `n`. If it does, it confirms that all nodes were connected during the traversal.

This solution employs a DFS approach, ensuring we consider both connectivity and the absence of cycles efficiently. The overall time complexity is O(n + e) and space complexity is O(n), making it suitable for the problem constraints.

# 279. Perfect Squares

### Problem Description 
Given an integer `n`, return the least number of perfect square numbers that sum to `n`.

A perfect square is an integer that is the square of an integer; in other words, it is the product of some integer with itself. For example, `1`, `4`, `9`, and `16` are perfect squares while `3` and `11` are not.


Example 1:
Input: n = 12
Output: 3
Explanation: 12 = 4 + 4 + 4.


Example 2:
Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.


Constraints:
`1 <= n <= 104`

### Solution 
 To solve the problem of finding the least number of perfect square numbers that sum up to `n`, we can leverage dynamic programming. The key idea is to break the problem down into smaller subproblems, using previously computed results to build up to the final solution.

### Explanation

1. **Understanding Perfect Squares**: First, we need to understand that a perfect square is any integer that can be expressed as `k^2` for some integer `k`. For instance, if `n = 12`, the perfect squares less than or equal to `n` are `1^2 = 1`, `2^2 = 4`, and `3^2 = 9`.

2. **Dynamic Programming Approach**:
   - We'll create a dynamic programming array `dp` where `dp[i]` will hold the minimum number of perfect squares that sum up to `i`.
   - Initialize `dp[0] = 0` since no numbers are needed to sum up to zero.
   - For every number from `1` to `n`, we will check each perfect square that is less than or equal to that number, and update the `dp` array.

3. **Transition Formula**:
   - For each number `i` (from `1` to `n`), iterate through all perfect squares `j*j` where `j*j <= i`, and update:
     

```python
     dp[i] = min(dp[i], dp[i - j*j] + 1)
     
```
   This indicates that if we take the perfect square `j*j`, we need to add `1` to the number of perfect squares needed to sum to `i - j*j`.

4. **Final Result**: The answer for the least perfect squares that sum up to `n` will be found at `dp[n]`.

### Python Code

Here's how we can implement this approach in Python, suitable for LeetCode submission:



```python
class Solution:
    def numSquares(self, n: int) -> int:
        # Create a list to hold the minimum count of perfect squares
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case

        # Fill the dp array
        for i in range(1, n + 1):
            # Check all perfect squares less than or equal to i
            j = 1
            while j * j <= i:
                # Update dp[i] with the minimum squares needed
                dp[i] = min(dp[i], dp[i - j * j] + 1)
                j += 1
        
        return dp[n]

```

### Explanation of the Code

- **Initialization**: We initialize `dp` with `float('inf')` for all indices since we want to find the minimum, except for `dp[0]` which is `0`.
  
- **Filling the DP table**: We iterate through each number `i` from `1` to `n`, and for each `i`, we determine the minimum number of perfect squares by considering each perfect square `j*j` that is less than or equal to `i`.

- **Return the Result**: Finally, we return `dp[n]`, which contains the least number of perfect squares that sum to `n`. 

### Complexity
- **Time Complexity**: The time complexity is \(O(n \sqrt{n})\) because for every number from `1` to `n`, we may check up to \(\sqrt{n}\) perfect squares.
- **Space Complexity**: The space complexity is \(O(n)\) due to the storage required for the `dp` array. 

This solution efficiently computes the answer for `n` up to `10,000` as required.

# 286. Walls and Gates

### Problem Description 
You are given an `m x n` grid `rooms` initialized with these three possible values.

`-1` A wall or an obstacle.

`0` A gate.

`INF` Infinity means an empty room. We use the value `231 - 1 = 2147483647` to represent `INF` as you may assume that the distance to a gate is less than `2147483647`.

Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with `INF`.


Example 1:
Input: rooms = [[2147483647,-1,0,2147483647],[2147483647,2147483647,2147483647,-1],[2147483647,-1,2147483647,-1],[0,-1,2147483647,2147483647]]
Output: [[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]]

Example 2:
Input: rooms = [[-1]]
Output: [[-1]]

Example 3:
Input: rooms = [[2147483647]]
Output: [[2147483647]]

Example 4:
Input: rooms = [[0]]
Output: [[0]]

Constraints:
`m == rooms.length`
`n == rooms[i].length`
`1 <= m, n <= 250`
`rooms[i][j]` is `-1`, `0`, or `231 - 1`.

### Solution 
 To solve the problem of filling each empty room in a grid with the distance to its nearest gate, we can utilize a breadth-first search (BFS) approach. The BFS is particularly well-suited for this situation because it allows us to explore the shortest paths from multiple sources (the gates) effectively. 

The process is outlined below:

1. **Initialization**: Start by identifying the gates (cells containing `0`) and enqueuing their positions for BFS. The gates will serve as the starting points from which we will calculate distances to the surrounding empty rooms (`INF`).

2. **Queueing Gates**: For each gate detected, we will push its coordinates into a queue. This queue will help us keep track of which cells to explore next during our BFS traversal.

3. **BFS Traversal**: While there are cells in the queue, we will dequeue a cell and check its four possible adjacent cells (up, down, left, right). If an adjacent cell is an empty room (i.e., it has a value of `INF`), we will update the distance of that cell to be one more than the distance of the current cell, then enqueue the adjacent cell for further exploration.

4. **Boundary Conditions**: Throughout the process, we'll make sure to stay within the bounds of the grid and not to overwrite walls or previously computed distances.

The complete code according to the LeetCode format is as follows:



```python
from collections import deque

class Solution:
    def wallsAndGates(self, rooms):
        if not rooms:
            return
        
        # Dimensions of the grid
        m = len(rooms)
        n = len(rooms[0])
        
        # Directions for up, down, left, right moves
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Queue for BFS
        queue = deque()
        
        # Enqueue all the gates' positions
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:  # Found a gate
                    queue.append((i, j))
        
        # BFS to find the shortest distance to a gate
        while queue:
            x, y = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy  # Next cell coordinates
                
                # Check if the next cell is within bounds and is an empty room
                if 0 <= nx < m and 0 <= ny < n and rooms[nx][ny] == 2147483647:
                    # Update distance to the nearest gate (current distance + 1)
                    rooms[nx][ny] = rooms[x][y] + 1
                    queue.append((nx, ny))  # Add the updated cell to the queue
        
        # The rooms grid will be modified in place with distances filled

```

### Explanation of the Code:

- **Imports**: We import `deque` from the `collections` module to efficiently handle our queue.
  
- **Class Declaration**: We define a class `Solution` with a method `wallsAndGates`.

- **Grid Handling**: We first check if the input grid `rooms` is empty.

- **Queue Operation**: After initializing the queue with the positions of all gates, we process each position by dequeuing it and exploring all four directions.

- **Distance Assignment**: When we find an empty room, we update its distance, mark it as visited by assigning it the new distance, and enqueue it for further processing.

This approach runs efficiently in O(m*n) time complexity and ensures that we fill out the rooms as required.

# 301. Remove Invalid Parentheses

### Problem Description 
Given a string `s` that contains parentheses and letters, remove the minimum number of invalid parentheses to make the input string valid.

Return all the possible results. You may return the answer in any order.


Example 1:
Input: s = "()())()"
Output: ["(())()","()()()"]

Example 2:
Input: s = "(a)())()"
Output: ["(a())()","(a)()()"]

Example 3:
Input: s = ")("
Output: [""]

Constraints:
`1 <= s.length <= 25`
`s` consists of lowercase English letters and parentheses `'('` and `')'`.

There will be at most `20` parentheses in `s`.

### Solution 
 Certainly! The problem at hand involves finding valid combinations of parentheses from the given string by removing the minimum number of invalid parentheses. Let's break down the solution and implement it in the format suitable for LeetCode.

### Explanation

1. **Understanding Valid Parentheses**:
   A string with parentheses is valid if for every open parenthesis `'('` there is a corresponding close parenthesis `')'`, and it is correctly nested. 

2. **Finding Minimum Removals**:
   To validate the string and count how many parentheses need to be removed, we can traverse the string while maintaining a balance counter. Every time we encounter `'('`, we increase the counter, and for `')'`, we decrease it. If the counter drops below zero, it means we have an unmatched close parenthesis.

3. **Breadth-First Search (BFS)**:
   We can use BFS to explore all possible strings generated by removing parentheses. Starting from the original string, we will iteratively remove one parenthesis at a time and keep track of all generated strings. Each time we generate a new string, we check if it is valid. If it’s valid and we achieved the least removals needed, we can add it to our results.

4. **Avoiding Repetitions**:
   To avoid processing the same string multiple times, we will use a set to keep track of visited strings.

5. **Stopping Condition**:
   The BFS continues until we find valid strings that minimize the removals.

### Python Code

Below is the Python code that follows the LeetCode solution format:



```python
from collections import deque

class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        def is_valid(string: str) -> bool:
            balance = 0
            for char in string:
                if char == '(':
                    balance += 1
                elif char == ')':
                    balance -= 1
                if balance < 0:
                    return False
            return balance == 0

        # Find the minimum number of removals needed
        def min_removals(string: str) -> int:
            balance = 0
            remove_open = 0
            remove_close = 0
            for char in string:
                if char == '(':
                    balance += 1
                elif char == ')':
                    if balance == 0:
                        remove_close += 1
                    else:
                        balance -= 1
            return balance + remove_close
                    
        # BFS to generate all possible states
        results = set()
        queue = deque([s])
        visited = set([s])
        found = False
        
        while queue:
            current = queue.popleft()
            
            if is_valid(current):
                results.add(current)
                found = True
                
            if found:
                continue
            
            for i in range(len(current)):
                if current[i] in "()":
                    next_state = current[:i] + current[i + 1:]  # Remove the parenthesis
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append(next_state)

        return list(results)


```

### Key Points in the Code

- **Helper Function `is_valid`**: This function checks if a string has valid parentheses by counting the balance of opening and closing brackets.
- **Helper Function `min_removals`**: It calculates how many parentheses need to be removed for the string to be valid.
- **BFS Implementation**: The `deque` is utilized to explore all the possible strings generated by removing one parentheses at a time. Each valid string upon encountering will be added to results.
- **Results**: Finally, we convert the results set to a list and return all valid configurations.

### Conclusion

This solution efficiently finds all valid parentheses combinations by systematically exploring possible strings and ensures that we only explore the necessary paths using BFS. The complexity is manageable due to the relatively short length of the string (max 25 characters). Hence, the solution adheres to the problem's constraints and runs efficiently on LeetCode.

# 310. Minimum Height Trees

### Problem Description 
A tree is an undirected graph in which any two vertices are connected by exactly one path. In other words, any connected graph without simple cycles is a tree.

Given a tree of `n` nodes labelled from `0` to `n - 1`, and an array of `n - 1` `edges` where `edges[i] = [ai, bi]` indicates that there is an undirected edge between the two nodes `ai` and `bi` in the tree, you can choose any node of the tree as the root. When you select a node `x` as the root, the result tree has height `h`. Among all possible rooted trees, those with minimum height (i.e. `min(h)`)  are called minimum height trees (MHTs).

Return a list of all MHTs' root labels. You can return the answer in any order.

The height of a rooted tree is the number of edges on the longest downward path between the root and a leaf.


Example 1:
Input: n = 4, edges = [[1,0],[1,2],[1,3]]
Output: [1]
Explanation: As shown, the height of the tree is 1 when the root is the node with label 1 which is the only MHT.


Example 2:
Input: n = 6, edges = [[3,0],[3,1],[3,2],[3,4],[5,4]]
Output: [3,4]

Example 3:
Input: n = 1, edges = []
Output: [0]

Example 4:
Input: n = 2, edges = [[0,1]]
Output: [0,1]

Constraints:
`1 <= n <= 2 * 104`
`edges.length == n - 1`
`0 <= ai, bi < n`
`ai != bi`
All the pairs `(ai, bi)` are distinct.

The given input is guaranteed to be a tree and there will be no repeated edges.

### Solution 
 To solve the problem of finding Minimum Height Trees (MHTs) in a given undirected tree represented by its nodes and edges, we can follow a systematic approach utilizing a topological sort-like strategy. The main idea is to iteratively prune the leaves of the tree until we reach the center nodes, which will be the roots of the minimum height trees.

### Step-by-step Explanation:

1. **Understanding Trees**: A tree with `n` nodes will always have exactly `n - 1` edges. Each node can be connected to multiple other nodes, but there are no cycles.

2. **Leaf Nodes**: Initially, we can identify leaf nodes as those which have only one connection (degree of one).

3. **Trimming Leaves**: By removing the leaf nodes layer by layer, we can gradually expose the next set of nodes, which might become leaves in the upcoming iterations. We repeat this process until we have 1 or 2 nodes left. These remaining nodes are the roots of the minimum height trees.

4. **Construction of the Graph**: We can represent the tree using an adjacency list. This allows us to efficiently manage and track the degrees of nodes.

5. **Edge Cases**:
   - If there is only one node (`n = 1`), that node is an MHT by default.
   - If there are two nodes (`n = 2`), both nodes are MHTs (since each can be root).

Now, let’s implement this in Python as per the LeetCode format.

### Python Code:



```python
from collections import defaultdict, deque
from typing import List

class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        # Edge case: if there are no nodes, return an empty list
        if n == 0:
            return []
        
        # Edge case: if there's only one node, return that node
        if n == 1:
            return [0]
        
        # Create an adjacency list to represent the tree
        graph = defaultdict(list)
        
        # Fill the adjacency list based on the edges
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
        
        # Initialize a queue for BFS and a list to track leaf nodes
        queue = deque()
        
        # Start with all leaf nodes
        for node in range(n):
            if len(graph[node]) == 1:  # Degree of 1 means it's a leaf
                queue.append(node)
        
        # Remove leaf nodes layer by layer
        remaining_nodes = n
        
        while remaining_nodes > 2:
            size = len(queue)
            remaining_nodes -= size  # Number of nodes being removed

            for _ in range(size):
                leaf = queue.popleft()
                for neighbor in graph[leaf]:
                    graph[neighbor].remove(leaf)  # Remove leaf from its neighbor's graph
                    if len(graph[neighbor]) == 1:  # If neighbor is now a leaf
                        queue.append(neighbor)
        
        # The remaining nodes in the queue are the roots of the minimum height trees
        return list(queue)

```

### How This Code Works:
- **Graph Representation**: It uses a `defaultdict` to create an adjacency list for the graph based on the given edges.
- **Identify Leaves**: Initially, it identifies all the leaf nodes and pushes them into a queue.
- **Prune the Tree**: It enters a loop where it keeps removing leaf nodes until only 1 or 2 nodes are left.
- **Return Result**: Finally, it returns the remaining nodes in the queue, which are the roots of the minimum height trees.

### Complexity:
- **Time Complexity**: \(O(n)\), since each node and edge is processed a constant number of times.
- **Space Complexity**: \(O(n)\), for the graph representation and queue storage.

This approach is efficient and effective for finding MHTs in a given tree structure.

# 314. Binary Tree Vertical Order Traversal

### Problem Description 
Given the `root` of a binary tree, return the vertical order traversal of its nodes' values. (i.e., from top to bottom, column by column).

If two nodes are in the same row and column, the order should be from left to right.


Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: [[9],[3,15],[20],[7]]

Example 2:
Input: root = [3,9,8,4,0,1,7]
Output: [[4],[9],[3,0,1],[8],[7]]

Example 3:
Input: root = [3,9,8,4,0,1,7,null,null,null,2,5]
Output: [[4],[9,5],[3,0,1],[8,2],[7]]

Example 4:
Input: root = []
Output: []

Constraints:
The number of nodes in the tree is in the range `[0, 100]`.

`-100 <= Node.val <= 100`

### Solution 
 To solve the problem of vertical order traversal of a binary tree, we can use a breadth-first search (BFS) approach. The idea is to traverse the tree while keeping track of the vertical columns. We can assign a column index to each node as we traverse the tree, starting from the root node at column index `0`. Nodes to the left of the current node will be assigned a column index of `current_index - 1`, and nodes to the right will be assigned `current_index + 1`.

### Steps to Solve the Problem:

1. **Use a Queue**: We'll employ a queue to perform BFS and a dictionary to keep track of nodes in their respective vertical columns.

2. **Column Mapping**: Use a dictionary where keys are column indices and values are lists of nodes' values that fall into that column. We'll also need to track the minimum and maximum column indices to construct the final result.

3. **BFS Traversal**: While traversing, for each node:
   - Add its value to the corresponding column list in the dictionary.
   - Enqueue its left and right children with updated column indices.

4. **Result Construction**: After we complete the BFS traversal, we'll extract the values from the dictionary in column order (from the minimum to the maximum column index).

5. **Edge Cases**: Handle the case where the tree is empty.

### Code Implementation

Here's the implementation in Python following the LeetCode solution format:



```python
from collections import defaultdict, deque

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def verticalOrder(self, root: TreeNode):
        if not root:
            return []
        
        # A dictionary to hold the vertical order traversal
        column_table = defaultdict(list)
        # A queue to hold nodes along with their column index
        queue = deque([(root, 0)])  # (node, column_index)
        min_column = 0
        max_column = 0
        
        while queue:
            node, column = queue.popleft()
            
            if node is not None:
                # Append the node's value to the corresponding column index list
                column_table[column].append(node.val)
                # Update min and max column range
                min_column = min(min_column, column)
                max_column = max(max_column, column)
                
                # Add the left and right children to the queue with updated indices
                queue.append((node.left, column - 1))
                queue.append((node.right, column + 1))
        
        # Extract the vertical order from the column table
        result = []
        for col in range(min_column, max_column + 1):
            result.append(column_table[col])
        
        return result

```

### Explanation of the Code
1. **TreeNode Class**: Defines the structure of a binary tree node with `val`, `left`, and `right` attributes.

2. **Solution Class and verticalOrder Method**:
   - Check if the root is `None` and return an empty list immediately if true.
   - A `defaultdict` is used to store lists of node values indexed by their column indices.
   - A deque is initialized with the root node and its column index (0).
   - In the BFS loop, nodes are processed until the queue is empty: retrieve the node and its column, store its value in the column table, and enqueue its children with updated column indices.
   - Finally, compile the results from the `column_table` based on index range and return it.

This solution efficiently performs the vertical order traversal of the binary tree while adhering to the required constraints, and will perform well within the given input limits.

# 317. Shortest Distance from All Buildings

### Problem Description 
You are given an `m x n` grid `grid` of values `0`, `1`, or `2`, where:
each `0` marks an empty land that you can pass by freely,
each `1` marks a building that you cannot pass through, and
each `2` marks an obstacle that you cannot pass through.

You want to build a house on an empty land that reaches all buildings in the shortest total travel distance. You can only move up, down, left, and right.

Return the shortest travel distance for such a house. If it is not possible to build such a house according to the above rules, return `-1`.

The total travel distance is the sum of the distances between the houses of the friends and the meeting point.

The distance is calculated using Manhattan Distance, where `distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|`.


Example 1:
Input: grid = [[1,0,2,0,1],[0,0,0,0,0],[0,0,1,0,0]]
Output: 7
Explanation: Given three buildings at (0,0), (0,4), (2,2), and an obstacle at (0,2).

The point (1,2) is an ideal empty land to build a house, as the total travel distance of 3+3+1=7 is minimal.

So return 7.


Example 2:
Input: grid = [[1,0]]
Output: 1

Example 3:
Input: grid = [[1]]
Output: -1

Constraints:
`m == grid.length`
`n == grid[i].length`
`1 <= m, n <= 100`
`grid[i][j]` is either `0`, `1`, or `2`.

There will be at least one building in the `grid`.

### Solution 
 To solve the problem of finding the shortest travel distance for a house placed on an empty land that reaches all buildings in the grid, we can follow these steps:

1. **Understand the Problem**: We need to find an optimal position for a house where the total distance to all buildings is minimized, while avoiding obstacles.

2. **Use BFS for Distance Calculation**: We can simulate the BFS (Breadth-First Search) from every building to calculate the distance to all reachable empty lands. We will also keep track of how many buildings each empty land can reach.

3. **Calculate Minimum Distance**:
   - Use a 2D list (`distance`) to store the total distance from each empty cell to all buildings.
   - Use another 2D list (`reach`) to count how many buildings can reach each empty cell.
   - Iterate through each building and perform a BFS, updating the `distance` and `reach` for every empty land found.

4. **Find the Optimal Position**: After processing all buildings, iterate through the grid again to find an empty land cell that can reach all buildings and has the minimal distance.

5. **Return the Result**: If no valid position was found, return `-1`. Otherwise, return the minimal distance.

Here's the code that follows the above logic and fits the LeetCode solution format:



```python
from collections import deque

class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        if not grid:
            return -1
        
        m, n = len(grid), len(grid[0])
        distance = [[0] * n for _ in range(m)]
        reach = [[0] * n for _ in range(m)]
        
        buildings = 0

        # Directions for moving in 4 directions.
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # BFS function to update distances and reach counts
        def bfs(x, y):
            visited = [[False] * n for _ in range(m)]
            queue = deque([(x, y, 0)])  # stores (current x, current y, distance)
            visited[x][y] = True
            
            while queue:
                curr_x, curr_y, dist = queue.popleft()
                
                for dx, dy in directions:
                    new_x, new_y = curr_x + dx, curr_y + dy
                    if 0 <= new_x < m and 0 <= new_y < n and not visited[new_x][new_y]:
                        if grid[new_x][new_y] == 0:  # only process empty land
                            visited[new_x][new_y] = True
                            distance[new_x][new_y] += dist + 1
                            reach[new_x][new_y] += 1
                            queue.append((new_x, new_y, dist + 1))
        
        # Step 1: BFS from each building
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:  # a building found
                    buildings += 1
                    bfs(i, j)
        
        # Step 2: find the minimum distance
        min_distance = float('inf')
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0 and reach[i][j] == buildings:  # valid empty land
                    min_distance = min(min_distance, distance[i][j])
        
        return min_distance if min_distance != float('inf') else -1

```

### Explanation:
- **Data Structures**:
    - `distance`: This will keep track of the total distance from each empty cell to all buildings.
    - `reach`: This will count how many buildings can reach each empty cell.
  
- **BFS Implementation**: A helper function called `bfs` is used to perform a breadth-first search starting from each building. It updates the `distance` and keeps track of how many buildings can reach each empty land.

- **Final Calculation**: After performing BFS for all buildings, we iterate over the grid to find the cell (empty land) with the minimum distance that can be reached by all buildings.

This straightforward approach ensures that we consider all possible paths from each building to all empty lands efficiently. Given the constraints, this solution should work within the limits of the problem.

# 323. Number of Connected Components in an Undirected Graph

### Problem Description 
You have a graph of `n` nodes. You are given an integer `n` and an array `edges` where `edges[i] = [ai, bi]` indicates that there is an edge between `ai` and `bi` in the graph.

Return the number of connected components in the graph.


Example 1:
Input: n = 5, edges = [[0,1],[1,2],[3,4]]
Output: 2

Example 2:
Input: n = 5, edges = [[0,1],[1,2],[2,3],[3,4]]
Output: 1

Constraints:
`1 <= n <= 2000`
`1 <= edges.length <= 5000`
`edges[i].length == 2`
`0 <= ai <= bi < n`
`ai != bi`
There are no repeated edges.

### Solution 
 To solve the problem of counting the number of connected components in a graph, we can utilize Depth-First Search (DFS) or Breadth-First Search (BFS) algorithms. Here's a detailed explanation of how to approach this problem, followed by a Python function formatted for LeetCode.

### Explanation:

1. **Graph Representation**:
   We can represent the graph using an adjacency list. Each node will have a list of its neighboring nodes based on the edges provided.

2. **Traversal**:
   Using either DFS or BFS, we will traverse the graph starting from each unvisited node. Each time we initiate a traversal from an unvisited node, we find a new connected component.

3. **Counting Components**:
   We maintain a count of how many times we start a new traversal (indicating a new connected component).

4. **Visited Nodes**:
   To keep track of which nodes we have already visited, we can use a boolean array.

### Steps in the Code:

1. Initialize the adjacency list based on the `edges`.
2. Create a `visited` array to track which nodes have been visited.
3. Iterate through each node, and if it hasn't been visited, perform DFS/BFS from that node and mark all reachable nodes as visited.
4. Increment the count of connected components each time a new DFS/BFS is initiated.

### Python Code:

Here’s the implementation using DFS:



```python
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        from collections import defaultdict
        
        # Step 1: Build the graph using adjacency list
        graph = defaultdict(list)
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
        
        # Step 2: Initialize visited array and component count
        visited = [False] * n
        components_count = 0
        
        # Step 3: Define the DFS function
        def dfs(node):
            stack = [node]
            while stack:
                current = stack.pop()
                for neighbor in graph[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
        
        # Step 4: Iterate through each node and count components
        for i in range(n):
            if not visited[i]:
                visited[i] = True  # mark current node as visited
                components_count += 1  # found a new component
                dfs(i)  # perform DFS from this node
        
        return components_count

```

### How to Submit:

1. Copy the above code into your Python environment on LeetCode.
2. Ensure that you import `List` from the `typing` library at the beginning of your code if it's not already done.
3. Run the code against the provided test cases to ensure it behaves as expected.

### Complexity Analysis:

- **Time Complexity**: O(V + E), where V is the number of vertices (nodes) and E is the number of edges. Each node and edge will be visited once.
- **Space Complexity**: O(V), where V is for the visited list and the adjacency list storage for the graph.

This approach efficiently counts the number of connected components in the graph based on the given nodes and edges.

# 339. Nested List Weight Sum

### Problem Description 
You are given a nested list of integers `nestedList`. Each element is either an integer or a list whose elements may also be integers or other lists.

The depth of an integer is the number of lists that it is inside of. For example, the nested list `[1,[2,2],[[3],2],1]` has each integer's value set to its depth.

Return the sum of each integer in `nestedList` multiplied by its depth.


Example 1:
Input: nestedList = [[1,1],2,[1,1]]
Output: 10
Explanation: Four 1's at depth 2, one 2 at depth 1. 1*2 + 1*2 + 2*1 + 1*2 + 1*2 = 10.


Example 2:
Input: nestedList = [1,[4,[6]]]
Output: 27
Explanation: One 1 at depth 1, one 4 at depth 2, and one 6 at depth 3. 1*1 + 4*2 + 6*3 = 27.


Example 3:
Input: nestedList = [0]
Output: 0

Constraints:
`1 <= nestedList.length <= 50`
The values of the integers in the nested list is in the range `[-100, 100]`.

The maximum depth of any integer is less than or equal to `50`.

### Solution 
 To solve the problem of calculating the sum of each integer in a nested list multiplied by its depth, we can define a recursive approach. The main idea is to iterate through the elements of the nested list and check whether each element is an integer or a nested list. If it's an integer, we add its value multiplied by the current depth to a total sum. If it's a nested list, we recursively call our function with an incremented depth.

Here is a step-by-step explanation followed by the code:

1. **Define Depth**: Keep track of the current depth as we explore the nested structure.
2. **Recursive Exploration**: If we encounter an integer, we add its contribution (value * depth) to the total. If we encounter a nested list, we call the function recursively, increasing the depth by 1.
3. **Base Case**: When we reach the end of a list, we return the accumulated sum.

We will implement a helper function that can handle this recursively.

Here's the Python code that adheres to LeetCode's requirements:



```python
class NestedInteger:
    # This class typically provides methods to check if the NestedInteger is an integer 
    # or a single nested list, but for this problem, we'll define our interface.
    
    def isInteger(self):
        # Return True if this NestedInteger holds a single integer, otherwise False.
        pass

    def getInteger(self):
        # Return the single integer that this NestedInteger holds, if it holds an integer.
        pass

    def getList(self):
        # Return the list that this NestedInteger holds, if it holds a nested list.
        pass

class Solution:
    def depthSum(self, nestedList: [NestedInteger]) -> int:
        def calculateDepthSum(nested_list, depth):
            total = 0  # Initialize total sum for this level
            for ni in nested_list:
                if ni.isInteger():  # Check if this is an integer
                    total += ni.getInteger() * depth  # Multiply by current depth
                else:
                    # It's a nested list, so we recurse with depth + 1
                    total += calculateDepthSum(ni.getList(), depth + 1)
            return total  # Return the accumulated sum at this depth
        
        return calculateDepthSum(nestedList, 1)  # Start from depth 1

# Example usage case:
# nestedList = [[1,1],2,[1,1]]
# solution = Solution()
# result = solution.depthSum(nestedList)
# print(result)  # Output: 10

```

### Explanation of the Code:
- We define a `NestedInteger` class which is given by LeetCode (assuming its methods are properly defined).
- Inside the `Solution` class, we implement the `depthSum` method which takes a list of `NestedInteger`.
- We create a helper function `calculateDepthSum` that performs the actual calculation.
- For each element in the list, we check:
  - If it is an integer via `isInteger()`, we grab its value with `getInteger()`, multiply by the current depth, and add to the total.
  - If it is a nested list, we call `calculateDepthSum` recursively with an increased depth (`depth + 1`).
- The recursive calls allow us to navigate through the nested structure seamlessly.
- Finally, we start our calculations by calling `calculateDepthSum` with the initial depth of 1.

This solution efficiently calculates the desired sum and works within the constraints provided, handling various levels of nesting.

# 407. Trapping Rain Water II

### Problem Description 
Given an `m x n` matrix of positive integers representing the height of each unit cell in a 2D elevation map, compute the volume of water it is able to trap after raining.


Example:
Given the following 3x6 height map:
[
  [1,4,3,1,3,2],
  [3,2,1,3,2,4],
  [2,3,3,2,3,1]
]
Return 4.

The above image represents the elevation map `[[1,4,3,1,3,2],[3,2,1,3,2,4],[2,3,3,2,3,1]]` before the rain.

After the rain, water is trapped between the blocks. The total volume of water trapped is 4.


Constraints:
`1 <= m, n <= 110`
`0 <= heightMap[i][j] <= 20000`

### Solution 
 To solve the problem of computing the volume of water trapped in a 2D elevation map, we can use a priority queue (min-heap) to efficiently evaluate the heights of the surrounding cells and determine where water can be trapped. 

### Explanation:

1. **Understanding the Problem**: When it rains, water will be trapped in low-lying areas surrounded by higher elevations. The volume of water that can be trapped is determined by the difference in height between the surrounding cells and the lowest height of the border surrounding the trapped area.

2. **Using a Min-Heap**: The algorithm will utilize a min-heap to keep track of the current boundary of the trapped regions. Start by adding all border cells to the heap since they can't trap any water themselves, and initialize the volume to 0.

3. **Exploring Neighbor Cells**: While extracting the cell with the smallest height from the heap, we check its neighboring cells (up, down, left, right). If a neighboring cell is lower than the current cell, it means water can be trapped above it. The amount of water trapped is given by the difference between the current height and the neighbor's height.

4. **Updating Heights**: After checking a neighbor cell, if it's lower than the current height, update that neighbor's height to the current height and add it back to the heap.

5. **Repeat Until All Cells are Processed**: Continue this process until all reachable cells are processed.

### Python Code:
Here's the complete implementation that you can run directly on LeetCode.



```python
import heapq

class Solution:
    def trapRainWater(self, heightMap):
        if not heightMap or not heightMap[0]:
            return 0
        
        # Initialize variables
        m, n = len(heightMap), len(heightMap[0])
        visited = [[False] * n for _ in range(m)]
        min_heap = []
        
        # Add all the border cells to the heap and mark them as visited
        for i in range(m):
            for j in range(n):
                if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                    heapq.heappush(min_heap, (heightMap[i][j], i, j))
                    visited[i][j] = True
        
        water_trapped = 0
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        # Process the cells in the min-heap
        while min_heap:
            current_height, x, y = heapq.heappop(min_heap)
            
            for direction in directions:
                nx, ny = x + direction[0], y + direction[1]
                
                # Check if the neighbor is within bounds
                if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny]:
                    # If the neighbor is lower than the current height
                    # water can be trapped above it
                    if heightMap[nx][ny] < current_height:
                        water_trapped += current_height - heightMap[nx][ny]
                    
                    # Update the height of the neighbor to be at least the current height
                    # and add it to the min-heap
                    heapq.heappush(min_heap, (max(heightMap[nx][ny], current_height), nx, ny))
                    visited[nx][ny] = True
        
        return water_trapped

```

### Steps to Run on LeetCode:
1. Copy the provided solution code.
2. Paste it into the code editor on the LeetCode problem page for trapping rainwater.
3. Ensure you have the problem statement and example test cases ready.
4. Run the code to see if it passes the tests.

This solution efficiently calculates the amount of water that can be trapped using O(m * n log(m * n)) time complexity due to the heap operations, which is feasible given the constraints.

# 417. Pacific Atlantic Water Flow

### Problem Description 
You are given an `m x n` integer matrix `heights` representing the height of each unit cell in a continent. The Pacific ocean touches the continent's left and top edges, and the Atlantic ocean touches the continent's right and bottom edges.

Water can only flow in four directions: up, down, left, and right. Water flows from a cell to an adjacent one with an equal or lower height.

Return a list of grid coordinates where water can flow to both the Pacific and Atlantic oceans.


Example 1:
Input: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]

Example 2:
Input: heights = [[2,1],[1,2]]
Output: [[0,0],[0,1],[1,0],[1,1]]

Constraints:
`m == heights.length`
`n == heights[i].length`
`1 <= m, n <= 200`
`1 <= heights[i][j] <= 105`

### Solution 
 To solve the problem of determining which cells in the matrix can flow into both the Pacific and Atlantic oceans, we'll implement a depth-first search (DFS) approach. The key here is to perform DFS from both oceans simultaneously and identify which cells can reach both.

### Explanation of the Approach

1. **Matrix Definition**: The input matrix `heights` represents the heights of different cells on the continent.

2. **Oceans Definition**:
    - The Pacific Ocean touches the top and left edges of the matrix.
    - The Atlantic Ocean touches the bottom and right edges of the matrix.

3. **Flow Condition**: Water can flow from a cell to one of its adjacent cells (up, down, left, right) only if the adjacent cell has a height less than or equal to the height of the current cell.

4. **Two-pass DFS**: 
    - First, we will start DFS from all cells that touch the Pacific Ocean and store the reachable cells in a set.
    - Second, we will perform DFS from all cells that touch the Atlantic Ocean, storing these reachable cells in another set.
  
5. **Intersecting Results**: The answer will be the intersection of both sets, which represents cells that can flow to both oceans.

### Implementation

Here is how you can implement the solution in Python following the LeetCode format:



```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights:
            return []

        rows, cols = len(heights), len(heights[0])
        pacific_reachable = set()
        atlantic_reachable = set()

        def dfs(r, c, visited):
            visited.add((r, c))
            # Directions for up, down, left, right movements
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and heights[nr][nc] >= heights[r][c]:
                    dfs(nr, nc, visited)

        # Start DFS from the borders that touch the Pacific Ocean
        for c in range(cols):
            dfs(0, c, pacific_reachable)
        for r in range(rows):
            dfs(r, 0, pacific_reachable)

        # Start DFS from the borders that touch the Atlantic Ocean
        for c in range(cols):
            dfs(rows - 1, c, atlantic_reachable)
        for r in range(rows):
            dfs(r, cols - 1, atlantic_reachable)

        # The intersection of pacific_reachable and atlantic_reachable gives the result
        result = list(pacific_reachable.intersection(atlantic_reachable))
        return result

```

### Explanation of the Code:

1. **Initialization**: 
    - We check if `heights` is empty, and if so, return an empty list.
    - We derive the number of rows (`rows`) and columns (`cols`).
    - Two sets (`pacific_reachable` and `atlantic_reachable`) are initialized to keep track of reachable cells for each ocean.

2. **DFS Function**: 
    - The `dfs` function takes current cell coordinates and the set of visited coordinates as parameters.
    - It marks the current cell as visited and iterates through each potential direction to move (right, down, left, up).
    - If a neighboring cell is within bounds, not visited, and its height is greater than or equal to the current height, we call `dfs` for that neighboring cell.

3. **Flood Fill from Oceans**:
    - We invoke `dfs` starting from all cells that are on the Pacific border (first row and first column).
    - Similarly, we invoke `dfs` for all cells on the Atlantic border (last row and last column).

4. **Intersection**: Finally, we return the cells that are reachable from both sets.

### Results
You can run this code directly on the LeetCode website where the problem is hosted by copying and pasting this class into the provided solution editor. The algorithm has a complexity of O(m * n) as each cell is processed once during the DFS.

# 429. N-ary Tree Level Order Traversal

### Problem Description 
Given an n-ary tree, return the level order traversal of its nodes' values.

Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples).


Example 1:
Input: root = [1,null,3,2,4,null,5,6]
Output: [[1],[3,2,4],[5,6]]

Example 2:
Input: root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
Output: [[1],[2,3,4,5],[6,7,8,9,10],[11,12,13],[14]]

Constraints:
The height of the n-ary tree is less than or equal to `1000`
The total number of nodes is between `[0, 104]`

### Solution 
 To solve the problem of performing a level order traversal on an n-ary tree, we can utilize a breadth-first search (BFS) approach. We'll employ a queue to help us navigate through the tree level by level. 

### Explanation

1. **Data Structure**:
   - We'll utilize a queue (FIFO) to track nodes at each level.
   - We'll also use a list to hold the result of the level order traversal.

2. **Traversal Logic**:
   - Start by checking if the input root is `None`. If it is, we return an empty list.
   - Initialize the queue with the root node and prepare an empty list to store the results.
   - While there are nodes in the queue:
     - Determine the number of nodes at the current level using the queue's size.
     - For each node at this level:
       - Dequeue the node and add its value to a temporary list.
       - Enqueue all of its children to the queue.
     - After processing all nodes at the current level, append the temporary list to the results list.

3. **Complexity**:
   - The time complexity is O(N), where N is the number of nodes, since we visit each node exactly once.
   - The space complexity can be O(M), where M is the maximum number of nodes at any level in the tree (this can occur at the last level).

Here's a Python implementation of the above logic:



```python
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []

class Solution:
    def levelOrder(self, root: Node) -> List[List[int]]:
        if root is None:
            return []
        
        result = []
        queue = [root]  # Start with the root in the queue
        
        while queue:
            level_size = len(queue)  # Number of nodes at the current level
            level_values = []  # Temporary list to store values of the current level
            
            for _ in range(level_size):
                # Dequeue the front node
                node = queue.pop(0)
                level_values.append(node.val)  # Store the value
                
                # Enqueue all children
                for child in node.children:
                    queue.append(child)
            
            result.append(level_values)  # Add the current level values to the results
        
        return result

```

### How to Run:
The code defines the `Node` class and `Solution` class with the required `levelOrder` method. You can create an instance of `Node` for your n-ary tree and call the `levelOrder` method of the `Solution` class to get the level order traversal.

### Example Usage:


```python
# Example 1
root = Node(1, [Node(3, [Node(5), Node(6)]), Node(2), Node(4)])
solution = Solution()
print(solution.levelOrder(root))  # Output: [[1], [3, 2, 4], [5, 6]]

# Example 2
root = Node(1, [
    Node(2), Node(3), Node(4), Node(5, [
        Node(6, [Node(8)]),
        Node(7, [Node(9), Node(10)])
    ])
])
solution = Solution()
print(solution.levelOrder(root))  # Output: [[1], [2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13], [14]]

```

This code adheres to the LeetCode format and can be run directly in a LeetCode environment where appropriate.

# 490. The Maze

### Problem Description 
There is a ball in a `maze` with empty spaces (represented as `0`) and walls (represented as `1`). The ball can go through the empty spaces by rolling up, down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, it could choose the next direction.

Given the `m x n` `maze`, the ball's `start` position and the `destination`, where `start = [startrow, startcol]` and `destination = [destinationrow, destinationcol]`, return `true` if the ball can stop at the destination, otherwise return `false`.

You may assume that the borders of the maze are all walls (see examples).


Example 1:
Input: maze = [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]], start = [0,4], destination = [4,4]
Output: true
Explanation: One possible way is : left -> down -> left -> down -> right -> down -> right.


Example 2:
Input: maze = [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]], start = [0,4], destination = [3,2]
Output: false
Explanation: There is no way for the ball to stop at the destination. Notice that you can pass through the destination but you cannot stop there.


Example 3:
Input: maze = [[0,0,0,0,0],[1,1,0,0,1],[0,0,0,0,0],[0,1,0,0,1],[0,1,0,0,0]], start = [4,3], destination = [0,1]
Output: false

Constraints:
`m == maze.length`
`n == maze[i].length`
`1 <= m, n <= 100`
`maze[i][j]` is `0` or `1`.

`start.length == 2`
`destination.length == 2`
`0 <= startrow, destinationrow <= m`
`0 <= startcol, destinationcol <= n`
Both the ball and the destination exist in an empty space, and they will not be in the same position initially.

The maze contains at least 2 empty spaces.

### Solution 
 To solve the problem of determining whether the ball can reach the destination in the maze, we can employ a depth-first search (DFS) or breadth-first search (BFS) algorithm. The fundamental idea is to simulate the rolling of the ball through the empty spaces of the maze until it hits a wall, and then check whether the destination is reachable.

Here’s a step-by-step breakdown of the approach:

1. **Initialize Variables**: Set up the maze dimensions and define the possible directions the ball can roll (up, down, left, right).

2. **DFS Function**: Create a recursive DFS function that will navigate through the maze starting from the current ball position. The DFS will:
   - Check if the current position is the destination. If it is, return `true`.
   - If the current position has already been visited, return `false` to avoid cycles.
   - Mark the current position as visited.

3. **Rolling the Ball**: In the DFS function, simulate rolling the ball in all four directions:
   - For each direction, continue moving until hitting a wall (a `1`).
   - Save the new position and check if it leads to the destination.

4. **Boundary Conditions**: Ensure that the DFS does not go out of bounds of the maze.

5. **Main Function**: In the main function, call the DFS beginning from the start position and return the result.

Here's the actual implementation in Python that you can use directly on LeetCode:



```python
class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        m, n = len(maze), len(maze[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] # right, down, left, up
        visited = set()
        
        def dfs(x, y):
            # Check if we reached the destination
            if [x, y] == destination:
                return True
            
            # Mark the current cell as visited
            if (x, y) in visited:
                return False
            visited.add((x, y))
            
            # Roll the ball in each direction
            for dx, dy in directions:
                nx, ny = x, y
                
                # Roll until hitting a wall
                while 0 <= nx + dx < m and 0 <= ny + dy < n and maze[nx + dx][ny + dy] == 0:
                    nx += dx
                    ny += dy
                
                # Check the next position after stopping
                if dfs(nx, ny):
                    return True
            
            return False
        
        return dfs(start[0], start[1])

```

### Explanation of the Implementation:

- **Class and Function Definition**: The `Solution` class contains the method `hasPath`. This is the required function format for LeetCode.
- **Directional Moves**: We define directions as a list of tuples representing movement in four possible directions.
- **Visited Set**: We use a set to keep track of visited positions to avoid infinite loops.
- **DFS Implementation**: The `dfs` function takes the current position `(x, y)`, checks if it's the destination, marks it as visited, and rolls the ball in all four directions.
- **Stopping Condition**: When the ball rolls, it will stop at the wall. The position is then checked recursively.
- **Final Output**: The main function returns the result of the DFS starting from the `start` position.

This method efficiently navigates the maze while preventing unnecessary revisits, ensuring we can accurately determine if the ball can reach the destination.

# 499. The Maze III

### Problem Description 
There is a ball in a `maze` with empty spaces (represented as `0`) and walls (represented as `1`). The ball can go through the empty spaces by rolling up, down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, it could choose the next direction. There is also a hole in this maze. The ball will drop into the hole if it rolls onto the hole.

Given the `m x n` `maze`, the ball's position `ball` and the hole's position `hole`, where `ball = [ballrow, ballcol]` and `hole = [holerow, holecol]`, return a string `instructions` of all the instructions that the ball should follow to drop in the hole with the shortest distance possible. If there are multiple valid instructions, return the lexicographically minimum one. If the ball can't drop in the hole, return `"impossible"`.

If there is a way for the ball to drop in the hole, the answer `instructions` should contain the characters `'u'` (i.e., up), `'d'` (i.e., down), `'l'` (i.e., left), and `'r'` (i.e., right).

The distance is the number of empty spaces traveled by the ball from the start position (excluded) to the destination (included).

You may assume that the borders of the maze are all walls (see examples).


Example 1:
Input: maze = [[0,0,0,0,0],[1,1,0,0,1],[0,0,0,0,0],[0,1,0,0,1],[0,1,0,0,0]], ball = [4,3], hole = [0,1]
Output: "lul"
Explanation: There are two shortest ways for the ball to drop into the hole.

The first way is left -> up -> left, represented by "lul".

The second way is up -> left, represented by 'ul'.

Both ways have shortest distance 6, but the first way is lexicographically smaller because 'l' < 'u'. So the output is "lul".


Example 2:
Input: maze = [[0,0,0,0,0],[1,1,0,0,1],[0,0,0,0,0],[0,1,0,0,1],[0,1,0,0,0]], ball = [4,3], hole = [3,0]
Output: "impossible"
Explanation: The ball cannot reach the hole.


Example 3:
Input: maze = [[0,0,0,0,0,0,0],[0,0,1,0,0,1,0],[0,0,0,0,1,0,0],[0,0,0,0,0,0,1]], ball = [0,4], hole = [3,5]
Output: "dldr"

Constraints:
`m == maze.length`
`n == maze[i].length`
`1 <= m, n <= 100`
`maze[i][j]` is `0` or `1`.

`ball.length == 2`
`hole.length == 2`
`0 <= ballrow, holerow <= m`
`0 <= ballcol, holecol <= n`
Both the ball and the hole exist in an empty space, and they will not be in the same position initially.

The maze contains at least 2 empty spaces.

### Solution 
 To solve the given problem of guiding a ball to a hole in a maze while adhering to constraints about movement, we'll implement a breadth-first search (BFS) strategy to explore possible paths from the ball’s starting position to the hole. The BFS approach is well-suited for this problem due to the requirement to find the shortest path (in terms of number of moves) first.

Here's a step-by-step explanation of how we will implement the solution, followed by the full code that can be run directly on the LeetCode platform.

### Step-by-step Explanation:

1. **Initialization**:
   - Define the directions for movement: up, down, left, right, associating each with its respective character (`'u'`, `'d'`, `'l'`, and `'r'`).
   - Use a priority queue (min-heap) to explore paths in order of their distance travelled. This will also help ensure that we find the lexicographically smallest path if multiple paths of the same distance exist.
   - Create a `visited` set to keep track of visited positions, so we don't reprocess them.

2. **BFS Exploration**:
   - Start the BFS from the ball's initial position.
   - For each direction, simulate rolling the ball until it hits a wall. While rolling:
     - Count the number of spaces rolled.
     - Stop rolling if the ball arrives at the hole; record the path taken.
   - If the ball enters a new position, push the new state into the priority queue.

3. **Finding the Optimal Path**:
   - Continue processing the queue until all possible paths are evaluated or until reaching the hole.
   - If the hole is reached, check if the current path is better than previously recorded paths (shorter or lexicographically smaller).
   - Return the best path found or "impossible" if no path can reach the hole.

### Python Code:

Here’s how the above logic can be implemented in Python:



```python
import heapq

class Solution:
    def findShortestWay(self, maze: List[List[int]], ball: List[int], hole: List[int]) -> str:
        m, n = len(maze), len(maze[0])
        directions = [(0, 1, 'r'), (0, -1, 'l'), (1, 0, 'd'), (-1, 0, 'u')]  # (dx, dy, direction)
        start = (ball[0], ball[1], 0, "")  # (row, col, distance, path)
        pq = []
        heapq.heappush(pq, start)
        
        visited = set()
        visited.add((ball[0], ball[1]))
        
        best_path = None
        best_distance = float('inf')
        
        while pq:
            row, col, dist, path = heapq.heappop(pq)
            
            # Check if we are at the hole
            if (row, col) == (hole[0], hole[1]):
                if dist < best_distance:
                    best_distance = dist
                    best_path = path
                elif dist == best_distance and (best_path is None or path < best_path):
                    best_path = path
                continue
            
            for dx, dy, d in directions:
                new_row, new_col = row, col
                steps = 0
                
                while 0 <= new_row + dx < m and 0 <= new_col + dy < n and maze[new_row + dx][new_col + dy] == 0:
                    new_row += dx
                    new_col += dy
                    steps += 1
                    
                    # If we reach the hole
                    if (new_row, new_col) == (hole[0], hole[1]):
                        break
                
                if steps > 0 and (new_row, new_col) not in visited:
                    visited.add((new_row, new_col))
                    heapq.heappush(pq, (new_row, new_col, dist + steps, path + d))
        
        return best_path if best_path is not None else "impossible"

```

### Explanation of Code:
- The `findShortestWay` method receives the maze, ball's position, and hole's position as inputs.
- We initialize a priority queue with the starting position of the ball and an empty path string.
- The BFS continues until the queue is empty, exploring all valid moves, rolling the ball until it either hits a wall or reaches the hole.
- If the hole is reached, the code checks if the current path is a new best path based on distance and lexicographical order before continuing.
- Finally, the method returns the best path found or "impossible" if it cannot reach the hole.

This implementation efficiently finds the shortest path to the hole and handles the conditions laid out in the problem statement.

# 505. The Maze II

### Problem Description 
There is a ball in a `maze` with empty spaces (represented as `0`) and walls (represented as `1`). The ball can go through the empty spaces by rolling up, down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, it could choose the next direction.

Given the `m x n` `maze`, the ball's `start` position and the `destination`, where `start = [startrow, startcol]` and `destination = [destinationrow, destinationcol]`, return the shortest distance for the ball to stop at the destination. If the ball cannot stop at `destination`, return `-1`.

The distance is the number of empty spaces traveled by the ball from the start position (excluded) to the destination (included).

You may assume that the borders of the maze are all walls (see examples).


Example 1:
Input: maze = [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]], start = [0,4], destination = [4,4]
Output: 12
Explanation: One possible way is : left -> down -> left -> down -> right -> down -> right.

The length of the path is 1 + 1 + 3 + 1 + 2 + 2 + 2 = 12.


Example 2:
Input: maze = [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]], start = [0,4], destination = [3,2]
Output: -1
Explanation: There is no way for the ball to stop at the destination. Notice that you can pass through the destination but you cannot stop there.


Example 3:
Input: maze = [[0,0,0,0,0],[1,1,0,0,1],[0,0,0,0,0],[0,1,0,0,1],[0,1,0,0,0]], start = [4,3], destination = [0,1]
Output: -1

Constraints:
`m == maze.length`
`n == maze[i].length`
`1 <= m, n <= 100`
`maze[i][j]` is `0` or `1`.

`start.length == 2`
`destination.length == 2`
`0 <= startrow, destinationrow <= m`
`0 <= startcol, destinationcol <= n`
Both the ball and the destination exist in an empty space, and they will not be in the same position initially.

The maze contains at least 2 empty spaces.

### Solution 
 To solve the problem of finding the shortest distance for a ball to stop at a destination in a maze, we can utilize a Breadth-First Search (BFS) approach. Here’s a detailed breakdown of the solution strategy:

### Problem Explanation

The ball in the maze rolls in one of four directions until it hits a wall, and it can stop rolling there in order to choose the next direction. We need to calculate the minimum distance from the starting position of the ball to the destination cell. If the ball cannot reach the destination with the rules given, we should return -1.

### Key Points

1. **Direction Rolling**: The ball can roll in four directions - up, down, left, right. It won't stop until reaching the next wall.
2. **BFS Usage**: BFS is ideal here as it explores all possible positions layer by layer; thus it guarantees that the first time we reach the destination cell is the shortest path.
3. **Visited Tracking**: We need to keep track of the cells we’ve already visited to avoid infinite loops.
4. **Edge Cases**: If the starting or destination cell is invalid, we simply return -1.

### Solution Steps

1. **Initialize BFS**: Start the BFS from the starting position of the ball, using a queue to explore all possible positions.
2. **Direction Vectors**: Define direction vectors for rolling the ball in all four directions.
3. **Rolling Mechanism**: For each direction, continuously roll the ball until hitting a wall, capturing the path length.
4. **Check Destination**: If during the BFS we reach the destination, return the path length; if we exhaust all possibilities without reaching the destination, return -1.

### Python Code Implementation

Here's the Python code adhering to the LeetCode format:



```python
from collections import deque

class Solution:
    def findShortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        # Directions for rolling the ball: [up, down, left, right]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        m, n = len(maze), len(maze[0])
        queue = deque([(start[0], start[1], 0)])  # (row, col, distance)
        visited = [[False for _ in range(n)] for _ in range(m)]
        visited[start[0]][start[1]] = True
        min_distance = float('inf')
        
        while queue:
            x, y, distance = queue.popleft()
            
            # Check if we reached the destination
            if (x, y) == (destination[0], destination[1]):
                min_distance = min(min_distance, distance)
                continue
            
            # Explore all four directions
            for dx, dy in directions:
                new_x, new_y, path_length = x, y, distance
                
                # Roll the ball until hitting a wall
                while 0 <= new_x + dx < m and 0 <= new_y + dy < n and maze[new_x + dx][new_y + dy] == 0:
                    new_x += dx
                    new_y += dy
                    path_length += 1
                
                # Check if we can visit this position
                if not visited[new_x][new_y]:
                    visited[new_x][new_y] = True
                    queue.append((new_x, new_y, path_length))
        
        return min_distance if min_distance != float('inf') else -1

```

### Explanation of the Code

- We first create a BFS queue initialized with the starting position and a distance of 0.
- A `visited` matrix keeps track of cells we've already explored to prevent revisiting.
- As we explore each direction, we roll the ball until it hits a wall, updating the new position and calculated distance.
- If we reach the destination, we store the minimum distance found.
- Finally, we check if we found any valid path to the destination; if not, return -1.

### Complexity Analysis

- **Time Complexity**: O(m * n) where m and n are the dimensions of the maze, as in the worst case we might explore every cell.
- **Space Complexity**: O(m * n) for the visited array and the queue.

This efficient approach ensures we can tackle the problem within the constraints provided.

# 513. Find Bottom Left Tree Value

### Problem Description 
Given the `root` of a binary tree, return the leftmost value in the last row of the tree.


Example 1:
Input: root = [2,1,3]
Output: 1

Example 2:
Input: root = [1,2,3,4,null,5,6,null,null,7]
Output: 7

Constraints:
The number of nodes in the tree is in the range `[1, 104]`.

`-231 <= Node.val <= 231 - 1`

### Solution 
 To solve the problem of finding the leftmost value in the last row of a binary tree, we can use a breadth-first search (BFS) approach. The reason BFS is suitable for this problem is that it traverses the tree level by level, allowing us to track the nodes in each row easily.

We will maintain a queue to explore the tree level by level. At each level, we will store the nodes, and when we reach the last level, the first node we encounter will be the leftmost node of that level.

Here’s a step-by-step breakdown of the approach:

1. **Initialization**: Create a queue using `collections.deque` to handle nodes for BFS. Start by adding the root node to the queue.
  
2. **BFS Loop**: While the queue is not empty, execute a loop where:
   - Get the current size of the queue (`level_size`), which tells us how many nodes are at the current level.
   - Initialize a variable to hold the leftmost value of this level.
   - Iterate through all nodes at the current level:
     - For the first node processed at the current level, we set it as the leftmost value.
     - Dequeue the current node and enqueue its children (left child before right child) for the next level.
   
3. **Result**: After processing all levels, the last leftmost value saved will be our answer.

This process ensures that we only keep track of the leftmost value as we traverse down the tree, and by the time we finish the BFS, that value will be the answer.

Here's the implementation of the above logic in Python, formatted according to LeetCode's expectations:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from collections import deque

class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        # Initialize a queue for BFS
        queue = deque([root])
        leftmost_value = root.val  # Initialize the leftmost value
        
        while queue:
            level_size = len(queue)  # Number of nodes at the current level
            
            for i in range(level_size):
                node = queue.popleft()  # Get the current node
                
                # For the first node of the current level, update leftmost_value
                if i == 0:
                    leftmost_value = node.val
                
                # Enqueue left child first, then right child
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return leftmost_value  # Return the last recorded leftmost value

```

### Explanation of the Code:
- We define a `TreeNode` class that constructs a binary tree node.
- The `Solution` class contains our method `findBottomLeftValue`, which takes the root of the tree as input.
- We initialize a queue to perform BFS and store the leftmost value's potential answer.
- Using a while loop, we process each level of the tree; we grab the size of the current level to control how many nodes we process.
- During the inner loop, we dequeue (pop) nodes from our queue and check if they're the first node at that level (to potentially update our leftmost value).
- Finally, after processing all nodes, we return the last recorded leftmost value, which, by the algorithm's design, will be the leftmost value in the last row of the binary tree.

This approach efficiently finds the solution while conforming to the constraints of the problem.

# 515. Find Largest Value in Each Tree Row

### Problem Description 
Given the `root` of a binary tree, return an array of the largest value in each row of the tree (0-indexed).


Example 1:
Input: root = [1,3,2,5,3,null,9]
Output: [1,3,9]

Example 2:
Input: root = [1,2,3]
Output: [1,3]

Example 3:
Input: root = [1]
Output: [1]

Example 4:
Input: root = [1,null,2]
Output: [1,2]

Example 5:
Input: root = []
Output: []

Constraints:
The number of nodes in the tree will be in the range `[0, 104]`.

`-231 <= Node.val <= 231 - 1`

### Solution 
 To solve the problem of finding the largest value in each row of a binary tree, we can utilize a level order traversal (also known as breadth-first traversal) of the tree. The approach involves:

1. Storing the largest value found on each level in an array.
2. Using a queue to assist with the level-order traversal, processing nodes level by level.

Let’s break down the solution:

### Steps:
1. **Handle Edge Cases**: If the tree is empty (i.e., the `root` is `None`), we simply return an empty list.
2. **Use a Queue**: We'll use a queue to facilitate level-order traversal. We begin by enqueueing the root node.
3. **Iterate Through Levels**: For each level of the tree, keep track of the maximum value:
   - Dequeue nodes level by level.
   - For each node, update the maximum value for that level.
   - Enqueue the left and right children of the node, if they exist.
4. **Store Results**: After processing all nodes at a given level, append the maximum value for that level to our results list.
5. **Return Results**: Once all levels have been processed, return the list of maximum values.

Here’s the implementation in Python, formatted in a way that can be directly used on the LeetCode platform:



```python
from collections import deque

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def largestValues(self, root: TreeNode) -> list[int]:
        if not root:
            return []
        
        result = []
        queue = deque([root])  # Use a deque as our queue
        
        while queue:
            level_size = len(queue)
            max_value = float('-inf')
            
            for _ in range(level_size):
                node = queue.popleft()  # Pop the node at the front of the queue
                
                # Update the maximum value for this level
                if node.val > max_value:
                    max_value = node.val
                
                # Enqueue the left and right children
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            # Append the largest value of this level to result
            result.append(max_value)
        
        return result

```

### Explanation of the Code:
- **TreeNode Class**: This is the definition for the binary tree node structure with attributes for the node's value and pointers to the left and right children.
- **Solution Class**: This class contains the method `largestValues` which returns the largest values per row of the binary tree.
- **Input Safety**: The method first checks if the `root` is `None` and returns an empty list if true.
- **Deque**: A deque is used to hold nodes as we traverse the tree in a breadth-first manner.
- **Loop Through Levels**: For each level, we determine the size of the current level using `len(queue)`, and we initialize `max_value` to negative infinity to ensure any node value will be larger.
- **Update Maximum**: As we dequeue nodes, we check if their value is larger than the current `max_value`, and if so, we update it. We also enqueue any left and right children to continue to the next level.
- **Final Result**: After processing all levels, the collected `result` list is returned containing the largest values from each row.

This solution efficiently captures the required values with a time complexity of O(N), where N is the number of nodes in the tree.

# 529. Minesweeper

### Problem Description 
Let's play the minesweeper game (Wikipedia, online game)!
You are given a 2D char matrix representing the game board. 'M' represents an unrevealed mine, 'E' represents an unrevealed empty square, 'B' represents a revealed blank square that has no adjacent (above, below, left, right, and all 4 diagonals) mines, digit ('1' to '8') represents how many mines are adjacent to this revealed square, and finally 'X' represents a revealed mine.

Now given the next click position (row and column indices) among all the unrevealed squares ('M' or 'E'), return the board after revealing this position according to the following rules:
If a mine ('M') is revealed, then the game is over - change it to 'X'.

If an empty square ('E') with no adjacent mines is revealed, then change it to revealed blank ('B') and all of its adjacent unrevealed squares should be revealed recursively.

If an empty square ('E') with at least one adjacent mine is revealed, then change it to a digit ('1' to '8') representing the number of adjacent mines.

Return the board when no more squares will be revealed.


Example 1:
Input: 
[['E', 'E', 'E', 'E', 'E'],
 ['E', 'E', 'M', 'E', 'E'],
 ['E', 'E', 'E', 'E', 'E'],
 ['E', 'E', 'E', 'E', 'E']]
Click : [3,0]
Output: 
[['B', '1', 'E', '1', 'B'],
 ['B', '1', 'M', '1', 'B'],
 ['B', '1', '1', '1', 'B'],
 ['B', 'B', 'B', 'B', 'B']]
Explanation:

Example 2:
Input: 
[['B', '1', 'E', '1', 'B'],
 ['B', '1', 'M', '1', 'B'],
 ['B', '1', '1', '1', 'B'],
 ['B', 'B', 'B', 'B', 'B']]
Click : [1,2]
Output: 
[['B', '1', 'E', '1', 'B'],
 ['B', '1', 'X', '1', 'B'],
 ['B', '1', '1', '1', 'B'],
 ['B', 'B', 'B', 'B', 'B']]
Explanation:
Note:
The range of the input matrix's height and width is [1,50].

The click position will only be an unrevealed square ('M' or 'E'), which also means the input board contains at least one clickable square.

The input board won't be a stage when game is over (some mines have been revealed).

For simplicity, not mentioned rules should be ignored in this problem. For example, you don't need to reveal all the unrevealed mines when the game is over, consider any cases that you will win the game or flag any squares.

### Solution 
 Sure! Let's solve the Minesweeper problem step by step. The goal is to manipulate a 2D character matrix based on a player's click according to the described rules. 

### Steps to Solve the Problem:
1. **Check the Clicked Position**: Determine if the clicked position is a mine ('M') or an empty square ('E').
2. **Game Over Logic**: If it's a mine, we simply change it to 'X' and return the board.
3. **Count Adjacent Mines**: If it's an empty square, count how many mines surround it (in all 8 possible directions).
4. **Update the Board**:
   - If there are adjacent mines, replace 'E' with the count (1 to 8).
   - If there are no adjacent mines, change 'E' to 'B' and recursively reveal all adjacent squares.
5. **Return the Updated Board**.

### Detailed Code Implementation:
This implementation will follow the above logic while providing a clear structure so that it can be used directly in LeetCode.



```python
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        # Directions: (row_change, column_change)
        directions = [(-1, -1), (-1, 0), (-1, 1), 
                      (0, -1),          (0, 1), 
                      (1, -1), (1, 0), (1, 1)]
        
        # Get the clicked cell position
        row, col = click
        
        # Check if the clicked cell is a mine
        if board[row][col] == 'M':
            board[row][col] = 'X'  # Game over logic
            return board
        
        # Function to count adjacent mines
        def count_mines(r, c):
            count = 0
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < len(board) and 0 <= nc < len(board[0]) and board[nr][nc] == 'M':
                    count += 1
            return count
        
        def reveal(r, c):
            # Count adjacent mines
            mine_count = count_mines(r, c)

            # If there's a mine nearby, reveal the number
            if mine_count > 0:
                board[r][c] = str(mine_count)
            else:
                # No mines adjacent, reveal a blank and recursively reveal neighbors
                board[r][c] = 'B'
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < len(board) and 0 <= nc < len(board[0]) and board[nr][nc] == 'E':
                        reveal(nr, nc)  # Recursive reveal
        
        # Start the revealing process
        reveal(row, col)

        return board

```

### Explanation of the Code:

1. **Direction Handling**: An array `directions` contains all possible moves (8 directions) which will be used to check adjacent cells.
  
2. **Game Over Check**: If the clicked cell contains 'M', it is marked as 'X'.

3. **Counting Mines**: The `count_mines` function checks each of the 8 directions to determine if there are any adjacent mines and returns that count.

4. **Recursive Reveal**: The `reveal` function checks if the current square has adjacent mines. If it does, it updates the board with that count; otherwise, it marks the square as 'B' and continues to reveal neighboring cells recursively.

5. **Final Output**: The modified board is returned at the end.

### Usage:
- To use this function, simply create an instance of the `Solution` class and call the `updateBoard` method with the game board and the clicked position. The method will return the updated board.

This implementation is efficient and adheres to the problem's constraints, ensuring correct behavior for all specified edge cases.

# 542. 01 Matrix

### Problem Description 
Given a matrix consists of 0 and 1, find the distance of the nearest 0 for each cell.

The distance between two adjacent cells is 1.


Example 1: 
Input:
[[0,0,0],
 [0,1,0],
 [0,0,0]]
Output:
[[0,0,0],
 [0,1,0],
 [0,0,0]]

Example 2: 
Input:
[[0,0,0],
 [0,1,0],
 [1,1,1]]
Output:
[[0,0,0],
 [0,1,0],
 [1,2,1]]
Note:
The number of elements of the given matrix will not exceed 10,000.

There are at least one 0 in the given matrix.

The cells are adjacent in only four directions: up, down, left and right.

### Solution 
 To solve the problem of finding the distance of the nearest 0 for each cell in a matrix filled with 0s and 1s, we can utilize a Breadth-First Search (BFS) approach. The key idea is to begin the BFS from all the cells that contain 0s simultaneously, which allows us to compute the distance to the nearest 0 for all cells in a single pass.

Here’s how we can break down the solution:

### Steps to Solve the Problem:

1. **Initialization**:
   - Create a result matrix that has the same dimensions as the input matrix.
   - Use a queue to help with the BFS. Initially, we will enqueue all the locations of the 0s in the matrix.

2. **BFS Traversal**:
   - While there are elements in the queue, pop the front element, which will give us the current cell's coordinates.
   - For each of the four possible directions (up, down, left, right), calculate the coordinates of the adjacent cell.
   - If the adjacent cell contains a 1 and has not been visited (or updated), update its distance and enqueue the cell.

3. **Output**:
   - After processing all cells, the result matrix will contain the distance from each cell to the nearest 0.

### Python Code Implementation:

Here is the complete Python code that follows the LeetCode format:



```python
from collections import deque

class Solution:
    def updateMatrix(self, mat):
        if not mat:
            return []
        
        rows, cols = len(mat), len(mat[0])
        # Initialize the result matrix with inf for 1s and 0 for 0s
        result = [[float('inf')] * cols for _ in range(rows)]
        queue = deque()

        # Add all the 0 cells to the queue and initialize their distance to 0 in the result
        for r in range(rows):
            for c in range(cols):
                if mat[r][c] == 0:
                    result[r][c] = 0
                    queue.append((r, c))

        # Directions for moving up, down, left, and right
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # Begin BFS
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                # Check if the neighbor is within bounds
                if 0 <= nr < rows and 0 <= nc < cols:
                    # If we found a shorter distance, update and enqueue the neighbor
                    if result[nr][nc] > result[r][c] + 1:
                        result[nr][nc] = result[r][c] + 1
                        queue.append((nr, nc))

        return result

```

### Explanation of the Code:

1. **Initialization**: 
   - We check if the input matrix `mat` is empty. If it is, we simply return an empty list.
   - We initialize the result matrix to hold the distances, filled initially with `float('inf')` to signify unreachable cells (1s). Cells with 0 start with a distance of 0.
   - We push all 0 cell coordinates (r, c) into our queue for BFS processing.

2. **BFS Logic**: 
   - We process cells in the queue until it's empty. For each cell, we explore all four directions.
   - If a neighboring cell has a distance greater than the current cell's distance + 1, we update its distance and add it to the queue.

3. **Output**: 
   - Finally, we return the resulting distance matrix.

This solution is efficient and handles the constraints well, producing results in O(m * n) time complexity where m and n are the dimensions of the matrix, making it optimal for the problem's constraints.

# 559. Maximum Depth of N-ary Tree

### Problem Description 
Given a n-ary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples).


Example 1:
Input: root = [1,null,3,2,4,null,5,6]
Output: 3

Example 2:
Input: root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
Output: 5

Constraints:
The depth of the n-ary tree is less than or equal to `1000`.

The total number of nodes is between `[0, 104]`.

### Solution 
 To solve the problem of finding the maximum depth of an n-ary tree, we can use a recursive depth-first search (DFS) approach. An n-ary tree is a tree where each node can have multiple children, and the input serialization is provided as a level order representation where groups of children are separated by `null`.

### Solution Explanation

1. **N-ary Tree Structure**: We first define a class for the n-ary tree node. Each node will consist of a value and a list of children.

2. **Depth Calculation**: To calculate the maximum depth:
   - If the node is `None`, we return a depth of 0.
   - If the node has no children, we return a depth of 1 (the node itself).
   - If the node has children, we compute the depth for each child, then take the maximum of those depths and add 1 (to account for the current node).

3. **Implementation**: We will define a method `maxDepth` which performs the above steps. We will also define the input structure based on the problem statement.

### Python Code

Here's the implementation in Python formatted for LeetCode:



```python
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []

class Solution:
    def maxDepth(self, root: Node) -> int:
        # If the root is None, the depth is 0
        if root is None:
            return 0
        
        # If the node has no children, its depth is 1
        if not root.children:
            return 1
        
        # Initialize max depth
        max_depth = 0
        
        # Calculate the depth for each child and find the maximum
        for child in root.children:
            child_depth = self.maxDepth(child)
            max_depth = max(max_depth, child_depth)
        
        # Return the maximum depth of children plus 1 for the current node
        return max_depth + 1

```

### Detailed Steps

1. **Node Class Definition**: 
   - `Node` class is defined with a constructor that takes a value and optionally a list of children.

2. **Solution Class**: 
   - The `Solution` class contains the method `maxDepth`.
   - The method takes a `Node` type as input.
   - We check if the node is `None`, returning 0 in that case.
   - If it has no children, we return 1.
   - If the node has children, we iterate over each child, recursively calculating the depth of each child, keeping track of the maximum depth found.

3. **Returning Depth**: 
   - At the end of processing all children, we return `max_depth + 1` to include the current node in the depth calculation.

### Edge Cases
- An empty tree (root is `None`) will return a maximum depth of 0.
- A tree where all nodes are leaf nodes will still correctly return a depth of 1 for each leaf.

### Complexity Analysis
- **Time Complexity**: O(N), where N is the number of nodes in the tree since we visit each node exactly once.
- **Space Complexity**: O(H) where H is the height of the tree, due to the recursive call stack.

This solution is efficient and straightforward for calculating the maximum depth of an n-ary tree, and is ready for use in LeetCode tests.

# 675. Cut Off Trees for Golf Event

### Problem Description 
You are asked to cut off all the trees in a forest for a golf event. The forest is represented as an `m x n` matrix. In this matrix:
`0` means the cell cannot be walked through.

`1` represents an empty cell that can be walked through.

A number greater than `1` represents a tree in a cell that can be walked through, and this number is the tree's height.

In one step, you can walk in any of the four directions: north, east, south, and west. If you are standing in a cell with a tree, you can choose whether to cut it off.

You must cut off the trees in order from shortest to tallest. When you cut off a tree, the value at its cell becomes `1` (an empty cell).

Starting from the point `(0, 0)`, return the minimum steps you need to walk to cut off all the trees. If you cannot cut off all the trees, return `-1`.

You are guaranteed that no two trees have the same height, and there is at least one tree needs to be cut off.


Example 1:
Input: forest = [[1,2,3],[0,0,4],[7,6,5]]
Output: 6
Explanation: Following the path above allows you to cut off the trees from shortest to tallest in 6 steps.


Example 2:
Input: forest = [[1,2,3],[0,0,0],[7,6,5]]
Output: -1
Explanation: The trees in the bottom row cannot be accessed as the middle row is blocked.


Example 3:
Input: forest = [[2,3,4],[0,0,5],[8,7,6]]
Output: 6

Explanation: You can follow the same path as Example 1 to cut off all the trees.

Note that you can cut off the first tree at (0, 0) before making any steps.


Constraints:
`m == forest.length`
`n == forest[i].length`
`1 <= m, n <= 50`
`0 <= forest[i][j] <= 109`

### Solution 
 To solve the problem of cutting down trees in a forest represented as an `m x n` matrix, where each tree has a height and can only be cut down in order of height, we'll follow a structured approach laid out below.

### Approach

1. **Prepare the Trees List**: Traverse the matrix to collect the coordinates and heights of all the trees in a list. We only care about cells with values greater than `1`, which represent the trees.

2. **Sort Trees by Height**: Sort the list of trees based on their heights because we need to cut down the trees from shortest to tallest.

3. **BFS for Pathfinding**: To find the minimum steps required to move from one tree to another, we'll use Breadth-First Search (BFS). BFS is suitable for this problem as it explores all possible directions uniformly and guarantees the shortest path in an unweighted grid.

4. **Calculate Steps**: For each tree, calculate the steps required to reach it from the current position, cut it down (which converts the cell into a walkable area), and update our current position to the location of the cut tree.

5. **Handle Inaccessibility**: If at any point a tree is not reachable, we return `-1`.

### Detailed Implementation

Here's the complete Python code following the above logic:



```python
from collections import deque

class Solution:
    def cutOffTree(self, forest):
        if not forest or not forest[0]:
            return -1
        
        m, n = len(forest), len(forest[0])
        trees = []
        
        # Gather all trees with their heights
        for i in range(m):
            for j in range(n):
                if forest[i][j] > 1:
                    trees.append((forest[i][j], i, j))
        
        # Sort trees by height
        trees.sort()
        
        # Initialize starting position
        start_x, start_y = 0, 0
        
        def bfs(start, target):
            queue = deque([start])
            visited = set()
            visited.add(start)
            steps = 0
            
            # Directions for N, E, S, W
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            
            while queue:
                for _ in range(len(queue)):
                    x, y = queue.popleft()
                    
                    if (x, y) == target:
                        return steps
                    
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        
                        # Check bounds and whether the cell is walkable
                        if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited and forest[nx][ny] != 0:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                
                steps += 1
            
            return -1  # Target not reachable
        
        total_steps = 0
        
        # Go through each tree in sorted order
        for height, x, y in trees:
            steps = bfs((start_x, start_y), (x, y))
            if steps == -1:
                return -1
            total_steps += steps
            start_x, start_y = x, y  # Move to the position of the cut tree
            forest[x][y] = 1  # The tree is cut down, now it's an empty cell
        
        return total_steps

```

### Explanation of the Code

1. **Collecting Trees**: The `for` loops at the start go through the matrix to create a list `trees` containing tuples of (height, row_index, column_index) for each tree.

2. **Sorting**: We sort the `trees` list so that we can process them in order of their height.

3. **Breadth-First Search (BFS)**: The function `bfs(start, target)` calculates the shortest path from the current position to the target tree’s position. It uses a queue to explore all possible movements until it finds the target tree or exhausts all possibilities.

4. **Cumulative Steps**: We accumulate `total_steps` every time we successfully cut down a tree. If any tree is unreachable, the function returns `-1`.

5. **Updating the Forest**: After cutting down a tree, we update its position in the forest to `1` indicating it is now empty.

This implementation solves the problem efficiently within the constraints provided.

# 690. Employee Importance

### Problem Description 
You are given a data structure of employee information, which includes the employee's unique id, their importance value and their direct subordinates' id.

For example, employee 1 is the leader of employee 2, and employee 2 is the leader of employee 3. They have importance value 15, 10 and 5, respectively. Then employee 1 has a data structure like [1, 15, [2]], and employee 2 has [2, 10, [3]], and employee 3 has [3, 5, []]. Note that although employee 3 is also a subordinate of employee 1, the relationship is not direct.

Now given the employee information of a company, and an employee id, you need to return the total importance value of this employee and all their subordinates.


Example 1:
Input: [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1
Output: 11
Explanation:
Employee 1 has importance value 5, and he has two direct subordinates: employee 2 and employee 3. They both have importance value 3. So the total importance value of employee 1 is 5 + 3 + 3 = 11.

Note:
One employee has at most one direct leader and may have several subordinates.

The maximum number of employees won't exceed 2000.

### Solution 
 To solve the problem of calculating the total importance of an employee and all their subordinates in a company, we canModel the employee hierarchy using a graph-like structure. Each employee is represented as an object containing their unique ID, importance value, and a list of IDs for their direct subordinates. 

We'll structure our solution as follows:

1. Parse the input to construct a mapping of employees based on their IDs. This allows for quick lookups of any employee's information.
2. Utilize Depth First Search (DFS) or Breadth First Search (BFS) to traverse through the employee and their subordinates, aggregating importance values as we go.

### Detailed Explanation of the Approach
1. **Data Structure Design**: Define a class (`Employee`) to store each employee’s details: id, importance, and a list of subordinates. 
2. **Mapping Employees**: Convert the input list into a dictionary where the keys are employee IDs and the values are their respective `Employee` objects. This allows for O(1) lookup time when querying an employee's information.
3. **Calculate Total Importance**: Implement a recursive function (DFS) or iterative function (using a stack or queue) that takes the employee ID as input, retrieves the employee's importance, and recursively retrieves the importance of each of their subordinates, summing these values.

Here’s the Python code implementing the above logic in the LeetCode solution format:



```python
class Employee:
    def __init__(self, id, importance, subordinates):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates

class Solution:
    def getImportance(self, employees: List[Employee], id: int) -> int:
        # Create a mapping from employee ID to Employee object
        emp_map = {employee.id: employee for employee in employees}

        # Function to calculate total importance using Depth First Search
        def dfs(emp_id):
            # Retrieve the employee object
            employee = emp_map[emp_id]
            # Start with the employee's own importance
            total_importance = employee.importance
            # Add the importance of each subordinate recursively
            for subordinate_id in employee.subordinates:
                total_importance += dfs(subordinate_id)
            return total_importance

        # Start DFS from the given employee ID
        return dfs(id)

# Your input data would look like below to test the function
# employees = [Employee(1, 5, [2, 3]), Employee(2, 3, []), Employee(3, 3, [])]
# sol = Solution()
# print(sol.getImportance(employees, 1))  # Output should be 11

```

### Explanation of the Code:
- **Employee Class**: This class initializes each employee instance with the attributes `id`, `importance`, and `subordinates`.
- **Solution Class**: The `getImportance` method calculates the total importance:
  - Creates a dictionary `emp_map` to allow for quick access to any employee via their ID.
  - Defines a helper function `dfs` that recursively accumulates importance values starting from the specified employee, adding the importance of each subordinate.
- **Running the Code**: You can create instances of `Employee` and call the method on an instance of `Solution` to get the total importance.

This solution is efficient and can execute within the provided constraints. It avoids global variables and uses recursion effectively to handle the nested structure of subordinates.

# 694. Number of Distinct Islands

### Problem Description 
Given a non-empty 2D array `grid` of 0's and 1's, an island is a group of `1`'s (representing land) connected 4-directionally (horizontal or vertical.)  You may assume all four edges of the grid are surrounded by water.

Count the number of distinct islands.  An island is considered to be the same as another if and only if one island can be translated (and not rotated or reflected) to equal the other.


Example 1:
11000
11000
00011
00011
Given the above grid map, return `1`.


Example 2:
11011
10000
00001
11011
Given the above grid map, return `3`.

Notice that:
11
1
and
 1
11
are considered different island shapes, because we do not consider reflection / rotation.

Note:
The length of each dimension in the given `grid` does not exceed 50.

### Solution 
 To solve the problem of counting distinct islands in a 2D grid of `0`s and `1`s, we need to identify groups of connected `1`s (land) and determine their shapes without concern for rotation or reflection. 

Here's a breakdown of our approach, followed by the complete Python code:

### Approach

1. **Explore the Island**: We'll perform a Depth-First Search (DFS) or Breadth-First Search (BFS) to explore all parts of an island starting from any given `1`. As we explore, we'll keep track of the shape of the island relative to its starting point. This way, we can represent the shape using a unique tuple.

2. **Shape Representation**: As we traverse the island, we will record the relative positions of the land (`1`s) we encounter, using a set to store these coordinates relative to the starting position. This enables us to create a standardized shape representation for each island.

3. **Storing Unique Shapes**: We will use a set to store the unique shapes of islands we find. Since sets inherently do not allow duplicates, this will help us easily count distinct shapes.

4. **Boundary Conditions**: We will ensure that we respect the boundaries of the grid to avoid indexing errors.

5. **Return Count of Unique Islands**: At the end, the size of our set will give us the number of distinct islands.

### Complete Python Code

Here’s the complete code formatted to be run directly on LeetCode:



```python
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        visited = set()
        unique_islands = set()  # To store unique island shapes
        
        def dfs(r, c, shape, base_r, base_c):
            if (r < 0 or r >= rows or c < 0 or c >= cols or 
                grid[r][c] == 0 or (r, c) in visited):
                return
            
            visited.add((r, c))  # Mark the cell as visited
            shape.append((r - base_r, c - base_c))  # Record the shape relative to the base
            
            # Explore the four directions
            dfs(r + 1, c, shape, base_r, base_c)
            dfs(r - 1, c, shape, base_r, base_c)
            dfs(r, c + 1, shape, base_r, base_c)
            dfs(r, c - 1, shape, base_r, base_c)
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1 and (i, j) not in visited:
                    shape = []  
                    dfs(i, j, shape, i, j)  # Start DFS and pass the base coordinates
                    unique_islands.add(tuple(shape))  # Use tuple to store the shape in the set
        
        return len(unique_islands)


```

### Explanation of the Code

- **Initialization**: We start by checking if the grid is empty. Then we initialize `visited` to keep track of the cells we've explored and `unique_islands` to hold the unique shapes of islands we discover.
  
- **DFS Function**: The `dfs` function takes the current coordinates and the shape we're building. It marks the current cell as visited and appends its relative position to the shape list.

- **Main Logic**: We iterate through each cell in the grid. If we encounter a `1` (land) that hasn't been visited, we create an empty list for the shape and call `dfs` to start exploring. After we fully explore an island, we convert the shape list to a tuple and add it to the `unique_islands` set.

- **Result**: Finally, we return the size of `unique_islands`, which gives us the count of distinct islands.

This approach efficiently counts distinct islands while ensuring we respect the constraints of the problem, and it should work well within the provided limits.

# 743. Network Delay Time

### Problem Description 
You are given a network of `n` nodes, labeled from `1` to `n`. You are also given `times`, a list of travel times as directed edges `times[i] = (ui, vi, wi)`, where `ui` is the source node, `vi` is the target node, and `wi` is the time it takes for a signal to travel from source to target.

We will send a signal from a given node `k`. Return the time it takes for all the `n` nodes to receive the signal. If it is impossible for all the `n` nodes to receive the signal, return `-1`.


Example 1:
Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
Output: 2

Example 2:
Input: times = [[1,2,1]], n = 2, k = 1
Output: 1

Example 3:
Input: times = [[1,2,1]], n = 2, k = 2
Output: -1

Constraints:
`1 <= k <= n <= 100`
`1 <= times.length <= 6000`
`times[i].length == 3`
`1 <= ui, vi <= n`
`ui != vi`
`0 <= wi <= 100`
All the pairs `(ui, vi)` are unique. (i.e., no multiple edges.)

### Solution 
 To solve this problem, we can use Dijkstra's algorithm to find the shortest time for a signal to reach all nodes starting from a given node `k`. The algorithm is suitable for this task because it efficiently finds the shortest paths in a weighted graph, which fits our problem description where nodes represent locations and travel times between them represent edge weights.

The key steps to implementing the solution are as follows:

1. **Graph Representation:** We will represent the network as a directed graph using an adjacency list. Each node will have a list of edges that indicate where signals can be sent and their corresponding travel times.
  
2. **Dijkstra's Algorithm:** We will use a priority queue (min-heap) to always expand the least costly path available. Starting from the node `k`, we will update the shortest time required to reach each node.

3. **Result Calculation:** After processing all reachable nodes, we will check if all nodes were reachable. If any node is unreachable, we return `-1`. Otherwise, we return the maximum travel time taken to reach any node.

Here’s the complete Python code that implements the above logic. This code follows the LeetCode submission format, where we define a function so that it can be directly run on the LeetCode platform:



```python
import heapq
from typing import List

class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # Build the graph as an adjacency list
        graph = {i: [] for i in range(1, n + 1)}
        for u, v, w in times:
            graph[u].append((v, w))  # Add the directed edge with its weight

        # Create a min-heap to act as a priority queue
        min_heap = []
        # Start with the signal sent from node k with time 0
        heapq.heappush(min_heap, (0, k))

        # Distance dictionary to hold the minimum time to reach each node
        min_time = {i: float('inf') for i in range(1, n + 1)}
        min_time[k] = 0  # Time to reach source node k is 0

        while min_heap:
            current_time, current_node = heapq.heappop(min_heap)

            # If the current time is already greater than the recorded minimum time, skip
            if current_time > min_time[current_node]:
                continue

            # Explore neighbors
            for neighbor, travel_time in graph[current_node]:
                time_to_neighbor = current_time + travel_time
                if time_to_neighbor < min_time[neighbor]:  # Found a shorter path
                    min_time[neighbor] = time_to_neighbor
                    heapq.heappush(min_heap, (time_to_neighbor, neighbor))

        # Calculate the maximum time to reach all nodes
        max_time = max(min_time.values())

        return max_time if max_time != float('inf') else -1

```

### Explanation:

- **Graph Construction (Lines 7-9):**
  We create a dictionary `graph` where each key represents a node and the corresponding value is a list of tuples (target, weight) indicating connected nodes and travel times.

- **Min-Heap Initialization (Lines 12-14):**
  We initialize our min-heap with a tuple containing the time to reach node `k` (which is initially 0) and the node itself.

- **Dijkstra's Algorithm (Lines 16-28):**
  We repeatedly pop the smallest time entry from the heap. For each node, we check its neighbors and calculate the cumulative time to reach them. If we find a shorter path to a neighbor, we update its minimum time and push this updated information back into the heap.

- **Final Time Calculation (Lines 30-33):**
  After processing all reachable nodes, we determine the maximum travel time from the `min_time` values. If all nodes are reachable, we return this time; if any node remains unreachable (inf), we return `-1`.

This implementation efficiently handles the problem with a time complexity of O((E + V) log V), where E is the number of edges and V is the number of vertices (nodes), making it suitable for the problem's constraints.

# 752. Open the Lock

### Problem Description 
You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: `'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'`. The wheels can rotate freely and wrap around: for example we can turn `'9'` to be `'0'`, or `'0'` to be `'9'`. Each move consists of turning one wheel one slot.

The lock initially starts at `'0000'`, a string representing the state of the 4 wheels.

You are given a list of `deadends` dead ends, meaning if the lock displays any of these codes, the wheels of the lock will stop turning and you will be unable to open it.

Given a `target` representing the value of the wheels that will unlock the lock, return the minimum total number of turns required to open the lock, or -1 if it is impossible.


Example 1:
Input: deadends = ["0201","0101","0102","1212","2002"], target = "0202"
Output: 6
Explanation:
A sequence of valid moves would be "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202".

Note that a sequence like "0000" -> "0001" -> "0002" -> "0102" -> "0202" would be invalid,
because the wheels of the lock become stuck after the display becomes the dead end "0102".


Example 2:
Input: deadends = ["8888"], target = "0009"
Output: 1
Explanation:
We can turn the last wheel in reverse to move from "0000" -> "0009".


Example 3:
Input: deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"], target = "8888"
Output: -1
Explanation:
We can't reach the target without getting stuck.


Example 4:
Input: deadends = ["0000"], target = "8888"
Output: -1

Constraints:
`1 <= deadends.length <= 500`
`deadends[i].length == 4`
`target.length == 4`
target will not be in the list `deadends`.

`target` and `deadends[i]` consist of digits only.

### Solution 
 To solve the problem of unlocking a lock represented with wheels that can rotate, we can apply a breadth-first search (BFS) algorithm. This algorithm is suitable as we are looking for the shortest path (in terms of moves) to reach the `target` from the initial state of the lock `'0000'`, while avoiding any dead ends.

### Understanding the Problem
1. **Lock Representation**: The lock can be represented as a 4-digit string where each digit can rotate from `'0'` to `'9'`.
2. **Dead ends**: A list of codes where if the lock reaches any of these states, it becomes stuck.
3. **Target**: The desired configuration of the lock that we want to reach from the initial state.
4. **Objective**: Calculate the minimum number of moves required to open the lock. If it is impossible, return `-1`.

### Approach
1. **Breadth-First Search (BFS)**:
   - We start from the initial state `'0000'` and explore all possible moves.
   - For each state, we will generate potential next states by incrementing or decrementing each of the four digits.
   - Use a queue to manage the states to explore and a set to keep track of visited states to prevent cycling through already checked states.
2. **Handling Dead Ends**: Before adding a new state to the queue, we will check if it is in the list of dead ends or has been visited.
3. **Terminate Condition**: If we reach the `target` state, we will return the number of moves taken to reach there.

### Code Implementation
Here's the Python code that implements the above approach:



```python
from collections import deque

class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        # Initial state
        start = '0000'
        
        # Store deadends in a set for O(1) lookup
        deadend_set = set(deadends)
        
        # Check if the start itself is a deadend
        if start in deadend_set:
            return -1
        
        # A queue for BFS
        queue = deque([(start, 0)])  # (current_state, moves)
        visited = set([start])
        
        while queue:
            current_state, moves = queue.popleft()
            
            # Check if we reached the target
            if current_state == target:
                return moves
            
            # Generate all possible next moves
            for i in range(4):
                for direction in [-1, 1]:  # -1 for decrement, +1 for increment
                    next_digit = (int(current_state[i]) + direction) % 10
                    next_state = current_state[:i] + str(next_digit) + current_state[i+1:]
                    
                    # If the next state is a deadend or already visited, skip it
                    if next_state not in visited and next_state not in deadend_set:
                        visited.add(next_state)
                        queue.append((next_state, moves + 1))

        # If we exhausted all options and didn't find the target
        return -1

```

### Explanation of the Code
- The code defines a `Solution` class with a method `openLock`.
- It initializes the BFS queue and checks if the starting state is a dead end.
- The main part of the algorithm loops over all possible next states derived from the current state. It modifies each wheel of the lock by either increasing or decreasing its digit and then checks if the new state is valid (not a dead end, not visited).
- If the target state is reached during the BFS, it returns the number of moves taken. If the queue is exhausted and the target is not found, it returns `-1`.

### Conclusion
This approach efficiently explores all possibilities using BFS, ensuring that we find the shortest path to the target while avoiding dead ends. The complexity of BFS in this case, limited by the maximum number of possible states (10,000), ensures it runs within acceptable time limits for the given constraints.

# 773. Sliding Puzzle

### Problem Description 
On a 2x3 `board`, there are 5 tiles represented by the integers 1 through 5, and an empty square represented by 0.

A move consists of choosing `0` and a 4-directionally adjacent number and swapping it.

The state of the board is solved if and only if the `board` is `[[1,2,3],[4,5,0]].`
Given a puzzle board, return the least number of moves required so that the state of the board is solved. If it is impossible for the state of the board to be solved, return -1.


Examples:
Input: board = [[1,2,3],[4,0,5]]
Output: 1
Explanation: Swap the 0 and the 5 in one move.

Input: board = [[1,2,3],[5,4,0]]
Output: -1
Explanation: No number of moves will make the board solved.

Input: board = [[4,1,2],[5,0,3]]
Output: 5
Explanation: 5 is the smallest number of moves that solves the board.

An example path:
After move 0: [[4,1,2],[5,0,3]]
After move 1: [[4,1,2],[0,5,3]]
After move 2: [[0,1,2],[4,5,3]]
After move 3: [[1,0,2],[4,5,3]]
After move 4: [[1,2,0],[4,5,3]]
After move 5: [[1,2,3],[4,5,0]]
Input: board = [[3,2,4],[1,5,0]]
Output: 14
Note:
`board` will be a 2 x 3 array as described above.

`board[i][j]` will be a permutation of `[0, 1, 2, 3, 4, 5]`.

### Solution 
 To solve this problem, we can utilize the Breadth-First Search (BFS) algorithm. The BFS algorithm is particularly suitable for this problem because it explores all possible configurations of the board in increasing order of depth, which directly corresponds to the number of moves taken to reach a configuration. 

### Steps to Solve the Problem:

1. **Representation of the Board**: We'll convert the 2x3 board into a single string for easier manipulation and comparison.
2. **Identify Possible States**: We will generate new board configurations by swapping the `0` space with its adjacent tiles.
3. **Use BFS for Exploration**: We'll maintain a queue to explore each configuration. Each state will keep track of the number of moves taken to reach it.
4. **Check for the Goal State**: The configuration `[[1, 2, 3], [4, 5, 0]]` will be our goal, represented as the string `123450`.
5. **Utilize a Set for Visited States**: To avoid revisiting configurations, we'll maintain a set of visited states.
6. **Return Result**: If we reach the goal state, we'll return the count of moves taken. If there are no configurations left to explore and we have not reached the goal, we'll return -1.

### Python Code:

Here's the complete Python code based on the above plan. It is formatted for submission to LeetCode.



```python
from collections import deque

class Solution:
    def slidingPuzzle(self, board):
        # Convert board to string
        start = ''.join(map(str, sum(board, [])))
        target = '123450'
        if start == target:
            return 0
        
        # Directions for possible moves
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        # Indexes of the blank space (0)
        zero_index = start.index('0')
        queue = deque([(start, zero_index, 0)])  # state, position of 0, steps
        visited = set()
        visited.add(start)
        
        while queue:
            current_state, zero_pos, steps = queue.popleft()
            
            # Calculate the row and column of the zero position
            row, col = divmod(zero_pos, 3)
            
            # Explore all possible movements
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check if the new position is valid
                if 0 <= new_row < 2 and 0 <= new_col < 3:
                    new_zero_pos = new_row * 3 + new_col
                    # Create new state by swapping
                    new_state_list = list(current_state)
                    # Swap zero with the number in the new position
                    new_state_list[zero_pos], new_state_list[new_zero_pos] = new_state_list[new_zero_pos], new_state_list[zero_pos]
                    new_state = ''.join(new_state_list)

                    if new_state == target:
                        return steps + 1   # Return the number of steps taken
                    
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_state, new_zero_pos, steps + 1))
        
        return -1  # If we exhaust all options without finding the target

```

### Explanation of the Code:

- **Input Handling**: The input board is converted into a string to represent the state easily.
- **The BFS Queue**: A deque is used for efficient pop from the left side.
- **Zero Movement**: For each configuration, we check all four possible moves (up, down, left, right) and swap the `0` with the adjacent number if the move is valid.
- **Goal Check**: We check if we've reached the configuration that solves the puzzle.
- **Visited States**: We store each unique configuration to prevent cycles.
- **Result**: We return the number of steps if the target is found, or -1 if we exhaust all options.

This BFS approach guarantees that we find the shortest path because we explore every configuration at a given distance before moving deeper. It's efficient and straightforward, perfectly suited to the constraints of this problem.

# 785. Is Graph Bipartite?

### Problem Description 
There is an undirected graph with `n` nodes, where each node is numbered between `0` and `n - 1`. You are given a 2D array `graph`, where `graph[u]` is an array of nodes that node `u` is adjacent to. More formally, for each `v` in `graph[u]`, there is an undirected edge between node `u` and node `v`. The graph has the following properties:
There are no self-edges (`graph[u]` does not contain `u`).

There are no parallel edges (`graph[u]` does not contain duplicate values).

If `v` is in `graph[u]`, then `u` is in `graph[v]` (the graph is undirected).

The graph may not be connected, meaning there may be two nodes `u` and `v` such that there is no path between them.

A graph is bipartite if the nodes can be partitioned into two independent sets `A` and `B` such that every edge in the graph connects a node in set `A` and a node in set `B`.

Return `true` if and only if it is bipartite.


Example 1:
Input: graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
Output: false
Explanation: There is no way to partition the nodes into two independent sets such that every edge connects a node in one and a node in the other.


Example 2:
Input: graph = [[1,3],[0,2],[1,3],[0,2]]
Output: true
Explanation: We can partition the nodes into two sets: {0, 2} and {1, 3}.


Constraints:
`graph.length == n`
`1 <= n <= 100`
`0 <= graph[u].length < n`
`0 <= graph[u][i] <= n - 1`
`graph[u]` does not contain `u`.

All the values of `graph[u]` are unique.

If `graph[u]` contains `v`, then `graph[v]` contains `u`.

### Solution 
 To determine whether an undirected graph is bipartite, we can use a graph traversal method such as Depth-First Search (DFS) or Breadth-First Search (BFS). The basic idea is to attempt to color the graph using two colors, where no two adjacent nodes share the same color. If we can successfully color the graph in this manner, it is bipartite; otherwise, it is not.

Here's how we can implement this in Python following the LeetCode solution format:



```python
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)  # Number of nodes in the graph
        colors = [-1] * n  # -1 indicates uncolored

        def bfs(node: int) -> bool:
            queue = deque([node])
            colors[node] = 0  # Start coloring with color 0
            
            while queue:
                current = queue.popleft()
                for neighbor in graph[current]:
                    # If the neighbor hasn't been colored yet
                    if colors[neighbor] == -1:
                        # Color it with the opposite color
                        colors[neighbor] = 1 - colors[current]
                        queue.append(neighbor)
                    # If the neighbor has the same color as the current node
                    elif colors[neighbor] == colors[current]:
                        return False  # Not bipartite
            return True

        for i in range(n):
            if colors[i] == -1:  # If the node is uncolored, start a BFS
                if not bfs(i):
                    return False  # If any component is not bipartite

        return True  # All components checked and are bipartite


```

### Explanation of the Code:

1. **Initialization**: 
   - We start by determining the number of nodes `n` and creating a color array initialized to `-1`, which denotes that the node is uncolored.

2. **BFS Function**:
   - The `bfs` function takes a starting node and attempts to color the graph using queues (BFS).
   - It starts coloring the given node with color `0`.
   - For each node processed, it looks at its neighbors:
     - If a neighbor hasn't been colored, it colors it with the opposite color (if current is `0`, we color neighbor as `1`).
     - If a neighbor has already been colored and its color is the same as the current node, we return `False` because it indicates that the graph is not bipartite.

3. **Main Loop**:
   - We iterate through all nodes in the graph. If a node has not been visited (still marked as `-1`), we call the `bfs` function.
   - If any call to `bfs` returns `False`, we conclude that the graph is not bipartite and return `False`.

4. **Final Check**:
   - If all components can be colored successfully, we return `True`, indicating the graph is bipartite.

This solution operates efficiently with a time complexity of O(V + E), where V is the number of vertices and E is the number of edges, which is suitable for the input constraints provided.

# 787. Cheapest Flights Within K Stops

### Problem Description 
There are `n` cities connected by `m` flights. Each flight starts from city `u` and arrives at `v` with a price `w`.

Now given all the cities and flights, together with starting city `src` and the destination `dst`, your task is to find the cheapest price from `src` to `dst` with up to `k` stops. If there is no such route, output `-1`.


Example 1:
Input: 
n = 3, edges = [[0,1,100],[1,2,100],[0,2,500]]
src = 0, dst = 2, k = 1
Output: 200
Explanation: 
The graph looks like this:
The cheapest price from city `0` to city `2` with at most 1 stop costs 200, as marked red in the picture.


Example 2:
Input: 
n = 3, edges = [[0,1,100],[1,2,100],[0,2,500]]
src = 0, dst = 2, k = 0
Output: 500
Explanation: 
The graph looks like this:
The cheapest price from city `0` to city `2` with at most 0 stop costs 500, as marked blue in the picture.


Constraints:
The number of nodes `n` will be in range `[1, 100]`, with nodes labeled from `0` to `n`` - 1`.

The size of `flights` will be in range `[0, n * (n - 1) / 2]`.

The format of each flight will be `(src, ``dst``, price)`.

The price of each flight will be in the range `[1, 10000]`.

`k` is in the range of `[0, n - 1]`.

There will not be any duplicated flights or self cycles.

### Solution 
 To solve the problem of finding the cheapest flight from a source city to a destination city with up to `k` stops, we can utilize a graph traversal technique. A good approach for this problem is to use modified Dijkstra's algorithm (also known as a priority queue approach) with a twist that considers the number of stops.

Here's a detailed step-by-step explanation followed by the complete solution:

### Problem Breakdown

1. **Graph Representation**: We can represent the cities and flights as a directed graph where each city is a node, and each flight is a directed edge with an associated weight (the cost of the flight).

2. **Input Parameters**:
   - `n`: The number of cities (vertices).
   - `edges`: A list of flights where each flight is represented as `[u, v, w]` meaning there is a flight from city `u` to city `v` with cost `w`.
   - `src`: The starting city (node).
   - `dst`: The destination city (node).
   - `k`: The maximum number of stops allowed.

3. **Algorithm**:
   - Use a priority queue (min-heap) to repeatedly explore the cheapest flight option at any given moment.
   - Keep track of the current city, the total cost to reach that city, and the number of stops made.
   - If the current city is the destination and the number of stops made is within the allowed limit, return the cost.
   - If we expand a city, only push new states into the queue if they respect the maximum stops constraint.

4. **Edge Cases**: If there are no possible routes from `src` to `dst` within the allowed stops, return -1.

### Python Solution

Below is the Python code implementing the described approach, following the format suitable for LeetCode.



```python
import heapq
from collections import defaultdict

class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        # Create a graph from the flights
        graph = defaultdict(list)
        for u, v, w in flights:
            graph[u].append((v, w))
        
        # Min-heap for the (cost, city, stops made)
        min_heap = [(0, src, 0)]  # (cost, current city, stops made)
        
        # While there are items in the min-heap
        while min_heap:
            cost, city, stops = heapq.heappop(min_heap)
            
            # If we reach the destination and stops are valid, return the cost
            if city == dst:
                return cost
            
            # If there are still stops left, explore the neighbors
            if stops <= k:
                for neighbor, price in graph[city]:
                    heapq.heappush(min_heap, (cost + price, neighbor, stops + 1))
        
        # If we exhaust the heap without finding the destination
        return -1

```

### Explanation of the Code

- **Graph Construction**: We utilize a `defaultdict` to create an adjacency list from the `flights` input.

- **Priority Queue (`min_heap`)**: This is initialized with a tuple containing the total cost (`0`), the `src` city, and the number of stops (`0`).

- **Main Loop**: The loop continues until the priority queue is empty. We:
  - Pop the smallest cost entry.
  - Check if the current city is the destination. If so, return the cost.
  - If the number of stops made is less than or equal to `k`, explore all connected cities (neighbors) from the current city and push new states into the priority queue with updated costs and stops.

- **No Valid Path Found**: If we exhaust all possibilities and never reach the destination, we return `-1`.

This code effectively balances efficient cost management with the number of stops to find the optimal route.

# 815. Bus Routes

### Problem Description 
You are given an array `routes` representing bus routes where `routes[i]` is a bus route that the `ith` bus repeats forever.

For example, if `routes[0] = [1, 5, 7]`, this means that the `0th` bus travels in the sequence `1 -> 5 -> 7 -> 1 -> 5 -> 7 -> 1 -> ...` forever.

You will start at the bus stop `source` (You are not on any bus initially), and you want to go to the bus stop `target`. You can travel between bus stops by buses only.

Return the least number of buses you must take to travel from `source` to `target`. Return `-1` if it is not possible.


Example 1:
Input: routes = [[1,2,7],[3,6,7]], source = 1, target = 6
Output: 2
Explanation: The best strategy is take the first bus to the bus stop 7, then take the second bus to the bus stop 6.


Example 2:
Input: routes = [[7,12],[4,5,15],[6],[15,19],[9,12,13]], source = 15, target = 12
Output: -1

Constraints:
`1 <= routes.length <= 500`.

`1 <= routes[i].length <= 105`
All the values of `routes[i]` are unique.

`sum(routes[i].length) <= 105`
`0 <= routes[i][j] < 106`
`0 <= source, target < 106`

### Solution 
 To solve the bus route problem where we need to determine the least number of buses to take from a source stop to a target stop, we can utilize a breadth-first search (BFS) approach. This is effective because we want to find the shortest path in terms of the number of bus transfers needed.

### Detailed Explanation:

1. **Understanding the Problem**:
   - We have several bus routes, each defined as a list of stops.
   - We need to get from one stop (`source`) to another stop (`target`), using these bus routes.
   - We are interested in the minimum number of bus transfers needed to get to the target.

2. **BFS Approach**:
   - We will treat each bus route as a "node" and each bus stop as a "connection" in our graph traversal. 
   - We will use BFS because it naturally explores all stops reachable with the minimum number of transfers first.

3. **Data Structures**:
   - A queue to manage the BFS.
   - A set to track visited bus stops and to avoid re-processing.
   - A dictionary to map each bus stop to the bus routes that service that stop for quick access.

4. **Algorithm**:
   - If the `source` is the same as `target`, we can return 0 immediately since no buses are needed.
   - Build a map of stops to the routes that service them.
   - Initiate BFS from the `source`, exploring all reachable bus stops via the available bus routes.
   - Keep track of the number of bus transfers made while processing stops.
   - If we reach the `target` during the traversal, we return the number of transfers.

5. **Edge Cases**:
   - If `source` or `target` are unreachable from the provided bus routes, we return `-1`.

### Python Code:

Here’s how the solution can be structured to run on LeetCode:



```python
from collections import defaultdict, deque

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:
        if source == target:
            return 0
        
        # Create a map from bus stops to bus routes
        stop_to_routes = defaultdict(set)
        for bus_index, route in enumerate(routes):
            for stop in route:
                stop_to_routes[stop].add(bus_index)

        # BFS setup
        queue = deque([source])
        visited_stops = set([source])
        visited_routes = set()
        buses_taken = 0

        while queue:
            buses_taken += 1
            for _ in range(len(queue)):
                current_stop = queue.popleft()
                
                # Check all routes that use the current stop
                for bus_index in stop_to_routes[current_stop]:
                    if bus_index in visited_routes:
                        continue

                    # Mark this route as visited
                    visited_routes.add(bus_index)

                    # Check all stops on this bus route
                    for next_stop in routes[bus_index]:
                        if next_stop == target:
                            return buses_taken
                        if next_stop not in visited_stops:
                            visited_stops.add(next_stop)
                            queue.append(next_stop)

        return -1  # If we complete the BFS without finding the target

```

### Explanation of the Code:

- We create a `defaultdict` to map each stop to the bus routes that can reach it.
- We initiate the BFS from the `source` and keep track of stops visited to avoid cycles.
- Each level of BFS corresponds to taking another bus, so we increase the count of `buses_taken` as we move deeper into the BFS.
- If we find the `target` stop, we return the number of buses taken.
- If the BFS completes without finding the target, it returns `-1`.

This code is structured to be optimal and adheres to the constraints defined in the problem.

# 827. Making A Large Island

### Problem Description 
You are given an `n x n` binary matrix `grid`. You are allowed to change at most one `0` to be `1`.

Return the size of the largest island in `grid` after applying this operation.

An island is a 4-directionally connected group of `1`s.


Example 1:
Input: grid = [[1,0],[0,1]]
Output: 3
Explanation: Change one 0 to 1 and connect two 1s, then we get an island with area = 3.


Example 2:
Input: grid = [[1,1],[1,0]]
Output: 4
Explanation: Change the 0 to 1 and make the island bigger, only one island with area = 4.


Example 3:
Input: grid = [[1,1],[1,1]]
Output: 4
Explanation: Can't change any 0 to 1, only one island with area = 4.


Constraints:
`n == grid.length`
`n == grid[i].length`
`1 <= n <= 500`
`grid[i][j]` is either `0` or `1`.

### Solution 
 To solve the problem of finding the size of the largest island in a binary matrix after changing at most one `0` to `1`, we can implement the following approach using Depth-First Search (DFS) or Breadth-First Search (BFS).

### Solution Explanation:

1. **Understanding the Problem**:
   - An island is defined as a group of `1`s connected in 4 directions (up, down, left, right).
   - You need to maximize the size of the island by changing one `0` to `1`.

2. **Algorithm Steps**:
   - First, we will traverse the grid to find all existing islands and compute their sizes.
   - We'll keep track of the sizes of these islands in a dictionary where the key is the island's identifier (a unique index).
   - By using DFS, each time we visit a `1`, we can compute the area (size) of that island and store it.
   - Next, we will iterate again over the grid, and whenever we encounter a `0`, we will:
     - Check the unique identifiers of adjacent islands and calculate the potential size of a new island by summing the sizes of these adjacent islands plus one (for the `0` we're changing to `1`).
   - The answer will be the largest size obtained from the above iterations.

3. **Implementation**: Below is the Python code that adheres to LeetCode submission requirements.



```python
class Solution:
    def largestIsland(self, grid):
        n = len(grid)
        island_id = [[-1] * n for _ in range(n)]
        island_size = {}
        
        def dfs(x, y, index):
            # Base case: Check boundaries and whether it's part of the island
            if x < 0 or x >= n or y < 0 or y >= n or grid[x][y] == 0 or island_id[x][y] != -1:
                return 0
            
            island_id[x][y] = index
            size = 1  # Current cell
            
            # Visit all 4 directions
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                size += dfs(x + dx, y + dy, index)
                
            return size
        
        index = 0
        
        # First pass: identify islands and calculate sizes
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1 and island_id[i][j] == -1:
                    size = dfs(i, j, index)
                    island_size[index] = size
                    index += 1
                    
        max_island_size = max(island_size.values(), default=0)  # Maximum island size without changes
        
        # Set of island indices around each '0'
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    seen = set()  # To avoid double counting the same island
                    potential_size = 1  # Including the current cell (0 -> 1)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < n and 0 <= nj < n and island_id[ni][nj] != -1:
                            seen.add(island_id[ni][nj])
                    
                    for island in seen:
                        potential_size += island_size[island]
                    
                    max_island_size = max(max_island_size, potential_size)
        
        return max_island_size

# Example Usage
# solution = Solution()
# print(solution.largestIsland([[1, 0], [0, 1]]))  # Output: 3

```

### Explanation of the Code:
- The `dfs` function is responsible for marking the visited cells of each island and calculating its size.
- The first loop identifies each island in the grid and records its size.
- The second loop evaluates each `0` in the grid, checks adjacent `1`s, and calculates the potential size of the island that could form if that `0` were changed to `1`.
- Finally, we return the maximum island size found.

### Efficiency:
- The solution runs in O(n^2) time, where n is the dimension of the square grid, which is efficient enough given the problem constraints.

# 847. Shortest Path Visiting All Nodes

### Problem Description 
An undirected, connected graph of N nodes (labeled `0, 1, 2, ..., N-1`) is given as `graph`.

`graph.length = N`, and `j != i` is in the list `graph[i]` exactly once, if and only if nodes `i` and `j` are connected.

Return the length of the shortest path that visits every node. You may start and stop at any node, you may revisit nodes multiple times, and you may reuse edges.


Example 1:
Input: [[1,2,3],[0],[0],[0]]
Output: 4
Explanation: One possible path is [1,0,2,0,3]

Example 2:
Input: [[1],[0,2,4],[1,3,4],[2],[1,2]]
Output: 4
Explanation: One possible path is [0,1,4,2,3]
Note:
`1 <= graph.length <= 12`
`0 <= graph[i].length < graph.length`

### Solution 
 To solve this problem, we will apply the concept of Depth-First Search (DFS) combined with dynamic programming using bit masking. The task is to find the length of the shortest path that visits all nodes in an undirected graph where we can start and stop at any node.

Here’s a step-by-step breakdown of how we will approach the problem:

### Problem Breakdown

1. **Graph Representation**: The graph is provided as an adjacency list. Each index represents a node, and the list at that index contains all nodes connected to it.

2. **Bitmasking**: We will use a bitmask to represent the set of nodes that have been visited. If there are N nodes, the bitmask can be represented as an integer from `0` to `(1 << N) - 1`, where each bit in the integer indicates whether a specific node has been visited.

3. **Dynamic Programming Table**: We will maintain a DP table where `dp[mask][i]` represents the minimum path length to visit all nodes represented in `mask`, ending at node `i`.

4. **DFS for Exploration**: We will use DFS to explore all paths from each node and use our DP table to minimize the path length. This will require traversing through all nodes and maintaining the state of visited nodes.

### Implementation

Now let's translate this idea into Python code. Below is the function formatted according to LeetCode's requirements.



```python
class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        from collections import deque
        
        n = len(graph)
        
        # All possible states (visited nodes) for n nodes.
        target_mask = (1 << n) - 1
        
        # This list will hold the minimum distance to cover each state ending at each node.
        dp = [[float('inf')] * n for _ in range(1 << n)]
        
        # Start by initializing the state where each node is visited individually.
        for i in range(n):
            dp[1 << i][i] = 0
        
        # We will perform a BFS from all starting nodes.
        queue = deque([(1 << i, i) for i in range(n)])  # (visited_mask, current_node)

        while queue:
            mask, node = queue.popleft()
            
            # Explore neighbors (next nodes)
            for neighbor in graph[node]:
                new_mask = mask | (1 << neighbor)
                
                # If we found a shorter way to reach this new state ending at 'neighbor'
                if dp[new_mask][neighbor] > dp[mask][node] + 1:
                    dp[new_mask][neighbor] = dp[mask][node] + 1
                    queue.append((new_mask, neighbor))
        
        # Find the minimum distance to visit all nodes
        return min(dp[target_mask][i] for i in range(n))

```

### Explanation of the Code

1. **Initialization**: We initialize the DP table, where `dp[mask][i]` is set to infinity. We then populate the DP table for single node visits.

2. **BFS Queue**: We use BFS from each node, represented by their bitmask and current node index. The queue is initialized with each node as the starting point.

3. **Exploration**: For each node, we check all its neighbors. If moving to a neighbor results in a new state (new mask) with a shorter path, we update the DP table and add this state to our queue.

4. **Result Computation**: After exploring all possibilities, we compute the result by finding the minimum path length from the DP table that visits all nodes.

This solution is optimized to handle the maximum constraint of `N <= 12`, making the overall complexity manageable due to the limited size of the problem space.

# 854. K-Similar Strings

### Problem Description 
Strings `s1` and `s2` are `k`-similar (for some non-negative integer `k`) if we can swap the positions of two letters in `s1` exactly `k` times so that the resulting string equals `s2`.

Given two anagrams `s1` and `s2`, return the smallest `k` for which `s1` and `s2` are `k`-similar.


Example 1:
Input: s1 = "ab", s2 = "ba"
Output: 1

Example 2:
Input: s1 = "abc", s2 = "bca"
Output: 2

Example 3:
Input: s1 = "abac", s2 = "baca"
Output: 2

Example 4:
Input: s1 = "aabc", s2 = "abca"
Output: 2

Constraints:
`1 <= s1.length <= 20`
`s2.length == s1.length`
`s1` and `s2` contain only lowercase letters from the set `{'a', 'b', 'c', 'd', 'e', 'f'}`.

`s2` is an anagram of `s1`.

### Solution 
 To solve the problem of finding the minimum number of swaps (`k`) needed to make two anagrams `s1` and `s2` equal, we can use a breadth-first search (BFS) approach. 

### Detailed Explanation

1. **Understanding k-similarity**: Two strings being `k`-similar means that you can transform one string into the other by performing exactly `k` swaps of character positions.

2. **Using BFS**: We will use BFS to explore the different states of the string `s1`. Every time we swap two characters, we create a new state. The goal is to find the shortest path (in terms of swaps) from `s1` to `s2`.

3. **Queue for BFS**: We will use a queue to hold our current states of `s1` along with the number of swaps taken to reach that state.

4. **Visited States**: To avoid processing the same state multiple times, we keep track of visited strings.

5. **Valid Swaps**: Since `s1` and `s2` are anagrams, the only swaps we need to consider are those that help align `s1` closer to `s2`. We will look for the first position where `s1` and `s2` differ, then try swapping it with others to create new valid states.

6. **Stopping Condition**: We will stop when we find `s1` equal to `s2`.

Now, let’s implement this logic in Python.

### Python Code

Here is the implementation following the LeetCode solution format:



```python
from collections import deque

class Solution:
    def kSimilarity(self, s1: str, s2: str) -> int:
        if s1 == s2:
            return 0

        queue = deque([(s1, 0)])  # (current string state, number of swaps)
        visited = {s1}  # keep track of visited states

        while queue:
            current, swaps = queue.popleft()  # get the current state and swap count

            # Find first index where characters differ
            for i in range(len(s1)):
                if current[i] != s2[i]:
                    break
            
            # Try swapping the current character with characters at all other indices
            for j in range(i + 1, len(s1)):
                if current[j] == s2[i]:  # We can only swap to match the target character in s2
                    # Swap them to create a new state
                    next_state = list(current)
                    next_state[i], next_state[j] = next_state[j], next_state[i]
                    next_state = ''.join(next_state)

                    if next_state == s2:  # If we made it to s2
                        return swaps + 1
                    
                    if next_state not in visited:  # If this state has not been visited
                        visited.add(next_state)
                        queue.append((next_state, swaps + 1))
        
        return -1  # This line will never be reached if inputs are valid (since s2 is an anagram of s1)

```

### Explanation of the Code

- The BFS begins by initializing the queue with the original string `s1` and zero swaps.
- The algorithm checks each character one-at-a-time to find the first index where the strings differ.
- For each mismatch found, it tries to swap that character with every character that matches the expected character in `s2`.
- Each new configuration is added to the queue if it hasn't been visited yet.
- When a configuration matches `s2`, it returns the total swaps taken to reach that configuration.

### Complexity Analysis

- The time complexity is `O(n^2 * n!)`, where `n` is the length of the string (`s1` or `s2`), due to the combinations of character swaps and BFS traversal.
- The space complexity is `O(n!)` in the worst case to store the visited states.

Overall, this approach efficiently finds the minimum number of swaps needed to transform `s1` into `s2`.

# 863. All Nodes Distance K in Binary Tree

### Problem Description 
We are given a binary tree (with root node `root`), a `target` node, and an integer value `K`.

Return a list of the values of all nodes that have a distance `K` from the `target` node.  The answer can be returned in any order.


Example 1:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, K = 2
Output: [7,4,1]
Explanation: 
The nodes that are a distance 2 from the target node (with value 5)
have values 7, 4, and 1.

Note that the inputs "root" and "target" are actually TreeNodes.

The descriptions of the inputs above are just serializations of these objects.

Note:
The given tree is non-empty.

Each node in the tree has unique values `0 <= node.val <= 500`.

The `target` node is a node in the tree.

`0 <= K <= 1000`.

### Solution 
 To solve the problem of finding all the nodes that are a distance `K` from a given `target` node in a binary tree, we can approach the problem by treating it as a graph problem. The binary tree can be considered as an undirected graph where each node is connected to its child nodes as well as its parent node.

Here’s a step-by-step explanation of how we will solve this problem:

1. **Graph Representation**: Convert the binary tree into an undirected graph using a depth-first search (DFS) algorithm. While traversing the tree, keep track of the parent node for each child node. This allows us to easily move back to the parent later.

2. **BFS for Distance Calculation**: Use a breadth-first search (BFS) starting from the `target` node to find all nodes that are `K` distance away. BFS is suitable here because it explores all nodes at the present depth level before moving onto nodes at the next depth level.

3. **Result Collection**: During the BFS, we'll maintain a set of visited nodes to avoid processing any node more than once. We will also collect the nodes that we encounter exactly `K` steps away from the target.

Here’s how we can implement that in Python, adhering to the LeetCode format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from collections import defaultdict, deque

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, K: int) -> List[int]:
        # Step 1: Create a graph representation of the binary tree
        graph = defaultdict(list)

        # DFS to build the graph
        def dfs(node, parent):
            if node:
                if parent:
                    graph[node].append(parent)  # connect to parent
                    graph[parent].append(node)  # connect back to child
                dfs(node.left, node)
                dfs(node.right, node)

        # Step 1: Build the graph from the binary tree
        dfs(root, None)

        # Step 2: Use BFS to find all nodes at distance K from the target
        queue = deque([target])
        visited = set([target])
        distance = 0
        result = []

        while queue:
            if distance == K:
                # At distance K, collect all node values
                result.extend(node.val for node in queue)
                break
            
            # Process current level
            for _ in range(len(queue)):
                current_node = queue.popleft()
                # Explore all neighbors (children and parent)
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            distance += 1

        return result

```

### Explanation of the Code:
- **TreeNode Class**: This is a standard definition of a tree node with properties for the value (`val`) and pointers to the left and right children (`left` and `right`).
  
- **Graph Construction**: In the `dfs` function, we add an edge between the node and its parent to create an undirected graph representation of the binary tree.

- **Breadth-First Search (BFS)**: We use a queue to explore nodes starting from the `target`. We keep track of the distance with a simple counter. When we reach the desired distance `K`, we collect the values of all nodes at that level.

- **Result Return**: Finally, we return the collected values as the result.

This code can be directly run in a LeetCode environment while creating the nodes of the binary tree and providing the `target` and `K` as input.

# 864. Shortest Path to Get All Keys

### Problem Description 
We are given a 2-dimensional `grid`. `"."` is an empty cell, `"#"` is a wall, `"@"` is the starting point, (`"a"`, `"b"`, ...) are keys, and (`"A"`, `"B"`, ...) are locks.

We start at the starting point, and one move consists of walking one space in one of the 4 cardinal directions.  We cannot walk outside the grid, or walk into a wall.  If we walk over a key, we pick it up.  We can't walk over a lock unless we have the corresponding key.

For some 1 <= K <= 6, there is exactly one lowercase and one uppercase letter of the first `K` letters of the English alphabet in the grid.  This means that there is exactly one key for each lock, and one lock for each key; and also that the letters used to represent the keys and locks were chosen in the same order as the English alphabet.

Return the lowest number of moves to acquire all keys.  If it's impossible, return `-1`.


Example 1:
Input: ["@.a.#","###.#","b.A.B"]
Output: 8

Example 2:
Input: ["@..aA","..B#.","....b"]
Output: 6
Note:
`1 <= grid.length <= 30`
`1 <= grid[0].length <= 30`
`grid[i][j]` contains only` '.'`, `'#'`, `'@'`, `'a'-``'f``'` and `'A'-'F'`
The number of keys is in `[1, 6]`.  Each key has a different letter and opens exactly one lock.

### Solution 
 Certainly! Let's solve the given problem by implementing a solution that utilizes breadth-first search (BFS) to navigate the grid. The approach involves exploring all potential paths, collecting keys, and ensuring we can unlock doors using the previously collected keys.

Here's the detailed solution in the format suitable for LeetCode:



```python
from collections import deque

class Solution:
    def shortestPathAllKeys(self, grid):
        rows, cols = len(grid), len(grid[0])
        # Directions for moving up, down, left, right
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Find the starting point and the number of keys
        start_row, start_col = 0, 0
        keys_count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '@':
                    start_row, start_col = r, c
                elif 'a' <= grid[r][c] <= 'f':
                    keys_count += 1
        
        # The target state where all keys have been collected
        target_state = (1 << keys_count) - 1
        
        # BFS queue initialized with starting point and the initial key state
        queue = deque([(start_row, start_col, 0, 0)])  # (row, col, keys collected, steps)
        visited = set((start_row, start_col, 0))  # to avoid revisiting the same state

        while queue:
            row, col, keys, steps = queue.popleft()
            # If we have collected all keys, return the number of steps
            if keys == target_state:
                return steps
            
            # Explore all 4 directions
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                # Check if in bounds
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    cell = grid[new_row][new_col]
                    # Check for walls
                    if cell == '#':
                        continue
                    
                    new_keys = keys
                    # Check for keys and update the key state
                    if 'a' <= cell <= 'f':
                        new_keys |= (1 << (ord(cell) - ord('a')))
                    # Check for locks; we can only pass if we have the key
                    elif 'A' <= cell <= 'F':
                        if not (keys & (1 << (ord(cell) - ord('A')))):
                            continue  # We don't have the key for this lock
                    
                    # If we haven't visited this state (row, col, new_keys)
                    if (new_row, new_col, new_keys) not in visited:
                        visited.add((new_row, new_col, new_keys))
                        queue.append((new_row, new_col, new_keys, steps + 1))

        # If we exit the loop without finding all keys, return -1
        return -1

```

### Explanation:
1. **Initialization**:
    - We determine the dimensions of the grid and identify the starting position represented by `@`.
    - We also count the number of keys, which corresponds to lowercase letters (`'a'` to `'f'`).
    - The target state (all keys collected) is represented as a bitmask where each bit corresponds to a key.

2. **BFS Setup**:
    - We use a deque (`queue`) to facilitate the BFS and keep track of the current position, keys collected so far, and step count.
    - We initialize the `visited` set to keep track of which states, represented by `(row, col, keys)`, we’ve already processed to avoid cycles.

3. **BFS Execution**:
    - For each position, we check neighboring cells according to 4 possible directions.
    - Walls are skipped, and upon encountering keys, we update our keys state using bit manipulation.
    - If a lock is encountered, we check if the corresponding key is already collected.

4. **End Condition**:
    - If we gather all the keys (reached the target state) at any point, we return the number of steps taken; if not, we return `-1` after exhausting all possibilities.

This approach efficiently navigates through the grid and provides the correct path length for key collection in the constraints given.

# 865. Smallest Subtree with all the Deepest Nodes

### Problem Description 
Given the `root` of a binary tree, the depth of each node is the shortest distance to the root.

Return the smallest subtree such that it contains all the deepest nodes in the original tree.

A node is called the deepest if it has the largest depth possible among any node in the entire tree.

The subtree of a node is tree consisting of that node, plus the set of all descendants of that node.

Note: This question is the same as 1123: https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/

Example 1:
Input: root = [3,5,1,6,2,0,8,null,null,7,4]
Output: [2,7,4]
Explanation: We return the node with value 2, colored in yellow in the diagram.

The nodes coloured in blue are the deepest nodes of the tree.

Notice that nodes 5, 3 and 2 contain the deepest nodes in the tree but node 2 is the smallest subtree among them, so we return it.


Example 2:
Input: root = [1]
Output: [1]
Explanation: The root is the deepest node in the tree.


Example 3:
Input: root = [0,1,3,null,2]
Output: [2]
Explanation: The deepest node in the tree is 2, the valid subtrees are the subtrees of nodes 2, 1 and 0 but the subtree of node 2 is the smallest.


Constraints:
The number of nodes in the tree will be in the range `[1, 500]`.

`0 <= Node.val <= 500`
The values of the nodes in the tree are unique.

### Solution 
 To solve the problem of finding the smallest subtree containing all the deepest nodes in a binary tree, we can use a depth-first search (DFS) strategy. The idea is to traverse the tree while keeping track of the depth of each node, and determine which nodes are the deepest. We then find the lowest common ancestor (LCA) of these deepest nodes, which will be our output.

Here's a step-by-step breakdown of how to implement the solution:

### Steps:
1. **Traverse the Tree Using DFS:** We will traverse the tree using DFS to calculate the depth of each node. While traversing, we'll keep track of the deepest nodes we encounter.
  
2. **Determine the Maximum Depth:** During the traversal, also record the maximum depth reached, so we can identify which nodes are the deepest.

3. **Find the Smallest Subtree:** We will perform another traversal to find the LCA of all the deepest nodes. The LCA will be our result since it is the smallest subtree containing all deepest nodes.

4. **Return the LCA Node:** Finally, return the node that represents the smallest subtree containing the deepest nodes.

### Python Code:

Here is the code implementation in Python following the LeetCode format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def subtreeWithAllDeepest(self, root: TreeNode) -> TreeNode:
        def find_depth(node):
            if not node:
                return 0
            
            left_depth = find_depth(node.left)
            right_depth = find_depth(node.right)
            
            # Update the maximum depth
            self.max_depth = max(self.max_depth, left_depth + 1, right_depth + 1)
            
            return max(left_depth, right_depth) + 1

        def lca_deepest(node, depth):
            if not node:
                return None
            
            if depth == 1:
                return node
            
            left = lca_deepest(node.left, depth - 1)
            right = lca_deepest(node.right, depth - 1)
            
            if left and right:
                return node
            
            return left if left else right
            
        self.max_depth = 0
        find_depth(root)
        return lca_deepest(root, self.max_depth)


```

### Explanation:

1. **TreeNode Class:** The `TreeNode` class is a standard representation of a node in a binary tree.

2. **Solution Class:** 
   - We define the `subtreeWithAllDeepest` method inside the `Solution` class.
   - It uses a helper function `find_depth` to calculate the maximum depth of the tree while also updating a class variable `self.max_depth`.
  
3. **Depth Calculation:** 
   - `find_depth` function recursively computes the depth of each node. For each visit, we determine if the current depth is the maximum found so far.

4. **Finding the LCA:** 
   - The `lca_deepest` function identifies the lowest common ancestor for the deepest leaves. It checks left and right children and returns the closest ancestor with both children populated.

5. **Return Result:** 
   - Finally, the method `subtreeWithAllDeepest` returns the LCA of the deepest nodes found in the tree.

### Complexity:
Time Complexity: O(N), where N is the number of nodes in the tree, since we traverse each node at least once.

Space Complexity: O(H), where H is the height of the tree, due to recursion stack space. In the worst case, it could be O(N) for skewed trees.

This implementation is efficient and ensures that we return the desired subtree containing all the deepest nodes in the binary tree.

# 882. Reachable Nodes In Subdivided Graph

### Problem Description 
You are given an undirected graph (the "original graph") with `n` nodes labeled from `0` to `n - 1`. You decide to subdivide each edge in the graph into a chain of nodes, with the number of new nodes varying between each edge.

The graph is given as a 2D array of `edges` where `edges[i] = [ui, vi, cnti]` indicates that there is an edge between nodes `ui` and `vi` in the original graph, and `cnti` is the total number of new nodes that you will subdivide the edge into. Note that `cnti == 0` means you will not subdivide the edge.

To subdivide the edge `[ui, vi]`, replace it with `(cnti + 1)` new edges and `cnti` new nodes. The new nodes are `x1`, `x2`, ..., `xcnti`, and the new edges are `[ui, x1]`, `[x1, x2]`, `[x2, x3]`, ..., `[xcnti+1, xcnti]`, `[xcnti, vi]`.

In this new graph, you want to know how many nodes are reachable from the node `0`, where a node is reachable if the distance is `maxMoves` or less.

Given the original graph and `maxMoves`, return the number of nodes that are reachable from node `0` in the new graph.


Example 1:
Input: edges = [[0,1,10],[0,2,1],[1,2,2]], maxMoves = 6, n = 3
Output: 13
Explanation: The edge subdivisions are shown in the image above.

The nodes that are reachable are highlighted in yellow.


Example 2:
Input: edges = [[0,1,4],[1,2,6],[0,2,8],[1,3,1]], maxMoves = 10, n = 4
Output: 23

Example 3:
Input: edges = [[1,2,4],[1,4,5],[1,3,1],[2,3,4],[3,4,5]], maxMoves = 17, n = 5
Output: 1
Explanation: Node 0 is disconnected from the rest of the graph, so only node 0 is reachable.


Constraints:
`0 <= edges.length <= min(n * (n - 1) / 2, 104)`
`edges[i].length == 3`
`0 <= ui < vi < n`
There are no multiple edges in the graph.

`0 <= cnti <= 104`
`0 <= maxMoves <= 109`
`1 <= n <= 3000`

### Solution 
 To solve the problem of determining how many nodes are reachable from node `0` in the modified graph after subdividing edges into chains of nodes, we'll use a combination of a breadth-first search (BFS) algorithm and careful management of the distances based on the edge subdivisions. Here’s how to approach it step-by-step.

### Problem Breakdown:
1. **Graph Construction**:
   We need to build a mapping (like an adjacency list) from the provided edges to understand how the original nodes connect and how the subdivisions affect the reach to new nodes.

2. **Distance Calculation**:
   For each edge `(u, v)`, the number of new nodes defined by `cnt` needs to determine the total distance to traverse. The distance from `u` to `v` will now include `cnt + 1` edges in total.

3. **BFS for Reachability**:
   We will use a BFS approach, which allows us to effectively manage how many moves are used to reach adjacent nodes. As we traverse, we can keep track of the cumulative distance (or moves used) and terminate the search when we exceed `maxMoves`.

### Code Implementation:
Now, let's write the code following these steps:



```python
from collections import defaultdict, deque

def reachableNodes(edges, maxMoves, n):
    # Create an adjacency list for the graph
    graph = defaultdict(list)
    
    for u, v, cnt in edges:
        graph[u].append((v, cnt + 1))  # Add v with the distance count + 1
        graph[v].append((u, cnt + 1))  # Add u with the same distance

    # BFS to count reachable nodes
    reachable_count = 0
    queue = deque([(0, maxMoves)])  # Start from node 0 with maxMoves
    visited = set()  # To keep track of visited nodes
    
    while queue:
        current_node, remaining_moves = queue.popleft()
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        reachable_count += 1  # Count this node as reachable
        
        for neighbor, distance in graph[current_node]:
            if remaining_moves >= distance:  # We can reach this neighbor
                queue.append((neighbor, remaining_moves - distance))
    
    # Calculate additional reachable nodes added from subdivisions
    additional_reachable = 0
    
    for u, v, cnt in edges:
        if u in visited and v in visited:
            # Count all subdivisions because both ends are reachable
            additional_reachable += cnt
        elif u in visited or v in visited:
            # Count only the reachable subdivisions (from the reachable node)
            additional_reachable += max(0, cnt - (maxMoves - abs(cnt + 1)))
    
    return reachable_count + additional_reachable

# Example usage:
# edges = [[0, 1, 10], [0, 2, 1], [1, 2, 2]]
# maxMoves = 6
# n = 3
# print(reachableNodes(edges, maxMoves, n))  # Output: 13

```

### Explanation:
1. **Graph Construction**:
   The graph is built using a `defaultdict` where each node points to its neighbors along with the distance we need to traverse (which accounts for subdivisions).

2. **Breadth-First Search (BFS)**:
   We use a deque for BFS, starting from node `0`, which allows us to explore all reachable nodes without exceeding `maxMoves`. Each time we explore a new node, we check how many moves we have remaining after reaching that node.

3. **Count Additional Nodes**:
   After identifying the nodes reachable within the allowed moves, we also need to account for additional subdivisions that can be "reachable" due to connections from visited nodes to unvisited subdivisions.

This approach ensures edge cases are handled efficiently, and the overall solution is optimized for the constraints given in the problem. The output of reachable nodes is returned after combining the initially reachable nodes with the additional subdivisions reachable through those nodes.

