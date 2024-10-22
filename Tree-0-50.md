# 94. Binary Tree Inorder Traversal

### Problem Description 
Given the `root` of a binary tree, return the inorder traversal of its nodes' values.


Example 1:
Input: root = [1,null,2,3]
Output: [1,3,2]

Example 2:
Input: root = []
Output: []

Example 3:
Input: root = [1]
Output: [1]

Example 4:
Input: root = [1,2]
Output: [2,1]

Example 5:
Input: root = [1,null,2]
Output: [1,2]

Constraints:
The number of nodes in the tree is in the range `[0, 100]`.

`-100 <= Node.val <= 100`
Follow up:
Recursive solution is trivial, could you do it iteratively?

### Solution 
 To solve the problem of performing an inorder traversal of a binary tree, we can use both recursive and iterative approaches. However, given the follow-up request, I'll be focusing on the iterative solution. 

### Inorder Traversal Explanation
Inorder traversal of a binary tree visits nodes in the following order:
1. Traverse the left subtree.
2. Visit the root node.
3. Traverse the right subtree.

The iterative approach uses a stack to accomplish this without the need for recursion.

### Steps for the Iterative Approach:
1. Initialize an empty stack and set the current node as the root.
2. Use a loop that continues until the stack is empty and the current node is `None`.
3. Traverse to the leftmost node, pushing each visited node to the stack.
4. When you reach a `None` node, pop the top node from the stack and add its value to the result list.
5. Switch to the right subtree of the popped node and repeat until all nodes have been visited.

### Python Code Implementation

Here's how the code is structured in accordance with LeetCode's requirement:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def inorderTraversal(self, root: TreeNode) -> list[int]:
        result = []  # This will hold the inorder traversal result
        stack = []   # Stack for iteratively traversing the tree
        current = root  # Start with the root node
        
        while current is not None or stack:
            # Reach the leftmost node of the current node
            while current is not None:
                stack.append(current)  # Push current node to stack
                current = current.left  # Move to left child
            
            # Current must be None at this point, we process the nodes
            current = stack.pop()  # Pop the top node from the stack
            result.append(current.val)  # Add the node's value to the result
            
            # Now visit the right subtree
            current = current.right
        
        return result  # Return the collected values

```

### Explanation of the Code:
- **TreeNode Class**: This defines the structure of each node in the tree.
- **Solution Class**: Contains the method `inorderTraversal()` which will perform the inorder traversal.
    - `result`: A list to store the values of the nodes in inorder.
    - `stack`: A stack to hold the nodes we visit as we go down the tree.
    - `current`: Initially points to the root of the binary tree.
  
- **Outer While Loop**: Continues until we've processed all nodes.
- **Inner While Loop**: It travels down the left side of the tree, adding nodes to the stack.
- **Node Processing**: Once we reach a `None` node, we pop the top node from the stack (which is the last left node we found), add its value to the result list, and then move to its right child.

### Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes in the binary tree. We visit each node exactly once.
- **Space Complexity**: O(h), where h is the height of the tree. This space is used for the stack, in the worst case (for a skewed tree), the height could be equal to n.

This solution will function effectively within the constraints provided and can be directly run on the LeetCode platform.

# 95. Unique Binary Search Trees II

### Problem Description 
Given an integer `n`, return all the structurally unique BST's (binary search trees), which has exactly `n` nodes of unique values from `1` to `n`. Return the answer in any order.


Example 1:
Input: n = 3
Output: [[1,null,2,null,3],[1,null,3,2],[2,1,3],[3,1,null,null,2],[3,2,null,1]]

Example 2:
Input: n = 1
Output: [[1]]

Constraints:
`1 <= n <= 8`

### Solution 
 To solve the problem of generating all structurally unique binary search trees (BSTs) with `n` nodes containing unique values from `1` to `n`, we can use a recursive approach with a function that constructs trees using each number from `1` to `n` as the root once, and recursively constructing the left and right subtrees from the remaining numbers.

Here's the step-by-step outline of the algorithm:

1. **Base Case**: If `n` is `0`, return an empty list, as there are no trees to be formed. If `n` is `1`, return a list containing a tree with a single node.

2. **Recursive Case**: Iterate through each integer `i` from `1` to `n`. Treat `i` as the root of the BST.
   - All integers less than `i` will form the left subtree.
   - All integers greater than `i` will form the right subtree.
   
   For each `i`:
   - Recursively generate all possible left subtrees using integers `1` to `i-1`.
   - Recursively generate all possible right subtrees using integers `i+1` to `n`.
   - Combine each left subtree with each right subtree and form a unique tree.

3. **Construct the Tree**: Use a tree node class to represent the binary tree nodes. Each node will have a value, and pointers to the left and right children.

4. **Return the Result**: Store all unique trees in a list and return that list.

Here's how the solution can be implemented in Python following the LeetCode solution format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        if n == 0:
            return []
        
        def generate_trees(start: int, end: int) -> List[TreeNode]:
            if start > end:
                return [None]  # Return a list containing a None tree
            
            all_trees = []
            for i in range(start, end + 1):
                # Generate all left and right subtrees
                left_trees = generate_trees(start, i - 1)
                right_trees = generate_trees(i + 1, end)
                
                # Combine left and right subtrees with the current root node
                for left in left_trees:
                    for right in right_trees:
                        root = TreeNode(i)  # Create a new tree node
                        root.left = left   # Set the left subtree
                        root.right = right  # Set the right subtree
                        all_trees.append(root)  # Add the tree to the list
            
            return all_trees
        
        return generate_trees(1, n)

# Example usage:
# sol = Solution()
# results = sol.generateTrees(3)
# for tree in results:
#     print(tree)  # This will print the tree structures, but you'd need a custom function to display them nicely.

```

### Explanation of the Code:

1. We define a `TreeNode` class which represents each node in the BST.
2. The `Solution` class contains the `generateTrees` method which initializes the recursion.
3. The `generate_trees` function is defined inside `generateTrees` to keep track of the range `(start, end)` for which we are generating trees.
4. We loop through all possible root values using `i`, generating all combinations of left and right subtrees recursively.
5. Every time we create a new root, we construct a unique BST and add it to the `all_trees` list.
6. Finally, the constructed list of trees is returned.

This solution efficiently generates all unique BSTs for a given `n` and adheres to the LeetCode format required for submission.

# 96. Unique Binary Search Trees

### Problem Description 
Given an integer `n`, return the number of structurally unique BST's (binary search trees) which has exactly `n` nodes of unique values from `1` to `n`.


Example 1:
Input: n = 3
Output: 5

Example 2:
Input: n = 1
Output: 1

Constraints:
`1 <= n <= 19`

### Solution 
 To solve the problem of counting the number of structurally unique binary search trees (BSTs) that can be created with `n` distinct nodes with values from `1` to `n`, we can use dynamic programming. 

### Explanation

The number of unique BSTs that can be formed with `n` nodes is based on the concept of Catalan numbers. The main idea is:

1. **Choose Node as Root**: We can pick any of the `n` nodes as the root of the tree. If we choose a node `i` (where `1 <= i <= n`), then all nodes with smaller values (`1` to `i-1`) will form the left subtree, and all nodes with larger values (`i+1` to `n`) will form the right subtree.

2. **Count Left and Right Subtrees**: The number of BSTs that can be formed with `i` as the root is the product of the number of unique BSTs that can be formed with the left nodes and the right nodes:
   - Left subtrees: Formed by the nodes from `1` to `i-1` which counts to `i-1` nodes.
   - Right subtrees: Formed by the nodes from `i+1` to `n` which counts to `n-i` nodes.

3. **Dynamic Programming**: We define a DP array `dp` where `dp[i]` represents the number of unique BSTs that can be formed with `i` nodes. Using a nested loop, we can calculate the total unique BSTs by summing up the products of left and right subtrees for every possible root.

4. **Base Case**: There's one unique BST for `0` nodes (an empty tree) and one unique BST for `1` node.

### Implementation

Here’s the Python code to implement the aforementioned logic:



```python
class Solution:
    def numTrees(self, n: int) -> int:
        # dp[i] will hold the number of unique BSTs for 'i' nodes
        dp = [0] * (n + 1)
        
        # Base cases: 0 nodes -> 1 empty tree, 1 node -> 1 single-node tree
        dp[0] = 1
        dp[1] = 1
        
        # Fill dp array for all counts from 2 to n
        for nodes in range(2, n + 1):
            total_trees = 0
            for root in range(1, nodes + 1): 
                # left = nodes before root, right = nodes after root
                left_trees = dp[root - 1]  # count of trees from 1 to root-1
                right_trees = dp[nodes - root]  # count of trees from root+1 to n
                total_trees += left_trees * right_trees
                
            dp[nodes] = total_trees
        
        return dp[n]  # Return the number of unique BSTs for 'n' nodes

```

### How It Works

1. We initialize a list `dp` of size `n+1`, where `dp[0]` and `dp[1]` are set to `1`.

2. For each number of nodes from `2` to `n`, we calculate the total number of unique BSTs by iterating over each possible root value.

3. For each root, we compute the number of BSTs that can be formed with the left and right children and sum their products.

4. Finally, we return `dp[n]`, which contains the count of unique BSTs for `n` nodes.

### Complexity

- **Time Complexity**: O(n²), because for each `nodes` (up to `n`), we iterate over each possible root.
- **Space Complexity**: O(n), due to the storage needed for the `dp` array. 

This solution is efficient given the constraints and makes use of dynamic programming to systematically build up the solution from smaller subproblems.

# 99. Recover Binary Search Tree

### Problem Description 
You are given the `root` of a binary search tree (BST), where exactly two nodes of the tree were swapped by mistake. Recover the tree without changing its structure.

Follow up: A solution using `O(n)` space is pretty straight forward. Could you devise a constant space solution?

Example 1:
Input: root = [1,3,null,null,2]
Output: [3,1,null,null,2]
Explanation: 3 cannot be a left child of 1 because 3 > 1. Swapping 1 and 3 makes the BST valid.


Example 2:
Input: root = [3,1,4,null,null,2]
Output: [2,1,4,null,null,3]
Explanation: 2 cannot be in the right subtree of 3 because 2 < 3. Swapping 2 and 3 makes the BST valid.


Constraints:
The number of nodes in the tree is in the range `[2, 1000]`.

`-231 <= Node.val <= 231 - 1`

### Solution 
 To solve the problem of recovering a binary search tree (BST) where two nodes have been swapped, we can use an in-order traversal approach. In a BST, an in-order traversal yields the values in sorted order. After the swap, this order is disrupted.

Here’s a detailed explanation of how we can approach this using the in-order traversal method, as well as how to implement it in Python for the LeetCode platform.

### Steps to Solution:
1. **In-order Traversal**: Perform an in-order traversal of the BST while maintaining a record of the previously seen node's value. This helps in identifying the two nodes that are swapped.
2. **Identify Nodes**: During the traversal:
   - If we ever find that the current node's value is less than the previous node's value, it indicates a problem:
     - The first time this occurs, we might have found the first swapped node, and we record it as `first`.
     - The second time this occurs, we record the second swapped node as `second`.
3. **Swap the Values**: After the traversal is complete, we just need to swap the values of these two nodes `first` and `second`.

### Implementation:
Here’s the implementation of the above logic in Python:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def recoverTree(self, root: TreeNode) -> None:
        # Variables to keep track of the nodes to be swapped
        first = second = prev = None
        
        # In-order Traversal Helper Function
        def inorder(node):
            nonlocal first, second, prev
            
            if not node:
                return
            
            # Traverse the left subtree
            inorder(node.left)
            
            # Check for the misplaced nodes
            if prev and node.val < prev.val:
                # If first is not set, we have found the first node
                if not first:
                    first = prev
                # We will update the 'second' node every time we find a misplaced node
                second = node
            
            # Mark this node as previous for the next iteration
            prev = node
            
            # Traverse the right subtree
            inorder(node.right)
        
        # Start the inorder traversal
        inorder(root)
        
        # Swap the values of the two nodes
        if first and second:
            first.val, second.val = second.val, first.val

```

### Explanation:
- **TreeNode Class**: This defines the structure of the nodes in our BST.
- **Solution Class**: This encapsulates our solution where `recoverTree` function will perform the recovery operation.
- **recoverTree Method**:
  - We declare variables `first`, `second`, and `prev` to keep track of the erroneous nodes and the previous node in traversal.
  - We define and call an internal `inorder` function that performs the in-order traversal while maintaining the identified nodes.
- **Value Swap**: Finally, if both `first` and `second` nodes are identified as swapped, we swap their values to correct the BST.

By this approach, we ensure the solution runs in `O(n)` time and uses `O(1)` additional space (aside from recursion stack), which fulfills the problem's constraints. This implementation can be copied directly to the LeetCode environment for testing.

# 100. Same Tree

### Problem Description 
Given the roots of two binary trees `p` and `q`, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.


Example 1:
Input: p = [1,2,3], q = [1,2,3]
Output: true

Example 2:
Input: p = [1,2], q = [1,null,2]
Output: false

Example 3:
Input: p = [1,2,1], q = [1,1,2]
Output: false

Constraints:
The number of nodes in both trees is in the range `[0, 100]`.

`-104 <= Node.val <= 104`

### Solution 
 To solve the problem of checking if two binary trees are the same, we will perform a recursive comparison of both trees. We will check the following conditions for both trees:

1. If both nodes are `None`, then they are the same (i.e., both trees are empty).
2. If one of the nodes is `None` but the other is not, then the trees are not the same.
3. If both nodes are not `None`, we need to check if the values of both nodes are equal. If they are not equal, the trees are not the same.
4. If the values are equal, we recursively check their left and right children.

Using this approach, we can ensure that we cover both structure and value comparisons in our trees. The algorithm operates in O(n) time complexity, where n is the number of nodes since it needs to potentially visit every node.

Here's how you can implement this in Python in a format suitable for LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        # Both nodes are None, trees are the same
        if not p and not q:
            return True
        # One of the nodes is None, trees are not the same
        if not p or not q:
            return False
        # Check if the current nodes' values are the same
        if p.val != q.val:
            return False
        # Recursively check the left and right subtree
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

# Example usage:
# p = TreeNode(1, TreeNode(2), TreeNode(3))
# q = TreeNode(1, TreeNode(2), TreeNode(3))
# sol = Solution()
# print(sol.isSameTree(p, q))  # Output: True

```

### Explanation of the Code:
1. **TreeNode definition:** We define a class `TreeNode` representing each node in the binary tree with constructor parameters for the node's value and pointers to left and right children.

2. **Solution Class:** The `Solution` class contains our main method, `isSameTree`, which will take two tree roots as parameters.

3. **Base Cases:**
   - If both nodes `p` and `q` are `None`, we return `True`, indicating both trees are the same (both are empty).
   - If either `p` or `q` is `None` (but not both), we return `False`, indicating the trees aren't the same due to structural difference.
   - If the values of `p` and `q` are different, we return `False` immediately.

4. **Recursive Checks:** If the above conditions are not triggered, we recursively check whether both their left and right subtrees are the same by calling `isSameTree` on the left and right children of both nodes.

### Example Usage:
- The example shows how to create two identical trees and check if they are the same using the `isSameTree` method from the `Solution` class.

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
 To solve the problem of checking whether a binary tree is symmetric (i.e., a mirror of itself), we can approach this in both recursive and iterative manners. Below, we will present both approaches with their respective implementations and explanations.

### Problem Explanation

A binary tree is symmetric if the left subtree is a mirror reflection of the right subtree. To check this, we can define a helper function that compares two trees to determine if they are mirrors. The criteria for two trees to be mirrors are:
1. The values of the nodes at the current positions must be equal.
2. The left subtree of one tree must be a mirror reflection of the right subtree of the other tree, and vice versa.

### Recursive Approach

The recursive approach is straightforward and involves traversing the tree in pairs. Here’s the plan for our recursive solution:
1. If both nodes are `None`, return `True` (because two empty trees are mirrors).
2. If one of the nodes is `None` and the other is not, return `False` (because one empty tree cannot mirror a non-empty tree).
3. Compare the values of the two nodes. If they are equal, proceed to compare the left child of the first node with the right child of the second node, and the right child of the first node with the left child of the second node recursively.

Here is the code for the recursive solution:

### Code for Recursive Solution



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        def isMirror(t1: TreeNode, t2: TreeNode) -> bool:
            if not t1 and not t2:  # Both are None
                return True
            if not t1 or not t2:  # One of them is None
                return False
            return (t1.val == t2.val and isMirror(t1.right, t2.left) and isMirror(t1.left, t2.right))

        return isMirror(root, root)  # Start with the root node

```

### Iterative Approach

For the iterative approach, we can use a queue (or a stack) to perform a level-order traversal of the tree while checking the symmetry conditions:
1. Start by adding the two root nodes to a queue.
2. While there are nodes in the queue, do the following:
   - Dequeue two nodes at a time.
   - If both nodes are `None`, continue.
   - If one is `None` and the other is not, return `False`.
   - If their values are not equal, return `False`.
   - Enqueue the children of the nodes in a specific order — enqueue the left child of the first node, then the right child of the second node, and vice versa for the other node.

Here is the code for the iterative solution:

### Code for Iterative Solution



```python
from collections import deque

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        queue = deque([root, root])
        
        while queue:
            t1 = queue.popleft()
            t2 = queue.popleft()
            
            if not t1 and not t2:
                continue
            if not t1 or not t2:
                return False
            if t1.val != t2.val:
                return False
            
            queue.append(t1.left)
            queue.append(t2.right)
            queue.append(t1.right)
            queue.append(t2.left)
        
        return True

```

### Summary

- The recursive approach provides a clean and elegant solution that intuitively expresses the problem of symmetry through recursion.
- The iterative approach utilizes a queue to traverse the tree and check symmetry, which can be more suitable for large trees as it avoids deep recursion, thus preventing stack overflow issues.

Both implementations offer efficient O(n) time complexity, where n is the number of nodes in the tree, and O(h) space complexity, where h is the height of the tree.

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
 To solve the problem of level order traversal of a binary tree, we can utilize a breadth-first search (BFS) approach. This type of traversal visits each level of the tree in order, collecting the node values level by level.

Here’s how we can implement this:

1. **Use a Queue**: We will employ a queue to facilitate the BFS. We start by enqueueing the root node of the tree. This allows us to process nodes in the order they are added to the queue, which corresponds to their level in the tree.

2. **Process Each Level**: As long as there are nodes in the queue, we will dequeue all the nodes at the current level, collect their values, and enqueue their children to process the next level.

3. **Store the Results**: We will maintain a list that contains sublists, where each sublist represents the values of nodes at a specific level.

Here's the code that accomplishes this:



```python
from collections import deque

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

        result = []
        queue = deque([root])  # Initialize the queue with the root
        
        while queue:
            level_size = len(queue)  # Number of nodes at the current level
            current_level = []  # List to store values of nodes at this level
            
            for _ in range(level_size):
                node = queue.popleft()  # Dequeue the front node
                current_level.append(node.val)  # Add its value to the current level list
                
                # Enqueue the left and right children (if they exist)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(current_level)  # Add the current level list to the result
        
        return result

```

### Explanation of the Code

1. **TreeNode Class**: We define a `TreeNode` class that represents each node in the binary tree. This class has a constructor that initializes the node's value and its left and right children.

2. **Solution Class**: The `Solution` class contains the `levelOrder` method which is responsible for performing the level order traversal.

3. **Initial Check**: If the root is `None`, we return an empty list since there are no nodes to traverse.

4. **Queue Initialization**: We use a deque (`queue`) from the `collections` module to store nodes. The root is initially added to the queue.

5. **While Loop**: The outer loop continues until the queue is empty, meaning all nodes have been processed.

6. **Level Size**: Before starting to process nodes at the current level, we determine how many nodes are in the queue (i.e., at the current level).

7. **Inner Loop**: For each node in the current level:
   - We dequeue a node and add its value to the `current_level` list.
   - If the node has a left child, we enqueue it. Similarly, we enqueue the right child if it exists.

8. **Storing Current Level**: After processing all nodes at the current level, we append the `current_level` list to the `result`.

9. **Return Result**: Finally, once the loop completes, we return the `result`, which contains lists of node values collected level by level.

This algorithm has a time complexity of O(N), where N is the number of nodes in the tree, since each node is processed once. The space complexity is also O(N) in the worst case for the function call stack and queue storage.

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
 To solve the problem of zigzag level order traversal of a binary tree, we can use a breadth-first search (BFS) approach with a queue. The idea is to traverse the tree level by level, and for each level, we will determine the order in which we collect the node values. This order alternates between left-to-right and right-to-left for each subsequent level.

Here’s a step-by-step explanation of the approach:

1. **Initialize a Queue**: We'll use a queue to facilitate the breadth-first traversal of the tree. The queue will hold the nodes of the tree that we need to process at each level.

2. **Zigzag Order Boolean**: We will maintain a boolean flag that tells us whether we should append the values from left to right or right to left for the current level.

3. **Process Nodes Level by Level**: For each level, we will:
   - Determine the number of nodes to process at that level.
   - Collect the values of the nodes based on the current order (left to right or right to left).
   - Add the children of the current node to the queue for processing in the next level.

4. **Switch Order for Next Level**: After processing a level, we will toggle the order for the next level.

The following is the Python code implementation that adheres to the LeetCode format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from collections import deque

class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        
        results = []
        queue = deque([root])
        left_to_right = True
        
        while queue:
            level_size = len(queue)
            level_values = []
            
            for _ in range(level_size):
                node = queue.popleft()
                # Add the node's value based on the zigzag order
                if left_to_right:
                    level_values.append(node.val)
                else:
                    level_values.insert(0, node.val)  # Insert at the beginning
                
                # Add children to the queue for the next level
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            # Toggle the direction for the next level
            left_to_right = not left_to_right
            results.append(level_values)
        
        return results

```

### Explanation of the Code:

1. **TreeNode Definition**: We start by defining the `TreeNode` class that represents a node in the binary tree.

2. **Solution Class**: The `Solution` class contains the method `zigzagLevelOrder`.

3. **Input Handling**: We check if the `root` is `None`, and if so, we return an empty list.

4. **Queue Initialization**: We initialize a deque (double-ended queue) that will hold the nodes to be processed.

5. **While Loop**: We continue processing while there are nodes in the queue.
   - We record the size of the current level.
   - We then process each node at that level. Depending on the `left_to_right` boolean, we either append to the end of `level_values` (for left-to-right) or insert at the beginning (for right-to-left).

6. **Adding Children**: For each node processed, we add its left and right children to the queue for the upcoming level processing.

7. **Toggle Direction**: Finally, after collecting the current level's values, we toggle the `left_to_right` flag for the next level before appending `level_values` to `results`.

8. **Return Results**: After completing the traversal, we return the `results` list, which contains the zigzag level order traversal of the tree.

This implementation efficiently traverses the tree and collects the values in the required zigzag order.

# 104. Maximum Depth of Binary Tree

### Problem Description 
Given the `root` of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.


Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: 3

Example 2:
Input: root = [1,null,2]
Output: 2

Example 3:
Input: root = []
Output: 0

Example 4:
Input: root = [0]
Output: 1

Constraints:
The number of nodes in the tree is in the range `[0, 104]`.

`-100 <= Node.val <= 100`

### Solution 
 To solve the problem of finding the maximum depth of a binary tree, we can utilize a depth-first search (DFS) approach. The idea is to traverse the tree and keep track of the depth of each path until we reach a leaf node (a node without children). When we reach a leaf node, we compare the current depth with the maximum depth recorded so far and update accordingly.

### Explanation of the Approach:

1. **Base Case**:
   - If the root is `None` (i.e., the tree is empty), the maximum depth is `0`.
  
2. **Recursive Case**:
   - If the root is not `None`, we perform a recursive search on both the left and right subtrees.
   - For each subtree, we increment the depth and compute the maximum depth from both left and right children, then take the maximum of those two values.

3. **Return Value**:
   - The maximum depth encountered during the traversal is returned.

This recursive approach is efficient and works within the constraints given.

### Implementation:

Let's implement the solution using the structure required by LeetCode.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # Base case: if the tree is empty
        if not root:
            return 0
        
        # Recursive case: compute the depth of left and right subtrees
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)
        
        # Current depth is max of left depth and right depth, plus one for the current node
        return max(left_depth, right_depth) + 1

```

### Explanation of the Code:

1. **TreeNode Class**: A simple class to define the structure of a binary tree node.
   - Each node has a value `val`, a left child `left`, and a right child `right`.

2. **maxDepth Function**:
   - This is a method of the `Solution` class designed to find the maximum depth of a binary tree.
   - It first checks if `root` is `None`. If so, we return `0`, indicating the tree is empty.
   - If the node is not `None`, it recursively calls `maxDepth` on the left and right children and stores their depths.
   - The maximum of these depths is then calculated, and we return it incremented by `1` (to account for the current node).

### Test Cases:
Here are some test cases to validate the solution:

1. **Example 1**:
   Input: `[3,9,20,null,null,15,7]`  
   Output: `3`

2. **Example 2**:
   Input: `[1,null,2]`  
   Output: `2`

3. **Example 3**:
   Input: `[]`  
   Output: `0`

4. **Example 4**:
   Input: `[0]`  
   Output: `1`

### Complexity Analysis:
- **Time Complexity**: O(N), where N is the number of nodes in the binary tree. We visit every node once.
- **Space Complexity**: O(H), where H is the height of the tree, which is required for the recursion stack space. In the worst case (skewed tree), this can be O(N).

This implementation adheres to the structure and requirements for LeetCode submissions and can be tested directly on the LeetCode platform.

# 105. Construct Binary Tree from Preorder and Inorder Traversal

### Problem Description 
Given two integer arrays `preorder` and `inorder` where `preorder` is the preorder traversal of a binary tree and `inorder` is the inorder traversal of the same tree, construct and return the binary tree.


Example 1:
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]

Example 2:
Input: preorder = [-1], inorder = [-1]
Output: [-1]

Constraints:
`1 <= preorder.length <= 3000`
`inorder.length == preorder.length`
`-3000 <= preorder[i], inorder[i] <= 3000`
`preorder` and `inorder` consist of unique values.

Each value of `inorder` also appears in `preorder`.

`preorder` is guaranteed to be the preorder traversal of the tree.

`inorder` is guaranteed to be the inorder traversal of the tree.

### Solution 
 To solve the problem of reconstructing a binary tree from its preorder and inorder traversals, we can leverage the properties of these traversal methods:

1. **Preorder Traversal**: The first element of the preorder array is always the root of the tree. Subsequent elements are the nodes in the left and right subtrees.
2. **Inorder Traversal**: The elements of the inorder array are arranged such that all elements to the left of the root node belong to the left subtree, and all elements to the right belong to the right subtree.

### Approach:

1. Use the first element of the `preorder` array as the root of the tree.
2. Find the index of the root in the `inorder` array. This index helps us determine which elements belong to the left and right subtrees.
3. Recursively build the left subtree using the left part of the `inorder` array and the corresponding elements in the `preorder` array.
4. Similarly, build the right subtree using the right part of the `inorder` array and the corresponding elements in the `preorder` array.
5. Continue this process until all elements are processed and the whole tree is constructed.

### Implementation:

Here's the Python code that adheres to the LeetCode solution format. This includes the definition of a `TreeNode` class to represent the nodes of the binary tree, and a `buildTree` function that constructs the binary tree from the given preorder and inorder arrays.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder or not inorder:
            return None
        
        # The first element of preorder list is the root.
        root_val = preorder[0]
        root = TreeNode(root_val)
        
        # Find the index of the root in the inorder list.
        root_index = inorder.index(root_val)
        
        # Build left and right subtrees recursively.
        root.left = self.buildTree(preorder[1:root_index+1], inorder[:root_index])
        root.right = self.buildTree(preorder[root_index+1:], inorder[root_index+1:])
        
        return root

```

### Explanation of the Code:

- **TreeNode Class**: This is a simple structure to hold the value and references to the left and right children of each node.
  
- **Solution Class**: This contains the `buildTree` method which constructs the tree:

1. **Base Case**: If either `preorder` or `inorder` is empty, return `None` as there are no nodes to process.
2. **Identify Root**: The first element of `preorder` gives us the root's value.
3. **Create TreeNode**: A new `TreeNode` is created with the root's value.
4. **Find Root in Inorder**: The index of the root value in `inorder` divides the list into left and right subtrees.
5. **Recursive Calls**: The left subtree is built from the parts of `preorder` and `inorder` that correspond to the left subtree. Similarly, the right subtree is built from the remaining parts of the lists.

### Remarks:
This code runs efficiently as it recursively divides the problem and builds the tree until all elements are processed. The constraints ensure that all elements are unique and valid, so the index searches and recursive calls will not create any issues.

# 106. Construct Binary Tree from Inorder and Postorder Traversal

### Problem Description 
Given two integer arrays `inorder` and `postorder` where `inorder` is the inorder traversal of a binary tree and `postorder` is the postorder traversal of the same tree, construct and return the binary tree.


Example 1:
Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
Output: [3,9,20,null,null,15,7]

Example 2:
Input: inorder = [-1], postorder = [-1]
Output: [-1]

Constraints:
`1 <= inorder.length <= 3000`
`postorder.length == inorder.length`
`-3000 <= inorder[i], postorder[i] <= 3000`
`inorder` and `postorder` consist of unique values.

Each value of `postorder` also appears in `inorder`.

`inorder` is guaranteed to be the inorder traversal of the tree.

`postorder` is guaranteed to be the postorder traversal of the tree.

### Solution 
 To solve the problem of constructing a binary tree from its inorder and postorder traversals, we can leverage the properties of these traversals:

1. In a postorder traversal, the last element is always the root of the tree.
2. The elements that appear before this root in the postorder list can be divided into left and right subtrees based on their positions in the inorder list.

Here's the step-by-step approach to solve this problem:

1. **Identify the Root**: The last element in the postorder array is the root of the binary tree (or subtree).

2. **Find Left and Right Subtrees**: 
   - Locate the index of the root in the inorder array; everything to the left of this index is part of the left subtree, and everything to the right is part of the right subtree.

3. **Recursion**: 
   - We will recursively construct the left and right subtrees by repeating the process for the corresponding segments of the inorder and postorder arrays.

4. **Base Case**: When there are no elements left to process (the start index exceeds the end index), we return `None`.

Here is the Python code that implements this approach:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def buildTree(self, inorder: list[int], postorder: list[int]) -> TreeNode:
        # Create a hashmap to store the index of each value in the inorder array for quick access
        in_index_map = {value: index for index, value in enumerate(inorder)}
        
        # Inner function to build tree recursively
        def build(in_start, in_end, post_start, post_end):
            # Base case: if there are no elements to form the subtree
            if in_start > in_end or post_start > post_end:
                return None
            
            # The last element in the postorder array is the root of the current subtree
            root_value = postorder[post_end]
            root = TreeNode(root_value)

            # Split the inorder array into left and right subtrees
            in_root_index = in_index_map[root_value]
            left_tree_size = in_root_index - in_start

            # Construct the left and right subtrees recursively
            root.left = build(in_start, in_root_index - 1, post_start, post_start + left_tree_size - 1)
            root.right = build(in_root_index + 1, in_end, post_start + left_tree_size, post_end - 1)

            return root

        # Start the recursion from the full range of both traversals
        return build(0, len(inorder) - 1, 0, len(postorder) - 1)

```

### Explanation of the Code
- We define a `TreeNode` class to represent each node of the binary tree.
- The `buildTree` method of the `Solution` class is where we perform the tree construction.
- We create a hashmap (`in_index_map`) that finds the index of each element from the inorder traversal in O(1) time.
- The `build` function is defined to handle the recursion. It takes the current bounding indices for inorder and postorder traversals.
- For each recursive call:
   - We check if the current bounds are valid (base case).
   - We identify the root from the postorder traversal.
   - Using the root's index in the inorder traversal, we calculate the size of the left subtree.
   - We then recursively build the left and right subtrees.

By following these steps, we can successfully reconstruct the binary tree from its inorder and postorder traversals.

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
 To solve the problem of returning the bottom-up level order traversal of the nodes' values in a binary tree, we will follow a methodical approach using a breadth-first search (BFS) strategy with the help of a queue. We will gather values level by level, and finally reverse the order to produce the desired bottom-up result.

Here’s a detailed explanation of the solution:

### Steps to Solve the Problem:

1. **Initialization**:
   - Use a queue (from the `collections` module) to facilitate the BFS traversal of the tree.
   - Use a list to hold the values of each level.
   - We can utilize a list of lists (`result`) to store the values at each level, starting from the top level.

2. **BFS Traversal**:
   - Start by enqueuing the root node into the queue.
   - While the queue is not empty, continue processing:
     - Determine the number of nodes at the current level (this can be tracked by the queue's length).
     - Create a temporary list to store the values for the current level.
     - For each node at the current level, deque the node, append its value to the temporary list, and enqueue its left and right children if they exist.

3. **Building the Result**:
   - After processing all nodes at the current level, append the temporary list to the `result`.
   - Once the BFS is complete, the `result` will have the levels from top to bottom, so we need to reverse it to get the bottom-up order.

4. **Return the Result**:
   - Finally, return the reversed `result`.

### Complete Code:

Here is the implementation of the above approach in Python, formatted according to LeetCode standards:



```python
from collections import deque
from typing import List, Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            current_level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(current_level)
        
        return result[::-1]  # Reverse the order of levels

# Example usage:
# root = TreeNode(3)
# root.left = TreeNode(9)
# root.right = TreeNode(20)
# root.right.left = TreeNode(15)
# root.right.right = TreeNode(7)
# solution = Solution()
# print(solution.levelOrderBottom(root))  # Output: [[15,7],[9,20],[3]]

```

### Explanation of the Code:

- **TreeNode Definition**: This class represents the structure of each node in the binary tree, containing a value and pointers to its left and right children.
- **Solution Class**: The `Solution` class method `levelOrderBottom` is where the logic for the bottom-up level order traversal is implemented.
- **Checking for Empty Tree**: If the root is `None`, return an empty list immediately.
- **Deque for BFS**: A `deque` is initialized with the root node for efficient pop from the front and append at the back operations.
- **Level Processing**: Each level is processed independently, and the values are collected in `current_level`.
- **Results Reversal**: Finally, we reverse `result` before returning it, thus giving a bottom-up traversal output.

This complete method will efficiently traverse the binary tree and provide the required result as stated in the problem description.

# 108. Convert Sorted Array to Binary Search Tree

### Problem Description 
Given an integer array `nums` where the elements are sorted in ascending order, convert it to a height-balanced binary search tree.

A height-balanced binary tree is a binary tree in which the depth of the two subtrees of every node never differs by more than one.


Example 1:
Input: nums = [-10,-3,0,5,9]
Output: [0,-3,9,-10,null,5]
Explanation: [0,-10,5,null,-3,null,9] is also accepted:

Example 2:
Input: nums = [1,3]
Output: [3,1]
Explanation: [1,3] and [3,1] are both a height-balanced BSTs.


Constraints:
`1 <= nums.length <= 104`
`-104 <= nums[i] <= 104`
`nums` is sorted in a strictly increasing order.

### Solution 
 To solve the problem of converting a sorted array into a height-balanced binary search tree (BST), we can leverage the properties of BSTs and the nature of sorted arrays.

### Key Insights:

1. **Height-Balanced BST**: A height-balanced BST is one where the left and right subtrees of any node differ in height by no more than one. This is crucial for maintaining optimal search times.

2. **Using a Sorted Array**: Since the input array is sorted, the middle element can be picked to be the root of the BST. The elements to the left of the middle element form the left subtree, and those to the right form the right subtree.

3. **Divide and Conquer**: This task can be approached recursively:
   - Choose the middle element as the root.
   - Recursively apply the same logic to the left half and the right half of the array to construct the left and right subtrees.

### Steps to Implement:

1. **Define a TreeNode class**: This will represent each node in the tree.
2. **Build the BST recursively**:
   - Base case: If the left index exceeds the right index, return `None` (indicating no subtree).
   - Calculate the middle index and create a node with that value.
   - Recursively build the left subtree using the left half of the array and the right subtree using the right half of the array.
3. **Manage the recursion** using a helper function to handle indices.

### Python Code:
Here is the Python implementation that can be directly run on the LeetCode platform:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def sortedArrayToBST(self, nums):
        # Helper function to construct BST
        def convert_to_bst(left, right):
            if left > right:
                return None
            
            # Middle element to maintain balance
            mid = (left + right) // 2
            node = TreeNode(nums[mid])
            # Recursively construct the left and right subtrees
            node.left = convert_to_bst(left, mid - 1)  # left half
            node.right = convert_to_bst(mid + 1, right)  # right half
            
            return node
        
        return convert_to_bst(0, len(nums) - 1)

```

### Explanation of the Code:

- **TreeNode Class**: A simple class defining the structure of a tree node with `val`, `left`, and `right`.
- **Solution Class**: 
  - The `sortedArrayToBST` method takes a sorted integer array `nums` as input.
  - Inside this method, a nested helper function `convert_to_bst` performs the actual recursive construction of the tree:
    - It checks if the left index is greater than the right index, in which case it returns `None`.
    - It calculates the middle index, creates a new `TreeNode`, and recursively sets the left and right child nodes using the portions of the array split at the middle.
- Finally, the method returns the root of the constructed BST.

### Complexity Analysis:
- **Time Complexity**: O(n), where n is the number of elements in `nums`. Each element is processed once to create a node in the BST.
- **Space Complexity**: O(log n) due to the recursion stack in the worst case (for balanced trees).

This implementation efficiently constructs the desired height-balanced binary search tree from a sorted array while conforming to typical coding challenge standards.

# 110. Balanced Binary Tree

### Problem Description 
Given a binary tree, determine if it is height-balanced.

For this problem, a height-balanced binary tree is defined as:
a binary tree in which the left and right subtrees of every node differ in height by no more than 1.


Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: true

Example 2:
Input: root = [1,2,2,3,3,null,null,4,4]
Output: false

Example 3:
Input: root = []
Output: true

Constraints:
The number of nodes in the tree is in the range `[0, 5000]`.

`-104 <= Node.val <= 104`

### Solution 
 To determine if a binary tree is height-balanced, we need to implement a function that checks the properties of balance across the tree's nodes. A binary tree is considered height-balanced if for any node, the height of its left and right child subtrees differ by no more than 1.

Here, we'll provide a Python solution that follows the LeetCode format, including the definition of the TreeNode class and the function that checks if the binary tree is balanced.

### Python Implementation



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        """
        Determine if the binary tree is height-balanced.
        """

        # Helper function to check height and balance
        def check_height(node):
            # An empty node is height-balanced and has height 0
            if not node:
                return 0

            # Recursively check the height of the left and right subtrees
            left_height = check_height(node.left)
            if left_height == -1:
                return -1  # Left subtree is not balanced

            right_height = check_height(node.right)
            if right_height == -1:
                return -1  # Right subtree is not balanced

            # Check if current node is balanced
            if abs(left_height - right_height) > 1:
                return -1  # Current node is not balanced

            # Return the height of the tree rooted at this node
            return max(left_height, right_height) + 1

        # Call the helper function on the root
        return check_height(root) != -1

```

### Explanation

1. **TreeNode Class**: We define a simple binary tree node class that contains a value (`val`), a pointer to the left child (`left`), and a pointer to the right child (`right`).

2. **Solution Class**: Within this class, we define the method `isBalanced`.

3. **check_height Function**: This is a recursive helper function that calculates the height of each subtree while verifying if it's balanced.
   - If the current node is `None` (i.e., we are at a leaf's child), we return `0` because the height of an empty tree is `0`.
   - We recursively compute the height of the left subtree first. If the left subtree is found to be unbalanced (indicated by a return value of `-1`), we propagate that signal up by returning `-1`.
   - The same process is repeated for the right subtree.
   - After obtaining heights from both left and right subtrees, we check the balance condition: the absolute difference between the left and right heights must not exceed `1`. If it does, we return `-1` indicating that the tree is not balanced at this node.
   - If it is balanced, we return the height of the tree rooted at the current node, which is `1` greater than the maximum of the heights of the left and right subtrees.

4. **Final Call**: The initial call to `check_height(root)` determines if the tree is height-balanced and returns `True` or `False` based on the final result. If the height check returns anything other than `-1`, the tree is balanced.

This method efficiently checks all nodes of the tree in a single traversal, making it suitable given the constraints provided.

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
 To solve the problem of finding the minimum depth of a binary tree, we can utilize a breadth-first search (BFS) or a depth-first search (DFS) approach. In this explanation, I will implement the BFS method, which is more intuitive for this kind of problem since it explores nodes layer by layer.

### Problem Breakdown
1. **Definition**: The minimum depth of a binary tree is the number of nodes along the shortest path from the root node to the nearest leaf node (a node with no children).
2. **Leaf Nodes**: A leaf node is defined as a node that does not have any children (both left and right children are `None`).
3. **Edge Cases**:
   - If the tree is empty (i.e., the root is `None`), the minimum depth should be `0`.
   - A tree with only one node has a minimum depth of `1`.

### Approach
1. **BFS Algorithm**:
   - We use a queue to traverse the tree level by level.
   - Start from the root and, at each level, check for leaf nodes.
   - Once we encounter a leaf node, we can return the current depth because it's guaranteed to be the shortest path to a leaf found so far.
   
2. **Implementation**:
   - We will check each node, add its children to the queue, and increment the depth until we find a leaf node.
   - In our solution, we'll use Python's `collections.deque` for efficient queue operations.

Here’s how to implement it in a LeetCode solution format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
            
        # Initialize a queue for BFS
        queue = collections.deque([(root, 1)])  # Tuple of (node, current depth)
        
        while queue:
            node, depth = queue.popleft()
            
            # Check if node is a leaf node
            if not node.left and not node.right:
                return depth  # Return the minimum depth as soon as a leaf is found
            
            # Add left child to the queue if it exists
            if node.left:
                queue.append((node.left, depth + 1))
            
            # Add right child to the queue if it exists
            if node.right:
                queue.append((node.right, depth + 1))
                
        return 0  # Just a fallback, it should never happen for valid input

```

### Explanation of the Code
1. **TreeNode Class**: This class defines the structure of each node in the tree.
2. **Solution Class**: We define a function `minDepth` which takes the tree's root as an input.
3. **Edge Case Handling**: If the root is `None`, we immediately return `0` since there are no nodes in the tree.
4. **Queue Initialization**: We initialize a queue and begin with the root node and a depth of `1`.
5. **While Loop**: We continue processing until the queue is empty.
   - Dequeue the current node and its depth.
   - Check if it’s a leaf node. If yes, return the current depth.
   - If the left child exists, enqueue it with an incremented depth.
   - If the right child exists, enqueue it similarly.

This solution is efficient and operates in O(N) time complexity, where N is the number of nodes in the binary tree, while using O(N) space in the worst-case scenario due to the queue.

# 112. Path Sum

### Problem Description 
Given the `root` of a binary tree and an integer `targetSum`, return `true` if the tree has a root-to-leaf path such that adding up all the values along the path equals `targetSum`.

A leaf is a node with no children.


Example 1:
Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
Output: true

Example 2:
Input: root = [1,2,3], targetSum = 5
Output: false

Example 3:
Input: root = [1,2], targetSum = 0
Output: false

Constraints:
The number of nodes in the tree is in the range `[0, 5000]`.

`-1000 <= Node.val <= 1000`
`-1000 <= targetSum <= 1000`

### Solution 
 To solve the problem of finding if there exists a root-to-leaf path in a binary tree such that the sum of the values along the path equals a given target sum, we can use a depth-first search (DFS) approach. 

In this problem:

1. A root-to-leaf path is defined as a path that starts at the root and ends at any leaf node (a node with no children).
2. We will traverse the tree using recursion, maintaining a running sum of the values from the root to the current node.
3. Once we reach a leaf node, we will check if the running sum equals the target sum.
4. If a path is found that meets the criteria, we return `true`. If we exhaust all paths without finding a valid one, we return `false`.

Here’s how we can implement this in Python using the LeetCode submission format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        # Helper function to perform DFS
        def dfs(node: TreeNode, currentSum: int) -> bool:
            if not node:  # If the node is None, return False
                return False

            currentSum += node.val  # Add the current node's value to the running sum
            
            # Check if we're at the leaf node and if the current sum equals the target sum
            if not node.left and not node.right: 
                return currentSum == targetSum
            
            # Recursively check the left and right subtrees
            return dfs(node.left, currentSum) or dfs(node.right, currentSum)

        return dfs(root, 0)  # Start DFS with initial sum of 0

```

### Explanation of the Code:

- **TreeNode Definition**: We define a `TreeNode` class to represent each node in the binary tree.

- **Solution Class**: A `Solution` class encapsulates the method `hasPathSum`.

- **hasPathSum Method**:
  - It initializes a depth-first search (DFS) by calling the inner helper function `dfs` with the root node and an initial sum of `0`.
  
- **dfs Function**:
  - If the current node is `None`, the function returns `False` because it cannot represent a valid path.
  - The value of the current node is added to `currentSum`.
  - If we reach a leaf node (checked by confirming that both left and right children of the node are `None`), we check whether the `currentSum` equals the `targetSum`. If it does, we return `True`.
  - If the current node is not a leaf, the function makes recursive calls to explore both the left and right child nodes, accumulating the `currentSum`.
  - The result of the recursive calls is combined with a logical OR (`or`), meaning if either subtree returns `True`, it implies that there is a valid root-to-leaf path in that subtree.
  
This simple yet effective approach ensures we explore all potential paths in the tree and adhere to the requirements provided in the problem statement.

# 113. Path Sum II

### Problem Description 
Given the `root` of a binary tree and an integer `targetSum`, return all root-to-leaf paths where each path's sum equals `targetSum`.

A leaf is a node with no children.


Example 1:
Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
Output: [[5,4,11,2],[5,8,4,5]]

Example 2:
Input: root = [1,2,3], targetSum = 5
Output: []

Example 3:
Input: root = [1,2], targetSum = 0
Output: []

Constraints:
The number of nodes in the tree is in the range `[0, 5000]`.

`-1000 <= Node.val <= 1000`
`-1000 <= targetSum <= 1000`

### Solution 
 To solve the problem of finding all root-to-leaf paths in a binary tree that sum up to a given target, we can use a depth-first search (DFS) approach. The idea is to traverse the tree while keeping track of the current path and current sum. Whenever we reach a leaf node, we check if the current sum matches the target sum, and if it does, we record the path.

Here's a detailed breakdown of the solution:

1. **TreeNode Definition**: We need a class to define the structure of our binary tree nodes.

2. **DFS Traversal**: We will create a recursive function to perform the DFS. This function will:
   - Update the current path with the current node's value.
   - Update the current sum with the current node's value.
   - Check if the current node is a leaf. If yes, compare the current sum with `targetSum`. If they match, add the current path to our results.
   - If not a leaf, recursively call the function for the left and right children.
   - Backtrack by removing the current node from the path after exploring its children.

3. **Data Structures**: Use a list to keep track of the current path and another list to store the result paths.

4. **Return Result**: Once the DFS completes, return the collected paths.

Now, here’s the complete code structured to fit the LeetCode environment:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        # List to hold the result
        self.result = []
        
        # Helper function for DFS traversal
        def dfs(node, currentPath, currentSum):
            if not node:
                return
            
            # Include the current node to the path and update the sum
            currentPath.append(node.val)
            currentSum += node.val
            
            # Check if it's a leaf node and the current sum matches targetSum
            if not node.left and not node.right:
                if currentSum == targetSum:
                    self.result.append(list(currentPath))

            # If not leaf, traverse left and right children
            else:
                dfs(node.left, currentPath, currentSum)
                dfs(node.right, currentPath, currentSum)

            # Backtrack
            currentPath.pop()

        # Start DFS from the root
        dfs(root, [], 0)
        return self.result

```

### Explanation of the Code:
1. **TreeNode Class**: This class defines the structure of each node in the binary tree. It has attributes for the node's value, and pointers to its left and right children.

2. **Solution Class**: Inside the `Solution` class, we define the `pathSum` method which is our main function.
   - It initializes an empty result list.
   - It then calls a helper DFS function with the root of the tree, an empty path list, and a sum initialized to 0.

3. **DFS Function**: This recursive function processes each node:
   - If the current node is `None`, it returns immediately.
   - It adds the node's value to the current path and updates the running sum.
   - If it’s a leaf node (no left or right children), it checks if the running sum matches the target. If it does, it saves the path.
   - If not a leaf, it makes recursive calls for the left and right children.
   - Finally, it backtracks by removing the last added node from the current path.

4. **Return Statement**: After finishing the DFS, we return the accumulated results.

This implementation efficiently finds all valid root-to-leaf paths that sum to the specified target, adhering to the constraints and expectations set forth by the problem.

# 114. Flatten Binary Tree to Linked List

### Problem Description 
Given the `root` of a binary tree, flatten the tree into a "linked list":
The "linked list" should use the same `TreeNode` class where the `right` child pointer points to the next node in the list and the `left` child pointer is always `null`.

The "linked list" should be in the same order as a pre-order traversal of the binary tree.


Example 1:
Input: root = [1,2,5,3,4,null,6]
Output: [1,null,2,null,3,null,4,null,5,null,6]

Example 2:
Input: root = []
Output: []

Example 3:
Input: root = [0]
Output: [0]

Constraints:
The number of nodes in the tree is in the range `[0, 2000]`.

`-100 <= Node.val <= 100`
Follow up: Can you flatten the tree in-place (with `O(1)` extra space)?

### Solution 
 To solve the problem of flattening a binary tree into a "linked list" using the same `TreeNode` class, we will implement a pre-order traversal of the tree and rearrange the nodes accordingly. The key point is to make sure that the left child of each node becomes `null` and the right child points to the next node in pre-order traversal.

### Detailed Explanation

1. **Understanding the Structure**: 
   - Each `TreeNode` can be represented as:
     

```python
     class TreeNode:
         def __init__(self, val=0, left=None, right=None):
             self.val = val
             self.left = left
             self.right = right
     
```
   - In pre-order traversal, we visit the root first, then the left subtree, followed by the right subtree.

2. **In-Place Flattening**:
   - We'll maintain a pointer to the current node that we are working on. 
   - As we traverse the tree, we will adjust each node's left child to `null` and link the right child to the next node in the pre-order sequence.
   - We can achieve this in O(1) extra space by altering the tree directly without using any additional data structures like lists or stacks.

3. **Algorithm Steps**:
   - Start from the root.
   - Utilize a recursive or iterative approach to process each node.
   - For each node:
     - Save the right child (since we'll be modifying pointers).
     - Recursively flatten the left subtree.
     - Move to the end of the newly flattened left subtree to link to the right child.

### Implementation

Here's the implementation that follows the specified format for LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        # Base case: if the tree is empty, return
        if not root:
            return

        # Flatten the left and right subtree
        self.flatten(root.left)
        self.flatten(root.right)

        # Store the right subtree to be processed later
        right_subtree = root.right

        # After flattening, the left subtree is already flattened,
        # attach it to the right
        root.right = root.left
        root.left = None  # Set left to None

        # Find the end of the new right subtree (which is the flattened left subtree)
        while root.right:
            root = root.right
        
        # Attach the previously stored right subtree
        root.right = right_subtree

```

### Explanation of the Code
- The `flatten` function is defined within the `Solution` class.
- It checks if the current node (`root`) is `None`. If so, it simply returns.
- We recursively call `flatten` on the left and right children of the current node.
- We temporarily store the right subtree because we will modify the `right` pointer of the current node.
- After flattening, we point the `right` child of the current node to the flatten left subtree and set the `left` child to `None`.
- Finally, we traverse to the end of the modified right child to append the previously stored right subtree.

### Time Complexity
- The algorithm runs in O(n) time where n is the number of nodes in the tree since we deal with each node exactly once.

### Space Complexity
- The space complexity is O(1) as we are modifying the tree in place regardless of the input size. The recursion stack in the worst case will take O(h) space where h is the height of the tree, but that's part of the processing of tree nodes.

This solution effectively flattens the binary tree into a linked list in the required order.

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
 To solve the problem of populating each `next` pointer in a perfect binary tree, we can take advantage of the properties of a perfect binary tree. In a perfect binary tree, all leaf nodes are at the same level, and each non-leaf node has exactly two children. Given these properties, we can use a recursive approach to traverse the tree and establish the `next` pointers.

### Solution Explanation

1. **Understanding Node Structure**: Each node has a value (`val`), a pointer to the left child (`left`), a pointer to the right child (`right`), and a pointer to the next right node (`next`).

2. **Recursive Approach**: We can establish the `next` pointers using depth-first traversal:
   - For each node, we link its left child's `next` pointer to its right child.
   - Additionally, if the node has a `next` pointer, we link the right child of the current node to the left child of the next node.

3. **Base Case**: We stop the recursion when we reach a null node.

4. **Space Complexity**: The algorithm runs in constant space, besides the recursion stack space which is acceptable in this problem.

### Implementation

Here’s how we can implement this solution in a format suitable for LeetCode:



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
        
        # Connect the left and right child
        if root.left:
            root.left.next = root.right
            
            # Also connect the right child's next to the left child of the next node
            if root.next:
                root.right.next = root.next.left
            
            # Recursively connect the left and right subtrees
            self.connect(root.left)
            self.connect(root.right)
        
        return root

```

### Explanation of the Code

- We define the `Node` class with the required properties.
- Inside the `Solution` class, the method `connect` takes the root of the binary tree as input.
- The base condition checks if the `root` is `None`. If it is, we return `None`.
- We then connect the left child's `next` pointer to the right child.
- If the current node has a `next` (i.e., it is not the rightmost node at its level), we connect the right child's `next` to the left child of the node's `next`.
- Finally, we recursively call the `connect` function for the left and right children.

### Example Usage

For example, given the tree represented by `root = [1,2,3,4,5,6,7]`, after running `connect`, all `next` pointers will be populated accordingly, allowing us to traverse the tree using these pointers. The output of the method will be the root of the modified tree.

This implementation meets the problem requirements, runs in O(n) time, where n is the number of nodes in the tree, and uses O(1) space beyond the recursion stack.

# 117. Populating Next Right Pointers in Each Node II

### Problem Description 
Given a binary tree
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
Input: root = [1,2,3,4,5,null,7]
Output: [1,#,2,3,#,4,5,7,#]
Explanation: Given the above binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.


Constraints:
The number of nodes in the given tree is less than `6000`.

`-100 <= node.val <= 100`

### Solution 
 To solve the problem of populating the `next` pointers for each node in a perfect binary tree, we can use a recursive depth-first search (DFS) approach. We must ensure that we use O(1) extra space, which implies we cannot use additional data structures like lists or queues, but we can use the implicit stack space from recursion.

Here's a step-by-step breakdown of how we can achieve this:

### Problem Breakdown
1. **Tree Structure**: Each node has a value (`val`), and pointers to its left child (`left`), right child (`right`), and the next right node (`next`).
2. **Next Pointer Logic**:
   - For a given node, if it has a left and a right child, the left child's next pointer should point to the right child.
   - If the node has a right child but no left child, it should point to the next node on the same level (which we will handle recursively).
   - If the node has a left child, that child should also point to the next node as we connect nodes level by level.
3. **Base Case for Recursion**: If the node is null, simply return.

### Implementation Steps
1. Start at the root node.
2. Use a recursive function to connect the `next` pointers.
3. For each node, connect its left child to its right child, or to the next node in its level, if it exists.
4. Call the function recursively on the left and right children.

### Python Code
Here is the code that implements the above logic:



```python
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
        # Connect the left child to the right child directly
        if root.left:
            root.left.next = root.right
            # Connect the right child to the next node's left child if exists
            if root.next:
                root.right.next = root.next.left
                
        # Recur for left and right children
        self.connect(root.left)
        self.connect(root.right)
        
        return root

```

### Explanation
- The `Node` class is defined to represent each node in the tree with a constructor.
- In the `Solution` class, the `connect` method is implemented to perform the following:
  - If the root is `None`, we return it immediately.
  - We connect the left child to the right child of the current node.
  - If the current node has a `next` pointer, we set the right child's `next` pointer to point to the next node's left child.
  - We then call the `connect` function recursively for the left and right children.
- The overall complexity of this approach is O(N) where N is the number of nodes in the tree, and we utilize constant space aside from the recursive stack.

### Example
Given a tree structure `[1,2,3,4,5,null,7]`, this algorithm will correctly populate the `next` pointers so that:
- Node 1's `next` will be `None`,
- Node 2's `next` will be Node 3,
- Node 4's `next` will be Node 5, and so on, creating proper connections across the same level.

You can copy and paste this code directly into a LeetCode submission to test it with given inputs.

# 124. Binary Tree Maximum Path Sum

### Problem Description 
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the `root` of a binary tree, return the maximum path sum of any path.


Example 1:
Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.


Example 2:
Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.


Constraints:
The number of nodes in the tree is in the range `[1, 3 * 104]`.

`-1000 <= Node.val <= 1000`

### Solution 
 To solve the problem of finding the maximum path sum in a binary tree, we can utilize a Depth First Search (DFS) approach. The idea is to traverse the tree and calculate the maximum path sum that can be obtained through each node, considering that the path can start and end at any node.

### Explanation of the Solution

1. **Definition of the Problem**:
   We need to calculate the maximum path sum of any path in the binary tree. A path can be defined by any node and can traverse through its children nodes. 

2. **Understanding the Path Sum**:
   The path sum for a specific node can be calculated as follows:
   - We can choose to navigate left and right from a node.
   - The maximum contribution to the path sum from each side (left or right) should be considered because a path can only continue downward.

3. **Recursive Function**:
   We will define a recursive function that:
   - Computes the maximum path sum starting from a given node.
   - Updates the global maximum path sum whenever we compute a path sum that exceeds the current maximum.

4. **Base Case**:
   If the current node is `None`, we return `0` since there are no values to contribute.

5. **Edge Case**:
   - If a subtree's maximum contribution is negative, it should not be included in the path sum calculation. Therefore, we take the maximum of `0` and the computed contributions from the left and right children.

6. **Global Maximum**:
   We will keep track of the maximum value found during the traversal by using a list to allow for modification within our nested function.

Here’s how the code will be constructed based on the above understanding.

### Python Code



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        self.max_sum = float('-inf')  # Initialize to negative infinity to handle all values

        def max_gain(node):
            if not node:
                return 0  # Base case: no contribution if the node is None
            
            # Recursively get the maximum path sum from left and right children
            left_gain = max(max_gain(node.left), 0)  # Only take positive gains
            right_gain = max(max_gain(node.right), 0)  # Only take positive gains
            
            # Current path price, considering both left and right contributions
            current_path_sum = node.val + left_gain + right_gain
            
            # Update the global maximum path sum
            self.max_sum = max(self.max_sum, current_path_sum)
            
            # Return the maximum gain the current node and its parent can provide
            return node.val + max(left_gain, right_gain)
        
        max_gain(root)  # Start the DFS from the root
        return self.max_sum  # Return the maximum path sum found

```

### Explanation of the Code

- **TreeNode Class**: This class is a standard representation of a node in a binary tree.
- **Solution Class**:
  - `maxPathSum` method initializes the maximum sum and defines the recursive `max_gain` function.
  - `max_gain` function:
    - If the node is `None`, it returns `0`.
    - It calculates the maximum gain from the left and right children.
    - Computes `current_path_sum` to include the value of the current node and its children's contributions.
    - Updates the global `self.max_sum` if `current_path_sum` is greater than the previously recorded maximum.
- It starts the recursive function from the root and finally returns the maximum path sum found.

This solution effectively computes the maximum path sum with a time complexity of O(n), where n is the number of nodes, and a space complexity of O(h) for the recursion stack, h being the height of the tree. This is efficient and suitable for the problem constraints.

# 129. Sum Root to Leaf Numbers

### Problem Description 
You are given the `root` of a binary tree containing digits from `0` to `9` only.

Each root-to-leaf path in the tree represents a number.

For example, the root-to-leaf path `1 -> 2 -> 3` represents the number `123`.

Return the total sum of all root-to-leaf numbers.

A leaf node is a node with no children.


Example 1:
Input: root = [1,2,3]
Output: 25
Explanation:
The root-to-leaf path `1->2` represents the number `12`.

The root-to-leaf path `1->3` represents the number `13`.

Therefore, sum = 12 + 13 = `25`.


Example 2:
Input: root = [4,9,0,5,1]
Output: 1026
Explanation:
The root-to-leaf path `4->9->5` represents the number 495.

The root-to-leaf path `4->9->1` represents the number 491.

The root-to-leaf path `4->0` represents the number 40.

Therefore, sum = 495 + 491 + 40 = `1026`.


Constraints:
The number of nodes in the tree is in the range `[1, 1000]`.

`0 <= Node.val <= 9`
The depth of the tree will not exceed `10`.

### Solution 
 To solve the problem of summing all root-to-leaf numbers in a binary tree, we can use a depth-first search (DFS) approach. For each root-to-leaf path, we will construct the number represented by the path and accumulate it when we reach a leaf node.

Here’s a breakdown of the solution:

1. **Tree Node Structure**: We first define the structure of the binary tree node (`TreeNode`) which has a value and pointers to its left and right children.

2. **DFS Function**: We define a recursive helper function that traverses the tree. This function will take two parameters:
   - The current node being visited.
   - The current number constructed from the root to the current node.

3. **Leaf Node Check**: The function will check if the current node is a leaf (i.e., both left and right children are `None`). If it is a leaf, it adds the current number to a global total.

4. **Recursive Calls**: If the current node has left or right children, we recursively call the DFS function on those children, passing the updated number (formed by appending the current node’s value).

5. **Base Case and Accummulation**: Start the DFS from root and initialize the number as `0`. Once the traversal is complete, the total sum will represent the total sum of all numbers formed by root-to-leaf paths.

Here is the complete Python code formatted for LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        self.total_sum = 0
        
        def dfs(node: TreeNode, current_number: int):
            if not node:  # Base case, return if node is None.
                return
            
            # Update the current number by appending the node's value.
            current_number = current_number * 10 + node.val
            
            # Check if it's a leaf node.
            if not node.left and not node.right:
                self.total_sum += current_number  # Add to total sum if it's a leaf.
                return
            
            # Recursive calls for left and right children.
            dfs(node.left, current_number)
            dfs(node.right, current_number)
        
        # Start DFS from root with initial number 0.
        dfs(root, 0)
        
        return self.total_sum

```

### Explanation of the Code:
- **TreeNode class**: This class defines the structure of each node in the binary tree.
  
- **sumNumbers method**: This method initializes a variable `total_sum` to keep track of the accumulated sum and invokes the `dfs` helper function starting from the root.

- **dfs function**:
  - Takes the current node and the `current_number` as parameters.
  - If the current node is `None`, it simply returns (base case).
  - Updates the `current_number` by multiplying the existing number by `10` (shifting left) and adding the current node's value.
  - If the node is a leaf node, it adds the `current_number` to `total_sum`.
  - Recursively calls itself for the left and right children.

This solution efficiently traverses the binary tree and computes the sum in O(N) time complexity, where N is the number of nodes in the tree, and it uses O(H) space for the recursive call stack, where H is the height of the tree.

# 144. Binary Tree Preorder Traversal

### Problem Description 
Given the `root` of a binary tree, return the preorder traversal of its nodes' values.


Example 1:
Input: root = [1,null,2,3]
Output: [1,2,3]

Example 2:
Input: root = []
Output: []

Example 3:
Input: root = [1]
Output: [1]

Example 4:
Input: root = [1,2]
Output: [1,2]

Example 5:
Input: root = [1,null,2]
Output: [1,2]

Constraints:
The number of nodes in the tree is in the range `[0, 100]`.

`-100 <= Node.val <= 100`
Follow up: Recursive solution is trivial, could you do it iteratively?

### Solution 
 To solve the problem of returning the preorder traversal of a binary tree, both recursive and iterative methods can be employed. In this explanation, I'll focus on the iterative method, as it is typically more challenging and provides a good approach for understanding tree traversal without the use of the call stack.

### Preorder Traversal

In a preorder traversal, we visit the nodes of the tree in the following order:
1. Visit the root node.
2. Traverse the left subtree.
3. Traverse the right subtree.

### Iterative Approach

To achieve an iterative preorder traversal, we can use a stack to keep track of nodes. The strategy is as follows:
1. Start with the root node and push it onto the stack.
2. While there are nodes in the stack:
   - Pop the top node from the stack and add its value to the result list.
   - Push the right child of the popped node onto the stack (if it exists).
   - Push the left child of the popped node onto the stack (if it exists).
3. Continue until the stack is empty.

This method ensures that we traverse the root before its children, maintaining the preorder traversal order.

### Implementation in Python

Below is the Python implementation formatted to be directly runnable on the LeetCode platform:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        # List to store the preorder traversal
        result = []
        # Stack to keep track of nodes
        stack = []
        
        # Start with the root node (if it's not null)
        if root:
            stack.append(root)
        
        # While there are still nodes to process
        while stack:
            # Pop the node from the top of the stack
            node = stack.pop()
            # Append the value of the node to the result
            result.append(node.val)
            
            # Push the right child first so that the left child is processed next
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        
        # Return the preorder traversal result
        return result

```

### Explanation of the Code

1. **TreeNode Definition**: This class represents each node in the binary tree. It has properties for `val`, `left`, and `right`.
   
2. **Solution Class**: The `Solution` class contains the method `preorderTraversal` which implements the preorder traversal.

3. **Initialization**:
   - An empty list `result` is created to store the traversal result.
   - A `stack` is initialized to manage the nodes.

4. **Processing the Root**:
   - If the `root` is not null, it is pushed onto the stack.

5. **Traversal Loop**:
   - The loop continues until the stack is empty.
   - The top node is popped from the stack, and its value is appended to the `result` list.
   - The right child is pushed onto the stack first (if it exists) followed by the left child. This is done because the stack is LIFO (Last In, First Out), meaning the left child will be processed before the right child as required in preorder.

6. **Return the Result**: Finally, the `result` list containing the preorder traversal of the tree is returned.

### Complexity Analysis

- **Time Complexity**: O(N), where N is the number of nodes in the tree, since we visit each node once.
- **Space Complexity**: O(H), where H is the height of the tree due to the stack space used. In the worst case, this is O(N) for a skewed tree.

This code can be submitted directly on LeetCode, and it should work correctly for the provided input formats.

# 145. Binary Tree Postorder Traversal

### Problem Description 
Given the `root` of a binary tree, return the postorder traversal of its nodes' values.


Example 1:
Input: root = [1,null,2,3]
Output: [3,2,1]

Example 2:
Input: root = []
Output: []

Example 3:
Input: root = [1]
Output: [1]

Example 4:
Input: root = [1,2]
Output: [2,1]

Example 5:
Input: root = [1,null,2]
Output: [2,1]

Constraints:
The number of the nodes in the tree is in the range `[0, 100]`.

`-100 <= Node.val <= 100`
Follow up:
Recursive solution is trivial, could you do it iteratively?

### Solution 
 To solve the problem of performing a postorder traversal of a binary tree, we can implement both recursive and iterative solutions in Python. Here, I'll provide the iterative solution since the follow-up specifically requests it.

### Explanation of Postorder Traversal

Postorder traversal visits the nodes in the following order:
1. Traverse the left subtree.
2. Traverse the right subtree.
3. Visit the root node.

For the iterative solution, we will use a stack to simulate the recursive behavior. The main idea is to traverse the tree in a way that allows us to visit the root after the left and right children have been processed.

### Iterative Postorder Traversal Algorithm

1. If the root is `None`, return an empty list.
2. Use a stack to keep track of nodes, and a result list to store the postorder traversal.
3. We push the root to the stack.
4. Pop nodes from the stack and push their children onto the stack. However, we will maintain the order by pushing the root last.
5. Once we've processed all nodes, we can reverse the result list to achieve postorder since we pushed the root last.
6. Lastly, return the result list.

### Python Code

We will create a `TreeNode` class for our binary tree nodes and then implement the postorder traversal function. Here's the code structured to work directly on the LeetCode platform:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def postorderTraversal(self, root: TreeNode) -> list[int]:
        if root is None:
            return []

        stack = [root]
        result = []

        while stack:
            node = stack.pop()
            result.append(node.val)  # Process the current node

            # Push left and then right to stack so that right is processed first
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)

        # The result is currently in "root-right-left" order, so we reverse it
        return result[::-1]

```

### Explanation of the Code

1. **TreeNode Class**: We define a `TreeNode` class to represent nodes in the binary tree, which has a value, a left child, and a right child.
2. **Solution Class**: The `Solution` class contains the method `postorderTraversal`.
3. **Base Case**: We immediately check if the root is `None` and return an empty list.
4. **Stack Initialization**: A stack is initialized with the root node.
5. **Processing Loop**: 
   - We pop the top node from the stack, append its value to the result list.
   - We push the left child first (if it exists), followed by the right child. This ensures the right child is processed before the left child when we pop from the stack.
6. **Reverse Result List**: Since we have collected the values in "root-right-left" order, we reverse the result to get "left-right-root" (postorder) order before returning it.

### Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes in the tree, as we visit each node exactly once.
- **Space Complexity**: O(n) in the worst case for the stack (in a skewed tree) and O(n) for the result list.

This code can be run directly on the LeetCode platform and should produce the desired postorder traversal output for the binary tree input.

# 156. Binary Tree Upside Down

### Problem Description 
Given the `root` of a binary tree, turn the tree upside down and return the new root.

You can turn a binary tree upside down with the following steps:
The original left child becomes the new root.

The original root becomes the new right child.

The original right child becomes the new left child.

The mentioned steps are done level by level, it is guaranteed that every node in the given tree has either 0 or 2 children.


Example 1:
Input: root = [1,2,3,4,5]
Output: [4,5,2,null,null,3,1]

Example 2:
Input: root = []
Output: []

Example 3:
Input: root = [1]
Output: [1]

Constraints:
The number of nodes in the tree will be in the range `[0, 10]`.

`1 <= Node.val <= 10`
`Every node has either 0 or 2 children.`

### Solution 
 To solve the problem of turning a binary tree upside down, we need to follow the transformation steps outlined: where the left child of each node becomes the new root, the original root becomes the new right child, and the right child becomes the new left child.

The algorithm can be implemented using recursion to effectively traverse the tree and reassign the child nodes as required. This can be achieved in the following steps:

1. **Base Case**: If the current node (`root`) is `None`, return `None` as we have no tree to transform. If the current node has no left child (leaf node case), return the node itself, as it remains the same.

2. **Recursive Case**: 
   - Recursively call the function on the left child of the current node. This will eventually lead us to the bottom left child of the tree, which will become the new root of the upside-down tree.
   - After the recursive call, we need to adjust pointers:
     - Set the right child of the left child to the original root.
     - Set the left child of the left child to the original right child.
     - Finally, set the left and right pointers of the original root to `None` to avoid incorrect references.
    
3. **Return**: The result of the recursive call, which is the new root of the transformed tree.

Here is the Python code implementing the above logic in LeetCode format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def upsideDownBinaryTree(self, root: TreeNode) -> TreeNode:
        if not root:  # If the root is None
            return None
        if not root.left:  # If there's no left child, this is a leaf node
            return root
        
        new_root = self.upsideDownBinaryTree(root.left)  # Recurse down to the leftmost node
        
        # After recursion, flip the pointers
        root.left.left = root.right  # Original left becomes the new root's left child
        root.left.right = root  # Original root becomes the new root's right child
        
        # Set left and right of original root to None
        root.left = None
        root.right = None
        
        return new_root  # The leftmost node is now the new root

```

### Explanation of Code:
- **TreeNode Class**: This is the basic structure for our tree nodes, where each node has a value (`val`), a left child (`left`), and a right child (`right`).
- **Solution Class**: Contains the method `upsideDownBinaryTree`.
- **Function Logic**:
  - The function checks for empty nodes and leaf nodes that do not require flipping.
  - It recursively processes the left subtree until it finds the new root.
  - After reaching the leftmost node, it rewires the children to create the upside-down structure.
  - Finally, it returns the new root of the flipped tree.

### Complexity:
- **Time Complexity**: O(n), where n is the number of nodes in the tree, as we visit each node once.
- **Space Complexity**: O(h), where h is the height of the tree due to recursion stack space.

# 173. Binary Search Tree Iterator

### Problem Description 
Implement the `BSTIterator` class that represents an iterator over the in-order traversal of a binary search tree (BST):
`BSTIterator(TreeNode root)` Initializes an object of the `BSTIterator` class. The `root` of the BST is given as part of the constructor. The pointer should be initialized to a non-existent number smaller than any element in the BST.

`boolean hasNext()` Returns `true` if there exists a number in the traversal to the right of the pointer, otherwise returns `false`.

`int next()` Moves the pointer to the right, then returns the number at the pointer.

Notice that by initializing the pointer to a non-existent smallest number, the first call to `next()` will return the smallest element in the BST.

You may assume that `next()` calls will always be valid. That is, there will be at least a next number in the in-order traversal when `next()` is called.


Example 1:
Input
["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
Output
[null, 3, 7, true, 9, true, 15, true, 20, false]
Explanation
BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
bSTIterator.next();    // return 3
bSTIterator.next();    // return 7
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 9
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 15
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 20
bSTIterator.hasNext(); // return False

Constraints:
The number of nodes in the tree is in the range `[1, 105]`.

`0 <= Node.val <= 106`
At most `105` calls will be made to `hasNext`, and `next`.

Follow up:
Could you implement `next()` and `hasNext()` to run in average `O(1)` time and use `O(h)` memory, where `h` is the height of the tree?

### Solution 
 To implement the `BSTIterator` class that allows for in-order traversal of a binary search tree (BST), we can take advantage of the properties of the BST and the stack data structure for managing the traversal process. Here’s a detailed explanation of the implementation, followed by the complete code formatted to run directly on LeetCode.

### Explanation:

1. **In-Order Traversal**: In a binary search tree, an in-order traversal yields values in non-decreasing order (left-root-right). We need to be able to keep track of the current position in this traversal.

2. **Using a Stack**: The basic idea is to use a stack to help us traverse down to the leftmost node whenever we're looking for the next smallest element. The stack will hold the nodes we haven't fully processed yet.

3. **Initialization**: When we create the `BSTIterator`, we traverse to the leftmost node of the BST, pushing nodes onto the stack as we go. This way, the top of the stack always gives us the next smallest element.

4. **`hasNext()` Method**: This method simply checks if there are any nodes left in the stack. If the stack is not empty, it means there are more elements to traverse.

5. **`next()` Method**: This method pops the top node from the stack (which is the next smallest node), checks if this node has a right child (which we then need to process), and if it has a right child, push all left children of that right child onto the stack for future calls.

### Complete Code:

Here is the complete implementation of the `BSTIterator` class:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BSTIterator:
    def __init__(self, root: TreeNode):
        self.stack = []
        self.push_left_nodes(root)
    
    def push_left_nodes(self, node):
        while node:
            self.stack.append(node)
            node = node.left
            
    def hasNext(self) -> bool:
        return len(self.stack) > 0
    
    def next(self) -> int:
        # Pop the top node from the stack
        next_node = self.stack.pop()
        # If there is a right child, push all its left nodes to the stack
        if next_node.right:
            self.push_left_nodes(next_node.right)
        # Return the value of the next node
        return next_node.val

```

### How to Use:

1. **Initialization**: To initialize the iterator, you'll create a `TreeNode` structure for your BST and pass the root node to `BSTIterator`.
   
2. **Calling Methods**: You will call `hasNext()` to check if there are more elements to traverse and `next()` to get the next element in the in-order traversal order.

### Example Usage:

When you want to test or use the `BSTIterator`, you could use it like this:



```python
# Example tree creation
#        7
#       / \
#      3   15
#         /  \
#        9   20
root = TreeNode(7)
root.left = TreeNode(3)
root.right = TreeNode(15, TreeNode(9), TreeNode(20))

iterator = BSTIterator(root)

results = []
results.append(iterator.next())    # returns 3
results.append(iterator.next())    # returns 7
results.append(iterator.hasNext()) # returns True
results.append(iterator.next())    # returns 9
results.append(iterator.hasNext()) # returns True
results.append(iterator.next())    # returns 15
results.append(iterator.hasNext()) # returns True
results.append(iterator.next())    # returns 20
results.append(iterator.hasNext()) # returns False

```

This implementation is efficient with an average time complexity of \( O(1) \) for `next()` and `hasNext()` calls and uses an \( O(h) \) space complexity, where \( h \) is the height of the tree.

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
 To solve the problem of viewing a binary tree from the right side and returning the visible nodes from top to bottom, we can perform a level order traversal of the binary tree and record the last node encountered at each level. This will give us the values of the nodes that are visible from the right side.

### Explanation

1. **Level Order Traversal**: We will use a queue to help us traverse the tree level by level (breadth-first search).
  
2. **Track the Last Node**: For each level, we'll keep track of the last node encountered. This node will be the one that is visible from the right side.

3. **Edge Cases**: If the root is `None`, we will return an empty list.

### Implementation Steps

- Check if the `root` is `None`. If yes, return an empty list.
- Use a queue to traverse the binary tree. Start by adding the `root` node to the queue.
- For each level, iterate through the nodes until the queue is empty, while also counting the number of nodes at the current level.
- For each node processed at the current level, we add its left child (if it exists) and its right child (if it exists) to the queue for processing in the next level.
- Record only the last node's value at each level into the result list.
- Return the result list at the end of the traversal.

### LeetCode Format

Below is the Python code provided in the LeetCode solution format.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        # List to store the right side view
        result = []
        
        if not root:
            return result
        
        # Queue for level order traversal
        queue = [root]

        while queue:
            level_length = len(queue)  # Number of nodes at the current level
            
            for i in range(level_length):
                node = queue.pop(0)  # Get the front node in the queue
                
                # If it's the last node of this level, add its value to result
                if i == level_length - 1:
                    result.append(node.val)
                
                # Add left and right children of the current node to the queue
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return result

```

### Explanation of the Code

- **TreeNode Class**: This class defines the structure for each node in the binary tree, with optional left and right child nodes.
  
- **Solution Class**: Contains the `rightSideView` function.
  
- **Function Logic**:
  - Initializes an empty list `result` to collect visible nodes.
  - Checks if `root` is `None`, in which case it returns an empty list.
  - Initializes a queue with the root node.
  - Processes nodes level by level. For each level, it updates the queue with the children of the currently processed nodes.
  - At each level, only the last node's value is appended to the result list.
  
By following the above steps, this code accurately captures the visible nodes from the right side of the binary tree and adheres to the expected format used in LeetCode submissions.

# 222. Count Complete Tree Nodes

### Problem Description 
Given the `root` of a complete binary tree, return the number of the nodes in the tree.

According to Wikipedia, every level, except possibly the last, is completely filled in a complete binary tree, and all nodes in the last level are as far left as possible. It can have between `1` and `2h` nodes inclusive at the last level `h`.


Example 1:
Input: root = [1,2,3,4,5,6]
Output: 6

Example 2:
Input: root = []
Output: 0

Example 3:
Input: root = [1]
Output: 1

Constraints:
The number of nodes in the tree is in the range `[0, 5 * 104]`.

`0 <= Node.val <= 5 * 104`
The tree is guaranteed to be complete.

Follow up: Traversing the tree to count the number of nodes in the tree is an easy solution but with `O(n)` complexity. Could you find a faster algorithm?

### Solution 
 To solve the problem of counting nodes in a complete binary tree with a more efficient approach than `O(n)`, we can leverage the properties of complete binary trees. 

In a complete binary tree:
- Every level is fully filled except possibly the last level.
- All nodes in the last level are as far left as possible.

### Approach

1. **Calculate the Height of the Tree**: We can determine the tree's height by traversing down the leftmost path, counting the number of nodes. This operation takes `O(log n)` time because a complete binary tree's height is logarithmic relative to the number of nodes.

2. **Count Nodes Using Binary Search**: Given the height, we use binary search to determine if a node exists at a specific index in the last level. This can be done with the following steps:
   - For each potential index in the last level (from 0 to `2^h - 1`), check if the node at that index exists by traversing the tree using its height to locate the index.
   - If a node exists, we can increment our count.

This combined approach of height calculation and binary search allows us to achieve an overall time complexity of `O(log^2 n)`.

### Implementation

Here is the Python code formatted for LeetCode:



```python
# Definition for a complete binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        # Function to calculate the height of the tree
        def getHeight(node):
            height = 0
            while node.left:
                height += 1
                node = node.left
            return height
        
        height = getHeight(root)
        if height == 0:
            return 1  # The tree has only the root node
        
        # Binary search to count nodes in the last level
        left, right = 0, (1 << height) - 1  # range [0, 2^height - 1]
        while left <= right:
            mid = (left + right) // 2
            if self.exists(mid, height, root):
                left = mid + 1  # This means this mid index exists, move to the right half
            else:
                right = mid - 1  # Move to the left half
        
        return (1 << height) - 1 + left  # Total nodes = full tree + left valid nodes count
    
    def exists(self, index: int, height: int, node: TreeNode) -> bool:
        """Check if the index-th node exists in the last level of the tree."""
        left, right = 0, (1 << height) - 1  # range [0, 2^height - 1]
        for _ in range(height):
            mid = (left + right) // 2
            if index <= mid:
                node = node.left
                right = mid
            else:
                node = node.right
                left = mid + 1
        return node is not None

```

### Explanation of the Code

- **TreeNode Class**: This is a standard definition of a binary tree node holding a value and pointers to left and right child nodes.
  
- **countNodes Function**: This is the main function to count the nodes, using the helper function `getHeight` to get the height of the tree.

- **getHeight Function**: This function increments the height by traversing left until there are no more left children.

- **exists Function**: This function checks whether a node exists at a given index using binary search in the last level and returns a boolean indicating its existence.

- **Counting Nodes**: The final count of nodes combines the nodes in the fully filled upper levels `(1 << height) - 1` with the last level nodes counted with `left`.

This solution is efficient and works well within the constraints provided, ensuring you can find the count without visiting every node, achieving a log-squared performance.

# 226. Invert Binary Tree

### Problem Description 
Given the `root` of a binary tree, invert the tree, and return its root.


Example 1:
Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]

Example 2:
Input: root = [2,1,3]
Output: [2,3,1]

Example 3:
Input: root = []
Output: []

Constraints:
The number of nodes in the tree is in the range `[0, 100]`.

`-100 <= Node.val <= 100`

### Solution 
 To solve the problem of inverting a binary tree, we will employ a simple recursive approach. The idea is to swap the left and right child nodes of each node in the tree. This gives a mirrored version of the original binary tree.

Here's how we will approach this:

1. **Base Case**: If the current node is `None`, we simply return `None` since there is nothing to invert.

2. **Swap Children**: For each node we encounter, we swap its left and right children.

3. **Recursive Call**: We then recursively call the function for the left and right children (which have been swapped).

4. **Return the Root**: Finally, we return the root of the inverted tree.

The algorithm inherently visits each node exactly once, resulting in a time complexity of O(n), where n is the number of nodes in the tree. The space complexity is O(h) due to the recursive stack, where h is the height of the tree.

Now, let's implement this in the format required by LeetCode.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        # If the current node is None, return None
        if root is None:
            return None
        
        # Swap the left and right children
        root.left, root.right = root.right, root.left
        
        # Recursively invert the left and right subtree
        self.invertTree(root.left)
        self.invertTree(root.right)
        
        # Return the root of the inverted tree
        return root

```

### Explanation of the Code

- **TreeNode Class**: We define a class `TreeNode` with an initializer that sets the value of the node (`val`) and its left and right children (`left` and `right`).

- **Solution Class**: Within the `Solution` class, we define the method `invertTree` which takes the root of the binary tree as an argument.

- **Checking for None**: The first line inside the `invertTree` function checks if the root is `None`. If it is, we return `None`.

- **Swapping**: The line `root.left, root.right = root.right, root.left` swaps the left and right children of the current node.

- **Recursive Calls**: We call `invertTree` recursively on the left and right children of the current node. 

- **Return Statement**: Finally, we return the root of the newly inverted tree.

### Usage
To use this code, you would typically create a binary tree using the `TreeNode` class and call `invertTree` with the root node. The provided code can be directly run on LeetCode to check for correctness against the various test cases given.

# 230. Kth Smallest Element in a BST

### Problem Description 
Given the `root` of a binary search tree, and an integer `k`, return the `kth` (1-indexed) smallest element in the tree.


Example 1:
Input: root = [3,1,4,null,2], k = 1
Output: 1

Example 2:
Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3

Constraints:
The number of nodes in the tree is `n`.

`1 <= k <= n <= 104`
`0 <= Node.val <= 104`
Follow up: If the BST is modified often (i.e., we can do insert and delete operations) and you need to find the kth smallest frequently, how would you optimize?

### Solution 
 To solve the problem of finding the k-th smallest element in a binary search tree (BST), we can take advantage of the properties of the BST. In a BST:

- The left subtree of a node contains only nodes with values less than the node's value.
- The right subtree of a node contains only nodes with values greater than the node's value.
- Both the left and right subtrees are also binary search trees.

This property enables us to perform an in-order traversal (which visits nodes in ascending order) to find the k-th smallest element efficiently.

### Plan:
1. Perform an in-order traversal of the BST while counting the visited nodes.
2. When we reach the k-th node during this traversal, we return its value.
3. To prevent excessive branching and make the solution clean, we can use a generator to yield nodes one by one during traversal.

### Implementation:

Here's the Python code structured to fit the LeetCode solution format. We will define a `TreeNode` class to represent nodes in the tree and a function `kthSmallest` to find the k-th smallest element.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        # Generator function for in-order traversal of the BST.
        def in_order_traversal(node):
            if node:
                yield from in_order_traversal(node.left)  # Left subtree
                yield node.val  # Current node
                yield from in_order_traversal(node.right)  # Right subtree

        # Create a generator from the in-order traversal
        gen = in_order_traversal(root)
        
        # Return the k-th smallest element by advancing the generator k times
        for _ in range(k):
            kth_smallest_value = next(gen)
        return kth_smallest_value

```

### Explanation:
- **TreeNode Class**: This class defines the structure of a node in a binary tree, with a constructor initializing the node's value and its left and right children.
  
- **kthSmallest Method**: This method is the main function that:
  - Defines an inner function `in_order_traversal(node)`, which is a generator for an in-order traversal of the BST.
  - The generator uses `yield from` to recursively yield values from the left subtree, the current node value, and then from the right subtree.
  
- **Finding the k-th Smallest Element**: 
  - We create a generator object `gen` to start the in-order traversal from the root.
  - Using a loop, we call `next(gen)` k times to obtain the k-th smallest value and return it.

### Complexity:
- **Time Complexity**: O(H + k), where H is the height of the tree. In the worst case, this could be O(n) if the tree is unbalanced (degenerate).
- **Space Complexity**: O(H) for the recursion stack during the traversal.

### Follow Up Optimization:
If the BST is frequently modified (insertion and deletion), maintaining an augmented tree structure or using balanced BSTs like AVL trees or Red-Black trees, where you also store the size of the subtree at each node, could optimize finding the k-th smallest element, yielding O(log n) time complexity for each insert/delete operation and also O(log n) for finding the k-th smallest element.

# 235. Lowest Common Ancestor of a Binary Search Tree

### Problem Description 
Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes `p` and `q` as the lowest node in `T` that has both `p` and `q` as descendants (where we allow a node to be a descendant of itself).”

Example 1:
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.


Example 2:
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
Output: 2
Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.


Example 3:
Input: root = [2,1], p = 2, q = 1
Output: 2

Constraints:
The number of nodes in the tree is in the range `[2, 105]`.

`-109 <= Node.val <= 109`
All `Node.val` are unique.

`p != q`
`p` and `q` will exist in the BST.

### Solution 
 To solve the problem of finding the Lowest Common Ancestor (LCA) of two nodes in a Binary Search Tree (BST), we can utilize the properties of BSTs, which allow us to efficiently determine the relationship between the nodes `p` and `q`.

### Key Insights:

1. **Properties of a BST**:
   - For any given node `n`, all nodes in the left subtree of `n` have values less than `n.val`.
   - All nodes in the right subtree have values greater than `n.val`.

2. **Finding the LCA**:
   - Starting from the root of the BST, we can navigate the tree based on the values of `p` and `q`:
     - If both `p` and `q` are less than the current node's value, then the LCA lies in the left subtree.
     - If both `p` and `q` are greater than the current node's value, then the LCA lies in the right subtree.
     - If one of `p` or `q` is equal to the current node's value, or if they lie on opposite sides of the current node, then the current node is the LCA.

### Implementation:

Here's the Python function to find the LCA of two nodes in a BST. This implementation follows the format typically used in LeetCode solutions.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        # Start from the root node
        while root:
            # If both p and q are less than root, recurse left
            if p.val < root.val and q.val < root.val:
                root = root.left
            # If both p and q are greater than root, recurse right
            elif p.val > root.val and q.val > root.val:
                root = root.right
            # We have found the split point, i.e., the LCA
            else:
                return root

```

### Explanation of the Code:

1. **TreeNode Class**: 
   - We define a `TreeNode` class representing each node in the binary tree, initialized with a value `val`, and pointers to its left and right children.

2. **Solution Class**:
   - We create a `Solution` class with a method `lowestCommonAncestor` that takes the root of the tree and two nodes `p` and `q`.

3. **Finding LCA**:
   - We use a while loop to traverse the tree starting from the `root`.
   - Depending on the values of `p` and `q` in relation to the current `root`, we decide to navigate left or right.
   - If we encounter a situation where `p` and `q` split (meaning one is less and the other is greater, or one is equal to the current node), we return the current node as the LCA.

### Complexity:
- **Time Complexity**: O(h), where `h` is the height of the tree. In the worst case (unbalanced tree), this could be O(n).
- **Space Complexity**: O(1) for the iterative approach since we are not using any additional data structures.

### Example Usage:
To use the above code on LeetCode, you simply need to instantiate the `TreeNode` objects to define your BST and call the `lowestCommonAncestor` method with the root and the two nodes.

This structure adheres to the expected LeetCode format and can be run directly to find the LCA of two nodes in a BST.

# 236. Lowest Common Ancestor of a Binary Tree

### Problem Description 
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes `p` and `q` as the lowest node in `T` that has both `p` and `q` as descendants (where we allow a node to be a descendant of itself).”

Example 1:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.


Example 2:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.


Example 3:
Input: root = [1,2], p = 1, q = 2
Output: 1

Constraints:
The number of nodes in the tree is in the range `[2, 105]`.

`-109 <= Node.val <= 109`
All `Node.val` are unique.

`p != q`
`p` and `q` will exist in the tree.

### Solution 
 Certainly! Below is a Python code to solve the problem of finding the lowest common ancestor (LCA) in a binary tree. The code is structured to be run directly on the LeetCode platform, and includes detailed explanations of the logic behind the solution.

### Problem Explanation

The objective is to find the Lowest Common Ancestor (LCA) of two nodes in a binary tree. The LCA is the deepest node that is an ancestor of both given nodes. We can approach this problem using a recursive depth-first search (DFS) strategy.

### Approach

1. **Base Case**: If the current node is `None`, we return `None`. If the current node is either `p` or `q`, we return the current node.
   
2. **Recursive Search**: We perform DFS on both the left and right children of the current node. 
   - If both left and right recursive calls return a non-null value, it means `p` and `q` are located in the left and right subtrees respectively, thus the current node is their LCA.
   - If one of the calls returns a non-null value, we propagate that value upwards since it means either `p` or `q` was found in that subtree.

3. **Final Return**: When we reach the root, if we receive a value that is not `None`, it will be the LCA.

### Implementation



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        # Base case: If the tree is empty or we found one of the nodes
        if root is None:
            return None
        if root == p or root == q:
            return root
        
        # Search in the left and right subtree
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        # If both left and right are not None, it means p and q are found in different subtrees
        if left and right:
            return root
        
        # Otherwise, return the non-None child (it could be either the left or right)
        return left if left is not None else right

```

### Example Explanation

1. **Example 1**:
   - Input: `root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1`
   - Output: `3`
   - Explanation: The path is `3 -> 5` for node 5 and `3 -> 1` for node 1, thus LCA is `3`.

2. **Example 2**:
   - Input: `root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4`
   - Output: `5`
   - Explanation: Node 5 is a descendant of itself, so the LCA is `5`.

3. **Example 3**:
   - Input: `root = [1,2], p = 1, q = 2`
   - Output: `1`
   - Explanation: Node 1 is the ancestor of itself and also of node 2, so LCA is `1`.

### Complexity Analysis

- **Time Complexity**: O(N), where N is the number of nodes in the binary tree, because we may need to visit each node to find the two nodes.
- **Space Complexity**: O(H), where H is the height of the tree, due to the call stack used in recursion.

This implementation follows a clear and structured approach to diagnose and retrieve the lowest common ancestor, while maintaining efficiency.

# 250. Count Univalue Subtrees

### Problem Description 
Given the `root` of a binary tree, return the number of uni-value subtrees.

A uni-value subtree means all nodes of the subtree have the same value.


Example 1:
Input: root = [5,1,5,5,5,null,5]
Output: 4

Example 2:
Input: root = []
Output: 0

Example 3:
Input: root = [5,5,5,5,5,null,5]
Output: 6

Constraints:
The numbrt of the node in the tree will be in the range `[0, 1000]`.

`-1000 <= Node.val <= 1000`

### Solution 
 To solve the problem of counting the number of uni-value subtrees in a binary tree, we will use a recursive approach, performing a post-order traversal on the tree. A uni-value subtree is defined as a subtree in which all nodes have the same value.

### Steps to solve the problem:

1. **Definition**: We will define a recursive helper function, which will determine if a subtree is a uni-value subtree while counting them.
  
2. **Base Case**: If the node is `None`, we consider it to be a uni-value subtree (return `True` and increment count by 0).

3. **Recursive Case**: For a given node:
   - Recursively check its left and right subtrees.
   - If both left and right subtrees are uni-value, then we proceed to check:
     - If the left subtree exists, check if its value matches the current node’s value.
     - If the right subtree exists, check if its value matches the current node’s value.
   - If all conditions are satisfied, we have found a uni-value subtree and increment our count.

4. **Counting**: Maintain a count variable which we will return from the helper function.

5. **Final Count**: Return the total count of uni-value subtrees when the initial function call completes.

Here is the complete Python code for the solution:



```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def countUniValSubtrees(self, root: TreeNode) -> int:
        self.count = 0  # Initialize the count of uni-value subtrees

        def is_uni_value_subtree(node: TreeNode) -> bool:
            # Base case: an empty subtree is considered a uni-value subtree
            if not node:
                return True
            
            # Check if left and right subtrees are uni-value
            left_is_uni = is_uni_value_subtree(node.left)
            right_is_uni = is_uni_value_subtree(node.right)

            # If both subtrees are uni-value, check current node's value
            if left_is_uni and right_is_uni:
                # If left child exists, check its value
                if node.left and node.left.val != node.val:
                    return False
                # If right child exists, check its value
                if node.right and node.right.val != node.val:
                    return False
                
                # If it is a uni-value subtree, increment the count
                self.count += 1
                return True
            
            return False

        is_uni_value_subtree(root)  # Start the recursion
        return self.count  # Return the total count of uni-value subtrees

```

### Explanation of the Code:

- **TreeNode Class**: A simple definition for tree nodes with `val`, `left`, and `right`.
  
- **countUniValSubtrees Method**:
  - Initializes a `count` attribute to store the number of uni-value subtrees.
  - Defines the helper function `is_uni_value_subtree` which checks if a given subtree is uni-value and updates the count accordingly.
  
- **Function Call**: The helper function is initially called with `root`, and once the processing is complete, the count of uni-value subtrees is returned.

### Complexity Analysis:
- **Time Complexity**: O(n), where n is the number of nodes in the tree. Each node is processed once.
- **Space Complexity**: O(h), where h is the height of the tree. This accounts for the recursive stack space used during the traversal.

This solution can be directly used on the LeetCode platform by creating a test for the `countUniValSubtrees` method using instances of `TreeNode` as inputs.

# 255. Verify Preorder Sequence in Binary Search Tree

### Problem Description 
Given an array of unique integers `preorder`, return `true` if it is the correct preorder traversal sequence of a binary search tree.


Example 1:
Input: preorder = [5,2,1,3,6]
Output: true

Example 2:
Input: preorder = [5,2,6,1,3]
Output: false

Constraints:
`1 <= preorder.length <= 104`
`1 <= preorder[i] <= 104`
All the elements of `preorder` are unique.

Follow up: Could you do it using only constant space complexity?

### Solution 
 To determine whether a given array represents the preorder traversal of a binary search tree (BST), we need to consider the properties of a BST:

1. For any node in a BST:
   - All values in the left subtree are less than the node's value.
   - All values in the right subtree are greater than the node's value.
   
In a preorder traversal, we visit the root node first, then recursively visit the left subtree, followed by the right subtree. This order helps us maintain the BST properties.

Let's break down the algorithm to verify if the given `preorder` sequence is indeed a valid preorder traversal of a BST:

### Steps:

1. **Initialize Boundaries**: Start with a range where the first element can be considered as the root of the BST. Initially, the lower bound is negative infinity and the upper bound is positive infinity.

2. **Recursively Check Validity**:
   - For each node, check if it is within the valid bounds.
   - If it is, proceed to check the next nodes:
     - All left children must be smaller than the current node (set the upper bound to the current node value).
     - All right children must be larger than the current node (set the lower bound to the current node value).
   - Use a stack or an index to track the current position in the array while maintaining the boundaries.

3. **Constant Space Complexity**: Instead of using a separate stack for recursion, we can use the input list itself to simulate a stack and keep track of the current index. This allows us to maintain O(1) space complexity with respect to the input size.

### Python Code:

Now, let's implement the solution in Python. This solution uses an iterative approach with constant space complexity.



```python
class Solution:
    def verifyPreorder(self, preorder: List[int]) -> bool:
        low = float('-inf')  # Starting lower bound
        stack = []  # Use stack to track the path in the BST

        for value in preorder:
            # A violation of BST properties
            if value < low:
                return False
            
            # If current value is greater than the stack's last, 
            # pop until we find a value higher than the current value.
            while stack and value > stack[-1]:
                low = stack.pop()  # Update the lower bound
            
            # Push the current value to the stack
            stack.append(value)

        return True

```

### Explanation:
1. **Initialization**: We start by defining `low` as negative infinity to allow the root node to be any valid integer from the input array.

2. **Loop through the preorder array**:
   - Check if the current value is less than `low`. If yes, it means we found a node that violates the BST properties, and we return `False`.
   - While the `stack` is not empty and the current value exceeds the last element in the `stack`, we pop the `stack` to backtrack and update `low` to the last popped value. This signifies that we are moving to the right subtree where values must be greater.
   - Finally, we push the current value onto `stack`.

3. **Return Result**: If we iterate through all values without violations, we return `True`.

### Complexity:
- **Time Complexity**: O(N), where N is the number of elements in the `preorder` array, since we are iterating through the list once.
- **Space Complexity**: O(1) auxiliary space (ignoring the input size) since we are only using a stack for tracking the current nodes.

This approach effectively checks if the `preorder` array corresponds to a valid BST's preorder traversal, adhering to the constraints and recommendation for constant space use.

# 257. Binary Tree Paths

### Problem Description 
Given the `root` of a binary tree, return all root-to-leaf paths in any order.

A leaf is a node with no children.


Example 1:
Input: root = [1,2,3,null,5]
Output: ["1->2->5","1->3"]

Example 2:
Input: root = [1]
Output: ["1"]

Constraints:
The number of nodes in the tree is in the range `[1, 100]`.

`-100 <= Node.val <= 100`

### Solution 
 To solve the problem of finding all root-to-leaf paths in a binary tree, we can use a depth-first search (DFS) approach. The idea is to traverse the tree from the root to the leaves, keeping track of the current path as we go deeper into the tree.

### Steps to Solve:
1. **Depth-First Search (DFS)**: We will use a standard DFS approach where we explore each branch of the tree recursively.
2. **Track the Current Path**: As we traverse the tree, we will maintain the current path as a string format. When we reach a leaf node (a node with no left and right children), we will add that path to our results list.
3. **Backtrack**: After exploring one node, we will backtrack to explore another node, hence we will need to manage the current path accordingly.

Here's how we can implement this in Python:

### Python Code:


```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def binaryTreePaths(self, root: TreeNode) -> list[str]:
        if not root:
            return []

        paths = []  # This will hold all the root-to-leaf paths
        
        def dfs(node, current_path):
            # Add current node's value to the path
            current_path += str(node.val)
            
            # Check if it's a leaf node
            if not node.left and not node.right:
                paths.append(current_path)  # We reached a leaf, add the path to the results
                return
            
            # If not a leaf, continue DFS on children
            if node.left:
                dfs(node.left, current_path + "->")
            if node.right:
                dfs(node.right, current_path + "->")
        
        # Start the DFS from the root
        dfs(root, "")
        
        return paths

```

### Explanation:
1. **TreeNode Class**: This is a standard definition of a binary tree node, which includes a value (`val`) and pointers to the left and right children.
2. **Solution Class**: This class contains the method `binaryTreePaths`, which will return all root-to-leaf paths in the given binary tree.
3. **DFS Function**: Inside the `binaryTreePaths` method, we define a nested helper function `dfs` which takes a node and the current path as parameters.
   - We concatenate the current node’s value to `current_path`.
   - If the node is a leaf (both left and right children are None), we add the current path to the list of paths.
   - If not, we recursively call `dfs` on the left and right children (if they exist), passing the updated path (appending "->" before adding the next node's value).
4. **Initial Call**: The DFS is initiated with the root node and an empty string for the current path.
5. **Return Result**: Finally, we return the collected paths.

### Complexity:
- **Time Complexity**: O(N) where N is the number of nodes in the tree, since we traverse each node exactly once.
- **Space Complexity**: O(H) where H is the height of the tree due to the recursion call stack.

This code can be directly run on the LeetCode platform, and it adheres to their specific format for solving binary tree-related problems. Adjustments for input reading and providing the necessary tree structure will be handled by the LeetCode environment itself.

# 270. Closest Binary Search Tree Value

### Problem Description 
Given the `root` of a binary search tree and a `target` value, return the value in the BST that is closest to the `target`.


Example 1:
Input: root = [4,2,5,1,3], target = 3.714286
Output: 4

Example 2:
Input: root = [1], target = 4.428571
Output: 1

Constraints:
The number of nodes in the tree is in the range `[1, 104]`.

`0 <= Node.val <= 109`
`-109 <= target <= 109`

### Solution 
 To solve the problem of finding the value in a binary search tree (BST) that is closest to a given target value, we will implement a function that traverses the BST. Because the tree is a binary search tree, we can leverage the properties of the tree to make our search more efficient by pruning unnecessary branches based on comparisons with the target.

Here’s how we can approach this:

1. **Initialization**: Start with a variable to keep track of the closest value found so far. Initialize it to a very large value or the root value itself.
  
2. **Traversal**: We'll use a while loop to traverse the tree. At each node:
   - Compare the node's value to the target.
   - If the current node's value is closer to the target than our current closest value, update the closest value.
   - Depending on whether the current node’s value is greater than or less than the target, decide which subtree to explore next. If it's less than the target, move to the right child; otherwise, move to the left child.

3. **Return the result**: After we've finished traversing the tree, return the closest value found.

Here’s the implementation of the above logic in Python, following the LeetCode solution format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def findClosestValueInBST(self, root: TreeNode, target: float) -> int:
        # Initial closest value set to the root value
        closest = root.val
        
        # Start traversing the tree
        current_node = root
        
        while current_node:
            # If the current node's value is closer to the target than the closest value, update closest
            if abs(current_node.val - target) < abs(closest - target):
                closest = current_node.val
            
            # Decide to go left or right in the BST
            if target < current_node.val:
                current_node = current_node.left
            else:
                current_node = current_node.right
        
        return closest

```

### Explanation of the Code:

- **TreeNode Class**: This defines the structure of a node in the BST. Each node has a value `val`, a left child `left`, and a right child `right`.

- **findClosestValueInBST Function**:
  - `closest` is initialized to the value of the root. This will hold the closest value found during the traversal.
  - `current_node` is initialized to the root, and we begin traversing the tree.
  - A while loop runs as long as `current_node` is not `None`. Inside the loop:
    - We calculate the absolute difference between the `current_node.val` and the `target` and compare it with the difference between `closest` and `target`. If the current node is closer, we update `closest`.
    - We then decide whether to move to the left or right child based on the value of `current_node` compared to `target`.
- Once the loop ends (when we reach a leaf node), we return the closest value.

### Complexity Analysis:

- **Time Complexity**: O(h), where h is the height of the tree. In the worst case, h can be equal to n (the number of nodes) if the tree is unbalanced; however, in a balanced BST, h would be log(n).
- **Space Complexity**: O(1), because we are using a constant amount of space for variables. 

This solution should run efficiently within the given constraints and provides the closest value in the BST relative to the target.

# 272. Closest Binary Search Tree Value II

### Problem Description 
Given the `root` of a binary search tree, a `target` value, and an integer `k`, return the `k` values in the BST that are closest to the `target`. You may return the answer in any order.

You are guaranteed to have only one unique set of `k` values in the BST that are closest to the `target`.


Example 1:
Input: root = [4,2,5,1,3], target = 3.714286, k = 2
Output: [4,3]

Example 2:
Input: root = [1], target = 0.000000, k = 1
Output: [1]

Constraints:
The number of nodes in the tree is `n`.

`1 <= k <= n <= 104`.

`0 <= Node.val <= 109`
`-109 <= target <= 109`
Follow up: Assume that the BST is balanced. Could you solve it in less than `O(n)` runtime (where `n = total nodes`)?

### Solution 
 To solve the problem of finding the `k` values in a binary search tree (BST) that are closest to a given `target` value, we can employ a strategy utilizing the properties of a BST and a max-heap to efficiently maintain the `k` closest values as we traverse the tree.

### Approach:

1. **In-Order Traversal**: Since the elements of a BST are stored in a sorted order, we can perform an in-order traversal which will yield the nodes in sorted order (left-root-right).

2. **Max-Heap**: We can use a max-heap (which in Python can be implemented using the `heapq` library by storing negative distances) to keep track of the closest `k` values. As we traverse, we push the distance from the target to each node's value into the heap. If the size of the heap exceeds `k`, we pop the largest element, ensuring that only the closest `k` elements remain.

3. **Collecting Results**: Once the traversal is complete, the heap will contain the `k` closest values which we can extract and return.

### Implementation:

Here’s the code implementation in Python:



```python
import heapq

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def closestKValues(self, root: TreeNode, target: float, k: int) -> list[int]:
        # This is the max-heap to keep track of the closest k values
        max_heap = []

        def inorder(node):
            if not node:
                return
            
            # Inorder traversal: left -> node -> right
            inorder(node.left)
            
            # Calculate the absolute difference to target
            dist = abs(node.val - target)
            # Push the (distance, value) pair to max-heap
            heapq.heappush(max_heap, (-dist, node.val))
            
            # If the size of the heap is greater than k, remove the farthest value
            if len(max_heap) > k:
                heapq.heappop(max_heap)
            
            inorder(node.right)
        
        inorder(root)
        
        # Extract the values from the max-heap
        result = []
        while max_heap:
            result.append(heapq.heappop(max_heap)[1])
        
        return result

# Example usage:
# root = TreeNode(4, TreeNode(2, TreeNode(1), TreeNode(3)), TreeNode(5))
# target = 3.714286
# k = 2
# sol = Solution()
# print(sol.closestKValues(root, target, k))  # Output could be [4, 3]

```

### Explanation of the Code:

- **TreeNode Class**: This defines the structure of each node in the binary tree.
- **Solution Class**: Contains the method `closestKValues`:
  - `max_heap` is initialized to track the k closest values.
  - The inner function `inorder(node)` is called recursively to perform an in-order traversal.
  - For each node, we calculate the absolute distance from the `target` and push it into the heap along with its value (negated distance for max-heap behavior).
  - If the size of the heap exceeds `k`, we pop the largest distance (max value) so only the closest remain.
- Finally, we pop all elements from the max-heap to gather the results and return them.

### Complexity:
- **Time Complexity**: The traversal takes `O(n)` time, but since we only maintain `k` elements in the heap, the heap operations add an additional `O(k log k)` in the worst case. However, typically for binary trees, `k << n`, leading to practical performance improvements.
- **Space Complexity**: The space used by the heap is `O(k)`.

The above code meets the constraints and specifications given in the problem statement and should work directly on LeetCode.

# 285. Inorder Successor in BST

### Problem Description 
Given the `root` of a binary search tree and a node `p` in it, return the in-order successor of that node in the BST. If the given node has no in-order successor in the tree, return `null`.

The successor of a node `p` is the node with the smallest key greater than `p.val`.


Example 1:
Input: root = [2,1,3], p = 1
Output: 2
Explanation: 1's in-order successor node is 2. Note that both p and the return value is of TreeNode type.


Example 2:
Input: root = [5,3,6,2,4,null,null,1], p = 6
Output: null
Explanation: There is no in-order successor of the current node, so the answer is `null`.


Constraints:
The number of nodes in the tree is in the range `[1, 104]`.

`-105 <= Node.val <= 105`
All Nodes will have unique values.

### Solution 
 To solve the problem of finding the in-order successor of a given node `p` in a binary search tree (BST), we can use the properties of the BST and the relationship between the nodes. An in-order traversal of a BST provides the nodes' values in sorted order, which is key to finding the successor.

### Explanation:
1. **Definition of In-Order Successor**: The in-order successor of a node `p` is defined as the node with the smallest key that is greater than `p.val`. 

2. **Case Analysis**:
   - If `p` has a right subtree, the successor is the minimum value node in that right subtree. We traverse to the right child of `p` and then keep going left until we reach a leaf node.
   - If `p` does not have a right subtree, we need to find the closest ancestor of `p` for which `p` would be in the left subtree. This involves moving up the tree until we find a node that is the left child of its parent. The parent of this node will be the successor.

### Implementation Steps:
1. Start at the root of the BST.
2. If `p` has a right child, traverse to the right child and then continue to the left until we reach the leftmost node (the minimum in that subtree).
3. If `p` does not have a right child, traverse from the root to find `p`, and while doing so, track the potential successor nodes.
4. When you find the node `p`, check the parent-child relationships to determine the successor.

### Python Code:
Here's the Python code that implements the above logic. This code follows the standard function signature and format used in LeetCode.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> TreeNode:
        # Initial assumption: successor is None
        successor = None
        
        # Start from the root and traverse the tree
        while root:
            # If the value of p is less than the current node's value
            if p.val < root.val:
                # Update successor to the current node
                successor = root
                # Move to the left subtree to find possibly smaller successor
                root = root.left
            else:
                # Move to right subtree since p's value is greater than or equal to root's
                root = root.right
        
        return successor

```

### Explanation of the Code:
- **TreeNode Class**: This defines the structure of each node in the tree (value, left child, right child).
- **Solution Class**: Contains the method to find the in-order successor.
  - `inorderSuccessor` method takes the root of the tree and the target node `p`.
  - It initializes `successor` as `None`.
  - It uses a while loop to navigate the BST:
    - If the current node's value is greater than `p`'s value, it updates `successor` and moves left to find a potentially smaller successor.
    - If the current node's value is less than or equal to `p`'s value, it moves right.
- Finally, it returns the found successor or `None` if no successor exists.

### Complexity:
- **Time Complexity**: O(h), where h is the height of the tree. In a balanced tree, this would be O(log n), while in a degenerate tree (like a linked list), it can lead to O(n).
- **Space Complexity**: O(1) since we are using only a constant amount of additional storage. 

This solution is efficient and works well within the constraints provided by the problem.


# 297. Serialize and Deserialize Binary Tree

### Problem Description 
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Clarification: The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.


Example 1:
Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]

Example 2:
Input: root = []
Output: []

Example 3:
Input: root = [1]
Output: [1]

Example 4:
Input: root = [1,2]
Output: [1,2]

Constraints:
The number of nodes in the tree is in the range `[0, 104]`.

`-1000 <= Node.val <= 1000`

### Solution 
 To solve the problem of serializing and deserializing a binary tree, we will implement two main functions: `serialize` and `deserialize`. 

1. **Serialization**: This process will convert the binary tree into a string format. We can use a pre-order traversal method (root, left, right) to traverse the tree and convert it into a string representation. Here, we will represent `null` values for missing children.

2. **Deserialization**: This will convert the serialized string back into the original binary tree structure. We will split the string into individual elements and reconstruct the tree using a queue to efficiently provide the next value while constructing the nodes.

Here is the detailed implementation using the LeetCode format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Codec:
    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string."""
        def preorder(node):
            if not node:
                return "null,"
            return str(node.val) + "," + preorder(node.left) + preorder(node.right)

        return preorder(root)

    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree."""
        def build_tree(nodes):
            val = next(nodes)
            if val == "null":
                return None
            node = TreeNode(int(val))
            node.left = build_tree(nodes)
            node.right = build_tree(nodes)
            return node
        
        nodes = iter(data.split(','))
        return build_tree(nodes)


# Example usage:
# codec = Codec()
# tree = codec.deserialize("[1,2,3,null,null,4,5]")
# serialized = codec.serialize(tree)
# print(serialized)  # Output should match the input string

```

### Explanation:

1. **TreeNode Class**: 
   - The `TreeNode` class represents each node in the binary tree, having a value (`val`), a left child (`left`), and a right child (`right`). 

2. **Codec Class**: 
   - `Codec` contains two methods: `serialize` and `deserialize`.

3. **Serialization Process** (`serialize` method):
   - We define a nested function `preorder` which takes a node and performs a pre-order traversal.
   - If a node is `None`, we return the string "null," to indicate a null node.
   - If the node is not null, we convert its value to a string and concatenate it with:
     - The result of the left subtree serialized.
     - The result of the right subtree serialized.
   - Finally, the result is returned as a single string.

4. **Deserialization Process** (`deserialize` method):
   - We split the serialized string into a list of node values using `split(',')`.
   - We create an iterator from this list called `nodes`.
   - We define a nested function `build_tree` that builds the tree recursively:
     - It retrieves the next value from the `nodes` iterator.
     - If it's "null", it returns `None`.
     - Otherwise, it creates a new `TreeNode` with the integer value and recursively builds the left and right subtrees.
   - The root of the tree is returned after constructing it completely.

### Complexity Analysis:
- **Time Complexity**: 
  - Both serialization and deserialization operations traverse all nodes exactly once, resulting in a time complexity of O(n), where n is the number of nodes in the tree.
  
- **Space Complexity**: 
  - The space complexity is O(n) due to the space needed to store the serialized string in memory, and also the recursive call stack space during deserialization in the worst case (for a skewed tree).

This implementation meets the requirements of the problem statement and should perform efficiently within the given constraints.

# 298. Binary Tree Longest Consecutive Sequence

### Problem Description 
Given the `root` of a binary tree, return the length of the longest consecutive sequence path.

The path refers to any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The longest consecutive path needs to be from parent to child (cannot be the reverse).


Example 1:
Input: root = [1,null,3,2,4,null,null,null,5]
Output: 3
Explanation: Longest consecutive sequence path is 3-4-5, so return 3.


Example 2:
Input: root = [2,null,3,2,null,1]
Output: 2
Explanation: Longest consecutive sequence path is 2-3, not 3-2-1, so return 2.


Constraints:
The number of nodes in the tree is in the range `[1, 3 * 104]`.

`-3 * 104 <= Node.val <= 3 * 104`

### Solution 
 To solve the problem of finding the length of the longest consecutive sequence path in a binary tree, we can use a depth-first search (DFS) approach. The idea is to traverse the tree while keeping track of the current streak of consecutive values.

Here's a detailed step-by-step explanation of how we will approach the solution:

1. **DFS Traversal**: We will use a helper function that performs a DFS traversal on the binary tree. This function will take two parameters: the current node and the current length of the consecutive sequence.

2. **Checking Consecutiveness**: For each node, we will check if its value is consecutive with respect to its parent node's value. Specifically, we will check if the current node's value is exactly one greater than the parent node's value.

3. **Updating the Current Length**: If the value is consecutive, we increase the current length. If it's not consecutive, we reset the current length to 1 (since we can start a new sequence from this node).

4. **Tracking the Maximum Length**: We maintain a variable to keep track of the maximum length of consecutive paths encountered during the traversal.

5. **Base Case**: If we reach a null node, we return to the previous level since there's nothing to process.

### Implementation on LeetCode

Now, let's implement this logic in Python as per the LeetCode solution format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def longestConsecutive(self, root: TreeNode) -> int:
        # Initialize the maximum length variable
        self.max_length = 0
        
        # Helper function to perform DFS
        def dfs(node, parent_val, current_length):
            if not node:
                return
            
            # Check if the current node is consecutive to the parent
            if parent_val is not None and node.val == parent_val + 1:
                current_length += 1
            else:
                current_length = 1  # Reset the length
            
            # Update the maximum length if current is longer
            self.max_length = max(self.max_length, current_length)

            # Continue DFS for left and right children
            dfs(node.left, node.val, current_length)
            dfs(node.right, node.val, current_length)
        
        # Start DFS from the root
        dfs(root, None, 0)
        
        return self.max_length

# Example usage:
# root = TreeNode(1)
# root.right = TreeNode(3)
# root.right.left = TreeNode(2)
# root.right.right = TreeNode(4)
# root.right.right.right = TreeNode(5)
# sol = Solution()
# print(sol.longestConsecutive(root)) # Output: 3

```

### Explanation of the Code:

- **TreeNode Class**: This is a basic definition of a binary tree node with `val`, `left`, and `right`.

- **Solution Class**: We define the `Solution` class with a method `longestConsecutive` which initializes a variable to keep track of the maximum path length.

- **DFS Function**: Inside the `longestConsecutive` method, we define the `dfs` function that traverses the tree. It checks each node for consecutiveness with its parent, updates the current length accordingly, and checks if we have a new maximum length.

- **Base Case**: If the current node is None, the function simply returns.

- **Initiating DFS**: We start the DFS at the root node, initially passing `None` for the parent value and 0 for the current length since there are no nodes processed yet.

- **Return Value**: Finally, we return the maximum length of consecutive paths that we computed during the traversal. 

This solution runs in O(N) time where N is the number of nodes in the tree, which is efficient given the problem constraints.

# 333. Largest BST Subtree

### Problem Description 
Given the root of a binary tree, find the largest subtree, which is also a Binary Search Tree (BST), where the largest means subtree has the largest number of nodes.

A Binary Search Tree (BST) is a tree in which all the nodes follow the below-mentioned properties:
The left subtree values are less than the value of their parent (root) node's value.

The right subtree values are greater than the value of their parent (root) node's value.

Note: A subtree must include all of its descendants.

Follow up: Can you figure out ways to solve it with O(n) time complexity?

Example 1:
Input: root = [10,5,15,1,8,null,7]
Output: 3
Explanation: The Largest BST Subtree in this case is the highlighted one. The return value is the subtree's size, which is 3.


Example 2:
Input: root = [4,2,7,2,3,5,null,2,null,null,null,null,null,1]
Output: 2

Constraints:
The number of nodes in the tree is in the range `[0, 104]`.

`-104 <= Node.val <= 104`

### Solution 
 To solve the problem of finding the largest BST subtree in a binary tree, we can leverage a depth-first search (DFS) approach. The idea is to traverse the tree while keeping track of properties that define a valid BST and the size of the valid BSTs that we encounter.

### Key Concepts:
1. **Binary Search Tree Properties**: For a subtree to qualify as a BST:
   - All nodes in the left subtree must be less than the root node's value.
   - All nodes in the right subtree must be greater than the root node's value.

2. **DFS Traversal**: By conducting a post-order traversal, we can evaluate the left and right subtrees first before checking the current subtree.

3. **Tracking Information**: For each subtree rooted at a node, we will keep track of:
   - Whether it is a valid BST.
   - The minimum and maximum values within that subtree.
   - The size of the subtree.

### Approach:
1. Define a helper function that:
   - Returns a tuple with four values: (is_bst, size_of_bst, min_value, max_value).
   - If a subtree is a valid BST, update the size of the largest BST found so far.
 
2. During the DFS traversal:
   - If a node's left and right subtrees are BSTs and satisfy the BST property with respect to the current node, then the current subtree is also a BST.
   - Update the minimum and maximum values from the current node and its children.

3. Maintain a global variable to track the maximum size of any valid BST subtree found during the traversal.

### Implementation:

Here's how we can implement this in Python:



```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def largestBSTSubtree(self, root: TreeNode) -> int:
        self.max_size = 0
        
        def dfs(node):
            if not node:
                return (True, 0, float('inf'), float('-inf'))
            
            left_is_bst, left_size, left_min, left_max = dfs(node.left)
            right_is_bst, right_size, right_min, right_max = dfs(node.right)

            # Check if the current subtree is a BST
            if left_is_bst and right_is_bst and left_max < node.val < right_min:
                # Calculate the size of the current BST
                current_size = left_size + right_size + 1
                self.max_size = max(self.max_size, current_size)
                # Return True indicating it's a valid BST
                return (True, current_size, min(left_min, node.val), max(right_max, node.val))
            else:
                # Not a BST, return False and size of the largest BST found so far
                return (False, 0, 0, 0)
        
        dfs(root)
        return self.max_size

```

### Explanation of the Code:
- The `TreeNode` class represents a node in the binary tree.
- The `largestBSTSubtree` method initializes a variable to keep track of the largest BST found and calls the helper function `dfs`.
- The `dfs` function:
  - Handles the base case of a null node by returning a tuple indicating it is a BST with size 0.
  - Recursively examines the left and right children and checks if the current node can form a BST with its children.
  - Updates the maximum size of any valid BST found so far.
- Finally, we return `self.max_size`, which holds the size of the largest BST subtree found.

### Complexity:
The solution runs in O(n) time complexity since we visit each node exactly once. The space complexity is O(h) for the call stack in the worst case (where h is the height of the tree). This should efficiently handle trees with up to 10,000 nodes as specified in the constraint.

# 337. House Robber III

### Problem Description 
The thief has found himself a new place for his thievery again. There is only one entrance to this area, called `root`.

Besides the `root`, each house has one and only one parent house. After a tour, the smart thief realized that all houses in this place form a binary tree. It will automatically contact the police if two directly-linked houses were broken into on the same night.

Given the `root` of the binary tree, return the maximum amount of money the thief can rob without alerting the police.


Example 1:
Input: root = [3,2,3,null,3,null,1]
Output: 7
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.


Example 2:
Input: root = [3,4,5,1,3,null,1]
Output: 9
Explanation: Maximum amount of money the thief can rob = 4 + 5 = 9.


Constraints:
The number of nodes in the tree is in the range `[1, 104]`.

`0 <= Node.val <= 104`

### Solution 
 To solve the problem of maximizing the amount of money the thief can rob from a binary tree without triggering alarms by robbing two directly connected houses, we can use a depth-first search (DFS) approach with dynamic programming.

### Explanation:

1. **Understanding the tree structure**: Each node can either be robbed or not. If a node is robbed, its direct children (parent nodes) cannot be robbed. Conversely, if a node is not robbed, then the thief can decide to rob its children.

2. **Dynamic programming with memoization**: We will define a recursive function that returns two values for each node:
   - The maximum money that can be robbed if the current node is robbed.
   - The maximum money that can be robbed if the current node is not robbed.

3. **DFS Traversal**: We will traverse the tree using DFS, and for each node, we will calculate the maximum money obtainable both if we rob it and if we don’t.

4. **Base case**: If the current node is `null`, we can return zero for both scenarios (robbed and not robbed).

5. **Calculation at each node**:
   - If we rob the current node, we add its value to the maximum money obtainable from its grandchildren (i.e., `left.left + left.right` and `right.left + right.right`).
   - If we don’t rob the current node, we can simply take the maximum money obtainable from its left and right children.

6. **Final Result**: The maximum of the two values (robbed and not robbed) gives the solution for the binary tree rooted at the given node.

### Implementation:

Here is the Python code that follows the above explanation and adheres to the LeetCode solution format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def rob(self, root: TreeNode) -> int:
        def dfs(node):
            if not node:
                return (0, 0)
            
            left = dfs(node.left)
            right = dfs(node.right)
            
            # If we rob this node
            rob_this = node.val + left[1] + right[1]
            # If we do not rob this node
            not_rob_this = max(left) + max(right)
            
            return (rob_this, not_rob_this)
        
        return max(dfs(root))


```

### Explanation of the Code:

- **TreeNode Class**: We define a `TreeNode` class which represents nodes of the binary tree.
- **Solution Class**: The `Solution` class contains the method `rob` which takes the tree's root as input.
- **dfs function**: This is a helper function that recursively calculates the maximum money that can be robbed from the tree nodes:
  - For each node, it checks the left and right children via recursive calls to `dfs`.
  - It computes two scenarios: robbing the current node and not robbing it.
- **Return statement**: Finally, we return the greater of the two scenarios from the root node, which gives the maximum money the thief can rob throughout the entire tree.

### Complexity Analysis:

- **Time Complexity**: O(N), where N is the number of nodes in the tree, since we visit each node exactly once.
- **Space Complexity**: O(H), where H is the height of the tree, for the recursion stack.

This solution is efficient and robust for the given constraints.

# 366. Find Leaves of Binary Tree

### Problem Description 
Given the `root` of a binary tree, collect a tree's nodes as if you were doing this:
Collect all the leaf nodes.

Remove all the leaf nodes.

Repeat until the tree is empty.


Example 1:
Input: root = [1,2,3,4,5]
Output: [[4,5,3],[2],[1]]
Explanation:
[[3,5,4],[2],[1]] and [[3,4,5],[2],[1]] are also considered correct answers since per each level it does not matter the order on which elements are returned.


Example 2:
Input: root = [1]
Output: [[1]]

Constraints:
The number of nodes in the tree is in the range `[1, 100]`.

`1 <= Node.val <= 100`

### Solution 
 To solve the problem of collecting leaf nodes from a binary tree and then removing them iteratively until the tree is empty, we can use a depth-first search (DFS) approach. Below, I will outline the solution in a format suitable for LeetCode submissions, along with detailed explanations of the key parts of the code.

### Solution Explanation

1. **Understanding the Problem**: We need to traverse the binary tree and collect all the leaf nodes. Once collected, we remove these nodes and continue the process until all nodes have been processed. The result should be a list of lists where each inner list represents the leaf nodes collected at each iteration.

2. **Tree Structure**: We will use the `TreeNode` class as given in typical binary tree problems. Each node has a value and can have a left and a right child.

3. **DFS Traversal**: We will perform a depth-first traversal to find the leaf nodes. Once we find them, we will store them in a list and then construct the new tree without these leaves.

4. **Loop Until Empty**: We will repeat the process until there are no more nodes left in the tree, which means our tree is empty.

### Python Code

Here’s the complete Python solution:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def findLeaves(self, root: TreeNode) -> List[List[int]]:
        res = []
        
        def dfs(node):
            if not node:
                return -1  # Base case: null node returns -1

            level = 1 + max(dfs(node.left), dfs(node.right))  # Get the level from left and right subtrees
            if level == len(res):
                res.append([])  # Create a new level if it doesn't exist
            res[level].append(node.val)  # Append the node value to the corresponding level

            return level  # Return the current level

        dfs(root)  # Start the DFS traversal
        return res  # Return the collected leaves in the correct format

```

### Key Parts of the Code Explanation

1. **TreeNode Class**: This is a simple class to represent nodes in the binary tree.

2. **DFS Function**: The `dfs` function is a recursive helper function that:
   - Takes a `node` as an argument (initially the root of the tree).
   - Returns the depth level of the current node based on the maximum level of its children.
   - If a node is null, it returns `-1` to signify no nodes were found.
   - It calculates the current level by taking 1 plus the maximum depth from the left and right children.
   - If the level is equal to the length of the result list `res`, it means we need to create a new list for that level.
   - We then append the current node's value to the appropriate level in `res`.

3. **Result Construction**: After the DFS completes, we have a structured result with leaf nodes collected by their depth, which represents the leaves removed in each iteration until the tree is empty.

4. **Return Value**: Finally, the `res` list is returned as the output, formatted correctly as required by the problem statement.

### Example Usage
Assuming an instance of `TreeNode` is created representing the input tree, this solution can be directly run on LeetCode to receive the expected output based on the tree structure provided.

# 404. Sum of Left Leaves

### Problem Description 
Find the sum of all left leaves in a given binary tree.


Example:
    3
   / \
  9  20
    /  \
   15   7
There are two left leaves in the binary tree, with values 9 and 15 respectively. Return 24.

### Solution 
 To solve the problem of finding the sum of all left leaves in a given binary tree, we can traverse the tree while checking whether each node is a left leaf. A left leaf is defined as a node that is a left child of its parent and has no children (both left and right children are `None`).

Here’s how we can implement the solution:

1. **Define the TreeNode class**: This class will represent each node in the binary tree.
2. **Create the sum_of_left_leaves function**: This function will carry out the traversal of the binary tree and calculate the sum of the left leaves. We will utilize a helper function to perform a depth-first traversal of the tree.
3. **Perform DFS**: During the traversal, we will check if a node is a left leaf and if so, add its value to the sum.
4. **Return the sum**: After completing the traversal, we will return the computed sum of all left leaves.

Here's the implementation in Python:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        # Helper function to perform DFS
        def dfs(node: TreeNode, is_left: bool) -> int:
            # Base case: if the node is None, return 0
            if not node:
                return 0
            
            # Check if the node is a left leaf
            if is_left and not node.left and not node.right:
                return node.val
            
            # Continue DFS for left and right children
            left_sum = dfs(node.left, True)   # Mark left child as left
            right_sum = dfs(node.right, False) # Mark right child as not left
            
            return left_sum + right_sum
        
        # Start DFS traversal from the root
        return dfs(root, False)

```

### Explanation of the Code:

1. **TreeNode Class**: This class defines nodes in the binary tree. Each node has a value `val`, and references to its left and right children.

2. **sumOfLeftLeaves Method**:
   - This is the main method that will be called to get the sum of left leaves.
   - The method initializes the DFS traversal starting from the root, marking it as not a left child.
   
3. **dfs Function**:
   - This is a nested helper function that takes a `node` and a boolean `is_left` indicating if the node is a left child.
   - If the current node is `None`, it returns 0 (base case for recursion).
   - If the current node is a left child and a leaf (both children are `None`), it returns the value of that node (since it’s a left leaf).
   - The function recursively computes the sum from the left and right children by calling itself with the appropriate flags for left and right.
   - Finally, it returns the total sum by adding the results from both left and right children.

### Complexity Analysis:
- **Time Complexity**: O(N), where N is the number of nodes in the tree. We visit each node once.
- **Space Complexity**: O(H), where H is the height of the tree. This accounts for the recursion stack.

This code can be directly run on LeetCode as it follows their specified format for defining a problem. You can replace the `root` variable with the input binary tree structure during your test cases.

# 426. Convert Binary Search Tree to Sorted Doubly Linked List

### Problem Description 
Convert a Binary Search Tree to a sorted Circular Doubly-Linked List in place.

You can think of the left and right pointers as synonymous to the predecessor and successor pointers in a doubly-linked list. For a circular doubly linked list, the predecessor of the first element is the last element, and the successor of the last element is the first element.

We want to do the transformation in place. After the transformation, the left pointer of the tree node should point to its predecessor, and the right pointer should point to its successor. You should return the pointer to the smallest element of the linked list.


Example 1:
Input: root = [4,2,5,1,3]
Output: [1,2,3,4,5]
Explanation: The figure below shows the transformed BST. The solid line indicates the successor relationship, while the dashed line means the predecessor relationship.


Example 2:
Input: root = [2,1,3]
Output: [1,2,3]

Example 3:
Input: root = []
Output: []
Explanation: Input is an empty tree. Output is also an empty Linked List.

Example 4:
Input: root = [1]
Output: [1]

Constraints:
The number of nodes in the tree is in the range `[0, 2000]`.

`-1000 <= Node.val <= 1000`
All the values of the tree are unique.

### Solution 
 To solve the problem of converting a Binary Search Tree (BST) into a sorted circular doubly-linked list in place, we can utilize an in-order traversal of the tree. During this traversal, we can rearrange the pointers of the nodes to create the required doubly-linked list structure. 

Here's the step-by-step approach we will follow to implement the solution:

1. **In-Order Traversal**: Perform an in-order traversal of the BST. This traversal visits the nodes in ascending order, which is perfect for creating a sorted list.
  
2. **Linking Nodes**: While traversing, we will adjust the `left` and `right` pointers of each node:
   - The `left` pointer of a node will point to its predecessor (the previous node in in-order traversal).
   - The `right` pointer of a node will point to its successor (the next node in in-order traversal).
  
3. **Circular Linking**: To make it circular, after processing all nodes, we need to link the last processed node's right pointer to the first processed node and the first processed node's left pointer to the last processed node.

4. **Return the Head**: Finally, we will return the pointer to the smallest element of the list, which is the first node processed during the traversal.

Here's the Python code for the solution:



```python
class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def treeToDoublyList(self, root: Node) -> Node:
        if not root:
            return None
        
        # Pointers for the head and the previous processed node in in-order traversal
        self.head = None
        self.prev = None
        
        def in_order(node):
            if not node:
                return
            
            # Traverse the left subtree
            in_order(node.left)
            
            # Process the current node
            if self.prev:
                # Link the previous node (predecessor) to current node
                self.prev.right = node
                node.left = self.prev
            else:
                # If prev is None, it means we are at the leftmost node
                self.head = node
            
            # Update prev to current node before moving to the right subtree
            self.prev = node
            
            # Traverse the right subtree
            in_order(node.right)
        
        # Perform the in-order traversal to link the nodes
        in_order(root)
        
        # Make the list circular
        self.head.left = self.prev
        self.prev.right = self.head
        
        return self.head

```

### Explanation of the Code:
1. **Node Class**: We define a `Node` class that represents each node in the BST with attributes for `val`, `left`, and `right`.

2. **Solution Class**: 
   - The main logic is implemented in the `Solution` class under the `treeToDoublyList` method. 
   - We check if the `root` is `None`. If it is, we return `None` since there's no tree to convert.

3. **In-Order Traversal Function**:
   - We define an inner function `in_order` that takes a node as an argument.
   - This function is recursively called to traverse the left subtree, process the current node, and then traverse the right subtree.
   - During processing, we check if `self.prev` is not `None` to link the current node to its predecessor. If it is `None`, we assign the current node to `self.head`, which becomes the head of our doubly-linked list.

4. **Circular Linking**: After completing the traversal, we link the head and the last processed node to make the list circular.

5. **Return the Head**: Finally, we return `self.head`, which points to the smallest element in the linked list.

### Complexity:
- **Time complexity**: O(n), where n is the number of nodes in the tree.
- **Space complexity**: O(n) in the implicit recursion stack.

This solution works efficiently for the given constraints, and utilizing in-order traversal ensures that we create a correctly ordered doubly-linked list.

# 428. Serialize and Deserialize N-ary Tree

### Problem Description 
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize an N-ary tree. An N-ary tree is a rooted tree in which each node has no more than N children. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that an N-ary tree can be serialized to a string and this string can be deserialized to the original tree structure.

For example, you may serialize the following `3-ary` tree
as `[1 [3[5 6] 2 4]]`. Note that this is just an example, you do not necessarily need to follow this format.

Or you can follow LeetCode's level order traversal serialization format, where each group of children is separated by the null value.

For example, the above tree may be serialized as `[1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]`.

You do not necessarily need to follow the above suggested formats, there are many more different formats that work so please be creative and come up with different approaches yourself.


Constraints:
The number of nodes in the tree is in the range `[0, 104]`.

`0 <= Node.val <= 104`
The height of the n-ary tree is less than or equal to `1000`
Do not use class member/global/static variables to store states. Your encode and decode algorithms should be stateless.

### Solution 
 To solve the problem of serializing and deserializing an N-ary tree, we can follow a structured approach. We will define a simple N-ary tree node class, implement the serialization process to convert the tree into a string format, and implement the deserialization process to restore the original tree structure from that string.

### N-ary Tree Node Definition
First, we need to define the structure of our N-ary tree node. Each node will have a value and a list of its children.

### Serialization Process
For serialization, we can use a depth-first traversal of the tree:
- First, write the value of the node.
- Then, write the number of children and serialize each child recursively.

### Deserialization Process
For deserialization, we can parse the string:
- Read the node value.
- Read the number of children and recursively construct each child node.

### Python Code
Here is the complete Python code that follows the LeetCode solution format:



```python
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []

class Codec:
    def serialize(self, root: 'Node') -> str:
        """Encodes an N-ary tree to a single string."""
        if not root:
            return ""
        
        # Initialize the serialization string
        result = []
        
        # Helper function to perform DFS for serialization
        def dfs_encode(node):
            result.append(str(node.val))  # Serialize the node value
            result.append(str(len(node.children)))  # Serialize the count of children
            for child in node.children:  # Serialize each child
                dfs_encode(child)
        
        dfs_encode(root)
        return ",".join(result)  # Join the result with commas

    def deserialize(self, data: str) -> 'Node':
        """Decodes your encoded data to tree."""
        if not data:
            return None
        
        # Split the serialized string into elements
        values = data.split(",")
        index = 0
        
        # Helper function to perform DFS for deserialization
        def dfs_decode():
            nonlocal index
            if index >= len(values):
                return None
            
            # Read the node value and count of children
            node_val = int(values[index])
            index += 1
            num_children = int(values[index])
            index += 1
            
            # Create a new node
            node = Node(node_val)
            # Deserialize each child
            for _ in range(num_children):
                node.children.append(dfs_decode())
            return node
        
        return dfs_decode()  # Start the deserialization process

# Example usage:
# root = Node(1, [Node(3, [Node(5), Node(6)]), Node(2), Node(4)])
# codec = Codec()
# serialized = codec.serialize(root)
# print(serialized)  # Example output: "1,3,2,5,0,6,0,0,2,0,4,0"
# new_root = codec.deserialize(serialized)

```

### Explanation
- **Node class**: Represents each node of the N-ary tree.
- **Codec class**:
  - **serialize** method: Uses a helper function `dfs_encode` to perform a depth-first traversal of the tree and constructs a string in a comma-separated format. Each node's value is followed by the number of its children.
  - **deserialize** method: Uses a helper function `dfs_decode` to reconstruct the tree from the comma-separated string. It reads the current node's value and number of children, then recursively builds each child's subtree.

### Complexity Analysis
- **Time complexity**: Both serialization and deserialization run in O(N), where N is the number of nodes in the tree, as each node is processed exactly once.
- **Space complexity**: The space used is O(N) for the string representation during serialization and O(H) for function call stack during deserialization, where H is the height of the tree.

This implementation provides a firm foundation for serializing and deserializing N-ary trees while adhering to the given constraints.

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
 To solve the problem of performing a level order traversal on an n-ary tree, we need to represent the n-ary tree structure and implement a breadth-first search (BFS) algorithm to collect the values of nodes at each level.

### Explanation
1. **N-ary Tree Definition**: An n-ary tree is a tree where each node can have zero or more children. We represent the tree using a class `Node` which has a `val` to store the value of the node and a `children` list to store references to its child nodes.

2. **Level Order Traversal**: In a level order traversal, we visit nodes level by level starting from the root. For each level, we collect the values of nodes and store them in a result list.

3. **Using a Queue**: We will use a queue (FIFO structure) to help us traverse the tree level by level. We will start from the root node, enqueue it, then repeatedly dequeue nodes from the front, process its value, and enqueue all its children until the queue is empty.

4. **Implementation Steps**:
   - Initialize an empty result list and a queue.
   - If the root is `None`, return an empty list immediately.
   - Start with the root node in the queue.
   - While the queue is not empty:
     - Count the number of nodes at the current level.
     - Dequeue nodes level by level, collect their values, and enqueue their children.
   - Append the collected values of the current level to the result list.

### Code Implementation

Here's the implementation of the above logic, formatted according to LeetCode requirements:



```python
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []

class Solution:
    def levelOrder(self, root: Node) -> list[list[int]]:
        if root is None:
            return []
        
        result = []
        queue = [root]  # Initialize the queue with the root node

        while queue:
            level_size = len(queue)  # Number of nodes at the current level
            current_level = []
            
            for _ in range(level_size):
                node = queue.pop(0)  # Dequeue the front node
                current_level.append(node.val)  # Process the current node
                for child in node.children:  # Enqueue all children
                    queue.append(child)
            
            result.append(current_level)  # Add the current level to the result
            
        return result

```

### Usage
- To use the given `Solution` class and check the level order traversal implementation, create instances of `Node` to build the n-ary tree as shown in the examples.
- Then create a `Solution` object and call the `levelOrder` method passing the root of the tree. 

### Example


```python
# Example of how to create the tree and use the function

# Creating the tree for input: root = [1,null,3,2,4,null,5,6]
root = Node(1, [
    Node(3, [Node(5), Node(6)]),
    Node(2),
    Node(4)
])

solution = Solution()
output = solution.levelOrder(root)
print(output)  # Output should be [[1], [3, 2, 4], [5, 6]]

```

### Conclusion
The above method ensures we can traverse an n-ary tree in level order efficiently, utilizing BFS with a queue. This approach guarantees that we visit each node exactly once, making the time complexity O(N), where N is the number of nodes in the tree. The space complexity is also O(N) in the worst case due to storing the nodes in the queue.

# 431. Encode N-ary Tree to Binary Tree

### Problem Description 
Design an algorithm to encode an N-ary tree into a binary tree and decode the binary tree to get the original N-ary tree. An N-ary tree is a rooted tree in which each node has no more than N children. Similarly, a binary tree is a rooted tree in which each node has no more than 2 children. There is no restriction on how your encode/decode algorithm should work. You just need to ensure that an N-ary tree can be encoded to a binary tree and this binary tree can be decoded to the original N-nary tree structure.

Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See following example).

For example, you may encode the following `3-ary` tree to a binary tree in this way:
Input: root = [1,null,3,2,4,null,5,6]
Note that the above is just an example which might or might not work. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.


Constraints:
The height of the n-ary tree is less than or equal to `1000`
The total number of nodes is between `[0, 10^4]`
Do not use class member/global/static variables to store states. Your encode and decode algorithms should be stateless.

### Solution 
 To solve the problem of encoding an N-ary tree into a binary tree and then decoding it back into the original N-ary tree, we need to define the structures for both tree types, and then implement encoding and decoding methods.

### Step 1: Definition of Tree Structures

**TreeNode for N-ary Tree:**


```python
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []

```

**TreeNode for Binary Tree:**


```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

```

### Step 2: Encoding Function

The main idea behind encoding is to treat the first child of an N-ary node as the left child of a binary node and the next sibling as the right child of the same binary node.

Here's how we can implement the encoding:


```python
def encode(root: Node) -> TreeNode:
    if not root:
        return None
    
    # Create a new binary node with the value from N-ary node
    binary_root = TreeNode(root.val)
    
    # If there are children, encode the first child
    if root.children:
        binary_root.left = encode(root.children[0])
    
    # Iterate through the rest of the children and link as right children
    curr = binary_root.left
    for child in root.children[1:]:
        curr.right = encode(child)
        curr = curr.right
    
    return binary_root

```

### Step 3: Decoding Function

For decoding, we will traverse the binary tree and reconstruct the N-ary tree:
1. The left child of the binary node corresponds to the first child of the N-ary node.
2. The right child serves as the next sibling node.

Here's how this can be implemented:


```python
def decode(root: TreeNode) -> Node:
    if not root:
        return None
    
    # Create an N-ary node from the binary node
    nary_node = Node(root.val)
    
    # Decode children from the binary tree
    if root.left:
        nary_node.children.append(decode(root.left))
    
    # Using a variable to keep track of sibling nodes
    curr = nary_node.children[0] if nary_node.children else None
    while curr and curr.right:
        nary_node.children.append(decode(curr.right))
        curr = curr.right
    
    return nary_node

```

### Complete Solution Code

Now we can combine everything into a complete solution with encoding and decoding functions ready for use:


```python
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def encode(root: Node) -> TreeNode:
    if not root:
        return None
    
    binary_root = TreeNode(root.val)
    
    if root.children:
        binary_root.left = encode(root.children[0])
    
    curr = binary_root.left
    for child in root.children[1:]:
        curr.right = encode(child)
        curr = curr.right
    
    return binary_root

def decode(root: TreeNode) -> Node:
    if not root:
        return None
    
    nary_node = Node(root.val)
    
    if root.left:
        nary_node.children.append(decode(root.left))
        
    curr = nary_node.children[0] if nary_node.children else None
    while curr and curr.right:
        nary_node.children.append(decode(curr.right))
        curr = curr.right
    
    return nary_node

```

### Explanation

1. **Structuring the Trees**: We create two classes, `Node` for the N-ary tree and `TreeNode` for the binary tree.
  
2. **Encoding Process**:
   - For every node in the N-ary tree, we create a corresponding binary tree node.
   - The first child of the N-ary node becomes the left child of the binary node, and subsequent children become right siblings.

3. **Decoding Process**:
   - For every binary tree node, the left child represents the first child in N-ary, and the right child represents the next sibling.
   - This process constructs the N-ary tree back to its original form.

With this code and explanation, you should be able to run this solution on LeetCode without issues.

# 437. Path Sum III

### Problem Description 
You are given a binary tree in which each node contains an integer value.

Find the number of paths that sum to a given value.

The path does not need to start or end at the root or a leaf, but it must go downwards
(traveling only from parent nodes to child nodes).

The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.


Example:
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8
      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1
Return 3. The paths that sum to 8 are:
1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11

### Solution 
 To solve this problem, we will implement a function that counts the number of paths that sum to a given value in a binary tree. The function will use a depth-first search (DFS) approach to explore all potential paths in the binary tree. The key is to keep track of the cumulative sum of the paths as we traverse from the root to any node in the tree.

### Approach

1. **DFS Traversal**: We'll traverse the tree using a recursive DFS function that explores each node.
2. **Path Tracking**: As we explore nodes, we'll maintain a dictionary to store the cumulative sum that has been seen so far. This allows us to check how many times the required sum has been encountered so far.
3. **Count Paths**: For each node, we determine how many times the difference between the current cumulative sum and the target sum has appeared in our path sums. This difference indicates how many times we have seen a sub-path that sums to our desired value.

### Detailed Steps

1. **Define the `TreeNode` class**: This is a standard class representing nodes in the binary tree.
2. **Define the main function `pathSum`**: This function initializes variables and calls the recursive helper function to start the DFS traversal.
3. **Define the helper function `dfs`**: This function does the recursive traversal, keeps track of the cumulative sum, and updates the counts of paths that sum to the given value.

### Python Code

Here is the complete code for the solution:



```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> int:
        # Dictionary to record the cumulative sum frequencies
        cumulative_sum_count = {0: 1}
        self.count = 0
        
        # Helper function for DFS
        def dfs(node, current_sum):
            if not node:
                return
            
            # Update the current path sum
            current_sum += node.val
            
            # Check how many times the (current_sum - sum) has been seen
            self.count += cumulative_sum_count.get(current_sum - sum, 0)
            
            # Update the count of the current_sum seen
            cumulative_sum_count[current_sum] = cumulative_sum_count.get(current_sum, 0) + 1
            
            # Traverse left and right children
            dfs(node.left, current_sum)
            dfs(node.right, current_sum)
            
            # After returning from the call, we need to backtrack
            # Remove the current sum from the dictionary to prevent affecting other paths
            cumulative_sum_count[current_sum] -= 1
        
        # Start DFS from the root with initial sum of 0
        dfs(root, 0)
        return self.count

```

### Explanation of the Code

- `TreeNode`: This class is a standard representation for the binary tree node with attributes for storing the value of the node and references to the left and right child nodes.
  
- `class Solution`: This contains the method `pathSum`, which takes the root of the tree and the target sum as inputs.

- `cumulative_sum_count`: A dictionary that maps each cumulative sum to the number of times it has been encountered.

- `dfs`: The recursive function that performs a depth-first search. For each node visited:
  - It updates the cumulative sum.
  - It checks if there is a previously seen cumulative sum that complements the current path to reach the desired target sum.
  - It adds or updates the current cumulative sum in the dictionary.
  - It continues the DFS on the left and right child nodes.
  - Backtracks by decrementing the count of the current cumulative sum in the dictionary.

- Finally, `return self.count` gives the total number of paths that sum to the target value.

The solution effectively counts all the valid paths in the tree and maintains a low space complexity due to the use of a dictionary for cumulative sums only while traversing down. This ensures it runs efficiently even with the constraint of up to 1,000 nodes.

# 449. Serialize and Deserialize BST

### Problem Description 
Serialization is converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary search tree. There is no restriction on how your serialization/deserialization algorithm should work. You need to ensure that a binary search tree can be serialized to a string, and this string can be deserialized to the original tree structure.

The encoded string should be as compact as possible.


Example 1:
Input: root = [2,1,3]
Output: [2,1,3]

Example 2:
Input: root = []
Output: []

Constraints:
The number of nodes in the tree is in the range `[0, 104]`.

`0 <= Node.val <= 104`
The input tree is guaranteed to be a binary search tree.

### Solution 
 To solve the problem of serializing and deserializing a binary search tree (BST), we can use a combination of pre-order traversal for serialization and constructing the tree back using the properties of BST during deserialization. 

### Key Concepts:
1. **Serialization**: We will convert the BST into a string representation using pre-order traversal. In a pre-order traversal, we first visit the root, then the left subtree, and finally the right subtree.
2. **Deserialization**: We will reconstruct the BST from the string. During this process, we will use the properties of the BST, where the left child is always less than the parent node, and the right child is always greater than the parent node.

### Approach:
- **Serialize**:
    - Use a recursive function to traverse the tree in pre-order. 
    - For each node, append its value to a list and return that list joined by commas to represent the serialized string.
    
- **Deserialize**:
    - Split the string back into integers.
    - Use a helper function that takes the boundaries for the current node (to ensure we maintain the BST properties) and recursively constructs the tree by finding the appropriate nodes from the list of values.

Here is the complete Python code implementing this concept, formatted according to LeetCode standards:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Codec:

    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string."""
        def preorder(node):
            if not node:
                return ["#"]  # Using '#' to indicate null nodes
            return [str(node.val)] + preorder(node.left) + preorder(node.right)
        
        return ','.join(preorder(root))

    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree."""
        def build_tree(vals):
            val = vals.pop(0)
            if val == "#":
                return None
            node = TreeNode(int(val))
            node.left = build_tree(vals)
            node.right = build_tree(vals)
            return node
        
        vals = data.split(',')
        return build_tree(vals)

# Example usage:
# codec = Codec()
# tree = codec.deserialize(codec.serialize(root))

```

### Explanation:
1. **TreeNode Class**: This is a basic structure to represent a node of the binary tree with a value, a left child, and a right child.
  
2. **Codec Class**:
    - `serialize`: This method defines a helper function `preorder` that traverses the tree and appends each node's value to a list. If a node is null, we append a placeholder (`#`).
    - `deserialize`: This method first splits the serialized string and then uses the helper function `build_tree`, which pops values from the list, creates a node for non-null values, and recursively assigns left and right children.

### Complexity:
- Time complexity for both serialization and deserialization is O(n), where n is the number of nodes, because we visit each node exactly once.
- Space complexity is also O(n) due to the space required to store the serialized string and the recursion stack during deserialization. 

This approach ensures that the serialized format is compact and we can recover the original BST efficiently.

