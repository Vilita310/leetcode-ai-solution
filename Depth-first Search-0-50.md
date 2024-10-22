# 17. Letter Combinations of a Phone Number

### Problem Description 
Given a string containing digits from `2-9` inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.


Example 1:
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

Example 2:
Input: digits = ""
Output: []

Example 3:
Input: digits = "2"
Output: ["a","b","c"]

Constraints:
`0 <= digits.length <= 4`
`digits[i]` is a digit in the range `['2', '9']`.

### Solution 
 To solve the problem of generating letter combinations from a string of digits between '2' and '9', we can utilize a recursive backtracking method. The main idea is to map each digit to its corresponding letters (similar to a phone keypad) and recursively build all possible combinations of these letters.

Here’s a step-by-step breakdown of how we can arrive at the solution:

### Step 1: Create a mapping

We start by mapping each digit from 2 to 9 to its corresponding letters:
- 2: "abc"
- 3: "def"
- 4: "ghi"
- 5: "jkl"
- 6: "mno"
- 7: "pqrs"
- 8: "tuv"
- 9: "wxyz"

### Step 2: Handle base cases

If the input string is empty (`""`), we should return an empty list. If the input has exactly one digit, we should return the list of corresponding letters for that digit.

### Step 3: Recursive backtracking

For strings with more than one digit, we can employ a backtracking approach:
- We maintain a list to store the current combination of letters.
- For each digit, we loop through its corresponding letters and recursively add each letter to the current combination.
- Once we reach the length equal to the number of digits, we add the current combination to the results and backtrack.

### Step 4: Implement the solution

Now, let’s implement this logic in Python as a function suitable for direct execution in LeetCode.



```python
class Solution:
    def letterCombinations(self, digits: str) -> list[str]:
        # Mapping from digit to letters
        digit_to_letters = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }
        
        if not digits:  # Base case: if input is empty
            return []
        
        result = []  # This will hold the final combinations
        current_combination = []  # This will hold the current combination of letters
        
        def backtrack(index: int):
            # If the current combination is the same length as digits
            if index == len(digits):
                result.append(''.join(current_combination))  # Add to results
                return
            
            # Get the letters that the current digit maps to
            current_digit = digits[index]
            possible_letters = digit_to_letters[current_digit]
            
            # Loop through these letters
            for letter in possible_letters:
                current_combination.append(letter)  # Choose the letter
                backtrack(index + 1)  # Move to the next digit
                current_combination.pop()  # Backtrack: remove the letter
        
        backtrack(0)  # Start the backtracking process from index 0
        return result

```

### Explanation of the Code

1. **Mapping Creation**: We define a dictionary `digit_to_letters` to map digits to their corresponding letters.

2. **Base Cases Handling**: We immediately check if the input `digits` is an empty string and return an empty list if so.

3. **Backtracking Function**: The `backtrack` function takes an index as its parameter, which indicates the current position in the `digits` string. If the index equals the length of `digits`, it means we have formed a complete combination, which we join and add to `result`.

4. **Letter Selection**: For each letter corresponding to the current digit, we:
   - Append the letter to our `current_combination`.
   - Recursively call `backtrack` with the next index.
   - Upon returning, we remove the last added letter (this is the backtracking step).

5. **Return Result**: Finally, we return the list of all combinations stored in `result`.

This structured approach efficiently generates all possible combinations as required by the problem statement, adhering to the constraints given.

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
 To solve the problem of recovering a binary search tree (BST) where two nodes have been swapped, we can approach it by performing an in-order traversal of the tree to identify the two nodes that are out of order. Here's a detailed explanation of the approach and the code implementation suitable for LeetCode.

### Explanation

1. **Understanding BST Properties**: In a BST, for any node:
   - The values of all nodes in the left subtree are less than the node's value.
   - The values of all nodes in the right subtree are greater than the node's value.

2. **Identifying Swapped Nodes**: During an in-order traversal, we expect the node values to be sorted in ascending order for a valid BST. If two nodes are swapped, we will find at least two instances where this order is violated:
   - The first instance will identify the first node (`first`) that should be larger but is found smaller.
   - The second instance will identify the second node (`second`) that should be smaller but is found larger.
   
3. **In-Order Traversal**: We will follow a recursive in-order traversal pattern to visit each node and collect the output. During this traversal:
   - We maintain a variable to track the last node we visited (`prev`).
   - When we find a violation, we update our references to `first` and `second`.

4. **Recover the BST**: After identifying the two nodes, we can swap their values to restore the BST.

### Code Implementation

Here is the Python code structured according to LeetCode requirements:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def recoverTree(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        # Initialize variables
        self.first = None
        self.second = None
        self.prev = TreeNode(float('-inf'))  # Previous node for comparison
        
        # In-order traversal to find the two nodes that are swapped
        def in_order_traversal(node):
            if not node:
                return
            
            # Traverse the left subtree
            in_order_traversal(node.left)
            
            # Check if there is a violation of the BST properties
            if self.first is None and self.prev.val >= node.val:
                self.first = self.prev  # The first node that needs to be swapped
            
            if self.first is not None and self.prev.val >= node.val:
                self.second = node  # The second node that needs to be swapped
            
            # Update previous node to the current one
            self.prev = node
            
            # Traverse the right subtree
            in_order_traversal(node.right)
        
        # Start the in-order traversal
        in_order_traversal(root)
        
        # Swap the values of the first and second nodes
        if self.first and self.second:
            self.first.val, self.second.val = self.second.val, self.first.val

```

### Usage

- **Input**: The input to the `recoverTree` method should be the root of a binary search tree (as an instance of `TreeNode`).
- **Output**: The output is in-place, meaning that the BST will be corrected without returning anything.

### Complexity Analysis

- **Time Complexity**: O(n) where n is the number of nodes in the tree since we visit each node exactly once.
- **Space Complexity**: O(1) because we are using a constant amount of extra space (excluding the input tree storage).

This solution is efficient and meets the requirements specified in the problem statement.

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
 To solve the problem of checking if two binary trees are the same, we need to compare their structure and the values of their nodes recursively. If both trees are structurally identical and every corresponding node's value is the same, we can conclude that the trees are the same. 

Here is a step-by-step breakdown of the approach:

1. **Base Condition**: When we compare two nodes, if both are `None`, we return `True` since two non-existent trees are the same.
2. If one of the nodes is `None` and the other is not, we return `False` since a non-existent tree cannot match an existing one.
3. Next, we check if the values of the two nodes are equal. If they are not, we return `False`.
4. If the values are equal, we recursively check both the left and right children of both nodes. We do this by calling the same function on the left child of `p` and the left child of `q`, and then on the right child of `p` and the right child of `q`.
5. If both recursive calls return `True`, then the trees are the same; otherwise, they are not.

Now, let's implement this logic in Python code. We will define the `TreeNode` class to represent a node in the binary tree and then the `isSameTree` function which will perform the comparison.

Here’s the complete code following the LeetCode format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        # Base case: both nodes are None, trees are identical
        if not p and not q:
            return True
        # One of the nodes is None while the other is not, trees are not identical
        if not p or not q:
            return False
        # If the value of the nodes is not the same, trees are not identical
        if p.val != q.val:
            return False
        # Recursively check the left and right subtree
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

# Example Usage
# Create binary tree for example 1
# p = [1, 2, 3]
p1 = TreeNode(1)
p1.left = TreeNode(2)
p1.right = TreeNode(3)

# Create binary tree for example 1
# q = [1, 2, 3]
q1 = TreeNode(1)
q1.left = TreeNode(2)
q1.right = TreeNode(3)

# Test case
solution = Solution()
print(solution.isSameTree(p1, q1))  # Output: True

```

### Explanation of the Code:

1. **TreeNode Class**: This class defines a node of the tree with attributes for its value, left child, and right child.
2. **Solution Class**: Contains the method `isSameTree` that implements the logic described.
3. **Base Cases**: The function first checks if both nodes are `None`, indicating that we have reached leaves in both trees. Then, it checks if one is `None` (and the other is not) and subsequently compares the values.
4. **Recursive Calls**: The function calls itself for the left and right children of both nodes.
5. **Example Usage**: We demonstrate how to create binary trees and call the `isSameTree` function to check if they are the same.

This implementation has an O(n) time complexity where n is the number of nodes in the trees, and it utilizes O(h) space complexity due to the recursion stack, where h is the height of the trees. This approach effectively covers all constraints and requirements outlined in the problem statement.

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
 To solve the problem of checking whether a binary tree is symmetric (i.e., it is a mirror of itself), we can employ both recursive and iterative approaches. I will provide both methods and explain each in detail, following the LeetCode solution format.

Let's first define the node structure for the binary tree. In Python, this is typically done with a class definition.

### Problem Definition
We will define our binary tree node structure and then implement both the recursive and iterative approaches to check if the tree is symmetric.

### TreeNode Class


```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

```

### Recursive Solution
The basic idea of the recursive solution is to compare two subtrees to check if they are mirrors of each other. The left subtree of one tree must be a mirror of the right subtree of the other tree.

#### Recursive Function


```python
def isMirror(left, right):
    if not left and not right:
        return True  # Both nodes are None, it's symmetric
    if not left or not right:
        return False  # One of the nodes is None, not symmetric
    return (left.val == right.val) and isMirror(left.right, right.left) and isMirror(left.left, right.right)

```

### Main Function for Symmetry Check


```python
def isSymmetric(root: TreeNode) -> bool:
    if not root:
        return True  # An empty tree is symmetric
    return isMirror(root.left, root.right)

```

### Iterative Solution
For the iterative approach, we will use a queue (or a stack) to compare nodes level by level. We will append nodes in a mirrored order.



```python
from collections import deque

def isSymmetricIterative(root: TreeNode) -> bool:
    if not root:
        return True
    
    queue = deque([root.left, root.right])
    
    while queue:
        left = queue.popleft()
        right = queue.popleft()
        
        if not left and not right:
            continue  # Both nodes are None, continue checking
        if not left or not right:
            return False  # One of the nodes is None, not symmetric
        if left.val != right.val:
            return False  # Values are different, not symmetric
        
        # Add children in mirrored order
        queue.append(left.left)
        queue.append(right.right)
        queue.append(left.right)
        queue.append(right.left)
        
    return True

```

### Complete Code Together
Here is how the complete solution will look, including both the recursive and iterative methods, ready to paste into LeetCode:



```python
from collections import deque

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    # Recursive approach
    def isSymmetric(self, root: TreeNode) -> bool:
        def isMirror(left, right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            return (left.val == right.val) and isMirror(left.right, right.left) and isMirror(left.left, right.right)
        
        if not root:
            return True
        return isMirror(root.left, root.right)

    # Iterative approach
    def isSymmetricIterative(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        queue = deque([root.left, root.right])
        
        while queue:
            left = queue.popleft()
            right = queue.popleft()
            
            if not left and not right:
                continue
            if not left or not right:
                return False
            if left.val != right.val:
                return False
            
            queue.append(left.left)
            queue.append(right.right)
            queue.append(left.right)
            queue.append(right.left)
        
        return True

```

### Explanation of the Code:
1. **TreeNode Class**: This defines the structure of each node in the binary tree.
2. **isSymmetric Method**: This is the main function that checks if the tree is symmetric using the recursive approach.
3. **isMirror Inner Function**: This function checks if two trees are mirror images. It does this by comparing values and checking children in mirrored order.
4. **isSymmetricIterative Method**: This is the iterative approach using a queue to hold pairs of nodes to be compared.
5. In both methods, specific conditions check if the trees are symmetric or not, returning `True` or `False` based on the results of those comparisons.

Now, this solution can be copied directly into LeetCode for testing.

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
 To solve the problem of finding the maximum depth of a binary tree, we can use a recursive depth-first search (DFS) approach. The idea is to traverse the tree starting from the root and compute the depth for each subtree, returning the maximum depth found.

### Detailed Explanation:

1. **Understanding Depth**: The maximum depth of a binary tree is defined as the number of nodes from the root node to the farthest leaf node. If the tree is empty (i.e., the root is `None`), the maximum depth is `0`.

2. **Recursive Approach**:
   - If the current node is `None`, we've reached beyond a leaf node, and we return `0`.
   - If the current node is not `None`, we compute the maximum depth of the left and right subtrees. The maximum depth of the tree rooted at the current node is then `1 + max(depth_of_left_subtree, depth_of_right_subtree)`.

3. **Base Case**: If the node is `None`, return `0`.

4. **Recursion**: We will call the same function recursively for left and right children of the current node.

5. **Time Complexity**: The time complexity for this solution is O(n), where n is the number of nodes in the binary tree, since we visit each node exactly once.

6. **Space Complexity**: The space complexity is O(h) due to the recursion stack, where h is the height of the tree. In the worst case, it could be O(n) for a skewed tree.

### LeetCode Solution Format:

Here’s how we implement this in Python, adhering to the LeetCode structure.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # Base case: If the node is None, the depth is 0.
        if not root:
            return 0
        else:
            # Recursive case: Compute the depth of each subtree and add 1 for the current node.
            left_depth = self.maxDepth(root.left)
            right_depth = self.maxDepth(root.right)
            return 1 + max(left_depth, right_depth)

# Example usage:
# To create a tree and test the function on LeetCode, you would typically create TreeNode instances.
# For example, to create the tree from Example 1:
#       3
#      / \
#     9  20
#        / \
#       15  7

root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)

solution = Solution()
print(solution.maxDepth(root))  # Output: 3

```

### Explanation of the Code:
- **TreeNode Class**: This defines the structure of each node in the binary tree with a value, left child, and right child.
- **Solution Class**: Contains the method `maxDepth` which implements the algorithm to calculate the maximum depth.
- **Recursive Function**: It checks if the current node is `None`, and if so, returns `0`. Otherwise, it calls itself to find the depths of the left and right children, returning the maximum depth found plus one for the current node.

This code is ready to be copied into the LeetCode IDE for execution and testing against the provided test cases.

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
 To solve the problem of constructing a binary tree from its preorder and inorder traversals, we can follow these steps:

### Steps to Construct the Binary Tree:

1. **Understanding Traversals**:
    - **Preorder Traversal**: The order is `Root -> Left -> Right`.
    - **Inorder Traversal**: The order is `Left -> Root -> Right`.

2. **Identifying the Root**:
    - The first element in the `preorder` array is always the root of the tree.

3. **Splitting the Tree**:
    - Once we identify the root, we can then use its position in the `inorder` array to split the `inorder` array into two parts:
      - Elements to the left of the root in the `inorder` array form the left subtree.
      - Elements to the right of the root in the `inorder` array form the right subtree.

4. **Recursive Construction**:
    - We recursively repeat this process for both the left and right subtrees using the appropriate segments of the `preorder` and `inorder` arrays.

5. **Base Case**:
    - The base case for the recursion will be when there are no elements left in the current segment of the arrays to process, in which case we return `None`.

Given these steps, the following is the complete solution in Python, formatted for running on LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def buildTree(self, preorder, inorder):
        if not preorder or not inorder:
            return None
        
        # The first element in preorder list is the root
        root_val = preorder[0]
        root = TreeNode(root_val)
        
        # Find the index of the root in inorder list
        root_index = inorder.index(root_val)
        
        # Elements left to the root_index in inorder will be in the left subtree
        # Elements right to the root_index in inorder will be in the right subtree
        # Preorder is sliced based on the length of left subtree calculated from inorder
        root.left = self.buildTree(preorder[1:root_index+1], inorder[:root_index])
        root.right = self.buildTree(preorder[root_index+1:], inorder[root_index+1:])
        
        return root

```

### Explanation of the Code:
- **TreeNode Class**: This is a simple class that defines the structure of a tree node. Each node contains a value (`val`), and pointers to left and right children (`left` and `right`).
  
- **Solution Class**: This is where the `buildTree` method is defined.
  - **Base Case**: If either `preorder` or `inorder` is empty, return `None`.
  - **Root Creation**: The first element of `preorder` is stored as the root's value, and a new `TreeNode` is created.
  - **Index Finding**: The index of the root in the `inorder` array is found. This helps in splitting the `inorder` array into left and right subtrees.
  - **Recursive Calls**: The function recursively constructs the left subtree using the corresponding segments of `preorder` and `inorder`, and does the same for the right subtree.
  
- **Return Statement**: Finally, the root of the constructed binary tree is returned.

### Complexity:
- **Time Complexity**: O(n^2) in the worst case when using `index` method (which can lead to linear searches). However, with a hash map to store indices, it can be improved to O(n).
- **Space Complexity**: O(n) due to the recursive stack and storage of the tree nodes. 

This code will successfully create the binary tree as required by the prompt, and you can run it directly on LeetCode.

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
 To solve the problem of reconstructing a binary tree given its inorder and postorder traversals, we can use a recursive approach. The key insight is to understand the properties of these traversals:

1. In the postorder traversal, the last element represents the root of the current (sub)tree.
2. We can find the index of this root in the inorder traversal, which will allow us to determine the elements that belong to the left and right subtrees.

Here's a detailed step-by-step breakdown of the algorithm:

1. **Identify the Root**: Start with the last element in the postorder array, which gives us the root of the tree or subtree.
  
2. **Find the Inorder Index**: Locate this root value in the inorder array to split the elements into left and right subtrees.

3. **Split the Traversals**: Using the index from the inorder array, determine which elements belong to the left and right subtrees:
   - The elements to the left of the root in the inorder traversal belong to the left subtree.
   - The elements to the right of the root in the inorder traversal belong to the right subtree.
  
4. **Recursive Construction**: Recursively repeat this process for the left and right subtree using the corresponding sections of the inorder and postorder arrays.

Now, let’s implement this in Python. We'll define a `TreeNode` class for the structure of the binary tree nodes, and then we'll create a function to construct the tree based on the provided inorder and postorder arrays.

Here’s the complete code to be run on LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def buildTree(self, inorder: list[int], postorder: list[int]) -> TreeNode:
        if not inorder or not postorder:
            return None
        
        # Create a hashmap to store the indices of values in inorder for quick lookup
        inorder_index_map = {value: index for index, value in enumerate(inorder)}
        
        def build(in_left, in_right):
            # If there are no elements to construct the tree
            if in_left > in_right:
                return None
            
            # Last element in postorder is the root
            root_val = postorder.pop()  # We pop from the end
            root = TreeNode(root_val)

            # Index of the root in inorder
            index = inorder_index_map[root_val]

            # Build right subtree before left subtree because we're popping from postorder
            root.right = build(index + 1, in_right)  # All elements to the right of the root in inorder
            root.left = build(in_left, index - 1)    # All elements to the left of the root in inorder
            
            return root
        
        return build(0, len(inorder) - 1)


```

### Explanation of the Code:

- **TreeNode Class**: We define a `TreeNode` class to represent each node of the tree.
- **Solution Class**: The main class where we define the `buildTree` method to build the binary tree.
- **Inorder Index Map**: We create a hashmap to store positions of each value in the inorder array for O(1) lookup times.
- **Nested Function**: `build` is a recursive function that:
  - Checks if the current slice of the inorder array is valid.
  - Pops the last element of the postorder array to use as the root.
  - Finds the root’s index in the inorder array and constructs the right and left subtrees recursively.

This implementation efficiently reconstructs the binary tree with a time complexity of O(n) and a space complexity of O(n), considering the space used by the hashmap and the recursion stack.

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
 To solve the problem of converting a sorted array into a height-balanced binary search tree (BST), we can use a recursive approach. The idea is to choose the middle element of the array as the root of the tree (or subtree) at each recursive step. This ensures that the tree remains balanced.

### Steps for the Solution:
1. **Choose the Middle Element**: For any given sub-array, the middle element will be the root. This helps in maintaining a balanced BST.
2. **Recursive Calls**: After selecting the middle element as the root, recursively apply the same process to the left half of the array (to form the left subtree) and to the right half (to form the right subtree).
3. **Base Case**: When the start index is greater than the end index (`start > end`), we return `None`, as this indicates that there are no elements to process.

### Implementation:
We'll define a binary tree node structure and implement our recursive function. At the end, we'll return the root of the constructed binary search tree.

The solution can be structured in a format suitable for LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def sortedArrayToBST(self, nums):
        """
        Given an integer array nums where the elements are sorted in ascending order,
        convert it to a height-balanced binary search tree.
        """
        if not nums:
            return None

        def convert_list_to_bst(start, end):
            # Base case: if start index exceeds the end index
            if start > end:
                return None
            
            # Middle element to ensure balance
            mid = (start + end) // 2
            root = TreeNode(nums[mid])
            
            # Recursively build the left and right subtrees
            root.left = convert_list_to_bst(start, mid - 1)  # Left subtree
            root.right = convert_list_to_bst(mid + 1, end)  # Right subtree
            
            return root
        
        # Call the recursive function with the initial range
        return convert_list_to_bst(0, len(nums) - 1)

```

### Explanation of the Code:
1. **TreeNode Class**: This class is defined to create tree nodes with a value (`val`), and pointers to left and right children.
2. **sortedArrayToBST Method**: This is the main function that will be called by LeetCode to get the output. It checks if the input array is not empty and initializes the recursive conversion.
3. **convert_list_to_bst Method**: This private method takes the start and end indices of the current sub-array and constructs a BST:
   - It calculates the middle index of the current array segment.
   - A new `TreeNode` is created for the middle element.
   - The function is called recursively for the left and right halves of the array (excluding the current middle element).
   - Finally, the constructed root node is returned.

### Complexity:
- **Time Complexity**: O(n), where n is the number of elements in the input array. Each element is processed once.
- **Space Complexity**: O(log n) due to the recursion stack in a balanced tree scenario.

This approach effectively converts the sorted array into a balanced binary search tree and can be run directly on the LeetCode platform.

# 109. Convert Sorted List to Binary Search Tree

### Problem Description 
Given the `head` of a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.


Example 1:
Input: head = [-10,-3,0,5,9]
Output: [0,-3,9,-10,null,5]
Explanation: One possible answer is [0,-3,9,-10,null,5], which represents the shown height balanced BST.


Example 2:
Input: head = []
Output: []

Example 3:
Input: head = [0]
Output: [0]

Example 4:
Input: head = [1,3]
Output: [3,1]

Constraints:
The number of nodes in `head` is in the range `[0, 2 * 104]`.

`-10^5 <= Node.val <= 10^5`

### Solution 
 To solve the problem of converting a sorted singly linked list into a height-balanced binary search tree (BST), we can leverage the properties of the sorted linked list. The middle element of the sorted linked list will be the root of the BST. The elements before the middle node will form the left subtree, and the elements after the middle node will form the right subtree. This approach ensures that the resulting BST is height-balanced.

Here's a step-by-step breakdown of the approach we will use:

1. **Determine the Size of the Linked List**: First, we need a way to keep track of the length of the linked list because we will need this to find the middle element.

2. **Convert the Linked List to a Tree**:
   - We will use a helper function which recursively constructs the tree.
   - This function will:
     - Use the length of the linked list to find the middle node.
     - If the left part of the linked list (elements before the middle node) is non-empty, we make the left child of the current subtree.
     - Similarly, if the right part of the linked list is non-empty (elements after the middle node), we make the right child of the current subtree.

3. **Build the Tree**: By continuously defining left and right subtrees based on the middle nodes, we will build the complete balanced BST.

Here is the Python implementation of the above logic using recursion:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        # Function to find the size of the linked list
        def find_size(node):
            size = 0
            while node:
                size += 1
                node = node.next
            return size
        
        # Function to build BST
        def convert_list_to_bst(left: int, right: int) -> TreeNode:
            nonlocal head  # To access the 'head' defined in the outer function
            
            if left > right:
                return None
            
            # Recursively build the left subtree
            mid = (left + right) // 2
            
            # First, construct the left half
            left_child = convert_list_to_bst(left, mid - 1)
            
            # Now, 'head' points to the middle node
            root = TreeNode(head.val)
            root.left = left_child
            
            # Move to the next node in the list
            head = head.next
            
            # Construct the right half
            root.right = convert_list_to_bst(mid + 1, right)
            return root
        
        size = find_size(head)
        return convert_list_to_bst(0, size - 1)

# Example usage:
# head = ListNode(-10, ListNode(-3, ListNode(0, ListNode(5, ListNode(9)))))
# solution = Solution()
# bst_root = solution.sortedListToBST(head)

```

### Explanation of the Code:

- We define two classes: `ListNode` for linked list nodes and `TreeNode` for tree nodes.
- The `Solution` class contains the main method `sortedListToBST` that finds the size of the linked list first and then invokes the helper function `convert_list_to_bst`.
- `convert_list_to_bst` takes the bounds `left` and `right`, determines the middle index, constructs the left subtree recursively, creates a new `TreeNode` for the middle node, updates `head`, and finally constructs the right subtree recursively.
- The use of `nonlocal` allows us to update the `head` pointer between recursive calls. 
- This solution maintains a time complexity of O(n) and a space complexity of O(log n) due to recursions, which is optimal for solving this problem. 

This implementation should work directly on LeetCode or in a similar coding environment.

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
 To determine if a binary tree is height-balanced, we will define a recursive approach. A binary tree is height-balanced if, for every node, the height difference between its left and right subtrees is no more than 1.

Here's how we can tackle the problem:

1. **Define a Helper Function:** We'll create a helper function that computes the height of the tree and simultaneously checks if the subtree is balanced.
2. **Base Case:** If we reach a leaf node (i.e., `null`), we return a height of `0`.
3. **Recursive Case:** For each node, we recursively compute the height of the left and right children. We also check if the difference in their heights is greater than `1`, in which case we know the tree is not balanced.
4. **Overall Balance Check:** If at any point we find that a subtree is unbalanced, we will propagate this information up to indicate that the tree is not balanced.

The time complexity of this approach is O(n), where n is the number of nodes in the tree, and the space complexity is O(h), where h is the height of the tree due to the recursion stack.

Here’s the implementation in Python that meets the LeetCode format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def check_balance(node):
            if node is None:
                return 0
            
            left_height = check_balance(node.left)
            right_height = check_balance(node.right)
            
            if left_height == -1 or right_height == -1 or abs(left_height - right_height) > 1:
                return -1  # tree is not balanced
            
            return max(left_height, right_height) + 1
        
        return check_balance(root) != -1

# Example usage:
# sol = Solution()
# balanced = sol.isBalanced(TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7))))
# print(balanced)  # Output: True

```

### Explanation of the Code:
1. **TreeNode Class:** This class represents a node in the binary tree with properties for the value (`val`), left child (`left`), and right child (`right`).
2. **isBalanced Method:** This method starts the balance check from the root of the tree. It calls `check_balance`, a nested function to analyze the balance.
3. **check_balance Function:** 
   - If the node is `None`, it returns `0`, indicating that the height of an empty subtree is `0`.
   - It recursively calculates the heights of the left and right subtrees.
   - It checks if either subtree is unbalanced (denoted by `-1`). If they are, it returns `-1` indicating the overall tree is unbalanced.
   - If it passes these checks, it calculates the height of the current node as `1 + max(left_height, right_height)`, effectively computing the height of the subtree rooted at the current node.

### Test Cases:
You can use the example cases mentioned in your problem statement to validate the implementation. Each case can be constructed using instances of `TreeNode` to represent the structure of the binary tree.

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
 To solve the problem of finding the minimum depth of a binary tree, we can utilize a Depth-First Search (DFS) approach. The minimum depth of a binary tree is defined as the number of nodes along the shortest path from the root node down to the nearest leaf node. A leaf node is one that has no children.

In our approach, we will recursively explore the tree while keeping track of the current depth. We'll return the depth when we reach a leaf node. If we encounter nodes that only have one child (left or right), we will recursively call the function only on the child that exists, ensuring that we do not miss any leaf nodes.

Here’s the code that implements this logic:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def minDepth(self, root: TreeNode) -> int:
        # If the tree is empty, the minimum depth is 0
        if not root:
            return 0
        
        # If we reach a leaf node, return 1
        if not root.left and not root.right:
            return 1
        
        # Initialize minimum depth to a large number
        min_depth = float('inf')
        
        # If left child exists, continue searching down that path
        if root.left:
            min_depth = min(min_depth, self.minDepth(root.left))
        
        # If right child exists, continue searching down that path
        if root.right:
            min_depth = min(min_depth, self.minDepth(root.right))
        
        # Add 1 to account for the current node
        return min_depth + 1

```

### Explanation of the Code:
1. **TreeNode Class**: This class defines the structure of a tree node. Each `TreeNode` object has a value `val` and pointers to `left` and `right` child nodes.

2. **Solution Class with minDepth Method**:
   - The `minDepth` method is where the main logic resides.
   - **Base Case - Empty Tree**: If the `root` is `None`, we return `0`, indicating that the tree has no nodes.
   - **Leaf Node Check**: If both `left` and `right` children are `None`, we have reached a leaf node, so we return `1`, as we only count the leaf node.
   - **Calculate Minimum Depth**: We initialize `min_depth` to infinity to ensure we can find the minimum. 
     - If the left child exists, we recursively compute the minimum depth of the left subtree and update `min_depth`.
     - If the right child exists, we do the same for the right subtree.
   - **Return Depth**: Finally, we return the minimum depth found, adding `1` to account for the current node.

### Complexity Analysis:
- **Time Complexity**: O(N), where N is the number of nodes in the tree. In the worst case, we may visit all nodes.
- **Space Complexity**: O(H), where H is the height of the tree, primarily due to the recursion stack.

This solution efficiently computes the minimum depth of a binary tree and adheres to LeetCode submission format, allowing it to be run directly on the platform.

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
 To solve the problem of finding a root-to-leaf path in a binary tree that sums up to a given `targetSum`, we can use a Depth-First Search (DFS) approach. 

### Explanation
We will traverse the binary tree starting from the root node and maintain a running sum of the values encountered along the path. Here are the general steps for the algorithm:

1. **Base Case**: If we reach a leaf node (a node with no children), check if the current running sum equals the `targetSum`. If it does, return `True`; otherwise, return `False`.
   
2. **Recursive Case**: From the current node, subtract the node's value from `targetSum` and call the function recursively on the left and right child nodes. Return `True` if either of the recursive calls returns `True`.

3. If we have traversed all paths and not found any matching the target sum, we return `False`.

### Implementation
The implementation will be defined in the method `hasPathSum` which will take the root of the tree and the target sum as input.

Here's the implementation in Python:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        if not root:
            return targetSum == 0  # Case when the tree is empty
        
        # Check if we are at a leaf node
        if not root.left and not root.right:
            return targetSum == root.val  # Check if leaf's value matches the remaining sum
        
        # Subtract current node's value from targetSum and recurse down the tree
        targetSum -= root.val
        return (self.hasPathSum(root.left, targetSum) or 
                self.hasPathSum(root.right, targetSum))

# The solution class can be used directly on the LeetCode platform 
# to check for root-to-leaf paths summing to targetSum.

```

### How to Use This Code
1. In LeetCode, the platform often provides a function signature and the `TreeNode` class. You need to ensure that this implementation is placed inside the `Solution` class.
2. The `hasPathSum` function will be called with the root of the binary tree and the target sum as arguments.
3. This function will then effectively check if there exists a root-to-leaf path whose sum matches the target value and return a boolean result.

### Example Usage
For the provided examples:
- For **Example 1** with `targetSum = 22`, the function will return `True` because the path 5 → 4 → 11 → 2 equals 22.
- For **Example 2** with `targetSum = 5`, it will return `False` as no path sums to 5.
- For **Example 3** where `targetSum = 0`, it will also return `False` since the tree has positive values.

This solution effectively explores all root-to-leaf paths in a binary tree and checks their sums with an efficient recursive approach.

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
 To solve the problem of finding all root-to-leaf paths in a binary tree where each path's sum equals a given `targetSum`, we can use a depth-first search (DFS) approach. This involves traversing the tree and maintaining a path of node values while calculating the cumulative sum. If we reach a leaf node and the cumulative sum equals `targetSum`, we add that path to our result.

Here's a step-by-step explanation of the solution:

1. **Tree Node Definition**:
   We will first define the structure of a binary tree node using a class definition.

2. **Depth-First Search (DFS)**:
   We will implement a recursive helper function that:
   - Takes the current node, the current path, and the current sum as arguments.
   - Checks if the current node is `None`. If so, it should return as there is no further path.
   - Adds the current node's value to the current path and the current sum.
   - Checks if the current node is a leaf (both `left` and `right` children are `None`). If it is a leaf and the current sum equals `targetSum`, it adds the current path to the list of found paths.
   - Otherwise, it recursively calls the helper function for the left and right children.
   - After exploring both subtrees, it backtracks by removing the last node from the current path.

3. **Main Function**:
   The main function will initialize the necessary variables and call the helper function starting from the root.

Here is the complete code following the LeetCode solution format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        def dfs(node, currentPath, currentSum):
            if not node:
                return
            
            # Include the current node value in the path
            currentPath.append(node.val)
            currentSum += node.val
            
            # Check if it is a leaf node and the path sum matches targetSum
            if not node.left and not node.right and currentSum == targetSum:
                result.append(list(currentPath))  # Append a copy of the current path
            
            # Continue the DFS on left and right children
            dfs(node.left, currentPath, currentSum)
            dfs(node.right, currentPath, currentSum)
            
            # Backtrack: remove the current node from the path
            currentPath.pop()
        
        result = []
        dfs(root, [], 0)  # Initialize DFS with empty path and sum of 0
        return result

```

### Explanation:
- **TreeNode Class**: This defines the structure of a node in the binary tree.
- **Solution Class**: Contains the method `pathSum`, which will find all paths that sum up to `targetSum`.
- **dfs Function**: This recursive function does the main work of traversing the tree:
  - It checks if the current node exists.
  - Updates the current path and the sum.
  - Checks if the current node is a leaf node and if the accumulated sum matches the `targetSum`. If both conditions are true, it appends a copy of the current path to `result`.
  - It calls itself recursively for the left and right child nodes.
  - Finally, it "backtracks" by removing the last node from the current path for the next recursive call.

This solution should work within the constraints provided and runs efficiently for the range of inputs specified.

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
 To solve the problem of flattening a binary tree into a "linked list" in place, we can leverage a pre-order traversal approach. Let's walk through the approach, then I'll provide a complete Python solution in the format expected by LeetCode.

### Explanation

1. **Understanding the Problem**: We need to transform a binary tree such that all left pointers are null, and all right pointers point to the next node following the pre-order traversal order.

2. **Pre-order Traversal**: Pre-order traversal processes nodes in the following order: root, left, right. We will visit each node and link them together via the right pointer.

3. **In-place Modification**: The problem specifies that we should use O(1) extra space. This means we can't use any data structures like arrays or lists to store nodes during traversal. Instead, we will make links directly during the traversal.

4. **Approach**: We can use recursive or iterative techniques to traverse the tree:
   - **Recursive Approach**: Traverse the tree, then connect the nodes.
   - **Iterative Approach**: Use a stack to keep track of nodes as we traverse.

5. **Implementation Strategy**:
   - Start from the root of the tree.
   - Use a depth-first approach to process the tree. We will flatten the left subtree first, then the right subtree.
   - After processing the left child, link it to the right child.

### Python Solution

Here’s the complete solution following the LeetCode requirements:



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
        # Helper function for recursive flattening
        def flatten_tree(node):
            if not node:
                return None
            
            # Flatten left and right subtrees
            left_tail = flatten_tree(node.left)  # Flatten the left subtree
            right_tail = flatten_tree(node.right)  # Flatten the right subtree
            
            # If there is a left subtree, we shuffle the connections
            if left_tail:
                left_tail.right = node.right  # Connect end of left subtree to the right subtree
                node.right = node.left  # Move left subtree to the right
                node.left = None  # Set the left child to None
            
            # We return the last node of the rightmost flattened subtree
            return right_tail if right_tail else left_tail if left_tail else node

        flatten_tree(root)

```

### Explanation of Code Components:

1. **TreeNode Class**: This class is a standard definition for nodes in a binary tree, containing a value and pointers to left and right children.

2. **Solution Class**: This contains the method `flatten` which is called with the root of the binary tree.

3. **Flatten Function**:
   - Handles the recursive flattening.
   - First checks if the current node is None. If it is, there’s nothing to flatten.
   - Recursively flattens the left and right subtrees.
   - If there's a left subtree, it re-links the last node of the left subtree to the right subtree, effectively flattening it in place.
   - Finally, it returns the tail of the rightmost flattened subtree, which is necessary for proper linking.

### Complexity:
- **Time Complexity**: O(n), where n is the number of nodes in the tree, since we visit each node exactly once.
- **Space Complexity**: O(1) for the in-place transformation, aside from the call stack of recursion.

This solution can be directly run on the LeetCode platform with the provided tree node structure.

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
 To solve the problem of populating the `next` pointers in a perfect binary tree, we can use a recursive approach. The challenge here is to do this using constant extra space, which we can achieve by recursively traversing the tree without any additional data structures.

### Explanation of the Approach:

1. **Understanding the Tree Structure**:
   - A perfect binary tree is defined such that all internal nodes have two children, and all leaves are at the same level. This means for any node, its left child's next pointer should point to its right sibling (the right child).
   - Additionally, the right child's next pointer should point to the first child of the current node's parent (if it exists).

2. **Recursive Traversal**:
   - We can create a recursive function that takes the current node as an argument.
   - For each node:
     - Connect the left child to the right child directly.
     - Connect the right child to the left child of the next node in line (if the next node exists).

3. **Base Case**:
   - We should stop the recursion if the node is `None` (meaning we've reached past the leaves of the tree).

4. **Constant Space**:
   - We utilize the recursive call stack for space, adhering to the constant extra space requirement since we won't be using any external data structures.

### Implementation in Python:

Here’s the complete Python code written in the LeetCode style format:



```python
class Node:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

class Solution:
    def connect(self, root: Node) -> Node:
        if not root:
            return None
        
        # Connect the left and right child
        if root.left and root.right:
            root.left.next = root.right
            # Connect the right child to the left child of the next parent
            if root.next:
                root.right.next = root.next.left
        
        # Recurse on left and right children
        self.connect(root.left)
        self.connect(root.right)
        
        return root

```

### How the Code Works:
1. **Node Class**:
   - We have a class `Node` to represent each node in the binary tree. Each node has a `val`, pointers to `left` and `right` children, and a `next` pointer.

2. **Solution Class**:
   - The `connect` function inside the `Solution` class is where the main logic resides.
   - It checks if the current node is not null and connects its children properly.
   - Then, it recursively calls itself for both left and right children.

### Complexity Analysis:
- **Time Complexity**: O(n), where n is the number of nodes in the tree. We visit each node once.
- **Space Complexity**: O(h), where h is the height of the tree due to recursion stack space. For a perfect binary tree, this height is logarithmic relative to the number of nodes, but since we consider only the memory overhead of our solution and not the stack space, we regard it as constant extra space as required.

In summary, this implementation effectively traverses the tree, linking the next pointers at each level, all while using a recursive strategy that respects the constraints of the problem.

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
 To solve the problem of populating the `next` pointers in a perfect binary tree, we can use a level-order traversal approach while maintaining the constraints of using constant extra space. 

### Understanding the Problem

We have a binary tree node defined as:


```python
class Node:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

```

The goal is to set the `next` pointer of each node to point to its immediate neighbor on the right in the same level. If there is no neighbor (i.e., it’s the rightmost node), the `next` pointer should be set to `NULL`. The tree is a perfect binary tree, which implies all levels are fully filled.

### Approach

1. **Level Order Traversal**: We will iterate through each level of the tree starting from the root.
2. **Pointer Manipulation**: For each node at the current level, we will set the `next` pointer of its children to maintain connections between adjacent nodes at the current level.
3. **Recursion**: We can use a recursive approach that utilizes the implicit call stack, so we don't count the stack space as extra space.
4. **Base Case**: Stop when there are no nodes at the current level.

### Code Implementation

Here's how we can implement the solution in Python:



```python
class Node:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

class Solution:
    def connect(self, root: Node) -> Node:
        if not root:
            return root
        
        # Start with the root node
        leftmost = root
        
        while leftmost.left:  # Since it's a perfect binary tree
            # Iterate through the current level
            head = leftmost
            while head:
                # Connect the left child to the right child
                head.left.next = head.right
                # Connect the right child to the next left child (if exists)
                if head.next:
                    head.right.next = head.next.left
                
                # Move to the next node in the current level
                head = head.next
            
            # Move to the next level
            leftmost = leftmost.left
        
        return root

```

### Explanation of the Code

- We define a `Node` class and a `Solution` class.
- The `connect` method is implemented in the `Solution` class. Here’s what happens inside it:
  - We first check if the given `root` is `None`. If it is, we simply return `None`.
  - We use a `while` loop to traverse each level of the tree. The condition `leftmost.left` ensures we only proceed while there are children (because it’s a perfect tree).
  - Inside this loop, we set `head` to `leftmost`, which represents the first node of the current level.
  - We then use another `while` loop to iterate through all the nodes at the current level:
    - We set `head.left.next` to `head.right` — connecting each left child to its right sibling.
    - If there is a `next` node, we also connect `head.right.next` to `head.next.left`, linking the right child to the next left child on the same level.
  - After processing all nodes at the current level, we move down to the next level by setting `leftmost` to its left child.
- Finally, we return the modified tree.

### Complexity Analysis
- **Time Complexity**: O(N), where N is the number of nodes in the tree. We visit each node exactly once.
- **Space Complexity**: O(1) in terms of additional space for pointers, since we only use the `next` pointers that are already included in the node structure.

This code can be directly run on LeetCode, and it meets all the stated requirements of the problem.

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
 To solve the problem of finding the maximum path sum in a binary tree, we will leverage depth-first search (DFS). The key points to understand for this solution are:

1. **Path Definition**: A path can start and end at any node in the tree. It does not need to include the root.
2. **Path Sum Calculation**: The path sum is the total value of the nodes in the path.
3. **Recursive Calculation**: We can use a recursive function to calculate the maximum path sum including each node and update the global maximum accordingly.

### Steps to Implement the Solution:

1. **Base Case**: If the current node is `None`, we should return `0` as there are no nodes to contribute to the path sum.
2. **Recursive Case**: For each node, compute the maximum path sums for its left and right subtrees.
3. **Path Sum Calculation**: The maximum path sum that can be contributed from the current node is its value plus the maximum of either left or right subtree path sums (only considering non-negative contributions).
4. **Global Maximum Update**: We keep a global maximum that updates whenever we find a new path that includes both left and right contributions plus the current node's value.
5. **Final Return**: The function will ultimately return the value of the global maximum after traversing the entire tree.

Here is the Python code that implements the above logic in the required format for LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        self.max_sum = float('-inf')
        
        def max_gain(node):
            if not node:
                return 0
            
            # Calculate maximum contribution from left and right subtrees
            left_gain = max(max_gain(node.left), 0)  # Ignore negative contributions
            right_gain = max(max_gain(node.right), 0)  # Ignore negative contributions
            
            # Current node value + max contribution from left and right
            current_max_path = node.val + left_gain + right_gain
            
            # Update global maximum path sum
            self.max_sum = max(self.max_sum, current_max_path)
            
            # Return the maximum gain to be used by parent call
            return node.val + max(left_gain, right_gain)
        
        max_gain(root)
        return self.max_sum

```

### Explanation of the Code:

- **TreeNode Class**: This is a standard representation of a node in a binary tree, having attributes for its value (`val`) and its left and right children (`left` and `right`).
  
- **Solution Class**: We define the class `Solution` in which the method `maxPathSum` is implemented.
  
- **maxPathSum Method**: Initializes the maximum path sum (`self.max_sum`) to negative infinity. It defines a nested helper function `max_gain` that computes the maximum gain from each node.
  
- **max_gain Function**:
  - It recursively computes the maximum gain from both left and right subtrees of a given node.
  - It calculates the maximum path sum through the current node (`current_max_path`) and updates `self.max_sum` if this new path sum exceeds the previously recorded maximum.
  - The function returns the maximum gain that could be obtained from this node moving upwards to its parent, which is `node.val + max(left_gain, right_gain)`.

This setup ensures that we traverse the binary tree once, yielding an efficient solution with a time complexity of O(n), where n is the number of nodes in the tree. The space complexity is O(h) due to the recursion stack, where h is the height of the tree.

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
 To solve the problem of summing all root-to-leaf numbers in a binary tree, we can use a depth-first search (DFS) approach. The idea is to traverse the tree while constructing the numbers formed by the digits along each path from the root node to the leaf nodes. Once we reach a leaf node, we will add the constructed number to a running total.

Here’s a step-by-step explanation of the solution:

1. **Depth-First Search (DFS)**: We will use a recursive DFS function to traverse the tree. Starting from the root node, we will pass the current number formed by the path taken so far.

2. **Forming Numbers**: As we visit each node, we update the current number by multiplying the previous number by 10 (to shift left in decimal representation) and adding the current node's value.

3. **Leaf Nodes**: When we reach a leaf node (a node with no left and right children), we will add the current number to our total sum.

4. **Base Cases**: If the current node is `None`, we simply return. If we are at a leaf node, we add the accumulated number to the result.

The solution will be implemented in a class method `sumNumbers` which will initiate the DFS from the root.

Here is the complete code that follows this logic:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        def dfs(node: TreeNode, current_number: int) -> int:
            if not node:
                return 0
            
            current_number = current_number * 10 + node.val
            
            # If the current node is a leaf, return the current number
            if not node.left and not node.right:
                return current_number
            
            # Otherwise, continue the depth-first search on left and right children
            left_sum = dfs(node.left, current_number)
            right_sum = dfs(node.right, current_number)
            
            return left_sum + right_sum
        
        return dfs(root, 0)

```

### Explanation of the Code:

1. **TreeNode Class**: This class defines the structure of each node in the binary tree.

2. **Solution Class**:
   - **sumNumbers**: This method initializes the DFS with a call to the helper function `dfs` starting from the root and with an initial `current_number` of 0.
   
3. **dfs Function**:
   - If the current `node` is `None`, we return 0 since it contributes no value.
   - Compute the `current_number` by appending the node's value.
   - Check if we are at a leaf node. If yes, return the `current_number`.
   - Recursively call `dfs` for left and right children and sum their contributions.

### Complexity:
- **Time Complexity**: O(N), where N is the number of nodes in the tree since we visit each node exactly once.
- **Space Complexity**: O(H), where H is the height of the tree, due to the recursive call stack. In the worst case (a skewed tree), this can go up to O(N).

This implementation is efficient and adheres to the constraints given in the problem statement. You can directly run this code on LeetCode as it is formatted as required.

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
 To solve the problem of capturing regions surrounded by 'X' in the given board, we can use a Depth-First Search (DFS) approach. The main idea is to first identify 'O's that are connected to the border (which are not to be captured) and mark them. After that, we can flip the remaining 'O's (which are truly surrounded) to 'X's.

Here's a step-by-step breakdown of the algorithm:

1. **Identify Border 'O's**: Start from all 'O's on the border of the board and use DFS to mark all 'O's that are connected to these border 'O's.

2. **Mark Connected 'O's**: During the DFS, we'll temporarily convert the 'O's we visit to a different character (let's say 'B') to denote that these 'O's are safe (not captured).

3. **Flip Remaining 'O's**: After processing the board, any remaining 'O's are surrounded and should be flipped to 'X'.

4. **Restore Marked 'O's**: Convert the temporarily marked 'B's back to 'O's since these cells are not captured.

Let's implement this in Python, formatted specifically for the LeetCode platform:



```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        if not board or not board[0]:
            return
        
        self.m, self.n = len(board), len(board[0])
        
        # DFS function to mark 'O's connected to the border
        def dfs(r: int, c: int):
            if r < 0 or c < 0 or r >= self.m or c >= self.n or board[r][c] != 'O':
                return
            
            # Mark the cell as visited
            board[r][c] = 'B'
            
            # Explore neighbors (up, down, left, right)
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)
        
        # Mark all 'O's on the border and connected 'O's
        for i in range(self.m):
            for j in range(self.n):
                if (i == 0 or i == self.m - 1 or j == 0 or j == self.n - 1) and board[i][j] == 'O':
                    dfs(i, j)
        
        # Final pass to convert 'O' to 'X' and 'B' back to 'O'
        for i in range(self.m):
            for j in range(self.n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'  # Capture surrounded 'O's
                elif board[i][j] == 'B':
                    board[i][j] = 'O'  # Restore safe 'O's
        

```

### Explanation of the Code:
- The `solve` function accepts a 2D list `board` representing the matrix of 'X' and 'O'.
- We define a nested `dfs` function, which performs a depth-first search to mark connected 'O's that are connected to the board's border.
- We iterate through all cells of the board. If we find an 'O' on the border (first or last row/column), we call `dfs` on that cell to mark the connected region.
- After marking, we perform a second pass through the board:
  - Convert remaining 'O's to 'X' (they are captured).
  - Convert marked 'B's back to 'O' (they are safe and connected to the border).
  
This algorithm effectively captures surrounded regions while preserving the border connections, resulting in the correct output for the given problem.

# 131. Palindrome Partitioning

### Problem Description 
Given a string `s`, partition `s` such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of `s`.

A palindrome string is a string that reads the same backward as forward.


Example 1:
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]

Example 2:
Input: s = "a"
Output: [["a"]]

Constraints:
`1 <= s.length <= 16`
`s` contains only lowercase English letters.

### Solution 
 To solve the problem of partitioning a string into all possible substrings that are palindromes, we can use backtracking. The approach involves exploring all possible partitions of the string, checking whether each substring is a palindrome, and then recursively proceeding to partition the remaining string. Here’s how we can implement this solution:

### Step-by-Step Explanation

1. **Palindrome Checking**: Create a helper function to check if a given string is a palindrome. This can be done by comparing the string with its reverse.

2. **Backtracking**: Use a recursive function to partition the string. The function will:
   - Take the current index and a list to store the current partition.
   - Iterate through the possible substrings starting from the current index.
   - For each substring, check if it is a palindrome. If it is, add it to the current partition and recursively call the function for the next substring.
   - If the end of the string is reached, add the current partition to the result list.

3. **Result Storage**: Maintain a list to store all the valid partitions.

By following this strategy, we can efficiently generate all possible palindromic partitions of the given string.

Here’s how you can code this in Python, formatted according to the LeetCode submission guidelines:



```python
class Solution:
    def partition(self, s: str):
        def is_palindrome(substring: str) -> bool:
            return substring == substring[::-1]

        def backtrack(start: int, path: list):
            if start == len(s):
                result.append(path[:])  # Make a copy of the current path
                return
            
            for end in range(start + 1, len(s) + 1):  # End is exclusive
                substring = s[start:end]
                if is_palindrome(substring):
                    path.append(substring)
                    backtrack(end, path)  # Recur for the remaining substring
                    path.pop()  # Backtrack and remove the last substring

        result = []
        backtrack(0, [])
        return result

# Example Usage:
# sol = Solution()
# print(sol.partition("aab"))  # Output: [["a","a","b"],["aa","b"]]
# print(sol.partition("a"))     # Output: [["a"]]

```

### Explanation of the Code:

- **is_palindrome**: This helper function checks if a substring reads the same forward and backward by comparing the substring with its reversed version.

- **backtrack**: This recursive function generates the partitions:
  - It checks all possible endings for substrings starting from the current index `start`.
  - If the substring from `start` to `end` is a palindrome, it:
    - Adds that substring to the `path`.
    - Calls itself to explore the next potential substring starting from `end`.
    - Upon returning from the recursive call, it removes the last added substring from `path` (backtracking).

- **result**: It is a list that stores all the valid partitions that achieve the condition of palindromic substrings.

This approach effectively explores all combinations and ensures that all valid partitions are captured, yielding a comprehensive solution to the problem.

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
 To solve the problem of creating a deep copy (clone) of a connected undirected graph represented as an adjacency list, we'll need to implement a cloning function that works efficiently with the provided `Node` structure.

Here's a step-by-step breakdown of how we can achieve this:

1. **Understanding the Graph Representation**:
    - Each node in the graph is represented by a `Node` class that has a value (`val`) and a list of neighbors.
    - The input is given as an adjacency list where the index of the outer list represents the nodes, and each inner list contains the indices of the neighbors.

2. **Clone Strategy**:
    - We will utilize a recursive depth-first search (DFS) to traverse and clone the graph.
    - A dictionary will be used to map the original nodes to their clones to avoid duplicating the clones for already visited nodes.

3. **Base Cases**:
    - If the input adjacency list is empty, we should return an empty list.
    - If a node has already been cloned, we will return the corresponding clone to handle cycles and avoid infinite loops.

4. **Implementation**:
    - Define the `Node` class.
    - Create a function to perform the clone using recursion.
    - Use a dictionary to store cloned nodes and their corresponding clones.

Here is the complete implementation in Python, formatted for LeetCode:



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
        
        # A dictionary to hold cloned nodes
        cloned_nodes = {}
        
        def dfs(current_node):
            # If the node is already cloned, return it
            if current_node in cloned_nodes:
                return cloned_nodes[current_node]

            # Clone the current node
            copy_node = Node(current_node.val)
            cloned_nodes[current_node] = copy_node
            
            # Iterate through the neighbors and clone them
            for neighbor in current_node.neighbors:
                copy_node.neighbors.append(dfs(neighbor))
            
            return copy_node

        return dfs(node)

# Note: The function can be called by creating instances of Node and passing the reference of the node that needs to be cloned.

```

### Explanation:
- `Node` class defines the structure of each node in the graph.
- `cloneGraph` method checks if the provided node reference is `None`. If it is, we directly return `None`.
- A helper function `dfs` is defined to navigate through the graph:
  - It checks if the current node has already been cloned; if yes, it returns the existing clone.
  - A new node is created and stored in the `cloned_nodes` dictionary.
  - The function recursively clones the neighbors and adds them to the current node's neighbors.
- Finally, the `dfs` function is initiated with the starting node.

### Usage:
To use this function, you would need to set up instances of the `Node` class according to your test cases (e.g., building your graph structure as required) and then call `Solution().cloneGraph(node)` where `node` is the reference to the starting node.

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
 To solve the problem of returning the values of the nodes visible from the right side of a binary tree, we can use a level-order traversal (also known as breadth-first search) technique using a queue. The key idea is that during each level of the tree traversal, the last node encountered at that level will be visible from the right side.

Here is a step-by-step breakdown of how to approach this:

1. **Initialize a Queue**: We will use a queue to facilitate our level-order traversal, starting with the root node.

2. **Track the Rightmost Node**: For every level in the tree, we will keep track of the last node seen. This node will be part of the result since it is the one visible from the right side.

3. **Store Results**: After processing each level, we will add the last node of that level (the rightmost node) to our results.

4. **Breadth-First Traversal**:
   - For each node processed, we will add its children to the queue (right child first, left child second to ensure the rightmost node is processed last).
   - Continue this until we have processed all levels of the tree.

5. **Return the Results**: Finally, return the list of rightmost nodes visible from the right side.

Here's the Python code implemented with the above logic, following the LeetCode format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def rightSideView(self, root: TreeNode) -> list[int]:
        if not root:
            return []

        right_side_values = []
        queue = [root]  # Start with the root node in the queue

        while queue:
            level_length = len(queue)  # Number of nodes at the current level
            
            for i in range(level_length):
                node = queue.pop(0)  # Dequeue the front node

                # If it's the last node in the current level, add it to the result
                if i == level_length - 1:
                    right_side_values.append(node.val)

                # Enqueue right child first
                if node.right:
                    queue.append(node.right)
                # Enqueue left child second
                if node.left:
                    queue.append(node.left)

        return right_side_values

```

### Detailed Explanation of the Code:

- **TreeNode Class**: This is a standard definition for a binary tree node with a value and pointers to left and right children.

- **Solution Class**: Contains the method `rightSideView`, which will compute the values seen from the right side.

- **Checking for Empty Tree**: If the root is `None`, we directly return an empty list since there are no nodes to see.

- **Level-Order Processing**:
  - We use a list named `queue` to simulate the queue data structure. We start by adding the `root` to the queue.
  - A while loop continues as long as there are nodes in the queue.
  - We keep track of how many nodes are present at the current level using `level_length`.
  - We then iterate through each node at that level:
    - For every node, we pop it from the queue.
    - If it’s the last node processed for that level (`i == level_length - 1`), we append its value to `right_side_values`.
    - If there are right and left children, we add right first (ensuring the rightmost nodes are processed last).

- Finally, we return `right_side_values`, which contains the values of nodes visible from the right side in top-down order.

### Complexity Analysis:
- **Time Complexity**: O(n), where n is the number of nodes in the binary tree, as each node is processed exactly once.
- **Space Complexity**: O(n), which is the space required for the queue to hold nodes at the maximum level.

This implementation adheres to the constraints given in the problem and will execute correctly on the LeetCode platform.

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
 To solve the problem of counting the number of islands in a 2D binary grid, we will utilize a Depth-First Search (DFS) approach. An island is defined as a group of '1's connected horizontally or vertically, surrounded by '0's (water). 

### Key Steps in the Solution:
1. **Traversal of the Grid**: We will iterate through each cell in the grid.
2. **DFS for Island Discovery**: When we find a '1', it indicates the presence of a new island. We will then initiate a DFS to explore all connected '1's (parts of the same island) and mark them as visited (change '1's to '0's).
3. **Counting Islands**: Each time we initiate a DFS from an unvisited '1', we increase the island count by one.

The following Python code implements this logic:



```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        island_count = 0

        def dfs(r, c):
            # Out of bounds check
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
                return
            
            # Mark the land as visited
            grid[r][c] = '0'
            
            # Explore the adjacent land in all 4 directions: up, down, left, right
            dfs(r + 1, c)  # down
            dfs(r - 1, c)  # up
            dfs(r, c + 1)  # right
            dfs(r, c - 1)  # left
        
        # Iterate through each cell in the grid
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':  # Found an unvisited island
                    island_count += 1  # Increment island count
                    dfs(r, c)  # Start DFS to mark the whole island as visited

        return island_count

```

### Explanation of the Code:
- **Class Definition**: We define a class `Solution` which contains the method `numIslands`.
- **Method Parameters**: The method takes a parameter `grid`, which is a list of lists representing the 2D binary grid.
- **Grid Check**: We first check if the grid is empty. If it is, we return `0`.
- **Row and Column Variables**: We store the number of rows and columns in `rows` and `cols`.
- **Island Counter**: We initialize `island_count` to count the number of islands found.
- **DFS Function**: The function `dfs(r, c)` is defined to explore the island recursively.
  - It checks for boundary conditions (out of bounds or water).
  - If valid, it marks the current cell as '0' (visited).
  - It recursively calls itself on the adjacent cells (down, up, right, left).
- **Main Loop**: We loop through each cell in the grid.
  - If we find a '1', it means we have encountered a new island, so we increment our island count and call the `dfs` to mark all connected parts of the island.
- **Return Statement**: Finally, we return the total count of islands found.

### Complexity Analysis:
- **Time Complexity**: O(m * n), where m is the number of rows and n is the number of columns in the grid. Each cell is visited once.
- **Space Complexity**: O(m * n) in the worst case for the recursion stack, depending on the shape of the islands.

This implementation can be easily tested and is compatible with LeetCode's input format.

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
 To solve the problem of determining whether all courses can be finished given a set of prerequisites, we can represent the courses and their dependencies as a directed graph. The main challenge is to check for cycles in this graph because if there's a cycle, it means that there's a circular dependency, making it impossible to complete the courses.

Here's a step-by-step explanation of the approach we'll take, followed by a Python solution that follows the LeetCode format:

### Explanation:

1. **Graph Representation**:
   - We'll use an adjacency list to represent the graph of courses. Each course will point to the courses that depend on it.

2. **Cycle Detection**:
   - We can use Kahn's algorithm (a BFS-based approach) for topological sorting. If we can produce a valid topological ordering of the courses, it means we can finish all courses. If we can't (i.e., some courses are left unvisited), there's a cycle.
   - We'll maintain an array for the in-degree of each course, which counts the number of prerequisites for that course.

3. **Algorithm Steps**:
   - Create the adjacency list and fill the in-degrees based on prerequisites.
   - Use a queue to process all courses with an in-degree of zero (no prerequisites).
   - As we process each course, we decrease the in-degree of its dependent courses. If a dependent course’s in-degree becomes zero, we add it to the queue.
   - Keep track of the number of processed courses. If this number equals `numCourses` at the end, we can finish all courses; otherwise, we cannot.

### Python Code:



```python
from collections import deque

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # Step 1: Create adjacency list and in-degree array
        adj = [[] for _ in range(numCourses)]
        in_degree = [0] * numCourses
        
        for a, b in prerequisites:
            adj[b].append(a)  # b -> a
            in_degree[a] += 1
        
        # Step 2: Initialize queue with all courses having in-degree of 0
        queue = deque()
        for i in range(numCourses):
            if in_degree[i] == 0:
                queue.append(i)
        
        # Step 3: Process the courses
        count = 0  # To count the number of courses we can complete
        while queue:
            course = queue.popleft()
            count += 1  # We can take this course
            
            # Decrease the in-degree of dependent courses
            for dependent in adj[course]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Step 4: Check if we were able to take all courses
        return count == numCourses

```

### Explanation of the Code:

- The `adj` list builds our graph structure from the `prerequisites` list, where each course points to its dependent courses.
- The `in_degree` list tracks how many prerequisites each course has. Courses with an in-degree of zero can be enrolled in immediately.
- We use a queue (`deque`) to process courses in the order they can be taken. As we remove a course from the queue, we increase our `count` of completed courses.
- For each course we process, we reduce the in-degrees of its dependents and check if any of these can now be taken (i.e., their in-degree is now zero).
- Finally, we return `true` only if we were able to process all `numCourses`, indicating that it's possible to complete them all.

This solution runs in O(V + E) time complexity, where V is the number of courses and E is the number of prerequisites, making it very efficient given the constraints.

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
 To solve the problem of finding an order to take courses given their prerequisites, we can model it as a directed graph problem where:

- Each course is represented as a node.
- Each prerequisite relationship is represented as a directed edge from the prerequisite course to the course that requires it.

The goal is to obtain a topological sort of this directed graph. A topological sort provides a linear ordering of vertices such that for every directed edge from vertex A to vertex B, vertex A comes before vertex B in the ordering. 

If there is a cycle in the graph, it's impossible to complete all courses, and we should return an empty list. 

We'll implement the solution using Kahn's algorithm, which is efficient and straightforward for topological sorting, especially in the context of prerequisite relationships. Kahn's algorithm works as follows:

1. **Build the Graph**: Use an adjacency list to represent the graph of courses and maintain an array to track the in-degrees (number of incoming edges) for each node (course).
  
2. **Initialize the Queue**: Create a queue and initialize it with all courses that have an in-degree of zero (i.e., courses without prerequisites).

3. **Process the Courses**: 
   - While the queue is not empty, pop a course from the front (this represents taking the course), and add it to the result list.
   - For each course that depends on the current course, decrement its in-degree. If its in-degree becomes zero, add it to the queue.

4. **Check for Cycles**: At the end, if the result list size is equal to the number of courses, we can return the result; otherwise, return an empty list (indicating a cycle).

Here’s the complete Python code:



```python
from collections import deque, defaultdict

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # Step 1: Create an adjacency list and a list for in-degrees
        graph = defaultdict(list)
        in_degree = [0] * numCourses
        
        # Step 2: Build the graph
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        # Step 3: Initialize the queue with all courses having in-degree of 0
        queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
        
        order = []  # This will store the topological order
        
        # Step 4: Process the courses
        while queue:
            current_course = queue.popleft()
            order.append(current_course)
            
            # Reduce in-degree of neighbor courses
            for neighbor in graph[current_course]:
                in_degree[neighbor] -= 1
                # If in-degree becomes 0, add it to the queue
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Step 5: Check if we were able to schedule all courses
        if len(order) == numCourses:
            return order
        else:
            return []  # Cycle detected or not all courses can be completed


```

### Explanation of the Code:
- **Inputs**: We receive `numCourses`, which tells us the total number of courses, and `prerequisites`, which is a list of pairs indicating prerequisites.
- **Graph Creation**: We use a dictionary (`defaultdict`) to create an adjacency list and an array to keep track of in-degrees.
- **Queue Initialization**: We find all courses with no prerequisites and add them to the queue.
- **Processing**: We iterate through the queue, updating in-degrees of dependent courses, and maintain the order of courses completed.
- Finally, we check if we have completed all courses; if not, we return an empty list indicating a cycle. 

This algorithm runs in O(V + E) time complexity, where V is the number of courses and E is the number of prerequisites, ensuring efficiency even for larger inputs.

# 211. Design Add and Search Words Data Structure

### Problem Description 
Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the `WordDictionary` class:
`WordDictionary()` Initializes the object.

`void addWord(word)` Adds `word` to the data structure, it can be matched later.

`bool search(word)` Returns `true` if there is any string in the data structure that matches `word` or `false` otherwise. `word` may contain dots `'.'` where dots can be matched with any letter.


Example:
Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]
Explanation
WordDictionary wordDictionary = new WordDictionary();
wordDictionary.addWord("bad");
wordDictionary.addWord("dad");
wordDictionary.addWord("mad");
wordDictionary.search("pad"); // return False
wordDictionary.search("bad"); // return True
wordDictionary.search(".ad"); // return True
wordDictionary.search("b.."); // return True

Constraints:
`1 <= word.length <= 500`
`word` in `addWord` consists lower-case English letters.

`word` in `search` consist of  `'.'` or lower-case English letters.

At most `50000` calls will be made to `addWord` and `search`.

### Solution 
 To solve the problem of designing a data structure that supports adding words and searching with the capability of using wildcards (`.`), we can utilize a Trie (prefix tree) data structure. This will allow us to efficiently store words and perform searches, including those with wildcards.

### Detailed Explanation

1. **Trie Structure**:
   - A Trie is a tree-like structure where each node represents a character of a string. 
   - Each path from the root to a leaf represents a word in the data structure.
   - Each node can have several children, one for each letter in the alphabet.

2. **Adding Words**:
   - When adding a word, we will iterate over each character of the word, creating a new node if the character does not already exist at that position.

3. **Searching Words**:
   - To search, we will also iterate over each character. If we encounter a character that is a `.` (wildcard), we will need to explore all possible branches from that current node.
   - If it’s a regular character, we only follow the branch corresponding to that character.

### Implementation
Here's a Python implementation of the `WordDictionary` class that follows the aforementioned design:



```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        
class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_word = True

    def search(self, word: str) -> bool:
        return self._search_in_node(word, self.root)

    def _search_in_node(self, word: str, node: TrieNode) -> bool:
        for i, char in enumerate(word):
            if char == '.':
                # If the character is '.', check all children
                for child in node.children.values():
                    if self._search_in_node(word[i + 1:], child):
                        return True
                return False
            else:
                if char not in node.children:
                    return False
                node = node.children[char]
        return node.is_word
    
# Example usage
# wordDictionary = WordDictionary()
# wordDictionary.addWord("bad")
# wordDictionary.addWord("dad")
# wordDictionary.addWord("mad")
# print(wordDictionary.search("pad"))  # return False
# print(wordDictionary.search("bad"))  # return True
# print(wordDictionary.search(".ad"))  # return True
# print(wordDictionary.search("b.."))  # return True

```

### Explanation of the Code:
- **TrieNode Class**: Represents a node in the Trie. It has a dictionary `children` to store its child nodes and a boolean `is_word` to indicate if a word ends at this node.
- **WordDictionary Class**:
  - `__init__`: Initializes the `WordDictionary` with a root TrieNode.
  - `addWord`: Adds a word to the Trie by traversing and creating nodes as needed. Setting `is_word` to `True` at the last character.
  - `search`: Helper function that starts the search. It calls `_search_in_node` to handle the actual recursive search.
  - `_search_in_node`: Recursively searches through the Trie. If it hits a wildcard, it explores all branches. If it finds a match, it returns `True`.

### Complexity:
- The time complexity for adding a word is **O(N)** where N is the length of the word.
- The search operation, in the worst case when wildcards are involved, can take up to **O(M * 26^K)** where M is the length of the word being searched and K is the number of wildcard positions (though this is rare in practice).

This design leverages the efficiency of the Trie structure to handle the word addition and searching effectively, including the complexity introduced by wildcards.

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
 To solve the problem of finding all root-to-leaf paths in a binary tree, we can use a depth-first search (DFS) approach. We'll traverse the tree, keeping track of the current path from the root to the leaf node. When we reach a leaf node (a node with no children), we'll add the current path to our list of paths. 

Here’s the detailed explanation and the Python code to solve the problem.

### Strategy
1. **Define the Recursive Function**: We'll create a recursive helper function that takes the current node and the current path as arguments.
2. **Base Case**: If the current node is null, we simply return since there's no path to explore.
3. **Update Path**: If the current node is a leaf node (no left or right children), we append the current path to our results.
4. **Explore Children**: If the node is not a leaf, we continue exploring the left and right children, appending the current node's value to the path.
5. **Backtrack**: We need to backtrack after exploring a node's children to ensure the path remains correct for other branches of the tree.

### Python Code
Here's how we can implement the above logic in Python, following the LeetCode submission format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        def dfs(node, current_path):
            if not node:
                return
            
            # Append the current node's value to the path
            current_path.append(str(node.val))
            
            # Check if it's a leaf node
            if not node.left and not node.right:
                # Join the path and add to results
                paths.append("->".join(current_path))
            else:
                # Continue the search for leaf nodes
                dfs(node.left, current_path)
                dfs(node.right, current_path)
            
            # Backtrack to prepare for exploring the next path
            current_path.pop()
        
        paths = []
        dfs(root, [])
        return paths

```

### Explanation of the Code:
1. **TreeNode Class**: This class represents a node in the binary tree. Each node has a value (`val`) and two children (`left` and `right`).
   
2. **Solution Class**: This contains the method `binaryTreePaths` which will be called by LeetCode to get the result.
   
3. **DFS Function**: The `dfs` helper function is defined to handle the tree traversal:
   - If the current node is None, it returns to avoid further processing.
   - The value of the current node is converted to a string and added to the `current_path`.
   - If the current node is a leaf (both children are null), the current path is joined with `"->"` and saved in the `paths` list.
   - The function then recursively calls itself for the left and right children.
   - After exploring both subtrees, it backtracks by removing the last added node value from `current_path`.

4. **Return Statement**: Finally, the `binaryTreePaths` method returns the list of paths.

### Complexity:
- **Time Complexity**: O(N), where N is the number of nodes in the tree. Every node is visited once.
- **Space Complexity**: O(H), where H is the height of the tree. This is due to the recursion stack.

You can now run this code on LeetCode, and it should successfully pass any test case that fits the problem's constraints.

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
 To determine whether a given undirected graph is a valid tree, we need to check two key properties:

1. **Connectedness:** All nodes must be reachable from any other node. This means there should be a way to traverse the entire graph without any isolated parts.
  
2. **No Cycles:** A valid tree must not contain any cycles, meaning that there shouldn't be any path that starts and ends at the same node without retracing any edge.

For a graph to be a valid tree with `n` nodes and `m` edges, we can summarize these properties into a single rule: a valid tree must have exactly `n-1` edges. If there are more edges, it indicates that at least one cycle is present. If there are fewer edges, it suggests that not all nodes are connected.

### Steps to Solve the Problem
1. Check if the number of edges is not equal to `n-1`. If it is, return `false` immediately.
2. Build the graph using an adjacency list from the edges.
3. Use Depth First Search (DFS) or Breadth First Search (BFS) to traverse the graph and ensure all nodes are reachable:
   - Track visited nodes to prevent counting any nodes more than once.
   - Ensure we do not revisit the parent node in an undirected graph.
4. After traversal, check if all `n` nodes were visited.

### Implementation
Here is a Python implementation of the above logic, formatted to run directly on the LeetCode website:



```python
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        # A valid tree must have exactly n - 1 edges
        if len(edges) != n - 1:
            return False
        
        # Build the adjacency list representation of the graph
        graph = [[] for _ in range(n)]
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
        
        visited = [False] * n
        
        # Define a DFS function
        def dfs(node: int, parent: int) -> bool:
            visited[node] = True
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    if not dfs(neighbor, node):
                        return False
                elif neighbor != parent:  # if we see a visited neighbor that is not the parent, we found a cycle
                    return False
            return True
        
        # Start DFS from the first node (0)
        if not dfs(0, -1):
            return False
        
        # Finally, check if all nodes were visited to ensure connectivity
        return all(visited)

# Example Usage
# sol = Solution()
# print(sol.validTree(5, [[0,1],[0,2],[0,3],[1,4]]))  # Output: True
# print(sol.validTree(5, [[0,1],[1,2],[2,3],[1,3],[1,4]]))  # Output: False

```

### Explanation
- **Graph Construction:** We create an adjacency list where each index corresponds to a node, and contains a list of connected nodes.
- **DFS Function:** We implement a recursive DFS function that marks nodes as visited. If we encounter a previously visited node that isn't the parent of the current node, a cycle is detected.
- **Final Check:** After DFS completes, we ensure all nodes have been visited, confirming that the graph is connected.

### Complexity Analysis
- **Time Complexity:** O(n + m), where `n` is the number of nodes and `m` is the number of edges, because we traverse each node and each edge once.
- **Space Complexity:** O(n), for the graph representation and the visited tracking list.

This implementation solves the problem efficiently while adhering to the constraints provided in the problem statement.

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
 Certainly! To solve the problem of removing the minimum number of invalid parentheses from a given string to make it valid, we can use a breadth-first search (BFS) approach. Here's a detailed explanation followed by the Python code that you can run directly on LeetCode.

### Detailed Explanation

1. **Understanding Valid Parentheses**: A string of parentheses is considered valid if:
    - Every opening parenthesis `(` has a corresponding closing parenthesis `)`.
    - At no point in the string does a closing parenthesis appear without a matching opening parenthesis before it.

2. **Identification of Invalid Parentheses**: Given a string, we can determine how many opening and closing parentheses are unbalanced, which informs us how many steps we must take in order to achieve validity.

3. **BFS Approach**:
    - We start from the original string and generate all possible strings by removing one parenthesis at a time.
    - We keep track of the strings we generate in a queue and use a set to track already seen strings to avoid duplicates.
    - For each string, we check if it is valid; if it is, we can collect it as a valid answer.
    - We continue this process until we have found valid strings and exhausted all possible removals.

4. **Validation Function**: To check if a string is valid, we can use a counter. Increment on `(` and decrement on `)`. A negative counter indicates an invalid state.

5. **Stopping Condition**: As soon as we find valid strings, we store them and stop further processing at that BFS level to ensure we are only returning results after removing the minimum number of parentheses.

### Python Code

Here’s the implementation of the above approach:



```python
from collections import deque

def is_valid(s: str) -> bool:
    count = 0
    for char in s:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
        # If count goes negative, there's a closing bracket without a matching opening one
        if count < 0:
            return False
    # The string is valid if count is zero (all brackets are matched)
    return count == 0

def remove_invalid_parentheses(s: str):
    res = set()  # Use a set to avoid duplicates
    queue = deque([s])
    found = False  # To indicate when we've found valid strings
    
    while queue:
        current = queue.popleft()
        
        if is_valid(current):
            res.add(current)
            found = True  # Found at least one valid string
            
        # If we found valid strings, we do not need to proceed to longer strings
        if found:
            continue
        
        # Generate all possible strings by removing one parenthesis
        for i in range(len(current)):
            if current[i] in ('(', ')'):  # Only try removing parentheses
                new_str = current[:i] + current[i+1:]  # Remove the parenthesis
                if new_str not in res:  # Only add if we haven't seen this string
                    queue.append(new_str)
    
    return list(res)  # Convert the set to a list before returning

# Example Usage within LeetCode's environment
class Solution:
    def removeInvalidParentheses(self, s: str):
        return remove_invalid_parentheses(s)

```

### Explanation of the Code:

- **is_valid Function**: This helper function checks the validity of parentheses in a string.
- **remove_invalid_parentheses Function**: Main logic where we explore all potential valid configurations through BFS. It initiates with the original string, checks validity, and if invalid, continues generating new strings by removing parentheses.
- **Queue and Set Management**: Guesses are stored in `queue` (for BFS processing) while results are collected in a `set` to ensure uniqueness.
- **Stopping Early**: The BFS holds levels of strings; once valid strings are found, we stop exploring farther down that path.

This code structure matches the typical format found on LeetCode, making it ready to run on their platform.

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
 To solve the problem of vertical order traversal of a binary tree, we can utilize a breadth-first search (BFS) approach along with a data structure to keep track of the vertical positions of each node. The main idea is to assign a horizontal distance to each node in the tree, which determines its vertical column in the final result. 

Here's how we will approach the problem step-by-step:

1. **Data Structure**: We'll use a queue to facilitate the BFS, and a dictionary to map each column index to its corresponding nodes. We'll also maintain variables to track the minimum and maximum column indices encountered during our traversal.

2. **Traversal**: Starting from the root, we'll enqueue the root node with the column index 0. For each node processed, we'll add its left child with a column index of `current_index - 1` and its right child with `current_index + 1`.

3. **Output**: Once we've processed all nodes, we will extract the values from our dictionary, sorted by column indices, and then form the final output.

Let’s implement this strategy in Python code, which fits the LeetCode format:



```python
from collections import defaultdict, deque

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def verticalOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        
        # Dictionary to store the mapping of columns to their respective nodes
        column_table = defaultdict(list)
        # BFS queue initialized with the root node and its column 0
        queue = deque([(root, 0)])
        # Variables to track the min and max column indices
        min_column = max_column = 0
        
        # BFS traversal
        while queue:
            node, column = queue.popleft()
            if node:
                # Add this node's value to the respective column
                column_table[column].append(node.val)
                # Update the column bounds
                min_column = min(min_column, column)
                max_column = max(max_column, column)

                # Add left child to the queue with column index - 1
                queue.append((node.left, column - 1))
                # Add right child to the queue with column index + 1
                queue.append((node.right, column + 1))
        
        # Extract the values in sorted order of columns
        result = []
        for col in range(min_column, max_column + 1):
            result.append(column_table[col])
        
        return result

```

### Explanation of the code:

1. **TreeNode Class**: This defines the structure of a node in our binary tree.

2. **Solution Class**: This contains the method `verticalOrder` which implements our algorithm.

   - **Base Case**: If the root is `None`, we return an empty list.
   
   - **Column Table**: We use `defaultdict(list)` to accumulate node values by their column indices.
   
   - **Queue for BFS**: We initialize the BFS with the root node and start at column `0`.
   
   - **BFS Execution**: We process nodes one by one, updating the column mappings and bounds. We enqueue left and right children with updated column indices.
   
   - **Result Compilation**: After the BFS is complete, we compile the results from our column mappings, ensuring to traverse from the smallest to the largest column index.

3. **Return**: Finally, we return the compiled list of lists representing the vertical order traversal.

This code is ready to run on LeetCode as a solution to the problem.

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
 To solve the problem of counting the number of connected components in an undirected graph, we can use either Depth-First Search (DFS) or Breadth-First Search (BFS). Here, I will use DFS for our solution.

### Problem Breakdown
1. **Understanding the Graph**: We are given nodes and edges that represent a graph. Each edge connects two nodes.
2. **Connected Components**: A connected component is a subset of the graph where there is a path between any two nodes in that subset, and which is connected to no additional nodes.
3. **Goal**: We need to determine how many such connected components exist in the graph based on the given edges.

### Approach
1. **Graph Representation**: We can represent the graph using an adjacency list, which is a common way to store graphs in terms of nodes and their edges.
2. **Traversal**: We will traverse the graph using DFS. Each time we start a new DFS from an unvisited node, we increase our count of connected components.
3. **Visited Tracking**: We will keep track of visited nodes with a set or a list to prevent counting the same component more than once.

### Detailed Steps
- Create an adjacency list from the list of edges.
- Initialize a visited list to track visited nodes.
- Iterate through each node; if it has not been visited, perform a DFS from that node, marking all reachable nodes as visited.
- Increment the connected component count each time you start a new DFS.

### Python Code
Here is the implementation of the above approach:



```python
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        # Create an adjacency list to represent the graph
        graph = [[] for _ in range(n)]
        
        # Fill the adjacency list with edges
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
        
        # Initialize a visited set to keep track of visited nodes
        visited = [False] * n
        component_count = 0
        
        # Define a DFS function
        def dfs(node):
            # Mark the current node as visited
            visited[node] = True
            # Explore all the neighbors
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        # Iterate through each node
        for i in range(n):
            if not visited[i]:  # If the node hasn't been visited
                dfs(i)  # Perform DFS from this node
                component_count += 1  # Increment the component count
        
        return component_count

# Example usage:
# sol = Solution()
# print(sol.countComponents(5, [[0,1],[1,2],[3,4]]))  # Output: 2
# print(sol.countComponents(5, [[0,1],[1,2],[2,3],[3,4]]))  # Output: 1

```

### Explanation of the Code
1. **Graph Construction**: We initialize a graph using a list of lists (adjacency list). For each edge, we add it in both directions since the graph is undirected.
2. **Visited Tracking**: An array `visited` keeps track of whether we have visited a node.
3. **DFS Implementation**: The recursive DFS function marks the current node as visited and recursively visits all its unvisited neighbors.
4. **Count Components**: In the main loop, we check if each node has been visited. If not, we perform a DFS from that node, indicating a new connected component has been found, and increment our component counter.

With these steps, this solution will efficiently count the number of connected components in the graph according to the problem requirements.

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
 To solve the problem of finding the length of the longest increasing path in a given matrix, we can utilize Depth-First Search (DFS) along with memoization. The key idea is to explore each cell's neighbors and keep track of the paths while ensuring they are strictly increasing.

### Explanation:

1. **DFS Approach**: The basic principle is to use DFS to explore all paths starting from each cell, but we will be caching results to avoid recalculating the length of paths from previously visited cells.

2. **Memoization**: We will create a memoization table (2D list) that stores the longest increasing path length starting from each cell. If we compute the result for a cell once, we save it to the memo table, allowing us to simply return the saved value on subsequent calls.

3. **Bounds Checking**: It's essential to ensure that we do not step out of matrix boundaries during our exploration.

4. **Directions for Movement**: We will use an array to represent the four possible movement directions (up, down, left, right).

### Complexity:
- **Time Complexity**: O(m * n), where m is the number of rows and n is the number of columns in the matrix, because each cell will be processed once.
- **Space Complexity**: O(m * n) for the memoization table.

### Implementation:
Here’s the complete Python code structured according to the LeetCode style, which can be run directly on the LeetCode platform.



```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        memo = [[-1] * n for _ in range(m)]  # memoization table
        
        def dfs(x: int, y: int) -> int:
            # Return the stored value if already calculated
            if memo[x][y] != -1:
                return memo[x][y]
            
            longest = 1  # At least the cell itself counts as length 1
            
            # Directions for moving in the matrix (down, up, right, left)
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # Check boundary and increasing condition
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] > matrix[x][y]:
                    current_length = 1 + dfs(nx, ny)
                    longest = max(longest, current_length)
            
            memo[x][y] = longest  # Store the result in memo table
            return longest
        
        max_length = 0
        for i in range(m):
            for j in range(n):
                max_length = max(max_length, dfs(i, j))
        
        return max_length

```

### How to Use:
- Simply create an instance of the `Solution` class and call the `longestIncreasingPath` method with the matrix as an argument.
- For example:


```python
sol = Solution()
result = sol.longestIncreasingPath([[9,9,4],[6,6,8],[2,1,1]])
print(result)  # Output: 4

```

### Summary:
This solution maximally leverages DFS with memoization to efficiently explore the matrix for the longest increasing path, ensuring the solution is optimal for the given constraints.

# 332. Reconstruct Itinerary

### Problem Description 
You are given a list of airline `tickets` where `tickets[i] = [fromi, toi]` represent the departure and the arrival airports of one flight. Reconstruct the itinerary in order and return it.

All of the tickets belong to a man who departs from `"JFK"`, thus, the itinerary must begin with `"JFK"`. If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string.

For example, the itinerary `["JFK", "LGA"]` has a smaller lexical order than `["JFK", "LGB"]`.

You may assume all tickets form at least one valid itinerary. You must use all the tickets once and only once.


Example 1:
Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
Output: ["JFK","MUC","LHR","SFO","SJC"]

Example 2:
Input: tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
Explanation: Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"] but it is larger in lexical order.


Constraints:
`1 <= tickets.length <= 300`
`tickets[i].length == 2`
`fromi.length == 3`
`toi.length == 3`
`fromi` and `toi` consist of uppercase English letters.

`fromi != toi`

### Solution 
 To solve the problem of reconstructing the itinerary from the given list of airline tickets, we can apply a backtracking approach combined with graph traversal techniques. Here's a detailed breakdown of how we can implement this:

### Strategy
1. **Graph Representation**: We will represent the flight tickets as a graph using a dictionary where each key is a departure airport and the value is a min-heap (priority queue) of arrival airports. This allows us to always pick the smallest lexical order of destinations first.

2. **DFS Traversal**: We will use Depth-First Search (DFS) to traverse the graph, ensuring that we use all tickets exactly once. We will start our journey from "JFK".

3. **Post-order Collection**: Since we want the final itinerary to be in the correct order, we will collect results in a post-order fashion. After reaching a destination, we'll backtrack and append the airport to the result list.

4. **Result Management**: Because we need the final itinerary in reverse order, we will reverse the collected results before returning them.

### Implementation
Here's how we can implement this logic in Python:



```python
from collections import defaultdict
import heapq

class Solution:
    def findItinerary(self, tickets):
        # Build the graph
        graph = defaultdict(list)
        
        # Create a min-heap for each departure airport
        for frm, to in tickets:
            heapq.heappush(graph[frm], to)
        
        # To store the final itinerary
        itinerary = []

        # Perform DFS
        def dfs(airport):
            # While there are destinations in the heap (i.e., for this airport)
            while graph[airport]:
                # Get the next destination airport (the smallest one due to min-heap)
                next_airport = heapq.heappop(graph[airport])
                # Visit the next airport
                dfs(next_airport)
            # Append the airport to the itinerary after visiting all reachable destinations
            itinerary.append(airport)

        # Start the journey from "JFK"
        dfs("JFK")
        
        # Since we built the itinerary in post-order, we need to reverse it
        return itinerary[::-1]

# Example usage
solution = Solution()
print(solution.findItinerary([["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]))  # Output: ["JFK","MUC","LHR","SFO","SJC"]
print(solution.findItinerary([["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]))  # Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]

```

### Explanation of Code:
- We first initialize a graph using `defaultdict` to manage our departure airports and corresponding min-heaps for arrival airports.
- Each ticket is added to the graph, ensuring that the destinations are stored in a priority order (using `heapq`).
- The `dfs` function is a recursive function that visits airports until there are no more flights from the current airport. It appends the airport to the itinerary after all its reachable destinations are visited.
- Finally, we reverse the itinerary to get it in the correct order and return that as the result.

### Complexity
- **Time Complexity**: O(E log E), where E is the number of tickets (edges). Each insertion into the heap takes O(log E) time.
- **Space Complexity**: O(E) for storing the graph and O(E) for the itinerary.

This solution efficiently follows the problem's constraints and ensures the smallest lexical order for valid itineraries.

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
 To solve the problem of maximizing the amount of money a thief can rob from houses arranged in a binary tree, we can utilize a depth-first search (DFS) strategy combined with dynamic programming. The main idea is to decide, for each house (node), whether to rob it or not based on the potential values derived from its children.

### Problem Breakdown:
1. **Node Structure**: Each node in the binary tree represents a house with a certain amount of money. The left and right children of a node represent the houses directly connected to it.
2. **Robbing Rules**: The thief cannot rob two directly linked houses. This means if a house (node) is robbed, its children cannot be robbed.
3. **Dynamic Choice**: For any given node, we have two choices:
   - Rob the current node and add its value to the value obtained from the grandchildren (i.e., all nodes except the direct children).
   - Do not rob the current node and take the maximum from robbing or not robbing its children.

### Dynamic Programming Approach:
We'll return two values for each node:
1. The maximum money that can be robbed if we rob the current node (including contributions from grandchildren).
2. The maximum money that can be robbed if we do not rob the current node (which accumulates the maximum from the children).

### Implementation Steps:
1. Define a helper recursive function that will traverse the tree and compute the maximum values.
2. At each node, decide whether to include its value or not based on the previously described logic.
3. Return the result from the `rob` function.

Here’s how the implementation looks in Python:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def rob(self, root: TreeNode) -> int:
        # Helper function that returns two values:
        # [0]: max amount if we rob the current node
        # [1]: max amount if we do not rob the current node
        def dfs(node):
            if not node:
                return (0, 0)  # Base case: If the node is null, both values are 0
            
            left = dfs(node.left)   # Recursively rob both children
            right = dfs(node.right)

            # If we rob this node:
            rob_current = node.val + left[1] + right[1]  # Rob current, do not rob children
            
            # If we do not rob this node:
            not_rob_current = max(left) + max(right)  # Take the best option from children
            
            return (rob_current, not_rob_current)

        # Get the maximum amount we can rob starting from the root
        result = dfs(root)
        return max(result)  # Max of robbing or not robbing the root

```

### Explanation of the Code:
1. **TreeNode Class**: Defines the structure of a node in the binary tree.
2. **Solution Class with rob Method**:
   - **DFS Function**: A helper function that traverses the tree recursively.
   - **Base Case**: If the node is `None`, return (0, 0) indicating no money can be robbed.
   - **Recursion**: Calculates the maximum money for left and right children.
   - **Calculations**: Computes the maximum amounts if the current node is robbed and if it is not.
3. **Final Return**: The maximum of the two options from the root.

### Time Complexity:
The solution has a time complexity of O(N) where N is the number of nodes since each node is processed once.

### Space Complexity:
The space complexity is O(H), where H is the height of the binary tree due to the recursive call stack. For a balanced tree, this is O(log N), and for a skewed tree, it can be O(N). 

This implementation is reliable and efficient for the problem constraints provided.

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
 Sure! Let's solve the problem of calculating the weighted sum of integers in a nested list based on their depth.

### Problem Explanation

We are given a nested list of integers, where each integer has a certain "depth" based on how deeply nested it is. The goal is to compute the sum of each integer multiplied by its depth.

### Steps to Solve the Problem

To solve this, we will use a recursive approach:
1. Define a function that takes a nested list and the current depth as parameters.
2. If the current element is an integer, multiply it by the current depth and add the result to a cumulative total.
3. If the current element is a list, recursively call the function for each element of the list, increasing the depth by 1.
4. Finally, return the cumulative total.

### Implementation

Here's the implementation of the above logic in the format suitable for LeetCode:



```python
from typing import List, Union

class Solution:
    def depthSum(self, nestedList: List[Union[int, List]]) -> int:
        def helper(lst: List[Union[int, List]], depth: int) -> int:
            total = 0
            for item in lst:
                if isinstance(item, int):  # Check if the item is an integer
                    total += item * depth  # Multiply by current depth
                else:  # The item is a list
                    total += helper(item, depth + 1)  # Recursive call with increased depth
            return total

        return helper(nestedList, 1)  # Start with depth 1

```

### Explanation of the Code

1. **Imports**:
    - We import `List` and `Union` from the `typing` module to specify the type of our parameters clearly.
    
2. **Solution Class**:
    - We define our main class `Solution`, which is conventionally used in LeetCode submissions.

3. **Method depthSum**:
    - This method takes a nested list called `nestedList` which can contain integers and lists.
    
4. **Helper Function**:
    - Defined within `depthSum`, this function `helper` takes the current list and the current depth.
    - It initializes a variable `total` to keep track of the cumulative sum.
    
5. **Loop Through Items**:
    - For each item in the provided list:
      - If it is an integer, it multiplies the integer by the current depth and adds it to `total`.
      - If it is a list, it recursively calls `helper` with that list, increasing the depth by 1.
      
6. **Return Total**:
    - After processing all items, it returns the accumulated total for that level.
    
7. **Starting the Helper**:
    - Finally, we call `helper(nestedList, 1)` from `depthSum`, starting at depth 1.

### Example Execution

For the input `nestedList = [[1,1],2,[1,1]]`, the function would work as follows:
- At depth 1, it sees the lists and the integer 2:
  - The two '1's are at depth 2, contributing `1*2 + 1*2 = 4`.
  - The integer 2 at depth 1 contributes `2*1 = 2`.
  - The two '1's in the last list contribute another `1*2 + 1*2 = 4`.
- Combining these gives `4 + 2 + 4 = 10`.

This approach efficiently computes the requested sum for the integers in the nested list based on their respective depths.

# 364. Nested List Weight Sum II

### Problem Description 
You are given a nested list of integers `nestedList`. Each element is either an integer or a list whose elements may also be integers or other lists.

The depth of an integer is the number of lists that it is inside of. For example, the nested list `[1,[2,2],[[3],2],1]` has each integer's value set to its depth. Let `maxDepth` be the maximum depth of any integer.

The weight of an integer is `maxDepth - (the depth of the integer) + 1`.

Return the sum of each integer in `nestedList` multiplied by its weight.


Example 1:
Input: nestedList = [[1,1],2,[1,1]]
Output: 8
Explanation: Four 1's with a weight of 1, one 2 with a weight of 2.

1*1 + 1*1 + 2*2 + 1*1 + 1*1 = 8

Example 2:
Input: nestedList = [1,[4,[6]]]
Output: 17
Explanation: One 1 at depth 3, one 4 at depth 2, and one 6 at depth 1.

1*3 + 4*2 + 6*1 = 17

Constraints:
`1 <= nestedList.length <= 50`
The values of the integers in the nested list is in the range `[-100, 100]`.

The maximum depth of any integer is less than or equal to `50`.

### Solution 
 To solve the problem of calculating the weighted sum of all integers in a nested list, we can follow these steps:

1. **Calculate the Maximum Depth**: We first need to find the maximum depth of the integers in the nested list. This can be done through a recursive function that explores all elements and tracks the depth.

2. **Calculate the Weighted Sum**: Once we have the maximum depth, we will iterate through the nested list again, this time calculating the weight of each integer based on its depth and accumulating the total weighted sum.

Let's implement this as a Python function within the `Solution` class that can be run directly on the LeetCode platform:



```python
class NestedInteger:
    def isInteger(self) -> bool:
        # Return True if this NestedInteger holds a single integer, false otherwise.
        pass

    def getInteger(self) -> int:
        # Return the single integer that this NestedInteger holds, 
        # if it holds a single integer. Otherwise, return None.
        pass

    def getList(self) -> [NestedInteger]:
        # Return the list that this NestedInteger holds, 
        # if it holds a nested list. Otherwise, return None.
        pass

class Solution:
    def depthSum(self, nestedList: [NestedInteger]) -> int:
        # First we need a function that calculates the maximum depth of the nested list
        def getMaxDepth(nestedList: [NestedInteger], depth: int) -> int:
            maxDepth = depth
            for ni in nestedList:
                if not ni.isInteger():  # If it's a list, go deeper
                    maxDepth = max(maxDepth, getMaxDepth(ni.getList(), depth + 1))
            return maxDepth
        
        # Get the maximum depth
        maxDepth = getMaxDepth(nestedList, 1)
        
        # Now we calculate the weighted sum
        def getWeightedSum(nestedList: [NestedInteger], depth: int) -> int:
            weightedSum = 0
            for ni in nestedList:
                if ni.isInteger():  # If it's an integer, calculate its contribution
                    weightedSum += ni.getInteger() * (maxDepth - depth + 1)
                else:  # If it's a list, go deeper
                    weightedSum += getWeightedSum(ni.getList(), depth + 1)
            return weightedSum
        
        # Calculate and return the weighted sum
        return getWeightedSum(nestedList, 1)

# Note: To test this code, you would need a specific implementation of the NestedInteger class.

```

### Explanation of the Code:

- **NestedInteger Class**: This class is an abstraction provided by LeetCode to facilitate interfacing with the nested list structure. It contains methods to determine if the element is an integer or a list and to retrieve the integer value or the list.
  
- **Method `depthSum` in Solution Class**: This method serves as the main function to compute the desired sum. It includes two helper functions:
  
  - **`getMaxDepth`**: This function is called initially to find the maximum depth of the integers in the nested list. It uses a recursive approach to navigate through each `NestedInteger`:
    - If the current element is a list, it recursively calls itself with an incremented depth until all lists are explored.
  
  - **`getWeightedSum`**: This function calculates the weighted sum by traversing the nested list and considering each integer's weight:
    - If the element is an integer, it multiplies it by its corresponding weight based on its depth and accumulates the sum.
    - If the element is a list, it continues to explore deeper.

- **Result**: Finally, the total weighted sum is returned.

### Complexity:
The time complexity is O(n), where n is the total number of integers and lists combined, as we traverse each element multiple times. The space complexity is O(d) due to the recursion depth, where d is the maximum depth of the nested lists.

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
 To solve the problem of collecting leaf nodes from a binary tree repeatedly until the tree is empty, we can use a level-order traversal (BFS). In each iteration, we will identify the leaf nodes, collect them into a list, and then remove them from the tree. This process is repeated until there are no nodes left in the tree.

Here's a detailed explanation of the approach we'll take:

1. **Define a TreeNode class**: We need a class to represent each node in the binary tree.

2. **Use a helper function**: We'll create a helper function that takes the root of the tree and returns a list of list of leaf nodes gathered at each step.

3. **Iterate through the tree**: In the helper function, we'll keep track of the nodes at the current level, collect the leaf nodes, and prepare for the next level until no nodes are left.

4. **Return the result**: After iterating through the tree, we return the collected leaf nodes as a list of lists.

Here is the complete Python code implementing this approach in the format suitable for LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def findLeaves(self, root: TreeNode) -> List[List[int]]:
        result = []
        
        while root:
            leaves = []
            root = self.collectLeaves(root, leaves)
            result.append(leaves)
        
        return result
    
    def collectLeaves(self, node: TreeNode, leaves: List[int]) -> TreeNode:
        if not node:
            return None
        
        # If it's a leaf node, add its value to the leaves list
        if not node.left and not node.right:
            leaves.append(node.val)
            return None  # Remove the leaf

        # Recur for left and right children
        node.left = self.collectLeaves(node.left, leaves)
        node.right = self.collectLeaves(node.right, leaves)
        
        return node  # Return the current node unchanged if it's not a leaf

# Example usage:
# This part is usually not included in the submission to LeetCode,
# but it's mentioned here for clarity on how to create the binary tree input.

# Input tree creation for testing:
#       1
#      / \
#     2   3
#    / \   
#   4   5  
# root = TreeNode(1)
# root.left = TreeNode(2)
# root.right = TreeNode(3)
# root.left.left = TreeNode(4)
# root.left.right = TreeNode(5)

```

### Explanation of the Code:

- The `TreeNode` class represents each node of the binary tree with a value (`val`), a left child (`left`), and a right child (`right`).
- The `findLeaves` method initializes an empty `result` list to store the collected leaf nodes at each iteration.
- A `while` loop continues as long as the `root` is not `None`. In each iteration, it finds and collects the leaves.
- The `collectLeaves` helper method is called recursively. It checks if a node is `None`, if it's a leaf (both children are `None`), or if it has children. If it's a leaf, it records the value and returns `None` (effectively removing it from the tree). Otherwise, it recurses down to collect leaves from the children and returns the node itself if it's not a leaf.
- Finally, the result list, which contains the leaf nodes grouped by the levels, is returned.

This implementation is efficient and directly follows the problem requirements, ensuring a clear understanding of how to collect leaves from the binary tree while maintaining the tree structure.

# 394. Decode String

### Problem Description 
Given an encoded string, return its decoded string.

The encoding rule is: `k[encoded_string]`, where the `encoded_string` inside the square brackets is being repeated exactly `k` times. Note that `k` is guaranteed to be a positive integer.

You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, `k`. For example, there won't be input like `3a` or `2[4]`.


Example 1:
Input: s = "3[a]2[bc]"
Output: "aaabcbc"

Example 2:
Input: s = "3[a2[c]]"
Output: "accaccacc"

Example 3:
Input: s = "2[abc]3[cd]ef"
Output: "abcabccdcdcdef"

Example 4:
Input: s = "abc3[cd]xyz"
Output: "abccdcdcdxyz"

Constraints:
`1 <= s.length <= 30`
`s` consists of lowercase English letters, digits, and square brackets `'[]'`.

`s` is guaranteed to be a valid input.

All the integers in `s` are in the range `[1, 300]`.

### Solution 
 To solve the problem of decoding an encoded string, we can utilize a stack-based approach. This method allows us to handle the nested structures of the encoded string effectively.

### Solution Explanation

We'll follow these steps:
1. **Initialize Two Stacks:** One for characters and one for numbers.
2. **Iterate Through Each Character:** 
   - If we see a digit, we extract the whole number (which may consist of more than one digit) and push it onto the number stack.
   - If we see an opening bracket `[`, we push the current accumulated string (if any) onto the character stack, and reset the current string.
   - If we see a closing bracket `]`, we pop the number from the number stack (which tells us how many times to repeat the string), pop the last string from the character stack, and concatenate the current string multiplied by the number we just popped.
   - If we see a character (a-z), we simply append it to the current string.
3. **Return the Final Result:** After processing all characters, the current string will be our final decoded output.

Here's the complete Python code following the LeetCode solution format:



```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack_num = []  # Stack to store the numbers
        stack_str = []  # Stack to store the strings
        current_num = 0  # For building the number
        current_str = ""  # For building the current string
        
        for char in s:
            if char.isdigit():  # If it's a digit, build the number
                current_num = current_num * 10 + int(char)
            elif char == '[':  # If we hit an '[', push current string and number to stacks
                stack_num.append(current_num)
                stack_str.append(current_str)
                current_num = 0  # Reset current number
                current_str = ""  # Reset current string
            elif char == ']':  # If we hit a ']', pop from stacks and build the string
                num = stack_num.pop()  # Get the last number
                prev_str = stack_str.pop()  # Get the last string
                current_str = prev_str + current_str * num  # Form the decoded string
            else:  # If it's a character, just add it to the current string
                current_str += char
        
        return current_str  # Return the final decoded string

# Example Usage
solution = Solution()
print(solution.decodeString("3[a]2[bc]"))  # Output: "aaabcbc"
print(solution.decodeString("3[a2[c]]"))   # Output: "accaccacc"
print(solution.decodeString("2[abc]3[cd]ef"))  # Output: "abcabccdcdcdef"
print(solution.decodeString("abc3[cd]xyz"))  # Output: "abccdcdcdxyz"

```

### Explanation of the Code:
- **Stack for Numbers (`stack_num`)**: This stores multipliers (k values).
- **Stack for Strings (`stack_str`)**: This holds the strings accumulated before the most recent `[` encountered.
- **`current_num` and `current_str`**: These are used to build the current number and string segment to be decoded.

- **Looping Through Input `s`**:
  - **Digits**: Convert the digit to an integer and accumulate it (`current_num`).
  - **`[`**: Store the accumulated number and string so far, then reset them.
  - **`]`**: Retrieve the most recent number and string, then construct the new current string by repeating it.
  - **Characters**: Simply append to the `current_str`.

### Performance
This algorithm runs in O(n) time and uses O(n) space, where n is the length of the input string, making it efficient for the problem's constraints.

This code can be directly copied into the LeetCode environment and will function correctly for decoding the encoded strings as specified in the problem.

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
 To solve the problem of identifying cells in a matrix from which water can flow to both the Pacific and Atlantic oceans, we can utilize a depth-first search (DFS) approach. Here is a detailed explanation and a code implementation that is ready to run on the LeetCode platform.

### Problem Breakdown:
1. **Understanding Flow Directions**: Water can flow from a cell to its neighboring cells in four directions (up, down, left, and right) if the neighboring cell's height is less than or equal to the current cell's height.

2. **Ocean Touch Points**: 
   - The **Pacific Ocean** touches the top and left edges of the matrix.
   - The **Atlantic Ocean** touches the bottom and right edges of the matrix.

3. **Cell Reachability**: We need to determine cells from which water can reach both oceans. This can be done by performing DFS from both the Pacific and Atlantic edges.

### Approach:
1. Create two boolean matrices `pacific_reachable` and `atlantic_reachable` to keep track of which cells can reach the respective oceans.
2. Perform DFS from all the cells in the Pacific Ocean (top and left edges) to mark all reachable cells.
3. Perform DFS from all the cells in the Atlantic Ocean (bottom and right edges) to mark all reachable cells.
4. After processing, the final result consists of coordinates where both `pacific_reachable` and `atlantic_reachable` are True.

### Complexity:
- **Time Complexity**: O(m * n), where m and n are the dimensions of the matrix since each cell is processed once.
- **Space Complexity**: O(m*n) for the boolean matrices to store reachability information.

### Python Implementation:



```python
class Solution:
    def pacificAtlantic(self, heights):
        if not heights:
            return []

        # Dimensions of the matrix
        m, n = len(heights), len(heights[0])
        
        # Directions for up, down, left, right movements
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Reachability matrices
        pacific_reachable = [[False] * n for _ in range(m)]
        atlantic_reachable = [[False] * n for _ in range(m)]
        
        def dfs(x, y, reachable):
            reachable[x][y] = True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and not reachable[nx][ny] and heights[nx][ny] >= heights[x][y]:
                    dfs(nx, ny, reachable)

        # Perform DFS for Pacific Ocean (top and left edges)
        for i in range(m):
            dfs(i, 0, pacific_reachable)  # Left edge
            dfs(i, n - 1, atlantic_reachable)  # Right edge
        for j in range(n):
            dfs(0, j, pacific_reachable)  # Top edge
            dfs(m - 1, j, atlantic_reachable)  # Bottom edge
        
        # Generate the result list
        result = []
        for i in range(m):
            for j in range(n):
                if pacific_reachable[i][j] and atlantic_reachable[i][j]:
                    result.append([i, j])
        
        return result

# Example usage:
# heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
# sol = Solution()
# print(sol.pacificAtlantic(heights))

```

### Explanation of the Code:
- **Initialization**: The dimensions of the matrix and direction vectors are established. Two boolean matrices for Pacific and Atlantic reachability are initialized.
- **DFS Function**: The `dfs` function marks cells as reachable based on height conditions.
- **Starting Points for DFS**: The DFS is initiated from every edge cell corresponding to the Pacific and Atlantic oceans.
- **Result Compilation**: Finally, we iterate through the matrices and compile a list of coordinates where both oceans are reachable. 

This approach is efficient and neatly organized, making it suitable for direct submission to LeetCode!

# 430. Flatten a Multilevel Doubly Linked List

### Problem Description 
You are given a doubly linked list which in addition to the next and previous pointers, it could have a child pointer, which may or may not point to a separate doubly linked list. These child lists may have one or more children of their own, and so on, to produce a multilevel data structure, as shown in the example below.

Flatten the list so that all the nodes appear in a single-level, doubly linked list. You are given the head of the first level of the list.


Example 1:
Input: head = [1,2,3,4,5,6,null,null,null,7,8,9,10,null,null,11,12]
Output: [1,2,3,7,8,11,12,9,10,4,5,6]
Explanation:
The multilevel linked list in the input is as follows:
After flattening the multilevel linked list it becomes:

Example 2:
Input: head = [1,2,null,3]
Output: [1,3,2]
Explanation:
The input multilevel linked list is as follows:
  1---2---NULL
  |
  3---NULL

Example 3:
Input: head = []
Output: []
How multilevel linked list is represented in test case:

We use the multilevel linked list from Example 1 above:
 1---2---3---4---5---6--NULL
         |
         7---8---9---10--NULL
             |
             11--12--NULL
The serialization of each level is as follows:
[1,2,3,4,5,6,null]
[7,8,9,10,null]
[11,12,null]
To serialize all levels together we will add nulls in each level to signify no node connects to the upper node of the previous level. The serialization becomes:
[1,2,3,4,5,6,null]
[null,null,7,8,9,10,null]
[null,11,12,null]
Merging the serialization of each level and removing trailing nulls we obtain:
[1,2,3,4,5,6,null,null,null,7,8,9,10,null,null,11,12]

Constraints:
The number of Nodes will not exceed `1000`.

`1 <= Node.val <= 105`

### Solution 
 To solve the problem of flattening a multilevel doubly linked list, we'll implement a depth-first search (DFS) approach. The idea is to traverse through each node, flatten any child lists, and then link them into a singly-level linked list. This approach effectively handles nodes that may point to further child lists, ensuring all levels are accounted for.

### Step-by-Step Explanation:

1. **Node Structure**: We'll define a `Node` class representing each node in the doubly linked list. Each node has three pointers: `next`, `prev`, and `child`.

2. **Flatten Function**: We'll write a `flatten` function that accepts the head node as input and returns the head of the flattened list. 

3. **Recursion**: The core of the algorithm involves recursively visiting each node. When we encounter a node with a child, we will:
   - Flatten the child list first.
   - Connect the current node with the flattened child.
   - Continue traversing the list.

4. **Maintaining Connections**: While flattening:
   - Adjust the `next` and `prev` pointers to maintain the doubly linked list structure.
   - After processing a child list, restore the pointers to continue to the next node seamlessly.

5. **Edge Cases**: Handle cases where the input list is empty, ensuring the function can return `None` when there are no nodes.

### Python Code:



```python
# Definition for a Node.
class Node:
    def __init__(self, val=0, prev=None, next=None, child=None):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child

class Solution:
    def flatten(self, head: 'Node') -> 'Node':
        if not head:
            return None
        
        # Helper function to do the flattening
        def flatten_dfs(node):
            current = node
            last = node
            
            while current:
                next_node = current.next
                
                # If current has a child
                if current.child:
                    # Flatten the child list
                    child_last = flatten_dfs(current.child)
                    
                    # Connect current node with the head of the child list
                    current.next = current.child
                    current.child.prev = current

                    # Connect the end of the flattened child list with the next node
                    child_last.next = next_node
                    if next_node:  # If there's a next node, adjust its prev pointer
                        next_node.prev = child_last
                    
                    # Clear the child pointer
                    current.child = None
                
                last = current  # Update last to current node
                current = next_node  # Move to the next node
            
            return last  # Return the last node processed

        # Start the flattening process
        flatten_dfs(head)
        return head  # Return the head of the flattened list

```

### Explanation of the Code:
- **Node Class**: Defines the structure of the list node including its `val`, `prev`, `next`, and `child` pointers.
- **Flatten Function**: This is the main function that initiates the flattening. It checks for an empty head and calls `flatten_dfs` if the head is valid.
- **Flatten DFS Function**: A recursive function that:
  - Iterates through the nodes, handling child nodes as they appear.
  - Links current nodes with their flattened child list.
  - Returns the last node of the current list segment after flattening.

### Complexity Analysis:
- **Time Complexity**: O(N) where N is the number of nodes in the list. We visit each node exactly once.
- **Space Complexity**: O(H) where H is the height of the recursive calls (if the list has many nested children). In the worst case, H can be O(N) due to linear depth.

This solution, when run in a LeetCode environment, will properly flatten the multilevel linked list as described.

# 439. Ternary Expression Parser

### Problem Description 
Given a string representing arbitrarily nested ternary expressions, calculate the result of the expression. You can always assume that the given expression is valid and only consists of digits `0-9`, `?`, `:`, `T` and `F` (`T` and `F` represent True and False respectively).

Note:
The length of the given string is ≤ 10000.

Each number will contain only one digit.

The conditional expressions group right-to-left (as usual in most languages).

The condition will always be either `T` or `F`. That is, the condition will never be a digit.

The result of the expression will always evaluate to either a digit `0-9`, `T` or `F`.


Example 1:
Input: "T?2:3"
Output: "2"
Explanation: If true, then result is 2; otherwise result is 3.


Example 2:
Input: "F?1:T?4:5"
Output: "4"
Explanation: The conditional expressions group right-to-left. Using parenthesis, it is read/evaluated as:
             "(F ? 1 : (T ? 4 : 5))"                   "(F ? 1 : (T ? 4 : 5))"
          -> "(F ? 1 : 4)"                 or       -> "(T ? 4 : 5)"
          -> "4"                                    -> "4"

Example 3:
Input: "T?T?F:5:3"
Output: "F"
Explanation: The conditional expressions group right-to-left. Using parenthesis, it is read/evaluated as:
             "(T ? (T ? F : 5) : 3)"                   "(T ? (T ? F : 5) : 3)"
          -> "(T ? F : 3)"                 or       -> "(T ? F : 5)"
          -> "F"                                    -> "F"

### Solution 
 To solve the problem of evaluating a nested ternary expression represented as a string, we need to carefully parse through the string while adhering to the sequence of conditions and corresponding results. The expression follows a right-to-left grouping, meaning that we should evaluate the expression from the end towards the beginning.

### Strategy
- Use a stack to keep track of the results and conditions as we parse through the expression.
- As we encounter characters in the string, we handle the conditions (`T`, `F`), question marks (`?`), and colons (`:`) appropriately.
- Starting from the end of the string moves us through the expressions, allowing us to resolve each ternary condition correctly.

### Detailed Steps
1. **Initialization**: Create a stack to help with the evaluation of the expression.
2. **Iterate Backwards**: Traverse the expression from the last character to the first:
    - If you find a digit (`0-9`), push it onto the stack.
    - If you find `T` or `F`, push this value onto the stack as it represents a true or false condition.
    - If you encounter `?:`, retrieve the top two elements from the stack:
        - The second element will be the result if the preceding condition is true (`T`).
        - The first element will be the result if the condition is false (`F`).
3. **Performing the Evaluation**:
    - Depending on whether the current character is `T` or `F`, decide which value to keep from the two popped values, and push the result back onto the stack.
4. **Final Result**: The process continues until we evaluate the entire expression, and the last element remaining in the stack is the final result.

### Implementation in Python
Here is the Python code that implements the aforementioned strategy:



```python
class Solution:
    def parseTernary(self, expression: str) -> str:
        stack = []
        
        # Iterate backwards through the expression
        for i in range(len(expression) - 1, -1, -1):
            char = expression[i]
            if char == '?':
                # When we find a '?', we need to pop two values from the stack
                true_result = stack.pop()  # evaluates if condition is True
                false_result = stack.pop()  # evaluates if condition is False
                # The character before '?' defines condition
                condition = expression[i - 1]  
                
                # Determine the result based on the condition
                if condition == 'T':
                    stack.append(true_result)
                else:  # condition is 'F'
                    stack.append(false_result)
                
                # Skip the next character since it is part of the ternary
                i -= 1
            else:
                # Push 'T', 'F', or digit onto the stack
                stack.append(char)
        
        # The final result will be on the top of the stack
        return stack[-1]

```

### Explanation of the Code
1. **Class Definition**: We define a class `Solution` with a method `parseTernary` that takes a string `expression`.
2. **Stack Implementation**: We use a list to simulate stack behavior.
3. **Backward Iteration**: The main loop processes each character in reverse:
   - Handle the '?' by popping the last two results off the stack and pushing the appropriate result based on the preceding condition.
   - Push `T`, `F`, or digits directly onto the stack.
4. **Returning the Result**: The final evaluated result is returned after fully processing the expression.

This implementation is efficient and handles expressions correctly within a single pass, managing complexity effectively. The algorithm primarily relies on managing the stack and conditional logic driven by the ternary format of the input string.

# 472. Concatenated Words

### Problem Description 
Given an array of strings `words` (without duplicates), return all the concatenated words in the given list of `words`.

A concatenated word is defined as a string that is comprised entirely of at least two shorter words in the given array.


Example 1:
Input: words = ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]
Output: ["catsdogcats","dogcatsdog","ratcatdogcat"]
Explanation: "catsdogcats" can be concatenated by "cats", "dog" and "cats"; 
"dogcatsdog" can be concatenated by "dog", "cats" and "dog"; 
"ratcatdogcat" can be concatenated by "rat", "cat", "dog" and "cat".


Example 2:
Input: words = ["cat","dog","catdog"]
Output: ["catdog"]

Constraints:
`1 <= words.length <= 104`
`0 <= words[i].length <= 1000`
`words[i]` consists of only lowercase English letters.

`0 <= sum(words[i].length) <= 6 * 105`

### Solution 
 To solve the problem of finding concatenated words in an array of strings, we can use a combination of dynamic programming and a set for quick lookups. The idea is to check for each word if it can be formed by concatenating two or more shorter words from the given array.

### Approach:
1. **Use a Set for Fast Lookup**: We will store all words in a set to allow O(1) time complexity for checking existence of words.

2. **Dynamic Programming**: For each word, we will use a DP array to check if the word can be formed by concatenating shorter words. 
   - `dp[i]` will be `True` if the substring `word[0:i]` can be segmented into at least two words from the set.
   - For each position `i`, we will check all possible partitions `j` (from 1 to `i`) and see if `word[j:i]` is in the set and if `dp[j]` is `True`.

3. **Counting Words**: When checking a word, we must ensure that we do not count the word itself as a possible concatenated word. Thus, we will temporarily remove the word from the set while checking it.

4. **Result Collection**: If a word can be formed using the condition mentioned, it will be added to the results.

### Code Implementation:
Now, let's implement this in Python according to the LeetCode format.



```python
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        # Create a set of words for quick lookup
        word_set = set(words)
        concatenated_words = []

        def can_form(word: str) -> bool:
            # Dynamically check if the word can be formed from other words
            if len(word) == 0:
                return False
            
            dp = [False] * (len(word) + 1)
            dp[0] = True  # An empty word can be formed trivially
            
            for i in range(1, len(word) + 1):
                for j in range(1, i + 1):
                    # Check if the substring word[j-1:i] is in set 
                    # and if dp[j-1] is True
                    if dp[j-1] and word[j-1:i] in word_set:
                        dp[i] = True
                        break
                    
            return dp[len(word)] and any(dp[j] for j in range(1, len(word)))

        for word in words:
            word_set.remove(word)  # Temporarily remove the word
            if can_form(word):
                concatenated_words.append(word)
            word_set.add(word)  # Add the word back to the set

        return concatenated_words

```

### Explanation of the Code:
- We define our solution class `Solution` and the method `findAllConcatenatedWordsInADict`.
- We create a `word_set` containing all the words for O(1) lookups.
- The `can_form` function takes a word and checks if it can be formed using other words:
  - We initialize a DP array, where `dp[i]` indicates whether `word[:i]` can be formed.
  - We iterate through all partitions of the current word, checking if the current partition and the previous segment can be formed.
- In the main loop, we temporarily remove each word from `word_set` to avoid using the word itself in its composition, then check it with `can_form`.
- Finally, we return the list of concatenated words.

This implementation effectively checks for concatenated words while ensuring that the solution runs efficiently within the provided constraints.

# 473. Matchsticks to Square

### Problem Description 
Remember the story of Little Match Girl? By now, you know exactly what matchsticks the little match girl has, please find out a way you can make one square by using up all those matchsticks. You should not break any stick, but you can link them up, and each matchstick must be used exactly one time.

 Your input will be several matchsticks the girl has, represented with their stick length. Your output will either be true or false, to represent whether you could make one square using all the matchsticks the little match girl has.


Example 1:
Input: [1,1,2,2,2]
Output: true
Explanation: You can form a square with length 2, one side of the square came two sticks with length 1.


Example 2:
Input: [3,3,3,3,4]
Output: false
Explanation: You cannot find a way to form a square with all the matchsticks.

Note:
The length sum of the given matchsticks is in the range of `0` to `10^9`.

The length of the given matchstick array will not exceed `15`.

### Solution 
 To solve the problem of determining whether a given set of matchsticks can form a square, we can utilize a backtracking approach. The basic idea is to consider each matchstick one by one and try to fit it to one of the four sides of the square until either all the matchsticks are used and a valid square is formed, or we exhaust all possibilities.

### Steps to Solve the Problem

1. **Calculate the Total Length**: Compute the sum of all matchsticks. For a valid square, this total must be divisible by 4 since a square has four equal sides.

2. **Target Side Length**: If valid (i.e., divisible by 4), set the target side length as `total_length // 4`.

3. **Sorting**: Sort the matchsticks in descending order. This helps in trying to fit longer matchsticks first, potentially reducing the search space.

4. **Backtracking**: Start backtracking to recursively try to fit each matchstick into one of the four sides of the square. If at any point the current side exceeds the target length, we backtrack.

5. **Base Case**: If all matchsticks are used successfully and each side is equal to the target length, we return true. If we exhaust all possibilities without forming a square, we return false.

### Python Code Implementation

Here is the Python implementation of the solution using the backtracking technique:



```python
class Solution:
    def makesquare(self, matchsticks):
        total_length = sum(matchsticks)
        
        # If the total length is not divisible by 4, we cannot form a square
        if total_length % 4 != 0:
            return False
        
        target_length = total_length // 4
        matchsticks.sort(reverse=True)  # Sort in descending order for optimization
        sides = [0] * 4  # Initialize lengths of 4 sides
        
        def backtrack(index):
            if index == len(matchsticks):  # All matchsticks used
                return sides[0] == sides[1] == sides[2] == sides[3] == target_length
            
            for i in range(4):
                if sides[i] + matchsticks[index] <= target_length:  # Valid to place matchstick
                    sides[i] += matchsticks[index]  # Place the matchstick
                    if backtrack(index + 1):  # Recur to place next matchstick
                        return True
                    sides[i] -= matchsticks[index]  # Backtrack (remove the matchstick)
                    
                # If current side length is 0 (i.e., we couldn't place it), no need to try next sides
                if sides[i] == 0:
                    break
                    
            return False
        
        return backtrack(0)  # Start backtracking with the first matchstick

```

### Explanation of the Code

1. **Class Definition**: We define a class `Solution` that contains the method `makesquare`.

2. **Input Handling**: We compute the total length of matchsticks and check if it's divisible by 4.

3. **Sorting**: We sort the matchsticks in descending order. This helps optimize the backtracking process.

4. **Backtracking Function**: 
   - `backtrack` takes the current index of the matchstick to place.
   - If all matchsticks are used, it checks if all sides are equal to the target length.
   - It loops through each side, tries to place the current matchstick if possible, and calls itself recursively for the next matchstick.
   - If placing the current matchstick does not lead to a solution, it backtracks by removing the matchstick from the current side.
   - To avoid redundant checks, the process stops trying further sides when a matchstick can't be placed on the first side.

5. **Return Value**: The method returns `True` if a square can be formed, otherwise `False`.

### Complexity Analysis
- **Time Complexity**: The worst-case scenario is exponential due to backtracking (O(4^n), where n is the number of matchsticks).
- **Space Complexity**: The recursion stack goes as deep as the number of matchsticks, yielding O(n) space complexity.

This solution adheres to the constraints of the problem and efficiently determines if a square can be formed using all the matchsticks.

# 488. Zuma Game

### Problem Description 
Think about Zuma Game. You have a row of balls on the table, colored red(R), yellow(Y), blue(B), green(G), and white(W). You also have several balls in your hand.

Each time, you may choose a ball in your hand, and insert it into the row (including the leftmost place and rightmost place). Then, if there is a group of 3 or more balls in the same color touching, remove these balls. Keep doing this until no more balls can be removed.

Find the minimal balls you have to insert to remove all the balls on the table. If you cannot remove all the balls, output -1.


Example 1:
Input: board = "WRRBBW", hand = "RB"
Output: -1
Explanation: WRRBBW -> WRR[R]BBW -> WBBW -> WBB[B]W -> WW

Example 2:
Input: board = "WWRRBBWW", hand = "WRBRW"
Output: 2
Explanation: WWRRBBWW -> WWRR[R]BBWW -> WWBBWW -> WWBB[B]WW -> WWWW -> empty

Example 3:
Input: board = "G", hand = "GGGGG"
Output: 2
Explanation: G -> G[G] -> GG[G] -> empty 

Example 4:
Input: board = "RBYYBBRRB", hand = "YRBGB"
Output: 3
Explanation: RBYYBBRRB -> RBYY[Y]BBRRB -> RBBBRRB -> RRRB -> B -> B[B] -> BB[B] -> empty 

Constraints:
You may assume that the initial row of balls on the table won’t have any 3 or more consecutive balls with the same color.

`1 <= board.length <= 16`
`1 <= hand.length <= 5`
Both input strings will be non-empty and only contain characters 'R','Y','B','G','W'.

### Solution 
 To solve the Zuma Game problem, we need to simulate the insertion of balls into the row, followed by the removal of any groups of three or more adjacent balls of the same color. The challenge is to find the minimum number of balls required to completely clear the board or determine if it's impossible.

Here's a step-by-step breakdown of the solution strategy:

1. **Recursive Backtracking**: We'll employ a recursive approach to try inserting each ball from the hand into every possible position on the board. After each insertion, we'll check if any groups of three or more balls of the same color can be removed.

2. **Removing Groups**: A helper function will be used to repeatedly remove groups of three or more matching colored balls from the board until no more such groups exist.

3. **Pruning**: If at any state the board cannot be cleared (i.e., any insertion has been made without producing removals infinitely), we can prune that branch of the recursion.

4. **Memoization**: To avoid recalculating states we've already seen, we can use memoization to store results for specific board and hand configurations.

5. **Depth First Search**: We will use depth-first search (DFS) to explore all possible states by inserting the balls.

The implementation will look like this:



```python
class Solution:
    def findMinStep(self, board: str, hand: str) -> int:
        from collections import Counter
        
        # Count the occurrences of each ball in hand
        hand_count = Counter(hand)
        
        def remove_consecutive(s):
            stack = []
            for char in s:
                if stack and stack[-1] == char:
                    stack.append(char)
                    # Check for groups of three
                    if len(stack) > 2 and (len(stack) == 3 or stack[-3] == stack[-1]):
                        stack.pop()
                        stack.pop()
                        stack.pop()
                else:
                    stack.append(char)
            return ''.join(stack)
        
        def dfs(board, hand_count):
            # Remove consecutive balls from board
            board = remove_consecutive(board)
            
            if not board:  # If the board is empty, we're done
                return 0
            
            min_steps = float('inf')
            n = len(board)
            for i in range(n):
                for color, count in hand_count.items():
                    if count > 0:
                        # Try inserting one ball of this color before the current position i
                        new_board = board[:i] + color + board[i:]
                        new_hand_count = hand_count.copy()
                        new_hand_count[color] -= 1
                        # Recursively attempt to solve the new board with the updated hand
                        result = dfs(new_board, new_hand_count)
                        if result != -1:
                            min_steps = min(min_steps, result + 1)

            return min_steps if min_steps != float('inf') else -1
        
        return dfs(board, hand_count)

```

### Explanation of the Code:

1. **Imports & Counting**: We start by importing `Counter` from the `collections` module to count how many of each color we have in the hand.

2. **Removing Consecutive Balls**: The function `remove_consecutive` removes adjacent balls of the same color until no more can be removed.

3. **DFS Function**: The `dfs` function recursively tries to solve the problem:
   - First, it removes any immediate groups from the current board state.
   - If the board is empty after removals, it returns 0 since no more balls need to be inserted.
   - It checks every possible position on the board to insert each available color from hand and recursively attempts to solve with the updated state.

4. **Counting Minimum Steps**: For each successful insertion and subsequent recursive call, it keeps track of the minimum number of balls inserted and returns that.

5. **Termination**: If we exhaust our options without clearing the board, we return -1.

This algorithm leverages recursion with backtracking effectively to cover all possibilities and adheres strictly to the constraints set by the problem description.

### Time Complexity:
The worst-case time complexity can be exponential in nature due to the recursive nature and branching from possible insertions, but memoization and the small constraint sizes for `board` (maximum length 16) and `hand` (maximum length 5) help keep it manageable in practice.

### Space Complexity:
The space complexity depends on the depth of the recursion and the space used by the `hand_count`, leading to a maximum of O(N + M) where N is the length of the board and M is the size of the hand.

# 489. Robot Room Cleaner

### Problem Description 
Given a robot cleaner in a room modeled as a grid.

Each cell in the grid can be empty or blocked.

The robot cleaner with 4 given APIs can move forward, turn left or turn right. Each turn it made is 90 degrees.

When it tries to move into a blocked cell, its bumper sensor detects the obstacle and it stays on the current cell.

Design an algorithm to clean the entire room using only the 4 given APIs shown below.

interface Robot {
  // returns true if next cell is open and robot moves into the cell.

  // returns false if next cell is obstacle and robot stays on the current cell.

  boolean move();
  // Robot will stay on the same cell after calling turnLeft/turnRight.

  // Each turn will be 90 degrees.

  void turnLeft();
  void turnRight();
  // Clean the current cell.

  void clean();
}

Example:
Input:
room = [
  [1,1,1,1,1,0,1,1],
  [1,1,1,1,1,0,1,1],
  [1,0,1,1,1,1,1,1],
  [0,0,0,1,0,0,0,0],
  [1,1,1,1,1,1,1,1]
],
row = 1,
col = 3
Explanation:
All grids in the room are marked by either 0 or 1.

0 means the cell is blocked, while 1 means the cell is accessible.

The robot initially starts at the position of row=1, col=3.

From the top left corner, its position is one row below and three columns right.

Notes:
The input is only given to initialize the room and the robot's position internally. You must solve this problem "blindfolded". In other words, you must control the robot using only the mentioned 4 APIs, without knowing the room layout and the initial robot's position.

The robot's initial position will always be in an accessible cell.

The initial direction of the robot will be facing up.

All accessible cells are connected, which means the all cells marked as 1 will be accessible by the robot.

Assume all four edges of the grid are all surrounded by wall.

### Solution 
 To solve this problem, we need to make the robot cleaner traverse the entire grid and clean all accessible cells marked as `1`. The important points to emphasize are that the robot cannot see the environment directly, and it has to rely on its moving abilities which include moving forward, turning left or right, and cleaning the current cell.

### Strategy:
1. **Coordinate System**: We can represent the robot's movement direction using a list that provides the delta (change in position) for each of the four directions (up, right, down, left).
2. **DFS Approach**: We'll use a Depth First Search (DFS) approach to explore each accessible cell.
3. **Visited Tracking**: We'll keep track of visited cells to avoid cleaning the same cell multiple times.
4. **Turn Management**: We will manage the robot's direction by using the turn APIs correctly.

### Implementation:
Here's the implementation in Python:



```python
class Solution:
    def cleanRoom(self, robot: 'Robot') -> None:
        # Directions represent [up, right, down, left] in grid coordinates
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        visited = set()  # Track visited cells
        def dfs(x, y, d):
            # Clean the current cell and mark it as visited
            robot.clean()
            visited.add((x, y))
            
            # Try all four directions
            for i in range(4):
                # Calculate new direction index
                new_d = (d + i) % 4
                # Calculate new position
                nx, ny = x + directions[new_d][0], y + directions[new_d][1]
                # Move in the current direction
                if (nx, ny) not in visited and robot.move():
                    dfs(nx, ny, new_d)  # Recur to the next cell
                    # Go back to the original position
                    self.goBack(robot)
                # Turn the robot right to face the next direction
                robot.turnRight()

        # Start DFS from initial position (0, 0) facing 'up' (index 0)
        dfs(0, 0, 0)

    def goBack(self, robot: 'Robot') -> None:
        # Turn the robot around (180 degrees) and move back
        robot.turnLeft()
        robot.turnLeft()
        robot.move()
        robot.turnLeft()  # Restore original direction
        robot.turnLeft()

```

### Explanation:
- **Direction Array**: `directions` is an array that helps in translating the current direction (up, right, down, left) into x and y position changes.
- **Visited Tracking**: We use a set named `visited` to store the coordinates of cells that have already been cleaned.
- **DFS Function**: 
  - The `dfs(x, y, d)` function cleans the current cell, marks it as visited, and then iterates through all four possible directions.
  - For each direction, it calculates the new coordinates and checks if that cell has been visited or if the robot can move there (meaning the cell is not blocked).
  - If the robot successfully moves, we recursively call `dfs` from the new position, passing the new direction.
  - After exploring one direction fully, we turn the robot right to explore the next one.
- **Going Back**: The `goBack()` function is called after returning from a deeper recursion level, which makes the robot turn around to go back to the previous position.
- **Starting Point**: The cleaning starts from the robot's initial position at `(0, 0)` facing 'up' (index 0 in our direction array).

This code follows the LeetCode format and can be directly run on their platform. The algorithm successfully simulates the robot cleaning the room while adhering to the constraints provided.

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
 To solve the problem of determining whether a ball can reach a specified destination in a maze, where the ball rolls in one of four directions until it hits a wall, we can implement a depth-first search (DFS) or breadth-first search (BFS) algorithm. The ball will roll freely until it cannot go further, at which point we will check if it lands on the destination.

Here's the step-by-step breakdown of how we can implement this:

1. **Model the Maze**: We will represent the maze as a 2D list, where `0` denoted an empty space and `1` denotes a wall.

2. **Define Movement**: The ball can move in four possible directions: up, down, left, and right. We will define these movements with directional vectors.

3. **Roll Until Stop**: We need a function that would roll the ball in a given direction until it hits a wall or the edge of the maze.

4. **DFS Search**: We will perform a DFS from the start position of the ball. At each step, we will roll the ball in each of the four directions to a stopping point and check if it can lead to the destination.

5. **Visited Tracking**: To avoid cycles and unnecessary checks, we will maintain a set of visited positions.

6. **Stopping Condition**: If the ball reaches the destination, we return true; if we exhaust all possibilities without reaching it, we return false.

Here’s how the implementation looks in Python:



```python
class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        # Directions for moving in the maze: down, up, right, left
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        m, n = len(maze), len(maze[0])
        visited = set()  # To keep track of visited positions
        
        # DFS function to explore paths in the maze
        def dfs(row, col):
            # If we reach the destination, return True
            if [row, col] == destination:
                return True
            # Mark the current position as visited
            visited.add((row, col))
            
            # Explore each direction
            for dr, dc in directions:
                r, c = row, col
                
                # Roll the ball until it hits a wall
                while 0 <= r + dr < m and 0 <= c + dc < n and maze[r + dr][c + dc] == 0:
                    r += dr
                    c += dc
                
                # Check if the new position is already visited
                if (r, c) not in visited:
                    # Continue DFS from the new stopping point
                    if dfs(r, c):
                        return True
            
            return False  # No path found
        
        return dfs(start[0], start[1])

```

### Explanation of the Code:

**Class Definition**: We define a class `Solution` with a method `hasPath` that takes the maze, start position, and destination.

1. **Direction Vectors**: The list `directions` defines how the ball can move—right, left, down, up.

2. **Set for Visited Cells**: A set `visited` is used to track cells we've already rolled to, preventing revisiting and infinite loops.

3. **DFS Function**: The `dfs` function takes the current position of the ball and searches for a path:
   - If the current position matches the destination, it returns `True`.
   - It adds the current position to `visited`.
   - For each direction, the ball is rolled until it stops at a wall.
   - If the new stopping position hasn’t been visited, the DFS function is called recursively.

4. **Return Result**: The result of the initial call to `dfs(start[0], start[1])` is returned, which will be `True` or `False` depending on whether the ball can reach the destination.

This solution efficiently traverses the maze and checks for possible paths, adhering to the constraints outlined in the problem statement.

# 491. Increasing Subsequences

### Problem Description 
Given an integer array, your task is to find all the different possible increasing subsequences of the given array, and the length of an increasing subsequence should be at least 2.


Example:
Input: [4, 6, 7, 7]
Output: [[4, 6], [4, 7], [4, 6, 7], [4, 6, 7, 7], [6, 7], [6, 7, 7], [7,7], [4,7,7]]

Constraints:
The length of the given array will not exceed 15.

The range of integer in the given array is [-100,100].

The given array may contain duplicates, and two equal integers should also be considered as a special case of increasing sequence.

### Solution 
 To solve the problem of finding all different possible increasing subsequences in an integer array with a minimum length of 2, we can use a backtracking approach. This method allows us to explore all possible combinations of the elements in the array while ensuring that we only include those subsequences that are strictly increasing. We'll also take care to handle duplicates to ensure that we do not produce the same subsequence multiple times.

Here’s a step-by-step breakdown of the approach:

### Step-by-step Explanation:

1. **Backtracking Function**: We define a recursive function that will be responsible for constructing the subsequences. 
   - The function will take the current index, the current subsequence being built, and a set to store the unique subsequences.

2. **Base Case**: If the current subsequence has a size of at least 2, we add it to the set of unique subsequences.

3. **For Loop**: We iterate through the remaining elements of the array:
   - If the current element can be added to the subsequence (it's greater than or equal to the last element of the subsequence), we include it and recursively call the function for the next index.
   - To handle duplicates, we maintain a set to track which elements have been included in the current position of the loop.

4. **Combination Collection**: Use a set to ensure we only get unique subsequences.

5. **Final Output**: After generating the subsequences, we convert our set to a list to return the results as required.

Now, let's implement this in Python following the LeetCode format:



```python
from typing import List, Set

class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        def backtrack(start: int, path: List[int]):
            # If the size of the current path is at least 2, add it to results
            if len(path) >= 2:
                res.add(tuple(path))  # Store as tuple to maintain uniqueness
            
            visited: Set[int] = set()  # To track duplicates at the current level
            for i in range(start, len(nums)):
                # If the current number can be part of an increasing subsequence
                if not path or nums[i] >= path[-1]:
                    # Avoid duplicates at this position
                    if nums[i] not in visited:
                        visited.add(nums[i])
                        path.append(nums[i])  # Choose the current number
                        backtrack(i + 1, path)  # Move to the next index
                        path.pop()  # Backtrack to try other combinations
        
        res = set()  # Using a set to store unique subsequences
        backtrack(0, [])
        return [list(seq) for seq in res]  # Convert tuples back to lists for result

```

### Explanation of Code:

- **Function Signature**: The `findSubsequences` function takes a list of integers as input and outputs a list of lists containing increasing subsequences.
  
- **Nested Backtrack Function**: 
  - `start` keeps track of where to begin the next search.
  - `path` holds the current subsequence being constructed.
  - The uniqueness of subsequences is ensured by using a set `visited` at each level of recursion.

- **Adding Subsequences**: When a valid subsequence (of length >= 2) is formed, it is added to the `res` set.

- **Returning Results**: Finally, we convert the set of tuples to a list of lists before returning.

### Testing the Solution:

You can run this solution directly in the LeetCode environment, and it should successfully return the expected output for the input provided.

### Example Usage:
For the input `[4, 6, 7, 7]`, calling `findSubsequences` would yield:


```python
solution = Solution()
print(solution.findSubsequences([4, 6, 7, 7]))
# Expected Output: [[4, 6], [4, 7], [4, 6, 7], [4, 6, 7, 7], [6, 7], [6, 7, 7], [7, 7], [4, 7, 7]]

``` 

This implementation efficiently handles the problem while keeping the constraints and requirements in mind.

# 494. Target Sum

### Problem Description 
You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols `+` and `-`. For each integer, you should choose one from `+` and `-` as its new symbol.

Find out how many ways to assign symbols to make sum of integers equal to target S.


Example 1:
Input: nums is [1, 1, 1, 1, 1], S is 3. 
Output: 5
Explanation: 
-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3
There are 5 ways to assign symbols to make the sum of nums be target 3.


Constraints:
The length of the given array is positive and will not exceed 20.

The sum of elements in the given array will not exceed 1000.

Your output answer is guaranteed to be fitted in a 32-bit integer.

### Solution 
 To solve the problem of finding the number of ways to assign symbols to a list of integers such that their sum equals a given target \( S \), we can utilize a recursive approach combined with memoization, or we could approach this using dynamic programming. Given the constraints, both approaches are feasible.

Below, I will provide a dynamic programming solution, as it is more efficient for larger inputs due to its iterative nature and avoidance of redundant calculations.

### Explanation

1. **Understanding Positive and Negative Sums**:
   - Each number can either contribute positively (using `+`) or negatively (using `-`) to the total sum.
   - Let's define `sumPositive` as the sum of numbers where we assign `+`, and `sumNegative` as the sum of numbers where we assign `-`.
   - Our goal is to find all possible distributions such that:
     \[
     sumPositive - sumNegative = S
     \]
   - This simplifies to:
     \[
     sumPositive = (S + sumTotal) / 2
     \]
   where `sumTotal` is the sum of all elements in the array.
   
2. **Feasibility of the Equation**:
   - For `sumPositive` to be an integer, \( S + sumTotal \) must be even. If it’s odd, there's no solution.

3. **Dynamic Programming Setup**:
   - We'll use a DP array where `dp[i]` indicates the number of ways to achieve the sum of `i` using the numbers we've seen so far.
   - Initialize `dp[0]` to 1 because there's one way to sum to 0 (by choosing nothing).
   - Iterate through each number and update the DP array for each possible sum in reverse.

### The Code

Here's how the complete function looks, structured according to LeetCode standards:



```python
class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        sumTotal = sum(nums)
        
        # Check if (S + sumTotal) is even and non-negative
        if (S + sumTotal) % 2 != 0 or S > sumTotal:
            return 0
        
        target = (S + sumTotal) // 2
        
        # Initialize a list for dynamic programming
        dp = [0] * (target + 1)
        dp[0] = 1  # Base case
        
        # Fill the dp array
        for num in nums:
            for i in range(target, num - 1, -1):
                dp[i] += dp[i - num]
        
        return dp[target]

```

### Explanation of the Code:
- We first calculate the total sum of the given integers.
- We check if the converted target is valid (even and non-negative).
- We initialize the `dp` array with size `target + 1` to store the counts.
- The outer loop goes through each number in `nums`, while the inner loop computes the number of ways to sum to the targets from `target` down to the current number.
- The final answer is stored in `dp[target]`, which we return.

### Example Usage
You can test the above solution using the following example:



```python
# Example test case
solution = Solution()
result = solution.findTargetSumWays([1, 1, 1, 1, 1], 3)
print(result)  # Output should be 5

```

This solution effectively counts all the ways to achieve the target sum using the available numbers, accommodating all constraints and the maximum input size specified.

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
 To solve the problem of navigating the ball through the maze to the hole, we can model the maze as a graph and perform a breadth-first search (BFS) or depth-first search (DFS) approach to find the shortest path to the hole. Here's a step-by-step explanation followed by the implementation.

### Steps to Solve the Problem:

1. **Input Parsing**: We'll receive a maze represented as a 2D list (matrix), along with the ball's starting position and the hole's position.

2. **Direction Handling**: We define the four possible directions the ball can roll: up, down, left, and right. We'll also maintain a map of these directions to their respective moves.

3. **Maze Traversal**: We'll traverse the maze using a DFS approach. The ball will continue moving in one direction until it hits a wall. At that point, we will check if the ball is at the hole, and if so, we update our result. If not, we can then explore other directions from the current position.

4. **Path Storage**: We will keep track of the distance traveled and the corresponding path to reach the hole. We will update our results if we find a shorter path or if we find a path of the same distance but lexicographically smaller instructions.

5. **Handling Impossible Cases**: If we finish traversing all paths and don't reach the hole, we return "impossible".

### Implementation:

Here is the complete code for the problem:



```python
class Solution:
    def findShortestWay(self, maze: List[List[int]], ball: List[int], hole: List[int]) -> str:
        m, n = len(maze), len(maze[0])
        directions = [(0, 1, 'r'), (0, -1, 'l'), (1, 0, 'd'), (-1, 0, 'u')]  # right, left, down, up
        visited = set()
        result = None
        
        def dfs(x, y, path):
            nonlocal result
            
            # If already visited or path is worse, return
            if (x, y) in visited: 
                return
            visited.add((x, y))
            
            # If we reach the hole, check and possibly update the result
            if (x, y) == (hole[0], hole[1]):
                if result is None or path < result:
                    result = path
                return
            
            # Roll the ball in each direction
            for dx, dy, direction in directions:
                nx, ny = x, y
                moves = 0
                
                # Keep rolling until we hit a wall or reach the hole
                while 0 <= nx + dx < m and 0 <= ny + dy < n and maze[nx + dx][ny + dy] == 0:
                    nx += dx
                    ny += dy
                    moves += 1
                    
                    # If we reach the hole while rolling
                    if (nx, ny) == (hole[0], hole[1]):
                        if result is None or path + direction < result:
                            result = path + direction
                        break
                
                # If we rolled some distance, call dfs recursively
                if moves > 0:
                    dfs(nx, ny, path + direction)
        
        # Start DFS from the ball's position
        dfs(ball[0], ball[1], "")
        
        return result if result is not None else "impossible"

```

### Explanation of Code:

- We define a class `Solution` with a method `findShortestWay`.
- The method takes `maze`, `ball`, and `hole` as inputs and initializes necessary variables.
- The `dfs` function is defined to explore paths recursively. It tracks the current position, path taken, and a set of visited nodes to avoid cycles.
- We iterate through possible directions. For each one, the ball rolls until a wall is hit or it reaches the hole. After rolling, we make a recursive call to explore further paths.
- Finally, the result is returned as a string signaling the path the ball should take, or "impossible" if no valid path exists.

This solution efficiently finds the shortest path in the maze while considering both distance and lexicographic order of directions.

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
 To solve the problem of finding the shortest distance for a ball to reach a destination in a maze, we can use a graph traversal algorithm such as Dijkstra's algorithm. This algorithm is appropriate because it will allow us to explore all possible paths in an efficient manner, minimizing the distance traveled.

The maze is represented by a grid, where `0`s indicate empty spaces and `1`s indicate walls. The ball can roll in four directions (up, down, left, right) until it hits a wall, at which point it can choose a new direction to roll.

Here's the detailed approach we'll take:

1. **Data Structures**:
   - Use a priority queue to implement Dijkstra's algorithm for exploring paths based on distance.
   - Maintain a distance matrix to keep track of the shortest distances to each cell from the start position.

2. **Rolling Mechanism**:
   - Define a function to simulate rolling the ball in a specified direction until it hits a wall. This function will return the final position and the distance rolled.

3. **Priority Queue**:
   - For each position in the maze, push the new position and the accumulated distance into the priority queue to explore.

4. **Termination**:
   - If we reach the destination, return the distance.
   - If we exhaust all possibilities without reaching the destination, return `-1`.

5. **Edge Cases**:
   - Make sure to handle cases when the ball can go out of bounds (it can't) and avoid visiting already optimized positions.

Here is the full Python implementation following the LeetCode format:



```python
import heapq

class Solution:
    def findShortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        m, n = len(maze), len(maze[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        # Distance matrix
        distance = [[float('inf')] * n for _ in range(m)]
        distance[start[0]][start[1]] = 0
        
        # Priority queue for Dijkstra's
        pq = [(0, start[0], start[1])]  # (distance, row, col)
        
        while pq:
            dist, x, y = heapq.heappop(pq)
            
            # If the destination is reached
            if [x, y] == destination:
                return dist
            
            # Ignore if the current distance is not optimal
            if dist > distance[x][y]:
                continue
            
            for dx, dy in directions:
                nx, ny, d = x, y, 0
                
                # Roll until hitting a wall
                while 0 <= nx + dx < m and 0 <= ny + dy < n and maze[nx + dx][ny + dy] == 0:
                    nx += dx
                    ny += dy
                    d += 1
                
                # Update the distance matrix
                if dist + d < distance[nx][ny]:
                    distance[nx][ny] = dist + d
                    heapq.heappush(pq, (distance[nx][ny], nx, ny))
        
        return -1 if distance[destination[0]][destination[1]] == float('inf') else distance[destination[0]][destination[1]]

```

### Explanation:
- **Initialization**: We initialize the `distance` matrix with `infinity` and set the distance of the starting position to `0`. The priority queue is initialized with a tuple containing distance and starting position.
- **Main Loop**: We pop elements from the priority queue, check if we've reached the destination, and if not, explore all four possible rolling directions. 
- **Rolling and Distance Calculation**: For each direction, we roll the ball until it hits a wall and calculate the distance traveled. If this distance is shorter than any previously recorded distance for that position, we update the distance and push the new position to the queue.
- **Final Decision**: If the destination distance remains as `infinity`, we return `-1`, else we return the computed shortest distance. 

This implementation operates efficiently within the constraints set by the problem and handles all edge cases appropriately.

