# 538. Convert BST to Greater Tree

### Problem Description 
Given the `root` of a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus sum of all keys greater than the original key in BST.

As a reminder, a binary search tree is a tree that satisfies these constraints:
The left subtree of a node contains only nodes with keys less than the node's key.

The right subtree of a node contains only nodes with keys greater than the node's key.

Both the left and right subtrees must also be binary search trees.

Note: This question is the same as 1038: https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/

Example 1:
Input: root = [4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
Output: [30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]

Example 2:
Input: root = [0,null,1]
Output: [1,null,1]

Example 3:
Input: root = [1,0,2]
Output: [3,3,2]

Example 4:
Input: root = [3,2,4,1]
Output: [7,9,4,10]

Constraints:
The number of nodes in the tree is in the range `[0, 104]`.

`-104 <= Node.val <= 104`
All the values in the tree are unique.

`root` is guaranteed to be a valid binary search tree.

### Solution 
 To solve the problem of converting a Binary Search Tree (BST) to a Greater Tree, we need to adjust the values of the nodes such that each node's value is replaced by the sum of its original value and the values of all nodes that are greater than it. 

### Strategy:
Since the properties of the BST allow us to traverse it in a way that we process the nodes in a decreasing order (i.e., visiting the right subtree first), we can perform a reverse in-order traversal. This will give us a pathway to accumulate the values of the nodes we have seen so far, allowing us to directly compute the new value for each node based on this running sum.

### Algorithm:
1. Initialize a variable to keep track of the cumulative sum of the node values.
2. Define a recursive function that:
   - Visits the right subtree.
   - Updates the current node’s value by adding the cumulative sum.
   - Update the cumulative sum to include the current node’s original value.
   - Visits the left subtree.
3. Start this traversal from the root.

### Complexity:
- **Time Complexity**: O(N), where N is the number of nodes in the tree, as we traverse each node exactly once.
- **Space Complexity**: O(H), where H is the height of the tree (due to the recursion stack).

### Python Code:
Here is the implementation in Python:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        # Initialize the cumulative sum
        self.cumulative_sum = 0
        
        def reverse_inorder(node: TreeNode):
            if not node:
                return
            
            # Traverse the right subtree first (greater values)
            reverse_inorder(node.right)
            
            # Update the cumulative sum and the current node's value
            self.cumulative_sum += node.val
            node.val = self.cumulative_sum
            
            # Traverse the left subtree (smaller values)
            reverse_inorder(node.left)

        # Start the reverse in-order traversal
        reverse_inorder(root)
        return root

```

### Explanation of the Code:
1. **TreeNode Class**: This is the definition of a TreeNode, which includes the value (`val`) and pointers to the left and right children.
2. **Solution Class and Method**:
   - `convertBST`: This method initializes the `cumulative_sum` variable and starts the `reverse_inorder` traversal.
3. **Reverse In-order Function**:
   - It checks if the node is `None`. If it is not, it first processes the right child.
   - The current node's value is updated by adding the existing `cumulative_sum`.
   - The cumulative sum is then updated to include the current node's original value, making it ready for the next node.
   - Finally, it processes the left child.

### Usage:
To use this code on LeetCode, simply copy and paste it into the solution editor. The provided `TreeNode` class will be correctly instantiated by LeetCode when you run the test cases against your solution.

This solution effectively builds the Greater Tree by leveraging the properties of the BST and managing the cumulative sums in a spirit of efficiency and clarity.

# 1038. Binary Search Tree to Greater Sum Tree

### Problem Description 
Given the `root` of a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus sum of all keys greater than the original key in BST.

As a reminder, a binary search tree is a tree that satisfies these constraints:
The left subtree of a node contains only nodes with keys less than the node's key.

The right subtree of a node contains only nodes with keys greater than the node's key.

Both the left and right subtrees must also be binary search trees.

Note: This question is the same as 538: https://leetcode.com/problems/convert-bst-to-greater-tree/

Example 1:
Input: root = [4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
Output: [30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]

Example 2:
Input: root = [0,null,1]
Output: [1,null,1]

Example 3:
Input: root = [1,0,2]
Output: [3,3,2]

Example 4:
Input: root = [3,2,4,1]
Output: [7,9,4,10]

Constraints:
The number of nodes in the tree is in the range `[1, 100]`.

`0 <= Node.val <= 100`
All the values in the tree are unique.

`root` is guaranteed to be a valid binary search tree.

### Solution 
 To solve the problem of converting a Binary Search Tree (BST) into a Greater Tree, we will leverage the properties of the BST and perform a reverse in-order traversal (right-root-left) which allows us to efficiently accumulate sums of greater values.

### Detailed Explanation:

1. **In-Order Traversal**: In a BST, an in-order traversal outputs values in increasing order. For the Greater Tree, we need the values in decreasing order, so we will perform a reverse in-order traversal.

2. **Accumulate Greater Values**: During the traversal, we maintain a cumulative sum of the values encountered. When we visit a node, we add this cumulative sum to the node's value.

3. **Tree Structure**: We will use a recursive helper function to conduct the traversal. We need to keep track of an accumulator variable to hold the running sum of the greater values.

### Python Implementation

Here is the Python code that implements the described approach. It follows the LeetCode format, and is structured so that it can run directly on the LeetCode platform:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        # Initialize the cumulative sum variable
        self.cumulative_sum = 0
        
        # Helper function to perform reverse in-order traversal
        def traverse(node):
            if not node:
                return
            
            # Traverse the right subtree first (greater values)
            traverse(node.right)
            
            # Update the cumulative sum and the current node's value
            self.cumulative_sum += node.val
            node.val = self.cumulative_sum
            
            # Finally, traverse the left subtree
            traverse(node.left)
        
        # Start the traversal from the root
        traverse(root)
        return root

```

### Explanation of the Code:

1. **TreeNode Class**: This class defines the structure of each node in the binary tree.

2. **Solution Class**: 
   - **convertBST Method**: This is the main method that will be invoked. It initializes the `cumulative_sum` to zero and calls the recursive `traverse` function.
   - **traverse Function**: It takes a node as an argument and:
     - Returns if the node is None (base case).
     - Recursively visits the right child (greater values).
     - Updates the `cumulative_sum` with the current node's value and assigns it to the node's value.
     - Recursively visits the left child (lesser values).

### Complexity Analysis:

- **Time Complexity**: O(N), where N is the number of nodes in the tree. We traverse each node exactly once.
- **Space Complexity**: O(H), where H is the height of the tree. This is due to the recursion stack space. In the worst case (a skewed tree), this can be O(N), but in a balanced tree, it will be O(log N).

This code meets the problem requirements and utilizes the properties of BSTs effectively to produce the desired Greater Tree representation.

# 1214. Two Sum BSTs

### Problem Description 
Given the roots of two binary search trees, `root1` and `root2`, return `true` if and only if there is a node in the first tree and a node in the second tree whose values sum up to a given integer `target`.


Example 1:
Input: root1 = [2,1,4], root2 = [1,0,3], target = 5
Output: true
Explanation: 2 and 3 sum up to 5.


Example 2:
Input: root1 = [0,-10,10], root2 = [5,1,7,0,2], target = 18
Output: false

Constraints:
The number of nodes in each tree is in the range `[1, 5000]`.

`-109 <= Node.val, target <= 109`

### Solution 
 To solve the problem of finding two nodes (one from each of the two binary search trees) that sum up to a given target, we can utilize the properties of binary search trees (BSTs) and a complementary approach with a hash set.

### Plan
1. **In-Order Traversal**: We traverse the first BST (`root1`) and store all its values in a hash set. Since BSTs are ordered, this traversal allows us to efficiently check for required values later.
2. **Searching for Complement**: Next, we traverse the second BST (`root2`). For each value we encounter in `root2`, we compute the complement that would yield the target sum when added to the BST values found in `root1`. We check if this complement exists in our hash set.
3. **Output**: If we find a match during the traversal of `root2`, we return `true`; if we finish both traversals without finding any such pair, we return `false`.

### Complexity
- Time Complexity: O(N + M), where N and M are the number of nodes in `root1` and `root2`, respectively.
- Space Complexity: O(N), due to the hash set used to store values from the first BST.

### Implementation
Here's the Python code that implements the above logic, formatted for LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def findTarget(self, root1: TreeNode, root2: TreeNode, target: int) -> bool:
        # Helper function to perform in-order traversal and store values in a set
        def inorder(node, values):
            if not node:
                return
            inorder(node.left, values)
            values.add(node.val)
            inorder(node.right, values)
        
        # Step 1: Store all values from the first tree in a hash set
        values_in_root1 = set()
        inorder(root1, values_in_root1)
        
        # Step 2: Search for the complement in the second tree
        def search(root):
            if not root:
                return False
            complement = target - root.val
            if complement in values_in_root1:
                return True
            return search(root.left) or search(root.right)

        # Perform the search in the second tree
        return search(root2)

```

### Explanation:
1. **TreeNode Class**: This defines the structure of nodes in the binary tree.
2. **Solution Class**: Contains the function `findTarget` that will implement our solution.
    - The `inorder()` function is a recursive helper that performs in-order traversal on the first tree to collect node values in a set.
    - The `search()` function checks the second tree and looks for the required complement of each node's value.
3. **Final Return**: The `findTarget` function orchestrates the process and returns whether such a pair exists.

You can run this solution directly on LeetCode by pasting it into the submission area for the problem. Make sure to correctly define the inputs for `root1`, `root2`, and `target` according to the format provided in the problem statement.

# 1373. Maximum Sum BST in Binary Tree

### Problem Description 
Given a binary tree `root`, the task is to return the maximum sum of all keys of any sub-tree which is also a Binary Search Tree (BST).

Assume a BST is defined as follows:
The left subtree of a node contains only nodes with keys less than the node's key.

The right subtree of a node contains only nodes with keys greater than the node's key.

Both the left and right subtrees must also be binary search trees.


Example 1:
Input: root = [1,4,3,2,4,2,5,null,null,null,null,null,null,4,6]
Output: 20
Explanation: Maximum sum in a valid Binary search tree is obtained in root node with key equal to 3.


Example 2:
Input: root = [4,3,null,1,2]
Output: 2
Explanation: Maximum sum in a valid Binary search tree is obtained in a single root node with key equal to 2.


Example 3:
Input: root = [-4,-2,-5]
Output: 0
Explanation: All values are negatives. Return an empty BST.


Example 4:
Input: root = [2,1,3]
Output: 6

Example 5:
Input: root = [5,4,8,3,null,6,3]
Output: 7

Constraints:
The given binary tree will have between `1` and `40000` nodes.

Each node's value is between `[-4 * 10^4 , 4 * 10^4]`.

### Solution 
 To solve the problem of finding the maximum sum of all keys in any subtree that forms a Binary Search Tree (BST), we can use a recursive depth-first search (DFS) approach. Here's a detailed explanation of the algorithm we will implement:

### Explanation:

1. **Definition of BST**: We need to ensure that for each node in a subtree:
   - All nodes in the left subtree must be less than the node’s value.
   - All nodes in the right subtree must be greater than the node’s value.

2. **Total Sum Calculation**: While traversing the tree, we will calculate the sum of the keys in each subtree and check if it forms a valid BST. If it is a valid BST, we will compare its sum with the current maximum sum encountered.

3. **Utilizing Recursion**: We will use recursion to navigate through the tree:
   - For each node, we will gather information from its left and right children.
   - We'll check whether the left and right subtrees are valid BSTs and whether they fall within the acceptable value range defined by the current node's value.

4. **Return Values**: For each subtree rooted at a node, we will return:
   - Whether that subtree is a BST.
   - The sum of the node values in that subtree if it is a BST.
   - The minimum and maximum values in that subtree to help validate the BST condition.

5. **Base Cases**: We handle the smallest cases (leaf nodes or `None`) separately to avoid errors in calculations.

### Implementation:

Now we can implement this logic in Python, following the LeetCode solution format.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxSumBST(self, root: TreeNode) -> int:
        self.max_sum = 0

        def findMaxBST(node):
            if not node:
                return (True, 0, float('inf'), float('-inf'))
            
            left_is_bst, left_sum, left_min, left_max = findMaxBST(node.left)
            right_is_bst, right_sum, right_min, right_max = findMaxBST(node.right)

            # Check if the current node forms a BST
            if left_is_bst and right_is_bst and (node.val > left_max and node.val < right_min):
                # Compute the total sum of this BST
                current_sum = node.val + left_sum + right_sum
                # Update the maximum sum found so far
                self.max_sum = max(self.max_sum, current_sum)
                
                # Return values for the current node
                return (True, current_sum, min(left_min, node.val), max(right_max, node.val))
            else:
                # If it does not form a BST, return False with appropriate sums and min/max values
                return (False, 0, 0, 0)

        findMaxBST(root)
        return self.max_sum

```

### Explanation of the Code:

- We define a class `TreeNode` to represent each node in the binary tree.
- The `Solution` class contains the method `maxSumBST`, where `self.max_sum` is initialized to 0 to store the maximum sum found.
- The inner function `findMaxBST()` performs the recursive checks for each node:
  - It checks base conditions first.
  - It gathers information about the left and right children.
  - It checks if the current subtree is a BST.
  - If it is a BST, it calculates the sum and updates `self.max_sum`.
  - It returns whether this subtree is a BST, its sum, the minimum value, and the maximum value.

Finally, after the function executes, `maxSumBST` returns the maximum sum of any valid BST subtree from the given binary tree.

# 1382. Balance a Binary Search Tree

### Problem Description 
Given a binary search tree, return a balanced binary search tree with the same node values.

A binary search tree is balanced if and only if the depth of the two subtrees of every node never differ by more than 1.

If there is more than one answer, return any of them.


Example 1:
Input: root = [1,null,2,null,3,null,4,null,null]
Output: [2,1,3,null,null,null,4]
Explanation: This is not the only correct answer, [3,1,4,null,2,null,null] is also correct.


Constraints:
The number of nodes in the tree is between `1` and `10^4`.

The tree nodes will have distinct values between `1` and `10^5`.

### Solution 
 To tackle the problem of converting a given binary search tree (BST) into a balanced binary search tree (BST) with the same node values, we will follow these steps:

1. **Traversal of the BST:** We need to extract the node values from the given BST. In a BST, an in-order traversal will give us the values in sorted order.

2. **Building a Balanced BST:** Using the sorted list of values, we can recursively build a balanced BST. The middle element of the list will serve as the root of the BST. The elements to the left will form the left subtree, and the elements to the right will form the right subtree.

3. **Returning the Result:** Finally, we will construct the tree by creating nodes and linking them based on our recursive structure.

Here's how we can implement this in Python, following the format typically used in LeetCode submissions:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def balancedBST(self, root: TreeNode) -> TreeNode:
        # Step 1: Perform in-order traversal to extract the values in sorted order
        values = []
        self.in_order_traversal(root, values)

        # Step 2: Build a balanced BST from the sorted values
        return self.build_balanced_bst(values)

    def in_order_traversal(self, node: TreeNode, values: List[int]):
        if not node:
            return
        self.in_order_traversal(node.left, values)
        values.append(node.val)
        self.in_order_traversal(node.right, values)

    def build_balanced_bst(self, values: List[int]) -> TreeNode:
        if not values:
            return None
        mid = len(values) // 2
        root = TreeNode(values[mid])  # The middle element becomes the root
        root.left = self.build_balanced_bst(values[:mid])  # Left subtree
        root.right = self.build_balanced_bst(values[mid + 1:])  # Right subtree
        return root

```

### Explanation of the Code:

- **TreeNode Class:** This defines the structure of each node in our binary tree, initialized with value, left child, and right child.

- **Solution Class:** This is where we implement the solution. The main method to be called is `balancedBST`, which:
  - Calls a helper function `in_order_traversal` to fill `values` list with the node values in sorted order.
  - Calls another helper function `build_balanced_bst` to construct the balanced BST from the sorted list.

- **In-Order Traversal:** This method recursively visits the left child, appends the current node value, and then visits the right child. This ensures that we collect values in sorted order.

- **Building the Balanced BST:** This method takes the sorted list of values and recursively builds the BST. It finds the middle index, creates a `TreeNode` from the middle value, and recursively creates the left and right subtrees from the left and right segments of the list.

### Complexity Analysis:
- **Time Complexity:** The time complexity of the algorithm is O(n) where n is the number of nodes in the tree, due to the in-order traversal and the recursive tree construction.
- **Space Complexity:** The space complexity is O(n) for storing the values, plus O(h) for the recursive call stack where h is the height of the tree.

This solution is efficient and suitable for the given constraints. It will run correctly on LeetCode as per the expected format.

