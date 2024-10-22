# 2. Add Two Numbers

### Problem Description 
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.


Example 1:
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.


Example 2:
Input: l1 = [0], l2 = [0]
Output: [0]

Example 3:
Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]

Constraints:
The number of nodes in each linked list is in the range `[1, 100]`.

`0 <= Node.val <= 9`
It is guaranteed that the list represents a number that does not have leading zeros.

### Solution 
 To solve the problem of adding two numbers represented as linked lists in reverse order, we'll define a `ListNode` class to represent each node in the linked list and then implement a function to add the two numbers together. Below is the solution formatted according to the LeetCode standards.

### Step-by-step Explanation

1. **Class Definition**:
   We'll define a `ListNode` class that represents a single node in the linked list. This will allow us to create our linked list structure.

2. **Function Implementation**:
   We will implement a function called `addTwoNumbers` that takes two linked lists (`l1` and `l2`) as input. The function will traverse both linked lists, adding the respective digits along with the carry from the previous addition. It will create a new linked list to store the result.

3. **Adding Digits**:
   - We'll initialize a `dummy_head` to help build the resultant linked list easily and a variable `carry` to handle sums that exceed 9.
   - We'll loop through both linked lists until we have processed all nodes and any remaining carry.
   - For each pair of corresponding nodes (or 0 if one list is shorter), we calculate the sum, update the carry for the next position, and create a new node with the result's last digit.

4. **Return the Result**:
   Finally, we'll return the linked list starting from the next node of the dummy head (to skip the initial dummy node).

### Python Code

Here's the Python implementation in the format suitable for LeetCode:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # Create a dummy head to ease the result list construction
        dummy_head = ListNode(0)
        current = dummy_head
        carry = 0
        
        # Traverse both lists
        while l1 is not None or l2 is not None or carry:
            # Get the values from the current nodes, default is 0 if none
            val1 = (l1.val if l1 is not None else 0)
            val2 = (l2.val if l2 is not None else 0)
            
            # Calculate the sum
            total = val1 + val2 + carry
            carry = total // 10  # Update carry for next iteration
            current.next = ListNode(total % 10)  # Create the new node
            
            # Move current pointer to the next node
            current = current.next
            
            # Advance l1 and l2 if they are not None
            if l1 is not None:
                l1 = l1.next
            if l2 is not None:
                l2 = l2.next

        # The next of dummy head is the resultant linked list
        return dummy_head.next

```

### Key Points:
- **Node Creation**: We initialize a dummy node to facilitate the construction of the result linked list.
- **Carry Management**: We handle the carry from additions greater than 9 properly.
- **Avoid Null Pointer Dereference**: We check for None on linked list nodes to prevent errors.
- **Efficiency**: The time complexity is O(max(N, M)), where N and M are the lengths of the two lists, and the space complexity is O(max(N, M)) for the resulting linked list.

This solution adheres to the LeetCode format and can be directly run as a function on that platform.

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
 To solve the problem of generating all possible letter combinations that a digit string can represent, we can utilize a backtracking algorithm. This approach is efficient for this problem given the constraints (the maximum length of the input string is 4), and allows us to explore all possible combinations of the letters that correspond to the digits.

### Step-by-Step Explanation

1. **Mapping Digits to Letters**: We first need to create a mapping of digits to the corresponding letters, similar to how old mobile phones mapped numbers to letters. The mapping is as follows:
   - 2 -> "abc"
   - 3 -> "def"
   - 4 -> "ghi"
   - 5 -> "jkl"
   - 6 -> "mno"
   - 7 -> "pqrs"
   - 8 -> "tuv"
   - 9 -> "wxyz"

2. **Backtracking**: We will define a recursive backtracking function that:
   - Takes the current index in the `digits` string and builds combinations by appending letters one by one until we reach the end of the string.
   - If we reach the end of the digits string, we add the current combination to a results list.

3. **Handling Edge Cases**: If the input string is empty, we should return an empty list.

Let's put all this together in a Python function formatted for LeetCode.

### Python Code



```python
class Solution:
    def letterCombinations(self, digits: str) -> list[str]:
        if not digits:
            return []

        # Mapping of digits to letters
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

        results = []

        def backtrack(index: int, path: str):
            # If the path length is equal to the length of digits, we found a combination
            if index == len(digits):
                results.append(path)
                return
            
            # Get the current digit and its corresponding letters
            current_digit = digits[index]
            possible_letters = digit_to_letters[current_digit]
            
            # Explore all possible letters for the current digit
            for letter in possible_letters:
                # Append the letter to the current path and proceed to the next digit
                backtrack(index + 1, path + letter)

        # Start backtracking from the first digit
        backtrack(0, "")
        return results

```

### Explanation of the Code:

- **Initialization**: We check if the input `digits` is empty and return an empty list if it is. We also set up our mapping from digits to letters.
- **Backtrack Function**: This inner function (`backtrack`) is responsible for building up combinations:
  - It starts with the current `index` and `path` (the current combination of letters being built).
  - When the current `index` equals the length of `digits`, it means we have built a complete combination, and we append it to the `results`.
  - We fetch the possible letters corresponding to the current digit and iterate over them, calling `backtrack` recursively for the next index.
- **Result Collection**: Finally, we return the accumulated results after exploring all combinations.

This algorithm efficiently explores all combinations through recursion, making it suitable given the constraints. Each digit contributes a unique set of letters, and through backtracking, all combinations are generated systematically.

# 21. Merge Two Sorted Lists

### Problem Description 
Merge two sorted linked lists and return it as a sorted list. The list should be made by splicing together the nodes of the first two lists.


Example 1:
Input: l1 = [1,2,4], l2 = [1,3,4]
Output: [1,1,2,3,4,4]

Example 2:
Input: l1 = [], l2 = []
Output: []

Example 3:
Input: l1 = [], l2 = [0]
Output: [0]

Constraints:
The number of nodes in both lists is in the range `[0, 50]`.

`-100 <= Node.val <= 100`
Both `l1` and `l2` are sorted in non-decreasing order.

### Solution 
 To solve the problem of merging two sorted linked lists, we need to create a new sorted linked list that contains all elements from both input linked lists while preserving their order. The approach is straightforward:

1. **Initialize a Dummy Node**: We can simplify the process of merging by using a dummy node, which acts as a placeholder for the head of the merged list. This avoids special checks for the head node during the merging process.

2. **Use Two Pointers**: Maintain two pointers (or references), one for each list (let's call them `l1` and `l2`). Traverse both lists, comparing the current nodes pointed to by `l1` and `l2`, and append the smaller node to the new merged list.

3. **Advance the Pointers**: After adding the smaller node to the merged list, advance the pointer of that list (either `l1` or `l2`).

4. **Handle Remaining Nodes**: After exiting the loop, if there are any remaining nodes in either `l1` or `l2`, append them to the merged list.

5. **Return the Merged List**: The merged list is found by returning the next node of the dummy node (since the dummy node itself is just a placeholder).

Here's how to implement this logic in Python, following the LeetCode format for defining the function and working with linked list nodes.



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        # Initialize a dummy node to simplify the merge process
        dummy = ListNode()
        current = dummy  # This pointer will track the end of the merged list
        
        # Loop until either l1 or l2 is exhausted
        while l1 is not None and l2 is not None:
            if l1.val < l2.val:
                current.next = l1  # Link the smaller node to the merged list
                l1 = l1.next  # Move to the next node in l1
            else:
                current.next = l2  # Link the smaller node to the merged list
                l2 = l2.next  # Move to the next node in l2
            current = current.next  # Move the current pointer forward

        # If there are remaining nodes in either l1 or l2, append them
        if l1 is not None:
            current.next = l1
        elif l2 is not None:
            current.next = l2
        
        # The merged list is in the next of the dummy node
        return dummy.next

```

### Explanation

- **Class Definition**: We define a `ListNode` class to represent each node in the linked list, with attributes for the node's value (`val`) and the next node (`next`).
  
- **Merging Logic**: 
  - We create a `dummy` node to simplify merging nodes.
  - A `current` pointer starts at `dummy` and will be used to build the new merged list by linking nodes.
  - In a loop, we compare the values of nodes from both lists (`l1` and `l2`). The smaller node gets added to `current`, and the corresponding pointer is moved ahead.
  - After one of the lists is exhausted, we directly link any remaining nodes from the other list.

- **Return the Result**: Finally, we return `dummy.next`, which points to the head of the newly merged sorted linked list.

This solution efficiently merges two sorted linked lists with a time complexity of O(n + m), where n and m are the lengths of the input lists, making it optimal given the constraints.

# 24. Swap Nodes in Pairs

### Problem Description 
Given a linked list, swap every two adjacent nodes and return its head.


Example 1:
Input: head = [1,2,3,4]
Output: [2,1,4,3]

Example 2:
Input: head = []
Output: []

Example 3:
Input: head = [1]
Output: [1]

Constraints:
The number of nodes in the list is in the range `[0, 100]`.

`0 <= Node.val <= 100`
Follow up: Can you solve the problem without modifying the values in the list's nodes? (i.e., Only nodes themselves may be changed.)

### Solution 
 To solve the problem of swapping every two adjacent nodes in a linked list, we'll follow a structured approach. Below is a Python implementation that adheres to the LeetCode format and directly addresses the problem.

### Python Solution



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        # Create a dummy node to ease the swapping process
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy  # Pointer to the last node of the previous pair

        # Iterate through the list while there are pairs to swap
        while prev.next and prev.next.next:
            # Identifying the two nodes to swap
            first = prev.next
            second = prev.next.next
            
            # Performing the swap
            first.next = second.next  # Connect first to the node after second
            second.next = first       # Connect second to first
            prev.next = second        # Connect the previous pair's last node to second

            # Move the prev pointer two nodes ahead
            prev = first

        # Return the new head which is dummy's next
        return dummy.next

```

### Explanation

1. **ListNode Class Definition**: 
   - The `ListNode` class is defined to represent a node in a linked list. It has two properties: `val` (to hold the node's value) and `next` (to point to the next node in the list).

2. **Solution Class**: 
   - The `Solution` class contains the method `swapPairs` which takes the head of the linked list as an argument.

3. **Dummy Node Initialization**: 
   - A dummy node is created with a value of 0, which helps simplify edge cases (like swapping the head of the list). The `dummy.next` is set to the input `head`.

4. **Prev Pointer**: 
   - We use a `prev` pointer which initially points to the dummy node. It helps track the last node of the previous pair.

5. **Swapping Logic**: 
   - We loop through the linked list and check if there are at least two nodes left to swap (`prev.next` and `prev.next.next`).
   - We identify the two nodes we need to swap: `first` (the first node of the pair) and `second` (the second node of the pair).
   - The swapping is done by rearranging the `next` pointers:
     - `first.next` is connected to the node after `second`.
     - `second.next` is connected back to `first`.
     - Finally, `prev.next` is updated to point to `second`, linking it correctly in the larger list.

6. **Moving the Prev Pointer**: 
   - After swapping, we move `prev` two nodes ahead to continue checking for more pairs.

7. **Return Statement**: 
   - At the end of the process, we return `dummy.next`, which points to the new head of the modified linked list.

### Complexity Analysis:
- **Time Complexity**: O(n), where n is the number of nodes in the linked list since we traverse the list once.
- **Space Complexity**: O(1), since we are using a constant amount of extra space.

This solution efficiently handles swapping adjacent nodes without modifying the values, ensuring that the structure of the linked list is preserved.

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
 To solve the problem of finding the maximum depth of a binary tree, we'll follow a typical depth-first search (DFS) approach. The maximum depth can be defined as the longest path from the root node down to the farthest leaf node.

### Explanation

1. **Understanding the Problem**: 
   - The binary tree is represented using nodes, where each node can have a left child and a right child.
   - A leaf node is a node that does not have any children (both left and right child are `None`).
   - The depth of the tree is the number of nodes along the longest path from the root down to a leaf.

2. **Base Cases**:
   - If the `root` is `None`, the maximum depth is `0` because there are no nodes.
   - If the `root` is not `None`, we will recursively calculate the depth of its left and right subtrees.

3. **Recursive Calculation**:
   - For each node, we will calculate the maximum depth of its left and right children. 
   - The depth of the current node will be `1 + max(depth of left subtree, depth of right subtree)`.
   - This means we will add one for the current node itself and take the maximum of the depths of the two subtrees.

4. **Implementation using Depth-First Search**:
   - We can implement this using a recursive function that traverses the tree.

### Python Code

The following Python code implements the above logic and adheres to the LeetCode solution format. 

Here’s the complete implementation:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:  # Base case: if the node is None
            return 0
        
        # Recursive case: compute the depth of each subtree
        left_depth = self.maxDepth(root.left)  # depth of left subtree
        right_depth = self.maxDepth(root.right)  # depth of right subtree
        
        # Return the max depth between the two subtrees + 1 for the current node
        return 1 + max(left_depth, right_depth)

```

### How to Run This Code on LeetCode

1. Copy the above code to the code editor on LeetCode.
2. Ensure that the input format matches the problem description when you're testing. LeetCode will automatically provide the `TreeNode` structure to represent the binary tree when you run the tests.
3. Click on "Run Code" or "Submit" once you are satisfied with the solution.

### Complexity Analysis

- **Time Complexity**: The time complexity of this solution is O(N), where N is the total number of nodes in the tree, since we visit each node exactly once.
- **Space Complexity**: The space complexity is O(H), where H is the height of the tree. This accounts for the space used by the recursion stack. In the worst case of a skewed tree, H could be equal to N.

This solution efficiently finds the maximum depth of a binary tree using a recursive approach.

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
 To determine if a binary tree is height-balanced, we can utilize a depth-first search approach. The idea is to traverse the tree and calculate the height of each subtree. While calculating the height, we will also check if the left and right subtrees of each node differ in height by more than 1.

### Detailed Explanation:

1. **Definition of Height-Balanced**: A binary tree is height-balanced if at every node, the difference in height between its left and right subtrees is no more than 1.

2. **Height Calculation**: We need to compute the height of a node, which is defined as the number of edges in the longest path from that node to a leaf node. A leaf node has a height of 0.

3. **Postorder Traversal**: Using postorder traversal (left-right-root), we can calculate the height of each subtree and determine if the tree is balanced at the same time.

4. **Balanced Check**: During the height computation, if we find that any subtree is unbalanced (difference in heights > 1), we can propagate that information upward and return early.

### Implementation Steps:
- We create a helper function that returns the height of a subtree and checks if it is balanced.
- If the subtree rooted at any node is unbalanced, we return a special value indicating this.
- Finally, in the main function, we call this helper function on the root and check if the tree is balanced.

### LeetCode Solution Format:
Here is the Python code that implements the above logic:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        
        def checkHeight(node):
            # Base case: if node is None, it's height is 0
            if not node:
                return 0
            
            # Recursively check the height of left and right subtrees
            left_height = checkHeight(node.left)
            right_height = checkHeight(node.right)
            
            # If left subtree is unbalanced, propagate -1 upwards
            if left_height == -1:
                return -1
            
            # If right subtree is unbalanced, propagate -1 upwards
            if right_height == -1:
                return -1
            
            # If the current node is unbalanced, return -1
            if abs(left_height - right_height) > 1:
                return -1
            
            # Return the height of the current node
            return max(left_height, right_height) + 1
            
        # Initiate the check from the root and check if the height function returns -1
        return checkHeight(root) != -1

```

### Explanation of the Code:
1. **TreeNode Class**: This defines the structure of the tree nodes with a value, a left child, and a right child.
2. **Solution Class**: Contains the method `isBalanced`.
3. **checkHeight Function**: 
   - It returns the height of the subtree if balanced; otherwise, it returns `-1`.
   - It checks for the base case when the node is `None`.
   - It computes heights for the left and right children and checks the balance condition.
4. **Return Statement**: The main function calls `checkHeight` on the root and checks for the return value to determine if the tree is balanced.

### Conclusion:
This solution efficiently checks if a binary tree is height-balanced with a time complexity of O(n), where n is the number of nodes in the tree, as it processes each node once. It uses O(h) space for the call stack in the case of a skewed tree, where h is the height of the tree.

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
 To solve the problem of finding the maximum path sum in a binary tree, we can use a depth-first search (DFS) approach. The idea is to traverse the tree and calculate the maximum path sum that can be obtained through each node, while keeping track of the global maximum path sum found so far.

### Key Points:
1. A path is defined as any sequence of nodes where each pair of adjacent nodes has an edge connecting them.
2. A node can be included in multiple paths; however, once we include a node in a path sum, we cannot include it again in that path.
3. The maximum path sum is defined as the highest sum that can be obtained by any valid path in the tree.

### Approach:
- We'll perform a depth-first search (DFS) from each node, calculating two things:
  - The maximum path sum that includes the node and extends to one of its children (which can be returned to the parent node).
  - The maximum path sum found starting from that node, which could include both children and the node itself (used to update the global maximum path sum).

### Steps:
1. Define a recursive function that will return the maximum path sum extending upward from the current node to its parent.
2. In this function, update the global maximum path sum using the current node's value and the maximum path sums of its left and right subtrees.
3. Return the maximum path sum extending from the current node to its parent.

### Implementation:
Here is the Python code implementing the above logic:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        self.max_sum = float('-inf')  # Initialize the global maximum to negative infinity

        def dfs(node: TreeNode) -> int:
            if not node:
                return 0  # Base case: If the node is None, return 0

            # Compute the maximum path sum from the left and right children
            left_max = max(dfs(node.left), 0)  # Only consider positive contributions
            right_max = max(dfs(node.right), 0)  # Only consider positive contributions

            # Calculate the maximum path sum including the current node
            current_path_sum = node.val + left_max + right_max
            
            # Update the global maximum path sum found
            self.max_sum = max(self.max_sum, current_path_sum)

            # Return the maximum sum that can be extended to the parent
            return node.val + max(left_max, right_max)

        dfs(root)  # Start the DFS from the root
        return self.max_sum  # Return the maximum path sum found

```

### Explanation:
- **TreeNode Class**: Represents a node in the binary tree.
- **Solution Class**: Contains the method `maxPathSum` which computes the maximum path sum.
- **maxPathSum Method**: Initializes the `max_sum` to negative infinity and calls the helper `dfs` function.
- **dfs Function**:
  - If the node is `None`, it returns 0.
  - It computes the maximum path sums for the left and right children (if they are positive).
  - It calculates the `current_path_sum` that includes the current node and its children.
  - Updates the `max_sum` with the maximum value found so far.
  - Returns the maximum sum to be used by its parent (includes the current node's value and the higher of the left or right paths).

### Complexity:
- **Time Complexity**: O(N), where N is the number of nodes in the tree because we visit each node once.
- **Space Complexity**: O(H), where H is the height of the tree, due to the recursion stack. In the worst case of a skewed tree, this can be O(N).

This implementation adheres to LeetCode's requirements and will run successfully on their platform.

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
 To solve the problem of returning the visible values of the nodes from the right side of a binary tree, we can perform a level-order traversal (BFS) while keeping track of the last node at each level since that node will be the one visible from the right side. Here's how to approach the problem:

### Step-by-Step Explanation

1. **Understanding the Input and Output**:
   - We are working with a binary tree represented by the `root`.
   - The output should be a list of node values that are visible from the right side of the tree, ordered from top to bottom.

2. **Using Level-Order Traversal**:
   - We’ll use a queue to facilitate the level-order traversal.
   - For each level of the tree, we'll store the last node's value that we encounter as this corresponds to the rightmost visible node for that level.

3. **Algorithm**:
   - Initialize an empty list to store the visible values.
   - Use a queue to perform a breadth-first search. Start by adding the root node to the queue.
   - For each level, iterate through the nodes, and while processing, keep track of the last node's value as you traverse each level.
   - After completing the traversal for a level, append the last node's value to the result list.
   - Continue this until all levels of the tree have been processed.

4. **Edge Cases**:
   - If the tree is empty (i.e., `root` is `None`), return an empty list.

Now, let's implement this in Python in a format suitable for LeetCode.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:  # Edge case for an empty tree
            return []

        result = []
        queue = [root]

        while queue:
            level_size = len(queue)  # Number of nodes at the current level
            for i in range(level_size):
                node = queue.pop(0)  # Dequeue the front node
                # If this is the last node in the level, add it to result
                if i == level_size - 1:
                    result.append(node.val)
                # Enqueue left and right children
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

        return result  # Final list containing right side view

```

### Explanation of the Code:
- We define a `TreeNode` class to represent the nodes of the binary tree.
- The `rightSideView` method takes the root of the binary tree and returns a list of values seen from the right.
- We check if the root is `None`. If so, we return an empty list.
- We use a queue to process the nodes level by level.
- For each level, we determine how many nodes are present (`level_size`). As we process each node, if it’s the last node for that level (i.e., the rightmost node), we append its value to the result list.
- Finally, we return the result list which contains the values seen from the right side of the tree.

This code should work directly on LeetCode and handles all edge cases as specified in the problem description.

# 247. Strobogrammatic Number II

### Problem Description 
Given an integer `n`, return all the strobogrammatic numbers that are of length `n`. You may return the answer in any order.

A strobogrammatic number is a number that looks the same when rotated `180` degrees (looked at upside down).


Example 1:
Input: n = 2
Output: ["11","69","88","96"]

Example 2:
Input: n = 1
Output: ["0","1","8"]

Constraints:
`1 <= n <= 14`

### Solution 
 To solve the problem of generating strobogrammatic numbers of a given length `n`, we need to understand the properties of strobogrammatic numbers. A strobogrammatic number remains the same when rotated 180 degrees. The valid pairs of digits that can produce this effect are:

- (0, 0)
- (1, 1)
- (8, 8)
- (6, 9)
- (9, 6)

Given these pairs, we can deduce the approach to generate strobogrammatic numbers of length `n`. We can build the numbers recursively by constructing the digits from both ends towards the middle.

### Approach:

1. **Base Cases**: 
   - For `n=0`, return `[""]` (the empty string).
   - For `n=1`, return `["0", "1", "8"]`.

2. **Recursion**:
   - For lengths greater than 1, recursively generate strobogrammatic numbers of length `n-2` and then "wrap" them using the pairs mentioned above.
   - To ensure valid strobogrammatic numbers, we need to be cautious with the first digit. If the number is being constructed for lengths greater than 1, we should avoid leading zeros.

3. **Building the Result**:
   - For each length `n`, combine valid pairs with the results from the previous recursion step.

By employing this recursive strategy, we can efficiently generate the required numbers.

Here’s the Python code implementing this strategy in the format acceptable for LeetCode:



```python
class Solution:
    def strobogrammatic(self, n: int) -> List[str]:
        # Base cases for lengths 0 and 1
        if n == 0:
            return [""]
        if n == 1:
            return ["0", "1", "8"]
        
        # Recursive function to generate strobogrammatic numbers
        def helper(length: int) -> List[str]:
            if length == 0:
                return [""]
            if length == 1:
                return ["0", "1", "8"]
            
            # Generate strobogrammatic numbers of length - 2
            smaller_pairs = helper(length - 2)
            result = []
            for number in smaller_pairs:
                # Add valid pairs to the front and back
                if length != n:  # Avoid leading zero case
                    result.append("0" + number + "0")
                result.append("1" + number + "1")
                result.append("8" + number + "8")
                result.append("6" + number + "9")
                result.append("9" + number + "6")
            return result
        
        return helper(n)

```

### Explanation of the Code:

- We define a `Solution` class with the `strobogrammatic` method, which accepts an integer `n` as an argument.
- We handle base cases directly for lengths 0 and 1.
- The `helper` function recursively builds strobogrammatic numbers:
  - If the current length is 0 or 1, it returns corresponding strobogrammatic numbers.
  - It calculates strobogrammatic numbers for `length - 2`, then builds new numbers by wrapping these smaller numbers with valid pairs.
  - We ensure we can't start with '0' unless the resulting length is strictly 0 (to avoid leading zeros in numbers longer than 1).
- Finally, we return the result of calling `helper(n)`.

This solution efficiently generates all strobogrammatic numbers of the specified length.

# 248. Strobogrammatic Number III

### Problem Description 
Given two strings low and high that represent two integers `low` and `high` where `low <= high`, return the number of strobogrammatic numbers in the range `[low, high]`.

A strobogrammatic number is a number that looks the same when rotated `180` degrees (looked at upside down).


Example 1:
Input: low = "50", high = "100"
Output: 3

Example 2:
Input: low = "0", high = "0"
Output: 1

Constraints:
`1 <= low.length, high.length <= 15`
`low` and `high` consist of only digits.

`low <= high`
`low` and `high` do not contain any leading zeros except for zero itself.

### Solution 
 To solve the problem of counting strobogrammatic numbers within the range defined by the strings `low` and `high`, we need to understand both the properties of strobogrammatic numbers and how to efficiently iterate through possible candidates.

### Strobogrammatic Numbers
A strobogrammatic number remains the same when rotated 180 degrees upside down. The valid pairs of digits that can form strobogrammatic numbers are:
- `0` ↔ `0`
- `1` ↔ `1`
- `6` ↔ `9`
- `8` ↔ `8`

### Approach
1. **Recursive Generation**: We can recursively generate all strobogrammatic numbers of lengths between the length of `low` and `high`.
2. **Valid Range Check**: After generating the candidates, check if they fall within the range [`low`, `high`].
3. **Count Valid Strobogrammatic Numbers**: Maintain a count of how many valid numbers fall within the specified range.

### Steps in the Implementation
1. **Generate Strobogrammatic Numbers**: Create a function that uses recursion to build strobogrammatic numbers, starting from an empty string and building outward.
2. **Range Comparison**: Once a number is generated, compare it against `low` and `high` to see if it is valid.
3. **Return the Count**: Finally, return the total count of valid strobogrammatic numbers.

### Python Implementation
Here's how we can implement this in a way suitable for LeetCode:



```python
class Solution:
    def strobogrammaticCountInRange(self, low: str, high: str) -> int:
        def generateStrobogrammatic(n: int, m: int) -> List[str]:
            if n == 0: return [""]   # Base case, no digits
            if n == 1: return ["0", "1", "8"]  # Single digit strobogrammatic numbers
            
            # Recursive generation for strobogrammatic numbers
            prev = generateStrobogrammatic(n - 2, m)
            result = []
            for num in prev:
                if n != m:  # Don't add leading zeros for multi-digit numbers
                    result.append("0" + num + "0")
                result.append("1" + num + "1")
                result.append("6" + num + "9")
                result.append("8" + num + "8")
                result.append("9" + num + "6")
            return result
        
        def is_valid_in_range(num: str, low: str, high: str) -> bool:
            # Check if the number is in range
            return (len(num) > len(low) and len(num) > len(high)) or \
                   (len(num) < len(low) and len(num) < len(high)) or \
                   (len(num) == len(low) and num >= low and num <= high)

        count = 0
        for length in range(len(low), len(high) + 1):
            for num in generateStrobogrammatic(length, length):
                if is_valid_in_range(num, low, high):
                    count += 1

        return count

```

### Explanation of the Code
- **Function `generateStrobogrammatic`**: This recursive function constructs strobogrammatic numbers of a given length `n`. For each recursion, if `n` is greater than 1, it recursively generates strobogrammatic numbers of length `n-2` and results in appending valid pairs to form strobogrammatic numbers.
- **Function `is_valid_in_range`**: This checks if the generated number falls within the bounds defined by `low` and `high`.
- **Counting**: We loop over all lengths from `len(low)` to `len(high)` and generate all possible strobogrammatic numbers for each length. Each valid number is counted.
  
### Complexity
- The time complexity is exponential in terms of the number of strobogrammatic numbers generated, but given the constraints (maximum length of 15), it runs efficiently within acceptable limits.

This algorithm effectively counts strobogrammatic numbers in the specified range using recursion and simple string comparisons, making it suitable for the problem scenario.

# 369. Plus One Linked List

### Problem Description 
Given a non-negative integer represented as a linked list of digits, plus one to the integer.

The digits are stored such that the most significant digit is at the `head` of the list.


Example 1:
Input: head = [1,2,3]
Output: [1,2,4]

Example 2:
Input: head = [0]
Output: [1]

Constraints:
The number of nodes in the linked list is in the range `[1, 100]`.

`0 <= Node.val <= 9`
The number represented by the linked list does not contain leading zeros except for the zero itself.

### Solution 
 To solve the problem of adding one to a non-negative integer represented as a linked list of digits, we can follow these steps:

1. **Reverse the Linked List**: Since we want to add one to the least significant digit, it’s easier to manipulate the list by reversing it first. This way, we can easily traverse from the least significant digit to the most significant.

2. **Add One**: Traverse through the reversed list and add one to the least significant digit (the first node after reversing). If this addition causes a carry (i.e., if it becomes 10), we set the current node’s value to 0 and carry over 1 to the next node.

3. **Handle Remaining Carry**: Continue this process until all nodes are processed. If there is still a carry left after processing all nodes (which only can happen if the original number contains only 9s), we will need to append a new node with value 1.

4. **Reverse the Linked List Back**: Finally, we reverse the linked list again to restore the original order of digits.

Here’s the Python code implementing the above logic, formatted to run on LeetCode:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def plusOne(self, head: ListNode) -> ListNode:
        # Helper function to reverse the linked list
        def reverse_linked_list(node):
            prev = None
            current = node
            while current:
                next_node = current.next
                current.next = prev
                prev = current
                current = next_node
            return prev  # New head of the reversed list
        
        # Step 1: Reverse the linked list
        head = reverse_linked_list(head)

        # Step 2: Add one to the reversed list
        current = head
        carry = 1  # We want to add one
        
        while current and carry:
            current.val += carry
            if current.val == 10:
                current.val = 0
                carry = 1  # Carry remains
            else:
                carry = 0  # No carry, we can stop
            current = current.next
        
        # Step 3: If there's still a carry left, we need to add a new node
        if carry:
            new_node = ListNode(1)
            new_node.next = head
            head = new_node
        
        # Step 4: Reverse back to original order
        head = reverse_linked_list(head)
        
        return head  # Return the new head of the linked list

```

### Explanation:
- **ListNode**: We define a simple class `ListNode` that has properties for value (`val`) and pointing to the next node (`next`).
- **Solution**: This class contains the method `plusOne` which takes the head of the linked list as input and returns a new head after performing the operation.
- **reverse_linked_list**: A helper function that reverses the linked list and returns the new head.
- **Adding Logic**: We process the reversed linked list, adding one while handling any carry overflow.
- **Final Steps**: After processing, if there's any carry left, we create a new node with value 1. Finally, we reverse the linked list again to restore the original order.

The above implementation meets the constraints and efficiently solves the problem within O(n) time complexity, where n is the number of nodes in the linked list. The solution handles edge cases such as the list being all nines effectively.

# 395. Longest Substring with At Least K Repeating Characters

### Problem Description 
Given a string `s` and an integer `k`, return the length of the longest substring of `s` such that the frequency of each character in this substring is greater than or equal to `k`.


Example 1:
Input: s = "aaabb", k = 3
Output: 3
Explanation: The longest substring is "aaa", as 'a' is repeated 3 times.


Example 2:
Input: s = "ababbc", k = 2
Output: 5
Explanation: The longest substring is "ababb", as 'a' is repeated 2 times and 'b' is repeated 3 times.


Constraints:
`1 <= s.length <= 104`
`s` consists of only lowercase English letters.

`1 <= k <= 105`

### Solution 
 To solve the problem of finding the length of the longest substring in a string `s` such that each character in that substring appears at least `k` times, we can utilize a recursive divide-and-conquer approach. The core idea is to partition the string based on characters that do not meet the required frequency and recursively apply the logic on the resulting substrings.

Here’s a detailed explanation of the solution approach:

### Approach:
1. **Frequency Count**: First, we count the frequency of each character in the string using a dictionary.

2. **Partitioning**: We then scan through the string to identify characters that appear less than `k` times. If we find such a character, we treat it as a boundary to split the string into multiple substrings.

3. **Recursive Process**: For each substring created (bounded by the characters that are below the required frequency), we recursively apply the same logic to find the maximum length of substrings where every character meets the frequency requirement.

4. **Base Cases**: If the entire substring has all characters with occurrences greater than or equal to `k`, then we can directly return its length. If no characters meet the criteria, return zero for that substring.

5. **Final Answer**: The answer will be the maximum length obtained from these recursive calls.

### Python Code:
Below is the Python implementation that follows the above approach which you can run directly on the LeetCode platform.



```python
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        # Base case: if the string length is less than k, return 0
        if len(s) < k:
            return 0
        
        # Count frequency of each character
        char_count = {}
        for char in s:
            char_count[char] = char_count.get(char, 0) + 1
        
        # Split the string by characters that do not meet the frequency requirement
        for char in char_count:
            if char_count[char] < k:
                # Recur for all substrings created by this split
                return max(self.longestSubstring(substr, k) for substr in s.split(char))
        
        # If we reach here, all characters meet the requirement
        return len(s)

# Example usage
solution = Solution()
print(solution.longestSubstring("aaabb", 3))  # Output: 3
print(solution.longestSubstring("ababbc", 2))  # Output: 5

```

### Explanation of the Code:
- We define the function `longestSubstring` within the `Solution` class, which is a standard format for LeetCode problems.
- The function starts with a base case check; if the string length is less than `k`, it returns `0`.
- We construct a frequency dictionary to count the occurrences of each character.
- The main loop checks for characters with frequency less than `k`:
  - If found, it splits the string at those characters and applies the function recursively to those substrings.
- If no characters are underrepresented, the entire string is valid, and we return its length.
- Finally, test cases are provided as examples demonstrating how the function works.

This approach efficiently handles the problem while adhering to the constraints and requirements of the prompt.

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
 To solve the problem of converting a Binary Search Tree (BST) to a Greater Tree, we'll utilize a reverse in-order traversal method. This means we will first visit the right subtree, then the current node, and finally the left subtree. This traversal order is crucial because in a BST, the right subtree contains nodes with values greater than the current node.

Here's a step-by-step breakdown of the approach:

1. **Reverse In-Order Traversal**: Start from the root and traverse to the rightmost node first, collecting the cumulative sum of all visited nodes. 

2. **Update Each Node's Value**: As we visit each node during the traversal, we will update the node's value to itself plus the cumulative sum (which represents the sum of all greater nodes).

3. **Maintain a Running Total**: We will keep a running total that starts at 0 and is updated on each visit to a node.

### Python Implementation

Here is a complete implementation in Python that follows the required format for LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def __init__(self):
        self.sum = 0  # Initialize the variable to hold the cumulative sum.
    
    def convertBST(self, root: TreeNode) -> TreeNode:
        # Start the reverse in-order traversal.
        self._reverse_inorder_traversal(root)
        return root
    
    def _reverse_inorder_traversal(self, node: TreeNode):
        if not node:
            return
        
        # Traverse the right subtree first.
        self._reverse_inorder_traversal(node.right)
        
        # Update the sum and the current node's value.
        self.sum += node.val  # Accumulate the current node's value.
        node.val = self.sum   # Update current node's value to be the accumulated sum.
        
        # Traverse the left subtree.
        self._reverse_inorder_traversal(node.left)

```

### Explanation of the Code

- **TreeNode Class**: This defines the structure for the nodes of the tree. Each node has a value and pointers to its left and right children.

- **Solution Class**: We define the `Solution` class, which will contain the methods to perform the conversion.

- **Initialization**: In the `__init__` method, we initialize `self.sum` to zero, which will hold the cumulative sum during the traversal.

- **convertBST Method**: This method starts the reverse in-order traversal by calling `_reverse_inorder_traversal` on the root node. It returns the modified root after processing.

- **_reverse_inorder_traversal Method**: This private method does the following:
  - If the current node is `None`, it simply returns.
  - It first recursively visits the right child (greater values).
  - Updates the running sum with the current node's value and then updates the node's value.
  - Finally, it visits the left child.

### Complexity Analysis
- **Time Complexity**: O(N), where N is the number of nodes in the BST since we visit each node once.
- **Space Complexity**: O(H), where H is the height of the tree, due to the recursive stack. In the worst case of a skewed tree, this could go up to O(N).

This implementation is efficient and leverages the properties of the BST to perform the conversion in a single traversal. It can be directly run on the LeetCode platform as it conforms to the expected input and output format.

# 544. Output Contest Matches

### Problem Description 
During the NBA playoffs, we always arrange the rather strong team to play with the rather weak team, like make the rank 1 team play with the rank nth team, which is a good strategy to make the contest more interesting. Now, you're given n teams, you need to output their final contest matches in the form of a string.

The n teams are given in the form of positive integers from 1 to n, which represents their initial rank. (Rank 1 is the strongest team and Rank n is the weakest team.) We'll use parentheses('(', ')') and commas(',') to represent the contest team pairing - parentheses('(' , ')') for pairing and commas(',') for partition. During the pairing process in each round, you always need to follow the strategy of making the rather strong one pair with the rather weak one.


Example 1:
Input: 2
Output: (1,2)
Explanation: 
Initially, we have the team 1 and the team 2, placed like: 1,2.

Then we pair the team (1,2) together with '(', ')' and ',', which is the final answer.


Example 2:
Input: 4
Output: ((1,4),(2,3))
Explanation: 
In the first round, we pair the team 1 and 4, the team 2 and 3 together, as we need to make the strong team and weak team together.

And we got (1,4),(2,3).

In the second round, the winners of (1,4) and (2,3) need to play again to generate the final winner, so you need to add the paratheses outside them.

And we got the final answer ((1,4),(2,3)).


Example 3:
Input: 8
Output: (((1,8),(4,5)),((2,7),(3,6)))
Explanation: 
First round: (1,8),(2,7),(3,6),(4,5)
Second round: ((1,8),(4,5)),((2,7),(3,6))
Third round: (((1,8),(4,5)),((2,7),(3,6)))
Since the third round will generate the final winner, you need to output the answer (((1,8),(4,5)),((2,7),(3,6))).

Note:
The n is in range [2, 212].

We ensure that the input n can be converted into the form 2k, where k is a positive integer.

### Solution 
 To tackle the problem of pairing teams according to the given rules, we will implement a recursive strategy. The idea is to simulate the matches round by round, where in each round, the strongest available team is paired with the weakest one. The winners of these matchups will then move to the next round, and we repeat the process until we reach a final winner.

Let's break down the approach step-by-step:

1. **Recursive Function**: We'll define a recursive function to create the matches for a given number of teams `n`. This function will take two arguments: `start` and `end`, representing the range of team ranks available for that round.

2. **Base Condition**: If there's only one match to be made (when `start + 1 == end`), we'll return the match in the format `(start, end)`.

3. **Pairing Teams**: For each round, we'll pair the strongest team (represented by `start`) with the weakest team (represented by `end`) and proceed to form pairs for the remaining teams recursively. Specifically, we will:
   - Pair the teams as `(start, end)`.
   - Recursively pair teams in the range from `start + 1` to `end - 1`.

4. **Combining Results**: After forming pairs for the weaker teams, we will combine these match results into a string format, adding parentheses as needed to reflect the pairing structure accurately.

5. **String Formatting**: We need to ensure that the final output string matches the required format, which is achieved by carefully adding parentheses before and after the pairs.

Below is the Python code that implements this solution according to the structure required for LeetCode:



```python
class Solution:
    def generateMatches(self, start: int, end: int) -> str:
        if start + 1 == end:
            # Base case: only one match to make
            return f"({start},{end})"
        
        # Pair the strongest team with the weakest team
        pair = f"({start},{end})"
        
        # Generate matches for the remaining teams
        # Teams left are from start + 1 to end - 1
        next_matches = self.generateMatches(start + 1, end - 1)
        
        # Combine the pairs and return the formatted string
        return f"{pair},{next_matches}"

    def findContestMatches(self, n: int) -> str:
        return self.generateMatches(1, n)

# Example usage:
# sol = Solution()
# print(sol.findContestMatches(4))  # Outputs: ((1,4),(2,3))
# print(sol.findContestMatches(8))  # Outputs: (((1,8),(4,5)),((2,7),(3,6)))

```

### Explanation of Code:
1. **Class Definition**: The `Solution` class contains two methods: `generateMatches` and `findContestMatches`.

2. **`generateMatches` Method**:
   - This method generates the contests recursively. It checks if there's only one match left to form, and if so, returns the pair as a string. Otherwise, it combines the strongest and the weakest teams into a pair, recursively calls itself for the next available teams, and formats the output string.

3. **`findContestMatches` Method**:
   - This method serves as the entry point, starting the match pair generation from team 1 to team `n`.

The given solution efficiently organizes the matchups in a structured format utilizing recursion, and it adheres to the requirement of generating matches in the format specified.

# 563. Binary Tree Tilt

### Problem Description 
Given the `root` of a binary tree, return the sum of every tree node's tilt.

The tilt of a tree node is the absolute difference between the sum of all left subtree node values and all right subtree node values. If a node does not have a left child, then the sum of the left subtree node values is treated as `0`. The rule is similar if there the node does not have a right child.


Example 1:
Input: root = [1,2,3]
Output: 1
Explanation: 
Tilt of node 2 : |0-0| = 0 (no children)
Tilt of node 3 : |0-0| = 0 (no children)
Tilt of node 1 : |2-3| = 1 (left subtree is just left child, so sum is 2; right subtree is just right child, so sum is 3)
Sum of every tilt : 0 + 0 + 1 = 1

Example 2:
Input: root = [4,2,9,3,5,null,7]
Output: 15
Explanation: 
Tilt of node 3 : |0-0| = 0 (no children)
Tilt of node 5 : |0-0| = 0 (no children)
Tilt of node 7 : |0-0| = 0 (no children)
Tilt of node 2 : |3-5| = 2 (left subtree is just left child, so sum is 3; right subtree is just right child, so sum is 5)
Tilt of node 9 : |0-7| = 7 (no left child, so sum is 0; right subtree is just right child, so sum is 7)
Tilt of node 4 : |(3+5+2)-(9+7)| = |10-16| = 6 (left subtree values are 3, 5, and 2, which sums to 10; right subtree values are 9 and 7, which sums to 16)
Sum of every tilt : 0 + 0 + 0 + 2 + 7 + 6 = 15

Example 3:
Input: root = [21,7,14,1,1,2,2,3,3]
Output: 9

Constraints:
The number of nodes in the tree is in the range `[0, 104]`.

`-1000 <= Node.val <= 1000`

### Solution 
 To solve the problem of calculating the sum of every tree node's tilt, we need to perform a depth-first traversal (DFS) of the binary tree. For each node, we will compute its tilt based on the sums of the values of its left and right subtrees. Let's break down the approach and then provide the implementation in Python.

### Approach:

1. **Define the Tree Node**: We will first define our binary tree node class, `TreeNode`, which will represent each node in the binary tree.

2. **Recursive Function**: We will create a function `calculate_tilt(node)` that:
   - Calculates the sum of the subtree rooted at `node`.
   - Computes the tilt of that node using the absolute difference between the sums of its left and right children.
   - Accumulates the total tilt in a class-level variable for final output.

3. **Base Case**: If the current node is `None`, we return a sum of `0`, which would also help in the recursion termination.

4. **Calculation**:
   - Recursively calculate the sum of the left subtree.
   - Recursively calculate the sum of the right subtree.
   - Calculate the tilt for the current node using the formula: `tilt = abs(left_sum - right_sum)`.
   - Update the total tilt with the current node's tilt.

5. **Invocation**: We call the recursive function on the root node and return the accumulated total tilt.

### Implementation:
Here's how we can implement this in Python, formatted in a way that is compatible with LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def __init__(self):
        self.total_tilt = 0  # Variable to hold the total tilt

    def findTilt(self, root: TreeNode) -> int:
        # Helper function to compute subtree sums and tilts
        def calculate_tilt(node):
            if not node:
                return 0
            
            # Calculate sum of left subtree
            left_sum = calculate_tilt(node.left)
            
            # Calculate sum of right subtree
            right_sum = calculate_tilt(node.right)
            
            # Calculate the tilt for the current node
            tilt = abs(left_sum - right_sum)
            self.total_tilt += tilt
            
            # Return the total sum for this subtree
            return left_sum + right_sum + node.val
            
        calculate_tilt(root)  # Start the recursion from the root
        return self.total_tilt  # Return the total calculated tilt

```

### Explanation of the Code:
- **TreeNode Class**: This class defines the structure of each node in the binary tree.
- **Solution Class**: It encapsulates the method to find the total tilt.
- **findTilt Method**: This public method initializes the tilt calculation by calling a recursive helper method.
- **calculate_tilt Method**: This recursive method calculates the sums, updates the tilt, and returns the sum of values for any given node's subtree.
- At the end, the method `findTilt` returns the accumulated tilt value.

### Complexity:
- **Time Complexity**: O(N), where N is the number of nodes in the tree since we visit each node exactly once.
- **Space Complexity**: O(H) for the recursion stack, where H is the height of the tree. In the worst case of a skewed tree, this could be O(N).

This implementation is efficient and handles all constraints specified in the problem.

# 625. Minimum Factorization

### Problem Description 
Given a positive integer `a`, find the smallest positive integer `b` whose multiplication of each digit equals to `a`. 
If there is no answer or the answer is not fit in 32-bit signed integer, then return 0.


Example 1
Input:
48 
Output:
68

Example 2
Input: 
15
Output:
35

### Solution 
 ### Problem Explanation

The problem requires us to take a positive integer `a` and find the smallest positive integer `b` such that the product of its digits equals `a`. If such a `b` cannot be formed or if the result exceeds the range of a 32-bit signed integer, we should return `0`.

To determine `b`, the multiplication of its digits must equal `a`. The approach entails breaking down `a` into its prime factors, ideally in the range of digits, which are 1 through 9. Larger factors need to be broken down until they fit within the range.

### Steps to Solve the Problem

1. **Handle the special case**: If `a` is 0 or 1, we can directly return 10 or 1, respectively.
2. **Factorization**: Start dividing `a` from 9 to 2 (the valid digits) to derive the digits of `b`. We will store the resultant digits in a list.
3. **Rebuild the number**: The collected digits are then sorted in ascending order to get the smallest possible number.
4. **Check constraints**: Finally, we will check if the resulting number exceeds the limits of a 32-bit signed integer.

### Python Code

The following code implements the above logic and is structured according to the LeetCode format.



```python
class Solution:
    def smallestNumber(self, a: int) -> int:
        # Step 1: Handle special cases
        if a == 0:
            return 10  # There is no positive integer whose digits multiply to 0
        if a == 1:
            return 1   # The smallest number whose digits multiply to 1 is 1
        
        # Step 2: Factorization from 9 to 2
        digits = []
        for i in range(9, 1, -1):
            while a % i == 0:
                digits.append(i)
                a //= i
        
        # Step 3: Check if we could fully factor a
        if a > 1:
            return 0  # Not possible to form the number b
        
        # Step 4: Sort digits to form the smallest number
        digits.sort()
        
        # Convert the digits list into a single number
        result = 0
        for digit in digits:
            result = result * 10 + digit
            
        # Step 5: Check if result exceeds 32-bit signed integer
        if result > 2**31 - 1:
            return 0
        
        return result

```

### Explanation of the Code:

1. **Special Cases**:
   - When `a` is `0`, we return `10` as no positive integer can produce a product of `0` through its digits.
   - When `a` is `1`, the answer is simply `1`.
   
2. **Digit Extraction**:
   - We loop from `9` to `2` checking if `a` can be divided by the digit. If it can be divided (`a % i == 0`), we append that digit to the `digits` list and continue dividing `a` by `i`.
   
3. **Feasibility Check**:
   - If after fully processing, `a` is greater than `1`, it means `a` had a prime factor larger than `9` which cannot form a valid digit, hence return `0`.
   
4. **Sorting and Number Formation**:
   - Sorting the `digits` will help create the smallest number possible. We construct the result from the sorted list of digits.
   
5. **Final Validation**:
   - We ensure the derived number does not exceed the maximum limit of a 32-bit signed integer.

### Complexity:
- **Time Complexity**: \(O(\log a)\) since we are continuously dividing `a` by digits from `9` to `2`.
- **Space Complexity**: \(O(1)\) in terms of storage, as the maximum number of digits can be at most 10 (the digits `2` through `9`).

# 669. Trim a Binary Search Tree

### Problem Description 
Given the `root` of a binary search tree and the lowest and highest boundaries as `low` and `high`, trim the tree so that all its elements lies in `[low, high]`. Trimming the tree should not change the relative structure of the elements that will remain in the tree (i.e., any node's descendant should remain a descendant). It can be proven that there is a unique answer.

Return the root of the trimmed binary search tree. Note that the root may change depending on the given bounds.


Example 1:
Input: root = [1,0,2], low = 1, high = 2
Output: [1,null,2]

Example 2:
Input: root = [3,0,4,null,2,null,null,1], low = 1, high = 3
Output: [3,2,null,1]

Example 3:
Input: root = [1], low = 1, high = 2
Output: [1]

Example 4:
Input: root = [1,null,2], low = 1, high = 3
Output: [1,null,2]

Example 5:
Input: root = [1,null,2], low = 2, high = 4
Output: [2]

Constraints:
The number of nodes in the tree in the range `[1, 104]`.

`0 <= Node.val <= 104`
The value of each node in the tree is unique.

`root` is guaranteed to be a valid binary search tree.

`0 <= low <= high <= 104`

### Solution 
 To tackle the problem of trimming a binary search tree based on given boundaries (`low` and `high`), we can utilize a recursive approach that checks each node and determines whether it should be retained in the trimmed tree or discarded. The main idea is to leverage the properties of the binary search tree (BST) where the left child of a node is always less than the node's value, and the right child is always greater.

Here's a streamlined approach to solve the problem:

1. **Base Case**: If the current node is `None`, return `None`. This indicates that we've reached the end of a branch.
  
2. **Check Node Value**:
   - If the value of the current node is less than `low`, then all values in the left subtree will also be less than `low`. Thus, we discard the current node and recursively trim its right subtree (which might contain valid nodes).
   - If the value is greater than `high`, then all values in the right subtree will also be greater than `high`. Thus, we discard the current node and recursively trim its left subtree.
   - If the value is within the range `[low, high]`, we keep the node and need to recursively trim both the left and right subtrees.

3. **Return the trimmed tree**: After performing the above checks, continue returning the new roots of the trimmed left and right subtrees.

The following Python code implements this logic in a format compatible with LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def trimBST(self, root: TreeNode, low: int, high: int) -> TreeNode:
        # Base case: if the current node is None
        if not root:
            return None
        
        # If the current node's value is less than low, discard the left subtree
        if root.val < low:
            return self.trimBST(root.right, low, high)
        
        # If the current node's value is greater than high, discard the right subtree
        if root.val > high:
            return self.trimBST(root.left, low, high)
        
        # If the current node's value is within [low, high], trim both subtrees
        root.left = self.trimBST(root.left, low, high)
        root.right = self.trimBST(root.right, low, high)
        
        return root

```

### Explanation of the Code:

1. **TreeNode Class**: This class defines the structure of a node in the binary tree. Each node has a value, a left child, and a right child.

2. **Solution Class**: This encapsulates the method `trimBST` which performs the trimming operation.
   - The method takes three parameters: the root of the tree and the boundaries `low` and `high`.
   - We check the root node:
     - If it’s `None`, we return `None` (base case).
     - If the root’s value is below `low`, we disregard the left subtree and only consider the right subtree.
     - If the root’s value is above `high`, we disregard the right subtree and only consider the left subtree.
     - If the value is within range, we recursively trim both the left and right children and reassign them to the current node's children.

3. **Returning the Root**: After trimming, the modified root is returned, which maintains the structure of the original tree while fitting the specified conditions.

This algorithm runs in O(n) time complexity, where n is the number of nodes in the tree, as each node is visited once. The space complexity is O(h), where h is the height of the tree due to recursion stack. In the worst case (unbalanced tree), h could be n. In the best case (balanced tree), h would be log(n).

# 687. Longest Univalue Path

### Problem Description 
Given the `root` of a binary tree, return the length of the longest path, where each node in the path has the same value. This path may or may not pass through the root.

The length of the path between two nodes is represented by the number of edges between them.


Example 1:
Input: root = [5,4,5,1,1,5]
Output: 2

Example 2:
Input: root = [1,4,5,4,4,5]
Output: 2

Constraints:
The number of nodes in the tree is in the range `[0, 104]`.

`-1000 <= Node.val <= 1000`
The depth of the tree will not exceed `1000`.

### Solution 
 To solve the problem of finding the longest path in a binary tree where all nodes have the same value, we'll create a function that utilizes a depth-first search (DFS) approach. The goal is to traverse the tree and calculate the longest path length for paths with equal values.

### Plan:

1. **Define the TreeNode Class:** First, we define a simple TreeNode class that represents each node of the binary tree.

2. **Using DFS for Traversal:** Create a helper function that utilizes DFS to explore each node. The function will track paths from the current node to its child nodes that have the same value.

3. **Calculate Path Length:** For each node, determine the longest path length in both the left and right subtrees (children nodes). If the child node has the same value as the current node, it contributes to the path length.

4. **Return the Max Path Length:** Throughout the traversal, keep track of the maximum path length encountered.

5. **Handle Edge Cases:** Make sure to handle the cases where the tree is empty (no nodes) or when there's a single node.

Here's the implementation:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def longestUnivaluePath(self, root: TreeNode) -> int:
        self.max_length = 0  # This will hold the maximum path length found
        
        def dfs(node: TreeNode) -> int:
            if not node:
                return 0  # If the node is None, return 0
            
            # Recursively find the longest path in both left and right children
            left_length = dfs(node.left)
            right_length = dfs(node.right)
            
            # Initialize lengths for left and right sections of the same value
            left_path = right_path = 0
            
            # If the left child is the same value, consider it part of the path
            if node.left and node.left.val == node.val:
                left_path = left_length + 1
            
            # If the right child is the same value, consider it part of the path
            if node.right and node.right.val == node.val:
                right_path = right_length + 1
            
            # Update the maximum path length found (max combination of left and right)
            self.max_length = max(self.max_length, left_path + right_path)
            
            # Return the longest path extending from this node
            return max(left_path, right_path)

        # Start the DFS from the root
        dfs(root)
        return self.max_length

```

### Explanation of the Code:

1. **TreeNode Class:** Defines a binary tree node with a value, a left child, and a right child.
  
2. **Solution Class:** Contains the method `longestUnivaluePath`. 

3. **max_length Variable:** This variable keeps track of the longest path found during the traversal.

4. **dfs Function:** This nested helper function performs the DFS:
   - If the node is None, it returns 0.
   - It recursively retrieves the lengths of the longest univalue paths from the left and right child nodes.
   - It checks if the left or right child has the same value as the current node. If they do, the respective path lengths are incremented.
   - Finally, it updates the `max_length` with the maximum length found at that node and returns the maximum length of paths that can be extended from that node.

5. **Return Value:** After the DFS completes, the method returns the `max_length`.

This implementation is efficient, running in O(N) time complexity, where N is the number of nodes in the tree, as each node is processed exactly once. The space complexity is O(H) due to the recursion stack, where H is the height of the tree.

# 698. Partition to K Equal Sum Subsets

### Problem Description 
Given an array of integers `nums` and a positive integer `k`, find whether it's possible to divide this array into `k` non-empty subsets whose sums are all equal.


Example 1:
Input: nums = [4, 3, 2, 3, 5, 2, 1], k = 4
Output: True
Explanation: It's possible to divide it into 4 subsets (5), (1, 4), (2,3), (2,3) with equal sums.

Note:
`1 <= k <= len(nums) <= 16`.

`0 < nums[i] < 10000`.

### Solution 
 To solve the problem of partitioning the array into `k` subsets with equal sums, we can use a backtracking approach. The main idea is to calculate the total sum of the array and check if it can be evenly divided by `k`. If it can, we will then use backtracking to check if we can form `k` subsets each summing to the target value (which is the total sum divided by `k`).

Here's a step-by-step explanation of how we can implement this:

1. **Calculate Total Sum**: First, we compute the total sum of the array. If the total sum is not divisible by `k`, we cannot divide the array into `k` equal-sum subsets, and we can return `False` immediately.

2. **Determine Target Sum**: If the total sum is divisible by `k`, we define our target sum for each subset. This is simply the total sum divided by `k`.

3. **Backtracking Setup**: We create a list to track the current sum of each of the `k` subsets. We will attempt to fill these subsets with the numbers from the `nums` array.

4. **Recursive Backtracking**: We define a recursive function that tries to place each number in one of the `k` subsets. If placing a number doesn't exceed the target sum for that subset, we can add it and recursively attempt to fill the next subset. If we reach the point where all subsets are filled perfectly, we return `True`.

5. **Base Cases**: If a subset reaches the target sum, we move to the next subset. If we manage to fill all `k` subsets successfully, we return `True`.

6. **Backtracking Logic**: If placing a number in a particular subset doesn't lead to a solution, we backtrack by removing the number and trying the next possibility.

Here's how you can implement this logic in Python in a format suitable for LeetCode:



```python
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        total_sum = sum(nums)
        
        # If total sum is not divisible by k, we cannot partition
        if total_sum % k != 0:
            return False
        
        target_sum = total_sum // k
        nums.sort(reverse=True)  # Sort in descending order for optimization
        subset_sums = [0] * k    # Array to track the sum of each subset
        
        def backtrack(index):
            # If we've placed all numbers, check if all subsets have the target_sum
            if index == len(nums):
                return all(s == target_sum for s in subset_sums)
            
            for i in range(k):
                if subset_sums[i] + nums[index] <= target_sum:
                    subset_sums[i] += nums[index]
                    
                    if backtrack(index + 1):
                        return True
                    
                    # Backtrack
                    subset_sums[i] -= nums[index]
                    
                # If an empty subset couldn't take this number, then no need to try others
                if subset_sums[i] == 0:
                    break
                    
            return False
        
        return backtrack(0)

```

### Explanation of the Code:

1. **Imports**: We need to import `List` from the `typing` module since we're using type hints.

2. **canPartitionKSubsets method**: This is the main method that implements the logic described above. It initializes the backtracking process.

3. **Backtracking Function**: The `backtrack` function takes an `index` which indicates the current number in `nums` that we are trying to place into one of the `k` subsets. 

4. **Iterating Through Subsets**: For each number, we attempt to add it to each of the `k` subsets. If we successfully add it and can continue to find a solution, we keep going. If not, we backtrack and remove the number from that subset.

5. **Base Condition**: The base case checks if all numbers have been placed. If yes, we verify that all subsets have reached the target sum.

This solution is efficient enough given the constraints (1 ≤ k ≤ len(nums) ≤ 16) and employs backtracking to explore all potential partitions of the array.

# 726. Number of Atoms

### Problem Description 
Given a chemical `formula` (given as a string), return the count of each atom.

The atomic element always starts with an uppercase character, then zero or more lowercase letters, representing the name.

One or more digits representing that element's count may follow if the count is greater than 1. If the count is 1, no digits will follow. For example, H2O and H2O2 are possible, but H1O2 is impossible.

Two formulas concatenated together to produce another formula. For example, H2O2He3Mg4 is also a formula.

A formula placed in parentheses, and a count (optionally added) is also a formula. For example, (H2O2) and (H2O2)3 are formulas.

Given a `formula`, return the count of all elements as a string in the following form: the first name (in sorted order), followed by its count (if that count is more than 1), followed by the second name (in sorted order), followed by its count (if that count is more than 1), and so on.


Example 1:
Input: formula = "H2O"
Output: "H2O"
Explanation: The count of elements are {'H': 2, 'O': 1}.


Example 2:
Input: formula = "Mg(OH)2"
Output: "H2MgO2"
Explanation: The count of elements are {'H': 2, 'Mg': 1, 'O': 2}.


Example 3:
Input: formula = "K4(ON(SO3)2)2"
Output: "K4N2O14S4"
Explanation: The count of elements are {'K': 4, 'N': 2, 'O': 14, 'S': 4}.


Example 4:
Input: formula = "Be32"
Output: "Be32"

Constraints:
`1 <= formula.length <= 1000`
`formula` consists of English letters, digits, `'('`, and `')'`.

`formula` is always valid.

### Solution 
 To solve the problem of counting atoms in a chemical formula, we can follow a systematic approach using a stack to handle the nested structures represented by parentheses. We will implement a function that scans through the formula, counts the atoms and their respective quantities, and outputs the result in a sorted format.

Here's the step-by-step breakdown of the solution:

1. **Use a Stack**: The main idea is to use a stack to keep track of the counts of atoms when we deal with nested parentheses. When we encounter a closing parenthesis `)`, we pop from the stack to calculate the total count of the atoms until the corresponding opening parenthesis `(`.

2. **Parsing Elements and Numbers**: We'll need to differentiate between the element names (which can be one uppercase letter followed by zero or more lowercase letters) and the numbers that follow (which indicate the count of that element). 

3. **Combining Counts**: When we process the characters in the formula, we need to handle cases where an element appears more than once or is multiplied by a number, especially after parentheses.

4. **Sorting and Formatting the Output**: After accumulating counts in a dictionary, we will sort the elements alphabetically and format them according to the required output.

Here is the Python solution implementing the above logic:



```python
from collections import defaultdict
import re

class Solution:
    def countOfAtoms(self, formula: str) -> str:
        stack = []
        atom_count = defaultdict(int)
        
        i = 0
        n = len(formula)
        
        while i < n:
            if formula[i].isupper():  # Start of a new atom
                j = i + 1
                while j < n and formula[j].islower():  # Read the whole atom name
                    j += 1
                atom = formula[i:j]
                i = j
                
                count = 0
                while i < n and formula[i].isdigit():  # Read the count
                    count = count * 10 + int(formula[i])
                    i += 1
                if count == 0:
                    count = 1
                atom_count[atom] += count
            
            elif formula[i] == '(':  # New scope
                stack.append(atom_count)
                atom_count = defaultdict(int)
                i += 1
            
            elif formula[i] == ')':  # Ending scope
                i += 1
                count = 0
                while i < n and formula[i].isdigit():  # Read the following count (if any)
                    count = count * 10 + int(formula[i])
                    i += 1
                if count == 0:
                    count = 1
                
                for atom in atom_count.keys():
                    atom_count[atom] *= count
                
                if stack:
                    prev_count = stack.pop()
                    for atom, cnt in atom_count.items():
                        prev_count[atom] += cnt
                    atom_count = prev_count
            
        # Sort atoms and format the result
        result = []
        for atom in sorted(atom_count.keys()):
            result.append(atom)
            if atom_count[atom] > 1:
                result.append(str(atom_count[atom]))

        return ''.join(result)

# Testing the function with example cases
sol = Solution()
print(sol.countOfAtoms("H2O"))          # Output: "H2O"
print(sol.countOfAtoms("Mg(OH)2"))      # Output: "H2MgO2"
print(sol.countOfAtoms("K4(ON(SO3)2)2"))# Output: "K4N2O14S4"
print(sol.countOfAtoms("Be32"))         # Output: "Be32"

```

### Explanation:
1. **Initialization**: A stack is initialized to hold the counts of atoms. We also use a `defaultdict` for atomic counts (`atom_count`).

2. **Parsing Logic**: As we iterate over the formula:
   - If we encounter an atom, we extract it and its quantity.
   - If we hit a `(`, we push the current `atom_count` onto the stack.
   - If we encounter a `)`, we pop from the stack, multiply the counts by the number following the `)`, and combine them with the previous counts.

3. **Output Formatting**: Finally, we sort the element names and construct the output string accordingly.

This method efficiently handles the parsing and counting of atoms, including nested structures, while ensuring correctness according to the problem constraints.

# 761. Special Binary String

### Problem Description 
Special binary strings are binary strings with the following two properties:
The number of 0's is equal to the number of 1's.

Every prefix of the binary string has at least as many 1's as 0's.

Given a special string `S`, a move consists of choosing two consecutive, non-empty, special substrings of `S`, and swapping them.  (Two strings are consecutive if the last character of the first string is exactly one index before the first character of the second string.)
At the end of any number of moves, what is the lexicographically largest resulting string possible?

Example 1:
Input: S = "11011000"
Output: "11100100"
Explanation:
The strings "10" [occuring at S[1]] and "1100" [at S[3]] are swapped.

This is the lexicographically largest string possible after some number of swaps.

Note:
`S` has length at most `50`.

`S` is guaranteed to be a special binary string as defined above.

### Solution 
 To solve the problem of rearranging a special binary string to achieve the lexicographically largest result, we can leverage the properties of special binary strings and employ a recursive approach. Let's break down the steps to arrive at the solution.

### Steps to Approach:

1. **Understanding Special Binary Strings**: A special binary string has equal numbers of `0`s and `1`s and maintains the condition that in any prefix, the number of `1`s is always greater than or equal to the number of `0`s. 

2. **Recursion and Sliding Window**: We can divide the string into special bracketed substrings:
   - For every substring that can be split, we recursively solve for its inner parts, ensuring we can maintain the special string property.
   - After solving the inner substrings, we can collect the results and sort them in descending order to help achieve the largest lexicographical order.

3. **Reconstruction**: Finally, we reconstruct our string by combining the sorted special strings along with their leading `1`s and trailing `0`s.

### Detailed Solution:

The implementation consists of a recursive function that identifies and organizes these substrings. Once they are processed, they are combined in such a way as to form the largest possible string.

Here is the Python code implementing this approach:



```python
class Solution:
    def makeLargestSpecial(self, S: str) -> str:
        count = i = 0
        substrings = []
        
        # Split S into special substrings
        for j in range(len(S)):
            if S[j] == '1':
                count += 1
            else:
                count -= 1
            
            # Whenever we find a complete special string
            if count == 0:
                # This is a valid special substring
                # Recursively process the inside and add leading/trailing characters
                substrings.append('1' + self.makeLargestSpecial(S[i + 1:j]) + '0')
                i = j + 1
        
        # Sort in reverse order to get the lexicographically largest string
        substrings.sort(reverse=True)
        
        # Join them to form the final largest special string
        return ''.join(substrings)

```

### Explanation of the Code:

1. **Count and Initialize**: We start by initializing variables to keep track of the number of `1`s and `0`s. The variable `i` will track the start of potential special substrings.

2. **Iterate Over the String**: We iterate over each character in the string:
   - Increase the count for `1` and decrease it for `0`.
   - When `count` reaches zero, we have found a complete special substring from `i` to `j`.

3. **Recursively Process**: For each special substring identified, we call the function recursively on the inner substring (`S[i + 1:j]`) to ensure we still follow the special string properties.

4. **Sorting and Joining**: After processing, we sort all the substrings in reverse order and join them together to form the largest possible special string.

### Complexity:

- The time complexity is roughly O(n log n) primarily due to sorting the substrings, where n is the length of the string. The space complexity is O(n) due to storing the substrings.

This approach effectively takes advantage of the recursive properties of special binary strings, ensuring that every substring is analyzed properly to achieve the desired result.

The solution meets LeetCode's requirements and can be directly tested on their platform.

# 776. Split BST

### Problem Description 
Given a Binary Search Tree (BST) with root node `root`, and a target value `V`, split the tree into two subtrees where one subtree has nodes that are all smaller or equal to the target value, while the other subtree has all nodes that are greater than the target value.  It's not necessarily the case that the tree contains a node with value `V`.

Additionally, most of the structure of the original tree should remain.  Formally, for any child C with parent P in the original tree, if they are both in the same subtree after the split, then node C should still have the parent P.

You should output the root TreeNode of both subtrees after splitting, in any order.


Example 1:
Input: root = [4,2,6,1,3,5,7], V = 2
Output: [[2,1],[4,3,6,null,null,5,7]]
Explanation:
Note that root, output[0], and output[1] are TreeNode objects, not arrays.

The given tree [4,2,6,1,3,5,7] is represented by the following diagram:
          4
        /   \
      2      6
     / \    / \
    1   3  5   7
while the diagrams for the outputs are:
          4
        /   \
      3      6      and    2
            / \           /
           5   7         1
Note:
The size of the BST will not exceed `50`.

The BST is always valid and each node's value is different.

### Solution 
 To solve the problem of splitting a Binary Search Tree (BST) into two subtrees based on a target value \( V \), we will create a solution in Python that adheres to the structure commonly expected on platforms like LeetCode.

### Problem Breakdown

1. **Input**: A BST represented by its root node and an integer \( V \).
2. **Output**: Two subtrees:
   - One containing all nodes less than or equal to \( V \).
   - Another containing all nodes greater than \( V \).

3. **Rules**:
   - Each subtree should maintain the BST properties.
   - The output should consist of the root nodes of these two subtrees.

### Approach

1. **Traversal**: We will traverse the tree starting from the root node. For each node, we will:
   - If its value is less than or equal to \( V \), it belongs in the left subtree.
   - If its value is greater than \( V \), it belongs in the right subtree.

2. **Recursive Function**: We will use a recursive function to handle the splitting:
   - If `node` is `None`, return `None`.
   - Depending on the value of `node`, we will recursively split the left and right subtrees.
   - Adjust the pointers to create new subtree roots appropriately based on comparisons with \( V \).

3. **Returning Results**: Finally, we will return the roots of the two split subtrees.

### Code Implementation

Here is the implementation following the format for LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def splitBST(self, root: TreeNode, V: int) -> List[TreeNode]:
        if not root:
            return [None, None]
        
        if root.val <= V:
            # Current root goes into the left tree
            left, right = self.splitBST(root.right, V)
            root.right = left  # Reassign the right child of the current root
            return [root, right]  # The current root is part of the left subtree
        else:
            # Current root goes into the right tree
            left, right = self.splitBST(root.left, V)
            root.left = right  # Reassign the left child of the current root
            return [left, root]  # The current root is part of the right subtree

```

### Explanation of the Code

1. **TreeNode Class**: This is a basic structure for a binary tree node containing a value and pointers to left and right children.

2. **Solution Class**: 
   - We define a method `splitBST` which takes in the `root` of the BST and integer `V`.
   - If the node is `None`, we return two `None` values, indicating both subtrees are empty.
   - Depending on the value of the current node in relation to \( V \):
     - If it's less than or equal, we recursively split the right subtree.
     - If it's greater, we recursively split the left subtree.
   - Finally, we connect the appropriate children based on the comparisons and return the roots of the two subtrees.

### Time Complexity
- The time complexity is \( O(n) \), as we visit each node exactly once.

### Space Complexity
- The space complexity is \( O(h) \) for the recursion stack, where \( h \) is the height of the tree, which can go up to \( O(n) \) in the worst case (unbalanced tree). 

This implementation effectively solves the problem while maintaining the structure of the originally given BST.

# 779. K-th Symbol in Grammar

### Problem Description 
On the first row, we write a `0`. Now in every subsequent row, we look at the previous row and replace each occurrence of `0` with `01`, and each occurrence of `1` with `10`.

Given row `N` and index `K`, return the `K`-th indexed symbol in row `N`. (The values of `K` are 1-indexed.) (1 indexed).


Examples:
Input: N = 1, K = 1
Output: 0
Input: N = 2, K = 1
Output: 0
Input: N = 2, K = 2
Output: 1
Input: N = 4, K = 5
Output: 1
Explanation:
row 1: 0
row 2: 01
row 3: 0110
row 4: 01101001
Note:
`N` will be an integer in the range `[1, 30]`.

`K` will be an integer in the range `[1, 2^(N-1)]`.

### Solution 
 To solve the problem of finding the K-th indexed symbol in row N of a constructed sequence derived from the previous row, we can use a clever approach to avoid generating the entire row explicitly. The sequence follows a recursive pattern where, for each row:

- `0` becomes `01`
- `1` becomes `10`

This transformation gives us an idea that we can represent this generation process without needing to build large strings, especially considering the constraints where N can go up to 30.

### Explanation:

1. **Understanding the Sequence Construction**:
    - The transformation can be seen as a binary tree where:
        - The root starts with `0`.
        - Every `0` in the current level transforms into `01` in the next level.
        - Every `1` transforms into `10`.

2. **Recursive Relation**:
    - Each position in the row can be determined based on the position's parent. We can trace back through the 'generations' to find out what the original symbol (from row 1) would be. 
    - The key observation is that the position can be traced back to find out whether it's in a string formed from `0` (which gives the left child) or `1` (which gives the right child).

3. **Using Binary Representation**:
    - The path to a character is determined by the binary representation of K. If we continuously divide K by 2 and take the remainder, we determine if we are on a `0` or `1` based on the depth in the 'tree'.
    - The parity (even or odd) of the number of steps taken from the initial `0` can also determine the final character.

### Implementation:

Following the insights above, we can implement this in Python directly suitable for a LeetCode submission:



```python
class Solution:
    def kthGrammar(self, N: int, K: int) -> int:
        # K is 1-indexed, we can work with a 0-indexed logic by reducing K by 1
        K -= 1
        
        # We will determine the result by counting how many times we need to flip
        # We reduce K and check if it's even or odd to determine if we flip.
        result = 0  # Starting at '0' which represents the root
        
        while K > 0:  # While we still have positions left to consider
            # Check if K is odd or even
            # When K is odd, it represents a '1', thus we increment the flips
            if K % 2 == 1:
                result ^= 1  # Flip between 0 and 1
            # Move to the parent node
            K //= 2
        
        return result

```

### Explanation of Code:

- We start by converting K to a 0-indexed number since Python lists and calculations generally use 0-based indices.
- We loop while K is greater than 0, which means we are tracing back to the original `0`.
- In each iteration, we check if K is odd or even:
    - If odd, it means we encountered a transformation `1` at this level, so we perform an XOR with `1` (flipping our current result).
- We then divide K by 2 to move up the binary tree (to the parent).
- The final result will be `0` or `1`, which is what we want to return.

### Complexity:
- This approach has a time complexity of O(log K), which is logarithmic in terms of the depth of the row, and a space complexity of O(1) since we are using constant space. The efficient nature of this approach allows it to handle the maximum constraints comfortably.

# 783. Minimum Distance Between BST Nodes

### Problem Description 
Given the `root` of a Binary Search Tree (BST), return the minimum difference between the values of any two different nodes in the tree.

Note: This question is the same as 530: https://leetcode.com/problems/minimum-absolute-difference-in-bst/

Example 1:
Input: root = [4,2,6,1,3]
Output: 1

Example 2:
Input: root = [1,0,48,null,null,12,49]
Output: 1

Constraints:
The number of nodes in the tree is in the range `[2, 100]`.

`0 <= Node.val <= 105`

### Solution 
 To solve the problem of finding the minimum difference between the values of any two different nodes in a Binary Search Tree (BST), we can take advantage of the properties of a BST. In a BST, the left subtree of any node contains values less than that node, while the right subtree contains values greater. This property allows us to traverse the tree in an ordered manner using an in-order traversal, which visits the nodes in increasing order of their values.

### Approach:
1. **In-order Traversal**: Perform an in-order traversal of the tree to retrieve the values in sorted order. This will give us a list of the node values in ascending order.
2. **Calculate Differences**: Once we have the sorted list of node values, we can iterate through the list and compute the absolute differences between consecutive values. The minimum of these differences will be our answer.

### Implementation Steps:
1. Define a helper function to traverse the BST in-order and collect the values in a list.
2. After collecting the values, compute the minimum difference between consecutive values in the list.
3. Return the minimum difference as the final answer.

### Python Code:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def getMinimumDifference(self, root: TreeNode) -> int:
        # To hold the in-order traversal values
        values = []

        # Helper function for in-order traversal
        def in_order_traversal(node):
            if not node:
                return
            in_order_traversal(node.left)  # Visit left subtree
            values.append(node.val)         # Visit node
            in_order_traversal(node.right) # Visit right subtree
            
        # Perform in-order traversal of the BST
        in_order_traversal(root)

        # Calculate the minimum difference
        min_difference = float('inf')
        for i in range(1, len(values)):
            # Difference between the current and previous values
            min_difference = min(min_difference, values[i] - values[i - 1])
        
        return min_difference

```

### Explanation of the Code:
1. **TreeNode Class**: This is a standard definition for a node in a binary tree, with a value `val`, and pointers to its left and right children.
2. **Solution Class**: Contains the method `getMinimumDifference` that implements the solution.
3. **In-order Traversal**: The `in_order_traversal` function is defined within `getMinimumDifference`. It recursively traverses the tree in order and appends each node's value to the `values` list.
4. **Calculating the Minimum Difference**: After collecting all values, we compute the minimum difference by iterating through the sorted list and checking the differences between each pair of consecutive elements.
5. **Return Statement**: Finally, the minimum difference found is returned.

### Complexity Analysis:
- **Time Complexity**: The time complexity of the solution is O(n), where n is the number of nodes in the BST. This is because we visit each node exactly once during the in-order traversal.
- **Space Complexity**: The space complexity is O(n) as well, due to storing the values of the nodes in a list.

This code can be run directly on LeetCode and will produce the correct results for the given problem.

# 794. Valid Tic-Tac-Toe State

### Problem Description 
A Tic-Tac-Toe board is given as a string array `board`. Return True if and only if it is possible to reach this board position during the course of a valid tic-tac-toe game.

The `board` is a 3 x 3 array, and consists of characters `" "`, `"X"`, and `"O"`.  The " " character represents an empty square.

Here are the rules of Tic-Tac-Toe:
Players take turns placing characters into empty squares (" ").

The first player always places "X" characters, while the second player always places "O" characters.

"X" and "O" characters are always placed into empty squares, never filled ones.

The game ends when there are 3 of the same (non-empty) character filling any row, column, or diagonal.

The game also ends if all squares are non-empty.

No more moves can be played if the game is over.


Example 1:
Input: board = ["O  ", "   ", "   "]
Output: false
Explanation: The first player always plays "X".


Example 2:
Input: board = ["XOX", " X ", "   "]
Output: false
Explanation: Players take turns making moves.


Example 3:
Input: board = ["XXX", "   ", "OOO"]
Output: false

Example 4:
Input: board = ["XOX", "O O", "XOX"]
Output: true
Note:
`board` is a length-3 array of strings, where each string `board[i]` has length 3.

Each `board[i][j]` is a character in the set `{" ", "X", "O"}`.

### Solution 
 To solve the Tic-Tac-Toe board verification problem, we need to ensure that the board configuration is valid given the rules of the game. Here’s how we can approach this:

### Steps to Follow:

1. **Count 'X' and 'O':** We need to ensure the counts of 'X' and 'O' are correct. The game starts with 'X' and alternates turns, so the following rules apply:
   - The number of 'X's can be equal to or one more than the number of 'O's.
   - The number of 'O's can never exceed the number of 'X's.

2. **Check Winning Conditions:** We need to check if either player has won the game.
   - Winning can occur in rows, columns, or diagonals.
   - If 'X' has won, then the count of 'X' should be exactly one more than the count of 'O'.
   - If 'O' has won, then the count of 'X' must be equal to the count of 'O'.

3. **Return Results:** Based on the above checks, we will return `True` if the board is valid, otherwise `False`.

### Implementation

Let’s implement the solution in Python following the structure required by LeetCode.



```python
class Solution:
    def validTicTacToe(self, board: List[str]) -> bool:
        x_count = sum(row.count('X') for row in board)
        o_count = sum(row.count('O') for row in board)

        # Rule 1: Check if counts of 'X' and 'O' are valid
        if o_count > x_count or x_count > o_count + 1:
            return False
        
        # Winning conditions
        def check_winner(char):
            return (board[0][0] == board[0][1] == board[0][2] == char or
                    board[1][0] == board[1][1] == board[1][2] == char or
                    board[2][0] == board[2][1] == board[2][2] == char or
                    board[0][0] == board[1][0] == board[2][0] == char or
                    board[0][1] == board[1][1] == board[2][1] == char or
                    board[0][2] == board[1][2] == board[2][2] == char or
                    board[0][0] == board[1][1] == board[2][2] == char or
                    board[0][2] == board[1][1] == board[2][0] == char)

        x_wins = check_winner('X')
        o_wins = check_winner('O')

        # Rule 2: Check if the winning conditions for 'X' and 'O' are valid
        if x_wins and o_wins:
            return False
        if x_wins and x_count != o_count + 1:
            return False
        if o_wins and x_count != o_count:
            return False

        return True

```

### Explanation:

- **Counting Characters:** We use list comprehensions to count 'X' and 'O' on the board. This gives us the current player turns.
- **Winning Check Function:** The `check_winner` function determines if a player has met any winning condition by checking rows, columns, and diagonals.
- **Validation Logic:**
  - First, we check the counts of 'X' and 'O'.
  - Then, we check if both players have been declared winners (which is impossible).
  - Finally, we check the win conditions only if applicable to see if counts align with the game rules.

This method ensures that we comprehensively check the board for validity in terms of gameplay while allowing for efficient verification of rules.

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
 To solve the problem of finding the smallest subtree that contains all the deepest nodes in a binary tree, we can perform a depth-first search (DFS) on the tree. The approach involves counting the depth of each node and keeping track of the deepest nodes as we traverse the tree.

### Solution Explanation:

1. **Depth Calculation**: We will traverse the tree using DFS and calculate the depth of each node. We'll also keep track of the maximum depth encountered.
  
2. **Finding Deepest Nodes**: While performing the DFS, we will also identify the nodes that are at the maximum depth.

3. **Finding the Smallest Subtree**: We'll check the nodes in the last layer of the tree (those at the maximum depth) and find their lowest common ancestor (LCA) which will be our answer.

4. **Edge Cases**: We should handle cases where the tree has only one node, as well as trees that may have deeper nodes positioned differently.

### Python Code:

Now, let's implement the solution in Python following the LeetCode format.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def subtreeWithAllDeepest(self, root: TreeNode) -> TreeNode:
        def dfs(node):
            if not node:
                return 0
            
            left_depth = dfs(node.left)
            right_depth = dfs(node.right)
            
            # If the left and right depths are the same, this node is the LCA of deepest nodes
            if left_depth == right_depth:
                return left_depth + 1
            
            # Return depth based on which side is deeper
            if left_depth > right_depth:
                self.lca = node.left if self.lca is None else self.lca
                return left_depth + 1
            else:
                self.lca = node.right if self.lca is None else self.lca
                return right_depth + 1
        
        self.lca = None  # This will hold the LCA of the deepest nodes
        dfs(root)
        return self.lca

# The code can now be tested with various tree inputs by constructing TreeNode objects.

```

### How the Code Works:

1. **DFS Function**: We define a nested `dfs` function that:
   - Returns 0 depth if the node is `None`.
   - Recursively calculates depths of left and right subtrees.
   - If both left and right depths are equal, it updates the `lca` to the current node (indicating this node is the LCA of the deepest nodes).
   - If one side is deeper, it determines which side is deeper and updates the `lca` accordingly.

2. **`self.lca` Variable**: We use an instance variable `self.lca` to keep track of the lowest common ancestor of the deepest nodes as we traverse the tree.

3. **Returning the Result**: After completing the DFS, `self.lca` will contain the root of the smallest subtree that contains all the deepest nodes.

### Example Usage:

To test the solution, create a binary tree by instantiating `TreeNode` objects, and then call `subtreeWithAllDeepest` with the root of the tree.

This structured approach ensures that we effectively find the subtree containing all the deepest nodes while maintaining clarity and correctness throughout the implementation.

# 894. All Possible Full Binary Trees

### Problem Description 
Given an integer `n`, return a list of all possible full binary trees with `n` nodes. Each node of each tree in the answer must have `Node.val == 0`.

Each element of the answer is the root node of one possible tree. You may return the final list of trees in any order.

A full binary tree is a binary tree where each node has exactly `0` or `2` children.


Example 1:
Input: n = 7
Output: [[0,0,0,null,null,0,0,null,null,0,0],[0,0,0,null,null,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,null,null,null,null,0,0],[0,0,0,0,0,null,null,0,0]]

Example 2:
Input: n = 3
Output: [[0,0,0]]

Constraints:
`1 <= n <= 20`

### Solution 
 To solve the problem of generating all possible full binary trees with `n` nodes, we can use a recursive approach. A full binary tree is defined such that every node has either 0 or 2 children. Given this, the number of nodes in a full binary tree must always be an odd number. Therefore, if `n` is even, we cannot form any full binary trees and should return an empty list.

The key idea is to break down the problem as follows:
1. If we have `n` nodes for a full binary tree, the root node uses up one node, and we must distribute the remaining `n-1` nodes between the left and right subtrees.
2. The number of nodes in the left subtree can be any odd number less than `n`, i.e., `1, 3, 5, ..., n-2`.
3. For each possible split of nodes into left and right subtrees, we recursively generate all possible left and right subtree combinations.
4. We then combine the left and right subtrees with a new root node to form full binary trees.

Here’s the complete Python code to generate all possible full binary trees using this approach. It also includes a definition for the `TreeNode` class as required by LeetCode.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def allPossibleFBT(self, n: int) -> List[TreeNode]:
        if n % 2 == 0:  # If n is even, return an empty list.
            return []
        
        if n == 1:  # Base case: A full binary tree with one node (the root).
            return [TreeNode(0)]
        
        result = []
        
        # Iterate over all possible odd numbers of nodes for the left subtree
        for left_nodes in range(1, n, 2):
            right_nodes = n - 1 - left_nodes  # Remaining nodes go to the right subtree
            # Generate all left and right subtrees
            left_subtrees = self.allPossibleFBT(left_nodes)
            right_subtrees = self.allPossibleFBT(right_nodes)
            
            # Combine each left and right subtree with a new root node
            for left in left_subtrees:
                for right in right_subtrees:
                    root = TreeNode(0)  # Create a new root node
                    root.left = left
                    root.right = right
                    result.append(root)  # Add this subtree to the result
        
        return result

# Example usage:
# solution = Solution()
# trees = solution.allPossibleFBT(7)
# This will generate all possible full binary trees with 7 nodes.

```

### Explanation of the Code:
1. **TreeNode Class**: This is a simple definition of a tree node. Each node has a value (`val`), a left child (`left`), and a right child (`right`).
  
2. **allPossibleFBT Function**: This is the main function that returns all full binary trees with `n` nodes.
   - If `n` is even, it immediately returns an empty list as we cannot form a full binary tree.
   - If `n` is 1, a single-node tree is returned.
   - For odd `n`, we iterate through possible numbers of nodes for the left subtree (`left_nodes`). The right subtree nodes are simply calculated as `n - 1 - left_nodes`.
   - We recursively get all possible left and right subtrees and then combine them with the newly created root node.

3. **Combining Trees**: For each combination of left and right subtrees, a new root node is created, and the left and right children are assigned. The resulting tree is added to the list of results.

This solution is efficient for the given constraints (`1 <= n <= 20`) and successfully generates all possible full binary trees of `n` nodes.

# 897. Increasing Order Search Tree

### Problem Description 
Given the `root` of a binary search tree, rearrange the tree in in-order so that the leftmost node in the tree is now the root of the tree, and every node has no left child and only one right child.


Example 1:
Input: root = [5,3,6,2,4,null,8,1,null,null,null,7,9]
Output: [1,null,2,null,3,null,4,null,5,null,6,null,7,null,8,null,9]

Example 2:
Input: root = [5,1,7]
Output: [1,null,5,null,7]

Constraints:
The number of nodes in the given tree will be in the range `[1, 100]`.

`0 <= Node.val <= 1000`

### Solution 
 To solve the problem of rearranging a binary search tree (BST) in the specified manner, we need to transform the tree into a structure where every node has no left child and only one right child, effectively creating a "linked list" from the in-order traversal of the BST.

### Step-by-Step Explanation

1. **Understanding In-order Traversal**:
   - The in-order traversal of a BST yields the nodes in sorted order. For a given BST, if we traverse it in-order, we will retrieve the node values in ascending order.

2. **Rearranging the Tree**:
   - We want to take the result of the in-order traversal and rearrange the tree such that the leftmost node becomes the new root and each subsequent node becomes the right child of the previous node.

3. **Implementation Plan**:
   - We can use a combination of recursive depth-first search (DFS) for in-order traversal and a linked list-like structure to rearrange the nodes.
   - We'll maintain a `prev` pointer to keep track of the last processed node, and for each node processed, set its left child to `None` and the right child to the next node.

### Python Code

Below is the Python code following the LeetCode format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        # This will hold the new root of the rearranged tree
        self.new_root = TreeNode(0)  # Dummy node to facilitate easy linking
        self.prev = self.new_root      # This will help keep track of the last node processed

        # Perform in-order traversal and rearrange
        def in_order(node: TreeNode):
            if not node:
                return
            # Visit the left subtree
            in_order(node.left)
            # Rearrange the node
            # The left child should be None
            node.left = None
            # The right child should point to the next node
            self.prev.right = node
            # Move the prev pointer
            self.prev = node
            # Visit the right subtree
            in_order(node.right)

        in_order(root)
        # The actual root of the rearranged tree is the right child of the dummy node
        return self.new_root.right

```

### Explanation of the Code

1. **TreeNode Class**: This is the definition for each node in the binary tree. Each node has a value (`val`), a left child (`left`), and a right child (`right`).

2. **Solution Class**: This contains the `increasingBST` method which performs the rearrangement of the tree.
   - **Dummy Node**: We create a dummy node `self.new_root` to simplify the linking of nodes.
   - **Recursive Function - `in_order`**: This function performs the in-order traversal.
     - It checks if the current node is `None`.
     - It recursively visits the left subtree.
     - We then rearrange the current node: set its left child to `None` and link it to the right of the last processed node (tracked by `self.prev`).
     - Finally, we recursively visit the right subtree.

3. **Returning the Result**: After the traversal and rearrangement, we return `self.new_root.right` as the new root of the rearranged tree.

### Conclusion

This solution efficiently rearranges the BST into the desired format with an in-order traversal, ensuring that the complexities remain manageable within the given constraints. The time complexity of this approach is O(n), where n is the number of nodes in the tree, since each node is processed exactly once.

# 938. Range Sum of BST

### Problem Description 
Given the `root` node of a binary search tree, return the sum of values of all nodes with a value in the range `[low, high]`.


Example 1:
Input: root = [10,5,15,3,7,null,18], low = 7, high = 15
Output: 32

Example 2:
Input: root = [10,5,15,3,7,13,18,1,null,6], low = 6, high = 10
Output: 23

Constraints:
The number of nodes in the tree is in the range `[1, 2 * 104]`.

`1 <= Node.val <= 105`
`1 <= low <= high <= 105`
All `Node.val` are unique.

### Solution 
 To solve the problem of finding the sum of values of all nodes within a given range `[low, high]` in a binary search tree (BST), we can leverage the properties of a BST. In a BST, for any given node:
- All values in the left subtree are less than the node's value.
- All values in the right subtree are greater than the node's value.

This characteristic allows us to traverse the tree efficiently, skipping entire subtrees when their values fall outside of the specified range.

### Approach:
1. Start at the root node and check its value.
2. If the node's value is less than `low`, it means all values in the left subtree will also be less than `low`, so we can skip the left subtree and move to the right subtree.
3. If the node's value is greater than `high`, similarly, we can skip the right subtree and only continue with the left subtree.
4. If the node's value is within the inclusive range `[low, high]`, we add its value to our running total and proceed to check both left and right subtrees.
5. We use a recursive approach to implement this or a stack for an iterative approach if preferred.

### Implementation:
Here is the Python code following the LeetCode format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        # Initialize the sum
        total_sum = 0
        
        # Helper function for DFS traversal
        def dfs(node):
            nonlocal total_sum
            if not node:
                return
            
            # If the node's value is within the range
            if low <= node.val <= high:
                total_sum += node.val
            
            # If the node's value is greater than low, check the left subtree
            if node.val > low:
                dfs(node.left)
                
            # If the node's value is less than high, check the right subtree
            if node.val < high:
                dfs(node.right)
        
        # Start the DFS traversal from the root
        dfs(root)
        
        return total_sum

```

### Explanation of the Code:
- We define a `TreeNode` class for the nodes of the tree.
- We create a `Solution` class that contains the `rangeSumBST` method.
- Inside `rangeSumBST`, we initialize `total_sum` to accumulate the values of the nodes within the specified range.
- We define a nested `dfs` (depth-first search) function that takes a node as its parameter:
  - When the node is `None`, we return immediately (base case).
  - If the current node's value is within the range `[low, high]`, we add it to `total_sum`.
  - We continue searching the left subtree if the current node's value is greater than `low`.
  - We continue searching the right subtree if the current node's value is less than `high`.
- Finally, we invoke the `dfs` function starting from the `root` and then return the `total_sum` which gives us the required sum of values within the specified range.

This solution has a time complexity of O(N) in the worst case if all nodes are within the range, but on average it will be O(log N) if the tree is balanced, making it efficient for large input sizes within the constraints provided.

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
 To solve the problem of finding all non-negative integers of length `n` such that the absolute difference between every two consecutive digits is `k`, we can take a backtracking approach. The key characteristics we need to keep in mind are:

1. **Non-Negative Integer**: The number should not have leading zeros unless the number itself is zero (which won't happen here due to the length constraints).
2. **Consecutive Digit Difference**: For any chosen digit, the next digit should either be greater than or equal to `k` or less than or equal to `k` based on the value of the current digit. 

Here's a structured solution following the LeetCode format:



```python
class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        if n == 1:  # If length is 1, return 0 through 9
            return list(range(10))
        
        results = []
        
        # Helper function for backtracking
        def backtrack(num, length):
            if length == n:
                results.append(num)
                return
            
            last_digit = num % 10
            
            # Two possible next digits
            next_digits = set()
            if last_digit + k <= 9:  # Valid digit greater than current
                next_digits.add(last_digit + k)
            if last_digit - k >= 0:  # Valid digit less than current
                next_digits.add(last_digit - k)
            
            for next_digit in next_digits:
                backtrack(num * 10 + next_digit, length + 1)

        # Start backtracking from each digit from 1 to 9 (to prevent leading zeros)
        for start_digit in range(1, 10):
            backtrack(start_digit, 1)
        
        return results

```

### Explanation of the Code:

1. **Class Definition**: We define a class `Solution` which contains the method `numsSameConsecDiff`.

2. **Input Handling**: 
   - If `n` is `1`, we simply return all single-digit numbers (0 through 9).

3. **Backtracking Function**:
   - The `backtrack(num, length)` function is defined to recursively build valid numbers.
   - It takes the current number (`num`) being formed and its `length` (number of digits formed so far).
   - If we have reached the required length `n`, we append the formed number to the results list.

4. **Finding Next Digits**:
   - For the last digit of the current number (extracted using `num % 10`), we compute the next possible digits that can be appended:
     - `last_digit + k`: Must be less than or equal to `9`.
     - `last_digit - k`: Must be greater than or equal to `0`.

5. **Iteration Through Next Digits**:
   - The method iterates through the valid next digits and calls `backtrack` recursively, forming new numbers by appending each valid next digit.

6. **Starting the Backtracking**:
   - The loop initializes starting points for valid numbers (1 to 9) to ensure no leading zeros.

7. **Return Result**:
   - Finally, the constructed list of results is returned.

This approach efficiently constructs all valid numbers of length `n` while adhering to the given constraints.

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
 To solve the problem of converting a Binary Search Tree (BST) into a Greater Tree, we need to make sure that each node in the tree is updated to the sum of its original value plus the values of all nodes that are greater than it.

Given the properties of a BST:
- The left subtree contains only nodes with keys lesser than the node's key.
- The right subtree contains only nodes with keys greater than the node's key.

We can utilize a reverse in-order traversal (visit right subtree, then the current node, then the left subtree) to achieve this conversion. By traversing the tree in this manner, we can keep a cumulative sum of the values of the nodes we have visited, allowing us to update each node's value accordingly.

Here's a detailed explanation of the approach:

1. **Reverse In-Order Traversal**: This traversal method will allow us to visit all the nodes in descending order. Thus, by the time we reach a node, we have already processed all the nodes that have greater values.

2. **Cumulative Sum**: We maintain a variable to keep a running total of the values we've encountered. When we visit a node during our traversal, we will add the node's original value to this cumulative sum and update the node's value to this new sum.

3. **Recursion**: We will use a helper function that takes a node and the cumulative sum as parameters, allowing us to traverse and update the tree efficiently.

Now, let's implement this in Python in the required LeetCode solution format.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        # Initialize a variable to keep track of the cumulative sum
        self.cumulative_sum = 0
        
        # Define the recursive function for reverse in-order traversal
        def reverse_in_order(node: TreeNode):
            if not node:
                return
            
            # Traverse the right subtree first
            reverse_in_order(node.right)
            
            # Process the current node
            self.cumulative_sum += node.val
            node.val = self.cumulative_sum
            
            # Traverse the left subtree
            reverse_in_order(node.left)
        
        # Start the reverse in-order traversal from the root
        reverse_in_order(root)
        return root

```

### Explanation of the Code:
1. **TreeNode Class**: This class is a standard definition of a tree node that contains a value (`val`), a left child (`left`), and a right child (`right`).

2. **Solution Class**: The `Solution` class encapsulates our algorithm.
   - We define an instance variable `self.cumulative_sum` to store the ongoing total as we traverse the tree.
   - The nested function `reverse_in_order(node)` performs the reverse in-order traversal. It processes the right child, updates the current node's value, and processes the left child.

3. **Handling the Root**: Finally, we invoke the `reverse_in_order` function starting at the `root`, and we return the modified root node.

### Complexity Analysis:
- **Time Complexity**: O(n), where n is the number of nodes in the tree, since we visit each node exactly once.
- **Space Complexity**: O(h), where h is the height of the tree, which accounts for the space used by the recursion stack. In the worst case, this can be O(n) for a skewed tree.

The algorithm effectively converts the BST into a Greater Tree in a clean and efficient manner.

# 1137. N-th Tribonacci Number

### Problem Description 
The Tribonacci sequence Tn is defined as follows: 
T0 = 0, T1 = 1, T2 = 1, and Tn+3 = Tn + Tn+1 + Tn+2 for n >= 0.

Given `n`, return the value of Tn.


Example 1:
Input: n = 4
Output: 4
Explanation:
T_3 = 0 + 1 + 1 = 2
T_4 = 1 + 1 + 2 = 4

Example 2:
Input: n = 25
Output: 1389537

Constraints:
`0 <= n <= 37`
The answer is guaranteed to fit within a 32-bit integer, ie. `answer <= 2^31 - 1`.

### Solution 
 To solve the problem of computing the nth value of the Tribonacci sequence, we can use an iterative approach. Here's a breakdown of the steps involved and the implementation in Python, formatted according to the LeetCode standards.

### Understanding the Problem

The Tribonacci sequence is similar to the Fibonacci sequence, where each term after the first three is the sum of the previous three terms:
- **Base cases**:
  - T0 = 0
  - T1 = 1
  - T2 = 1
- **Recurrence relation**:
  - Tn = T(n-1) + T(n-2) + T(n-3) for n >= 3

We need to compute Tn for any given n in the range [0, 37].

### Approach

1. **Base Cases**: Directly return the result for n = 0, 1, or 2.
2. **Iterative Computation**: For n >= 3, iterate from 3 to n while maintaining the last three computed values of the sequence.
3. **Space Efficiency**: Instead of storing the entire sequence, we only need the last three numbers at any time, which reduces our space complexity.

### Implementation

Here is the Python code implementing this approach:



```python
class Solution:
    def tribonacci(self, n: int) -> int:
        # Base cases
        if n == 0:
            return 0
        elif n == 1 or n == 2:
            return 1
        
        # Initialize the first three Tribonacci numbers
        T0, T1, T2 = 0, 1, 1
        
        # Compute Tn iteratively
        for i in range(3, n + 1):
            Tn = T0 + T1 + T2  # Tn = T(n-1) + T(n-2) + T(n-3)
            # Update the last three Tribonacci numbers
            T0, T1, T2 = T1, T2, Tn
        
        return Tn

```

### Explanation of the Code

- **Class `Solution`**: This adheres to LeetCode's requirement to encapsulate the solution in a class.
- **Method `tribonacci`**: This is the main method to compute the nth Tribonacci number.
- **Input**: An integer `n`, which is the index for the Tribonacci number.
- **Base Cases**: We handle the simplest cases (n = 0, 1, 2) directly at the beginning for efficiency.
- **Looping**: For values of n >= 3, we iterate from 3 to n, calculating each successive Tribonacci number using the three most recent values.
- **Return Value**: Finally, the nth Tribonacci number is returned.

### Time and Space Complexity
- **Time Complexity**: O(n) — We compute Tn in a linear number of steps relative to n.
- **Space Complexity**: O(1) — We only utilize a constant amount of space for storing the last three computed values. 

You can directly copy this code into the LeetCode platform, and it should run correctly for the given problem statement.

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
 To solve the problem of determining if we can reach an index with a value of `0` in the given array by jumping from a starting index, we can utilize a graph traversal approach. Specifically, we can use Breadth-First Search (BFS) to explore all possible indices we can reach from the starting position.

### Explanation

1. **Initialization**: 
   - We start by checking if the array is valid. If the starting index itself contains `0`, we return `True`.
   - We will use a queue to manage our exploration of indices and a set to keep track of visited indices to avoid cycles and repeated work.

2. **BFS Traversal**:
   - For the current index, determine the next potential indices we could reach by moving either `i + arr[i]` or `i - arr[i]`.
   - If either of these indices contains `0`, we return `True`.
   - If the next index is within bounds and has not been visited yet, we enqueue it for further exploration.

3. **Completion**:
   - If we exhaust our queue without reaching any index with a value of `0`, we return `False`.

### Python code

Here’s the complete code in the format as it would be presented in LeetCode:



```python
from collections import deque

class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        # Check if the starting position is already 0
        if arr[start] == 0:
            return True
        
        n = len(arr)
        visited = set()  # To keep track of visited indices
        queue = deque([start])  # BFS queue initialized with the start index

        while queue:
            index = queue.popleft()
            
            # Calculate the two possible jumps
            jump_forward = index + arr[index]
            jump_backward = index - arr[index]

            # Check the forward jump
            if jump_forward < n:
                if arr[jump_forward] == 0:
                    return True
                if jump_forward not in visited:
                    visited.add(jump_forward)
                    queue.append(jump_forward)

            # Check the backward jump
            if jump_backward >= 0:
                if arr[jump_backward] == 0:
                    return True
                if jump_backward not in visited:
                    visited.add(jump_backward)
                    queue.append(jump_backward)

        return False

```

### Key Considerations:
- **Complexity**: The time complexity is O(n), where n is the length of the array, as each index will be processed at most once.
- **Space Complexity**: The space complexity is also O(n) for the queue and visited set storage.

### Usage
You can simply copy and paste the code into LeetCode's online editor for the problem and run it against the provided test cases. The function `canReach` will determine if it's possible to reach an index with a value of zero from the specified starting index.

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
 To solve the problem of finding the corresponding node in the cloned binary tree that is identical to a given target node in the original binary tree, we can utilize a simple tree traversal technique, like Depth-First Search (DFS). Given that both trees are identical in structure and the node values are unique, we'll be able to traverse the cloned tree in parallel with the original tree until we find the target in the original tree, at which point we can return the corresponding node from the cloned tree.

Here is the step-by-step approach we'll take:

1. **Input Structure**: We'll define the structure of a tree node (if not already provided by LeetCode) using a class.

2. **DFS Traversal**: We'll implement a recursive function that traverses both trees simultaneously. At each node, we will:
   - If the current node in the original tree is equal to the target node, return the corresponding node in the cloned tree.
   - If not, we will recursively traverse the left and right children of both trees.

3. **Base Case**: If we reach a leaf node (i.e., a null node), we will return `None`.

4. **Return the Result**: After performing the DFS, we will return the found node from the cloned tree.

Here's the implementation, formatted for direct use in LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
        # Define the DFS function
        def dfs(o_node, c_node):
            if not o_node:
                return None
            
            # If we found the target node in the original tree, return the corresponding cloned node
            if o_node is target:
                return c_node
            
            # Perform DFS on the left subtree
            left_result = dfs(o_node.left, c_node.left)
            if left_result:  # If we found it in the left subtree, return the result
                return left_result
            
            # Perform DFS on the right subtree
            return dfs(o_node.right, c_node.right)

        # Start DFS from the root of both trees
        return dfs(original, cloned)

```

### Explanation of the Code:

1. **TreeNode Class**: We define a `TreeNode` class that represents each node in the binary tree.

2. **Solution Class**: We create a `Solution` class that includes our main method, `getTargetCopy`.

3. **DFS Function**: The `dfs` function takes two parameters:
   - `o_node`: the current node in the original tree.
   - `c_node`: the current node in the cloned tree.

4. **Base Case**: If `o_node` is `None`, we return `None`, indicating we've reached a leaf node.

5. **Target Check**: If the current original node (`o_node`) is the same as `target`, we return the corresponding cloned node (`c_node`).

6. **Recursive Calls**: We first search in the left subtree. If we find our target in the left side, we return that result. If not, we continue our search in the right subtree.

7. **Execution**: Finally, we call the `dfs` method starting from the root of both trees.

This code remains efficient and straightforward, and it adheres to the constraints provided in the problem statement. The DFS ensures that we find the target node directly, making it well-suited to the binary tree traversal structure.

# 1718. Construct the Lexicographically Largest Valid Sequence

### Problem Description 
Given an integer `n`, find a sequence that satisfies all of the following:
The integer `1` occurs once in the sequence.

Each integer between `2` and `n` occurs twice in the sequence.

For every integer `i` between `2` and `n`, the distance between the two occurrences of `i` is exactly `i`.

The distance between two numbers on the sequence, `a[i]` and `a[j]`, is the absolute difference of their indices, `|j - i|`.

Return the lexicographically largest sequence. It is guaranteed that under the given constraints, there is always a solution. 
A sequence `a` is lexicographically larger than a sequence `b` (of the same length) if in the first position where `a` and `b` differ, sequence `a` has a number greater than the corresponding number in `b`. For example, `[0,1,9,0]` is lexicographically larger than `[0,1,5,6]` because the first position they differ is at the third number, and `9` is greater than `5`.


Example 1:
Input: n = 3
Output: [3,1,2,3,2]
Explanation: [2,3,2,1,3] is also a valid sequence, but [3,1,2,3,2] is the lexicographically largest valid sequence.


Example 2:
Input: n = 5
Output: [5,3,1,4,3,5,2,4,2]

Constraints:
`1 <= n <= 20`

### Solution 
 To solve the problem of creating a lexicographically largest sequence under the defined constraints, we can utilize a systematic approach to generate the sequence step by step. Below is an explanation of the solution along with the actual Python code that is compatible with the LeetCode platform.

### Detailed Explanation
1. **Understanding Requirements**:
   - The integer `1` must appear once.
   - Each integer from `2` to `n` must appear exactly twice.
   - For any integer `i`, the two occurrences must be exactly `i` indices apart.
   - Our goal is to maximize the sequence lexicographically, meaning that larger numbers should be placed first whenever possible.

2. **Sequence Structure**:
   - We will create an empty list to hold our sequence.
   - We will place the integers from `n` down to `2` in the sequence to ensure that we achieve the lexicographically largest arrangement.
   - We will carefully place each integer `i` at indices that are `i` apart.

3. **Implementation Steps**:
   - Start by initializing the sequence with zeros (or some placeholder) of length equal to the required size.
   - For each number from `n` down to `2`, determine the best indices to place them such that the spacing condition is met.
   - Finally, insert the integer `1` in the middle (or appropriate) position.

The following is the complete Python code implementing the above approach:



```python
def constructDistancedSequence(n):
    # Initialize the resulting sequence with placeholders (0)
    size = n + (n - 1)  # Size of the output array
    sequence = [0] * size
    
    # Function to place the number 'num' in the sequence
    def place_number(num):
        # Find a suitable place for the number 'num'
        for i in range(len(sequence) - 1):
            if sequence[i] == 0 and i + num < len(sequence) and sequence[i + num] == 0:
                sequence[i] = num
                sequence[i + num] = num
                return True
        return False

    # Place numbers from n to 2
    for num in range(n, 1, -1):
        place_number(num)

    # Place number '1'
    for i in range(len(sequence)):
        if sequence[i] == 0:
            sequence[i] = 1
            break

    return sequence

```

### Explanation of the Code:
- **Initialization**: The sequence is initialized with zeroes. The length is calculated as `n + (n - 1)` because we need space for all numbers including the required distances.
- **Placing Numbers**: 
  - The function `place_number(num)` looks for available indices to place `num`. It checks if both positions `i` and `i + num` are free (i.e., contain zero). When found, it assigns the value to both indices.
- **Looping**: We iterate from `n` down to `2` ensuring that we keep the sequence lexicographically large. After placing all `2` to `n` integers, we place `1` in the first available spot.
- **Output**: Finally, the function returns the constructed sequence.

By following this structured approach, we ensure that we fulfill all the given constraints while also achieving the desired order of numbers in the sequence. This solution will be efficient and works within the problem's constraints as `n` ranges from `1` to `20`.

# 1723. Find Minimum Time to Finish All Jobs

### Problem Description 
You are given an integer array `jobs`, where `jobs[i]` is the amount of time it takes to complete the `ith` job.

There are `k` workers that you can assign jobs to. Each job should be assigned to exactly one worker. The working time of a worker is the sum of the time it takes to complete all jobs assigned to them. Your goal is to devise an optimal assignment such that the maximum working time of any worker is minimized.

Return the minimum possible maximum working time of any assignment. 

Example 1:
Input: jobs = [3,2,3], k = 3
Output: 3
Explanation: By assigning each person one job, the maximum time is 3.


Example 2:
Input: jobs = [1,2,4,7,8], k = 2
Output: 11
Explanation: Assign the jobs the following way:
Worker 1: 1, 2, 8 (working time = 1 + 2 + 8 = 11)
Worker 2: 4, 7 (working time = 4 + 7 = 11)
The maximum working time is 11.


Constraints:
`1 <= k <= jobs.length <= 12`
`1 <= jobs[i] <= 107`

### Solution 
 To solve the problem of minimizing the maximum working time among workers assigned jobs, we can use a backtracking approach along with a binary search technique to efficiently find the optimal solution. 

Here is a step-by-step breakdown of the solution:

### Step 1: Understanding the Problem

We need to partition the jobs into `k` groups (or assignments) such that the maximum sum of any group is minimized. Essentially, we want to distribute the jobs as evenly as possible among the workers.

### Step 2: Using Binary Search

Given the constraints (`1 <= k <= 12` and the length of `jobs` at most 12), we can use binary search to find the answer for the minimum possible maximum working time. 

1. The lower bound of the binary search can be the maximum time of the jobs array since one worker has to take at least one job.
2. The upper bound can be the sum of all job times since that's the scenario where one worker does all the jobs.

### Step 3: Validating Assignments

For a given `mid` value (which represents a potential maximum working time), we need to check if it's feasible to assign jobs such that no worker exceeds this `mid` time. 

We'll implement a backtracking approach where we try to assign each job to one of the workers while keeping track of their respective working times.

### Implementation

Here’s how the code can be structured based on the explanation above, formatted for LeetCode submission:



```python
class Solution:
    def minimumTimeRequired(self, jobs: List[int], k: int) -> int:
        # Function to check if we can partition jobs into k workers with max time <= mid
        def canAssign(jobs, k, mid):
            workers = [0] * k
            
            def backtrack(i):
                if i == len(jobs):
                    return True
                for w in range(k):
                    if workers[w] + jobs[i] <= mid:
                        workers[w] += jobs[i]
                        if backtrack(i + 1):
                            return True
                        workers[w] -= jobs[i]
                    # If current worker has no job assigned, break to avoid unnecessary checks
                    if workers[w] == 0:
                        break
                return False
            
            return backtrack(0)
        
        # Binary search for the minimum maximum working time
        left, right = max(jobs), sum(jobs)
        while left < right:
            mid = (left + right) // 2
            if canAssign(jobs, k, mid):
                right = mid
            else:
                left = mid + 1
        
        return left

```

### Explanation of the Code:

1. **Binary Search**: The `minimumTimeRequired` function employs binary search. We set `left` to the maximum job time (the lowest possible maximum working time) and `right` to the sum of all job times (the highest possible maximum working time).
  
2. **Checking Feasibility (`canAssign` function)**: 
   - This function attempts to assign jobs using a backtracking method. We maintain an array `workers` to track the total working time assigned to each worker.
   - For each job, we attempt to assign it to each worker (if the current addition does not exceed the `mid`). If we can assign all jobs while respecting the `mid` condition, we return true.

3. **Final Return**: The binary search will eventually converge to the minimum possible maximum working time, which is returned at the end.

### Conclusion

This code efficiently minimizes the maximum working time across multiple worker assignments using a combination of binary search and backtracking. This approach works well within the given constraints, providing an optimal solution.

# 1780. Check if Number is a Sum of Powers of Three

### Problem Description 
Given an integer `n`, return `true` if it is possible to represent `n` as the sum of distinct powers of three. Otherwise, return `false`.

An integer `y` is a power of three if there exists an integer `x` such that `y == 3x`.


Example 1:
Input: n = 12
Output: true
Explanation: 12 = 31 + 32

Example 2:
Input: n = 91
Output: true
Explanation: 91 = 30 + 32 + 34

Example 3:
Input: n = 21
Output: false

Constraints:
`1 <= n <= 107`

### Solution 
 To solve the problem of determining if an integer `n` can be expressed as the sum of distinct powers of three, we can take advantage of some properties related to the representation of numbers in base 3.

### Explanation:

1. **Understanding Powers of Three**:
   Powers of three are represented as \(3^0, 3^1, 3^2, \dots\). The distinct sums of these powers create a unique combination that can yield different numbers.

2. **Base 3 Representation**:
   Every integer can be represented uniquely in a numeral system with a base. For our problem, we can look at the representation of `n` in base 3. Specifically, if `n` can be represented as a sum of distinct powers of three, then in base 3, the representation should have digits that are either `0` or `1`. Any digit that is `2` would indicate a non-distinct sum (e.g., using the same power of three twice).

3. **Algorithm**:
   - Convert the integer `n` to its base 3 representation.
   - Check each digit of this representation.
   - If any digit is greater than `1`, return `false`, otherwise, return `true`.

### Complexity:
- The time complexity of converting a number to base 3 is \(O(\log_3 n)\), which is efficient given the constraints.

### Implementation in Python:

Here’s how you would implement this solution in Python, following the LeetCode solution format:



```python
class Solution:
    def checkPowersOfThree(self, n: int) -> bool:
        # To convert n to its base 3 representation
        while n > 0:
            # Get the remainder when divided by 3
            if n % 3 == 2:
                return False  # If we get a 2, we cannot have distinct powers of 3
            n //= 3  # Move to the next base 3 digit
        return True  # If we never saw a '2', return True

# Example usage
solution = Solution()
print(solution.checkPowersOfThree(12))  # Output: True
print(solution.checkPowersOfThree(91))  # Output: True
print(solution.checkPowersOfThree(21))  # Output: False

```

### Explanation of Code:
- The `checkPowersOfThree` method is defined within the `Solution` class.
- We enter a loop that continues as long as `n` is greater than zero.
- Inside the loop, we check the remainder of `n` when divided by 3:
  - If the remainder is `2`, we immediately return `false`.
  - If the remainder is `0` or `1`, we continue the process by dividing `n` by 3 (which effectively shifts to the next base 3 digit).
- If we finish the loop without encountering a `2`, we return `true`, indicating that `n` can be expressed as a sum of distinct powers of three.

This code can be submitted directly on the LeetCode website for the specified problem.

# 1799. Maximize Score After N Operations

### Problem Description 
You are given `nums`, an array of positive integers of size `2 * n`. You must perform `n` operations on this array.

In the `ith` operation (1-indexed), you will:
Choose two elements, `x` and `y`.

Receive a score of `i * gcd(x, y)`.

Remove `x` and `y` from `nums`.

Return the maximum score you can receive after performing `n` operations.

The function `gcd(x, y)` is the greatest common divisor of `x` and `y`.


Example 1:
Input: nums = [1,2]
Output: 1
Explanation: The optimal choice of operations is:
(1 * gcd(1, 2)) = 1

Example 2:
Input: nums = [3,4,6,8]
Output: 11
Explanation: The optimal choice of operations is:
(1 * gcd(3, 6)) + (2 * gcd(4, 8)) = 3 + 8 = 11

Example 3:
Input: nums = [1,2,3,4,5,6]
Output: 14
Explanation: The optimal choice of operations is:
(1 * gcd(1, 5)) + (2 * gcd(2, 4)) + (3 * gcd(3, 6)) = 1 + 4 + 9 = 14

Constraints:
`1 <= n <= 7`
`nums.length == 2 * n`
`1 <= nums[i] <= 106`

### Solution 
 To solve the problem of maximizing the score using the specified operations with the array of integers, we can use a backtracking approach. The basic idea is to always choose two elements from the array, calculate their GCD (greatest common divisor), and then recursively compute the maximum score that can be achieved thereafter.

Here's a breakdown of our approach:

1. **Understanding GCD**: The `gcd(x, y)` function calculates the greatest common divisor of two integers `x` and `y`. Python's `math` library provides a built-in function for this.

2. **Backtracking**: We need to try all possible pairs of numbers to form operations. After choosing a pair, we'll remove them and calculate the subsequent score recursively.

3. **Score Calculation**: The score increases as operations are performed, meaning the first operation contributes `1 * gcd`, the second operation contributes `2 * gcd`, and so on.

4. **Memoization**: Although our constraints are relatively small (`n` can go up to 7 leading to `2n = 14`), it's beneficial to cache results of calculated states to avoid redundant calculations.

5. **Bitmasking**: We can use a bitmask to represent which numbers have been selected (or removed) from the array, allowing us to easily check which elements are left to choose from in our recursive calls.

Here's the implementation of the solution:



```python
import math
from itertools import combinations

class Solution:
    def maxScore(self, nums) -> int:
        n = len(nums) // 2
        dp = {}
        
        def backtrack(mask, step):
            if step > n:
                return 0
            if mask in dp:
                return dp[mask]

            max_score = 0
            for i, j in combinations(range(2 * n), 2):
                if (mask & (1 << i)) == 0 and (mask & (1 << j)) == 0:  # if i and j are not picked yet
                    new_mask = mask | (1 << i) | (1 << j)
                    current_gcd = math.gcd(nums[i], nums[j])
                    score = step * current_gcd
                    max_score = max(max_score, score + backtrack(new_mask, step + 1))
            
            dp[mask] = max_score
            return max_score
        
        return backtrack(0, 1)

# Example usage:
# solution = Solution()
# print(solution.maxScore([1, 2]))  # Output: 1
# print(solution.maxScore([3, 4, 6, 8]))  # Output: 11
# print(solution.maxScore([1, 2, 3, 4, 5, 6]))  # Output: 14

```

### Explanation of the Code:

1. **Imports**: We import the `math` module for the `gcd` function and `combinations` from `itertools` to generate pairs.

2. **Class Declaration**: We define a class `Solution` which contains the method `maxScore`.

3. **Recursion & Memoization**: 
   - The function `backtrack(mask, step)` is defined to find the maximum score.
   - `mask` is a bitmask indicating which elements are already chosen. 
   - `step` keeps track of the current operation number.
   - We check if the current state (mask) has been computed already using `dp`.

4. **Generating Pairs**: We use `itertools.combinations` to generate all unique pairs of indices that are not yet picked (i.e., their corresponding bits in the mask are unset).

5. **Calculating New States**: For each pair, we compute the new mask after selecting the pair and calculate the score. We then recursively call `backtrack` with this new state.

6. **Result Storage**: We store the maximum score in the `dp` dictionary for the current mask before returning the result.

This implementation explores all possible pair selections with the optimal scoring mechanism using backtracking, ensuring that we find the maximum score possible.

