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
 To tackle this problem, we will simulate the addition process similar to how you would add numbers on paper. Each node in the linked list represents a digit. Since the digits are stored in reverse order, each node represents a digit moving from the least significant to the most significant digit.

We will iterate through both linked lists simultaneously, extract their values, and compute the sum digit by digit considering the carry from the previous addition. If the linked list lengths are not equal, we continue with the longer list and the existing carry.

Here’s the plan:
1. Start from the head of both linked lists.
2. Create a dummy node which will be the starting point of the resulting linked list and a current pointer initialized at the dummy.
3. Keep track of a carry value (initially set to 0).
4. Traverse both linked lists while there are nodes left in either list or while there is a carry left.
5. For each node:
   - Add values from both linked lists if they exist and add the carry.
   - Calculate the new digit (value % 10) and update the carry (`value // 10`).
   - Create a new node with the new digit and moving the current pointer to this new node.
6. Return the next of the dummy node (since the dummy node is a placeholder).

Here's the Python code following LeetCode’s format:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # Dummy head for the resultant linked list
        dummyHead = ListNode(0)
        # Initialize pointers
        current = dummyHead
        carry = 0
        
        # While there are still nodes to process or there is a carry
        while l1 is not None or l2 is not None or carry != 0:
            x = l1.val if l1 is not None else 0
            y = l2.val if l2 is not None else 0
            
            # Calculate the sum and carry
            total = x + y + carry
            carry = total // 10
            newNode = ListNode(total % 10)
            
            # Link it in the resulting list
            current.next = newNode
            current = current.next
            
            # Move the l1 and l2 pointers if possible
            if l1 is not None:
                l1 = l1.next
            if l2 is not None:
                l2 = l2.next
        
        return dummyHead.next

# Example usage, you would replace this part with LeetCode test
# l1: 2 -> 4 -> 3
# l2: 5 -> 6 -> 4
l1 = ListNode(2, ListNode(4, ListNode(3)))
l2 = ListNode(5, ListNode(6, ListNode(4)))
sol = Solution()
result = sol.addTwoNumbers(l1, l2)
# Printing result for verification (outside scope in LeetCode normally)
while result:
    print(result.val, end=" -> " if result.next else "")
    result = result.next

```

This code can directly be pasted into LeetCode's Python script editor for the "Add Two Numbers" problem and should successfully execute the test cases provided.

# 19. Remove Nth Node From End of List

### Problem Description 
Given the `head` of a linked list, remove the `nth` node from the end of the list and return its head.

Follow up: Could you do this in one pass?

Example 1:
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]

Example 2:
Input: head = [1], n = 1
Output: []

Example 3:
Input: head = [1,2], n = 1
Output: [1]

Constraints:
The number of nodes in the list is `sz`.

`1 <= sz <= 30`
`0 <= Node.val <= 100`
`1 <= n <= sz`

### Solution 
 To address the given problem, you can employ the "two-pointer" technique to efficiently remove the nth node from the end of a linked list in a single pass. Here's a step-by-step explanation of the approach:

1. **Initialization**: Start with two pointers, `slow` and `fast`. Both are initially pointing to the head of the list.

2. **Advance the fast pointer**: Move the `fast` pointer `n` steps ahead. This will create a gap of `n` nodes between the `slow` and the `fast` pointers.

3. **Move both pointers**: Continue to advance both `slow` and `fast` pointers together until the `fast` pointer reaches the end of the list. By this time, the `slow` pointer will be just before the node that needs to be removed.

4. **Node removal**: Adjust pointers to skip the node after `slow`.

5. **Edge cases**: If you need to remove the head of the list, the head pointer adjustments should be handled separately.

Below is the Python code for the solution on a LeetCode-compatible format, including the definition of the ListNode class as often seen in linked list problems on LeetCode.



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0)  # Create a dummy node that points to the head
        dummy.next = head
        slow = fast = dummy
        
        # Move fast pointer n+1 steps ahead
        for _ in range(n + 1):
            fast = fast.next
        
        # Move both pointers until fast reaches the end
        while fast:
            slow = slow.next
            fast = fast.next
        
        # Remove the nth node from end
        slow.next = slow.next.next
        
        return dummy.next  # Return new head in case original head was removed

# Example usage:
# Construct the list: 1 -> 2 -> 3 -> 4 -> 5
head = ListNode(1)
current = head
for value in [2, 3, 4, 5]:
    current.next = ListNode(value)
    current = current.next

# Create an instance of the solution class
sol = Solution()
new_head = sol.removeNthFromEnd(head, 2)

# Print the resulting list
current = new_head
while current:
    print(current.val, end=" ")
    current = current.next

```

This Python script defines the `ListNode` class necessary for list node operations, outlines a `Solution` class with the function `removeNthFromEnd`, and provides example code to use and test this function. Note that because of the constraints (list size between 1 and 30), this solution is both efficient and practical, adhering to O(n) time complexity as it traverses the list at most twice through the staggered dual-pointer method.

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
 To solve the problem of merging two sorted linked lists into a single sorted linked list using Python, we need to address the structure of the ListNode provided typically in LeetCode problems.

We first define the class `ListNode` which is used to represent a node in a singly linked list. Each node contains an integer data value (`val`) and a reference/link (`next`) to the next node in the list.

The algorithm to merge the two lists involves:
1. Creating a dummy node to help simplify the code by providing a non-null node we can use as the start of our new list.
2. Maintaining a `current` reference that points to the dummy node initially, and is used to build the merged list step-by-step by choosing the smaller element from the front of the lists `l1` and `l2`.
3. Iterating while there are still elements in both `l1` and `l2`. Comparing the front element of each and linking the node with the smaller value to the `current.next`, then moving `current` forward and the reference (`l1` or `l2`) to the next node in its list.
4. By the end of the iteration, one might still have remaining nodes in either `l1` or `l2` (not both since one will be empty). Point `current.next` to the non-empty list. This operation will add all remaining elements as they are already sorted.
5. The `head` of the merged linked list is `dummy.next`.

Here is the code in the format suitable for execution on LeetCode:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        # Dummy node that acts as the start of the merged list
        dummy = ListNode(-1)
        current = dummy
        
        # As long as either list still has elements, perform merge operation
        while l1 and l2:
            if l1.val <= l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next
        
        # Connect the remaining nodes from whichever list is non-empty
        current.next = l1 if l1 else l2
        
        return dummy.next

```

This implementation is both straightforward and efficient, with a time complexity of O(n + m) and space complexity of O(1), where n and m are the lengths of l1 and l2 respectively. Because the solution only constructs a few additional nodes (mainly the dummy node), it efficiently manages memory without duplicating the entire lists.

# 23. Merge k Sorted Lists

### Problem Description 
You are given an array of `k` linked-lists `lists`, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.


Example 1:
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6

Example 2:
Input: lists = []
Output: []

Example 3:
Input: lists = [[]]
Output: []

Constraints:
`k == lists.length`
`0 <= k <= 10^4`
`0 <= lists[i].length <= 500`
`-10^4 <= lists[i][j] <= 10^4`
`lists[i]` is sorted in ascending order.

The sum of `lists[i].length` won't exceed `10^4`.

### Solution 
 To solve the problem of merging `k` sorted linked-lists into one sorted linked-list on the LeetCode platform, we'll use a Python implementation provided below.

The code will utilize Python's `heapq` module, which implements a min-heap, to efficiently manage the merging process. Min-heaps help in retrieving the smallest element in constant time, which is essential for merging sorted lists. The algorithm consists of the following steps:

1. Initialize a heap to store pairs of (node value, list index, node) for the head of each linked list.
2. Extract the smallest element from the heap and append it to the merged list.
3. Move to the next node in the linked list from which the smallest element was extracted, and insert the next node's value into the heap.
4. Repeat the process until the heap is empty.

We also need to define a ListNode class according to the structure used in LeetCode.

Here's the Python code:



```python
from heapq import heappop, heappush
from typing import List
from typing import Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return None

        # Priority queue/min-heap
        heap = []

        # Initialize the heap with the first element of each list
        for index, lst in enumerate(lists):
            if lst:
                heappush(heap, (lst.val, index, lst))

        # Dummy node to help easily return the result
        dummy = ListNode()
        current = dummy

        # Process the heap
        while heap:
            val, idx, node = heappop(heap)
            # Link the smallest element to the current node
            current.next = ListNode(val)
            current = current.next

            # Move to the next element in the list and add it to the heap
            if node.next:
                heappush(heap, (node.next.val, idx, node.next))

        return dummy.next

# Helper function to convert list of lists to list of ListNode
def list_to_nodes(lists):
    result = []
    for lst in lists:
        dummy = ListNode()
        current = dummy
        for val in lst:
            current.next = ListNode(val)
            current = current.next
        result.append(dummy.next)
    return result

# To visualize the result for testing:
def nodes_to_list(node):
    result = []
    while node:
        result.append(node.val)
        node = node.next
    return result

# Testing the function:
lists_input = [[1,4,5], [1,3,4], [2,6]]
nodes_input = list_to_nodes(lists_input)
solution = Solution()
merged_head = solution.mergeKLists(nodes_input)
print(nodes_to_list(merged_head))  # Output: [1, 1, 2, 3, 4, 4, 5, 6]

```

This solution handles all edge cases, such as empty lists and single-element lists. It has a time complexity of \(O(N \log k)\), where \(N\) is the total number of elements across all lists, and \(k\) is the number of lists. This is because each element is processed through the heap, which takes \(O(\log k)\) time. This is particularly efficient given the constraints.

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
 To solve the problem of swapping every two adjacent nodes in a linked list, we will need to adjust the links between the nodes directly, without modifying the values within the nodes. This requirement is in line with the follow-up constraint. Here's how we can approach this:

### Approach

1. **Iterative Approach**: 
    - Use two pointers, one to track the current node and another to track the previous node's next link (to be updated after the swap).
    - For each pair, swap the nodes by adjusting the links.
    - Use a dummy node to simplify the head management and initial swap handling.
    - Move the pointers forward by two positions after each swap.

### Step-by-Step Explanation
1. If the head is `None` or has only one node, return the head as there are no nodes to swap.
2. Initialize a dummy node and point its next to the head. This will help maintain the head of the resultant swapped list.
3. Use three pointers:
    - `prev_node` initialized to point to the dummy node.
    - `current` initialized to point to the first node of the list.
4. In each iteration:
    - Check if the current node and the next node are available for swapping.
    - Swap the two nodes by adjusting the links.
    - Move the `prev_node` two steps forward (to the node just before the next pair).
    - Move the `current` two steps forward (to the next pair's first node).
5. The loop continues until there are fewer than two nodes available for swapping.
6. Finally, return the list starting from `dummy_head.next`.

### Implementation in Python


```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy_head = ListNode(0)
        dummy_head.next = head
        
        prev_node = dummy_head
        
        while head and head.next:
            # Nodes to be swapped
            first_node = head
            second_node = head.next
            
            # Swapping
            prev_node.next = second_node
            first_node.next = second_node.next
            second_node.next = first_node
            
            # Reinitializing the head and prev_node for next swap
            prev_node = first_node
            head = first_node.next
        
        # Return the new head node.
        return dummy_head.next

```

### Example Usage
To use this code in a practical example, you would first need to create a linked list:


```python
# Helper function to create a linked list from a list of values
def create_linked_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for value in values[1:]:
        current.next = ListNode(value)
        current = current.next
    return head

# Helper function to print the linked list
def print_linked_list(head):
    current = head
    while current:
        print(current.val, end=' -> ')
        current = current.next
    print('None')

# Example Usage
sol = Solution()
ll = create_linked_list([1, 2, 3, 4])
result = sol.swapPairs(ll)
print_linked_list(result)  # Expected Output: 2 -> 1 -> 4 -> 3 -> None

```

This solution adheres to the problem's constraints and efficiently swaps adjacent nodes with a time complexity of O(n) and a space complexity of O(1).

# 25. Reverse Nodes in k-Group

### Problem Description 
Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.

k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.

Follow up:
Could you solve the problem in `O(1)` extra memory space?
You may not alter the values in the list's nodes, only nodes itself may be changed.


Example 1:
Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]

Example 2:
Input: head = [1,2,3,4,5], k = 3
Output: [3,2,1,4,5]

Example 3:
Input: head = [1,2,3,4,5], k = 1
Output: [1,2,3,4,5]

Example 4:
Input: head = [1], k = 1
Output: [1]

Constraints:
The number of nodes in the list is in the range `sz`.

`1 <= sz <= 5000`
`0 <= Node.val <= 1000`
`1 <= k <= sz`

### Solution 
 To solve the problem of reversing a linked list `k` nodes at a time, we can proceed with the following approach using O(1) extra memory space:

1. **Count Nodes**: Determine the length of the list to know when to stop reversing if the remainder of the list is smaller than `k`.
2. **Iterative Reversal**: Reverse every consecutive set of `k` nodes. Use three-pointers (`prev`, `current`, and `next`) to handle the node reversal:
   - `prev` points to the last node of the previous part of the linked list that has been reversed or, initially, a dummy node.
   - `current` points to the current node being processed.
   - `next` indicates the next node in the list.
3. **Connectivity**: Connect reversed parts with the non-reversed parts. The first node in each batch of `k` nodes before reversal becomes the last after reversal, and it should point to the head of the next batch after its reversal.
4. **Handle Remainders**: If the count of nodes is not a multiple of `k`, the remainder should remain untouched as it is.

The algorithm will handle node pointer rearrangement to achieve in-place reversal without using more than constant extra space.

Let's implement this solution in Python suitable to be run directly on LeetCode:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if not head or k == 1:
            return head

        # Dummy node initialization which will point to the start of the list
        dummy = ListNode(0)
        dummy.next = head
        prev_end = dummy
        
        # Count total number of nodes in the list
        current = head
        length = 0
        while current:
            length += 1
            current = current.next
        
        # Perform the reversal
        while length >= k:
            current = prev_end.next
            next = None
            prev = None
            
            # Reverse k nodes
            for i in range(k):
                next = current.next
                current.next = prev
                prev = current
                current = next
            
            # Connect the previous part with the newly reversed list
            front = prev_end.next   # front of the (sub)list that has been reversed
            prev_end.next.next = current  # previous 'front' should now point to 'current'
            prev_end.next = prev  # 'prev_end' should now point to the beginning of the reversed sublist
            
            prev_end = front  # move the 'prev_end' to the end of the reversed part
            length -= k  # decrease the remaining length by k since we reversed k elements
        
        return dummy.next  # 'dummy.next' now points to the head of the modified list

# Example usage
if __name__ == "__main__":
    # Example: reversing every 2 nodes in the list [1, 2, 3, 4, 5]
    node5 = ListNode(5)
    node4 = ListNode(4, node5)
    node3 = ListNode(3, node4)
    node2 = ListNode(2, node3)
    head = ListNode(1, node2)
    
    solution = Solution()
    new_head = solution.reverseKGroup(head, 2)
    
    # Print new list
    current = new_head
    while current:
        print(current.val, end=" -> ")
        current = current.next
    print("None")

```

This code defines the `ListNode` class (as required by LeetCode for linked list problems) and a `Solution` class with the `reverseKGroup` method implemented as described above. The reversal logic is illustrated clearly, and the example usage shows how to create a tree and print the reversed result.

# 61. Rotate List

### Problem Description 
Given the `head` of a linked list, rotate the list to the right by `k` places.


Example 1:
Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]

Example 2:
Input: head = [0,1,2], k = 4
Output: [2,0,1]

Constraints:
The number of nodes in the list is in the range `[0, 500]`.

`-100 <= Node.val <= 100`
`0 <= k <= 2 * 109`

### Solution 
 

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head or not head.next or k == 0:
            return head
        
        # First find the length of the linked list
        length = 1
        current = head
        while current.next:  # Traverse the list to find the tail and count the length
            current = current.next
            length += 1
        
        # Link the tail back to the head (circle the list)
        current.next = head
        
        # Find the effective rotations needed (as rotating length times doesn't change the list)
        k = k % length
        
        # Find the new tail, which is (length - k) elements from the head
        steps_to_new_tail = length - k
        new_tail = head
        for _ in range(steps_to_new_tail - 1):
            new_tail = new_tail.next
        
        # new head will be next of new_tail
        new_head = new_tail.next
        new_tail.next = None  # Break the circle
        
        return new_head

# To use the above solution, you need to set up some utility functions to convert between list structures used in LeetCode examples and ListNode structures.

def list_to_linkedlist(elements):
    """Helper function to convert a list to a linked list."""
    dummy = ListNode(0)
    ptr = dummy
    for element in elements:
        ptr.next = ListNode(element)
        ptr = ptr.next
    return dummy.next

def linkedlist_to_list(node):
    """Helper function to convert a linked list to a list."""
    result = []
    while node:
        result.append(node.val)
        node = node.next
    return result

# Example usage:
# Convert list to linkedlist
head = list_to_linkedlist([1,2,3,4,5])
k = 2

# Instantiate the Solution class and use the rotateRight method
sol = Solution()
result = sol.rotateRight(head, k)

# Convert linkedlist back to list to check the result
output = linkedlist_to_list(result)
print(output)  # Output should be: [4,5,1,2,3]


```

### Explanation

1. **Edge Cases Handling**: Firstly, the function checks if the `head` is `None`, or if there’s only one node, or if `k` is zero. In these cases, no rotation is needed, and the original list can simply be returned.

2. **Counting Length with Tail Discovery**: We scan through the list to determine the list's length and locate the tail element.

3. **Circular Link Creation**: By connecting the tail of the list back to the head, we enable a seamless transition from the end back to the start of the list, making rotations straightforward.

4. **Effective Rotations Calculation**: Since rotating the list by its length `n` times results in the same list, we compute `k modulo n` to find out the necessary number of practical rotations.

5. **New Tail and Head Determination**: We locate the new tail, which is exactly `(length - k)` nodes away from the original head, to accommodate `k` rotations. The element right after this new tail is considered the new head after rotation.

6. **Breaking the Loop**: Finally, to maintain a standard singly linked list structure, we sever the connection between the new tail and the new head.

This solution is efficient because it involves a simple linear scan to determine the list's size and then another pass to adjust the relevant links, leading to a time complexity of O(n), which is efficient given the constraints.

# 82. Remove Duplicates from Sorted List II

### Problem Description 
Given the `head` of a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list. Return the linked list sorted as well.


Example 1:
Input: head = [1,2,3,3,4,4,5]
Output: [1,2,5]

Example 2:
Input: head = [1,1,1,2,3]
Output: [2,3]

Constraints:
The number of nodes in the list is in the range `[0, 300]`.

`-100 <= Node.val <= 100`
The list is guaranteed to be sorted in ascending order.

### Solution 
 To solve the problem of removing duplicate nodes from a sorted linked list, we need an efficient way to identify and skip nodes that appear more than once. A simple approach involves using a sentinel (or dummy) node and two pointers to traverse the linked list: one for iterating through the list (`current`) and another to indicate the last known unique node (`prev`). 

Here's a stepwise break-down of the Python solution:

1. Create a dummy node (`dummy`) and let it point to the `head`. This dummy node simplifies edge cases, especially when the head itself might need deletion.
2. Initialize `prev` to this dummy node.
3. Use a pointer `current` to iterate through the list starting from the head node.
4. While moving through the list:
   - If `current` and its next node have the same value (indicating a duplicate):
     - Use a nested loop to skip all nodes having this duplicate value.
     - Once all duplicates are skipped, link `prev.next` to the node right after the last duplicate (essentially, remove all duplicates from the main list).
   - Otherwise, if no duplicates are found, just move `prev` to `current`.
   - In both cases, move `current` forward.
5. The `next` node of the dummy will point to the head of the modified linked list without duplicates.
6. Return this result.

Below is the LeetCode-compatible Python code to achieve the above steps:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        # Edge case: when the list is empty or has only one node, just return it.
        if not head or not head.next:
            return head
        
        # Create a dummy node and initiate prev to dummy.
        dummy = ListNode(0, head)
        prev = dummy
        
        # Initialize current to head, which will iterate over the list.
        current = head
        
        while current:
            # Check if current node and the next node have same value
            if current.next and current.val == current.next.val:
                # Skip all nodes that have the same value.
                while current.next and current.val == current.next.val:
                    current = current.next
                # Connect prev's next to current's next (skip all duplicates)
                prev.next = current.next
            else:
                # No duplication, we can safely move prev to current
                prev = current
            # Move current to the next node.
            current = current.next
        
        # Return the next of dummy, which is the head of the new list without duplicates.
        return dummy.next

# Example usage:
# List: 1->2->3->3->4->4->5
# After processing, should become 1->2->5.

```

This code will handle cases where duplicates appear right at the beginning of the list, in the middle, and at the end, effectively removing any and all consecutive duplicate nodes and leaving only unique numbers from the original sorted list.

# 83. Remove Duplicates from Sorted List

### Problem Description 
Given the `head` of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.


Example 1:
Input: head = [1,1,2]
Output: [1,2]

Example 2:
Input: head = [1,1,2,3,3]
Output: [1,2,3]

Constraints:
The number of nodes in the list is in the range `[0, 300]`.

`-100 <= Node.val <= 100`
The list is guaranteed to be sorted in ascending order.

### Solution 
 To solve this problem, we can use a straightforward approach that involves iterating through the linked list and deleting nodes that have duplicate values. Since the list is already sorted, all duplicates will be adjacent. We can leverage this property to efficiently remove duplicates by keeping track of the previous distinct node and continuously comparing it with the current node.

Here's a step-by-step process of what the solution will do:

1. If the head of the list is `None` (indicating the list is empty), immediately return `None` as there can't be any duplicates in an empty list.
2. Use two pointers: `current` to iterate over the list and `prev` to keep a reference to the last node that wasn't a duplicate.
3. Traverse the linked list from the head using `current`. For each node:
   a. Check if it's a duplicate by comparing its value with `prev`.
   b. If it's not a duplicate, update `prev` to this node.
   c. If it is a duplicate, modify the next pointer of `prev` to skip the current node, effectively removing it from the list.
4. Return the modified list starting from the head.

This approach works directly on the list without using any extra space (except pointers), and runs in O(n) time, where n is the number of nodes in the list, since we simply traverse the list once.

Let's write the Python code matching this description:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return head
        
        prev = head
        current = head.next
        
        while current:
            if current.val == prev.val:
                # Skip the duplicate
                prev.next = current.next
                current = current.next
            else:
                # Not a duplicate, move the prev pointer
                prev = current
                current = current.next
        
        return head

```

For the given problem, you can test this code on the LeetCode platform by copying the class `Solution` and the definition `ListNode` into their coding environment. This implementation efficiently handles the linked list in place and conforms to typical constraints and average problem requirements on the platform.

# 86. Partition List

### Problem Description 
Given the `head` of a linked list and a value `x`, partition it such that all nodes less than `x` come before nodes greater than or equal to `x`.

You should preserve the original relative order of the nodes in each of the two partitions.


Example 1:
Input: head = [1,4,3,2,5,2], x = 3
Output: [1,2,2,4,3,5]

Example 2:
Input: head = [2,1], x = 2
Output: [1,2]

Constraints:
The number of nodes in the list is in the range `[0, 200]`.

`-100 <= Node.val <= 100`
`-200 <= x <= 200`

### Solution 
 To solve the problem of partitioning a linked list around a given value `x`, we can use a two-pointer approach to manage the two partitions:
1. One pointer for the part of the linked list with values less than `x`.
2. Another pointer for the part with values greater than or equal to `x`.

The basic steps will involve:
- Initializing two dummy nodes: "before" to track nodes less than `x` and "after" to track nodes greater or equal to `x`.
- Iterating through the linked list and appending each node to either the "before" list or the "after" list based on its value.
- Connecting the end of the "before" list to the start of the "after" list, ensuring the ordering requirements are met.
- Finally, returning the head of the newly formed partitioned list starting from the "before" dummy node's next reference.

Here’s the Python code using the mentioned algorithm:



```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        # Create dummy nodes to start the less and greater/equal lists
        before = before_head = ListNode(0)
        after = after_head = ListNode(0)
        
        # Iterate through the original list and divide the nodes into before and after according to x
        while head:
            if head.val < x:
                before.next = head
                before = before.next
            else:
                after.next = head
                after = after.next
            head = head.next
        
        # Ensure the last node of the 'after' list points to None
        after.next = None
        # Connect the 'before' list to the 'after' list
        before.next = after_head.next
        
        return before_head.next

# Optionally, here's a helper function to print list nodes for easy testing
def print_list(node):
    while node:
        print(node.val, end=" ")
        node = node.next
    print()
    
# Example usage
# Constructing the list [1,4,3,2,5,2]
head = ListNode(1, ListNode(4, ListNode(3, ListNode(2, ListNode(5, ListNode(2))))))
x = 3

sol = Solution()
result = sol.partition(head, x)
print_list(result)  # Output: 1 2 2 4 3 5

```

This Python code defines a `ListNode` class for basic linked list operations, a `Solution` class with the `partition` method, and finally a small helper to visualize the output list "print_list". This should effectively split and reorder nodes around the specified pivot value while maintaining internal order in the partitions, and it can be run directly on LeetCode given their interface for linked list problems.

# 92. Reverse Linked List II

### Problem Description 
Given the `head` of a singly linked list and two integers `left` and `right` where `left <= right`, reverse the nodes of the list from position `left` to position `right`, and return the reversed list.


Example 1:
Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]

Example 2:
Input: head = [5], left = 1, right = 1
Output: [5]

Constraints:
The number of nodes in the list is `n`.

`1 <= n <= 500`
`-500 <= Node.val <= 500`
`1 <= left <= right <= n`
Follow up: Could you do it in one pass?

### Solution 
 Here's the Python code that solves the problem described using the LeetCode format. The solution involves reversing the sublist defined between positions `left` and `right` and ensuring that the rest of the list remains unchanged.



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        if not head or left == right:
            return head
        
        # Create a dummy node which acts as the prev node for the head node
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        
        # Move `prev` to just before the `left`th node
        for _ in range(left - 1):
            prev = prev.next
        
        # Reverse the segment from `left` to `right`
        reverse = None
        current = prev.next
        for _ in range(right - left + 1):
            next = current.next
            current.next = reverse
            reverse = current
            current = next
        
        # Connect the reversed sublist back to the rest of the list
        prev.next.next = current  # Connect the end of the reversed list to the node after `right`
        prev.next = reverse  # Connect the node before `left` to the start of the reversed list
        
        return dummy.next

# Helper function to convert list to linked list
def list_to_linkedlist(input_list):
    dummy = ListNode(0)
    temp = dummy
    for elem in input_list:
        temp.next = ListNode(elem)
        temp = temp.next
    return dummy.next

# Helper function to convert linked list to list
def linkedlist_to_list(node):
    result = []
    while node:
        result.append(node.val)
        node = node.next
    return result

# Example usage:
# input_list = [1, 2, 3, 4, 5]
# head = list_to_linkedlist(input_list)
# solution = Solution()
# new_head = solution.reverseBetween(head, 2, 4)
# output_list = linkedlist_to_list(new_head)
# print(output_list)  # Output: [1, 4, 3, 2, 5]

```

### Explanation:
- **ListNode class**: Defines the singly linked list nodes.
- **Solution class with reverseBetween method**: Contains the logic for reversing the nodes from position `left` to `right`.
  - A dummy node is used to simplify edge cases, such as when reversing starting from the first node.
  - `prev` points to the node just before the `left`th node.
  - The sublist is reversed using a simple iterative approach.
  - After reversal, the list is reconnected properly.
  
This solution ensures that the reversal is achieved using a one-pass solution, respecting the problem's constraints and optimizing for a single traversal of the critical list segments.

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
 To solve this problem, we'll convert a sorted linked list into a height-balanced binary search tree (BST). 

**Key Insight**: The middle node of a sorted linked list should serve as the root of a height-balanced BST, with the left half forming the left subtree and the right half forming the right subtree.

**Challenges**:
1. **Traversal**: Linked lists cannot be accessed by index as arrays can, so we need a method to find the middle element efficiently.
2. **Recursion**: Building a BST recursively using the divide-and-conquer strategy to ensure height balance.

**Approach**:
- Use the **two-pointer technique** ("slow and fast pointer" method) to locate the middle node of the linked list. This will serve as the root of the BST.
- Recursively apply the same technique to construct the left and right subtrees:
  - The part of the list from the start to the node before the middle becomes the left subtree.
  - The part from after the middle to the end becomes the right subtree.
- Handle base cases:
  - If the list is empty, return `None`.
  - If the list consists of a single node, return it as a leaf node.
  
This approach efficiently divides the list and constructs the BST in a manner similar to binary search, ensuring both log(n) depth (by a balanced division) and optimal tree construction.

Here's the Python code implementing the above approach suitable for direct execution in LeetCode:



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
        if not head:
            return None
        if not head.next:
            return TreeNode(head.val)
        
        # Function to find the middle element of the linked list
        def findMiddle(left, right):
            slow = fast = left
            prev = None
            
            while fast != right and fast.next != right:
                prev = slow
                slow = slow.next
                fast = fast.next
                if fast.next:
                    fast = fast.next
                    
            # 'prev' points to the element before the middle
            if prev:
                prev.next = None  # Break the link
            
            return slow
        
        # Recursive function to convert the current linked list range into a BST
        def convertListToBST(left, right):
            if left == right:
                return None
            
            mid = findMiddle(left, right)
            node = TreeNode(mid.val)
            
            node.left = convertListToBST(left, mid)
            node.right = convertListToBST(mid.next, right)
            
            return node
        
        return convertListToBST(head, None)

# Below here would be additional handling if main testing or for system integration:
# For example, list and tree handling functions

```

In this code:
1. `findMiddle(left, right)` function finds the middle node in the list section between `left` and `right`.
2. `convertListToBST(left, right)` function recursively builds the BST using the above-found middle node.
3. Finally, `sortedListToBST` on the entire list gives the required height-balanced BST.

# 138. Copy List with Random Pointer

### Problem Description 
A linked list of length `n` is given such that each node contains an additional random pointer, which could point to any node in the list, or `null`.

Construct a deep copy of the list. The deep copy should consist of exactly `n` brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the `next` and `random` pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.

For example, if there are two nodes `X` and `Y` in the original list, where `X.random --> Y`, then for the corresponding two nodes `x` and `y` in the copied list, `x.random --> y`.

Return the head of the copied linked list.

The linked list is represented in the input/output as a list of `n` nodes. Each node is represented as a pair of `[val, random_index]` where:
`val`: an integer representing `Node.val`
`random_index`: the index of the node (range from `0` to `n-1`) that the `random` pointer points to, or `null` if it does not point to any node.

Your code will only be given the `head` of the original linked list.


Example 1:
Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]

Example 2:
Input: head = [[1,1],[2,1]]
Output: [[1,1],[2,1]]

Example 3:
Input: head = [[3,null],[3,0],[3,null]]
Output: [[3,null],[3,0],[3,null]]

Example 4:
Input: head = []
Output: []
Explanation: The given linked list is empty (null pointer), so return null.


Constraints:
`0 <= n <= 1000`
`-10000 <= Node.val <= 10000`
`Node.random` is `null` or is pointing to some node in the linked list.

### Solution 
 To solve this problem, we follow a two-step approach often used for deep copying linked lists with additional pointers, such as the `random` pointer in our case.

1. **Interweaving Copying Phase**: Go through the original list and create a new copy of each node, and interweave them within the original list. That is, for every original node 'A', create a copied node 'A'', and set 'A.next = A''', and 'A''.next = A.next'. This creates an alternating pattern of original and copied nodes.
   
2. **Assign Random Pointers**: Traverse through the list again. Utilize the `next` connections to assign the `random` pointers for the copied nodes. If `A.random` exists in the original list, let `A''` (the copied node of 'A') point to `A.random.next` which is the cloned node equivalent of `A.random`.

3. **Uninterweaving Split Phase**: After setting up all the pointers correctly, restore the original list and extract the deep copied list by splitting the interwoven nodes.

Now, let's implement this approach:



```python
# Definition for a Node.
class Node:
    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return None
        
        # Step 1: Create the interwoven list of old and copied nodes
        cur = head
        while cur:
            new_node = Node(cur.val, cur.next)
            cur.next = new_node
            cur = new_node.next
            
        # Step 2: Assign random pointers to the copied nodes
        cur = head
        while cur:
            if cur.random:
                cur.next.random = cur.random.next
            cur = cur.next.next
        
        # Step 3: Split the lists
        cur = head
        copy_head = cur.next
        copy = copy_head
        while cur:
            cur.next = cur.next.next
            if copy.next:
                copy.next = copy.next.next
            cur = cur.next
            copy = copy.next
        
        return copy_head

```

This code follows the LeetCode format and can be run directly on their website provided that the definition for `Node` class is used. Each phase of the algorithm ensures that the copied list reflects the same 'next' and 'random' configurations as the original list, with no connections between the copied nodes and the original nodes. This implementation efficiently performs in linear time, O(n), with constant space, O(1), ignoring the space required for the new nodes themselves.

# 141. Linked List Cycle

### Problem Description 
Given `head`, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to. Note that `pos` is not passed as a parameter.

Return `true` if there is a cycle in the linked list. Otherwise, return `false`.


Example 1:
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).


Example 2:
Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.


Example 3:
Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.


Constraints:
The number of the nodes in the list is in the range `[0, 104]`.

`-105 <= Node.val <= 105`
`pos` is `-1` or a valid index in the linked-list.

Follow up: Can you solve it using `O(1)` (i.e. constant) memory?

### Solution 
 

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        """
        To determine if a linked list has a cycle, we'll use the Floyd's Tortoise and Hare algorithm.
        This method uses two pointers, a slow pointer and a fast pointer. The slow pointer moves one step
        at a time while the fast pointer moves two steps at a time. If there is no cycle, the fast pointer
        will reach the end of the list. If there is a cycle, the fast pointer will eventually meet the slow
        pointer inside the cycle.

        :type head: ListNode
        :rtype: bool
        """

        if head is None:
            return False
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
            
        return False

# Explanation:
# 1. We initialize two pointers, `slow` and `fast`, both pointing to the head of the linked list.
# 2. We traverse the list in a loop. In each iteration, `slow` moves one step and `fast` moves two steps.
# 3. If at any point `slow` equals `fast`, it means that `fast` has caught up with `slow` within a cycle,
#    so there is a cycle.
# 4. The loop terminates if `fast` or `fast.next` becomes `None`, meaning we reached the end without finding a cycle.
# 5. The function returns `True` if a cycle is found, otherwise it returns `False`.

``` 

The provided code implements a widely known two-pointer technique called "Floyd's Tortoise and Hare" algorithm to detect cycles in a linked list. This solution is efficient as it only traverses the list in linear time, i.e., O(n), with a space complexity of O(1), which meets the constraint of using constant memory. This algorithm is optimal for detecting cycles in terms of both time and space efficiency, especially considering the follow-up constraint.

# 142. Linked List Cycle II

### Problem Description 
Given a linked list, return the node where the cycle begins. If there is no cycle, return `null`.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to. Note that `pos` is not passed as a parameter.

Notice that you should not modify the linked list.


Example 1:
Input: head = [3,2,0,-4], pos = 1
Output: tail connects to node index 1
Explanation: There is a cycle in the linked list, where tail connects to the second node.


Example 2:
Input: head = [1,2], pos = 0
Output: tail connects to node index 0
Explanation: There is a cycle in the linked list, where tail connects to the first node.


Example 3:
Input: head = [1], pos = -1
Output: no cycle
Explanation: There is no cycle in the linked list.


Constraints:
The number of the nodes in the list is in the range `[0, 104]`.

`-105 <= Node.val <= 105`
`pos` is `-1` or a valid index in the linked-list.

Follow up: Can you solve it using `O(1)` (i.e. constant) memory?

### Solution 
 To solve the problem of finding the node where the cycle begins in a linked list, we'll use a common technique known as Floyd's Cycle Detection Algorithm, also known as the "tortoise and the hare" approach. This method uses two pointers progressing through the list at different speeds — one moves one step at a time (slow pointer) and the other moves two steps at a time (fast pointer). If there's a cycle, the fast pointer will eventually meet the slow pointer within the cycle.

Once the two pointers meet within the cycle, the starting point of the cycle can be found by moving one pointer to the start of the list and then moving both pointers at the same speed; both pointers will meet at the starting node of the cycle.

This algorithm operates in O(n) time complexity with O(1) space complexity because it only uses two pointers regardless of the size of the input list.

Here's the solution formatted for a direct run on LeetCode:



```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if not head:
            return None
        
        slow = fast = head
        has_cycle = False
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                has_cycle = True
                break
        
        if not has_cycle:
            return None
        
        # Find the entry point to the cycle
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        
        return slow

```

**Explanation:**
1. First, we check if the list is empty. If it is, we return `None`.
2. We then initialize two pointers `slow` and `fast` at the head of the list.
3. We iterate through the list with these pointers: `slow` moves one step at a time while `fast` moves two steps at a time. If they meet, this confirms the presence of a cycle (`has_cycle` is set to `True`).
4. If no cycle is detected (i.e., the fast pointer reaches the end of the list), return `None`.
5. To find where the cycle starts, move the `slow` pointer back to the head of the list and then move both pointers one step at a time. The point at which they meet again is the start of the cycle.

This solution respects the constraints and follow-up condition, providing an efficient method to detect the start of a cycle in the linked list.

# 143. Reorder List

### Problem Description 
You are given the head of a singly linked-list. The list can be represented as:
L0 → L1 → ... → Ln - 1 → Ln
Reorder the list to be on the following form:
L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → ...

You may not modify the values in the list's nodes. Only nodes themselves may be changed.


Example 1:
Input: head = [1,2,3,4]
Output: [1,4,2,3]

Example 2:
Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]

Constraints:
The number of nodes in the list is in the range `[1, 5 * 104]`.

`1 <= Node.val <= 1000`

### Solution 
 To solve the problem where we reorder a singly linked list according to the given pattern, we can follow these steps:

1. **Find the Middle of the List:** Use the fast and slow pointer technique (Tortoise and Hare Algorithm) to determine the middle of the list. This is because the operation will be more accessible when we split the list.

2. **Reverse the Second Part of the List:** Once the middle is found, reverse the second half of the list. This will make appending the last element to the first and so on, more straightforward.

3. **Merge Two Halves:** Start from the head of the first half and the head of the reversed second half. Alternately connect nodes from each part until the reordered list is achieved.

Here is the Python code implementing the above logic that can be run directly on the LeetCode platform:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head or not head.next or not head.next.next:
            return
        
        # Step 1: Find the middle of the linked list
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # Step 2: Reverse the second half of the list
        second = slow.next
        slow.next = None
        prev = None
        while second:
            tmp = second.next
            second.next = prev
            prev = second
            second = tmp
            
        # Step 3: Merge the first and the reversed second half
        first, second = head, prev
        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first, second = tmp1, tmp2

# The code does not return anything because it modifies the list in place. The function expects a ListNode as input.

```

This code achieves in-place reordering with a time complexity of O(n), where n is the number of nodes in the list due to the traversal of the list a constant number of times (up to three times). The space complexity is O(1) since we are only using pointers and not additional data structures for our operations.

To use this code, input configurations can be managed within a test harness that constructs linked list instances from arrays and then invokes `reorderList`.
For instance:


```python
def list_to_nodes(lst):
    head = ListNode(lst[0])
    current = head
    for value in lst[1:]:
        current.next = ListNode(value)
        current = current.next
    return head

def nodes_to_list(head):
    lst = []
    while head:
        lst.append(head.val)
        head = head.next
    return lst

# Example run
lst = [1, 2, 3, 4, 5]
head = list_to_nodes(lst)
sol = Solution()
sol.reorderList(head)

# Convert nodes back to list and print result
print(nodes_to_list(head))

```

This helper will convert a list to a linked list and vice versa, allowing us to easily see the results of our function.

# 147. Insertion Sort List

### Problem Description 
Given the `head` of a singly linked list, sort the list using insertion sort, and return the sorted list's head.

The steps of the insertion sort algorithm:
Insertion sort iterates, consuming one input element each repetition and growing a sorted output list.

At each iteration, insertion sort removes one element from the input data, finds the location it belongs within the sorted list and inserts it there.

It repeats until no input elements remain.

The following is a graphical example of the insertion sort algorithm. The partially sorted list (black) initially contains only the first element in the list. One element (red) is removed from the input data and inserted in-place into the sorted list with each iteration.


Example 1:
Input: head = [4,2,1,3]
Output: [1,2,3,4]

Example 2:
Input: head = [-1,5,3,4,0]
Output: [-1,0,3,4,5]

Constraints:
The number of nodes in the list is in the range `[1, 5000]`.

`-5000 <= Node.val <= 5000`

### Solution 
 

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        # Base case: if the list is empty or has one element, it is already sorted.
        if not head or not head.next:
            return head
        
        # Create a dummy node to make edge case handling easier
        dummy = ListNode(0)
        # Pointer to the sorted part of the list
        sorted_tail = dummy
        
        while head:
            current = head
            # Move head to the next element because we are processing the current.
            head = head.next
            
            # Insert current in the sorted list.
            # If current is less than the last element in the sorted list,
            # start from the dummy node again. This happens when the current
            # element is smaller than any in the sorted part.
            if sorted_tail.val > current.val:
                sorted_tail = dummy
            
            # Traverse the sorted part to find the right position for current.
            while sorted_tail.next and sorted_tail.next.val < current.val:
                sorted_tail = sorted_tail.next
            
            # Insert current between sorted_tail and sorted_tail.next
            current.next = sorted_tail.next
            sorted_tail.next = current
        
        return dummy.next

# Helper to create a linked list from a list
def create_linked_list(lst):
    head = ListNode(lst[0])
    current = head
    for value in lst[1:]:
        current.next = ListNode(value)
        current = current.next
    return head

# Helper to print linked list
def print_linked_list(head):
    current = head
    while current:
        print(current.val, end=" -> ")
        current = current.next
    print("None")

# Example Usage
# head = [4,2,1,3]
head = create_linked_list([4,2,1,3])
solution = Solution()
sorted_head = solution.insertionSortList(head)
print_linked_list(sorted_head)  # Output: 1 -> 2 -> 3 -> 4 -> None

# head = [-1,5,3,4,0]
head = create_linked_list([-1,5,3,4,0])
sorted_head = solution.insertionSortList(head)
print_linked_list(sorted_head)  # Output: -1 -> 0 -> 3 -> 4 -> 5 -> None

```

### Explanation

1. **Node Definition**: First, a ListNode class is defined with typical attributes for value and next pointer.

2. **Solution Class and Method**: `insertionSortList` method in `Solution` class takes the head of a singly linked list as its argument.

3. **Edge Case Handling**: If the list is empty (`head is None`) or consists of a single node (`head.next is None`), it returns the head directly since no sorting is needed.

4. **Initialization**: A dummy node is created to simplify insertion when handling the smallest elements or initializing the list. `sorted_tail` helps in keeping track of where to insert the new elements efficiently from the original list.

5. **Insertion Logic**:
   - Iterates through each node (`current`) in the original list.
   - If the current node's value is lower than any part of the newly forming sorted list (detected by comparing with `sorted_tail`), then move `sorted_tail` to start at dummy to search from the beginning.
   - Use `sorted_tail` to find where in the sorted sublist the current node should be inserted.
   - Adjust the pointers such that `current` is appropriately inserted between `sorted_tail` and `sorted_tail.next`.

6. **Final**: The sorted list's head is returned as `dummy.next`, effectively ignoring the dummy node itself.

7. **Utility Functions**: `create_linked_list` and `print_linked_list` are used to simplify list creation and visualization for test cases. 

This solution effectively sorts a linked list in O(n^2) time (due to the nature of insertion sort), which is acceptable given LeetCode's constraints, but it uses O(1) additional space (excluding the input).

# 148. Sort List

### Problem Description 
Given the `head` of a linked list, return the list after sorting it in ascending order.

Follow up: Can you sort the linked list in `O(n logn)` time and `O(1)` memory (i.e. constant space)?

Example 1:
Input: head = [4,2,1,3]
Output: [1,2,3,4]

Example 2:
Input: head = [-1,5,3,4,0]
Output: [-1,0,3,4,5]

Example 3:
Input: head = []
Output: []

Constraints:
The number of nodes in the list is in the range `[0, 5 * 104]`.

`-105 <= Node.val <= 105`

### Solution 
 To solve the problem of sorting a linked list in ascending order, we'll employ the merge sort algorithm. Merge sort is ideal because it guarantees \(O(n \log n)\) time complexity and can be implemented in \(O(1)\) space complexity when sorting linked lists.

Using merge sort entails:
1. Splitting the linked list into two halves.
2. Recursively sorting the two halves.
3. Merging the two sorted halves.

We'll implement this in a way that can be run directly on the LeetCode platform by providing a `ListNode` class definition as seen in typical LeetCode challenges.

Below is the complete solution including explanations within the code comments:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        # Base condition checks
        if not head or not head.next:
            return head
        
        # Function to get the mid-point of the list
        def getMid(head):
            slow, fast = head, head.next
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
            return slow
        
        # Recursive merge sort function
        def mergeSort(head):
            if not head or not head.next:
                return head
            
            mid = getMid(head)
            right_start = mid.next
            mid.next = None
            
            left_sorted = mergeSort(head)
            right_sorted = mergeSort(right_start)
            
            return merge(left_sorted, right_sorted)
        
        # Function to merge two sorted lists
        def merge(left, right):
            dummy = ListNode(0)
            tail = dummy
            while left and right:
                if left.val < right.val:
                    tail.next = left
                    left = left.next
                else:
                    tail.next = right
                    right = right.next
                tail = tail.next
            tail.next = left if left else right
            return dummy.next
        
        return mergeSort(head)

```

This solution works directly with the node pointers of the linked list, avoiding any additional space apart from function call stack during recursion (which is implicit and doesn't count against the \(O(1)\) space requirement for this problem).

The given code defines:
- A `ListNode` class for individual nodes in the linked list.
- A `Solution` class with a method `sortList` that implements the sorting via the discussed merge sort algorithm.

This approach is optimal and meets the \(O(n \log n)\) time complexity and \(O(1)\) space complexity requirements.

# 160. Intersection of Two Linked Lists

### Problem Description 
Given the heads of two singly linked-lists `headA` and `headB`, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return `null`.

For example, the following two linked lists begin to intersect at node `c1`:
It is guaranteed that there are no cycles anywhere in the entire linked structure.

Note that the linked lists must retain their original structure after the function returns.


Example 1:
Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
Output: Intersected at '8'
Explanation: The intersected node's value is 8 (note that this must not be 0 if the two lists intersect).

From the head of A, it reads as [4,1,8,4,5]. From the head of B, it reads as [5,6,1,8,4,5]. There are 2 nodes before the intersected node in A; There are 3 nodes before the intersected node in B.


Example 2:
Input: intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
Output: Intersected at '2'
Explanation: The intersected node's value is 2 (note that this must not be 0 if the two lists intersect).

From the head of A, it reads as [1,9,1,2,4]. From the head of B, it reads as [3,2,4]. There are 3 nodes before the intersected node in A; There are 1 node before the intersected node in B.


Example 3:
Input: intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
Output: No intersection
Explanation: From the head of A, it reads as [2,6,4]. From the head of B, it reads as [1,5]. Since the two lists do not intersect, intersectVal must be 0, while skipA and skipB can be arbitrary values.

Explanation: The two lists do not intersect, so return null.


Constraints:
The number of nodes of `listA` is in the `m`.

The number of nodes of `listB` is in the `n`.

`0 <= m, n <= 3 * 104`
`1 <= Node.val <= 105`
`0 <= skipA <= m`
`0 <= skipB <= n`
`intersectVal` is `0` if `listA` and `listB` do not intersect.

`intersectVal == listA[skipA + 1] == listB[skipB + 1]` if `listA` and `listB` intersect.

Follow up: Could you write a solution that runs in `O(n)` time and use only `O(1)` memory?

### Solution 
 To address the problem of finding the intersection of two singly linked lists efficiently, we can employ a two-pointer technique. This technique ensures a time complexity of O(n) and uses only O(1) extra space, satisfying the constraints and follow-up requirements.

**Explanation of the Algorithm:**

1. **Two-Pointer Technique:** Utilize two pointers, `pA` and `pB`, initialized at the heads of `headA` and `headB`, respectively. Traverse through the lists, and when a pointer reaches the end of one list, redirect it to the head of the other list. If the lists intersect, `pA` and `pB` will eventually meet at the intersection node due to their simultaneous traversal of both lists. If the lists do not intersect, redirecting the pointers will make them equal at the end of their combined traversal, which is `None`.

2. **Reasoning:** By redirecting the pointers, we ensure that both pointers traverse exactly `m + n` nodes combined, where `m` is the length of `listA` and `n` is the length of `listB`. This balancing allows the pointers from different starting points and lengths to synchronize in terms of their remaining nodes when they converge or prove non-intersection.

Here's how this solution might be written in Python:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:
            return None
        
        # Initialize two pointers
        pA, pB = headA, headB
        
        # While the pointers don't meet
        while pA is not pB:
            # Switch to the other list's head when reaching the end
            pA = pA.next if pA else headB
            pB = pB.next if pB else headA
        
        # If they meet at None, there's no intersection, else they meet at the intersection node
        return pA

# Example of instantiating the linked list and finding intersection:
# Normally, you would need to construct an example like in the problem statement, 
# where part of one list is physically linked to the other and then apply getIntersectionNode.

```

This code defines the `ListNode` class as a structure for the linked list node and implements the `Solution` class with the method `getIntersectionNode()`. The provided method uses the two-pointer approach, which efficiently identifies the intersection, if any, between the two lists.

**Testing:**  
In the LeetCode environment, nodes would be constructed and linked together in a specific pattern to represent the problem constraints. The solution provided above accommodates these setup variations inherently. Note that for direct execution, one might want to set up specific linked list structures that physically share nodes to fully demonstrate intersection testing.

# 203. Remove Linked List Elements

### Problem Description 
Given the `head` of a linked list and an integer `val`, remove all the nodes of the linked list that has `Node.val == val`, and return the new head.


Example 1:
Input: head = [1,2,6,3,4,5,6], val = 6
Output: [1,2,3,4,5]

Example 2:
Input: head = [], val = 1
Output: []

Example 3:
Input: head = [7,7,7,7], val = 7
Output: []

Constraints:
The number of nodes in the list is in the range `[0, 104]`.

`1 <= Node.val <= 50`
`0 <= k <= 50`

### Solution 
 Below is a Python solution for the given problem that adheres to the LeetCode format. It includes the class definition for the linked list node and the function to remove nodes from the linked list based on a specified value.



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        # Initialize a dummy node. This simplifies code when the first node(s)
        # need to be removed or the list is empty.
        dummy = ListNode(next=head)
        
        # Use two pointers: current and previous.
        prev = dummy
        current = head
        
        # Iterate through the linked list.
        while current:
            if current.val == val:
                # If current node needs to be removed,
                # adjust pointers to skip it.
                prev.next = current.next
            else:
                # Otherwise, just move previous pointer.
                prev = current
            # Move to the next node.
            current = current.next
        
        # The head of the new list is the next of dummy node.
        return dummy.next

# Helper function to convert list to linked list.
def list_to_linked_list(elements):
    head = current = ListNode()
    for element in elements:
        current.next = ListNode(element)
        current = current.next
    return head.next

# Helper function to print linked list.
def print_linked_list(head):
    while head:
        print(head.val, end=" -> ")
        head = head.next
    print("None")

# Example usage of the code:
sol = Solution()
# Example 1
lst = [1,2,6,3,4,5,6]
val = 6
head = list_to_linked_list(lst)
new_head = sol.removeElements(head, val)
print_linked_list(new_head)  # Expected Output: 1 -> 2 -> 3 -> 4 -> 5 -> None

# Example 2
lst = []
val = 1
head = list_to_linked_list(lst)
new_head = sol.removeElements(head, val)
print_linked_list(new_head)  # Expected Output: None

# Example 3
lst = [7, 7, 7, 7]
val = 7
head = list_to_linked_list(lst)
new_head = sol.removeElements(head, val)
print_linked_list(new_head)  # Expected Output: None

```

### Explanation
1. **Initialization**: We create a dummy node that points to the head of the list. This allows us to easily handle cases where the first node (or multiple initial nodes) need to be removed.
2. **Iteration**: We iterate through the linked list using a `current` pointer for traversal and a `prev` pointer to make connection adjustments when nodes are skipped.
3. **Node Skipping**: If a node's value matches the target `val`, we need to remove it by adjusting the `prev.next` pointer to skip the current node.
4. **Completion**: Once all nodes have been examined, the modified list will start from `dummy.next`, since `dummy` itself is not part of the original list.

The `list_to_linked_list` and `print_linked_list` functions are helper functions used in the example to easily build and display linked lists.

# 206. Reverse Linked List

### Problem Description 
Given the `head` of a singly linked list, reverse the list, and return the reversed list.


Example 1:
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Example 2:
Input: head = [1,2]
Output: [2,1]

Example 3:
Input: head = []
Output: []

Constraints:
The number of nodes in the list is the range `[0, 5000]`.

`-5000 <= Node.val <= 5000`
Follow up: A linked list can be reversed either iteratively or recursively. Could you implement both?

### Solution 
 Below is the Python code, ready to be tested on LeetCode, to reverse a singly linked list both iteratively and recursively. First, let’s define the ListNode class as per the usual structure used on LeetCode. Then, two solutions are provided: one iterative and one recursive.

### Python Code:


```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        # Iterative Solution
        prev = None
        curr = head
        while curr:
            next_temp = curr.next  # Store reference to the next node
            curr.next = prev       # Reverse the current node's pointer
            prev = curr            # Move pointers one position ahead
            curr = next_temp
        return prev
    
    def reverseListRecursive(self, head: ListNode) -> ListNode:
        # Recursive Solution
        if head is None or head.next is None:
            return head
        p = self.reverseListRecursive(head.next)
        head.next.next = head
        head.next = None
        return p

```

### Explanation:

#### Iterative Method:
1. **Initialization**: Start with `prev` as `None`. This will eventually be the new head of the reversed linked list. Use `curr` to traverse the list.
2. **Reversing Pointers**: Iterate through the list. For each node, temporarily save the next node (`next_temp`), then reverse the `curr.next` pointer to point to `prev`. Move `prev` and `curr` forward one step.
3. **Return**: When the loop exits (i.e., `curr` is `None`), `prev` will be at the new head of the reversed list, which we return.

#### Recursive Method:
1. **Base Case**: If the list is empty (`head` is `None`) or has only one element (`head.next` is `None`), return `head` as no further reversing is needed.
2. **Recursive Call**: Call the function recursively with `head.next` to handle the rest of the list.
3. **Reversing Pointers**: Once reaching the base case, the stack starts unwinding. Set the next node's (`head.next.next`) next pointer to `head`, and then set `head.next` to `None` to complete the reversal of the current node.
4. **Return**: Return the new head of the reversed list (returned by deeper calls) back up the call stack.

### Usage:
To utilize this code in a real scenario (e.g., on LeetCode), input is typically managed by LeetCode itself. This structure can directly handle input given in terms of linked list nodes and will output the linked list in reverse order.

# 234. Palindrome Linked List

### Problem Description 
Given the `head` of a singly linked list, return `true` if it is a palindrome.


Example 1:
Input: head = [1,2,2,1]
Output: true

Example 2:
Input: head = [1,2]
Output: false

Constraints:
The number of nodes in the list is in the range `[1, 105]`.

`0 <= Node.val <= 9`
Follow up: Could you do it in `O(n)` time and `O(1)` space?

### Solution 
 Below is the Python code to determine whether a singly linked list is a palindrome. The code must be executed on the LeetCode platform within a Python programming environment. The provided solution meets the requirement of O(n) time complexity and strives to achieve O(1) extra space complexity, which follows the given follow-up question. The strategy mainly involves finding the middle of the list, reversing the second half, and then comparing the first half with the reversed second half.



```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        if not head or not head.next:
            return True
        
        # Find the middle of the linked list
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # Reverse the second half of the linked list
        prev = None
        while slow:
            next_node = slow.next
            slow.next = prev
            prev = slow
            slow = next_node
        
        # Check if the linked list is a palindrome
        left, right = head, prev
        while right:  # We only need to check half of the list
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        
        return True

# The above class definition and method can be tested using LeetCode's online platform by submitting the code.

```

**Explanation:**
1. **Edge Case Handling:** If the list is empty or contains a single node, it's inherently a palindrome.
2. **Middle Finding:** Use the two-pointer technique where `slow` moves one step and `fast` moves two steps. When `fast` reaches the end, `slow` will be at the middle.
3. **Reverse Second Half:** After finding the middle, reverse the latter half of the list starting from `slow`.
4. **Palindrome Comparison:** Traverse from the start and from the middle (using two pointers again) to check if the values are the same till the end of the right half.
5. **Time Complexity:** The operations for finding the middle, reversing, and then comparing are all linear, leading to an O(n) overall time complexity. 
6. **Space Complexity:** Since only a few extra variables are used, and transformations are done in-place, the space complexity is O(1), as required.

This solution respects all constraints and adheres to the performance requirements stated in the problem's follow-up question.

# 237. Delete Node in a Linked List

### Problem Description 
Write a function to delete a node in a singly-linked list. You will not be given access to the `head` of the list, instead you will be given access to the node to be deleted directly.

It is guaranteed that the node to be deleted is not a tail node in the list.


Example 1:
Input: head = [4,5,1,9], node = 5
Output: [4,1,9]
Explanation: You are given the second node with value 5, the linked list should become 4 -> 1 -> 9 after calling your function.


Example 2:
Input: head = [4,5,1,9], node = 1
Output: [4,5,9]
Explanation: You are given the third node with value 1, the linked list should become 4 -> 5 -> 9 after calling your function.


Example 3:
Input: head = [1,2,3,4], node = 3
Output: [1,2,4]

Example 4:
Input: head = [0,1], node = 0
Output: [1]

Example 5:
Input: head = [-3,5,-99], node = -3
Output: [5,-99]

Constraints:
The number of the nodes in the given list is in the range `[2, 1000]`.

`-1000 <= Node.val <= 1000`
The value of each node in the list is unique.

The `node` to be deleted is in the list and is not a tail node

### Solution 
 When providing a solution for deleting a node from a singly-linked list directly in LeetCode, consider the specific requirements and constraints. Here is the thought process as well as the Python code that can be used directly in LeetCode:

## Thought Process
In traditional deletion, we often have access to the head of the linked list and we traverse it to find and remove a given node. However, in this problem, we only have direct access to the node that needs to be deleted, but not to the head of the list itself. The challenge is that we can't easily modify the links of the previous node as we have no way of reaching it.

Given that we know the node to be deleted is guaranteed not to be the tail node, we can use the following trick:
1. Copy the value from the next node into the node to be deleted.
2. Delete the next node by changing the next pointer of the node to be deleted to point to the next node's next pointer.



```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def deleteNode(node):
    """
    :type node: ListNode
    :rtype: void Do not return anything, modify node in-place instead.
    """
    # Copy the value of the next node into the current node
    node.val = node.next.val
    # Skip the next node
    node.next = node.next.next

```

### Code Walkthrough
- The function `deleteNode` accepts a parameter `node`, which is the node to be deleted.
- It modifies the value of `node` to the value of `node.next` which effectively overwrites the current node's value with the next node's value.
- Then, it adjusts the `next` reference of the `node` to skip over `node.next`, hence linking directly to `node.next.next`. This removal effectively deletes `node.next` from the list, though we have moved its value one step earlier in the list.

This approach works under the constraints given because:
- The node to delete is ensured to not be a tail node, hence `node.next` is always valid.
- The space complexity is O(1) since no additional space is required.
- The time complexity is also O(1) because we are performing a constant number of operations.

### Examples
Using the function on a linked list constructed to represent the test cases would reflect the given examples, such as transforming [4,5,1,9] into [4,1,9] when the node with value 5 is passed to `deleteNode`.

This solution can be tested directly in LeetCode under a problem designed as described with ListNode definition and function signature provided like above.

# 328. Odd Even Linked List

### Problem Description 
Given the `head` of a singly linked list, group all the nodes with odd indices together followed by the nodes with even indices, and return the reordered list.

The first node is considered odd, and the second node is even, and so on.

Note that the relative order inside both the even and odd groups should remain as it was in the input.


Example 1:
Input: head = [1,2,3,4,5]
Output: [1,3,5,2,4]

Example 2:
Input: head = [2,1,3,5,6,4,7]
Output: [2,3,6,7,1,5,4]

Constraints:
The number of nodes in the linked list is in the range `[0, 104]`.

`-106 <= Node.val <= 106`
Follow up: Could you solve it in `O(1)` space complexity and `O(nodes)` time complexity?

### Solution 
 You can solve this problem by iterating through the linked list and constructing two separate lists: one for the nodes at odd positions and one for the nodes at even positions. Finally, you'll combine these two lists.

In Python, the solution can be formatted in the LeetCode's required class structure as follows:



```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#        self.val = val
#        self.next = next

class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        # Initialize pointers for odd and even
        odd = head
        even = head.next
        even_head = even

        # Process all the nodes in the list
        while even and even.next:
            # Jump the odd node's next to the subsequent odd node 
            odd.next = odd.next.next
            odd = odd.next
            
            # Similarly, jump the even node's next to the subsequent even node
            even.next = even.next.next
            even = even.next

        # Connect the end of the odd-linked nodes to the head of the even-linked nodes
        odd.next = even_head

        return head

```

### Explanation:

- This code snippet begins by defining a LinkedList node class, which we typically find prepared in coding interview environments like LeetCode. It represents elements of the list with `val` and `next` pointers.
- The `oddEvenList` function checks if the list is empty or consists of a single node, in which case it directly returns the `head`.
- Two pointers, `odd` and `even`, are initialized to start at the first and second nodes of the list, respectively.
- Another pointer, `even_head`, is used to remember the start of the even-index nodes.
- A loop then runs through the list as long as there are nodes to process in the even list (`even` and `even.next`). The pointers skip one node during each iteration, effectively grouping odd nodes together and even nodes together.
- Finally, `odd.next` is set to `even_head` to connect the reordered groups.
- The function returns the modified list starting from `head`.

This solution meets the requirements of being `O(n)` in time complexity — where `n` is the number of nodes in the list — since it makes a single pass through the list. It's also `O(1)` in space complexity since no additional data structures are used for storing nodes, but just a few extra pointers.

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
 To solve this problem, the key is to add one to the last digit of the number represented by a linked list and properly handle any carry that might result. The main challenge is that the most significant digit is stored at the head, and digits follow in order, which means we need to traverse to the end of the linked list to start adjusting digits or handle carries.

For this solution, we'll first reverse the linked list for easier manipulation (since operations like adding can then start from the least significant digit). After processing the addition and any carries, we'll reverse the list again to restore the original order before returning the result. Alternatively, we can use a recursive approach to handle the addition without reversing the list, but we'll go with the iterative reversal for clarity and ease.

Here's the detailed Python code using the recommended LeetCode format:



```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def plusOne(self, head: ListNode) -> ListNode:
        # Helper function to reverse a linked list
        def reverse(head):
            prev = None
            current = head
            while current:
                next_node = current.next
                current.next = prev
                prev = current
                current = next_node
            return prev
        
        # Step 1: Reverse the linked list
        reversed_head = reverse(head)
        
        # Step 2: Add one to the number
        current = reversed_head
        carry = 1
        while current and carry:
            current_val = current.val + carry
            current.val = current_val % 10
            carry = current_val // 10
            if current.next is None and carry != 0:
                # Handle the case where we have a carry at the last digit
                current.next = ListNode(carry)
                carry = 0
            current = current.next
        
        # Step 3: Reverse the linked list back to its original order
        final_head = reverse(reversed_head)
        
        return final_head

```

### Explanation:
1. **Reversing the Linked List**: This function takes a linked list and reverses it, so the least significant digit moves to the front. This is done using a simple while loop by adjusting the `next` pointers.

2. **Adding One**: We start from the reversed list's head and add one (initialize `carry = 1`). Each node's value is updated in place, and we propagate the carry forward. If the final node has a carry, a new node is created.

3. **Restoring Order**: Finally, the list is reversed back to its original order (most significant digit at the head) and returned.

This solution runs in O(n) time due to the need to potentially traverse the list multiple times (up to two reversals and one pass for addition), and uses O(1) additional space not counting the input and output space. This meets the problem's constraints and handles all edge cases, such as handling a carry that extends the list length (`[9, 9, 9] -> [1, 0, 0, 0]`).

# 379. Design Phone Directory

### Problem Description 
Design a phone directory that initially has `maxNumbers` empty slots that can store numbers. The directory should store numbers, check if a certain slot is empty or not, and empty a given slot.

Implement the `PhoneDirectory` class:
`PhoneDirectory(int maxNumbers)` Initializes the phone directory with the number of available slots `maxNumbers`.

`int get()` Provides a number that is not assigned to anyone. Returns `-1` if no number is available.

`bool check(int number)` Returns `true` if the slot `number` is available and `false` otherwise.

`void release(int number)` Recycles or releases the slot `number`.


Example 1:
Input
["PhoneDirectory", "get", "get", "check", "get", "check", "release", "check"]
[[3], [], [], [2], [], [2], [2], [2]]
Output
[null, 0, 1, true, 2, false, null, true]
Explanation
PhoneDirectory phoneDirectory = new PhoneDirectory(3);
phoneDirectory.get();      // It can return any available phone number. Here we assume it returns 0.

phoneDirectory.get();      // Assume it returns 1.

phoneDirectory.check(2);   // The number 2 is available, so return true.

phoneDirectory.get();      // It returns 2, the only number that is left.

phoneDirectory.check(2);   // The number 2 is no longer available, so return false.

phoneDirectory.release(2); // Release number 2 back to the pool.

phoneDirectory.check(2);   // Number 2 is available again, return true.


Constraints:
`1 <= maxNumbers <= 104`
`0 <= number < maxNumbers`
At most `2 * 104` calls will be made to `get`, `check`, and `release`.

### Solution 
 To solve this problem, the `PhoneDirectory` class should efficiently manage available and unavailable numbers using a set (for quick lookup) and a queue (for providing numbers in an organized manner). Below is the implementation along with detailed comments.

Here is how you can implement the `PhoneDirectory` class:



```python
class PhoneDirectory:
    def __init__(self, maxNumbers: int):
        # Initialize phone directory.
        self.available = set(range(maxNumbers))
        self.queue = list(range(maxNumbers))

    def get(self) -> int:
        # Provide an available number.
        if self.queue:
            number = self.queue.pop(0)
            self.available.remove(number)
            return number
        else:
            return -1  # Return -1 if no number is available.

    def check(self, number: int) -> bool:
        # Check if the number is available.
        return number in self.available

    def release(self, number: int) -> None:
        # Release the number back to the pool if it is not currently available.
        if number not in self.available:
            self.available.add(number)
            self.queue.append(number)

# Example usage
# Your PhoneDirectory object will be instantiated and called as such:
# obj = PhoneDirectory(maxNumbers)
# param_1 = obj.get()
# param_2 = obj.check(number)
# obj.release(number)

```

### Explanation:

- **__init__ method**: Initializes the PhoneDirectory. It uses:
  - **self.available**: A set containing all available numbers, useful for O(1) lookups with `check`.
  - **self.queue**: A list holding all available numbers, to provide the next available number with `get()` method.

- **get method**: Dequeues a number from the queue (thus giving an available number) and removes it from the set of available numbers. If the queue is empty, it returns -1 indicating no numbers are available.

- **check method**: Simply checks if a number is in the `available` set, providing an instantaneous response regarding its availability.

- **release method**: Adds the number back to the available set and appends it to the queue if it's not already present there.

### Complexity Analysis:
- **Time Complexity**: The operations `get`, `check`, and `release` are O(1) due to the use of set for checking elements and lists for managing available numbers.
- **Space Complexity**: O(maxNumbers) due to storage used for `available` set and `queue`.

This solution adheres to the constraints and efficiently handles up to the maximum number of allowable calls as specified.

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
 

```python
class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if not root:
            return None

        # Initializing the first and last nodes as None
        first, last = None, None
        
        # Helper function to perform inorder traversal
        def inorder(node):
            nonlocal first, last
            if not node:
                return
            
            # Visit the left child
            inorder(node.left)
            
            # Process the current node
            if last:
                # Link the last node with the current node
                last.right = node
                node.left = last
            else:
                # Initialize the first node when `last` is None
                first = node
            
            # Update the last node to the current node
            last = node
            
            # Visit the right child
            inorder(node.right)

        # Execute the inorder traversal starting from the root
        inorder(root)
        
        # Linking the last element with the first to make the list circular
        last.right = first
        first.left = last
        
        return first

```

**Explanation:**
1. **Node Class:** Defined a tree 'Node' with `val`, `left`, and `right` to represent the BST.

2. **Solution Class:** Implements function `treeToDoublyList` to convert the BST to a circular doubly linked list.

3. **Helper Function (inorder):** Performs an inorder traversal which processes the tree in sorted order.
   - If there is a left child, it recursively visits the left subtree.
   - Processes the current node:
     - If a predecessor (`last`) exists, sets up the current node as the successor of `last` and set `last.right` pointer to the current node and `node.left` to `last`.
     - If `last` does not exist, it means the current node is the smallest node, and so assigns `first` to this node.
   - Updates the `last` node to the current node.
   - If there is a right child, it recursively visits the right subtree.

4. **Circular Linking:** After the traversal, `first` points to the smallest element and `last` points to the largest. These are linked to each other to form a circular doubly linked list.

5. **Return Value:** Returns `first` pointer, which is the smallest element of the list, as the entry point to the doubly linked list.

This solution works in-place to transform the BST into a sorted circular doubly-linked list using an in-order traversal, following constraints up to 2000 nodes while keeping track of first and last nodes to create circular links effectively. The methodology ensures that no additional space, other than the needed pointers, is utilized, hence the space complexity excluding the input is O(1).

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
 To implement a solution for the problem outlined, we will use a depth-first search (DFS) approach to flatten the multilevel doubly linked list. This will involve using a stack to manage nodes, ensuring that we traverse the child nodes before continuing with the next nodes in the list.

Here is a structured approach to solve the problem:
1. Start with the head of the linked list.
2. Use a stack to keep track of nodes to visit.
3. Start with the head node and push it to the stack.
4. Use a loop to process each node popped from the stack:
   - Attach the last visited node to the current node via next pointers, and fix the previous pointers accordingly.
   - If the current node has a child, push the next node (if it exists) to the stack, then set the current node's next to its child. The end of the child chain is then linked back to the next node (which is on top of the stack).
   - Continue this process until the stack is exhausted.
5. Ensure all child pointers are set to null since all node relations will now be expressed via next pointers only.

Here's the code implementing the above steps:



```python
class Node:
    def __init__(self, val=0, prev=None, next=None, child=None):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child

def flatten(head: 'Node') -> 'Node':
    if not head:
        return head
    
    pseudoHead = Node(0, None, head, None)
    prev = pseudoHead
    
    stack = [head]
    
    while stack:
        curr = stack.pop()
        
        prev.next = curr
        curr.prev = prev
        
        if curr.next:
            stack.append(curr.next)
            curr.next = None  # This is crucial to prevent cycles
        
        if curr.child:
            stack.append(curr.child)
            curr.child = None  # Prevent looping through child again
        
        prev = curr
    
    pseudoHead.next.prev = None  # Detach pseudo head from the real head
    return pseudoHead.next

# Example usage (this will need a setup to create a multilevel doubly linked list as per input)
# head = Node(1)
# head.next = Node(2)
# head.next.prev = head
# head.next.next = Node(3)
# head.next.next.prev = head.next
# head.child = Node(7)
# Flatten and test your linked list here.

```

This approach efficiently flattens the linked list in-place with O(N) time complexity, where N is the total number of nodes in the list, and O(N) space complexity from use of the stack, ensuring it handles deep nesting of child nodes efficiently.

# 445. Add Two Numbers II

### Problem Description 
You are given two non-empty linked lists representing two non-negative integers. The most significant digit comes first and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Follow up:
What if you cannot modify the input lists? In other words, reversing the lists is not allowed.


Example:
Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 8 -> 0 -> 7

### Solution 
 To solve this problem using Python for LeetCode, you'll need to create a linked list to represent the non-negative integers. The digits of these integers are stored in reverse order, such that the least significant digit is at the head of the list. Since we are not allowed to modify the input lists for the follow-up, the approach would involve using a stack to reverse the order of traversal without physically reversing the linked lists.

Here's how we can approach this:

1. **Use stacks to collect digits:** As we traverse the two linked lists from head to tail, we'll push the digits onto the stacks. This will help us to retrieve the digits from the least significant to the most significant by using the stack's Last-In-First-Out (LIFO) nature.

2. **Perform digit-wise addition with carry handling:** Pop digits from each stack one by one, adding them together while managing any carry that arises from the addition.

3. **Create the resultant linked list:** As we continue adding the numbers, we will create new nodes of a resultant linked list from the most significant to the least significant digit (as a result, we will be creating the linked list from tail to head).

4. **Handle any remaining carry:** After the main addition phase is over, if there's any carry left, it should be added as a new leading node.

Here’s how we can implement this using Python:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        stack1, stack2 = [], []
        
        # Fill the stacks
        while l1:
            stack1.append(l1.val)
            l1 = l1.next
        while l2:
            stack2.append(l2.val)
            l2 = l2.next
        
        carry = 0
        head = None
        
        # Process both stacks
        while stack1 or stack2 or carry:
            sum = carry
            
            if stack1:
                sum += stack1.pop()
            if stack2:
                sum += stack2.pop()
            
            carry = sum // 10
            node = ListNode(sum % 10)
            node.next = head
            head = node
        
        return head

```

This function is ready to be run on LeetCode to solve the problem.

- A `ListNode` class is defined to use as the nodes of the linked list.
- Function `addTwoNumbers` takes two linked lists (`l1` and `l2`) and performs digit-wise addition from the least significant digit to the most significant by utilizing stacks.
- We create a new list by continually updating the `head`, which ensures that the resultant list stores values from the least significant to the most significant digit just as required.
 
The time complexity of this solution is O(m + n), where `m` and `n` are the lengths of the two linked lists. Space complexity is O(m + n) as well because of the space required for the stacks and the new list.

# 707. Design Linked List

### Problem Description 
Design your implementation of the linked list. You can choose to use a singly or doubly linked list.

A node in a singly linked list should have two attributes: `val` and `next`. `val` is the value of the current node, and `next` is a pointer/reference to the next node.

If you want to use the doubly linked list, you will need one more attribute `prev` to indicate the previous node in the linked list. Assume all nodes in the linked list are 0-indexed.

Implement the `MyLinkedList` class:
`MyLinkedList()` Initializes the `MyLinkedList` object.

`int get(int index)` Get the value of the `indexth` node in the linked list. If the index is invalid, return `-1`.

`void addAtHead(int val)` Add a node of value `val` before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.

`void addAtTail(int val)` Append a node of value `val` as the last element of the linked list.

`void addAtIndex(int index, int val)` Add a node of value `val` before the `indexth` node in the linked list. If `index` equals the length of the linked list, the node will be appended to the end of the linked list. If `index` is greater than the length, the node will not be inserted.

`void deleteAtIndex(int index)` Delete the `indexth` node in the linked list, if the index is valid.


Example 1:
Input
["MyLinkedList", "addAtHead", "addAtTail", "addAtIndex", "get", "deleteAtIndex", "get"]
[[], [1], [3], [1, 2], [1], [1], [1]]
Output
[null, null, null, null, 2, null, 3]
Explanation
MyLinkedList myLinkedList = new MyLinkedList();
myLinkedList.addAtHead(1);
myLinkedList.addAtTail(3);
myLinkedList.addAtIndex(1, 2);    // linked list becomes 1->2->3
myLinkedList.get(1);              // return 2
myLinkedList.deleteAtIndex(1);    // now the linked list is 1->3
myLinkedList.get(1);              // return 3

Constraints:
`0 <= index, val <= 1000`
Please do not use the built-in LinkedList library.

At most `2000` calls will be made to `get`, `addAtHead`, `addAtTail`, `addAtIndex` and `deleteAtIndex`.

### Solution 
 Here's a Python solution to the described problem using a singly linked list along with detailed explanations embedded in the comments. This code can be directly used in LeetCode's environment:



```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class MyLinkedList:
    def __init__(self):
        self.size = 0
        self.head = ListNode(0)  # Sentinel node as pseudo-head

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        
        current = self.head
        # get the index-th node (we're starting with pseudo-head node `self.head`)
        for _ in range(index + 1):
            current = current.next
        return current.val

    def addAtHead(self, val: int) -> None:
        self.addAtIndex(0, val)

    def addAtTail(self, val: int) -> None:
        self.addAtIndex(self.size, val)

    def addAtIndex(self, index: int, val: int) -> None:
        if index > self.size:
            return
        
        if index < 0:
            index = 0
        
        self.size += 1
        # find predecessor of the node to be added
        pred = self.head
        for _ in range(index):
            pred = pred.next
        
        # node to be added
        to_add = ListNode(val)
        # insertion itself
        to_add.next = pred.next
        pred.next = to_add

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return
        
        self.size -= 1
        # find predecessor of the node to be deleted
        pred = self.head
        for _ in range(index):
            pred = pred.next
        
        # delete pred.next 
        pred.next = pred.next.next

# Example/Test usage:
# Using the sample operations and expected output described in the problem statement
# myLinkedList = MyLinkedList()
# myLinkedList.addAtHead(1)
# myLinkedList.addAtTail(3)
# myLinkedList.addAtIndex(1, 2)
# print(myLinkedList.get(1))  # Output is 2
# myLinkedList.deleteAtIndex(1)
# print(myLinkedList.get(1))  # Output is 3

```

### Explanation:

1. `ListNode` is an inner class representing a node in the linked list, having a value and a reference to the next node.

2. The `MyLinkedList` class maintains a pseudo-head (sentinel node) for simplicity to handle edge cases with adding or removing nodes from the start of the linked list.

3. The methods `addAtHead`, `addAtTail`, and `addAtIndex` all use the `addAtIndex` method to simplify inserting nodes by finding the correct preceding node and modifying pointers accordingly.

4. The `deleteAtIndex` method ensures that the pointer of the previous node skips the node to be deleted, thus removing it from the list. Proper bounds checking is performed to ensure index validity.

The use of a pseudo-head node simplifies operations, especially when adding or removing from the head, as it reduces the number of conditional checks required.

# 708. Insert into a Sorted Circular Linked List

### Problem Description 
Given a node from a Circular Linked List which is sorted in ascending order, write a function to insert a value `insertVal` into the list such that it remains a sorted circular list. The given node can be a reference to any single node in the list, and may not be necessarily the smallest value in the circular list.

If there are multiple suitable places for insertion, you may choose any place to insert the new value. After the insertion, the circular list should remain sorted.

If the list is empty (i.e., given node is `null`), you should create a new single circular list and return the reference to that single node. Otherwise, you should return the original given node.


Example 1:
Input: head = [3,4,1], insertVal = 2
Output: [3,4,1,2]
Explanation: In the figure above, there is a sorted circular list of three elements. You are given a reference to the node with value 3, and we need to insert 2 into the list. The new node should be inserted between node 1 and node 3. After the insertion, the list should look like this, and we should still return node 3.


Example 2:
Input: head = [], insertVal = 1
Output: [1]
Explanation: The list is empty (given head is `null`). We create a new single circular list and return the reference to that single node.


Example 3:
Input: head = [1], insertVal = 0
Output: [1,0]

Constraints:
`0 <= Number of Nodes <= 5 * 10^4`
`-10^6 <= Node.val <= 10^6`
`-10^6 <= insertVal <= 10^6`

### Solution 
 Here's a Python solution for the described problem along with detailed comments to ensure it's clear and follows the constraints given:



```python
class Node:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def insertIntoSortedCircularList(head, insertVal):
    new_node = Node(insertVal)
    
    if not head:
        # If the list is empty, create a new node pointing to itself
        new_node.next = new_node
        return new_node
    
    prev, curr = head, head.next
    insertion_completed = False
    
    while True:
        # Check if we are at the appropriate insertion point
        if prev.val <= insertVal <= curr.val:
            # Case 1: Node fits between two sorted nodes
            prev.next = new_node
            new_node.next = curr
            insertion_completed = True
            break
        elif prev.val > curr.val:
            # Case 2: We are at the boundary between the largest and smallest values
            if insertVal >= prev.val or insertVal <= curr.val:
                # Either insertVal is greater than the max or smaller than the min
                prev.next = new_node
                new_node.next = curr
                insertion_completed = True
                break
        
        # Move pointers
        prev, curr = curr, curr.next
        
        # If we traversed the whole list and didn't find insertion point,
        # which can happen if all elements are the same and do not satisfy
        # any of the above conditions
        if prev == head:
            break
    
    if not insertion_completed:
        # If all nodes are the same and we never inserted,
        # just insert the new node somewhere, e.g., after head
        prev.next = new_node
        new_node.next = curr
    
    return head

# To directly run on LeetCode, the class name and function signature must match
# the expectation and you can use the above Node class and the function as is.

```

### Explanation

This solution provides a complete implementation for inserting a value into a sorted circular linked list, maintaining its sorted order.

1. **Node Class**: A helper class to represent the linked list node which stores the value and a reference to the next node.
2. **Function insertIntoSortedCircularList**: The core logic:
   - Handles an empty list scenario by creating a self-referential single node.
   - Traverses the circular list to find the correct insertion location:
     - It may insert between two nodes if the `insertVal` fits in between their values (sorted order).
     - It may insert at the boundary between max and min values in the sorted circular list.
     - If all values in the list are the same or it completes a full loop without finding a spot, it will insert the new value next to the head.
   - The operation ensures the circular nature of the list remains intact.

The provided solution works for all the given constraints and typical edge cases, including the list being empty or containing a single node.

# 725. Split Linked List in Parts

### Problem Description 
Given a (singly) linked list with head node `root`, write a function to split the linked list into `k` consecutive linked list "parts".

The length of each part should be as equal as possible: no two parts should have a size differing by more than 1.  This may lead to some parts being null.

The parts should be in order of occurrence in the input list, and parts occurring earlier should always have a size greater than or equal parts occurring later.

Return a List of ListNode's representing the linked list parts that are formed.


Examples
1->2->3->4, k = 5 // 5 equal parts
[ [1], 
[2],
[3],
[4],
null ]

Example 1:
Input: 
root = [1, 2, 3], k = 5
Output: [[1],[2],[3],[],[]]
Explanation:
The input and each element of the output are ListNodes, not arrays.

For example, the input root has root.val = 1, root.next.val = 2, \root.next.next.val = 3, and root.next.next.next = null.

The first element output[0] has output[0].val = 1, output[0].next = null.

The last element output[4] is null, but it's string representation as a ListNode is [].


Example 2:
Input: 
root = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], k = 3
Output: [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]
Explanation:
The input has been split into consecutive parts with size difference at most 1, and earlier parts are a larger size than the later parts.

Note:
The length of `root` will be in the range `[0, 1000]`.

Each value of a node in the input will be an integer in the range `[0, 999]`.

`k` will be an integer in the range `[1, 50]`.

### Solution 
 To solve this problem, we can start by evaluating the length of the provided linked list. Using this length, we will determine the size of each part when the list is broken into `k` parts. Given that we can only split the linked list into whole nodes, perfectly equal divisions might not always be possible. However, we should aim for the length of consecutive parts to differ by at most 1 element. Once we've determined the sizes, we convert these sizes into actual linked list parts.

Here's the plan:

1. Traverse the linked list to find its total length.
2. Determine the base size of each part (`minimum size`) and determine how many parts need an `extra node` by computing the remainder when the total length is divided by `k`.
3. Traverse the linked list again, this time breaking it up into parts as determined in the previous step.
4. Return the list of parts.

Let's implement this solution in Python.



```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def splitListToParts(self, root: ListNode, k: int):
        # Step 1: Calculate the length of the linked list
        current = root
        total_length = 0
        while current:
            total_length += 1
            current = current.next
        
        # Step 2: Determine the minimum size of each part and how many need an extra node
        part_length = total_length // k
        remainder = total_length % k
        
        # Step 3: Break the linked list into parts
        parts = []
        current = root
        for i in range(k):
            # Each part has part_length nodes, and the first "remainder" parts get one extra node
            part_head = current
            part_size = part_length + (1 if i < remainder else 0)
            
            # To separate the part from the list, we need to move forward part_size nodes
            for j in range(part_size - 1) if part_size > 0 else []:
                if current:
                    current = current.next
            
            if current:
                # Separate the current part from the rest of the list
                next_part = current.next
                current.next = None
                current = next_part
            
            # Add the current part head to the result
            parts.append(part_head if part_size > 0 else None)
        
        return parts

# Helper function to create a linked list from a list of values
def createLinkedList(lst):
    dummy = ListNode(0)
    current = dummy
    for value in lst:
        current.next = ListNode(value)
        current = current.next
    return dummy.next

# Example usage
if __name__ == "__main__":
    # Example 1
    root = createLinkedList([1, 2, 3])
    k = 5
    solution = Solution()
    parts = solution.splitListToParts(root, k)
    output = [[node.val for node in part] if part else [] for part in parts]
    print(output)  # Output should be [[1], [2], [3], [], []]

    # Example 2
    root = createLinkedList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    k = 3
    parts = solution.splitListToParts(root, k)
    output = [[node.val for node in part] if part else [] for part in parts]
    print(output)  # Output should be [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]

```

In this implementation, we used a helper function `createLinkedList` to facilitate easy testing by converting a list of integers to a linked list. The main solution is in the `splitListToParts` method of the `Solution` class, structured to be directly usable in a LeetCode submission. This method handles cases where the number of requested parts `k` exceeds the length of the linked list gracefully by making use of empty parts represented as `None`.

# 817. Linked List Components

### Problem Description 
We are given `head`, the head node of a linked list containing unique integer values.

We are also given the list `G`, a subset of the values in the linked list.

Return the number of connected components in `G`, where two values are connected if they appear consecutively in the linked list.


Example 1:
Input: 
head: 0->1->2->3
G = [0, 1, 3]
Output: 2
Explanation: 
0 and 1 are connected, so [0, 1] and [3] are the two connected components.


Example 2:
Input: 
head: 0->1->2->3->4
G = [0, 3, 1, 4]
Output: 2
Explanation: 
0 and 1 are connected, 3 and 4 are connected, so [0, 1] and [3, 4] are the two connected components.

Note: 
If `N` is the length of the linked list given by `head`, `1 <= N <= 10000`.

The value of each node in the linked list will be in the range` [0, N - 1]`.

`1 <= G.length <= 10000`.

`G` is a subset of all values in the linked list.

### Solution 
 Below, you'll find the Python code to solve the problem of finding the number of connected components in a linked list, based on the nodes specified in the list `G`. The solution involves iterating through the linked list while maintaining a flag that indicates whether the current segment of visited nodes is considered a connected component of `G`.

We'll start by defining the linked list node class and then proceed with the solution function:



```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def numComponents(self, head: ListNode, G: List[int]) -> int:
        # First, convert the list G into a set for fast lookup
        g_set = set(G)
        current = head
        count = 0
        in_component = False
        
        # Traverse the linked list
        while current:
            # Check if the current node is in G
            if current.val in g_set:
                # Check if we are not already counting a component
                if not in_component:
                    # We are starting a new component
                    count += 1
                    in_component = True
            else:
                # If not in G, reset in_component flag if it was set
                in_component = False
                
            # Move to the next node in the list
            current = current.next
        
        return count

# Helper function to create a linked list from a list of values
def create_linked_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for value in values[1:]:
        current.next = ListNode(value)
        current = current.next
    return head

```

To test the function, you can create a linked list and a list `G`, and then instantiate the `Solution` class to use the `numComponents` method:



```python
# Example usage:
head_values = [0, 1, 2, 3, 4]
head = create_linked_list(head_values)
G = [0, 3, 1, 4]

sol = Solution()
print(sol.numComponents(head, G))  # Output should match the expected result from the examples

```

#### Explanation:
- **Data Structures**: We use a set for `G` for O(1) average-time complexity look-up.
- **Process**: We iterate through the nodes of the linked list. When we find a node's value in `G`, we check whether it starts a new connected component by using the boolean flag `in_component`. If a node is part of `G` but the previous node is not (or does not exist), it signifies the beginning of a new connected component.
- **Variables**:
  - `g_set` to store the values of `G` for quick access.
  - `count` to count the number of connected components.
  - `in_component` is a boolean flag indicating if the current sequence of nodes is counted as one connected component.

This methodology ensures an efficient traversal with a time complexity of O(N), where N is the number of nodes in the linked list.

# 876. Middle of the Linked List

### Problem Description 
Given a non-empty, singly linked list with head node `head`, return a middle node of linked list.

If there are two middle nodes, return the second middle node.


Example 1:
Input: [1,2,3,4,5]
Output: Node 3 from this list (Serialization: [3,4,5])
The returned node has value 3.  (The judge's serialization of this node is [3,4,5]).

Note that we returned a ListNode object ans, such that:
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, and ans.next.next.next = NULL.


Example 2:
Input: [1,2,3,4,5,6]
Output: Node 4 from this list (Serialization: [4,5,6])
Since the list has two middle nodes with values 3 and 4, we return the second one.

Note:
The number of nodes in the given list will be between `1` and `100`.

### Solution 
 

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        # Fast and slow pointers initialization
        slow = fast = head
        
        # Move the fast pointer two steps and the slow pointer one step each time
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # By the time the fast pointer reaches the end of the list,
        # the slow pointer will be at the middle
        return slow

# Example usage:
# Let's construct a linked list manually for demonstration.
# If running on LeetCode, this part isn't needed as LeetCode will prepare the list.

# List corresponding to example 1: [1,2,3,4,5]
node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(3)
node4 = ListNode(4)
node5 = ListNode(5)
node1.next = node2
node2.next = node3
node3.next = node4
node4.next = node5

sol = Solution()
middle = sol.middleNode(node1)

# Print the values from the middle to the end of the list
while middle:
    print(middle.val, end=" ")
    middle = middle.next
# Output: 3 4 5

```

This Python code defines a function, `middleNode`, which takes a singly linked list and returns the middle node according to the problem's requirements. Here's how it works:

1. **ListNode Class**: This is a simple definition of a node in a singly linked list, having a value `val` and a pointer `next` to the next node.

2. **Solution Class with middleNode Method**:
    - We employ the two-pointer technique using `slow` and `fast` pointers both initiated at the head of the list.
    - The `fast` pointer traverses the list at twice the speed (two nodes at a time) compared to the `slow` pointer (one node at a time).
    - The loop continues until `fast` or `fast.next` becomes `None`, indicating the end of the list has been reached or exceeded.
    - At the end of these iterations, `slow` is guaranteed to point at the middle of the list:
        - If the number of elements is odd, `slow` stops exactly at the center.
        - If even, `slow` will stop at the second of the two central nodes.

3. **Comments and Formatting**:
   - Comments are used within the code to make it clear what each major step is responsible for.
   - The format adheres to Python best practices, making it both readable and ready for incorporation into a broader codebase or for direct submissions to competitive coding platforms like LeetCode.

# 1019. Next Greater Node In Linked List

### Problem Description 
We are given a linked list with `head` as the first node.  Let's number the nodes in the list: `node_1, node_2, node_3, ...` etc.

Each node may have a next larger value: for `node_i`, `next_larger(node_i)` is the `node_j.val` such that `j > i`, `node_j.val > node_i.val`, and `j` is the smallest possible choice.  If such a `j` does not exist, the next larger value is `0`.

Return an array of integers `answer`, where `answer[i] = next_larger(node_{i+1})`.

Note that in the example inputs (not outputs) below, arrays such as `[2,1,5]` represent the serialization of a linked list with a head node value of 2, second node value of 1, and third node value of 5.


Example 1:
Input: [2,1,5]
Output: [5,5,0]

Example 2:
Input: [2,7,4,3,5]
Output: [7,0,5,5,0]

Example 3:
Input: [1,7,5,1,9,2,5,1]
Output: [7,9,9,9,0,5,0,0]
Note:
`1 <= node.val <= 10^9` for each node in the linked list.

The given list has length in the range `[0, 10000]`.

### Solution 
 To solve this problem, we will use a stack to keep track of nodes for which we've yet to find the next larger value. Specifically, while traversing the list, we will use this stack to store the indices from the linked list, and we'll compare the node's value with the value of the node at the stored index. If the current node's value is greater, it means we've found the next largest element for the node corresponding to the index at the top of the stack.

Here's how the algorithm will work:
1. Traverse the linked list and create an array of values while keeping the indices in the stack.
2. For each node:
   - Check if the stack is not empty and the current node's value is greater than the value of the node at the index present at the top of the stack.
   - If true, that means we found the next greater value for the elements indexed in the stack. Update the respective positions in the result array and pop the indices from the stack.
   - Push the index of the current node into the stack.
3. Nodes left in the stack by the end of the traversal don't have a next greater value, so set their corresponding positions in the result array to 0.

This approach leverages the stack to manage a decreasing order of node values effectively and updates the results in O(n) time complexity.

Here is a Python implementation suitable for a LeetCode submission:



```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def nextLargerNodes(self, head: ListNode):
        values = []
        stack = []
        result = []
        
        # First, collect all values from the linked list and prepare the results array.
        node = head
        while node:
            values.append(node.val)
            result.append(0)
            node = node.next
        
        # Use stack to keep track of the indices where the 'next larger' hasn't been found yet.
        for i, val in enumerate(values):
            while stack and values[stack[-1]] < val:
                result[stack.pop()] = val
            stack.append(i)
        
        return result

```

This solution first converts the linked list into a values array and initializes a result list (initially filled with zeros). Using a stack, we then fill the result list with next larger values in one pass over these values.

To test the solution on LeetCode, ensure that the class `ListNode` and structure is provided or compatible with their environment. The class`Solution()` and its method `nextLargerNodes(head)` will be applied directly to linked list inputs.

# 1171. Remove Zero Sum Consecutive Nodes from Linked List

### Problem Description 
Given the `head` of a linked list, we repeatedly delete consecutive sequences of nodes that sum to `0` until there are no such sequences.

After doing so, return the head of the final linked list.  You may return any such answer.

(Note that in the examples below, all sequences are serializations of `ListNode` objects.)

Example 1:
Input: head = [1,2,-3,3,1]
Output: [3,1]
Note: The answer [1,2,1] would also be accepted.


Example 2:
Input: head = [1,2,3,-3,4]
Output: [1,2,4]

Example 3:
Input: head = [1,2,3,-3,-2]
Output: [1]

Constraints:
The given linked list will contain between `1` and `1000` nodes.

Each node in the linked list has `-1000 <= node.val <= 1000`.

### Solution 
 To solve the problem using Python on LeetCode, we can use a strategy that involves prefix sums and a hashmap. By using this approach, we can efficiently identify and remove sublists that sum to zero. Here's how it works:

1. Maintain a running sum of the node values as you iterate through the linked list.
2. Use a hashmap to store the first occurrence of each prefix sum along with the corresponding node.
3. If a prefix sum repeats (i.e., it's found again in the hashmap), it means the sum of the nodes between these two occurrences is zero. We can then adjust the pointers to remove this range of nodes.
4. The algorithm will involve a single pass to identify nodes to remove and another pass to actually rebuild the list excluding the removed parts.

This approach minimally requires revisiting each node twice, making it quite efficient.

Here's the Python code using the described approach:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def removeZeroSumSublists(self, head: ListNode) -> ListNode:
        dummy = ListNode(0)  # create a dummy node to handle edge cases more uniformly
        dummy.next = head
        prefix_sum = 0
        sum_to_node = {0: dummy}

        # First pass: Record the first occurrence of each prefix sum 
        # and identify the nodes that reset the sum to a seen value
        current = head
        while current:
            prefix_sum += current.val
            if prefix_sum in sum_to_node:
                # Find the last node before the first occurrence of this sum
                prev = sum_to_node[prefix_sum]
                # Remove all nodes between this range
                temp = prev.next
                temp_sum = prefix_sum
                while temp != current:
                    temp_sum += temp.val
                    del sum_to_node[temp_sum]
                    temp = temp.next
                prev.next = current.next
            else:
                sum_to_node[prefix_sum] = current
            current = current.next
            
        # Cleansed list reconstruction from dummy
        return dummy.next

```

In this code:
- `dummy` is a dummy node added to simplify edge cases, such as when the entire list or the lead portion of the list sums to zero.
- `prefix_sum` keeps track of the cumulative sum from the dummy node to the current node.
- `sum_to_node` is a dictionary that maps a prefix sum to its corresponding node.

This solution efficiently handles the list by removing consecutive nodes summing to zero as soon as they're detected, without unnecessary re-traversals of the list.

# 1290. Convert Binary Number in a Linked List to Integer

### Problem Description 
Given `head` which is a reference node to a singly-linked list. The value of each node in the linked list is either 0 or 1. The linked list holds the binary representation of a number.

Return the decimal value of the number in the linked list.


Example 1:
Input: head = [1,0,1]
Output: 5
Explanation: (101) in base 2 = (5) in base 10

Example 2:
Input: head = [0]
Output: 0

Example 3:
Input: head = [1]
Output: 1

Example 4:
Input: head = [1,0,0,1,0,0,1,1,1,0,0,0,0,0,0]
Output: 18880

Example 5:
Input: head = [0,0]
Output: 0

Constraints:
The Linked List is not empty.

Number of nodes will not exceed `30`.

Each node's value is either `0` or `1`.

### Solution 
 To solve this problem in Python for a LeetCode format, we'll be using the properties of bitwise operations to convert the binary number represented by the linked list into a decimal value.

The idea is simple: 

1. Traverse the linked list from head to tail.
2. Take each node's value and add it to an accumulator.
3. For each new node after the first, left shift the accumulator (as though building the binary number from left to right).

Binary value aggregation using bit manipulation:
- If your current binary number is '101' and the next bit is '1':
  - Left shift the number by 1, resulting in '1010'.
  - Add the next bit, resulting in '1011'.

For the implementation, we start by defining the structure of a linked list node, then create the solution class to perform the operations described.

Here's how the solution can be implemented directly:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        result = 0
        while head:
            result = (result << 1) | head.val  # left-shift result and add current node's value
            head = head.next  # move to the next node
        return result

# Additional code below is just for running the example and will not be included when submitting on LeetCode:
def create_linked_list(values):
    dummy_head = ListNode(0)
    current = dummy_head
    for value in values:
        current.next = ListNode(value)
        current = current.next
    return dummy_head.next

# Test cases:
test_cases = [
    ([1,0,1], 5),
    ([0], 0),
    ([1], 1),
    ([1,0,0,1,0,0,1,1,1,0,0,0,0,0,0], 18880),
    ([0,0], 0)
]

sol = Solution()
for case in test_cases:
    head = create_linked_list(case[0])
    assert sol.getDecimalValue(head) == case[1], "Failed for input: {}".format(case[0])
print("All test cases passed.")

```

This script not only defines the necessary `ListNode` and `Solution` classes as required by LeetCode but also provides a simple way to create a linked list and test the solution with the example cases provided. When using this in LeetCode, you would only need to include definitions of the `ListNode` and `Solution` classes and the method to test (in an appropriate format such as test cases can be omitted).

# 1367. Linked List in Binary Tree

### Problem Description 
Given a binary tree `root` and a linked list with `head` as the first node. 
Return True if all the elements in the linked list starting from the `head` correspond to some downward path connected in the binary tree otherwise return False.

In this context downward path means a path that starts at some node and goes downwards.


Example 1:
Input: head = [4,2,8], root = [1,4,4,null,2,2,null,1,null,6,8,null,null,null,null,1,3]
Output: true
Explanation: Nodes in blue form a subpath in the binary Tree.  

Example 2:
Input: head = [1,4,2,6], root = [1,4,4,null,2,2,null,1,null,6,8,null,null,null,null,1,3]
Output: true

Example 3:
Input: head = [1,4,2,6,8], root = [1,4,4,null,2,2,null,1,null,6,8,null,null,null,null,1,3]
Output: false
Explanation: There is no path in the binary tree that contains all the elements of the linked list from `head`.


Constraints:
The number of nodes in the tree will be in the range `[1, 2500]`.

The number of nodes in the list will be in the range `[1, 100]`.

`1 <= Node.val <= 100` for each node in the linked list and binary tree.

### Solution 
 To solve this problem on LeetCode, we need to check if the sequence of nodes in the given linked list matches with any downward path (from top to bottom) in the binary tree.

### Python Solution

To achieve this, I'll follow these steps:
1. **DFS on Tree Nodes:** For every tree node, start a path-checking process (using another helper function) to verify if starting from this tree node, a downward path can match the linked list beginning from its head.
2. **Recursive Linked List Matching:** For each node where the value matches the current linked list node, recursively confirm if subsequent nodes (in both the linked list and tree) continue to match.

Here's the code with an explanation embedded as comments:



```python
# LeetCode requires definitions for ListNode and TreeNode. Assuming these are provided like:
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def isSubPath(self, head: ListNode, root: TreeNode) -> bool:
        
        if not root:
            return False
        
        # A helper function that checks if the linked list with this head
        # is a subpath starting from this given tree node
        def dfs(tree_node, list_node):
            # If the list is fully matched, return True
            if not list_node:
                return True
            # If tree is exhausted but list is still there, return False
            if not tree_node:
                return False
            # Check current node match
            if tree_node.val == list_node.val:
                # Continue with left and right subtree and the next node in the list
                return dfs(tree_node.left, list_node.next) or dfs(tree_node.right, list_node.next)
            else:
                return False
        
        # Traverse each node in the tree and start the path matching process
        return (dfs(root, head) or 
                self.isSubPath(head, root.left) or 
                self.isSubPath(head, root.right))

```

### Analysis
- **Time Complexity:** Each tree node is checked potentially up to `L` times where `L` is the linked list's length. Given `N` as the number of tree nodes, the worst-case time complexity becomes O(N * L).
- **Space Complexity:** In the worst case, the depth of the recursive stack could reach `L` for linked list and `H` for tree height, leading to a space complexity of O(H+L).

### How to Test and Submit on LeetCode:
- Copy the aforementioned class, `Solution`, into your LeetCode Python environment.
- The provided test cases by LeetCode and any additional test cases you create should be run to ensure accuracy.
- Click 'Submit' once the results match expected outcomes.

# 1474. Delete N Nodes After M Nodes of a Linked List

### Problem Description 
Given the `head` of a linked list and two integers `m` and `n`. Traverse the linked list and remove some nodes in the following way:
Start with the head as the current node.

Keep the first `m` nodes starting with the current node.

Remove the next `n` nodes
Keep repeating steps 2 and 3 until you reach the end of the list.

Return the head of the modified list after removing the mentioned nodes.

Follow up question: How can you solve this problem by modifying the list in-place?

Example 1:
Input: head = [1,2,3,4,5,6,7,8,9,10,11,12,13], m = 2, n = 3
Output: [1,2,6,7,11,12]
Explanation: Keep the first (m = 2) nodes starting from the head of the linked List  (1 ->2) show in black nodes.

Delete the next (n = 3) nodes (3 -> 4 -> 5) show in read nodes.

Continue with the same procedure until reaching the tail of the Linked List.

Head of linked list after removing nodes is returned.


Example 2:
Input: head = [1,2,3,4,5,6,7,8,9,10,11], m = 1, n = 3
Output: [1,5,9]
Explanation: Head of linked list after removing nodes is returned.


Example 3:
Input: head = [1,2,3,4,5,6,7,8,9,10,11], m = 3, n = 1
Output: [1,2,3,5,6,7,9,10,11]

Example 4:
Input: head = [9,3,7,7,9,10,8,2], m = 1, n = 2
Output: [9,7,8]

Constraints:
The given linked list will contain between `1` and `10^4` nodes.

The value of each node in the linked list will be in the range` [1, 10^6]`.

`1 <= m,n <= 1000`

### Solution 
 To solve the problem on LeetCode, we can use a direct approach of modifying the linked list in-place. We traverse the linked list using a pointer, maintaining nodes based on the counts `m` and `n`. For every `m` nodes that we retain, we skip the next `n` nodes, thereby effectively removing them from the list.

Here's a Python solution following the constraints and examples provided:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def deleteNodes(self, head: ListNode, m: int, n: int) -> ListNode:
        # Ensure the head is not None
        if not head:
            return head
        
        # Initialize the current node (pointer)
        current = head
        
        while current:
            # First, keep 'm' nodes
            for i in range(m - 1):
                if not current.next:
                    return head
                current = current.next
            
            # Now, we need to remove 'n' nodes
            remove = current
            for j in range(n):
                if not remove.next:
                    break
                remove = remove.next
            
            # Link the last kept node to the node after the removed ones
            current.next = remove.next
            current = current.next
        
        return head

# Helper function to create a linked list from a list of values
def create_linked_list(lst):
    if not lst:
        return None
    head = ListNode(lst[0])
    current = head
    for value in lst[1:]:
        current.next = ListNode(value)
        current = current.next
    return head

# Helper function to print linked list values
def print_linked_list(head):
    current = head
    while current:
        print(current.val, end=" -> ")
        current = current.next
    print("None")

# Example usage:
head = create_linked_list([1,2,3,4,5,6,7,8,9,10,11,12,13])
sol = Solution()
modified_head = sol.deleteNodes(head, 2, 3)
print_linked_list(modified_head)

```

Explanation:
- We define a `ListNode` class to simulate the linked list node.
- The `deleteNodes` function starts by checking if the head is not `None`.
- We then iterate over the linked list: for every group of `m` nodes we wish to keep, we directly skip `n` nodes and modify the `next` pointer to link non-deleted nodes.
- This is done in-place, adjusting the `next` references without creating a new list.
- Helper functions `create_linked_list` and `print_linked_list` provide simple utilities for testing the linked list manipulations.

This solution directly modifies the list in place, following the "in-place" follow-up question. It handles edge cases where the list may end during either the keep or remove phases by checking if `current.next` or `remove.next` is `None`.

# 1634. Add Two Polynomials Represented as Linked Lists

### Problem Description 
A polynomial linked list is a special type of linked list where every node represents a term in a polynomial expression.

Each node has three attributes:
`coefficient`: an integer representing the number multiplier of the term. The coefficient of the term `9x4` is `9`.

`power`: an integer representing the exponent. The power of the term `9x4` is `4`.

`next`: a pointer to the next node in the list, or `null` if it is the last node of the list.

For example, the polynomial `5x3 + 4x - 7` is represented by the polynomial linked list illustrated below:
The polynomial linked list must be in its standard form: the polynomial must be in strictly descending order by its `power` value. Also, terms with a `coefficient` of `0` are omitted.

Given two polynomial linked list heads, `poly1` and `poly2`, add the polynomials together and return the head of the sum of the polynomials.

`PolyNode` format:
The input/output format is as a list of `n` nodes, where each node is represented as its `[coefficient, power]`. For example, the polynomial `5x3 + 4x - 7` would be represented as: `[[5,3],[4,1],[-7,0]]`.


Example 1:
Input: poly1 = [[1,1]], poly2 = [[1,0]]
Output: [[1,1],[1,0]]
Explanation: poly1 = x. poly2 = 1. The sum is x + 1.


Example 2:
Input: poly1 = [[2,2],[4,1],[3,0]], poly2 = [[3,2],[-4,1],[-1,0]]
Output: [[5,2],[2,0]]
Explanation: poly1 = 2x2 + 4x + 3. poly2 = 3x2 - 4x - 1. The sum is 5x2 + 2. Notice that we omit the "0x" term.


Example 3:
Input: poly1 = [[1,2]], poly2 = [[-1,2]]
Output: []
Explanation: The sum is 0. We return an empty list.


Constraints:
`0 <= n <= 104`
`-109 <= PolyNode.coefficient <= 109`
`PolyNode.coefficient != 0`
`0 <= PolyNode.power <= 109`
`PolyNode.power > PolyNode.next.power`

### Solution 
 

```python
# Definition for polynomial singly-linked list.
class PolyNode:
    def __init__(self, x=0, y=0, next=None):
        self.coefficient = x
        self.power = y
        self.next = next

class Solution:
    def addPoly(self, poly1: 'PolyNode', poly2: 'PolyNode') -> 'PolyNode':
        # Initialize a dummy node to make result list management easier
        dummy = PolyNode()
        current = dummy

        # Traverse both linked lists
        while poly1 and poly2:
            if poly1.power > poly2.power:
                # poly1 term comes before poly2 term
                current.next = PolyNode(poly1.coefficient, poly1.power)
                poly1 = poly1.next
            elif poly1.power < poly2.power:
                # poly2 term comes before poly1 term
                current.next = PolyNode(poly2.coefficient, poly2.power)
                poly2 = poly2.next
            else:
                # Both are the same power, calculate sum of coefficients
                coeff_sum = poly1.coefficient + poly2.coefficient
                if coeff_sum != 0:  # Only add non-zero coefficients
                    current.next = PolyNode(coeff_sum, poly1.power)
                poly1 = poly1.next
                poly2 = poly2.next

            # If new node was added and it's not the final zero-sum case
            if current.next:
                current = current.next

        # If one of the lists still has terms, attach the remaining part
        if poly1:
            current.next = poly1
        elif poly2:
            current.next = poly2

        # Return the next of dummy, as dummy itself is not part of result
        return dummy.next

```

### Explanation:
1. **Initialization**: We create a dummy `PolyNode` to simplify the list handling by allowing us to always have a `current` node, avoiding the need to handle special cases for beginning of list.

2. **Traversal**: Using a while loop until both `poly1` and `poly2` are `None`, we determine which of the polynomials has the next term that should be added based on the order of `power`.

3. **Comparison and Addition**:
   - If the powers of current `poly1` and `poly2` are different, we simply add the node with a greater power to the resultant polynomial.
   - If they have the same power, we calculate the coefficient sum. If the sum is nonzero, we add this term to our result list.

4. **Merging Remaining Part**: After exiting the loop, at least one of the linked lists might still have nodes left (since they can be of unequal lengths or due to differing power ranges). We attach this remaining part directly to the result since it is already in the correct order.

5. **Edge Cases**: If coefficient sums become zero for a particular power, we do not add that term (`coeff_sum != 0` check).

This solution effectively adds two polynomial lists ensuring the resulting polynomial is in the fully combined and ordered form as specified by the constraints.

# 1669. Merge In Between Linked Lists

### Problem Description 
You are given two linked lists: `list1` and `list2` of sizes `n` and `m` respectively.

Remove `list1`'s nodes from the `ath` node to the `bth` node, and put `list2` in their place.

The blue edges and nodes in the following figure incidate the result:
Build the result list and return its head.


Example 1:
Input: list1 = [0,1,2,3,4,5], a = 3, b = 4, list2 = [1000000,1000001,1000002]
Output: [0,1,2,1000000,1000001,1000002,5]
Explanation: We remove the nodes 3 and 4 and put the entire list2 in their place. The blue edges and nodes in the above figure indicate the result.


Example 2:
Input: list1 = [0,1,2,3,4,5,6], a = 2, b = 5, list2 = [1000000,1000001,1000002,1000003,1000004]
Output: [0,1,1000000,1000001,1000002,1000003,1000004,6]
Explanation: The blue edges and nodes in the above figure indicate the result.


Constraints:
`3 <= list1.length <= 104`
`1 <= a <= b < list1.length - 1`
`1 <= list2.length <= 104`

### Solution 
 To solve this problem, we need to modify the first linked list (`list1`) by replacing a portion of its nodes (from the `ath` to the `bth` node) with another linked list (`list2`). Here's the step-by-step algorithm:

1. Traverse `list1` until reaching the node just before the `ath` node (`prevNode`). Keep a reference to this `prevNode`.
2. Continue traversing `list1` until reaching the `bth` node. Keep a reference to the node just after `bth` (`nextNode`).
3. Link the `prevNode.next` to the head of `list2`.
4. Traverse to the end of `list2`.
5. Link the last node of `list2` to `nextNode`.

Now, let's directly implement the solution.



```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
        dummy = ListNode(0)  # Create a dummy node to simplify edge cases
        dummy.next = list1
        current = dummy
        
        # Step 1: Find the node before `a`
        for i in range(a):
            current = current.next
        nodeBeforeA = current
        
        # Step 2: Find the node after `b`
        current = list1
        for i in range(b + 1):
            current = current.next
        nodeAfterB = current
        
        # Step 3: Attach list2 to `nodeBeforeA`
        nodeBeforeA.next = list2
        
        # Step 4: Traverse to the end of `list2` 
        while list2.next:
            list2 = list2.next
        
        # Step 5: Attach `nodeAfterB` to the end of list2
        list2.next = nodeAfterB
        
        return dummy.next

# Example usage:
# Let's construct the lists from the example:
# It would be used like this in a test function or a main function outside.
head1 = ListNode(0)
current = head1
for val in [1, 2, 3, 4, 5]:
    current.next = ListNode(val)
    current = current.next

head2 = ListNode(1000000)
current = head2
for val in [1000001, 1000002]:
    current.next = ListNode(val)
    current = current.next

sol = Solution()
result = sol.mergeInBetween(head1, 3, 4, head2)
while result:
    print(result.val)
    result = result.next

```

The provided code defines the `ListNode` class which represents a node in a singly-linked list. The `Solution` class contains one method, `mergeInBetween`, which performs the described operation. The test at the end constructs the linked lists according to the example in the problem description, and outputs the modified linked list's values which should print out `0, 1, 2, 1000000, 1000001, 1000002, 5` as in Example 1.

# 1670. Design Front Middle Back Queue

### Problem Description 
Design a queue that supports `push` and `pop` operations in the front, middle, and back.

Implement the `FrontMiddleBack` class:
`FrontMiddleBack()` Initializes the queue.

`void pushFront(int val)` Adds `val` to the front of the queue.

`void pushMiddle(int val)` Adds `val` to the middle of the queue.

`void pushBack(int val)` Adds `val` to the back of the queue.

`int popFront()` Removes the front element of the queue and returns it. If the queue is empty, return `-1`.

`int popMiddle()` Removes the middle element of the queue and returns it. If the queue is empty, return `-1`.

`int popBack()` Removes the back element of the queue and returns it. If the queue is empty, return `-1`.

Notice that when there are two middle position choices, the operation is performed on the frontmost middle position choice. For example:
Pushing `6` into the middle of `[1, 2, 3, 4, 5]` results in `[1, 2, 6, 3, 4, 5]`.

Popping the middle from `[1, 2, 3, 4, 5, 6]` returns `3` and results in `[1, 2, 4, 5, 6]`.


Example 1:
Input:
["FrontMiddleBackQueue", "pushFront", "pushBack", "pushMiddle", "pushMiddle", "popFront", "popMiddle", "popMiddle", "popBack", "popFront"]
[[], [1], [2], [3], [4], [], [], [], [], []]
Output:
[null, null, null, null, null, 1, 3, 4, 2, -1]
Explanation:
FrontMiddleBackQueue q = new FrontMiddleBackQueue();
q.pushFront(1);   // [1]
q.pushBack(2);    // [1, 2]
q.pushMiddle(3);  // [1, 3, 2]
q.pushMiddle(4);  // [1, 4, 3, 2]
q.popFront();     // return 1 -> [4, 3, 2]
q.popMiddle();    // return 3 -> [4, 2]
q.popMiddle();    // return 4 -> [2]
q.popBack();      // return 2 -> []
q.popFront();     // return -1 -> [] (The queue is empty)

Constraints:
`1 <= val <= 109`
At most `1000` calls will be made to `pushFront`, `pushMiddle`, `pushBack`, `popFront`, `popMiddle`, and `popBack`.

### Solution 
 Here is a Python code solution to the given problem compatible with the LeetCode platform. This code uses Python lists to simulate the queue operations effectively. We handle each operation by directly manipulating the queue list, adjusting items as necessary for front, middle, and back operations.



```python
class FrontMiddleBackQueue:

    def __init__(self):
        self.queue = []

    def pushFront(self, val: int) -> None:
        self.queue.insert(0, val)

    def pushMiddle(self, val: int) -> None:
        middle_index = len(self.queue) // 2
        self.queue.insert(middle_index, val)

    def pushBack(self, val: int) -> None:
        self.queue.append(val)

    def popFront(self) -> int:
        return self.queue.pop(0) if self.queue else -1

    def popMiddle(self) -> int:
        if not self.queue:
            return -1
        middle_index = (len(self.queue) - 1) // 2 # -1 to handle when count is even, we need to take the front most of two middles
        return self.queue.pop(middle_index)

    def popBack(self) -> int:
        return self.queue.pop() if self.queue else -1


# Your FrontMiddleBackQueue object will be instantiated and called as such:
# obj = FrontMiddleBackQueue()
# obj.pushFront(val)
# obj.pushMiddle(val)
# obj.pushBack(val)
# param_4 = obj.popFront()
# param_5 = obj.popMiddle()
# param_6 = obj.popBack()

```

Explanation:

1. **Initialization**: The constructor `__init__()` initializes an empty list that will act as the queue.
2. **pushFront**: By using the list method `insert()`, we add elements at the 0th position to push them to the front.
3. **pushMiddle**: The middle index for insertion is computed as half the length of the list. This ensures elements are inserted in the middle, or towards the end of the first half of the list if the list length is even.
4. **pushBack**: Utilize the list method `append()` to add elements at the back of the list.
5. **popFront**: Remove and return the element from the front using `pop(0)`. If the list is empty, we return `-1`.
6. **popMiddle**: If the list is not empty, the middle element's index is identified and removed using the `pop()` method. When the list length is even, we adjust to remove the front middle.
7. **popBack**: Remove and return the last element with `pop()`, accommodating an empty list scenario by returning `-1`.

Each function efficiently modifies the list to perform the specified queue operation. This code should be suitable for direct execution and testing on LeetCode under the constraints and examples provided.

# 1721. Swapping Nodes in a Linked List

### Problem Description 
You are given the `head` of a linked list, and an integer `k`.

Return the head of the linked list after swapping the values of the `kth` node from the beginning and the `kth` node from the end (the list is 1-indexed).


Example 1:
Input: head = [1,2,3,4,5], k = 2
Output: [1,4,3,2,5]

Example 2:
Input: head = [7,9,6,6,7,8,3,0,9,5], k = 5
Output: [7,9,6,6,8,7,3,0,9,5]

Example 3:
Input: head = [1], k = 1
Output: [1]

Example 4:
Input: head = [1,2], k = 1
Output: [2,1]

Example 5:
Input: head = [1,2,3], k = 2
Output: [1,2,3]

Constraints:
The number of nodes in the list is `n`.

`1 <= k <= n <= 105`
`0 <= Node.val <= 100`

### Solution 
 To solve the problem of swapping the `kth` node from the beginning and `kth` node from the end in a linked list, we can approach it by using pointers to traverse the list. Here's a step-by-step breakdown:

1. **Traversal and Length**: Begin by traversing the linked list once to determine the total length `n` of the list.
2. **Identify Nodes for Swap**: Knowing the total length allows us to easily compute the positions of the `kth` node from the beginning and the `kth` node from the end (which is `n-k+1` from the beginning). 
3. **Node Swap**: If we denote `nodeK` as our `kth` node from the beginning and `nodeNMinusKPlus1` as our `kth` node from the end, we find these nodes in a second pass and swap their values.

This approach has a time complexity of O(n) due to two traversals (both linear traversals of the list), but we maintain a space complexity of O(1) because we're only swapping node values rather than manipulating the nodes themselves.

Let's proceed with the Python implementation using this plan:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def swapNodes(self, head: ListNode, k: int) -> ListNode:
        # Initialize pointers and a counter to determine the length of the list
        fast = slow = head
        counter = 1
        
        # First pass to determine `n` and find the `k`th node from the beginning
        while counter < k:
            fast = fast.next
            counter += 1
        
        # `kth` node from the beginning
        nodeK = fast
        
        # Continue to the end to determine `n` and find the `k`th node from the end
        while fast.next:
            fast = fast.next
            slow = slow.next
        
        # `kth` node from the end
        nodeNMinusKPlus1 = slow
        
        # Swap values
        nodeK.val, nodeNMinusKPlus1.val = nodeNMinusKPlus1.val, nodeK.val
        
        # Return the updated list
        return head

```

Explanation:
- The `ListNode` class is a definition for linked list nodes.
- `swapNodes(self, head, k)` function swaps the `k`th node from the beginning and the end of the list, as described, and returns the modified list.
- `fast` pointer is used to locate the end of the list and the `slow` pointer will point to the `kth` node from the end after we find `k`th node from the beginning.
- The swapping is simply value-based, not node reference-based, ensuring minimal changes necessary to achieve the desired outcome.

This solution will work efficiently within the provided constraints and is ready to be tested on the LeetCode platform for verification.

