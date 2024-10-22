# 382. Linked List Random Node

### Problem Description 
Given a singly linked list, return a random node's value from the linked list. Each node must have the same probability of being chosen.


Example 1:
Input
["Solution", "getRandom", "getRandom", "getRandom", "getRandom", "getRandom"]
[[[1, 2, 3]], [], [], [], [], []]
Output
[null, 1, 3, 2, 2, 3]
Explanation
Solution solution = new Solution([1, 2, 3]);
solution.getRandom(); // return 1
solution.getRandom(); // return 3
solution.getRandom(); // return 2
solution.getRandom(); // return 2
solution.getRandom(); // return 3
// getRandom() should return either 1, 2, or 3 randomly. Each element should have equal probability of returning.


Constraints:
The number of nodes in the linked list will be in the range `[1, 104]`.

`-104 <= Node.val <= 104`
At most `104` calls will be made to `getRandom`.

Follow up:
What if the linked list is extremely large and its length is unknown to you?
Could you solve this efficiently without using extra space?

### Solution 
 To solve the problem of returning a random node's value from a singly linked list with equal probability for each node, we can use a reservoir sampling technique. This allows us to traverse the linked list just once while keeping the memory usage constant. Here's how we can implement it in Python:

### Explanation

1. **Data Structure**:
   We need a class to represent our linked list (`ListNode`) and another class to implement the solution (`Solution`).

2. **Reservoir Sampling**:
   - Initialize a counter to count the nodes as we traverse the list.
   - When we encounter each node, we decide with a probability whether to select the current node's value.
   - Specifically, for every node visited (let's say the i-th node), we generate a random number. If this number is `0`, we chose the current node's value. This ensures that each node has an equal probability of being selected.

3. **Random Selection**:
   - Use the `random.randint` function to generate a random number.
   - Since we want to give equal probability to each of the nodes, we check if the generated random number equals the current index (i.e., if we're on the first node selected, we'd use 1 as the generated random number).

### Code Implementation

Here’s how the code would look, structured in a way that's compatible with LeetCode:



```python
import random

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def __init__(self, head: ListNode):
        self.head = head

    def getRandom(self) -> int:
        current = self.head
        result = current.val
        count = 1  # Start counting from the first node
        
        while current is not None:
            if random.randint(0, count - 1) == 0:  # Randomly select this node
                result = current.val
            current = current.next
            count += 1
            
        return result

# Example usage:
# head = ListNode(1, ListNode(2, ListNode(3)))
# solution = Solution(head)
# print(solution.getRandom())  # It should return 1, 2, or 3 with equal probability.

```

### Key Points of the Code:

1. **ListNode Class**: Represents each node of the singly linked list.
2. **Solution Class**:
   - The `__init__` method initializes the `Solution` object with the head of the linked list.
   - The `getRandom` method traverses the linked list using a while loop and applies reservoir sampling as explained.
3. **Random Selection**: The line `if random.randint(0, count - 1) == 0:` ensures that each node has a \( \frac{1}{n} \) chance of being selected, where \( n \) is the number of nodes seen so far.

### Complexity Analysis:
- **Time Complexity**: \( O(n) \), where \( n \) is the number of nodes in the linked list, because we need to traverse all nodes.
- **Space Complexity**: \( O(1) \), as we are using a constant amount of extra space regardless of the input size. 

This method efficiently handles large linked lists and maintains equal probability for each node's value when selected randomly.

# 398. Random Pick Index

### Problem Description 
Given an integer array `nums` with possible duplicates, randomly output the index of a given `target` number. You can assume that the given target number must exist in the array.

Implement the `Solution` class:
`Solution(int[] nums)` Initializes the object with the array `nums`.

`int pick(int target)` Picks a random index `i` from `nums` where `nums[i] == target`. If there are multiple valid i's, then each index should have an equal probability of returning.


Example 1:
Input
["Solution", "pick", "pick", "pick"]
[[[1, 2, 3, 3, 3]], [3], [1], [3]]
Output
[null, 4, 0, 2]
Explanation
Solution solution = new Solution([1, 2, 3, 3, 3]);
solution.pick(3); // It should return either index 2, 3, or 4 randomly. Each index should have equal probability of returning.

solution.pick(1); // It should return 0. Since in the array only nums[0] is equal to 1.

solution.pick(3); // It should return either index 2, 3, or 4 randomly. Each index should have equal probability of returning.


Constraints:
`1 <= nums.length <= 2 * 104`
`-231 <= nums[i] <= 231 - 1`
`target` is an integer from `nums`.

At most `104` calls will be made to `pick`.

### Solution 
 To solve the problem of randomly picking an index of a given target number from an array containing possible duplicates, we will utilize a dictionary to map each unique number to a list of indices where that number appears. This will allow us to efficiently retrieve indices for the target number and return a random index among those.

Here's a detailed step-by-step explanation and implementation using the provided format:

### Step-by-Step Explanation

1. **Initialization**: When initializing the `Solution` class with the array `nums`, we will create a dictionary where each key is a unique number from the array, and the corresponding value is a list of indices where that number appears in the array.

2. **Picking Method**: The `pick` method will accept a target number. We will retrieve the list of indices from our dictionary that corresponds to the target. Then, we will use Python’s `random.choice` to randomly select an index from the list of indices.

3. **Randomness**: This approach ensures that every index associated with the target has an equal probability of being returned, as `random.choice` selects randomly from the list of indices.

4. **Complexity**: The initialization takes O(n) time to create the index mapping, and each `pick` operation takes O(1) on average for accessing the dictionary and O(1) for choosing a random element from a small list.

### Python Solution

Here is the implementation as requested:



```python
import random
from collections import defaultdict

class Solution:

    def __init__(self, nums: List[int]):
        self.index_map = defaultdict(list)
        
        # Build the index map
        for index, num in enumerate(nums):
            self.index_map[num].append(index)

    def pick(self, target: int) -> int:
        # Get the list of indices for the target
        indices = self.index_map[target]
        # Pick a random index from the list of indices
        return random.choice(indices)

```

### How to Use This Class

When you create an instance of `Solution` with a specific array, the initialization step builds the dictionary of indices. You can then call the `pick` method with a target number to get a random index each time.

### Example Usage



```python
# Example usage:
solution = Solution([1, 2, 3, 3, 3])
print(solution.pick(3))  # This may return 2, 3, or 4 randomly
print(solution.pick(1))  # This will always return 0
print(solution.pick(3))  # This may return 2, 3, or 4 randomly again

```

This structure ensures that the code runs efficiently and meets the problem constraints. Each time the `pick` method is called, it will return a valid index where the target appears, adhering to the problem’s requirements.

