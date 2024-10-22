# 239. Sliding Window Maximum

### Problem Description 
You are given an array of integers `nums`, there is a sliding window of size `k` which is moving from the very left of the array to the very right. You can only see the `k` numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window.


Example 1:
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

Example 2:
Input: nums = [1], k = 1
Output: [1]

Example 3:
Input: nums = [1,-1], k = 1
Output: [1,-1]

Example 4:
Input: nums = [9,11], k = 2
Output: [11]

Example 5:
Input: nums = [4,-2], k = 2
Output: [4]

Constraints:
`1 <= nums.length <= 105`
`-104 <= nums[i] <= 104`
`1 <= k <= nums.length`

### Solution 
 To solve the problem of finding the maximum in each sliding window of size `k` in an array of integers `nums`, we can employ an efficient algorithm using a deque (double-ended queue). This algorithm efficiently keeps track of the indices of the maximum elements, ensuring that we traverse the array in linear time. 

Here’s a breakdown of the algorithm:

1. **Initialization**: We'll use a deque to store the indices of the elements in the current window. This deque will maintain the indices in decreasing order of their values in `nums`, so the front of the deque will always be the index of the maximum value for the current window.

2. **Sliding the Window**:
   - We iterate through each element in the `nums` array.
   - For each element, we will:
     - Remove indices from the front of the deque that are out of the bounds of the sliding window.
     - Remove indices from the back of the deque for elements that are less than or equal to the current element, since they can never be the maximum when the current element is in the window.
     - Append the current index to the deque.
     - If we have filled our first `k` elements (i.e., at least the window size has been reached), we will add the maximum (which is at the front of the deque) to our result list.

3. **Result Construction**: Finally, we return the result list which contains the maximums of each sliding window.

The time complexity of this solution is O(n), where n is the number of elements in `nums`, since each element is added and removed from the deque at most once.

Here's the Python code implementing this algorithm formatted for LeetCode:



```python
from collections import deque
from typing import List

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums:
            return []

        n = len(nums)
        result = []
        dq = deque()  # This will hold the indices of nums

        for i in range(n):
            # Remove indices of elements not in the sliding window
            if dq and dq[0] < i - k + 1:
                dq.popleft()

            # Remove indices of all elements in the deque that are smaller than
            # the current element (as they cannot be maximum for this or any
            # future window).
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()

            # Add the current element index to the deque
            dq.append(i)

            # Starting from the index k-1, we can start adding to the result as
            # the first valid maximum is established at that point
            if i >= k - 1:
                result.append(nums[dq[0]])  # The front of the deque is the maximum

        return result

```

### Explanation of the Code:
- We start by importing the necessary modules and defining the `Solution` class.
- The `maxSlidingWindow` method is defined to accept a list of integers `nums` and an integer `k`.
- We handle cases where `nums` is empty.
- We initialize our result list and the deque.
- We iterate through `nums` with an index `i`:
  - We check for indices that are out of the current window (`i - k + 1`).
  - We remove from the back of the deque all elements that are smaller than the current element to maintain the decreasing order.
  - We append the current index `i` to the deque.
  - Once we reach a position where the first window is fully formed (`i >= k - 1`), we append the maximum (at the front of the deque) to the result list.
- Finally, we return the result list containing the maximums of each sliding window. 

This solution runs efficiently within the constraints provided and can handle the maximum input sizes specified in the problem.

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
 To solve the problem of implementing a `FrontMiddleBackQueue` class that supports special operations at the front, middle, and back of a queue, we need to ensure that all operations are efficient. We can achieve this with two deques—one for the front half of the queue and another for the back half. This design allows us to quickly access and modify the front, middle, and back elements.

Here's the complete Python implementation along with a detailed explanation.

### Python Code Implementation



```python
from collections import deque

class FrontMiddleBackQueue:
    def __init__(self):
        # Two deques to represent the front and back halves of the queue
        self.front = deque()
        self.back = deque()

    def pushFront(self, val: int) -> None:
        self.front.appendleft(val)
        self._rebalance()

    def pushMiddle(self, val: int) -> None:
        if len(self.front) > len(self.back):
            self.back.appendleft(self.front.pop())
            self.front.append(val)
        else:
            self.front.append(val)
        self._rebalance()

    def pushBack(self, val: int) -> None:
        self.back.append(val)
        self._rebalance()

    def popFront(self) -> int:
        if not self.front and not self.back:
            return -1
        if self.front:
            result = self.front.popleft()
        else:
            result = self.back.popleft()
        self._rebalance()
        return result

    def popMiddle(self) -> int:
        if not self.front and not self.back:
            return -1
        if len(self.front) > len(self.back):
            result = self.front.pop()
        else:
            result = self.back.popleft()
        self._rebalance()
        return result

    def popBack(self) -> int:
        if not self.back and not self.front:
            return -1
        if self.back:
            result = self.back.pop()
        else:
            result = self.front.pop()
        self._rebalance()
        return result

    def _rebalance(self):
        # Ensure the sizes of the two halves are balanced
        if len(self.front) > len(self.back) + 1:
            self.back.appendleft(self.front.pop())
        elif len(self.back) > len(self.front):
            self.front.append(self.back.popleft())

```

### Detailed Explanation

1. **Data Structure**:
   - We utilize two `deque` (double-ended queues): `front` and `back`. This allows us to efficiently perform operations on both ends and effectively manage the middle element.

2. **Push Operations**:
   - **pushFront**: Adds a value to the front of the `front` deque and then calls `_rebalance` to maintain size constraints.
   - **pushMiddle**: If the `front` deque has more elements than `back`, we move the last element of `front` to `back` before adding the new middle element to `front`. This ensures that the middle operation always affects the correct "middle" element. If `front` and `back` are balanced or `back` has fewer elements, the new value gets directly added to `front`.
   - **pushBack**: Simply appends a value to the `back` deque and then rebalances.

3. **Pop Operations**:
   - **popFront**: If there are elements in `front`, it pops from there. If `front` is empty, we pop from `back`. After popping, we rebalance.
   - **popMiddle**: Similar to `popFront`, it pops the middle element based on the size of `front` and `back`. The procedure handles the balance to ensure middle element operations always work correctly.
   - **popBack**: Pops from the `back` deque if it has elements. If `back` is empty, it pops from `front`.

4. **Rebalance Method**:
   - This auxiliary method maintains the size balance between `front` and `back`. After every push or pop operation, we ensure that `front` can have at most one more element than `back`, which helps in determining the middle element correctly.

This design ensures that all operations are efficient, leading to an average time complexity of O(1) per operation due to the properties of the deque structure. This implementation is directly runnable on LeetCode or any similar environment.

# 1687. Delivering Boxes from Storage to Ports

### Problem Description 
You have the task of delivering some boxes from storage to their ports using only one ship. However, this ship has a limit on the number of boxes and the total weight that it can carry.

You are given an array `boxes`, where `boxes[i] = [ports​​i​, weighti]`, and three integers `portsCount`, `maxBoxes`, and `maxWeight`.

`ports​​i` is the port where you need to deliver the `ith` box and `weightsi` is the weight of the `ith` box.

`portsCount` is the number of ports.

`maxBoxes` and `maxWeight` are the respective box and weight limits of the ship.

The boxes need to be delivered in the order they are given. The ship will follow these steps:
The ship will take some number of boxes from the `boxes` queue, not violating the `maxBoxes` and `maxWeight` constraints.

For each loaded box in order, the ship will make a trip to the port the box needs to be delivered to and deliver it. If the ship is already at the correct port, no trip is needed, and the box can immediately be delivered.

The ship then makes a return trip to storage to take more boxes from the queue.

The ship must end at storage after all the boxes have been delivered.

Return the minimum number of trips the ship needs to make to deliver all boxes to their respective ports.


Example 1:
Input: boxes = [[1,1],[2,1],[1,1]], portsCount = 2, maxBoxes = 3, maxWeight = 3
Output: 4
Explanation: The optimal strategy is as follows: 
- The ship takes all the boxes in the queue, goes to port 1, then port 2, then port 1 again, then returns to storage. 4 trips.

So the total number of trips is 4.

Note that the first and third boxes cannot be delivered together because the boxes need to be delivered in order (i.e. the second box needs to be delivered at port 2 before the third box).


Example 2:
Input: boxes = [[1,2],[3,3],[3,1],[3,1],[2,4]], portsCount = 3, maxBoxes = 3, maxWeight = 6
Output: 6
Explanation: The optimal strategy is as follows: 
- The ship takes the first box, goes to port 1, then returns to storage. 2 trips.

- The ship takes the second, third and fourth boxes, goes to port 3, then returns to storage. 2 trips.

- The ship takes the fifth box, goes to port 3, then returns to storage. 2 trips.

So the total number of trips is 2 + 2 + 2 = 6.


Example 3:
Input: boxes = [[1,4],[1,2],[2,1],[2,1],[3,2],[3,4]], portsCount = 3, maxBoxes = 6, maxWeight = 7
Output: 6
Explanation: The optimal strategy is as follows:
- The ship takes the first and second boxes, goes to port 1, then returns to storage. 2 trips.

- The ship takes the third and fourth boxes, goes to port 2, then returns to storage. 2 trips.

- The ship takes the fifth and sixth boxes, goes to port 3, then returns to storage. 2 trips.

So the total number of trips is 2 + 2 + 2 = 6.


Example 4:
Input: boxes = [[2,4],[2,5],[3,1],[3,2],[3,7],[3,1],[4,4],[1,3],[5,2]], portsCount = 5, maxBoxes = 5, maxWeight = 7
Output: 14
Explanation: The optimal strategy is as follows:
- The ship takes the first box, goes to port 2, then storage. 2 trips.

- The ship takes the second box, goes to port 2, then storage. 2 trips.

- The ship takes the third and fourth boxes, goes to port 3, then storage. 2 trips.

- The ship takes the fifth box, goes to port 3, then storage. 2 trips.

- The ship takes the sixth and seventh boxes, goes to port 3, then port 4, then storage. 3 trips. 
- The ship takes the eighth and ninth boxes, goes to port 1, then port 5, then storage. 3 trips.

So the total number of trips is 2 + 2 + 2 + 2 + 3 + 3 = 14.


Constraints:
`1 <= boxes.length <= 105`
`1 <= portsCount, maxBoxes, maxWeight <= 105`
`1 <= ports​​i <= portsCount`
`1 <= weightsi <= maxWeight`

### Solution 
 Certainly! The problem you're dealing with requires simulating the process of delivering boxes from storage to their respective ports while adhering to certain constraints. Below, I'll provide a Python solution that follows the LeetCode format.

### Problem Breakdown
1. **Inputs**:
    - `boxes`: A list containing pairs of (port, weight).
    - `portsCount`: The total number of unique ports.
    - `maxBoxes`: The maximum number of boxes the ship can carry at once.
    - `maxWeight`: The maximum weight the ship can carry at once.

2. **Constraints**:
    - Boxes must be delivered in the order they are given.
    - The ship needs to return to storage after delivering boxes.

3. **Output**:
    - Return the minimum number of trips the ship needs to make to deliver all boxes.

### Plan
To solve this, we can use a greedy approach:
- Use a loop to iterate through the boxes.
- Maintain counters for `current_weight` and `current_boxes`.
- When the ship's weight capacity or box limit is reached, count a trip and reset for the next round.
- Always factor in the trips to the respective ports and back to storage.

### Python Code


```python
def boxDelivering(boxes, portsCount, maxBoxes, maxWeight):
    trips = 0  # Total trips made
    current_weight = 0  # Current weight of boxes loaded
    current_boxes = 0  # Current number of boxes loaded
    n = len(boxes)  # Number of boxes
    last_port = -1  # Keep track of the last port visited
    
    i = 0
    while i < n:
        trips += 1  # Starting a trip
        current_weight = 0
        current_boxes = 0
        
        while i < n and current_boxes < maxBoxes and current_weight + boxes[i][1] <= maxWeight:
            current_weight += boxes[i][1]
            current_boxes += 1
            
            # If we are at a different port, we can count the delivery. We need to go to this port.
            if last_port != boxes[i][0]:
                last_port = boxes[i][0]
                
            i += 1
        
        # After delivering all boxes in this trip, we return to storage.
    
    return trips

```

### Explanation
- Initialize the `trips` counter to keep track of the number of trips made by the ship.
- Loop over the boxes:
  - Start a new trip and reset `current_weight` and `current_boxes`.
  - Load boxes onto the ship as long as it doesn’t exceed `maxBoxes` or `maxWeight`.
  - Each time a port is visited, update `last_port`.
- For each trip initiated, increment the `trips` count.
- Finally, return the total number of trips made.

### Complexity Analysis
- **Time Complexity**: O(n), where n is the number of boxes. Each box is processed at most twice (once when loading and once when counted in trips).
- **Space Complexity**: O(1), as we are using a fixed amount of extra space.

This solution effectively manages the constraints of the problem and ensures that all boxes are delivered in optimal trips.

# 1696. Jump Game VI

### Problem Description 
You are given a 0-indexed integer array `nums` and an integer `k`.

You are initially standing at index `0`. In one move, you can jump at most `k` steps forward without going outside the boundaries of the array. That is, you can jump from index `i` to any index in the range `[i + 1, min(n - 1, i + k)]` inclusive.

You want to reach the last index of the array (index `n - 1`). Your score is the sum of all `nums[j]` for each index `j` you visited in the array.

Return the maximum score you can get.


Example 1:
Input: nums = [1,-1,-2,4,-7,3], k = 2
Output: 7
Explanation: You can choose your jumps forming the subsequence [1,-1,4,3] (underlined above). The sum is 7.


Example 2:
Input: nums = [10,-5,-2,4,0,3], k = 3
Output: 17
Explanation: You can choose your jumps forming the subsequence [10,4,3] (underlined above). The sum is 17.


Example 3:
Input: nums = [1,-5,-20,4,-1,3,-6,-3], k = 2
Output: 0

Constraints:
 `1 <= nums.length, k <= 105`
`-104 <= nums[i] <= 104`

### Solution 
 To solve this problem, we can use Dynamic Programming along with a deque (double-ended queue) to maintain the maximum scores of reachable indices efficiently. The idea is to iteratively build up the maximum score possible at each index, using previously calculated scores of indices that can be jumped from.

### Explanation

1. **Dynamic Programming Array**: We'll create an array `dp` where `dp[i]` will store the maximum score we can achieve at index `i`.

2. **Initialization**: Start by setting `dp[0]` to `nums[0]`, as that is where we start.

3. **Deque for Maximum Scores**: Use a deque to maintain indices of the `dp` array. This deque will help us efficiently retrieve the maximum score from previous indices that can jump to the current index. It will store indices in a way that the highest score is always at the front.

4. **Iteration**: For each index `i` from `1` to `n - 1`, we'll:
   - Remove indices from the front of the deque that are out of the jump range (i.e., indices less than `i - k`).
   - The maximum score to come to `i` can then be fetched from the front of the deque.
   - Calculate the score at `i` using `dp[i] = max(dp[i], dp[max_index] + nums[i])`, where `max_index` is the index at the front of the deque.
   - Add the current index `i` to the deque, maintaining the property that the indices in it are sorted by their scores in `dp`.

5. **Result**: The answer is found at `dp[n - 1]`, which contains the maximum score obtainable at the last index.

### Implementation

Here is the Python implementation following the LeetCode format:



```python
from collections import deque

class Solution:
    def maxScore(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [0] * n  # DP array to store maximum scores
        dp[0] = nums[0]  # Starting at index 0
        
        dq = deque([0])  # Deque to maintain indices of dp array
        
        for i in range(1, n):
            # Remove indices that are outside the jump range
            while dq and dq[0] < i - k:
                dq.popleft()

            # Calculate the maximum score at index i
            dp[i] = dp[dq[0]] + nums[i]
            
            # Maintain the deque to keep it ordered by dp values
            while dq and dp[dq[-1]] < dp[i]:
                dq.pop()
                
            dq.append(i)
        
        return dp[-1]  # Return the maximum score at the last index

# Example usage:
# sol = Solution()
# print(sol.maxScore([1,-1,-2,4,-7,3], 2))  # Output: 7
# print(sol.maxScore([10,-5,-2,4,0,3], 3))  # Output: 17
# print(sol.maxScore([1,-5,-20,4,-1,3,-6,-3], 2))  # Output: 0

```

### Complexity

- **Time Complexity**: O(n), because each index is processed once and each index is added and removed from the deque at most once.
- **Space Complexity**: O(n), for the `dp` array and the deque.

This efficient algorithm allows us to compute the maximum score at the last index of the array while adhering to the jumping constraints of `k`.

# 1756. Design Most Recently Used Queue

### Problem Description 
Design a queue-like data structure that moves the most recently used element to the end of the queue.

Implement the `MRUQueue` class:
`MRUQueue(int n)` constructs the `MRUQueue` with `n` elements: `[1,2,3,...,n]`.

`fetch(int k)` moves the `kth` element (1-indexed) to the end of the queue and returns it.


Example 1:
Input:
["MRUQueue", "fetch", "fetch", "fetch", "fetch"]
[[8], [3], [5], [2], [8]]
Output:
[null, 3, 6, 2, 2]
Explanation:
MRUQueue mRUQueue = new MRUQueue(8); // Initializes the queue to [1,2,3,4,5,6,7,8].

mRUQueue.fetch(3); // Moves the 3rd element (3) to the end of the queue to become [1,2,4,5,6,7,8,3] and returns it.

mRUQueue.fetch(5); // Moves the 5th element (6) to the end of the queue to become [1,2,4,5,7,8,3,6] and returns it.

mRUQueue.fetch(2); // Moves the 2nd element (2) to the end of the queue to become [1,4,5,7,8,3,6,2] and returns it.

mRUQueue.fetch(8); // The 8th element (2) is already at the end of the queue so just return it.


Constraints:
`1 <= n <= 2000`
`1 <= k <= n`
At most `2000` calls will be made to `fetch`.

Follow up: Finding an `O(n)` algorithm per `fetch` is a bit easy. Can you find an algorithm with a better complexity for each `fetch` call?

### Solution 
 To implement the `MRUQueue` class, we need to maintain a data structure that allows us to efficiently move elements around, particularly moving the most recently used element to the end of the queue. We'll employ a list to represent the queue and a dictionary to allow for quicker access when fetching elements.

Here's a breakdown of how we can implement this:

### Data Structure
- **List**: We'll use a list to maintain the order of elements. The list will represent the queue and will start with the elements `[1, 2, ..., n]`.
- **Dictionary**: We will also maintain a dictionary that keeps track of the indices of elements. This will help us fetch any element in O(1) time, allowing us to move it to the end of the queue efficiently.

### Operations
1. **Constructor**: Initialize the elements from 1 to `n` into a list.
2. **Fetch Operation**: For the `fetch(k)` method, we'll pull the element at index `k-1` since our list is 0-indexed. We'll then move this element to the end of the list. If the element is already at the end, we simply return it without moving.
3. **Updating Indices**: After moving an element, we need to update our dictionary to reflect the new indices.

### Python Code
Below is the implementation of the `MRUQueue` class following the LeetCode format.



```python
class MRUQueue:

    def __init__(self, n: int):
        self.queue = [i for i in range(1, n + 1)]
        self.index_map = {i: i - 1 for i in range(1, n + 1)}  # Maps element to its index in queue

    def fetch(self, k: int) -> int:
        idx = self.index_map[k]  # Get the index of the k-th element
        value = self.queue[idx]  # Get the value at that index
        
        # Move the element to the end
        if idx != len(self.queue) - 1:  # Check if it's not already at the end
            # Remove the element from its current position
            self.queue.pop(idx)
            # Append it to the end
            self.queue.append(value)
            
            # Update the index map accordingly
            # All elements after `idx` must have their index decreased by 1
            for i in range(idx, len(self.queue) - 1):
                self.index_map[self.queue[i]] = i
            
            # The fetched element now has the last index
            self.index_map[value] = len(self.queue) - 1
        
        return value

```

### Explanation
1. **Initialization**: Upon initializing, we create a list `self.queue` containing elements from 1 to `n` and a dictionary `self.index_map` which stores the current index of each element for O(1) access.

2. **Fetch Method**: When we call `fetch(k)`:
   - We get the index of the k-th element using `self.index_map[k]`.
   - We retrieve the value at that index.
   - If the index is not the last one, we remove the element from its current position and append it to the end of the list. 
   - We then update the indices in our dictionary to reflect the new positions of the elements in the list.

### Complexity
- **Time Complexity**: Each `fetch` call operates in O(n) because we potentially need to adjust the indices after moving an element. 
- **Space Complexity**: O(n) for storing the list and the dictionary.

This approach correctly implements the desired functionality while being efficient in accessing and repositioning elements as required by the MRUQueue problem.

