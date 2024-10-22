# 4. Median of Two Sorted Arrays

### Problem Description 
Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return the median of the two sorted arrays.


Example 1:
Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.


Example 2:
Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.


Example 3:
Input: nums1 = [0,0], nums2 = [0,0]
Output: 0.00000

Example 4:
Input: nums1 = [], nums2 = [1]
Output: 1.00000

Example 5:
Input: nums1 = [2], nums2 = []
Output: 2.00000

Constraints:
`nums1.length == m`
`nums2.length == n`
`0 <= m <= 1000`
`0 <= n <= 1000`
`1 <= m + n <= 2000`
`-106 <= nums1[i], nums2[i] <= 106`
Follow up: The overall run time complexity should be `O(log (m+n))`.

### Solution 
 To solve the problem of finding the median of two sorted arrays, we can utilize a binary search approach for optimal performance. The median is defined as the middle value in an ordered list of numbers, and it can be calculated differently based on whether there's an odd or even number of values.

### Explanation of the Approach

1. **Understanding the median**: When we merge two sorted arrays, the median is the middle element (if the total number of elements is odd) or the average of the two middle elements (if the total number of elements is even).

2. **Binary Search**: Instead of merging the arrays which would take O(m+n) time, we will employ binary search on the smaller of the two arrays. This will give us an efficient O(log(min(m, n))) solution.

3. **Partitioning**: We maintain a partitioning of the two arrays:
   - Let `X` and `Y` be the two arrays, where `X` is the smaller array. 
   - We find a partition such that:
     - The left part of the combined array (from both arrays) contains half of the total numbers.
     - This means we can use the partitions to create conditions for the median.

4. **Conditions to Adjust Partitions**: 
   - We need to ensure that all elements in the left partition are less than or equal to all elements in the right partition.
   - We can calculate:
     - `max_left_X`, `min_right_X` for array `X`.
     - `max_left_Y`, `min_right_Y` for array `Y`.
   - Adjust the partitions based on the max and min values until the conditions are satisfied.

5. **Calculating the Median**: 
   - Once the correct partition is found, we can calculate the median based on whether the total length is odd or even.

### Python Code

Here is the Python solution that follows the steps described above:



```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # Ensure nums1 is the smaller array
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
            
        m, n = len(nums1), len(nums2)
        total_left = (m + n + 1) // 2
        
        low, high = 0, m
        
        while low <= high:
            partitionX = (low + high) // 2
            partitionY = total_left - partitionX
            
            # Edge cases for partition X
            maxLeftX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]
            minRightX = float('inf') if partitionX == m else nums1[partitionX]
            
            # Edge cases for partition Y
            maxLeftY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]
            minRightY = float('inf') if partitionY == n else nums2[partitionY]

            # Check if we have found the correct partition
            if maxLeftX <= minRightY and maxLeftY <= minRightX:
                # We have a valid partition
                if (m + n) % 2 == 0:
                    return (max(maxLeftX, maxLeftY) + min(minRightX, minRightY)) / 2
                else:
                    return max(maxLeftX, maxLeftY)
            elif maxLeftX > minRightY:
                # Move towards left in nums1
                high = partitionX - 1
            else:
                # Move towards right in nums1
                low = partitionX + 1
        
        raise ValueError("Input arrays are not sorted properly or the length constraints are violated.")


```

### Illustrating the Code
- The function begins by ensuring that `nums1` is the smaller array to minimize the time complexity.
- We perform a binary search through `nums1`, adjusting our partitions based on the comparisons between the max left and min right values of both arrays.
- Finally, once the right partition is found, we calculate and return the median based on whether the combined length of the arrays is even or odd.

This solution runs in O(log(min(m, n))) time, ensuring efficiency even for the largest inputs specified in the problem constraints.

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
 To solve the problem of merging `k` sorted linked lists, we can utilize a min-heap (or priority queue) to efficiently manage the heads of the linked lists. The main idea is to always extract the smallest element from the heap and add it to the result list. We will then move to the next node of the corresponding linked-list and continue the process until all linked lists are processed.

Here's a step-by-step breakdown of the solution:

1. **Initialization**: 
   - We will start by creating a min-heap to keep track of the smallest head of each linked list.
   - We will also create a dummy node that helps recursively build the result linked list.

2. **Heap Population**: 
   - We iterate through each linked list and push the head nodes into the min-heap. Each heap entry should store both the node value and the node itself so that we can construct the resultant linked list.

3. **Merging Process**:
   - While the min-heap is not empty, we will:
     - Extract the smallest node from the heap.
     - Add it to the result linked list.
     - If there are more nodes in the same linked list, push the next node into the heap.

4. **Return the Result**: 
   - Finally, we return the next node of the dummy node, which represents the head of the merged linked list.

Here is the finalized Python code in accordance with the LeetCode submission format:



```python
import heapq
from typing import List, Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # Min-Heap to keep the smallest head of the linked-lists
        min_heap = []
        
        # Initialize the heap with the head of each linked list
        for index, linked_list in enumerate(lists):
            if linked_list:
                # Push (value, node) tuple into the min-heap
                heapq.heappush(min_heap, (linked_list.val, index, linked_list))
        
        # Dummy node to help with the merged linked list
        dummy = ListNode(0)
        current = dummy
        
        # Process the heap until it's empty
        while min_heap:
            # Get the smallest node from heap
            val, index, node = heapq.heappop(min_heap)
            current.next = ListNode(val)  # Create a new node in the merged list
            current = current.next  # Move the pointer forward
            
            # If there's more nodes in the linked list, push the next node into the min-heap
            if node.next:
                heapq.heappush(min_heap, (node.next.val, index, node.next))
        
        # Return the head of the merged linked list
        return dummy.next

```

### Explanation of the Code:
- **Heap Operations**: We use `heapq` in Python, which provides an efficient way to maintain a min-heap. The heap operations (`heappush` and `heappop`) give us O(log k) time complexity, where `k` is the number of linked lists.
- **Node Handling**: Each time we extract a node, we also check if there is a next node in the same linked list to continue merging.
- **Edge Cases**: The implementation correctly handles edge cases like when the list of linked lists is empty or contains lists that are themselves empty.

### Complexity:
- **Time Complexity**: O(N log k), where N is the total number of nodes across all linked lists, and `k` is the number of linked lists.
- **Space Complexity**: O(k), for the min-heap storing up to `k` nodes.

This implementation efficiently merges the k sorted linked lists into one sorted linked list following the rules specified in the problem statement.

# 53. Maximum Subarray

### Problem Description 
Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.


Example 1:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.


Example 2:
Input: nums = [1]
Output: 1

Example 3:
Input: nums = [5,4,-1,7,8]
Output: 23

Constraints:
`1 <= nums.length <= 3 * 104`
`-105 <= nums[i] <= 105`
Follow up: If you have figured out the `O(n)` solution, try coding another solution using the divide and conquer approach, which is more subtle.

### Solution 
 To solve the problem of finding the contiguous subarray with the largest sum, we can utilize a well-known algorithm called Kadane's Algorithm, which runs in O(n) time complexity.

### Explanation of Kadane's Algorithm:
1. **Initialization**: We initialize two variables:
   - `max_current`: This represents the maximum sum of the subarray that ends at the current index. We start this at the first element.
   - `max_global`: This keeps track of the maximum sum we have seen so far. We start this also at the first element.

2. **Iteration**: We iterate through the array starting from the second element.
   - For each element `num` in the array:
     - We update `max_current` to be the maximum of `num` itself or `max_current + num`. This means we either start a new subarray at the current element or extend the previous subarray to include the current element.
     - We then update `max_global` to be the maximum of `max_global` and `max_current`. This keeps track of the largest sum found.

3. **Return**: After processing all elements, `max_global` contains the sum of the subarray with the largest sum.

### Code Implementation:
Here is the Python code implementing the above logic which should run directly on LeetCode:



```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # Initialize the maximum sums
        max_current = max_global = nums[0]
        
        # Loop through the array starting from index 1
        for num in nums[1:]:
            # Update max_current to be the maximum of num itself or extending the previous subarray
            max_current = max(num, max_current + num)
            # Update max_global if we found a new maximum
            max_global = max(max_global, max_current)
        
        return max_global

```

### How to Run on LeetCode:
- Open the LeetCode platform, and search for the problem titled "Maximum Subarray."
- Choose the Python language option.
- Copy and paste the code into the provided code editor on the platform.
- Make sure the `List` type hint is imported or modify the code to use standard Python lists if necessary.
- Click the submit button to see if your solution passes all the test cases.

### Follow-Up (Divide and Conquer Approach):
If you're interested in the divide and conquer approach, it involves recursively finding the maximum subarray in the left half, the right half, and across the midpoint. However, this solution is more complex and typically runs in O(n log n) time. Should you want that implementation as well, let me know!

# 169. Majority Element

### Problem Description 
Given an array `nums` of size `n`, return the majority element.

The majority element is the element that appears more than `⌊n / 2⌋` times. You may assume that the majority element always exists in the array.


Example 1:
Input: nums = [3,2,3]
Output: 3

Example 2:
Input: nums = [2,2,1,1,1,2,2]
Output: 2

Constraints:
`n == nums.length`
`1 <= n <= 5 * 104`
`-231 <= nums[i] <= 231 - 1`
Follow-up: Could you solve the problem in linear time and in `O(1)` space?

### Solution 
 To solve the problem of finding the majority element in an array, we can utilize the Boyer-Moore Voting Algorithm, which efficiently determines the majority element with a time complexity of O(n) and space complexity of O(1). 

### Explanation:
The Boyer-Moore Voting Algorithm works by maintaining a count and a candidate for the majority element. Here’s how it works:

1. **Candidate Selection**: We iterate through the array while keeping a count of how many times we see the current candidate. If the count drops to zero, we select a new candidate.
2. **Counting**: For each number:
   - If the count is zero, we set the current number as the candidate and set the count to 1.
   - If the current number is the same as the candidate, we increment the count by 1.
   - If the current number is different, we decrement the count by 1.
3. **Result**: At the end of this process, the candidate will be our majority element since we know one exists in the problem constraints.

### Python Code:
Here is how you can implement the described algorithm in Python, which can be directly executed on the LeetCode platform:



```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        candidate = None
        count = 0
        
        for num in nums:
            if count == 0:
                candidate = num
                count = 1
            elif num == candidate:
                count += 1
            else:
                count -= 1
        
        return candidate

```

### Breakdown of the Code:
1. **Class Definition**: We define a class `Solution` as required by LeetCode.
2. **Method Definition**: The method `majorityElement` takes a list of integers `nums` as input. It returns an integer, which is the majority element.
3. **Initialize Variables**:
   - `candidate`: This variable will store our potential majority element.
   - `count`: This variable tracks the count of occurrences of the candidate.
4. **Iterate Through the Array**:
   - Check if `count` is zero. If it is, set the current number as the new candidate and reset the count to 1.
   - If the current number matches the candidate, increment the count.
   - If it does not match, decrement the count.
5. **Return the Candidate**: After the loop ends, the candidate will be the majority element, which we return.

### Time and Space Complexity:
- **Time Complexity**: O(n), where n is the length of the input array. We make one pass through the array.
- **Space Complexity**: O(1) as we are using only a fixed number of variables.

This solution efficiently finds the majority element while ensuring that we meet the constraints provided in the problem statement. This makes it suitable for large arrays, up to 50,000 elements as specified.

# 215. Kth Largest Element in an Array

### Problem Description 
Given an integer array `nums` and an integer `k`, return the `kth` largest element in the array.

Note that it is the `kth` largest element in the sorted order, not the `kth` distinct element.


Example 1:
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5

Example 2:
Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4

Constraints:
`1 <= k <= nums.length <= 104`
`-104 <= nums[i] <= 104`

### Solution 
 To solve the problem of finding the `kth` largest element in an integer array `nums`, we can utilize several approaches. However, a very efficient way to do this is through the use of a min-heap, which allows us to maintain the `k` largest elements seen so far. This way, the root of the min-heap will always be the `kth` largest element when the array is fully processed.

### Explanation

1. **Understanding the Problem**: We need to find the `kth` largest element in the given list. Given the sorted nature of the task, we should recognize that the `kth` largest element corresponds to the element that has `k-1` elements larger than it.

2. **Using a Min-Heap**: By using a min-heap of size `k`, we can efficiently keep track of the `k` largest elements. The `kth` largest element will be the smallest element in this min-heap once it contains `k` elements.

3. **Algorithm**:
    - Initialize a min-heap.
    - Iterate through each number in the `nums` list.
    - Push each number into the heap until it contains more than `k` elements.
    - If the heap size exceeds `k`, pop the smallest element (root of the min-heap).
    - After processing all elements, the root of the heap will give us the `kth` largest element.

4. **Complexity**: 
    - The time complexity of this approach is O(n log k), where `n` is the number of elements in `nums` because each insertion into the heap is O(log k).
    - The space complexity is O(k) for storing the `k` largest elements in the min-heap.

### Python Code

Here's how you would implement the above logic in Python, formatted for direct use on LeetCode:



```python
import heapq

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # Create a min-heap
        min_heap = []
        
        # Iterate over the numbers in nums
        for num in nums:
            # Push the number into the heap
            heapq.heappush(min_heap, num)
            
            # If the size of the heap exceeds k, pop the smallest element
            if len(min_heap) > k:
                heapq.heappop(min_heap)
        
        # The root of the min-heap is the kth largest element
        return min_heap[0]

```

### How to Use This Code
1. You would typically place this code in the environment provided by LeetCode.
2. Ensure that the proper imports are included, particularly `heapq` and `List` from the `typing` module.
3. You can run this code in the function testing area of LeetCode by providing the input as specified in the problem description.

### Test the Function
You can run the function with sample inputs such as:



```python
# Example 1
sol = Solution()
print(sol.findKthLargest([3, 2, 1, 5, 6, 4], 2))  # Output: 5

# Example 2
print(sol.findKthLargest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4))  # Output: 4

```

This will confirm that your implementation correctly identifies the `kth` largest element for the provided examples.

# 218. The Skyline Problem

### Problem Description 
A city's skyline is the outer contour of the silhouette formed by all the buildings in that city when viewed from a distance. Given the locations and heights of all the buildings, return the skyline formed by these buildings collectively.

The geometric information of each building is given in the array `buildings` where `buildings[i] = [lefti, righti, heighti]`:
`lefti` is the x coordinate of the left edge of the `ith` building.

`righti` is the x coordinate of the right edge of the `ith` building.

`heighti` is the height of the `ith` building.

You may assume all buildings are perfect rectangles grounded on an absolutely flat surface at height `0`.

The skyline should be represented as a list of "key points" sorted by their x-coordinate in the form `[[x1,y1],[x2,y2],...]`. Each key point is the left endpoint of some horizontal segment in the skyline except the last point in the list, which always has a y-coordinate `0` and is used to mark the skyline's termination where the rightmost building ends. Any ground between the leftmost and rightmost buildings should be part of the skyline's contour.

Note: There must be no consecutive horizontal lines of equal height in the output skyline. For instance, `[...,[2 3],[4 5],[7 5],[11 5],[12 7],...]` is not acceptable; the three lines of height 5 should be merged into one in the final output as such: `[...,[2 3],[4 5],[12 7],...]`

Example 1:
Input: buildings = [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]
Output: [[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]
Explanation:
Figure A shows the buildings of the input.

Figure B shows the skyline formed by those buildings. The red points in figure B represent the key points in the output list.


Example 2:
Input: buildings = [[0,2,3],[2,5,3]]
Output: [[0,3],[5,0]]

Constraints:
`1 <= buildings.length <= 104`
`0 <= lefti < righti <= 231 - 1`
`1 <= heighti <= 231 - 1`
`buildings` is sorted by `lefti` in non-decreasing order.

### Solution 
 To solve the skyline problem, we can use a combination of a heap (priority queue) and a list to store the result. The main idea is to process all the building edges (both the left and right edges) in a sorted manner and maintain the current heights of the buildings that are "active" at any given x-coordinate. Let's break down the steps in detail.

### Step-by-step Explanation:

1. **Understand the Input and Output**: 
    - We have a list of buildings represented as `[lefti, righti, heighti]`.
    - Our output should be a list of unique key points that form the skyline.

2. **Data Structure**:
    - Use a list to hold all events (edges) where an event can be a building's left edge (which will add a height) or a right edge (which will remove a height).
    - Use a max-heap to keep track of the current maximum height efficiently.

3. **Event Creation**:
    - For each building, create two events: one for when the building starts (left edge) and one for when it ends (right edge).
    - Store these events as tuples in the form `(x-coordinate, height, type)`, where `type` is `-1` for start and `1` for end.

4. **Processing Events**:
    - Sort the events by x-coordinate. If two events have the same x-coordinate, sort by height (with higher buildings coming first for starts, and lower buildings coming first for ends).
    - Use the heap to manage the current heights of active buildings. 
    - Whenever we encounter a left edge, we add the height to the heap, and when we encounter a right edge, we remove the height.

5. **Building the Skyline**:
    - While processing events, whenever we have a new maximum height (i.e., the height changes), we add the x-coordinate and the new height to the result.

6. **Termination**:
    - Always add an endpoint (0 height) for the final skyline termination.

Here’s the implementation of the above steps:



```python
import heapq

class Solution:
    def getSkyline(self, buildings):
        # Step 1: Create events (start and end)
        events = []
        
        for left, right, height in buildings:
            events.append((left, -height))  # Start of the building
            events.append((right, height))   # End of the building
            
        # Step 2: Sort events
        events.sort()
        
        # Step 3: Use a max-heap to track the current heights
        result = []
        max_heap = [(0, float('inf'))]  # (height, end)
        
        for x, h in events:
            # If it's a start of a building, add it to the heap
            if h < 0:  # Start of a building
                heapq.heappush(max_heap, (h, x))
            else:  # End of a building
                # Remove the height from the heap; heapq does not support direct removal
                max_heap = [(height, end) for height, end in max_heap if end > x]
                heapq.heapify(max_heap)
            
            # Get current maximum height
            current_max_height = -max_heap[0][0] if max_heap else 0
            
            # If this is a new point in the skyline
            if not result or current_max_height != result[-1][1]:
                result.append([x, current_max_height])
        
        return result

```

### Explanation of the Code:
- We first gather all events in a list and sort them.
- We maintain a max-heap to keep track of building heights as we process each event.
- For each event:
  - If it's a start event, we add the height to the heap.
  - If it's an end event, we remove the corresponding height by filtering the heap and then re-heapifying it.
- When the current max height changes, it indicates a new key point in the skyline, which we then append to the result.
- Finally, we return the result after processing all events.

The solution efficiently handles the required operations and ensures the constraints are satisfied, making this method suitable for large input sizes.

# 240. Search a 2D Matrix II

### Problem Description 
Write an efficient algorithm that searches for a `target` value in an `m x n` integer `matrix`. The `matrix` has the following properties:
Integers in each row are sorted in ascending from left to right.

Integers in each column are sorted in ascending from top to bottom.


Example 1:
Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
Output: true

Example 2:
Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
Output: false

Constraints:
`m == matrix.length`
`n == matrix[i].length`
`1 <= n, m <= 300`
`-109 <= matix[i][j] <= 109`
All the integers in each row are sorted in ascending order.

All the integers in each column are sorted in ascending order.

`-109 <= target <= 109`

### Solution 
 To solve the problem of searching for a target value in an `m x n` integer matrix with particular sorted properties, we can use an efficient approach that exploits these properties.

### Overview of the Approach
1. **Starting Point**: Begin the search from the top-right corner of the matrix.
2. **Comparison**:
   - If the current element is equal to the target, return `True`.
   - If the current element is greater than the target, move left (to a smaller value).
   - If the current element is less than the target, move down (to a larger value).
3. **Termination**: If we move out of the bounds of the matrix without finding the target, return `False`.

### Complexity
- **Time Complexity**: O(m + n), where m is the number of rows and n is the number of columns. In the worst case, we traverse each row and column once.
- **Space Complexity**: O(1), since we're using a constant amount of space.

### Python Implementation
Here is the implementation following the LeetCode format:



```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]:  # Check if the matrix is empty
            return False
        
        m, n = len(matrix), len(matrix[0])
        row, col = 0, n - 1  # Start from top-right corner
        
        while row < m and col >= 0:
            current = matrix[row][col]
            if current == target:
                return True
            elif current > target:
                col -= 1  # Move left
            else:
                row += 1  # Move down
        
        return False  # Target not found

```

### Explanation of the Code
1. **Input Validation**: We first check if the `matrix` is empty. If it is, we immediately return `False`.
2. **Initialize Variables**: We determine the dimensions of the matrix (`m` for rows and `n` for columns) and set our starting position at the top-right corner of the matrix (`row = 0` and `col = n - 1`).
3. **Search Loop**: We employ a `while` loop that continues as long as `row` is in bounds (less than `m`) and `col` is in bounds (greater than or equal to `0`):
   - Check the current element at `matrix[row][col]`.
   - If it matches the `target`, we return `True`.
   - If it is greater than the `target`, we move left by decrementing the `col`.
   - If it is less than the `target`, we move down by incrementing the `row`.
4. **Final Check**: If we exit the loop without having found the target, we return `False`.

### Conclusion
This implementation is efficient and leverages the matrix's properties to minimize the number of comparisons needed to find the target element. This approach is particularly suitable for the constraints provided in the problem, ensuring that it runs efficiently even for the largest matrix sizes.

# 241. Different Ways to Add Parentheses

### Problem Description 
Given a string `expression` of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. You may return the answer in any order.


Example 1:
Input: expression = "2-1-1"
Output: [0,2]
Explanation:
((2-1)-1) = 0 
(2-(1-1)) = 2

Example 2:
Input: expression = "2*3-4*5"
Output: [-34,-14,-10,-10,10]
Explanation:
(2*(3-(4*5))) = -34 
((2*3)-(4*5)) = -14 
((2*(3-4))*5) = -10 
(2*((3-4)*5)) = -10 
(((2*3)-4)*5) = 10

Constraints:
`1 <= expression.length <= 20`
`expression` consists of digits and the operator `'+'`, `'-'`, and `'*'`.

### Solution 
 To solve the problem, we need to compute all possible outcomes from the given expression by exploring different ways to group numbers and operators. This is a classic problem that can be effectively solved using recursion and memoization to avoid redundant calculations.

### Approach

1. **Recursive Function**: We create a recursive function that processes the expression. The function will:
   - Traverse the expression looking for operators.
   - For every operator found, it splits the expression into two parts: the left part (before the operator) and the right part (after the operator).
   - It then recursively computes all possible results for both parts.
   - Combines the results from both parts by applying the operator.
   
2. **Memoization**: Since the recursive function might be called with the same substring multiple times (due to overlapping subproblems), we use a dictionary to cache results for already computed expressions. This significantly reduces the computational overhead.

3. **Base Case**: If the current expression contains only a number (no operators), we return that number as the only result.

4. **Combining Results**: When summing the results of the left and the right parts, we will consider each operator and compute the combined results.

### Implementation

Here's the Python implementation that adheres to the LeetCode format:



```python
from typing import List

class Solution:
    def diffWaysToCompute(self, expression: str) -> List[int]:
        memo = {}

        def compute(expr):
            if expr in memo:
                return memo[expr]
            if expr.isdigit():  # Base case: the expression is a number
                return [int(expr)]
            
            result = []
            for i in range(len(expr)):
                if expr[i] in "+-*":  # Check if the character is an operator
                    left = compute(expr[:i])  # Compute left side
                    right = compute(expr[i + 1:])  # Compute right side
                    # Combine results
                    for l in left:
                        for r in right:
                            if expr[i] == '+':
                                result.append(l + r)
                            elif expr[i] == '-':
                                result.append(l - r)
                            elif expr[i] == '*':
                                result.append(l * r)

            memo[expr] = result  # Store the result in memo
            return result
        
        return compute(expression)

# Example usage:
sol = Solution()
print(sol.diffWaysToCompute("2-1-1"))  # Output: [0, 2]
print(sol.diffWaysToCompute("2*3-4*5"))  # Output: [-34, -14, -10, -10, 10]

```

### Explanation of the Code:

1. **Function Definition**: The main function is `diffWaysToCompute`, which takes in the expression as a string and initializes a memoization dictionary.

2. **Inner Compute Function**: The inner function `compute` performs the recursive logic:
   - It checks if the expression is already computed and stored in `memo`. If so, it returns that result to avoid recomputation.
   - If the expression consists only of digits (i.e., no operators), it converts it to an integer and returns it as a single-item list.
   - The main loop iterates through each character in the expression:
     - When it finds an operator, it recursively computes the possible results for the left and right sub-expressions.
     - It then combines the results according to the current operator and appends the final results to the list.

3. **Memoization**: The results of computed expressions are stored to prevent recalculation, significantly optimizing performance.

4. **Execution**: The function finally returns all possible results as a list.

### Complexity

The time complexity of this algorithm can be exponential in the worst case due to the number of ways to split and compute expressions. However, memoization helps greatly reduce repeated calculations, making it feasible for the given constraint \(1 \leq \text{expression.length} \leq 20\).

# 282. Expression Add Operators

### Problem Description 
Given a string `num` that contains only digits and an integer `target`, return all possibilities to add the binary operators `'+'`, `'-'`, or `'*'` between the digits of `num` so that the resultant expression evaluates to the `target` value.


Example 1:
Input: num = "123", target = 6
Output: ["1*2*3","1+2+3"]

Example 2:
Input: num = "232", target = 8
Output: ["2*3+2","2+3*2"]

Example 3:
Input: num = "105", target = 5
Output: ["1*0+5","10-5"]

Example 4:
Input: num = "00", target = 0
Output: ["0*0","0+0","0-0"]

Example 5:
Input: num = "3456237490", target = 9191
Output: []

Constraints:
`1 <= num.length <= 10`
`num` consists of only digits.

`-231 <= target <= 231 - 1`

### Solution 
 To solve the problem of generating all possible expressions by adding the binary operators `'+'`, `'-'`, or `'*'` between the digits in the provided string `num`, we can employ a backtracking approach. This method allows us to explore all possible ways to place the operators in the string and evaluate the expressions to check if they equal the target value.

### Key Steps in the Solution:
1. **Exploration of Expressions**: We will explore every possible way of placing operators between the digits using recursive backtracking.
2. **Evaluation of Expressions**: While constructing the expressions, we will also maintain a running total and compute the value while considering operator precedence, particularly for multiplication, which has a higher precedence than addition and subtraction.
3. **Base Case**: When we've placed all operators and reached the end of the string, we will check if the evaluated expression equals the target.
4. **Recursive Function**: A helper function to explore choices will take the current index in the string, the current expression being constructed, the current value for evaluation, and the last value (used to handle operator precedence).

### Python Code Implementation:
Below is the Python code that follows the LeetCode solution format, including detailed comments for clarity.



```python
class Solution:
    def addOperators(self, num: str, target: int):
        def backtrack(index, current_expr, current_value, last_value):
            # If we have reached the end of the string and current_value equals target
            if index == len(num):
                if current_value == target:
                    results.append(current_expr)
                return
            
            for i in range(index, len(num)):
                # Avoid numbers with leading zeros
                if i > index and num[index] == '0':
                    break
                
                current_num = num[index:i + 1]
                num_value = int(current_num)

                # Adding operator '+'
                backtrack(i + 1, current_expr + '+' + current_num, current_value + num_value, num_value)
                # Adding operator '-'
                backtrack(i + 1, current_expr + '-' + current_num, current_value - num_value, -num_value)
                # Adding operator '*'
                backtrack(i + 1, current_expr + '*' + current_num,
                          current_value - last_value + last_value * num_value, last_value * num_value)

            # Base case automatically handled when index == len(num)
        
        results = []
        for i in range(len(num)):
            # Avoid leading zeros
            if i > 0 and num[0] == '0':
                break
            
            first_num = num[:i + 1]
            backtrack(i + 1, first_num, int(first_num), int(first_num))
        
        return results

```

### Explanation of the Code:
1. **Class Definition**: We define a class `Solution` which contains the main method `addOperators`.
2. **Backtrack Function**: This function takes the current index, the current expression string, the current total evaluated value, and the last evaluated value to handle operator precedence. 
3. **Loop through Digits**: We use a for loop to try every possible position in the string for placing operators, ensuring we skip leading zeros by checking if the number's first character is zero.
4. **Expressions Evaluation**:
   - For addition (`+`), we add the current number to the total.
   - For subtraction (`-`), we subtract it from the total.
   - For multiplication (`*`), we adjust the total to consider the last number with multiplication.
5. **Base Case Check**: When we reach the end of `num`, we check if the evaluated expression matches the target, adding it to the results list if it does.

### Edge Cases:
- Handling leading zeros correctly, as seen in the loops.
- Addressing various combinations of operators and ensuring correct operator precedence during evaluation.

This code can be submitted directly to LeetCode and should work correctly given the constraints provided in the problem statement.

# 312. Burst Balloons

### Problem Description 
You are given `n` balloons, indexed from `0` to `n - 1`. Each balloon is painted with a number on it represented by an array `nums`. You are asked to burst all the balloons.

If you burst the `ith` balloon, you will get `nums[i - 1] * nums[i] * nums[i + 1]` coins. If `i - 1` or `i + 1` goes out of bounds of the array, then treat it as if there is a balloon with a `1` painted on it.

Return the maximum coins you can collect by bursting the balloons wisely.


Example 1:
Input: nums = [3,1,5,8]
Output: 167
Explanation:
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167

Example 2:
Input: nums = [1,5]
Output: 10

Constraints:
`n == nums.length`
`1 <= n <= 500`
`0 <= nums[i] <= 100`

### Solution 
 To solve the problem of bursting balloons for maximum coins collected, we can use a dynamic programming approach. The idea is to break down the problem into smaller subproblems by considering the maximum coins we can collect by bursting balloons in a certain range.

### Detailed Explanation

Let's break down the steps involved in solving the problem:

1. **Understanding the Points of Bursting**:
   When you burst a balloon (at index `i`), you earn coins based on the balloons adjacent to it:
   - If you burst balloon `i`, you earn `nums[i - 1] * nums[i] * nums[i + 1]` coins, where `nums[-1]` and `nums[n]` are treated as `1` if they go out of bounds.

2. **Dynamic Programming Setup**:
   - We'll create a DP array `dp` where `dp[l][r]` represents the maximum coins you can collect by bursting all the balloons in the range from `l` to `r`.
   - We will iterate through all possible lengths of intervals and compute the maximum coins for each interval.

3. **Transitional Formula**:
   - For each subproblem defined by the range `[l, r]`, we will consider bursting each balloon `k` within this range. The coins collected from bursting balloon `k` can be calculated, and we'll add the maximum coins achievable from the two remaining subproblems:
     
```
     dp[l][r] = max(dp[l][r], dp[l][k-1] + dp[k+1][r] + nums[l-1]*nums[k]*nums[r+1])
     
```
   - Here `nums[l-1]` and `nums[r+1]` are the values that would be treated as `1` if out of boundaries.

4. **Base Case**:
   - If there are no balloons to burst between indices, the collected coins will be `0`.

5. **Implementation**:
   We'll initiate our `dp` table and start filling it based on the lengths of the intervals. Finally, the maximum coins you can collect will be in `dp[0][n-1]`.

Here is the implementation in Python:



```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        # Add boundaries (1s) to the original list
        nums = [1] + nums + [1]
        n = len(nums)
        dp = [[0] * n for _ in range(n)]
        
        for length in range(1, n - 1):  # length of the interval
            for left in range(1, n - length):  # left boundary of the interval
                right = left + length - 1  # right boundary of the interval
                # Try bursting each balloon in the range [left, right]
                for k in range(left, right + 1):
                    dp[left][right] = max(dp[left][right], 
                                           dp[left][k - 1] + dp[k + 1][right] + nums[left - 1] * nums[k] * nums[right + 1])
        
        return dp[1][n - 2]  # do not include the added boundaries

```

### Conclusion
The above code efficiently calculates the maximum coins that can be collected by strategically bursting balloons using dynamic programming principles. The complexity of the solution is \(O(n^3)\), which is feasible within the given constraints of \(1 \leq n \leq 500\).

# 315. Count of Smaller Numbers After Self

### Problem Description 
You are given an integer array `nums` and you have to return a new `counts` array. The `counts` array has the property where `counts[i]` is the number of smaller elements to the right of `nums[i]`.


Example 1:
Input: nums = [5,2,6,1]
Output: [2,1,1,0]
Explanation:
To the right of 5 there are 2 smaller elements (2 and 1).

To the right of 2 there is only 1 smaller element (1).

To the right of 6 there is 1 smaller element (1).

To the right of 1 there is 0 smaller element.


Example 2:
Input: nums = [-1]
Output: [0]

Example 3:
Input: nums = [-1,-1]
Output: [0,0]

Constraints:
`1 <= nums.length <= 105`
`-104 <= nums[i] <= 104`

### Solution 
 To tackle the problem of finding the number of smaller elements to the right for each element in the given list `nums`, we can use a more efficient approach than the brute-force method, which would typically involve nested loops and result in a time complexity of \(O(n^2)\). 

Instead, we will use a modified merge sort algorithm to count the smaller elements in \(O(n \log n)\) time complexity. This approach works by sorting the elements while simultaneously counting the number of elements that are smaller and to the right of the current element.

### Approach Overview:
1. **Coordinate Compression:** Since the elements can range from -10,000 to 10,000, we can map these values to a compressed range. This is helpful when we want to implement tree data structures for counting.

2. **Merge Sort:** We will implement the merge sort algorithm but modify it to count how many numbers are smaller while merging.

3. **Count Array:** We will maintain an array `counts` to store the result for each element.


### Implementation:



```python
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        if not nums:
            return []

        n = len(nums)
        counts = [0] * n
        indices = list(range(n))

        def merge_sort(enum):
            mid = len(enum) // 2
            if mid:
                left = enum[:mid]
                right = enum[mid:]
                merge_sort(left)
                merge_sort(right)

                l = r = 0
                right_count = 0

                while l < len(left) and r < len(right):
                    if left[l][1] > right[r][1]:  # Count numbers in left greater than number in right
                        right_count += 1
                        r += 1
                    else:
                        counts[left[l][0]] += right_count  # Update count
                        l += 1
                
                # Collect the remaining items
                while l < len(left):
                    counts[left[l][0]] += right_count
                    l += 1
                while r < len(right):
                    r += 1

                # Merge back
                for i in range(len(enum)):
                    if i < len(left):
                        enum[i] = left[i]
                    else:
                        enum[i] = right[i - len(left)]

        enum = list(enumerate(nums))
        merge_sort(enum)
        return counts

```

### Explanation of the Code:
1. **Initialization**: 
   - We create a list `counts` to store the results with the same length as `nums`.
   - We create an `indices` list to maintain the original indices of the numbers alongside their values for sorting.

2. **Merge Sort with Counting**:
   - The `merge_sort` function takes a list of tuples `(index, value)`, where `index` is the original index of the number, and `value` is the number itself.
   - We recursively split the array until we reach the base case where the subarrays have length 1.
   - During the merging step, we compare elements in the left and right halves. If an element in the left half is greater than an element in the right half, it means all subsequent elements in the left half (to the right of the current left element) will also be greater. We count how many such elements exist and adjust their counts accordingly.

3. **Final Result**: 
   - The `counts` array is returned, reflecting the number of smaller elements to the right for each element in the input array.

### Complexity:
- **Time Complexity**: \(O(n \log n)\) due to the merge sort.
- **Space Complexity**: \(O(n)\) for the counts array and the enumeration.

This method is efficient and works well within the provided constraints, ensuring that we can handle up to \(10^5\) elements smoothly.

# 327. Count of Range Sum

### Problem Description 
Given an integer array `nums` and two integers `lower` and `upper`, return the number of range sums that lie in `[lower, upper]` inclusive.

Range sum `S(i, j)` is defined as the sum of the elements in `nums` between indices `i` and `j` inclusive, where `i <= j`.


Example 1:
Input: nums = [-2,5,-1], lower = -2, upper = 2
Output: 3
Explanation: The three ranges are: [0,0], [2,2], and [0,2] and their respective sums are: -2, -1, 2.


Example 2:
Input: nums = [0], lower = 0, upper = 0
Output: 1

Constraints:
`1 <= nums.length <= 104`
`-231 <= nums[i] <= 231 - 1`
`-3 * 104 <= lower <= upper <= 3 * 104`
Follow up: A naive algorithm of `O(n2)` is trivial, Could you do better than that?

### Solution 
 To solve the problem of finding the number of range sums that lie within a specified range `[lower, upper]`, we can use a more efficient algorithm than the naive O(n²) approach. The optimal way to approach this is through a combination of prefix sums and a modified merge sort. This way, we can effectively count the number of valid range sums in O(n log n) time.

### Explanation:

1. **Prefix Sum**: 
   We first calculate the prefix sums such that `prefix[i] = nums[0] + nums[1] + ... + nums[i]`. The range sum `S(i, j)` can be expressed in terms of prefix sums as:
   \[
   S(i, j) = prefix[j] - prefix[i-1]
   \]
   This indicates that the sum of the subarray from index `i` to `j` can be derived using prefix sums.

2. **Counting Ranges**:
   We need to count how many of these prefix sums fall within the range `[lower, upper]`.
   We can rewrite our condition:
   \[
   lower \leq prefix[j] - prefix[i-1] \leq upper
   \]
   Which simplifies to two conditions:
   \[
   prefix[j] - upper \leq prefix[i-1] \leq prefix[j] - lower
   \]

3. **Using Merge Sort**:
   We will perform a modified merge sort to:
   - Sort the prefix sums.
   - Count how many prefix sums fall in the desired range using binary search.

The algorithm will be implemented as follows:

### Code:



```python
class Solution:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        # This function will perform the merge sort and count the range sums
        def merge_count_and_sort(pfx_sums, left, mid, right):
            count = 0
            j = mid + 1
            # Count the number of valid range sums
            for i in range(left, mid + 1):
                while j <= right and pfx_sums[j] - pfx_sums[i] < lower:
                    j += 1
                k = j
                while k <= right and pfx_sums[k] - pfx_sums[i] <= upper:
                    k += 1
                count += k - j
            
            # Sort the prefix sums
            pfx_sums[left:right + 1] = sorted(pfx_sums[left:right + 1])
            return count

        def count_range_sums(pfx_sums, left, right):
            if left >= right:
                return 0
            mid = (left + right) // 2
            # Count on the left and right side plus the cross range count
            count = count_range_sums(pfx_sums, left, mid) + count_range_sums(pfx_sums, mid + 1, right)
            count += merge_count_and_sort(pfx_sums, left, mid, right)
            return count
        
        # Compute prefix sums
        pfx_sums = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            pfx_sums[i + 1] = pfx_sums[i] + nums[i]
        
        # Count the number of range sums
        return count_range_sums(pfx_sums, 0, len(pfx_sums) - 1)

```

### Detailed Explanation of the Code:
- We define an inner function `merge_count_and_sort` that:
  - Counts how many `prefix` sums fall into the desired range when dividing around the midpoint `mid`.
  - Merges and sorts the prefix sums in that range.
  
- We have another recursive function `count_range_sums` that:
  - Calls itself to divide the problem into smaller parts and count the valid range sums recursively.

- Finally, we compute the prefix sums and call the counting function with the initial range.

### Performance:
This approach has a time complexity of O(n log n) due to the merge sort logic combined with the counting process, which is efficient for the input limits provided in the problem.

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
 To solve the problem of finding the length of the longest substring in which each character appears at least `k` times, we can utilize a divide-and-conquer approach. The main idea is to recursively divide the string into parts based on characters that do not meet the frequency requirements.

### Explanation of the Approach:

1. **Character Frequency Count**: First, we will count the frequency of each character in the string.

2. **Divide and Conquer**: We will check each character's frequency:
   - If a character appears less than `k` times, it cannot be part of the desired substring. So, we will split the string into substrings based on the positions of this character and recursively evaluate those substrings.
   - If a character meets the frequency requirement, then we can consider that entire substring as a candidate for the longest substring that meets the criteria.

3. **Base Cases**:
   - If the string is empty, return `0`.
   - If all characters in the string meet the frequency requirement, return the length of the string.

4. **Recursion**: For each split, recursively apply the same logic to both parts of the split.

### Code Implementation:

Here’s the code implementing the above logic structured in the LeetCode format:



```python
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        # Function to find the longest valid substring
        def get_longest_substring(start: int, end: int) -> int:
            if end - start < k:  # When length of substring is less than k
                return 0
            
            # Count the frequency of each character in the current substring
            char_count = {}
            for i in range(start, end):
                char_count[s[i]] = char_count.get(s[i], 0) + 1
            
            # Check for any character that does not meet the frequency requirement
            for mid in range(start, end):
                if char_count[s[mid]] < k:
                    # Split and recursively solve for the left and right substrings
                    next_start = mid + 1
                    # Find the longest valid substring in both halves
                    while next_start < end and char_count[s[next_start]] < k:
                        next_start += 1
                    return max(get_longest_substring(start, mid), 
                               get_longest_substring(next_start, end))
            
            # If all characters meet the requirement, return the length of the substring
            return end - start
        
        # Start the recursive process from the entire string
        return get_longest_substring(0, len(s))

```

### Explanation of Code:
- **Class Definition**: We define a class `Solution` which is a common structure for LeetCode problems.
- **Function `longestSubstring`**: This is the main function that takes `s` and `k` as inputs and initiates the recursive function.
- **Inner Function `get_longest_substring`**:
  - Base case checks for the substring length being less than `k`.
  - Constructs a frequency dictionary for characters in the current substring.
  - Determines if any character has insufficient frequency (`< k`). If so, split the substring and recursively check the left and right parts.
  - If all characters meet the requirements, return the total length of the substring.
  
This code will accurately and efficiently find the length of the longest substring where every character appears at least `k` times, adhering to the constraints provided in the problem.

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
 To solve the problem of converting a Binary Search Tree (BST) into a sorted circular doubly-linked list, we will use a depth-first search (DFS) traversal approach. The main idea is to traverse the tree in an in-order manner while adjusting the pointers of each node to create the desired doubly-linked list structure.

### Explanation:
1. **In-Order Traversal**: In a BST, an in-order traversal yields the nodes in sorted order. We will traverse the BST using an in-order approach to capture the node values in order while linking them to form a doubly-linked list.

2. **Node Linking**: For each node visited during the traversal:
    - Set the `right` pointer of the current node to point to the next node.
    - Set the `left` pointer of the next node to point back to the current node.
    
3. **Circular Linking**: After traversing the nodes, we will link the last node back to the first node to make the list circular.

4. **Return the Smallest Element**: The smallest element in the BST is the leftmost node, which will eventually become the head of our circular doubly-linked list.

### Implementation:
Here’s how we can implement the above logic in Python according to the LeetCode submission format:



```python
# Definition for a Node.
class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def treeToDoublyList(self, root: Node) -> Node:
        if not root:
            return None
        
        # Helper function to perform in-order traversal and link nodes
        def in_order_traversal(node):
            nonlocal last, head
            
            if not node:
                return
            
            # Visit left subtree
            in_order_traversal(node.left)
            
            # Link the previous node (last) and current node (node)
            if last:
                last.right = node  # Set the right pointer
                node.left = last   # Set the left pointer
            else:
                head = node  # Mark the head for the first node
            
            # Update last to the current node
            last = node
            
            # Visit right subtree
            in_order_traversal(node.right)
        
        head = None  # To keep track of head of the doubly linked list
        last = None  # To keep track of the last processed node
        in_order_traversal(root)
        
        # After the traversal, connect the head and last to make it circular
        last.right = head
        head.left = last
        
        return head

```

### Complexities:
- **Time Complexity**: O(n), where n is the number of nodes in the BST, since we visit each node exactly once.
- **Space Complexity**: O(h), where h is the height of the tree. This is due to the stack space utilized during the recursive in-order traversal. In the worst case (i.e. a skewed tree), this could be O(n), but for balanced trees, it would be O(log n).

### Example Cases:
1. For `root = [4,2,5,1,3]`, the output will be the circular doubly linked list starting from 1 -> 2 -> 3 -> 4 -> 5 -> back to 1.
2. For an empty tree, `root = []`, the output is `[]`.
3. For `root = [1]`, the output will simply be `1`.

This code is written in a way that it can be directly copied into the LeetCode platform under the provided problem statement and should work correctly to solve the problem as required.

# 493. Reverse Pairs

### Problem Description 
Given an array `nums`, we call `(i, j)` an important reverse pair if `i < j` and `nums[i] > 2*nums[j]`.

You need to return the number of important reverse pairs in the given array.


Example1:
Input: [1,3,2,3,1]
Output: 2

Example2:
Input: [2,4,3,5,1]
Output: 3
Note:
The length of the given array will not exceed `50,000`.

All the numbers in the input array are in the range of 32-bit integer.

### Solution 
 To solve the problem of counting important reverse pairs in the array, we can utilize a modified merge sort algorithm. The naive approach would involve using a nested loop to check each pair `(i, j)` for the condition `nums[i] > 2 * nums[j]`, which would lead to an O(n^2) time complexity. However, such an approach is infeasible for larger arrays, especially given the constraint that the length of the array can be as high as 50,000.

Instead, we will take advantage of the merge sort's divide-and-conquer approach. This allows us to count the important reverse pairs while simultaneously sorting the array, thus achieving a time complexity of O(n log n).

### Detailed Explanation:

1. **Merge Sort**: We will implement a merge sort algorithm, where in the merge step, we will count how many important reverse pairs can be formed with elements from the left and right halves of the array.

2. **Count Important Reverse Pairs**: While merging two sorted halves, we will start comparing elements from the left half (`left[i]`) and the right half (`right[j]`). For any element `left[i]`, we need to find how many elements `right[j]` satisfy the condition `left[i] > 2 * right[j]`. This can be done efficiently since both halves are sorted.

3. **Merge Step**: During the merge step:
   - Maintain a count of valid pairs for every element in the left half.
   - After counting, merge the two halves back together while ensuring the entire array remains sorted.

4. **Final Count**: The total number of pairs counted during the merge steps will provide the final answer.

Here is the implementation of the described approach:



```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        def merge_sort_and_count(arr, left, right):
            if left >= right:
                return 0
            
            mid = (left + right) // 2
            count = merge_sort_and_count(arr, left, mid) + merge_sort_and_count(arr, mid + 1, right)
            
            # Count important reverse pairs
            j = mid + 1
            for i in range(left, mid + 1):
                while j <= right and arr[i] > 2 * arr[j]:
                    j += 1
                count += (j - (mid + 1))  # All elements from mid+1 to j-1 are valid
            
            # Merge step
            sorted_arr = []
            i, j = left, mid + 1
            
            while i <= mid and j <= right:
                if arr[i] <= arr[j]:
                    sorted_arr.append(arr[i])
                    i += 1
                else:
                    sorted_arr.append(arr[j])
                    j += 1
            
            while i <= mid:
                sorted_arr.append(arr[i])
                i += 1
            
            while j <= right:
                sorted_arr.append(arr[j])
                j += 1
            
            for i in range(len(sorted_arr)):
                arr[left + i] = sorted_arr[i]
            
            return count
            
        return merge_sort_and_count(nums, 0, len(nums) - 1)

```

### How It Works
- The `reversePairs` function starts the merge sort with counting by calling `merge_sort_and_count()`.
- The recursive calls break down the problem into smaller problems until we reach single elements (which have no pairs).
- During the merging, we count pairs for the left and right halves of the array using a two-pointer approach.
- Finally, we replace the original array's elements with the sorted elements.

### Example Walkthrough
For the input `[2, 4, 3, 5, 1]`:
1. Split: `[2, 4]` and `[3, 5, 1]`
2. Further split and count pairs during merging on the way back up:
   - Find pairs during merge of `[2, 4]` and `[3, 5]`, and later on, counting pairs with `1` would yield the total count of reverse pairs as `3` for this example.

This solution is efficient and can handle the upper limits of the input size constraint.



# 514. Freedom Trail

### Problem Description 
In the video game Fallout 4, the quest "Road to Freedom" requires players to reach a metal dial called the "Freedom Trail Ring", and use the dial to spell a specific keyword in order to open the door.

Given a string ring, which represents the code engraved on the outer ring and another string key, which represents the keyword needs to be spelled. You need to find the minimum number of steps in order to spell all the characters in the keyword.

Initially, the first character of the ring is aligned at 12:00 direction. You need to spell all the characters in the string key one by one by rotating the ring clockwise or anticlockwise to make each character of the string key aligned at 12:00 direction and then by pressing the center button.

At the stage of rotating the ring to spell the key character key[i]:
You can rotate the ring clockwise or anticlockwise one place, which counts as 1 step. The final purpose of the rotation is to align one of the string ring's characters at the 12:00 direction, where this character must equal to the character key[i].

If the character key[i] has been aligned at the 12:00 direction, you need to press the center button to spell, which also counts as 1 step. After the pressing, you could begin to spell the next character in the key (next stage), otherwise, you've finished all the spelling.


Example:
Input: ring = "godding", key = "gd"
Output: 4
Explanation:
For the first key character 'g', since it is already in place, we just need 1 step to spell this character. 
For the second key character 'd', we need to rotate the ring "godding" anticlockwise by two steps to make it become "ddinggo".

Also, we need 1 more step for spelling.

So the final output is 4.

Note:
Length of both ring and key will be in range 1 to 100.

There are only lowercase letters in both strings and might be some duplcate characters in both strings.

It's guaranteed that string key could always be spelled by rotating the string ring.

### Solution 
 To solve the problem of finding the minimum number of steps needed to spell a given keyword using a rotating ring, we can apply dynamic programming (DP). The idea is to maintain a DP table where each entry corresponds to the minimum steps needed to spell the keyword up to a certain character, starting from a specific position on the ring.

### Detailed Explanation

1. **Understanding the Problem**:
   - We have a string `ring` that can be rotated either clockwise or counterclockwise.
   - We have a string `key` that represents the word we want to spell, character by character.
   - For each character in `key`, we need to rotate the ring to align the desired character at the 12:00 position, and then press a button to "spell" it.

2. **Dynamic Programming Approach**:
   - We'll create a DP table where `dp[i][j]` represents the minimum steps required to spell the first `i` characters of the `key` starting from the position of the `j`th character of `ring`.
   - There are a few key actions for each character in `key`:
     - Calculate the distance (steps) to rotate the `ring` to bring the target character of `key` to the 12:00 position.
     - Add a step for pressing the button.
   - We'll look at all positions in the `ring` that match the current character in `key`.

3. **Transition**:
   - For each character in `key`, we calculate the positions of all occurrences of that character in `ring`.
   - For each position in `ring` that matches the current character in `key`, we determine the minimum steps from all possible previous positions.

4. **Initialization**:
   - Start from the initial position (0th index of `ring`) and spell the first character of `key`.

5. **Final Result**:
   - After processing all characters in `key`, the answer will be the minimum value in the last row of the DP table.

### Python Code
Here’s the Python code that implements the above logic:



```python
class Solution:
    def findRotateSteps(self, ring: str, key: str) -> int:
        from collections import defaultdict
        
        n = len(ring)
        m = len(key)
        
        # Create a dictionary to store the indices of characters in ring
        char_indices = defaultdict(list)
        for i in range(n):
            char_indices[ring[i]].append(i)
        
        # DP table where dp[i] holds the minimum steps to spell key[0..i] with ending at ring position
        dp = [[float('inf')] * n for _ in range(m + 1)]
        
        # Base case
        for index in char_indices[key[0]]:
            dp[1][index] = min(dp[1][index], abs(index) + 1)
        
        # Fill the DP table
        for i in range(2, m + 1):
            for index in char_indices[key[i - 1]]:
                for prev_index in char_indices[key[i - 2]]:
                    steps_to_rotate = min(abs(index - prev_index), n - abs(index - prev_index))
                    dp[i][index] = min(dp[i][index], dp[i - 1][prev_index] + steps_to_rotate + 1)
        
        # Extract the minimum steps required to spell the complete key
        return min(dp[m])

# Example usage:
# sol = Solution()
# print(sol.findRotateSteps("godding", "gd"))  # Output: 4

```

### Explanation of the Code:
1. **Data Structures**:
   - We use a list of lists to store the DP table and a dictionary to map each character in `ring` to its indices.
2. **Base Case**:
   - Initialize the steps required to spell the first character of `key` from all possible positions in `ring`.
3. **DP Population**:
   - Iteratively fill in the DP table using the rules outlined for transition.
4. **Final Return**:
   - We compute the final answer by extracting the minimum from the last entry in the DP table.

With this approach, we efficiently compute the minimum steps needed to spell the entire `key` using the given `ring`.

# 932. Beautiful Array

### Problem Description 
For some fixed `N`, an array `A` is beautiful if it is a permutation of the integers `1, 2, ..., N`, such that:
For every `i < j`, there is no `k` with `i < k < j` such that `A[k] * 2 = A[i] + A[j]`.

Given `N`, return any beautiful array `A`.  (It is guaranteed that one exists.)

Example 1:
Input: 4
Output: [2,1,4,3]

Example 2:
Input: 5
Output: [3,1,2,5,4]
Note:
`1 <= N <= 1000`

### Solution 
 To solve the problem of generating a beautiful array according to the given conditions, let's first take a closer look at what a beautiful array is. The requirement states that for every pair of indices \(i < j\), there should be no index \(k\) such that \(A[k] * 2 = A[i] + A[j]\). This means that the sum of the elements at indices \(i\) and \(j\) should not be twice the value of any element that lies between them in the array.

### Approach to Solution

**Key Insight**: 
An effective approach is to use the concept of separating the numbers based on their parity (odd or even). By doing this, we can ensure that the condition will hold. Here’s a breakdown of the reason this works:

1. **Separating Odds and Evens**:
   - If we create one part of the array with all odd numbers and another part with all even numbers, we guarantee that there won’t be any \(k\) such that \(A[k] * 2 = A[i] + A[j]\) for \(i < j\) where both \(A[i]\) and \(A[j]\) are taken from the same parity (either both odd or both even).

2. **Constructing the Array**:
   - First, generate the odd numbers from \(1\) to \(N\).
   - Next, generate the even numbers from \(1\) to \(N\).
   - Concatenate the list of odd numbers with the list of even numbers.

3. **Complexity**: 
   - The time complexity for this approach is \(O(N)\) since we are simply iterating through the numbers from \(1\) to \(N\) to collect odds and evens. The space complexity is also \(O(N)\) because we are storing \(N\) elements.

Now, let's see how we can implement this in Python.

### Python Code Implementation

The following code fulfills the requirement as stated in the problem:



```python
class Solution:
    def beautifulArray(self, N: int) -> List[int]:
        odds = []
        evens = []
        
        # Collect odd numbers
        for i in range(1, N + 1, 2):
            odds.append(i)
        
        # Collect even numbers
        for i in range(2, N + 1, 2):
            evens.append(i)
        
        # The beautiful array can be formed by combining odds then evens
        return odds + evens

# Example usage:
# sol = Solution()
# print(sol.beautifulArray(4))  # Outputs: [1, 3, 2, 4] or any other valid beautiful array
# print(sol.beautifulArray(5))  # Outputs: [1, 3, 5, 2, 4] or any other valid beautiful array

```

### Explanation of Code:

1. `class Solution`: This defines a solution class which is typical in LeetCode problems.
2. `def beautifulArray(self, N: int) -> List[int]`: This is the method that will return the beautiful array. It takes `N` as input.
3. `odds` and `evens`: Two lists to hold odd and even integers.
4. The `for` loops: The first loop collects all odd numbers from `1` to `N` in the `odds` list, while the second loop collects all even numbers from `2` to `N`.
5. `return odds + evens`: Finally, we return the concatenation of the two lists which forms a valid beautiful array.

This code can be directly run on the LeetCode website under the specified problem, and it should pass all test cases as described.

# 973. K Closest Points to Origin

### Problem Description 
Given an array of `points` where `points[i] = [xi, yi]` represents a point on the X-Y plane and an integer `k`, return the `k` closest points to the origin `(0, 0)`.

The distance between two points on the X-Y plane is the Euclidean distance (i.e., `√(x1 - x2)2 + (y1 - y2)2`).

You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).


Example 1:
Input: points = [[1,3],[-2,2]], k = 1
Output: [[-2,2]]
Explanation:
The distance between (1, 3) and the origin is sqrt(10).

The distance between (-2, 2) and the origin is sqrt(8).

Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.

We only want the closest k = 1 points from the origin, so the answer is just [[-2,2]].


Example 2:
Input: points = [[3,3],[5,-1],[-2,4]], k = 2
Output: [[3,3],[-2,4]]
Explanation: The answer [[-2,4],[3,3]] would also be accepted.


Constraints:
`1 <= k <= points.length <= 104`
`-104 < xi, yi < 104`

### Solution 
 To solve the problem of finding the `k` closest points to the origin on the X-Y plane, we can follow these steps in our implementation:

1. **Calculate the Euclidean Distance**: While we don't need the precise distance (since we're only comparing distances), we can compare the squared distances to avoid unnecessary computation of the square root. The squared distance for a point `(x, y)` can be calculated as `d² = x² + y²`.

2. **Use a Min-Heap**: We can use a min-heap (or priority queue) to efficiently retrieve the `k` closest points. In Python, the `heapq` library allows us to maintain a heap structure. Since Python's `heapq` is a min-heap, it will maintain the smallest elements at the top.

3. **Extract the Closest Points**: After pushing all points into the heap based on their distances, we can pop the smallest `k` elements to get the closest points.

4. **Handle the Output**: Finally, we prepare the list of the `k` closest points to return.

Here's how we can implement the above plan in Python, following the LeetCode solution format:



```python
import heapq

class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        # Create a min-heap based on the squared distance
        heap = []
        
        for point in points:
            x, y = point
            # Compute squared distance from the origin
            distance = x*x + y*y
            
            # Push the distance and point onto the heap
            heapq.heappush(heap, (distance, point))
        
        # Extract the k closest points
        closest_points = []
        for _ in range(k):
            closest_points.append(heapq.heappop(heap)[1])  # Get the point part of the tuple
        
        return closest_points

```

### Explanation of the Code:

1. **Importing Libraries**: We import `heapq` for using the min-heap data structure and `List` from `typing` for type hints.

2. **Defining the Class and Method**: The method `kClosest` belongs to the `Solution` class. It takes two parameters: `points` (a list of points on the plane) and `k` (the number of closest points to find).

3. **Creating the Min-Heap**: We initialize an empty list `heap` to act as our min-heap. 

4. **Populating the Heap**: We loop through each `point` in the `points` array. For each point, we calculate the squared distance from the origin using `x*x + y*y`, and then we push a tuple containing the distance and the point itself into the heap.

5. **Extracting the Closest Points**: After building the heap, we create an empty list `closest_points`. We then execute a loop `k` times to pop the smallest element from the heap and append the point (the second element of the tuple) to our list of closest points.

6. **Returning the Result**: Finally, we return the list of `k` closest points.

### Time Complexity:

- The time complexity of this solution is `O(N log N)`, where `N` is the number of points. This is due to inserting each point into the heap.
  
### Space Complexity:

- The space complexity is `O(N)` to store the heap.

This solution is efficient for the given constraints and runs well within the limits for large `N`.

# 1274. Number of Ships in a Rectangle

### Problem Description 
(This problem is an interactive problem.)
Each ship is located at an integer point on the sea represented by a cartesian plane, and each integer point may contain at most 1 ship.

You have a function `Sea.hasShips(topRight, bottomLeft)` which takes two points as arguments and returns `true` If there is at least one ship in the rectangle represented by the two points, including on the boundary.

Given two points: the top right and bottom left corners of a rectangle, return the number of ships present in that rectangle. It is guaranteed that there are at most 10 ships in that rectangle.

Submissions making more than 400 calls to `hasShips` will be judged Wrong Answer. Also, any solutions that attempt to circumvent the judge will result in disqualification.


Example :
Input: 
ships = [[1,1],[2,2],[3,3],[5,5]], topRight = [4,4], bottomLeft = [0,0]
Output: 3
Explanation: From [0,0] to [4,4] we can count 3 ships within the range.


Constraints:
On the input `ships` is only given to initialize the map internally. You must solve this problem "blindfolded". In other words, you must find the answer using the given `hasShips` API, without knowing the `ships` position.

`0 <= bottomLeft[0] <= topRight[0] <= 1000`
`0 <= bottomLeft[1] <= topRight[1] <= 1000`
`topRight != bottomLeft`

### Solution 
 To solve the problem of counting the number of ships in a given rectangular area on a Cartesian plane using the provided `Sea.hasShips(topRight, bottomLeft)` function, we need a strategic approach that ensures efficiency and adhere to the constraints given, particularly the limit of 400 calls to the `hasShips` function.

### Solution Explanation

We'll use a recursive divide-and-conquer strategy, similar to the approach used in binary search:
1. **Base Condition:** If the rectangle defined by `topRight` and `bottomLeft` does not contain any ships, directly return 0.
2. **Boundary Check:** We check if the current rectangle is valid (not empty). This is determined by checking if `topRight` is actually above and to the right of `bottomLeft`.
3. **Count Ships:** If ships exist in the current rectangle, we will further subdivide the rectangle into smaller quadrants (sub-rectangles).
4. **Recursive Division:** 
   - We compute the midpoints of the rectangle to create four smaller rectangles: top left, top right, bottom left, and bottom right.
   - Recursively call the `hasShips` function on each quadrant to determine if they contain ships. 
   - If they do, we recursively count the ships in those quadrants.

### Implementation

Below is the Python code that implements the algorithm described above. The code is encapsulated within a `Solution` class and adheres to the format that can be submitted directly to LeetCode.



```python
class Sea:
    def hasShips(self, topRight: List[int], bottomLeft: List[int]) -> bool:
        # This function is given and simulates the interaction. 
        pass 

class Solution:
    def countShips(self, sea: Sea, topRight: List[int], bottomLeft: List[int]) -> int:
        # Base case: if there are no ships in the area, return 0
        if not sea.hasShips(topRight, bottomLeft):
            return 0
        
        # If the rectangle reduces to one point, return 1 because we know there's a ship
        if topRight == bottomLeft:
            return 1
        
        # Initialize mid points
        midX = (topRight[0] + bottomLeft[0]) // 2
        midY = (topRight[1] + bottomLeft[1]) // 2
        
        # Initialize count of ships
        count = 0
        
        # Check each quadrant
        if midX >= bottomLeft[0] and midY + 1 <= topRight[1]:  # Top Left
            count += self.countShips(sea, [midX, topRight[1]], bottomLeft)
        
        if midX + 1 <= topRight[0] and midY + 1 <= topRight[1]:  # Top Right
            count += self.countShips(sea, topRight, [midX + 1, midY + 1])
        
        if midX >= bottomLeft[0] and midY >= bottomLeft[1]:  # Bottom Left
            count += self.countShips(sea, [midX, midY], bottomLeft)
        
        if midX + 1 <= topRight[0] and midY >= bottomLeft[1]:  # Bottom Right
            count += self.countShips(sea, [topRight[0], midY], [midX + 1, bottomLeft[1]])
        
        return count

```

### Key Points:
- The `countShips` function recursively divides the current rectangle into quadrants until it reaches the base case where the rectangle is either confirmed to have no ships or is reduced to a single point holding a ship.
- This approach ensures that we efficiently minimize the number of calls to `hasShips`, adhering to the limit of 400 calls, by halving the search space iteratively.
- The provided structure is suitable for the interactive environment expected in a LeetCode problem, where the `hasShips` method simulates the interface to count ships without prior knowledge of their locations. 

### Complexity:
The time complexity is approximately O(log(n * m)) for each ship check due to the recursive division of the area and the worst-case scenario of checking all quadrants. This approach is efficient given the constraints of the problem.

# 1755. Closest Subsequence Sum

### Problem Description 
You are given an integer array `nums` and an integer `goal`.

You want to choose a subsequence of `nums` such that the sum of its elements is the closest possible to `goal`. That is, if the sum of the subsequence's elements is `sum`, then you want to minimize the absolute difference `abs(sum - goal)`.

Return the minimum possible value of `abs(sum - goal)`.

Note that a subsequence of an array is an array formed by removing some elements (possibly all or none) of the original array.


Example 1:
Input: nums = [5,-7,3,5], goal = 6
Output: 0
Explanation: Choose the whole array as a subsequence, with a sum of 6.

This is equal to the goal, so the absolute difference is 0.


Example 2:
Input: nums = [7,-9,15,-2], goal = -5
Output: 1
Explanation: Choose the subsequence [7,-9,-2], with a sum of -4.

The absolute difference is abs(-4 - (-5)) = abs(1) = 1, which is the minimum.


Example 3:
Input: nums = [1,2,3], goal = -7
Output: 7

Constraints:
`1 <= nums.length <= 40`
`-107 <= nums[i] <= 107`
`-109 <= goal <= 109`

### Solution 
 To solve the problem of finding a subsequence such that the sum of its elements is closest to a given goal, we can utilize a technique called "meet in the middle." This method is particularly effective for problems with constraints that allow for such an approach, especially with constraints up to around 40 elements.

### Explanation:

1. **Divide and Conquer**: We will split the array `nums` into two halves. For each half, we will generate all possible sums of its subsequences. This results in two sets of sums that we can later combine to find the optimal solution.

2. **Subset Sums Generation**: For each half of the array, we will generate all possible sums of subsets using a simple iterative approach. For an array of length `n`, there are `2^n` possible subsets, which is feasible given our constraints.

3. **Minimizing the Difference**: After calculating all possible sums for both halves, we'll then look for pairs of sums (one from the first half and one from the second half) that together are as close as possible to the `goal`. This will involve sorting the sums from one half and using binary search (or two-pointer technique) to efficiently find the best possible candidate from the second half that, when added to a sum from the first half, gets close to `goal`.

4. **Return Result**: The final step is to return the minimum absolute difference found during the search.

Here's the implementation of the given approach:



```python
from itertools import combinations
from bisect import bisect_left

class Solution:
    def minAbsDifference(self, nums: List[int], goal: int) -> int:

        # Function to compute all subset sums of a given list
        def subset_sums(arr):
            n = len(arr)
            sums = set()
            for i in range(n + 1):
                for comb in combinations(arr, i):
                    sums.add(sum(comb))
            return list(sums)

        # Split nums into two halves
        n = len(nums)
        first_half = nums[:n//2]
        second_half = nums[n//2:]

        # Compute all subset sums for both halves
        sum_first_half = subset_sums(first_half)
        sum_second_half = subset_sums(second_half)

        # Sort the second half sums for binary search
        sum_second_half.sort()

        # We want to find the minimum abs(sum - goal)
        min_diff = float('inf')

        # Search for the best possible pairing
        for sum1 in sum_first_half:
            # We want to find the closest sum2 such that sum1 + sum2 is close to goal
            target = goal - sum1
            
            # Binary search for the closest sum2 in sum_second_half to target
            idx = bisect_left(sum_second_half, target)
            
            # Check the closest candidates
            if idx < len(sum_second_half):
                sum2 = sum_second_half[idx]
                min_diff = min(min_diff, abs(sum1 + sum2 - goal))
            if idx > 0:
                sum2 = sum_second_half[idx - 1]
                min_diff = min(min_diff, abs(sum1 + sum2 - goal))

        return min_diff

```

### Key Points:

- The `subset_sums` function generates all possible sums from the subsequences of a given array.
- We sort the sums of the second half to leverage binary search for finding close matches.
- The `bisect_left` function is used to efficiently locate the insertion point of the target in the sorted list, allowing us to explore the closest sums.

### Complexity:

- The time complexity is mainly governed by generating subsets, which leads to `O(2^(n/2))` for each half, resulting in `O(2^(n/2) * log(2^(n/2)))` for summation and subsequent searching.
- This method is feasible due to the constraints and handles the problem efficiently within acceptable limits for input size.

Overall, this solution is efficient and scales well within the provided constraints.

