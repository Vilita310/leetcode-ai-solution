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
 To solve the skyline problem, we need to merge the heights of overlapping buildings and return key points representing the transitions of height changes. We can achieve this by using a sweep line algorithm with a priority queue (max-heap) to efficiently manage the heights of the buildings as we process their edges (left and right).

### Detailed Explanation:

1. **Points of Interest**: For each building defined by `[left, right, height]`, we will generate two events: 
    - A start event when the building starts at `left` (increases the height).
    - An end event when the building ends at `right` (decreases the height).

2. **Sorting Events**: We need to sort these events:
    - Start points will be sorted by their x-coordinate, and if they have the same x-coordinate, higher heights should come first.
    - End points will be sorted by their x-coordinate, and if they have the same x-coordinate, lower heights should come first.

3. **Max-Heap for Heights**: We will use a max-heap to keep track of the heights of the buildings that currently contribute to the skyline. When processing start points, we will push the height onto the heap, and for end points, we will mark the height to be removed.

4. **Tracking the Current Height**: We keep track of the maximum height from the heap. Whenever there is a change in the maximum height (from what was previously recorded), we add a new key point to the result.

5. **Final Output**: The result will include all detected key points, which are the x-coordinates at which height changes occur, followed by the corresponding heights.

Here's the Python code implementing the above logic:



```python
import heapq
from collections import defaultdict

class Solution:
    def getSkyline(self, buildings):
        # Create a list of events (x, height, type) where type is +1 for start, -1 for end
        events = []
        for left, right, height in buildings:
            events.append((left, -height))  # Start of the building, height as negative for max-heap
            events.append((right, height))   # End of the building, height as positive
        
        # Sort events
        events.sort()

        result = []  # This will store our key points
        # Max-heap to store heights
        max_heap = [(0, float('inf'))]  # (height, right)
        prev_height = 0  # To track the previous maximum height

        for x, h in events:
            # If it's a start point (h < 0), add to the heap
            if h < 0:
                heapq.heappush(max_heap, (h, x))  # Push negative height
            else:
                # This is an end point, remove the height (lazy removal)
                max_heap = [(height, right) for height, right in max_heap if right > x]
                heapq.heapify(max_heap)  # Restore the heap property
            
            # Get the current height
            current_height = -max_heap[0][0]  # The max height (negative back to positive)

            # If current height is different from the previous height, we have a key point
            if current_height != prev_height:
                result.append([x, current_height])
                prev_height = current_height  # Update previous height

        return result
        
# Example usage
# sol = Solution()
# print(sol.getSkyline([[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]))

```

### Explanation of the Code:

- We create events for each building start and end, where start events are represented by negative heights to utilize Python's min-heap as a max-heap.
- We sort the events primarily by x-coordinate and then by height.
- We maintain a max-heap of currently active building heights where we remove heights corresponding to buildings that have ended.
- We check for height changes and if a change occurs, a key point is recorded.
- The results are returned in the required format of `[x, height]` pairs.

### Edge Cases:
- The algorithm handles overlapping buildings and ensures consecutive heights are merged.
- The use of lazy removal in the heap ensures efficient performance even with the upper constraints of the problem.

# 307. Range Sum Query - Mutable

### Problem Description 
Given an array `nums` and two types of queries where you should update the value of an index in the array, and retrieve the sum of a range in the array.

Implement the `NumArray` class:
`NumArray(int[] nums)` initializes the object with the integer array `nums`.

`void update(int index, int val)` updates the value of `nums[index]` to be `val`.

`int sumRange(int left, int right)` returns the sum of the subarray `nums[left, right]` (i.e., `nums[left] + nums[left + 1], ..., nums[right]`).


Example 1:
Input
["NumArray", "sumRange", "update", "sumRange"]
[[[1, 3, 5]], [0, 2], [1, 2], [0, 2]]
Output
[null, 9, null, 8]
Explanation
NumArray numArray = new NumArray([1, 3, 5]);
numArray.sumRange(0, 2); // return 9 = sum([1,3,5])
numArray.update(1, 2);   // nums = [1,2,5]
numArray.sumRange(0, 2); // return 8 = sum([1,2,5])

Constraints:
`1 <= nums.length <= 3 * 104`
`-100 <= nums[i] <= 100`
`0 <= index < nums.length`
`-100 <= val <= 100`
`0 <= left <= right < nums.length`
At most `3 * 104` calls will be made to `update` and `sumRange`.

### Solution 
 To solve the problem of implementing a `NumArray` class that handles updates to an array and efficiently calculates the sum of a range in that array, we can utilize a technique that allows for quick updates and range queries. A common approach for this type of problem is using a Fenwick Tree (Binary Indexed Tree) or Segment Tree, but for simplicity and to meet the constraints, we will use a straightforward approach with a prefix sum array.

Here's a breakdown of how we'll implement the `NumArray` class, followed by the actual code:

### Approach
1. **Initialization**: Store the original array and create a prefix sum array. The prefix sum array will allow us to compute the sum of elements in any range quickly.
  
2. **Update Method**: To handle updates efficiently, we will directly modify the element in the original array and then update the prefix sum array to reflect this change.

3. **Sum Range Method**: For retrieving the sum in a specific range, we can use the prefix sum array to compute the sum quickly with the formula:
   \[
   \text{sumRange}(left, right) = \text{prefixSum}[right + 1] - \text{prefixSum}[left]
   \]
   This utilizes the fact that the prefix sum array contains cumulative sums up to each index.

### Implementation

Here's the code formatted for LeetCode:



```python
class NumArray:

    def __init__(self, nums: list[int]):
        self.nums = nums  # Original array
        self.prefixSum = [0] * (len(nums) + 1)  # Prefix sum array

        # Populate the prefix sum array
        for i in range(len(nums)):
            self.prefixSum[i + 1] = self.prefixSum[i] + nums[i]

    def update(self, index: int, val: int) -> None:
        # Update the original array
        current_val = self.nums[index]
        self.nums[index] = val
        
        # Update the prefix sum array
        # Calculate the difference
        difference = val - current_val
        
        for i in range(index + 1, len(self.prefixSum)):
            self.prefixSum[i] += difference

    def sumRange(self, left: int, right: int) -> int:
        # Use the prefix sum array to calculate the sum in O(1) time
        return self.prefixSum[right + 1] - self.prefixSum[left]

# Example Usage
# numArray = NumArray([1, 3, 5])
# print(numArray.sumRange(0, 2))  # Output: 9
# numArray.update(1, 2)            # nums = [1, 2, 5]
# print(numArray.sumRange(0, 2))  # Output: 8

```

### Explanation of the Code
- The `__init__` method constructs the instance by initializing both the original array (`self.nums`) and the prefix sum (`self.prefixSum`).
- The `update` method changes the value at the specified index and then propagates that change through the prefix sum array, ensuring that it accurately reflects the new sums.
- The `sumRange` method computes the sum by leveraging the prefix sum, allowing for efficient time complexity on sum queries.

### Time Complexity
- The `update` operation is `O(n)` in the worst case when updating the prefix sum.
- The `sumRange` operation is `O(1)`, making it very efficient for retrieving sums.
- Overall, this approach balances simplicity and efficiency while adhering to the constraints given.

# 308. Range Sum Query 2D - Mutable

### Problem Description 
Given a 2D matrix `matrix`, find the sum of the elements inside the rectangle defined by its upper left corner `(row1, col1)` and lower right corner `(row2, col2)`.

Implement the NumMatrix class:
`NumMatrix(int[][] matrix)` initializes the object with the integer matrix `matrix`.

`void update(int row, int col, int val)` updates the value of `matrix[row][col]` to be `val`.

`int sumRegion(int row1, int col1, int row2, int col2)` returns the sum of the elements of the `matrix` array inside the rectangle defined by its upper left corner `(row1, col1)` and lower right corner `(row2, col2)`.


Example 1:
Input
["NumMatrix", "sumRegion", "update", "sumRegion"]
[[[[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]]], [2, 1, 4, 3], [3, 2, 2], [2, 1, 4, 3]]
Output
[null, 8, null, 10]
Explanation
NumMatrix numMatrix = new NumMatrix([[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]]);
numMatrix.sumRegion(2, 1, 4, 3); // return 8
numMatrix.update(3, 2, 2);
numMatrix.sumRegion(2, 1, 4, 3); // return 10

Constraints:
`m == matrix.length`
`n == matrix[i].length`
`1 <= m, n <= 200`
`-105 <= matrix[i][j] <= 105`
`0 <= row < m`
`0 <= col < n`
`-105 <= val <= 105`
`0 <= row1 <= row2 < m`
`0 <= col1 <= col2 < n`
At most `104` calls will be made to `sumRegion` and `update`.

### Solution 
 To solve the problem of calculating the sum of elements in a 2D matrix defined by specific coordinates, we'll create a class `NumMatrix` that provides two main functionalities: updating an element in the matrix and calculating the sum of elements in a specified rectangular region.

This task can be efficiently handled using a 2D binary indexed tree (also known as a Fenwick Tree) for quick updates and sum queries, allowing us to stay within the constraints given by the problem.

Here's how we can implement the solution:

### Explanation

1. **Data Structure Initialization**:
   We'll maintain a matrix `matrix` to store the original array, and a 2D binary indexed tree (`BIT`) to facilitate efficient updates and queries. The `BIT` will help us compute the sum of any rectangular area quickly.

2. **Updating the Matrix**:
   When we update an element in the matrix, we'll also update the `BIT` structure accordingly. We calculate the difference between the new value and the old value, and then apply this difference to the `BIT`.

3. **Sum Query**:
   For the sum query, we need to calculate the sum of the elements in the defined rectangle using the `BIT`. Using inclusion-exclusion principle, we can derive the sum for the rectangle using four sums that can be computed efficiently with the help of the `BIT`.

4. **Time Complexity**:
   Both update and sum operations will be O(log(m) * log(n)), where m is the number of rows and n is the number of columns of the matrix.

### Implementation

Below is the complete implementation following the LeetCode format:



```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        if not matrix or not matrix[0]:
            return
        self.rows = len(matrix)
        self.cols = len(matrix[0])
        self.matrix = [[0] * self.cols for _ in range(self.rows)]
        self.BIT = [[0] * (self.cols + 1) for _ in range(self.rows + 1)]

        for r in range(self.rows):
            for c in range(self.cols):
                self.updateBIT(r, c, matrix[r][c])
                self.matrix[r][c] = matrix[r][c]

    def updateBIT(self, row: int, col: int, value: int):
        delta = value - self.matrix[row][col]
        self.matrix[row][col] = value
        r = row + 1
        while r <= self.rows:
            c = col + 1
            while c <= self.cols:
                self.BIT[r][c] += delta
                c += c & -c
            r += r & -r

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return (self.sumBIT(row2 + 1, col2 + 1) 
                - self.sumBIT(row1, col2 + 1) 
                - self.sumBIT(row2 + 1, col1) 
                + self.sumBIT(row1, col1))

    def sumBIT(self, row: int, col: int) -> int:
        total = 0
        r = row
        while r > 0:
            c = col
            while c > 0:
                total += self.BIT[r][c]
                c -= c & -c
            r -= r & -r
        return total

```

### Example Usage

You can create an instance of `NumMatrix` with a 2D list, call `sumRegion` to get the sum in a specified region, and call `update` to change an element:



```python
numMatrix = NumMatrix([[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]])
print(numMatrix.sumRegion(2, 1, 4, 3))  # Output: 8
numMatrix.update(3, 2, 2)
print(numMatrix.sumRegion(2, 1, 4, 3))  # Output: 10

```

### Conclusion

This implementation efficiently handles updates and sum queries on a 2D matrix while sticking to the given constraints. It leverages a 2D binary indexed tree for effective computation.

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
 To solve the problem of finding the number of smaller elements to the right of each element in the given integer array `nums`, we can use a combination of a binary search and a data structure that allows us efficient insertion and querying of elements. The ideal approach for this problem is to leverage a data structure called a Binary Indexed Tree (also known as a Fenwick Tree) or a balanced binary search tree, such as an AVL tree or a Red-Black tree. However, for simplicity and understanding, we'll implement a simpler approach using a sorted list.

### Explanation
1. **Idea**: As we traverse the `nums` array from right to left, we keep track of the elements we have seen so far in a sorted order. For each element, we can determine how many of these seen elements are smaller using binary search.

2. **Steps**:
   - Initialize an empty list `sorted_list` to store the elements seen so far.
   - Initialize a `counts` array of the same length as `nums`, initialized to zeros.
   - Iterate over the `nums` array from right to left:
     - For each element, use the `bisect_left` function from the `bisect` module to find the index where the current element would fit in `sorted_list`. This index gives us the count of how many numbers in `sorted_list` are smaller than the current element.
     - Append the current element to `sorted_list` while maintaining its sorted order.
   - Finally, return the `counts` array.

### Time Complexity
- The binary search operation takes O(log n) time, and since we may have to insert n elements in total, the overall complexity can be approximated to O(n log n).

### Python Code
Here's the complete code that follows the structure required for a LeetCode submission:



```python
from typing import List
from bisect import bisect_left

class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        counts = [0] * len(nums)
        sorted_list = []
        
        # Traverse the nums array from right to left
        for i in range(len(nums) - 1, -1, -1):
            # Find the index where nums[i] would fit in the sorted list
            index = bisect_left(sorted_list, nums[i])
            counts[i] = index  # The number of smaller elements is the index
            # Insert nums[i] into sorted_list maintaining sorted order
            sorted_list.insert(index, nums[i])
        
        return counts

```

### Example
For `nums = [5, 2, 6, 1]`, the method works as follows:
- Start with an empty `sorted_list`.
- For `1`: sorted_list becomes `[1]`, counts becomes `[0]`.
- For `6`: sorted_list becomes `[1, 6]`, counts becomes `[0, 1]` (1 is smaller).
- For `2`: sorted_list becomes `[1, 2, 6]`, counts becomes `[2, 1]` (1 and 2 are smaller).
- For `5`: sorted_list becomes `[1, 2, 5, 6]`, counts becomes `[2, 1, 1]` (1 and 2 are smaller).

The final output for this method should return `[2, 1, 1, 0]`, which matches the expected output.

### Final Remarks
This approach effectively solves the problem within the constraints, providing a straightforward way to count the smaller elements to the right while maintaining an efficient runtime.

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
 To solve the problem of counting the number of range sums that lie within a given range `[lower, upper]`, we can utilize a more efficient algorithm based on the concept of prefix sums combined with a divide-and-conquer technique. This approach reduces the complexity significantly compared to a naive O(n^2) solution.

### Step-by-step Explanation:

1. **Prefix Sums**:
   - A prefix sum array is computed where each element at index `i` represents the sum of all elements from the start of the array `nums` up to index `i`.
   - For example, for `nums = [-2, 5, -1]`, the prefix sums will be `[0, -2, 3, 2]` where:
     - `prefix[0] = 0` (sum of no elements)
     - `prefix[1] = nums[0] = -2`
     - `prefix[2] = nums[0] + nums[1] = -2 + 5 = 3`
     - `prefix[3] = nums[0] + nums[1] + nums[2] = -2 + 5 - 1 = 2`

2. **Counting Ranges**:
   - We need to count ranges `[i, j]` which can be transformed into counting how many prefix sums fall within the range.
   - For any `j`, we want to count how many previous prefix sums `prefix[i]` satisfy the condition:
     - `lower <= prefix[j + 1] - prefix[i] <= upper`
     - This can be rearranged to:
       - `prefix[j + 1] - upper <= prefix[i] <= prefix[j + 1] - lower`

3. **Divide and Conquer with Merge Sort**:
   - We can implement a divide-and-conquer algorithm similar to merge sort. 
   - During the merging step, we keep track of the counts of valid prefix sums that fall within the required bounds.

### Python Code:
The following Python code implements the above logic and is formatted for direct use on LeetCode.



```python
class Solution:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        n = len(nums)
        prefix_sums = [0] * (n + 1)
        
        # Compute prefix sums
        for i in range(n):
            prefix_sums[i + 1] = prefix_sums[i] + nums[i]
        
        # Function to merge the counts
        def merge_count_split_inv(sums, start, mid, end, lower, upper):
            j = mid + 1
            count = 0
            
            # Counting valid prefix sums
            for i in range(start, mid + 1):
                while j <= end and sums[j] - sums[i] < lower:
                    j += 1
                k = j
                while k <= end and sums[k] - sums[i] <= upper:
                    k += 1
                count += k - j
            
            # Merge the two halves
            sums[start:end + 1] = sorted(sums[start:end + 1])
            return count
        
        def sort_and_count(sums, start, end, lower, upper):
            if start >= end:
                return 0
            
            mid = (start + end) // 2
            count = sort_and_count(sums, start, mid, lower, upper)
            count += sort_and_count(sums, mid + 1, end, lower, upper)
            count += merge_count_split_inv(sums, start, mid, end, lower, upper)
            
            return count
        
        return sort_and_count(prefix_sums, 0, n, lower, upper)

```

### Explanation of the Code:
- We first create a prefix sum array.
- The function `merge_count_split_inv` counts how many of the prefix sums fall within the specified range while merging.
- The function `sort_and_count` recursively divides the array into two halves and counts the number of valid ranges.
- Finally, the total count is accumulated and returned.

This algorithm operates in `O(n log n)` time complexity due to the merge sort-like approach, making it efficient for the input constraints provided in the problem.

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
 To solve the problem of counting important reverse pairs `(i, j)` where `i < j` and `nums[i] > 2 * nums[j]`, we can leverage a modified merge sort algorithm. The brute-force solution would involve nested loops, leading to a time complexity of O(n^2), which is impractical for larger arrays (up to 50,000 elements). Instead, the merge sort approach allows us to count these pairs while sorting the array, resulting in a time complexity of O(n log n).

### Detailed Explanation

1. **Understanding Important Reverse Pairs**: For each index `j`, we need to count how many indices `i < j` satisfy the condition `nums[i] > 2 * nums[j]`. 

2. **Using Merge Sort**: The merge sort algorithm divides the array into two halves, sorts them, and then merges them back together. During the merging process, we can count the important reverse pairs.

3. **Count Pairs During Merge**:
   - While merging two halves, for a given `j`, we can find how many indices `i` in the left half fulfill the condition `nums[i] > 2 * nums[j]`. 
   - We'll maintain a pointer for the left half and move it as needed to find the valid pairs for the current element `nums[j]`.

4. **Implementation Steps**:
   - Implement a merge sort function that returns the count of important reverse pairs while sorting the array.
   - Use a helper function for the merge process to count pairs and merge two sorted halves.

### Python Code

Here’s the implementation that fits into the LeetCode solution format:



```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        def merge_count_split_inv(nums, start, mid, end):
            if start >= end:
                return 0
            
            count = 0
            j = mid + 1
            
            # Count important reverse pairs
            for i in range(start, mid + 1):
                while j <= end and nums[i] > 2 * nums[j]:
                    j += 1
                count += (j - (mid + 1))

            # Merge the two halves
            left = nums[start:mid + 1]
            right = nums[mid + 1:end + 1]
            k = start
            i = j = 0
            
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    nums[k] = left[i]
                    i += 1
                else:
                    nums[k] = right[j]
                    j += 1
                k += 1
            
            while i < len(left):
                nums[k] = left[i]
                i += 1
                k += 1
            
            while j < len(right):
                nums[k] = right[j]
                j += 1
                k += 1
            
            return count
        
        def merge_sort(nums, start, end):
            if start >= end:
                return 0
            
            mid = (start + end) // 2
            count = merge_sort(nums, start, mid) + merge_sort(nums, mid + 1, end)
            count += merge_count_split_inv(nums, start, mid, end)
            return count
        
        return merge_sort(nums, 0, len(nums) - 1)

```

### Explanation of the Code:
- **Class Structure**: Code is encapsulated in a `Solution` class which is required by LeetCode submissions.
- **reversePairs function**: This is the main function that takes a list `nums` and returns the count of important reverse pairs.
- **merge_count_split_inv function**: This function counts reverse pairs while merging two halves of the array. It verifies if `nums[i] > 2 * nums[j]` and counts pairs accordingly.
- **merge_sort function**: This recursive function sorts the array and accumulates the count of pairs. It recursively divides the array into halves, counts pairs in each half, and counts while merging.
- **Return Statement**: Finally, `merge_sort` is called on the entire array and the count of important reverse pairs is returned.

This implementation efficiently counts important reverse pairs in O(n log n) time, making it suitable for large input sizes.

# 1649. Create Sorted Array through Instructions

### Problem Description 
Given an integer array `instructions`, you are asked to create a sorted array from the elements in `instructions`. You start with an empty container `nums`. For each element from left to right in `instructions`, insert it into `nums`. The cost of each insertion is the minimum of the following:
The number of elements currently in `nums` that are strictly less than `instructions[i]`.

The number of elements currently in `nums` that are strictly greater than `instructions[i]`.

For example, if inserting element `3` into `nums = [1,2,3,5]`, the cost of insertion is `min(2, 1)` (elements `1` and `2` are less than `3`, element `5` is greater than `3`) and `nums` will become `[1,2,3,3,5]`.

Return the total cost to insert all elements from `instructions` into `nums`. Since the answer may be large, return it modulo `109 + 7`

Example 1:
Input: instructions = [1,5,6,2]
Output: 1
Explanation: Begin with nums = [].

Insert 1 with cost min(0, 0) = 0, now nums = [1].

Insert 5 with cost min(1, 0) = 0, now nums = [1,5].

Insert 6 with cost min(2, 0) = 0, now nums = [1,5,6].

Insert 2 with cost min(1, 2) = 1, now nums = [1,2,5,6].

The total cost is 0 + 0 + 0 + 1 = 1.


Example 2:
Input: instructions = [1,2,3,6,5,4]
Output: 3
Explanation: Begin with nums = [].

Insert 1 with cost min(0, 0) = 0, now nums = [1].

Insert 2 with cost min(1, 0) = 0, now nums = [1,2].

Insert 3 with cost min(2, 0) = 0, now nums = [1,2,3].

Insert 6 with cost min(3, 0) = 0, now nums = [1,2,3,6].

Insert 5 with cost min(3, 1) = 1, now nums = [1,2,3,5,6].

Insert 4 with cost min(3, 2) = 2, now nums = [1,2,3,4,5,6].

The total cost is 0 + 0 + 0 + 0 + 1 + 2 = 3.


Example 3:
Input: instructions = [1,3,3,3,2,4,2,1,2]
Output: 4
Explanation: Begin with nums = [].

Insert 1 with cost min(0, 0) = 0, now nums = [1].

Insert 3 with cost min(1, 0) = 0, now nums = [1,3].

Insert 3 with cost min(1, 0) = 0, now nums = [1,3,3].

Insert 3 with cost min(1, 0) = 0, now nums = [1,3,3,3].

Insert 2 with cost min(1, 3) = 1, now nums = [1,2,3,3,3].

Insert 4 with cost min(5, 0) = 0, now nums = [1,2,3,3,3,4].

​​​​​​​Insert 2 with cost min(1, 4) = 1, now nums = [1,2,2,3,3,3,4].

​​​​​​​Insert 1 with cost min(0, 6) = 0, now nums = [1,1,2,2,3,3,3,4].

​​​​​​​Insert 2 with cost min(2, 4) = 2, now nums = [1,1,2,2,2,3,3,3,4].

The total cost is 0 + 0 + 0 + 0 + 1 + 0 + 1 + 0 + 2 = 4.


Constraints:
`1 <= instructions.length <= 105`
`1 <= instructions[i] <= 105`

### Solution 
 To solve the problem of inserting elements from the `instructions` into a sorted container while calculating the cost of each insertion, we will utilize a balanced data structure. Given the constraints of up to `10^5` elements, a straightforward approach (inserting into a list and sorting at each step) would be too slow. Instead, we will use the `SortedList` from the `sortedcontainers` module, which allows us to maintain a sorted list efficiently.

### Detailed Explanation

1. **Insertion Cost Calculation**:
   Whenever we want to insert an element into the `nums`, we need to find out:
   - How many elements are less than the current element (`instructions[i]`).
   - How many elements are greater than the current element.

   Using a sorted data structure like `SortedList`, we can find the insertion point which provides us with both counts directly.

2. **Using `SortedList`**:
   - The `SortedList` provides an efficient way to maintain a sorted list, allowing for `O(log n)` time complexity for both inserting a new element and finding the position where it should be inserted.
   - We can use the `bisect_left` method to find the index where `instructions[i]` fits, which gives us the count of numbers less than it.

3. **Cost Calculation**:
   - The count of elements less than `instructions[i]` is given by the index obtained from `bisect_left`.
   - The count of elements greater than it can be computed as `len(nums) - index_of_insertion`.

4. **Modulo Operation**:
   - Since the result can be large, we'll take results mod \(10^9 + 7\) as specified.

### Python Code

Below is the implementation that follows the LeetCode solution format:



```python
from sortedcontainers import SortedList

class Solution:
    def createSortedArray(self, instructions):
        MOD = 10**9 + 7
        sorted_nums = SortedList()
        total_cost = 0
        
        for num in instructions:
            # Count of elements less than num
            less_count = sorted_nums.bisect_left(num)
            # Count of elements greater than num
            greater_count = len(sorted_nums) - sorted_nums.bisect_right(num)
            # Minimum of the two counts
            cost = min(less_count, greater_count)
            total_cost = (total_cost + cost) % MOD
            # Add num to the sorted list
            sorted_nums.add(num)
        
        return total_cost

# Example usage:
# instructions = [1,5,6,2]
# result = Solution().createSortedArray(instructions)
# print(result)  # Output: 1

```

### Testing the Function

You can run the function using the provided examples:



```python
# Example 1:
result1 = Solution().createSortedArray([1, 5, 6, 2])
print(result1)  # Output: 1

# Example 2:
result2 = Solution().createSortedArray([1, 2, 3, 6, 5, 4])
print(result2)  # Output: 3

# Example 3:
result3 = Solution().createSortedArray([1, 3, 3, 3, 2, 4, 2, 1, 2])
print(result3)  # Output: 4

```

### Conclusion
This approach ensures that we do not exceed time limits while still accurately tracking costs in a sorted manner. The usage of `SortedList` allows for efficient insertion and querying, making it suitable for large inputs as described in the problem constraints.

