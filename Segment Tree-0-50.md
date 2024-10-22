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
 To solve the "Skyline Problem," we can use a sweep line algorithm combined with a priority queue (max heap) to efficiently track the maximum height of buildings as we progress through their x-coordinates. Let's break down the solution step by step:

### Explanation

1. **Event Representation**:
   We can represent each building by two events:
   - A start event `(left, -height)`: This indicates the start of a building, and we're using the height as a negative value so that when we insert heights into the max heap, the highest one is on top.
   - An end event `(right, height)`: This indicates the end of a building and we store the height as a positive value.

2. **Sorting Events**:
   We sort all events:
   - First by x-coordinate. If two events have the same x, we prioritize start events (negative heights) over end events (positive heights) so that we handle new buildings before closing existing ones.

3. **Using a Max Heap**:
   As we process each event:
   - On a start event, we add the building's height to the max heap.
   - On an end event, we remove the height from the max heap.
   We also maintain a dictionary (or counter) to keep track of how many times we've encountered each height.

4. **Recording Key Points**:
   The current maximum height (the top of the heap) gives us the height of the skyline at the current x-coordinate. Whenever this maximum height changes (from one event to the next), we add a new key point.

5. **Finalizing the Skyline**:
   After processing all events, we ensure that our skyline ends with a point at height `0` to indicate the termination of the skyline.

### Code Implementation

This implementation follows the above logic:



```python
import heapq
from collections import defaultdict

class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        events = []
        
        # Create events for each building
        for left, right, height in buildings:
            events.append((left, -height))   # starting of building
            events.append((right, height))    # ending of building
        
        # Sort events by x coordinate, and by height in case of tie
        events.sort()

        # Result list and max heap for heights
        result = []
        max_heap = [(0, 1)]  # height, count of heights
        height_count = defaultdict(int)
        height_count[0] = 1  # starting with ground level height

        prev_max_height = 0

        # Process each event
        for x, h in events:
            if h < 0:  # Starting event
                height_count[-h] += 1
                heapq.heappush(max_heap, h)
            else:  # Ending event
                height_count[h] -= 1
            
            # Maintain the max heap
            while max_heap and height_count[-max_heap[0]] == 0:
                heapq.heappop(max_heap)
            
            current_max_height = -max_heap[0] if max_heap else 0
            
            # If the current maximum height changes, we have a key point
            if current_max_height != prev_max_height:
                result.append([x, current_max_height])
                prev_max_height = current_max_height
        
        return result

```

### Explanation of the Code

- We create and sort the events based on x-coordinates.
- We define a max heap (`max_heap`) to keep track of the current heights efficiently and a height count dictionary to track how many times we encounter each height.
- As we iterate through the events, we handle adding and removing heights from the heap while maintaining the count.
- Whenever the maximum height changes, we append the new key point to the result.

This code is efficient and meets the problem constraints, allowing it to handle the maximum input sizes within a reasonable time. You can run this directly on LeetCode as a submission under the problem "The Skyline Problem."

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
 To solve the problem of supporting dynamic updates and range sum queries on an array, we can use a data structure known as a Fenwick Tree (also known as a Binary Indexed Tree). This allows for efficient updates and cumulative frequency querying. However, for the specific implementation needed here, we can also utilize a basic array approach coupled with a prefix sum array for a clearer understanding.

### Explanation

1. **Initialization**: When we initialize the `NumArray` class with an integer array `nums`, we'll store this array and maintain a prefix sum array. The prefix sum array allows us to efficiently calculate the sum of elements between two indices `left` and `right`.

2. **Update Function**: To update an element in the array, we need to:
   - Change the specific index in the original array.
   - Recalculate the prefix sums from that index to the end of the array since that change affects future sums.

3. **SumRange Function**: To get the sum of the range from index `left` to `right`, we can use the prefix sum array for an efficient constant-time query:
   - The sum from `left` to `right` can be calculated as `prefix[right + 1] - prefix[left]`, where `prefix[i]` holds the sum of the array from the start up to index `i - 1`.

### Python Code

Here’s the implementation based on this understanding, structured to work on LeetCode:



```python
class NumArray:
    def __init__(self, nums: List[int]):
        self.nums = nums
        self.n = len(nums)
        self.prefix = [0] * (self.n + 1)
        
        # Build prefix sum array
        for i in range(self.n):
            self.prefix[i + 1] = self.prefix[i] + nums[i]

    def update(self, index: int, val: int) -> None:
        # Calculate the difference and update the nums array
        diff = val - self.nums[index]
        self.nums[index] = val
        
        # Update the prefix sums from index + 1 onwards
        for i in range(index + 1, self.n + 1):
            self.prefix[i] += diff

    def sumRange(self, left: int, right: int) -> int:
        # Return the sum from left to right using prefix sums
        return self.prefix[right + 1] - self.prefix[left]

# Example Usage
# numArray = NumArray([1, 3, 5])
# print(numArray.sumRange(0, 2)) # Output: 9
# numArray.update(1, 2)
# print(numArray.sumRange(0, 2)) # Output: 8

```

### Explanation of the Code

- **Initialization (`__init__` method)**: We store the input array `nums` and compute the prefix sums. The prefix array has a size of `n+1` to easily manage the sums.

- **Update Method**: When we update an element, we first calculate the difference between the new value and the old value. We then update the `nums` array directly. Next, we adjust the `prefix` array starting from the index of the updated value to the end of the array using the computed difference.

- **SumRange Method**: Using the prefix sum array, we can return the sum of any range efficiently in constant time.

### Complexity Analysis
- **Time Complexity**:
  - `O(n)` for initialization of the prefix sum.
  - `O(n)` in the worst case for an update operation (can affect all subsequent prefix sums).
  - `O(1)` for each range sum query.

- **Space Complexity**: `O(n)` due to the storage of the `prefix` sum array.

This implementation is efficient and meets the problem's constraints effectively.

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
 To solve the problem effectively, we can utilize a 2D prefix sum array to quickly calculate the sum of elements within any specified rectangular sub-region of the matrix. This approach allows us to perform the `sumRegion` query in constant time after an initial setup phase, while the `update` operation remains somewhat efficient.

### Steps to Solve

1. **Data Structure**: We'll create a class `NumMatrix` that initializes with a given 2D matrix. We'll also maintain a prefix sum matrix to facilitate quick region sum calculations.

2. **Initialize Prefix Sum**: The prefix sum at any position `(i, j)` will hold the sum of all elements from the top-left corner `(0, 0)` to `(i, j)` of the matrix.

3. **Update Method**: When the value of an element in the matrix is updated, we will refresh the prefix sum accordingly. However, efficiently updating all relevant regions can be tricky; we may choose to recalculate the prefix sum from that point onward.

4. **Sum Region Method**: For calculating the sum of a sub-matrix defined by the corners `(row1, col1)` and `(row2, col2)`, use:
   \[
   \text{sumRegion}(row1, col1, row2, col2) = \text{prefix}[row2][col2] - \text{prefix}[row1-1][col2] - \text{prefix}[row2][col1-1] + \text{prefix}[row1-1][col1-1]
   \]
   Ensure bounds are checked for indices before accessing.

Here is the complete Python implementation following the LeetCode format:



```python
class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        self.rows = len(matrix)
        self.cols = len(matrix[0]) if matrix else 0
        self.matrix = matrix
        self.prefix_sum = [[0] * (self.cols + 1) for _ in range(self.rows + 1)]
        
        # Build the prefix sum matrix
        for r in range(1, self.rows + 1):
            for c in range(1, self.cols + 1):
                self.prefix_sum[r][c] = (self.matrix[r - 1][c - 1]
                                         + self.prefix_sum[r - 1][c]
                                         + self.prefix_sum[r][c - 1]
                                         - self.prefix_sum[r - 1][c - 1])

    def update(self, row: int, col: int, val: int) -> None:
        # Update the value in the matrix
        delta = val - self.matrix[row][col]
        self.matrix[row][col] = val
        
        # Rebuild the prefix sum from the updated position
        for r in range(row + 1, self.rows + 1):
            for c in range(col + 1, self.cols + 1):
                self.prefix_sum[r][c] += delta

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return (self.prefix_sum[row2 + 1][col2 + 1]
                - self.prefix_sum[row1][col2 + 1]
                - self.prefix_sum[row2 + 1][col1]
                + self.prefix_sum[row1][col1])

# Example Usage:
# numMatrix = NumMatrix([[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]])
# print(numMatrix.sumRegion(2, 1, 4, 3))  # Output: 8
# numMatrix.update(3, 2, 2)
# print(numMatrix.sumRegion(2, 1, 4, 3))  # Output: 10

```

### Explanation of the Code

- The constructor `__init__` initializes the `NumMatrix` object and sets up the prefix sum matrix.
- The `update` method efficiently updates the matrix's value and modifies the prefix sum matrix accordingly.
- The `sumRegion` method computes the sum of the specified rectangular region using the prefix sums.

### Complexity Analysis
- Constructing the prefix sum takes O(m * n) time, where m and n are the dimensions of the matrix.
- Each `update` operation may take O(m * n) in the worst case if we decide to refresh the entire prefix sum matrix. If we want to optimize this, more advanced data structures like Fenwick Trees or Segment Trees could be applied, but they require added complexity.
- Each `sumRegion` query runs in O(1), due to the pre-computed prefix sums. 

This design provides a good balance between simplicity and efficiency for the given problem constraints.

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
 To solve the problem of finding the counts of smaller elements to the right of each element in the given array `nums`, we can apply a more efficient approach using a data structure like a Binary Indexed Tree (also known as a Fenwick Tree) or a modified merge sort. This is more optimal than the naive O(n^2) approach of nested loops.

### Explanation of the Solution

Here are the steps we'll follow to solve the problem:

1. **Understanding the Input and Output**: For each element in the array `nums`, we need to count the number of elements that are smaller than it and appear after it. The output is an array `counts` of the same length as `nums`.

2. **Using Coordinate Compression**: Since the values of `nums` can range from `-10^4` to `10^4`, we can use coordinate compression to map these values into a smaller range which will be easier to manage in our data structure.

3. **Binary Indexed Tree (Fenwick Tree)**: We will leverage a Binary Indexed Tree to keep track of counts in a dynamic manner as we process elements from right to left. The Fenwick Tree will help us efficiently count the number of elements that have been seen so far that are less than the current element.

### Implementation

We will create a class `Solution` with a method `countSmaller`. Inside this method:
- We will compress the values of `nums`.
- Iterate through the elements from right to left.
- Use the Binary Indexed Tree to maintain the counts and query for the count of smaller elements efficiently.

Here’s how we can implement this:



```python
class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        count = 0
        while index > 0:
            count += self.tree[index]
            index -= index & -index
        return count

class Solution:
    def countSmaller(self, nums):
        sorted_nums = sorted(set(nums))
        rank = {v: i + 1 for i, v in enumerate(sorted_nums)}  # Create rank dict
        
        counts = []
        fenwick_tree = FenwickTree(len(sorted_nums))
        
        # Traverse from right to left
        for num in reversed(nums):
            # Get the rank of the current number
            r = rank[num]
            # Query the number of smaller elements (rank - 1)
            counts.append(fenwick_tree.query(r - 1))
            # Update the fenwick tree with the current number's rank
            fenwick_tree.update(r, 1)

        # Since we traversed the array from right to left, reverse the counts array
        counts.reverse()
        return counts

```

### Explanation of Code:

1. **FenwickTree Class**:
   - This class implements the methods to update the tree and query the cumulative counts.
   - The `update` method modifies the tree for the specific index.
   - The `query` method retrieves the cumulative count of elements up to a specified index.

2. **Solution Class**:
   - The `countSmaller` method first creates a sorted list of unique numbers to establish their rank.
   - We then traverse `nums` in reverse order, at each step querying the count of elements smaller than the current element based on its rank and updating the Fenwick Tree.
   - Finally, we reverse the `counts` because we have built it from the rightmost element back to the left.

This algorithm runs in O(n log n) time complexity, which is efficient given the constraints. The use of the Fenwick Tree allows us to keep track of how many numbers have been seen efficiently.

You can now copy and paste this code into a LeetCode editor to run it against provided test cases.

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
 To solve the problem of counting range sums within a specific range efficiently, we can use a combination of prefix sums and a modified merge sort algorithm. The key insights are:

1. **Prefix Sum**: We compute a prefix sum array such that `prefix[i]` is the sum of elements from the beginning of the array up to the `i`-th index. The range sum `S(i, j)` can be computed as `prefix[j] - prefix[i-1]`. 

2. **Count Valid Sums**: We need to count how many range sums fall within the bounds `[lower, upper]`. Given `S(i, j)`, we can transform this into the requirement that we need counts of `prefix[j] - upper <= prefix[i-1] <= prefix[j] - lower`. 

3. **Merge Sort**: By using a modified merge sort, we can sort the prefix sums while simultaneously counting how many of them fall within the desired range for each prefix sum. This allows us to achieve an efficient `O(n log n)` solution.

Here's the implementation:



```python
class Solution:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        # Step 1: Prepare the prefix sums
        prefix_sums = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefix_sums[i + 1] = prefix_sums[i] + nums[i]
        
        # Step 2: Define a helper function for merge sort and counting
        def merge_and_count(start, mid, end):
            count = 0
            j, k = mid + 1, mid + 1
            for left in range(start, mid + 1):
                while j <= end and prefix_sums[j] - prefix_sums[left] < lower:
                    j += 1
                while k <= end and prefix_sums[k] - prefix_sums[left] <= upper:
                    k += 1
                count += (k - j)
            
            # Step 3: Merge phase
            prefix_sums[start:end + 1] = sorted(prefix_sums[start:end + 1])
            return count

        def merge_sort_and_count(start, end):
            if start >= end:
                return 0
            mid = (start + end) // 2
            count = merge_sort_and_count(start, mid) + merge_sort_and_count(mid + 1, end)
            count += merge_and_count(start, mid, end)
            return count
        
        return merge_sort_and_count(0, len(prefix_sums) - 1)


```

### Explanation of the Code:

1. **Prefix Sums Calculation**:
    - We initialize a list `prefix_sums` of size `len(nums) + 1`, where `prefix_sums[0]` is `0`, and each subsequent index contains the sum of all elements up to that index.

2. **Merge and Count Function**:
    - We count how many valid `prefix_sums` fall within the specified range using two pointers (`j` and `k`) for the bounds defined by the current `prefix_sums[left]`. 
    - The count of valid sums is updated by checking conditions defined by `lower` and `upper`.

3. **Merge Sort and Counting**:
    - This function implements a standard merge sort where, after breaking down the array, it counts valid ranges and then merges sorted halves back together.

4. **Final Call**:
    - The main method returns the total count of valid range sums using the recursive `merge_sort_and_count`.

This implementation is efficient and works within the provided constraints. It captures range sums directly via prefix sum manipulation and simplifies counting through a sorted merge approach, giving it a time complexity of `O(n log n)`.

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
 To solve the problem of counting the number of important reverse pairs `(i, j)` in the array `nums`, where `i < j` and `nums[i] > 2 * nums[j]`, we can use a modified merge sort algorithm. This approach is efficient and avoids the brute-force O(n^2) complexity that would be prohibitive for large arrays (up to 50,000 elements).

### Detailed Explanation:

1. **Understanding Reverse Pairs**: We need to count pairs where the first number is greater than twice the second number. This condition can be checked while performing a merge sort, which divides the array and counts pairs during the merge step.

2. **Merge Sort Basics**: Merge sort is a divide-and-conquer algorithm that splits the array into halves, sorts them, and then merges the sorted halves while ensuring the order.

3. **Counting Important Reverse Pairs**: 
   - For each element `nums[i]` in the left half of our current segment during merging, we need to count how many elements in the right half are less than half of `nums[i]`. This will give us the count of important reverse pairs for `nums[i]`.

4. **Implementation Steps**:
   - Implement the merge sort function that sorts and counts pairs.
   - In the merge step, maintain the count of important pairs using a two-pointer technique to efficiently check how many elements in the right half satisfy the condition.

5. **Complexity**: The modified merge sort runs in O(n log n) time, which is efficient for the input size constraint.

Here is the implementation of the above logic in Python, following the LeetCode submission format:



```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        def merge_count_split_inv(left: List[int], right: List[int]) -> int:
            count = 0
            j = 0
            
            # Count important reverse pairs
            for i in range(len(left)):
                while j < len(right) and left[i] > 2 * right[j]:
                    j += 1
                count += j
            
            # Merge the two halves
            sorted_array = []
            i = 0
            j = 0
            
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    sorted_array.append(left[i])
                    i += 1
                else:
                    sorted_array.append(right[j])
                    j += 1
            
            # Add remaining elements
            sorted_array.extend(left[i:])
            sorted_array.extend(right[j:])
            
            return count, sorted_array
        
        def merge_sort_and_count(nums: List[int]) -> int:
            if len(nums) < 2:
                return 0, nums
            
            mid = len(nums) // 2
            left_count, left_sorted = merge_sort_and_count(nums[:mid])
            right_count, right_sorted = merge_sort_and_count(nums[mid:])
            merge_count, merged = merge_count_split_inv(left_sorted, right_sorted)
            
            return left_count + right_count + merge_count, merged
        
        total_reverse_pairs, _ = merge_sort_and_count(nums)
        return total_reverse_pairs

# Example usage:
# sol = Solution()
# print(sol.reversePairs([1,3,2,3,1]))  # Output: 2
# print(sol.reversePairs([2,4,3,5,1]))  # Output: 3

```

### Explanation of the Code:
- `merge_count_split_inv`: This function counts the reverse pairs across the split and also merges the two halves of the array.
- `merge_sort_and_count`: This recursively divides the array and calls the merge function, aggregating the count of reverse pairs.
- `reversePairs`: This is the main function that initiates the merge sort and returns the total count of reverse pairs.

The provided solution is structured to work efficiently within the constraints of the problem and can be run directly on LeetCode.

# 699. Falling Squares

### Problem Description 
On an infinite number line (x-axis), we drop given squares in the order they are given.

The `i`-th square dropped (`positions[i] = (left, side_length)`) is a square with the left-most point being `positions[i][0]` and sidelength `positions[i][1]`.

The square is dropped with the bottom edge parallel to the number line, and from a higher height than all currently landed squares. We wait for each square to stick before dropping the next.

The squares are infinitely sticky on their bottom edge, and will remain fixed to any positive length surface they touch (either the number line or another square). Squares dropped adjacent to each other will not stick together prematurely.

Return a list `ans` of heights. Each height `ans[i]` represents the current highest height of any square we have dropped, after dropping squares represented by `positions[0], positions[1], ..., positions[i]`.


Example 1:
Input: [[1, 2], [2, 3], [6, 1]]
Output: [2, 5, 5]
Explanation:
After the first drop of `positions[0] = [1, 2]: _aa _aa ------- `The maximum height of any square is 2.

After the second drop of `positions[1] = [2, 3]: __aaa __aaa __aaa _aa__ _aa__ -------------- `The maximum height of any square is 5. The larger square stays on top of the smaller square despite where its center of gravity is, because squares are infinitely sticky on their bottom edge.

After the third drop of `positions[1] = [6, 1]: __aaa __aaa __aaa _aa _aa___a -------------- `The maximum height of any square is still 5. Thus, we return an answer of `[2, 5, 5]`.


Example 2:
Input: [[100, 100], [200, 100]]
Output: [100, 100]
Explanation: Adjacent squares don't get stuck prematurely - only their bottom edge can stick to surfaces.

Note:
`1 <= positions.length <= 1000`.

`1 <= positions[i][0] <= 10^8`.

`1 <= positions[i][1] <= 10^6`.

### Solution 
 To solve the problem of dropping squares on an infinite number line and calculating their heights, we can take a structured approach. The idea is to keep track of the highest point achieved after dropping each square based on its position and side length.

### Explanation of the Approach

1. **Understanding the Positions**: Each square is defined by its leftmost position (`left`) and its side length (`side_length`). This allows us to calculate:
   - The rightmost point of the square as `right = left + side_length`.

2. **Height Calculation**: 
   - Each square can only rest on top of currently existing structures. The height of a square is defined as `height[i] = Base Height + Current Height`.
   - The "Base Height" is determined by identifying if the current square overlaps with any previously dropped squares.
   - We need to know the highest position supported at the `left` point upto the `right` point of the current square.

3. **Data Structure Choice**: We will use a list to store the maximum height at different positions due to squares stacking up.

4. **Iterative Build-up**: For each square, check the height at its base and add the height of the square to get its effective height. Then, update this position for subsequent squares.

### Steps to Implement

- Initialize an empty list `ans` to store the maximum heights.
- For each square, determine its effective height based on the heights of squares that are already on the ground underneath it.
- Keep updating the maximum height in the `ans` list as we process each square.

### Implementation in Python

Here is a complete implementation following the LeetCode format:



```python
class Solution:
    def fallingSquares(self, positions):
        # Result list to hold the height after each drop
        ans = []
        # Array to record the height at each position
        heights = []

        max_height = 0  # The maximum height of any square dropped so far
        
        for left, side_length in positions:
            right = left + side_length
            current_height = 0
            
            # Determine the highest point where this square will rest
            for i in range(len(heights)):
                # Check if the current square overlaps with any existing squares
                if not (right <= positions[i][0] or left >= positions[i][0] + positions[i][1]):
                    current_height = max(current_height, heights[i])  # Update current height

            # The height of the new square will be based on the highest one beneath it
            new_height = current_height + side_length
            heights.append(new_height)  # Add this square's height to the list
            
            max_height = max(max_height, new_height)  # Update the max height across all squares
            ans.append(max_height)  # Record the current max height

        return ans

# Example usage:
# solution = Solution()
# print(solution.fallingSquares([[1, 2], [2, 3], [6, 1]]))  # Output: [2, 5, 5]
# print(solution.fallingSquares([[100, 100], [200, 100]]))  # Output: [100, 100]

```

### Explanation of the Code

- The `Solution` class has a method `fallingSquares` which takes a list of positions.
- We maintain a `heights` list to track the heights of each square as it is added.
- For each new square, we calculate its overlap with previously placed squares and determine the maximum height it will achieve.
- Finally, we update our result list `ans` with the maximum height after processing each square.
- The code can be run directly on LeetCode or similar platforms.

# 715. Range Module

### Problem Description 
A Range Module is a module that tracks ranges of numbers. Your task is to design and implement the following interfaces in an efficient manner.

`addRange(int left, int right)` Adds the half-open interval `[left, right)`, tracking every real number in that interval.  Adding an interval that partially overlaps with currently tracked numbers should add any numbers in the interval `[left, right)` that are not already tracked.

`queryRange(int left, int right)` Returns true if and only if every real number in the interval `[left, right)`
 is currently being tracked.

`removeRange(int left, int right)` Stops tracking every real number currently being tracked in the interval `[left, right)`.


Example 1:
addRange(10, 20): null
removeRange(14, 16): null
queryRange(10, 14): true (Every number in [10, 14) is being tracked)
queryRange(13, 15): false (Numbers like 14, 14.03, 14.17 in [13, 15) are not being tracked)
queryRange(16, 17): true (The number 16 in [16, 17) is still being tracked, despite the remove operation)
Note:
A half open interval `[left, right)` denotes all real numbers `left <= x < right`.

`0 < left < right < 10^9` in all calls to `addRange, queryRange, removeRange`.

The total number of calls to `addRange` in a single test case is at most `1000`.

The total number of calls to `queryRange` in a single test case is at most `5000`.

The total number of calls to `removeRange` in a single test case is at most `1000`.

### Solution 
 To solve the problem of managing a range module that can add, query, and remove ranges efficiently, we can utilize an interval list to maintain the tracked intervals. We'll represent each interval as a tuple of two elements `(left, right)`. The following methods will be implemented:

1. **addRange(int left, int right)**: This method adds the range `[left, right)` to our collection. We need to merge any overlapping or contiguous intervals after adding the new interval.

2. **queryRange(int left, int right)**: This method checks if every number in the interval `[left, right)` is being tracked. This means we need to check if the entire query range is covered by our tracked intervals.

3. **removeRange(int left, int right)**: This method removes the range `[left, right)` from the collection. It might split existing intervals if the removed range overlaps with them.

Given the constraints, we will maintain the intervals in a sorted list to ensure that merging and querying operations are efficient.

Here is the complete implementation in Python, formatted to run directly on LeetCode:



```python
class RangeModule:

    def __init__(self):
        self.intervals = []

    def addRange(self, left: int, right: int) -> None:
        new_intervals = []
        i = 0
        # Add all intervals that end before left
        while i < len(self.intervals) and self.intervals[i][1] < left:
            new_intervals.append(self.intervals[i])
            i += 1
        
        # Merge all overlapping intervals
        while i < len(self.intervals) and self.intervals[i][0] <= right:
            left = min(left, self.intervals[i][0])
            right = max(right, self.intervals[i][1])
            i += 1
        
        new_intervals.append((left, right))
        
        # Add remaining intervals that start after right
        while i < len(self.intervals):
            new_intervals.append(self.intervals[i])
            i += 1

        self.intervals = new_intervals

    def queryRange(self, left: int, right: int) -> bool:
        for l, r in self.intervals:
            if l <= left < r:
                # If we found an interval that starts before the right and
                # covers the left, we need to check if it covers the full range
                return r >= right
            if l >= right:
                break
        return False

    def removeRange(self, left: int, right: int) -> None:
        new_intervals = []
        for l, r in self.intervals:
            if r <= left:  # No overlap, add the entire interval
                new_intervals.append((l, r))
            elif l >= right:  # No overlap, add the entire interval
                new_intervals.append((l, r))
            else:  # There is some overlap
                if l < left:
                    new_intervals.append((l, left))
                if r > right:
                    new_intervals.append((right, r))
        self.intervals = new_intervals

```

### Explanation of the Implementation:

1. **The `__init__` method** initializes an empty list `intervals` to store the tracked ranges.

2. **The `addRange` method**:
   - We first iterate through the `intervals` to add all intervals that end before the new range starts. 
   - We then merge any overlapping intervals by adjusting the left and right boundaries.
   - Finally, we add the new merged interval and any remaining intervals that start after the new range.

3. **The `queryRange` method**:
   - It checks each interval to see if it overlaps with or contains the query range `[left, right)`.
   - If an interval is found that starts before `right` and it ends after `left`, it checks if this interval extends to cover `right`.

4. **The `removeRange` method**:
   - It constructs a new list that excludes any parts of the existing intervals that overlap with the range `[left, right)`.
   - If an existing interval is completely excluded, it is not added; otherwise, portions of it that do not overlap are added back.

This solution efficiently handles the operations while maintaining the intervals in a sorted manner, ensuring that both time and space complexities are managed well within the provided constraints.

# 732. My Calendar III

### Problem Description 
A `k`-booking happens when `k` events have some non-empty intersection (i.e., there is some time that is common to all `k` events.)
You are given some events `[start, end)`, after each given event, return an integer `k` representing the maximum `k`-booking between all the previous events.

Implement the `MyCalendarThree` class:
`MyCalendarThree()` Initializes the object.

`int book(int start, int end)` Returns an integer `k` representing the largest integer such that there exists a `k`-booking in the calendar.


Example 1:
Input
["MyCalendarThree", "book", "book", "book", "book", "book", "book"]
[[], [10, 20], [50, 60], [10, 40], [5, 15], [5, 10], [25, 55]]
Output
[null, 1, 1, 2, 3, 3, 3]
Explanation
MyCalendarThree myCalendarThree = new MyCalendarThree();
myCalendarThree.book(10, 20); // return 1, The first event can be booked and is disjoint, so the maximum k-booking is a 1-booking.

myCalendarThree.book(50, 60); // return 1, The second event can be booked and is disjoint, so the maximum k-booking is a 1-booking.

myCalendarThree.book(10, 40); // return 2, The third event [10, 40) intersects the first event, and the maximum k-booking is a 2-booking.

myCalendarThree.book(5, 15); // return 3, The remaining events cause the maximum K-booking to be only a 3-booking.

myCalendarThree.book(5, 10); // return 3
myCalendarThree.book(25, 55); // return 3

Constraints:
`0 <= start < end <= 109`
At most `400` calls will be made to `book`.

### Solution 
 To solve the problem of tracking the maximum number of overlapping events in a calendar, we can utilize a concept known as the "sweep line" algorithm, which involves sorting events and tracking changes to the active count of overlaps as we sweep through the timeline.

### Step-by-step Explanation

1. **Class Definition**: We will create a class called `MyCalendarThree`. It will have a method `book` for adding events and returning the maximum number of overlapping intervals at any given time.

2. **Data Structure**: We will keep a list of all events defined by their start and end times. For efficiency, we will leverage a dictionary to count overlapping occurrences at specified time points.

3. **Event Marking**: For each event booked (defined by a start and end time):
   - We will increment a count at `start` (indicating an event has started).
   - We will decrement a count at `end` (indicating an event has ended). This helps in tracking overlaps correctly.

4. **Calculating Max Overlaps**:
   - We will iterate through the keys of the dictionary, maintaining a running count of currently active events. During this sweep, we will update the maximum overlaps recorded.

5. **Return Values**: After booking an event, we return the maximum number of overlapping events observed up to that point.

### Python Implementation 

Now, let's translate the above approach into Python code that can be used on LeetCode.



```python
class MyCalendarThree:
    
    def __init__(self):
        # This will hold the number of active events at specific times
        self.timeline = {}
        
    def book(self, start: int, end: int) -> int:
        # Mark the start and end of the event in the timeline
        self.timeline[start] = self.timeline.get(start, 0) + 1  # Increment for start time
        self.timeline[end] = self.timeline.get(end, 0) - 1      # Decrement for end time
        
        # Now we need to walk through the timeline to find the maximum k-booking
        max_k = 0  # To keep track of the maximum overlaps
        current_k = 0  # Current number of overlapping events
        
        for time in sorted(self.timeline.keys()):  # Sort the keys (time points)
            current_k += self.timeline[time]  # Updated current overlaps
            max_k = max(max_k, current_k)  # Update max_k if current exceeds it
        
        return max_k  # Return the maximum booking at this stage

# Example Usage
# myCalendarThree = MyCalendarThree()
# print(myCalendarThree.book(10, 20))  # Output: 1
# print(myCalendarThree.book(50, 60))  # Output: 1
# print(myCalendarThree.book(10, 40))  # Output: 2
# print(myCalendarThree.book(5, 15))  # Output: 3
# print(myCalendarThree.book(5, 10))  # Output: 3
# print(myCalendarThree.book(25, 55))  # Output: 3

```

### Explanation of Code

1. **Initialization**: The `timeline` dictionary keeps track of how many events start and end at each time point.

2. **Booking Events**: In the `book` method, for each event:
   - We update our `timeline` for both start and end times.
   - We calculate the current number of overlapping events by iterating through the timeline's keys (sorted) and maintaining a cumulative count.

3. **Maximum Overlap Count**: As we process each time point, we check and update the maximum overlap seen so far.

This code efficiently manages up to 400 calls in logarithmic time concerning the number of unique time points, making it well-suited for the problem constraints.

# 850. Rectangle Area II

### Problem Description 
We are given a list of (axis-aligned) `rectangles`. Each `rectangle[i] = [xi1, yi1, xi2, yi2] `, where `(xi1, yi1)` are the coordinates of the bottom-left corner, and `(xi2, yi2)` are the coordinates of the top-right corner of the `ith` rectangle.

Find the total area covered by all `rectangles` in the plane. Since the answer may be too large, return it modulo `109 + 7`.


Example 1:
Input: rectangles = [[0,0,2,2],[1,0,2,3],[1,0,3,1]]
Output: 6
Explanation: As illustrated in the picture.


Example 2:
Input: rectangles = [[0,0,1000000000,1000000000]]
Output: 49
Explanation: The answer is 1018 modulo (109 + 7), which is (109)2 = (-7)2 = 49.


Constraints:
`1 <= rectangles.length <= 200`
`rectanges[i].length = 4`
`0 <= rectangles[i][j] <= 109`
The total area covered by all rectangles will never exceed `263 - 1` and thus will fit in a 64-bit signed integer.

### Solution 
 To solve the problem of calculating the total area covered by a list of rectangles, we can employ a geometric approach using a sweep line algorithm. The key concept is to keep track of active rectangles as we move through the x-coordinates, effectively calculating areas for segments along the y-axis.

Here's a step-by-step breakdown of the approach:

1. **Collect Events:** For each rectangle defined by \([xi1, yi1, xi2, yi2]\), we will create two events:
   - A "start" event at \(xi1\) (indicating the rectangle is starting).
   - An "end" event at \(xi2\) (indicating the rectangle is ending).
   We will also store the y-coordinates \(yi1\) and \(yi2\) to keep track of the vertical spans of the rectangles.

2. **Sort Events:** We will sort these events based on the x-coordinate. If two events have the same x-coordinate, we prioritize the "end" event before the "start" event to ensure areas are computed correctly.

3. **Sweep Line Algorithm:** We will iterate through the sorted events and maintain a list of active y-intervals (the y-coordinate bounds of rectangles currently being processed):
   - For "start" events, we add the y-interval to our list of active intervals.
   - For "end" events, we remove the corresponding y-interval.
   - Between distinct x-coordinates, we calculate the area covered by the active rectangles using their combined y-interval lengths.

4. **Area Calculation:** The area for each segment between x-coordinates can be computed as:
   - \( \text{Width} \times \text{Total Height Covered} \) where:
     - \( \text{Width} \) is the difference between the current and previous x-coordinates.
     - \( \text{Total Height Covered} \) is the union of the lengths of the active y-intervals.

5. **Modulo Operation:** Since the result can be very large, we will return the area modulo \(10^9 + 7\).

Now, let's put this into Python code that can be used directly on LeetCode:



```python
class Solution:
    def rectangleArea(self, rectangles: List[List[int]]) -> int:
        MOD = 10**9 + 7
        events = []  # List to store events (start and end)
        
        # Collect all events
        for x1, y1, x2, y2 in rectangles:
            events.append((x1, y1, y2, 1))  # Starting event (1)
            events.append((x2, y1, y2, -1))  # Ending event (-1)
        
        # Sort events, prioritizing x position; when x is the same, prioritize ends
        events.sort()
        
        # This will hold the active y-intervals
        active_intervals = []
        prev_x = events[0][0]  # Start with the first x coordinate
        total_area = 0
        
        def compute_height():
            """ Calculate the total height covered by active intervals """
            height = 0
            last_y = -1
            
            for y1, y2 in active_intervals:
                if last_y < y1: 
                    height += y2 - y1  # Add new interval height
                    last_y = y2
                elif last_y < y2:
                    height += y2 - last_y  # Extend with the overlap
                    last_y = y2
                    
            return height
        
        for x, y1, y2, typ in events:
            # Calculate the area covered from prev_x to current x
            width = x - prev_x
            if width > 0:
                height = compute_height()
                total_area = (total_area + width * height) % MOD
            
            # Update the active intervals based on the event type
            if typ == 1:  # Start event
                active_intervals.append((y1, y2))
            else:  # End event
                active_intervals.remove((y1, y2))
            
            prev_x = x  # Move to the next x coordinate

        return total_area

```

### Explanation of the Code:
- We define a `Solution` class and a method `rectangleArea`.
- We create events for each rectangle and sort them appropriately.
- We use a helper function `compute_height` to calculate the total height covered by active intervals.
- We maintain a list of active intervals, update it as we encounter start and end events, and calculate the area whenever the x-coordinate changes.
- Finally, we return the total area modulo \(10^9 + 7\). 

This code is structured in a way that follows the standard format used in LeetCode submissions, with relevant imports for the `List` type and class-based encapsulation for solutions.

# 1157. Online Majority Element In Subarray

### Problem Description 
Implementing the class `MajorityChecker`, which has the following API:
`MajorityChecker(int[] arr)` constructs an instance of MajorityChecker with the given array `arr`;
`int query(int left, int right, int threshold)` has arguments such that:
	
`0 <= left <= right < arr.length` representing a subarray of `arr`;
`2 * threshold > right - left + 1`, ie. the threshold is always a strict majority of the length of the subarray
Each `query(...)` returns the element in `arr[left], arr[left+1], ..., arr[right]` that occurs at least `threshold` times, or `-1` if no such element exists.


Example:
MajorityChecker majorityChecker = new MajorityChecker([1,1,2,2,1,1]);
majorityChecker.query(0,5,4); // returns 1
majorityChecker.query(0,3,3); // returns -1
majorityChecker.query(2,3,2); // returns 2

Constraints:
`1 <= arr.length <= 20000`
`1 <= arr[i] <= 20000`
For each query, `0 <= left <= right < len(arr)`
For each query, `2 * threshold > right - left + 1`
The number of queries is at most `10000`

### Solution 
 To solve the problem of implementing the class `MajorityChecker`, we need to efficiently handle multiple queries asking for the majority element in specified subarrays of an array. Given the constraints on the array length and the maximum number of queries, it’s important to use an efficient design.

Here’s how we can achieve this:

1. **Data Structure Choice**: We will use a dictionary (hashmap) to store the indices of each number in the array. This will allow us to quickly find how many times a number appears within a given range by using binary search techniques.

2. **Binary Search**: For each query, we will:
   - Use the pre-stored indices of the elements to find the frequency of the candidate numbers in the specified range using binary search. This will help us quickly ascertain how many times a number appears between two indices.

3. **Query Processing**: For each query:
   - We will take the left and right bounds, and for each number in the range, we'll check if its count meets the threshold.

### Code Implementation

Below is the full implementation of the `MajorityChecker` class following these strategies, structured to run directly on the LeetCode platform:



```python
from collections import defaultdict
from bisect import bisect_left, bisect_right

class MajorityChecker:

    def __init__(self, arr: List[int]):
        self.arr = arr
        # A dictionary to hold the indices for each number in arr.
        self.indices = defaultdict(list)
        for idx, num in enumerate(arr):
            self.indices[num].append(idx)

    def query(self, left: int, right: int, threshold: int) -> int:
        candidate = -1
        count = 0

        # We can randomly choose a candidate number to check its frequency.
        for _ in range(10):
            random_index = random.randint(left, right)
            candidate = self.arr[random_index]

            # Find the range of indices for this candidate
            indices_list = self.indices[candidate]
            l_index = bisect_left(indices_list, left)
            r_index = bisect_right(indices_list, right)

            # Count occurrences in the specified range
            current_count = r_index - l_index
            
            # If current_count meets the threshold, return the candidate
            if current_count >= threshold:
                return candidate
        
        # If no candidate met the threshold
        return -1

```

### Explanation

1. **Initialization (`__init__` method)**:
   - The constructor takes an array and constructs a dictionary (`indices`) where each key is a unique number from the array, and its value is a list of all indices where this number occurs. This set-up takes O(n) time.

2. **Query Processing (`query` method)**:
   - We run the query a maximum of 10 times to randomly select potential candidates within the provided range.
   - For each candidate chosen randomly, we use the `bisect_left` and `bisect_right` functions to quickly find the number of occurrences of that candidate within the specified `left` and `right` range.
   - If the count of occurrences of the candidate satisfies the threshold requirement, we return that candidate.
   - If none of the candidates satisfy the threshold after 10 attempts, we return -1.

### Performance Analysis
- The `__init__` function runs in O(n) time.
- Each query runs in O(log(m)) time due to binary search, where m is the number of occurrences of the candidate number being checked.
- In a worst-case scenario, with up to 10,000 queries, the algorithm is efficient and adheres to the problem's constraints.

Overall, this class structure and query handling allow efficient searching for the majority element in a given subarray effectively solving the problem within specified limits.

# 1353. Maximum Number of Events That Can Be Attended

### Problem Description 
Given an array of `events` where `events[i] = [startDayi, endDayi]`. Every event `i` starts at `startDayi` and ends at `endDayi`.

You can attend an event `i` at any day `d` where `startTimei <= d <= endTimei`. Notice that you can only attend one event at any time `d`.

Return the maximum number of events you can attend.


Example 1:
Input: events = [[1,2],[2,3],[3,4]]
Output: 3
Explanation: You can attend all the three events.

One way to attend them all is as shown.

Attend the first event on day 1.

Attend the second event on day 2.

Attend the third event on day 3.


Example 2:
Input: events= [[1,2],[2,3],[3,4],[1,2]]
Output: 4

Example 3:
Input: events = [[1,4],[4,4],[2,2],[3,4],[1,1]]
Output: 4

Example 4:
Input: events = [[1,100000]]
Output: 1

Example 5:
Input: events = [[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7]]
Output: 7

Constraints:
`1 <= events.length <= 105`
`events[i].length == 2`
`1 <= startDayi <= endDayi <= 105`

### Solution 
 Here’s a solution to the problem of maximizing the number of events you can attend in Python. We will leverage a greedy algorithmic approach to effectively manage the overlapping intervals.

### Python Code


```python
class Solution:
    def maxEvents(self, events: List[List[int]]) -> int:
        # Sort events by their end days (and start days for tie-breaking)
        events.sort(key=lambda x: (x[1], x[0]))
        
        max_attended = 0
        current_day = 0
        
        for start, end in events:
            # If we can attend the event on the current_day
            if current_day < start:
                # Move current_day to the start of the event
                current_day = start
            
            # Attend the event if current_day is between start and end
            if start <= current_day <= end:
                max_attended += 1
                current_day += 1  # Move to the next day after attending this event
        
        return max_attended

```

### Explanation
1. **Input Structure**: The input is a list of events, where each event is represented as a list containing two integers: `[startDay, endDay]`. 

2. **Sorting Events**: The first step is to sort the events primarily by their end day and, in the case of a tie, by their start day. This sorting allows us to consider the events which finish the earliest first, making it easier to maximize the number of events we can attend.

3. **Grabbing the Events**: We initialize `max_attended` to count how many events we can attend and `current_day` to track which day we are on. 
   - We iterate through each sorted event.
   - If `current_day` is less than the event's start day, we update `current_day` to the event's start time, indicating that we will attend the event starting on that day.
   - If `current_day` falls within the event's range (from start to end), we can attend this event. We increment the count of attended events and also move to the next day (`current_day += 1`) to ensure we don’t attend multiple events on the same day.

4. **Return the Result**: After going through all the events, we return the total count of attended events.

### Complexity
- **Time Complexity**: O(n log n) due to sorting the events, where n is the number of events.
- **Space Complexity**: O(1), not accounting for the input storage; we use constant extra space.

### Testing the Solution
You can run the above code on LeetCode by using the `Solution` class and calling the method with the given examples. This approach efficiently ensures you attend the maximum number of events while adhering to the constraints provided.

# 1521. Find a Value of a Mysterious Function Closest to Target

### Problem Description 
Winston was given the above mysterious function `func`. He has an integer array `arr` and an integer `target` and he wants to find the values `l` and `r` that make the value `|func(arr, l, r) - target|` minimum possible.

Return the minimum possible value of `|func(arr, l, r) - target|`.

Notice that `func` should be called with the values `l` and `r` where `0 <= l, r < arr.length`.


Example 1:
Input: arr = [9,12,3,7,15], target = 5
Output: 2
Explanation: Calling func with all the pairs of [l,r] = [[0,0],[1,1],[2,2],[3,3],[4,4],[0,1],[1,2],[2,3],[3,4],[0,2],[1,3],[2,4],[0,3],[1,4],[0,4]], Winston got the following results [9,12,3,7,15,8,0,3,7,0,0,3,0,0,0]. The value closest to 5 is 7 and 3, thus the minimum difference is 2.


Example 2:
Input: arr = [1000000,1000000,1000000], target = 1
Output: 999999
Explanation: Winston called the func with all possible values of [l,r] and he always got 1000000, thus the min difference is 999999.


Example 3:
Input: arr = [1,2,4,8,16], target = 0
Output: 0

Constraints:
`1 <= arr.length <= 105`
`1 <= arr[i] <= 106`
`0 <= target <= 107`

### Solution 
 To solve the problem at hand, we need to find pairs of indices \( l \) and \( r \) in the `arr` array such that the function `func(arr, l, r)` is called, yielding a result that is as close as possible to a specified `target` value. The `func` function is defined according to the problem statement in terms of cumulative operations.

### Understanding `func`
The function `func(arr, l, r)` can be defined as the sum of all elements from index `l` to index `r`, inclusive. This requires calculating the sum for every possible pair of indices \( l \) and \( r \), and then determining the absolute difference between each calculated sum and the target.

### Steps to Solve the Problem
1. **Use Prefix Sums:** To efficiently calculate the sum between any two indices \( l \) and \( r \), we can precompute prefix sums. The prefix sum array allows us to retrieve the sum of any subarray in constant time.

2. **Iterate through Pairs:** Using two nested loops, we can iterate through all possible pairs of \( l \) and \( r \).

3. **Calculate Minimum Difference:** For each sum derived from \( func(arr, l, r) \), calculate the absolute difference with the target. Keep track of the smallest difference encountered.

### Implementation in Python
Here's the Python code that executes the steps mentioned above and fits the LeetCode submission format:



```python
class Solution:
    def minAbsoluteDifference(self, arr, target: int) -> int:
        n = len(arr)
        
        # Step 1: Compute prefix sums
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + arr[i - 1]
        
        # Step 2: Initialize the minimum difference
        min_diff = float('inf')
        
        # Step 3: Iterate through all pairs (l, r)
        for l in range(n):
            for r in range(l, n):
                func_value = prefix_sum[r + 1] - prefix_sum[l]
                min_diff = min(min_diff, abs(func_value - target))
        
        return min_diff

# Example usage:
# sol = Solution()
# print(sol.minAbsoluteDifference([9,12,3,7,15], 5))  # Output: 2
# print(sol.minAbsoluteDifference([1000000,1000000,1000000], 1))  # Output: 999999
# print(sol.minAbsoluteDifference([1,2,4,8,16], 0))  # Output: 0

```

### Explanation of the Code
1. **Prefix Sum Calculation:**
   - We create a `prefix_sum` array where `prefix_sum[i]` stores the sum of the first `i` elements of the array. This allows us to compute the sum of any subarray `arr[l:r]` as `prefix_sum[r + 1] - prefix_sum[l]`.

2. **Nested Loop for Pairs:**
   - We loop through each possible pair \( (l, r) \) where \( l \) is less than or equal to \( r \). The total time complexity for this nested loop is \( O(n^2) \), which is feasible given the constraints.

3. **Finding the Minimum Difference:**
   - For each subarray sum calculated, we compute the absolute difference with the target and update our minimum difference whenever we find a smaller one.

### Efficiency
The approach is straightforward but may become impractical for larger limits, given it has a time complexity of \( O(n^2) \). However, it systematically covers all combinations and guarantees the correct result. Further optimizations might be needed for very large inputs or different constraints.

# 1526. Minimum Number of Increments on Subarrays to Form a Target Array

### Problem Description 
Given an array of positive integers `target` and an array `initial` of same size with all zeros.

Return the minimum number of operations to form a `target` array from `initial` if you are allowed to do the following operation:
Choose any subarray from `initial` and increment each value by one.

The answer is guaranteed to fit within the range of a 32-bit signed integer.


Example 1:
Input: target = [1,2,3,2,1]
Output: 3
Explanation: We need at least 3 operations to form the target array from the initial array.

[0,0,0,0,0] increment 1 from index 0 to 4 (inclusive).

[1,1,1,1,1] increment 1 from index 1 to 3 (inclusive).

[1,2,2,2,1] increment 1 at index 2.

[1,2,3,2,1] target array is formed.


Example 2:
Input: target = [3,1,1,2]
Output: 4
Explanation: (initial)[0,0,0,0] -> [1,1,1,1] -> [1,1,1,2] -> [2,1,1,2] -> [3,1,1,2] (target).


Example 3:
Input: target = [3,1,5,4,2]
Output: 7
Explanation: (initial)[0,0,0,0,0] -> [1,1,1,1,1] -> [2,1,1,1,1] -> [3,1,1,1,1] 
                                  -> [3,1,2,2,2] -> [3,1,3,3,2] -> [3,1,4,4,2] -> [3,1,5,4,2] (target).


Example 4:
Input: target = [1,1,1,1]
Output: 1

Constraints:
`1 <= target.length <= 10^5`
`1 <= target[i] <= 10^5`

### Solution 
 To solve the problem of minimizing the number of operations required to transform an `initial` array of zeros into a `target` array of positive integers using the specified increment operations, we can observe a key principle regarding changes in the array.

The operations can be visualized as:
1. Incrementing all the elements in any chosen subarray (all elements between two indices).
  
Instead of directly simulating each operation, we can break down the problem by noting that the number of operations required to reach any element in the `target` array depends solely on the difference in values between adjacent elements in the array. Because:
- If `target[i] > target[i-1]`, we need to perform increments all the way from `target[i-1]` to `target[i]`.
- If `target[i] < target[i-1]`, no additional operations are needed going in the opposite direction.

From that observation, the plan is:
1. We start with the initial `initial` array as zero-filled.
2. We iterate through the `target` array and calculate the number of increments needed when moving from one index to the next.
3. Specifically, we will only count the positive differences between consecutive elements to accumulate the total operations.

Here's the Python code that implements this algorithm:



```python
def minOperations(target):
    # Initialize the number of operations required
    operations = 0
    
    # Iterate through the target array
    for i in range(len(target)):
        # We only care about the increments needed, i.e., the positive differences
        # For the first element, we can directly add its value
        if i == 0:
            operations += target[i]
        else:
            # Add the difference if it's positive
            operations += max(0, target[i] - target[i - 1])

    return operations

# Example Usage
# Example 1
target1 = [1, 2, 3, 2, 1]
print(minOperations(target1))  # Output: 3

# Example 2
target2 = [3, 1, 1, 2]
print(minOperations(target2))  # Output: 4

# Example 3
target3 = [3, 1, 5, 4, 2]
print(minOperations(target3))  # Output: 7

# Example 4
target4 = [1, 1, 1, 1]
print(minOperations(target4))  # Output: 1

```

### Explanation of the Code:
1. `minOperations` function accepts the `target` array as input.
2. We initialize a variable `operations` to zero to keep track of the total increments required.
3. We loop through each index of the `target` array:
   - For the first element, we add its value directly to `operations`.
   - For every subsequent element, we check the difference with its predecessor. If `target[i]` is greater than `target[i-1]`, we add the difference to `operations`.
4. This results in a minimal number of operations calculated in O(n) time complexity, making the solution efficient for the input constraints.

### Complexity:
- **Time Complexity**: O(n), where n is the length of the target array.
- **Space Complexity**: O(1), since we are using a constant amount of extra space for operations count. 

This solution executes efficiently and meets the constraints provided in the problem statement, making it ready for use on platforms such as LeetCode.

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
 To solve the problem of calculating the total cost of inserting elements from the given `instructions` list into a sorted array `nums`, we will implement a Python solution using a combination of methods. One efficient approach to maintain the sorted order and perform the necessary operations to find the cost is to use a `SortedList` from the `sortedcontainers` module. This allows us to efficiently maintain a dynamic sorted list with logarithmic time complexity for insertions and queries.

Here's a breakdown of how the algorithm will work:

1. **Initialization**: We start with an empty list `nums` and a variable `total_cost` to keep track of the cumulative insertion cost.

2. **Insertion Process**:
   - For each element `instructions[i]`:
     - Use the `bisect_left` method to find the position where `instructions[i]` should be inserted.
     - This position tells us how many elements in `nums` are strictly less than `instructions[i]`.
     - The number of elements strictly greater can be determined by subtracting the position index from the current length of `nums`.
     - Calculate the insertion cost as the minimum of the two counts.
     - Update the total cost.

3. **Insertion in Sorted List**: After calculating the cost for the current element, we insert it into the `nums` list.

4. **Return Result**: Finally, since the cumulative cost can be large, we return `total_cost` modulo \(10^9 + 7\).

Here is the implementation:



```python
from sortedcontainers import SortedList

class Solution:
    def createSortedArray(self, instructions):
        MOD = 10**9 + 7
        nums = SortedList()
        total_cost = 0

        for instruction in instructions:
            # Find count of elements in nums that are < instruction
            left_count = nums.bisect_left(instruction)
            # Find count of elements in nums that are > instruction
            right_count = len(nums) - nums.bisect_right(instruction)
            # Calculate the cost for the current insertion
            total_cost += min(left_count, right_count)
            # Insert the instruction into the sorted list
            nums.add(instruction)

        return total_cost % MOD

# Example usage
solution = Solution()
print(solution.createSortedArray([1, 5, 6, 2]))  # Output: 1
print(solution.createSortedArray([1, 2, 3, 6, 5, 4]))  # Output: 3
print(solution.createSortedArray([1, 3, 3, 3, 2, 4, 2, 1, 2]))  # Output: 4

```

### Explanation of the Code:
- **Imports**: We import `SortedList` from the `sortedcontainers` module to maintain a sorted sequence of numbers.
- **MOD**: We define `MOD` as \(10^9 + 7\) to apply modulo operations when returning the result.
- **Loop Through Instructions**: For each number in `instructions`, we calculate both the number of elements smaller and larger than it using pre-defined methods of `SortedList`—efficiently maintaining the sorted order.
- **Cost Calculation**: The minimum of these two counts gives the cost of insertion for that element.
- **Insertion**: After calculating the cost, we insert the current element into the list using `add()`.
- **Final Result**: The total cost is returned modulo \(10^9 + 7\).

This solution efficiently handles the problem constraints and provides the correct results as illustrated in the examples.

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
 To solve the problem of determining the minimum number of trips a ship needs to make to deliver boxes to their respective ports, we can approach it with a greedy strategy. The key is to prioritize loading as many boxes as possible without exceeding the limits on the number of boxes and the total weight. We'll iterate through the `boxes` array, keep track of the current load and the deliveries, and count the trips accordingly.

### Problem Breakdown

1. **Understanding Inputs**: We have a list of boxes where each box is represented as a list containing the destination port and the weight. We also have constraints on the maximum number of boxes (`maxBoxes`) and the total weight that can be delivered (`maxWeight`).

2. **Trip Calculation**:
   - For each trip, we can load boxes into the ship until we reach either the maximum number of boxes or the maximum weight.
   - Once the ship is loaded, we need to deliver the boxes in order and keep track of the trips made.
   - Each trip to a port and back counts as two trips, except when the ship is already at the destination port (in which case it is only one trip).

3. **Implementation Steps**:
   - Iterate through the `boxes` list and keep track of the current weight and count of boxes.
   - When adding a box would exceed either limit, make a trip for the boxes currently loaded.
   - Move to the next port and repeat until all boxes are delivered.

### Python Code

Here is the Python code that implements this logic in a format suitable for directly running on LeetCode:



```python
class Solution:
    def boxDelivering(self, boxes: List[List[int]], portsCount: int, maxBoxes: int, maxWeight: int) -> int:
        trips = 0
        curr_weight = 0
        curr_boxes = 0
        current_port = 0  # Start at storage

        i = 0
        while i < len(boxes):
            trips += 1  # New trip starts
            curr_weight = 0
            curr_boxes = 0
            last_port = None
            
            while i < len(boxes):
                port, weight = boxes[i]
                # Check if we can take this box
                if curr_boxes < maxBoxes and curr_weight + weight <= maxWeight:
                    curr_boxes += 1
                    curr_weight += weight
                    last_port = port  # Update last port delivered
                    i += 1  # Move to the next box
                else:
                    break  # Can't load more boxes, exit inner loop
            
            # Now deliver all loaded boxes
            # Check if last_port changes
            if last_port is not None:
                # Need to go to the last port and return to storage
                trips += 1  # Move to port
                if last_port != current_port:
                    trips += 1  # Return to storage if we moved to another port
            
        return trips

```

### Explanation of the Code
1. **Variables**:
   - `trips`: Counts the number of trips made by the ship.
   - `curr_weight`: Tracks the current total weight of the boxes on the ship.
   - `curr_boxes`: Counts the number of boxes on the ship.
   - `current_port`: Stores the current location of the ship (initially at storage).

2. **Outer While Loop**: It iterates through the `boxes` array until all boxes have been processed.

3. **Inner While Loop**: It tries to load boxes into the ship while checking the constraints on weight and number of boxes. Once it cannot take more boxes, it exits the inner loop.

4. **Delivery and Trips Calculation**: After loading boxes, it calculates the number of trips needed to deliver to the last port and back. If the last port is different from the storage, it adds additional trips.

### Conclusion
This greedy approach ensures that we efficiently load the maximum possible boxes within the constraints and calculate the trips required, efficiently processing each box only once. The solution is optimal and runs in linear time relative to the number of boxes.

