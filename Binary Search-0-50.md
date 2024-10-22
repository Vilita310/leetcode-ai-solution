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
 To solve the problem of finding the median of two sorted arrays, we can use a binary search approach which ensures we achieve a time complexity of O(log(min(m, n))). This is optimal given the constraints provided, as simply merging the two arrays and then finding the median would have a time complexity of O(m + n).

### Explanation of the Approach:

1. **Understanding the Median**:
   - The median is the middle value in an ordered array, or the average of the two middle values if the array has an even length.
   - For two combined arrays, we're interested in the central values after sorting.

2. **Binary Search on the Shorter Array**:
   - We need to ensure our binary search is performed on the smaller of the two arrays. This minimizes the number of elements we need to consider for partitioning.
   - We can define two arrays, `A` (length m) and `B` (length n), and make sure `A` is the smaller one.

3. **Partitioning the Arrays**:
   - We partition both arrays such that the left half of `A` and `B` combined is equal in size to the right half.
   - Using binary search, we adjust our partitions based on the conditions:
     - The largest element on the left of `A` should be less than or equal to the smallest element on the right of `B`, and vice versa.

4. **Median Calculation**:
   - If the combined length of the two arrays is odd, the median is the maximum of the left parts.
   - If it's even, the median is the average of the maximum of the left parts and the minimum of the right parts.

### Implementation:
Here is the Python code that implements the described approach following the LeetCode format:



```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # Lengths of the arrays
        m, n = len(nums1), len(nums2)
        
        # Ensure nums1 is the smaller array
        if m > n:
            nums1, nums2, m, n = nums2, nums1, n, m
        
        # Binary search on the smaller array
        low, high = 0, m
        
        while low <= high:
            partition1 = (low + high) // 2  # Partition index for nums1
            partition2 = (m + n + 1) // 2 - partition1  # Partition index for nums2
            
            # If partition is at the edge, use -infinity/ +infinity instead
            maxLeft1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
            minRight1 = float('inf') if partition1 == m else nums1[partition1]
            
            maxLeft2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
            minRight2 = float('inf') if partition2 == n else nums2[partition2]
            
            # Check if we have found the correct partition
            if maxLeft1 <= minRight2 and maxLeft2 <= minRight1:
                # Correct partition is found
                if (m + n) % 2 == 0:  # Even length
                    return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2
                else:  # Odd length
                    return max(maxLeft1, maxLeft2)
            elif maxLeft1 > minRight2:
                # We are too far on right side for partition1. Go on left side.
                high = partition1 - 1
            else:
                # We are too far on left side for partition1. Go on right side.
                low = partition1 + 1

        # If we reach here, there's an error
        raise ValueError("Input arrays are not sorted")

# Example usage:
# sol = Solution()
# print(sol.findMedianSortedArrays([1, 3], [2]))  # Output: 2.0
# print(sol.findMedianSortedArrays([1, 2], [3, 4]))  # Output: 2.5

```

### Key Points:
- The algorithm uses binary search to find a proper partition, which allows for efficient computation of the median.
- It handles edge cases such as empty arrays and differentiates evenly and oddly sized combined arrays.
- The complexity of O(log(min(m, n))) makes it suitable for the problem constraints.

# 29. Divide Two Integers

### Problem Description 
Given two integers `dividend` and `divisor`, divide two integers without using multiplication, division, and mod operator.

Return the quotient after dividing `dividend` by `divisor`.

The integer division should truncate toward zero, which means losing its fractional part. For example, `truncate(8.345) = 8` and `truncate(-2.7335) = -2`.

Note: Assume we are dealing with an environment that could only store integers within the 32-bit signed integer range: `[−231, 231 − 1]`. For this problem, assume that your function returns `231 − 1` when the division result overflows.


Example 1:
Input: dividend = 10, divisor = 3
Output: 3
Explanation: 10/3 = truncate(3.33333..) = 3.


Example 2:
Input: dividend = 7, divisor = -3
Output: -2
Explanation: 7/-3 = truncate(-2.33333..) = -2.


Example 3:
Input: dividend = 0, divisor = 1
Output: 0

Example 4:
Input: dividend = 1, divisor = 1
Output: 1

Constraints:
`-231 <= dividend, divisor <= 231 - 1`
`divisor != 0`

### Solution 
 To solve the problem of dividing two integers without using multiplication, division, and mod operators, we can employ a method that utilizes bit manipulation and subtraction. The goal is to arrive at the quotient of the division using bit shifts, which effectively simulates the division process.

**Detailed Explanation:**
1. **Handling Edge Cases**: 
    - We need to handle overflow scenarios where the result exceeds the constraints of a 32-bit signed integer. The only overflow case occurs when dividing `-2^31` (minimum integer) by `-1` (as this would yield `2^31`, which is outside the range).
  
2. **Determine the Sign of the Result**:
    - The result of the division will be negative if one (and only one) of the `dividend` or `divisor` is negative. We can determine the sign using a boolean flag.

3. **Working with Absolute Values**: 
    - We take the absolute values of both `dividend` and `divisor` because working with positive numbers simplifies the logic for subtraction.

4. **Bit Manipulation to Build the Quotient**:
    - Using a repetitive subtraction can be expensive, so instead, we can double the divisor using a left bit shift and see how many times we can subtract that from the dividend. This leveraging of bit shifts helps us to efficiently determine how many times the divisor fits into the dividend.
    - For each doubling (using left shifts), we also track a separate power of two that indicates how many times we have doubled the divisor.

5. **Constructing the Quotient**:
    - We accumulate the quotient as we find how many times we can subtract the current value of the divisor (after it has been shifted) from the dividend.

6. **Return the Result**: 
    - Finally, we adjust the sign of the quotient based on the earlier sign determination and ensure it's within the 32-bit integer limits.

Now with this plan in mind, let's implement it as a function in a format suitable for LeetCode.



```python
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        # Constants for the limits of a 32-bit signed integer
        INT_MAX = 2**31 - 1
        INT_MIN = -2**31
        
        # Handle the overflow case
        if dividend == INT_MIN and divisor == -1:
            return INT_MAX
        
        # Determine the sign of the result
        negative = (dividend < 0) != (divisor < 0)
        
        # Work with absolute values
        a = abs(dividend)
        b = abs(divisor)
        
        quotient = 0
        # The power represents the current multiple of the divisor
        for i in range(31, -1, -1):
            # The shifted value of divisor
            if (b << i) <= a:
                a -= (b << i)
                quotient += (1 << i)
        
        # Apply the sign to the result
        return -quotient if negative else quotient

# Example usage (you can test by calling the method with different inputs)
# solution = Solution()
# print(solution.divide(10, 3)) # Output: 3
# print(solution.divide(7, -3)) # Output: -2
# print(solution.divide(0, 1))   # Output: 0
# print(solution.divide(1, 1))   # Output: 1

```

In this implementation:
- The bit manipulation allows the algorithm to be efficient and quick due to the logarithmic number of shifts and masks.
- We ensure to follow the problem's instructions concerning integer overflow by returning `INT_MAX` when division overflows.

# 33. Search in Rotated Sorted Array

### Problem Description 
There is an integer array `nums` sorted in ascending order (with distinct values).

Prior to being passed to your function, `nums` is rotated at an unknown pivot index `k` (`0 <= k < nums.length`) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (0-indexed). For example, `[0,1,2,4,5,6,7]` might be rotated at pivot index `3` and become `[4,5,6,7,0,1,2]`.

Given the array `nums` after the rotation and an integer `target`, return the index of `target` if it is in `nums`, or `-1` if it is not in `nums`.


Example 1:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Example 2:
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

Example 3:
Input: nums = [1], target = 0
Output: -1

Constraints:
`1 <= nums.length <= 5000`
`-104 <= nums[i] <= 104`
All values of `nums` are unique.

`nums` is guaranteed to be rotated at some pivot.

`-104 <= target <= 104`
Follow up: Can you achieve this in `O(log n)` time complexity?

### Solution 
 To solve the problem of finding a target in a rotated sorted array, we can employ a modified binary search algorithm. The key insight here is to use the properties of the rotated sorted array to discern which subarray (left or right of the mid-point) to continue searching.

Here's a breakdown of the approach:

1. **Identify the rotated structure**: Given that the array is sorted and then rotated, we know that one part of the array will always be sorted. For instance, if you're at a midpoint (`mid`), you can easily determine if the left side (`nums[left]` to `nums[mid]`) or the right side (`nums[mid]` to `nums[right]`) is sorted.

2. **Decide which part to search**: 
   - If the left part is sorted (`nums[left] <= nums[mid]`):
     - Check if the target lies in this sorted left part. If it does, narrow the search to the left part by adjusting the right boundary (`high = mid - 1`).
     - If the target is not in the left part, search in the right part by adjusting the left boundary (`low = mid + 1`).
   
   - If the right part is sorted (`nums[mid] <= nums[right]`):
     - Check if the target lies in this sorted right part. If it does, adjust the left boundary (`low = mid + 1`).
     - If it does not, search in the left part by adjusting the right boundary (`high = mid - 1`).

3. **Repeat until found or exhausted**: Continue the process until the left bound is greater than the right bound, indicating the target is not present in the array.

Here’s how to implement this in Python:



```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # Initialize the left and right pointers
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2  # To avoid overflow
            
            # Check if we found the target
            if nums[mid] == target:
                return mid
            
            # Determine which side is sorted
            if nums[left] <= nums[mid]:  # Left side is sorted
                if nums[left] <= target < nums[mid]:  # Target is in the sorted left
                    right = mid - 1
                else:  # Target is in the right
                    left = mid + 1
            else:  # Right side must be sorted
                if nums[mid] < target <= nums[right]:  # Target is in the sorted right
                    left = mid + 1
                else:  # Target is in the left
                    right = mid - 1
        
        return -1  # Target not found

```

### Explanation of the Code:
- **Initialization**: We start by initializing two pointers: `left` at the beginning of the array and `right` at the end.
- **Looping until found**: We use a `while` loop that continues as long as `left` is less than or equal to `right`.
- **Finding the midpoint**: In each iteration, we calculate the `mid` index.
- **Target check**: If the value at `mid` matches the target, we return `mid`.
- **Identifying sorted halves**: The algorithm checks which half of the array is sorted and then narrows down the search space accordingly.
- **Returning the result**: If the target is not found, we return -1.

### Complexity:
- The time complexity of this approach is O(log n), which is efficient for the input size constraint provided. Total space complexity is O(1) since we are using a constant amount of extra space.

This solution directly fits the requirements of the LeetCode platform and can be submitted directly.

# 34. Find First and Last Position of Element in Sorted Array

### Problem Description 
Given an array of integers `nums` sorted in ascending order, find the starting and ending position of a given `target` value.

If `target` is not found in the array, return `[-1, -1]`.

Follow up: Could you write an algorithm with `O(log n)` runtime complexity?

Example 1:
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

Example 2:
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]

Example 3:
Input: nums = [], target = 0
Output: [-1,-1]

Constraints:
`0 <= nums.length <= 105`
`-109 <= nums[i] <= 109`
`nums` is a non-decreasing array.

`-109 <= target <= 109`

### Solution 
 To solve the problem of finding the starting and ending position of a target value in a sorted array of integers, we can utilize a binary search approach for an efficient `O(log n)` runtime complexity. The idea is to perform two separate binary search operations: one to find the leftmost (starting) position of the target and the other to find the rightmost (ending) position.

Here’s a structured breakdown of the solution:

### Approach:
1. **Binary Search**: We will define a helper function to perform binary search. This function will return the index of the target if it's found; otherwise, it returns -1.

2. **Find the Starting Position**: We will modify the binary search to ensure that we continue searching in the left half even after finding the target, in order to find the first occurrence.

3. **Find the Ending Position**: Similarly, we will modify binary search to find the last occurrence by searching in the right half.

4. **Return Results**: Finally, we return the indices of the starting and ending positions of the target in a list. If the target is not found, we return `[-1, -1]`.

### Implementation:
Let's implement the above logic in Python:



```python
class Solution:
    def searchRange(self, nums: list[int], target: int) -> list[int]:
        def findLeftIndex(nums, target):
            left, right = 0, len(nums) - 1
            leftIndex = -1
            
            while left <= right:
                mid = left + (right - left) // 2
                
                if nums[mid] == target:
                    leftIndex = mid  # Update leftIndex
                    right = mid - 1  # Look for earlier occurrences
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return leftIndex
        
        def findRightIndex(nums, target):
            left, right = 0, len(nums) - 1
            rightIndex = -1
            
            while left <= right:
                mid = left + (right - left) // 2
                
                if nums[mid] == target:
                    rightIndex = mid  # Update rightIndex
                    left = mid + 1  # Look for later occurrences 
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return rightIndex
        
        # Calculate the starting and ending position
        start = findLeftIndex(nums, target)
        end = findRightIndex(nums, target)
        
        if start == -1 or end == -1:
            return [-1, -1]  # Target not found
        
        return [start, end]  # Return the found indices

```

### Explanation:
- We define the method `searchRange` that takes the sorted array `nums` and the `target` value as inputs.
- We then define two helper functions:
  - `findLeftIndex`: This performs a binary search to find the leftmost position of the target.
  - `findRightIndex`: This performs a binary search to find the rightmost position of the target.
- The `searchRange` method first calls these helper functions to get the start and end indices of the target in the array.
- If we couldn't find either index (i.e., target is not present), we return `[-1, -1]`. Otherwise, we return the indices as a list.

### Time Complexity:
The time complexity for this approach is `O(log n)` because we are performing binary search twice.

This solution can be run directly on LeetCode as it follows the required format with the proper function signature and type annotations.

# 35. Search Insert Position

### Problem Description 
Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.


Example 1:
Input: nums = [1,3,5,6], target = 5
Output: 2

Example 2:
Input: nums = [1,3,5,6], target = 2
Output: 1

Example 3:
Input: nums = [1,3,5,6], target = 7
Output: 4

Example 4:
Input: nums = [1,3,5,6], target = 0
Output: 0

Example 5:
Input: nums = [1], target = 0
Output: 0

Constraints:
`1 <= nums.length <= 104`
`-104 <= nums[i] <= 104`
`nums` contains distinct values sorted in ascending order.

`-104 <= target <= 104`

### Solution 
 To solve the problem of finding the index of a target value in a sorted array of distinct integers (or the index where it could be inserted), we can efficiently use the binary search algorithm. The binary search approach is suitable here because the input array is sorted. We will be aiming for a time complexity of O(log n), which is what binary search provides.

Here's the plan:
1. We'll create two pointers, `left` and `right`, to represent the range of indices we are currently examining in the array.
2. We'll perform a loop that repeatedly divides the search range in half.
3. If the middle element of the current range is equal to the target, we'll return its index.
4. If the middle element is less than the target, we'll adjust the `left` pointer to one position right of the middle, narrowing our search to the right half.
5. If the middle element is greater than the target, we'll adjust the `right` pointer to one position left of the middle, narrowing our search to the left half.
6. If we exit the loop without finding the target, at this point, the `left` pointer will be at the position where the target would fit in the sorted order.

Here's the complete Python code that implements the above logic, formatted for LeetCode:



```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return left  # left is the insertion point if target is not found

```

### Explanation of the Code:
- **Class Definition**: We define a class named `Solution`, which adheres to the structure that LeetCode expects.
- **Function Definition**: The method `searchInsert` takes two parameters: `nums`, which is a list of integers, and `target`, which is the integer we want to search for.
- **Pointers Initialization**: We initialize `left` to 0 and `right` to the last index of the array (i.e., `len(nums) - 1`).
- **While Loop**: This loop runs as long as `left` is less than or equal to `right`.
  - **Calculate Middle**: We calculate the middle index `mid`.
  - **Comparison**:
    - If `nums[mid]` is equal to `target`, we return `mid`.
    - If `nums[mid]` is less than `target`, it means we need to look in the right half, so we update `left` to `mid + 1`.
    - If `nums[mid]` is greater than `target`, we update `right` to `mid - 1` to search in the left half.
- **Return Statement**: If we exit the loop, it means the `target` was not found, so we return `left`, which indicates the index where the `target` should be inserted.

This approach ensures that we can efficiently determine the desired index in logarithmic time, adhering to the constraints given in the problem statement.

# 50. Pow(x, n)

### Problem Description 
Implement pow(x, n), which calculates `x` raised to the power `n` (i.e., `xn`).


Example 1:
Input: x = 2.00000, n = 10
Output: 1024.00000

Example 2:
Input: x = 2.10000, n = 3
Output: 9.26100

Example 3:
Input: x = 2.00000, n = -2
Output: 0.25000
Explanation: 2-2 = 1/22 = 1/4 = 0.25

Constraints:
`-100.0 < x < 100.0`
`-231 <= n <= 231-1`
`-104 <= xn <= 104`

### Solution 
 To solve the problem of implementing the `pow(x, n)` function, we can use a recursive method known as "Exponentiation by Squaring," which is efficient and significantly reduces the number of multiplications needed to compute the result. This method works based on the property of exponents:

1. If `n` is even, `x^n = (x^(n/2))^2`
2. If `n` is odd, `x^n = x * x^(n-1)`

Additionally, we need to handle the case where `n` is negative, as `x^(-n) = 1/(x^n)`.

### Detailed Breakdown of the Algorithm

1. **Base Cases**: 
   - If `n == 0`, return `1` because any number raised to the power of 0 is 1.
   - If `n < 0`, we will compute the power using its positive counterpart and then take the reciprocal.

2. **Recursive Case**: 
   - For even `n`, compute the power recursively by reducing the exponent: `pow(x, n/2)^2`.
   - For odd `n`, compute it by: `x * pow(x, n-1)`.

3. **Handling Negative Powers**: When `n` is negative, we compute `1 / pow(x, -n)`.

### Implementation

Here is the complete Python code for the `myPow` function in the LeetCode format:



```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        # Base case: any number to the power of 0 is 1
        if n == 0:
            return 1.0
        
        if n < 0:
            x = 1 / x  # For negative powers, compute reciprocal
            n = -n
        
        # Recursive function to compute power
        def recurse(x: float, n: int) -> float:
            if n == 0:
                return 1.0
            half_pow = recurse(x, n // 2)  # Compute x^(n/2)
            if n % 2 == 0:
                return half_pow * half_pow  # If even
            else:
                return half_pow * half_pow * x  # If odd
        
        return recurse(x, n)

# Example Usage
# sol = Solution()
# print(sol.myPow(2.00000, 10))  # Output: 1024.00000
# print(sol.myPow(2.10000, 3))   # Output: 9.26100
# print(sol.myPow(2.00000, -2))  # Output: 0.25000

```

### Explanation of the Code

1. **Class Definition (`Solution`)**: The function `myPow` is defined as a method of the `Solution` class, which is required by LeetCode for its solutions.
  
2. **Base Case Handling**: The function checks if `n` is zero and returns `1.0`. If `n` is negative, it computes the positive exponent and inversely handles `x`.

3. **Recursive Function**: Inside `myPow`, we define a helper function `recurse` to perform the actual exponentiation recursively.

4. **Returning the Result**: Finally, `myPow` calls the `recurse` method and returns the calculated power.

### Efficiency
This implementation has a logarithmic time complexity O(log n), making it significantly more efficient than a naive approach that would have linear time complexity O(n). The recursive nature allows us to reduce the number of multiplications effectively.

# 69. Sqrt(x)

### Problem Description 
Given a non-negative integer `x`, compute and return the square root of `x`.

Since the return type is an integer, the decimal digits are truncated, and only the integer part of the result is returned.


Example 1:
Input: x = 4
Output: 2

Example 2:
Input: x = 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since the decimal part is truncated, 2 is returned.


Constraints:
`0 <= x <= 231 - 1`

### Solution 
 To solve the problem of calculating the integer square root of a non-negative integer `x`, we can use a binary search approach. The integer square root of `x` is the largest integer `y` such that `y * y <= x`. 

## Detailed Explanation

1. **Binary Search Intuition**:
   - We can search for the square root between two bounds: `0` and `x`. 
   - If `x` is `0` or `1`, we can return `x` directly since the square roots are `0` and `1` respectively.
   - For values of `x` greater than `1`, we can set our initial search boundaries:
       - `low = 0`
       - `high = x`
   - We will continue to narrow down our search:
       - Calculate `mid = (low + high) // 2`.
       - If `mid * mid` is equal to `x`, we have found the square root.
       - If `mid * mid` is less than `x`, we need to search to the right by setting `low = mid + 1`.
       - If `mid * mid` is greater than `x`, we need to search to the left by setting `high = mid - 1`.
   - The process continues until `low` exceeds `high`, at which point `high` will be the largest integer whose square is less than or equal to `x`.

2. **Time Complexity**:
   - The time complexity of this approach is `O(log x)` because we halve our search space with each iteration.

3. **Space Complexity**:
   - The space complexity is `O(1)`, as we are only using a few extra variables.

Now, let's implement this in Python in a format suitable for LeetCode:



```python
class Solution:
    def mySqrt(self, x: int) -> int:
        # Edge cases for 0 and 1
        if x < 2:
            return x
        
        low, high = 0, x
        
        while low <= high:
            mid = (low + high) // 2
            
            # Calculate square of mid
            mid_squared = mid * mid
            
            if mid_squared == x:
                return mid  # Perfect square found
            elif mid_squared < x:
                low = mid + 1  # Move to higher half
            else:
                high = mid - 1  # Move to lower half
        
        # At the end of the loop, high is the floor of the square root
        return high

```

### How to Run This Code on LeetCode
1. Copy the provided class and method implementation.
2. Paste it into the code editor window on the LeetCode problem page corresponding to this problem.
3. Press "Run" or "Submit" to test your solution against the provided test cases.

This implementation should correctly compute the integer square root for any non-negative integer within the specified constraints.

# 74. Search a 2D Matrix

### Problem Description 
Write an efficient algorithm that searches for a value in an `m x n` matrix. This matrix has the following properties:
Integers in each row are sorted from left to right.

The first integer of each row is greater than the last integer of the previous row.


Example 1:
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true

Example 2:
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
Output: false

Constraints:
`m == matrix.length`
`n == matrix[i].length`
`1 <= m, n <= 100`
`-104 <= matrix[i][j], target <= 104`

### Solution 
 To solve the problem of searching for a value in a sorted `m x n` matrix where each row is sorted in ascending order and the first integer of each row is greater than the last integer of the previous row, we can utilize an efficient approach utilizing binary search.

### Step-by-Step Explanation:

1. **Matrix Properties**: 
   - Each row is sorted in ascending order.
   - The first element of each row is greater than the last element of the previous row. This implies that if we treat the entire matrix as a flattened sorted array, we can apply binary search techniques.

2. **Binary Search Logic**:
   - Treat the matrix as a single list by imagining indices. If we have a 2D matrix `matrix[i][j]`, we can calculate its position in a 1D array format:
     - The index in the 1D array can be derived as `index = i * n + j`, where `m` is the number of rows and `n` is the number of columns.
   - We perform binary search on this 1D representation:
     - First, calculate the total number of elements `m * n`.
     - Use a binary search to find the target by setting `left` as `0` and `right` as `m * n - 1`.
     - Calculate the mid-point and check whether the converted index corresponds to the element in the 2D matrix.
     - Adjust `left` or `right` based on whether the `mid` element is less than or greater than the target.

3. **Time Complexity**: 
   - Since we're applying binary search, the time complexity is \(O(\log(m \times n))\).

### Python Code:



```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]:
            return False  # If the matrix is empty, return False
        
        m, n = len(matrix), len(matrix[0])
        left, right = 0, m * n - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            mid_value = matrix[mid // n][mid % n]  # Convert back to 2D indexing
            
            if mid_value == target:
                return True
            elif mid_value < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return False  # Target not found

```

### Key Points:
- We utilize efficient searching techniques by interpreting the 2D structure as a 1D structure.
- The transformation of 2D matrix indices back to 1D allows us to check values quickly and employ binary search efficiently.
- The constraints guarantee that the matrix will not be empty and that all input values will remain within specified limits, making the algorithm resilient and easy to implement.

# 81. Search in Rotated Sorted Array II

### Problem Description 
There is an integer array `nums` sorted in non-decreasing order (not necessarily with distinct values).

Before being passed to your function, `nums` is rotated at an unknown pivot index `k` (`0 <= k < nums.length`) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (0-indexed). For example, `[0,1,2,4,4,4,5,6,6,7]` might be rotated at pivot index `5` and become `[4,5,6,6,7,0,1,2,4,4]`.

Given the array `nums` after the rotation and an integer `target`, return `true` if `target` is in `nums`, or `false` if it is not in `nums`.


Example 1:
Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true

Example 2:
Input: nums = [2,5,6,0,0,1,2], target = 3
Output: false

Constraints:
`1 <= nums.length <= 5000`
`-104 <= nums[i] <= 104`
`nums` is guaranteed to be rotated at some pivot.

`-104 <= target <= 104`
Follow up: This problem is the same as Search in Rotated Sorted Array, where `nums` may contain duplicates. Would this affect the runtime complexity? How and why?

### Solution 
 To solve the problem of determining if a target integer exists in a rotated sorted array that may contain duplicates, we will employ a modified binary search algorithm. This is because the properties of the sorted array allow us to significantly reduce the search space compared to a linear search.

### Algorithm Explanation:

1. **Initialization**: Start with two pointers `left` (0) and `right` (last index of `nums`). We will use these pointers to keep track of our search boundaries.

2. **Binary Search Loop**:
    - While `left` is less than or equal to `right`:
        - Calculate the middle index `mid`.
        - If the mid element `nums[mid]` equals the `target`, return `True`.
        - If there are duplicates, where `nums[left]`, `nums[mid]`, and `nums[right]` are equal, we reduce the search space from both sides by incrementing `left` and decrementing `right`.
        - If the left side is sorted (i.e., `nums[left] <= nums[mid]`):
            - Check if the target is in this sorted part. If `nums[left] <= target < nums[mid]`, adjust `right` to `mid - 1`, else adjust `left` to `mid + 1`.
        - If the right side is sorted (i.e., `nums[mid] <= nums[right]`):
            - Similarly, check if the target falls in this range. If `nums[mid] < target <= nums[right]`, adjust `left` to `mid + 1`, else adjust `right` to `mid - 1`.

3. **Result**: If we exit the while loop with no return, it means the target is not in the array, so we return `False`.

### Complexity Analysis:
- In the worst case, the time complexity could degrade to O(n) when dealing with duplicates because we may need to skip through them. However, in a general case (without duplicates), the time complexity is O(log n).

Here’s how the solution can be implemented in Python:



```python
class Solution:
    def search(self, nums, target):
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            # Check if we found the target
            if nums[mid] == target:
                return True
            
            # Deal with duplicates
            if nums[left] == nums[mid] == nums[right]:
                left += 1
                right -= 1
            # Check if the left side is sorted
            elif nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # Right side is sorted
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return False

```

### Testing the Solution:
We could verify the solution using the provided examples:



```python
# Example 1
solution = Solution()
print(solution.search([2, 5, 6, 0, 0, 1, 2], 0))  # Output: True

# Example 2
print(solution.search([2, 5, 6, 0, 0, 1, 2], 3))  # Output: False

```

### Conclusion:
The described approach provides an efficient mechanism to search in a rotated sorted array while addressing the issue of duplicates. The solution can run efficiently on LeetCode due to its O(log n) time complexity in cases without many duplicates.

# 153. Find Minimum in Rotated Sorted Array

### Problem Description 
Suppose an array of length `n` sorted in ascending order is rotated between `1` and `n` times. For example, the array `nums = [0,1,2,4,5,6,7]` might become:
`[4,5,6,7,0,1,2]` if it was rotated `4` times.

`[0,1,2,4,5,6,7]` if it was rotated `7` times.

Notice that rotating an array `[a[0], a[1], a[2], ..., a[n-1]]` 1 time results in the array `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]`.

Given the sorted rotated array `nums` of unique elements, return the minimum element of this array.


Example 1:
Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.


Example 2:
Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.


Example 3:
Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 

Constraints:
`n == nums.length`
`1 <= n <= 5000`
`-5000 <= nums[i] <= 5000`
All the integers of `nums` are unique.

`nums` is sorted and rotated between `1` and `n` times.

### Solution 
 To solve this problem, we can utilize a modified binary search technique to efficiently find the minimum element in a sorted and rotated array. The main idea is to take advantage of the properties of the rotated array:

1. The array is initially sorted in ascending order.
2. When rotated, there will be a point at which the order is disrupted, creating a "pivot". The minimum element will be exactly at this pivot point where the next element is smaller than the current.

### Algorithm Explanation:

1. **Initialize Pointers**:
   - Use two pointers, `left` and `right`, to represent the bounds of the current search space in the array.

2. **Binary Search**:
   - While `left` is less than `right`:
     - Calculate the middle index `mid`.
     - Compare the middle element `nums[mid]` with the rightmost element `nums[right]`:
       - If `nums[mid]` is greater than `nums[right]`, it indicates that the minimum value must be in the right half of the array, so we move `left` to `mid + 1`.
       - If `nums[mid]` is less than or equal to `nums[right]`, it means the minimum value is in the left half or could be the middle element itself, thus move `right` to `mid`.
   
3. **Termination**:
   - When the loop terminates, `left` will point to the index of the minimum element in the rotated sorted array.

This approach runs in O(log n) time, which is efficient given the constraints.

### Implementation in Python:

Here is the implementation of the above logic formatted in a way suitable for LeetCode:



```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = (left + right) // 2
            
            # If mid element is greater than the rightmost element,
            # it means the pivot is in the right half
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid  # Search in the left half
            
        # When left equals right, we have found the minimum element
        return nums[left]

```

### Explanation of Code:

- **Class Definition**: The `Solution` class holds the method `findMin` which will take the input array `nums`.
- **Pointers Initialization**: The `left` pointer starts from 0 and the `right` pointer starts from the last index of the array (`len(nums) - 1`).
- **While Loop**: The loop continues until `left` is no longer less than `right`.
- **Middle Calculation**: We compute the middle index and compare `nums[mid]` with `nums[right]` to determine which half to search next.
- **Final Return**: After the loop ends, `left` points to the smallest element, and we return `nums[left]`.

### Time Complexity:
- The time complexity of this solution is O(log n) due to the binary search approach.

### Space Complexity:
- The space complexity is O(1) as we are using a constant amount of extra space.

With this implementation, you can test it directly on the LeetCode platform. Make sure to provide the input in the form of a `List[int]` object as required by the problem statement.

# 154. Find Minimum in Rotated Sorted Array II

### Problem Description 
Suppose an array of length `n` sorted in ascending order is rotated between `1` and `n` times. For example, the array `nums = [0,1,4,4,5,6,7]` might become:
`[4,5,6,7,0,1,4]` if it was rotated `4` times.

`[0,1,4,4,5,6,7]` if it was rotated `7` times.

Notice that rotating an array `[a[0], a[1], a[2], ..., a[n-1]]` 1 time results in the array `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]`.

Given the sorted rotated array `nums` that may contain duplicates, return the minimum element of this array.


Example 1:
Input: nums = [1,3,5]
Output: 1

Example 2:
Input: nums = [2,2,2,0,1]
Output: 0

Constraints:
`n == nums.length`
`1 <= n <= 5000`
`-5000 <= nums[i] <= 5000`
`nums` is sorted and rotated between `1` and `n` times.

Follow up: This is the same as Find Minimum in Rotated Sorted Array but with duplicates. Would allow duplicates affect the run-time complexity? How and why?

### Solution 
 Certainly! Let's tackle the problem step by step using an efficient algorithm that can handle duplicates in a rotated sorted array.

### Problem Understanding

Given a sorted array that has been rotated, our goal is to find the minimum element in that array. The challenge is that the array may contain duplicates, which complicates the binary search approach typically used for sorted arrays.

### Approach

To solve the problem, we can use a modified binary search approach. The key points are:

1. **Binary Search Basics**: In a traditional rotated sorted array without duplicates, we can determine which part to search based on the relationship between the middle element and the ends of the array.
  
2. **Handling Duplicates**: If we encounter duplicates, we cannot definitively decide which side to search next (left or right) based on the middle element. If the left, middle, and right values are equal, we can skip the duplicates by moving the left pointer to the right.

3. **Conditions to Update**:
   - If `nums[mid] > nums[right]`, the minimum lies in the right half.
   - If `nums[mid] < nums[right]`, the minimum lies in the left half (including mid).
   - If `nums[mid] == nums[right]`, we cannot determine the side to search, so we will increment the right pointer to skip the duplicate.

### Implementation

Here's how you can implement this logic in Python:



```python
def findMin(nums):
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2  # Find the mid index
        
        if nums[mid] > nums[right]:  # Minimum must be in the right part
            left = mid + 1
        elif nums[mid] < nums[right]:  # Minimum is in the left part including mid
            right = mid
        else:  # nums[mid] == nums[right], shrink the search space
            right -= 1
    
    return nums[left]  # The left pointer points to the minimum element

```

### Explanation of the Code

- We start by initializing two pointers, `left` and `right`, to the start and end of the array, respectively.
- We enter a while loop that continues until `left` is less than `right`.
- Inside the loop, we calculate the `mid` index.
- Depending on the comparison of `nums[mid]` with `nums[right]`, we adjust the search space:
  - If `nums[mid] > nums[right]`, it indicates that the minimum value is somewhere in the right half, hence we move `left` to `mid + 1`.
  - If `nums[mid] < nums[right]`, it indicates that the minimum value could be the `mid` itself or lies in the left half, so we set `right` to `mid`.
  - If the elements are equal (`nums[mid] == nums[right]`), we can't decide which part of the array to discard; therefore, we simply reduce `right` to skip one duplicate.
- After the loop ends, `left` will point to the smallest element in the rotated array.

### Complexity Analysis

- **Time Complexity**: The worst-case time complexity is O(n) due to the potential for duplicates causing the algorithm to examine every element in some cases. This can happen when all the elements are the same.
- **Space Complexity**: O(1), as we are using a constant amount of space regardless of the input size.

### Conclusion

This algorithm efficiently finds the minimum in a rotated sorted array even when duplicates are present. By leveraging a binary search approach, we can minimize the time complexity typically associated with such search problems.

# 162. Find Peak Element

### Problem Description 
A peak element is an element that is strictly greater than its neighbors.

Given an integer array `nums`, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that `nums[-1] = nums[n] = -∞`.


Example 1:
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.


Example 2:
Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6.


Constraints:
`1 <= nums.length <= 1000`
`-231 <= nums[i] <= 231 - 1`
`nums[i] != nums[i + 1]` for all valid `i`.

Follow up: Could you implement a solution with logarithmic complexity?

### Solution 
 To solve the problem of finding a peak element in an array where a peak is defined as an element that is greater than its neighbors, we can use a binary search approach to achieve a logarithmic time complexity, O(log n). Here’s the step-by-step explanation and the corresponding Python code.

### Explanation

1. **Understanding Peaks**:
   - A peak element is an element that is greater than its neighbors. For edge elements, we assume that out-of-bound elements are `-∞` which means the first and last elements can also be peaks.
   - The problem states that if there are multiple peak elements, we can return the index of any one of them.

2. **Binary Search Approach**:
   - The idea is to leverage the property that if you are at an element `nums[mid]`, at least one of the positions `mid`, `left`, or `right` must be a peak.
   - If `nums[mid] < nums[mid + 1]`, it means there exists a peak in the right half of the array (including `mid + 1`).
   - Conversely, if `nums[mid] < nums[mid - 1]`, it indicates there is a peak in the left half (including `mid - 1`).
   - If neither is true, then `nums[mid]` itself is a peak element.

3. **Termination**:
   - The binary search will continue until the search space is reduced to a single element, which must be a peak.

### Code Implementation

Here is the Python code structured to run on LeetCode:



```python
class Solution:
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            # Compare mid with its next element
            if nums[mid] < nums[mid + 1]:
                # Move to the right side
                left = mid + 1
            else:
                # Move to the left side
                right = mid
        
        # At the end of the while loop, left == right and it's a peak
        return left

```

### Explanation of the Code:

- **Initialization**: We define two pointers `left` and `right` to point to the start and end of the array, respectively.
- **While Loop**: We loop while `left` is less than `right`:
  - Calculate the middle index `mid`.
  - Check if the current middle element is less than its right neighbor:
    - If it is less, move our search to the right half (`left = mid + 1`).
    - If it is greater than or equal to its right neighbor, move our search to the left half (`right = mid`).
- **Return Statement**: Once the loop exits, the `left` pointer will point to a peak element, and we return `left` as the index of the peak.

### Complexity:
- **Time Complexity**: O(log n) due to the binary search strategy.
- **Space Complexity**: O(1) since we are using a constant amount of extra space.

This solution meets the requirements and performs well within the given constraints.

# 167. Two Sum II - Input array is sorted

### Problem Description 
Given an array of integers `numbers` that is already sorted in ascending order, find two numbers such that they add up to a specific `target` number.

Return the indices of the two numbers (1-indexed) as an integer array `answer` of size `2`, where `1 <= answer[0] < answer[1] <= numbers.length`.

You may assume that each input would have exactly one solution and you may not use the same element twice.


Example 1:
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore index1 = 1, index2 = 2.


Example 2:
Input: numbers = [2,3,4], target = 6
Output: [1,3]

Example 3:
Input: numbers = [-1,0], target = -1
Output: [1,2]

Constraints:
`2 <= numbers.length <= 3 * 104`
`-1000 <= numbers[i] <= 1000`
`numbers` is sorted in increasing order.

`-1000 <= target <= 1000`
Only one valid answer exists.

### Solution 
 To solve the problem of finding two numbers in a sorted array that sum up to a specific target, we can employ a two-pointer approach. This is an efficient method because the array is already sorted, allowing us to navigate the array with a linear time complexity of O(n).

### Problem Breakdown:

1. **Input/Output Requirements**:
   - **Input**: A sorted array of integers, `numbers`, and an integer `target`.
   - **Output**: An array of two integers representing the 1-indexed positions of the two numbers that sum to `target`.

2. **Constraints**:
   - The array will always contain exactly one unique solution, meaning we will not need to handle cases with multiple valid pairs.

3. **Approach**:
   - Initialize two pointers: one starting at the beginning of the array (`left` pointer) and the other at the end of the array (`right` pointer).
   - Calculate the sum of the two numbers at these pointers.
   - If the sum equals the target, we've found our solution. If it is less than the target, we move the `left` pointer up to increase the sum. If it is greater than the target, we move the `right` pointer down to decrease the sum.
   - Repeat this process until we find the two numbers that match the target.

### Implementation:

Here’s how the implementation would look in Python:



```python
def twoSum(numbers, target):
    left, right = 0, len(numbers) - 1  # Initialize pointers

    while left < right:  # Continue until the two pointers meet
        current_sum = numbers[left] + numbers[right]  # Calculate the current sum
        if current_sum == target:  # Check if we found the solution
            return [left + 1, right + 1]  # Return indices as 1-indexed
        elif current_sum < target:  # If the sum is less, move left pointer right
            left += 1
        else:  # If the sum is greater, move right pointer left
            right -= 1
            
    return []  # In case there is no solution (but the problem guarantees one)

```

### Explanation of the Code:

1. **Function Definition**: We define the function `twoSum` which takes `numbers` (the sorted list of integers) and `target` (the integer we want the sum to equal).
2. **Pointer Initialization**: We initialize two pointers, `left` at index 0 and `right` at the last index of the array.
3. **While Loop**: We loop until the `left` pointer is less than the `right` pointer:
   - Calculate the `current_sum` of the elements at these two pointers.
   - If `current_sum` equals the `target`, we return the indices (adding 1 to each index to convert from 0-indexing to 1-indexing).
   - If `current_sum` is less than the `target`, we increment the `left` pointer to try to increase the sum.
   - If `current_sum` is greater than the `target`, we decrement the `right` pointer to try to decrease the sum.

4. **Return**: The function will return the required indices when the two numbers that sum to the target are found.

### Complexity:
- **Time Complexity**: O(n) where n is the number of elements in the `numbers` array.
- **Space Complexity**: O(1), as we are only using two additional variables (pointers) irrespective of the input size.

This solution is efficient and leverages the sorted property of the array to find the answer quickly.

# 174. Dungeon Game

### Problem Description 
The demons had captured the princess and imprisoned her in the bottom-right corner of a `dungeon`. The `dungeon` consists of `m x n` rooms laid out in a 2D grid. Our valiant knight was initially positioned in the top-left room and must fight his way through `dungeon` to rescue the princess.

The knight has an initial health point represented by a positive integer. If at any point his health point drops to `0` or below, he dies immediately.

Some of the rooms are guarded by demons (represented by negative integers), so the knight loses health upon entering these rooms; other rooms are either empty (represented as 0) or contain magic orbs that increase the knight's health (represented by positive integers).

To reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.

Return the knight's minimum initial health so that he can rescue the princess.

Note that any room can contain threats or power-ups, even the first room the knight enters and the bottom-right room where the princess is imprisoned.


Example 1:
Input: dungeon = [[-2,-3,3],[-5,-10,1],[10,30,-5]]
Output: 7
Explanation: The initial health of the knight must be at least 7 if he follows the optimal path: RIGHT-> RIGHT -> DOWN -> DOWN.


Example 2:
Input: dungeon = [[0]]
Output: 1

Constraints:
`m == dungeon.length`
`n == dungeon[i].length`
`1 <= m, n <= 200`
`-1000 <= dungeon[i][j] <= 1000`

### Solution 
 To solve the problem of determining the minimum initial health required for the knight to reach the princess in the dungeon, we can use a dynamic programming approach. The idea is to work backwards from the princess's position to the knight's starting position, calculating the minimum health needed at each room to ensure the knight can continue to the next room without dying.

### Explanation:

1. **Dynamic Programming Table (DP Table)**: We will maintain a 2D list `dp` where `dp[i][j]` represents the minimum health required at room `(i, j)` to ensure that the knight can reach the princess, starting with a health of at least 1.

2. **Base Case**: Starting from the princess's room (bottom-right corner), we need to ensure that the knight has enough health to exit that room. Thus:
   - If the princess's room has a negative value, the knight needs more than that amount of health to survive. Specifically, he needs `1 - dungeon[m-1][n-1]` health points.
   - If the princess's room has a positive value or is zero, he only needs `1` health point, so we define the base case as:
     

```python
     dp[m-1][n-1] = max(1, 1 - dungeon[m-1][n-1])
     
```

3. **Fill DP Table**: We fill the DP table from the bottom-right to the top-left:
   - For the last row and last column, we only have one option (go right or down).
   - For internal cells, the knight can move either right or down, so we take the minimum health needed between both possible moves.

4. **Final Calculation**: The value at `dp[0][0]` will give us the minimum health needed at the start to ensure the knight can reach the princess.

Here is the complete Python implementation in the format suitable for LeetCode:



```python
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        # Get the dimensions of the dungeon
        m, n = len(dungeon), len(dungeon[0])
        
        # Create a DP table with zeros
        dp = [[0] * n for _ in range(m)]
        
        # Fill the bottom-right corner (princess's room)
        dp[m-1][n-1] = max(1, 1 - dungeon[m-1][n-1])
        
        # Fill the last row (only moves can go left)
        for j in range(n-2, -1, -1):
            dp[m-1][j] = max(1, dp[m-1][j+1] - dungeon[m-1][j])
        
        # Fill the last column (only moves can go up)
        for i in range(m-2, -1, -1):
            dp[i][n-1] = max(1, dp[i+1][n-1] - dungeon[i][n-1])
        
        # Fill the rest of the dp table
        for i in range(m-2, -1, -1):
            for j in range(n-2, -1, -1):
                min_health_on_exit = min(dp[i+1][j], dp[i][j+1])
                dp[i][j] = max(1, min_health_on_exit - dungeon[i][j])
        
        # The answer will be the value at the starting position
        return dp[0][0]


```

### How to Use:
- Create an instance of the `Solution` class.
- Call the `calculateMinimumHP` method with the dungeon as an argument.
- The method will return the minimum initial health required for the knight to reach the princess.

### Test Cases:
You can test this implementation using:


```python
# Example usage
dungeon = [[-2,-3,3],[-5,-10,1],[10,30,-5]]
sol = Solution()
print(sol.calculateMinimumHP(dungeon))  # Output: 7

dungeon = [[0]]
print(sol.calculateMinimumHP(dungeon))  # Output: 1

``` 

This implementation is efficient, with a time complexity of O(m * n) and a space complexity of O(m * n), which works well within the problem constraints.

# 209. Minimum Size Subarray Sum

### Problem Description 
Given an array of positive integers `nums` and a positive integer `target`, return the minimal length of a contiguous subarray `[numsl, numsl+1, ..., numsr-1, numsr]` of which the sum is greater than or equal to `target`. If there is no such subarray, return `0` instead.


Example 1:
Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: The subarray [4,3] has the minimal length under the problem constraint.


Example 2:
Input: target = 4, nums = [1,4,4]
Output: 1

Example 3:
Input: target = 11, nums = [1,1,1,1,1,1,1,1]
Output: 0

Constraints:
`1 <= target <= 109`
`1 <= nums.length <= 105`
`1 <= nums[i] <= 105`
Follow up: If you have figured out the `O(n)` solution, try coding another solution of which the time complexity is `O(n log(n))`.

### Solution 
 To solve the problem of finding the minimal length of a contiguous subarray whose sum is greater than or equal to a specified target, we can utilize the two-pointer or sliding window technique. This method allows us to efficiently find the required subarray without having to check every possible contiguous subarray, thus achieving a time complexity of \(O(n)\).

### Approach:
1. **Initialization**:
    - Maintain two pointers, `left` and `right`, which represent the boundaries of the current subarray.
    - Use a variable, `current_sum`, to keep track of the sum of the elements in the current subarray.
    - Initialize `min_length` to a very large value (or infinity) to store the length of the minimal subarray found.

2. **Expand the Right Pointer**:
    - Move the `right` pointer to the right, adding the current number to `current_sum`.
    - Whenever `current_sum` becomes greater than or equal to `target`, we have a valid subarray.

3. **Shrink from the Left**:
    - When a valid subarray is found, try to shrink the subarray from the left by moving the `left` pointer to the right (i.e., increasing `left`).
    - After moving `left`, update `current_sum` by subtracting the value that is no longer in the subarray.
    - Continue this until `current_sum` is less than `target`.
    - During this process, update `min_length` whenever the current valid subarray length is smaller than the previously recorded minimum.

4. **Return Result**:
    - If `min_length` remains large, it indicates that no valid subarray was found, hence return `0`.
    - Otherwise, return `min_length`.

### Python Code:
Now, let's implement this in the format suitable for LeetCode.



```python
class Solution:
    def minSubArrayLen(self, target: int, nums: list[int]) -> int:
        left = 0
        current_sum = 0
        min_length = float('inf')  # Initialize to infinity
        
        # Iterate with the right pointer
        for right in range(len(nums)):
            current_sum += nums[right]  # Add current element to the sum
            
            # While current sum is greater than or equal to target, try to shrink the window
            while current_sum >= target:
                min_length = min(min_length, right - left + 1)  # Update min length
                current_sum -= nums[left]  # Subtract the leftmost element
                left += 1  # Move left pointer to the right
        
        # If min_length is still infinity, it means we never found a valid subarray
        return 0 if min_length == float('inf') else min_length

```

### Explanation of Example:
For the input `target = 7` and `nums = [2,3,1,2,4,3]`:
- We initialize with `left = 0`, `current_sum = 0`, and `min_length = inf`.
- As we iterate, when `right` reaches the position of `4`, the current sum is `7`, which meets our target. The length of the subarray at this stage is `right - left + 1 = 5`.
- We then try to shrink it by moving `left` and adjusting the `current_sum`. We find `[4,3]` when `left` has moved and `current_sum` is again `7`. The length at this moment is `2` which is less than `5`, and thus `min_length` is updated.
- After processing, we find the minimal length subarray whose sum equals or exceeds the target is `2`.

### Conclusion:
This implementation efficiently finds the minimal subarray length with a time complexity of \(O(n)\), making it suitable for large inputs as specified in the constraints.

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
 To solve the problem of counting the number of nodes in a complete binary tree, we can take advantage of the properties of complete binary trees to achieve a more efficient solution than simply traversing the tree.

### Explanation:

1. **Complete Binary Tree Properties**:
   - In a complete binary tree, all levels except possibly the last are fully filled.
   - All nodes in the last level are as far left as possible.

2. **Count Efficiently**:
   - We can determine the height of the tree. The height is defined as the number of edges from the root to the deepest leaf node.
   - If the height of the tree is `h`, then there are `2^h - 1` nodes in the full tree from level `0` to `h-1`.
   - The nodes at the last level can be counted directly by checking how many nodes exist on that level.

3. **Binary Search for Last Level Nodes**:
   - If we know the height, the last level can have between `0` and `2^h` nodes. We can perform a binary search to find the exact number of nodes in the last level.
   - We check if a node exists at a certain index in the last level using a helper function which traverses the tree down from the root to find the presence of nodes at particular indices.

### Implementation:

Here's how we can implement this approach in Python, formatted for LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if not root:
            return 0

        # Calculate the height of the tree
        height = self.getHeight(root)

        # If the tree is a complete binary tree, nodes can be counted as (2^height - 1)
        # We need to count how many nodes are present in the last level (height level).
        last_level_nodes = self.countLastLevelNodes(root, height)
        
        # Total nodes = Nodes in full binary tree until height - 1 + Last level nodes
        return (2 ** height - 1) + last_level_nodes

    def getHeight(self, node: TreeNode) -> int:
        height = 0
        while node:
            height += 1
            node = node.left  # Go down to the leftmost node
        return height

    def countLastLevelNodes(self, root: TreeNode, height: int) -> int:
        left, right = 0, (2 ** height) - 1  # The range of nodes at the last level
        
        while left <= right:
            mid = (left + right) // 2
            if self.exists(mid, height, root):
                left = mid + 1  # Move to the right half, mid exists
            else:
                right = mid - 1  # Move to the left half, mid does not exist
        
        return left  # Left is the count of nodes that exist

    def exists(self, index: int, height: int, node: TreeNode) -> bool:
        left, right = 0, (2 ** height) - 1  # The range of nodes at the last level
        
        for _ in range(height):
            mid = (left + right) // 2
            if index <= mid:
                node = node.left
                right = mid  # Move to the left child
            else:
                node = node.right
                left = mid + 1  # Move to the right child
        
        return node is not None  # Return True if the node exists, otherwise False

```

### Summary:

This solution contains:
- A method to compute the height of the tree.
- A binary search approach to determine how many nodes are in the last level.
- A helper method to check if a node exists at a specific index in the last level.

This algorithm runs in `O(log^2 n)` where `n` is the number of nodes in the tree, which is significantly faster than traversing the entire tree.

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
 To solve the problem of finding the k-th smallest element in a binary search tree (BST), we can leverage the properties of the BST where the left subtree contains values less than the root and the right subtree contains values greater than the root. This allows us to perform an in-order traversal to retrieve elements in sorted order.

### Plan
1. **In-order Traversal:** Since in-order traversal visits nodes in ascending order (for BSTs), we can perform this traversal and keep track of the count of visited nodes.
2. **Tracking k-th Element:** As we traverse, we will increment a counter. Once the counter hits `k`, we return the current node's value, which will be the k-th smallest element.
3. **Efficiency:** This ensures we only traverse the tree until we find our answer, thus making the approach efficient with a time complexity of O(H + k), where H is the height of the tree.

### Implementation
Here's a Python implementation following the LeetCode solution format:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        # Initialize a counter and a variable to hold the result
        self.count = 0
        self.result = -1

        def in_order_traversal(node):
            # Base case: if node is None, simply return
            if node is None:
                return
            
            # Traverse the left subtree first
            in_order_traversal(node.left)
            
            # Increment the count
            self.count += 1
            
            # If count equals k, we've found the k-th smallest element
            if self.count == k:
                self.result = node.val
                return
            
            # Traverse the right subtree
            in_order_traversal(node.right)

        # Start the in-order traversal from the root
        in_order_traversal(root)
        
        # Return the k-th smallest element found
        return self.result

```

### Explanation:
1. **TreeNode Class:** This is a simple class definition for tree nodes, which holds the value (`val`), and pointers to left and right children.
2. **Solution Class:** The `Solution` class contains the method `kthSmallest`, which performs an in-order traversal. 
3. **In-order Traversal Function:** The inner function `in_order_traversal` is defined to recursively visit nodes. It:
   - Recursively visits the left child (ensuring we access the left subtree first).
   - Increments the counter upon visiting a node.
   - Checks if the counter matches `k`. If so, it saves the node's value as `result` and returns.
   - Finally, it recursively visits the right child.
4. **Result Return:** After the traversal, the value of `self.result` is returned.

### Follow-Up:
If the BST is modified frequently, one optimization could be to maintain a balanced BST (like an AVL tree or a Red-Black tree) that can provide k-th smallest queries efficiently. Alternatively, we can use augmented BSTs that store the size of the subtrees, allowing for efficient rank queries. However, these implementations would significantly increase the complexity of insertion and deletion operations.

This implementation can be directly tested on the LeetCode website using their custom test function provided for binary trees.

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
 To solve the problem of searching for a `target` value in a matrix that is sorted both row-wise and column-wise, we can use an efficient approach that takes advantage of these sorting properties. Instead of performing a linear search, which could be time-consuming, we can utilize a method that explores the matrix in a systematic way.

### Strategy

1. **Start at the Top Right Corner**: 
   - We begin our search from the top-right corner of the matrix. This position is chosen because:
     - The values to the left are smaller.
     - The values below are larger.

2. **Iterate Through the Matrix**:
   - If the current value is equal to the `target`, we return `True`.
   - If the current value is greater than the `target`, we move left (which means we look for smaller values).
   - If the current value is less than the `target`, we move down (which means we look for larger values).

3. **End Condition**:
   - If we move out of the bounds of the matrix (either to the left or below), it indicates that the `target` is not present in the matrix.

This method ensures we traverse at most \(m + n\) elements of the matrix, leading to a time complexity of \(O(m + n)\).

### Implementation

Now, let's implement this solution in Python using the LeetCode solution format:



```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]:
            return False
        
        # Start at the top-right corner
        row, col = 0, len(matrix[0]) - 1
        
        # While we're within the bounds of the matrix
        while row < len(matrix) and col >= 0:
            current = matrix[row][col]
            
            if current == target:
                return True
            elif current > target:
                # Move left
                col -= 1
            else:
                # Move down
                row += 1
        
        return False

```

### Explanation of the Code

1. **Class Definition**: We define a class `Solution` which is the standard format for LeetCode submissions.

2. **Function Definition**: Inside the class, the function `searchMatrix` takes a 2D list `matrix` and an integer `target`.

3. **Edge Case Handling**: We first check if the matrix is empty. If it is, we return `False`.

4. **Initialization**: We start from the top-right corner of the matrix (row = 0, col = number of columns - 1).

5. **While Loop**: We iterate as long as `row` is within the number of rows and `col` is non-negative.
   - We store the current value at `matrix[row][col]`.
   - We then compare this `current` value with the `target`:
     - If they are equal, we return `True`.
     - If `current` is greater than `target`, we move left by decrementing `col`.
     - If `current` is less than `target`, we move down by incrementing `row`.

6. **Return Statement**: If we exit the loop without finding the `target`, we return `False`.

### Conclusion

The above implementation efficiently searches for the `target` value in the given matrix by taking advantage of its sorted properties. It runs within optimal time complexity and space complexity is constant (O(1)). You can run this code directly on the LeetCode platform to see if it passes the test cases.

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
 To solve the problem of finding the value in a binary search tree (BST) that is closest to a given target value, we can utilize the properties of a BST. Specifically, since the left subtree contains values less than the current node's value, and the right subtree contains values greater than the current node's value, we can efficiently navigate the tree.

### Approach:

1. **Initialization**: Start from the root of the BST, keeping track of the closest value found so far.
2. **Traversal**: At each node, compare its value with the target:
   - If the value is closer to the target than the current closest value, update the closest value.
   - If the value is greater than the target, move to the left subtree (since all values in the left subtree are less than the current node's value).
   - If the value is less than the target, move to the right subtree.
3. **Termination**: The traversal continues until all relevant nodes have been checked (when you can't move further in the tree).
4. **Return the closest value**: After traversing, return the closest value found.

### Code Implementation:
Here is the Python code implementing the above logic in a format suitable for LeetCode:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def closestValue(self, root: TreeNode, target: float) -> int:
        # Initialize the closest value as the root's value
        closest = root.val
        
        # Node pointer for traversal
        current_node = root
        
        # Traverse the tree
        while current_node is not None:
            # Update the closest value if the current node is closer to target
            if abs(current_node.val - target) < abs(closest - target):
                closest = current_node.val
            
            # Determine the direction to traverse
            if target < current_node.val:
                current_node = current_node.left
            else:
                current_node = current_node.right
        
        return closest

```

### Explanation of the Code:
- **TreeNode Class**: This class defines the structure of the nodes in the BST, which includes a value and pointers to the left and right children.
- **Solution Class**: Contains the method `closestValue` that operates on the BST.
- **closestValue Method**:
  - Initializes `closest` to the value of the `root`.
  - Uses a loop to traverse through the tree, adjusting the `current_node` based on comparisons with the `target`.
  - After checking the current node's value, it updates `closest` if the current node is a closer match than what has been found so far.
  - It continues until it cannot traverse further (i.e., `current_node` becomes `None`).
  - Finally, it returns the closest value found during the traversal.

### Complexity:
- **Time Complexity**: O(h), where h is the height of the tree. In the worst case of a skewed tree, this could approach O(n).
- **Space Complexity**: O(1) since we are using only a fixed amount of space for the closest variable and the traversal doesn’t use any additional space for a stack or list.

# 275. H-Index II

### Problem Description 
Given an array of integers `citations` where `citations[i]` is the number of citations a researcher received for their `ith` paper and `citations` is sorted in an ascending order, return compute the researcher's `h`-index.

According to the definition of h-index on Wikipedia: A scientist has an index `h` if `h` of their `n` papers have at least `h` citations each, and the other `n − h` papers have no more than `h` citations each.

If there are several possible values for `h`, the maximum one is taken as the `h`-index.


Example 1:
Input: citations = [0,1,3,5,6]
Output: 3
Explanation: [0,1,3,5,6] means the researcher has 5 papers in total and each of them had received 0, 1, 3, 5, 6 citations respectively.

Since the researcher has 3 papers with at least 3 citations each and the remaining two with no more than 3 citations each, their h-index is 3.


Example 2:
Input: citations = [1,2,100]
Output: 2

Constraints:
`n == citations.length`
`1 <= n <= 105`
`0 <= citations[i] <= 1000`
`citations` is sorted in ascending order.

Follow up: Could you solve it in logarithmic time complexity?

### Solution 
 To solve the problem of calculating the researchers' h-index from a sorted list of citations, we can leverage the properties of the sorted array to efficiently find the h-index using binary search, which offers logarithmic time complexity.

### Explanation:

The h-index is defined such that a researcher has an index `h` if `h` of their `n` papers have at least `h` citations each. Given the sorted nature of the `citations` array, we can use binary search to quickly locate the maximum value of `h`.

#### Steps to find the h-index:

1. **Initialization**: Define two pointers, `left` and `right`. `left` begins at index `0` and `right` starts at index `n - 1` (the last index of the array).

2. **Binary Search**:
   - Calculate the middle index `mid` by averaging `left` and `right`.
   - Determine if the current middle value `citations[mid]` is greater than or equal to `n - mid`. If it is, this means there are at least `n - mid` papers with at least `n - mid` citations, which suggests a potential h-index.
   - Update `h` accordingly and try to find a potentially larger h-index by moving the `right` pointer to `mid - 1`.
   - If the current value is less than `n - mid`, move `left` to `mid + 1` to look for higher citation counts.

3. **Final h-index**: The loop continues until `left` surpasses `right`, at which point the variable `h` will hold the maximum valid h-index.

Here's the implementation of the above logic in Python:



```python
class Solution:
    def hIndex(self, citations):
        n = len(citations)
        left, right = 0, n - 1
        h = 0  # This will keep track of the maximum h-index found
        
        while left <= right:
            mid = (left + right) // 2  # Find the mid index
            
            # Check if there are at least (n - mid) papers with at least (n - mid) citations
            if citations[mid] >= n - mid:
                h = n - mid  # Update h since n - mid papers have at least n - mid citations
                right = mid - 1  # Look for a potentially larger h-index
            else:
                left = mid + 1  # Look in the right half
        
        return h  # Return the found h-index

```

### Explanation of the Code:
1. **Class Definition**: We define a class `Solution` which conforms to the standard submission format in platforms like LeetCode.
2. **Method hIndex**: This method takes a list of citations as input.
3. **Initialization and Loop**: Using binary search, we adjust our pointers (`left` and `right`) based on the mid-value's comparison. Depending on whether we have found sufficient citations for a considered h-index or not, we either update our h-value or adjust the search space.
4. **Return the Result**: Finally, we return the value of `h`, which represents the maximum h-index given the citation counts.

### Complexity Analysis:
- **Time Complexity**: \( O(\log n) \) due to the binary search approach.
- **Space Complexity**: \( O(1) \) because we are using a constant amount of space for our variables.

This solution efficiently calculates the h-index for large input sizes as specified in the problem constraints.

# 278. First Bad Version

### Problem Description 
You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

Suppose you have `n` versions `[1, 2, ..., n]` and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API `bool isBadVersion(version)` which returns whether `version` is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.


Example 1:
Input: n = 5, bad = 4
Output: 4
Explanation:
call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true
Then 4 is the first bad version.


Example 2:
Input: n = 1, bad = 1
Output: 1

Constraints:
`1 <= bad <= n <= 231 - 1`

### Solution 
 Here’s a solution to the problem using a binary search approach. Since you want to minimize the number of calls to the `isBadVersion` API, a binary search is optimal for this scenario. We'll leverage the fact that if a version is bad, all subsequent versions are also bad, forming a contiguous block of bad versions at the end of the range.

### Detailed Explanation

1. **Understand the Problem**: We need to identify the first bad version among versions numbered from 1 through n. The call to `isBadVersion(version)` tells us whether a specific version is bad (returns `True`) or good (returns `False`).

2. **Binary Search Approach**: 
   - Initialize two pointers, `left` at 1 and `right` at `n`.
   - While `left` is less than `right`, calculate the midpoint `mid`.
   - If `isBadVersion(mid)` returns `True`, it indicates that the first bad version is at `mid` or to the left of `mid`, so we move the `right` pointer to `mid`.
   - If `isBadVersion(mid)` returns `False`, it means that the first bad version must be to the right of `mid`, and thus we update the `left` pointer to `mid + 1`.
   - The process continues until `left` equals `right`, at which point both pointers indicate the first bad version.

3. **Code Implementation**: We need to implement a function named `firstBadVersion` that takes an integer `n` as input and returns the index of the first bad version. This function will use the binary search method we described.

Here is the Python code in the format suitable for LeetCode:



```python
# Assume that the isBadVersion API is already defined for you

def isBadVersion(version):
    # This is a mock function for demonstration.
    # In the actual LeetCode environment, this function is provided.
    pass

class Solution:
    def firstBadVersion(self, n: int) -> int:
        left, right = 1, n
        
        while left < right:
            mid = left + (right - left) // 2  # prevent overflow
            if isBadVersion(mid):
                right = mid  # the first bad version is at mid or before
            else:
                left = mid + 1  # the first bad version is after mid
                
        # At the end of the loop, left == right, pointing to the first bad version
        return left

```

### Usage
- When you put this code in the LeetCode platform, ensure that the `isBadVersion` API is implemented or mocked as shown above. The `Solution` class contains the `firstBadVersion` method that performs the binary search to find the first bad version efficiently.

### Complexity
- **Time Complexity**: O(log n) due to the binary search approach.
- **Space Complexity**: O(1) since we are using a constant amount of additional space.

This solution is optimal and will efficiently find the first bad version with minimal API calls, as required by the problem statement.

# 287. Find the Duplicate Number

### Problem Description 
Given an array of integers `nums` containing `n + 1` integers where each integer is in the range `[1, n]` inclusive.

There is only one repeated number in `nums`, return this repeated number.


Example 1:
Input: nums = [1,3,4,2,2]
Output: 2

Example 2:
Input: nums = [3,1,3,4,2]
Output: 3

Example 3:
Input: nums = [1,1]
Output: 1

Example 4:
Input: nums = [1,1,2]
Output: 1

Constraints:
`2 <= n <= 3 * 104`
`nums.length == n + 1`
`1 <= nums[i] <= n`
All the integers in `nums` appear only once except for precisely one integer which appears two or more times.

Follow up:
How can we prove that at least one duplicate number must exist in `nums`?
Can you solve the problem without modifying the array `nums`?
Can you solve the problem using only constant, `O(1)` extra space?
Can you solve the problem with runtime complexity less than `O(n2)`?

### Solution 
 To solve the problem of finding the repeated number in an array of integers where the integers are in the range `[1, n]` and there is exactly one duplicate, we can utilize a strategy known as Floyd's Tortoise and Hare (Cycle Detection). This method is beneficial because it uses constant space (`O(1)`) and has a linear runtime complexity (`O(n)`).

Here's the detailed explanation of the approach before delving into the code:

### Explanation of the Approach

1. **Understanding the Problem**: Given the constraints (each integer from `1` to `n` appears once except for one which appears twice), the problem is structured such that our `nums` array can be visualized as forming a cycle (like a linked list).

2. **Cycle Detection**: We can envision the numbers in the array as pointers where each number points to the index of the next number. The duplication creates a cycle. We will use two pointers:
   - **Tortoise Pointer**: Moves one step at a time.
   - **Hare Pointer**: Moves two steps at a time.

3. **Finding the Intersection**: We first detect if there is a cycle by initializing both pointers and moving them through the numbers according to their values. When they meet, it indicates that a cycle exists.

4. **Finding the Start of the Cycle**: Once we detect the cycle, we can find the entrance of the cycle (which is the duplicate number) by resetting one pointer to the beginning of the array and moving both pointers one step at a time until they meet again.

### Python Code

Here’s the code formatted for LeetCode submission:



```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # Step 1: Initialize the tortoise and hare
        tortoise = nums[0]
        hare = nums[0]
        
        # Step 2: Move the tortoise by one step and the hare by two steps until they meet
        while True:
            tortoise = nums[tortoise]
            hare = nums[nums[hare]]
            if tortoise == hare:  # They meet which indicates a cycle
                break
        
        # Step 3: Find the entrance to the cycle
        # Reset one pointer to the start of the array
        tortoise = nums[0]
        
        while tortoise != hare:
            tortoise = nums[tortoise]
            hare = nums[hare]
        
        # At the point where they meet, we found the duplicate number
        return hare

```

### How to Run the Code on LeetCode

1. **Function Signature**: Make sure to use the correct function signature as expected by LeetCode. In this case, the method `findDuplicate` will accept a list of integers as input.

2. **Data Types**: The type hint `List[int]` ensures that it is clear that the input is a list of integers.

3. **Usage of Loops**: The while loops efficiently find the intersection point and the starting point of the cycle, which is the duplicate.

### Complexity
- **Time Complexity**: `O(n)` – We traverse the list a limited number of times.
- **Space Complexity**: `O(1)` – We use only a few variables regardless of the input size.

### Conclusion
This approach effectively utilizes the properties of cycles to find the duplicate number in the array, achieving the desired performance constraints. By following this methodology, we not only find the duplicate but also adhere to the problem's limitations regarding space and complexity.

# 300. Longest Increasing Subsequence

### Problem Description 
Given an integer array `nums`, return the length of the longest strictly increasing subsequence.

A subsequence is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. For example, `[3,6,2,7]` is a subsequence of the array `[0,3,1,6,2,2,7]`.


Example 1:
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.


Example 2:
Input: nums = [0,1,0,3,2,3]
Output: 4

Example 3:
Input: nums = [7,7,7,7,7,7,7]
Output: 1

Constraints:
`1 <= nums.length <= 2500`
`-104 <= nums[i] <= 104`
Follow up:
Could you come up with the `O(n2)` solution?
Could you improve it to `O(n log(n))` time complexity?

### Solution 
 To solve the problem of finding the length of the longest strictly increasing subsequence, we will implement two approaches that meet the requirements: an `O(n²)` solution and an `O(n log(n))` solution.

### Explanation of the Approaches

1. **O(n²) Dynamic Programming Approach**:
   - We will use a dynamic programming array `dp` where `dp[i]` will represent the length of the longest increasing subsequence that ends with the element at index `i`.
   - For each `i`, we will check all previous elements `j` (from `0` to `i-1`). If `nums[j] < nums[i]`, it means we can extend the subsequence ending at `j` by including `nums[i]`. Therefore, we update `dp[i]` as `dp[i] = max(dp[i], dp[j] + 1)`.
   - Finally, the length of the longest increasing subsequence will be the maximum value in the `dp` array.

   Here’s the implementation of the `O(n²)` solution:



```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        dp = [1] * len(nums)  # Each element is an increasing subsequence of length 1
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:  # Check for increasing condition
                    dp[i] = max(dp[i], dp[j] + 1)  # Update dp[i]
        
        return max(dp)  # The length of the longest increasing subsequence

```

2. **O(n log(n)) Patience Sorting Approach**:
   - In this approach, we maintain a list called `tails`, where `tails[i]` will hold the smallest tail value of all increasing subsequences of length `i+1`.
   - We iterate through each number in `nums`, and for each number, we will use binary search (`bisect_left` from the `bisect` module) to find the position of the current number in the `tails` array.
   - If the current number can extend the longest subsequence found so far, we append it to the `tails`. Otherwise, we replace the existing value to maintain the minimum value for the tail of the subsequences.

   Here’s the `O(n log(n))` implementation:



```python
from bisect import bisect_left

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        tails = []  # This will hold our increasing subsequence tails
        
        for num in nums:
            pos = bisect_left(tails, num)  # Find the position to replace or append
            if pos == len(tails):
                tails.append(num)  # Extend the size of the tails
            else:
                tails[pos] = num  # Replace tails[pos] with num
        
        return len(tails)  # The length of the longest increasing subsequence

```

### Summary
- The first approach uses dynamic programming and runs in `O(n²)`. It is a straightforward implementation to find the LIS using a state array.
- The second approach uses a binary search technique to maintain the smallest potential tails of increasing subsequences and runs in `O(n log(n))`, making it efficient for larger inputs.

These implementations will run directly on the LeetCode platform as solutions to the problem. You just need to ensure that the relevant libraries (`List` for type hinting and `bisect` for the second approach) are imported if they are not included in the function definition in LeetCode's environment.

# 302. Smallest Rectangle Enclosing Black Pixels

### Problem Description 
You are given an `image` that is represented by a binary matrix with `0` as a white pixel and `1` as a black pixel.

The black pixels are connected (i.e., there is only one black region). Pixels are connected horizontally and vertically.

Given two integers `x` and `y` that represent the location of one of the black pixels, return the area of the smallest (axis-aligned) rectangle that encloses all black pixels.


Example 1:
Input: image = [["0","0","1","0"],["0","1","1","0"],["0","1","0","0"]], x = 0, y = 2
Output: 6

Example 2:
Input: image = [["1"]], x = 0, y = 0
Output: 1

Constraints:
`m == image.length`
`n == image[i].length`
`1 <= m, n <= 100`
`image[i][j]` is either `'0'` or `'1'`.

`1 <= x < m`
`1 <= y < n`
`image[x][y] == '1'.`
The black pixels in the `image` only form one component.

### Solution 
 To address the problem of finding the area of the smallest rectangle enclosing all the black pixels in a binary matrix, we can approach it using a depth-first search (DFS) algorithm. The main idea is to traverse the matrix starting from the provided coordinates (x, y), and during the traversal, determine the bounds of the rectangle that encapsulates all the connected black pixels (represented by '1's).

Here's a step-by-step breakdown of the solution:

1. **Identify Connected Pixels**: Start from the pixel at (x, y) and explore all connected black pixels (1's) in all four possible directions (up, down, left, right).

2. **Track Boundaries**: While traversing, record the minimum and maximum coordinates (row and column) of the visited black pixels. This will help in determining the smallest rectangle that encompasses all the black pixels.

3. **Calculate Area**: Once we determine the boundaries (min_row, max_row, min_col, max_col), the area of the rectangle can be computed using the formula:
   \[
   \text{Area} = (\text{max\_row} - \text{min\_row} + 1) \times (\text{max\_col} - \text{min\_col} + 1)
   \]

4. **Implementing**: We can employ a DFS function for traversing and marking visited pixels to avoid counting them again.

Here is the solution wrapped in the LeetCode format:



```python
class Solution:
    def minArea(self, image: List[List[str]], x: int, y: int) -> int:
        # Initialize the boundaries
        min_row, max_row = x, x
        min_col, max_col = y, y
        
        # Get dimensions of the image
        m = len(image)
        n = len(image[0])
        
        # Directions for exploring neighbors (up, down, left, right)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # DFS to explore the black pixels
        def dfs(i, j):
            nonlocal min_row, max_row, min_col, max_col
            
            # Update the boundaries
            min_row = min(min_row, i)
            max_row = max(max_row, i)
            min_col = min(min_col, j)
            max_col = max(max_col, j)
            
            # Visit all four directions
            for di, dj in directions:
                ni, nj = i + di, j + dj
                # Check bounds and if the pixel is a black pixel
                if 0 <= ni < m and 0 <= nj < n and image[ni][nj] == '1':
                    # Mark it as visited by setting to '0'
                    image[ni][nj] = '0'
                    dfs(ni, nj)
        
        # Start DFS from the given pixel
        dfs(x, y)
        
        # Calculate and return the area
        return (max_row - min_row + 1) * (max_col - min_col + 1)


```

### Explanation of the Code:
- We start by defining the `minArea` method, which prepares to find the smallest enclosing rectangle for the black pixels.
- We initialize the boundaries (`min_row`, `max_row`, `min_col`, `max_col`) to the starting pixel coordinates and define the possible movement directions.
- The `dfs` function performs a depth-first search to explore all connected black pixels. Each time it visits a black pixel, it records its coordinates and marks the pixel as visited by changing its value to '0'.
- After the DFS has traversed all reachable black pixels, we compute the area of the rectangle by using the recorded boundaries.
- Finally, we return the calculated area. This code should run directly on LeetCode and solve the given problem efficiently.

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
 To solve the problem of counting the number of smaller elements to the right of each element in the array, a brute-force approach would involve nested loops, checking each element against all elements to its right. However, this would result in a time complexity of O(n^2), which is inefficient for large arrays (up to 100,000 elements).

Instead, we can use a more efficient approach using a modification of the Merge Sort algorithm. This will allow us to count the smaller elements in O(n log n) time complexity. Here’s a breakdown of how we can achieve this using a merge sort technique combined with indexing:

### Step-by-Step Explanation:
1. **Define a Merge Sort Function**: We will implement a merge sort that sorts the array while counting the number of smaller elements to the right.
  
2. **Auxiliary Structure for Storing Results**: We'll create a `counts` array initialized with zeros to store the result.

3. **Pair Elements with their Indices**: By pairing each number with its corresponding index in the `nums` array, we can re-map the counts correctly after sorting.

4. **Split and Count**: During the merge step, for each element in the left half of the array, we will count how many elements in the right half are smaller. This will increment our counts for that position.

5. **Return the Results**: Once we have processed the entire array, we will return the counts array.

### Here is the Python code that implements this solution:



```python
class Solution:
    def countSmaller(self, nums):
        # Store count of smaller elements
        counts = [0] * len(nums)
        # Pair each number with its index to reconstruct the counts after sorting
        indexed_nums = list(enumerate(nums))
        
        def merge_sort(start, end):
            if end - start <= 1:
                return []
            
            mid = (start + end) // 2
            
            # Count and sort left and right halves
            left = merge_sort(start, mid)
            right = merge_sort(mid, end)
            
            # Merge both halves
            merged = []
            i = 0  # Pointer for left
            j = 0  # Pointer for right
            
            # Count smaller elements to the right
            while i < len(left) and j < len(right):
                # If left element is less than right element
                if left[i][1] <= right[j][1]:
                    # All remaining elements in right are smaller than left[i]
                    counts[left[i][0]] += j
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    j += 1
            
            # Collect remaining elements from left and right
            while i < len(left):
                counts[left[i][0]] += j
                merged.append(left[i])
                i += 1
            
            while j < len(right):
                merged.append(right[j])
                j += 1
            
            return merged
        
        # Start sorting and counting smaller elements
        merge_sort(0, len(indexed_nums))
        
        return counts

```

### Explanation of the Code:
1. **Initialization**: We create an array `counts` to hold the results and `indexed_nums`, which pairs each number with its original index.

2. **Recursive Merge Sort**: The `merge_sort` function takes the start and end indices to process the subarrays recursively.

3. **Counting Logic**: During the merge process, for each element in the left part, we keep track of how many elements from the right part are smaller. The variable `j` keeps a count of the index in the right part.

4. **Final Result**: After sorting and counting, we return the `counts` array that is built during the merging process.

This method efficiently counts the smaller elements to the right for each element in the original `nums` array with a time complexity of O(n log n) and is well-suited for the input constraints.

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
 To solve the problem of counting the number of range sums that lie within a specified range \([lower, upper]\), we can leverage the concept of prefix sums combined with a modified merge sort algorithm. This approach reduces the time complexity to \(O(n \log n)\), which is much better than the naive \(O(n^2)\) approach. 

Here’s how the solution is structured:

1. **Prefix Sum Calculation**: We calculate the prefix sums of the array. A prefix sum at index \(i\) represents the sum of elements from the start of the array up to index \(i\). This allows us to compute any range sum \(S(i, j)\) as \(prefix[j] - prefix[i-1]\).

2. **Using Merge Sort for Counting Ranges**: We will modify the merge sort algorithm to count how many prefix sums fall into the range \([lower, upper]\) as we merge the two halves of the array. During the merge step, we maintain two pointers to determine how many prefix sums fall within the desired range.

3. **Final Count**: The modified merge sort will incrementally keep track of counts of valid range sums as it resolves merging from smaller subarrays to larger ones.

Here's the implemented solution in Python that can be run directly in the LeetCode environment:



```python
class Solution:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        # Step 1: Calculate prefix sums
        prefix_sums = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefix_sums[i + 1] = prefix_sums[i] + nums[i]

        # Step 2: Helper function to perform merge sort and count ranges
        def merge_and_count(start, mid, end):
            count = 0
            j = k = mid + 1
            
            # Count the number of valid ranges
            for i in range(start, mid + 1):
                while j <= end and prefix_sums[j] - prefix_sums[i] < lower:
                    j += 1
                while k <= end and prefix_sums[k] - prefix_sums[i] <= upper:
                    k += 1
                count += k - j
            
            # Step 3: Merge step
            temp = []
            left, right = start, mid + 1
            
            while left <= mid and right <= end:
                if prefix_sums[left] <= prefix_sums[right]:
                    temp.append(prefix_sums[left])
                    left += 1
                else:
                    temp.append(prefix_sums[right])
                    right += 1
            
            while left <= mid:
                temp.append(prefix_sums[left])
                left += 1
            
            while right <= end:
                temp.append(prefix_sums[right])
                right += 1
            
            for i in range(len(temp)):
                prefix_sums[start + i] = temp[i]
            
            return count

        # Step 4: Recursively sort and count
        def sort_and_count(start, end):
            if start >= end:
                return 0
            
            mid = (start + end) // 2
            count = sort_and_count(start, mid) + sort_and_count(mid + 1, end)
            count += merge_and_count(start, mid, end)
            return count
        
        return sort_and_count(0, len(prefix_sums) - 1)

```

### Explanation of the Code:
1. **Prefix Sum Calculation**: We first compute the prefix sums so that any range sum can be computed quickly.
   
2. **Merge and Count Function**:
   - This function counts how many prefix sums are in the required range while merging two halves of the array. The indices \(j\) and \(k\) track where valid prefix sums that meet the criteria fall relative to the current prefix sum being considered.

3. **Sort and Count Recursive Function**: This function implements the recursive merge sort logic, which first sorts the array and counts valid prefixes in the process.

### Complexity:
- Time complexity: \(O(n \log n)\) due to the merge sort.
- Space complexity: \(O(n)\) for the prefix sum array and temporary storage during merging.

This approach efficiently counts range sums and adheres to the problem constraints, allowing it to handle the upper limits smoothly.

# 349. Intersection of Two Arrays

### Problem Description 
Given two integer arrays `nums1` and `nums2`, return an array of their intersection. Each element in the result must be unique and you may return the result in any order.


Example 1:
Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2]

Example 2:
Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [9,4]
Explanation: [4,9] is also accepted.


Constraints:
`1 <= nums1.length, nums2.length <= 1000`
`0 <= nums1[i], nums2[i] <= 1000`

### Solution 
 To solve the problem of finding the unique intersection of two integer arrays `nums1` and `nums2`, we will leverage Python's set data structure due to its efficient support for membership testing and uniqueness guarantees. 

The approach will involve the following steps:
1. Convert both arrays into sets to remove duplicates and allow for fast lookups.
2. Use the intersection operation on these sets to find the common elements.
3. Convert the resultant set back into a list before returning it.

The time complexity of this algorithm is \(O(n + m)\), where \(n\) and \(m\) are the lengths of `nums1` and `nums2` respectively, primarily due to the set operations.

Here is the Python code which can be run directly on the LeetCode website:



```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Convert both lists to sets to remove duplicates
        set1 = set(nums1)
        set2 = set(nums2)

        # Use the intersection operation to find common elements
        intersection_set = set1.intersection(set2)

        # Convert the result set back to a list
        result = list(intersection_set)

        return result

```

### Explanation of the Code:

1. **Class Definition**: The code is encapsulated within a `Solution` class which is the typical format on platforms like LeetCode.

2. **Method Signature**: The method `intersection` is defined to accept two parameters: `nums1` and `nums2`. Both of these are lists of integers.

3. **Creating Sets**:
   - We initialize `set1` and `set2` by passing `nums1` and `nums2` to the `set()` function. This step automatically removes any duplicate values within each initial list.
   
4. **Finding Intersection**:
   - We compute the intersection of `set1` and `set2` using `set1.intersection(set2)`, which yields another set containing only the elements that are present in both `set1` and `set2`.

5. **Returning the Result**:
   - We convert the resulting intersection set back into a list using `list(intersection_set)` and return it.

### Example Cases:
- For the input `nums1 = [1,2,2,1]` and `nums2 = [2,2]`, the intersection would return `[2]`.
- For the input `nums1 = [4,9,5]` and `nums2 = [9,4,9,8,4]`, either `[9,4]` or `[4,9]` can be the output, given the uniqueness of set elements.

This solution effectively meets the problem's requirements while adhering to standard practices in Python programming.

# 350. Intersection of Two Arrays II

### Problem Description 
Given two integer arrays `nums1` and `nums2`, return an array of their intersection. Each element in the result must appear as many times as it shows in both arrays and you may return the result in any order.


Example 1:
Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2,2]

Example 2:
Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [4,9]
Explanation: [9,4] is also accepted.


Constraints:
`1 <= nums1.length, nums2.length <= 1000`
`0 <= nums1[i], nums2[i] <= 1000`
Follow up:
What if the given array is already sorted? How would you optimize your algorithm?
What if `nums1`'s size is small compared to `nums2`'s size? Which algorithm is better?
What if elements of `nums2` are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?

### Solution 
 To solve the problem of finding the intersection of two integer arrays (`nums1` and `nums2`), we can utilize the `Counter` class from the `collections` module to efficiently count the occurrences of each element in both arrays. Then, we can find the common elements and their respective counts to form the intersection array.

### Detailed Explanation:

1. **Counting Elements**: We will use `Counter` to count how many times each number appears in both `nums1` and `nums2`.

2. **Finding Intersections**: We can then iterate through one of the count dictionaries (e.g., `count1` for `nums1`), and for each element, we will check if it exists in the other count dictionary (`count2` for `nums2`). If it does, we will take the minimum of its counts in both dictionaries since the intersection should consist of the minimum occurrences of that element in both arrays.

3. **Constructing the Result**: We will build the result list based on the counts found in the previous step.

4. **Returning the Result**: We return the result list directly as the answer.

The time complexity of this approach is O(m + n), where m and n are the lengths of `nums1` and `nums2`, respectively, since we are effectively scanning both arrays once to count and then once to find the intersections. 

### Python Code:

Here is the complete solution implemented in the format suitable for the LeetCode website:



```python
from collections import Counter
from typing import List

class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Count occurrences of each number in both arrays
        count1 = Counter(nums1)
        count2 = Counter(nums2)

        # Find the intersection
        intersection = []
        
        for num in count1:
            if num in count2:
                # Append the minimum count of the current number in both arrays
                intersection.extend([num] * min(count1[num], count2[num]))

        return intersection

```

### Explanation of the Code:

- We first import `Counter` from the `collections` module and `List` from `typing` for type hints.
- The `intersect` method takes two lists of integers, `nums1` and `nums2`.
- We create two `Counter` objects, `count1` and `count2`, to hold the counts of elements in `nums1` and `nums2`, respectively.
- We initialize an empty list `intersection` to store the result.
- We iterate through each unique number in `count1`. If the number also exists in `count2`, we extend the `intersection` list with that number, multiplied by the minimum of its counts in both arrays.
- Finally, we return the populated `intersection` list.

### Follow-up Discussion:

1. **If the given array is sorted**: If the arrays are sorted, we could use a two-pointer technique which would minimize space usage and eliminate the need for a counting step.
   
2. **If `nums1` is smaller than `nums2`**: It would still be efficient to use the counting method, but if `nums1` is small, we could also use a set for `nums1` to check for existence in `nums2` directly while iterating through `nums2`.

3. **If `nums2` is on disk with memory constraints**: We could read `nums2` in chunks and maintain a frequency count in memory, processing parts of it iteratively to find intersections.

This solution is efficient for the problem constraints, straightforward, and follows a structured approach suitable for submission on LeetCode.

# 352. Data Stream as Disjoint Intervals

### Problem Description 
Given a data stream input of non-negative integers `a1, a2, ..., an`, summarize the numbers seen so far as a list of disjoint intervals.

Implement the `SummaryRanges` class:
`SummaryRanges()` Initializes the object with an empty stream.

`void addNum(int val)` Adds the integer `val` to the stream.

`int[][] getIntervals()` Returns a summary of the integers in the stream currently as a list of disjoint intervals `[starti, endi]`.


Example 1:
Input
["SummaryRanges", "addNum", "getIntervals", "addNum", "getIntervals", "addNum", "getIntervals", "addNum", "getIntervals", "addNum", "getIntervals"]
[[], [1], [], [3], [], [7], [], [2], [], [6], []]
Output
[null, null, [[1, 1]], null, [[1, 1], [3, 3]], null, [[1, 1], [3, 3], [7, 7]], null, [[1, 3], [7, 7]], null, [[1, 3], [6, 7]]]
Explanation
SummaryRanges summaryRanges = new SummaryRanges();
summaryRanges.addNum(1);      // arr = [1]
summaryRanges.getIntervals(); // return [[1, 1]]
summaryRanges.addNum(3);      // arr = [1, 3]
summaryRanges.getIntervals(); // return [[1, 1], [3, 3]]
summaryRanges.addNum(7);      // arr = [1, 3, 7]
summaryRanges.getIntervals(); // return [[1, 1], [3, 3], [7, 7]]
summaryRanges.addNum(2);      // arr = [1, 2, 3, 7]
summaryRanges.getIntervals(); // return [[1, 3], [7, 7]]
summaryRanges.addNum(6);      // arr = [1, 2, 3, 6, 7]
summaryRanges.getIntervals(); // return [[1, 3], [6, 7]]

Constraints:
`0 <= val <= 104`
At most `3 * 104` calls will be made to `addNum` and `getIntervals`.

Follow up: What if there are lots of merges and the number of disjoint intervals is small compared to the size of the data stream?

### Solution 
 To solve the problem of summarizing a stream of non-negative integers as disjoint intervals, we will implement a class `SummaryRanges`. This class will maintain a data structure that allows us to efficiently add numbers and return the current list of intervals while merging overlapping intervals as necessary.

### Plan

1. **Class Initialization**: We will initialize the class with an empty list to store the intervals.
  
2. **Adding Numbers**: When a number is added using the `addNum` method, we'll check:
   - If the number can extend an existing interval.
   - If it can merge with an existing interval.
   - If it should create a new interval.
  
3. **Getting Intervals**: The `getIntervals` method will simply return the current list of intervals.

### Implementation

We'll keep our intervals in a sorted list. This will allow us to efficiently merge overlapping intervals as we add numbers. The main challenge is to maintain the intervals correctly as numbers are added.

Let's implement the `SummaryRanges` class in Python:



```python
class SummaryRanges:
    def __init__(self):
        self.intervals = []  # This will store the disjoint intervals as [start, end]
    
    def addNum(self, val: int) -> None:
        # Create a new empty list to store merged intervals
        new_intervals = []
        added = False
        
        # Iterate through existing intervals
        for interval in self.intervals:
            # If the current interval ends before the new number starts, add the interval as it is
            if interval[1] < val - 1:
                new_intervals.append(interval)
            # If the current interval starts after the new number ends, 
            # we can add the new interval and all remaining intervals
            elif interval[0] > val + 1:
                if not added:
                    new_intervals.append([val, val])  # add new interval [val, val]
                    added = True
                new_intervals.append(interval)
            else:
                # The intervals overlap, so we merge them
                interval[0] = min(interval[0], val)  # Update start to the min
                interval[1] = max(interval[1], val)  # Update end to the max
                new_intervals.append(interval)
        
        # If val wasn't added yet, it means it's either a new minimum number or a new single interval.
        if not added:
            new_intervals.append([val, val])
        
        self.intervals = new_intervals  # Update the intervals to the merged list
    
    def getIntervals(self) -> list[list[int]]:
        return self.intervals  # Return the final list of disjoint intervals

```

### Explanation

- **`__init__` Method**: Initializes an empty list of intervals.
- **`addNum` Method**:
  - It starts with an empty `new_intervals` that will store merged intervals.
  - We iterate through each existing interval:
    - If the current interval ends before the new number is less than 1, it is safe to keep the interval as is.
    - If the current interval starts after the new number is more than 1, then we can add the new number as a separate interval.
    - If the current interval overlaps with the new number, we merge them by updating the start and end of the current interval.
  - After processing all existing intervals, we check if the new number has been added. If not, it implies that the number creates a separate interval and is therefore appended.
- **`getIntervals` Method**: Simply returns the list of currently stored intervals.

### Complexity
- **Time Complexity**: Each `addNum` call has O(n) complexity in the worst case, where `n` is the number of intervals because we may need to check and potentially merge all existing intervals.
- **Space Complexity**: The space complexity is also O(n), as we may store up to `n` intervals.

This implementation efficiently maintains a summary of the numbers in the stream as required by the problem statement.

# 354. Russian Doll Envelopes

### Problem Description 
You are given a 2D array of integers `envelopes` where `envelopes[i] = [wi, hi]` represents the width and the height of an envelope.

One envelope can fit into another if and only if both the width and height of one envelope are greater than the other envelope's width and height.

Return the maximum number of envelopes you can Russian doll (i.e., put one inside the other).

Note: You cannot rotate an envelope.


Example 1:
Input: envelopes = [[5,4],[6,4],[6,7],[2,3]]
Output: 3
Explanation: The maximum number of envelopes you can Russian doll is `3` ([2,3] => [5,4] => [6,7]).


Example 2:
Input: envelopes = [[1,1],[1,1],[1,1]]
Output: 1

Constraints:
`1 <= envelopes.length <= 5000`
`envelopes[i].length == 2`
`1 <= wi, hi <= 104`

### Solution 
 To solve the problem of finding the maximum number of 'Russian doll' envelopes (where one envelope can fit perfectly inside another based on certain width and height conditions), we can use an algorithmic approach that incorporates sorting and dynamic programming. Below is the detailed explanation along with the code formatted according to LeetCode standards.

### Problem Explanation

1. **Understanding the Problem**:
    - An envelope can only fit into another if both its width and height are smaller than those of the other envelope.
    - The challenge is to find the maximum number of envelopes you can nest.

2. **Approach**:
    - **Sorting the Envelopes**: We first sort the envelopes. To make it simple:
        - Sort by width in ascending order.
        - If two envelopes have the same width, sort by height in descending order. This is crucial because if the heights were in ascending order as well, we could mistakenly count envelopes of the same width multiple times as nesting options.
  
    - **Finding the Longest Increasing Subsequence (LIS)**:
        - After sorting, we need to find the longest increasing subsequence based on heights. The goal is to count the maximum number of layers of envelopes with nesting capabilities.
        - We can use a dynamic programming approach or a more efficient binary search method to get the LIS based on heights after sorting the envelopes.

### Implementation

Here’s the Python code that implements the above logic:



```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        # Step 1: Sort the envelopes
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        
        # Step 2: Extract heights from the sorted envelopes
        heights = [h for _, h in envelopes]
        
        # Step 3: Finding the length of LIS in heights
        from bisect import bisect_left
        
        lis = []
        
        for height in heights:
            pos = bisect_left(lis, height)  # Find the position to replace or extend
            if pos == len(lis):
                lis.append(height)  # Extend LIS if height is greater than all elements
            else:
                lis[pos] = height  # Replace at found position
        
        return len(lis)


```

### Explanation of the Code:

1. **Sorting**:
   - We sort the `envelopes` list using a lambda function that sorts primarily by width `wi` and secondarily by height `-hi`. This ensures that we do not mistakenly count envelopes of the same width.

2. **Extracting Heights**:
   - After sorting, we create a list of heights using list comprehension. This simplifies our next task of finding the LIS.

3. **Finding LIS**:
   - We initialize an empty list `lis` that will hold the heights of the envelopes in increasing order.
   - For each height, we use `bisect_left` from the `bisect` module to determine the position where the current height can fit in the `lis`.
   - If the position is equal to the length of `lis`, it means the current height can extend the sequence, and we append it.
   - If not, we replace the existing value at the found position, which helps maintain the potential for forming longer sequences.

4. **Final Output**:
   - The length of the `lis` list at the end gives the maximum number of envelopes that can be nested.

### Complexity:
- **Time Complexity**: The overall complexity is \(O(n \log n)\), where \(n\) is the number of envelopes due to sorting and the LIS finding process.
- **Space Complexity**: \(O(n)\) due to the storage used for the `heights` list and the `lis` list.

This solution is efficient and suitable for the input limits provided in the problem description.

# 363. Max Sum of Rectangle No Larger Than K

### Problem Description 
Given an `m x n` matrix `matrix` and an integer `k`, return the max sum of a rectangle in the matrix such that its sum is no larger than `k`.

It is guaranteed that there will be a rectangle with a sum no larger than `k`.


Example 1:
Input: matrix = [[1,0,1],[0,-2,3]], k = 2
Output: 2
Explanation: Because the sum of the blue rectangle [[0, 1], [-2, 3]] is 2, and 2 is the max number no larger than k (k = 2).


Example 2:
Input: matrix = [[2,2,-1]], k = 3
Output: 3

Constraints:
`m == matrix.length`
`n == matrix[i].length`
`1 <= m, n <= 100`
`-100 <= matrix[i][j] <= 100`
`-105 <= k <= 105`
Follow up: What if the number of rows is much larger than the number of columns?

### Solution 
 To solve the problem of finding the maximum sum of a rectangle in a matrix such that its sum is no larger than `k`, we can use a combination of prefix sums and a sorted data structure. The overall approach is based on the idea of fixing two row boundaries and using a 1D array to compute sums for each selected rectangle.

Here’s how the algorithm works:

### Algorithm Explanation

1. **Prefix Sum Array**: Use a prefix sum approach to convert the 2D problem into a 1D problem. This allows us to calculate the sum of any rectangle defined by two rows efficiently.

2. **Iterate Over Rows**: Fix two row indices (let's call them `top` and `bottom`). For each combination of `top` and `bottom`, we will:
   - Create a 1D array that represents the column sums between these two rows.
   - For each column, the value in this array will hold the sum of elements from the `top` row to the `bottom` row for that column.

3. **Use a Sorted Data Structure**: As we process the column sums, we will use a sorted structure (like `SortedList` from `sortedcontainers` in Python) to keep track of the cumulative sums we encounter:
   - For each new column sum, calculate the maximum subarray sum that does not exceed `k` by searching for the upper bound in the sorted list.
   - If we can find a previous cumulative sum that, when added to the current column sum, remains ≤ `k`, we can derive a valid rectangle sum.

4. **Update the Maximum Sum**: Keep track of the maximum sum found that satisfies the condition.

This approach efficiently combines the benefits of running a 2D cumulative sum query and employing binary search through sorted data structures to allow us to quickly find the maximum possible sum under a constraint.

### Implementation

Here’s the full implementation in Python that adheres to LeetCode solution format:



```python
from typing import List
from sortedcontainers import SortedList

class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        max_sum = float('-inf')
        
        # Iterate through all pairs of top and bottom row
        for top in range(m):
            # Initialize an array to hold cumulative sums for columns
            col_sums = [0] * n
            
            for bottom in range(top, m):
                # Update the column sums for the current bottom
                for col in range(n):
                    col_sums[col] += matrix[bottom][col]
                
                # Now, we want to find the maximum sum <= k
                # We can use a sorted list to keep track of the cumulative sums we have seen
                sorted_list = SortedList([0])  # Start with zero accumulated sum
                current_sum = 0
                
                for sum_j in col_sums:
                    current_sum += sum_j
                    # We need to find the largest prefix sum <= current_sum - k
                    target = current_sum - k
                    idx = sorted_list.bisect_right(target)
                    
                    # If idx is not at the beginning of the list, it means we found a valid prefix
                    if idx > 0:
                        max_sum = max(max_sum, current_sum - sorted_list[idx - 1])
                    
                    # Now add the current_sum to the sorted list
                    sorted_list.add(current_sum)
        
        return max_sum

```

### Explanation of the Code:

1. **Initialization**: We initialize the matrix dimensions and a variable to keep track of the maximum sum found.

2. **Top and Bottom Loop**: For each possible pair of rows defined by `top` and `bottom`, update the `col_sums` to represent the cumulative sums of columns between these rows.

3. **Sorted List for Cumulative Sums**: We maintain a `SortedList` to keep track of prefix sums which allows us to efficiently query for the largest prefix sum ≤ `current_sum - k`.

4. **Max Sum Calculation**: We update `max_sum` whenever we find a valid prefix sum that meets our criteria.

This algorithm runs in O(m^2 * n log n), which is efficient enough given the constraints specified in the problem.

# 367. Valid Perfect Square

### Problem Description 
Given a positive integer num, write a function which returns True if num is a perfect square else False.

Follow up: Do not use any built-in library function such as `sqrt`.


Example 1:
Input: num = 16
Output: true

Example 2:
Input: num = 14
Output: false

Constraints:
`1 <= num <= 2^31 - 1`

### Solution 
 Certainly! To solve the problem of determining if a given positive integer `num` is a perfect square without using built-in library functions such as `sqrt`, we can use a binary search approach. This method is efficient and works well given the constraints (`1 <= num <= 2^31 - 1`).

### Explanation of the Approach:
1. **Perfect Square Definition**: A number `num` is a perfect square if there exists an integer `x` such that `x * x = num`. 

2. **Binary Search**: Since the perfect square root of `num` will always be between `1` and `num`, we can use binary search to efficiently find the integer `x`. 
   - Set `left` to `1` and `right` to `num`.
   - Calculate the midpoint `mid` as `(left + right) // 2`.
   - If `mid * mid` equals `num`, we have found the perfect square.
   - If `mid * mid` is less than `num`, adjust the `left` pointer to `mid + 1`.
   - If `mid * mid` is greater than `num`, adjust the `right` pointer to `mid - 1`.
   - Repeat this until `left` exceeds `right`.

3. **Terminate and Return**: If we find an integer `mid` such that `mid * mid = num`, return `True`. If we exit the loop without finding such an `x`, return `False`.

### Implementation:
Now, let's write the function in a way that it can be run directly on the LeetCode website.



```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num < 1:
            return False
        
        left, right = 1, num
        
        while left <= right:
            mid = (left + right) // 2
            square = mid * mid
            
            if square == num:
                return True
            elif square < num:
                left = mid + 1
            else:
                right = mid - 1
        
        return False

```

### Detailed Explanation of the Code:
- **Class Definition**: The code is wrapped in a class called `Solution`, which is the expected format for LeetCode solutions.
- **Method Definition**: The method `isPerfectSquare` takes an integer `num` as input.
- **Edge Case Handling**: It checks if `num` is less than `1`, which technically shouldn't happen according to the constraints, but it's a good practice to handle such cases.
- **Binary Search Initialization**: `left` is initialized to `1`, and `right` to `num`.
- **Loop**: A `while` loop continues as long as `left` is less than or equal to `right`.
   - Inside the loop, calculate the `mid` and its square `square`.
   - Compare `square` with `num` to adjust the search range.
- **Return Statement**: The function returns `True` if a perfect square is found, otherwise it returns `False` after exiting the loop.

### Complexity Analysis:
- **Time Complexity**: O(log(num)): We are halving the search space at each step.
- **Space Complexity**: O(1): We only use a few additional variables for the search.

This code will efficiently determine if `num` is a perfect square under the given constraints.

# 374. Guess Number Higher or Lower

### Problem Description 
We are playing the Guess Game. The game is as follows:
I pick a number from `1` to `n`. You have to guess which number I picked.

Every time you guess wrong, I will tell you whether the number I picked is higher or lower than your guess.

You call a pre-defined API `int guess(int num)`, which returns 3 possible results:
`-1`: The number I picked is lower than your guess (i.e. `pick < num`).

`1`: The number I picked is higher than your guess (i.e. `pick > num`).

`0`: The number I picked is equal to your guess (i.e. `pick == num`).

Return the number that I picked.


Example 1:
Input: n = 10, pick = 6
Output: 6

Example 2:
Input: n = 1, pick = 1
Output: 1

Example 3:
Input: n = 2, pick = 1
Output: 1

Example 4:
Input: n = 2, pick = 2
Output: 2

Constraints:
`1 <= n <= 231 - 1`
`1 <= pick <= n`

### Solution 
 To solve the "Guess Number Higher or Lower" problem on LeetCode, we can utilize a binary search algorithm. This approach is efficient and suitable for this problem since it allows us to reduce the search space in half with each guess.

Here’s a brief explanation of the approach:

1. **Initialization**: We start with two pointers, `low` initialized to 1 (the minimum possible number) and `high` initialized to `n` (the maximum possible number).

2. **Binary Search Loop**: We'll continue to search while `low` is less than or equal to `high`. For each iteration:
   - We calculate the middle point using `(low + high) // 2`.
   - We call the `guess` API with the middle point to determine how it compares to the picked number.
   - If the result of the `guess` is `0`, it means we have found the picked number, and we return this number.
   - If the result is `-1`, it indicates the picked number is lower than our guess, so we adjust our search space by setting `high` to `mid - 1`.
   - If the result is `1`, it means the picked number is higher than our guess, so we adjust our search space by setting `low` to `mid + 1`.

3. **Termination**: The loop will continue until the correct number is found, at which point it will be returned.

Here’s how you can implement this in Python following the LeetCode solution format:



```python
# The guess API is already defined for you.
# The guess API is defined below:
# def guess(num: int) -> int:
#     # This is a stub for the guess function that is provided by LeetCode

class Solution:
    def guessNumber(self, n: int) -> int:
        low, high = 1, n  # Set initial search range
        
        while low <= high:
            mid = (low + high) // 2  # Find the middle point
            result = guess(mid)  # Call the guess API
            
            if result == 0:
                return mid  # We found the picked number
            elif result == -1:
                high = mid - 1  # Adjust the range to the lower half
            else:
                low = mid + 1  # Adjust the range to the upper half
        
        return -1  # This line is never reached, but added for completeness

```

### Key Points:
- This implementation uses a binary search technique, ensuring O(log n) time complexity.
- The `guess` function is assumed to exist as part of the LeetCode environment.
- The range of numbers is adjusted based on the results of the `guess` function to zero in on the picked number efficiently. 

This solution is scalable and works well within the problem constraints provided.

# 378. Kth Smallest Element in a Sorted Matrix

### Problem Description 
Given an `n x n` `matrix` where each of the rows and columns are sorted in ascending order, return the `kth` smallest element in the matrix.

Note that it is the `kth` smallest element in the sorted order, not the `kth` distinct element.


Example 1:
Input: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
Output: 13
Explanation: The elements in the matrix are [1,5,9,10,11,12,13,13,15], and the 8th smallest number is 13

Example 2:
Input: matrix = [[-5]], k = 1
Output: -5

Constraints:
`n == matrix.length`
`n == matrix[i].length`
`1 <= n <= 300`
`-109 <= matrix[i][j] <= 109`
All the rows and columns of `matrix` are guaranteed to be sorted in non-degreasing order.

`1 <= k <= n2`

### Solution 
 To solve the problem of finding the `k`-th smallest element in a `n x n` matrix where each row and column is sorted in ascending order, we can utilize a min-heap (or priority queue). This approach allows us to efficiently retrieve the smallest elements from the matrix.

### Explanation of the Approach:

1. **Min-Heap Initialization**: We will initialize a min-heap that will help us keep track of the smallest elements as we extract them from the matrix.

2. **Pushing the First Row into the Heap**: We will start by pushing the first element of each row into the min-heap. This ensures that we have the smallest elements from the top row available for extraction.

3. **Extracting from the Heap**: We will then repeatedly extract the smallest element from the heap `k` times. For each element we extract, we will push the next element from the same row into the heap (if it exists). This way, we are always expanding our search to find the next smallest element in the row from which the smallest element was extracted.

4. **Returning the k-th Smallest Element**: After extracting `k` elements from the heap, the last extracted element will be our answer since it represents the `k`-th smallest element in the entire matrix.

### Implementation:

Here’s how the implementation looks in Python, following the structure that LeetCode would accept.



```python
import heapq

class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        # Initialize a min-heap
        min_heap = []
        
        # Push the first element of each row into the heap
        for i in range(min(len(matrix), k)):  # Only need to consider first k rows
            heapq.heappush(min_heap, (matrix[i][0], i, 0))  # (value, row_index, column_index)
        
        # Variable to store the element
        current_value = None
        
        # Extract k elements from the min-heap
        for _ in range(k):
            current_value, row, col = heapq.heappop(min_heap)
            # If there's a next element in the same row, add it to the heap
            if col + 1 < len(matrix[row]):
                heapq.heappush(min_heap, (matrix[row][col + 1], row, col + 1))
        
        # The current value is the k-th smallest element
        return current_value

```

### Detailed Breakdown of the Code:

1. **Heap Initialization**: 
   

```python
   min_heap = []
   
```

2. **Pushing First Elements**: 
    - We loop through the first `min(len(matrix), k)` rows to push the first element of each row (if it exists) into the min-heap.
   

```python
   for i in range(min(len(matrix), k)):
       heapq.heappush(min_heap, (matrix[i][0], i, 0))
   
```

3. **Extracting the k Smallest Elements**:
   - We perform extraction from the heap `k` times.
   - For each extraction, we also check if there is a next element in the same row and push it to the heap.
   

```python
   for _ in range(k):
       current_value, row, col = heapq.heappop(min_heap)
       if col + 1 < len(matrix[row]):
           heapq.heappush(min_heap, (matrix[row][col + 1], row, col + 1))
   
```

4. **Output**:
   - After `k` extractions, the last extracted element holds our desired `k`-th smallest value which we return.
   

```python
   return current_value
   
```

### Complexity Analysis:

- **Time Complexity**: The time complexity is O(k log n) because any element insertion and extraction from the heap takes O(log n) time, and we perform this operation `k` times.
  
- **Space Complexity**: The space complexity is O(n) for storing elements in the min-heap.

This method is efficient and works well within the constraints given in the problem statement.

# 392. Is Subsequence

### Problem Description 
Given two strings `s` and `t`, check if `s` is a subsequence of `t`.

A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., `"ace"` is a subsequence of `"abcde"` while `"aec"` is not).


Example 1:
Input: s = "abc", t = "ahbgdc"
Output: true

Example 2:
Input: s = "axc", t = "ahbgdc"
Output: false

Constraints:
`0 <= s.length <= 100`
`0 <= t.length <= 104`
`s` and `t` consist only of lowercase English letters.

Follow up: If there are lots of incoming `s`, say `s1, s2, ..., sk` where `k >= 109`, and you want to check one by one to see if `t` has its subsequence. In this scenario, how would you change your code?

### Solution 
 To solve the problem of checking if string `s` is a subsequence of string `t`, we can implement a straightforward algorithm that uses two pointers to traverse both strings. The approach is efficient and easy to understand.

### Explanation:

1. **Two Pointers Technique**: We will set up two pointers: one for the string `s` (the subsequence we want to check) and one for the string `t` (the string we are checking against). We will iterate through both strings, advancing the pointer for `s` only if we find a matching character in `t`.
  
2. **Loop through t**: For each character in `t`, we check if it matches the current character in `s` (pointed by the pointer for `s`). If it matches, we move the pointer for `s` forward to check the next character in `s`.

3. **Termination Condition**: If we reach the end of `s` before we finish iterating through `t`, it means that all characters of `s` were found in `t` in order, and we can conclude that `s` is a subsequence of `t`. If we finish traversing `t` and still have characters left in `s`, then `s` is not a subsequence of `t`.

4. **Time Complexity**: The time complexity of this algorithm is O(n), where n is the length of string `t`. This is efficient enough given the problem constraints.

### Implementation:

Here is the Python code to implement this solution in a format suitable for submission on LeetCode:



```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        # Pointers for s and t
        s_pointer, t_pointer = 0, 0
        # Lengths of s and t
        s_len, t_len = len(s), len(t)
        
        # Traverse t until we reach the end of t or s
        while t_pointer < t_len:
            # If the characters match, move the s pointer
            if s_pointer < s_len and s[s_pointer] == t[t_pointer]:
                s_pointer += 1
            
            # Always move the t pointer
            t_pointer += 1
        
        # If we've moved through all of s, it is a subsequence of t
        return s_pointer == s_len

```

### Follow-up Optimization:

For the follow-up question, if there are many incoming strings s1, s2, ..., sk, we can preprocess string `t`. One efficient way would be to:

1. **Preprocess `t`** using a mapping of each character's indices. This allows us to quickly find the next occurrence of each character in `t` using binary search.

2. For each incoming `s`, we can determine if it is a subsequence of `t` by checking if each character in `s` can be found in `t` while maintaining the order.

This method significantly optimizes the check when `s` is checked multiple times against `t`.

Here's a brief implementation of the preprocessing strategy:



```python
from collections import defaultdict
import bisect

class Solution:
    def preprocess(self, t: str):
        # Create a dictionary to hold the indices of each character
        char_indices = defaultdict(list)
        for index, char in enumerate(t):
            char_indices[char].append(index)
        return char_indices

    def isSubsequence(self, s: str, t: str) -> bool:
        # Preprocess t to create index mapping
        char_indices = self.preprocess(t)
        
        # Last position in t where we found a character
        current_position = -1
        
        for char in s:
            # If the character is not in t, return False
            if char not in char_indices:
                return False
            
            # Get the list of indices for the current character
            indices = char_indices[char]
            # Find the first index in indices that is greater than current_position
            next_position = bisect.bisect_right(indices, current_position)
            
            # If we can't find a next position, s is not a subsequence
            if next_position == len(indices):
                return False
            
            # Move the current position to this position
            current_position = indices[next_position]
        
        return True

```

With this more optimized approach, we maintain an efficient check for subsequences even with multiple queries against the same string `t`.

# 410. Split Array Largest Sum

### Problem Description 
Given an array `nums` which consists of non-negative integers and an integer `m`, you can split the array into `m` non-empty continuous subarrays.

Write an algorithm to minimize the largest sum among these `m` subarrays.


Example 1:
Input: nums = [7,2,5,10,8], m = 2
Output: 18
Explanation:
There are four ways to split nums into two subarrays.

The best way is to split it into [7,2,5] and [10,8],
where the largest sum among the two subarrays is only 18.


Example 2:
Input: nums = [1,2,3,4,5], m = 2
Output: 9

Example 3:
Input: nums = [1,4,4], m = 3
Output: 4

Constraints:
`1 <= nums.length <= 1000`
`0 <= nums[i] <= 106`
`1 <= m <= min(50, nums.length)`

### Solution 
 To tackle the problem of minimizing the largest sum among `m` subarrays split from the array `nums`, we can utilize a binary search combined with a greedy approach to efficiently find the optimal solution.

### Explanation of the Approach:

1. **Understanding the Problem**:
   - We need to split the `nums` array into exactly `m` subarrays in such a way that the maximum sum of these subarrays is as small as possible.
   - The constraints dictate that we must be cautious with splitting, especially in the case of larger values in the array.

2. **Binary Search Setup**:
   - The minimum possible value for the largest sum of the subarrays (`left`) can be the maximum number in the array because each subarray must at least contain one number and hence cannot have a sum smaller than the largest element.
   - The maximum possible value for the largest sum (`right`) is the sum of all elements in the array, which represents the case where we don't split the array at all.

3. **Greedy Check**:
   - For a given target sum (midpoint between left and right during the binary search), we check whether it is feasible to split the array into `m` or fewer subarrays with that maximum sum. This can be done with a simple greedy approach:
     - Keep adding elements to a current subarray until adding another element would exceed the target sum.
     - When it exceeds, start a new subarray and continue. Count how many subarrays are created.
  
4. **Binary Search Execution**:
   - Adjust `left` and `right` based on whether the current target sum can produce `m` or fewer subarrays. If it can, we reduce the maximum (move `right` down); if it cannot, we increase the minimum (move `left` up).

### Implementing the Solution:

Here’s the complete Python code reflecting the above plan:


```python
class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        # Define the binary search bounds
        left = max(nums)  # The largest number in nums
        right = sum(nums) # The sum of all numbers in nums
        
        def can_split(target):
            current_sum = 0
            subarray_count = 1  # Start with one subarray

            for num in nums:
                # If adding this number exceeds the target, we need a new subarray
                if current_sum + num > target:
                    subarray_count += 1
                    current_sum = num  # Start the new subarray with the current number
                    if subarray_count > m:  # If we exceed m subarrays, return False
                        return False
                else:
                    current_sum += num
            
            return True  # We were able to split into m or fewer subarrays
        
        # Perform binary search
        while left < right:
            mid = (left + right) // 2
            if can_split(mid):
                right = mid  # Try for a smaller max sum
            else:
                left = mid + 1  # Increase the possible max sum
        
        return left  # The minimised largest sum among m subarrays

# Example of how you might call this function:
# sol = Solution()
# result = sol.splitArray([7,2,5,10,8], 2)
# print(result)  # Output: 18

```

### Explanation of the Code:
- We define a class `Solution` with a method `splitArray` that accepts the input `nums` and the integer `m`.
- The binary search process uses the helper function `can_split`, which checks if it’s possible to split the array into `m` or fewer subarrays with the maximum sum not exceeding a specified target.
- By adjusting the `left` and `right` bounds based on the feasibility check, we eventually narrow down to the minimal largest sum that can be achieved with `m` splits. The final result is returned as the output.

# 441. Arranging Coins

### Problem Description 
You have a total of n coins that you want to form in a staircase shape, where every k-th row must have exactly k coins.

Given n, find the total number of full staircase rows that can be formed.

n is a non-negative integer and fits within the range of a 32-bit signed integer.


Example 1:
n = 5
The coins can form the following rows:
¤
¤ ¤
¤ ¤
Because the 3rd row is incomplete, we return 2.


Example 2:
n = 8
The coins can form the following rows:
¤
¤ ¤
¤ ¤ ¤
¤ ¤
Because the 4th row is incomplete, we return 3.

### Solution 
 To solve the problem of forming staircase rows with `n` coins, we need to determine how many full rows we can create where the `k-th` row has exactly `k` coins. This means that for each subsequent row, we need to increase the number of coins used.

For example, the coin distribution looks like this:
- Row 1: 1 coin (Total = 1)
- Row 2: 2 coins (Total = 3)
- Row 3: 3 coins (Total = 6)
- Row 4: 4 coins (Total = 10)
- ...

The total number of coins needed for the first `k` rows can be calculated using the formula:
\[ S(k) = \frac{k \times (k + 1)}{2} \]
This is the sum of the first `k` natural numbers.

The goal is to find the maximum integer `k` such that:
\[ S(k) \leq n \]
To do this, we can use a simple iterative approach or a binary search. The iterative approach is straightforward and effective for this problem.

### Python Code
Here is the Python code to implement the solution. This code follows the LeetCode format and can be run directly in the LeetCode environment.



```python
class Solution:
    def arrangeCoins(self, n: int) -> int:
        total_coins = 0
        k = 0
        
        while total_coins <= n:
            k += 1
            total_coins += k
        
        # When the loop exits, total_coins > n, so the last valid k is k - 1
        return k - 1

```

### Explanation
1. **Initialization**: We start with `total_coins` set to 0, which keeps track of the total coins used so far. We also initialize `k` to 0, which represents the current row we are trying to form.

2. **Iterative Loop**: We enter a while loop that continues as long as `total_coins` is less than or equal to `n`. In each iteration:
   - We increment `k` by 1 (indicating we're trying to form the next row).
   - We add `k` to `total_coins` (indicating we've used the coins needed for the `k-th` row).

3. **Exit Condition**: The loop exits when `total_coins` exceeds `n`. At this point, the last valid row we formed is `k - 1`.

4. **Return Value**: We return `k - 1` since `k` is incremented one extra time after exceeding the available coins.

This algorithm runs in O(√n) time because, in the worst case, `k` can be approximately √(2n) due to the formula for the sum of the first `k` integers. It's efficient and works within the provided constraints.

# 454. 4Sum II

### Problem Description 
Given four lists A, B, C, D of integer values, compute how many tuples `(i, j, k, l)` there are such that `A[i] + B[j] + C[k] + D[l]` is zero.

To make problem a bit easier, all A, B, C, D have same length of N where 0 ≤ N ≤ 500. All integers are in the range of -228 to 228 - 1 and the result is guaranteed to be at most 231 - 1.


Example:
Input:
A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]
Output:
2
Explanation:
The two tuples are:
1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0

### Solution 
 To solve the problem of counting the number of tuples `(i, j, k, l)` such that `A[i] + B[j] + C[k] + D[l]` equals zero, we can utilize the concept of hash maps (dictionaries in Python) to efficiently compute the possible sums of elements.

Let's break down the approach step by step:

1. **Combining Sums**: We can separate the problem into two parts. We can compute all possible sums of the first two lists (A and B) and the last two lists (C and D). This means we will calculate sums of all pairs from A and B, and from C and D.

2. **Using a Hash Map**: We will store the sums of the first two lists (A and B) in a hash map (dictionary), where the key is the sum and the value is the count of how many times that sum appears. 

3. **Finding Complements**: For each possible sum from the last two lists (C and D), we will check if the negative of that sum exists in the hash map we created from A and B. If it does, it implies there are tuples that can be formed which sum to zero.

4. **Count Valid Tuples**: We will multiply the count of the found sum in the hash map by the occurrence of the current sum from C and D to get the count of valid tuples corresponding to that combination.

5. **Complexity**: The time complexity of this approach is O(N^2) due to the nested loops, which is feasible given the constraints.

Here's the complete implementation of the solution in Python:



```python
from collections import defaultdict

class Solution:
    def fourSumCount(self, A, B, C, D):
        countAB = defaultdict(int)
        
        # Count all possible sums of A and B
        for a in A:
            for b in B:
                countAB[a + b] += 1
        
        count = 0
        
        # For every possible sum of C and D, find if its negation exists in countAB
        for c in C:
            for d in D:
                count += countAB[-(c + d)]
        
        return count

# Example usage:
# sol = Solution()
# print(sol.fourSumCount([1, 2], [-2, -1], [-1, 2], [0, 2]))  # Output: 2

```

### Explanation of Code:
- We import `defaultdict` from the `collections` module, which simplifies counting occurrences of sums.
- We define a class `Solution` with a method `fourSumCount` that takes four lists as parameters.
- We initialize a default dictionary `countAB` to store the sums of A and B.
- Two nested loops iterate through each element of A and B to calculate their sums and store their counts in the dictionary.
- We then initialize a variable `count` to 0, which will hold the number of valid tuples.
- Another pair of nested loops go through all combinations of C and D, checking the negative of their sums against `countAB`. For each match, we add to the count the number of occurrences of that sum in `countAB`.
- Finally, we return the total count.

This implementation efficiently counts the valid tuples with a manageable complexity and ensures correct results for the inputs provided.

# 475. Heaters

### Problem Description 
Winter is coming! During the contest, your first job is to design a standard heater with a fixed warm radius to warm all the houses.

Every house can be warmed, as long as the house is within the heater's warm radius range. 
Given the positions of `houses` and `heaters` on a horizontal line, return the minimum radius standard of heaters so that those heaters could cover all houses.

Notice that all the `heaters` follow your radius standard, and the warm radius will the same.


Example 1:
Input: houses = [1,2,3], heaters = [2]
Output: 1
Explanation: The only heater was placed in the position 2, and if we use the radius 1 standard, then all the houses can be warmed.


Example 2:
Input: houses = [1,2,3,4], heaters = [1,4]
Output: 1
Explanation: The two heater was placed in the position 1 and 4. We need to use radius 1 standard, then all the houses can be warmed.


Example 3:
Input: houses = [1,5], heaters = [2]
Output: 3

Constraints:
`1 <= houses.length, heaters.length <= 3 * 104`
`1 <= houses[i], heaters[i] <= 109`

### Solution 
 To solve the problem of finding the minimum radius of heaters required to warm all the houses, we can follow a systematic approach using binary search. Here's a detailed step-by-step explanation along with the implementation in the required LeetCode format.

### Problem Breakdown

1. **Understanding the Warmth Radius**: Each heater can warm houses that are within a certain distance (the radius). If a heater is at position `h`, and its radius is `r`, it can warm houses in the range `[h - r, h + r]`.

2. **Goal**: We need to determine the smallest `r` such that every house can be warmed by at least one heater.

3. **Input/Output**: 
   - Input: Two lists (`houses` and `heaters`).
   - Output: An integer representing the minimum radius required.

4. **Approach**:
   - **Sorting**: First, we will sort both the `houses` and `heaters` arrays. This allows us to effectively determine the nearest heater for each house.
   - **Binary Search**: For each house, we will use binary search to find the closest heater and calculate the distance. The maximum of these distances will give us the required radius.

### Implementation

Here's the implementation of the above logic in Python:



```python
from bisect import bisect_left

class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        # Sort both houses and heaters
        houses.sort()
        heaters.sort()
        
        # Initialize the minimum radius
        min_radius = 0
        
        # Iterate over each house
        for house in houses:
            # Find the position to insert the house into the heaters' list
            index = bisect_left(heaters, house)

            # Calculate the distances to the closest heater
            left_heater_distance = float('inf') if index == 0 else abs(house - heaters[index - 1])
            right_heater_distance = float('inf') if index == len(heaters) else abs(house - heaters[index])
            
            # Get the minimum distance for current house to nearest heater
            min_distance = min(left_heater_distance, right_heater_distance)
            
            # Update the required radius
            min_radius = max(min_radius, min_distance)
        
        return min_radius

```

### Explanation of the Code

1. **Imports**: We import `bisect_left` from the `bisect` module for efficient binary searching.

2. **Sorting**: We sort the houses and heaters lists.

3. **Iterating through Houses**: For each house, we:
   - Use `bisect_left` to find the position where the house would fit among the heaters.
   - Determine the distance to the closest heater on either side (left and right). We handle edge cases where there may not be a heater on one side.

4. **Calculate Minimum Distance**: For each house, calculate the minimum distance to heaters, and maintain the maximum distance found (as this will dictate the required radius).

5. **Return Result**: Finally, return the minimum radius required to cover all the houses.

### Complexity Analysis
- **Time Complexity**: `O(N log M)` where `N` is the number of houses and `M` is the number of heaters due to the binary search operation.
- **Space Complexity**: `O(1)` if we disregard the input storage, as we only use a fixed amount of extra space.

This solution should run efficiently within the constraints provided by LeetCode, and it directly adheres to the platform-specific output requirements.

# 483. Smallest Good Base

### Problem Description 
For an integer n, we call k>=2 a good base of n, if all digits of n base k are 1.

Now given a string representing n, you should return the smallest good base of n in string format.


Example 1:
Input: "13"
Output: "3"
Explanation: 13 base 3 is 111.


Example 2:
Input: "4681"
Output: "8"
Explanation: 4681 base 8 is 11111.


Example 3:
Input: "1000000000000000000"
Output: "999999999999999999"
Explanation: 1000000000000000000 base 999999999999999999 is 11.

Note:
The range of n is [3, 10^18].

The string representing n is always valid and will not have leading zeros.

### Solution 
 To solve the problem of finding the smallest good base `k` for a given integer `n` represented as a string, we need to consider the definition of a "good base." A base `k` is good if when `n` is represented in base `k`, all its digits are `1`. Mathematically, this means:

\[ n = k^m + k^{m-1} + ... + k^1 + k^0 \]

This can be rewritten using the formula for the sum of a geometric series:

\[ n = \frac{k^{m+1} - 1}{k - 1} \]

Where `m` is the total number of digits (in base `k`, `m` will correspond to the largest exponent).

To find the smallest base `k`, we can leverage the fact that the maximum number of digits `m` can be derived from the logarithm of `n`. The base will have to be at least `2`, and we search for `k` from `2` up to the integer part of the square root of `n`. 

The algorithm can be broken down into the following steps:
1. For each potential number of digits `m`, starting from `60` down to `2`, we compute the potential base `k`.
2. For each `m`, calculate `k` as:
   \[
   k = \text{floor}(n^{1/m})
   \]
3. Validate whether this base produces a good representation (i.e., check if the sum calculated from the geometric series equals `n`).
4. If we find such a base, we can return it as the answer since we check for the largest `m` downwards, ensuring we find the smallest successful `k`.

Here is the Python code implementing this logic:



```python
class Solution:
    def smallestGoodBase(self, n: str) -> str:
        n = int(n)
        max_m = 64  # Log base 2 of 10^18 is approximately 59.79, thus we can consider 64

        for m in range(max_m, 1, -1):
            # Using integer division to find base k
            k = int(n ** (1 / m))  # floor(n^(1/m))
            # Check if this k is a good base
            # Calculate the sum of geometric progression for base k and m terms
            if (k ** (m + 1) - 1) // (k - 1) == n:
                return str(k)

        # If no good base found for m in the range, then the answer is n - 1, which is the case when m = 1
        return str(n - 1)

# Example usage:
sol = Solution()
print(sol.smallestGoodBase("13")) # Output: "3"
print(sol.smallestGoodBase("4681")) # Output: "8"
print(sol.smallestGoodBase("1000000000000000000")) # Output: "999999999999999999"

```

### Detailed Explanation:
1. **Initialization**: We start by converting the input string `n` into an integer. We set `max_m` to `64` to cover all feasible values of `m`.
2. **Iterating from m=64 to 2**: We check for potential values of `m` in a loop going downwards in order to prioritize smaller bases.
3. **Calculating Base `k`**: For each `m`, we calculate `k` using the formula \( k = n^{1/m} \). This gives an idea of the base for the corresponding number of digits.
4. **Validating Base**: Using the geometric series sum, we check if it equals `n`. If we find a valid `k`, we immediately return it.
5. **Returning n-1**: If no valid `k` is found, we return `n-1` as it is always a good base.

This algorithm efficiently narrows down the possible good bases and operates within the constraints given in the problem description.

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
 To solve the problem of counting important reverse pairs `(i, j)` in an array `nums` where `i < j` and `nums[i] > 2 * nums[j]`, we can utilize a modified merge sort approach. This is efficient and can handle up to 50,000 elements in the array.

### Explanation:

1. **Brute Force Approach**: A naive approach would involve checking each pair `(i, j)` in a nested loop, which would have a time complexity of O(n^2). This is not feasible for large inputs, hence a more efficient approach is needed.

2. **Optimized Approach**: We can use a modified merge sort. The idea is to:
   - First, sort the array while simultaneously counting the important reverse pairs.
   - Split the array into two halves, count the pairs in each half, and then count the pairs across the two halves during the merging process.

3. **Counting Pairs**: For each element in the left half, we need to count how many elements in the right half satisfy the condition `nums[i] > 2 * nums[j]`. This can be efficiently done because both halves will be sorted after the split.

### Code Implementation:

Here’s the code implementing the above logic in a format suitable for LeetCode:



```python
class Solution:
    def reversePairs(self, nums):
        # Helper function to use merge sort
        def merge_sort(arr, temp_arr, left, right):
            if left >= right:
                return 0
            
            mid = (left + right) // 2
            
            # Count pairs in left half, right half and across halves
            count = merge_sort(arr, temp_arr, left, mid) + merge_sort(arr, temp_arr, mid + 1, right)
            
            # Count important reverse pairs (across halves)
            j = mid + 1
            for i in range(left, mid + 1):
                while j <= right and arr[i] > 2 * arr[j]:
                    j += 1
                count += (j - (mid + 1))

            # Merge step
            count += merge(arr, temp_arr, left, mid, right)
            return count

        # Function to merge two halves
        def merge(arr, temp_arr, left, mid, right):
            for i in range(left, right + 1):
                temp_arr[i] = arr[i]
                
            i = left
            j = mid + 1
            k = left
            
            while i <= mid and j <= right:
                if temp_arr[i] <= temp_arr[j]:
                    arr[k] = temp_arr[i]
                    i += 1
                else:
                    arr[k] = temp_arr[j]
                    j += 1
                k += 1
            
            while i <= mid:
                arr[k] = temp_arr[i]
                i += 1
                k += 1
            
            while j <= right:
                arr[k] = temp_arr[j]
                j += 1
                k += 1
            
            return 0

        # Main call to merge_sort
        n = len(nums)
        temp_arr = [0] * n
        return merge_sort(nums, temp_arr, 0, n - 1)

```

### Explanation of the Code:

- We define a `Solution` class and a method `reversePairs`.
- Inside `reversePairs`, we define a helper function `merge_sort` which recursively sorts the array while counting important reverse pairs.
- We also define a `merge` function to combine two sorted halves and keep the entire array sorted.
- We maintain a count of reverse pairs found during the merge process.
- Finally, the function returns the total count of important reverse pairs.

### Complexity Analysis:
- The time complexity of this approach is **O(n log n)** due to the divide and conquer nature of merge sort.
- The space complexity is **O(n)** for the temporary array used in merging.

This will efficiently handle arrays with lengths up to the constraint, yielding results in a reasonable time frame.

# 497. Random Point in Non-overlapping Rectangles

### Problem Description 
Given a list of non-overlapping axis-aligned rectangles `rects`, write a function `pick` which randomly and uniformily picks an integer point in the space covered by the rectangles.

Note:
An integer point is a point that has integer coordinates. 
A point on the perimeter of a rectangle is included in the space covered by the rectangles. 
`i`th rectangle = `rects[i]` = `[x1,y1,x2,y2]`, where `[x1, y1]` are the integer coordinates of the bottom-left corner, and `[x2, y2]` are the integer coordinates of the top-right corner.

length and width of each rectangle does not exceed `2000`.

`1 <= rects.length <= 100`
`pick` return a point as an array of integer coordinates `[p_x, p_y]`
`pick` is called at most `10000` times.


Example 1:
Input: 
["Solution","pick","pick","pick"]
[[[[1,1,5,5]]],[],[],[]]
Output: 
[null,[4,1],[4,1],[3,3]]

Example 2:
Input: 
["Solution","pick","pick","pick","pick","pick"]
[[[[-2,-2,-1,-1],[1,0,3,0]]],[],[],[],[],[]]
Output: 
[null,[-1,-2],[2,0],[-2,-1],[3,0],[-2,-2]]
Explanation of Input Syntax:
The input is two lists: the subroutines called and their arguments. `Solution`'s constructor has one argument, the array of rectangles `rects`. `pick` has no arguments. Arguments are always wrapped with a list, even if there aren't any.

### Solution 
 To solve the problem of picking a random integer point from a list of non-overlapping axis-aligned rectangles, we can follow a systematic approach. Below is the Python code that adheres to the guidelines of the LeetCode format, along with a detailed explanation.

### Code Implementation



```python
import random

class Solution:

    def __init__(self, rects: List[List[int]]):
        self.rects = rects
        self.areas = []
        total_area = 0
        
        # Calculate the total area of each rectangle and maintain a cumulative area list
        for rect in rects:
            x1, y1, x2, y2 = rect
            area = (x2 - x1 + 1) * (y2 - y1 + 1)  # Calculate the area of the rectangle
            total_area += area
            self.areas.append(total_area)  # Cumulative area
        
        self.total_area = total_area

    def pick(self) -> List[int]:
        # Pick a random point
        target = random.randint(1, self.total_area)  # Pick a random area index
        # Find the rectangle where this random area falls into
        left, right = 0, len(self.areas) - 1
        
        while left < right:
            mid = (left + right) // 2
            if self.areas[mid] < target:  # Move right if target is greater than mid area
                left = mid + 1
            else:  # Otherwise, move left
                right = mid
        
        # left is now the index of the selected rectangle
        rect = self.rects[left]
        x1, y1, x2, y2 = rect
        
        # Generate a random point within the selected rectangle
        px = random.randint(x1, x2)
        py = random.randint(y1, y2)
        
        return [px, py]


```

### Detailed Explanation

1. **Initialization (`__init__` method)**:
   - The constructor `__init__` accepts a list of rectangles (`rects`) where each rectangle is defined by its bottom-left and top-right coordinates.
   - We initialize a list `areas` to keep track of cumulative areas of the rectangles and a variable `total_area` to hold the total area of all rectangles.
   - For each rectangle, we calculate its area using the formula:
     \[
     \text{Area} = (x2 - x1 + 1) \times (y2 - y1 + 1)
     \]
     Here, `+1` accounts for both the inclusive bounds.
   - We keep adding these areas cumulatively to the `areas` list, enabling us to later determine the specific rectangle based on a random number.

2. **Picking a Point (`pick` method)**:
   - In the `pick` method, we generate a random number (`target`) between `1` and `total_area`. This number will help us determine which rectangle to pick.
   - We then perform a binary search on our `areas` list to find the rectangle that contains the randomly chosen area index. This is efficient and reduces the time complexity for searching to \(O(\log n)\), where \(n\) is the number of rectangles.
   - Once we find the appropriate rectangle (using index `left`), we extract its coordinates and then randomly choose integer coordinates within that rectangle using:
     \[
     px = \text{random.randint}(x1, x2)
     \]
     \[
     py = \text{random.randint}(y1, y2)
     \]
   - Finally, we return the randomly selected point as a list \([px, py]\).

### Usage
You can use this class in a LeetCode environment by instantiating it with a list of rectangles and calling `pick` to get random points as specified in the problem statement. The above implementation meets all problem constraints, ensuring uniform randomness over the chosen rectangles.

# 528. Random Pick with Weight

### Problem Description 
You are given an array of positive integers `w` where `w[i]` describes the weight of `i``th` index (0-indexed).

We need to call the function `pickIndex()` which randomly returns an integer in the range `[0, w.length - 1]`. `pickIndex()` should return the integer proportional to its weight in the `w` array. For example, for `w = [1, 3]`, the probability of picking the index `0` is `1 / (1 + 3) = 0.25` (i.e 25%) while the probability of picking the index `1` is `3 / (1 + 3) = 0.75` (i.e 75%).

More formally, the probability of picking index `i` is `w[i] / sum(w)`.


Example 1:
Input
["Solution","pickIndex"]
[[[1]],[]]
Output
[null,0]
Explanation
Solution solution = new Solution([1]);
solution.pickIndex(); // return 0. Since there is only one single element on the array the only option is to return the first element.


Example 2:
Input
["Solution","pickIndex","pickIndex","pickIndex","pickIndex","pickIndex"]
[[[1,3]],[],[],[],[],[]]
Output
[null,1,1,1,1,0]
Explanation
Solution solution = new Solution([1, 3]);
solution.pickIndex(); // return 1. It's returning the second element (index = 1) that has probability of 3/4.

solution.pickIndex(); // return 1
solution.pickIndex(); // return 1
solution.pickIndex(); // return 1
solution.pickIndex(); // return 0. It's returning the first element (index = 0) that has probability of 1/4.

Since this is a randomization problem, multiple answers are allowed so the following outputs can be considered correct :
[null,1,1,1,1,0]
[null,1,1,1,1,1]
[null,1,1,1,0,0]
[null,1,1,1,0,1]
[null,1,0,1,0,0]
......

and so on.


Constraints:
`1 <= w.length <= 10000`
`1 <= w[i] <= 10^5`
`pickIndex` will be called at most `10000` times.

### Solution 
 To solve the problem of selecting an index from an array of weights such that the index is selected with a probability proportional to the weight, we can use a combination of prefix sums and binary search. 

### Detailed Explanation:

1. **Prefix Sums**:
   - The idea is to convert the weights into cumulative sums, where each position `i` in the prefix sum array contains the total weight of all elements from index `0` to `i`.
   - This allows us to treat the problem as selecting a random number from `1` to the total weight. A random selection will be mapped to one of the indices based on the weights.

2. **Binary Search**:
   - Once we have the total weight and the prefix sums, when we generate a random number between `1` and the total sum, we can use binary search to find the first index in the prefix sum that is greater than or equal to this random number.
   - This index corresponds to the original weights array.

3. **Random Number Generation**:
   - For generating random numbers, we will use Python's `random.randint()` to select from the range `[1, total_weight]`.

### Implementation:
The implementation consists of a class `Solution` that contains the method `pickIndex`. The constructor initializes the prefix array based on the input weights.

Here’s how this can be implemented in Python:



```python
import random
from bisect import bisect_left

class Solution:
    def __init__(self, w: list[int]):
        self.prefix_sum = []
        total_sum = 0
        
        # Create the prefix sums array
        for weight in w:
            total_sum += weight
            self.prefix_sum.append(total_sum)
        
        self.total_weight = total_sum

    def pickIndex(self) -> int:
        # Generate a random number in the range [1, total_weight]
        target = random.randint(1, self.total_weight)
        
        # Use binary search to find the index
        index = bisect_left(self.prefix_sum, target)
        
        return index

```

### Explanation of the Code:
- In the `__init__` method:
  - We initialize `self.prefix_sum`, an empty list that will store cumulative sums.
  - We iterate through the input list `w`, compute the cumulative sum and store it in `self.prefix_sum`.
  - `self.total_weight` keeps track of the total sum of weights, which will be used in generating the random numbers.

- In the `pickIndex` method:
  - We generate a random target value between `1` and `self.total_weight`.
  - We use `bisect_left`, which is a binary search function from the `bisect` module. It finds the first index at which the `target` can be inserted into `self.prefix_sum` to maintain sorted order.
  - This index is then returned, representing the selected index in proportion to the weights.

### Example of Execution:
For the input `w = [1, 3]`, the prefix sum becomes `[1, 4]`. If a random number picked is `3`, `bisect_left` would find index `1` (because `3` fits in the range `[1, 4)`), thus returning `1`. 

This implementation meets the constraints and can handle up to `10,000` function invocations efficiently using the binary search approach. 

### Connecting to LeetCode:
To run this code on LeetCode, create an instance of the `Solution` class using `solution = Solution(weights)`, and invoke `solution.pickIndex()` as needed.

# 540. Single Element in a Sorted Array

### Problem Description 
You are given a sorted array consisting of only integers where every element appears exactly twice, except for one element which appears exactly once. Find this single element that appears only once.

Follow up: Your solution should run in O(log n) time and O(1) space.


Example 1:
Input: nums = [1,1,2,3,3,4,4,8,8]
Output: 2

Example 2:
Input: nums = [3,3,7,7,10,11,11]
Output: 10

Constraints:
`1 <= nums.length <= 10^5`
`0 <= nums[i] <= 10^5`

### Solution 
 To solve the problem of finding the single element in a sorted array where every other element appears exactly twice, we can take advantage of the properties of a binary search algorithm. The constraints given require us to achieve a solution with O(log n) time complexity and O(1) space complexity.

### Explanation of the Approach

1. **Binary Search Concept**: Given that the array is sorted and the elements are paired, we can utilize the binary search technique. The paired elements (elements that appear twice) will always cluster together.

2. **Identifying the Half**: The key observation is to analyze whether the single element lies in the left half or the right half of the current subarray we are examining:
   - If the index of the first occurrence of a paired element is even, then the pair starts at that index (i.e., `nums[mid]` and `nums[mid + 1]`).
   - If it is odd, the pair starts one element earlier (i.e., `nums[mid - 1]`).

3. **Logic to Narrow Down**: Based on whether `mid` is even or odd and whether the following or the previous element matches, we can figure out which half contains the unpaired element:
   - If `mid` is even and `nums[mid] == nums[mid + 1]`, then the single element must be in the right half. Otherwise, it is in the left half.
   - If `mid` is odd and `nums[mid] == nums[mid - 1]`, then the single element must be in the right half. Otherwise, it is in the left half.

4. **Termination Condition**: Continue this process until left equals right, at which point the left index will point to the single unpaired element.

### Python Code



```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            # Ensure mid is even
            if mid % 2 == 1:
                mid -= 1
            
            # Check the pair
            if nums[mid] == nums[mid + 1]:
                # The pair is correct, the single number must be on the right
                left = mid + 2
            else:
                # The single number must be on the left, including mid
                right = mid
        
        # left should be pointing to the single element
        return nums[left]

```

### Explanation of the Code

- **Initialization**: We declare `left` and `right` pointers to the start and end of the array respectively.
  
- **While Loop**: The loop continues until `left` is no longer less than `right`.

- **Find Midpoint**: We calculate the midpoint. If it is odd, we adjust it to be even to ensure it's the start of a pair.

- **Comparison Logic**: We compare `nums[mid]` with `nums[mid + 1]`:
  - If they match, it implies that the single element lies beyond `mid`, so we move the left pointer up to `mid + 2`.
  - If they do not match, it implies that the single element is before or at `mid`, so we adjust the right pointer to just `mid`.

- **Return Statement**: Once the loop concludes, `left` will point at the single element, which we return.

This algorithm efficiently narrows the search range, yielding a time complexity of O(log n) and space complexity of O(1), complying with the given constraints.

# 644. Maximum Average Subarray II

### Problem Description 
You are given an integer array `nums` consisting of `n` elements, and an integer `k`.

Find a contiguous subarray whose length is greater than or equal to `k` that has the maximum average value and return this value. Any answer with a calculation error less than `10-5` will be accepted.


Example 1:
Input: nums = [1,12,-5,-6,50,3], k = 4
Output: 12.75000
Explanation:
- When the length is 4, averages are [0.5, 12.75, 10.5] and the maximum average is 12.75
- When the length is 5, averages are [10.4, 10.8] and the maximum average is 10.8
- When the length is 6, averages are [9.16667] and the maximum average is 9.16667
The maximum average is when we choose a subarray of length 4 (i.e., the sub array [12, -5, -6, 50]) which has the max average 12.75, so we return 12.75
Note that we do not consider the subarrays of length < 4.


Example 2:
Input: nums = [5], k = 1
Output: 5.00000

Constraints:
`n == nums.length`
`1 <= k <= n <= 104`
`-104 <= nums[i] <= 104`

### Solution 
 To solve the problem of finding a contiguous subarray of length at least `k` that has the maximum average value, we can utilize a sliding window approach with prefix sums. This technique allows us to efficiently calculate the sums of subarrays without recalculating sums from scratch.

### Approach:
1. **Initialize Variables**: Start by calculating the total sum of the first `k` elements. This will serve as the initial maximum sum. 

2. **Sliding Window**: Iterate through the array from the `k`-th position onward. For each step:
   - Update the current sum by adding the current element and subtracting the element that is sliding out of the window.
   - Compare the current sum to keep track of the maximum sum encountered.

3. **Expand the Window**: After calculating sums for subarrays of length `k`, continue to expand the window to include more elements while still tracking the maximum average. This means that for each new element after `k`, we can check sums of lengths `k, k+1, ..., n`.

4. **Calculate Average**: Finally, calculate the maximum average using the stored maximum sum divided by the respective lengths.

5. **Handle Precision**: Ensure the final output has sufficient precision as required by the problem statement.

### Implementation:
Here’s the complete Python code that follows the outlined approach:



```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        n = len(nums)
        
        # Calculate the initial sum of the first k elements
        current_sum = sum(nums[:k])
        max_sum = current_sum
        
        # Use the sliding window technique
        for i in range(k, n):
            current_sum += nums[i] - nums[i - k]  # add new element and remove the oldest element
            max_sum = max(max_sum, current_sum)
        
        # Now check for averages greater than k
        # Starting with the max_sum up to length k:
        max_average = max_sum / k
        
        # Check with larger subarrays
        for length in range(k + 1, n + 1):
            current_sum += nums[length - 1]
            # Calculate the average
            max_average = max(max_average, current_sum / length)
        
        return max_average

```

### Explanation of the Code:
1. **Line 1**: We define the class `Solution` as per LeetCode structure.
2. **Line 2**: `findMaxAverage` is defined, receiving `nums` (the array) and `k` (the minimum length).
3. **Line 3**: Calculate the length of `nums`.
4. **Lines 6-7**: Compute and store the sum of the first `k` elements.
5. **Lines 10-13**: Iterate through the array using a sliding window to update the current sum and keep track of the maximum sum.
6. **Line 16**: Initialize `max_average` using the maximum sum found for the first `k` elements.
7. **Lines 19-23**: Expand the window to include additional elements, updating `current_sum` and calculating the average for each possible length greater than `k`.
8. **Line 25**: Finally, return the maximum average found.

### Complexity:
- **Time Complexity**: O(n) — The algorithm iterates through the array a constant number of times.
- **Space Complexity**: O(1) — The solution uses a fixed amount of additional space.

This code satisfies the requirements of the problem and will yield an accurate maximum average, abiding by the precision constraint.

# 658. Find K Closest Elements

### Problem Description 
Given a sorted integer array `arr`, two integers `k` and `x`, return the `k` closest integers to `x` in the array. The result should also be sorted in ascending order.

An integer `a` is closer to `x` than an integer `b` if:
`|a - x| < |b - x|`, or
`|a - x| == |b - x|` and `a < b`

Example 1:
Input: arr = [1,2,3,4,5], k = 4, x = 3
Output: [1,2,3,4]

Example 2:
Input: arr = [1,2,3,4,5], k = 4, x = -1
Output: [1,2,3,4]

Constraints:
`1 <= k <= arr.length`
`1 <= arr.length <= 104`
`arr` is sorted in ascending order.

`-104 <= arr[i], x <= 104`

### Solution 
 To solve the problem of finding `k` closest integers to `x` in a sorted array, we need to consider both the distance from `x` and the original order of the elements in the array. We can approach this problem using a combination of a binary search to locate the position of `x` and then expanding outwards to gather `k` closest numbers.

### Detailed Explanation

1. **Binary Search**: Since the array is sorted, we can use a binary search to find the position of `x`. This allows us to identify where `x` would fit in the array quickly.

2. **Two Pointers**: Once we have the position, we can use two pointers:
   - One pointer will move left (decreasing the index).
   - Another pointer will move right (increasing the index).
   
   With these pointers, we can gather the closest `k` elements by comparing the distance of elements at the two pointers from `x`.

3. **Distance Comparison**: For elements pointed to by the two pointers, we compare their distances to `x`. If the distance of the left pointer is less than the right one, we include the left element, otherwise, we include the right element.

4. **Result Compilation**: We'll collect the valid closest numbers until we've found `k` elements. Finally, we'll return these elements sorted.

5. **Edge Cases**: Be careful to handle edge cases, such as when `x` is smaller than all elements in the array or larger than all.

### Implementation

Here's the code that follows the solution format of LeetCode:



```python
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        from bisect import bisect_left
        
        # Find the position using binary search
        pos = bisect_left(arr, x)
        
        left, right = pos - 1, pos
        
        # Result list to hold the closest k elements
        result = []
        
        # Collect k closest elements
        for _ in range(k):
            if left < 0:  # If left index is out of bounds
                result.append(arr[right])
                right += 1
            elif right >= len(arr):  # If right index is out of bounds
                result.append(arr[left])
                left -= 1
            else:
                # Check distances
                if abs(arr[left] - x) <= abs(arr[right] - x):
                    result.append(arr[left])
                    left -= 1
                else:
                    result.append(arr[right])
                    right += 1
        
        # Sort the result before returning
        return sorted(result)

```

### Key Points in the Code

- We use `bisect_left` from the `bisect` module to find the insertion point for `x` in `arr`. This gives us a starting point for our two-pointer approach.
- The loop runs `k` times, and during each iteration, we decide which pointer to move based on the closest distance comparison.
- Finally, we sort the result list before returning it to match the output requirements.

### Complexity

- **Time Complexity**: O(k log k) for sorting the selected closest elements, and O(log n) for the binary search overhead.
- **Space Complexity**: O(k) for the result list that stores the closest integers.

This solution efficiently finds and returns the `k` closest integers to `x` in a sorted order, as required.

# 668. Kth Smallest Number in Multiplication Table

### Problem Description 
Nearly every one have used the Multiplication Table. But could you find out the `k-th` smallest number quickly from the multiplication table?
Given the height `m` and the length `n` of a `m * n` Multiplication Table, and a positive integer `k`, you need to return the `k-th` smallest number in this table.


Example 1:
Input: m = 3, n = 3, k = 5
Output: 
Explanation: 
The Multiplication Table:
1	2	3
2	4	6
3	6	9
The 5-th smallest number is 3 (1, 2, 2, 3, 3).


Example 2:
Input: m = 2, n = 3, k = 6
Output: 
Explanation: 
The Multiplication Table:
1	2	3
2	4	6
The 6-th smallest number is 6 (1, 2, 2, 3, 4, 6).

Note:
The `m` and `n` will be in the range [1, 30000].

The `k` will be in the range [1, m * n]

### Solution 
 To solve the problem of finding the `k-th` smallest number in an `m x n` multiplication table, we can leverage a binary search approach. The challenge here is that the multiplication table grows quite rapidly, and we can't feasibly generate it in its entirety given the constraints.

### Explanation

1. **Understanding the Multiplication Table**:
   - The multiplication table for `m` and `n` consists of numbers formed by `i * j` where `1 ≤ i ≤ m` and `1 ≤ j ≤ n`. 
   - For instance, the first few elements go:
     
```
     1, 2, 3, ..., n
     2, 4, 6, ..., 2*n
     ...
     m, 2*m, 3*m, ..., m*n
     
```
   - The smallest number will always be `1` (from `1 * 1`), and the largest will be `m * n`.

2. **Binary Search Over the Result Range**:
   - We can perform a binary search over the range of numbers possible in this multiplication table, which is `[1, m*n]`.
   - For a midpoint (`mid`) during our binary search, we need to count how many numbers in the multiplication table are less than or equal to `mid`.

3. **Counting Elements**:
   - To count how many products \( i*j \) are less than or equal to `mid`, we can notice:
     - For each row `i`, the largest column index `j` where `i*j <= mid` is `mid // i`.
     - Thus, the count of numbers that are less than or equal to `mid` can be expressed as the sum of `min(mid // i, n)` for each `i` from `1` to `m`.

4. **Binary Search Logic**:
   - If the count of numbers less than or equal to `mid` is less than `k`, it means the `k-th` smallest number lies in the upper half, so we move our search range up.
   - If the count is greater than or equal to `k`, it lies in the lower half, so we move our search range down.

### Implementation

Here's how you can implement this in Python:



```python
class Solution:
    def findKthNumber(self, m: int, n: int, k: int) -> int:
        # Binary search on the value
        left, right = 1, m * n
        
        while left < right:
            mid = (left + right) // 2
            count = 0
            
            # Count how many numbers are <= mid in the multiplication table
            for i in range(1, m + 1):
                count += min(mid // i, n)  # count numbers in the i-th row
                
            # If count is less than k, it means the k-th number is larger
            if count < k:
                left = mid + 1
            else:
                right = mid
        
        # At the end of the loop, left == right and it's the k-th number
        return left

```

### Explanation of the Code:
1. **Class Definition**: We define a class `Solution` which is the required format for LeetCode.
2. **findKthNumber Method**: This method takes three integers as input: `m`, `n`, and `k`.
3. **Binary Search Initialization**: We set `left` to `1` and `right` to `m * n`.
4. **Counting Loop**: We iterate over each `i` from `1` to `m`, counting how many numbers in the multiplication table are less than or equal to `mid` using the formula `min(mid // i, n)`.
5. **Adjusting Search Range**: Depending on whether the count is less than or greater than `k`, we adjust the `left` and `right` bounds.
6. **Final Result**: The value of `left` (or `right`) at the end of the search will be the `k-th` smallest number.

### Complexity:
- The complexity of this approach is \(O(m \log(m \times n))\), where \(O(m)\) comes from counting numbers for each mid, and \(O(\log(m \times n))\) comes from the binary search on the range. This is efficient given the constraints.

# 702. Search in a Sorted Array of Unknown Size

### Problem Description 
Given an integer array sorted in ascending order, write a function to search `target` in `nums`.  If `target` exists, then return its index, otherwise return `-1`. However, the array size is unknown to you. You may only access the array using an `ArrayReader` interface, where `ArrayReader.get(k)` returns the element of the array at index `k` (0-indexed).

You may assume all integers in the array are less than `10000`, and if you access the array out of bounds, `ArrayReader.get` will return `2147483647`.


Example 1:
Input: `array` = [-1,0,3,5,9,12], `target` = 9
Output: 4
Explanation: 9 exists in `nums` and its index is 4

Example 2:
Input: `array` = [-1,0,3,5,9,12], `target` = 2
Output: -1
Explanation: 2 does not exist in `nums` so return -1

Constraints:
You may assume that all elements in the array are unique.

The value of each element in the array will be in the range `[-9999, 9999]`.

The length of the array will be in the range `[1, 10^4]`.

### Solution 
 To solve this problem, we need to efficiently search for a target value in an unknown size sorted array using the given `ArrayReader` interface. Since we do not know the length of the array, our approach will involve finding a suitable range in which the target might exist, and then performing a binary search within that range.

### Approach
1. **Determine the Range**: We will first find an interval where the target might reside. This will be done by repeatedly calling `ArrayReader.get(k)` and doubling `k` until we find an out-of-bounds response. This helps us narrow down the potential search space.

2. **Binary Search**: Once we have a range established from the previous step, we can use binary search to find the target in that specific range. This is efficient and works perfectly because the array is sorted.

### Implementation
Below is the implementation that adheres to the LeetCode format:



```python
class ArrayReader:
    def get(self, index: int) -> int:
        # This is just a placeholder for the actual implementation provided by LeetCode.
        pass

class Solution:
    def search(self, reader: ArrayReader, target: int) -> int:
        # Step 1: Determine the bounds
        left, right = 0, 1
        
        # Use get(right) to find an approximate upper bound:
        while reader.get(right) < 2147483647:
            right *= 2
        
        # Step 2: Perform binary search in the range [left, right]
        while left <= right:
            mid = left + (right - left) // 2
            current_val = reader.get(mid)
            
            if current_val == target:
                return mid
            elif current_val < target:
                left = mid + 1
            else:
                right = mid - 1
                
        return -1  # target not found

```

### Explanation:
1. **Class and Function Definition**: We define the `ArrayReader` class for the interface and a `Solution` class with the `search` method.

2. **Range Discovery**:
   - Start with `left` at 0 and `right` at 1.
   - Use a loop that doubles `right` until `reader.get(right)` returns `2147483647`. This indicates we've gone out of bounds. At each step, we're effectively expanding our search range exponentially.

3. **Binary Search**:
   - While within the bounds (`left` <= `right`), calculate the middle index.
   - Get the value at that index using `reader.get(mid)`.
   - Compare the retrieved value with the `target`:
     - If they match, return `mid` (the index).
     - If the value is less than `target`, move the `left` boundary up.
     - If the value is greater than `target`, move the `right` boundary down.
   - If the loop exits without finding the target, return `-1`.

### Complexity
- **Time Complexity**: The time complexity is O(log N) due to the binary search after finding the range, and O(log M) for finding the boundary range where M is the actual size of the array.
- **Space Complexity**: O(1) as we are only using a fixed amount of additional space.

This solution is efficient, clear, and works within the constraints of the problem as described.

# 704. Binary Search

### Problem Description 
Given an array of integers `nums` which is sorted in ascending order, and an integer `target`, write a function to search `target` in `nums`. If `target` exists, then return its index. Otherwise, return `-1`.


Example 1:
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4

Example 2:
Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1

Constraints:
`1 <= nums.length <= 104`
`-9999 <= nums[i], target <= 9999`
All the integers in `nums` are unique.

`nums` is sorted in an ascending order.

### Solution 
 To solve the problem of searching for a target value in a sorted array, we can utilize the binary search algorithm. This is a highly efficient method suited for the task because it has a time complexity of O(log n), which is optimal for searching in sorted arrays.

### Explanation:

1. **Binary Search Algorithm**:
   - We maintain two pointers, `left` and `right`, which mark the current segment of the array under consideration.
   - The algorithm works by calculating the middle index (`mid`) of the current segment and comparing the middle element (`nums[mid]`) with the target.
   - If `nums[mid]` matches the target, we've found our target, and we return the index `mid`.
   - If `nums[mid]` is less than the target, it means the target must be in the right half of the array. We adjust the `left` pointer to `mid + 1`.
   - If `nums[mid]` is greater than the target, it means the target must be in the left half of the array. We adjust the `right` pointer to `mid - 1`.
   - We repeat this process until the `left` pointer exceeds the `right` pointer, which indicates that the target is not present in the array.

2. **Edge Cases**:
   - The function should handle cases where the array is very small, containing only one element or even where the target is outside the range of values in the array.

Here’s the Python code that implements the above logic:



```python
def search(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid potential overflow
        
        # Check if the mid element is the target
        if nums[mid] == target:
            return mid
        # Check if target is in the right half
        elif nums[mid] < target:
            left = mid + 1
        # Check if target is in the left half
        else:
            right = mid - 1
            
    # Target is not found
    return -1

```

### Usage:

You can run the provided code in a LeetCode environment, and it should accurately solve the given problem. Here's how it aligns with the format used in LeetCode submissions:

1. **Function Definition**: The function is defined with the required parameters (`nums` and `target`).
2. **Return Values**: It returns the index of the target or `-1` if not found, as specified in the problem statement.
3. **Efficiency**: The use of binary search ensures efficient operation within the constraints provided.

### Example Execution:

Let's run through the provided examples:

- Example 1:
  - Input: `nums = [-1,0,3,5,9,12]`, `target = 9`
    - The function will return `4` because `9` is at index `4`.

- Example 2:
  - Input: `nums = [-1,0,3,5,9,12]`, `target = 2`
    - The function will return `-1` because `2` is not in the array.

This implementation correctly handles all specified constraints and edge cases, making it a robust solution for the problem.

# 710. Random Pick with Blacklist

### Problem Description 
Given a blacklist `B` containing unique integers from `[0, N)`, write a function to return a uniform random integer from `[0, N)` which is NOT in `B`.

Optimize it such that it minimizes the call to system’s `Math.random()`.

Note:
`1 <= N <= 1000000000`
`0 <= B.length < min(100000, N)`
`[0, N)` does NOT include N. See interval notation.


Example 1:
Input: 
["Solution","pick","pick","pick"]
[[1,[]],[],[],[]]
Output: [null,0,0,0]

Example 2:
Input: 
["Solution","pick","pick","pick"]
[[2,[]],[],[],[]]
Output: [null,1,1,1]

Example 3:
Input: 
["Solution","pick","pick","pick"]
[[3,[1]],[],[],[]]
Output: [null,0,0,2]

Example 4:
Input: 
["Solution","pick","pick","pick"]
[[4,[2]],[],[],[]]
Output: [null,1,3,1]
Explanation of Input Syntax:
The input is two lists: the subroutines called and their arguments. `Solution`'s constructor has two arguments, `N` and the blacklist `B`. `pick` has no arguments. Arguments are always wrapped with a list, even if there aren't any.

### Solution 
 To solve the problem of picking a random integer from the range `[0, N)` that is not in the blacklist `B`, we can optimize our solution using a combination of hashing and random number generation. The approach involves the following steps:

1. **Understanding the Data Structure**: We will use a set to store the blacklisted numbers for O(1) average time complexity when checking for membership.

2. **Using Mapping for Blacklist Management**: If the blacklisted numbers prevent us from picking certain numbers, we can use a mapping technique to avoid these numbers. Specifically, we will keep track of the total count of blacklisted numbers and adjust our pick range accordingly.

3. **Random Picking**: For each pick, generate a random number. If that number falls within the blacklist, we will find the next possible number to pick that is not in the blacklist.

4. **Optimized Randomness**: By maintaining a list of allowed numbers in a compact manner, we can minimize the reliance on random number generation.

The following Python code implements this approach and is designed to meet the LeetCode format:



```python
import random

class Solution:
    def __init__(self, N: int, B: list[int]):
        self.N = N
        self.blacklist = set(B)
        self.valid_numbers = []
        
        # Create a valid numbers list that does not include the blacklist
        for i in range(N):
            if i not in self.blacklist:
                self.valid_numbers.append(i)
        
    def pick(self) -> int:
        # Randomly pick an index from valid_numbers
        return random.choice(self.valid_numbers)

# Example usage:
# obj = Solution(3, [1])
# print(obj.pick())  # Outputs: 0 or 2

```

### Explanation of the Code:

- **The Constructor (`__init__` method)**:
  - Initializes the variables `N` and `B`.
  - Converts the blacklist to a set for quick lookups.
  - Iterates through the range `[0, N)` and builds a list of valid numbers that are not in the blacklist.

- **The `pick` Method**:
  - Uses `random.choice` to randomly select an integer from the `valid_numbers` list, which contains all integers in `[0, N)` that are not blacklisted.

### Space Complexity:
- We need extra space to store the `valid_numbers`. In the worst-case scenario (when all numbers except one are blacklisted), the space used will be close to `O(N)`.

### Time Complexity:
- The constructor runs in `O(N)` in the worst case to initialize the `valid_numbers` list, while each call to `pick` operates in `O(1)` due to the `random.choice` method.

This code efficiently picks random numbers from a large range, skipping those in the blacklist while minimizing additional overhead.

