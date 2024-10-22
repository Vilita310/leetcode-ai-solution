# 1802. Maximum Value at a Given Index in a Bounded Array

### Problem Description 
You are given three positive integers: `n`, `index`, and `maxSum`. You want to construct an array `nums` (0-indexed) that satisfies the following conditions:
`nums.length == n`
`nums[i]` is a positive integer where `0 <= i < n`.

`abs(nums[i] - nums[i+1]) <= 1` where `0 <= i < n-1`.

The sum of all the elements of `nums` does not exceed `maxSum`.

`nums[index]` is maximized.

Return `nums[index]` of the constructed array.

Note that `abs(x)` equals `x` if `x >= 0`, and `-x` otherwise.


Example 1:
Input: n = 4, index = 2,  maxSum = 6
Output: 2
Explanation: nums = [1,2,2,1] is one array that satisfies all the conditions.

There are no arrays that satisfy all the conditions and have nums[2] == 3, so 2 is the maximum nums[2].


Example 2:
Input: n = 6, index = 1,  maxSum = 10
Output: 3

Constraints:
`1 <= n <= maxSum <= 109`
`0 <= index < n`

### Solution 
 To solve this problem, we need to construct an array `nums` of length `n` such that:
1. The values of `nums` must be positive integers that differ by no more than `1` between adjacent elements.
2. The sum of the elements of `nums` must not exceed `maxSum`.
3. The value of `nums[index]` must be maximized.

The approach to solve this problem can be summarized as follows:

1. **Binary Search**: We'll use binary search to maximize the value at `nums[index]`. We need to find the maximum feasible value `x`, which can be assigned to `nums[index]`. We'll then check if it is possible to construct the array `nums` such that `nums[index] = x` while satisfying the other conditions.

2. **Check Feasibility**: For a candidate value `x`, we'll calculate how much we can take from both the left and right sides of `index` to keep the sum under `maxSum`. We need to make sure that we can fill the array with positive integers while maintaining the differences of at most `1`.

3. **Fill Numbers**: We can determine how many numbers we can fill to the left of `index` and how much we can fill to the right, starting from `x`.

Now, let's implement this logic in Python in accordance with the LeetCode solution format.



```python
class Solution:
    def maxValue(self, n: int, index: int: maxSum: int) -> int:
        # Function to check if we can achieve a certain middle value with the given constraints
        def canAchieve(value: int) -> bool:
            # Calculate the total sum based on the potential middle value
            total = value
            
            # Calculate left side sum
            left = max(0, value - 1)  # the max value to the left of index
            count_left = index  # how many elements can we fill to the left
            
            if count_left > 0:
                # If we can fill to the left
                if left < count_left:
                    total += sum(range(1, left + 1))  # fill with 1 to left
                    total += count_left - left  # remaining numbers as 1
                else:
                    total += sum(range(left - count_left + 1, left + 1))  # fill using a series
                
            # Calculate right side sum
            right = max(0, value - 1)  # the max value to the right of index
            count_right = n - index - 1  # how many elements can we fill to the right
            
            if count_right > 0:
                # If we can fill to the right
                if right < count_right:
                    total += sum(range(1, right + 1))  # fill with 1 to right
                    total += count_right - right  # remaining numbers as 1
                else:
                    total += sum(range(right - count_right + 1, right + 1))  # fill using a series
            
            return total <= maxSum
        
        # Binary search for the maximum possible value at index
        left, right = 1, maxSum
        while left < right:
            mid = (left + right + 1) // 2  # Midpoint, knowing we need to go higher
            
            if canAchieve(mid):
                left = mid  # If achievable, we can try for a higher value
            else:
                right = mid - 1  # If not achievable, go lower
        
        return left  # This will be the maximum nums[index]

```

### Explanation:
- **canAchieve Function**: This checks if `nums[index]` can equal `value` while respecting both constraints. It calculates how much sum can be filled on both sides of `index`.
- We utilize a binary search to effectively narrow down the maximum possible value of `nums[index]`.
- The function returns the largest value of `nums[index]` once our binary search concludes.

This implementation is efficient enough to handle the problem's constraints (up to `10^9`) due to the logarithmic nature of the binary search.

# 1818. Minimum Absolute Sum Difference

### Problem Description 
You are given two positive integer arrays `nums1` and `nums2`, both of length `n`.

The absolute sum difference of arrays `nums1` and `nums2` is defined as the sum of `|nums1[i] - nums2[i]|` for each `0 <= i < n` (0-indexed).

You can replace at most one element of `nums1` with any other element in `nums1` to minimize the absolute sum difference.

Return the minimum absolute sum difference after replacing at most one element in the array `nums1`. Since the answer may be large, return it modulo `109 + 7`.

`|x|` is defined as:
`x` if `x >= 0`, or
`-x` if `x < 0`.


Example 1:
Input: nums1 = [1,7,5], nums2 = [2,3,5]
Output: 3
Explanation: There are two possible optimal solutions:
- Replace the second element with the first: [1,7,5] => [1,1,5], or
- Replace the second element with the third: [1,7,5] => [1,5,5].

Both will yield an absolute sum difference of `|1-2| + (|1-3| or |5-3|) + |5-5| = `3.


Example 2:
Input: nums1 = [2,4,6,8,10], nums2 = [2,4,6,8,10]
Output: 0
Explanation: nums1 is equal to nums2 so no replacement is needed. This will result in an 
absolute sum difference of 0.


Example 3:
Input: nums1 = [1,10,4,4,2,7], nums2 = [9,3,5,1,7,4]
Output: 20
Explanation: Replace the first element with the second: [1,10,4,4,2,7] => [10,10,4,4,2,7].

This yields an absolute sum difference of `|10-9| + |10-3| + |4-5| + |4-1| + |2-7| + |7-4| = 20`

Constraints:
`n == nums1.length`
`n == nums2.length`
`1 <= n <= 105`
`1 <= nums1[i], nums2[i] <= 105`

### Solution 
 To solve the problem of minimizing the absolute sum difference between two arrays `nums1` and `nums2` by changing at most one element of `nums1`, we can utilize the following approach:

### Approach:
1. **Calculate Initial Sum**: First, we need to compute the initial absolute sum difference without any modifications. This is simply the sum of `|nums1[i] - nums2[i]|` for all valid indices `i`.

2. **Find Optimal Replacement**: For each element in `nums2`, determine the best possible replacement in `nums1` that minimizes the sum difference. To efficiently find the closest value in `nums1`, we can:
   - Sort `nums1`.
   - For each `nums2[i]`, use binary search to find the closest number in `nums1` that minimizes `|nums1[j] - nums2[i]|`.

3. **Update the Sum**: For each potential replacement found, calculate the new sum difference and track the minimum value encountered.

4. **Result**: Return the minimum absolute sum difference computed modulo \(10^9 + 7\).

Here's the implementation of the above approach in Python:



```python
class Solution:
    def minAbsoluteSumDiff(self, nums1: List[int], nums2: List[int]) -> int:
        import bisect

        MOD = 10**9 + 7
        
        # Step 1: Calculate the initial absolute sum difference
        n = len(nums1)
        initial_sum = sum(abs(nums1[i] - nums2[i]) for i in range(n))
        
        # Step 2: Prepare nums1 for binary search
        sorted_nums1 = sorted(nums1)
        
        # Step 3: Try to minimize the absolute sum
        min_sum = initial_sum
        
        for i in range(n):
            target = nums2[i]
            # Find the closest value to nums2[i] in sorted_nums1
            idx = bisect.bisect_left(sorted_nums1, target)
            
            # Check the closest value on the left and right if available
            for j in (idx - 1, idx):
                if 0 <= j < n:  # Make sure the index is valid
                    new_number = sorted_nums1[j]
                    new_abs_diff = abs(new_number - target)
                    # Calculate new sum by adjusting the original abs differences
                    new_sum = initial_sum - abs(nums1[i] - target) + new_abs_diff
                    min_sum = min(min_sum, new_sum)
        
        return min_sum % MOD

```

### Explanation of the Code:
- **Input Processing**: We define a class `Solution` and a method `minAbsoluteSumDiff` which takes two lists of integers as input.
- **Initial Sum Calculation**: We calculate the current absolute sum difference using a generator expression.
- **Binary Search for Optimal Replacement**: We sort `nums1` to enable efficient searching. For each element in `nums2`, we find the closest numbers in `sorted_nums1` using `bisect_left` to maintain order and efficiency.
- **Determine Minimum Sum**: For each closest number found, we compute the potential new sum and keep track of the minimum over all trials.
- **Return Value**: We return the minimum sum modulo \(10^9 + 7\).

### Complexity:
- **Time Complexity**: Sorting `nums1` takes \(O(n \log n)\), and for each element in `nums2`, we perform a binary search which takes \(O(\log n)\). Thus, the overall complexity is \(O(n \log n)\).
- **Space Complexity**: We require additional space for the sorted version of `nums1`, which also takes \(O(n)\).

This implementation is efficient and meets the constraints provided in the problem statement.

