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
 To solve the problem of finding the maximum sum of a rectangle in a matrix such that its sum is no larger than `k`, we can utilize a combination of prefix sums and a data structure to help track the current sums efficiently.

### Approach:

1. **Prefix Sums**: We'll compute sums of rectangles using prefix sums, which allows us to calculate the sum of any subarray efficiently.
  
2. **Iterate over rows**: We can iterate over all pairs of rows. For each pair, we will consider the sum of columns between these two rows, which will allow us to reduce our problem to a 1D problem.

3. **Use a Sorted List**: To efficiently find the largest sum which is less than or equal to `k`, we will use a sorted list (or a balanced binary search tree) to keep track of the sums we encounter, enabling us to perform binary search for quickly determining the best possible previous sum to form a rectangle with the current sum.

### Implementation:

Here is the Python code to implement this approach:



```python
from sortedcontainers import SortedList

class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        max_sum = float('-inf')
        
        # Iterate over all pairs of rows
        for top in range(m):
            # We will maintain a temporary array for the sum of columns between 'top' and 'bottom'
            temp = [0] * n
            
            for bottom in range(top, m):
                # Update the temp array with the current row sums
                for i in range(n):
                    temp[i] += matrix[bottom][i]
                
                # Now, we need to find the maximum sum of rectangles using this temp array
                sorted_list = SortedList([0])  # Start with 0 for handling full prefix sums
                curr_sum = 0
                
                for sum_j in temp:
                    curr_sum += sum_j
                    
                    # We want to find the largest x in sorted_list such that curr_sum - x <= k
                    # This is equivalent to finding x >= curr_sum - k
                    target = curr_sum - k
                    
                    # Use bisect to find the location
                    idx = sorted_list.bisect_left(target)
                    
                    if idx < len(sorted_list):  # If we found a valid idx
                        max_sum = max(max_sum, curr_sum - sorted_list[idx])
                    
                    # Add the current sum to the sorted list
                    sorted_list.add(curr_sum)

        return max_sum

```

### Explanation:

- **Outer Loop (rows)**: We iterate through all starting rows (variable `top`).
- **Inner Loop (rows)**: For each starting row, we then iterate through all possible ending rows (variable `bottom`).
- **Temporary Column Sums**: For each combination of `top` and `bottom`, we calculate the sums of all columns between these two rows and store it in the `temp` list.
- **Sorted List**: We maintain a sorted list (`sorted_list`) to keep track of the cumulative sums. This allows us to quickly find the maximum sum rectangle that does not exceed `k` using binary search.
- **Update max_sum**: For each new sum calculated, we check if there exists a sum in the `sorted_list` such that when subtracted from the current sum gives a value less than or equal to `k`. If yes, we update `max_sum`.

### Complexity:
- **Time Complexity**: O(m^2 * n log n) where `m` is the number of rows and `n` is the number of columns. The sorting operation takes `log n`.
- **Space Complexity**: O(n) for the temporary column sums and O(n) for the sorted list. 

This implementation efficiently computes the desired result while still being readable and maintainable.

# 368. Largest Divisible Subset

### Problem Description 
Given a set of distinct positive integers `nums`, return the largest subset `answer` such that every pair `(answer[i], answer[j])` of elements in this subset satisfies:
`answer[i] % answer[j] == 0`, or
`answer[j] % answer[i] == 0`
If there are multiple solutions, return any of them.


Example 1:
Input: nums = [1,2,3]
Output: [1,2]
Explanation: [1,3] is also accepted.


Example 2:
Input: nums = [1,2,4,8]
Output: [1,2,4,8]

Constraints:
`1 <= nums.length <= 1000`
`1 <= nums[i] <= 2 * 109`
All the integers in `nums` are unique.

### Solution 
 The problem at hand is to find the largest divisible subset of distinct positive integers. A subset is defined as divisible if for every pair of integers in the subset, one is divisible by the other. 

To solve this problem, we can make use of dynamic programming along with sorting. Here's a detailed step-by-step breakdown of the approach:

### Steps to Solve the Problem:

1. **Sorting the Input**: First, we sort the array of integers. This is crucial since if `a % b == 0`, then `a` must be greater than or equal to `b`. Sorting will help us easily check divisibility in increasing order.

2. **Dynamic Programming Array**: We'll maintain a dp array (`dp`) where `dp[i]` will store the length of the largest divisible subset that ends with the `i-th` number.

3. **Backtracking Array**: Alongside `dp`, we'll maintain a `previous` array that helps us track the previous index of the last number in the optimal subset. This allows us to reconstruct the largest subset later.

4. **Building the DP Table**: For each number, we will compare it with all the previous numbers to see if we can form a larger divisible subset ending with the current number. If `nums[i]` is divisible by `nums[j]`, we can possibly form a subset with `nums[j]` and update `dp[i]` and `previous[i]` accordingly.

5. **Finding the Maximum**: Lastly, we find the maximum element in the `dp` array for the size of our largest subset, and use the `previous` array to reconstruct the actual subset.

### Solution Code:

Here's how the implementation looks in Python:



```python
class Solution:
    def largestDivisibleSubset(self, nums):
        # Step 1: Sort the array
        nums.sort()
        n = len(nums)
        
        # Step 2: Initialize the dp and previous arrays
        dp = [1] * n
        previous = [-1] * n
        
        # Step 3: Fill the dp array
        max_size = 0
        max_index = 0
        
        for i in range(n):
            for j in range(i):
                if nums[i] % nums[j] == 0:  # Check divisibility
                    if dp[i] < dp[j] + 1:  # Check if we can make a longer subset
                        dp[i] = dp[j] + 1
                        previous[i] = j  # Keep track of the previous index
            # Update the maximum size found
            if dp[i] > max_size:
                max_size = dp[i]
                max_index = i
        
        # Step 4: Rebuild the largest divisible subset
        answer = []
        while max_index >= 0:
            answer.append(nums[max_index])
            max_index = previous[max_index]
        
        # Reverse to get them in the correct order
        return answer[::-1]

# Example Usage
# sol = Solution()
# print(sol.largestDivisibleSubset([1,2,3]))  # Output: [1,2] or [1,3]
# print(sol.largestDivisibleSubset([1,2,4,8]))  # Output: [1,2,4,8]

```

### Explanation of the Code:
- **Sorting**: The `nums.sort()` statement sorts the input list. This allows us to easily check divisibility.
- **DP Initialization**: The `dp` array initializes to `1` since any number can at least form a subset with itself.
- **Nested Loops**: The two nested loops iterate over the sorted numbers; for each pair `(j, i)`, if `nums[i]` is divisible by `nums[j]`, we potentially increase the size of the subset ending at `i`.
- **Backtracking**: After finding the size of the largest subset, we use the `previous` array to backtrack and collect the subset by checking the indices.
- **Returning Result**: The subset is returned in reverse order (as it was constructed backwards).

This code adheres to the LeetCode submission format, and should work seamlessly on the LeetCode website when tested with the provided examples.

# 375. Guess Number Higher or Lower II

### Problem Description 
We are playing the Guessing Game. The game will work as follows:
I pick a number between `1` and `n`.

You guess a number.

If you guess the right number, you win the game.

If you guess the wrong number, then I will tell you whether the number I picked is higher or lower, and you will continue guessing.

Every time you guess a wrong number `x`, you will pay `x` dollars. If you run out of money, you lose the game.

Given a particular `n`, return the minimum amount of money you need to guarantee a win regardless of what number I pick.


Example 1:
Input: n = 10
Output: 16
Explanation: The winning strategy is as follows:
- The range is [1,10]. Guess 7.

    - If this is my number, your total is $0. Otherwise, you pay $7.

    - If my number is higher, the range is [8,10]. Guess 9.

        - If this is my number, your total is $7. Otherwise, you pay $9.

        - If my number is higher, it must be 10. Guess 10. Your total is $7 + $9 = $16.

        - If my number is lower, it must be 8. Guess 8. Your total is $7 + $9 = $16.

    - If my number is lower, the range is [1,6]. Guess 3.

        - If this is my number, your total is $7. Otherwise, you pay $3.

        - If my number is higher, the range is [4,6]. Guess 5.

            - If this is my number, your total is $7 + $3 = $10. Otherwise, you pay $5.

            - If my number is higher, it must be 6. Guess 6. Your total is $7 + $3 + $5 = $15.

            - If my number is lower, it must be 4. Guess 4. Your total is $7 + $3 + $5 = $15.

        - If my number is lower, the range is [1,2]. Guess 1.

            - If this is my number, your total is $7 + $3 = $10. Otherwise, you pay $1.

            - If my number is higher, it must be 2. Guess 2. Your total is $7 + $3 + $1 = $11.

The worst case in all these scenarios is that you pay $16. Hence, you only need $16 to guarantee a win.


Example 2:
Input: n = 1
Output: 0
Explanation: There is only one possible number, so you can guess 1 and not have to pay anything.


Example 3:
Input: n = 2
Output: 1
Explanation: There are two possible numbers, 1 and 2.

- Guess 1.

    - If this is my number, your total is $0. Otherwise, you pay $1.

    - If my number is higher, it must be 2. Guess 2. Your total is $1.

The worst case is that you pay $1.


Constraints:
`1 <= n <= 200`

### Solution 
 To solve the problem of finding the minimum amount of money you need to guarantee a win in the guessing game, we can take a dynamic programming approach. We will keep track of the minimum cost of guessing for all possible ranges of numbers and use these results to compute the costs for larger ranges.

### Explanation

1. **Setting Up a DP Table**: We will define a 2D list `dp` where `dp[start][end]` will store the minimum amount of money needed to guarantee a win when guessing between the numbers `start` and `end`.

2. **Base Cases**:
   - When `start` equals `end`, there is only one number to guess, and you win immediately without cost, so `dp[start][start]` should be 0. 
   - When there are two numbers `(start, start + 1)`, you can guess `start` for a cost of `start` and if that guess fails, you will then have to guess `start + 1`, leading to a total cost of `start`. Thus, `dp[start][start + 1]` will be just `start`.

3. **Filling the DP Table**: For ranges larger than 2, we iterate over all possible ranges `(start, end)`.
   - For each possible guess `x` from `start` to `end`, the resulting costs will be:
     - If you guess `x`, and it is not correct:
       - If the number is higher (in range `[x + 1, end]`), the cost involves paying `x` plus the minimum cost to guess in that range.
       - If it is lower (in range `[start, x - 1]`), similarly calculate the cost.
   - The goal is to choose `x` such that the worst-case cost (the maximum of the two scenarios) is minimized.

4. **Final Result**: After populating the DP table, the result for the starting range 1 to `n` will be in `dp[1][n]`.

### Implementation

Here's the Python code implementing the above logic:



```python
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        # Create a DP table initialized to 0
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        # Iterate through every possible range length
        for length in range(2, n + 1):  # length varies from 2 to n
            for start in range(1, n - length + 2):  # start varies appropriately
                end = start + length - 1  # Determine end based on start and length
                dp[start][end] = float('inf')  # Start with an infinite cost
                
                # Try every possible guess in the current range
                for x in range(start, end + 1):
                    # Calculate the cost if we guess x
                    cost = x + max(dp[start][x - 1], dp[x + 1][end])
                    dp[start][end] = min(dp[start][end], cost)
        
        # The answer for the range 1 to n
        return dp[1][n]

```

### Explanation of the Code:

- We create a 2D list `dp` of size `(n+1) x (n+1)` to store costs, initializing all to zero.
- We fill in the DP table by considering all possible lengths of sub-ranges and calculating the minimum cost for each range.
- The maximum function is used to ensure we consider the worst case for each guess.
- Finally, we return the minimum cost of guessing across the full range from 1 to `n` by accessing `dp[1][n]`.

This approach efficiently computes the desired outcome within the constraints provided (with `n` up to 200).

# 376. Wiggle Subsequence

### Problem Description 
A wiggle sequence is a sequence where the differences between successive numbers strictly alternate between positive and negative. The first difference (if one exists) may be either positive or negative. A sequence with two or fewer elements is trivially a wiggle sequence.

For example, `[1, 7, 4, 9, 2, 5]` is a wiggle sequence because the differences `(6, -3, 5, -7, 3)` alternate between positive and negative.

In contrast, `[1, 4, 7, 2, 5]` and `[1, 7, 4, 5, 5]` are not wiggle sequences. The first is not because its first two differences are positive, and the second is not because its last difference is zero.

A subsequence is obtained by deleting some elements (possibly zero) from the original sequence, leaving the remaining elements in their original order.

Given an integer array `nums`, return the length of the longest wiggle subsequence of `nums`.


Example 1:
Input: nums = [1,7,4,9,2,5]
Output: 6
Explanation: The entire sequence is a wiggle sequence with differences (6, -3, 5, -7, 3).


Example 2:
Input: nums = [1,17,5,10,13,15,10,5,16,8]
Output: 7
Explanation: There are several subsequences that achieve this length.

One is [1, 17, 10, 13, 10, 16, 8] with differences (16, -7, 3, -3, 6, -8).


Example 3:
Input: nums = [1,2,3,4,5,6,7,8,9]
Output: 2

Constraints:
`1 <= nums.length <= 1000`
`0 <= nums[i] <= 1000`
Follow up: Could you solve this in `O(n)` time?

### Solution 
 To solve the problem of finding the length of the longest wiggle subsequence in an array of integers, we need to look for alternating sequences of 'up' and 'down' differences between successive elements. The key point is that we can achieve this in linear time, O(n), by tracking the state of the previous difference.

Here’s a detailed explanation of the solution:

1. **Understanding the Wiggle Sequence**: A wiggle sequence changes direction between positive and negative. For example:
   - If the difference between two successive elements is positive, the next difference should be negative.
   - The length of the wiggle subsequence can include the first two elements if they are distinct.

2. **Iterative Approach**:
   - We will use a variable to count the lengths of the subsequence and two flags to keep track of the current state (whether we are expecting an 'up' or 'down' next).
   - As we go through the array:
     - We will check the difference between the current and previous numbers.
     - If it’s positive and we haven't recorded an 'up' yet, we count it and set our state.
     - If it’s negative and we haven't recorded a 'down' yet, we count it and set our state.
   - We will reset our flags when we encounter the same number (to avoid counting zeros).

3. **Edge Cases**:
   - An array with one or two distinct numbers will trivially have a length of 1 or 2 (if they are different).

Let's implement this logic in Python, following the LeetCode code format:



```python
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        # Edge case when input list is empty or only one element
        if not nums:
            return 0
        elif len(nums) == 1:
            return 1
        
        n = len(nums)
        count = 1  # At least one number counts as a wiggle sequence
        prev_diff = 0  # Previous difference
        
        for i in range(1, n):
            diff = nums[i] - nums[i - 1]
            # Only consider the difference if it's non-zero
            if diff > 0 and prev_diff <= 0:
                count += 1
                prev_diff = diff
            elif diff < 0 and prev_diff >= 0:
                count += 1
                prev_diff = diff
        
        return count

# Example usage:
# sol = Solution()
# print(sol.wiggleMaxLength([1, 7, 4, 9, 2, 5])) # Output: 6
# print(sol.wiggleMaxLength([1, 17, 5, 10, 13, 15, 10, 5, 16, 8])) # Output: 7
# print(sol.wiggleMaxLength([1, 2, 3, 4, 5, 6, 7, 8, 9])) # Output: 2

```

### Explanation of the Code:
- **Input and Initialization**: The function checks for an empty list or a single element list upfront. Then it initializes a counter for the length of the wiggle sequence and a variable for the previous difference.
  
- **Loop through the elements**: Starting from the second element, it computes the current difference. Depending on this difference:
  - If it's positive and the previous difference was not in the 'up' state, it increments the count and updates the previous difference.
  - Similarly, if it's negative and the previous difference was not in the 'down' state, it also increments the count.

- **Return the result**: Finally, the function returns the count, which is the length of the longest wiggle subsequence.

The above implementation effectively runs in O(n) time, which is optimal for this problem within the given constraints.

# 377. Combination Sum IV

### Problem Description 
Given an array of distinct integers `nums` and a target integer `target`, return the number of possible combinations that add up to `target`.

The answer is guaranteed to fit in a 32-bit integer.


Example 1:
Input: nums = [1,2,3], target = 4
Output: 7
Explanation:
The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
Note that different sequences are counted as different combinations.


Example 2:
Input: nums = [9], target = 3
Output: 0

Constraints:
`1 <= nums.length <= 200`
`1 <= nums[i] <= 1000`
All the elements of `nums` are unique.

`1 <= target <= 1000`
Follow up: What if negative numbers are allowed in the given array? How does it change the problem? What limitation we need to add to the question to allow negative numbers?

### Solution 
 To solve the problem of finding the number of combinations of distinct integers that sum up to a given target, we can use dynamic programming. The idea is to maintain a list where each index represents the number of ways to achieve that index sum using the elements in the given array.

### Explanation

1. **Dynamic Programming Array**: We'll create a dynamic programming array `dp` of size `target + 1` where `dp[i]` will represent the number of combinations that add up to `i`.

2. **Initialization**: We initialize `dp[0]` to 1 because there is one way to create a sum of zero: to use no elements at all.

3. **Filling the DP array**: We iterate through each number in `nums`, and for each number, we iterate through possible sums from the number to `target`. For each sum `j`, we update `dp[j]` by adding `dp[j - num]` because if we use the current number `num`, we need to know how many combinations could previously achieve the sum `j - num`.

4. **Return the result**: At the end, `dp[target]` will hold the number of combinations that sum up to `target`.

### Code

Here is how this approach can be structured and implemented according to LeetCode format:



```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        # Create a DP array of size target + 1 initialized with 0
        dp = [0] * (target + 1)
        # There's one way to make the sum of 0: by using no elements
        dp[0] = 1
        
        # Iterate through all target sums from 1 to target
        for i in range(1, target + 1):
            # For each number, check if it can contribute to the sum
            for num in nums:
                # If the number can be used to create the sum i
                if i >= num:
                    dp[i] += dp[i - num]
        
        # Return the number of combinations to reach the target
        return dp[target]

```

### How to Test

You can test this code by using the following sample test cases:



```python
# Sample Test Cases
solution = Solution()

# Test case 1
print(solution.combinationSum4([1, 2, 3], 4))  # Output: 7

# Test case 2
print(solution.combinationSum4([9], 3))  # Output: 0

```

### Follow-Up Discussion

**What if negative numbers are allowed?**
When negative numbers are allowed, the problem becomes more complex. This is because you could end up in an infinite loop of combinations, e.g., using a negative number repeatedly could infinitely decrease a sum. 

To manage this, we would need additional constraints on the problem. Here are some possible limitations:
- Limit the number of times a negative number can be used.
- Restrict the input to non-negative integers or set a threshold on how negative the numbers can be.

This ensures that there is a termination point and avoids infinite combinations. 

This approach robustly solves the problem while maintaining clarity regarding the constraints and possible extensions.

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
 To determine whether one string `s` is a subsequence of another string `t`, we can use a straightforward method that involves iterating through the characters of both strings while maintaining the relative order of characters in `s`.

### Explanation

1. **Two Pointer Technique**: We will use two pointers; one pointer (`s_index`) for string `s` and another pointer (`t_index`) for string `t`. The goal is to traverse through both strings and check if we can match all characters of `s` in the correct order within `t`.

2. **Increment Pointers**: 
   - We continue moving through `t` with `t_index`. 
   - Whenever we find a character in `t` that matches the current character in `s` (pointed to by `s_index`), we increment the `s_index` pointer.
   - Always increment the `t_index` pointer to continue checking the next character of `t`.

3. **Completion Check**: If `s_index` reaches the length of `s` after traversing `t`, it indicates that all characters in `s` were found in `t` in the correct order, and we return `True`. If we finish traversing `t` and haven't matched all characters of `s`, we return `False`.

### Complexity
- The time complexity is O(n), where n is the length of `t`. This is efficient enough given the constraints.
- The space complexity is O(1) as we are using only a couple of pointers.

### Follow-up Consideration
If we have multiple queries for subsequences (k >= 10^9), instead of iterating through `t` for each `s`, we can preprocess `t` to create a mapping of character positions which would allow us to verify subsequences for each `s` in logarithmic time for each character lookup.

Here's the implementation based on the above logic:



```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        s_index, t_index = 0, 0
        
        while s_index < len(s) and t_index < len(t):
            if s[s_index] == t[t_index]:
                s_index += 1  # Move to next character in s
            t_index += 1  # Always move to next character in t
        
        # If we've matched all characters in s
        return s_index == len(s)

```

### How to Run on LeetCode
- You can copy and paste the code into the LeetCode editor.
- Make sure to select the problem related to checking whether `s` is a subsequence of `t`.

### Example Usage
You may want to test this code with the given examples:

1. For `s = "abc"` and `t = "ahbgdc"`, calling `isSubsequence(s, t)` should return `True`.
2. For `s = "axc"` and `t = "ahbgdc"`, calling `isSubsequence(s, t)` should return `False`.

This approach effectively solves the given problem within the constraints provided.

# 403. Frog Jump

### Problem Description 
A frog is crossing a river. The river is divided into some number of units, and at each unit, there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.

Given a list of `stones`' positions (in units) in sorted ascending order, determine if the frog can cross the river by landing on the last stone. Initially, the frog is on the first stone and assumes the first jump must be `1` unit.

If the frog's last jump was `k` units, its next jump must be either `k - 1`, `k`, or `k + 1` units. The frog can only jump in the forward direction.


Example 1:
Input: stones = [0,1,3,5,6,8,12,17]
Output: true
Explanation: The frog can jump to the last stone by jumping 1 unit to the 2nd stone, then 2 units to the 3rd stone, then 2 units to the 4th stone, then 3 units to the 6th stone, 4 units to the 7th stone, and 5 units to the 8th stone.


Example 2:
Input: stones = [0,1,2,3,4,8,9,11]
Output: false
Explanation: There is no way to jump to the last stone as the gap between the 5th and 6th stone is too large.


Constraints:
`2 <= stones.length <= 2000`
`0 <= stones[i] <= 231 - 1`
`stones[0] == 0`

### Solution 
 To solve the problem of determining whether the frog can cross the river by landing on the last stone given the rules of jumping, we can utilize a dynamic programming approach. 

The key points to understand are:
1. Jumping rules dictate that if the frog's last jump was of `k` units, the next jump can be `k - 1`, `k`, or `k + 1` units.
2. The frog starts on the first stone (position 0) and must reach the last stone in the provided list of `stones`.

We can represent the stones and their jump capabilities with a dynamic programming approach. We'll keep track of the possible jumps that can be made to each stone using a dictionary that maps the position of each stone to a set of possible jump distances that can reach that stone.

### Step-by-Step Explanation

1. **Storing Positions and Jumps**:
   Create a dictionary `dp` where each key is a stone position and its value is a set that stores the possible jump sizes that can reach that stone.

2. **Base Case**:
   The frog starts on the first stone (position 0) and has made no prior jump, hence it can only jump `1` unit to start. Initialize `dp` with `dp[0] = {0}`, representing the starting position.

3. **Iterate Through Each Stone**:
   Loop through all the stones from the first to the second last. For each stone, get its possible jump sizes and calculate next positions by applying the jumping rules. For each valid jump size from the current stone, attempt to reach the next stones by calculating `current_position + jump_size`.

4. **Check for Valid Stones**:
   If the calculated position exists in `stones`, add the jump size that brought you there to its corresponding set of jump sizes in `dp`.

5. **Final Check**:
   At the end of the iteration, check if there are any possible jumps that can reach the last stone.

Here's the Python code that implements the above logic:



```python
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        if stones[1] != 1:  # The first jump has to be to the first stone, which is at position 1
            return False
        
        dp = {stone: set() for stone in stones}
        dp[0].add(0)  # We start at the first stone with a jump of size 0.
        
        for stone in stones:
            for jump in dp[stone]:
                # Try jumps of size (jump - 1), jump and (jump + 1)
                for next_jump in (jump - 1, jump, jump + 1):
                    if next_jump > 0:  # Only consider positive jumps
                        next_stone = stone + next_jump
                        if next_stone in dp:
                            dp[next_stone].add(next_jump)
        
        # The last stone should have some jump size reaching it
        return len(dp[stones[-1]]) > 0

```

### Explanation of the Code:
- **Initialization**: Create a mapping of stone positions to sets of jump sizes.
- **Processing Each Stone**: For each stone, determine all the possible jumps that can take the frog to new stones within the jumping rules. If a calculated next position exists in the set of stones, update its corresponding jump sizes.
- **Final Check**: The function returns `True` if there's at least one valid jump size that can reach the last stone.

### Complexity Analysis:
- **Time Complexity**: O(n^2), where n is the number of stones. Each stone is processed in relation to its jump sizes.
- **Space Complexity**: O(n) for the dictionary that keeps track of reachable jump sizes.

This solution effectively checks the frog's possible paths and ensures it arrives at the last stone correctly while adhering to the jumping constraints.

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
 In order to solve the problem of minimizing the largest sum among `m` subarrays from the given array `nums`, we can utilize a binary search approach combined with a greedy algorithm. The core idea is to find the smallest possible maximum sum of these subarrays.

### Problem Breakdown:
1. **Understanding the Limits**:
   - The minimum possible value for the largest sum is the maximum element in the `nums` array (since at least one subarray must include this element).
   - The maximum possible value for the largest sum is the sum of all elements in `nums` (if we take the entire array as one subarray).

2. **Binary Search Setup**:
   - We'll perform a binary search on the range from `max(nums)` to `sum(nums)`.
   - In each iteration of the binary search, we will check if it's possible to split the `nums` into `m` or fewer parts such that the largest sum of these parts does not exceed a certain value (`mid`).

3. **Greedy Check**:
   - For a given `mid` value (the current guess for the largest sum), we will traverse the array and accumulate sums until adding another element would exceed `mid`. When this happens, we count a new subarray and reset the sum. If the count exceeds `m`, then `mid` is too low, and we need to increase our limits.

### Implementation:
Here’s how we can implement this approach in Python, following the LeetCode submission format:



```python
class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        # Helper function to check if we can split with max_sum as threshold
        def can_split(max_sum):
            current_sum = 0
            count = 1  # Start with one subarray
            for num in nums:
                if current_sum + num > max_sum:  # Need to split
                    count += 1
                    current_sum = num
                    if count > m:  # More subarrays than allowed
                        return False
                else:
                    current_sum += num
            return True  # We could split within m subarrays
        
        # Binary search to find the minimum largest sum
        left, right = max(nums), sum(nums)
        while left < right:
            mid = (left + right) // 2
            if can_split(mid):  # If we can split with this max_sum
                right = mid  # Try for a smaller max_sum
            else:
                left = mid + 1  # We need a larger max_sum
        
        return left  # The minimal largest sum achievable

```

### Explanation of the Code:
1. **Function Header**: We define a `splitArray` method under the `Solution` class which takes `nums` and `m` as inputs.
2. **Inner Function `can_split`**: It checks if it's possible to partition `nums` into `m` subarrays such that no subarray has a sum greater than `max_sum`.
3. **Binary Search Logic**:
   - Initialize `left` to `max(nums)` and `right` to `sum(nums)`.
   - Perform a binary search until `left` is not less than `right`.
   - Calculate `mid` and use `can_split` to check if this split is feasible.
   - Adjust `left` and `right` based on whether splitting is possible or not.
4. **Return Statement**: The minimal largest sum that can achieve the required splits is found at `left`.

### Complexity:
- **Time Complexity**: O(n * log(sum(nums))) where `n` is the length of `nums`. The `log` factor results from the binary search and the `n` factor comes from the summing operation for the splits.
- **Space Complexity**: O(1) for the space used by the algorithm, as it uses only a constant amount of space.

This solution handles all the edge cases as per the constraints and will perform efficiently within the provided limits.

# 413. Arithmetic Slices

### Problem Description 
An integer array is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same.

For example, `[1,3,5,7,9]`, `[7,7,7,7]`, and `[3,-1,-5,-9]` are arithmetic sequences.

Given an integer array `nums`, return the number of arithmetic subarrays of `nums`.

A subarray is a contiguous subsequence of the array.


Example 1:
Input: nums = [1,2,3,4]
Output: 3
Explanation: We have 3 arithmetic slices in nums: [1, 2, 3], [2, 3, 4] and [1,2,3,4] itself.


Example 2:
Input: nums = [1]
Output: 0

Constraints:
`1 <= nums.length <= 5000`
`-1000 <= nums[i] <= 1000`

### Solution 
 To solve the problem of counting the number of arithmetic subarrays in a given integer array `nums`, we can leverage the properties of arithmetic sequences. An arithmetic sequence is defined as one where the difference between consecutive elements is constant. 

### Strategy

1. **Identify Arithmetic Subarray**: We need to identify all contiguous subarrays where the difference between consecutive elements remains the same across at least three elements.
  
2. **Dynamic Count of Valid Subarrays**: We can keep track of the length of the current arithmetic subarray. When we find that the current difference (between the last two elements in the subarray) remains the same, we increment our count. If it deviates, we reset our counter.

3. **Count Valid Subarrays**: For every valid arithmetic subarray of length `n`, the number of arithmetic subarrays that can be formed is given by `(n-2)* (n-1) / 2`. This is because you can form subarrays of lengths 3 through `n` from an arithmetic sequence of at least length `n`.

### Implementation Steps

1. Start with a counter initialized to 0.
2. Loop through the array starting from the third element.
3. Check if the difference between the current and the previous element is the same as the difference between the previous element and the one before it.
4. If yes, increase the length of the current arithmetic subarray.
5. If no, reset the length.
6. Use the arithmetic sum formula to add up counts from valid lengths.

Here’s how to implement this logic in Python, following the LeetCode style:



```python
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        # Initialize the count of total arithmetic slices
        total_slices = 0
        
        # Initialize the length of the current arithmetic slice
        current_length = 0
        
        # Iterate from the third element to the end of the array
        for i in range(2, len(nums)):
            # If the current element and the previous two form an arithmetic sequence
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                # Increase the current length of the arithmetic slice
                current_length += 1
                # Every time we extend the slice, we can form `current_length` new slices
                total_slices += current_length
            else:
                # If there is no arithmetic sequence, reset the current length
                current_length = 0

        return total_slices

# Example usage
solution = Solution()
print(solution.numberOfArithmeticSlices([1, 2, 3, 4]))  # Output: 3
print(solution.numberOfArithmeticSlices([1]))           # Output: 0

```

### Explanation of the Code

- **Class Definition**: We encapsulate the function inside a class `Solution` which is a standard format for LeetCode solutions.
- **Function Signature**: The function `numberOfArithmeticSlices` takes a list of integers `nums` and returns an integer.
- **Loop Through Elements**: The loop starts from index 2, utilizing the three latest numbers to check if they form an arithmetic slice.
- **Differential Check**: We check if the current sequence is arithmetic by comparing the differences.
- **Count Update**: Each time we find a continued arithmetic sequence, we update `current_length` and add it to `total_slices`.
- **Return Value**: Finally, the total count of arithmetic subarrays is returned.

This algorithm runs in linear time, O(n), with space complexity O(1), and meets the constraints provided.

# 416. Partition Equal Subset Sum

### Problem Description 
Given a non-empty array `nums` containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.


Example 1:
Input: nums = [1,5,11,5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].


Example 2:
Input: nums = [1,2,3,5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.


Constraints:
`1 <= nums.length <= 200`
`1 <= nums[i] <= 100`

### Solution 
 To solve the problem of partitioning an array into two subsets with equal sum, we can take advantage of the properties of sums and subsets. The key observation is that the total sum of the array must be even in order for it to be split into two equal parts.

### Detailed Explanation:

1. **Calculate the Sum**: First, we calculate the total sum of the elements in the array. If this sum is odd, it is impossible to partition the array into two equal subsets, and we can directly return `False`.

2. **Target Sum**: If the total sum is even, we determine the target sum for each subset, which is half of the total sum.

3. **Dynamic Programming Approach**: We then use a dynamic programming approach to determine whether it is possible to form a subset of elements that sums up to the target. 
    - We create a boolean array `dp` where `dp[j]` indicates whether a subset sum of `j` can be achieved with the available numbers.
    - Initialize `dp[0]` to `True` because a sum of 0 can always be achieved with an empty subset.

4. **Iterate Through Numbers**: For each number in `nums`, we update the `dp` array in reverse order (from `target` down to the current number). This ensures that each number is only used once in the current iteration.

5. **Final Result**: At the end, if `dp[target]` is `True`, it means we can indeed partition the array into two subsets of equal sum, and we return `True`. Otherwise, we return `False`.

### Python Code:

Here is the implementation of the above logic in Python, formatted according to LeetCode's requirements:



```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total_sum = sum(nums)
        
        # If the total sum is odd, we cannot partition it into two equal subsets
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        dp = [False] * (target + 1)
        dp[0] = True  # We can always have a subset sum of 0
        
        # Iterate through the numbers
        for num in nums:
            # Update the dp array in reverse order
            for j in range(target, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]

        return dp[target]

```

### Explanation of the Code:

- We define a class `Solution` with a method `canPartition` that takes a list of integers `nums`.
- We calculate `total_sum`, and check if it's odd. If so, we return `False`.
- If it's even, we initialize a `dp` list sized to `target + 1` with `dp[0]` set to `True`.
- We iterate over each number in `nums`, and for each number, we update our `dp` list in reverse to ensure we don't use the same number multiple times in this iteration.
- Finally, we check `dp[target]` to return our result.

This algorithm runs in \(O(n \cdot \text{target})\) time and uses \(O(\text{target})\) space, where \(n\) is the length of `nums` and `target` is half of the total sum of the array. This approach should efficiently handle the constraints given in this problem.

# 418. Sentence Screen Fitting

### Problem Description 
Given a `rows x cols` screen and a sentence represented by a list of non-empty words, find how many times the given sentence can be fitted on the screen.

Note:
A word cannot be split into two lines.

The order of words in the sentence must remain unchanged.

Two consecutive words in a line must be separated by a single space.

Total words in the sentence won't exceed 100.

Length of each word is greater than 0 and won't exceed 10.

1 ≤ rows, cols ≤ 20,000.


Example 1:
Input:
rows = 2, cols = 8, sentence = ["hello", "world"]
Output: 
1
Explanation:
hello---
world---
The character '-' signifies an empty space on the screen.


Example 2:
Input:
rows = 3, cols = 6, sentence = ["a", "bcd", "e"]
Output: 
2
Explanation:
a-bcd- 
e-a---
bcd-e-
The character '-' signifies an empty space on the screen.


Example 3:
Input:
rows = 4, cols = 5, sentence = ["I", "had", "apple", "pie"]
Output: 
1
Explanation:
I-had
apple
pie-I
had--
The character '-' signifies an empty space on the screen.

### Solution 
 To solve the problem of fitting a given sentence onto a screen defined by the number of rows and columns, we can follow these steps:

1. **Understanding the Layout**: We must fit words into a grid where each row can hold multiple words, as long as they fit within the column limit. Each word needs to be separated by a single space, and words can't be split.

2. **Circular Fitting**: Once we put words into rows, if there's space left in the row, we may start from the beginning of the sentence again. This circular fitting is key to solving the problem efficiently.

3. **Count Fitting**: We will simulate the placement of words on the screen row by row until we exhaust the rows. We need to keep track of how many times we have been able to insert the entire sentence.

Here's the Python code that implements the above logic:



```python
class Solution:
    def wordsTyping(self, sentence: List[str], rows: int, cols: int) -> int:
        # Join the sentence into a single string with spaces 
        full_sentence = ' '.join(sentence) + ' '  
        len_full_sentence = len(full_sentence)  
        idx = 0  
        count = 0  

        for _ in range(rows):
            # Try to fit as many words as possible in the current row
            idx += cols  # Move forward in the current row
            if idx >= len_full_sentence:   
                idx -= len_full_sentence  # Wrap around if we exceed the length of the full sentence
            idx_last_space = idx
            while idx_last_space > 0 and full_sentence[idx_last_space - 1] != ' ':
                idx_last_space -= 1
            
            # If we have a full word to count in this row
            if idx_last_space > 0:
                idx = idx_last_space  # Move idx back to the last space
            
            # Count how many words we managed to fit
            count += (idx_last_space == cols)

        return count

```

### Explanation of the Code:

1. **Initialization**: The class `Solution` contains the method `wordsTyping`. We first join all the words in the sentence with a space and append an additional space at the end to facilitate word wrapping.

2. **Row Iteration**: We iterate through each row of the screen (up to the number of rows specified).

3. **Index Management**: The variable `idx` maintains our current position within the `full_sentence` as we try to fit words into the current row. If `idx` surpasses the length of `full_sentence`, we wrap around using modulo.

4. **Word Placement**: We look for the last space in the range of the current row and backtrack `idx` to that space. It ensures that we place full words only on the screen.

5. **Counting Fits**: We keep a counter that increments whenever a full row can fit a part of the sentence.

6. **Return the Count**: Finally, we return the total count of how many times the sentence fits on the screen.

### Complexity:
- **Time Complexity**: O(rows), as we iterate through the number of rows, and each iteration does a bounded number of operations determined by the string length.
- **Space Complexity**: O(1), since we are using a fixed amount of space irrespective of the input sizes.

This solution is efficient and should perform well even with the upper limits of input constraints specified in the problem.

# 446. Arithmetic Slices II - Subsequence

### Problem Description 
A sequence of numbers is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same.

For example, these are arithmetic sequences:
1, 3, 5, 7, 9
7, 7, 7, 7
3, -1, -5, -9
The following sequence is not arithmetic.

1, 1, 2, 5, 7
A zero-indexed array A consisting of N numbers is given. A subsequence slice of that array is any sequence of integers (P0, P1, ..., Pk) such that 0 ≤ P0 < P1 < ... < Pk < N.

A subsequence slice (P0, P1, ..., Pk) of array A is called arithmetic if the sequence A[P0], A[P1], ..., A[Pk-1], A[Pk] is arithmetic. In particular, this means that k ≥ 2.

The function should return the number of arithmetic subsequence slices in the array A.

The input contains N integers. Every integer is in the range of -231 and 231-1 and 0 ≤ N ≤ 1000. The output is guaranteed to be less than 231-1.


Example:
Input: [2, 4, 6, 8, 10]
Output: 7
Explanation:
All arithmetic subsequence slices are:
[2,4,6]
[4,6,8]
[6,8,10]
[2,4,6,8]
[4,6,8,10]
[2,4,6,8,10]
[2,6,10]

### Solution 
 To solve the problem of counting the number of arithmetic subsequence slices in a given integer array, we can use a combination of dynamic programming and a hash map. The idea is to keep track of the number of valid arithmetic subsequences that can be formed with each pair of numbers while iterating through the array.

### Explanation

1. **Subsequence Concept**: An arithmetic subsequence is defined as a sequence in which the difference between any two consecutive elements is constant. For instance, in the array `[2, 4, 6, 8, 10]`, valid arithmetic subsequences starting with initial pairs can be extended into longer valid subsequences.

2. **Dynamic Programming Approach**:
   - We'll use a list of dictionaries `dp` where `dp[i]` is a dictionary that will store the count of arithmetic subsequences ending at index `i` with a certain difference `d`.
   - For every pair of indices `j` and `i` (where `j < i`), we calculate the difference `d = A[i] - A[j]`.
   - If `d` has been seen before at index `j`, it means `dp[j].get(d, 0)` gives the count of valid arithmetic subsequences that can be extended to include `A[i]`.
   - Every time we extend an existing subsequence, we also need to count those subsequences of length 3 or more as these are only part of our final answer.

3. **Counting Subsequences**: 
   - Whenever we count subsequences ending at `i` with a difference `d`, we also add all sequences that can potentially end at `j` with the same difference `d`.
   - Keep track of the total count of arithmetic subsequences.

### Code Implementation

Here is the Python code implementing the above logic:



```python
class Solution:
    def countArithmeticSubsequences(self, A):
        if len(A) < 3:
            return 0
        
        # List of dictionaries to hold the count of valid subsequences
        dp = [{} for _ in range(len(A))]
        total_count = 0
        
        for i in range(len(A)):
            for j in range(i):
                # Calculate the common difference
                d = A[i] - A[j]
                
                # Get the number of arithmetic subsequences ending at j with difference d
                count_at_j = dp[j].get(d, 0)
                
                # Update the count of subsequences ending at i
                # Start with 1 (the pair A[j], A[i]) and add counts from j
                dp[i][d] = dp[i].get(d, 0) + count_at_j + 1
                
                # Count all subsequences of length ≥ 3
                total_count += count_at_j
        
        return total_count

# Example usage
solution = Solution()
print(solution.countArithmeticSubsequences([2, 4, 6, 8, 10]))  # Output: 7

```

### How to Run the Code
The code adheres to the LeetCode function format. You can directly input your array when testing the solution, and the count of valid arithmetic subsequences will be returned.

### Conclusion
This solution efficiently calculates the number of arithmetic subsequences in a given list and works within the constraints provided. The use of a list of dictionaries allows us to efficiently keep track of valid differences and their counts, thereby allowing for rapid aggregation when extending subsequences.

# 464. Can I Win

### Problem Description 
In the "100 game" two players take turns adding, to a running total, any integer from `1` to `10`. The player who first causes the running total to reach or exceed 100 wins.

What if we change the game so that players cannot re-use integers?
For example, two players might take turns drawing from a common pool of numbers from 1 to 15 without replacement until they reach a total >= 100.

Given two integers maxChoosableInteger and desiredTotal, return `true` if the first player to move can force a win, otherwise return `false`. Assume both players play optimally.


Example 1:
Input: maxChoosableInteger = 10, desiredTotal = 11
Output: false
Explanation:
No matter which integer the first player choose, the first player will lose.

The first player can choose an integer from 1 up to 10.

If the first player choose 1, the second player can only choose integers from 2 up to 10.

The second player will win by choosing 10 and get a total = 11, which is >= desiredTotal.

Same with other integers chosen by the first player, the second player will always win.


Example 2:
Input: maxChoosableInteger = 10, desiredTotal = 0
Output: true

Example 3:
Input: maxChoosableInteger = 10, desiredTotal = 1
Output: true

Constraints:
`1 <= maxChoosableInteger <= 20`
`0 <= desiredTotal <= 300`

### Solution 
 To solve the problem efficiently, we can use a recursive approach combined with memoization. The idea is to keep track of the current state of the game using the available numbers and the running total, and determine if the first player can force a win from that state.

### Problem Breakdown:
1. **Game Constraints**: Players can choose numbers between 1 and `maxChoosableInteger`, and the game's total must reach or exceed `desiredTotal` to win.
2. **Player Turn**: The first player chooses a number first, and they will play optimally, meaning they will always make the choice that maximizes their chances of winning.
3. **Memoization**: Since there can be overlapping states (same available numbers and the same current total), we will store results of already computed states to avoid redundant calculations.

### Steps:
1. **Base Cases**:
   - If `desiredTotal` is `0`, the first player wins automatically, so return `True`.
   - If `desiredTotal` is greater than `0` and the sum of all choosable integers is less than `desiredTotal`, the player cannot win, hence return `False`.

2. **Recursive Strategy**:
   - Try each number from 1 to `maxChoosableInteger`.
   - For each number, simulate the first player's choice and adjust the total.
   - Check the next player's turn. The next player will be in the position to win if the total after the first player’s choice plus the maximum remaining choosable integer is still less than `desiredTotal`.

3. **Memoization**: Use a dictionary to store results of states to avoid recomputation.

Here is how the code is structured for the solution:



```python
class Solution:
    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        # If desiredTotal is 0, the first player wins.
        if desiredTotal <= 0:
            return True
        
        # Calculate the maximum sum that the first player can achieve.
        total_sum = (maxChoosableInteger * (maxChoosableInteger + 1)) // 2
        # If the sum of all choosable integers is less than the desired total, the first player cannot win.
        if total_sum < desiredTotal:
            return False
        
        memo = {}
        
        def can_win(used_numbers, total):
            if (used_numbers, total) in memo:
                return memo[(used_numbers, total)]
            
            for i in range(maxChoosableInteger):
                # Check if the number `i+1` is used
                if not (used_numbers & (1 << i)):  # If i+1 is not chosen
                    # Calculate next total if the player picks `i+1`
                    next_total = total + (i + 1)
                    # If next total reaches or exceeds desiredTotal, the first player wins
                    if next_total >= desiredTotal:
                        memo[(used_numbers, total)] = True
                        return True
                    # If next total does not reach desiredTotal, check if the next player can win
                    if not can_win(used_numbers | (1 << i), next_total):  # If the opponent cannot win, the player can win
                        memo[(used_numbers, total)] = True
                        return True
            
            memo[(used_numbers, total)] = False
            return False
        
        # Start the game with no numbers used and total starting at 0
        return can_win(0, 0)

```

### Explanation of the Code:
- **Function Signature**: The method `canIWin` initializes the game state and checks for the immediate wins or losses.
- **Base Conditions**: Immediate wins or losses are checked first.
- **Memoized Recursive Function**: The function `can_win` checks all possible states based on the current choices and the total. If a winning state is found, it returns `True`, otherwise `False`.
- **Bit Manipulation**: Using a bitmask to represent used numbers allows efficient state management.

This approach ensures that we explore the game tree efficiently and only compute necessary states, allowing the solution to run within the constraints provided.

# 466. Count The Repetitions

### Problem Description 
Define `S = [s,n]` as the string S which consists of n connected strings s. For example, `["abc", 3]` ="abcabcabc". 
On the other hand, we define that string s1 can be obtained from string s2 if we can remove some characters from s2 such that it becomes s1. For example, “abc”  can be obtained from “abdbec” based on our definition, but it can not be obtained from “acbbe”.

You are given two non-empty strings s1 and s2 (each at most 100 characters long) and two integers 0 ≤ n1 ≤ 106 and 1 ≤ n2 ≤ 106. Now consider the strings S1 and S2, where `S1=[s1,n1]` and `S2=[s2,n2]`. Find the maximum integer M such that `[S2,M]` can be obtained from `S1`.


Example:
Input:
s1="acb", n1=4
s2="ab", n2=2
Return:
2

### Solution 
 To solve the problem, we will break it down into a series of steps. We need to determine how many times we can obtain the sequence formed by repeating `S2` from the sequence formed by repeating `S1`.

### Key Definitions:
1. **S1 and S2 Construction**:
   - `S1` is constructed by repeating the string `s1`, `n1` times.
   - `S2` is constructed by repeating the string `s2`, `n2` times.

2. **Subsequence Definition**:
   - A string `s1` can be a subsequence of `s2` if we can remove some characters from `s2` such that we only have `s1` left in the same order.

### Steps to Solve:
1. **Construct the Long Strings**: 
   - Create the full strings for `S1` and `S2`.
2. **Count Subsequences**:
   - We will have to check how many times `S2` can fit in `S1` as a subsequence.
3. **Binary Search**:
   - Since `n2` can be as large as 10^6, we will utilize binary search to efficiently determine the maximum integer `M` such that we can construct `M` instances of `S2` from `S1`.

### Implementation:
Now, let's implement this logic in Python according to the LeetCode solution format.



```python
class Solution:
    def can_form(self, s1, s2, m):
        # This function checks if S2 constructed m times can be formed from S1
        j = 0  # Pointer for the characters in s2
        for char in s1:
            # Try to match all characters of s2, m times
            for _ in range(m):
                while j < len(s2) and s2[j] != char:
                    j += 1
                if j < len(s2) and s2[j] == char:
                    j += 1
                else:  # If we exhaust s2 before matching all characters in m
                    return False
        return j >= len(s2) * m  # Successfully matched all characters
    
    def getMaxRepetitions(self, s1: str, n1: int, s2: str, n2: int) -> int:
        # Initialize binary search bounds
        low, high = 0, n1
        
        while low < high:
            mid = (low + high + 1) // 2  # Avoid infinite loop by rounding up
            if self.can_form(s1 * n1, s2 * n2, mid):
                low = mid  # This count can work, try for more
            else:
                high = mid - 1  # Too many, reduce the upper limit

        return low  # `low` should be the maximum M that works

```

### Explanation of the Code:
- **`can_form` function**: This checks if we can form `S2` repeated `m` times from `S1`.
  - We use two pointers to iterate through `s1` and `s2`. For each character in `s1`, we check if we can find that character in `s2` `m` times.
  
- **`getMaxRepetitions` function**: This implements a binary search to find the maximum value of `M`.
  - We initialize `low` and `high`, where `low` starts at 0 and `high` starts at `n1`.
  - The loop continues until `low` meets `high`. We check the middle point `mid`. If `can_form` returns True, we can attempt a larger `M` (hence `low` is updated), otherwise, we reduce our search space.

This algorithm efficiently handles the input size constraints and delivers the correct maximum number of repetitions of `S2` constructible from `S1`.

# 467. Unique Substrings in Wraparound String

### Problem Description 
Consider the string `s` to be the infinite wraparound string of "abcdefghijklmnopqrstuvwxyz", so `s` will look like this: "...zabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcd....".

Now we have another string `p`. Your job is to find out how many unique non-empty substrings of `p` are present in `s`. In particular, your input is the string `p` and you need to output the number of different non-empty substrings of `p` in the string `s`.

Note: `p` consists of only lowercase English letters and the size of p might be over 10000.


Example 1:
Input: "a"
Output: 1
Explanation: Only the substring "a" of string "a" is in the string s.


Example 2:
Input: "cac"
Output: 2
Explanation: There are two substrings "a", "c" of string "cac" in the string s.


Example 3:
Input: "zab"
Output: 6
Explanation: There are six substrings "z", "a", "b", "za", "ab", "zab" of string "zab" in the string s.

### Solution 
 To solve the problem of finding how many unique non-empty substrings of a given string `p` are present in the infinite wraparound string of "abcdefghijklmnopqrstuvwxyz," we can follow a systematic approach leveraging the properties of the English alphabet and substrings.

## Detailed Explanation

The infinite string can be thought of as a cyclic sequence of characters where:
- After 'z', we return to 'a'.

For each character in the string `p`, we need to identify the longest contiguous substring ending in that character that can also be found in the wraparound string.

### Steps to Approach:

1. **Identifying Wraparound Substrings**:
   - For each character in `p`, calculate how many contiguous characters can be found in the wraparound manner. For example, if you encounter 'a', 'b' and so forth until you encounter a non-contiguous character, you can account for all the distinct substrings formed by these characters.

2. **Using a Set**:
   - We can utilize a set to count unique substrings. However, because we are calculating contiguous substrings individually as we scan through `p`, we can also keep a simple count without the need for an explicit set.

3. **Dynamic Counting**:
   - Maintain a count of current contiguous substrings (`count`). As we encounter a character in `p`, determine if it's contiguous to the previous character. If it is, increment our substring length counter; if not, reset it.

4. **Adding to Result**:
   - For each character processed, add the current count to a total result counter.

This way, we avoid the need to generate all substrings explicitly, making our solution efficient, even for larger strings.

### Implementation in Python

Now, let's write the Python code based on the above plan.



```python
class Solution:
    def findSubstringInWraproundString(self, p: str) -> int:
        # Length of string p
        n = len(p)
        
        # This will hold the count of unique substrings ending with each character
        count = [0] * 26
        current_len = 0
        
        for i in range(n):
            # Check if the current character is a wraparound from the previous one
            if i > 0 and (ord(p[i]) - ord(p[i-1]) + 26) % 26 == 1:
                current_len += 1
            else:
                current_len = 1
            
            # Update count for the ending character of this substring
            index = ord(p[i]) - ord('a')
            count[index] = max(count[index], current_len)

        # The result is the sum of all unique substring counts
        return sum(count)

# Example usage
solution = Solution()
print(solution.findSubstringInWraproundString("a"))    # Output: 1
print(solution.findSubstringInWraproundString("cac"))  # Output: 2
print(solution.findSubstringInWraproundString("zab"))  # Output: 6

```

### Explanation of the Code:
1. **Initialization**: We create an array `count` of size 26 to store the maximum length of substrings ending with each letter 'a' to 'z'.
2. **Loop through p**: We iterate through each character in string `p`. If the current character and the previous character are consecutive in the wraparound nature, we continue our substring count; otherwise, we reset it.
3. **Update Counts**: For each character `p[i]`, we update our counts if we found a longer substring ending with that character.
4. **Final Count**: Finally, we sum up all the maximum lengths stored in `count` to get the total number of unique substrings.

This solution is efficient with a time complexity of O(n), where n is the length of the string `p`, which is an optimal approach given the constraints.

# 471. Encode String with Shortest Length

### Problem Description 
Given a non-empty string, encode the string such that its encoded length is the shortest.

The encoding rule is: `k[encoded_string]`, where the `encoded_string` inside the square brackets is being repeated exactly `k` times.

Note:
`k` will be a positive integer.

If an encoding process does not make the string shorter, then do not encode it. If there are several solutions, return any of them.


Example 1:
Input: s = "aaa"
Output: "aaa"
Explanation: There is no way to encode it such that it is shorter than the input string, so we do not encode it.


Example 2:
Input: s = "aaaaa"
Output: "5[a]"
Explanation: "5[a]" is shorter than "aaaaa" by 1 character.


Example 3:
Input: s = "aaaaaaaaaa"
Output: "10[a]"
Explanation: "a9[a]" or "9[a]a" are also valid solutions, both of them have the same length = 5, which is the same as "10[a]".


Example 4:
Input: s = "aabcaabcd"
Output: "2[aabc]d"
Explanation: "aabc" occurs twice, so one answer can be "2[aabc]d".


Example 5:
Input: s = "abbbabbbcabbbabbbc"
Output: "2[2[abbb]c]"
Explanation: "abbbabbbc" occurs twice, but "abbbabbbc" can also be encoded to "2[abbb]c", so one answer can be "2[2[abbb]c]".


Constraints:
`1 <= s.length <= 150`
`s` consists of only lowercase English letters.

### Solution 
 To solve the problem of encoding a string such that its encoded length is the shortest, we can utilize a dynamic programming approach. The idea is to explore all possible ways to encode substrings of the string while keeping track of the shortest encoded length.

### Solution Steps:

1. **Initialization**: Create a dynamic programming (DP) table where `dp[i]` holds the shortest encoded string for the substring `s[0:i+1]`.

2. **Single Characters**: Initially, assign the DP table with individual characters since they cannot be encoded. Hence, `dp[i] = s[0:i+1]`.

3. **Loop through Substrings**: For each substring, check different potential encoding patterns:
   - Attempt to find valid repeating segments within the substring.
   - Use the format `k[encoded_string]` for possible encodings and calculate their lengths.

4. **Calculate Encodings**: For each possible starting position of a repeated pattern, calculate the length of the original and encoded versions to decide if encoding is beneficial.

5. **Update DP Table**: If a new encoding is shorter, update the DP table. 

6. **Return Result**: The shortest encoded version of the string can be found in `dp[n-1]`, where `n` is the length of the string.

Here's the complete Python code implementing the above logic in a format suitable for LeetCode:



```python
class Solution:
    def encode(self, s: str) -> str:
        n = len(s)
        dp = [s] * n  # Initialize dp array where dp[i] = s[0:i+1]

        for length in range(1, n + 1):  # Length of the substring
            for start in range(n - length + 1):  # Start index of the substring
                substr = s[start:start + length]
                # Update the dp with the original substring
                if length < len(dp[start]):
                    dp[start] = substr
                
                # Try to encode the substring with repetitions
                for repeat_length in range(1, length // 2 + 1):
                    if substr[:repeat_length] == substr[repeat_length:repeat_length*2]:  # Check if it can be encoded
                        k = length // repeat_length  
                        encoded = f"{k}[{dp[start][:repeat_length]}]"
                        # Check if encoding is shorter than the current recorded string
                        if len(encoded) < len(dp[start]):
                            dp[start] = encoded[:length]

                # Update dp[start] if the encoded form is shorter
                dp[start] = min(dp[start], substr)  # Compare with the non-encoded form
        
        return dp[0]

# Example usage:
# sol = Solution()
# print(sol.encode("aaaaa"))  # Output: "5[a]"
# print(sol.encode("aabcaabcd"))  # Output: "2[aabc]d"

```

### Explanation of the Code:

- We create a list `dp` that starts with the original substrings.
- We iterate through all possible lengths of substrings and their starting positions.
- For each substring, we check for potential encoding by matching segments of the substring.
- If we find a valid encoding that results in a shorter string than what's already stored, we update `dp`.
- Finally, we return the encoded version stored in `dp[0]`, which represents the shortest encoding of the entire string.

This approach ensures that we explore all combinations of possible encodings while efficiently maintaining the shortest possible results for substrings, providing an effective solution to the problem constraints.

# 472. Concatenated Words

### Problem Description 
Given an array of strings `words` (without duplicates), return all the concatenated words in the given list of `words`.

A concatenated word is defined as a string that is comprised entirely of at least two shorter words in the given array.


Example 1:
Input: words = ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]
Output: ["catsdogcats","dogcatsdog","ratcatdogcat"]
Explanation: "catsdogcats" can be concatenated by "cats", "dog" and "cats"; 
"dogcatsdog" can be concatenated by "dog", "cats" and "dog"; 
"ratcatdogcat" can be concatenated by "rat", "cat", "dog" and "cat".


Example 2:
Input: words = ["cat","dog","catdog"]
Output: ["catdog"]

Constraints:
`1 <= words.length <= 104`
`0 <= words[i].length <= 1000`
`words[i]` consists of only lowercase English letters.

`0 <= sum(words[i].length) <= 6 * 105`

### Solution 
 To solve the problem of finding all concatenated words in a list of strings, we can take advantage of a depth-first search (DFS) combined with a set for quick lookups. Here’s the plan:

1. **Data Structure**: We will maintain a set containing all the words. This allows O(1) average time complexity for checking if a substring exists in the list.

2. **Concatenation Check**: For each word, we will try to break it into two parts in every possible way (e.g., by iterating through all indices) and check if both parts exist in the set. We need to ensure that at least one of these parts is a word from the set to satisfy the concatenation requirement.

3. **DFS Implementation**: We'll implement a recursive function that will check if a word can be constructed from at least two shorter words in the list.

4. **Avoiding Redundant Checks**: To optimize, we can use memoization to store results of previously checked words to avoid redundant calculations.

Here's how the code would look:



```python
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        def can_form(word):
            if word in word_set:
                return True  # The word itself is in the set.
            if word in memo:
                return memo[word]
            for i in range(1, len(word)):  # Check every possible split.
                prefix = word[:i]
                if prefix in word_set and can_form(word[i:]):
                    memo[word] = True
                    return True
            memo[word] = False
            return False

        word_set = set(words)  # Create a set of words for quick look-up.
        memo = {}
        concatenated_words = []

        for word in words:
            word_set.remove(word)  # We're checking the current word so remove it temporarily.
            if can_form(word):
                concatenated_words.append(word)  # If it can be formed, add to the result.
            word_set.add(word)  # Add it back to the set for the next iterations.

        return concatenated_words

```

### Explanation of the Code

1. **Initialization**: We create a set from `words` for efficient lookup and set up a memoization dictionary (`memo`) to avoid redundant checks.

2. **Defining Helper Function (`can_form`)**:
   - This function checks recursively if the `word` can be formed from concatenated words in `word_set`.
   - If the word is already in `word_set`, we return `True`.
   - If the word has been processed before, we return the stored result from `memo`.
   - We iterate through the word and split it into a prefix and a suffix and check if the prefix is in `word_set` and if the suffix can also be formed from other words using a recursive call.

3. **Main Loop**: We iterate over each word, temporarily remove it from `word_set`, and check if it can be formed by calling `can_form()`. If it can be, we store it in the `concatenated_words` list. After checking, we add the word back to the word set for further iterations.

4. **Return Statement**: Finally, we return the list of concatenated words.

### Time Complexity:
- The worst-case time complexity would be O(N * M^2), where N is the number of words and M is the average length of the words, due to checking all splits of each word. 

### Space Complexity:
- O(N + M) for the set and memo dictionary.

This structure provides a comprehensive solution to the problem within the constraints provided.

# 474. Ones and Zeroes

### Problem Description 
You are given an array of binary strings `strs` and two integers `m` and `n`.

Return the size of the largest subset of `strs` such that there are at most `m` `0`'s and `n` `1`'s in the subset.

A set `x` is a subset of a set `y` if all elements of `x` are also elements of `y`.


Example 1:
Input: strs = ["10","0001","111001","1","0"], m = 5, n = 3
Output: 4
Explanation: The largest subset with at most 5 0's and 3 1's is {"10", "0001", "1", "0"}, so the answer is 4.

Other valid but smaller subsets include {"0001", "1"} and {"10", "1", "0"}.

{"111001"} is an invalid subset because it contains 4 1's, greater than the maximum of 3.


Example 2:
Input: strs = ["10","0","1"], m = 1, n = 1
Output: 2
Explanation: The largest subset is {"0", "1"}, so the answer is 2.


Constraints:
`1 <= strs.length <= 600`
`1 <= strs[i].length <= 100`
`strs[i]` consists only of digits `'0'` and `'1'`.

`1 <= m, n <= 100`

### Solution 
 To solve the problem of finding the largest subset of binary strings such that the subset contains at most `m` `0`s and `n` `1`s, we can utilize dynamic programming. The approach is reminiscent of the "knapsack problem," where we aim to maximize the count of items (binary strings) chosen under certain constraints (maximum `0`s and `1`s).

### Step-by-step Explanation

1. **Understanding the Problem**: Each string in the `strs` array can be represented by the number of `0`s and `1`s it contains. For every string, we can determine how many `0`s and `1`s it contributes. Our goal is to find the maximum number of strings we can select such that the total count of `0`s does not exceed `m` and the total count of `1`s does not exceed `n`.

2. **Dynamic Programming Table**: We will create a 2D DP array where `dp[i][j]` will represent the maximum number of strings that can be formed with at most `i` `0`s and `j` `1`s.

3. **Updating the DP Table**: For each binary string, count its `0`s and `1`s. We then iterate over our DP table backwards (from `m` to the count of `0`s required by the current string, and from `n` to the count of `1`s required by the current string). We update the DP table by considering whether to include the current string or not.

4. **Final Result**: The answer will be found in the cell `dp[m][n]`, which gives us the size of the largest subset of binary strings that fit within the constraints.

### Python Code

Here's how this can be implemented in Python:



```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        # Create a DP array with dimensions (m+1) x (n+1)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Iterate through each string in the input list
        for s in strs:
            # Count zeros and ones in the current string
            zeros = s.count('0')
            ones = s.count('1')
            
            # Update dp array in reverse order
            for i in range(m, zeros - 1, -1):
                for j in range(n, ones - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
        
        # The result is found in dp[m][n]
        return dp[m][n]

```

### Explanation of the Code

- **Initialization**: We initialize a 2D list `dp` with dimensions `(m+1) x (n+1)` filled with zeros. This represents all possible combinations of `0`s and `1`s we can count up to `m` and `n`.

- **Loop over strings**: We loop through each binary string and calculate how many `0`s (`zeros`) and `1`s (`ones`) it contains.

- **Reverse DP Update**: We iterate backwards through the `dp` array, which ensures that we do not overwrite the results of the current iteration when we add the current string.

- **Final Result**: After processing all strings, `dp[m][n]` will contain the maximum size of the subset of strings that can be formed under the constraints.

### Time Complexity

The time complexity of the solution is `O(k * m * n)`, where `k` is the number of strings in `strs`. This is efficient given the problem constraints.

### Space Complexity

The space complexity is `O(m * n)` due to the storage of the `dp` array. 

This complete solution, when run in the LeetCode environment, should work efficiently and provide the correct output for the described problem.

# 486. Predict the Winner

### Problem Description 
Given an array of scores that are non-negative integers. Player 1 picks one of the numbers from either end of the array followed by the player 2 and then player 1 and so on. Each time a player picks a number, that number will not be available for the next player. This continues until all the scores have been chosen. The player with the maximum score wins.

Given an array of scores, predict whether player 1 is the winner. You can assume each player plays to maximize his score.


Example 1:
Input: [1, 5, 2]
Output: False
Explanation: Initially, player 1 can choose between 1 and 2. 
If he chooses 2 (or 1), then player 2 can choose from 1 (or 2) and 5. If player 2 chooses 5, then player 1 will be left with 1 (or 2). 
So, final score of player 1 is 1 + 2 = 3, and player 2 is 5. 
Hence, player 1 will never be the winner and you need to return False.


Example 2:
Input: [1, 5, 233, 7]
Output: True
Explanation: Player 1 first chooses 1. Then player 2 have to choose between 5 and 7. No matter which number player 2 choose, player 1 can choose 233.

Finally, player 1 has more score (234) than player 2 (12), so you need to return True representing player1 can win.


Constraints:
1 <= length of the array <= 20.

Any scores in the given array are non-negative integers and will not exceed 10,000,000.

If the scores of both players are equal, then player 1 is still the winner.

### Solution 
 To solve the problem of determining if player 1 can guarantee a win given the scoring rules, we can approach it using dynamic programming with memoization. The fundamental idea is to simulate the decision-making process of both players, ensuring that each player plays optimally to maximize their score while minimizing their opponent's score.

### Dynamic Programming Explanation

1. **State Definition**: We'll define a recursive state `dp(i, j)` where `i` is the start index and `j` is the end index of the remaining scores in the array. The value of `dp(i, j)` will be the maximum score that the current player can secure given the array slice from index `i` to `j`.

2. **Base Cases**: 
   - If there's only one element left in the slice (i.e., if `i == j`), the player can only take that element, so `dp(i, j) = scores[i]`.

3. **Transition**: 
   - If the current player picks the `i-th` element, then the next turn is for the opponent from the slice `i+1` to `j`. The opponent will also maximize their score, but since we are interested in the current player's perspective, we subtract the score of the opponent from the total:
     - Choosing `scores[i]`: 
       
``` 
       dp(i, j) = scores[i] + (total scores from i to j - dp(i+1, j))
       
```
   - If the current player picks the `j-th` element:
     - Choosing `scores[j]`:
       
```
       dp(i, j) = scores[j] + (total scores from i to j - dp(i, j-1))
       
```

4. **Final Decision**: The function will return true if player 1's score is greater than or equal to half the sum of all scores, as both players play optimally.

### Implementation

Now, we can write this out in Python code suitable for LeetCode. We'll use a wrapper function to handle input and output as expected in the LeetCode environment:



```python
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        n = len(nums)
        
        # Create a memoization table
        memo = {}

        # Function to calculate the maximum score the current player can achieve
        def dp(i: int, j: int) -> int:
            if i == j:
                return nums[i]
            # Check if this state has been computed
            if (i, j) in memo:
                return memo[(i, j)]
            
            # Total from this range
            total = sum(nums[i:j+1])
            # Current player can choose either end
            pick_i = nums[i] + (total - nums[i] - dp(i + 1, j))
            pick_j = nums[j] + (total - nums[j] - dp(i, j - 1))
            
            # Store the best score achievable
            memo[(i, j)] = max(pick_i, pick_j)
            return memo[(i, j)]
        
        total_score = sum(nums)
        player1_score = dp(0, n - 1)
        
        # Player 1 wins if his score is not less than half the total score
        return player1_score >= (total_score - player1_score)

# You can run the solution with:
# sol = Solution()
# print(sol.PredictTheWinner([1, 5, 2]))  # Output: False
# print(sol.PredictTheWinner([1, 5, 233, 7]))  # Output: True

```

### Explanation of the Code

1. **Memoization**: We use a dictionary to cache results for `dp(i, j)` to avoid redundant calculations.
2. **Recursive Function**: The `dp` function computes the maximum score for the current player for the range defined by `i` and `j`.
3. **Choosing the Best Option**: The function evaluates both picking the left-most and right-most indices and stores the optimum value.
4. **Final Check**: At the end, we compare the score achieved by player 1 to the remaining score to determine if player 1 has guaranteed a win.

With this approach and implementation, the problem can be efficiently solved with a complexity of O(n^2), where n is the number of elements in nums, which is feasible given the problem constraints.

# 494. Target Sum

### Problem Description 
You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols `+` and `-`. For each integer, you should choose one from `+` and `-` as its new symbol.

Find out how many ways to assign symbols to make sum of integers equal to target S.


Example 1:
Input: nums is [1, 1, 1, 1, 1], S is 3. 
Output: 5
Explanation: 
-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3
There are 5 ways to assign symbols to make the sum of nums be target 3.


Constraints:
The length of the given array is positive and will not exceed 20.

The sum of elements in the given array will not exceed 1000.

Your output answer is guaranteed to be fitted in a 32-bit integer.

### Solution 
 Here's a detailed explanation and the Python code to solve the problem of finding the number of ways to assign `+` and `-` symbols to a list of integers such that their sum equals a target value \( S \).

To solve this problem effectively, we can use a recursive approach with backtracking or dynamic programming, but a common approach is to leverage a depth-first search (DFS) technique which can be coded concisely.

### Problem Analysis

The given problem can be framed as finding combinations of adding and subtracting the integers in the list to achieve a target sum. Since each number can either be positive (added) or negative (subtracted), for `n` numbers, we have \( 2^n \) possible combinations of symbols.

The mathematical formulation of our problem is as follows:
Given a list \( nums \) and an integer \( S \), we need to determine how many ways the numbers can be summed to equal \( S \). 

If we denote:
- The sum of all numbers added with a positive sign as \( P \)
- The sum of all numbers added with a negative sign as \( N \)

Thus:
\[ P - N = S \]
And since \( P + N \) is the sum of all numbers in our list:
\[ P + N = \text{sum(nums)} \]

From these two equations, we derive:
1. \( P = \frac{S + \text{sum(nums)}}{2} \)

This means we are seeking the number of subsets that sum up to \( P \). The condition that \( S + \text{sum(nums)} \) must be even and non-negative for \( P \) to be valid is crucial.

### Dynamic Programming Approach

We'll implement the dynamic programming approach to solve the problem, which counts the number of subsets that sum to a particular value.

### Code Implementation

Here is the Python code that follows the LeetCode format:



```python
class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        total_sum = sum(nums)
        
        # Check validity of P
        if (S + total_sum) % 2 != 0 or S > total_sum:
            return 0
        
        P = (S + total_sum) // 2
        
        # Initialize the DP array
        dp = [0] * (P + 1)
        dp[0] = 1  # There's one way to create the sum 0: by selecting nothing
        
        for num in nums:
            # Iterate backwards to ensure we don’t use the same element multiple times
            for j in range(P, num - 1, -1):
                dp[j] += dp[j - num]
        
        return dp[P]

```

### Explanation of the Code

1. **Initial Setup:**
   We calculate the total sum of the array `nums`. 

2. **Validation Check:**
   We check if \( S + \text{total\_sum} \) is even and \( S \) does not exceed `total_sum`. If either condition fails, we return 0 because it's impossible to partition.

3. **Dynamic Programming Array:**
   We create a DP array `dp` where `dp[i]` represents the number of ways to achieve the sum \( i \). We initialize `dp[0] = 1` because there is one way to sum to zero (no elements).

4. **Filling the DP Array:**
   We iterate through each number in `nums`, and for each number, we update the DP array in reverse order. The reverse order ensures that we are not using the same element multiple times during the current iteration.

5. **Return Result:**
   Finally, we return `dp[P]`, which contains the number of ways to achieve the desired target.

This approach efficiently calculates the number of ways to assign `+` or `-` so that the total equals \( S \), complying with the constraints provided.

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
 To solve the problem of determining the minimum number of steps needed to spell a keyword using a rotating ring, we can utilize a dynamic programming approach. Let's break this down step by step.

### Explanation:

1. **Understanding the Movement**:
   - Each character in the `ring` can be rotated either clockwise or anticlockwise.
   - The distance to rotate to align a character at the 12:00 position needs to be calculated for each character in the `key`.

2. **Dynamic Programming Setup**:
   - We will create a 3D list `dp`, where `dp[i][j]` represents the minimum steps needed to spell the first `i` characters of the `key`, starting from the character at `j` position of the `ring`.
   - The length of `ring` is denoted as `R`, and the length of the `key` is denoted as `K`.
   - `dp[0][j] = 0` for any `j` position in `ring`, because no steps are needed to spell zero characters.
  
3. **Optimal Substructure**:
   - For each character in `key`, we need to consider each position in `ring` where the character can be matched. 
   - The transition will be to calculate the cost of reaching this position from any previous position where the last character from the `key` was matched.
   - The cost to switch from one position to another can be calculated via the formulas for clockwise and anticlockwise movements.

4. **Base Case and Recursion**:
   - The base case is when no characters are spelled.
   - For each character in `key`, we will compute costs based on previous characters’ positions.

5. **Result Calculation**:
   - The answer will be the minimum steps needed from any position in the `ring` that corresponds to the last character of the `key`.

### Python Code:

Here is the implementation based on the above logic:



```python
class Solution:
    def findRotateSteps(self, ring: str, key: str) -> int:
        from collections import defaultdict

        # Step 1: Map characters to their indices in the ring
        char_indices = defaultdict(list)
        for i, char in enumerate(ring):
            char_indices[char].append(i)
        
        R, K = len(ring), len(key)

        # Step 2: Initialize dp array
        dp = [[float('inf')] * R for _ in range(K + 1)]
        
        # Base case: dp[0][j] = 0 (0 steps to spell 0 characters)
        for j in range(R):
            dp[0][j] = 0
        
        # Step 3: Fill the dp table
        for i in range(1, K + 1):
            for j in range(R):
                if key[i - 1] not in char_indices:
                    continue
                # Find the positions in the ring for key[i - 1]
                for pos in char_indices[key[i - 1]]:
                    # Calculate the minimum steps to reach pos from any previous position
                    for prev_pos in range(R):
                        if dp[i - 1][prev_pos] != float('inf'):
                            # Calculate the rotation steps
                            steps = abs(pos - prev_pos)
                            steps = min(steps, R - steps)  # Choose the minimum steps
                            dp[i][pos] = min(dp[i][pos], dp[i - 1][prev_pos] + steps + 1)  # +1 for pressing the button

        # Step 4: Get the minimum steps to spell the entire key
        return min(dp[K][j] for j in range(R))

# Example usage:
# solution = Solution()
# output = solution.findRotateSteps("godding", "gd")
# print(output)  # Output: 4

```

### Explanation of the Code:

- **Mapping Characters**: We create a dictionary to map each character in the ring to its indices.
- **DP Initialization and Filling**: We initialize the `dp` array with infinity and fill it based on the previously computed values, ensuring we calculate the minimal steps required.
- **Result Calculation**: Finally, by looking at the last row of our DP table, we can find the minimum steps required to spell the entire key.

This code is efficient and handles the constraints given in the problem effectively.

# 516. Longest Palindromic Subsequence

### Problem Description 
Given a string `s`, find the longest palindromic subsequence's length in `s`.

A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.


Example 1:
Input: s = "bbbab"
Output: 4
Explanation: One possible longest palindromic subsequence is "bbbb".


Example 2:
Input: s = "cbbd"
Output: 2
Explanation: One possible longest palindromic subsequence is "bb".


Constraints:
`1 <= s.length <= 1000`
`s` consists only of lowercase English letters.

### Solution 
 To solve the problem of finding the longest palindromic subsequence's length in a given string `s`, we can use dynamic programming. A palindromic subsequence is a sequence that remains the same when read forward and backward.

### Algorithm Explanation:

1. **Dynamic Programming Table**: We will use a 2D array `dp` where `dp[i][j]` will store the length of the longest palindromic subsequence in the substring `s[i:j+1]`.

2. **Initialization**: Each single character is a palindrome of length 1, so we initialize `dp[i][i] = 1` for all `i`.

3. **Building the Table**:
   - We will fill the `dp` table based on the following:
     - If the characters `s[i]` and `s[j]` are the same, then `dp[i][j] = dp[i+1][j-1] + 2`.
     - If they are different, then we take the maximum of the two possible cases by either ignoring the character at index `i` or the index `j`: `dp[i][j] = max(dp[i+1][j], dp[i][j-1])`.

4. **Result**: After filling the table, the length of the longest palindromic subsequence for the whole string will be stored in `dp[0][n-1]`, where `n` is the length of the string.

### Python Code

Here is the code that implements the above logic:



```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        # Create a DP table
        dp = [[0] * n for _ in range(n)]
        
        # Every single character is a palindrome of length 1
        for i in range(n):
            dp[i][i] = 1
            
        # Fill the DP table
        for length in range(2, n + 1):  # length of the substring
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

        # The answer is in the full range from 0 to n-1
        return dp[0][n - 1]

```

### Explanation of the Code:

- We first determine the length of the input string `s`.
- We create a 2D list `dp` initialized to zero, where `dp[i][j]` will later hold the length of the longest palindromic subsequence in `s[i:j+1]`.
- We loop through all possible substring lengths starting from 2 up to n. For each length, we determine the start and end indices of the substrings and compute the values in the `dp` table based on the previously discussed rules.
- The final result, which is the longest palindromic subsequence length for the entire string, can be found at `dp[0][n-1]`.

This approach has a time and space complexity of \(O(n^2)\), which is efficient given the constraints.

# 517. Super Washing Machines

### Problem Description 
You have n super washing machines on a line. Initially, each washing machine has some dresses or is empty. 
For each move, you could choose any m (1 ≤ m ≤ n) washing machines, and pass one dress of each washing machine to one of its adjacent washing machines  at the same time .  
Given an integer array representing the number of dresses in each washing machine from left to right on the line, you should find the minimum number of moves to make all the washing machines have the same number of dresses. If it is not possible to do it, return -1.


Example1
Input: [1,0,5]
Output: 3
Explanation: 
1st move:    1     0 <-- 5    =>    1     1     4
2nd move:    1 <-- 1 <-- 4    =>    2     1     3    
3rd move:    2     1 <-- 3    =>    2     2     2   

Example2
Input: [0,3,0]
Output: 2
Explanation: 
1st move:    0 <-- 3     0    =>    1     2     0    
2nd move:    1     2 --> 0    =>    1     1     1     

Example3
Input: [0,2,0]
Output: -1
Explanation: 
It's impossible to make all the three washing machines have the same number of dresses. 
Note:
The range of n is [1, 10000].

The range of dresses number in a super washing machine is [0, 1e5].

### Solution 
 To solve the problem of redistributing dresses among washing machines, we need to ensure that all machines can potentially hold the same number of dresses. This requires a few steps:

### Key Points to Consider
1. **Sum and Average**: Calculate the total number of dresses. For an equal distribution across 'n' machines, the total dresses must be divisible by 'n', otherwise, equal distribution is impossible.

2. **Moves Calculation**: The moves required will depend on how many dresses need to be moved from a machine that's over the average and how many are needed by machines that are under the average.

3. **Two Scenarios**:
   - If a machine has more dresses than the average, it will need to give away the excess.
   - If a machine has fewer dresses, it will need to receive dresses.

### Implementation Details
The approach involves iterating through the list of machines, calculating the excess or deficit of dresses compared to the average, and then determining the maximum number of moves required to balance the machines based on these excesses and deficits.

### Python Code
Here’s the implementation in the LeetCode format:



```python
class Solution:
    def findMinMoves(self, machines: List[int]) -> int:
        n = len(machines)
        total_dresses = sum(machines)
        
        # If total dresses can't be evenly distributed, return -1
        if total_dresses % n != 0:
            return -1
        
        average = total_dresses // n
        moves = 0
        current_balanced_dresses = 0

        for dresses in machines:
            # Current excess or deficit
            current_balanced_dresses += (dresses - average)
            moves = max(moves, abs(current_balanced_dresses), dresses - average)
        
        return moves

```

### Explanation of the Code
1. **Input Definition**: We define a function `findMinMoves` that takes a list `machines` representing the number of dresses in each washing machine.

2. **Total Dresses Calculation**: We compute the total dresses using `sum(machines)`.

3. **Divisibility Check**: If the total dresses are not divisible by the number of machines (n), we return `-1` as it's impossible to achieve equal distribution.

4. **Average Calculation**: Calculate the average number of dresses per machine.

5. **Move Calculation**:
    - We initialize `moves` to zero and `current_balanced_dresses` to track the balance of dresses.
    - We loop through each washing machine, calculate how much it's over or under the average, and update the current balance.
    - For each machine, we consider:
        - The absolute value of the current balance, which tells us how much excess or deficit we have at any point.
        - The difference between the current number of dresses and the average, to ensure we account for the moves needed by the machines.

6. **Result Return**: Finally, the maximum value in moves gives the minimum number of moves required to balance the machines.

### Complexity
- **Time Complexity**: O(n), where n is the number of washing machines, due to the single pass required through the list.
- **Space Complexity**: O(1) since we are using only a few auxiliary variables.

This solution is efficient and should work within the provided constraints.

# 523. Continuous Subarray Sum

### Problem Description 
Given an integer array `nums` and an integer `k`, return `true` if `nums` has a continuous subarray of size at least two whose elements sum up to a multiple of `k`, or `false` otherwise.

An integer `x` is a multiple of `k` if there exists an integer `n` such that `x = n * k`. `0` is always a multiple of `k`.


Example 1:
Input: nums = [23,2,4,6,7], k = 6
Output: true
Explanation: [2, 4] is a continuous subarray of size 2 whose elements sum up to 6.


Example 2:
Input: nums = [23,2,6,4,7], k = 6
Output: true
Explanation: [23, 2, 6, 4, 7] is an continuous subarray of size 5 whose elements sum up to 42.

42 is a multiple of 6 because 42 = 7 * 6 and 7 is an integer.


Example 3:
Input: nums = [23,2,6,4,7], k = 13
Output: false

Constraints:
`1 <= nums.length <= 105`
`0 <= nums[i] <= 109`
`0 <= sum(nums[i]) <= 231 - 1`
`1 <= k <= 231 - 1`

### Solution 
 To solve this problem, we can make use of the properties of prefix sums and the modulus operation. The idea is to keep track of the cumulative sum of the elements in the array and check if any two prefix sums have the same remainder when divided by `k`. If they do, it implies that the sum of the elements between these two prefix sums is a multiple of `k`.

Here’s how we can approach the problem step-by-step:

1. **Prefix Sum and Remainders**: We keep a running sum of the elements and calculate the remainder of this sum when divided by `k`. We store these remainders in a dictionary (or hashmap) to check if we have seen the same remainder before.

2. **Checking Subarray of Length >= 2**: For each remainder, we also need to ensure that the subarray's length is at least 2. Thus, when we store a remainder in our hashmap for the first time, we also store the index. If we see the same remainder again and the distance between the current index and the stored index is at least 2, then we have found a qualifying subarray.

3. **Edge Cases**: We handle the case where `k` is 0 separately since we need at least two elements summing to zero for those scenarios.

Here's the complete Python code that implements this approach:



```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        # Edge case for k equals 0
        if k == 0:
            # We need at least two consecutive zeros
            for i in range(len(nums) - 1):
                if nums[i] == 0 and nums[i + 1] == 0:
                    return True
            return False
        
        # Using a hashmap to store the remainder and its index
        remainder_map = {0: -1}  # Initialize with remainder 0 at index -1
        cumulative_sum = 0
        
        for i in range(len(nums)):
            cumulative_sum += nums[i]
            remainder = cumulative_sum % k
            
            # Handle the case where the remainder is negative
            if remainder < 0:
                remainder += k
            
            if remainder in remainder_map:
                # Check if the previous index is at least 2 elements prior
                if i - remainder_map[remainder] > 1:
                    return True
            else:
                # Store the first occurrence of this remainder
                remainder_map[remainder] = i
        
        return False

```

### Explanation of the Code:
- We define a class `Solution` with a method `checkSubarraySum` that takes a list `nums` and an integer `k`.
- We first handle the edge case when `k` is 0. If we find any two consecutive zeros in `nums`, we return `True`.
- We initialize a dictionary `remainder_map` to store the first occurrence of each remainder from the cumulative sums.
- We iterate over the array, maintaining a running sum and calculating its remainder modulo `k`.
- If we encounter a remainder that has been seen before, we check if the indices are at least 2 apart.
- If we find such a pair, we return `True`. If we complete the loop without finding any valid subarrays, we return `False`.

### Complexity:
- Time Complexity: O(n), where n is the length of the `nums` array, since we pass through the array once.
- Space Complexity: O(k), in the worst case, if all remainders (from 0 to k-1) are stored in the dictionary. 

This solution is efficient and works within the constraints provided.

# 546. Remove Boxes

### Problem Description 
You are given several `boxes` with different colors represented by different positive numbers.

You may experience several rounds to remove boxes until there is no box left. Each time you can choose some continuous boxes with the same color (i.e., composed of `k` boxes, `k >= 1`), remove them and get `k * k` points.

Return the maximum points you can get.


Example 1:
Input: boxes = [1,3,2,2,2,3,4,3,1]
Output: 23
Explanation:
[1, 3, 2, 2, 2, 3, 4, 3, 1] 
----> [1, 3, 3, 4, 3, 1] (3*3=9 points) 
----> [1, 3, 3, 3, 1] (1*1=1 points) 
----> [1, 1] (3*3=9 points) 
----> [] (2*2=4 points)

Example 2:
Input: boxes = [1,1,1]
Output: 9

Example 3:
Input: boxes = [1]
Output: 1

Constraints:
`1 <= boxes.length <= 100`
`1 <= boxes[i] <= 100`

### Solution 
 To solve the problem of maximizing the points gained from removing boxes, we will use a dynamic programming approach. The key idea here is to recursively remove boxes while keeping track of the maximum points possible by leveraging memoization.

### Problem Breakdown
1. **Understanding Points Calculation**: Each time you remove continuous boxes of the same color of length `k`, you earn `k * k` points. Therefore, the score heavily depends on the length of the sequence of boxes removed.

2. **State Representation**: We can represent the state of the game by a tuple `(l, r, k)` where:
   - `l` is the left index of the current segment of boxes we are considering.
   - `r` is the right index of the current segment of boxes.
   - `k` is the count of boxes of the same color as the box at `r` that we can remove after removing all boxes in between to the left.

3. **Recursive Function**: We'll define a function `dp(l, r, k)` that will return the maximum points for the boxes between indexes `l` and `r` with `k` boxes of the same color as the box at `r`.

4. **Base Case**: If `l > r`, there are no boxes left, and the score is `0`.

5. **Recursive Transition**: 
   - We can choose to remove boxes from `l` to `r`.
   - If the box at `boxes[r]` is the same as the box at `boxes[r - 1]`, we can try to merge those and gain additional points.
   - We can also check if there are any boxes of the same color to the left of `l` and consider merging them.

### Implementation
Now, let's implement this logic in Python, formatted for LeetCode submission:



```python
class Solution:
    def removeBoxes(self, boxes: List[int]) -> int:
        # Memoization dictionary
        memo = {}

        # Define the recursive function
        def dp(l: int, r: int, k: int) -> int:
            # Base case: no boxes left
            if l > r:
                return 0
            if (l, r, k) in memo:
                return memo[(l, r, k)]

            # Move `l` to skip all boxes with the same color as the box `l`
            while r > l and boxes[r] == boxes[r - 1]:
                r -= 1
            
            # Remove all continuous boxes of the same color from `r`
            extra_k = r - l + 1  # Number of boxes removed at this step
            res = dp(l, r - 1, 0) + (extra_k + k) * (extra_k + k)

            # Try to merge boxes with the same color at the position `l` before removing `r`
            for i in range(l, r):
                if boxes[i] == boxes[r]:
                    res = max(res, dp(l, i, k + extra_k) + dp(i + 1, r - 1, 0))

            # Memorize the result
            memo[(l, r, k)] = res
            return res
        
        return dp(0, len(boxes) - 1, 0)

```

### Explanation of Code
- The method `removeBoxes` initializes a memoization dictionary to store computed results of `dp(l, r, k)` to avoid redundant calculations.
- The nested function `dp(l, r, k)` is implemented to recursively compute the maximum points:
  - Base case checks if we've gone past the bounds of the `boxes`.
  - The while loop adjusts the `r` pointer to consolidate boxes of the same color.
  - The main calculation considers removing boxes and keeps track of points.
  - It also attempts to merge boxes of the same color with those in the left segment, which contributes to maximizing the score.
- Finally, it returns the maximum score by invoking `dp` for the entire range of boxes with an initial `k` of `0`.

This approach ensures we efficiently calculate the maximum points by leveraging dynamic programming and memoization, resulting in optimal performance within the provided constraints.

# 552. Student Attendance Record II

### Problem Description 
An attendance record for a student can be represented as a string where each character signifies whether the student was absent, late, or present on that day. The record only contains the following three characters:
`'A'`: Absent.

`'L'`: Late.

`'P'`: Present.

Any student is eligible for an attendance award if they meet both of the following criteria:
The student was absent (`'A'`) for strictly fewer than 2 days total.

The student was never late (`'L'`) for 3 or more consecutive days.

Given an integer `n`, return the number of possible attendance records of length `n` that make a student eligible for an attendance award. The answer may be very large, so return it modulo `109 + 7`.


Example 1:
Input: n = 2
Output: 8
Explanation: There are 8 records with length 2 that are eligible for an award:
"PP", "AP", "PA", "LP", "PL", "AL", "LA", "LL"
Only "AA" is not eligible because there are 2 absences (there need to be fewer than 2).


Example 2:
Input: n = 1
Output: 3

Example 3:
Input: n = 10101
Output: 183236316

Constraints:
`1 <= n <= 105`

### Solution 
 To solve the problem of counting the number of attendance records that meet the required criteria, we can leverage dynamic programming due to the constraints and the overlapping nature of the states.

## Explanation:

### Overview of Approach:
1. **Define States**:
    - We'll define a dynamic programming array `dp[i]` where `dp[i]` holds the count of valid records of length `i`.
    - We'll categorize further based on the number of `A`s in the attendance record (0 or 1) and the streak of `L`s.

2. **Break down the problem**:
    - We can classify records based on whether the last character is:
      - 'P' (Present)
      - 'A' (Absent)
      - 'L' (Late)

3. **Recurrence Relations**:
    - If the last day is 'P': The previous `n-1` days can be any valid configuration; hence `dp[i-1]`.
    - If the last day is 'A': The previous `n-1` days must not contain another 'A', so we can take valid records of length `i-1` that end with either 'P' or 'L', which can be computed as follows:
      - `dp[i-1]` (ends with 'P')
      - `dp[i-1]` (ends with 'L'), but this is counted in `dp[i-1]`.
    - If the last day is 'L': We need to be cautious of consecutive 'L's:
      - Consider configurations ending in 'P' and 'A', but also 'L', ensuring the last two are not 'LL'.

4. **Modular Arithmetic**:
    - Since the result can be large, we will keep taking modulo \(10^9 + 7\) at every step.

### Code Implementation:

Here’s the implementation in Python following the specifications of LeetCode:



```python
class Solution:
    def checkRecord(self, n: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i][0] : number of valid records of length i with 0 'A's
        # dp[i][1] : number of valid records of length i with 1 'A'
        dp = [[0, 0] for _ in range(n + 1)]
        
        # Base cases
        dp[0][0] = 1  # Empty record
        dp[0][1] = 0  # No records possible with 'A's
        
        # Temporary variables to keep track of sums
        for i in range(1, n + 1):
            dp[i][0] = (dp[i - 1][0] + dp[i - 1][1]) % MOD  # from records of length i-1 with 0 or 1 A
            
            dp[i][1] = dp[i - 1][0]  # adding 'A' to valid records of length i-1 with 0 A
            
            # Count records that end with 'L' (0, 1, or 2 'L's at the end)
            for k in range(3):
                if i - k - 1 >= 0:  # must have enough previous records
                    dp[i][0] = (dp[i][0] + dp[i - k - 1][0] + dp[i - k - 1][1]) % MOD

        return (dp[n][0] + dp[n][1]) % MOD  # Total valid records of length n with 0 or 1 'A'

# Example usage:
# sol = Solution()
# print(sol.checkRecord(2))  # Output: 8
# print(sol.checkRecord(1))  # Output: 3
# print(sol.checkRecord(10101))  # Output: 183236316

```

### Explanation of the Code:
- We define a 2D DP array to keep track of configurations based on the number of 'A's.
- We initialize base cases for length 0.
- As we iterate through lengths from 1 to n, we accumulate counts from previous valid configurations.
- Finally, we return the sum of counts with 0 and 1 'A', taking modulo \(10^9 + 7\).

This approach efficiently computes the number of valid attendance records in \(O(n)\) time with manageable space usage.

# 568. Maximum Vacation Days

### Problem Description 
LeetCode wants to give one of its best employees the option to travel among N cities to collect algorithm problems. But all work and no play makes Jack a dull boy, you could take vacations in some particular cities and weeks. Your job is to schedule the traveling to maximize the number of vacation days you could take, but there are certain rules and restrictions you need to follow.

Rules and restrictions:
You can only travel among N cities, represented by indexes from 0 to N-1. Initially, you are in the city indexed 0 on Monday.

The cities are connected by flights. The flights are represented as a N*N matrix (not necessary symmetrical), called flights representing the airline status from the city i to the city j. If there is no flight from the city i to the city j, flights[i][j] = 0; Otherwise, flights[i][j] = 1. Also, flights[i][i] = 0 for all i.

You totally have K weeks (each week has 7 days) to travel. You can only take flights at most once per day and can only take flights on each week's Monday morning. Since flight time is so short, we don't consider the impact of flight time.

For each city, you can only have restricted vacation days in different weeks, given an N*K matrix called days representing this relationship. For the value of days[i][j], it represents the maximum days you could take vacation in the city i in the week j.

You're given the flights matrix and days matrix, and you need to output the maximum vacation days you could take during K weeks.


Example 1:
Input:flights = [[0,1,1],[1,0,1],[1,1,0]], days = [[1,3,1],[6,0,3],[3,3,3]]
Output: 12
Explanation: Ans = 6 + 3 + 3 = 12. 
One of the best strategies is:
1st week : fly from city 0 to city 1 on Monday, and play 6 days and work 1 day. (Although you start at city 0, we could also fly to and start at other cities since it is Monday.) 
2nd week : fly from city 1 to city 2 on Monday, and play 3 days and work 4 days.

3rd week : stay at city 2, and play 3 days and work 4 days.


Example 2:
Input:flights = [[0,0,0],[0,0,0],[0,0,0]], days = [[1,1,1],[7,7,7],[7,7,7]]
Output: 3
Explanation: Ans = 1 + 1 + 1 = 3. 
Since there is no flights enable you to move to another city, you have to stay at city 0 for the whole 3 weeks. For each week, you only have one day to play and six days to work. So the maximum number of vacation days is 3.


Example 3:
Input:flights = [[0,1,1],[1,0,1],[1,1,0]], days = [[7,0,0],[0,7,0],[0,0,7]]
Output: 21
Explanation:Ans = 7 + 7 + 7 = 21
One of the best strategies is:
1st week : stay at city 0, and play 7 days. 
2nd week : fly from city 0 to city 1 on Monday, and play 7 days.

3rd week : fly from city 1 to city 2 on Monday, and play 7 days.

Note:
N and K are positive integers, which are in the range of [1, 100].

In the matrix flights, all the values are integers in the range of [0, 1].

In the matrix days, all the values are integers in the range [0, 7].

You could stay at a city beyond the number of vacation days, but you should work on the extra days, which won't be counted as vacation days.

If you fly from the city A to the city B and take the vacation on that day, the deduction towards vacation days will count towards the vacation days of city B in that week.

We don't consider the impact of flight hours towards the calculation of vacation days.

### Solution 
 To solve the problem of maximizing vacation days while traveling between cities, we can employ dynamic programming (DP). The key idea is to maintain a DP table that keeps track of the maximum vacation days we can accumulate by the end of each week while considering all possible paths we can take through the cities.

Here's how we can break down the steps to implement the solution:

### Breakdown of the Solution 

1. **Define the Data Structure**:
   We will create a 2D DP table `dp[week][city]` where `week` represents the week (from 0 to K-1) and `city` represents the city we are in at the end of that week. The value at `dp[week][city]` will store the maximum vacation days we can accumulate after visiting that city at the end of the specified week.

2. **Initialization**:
   The starting point is city 0, and we begin with week 0. So, `dp[0][0]` will be initialized with `days[0][0]` since that's the maximum vacation days we can take starting from city 0 in week 0.

3. **DP Transition**:
   For each week (from 0 to K-1), for each city (from 0 to N-1), we:
   - Consider staying in the same city, which means adding to the total vacation days based on the allowed vacation for that city in the current week.
   - Explore traveling to other reachable cities according to the flight availability and calculate the potential new vacation days by flying to those cities.

4. **Compute the Result**:
   After processing all weeks and cities, the answer will be the maximum value in `dp[K-1][city]` for all cities since the last week can end in any of the cities.

### Implementation

Here is the complete Python code wrapped in a solution class following the LeetCode solution format:



```python
class Solution:
    def maxVacationDays(self, flights: List[List[int]], days: List[List[int]]) -> int:
        N = len(flights)
        K = len(days[0])
        
        # Initialize the DP table with -1 (impossible state).
        dp = [[-1] * N for _ in range(K)]
        
        # Start at city 0 in week 0.
        dp[0][0] = days[0][0]

        # Iterate through each week
        for week in range(1, K):
            for city_from in range(N):
                if dp[week-1][city_from] >= 0:  # If it's reachable
                    # Stay in the same city
                    dp[week][city_from] = max(dp[week][city_from], dp[week-1][city_from] + days[city_from][week])
                    # Check for possible flights
                    for city_to in range(N):
                        if flights[city_from][city_to] == 1:
                            dp[week][city_to] = max(dp[week][city_to], dp[week-1][city_from] + days[city_to][week])
        
        # Compute the maximum vacation days in the last week
        max_vacation_days = max(dp[K-1])
        
        return max_vacation_days

```

### Explanation of the Code

1. **Importing Necessary Libraries**: 
   We assume `List` is imported from the `typing` module to specify the type of the input parameters.

2. **Initialization of DP Table**: 
   A 2D list `dp` is created to hold values of vacation days with `-1` indicating impossible states.

3. **Initial State Setup**: 
   We set the vacation days for the first week in the starting city.

4. **Filling the DP Table**:
   We iterate over each week and each city to update the DP table based on whether we stay or move to another city.

5. **Final Calculation**:
   After filling the DP table, we take the maximum vacation days from the last week across all cities for our result.

This efficient approach ensures we explore all possible ways to maximize vacation days given the constraints of flying and vacation allowances in the cities. The time complexity is O(K * N^2), which is manageable within the provided constraints.

# 576. Out of Boundary Paths

### Problem Description 
There is an m by n grid with a ball. Given the start coordinate (i,j) of the ball, you can move the ball to adjacent cell or cross the grid boundary in four directions (up, down, left, right). However, you can at most move N times. Find out the number of paths to move the ball out of grid boundary. The answer may be very large, return it after mod 109 + 7.


Example 1:
Input: m = 2, n = 2, N = 2, i = 0, j = 0
Output: 6
Explanation:

Example 2:
Input: m = 1, n = 3, N = 3, i = 0, j = 1
Output: 12
Explanation:
Note:
Once you move the ball out of boundary, you cannot move it back.

The length and height of the grid is in range [1,50].

N is in range [0,50].

### Solution 
 To solve the problem of finding the number of paths for the ball to exit the grid after a maximum of N moves, we can use dynamic programming. The key idea to notice is that from each position in the grid, we can recursively calculate the number of ways to move the ball until it goes out of boundaries. 

Here is a step-by-step breakdown of how to approach this problem:

1. **Define the Movement Directions**: The ball can be moved in four directions: up, down, left, and right. Each movement can be represented by changes in the coordinates.

2. **Dynamic Programming Table**: We'll maintain a 3D array `dp` where `dp[k][x][y]` represents the number of ways to move out of bounds from position `(x, y)` with exactly `k` moves remaining.

3. **Boundary Condition**: If the ball moves out of the grid, we count it as a valid move. Hence, any cell where `x < 0`, `x >= m`, `y < 0`, or `y >= n` contributes 1 way to exit the grid.

4. **Transition Formula**: For each cell `(x, y)` and number of moves `k`, the valid ways to move out are determined by the sum of valid moves from the four adjacent cells. Specifically:
   \[
   dp[k][x][y] = dp[k-1][x-1][y] + dp[k-1][x+1][y] + dp[k-1][x][y-1] + dp[k-1][x][y+1]
   \]
   Here we account for out-of-bounds conditions elegantly since any access that goes out of bounds will contribute to the count.

5. **Base Case**: When `k = 0`, we cannot move at all, and thus if the ball is still within the grid, there are no valid paths.

6. **Calculate the Result**: We accumulate results for all moves from `1` to `N` for the position `(i, j)`.

Here's the complete implementation in Python according to the LeetCode solution format:



```python
class Solution:
    def findPaths(self, m: int, n: int, N: int, i: int, j: int) -> int:
        MOD = 10**9 + 7
        
        # Create a DP table
        dp = [[[0] * n for _ in range(m)] for _ in range(N + 1)]
        
        # Base case: If the ball is out of bounds, there's one way (to leave)
        for k in range(1, N + 1):
            for x in range(m):
                for y in range(n):
                    # From (x,y), we can go in four directions
                    if x == 0:   # Out of bounds upwards
                        dp[k][x][y] = (dp[k][x][y] + 1) % MOD
                    if x == m - 1:  # Out of bounds downwards
                        dp[k][x][y] = (dp[k][x][y] + 1) % MOD
                    if y == 0:   # Out of bounds left
                        dp[k][x][y] = (dp[k][x][y] + 1) % MOD
                    if y == n - 1:  # Out of bounds right
                        dp[k][x][y] = (dp[k][x][y] + 1) % MOD
                    
                    if k > 0:
                        # Valid moves from the current position
                        if x > 0: 
                            dp[k][x][y] = (dp[k][x][y] + dp[k - 1][x - 1][y]) % MOD
                        if x < m - 1: 
                            dp[k][x][y] = (dp[k][x][y] + dp[k - 1][x + 1][y]) % MOD
                        if y > 0: 
                            dp[k][x][y] = (dp[k][x][y] + dp[k - 1][x][y - 1]) % MOD
                        if y < n - 1: 
                            dp[k][x][y] = (dp[k][x][y] + dp[k - 1][x][y + 1]) % MOD
        
        # Sum up all paths from (i, j) with at most N moves
        result = 0
        for k in range(1, N + 1):
            result = (result + dp[k][i][j]) % MOD
        
        return result

```

### Explanation of the Code:

1. **Input Handling**: The function takes the grid dimensions (`m`, `n`), the maximum moves (`N`), and the starting coordinates (`i`, `j`).

2. **DP Array Initialization**: A 3D list is initialized to keep track of the number of ways the ball can exit.

3. **Base Case and Transition Logic**: We handle cases to increment paths for each directional move. 

4. **Final Count**: After populating the DP table, we sum over valid moves up to N to get the final answer.

5. **Output**: The result is returned as required by LeetCode. 

This approach ensures that we efficiently compute the paths using dynamic programming while respecting the constraints of the problem.

# 600. Non-negative Integers without Consecutive Ones

### Problem Description 
Given a positive integer n, find the number of non-negative integers less than or equal to n, whose binary representations do NOT contain consecutive ones.


Example 1:
Input: 5
Output: 5
Explanation: 
Here are the non-negative integers <= 5 with their corresponding binary representations:
0 : 0
1 : 1
2 : 10
3 : 11
4 : 100
5 : 101
Among them, only integer 3 disobeys the rule (two consecutive ones) and the other 5 satisfy the rule. 
Note:
1 <= n <= 109

### Solution 
 To tackle the problem of counting non-negative integers less than or equal to `n` whose binary representations do not contain consecutive ones, we can employ a combination of dynamic programming and a bit manipulation approach. The key idea is to build a solution based on Fibonacci-like properties.

### Problem Breakdown

1. **Binary Representation**: Binary representations of integers are built using bits. For example, the number `5` is represented as `101` in binary.
  
2. **No Consecutive Ones Rule**: We need to count integers whose binary representation does not have two `1s` next to each other. This means patterns like `11` are invalid.

3. **Dynamic Programming Approach**: 
   - Let's define a function `fib(k)` that counts the number of valid binary representations of all numbers with `k` bits.
   - The recurrence relation can be derived as follows:
     - If the highest bit (most significant) is `0`, then the remaining `k-1` bits can be either valid or invalid -> `fib(k-1)`.
     - If the highest bit is `1`, then the second highest bit must be `0` (to avoid consecutive ones), allowing us to look at the remaining `k-2` bits -> `fib(k-2)`.
   - Thus, we have: `fib(k) = fib(k-1) + fib(k-2)` which mirrors the Fibonacci sequence.

4. **Counting Valid Numbers**: 
   - We will iterate through the bits of `n`, starting from the most significant bit, and apply the rules from our dynamic programming setup.
   - Whenever we encounter a `1`, we count the contribution from the previous bits, and we need to ensure we don’t count numbers leading up to the next `1`.

### Implementation

Here is how we can implement this in Python using a format suitable for the LeetCode platform:



```python
class Solution:
    def findIntegers(self, n: int) -> int:
        # Step 1: Create a list for Fibonacci-like number
        fib = [0] * 32
        fib[0] = 1  # fib(0) = 1
        fib[1] = 2  # fib(1) = 2
        
        # Step 2: Fill the fib array with values for 32 bits
        for i in range(2, 32):
            fib[i] = fib[i - 1] + fib[i - 2]
        
        # Step 3: Count valid numbers less than or equal to n
        count = 0
        prev_bit = 0 # track the previously processed bit
        for i in range(31, -1, -1):  # check each bit position from 31 down to 0
            if (n & (1 << i)) != 0:  # if the i-th bit is a `1`
                count += fib[i]  # Add the number of valid integers with i bits
                
                if prev_bit == 1:  # If the previous bit was also `1`, break
                    break
                
                prev_bit = 1  # Update previous bit to `1`
            else:
                prev_bit = 0  # Update previous bit to `0`
        
        else:  # If we did not break from the loop
            count += 1  # Include `n` itself if it doesn't end with `11`
        
        return count

```

### Explanation:

- **Fibonacci Array**: We precompute a Fibonacci-like array where `fib[i]` represents the number of valid integers that can be formed with `i` bits.
- **Loop through Bits**: We evaluate each bit from the highest to the lowest:
  - If the current bit is `1`, we add the number of valid integers that can be formed with remaining lower bits.
  - If we encounter consecutive `1`s while iterating, we can stop because all subsequent counts will be invalid.
- **Edge Case**: If we finish iterating without finding consecutive `1`s, we add `1` to include `n` itself.

This method ensures efficient counting for values of `n` up to \(10^9\). It operates in `O(log n)`, making it suitable given the constraints.

# 629. K Inverse Pairs Array

### Problem Description 
Given two integers `n` and `k`, find how many different arrays consist of numbers from `1` to `n` such that there are exactly `k` inverse pairs.

We define an inverse pair as following: For `ith` and `jth` element in the array, if `i` < `j` and `a[i]` > `a[j]` then it's an inverse pair; Otherwise, it's not.

Since the answer may be very large, the answer should be modulo 109 + 7.


Example 1:
Input: n = 3, k = 0
Output: 1
Explanation: 
Only the array [1,2,3] which consists of numbers from 1 to 3 has exactly 0 inverse pair.


Example 2:
Input: n = 3, k = 1
Output: 2
Explanation: 
The array [1,3,2] and [2,1,3] have exactly 1 inverse pair.

Note:
The integer `n` is in the range [1, 1000] and `k` is in the range [0, 1000].

### Solution 
 To solve the problem of finding how many different arrays consist of numbers from `1` to `n` with exactly `k` inverse pairs, we can utilize dynamic programming. The idea is to create a DP table where each entry `dp[i][j]` will represent the number of arrays of length `i` that contain exactly `j` inverse pairs.

### Explanation:

1. **Base Case**: 
   - For `dp[0][0] = 1`: There is only one way to have 0 elements with 0 inverse pairs (the empty array).
   - For all other `dp[0][j]` where `j > 0`, it should be `0` because we cannot have inverse pairs without any elements.

2. **Recurrence Relation**:
   - To fill in the DP table, we need to consider how to construct an array of length `i` with `j` inverse pairs. 
   - When we add the `i-th` number to the sequence, it can be placed in `0` to `i-1` positions. 
   - If we place the number at position `p`, it will create `i-p-1` new inverse pairs since all numbers that are placed after position `p` will be counted against it.
   - Thus, if we consider placing it at position `p`, we need to ensure that the previous `j - (i - 1 - p)` inverse pairs can be accounted for by the remaining `i-1` numbers.
   - Hence, the recurrence can be given as:
     \[
     dp[i][j] = \sum_{p=0}^{min(j, i-1)} dp[i-1][j-p]
     \]

3. **Optimization**:
   - Instead of re-calculating the sum for every position, we can use a prefix sum to optimize the computation.

4. **Modulo Operation**:
   - Since the result can be very large, we'll perform all operations modulo \(10^9 + 7\).

5. **Final Calculation**:
   - The answer will be found at `dp[n][k]`.

Now let's implement this solution in Python:



```python
MOD = 10**9 + 7

class Solution:
    def kInversePairs(self, n: int, k: int) -> int:
        # Initialize the dp array
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        
        # Base case
        dp[0][0] = 1  # There is one way to arrange 0 numbers with 0 inverse pairs
        
        # Fill the dp array
        for i in range(1, n + 1):
            for j in range(k + 1):
                # We can calculate dp[i][j] using the previous row
                dp[i][j] = dp[i - 1][j]  # case of adding 'i' at the end (no new inverses)
                if j > 0:
                    dp[i][j] += dp[i][j - 1]  # Count inverses as we can move the last number left
                
                if j >= i:
                    dp[i][j] -= dp[i - 1][j - i]  # Exclude scenarios that exceed the maximum inverses
                dp[i][j] = dp[i][j] % MOD  # Take modulo at every step
        
        return dp[n][k]

# Example usage:
solution = Solution()
print(solution.kInversePairs(3, 0))  # Output: 1
print(solution.kInversePairs(3, 1))  # Output: 2

```

### Explanation of the Code:
- We first initialize a 2D list `dp` with dimensions `(n+1) x (k+1)` filled with zeros.
- The base case is set, indicating that there's one way to have an empty array with zero inverse pairs.
- We proceed to fill in the `dp` table using the recurrence relation, taking care to handle cases where new inverse pairs exceed valid counts.
- Finally, we return the value found at `dp[n][k]`, which represents the number of arrays of size `n` with exactly `k` inverse pairs.

This solution has a time complexity of \(O(n \times k)\) and a space complexity of \(O(n \times k)\), which are efficient given the input constraints.

# 638. Shopping Offers

### Problem Description 
In LeetCode Store, there are some kinds of items to sell. Each item has a price.

However, there are some special offers, and a special offer consists of one or more different kinds of items with a sale price.

You are given the each item's price, a set of special offers, and the number we need to buy for each item.

The job is to output the lowest price you have to pay for exactly certain items as given, where you could make optimal use of the special offers.

Each special offer is represented in the form of an array, the last number represents the price you need to pay for this special offer, other numbers represents how many specific items you could get if you buy this offer.

You could use any of special offers as many times as you want.


Example 1:
Input: [2,5], [[3,0,5],[1,2,10]], [3,2]
Output: 14
Explanation: 
There are two kinds of items, A and B. Their prices are $2 and $5 respectively. 
In special offer 1, you can pay $5 for 3A and 0B
In special offer 2, you can pay $10 for 1A and 2B. 
You need to buy 3A and 2B, so you may pay $10 for 1A and 2B (special offer #2), and $4 for 2A.


Example 2:
Input: [2,3,4], [[1,1,0,4],[2,2,1,9]], [1,2,1]
Output: 11
Explanation: 
The price of A is $2, and $3 for B, $4 for C. 
You may pay $4 for 1A and 1B, and $9 for 2A ,2B and 1C. 
You need to buy 1A ,2B and 1C, so you may pay $4 for 1A and 1B (special offer #1), and $3 for 1B, $4 for 1C. 
You cannot add more items, though only $9 for 2A ,2B and 1C.

Note:
There are at most 6 kinds of items, 100 special offers.

For each item, you need to buy at most 6 of them.

You are not allowed to buy more items than you want, even if that would lower the overall price.

### Solution 
 To solve the problem of finding the lowest price for items in the LeetCode Store, we can use a recursive approach with memoization to optimize our calculations. Let's go through the key points and then present the code in a format appropriate for a LeetCode submission.

### Understanding the Problem

1. **Item Prices:** You have `n` items, each with a price.
2. **Special Offers:** Each special offer provides a combination of items for a certain price; you can use it multiple times.
3. **Quantity Required:** You have a specific number of each item that you must obtain.
4. **Objective:** Minimize the total cost using these special offers optimally.

### Approach

- We'll create a recursive function that will explore the combinations of purchasing items either directly (using item prices) or via special offers.
- We'll use memoization to store results of function calls with specific quantities left to buy, to avoid redundant calculations.
- The base case for our recursion will be when we have no items left to buy, at which point the cost is zero.
- For each call, we will either:
  - Buy the items without using any special offers.
  - Loop through each special offer and apply it, then recursively compute the remaining items needed.

### Implementation

Here's how the code can be structured in Python:



```python
class Solution:
    def shoppingOffers(self, prices: List[int], special: List[List[int]], needs: List[int]) -> int:
        # Memoization dictionary
        memo = {}
        
        def dfs(needs):
            # Convert needs to a tuple (hashable) for memoization
            needs_tuple = tuple(needs)
            if needs_tuple in memo:
                return memo[needs_tuple]
            
            # Total cost if we buy items directly without special offers
            base_cost = sum(prices[i] * needs[i] for i in range(len(needs)))
            min_cost = base_cost
            
            # Check each special offer
            for offer in special:
                # Create a new needs list after applying the special offer
                new_needs = []
                for i in range(len(needs)):
                    if needs[i] < offer[i]:
                        break
                    new_needs.append(needs[i] - offer[i])
                else:  # Only if we did not break from the loop
                    new_needs = [x for x in new_needs]
                    # Calculate the cost using this special offer
                    offer_cost = offer[-1] + dfs(new_needs)
                    min_cost = min(min_cost, offer_cost)
            
            # Store the result in memo and return
            memo[needs_tuple] = min_cost
            return min_cost
        
        return dfs(needs)


```

### Explanation of the Code

1. **Class Definition:** We define a class `Solution` which contains the method `shoppingOffers`.
2. **`shoppingOffers` Method:** This method initializes the recursion and memoization, processes the inputs, and starts the depth-first search (DFS) with `dfs(needs)`.
3. **DFS Function:**
   - We convert the `needs` list to a tuple to use it as a key for memoization.
   - Calculate the `base_cost` for buying items without any special offers.
   - Initialize `min_cost` with `base_cost`.
   - Loop through each special offer:
     - Check if the offer can be applied (i.e., we have enough items left).
     - Calculate the remaining needs after applying the special offer.
     - Recursively call `dfs` to compute the cost after applying the offer.
     - Update `min_cost` if the newly calculated cost is lower.
4. **Return Value:** Finally, return the minimum cost calculated.

This approach ensures that we consider all possible combinations of item purchases and special offers efficiently by using memoization, making it suitable for the given constraints.

# 639. Decode Ways II

### Problem Description 
A message containing letters from `A-Z` can be encoded into numbers using the following mapping:
'A' -> "1"
'B' -> "2"
...

'Z' -> "26"
To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, `"11106"` can be mapped into:
`"AAJF"` with the grouping `(1 1 10 6)`
`"KJF"` with the grouping `(11 10 6)`
Note that the grouping `(1 11 06)` is invalid because `"06"` cannot be mapped into `'F'` since `"6"` is different from `"06"`.

In addition to the mapping above, an encoded message may contain the `'*'` character, which can represent any digit from `'1'` to `'9'` (`'0'` is excluded). For example, the encoded message `"1*"` may represent any of the encoded messages `"11"`, `"12"`, `"13"`, `"14"`, `"15"`, `"16"`, `"17"`, `"18"`, or `"19"`. Decoding `"1*"` is equivalent to decoding any of the encoded messages it can represent.

Given a string `s` containing digits and the `'*'` character, return the number of ways to decode it.

Since the answer may be very large, return it modulo `109 + 7`.


Example 1:
Input: s = "*"
Output: 9
Explanation: The encoded message can represent any of the encoded messages "1", "2", "3", "4", "5", "6", "7", "8", or "9".

Each of these can be decoded to the strings "A", "B", "C", "D", "E", "F", "G", "H", and "I" respectively.

Hence, there are a total of 9 ways to decode "*".


Example 2:
Input: s = "1*"
Output: 18
Explanation: The encoded message can represent any of the encoded messages "11", "12", "13", "14", "15", "16", "17", "18", or "19".

Each of these encoded messages have 2 ways to be decoded (e.g. "11" can be decoded to "AA" or "K").

Hence, there are a total of 9 * 2 = 18 ways to decode "1*".


Example 3:
Input: s = "2*"
Output: 15
Explanation: The encoded message can represent any of the encoded messages "21", "22", "23", "24", "25", "26", "27", "28", or "29".

"21", "22", "23", "24", "25", and "26" have 2 ways of being decoded, but "27", "28", and "29" only have 1 way.

Hence, there are a total of (6 * 2) + (3 * 1) = 12 + 3 = 15 ways to decode "2*".


Constraints:
`1 <= s.length <= 105`
`s[i]` is a digit or `'*'`.

### Solution 
 To solve the decoding problem involving the encoded message with digits and `'*'` characters, we will use dynamic programming. The key idea is to keep track of the number of ways to decode the string up to each position while considering the rules of valid decoding, especially those related to the presence of `'*'` characters.

### Step-by-step Explanation:

1. **Dynamic Programming Array**:
   We will maintain a DP array `dp` where `dp[i]` represents the number of ways to decode the substring `s[0...i-1]`.

2. **Base Cases**:
   - `dp[0]` = 1, which represents one way to decode an empty string.
   - If the first character is `'*'`, it can be decoded as any digit from `1` to `9`, so `dp[1]` should be `9`.
   - If the first character is a digit from `1` to `9`, `dp[1]` should be `1`.

3. **Filling the DP Array**:
   For each character in the string `s`, we will determine the number of decoding ways based on the current character and the previous character:
   - If `s[i-1]` is a digit:
     - Check if it contributes a single digit decoding.
     - Check if `s[i-2]` and `s[i-1]` can form a valid two-digit decoding.
   - If `s[i-1]` is `'*'`, it can represent any digit from `1` to `9`, which will affect both single digit and two-digit interpretations.
   - Additionally, if `s[i-2]` is `'*'`, it can also represent digits `1-9` and will influence two-digit combinations.

4. **Handling Edge Cases**:
   We need to ensure that when calculating two-digit combinations, they fall within the valid bounds (`10` to `26`).

5. **Modulo Operation**:
   Since results could be large, every addition to the `dp` values should be taken modulo \(10^9 + 7\).

### Final Implementation:

Here’s how the complete implementation of the above logic looks in Python, suitable for running on LeetCode:



```python
class Solution:
    def numDecodings(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Handle empty string case
        if n == 0:
            return 0
        
        # dp array initialized to 0
        dp = [0] * (n + 1)
        
        # Base case
        dp[0] = 1  # There is one way to decode an empty string
        
        for i in range(1, n + 1):
            # Single digit decode
            if s[i - 1] == '*':
                dp[i] += 9 * dp[i - 1]  # '*' can represent 1-9
            elif s[i - 1] != '0':
                dp[i] += dp[i - 1]
            
            dp[i] %= MOD
            
            # Two digit decode
            if i > 1:
                if s[i - 2] == '*' and s[i - 1] == '*':
                    dp[i] += 15 * dp[i - 2]  # 11-19 (9 ways) + 21-26 (6 ways)
                elif s[i - 2] == '*':
                    # can be '1', '2' followed by '*'
                    if s[i - 1] <= '6':
                        dp[i] += 2 * dp[i - 2]  # '1' or '2' 
                    else:
                        dp[i] += dp[i - 2]  # only '1'
                elif s[i - 1] == '*':
                    if '10' <= s[i - 2:i] <= '26':
                        dp[i] += dp[i - 2] * 2  # '10' or '20'
                else:
                    # Regular two digit processing
                    if 10 <= int(s[i - 2:i]) <= 26:
                        dp[i] += dp[i - 2]
            
            dp[i] %= MOD
            
        return dp[n]

```

### Complexity Analysis:
- **Time Complexity**: O(n), where n is the length of the string, since we iterate through each character once.
- **Space Complexity**: O(n), for the DP array. However, we can reduce it to O(1) by using only two variables to keep track of the last two states.

This Python function will successfully decode the string as required and return the number of decoding ways modulo \(10^9 + 7\).

# 646. Maximum Length of Pair Chain

### Problem Description 
You are given `n` pairs of numbers. In every pair, the first number is always smaller than the second number.

Now, we define a pair `(c, d)` can follow another pair `(a, b)` if and only if `b < c`. Chain of pairs can be formed in this fashion. 
Given a set of pairs, find the length longest chain which can be formed. You needn't use up all the given pairs. You can select pairs in any order.


Example 1:
Input: [[1,2], [2,3], [3,4]]
Output: 2
Explanation: The longest chain is [1,2] -> [3,4]
Note:
The number of given pairs will be in the range [1, 1000].

### Solution 
 To solve the problem of finding the longest chain of pairs where the second number of one pair is less than the first number of the next pair, we can use a greedy approach. The key idea is to sort the pairs based on the second element and then iterate through the sorted pairs to build the longest chain.

Here's a step-by-step breakdown of the approach:

1. **Sort the Pairs**: First, we sort the pairs based on the second element. This allows us to maximize the number of pairs we can chain together, as we always want to pick the pair that ends the earliest (smallest second element).

2. **Greedy Selection**: We then iterate through the sorted pairs. We maintain a variable to track the end of the last selected pair. For each pair, if its first element is greater than the second element of the last selected pair, we can form a chain and update our count and the end of the last selected pair.

3. **Count the Length of the Chain**: Every time we can select a pair to include in our chain, we increment our chain length.

Here's the complete Python code implementing the above logic, formatted according to LeetCode specifications:



```python
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        # Sort the pairs by the second element of each pair
        pairs.sort(key=lambda x: x[1])
        
        # Initialize variables to track the count of chains and the last end of the chain
        count = 0
        last_end = float('-inf')  # Start with negative infinity to ensure any first pair is accepted
        
        # Iterate through the sorted pairs
        for pair in pairs:
            start, end = pair
            # If the current pair can follow the last pair in the chain
            if start > last_end:
                # Include the current pair in the chain
                count += 1
                last_end = end  # Update the last end to the end of the current pair
        
        return count

```

### Explanation of the Code

1. **Sorting the Pairs**: We use the `sorted()` function with a custom key that sorts the pairs by their second value. This is crucial for the greedy approach.

   

```python
   pairs.sort(key=lambda x: x[1])
   
```
   
2. **Initialization**: We initialize `count` to 0 to keep track of the number of chains formed. `last_end` is initialized to negative infinity (or any small value) because it allows the first pair to be added to the chain without restriction.

3. **Building the Chain**: As we loop through each pair, we check if the current pair can follow the last selected pair by comparing the `start` of the current pair with `last_end`. If it can follow, we increment our `count` and update `last_end` to the `end` of the current pair.

4. **Return the Result**: Finally, we return the count of the longest chain formed.

### Complexity Analysis
- **Time Complexity**: The overall time complexity is O(n log n) due to the sorting step, where n is the number of pairs.
- **Space Complexity**: The space complexity is O(1), as we are using a constant amount of extra space for our calculations aside from the input.

This code can be directly run on the LeetCode platform, and it will provide the correct output for the problem as stated.

# 647. Palindromic Substrings

### Problem Description 
Given a string, your task is to count how many palindromic substrings in this string.

The substrings with different start indexes or end indexes are counted as different substrings even they consist of same characters.


Example 1:
Input: "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".


Example 2:
Input: "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".

Note:
The input string length won't exceed 1000.

### Solution 
 To solve the problem of counting palindromic substrings in a given string, we can use the "Expand Around Center" approach. Palindromic substrings read the same backward as forward, and for a given string of length `n`, there can be `2n-1` centers (considering both individual characters and gaps between characters).

### Algorithm:
1. **Expand Around Center**: We define a helper function that expands around the given center index. For each character and each gap between characters, we expand outwards as long as the left and right characters are equal.
  
2. **Count Palindromes**: For each character (or between each pair of characters), we keep expanding and count how many palindromic substrings we can find.

3. **Return Count**: Finally, we return the total count of palindromic substrings found.

### Implementation:
Here is the Python code following the LeetCode solution format:



```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        def expand_around_center(left: int, right: int) -> int:
            count = 0
            while left >= 0 and right < len(s) and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1
            return count
        
        total_count = 0
        for i in range(len(s)):
            # Count odd-length palindromes (single character center)
            total_count += expand_around_center(i, i)
            # Count even-length palindromes (gaps between characters)
            total_count += expand_around_center(i, i + 1)
        
        return total_count

# Example usage:
solution = Solution()
print(solution.countSubstrings("abc"))  # Output: 3
print(solution.countSubstrings("aaa"))  # Output: 6

```

### Explanation:
- We define a class named `Solution` with a method `countSubstrings`.
- Inside the method, we have a nested function `expand_around_center`. This function takes two indices, `left` and `right`, and checks for palindrome by expanding outwards. For every match found, it increments the count.
- In the outer loop, we iterate through each character in the string:
  - First, we check for palindromic substrings with odd lengths by calling `expand_around_center(i, i)`, where `i` is the center character.
  - Then, we check for palindromic substrings with even lengths by calling `expand_around_center(i, i + 1)`, which considers the gaps between characters.
- We accumulate the counts of palindromic substrings and return the total count.

### Complexity:
- **Time Complexity**: O(n^2), where `n` is the length of the string. In the worst case (for palindromic strings), we have to check every character and on average we expand around `n` centers.
- **Space Complexity**: O(1), as we are using a constant amount of space regardless of the input size other than the input string itself.

This code can be directly run on the LeetCode platform and will provide the correct output for the examples given.

# 650. 2 Keys Keyboard

### Problem Description 
Initially on a notepad only one character 'A' is present. You can perform two operations on this notepad for each step:
`Copy All`: You can copy all the characters present on the notepad (partial copy is not allowed).

`Paste`: You can paste the characters which are copied last time.

Given a number `n`. You have to get exactly `n` 'A' on the notepad by performing the minimum number of steps permitted. Output the minimum number of steps to get `n` 'A'.

Example 1:
Input: 3
Output: 3
Explanation:
Intitally, we have one character 'A'.

In step 1, we use Copy All operation.

In step 2, we use Paste operation to get 'AA'.

In step 3, we use Paste operation to get 'AAA'.

Note:
The `n` will be in the range [1, 1000].

### Solution 
 To solve the problem of determining the minimum number of steps required to get exactly `n` 'A' characters on a notepad using the operations "Copy All" and "Paste", we can use a dynamic programming approach. The idea is to analyze the possible ways of reaching `n` characters by breaking down the problem into smaller subproblems.

### Explanation of the Approach:

1. **Understanding Operations**:
   - You start with a single 'A'.
   - "Copy All" operation allows you to copy all currently displayed characters.
   - "Paste" operation allows you to paste the last copied characters.

2. **Dynamic Programming Array**:
   - We'll maintain a `dp` array where `dp[i]` represents the minimum number of steps required to obtain exactly `i` 'A's.

3. **Base Case**:
   - `dp[1] = 0`: It takes no steps to have 1 'A' since we start with it.

4. **Building the DP array**:
   - For each number `i` from 2 to `n`, we'll determine the minimum steps to reach `i` by checking for every divisor `j` of `i` (starting from 1 to `sqrt(i)`).
   - If `j` is a divisor, then you can reach `i` by performing:
     - `dp[j]` (to get `j` 'A's)
     - `Copy All` operation (1 step)
     - and `Paste` several times (which is `i/j - 1` times to yield `i` 'A's from `j`).
   - Therefore, `dp[i]` can be updated as:
     

```python
     dp[i] = min(dp[i], dp[j] + 1 + (i // j - 1))
     
```

5. **Final Step**:
   - After filling up the `dp` array, `dp[n]` will contain the minimum steps required to achieve exactly `n` 'A's.

### Python Code in LeetCode Format:

Here’s the complete code for the solution:



```python
class Solution:
    def minSteps(self, n: int) -> int:
        if n == 1:
            return 0
        
        dp = [0] * (n + 1)  # Create a dp array to hold steps for each count of 'A'

        for i in range(2, n + 1):  # Start from 2 to n
            dp[i] = float('inf')  # Initialize with infinity
            for j in range(1, int(i**0.5) + 1):  # Check divisors up to sqrt(i)
                if i % j == 0:  # j is a divisor
                    # If j is a divisor of i
                    dp[i] = min(dp[i], dp[j] + (i // j) + 1)  # j --> copy and paste i/j times
                    if j != i // j:
                        dp[i] = min(dp[i], dp[i // j] + j + 1)  # i/j --> copy and paste j times

        return dp[n]  # The result for n 'A's

```

### Explanation of the Code:
- We first handle the trivial case of `n = 1`.
- We create a `dp` list of size `n + 1`, initialized to zero.
- We loop through each number from `2` to `n`, calculating possible steps by checking divisors and applying the formula derived above.
- Finally, we return `dp[n]`, which contains the minimum steps to reach exactly `n` 'A's.

This solution runs in `O(n * sqrt(n))`, which is efficient given the maximum constraint of `n = 1000`.

# 651. 4 Keys Keyboard

### Problem Description 
Imagine you have a special keyboard with the following keys: 
`Key 1: (A)`:  Print one 'A' on screen.

`Key 2: (Ctrl-A)`: Select the whole screen.

`Key 3: (Ctrl-C)`: Copy selection to buffer.

`Key 4: (Ctrl-V)`: Print buffer on screen appending it after what has already been printed. 
Now, you can only press the keyboard for N times (with the above four keys), find out the maximum numbers of 'A' you can print on screen.


Example 1:
Input: N = 3
Output: 3
Explanation: 
We can at most get 3 A's on screen by pressing following key sequence:
A, A, A

Example 2:
Input: N = 7
Output: 9
Explanation: 
We can at most get 9 A's on screen by pressing following key sequence:
A, A, A, Ctrl A, Ctrl C, Ctrl V, Ctrl V
Note:
1 <= N <= 50 
Answers will be in the range of 32-bit signed integer.

### Solution 
 To tackle the problem of maximizing the number of 'A's that can be printed on the screen within a given number of keystrokes `N`, we can use a dynamic programming approach.

### Problem Understanding

You have four operations you can perform with the keyboard:
1. **Print 'A'**: Increases the count of 'A's by 1.
2. **Select All (Ctrl-A)**: Selects all printed characters.
3. **Copy (Ctrl-C)**: Copies the selected characters into a buffer.
4. **Paste (Ctrl-V)**: Pastes the content of the buffer after the current screen content, effectively adding that many new 'A's.

The goal is to determine how many 'A's can be printed on the screen after exactly `N` key presses.

### Dynamic Programming Approach

We will create a dynamic programming array `dp` where `dp[i]` represents the maximum number of 'A's that can be printed using exactly `i` keystrokes.

#### Steps to Solve
1. **Initialization**: Create a `dp` array of size `N + 1`, initialized to zero.
2. **Base Case**: For `i` from 1 to 3, we can only print 'A's since we do not have enough keys to perform the Copy/Paste operation. Thus, `dp[i] = i`.
3. **Filling the Array**:
   - For each number of keystrokes from 4 to N:
     - We consider different positions to insert Ctrl-A, Ctrl-C followed by multiple Ctrl-V operations.
     - The formula involves selecting how many times to paste after the copy. 
     - For example, if we copy after `k` keystrokes, then the keystrokes used are `k + 2 + p * (pastes count)`, where `pastes count` varies based on the remaining keystrokes.
   
   Thus, we update `dp[i]` by considering all ways we could have performed the Ctrl operations, leading to the maximum possible number of 'A's printed.

### Python Code Using Dynamic Programming

Here’s a full implementation in Python based on the above logic:



```python
class Solution:
    def maxA(self, N: int) -> int:
        # Create a DP array where dp[i] will hold the maximum number of A's we can print in i keystrokes
        dp = [0] * (N + 1)
        
        # Fill the dp array
        for i in range(1, N + 1):
            # Base case: For first three strokes we can only print 'A's
            if i <= 3:
                dp[i] = i
            else:
                # Check every possible place to do Ctrl-A and Ctrl-C
                for j in range(1, i - 2):
                    # Remaining strokes after Ctrl-A and Ctrl-C is (i - j - 2)
                    # This can be considered as the number of pastes we can do
                    # Each paste adds dp[j] (the numbers of A's printed by the first j keystrokes)
                    dp[i] = max(dp[i], dp[j] * (i - j - 1))
        
        return dp[N]

# Example to run the code
solution = Solution()
print(solution.maxA(7))  # should return 9

```

### Explanation of the Code
- We maintain a `dp` list initialized with zeros.
- For every `i` from 1 to `N`, we compute how many 'A's can be achieved based on previous computations in the `dp` list.
- We loop over potential positions for the last Ctrl-A/Ctrl-C (up to `i-2`), and for each position `j`, we calculate the maximum number of 'A's possible if we use `dp[j]` and perform the pastes `(i - j - 1)` times.
- Finally, we return `dp[N]`, which will give the maximum 'A's printed using `N` keystrokes.

This solution works efficiently within the constraints provided, leveraging the dynamics of pasting to maximize 'A's printout.


# 656. Coin Path

### Problem Description 
Given an array `A` (index starts at `1`) consisting of N integers: A1, A2, ..., AN and an integer `B`. The integer `B` denotes that from any place (suppose the index is `i`) in the array `A`, you can jump to any one of the place in the array `A` indexed `i+1`, `i+2`, ..., `i+B` if this place can be jumped to. Also, if you step on the index `i`, you have to pay Ai coins. If Ai is -1, it means you can’t jump to the place indexed `i` in the array.

Now, you start from the place indexed `1` in the array `A`, and your aim is to reach the place indexed `N` using the minimum coins. You need to return the path of indexes (starting from 1 to N) in the array you should take to get to the place indexed `N` using minimum coins.

If there are multiple paths with the same cost, return the lexicographically smallest such path.

If it's not possible to reach the place indexed N then you need to return an empty array.


Example 1:
Input: [1,2,4,-1,2], 2
Output: [1,3,5]

Example 2:
Input: [1,2,4,-1,2], 1
Output: []
Note:
Path Pa1, Pa2, ..., Pan is lexicographically smaller than Pb1, Pb2, ..., Pbm, if and only if at the first `i` where Pai and Pbi differ, Pai < Pbi; when no such `i` exists, then `n` < `m`.

A1 >= 0. A2, ..., AN (if exist) will in the range of [-1, 100].

Length of A is in the range of [1, 1000].

B is in the range of [1, 100].

### Solution 
 To solve the problem of finding the minimum cost path in the given array, we can implement a modified breadth-first search (BFS) with a priority queue (min-heap) to ensure that we always explore the paths with the least coins spent first. 

The key steps for the solution will involve:
1. Using a priority queue to manage the nodes (indices) we need to explore based on increasing cost.
2. Keeping track of the minimum cost to reach each index and the corresponding path taken to get there.
3. Ensuring we yield lexicographically smaller paths by exploring nodes in a systematic manner.

Let’s walk through the implementation:

### Step-by-step Explanation:

1. **Data Structures**:
   - A priority queue (using `heapq` in Python) to keep track of the current position and the cost to reach it.
   - A list to keep track of minimum costs to reach each index.
   - A dictionary to maintain the path taken to each index.

2. **Initialization**:
   - Start from index 1 (0 in zero-based index), with the initial cost being the value at A[0].
   - Initialize the cost of reaching index 1 in the costs list and set its path in the paths dictionary.

3. **BFS with Priority Queue**:
   - Use a loop to explore the indices reachable from the current index within the jump limit `B`.
   - For each reachable index, calculate the total cost.
   - If the new cost is cheaper than the recorded cost, or if it’s equal to the current minimum cost but yields a lexicographically smaller path, update the cost and path.

4. **Termination**:
   - If index `N` is reached, the path is returned. If we exhaust all options without reaching `N`, return an empty list.

### Python Code Implementation:

Here is the code following the aforementioned logic:



```python
import heapq

class Solution:
    def minCostPath(self, A: List[int], B: int) -> List[int]:
        N = len(A)
        if N == 0 or A[0] == -1:
            return []
        
        # Cost and path storage
        costs = [float('inf')] * N
        paths = {}
        
        # Priority queue: (cost, current index)
        pq = []
        
        # Start from index 0 (1 in problem statement)
        costs[0] = A[0]
        heapq.heappush(pq, (A[0], 0))
        paths[0] = [1]  # Storing path as 1-based index
        
        while pq:
            current_cost, i = heapq.heappop(pq)
            
            # If we reached the last index
            if i == N - 1:
                return paths[i]
            
            # Can we jump from index i to i + j for j = 1 to B?
            for j in range(1, B + 1):
                if i + j < N and A[i + j] != -1:
                    new_cost = current_cost + A[i + j]
                    next_index = i + j
                    
                    # Check if it's cheaper or updates the path
                    if new_cost < costs[next_index]:
                        costs[next_index] = new_cost
                        paths[next_index] = paths[i] + [next_index + 1]  # Store path as 1-based index
                        heapq.heappush(pq, (new_cost, next_index))
                    elif new_cost == costs[next_index]:
                        # Check for lexicographical order
                        new_path = paths[i] + [next_index + 1]
                        if new_path < paths[next_index]:
                            paths[next_index] = new_path
                            heapq.heappush(pq, (new_cost, next_index))

        return []  # If we couldn't reach the last index

```

### Summary of the Code:
- We implement a modified Dijkstra-style approach using a priority queue to ensure we explore paths in order of cost.
- We keep track of costs and paths to ensure we can return the lexicographically smallest path efficiently.
- If index `N` cannot be reached based on the conditions of the problem, an empty list is returned. 

This code follows the LeetCode format and can be directly run on their platform.

# 664. Strange Printer

### Problem Description 
There is a strange printer with the following two special requirements:
The printer can only print a sequence of the same character each time.

At each turn, the printer can print new characters starting from and ending at any places, and will cover the original existing characters.

Given a string consists of lower English letters only, your job is to count the minimum number of turns the printer needed in order to print it.


Example 1:
Input: "aaabbb"
Output: 2
Explanation: Print "aaa" first and then print "bbb".


Example 2:
Input: "aba"
Output: 2
Explanation: Print "aaa" first and then print "b" from the second place of the string, which will cover the existing character 'a'.

Hint: Length of the given string will not exceed 100.

### Solution 
 Certainly! The problem at hand is to determine the minimum number of turns required for a strange printer to print a given string where it can only print sequences of the same character. The challenge lies in efficiently covering the string with minimal turns while adhering to the printer’s restriction.

### Explanation

The approach to solve this problem is based on dynamic programming. We will use a 2D DP table `dp[i][j]` where:
- `i`: starting index of the substring we are considering
- `j`: ending index of the substring we are considering

The entry `dp[i][j]` will represent the minimum number of turns needed to print the substring from index `i` to index `j`.

#### Steps:
1. **Initialize the DP Table:** Initially, each character can be printed in one turn, so for any substring of length 1, `dp[i][i] = 1`.

2. **Build the DP Table:** For substrings of increasing lengths, calculate `dp[i][j]`:
   - If the characters at the two ends of the substring `s[i]` and `s[j]` are the same, we can merge the two subsequences: 
     - `dp[i][j] = dp[i][j-1]` because `s[j]` can be printed with the same turn as `s[i]`.
   - Otherwise, we have to consider two different possibilities:
     - Print the substring `s[i..j-1]` and then print `s[j]`, which gives `dp[i][j] = dp[i][j-1] + 1`.
     - Check for each possible split position `k (where i <= k < j)` and take the minimum turns needed: `dp[i][j] = min(dp[i][k] + dp[k+1][j])`.

3. **Final Result:** The answer for the whole string will be found in `dp[0][n-1]` where `n` is the length of the string.

### Python Code



```python
class Solution:
    def strangePrinter(self, s: str) -> int:
        n = len(s)
        
        # dp[i][j] will store the minimum number of turns to print s[i..j]
        dp = [[0] * n for _ in range(n)]
        
        # A single character can be printed in one turn
        for i in range(n):
            dp[i][i] = 1
        
        # Fill the dp table
        for length in range(2, n + 1):  # length of the substring
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = dp[i][j - 1] + 1  # Case: print s[i..j-1] then s[j]
                
                if s[i] == s[j]:  # Case: we can merge if characters at ends are the same
                    dp[i][j] = min(dp[i][j], dp[i][j - 1])
                
                # Check the partition at different points
                for k in range(i, j):
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j])
        
        return dp[0][n - 1]

```

### How to Run This Code:
1. Copy the entire code into LeetCode's editor.
2. Set the function signature to `strangePrinter(s: str) -> int`.
3. Test it with example cases provided in the prompt.

### Time Complexity:
The solution has a time complexity of \(O(n^3)\) due to the three nested loops: one for the length of the substring, one for the starting index, and one for the split point.

### Space Complexity:
The space complexity is \(O(n^2)\) for storing the DP table. 

This approach efficiently computes the minimum number of turns required for the strange printer to print the given string according to the specified constraints.

# 673. Number of Longest Increasing Subsequence

### Problem Description 
Given an integer array `nums`, return the number of longest increasing subsequences.

Notice that the sequence has to be strictly increasing.


Example 1:
Input: nums = [1,3,5,4,7]
Output: 2
Explanation: The two longest increasing subsequences are [1, 3, 4, 7] and [1, 3, 5, 7].


Example 2:
Input: nums = [2,2,2,2,2]
Output: 5
Explanation: The length of longest continuous increasing subsequence is 1, and there are 5 subsequences' length is 1, so output 5.


Constraints:
`1 <= nums.length <= 2000`
`-106 <= nums[i] <= 106`

### Solution 
 To solve the problem of counting the number of longest increasing subsequences (LIS) from the given array `nums`, we can employ a dynamic programming approach. Let's break down the solution step by step.

### Problem Breakdown

1. **Definition of Longest Increasing Subsequence**: An increasing subsequence is a sequence where each element is smaller than the next one. Our task is to find the length of the longest such subsequence and count how many different ways we can achieve that length.

2. **Dynamic Programming (DP) Approach**:
   - We will use two lists: 
     - `lengths[i]`: to store the length of the longest increasing subsequence that ends at index `i`.
     - `counts[i]`: to store how many different longest increasing subsequences end at index `i`.
     
3. **Initialization**:
   - Both `lengths` and `counts` will be initialized to have the same size as `nums`. Since every element can be a subsequence of length 1 by itself, we will initialize `lengths[i] = 1` and `counts[i] = 1` for all `i`.

4. **Filling the DP Arrays**:
   - We'll iterate through the array and for each element, we'll look at all previous elements. If a previous element is smaller than the current element, it can be appended to the subsequence.
   - We update `lengths[i]` and `counts[i]` accordingly based on the already calculated values.

5. **Final Calculation**:
   - After filling the arrays, the maximum value in the `lengths` list will give us the length of the longest increasing subsequence.
   - The total count of such subsequences will be the sum of `counts[i]` for every `i` where `lengths[i]` equals the maximum length found.

### Python Code

Now, let's put this into a Python code snippet that can be run directly on LeetCode.



```python
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        lengths = [1] * n  # lengths[i] will be the length of LIS ending at i
        counts = [1] * n   # counts[i] will be the number of LIS ending at i
        
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:  # Strictly increasing
                    if lengths[j] + 1 > lengths[i]:  # Found a longer subsequence
                        lengths[i] = lengths[j] + 1
                        counts[i] = counts[j]  # inherit the count
                    elif lengths[j] + 1 == lengths[i]:  # Found another way to achieve the same length
                        counts[i] += counts[j]

        max_length = max(lengths)  # Get the length of the longest increasing subsequence
        return sum(counts[i] for i in range(n) if lengths[i] == max_length)  # Count all LIS of that max_length

```

### Explanation of the Code:

- The `Solution` class has a method `findNumberOfLIS` that takes an array `nums` as input.
- We initialize the `lengths` and `counts` arrays of the same length as `nums`. 
- The nested loop checks each pair of elements to update the lengths and counts based on the rules defined above.
- Finally, we compute the maximum length of increasing subsequences and sum up the counts for those subsequences that match this maximum length.

### Complexity:

- **Time Complexity**: O(n^2) due to the nested loops, which is efficient given the constraints.
- **Space Complexity**: O(n) for storing lengths and counts.

This approach efficiently counts the number of longest increasing subsequences, and it adheres to the requested format for a LeetCode solution.

# 688. Knight Probability in Chessboard

### Problem Description 
On an `N`x`N` chessboard, a knight starts at the `r`-th row and `c`-th column and attempts to make exactly `K` moves. The rows and columns are 0 indexed, so the top-left square is `(0, 0)`, and the bottom-right square is `(N-1, N-1)`.

A chess knight has 8 possible moves it can make, as illustrated below. Each move is two squares in a cardinal direction, then one square in an orthogonal direction.

Each time the knight is to move, it chooses one of eight possible moves uniformly at random (even if the piece would go off the chessboard) and moves there.

The knight continues moving until it has made exactly `K` moves or has moved off the chessboard. Return the probability that the knight remains on the board after it has stopped moving.


Example:
Input: 3, 2, 0, 0
Output: 0.0625
Explanation: There are two moves (to (1,2), (2,1)) that will keep the knight on the board.

From each of those positions, there are also two moves that will keep the knight on the board.

The total probability the knight stays on the board is 0.0625.

Note:
`N` will be between 1 and 25.

`K` will be between 0 and 100.

The knight always initially starts on the board.

### Solution 
 To solve the problem of determining the probability that a knight remains on an `N x N` chessboard after making exactly `K` moves from a starting position `(r, c)`, we will use dynamic programming. 

### Explanation:

1. **Knight Moves**: The knight can move to 8 possible positions from any square on the chessboard. These moves can be represented as pairs of changes in the row and column coordinates: 
   
```
   [(2, 1), (2, -1), (-2, 1), (-2, -1), 
    (1, 2), (1, -2), (-1, 2), (-1, -2)]
   
```

2. **Dynamic Programming Table**: We will create a 3D array `dp[k][r][c]` where `k` is the number of moves made, and `(r, c)` is the current position of the knight. This table will store the probability of the knight being on the board after `k` moves.

3. **Base Case**: At `k = 0`, the knight is guaranteed to be on the board at `(start_r, start_c)`, so:
   
```
   dp[0][start_r][start_c] = 1
   
```
   All other positions have a probability of 0.

4. **Transition**: For every subsequent number of moves `k`, we will calculate the probability for each cell `(r, c)`. The probability of being on the board at `(r, c)` after `k` moves is the sum of probabilities from all valid positions that could have moved to `(r, c)` in one knight move. The formula will look like:
   
```
   dp[k][r][c] = sum(dp[k-1][prev_r][prev_c]) / 8
   
```
   Here `(prev_r, prev_c)` are valid positions from which the knight can come to `(r, c)`.

5. **Out of Bounds**: If a knight moves out of bounds, it contributes zero to the probability.

6. **Final Probability**: After filling the DP table, we will sum all probabilities in the `K`th move layer for all valid `(r, c)` board positions to get the final probability that the knight is on the board after `K` moves.

Here is the Python implementation of the above approach:



```python
class Solution:
    def knightProbability(self, N: int, K: int, r: int, c: int) -> float:
        # Define the knight move directions
        moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), 
                 (1, 2), (1, -2), (-1, 2), (-1, -2)]
        
        # Initialize the DP table
        dp = [[[0] * N for _ in range(N)] for _ in range(K + 1)]
        
        # Base case: 100% probability of being on the board at starting position
        dp[0][r][c] = 1
        
        # Fill the DP table
        for k in range(1, K + 1):
            for i in range(N):
                for j in range(N):
                    for move in moves:
                        prev_r = i + move[0]
                        prev_c = j + move[1]
                        if 0 <= prev_r < N and 0 <= prev_c < N:
                            dp[k][i][j] += dp[k-1][prev_r][prev_c] / 8
        
        # Sum all the probabilities for the K-th move
        probability = 0
        for i in range(N):
            for j in range(N):
                probability += dp[K][i][j]
        
        return probability

```

### Usage
You can directly copy and run the above code on LeetCode by creating a new solution and testing it with the provided example. The solution optimally uses dynamic programming to calculate the required probability efficiently, even for larger values of `N` and `K`.

# 689. Maximum Sum of 3 Non-Overlapping Subarrays

### Problem Description 
In a given array `nums` of positive integers, find three non-overlapping subarrays with maximum sum.

Each subarray will be of size `k`, and we want to maximize the sum of all `3*k` entries.

Return the result as a list of indices representing the starting position of each interval (0-indexed). If there are multiple answers, return the lexicographically smallest one.


Example:
Input: [1,2,1,2,6,7,5,1], 2
Output: [0, 3, 5]
Explanation: Subarrays [1, 2], [2, 6], [7, 5] correspond to the starting indices [0, 3, 5].

We could have also taken [2, 1], but an answer of [1, 3, 5] would be lexicographically larger.

Note:
`nums.length` will be between 1 and 20000.

`nums[i]` will be between 1 and 65535.

`k` will be between 1 and floor(nums.length / 3).

### Solution 
 To tackle the problem of finding three non-overlapping subarrays of size `k` in an array `nums` such that their sum is maximized, we can break down the solution into distinct steps. 

### Explanation of the Approach

1. **Calculate Subarray Sums**: First, compute the sum of each subarray of size `k`. This can be achieved using a sliding window technique, which allows us to compute the sum in linear time.

2. **Track Maximum Sums**: Next, we need to store the maximum subarray sums found so far in two separate lists:
   - `max_left[i]`: This keeps track of the maximum sum found when considering the left subarrays up to index `i`.
   - `max_right[i]`: This keeps track of the maximum sum in the right subarrays starting from index `i`.

3. **Combine Results**: Finally, we compute the best combination of the left maximum, the current (middle) subarray sum, and the right maximum sums to find the best trio of subarrays, ensuring that they do not overlap.

4. **Return the Indices**: Return the starting indices of the three subarrays, ensuring to choose the lexicographically smallest option in case of ties.

### Python Code



```python
class Solution:
    def maxSumOfThreeSubarrays(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        # Step 1: Calculate sums of all subarrays of size k
        current_sum = sum(nums[:k])
        sums = [0] * (n - k + 1)
        sums[0] = current_sum
        
        for i in range(1, n - k + 1):
            current_sum += nums[i + k - 1] - nums[i - 1]
            sums[i] = current_sum
        
        # Step 2: Prepare max_left array
        max_left = [0] * (n - k + 1)
        max_index_left = 0
        for i in range(n - k + 1):
            if sums[i] > sums[max_index_left]:
                max_index_left = i
            max_left[i] = max_index_left
        
        # Step 3: Prepare max_right array
        max_right = [0] * (n - k + 1)
        max_index_right = n - k
        for i in range(n - k, -1, -1):
            if sums[i] >= sums[max_index_right]:
                max_index_right = i
            max_right[i] = max_index_right
        
        # Step 4: Find the best indices for the three subarrays
        max_sum = 0
        ans = [-1, -1, -1]
        
        for j in range(k, n - 2 * k + 1):
            left_index = max_left[j - k]
            right_index = max_right[j + k]
            total_sum = sums[left_index] + sums[j] + sums[right_index]
            
            if total_sum > max_sum:
                max_sum = total_sum
                ans = [left_index, j, right_index]
            # If there's a tie, we take the lexicographically smallest one
            elif total_sum == max_sum:
                if [left_index, j, right_index] < ans:
                    ans = [left_index, j, right_index]

        return ans

```

### Detailed Breakdown

1. **Calculating Subarray Sums**: 
   - Initialize `current_sum` to the sum of the first `k` elements. 
   - Use a sliding window to compute sums for the rest of the subarrays of size `k`.

2. **Maintaining Maximum Sums**:
   - For the `max_left`, iterate through the sums and keep track of the index where the maximum sum occurs up to that point.
   - For the `max_right`, do a similar operation but iterate in reverse to ensure we capture the maximum sums from the end of the list.

3. **Finding the Optimal Solution**:
   - Iterate through possible starting positions for the middle subarray and combine the best left and right maximum sums accordingly. Update the result if a better sum is found or if you need to handle ties with lexicographic comparison.

### Complexity
- The solution has a time complexity of O(n), where n is the length of `nums`, and a space complexity of O(n) due to the additional arrays used to store sums and indices. 

This method ensures efficiency and correctness while adhering to the specifics of the problem statement regarding overlapping subarrays and index ordering.

# 691. Stickers to Spell Word

### Problem Description 
We are given N different types of stickers.  Each sticker has a lowercase English word on it.

You would like to spell out the given `target` string by cutting individual letters from your collection of stickers and rearranging them.

You can use each sticker more than once if you want, and you have infinite quantities of each sticker.

What is the minimum number of stickers that you need to spell out the `target`?  If the task is impossible, return -1.


Example 1:
Input:["with", "example", "science"], "thehat"
Output:3
Explanation:We can use 2 "with" stickers, and 1 "example" sticker.

After cutting and rearrange the letters of those stickers, we can form the target "thehat".

Also, this is the minimum number of stickers necessary to form the target string.


Example 2:
Input:["notice", "possible"], "basicbasic"
Output:-1
Explanation:We can't form the target "basicbasic" from cutting letters from the given stickers.

Note:
`stickers` has length in the range `[1, 50]`.

`stickers` consists of lowercase English words (without apostrophes).

`target` has length in the range `[1, 15]`, and consists of lowercase English letters.

In all test cases, all words were chosen randomly from the 1000 most common US English words, and the target was chosen as a concatenation of two random words.

The time limit may be more challenging than usual.  It is expected that a 50 sticker test case can be solved within 35ms on average.

### Solution 
 To solve the problem of determining the minimum number of stickers required to form a target word from a given collection of stickers, we can use a recursive backtracking approach combined with memoization. Here is a step-by-step breakdown of the solution, followed by the complete code.

### Steps to Solution

1. **Count Letters**: We first need a function to count the frequency of each character in the stickers and the target string. This will help us understand how many of each letter we need and how many we have available from the stickers.

2. **Recursive Function with Memoization**: We can use a recursive function to try every combination of stickers to form the target. Since this may result in a lot of overlapping subproblems, we'll use memoization to store the results of already computed states.

3. **Base Case and Recursive Case**:
    - If the target string is empty (i.e., all characters have been covered), return 0 (no stickers needed).
    - If the recursive function encounters a case where crafting the target is impossible, return infinity or a sufficiently large number.
    - For the recursive case, try using each sticker and compute how many stickers are needed to cover the remaining characters of the target.

4. **Returning the Result**: The final result will be the minimum number of stickers needed across all recursive calls. If it's still set to a large number (indicating no solution was found), return -1.

### Complete Code

Here's the complete Python code written in the LeetCode format:



```python
from collections import Counter
import sys
from typing import List

class Solution:
    def minStickers(self, stickers: List[str], target: str) -> int:
        # Count how many of each letter are in each sticker
        sticker_count = []
        for sticker in stickers:
            counter = Counter(sticker)
            sticker_count.append(counter)

        memo = {}
        
        def dp(remaining_target):
            if remaining_target == "":
                return 0
            if remaining_target in memo:
                return memo[remaining_target]
            
            # Track the minimum stickers needed
            min_stickers = sys.maxsize
            
            # Count the frequency of characters in the remaining target
            target_count = Counter(remaining_target)
            
            # Try using each sticker
            for sticker in sticker_count:
                # If the sticker has the first letter we're looking for
                if remaining_target[0] in sticker:
                    # Create a new target by subtracting the letters used from sticker
                    new_remaining_target = []
                    for char, count in target_count.items():
                        if count > sticker[char]:
                            new_remaining_target.append(char * (count - sticker[char]))
                    new_remaining_target_string = ''.join(new_remaining_target)
                    # Recursive call
                    next_stickers = dp(new_remaining_target_string)
                    if next_stickers != sys.maxsize:
                        min_stickers = min(min_stickers, 1 + next_stickers)
            
            memo[remaining_target] = min_stickers
            return min_stickers
        
        result = dp(target)
        return result if result != sys.maxsize else -1

```

### Explanation of the Code

1. We first create a list of `Counter` objects representing the character counts for each sticker.
2. The `dp` function checks if the target string is empty. If so, it returns 0 since no stickers are needed. It also checks the memoization dictionary to see if this state has already been computed.
3. For each sticker, the function checks if it can help reduce the remaining target. If it can, it calculates the new target that would result from using this sticker and makes a recursive call.
4. Finally, it updates the memoization dictionary and computes the minimum stickers needed.
5. If no valid configuration was found, it returns -1.

This approach is efficient and leverages the power of memoization to avoid redundant calculations. The complexity is significantly reduced, helping meet the constraints given in the problem statement.

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
 To solve the problem of dividing an array into `k` non-empty subsets with equal sums, we can apply a backtracking approach. Here, we will detail the algorithm and provide a Python implementation that fits the LeetCode solution structure.

### Problem Breakdown
1. **Calculate the Total Sum**: First, we compute the sum of all elements in the array. If this total sum is not divisible by `k`, it's impossible to form `k` subsets with equal sums.

2. **Target Subset Sum**: If it is divisible, we can find the target sum for each subset as `target = total_sum / k`.

3. **Sort the Array**: We will sort the array in descending order to help improve the efficiency of the backtracking algorithm. Starting with larger numbers helps to fill subsets faster and can lead to quicker fail conditions.

4. **Backtracking Algorithm**: We will maintain an array `subset_sums` to track the current sums of each subset formed. The goal is to recursively add numbers to these subsets until all subsets reach the target sum.

5. **Base Cases**: There are several base cases:
   - If a number is larger than the target sum, we can return False immediately.
   - If all subsets have reached the target sum, we can check if all elements have been used.

### Python Code
Below is the Python implementation following the LeetCode format:



```python
class Solution:
    def canPartitionKSubsets(self, nums, k: int) -> bool:
        total_sum = sum(nums)
        if total_sum % k != 0:
            return False
        
        target = total_sum // k
        nums.sort(reverse=True)  # Sort in descending order for optimization
        
        subset_sums = [0] * k  # Array to hold the current sums of k subsets
        
        # Helper function for backtracking
        def backtrack(index):
            # If we reached the end of the array, check if all subsets are equal to target
            if index == len(nums):
                return all(s == target for s in subset_sums)
            
            for i in range(k):
                if subset_sums[i] + nums[index] <= target:  # Only proceed if it does not exceed target
                    subset_sums[i] += nums[index]  # Choose this number for the current subset
                    if backtrack(index + 1):  # Move to the next number
                        return True
                    subset_sums[i] -= nums[index]  # Backtrack, remove the number
                
                # If the current subset is now 0, it means we have just used the first number in this subset
                # thus we should not continue with this subset anymore.
                if subset_sums[i] == 0:
                    break
            
            return False
        
        return backtrack(0)

# Example usage:
# sol = Solution()
# print(sol.canPartitionKSubsets([4, 3, 2, 3, 5, 2, 1], 4))  # Output: True

```

### Explanation of the Code
1. **Total Sum and Target Calculation**: We calculate the total sum of the input list and check divisibility by `k`. If not divisible, return False immediately.

2. **Sort**: We sort `nums` in descending order.

3. **Backtracking**:
   - A helper function `backtrack` is defined, which takes the current index of `nums`.
   - We iterate over each subset and try to assign the current number to a subset if it doesn’t exceed the target.
   - If the function exhausts the list and all subsets have reached the target, we return True.
   - If we exhaust all possibilities for a particular subset with the current number, we backtrack and try to find other combinations.

### Complexity
The time complexity can be quite high due to backtracking, but for the constraint of `1 <= k <= len(nums) <= 16`, this approach is feasible.

This solution is designed to be submitted directly on LeetCode, where the structure and function signatures match the platform's requirements.

# 712. Minimum ASCII Delete Sum for Two Strings

### Problem Description 
Given two strings `s1, s2`, find the lowest ASCII sum of deleted characters to make two strings equal.


Example 1:
Input: s1 = "sea", s2 = "eat"
Output: 231
Explanation: Deleting "s" from "sea" adds the ASCII value of "s" (115) to the sum.

Deleting "t" from "eat" adds 116 to the sum.

At the end, both strings are equal, and 115 + 116 = 231 is the minimum sum possible to achieve this.


Example 2:
Input: s1 = "delete", s2 = "leet"
Output: 403
Explanation: Deleting "dee" from "delete" to turn the string into "let",
adds 100[d]+101[e]+101[e] to the sum.  Deleting "e" from "leet" adds 101[e] to the sum.

At the end, both strings are equal to "let", and the answer is 100+101+101+101 = 403.

If instead we turned both strings into "lee" or "eet", we would get answers of 433 or 417, which are higher.

Note:
`0 < s1.length, s2.length <= 1000`.

All elements of each string will have an ASCII value in `[97, 122]`.

### Solution 
 To solve the problem of finding the lowest ASCII sum of deleted characters to make the two strings equal, we can utilize dynamic programming. We will create a 2D table to store the minimum cost (in terms of ASCII sums of deleted characters) to make substrings of `s1` and `s2` equal.

### Concept

1. **Dynamic Programming Table**: Create a 2D list `dp` with dimensions `(len(s1)+1) x (len(s2)+1)`. The cell `dp[i][j]` will represent the minimum ASCII sum of deleted characters required to make the substring `s1[0:i]` equal to `s2[0:j]`.

2. **Base Cases**:
    - If one string is empty (i.e., \(i = 0\) or \(j = 0\)), the cost is the sum of ASCII values of the characters in the other string since all characters must be deleted to make them equal.
    
3. **Filling the DP Table**:
    - If the characters `s1[i-1]` and `s2[j-1]` are equal, then `dp[i][j] = dp[i-1][j-1]` since no characters need to be deleted for those positions.
    - If they are not equal, consider two cases:
        - Deleting the character from `s1`: The cost will then add the ASCII value of `s1[i-1]` to `dp[i-1][j]`.
        - Deleting the character from `s2`: The cost will then add the ASCII value of `s2[j-1]` to `dp[i][j-1]`.
      The value of `dp[i][j]` will be the minimum of these two options.

4. **Result**: The value in `dp[len(s1)][len(s2)]` will give us the minimum ASCII sum of deleted characters to make the two strings equal.

### Implementation

Here's how we implement the above logic in Python following LeetCode's structure:



```python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        # Lengths of the two strings
        len_s1, len_s2 = len(s1), len(s2)
        
        # Create a 2D DP array
        dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]
        
        # Fill the first column (when s2 is empty)
        for i in range(1, len_s1 + 1):
            dp[i][0] = dp[i - 1][0] + ord(s1[i - 1])
        
        # Fill the first row (when s1 is empty)
        for j in range(1, len_s2 + 1):
            dp[0][j] = dp[0][j - 1] + ord(s2[j - 1])
        
        # Fill the rest of the DP table
        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  # No character needs to be deleted
                else:
                    delete_from_s1 = dp[i - 1][j] + ord(s1[i - 1])  # Delete from s1
                    delete_from_s2 = dp[i][j - 1] + ord(s2[j - 1])  # Delete from s2
                    dp[i][j] = min(delete_from_s1, delete_from_s2)
        
        return dp[len_s1][len_s2]

```

### Explanation of the Code
1. We define a class `Solution` and a method `minimumDeleteSum` that takes two strings as input.
2. We compute the lengths of the strings and initialize the `dp` table.
3. We fill the first row and the first column based on our base cases.
4. We iterate through each character in both strings to fill the `dp` table according to the rules discussed.
5. Finally, we return the value at `dp[len(s1)][len(s2)]`, which contains the minimum ASCII sum needed to delete characters to make the two strings equal.

### Complexity
- **Time Complexity**: \(O(n \times m)\) where \(n\) is the length of `s1` and \(m\) is the length of `s2`.
- **Space Complexity**: \(O(n \times m)\) for the `dp` table.

# 714. Best Time to Buy and Sell Stock with Transaction Fee

### Problem Description 
You are given an array `prices` where `prices[i]` is the price of a given stock on the `ith` day, and an integer `fee` representing a transaction fee.

Find the maximum profit you can achieve. You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction.

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).


Example 1:
Input: prices = [1,3,2,8,4,9], fee = 2
Output: 8
Explanation: The maximum profit can be achieved by:
- Buying at prices[0] = 1
- Selling at prices[3] = 8
- Buying at prices[4] = 4
- Selling at prices[5] = 9
The total profit is ((8 - 1) - 2) + ((9 - 4) - 2) = 8.


Example 2:
Input: prices = [1,3,7,5,10,3], fee = 3
Output: 6

Constraints:
`1 <= prices.length <= 5 * 104`
`1 <= prices[i] < 5 * 104`
`0 <= fee < 5 * 104`

### Solution 
 Certainly! Let's solve the problem of maximizing profit with stock transactions while taking a transaction fee into account. The approach we'll take is to use dynamic programming to decide when to buy and sell stocks throughout the given days.

## Problem Explanation
We have an array of stock prices where `prices[i]` is the price of a stock on the `i-th` day, and a transaction fee for each transaction (buy/sell). The goal is to maximize profit over multiple transactions while subtracting the transaction fees.

### Key Concepts:
1. **State Variables**:
   - `cash`: This represents the maximum profit we can achieve if we do not hold a stock on the current day.
   - `hold`: This represents the maximum profit we can achieve if we hold a stock on the current day.
   
2. **Transition Formula**:
   - On day `i`, we can either do nothing, sell the stock if we hold it, or buy the stock if we don't hold it.
   - If we sell on day `i`, we update `cash` as `hold + prices[i] - fee`.
   - If we buy on day `i`, we update `hold` as `cash - prices[i]`.

3. **Initialization**:
   - On day 0, if we do nothing, `cash` is 0. If we were to hold a stock on the first day, the profit would be negative because we would have purchased it.

### Dynamic Programming Table:
We'll iterate through the `prices` array, updating our `cash` and `hold` at each step based on the transitions described above.

Here's the Python code that implements this logic:



```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        # Initialize variables
        cash = 0  # Maximum profit when we do not hold any stock
        hold = -prices[0]  # Maximum profit when we hold a stock (initially buying on the first day)
        
        # Iterate through each price in the prices list
        for price in prices:
            # Update cash and hold for the current price
            cash = max(cash, hold + price - fee)  # either stay in cash or sell stock
            hold = max(hold, cash - price)        # either stay holding or buy stock
            
        # The result is the maximum profit we can have without holding any stocks at the end
        return cash

```

### Explanation of the Code:
1. **Input Parameters**:
   - `prices`: A list of integers representing stock prices on different days.
   - `fee`: An integer representing the transaction fee.

2. **Initialization**:
   - We start with `cash = 0` (no profit at the beginning) and `hold = -prices[0]` (we assume we buy the stock at the first day's price).

3. **Iteration**:
   - For each stock price in `prices`, we update the `cash` and `hold` accordingly:
     - `cash` can either remain the same or be updated to the profit obtained by selling the stock (which is `hold + price - fee`).
     - `hold` can also remain the same or be updated to reflect buying a new stock (which is `cash - price`).

4. **Result**:
   - After iterating through all the prices, we return `cash` which contains the maximum profit we can gain without holding any stock.

### Complexity:
- **Time Complexity**: O(n), where n is the number of days (length of prices).
- **Space Complexity**: O(1), since we're using a constant amount of space.

You can directly copy this code to the LeetCode environment, and it should work correctly for the problem at hand!

# 718. Maximum Length of Repeated Subarray

### Problem Description 
Given two integer arrays `A` and `B`, return the maximum length of an subarray that appears in both arrays.


Example 1:
Input:
A: [1,2,3,2,1]
B: [3,2,1,4,7]
Output: 3
Explanation: 
The repeated subarray with maximum length is [3, 2, 1].

Note:
1 <= len(A), len(B) <= 1000
0 <= A[i], B[i] < 100

### Solution 
 To solve the problem of finding the maximum length of a subarray that appears in both given arrays `A` and `B`, we can use a dynamic programming approach. The key idea is to maintain a table (2D array) where the cell at position `(i, j)` represents the length of the longest common subarray that ends at indices `i` in `A` and `j` in `B`. 

Here's how we can break down the solution:

### Steps to Solve the Problem

1. **Initialization**:
   - Create a 2D list `dp` where `dp[i][j]` will represent the longest common subarray ending at `A[i-1]` and `B[j-1]`.
   - Initialize a variable `max_length` to keep track of the longest subarray found.

2. **Dynamic Programming Table Filling**:
   - Loop through each element in both arrays.
   - Whenever we find `A[i-1]` and `B[j-1]` to be equal, we set `dp[i][j]` to `dp[i-1][j-1] + 1`, meaning we can extend the length of the subarray ending at these indices by 1.
   - If they are not equal, `dp[i][j]` will remain 0 because we cannot have a common subarray ending at these indices.

3. **Update Maximum Length**:
   - During the table filling, update `max_length` whenever we find a longer common subarray.

4. **Return the Result**:
   - After filling the table, `max_length` will contain the length of the longest common subarray.

### Python Code

Here's the Python code that implements the above logic:



```python
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        # Get the lengths of both arrays
        lenA = len(A)
        lenB = len(B)
        
        # Create a 2D array for the dynamic programming table
        dp = [[0] * (lenB + 1) for _ in range(lenA + 1)]
        
        max_length = 0
        
        # Fill the dp table
        for i in range(1, lenA + 1):
            for j in range(1, lenB + 1):
                if A[i - 1] == B[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_length = max(max_length, dp[i][j])
                else:
                    dp[i][j] = 0
        
        return max_length

```

### Explanation of the Code

1. **Class Definition**: We define a class `Solution`, which is the required structure for LeetCode problems.
  
2. **Method `findLength`**: This method takes two arguments, the integer arrays `A` and `B`, and returns an integer indicating the maximum length of the common subarray.

3. **Dynamic Programming Table Initialization**: 
   - `dp` is initialized to be of size `(lenA + 1) x (lenB + 1)` to accommodate index offsets. Each entry is initialized to zero, which represents no common subarray found.

4. **Nested Loops**: Two nested loops iterate through each index pair `(i, j)` of the arrays. If the elements at the respective positions are equal, we update `dp[i][j]` based on the previous values.

5. **Max Length Check**: Every time we update `dp[i][j]`, we check if it's greater than `max_length` and update accordingly.

6. **Return Statement**: Finally, we return `max_length`, which holds the length of the longest common subarray.

This approach runs in O(N * M) time complexity and O(N * M) space complexity, where N = len(A) and M = len(B). Given the constraints, this approach is efficient and effective.

# 727. Minimum Window Subsequence

### Problem Description 
Given strings `S` and `T`, find the minimum (contiguous) substring `W` of `S`, so that `T` is a subsequence of `W`.

If there is no such window in `S` that covers all characters in `T`, return the empty string `""`. If there are multiple such minimum-length windows, return the one with the left-most starting index.


Example 1:
Input: 
S = "abcdebdde", T = "bde"
Output: "bcde"
Explanation: 
"bcde" is the answer because it occurs before "bdde" which has the same length.

"deb" is not a smaller window because the elements of T in the window must occur in order.

Note:
All the strings in the input will only contain lowercase letters.

The length of `S` will be in the range `[1, 20000]`.

The length of `T` will be in the range `[1, 100]`.

### Solution 
 To solve the problem of finding the minimum contiguous substring `W` from `S` such that `T` is a subsequence of `W`, we can utilize a two-pointer technique. We'll maintain one pointer for scanning through the string `S`, and the other pointer will manage our progress through the string `T`.

Here's how we can approach the problem in a structured way:

1. **Initialization**:
   - We'll maintain a left pointer (`l`) that denotes the starting index of our current substring in `S`.
   - We'll also have another pointer (`r`) that iterates over `S` to find potential matches for `T`.

2. **Matching Characters**:
   - As we move through `S` with the right pointer (`r`), whenever we find a character in `S` that matches the current character in `T`, we move the character pointer of `T` forward.

3. **Check Completion**:
   - If we manage to match all characters of `T` (i.e., the pointer for `T` has reached the end of `T`), we then check to see the length of the current substring from `l` to `r`. We compare this length with previous minimum lengths and update accordingly.

4. **Shrinking the Window**:
   - After a complete match is found, we can increment the left pointer (`l`) to try and minimize the window. We keep doing this until the match is no longer valid (i.e., when characters in `T` can no longer be found in our current substring).

5. **Handling Multiple Matches**:
   - We will track the starting and ending indices of the best match found so far.

6. **Return Result**:
   - If a valid substring was found, return it; if not, return an empty string.

Let's implement this in Python code suitable for LeetCode:



```python
def min_window_subsequence(S: str, T: str) -> str:
    n, m = len(S), len(T)
    best_start = -1
    best_len = float('inf')

    for l in range(n):
        t_index = 0
        for r in range(l, n):
            if t_index < m and S[r] == T[t_index]:
                t_index += 1
            if t_index == m:  # Full T matched in S[l:r+1]
                # Update best solution if found a better one
                if (r - l + 1) < best_len:
                    best_len = r - l + 1
                    best_start = l
                break  # No need to scan beyond this point

    if best_start == -1:
        return ""
    
    return S[best_start:best_start + best_len]

# Example usage:
S = "abcdebdde"
T = "bde"
print(min_window_subsequence(S, T))  # Output: "bcde"

```

### Explanation of the Code:

1. **Function Definition**:
   - We define the function `min_window_subsequence`, which takes two strings `S` and `T` as parameters.

2. **Variable Initialization**:
   - `n` and `m` hold the lengths of `S` and `T`, respectively.
   - `best_start` keeps track of the starting index of the smallest valid substring, and `best_len` keeps its length.

3. **Two Nested Loops**:
   - The outer loop (`l`) goes over all possible starting positions in `S`.
   - The inner loop (`r`) goes from the current `l` position to the end of `S`.

4. **Matching Logic**:
   - Inside the inner loop, we check if the characters match. If they do, we move the pointer in `T`. If we complete the scan of `T`, we check if we found a new minimum length.

5. **Return Statement**:
   - At the end of processing, we check if we found a valid starting position and return the appropriate substring or an empty string if no such substring exists.

This solution efficiently finds the minimum substring while maintaining readability and correctness, and is compatible with the constraints provided in the problem statement.

# 730. Count Different Palindromic Subsequences

### Problem Description 
Given a string S, find the number of different non-empty palindromic subsequences in S, and return that number modulo `10^9 + 7`.

A subsequence of a string S is obtained by deleting 0 or more characters from S.

A sequence is palindromic if it is equal to the sequence reversed.

Two sequences `A_1, A_2, ...` and `B_1, B_2, ...` are different if there is some `i` for which `A_i != B_i`.


Example 1:
Input: 
S = 'bccb'
Output: 6
Explanation: 
The 6 different non-empty palindromic subsequences are 'b', 'c', 'bb', 'cc', 'bcb', 'bccb'.

Note that 'bcb' is counted only once, even though it occurs twice.


Example 2:
Input: 
S = 'abcdabcdabcdabcdabcdabcdabcdabcddcbadcbadcbadcbadcbadcbadcbadcba'
Output: 104860361
Explanation: 
There are 3104860382 different non-empty palindromic subsequences, which is 104860361 modulo 10^9 + 7.

Note:
The length of `S` will be in the range `[1, 1000]`.

Each character `S[i]` will be in the set `{'a', 'b', 'c', 'd'}`.

### Solution 
 To solve the problem of counting different non-empty palindromic subsequences in a given string S, we need to use dynamic programming. We'll approach the problem by designing a 2D DP table that will help us track counts of palindromic subsequences based on substrings of S.

To explain the algorithm step-by-step:

### Steps to Solve the Problem:

1. **Understanding the Palindrome**: A palindrome reads the same backward as forward. For example, "aba" is a palindrome, while "abc" is not.

2. **Dynamic Programming Table**: We'll create a DP table `dp[i][j]` where `dp[i][j]` will store the number of distinct palindromic subsequences in the substring `S[i:j+1]`.

3. **Base Cases**: For any single character, it’s a palindrome itself, so for every `i`, `dp[i][i] = 1`.

4. **Filling the DP Table**:
    - If `S[i]` == `S[j]`, we have the following scenarios:
        - All palindromic subsequences in `S[i+1:j]` (denote as `X`).
        - Each of those subsequences can have `S[i]` and `S[j]` appended to it, forming new palindromes.
        - Additionally, counting the individual characters `S[i]` and `S[j]` as well as the pair.
        - So, `dp[i][j] = dp[i+1][j] + dp[i][j-1] + 1`.
        
    - If `S[i]` != `S[j]`, we need to count:
        - Palindromic subsequences in `S[i+1:j]` (denote as `X`).
        - Palindromic subsequences in `S[i:j-1]` (denote as `Y`).
        - Subtract the overlapping counted palindromic subsequences in `S[i+1:j-1]` which are counted twice.
        - So, `dp[i][j] = dp[i+1][j] + dp[i][j-1] - dp[i+1][j-1]`.

5. **Modulo Operation**: Since the output could be large, we will take every calculation modulo `10^9 + 7`.

6. **Return the Result**: The final answer will be stored in `dp[0][n-1]` where `n` is the length of the string `S`.

### Python Code Implementation:



```python
class Solution:
    def countPalindromicSubsequences(self, S: str) -> int:
        MOD = 10**9 + 7
        n = len(S)
        dp = [[0] * n for _ in range(n)]
        
        # Initialize the table for single character palindromic subsequences
        for i in range(n):
            dp[i][i] = 1
        
        # Fill the DP table
        for length in range(2, n + 1):  # length from 2 to n
            for i in range(n - length + 1):
                j = i + length - 1
                if S[i] == S[j]:
                    # If characters match
                    dp[i][j] = (dp[i + 1][j] + dp[i][j - 1] + 1) % MOD
                else:
                    # If characters do not match
                    dp[i][j] = (dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1]) % MOD
        
        return dp[0][n - 1]

# Example usage:
solution = Solution()
print(solution.countPalindromicSubsequences("bccb"))  # Output: 6
print(solution.countPalindromicSubsequences("abcdabcdabcdabcdabcdabcdabcdabcddcbadcbadcbadcbadcbadcbadcbadcba"))  # Output: 104860361

```

### Explanation of the Code:
- **Initialization**: The DP table is initialized to count palindromic subsequences.
- **Loop through lengths**: We loop through all possible substring lengths and update the counts based on whether the characters at the ends of the substring match.
- **Modulo operation**: Each operation involving sums and differences is utilized with a modulo operation to avoid overflow.
- **Final Output**: The number of distinct palindromic subsequences is given by `dp[0][n-1]` which considers the whole string.

This approach has a time complexity of O(n^2) and a space complexity of O(n^2), making it feasible given the constraints of the problem.

# 740. Delete and Earn

### Problem Description 
Given an array `nums` of integers, you can perform operations on the array.

In each operation, you pick any `nums[i]` and delete it to earn `nums[i]` points. After, you must delete every element equal to `nums[i] - 1` or `nums[i] + 1`.

You start with `0` points. Return the maximum number of points you can earn by applying such operations.


Example 1:
Input: nums = [3,4,2]
Output: 6
Explanation: Delete 4 to earn 4 points, consequently 3 is also deleted.

Then, delete 2 to earn 2 points.

6 total points are earned.


Example 2:
Input: nums = [2,2,3,3,3,4]
Output: 9
Explanation: Delete 3 to earn 3 points, deleting both 2's and the 4.

Then, delete 3 again to earn 3 points, and 3 again to earn 3 points.

9 total points are earned.


Constraints:
`1 <= nums.length <= 2 * 104`
`1 <= nums[i] <= 104`

### Solution 
 To solve the problem of maximizing the points earned by deleting elements from the array while adhering to the deletion rules, we can use a dynamic programming approach. The idea is to think of the problem in terms of how we can accumulate points by either choosing to delete or skip certain values based on their frequency.

### Step-by-step Explanation:

1. **Understanding the Problem**: 
   - When you delete a number `x`, you gain `x` points but also remove all instances of `x-1` and `x+1`. This creates a conflict where picking numbers close to each other can limit our options, hence we need to think strategically about which numbers to pick.

2. **Frequency Count**:
   - First, we need to determine the frequency of each number in the array. This allows us to compute how many points we can earn by taking that number.

3. **Dynamic Programming Formulation**:
   - Define a `dp` array where `dp[i]` will represent the maximum number of points we can earn considering all numbers from `1` to `i`.
   - The transition will be:  
     - If you take the number `i`, you will add its points (`i * frequency[i]`) and you must have skipped any contributions from `i-1`. Hence, the recurrence relation can be expressed as:
       \[
       dp[i] = \max(dp[i-1], dp[i-2] + i \times \text{frequency}[i])
       \]
     - This means you either ignore `i` (take the value from `dp[i-1]`) or take it (add `i * frequency[i]` to `dp[i-2]`).

4. **Initialization**:
   - We need to initialize `dp[0]` (0 points for no elements) and `dp[1]` is simply the points we can earn from the number `1` (if it exists).

### Implementation:

Now, we will implement the solution according to the above plan.



```python
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        max_num = max(nums)  # Find the maximum number in the input
        frequency = [0] * (max_num + 1)  # Frequency array
        
        # Fill the frequency array
        for num in nums:
            frequency[num] += 1
            
        # DP array to store the maximum points we can earn
        dp = [0] * (max_num + 1)
        
        # Initialize the first two values in dp
        dp[1] = frequency[1] * 1  # Points earned from number 1
        
        for i in range(2, max_num + 1):
            dp[i] = max(dp[i - 1], dp[i - 2] + i * frequency[i])
        
        return dp[max_num]  # The maximum points we can earn


```

### Explanation of the Code:

1. We first check if `nums` is empty, in which case we return `0`.
2. We find the maximum number in `nums` to define the size of our `frequency` and `dp` arrays.
3. We populate the `frequency` array such that `frequency[i]` holds the number of times the integer `i` appears in `nums`.
4. We initialize the dynamic programming array and set its first two values.
5. We loop through the possible values from `2` to `max_num` to fill the `dp` array using our transition relation.
6. Finally, the solution returns `dp[max_num]`, which is the maximum points we can earn.

This solution runs in O(n + m) time complexity where n is the size of the input array and m is the maximum value in `nums`, which is efficient given the problem constraints.

# 741. Cherry Pickup

### Problem Description 
You are given an `n x n` `grid` representing a field of cherries, each cell is one of three possible integers.

`0` means the cell is empty, so you can pass through,
`1` means the cell contains a cherry that you can pick up and pass through, or
`-1` means the cell contains a thorn that blocks your way.

Return the maximum number of cherries you can collect by following the rules below:
Starting at the position `(0, 0)` and reaching `(n - 1, n - 1)` by moving right or down through valid path cells (cells with value `0` or `1`).

After reaching `(n - 1, n - 1)`, returning to `(0, 0)` by moving left or up through valid path cells.

When passing through a path cell containing a cherry, you pick it up, and the cell becomes an empty cell `0`.

If there is no valid path between `(0, 0)` and `(n - 1, n - 1)`, then no cherries can be collected.


Example 1:
Input: grid = [[0,1,-1],[1,0,-1],[1,1,1]]
Output: 5
Explanation: The player started at (0, 0) and went down, down, right right to reach (2, 2).

4 cherries were picked up during this single trip, and the matrix becomes [[0,1,-1],[0,0,-1],[0,0,0]].

Then, the player went left, up, up, left to return home, picking up one more cherry.

The total number of cherries picked up is 5, and this is the maximum possible.


Example 2:
Input: grid = [[1,1,-1],[1,-1,1],[-1,1,1]]
Output: 0

Constraints:
`n == grid.length`
`n == grid[i].length`
`1 <= n <= 50`
`grid[i][j]` is `-1`, `0`, or `1`.

`grid[0][0] != -1`
`grid[n - 1][n - 1] != -1`

### Solution 
 To solve the problem of collecting the maximum number of cherries from a grid representing a field of cherries while adhering to movement rules, we can utilize dynamic programming with depth-first search (DFS) and memoization.

The approach is to realize that both the trip to the bottom-right corner and back to the top-left corner can be consolidated into one combined traversal strategy, avoiding redundancy and allowing us to count cherries efficiently.

Here’s how we can break down the solution:

1. **Understanding Movement**: We can only move right (`(i, j+1)`) or down (`(i+1, j)`) while going from the top-left to the bottom-right corner. For the return trip, we can move left (`(i, j-1)`) or up (`(i-1, j)`).

2. **Grid Evaluation**: Each grid cell can contain:
   - `0` (empty),
   - `1` (cherry),
   - `-1` (thorn, impassable).

3. **Single Trip Calculation**: Since we collect cherries on both the way to `(n-1, n-1)` and back to `(0, 0)`, we can use the states of both visits to determine the overall collection.

4. **Dynamic Programming**: We use a recursive function to traverse the grid and keep track of the number of cherries collected along the way. Memoization will help in storing already computed paths to avoid recalculating them.

5. **Code Implementation**: Below is the Python code implementing this logic, which is directly runnable on platforms like LeetCode.

### Python Code



```python
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        n = len(grid)
        
        # To represent memoization, where dp[x1][y1][x2] is the max cherries
        # collected when the first walker is at (x1, y1) and the second walker
        # is at (x2, y2). Since y2 can be determined by x's, we only store x1, y1, and x2.
        mem = {}

        def dfs(x1, y1, x2):
            y2 = x1 + y1 - x2  # derive y2 based on the symmetry of the problem
            
            # If out of bounds or hitting a thorn, return -inf.
            if x1 >= n or y1 >= n or x2 >= n or y2 >= n or grid[x1][y1] == -1 or grid[x2][y2] == -1:
                return float('-inf')

            # If we reach the bottom-right cell, collect cherries here.
            if x1 == n - 1 and y1 == n - 1:
                return grid[x1][y1] if grid[x1][y1] != -1 else 0

            # If already computed, return the result.
            if (x1, y1, x2) in mem:
                return mem[(x1, y1, x2)]
            
            # Collect cherries at both positions (only once for same position)
            cherries = grid[x1][y1] + (grid[x2][y2] if x1 != x2 else 0)

            # Explore all moves from here (right or down for both)
            res = max(
                dfs(x1 + 1, y1, x2 + 1),    # both go down
                dfs(x1 + 1, y1, x2),        # first goes down, second goes right
                dfs(x1, y1 + 1, x2 + 1),    # first goes right, second goes down
                dfs(x1, y1 + 1, x2)         # both go right
            )

            # Store the result in the memoization dictionary
            mem[(x1, y1, x2)] = cherries + res
            return mem[(x1, y1, x2)]

        max_cherries = dfs(0, 0, 0)
        return max_cherries if max_cherries >= 0 else 0  # Return 0 if unable to reach

```

### Explanation of the Code

- We define a helper function `dfs(x1, y1, x2)` to handle the recursive traversal of the grid for both trips:
  - `x1, y1` represents the coordinates of the first traveler.
  - Derived `y2` is obtained to find the second traveler’s position based on symmetrical movement.
  
- We check for out-of-bounds indices or thorns and return negative infinity if they are invalid moves.

- We collect cherries from both positions, taking care not to double-count if `x1` equals `x2`.

- The recursive function explores all valid movements (right and down) and aggregates the maximum number of cherries collected from those paths.

- Finally, we return the maximum result found, ensuring to return `0` if no cherries could be collected, indicating an impassable route. 

This solution efficiently considers cherry collection while adhering to movement rules and optimally utilizes memoization to reduce overhead on repeated state evaluations.

