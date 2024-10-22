# 923. 3Sum With Multiplicity

### Problem Description 
Given an integer array `arr`, and an integer `target`, return the number of tuples `i, j, k` such that `i < j < k` and `arr[i] + arr[j] + arr[k] == target`.

As the answer can be very large, return it modulo `109 + 7`.


Example 1:
Input: arr = [1,1,2,2,3,3,4,4,5,5], target = 8
Output: 20
Explanation: 
Enumerating by the values (arr[i], arr[j], arr[k]):
(1, 2, 5) occurs 8 times;
(1, 3, 4) occurs 8 times;
(2, 2, 4) occurs 2 times;
(2, 3, 3) occurs 2 times.


Example 2:
Input: arr = [1,1,2,2,2,2], target = 5
Output: 12
Explanation: 
arr[i] = 1, arr[j] = arr[k] = 2 occurs 12 times:
We choose one 1 from [1,1] in 2 ways,
and two 2s from [2,2,2,2] in 6 ways.


Constraints:
`3 <= arr.length <= 3000`
`0 <= arr[i] <= 100`
`0 <= target <= 300`

### Solution 
 To solve the problem of counting the number of tuples `(i, j, k)` such that `i < j < k` and `arr[i] + arr[j] + arr[k] == target`, we can follow a systematic approach. The strategy involves leveraging combinatorial counting, given the constraints on the values in the `arr`.

### Steps to the Solution

1. **Count Frequencies**: Since the values in the array are bounded (from 0 to 100), we can maintain a count of how many times each value appears in the array.

2. **Iterate Over Pairs**: We can iterate over each possible pair of `(arr[j], arr[k])` (where `j < k`) and then calculate what value `arr[i]` must be such that `arr[i] + arr[j] + arr[k] = target`. This value can be computed as `arr[i] = target - (arr[j] + arr[k])`.

3. **Calculate Valid Tuples**: Depending on the indices `i`, `j`, and `k` and the counts of respective values, we compute how many valid tuples we can form:
   - If `arr[i]`, `arr[j]`, and `arr[k]` are all the same, we need to choose 3 different indices from the available count.
   - If `arr[i]` equals `arr[j]` but not `arr[k]`, we choose one from the count of `arr[i]` and two from the count of `arr[k]`.
   - If all are different, we choose one from each count.

4. **Return Result Modulo**: Since the answer could be large, we return the result modulo `10^9 + 7`.

### Implementation

Here’s the Python code for the solution:



```python
def threeSumMulti(arr, target):
    MOD = 10**9 + 7

    # Count occurrences of each number in arr
    count = [0] * 101
    for num in arr:
        count[num] += 1
        
    result = 0
    
    # Iterate through all possible pairs (arr[j], arr[k])
    for j in range(101):
        for k in range(j, 101):
            sum_jk = j + k
            arr_i = target - sum_jk
            
            if arr_i < 0 or arr_i > 100:
                continue  # Skip if arr[i] is out of bounds
            
            if j == k:
                # If j and k are the same
                if arr_i == j:
                    # i, j, k all the same
                    if count[j] >= 3:
                        result += (count[j] * (count[j] - 1) * (count[j] - 2)) // 6
                else:
                    # j == k but arr[i] is different
                    if count[j] >= 2 and count[arr_i] > 0:
                        result += (count[j] * (count[j] - 1) // 2) * count[arr_i]
            else:
                # j and k are different
                if arr_i < j or arr_i > k:  # arr[i] must be less than j or between j and k
                    result += count[j] * count[k] * count[arr_i] 
    
    return result % MOD

```

### Explanation of the Code:

- We initialize a `count` array to store how many times each number appears in `arr`.
- The nested loops iterate through all possible pairs of values `(j, k)`.
- For each pair, we find what value of `arr[i]` is needed to achieve the target sum.
- Depending on whether `arr[i]`, `arr[j]`, and `arr[k]` are the same or different, we calculate the number of valid index combinations using combinatorial counting methods.
- Finally, we return the result modulo `10^9 + 7`.

This solution efficiently counts the valid tuples with a time complexity that is manageable given the constraints, and it is structured to fit directly in the LeetCode environment.

# 925. Long Pressed Name

### Problem Description 
Your friend is typing his `name` into a keyboard. Sometimes, when typing a character `c`, the key might get long pressed, and the character will be typed 1 or more times.

You examine the `typed` characters of the keyboard. Return `True` if it is possible that it was your friends name, with some characters (possibly none) being long pressed.


Example 1:
Input: name = "alex", typed = "aaleex"
Output: true
Explanation: 'a' and 'e' in 'alex' were long pressed.


Example 2:
Input: name = "saeed", typed = "ssaaedd"
Output: false
Explanation: 'e' must have been pressed twice, but it wasn't in the typed output.


Example 3:
Input: name = "leelee", typed = "lleeelee"
Output: true

Example 4:
Input: name = "laiden", typed = "laiden"
Output: true
Explanation: It's not necessary to long press any character.


Constraints:
`1 <= name.length <= 1000`
`1 <= typed.length <= 1000`
`name` and `typed` contain only lowercase English letters.

### Solution 
 To solve the problem of determining if the `typed` string could be derived from the `name` string by allowing for long presses of characters, we can use a two-pointer technique. This approach will effectively allow us to traverse both strings simultaneously to confirm whether `typed` is a valid modification of `name`.

### Detailed Explanation:

1. **Initialization**: We start by defining two pointers, `i` and `j`, both set to 0. The pointer `i` will iterate over the `name` string, while `j` will iterate over the `typed` string.

2. **Iterating through both strings**: We use a loop that runs while `j` is less than the length of `typed`. Inside this loop:
   - If the characters at the current positions of `name` and `typed` match (i.e., `name[i] == typed[j]`), we increment both pointers.
   - If they don't match, we check if we are still within bounds of `typed` and see if the current character `typed[j]` is equal to `typed[j-1]`. If it is, we assume a long press and move the pointer `j` forward without incrementing `i`.
   - If the characters don’t match and the long press condition is not satisfied, we immediately return `False`.

3. **End Condition**: After the loop, if we have finished processing the `name` string (i.e., `i` points to the length of `name`), we then check if the remaining characters in `typed` can be attributed to long presses. This is done by ensuring that any remaining characters in `typed` must be the same as the last character checked.

4. **Return the Result**: If all checks are fulfilled, we return `True`, indicating that `typed` can be derived from `name`.

Here's the implementation in Python in a format that can be run directly on the LeetCode website:



```python
class Solution:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        i, j = 0, 0

        while j < len(typed):
            if i < len(name) and name[i] == typed[j]:
                i += 1
                j += 1
            elif j > 0 and typed[j] == typed[j - 1]:
                j += 1
            else:
                return False
            
        # After we finish checking typed, we should have gone through all of name
        return i == len(name)

# Example usage:
solution = Solution()
print(solution.isLongPressedName("alex", "aaleex"))  # Output: True
print(solution.isLongPressedName("saeed", "ssaaedd"))  # Output: False
print(solution.isLongPressedName("leelee", "lleeelee"))  # Output: True
print(solution.isLongPressedName("laiden", "laiden"))  # Output: True

```

### Complexity Analysis:

- Time Complexity: O(n + m), where n is the length of `name` and m is the length of `typed`. This is because we are potentially examining each character in both strings once.
- Space Complexity: O(1), as we are using only a fixed amount of space for pointers and no additional data structures.

This approach is efficient and adheres to the constraints provided in the problem statement.

# 930. Binary Subarrays With Sum

### Problem Description 
In an array `A` of `0`s and `1`s, how many non-empty subarrays have sum `S`?

Example 1:
Input: A = [1,0,1,0,1], S = 2
Output: 4
Explanation: 
The 4 subarrays are bolded below:
[1,0,1,0,1]
[1,0,1,0,1]
[1,0,1,0,1]
[1,0,1,0,1]
Note:
`A.length <= 30000`
`0 <= S <= A.length`
`A[i]` is either `0` or `1`.

### Solution 
 To tackle the problem of counting non-empty subarrays with a specified sum `S` in a binary array made up of `0`s and `1`s, we can utilize a hashmap (or dictionary in Python) to keep track of the cumulative sums as we iterate through the array. This approach allows us to efficiently determine how many subarrays sum up to `S` without needing to consider every possible subarray explicitly, which is critical given the constraint of `A.length` being up to 30,000.

## Detailed Explanation

### Key Concepts:
1. **Cumulative Sum**: As we traverse the array, we keep a running total (cumulative sum) of the elements we've seen so far.
2. **HashMap for Count**: We maintain a hash map that records the frequency of each cumulative sum we've encountered. This allows us to quickly check how many times a particular sum has occurred.
3. **Subarray Condition**: For any index `i`, if we have a cumulative sum `current_sum` at that index (which is the sum of the array from index `0` to `i`), we want to know how many times `current_sum - S` has appeared before index `i`, as this indicates there is a subarray summing to `S` ending at index `i`.

### Implementation Steps:
1. Initialize a hashmap to store cumulative sums and initialize it with the sum `0` occurring once (to account for subarrays starting from index `0`).
2. Traverse through the array, updating the cumulative sum at each step.
3. For each cumulative sum, check if `current_sum - S` exists in the hashmap. If it does, add its count to our result.
4. Update the hashmap with the current cumulative sum.

### Code Implementation:
Below is the Python code that follows the LeetCode solution format:



```python
class Solution:
    def numSubarraysWithSum(self, A: List[int], S: int) -> int:
        from collections import defaultdict
        
        # Dictionary to hold the frequency of cumulative sums
        count_map = defaultdict(int)
        count_map[0] = 1  # To handle the case where a subarray equals S from the start
        current_sum = 0
        result = 0

        for num in A:
            current_sum += num  # Update the cumulative sum
            
            # Check if there exists a prefix with sum (current_sum - S)
            if (current_sum - S) in count_map:
                result += count_map[current_sum - S]
            
            # Store the current_cumulative_sum occurrence
            count_map[current_sum] += 1
        
        return result

```

### Explanation of the Code:
- We define a class `Solution` and within it, a method `numSubarraysWithSum` which takes in `A` and `S` as parameters.
- We use `defaultdict` from the `collections` module to create a convenient structure for counting occurrences of cumulative sums.
- The variable `current_sum` keeps track of our cumulative total as we iterate through the array, while `result` is used to store the final count of valid subarrays.
- For each element in the array `A`, we update `current_sum`, check for the existence of `current_sum - S` in `count_map`, and update our result accordingly.
- Finally, we return the count of subarrays whose sum equals `S`.

### Time Complexity
The time complexity of the solution is O(n), where n is the length of the array `A`, because we make a single pass through the array.

The space complexity is also O(n) in the worst case due to the storage of cumulative sums in `count_map`. 

This efficient approach ensures that we can handle the problem within the given constraints.

# 948. Bag of Tokens

### Problem Description 
You have an initial power of `P`, an initial score of `0`, and a bag of `tokens` where `tokens[i]` is the value of the `ith` token (0-indexed).

Your goal is to maximize your total score by potentially playing each token in one of two ways:
If your current power is at least `tokens[i]`, you may play the `ith` token face up, losing `tokens[i]` power and gaining `1` score.

If your current score is at least `1`, you may play the `ith` token face down, gaining `tokens[i]` power and losing `1` score.

Each token may be played at most once and in any order. You do not have to play all the tokens.

Return the largest possible score you can achieve after playing any number of tokens.


Example 1:
Input: tokens = [100], P = 50
Output: 0
Explanation: Playing the only token in the bag is impossible because you either have too little power or too little score.


Example 2:
Input: tokens = [100,200], P = 150
Output: 1
Explanation: Play the 0th token (100) face up, your power becomes 50 and score becomes 1.

There is no need to play the 1st token since you cannot play it face up to add to your score.


Example 3:
Input: tokens = [100,200,300,400], P = 200
Output: 2
Explanation: Play the tokens in this order to get a score of 2:
1. Play the 0th token (100) face up, your power becomes 100 and score becomes 1.

2. Play the 3rd token (400) face down, your power becomes 500 and score becomes 0.

3. Play the 1st token (200) face up, your power becomes 300 and score becomes 1.

4. Play the 2nd token (300) face up, your power becomes 0 and score becomes 2.


Constraints:
`0 <= tokens.length <= 1000`
`0 <= tokens[i], P < 104`

### Solution 
 To solve the problem of maximizing the score with the given tokens while managing the available power `P`, we can approach it using a greedy algorithm with a two-pointer technique. 

### Problem Breakdown:

1. **Understanding Token Play**: 
   - A token can be played "face up" to gain a score point by spending power equal to the token's value.
   - Alternatively, if we have at least one score, we can play a token "face down" to gain power equal to the token's value at the cost of losing a score point.

2. **Sorting Tokens**: 
   - We should first sort the tokens. This arrangement allows us to easily identify the weakest tokens that can be played face up first.

3. **Using Two Pointers**: 
   - One pointer (`left`) starts at the beginning of the sorted list (weakest token).
   - Another pointer (`right`) starts at the end of the sorted list (strongest token).
   - This allows us to balance between gaining score and managing power efficiently.

4. **Greedy Process**: 
   - We attempt to play tokens face up as long as we have enough power to do so.
   - If we cannot play a token face up but have enough score, we can redeem a token face down for power.

### Code Implementation:

Here is the complete code implementing the above logic following the LeetCode solution format:



```python
class Solution:
    def bagOfTokensScore(self, tokens: List[int], P: int) -> int:
        # First, we sort the tokens
        tokens.sort()
        
        # Initialize the two pointers and the score
        left, right = 0, len(tokens) - 1
        score = 0
        max_score = 0
        
        # While we have tokens to consider
        while left <= right:
            # If we can afford to play the leftmost token face up
            if P >= tokens[left]:
                P -= tokens[left]  # Pay the cost of the token
                score += 1        # Gain a score
                max_score = max(max_score, score)  # Update the max score
                left += 1          # Move the left pointer to the right
            # If we cannot afford the leftmost token
            elif score > 0:
                P += tokens[right]  # Pay the cost of the rightmost token to gain power
                score -= 1          # Lose a score
                right -= 1          # Move the right pointer to the left
            else:
                break  # Break the loop if we can't afford any tokens face up and have no score to redeem
        
        return max_score

```

### Explanation of the Code:

1. **Sorting**: 
   - The tokens are sorted to allow efficient access to the weakest and strongest tokens.
  
2. **Two Pointers**:
   - `left` starts from the weakest (smallest value).
   - `right` starts from the strongest (largest value).
  
3. **While Loop**:
   - Inside the loop, we check if we can play the token at `left` position. If yes, we update power and score accordingly.
   - If not, we check if we can redeem from the token at `right` only if we have a score to lose.
  
4. **Updating Max Score**: 
   - After each successful face-up token play, we update the maximum score achieved.
  
5. **Return**:
   - Finally, we return `max_score`, which is the largest score achieved during the process.

This greedy approach ensures we make the best decisions at each step, leading to an optimized solution to the problem.

# 977. Squares of a Sorted Array

### Problem Description 
Given an integer array `nums` sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.


Example 1:
Input: nums = [-4,-1,0,3,10]
Output: [0,1,9,16,100]
Explanation: After squaring, the array becomes [16,1,0,9,100].

After sorting, it becomes [0,1,9,16,100].


Example 2:
Input: nums = [-7,-3,2,3,11]
Output: [4,9,9,49,121]

Constraints:
`1 <= nums.length <= 104`
`-104 <= nums[i] <= 104`
`nums` is sorted in non-decreasing order.

Follow up: Squaring each element and sorting the new array is very trivial, could you find an `O(n)` solution using a different approach?

### Solution 
 To solve the problem of returning an array of the squares of each number from a sorted integer array `nums` in non-decreasing order, we can utilize a two-pointer approach. This method is efficient and operates in `O(n)` time complexity, where `n` is the length of the input array. 

### Explanation:

1. **Understanding Squares**: When we square negative and positive numbers, the resulting values will range from largest (for extreme negative and positive values) to smallest (for numbers approaching zero).

2. **Using Two Pointers**: Since the input array is sorted in non-decreasing order, the smallest squares will either come from squaring the largest negative number or the largest positive number. 
   - We can maintain two pointers: one starting at the beginning of the array (pointing to the smallest value in terms of magnitude) and the other at the end (pointing to the largest value). 
   - We will compare the squares of the two pointed values and place the larger square in the result list from the back, moving the respective pointer inwards.

3. **Result Array**: We will create a result array of the same size as `nums` and fill it from back to front as we evaluate the squares from both ends of the `nums` array.

### Implementation:

Here is how the solution is structured in Python, adhering to the LeetCode solution format.



```python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        # Initialize two pointers
        left = 0
        right = len(nums) - 1
        
        # Result array to store squares
        result = [0] * len(nums)
        
        # Fill the result array from the back
        for i in range(len(nums) - 1, -1, -1):
            left_square = nums[left] ** 2
            right_square = nums[right] ** 2
            
            if left_square > right_square:
                result[i] = left_square
                left += 1
            else:
                result[i] = right_square
                right -= 1
        
        return result

```

### Detailed Steps:

1. **Initialization**: We set the `left` pointer at index `0` and the `right` pointer at the last index of the `nums` array.

2. **Looping from Back to Front**:
   - We use a loop that iterates backwards through the result array from the last index to the first.
   - Inside the loop, we calculate the squares of the values pointed by `left` and `right`.
   - We compare these two squares:
     - If the square at `left` is larger, we place it in the current position of the result array and increment the `left` pointer.
     - Otherwise, we place the square at `right` in the current position and decrement the `right` pointer.

3. **Returning the Result**: Once the loop is complete, we return the fully populated `result` array.

### Time Complexity:
- The time complexity of this solution is `O(n)` since we traverse the array only once.
- The space complexity is also `O(n)` due to the result array used for storing the squared values.

This efficient method provides us with a sorted array of squares without the need to sort the squares afterwards, therefore meeting the problem's constraints and follow-up efficiently.

# 986. Interval List Intersections

### Problem Description 
You are given two lists of closed intervals, `firstList` and `secondList`, where `firstList[i] = [starti, endi]` and `secondList[j] = [startj, endj]`. Each list of intervals is pairwise disjoint and in sorted order.

Return the intersection of these two interval lists.

A closed interval `[a, b]` (with `a < b`) denotes the set of real numbers `x` with `a <= x <= b`.

The intersection of two closed intervals is a set of real numbers that are either empty or represented as a closed interval. For example, the intersection of `[1, 3]` and `[2, 4]` is `[2, 3]`.


Example 1:
Input: firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]

Example 2:
Input: firstList = [[1,3],[5,9]], secondList = []
Output: []

Example 3:
Input: firstList = [], secondList = [[4,8],[10,12]]
Output: []

Example 4:
Input: firstList = [[1,7]], secondList = [[3,10]]
Output: [[3,7]]

Constraints:
`0 <= firstList.length, secondList.length <= 1000`
`firstList.length + secondList.length >= 1`
`0 <= starti < endi <= 109`
`endi < starti+1`
`0 <= startj < endj <= 109 `
`endj < startj+1`

### Solution 
 To solve the problem of finding the intersection of two lists of closed intervals, we will employ a two-pointer technique given that both lists are sorted and pairwise disjoint. The two-pointer technique allows us to efficiently traverse both lists of intervals in a linear fashion.

### Explanation:

1. **Initialization**: We will maintain two pointers, `i` for `firstList` and `j` for `secondList`. We also need a list `result` to store the intersections.

2. **Iterate Through Both Lists**: We will iterate through both lists with the help of our two pointers. The key is to determine if the current intervals overlap:
   - An interval from `firstList` represented as `[starti, endi]` intersects with an interval from `secondList` represented as `[startj, endj]` if:
     
```
     starti <= endj and startj <= endi
     
```
   - The intersection will be the interval:
     
```
     [max(starti, startj), min(endi, endj)]
     
```
   
3. **Move Pointers**: After checking for an intersection:
   - If the current interval from `firstList` ends before the current interval from `secondList` starts (i.e., `endi < startj`), we increment the pointer `i`.
   - Otherwise, if the current interval from `secondList` ends before the current interval from `firstList` starts (i.e., `endj < starti`), we increment the pointer `j`.
   - If they intersect, we add the intersection to the result and adjust both pointers.

4. **Continue Until One List Is Exhausted**: We loop until either pointer goes beyond its respective list.

5. **Return Results**: Finally, we return the `result` list which contains all the intersections.

### Python Code:

Here is the Python code following the LeetCode format:



```python
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        i, j = 0, 0
        result = []

        while i < len(firstList) and j < len(secondList):
            # Get the current intervals
            start1, end1 = firstList[i]
            start2, end2 = secondList[j]

            # Check for intersection
            if start1 <= end2 and start2 <= end1:
                # There's an intersection
                result.append([max(start1, start2), min(end1, end2)])

            # Move to the next interval
            if end1 < end2:
                i += 1  # Move from firstList
            else:
                j += 1  # Move from secondList

        return result

```

### Explanation of the Code:

- We define a class `Solution` and implement the method `intervalIntersection` that takes in two lists of intervals.
- We initialize two pointers `i` and `j` along with the `result` list.
- Inside the `while` loop, we check if the intervals intersect based on the conditions outlined above.
- Depending on which interval ends first, we increment the appropriate pointer.
- Finally, we return the `result` list containing all the intervals that represent the intersections.

This implementation runs in O(m + n) time complexity, where `m` and `n` are the lengths of `firstList` and `secondList`, respectively. Given the constraints, this is efficient and suitable for the problem at hand.

# 992. Subarrays with K Different Integers

### Problem Description 
Given an array `A` of positive integers, call a (contiguous, not necessarily distinct) subarray of `A` good if the number of different integers in that subarray is exactly `K`.

(For example, `[1,2,3,1,2]` has `3` different integers: `1`, `2`, and `3`.)
Return the number of good subarrays of `A`.


Example 1:
Input: A = [1,2,1,2,3], K = 2
Output: 7
Explanation: Subarrays formed with exactly 2 different integers: [1,2], [2,1], [1,2], [2,3], [1,2,1], [2,1,2], [1,2,1,2].


Example 2:
Input: A = [1,2,1,3,4], K = 3
Output: 3
Explanation: Subarrays formed with exactly 3 different integers: [1,2,1,3], [2,1,3], [1,3,4].

Note:
`1 <= A.length <= 20000`
`1 <= A[i] <= A.length`
`1 <= K <= A.length`

### Solution 
 To solve the problem of finding the number of contiguous subarrays with exactly `K` different integers, we can use the sliding window technique combined with a two-pointer approach. The main idea is to count the number of subarrays with at most `K` different integers and then use this count to compute the number of subarrays with exactly `K` different integers.

### Explanation

1. **Subarrays with at most K different integers**: We will create a helper function that calculates the number of subarrays with at most `K` distinct integers. This will be done using a hashmap to count occurrences of each integer and maintaining two pointers, `start` and `end`:
   - Increment the `end` pointer to expand the window and add the integer at `A[end]` to the hashmap.
   - If the hashmap size (number of distinct integers) exceeds `K`, increment the `start` pointer to shrink the window until we have at most `K` distinct integers.
   - Count the number of valid subarrays ending at `end` which would be `end - start + 1`.

2. **Number of subarrays with exactly K different integers**: Once we have the count of subarrays with at most `K` and at most `K-1` distinct integers, the count of subarrays with exactly `K` different integers can be derived as:
   \[
   \text{exactly K} = \text{at most K} - \text{at most K-1}
   \]

### Implementation

Here is the Python code that adheres to the LeetCode format:



```python
class Solution:
    def subarraysWithKDistinct(self, A: List[int], K: int) -> int:
        def atMostK(k):
            count = {}
            start = 0
            result = 0
            for end in range(len(A)):
                # Add current element to the map
                if A[end] in count:
                    count[A[end]] += 1
                else:
                    count[A[end]] = 1
                
                # If we've exceeded K distinct integers, shrink from the start
                while len(count) > k:
                    count[A[start]] -= 1
                    if count[A[start]] == 0:
                        del count[A[start]]
                    start += 1
                
                # Count the number of subarrays ending at 'end'
                result += end - start + 1
            
            return result

        # Total subarrays with exactly K distinct integers
        return atMostK(K) - atMostK(K - 1)

```

### Example Walkthrough

- **Example 1**: `A = [1, 2, 1, 2, 3], K = 2`
    - Call `atMostK(2)`: This counts valid subarrays with at most 2 distinct integers.
    - Call `atMostK(1)`: This counts valid subarrays with at most 1 distinct integer.
    - The difference gives the result of 7.

- **Example 2**: `A = [1, 2, 1, 3, 4], K = 3`
    - Similar two calls to find subarrays with at most 3 and at most 2 distinct integers, yielding the final count of 3.

### Complexity
- **Time Complexity**: O(N), where N is the length of the array. Each element is processed at most twice (once added and once removed from the window).
- **Space Complexity**: O(K), where K is the number of distinct integers stored in the hashmap.

This approach is efficient and works well within the constraints provided in the problem statement.

# 1004. Max Consecutive Ones III

### Problem Description 
Given an array `A` of 0s and 1s, we may change up to `K` values from 0 to 1.

Return the length of the longest (contiguous) subarray that contains only 1s. 

Example 1:
Input: A = [1,1,1,0,0,0,1,1,1,1,0], K = 2
Output: 6
Explanation: 
[1,1,1,0,0,1,1,1,1,1,1]
Bolded numbers were flipped from 0 to 1.  The longest subarray is underlined.


Example 2:
Input: A = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], K = 3
Output: 10
Explanation: 
[0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1]
Bolded numbers were flipped from 0 to 1.  The longest subarray is underlined.

Note:
`1 <= A.length <= 20000`
`0 <= K <= A.length`
`A[i]` is `0` or `1`

### Solution 
 To solve the problem of finding the length of the longest contiguous subarray consisting only of 1s after changing up to `K` 0s to 1s, we can use a sliding window approach. This technique is efficient and allows us to maintain a window of the current subarray while ensuring that the number of 0s we can flip does not exceed `K`.

### Explanation:

1. **Sliding Window Technique**:
   - We will maintain two pointers, `left` and `right`, which represent the current window of interest in the array.
   - The `right` pointer will extend to the right to include more elements, while the `left` pointer will be adjusted to ensure that we do not flip more than `K` zeroes.

2. **Counting 0s**:
   - We'll keep a count of the number of zeros in the current window. If the count exceeds `K`, we will move the `left` pointer to the right until we have at most `K` zeroes in the current window.

3. **Tracking the Maximum Length**:
   - At each step, while adjusting the window, we calculate the current length of the window and update the maximum length found.

4. **Time Complexity**:
   - This algorithm runs in O(n) time complexity since each pointer (left and right) traverses the list at most once.

### Implementation:

Here is the Python code that implements the above logic in the required LeetCode format:



```python
class Solution:
    def longestOnes(self, A: List[int], K: int) -> int:
        left = 0
        count_zeroes = 0
        max_length = 0
        
        for right in range(len(A)):
            if A[right] == 0:
                count_zeroes += 1
            
            # If the count of zeroes exceeds K, move the left pointer
            while count_zeroes > K:
                if A[left] == 0:
                    count_zeroes -= 1
                left += 1
            
            # Update max_length with the current valid window size
            max_length = max(max_length, right - left + 1)
        
        return max_length

```

### Explanation of the Code:

- We define a class `Solution` with a method `longestOnes`.
- The method takes two parameters: `A` (the array of 0s and 1s) and `K` (the maximum number of 0s we can flip).
- We initialize pointers (`left`, `right`) and a counter (`count_zeroes`) to keep track of the number of 0s in the current window.
- We iterate through the array using the `right` pointer. If we encounter a zero, we increment `count_zeroes`.
- We use a while loop to check if `count_zeroes` exceeds `K`. If it does, we move the `left` pointer to the right and decrement the count of 0s appropriately.
- We continuously update `max_length` with the width of the current valid window.
- Finally, we return `max_length`, which gives us the length of the longest contiguous subarray of 1s after flipping at most `K` 0s.

### Usage:

This solution can be submitted directly to LeetCode, and it should work efficiently for the provided constraints.

# 1093. Statistics from a Large Sample

### Problem Description 
You are given a large sample of integers in the range `[0, 255]`. Since the sample is so large, it is represented by an array `count` where `count[k]` is the number of times that `k` appears in the sample.

Calculate the following statistics:
`minimum`: The minimum element in the sample.

`maximum`: The maximum element in the sample.

`mean`: The average of the sample, calculated as the total sum of all elements divided by the total number of elements.

`median`:
	
If the sample has an odd number of elements, then the `median` is the middle element once the sample is sorted.

If the sample has an even number of elements, then the `median` is the average of the two middle elements once the sample is sorted.

`mode`: The number that appears the most in the sample. It is guaranteed to be unique.

Return the statistics of the sample as an array of floating-point numbers `[minimum, maximum, mean, median, mode]`. Answers within `10-5` of the actual answer will be accepted.


Example 1:
Input: count = [0,1,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Output: [1.00000,3.00000,2.37500,2.50000,3.00000]
Explanation: The sample represented by count is [1,2,2,2,3,3,3,3].

The minimum and maximum are 1 and 3 respectively.

The mean is (1+2+2+2+3+3+3+3) / 8 = 19 / 8 = 2.375.

Since the size of the sample is even, the median is the average of the two middle elements 2 and 3, which is 2.5.

The mode is 3 as it appears the most in the sample.


Example 2:
Input: count = [0,4,3,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Output: [1.00000,4.00000,2.18182,2.00000,1.00000]
Explanation: The sample represented by count is [1,1,1,1,2,2,2,3,3,4,4].

The minimum and maximum are 1 and 4 respectively.

The mean is (1+1+1+1+2+2+2+3+3+4+4) / 11 = 24 / 11 = 2.18181818... (for display purposes, the output shows the rounded number 2.18182).

Since the size of the sample is odd, the median is the middle element 2.

The mode is 1 as it appears the most in the sample.


Constraints:
`count.length == 256`
`0 <= count[i] <= 109`
`1 <= sum(count) <= 109`
The mode of the sample that `count` represents is unique.

### Solution 
 Certainly! Let's tackle the problem step by step. We'll implement the necessary calculations for `minimum`, `maximum`, `mean`, `median`, and `mode` based on the provided frequency counts of integers.

### Step-by-Step Breakdown

1. **Minimum and Maximum**:
   - The minimum value will be the smallest index `k` where `count[k]` is greater than 0.
   - The maximum value will be the largest index `k` where `count[k]` is greater than 0.

2. **Mean**:
   - The mean is calculated using the formula:
   \[
   \text{mean} = \frac{\text{total sum of all elements}}{\text{total number of elements}}
   \]
   - The total sum can be computed as `k * count[k]` for all `k`.
   - The total number of elements is the sum of all values in `count`.

3. **Median**:
   - To find the median, we will need the sorted array of elements.
   - The median depends on whether the total count of elements is odd or even: 
     - If odd, it's the middle element.
     - If even, it's the average of the two middle elements.

4. **Mode**:
   - The mode is the index `k` with the highest count value (the frequency). Since the mode is guaranteed to be unique, we can find it directly.

### Implementation

Now, let’s see how this can be translated into code in the LeetCode format:



```python
class Solution:
    def sampleStats(self, count: List[int]) -> List[float]:
        # Initialize variables for min, max, sum, and total number of elements
        min_val = None
        max_val = None
        total_sum = 0
        total_count = 0
        
        # Calculate min, max, total_sum and total_count
        for k in range(256):
            if count[k] > 0:
                # Set min and max
                if min_val is None:
                    min_val = k
                max_val = k
            
            # Update total_sum and total_count
            total_sum += k * count[k]
            total_count += count[k]
        
        # Calculate mean
        mean = total_sum / total_count
        
        # Prepare to calculate median
        mid = total_count // 2
        cumulative_count = 0
        median = 0
        
        # Calculate median
        for k in range(256):
            cumulative_count += count[k]
            if total_count % 2 == 1:  # Odd total_count
                if cumulative_count > mid:
                    median = k
                    break
            else:  # Even total_count
                if cumulative_count == mid:  # Handle right middle element
                    first_middle = k
                if cumulative_count > mid:  # Handle left middle element
                    second_middle = k
                    median = (first_middle + second_middle) / 2
                    break
        
        # Calculate mode
        mode = max(range(256), key=lambda k: count[k])

        return [float(min_val), float(max_val), float(mean), float(median), float(mode)]

```

### Explanation

- **Initialization**: We prepare variables for minimum, maximum, total sum, and total count.
- **First Loop**: We iterate through `count`, updating the minimum/maximum values, total sum, and total count based on the frequency.
- **Mean Calculation**: This is straightforward, simply dividing the total sum by the total count.
- **Median Calculation**: Another loop is used to determine the median based on whether the count is odd or even by checking the cumulative counts.
- **Mode Calculation**: We find the mode by identifying the index with the highest frequency using the `max()` function combined with a key that looks for counts.
  
This method ensures that all calculations adhere to the constraints and run efficiently given the input restrictions.

# 1099. Two Sum Less Than K

### Problem Description 
Given an array `nums` of integers and integer `k`, return the maximum `sum` such that there exists `i < j` with `nums[i] + nums[j] = sum` and `sum < k`. If no `i`, `j` exist satisfying this equation, return `-1`.


Example 1:
Input: nums = [34,23,1,24,75,33,54,8], k = 60
Output: 58
Explanation: We can use 34 and 24 to sum 58 which is less than 60.


Example 2:
Input: nums = [10,20,30], k = 15
Output: -1
Explanation: In this case it is not possible to get a pair sum less that 15.


Constraints:
`1 <= nums.length <= 100`
`1 <= nums[i] <= 1000`
`1 <= k <= 2000`

### Solution 
 To solve the problem of finding the maximum sum of two distinct elements in the list `nums` such that their sum is less than a given integer `k`, we can use a straightforward approach involving nested loops. This brute-force method is suitable given the constraints, allowing us to examine all possible pairs of elements in the array.

### Steps to Solve the Problem:

1. **Initialize a Variable for Maximum Sum**: Start with a variable to keep track of the maximum sum found, initialized to `-1`. This will also serve as a flag in case no valid pair is found.

2. **Nested Loops to Find Pairs**: Use two nested loops to examine all pairs `(i, j)` where `i < j`. In Python, you can achieve this by iterating over the list with two indices:
   - The outer loop will iterate from the beginning of the list to the second last element.
   - The inner loop will iterate from the next element of the current element in the outer loop to the end of the list.

3. **Calculate the Sum**: For each pair of indices, calculate their sum.

4. **Check Conditions**: If the calculated sum is less than `k`, check if it is greater than the current maximum sum. If it is, update the maximum sum.

5. **Return Result**: After processing all pairs, if the maximum sum is still `-1`, it indicates that no valid pairs were found, so return `-1`. Otherwise, return the maximum sum.

### Python Code Implementation

Here's how the above logic can be translated into Python code following the LeetCode format:



```python
class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        max_sum = -1  # Initialize max_sum to -1
        
        n = len(nums)
        # Iterate through all pairs of indices i and j
        for i in range(n):
            for j in range(i + 1, n):
                current_sum = nums[i] + nums[j]  # Calculate the sum of the pair
                if current_sum < k:  # Check if the sum is less than k
                    max_sum = max(max_sum, current_sum)  # Update the max_sum if it's higher
        
        return max_sum  # Return the result

```

### Explanation of the Code:

- **Class Definition**: We define a `Solution` class as is standard on LeetCode.
- **Method Definition**: The method `maxSum` takes a list of integers `nums` and an integer `k`.
- **Initialization**: We start by initializing `max_sum` to `-1` to indicate no valid pair has been found yet.
- **Loops**: We loop through each possible pair `(i, j)` by using two nested loops. The outer loop goes through the indices, and the inner loop starts from `i + 1` to ensure that `i < j`.
- **Condition Check**: For every pair, we check if the sum of the two numbers is less than `k`. If it is, we update `max_sum` to be the maximum of the current `max_sum` and the sum of the current pair.
- **Final Return**: At the end, we return the `max_sum`, which will be `-1` if no valid pairs were found.

This solution will run efficiently within the given constraints, as it checks at most \( \frac{n(n-1)}{2} \) pairs, where \( n \) is the length of the input list `nums`.

# 1213. Intersection of Three Sorted Arrays

### Problem Description 
Given three integer arrays `arr1`, `arr2` and `arr3` sorted in strictly increasing order, return a sorted array of only the integers that appeared in all three arrays.


Example 1:
Input: arr1 = [1,2,3,4,5], arr2 = [1,2,5,7,9], arr3 = [1,3,4,5,8]
Output: [1,5]
Explanation: Only 1 and 5 appeared in the three arrays.


Example 2:
Input: arr1 = [197,418,523,876,1356], arr2 = [501,880,1593,1710,1870], arr3 = [521,682,1337,1395,1764]
Output: []

Constraints:
`1 <= arr1.length, arr2.length, arr3.length <= 1000`
`1 <= arr1[i], arr2[i], arr3[i] <= 2000`

### Solution 
 To solve the problem of finding the common elements among three sorted arrays, we can utilize a three-pointer technique. Since all three arrays are sorted in strictly increasing order, we can efficiently traverse through each array simultaneously, adjusting our pointers based on the values we encounter.

Here’s a step-by-step breakdown of the approach:

1. **Initialization**: Start with three indices/pointers set to the beginning of each array (`i`, `j`, `k` set to 0 for `arr1`, `arr2`, and `arr3`, respectively). Also, create a result list to store the common elements.

2. **Traversing the Arrays**: Use a while loop that continues until any of the pointers exceeds the length of their respective array. Within the loop:
   - Compare the elements at the current indices of the three arrays:
     - If all three elements are the same, it indicates a common element. Append it to the result list and increment all three pointers.
     - If the element in `arr1` is less than the element in `arr2`, increment `i`, since `arr1` needs a larger element to match.
     - If the element in `arr2` is less than the element in `arr3`, increment `j`.
     - If the element in `arr3` is less than the element in `arr1`, increment `k`.
  
3. **Return the Result**: Once the loop concludes, return the result list which contains all common elements.

This method is efficient with a time complexity of O(n) where n is the length of the longest array since we traverse each array at most once.

Here's how you would implement this in Python for LeetCode:



```python
class Solution:
    def arraysIntersection(self, arr1: List[int], arr2: List[int], arr3: List[int]) -> List[int]:
        # Pointers for each of the three arrays
        i = j = k = 0
        result = []
        
        # Use a while loop to traverse through all three arrays
        while i < len(arr1) and j < len(arr2) and k < len(arr3):
            if arr1[i] == arr2[j] == arr3[k]:
                # Found a common element
                result.append(arr1[i])
                # Move all pointers forward
                i += 1
                j += 1
                k += 1
            else:
                # Move the pointer that points to the smallest value
                if arr1[i] < arr2[j]:
                    i += 1
                elif arr2[j] < arr3[k]:
                    j += 1
                else:
                    k += 1
        
        return result

```

### Explanation:
1. **Class and Method Definition**: Following LeetCode's format, we define a class `Solution` and a method `arraysIntersection` that takes three parameters, which are the three integer arrays.

2. **Pointer Initialization**: We initialize the pointers `i`, `j`, and `k` to zero, and `result` to collect the common elements.

3. **Loop**: The main while loop checks the current indices for the arrays and performs the necessary comparisons.

4. **Common Element Detection**: If all elements at the current pointers are equal, we record the common element and increment all pointers.

5. **Pointer Adjustment**: If they are not equal, we increment the pointer of the smallest element.

This implementation is efficient and directly follows the structured approach suitable for the LeetCode submission format.

# 1229. Meeting Scheduler

### Problem Description 
Given the availability time slots arrays `slots1` and `slots2` of two people and a meeting duration `duration`, return the earliest time slot that works for both of them and is of duration `duration`.

If there is no common time slot that satisfies the requirements, return an empty array.

The format of a time slot is an array of two elements `[start, end]` representing an inclusive time range from `start` to `end`.

It is guaranteed that no two availability slots of the same person intersect with each other. That is, for any two time slots `[start1, end1]` and `[start2, end2]` of the same person, either `start1 > end2` or `start2 > end1`.


Example 1:
Input: slots1 = [[10,50],[60,120],[140,210]], slots2 = [[0,15],[60,70]], duration = 8
Output: [60,68]

Example 2:
Input: slots1 = [[10,50],[60,120],[140,210]], slots2 = [[0,15],[60,70]], duration = 12
Output: []

Constraints:
`1 <= slots1.length, slots2.length <= 104`
`slots1[i].length, slots2[i].length == 2`
`slots1[i][0] < slots1[i][1]`
`slots2[i][0] < slots2[i][1]`
`0 <= slots1[i][j], slots2[i][j] <= 109`
`1 <= duration <= 106`

### Solution 
 To solve the problem of finding a common time slot where two people can meet for a specified duration, we can use a two-pointer technique. The main idea is to iterate through the available time slots of both people, comparing them to find overlapping intervals that can accommodate the requested duration.

Here's a step-by-step breakdown of the approach:

1. **Sort the Time Slots**: Although the problem guarantees that there are no overlaps within individual lists, we first sort both `slots1` and `slots2` to ensure we process the earliest slots first.

2. **Initialize Pointers**: We'll use two pointers, one for each list of slots. This will help us iterate through the slots simultaneously.

3. **Check Overlapping Slots**:
   - For each pair of slots from `slots1` and `slots2`, check if there is an overlap.
   - An overlap exists if the `start` of one slot is less than the `end` of the other, and vice versa.
   - If they overlap, calculate the maximum of their two start times and the minimum of their two end times. This will give us the start and end of the overlapping interval.

4. **Check for Duration**: If the duration fits within the overlapping time (i.e., the length of the overlapping interval is greater than or equal to the desired duration), we can return the earliest viable slot.

5. **Move Pointers**: Depending on the end times of the current slots, move the pointer that points to the earlier ending slot to continue searching for more potential slots.

6. **Return Result**: If no suitable slot is found by the end of the iterations, return an empty array.

Here’s the Python code implementing the above logic:



```python
from typing import List

class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        # Step 1: Sort the time slots
        slots1.sort()
        slots2.sort()
        
        # Step 2: Initialize pointers for both lists
        i, j = 0, 0
        
        # Step 3: Iterate through both lists
        while i < len(slots1) and j < len(slots2):
            start1, end1 = slots1[i]
            start2, end2 = slots2[j]
            
            # Step 4: Calculate the overlapping interval
            start_overlap = max(start1, start2)
            end_overlap = min(end1, end2)
            
            # Step 5: Check if the overlapping interval can accommodate the duration
            if end_overlap - start_overlap >= duration:
                return [start_overlap, start_overlap + duration]
            
            # Step 6: Move the pointer that points to the earlier end time
            if end1 < end2:
                i += 1
            else:
                j += 1
        
        # Step 7: Return an empty array if no suitable slot is found
        return []

```

### Explanation of the Code:

- **Sorting**: The `slots1.sort()` and `slots2.sort()` lines ensure the slots are in increasing order of their start times.
- **Two-Pointer Technique**: The while loop continues as long as there are slots left in both arrays. The `i` and `j` pointers track our current position in each slot list.
- **Overlapping Calculation**: We calculate where the two slots might overlap using `max(start1, start2)` for start and `min(end1, end2)` for end. This gives us the interval that is common between the two slots.
- **Duration Check**: If the length of the overlapping interval is greater than or equal to `duration`, we return that interval appropriately modified to fit the requested duration.
- **Pointer Update**: Finally, depending on which slot ends first, we increment the respective pointer. This ensures we always check the next potential overlapping slot efficiently.

This solution runs in O(n + m) time complexity where n and m are the lengths of `slots1` and `slots2`, making it efficient and suitable for the input size limits given in the problem.

# 1234. Replace the Substring for Balanced String

### Problem Description 
You are given a string containing only 4 kinds of characters `'Q',` `'W', 'E'` and `'R'`.

A string is said to be balanced if each of its characters appears `n/4` times where `n` is the length of the string.

Return the minimum length of the substring that can be replaced with any other string of the same length to make the original string `s` balanced.

Return 0 if the string is already balanced.


Example 1:
Input: s = "QWER"
Output: 0
Explanation: s is already balanced.


Example 2:
Input: s = "QQWE"
Output: 1
Explanation: We need to replace a 'Q' to 'R', so that "RQWE" (or "QRWE") is balanced.


Example 3:
Input: s = "QQQW"
Output: 2
Explanation: We can replace the first "QQ" to "ER". 

Example 4:
Input: s = "QQQQ"
Output: 3
Explanation: We can replace the last 3 'Q' to make s = "QWER".


Constraints:
`1 <= s.length <= 10^5`
`s.length` is a multiple of `4`
`s `contains only `'Q'`, `'W'`, `'E'` and `'R'`.

### Solution 
 To solve the problem of determining the minimum length of a substring that can be replaced to make the string balanced, we can extend the sliding window technique for this purpose. Here's a step-by-step explanation of the approach we will take to implement the solution.

### Step-by-Step Explanation

1. **Character Count Calculation**:
   For a string to be balanced, each character ('Q', 'W', 'E', 'R') must appear `n/4` times, where `n` is the length of the string. We will first determine the count of each character in the string.

2. **Determine Excess Characters**:
   Calculate how many characters of each type exceed the required count (`n/4`). This gives us an idea of how many characters we need to replace to achieve balance.

3. **Sliding Window for Minimum Substring Replacement**:
   Utilize a sliding window (or two-pointer) approach to find the smallest substring that contains all the excess characters. As the window is expanded with a right pointer, whenever the window satisfies the condition of containing sufficient excess characters, we will check the current window size and attempt to contract it with a left pointer to minimize it.

4. **Edge Cases**:
   Handle cases where the string is already balanced upfront to return 0 immediately.

### Python Code Implementation

Here’s the implementation that can run directly on the LeetCode platform following the specified format:



```python
class Solution:
    def balancedString(self, s: str) -> int:
        n = len(s)
        target_count = n // 4
        
        # Count the characters
        char_count = {
            'Q': 0,
            'W': 0,
            'E': 0,
            'R': 0
        }
        
        for char in s:
            char_count[char] += 1
        
        # Determine the excess characters
        excess = {
            'Q': max(0, char_count['Q'] - target_count),
            'W': max(0, char_count['W'] - target_count),
            'E': max(0, char_count['E'] - target_count),
            'R': max(0, char_count['R'] - target_count)
        }
        
        # If there are no excess characters, the string is already balanced
        if all(value == 0 for value in excess.values()):
            return 0
        
        # Now we apply the sliding window technique
        left = 0
        min_length = n
        
        for right in range(n):
            # Reduce the count for the right pointer
            char_count[s[right]] -= 1
            
            # Check if we still need replacements
            while all(char_count[char] <= target_count for char in "QWER"):
                # Update the minimum length
                min_length = min(min_length, right - left + 1)
                # Move left pointer to try and reduce the size
                char_count[s[left]] += 1
                left += 1

        return min_length

# Example test cases for validation
# solution = Solution()
# print(solution.balancedString("QWER"))   # Output: 0
# print(solution.balancedString("QQWE"))   # Output: 1
# print(solution.balancedString("QQQW"))   # Output: 2
# print(solution.balancedString("QQQQ"))   # Output: 3

```

### Explanation of Code:

1. **Character Counting**:
   We build a dictionary `char_count` to count occurrences of each letter.

2. **Excess Character Calculation**:
   We calculate `excess` for each character, determining how many need to be replaced if the count exceeds the target.

3. **Sliding Window Logic**:
   A right pointer expands the window to include new characters and decrements counts in the `char_count` dictionary. Once all character counts are at or below the target in the window, we minimize the length by moving the left pointer to try to find a smaller valid window.

4. **Return the Final Result**:
   After finding and checking all windows, the minimum length needed to make the string balanced is returned.

This implementation effectively provides an optimal solution with a time complexity of O(n), appropriate given the constraints.

# 1248. Count Number of Nice Subarrays

### Problem Description 
Given an array of integers `nums` and an integer `k`. A continuous subarray is called nice if there are `k` odd numbers on it.

Return the number of nice sub-arrays.


Example 1:
Input: nums = [1,1,2,1,1], k = 3
Output: 2
Explanation: The only sub-arrays with 3 odd numbers are [1,1,2,1] and [1,2,1,1].


Example 2:
Input: nums = [2,4,6], k = 1
Output: 0
Explanation: There is no odd numbers in the array.


Example 3:
Input: nums = [2,2,2,1,2,2,1,2,2,2], k = 2
Output: 16

Constraints:
`1 <= nums.length <= 50000`
`1 <= nums[i] <= 10^5`
`1 <= k <= nums.length`

### Solution 
 To solve the problem of counting the number of nice subarrays with exactly `k` odd numbers, we'll employ a strategy using the two-pointer technique combined with prefix sums. Here’s a detailed breakdown of the approach and the implementation in the LeetCode format.

### Approach:

1. **Odd Count Calculation**: First, we need to determine the odd numbers in the array. We can define a helper function that counts how many subarrays contain at most `k` odd numbers. This helps us leverage the relationship between "exactly k odd numbers" and "at most k odd numbers".

2. **At Most k Odd Numbers**: We can count the number of subarrays that contain at most `k` odd numbers using a sliding window (or two-pointer) approach:
   - Maintain a pointer for the current end of the subarray and another pointer to denote the start of the subarray.
   - Expand the end pointer and when the count of odd numbers exceeds `k`, move the start pointer until we are back to at most `k` odd numbers.
   - Count the valid subarrays formed during this process.

3. **Calculate Result**: The number of valid subarrays with exactly `k` odd numbers can be obtained by calculating:
   \[
   \text{count of subarrays with at most } k \text{ odd numbers} - \text{count of subarrays with at most } (k-1) \text{ odd numbers}
   \]

### Implementation:

Here’s the Python implementation in the LeetCode required format:



```python
class Solution:
    def atMostK(self, nums: List[int], k: int) -> int:
        count = 0
        left = 0
        odd_count = 0
        
        for right in range(len(nums)):
            # If the current number is odd, we increment the count of odd numbers
            if nums[right] % 2 == 1:
                odd_count += 1
            
            # Shrink the window from the left until we have at most k odd numbers
            while odd_count > k:
                if nums[left] % 2 == 1:
                    odd_count -= 1
                left += 1
            
            # Count the number of subarrays that ends at 'right'
            count += right - left + 1
        
        return count
    
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        # The number of subarrays with exactly k odd numbers is 
        # the number with at most k odd numbers minus the number with at most (k-1) odd numbers.
        return self.atMostK(nums, k) - self.atMostK(nums, k - 1)

# Example usage:
# solution = Solution()
# print(solution.numberOfSubarrays([1,1,2,1,1], 3))  # Output: 2
# print(solution.numberOfSubarrays([2,4,6], 1))      # Output: 0
# print(solution.numberOfSubarrays([2,2,2,1,2,2,1,2,2,2], 2))  # Output: 16

```

### Explanation of the Code:
1. **`atMostK` function**: This function counts subarrays with at most `k` odd numbers using the sliding window approach.
   - `count` keeps track of the total valid subarrays found.
   - A for loop iterates through the indices of `nums`, and we check if the current number is odd.
   - If we exceed `k` odd numbers, we increment the `left` pointer to shrink the window until we are back to at most `k` odd numbers while adjusting `odd_count`.
   - Each time we have a valid window (determined by the `left` pointer), we calculate the number of subarrays ending at `right`.

2. **`numberOfSubarrays` function**: This function computes the final result by subtracting the number of subarrays with at most `k-1` odd numbers from those with at most `k`.

### Complexity:
The algorithm runs in O(n) time, where n is the length of the input array `nums`, making it efficient enough to handle the upper constraint. The space complexity is O(1), as we use a fixed number of extra variables.

This implementation will run directly on the LeetCode platform and should solve the problem as required.

# 1570. Dot Product of Two Sparse Vectors

### Problem Description 
Given two sparse vectors, compute their dot product.

Implement class `SparseVector`:
`SparseVector(nums)` Initializes the object with the vector `nums`
`dotProduct(vec)` Compute the dot product between the instance of SparseVector and `vec`
A sparse vector is a vector that has mostly zero values, you should store the sparse vector efficiently and compute the dot product between two SparseVector.

Follow up: What if only one of the vectors is sparse?

Example 1:
Input: nums1 = [1,0,0,2,3], nums2 = [0,3,0,4,0]
Output: 8
Explanation: v1 = SparseVector(nums1) , v2 = SparseVector(nums2)
v1.dotProduct(v2) = 1*0 + 0*3 + 0*0 + 2*4 + 3*0 = 8

Example 2:
Input: nums1 = [0,1,0,0,0], nums2 = [0,0,0,0,2]
Output: 0
Explanation: v1 = SparseVector(nums1) , v2 = SparseVector(nums2)
v1.dotProduct(v2) = 0*0 + 1*0 + 0*0 + 0*0 + 0*2 = 0

Example 3:
Input: nums1 = [0,1,0,0,2,0,0], nums2 = [1,0,0,0,3,0,4]
Output: 6

Constraints:
`n == nums1.length == nums2.length`
`1 <= n <= 10^5`
`0 <= nums1[i], nums2[i] <= 100`

### Solution 
 To solve the problem of computing the dot product between two sparse vectors efficiently, we can use the following approach:

1. **Sparse Representation**: We will represent the sparse vector by storing only the non-zero elements along with their corresponding indices. This allows us to save space and time as we only compute the dot product for the non-zero components.

2. **Data Structure**: We'll use a dictionary (or a list of tuples) to store the non-zero values and their indices in the given vector.

3. **Dot Product Calculation**: The dot product is computed by multiplying corresponding indices of both vectors and summing them up. We can achieve this by iterating only through the non-zero entries of one vector and checking if the indices exist in the other vector.

Here is the Python implementation for the class `SparseVector` with the methods we need:



```python
class SparseVector:
    def __init__(self, nums):
        """
        Initialize the SparseVector with the given list of numbers.
        Store only the non-zero elements in a dictionary for efficient access.
        """
        self.non_zero_elements = {}
        for index, value in enumerate(nums):
            if value != 0:
                self.non_zero_elements[index] = value

    def dotProduct(self, vec):
        """
        Compute the dot product of this SparseVector and another SparseVector vec.
        Iterate over the non-zero elements of the current vector and calculate the dot product.
        """
        result = 0
        for index, value in self.non_zero_elements.items():
            if index in vec.non_zero_elements:
                result += value * vec.non_zero_elements[index]
        return result

```

### Explanation:
- **Initialization (`__init__` method)**:
  - We initialize the `SparseVector` with a list of numbers (`nums`).
  - We create a dictionary (`non_zero_elements`) to map indices to their corresponding non-zero values. This ensures we only store values that contribute to the dot product calculation.

- **Dot Product Calculation (`dotProduct` method)**:
  - We iterate over the items in the `non_zero_elements` dictionary.
  - For each index and value in the current sparse vector, we check if that index exists in the `non_zero_elements` of the input vector (`vec`).
  - If it does exist, we compute the product of the two values and add it to the result.
  - Finally, we return the computed dot product.

### Example Usage:
To use the `SparseVector` class and compute the dot product, you would execute the code as follows:



```python
# Example 1
nums1 = [1,0,0,2,3]
nums2 = [0,3,0,4,0]
v1 = SparseVector(nums1)
v2 = SparseVector(nums2)
print(v1.dotProduct(v2))  # Output: 8

# Example 2
nums1 = [0,1,0,0,0]
nums2 = [0,0,0,0,2]
v1 = SparseVector(nums1)
v2 = SparseVector(nums2)
print(v1.dotProduct(v2))  # Output: 0

# Example 3
nums1 = [0,1,0,0,2,0,0]
nums2 = [1,0,0,0,3,0,4]
v1 = SparseVector(nums1)
v2 = SparseVector(nums2)
print(v1.dotProduct(v2))  # Output: 6

```

This code should fit perfectly in the LeetCode environment as it adheres to object-oriented design principles while being efficient with space and time complexities suitable for the constraints provided.

# 1610. Maximum Number of Visible Points

### Problem Description 
You are given an array `points`, an integer `angle`, and your `location`, where `location = [posx, posy]` and `points[i] = [xi, yi]` both denote integral coordinates on the X-Y plane.

Initially, you are facing directly east from your position. You cannot move from your position, but you can rotate. In other words, `posx` and `posy` cannot be changed. Your field of view in degrees is represented by `angle`, determining how wide you can see from any given view direction. Let `d` be the amount in degrees that you rotate counterclockwise. Then, your field of view is the inclusive range of angles `[d - angle/2, d + angle/2]`.

Your browser does not support the video tag or this video format.

You can see some set of points if, for each point, the angle formed by the point, your position, and the immediate east direction from your position is in your field of view.

There can be multiple points at one coordinate. There may be points at your location, and you can always see these points regardless of your rotation. Points do not obstruct your vision to other points.

Return the maximum number of points you can see.


Example 1:
Input: points = [[2,1],[2,2],[3,3]], angle = 90, location = [1,1]
Output: 3
Explanation: The shaded region represents your field of view. All points can be made visible in your field of view, including [3,3] even though [2,2] is in front and in the same line of sight.


Example 2:
Input: points = [[2,1],[2,2],[3,4],[1,1]], angle = 90, location = [1,1]
Output: 4
Explanation: All points can be made visible in your field of view, including the one at your location.


Example 3:
Input: points = [[1,0],[2,1]], angle = 13, location = [1,1]
Output: 1
Explanation: You can only see one of the two points, as shown above.


Constraints:
`1 <= points.length <= 105`
`points[i].length == 2`
`location.length == 2`
`0 <= angle < 360`
`0 <= posx, posy, xi, yi <= 100`

### Solution 
 To solve the problem of determining the maximum number of points visible from a given location with a specified field of view (angle), we will approach it as follows:

1. **Understanding Angles**: We'll need to calculate the angle that each point makes with regard to the positive x-axis (east direction). This can be done using the `atan2` function that computes the angle based on the differences in y and x coordinates.

2. **Sorting Angles**: Once we have computed the angles for each point, we will sort them. This helps in efficiently calculating which points fall within the field of view.

3. **Using Two Pointers**: To efficiently find the maximum number of points that can be seen within a continuous range of angles (i.e., the field of view), we'll use a two-pointer technique. This allows us to efficiently count how many angles fall within the given limit of `angle`.

4. **Handling Wrap-around**: Since angles are circular (0 to 360 degrees), we will need to handle cases where the end of our field of view wraps around back to the beginning. We can do this by duplicating our angle list with angles increased by 360 degrees.

Let's implement the logic in Python:



```python
import math
from typing import List

class Solution:
    def visiblePoints(self, points: List[List[int]], angle: int, location: List[int]) -> int:
        # Step 1: Calculate angles of points relative to the location
        angles = []
        # Count points that are at the exact location (which are always visible)
        same_location_points = 0
        
        for point in points:
            if point == location:
                same_location_points += 1
            else:
                dx = point[0] - location[0]
                dy = point[1] - location[1]
                angle_rad = math.atan2(dy, dx) * (180 / math.pi)  # Convert from radians to degrees.
                angles.append((angle_rad + 360) % 360)  # Normalize the angle to [0, 360)

        # Step 2: Sort the angles
        angles.sort()
        
        # Step 3: Duplicate the angles to handle the wrap-around
        angles += [a + 360 for a in angles]
        
        # Step 4: Use two pointers to find the maximum visible points
        max_visible = 0
        angle_rad = angle  # The field of view in the sorted angle list
        j = 0
        
        for i in range(len(angles) // 2):
            # While the point at j is within the field of view of angle_rad from point i
            while j < len(angles) and angles[j] <= angles[i] + angle_rad:
                j += 1
            # Calculate how many angles are in the current view
            max_visible = max(max_visible, j - i)
        
        # Adding points that are at the exact same location
        return max_visible + same_location_points

# Example usage:
# sol = Solution()
# result1 = sol.visiblePoints([[2,1],[2,2],[3,3]], 90, [1,1])  # Output: 3
# result2 = sol.visiblePoints([[2,1],[2,2],[3,4],[1,1]], 90, [1,1])  # Output: 4
# result3 = sol.visiblePoints([[1,0],[2,1]], 13, [1,1])  # Output: 1

```

### Explanation of the Code:
1. **Input Processing**: We initialize an empty list for angles and a counter for points at the same location.
2. **Angle Calculation**: For each point that isn't at the location, we calculate the angle using `atan2` and normalize it to the range [0, 360] degrees.
3. **Sorting and Duplication**: We sort our list of angles and then duplicate it by adding 360 to each angle to take care of the circular nature of angles.
4. **Two Pointer Technique**: For each starting angle, we use a second pointer to find the maximum number of angles that fall within the specified angle of view.
5. **Final Count**: We return the maximum number of visible points and add the count of points located directly at the observer's position.

This solution efficiently handles the maximum constraints and runs in O(n log n) time due to sorting, which is efficient enough given the problem constraints.

# 1616. Split Two Strings to Make Palindrome

### Problem Description 
You are given two strings `a` and `b` of the same length. Choose an index and split both strings at the same index, splitting `a` into two strings: `aprefix` and `asuffix` where `a = aprefix + asuffix`, and splitting `b` into two strings: `bprefix` and `bsuffix` where `b = bprefix + bsuffix`. Check if `aprefix + bsuffix` or `bprefix + asuffix` forms a palindrome.

When you split a string `s` into `sprefix` and `ssuffix`, either `ssuffix` or `sprefix` is allowed to be empty. For example, if `s = "abc"`, then `"" + "abc"`, `"a" + "bc"`, `"ab" + "c"` , and `"abc" + ""` are valid splits.

Return `true` if it is possible to form a palindrome string, otherwise return `false`.

Notice that `x + y` denotes the concatenation of strings `x` and `y`.


Example 1:
Input: a = "x", b = "y"
Output: true
Explaination: If either a or b are palindromes the answer is true since you can split in the following way:
aprefix = "", asuffix = "x"
bprefix = "", bsuffix = "y"
Then, aprefix + bsuffix = "" + "y" = "y", which is a palindrome.


Example 2:
Input: a = "abdef", b = "fecab"
Output: true

Example 3:
Input: a = "ulacfd", b = "jizalu"
Output: true
Explaination: Split them at index 3:
aprefix = "ula", asuffix = "cfd"
bprefix = "jiz", bsuffix = "alu"
Then, aprefix + bsuffix = "ula" + "alu" = "ulaalu", which is a palindrome.


Example 4:
Input: a = "xbdef", b = "xecab"
Output: false

Constraints:
`1 <= a.length, b.length <= 105`
`a.length == b.length`
`a` and `b` consist of lowercase English letters

### Solution 
 Certainly! Let's walk through a Python solution to the problem described, formatted for direct use on LeetCode. 

### Problem Analysis

Given two strings `a` and `b` of equal length, we want to check if we can split the strings at some index such that concatenating the two halves from both strings forms a palindrome. Specifically, we can create two new combinations:
1. `aprefix + bsuffix`
2. `bprefix + asuffix`

A palindrome is a string that reads the same forward and backward. 

### Approach

1. **Palindrome Function**: We will create a utility function to check if a string is a palindrome.
2. **Iterate Over Possible Splits**: We will iterate over all possible split indices (from 0 to the length of the strings), and for each index:
   - Compute `aprefix` and `asuffix` for string `a`
   - Compute `bprefix` and `bsuffix` for string `b`
   - Check if `aprefix + bsuffix` or `bprefix + asuffix` forms a palindrome using our utility function.
3. **Return Result**: If any combination forms a palindrome, we return `True`; if no combinations do, we return `False`.

### Implementation

Here's the code following LeetCode standards:



```python
class Solution:
    def checkPalindromeFormation(self, a: str, b: str) -> bool:
        def is_palindrome(s: str) -> bool:
            return s == s[::-1]

        n = len(a)
        
        for i in range(n + 1):
            # Splitting at index i
            aprefix = a[:i]
            asuffix = a[i:]
            bprefix = b[:i]
            bsuffix = b[i:]
            
            # Check the combinations
            if is_palindrome(aprefix + bsuffix) or is_palindrome(bprefix + asuffix):
                return True
        
        return False

```

### Explanation of the Code

1. **Function Definition**: We define a class `Solution` with the method `checkPalindromeFormation` which takes two strings `a` and `b`.
  
2. **Palindrome Checking**: The helper function `is_palindrome` checks if a string `s` is a palindrome by comparing it with its reverse (`s[::-1]`).

3. **Iterating Over Possible Splits**: We loop from `0` to `n` (inclusive), covering all split points:
   - For each split index `i`, we derive:
     - `aprefix`: First part of `a` (from start to index `i`)
     - `asuffix`: Remaining part of `a` (from index `i` to end)
     - `bprefix`: First part of `b`
     - `bsuffix`: Remaining part of `b`
   
4. **Checking Combinations**: For each split, we check both combinations (`aprefix + bsuffix` and `bprefix + asuffix`). If either is a palindrome, we return `True`.

5. **Final Return**: If the loop completes without finding any palindrome combinations, we return `False`.

### Complexity Analysis

- The time complexity is O(n^2) in the worst case due to potentially checking palindromes n times for up to n characters in the worst case.
- The space complexity is O(1), besides the input storage, as we are not using additional data structures proportional to the size of input strings.

This solution should run efficiently within the constraints provided (up to 100,000 characters in the strings).

# 1658. Minimum Operations to Reduce X to Zero

### Problem Description 
You are given an integer array `nums` and an integer `x`. In one operation, you can either remove the leftmost or the rightmost element from the array `nums` and subtract its value from `x`. Note that this modifies the array for future operations.

Return the minimum number of operations to reduce `x` to exactly `0` if it's possible, otherwise, return `-1`.


Example 1:
Input: nums = [1,1,4,2,3], x = 5
Output: 2
Explanation: The optimal solution is to remove the last two elements to reduce x to zero.


Example 2:
Input: nums = [5,6,7,8,9], x = 4
Output: -1

Example 3:
Input: nums = [3,2,20,1,1,3], x = 10
Output: 5
Explanation: The optimal solution is to remove the last three elements and the first two elements (5 operations in total) to reduce x to zero.


Constraints:
`1 <= nums.length <= 105`
`1 <= nums[i] <= 104`
`1 <= x <= 109`

### Solution 
 To solve the problem of reducing `x` to zero using the least number of operations on the `nums` array, we can utilize a two-pointer technique along with a prefix sum approach to identify the longest subarray that sums to `total - x`, where `total` is the sum of all elements in `nums`. The idea is that if we can find a contiguous subarray that sums to this value, the elements we leave out (from both ends) will total exactly `x`.

Here's a clear outline of how to approach the solution:

1. **Calculate Total Sum**: Compute the total sum of the array `nums`.

2. **Determine Target Sum**: Set the target sum as `total - x`. If the target sum is less than zero, return `-1` because it is impossible to reduce `x`.

3. **Use Two-Pointer Technique**: We will use a sliding window (or two-pointer) approach to find the longest subarray that sums to `target`. The window will expand when we add elements from the right and contract when we exceed the target.

4. **Check for Feasibility**: If we find a valid subarray, the number of operations will be the total length of `nums` minus the length of that longest subarray.

5. **Return Result**: If a valid subarray is found, return the number of operations; if not, return `-1`.

Here is the Python code that implements this logic in a format suitable for LeetCode:



```python
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        total = sum(nums)
        target = total - x
        
        # If the target is negative, it's impossible to reach zero
        if target < 0:
            return -1
            
        max_len = -1
        curr_sum = 0
        left = 0
        
        # Using a sliding window to find the longest subarray with sum == target
        for right in range(len(nums)):
            curr_sum += nums[right]
            
            # If curr_sum exceeds target, move the left pointer to contract the window
            while curr_sum > target and left <= right:
                curr_sum -= nums[left]
                left += 1
            
            # If we found a valid subarray
            if curr_sum == target:
                max_len = max(max_len, right - left + 1)
        
        if max_len == -1:
            return -1
        
        # The minimum operations is the total elements minus the length of the found subarray
        return len(nums) - max_len

```

### Explanation of the Code:

1. **Class Definition**: The class `Solution` contains the method `minOperations`, which takes `nums` and `x` as inputs.

2. **Total Calculation**: We calculate the `total` sum of the list `nums` and determine the `target` sum needed.

3. **Initial Check**: If the `target` is negative, immediately return `-1`.

4. **Sliding Window**: 
   - Initialize `max_len`, which keeps track of the longest valid subarray length found.
   - Iterate over the `nums` array with a `right` pointer, adding the value of `nums[right]` to `curr_sum`.
   - If `curr_sum` exceeds `target`, increment the `left` pointer to reduce the window size and subtract the value of `nums[left]`.
   - If `curr_sum` matches `target`, check if this window is the longest found so far.

5. **Return Value**: Finally, if a valid subarray exists, return the number of operations calculated as `len(nums) - max_len`, otherwise return `-1`.

### Complexity:
- **Time Complexity**: O(n) where n is the length of `nums`, since each element is processed at most twice.
- **Space Complexity**: O(1) since we are using a constant amount of space for variables. 

This approach efficiently finds the solution within the constraints provided by the problem statement.

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
 To solve the problem of delivering boxes using a ship with constraints on maximum box count and maximum weight, we can approach it using a greedy algorithm, which takes advantage of priorities: we will always try to load as many boxes as possible without violating the constraints.

Here’s how we can break down the problem:

1. **Initialization**: We need to keep track of trips made, the current weight of boxes, and the count of boxes loaded onto the ship.
2. **Load Boxes**: For each box in the list, we attempt to load it onto the ship if it doesn’t exceed the `maxBoxes` and `maxWeight`.
3. **Delivery Trips**: As soon as we can't load an additional box, we make a delivery trip:
   - For the current set of loaded boxes, if we have to go to a port, it's considered 1 trip to deliver them and then 1 trip back to the storage (2 trips).
   - If the next box requires a delivery to the same port as a previously delivered box, we need to account for that by possibly reducing trips when multiple boxes are delivered to the same port in succession.
4. **Repeat Until All Boxes Delivered**: Continue the loading and delivering process until all boxes are delivered.

### Python Code

Here's the implementation of the above logic:



```python
class Solution:
    def boxDelivering(self, boxes: List[List[int]], portsCount: int, maxBoxes: int, maxWeight: int) -> int:
        trips = 0  # Count of trips made
        current_weight = 0
        current_boxes = 0
        deliveries = []  # This will store the ports we need to go to
        
        i = 0
        while i < len(boxes):
            # Reset current weight and box count for this trip
            current_weight = 0
            current_boxes = 0
            ports_to_visit = set()  # To track distinct ports for this load
            
            # Load boxes
            while i < len(boxes):
                port, weight = boxes[i]
                if current_boxes < maxBoxes and current_weight + weight <= maxWeight:
                    current_boxes += 1
                    current_weight += weight
                    ports_to_visit.add(port)  # Mark this port for delivery
                    i += 1  # Move to the next box
                else:
                    break
            
            if ports_to_visit:
                # Each distinct port requires one additional trip
                trips += 1  # One trip to deliver boxes
                trips += 1  # One trip back to storage
        
        return trips

# Example usage:
# sol = Solution()
# result = sol.boxDelivering([[1,1],[2,1],[1,1]], 2, 3, 3)
# print(result)  # Output: 4

```

### Explanation of the Code
- We loop through the boxes and keep track of the total weight and number of boxes loaded.
- For every box loaded, we add its delivery port to a `ports_to_visit` set. This way, even if we load multiple boxes for the same port, we only count that port once in our trip calculations.
- When we reach a loading constraint (either weight or box count), we count the trips required based on how many distinct ports we need to deliver to.
- The while loop continues until all boxes are processed and deliveries are planned accordingly.

### Complexity Analysis
- **Time Complexity**: O(n) where n is the length of the boxes list since we loop through all boxes at most twice.
- **Space Complexity**: O(1) additional space is used since we only use a few variables to track states (ignoring input and output storage). 

This solution should be efficient and suitable for the problem's constraints.

# 1695. Maximum Erasure Value

### Problem Description 
You are given an array of positive integers `nums` and want to erase a subarray containing unique elements. The score you get by erasing the subarray is equal to the sum of its elements.

Return the maximum score you can get by erasing exactly one subarray.

An array `b` is called to be a subarray of `a` if it forms a contiguous subsequence of `a`, that is, if it is equal to `a[l],a[l+1],...,a[r]` for some `(l,r)`.


Example 1:
Input: nums = [4,2,4,5,6]
Output: 17
Explanation: The optimal subarray here is [2,4,5,6].


Example 2:
Input: nums = [5,2,1,2,5,2,1,2,5]
Output: 8
Explanation: The optimal subarray here is [5,2,1] or [1,2,5].


Constraints:
`1 <= nums.length <= 105`
`1 <= nums[i] <= 104`

### Solution 
 To solve the problem of erasing a subarray of unique elements from a given array and maximizing the score, we can utilize a sliding window technique combined with a set to keep track of unique elements.

Here, we will maintain a window defined by two pointers: `left` and `right`. The idea is to expand the `right` pointer to include more elements in our subarray until we encounter a duplicate, at which point we'll move the `left` pointer to make the subarray valid again.

Here's a detailed explanation of the approach we'll implement:

1. **Initialize Variables**:
   - Use a `set` to keep track of the unique elements currently in our window.
   - Use a variable `current_sum` to track the sum of the elements in the current window.
   - Use a variable `max_score` to store the maximum score obtained by erasing a subarray.

2. **Expand the Window**:
   - Iterate through the array using the `right` pointer.
   - If the current element `nums[right]` is not in the set of unique elements, add it to the set and update the `current_sum`.

3. **Handle Duplicates**:
   - If the current element is already in the set (i.e., it would form a duplicate), we'll start moving the `left` pointer to the right until the window is valid again. During this process, remove elements from the set and update the `current_sum`.

4. **Calculate the Score**:
   - After ensuring the window only contains unique elements, calculate the score we would get by erasing the subarray from the start of the array up to `left` and from `right+1` to the end of the array. This can be represented mathematically as `total_sum - current_sum`.
   - Update the `max_score` if this score is greater than the previously recorded maximum.

5. **Return Results**:
   - After going through the entire array, return the `max_score`.

Here is the implementation of the above approach:



```python
class Solution:
    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        total_sum = sum(nums)  # Calculate the total sum of the array
        current_sum = 0        # Current sum of the subarray 
        max_score = 0          # Max score found so far
        left = 0               # Left pointer for the sliding window
        seen = set()           # Set to track unique elements in the current window
        
        for right in range(len(nums)):
            # If we encounter a duplicate, move the left pointer
            while nums[right] in seen:
                seen.remove(nums[left])  # Remove the leftmost element
                current_sum -= nums[left]
                left += 1
            
            # Add the current element to the window
            seen.add(nums[right])
            current_sum += nums[right]
            
            # Calculate the maximum score by erasing the subarray
            max_score = max(max_score, total_sum - current_sum)
        
        return max_score

```

### Explanation of the Code:
- We first calculate the `total_sum` of `nums`, which will be used to calculate the score after erasing the unique subarray.
- We loop through each element using the `right` pointer, expanding our subarray.
- For each element, we check if it’s already in the `seen` set. If it is, we keep moving the `left` pointer to the right until we remove the duplicate.
- Each time we add a new element, we also update `max_score`.
- Finally, we return the maximum score found.

This solution efficiently computes the maximum score in O(n) time complexity, making it suitable for the input size constraints provided.

# 1711. Count Good Meals

### Problem Description 
A good meal is a meal that contains exactly two different food items with a sum of deliciousness equal to a power of two.

You can pick any two different foods to make a good meal.

Given an array of integers `deliciousness` where `deliciousness[i]` is the deliciousness of the `i​​​​​​th​​​​`​​​​ item of food, return the number of different good meals you can make from this list modulo `109 + 7`.

Note that items with different indices are considered different even if they have the same deliciousness value.


Example 1:
Input: deliciousness = [1,3,5,7,9]
Output: 4
Explanation: The good meals are (1,3), (1,7), (3,5) and, (7,9).

Their respective sums are 4, 8, 8, and 16, all of which are powers of 2.


Example 2:
Input: deliciousness = [1,1,1,3,3,3,7]
Output: 15
Explanation: The good meals are (1,1) with 3 ways, (1,3) with 9 ways, and (1,7) with 3 ways.


Constraints:
`1 <= deliciousness.length <= 105`
`0 <= deliciousness[i] <= 220`

### Solution 
 To solve the problem of counting "good meals" based on the sum of two different food items equaling a power of two, we can follow a systematic approach. Below is the Python code formatted for the LeetCode website, along with a detailed explanation of the solution.

### Python Code



```python
class Solution:
    def countPairs(self, deliciousness: List[int]) -> int:
        from collections import Counter
        MOD = 10**9 + 7
        
        # Generate a set of all possible powers of 2 up to 2^21, since 2^20 is more than 2 * 10^20
        powers_of_two = {2 ** i for i in range(22)}
        
        # Count the frequency of each deliciousness value
        delicious_count = Counter(deliciousness)
        result = 0
        
        # Iterate through each unique deliciousness value
        for d in delicious_count:
            for power in powers_of_two:
                complement = power - d
                if complement in delicious_count:
                    if complement == d:  # Same item; we can't pick the same index twice
                        count = delicious_count[d]
                        # For same items, choose 2 out of count: C(count, 2) = count * (count - 1) / 2
                        result += count * (count - 1) // 2
                    else:
                        count1 = delicious_count[d]
                        count2 = delicious_count[complement]
                        result += count1 * count2
            
            # To avoid double counting, we ensure we only count pairs once.
            result %= MOD
            
        return result

```

### Detailed Explanation

1. **Understanding Powers of Two**: The first step is to determine which sums can qualify as "good meals". We calculate all powers of two up to \(2^{21}\) because a maximum sum using two deliciousness values under the problem constraints (maximum of \(2^{20}\)) will not exceed this limit. The powers of two are stored in a set for fast lookup.

2. **Counting Deliciousness Values**: We then use Python's `Counter` from the `collections` module to create a frequency map of the deliciousness values. This will allow us to efficiently reference how many items of a certain deliciousness there are.

3. **Iterating Over Unique Values**: The main loop iterates through each unique deliciousness value \(d\). For each \(d\), we check all powers of two.

4. **Finding Complements**: We calculate the complement of \(d\) needed to reach each power of two, i.e., `complement = power - d`. If this complement exists in our frequency map (`delicious_count`), we can form pairs.

5. **Counting the Pairs**:
   - If \(d\) is equal to the complement (where we have pairs of the same food item), we can select pairs using combinations. The mathematical formula `C(n, 2) = n * (n - 1) / 2` counts the number of ways to pick 2 items out of \(n\).
   - If \(d\) is different from the complement, we simply multiply the counts of \(d\) and `complement` to get the number of unique pairs.

6. **Avoiding Double Counting**: Note that when we count `(d, complement)` and `(complement, d)`, they should count only once. Since we go through each value only once with respect to its complement, we ensure no duplicates are counted.

7. **Modulo Operation**: Since the output must be returned modulo \(10^9 + 7\), we apply this at every step as necessary to prevent overflow.

8. **Return the Result**: Finally, we return the total count of good meals.

This code can be directly used in LeetCode, and will efficiently count the pairs fulfilling the criteria under the problem's constraints.

# 1712. Ways to Split Array Into Three Subarrays

### Problem Description 
A split of an integer array is good if:
The array is split into three non-empty contiguous subarrays - named `left`, `mid`, `right` respectively from left to right.

The sum of the elements in `left` is less than or equal to the sum of the elements in `mid`, and the sum of the elements in `mid` is less than or equal to the sum of the elements in `right`.

Given `nums`, an array of non-negative integers, return the number of good ways to split `nums`. As the number may be too large, return it modulo `109 + 7`.


Example 1:
Input: nums = [1,1,1]
Output: 1
Explanation: The only good way to split nums is [1] [1] [1].


Example 2:
Input: nums = [1,2,2,2,5,0]
Output: 3
Explanation: There are three good ways of splitting nums:
[1] [2] [2,2,5,0]
[1] [2,2] [2,5,0]
[1,2] [2,2] [5,0]

Example 3:
Input: nums = [3,2,1]
Output: 0
Explanation: There is no good way to split nums.


Constraints:
`3 <= nums.length <= 105`
`0 <= nums[i] <= 104`

### Solution 
 To solve the problem of counting good ways to split an integer array into three contiguous subarrays (`left`, `mid`, and `right`), we need to ensure that the following conditions are satisfied:

1. The sum of the `left` subarray is less than or equal to the sum of the `mid` subarray.
2. The sum of the `mid` subarray is less than or equal to the sum of the `right` subarray.

Given the constraints on the input size (`3 ≤ nums.length ≤ 10^5` and `0 ≤ nums[i] ≤ 10^4`), a brute force approach that checks every possible split would be inefficient. Instead, we can use a more efficient approach based on prefix sums.

### Steps to Solve the Problem

1. **Compute the Total Sum**: First, compute the total sum of the input array.

2. **Prefix Sums**: Create a prefix sum array where each element at index `i` represents the sum of the array elements from the start up to index `i`.

3. **Iterate Over Possible Splits**: Use two pointers to find valid positions for `mid` as we iterate over possible positions for `left`. As we do this, we maintain counts of valid splits that fulfill the required conditions.

4. **Count Valid Combinations**: For each valid partition of `left`, count how many valid partitions of `mid` exist such that they adhere to the conditions with respect to the corresponding partitions of `right`.

5. **Return the Count**: Finally, return the count modulo \(10^9 + 7\).

Here's how the implementation looks in Python:



```python
class Solution:
    def waysToSplit(self, nums):
        MOD = 10**9 + 7
        
        # Step 1: Calculate prefix sums
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + nums[i - 1]
        
        total_ways = 0
        
        # Step 2: Iterate through possible positions of left
        for i in range(1, n - 1):
            left_sum = prefix_sum[i] # Sum of left
            
            # Step 3: Use binary search to find boundaries for mid
            # We need mid_sum such that left_sum <= mid_sum <= right_sum
            
            # Move j to find a valid start for mid
            j_start = i + 1
            while j_start < n - 1 and prefix_sum[j_start] - left_sum < left_sum:
                j_start += 1
            
            # Move j_end to find a valid end for mid
            j_end = j_start
            while j_end < n - 1 and prefix_sum[j_end] - left_sum <= prefix_sum[n] - prefix_sum[j_end]:
                j_end += 1
            
            # Number of valid mid ranges is (j_end - j_start)
            total_ways += (j_end - j_start)
            total_ways %= MOD
        
        return total_ways

# Example Usage:
# sol = Solution()
# print(sol.waysToSplit([1, 1, 1]))  # Output: 1
# print(sol.waysToSplit([1, 2, 2, 2, 5, 0]))  # Output: 3
# print(sol.waysToSplit([3, 2, 1]))  # Output: 0

```

### Explanation of the Code
- **Prefix Sum Calculation**: A prefix sum array is created to enable quick sum calculations of any subarray.
- **Outer Loop**: The outer loop iterates through possible end indices for the `left` subarray.
- **Binary Search (Two Inner Loops)**: We utilize two inner loops to determine valid ranges for the `mid` subarray based on the sums derived from the prefix array:
  - `j_start`: This finds the first index where the sum of `mid` can be potentially valid given the `left` sum.
  - `j_end`: This determines the last index up to where a valid `mid` can stretch before exceeding the `right` sum.
- **Count Valid Partitions**: The difference `j_end - j_start` gives the number of valid partitions for `mid` based on the current `left`.

This implementation is efficient and runs within the constraints provided, adhering to a complexity of O(n log n) due to the prefix sum and searching processes, which is suitable for `n` up to \(10^5\).

# 1750. Minimum Length of String After Deleting Similar Ends

### Problem Description 
Given a string `s` consisting only of characters `'a'`, `'b'`, and `'c'`. You are asked to apply the following algorithm on the string any number of times:
Pick a non-empty prefix from the string `s` where all the characters in the prefix are equal.

Pick a non-empty suffix from the string `s` where all the characters in this suffix are equal.

The prefix and the suffix should not intersect at any index.

The characters from the prefix and suffix must be the same.

Delete both the prefix and the suffix.

Return the minimum length of `s` after performing the above operation any number of times (possibly zero times).


Example 1:
Input: s = "ca"
Output: 2
Explanation: You can't remove any characters, so the string stays as is.


Example 2:
Input: s = "cabaabac"
Output: 0
Explanation: An optimal sequence of operations is:
- Take prefix = "c" and suffix = "c" and remove them, s = "abaaba".

- Take prefix = "a" and suffix = "a" and remove them, s = "baab".

- Take prefix = "b" and suffix = "b" and remove them, s = "aa".

- Take prefix = "a" and suffix = "a" and remove them, s = "".


Example 3:
Input: s = "aabccabba"
Output: 3
Explanation: An optimal sequence of operations is:
- Take prefix = "aa" and suffix = "a" and remove them, s = "bccabb".

- Take prefix = "b" and suffix = "bb" and remove them, s = "cca".


Constraints:
`1 <= s.length <= 105`
`s` only consists of characters `'a'`, `'b'`, and `'c'`.

### Solution 
 To solve the problem of reducing a string by removing equal non-empty prefix and suffix that are made of the same character, we can apply a two-pointer technique to efficiently determine how many characters can be removed. 

### Algorithm Explanation

1. **Two Pointers Initialization**:
   - We will initiate two pointers (let’s call them `left` and `right`):
     - `left` starts from the beginning of the string (index 0).
     - `right` starts from the end of the string (index length of the string - 1).

2. **Prefix and Suffix Counting**:
   - Count how many characters are the same starting from the left. This will be the prefix count.
   - Similarly, count how many characters are the same starting from the right. This will be the suffix count.
   - Stop counting when the characters at `left` and `right` indices are different or when the pointers intersect.

3. **Condition to Remove**:
   - If the characters at `left` and `right` pointers are the same, it means we can remove that prefix and suffix. Move the pointers `left` and `right` inward to potentially remove more characters.
   - The new `left` pointer should move forward by the number of same characters counted as the prefix, and likewise for the `right` pointer.

4. **Termination**:
   - The process continues until the pointers meet or cross each other. The remaining characters in the string that can't be matched for removal will give us the answer.

5. **Compute Remaining Length**:
   - The final length of the string after all possible removals would be the difference between `right` and `left` pointers + 1 (because we are counting inclusively).

### Code Implementation

Here's how you can implement this:



```python
class Solution:
    def minimumLength(self, s: str) -> int:
        left = 0
        right = len(s) - 1
        
        while left < right:
            # Count characters from the start (left)
            char_left = s[left]
            while left < right and s[left] == char_left:
                left += 1
            
            # Count characters from the end (right)
            char_right = s[right]
            while left < right and s[right] == char_right:
                right -= 1
            
            # If we have a matching character at both ends
            if char_left == char_right:
                continue
            
            # If characters don't match, break the loop as we cannot remove anymore
            break
        
        return right - left + 1

```

### Explanation of the Code:
- We define a class `Solution` and a method `minimumLength` that takes a string `s`.
- We initialize `left` and `right` pointers. Using a while loop, we keep checking for characters from both ends.
- Inside the loop, we keep counting characters from both ends until they either become unequal or cross each other.
- If a matching character is found at both ends, we iteration continues; otherwise, it stops, as we've removed all possible characters.
- Finally, we return the remaining length of the string, which is `right - left + 1`.

### Complexity
- **Time Complexity**: O(n), where n is the length of string `s`, as we are making a single pass from both ends with the two pointers.
- **Space Complexity**: O(1), since we are using only a constant amount of extra space.

This solution is efficient and works within the constraints provided in the problem statement.

# 1800. Maximum Ascending Subarray Sum

### Problem Description 
Given an array of positive integers `nums`, return the maximum possible sum of an ascending subarray in `nums`.

A subarray is defined as a contiguous sequence of numbers in an array.

A subarray `[numsl, numsl+1, ..., numsr-1, numsr]` is ascending if for all `i` where `l <= i < r`, `numsi  < numsi+1`. Note that a subarray of size `1` is ascending.


Example 1:
Input: nums = [10,20,30,5,10,50]
Output: 65
Explanation: [5,10,50] is the ascending subarray with the maximum sum of 65.


Example 2:
Input: nums = [10,20,30,40,50]
Output: 150
Explanation: [10,20,30,40,50] is the ascending subarray with the maximum sum of 150.


Example 3:
Input: nums = [12,17,15,13,10,11,12]
Output: 33
Explanation: [10,11,12] is the ascending subarray with the maximum sum of 33.


Example 4:
Input: nums = [100,10,1]
Output: 100

Constraints:
`1 <= nums.length <= 100`
`1 <= nums[i] <= 100`

### Solution 
 Here's a Python solution to the problem of finding the maximum possible sum of an ascending subarray in a given array of positive integers. The solution will traverse the array while maintaining the sum of the current ascending subarray and updating the maximum sum whenever necessary.

### Python Code



```python
class Solution:
    def maxAscendingSum(self, nums: List[int]) -> int:
        max_sum = 0
        current_sum = 0
        
        for i in range(len(nums)):
            # If this is the start of the ascending subarray or a continuation
            if i == 0 or nums[i] > nums[i - 1]:
                current_sum += nums[i]  # Add current number to the current ascending sum
            else:
                # Reset current sum since the ascending order is broken
                current_sum = nums[i]
            max_sum = max(max_sum, current_sum)  # Update maximum sum if current is greater
        
        return max_sum

```

### Explanation

1. **Initialization**: We start by initializing two variables: `max_sum` to store the maximum sum found so far and `current_sum` to accumulate the sum of the current ascending subarray.

2. **Iterating Through the Array**:
   - We loop through each number in the `nums` array using a for loop.
   - For each element, we check whether it is part of an ascending sequence:
     - If it is the first element or if it is greater than the previous element (`nums[i] > nums[i - 1]`), it means the current number can extend the current ascending subarray, so we add it to `current_sum`.
     - If it is not greater, that indicates the end of the current ascending sequence. Thus, we reset `current_sum` to the current number since it's the start of a new potential ascending subarray.
   
3. **Updating the Maximum Sum**:
   - After each inclusion of a number in `current_sum`, we compare `current_sum` to `max_sum`. If `current_sum` is greater than `max_sum`, we update `max_sum`.

4. **Returning the Result**: Finally, after we finish processing all elements, we return the value of `max_sum`, which holds the highest sum of any ascending subarray.

### Example Walkthrough
For the input `nums = [10,20,30,5,10,50]`:
- Start with `max_sum = 0` and `current_sum = 0`.
- Process `10`: `current_sum = 10`, `max_sum = 10`.
- Process `20`: `current_sum = 30`, `max_sum = 30`.
- Process `30`: `current_sum = 60`, `max_sum = 60`.
- Process `5`: `current_sum = 5` (reset), `max_sum = 60`.
- Process `10`: `current_sum = 15`, `max_sum = 60`.
- Process `50`: `current_sum = 65`, `max_sum = 65`.

The output for this example will be `65`.

### Constraints Handling
This code efficiently handles the constraints of the problem since it requires only one pass through the list, resulting in a time complexity of O(n) and a space complexity of O(1), making it suitable for the input limits provided.

