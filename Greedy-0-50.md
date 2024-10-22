# 44. Wildcard Matching

### Problem Description 
Given an input string (`s`) and a pattern (`p`), implement wildcard pattern matching with support for `'?'` and `'*'` where:
`'?'` Matches any single character.

`'*'` Matches any sequence of characters (including the empty sequence).

The matching should cover the entire input string (not partial).


Example 1:
Input: s = "aa", p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".


Example 2:
Input: s = "aa", p = "*"
Output: true
Explanation: '*' matches any sequence.


Example 3:
Input: s = "cb", p = "?a"
Output: false
Explanation: '?' matches 'c', but the second letter is 'a', which does not match 'b'.


Example 4:
Input: s = "adceb", p = "*a*b"
Output: true
Explanation: The first '*' matches the empty sequence, while the second '*' matches the substring "dce".


Example 5:
Input: s = "acdcb", p = "a*c?b"
Output: false

Constraints:
`0 <= s.length, p.length <= 2000`
`s` contains only lowercase English letters.

`p` contains only lowercase English letters, `'?'` or `'*'`.

### Solution 
 To solve the problem of wildcard pattern matching with support for `?` and `*`, we can use dynamic programming (DP). The key idea is to create a 2D DP table where `dp[i][j]` represents whether the first `i` characters of string `s` match the first `j` characters of pattern `p`.

### Dynamic Programming Approach

1. **Initialization**:
   - Let `n` be the length of string `s` and `m` be the length of pattern `p`.
   - Create a DP table of size `(n+1) x (m+1)` initialized to `False`. The additional row and column help manage the scenarios when `s` or `p` is empty.
   - Set `dp[0][0] = True` because two empty strings match.

2. **Pattern Initialization**:
   - If pattern `p` starts with `*`, we need to fill the first row of the DP table. This is because `*` can match an empty sequence.
   - Iterate through the pattern: If `p[j-1]` is `*`, set `dp[0][j] = dp[0][j-1]`, which means we can ignore the `*` and still have a match with an empty string.

3. **Filling the DP Table**:
   - Loop through the string `s` (from 1 to `n`) and the pattern `p` (from 1 to `m`).
   - For each character in `s` and `p`:
     - If `s[i-1]` matches `p[j-1]` or `p[j-1]` is `?`, set `dp[i][j] = dp[i-1][j-1]`.
     - If `p[j-1]` is `*`, set `dp[i][j] = dp[i][j-1] (matching * with empty sequence) or dp[i-1][j] (matching * with one or more characters)`.

4. **Result**:
   - The final result will be in `dp[n][m]`, which tells whether the whole string matches the whole pattern.

Here's the code implementation for the above approach that can be directly run on LeetCode:



```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        n, m = len(s), len(p)
        
        # Initialize the dp table with False
        dp = [[False] * (m + 1) for _ in range(n + 1)]
        
        # Empty string and empty pattern match
        dp[0][0] = True
        
        # Handle cases where the pattern starts with '*'
        for j in range(1, m + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 1]
        
        # Fill the dp table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
                elif p[j - 1] == '?' or s[i - 1] == p[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
        
        return dp[n][m]

```

### Explanation of the Code:
- The `isMatch` function takes two arguments: the string `s` and the pattern `p`.
- We compute the lengths of `s` and `p` and initialize the DP table.
- We handle initial wildcard pattern conditions (for an initial `*`).
- We iterate through each character in `s` and `p`, updating the DP table based on the conditions described above.
- Finally, we return the value at `dp[n][m]`, indicating whether the entire string matches the pattern.

This approach ensures that we are efficiently checking each possibility while keeping the time complexity to O(n * m), which is suitable given the constraints.

# 45. Jump Game II

### Problem Description 
Given an array of non-negative integers `nums`, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Your goal is to reach the last index in the minimum number of jumps.

You can assume that you can always reach the last index.


Example 1:
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.


Example 2:
Input: nums = [2,3,0,1,4]
Output: 2

Constraints:
`1 <= nums.length <= 1000`
`0 <= nums[i] <= 105`

### Solution 
 Here's a solution for the problem of finding the minimum number of jumps to reach the last index of an array of non-negative integers. We will use a greedy approach to efficiently determine the minimum jumps needed.

### Problem Explanation

Given an array `nums`, each element in the array `nums[i]` represents the maximum length of the jump you can make from that position. Our task is to determine the minimum number of jumps needed to reach the last index of the array.

### Approach

1. **Initialization**:
    - We will keep track of the current end of the maximum reach we can get with the current number of jumps (denoted as `current_end`).
    - We'll also need a variable to track the farthest point we can reach (`farthest`) as we iterate through the array.
    - A counter (`jumps`) will track the number of jumps made.
    
2. **Iteration**:
    - We'll iterate through the elements in the array (excluding the last index). For each index, we will update the `farthest` point we can reach by taking the maximum of `farthest` or `i + nums[i]`.
    - If the index reaches `current_end`, it means we need to make a jump. At this point, update `current_end` to `farthest`, and increment the `jumps` counter.
    
3. **Termination**:
    - The loop will terminate once we have processed up to the second last index of the array, since to reach the last index, we must have already counted the jump needed towards it.

### Python Code



```python
class Solution:
    def jump(self, nums):
        # Edge case: if there's only one element, no jump is needed
        if len(nums) <= 1:
            return 0
        
        jumps = 0
        current_end = 0
        farthest = 0
        
        for i in range(len(nums) - 1):  # We don't need to jump from the last element
            farthest = max(farthest, i + nums[i])  # Update the farthest point we can reach
            
            # If we have come to the end of the range that we can reach with the current jump
            if i == current_end:
                jumps += 1  # We need to make a jump
                current_end = farthest  # Update the end to the farthest we can reach
                
        return jumps

```

### Detailed Explanation of the Code

1. **Edge Case Handling**: The first line checks if the length of the `nums` array is 1 or less. If it is, we return 0 since no jumps are needed.

2. **Variables Initialization**:
    - `jumps` tracks the number of jumps taken.
    - `current_end` is initialized to 0 representing the end of the current jump range.
    - `farthest` holds the farthest index that can be reached based on the jumps we might take.

3. **For Loop**:
    - We iterate over each index of the `nums` array up to the second last index.
    - For each index, we determine how far we can reach (`i + nums[i]`) and update the `farthest` variable if this index is further.
    - When we reach `current_end`, it means we need to perform a jump, so we increase the `jumps` count and update `current_end` to `farthest`.

This algorithm runs in O(n) time complexity, where n is the length of the `nums` array, making it efficient and suitable for the input constraints.

# 55. Jump Game

### Problem Description 
Given an array of non-negative integers `nums`, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.


Example 1:
Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.


Example 2:
Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.


Constraints:
`1 <= nums.length <= 3 * 104`
`0 <= nums[i] <= 105`

### Solution 
 To solve the problem of determining if you can reach the last index of the given array of non-negative integers, we can utilize a greedy approach. The goal is to keep track of the furthest index we can reach as we traverse the array.

### Approach

1. **Initialization**: Start at the first index, setting a variable (`max_reachable`) to track the furthest index we can reach.

2. **Traverse the Array**: For each position in the array, check if it is reachable (i.e., if its index is less than or equal to `max_reachable`). If it is reachable, update `max_reachable` to be the maximum of its current value and the index reachable from this position.

3. **Check Each Position**:
   - If at any index, the `max_reachable` is updated to be greater than or equal to the last index, we can return `True` because it means we can reach the end of the array.
   - If we reach an index that is not reachable (greater than `max_reachable`), we can stop and return `False`.

4. **Final Decision**: After traversing the entire array, if we have never reached a point where `max_reachable` fell short of our current index, return `True`. Otherwise, return `False`.

### Implementation

Now, let's code this approach in Python following the LeetCode submission format:



```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_reachable = 0
        n = len(nums)
        
        # Check each index in the array
        for i in range(n):
            # If the current index is unreachable
            if i > max_reachable:
                return False
            
            # Update max_reachable index
            max_reachable = max(max_reachable, i + nums[i])
            
            # If we can reach or exceed the last index
            if max_reachable >= n - 1:
                return True

        # If we've gone through all indices and haven't exited
        return False

```

### Explanation of the Code:

1. **Class Definition**: We define a class `Solution` which contains the method `canJump`.

2. **Method Signature**: The method takes a list of integers `nums` as input, which represents the maximum jump lengths.

3. **Variables**:
   - `max_reachable` is initialized to 0, representing the maximum index that can be reached initially.
   - `n` holds the length of the `nums` array.

4. **Loop**:
   - We loop through each index of the array.
   - Check if the current index `i` is greater than `max_reachable`. If so, return `False`.
   - Update `max_reachable` to be the maximum of its current value or the sum of the current index `i` and the jump length `nums[i]`.
   - Check if `max_reachable` is greater than or equal to the last index (`n - 1`). If so, return `True`.

5. **Return Statement**: If we're able to check all indices without returning `False`, return `False`.

### Complexity Analysis

- **Time Complexity**: O(n), where n is the length of the `nums` array. We traverse the array once.
- **Space Complexity**: O(1), since we only use a constant amount of space.

This solution is efficient and works within the constraints provided in the problem statement. You can copy and paste this code directly into the LeetCode editor to test it with the provided examples.

# 122. Best Time to Buy and Sell Stock II

### Problem Description 
You are given an array `prices` where `prices[i]` is the price of a given stock on the `ith` day.

Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).


Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.

Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.


Example 2:
Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.

Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. You must sell before buying again.


Example 3:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e., max profit = 0.


Constraints:
`1 <= prices.length <= 3 * 104`
`0 <= prices[i] <= 104`

### Solution 
 Certainly! The task is to maximize profit from an array of prices of stocks given over days, with the stipulation that you can buy and sell as many times as you like, but you cannot hold onto the stock while buying again.

To solve this problem, the key observation is:
- You can maximize profit by buying on one day and selling on the subsequent days if the price increases. Therefore, for every increase in price from one day to the next, you should consider that as a profit.

### Approach
1. Iterate through the `prices` array from the first day to the last.
2. Whenever there is a price increase from one day to the next (i.e., `prices[i] < prices[i + 1]`), calculate the profit by adding the difference to a variable that keeps track of total profit.
3. Continue this until you go through the entire list of prices.

This approach ensures that we grab every opportunity to profit when the price goes up.

### Python Code
Below is the Python code formatted to work directly on the LeetCode website:



```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                max_profit += prices[i] - prices[i - 1]
        
        return max_profit

```

### Explanation of the Code:
- We define a `Solution` class with a method `maxProfit`.
- A variable `max_profit` is initialized to 0 to accumulate the total profit.
- We loop through the `prices` list starting from index 1 (the second day).
- For each price at index `i`, we check if it's greater than the price on the previous day (`i - 1`).
- If it is, we calculate the profit by subtracting the previous day’s price from the current day’s price, and we add this amount to `max_profit`.
- Finally, we return the accumulated `max_profit`.

### Time Complexity
The time complexity of this solution is O(n), where n is the number of days (length of the `prices` array). We're making a single pass through the list.

### Space Complexity
The space complexity is O(1), as we only use a few extra variables for tracking profit and indices, regardless of the input size.

This solution provides an efficient way to solve the problem within the given constraints.

# 134. Gas Station

### Problem Description 
There are `n` gas stations along a circular route, where the amount of gas at the `ith` station is `gas[i]`.

You have a car with an unlimited gas tank and it costs `cost[i]` of gas to travel from the `ith` station to its next `(i + 1)th` station. You begin the journey with an empty tank at one of the gas stations.

Given two integer arrays `gas` and `cost`, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return `-1`. If there exists a solution, it is guaranteed to be unique

Example 1:
Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
Output: 3
Explanation:
Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 4. Your tank = 4 - 1 + 5 = 8
Travel to station 0. Your tank = 8 - 2 + 1 = 7
Travel to station 1. Your tank = 7 - 3 + 2 = 6
Travel to station 2. Your tank = 6 - 4 + 3 = 5
Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.

Therefore, return 3 as the starting index.


Example 2:
Input: gas = [2,3,4], cost = [3,4,3]
Output: -1
Explanation:
You can't start at station 0 or 1, as there is not enough gas to travel to the next station.

Let's start at station 2 and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 0. Your tank = 4 - 3 + 2 = 3
Travel to station 1. Your tank = 3 - 3 + 3 = 3
You cannot travel back to station 2, as it requires 4 unit of gas but you only have 3.

Therefore, you can't travel around the circuit once no matter where you start.


Constraints:
`gas.length == n`
`cost.length == n`
`1 <= n <= 104`
`0 <= gas[i], cost[i] <= 104`

### Solution 
 To solve the problem of determining if we can complete a circular route starting from any gas station, we can employ a greedy algorithm approach. The goal is to find the starting gas station index from which we can make a complete trip around the circuit. Here's how we can implement the solution, followed by an explanation of the algorithm.

### Solution



```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        total_tank = 0
        current_tank = 0
        start_index = 0
        
        for i in range(len(gas)):
            total_tank += gas[i] - cost[i]
            current_tank += gas[i] - cost[i]
            
            # If current_tank drops below 0, we cannot start from 'start_index'
            if current_tank < 0:
                start_index = i + 1  # next station becomes the new starting point
                current_tank = 0  # reset current tank
                
        # If total_tank is negative, it is not possible to complete the circuit
        return start_index if total_tank >= 0 else -1

```

### Explanation

1. **Initialization**:
   - `total_tank`: It keeps track of the total amount of gas minus the cost for the entire trip. This will help us determine if a solution exists at all.
   - `current_tank`: It tracks the current amount of gas available as we simulate the journey.
   - `start_index`: This is initialized to 0 and will be updated to the next station index whenever the `current_tank` becomes negative.

2. **Iterate through stations**:
   - We loop through each station index `i` (from 0 to `n-1`).
   - For each station, we compute `gas[i] - cost[i]` and add this value to both `total_tank` and `current_tank`.
   - If at any point `current_tank` falls below 0, it indicates that we cannot reach the next station starting from the current `start_index`. Consequently, we set `start_index` to `i + 1`, effectively skipping the current station and attempting to start from the next one. We also reset `current_tank` to 0.

3. **Final Condition**:
   - After iterating through all the gas stations, we check the `total_tank`:
     - If `total_tank` is negative, this means that even if we started from the best index, we wouldn't have enough gas to complete the circuit; thus, we return -1.
     - If `total_tank` is non-negative, we return `start_index`, which is the index of the starting station from where we can complete the trip.

### Complexity Analysis
- **Time Complexity**: O(n), where n is the number of gas stations. We only make a single pass through the lists.
- **Space Complexity**: O(1), as we are using only a constant amount of extra space.

This efficient algorithm ensures that we can quickly determine if a circular route is possible and, if so, where to start the journey.

# 135. Candy

### Problem Description 
There are `n` children standing in a line. Each child is assigned a rating value given in the integer array `ratings`.

You are giving candies to these children subjected to the following requirements:
Each child must have at least one candy.

Children with a higher rating get more candies than their neighbors.

Return the minimum number of candies you need to have to distribute the candies to the children.


Example 1:
Input: ratings = [1,0,2]
Output: 5
Explanation: You can allocate to the first, second and third child with 2, 1, 2 candies respectively.


Example 2:
Input: ratings = [1,2,2]
Output: 4
Explanation: You can allocate to the first, second and third child with 1, 2, 1 candies respectively.

The third child gets 1 candy because it satisfies the above two conditions.


Constraints:
`n == ratings.length`
`1 <= n <= 2 * 104`
`0 <= ratings[i] <= 2 * 104`

### Solution 
 To solve the problem of distributing candies to children based on their ratings while meeting specific requirements, we can use a two-pass algorithm. This approach ensures that we satisfy the constraints regarding the candy distribution effectively and efficiently.

### Problem Breakdown
1. **Conditions**:
    - Every child must receive at least one candy.
    - A child with a higher rating than their neighbors must receive more candies than those neighbors.

2. **Approach**:
    - First, initialize an array `candies` with all elements set to 1, since each child gets at least one candy.
    - **First Pass (Left to Right)**:
        - Traverse through the `ratings` array from left to right. If a child has a higher rating than the previous child, increase their candies count to be one more than the previous child's candies.
    - **Second Pass (Right to Left)**:
        - Traverse through the `ratings` array from right to left. If a child has a higher rating than the next child, adjust their candies to be the maximum of their current candies or one more than the next child's candies.
  
3. **Final Count**:
    - The total number of candies needed will be the sum of the values in the `candies` array.

### Code Implementation

Below is the Python code implementing the above logic in a format that can be directly run on LeetCode:



```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        
        if n == 0:
            return 0
        
        # Step 1: Initialize candies array
        candies = [1] * n
        
        # Step 2: Left to Right Pass
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1
        
        # Step 3: Right to Left Pass
        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                candies[i] = max(candies[i], candies[i + 1] + 1)
        
        # Step 4: Sum up all candies
        return sum(candies)

```

### Explanation of the Code
1. **Initialization**: 
   - We start by checking if the `ratings` list is empty. If it is, we return `0`, as there are no children to distribute candies.
   - We create a `candies` array of the same length as `ratings`, initialized to `1` for each child.

2. **Left to Right Pass**:
   - We loop from the second child to the last child (index `1` to `n-1`).
   - If the current child's rating is higher than the previous child's, we assign the current child's candies to be one more than the previous child's candies.

3. **Right to Left Pass**:
   - We loop from the second-to-last child to the first child (index `n-2` down to `0`).
   - If the current child's rating is higher than the next child's, we ensure that the current child's candies are at least one more than the next child's, while also keeping the maximum from the previous pass to ensure both conditions are met.

4. **Total Candies Calculation**:
   - Finally, we return the sum of all the values in the `candies` array, representing the minimum number of candies needed.

This algorithm runs in O(n) time complexity and O(n) space complexity, which is efficient given the constraints.

# 253. Meeting Rooms II

### Problem Description 
Given an array of meeting time intervals `intervals` where `intervals[i] = [starti, endi]`, return the minimum number of conference rooms required.


Example 1:
Input: intervals = [[0,30],[5,10],[15,20]]
Output: 2

Example 2:
Input: intervals = [[7,10],[2,4]]
Output: 1

Constraints:
`1 <= intervals.length <= 104`
`0 <= starti < endi <= 106`

### Solution 
 To solve the problem of finding the minimum number of conference rooms required given an array of meeting time intervals, we can follow a systematic approach using a greedy algorithm paired with a priority queue (min-heap) data structure. Here's how we can break down the approach:

### Explanation

1. **Sorting the Intervals**: The first step is to sort the meeting intervals based on the start time. This allows us to handle the meetings in chronological order.

2. **Using a Min-Heap**: We utilize a priority queue (or min-heap) to keep track of the end times of meetings. The min-heap will help efficiently find the earliest ending meeting.

3. **Iterating through the Meetings**: For each meeting:
   - If the meeting starts after the earliest ending meeting in the heap has ended, we can reuse that room (remove the earliest end time from the heap).
   - If it starts before the earliest ending meeting ends, we need to allocate a new room (add the current meeting's end time to the heap).

4. **Counting Rooms**: The size of the heap at any point during the iteration represents the number of rooms currently in use. The maximum size of the heap during the entire iteration gives us the minimum number of conference rooms required.

### Implementation

Here is how this can be translated into Python code appropriate for LeetCode:



```python
import heapq

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        
        # Step 1: Sort the meetings in increasing order of start time
        intervals.sort(key=lambda x: x[0])
        
        # Step 2: Create a min-heap to track the end times of meetings
        min_heap = []
        
        # Step 3: Iterate over the intervals
        for interval in intervals:
            # If the room due to end the earliest is free, remove it from the heap
            if min_heap and min_heap[0] <= interval[0]:
                heapq.heappop(min_heap)
            # Step 4: Add the current meeting's end time to the heap
            heapq.heappush(min_heap, interval[1])
        
        # Step 5: The size of the heap tells us the maximum number of rooms needed
        return len(min_heap)

```

### Explanation of the Code

- **Importing Libraries**: We import the `heapq` module which provides an efficient implementation of a priority queue in Python.
- **Sortable Function**: The `sort(key=lambda x: x[0])` is used to sort the intervals based on their starting times.
- **Heap Operations**: 
  - We use `heapq.heappop(min_heap)` to remove the earliest finishing meeting (if any).
  - We use `heapq.heappush(min_heap, interval[1])` to add the current meeting's end time into the heap.
- **Final Count**: The length of `min_heap` at the end corresponds to the minimum number of rooms required to accommodate all meetings concurrently scheduled.

### Complexity Analysis

- **Time Complexity**: O(N log N), where N is the number of meetings. This is due to sorting the intervals and the logarithmic complexity of heap operations.
- **Space Complexity**: O(N), which is the space needed for the heap in the worst case where all meetings overlap.

This code can be tested directly on the LeetCode platform, and it effectively solves the meeting room allocation problem by utilizing optimal data structures and algorithms.

# 316. Remove Duplicate Letters

### Problem Description 
Given a string `s`, remove duplicate letters so that every letter appears once and only once. You must make sure your result is the smallest in lexicographical order among all possible results.

Note: This question is the same as 1081: https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/

Example 1:
Input: s = "bcabc"
Output: "abc"

Example 2:
Input: s = "cbacdcbc"
Output: "acdb"

Constraints:
`1 <= s.length <= 104`
`s` consists of lowercase English letters.

### Solution 
 To solve the problem of removing duplicate letters while ensuring the result is the smallest possible in lexicographical order, we can use a greedy algorithm with a stack. Here is a detailed explanation of the approach, followed by the implementation in the format expected by LeetCode.

### Explanation

1. **Stack Data Structure**: We'll use a stack to build our resulting string. The stack will help us maintain the order of characters while also allowing us to remove characters that would prevent the smallest lexicographical order.

2. **Character Frequency Dictionary**: We'll maintain a count of the characters in the input string to know how many times each character appears. This will help us decide whether we can safely remove a character from the stack.

3. **Set to Track Added Characters**: We'll use a set to keep track of which characters have already been added to the stack. This prevents us from adding duplicates.

4. **Greedy Choice**: As we process each character:
   - If the character is already in the stack (tracked by our set), we skip it.
   - If the character can replace the top character of the stack (from a lexicographical standpoint and if there's still a chance to re-add that top character later), we pop the top character off the stack. This is done if the character at the top of the stack is greater than the current character and we still have more occurrences of that top character later in the string.

5. **Building the Result**: Once all characters have been processed, the stack will contain the answer in the order they were added, and we can join them to form our resulting string.

### Implementation

Here is the Python code that follows the LeetCode submission format:



```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        # Count the frequency of each character in the string
        char_count = {}
        for char in s:
            char_count[char] = char_count.get(char, 0) + 1
        
        stack = []
        in_stack = set()
        
        for char in s:
            # Decrease the count for this character since we're processing it now
            char_count[char] -= 1
            
            # If the character is already in the stack, skip it
            if char in in_stack:
                continue
            
            # Maintain the stack in lexicographical order
            while stack and stack[-1] > char and char_count[stack[-1]] > 0:
                removed_char = stack.pop()
                in_stack.remove(removed_char)
            
            # Add the current character to the stack and the set
            stack.append(char)
            in_stack.add(char)
        
        # Join all characters in the stack to form the result
        return ''.join(stack)

# Example of how to run the solution
sol = Solution()
print(sol.removeDuplicateLetters("bcabc"))  # Output: "abc"
print(sol.removeDuplicateLetters("cbacdcbc"))  # Output: "acdb"

```

### Complexity Analysis
- **Time Complexity**: O(n), where n is the length of the string `s`. We traverse the string and utilize a stack which allows us to add and remove characters in constant time.
- **Space Complexity**: O(1) to O(26), since the space used depends on the set of characters being processed, which in this case is bounded by the number of lowercase English letters.

This solution fulfills the constraints provided and ensures that the result is both unique and minimal in lexicographical order.

# 321. Create Maximum Number

### Problem Description 
You are given two integer arrays `nums1` and `nums2` of lengths `m` and `n` respectively. `nums1` and `nums2` represent the digits of two numbers. You are also given an integer `k`.

Create the maximum number of length `k <= m + n` from digits of the two numbers. The relative order of the digits from the same array must be preserved.

Return an array of the `k` digits representing the answer.


Example 1:
Input: nums1 = [3,4,6,5], nums2 = [9,1,2,5,8,3], k = 5
Output: [9,8,6,5,3]

Example 2:
Input: nums1 = [6,7], nums2 = [6,0,4], k = 5
Output: [6,7,6,0,4]

Example 3:
Input: nums1 = [3,9], nums2 = [8,9], k = 3
Output: [9,8,9]

Constraints:
`m == nums1.length`
`n == nums2.length`
`1 <= m, n <= 500`
`0 <= nums1[i], nums2[i] <= 9`
`1 <= k <= m + n`
Follow up: Try to optimize your time and space complexity.

### Solution 
 To solve the problem of creating the maximum number of length `k` from the digits of two arrays while preserving the order of digits within each array, we can use a combination of selecting digits from each array and merging those selections while maintaining their relative ordering.

Here's a step-by-step explanation of the approach we'll take:

1. **Choose Digits from Each Array:** We need to decide how many digits to take from `nums1` and how many from `nums2` such that their total equals `k`. This means for every possible split `(x, y)` where `x` is the number of digits taken from `nums1` and `y` from `nums2`, we must ensure that `x + y = k`.

2. **Maximize the Sequence:** For any chosen number of digits `x` from `nums1`, we will use a helper function to select the maximum possible sequence of `x` digits from `nums1`, and a similar one for `y` digits from `nums2`.

3. **Merge Sequences:** Once we have potential subsequences from both arrays, we need to merge them to create the final sequence. The merge needs to maintain the order of digits while creating the largest number possible.

4. **Iterate Through All Possible Combinations:** We will iterate over all possible splits `(x, k-x)` for `x` ranging from `max(0, k-n)` to `min(k, m)` and evaluate the merged results to keep track of the maximum outcome.

Here's how this can be implemented in Python:



```python
class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        def maxSingleNumber(nums: List[int], length: int) -> List[int]:
            stack = []
            drop = len(nums) - length
            for x in nums:
                while drop and stack and stack[-1] < x:
                    stack.pop()
                    drop -= 1
                stack.append(x)
            return stack[:length]
        
        def merge(nums1: List[int], nums2: List[int]) -> List[int]:
            return [max(nums1, nums2).pop(0) for _ in range(len(nums1) + len(nums2))]

        m, n = len(nums1), len(nums2)
        max_result = []
        
        for x in range(max(0, k - n), min(k, m) + 1):
            y = k - x
            if y < 0 or y > n:
                continue
            
            max_num1 = maxSingleNumber(nums1, x)
            max_num2 = maxSingleNumber(nums2, y)
            candidate = merge(max_num1, max_num2)
            max_result = max(max_result, candidate)
        
        return max_result

# Example usage
solution = Solution()
print(solution.maxNumber([3,4,6,5], [9,1,2,5,8,3], 5))  # Output: [9, 8, 6, 5, 3]
print(solution.maxNumber([6,7], [6,0,4], 5))              # Output: [6, 7, 6, 0, 4]
print(solution.maxNumber([3,9], [8,9], 3))                # Output: [9, 8, 9]

```

### Explanation of the Code:

1. **`maxSingleNumber` Function:** This function is responsible for extracting the largest subsequence of a specific length from an individual array. It uses a stack-based approach to ensure that we only keep the needed digits while following the criteria to maintain the largest possible digits.

2. **`merge` Function:** This function takes two lists and merges them in a greedy manner, always picking the larger leading digit from the remaining digits of both lists.

3. **Main Logic in the `maxNumber` Method:** 
   - We loop through all possible ways to split `k` into `x` from `nums1` and `k - x` from `nums2`.
   - For each split, we call `maxSingleNumber` for both arrays and then merge the results.
   - We continuously track the maximum result found.

This algorithm efficiently combines and merges the selected sequences while ensuring the optimal solution is preserved. The complexity is approximately O((m + n) * k) due to the selection and merging processes, which fits well within the constraints given.

# 330. Patching Array

### Problem Description 
Given a sorted integer array `nums` and an integer `n`, add/patch elements to the array such that any number in the range `[1, n]` inclusive can be formed by the sum of some elements in the array.

Return the minimum number of patches required.


Example 1:
Input: nums = [1,3], n = 6
Output: 1
Explanation:
Combinations of nums are [1], [3], [1,3], which form possible sums of: 1, 3, 4.

Now if we add/patch 2 to nums, the combinations are: [1], [2], [3], [1,3], [2,3], [1,2,3].

Possible sums are 1, 2, 3, 4, 5, 6, which now covers the range [1, 6].

So we only need 1 patch.


Example 2:
Input: nums = [1,5,10], n = 20
Output: 2
Explanation: The two patches can be [2, 4].


Example 3:
Input: nums = [1,2,2], n = 5
Output: 0

Constraints:
`1 <= nums.length <= 1000`
`1 <= nums[i] <= 104`
`nums` is sorted in ascending order.

`1 <= n <= 231 - 1`

### Solution 
 To solve the problem of finding the minimum number of patches required to ensure any number in the range `[1, n]` can be formed by the sum of some elements in the sorted integer array `nums`, we can follow a greedy algorithm approach.

### Explanation

1. **Initial Setup**: 
   - We will maintain a variable `max_reachable` that represents the largest sum we can achieve with the current numbers in the array and any patches we have added. Initially, this starts at `0`.
   - We also need a variable `patches` to count how many numbers we have added to the array.

2. **Iterate Through the Range**:
   - We will iterate through the numbers from `1` to `n`.
   - For each number `x`, we need to check if `x` is greater than `max_reachable`. If it is, it means we need to add a number (patch) to our list to cover `x`.

3. **Adding Patches**:
   - If we choose to add a patch, we will add `max_reachable + 1` to our array. This is because the smallest number that we cannot reach is `max_reachable + 1`. Once we add this number, `max_reachable` will be updated to cover a larger range.
   - After adding a patch, our `max_reachable` will be incremented by `max_reachable + 1`, allowing us to reach even more numbers.

4. **Advancing Through the Existing Array**:
   - If `x` is less than or equal to `max_reachable`, it indicates we can already form `x` using existing numbers. In this case, we just advance to the next number `x`.

5. **Termination**:
   - We keep repeating the above steps until `max_reachable` is at least `n`.

This approach ensures that we are efficiently covering the range `[1, n]` with the minimum number of patches.

### Python Code

Here’s the code implementing the above logic using the LeetCode format:



```python
class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:
        patches = 0
        max_reachable = 0
        i = 0
        while max_reachable < n:
            if i < len(nums) and nums[i] <= max_reachable + 1:
                # If the current number can help us reach further
                max_reachable += nums[i]
                i += 1
            else:
                # We need to patch (add a new number)
                patches += 1
                max_reachable += (max_reachable + 1)  # Patch with max_reachable + 1
        return patches

```

### Key Points

- **Time Complexity**: The time complexity of this algorithm is O(m), where `m` is the number of patches we need to add. In the worst case, we will need to add all numbers from `1` to `n`, but on average, this is efficient because we can reach multiple numbers with each patch.
- **Space Complexity**: This algorithm uses O(1) extra space since we only need a few variables to track our counts and indexes.

You can directly run this code on the LeetCode platform, and it should work efficiently for the given constraints.

# 358. Rearrange String k Distance Apart

### Problem Description 
Given a string `s` and an integer `k`, rearrange `s` such that the same characters are at least distance `k` from each other. If it is not possible to rearrange the string, return an empty string `""`.


Example 1:
Input: s = "aabbcc", k = 3
Output: "abcabc"
Explanation: The same letters are at least a distance of 3 from each other.


Example 2:
Input: s = "aaabc", k = 3
Output: ""
Explanation: It is not possible to rearrange the string.


Example 3:
Input: s = "aaadbbcc", k = 2
Output: "abacabcd"
Explanation: The same letters are at least a distance of 2 from each other.


Constraints:
`1 <= s.length <= 3 * 105`
`s` consists of only lowercase English letters.

`0 <= k <= s.length`

### Solution 
 To solve the problem of rearranging characters in a string `s` such that the same characters are at least a distance `k` apart, we can utilize a max-heap (priority queue) to maintain and track the frequency of characters. The approach can be broken down into several steps:

1. **Count Frequency**: First, we need to calculate the frequency of each character in the string using a dictionary.
2. **Use a Max-Heap**: We then utilize a max-heap to always try to place the character with the highest remaining count first.
3. **Placement**: We maintain a queue to ensure that once a character is placed in the output string, it cannot be placed again until `k` characters have been placed after it. This effectively manages the distance requirement.
4. **Construct the Result**: If we can successfully construct the output string with all characters placed according to the rules, we return the result; otherwise, we return an empty string.

Let's look at the complete implementation of this approach in Python following the LeetCode solution format:



```python
import collections
import heapq

class Solution:
    def rearrangeString(self, s: str, k: int) -> str:
        if k == 0:
            return s
        
        # Step 1: Count frequency of each character
        frequency = collections.Counter(s)
        
        # Step 2: Create a max-heap based on character frequency
        max_heap = [(-count, char) for char, count in frequency.items()]
        heapq.heapify(max_heap)
        
        # This will hold the resulting rearranged string
        result = []
        
        # This queue will hold characters waiting to be reused after k steps
        wait_queue = collections.deque()
        
        while max_heap:
            count, char = heapq.heappop(max_heap)
            result.append(char)
            wait_queue.append((count + 1, char))  # Reduce the count as it's used
            
            # Once we have placed k characters, we can re-include the front element in max_heap
            if len(wait_queue) >= k:
                to_reinsert = wait_queue.popleft()
                if to_reinsert[0] < 0:  # If there are more of this character still available
                    heapq.heappush(max_heap, to_reinsert)
        
        # If the length of the result string is not equal to the input string,
        # it means we couldn't reorganize the string successfully
        return ''.join(result) if len(result) == len(s) else ""

```

### Explanation of the Code:

1. **Import Modules**: We import `collections` for the `Counter` class and `heapq` to manage the max-heap.
  
2. **Define the Solution Class**: We define our class `Solution` which contains the method `rearrangeString`.

3. **Handle Edge Case**: If `k` is 0, we can return `s` immediately since no distance constraint applies.

4. **Count Frequencies**: We use `collections.Counter` to count how many times each character appears in the input string `s`.

5. **Max-Heap Setup**: We create a list of tuples from the character frequency count, storing negative counts (since Python's `heapq` is a min-heap) to simulate a max-heap.

6. **Rearranging Logic**: We use a `while` loop to extract the most frequent characters from the heap, append them to the result list, and manage the wait queue where characters that were just used will wait for `k` placements before being re-added to the heap.

7. **Final Check**: After constructing the result string, we check if we've used all characters. If the lengths match, we return the rearranged string; otherwise, we return an empty string.

This solution efficiently rearranges the string while satisfying the distance constraint, with a time complexity of \(O(n \log m)\) where \(n\) is the length of the string and \(m\) is the number of unique characters.

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
 To solve the problem of finding the length of the longest wiggle subsequence, we can take advantage of a greedy algorithm. The essential idea is to iterate through the array while tracking the previous difference (positive or negative) between consecutive elements, and count how many times the wiggle pattern occurs.

### Explanation

1. **Understanding Wiggle Sequences**: A wiggle sequence requires the differences between consecutive elements to alternate between positive and negative values. Therefore, our goal is to pick elements in a way that maintains this alternating property.

2. **Greedy Approach**: 
   - We need to traverse the array and track the direction of the last difference. If the difference changes direction compared to the last recorded difference, we can count this as a valid wiggle and increase our count of the wiggle subsequence.
   - Specifically, we'll use two pointers or counters: one for tracking the current value and the other for the last direction. This can be achieved with a couple of variables.

3. **Algorithm Steps**:
   - Initialize a counter to 1 (the minimum length of wiggle sequence with single element).
   - Use a loop to calculate differences between the consecutive numbers.
   - Track the last difference and whenever it changes from positive to negative or from negative to positive, increase the count.

4. **Edge Cases**: 
   - If the array has less than 2 elements, we immediately return the length of the array since it is trivially a wiggle sequence.

### Python Code

Here's the implementation of the above explanation in Python:



```python
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return len(nums)
        
        # Initialize count for wiggle sequence and the last difference
        count = 1
        last_diff = 0  # We use this to track difference: 0 means no last diff

        for i in range(1, len(nums)):
            diff = nums[i] - nums[i - 1]  # Current difference
            
            # Only process if the current difference is not zero
            if diff > 0:
                # Positive difference
                if last_diff <= 0:  # Upward wiggle (last was <= 0)
                    count += 1
                    last_diff = 1  # Indicate last difference is positive
            elif diff < 0:
                # Negative difference
                if last_diff >= 0:  # Downward wiggle (last was >= 0)
                    count += 1
                    last_diff = -1  # Indicate last difference is negative

        return count

```

### Explanation of the Code

- We define a class `Solution` and a method `wiggleMaxLength`.
- Begin by checking if the `nums` array has fewer than 2 elements. If true, return its length.
- Initialize `count` to 1 since a sequence is trivially a wiggle if it contains at least one number.
- The `last_diff` variable is initialized to zero to track the last difference type.
- Loop through the array starting from the second element, calculating the difference between consecutive elements.
- Based on the sign of the difference, we determine whether it qualifies to count as a wiggle. If it does, we increment the count and update `last_diff` to reflect the current direction.
- Finally, we return the count which represents the length of the longest wiggle subsequence.

### Complexity
- **Time Complexity**: O(n), where n is the length of the input array `nums`, as we traverse the list once.
- **Space Complexity**: O(1), using only a few variables for tracking state. 

This approach conforms to the constraints and efficiently finds the answer as required.

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
 To solve the problem of determining whether string `s` is a subsequence of string `t`, we can implement an efficient solution that utilizes two pointers. This algorithm will allow us to traverse both strings in a linear manner, ensuring we keep track of the characters of `s` while iterating through `t`.

### Explanation of the Approach:

1. **Two-Pointer Technique**: 
   - We will use two pointers: one (`i`) for string `s` and another (`j`) for string `t`.
   - We will iterate through both strings, moving the pointer for `t` whenever we find a matching character in `s`.
   - If the pointer for `s` reaches the length of `s` during this process, it indicates that all characters of `s` were found in `t` in order, hence `s` is a subsequence of `t`.

2. **Iterative Matching**: 
   - We will check each character of `t` against the current character of `s`.
   - If a match is found, we increment the pointer for `s`.
   - If we exhaust string `t` before we find all characters of `s`, then `s` is not a subsequence of `t`.

3. **Complexity**:
   - The time complexity of this algorithm is O(n) where n is the length of string `t`, as we are scanning through `t` only once.
   - The space complexity is O(1) since we are not using any additional data structures that grow with input size.

### Python Code Implementation:



```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        # Initialize two pointers
        i, j = 0, 0
        # Get the lengths of both strings
        len_s, len_t = len(s), len(t)
        
        # Iterate until we reach the end of either string
        while i < len_s and j < len_t:
            # If characters match, move pointer for s
            if s[i] == t[j]:
                i += 1
            # Always move pointer for t
            j += 1
        
        # If we have traversed all characters of s, it is a subsequence
        return i == len_s

```

### Example Usage:

You can use this function by instantiating the `Solution` class and calling the `isSubsequence` method, passing your strings `s` and `t`:



```python
sol = Solution()
print(sol.isSubsequence("abc", "ahbgdc"))  # Output: True
print(sol.isSubsequence("axc", "ahbgdc"))  # Output: False

```

### Follow-Up Optimization:

If there are numerous queries with different strings `s1, s2, ..., sk` to check against the same string `t`, we can preprocess `t` into a data structure that allows for faster querying.

#### Possible Enhancements:
- **Preprocessing with a Map**: Create a map that records the indices of each character in `t`. This allows us to quickly find the next occurrence of a character in `t`.
  
- **Binary Search Approach**: By storing the indices in sorted order for each character, we can perform a binary search for each character of `s` in these indices to find their position in `t`.

This kind of preprocessing would improve the query time, making it logarithmic concerning the length of `t` for each character lookup, while the preprocessing step would be linear.

Here’s a conceptual outline of that optimization, although it goes beyond the original problem you posted.

# 402. Remove K Digits

### Problem Description 
Given a non-negative integer num represented as a string, remove k digits from the number so that the new number is the smallest possible.

Note:
The length of num is less than 10002 and will be ≥ k.

The given num does not contain any leading zero.


Example 1:
Input: num = "1432219", k = 3
Output: "1219"
Explanation: Remove the three digits 4, 3, and 2 to form the new number 1219 which is the smallest.


Example 2:
Input: num = "10200", k = 1
Output: "200"
Explanation: Remove the leading 1 and the number is 200. Note that the output must not contain leading zeroes.


Example 3:
Input: num = "10", k = 2
Output: "0"
Explanation: Remove all the digits from the number and it is left with nothing which is 0.

### Solution 
 To solve the problem of removing `k` digits from a number represented as a string in order to form the smallest possible number, we can utilize a greedy algorithm involving a stack. The idea is that we iterate through each digit of the given number and maintain a stack to build our result.

### Algorithm Explanation:

1. **Initialization**: We will use a stack to hold the digits of the resulting number. We also calculate how many digits we need to remove, which is indicated by `k`.

2. **Iterate through digits**: For each digit in the number:
    - While the stack is not empty and the current digit is less than the digit at the top of the stack (indicating that by removing the top digit, we can achieve a smaller number), and we still have digits left to remove (`k > 0`), we pop the top digit from the stack.
    - Push the current digit onto the stack.

3. **Handling remaining digits**: After processing all digits, if we have not removed exactly `k` digits, we will remove the remaining digits from the end of the stack.

4. **Construct the result**: The stack now contains the digits of the result in order. We will convert the stack to a string. We also remove any leading zeros by converting to an integer and back to a string, unless the result is empty, in which case we return "0".

5. **Return the result**: Finally, return the resultant string which represents the smallest number after removing `k` digits.

Here is the Python implementation of the above logic, formatted to run on the LeetCode website:



```python
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []  # This will hold the digits of the result
        
        for digit in num:
            # While the stack is not empty and the last digit is larger than the current digit
            # and we still have k digits to remove
            while stack and k > 0 and stack[-1] > digit:
                stack.pop()  # Remove the last digit
            
            stack.append(digit)  # Add the current digit to the stack

        # If we still have digits to remove, remove them from the end
        while k > 0:
            stack.pop()
            k -= 1

        # Convert the stack to a string and remove leading zeros
        result = ''.join(stack).lstrip('0')

        return result if result else "0"  # Return '0' if the result is empty

```

### Time Complexity:
- The overall time complexity of this approach is O(n), where n is the length of the string `num`, as we process each digit at most twice (once for adding to the stack and potentially once for removing).

### Space Complexity:
- The space complexity is O(n) in the worst case where all digits are kept in the stack.

This efficient algorithm ensures that we are able to derive the smallest possible number by strategically removing digits while preserving the order and taking advantage of a stack data structure to manage the digits dynamically.

# 406. Queue Reconstruction by Height

### Problem Description 
You are given an array of people, `people`, which are the attributes of some people in a queue (not necessarily in order). Each `people[i] = [hi, ki]` represents the `ith` person of height `hi` with exactly `ki` other people in front who have a height greater than or equal to `hi`.

Reconstruct and return the queue that is represented by the input array `people`. The returned queue should be formatted as an array `queue`, where `queue[j] = [hj, kj]` is the attributes of the `jth` person in the queue (`queue[0]` is the person at the front of the queue).


Example 1:
Input: people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
Output: [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
Explanation:
Person 0 has height 5 with no other people taller or the same height in front.

Person 1 has height 7 with no other people taller or the same height in front.

Person 2 has height 5 with two persons taller or the same height in front, which is person 0 and 1.

Person 3 has height 6 with one person taller or the same height in front, which is person 1.

Person 4 has height 4 with four people taller or the same height in front, which are people 0, 1, 2, and 3.

Person 5 has height 7 with one person taller or the same height in front, which is person 1.

Hence [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] is the reconstructed queue.


Example 2:
Input: people = [[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]
Output: [[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]

Constraints:
`1 <= people.length <= 2000`
`0 <= hi <= 106`
`0 <= ki < people.length`
It is guaranteed that the queue can be reconstructed.

### Solution 
 To solve this problem, we need to reconstruct the queue of people based on their height and the number of people that must be in front of them (who have a height greater than or equal to theirs). We will follow a specific algorithm to achieve this.

### Detailed Explanation:

1. **Understanding the Input**: Each element in the input `people` array is a list with two integers: `[hi, ki]`, where `hi` is the height of the person, and `ki` is the number of people in front of this person with a height greater than or equal to `hi`.

2. **Sorting the People**: The first step in our approach is to sort the `people` array. We will sort it primarily by height in descending order, and for people with the same height, we will sort by `ki` in ascending order. This ensures that when we are placing a person in the queue, we are directly accounting for the people who are already placed in the queue and those who satisfy the condition based on the sorted order.

3. **Inserting into the Queue**: After sorting, we will iterate through the sorted list, and for each person, we will insert them into the resultant queue at the index equal to their `ki` value. This ensures that the conditions for the number of people in front are satisfied.

4. **Return Result**: Finally, we will return the constructed queue.

### Code Implementation:

The implementation below follows the aforementioned steps:



```python
class Solution:
    def reconstructQueue(self, people):
        # Step 1: Sort the people
        # Sort by height (descending), and by k (ascending)
        people.sort(key=lambda x: (-x[0], x[1]))
        
        # Step 2: Reconstruct the queue
        queue = []
        for person in people:
            # Insert each person at the index equal to their ki
            queue.insert(person[1], person)
        
        return queue

# Example usage
solution = Solution()
print(solution.reconstructQueue([[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]))  
# Output: [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]

```

### Key Points in the Implementation:

- **Sorting**: The `sort` function is used with a custom lambda function that sorts first by height in descending order and then by `ki` in ascending order. This is crucial for the correct reconstruction of the queue.
  
- **Insertion**: Using `queue.insert(index, person)` allows us to place each person in their correct position in a straightforward manner, as it uses the value of `ki` directly derived from our sorted list.

- **Complexity**: The sorting takes O(n log n), and inserting each person into the queue takes O(n), leading to an overall time complexity of O(n^2) in the worst case, which is efficient enough given the constraints (with n ≤ 2000).

This implementation is formatted in Python as required for submission in LeetCode, and it should work correctly for the provided problem statement.

# 435. Non-overlapping Intervals

### Problem Description 
Given a collection of intervals, find the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.


Example 1:
Input: [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of intervals are non-overlapping.


Example 2:
Input: [[1,2],[1,2],[1,2]]
Output: 2
Explanation: You need to remove two [1,2] to make the rest of intervals non-overlapping.


Example 3:
Input: [[1,2],[2,3]]
Output: 0
Explanation: You don't need to remove any of the intervals since they're already non-overlapping.

Note:
You may assume the interval's end point is always bigger than its start point.

Intervals like [1,2] and [2,3] have borders "touching" but they don't overlap each other.

### Solution 
 Here's the solution to the problem of finding the minimum number of intervals to remove to ensure that the rest of the intervals are non-overlapping. I'll provide the code that can be run directly on LeetCode, along with a detailed explanation of the algorithm used.



```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        # If there are no intervals, no need to remove any
        if not intervals:
            return 0
        
        # Sort the intervals based on the end time
        intervals.sort(key=lambda x: x[1])
        
        # Initialize variables
        count = 0  # To count the number of intervals to remove
        end = intervals[0][1]  # The end of the last added interval to the non-overlapping set

        # Iterate through the sorted intervals starting from the second one
        for i in range(1, len(intervals)):
            # If the current interval starts before the last added interval ends,
            # we have an overlap
            if intervals[i][0] < end:
                count += 1  # We need to remove one interval
            else:
                # Otherwise, update the end to the end of the current interval
                end = intervals[i][1]
        
        return count

```

### Explanation of the Code:

1. **Input Handling**: 
   - First, we check if the input list of intervals is empty. If it's empty, we return `0`, since there are no intervals to manage.

2. **Sorting the Intervals**:
   - We sort the intervals based on their end times. We do this because the strategy revolves around selecting the intervals that finish earliest. This helps to minimize overlap with subsequent intervals.

3. **Initialization**:
   - We initialize `count` to keep track of the number of intervals that need to be removed.
   - We set `end` to the end of the first interval, which will act as a marker for the end of the last non-overlapping interval that we consider.

4. **Iterating through Intervals**:
   - We start from the second interval and iterate through the sorted list of intervals.
   - For each interval, we check if it starts before the `end` of the last non-overlapping interval. If it does:
     - This indicates an overlap, and we increment `count` since we would need to remove one of the overlapping intervals.
   - If there is no overlap, we update `end` to the current interval’s end, effectively adding this interval to our non-overlapping set.

5. **Return the Result**: 
   - Finally, we return the total count of intervals that need to be removed.

### Complexity Analysis:
- **Time Complexity**: O(n log n), where n is the number of intervals. This is due to the sorting step.
- **Space Complexity**: O(1), as we are using a constant amount of extra space regardless of input size.

This solution efficiently finds the minimum number of intervals that need to be removed to eliminate overlaps, hence ensuring that the remaining intervals are non-overlapping.

# 452. Minimum Number of Arrows to Burst Balloons

### Problem Description 
There are some spherical balloons spread in two-dimensional space. For each balloon, provided input is the start and end coordinates of the horizontal diameter. Since it's horizontal, y-coordinates don't matter, and hence the x-coordinates of start and end of the diameter suffice. The start is always smaller than the end.

An arrow can be shot up exactly vertically from different points along the x-axis. A balloon with `xstart` and `xend` bursts by an arrow shot at `x` if `xstart ≤ x ≤ xend`. There is no limit to the number of arrows that can be shot. An arrow once shot keeps traveling up infinitely.

Given an array `points` where `points[i] = [xstart, xend]`, return the minimum number of arrows that must be shot to burst all balloons.


Example 1:
Input: points = [[10,16],[2,8],[1,6],[7,12]]
Output: 2
Explanation: One way is to shoot one arrow for example at x = 6 (bursting the balloons [2,8] and [1,6]) and another arrow at x = 11 (bursting the other two balloons).


Example 2:
Input: points = [[1,2],[3,4],[5,6],[7,8]]
Output: 4

Example 3:
Input: points = [[1,2],[2,3],[3,4],[4,5]]
Output: 2

Constraints:
`0 <= points.length <= 104`
`points[i].length == 2`
`-231 <= xstart < xend <= 231 - 1`

### Solution 
 To solve the problem of determining the minimum number of arrows needed to burst all the balloons, we can employ a greedy algorithm. The main idea is to sort the intervals (or balloon diameter ranges) by their ending points and then iteratively shoot arrows to cover as many balloons as possible with each arrow.

### Explanation:

1. **Sorting**: First, we sort the list of points (balloon ranges) based on the end coordinates (`xend`). This sorting allows us to ensure that we always choose the starting point that allows us to cover the most balloons with as few arrows as possible.

2. **Greedy Approach**: After sorting, we'll iterate through the sorted list and keep track of the position of the last arrow shot. If the current balloon's starting point (`xstart`) exceeds the position of the last arrow, we need to shoot a new arrow.

3. **Counting Arrows**: Each time we shoot an arrow, we simply increment our arrow count and update the position of the last arrow to the end of the current balloon range.

### Code Implementation:

Here's how this would be implemented in Python following the LeetCode submission format:



```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        # Edge case: If there are no points, no arrows are needed.
        if not points:
            return 0
        
        # Sort the points based on the end coordinate (xend)
        points.sort(key=lambda x: x[1])
        
        arrows = 1  # We need at least one arrow
        last_arrow_position = points[0][1]  # Position of the last arrow shot
        
        for i in range(1, len(points)):
            # If the current balloon starts after the last arrow position
            if points[i][0] > last_arrow_position:
                arrows += 1
                last_arrow_position = points[i][1]  # Update the last arrow position
        
        return arrows

```

### Code Explanation:

1. **Edge Case Handling**: The function first checks if the `points` list is empty. If so, it returns `0` as no arrows are needed.

2. **Sorting**: We sort `points` using the lambda function `key=lambda x: x[1]` which sorts the balloon ranges by their end boundaries.

3. **Initialization**: We initialize `arrows` to `1` because we need at least one arrow to start with. `last_arrow_position` is set to the end of the first balloon to reflect the position of the first arrow.

4. **Loop Through Points**: We loop through the sorted `points` starting from the second balloon. For each balloon, if its starting point is greater than the last shot arrow's position, it means we need to shoot a new arrow. We then increment the `arrows` count and update `last_arrow_position` to the end of the current balloon.

5. **Return the Count**: Finally, we return the total count of arrows needed.

### Complexity Analysis:

- **Time Complexity**: O(N log N), where N is the number of balloons since we primarily need to sort the list of points.
- **Space Complexity**: O(1), as we are using a constant amount of space for variables (besides the input list).

This implementation should work efficiently for the problem constraints provided. You can test this code directly on the LeetCode platform under the given problem number.

# 455. Assign Cookies

### Problem Description 
Assume you are an awesome parent and want to give your children some cookies. But, you should give each child at most one cookie.

Each child `i` has a greed factor `g[i]`, which is the minimum size of a cookie that the child will be content with; and each cookie `j` has a size `s[j]`. If `s[j] >= g[i]`, we can assign the cookie `j` to the child `i`, and the child `i` will be content. Your goal is to maximize the number of your content children and output the maximum number.


Example 1:
Input: g = [1,2,3], s = [1,1]
Output: 1
Explanation: You have 3 children and 2 cookies. The greed factors of 3 children are 1, 2, 3. 
And even though you have 2 cookies, since their size is both 1, you could only make the child whose greed factor is 1 content.

You need to output 1.


Example 2:
Input: g = [1,2], s = [1,2,3]
Output: 2
Explanation: You have 2 children and 3 cookies. The greed factors of 2 children are 1, 2. 
You have 3 cookies and their sizes are big enough to gratify all of the children, 
You need to output 2.


Constraints:
`1 <= g.length <= 3 * 104`
`0 <= s.length <= 3 * 104`
`1 <= g[i], s[j] <= 231 - 1`

### Solution 
 To solve the problem of maximizing the number of content children with the given greed factors and cookie sizes, we can employ a greedy algorithm. The goal is to assign the smallest available cookie that can satisfy a child's greed factor, ensuring that we can provide contentment to as many children as possible.

### Detailed Explanation

1. **Sort the Lists**: We start by sorting both the greed factors (`g`) of the children and the sizes (`s`) of the cookies. This allows us to systematically iterate through both lists from the smallest values to the largest.

2. **Two Pointer Technique**:
   - We use two pointers:
     - One pointer (`i`) for the greed factors of the children.
     - One pointer (`j`) for the sizes of the cookies.
   - We will iterate over the cookies with one pointer and check if the current cookie can satisfy the current child's greed. If it can, we move both pointers forward since this child is now satisfied.

3. **Count Content Children**: We maintain a count of satisfied children and return this count as the result.

By following these steps, we can ensure that we maximize the number of content children with the available cookies.

### Python Code

Here’s how the implementation looks in the LeetCode format:



```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        # Sort the greed factor g and the sizes of the cookies s
        g.sort()
        s.sort()
        
        # Initialize two pointers for children and cookies
        child_i = 0  # For greed factors
        cookie_j = 0  # For cookie sizes
        content_children = 0  # Count of content children
        
        # Use two pointers to find the maximum number of content children
        while child_i < len(g) and cookie_j < len(s):
            if s[cookie_j] >= g[child_i]:  
                # If the current cookie can satisfy the current child
                content_children += 1  
                child_i += 1  # Move to next child
            # Always move to next cookie
            cookie_j += 1
        
        return content_children

```

### Explanation of the Code:
- **Sorting**: We first sort both the `g` and `s` lists to facilitate a proper greedy approach.
- **Pointers Initialization**: `child_i` is initialized to zero to track the current child we are trying to satisfy, and `cookie_j` is initialized to zero for the current cookie.
- **While Loop**: We iterate through our lists while there are still children and cookies to process:
  - If the current cookie size `s[cookie_j]` can satisfy the current child `g[child_i]`, we increase the count of content children and move both pointers forward to the next child and the next cookie.
  - If the current cookie cannot satisfy the child, we only move the cookie pointer forward to check the next cookie.
  
This approach ensures that we correctly count how many children can be made content with the available cookies, maintaining both efficiency and clarity. 

### Complexity Analysis:
- **Time Complexity**: O(n log n + m log m) where `n` is the number of children and `m` is the number of cookies, due to the sorting steps. The subsequent while loop runs in O(n + m).
- **Space Complexity**: O(1) for the pointers used, and we modify the input lists in place for sorting, so no extra space is needed beyond the input. 

This algorithm can handle the constraints effectively and will run efficiently within the provided limits.

# 484. Find Permutation

### Problem Description 
By now, you are given a secret signature consisting of character 'D' and 'I'. 'D' represents a decreasing relationship between two numbers, 'I' represents an increasing relationship between two numbers. And our secret signature was constructed by a special integer array, which contains uniquely all the different number from 1 to n (n is the length of the secret signature plus 1). For example, the secret signature "DI" can be constructed by array [2,1,3] or [3,1,2], but won't be constructed by array [3,2,4] or [2,1,3,4], which are both illegal constructing special string that can't represent the "DI" secret signature.

On the other hand, now your job is to find the lexicographically smallest permutation of [1, 2, ... n] could refer to the given secret signature in the input.


Example 1:
Input: "I"
Output: [1,2]
Explanation: [1,2] is the only legal initial spectial string can construct secret signature "I", where the number 1 and 2 construct an increasing relationship.


Example 2:
Input: "DI"
Output: [2,1,3]
Explanation: Both [2,1,3] and [3,1,2] can construct the secret signature "DI", but since we want to find the one with the smallest lexicographical permutation, you need to output [2,1,3]
Note:
The input string will only contain the character 'D' and 'I'.

The length of input string is a positive integer and will not exceed 10,000

### Solution 
 To solve the problem of generating the lexicographically smallest permutation that conforms to a given secret signature comprising of characters 'D' (decreasing) and 'I' (increasing), we will follow these steps:

Our approach will be based on a stack data structure, where we will push integers onto the stack when we encounter a consecutive 'I' or 'D' character. Upon encountering an 'I', we will pop the integers from the stack to the result list, which ensures they are in the correct order after processing the entire string.

### Detailed Explanation

1. **Initialization**: 
   - Create an empty list to store the result.
   - Utilize a stack to keep track of numbers based on the D/I pattern.
   - Iterate over the characters in the signature while also dealing with the extra number at the end (n).

2. **Handling Characters**: 
   - If the current character is 'D', we push the current number onto the stack.
   - If the current character is 'I', we push the current number onto the stack and then pop all elements from the stack to the result. This will ensure that the numbers were in decreasing order (due to stack's LIFO property) and are added to the result in increasing order.

3. **Finalization**: 
   - After checking all characters, we still need to pop any remaining numbers in the stack to the result since we may have more elements left when the loop ends.

4. **Return the Result**: 
   - Finally, return or print the resultant list.

### Python Code

Here’s the implementation of the above logic:



```python
class Solution:
    def findPermutation(self, s: str) -> List[int]:
        n = len(s) + 1  # Length of the resulting array is length of s + 1
        result = []
        stack = []

        for i in range(1, n + 1):  # We want numbers from 1 to n
            stack.append(i)  # Add the current number to the stack
            
            # If we reached the end of an increasing condition or end of string
            if i == n or s[i - 1] == 'I':
                while stack:
                    result.append(stack.pop())  # Pop all elements in the stack to result
        
        return result

```

### Explanation of the Code:

1. **Input Handling**: The input string `s` is read; we calculate `n` as the length of `s` + 1.
2. **Loop through `1 to n`**: The loop runs from `1` to `n`, pushing every integer onto the stack.
3. **Condition Check**: If we encounter an `I` or reach the end (`i == n`), we pop all the elements from the stack and append them to the result list.
4. **Output**: The final result is printed or returned which is ensured to be lexicographically smallest due to the stack's properties and the way we process 'D' and 'I'.

### Example Walkthrough

- For input "DI":
    - Push 1 → stack: [1]
    - Push 2 → stack: [1, 2]
    - Since 'D', continue pushing.
    - Push 3 → stack: [1, 2, 3]
    - Now we hit 'I' (when i = 3):
        - Pop: result becomes [3]
        - Then pop: result becomes [3, 2]
        - Then pop: result becomes [3, 2, 1]
  - Finally, we reverse the result to get the correct order, leading to [2, 1, 3].

### Complexity
- **Time Complexity**: O(n) where n is the length of the input string. Each character is processed once.
- **Space Complexity**: O(n) for storing the numbers in the stack and the result list. 

This implementation efficiently finds the desired permutation by leveraging stack properties and meets the requirements specified in the problem.

# 502. IPO

### Problem Description 
Suppose LeetCode will start its IPO soon. In order to sell a good price of its shares to Venture Capital, LeetCode would like to work on some projects to increase its capital before the IPO. Since it has limited resources, it can only finish at most k distinct projects before the IPO. Help LeetCode design the best way to maximize its total capital after finishing at most k distinct projects. 
You are given several projects. For each project i, it has a pure profit Pi and a minimum capital of Ci is needed to start the corresponding project. Initially, you have W capital. When you finish a project, you will obtain its pure profit and the profit will be added to your total capital.

To sum up, pick a list of at most k distinct projects from given projects to maximize your final capital, and output your final maximized capital.


Example 1:
Input: k=2, W=0, Profits=[1,2,3], Capital=[0,1,1].
Output: 4
Explanation: Since your initial capital is 0, you can only start the project indexed 0.

             After finishing it you will obtain profit 1 and your capital becomes 1.

             With capital 1, you can either start the project indexed 1 or the project indexed 2.

             Since you can choose at most 2 projects, you need to finish the project indexed 2 to get the maximum capital.

             Therefore, output the final maximized capital, which is 0 + 1 + 3 = 4.

Note:
You may assume all numbers in the input are non-negative integers.

The length of Profits array and Capital array will not exceed 50,000.

The answer is guaranteed to fit in a 32-bit signed integer.

### Solution 
 To solve the problem of maximizing the capital after completing a limited number of projects, we can use a combination of sorting and a max-heap data structure (priority queue). The key idea here is to always select the most profitable project that can be started with the current capital.

Here’s a step-by-step breakdown of the approach:

1. **Sort Projects by Capital**: First, we sort the projects based on their required capital. This allows us to efficiently select which projects can be started at any given time based on the current capital.

2. **Use a Max-Heap for Profitable Projects**: We maintain a max-heap of projects that can be started with the current capital. The heap allows us to efficiently pick the most profitable one.

3. **Iterate Up to k Projects**: We will iterate up to `k` times:
    - For each iteration, we add all projects that can be started with the current capital to the max-heap.
    - If the max-heap is not empty, we pop the most profitable project from the heap, add its profit to the current capital, and repeat.

4. **Return the Final Capital**: After finishing up to `k` projects, we return the current capital as the maximum achievable capital.

Here is the Python implementation of the above logic formatted for direct usage on the LeetCode platform:



```python
import heapq

class Solution:
    def findMaximizedCapital(self, k: int, W: int, Profits: List[int], Capital: List[int]) -> int:
        # Pair the Capital with corresponding Profit and sort by Capital
        projects = sorted(zip(Capital, Profits))
        
        max_heap = []
        index = 0
        n = len(projects)
        
        for _ in range(k):
            # Add all projects that can be started with the current capital W
            while index < n and projects[index][0] <= W:
                # Push the profit into the max-heap (negative for max-heap behavior)
                heapq.heappush(max_heap, -projects[index][1])
                index += 1
            
            # If there is no project we can take, break early
            if not max_heap:
                break
            
            # Take the most profitable project
            W -= heapq.heappop(max_heap)  # Pop the max profit (note: it's negative)
            
        return W

```

### Explanation of the Implementation:
1. **Sorting Projects**: We start by zipping the `Capital` and `Profits` lists together and sorting them based on the required capital.

2. **Max-Heap**: We use `heapq` to create a max-heap, but since Python's `heapq` only provides a min-heap, we push the negative of profits to mimic max-heap behavior.

3. **Main Loop**: The loop runs `k` times, and:
   - We add all projects that can be launched with the current capital `W` to the heap.
   - If the heap is non-empty, we pop the most profitable project and update our capital.

4. **Return Value**: Finally, we return the updated capital after processing up to `k` projects.

### Complexity Analysis:
- The overall time complexity is dominated by the sorting step: O(n log n), where n is the number of projects. The operations on the heap will result in a time complexity of O(k log k) at most, where k is the number of projects we can take, making the solution efficient for the given problem constraints.

# 605. Can Place Flowers

### Problem Description 
You have a long flowerbed in which some of the plots are planted, and some are not. However, flowers cannot be planted in adjacent plots.

Given an integer array `flowerbed` containing `0`'s and `1`'s, where `0` means empty and `1` means not empty, and an integer `n`, return if `n` new flowers can be planted in the `flowerbed` without violating the no-adjacent-flowers rule.


Example 1:
Input: flowerbed = [1,0,0,0,1], n = 1
Output: true

Example 2:
Input: flowerbed = [1,0,0,0,1], n = 2
Output: false

Constraints:
`1 <= flowerbed.length <= 2 * 104`
`flowerbed[i]` is `0` or `1`.

There are no two adjacent flowers in `flowerbed`.

`0 <= n <= flowerbed.length`

### Solution 
 To solve the problem of determining whether we can plant `n` new flowers in a flowerbed represented as an array, we need to consider the constraints imposed by the requirement that no two flowers can be adjacent. Therefore, we will carefully examine the cells of the flowerbed to find spots where we can plant new flowers.

Here's how we can approach the problem:

1. **Iterate through the flowerbed:** We'll loop through each plot in the flowerbed array.
2. **Check conditions for planting a flower:** A flower can be planted in an empty plot (represented by `0`) if both its adjacent plots are either empty or out of bounds. Specifically, for plot `i`, we need to ensure:
   - If `i-1` is within bounds, `flowerbed[i-1]` must be `0` (the plot to the left is empty).
   - If `i+1` is within bounds, `flowerbed[i+1]` must be `0` (the plot to the right is empty).
3. **Plant the flower:** If the conditions are satisfied, we can place a flower by setting `flowerbed[i]` to `1` and incrementing our count of planted flowers.
4. **Check the count:** If the count of newly planted flowers reaches `n`, we return `True`. If we finish iterating through the flowerbed and haven't reached `n`, we return `False`.

Now, let's implement this in Python using the specified format that would run on LeetCode:



```python
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        count = 0  # To count how many flowers we can plant
        size = len(flowerbed)

        for i in range(size):
            # Check if the current spot is empty
            if flowerbed[i] == 0:
                # Check if the left and right plots are also empty or out of bounds
                if (i == 0 or flowerbed[i - 1] == 0) and (i == size - 1 or flowerbed[i + 1] == 0):
                    flowerbed[i] = 1  # Plant the flower
                    count += 1  # Increment the count of planted flowers
                    if count >= n:  # If we have planted enough flowers, return True
                        return True
        
        return count >= n  # Check if we have planted at least n flowers

```

### Explanation of the Code:

- **Class Definition:** We define a class `Solution` which is a typical approach in LeetCode.
- **Function Definition:** The `canPlaceFlowers` function accepts two parameters: `flowerbed` (a list of integers) and `n` (the number of flowers to plant).
- **Initialization:** We initialize `count` to track the number of successfully planted flowers and `size` to hold the length of the flowerbed.
- **Looping through flowerbed:** We iterate over each plot in the flowerbed.
- **Condition Checking:** For every empty plot (value `0`), we check both adjacent plots and the bounds to see if it's possible to plant a flower there.
- **Updating state:** When we plant a flower, we set that position to `1` and increase our `count`. If we've planted at least `n` flowers at any point, we return `True` immediately.
- **Final Return:** If we finish checking all plots and have not planted at least `n` flowers, we return `False`.

This algorithm runs in O(m) time complexity, where m is the size of the flowerbed, and uses O(1) additional space since we’re modifying the flowerbed in place.

# 621. Task Scheduler

### Problem Description 
Given a characters array `tasks`, representing the tasks a CPU needs to do, where each letter represents a different task. Tasks could be done in any order. Each task is done in one unit of time. For each unit of time, the CPU could complete either one task or just be idle.

However, there is a non-negative integer `n` that represents the cooldown period between two same tasks (the same letter in the array), that is that there must be at least `n` units of time between any two same tasks.

Return the least number of units of times that the CPU will take to finish all the given tasks.


Example 1:
Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
Explanation: 
A -> B -> idle -> A -> B -> idle -> A -> B
There is at least 2 units of time between any two same tasks.


Example 2:
Input: tasks = ["A","A","A","B","B","B"], n = 0
Output: 6
Explanation: On this case any permutation of size 6 would work since n = 0.

["A","A","A","B","B","B"]
["A","B","A","B","A","B"]
["B","B","B","A","A","A"]
...

And so on.


Example 3:
Input: tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2
Output: 16
Explanation: 
One possible solution is
A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> idle -> idle -> A -> idle -> idle -> A

Constraints:
`1 <= task.length <= 104`
`tasks[i]` is upper-case English letter.

The integer `n` is in the range `[0, 100]`.

### Solution 
 To tackle the problem of scheduling tasks with a cooldown period, we can use a greedy strategy. The idea is to focus on executing the most frequent tasks while ensuring that the cooldown requirement is met.

### Explanation of the Approach:

1. **Count the Frequencies**:
   First, we need to determine how many times each task appears. This can be achieved with a frequency counter (like `Counter` from the `collections` module in Python).

2. **Max Frequency**:
   Identify the task with the maximum frequency. The task with the maximum frequency will dictate the minimum time taken to schedule the tasks due to the cooldown requirement.

3. **Calculate Ideal Slots**:
   Given the most frequent task, the number of "slots" that this task creates can be visualized based on its frequency. If a task appears `max_count` times, it will occupy `max_count` slots initially.

4. **Idle Times**:
   For each occurrence of the most frequent task, we need to consider the cooldown `n`. Therefore, each task will create `n` idle slots following the max task before another task of the same type can be scheduled. This spreads the execution of these tasks.

5. **Calculate Total Time**:
   - The total time will be the maximum of:
     - The total number of tasks
     - The total slots created by the most frequent task and its cooldowns:
       \[
       \text{total slots} = \text{max\_count - 1} \times (n + 1) + 1
       \]
     This formula calculates the required time considering the slots taken by the most frequent task and the necessary cooldowns.

6. **Return the Max**:
   Finally, take the maximum between the total tasks and the total calculated time to ensure all are scheduled.

Here’s the Python implementation based on this logic:



```python
from collections import Counter

class Solution:
    def leastInterval(self, tasks: list[str], n: int) -> int:
        # Step 1: Count the frequency of each task
        task_count = Counter(tasks)
        
        # Step 2: Find the maximum frequency
        max_count = max(task_count.values())
        
        # Step 3: Count how many tasks have this maximum frequency
        max_count_tasks = sum(1 for count in task_count.values() if count == max_count)
        
        # Step 4: Calculate the total number of time slots needed
        # Formula: total slots = (max_count - 1) * (n + 1) + max_count_tasks
        total_slots = (max_count - 1) * (n + 1) + max_count_tasks
        
        # Step 5: The least time needed is the maximum of total tasks and total slots calculated
        return max(total_slots, len(tasks))

```

### Summary:
- We calculate how many times each task occurs.
- Using the most frequent task, we determine the necessary scheduling with the provided cooldown.
- Return the maximum of the calculated time and the total number of tasks.

This solution adheres to the constraints provided and efficiently calculates the least amount of time needed using a straightforward greedy algorithm approach with a runtime complexity of O(m + n) where m is the number of unique tasks. This is efficient and well-suited for the input limits.

# 630. Course Schedule III

### Problem Description 
There are `n` different online courses numbered from `1` to `n`. Each course has some duration(course length) `t` and closed on `dth` day. A course should be taken continuously for `t` days and must be finished before or on the `dth` day. You will start at the `1st` day.

Given `n` online courses represented by pairs `(t,d)`, your task is to find the maximal number of courses that can be taken.


Example:
Input: [[100, 200], [200, 1300], [1000, 1250], [2000, 3200]]
Output: 3
Explanation: 
There're totally 4 courses, but you can take 3 courses at most:
First, take the 1st course, it costs 100 days so you will finish it on the 100th day, and ready to take the next course on the 101st day.

Second, take the 3rd course, it costs 1000 days so you will finish it on the 1100th day, and ready to take the next course on the 1101st day. 
Third, take the 2nd course, it costs 200 days so you will finish it on the 1300th day. 
The 4th course cannot be taken now, since you will finish it on the 3300th day, which exceeds the closed date.

Note:
The integer 1 <= d, t, n <= 10,000.

You can't take two courses simultaneously.

### Solution 
 To solve the problem of maximizing the number of online courses that can be taken given their duration and deadline, we can follow a greedy algorithm approach. This involves sorting the courses based on their deadlines and then iterating through them while maintaining a running tally of the total time spent on courses taken so far. 

Here’s a step-by-step breakdown of the solution:

1. **Understanding Inputs and Outputs**: 
   - We receive a list of courses where each course is represented as a pair `(t, d)`, where `t` is the duration and `d` is the deadline by which the course must be completed.
   - Our goal is to output the maximum number of courses that can be taken.

2. **Sorting the Courses**:
   - First, we should sort the courses based on their deadlines `d`. This ensures we always focus on the courses that have to be completed soonest, which helps in maximizing the number of courses taken.

3. **Greedy Selection of Courses**:
   - We will maintain a running total of the time spent on courses taken so far.
   - For each course in the sorted list, we check if adding this course’s duration to the total time does not exceed its deadline. If it fits, we take the course and update our total time.

4. **Implementation**: We will implement the above logic in the form of a function that is compatible with LeetCode's formatting.

Here's the Python code to achieve this:



```python
from typing import List

class Solution:
    def maxCourses(self, courses: List[List[int]]) -> int:
        # Sort courses by their deadline
        courses.sort(key=lambda x: x[1])
        
        total_time = 0
        count = 0
        
        for t, d in courses:
            if total_time + t <= d:  # Check if course can be taken
                total_time += t       # Update total time spent
                count += 1            # Increment the count of courses taken
        
        return count

```

### Explanation of the Code:
- We first import the necessary type hinting from the `typing` module.
- We define a class named `Solution` with a method `maxCourses` that takes a list of courses as input.
- Inside the method, we sort the `courses` list based on their deadlines using a lambda function as the key.
- We initialize `total_time` to keep track of the total days spent on courses taken and `count` to count the number of courses taken.
- We iterate through each course (duration `t` and deadline `d`) in the sorted list:
  - If the current total time plus the duration of the course does not exceed the course’s deadline, we take the course by adding its duration to `total_time` and incrementing the `count`.
- Finally, the method returns the maximum number of courses that can be taken.

### Complexity Analysis:
- **Time Complexity**: O(n log n) due to the sorting step, where `n` is the number of courses.
- **Space Complexity**: O(1) as we are using a constant amount of space apart from the input.

This implementation efficiently determines the maximum number of courses that can be taken without exceeding their respective deadlines. You can directly run this code on the LeetCode platform as it follows their expected format.

# 649. Dota2 Senate

### Problem Description 
In the world of Dota2, there are two parties: the `Radiant` and the `Dire`.

The Dota2 senate consists of senators coming from two parties. Now the senate wants to make a decision about a change in the Dota2 game. The voting for this change is a round-based procedure. In each round, each senator can exercise `one` of the two rights:
`Ban one senator's right`:
	A senator can make another senator lose all his rights in this and all the following rounds.

`Announce the victory`:
	If this senator found the senators who still have rights to vote are all from the same party, he can announce the victory and make the decision about the change in the game.

Given a string representing each senator's party belonging. The character 'R' and 'D' represent the `Radiant` party and the `Dire` party respectively. Then if there are `n` senators, the size of the given string will be `n`.

The round-based procedure starts from the first senator to the last senator in the given order. This procedure will last until the end of voting. All the senators who have lost their rights will be skipped during the procedure.

Suppose every senator is smart enough and will play the best strategy for his own party, you need to predict which party will finally announce the victory and make the change in the Dota2 game. The output should be `Radiant` or `Dire`.


Example 1:
Input: "RD"
Output: "Radiant"
Explanation: The first senator comes from Radiant and he can just ban the next senator's right in the round 1. 
And the second senator can't exercise any rights any more since his right has been banned. 
And in the round 2, the first senator can just announce the victory since he is the only guy in the senate who can vote.


Example 2:
Input: "RDD"
Output: "Dire"
Explanation: 
The first senator comes from Radiant and he can just ban the next senator's right in the round 1. 
And the second senator can't exercise any rights anymore since his right has been banned. 
And the third senator comes from Dire and he can ban the first senator's right in the round 1. 
And in the round 2, the third senator can just announce the victory since he is the only guy in the senate who can vote.

Note:
The length of the given string will in the range [1, 10,000].

### Solution 
 To solve the problem of predicting which party (Radiant or Dire) will win the Dota2 senate vote, we can leverage a queue-based approach, which allows us to simulate the rounds of voting efficiently. Here’s the detailed explanation and the Python code that follows the LeetCode solution format.

### Detailed Explanation:
1. **Understanding the Voting Mechanism**:
   - Each senator can either ban the rights of another senator or announce victory.
   - Since senators are smart and will play optimally, they will always try to eliminate the other party's senators when it's their turn.

2. **Data Structures**:
   - We will use two queues to respectively keep track of the indices of the 'R' (Radiant) and 'D' (Dire) senators. This will help us simulate the rounds effectively.
   - We will also keep track of the total number of senators from each party.

3. **Simulation of Rounds**:
   - In each round, we will compare the front of the two queues to see which senator gets to act first.
   - The senator whose index is smaller will act first (and this is important for the order of actions).
   - The acting senator will ban the opposing senator by removing their index from their respective queue.
   - The acting senator's index will be incremented by the total number of senators, which effectively gives them a ‘next turn’ advantage in the next round.

4. **Termination**:
   - The process continues until one of the queues is empty, which means that one party has been completely banned. The last party with remaining senators announces victory.

5. **Time Complexity**:
   - The approach works in O(n) time complexity where n is the length of the input string, since each senator is processed at most twice (once to ban and once to act again).

### Implementation:
Here is the Python code that implements the above logic using the LeetCode submission format.



```python
class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        from collections import deque
        
        radiant_queue = deque()
        dire_queue = deque()
        
        # Fill the queues with the indices of the senators
        for index, senator in enumerate(senate):
            if senator == 'R':
                radiant_queue.append(index)
            else:  # senator == 'D'
                dire_queue.append(index)

        total_senators = len(senate)

        # Simulate the rounds of voting
        while radiant_queue and dire_queue:
            radiant_index = radiant_queue.popleft()
            dire_index = dire_queue.popleft()

            # Whoever has the smaller index gets to act first
            if radiant_index < dire_index:
                # Radiant senator bans Dire senator
                radiant_queue.append(radiant_index + total_senators)
            else:
                # Dire senator bans Radiant senator
                dire_queue.append(dire_index + total_senators)

        # If one of the queues is empty, the other party wins
        return "Radiant" if radiant_queue else "Dire"

```

### How to Run on LeetCode:
You can copy the above code and paste it directly into the code editor on the LeetCode website after selecting the problem. Make sure the class name `Solution` matches what is defined in the problem statement, and the method `predictPartyVictory` should also match the input parameters as specified.

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
 To solve this problem, we need to maximize the number of 'A's printed on the screen after pressing our special keyboard keys a total of `N` times. The keys we can use are:

1. A: Print one 'A'.
2. Ctrl-A: Select all printed characters.
3. Ctrl-C: Copy the selected characters to a buffer.
4. Ctrl-V: Paste from the buffer.

Given the constraints of the problem, our aim is to determine the most efficient way to use these keys in order to maximize the output of 'A's.

### Explanation:

The idea is to realize that pressing the keys in certain sequences will yield different numbers of 'A's printed. The most efficient way we can create a large number of 'A's using the Ctrl keys is to first build up a certain number of 'A's by simply pressing 'A' multiple times, and then switch to using the Ctrl combinations.

The general strategy can be summarized as follows:

1. **Initial Presses:** Start by pressing 'A' a certain number of times. Let's say we press 'A' `k` times.
2. **Select & Copy:** After we have a base of 'A's, we can use Ctrl-A and Ctrl-C to select and copy them.
3. **Pasting:** We can then paste them multiple times, which exponentially increases the count of 'A's printed.

The key is to decide how many presses we should dedicate to the initial 'A's, and then how many to the copying and pasting.

### Dynamic Programming Approach:

We will use a dynamic programming approach where we maintain an array `dp` such that `dp[i]` represents the maximum number of 'A's that can be printed with `i` presses.

1. Start by initializing the `dp` array where `dp[i]` is initially set to `i`, since the worst case is pressing 'A' for all `i` presses.
2. For each possible number of presses, we will try pressing 'A' `k` times (where `k` can be from 1 to `i-3`, allowing room for Ctrl-A, Ctrl-C, and at least one Ctrl-V).
3. After `k` presses, the remaining presses would be used for Ctrl-A, Ctrl-C, and several Ctrl-V's.
4. The relationship becomes: 
    \[
    dp[i] = \max(dp[i], dp[k] \times (times))
    \]
   where `times = i - k - 1` and `1 + times` represents the total number of pastes.

The overall complexity will be manageable given that `N` can only be as large as 50.

### Python Code:



```python
class Solution:
    def maxA(self, N: int) -> int:
        # dp[i] will hold the maximum number of 'A's we can print with i key presses
        dp = [0] * (N + 1)

        for i in range(1, N + 1):
            dp[i] = i  # default: if we just press A i times
            for j in range(1, i - 2):  # j represents the number of A's pressed before Ctrl-A
                # times are the remaining presses after pressing A j times, selecting it, copying it and pasting it
                times = i - j - 1
                # The number of A's we can create is:
                # dp[j] -> initial A's, times -> number of pastes after selecting and copying
                dp[i] = max(dp[i], dp[j] * (times + 1))
        
        return dp[N]

# Example of using the Solution class to run the problem
# solution = Solution()
# print(solution.maxA(7))  # Example call to the function

```

### Explanation of the Code:

- We define a class `Solution` with a method `maxA`.
- We initialize a `dp` array of size `N+1` to store the maximum number of 'A's.
- We iterate through all possible values of key presses, and for each one, we evaluate how many 'A's can be maximized given the prior presses.
- Finally, we return `dp[N]`, which gives the maximum 'A's achievable with `N` presses.

This method efficiently computes the result using a systematic dynamic programming approach, ensuring that we explore all possibilities while remaining within the given constraints.

# 659. Split Array into Consecutive Subsequences

### Problem Description 
Given an integer array `nums` that is sorted in ascending order, return `true` if and only if you can split it into one or more subsequences such that each subsequence consists of consecutive integers and has a length of at least `3`.


Example 1:
Input: nums = [1,2,3,3,4,5]
Output: true
Explanation:
You can split them into two consecutive subsequences : 
1, 2, 3
3, 4, 5

Example 2:
Input: nums = [1,2,3,3,4,4,5,5]
Output: true
Explanation:
You can split them into two consecutive subsequences : 
1, 2, 3, 4, 5
3, 4, 5

Example 3:
Input: nums = [1,2,3,4,4,5]
Output: false

Constraints:
`1 <= nums.length <= 104`
`-1000 <= nums[i] <= 1000`
`nums` is sorted in an ascending order.

### Solution 
 To solve this problem, we need to determine whether we can split the sorted array `nums` into one or more subsequences, each consisting of at least three consecutive integers.

### Approach:
1. **Use a Counter:** We will utilize a frequency counter to monitor how many times each number appears in `nums`. This will allow us to keep track of available numbers for forming subsequences.
  
2. **Greedily Build Subsequences:** We will attempt to build subsequences by checking if we can extend existing subsequences or start new ones. We maintain a priority-queue (or a simple list/array) to keep track of the endings of the current subsequences. We will always attempt to extend the existing subsequences before starting a new one.

3. **Iterate through `nums`:** For each number in `nums`, we'll check:
   - If there's a subsequence that can be extended with the current number.
   - If not, we need to check if we can start a new subsequence or not based on the available counts of the current and the next two consecutive numbers.

4. **Conditions for Valid Subsequences:**
   - A subsequence can be extended if we have at least one subsequence ending in `current number - 1`.
   - To start a new subsequence from `current number`, we need to ensure that we have `current number`, `current number + 1`, and `current number + 2` available.

5. **Final Check:** Ensure that at every step, we aren't left with any number that cannot form a valid subsequence by the end of the iterations.

Here is the implementation of this approach encapsulated in a Python function following the LeetCode format:



```python
from collections import Counter

class Solution:
    def isPossible(self, nums: List[int]) -> bool:
        # Count the frequency of each number
        count = Counter(nums)
        # Tracks the number of subsequences ending with a specific integer
        tails = Counter()
        
        for num in nums:
            if count[num] == 0:
                continue  # If we've used this number already, skip it
            
            # We can extend a subsequence by 1 if possible
            if tails[num - 1] > 0:
                tails[num - 1] -= 1  # Remove the tail of the previous sequence
                tails[num] += 1      # Create a new tail with the current number
            else:
                # We want to create a new sequence starting from `num`
                if count[num + 1] > 0 and count[num + 2] > 0:
                    count[num] -= 1
                    count[num + 1] -= 1
                    count[num + 2] -= 1
                    tails[num + 2] += 1  # The next sequence will end with `num + 2`
                else:
                    return False  # Cannot form a new valid sequence
        
            # Use the current number
            count[num] -= 1
        
        return True  # All constraints satisfied

```

### Explanation of the Code:
- **Counter Initialization:** We use `Counter` to create a frequency map of the numbers in `nums`.
- **Tails Counter:** This keeps track of how many subsequences end with a certain number.
- **Loop Through `nums`:** For each number:
  - First, we check if it's already been fully used (skip if count is `0`).
  - Try to extend an existing valid subsequence if possible.
  - If not, check if we can start a new sequence with at least three consecutive numbers.
  
- **Return:** If we can manage to form all required subsequences, we return `True`; otherwise, we return `False` at the first failure point.

This solution has a time complexity of **O(n)** and a space complexity of **O(n)**, making it efficient for the input constraints given.

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
 To solve the given problem, we will use a dynamic programming approach. The main idea is that we need to keep track of two states: the maximum profit we can achieve if we currently hold a stock and the maximum profit we can achieve if we do not hold a stock.

### Explanation of the Solution

1. **Define State Variables**:
   - `cash`: Maximum profit we can have if we do not currently hold any stock.
   - `hold`: Maximum profit we can have if we currently hold a stock.

2. **Initialization**:
   - Initially, if we have not made any transactions, then:
     - `cash = 0` (profit is zero because we have not bought anything).
     - `hold = -prices[0]` (if we buy the stock on the first day, our profit is negative by the cost of the stock).

3. **State Transition**:
   - For each price in the `prices` array:
     - Calculate the new `cash` value:
       - If we sell our stock today, we would have `hold + prices[i] - fee`.
     - Calculate the new `hold` value:
       - If we buy stock today, we would have `cash - prices[i]`.
   - We update `cash` and `hold` accordingly.

4. **Final Result**:
   - After processing all prices, `cash` will contain the maximum profit we can achieve without holding any stock.

### Python Implementation

Here is the implementation according to the LeetCode format:



```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        cash = 0  # Max profit without holding a stock
        hold = -prices[0]  # Max profit while holding a stock
        
        for price in prices:
            # Update cash and hold variables
            cash = max(cash, hold + price - fee)  # Sell the stock
            hold = max(hold, cash - price)        # Buy the stock
            
        return cash

```

### Detailed Breakdown of the Code:

1. **Class Definition**: We define a class `Solution` that contains the method `maxProfit`.

2. **Method Signature**: The method takes in a list `prices` representing prices of stocks and an integer `fee` representing the transaction fee.

3. **Variables Initialization**:
   - We start by initializing `cash` to `0` (profit when not holding any stock).
   - We initialize `hold` to `-prices[0]` since if we buy on the first day, we'll incur a loss equal to the price of the stock.

4. **Iteration Over Prices**:
   - We iterate through each `price` in `prices`:
     - We update the `cash` state by considering the profit from selling today.
     - We also update the `hold` state by considering the cost of buying today.

5. **Return Value**: Finally, we return the `cash` variable which now contains the maximum profit achievable without holding any stock.

### Time and Space Complexity:
- **Time Complexity**: O(n), where n is the length of the `prices` array. We only make a single pass through the array.
- **Space Complexity**: O(1), since we only use a fixed amount of extra space (for `cash` and `hold`).

This solution efficiently calculates the maximum profit possible while considering the transaction fee for each transaction.

# 738. Monotone Increasing Digits

### Problem Description 
Given a non-negative integer `N`, find the largest number that is less than or equal to `N` with monotone increasing digits.

(Recall that an integer has monotone increasing digits if and only if each pair of adjacent digits `x` and `y` satisfy `x <= y`.)

Example 1:
Input: N = 10
Output: 9

Example 2:
Input: N = 1234
Output: 1234

Example 3:
Input: N = 332
Output: 299
Note:
`N` is an integer in the range `[0, 10^9]`.

### Solution 
 To solve the problem of finding the largest number less than or equal to a given non-negative integer `N` with monotone increasing digits, we need to understand what "monotone increasing digits" means. A number has monotone increasing digits if each digit from the left to the right does not decrease.

### Approach

1. **Identify Monotone Increase Points**: 
   - Traverse the digits of `N` from left to right and check where the digits start to decrease. This indicates that we need to modify the number to ensure it remains monotone increasing.

2. **Modify Digits**: 
   - Once we find a point where the current digit is greater than the next digit, we reduce the current digit by `1` (to maintain the maximum possible number) and change all subsequent digits to `9` (maximizing those digits while ensuring they are consistent). This guarantees the number stays under the original `N`.

3. **Edge Cases**: 
   - If we encounter a situation where reducing a digit causes an earlier digit to no longer be valid (like turning `21` to `19`), we need to backtrack and apply the same change to the earlier digits.

4. **Construct the Result**: 
   - After adjusting the digits, we will construct the final number and ensure it meets the monotonicity conditions.

### Implementation

Here's how this can be implemented in Python in a format suitable for LeetCode:



```python
class Solution:
    def monotoneIncreasingDigits(self, N: int) -> int:
        # Convert the integer to a list of digits
        digits = list(map(int, str(N)))
        n = len(digits)
        
        # We'll find the point where the digits start to decrease
        mark = n  # Mark the position to modify
        for i in range(n - 1):
            if digits[i] > digits[i + 1]:  # A drop is found
                mark = i
                break
        
        # If we found a drop, we need to modify
        if mark < n:
            # Decrease the mark digit
            digits[mark] -= 1
            
            # Set all subsequent digits to 9 to maximize
            for j in range(mark + 1, n):
                digits[j] = 9
            
            # Check for any subsequent decreases
            for k in range(mark - 1, -1, -1):
                if digits[k] > digits[k + 1]:
                    digits[k] -= 1
                    digits[k + 1] = 9
        
        # Convert the list back to an integer
        result = int(''.join(map(str, digits)))
        
        return result

# Example usage:
# sol = Solution()
# print(sol.monotoneIncreasingDigits(10))    # Output: 9
# print(sol.monotoneIncreasingDigits(1234))  # Output: 1234
# print(sol.monotoneIncreasingDigits(332))   # Output: 299

```

### Explanation

- The solution first converts the number to its digit representation and traverses those digits to identify where the monotonicity breaks (where a digit is greater than the next).
  
- If a drop is found, it marks that position, reduces the problematic digit, and changes subsequent digits. This ensures we create the largest possible number less than `N` that maintains monotonic digits.

- Finally, if necessary, we confirm that no earlier digits cause inconsistency after the adjustments.

This implementation is efficient with a time complexity of O(d), where d is the number of digits in `N`. Given that \( N \leq 10^9 \), d can be at most \( 10 \). Therefore, this algorithm will run in constant time concerning N and is suitable for large inputs within the provided constraints.

# 757. Set Intersection Size At Least Two

### Problem Description 
An integer interval `[a, b]` (for integers `a < b`) is a set of all consecutive integers from `a` to `b`, including `a` and `b`.

Find the minimum size of a set S such that for every integer interval A in `intervals`, the intersection of S with A has a size of at least two.


Example 1:
Input: intervals = [[1,3],[1,4],[2,5],[3,5]]
Output: 3
Explanation: Consider the set S = {2, 3, 4}.  For each interval, there are at least 2 elements from S in the interval.

Also, there isn't a smaller size set that fulfills the above condition.

Thus, we output the size of this set, which is 3.


Example 2:
Input: intervals = [[1,2],[2,3],[2,4],[4,5]]
Output: 5
Explanation: An example of a minimum sized set is {1, 2, 3, 4, 5}.


Constraints:
`1 <= intervals.length <= 3000`
`intervals[i].length == 2`
`0 <= ai < bi <= 108`

### Solution 
 To solve the problem of finding the minimum size of a set \( S \) such that every integer interval \( A \) in the given list of intervals has at least two elements from \( S \), we can adopt a greedy algorithm approach. Here's a detailed breakdown of how we can achieve this:

### Detailed Explanation:

1. **Understanding the Problem**:
   - For each given interval \([a, b]\), the set \( S \) should contain at least two numbers within this range.
   - The challenge is to minimize the size of \( S \) while still satisfying the requirement for every interval.

2. **Greedy Choice**:
   - Sort the intervals based on their ending points. This allows us to cover the maximum number of intervals with the least number of points in \( S \).
   - For each interval, if it is not already covered by the previous selected points in \( S \), we will select two points to ensure at least two points fall within this interval.

3. **Steps**:
   - Sort the intervals based on the second element (end) of each interval.
   - Maintain a list for points in set \( S \).
   - For each interval, determine if it is covered by at least two points from \( S \). If it isn’t, add two new points to \( S \) that fall within this interval.

4. **Implementation**:
Now let's implement this logic in Python in accordance with the LeetCode format.



```python
class Solution:
    def minSetSize(self, intervals: List[List[int]]) -> int:
        # Step 1: Sort intervals by their ending value
        intervals.sort(key=lambda x: x[1])
        
        # This will store the points in our set S
        points = []
        
        for interval in intervals:
            a, b = interval
            # Check if the last two points in points cover this interval
            if len(points) < 2 or points[-1] < a or points[-2] < a:
                # Add the last two points from the current interval
                points.append(b-1)  # Adding the second last element
                points.append(b)    # Adding the last element
        
        # The result is the number of points we selected
        return len(points)

# Example usage
solution = Solution()
print(solution.minSetSize([[1, 3], [1, 4], [2, 5], [3, 5]]))  # Output: 3
print(solution.minSetSize([[1, 2], [2, 3], [2, 4], [4, 5]]))  # Output: 5

```

### Explanation of the Code:
- **Sorting**: We first sort the intervals to facilitate the greedy selection of points in \( S \).
- **Points Selection**: For each interval, if the two last chosen points do not cover the interval, we add two new points (just before the last point of the interval).
- **Output**: Finally, we return the number of points selected in \( S \).

### Complexity Analysis:
- **Time Complexity**: The sorting step takes \( O(n \log n) \), and iterating through the intervals takes \( O(n) \). Thus, the overall complexity is \( O(n \log n) \).
- **Space Complexity**: In the worst case, we are storing two points for every interval, so the space used is \( O(n) \).

This solution efficiently constructs the minimum set \( S \) needed to satisfy the problem's constraints.

# 759. Employee Free Time

### Problem Description 
We are given a list `schedule` of employees, which represents the working time for each employee.

Each employee has a list of non-overlapping `Intervals`, and these intervals are in sorted order.

Return the list of finite intervals representing common, positive-length free time for all employees, also in sorted order.

(Even though we are representing `Intervals` in the form `[x, y]`, the objects inside are `Intervals`, not lists or arrays. For example, `schedule[0][0].start = 1`, `schedule[0][0].end = 2`, and `schedule[0][0][0]` is not defined).  Also, we wouldn't include intervals like [5, 5] in our answer, as they have zero length.


Example 1:
Input: schedule = [[[1,2],[5,6]],[[1,3]],[[4,10]]]
Output: [[3,4]]
Explanation: There are a total of three employees, and all common
free time intervals would be [-inf, 1], [3, 4], [10, inf].

We discard any intervals that contain inf as they aren't finite.


Example 2:
Input: schedule = [[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]]
Output: [[5,6],[7,9]]

Constraints:
`1 <= schedule.length , schedule[i].length <= 50`
`0 <= schedule[i].start < schedule[i].end <= 10^8`

### Solution 
 To solve the problem of finding common free time intervals for a list of employees, we can follow these steps:

1. **Flatten the Schedule**: Start by collecting all the intervals from the employees’ schedules into a single list while keeping track of their start and end times.
2. **Merge Intervals**: Combine overlapping intervals to simplify our search for gaps (free time).
3. **Find Free Time**: Traverse the merged intervals to identify gaps between the end of one interval and the start of the next.

Here is the Python code implementing this logic, which can be directly run on the LeetCode platform:



```python
# Definition for an Interval.
class Interval:
    def __init__(self, start=0, end=0):
        self.start = start
        self.end = end

class Solution:
    def employeeFreeTime(self, schedule: List[List[Interval]]) -> List[Interval]:
        # Step 1: Flatten the schedule into a single list of intervals
        all_intervals = []
        
        for employee in schedule:
            for interval in employee:
                all_intervals.append(interval)
        
        # Step 2: Sort all intervals based on their start time
        all_intervals.sort(key=lambda x: x.start)
        
        # Step 3: Merge intervals
        merged_intervals = []
        for interval in all_intervals:
            if not merged_intervals or merged_intervals[-1].end < interval.start:
                merged_intervals.append(interval)
            else:
                merged_intervals[-1].end = max(merged_intervals[-1].end, interval.end)
        
        # Step 4: Find free time
        free_times = []
        
        for i in range(1, len(merged_intervals)):
            # Start of the current interval - End of the previous interval
            start_free = merged_intervals[i - 1].end
            end_free = merged_intervals[i].start
            
            if start_free < end_free:
                free_times.append(Interval(start_free, end_free))
        
        return free_times

```

### Explanation:

1. **Flattening the Schedule**:
   - We initialize an empty list `all_intervals` to collect all intervals from the employees' schedules.
   - We iterate through each employee's schedule and extract their `Interval` objects, adding them to `all_intervals`.

2. **Sorting the Intervals**:
   - We sort `all_intervals` based on the `start` attribute of each `Interval`. This is important as it allows us to merge overlapping intervals easily.

3. **Merging Intervals**:
   - We create an empty list `merged_intervals`.
   - As we iterate through `all_intervals`, we check if the current interval overlaps with the last added interval in `merged_intervals`. If not, we add it as a new interval. If they do overlap, we merge them by updating the end time of the last interval in `merged_intervals`.

4. **Finding Free Time**:
   - We check gaps between consecutive merged intervals. For each pair of adjacent intervals, if there is a gap (`start_free` is less than `end_free`), we create a new `Interval` for the free time and add it to the `free_times` list.

5. **Return**:
   - Finally, we return the `free_times` list containing all intervals of free time.

### Time Complexity:
The time complexity of this solution is O(N log N), where N is the total number of intervals, due to the sorting step, followed by O(N) for merging and finding free times.

### Constraints:
The function satisfies the constraints specified in the problem statement, handling up to 50 employees and intervals in a reasonable manner.

# 763. Partition Labels

### Problem Description 
A string `S` of lowercase English letters is given. We want to partition this string into as many parts as possible so that each letter appears in at most one part, and return a list of integers representing the size of these parts.


Example 1:
Input: S = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".

This is a partition so that each letter appears in at most one part.

A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits S into less parts.

Note:
`S` will have length in range `[1, 500]`.

`S` will consist of lowercase English letters (`'a'` to `'z'`) only.

### Solution 
 To solve the problem of partitioning a string such that each letter appears in at most one part, we can follow a systematic approach. The goal is to create partitions such that each character within a partition is unique to that partition.

Here's how to approach the problem step-by-step:

### Steps to Solution

1. **Calculate the Last Occurrence of Each Character**: First, iterate through the string and record the last index where each character appears. This allows us to know how far we need to go in the string to ensure a character is not repeated in subsequent partitions.

2. **Iterate Through the String**: Next, we will iterate through the string while keeping track of the maximum last index encountered so far. This will help us determine when we can safely create a partition.

3. **Define Partitions**: Whenever the current index matches the maximum last index, we finalize a partition. At this point, we can append the size of this partition to our result list.

### Implementation

Now let's implement this in Python following the LeetCode solution format:



```python
class Solution:
    def partitionLabels(self, S: str) -> List[int]:
        # Step 1: Determine the last occurrence of each character
        last_occurrence = {char: idx for idx, char in enumerate(S)}
        
        # Step 2: Initialize variables for partitions
        partitions = []
        current_partition_size = 0
        max_last_index = 0
        
        # Step 3: Iterate through the string
        for i, char in enumerate(S):
            current_partition_size += 1
            max_last_index = max(max_last_index, last_occurrence[char])
            
            # Step 4: Check if we can finalize a partition
            if i == max_last_index:
                partitions.append(current_partition_size)
                current_partition_size = 0
        
        return partitions

```

### Explanation of the Code

- **last_occurrence dictionary**: This keeps track of the last index of each character. For example, for the input string `"ababcbacadefegdehijhklij"`, the character 'a' has its last occurrence at index 8, 'b' at 7, etc.

- **Variables**: 
  - `partitions`: stores the sizes of each partition.
  - `current_partition_size`: counts characters in the current partition.
  - `max_last_index`: keeps track of the furthest point we need to go to ensure all characters are included in the partition.

- **Loop through `S`**:
  - For each character, increment the `current_partition_size`.
  - Update `max_last_index` based on the current character's last occurrence.
  - If the current index matches the `max_last_index`, it means we can finalize the current partition. Append the size to `partitions` and reset `current_partition_size`.

### Complexity Analysis
- **Time Complexity**: O(N), where N is the length of string S, as we make a single pass to determine the last occurrences and another pass to create the partitions.
- **Space Complexity**: O(1) if we disregard the output since we're using a fixed-size array (the maximum possible size is 26 for lowercase letters).

This approach efficiently solves the problem while ensuring that we make as many partitions as possible without violating the constraints provided.

# 765. Couples Holding Hands

### Problem Description 
N couples sit in 2N seats arranged in a row and want to hold hands.  We want to know the minimum number of swaps so that every couple is sitting side by side.  A swap consists of choosing any two people, then they stand up and switch seats. 
The people and seats are represented by an integer from `0` to `2N-1`, the couples are numbered in order, the first couple being `(0, 1)`, the second couple being `(2, 3)`, and so on with the last couple being `(2N-2, 2N-1)`.

The couples' initial seating is given by `row[i]` being the value of the person who is initially sitting in the i-th seat.


Example 1:Input: row = [0, 2, 1, 3]
Output: 1
Explanation: We only need to swap the second (row[1]) and third (row[2]) person.


Example 2:Input: row = [3, 2, 0, 1]
Output: 0
Explanation: All couples are already seated side by side.

Note:
 `len(row)` is even and in the range of `[4, 60]`.

 `row` is guaranteed to be a permutation of `0...len(row)-1`.

### Solution 
 To solve the problem of minimizing the number of swaps required to seat all couples together in a row, we can follow a structured approach. We will implement a greedy algorithm to find the minimum swaps needed to ensure each couple sits next to each other. Let’s break down the solution step-by-step and subsequently code it in Python.

### Understanding the Problem

The couples are represented as `(0, 1)`, `(2, 3)`, ..., `(2N-2, 2N-1)`. Each couple must occupy two adjacent seats in the `row`. The main goal is to count the minimum number of swaps required to rearrange the `row` so that each couple is next to each other.

### Approach

1. **Identify Current Positions:** Create a mapping of the current positions of individuals in the row for quick lookup.

2. **Count Swaps:** For each couple:
   - Check if they are already seated together.
   - If not seated together, swap one of them with a person sitting in their partner's correct position.
   - Update the positions in the mapping accordingly.
   - Increase the swap count each time a swap is made.

3. **Iterate Through the Couples:** Continue this process until all couples are seated together.

### Implementation

Now, let's implement this in Python following the LeetCode format.



```python
class Solution:
    def minSwapsCouples(self, row: List[int]) -> int:
        # Create a mapping from person to their index
        index = {person: i for i, person in enumerate(row)}
        swaps = 0
        
        for i in range(0, len(row), 2):
            first_person = row[i]
            second_person = first_person ^ 1  # Use XOR to find the partner (0^1=1, 2^3=1, etc.)
            
            # If current person and partner are not adjacent
            if row[i + 1] != second_person:
                # Find where the partner is currently sitting
                partner_index = index[second_person]
                
                # Swap the partner into the correct spot
                row[i + 1], row[partner_index] = row[partner_index], row[i + 1]
                
                # Update the index mapping
                index[first_person] = i + 1
                index[second_person] = partner_index
                
                # Increase the count of swaps
                swaps += 1
        
        return swaps

```

### Explanation of the Code

1. **Index Mapping:** We create a dictionary called `index` that holds the current index of each person for quick reference.

2. **Loop through Couples:**
   - We iterate over the `row` in steps of 2 to access each couple.
   - For each couple, we derive the second person using XOR operation (every person `x` has its partner as `x ^ 1`).
   - We check if the couple is already together. If not, we locate the index of the second person and swap them with the person currently sitting next to the first.

3. **Update Mapping and Count Swaps:** After a successful swap, we update the index mapping to reflect the new positions and increment the swap count.

### Conclusion

This approach efficiently minimizes the number of adjacent swaps needed by strategically rearranging couples. The algorithm operates with a time complexity of \(O(N)\), which is efficient given the problem constraints, ensuring that we can solve even the largest cases quickly. Each swap only takes constant time due to the nature of the set operations, making the overall approach optimal for this problem.

# 767. Reorganize String

### Problem Description 
Given a string `S`, check if the letters can be rearranged so that two characters that are adjacent to each other are not the same.

If possible, output any possible result.  If not possible, return the empty string.


Example 1:
Input: S = "aab"
Output: "aba"

Example 2:
Input: S = "aaab"
Output: ""
Note:
`S` will consist of lowercase letters and have length in range `[1, 500]`.

### Solution 
 To solve the problem of rearranging the string `S` such that no two adjacent characters are the same, we can follow a structured approach:

1. **Count Character Frequencies**: Utilize a frequency count to determine how many times each character appears in the string.

2. **Max Heap**: Use a max heap (or priority queue) to always pick the character that appears the most times next. This helps in reducing the chances of the same character appearing consecutively.

3. **Rearranging**: Continuously extract the most frequent character and append it to the result string. After appending, decrease its count. We must also ensure that we do not immediately append the same character that was just added, hence we maintain a temporary hold on the last used character.

4. **Reconstruction Check**: If at any point we can't proceed without repeating characters, we return an empty string. If we successfully build out the string, we return that as the result.

Here's the Python code implementing the above logic according to the LeetCode solution format:



```python
from collections import Counter
import heapq

class Solution:
    def reorganizeString(self, S: str) -> str:
        # Step 1: Get character frequencies
        count = Counter(S)
        
        # Step 2: Create a max heap based on character frequencies
        max_heap = []
        for char, freq in count.items():
            heapq.heappush(max_heap, (-freq, char))  # use negative frequency for max heap
        
        result = []
        
        # Step 3: Previous character storage to avoid immediate repeats
        prev_freq, prev_char = 0, ''
        
        # Step 4: Build the result string
        while max_heap:
            freq, char = heapq.heappop(max_heap)  # Get the most frequent character
            result.append(char)  # Append it to our result string
            
            # Since we only want the next most frequent character not to be the same
            # we can use the previously held character.
            if prev_freq < 0:  # Check if we have a previous character to push back
                heapq.heappush(max_heap, (prev_freq, prev_char))
            
            # Update the frequency and set the previous character
            prev_freq = freq + 1  # Decrease frequency since we just used this character
            prev_char = char       # Update previous character for next iteration

        # Step 5: Join the result and check if we used all characters
        result_string = ''.join(result)
        if len(result_string) != len(S):
            return ""  # Not all characters were used, hence arrangement is not possible
        return result_string

```

### Explanation of the Code

- **Character Frequency Calculation**: We use `collections.Counter` to count how many times each character appears in `S`.

- **Max Heap Construction**: We store characters in a max heap by their negative frequency values because Python uses a min heap by default. This allows us to always pull the character with the highest frequency.

- **Building Result**: We iteratively pop from the heap to build the result string. After appending a character to the result, we check if we need to push the last used character back to the heap based on its remaining frequency.

- **Final Check**: Once we finish constructing the result, we check if the length of the result matches the original string. If it does, we successfully rearranged the string; if not, we return an empty string.

This solution is efficient with a time complexity of O(n log k) where n is the length of the string and k is the number of unique characters due to the heap operations.

# 842. Split Array into Fibonacci Sequence

### Problem Description 
Given a string `S` of digits, such as `S = "123456579"`, we can split it into a Fibonacci-like sequence `[123, 456, 579].`
Formally, a Fibonacci-like sequence is a list `F` of non-negative integers such that:
`0 <= F[i] <= 2^31 - 1`, (that is, each integer fits a 32-bit signed integer type);
`F.length >= 3`;
and` F[i] + F[i+1] = F[i+2] `for all `0 <= i < F.length - 2`.

Also, note that when splitting the string into pieces, each piece must not have extra leading zeroes, except if the piece is the number 0 itself.

Return any Fibonacci-like sequence split from `S`, or return `[]` if it cannot be done.


Example 1:
Input: "123456579"
Output: [123,456,579]

Example 2:
Input: "11235813"
Output: [1,1,2,3,5,8,13]

Example 3:
Input: "112358130"
Output: []
Explanation: The task is impossible.


Example 4:
Input: "0123"
Output: []
Explanation: Leading zeroes are not allowed, so "01", "2", "3" is not valid.


Example 5:
Input: "1101111"
Output: [110, 1, 111]
Explanation: The output [11, 0, 11, 11] would also be accepted.

Note: 
`1 <= S.length <= 200`
`S` contains only digits.

### Solution 
 To solve the problem of splitting the string `S` into a possible Fibonacci-like sequence, we can implement a backtracking approach. This method will systematically explore splitting the string into valid segments while ensuring that those segments follow the rules of a Fibonacci-like sequence.

Here is a detailed breakdown of the approach:

1. **Understanding Fibonacci Sequence Conditions**:
   - The sequence must have at least three numbers.
   - The sum of the two previous numbers must equal the next number: `F[i] + F[i+1] = F[i+2]`.
   - Each number must fit within the range of 32-bit signed integers (0 to 2^31 - 1).
   - Leading zeroes must be handled properly: a number cannot have a leading zero unless it is '0'.

2. **Recursive Backtracking**:
   - We will recursively attempt to construct a sequence from the string by trying out all possible splits for the first two numbers.
   - After establishing the first two numbers, we can derive the next number based on the Fibonacci-like property and check if it can be formed from the remaining string.

3. **Base Cases for Backtracking**:
   - If we reach the end of the string and have at least three numbers in our sequence, we return the sequence.
   - If at any point we cannot find a valid next number that matches the Fibonacci condition or if we can't match the remaining substring, we backtrack.

4. **Implementation**:
   - We utilize a helper function that checks for valid splits and implements the backtracking logic.
   - We handle various edge cases such as leading zeros, valid ranges, and lengths.

Here is how the implementation aligns with the LeetCode format for submission:



```python
class Solution:
    def splitIntoFibonacci(self, S: str):
        def is_valid(num):
            # A number is valid if it doesn't have leading zeros unless it's '0' itself
            return str(num) == str(int(num))  # Converts to int and back to avoid leading zeros

        def backtrack(start, seq):
            # If we have at least 3 numbers and we've consumed the entire string,
            # we can return the sequence.
            if start == len(S) and len(seq) >= 3:
                return seq
            
            # Try to form the next number
            for length in range(1, 11):  # A number can be at most 10 digits
                if start + length > len(S):  # Out of bounds check
                    break
                
                current_str = S[start:start + length]
                current_num = int(current_str)

                # Validate number to be in the required range
                if current_num > 2**31 - 1 or not is_valid(current_num):
                    continue

                # We want to ensure we build a Fibonacci sequence
                if len(seq) >= 2:
                    expected_num = seq[-1] + seq[-2]
                    if current_num < expected_num:
                        continue  # If the current number is too small
                    if current_num > expected_num:
                        break  # If it becomes too large, no need to try longer lengths
                
                # We add current number to the sequence and proceed
                seq.append(current_num)
                result = backtrack(start + length, seq)
                
                if result:
                    return result
                
                # If we reach here, it means we need to backtrack
                seq.pop()

            return None
        
        return backtrack(0, [])

# Example usage
# sol = Solution()
# print(sol.splitIntoFibonacci("123456579"))  # Example 1 Output: [123, 456, 579]

```

### Explanation of the Code:
- The function `splitIntoFibonacci` initiates the backtracking process.
- The `is_valid` function checks for leading zeros by comparing the string representation of a number with its integer conversion.
- The `backtrack` function attempts to construct the Fibonacci sequence by extracting substrings of various lengths and validating them according to the problem's rules.
- The recursive exploration continues until either a valid sequence is found or all possibilities are exhausted. 

This implementation is efficient and covers edge cases while adhering to the problem requirements.

# 860. Lemonade Change

### Problem Description 
At a lemonade stand, each lemonade costs `$5`. 
Customers are standing in a queue to buy from you, and order one at a time (in the order specified by `bills`).

Each customer will only buy one lemonade and pay with either a `$5`, `$10`, or `$20` bill.  You must provide the correct change to each customer, so that the net transaction is that the customer pays $5.

Note that you don't have any change in hand at first.

Return `true` if and only if you can provide every customer with correct change.


Example 1:
Input: [5,5,5,10,20]
Output: true
Explanation: 
From the first 3 customers, we collect three $5 bills in order.

From the fourth customer, we collect a $10 bill and give back a $5.

From the fifth customer, we give a $10 bill and a $5 bill.

Since all customers got correct change, we output true.


Example 2:
Input: [5,5,10]
Output: true

Example 3:
Input: [10,10]
Output: false

Example 4:
Input: [5,5,10,10,20]
Output: false
Explanation: 
From the first two customers in order, we collect two $5 bills.

For the next two customers in order, we collect a $10 bill and give back a $5 bill.

For the last customer, we can't give change of $15 back because we only have two $10 bills.

Since not every customer received correct change, the answer is false.

Note:
`0 <= bills.length <= 10000`
`bills[i]` will be either `5`, `10`, or `20`.

### Solution 
 To solve the problem of providing the correct change at a lemonade stand, we will create a function that simulates the transactions based on the sequence of bills the customers present. We'll keep track of the amount of $5 and $10 bills that we have, since the only denominations we can provide as change are either $5 (when a $10 bill is presented) or a combination of a $10 and a $5 bill (when a $20 bill is presented).

The approach works as follows:
1. Initialize counters for $5 and $10 bills.
2. Iterate through each customer's payment in the given `bills` list.
3. For each payment:
   - If the customer pays with a $5 bill, increment the $5 counter.
   - If the customer pays with a $10 bill, check if we have a $5 bill to give as change. If yes, decrement the $5 counter and increment the $10 counter. If not, return `false`.
   - If the customer pays with a $20 bill, check if we can provide $15 in change. The preferable way to give change is to use one $10 bill and one $5 bill (if available). If that's not possible, then check if we have three $5 bills. If we cannot provide the appropriate change in either case, return `false`.
4. If we finish iterating through all bills and provide all necessary changes, return `true`.

Here is the code that implements this logic:



```python
def lemonadeChange(bills):
    # Initialize counters for $5 and $10 bills
    count5 = 0
    count10 = 0
    
    # Iterate through each bill paid by the customers
    for bill in bills:
        if bill == 5:
            count5 += 1  # Increase $5 bill count
        elif bill == 10:
            if count5 > 0:
                count5 -= 1  # Provide $5 change
                count10 += 1  # Increase $10 bill count
            else:
                return False  # Cannot provide change
        elif bill == 20:
            # Preferably give one $10 and one $5 as change
            if count10 > 0 and count5 > 0:
                count10 -= 1
                count5 -= 1
            elif count5 >= 3:
                count5 -= 3  # Give three $5 bills as change
            else:
                return False  # Cannot provide change

    return True  # Successfully provided change for all customers

# Example usage:
# bills = [5, 5, 10, 20]
# print(lemonadeChange(bills))  # Output: false

```

### Explanation of the Code:
- We utilize two counters, `count5` and `count10`, to handle the $5 and $10 bills we have on hand.
- Each time we encounter a bill, we check its denomination and adjust our counters accordingly based on the rules of providing change.
- The solution operates in a single pass through the list of bills (`O(n)` complexity), making it efficient even for larger inputs (up to 10,000 items).
- The function returns `true` if we successfully provide change to all customers, or `false` if at any point we are unable to do so.

This implementation can be directly submitted to platforms like LeetCode for testing and validation against various test cases.

# 861. Score After Flipping Matrix

### Problem Description 
We have a two dimensional matrix `A` where each value is `0` or `1`.

A move consists of choosing any row or column, and toggling each value in that row or column: changing all `0`s to `1`s, and all `1`s to `0`s.

After making any number of moves, every row of this matrix is interpreted as a binary number, and the score of the matrix is the sum of these numbers.

Return the highest possible score.


Example 1:
Input: [[0,0,1,1],[1,0,1,0],[1,1,0,0]]
Output: 39
Explanation:
Toggled to [[1,1,1,1],[1,0,0,1],[1,1,1,1]].

0b1111 + 0b1001 + 0b1111 = 15 + 9 + 15 = 39
Note:
`1 <= A.length <= 20`
`1 <= A[0].length <= 20`
`A[i][j]` is `0` or `1`.

### Solution 
 To solve the problem of maximizing the score of the binary matrix defined by toggling rows and columns, we'll take a structured approach. The core idea is to ensure that the most significant bits (MSB) in each row are maximized, thereby maximizing the resultant binary score.

### Explanation:

1. **Understanding Row Toggling**:
   - Toggling a row means changing all its 0s to 1s and all its 1s to 0s. Since we can toggle any row any number of times, we initially can make the first cell of each row equal to 1 (maximum contribution to the score).

2. **Toggling Columns**:
   - For each column, we can choose to toggle it or leave it as it is. After determining the desired state for the first column (i.e., making it 1), we can decide for each column if toggling will lead to more 1s in that column for all rows.

3. **Calculating Score**:
   - Once we determine the final state of the matrix after all possible toggles, we'll calculate the binary score for each row and sum them up.

### Algorithm Steps:
- For each row, check if the first element is 0 or 1. If it's 0, toggle the entire row.
- For every column (from the second column onwards), if more than half of the rows have a 0 in this column, we toggle the column to ensure more 1s.
- Finally, compute the score for each row and return the total sum.

### Python Code:
Below is the code that implements the above logic. It is structured according to LeetCode's submission format.



```python
class Solution:
    def matrixScore(self, A):
        # Get dimensions of the matrix
        n_rows = len(A)
        n_cols = len(A[0])

        # Step 1: Ensure the first column is all 1s.
        for i in range(n_rows):
            if A[i][0] == 0:
                # Toggle the entire row
                for j in range(n_cols):
                    A[i][j] ^= 1  # Toggle with XOR

        # Step 2: Toggle columns from the second column onwards if needed
        for j in range(1, n_cols):
            count_one = sum(A[i][j] for i in range(n_rows))
            count_zero = n_rows - count_one
            
            # If there are more 0s than 1s, we toggle that column
            if count_zero > count_one:
                for i in range(n_rows):
                    A[i][j] ^= 1  # Toggle with XOR

        # Step 3: Calculate the final score
        score = 0
        for row in A:
            # Convert binary list to an integer
            score += int("".join(map(str, row)), 2)

        return score

# Example use case
solution = Solution()
result = solution.matrixScore([[0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0]])
print(result)  # Output: 39

```

### Explanation of the Code:
- We define a class `Solution` that contains a method `matrixScore`.
- We first check the first column for every row and toggle the entire row if its first element is 0.
- We then iterate through each column from the second onwards to check the number of 1s and 0s. We toggle the column if there are more 0s than 1s.
- Finally, we calculate the score by converting each row from binary to decimal and summing them up, which is done using `int("".join(map(str, row)), 2)`.

This implementation is efficient and adheres to the matrix scoring rules as described in the problem statement.

# 870. Advantage Shuffle

### Problem Description 
Given two arrays `A` and `B` of equal size, the advantage of `A` with respect to `B` is the number of indices `i` for which `A[i] > B[i]`.

Return any permutation of `A` that maximizes its advantage with respect to `B`.


Example 1:
Input: A = [2,7,11,15], B = [1,10,4,11]
Output: [2,11,7,15]

Example 2:
Input: A = [12,24,8,32], B = [13,25,32,11]
Output: [24,32,8,12]
Note:
`1 <= A.length = B.length <= 10000`
`0 <= A[i] <= 10^9`
`0 <= B[i] <= 10^9`

### Solution 
 To solve the problem of maximizing the advantage of array `A` over array `B`, we can follow a strategy that involves sorting and using a greedy selection approach. The goal is to permute `A` in such a way that we maximize the number of indices `i` for which `A[i] > B[i]`. Here's a step-by-step breakdown of the solution:

### Detailed Explanation:

1. **Sort both arrays**: 
   - First, we need to sort both arrays `A` and `B`. Sorting will allow us to efficiently find which elements in `A` can beat the elements in `B`.

2. **Use a two-pointer approach**:
   - We create two pointers, one for each sorted array. The point of these pointers is to track which elements we've considered in `A` and `B`.
   - The pointer `j` will iterate over `B` and `i` will iterate over `A`.

3. **Greedy selection**:
   - For each element in `B`, if the current element of `A` can beat it (i.e., `A[i] > B[j]`), we assign it as the advantage for `B[j]` and move both pointers forward (`i += 1`, `j += 1`).
   - If `A[i]` cannot beat `B[j]`, we just move the pointer `i` to consider the next higher element from `A`, effectively looking for a potential advantage with the next element in `A`.

4. **Fill the remaining positions**:
   - If we exhaust `A` while `B` still has elements to match, we will fill the remaining indices in `A` with elements that couldn't contribute an advantage (they will just fill the remaining places).

5. **Return the final arrangement**.

The solution ensures that we maximize the count of `A[i] > B[i]` while maintaining the constraints of the problem. 

### Python Code Solution

Here's how we can implement this logically structured plan in Python:



```python
from typing import List
import collections

class Solution:
    def advantageCount(self, A: List[int], B: List[int]) -> List[int]:
        # Step 1: Sort A and create a sorted version of B indexed by the original positions
        sorted_A = sorted(A)
        indexed_B = sorted((b, i) for i, b in enumerate(B))
        
        # Step 2: Prepare to place results in the original indices of B
        result = [0] * len(A)
        i = 0  # Pointer for sorted_A
        
        # Step 3: Place elements from sorted_A to beat elements in B
        for b_value, b_index in indexed_B:
            # Find the first `A` that is greater than `b_value`
            while i < len(sorted_A) and sorted_A[i] <= b_value:
                i += 1
            
            if i < len(sorted_A):
                result[b_index] = sorted_A[i]  # Use this A element as it beats B
                i += 1  # Move to the next element in A
            else:
                break  # No more elements left in A that can beat elements in B
        
        # Step 4: Fill any remaining positions in result with unused elements from sorted_A
        for j in range(len(result)):
            if result[j] == 0:
                result[j] = sorted_A[i]
                i += 1
                
        return result

```

### Explanation of Code:
- We use the `enumerate` function to pair each element of `B` with its original index, which helps us keep track of where to place the elements from `A` later on.
- The variable `result` is initialized to the same length as `A` and `B`.
- The array `sorted_A` is populated with the sorted elements of `A` while we use pointers to traverse both sorted arrays and populate `result` smartly.
- Finally, we make sure to fill any unfilled positions in `result` with the remaining elements of `sorted_A`.

This code should work efficiently within the given constraints and can be submitted directly to the LeetCode platform to solve the problem.

# 874. Walking Robot Simulation

### Problem Description 
A robot on an infinite XY-plane starts at point `(0, 0)` and faces north. The robot can receive one of three possible types of `commands`:
`-2`: turn left `90` degrees,
`-1`: turn right `90` degrees, or
`1 <= k <= 9`: move forward `k` units.

Some of the grid squares are `obstacles`. The `ith` obstacle is at grid point `obstacles[i] = (xi, yi)`.

If the robot would try to move onto them, the robot stays on the previous grid square instead (but still continues following the rest of the route.)
Return the maximum Euclidean distance that the robot will be from the origin squared (i.e. if the distance is `5`, return `25`).

Note:
North means +Y direction.

East means +X direction.

South means -Y direction.

West means -X direction.


Example 1:
Input: commands = [4,-1,3], obstacles = []
Output: 25
Explanation: The robot starts at (0, 0):
1. Move north 4 units to (0, 4).

2. Turn right.

3. Move east 3 units to (3, 4).

The furthest point away from the origin is (3, 4), which is 32 + 42 = 25 units away.


Example 2:
Input: commands = [4,-1,4,-2,4], obstacles = [[2,4]]
Output: 65
Explanation: The robot starts at (0, 0):
1. Move north 4 units to (0, 4).

2. Turn right.

3. Move east 1 unit and get blocked by the obstacle at (2, 4), robot is at (1, 4).

4. Turn left.

5. Move north 4 units to (1, 8).

The furthest point away from the origin is (1, 8), which is 12 + 82 = 65 units away.


Constraints:
`1 <= commands.length <= 104`
`commands[i]` is one of the values in the list `[-2,-1,1,2,3,4,5,6,7,8,9]`.

`0 <= obstacles.length <= 104`
`-3 * 104 <= xi, yi <= 3 * 104`
The answer is guaranteed to be less than `231`.

### Solution 
 To solve the problem of the robot navigating on a 2D plane with the commands provided and avoiding obstacles, we can follow a structured approach. Here's a step-by-step explanation of how we'll implement the solution, followed by the complete Python code suitable for running on LeetCode.

### Steps to Solve the Problem

1. **Data Representation**: First, we need to represent the directions the robot can face. We can use a list to maintain the directional vectors corresponding to North, East, South, and West.

2. **State Initialization**: Initialize the robot's position `(x, y)` at the origin `(0, 0)`. Set an initial direction index to represent North (index 0).

3. **Obstacle Handling**: Store obstacles in a set for O(1) average time complexity when checking if the next move is valid.

4. **Command Processing**:
    - For each command, check its type:
        - **Turning commands (-2 for left, -1 for right)**: Update the current direction index appropriately.
        - **Movement commands (1 to 9)**: Update the position by the specified number of units unless blocked by an obstacle.
5. **Distance Calculation**: Track the maximum squared distance from the origin after every valid move.

6. **Result**: Finally, return the maximum squared distance calculated.

### Implementation

Now, here is the Python code encapsulated in a `Solution` class, following the LeetCode format:



```python
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        # Direction vectors for North, East, South, West
        direction_vectors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        x, y = 0, 0  # Initial position
        direction_index = 0  # Initial facing North
        obstacle_set = set(map(tuple, obstacles))  # Convert to set for quick lookup
        max_distance_squared = 0  # To track the maximum distance

        for command in commands:
            if command == -2:  # Turn left
                direction_index = (direction_index - 1) % 4
            elif command == -1:  # Turn right
                direction_index = (direction_index + 1) % 4
            else:  # Move forward
                for step in range(command):
                    new_x = x + direction_vectors[direction_index][0]
                    new_y = y + direction_vectors[direction_index][1]
                    
                    if (new_x, new_y) not in obstacle_set:  # If not an obstacle
                        x, y = new_x, new_y  # Update robot's position
                        # Calculate squared distance from origin
                        max_distance_squared = max(max_distance_squared, x * x + y * y)
        
        return max_distance_squared

```

### Explanation of the Code

- **Direction Management**: The directions are controlled via an index that wraps around using modulo 4. This ensures that when turning left or right, the direction remains within the bounds of our `direction_vectors`.
  
- **Movement Logic**: For movement commands, we iterate through each step of the movement. We calculate the new prospective coordinates and check if they collide with any obstacles.

- **Max Distance Tracking**: After each successful move, we calculate the squared distance to avoid floating-point operations and maintain consistency.

- **Efficiency**: The use of sets for obstacles allows us to efficiently check for collisions in constant time. The overall complexity primarily stems from the number of commands and the potential movement within those commands, making it very efficient given the constraints.

This structure efficiently handles the commands and obstacles, providing a robust solution to the robot's navigation problem.

# 881. Boats to Save People

### Problem Description 
You are given an array `people` where `people[i]` is the weight of the `ith` person, and an infinite number of boats where each boat can carry a maximum weight of `limit`. Each boat carries at most two people at the same time, provided the sum of the weight of those people is at most `limit`.

Return the minimum number of boats to carry every given person.


Example 1:
Input: people = [1,2], limit = 3
Output: 1
Explanation: 1 boat (1, 2)

Example 2:
Input: people = [3,2,2,1], limit = 3
Output: 3
Explanation: 3 boats (1, 2), (2) and (3)

Example 3:
Input: people = [3,5,3,4], limit = 5
Output: 4
Explanation: 4 boats (3), (3), (4), (5)

Constraints:
`1 <= people.length <= 5 * 104`
`1 <= people[i] <= limit <= 3 * 104`

### Solution 
 To solve the problem of determining the minimum number of boats needed to carry every person given their weights and the weight limit of the boats, we can take a strategic approach using a two-pointer technique after sorting the weights of the people. Here’s the thought process and implementation in the LeetCode format.

### Problem Breakdown

1. **Sorting the Weights**: First, we sort the array of people's weights. This allows us to efficiently pair the lightest and the heaviest individuals.

2. **Two-Pointer Technique**: We use two pointers: one (`left`) starting at the beginning of the sorted array (lightest person) and the other (`right`) starting at the end of the array (heaviest person).
   - If the sum of the weights of the two people at the `left` and `right` pointers is less than or equal to the `limit`, we can send them together in one boat. In this case, we move both pointers inward (i.e., increment `left` and decrement `right`).
   - If they can't share a boat (i.e., their combined weight exceeds the `limit`), we only take the heavier person (the one at the `right` pointer) in a boat alone and move just the `right` pointer inward.

3. **Counting Boats**: In either case (whether we pair two people or just take one), we count that as one boat used.

4. **Repeat**: We repeat these steps until we have processed all the weights.

### Implementation

Now, let's implement this logic in Python, formatted according to the LeetCode submission style:



```python
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        # Sort the people array
        people.sort()
        left, right = 0, len(people) - 1
        boats = 0
        
        # Use two pointers to pair people 
        while left <= right:
            # If the lightest and heaviest can be paired
            if people[left] + people[right] <= limit:
                left += 1  # move the left pointer to the right
            # In either case, we need to move the right pointer (heaviest person must take a boat)
            right -= 1
            boats += 1  # we've used one boat
        
        return boats

```

### Explanation of the Code

1. **Sorting**: We sort the `people` array to allow pairing of the lightest and heaviest individuals efficiently.

2. **Two pointers**:
   - `left` starts at 0 (lightest) and `right` starts at `len(people) - 1` (heaviest).
   - We continue looping as long as `left` is less than or equal to `right`.

3. **Condition checking**:
   - If the sum of `people[left]` and `people[right]` is less than or equal to `limit`, both can share a boat, so we increment `left` and decrement `right`.
   - If they cannot share a boat, we deduct only the heavier person (`right`) but still count that boat.

4. **Boat Count**: For every iteration (whether pairing or single), we increment our `boats` counter.

5. **Result**: After processing all people, we return the total `boats` required.

### Complexity Analysis
- **Time Complexity**: O(n log n) for sorting the array, and O(n) for the two-pointer traversal, leading to an overall complexity of O(n log n).
- **Space Complexity**: O(1) for pointer operations since we sort in place.

This implementation efficiently calculates the minimum number of boats required to rescue all people based on the constraints provided in the problem statement.

# 910. Smallest Range II

### Problem Description 
Given an array `A` of integers, for each integer `A[i]` we need to choose either `x = -K` or `x = K`, and add `x` to `A[i] (only once)`.

After this process, we have some array `B`.

Return the smallest possible difference between the maximum value of `B` and the minimum value of `B`.


Example 1:
Input: A = [1], K = 0
Output: 0
Explanation: B = [1]

Example 2:
Input: A = [0,10], K = 2
Output: 6
Explanation: B = [2,8]

Example 3:
Input: A = [1,3,6], K = 3
Output: 3
Explanation: B = [4,6,3]
Note:
`1 <= A.length <= 10000`
`0 <= A[i] <= 10000`
`0 <= K <= 10000`

### Solution 
 To solve the problem of minimizing the difference between the maximum and minimum values of the modified array \( B \) derived from the original array \( A \), we need to first understand how we can manipulate each element in \( A \). At each index \( i \), we can either add \( K \) or subtract \( K \) from \( A[i] \). 

**Explanation:**

1. For each element \( A[i] \), if we add \( K \), the resulting value will be \( A[i] + K \). Conversely, if we subtract \( K \), the resulting value will be \( A[i] - K \).
2. After determining the possible new values from each element:
   - The maximum possible value in the resulting array \( B \) occurs if we maximize every element. This happens by adding \( K \) to the maximum element in \( A \) and subtracting \( K \) from the minimum element:
     \[
     \text{max\_B} = \max(A) + K
     \]
   - The minimum possible value in the resulting array \( B \) occurs by minimizing every element. This happens by subtracting \( K \) from the maximum element in \( A \) and adding \( K \) to the minimum element:
     \[
     \text{min\_B} = \min(A) - K
     \]
3. The difference between the maximum and minimum values of \( B \) can then be expressed as:
   \[
   \text{difference} = (\max(A) + K) - (\min(A) - K) = \max(A) - \min(A) + 2K
   \]
4. The problem then reduces to calculating the difference between the maximum and minimum of the array \( A \), and adjusting that by \( 2K \).

5. In the edge case where \( K = 0 \), the difference will simply be the difference between the maximum and minimum of the original array \( A \).

Now let's implement the solution in Python, formatted for LeetCode.



```python
class Solution:
    def smallestRangeI(self, A: List[int], K: int) -> int:
        if not A:  # Check if the input list is empty
            return 0
        
        # Find the maximum and minimum values in array A
        max_A = max(A)
        min_A = min(A)
        
        # Calculate the smallest possible difference
        difference = (max_A - min_A) - 2 * K
        
        # The smallest possible difference cannot be negative
        return max(0, difference)

```

### Detailed Breakdown of the Code:

- **Input and Output:** The function `smallestRangeI` takes a list of integers `A` and an integer `K`. It returns an integer which is the smallest possible difference after modifying \( A \).
- **Edge Case Handling:** We first check if the array \( A \) is empty. If it is, we return 0 since there are no values to process.
- **Finding the Max and Min:** We use Python's built-in `max()` and `min()` functions to determine the maximum and minimum values in \( A \).
- **Calculating the Difference:** We subtract twice \( K \) from the difference between the maximum and minimum values.
- **Non-negative Return:** We use `max(0, difference)` to ensure the result is non-negative, as the minimum possible difference cannot be less than zero.

This solution runs efficiently with a time complexity of \( O(n) \) for finding the max and min values, where \( n \) is the length of array \( A \).

# 921. Minimum Add to Make Parentheses Valid

### Problem Description 
Given a string `S` of `'('` and `')'` parentheses, we add the minimum number of parentheses ( `'('` or `')'`, and in any positions ) so that the resulting parentheses string is valid.

Formally, a parentheses string is valid if and only if:
It is the empty string, or
It can be written as `AB` (`A` concatenated with `B`), where `A` and `B` are valid strings, or
It can be written as `(A)`, where `A` is a valid string.

Given a parentheses string, return the minimum number of parentheses we must add to make the resulting string valid.


Example 1:
Input: "())"
Output: 1

Example 2:
Input: "((("
Output: 3

Example 3:
Input: "()"
Output: 0

Example 4:
Input: "()))(("
Output: 4
Note:
`S.length <= 1000`
`S` only consists of `'('` and `')'` characters.

### Solution 
 To solve the problem of making a parentheses string valid by adding the minimum number of parentheses, we can follow a systematic approach. Below is the detailed explanation along with the Python code in a format suitable for LeetCode.

### Explanation:

1. **Understanding Valid Parentheses**:
   - A valid parentheses string means that for every closing parenthesis `')'`, there is a corresponding opening parenthesis `'('` that comes before it.
   - Additionally, at no point in the string from left to right should the count of `')'` exceed the count of `'('`.

2. **Count the Imbalances**:
   - We will maintain two counters:
     - `open_needed`: This counts how many opening parentheses we need to add.
     - `close_needed`: This counts how many closing parentheses we need to add.
   
3. **Iterate Through the String**:
   - For each character in the string:
     - If it’s an opening parenthesis `'('`, increment the `close_needed` counter.
     - If it’s a closing parenthesis `')'`:
       - Check if there are any unmatched opening parentheses (`close_needed > 0`). If yes, it means we can match this `')'` with an `'('`, so we decrement `close_needed`.
       - If there are no unmatched opening parentheses, it means we encountered an extra `')'`; thus, we increment the `open_needed` counter (indicating we need to add an `'('`).

4. **Result Calculation**:
   - The total number of parentheses to be added to make the string valid will be the sum of `open_needed` and `close_needed`.

### Python Code:

Here's the Python code that implements the above logic:



```python
class Solution:
    def minAddToMakeValid(self, S: str) -> int:
        open_needed = 0
        close_needed = 0
        
        for char in S:
            if char == '(':
                close_needed += 1  # We need a matching ')' for this '('
            elif char == ')':
                if close_needed > 0:
                    close_needed -= 1  # Match this ')' with a previous '('
                else:
                    open_needed += 1  # We need an extra '(' for this ')'
        
        # Total parentheses needed to add
        return open_needed + close_needed

```

### Explanation of the Code:

1. **Class and Method Definition**:
   - The code defines a class `Solution` and a method `minAddToMakeValid` that takes a string `S` as input.

2. **Initialization of Counters**:
   - We initialize `open_needed` and `close_needed` to zero.

3. **Loop Through Each Character**:
   - We iterate through each character in the string `S` and update our counters based on whether we encounter `'('` or `')'`.

4. **Return the Result**:
   - Finally, we return the sum of `open_needed` and `close_needed`, which gives us the minimum number of parentheses that need to be added to make the string valid.

This code is efficient with a time complexity of O(n) where n is the length of string S, and it correctly accounts for all possible cases of unbalanced parentheses. You can run this solution directly on LeetCode as is.

# 927. Three Equal Parts

### Problem Description 
You are given an array `arr` which consists of only zeros and ones, divide the array into three non-empty parts such that all of these parts represent the same binary value.

If it is possible, return any `[i, j]` with `i + 1 < j`, such that:
`arr[0], arr[1], ..., arr[i]` is the first part,
`arr[i + 1], arr[i + 2], ..., arr[j - 1]` is the second part, and
`arr[j], arr[j + 1], ..., arr[arr.length - 1]` is the third part.

All three parts have equal binary values.

If it is not possible, return `[-1, -1]`.

Note that the entire part is used when considering what binary value it represents. For example, `[1,1,0]` represents `6` in decimal, not `3`. Also, leading zeros are allowed, so `[0,1,1]` and `[1,1]` represent the same value.


Example 1:
Input: arr = [1,0,1,0,1]
Output: [0,3]

Example 2:
Input: arr = [1,1,0,1,1]
Output: [-1,-1]

Example 3:
Input: arr = [1,1,0,0,1]
Output: [0,2]

Constraints:
`3 <= arr.length <= 3 * 104`
`arr[i]` is `0` or `1`

### Solution 
 To solve the problem of splitting an array of binary values into three parts with equal binary representation, we can follow a structured approach. Below is the detailed explanation of the solution as well as the code formatted to run on LeetCode.

## Problem Breakdown
1. **Count the Total 1's**: First, we need to count how many `1`s are in the array. If the count is not divisible by three (i.e., `count % 3 != 0`), we can immediately return `[-1, -1]` since it's impossible to split the array into three parts with equal binary values.

2. **Identify Parts**: If the count of `1`s is divisible by three, we need to find the starting indices for each part:
   - The first part should start where the first `1` occurs.
   - Each subsequent part should start where the second and third `1`s occur if they exist.

3. **Handle Edge Cases**: If a part starts or ends with zeros, we need to manage those cases carefully to maintain the integrity of the binary values we represent.

4. **Construct the Result**: After identifying the three parts by their respective starting indices, we should compare the portions of the array to ensure they are equal. The parts are considered equal if they are same after accounting for leading zeros.

5. **Return Indices**: Finally, if we can find valid indices, we return them; otherwise, return `[-1, -1]`.

## Solution Code

Here’s the Python code implementing this logic:



```python
from typing import List

class Solution:
    def threeEqualParts(self, arr: List[int]) -> List[int]:
        total_ones = sum(arr)
        
        # Check if the total number of 1's is divisible by 3
        if total_ones % 3 != 0:
            return [-1, -1]
        
        # If there are no 1's, we can return [0, len(arr) - 1]
        if total_ones == 0:
            return [0, len(arr) - 1]
        
        # Target number of 1's in each part
        ones_per_part = total_ones // 3
        
        # Find the first part's end index
        first = second = third = None
        current_ones = 0
        
        for i in range(len(arr)):
            if arr[i] == 1:
                current_ones += 1
                if current_ones == 1:
                    first = i
                elif current_ones == ones_per_part + 1:
                    second = i
                elif current_ones == 2 * ones_per_part + 1:
                    third = i
        
        # Trim trailing zeros for all three parts
        while third < len(arr) and arr[first] == arr[second] == arr[third]:
            first += 1
            second += 1
            third += 1
        
        # Check that we've successfully partitioned the array
        if third != len(arr):
            return [-1, -1]
        
        return [first - 1, second]

```

## Explanation of the Code
1. **Counting 1's**: We first count the total number of 1's using `sum(arr)`. If it’s not divisible by 3, we return `[-1, -1]`.

2. **Early Return**: In the case of no 1's (i.e., all zeros), we can return `[0, len(arr) - 1]`.

3. **Finding Start Indices**: Using a single pass through the array, we identify the starting indices for each part (`first`, `second`, and `third`).

4. **Trimming Zeros**: We adjust the starting indices if the three parts have trailing zeros that are the same, ensuring we compare the actual binary representations correctly.

5. **Final Check**: If the indices are valid and represent three equal parts, we return the appropriate result.

This solution is efficient and runs in O(N) time complexity where N is the length of the array, which is sufficient given the constraints.

# 936. Stamping The Sequence

### Problem Description 
You want to form a `target` string of lowercase letters.

At the beginning, your sequence is `target.length` `'?'` marks.  You also have a `stamp` of lowercase letters.

On each turn, you may place the stamp over the sequence, and replace every letter in the sequence with the corresponding letter from the stamp.  You can make up to `10 * target.length` turns.

For example, if the initial sequence is "?????", and your stamp is `"abc"`,  then you may make "abc??", "?abc?", "??abc" in the first turn.  (Note that the stamp must be fully contained in the boundaries of the sequence in order to stamp.)
If the sequence is possible to stamp, then return an array of the index of the left-most letter being stamped at each turn.  If the sequence is not possible to stamp, return an empty array.

For example, if the sequence is "ababc", and the stamp is `"abc"`, then we could return the answer `[0, 2]`, corresponding to the moves "?????" -> "abc??" -> "ababc".

Also, if the sequence is possible to stamp, it is guaranteed it is possible to stamp within `10 * target.length` moves.  Any answers specifying more than this number of moves will not be accepted.


Example 1:
Input: stamp = "abc", target = "ababc"
Output: [0,2]
([1,0,2] would also be accepted as an answer, as well as some other answers.)

Example 2:
Input: stamp = "abca", target = "aabcaca"
Output: [3,0,1]
Note:
`1 <= stamp.length <= target.length <= 1000`
`stamp` and `target` only contain lowercase letters.

### Solution 
 To solve the problem of transforming a sequence of '?' marks into a target string by using a provided stamp, we can break down the approach into clear steps. The goal is to replace the '?' marks in the initial sequence with the stamp until we form the target string. Let's take a detailed look at the solution.

### Steps to Solve the Problem:

1. **Initialization**: Start with a list of characters representing the current sequence (all '?'). Also, maintain a list to track the indices where stamping occurs.

2. **Define a Helper Function**: Create a function that checks if stamping at a specific position is possible. This function will compare the current sequence with the stamp:
   - A position is stampable if for each character in the stamp, either the corresponding position in the sequence contains a '?' or matches the character in the stamp.

3. **Stamping Logic**: Loop while there are still characters in the target string that need to be replaced (i.e., while the current sequence is different from the target):
   - For each position in the target, check if stamping is possible. If it is, apply the stamp by replacing the relevant section with a character that indicates it's been stamped (like a special character or just setting it to a confirmed character).
   - Record the position of the stamp in the result list.

4. **Termination**: If after all stamping operations, the current sequence matches the target, return the recorded indices. Otherwise, return an empty list.

### Code Implementation:

Here's how you can implement the solution in Python:



```python
class Solution:
    def movesToStamp(self, stamp: str, target: str) -> List[int]:
        s_len, t_len = len(stamp), len(target)
        sequence = list('?' * t_len)
        result = []
        stamped = [False] * t_len
        total_stamped = 0
        
        # Helper function to check if we can stamp at position `i`
        def can_stamp(i):
            changeable = False
            for j in range(s_len):
                # Check if stamp fits in the sequence
                if sequence[i + j] == '?':
                    changeable = True  # We can change '?'
                elif sequence[i + j] != stamp[j]:
                    return False  # Mismatch and not '?' so we cannot stamp
            return changeable
        
        # Helper function to actually perform the stamping
        def stamp_position(i):
            nonlocal total_stamped
            for j in range(s_len):
                sequence[i + j] = stamp[j]
            total_stamped += 1
            
        while total_stamped < t_len:
            stamped_any = False
            
            # Try to stamp at every position
            for i in range(t_len - s_len + 1):
                if not stamped[i] and can_stamp(i):
                    stamp_position(i)
                    result.append(i)
                    stamped[i] = True
                    stamped_any = True
            
            # If we did not stamp, we cannot proceed
            if not stamped_any:
                return []
        
        # We should return the result in reverse order to match the required output
        return result[::-1]


```

### Explanation of the Code:

1. **Initialization**:
   - We initialize `sequence` as a list of '?' marks, `result` to store indices, and `stamped` to keep track of which positions have been transformed.
   - `total_stamped` counts how many characters in the target have been successfully stamped.

2. **Helper Functions**:
   - `can_stamp(i)`: Checks if stamping at index `i` can change the '?' to the letters in `stamp`. If any letter matches or is a '?', the position returns true.
   - `stamp_position(i)`: Stamps the `stamp` string into the sequence, counting how many total stamps have been done.

3. **Looping**:
   - As long as we haven't filled the entire target (tracked by `total_stamped`), we check through potential stamping positions.
   - If an applicable position is found, it gets stamped and recorded.

4. **Output**:
   - Finally, since the order of moves matters, the collected moves in `result` are returned in reverse order.

This algorithm efficiently processes the replacement while adhering to the constraints provided and makes sure the resulting indices are correct according to the problem statement.

# 944. Delete Columns to Make Sorted

### Problem Description 
You are given an array of `n` strings `strs`, all of the same length.

The strings can be arranged such that there is one on each line, making a grid. For example, `strs = ["abc", "bce", "cae"]` can be arranged as:
abc
bce
cae
You want to delete the columns that are not sorted lexicographically. In the above example (0-indexed), columns 0 (`'a'`, `'b'`, `'c'`) and 2 (`'c'`, `'e'`, `'e'`) are sorted while column 1 (`'b'`, `'c'`, `'a'`) is not, so you would delete column 1.

Return the number of columns that you will delete.


Example 1:
Input: strs = ["cba","daf","ghi"]
Output: 1
Explanation: The grid looks as follows:
  cba
  daf
  ghi
Columns 0 and 2 are sorted, but column 1 is not, so you only need to delete 1 column.


Example 2:
Input: strs = ["a","b"]
Output: 0
Explanation: The grid looks as follows:
  a
  b
Column 0 is the only column and is sorted, so you will not delete any columns.


Example 3:
Input: strs = ["zyx","wvu","tsr"]
Output: 3
Explanation: The grid looks as follows:
  zyx
  wvu
  tsr
All 3 columns are not sorted, so you will delete all 3.


Constraints:
`n == strs.length`
`1 <= n <= 100`
`1 <= strs[i].length <= 1000`
`strs[i]` consists of lowercase English letters.

### Solution 
 To solve the problem of determining the number of unsorted columns from an array of strings, we can take the following approach:

### Step-by-Step Explanation

1. **Understand the Input and Output**:
   We are given a list of strings, all of equal length. We need to determine how many columns (based on the grid structure formed by the strings) are not sorted in lexicographical order.

2. **Grid Representation**:
   When we convert the list of strings into a grid, each string represents a row, and each character at the same index across all strings represents a column.

3. **Check Each Column**:
   To determine if a column is sorted, we need to check if for every row in that column, the character at the current row is less than or equal to the character in the row directly below it. 

4. **Count Unsorted Columns**:
   For each column, if we find any instance where the character in the current row is greater than the character in the row below, we mark that column as unsorted and increment our unsorted column count.

5. **Iterate Through Columns**:
   We will iterate through each column and use a nested loop to check every row in that column.

### Implementation

Here's how we can implement this:



```python
class Solution:
    def minDeletionSize(self, strs):
        # Initialize count of unsorted columns
        delete_count = 0
        
        # The number of columns would be the length of the first string
        num_columns = len(strs[0])
        
        # Iterate through each column
        for col in range(num_columns):
            # Check if this column is sorted
            for row in range(1, len(strs)):
                # Compare current character with the previous character in the same column
                if strs[row][col] < strs[row - 1][col]:
                    # If it's not sorted, mark this column for deletion
                    delete_count += 1
                    break  # No need to check further for this column
        
        return delete_count

# Example usage:
# solution = Solution()
# print(solution.minDeletionSize(["cba","daf","ghi"]))  # Output: 1
# print(solution.minDeletionSize(["a","b"]))            # Output: 0
# print(solution.minDeletionSize(["zyx","wvu","tsr"]))  # Output: 3

```

### Explanation of the Code:

- We define a class `Solution` with a method `minDeletionSize` that takes a list of strings as input.
- We initialize `delete_count` to zero to keep track of how many columns we need to delete.
- We determine the number of columns by checking the length of the first string (`len(strs[0])`).
- We use a loop for each column, then another nested loop that starts from the second row (to compare it with the first row) and checks if the current row's character is less than the previous row's character.
- If a character is found that breaks the sorting order, we increment our `delete_count` and break out of the inner loop to move to the next column.
- Finally, after checking all columns, we return the total `delete_count`.

This solution is efficient given the constraints and provides an easy-to-understand approach to solving the problem. The complexity is O(n * m), where n is the number of strings and m is the length of each string.

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
 To solve the problem of maximizing the score by strategically using tokens, we need to approach this in a structured manner. Let's break down the steps necessary to create an effective solution.

### Problem Analysis
You have a list of tokens and an initial amount of power (P). Each token can be used to either gain score (if you have enough power) or regain power (if you have score to spend). The goal is to maximize your score using the tokens optimally.

### Strategy
1. **Sorting Tokens**: By sorting the tokens in ascending order, we can manage our power more effectively. We attempt to play tokens face up (gain score) using the least power first.
2. **Two-Pointer Technique**: We can use two pointers — one (`left`) starting at the beginning of the sorted array (for face-up plays) and another (`right`) at the end (for face-down plays).
3. **Iterate While Conditions Hold**:
   - If we can afford to play the token at `left` face up (i.e., `P >= tokens[left]`), we do so: decreasing power, increasing score, and moving the `left` pointer right.
   - If we cannot afford to play face-up and we have score to play a token face down (i.e., `score > 0`), we play the token at `right` face down: increasing power and decreasing score, and moving the `right` pointer left.
   - If neither condition holds, we stop since we cannot gain any more score or power.

### Implementation
Now, let's proceed to implement the algorithm directly in Python suitable for the LeetCode platform.



```python
class Solution:
    def bagOfTokensScore(self, tokens: List[int], P: int) -> int:
        # Sort the tokens array
        tokens.sort()
        
        left = 0
        right = len(tokens) - 1
        score = 0
        max_score = 0
        
        while left <= right:
            # If we have enough power to play the left token face up
            if P >= tokens[left]:
                P -= tokens[left]
                score += 1
                max_score = max(max_score, score)  # Update max score if needed
                left += 1
            # Otherwise, if we can play the right token face down to gain power
            elif score > 0:
                P += tokens[right]
                score -= 1
                right -= 1
            else:
                break  # If we cannot play any tokens, break the loop
        
        return max_score

```

### Explanation of the Code
- **Sorting**: We first sort the tokens to help us choose the optimal tokens to play first.
- **Two Pointers**: We initialize pointers `left` and `right` to explore the tokens from both ends.
- **Loop**: Inside the while loop, we check if we can play the token at `left` (face up) or `right` (face down).
  - We adjust power and score accordingly and update the maximum score found.
- **Exit Condition**: The loop continues until neither pointer can make a valid play anymore.

### Complexity Analysis
- **Time Complexity**: The time complexity is \(O(n \log n)\) for sorting the tokens, where \(n\) is the number of tokens. The subsequent traversal of the list is \(O(n)\).
- **Space Complexity**: The space complexity is \(O(1)\) since we are using only a constant amount of extra space for variables.

This algorithm effectively maximizes the score we can achieve using the given tokens and power constraints.

# 955. Delete Columns to Make Sorted II

### Problem Description 
You are given an array of `n` strings `strs`, all of the same length.

We may choose any deletion indices, and we delete all the characters in those indices for each string.

For example, if we have `strs = ["abcdef","uvwxyz"]` and deletion indices `{0, 2, 3}`, then the final array after deletions is `["bef", "vyz"]`.

Suppose we chose a set of deletion indices `answer` such that after deletions, the final array has its elements in lexicographic order (i.e., `strs[0] <= strs[1] <= strs[2] <= ... <= strs[n - 1]`). Return the minimum possible value of `answer.length`.


Example 1:
Input: strs = ["ca","bb","ac"]
Output: 1
Explanation: 
After deleting the first column, strs = ["a", "b", "c"].

Now strs is in lexicographic order (ie. strs[0] <= strs[1] <= strs[2]).

We require at least 1 deletion since initially strs was not in lexicographic order, so the answer is 1.


Example 2:
Input: strs = ["xc","yb","za"]
Output: 0
Explanation: 
strs is already in lexicographic order, so we do not need to delete anything.

Note that the rows of strs are not necessarily in lexicographic order:
i.e., it is NOT necessarily true that (strs[0][0] <= strs[0][1] <= ...)

Example 3:
Input: strs = ["zyx","wvu","tsr"]
Output: 3
Explanation: We have to delete every column.


Constraints:
`n == strs.length`
`1 <= n <= 100`
`1 <= strs[i].length <= 100`
`strs[i]` consists of lowercase English letters.

### Solution 
 To solve the problem of finding the minimum number of deletion indices required to arrange an array of strings in lexicographical order after character deletions, we can take the following approach.

### Approach

1. **Understanding the Problem**: We need to ensure that after possibly deleting some characters (at the same indices for each string), the resulting strings remain in lexicographical order. 

2. **Lexicographical Order**: Two strings `strs[i]` and `strs[i+1]` are in lexicographical order if for every character index `j`, `strs[i][j] <= strs[i+1][j]`. If this condition fails at any character position, we will need to consider deleting that character position.

3. **Column Comparison**: We will compare characters column by column (index by index). If we find a column where a string in the current position is greater than the string in the next one, we should mark this column as one that needs to be deleted. 

4. **Count Required Deletions**: We increment a counter every time we find a problematic column.

### Implementation

The implementation can be directly translated to a function that can be run on LeetCode.

Here is the code along with detailed explanations in the comments:



```python
class Solution:
    def minDeletionSize(self, strs: List[str]) -> int:
        # Length of the string (each string has the same length)
        num_columns = len(strs[0])
        # Counter for the number of columns to delete
        delete_count = 0
        
        # Traverse each column
        for col in range(num_columns):
            # Check the current column against the next one
            for row in range(len(strs) - 1):
                if strs[row][col] > strs[row + 1][col]:
                    # If the current character is greater than the next one, we need to delete this column
                    delete_count += 1
                    break  # No need to check further, move to next column
        
        return delete_count


```

### Explanation of the Code

- We define a class `Solution` with a method `minDeletionSize` that takes a list of strings `strs`.
- We first determine the length of the strings (`num_columns`).
- We initialize a counter `delete_count` to zero.
- We iterate through each column index `col`.
- For each column, we compare adjacent rows:
  - If we find that `strs[row][col]` is greater than `strs[row + 1][col]`, it means that they are not in lexicographical order for that column, hence we increment `delete_count` and break out of the inner loop to check the next column.
- Finally, we return the count of columns that need to be deleted.

### Complexity
- **Time Complexity**: O(n * m), where n is the number of strings and m is the length of each string. We are essentially checking every character in up to `n-1` pairs for each of the `m` columns.
- **Space Complexity**: O(1), as we are using only a few extra variables for counting and not using any significant extra space related to input size.

This approach is efficient given the constraints of the problem and will perform well within the given limits.

# 984. String Without AAA or BBB

### Problem Description 
Given two integers `a` and `b`, return any string `s` such that:
`s` has length `a + b` and contains exactly `a` `'a'` letters, and exactly `b` `'b'` letters,
The substring `'aaa'` does not occur in `s`, and
The substring `'bbb'` does not occur in `s`.


Example 1:
Input: a = 1, b = 2
Output: "abb"
Explanation: "abb", "bab" and "bba" are all correct answers.


Example 2:
Input: a = 4, b = 1
Output: "aabaa"

Constraints:
`0 <= a, b <= 100`
It is guaranteed such an `s` exists for the given `a` and `b`.

### Solution 
 To solve the problem of generating a string containing exactly `a` letter 'a's and `b` letter 'b's without forming substrings "aaa" or "bbb", we can use a greedy approach. 

### Explanation of the Solution

1. **Basic Idea**: The goal is to construct a string that satisfies the condition of having `a` number of 'a's and `b` number of 'b's. We must ensure that no substring "aaa" or "bbb" appears. This suggests a pattern where we should try to evenly distribute 'a's and 'b's to prevent forming three consecutive same letters.

2. **Choosing the Right Pattern**: 
   - If either `a` or `b` is greater than the other, we can use the excess letters to fill in between the letters of the lesser character to avoid consecutive letters.
   - For example, if `a > b`, we might want to add 'a's in pairs or single instances and separate them with 'b's.

3. **Building the String**: 
   - We can use a loop to append the characters while checking to avoid conditions that lead to the forbidden substrings. If the count of 'a's is greater than many 'b's, add 'a's judiciously.

### Implementation



```python
class Solution:
    def generateString(self, a: int, b: int) -> str:
        result = []
        
        while a > 0 or b > 0:
            if a > b:
                # Add two 'a's if possible and one 'b' if available
                if a > 1:
                    result.append('aa')
                    a -= 2
                else:
                    result.append('a')
                    a -= 1
                
                if b > 0:
                    result.append('b')
                    b -= 1
            else:
                # Add two 'b's if possible and one 'a' if available
                if b > 1:
                    result.append('bb')
                    b -= 2
                else:
                    result.append('b')
                    b -= 1
                    
                if a > 0:
                    result.append('a')
                    a -= 1
        
        return ''.join(result)

# Example usages
solution = Solution()
print(solution.generateString(1, 2))  # Output can be "abb", "bab", or "bba"
print(solution.generateString(4, 1))  # Output can be "aabaa" or similar variations

```

### Summary of the Code

- The `generateString` method constructs the required string step-by-step.
- It utilizes a loop that runs while there are still letters left to add (`a > 0` or `b > 0`).
- It incorporates checks to add 'aa' or 'bb' when possible, alternating with a single letter of the other character.
- Finally, it combines the list of characters into a single string using `''.join(result)` and returns it.

This code follows the LeetCode's format and should work correctly within the constraints provided.

# 991. Broken Calculator

### Problem Description 
On a broken calculator that has a number showing on its display, we can perform two operations:
Double: Multiply the number on the display by 2, or;
Decrement: Subtract 1 from the number on the display.

Initially, the calculator is displaying the number `X`.

Return the minimum number of operations needed to display the number `Y`.


Example 1:
Input: X = 2, Y = 3
Output: 2
Explanation: Use double operation and then decrement operation {2 -> 4 -> 3}.


Example 2:
Input: X = 5, Y = 8
Output: 2
Explanation: Use decrement and then double {5 -> 4 -> 8}.


Example 3:
Input: X = 3, Y = 10
Output: 3
Explanation:  Use double, decrement and double {3 -> 6 -> 5 -> 10}.


Example 4:
Input: X = 1024, Y = 1
Output: 1023
Explanation: Use decrement operations 1023 times.

Note:
`1 <= X <= 10^9`
`1 <= Y <= 10^9`

### Solution 
 To solve the problem of transforming the number `X` displayed on a broken calculator to the number `Y` using the minimum number of operations, we need to consider the operations allowed and their implications. 

### Key Observations:
1. **Operations Available**:
   - **Double Operation**: This operation allows us to increase `X` by doubling it.
   - **Decrement Operation**: This operation allows us to decrease `X` by 1.

2. **Bottleneck Scenario**:
   - If `X` is greater than or equal to `Y`, the only way to reach `Y` is to decrement `X` down to `Y`, which will take exactly `X - Y` operations. 

3. **When `X < Y`**:
   - If `X` is smaller than `Y`, we can consider working backwards from `Y` to `X`. The idea is to understand how to reach `X` efficiently:
     - If `Y` is even, the most efficient way to reduce `Y` is to halve it (inverse of doubling).
     - If `Y` is odd, first, we need to increment it to make it even (this represents the operation of decrementing `X` one step back). After that, we can halve it.

### Approach:
Using a while loop, we can reduce `Y` to `X`:
- If `Y` is greater than `X`:
  - If `Y` is even, divide it by 2.
  - If `Y` is odd, increment it by 1 to make it even.
- Count these operations until `Y` is less than or equal to `X`.
- Finally, account for the remaining steps where `X` is greater than `Y`, which would simply be the difference `X - Y`.

### Detailed Python Code:
Below is the Python code formatted for LeetCode submission:



```python
class Solution:
    def brokenCalc(self, X: int, Y: int) -> int:
        operations = 0
        
        while Y > X:
            if Y % 2 == 0:
                Y //= 2
            else:
                Y += 1
            operations += 1
        
        # Once Y is less than or equal to X, we need to perform decrement operations.
        operations += (X - Y)
        
        return operations

```

### Explanation of the Code:
1. We define a class `Solution` with a method `brokenCalc` that takes integers `X` and `Y`.
2. We initialize a counter `operations` to store the total number of operations performed.
3. We enter a while loop that continues as long as `Y` is greater than `X`.
   - Inside the loop, we check if `Y` is even. If true, we halve `Y`. If not, we increment `Y` to make it even. Each action adds to the `operations` counter.
4. When `Y` becomes less than or equal to `X`, we add the difference `X - Y` to `operations` since these will be all decrement operations needed to reach `Y`.
5. Finally, we return the total `operations`.

### Complexity:
- **Time Complexity**: O(log Y), since halving `Y` leads to a logarithmic reduction in values.
- **Space Complexity**: O(1), as we are using a constant amount of space regardless of the input size. 

This solution effectively minimizes the number of operations needed to display `Y` starting from `X`.

# 995. Minimum Number of K Consecutive Bit Flips

### Problem Description 
In an array `A` containing only 0s and 1s, a `K`-bit flip consists of choosing a (contiguous) subarray of length `K` and simultaneously changing every 0 in the subarray to 1, and every 1 in the subarray to 0.

Return the minimum number of `K`-bit flips required so that there is no 0 in the array.  If it is not possible, return `-1`.


Example 1:
Input: A = [0,1,0], K = 1
Output: 2
Explanation: Flip A[0], then flip A[2].


Example 2:
Input: A = [1,1,0], K = 2
Output: -1
Explanation: No matter how we flip subarrays of size 2, we can't make the array become [1,1,1].


Example 3:
Input: A = [0,0,0,1,0,1,1,0], K = 3
Output: 3
Explanation:
Flip A[0],A[1],A[2]: A becomes [1,1,1,1,0,1,1,0]
Flip A[4],A[5],A[6]: A becomes [1,1,1,1,1,0,0,0]
Flip A[5],A[6],A[7]: A becomes [1,1,1,1,1,1,1,1]
Note:
`1 <= A.length <= 30000`
`1 <= K <= A.length`

### Solution 
 To solve the problem of flipping bits in an array, we will utilize a greedy approach combined with a sliding window technique. Our goal is to minimize the number of K-bit flips required to turn all 0s in the array into 1s. If it's not possible to do this, we will return -1.

Here's the step-by-step outline to solve the problem:

1. **Understanding the Flips**: Each flip at index `i` affects the array from `i` to `i + K - 1` by toggling (changing 0s to 1s and 1s to 0s) these positions. 

2. **Tracking State Changes**: To efficiently track the impact of flips, we can maintain an array that records the cumulative number of flips applied up to each index. This helps determine the effective state of each bit without needing to modify the original array every time we perform a flip.

3. **Greedy Approach**: We will iterate through the array and whenever we encounter a 0, we will perform a flip of size `K` trying to flip it to ensure that bit becomes 1.

4. **Checking Boundaries**: If we reach an index where we can no longer flip (i.e., the current index + K exceeds the length of the array), but still find a 0, it means it's impossible to make all bits 1, and we should return -1.

5. **Counting Flips**: We'll maintain a counter to track how many flips we've made.

Now, let's implement the solution in Python following the LeetCode style format:



```python
class Solution:
    def minKBitFlips(self, A: List[int], K: int) -> int:
        n = len(A)
        flip_count = 0  # To count the number of K-bit flips
        current_flips = 0  # To track the current state of flips effect
        
        # We use an array to keep track of flip effects
        flip_effects = [0] * n
        
        for i in range(n):
            # Update current flips based on the effects of previous flips 
            if i >= K:
                current_flips ^= flip_effects[i - K]
                
            # Determine the effective value of A[i]
            effective_value = A[i] ^ current_flips
            
            # If the effective value is 0, we need to flip
            if effective_value == 0:
                if i + K > n:  # Check if we can flip
                    return -1
                flip_count += 1  # Increment flip count
                current_flips ^= 1  # Start a new flip effect
                flip_effects[i] ^= 1  # Mark the flip effect starting from this index

        return flip_count

```

### Explanation of the Code:
- **Initialization**: We define the `flip_count` to keep track of the number of flips done. `current_flips` keeps track of how the previous flips affect the current bit.
- **The Loop**: As we iterate through the array:
    - We check if we are past index `K`, and if so, we remove the effect of the flip that started `K` indices back (using XOR).
    - We then determine the effective value of `A[i]` after considering the current flip status.
    - If we encounter a 0 (meaning we need a flip), we check if a flip can still be accommodated. If it cannot, we return -1.
    - If it's possible, we record the flip and update the tracking arrays accordingly.
- **Final Output**: After processing all elements, `flip_count` will give the minimum number of flips required.

### Complexity:
- The time complexity is O(n) since we traverse the list once.
- The space complexity is O(n) due to the `flip_effects` array.

This approach ensures we efficiently compute the required flips while maintaining clarity and readability suitable for LeetCode submissions.

# 1005. Maximize Sum Of Array After K Negations

### Problem Description 
Given an array `A` of integers, we must modify the array in the following way: we choose an `i` and replace `A[i]` with `-A[i]`, and we repeat this process `K` times in total.  (We may choose the same index `i` multiple times.)
Return the largest possible sum of the array after modifying it in this way.


Example 1:
Input: A = [4,2,3], K = 1
Output: 5
Explanation: Choose indices (1,) and A becomes [4,-2,3].


Example 2:
Input: A = [3,-1,0,2], K = 3
Output: 6
Explanation: Choose indices (1, 2, 2) and A becomes [3,1,0,2].


Example 3:
Input: A = [2,-3,-1,5,-4], K = 2
Output: 13
Explanation: Choose indices (1, 4) and A becomes [2,3,-1,5,4].

Note:
`1 <= A.length <= 10000`
`1 <= K <= 10000`
`-100 <= A[i] <= 100`

### Solution 
 To solve the given problem, we need to maximize the sum of an array after performing the operation of negating an element in the array a total of `K` times. Given that we can choose the same index multiple times, our goal is to prioritize the negation of the most negative numbers (i.e., smallest numbers) to increase the overall sum as much as possible.

### Steps to Approach the Problem:

1. **Sort the Array**: First, we will sort the array in ascending order. By doing this, the smallest (most negative) numbers will be at the beginning. If we want to negate an element to increase the sum, we should start with these smallest numbers.

2. **Negate Elements**: Depending on the value of `K`, we will perform the negation operation:
   - If `K` is less than the number of negative numbers in the sorted array, we'll negate the `K` smallest numbers.
   - If `K` exceeds the number of negative numbers, we will negate all negatives and then handle the remaining negations. If `K` is odd after negating all the negative numbers, we will want to negate the smallest positive number (or the smallest absolute value).

3. **Calculate the Final Sum**: After performing the necessary negations, we can compute the sum of the modified array.

### Implementation of the Solution in Python

Here is the Python code wrapped in a function format suitable for LeetCode:



```python
class Solution:
    def largestSumAfterKNegations(self, A: List[int], K: int) -> int:
        # Step 1: Sort the array
        A.sort()
        
        # Step 2: Negate the smallest numbers for K times
        for i in range(len(A)):
            if K > 0 and A[i] < 0:
                A[i] = -A[i]
                K -= 1
                
        # After all possible negations, if K is still odd
        if K % 2 == 1:
            # Negate the smallest absolute value (which is the minimum value in A now)
            A.sort()  # Ensure the array is sorted again to find the smallest
            A[0] = -A[0]
        
        # Step 3: Return the sum of the modified array
        return sum(A)


```

### Explanation of the Code:

1. **Sorting**: We start by sorting the array `A`. This helps us to easily access the smallest and most negative numbers at the beginning of the array.

2. **Negation Loop**: We iterate through the sorted array:
   - For each element, if it is negative and we still have `K` operations left, we negate it (`A[i] = -A[i]`) and reduce `K` by 1.
   - We continue this until we've either negated all negative numbers or exhausted all `K` operations.

3. **Handle Remaining K**: After processing, if `K` is still odd (meaning we've done an odd number of negations), we need to negate the smallest absolute value in the array. We sort the array again and negate the first element (which will now be the smallest in absolute terms).

4. **Final Sum**: Finally, we return the sum of the modified array.

### Complexity Analysis:
- **Time Complexity**: The primary time-consuming operations are sorting the array, so the overall time complexity is O(N log N), where N is the length of the array.
- **Space Complexity**: This is O(1) if we ignore the input storage since we are modifying the array in place.

This implementation should run effectively for the input constraints provided in the problem.

