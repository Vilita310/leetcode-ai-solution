# 292. Nim Game

### Problem Description 
You are playing the following Nim Game with your friend:
Initially, there is a heap of stones on the table.

You and your friend will alternate taking turns, and you go first.

On each turn, the person whose turn it is will remove 1 to 3 stones from the heap.

The one who removes the last stone is the winner.

Given `n`, the number of stones in the heap, return `true` if you can win the game assuming both you and your friend play optimally, otherwise return `false`.


Example 1:
Input: n = 4
Output: false
Explanation: These are the possible outcomes:
1. You remove 1 stone. Your friend removes 3 stones, including the last stone. Your friend wins.

2. You remove 2 stones. Your friend removes 2 stones, including the last stone. Your friend wins.

3. You remove 3 stones. Your friend removes the last stone. Your friend wins.

In all outcomes, your friend wins.


Example 2:
Input: n = 1
Output: true

Example 3:
Input: n = 2
Output: true

Constraints:
`1 <= n <= 231 - 1`

### Solution 
 To solve the Nim Game problem, we need to determine whether the first player (you) can secure a win given `n` stones, assuming both players play optimally. 

### Explanation:

Let's break down the game dynamics:

1. **Game Mechanics**: You can remove between 1 to 3 stones in each turn. The player who takes the last stone wins.

2. **Base Cases**: 
   - If there is 1 stone left (`n = 1`), you can take it and win. So `n = 1` returns `true`.
   - If there are 2 stones (`n = 2`), you can take both and win. So `n = 2` returns `true`.
   - If there are 3 stones (`n = 3`), you can take all three and win. So `n = 3` returns `true`.
   - If there are 4 stones (`n = 4`), no matter if you take 1, 2, or 3 stones, your opponent can always take the remaining stones and win. Thus, `n = 4` returns `false`.

3. **Pattern Recognition**: 
   - Notice the pattern: If `n % 4 == 0`, the first player cannot secure a win. The reason is that the first player leaves the opponent in a position `n = 4, 8, 12, ...` (multiples of 4), where the opponent can always respond optimally to maintain that position.
   - Conversely, for `n % 4 != 0`, the first player can adjust their move to leave the opponent with a multiple of 4.

### Conclusion:
Based on this observation, we can derive a simple rule: If `n % 4 == 0`, return `false`. Otherwise, return `true`.

### Python Code:

Below is the implementation that follows the LeetCode solution format:



```python
class Solution:
    def canWinNim(self, n: int) -> bool:
        # Check if n is a multiple of 4
        return n % 4 != 0

```

### Example Testing:

- For `n = 4`: 
  - The output will be `false`, as the first player will lose.
- For `n = 1`: 
  - The output will be `true`, as the first player can take the only stone and win.
- For `n = 2`: 
  - The output will be `true`, as the first player can take both stones and win.
- For `n = 3`: 
  - The output will be `true`, as the first player can take all three stones and win.

### Complexity Analysis:
- **Time Complexity**: O(1) because it only involves a modulo operation and a conditional check.
- **Space Complexity**: O(1) since we are using a constant amount of space.

This solution is efficient and handles the maximum constraints easily. You can directly run this method on the LeetCode platform.

# 319. Bulb Switcher

### Problem Description 
There are `n` bulbs that are initially off. You first turn on all the bulbs, then you turn off every second bulb.

On the third round, you toggle every third bulb (turning on if it's off or turning off if it's on). For the `ith` round, you toggle every `i` bulb. For the `nth` round, you only toggle the last bulb.

Return the number of bulbs that are on after `n` rounds.


Example 1:
Input: n = 3
Output: 1
Explanation: At first, the three bulbs are [off, off, off].

After the first round, the three bulbs are [on, on, on].

After the second round, the three bulbs are [on, off, on].

After the third round, the three bulbs are [on, off, off]. 
So you should return 1 because there is only one bulb is on.


Example 2:
Input: n = 0
Output: 0

Example 3:
Input: n = 1
Output: 1

Constraints:
`0 <= n <= 109`

### Solution 
 To solve the problem of counting the number of bulbs that are on after `n` rounds, let's break down the operations involved.

### Explanation:
1. **Initial State**: All bulbs are initially turned off. For `n` bulbs, the initial state can be represented as an array of `False` values (off state).

2. **Rounds of Operations**:
   - In the first round, all bulbs are turned on.
   - In the second round, every second bulb is toggled.
   - In the third round, every third bulb is toggled.
   - This process continues until the nth round, toggling every nth bulb.

3. **Toggling Logic**: A bulb remains on if it is toggled an odd number of times and off if toggled an even number of times. 

4. **Analysis of Toggling**: A bulb `k` is toggled in rounds where `i` is a divisor of `k`. For instance:
   - Bulb 6 is toggled in rounds 1, 2, 3, and 6 (its divisors).
   - A number has an odd count of divisors only when it is a perfect square. This is because divisors generally come in pairs (e.g., 1 and 6, 2 and 3), except for square numbers (e.g., for 9: 1, 3, 9).

5. **Conclusion**: The bulbs that remain on correspond to the perfect squares up to `n`. Therefore, the number of bulbs that remain on is simply the floor value of the square root of n.

### Implementation:
We'll implement this logic in Python as follows, using the integer square root function to handle large values of `n` efficiently.



```python
import math

class Solution:
    def bulbSwitch(self, n: int) -> int:
        # The number of bulbs that are on is equal to the number of perfect squares less than or equal to n
        return int(math.isqrt(n))

```

### Test Cases:
Let's verify the implementation with the test cases provided in the problem description.

1. **Example 1**: 
   - Input: `n = 3`
   - Output: `1` (since only bulb 1 is on)
   
2. **Example 2**: 
   - Input: `n = 0`
   - Output: `0` (since there are no bulbs)
   
3. **Example 3**: 
   - Input: `n = 1`
   - Output: `1` (the only bulb is on)

### Complexity Analysis:
- **Time Complexity**: O(1) - The time taken to calculate the square root is constant time.
- **Space Complexity**: O(1) - We only need a constant amount of space for the result.

This solution is efficient even for the maximum constraints provided (`n` up to `10^9`). 

You can now run this code directly on the LeetCode website as it follows their required function signature and format.

# 521. Longest Uncommon Subsequence I

### Problem Description 
Given two strings `a` and `b`, find the length of the longest uncommon subsequence between them.

A subsequence of a string `s` is a string that can be obtained after deleting any number of characters from `s`. For example, `"abc"` is a subsequence of `"aebdc"` because you can delete the underlined characters in `"aebdc"` to get `"abc"`. Other subsequences of `"aebdc"` include `"aebdc"`, `"aeb"`, and `""` (empty string).

An uncommon subsequence between two strings is a string that is a subsequence of one but not the other.

Return the length of the longest uncommon subsequence between `a` and `b`. If the longest uncommon subsequence doesn't exist, return `-1`.


Example 1:
Input: a = "aba", b = "cdc"
Output: 3
Explanation: One longest uncommon subsequence is "aba" because "aba" is a subsequence of "aba" but not "cdc".

Note that "cdc" is also a longest uncommon subsequence.


Example 2:
Input: a = "aaa", b = "bbb"
Output: 3
Explanation: The longest uncommon subsequences are "aaa" and "bbb".


Example 3:
Input: a = "aaa", b = "aaa"
Output: -1
Explanation: Every subsequence of string a is also a subsequence of string b. Similarly, every subsequence of string b is also a subsequence of string a.


Constraints:
`1 <= a.length, b.length <= 100`
`a` and `b` consist of lower-case English letters.

### Solution 
 To solve the problem of finding the length of the longest uncommon subsequence between two strings `a` and `b`, we first need to understand the definition of an uncommon subsequence. An uncommon subsequence from string `a` is either a subsequence that is not present in string `b` or vice versa. 

Given this definition, we can derive a straightforward approach to solve the problem:

1. **Compare the two strings**:
   - If `a` is equal to `b`, there are no uncommon subsequences, and we should return `-1`.
   - If `a` is not equal to `b`, the longest uncommon subsequence will either be the whole string `a` or the whole string `b` since both can’t be subsequences of each other. Thus, the result will be the maximum length of the two strings.

From the above logic, it follows that:
- If the two strings are the same, return `-1`.
- If they are different, return the length of the longer string.

Here is how the Python code would look, formatted according to LeetCode's style:



```python
class Solution:
    def findLUSlength(self, a: str, b: str) -> int:
        if a == b:
            return -1
        else:
            return max(len(a), len(b))

# Example usage:
# solution = Solution()
# print(solution.findLUSlength("aba", "cdc"))  # Output: 3
# print(solution.findLUSlength("aaa", "bbb"))  # Output: 3
# print(solution.findLUSlength("aaa", "aaa"))  # Output: -1

```

### Detailed Explanation:
1. **Class Definition**: The function `findLUSlength` is within the `Solution` class, as required by LeetCode.
2. **Input Parameters**:
   - The method takes two strings `a` and `b` as input parameters.
3. **Equality Check**: 
   - The first step in the function checks if `a` is equal to `b`. If they are the same, we return `-1` since there’s no uncommon subsequence.
4. **Length Calculation**:
   - If `a` and `b` are different, we compute the maximum of the lengths of both strings (`len(a)` and `len(b)`) to determine the length of the longest uncommon subsequence.
5. **Return Statement**: Finally, we return the computed value.

The time complexity of this algorithm is O(n), where n is the length of the longer string because we are potentially checking the lengths and equality of two strings. The space complexity is O(1) since we are only using a few extra variables for computation. 

This solution effectively handles all edge cases as outlined in the problem statement and leverages the simple properties of strings to achieve the result efficiently.

# 777. Swap Adjacent in LR String

### Problem Description 
In a string composed of `'L'`, `'R'`, and `'X'` characters, like `"RXXLRXRXL"`, a move consists of either replacing one occurrence of `"XL"` with `"LX"`, or replacing one occurrence of `"RX"` with `"XR"`. Given the starting string `start` and the ending string `end`, return `True` if and only if there exists a sequence of moves to transform one string to the other.


Example 1:
Input: start = "RXXLRXRXL", end = "XRLXXRRLX"
Output: true
Explanation: We can transform start to end following these steps:
RXXLRXRXL ->
XRXLRXRXL ->
XRLXRXRXL ->
XRLXXRRXL ->
XRLXXRRLX

Example 2:
Input: start = "X", end = "L"
Output: false

Example 3:
Input: start = "LLR", end = "RRL"
Output: false

Example 4:
Input: start = "XL", end = "LX"
Output: true

Example 5:
Input: start = "XLLR", end = "LXLX"
Output: false

Constraints:
`1 <= start.length <= 104`
`start.length == end.length`
Both `start` and `end` will only consist of characters in `'L'`, `'R'`, and `'X'`.

### Solution 
 To solve the problem of determining if the string `start` can be transformed into the string `end` through a series of allowed moves, we need to focus on how the characters can be repositioned. The only allowed moves are:

1. Replacing `"XL"` with `"LX"`
2. Replacing `"RX"` with `"XR"`

From these moves, we can deduce several important points:

1. The characters 'X' can be freely moved around, as they can be swapped with either 'L' or 'R'.
2. The order of character sets is essential. The number of 'L's and 'R's in both strings should match. If the counting of 'X's are disturbed due to the transformations, it may affect the sequence.

Overall, our approach revolves around maintaining the relative positioning of 'L's and 'R's while allowing 'X's to fill between them or replace them moving forward or backward.

Here's a step-by-step breakdown of the algorithm we will use:

1. **Counting 'L', 'R', and 'X'**: We will count the number of each character in both strings. It needs to be established that the counts of 'L' and 'R' match between `start` and `end`.

2. **Simulating the Moves**: Using the positions of 'L' and 'R' in both strings, we will check whether we can reconstruct the positions in such a way that we can move from `start` to `end`.

3. **Validation of Moves**: The movement of 'R' towards the right must maintain the count of 'X' in between 'L's and ensure that 'L' can precede 'R'.

4. **Returning the Result**: After validating the conditions through positional tracking, we'll return `True` if the transformation is possible, and `False` otherwise.

Now, here’s the complete code to solve the problem:



```python
class Solution:
    def canTransform(self, start: str, end: str) -> bool:
        # Check if lengths are the same and count characters
        if len(start) != len(end):
            return False
        
        # Counts of each character
        countX_start = start.count('X')
        countX_end = end.count('X')
        countL_start = start.count('L')
        countL_end = end.count('L')
        countR_start = start.count('R')
        countR_end = end.count('R')
        
        # Counts must match
        if countX_start + countL_start + countR_start != len(start) or countX_end + countL_end + countR_end != len(end):
            return False
        
        if countL_start != countL_end or countR_start != countR_end:
            return False
        
        # Positions of L and R in the strings
        i_start, i_end = 0, 0
        n = len(start)
        
        while i_start < n and i_end < n:
            # Move past 'X's in start
            while i_start < n and start[i_start] == 'X':
                i_start += 1
            # Move past 'X's in end
            while i_end < n and end[i_end] == 'X':
                i_end += 1
            
            # If we reached the end of one before the other
            if i_start < n and i_end < n:
                if start[i_start] != end[i_end]:
                    return False
                # Check the movements of L and R
                if start[i_start] == 'L' and i_start < i_end:
                    return False  # Cannot move L to the right
                if start[i_start] == 'R' and i_start > i_end:
                    return False  # Cannot move R to the left
            
            # Move to next characters
            i_start += 1
            i_end += 1
        
        return True

```

### Explanation of the Code:

1. **Initial Checks**: We first ensure that the strings have the same length and identical character counts for 'L' and 'R'.

2. **Positional Matching**: We utilize two pointers to scan through both strings, matching 'L' and 'R' while allowing 'X's to be skipped.

3. **Directional Validity with Context**: At each position match of 'L' or 'R', we ensure that the move is valid based on possible directions (L should not move right and R should not move left).

4. **Final Validation**: If all checks pass, the transformation is possible, and we return `True`. If any rule is violated, we return `False`. 

This solution adheres to the problem constraints and efficiently determines the validity of the transformation.

# 1033. Moving Stones Until Consecutive

### Problem Description 
Three stones are on a number line at positions `a`, `b`, and `c`.

Each turn, you pick up a stone at an endpoint (ie., either the lowest or highest position stone), and move it to an unoccupied position between those endpoints.  Formally, let's say the stones are currently at positions `x, y, z` with `x < y < z`.  You pick up the stone at either position `x` or position `z`, and move that stone to an integer position `k`, with `x < k < z` and `k != y`.

The game ends when you cannot make any more moves, ie. the stones are in consecutive positions.

When the game ends, what is the minimum and maximum number of moves that you could have made?  Return the answer as an length 2 array: `answer = [minimum_moves, maximum_moves]`

Example 1:
Input: a = 1, b = 2, c = 5
Output: [1,2]
Explanation: Move the stone from 5 to 3, or move the stone from 5 to 4 to 3.


Example 2:
Input: a = 4, b = 3, c = 2
Output: [0,0]
Explanation: We cannot make any moves.


Example 3:
Input: a = 3, b = 5, c = 1
Output: [1,2]
Explanation: Move the stone from 1 to 4; or move the stone from 1 to 2 to 4.

Note:
`1 <= a <= 100`
`1 <= b <= 100`
`1 <= c <= 100`
`a != b, b != c, c != a`

### Solution 
 To solve the problem of moving stones on a number line, we'll need to analyze the positions of the stones and determine the minimum and maximum number of moves possible until no further moves can be made (i.e., the stones occupy consecutive positions).

### Understanding the Problem

Given three distinct positions for stones:
1. We need to identify the minimum and maximum number of moves before they become consecutive.
2. The stones can be thought of as integers on a number line.
3. A move consists of picking up a stone from one end (either the smallest or the largest) and placing it in an unoccupied spot between the other two stones.

### Steps to the Solution

1. **Sorting the Positions:** First, sort the positions of the stones. Let `x`, `y`, and `z` be the sorted positions such that `x < y < z`.

2. **Minimum Moves Calculation:**
   - Check the gaps between `x`, `y`, and `z`.
   - If the stones are already consecutive, return `[0, 0]`.
   - If there is one gap, you can move one stone to make them consecutive, resulting in `1` move.
   - If there are two gaps, you can make one move to cover the larger gap and another move to cover the smaller gap, resulting in `2` moves.

3. **Maximum Moves Calculation:**
   - To determine the maximum moves, you can think of maximizing the number of moves by moving stones one position at a time until they are consecutive.
   - The maximum is generally equal to the total number of gaps, taking into account each gap potentially requiring an individual move.

### Implementation

Here’s how this logic can be implemented in Python, formatted for easy submission to LeetCode:



```python
class Solution:
    def numMovesStones(self, a: int, b: int, c: int) -> List[int]:
        # Step 1: Sort the positions
        positions = sorted([a, b, c])
        x, y, z = positions
        
        # Step 2: Determine the minimum moves
        if z - y == 1 and y - x == 1:
            min_moves = 0  # Already consecutive
        elif z - y <= 2 or y - x <= 2:
            min_moves = 1  # One move can make them consecutive
        else:
            min_moves = 2  # Two moves needed to make them consecutive
        
        # Step 3: Determine the maximum moves
        max_moves = (z - x - 2)  # Number of positions between x and z minus the stones in between
        
        return [min_moves, max_moves]

```

### Explanation of the Code:
- We first sort the positions of the stones to ensure we can easily access the smallest and largest positions.
- We then check how many moves are needed based on the gaps:
    - If the stones are consecutive, the minimum moves are `0`.
    - If they can be made consecutive in one move, the minimum is `1`.
    - Otherwise, it will take `2` moves.
- For the maximum moves, we calculate how many integer positions are unoccupied between the smallest and largest stones (`z - x - 2`).

This code is efficient and straightforward, providing the result in O(1) time complexity since we are only dealing with a fixed number of operations regardless of the input values, given the constraints.

# 1227. Airplane Seat Assignment Probability

### Problem Description 
n` passengers board an airplane with exactly n` seats. The first passenger has lost the ticket and picks a seat randomly. But after that, the rest of passengers will:
Take their own seat if it is still available, 
Pick other seats randomly when they find their seat occupied 
What is the probability that the n-th person can get his own seat?

Example 1:
Input: n = 1
Output: 1.00000
Explanation: The first person can only get the first seat.


Example 2:
Input: n = 2
Output: 0.50000
Explanation: The second person has a probability of 0.5 to get the second seat (when first person gets the first seat).


Constraints:
`1 <= n <= 10^5`

### Solution 
 To solve the problem of determining the probability that the n-th passenger can get their own seat after the first passenger has randomly picked a seat, we can observe a few key points:

1. **Understanding the Scenario**:
    - If there is only one passenger (n=1), they will take their own seat since there's no alternative. Therefore, the probability is 1.
    - If there are two passengers (n=2), the first passenger can either choose seat 1 (their own) or seat 2 (the second passenger's seat). If the first picks their own seat, the second gets their seat. If the first picks seat 2, then the second passenger cannot sit in their own seat leading to a 0.5 probability for the second person to sit in their own seat.

2. **Generalizing the Probability**:
    - As we extend this reasoning to more passengers, we observe that the first passenger's choice significantly affects the probability distribution.
    - If the first passenger picks their own seat, everyone else will sit in their own seat.
    - If the first passenger picks the n-th passenger’s seat, then the n-th passenger can no longer sit in their own seat.
    - If they choose any other seat (1 through n-1), a recursive situation arises just like the original but with the first person and one less seat.

3. **Recursive Formulation**:
    - For `n` passengers:
        - If the first passenger picks seat 1, the probability is 1 (everyone sits in their own seat).
        - If the first passenger picks seat n, the probability is 0 (the n-th passenger cannot sit in their own seat).
        - For each of the other seats (1 to n-1), you essentially have the same problem with one less passenger leading to recursive probabilities.

After analyzing the situation, we can draw the conclusion that the probability for the n-th passenger to sit in their seat converges to either 0 or 1.

After a careful analysis, irrespective of the value of `n`:
- The probability is:
  - 1 (if n=1)
  - 0.5 (if n=2)
  - For n >= 3, the probability will stabilize at 0.5. 

Thus viewing it through this lens, we can create a simple solution:

### Python Solution
Here is the Python code to solve this problem using the findings detailed above:



```python
class Solution:
    def nthPersonGetsNthSeat(self, n: int) -> float:
        if n == 1:
            return 1.0
        elif n == 2:
            return 0.5
        else:
            return 0.5  # For n >= 3, the probability stabilizes at 0.5

```

### Explanation:
- **Function Definition**: We define a class `Solution` with a method `nthPersonGetsNthSeat`, which takes an integer `n` as input.
- **Base Cases**:
  - If `n == 1`, the return value is `1.0` because the one passenger will always get their seat.
  - If `n == 2`, the return value is `0.5` because the first passenger has a 50% chance of taking either the first or second seat.
- **Returning the Probability**:
  - For any `n >= 3`, we can directly return `0.5` based on our earlier conclusion that the chances stabilize at this value.

This solution runs in constant time O(1) and space O(1), making it efficient even for the upper constraint of `n = 100000`. You can copy this code directly into the LeetCode submission area and run it to verify correctness.

# 1503. Last Moment Before All Ants Fall Out of a Plank

### Problem Description 
We have a wooden plank of the length `n` units. Some ants are walking on the plank, each ant moves with speed 1 unit per second. Some of the ants move to the left, the other move to the right.

When two ants moving in two different directions meet at some point, they change their directions and continue moving again. Assume changing directions doesn't take any additional time.

When an ant reaches one end of the plank at a time `t`, it falls out of the plank imediately.

Given an integer `n` and two integer arrays `left` and `right`, the positions of the ants moving to the left and the right. Return the moment when the last ant(s) fall out of the plank.


Example 1:
Input: n = 4, left = [4,3], right = [0,1]
Output: 4
Explanation: In the image above:
-The ant at index 0 is named A and going to the right.

-The ant at index 1 is named B and going to the right.

-The ant at index 3 is named C and going to the left.

-The ant at index 4 is named D and going to the left.

Note that the last moment when an ant was on the plank is t = 4 second, after that it falls imediately out of the plank. (i.e. We can say that at t = 4.0000000001, there is no ants on the plank).


Example 2:
Input: n = 7, left = [], right = [0,1,2,3,4,5,6,7]
Output: 7
Explanation: All ants are going to the right, the ant at index 0 needs 7 seconds to fall.


Example 3:
Input: n = 7, left = [0,1,2,3,4,5,6,7], right = []
Output: 7
Explanation: All ants are going to the left, the ant at index 7 needs 7 seconds to fall.


Example 4:
Input: n = 9, left = [5], right = [4]
Output: 5
Explanation: At t = 1 second, both ants will be at the same intial position but with different direction.


Example 5:
Input: n = 6, left = [6], right = [0]
Output: 6

Constraints:
`1 <= n <= 10^4`
`0 <= left.length <= n + 1`
`0 <= left[i] <= n`
`0 <= right.length <= n + 1`
`0 <= right[i] <= n`
`1 <= left.length + right.length <= n + 1`
All values of `left` and `right` are unique, and each value can appear only in one of the two arrays.

### Solution 
 To solve the problem of determining when the last ant falls off a wooden plank given their positions and directions, we can break down the problem as follows:

### Problem Breakdown:
1. **Understanding Movement**: An ant moving to the left will fall off the plank when it reaches position `0`, which takes a time equal to the position itself (if it starts at `x`, it falls off at time `t = x`). Similarly, an ant moving to the right will fall off at position `n`, which takes `n - position` time.

2. **Finding Maximum Time**: The overall time when the last ant falls off the plank is simply the maximum of the fall-off times calculated for all ants. 

3. **Input Processing**:
   - We are given a length of the plank `n`, positions of the ants moving left (in an array `left`), and positions of the ants moving right (in an array `right`).
   - For each position in `left`, compute the time it would take to fall off (which just equals the position). For each position in `right`, compute the time as `n - position`.

### Complexity Analysis:
The time complexity of our solution is O(m + k), where `m` is the length of `left` and `k` is the length of `right`, because we are iterating over both lists only once.

### Implementation:
Here is the complete Python code according to the LeetCode format:



```python
class Solution:
    def getLastMoment(self, n: int, left: List[int], right: List[int]) -> int:
        max_time = 0
        
        # Calculate the fall-off time for all ants moving left
        for pos in left:
            max_time = max(max_time, pos)
        
        # Calculate the fall-off time for all ants moving right
        for pos in right:
            max_time = max(max_time, n - pos)
        
        return max_time

```

### Explanation of the Code:
- We define a class `Solution` that contains the method `getLastMoment`.
- The method takes three parameters: `n` (length of the plank), `left` (list of positions of ants moving left), and `right` (list of positions of ants moving right).
- We initialize a variable `max_time` to store the maximum time when any ant falls off the plank.
- We then loop over each ant in the `left` list and calculate their fall-off time (which is equal to their position). If this time is greater than the current `max_time`, we update `max_time`.
- In the next loop, we evaluate the ants in the `right` list, calculating their fall-off time (which is `n - position`). We again update `max_time` whenever we find a larger fall-off time.
- Finally, we return `max_time` as the result, which represents when the last ant falls off the plank.

### Example Execution:
For `n = 4`, `left = [4, 3]`, and `right = [0, 1]`:
- The ants moving left will fall at times 4 and 3.
- The ants moving right will fall at times `4` (falling off from `0` to `4`).
- The maximum among these times is `4`, which is the correct output.

This solution is efficient and adheres to the constraints provided in the problem statement.

