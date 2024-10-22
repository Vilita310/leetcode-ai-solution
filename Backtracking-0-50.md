# 10. Regular Expression Matching

### Problem Description 
Given an input string (`s`) and a pattern (`p`), implement regular expression matching with support for `'.'` and `'*'` where:` `
`'.'` Matches any single character.​​​​
`'*'` Matches zero or more of the preceding element.

The matching should cover the entire input string (not partial).


Example 1:
Input: s = "aa", p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".


Example 2:
Input: s = "aa", p = "a*"
Output: true
Explanation: '*' means zero or more of the preceding element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".


Example 3:
Input: s = "ab", p = ".*"
Output: true
Explanation: ".*" means "zero or more (*) of any character (.)".


Example 4:
Input: s = "aab", p = "c*a*b"
Output: true
Explanation: c can be repeated 0 times, a can be repeated 1 time. Therefore, it matches "aab".


Example 5:
Input: s = "mississippi", p = "mis*is*p*."
Output: false

Constraints:
`0 <= s.length <= 20`
`0 <= p.length <= 30`
`s` contains only lowercase English letters.

`p` contains only lowercase English letters, `'.'`, and `'*'`.

It is guaranteed for each appearance of the character `'*'`, there will be a previous valid character to match.

### Solution 
 To solve the problem of regular expression matching using `.` and `*` in Python, we can utilize a dynamic programming approach. The idea is to create a 2D boolean array where each cell in the array represents whether the substring of `s` can match the substring of `p` up to that point.

Here's a step-by-step breakdown of how the algorithm works:

1. **Initialization**: We define a 2D array `dp` of size `(m+1) x (n+1)`, where `m` is the length of the string `s` and `n` is the length of the pattern `p`. The extra row and column will help us manage empty substrings.

2. **Base Case**: 
    - `dp[0][0]` is `True` because an empty string matches an empty pattern.
    - We then fill the first row for patterns that can match empty strings. This involves checking if we have patterns like `x*`, where `x` can be any character preceding `*`.

3. **Filling the DP Table**: 
    - We iterate through each character of `s` and `p`. If characters match (or if we have a `.` in the pattern), we take the value from the diagonal cell (i.e., `dp[i-1][j-1]`).
    - If we encounter a `*`, it can either represent zero occurrences of the preceding element or one or more occurrences:
        - Zero occurrences: We take the value from the cell directly to the left.
        - One or more occurrences: If the preceding character matches, we take the value from the cell directly above.

4. **Final Result**: The value at `dp[m][n]` will give us the result of whether the entire string `s` matches the entire pattern `p`.

Now, here is the Python code implementing the above logic:



```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True  # Both string and pattern are empty
        
        # Initialize the first row for patterns like a*, a*b*, a*b*c* etc.
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]

        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == s[i - 1] or p[j - 1] == '.':
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == '*':
                    # Use '*' as zero occurrence
                    dp[i][j] = dp[i][j - 2]
                    # Use '*' as one or more occurrences
                    if p[j - 2] == s[i - 1] or p[j - 2] == '.':
                        dp[i][j] = dp[i][j] or dp[i - 1][j]

        return dp[m][n]

```

### Explanation of the Code:

- We define the `isMatch` method which takes the string `s` and the pattern `p` as inputs.
- We initialize the `dp` table to store the boolean matching results.
- We process both the string and the pattern character by character to fill in the `dp` table according to the rules of matching.
- Finally, we return `dp[m][n]` which contains the result of whether `s` matches `p`.

### Complexity:
- **Time Complexity**: O(m * n), where `m` is the length of `s` and `n` is the length of `p`. We are filling a table of size (m+1) x (n+1).
- **Space Complexity**: O(m * n) for the `dp` table. This can be optimized to O(n) with further optimizations, but for simplicity, we kept it in the 2D array format.

This code should work directly on the LeetCode platform.

# 17. Letter Combinations of a Phone Number

### Problem Description 
Given a string containing digits from `2-9` inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.


Example 1:
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

Example 2:
Input: digits = ""
Output: []

Example 3:
Input: digits = "2"
Output: ["a","b","c"]

Constraints:
`0 <= digits.length <= 4`
`digits[i]` is a digit in the range `['2', '9']`.

### Solution 
 To solve the problem of generating all possible letter combinations from a given string of digits (from '2' to '9'), we can utilize a backtracking approach. The idea is to map each digit to its corresponding letters based on the phone keypad mapping and systematically explore all possible letter combinations these digits can generate.

Here's a breakdown of the approach:

1. **Mapping Digits to Letters**: Create a dictionary that maps each digit to a string of letters based on the phone keypad (for example, '2' maps to "abc", '3' maps to "def", etc.).

2. **Handle Edge Cases**: If the input string is empty, return an empty list as there are no combinations possible.

3. **Backtracking Function**: Implement a recursive function that builds combinations. This function takes the current combination being built and the index of the next digit to process.

4. **Base Case**: When the length of the current combination equals the length of the input digits, add the combination to the results list.

5. **Recursive Case**: Loop through the letters corresponding to the current digit, appending each letter to the current combination, and then recursively call the function for the next digit.

Let's implement this step by step in code:



```python
class Solution:
    def letterCombinations(self, digits: str) -> list[str]:
        # Base case
        if not digits:
            return []
        
        # Mapping of digit to letters
        phone_map = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }
        
        # Result list to store combinations
        result = []
        
        # Backtracking function
        def backtrack(index: int, current_combination: str):
            # If the current combination is the same length as digits, add it to results
            if index == len(digits):
                result.append(current_combination)
                return
            
            # Get the letters corresponding to the current digit
            current_digit = digits[index]
            possible_letters = phone_map[current_digit]
            
            # Iterate over the letters and continue building combinations
            for letter in possible_letters:
                backtrack(index + 1, current_combination + letter)
        
        # Start backtracking
        backtrack(0, "")
        
        return result

```

### Explanation of the Code:

1. **Mapping Initialization**: We define a dictionary `phone_map` which maps each digit from '2' to '9' to its corresponding letters.

2. **Base Case Check**: We immediately return an empty list if the input `digits` is empty.

3. **Backtrack Function**: We define a `backtrack` function that takes the index of the current digit being processed and the current combination of letters built so far. 

4. **Base Case in Backtrack**: If `index` equals the length of `digits`, we've constructed a complete combination, so we add it to the `result`.

5. **Recursive Exploration**: For each letter corresponding to the current digit, we call `backtrack` recursively incrementing the index.

6. **Start Backtracking**: The process begins by calling the `backtrack` function with the initial values.

By running the above code in the LeetCode environment, it will generate all possible combinations of letters for the provided string of digits.

The solution adheres to the constraints given and efficiently generates combinations by exploring tree-like structures of possibilities through backtracking.

# 22. Generate Parentheses

### Problem Description 
Given `n` pairs of parentheses, write a function to generate all combinations of well-formed parentheses.


Example 1:
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]

Example 2:
Input: n = 1
Output: ["()"]

Constraints:
`1 <= n <= 8`

### Solution 
 To solve the problem of generating all combinations of well-formed parentheses given `n` pairs of parentheses, we can use a backtracking approach. Backtracking allows us to explore potential solutions one step at a time and backtrack as soon as we determine that a step will not lead to a valid solution.

### Explanation of the Approach:

1. **Understanding Well-formed Parentheses**: 
   A string of parentheses is well-formed if:
   - The number of opening parentheses `(` is equal to the number of closing parentheses `)`.
   - At no point in the string, when reading from left to right, should the count of closing parentheses exceed the count of opening parentheses.

2. **Backtracking Steps**: 
   We can maintain a string that we build as we generate parentheses. The parameters we'll track during recursion are:
   - The current string of parentheses being built.
   - The number of `(` currently used and the number of `)` currently used.

   We start with an empty string and two counts (for `(` and `)`) set to zero.

3. **Adding Parentheses**: 
   - We can add a `(` as long as the count of `(` used is less than `n`.
   - We can add a `)` if its count is less than the count of `(` already used (to ensure it’s valid).

4. **Base Case**: 
   Once the length of the current string is `2 * n`, we have a valid well-formed parentheses combination and can add it to our results.

### Python Code Implementation:
Here’s how we can implement this logic using Python:



```python
class Solution:
    def generateParenthesis(self, n: int):
        def backtrack(current_string, open_count, close_count):
            # Base case: if the current string reaches the maximum length
            if len(current_string) == 2 * n:
                result.append(current_string)
                return
            
            # If we can add an opening parenthesis
            if open_count < n:
                backtrack(current_string + '(', open_count + 1, close_count)
            
            # If we can add a closing parenthesis
            if close_count < open_count:
                backtrack(current_string + ')', open_count, close_count + 1)

        result = []
        backtrack('', 0, 0)
        return result

```

### Explanation of the Code:
- We define a class `Solution` which contains the function `generateParenthesis`.
- Inside this function, we define a helper function `backtrack` that performs the recursive generation of parentheses.
- We initialize an empty list `result` to store the valid combinations.
- The `backtrack` function builds the string step by step:
  - It checks if the current string is of the required length (i.e., `2 * n`).
  - It then decides whether to add a `(` or `)` based on the counts of each.
  
### Complexity:
- **Time Complexity**: The time complexity is O(4^n / sqrt(n)), which is derived from the number of valid combinations generated.
- **Space Complexity**: O(n), which accounts for the depth of the recursion stack and the storage of the result.

### Running the Code:
This solution can be directly run on the LeetCode website as it conforms to the expected function signature and utilizes the required class structure. Simply copy the code into your LeetCode submission for the problem "Generate Parentheses".

# 37. Sudoku Solver

### Problem Description 
Write a program to solve a Sudoku puzzle by filling the empty cells.

A sudoku solution must satisfy all of the following rules:
Each of the digits `1-9` must occur exactly once in each row.

Each of the digits `1-9` must occur exactly once in each column.

Each of the digits `1-9` must occur exactly once in each of the 9 `3x3` sub-boxes of the grid.

The `'.'` character indicates empty cells.


Example 1:
Input: board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
Output: [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
Explanation: The input board is shown above and the only valid solution is shown below:

Constraints:
`board.length == 9`
`board[i].length == 9`
`board[i][j]` is a digit or `'.'`.

It is guaranteed that the input board has only one solution.

### Solution 
 To solve the Sudoku puzzle, we need to implement a backtracking algorithm. This approach tries to fill the empty cells (`'.'`) one by one by attempting to place digits (from `1` to `9`) in a valid spot. A valid position must satisfy the three Sudoku rules:

1. Each digit must occur exactly once in each row.
2. Each digit must occur exactly once in each column.
3. Each digit must occur exactly once in each of the 9 `3x3` sub-boxes of the grid.

If a placement does not lead to a solution, we backtrack and try the next possible number.

Here's a complete Python implementation suitable for LeetCode:



```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        
        def canPlace(board, row, col, num):
            # Check if num is not in the current row
            for j in range(9):
                if board[row][j] == num:
                    return False
            
            # Check if num is not in the current column
            for i in range(9):
                if board[i][col] == num:
                    return False
            
            # Check if num is not in the current 3x3 box
            box_row = (row // 3) * 3
            box_col = (col // 3) * 3
            for i in range(3):
                for j in range(3):
                    if board[box_row + i][box_col + j] == num:
                        return False
            
            return True

        def solve(board):
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        for num in map(str, range(1, 10)):  # Check digits 1 to 9
                            if canPlace(board, i, j, num):
                                board[i][j] = num  # Make the move
                                if solve(board):  # Recur
                                    return True
                                board[i][j] = '.'  # Backtrack
                        return False  # If no number can be placed, return False
            return True  # If the board is completely filled and valid

        solve(board)

```

### Detailed Explanation:

1. **Data structure used**: We utilize a 2D list (`board`), where each element can be a digit from '1' to '9' or a '.' indicating an empty cell.

2. **Helper function `canPlace`**: 
   - This function checks if a number can be placed in a specific cell without violating the Sudoku rules.
   - It checks the corresponding row, column, and the respective 3x3 sub-box to ensure that the number isn't already present.

3. **Recursive function `solve`**:
   - This function tries to fill the board recursively. It scans each cell to find an empty one ('.').
   - If an empty cell is found, it tests placing each number from `1` to `9`:
     - For each number, it checks if the placement is valid using `canPlace`. 
     - If valid, it places the number and recursively calls `solve` to try and fill the next empty cell.
     - If the recursive call returns `True`, it means the puzzle is solved; otherwise, it backtracks by resetting the cell to '.' and trying the next number.
   - If no number from `1` to `9` is valid for a cell, it returns `False`, signaling backtracking.

4. **Base case**: The recursion stops when all cells are filled, which the code identifies by checking if it has gone through all cells without returning `False`.

5. **In-place modification**: The solution modifies the input `board` directly.

This algorithm is efficient due to the constraints of the Sudoku, where solutions are guaranteed to exist and are unique for the given board configuration. The recursive backtracking ensures we explore all potential configurations to reach the solution.

# 39. Combination Sum

### Problem Description 
Given an array of distinct integers `candidates` and a target integer `target`, return a list of all unique combinations of `candidates` where the chosen numbers sum to `target`. You may return the combinations in any order.

The same number may be chosen from `candidates` an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

It is guaranteed that the number of unique combinations that sum up to `target` is less than `150` combinations for the given input.


Example 1:
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.

7 is a candidate, and 7 = 7.

These are the only two combinations.


Example 2:
Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]

Example 3:
Input: candidates = [2], target = 1
Output: []

Example 4:
Input: candidates = [1], target = 1
Output: [[1]]

Example 5:
Input: candidates = [1], target = 2
Output: [[1,1]]

Constraints:
`1 <= candidates.length <= 30`
`1 <= candidates[i] <= 200`
All elements of `candidates` are distinct.

`1 <= target <= 500`

### Solution 
 Here's a Python implementation to solve the problem of finding all unique combinations of numbers in an array that sum up to a given target. We will employ a backtracking approach to explore all possible combinations. Below is the code along with a detailed explanation.



```python
from typing import List

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        
        def backtrack(remaining: int, comb: List[int], start: int):
            # If the remaining target is 0, we found a valid combination
            if remaining == 0:
                result.append(list(comb))  # Append a copy of the current combination
                return
            # If the remaining target is less than 0, no valid combination can be formed
            elif remaining < 0:
                return
            
            # Loop through the candidates starting from the current index
            for i in range(start, len(candidates)):
                # Include the candidate[i] in the combination
                comb.append(candidates[i])
                # Give the current candidate another chance, rather than moving on
                backtrack(remaining - candidates[i], comb, i)
                # Backtrack and remove the candidate from the combination
                comb.pop()
        
        # Start backtracking with the initial target and an empty combination
        backtrack(target, [], 0)
        return result

```

### Explanation

1. **Function Signature**: 
   - The function `combinationSum` takes two arguments:
     - `candidates`: A list of distinct integers.
     - `target`: The sum we wish to achieve through combinations of elements from `candidates`.

2. **Result Storage**: 
   - An empty list named `result` is initialized to store valid combinations that sum up to the target.

3. **Backtracking Function**:
   - The inner helper function `backtrack` is defined which takes three parameters:
     - `remaining`: The amount left to reach the target.
     - `comb`: The current combination being explored.
     - `start`: The index in `candidates` to begin searching for new candidates to include in `comb`.

4. **Base Cases**:
   - If `remaining` is exactly zero, we've found a valid combination, so it is added to `result`.
   - If `remaining` goes below zero, we exit the function since it implies that the current combination is invalid.

5. **Exploring Candidates**:
   - A `for` loop iterates through `candidates` starting from the index `start`:
     - The current candidate (`candidates[i]`) is appended to `comb`.
     - The `backtrack` function is called recursively with the updated `remaining` (subtracting the candidate's value) and the current index `i` to allow the same candidate to be reused.
     - After returning from the recursive call, the last candidate is popped from `comb` to backtrack and explore other potential combinations.

6. **Starting the Backtracking Process**: 
   - The `backtrack` function is initially called with `target`, an empty combination, and starting index `0`.

7. **Return Result**: 
   - Finally, `result` is returned, containing all combinations that sum up to the target.

### Complexity Analysis
- **Time Complexity**: O(2^t), where t is the target, due to potentially exploring all combinations.
- **Space Complexity**: O(k), where k is the depth of recursion and can go up to the target value in the worst case.

This implementation should work efficiently within the constraints provided for the problem. You can directly run this code on LeetCode's platform by placing it in the code editor for the corresponding problem.

# 40. Combination Sum II

### Problem Description 
Given a collection of candidate numbers (`candidates`) and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sum to `target`.

Each number in `candidates` may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.


Example 1:
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]

Example 2:
Input: candidates = [2,5,2,1,2], target = 5
Output: 
[
[1,2,2],
[5]
]

Constraints:
`1 <= candidates.length <= 100`
`1 <= candidates[i] <= 50`
`1 <= target <= 30`

### Solution 
 To solve the problem of finding all unique combinations of candidate numbers that sum to a target, we can use a backtracking approach. The main idea is to explore each candidate number while keeping track of the current sum and the path taken so far. If the current sum equals the target, we add the current path to the results. If it exceeds the target, we stop exploring further down that path. 

To avoid duplicate combinations, we'll first sort the candidates. During our recursive backtracking algorithm, if we encounter the same number as the previous one, we skip it to avoid duplicates.

Here's the detailed Python code formatted for submission on LeetCode:



```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtrack(start: int, path: List[int], target: int):
            if target == 0:
                result.append(path)
                return
            for i in range(start, len(candidates)):
                # Skip the same elements to avoid duplicates
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                # If the current number is greater than the target, we can break,
                # since all subsequent numbers will be greater (due to sorting).
                if candidates[i] > target:
                    break
                # Recur with the next number and decrease the target
                backtrack(i + 1, path + [candidates[i]], target - candidates[i])

        # Sort candidates to facilitate skipping duplicates
        candidates.sort()
        result = []
        # Start backtracking
        backtrack(0, [], target)
        return result

```

### Explanation of the Code:

1. **Sorting the Candidates**: 
   - We first sort the `candidates` list. This helps in easily skipping duplicates and also allows us to stop early when the current candidate exceeds the target during backtracking.

2. **Backtracking Function**:
   - A helper function `backtrack` is defined, which takes three parameters:
     - `start`: The index at which to start looking for candidates.
     - `path`: The current combination being formed.
     - `target`: The remaining sum we need to achieve.

3. **Base Case**:
   - If `target` is zero, it means we've found a valid combination. We append the current `path` to the results.

4. **Iterating through Candidates**:
   - We iterate over the candidates starting from the `start` index. 
   - If the current candidate is the same as the previous one (and it's not the starting index), we skip it. This prevents adding the same combination multiple times.
   - If the candidate exceeds the target, we break out of the loop, as further candidates will be larger (because the list is sorted).

5. **Recursive Call**:
   - We make a recursive call to `backtrack`, advancing the `start` index to avoid using the same candidate again, and subtract the candidate's value from the target.

6. **Final Result**:
   - Finally, we return the `result` list containing all unique combinations.

### Time Complexity:
The time complexity is O(2^n) in the worst case, where n is the number of candidates, due to the exponential nature of the backtracking search.

### Space Complexity:
The space complexity is O(n), accounting for the recursion stack and the space needed to store results. 

You can now submit this code in LeetCode, and it should work correctly for the problem given.

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
 To solve this problem of wildcard pattern matching, we can use dynamic programming. The idea is to construct a 2D table that keeps track of whether substrings of `s` can match substrings of `p` based on the wildcard rules.

### Detailed Explanation:

1. **Understanding Characters**:
   - The character `?` can match any single character.
   - The character `*` can match any sequence of characters, including an empty sequence.

2. **Dynamic Programming Table Initialization**:
   - We'll create a 2D boolean array `dp`, where `dp[i][j]` will be `True` if the first `i` characters of string `s` can be matched by the first `j` characters of pattern `p`.
   - `dp[0][0]` will be `True` because an empty string matches an empty pattern.

3. **Handling Leading Asterisks**:
   - If the pattern starts with one or more `*`, then `dp[0][j]` for `j > 0` should be `True` because a pattern can match an empty string if it starts with `*`.

4. **Filling the DP Table**:
   - For each character in `s` and `p`, we will fill in the `dp` table based on the following rules:
     - If the current characters match (or pattern has `?`), the state will depend on the previous state: `dp[i][j] = dp[i-1][j-1]`.
     - If the current character in `p` is `*`, then it can either:
       - Match nothing (carry the state over from the left): `dp[i][j] = dp[i][j-1]`.
       - Match one more character from `s` (carry the state from above): `dp[i][j] = dp[i-1][j]`.

5. **Result Extraction**:
   - The value at `dp[len(s)][len(p)]` will give us the result for the entire strings.

### Final Code:

Here's the solution encapsulated in the `isMatch` function:



```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        # Create a 2D DP array
        dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
        
        # An empty pattern can match an empty string
        dp[0][0] = True
        
        # Fill the first row where pattern starts with *
        for j in range(1, len(p) + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 1]
        
        # Fill the DP table
        for i in range(1, len(s) + 1):
            for j in range(1, len(p) + 1):
                if p[j - 1] == '*':
                    # * can match empty (dp[i][j-1]) or one more character (dp[i-1][j])
                    dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
                elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                    # Match single character
                    dp[i][j] = dp[i - 1][j - 1]
        
        return dp[len(s)][len(p)]

```

### Explanation of Code:
- We initialize a DP table with dimensions based on the lengths of `s` and `p`.
- We then set the base case for an empty pattern and proceed to fill the table according to the rules mentioned above.
- Finally, the function returns the value that indicates if the entire string `s` can be matched by pattern `p`. 

This implementation is efficient and operates within the constraints provided, making it suitable for a LeetCode submission.

# 46. Permutations

### Problem Description 
Given an array `nums` of distinct integers, return all the possible permutations. You can return the answer in any order.


Example 1:
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

Example 2:
Input: nums = [0,1]
Output: [[0,1],[1,0]]

Example 3:
Input: nums = [1]
Output: [[1]]

Constraints:
`1 <= nums.length <= 6`
`-10 <= nums[i] <= 10`
All the integers of `nums` are unique.

### Solution 
 To solve the problem of generating all possible permutations of a given array of distinct integers, we can use a backtracking algorithm. The idea is to construct permutations step by step, choosing each number in the array one at a time and recursively adding the remaining numbers until all numbers are included in a permutation.

Here's the step-by-step explanation of the approach:

1. **Base Case**: If the current permutation length equals the length of the input array, we add it to the result list because we have formed a complete permutation.

2. **Backtracking Process**:
   - We iterate through each number in the input array.
   - For each number, we check if it is already used in the current permutation. If not, we mark it as used.
   - We then append this number to the current permutation and recursively call the function to continue building the permutation.
   - After the recursive call, we backtrack by removing the last number and marking it as unused, allowing the next iteration to consider that number.

3. **Collecting Results**: During the recursive calls, whenever a complete permutation is formed, we store it in a results list which will eventually contain all the permutations.

Here's how we can implement this in Python, formatted to be directly runnable on LeetCode:



```python
class Solution:
    def permute(self, nums):
        result = []
        self._backtrack(nums, [], result)
        return result
    
    def _backtrack(self, nums, current_permutation, result):
        # If the current permutation is the same length as the original, we found a valid permutation
        if len(current_permutation) == len(nums):
            result.append(current_permutation[:])  # Append a copy of current_permutation
            return
        
        for num in nums:
            if num in current_permutation:
                continue  # Skip used numbers
            
            # Choose the current number
            current_permutation.append(num)
            # Recurse with the new permutation
            self._backtrack(nums, current_permutation, result)
            # Backtrack by removing the number we just added
            current_permutation.pop()


```

### Explanation of the Code:
- The `permute` function initializes an empty `result` list and calls the `_backtrack` helper function.
- The `_backtrack` function takes in the `nums` array, the `current_permutation`, and the `result` list.
- If the size of `current_permutation` matches the size of `nums`, we have a complete permutation, and we add a copy of it to `result`.
- We loop through each number in `nums`. If it's already included in `current_permutation`, we skip it. Otherwise, we add it, call `_backtrack` recursively to fill in the next numbers, and then backtrack by removing the last added number.

### Complexity:
- **Time Complexity**: O(n!), where n is the size of `nums`. This is because there are n! permutations for n distinct numbers.
- **Space Complexity**: O(n), for the recursion stack and temporary storage of permutations in `current_permutation`.

This code is efficient for the input size limits specified in the problem (1 to 6 distinct integers).

# 47. Permutations II

### Problem Description 
Given a collection of numbers, `nums`, that might contain duplicates, return all possible unique permutations in any order.


Example 1:
Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]

Example 2:
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

Constraints:
`1 <= nums.length <= 8`
`-10 <= nums[i] <= 10`

### Solution 
 To solve the problem of generating all unique permutations from a list of numbers that may include duplicates, we can employ a backtracking approach. This method will allow us to explore all possible arrangements while ensuring that we do not repeat permutations that are identical due to the presence of duplicates.

Here's a step-by-step breakdown of the approach we will take:

1. **Sorting**: First, we will sort the input array `nums`. This helps in easily identifying duplicates because identical numbers will be adjacent after sorting.

2. **Backtracking Function**: We will create a recursive function that builds the permutations. We will keep track of the current permutation being constructed and the indices of the numbers that have been used so far.

3. **Using a Seen Set**: To avoid creating duplicate permutations, we will maintain a set of indices that we have used in the current recursive call. We will only proceed to use a number if it hasn’t been used before in the current position (ensured by checking the preceding number to avoid duplicates).

4. **Base Case**: When the current permutation reaches the length of the input list, we have found a valid permutation, and we add it to our results.

5. **Result Collection**: Finally, we'll return the list of unique permutations.

Here's the complete code following the LeetCode format:



```python
from typing import List

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # Sort the numbers to handle duplicates
        nums.sort()
        result = []
        visited = [False] * len(nums)
        self.backtrack(nums, result, [], visited)
        return result
    
    def backtrack(self, nums, result, current, visited):
        if len(current) == len(nums):  # Base case: if the current permutation is complete
            result.append(current[:])  # Add a copy of the current permutation to the result
            return
        
        for i in range(len(nums)):
            if visited[i]:  # Skip if this number has been used
                continue
            
            # To skip duplicates: if the current number is the same as the previous one 
            # and the previous number hasn't been used, we skip this iteration
            if i > 0 and nums[i] == nums[i - 1] and not visited[i - 1]:
                continue
            
            visited[i] = True  # Mark the number as used
            current.append(nums[i])  # Add the number to the current permutation
            
            self.backtrack(nums, result, current, visited)  # Recur with the updated current
            
            # Backtrack: unmark the number as used and remove it from current permutation
            visited[i] = False  
            current.pop()  

# Example usage
# sol = Solution()
# print(sol.permuteUnique([1,1,2]))
# print(sol.permuteUnique([1,2,3]))

```

### Explanation of the Code:

- **Sorting the nums**: Sorting is done to ensure that duplicates are adjacent. This allows us to implement the duplicate-checking logic effectively.

- **Visited Array**: We maintain an array `visited`, where each index corresponds to whether we have used the number in that position in the current path/potential permutation.

- **Recursive Backtracking**: The backtracking function constructs the permutations:
  - If the length of the current permutation matches the length of `nums`, it means we have a complete permutation to store.
  - We iterate through `nums`, skipping already visited indexes and ensuring we do not consider duplicate numbers in the same recursive hierarchy unless they are already included (i.e., the numbers are not adjacent, maintaining unique contributions).

- **Final Result**: The result contains all unique permutations, ready to be returned.

This implementation meets the constraints efficiently and ensures that all permutations are unique, suitable for submission on LeetCode.

# 51. N-Queens

### Problem Description 
The n-queens puzzle is the problem of placing `n` queens on an `n x n` chessboard such that no two queens attack each other.

Given an integer `n`, return all distinct solutions to the n-queens puzzle.

Each solution contains a distinct board configuration of the n-queens' placement, where `'Q'` and `'.'` both indicate a queen and an empty space, respectively.


Example 1:
Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above

Example 2:
Input: n = 1
Output: [["Q"]]

Constraints:
`1 <= n <= 9`

### Solution 
 To solve the N-Queens problem, we'll use a backtracking approach. The goal is to place `n` queens on an `n x n` chessboard such that no two queens threaten each other. This means no two queens can be in the same row, the same column, or on the same diagonal.

### Detailed Explanation:
1. **Backtracking**: We'll attempt to place queens in each row. For each row, we'll try to place a queen in every column one by one, and check if that position is safe.

2. **Checking Safety**: For a position `(row, col)`, we need to ensure that:
   - No queen is placed in the same column `col`.
   - No queen is placed on the upper-left diagonal (which can be checked by `(row - i == col - j)`).
   - No queen is placed on the upper-right diagonal (which can be checked by `(row - i == j - col)`).
   Where `i` is the row index of previously placed queens and `j` is their respective column positions.

3. **Board Representation**: We can represent the chessboard using strings for simplicity, where 'Q' denotes a queen and '.' denotes an empty space.

4. **Base Case**: When we reach the last row (i.e., `row == n`), we store the current configuration of the board as a valid solution.

5. **Output**: We will return a list of all possible configurations once all rows are processed.

Here’s the Python code that implements the above logic according to the LeetCode format:



```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def backtrack(row: int, columns: set, diagonals1: set, diagonals2: set, board: List[str]):
            if row == n:  # All queens are placed
                result.append(board[:])  # Add a copy of the current board to the result
                return
            
            for col in range(n):
                # Calculate the identifiers for the diagonals
                diagonal1 = row - col  # Left diagonal (from top-left to bottom-right)
                diagonal2 = row + col  # Right diagonal (from top-right to bottom-left)

                if col in columns or diagonal1 in diagonals1 or diagonal2 in diagonals2:
                    continue  # This position is under attack, skip it

                # Place the queen
                columns.add(col)
                diagonals1.add(diagonal1)
                diagonals2.add(diagonal2)
                board.append("." * col + "Q" + "." * (n - col - 1))  # Create the board configuration
                
                # Recur to place queens in the next row
                backtrack(row + 1, columns, diagonals1, diagonals2, board)
                
                # Backtrack: remove the queen and reset the board
                columns.remove(col)
                diagonals1.remove(diagonal1)
                diagonals2.remove(diagonal2)
                board.pop()

        result = []
        backtrack(0, set(), set(), set(), [])
        return result

```

### Explanation of the Code:
- **Class Definition**: We define a `Solution` class, as per LeetCode's requirements.
- **Main Function**: `solveNQueens` takes an integer `n` and initializes the backtracking process.
- **Backtracking Function**: The inner function `backtrack` handles the recursive placement of queens:
  - It checks for safety before placing a queen.
  - If a valid configuration is found (i.e., all queens placed), it appends a copy of the current board to the results.
  - After trying a position, it removes the queen (backtracks) to try other possibilities.
- **Result Collection**: The function returns all configurations found.

### Complexity:
- The time complexity is O(N!), as in the worst-case scenario, we might end up exploring every permutation of placing queens in rows.
- The space complexity is O(N), due to the recursion stack and storage for the solutions. 

This code can be directly run on the LeetCode platform and will produce the required output for the N-Queens problem.

# 52. N-Queens II

### Problem Description 
The n-queens puzzle is the problem of placing `n` queens on an `n x n` chessboard such that no two queens attack each other.

Given an integer `n`, return the number of distinct solutions to the n-queens puzzle.


Example 1:
Input: n = 4
Output: 2
Explanation: There are two distinct solutions to the 4-queens puzzle as shown.


Example 2:
Input: n = 1
Output: 1

Constraints:
`1 <= n <= 9`

### Solution 
 Certainly! To solve the n-queens problem, we can employ a backtracking algorithm. The idea is to place queens on the board one row at a time, ensuring that no two queens threaten each other. This can be achieved by checking column restrictions and the two diagonals where queens can attack.

### Detailed Explanation:

1. **Understanding the Problem**:
   - We need to place `n` queens on an `n x n` chessboard.
   - A queen can attack another queen if they are in the same column or if they share a diagonal.

2. **Backtracking Approach**:
   - We will define a recursive function that attempts to place a queen in each row one by one.
   - We will keep track of:
     - Columns that are already occupied by queens.
     - Both diagonals (the left diagonal and the right diagonal) that queens can threaten.
   - For each row, we will try placing a queen in each column and check if it is safe to do so.

3. **Base Case**:
   - If we have placed queens in all `n` rows, we have found a valid configuration, and we can increment our solution count.

4. **Recursive Exploration**:
   - For each column `c` in the current row `r`, check if placing a queen there is valid (not under attack).
   - If valid, mark the column and both diagonals as occupied and recurse to the next row.
   - After returning from recursion, unmark the column and diagonal to explore further possibilities.

### Implementation:

Here's how this logic translates into Python code, formatted to run directly on LeetCode:



```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        # This will hold the number of distinct solutions
        self.count = 0
        
        # Initialize sets to check for column and diagonal occupancies
        self.columns = set()
        self.left_diagonal = set()
        self.right_diagonal = set()
        
        def backtrack(row: int):
            # If all queens are placed, we've found a solution
            if row == n:
                self.count += 1
                return
            
            # Try placing the queen in each column for the given row
            for col in range(n):
                if col in self.columns or (row - col) in self.left_diagonal or (row + col) in self.right_diagonal:
                    continue  # Can't place here, it is under attack
                
                # Place the queen
                self.columns.add(col)
                self.left_diagonal.add(row - col)
                self.right_diagonal.add(row + col)
                
                # Move to the next row
                backtrack(row + 1)
                
                # Remove the queen and backtrack
                self.columns.remove(col)
                self.left_diagonal.remove(row - col)
                self.right_diagonal.remove(row + col)
        
        backtrack(0)  # Start from the first row
        return self.count  # Return the total number of solutions

```

### Explanation of the Code:
- We define a class `Solution` with a method `totalNQueens` that takes an integer `n`.
- We initialize a counter `self.count` to keep track of valid configurations and sets to keep track of occupied columns and diagonals.
- The nested function `backtrack(row)` implements the backtracking algorithm; it checks each column of the current row to see where a queen can be placed without conflict.
- If a queen is placed, we recursively call `backtrack` for the next row. This continues until all queens are placed (base case).
- After exploring all options, we remove the queen (backtrack) and proceed to alternate placements.
- Finally, the method returns the count of distinct solutions.

This solution can efficiently find the number of distinct solutions for `n` queens within the constraints provided.

# 60. Permutation Sequence

### Problem Description 
The set `[1, 2, 3, ..., n]` contains a total of `n!` unique permutations.

By listing and labeling all of the permutations in order, we get the following sequence for `n = 3`:
`"123"`
`"132"`
`"213"`
`"231"`
`"312"`
`"321"`
Given `n` and `k`, return the `kth` permutation sequence.


Example 1:
Input: n = 3, k = 3
Output: "213"

Example 2:
Input: n = 4, k = 9
Output: "2314"

Example 3:
Input: n = 3, k = 1
Output: "123"

Constraints:
`1 <= n <= 9`
`1 <= k <= n!`

### Solution 
 To solve the problem of finding the k-th permutation of the sequence `[1, 2, ..., n]`, we can utilize a systematic approach that leverages the properties of permutations and factorial number systems.

## Explanation

1. **Understanding Factorial Representation**: The permutations of the sequence can be thought of in terms of blocks based on the starting number. For `n` numbers, there are `n!` total permutations and every number `i` from `1` to `n` will lead a block of `(n-1)!` permutations when fixed at the front.

2. **Determine the Starting Number**:
   - When trying to build the k-th permutation, we can determine which number to place first by dividing `k-1` (zero-based index) by `(n-1)!`. This value gives us the index of the starting number in the list of available numbers.

3. **Update the K and Reduce the Problem Size**:
   - After determining which number should be placed first, we adjust `k` for the next position by taking the remainder of the division of `k-1` by `(n-1)!`. We also need to remove the chosen number from our list of available numbers.

4. **Repeat**: We reduce `n` by 1 and keep repeating this process until we build the full permutation sequence.

5. **Edge Cases**: Since `k` is guaranteed to be valid (between `1` and `n!`), we do not require extra checks for inputs.

Here is the Python implementation that follows the above logic:



```python
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        # Initialize the list of numbers from 1 to n
        numbers = list(range(1, n + 1))
        # Initialize the result
        permutation = []
        
        # Decrement k to convert to zero-based index
        k -= 1
        
        # Precompute factorials up to n
        factorial = [1] * n
        for i in range(1, n):
            factorial[i] = factorial[i - 1] * i
        
        for i in range(n, 0, -1):
            # Find the index of the current digit
            index = k // factorial[i - 1]
            # Append the current number to the result
            permutation.append(str(numbers[index]))
            # Remove the used number from the list
            numbers.pop(index)
            # Update k
            k %= factorial[i - 1]
        
        return ''.join(permutation)

# Example usage:
solution = Solution()
print(solution.getPermutation(3, 3))  # Output: "213"
print(solution.getPermutation(4, 9))  # Output: "2314"
print(solution.getPermutation(3, 1))  # Output: "123"

```

## Code Explanation:
- We start by initializing a list of numbers from `1` to `n`.
- We also initialize a list to hold the resulting permutation.
- We decrement `k` by 1 to use it in a zero-based manner.
- We compute the factorial values up to `n` to determine how many permutations start with each number.
- By looping from `n` down to `1`, we continually figure out which number to select and update `k`.
- Finally, we join the result list into a string and return it.

The overall time complexity of this solution is `O(n^2)` due to the pop operation, while the factorial precomputation is `O(n)`. This is efficient considering the constraints (`1 <= n <= 9`).

# 77. Combinations

### Problem Description 
Given two integers `n` and `k`, return all possible combinations of `k` numbers out of the range `[1, n]`.

You may return the answer in any order.


Example 1:
Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]

Example 2:
Input: n = 1, k = 1
Output: [[1]]

Constraints:
`1 <= n <= 20`
`1 <= k <= n`

### Solution 
 To solve the problem of finding all possible combinations of `k` numbers from the range `[1, n]`, we can use a backtracking approach. Backtracking is an effective method for generating combinations because it allows us to construct solutions incrementally and back out as needed.

### Detailed Explanation:

1. **Understanding the problem**: 
   - We need to select `k` unique numbers from the range `1` to `n`.
   - The order of numbers in the combinations does not matter. For example, `[1, 2]` is the same as `[2, 1]`.

2. **Approach**:
   - Use backtracking to generate combinations. We will maintain a current combination and explore adding one number at a time until we reach the required length `k`.
   - Start from the first number and try to build combinations by adding the next available numbers to the current combination.
   - Once the current combination reaches a length of `k`, we will add it to our results list.

3. **Implementation**:
   - We'll create a recursive function that takes the current combination, the start point for the next number, and the results list.
   - We will iterate over the possible numbers, add the current number to our combination, and recursively call our function to continue building the combination.
   - After exploring the combination, we backtrack by removing the last added number and trying the next number.

### Python Code:



```python
class Solution:
    def combine(self, n: int, k: int):
        def backtrack(start: int, current_combination: List[int]):
            # If the combination is complete
            if len(current_combination) == k:
                results.append(current_combination[:])
                return
            
            # Iterate from the current number (start) to n
            for i in range(start, n + 1):
                # Include the number in the current combination
                current_combination.append(i)
                # Continue to build combination with the next numbers
                backtrack(i + 1, current_combination)
                # Backtrack - remove the last added number
                current_combination.pop()
        
        results = []
        backtrack(1, [])
        return results

```

### Explanation of the Code:

- **Class and Method**: We define a class `Solution` and a method `combine` that takes `n` and `k` as arguments.
- **Backtracking Function**: Inside `combine`, we define a helper function `backtrack` which will do the main work:
  - `start`: This keeps track of the next number to consider for adding to the combination.
  - `current_combination`: This is our ongoing combination as we build it.
- **Base Case**: When the current combination reaches the length of `k`, we add a copy of it to the results.
- **Loop**: We loop from `start` to `n`, adding each number to `current_combination`, then calling `backtrack` recursively.
- **Backtracking**: After returning from the recursive call, we simply pop the last number from `current_combination` to explore other combinations.

### Complexity:

- The time complexity of this algorithm is O(C(n, k)), where C(n, k) is the binomial coefficient representing the number of combinations, and the space complexity is O(k) due to the recursion stack and the storage of combinations.

This code can be run directly on LeetCode under the problem that describes generating combinations and should work efficiently within the given constraints.

# 78. Subsets

### Problem Description 
Given an integer array `nums` of unique elements, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.


Example 1:
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

Example 2:
Input: nums = [0]
Output: [[],[0]]

Constraints:
`1 <= nums.length <= 10`
`-10 <= nums[i] <= 10`
All the numbers of `nums` are unique.

### Solution 
 To solve the problem of generating all possible subsets (also known as the power set) of a given list of unique integers, we can use a backtracking approach. This allows us to explore all potential combinations of the elements in the provided list.

### Explanation of the Approach

1. **Understanding Subsets**: A subset can either include an element or exclude it. Hence, for each element in the array, we have a choice: to include it in the current subset or to leave it out.

2. **Backtracking**: We'll use a backtracking function to build the subsets. The idea is to recursively explore each element, and at each step, decide whether to include the current element in the subset or not.

3. **Base Case**: When we've considered all elements (i.e., when we reach the end of the array), we'll add the current subset to our list of results.

4. **Result Collection**: We’ll maintain a list to collect all generated subsets, starting with an empty subset.

### Implementation

We'll define the function according to LeetCode's requirements, using a nested helper function for the backtracking.

Here is the implementation in Python:



```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = []  # This will store all the subsets
        current_subset = []  # This array keeps track of the current subset

        def backtrack(start: int):
            # Append the current subset to the result
            result.append(list(current_subset))

            # Explore further subsets
            for i in range(start, len(nums)):
                # Include nums[i] in the current subset
                current_subset.append(nums[i])
                
                # Move to the next element
                backtrack(i + 1)
                
                # Backtrack: remove the last added element
                current_subset.pop()

        # Start backtracking from the first index
        backtrack(0)
        
        return result

```

### Breakdown of the Code

1. **Class Definition**: The solution is encapsulated in the class `Solution`, which is a standard format for LeetCode submissions.

2. **Function Signature**: The function `subsets` takes a list of integers `nums` and returns a list of lists (subsets).

3. **Initialization**: 
   - `result`: This will store all subsets collected during the backtracking process.
   - `current_subset`: This keeps track of the current subset being built.

4. **Backtrack Function**: 
   - The nested `backtrack` function takes a starting index `start` as an argument.
   - It appends a copy of the current subset to `result` to save the current state.
   - It then iterates over the elements starting from `start` to the end of the list:
     - Adds the current element to `current_subset`.
     - Recursively calls `backtrack` to continue building the subset with the next elements.
     - After the recursive call, it removes the last added element to explore other subsets (backtracking).

5. **Starting the Process**: The `backtrack` function is initially called with `0`, indicating that we start considering elements from the beginning of the list.

6. **Return Statement**: Finally, return `result`, which will contain all possible subsets of `nums`.

### Complexity Analysis
- **Time Complexity**: O(2^n), where n is the number of elements in `nums`. This is because each element can either be included or excluded from a subset.
- **Space Complexity**: O(n), primarily due to the space used by the recursion stack and storing the current subset.

This method efficiently constructs all possible subsets while adhering to the problem's constraints and ensuring no duplicate subsets are created.

# 79. Word Search

### Problem Description 
Given an `m x n` grid of characters `board` and a string `word`, return `true` if `word` exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.


Example 1:
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true

Example 2:
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
Output: true

Example 3:
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
Output: false

Constraints:
`m == board.length`
`n = board[i].length`
`1 <= m, n <= 6`
`1 <= word.length <= 15`
`board` and `word` consists of only lowercase and uppercase English letters.

Follow up: Could you use search pruning to make your solution faster with a larger `board`?

### Solution 
 To solve the problem of checking if a given word exists in a 2D grid (board) of characters, we can employ a depth-first search (DFS) algorithm. This algorithm will explore all possible paths starting from each cell in the grid while adhering to the constraints that letters must be adjacent horizontally or vertically and that the same cell cannot be reused in constructing the word.

Here's the step-by-step breakdown of how we will implement the solution:

1. **Iterate through each cell in the grid**: For each starting point in the grid, we will invoke a DFS function that will try to match the word.

2. **Depth-First Search (DFS) function**: This function will take parameters including the current position in the grid, the current index of the word, and the visited status of cells to track which cells have already been utilized in the current path.

3. **Base Conditions**: 
   - If the index matches the length of the word, it means we have successfully matched the entire word, so we return `True`.
   - If the current position is out of bounds or the character in the grid does not match the current character of the word, we return `False`.

4. **Mark the Cell as Visited**: We will use a set of coordinates to keep track of visited cells in the current path to avoid reusing them.

5. **Recursive Calls**: From the current position, perform recursive calls to all four possible directions (up, down, left, right) to continue matching the next character of the word.

6. **Unmark the Cell**: After searching all possible directions from the current cell, we unmark it (allow it to be visited again in other paths).

7. **Return Result**: If we find the word starting from any cell, return `True`. If none of the paths match the word, return `False`.

Here is the implementation of the above logic in a format suitable for LeetCode:



```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board or not board[0]:
            return False
        
        # Dimensions of the board
        m, n = len(board), len(board[0])
        
        # Directions for moving in the board: up, down, left, right
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def dfs(x: int, y: int, index: int) -> bool:
            # If we have matched all characters of the word
            if index == len(word):
                return True
            
            # Out of bounds or character does not match
            if x < 0 or x >= m or y < 0 or y >= n or board[x][y] != word[index]:
                return False
            
            # To avoid using the same cell, we temporarily mark it
            temp = board[x][y]
            board[x][y] = "#"
            
            # Explore all possible directions
            for dx, dy in directions:
                if dfs(x + dx, y + dy, index + 1):
                    return True
            
            # Unmark the cell after exploring
            board[x][y] = temp
            
            return False
        
        # Try starting the search from each cell
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):  # Start searching for the word
                    return True
        
        return False

```

### Explanation:
- We define a helper function `dfs` that attempts to match the word starting from a particular cell.
- The tracking of visited cells is done using temporary marking (changing the cell value). This avoids using sets or additional data structures, simplifying the logic.
- The code runs efficiently within the problem's constraints, ensuring that each path is exhaustively explored up to the characters in the given word. The recursive nature of the solution allows us to explore all potential paths while adhering to the adjacency and usage rules.

# 89. Gray Code

### Problem Description 
The gray code is a binary numeral system where two successive values differ in only one bit.

Given an integer `n` representing the total number of bits in the code, return any sequence of gray code.

A gray code sequence must begin with `0`.


Example 1:
Input: n = 2
Output: [0,1,3,2]
Explanation:
00 - 0
01 - 1
11 - 3
10 - 2
[0,2,3,1] is also a valid gray code sequence.

00 - 0
10 - 2
11 - 3
01 - 1

Example 2:
Input: n = 1
Output: [0,1]

Constraints:
`1 <= n <= 16`

### Solution 
 To generate a Gray code sequence of `n` bits in Python, we can utilize a well-known mathematical pattern. The Gray code sequence can be constructed using the following approach:

1. **Understanding Gray Code**: In Gray code, every two successive values differ in one bit. For any integer `n`, the Gray code for a number `i` can be computed using the formula:  
   \[
   \text{Gray}(i) = i \oplus (i >> 1)
   \]
   where `⊕` is the bitwise XOR operation and `>>` represents the right shift operation.

2. **Building the Sequence**: For a given `n`, we need to generate numbers from `0` to `2^n - 1`, apply the Gray code formula to each number, and return the result as a list.

3. **Complexity**: The time complexity of generating the Gray code sequence is \(O(2^n)\), as we need to compute the Gray code for each number from `0` to `2^n - 1`.

Here is the implementation of the above steps in Python, formatted to be directly runnable on LeetCode:



```python
class Solution:
    def grayCode(self, n: int) -> List[int]:
        # Initialize the output list
        gray_codes = []
        
        # We loop through all numbers from 0 to 2^n - 1
        for i in range(1 << n):  # This is equal to 2**n
            # Compute the Gray code using the formula: i ^ (i >> 1)
            gray_code = i ^ (i >> 1)
            gray_codes.append(gray_code)
        
        return gray_codes

```

### Explanation of the Code:
1. **Initialization**: We create an empty list `gray_codes` that will hold our resulting Gray code sequence.
  
2. **Loop Through Numbers**: We use a loop that iterates from `0` to `2^n - 1`. The expression `1 << n` calculates \(2^n\) by shifting `1` to the left `n` times.

3. **Compute Gray Code**: For each integer `i`, we calculate its corresponding Gray code using the formula `i ^ (i >> 1)` and store that in the `gray_codes` list.

4. **Return the Result**: Finally, we return the list of Gray codes.

### Example Usage:
- When `n = 2`, the output will be `[0, 1, 3, 2]` or any other valid Gray code sequence.
- When `n = 1`, the output will be `[0, 1]`. 

This code runs efficiently for the given constraints and adheres to the requirements of the problem statement.

# 90. Subsets II

### Problem Description 
Given an integer array `nums` that may contain duplicates, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.


Example 1:
Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]

Example 2:
Input: nums = [0]
Output: [[],[0]]

Constraints:
`1 <= nums.length <= 10`
`-10 <= nums[i] <= 10`

### Solution 
 To solve the problem of generating all possible subsets from an integer array that may contain duplicates, we can use a backtracking approach. The key is to ensure that we only generate unique subsets, which can be achieved by sorting the input array and skipping over duplicates during the recursive exploration.

### Solution Outline:
1. **Sort the Input:** By sorting the array, we can easily identify duplicates. If we encounter a duplicate element in a certain recursion level, we can skip it to avoid generating the same subset again.
  
2. **Backtracking Function:** We define a recursive function that constructs subsets by including or excluding each element. When we include an element, we move to the next index, and when we exclude it, we also move to the next index.

3. **Base Case:** The base case for our recursion will involve adding the current subset to the results. We start by adding an empty subset.

4. **Handling Duplicates:** While iterating through the array, if we encounter the same number as the previous one, we skip over it when adding to our current subset.

### Implementation:
Below is the Python implementation formatted in a way that can be directly executed on LeetCode:



```python
from typing import List

class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        # Sort the array to handle duplicates
        nums.sort()
        res = []
        
        def backtrack(start: int, path: List[int]):
            # Add the current subset to the results
            res.append(path[:])  # Append a copy of the current path
            
            for i in range(start, len(nums)):
                # If we encounter a duplicate, we skip it
                if i > start and nums[i] == nums[i - 1]:
                    continue
                
                # Include nums[i] in the subset
                path.append(nums[i])
                # Move on to the next elements
                backtrack(i + 1, path)
                # Backtrack: Remove the last element added
                path.pop()
        
        backtrack(0, [])
        return res

```

### Explanation of the Code:
1. **Input Handling:** The class `Solution` contains the function `subsetsWithDup` which takes a list of integers `nums` as input.

2. **Sorting the Array:** The input array `nums` is sorted to ensure that duplicates are adjacent.

3. **Defining the Res Variable:** `res` is initialized as an empty list to store all the unique subsets.

4. **Backtracking Function `backtrack`:**
   - **Parameters:** It takes `start` (the index to start searching for subsets) and `path` (the current subset being built).
   - **Adding Current Path:** We append a copy of `path` to `res`.
   - **Looping Through Elements:** We loop over indices from `start` to the length of `nums`.
   - **Skipping Duplicates:** If the current element is the same as the previous one, we skip it to avoid duplicate subsets.
   - **Recursion:** We include the current element in `path`, then recursively call `backtrack` with the next index.
   - **Backtracking Step:** After returning from the recursion, we remove the last element added to the `path` to explore other possibilities.

5. **Return the Result:** Finally, the function returns the list of subsets stored in `res`.

### Complexity:
- **Time Complexity:** O(2^N), where N is the number of elements in `nums`. This is because each element can either be included or excluded from a subset.
- **Space Complexity:** O(N), due to the space used for the recursion stack and the result list.

This implementation effectively generates all unique subsets from the input array while ensuring no duplicates are included in the results.

# 93. Restore IP Addresses

### Problem Description 
Given a string `s` containing only digits, return all possible valid IP addresses that can be obtained from `s`. You can return them in any order.

A valid IP address consists of exactly four integers, each integer is between `0` and `255`, separated by single dots and cannot have leading zeros. For example, "0.1.2.201" and "192.168.1.1" are valid IP addresses and "0.011.255.245", "192.168.1.312" and "192.168@1.1" are invalid IP addresses. 

Example 1:
Input: s = "25525511135"
Output: ["255.255.11.135","255.255.111.35"]

Example 2:
Input: s = "0000"
Output: ["0.0.0.0"]

Example 3:
Input: s = "1111"
Output: ["1.1.1.1"]

Example 4:
Input: s = "010010"
Output: ["0.10.0.10","0.100.1.0"]

Example 5:
Input: s = "101023"
Output: ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]

Constraints:
`0 <= s.length <= 3000`
`s` consists of digits only.

### Solution 
 To solve the problem of generating all possible valid IP addresses from a given string of digits, we need to break the string into four parts, each representing one of the octets of the IP address. Each part must satisfy the following criteria:

1. It must be an integer in the range [0, 255].
2. It cannot contain leading zeros unless the part itself is '0'.

### Approach

We will use a backtracking approach to explore all the potential ways to split the input string into four segments. Here’s the step-by-step breakdown of our solution:

1. **Base Conditions**: If the length of the string does not allow for four valid IP parts, we can immediately return an empty list.
  
2. **Backtracking Function**:
   - We define a recursive function to build the address step by step.
   - We keep track of the current position in the string and the number of segments formed.
   - If we form four segments and have consumed the entire string, we consider the current segments as a valid IP address and store it.
  
3. **Segment Validation**:
   - For each segment, slice a portion of the string and check if it is valid:
     - It should not be longer than 3 digits.
     - It should be between 0 and 255.
     - It should not have leading zeros if it's more than one digit long.

4. **Recursive Calls**: For each valid segment, we recursively call our function to form the next segment with the remaining string.

5. **Result Collection**: Collect all valid IP addresses and return them.

Here is the Python code that implements the above logic in the required LeetCode format:



```python
class Solution:
    def restoreIpAddresses(self, s: str) -> list[str]:
        def backtrack(start: int, path: list[str]):
            # If we have 4 segments and we've used the entire string
            if len(path) == 4:
                if start == len(s):
                    result.append('.'.join(path))
                return
            
            # If we already have 4 segments and we are not at the end
            if len(path) > 4:
                return
            
            # Try every possible length for the next segment (1-3)
            for length in range(1, 4):
                if start + length > len(s):  # Out of bounds
                    break
                part = s[start:start + length]  # Get the next segment
                
                # Validate the part
                if (len(part) > 1 and part[0] == '0') or (int(part) > 255):  # Leading zero or invalid
                    continue
                
                # Proceed with the next part
                path.append(part)
                backtrack(start + length, path)  # Move to the next segment
                path.pop()  # Backtrack
        
        result = []
        backtrack(0, [])
        return result

```

### Explanation of the Code
- **Class and Method Definition**: We define a class `Solution` and a method `restoreIpAddresses` that takes a string `s` as input and returns a list of valid IP addresses.
- **Backtracking Helper Function**: The inner function `backtrack` is responsible for generating the IP segments.
  - It uses parameters `start` to track the current position in the string and `path` to build the currently formed segments.
- **Segment Checks**: For each recursive call, it checks if the segment formed is valid and decides whether to continue to the next part.
- **Result Aggregation**: Once a valid IP is formed, it is joined by dots and added to the result list.

This code will run efficiently for the stated input constraints and meet the requirements necessary to generate valid IP addresses from the input string.

# 126. Word Ladder II

### Problem Description 
A transformation sequence from word `beginWord` to word `endWord` using a dictionary `wordList` is a sequence of words `beginWord -> s1 -> s2 -> ... -> sk` such that:
Every adjacent pair of words differs by a single letter.

Every `si` for `1 <= i <= k` is in `wordList`. Note that `beginWord` does not need to be in `wordList`.

`sk == endWord`
Given two words, `beginWord` and `endWord`, and a dictionary `wordList`, return all the shortest transformation sequences from `beginWord` to `endWord`, or an empty list if no such sequence exists. Each sequence should be returned as a list of the words `[beginWord, s1, s2, ..., sk]`.


Example 1:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
Explanation: There are 2 shortest transformation sequences:
"hit" -> "hot" -> "dot" -> "dog" -> "cog"
"hit" -> "hot" -> "lot" -> "log" -> "cog"

Example 2:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
Output: []
Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.


Constraints:
`1 <= beginWord.length <= 10`
`endWord.length == beginWord.length`
`1 <= wordList.length <= 5000`
`wordList[i].length == beginWord.length`
`beginWord`, `endWord`, and `wordList[i]` consist of lowercase English letters.

`beginWord != endWord`
All the words in `wordList` are unique.

### Solution 
 To solve the problem of finding all the shortest transformation sequences from `beginWord` to `endWord`, we can use a combination of Breadth-First Search (BFS) to find the shortest paths and Depth-First Search (DFS) to construct the paths.

Here's a structured explanation of how we can tackle this problem:

### Approach

1. **BFS to Find All Shortest Paths**:
   - We use BFS to explore the word transformation graph level by level. For each word, we generate all possible transformations by changing one letter at a time and check if the transformed word is in the word list.
   - We also maintain a parent mapping that records from which word we arrived at each word. This will help us during the path reconstruction step.
   
2. **Path Reconstruction**:
   - Once we find the shortest length of the transformations, we can use DFS to reconstruct all valid paths from `beginWord` to `endWord` using the parent mapping created during the BFS.

3. **Edge Cases**:
   - If `endWord` is not in the `wordList`, we immediately return an empty list since no valid transformations can occur.

### Code Implementation

Here's the complete Python code following the LeetCode solution format:



```python
from collections import defaultdict, deque

class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: list[str]) -> list[list[str]]:
        wordSet = set(wordList)
        
        if endWord not in wordSet:
            return []

        # Step 1: BFS to find the shortest paths
        graph = defaultdict(list)
        found_end = False
        level = {beginWord}
        while level and not found_end:
            next_level = set()
            for word in level:
                for i in range(len(word)):
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        new_word = word[:i] + c + word[i+1:]
                        if new_word in wordSet:
                            if new_word == endWord:
                                found_end = True
                            if new_word not in graph[word]:  # Avoid adding duplicates
                                graph[word].append(new_word)
                            next_level.add(new_word)
            wordSet -= next_level  # Remove found words to prevent cycles
            level = next_level

        # Step 2: DFS to build the paths
        def dfs(word):
            if word == beginWord:
                return [[word]]
            return [[beginWord] + path for prev_word in graph[word] for path in dfs(prev_word)]

        return dfs(endWord)

# Example usage:
# sol = Solution()
# print(sol.findLadders("hit", "cog", ["hot","dot","dog","lot","log","cog"]))
# Output: [["hit","hot","dot","dog","cog"], ["hit","hot","lot","log","cog"]]

```

### Explanation of the Code

1. **Initialization**:
   - We use a `defaultdict` to create a graph where keys are words and values are lists of words that can be reached from the key by a single letter change.
   - We convert `wordList` into a set for O(1) lookups.

2. **BFS Loop**:
   - We initialize the BFS with the `beginWord`. For each word in the current level, we explore all possible one-letter transformations.
   - We build the graph by recording transformations only if they are present in the `wordSet`.
   - The `next_level` accumulates all the words that can be reached from the current level, facilitating breadth-first traversal.
   - We stop the BFS when `endWord` is found or when there are no more levels to explore.

3. **DFS function**:
   - This function recursively constructs all paths from `endWord` back to `beginWord` based on the relationships established during BFS.
   - For each previous word leading to the current word, we construct paths using list concatenation.

4. **Return Result**:
   - Finally, we obtain and return all paths found by traversing the graph.

### Conclusion
This solution combines BFS and DFS effectively to find all the shortest transformation sequences in a structured way. The use of a graph (adjacency list) allows efficient path retrieval for backtracking, leading to a clear implementation of the requirements given in the problem description.

# 131. Palindrome Partitioning

### Problem Description 
Given a string `s`, partition `s` such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of `s`.

A palindrome string is a string that reads the same backward as forward.


Example 1:
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]

Example 2:
Input: s = "a"
Output: [["a"]]

Constraints:
`1 <= s.length <= 16`
`s` contains only lowercase English letters.

### Solution 
 To solve the problem of partitioning a string such that every substring is a palindrome, we can use a backtracking approach. The main idea is to explore all possible partition combinations, and for each combination, check if all the substrings are palindromes. If they are, we add the combination to the result.

Here's how we can implement the solution step-by-step:

1. **Check for Palindrome**: We need a helper function to check if a substring is a palindrome.
2. **Backtracking Function**: We will use a recursive function to explore all partition combinations. This function will take the current starting index and the current path of substrings as input.
3. **Base Case**: If the starting index is equal to the length of the string, we've found a valid partition and we can add it to the results.
4. **Explore Partitions**: For each substring from the current starting point to the end of the string, check if it's a palindrome. If it is, we will include it in our current path and recursively call the function for the next starting index.

Here’s the complete code:



```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def is_palindrome(substring):
            return substring == substring[::-1]
        
        def backtrack(start, path):
            if start == len(s):
                result.append(path.copy())
                return
            
            for end in range(start + 1, len(s) + 1):
                substring = s[start:end]
                if is_palindrome(substring):
                    path.append(substring)  # choose
                    backtrack(end, path)     # explore
                    path.pop()               # unchoose
        
        result = []
        backtrack(0, [])
        return result

```

### Explanation of the Code:

- **is_palindrome Function**: It checks if a substring is a palindrome by comparing it to its reverse.
  
- **backtrack Function**: This is a recursive function that performs the following:
  - If the starting index `start` is equal to the length of the string `s`, it means we have a complete partition and we add a copy of the path to the `result`.
  - The `for` loop iterates over the possible end indices to create substrings from the current start index. 
  - If the substring formed is a palindrome, it is added to the current path. We then recursively call `backtrack` with the new starting index (which is `end`).
  - After exploring that path, we backtrack by removing the last added substring from the path.

- **Main Function**: It initializes an empty list `result` to hold all valid partitions and calls `backtrack` starting from index `0` with an empty path.

### Complexity Analysis:
The time complexity is O(2^n) because in the worst case, for every character, we can choose to include it or not in the current substring, resulting in 2^n possible substrings. The space complexity is O(n) for the recursion stack and the stored result.

This solution effectively generates all possible palindromic partitions for a given string and adheres to the LeetCode submission format.

# 140. Word Break II

### Problem Description 
Given a string `s` and a dictionary of strings `wordDict`, add spaces in `s` to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in any order.

Note that the same word in the dictionary may be reused multiple times in the segmentation.


Example 1:
Input: s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
Output: ["cats and dog","cat sand dog"]

Example 2:
Input: s = "pineapplepenapple", wordDict = ["apple","pen","applepen","pine","pineapple"]
Output: ["pine apple pen apple","pineapple pen apple","pine applepen apple"]
Explanation: Note that you are allowed to reuse a dictionary word.


Example 3:
Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: []

Constraints:
`1 <= s.length <= 20`
`1 <= wordDict.length <= 1000`
`1 <= wordDict[i].length <= 10`
`s` and `wordDict[i]` consist of only lowercase English letters.

All the strings of `wordDict` are unique.

### Solution 
 To solve the problem of segmenting a string `s` into valid words from a dictionary `wordDict`, we can use backtracking to explore all possible ways to split the string.

### Approach:
1. **Backtracking**: We will create a recursive function that tries to build valid sentences by checking every possible prefix of the string `s`. If a prefix is found in the `wordDict`, we recursively call the function on the remaining substring.

2. **Base Condition**: If we reach the end of the string, we add the constructed sentence to our results list.

3. **Using a Set for Fast Lookups**: We'll convert `wordDict` into a set for O(1) lookup time.

4. **Avoiding Redundant Work**: We can use memoization to store results for substrings we’ve already computed, reducing duplicate computations.

### Detailed Steps:
1. Initialize the results list to store valid sentences.
2. Define a recursive function that:
   - Takes the current index, the current sentence, and the result list as parameters.
   - Loops through possible end indices for substrings starting from the current index.
   - Checks if the substring is in `wordDict`.
   - If it is, recursively call the function with the new starting index (current end index).
   - If we reach the end of the string with a complete valid sentence, add it to the results list.
3. Finally, return the results list.

The final solution implementation is as follows:



```python
class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> list[str]:
        word_set = set(wordDict)
        memo = {}  # To memorize already computed results

        def backtrack(start):
            if start in memo:  # Return memoized results
                return memo[start]

            if start == len(s):  # If reached the end of the string
                return [""]  # Return a list with an empty string

            results = []

            # Try every possible end position for the current word
            for end in range(start + 1, len(s) + 1):
                word = s[start:end]
                
                if word in word_set:
                    # For valid word, continue backtracking with the remaining substring
                    for sub_sentence in backtrack(end):
                        # If sub_sentence is not empty, prepend a space
                        if sub_sentence:
                            results.append(word + " " + sub_sentence)
                        else:
                            results.append(word)  # For the last word which leads to end

            memo[start] = results  # Memoization
            return results

        return backtrack(0)

# Example usage:
# sol = Solution()
# print(sol.wordBreak("catsanddog", ["cat","cats","and","sand","dog"]))
# print(sol.wordBreak("pineapplepenapple", ["apple","pen","applepen","pine","pineapple"]))
# print(sol.wordBreak("catsandog", ["cats","dog","sand","and","cat"]))

```

### Explanation of the Implementation:
- We use a class `Solution` with a method `wordBreak`.
- Inside the `wordBreak` method, we convert `wordDict` into a set for quick access and initialize a memoization dictionary `memo`.
- The inner `backtrack` function handles the core logic:
  - If we've already computed sentences starting from `start`, we return those stored results.
  - When we reach the end of the string, we return a list containing an empty string.
  - We iterate over possible end indices to form words and check if they are in the `word_set`.
  - For each valid word, we recursively run `backtrack` on the remaining part of the string and combine results. 
- Finally, we call the `backtrack` starting from index `0` and return the complete list of segmented sentences.

This solution efficiently constructs all possible valid sentences from the given string and word dictionary, capturing all combinations allowed by the problem constraints.

# 211. Design Add and Search Words Data Structure

### Problem Description 
Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the `WordDictionary` class:
`WordDictionary()` Initializes the object.

`void addWord(word)` Adds `word` to the data structure, it can be matched later.

`bool search(word)` Returns `true` if there is any string in the data structure that matches `word` or `false` otherwise. `word` may contain dots `'.'` where dots can be matched with any letter.


Example:
Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]
Explanation
WordDictionary wordDictionary = new WordDictionary();
wordDictionary.addWord("bad");
wordDictionary.addWord("dad");
wordDictionary.addWord("mad");
wordDictionary.search("pad"); // return False
wordDictionary.search("bad"); // return True
wordDictionary.search(".ad"); // return True
wordDictionary.search("b.."); // return True

Constraints:
`1 <= word.length <= 500`
`word` in `addWord` consists lower-case English letters.

`word` in `search` consist of  `'.'` or lower-case English letters.

At most `50000` calls will be made to `addWord` and `search`.

### Solution 
 To solve this problem, we will implement a class `WordDictionary` that can efficiently add words and search for words with the potential presence of wildcard characters (dots `.`) that can match any character.

We will utilize a Trie (prefix tree) data structure to store the words as it allows efficient storage and search operations. Each node in the Trie will represent a character in a word. We will also add support for wildcard search, where a dot in the search query can match any character.

Here's a step-by-step explanation of our approach:

1. **Define the TrieNode Class**: This class represents each node in the Trie. Each node will maintain:
   - A dictionary of child nodes (for the next characters in words).
   - A boolean flag to indicate if a node represents the end of a word.

2. **Implement the WordDictionary Class**:
   - **Initialization**: Create a root node of the Trie.
   - **addWord(word)**: Traverse the Trie according to the characters in the word. If the character node does not exist, create a new node. After traversing all characters, mark the last node as the end of a word.
   - **search(word)**: This method handles both exact matches and wildcard searches. We will use a helper function that can perform backtracking to match each character and handle dots by recursively trying each possible character for the dot.

3. **Efficiency**: The `addWord` method operates in O(M) time complexity where M is the length of the word. The `search` operation, in the worst case, operates in O(M * 26^D) where D is the number of dots in the search string due to backtracking.

Now, let's implement this.



```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        return self._search_helper(word, 0, self.root)

    def _search_helper(self, word: str, index: int, node: TrieNode) -> bool:
        if index == len(word):
            return node.is_end_of_word
        
        char = word[index]
        if char == '.':
            for child in node.children.values():
                if self._search_helper(word, index + 1, child):
                    return True
            return False
        else:
            if char not in node.children:
                return False
            return self._search_helper(word, index + 1, node.children[char])

# Example usage:
# wordDictionary = WordDictionary()
# wordDictionary.addWord("bad")
# wordDictionary.addWord("dad")
# wordDictionary.addWord("mad")
# print(wordDictionary.search("pad")) # returns False
# print(wordDictionary.search("bad")) # returns True
# print(wordDictionary.search(".ad")) # returns True
# print(wordDictionary.search("b..")) # returns True

```

### Explanation of the Code:

- **TrieNode Class**: Contains a dictionary `children` to represent next characters and a boolean `is_end_of_word` to mark the end of a valid word.
  
- **WordDictionary Class**:
  - **`__init__`**: Initializes the root node for the Trie.
  - **`addWord`**: Iterates through each character in the input word and creates new nodes if they don't exist. Finally, it marks the endpoint of the word.
  - **`search`**: Initiates the recursive search helper method.
  - **`_search_helper`**: Recursively searches the Trie:
    - If we reach the end of the search string, we check if we've done an exact match up to this point.
    - If we encounter a dot, we explore all possible child nodes.
    - For any specific character, we progress through the corresponding child node.

This implementation supports adding and searching words efficiently and accommodates the wildcard character appropriately.

# 212. Word Search II

### Problem Description 
Given an `m x n` `board` of characters and a list of strings `words`, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.


Example 1:
Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]

Example 2:
Input: board = [["a","b"],["c","d"]], words = ["abcb"]
Output: []

Constraints:
`m == board.length`
`n == board[i].length`
`1 <= m, n <= 12`
`board[i][j]` is a lowercase English letter.

`1 <= words.length <= 3 * 104`
`1 <= words[i].length <= 10`
`words[i]` consists of lowercase English letters.

All the strings of `words` are unique.

### Solution 
 To solve the problem of finding all words from a given list that can be formed by sequentially adjacent letters on a board, we can use a combination of the Trie (prefix tree) data structure for efficient lookups and a Depth First Search (DFS) technique to explore the board. Below, I'll outline a detailed explanation of the approach, followed by the complete Python code that adheres to the LeetCode solution format.

### Approach:

1. **Trie Construction**:
   - First, we will construct a Trie to hold all the words from the given list. This will allow us to easily check if a sequence of letters from the board forms a valid prefix of any word in the list.

2. **Depth-First Search (DFS)**:
   - For each cell in the board, we will initiate a DFS to explore all possible paths. 
   - Starting from the current cell, we will move in four possible directions (up, down, left, right) while keeping track of the characters we visit to form potential words.
   - To prevent reusing the same cell in the board while constructing a word, we will mark cells as visited.

3. **Word Collection**:
   - During the DFS, if we construct a complete word that exists in our Trie and hasn't been found yet, we will add it to the results list.
   - We will also use a set to track which words have already been added to prevent duplicates.

4. **Early Stopping**:
   - As we traverse the Trie during the DFS, if we encounter a node that does not lead to any words, we stop exploring that path further.

### Complexity:
- The time complexity will primarily depend on the number of cells in the board and the number of words, leading to a complexity of O(M * N * 4^L), where M and N are the dimensions of the board and L is the length of the longest word, due to the DFS exploring all paths.
- Space complexity is mainly influenced by the Trie and the recursion stack, leading to O(W * K + M * N), where W is the number of words, K is the average length of words, and M * N for the board.

### Implementation:

Here is the Python code that implements the above approach:



```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        # Initialize the Trie
        trie = Trie()
        for word in words:
            trie.insert(word)
        
        self.m, self.n = len(board), len(board[0])
        self.result = set()  # To store found words
        self.visited = [[False] * self.n for _ in range(self.m)]
        
        # Directions for moving in the board (right, down, left, up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def dfs(x, y, node, path):
            if node.is_end_of_word:
                self.result.add(path)
                node.is_end_of_word = False  # Avoid duplicate entries
            
            # Temporarily mark the cell as visited
            self.visited[x][y] = True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.m and 0 <= ny < self.n and not self.visited[nx][ny]:
                    next_char = board[nx][ny]
                    if next_char in node.children:
                        dfs(nx, ny, node.children[next_char], path + next_char)
            # Unmark the cell as visited for other paths
            self.visited[x][y] = False

        for i in range(self.m):
            for j in range(self.n):
                start_char = board[i][j]
                if start_char in trie.root.children:
                    dfs(i, j, trie.root.children[start_char], start_char)
        
        return list(self.result)


```

### Explanation of the Code:
- **TrieNode** and **Trie** classes are used to create a trie structure to store all the input words.
- The `Solution` class has the key method `findWords` which executes the main logic.
- The nested DFS function explores possible words starting from each cell, and the use of a set `result` ensures no duplicates are collected.
- Finally, the list of found words is returned. 

This code should run efficiently within the constraints given in the problem statement.

# 216. Combination Sum III

### Problem Description 
Find all valid combinations of `k` numbers that sum up to `n` such that the following conditions are true:
Only numbers `1` through `9` are used.

Each number is used at most once.

Return a list of all possible valid combinations. The list must not contain the same combination twice, and the combinations may be returned in any order.


Example 1:
Input: k = 3, n = 7
Output: [[1,2,4]]
Explanation:
1 + 2 + 4 = 7
There are no other valid combinations.


Example 2:
Input: k = 3, n = 9
Output: [[1,2,6],[1,3,5],[2,3,4]]
Explanation:
1 + 2 + 6 = 9
1 + 3 + 5 = 9
2 + 3 + 4 = 9
There are no other valid combinations.


Example 3:
Input: k = 4, n = 1
Output: []
Explanation: There are no valid combinations. [1,2,1] is not valid because 1 is used twice.


Example 4:
Input: k = 3, n = 2
Output: []
Explanation: There are no valid combinations.


Example 5:
Input: k = 9, n = 45
Output: [[1,2,3,4,5,6,7,8,9]]
Explanation:
1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 = 45
​​​​​​​There are no other valid combinations.


Constraints:
`2 <= k <= 9`
`1 <= n <= 60`

### Solution 
 Here's a solution to the problem of finding all valid combinations of `k` numbers that sum up to `n`, using numbers from `1` to `9` without repetition. We will use backtracking to generate the combinations and gather those that meet the criteria.

## Python Code



```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        def backtrack(start, target, path):
            # If the combination is valid (k numbers and sums to n)
            if len(path) == k and target == 0:
                result.append(path[:])  # Add the current combination to result
            
            for num in range(start, 10):  # Numbers 1 through 9
                # If we have enough digits left to complete the combination
                if len(path) < k and num <= target:  
                    path.append(num)            # Choose the number
                    backtrack(num + 1, target - num, path)  # Explore further
                    path.pop()                  # Backtrack: remove the last chosen number

        result = []
        backtrack(1, n, [])  # Start from number 1, target sum n, empty path
        return result

```

## Explanation of the Code

### Class and Method Definition
- We define a class `Solution` which houses the method `combinationSum3`.
- This method takes in two parameters, `k` (the count of numbers) and `n` (the target sum).

### Backtracking Function
- A nested helper function `backtrack` is defined. This function takes parameters:
  - `start`: The starting number for the iteration (to ensure we only use numbers greater than the last chosen number).
  - `target`: The remaining sum we need to reach.
  - `path`: A list that keeps track of the current combination of numbers.

### Base Case
- Inside `backtrack`, we check if the current `path` has exactly `k` numbers and if the `target` is zero. If both conditions are satisfied, we record a copy of `path` into `result`.

### Loop and Recursive Backtracking
- We loop through numbers from `start` to `9`. For each `num`:
  - We check if adding `num` would keep our combination valid (if the length of `path` is less than `k` and `num` is less than or equal to `target`).
    - If valid, we choose `num` by appending it to `path`.
    - We then recursively call `backtrack`, moving to the next number (`num + 1`) and decreasing `target` by `num`.
    - This enables a depth-first search through valid combinations.
- After exploring, we backtrack by popping the last number from `path`, allowing us to explore other potential combinations.

### Result Initialization and Function Call
- An empty list `result` is initialized to store valid combinations.
- The backtracking starts with `1` as the initial number, our target `n`, and an empty combination `path`.
- Finally, we return the list of valid combinations found.

### Complexity
- The time complexity is combinatorial, as we explore subsets of numbers to find valid combinations. In practice, the number of combinations is significantly limited by the constraints of the problem (only using numbers between `1` and `9`).

This implementation effectively provides a structured approach to generate all combinations of `k` numbers that sum to `n` using numbers from `1` to `9`.

# 254. Factor Combinations

### Problem Description 
Numbers can be regarded as the product of their factors.

For example, `8 = 2 x 2 x 2 = 2 x 4`.

Given an integer `n`, return all possible combinations of its factors. You may return the answer in any order.

Note that the factors should be in the range `[2, n - 1]`.


Example 1:
Input: n = 1
Output: []

Example 2:
Input: n = 12
Output: [[2,6],[3,4],[2,2,3]]

Example 3:
Input: n = 37
Output: []

Example 4:
Input: n = 32
Output: [[2,16],[4,8],[2,2,8],[2,4,4],[2,2,2,4],[2,2,2,2,2]]

Constraints:
`1 <= n <= 108`

### Solution 
 To solve the problem of finding all possible combinations of factors of a given integer \( n \) (where the factors must be in the range [2, \( n - 1 \)]), we can use backtracking. Here's a detailed approach:

### Problem Breakdown

1. **Understanding the Factors**:
   - A factor of \( n \) is a number that divides \( n \) completely (without leaving a remainder).
   - Factors we consider must be within the range of 2 to \( n - 1 \).

2. **Backtracking Approach**:
   - The core idea is to recursively build combinations of factors.
   - We will start from the smallest possible factor and work our way up to the largest possible factor.
   - Each time we find a valid factor, we will divide \( n \) by that factor and proceed to find more factors for the resulting quotient.

3. **Stopping Condition**:
   - We stop our recursion when \( n \) is reduced to 1, as we have found a valid combination of factors.

4. **Eliminate Duplicates**:
   - To avoid duplicates, we can ensure that each factor we choose is less than or equal to the previous factor selected in the recursion.

5. **Return Result**:
   - We will store all unique combinations in a list and return that list.

### Implementation

Here is how we can implement the above logic in Python:



```python
from typing import List

class Solution:
    def getFactors(self, n: int) -> List[List[int]]:
        def backtrack(start: int, current: List[int], remaining: int):
            # If remaining is 1, it means we have found a valid combination
            if remaining == 1 and len(current) > 1:
                results.append(current.copy())
                return
            # Iterate through possible factors
            for i in range(start, remaining + 1):
                # If i is a factor of remaining, we can explore further
                if remaining % i == 0:
                    current.append(i)  # Make a choice
                    backtrack(i, current, remaining // i)  # Recur for the remaining/next level
                    current.pop()  # Remove the last element to backtrack

        results = []
        backtrack(2, [], n)  # Start from factor 2
        return results

```

### Explanation of the Code:

1. **Function Definition**:
   - The `getFactors` method takes an integer \( n \) and initializes the backtracking process.

2. **Backtracking Function**:
   - The `backtrack` function takes three parameters:
     - `start`: The current factor to start checking from,
     - `current`: The current list of factors forming a combination,
     - `remaining`: The number that remains to be factored further.

3. **Base Case**:
   - When `remaining` becomes 1, it indicates that we have a valid combination of factors stored in `current`. We ensure that it contains more than one element (as per problem conditions) before adding it to `results`.

4. **Iterating Over Factors**:
   - We loop through possible factors starting from `start`. For each potential factor \( i \):
     - We check if \( i \) divides `remaining` evenly.
     - If it does, \( i \) is added to `current`, and we call `backtrack` recursively with the new updated `remaining` which is `remaining // i`.

5. **Backtracking**:
   - After exploring further with \( i \), we pop \( i \) from `current` to explore the next possible factor.

### Conclusion

This approach efficiently finds all factor combinations of \( n \) by leveraging recursion and backtracking, ensuring that all combinations are formed and tracked in a clean manner. Feel free to run this code directly on the LeetCode platform.

# 267. Palindrome Permutation II

### Problem Description 
Given a string s, return all the palindromic permutations (without duplicates) of it.

You may return the answer in any order. If `s` has no palindromic permutation, return an empty list.


Example 1:
Input: s = "aabb"
Output: ["abba","baab"]

Example 2:
Input: s = "abc"
Output: []

Constraints:
`1 <= s.length <= 16`
`s` consists of only lowercase English letters.

### Solution 
 To solve the problem of finding all palindromic permutations of a given string, we can follow these steps:

1. **Character Frequency Count**: First, we need to count the frequency of each character in the string. A palindrome can be formed from the characters of the string if at most one character has an odd count.

2. **Check for Palindrome Possibility**: If more than one character has an odd count, it's impossible to form a palindromic permutation, so we return an empty list.

3. **Construct Half of the Palindrome**: To construct the palindromic permutations, we can take half of the count of each character (the other half will mirror it) and prepare for permutations.

4. **Generate Permutations**: We then generate permutations of these half strings. After generating each unique permutation, we can construct the full palindromic string by mirroring the half string around any odd-count character (if there is one).

5. **Return Results**: Finally, we will return all unique palindromic permutations as the result.

Here's how the implementation looks in Python, following the format suitable for LeetCode:



```python
from collections import Counter
from typing import List

class Solution:
    def generatePalindromicPermutations(self, s: str) -> List[str]:
        # Step 1: Count frequencies of each character
        char_count = Counter(s)
        odd_count = 0
        half_part = []
        middle_char = ''
        
        # Step 2: Determine if a palindrome permutation is possible
        for char, count in char_count.items():
            if count % 2 == 1:
                odd_count += 1
                middle_char = char  # Only one character can be in the middle of a palindrome
            half_part.append(char * (count // 2))  # Append half of the characters
        
        # If there are more than one odd character counts, return empty list
        if odd_count > 1:
            return []
        
        # Join half part to form the base for permutations
        half_string = ''.join(half_part)
        
        # Step 3: Generate unique permutations of the half string
        from itertools import permutations
        
        # Generate unique permutations
        unique_half_permutations = {''.join(p) for p in permutations(half_string)}
        
        # Step 4: Construct full palindromes from half permutations
        result = []
        for half in unique_half_permutations:
            if middle_char:  # If there's an odd character, insert it in the middle
                result.append(half + middle_char + half[::-1])
            else:  # If there's no odd character, just mirror the half
                result.append(half + half[::-1])
        
        return result

# Example usage:
# sol = Solution()
# print(sol.generatePalindromicPermutations("aabb"))  # returns ["abba", "baab"]
# print(sol.generatePalindromicPermutations("abc"))   # returns []

```

### Explanation:

- **Data Structures**:
  - We use a `Counter` to count character occurrences.
  - We use a list `half_part` to collect half the counts of each character which will be used to generate permutations.

- **Logic**:
  - We iterate over the character counts to count how many characters have odd occurrences and create the half of the string which will be used for permutation.
  - If the number of characters with odd counts is more than one, we return an empty list.
  
- **Permutation Generation**:
  - We use `itertools.permutations`, which generates all the permutations of a given iterable. However, to avoid duplicates when there are repeating characters, we convert the permutations to a set and join them back into strings.

- **Constructing the Palindrome**:
  - We construct potential palindromes based on half permutations and any middle character that might exist if there's an odd character count.

This approach ensures we get all unique palindromic permutations in an efficient and clear manner.

# 291. Word Pattern II

### Problem Description 
Given a `pattern` and a string `s`, return `true` if `s` matches the `pattern`.

A string `s` matches a `pattern` if there is some bijective mapping of single characters to strings such that if each character in `pattern` is replaced by the string it maps to, then the resulting string is `s`. A bijective mapping means that no two characters map to the same string, and no character maps to two different strings.


Example 1:
Input: pattern = "abab", s = "redblueredblue"
Output: true
Explanation: One possible mapping is as follows:
'a' -> "red"
'b' -> "blue"

Example 2:
Input: pattern = "aaaa", s = "asdasdasdasd"
Output: true
Explanation: One possible mapping is as follows:
'a' -> "asd"

Example 3:
Input: pattern = "abab", s = "asdasdasdasd"
Output: true
Explanation: One possible mapping is as follows:
'a' -> "a"
'b' -> "sdasd"
Note that 'a' and 'b' cannot both map to "asd" since the mapping is a bijection.


Example 4:
Input: pattern = "aabb", s = "xyzabcxzyabc"
Output: false

Constraints:
`1 <= pattern.length, s.length <= 20`
`pattern` and `s` consist of only lower-case English letters.

### Solution 
 To solve the problem of determining if a string `s` matches a given `pattern` under a bijective mapping of single characters to substrings, we will follow a structured approach. The main idea is to analyze the relationships between the characters of the pattern and substrings of `s`.

Here's how the algorithm can be implemented step-by-step:

1. **Mapping Creation**: We will create two dictionaries: one for mapping characters from the `pattern` to substrings in `s`, and another for mapping substrings in `s` back to characters in `pattern`. This makes it easy to enforce the bijective condition.

2. **Backtracking**: We will use a backtracking method that attempts to assign substrings from `s` to characters in `pattern`. At each step, we will check if the current assignment leads to a valid mapping.

3. **Recursive Function**: The recursive function will take the current index in the `pattern` and try to assign different possible substrings from `s` starting from the current position.

4. **Base Case**: If the entire pattern is processed and the entire string `s` has been consumed, we have found a valid mapping.

5. **Conflict Resolution**: If at any point the mapping contradicts our earlier assignments (either two characters map to the same substring or one character maps to two different substrings), we will backtrack.

Here's the implementation of the above steps in Python:



```python
class Solution:
    def wordPatternMatch(self, pattern: str, s: str) -> bool:
        # Two dictionaries for mappings
        char_to_str = {}
        str_to_char = {}
        
        # Helper function for backtracking
        def backtrack(pattern_index: int, s_index: int) -> bool:
            # If both indices reach the end, we successfully matched
            if pattern_index == len(pattern) and s_index == len(s):
                return True
            # If one reaches the end and the other does not, it's a mismatch
            if pattern_index == len(pattern) or s_index == len(s):
                return False
            
            pattern_char = pattern[pattern_index]
            
            # Try to match the current pattern char with various substrings
            for end_index in range(s_index + 1, len(s) + 1):  # +1 because we want non-empty substrings
                current_substring = s[s_index:end_index]
                
                # If the pattern char is already mapped
                if pattern_char in char_to_str:
                    # Check if the previous mapping matches the substring
                    if char_to_str[pattern_char] != current_substring:
                        continue  # Mismatch, try next substring
                else:
                    # If the current substring is already mapped to another char
                    if current_substring in str_to_char:
                        continue  # Conflict, try next substring
                    
                    # Establish the mapping
                    char_to_str[pattern_char] = current_substring
                    str_to_char[current_substring] = pattern_char
                    
                # Recur for the next character in the pattern and next character in s
                if backtrack(pattern_index + 1, end_index):
                    return True
                
                # Backtrack: Remove the last mapping
                if pattern_char in char_to_str:
                    del char_to_str[pattern_char]
                if current_substring in str_to_char:
                    del str_to_char[current_substring]
            
            return False
        
        # Start backtracking from the beginning
        return backtrack(0, 0)

# Example usage: Instantiate the class and call the method
solution = Solution()
print(solution.wordPatternMatch("abab", "redblueredblue"))  # Output: True
print(solution.wordPatternMatch("aaaa", "asdasdasdasd"))    # Output: True
print(solution.wordPatternMatch("abab", "asdasdasdasd"))    # Output: True
print(solution.wordPatternMatch("aabb", "xyzabcxzyabc"))    # Output: False

```

### Explanation of Code:

- The `Solution` class contains the method `wordPatternMatch` which is the entry point for the solution.
- Inside this method, we define a helper function `backtrack` that performs the recursive checks, keeping track of the current indices in `pattern` and `s`.
- For each unique character in the `pattern`, we attempt to match it with substrings of `s` by checking if either mapping already exists and if it matches when it does.
- The process continues recursively until all characters in `pattern` are matched or we determine that no valid mapping exists.
- Finally, the function returns `True` if the mapping is successful and `False` otherwise.

This approach efficiently tracks mappings and ensures the bijective nature of the relationship, making it a robust solution for the problem statement.

# 294. Flip Game II

### Problem Description 
You are playing a Flip Game with your friend.

You are given a string `currentState` that contains only `'+'` and `'-'`. You and your friend take turns to flip two consecutive `"++"` into `"--"`. The game ends when a person can no longer make a move, and therefore the other person will be the winner.

Return `true` if the starting player can guarantee a win, and `false` otherwise.


Example 1:
Input: currentState = "++++"
Output: true
Explanation: The starting player can guarantee a win by flipping the middle "++" to become "+--+".


Example 2:
Input: currentState = "+"
Output: false

Constraints:
`1 <= currentState.length <= 60`
`currentState[i]` is either `'+'` or `'-'`.

Follow up: Derive your algorithm's runtime complexity.

### Solution 
 To solve the Flip Game problem, we need to identify whether the starting player can guarantee a win based on the current state of the game represented by a string comprising '+' and '-'. The rules allow players to flip two consecutive '+' signs ('++') to two consecutive '-' signs ('--'), and the game ends when there are no more pairs of '++' available to flip.

## Key Insights for the Solution

1. **Recursion and Backtracking**: The problem can be approached using recursion to explore all possible states resulting from valid moves, which in this case are the flips of '++'.

2. **Base Case**: If the string does not contain '++', then the player cannot make a move, resulting in their defeat.

3. **Winning Condition**: The current player can guarantee a win if there exists at least one move leading to a state where the opponent cannot win. This necessitates exploring all potential moves and ensuring if there is any move that leads to a losing state for the opponent.

## Approach

We'll implement a recursive function that checks if there's a winning move for the current player. For each turn, we'll flip two '+' signs to '-' and recursively check if the opponent can win from the new state.

## Code Implementation

Here's the Python code that implements the solution as per LeetCode's format:



```python
class Solution:
    def canWin(self, currentState: str) -> bool:
        # A set to memoize the results for already computed states
        memo = {}

        def can_current_player_win(state):
            if state in memo:
                return memo[state]

            for i in range(len(state) - 1):
                if state[i] == '+' and state[i + 1] == '+':
                    # Create the new state by flipping '++' to '--'
                    new_state = state[:i] + '--' + state[i + 2:]

                    # If the opponent cannot win from the new state,
                    # it means the current player can guarantee a win
                    if not can_current_player_win(new_state):
                        memo[state] = True
                        return True
            
            memo[state] = False
            return False
        
        return can_current_player_win(currentState)

```

### Explanation of the Code

1. **Memoization**: We use a dictionary `memo` to store already computed results for further efficiency. This avoids repetitive calculations for the same game state.

2. **Recursive Function**: The function `can_current_player_win(state)` checks if the current player can win from the given state:
   - It iterates through the `state` looking for all occurrences of '++'.
   - Each time '++' is found, we generate a new state by flipping these to '--'.
   - We call the function recursively with the new state to see if the opponent can win from there.
   - If there's any new state that leads to a situation where the opponent cannot win (i.e., calling `can_current_player_win(new_state)` returns `False`), then the current player can guarantee a win, and we store it in the memo and return `True`.
   - If we finish the loop and find no winning move, we conclude that the current player cannot win from this state, thus returning `False`.

### Complexity Analysis

- **Time Complexity**: The worst-case scenario requires visiting each game state multiple times. Due to the memoization, we can reduce redundant calculations, resulting in an efficient search. The state space is bounded by the number of '+' and '-' arrangements; thus, the time complexity is approximately O(3^n) in practice, as every pair of '+' can spawn multiple states.
  
- **Space Complexity**: O(n) for the memoization storage of previously computed states.

This implementation is expected to run efficiently within the constraints provided (1 <= `currentState.length` <= 60).

# 306. Additive Number

### Problem Description 
Additive number is a string whose digits can form additive sequence.

A valid additive sequence should contain at least three numbers. Except for the first two numbers, each subsequent number in the sequence must be the sum of the preceding two.

Given a string containing only digits `'0'-'9'`, write a function to determine if it's an additive number.

Note: Numbers in the additive sequence cannot have leading zeros, so sequence `1, 2, 03` or `1, 02, 3` is invalid.


Example 1:
Input: "112358"
Output: true
Explanation: The digits can form an additive sequence: 1, 1, 2, 3, 5, 8. 
             1 + 1 = 2, 1 + 2 = 3, 2 + 3 = 5, 3 + 5 = 8

Example 2:
Input: "199100199"
Output: true
Explanation: The additive sequence is: 1, 99, 100, 199. 
             1 + 99 = 100, 99 + 100 = 199

Constraints:
`num` consists only of digits `'0'-'9'`.

`1 <= num.length <= 35`
Follow up:
How would you handle overflow for very large input integers?

### Solution 
 To determine if a given string of digits represents an additive number, we need to check if we can form a valid additive sequence from it. An additive sequence is a sequence of numbers where every number after the first two is the sum of the two preceding numbers. Additionally, numbers cannot have leading zeros unless they are the digit '0' itself.

### Approach

1. **Brute-Force Search**: We will take all possible pairs of starting numbers from the string and attempt to build the sequence.
2. **Validating Leading Zeros**: We must ensure that any part of the string we take as a number doesn't have leading zeros unless it is '0'.
3. **Sequence Validation**: After extracting the first two numbers, we continuously compute the next number by adding the last two, and we check if this number matches the subsequent characters in the string.
4. **Loop Control**: We will iterate through possible lengths for the first and second numbers based on the length of the string and try every combination.

### Implementation

Here's the Python implementation following the above approach that fits the LeetCode solution format:



```python
class Solution:
    def isAdditiveNumber(self, num: str) -> bool:
        n = len(num)
        
        # Function to check the sequence
        def check_sequence(first: str, second: str, remainder: str) -> bool:
            while remainder:
                # Calculate the expected next number in the additive sequence
                next_num = str(int(first) + int(second))
                # Check if it matches the start of the remainder
                if not remainder.startswith(next_num):
                    return False
                # Move the sequence ahead
                remainder = remainder[len(next_num):]
                first, second = second, next_num
            
            return True

        # Iterate over possible lengths for the first number
        for i in range(1, n):
            # Skip leading zeros
            if num[0] == '0' and i > 1:
                break
            
            first = num[:i]
            # Iterate over possible lengths for the second number
            for j in range(i + 1, n):
                # Skip leading zeros
                if num[i] == '0' and j > i + 1:
                    break
                
                second = num[i:j]
                # Check the sequence with the remaining part of the string
                remainder = num[j:]
                if check_sequence(first, second, remainder):
                    return True
        
        return False

# Example Usage:
# sol = Solution()
# print(sol.isAdditiveNumber("112358"))  # Output: True
# print(sol.isAdditiveNumber("199100199"))  # Output: True

```

### Explanation of the Code
1. The class `Solution` contains the method `isAdditiveNumber`, which takes string `num` as input.
2. Inside the method, we define a helper function `check_sequence` that verifies if a valid sequence can be formed starting from two numbers.
3. Two nested loops allow us to experiment with different split points for the first and second numbers.
4. When checking parts of the string:
   - We skip parts that would result in leading zeros if their length exceeds 1.
   - For valid pairs, we call `check_sequence` to determine if the rest of the string can be represented as an additive sequence.
5. The function returns `True` if a valid sequence is found; otherwise, it returns `False`.

### Considerations for Large Inputs
The constraints of the problem state that the maximum length of `num` is 35 characters. Python's integer type handles large integers, so we do not need to worry about integer overflow as it can handle very large numbers gracefully due to its internal representation.

# 320. Generalized Abbreviation

### Problem Description 
A word's generalized abbreviation can be constructed by taking any number of non-overlapping substrings and replacing them with their respective lengths. For example, `"abcde"` can be abbreviated into `"a3e"` (`"bcd"` turned into `"3"`), `"1bcd1"` (`"a"` and `"e"` both turned into `"1"`), and `"23"` (`"ab"` turned into `"2"` and `"cde"` turned into `"3"`).

Given a string `word`, return a list of all the possible generalized abbreviations of `word`. Return the answer in any order.


Example 1:
Input: word = "word"
Output: ["4","3d","2r1","2rd","1o2","1o1d","1or1","1ord","w3","w2d","w1r1","w1rd","wo2","wo1d","wor1","word"]

Example 2:
Input: word = "a"
Output: ["1","a"]

Constraints:
`1 <= word.length <= 15`
`word` consists of only lowercase English letters.

### Solution 
 To solve the problem of generating all possible generalized abbreviations for a given word, we can use a backtracking approach. The idea is to explore all the possibilities of abbreviating either the current character or keeping it as is, while also tracking the number of characters abbreviated.

### Explanation

1. **Backtracking Function**: We'll create a recursive function that takes the current index in the word and the current built abbreviation as parameters. At each index, we have two choices:
   - Keep the character and add it to the current abbreviation.
   - Abbreviate the character:
     - If we’ve already abbreviated some characters, increment the count and omit the current character.

2. **Base Case**: When we've processed all characters in the word, we add the current abbreviation to our results.

3. **Handling Counts**: Whenever we choose to abbreviate characters, we need to keep track of the count of abbreviated characters, which we reset once we decide to bring back a character.

### Python Code

We'll implement the above logic in the following Python code:



```python
class Solution:
    def generateAbbreviations(self, word: str) -> List[str]:
        res = []

        def backtrack(index: int, current: str, count: int):
            # If we reach the end of the word
            if index == len(word):
                # If we have abbreviated some characters, append the count
                if count > 0:
                    current += str(count)
                res.append(current)
                return

            # Option 1: Abbreviate the current character
            # We continue increasing the count of abbreviations
            backtrack(index + 1, current, count + 1)

            # Option 2: Keep the current character
            # If we have abbreviated some characters, we need to append the count
            if count > 0:
                current += str(count)
            current += word[index]  # Append the current character
            backtrack(index + 1, current, 0)  # Reset count since we are adding a character

        backtrack(0, "", 0)  # Start the backtracking process
        return res

```

### Usage

To run this code, simply create an instance of the `Solution` class and call the `generateAbbreviations` method with the desired input string.

### Example Usage



```python
solution = Solution()
result = solution.generateAbbreviations("word")
print(result)  # Output could be any order of the valid abbreviations

```

### Key Points

1. **Backtracking**: This method is ideal for problems where we need to explore many combinations or permutations, as it avoids generating invalid states.
2. **Handling Counts**: We keep track of count and efficiently reset it as needed when branching.
3. **Time Complexity**: The solution leverages the backtracking tree's properties, reaching a maximum of 2^n possible strings where n is the length of the input word.

### Output Nature

The outputs can be in any order as the problem does not specify a particular order for the results. The recursive nature ensures all combinations are explored systematically. 

You can run this code directly on LeetCode under the appropriate problem statement by structuring the input and output according to their requirements.

# 351. Android Unlock Patterns

### Problem Description 
Android devices have a special lock screen with a `3 x 3` grid of dots. Users can set an "unlock pattern" by connecting the dots in a specific sequence, forming a series of joined line segments where each segment's endpoints are two consecutive dots in the sequence. A sequence of `k` dots is a valid unlock pattern if both of the following are true:
All the dots in the sequence are distinct.

If the line segment connecting two consecutive dots in the sequence passes through any other dot, the other dot must have previously appeared in the sequence. No jumps through non-selected dots are allowed.

Here are some example valid and invalid unlock patterns:
The 1st pattern `[4,1,3,6]` is invalid because the line connecting dots `1` and `3` pass through dot `2`, but dot `2` did not previously appear in the sequence.

The 2nd pattern `[4,1,9,2]` is invalid because the line connecting dots `1` and `9` pass through dot `5`, but dot `5` did not previously appear in the sequence.

The 3rd pattern `[2,4,1,3,6]` is valid because it follows the conditions. The line connecting dots `1` and `3` meets the condition because dot `2` previously appeared in the sequence.

The 4th pattern `[6,5,4,1,9,2]` is valid because it follows the conditions. The line connecting dots `1` and `9` meets the condition because dot `5` previously appeared in the sequence.

Given two integers `m` and `n`, return the number of unique and valid unlock patterns of the Android grid lock screen that consist of at least `m` keys and at most `n` keys.

Two unlock patterns are considered unique if there is a dot in one sequence that is not in the other, or the order of the dots is different.


Example 1:
Input: m = 1, n = 1
Output: 9

Example 2:
Input: m = 1, n = 2
Output: 65

Constraints:
`1 <= m, n <= 9`

### Solution 
 To solve the problem of counting unique and valid unlock patterns on an Android lock screen with a 3x3 grid of dots, we need a systematic approach to explore various possible patterns while respecting the constraints. 

Here is a detailed explanation of the solution, followed by the Python code formatted for LeetCode.

## Problem Breakdown
1. **Understanding the Grid**: The dots are arranged in a 3x3 grid and represented by numbers as follows:
   
```
   1 2 3
   4 5 6
   7 8 9
   
```
   
2. **Constraints on Patterns**:
   - We require that all dots in the sequence are distinct.
   - If a line segment connecting two dots passes through another dot, that dot must have been visited earlier in the sequence. For example, if you traverse directly from dot 1 to 3, dot 2 must be in the sequence before either 1 or 3 has been included.

3. **Count Valid Patterns**: We need to count the patterns of lengths varying from `m` to `n`, where `m` is the minimum number of keys to use and `n` is the maximum.

## Approach
- **Backtracking**: We can use backtracking to explore all possible patterns starting from each dot.
- **Marking Visited Dots**: Utilize a set to track which dots have been visited in the current pattern.
- **Helper Function**: A recursive function will handle the formation of patterns, enforcing the rules as we expand the current pattern.
  
### Implementation Steps
1. Initialize a 3x3 grid representing the dots.
2. Create a function to check if moving from one dot to another is allowed, considering the rules.
3. Use a backtracking approach to build valid sequences and count them.

### The Code



```python
class Solution:
    def numberOfPatterns(self, m: int, n: int) -> int:
        # Predefine the skipping rules for dots that should be visited
        skip = {}
        skip[(1, 3)] = skip[(3, 1)] = 2
        skip[(1, 7)] = skip[(7, 1)] = 4
        skip[(1, 9)] = skip[(9, 1)] = 5
        skip[(2, 8)] = skip[(8, 2)] = 5
        skip[(3, 7)] = skip[(7, 3)] = 5
        skip[(3, 9)] = skip[(9, 3)] = 6
        skip[(4, 6)] = skip[(6, 4)] = 5
        skip[(7, 9)] = skip[(9, 7)] = 8
        
        def backtrack(used, last, length):
            if length > n:
                return 0
            if length >= m:
                total_patterns = 1
            else:
                total_patterns = 0
            
            for next_dot in range(1, 10):
                if next_dot in used:
                    continue
                if (last, next_dot) in skip:
                    if skip[(last, next_dot)] not in used:
                        continue
                
                used.add(next_dot)
                total_patterns += backtrack(used, next_dot, length + 1)
                used.remove(next_dot)
                
            return total_patterns
        
        total_count = 0
        for start in range(1, 10):
            used = {start}
            total_count += backtrack(used, start, 1)

        return total_count

# Example usage
# sol = Solution()
# print(sol.numberOfPatterns(1, 1))  # Output: 9
# print(sol.numberOfPatterns(1, 2))  # Output: 65

```

### Explanation of the Code
- **Initialization**: The `skip` dictionary holds the conditions under which a jump between dots is allowed based on the rules.
- **Backtracking Function**:
  - The function starts at a specified dot, keeping track of `used` dots and `length` of the current pattern.
  - It explores all valid next dots that haven't been used, and checks if a jump is valid according to the constraints.
- **Counting Results**: For each valid pattern length, it accumulates the valid pattern counts.

This code can be directly used in a LeetCode environment and will return the correct counts of unique unlock patterns for provided values of `m` and `n`.

# 357. Count Numbers with Unique Digits

### Problem Description 
Given an integer `n`, return the count of all numbers with unique digits, `x`, where `0 <= x < 10n`.


Example 1:
Input: n = 2
Output: 91
Explanation: The answer should be the total numbers in the range of 0 ≤ x < 100, excluding 11,22,33,44,55,66,77,88,99

Example 2:
Input: n = 0
Output: 1

Constraints:
`0 <= n <= 8`

### Solution 
 To solve the problem of counting all numbers with unique digits where \(0 \leq x < 10^n\), we can break down the problem step by step. 

### Explanation:

1. **Understanding Unique Digits**:
   - A number has unique digits if no digit appears more than once in that number. For example:
     - `12`, `345`, and `6789` are numbers with unique digits.
     - `11`, `22`, `111`, and `121` are NOT.

2. **Constraints on Digits**:
   - For a number range defined by \( n \):
     - If \( n = 0 \): Only the number `0` is valid. Thus, the output is `1`.
     - If \( n = 1 \): This includes `0` to `9`, so there are `10` valid numbers.
     - If \( n \geq 2 \): We’ll need to calculate valid unique-digit combinations.

3. **Counting Unique Digits**:
   - For each logical digit position, we can select unique digits such that they don’t repeat:
     - For the hundreds place (when \( n \) is at least 1), there are 9 different digits (1-9).
     - For the tens and lower places, the digits can be `0-9`, excluding those already chosen.
   - We can derive the formula using combinatorial logic:
     - For numbers with exactly `k` digits:
       - The first digit has 9 choices (1-9).
       - The second digit has 9 choices (0-9, excluding the first digit).
       - The third digit has 8 choices, and so forth.
   
The total count of numbers with unique digits is the sum of counts of unique digits for each digit length from `1` to `n`.

### Python Code:

Below is the implementation in Python following the LeetCode solution format.



```python
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        if n == 0:
            return 1  # Only '0' is available
        if n == 1:
            return 10  # '0' to '9' are valid

        count = 10  # Start by counting the single digit numbers
        unique_digits = 9  # For the first digit (non-zero), we have 1-9
        available_digits = 9  # Remaining digits (0-9, excluding the first one)

        for i in range(2, n + 1):
            count += unique_digits * available_digits
            unique_digits -= 1  # One less unique digit available for the next position
            available_digits -= 1  # One less available digit

        return count

# Example usage:
sol = Solution()
print(sol.countNumbersWithUniqueDigits(2))  # Output: 91
print(sol.countNumbersWithUniqueDigits(0))  # Output: 1

```

### Explanation of the Code:

- We start with a class `Solution` and define the required method `countNumbersWithUniqueDigits`.
- We handle special cases for `n=0` and `n=1`.
- For values of `n` greater than `1`, we initialize the count of unique digits and iterate through the range to calculate counts of unique numbers digit by digit. 
- In each iteration, we multiply the choices for the current digit to progressively build upon the count.
- Finally, we return the total count. 

This implementation runs efficiently for the given constraints \( 0 \leq n \leq 8 \).

# 401. Binary Watch

### Problem Description 
A binary watch has 4 LEDs on the top which represent the hours (0-11), and the 6 LEDs on the bottom represent the minutes (0-59).

Each LED represents a zero or one, with the least significant bit on the right.

For example, the above binary watch reads "3:25".

Given a non-negative integer n which represents the number of LEDs that are currently on, return all possible times the watch could represent.


Example:
Input: n = 1Return: ["1:00", "2:00", "4:00", "8:00", "0:01", "0:02", "0:04", "0:08", "0:16", "0:32"]
Note:
The order of output does not matter.

The hour must not contain a leading zero, for example "01:00" is not valid, it should be "1:00".

The minute must be consist of two digits and may contain a leading zero, for example "10:2" is not valid, it should be "10:02".

### Solution 
 To solve the problem of finding possible times that a binary watch can represent with a given number of LEDs lit, we need to consider how the LEDs represent the hours and minutes.

### Breakdown of the Problem

1. **Binary Representation**:
   - The watch has 4 LEDs for hours and 6 LEDs for minutes.
   - Hours can range from **0 to 11** (4 bits: `0000` to `1011`).
   - Minutes can range from **0 to 59** (6 bits: `000000` to `111011`).

2. **LED Counting**:
   - The number of lit LEDs is represented by `n`, which is the sum of the hours' and minutes' bits that are set to 1.

3. **Goal**:
   - Generate all combinations of hours and minutes where the total count of LEDs that are lit equals `n`.

### Steps to Implement

1. **Loop through all possible hours and minutes** using nested loops: 
   - For hours from `0` to `11`.
   - For minutes from `0` to `59`.

2. **Count the number of bits set (lit LEDs)**:
   - Use the `bin(x).count('1')` function to count the bits set in the binary representation of hours and minutes.

3. **Check if the total LED count equals `n`**:
   - If the sum of the LED counts for hours and minutes equals `n`, format the time string as "H:MM".

4. **Store results** and return from the function.

### Python Code

Here's the implementation based on the above logic:



```python
class Solution:
    def readBinaryWatch(self, n: int) -> List[str]:
        results = []
        
        # Iterate over hours (0-11)
        for hour in range(12):
            # Iterate over minutes (0-59)
            for minute in range(60):
                # Count the number of LEDs that are on
                # We use bin to convert the numbers to binary and count '1's.
                if bin(hour).count('1') + bin(minute).count('1') == n:
                    # Format the time as "H:MM"
                    results.append(f"{hour}:{minute:02d}")

        return results

```

### Explanation of the Code

1. **Class Definition**: We define a class `Solution` to encapsulate the method required by LeetCode.
   
2. **Function Declaration**: The `readBinaryWatch` method takes an integer `n` as input and returns a list of time strings.

3. **Nested Loops**: We loop through every hour (0-11) and every minute (0-59). 

4. **Bit Counting**: For each combination of hour and minute, we count the bits set to '1' using `bin(x).count('1')`:
   - `bin(hour).count('1')` counts the lit LEDs in the hour.
   - `bin(minute).count('1')` counts the lit LEDs in the minute.

5. **Check Condition**: If the total count from both hour and minute matches `n`, we format the results appropriately by ensuring minutes always have two digits using `minute:02d`, which pads with zero if necessary.

6. **Return Statement**: Finally, we return the accumulated `results`.

### Complexity Analysis
- **Time Complexity**: O(1), since the loop runs through a fixed number of hours (12) and a fixed number of minutes (60).
- **Space Complexity**: O(1), as we are using a list to store the results but the maximum possible time strings stored is also limited.

This solution is efficient and direct, making use of Python's built-in functions to simplify the binary counting process.

# 411. Minimum Unique Word Abbreviation

### Problem Description 
A string can be abbreviated by replacing any number of non-adjacent substrings with their lengths. For example, a string such as `"substitution"` could be abbreviated as (but not limited to):
`"s10n"` (`"s ubstitutio n"`)
`"sub4u4"` (`"sub stit u tion"`)
`"12"` (`"substitution"`)
`"su3i1u2on"` (`"su bst i t u ti on"`)
`"substitution"` (no substrings replaced)
Note that `"s55n"` (`"s ubsti tutio n"`) is not a valid abbreviation of `"substitution"` because the replaced substrings are adjacent.

The length of an abbreviation is the number of letters that were not replaced plus the number of substrings that were replaced. For example, the abbreviation `"s10n"` has a length of `3` (`2` letters + `1` substring) and `"su3i1u2on"` has a length of `9` (`6` letters + `3` substrings).

Given a target string `target` and an array of strings `dictionary`, return an abbreviation of `target` with the shortest possible length such that it is not an abbreviation of any string in `dictionary`. If there are multiple shortest abbreviations, return any of them.


Example 1:
Input: target = "apple", dictionary = ["blade"]
Output: "a4"
Explanation: The shortest abbreviation of "apple" is "5", but this is also an abbreviation of "blade".

The next shortest abbreviations are "a4" and "4e". "4e" is an abbreviation of blade while "a4" is not.

Hence, return "a4".


Example 2:
Input: target = "apple", dictionary = ["blade","plain","amber"]
Output: "1p3"
Explanation: "5" is an abbreviation of both "apple" but also every word in the dictionary.

"a4" is an abbreviation of "apple" but also "amber".

"4e" is an abbreviation of "apple" but also "blade".

"1p3", "2p2", and "3l1" are the next shortest abbreviations of "apple".

Since none of them are abbreviations of words in the dictionary, returning any of them is correct.


Constraints:
`target.length == m`
`dictionary.length == n`
`1 <= m <= 21`
`0 <= n <= 1000`
`1 <= dictionary[i] <= 100`
`log2(n) + m <= 21` if `n > 0`

### Solution 
 To solve the problem of finding the shortest valid abbreviation of a target string that does not conflict with any abbreviation for words in a given dictionary, we can use a backtracking approach combined with bit manipulation. The strategy is to generate all possible abbreviations for the target string, check their validity against each word in the dictionary, and select the shortest valid abbreviation.

### Explanation of the Approach

1. **Bitmask Representation**: We can represent each abbreviation as a bitmask, where the bits indicate whether a character at that position in the target string is kept or replaced. For example, if we have a target string of length `m`, the bitmask would be `0` (keep) or `1` (abbreviate) for each character.

2. **Generate Abbreviations**: For each combination of character replacements (defined by the bitmask), we construct the abbreviation and calculate its length.

3. **Check Validity**: For each generated abbreviation, check if it can be formed from any word in the dictionary by extending the abbreviation back to the original word.

4. **Sorting and Selecting**: Store valid abbreviations and their lengths, then sort to find the shortest one.

5. **Implementation Details**: Since the maximum length of the target string is limited to 21, the total combinations of abbreviations are manageable (2^21 in the worst case).

### Challenges to Address

- Efficiently checking if an abbreviation is valid against the dictionary.
- Ensuring the abbreviation length is minimal while still being distinct from the dictionary.

### Implementation

Here is how you can implement the solution using Python. This code follows the LeetCode submission format, which allows it to be executed directly on the platform.



```python
class Solution:
    def abbreviate(self, target: str, dictionary: List[str]) -> str:
        from itertools import combinations

        # Function to get the abbreviation of string with given mask
        def get_abbreviation(mask):
            abbr = []
            count = 0
            for i in range(len(target)):
                if mask[i] == '1':
                    count += 1
                else:
                    if count > 0:
                        abbr.append(str(count))
                        count = 0
                    abbr.append(target[i])
            if count > 0:
                abbr.append(str(count))
            return ''.join(abbr)

        # Function to check if abbreviation is valid against the dictionary
        def is_valid_abbr(abbr):
            for word in dictionary:
                if self.is_abbreviation_valid(word, abbr):
                    return False
            return True

        # Function to check if the abbr is a valid abbreviation for the word
        def is_abbreviation_valid(word, abbr):
            i, j = 0, 0
            while i < len(word) and j < len(abbr):
                if abbr[j].isdigit():
                    num = 0
                    while j < len(abbr) and abbr[j].isdigit():
                        num = num * 10 + int(abbr[j])
                        j += 1
                    i += num  # Skip the number of characters in word
                else:
                    if word[i] != abbr[j]:
                        return False
                    i += 1
                    j += 1
            return j == len(abbr)

        m = len(target)
        for length in range(1, m + 1):  # Start from length 1 to m
            for mask in combinations(range(m), length):
                bitmask = ['0'] * m
                for idx in mask:
                    bitmask[idx] = '1'
                abbr = get_abbreviation(bitmask)
                if is_valid_abbr(abbr):
                    return abbr
        
        return target # Return target itself if no abbreviation is valid

# Example usage
solution = Solution()
print(solution.abbreviate("apple", ["blade"]))  # Example output: "a4"
print(solution.abbreviate("apple", ["blade", "plain", "amber"]))  # Example output: "1p3"

```

### Summary

- **Generating Abbreviations**: We systematically combine keep/replace decisions for each character in the target string.
- **Validity Check**: By defining a robust function to check against the dictionary, we keep our approach efficient.
- **Output**: The shortest valid abbreviation is returned as required. 

This implementation balances between clarity and efficiency, adhering to the constraints provided in the prompt.

# 425. Word Squares

### Problem Description 
Given a set of words (without duplicates), find all word squares you can build from them.

A sequence of words forms a valid word square if the kth row and column read the exact same string, where 0 ≤ k < max(numRows, numColumns).

For example, the word sequence `["ball","area","lead","lady"]` forms a word square because each word reads the same both horizontally and vertically.

b a l l
a r e a
l e a d
l a d y
Note:
There are at least 1 and at most 1000 words.

All words will have the exact same length.

Word length is at least 1 and at most 5.

Each word contains only lowercase English alphabet `a-z`.


Example 1:
Input:
["area","lead","wall","lady","ball"]
Output:
[
  [ "wall",
    "area",
    "lead",
    "lady"
  ],
  [ "ball",
    "area",
    "lead",
    "lady"
  ]
]
Explanation:
The output consists of two word squares. The order of output does not matter (just the order of words in each word square matters).


Example 2:
Input:
["abat","baba","atan","atal"]
Output:
[
  [ "baba",
    "abat",
    "baba",
    "atan"
  ],
  [ "baba",
    "abat",
    "baba",
    "atal"
  ]
]
Explanation:
The output consists of two word squares. The order of output does not matter (just the order of words in each word square matters).

### Solution 
 To solve the problem of finding all possible word squares from a given set of words, we can utilize a backtracking approach combined with a prefix tree (Trie) to efficiently explore potential candidates for word squares.

A word square consists of words arranged in a square such that the word in the ith row matches the word in the ith column. Therefore, while adding words to the square, we need to ensure that the prefix formed by the columns matches the input words.

### Steps to Solve the Problem:
1. **Preprocessing:** We'll first reorganize the words into a prefix map. This will help in quickly finding potential candidates for the square based on the prefixes formed by the rows and columns.
2. **Backtracking:** We'll construct the word square using a depth-first search (DFS) approach, which will add words to the square one by one while checking for valid prefixes.
3. **Adding Results:** Once a valid word square is formed (i.e., when the square is full), we add this square to our results list.

### Implementation

Here is the Python code using the above approach:



```python
class Solution:
    def wordSquares(self, words):
        # Initialize a list for the result and a prefix map
        self.results = []
        self.prefix_map = {}
        self.word_length = len(words[0])
        
        # Build the prefix map
        for word in words:
            for i in range(1, self.word_length + 1):
                prefix = word[:i]
                if prefix not in self.prefix_map:
                    self.prefix_map[prefix] = []
                self.prefix_map[prefix].append(word)

        # Function to attempt to build the word square
        def backtrack(square):
            if len(square) == self.word_length:
                self.results.append(square[:])
                return
            
            # Determine the prefix for the next word
            prefix_length = len(square)
            current_prefix = ''.join([square[i][prefix_length] for i in range(prefix_length)])
            
            # Find words that match the current prefix
            if current_prefix in self.prefix_map:
                for next_word in self.prefix_map[current_prefix]:
                    square.append(next_word)
                    backtrack(square)
                    square.pop()

        # Start backtracking with each word as the first word in the square
        for word in words:
            backtrack([word])
        
        return self.results

# Example usage:
# sol = Solution()
# print(sol.wordSquares(["area","lead","wall","lady","ball"]))

```

### Explanation of the Code:
1. **Class Definition:** We define a `Solution` class as required by LeetCode.
2. **Prefix Map Creation:** We loop through each word and build a dictionary mapping prefixes of each word to the words that have those prefixes.
3. **Backtracking Function:** The `backtrack` function attempts to build the square:
    - When the length of the current square equals the word length, we copy the square into results.
    - We construct the current column prefix using the existing words in the square.
    - We look up possible next words using the prefix map and continue building the square recursively.
4. **Initial Call:** We call the `backtrack` starting with each word from the list, allowing all words to be a potential starting point for the square.

### Complexity Analysis:
- **Time Complexity:** O(N^4) in the worst case, where N is the number of words. Each word potentially requires O(N^2) for checking the prefix for each square row and column.
- **Space Complexity:** O(N^2) for maintaining the prefixes and the results list.

This code is structured to be run directly on the LeetCode platform and should function correctly for the given problem constraints.

# 526. Beautiful Arrangement

### Problem Description 
Suppose you have `n` integers labeled `1` through `n`. A permutation of those `n` integers `perm` (1-indexed) is considered a beautiful arrangement if for every `i` (`1 <= i <= n`), either of the following is true:
`perm[i]` is divisible by `i`.

`i` is divisible by `perm[i]`.

Given an integer `n`, return the number of the beautiful arrangements that you can construct.


Example 1:
Input: n = 2
Output: 2
Explanation: 
The first beautiful arrangement is [1,2]:
    - perm[1] = 1 is divisible by i = 1
    - perm[2] = 2 is divisible by i = 2
The second beautiful arrangement is [2,1]:
    - perm[1] = 2 is divisible by i = 1
    - i = 2 is divisible by perm[2] = 1

Example 2:
Input: n = 1
Output: 1

Constraints:
`1 <= n <= 15`

### Solution 
 To solve the problem of counting the number of beautiful arrangements for the given integer `n`, we'll use a backtracking approach. A backtracking solution allows us to explore all permutations of the numbers and count those that meet the defined criteria of being beautiful arrangements.

### Explanation:

1. **Understanding the Criteria**:
    - A permutation `perm` is called beautiful if for every index `i` (1-indexed), one of the following holds:
        - `perm[i]` is divisible by `i`, or
        - `i` is divisible by `perm[i]`.

2. **Backtracking Approach**:
    - We'll generate permutations of the numbers `[1, 2, ..., n]` recursively.
    - For each number we place in the permutation, we'll check if it satisfies the beautiful arrangement condition.
    - If it does, we proceed to the next position; if not, we backtrack.

3. **Implementation Steps**:
    - Keep track of the current permutation.
    - Use a boolean array (or a set) to track which numbers have already been used in the permutation.
    - Recursively fill the permutation, checking the conditions for each index.
    - Count each valid configuration that meets the criteria.

### Python Code Implementation:

Here’s how the above logic can be implemented in Python, following the LeetCode solution format:



```python
class Solution:
    def countArrangement(self, n: int) -> int:
        # Keep track of the count of beautiful arrangements
        self.count = 0
        
        def backtrack(pos, used):
            if pos > n:
                self.count += 1
                return
            
            for i in range(1, n + 1):
                if not used[i]:  # If i is not used yet
                    # Check the beautiful arrangement conditions
                    if i % pos == 0 or pos % i == 0:
                        used[i] = True  # Mark this number as used
                        backtrack(pos + 1, used)  # Recurse to the next position
                        used[i] = False  # Backtrack
                        
        # Create an array to track used numbers from 1 to n
        used = [False] * (n + 1)
        backtrack(1, used)  # Start the process with position 1
        
        return self.count

# Example Test Cases
sol = Solution()
print(sol.countArrangement(2))  # Output: 2
print(sol.countArrangement(1))  # Output: 1

```

### Explanation of Code:
1. **Class Definition**: We define a class `Solution` containing the method `countArrangement`.
2. **Count Variable**: `self.count` is a class variable used to store the number of beautiful arrangements found.
3. **Backtrack Function**: A recursive function that:
   - Takes the current position (`pos`) and a list of used numbers (`used`).
   - If `pos` exceeds `n`, we have found a valid arrangement, so we increment the count.
   - For each number `i` in the range `[1, n]`, if it hasn't been used yet and satisfies the beautiful arrangement's divisibility condition, we proceed to recursively fill the next position.
4. **Flagging Used Numbers**: The `used` list helps ensure that each number can be placed in the permutation only once.

This algorithm efficiently counts the beautiful arrangements through a depth-first search method, and given that `n` is at most `15`, this approach will run efficiently within the provided constraints.

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
 To solve the problem of finding the minimum number of stickers required to form a target string using available stickers, we can approach it using backtracking combined with memoization for efficiency. The key idea is to use a recursive function to explore the combinations of stickers used to construct the `target`, while keeping track of how many stickers we have used so far. 

Here is a detailed explanation followed by the code implementation in a format suitable for LeetCode:

### Step-by-Step Explanation:

1. **Character Count Preparation**: 
   - We need to count the frequency of each character in the `target` string since we will need to match these counts using the stickers.

2. **Dynamic Programming with Bitmasking**:
   - Since the target can have at most 15 characters, we can use a bitmask to represent the current state of how much of the target we have fulfilled. A simple int can represent our letters needed (where each bit represents whether a character is still needed).
   - We also use a dictionary for memoization to store the results of already computed states to avoid repeated calculations.

3. **Recursive Function**:
   - This function takes the current mask (which indicates the state of needed characters) and recursively tries to use each sticker to fulfill the characters remaining.
   - For each sticker, we determine how much it can address the remaining characters indicated by the current mask. We then compute the new mask after using the sticker and make a recursive call.

4. **Base Case**:
   - If the current mask equals 0, it implies that all characters in the target have been fulfilled, and thus no additional stickers are needed.
   - We return the number of stickers used in this case.

5. **Iterate through Stickers**:
   - For each sticker, we compute the contribution of each sticker to the characters needed and recursively explore further.

6. **Return Result**:
   - Finally, we return the minimum number of stickers needed to form the target or -1 if it's not possible to form the target.

### Python Code Implementation



```python
from collections import Counter
import functools

class Solution:
    def minStickers(self, stickers: List[str], target: str) -> int:
        # Count frequency of each letter in the target
        target_count = Counter(target)
        
        # Prepare a list of sticker frequencies
        sticker_counts = []
        for sticker in stickers:
            sticker_counts.append(Counter(sticker))
        
        # Memoization
        @functools.lru_cache(None)
        def dfs(remaining_count):
            # When no letters are needed
            if not remaining_count:
                return 0
            
            # Mimic the target count with a key
            remaining_count_s = str(remaining_count)
            min_stickers = float('inf')
            
            # Try each sticker
            for sticker_count in sticker_counts:
                # If the sticker can't help with the current letters
                if sticker_count[remaining_count[0]] == 0:
                    continue
                
                # Build a new remaining need count
                new_remaining_count = remaining_count.copy()
                
                for ch in sticker_count:
                    if ch in new_remaining_count:
                        new_remaining_count[ch] = max(0, new_remaining_count[ch] - sticker_count[ch])
                
                # Only search further if there are still letters remaining
                if sum(new_remaining_count.values()) != 0:
                    num_stickers = dfs(ftuple(sorted(new_remaining_count.items())))
                    if num_stickers != float('inf'):
                        min_stickers = min(min_stickers, num_stickers + 1)

            return min_stickers

        # Start DFS with the full target count
        result = dfs(tuple(sorted(target_count.items())))
        
        return result if result != float('inf') else -1

```

### Explanation of the Code:
- We use `Counter` to efficiently count character frequencies in both stickers and the target.
- The `dfs` function uses memoization to cache results based on the current need state (as a sorted tuple of character counts).
- We check if using each sticker can help reduce the needs and recursively compute further needs.
- The final answer is either the computed minimum number of stickers or -1 if it's impossible.

### Complexity:
- This approach is optimized by memoization and will work efficiently within the given constraints, ensuring that we do not recompute results unnecessarily.

You can now run this code directly on LeetCode by copying it into the provided solution editor for the problem.

# 784. Letter Case Permutation

### Problem Description 
Given a string S, we can transform every letter individually to be lowercase or uppercase to create another string.

Return a list of all possible strings we could create. You can return the output in any order.


Example 1:
Input: S = "a1b2"
Output: ["a1b2","a1B2","A1b2","A1B2"]

Example 2:
Input: S = "3z4"
Output: ["3z4","3Z4"]

Example 3:
Input: S = "12345"
Output: ["12345"]

Example 4:
Input: S = "0"
Output: ["0"]

Constraints:
`S` will be a string with length between `1` and `12`.

`S` will consist only of letters or digits.

### Solution 
 To solve the problem of transforming a string such that each letter can be either lowercase or uppercase (while digits remain unchanged), we can use a backtracking approach. This technique allows us to explore all possible combinations of letters in the string while keeping track of the current state of our transformation.

### Steps to Approach the Problem:

1. **Backtracking Method**: We will define a recursive function that builds the result string character by character.
2. **Character Processing**: For each character:
   - If it's a letter (a-z or A-Z), we can either:
     - Add it to our result in lowercase.
     - Add it to our result in uppercase.
   - If it's a digit, we only add it in its original form as digits do not change.
3. **Base Case**: When we have processed all characters, we add the constructed string to our list of results.
4. **Returning Results**: Finally, we return the list containing all variations of the transformed string.

Let's translate this approach into code.



```python
class Solution:
    def letterCasePermutation(self, S: str) -> List[str]:
        result = []
        
        def backtrack(index: int, path: str):
            if index == len(S):
                result.append(path)
                return
            
            # Get the current character
            current_char = S[index]
            
            # If it is a letter, we can use both lowercase and uppercase
            if current_char.isalpha():
                # Consider the lowercase option
                backtrack(index + 1, path + current_char.lower())
                # Consider the uppercase option
                backtrack(index + 1, path + current_char.upper())
            else:
                # If it's a digit, we only add it as is
                backtrack(index + 1, path + current_char)
        
        # Start backtracking from the first index
        backtrack(0, "")
        
        return result

```

### Explanation of the Code:

1. **Class Definition**: The solution is wrapped in a class named `Solution`, which is a common format for coding solutions on LeetCode.
2. **Function Definition**: The method `letterCasePermutation` receives a string `S` and initializes an empty list `result` to store all permutations.
3. **Backtrack Function**:
   - **Parameters**: `index` keeps track of the current character being processed, and `path` is the string constructed so far.
   - **Base Case**: When `index` equals the length of `S`, the complete string is formed and appended to `result`.
   - **Character Handling**:
     - For letters, we call `backtrack` twice—once for lowercase and once for uppercase.
     - For digits, we call `backtrack` once, appending the digit to the path.
4. **Starting the Process**: Backtracking starts from index `0` with an empty `path`.
  
### Time Complexity:
The time complexity of this solution is O(2^L), where L is the number of letters in the string. This is because each letter has two possible states (lowercase/uppercase), leading to exponential combinations.

### Space Complexity:
The space complexity is also O(2^L), as we need to store all permutations in the `result` list.

With this clear backtracking strategy, we can generate all possible letter-case permutations for the given string efficiently.

# 797. All Paths From Source to Target

### Problem Description 
Given a directed acyclic graph (DAG) of `n` nodes labeled from 0 to n - 1, find all possible paths from node `0` to node `n - 1`, and return them in any order.

The graph is given as follows: `graph[i]` is a list of all nodes you can visit from node `i` (i.e., there is a directed edge from node `i` to node `graph[i][j]`).


Example 1:
Input: graph = [[1,2],[3],[3],[]]
Output: [[0,1,3],[0,2,3]]
Explanation: There are two paths: 0 -> 1 -> 3 and 0 -> 2 -> 3.


Example 2:
Input: graph = [[4,3,1],[3,2,4],[3],[4],[]]
Output: [[0,4],[0,3,4],[0,1,3,4],[0,1,2,3,4],[0,1,4]]

Example 3:
Input: graph = [[1],[]]
Output: [[0,1]]

Example 4:
Input: graph = [[1,2,3],[2],[3],[]]
Output: [[0,1,2,3],[0,2,3],[0,3]]

Example 5:
Input: graph = [[1,3],[2],[3],[]]
Output: [[0,1,2,3],[0,3]]

Constraints:
`n == graph.length`
`2 <= n <= 15`
`0 <= graph[i][j] < n`
`graph[i][j] != i` (i.e., there will be no self-loops).

The input graph is guaranteed to be a DAG.

### Solution 
 To solve the problem of finding all paths from node `0` to node `n - 1` in a directed acyclic graph (DAG), we can employ Depth-First Search (DFS). We'll explore all possible paths from the starting node to the target node, keeping track of the current path along the way.

### Algorithm Explanation:

1. **Initialization:**
   - Start from node `0`.
   - Prepare a list to store all valid paths.
   - Use a helper function to perform DFS.

2. **DFS Function:**
   - The function takes the current node and the current path as its arguments.
   - If the current node is the target node (`n - 1`), add the current path to the path list and return.
   - Otherwise, iterate through all the neighbors of the current node.
   - For each neighbor, append it to the current path and recursively call DFS on the neighbor.
   - After returning from recursion, backtrack by removing the neighbor from the current path.

3. **Base Case:**
   - When reaching the target node, we add the current path to the result list.

4. **End Result:**
   - After all recursive calls are completed, we will have a list of all paths from node `0` to node `n - 1`.

### Python Code:
Below is the implementation of the above logic in the required format for LeetCode.



```python
from typing import List

class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        n = len(graph)  # Number of nodes
        result = []     # To store all paths
        
        def dfs(node: int, path: List[int]):
            path.append(node)  # Add current node to the path
            if node == n - 1:  # If we reached the target node
                result.append(path.copy())  # Store the current path
            else:
                for neighbor in graph[node]:  # Explore neighbors
                    dfs(neighbor, path)  # Recursive DFS call
            path.pop()  # Backtrack after exploring

        dfs(0, [])  # Start DFS from node 0
        return result  # Return the result

```

### Detailed Explanation of the Code:

- **Imports:** We import `List` from `typing` for type hinting to indicate our function's parameters and return types.
- **Class Declaration:** We define a class `Solution` with a method `allPathsSourceTarget`.
- **Graph Traversal:**
  - The method initializes `result`, a list to hold all found paths.
  - A helper function `dfs` is defined, which is responsible for the depth-first traversal.
- **Path Tracking:**
  - In the `dfs` function, we append the current node to the "path" list.
  - We check if the current node is the target. If it is, we append a copy of the current path to results.
  - If not, we loop through each neighbor of the current node and call `dfs` recursively.
- **Backtracking:**
  - After exploring all paths that can be formed from the current node, we remove the node from `path` (backtracking) to explore alternative paths.
- **Execution Start:** We call the `dfs` function with node `0` and an empty list.
- **Return Statement:** Finally, we return the `result` containing all paths found from source to target.

This code will successfully find all paths from node `0` to `n - 1` in samples of the specified graph with a time complexity of \(O(2^V)\), where \(V\) is the number of vertices, since the function explores all possible paths through the graph. The constraints ensure this approach will run efficiently.

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
 To solve the problem of finding a Fibonacci-like sequence in a string of digits, we'll utilize a systematic approach to split the string into possible candidate numbers and check if they fulfill the Fibonacci condition. Here's a detailed explanation of the solution, formatted as you would find in a LeetCode problem.

## Solution Outline:
1. **Understanding Fibonacci-like Sequence**: A Fibonacci-like sequence is one where each number is the sum of the two preceding numbers, starting with at least three numbers. The sequence must also not contain numbers with leading zeros (except for the number "0" itself).

2. **Generating Candidate Numbers**: We can split the input string `S` into potential numbers. We'll try different lengths for the first and the second numbers in the sequence. The third number will be inferred from the sum of the first two.

3. **Validation**: For each generated sequence:
   - Ensure there are no leading zeros in any number (except for the number "0").
   - Check if the inferred third number matches the next segment of the string.

4. **Implementation**: We'll iterate through the string, trying to split it in different ways, and check each resulting Fibonacci-like sequence.

### Python Code:


```python
class Solution:
    def splitIntoFibonacci(self, S: str):
        def is_valid(num_str):
            # A number is valid if it doesn't have leading zeros unless it is "0"
            return num_str == "0" or (not num_str.startswith("0") and int(num_str) <= 2**31 - 1)

        n = len(S)
        for i in range(1, n):
            for j in range(i + 1, n):
                # Get the first two numbers from the string
                first = S[:i]
                second = S[i:j]
                
                # Check if they are valid numbers
                if not is_valid(first) or not is_valid(second):
                    continue
                
                # Start building the Fibonacci-like sequence
                fib_sequence = [int(first), int(second)]
                k = j  # Current position in the string
                
                while k < n:
                    # Get the next number in the Fibonacci-like sequence
                    next_fib = fib_sequence[-1] + fib_sequence[-2]
                    next_fib_str = str(next_fib)
                    next_fib_length = len(next_fib_str)
                    
                    # Check if the next number matches the segment in the string
                    if S[k:k + next_fib_length] == next_fib_str:
                        fib_sequence.append(next_fib)
                        k += next_fib_length  # Move the index forward
                    else:
                        break  # Exit the loop if the sequence breaks

                # Check if we successfully used the entire string and have at least 3 numbers
                if k == n and len(fib_sequence) >= 3:
                    return fib_sequence
        
        return []  # If no valid sequence is found

```

### Explanation of the Code:
1. **is_valid function**: This helper function checks if a given string can be converted to a valid integer under the problem's constraints.

2. **Main Logic**: We iterate through possible ends for the first two numbers (`i` and `j`):
   - Convert these substrings into integers and validate them.
   - Construct the potential Fibonacci sequence by repeatedly summing the last two numbers and checking if this sum can be found in the remainder of the string.
   
3. **Condition Check**: After checking all possible pairs, we return the first valid Fibonacci-like sequence that we find. If no such sequence can be formed, we return an empty list.

### Complexity:
- **Time Complexity**: O(n^3) in the worst case, where `n` is the length of the string, due to the three nested loops for splitting and checking validity.
- **Space Complexity**: O(n) for storing the potential Fibonacci-like sequence.

This implementation directly adheres to the requirements of the LeetCode platform and should work efficiently given the constraints.

# 967. Numbers With Same Consecutive Differences

### Problem Description 
Return all non-negative integers of length `n` such that the absolute difference between every two consecutive digits is `k`.

Note that every number in the answer must not have leading zeros. For example, `01` has one leading zero and is invalid.

You may return the answer in any order.


Example 1:
Input: n = 3, k = 7
Output: [181,292,707,818,929]
Explanation: Note that 070 is not a valid number, because it has leading zeroes.


Example 2:
Input: n = 2, k = 1
Output: [10,12,21,23,32,34,43,45,54,56,65,67,76,78,87,89,98]

Example 3:
Input: n = 2, k = 0
Output: [11,22,33,44,55,66,77,88,99]

Example 4:
Input: n = 2, k = 2
Output: [13,20,24,31,35,42,46,53,57,64,68,75,79,86,97]

Constraints:
`2 <= n <= 9`
`0 <= k <= 9`

### Solution 
 Here's a Python code implementation for the stated problem, formatted to be directly executable in LeetCode. The solution uses a depth-first search (DFS) approach to generate the numbers based on the constraints given.



```python
class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        if n == 1:
            return [i for i in range(10)]  # single-digit numbers (0-9)

        def dfs(num, length):
            if length == n:
                result.append(num)
                return
            last_digit = num % 10
            # Generate the next possible digits based on k
            if last_digit + k < 10:  # Ensure the next digit is valid
                dfs(num * 10 + (last_digit + k), length + 1)
            if k > 0 and last_digit - k >= 0:  # Ensure the next digit is valid and avoid duplicates when k=0
                dfs(num * 10 + (last_digit - k), length + 1)
        
        result = []
        # Start from numbers 1 to 9 to avoid leading zeros
        for starting_digit in range(1, 10):
            dfs(starting_digit, 1)

        return result

```

### Explanation

1. **Base Case for n=1**: The first check is for the case when `n=1`. The valid numbers are single-digit integers from `0` to `9`. Hence, if `n=1`, we return `[0, 1, 2, ..., 9]`.

2. **Depth-First Search Function (dfs)**:
   - We define a recursive function `dfs(num, length)` that builds numbers digit by digit.
   - `num` is the current number being built, and `length` is the current length of that number.
   - When the length equals `n`, we append the number to our results and return.
   - We determine the last digit of the current number with `last_digit = num % 10`.
   - We look at two potential candidates for the next digit based on the last digit and `k`:
     - **Option 1**: Add `k` to the last digit if it remains a valid digit (i.e., less than 10).
     - **Option 2**: Subtract `k` from the last digit if it does not go negative and ensure that we don't do this if `k=0` to avoid duplicating the same digit.

3. **Initializing the Search**: 
   - The main function initializes an empty list called `result` to store valid numbers.
   - We iterate from `1` to `9` to serve as the starting digit (to avoid leading zeros).
   - We trigger the DFS for each valid starting digit.

4. **Return Statement**: After the DFS completes for all starting digits, we return the `result` list containing all valid numbers.

The algorithm is efficient for the input constraints (`2 <= n <= 9` and `0 <= k <= 9`), since it recursively builds valid numbers without leading zeros and adheres to the constraints on consecutive digit differences.

# 980. Unique Paths III

### Problem Description 
On a 2-dimensional `grid`, there are 4 types of squares:
`1` represents the starting square.  There is exactly one starting square.

`2` represents the ending square.  There is exactly one ending square.

`0` represents empty squares we can walk over.

`-1` represents obstacles that we cannot walk over.

Return the number of 4-directional walks from the starting square to the ending square, that walk over every non-obstacle square exactly once.


Example 1:
Input: [[1,0,0,0],[0,0,0,0],[0,0,2,-1]]
Output: 2
Explanation: We have the following two paths: 
1. (0,0),(0,1),(0,2),(0,3),(1,3),(1,2),(1,1),(1,0),(2,0),(2,1),(2,2)
2. (0,0),(1,0),(2,0),(2,1),(1,1),(0,1),(0,2),(0,3),(1,3),(1,2),(2,2)

Example 2:
Input: [[1,0,0,0],[0,0,0,0],[0,0,0,2]]
Output: 4
Explanation: We have the following four paths: 
1. (0,0),(0,1),(0,2),(0,3),(1,3),(1,2),(1,1),(1,0),(2,0),(2,1),(2,2),(2,3)
2. (0,0),(0,1),(1,1),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,3),(1,3),(2,3)
3. (0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(1,1),(0,1),(0,2),(0,3),(1,3),(2,3)
4. (0,0),(1,0),(2,0),(2,1),(1,1),(0,1),(0,2),(0,3),(1,3),(1,2),(2,2),(2,3)

Example 3:
Input: [[0,1],[2,0]]
Output: 0
Explanation: 
There is no path that walks over every empty square exactly once.

Note that the starting and ending square can be anywhere in the grid.

Note:
`1 <= grid.length * grid[0].length <= 20`

### Solution 
 To solve the problem of counting the number of unique paths from a starting square to an ending square on a grid while visiting every non-obstacle square exactly once, we can utilize a depth-first search (DFS) approach combined with backtracking. Below is the step-by-step explanation, followed by the complete code formatted for LeetCode.

### Explanation

1. **Understanding the Grid**: The grid contains squares represented by:
   - `1`: Starting square.
   - `2`: Ending square.
   - `0`: Empty squares that we can walk over.
   - `-1`: Obstacles that cannot be traversed.

2. **Identify Relevant Squares**: We need to first identify the locations of the starting square (1), the ending square (2), and count the number of `0` squares, since every `0` (along with `1` and `2`) must be visited exactly once.

3. **DFS Search**: Using a recursive DFS approach, we will:
   - Move in all four directions (up, down, left, right).
   - Keep track of the path to ensure we don't revisit any square.
   - A counter will keep track of how many squares we've visited.

4. **Base Case**: We stop when we reach the ending square and have visited all non-obstacle squares exactly once. If we reach `2` and the count of visited squares equals the total count we need to traverse, we increment our path count.

5. **Backtracking**: After exploring from a given square, we backtrack by marking the square as unvisited (i.e., return it to `0`) to allow for different path explorations.

6. **Edge Cases**: Handle scenarios where there are no valid paths early in the function.

### Implementation

Here's the Python code implementing the above logic, formatted according to the LeetCode submission format:



```python
class Solution:
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        # Variable to hold the number of unique paths
        self.path_count = 0
        self.start = None
        self.end = None
        self.empty_squares = 0
        
        rows, cols = len(grid), len(grid[0])
        
        # Identify the start, end positions and count empty squares
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    self.start = (r, c)
                elif grid[r][c] == 2:
                    self.end = (r, c)
                if grid[r][c] == 0 or grid[r][c] == 1 or grid[r][c] == 2:
                    self.empty_squares += 1
        
        # Start the DFS from the starting position
        self.dfs(grid, self.start[0], self.start[1], 0)
        return self.path_count
    
    def dfs(self, grid, r, c, count):
        # Base cases:
        # If out of bounds or hit an obstacle, return
        if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or grid[r][c] == -1:
            return
        
        # If we reach the ending square
        if grid[r][c] == 2:
            # Check if we have visited all non-obstacle squares
            if count == self.empty_squares - 1:
                self.path_count += 1
            return
        
        # Mark the square as visited
        temp = grid[r][c]
        grid[r][c] = -1  # Mark as visited
        
        # Explore all four directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            self.dfs(grid, r + dr, c + dc, count + 1)
        
        # Backtrack: unmark the square
        grid[r][c] = temp  # Return to original state

```

### How to Use This Code
- This function can be directly called with `grid` as an argument.
- The result will be an integer representing the total number of unique paths from the starting point to the endpoint while visiting every accessible square exactly once.

# 996. Number of Squareful Arrays

### Problem Description 
Given an array `A` of non-negative integers, the array is squareful if for every pair of adjacent elements, their sum is a perfect square.

Return the number of permutations of A that are squareful.  Two permutations `A1` and `A2` differ if and only if there is some index `i` such that `A1[i] != A2[i]`.


Example 1:
Input: [1,17,8]
Output: 2
Explanation: 
[1,8,17] and [17,8,1] are the valid permutations.


Example 2:
Input: [2,2,2]
Output: 1
Note:
`1 <= A.length <= 12`
`0 <= A[i] <= 1e9`

### Solution 
 To solve the problem of counting the number of squareful permutations of an array `A`, we need to follow a structured approach involving finding valid pairs of adjacent elements whose sum is a perfect square, and then using backtracking to generate all valid permutations while ensuring that duplicates are handled correctly.

Here’s the step-by-step explanation and Python code following the LeetCode submission format:

### Steps to Solve:
1. **Perfect Square Check**: Create a function to check if a number is a perfect square.
2. **Pairing**: Create a function to check if two numbers can be adjacent in a squareful permutation based on their sums being perfect squares.
3. **Backtracking**: Use backtracking to explore permutations of the array while ensuring the conditions for squareful permutations are met.
4. **Avoid Duplicates**: To avoid counting duplicate permutations, we can sort the array at the beginning and skip duplicates during the permutation generation.
5. **Count Valid Permutations**: Keep a count of all valid permutations found.

### Python Code:


```python
from collections import Counter
from math import isqrt

class Solution:
    def numSquarefulPerms(self, A: List[int]) -> int:
        def is_perfect_square(n):
            if n < 0:
                return False
            root = isqrt(n)
            return root * root == n
        
        def backtrack(curr_perm):
            if len(curr_perm) == len(A):
                return 1
            
            count = 0
            for num in unique_nums:
                if count_map[num] > 0:  # If this number can still be used
                    # Check if we can add this number to the current permutation
                    if len(curr_perm) == 0 or is_perfect_square(curr_perm[-1] + num):
                        count_map[num] -= 1
                        curr_perm.append(num)
                        
                        count += backtrack(curr_perm)  # Recursively build the permutation
                        
                        curr_perm.pop()
                        count_map[num] += 1
            
            return count
        
        A.sort()  # Sort to facilitate the handling of duplicates
        unique_nums = list(set(A))
        count_map = Counter(A)  # Count occurrences of each number
        return backtrack([])


```

### Explanation of the Code:

1. **Perfect Square Function**:
   - The function `is_perfect_square(n)` uses integer square root (`isqrt`) to check if `n` is a perfect square by ensuring that squaring the root yields the original number.

2. **Backtracking Function**:
   - The `backtrack(curr_perm)` function constructs valid permutations recursively.
   - If the length of `curr_perm` is equal to the length of `A`, it means a valid permutation was formed, so it counts that permutation.
   - It iterates over the unique numbers and checks if it can add the current number to the permutation based on the squareful condition.
   - If adding the number is possible, it reduces the count of that number (using `count_map`), adds it to `curr_perm`, and calls itself recursively.
   - After exploring that path, it backtracks by removing the last number added and restoring its count.

3. **Main Function Logic**:
   - The input array is sorted to manage duplicates effectively.
   - A `Counter` is used to maintain counts of each number, and we start the backtracking process with an empty current permutation.

4. **Return Value**:
   - The function finally returns the total count of all unique squareful permutations.

This code will run efficiently within the provided constraints and can be directly submitted to LeetCode.

# 1066. Campus Bikes II

### Problem Description 
On a campus represented as a 2D grid, there are `N` workers and `M` bikes, with `N <= M`. Each worker and bike is a 2D coordinate on this grid.

We assign one unique bike to each worker so that the sum of the Manhattan distances between each worker and their assigned bike is minimized.

The Manhattan distance between two points `p1` and `p2` is `Manhattan(p1, p2) = |p1.x - p2.x| + |p1.y - p2.y|`.

Return the minimum possible sum of Manhattan distances between each worker and their assigned bike.


Example 1:
Input: workers = [[0,0],[2,1]], bikes = [[1,2],[3,3]]
Output: 6
Explanation: 
We assign bike 0 to worker 0, bike 1 to worker 1. The Manhattan distance of both assignments is 3, so the output is 6.


Example 2:
Input: workers = [[0,0],[1,1],[2,0]], bikes = [[1,0],[2,2],[2,1]]
Output: 4
Explanation: 
We first assign bike 0 to worker 0, then assign bike 1 to worker 1 or worker 2, bike 2 to worker 2 or worker 1. Both assignments lead to sum of the Manhattan distances as 4.


Example 3:
Input: workers = [[0,0],[1,0],[2,0],[3,0],[4,0]], bikes = [[0,999],[1,999],[2,999],[3,999],[4,999]]
Output: 4995

Constraints:
`N == workers.length`
`M == bikes.length`
`1 <= N <= M <= 10`
`workers[i].length == 2`
`bikes[i].length == 2`
`0 <= workers[i][0], workers[i][1], bikes[i][0], bikes[i][1] < 1000`
All the workers and the bikes locations are unique.

### Solution 
 To solve the problem of assigning bikes to workers while minimizing the total Manhattan distance, we can utilize a backtracking approach with memoization. Given the constraints of `N` (workers) being equal to or less than `M` (bikes) and the maximum `N` being `10`, this approach is feasible and efficient within these limits.

### Explanation:

1. **Understanding Manhattan Distance**: The Manhattan distance between two points `(x1, y1)` and `(x2, y2)` is calculated as:
   \[
   \text{Manhattan}(p1, p2) = |x1 - x2| + |y1 - y2|
   \]
   This distance gives us the cost for assigning a bike to a worker.

2. **State Representation**: We will use a bitmask to keep track of which bikes have already been assigned to workers. Since we have at most `10` workers, a bitmask from `0` to `2^M - 1` (where `M` is the number of bikes) will represent the current assignment of bikes.

3. **Backtracking with Memoization**:
   - We'll define a function that takes the current index of the worker and the current mask of assigned bikes.
   - If we have assigned a bike to all workers, we return `0` since no more distance needs to be counted.
   - For the current worker, we can iterate through all bikes, check if a bike is already assigned (using our bitmask), and if it is not, we calculate the distance and recursively call our function to assign bikes to the remaining workers.
   - We store results in a memoization dictionary to avoid redundant calculations.

4. **Recursive Function**:
   - The recursive function tries all possible assignments of the current worker to various bikes, accumulating the distances and keeping track of the minimum distance.

Here’s the complete code following the LeetCode solution format:



```python
class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        from functools import lru_cache
        
        N, M = len(workers), len(bikes)
        
        # Helper function to calculate the Manhattan distance
        def manhattan_distance(worker, bike):
            return abs(worker[0] - bike[0]) + abs(worker[1] - bike[1])
        
        # Using memoization to cache results
        @lru_cache(None)
        def backtrack(worker_index, bike_mask):
            # If all workers have been assigned a bike
            if worker_index == N:
                return 0
            
            min_distance = float('inf')
            # Go through all bikes
            for bike_index in range(M):
                if bike_mask & (1 << bike_index) == 0:  # If bike_index is not assigned
                    # Calculate the distance for the current assignment
                    distance = manhattan_distance(workers[worker_index], bikes[bike_index])
                    # Recursively assign bikes to the remaining workers
                    total_distance = distance + backtrack(worker_index + 1, bike_mask | (1 << bike_index))
                    min_distance = min(min_distance, total_distance)
                    
            return min_distance
        
        return backtrack(0, 0)


```

### How the Code Works:
- The `assignBikes` method initializes the number of workers and bikes.
- It defines a helper function `manhattan_distance` to compute the distances.
- The `backtrack` function performs the recursive assignment, checking if a bike is assigned using a bitmask, and calculates cumulative distances while tracking the minimum distance across different assignments.
- Finally, we call the `backtrack` function starting from the first worker and with no bikes assigned (mask `0`).

### Time Complexity:
The time complexity of this solution is approximately `O(N * M * 2^M)` because for each worker, we try each bike which leads us to explore `2^M` states.

This solution is efficient and optimal given the problem constraints.

# 1079. Letter Tile Possibilities

### Problem Description 
You have `n`  `tiles`, where each tile has one letter `tiles[i]` printed on it.

Return the number of possible non-empty sequences of letters you can make using the letters printed on those `tiles`.


Example 1:
Input: tiles = "AAB"
Output: 8
Explanation: The possible sequences are "A", "B", "AA", "AB", "BA", "AAB", "ABA", "BAA".


Example 2:
Input: tiles = "AAABBC"
Output: 188

Example 3:
Input: tiles = "V"
Output: 1

Constraints:
`1 <= tiles.length <= 7`
`tiles` consists of uppercase English letters.

### Solution 
 To solve the problem of counting the number of distinct non-empty sequences that can be formed from the letters on the tiles, we can utilize a backtracking approach. The key steps involve generating all possible combinations of the characters while ensuring that duplicates are handled correctly. This will allow us to account for the fact that tiles with the same letter can create identical sequences.

Here's a detailed breakdown of the approach we'll take:

1. **Backtracking**: We'll use a helper function to recursively build sequences from the tiles. This function will take the current sequence being built and a frequency dictionary to track how many of each letter are left to use.

2. **Tracking Unique Sequences**: We need to ensure that we track unique sequences. We'll use a set to record all the unique sequences we generate.

3. **Frequency Dictionary**: We'll keep track of the count of each letter to avoid using the same letter more times than it appears in the original tiles.

4. **Base Cases**: Each time we add a letter to our current sequence, we'll consider that sequence as a valid combination. Therefore, we add it to our results before continuing to explore further combinations.

5. **Complexity Considerations**: The maximum length of the tiles is 7, which makes the problem manageable for a backtracking solution even though it may seem computationally intensive at first glance.

Here's how the implementation looks in Python, formatted for LeetCode:



```python
class Solution:
    def numTilePossibilities(self, tiles: str) -> int:
        from collections import Counter

        # Store frequencies of each character in the string
        freq = Counter(tiles)
        ans_set = set()

        def backtrack(current_seq):
            if current_seq:
                ans_set.add(current_seq)

            for char in freq:
                if freq[char] > 0:  # If we have this character available
                    # Use the character
                    freq[char] -= 1
                    # Recur to build sequences with the current character included
                    backtrack(current_seq + char)
                    # Backtrack, restore the character count
                    freq[char] += 1

        backtrack("")  # Start with an empty sequence
        return len(ans_set)  # Return the count of unique sequences

# Example usage:
# sol = Solution()
# print(sol.numTilePossibilities("AAB"))  # Output: 8
# print(sol.numTilePossibilities("AAABBC"))  # Output: 188
# print(sol.numTilePossibilities("V"))  # Output: 1

```

### Explanation of the Code:
- **Counter**: We create a frequency dictionary using `collections.Counter` to count how many times each character appears in `tiles`.
- **Set for Unique Sequences**: We use a set `ans_set` to store all unique sequences generated during backtracking.
- **Backtracking Logic**: The function `backtrack` constructs the sequences:
  - If `current_seq` is not empty, we add it to `ans_set`.
  - For each character, if we have any left (`freq[char] > 0`), we reduce its count, invoke the backtrack function adding this character, and then restore its count (backtrack).
- Finally, we return the size of `ans_set`, which gives the number of distinct non-empty sequences.

### Testing the Code:
You can run the provided example usage at the end of the code. The expected outputs for the test cases are also included as comments. This setup can be run directly on the LeetCode platform.

# 1087. Brace Expansion

### Problem Description 
You are given a string `s` representing a list of words. Each letter in the word has one or more options.

If there is one option, the letter is represented as is.

If there is more than one option, then curly braces delimit the options. For example, `"{a,b,c}"` represents options `["a", "b", "c"]`.

For example, if `s = "a{b,c}"`, the first character is always `'a'`, but the second character can be `'b'` or `'c'`. The original list is `["ab", "ac"]`.

Return all words that can be formed in this manner, sorted in lexicographical order.


Example 1:
Input: s = "{a,b}c{d,e}f"
Output: ["acdf","acef","bcdf","bcef"]

Example 2:
Input: s = "abcd"
Output: ["abcd"]

Constraints:
`1 <= s.length <= 50`
`s` consists of curly brackets `'{}'`, commas `','`, and lowercase English letters.

`s` is guaranteed to be a valid input.

There are no nested curly brackets.

All characters inside a pair of consecutive opening and ending curly brackets are different.

### Solution 
 To solve the given problem of generating all possible words based on the provided string format, we can use a backtracking approach or an iterative method to handle the combinations of characters specified within the curly braces. Let's implement a solution in Python.

### Explanation:

1. **Parsing the Input**:
   - We need to split the string into distinct parts where each part will be either a standalone character or a set of options enclosed in braces (`{}`).
   - We will maintain a list called `parts` that stores these segments, whether they are single characters or lists of options.

2. **Generating Combinations**:
   - After parsing the input string and collecting the segments in `parts`, we need to generate various combinations. We can do this using the `itertools.product` which will take the Cartesian product of the sets of options.

3. **Sorting the Output**:
   - The final result must be sorted lexicographically before returning.

### Solution Implementation:

Below is the code that accomplishes this using the mentioned steps:



```python
from itertools import product

def expand(s: str) -> list[str]:
    parts = []
    i = 0
    n = len(s)

    while i < n:
        if s[i] == '{':  # Start of options
            j = i + 1
            options = []

            while j < n and s[j] != '}':
                if s[j] != ',':
                    options.append(s[j])  # collect the individual options
                j += 1
            
            parts.append(options)  # append the options to parts
            i = j + 1  # move past '}'
        else:
            parts.append([s[i]])  # append the single character as a list
            i += 1

    # Generate all combinations using the Cartesian product of parts
    combinations = [''.join(option) for option in product(*parts)]

    return sorted(combinations)  # return sorted list of combinations

# Examples to test the function
print(expand("{a,b}c{d,e}f"))  # Output: ["acdf","acef","bcdf","bcef"]
print(expand("abcd"))          # Output: ["abcd"]

```

### Explanation of the Code:

1. **Importing the Product Function**:
   - We import `product` from the `itertools` module to help create the Cartesian product of the options.

2. **Expanding the Input**:
   - We iterate through the string and check if the character is a brace '{'. 
   - If it is, we gather all options until we find the corresponding closing brace '}'.
   - If it's a regular character, we simply append that character (wrapped in a list) to `parts`.

3. **Generating the Combinations**:
   - After constructing `parts`, which consists of either characters or lists of options, we use `product(*parts)` to get all possible combinations.

4. **Sorting the Results**:
   - Finally, we return the sorted list of combinations as the result.

This implementation works efficiently within the constraints provided and handles various edge cases such as strings without braces.

# 1088. Confusing Number II

### Problem Description 
We can rotate digits by 180 degrees to form new digits. When 0, 1, 6, 8, 9 are rotated 180 degrees, they become 0, 1, 9, 8, 6 respectively. When 2, 3, 4, 5 and 7 are rotated 180 degrees, they become invalid.

A confusing number is a number that when rotated 180 degrees becomes a different number with each digit valid.(Note that the rotated number can be greater than the original number.)
Given a positive integer `N`, return the number of confusing numbers between `1` and `N` inclusive.


Example 1:
Input: 20
Output: 6
Explanation: 
The confusing numbers are [6,9,10,16,18,19].

6 converts to 9.

9 converts to 6.

10 converts to 01 which is just 1.

16 converts to 91.

18 converts to 81.

19 converts to 61.


Example 2:
Input: 100
Output: 19
Explanation: 
The confusing numbers are [6,9,10,16,18,19,60,61,66,68,80,81,86,89,90,91,98,99,100].

Note:
`1 <= N <= 10^9`

### Solution 
 To solve the problem of counting confusing numbers between `1` and `N`, we start by identifying which digits can be rotated and what they transform into when viewed from a 180-degree angle.

### Step-by-step Breakdown:

1. **Understanding Rotations**: The valid digits and their transformations upon rotating 180 degrees are:
   - `0 -> 0`
   - `1 -> 1`
   - `6 -> 9`
   - `8 -> 8`
   - `9 -> 6`

   The digits `2, 3, 4, 5, 7` do not form valid numbers after rotation, making them invalid for our count of confusing numbers.

2. **Definition of a Confusing Number**:
   A confusing number is defined as a number that, when rotated, results in a different number. For example:
   - `6` becomes `9`, which is different.
   - `10` becomes `01`, which is treated as `1` (valid but not different).
   - `18` becomes `81`, which is different and valid.
   - `9` becomes `6`, which is valid and different.

3. **Constraints**: The problem states `1 <= N <= 10^9`, meaning our solution should be efficient and should not involve simple iteration from `1` to `N` due to performance issues.

4. **Recursion and Backtracking**: Given the potential number of digits in `N`, we can utilize a backtracking approach to generate numbers using the valid digits. We will recursively build numbers using the valid digits until we exceed `N`.

### Implementation:

Now, let's implement the solution in Python following the LeetCode problem structure:



```python
class Solution:
    def confusingNumberII(self, N: int) -> int:
        # Mapping of valid digits and their rotated counterparts
        valid_digits = ['0', '1', '6', '8', '9']
        rotation_map = {'0': '0', '1': '1', '6': '9', '8': '8', '9': '6'}
        
        # Set to store confusing numbers
        confusing_count = 0
        
        def is_confusing(num_str: str) -> bool:
            # Check if the rotated version is different
            rotated = ''.join(rotation_map[ch] for ch in reversed(num_str))
            return rotated != num_str and rotated[0] != '0'  # cannot start with '0'
        
        def backtrack(current_num_str: str) -> None:
            nonlocal confusing_count
            
            # Convert the current number string into an integer
            if current_num_str:
                num = int(current_num_str)
                if num > N:
                    return
                if is_confusing(current_num_str):
                    confusing_count += 1
            
            # Generate further numbers based on the valid digits
            for digit in valid_digits:
                # Skip leading zeroes
                if not current_num_str and digit == '0':
                    continue
                backtrack(current_num_str + digit)
        
        # Start backtracking from an empty string
        backtrack("")
        
        return confusing_count

```

### Explanation of the Code:

1. **`valid_digits` and `rotation_map`**: We define the digits we can use and their rotations.
  
2. **`is_confusing` Function**: This function checks if the rotated version of a given number string is different from the original and does not lead with `0`.

3. **`backtrack` Function**: This function recursively builds valid numbers. If a valid number is found, it checks if it's confusing and increments the count if so. The base case is when the number exceeds `N`.

4. **Recursive Backtracking**: It generates numbers by appending each valid digit to the current number string, ensuring we avoid leading zeros.

This structure is efficient, and by recursively exploring valid configurations of numbers up to `N`, we achieve the solution without oversaturating memory and maintaining a time complexity that is feasible within reasonable input sizes.

# 1215. Stepping Numbers

### Problem Description 
A Stepping Number is an integer such that all of its adjacent digits have an absolute difference of exactly `1`. For example, `321` is a Stepping Number while `421` is not.

Given two integers `low` and `high`, find and return a sorted list of all the Stepping Numbers in the range `[low, high]` inclusive.


Example 1:
Input: low = 0, high = 21
Output: [0,1,2,3,4,5,6,7,8,9,10,12,21]

Constraints:
`0 <= low <= high <= 2 * 10^9`

### Solution 
 Certainly! Let's tackle the problem of finding all Stepping Numbers in the given range `[low, high]`. 

### Problem Analysis
A "Stepping Number" is defined as an integer where every pair of adjacent digits has an absolute difference of exactly `1`. The task is to identify all such numbers falling within the specified range.

### Approach
To efficiently generate Stepping Numbers, we employ a Breadth-First Search (BFS) approach. Here's the step-by-step explanation of the method:

1. **Digit-Level Exploration**: Starting from each digit (0 through 9), treat these digits as nodes in a graph. Each node can traverse to another node if the difference between the digits is exactly `1`.
  
2. **Breadth-First Search (BFS)**: 
   - For each digit, if it's between the bounds of `low` and `high`, it’s added to the result list.
   - From a number, we can generate its "neighbors" (the next valid Stepping Numbers) by adding or subtracting `1` from its last digit and appending that to the number.
   - We ensure that the new digit is valid (i.e., remains between `0` and `9`).

3. **Termination Condition**: We continue this exploration until no new valid numbers can be generated.

4. **Sorting**: Finally, return the sorted results after BFS generation.

### Implementation
Here is the Python code that implements the above logic:



```python
from collections import deque

class Solution:
    def steppingNumbers(self, low: int, high: int) -> List[int]:
        result = []
        
        # BFS function to generate stepping numbers
        def bfs(start):
            queue = deque([start])
            while queue:
                number = queue.popleft()
                
                # If the number is within the bounds, add to result
                if low <= number <= high:
                    result.append(number)
                
                # Get the last digit of the current number
                last_digit = number % 10
                
                # Next possible stepping numbers
                if last_digit > 0:  # If last digit is greater than 0 we can subtract 1
                    next_number = number * 10 + (last_digit - 1)
                    if next_number <= high:
                        queue.append(next_number)
                
                if last_digit < 9:  # If last digit is less than 9 we can add 1
                    next_number = number * 10 + (last_digit + 1)
                    if next_number <= high:
                        queue.append(next_number)

        # Start BFS from each digit from 0 to 9
        for i in range(10):
            bfs(i)

        # Sort the result list before returning
        result.sort()
        return result


```

### Explanation of the Code
1. **Imports**: We import `deque` from `collections` for efficient queue operations.
2. **Function Definition**: The `steppingNumbers` function is defined within the `Solution` class as per LeetCode requirements.
3. **BFS Implementation**:
   - A helper function `bfs(start)` generates all Stepping Numbers from a given starting point.
   - We use a queue (FIFO structure) to explore each number and its potential neighbors.
4. **Boundary Checks**: Check if the generated number is in the specified range before adding to the results.
5. **Result Sorting**: Finally, we sort the results before returning, ensuring they are in ascending order.

### Complexity
- The time complexity is quite efficient for the range given the constraints, since we are generating numbers iteratively. Each number will branch into at most two new numbers, making the BFS manageable.
- The space complexity is also reasonable due to the limited size of the queue and result list.

This solution can be submitted directly on LeetCode and should handle the constraints effectively.

# 1219. Path with Maximum Gold

### Problem Description 
In a gold mine `grid` of size `m x n`, each cell in this mine has an integer representing the amount of gold in that cell, `0` if it is empty.

Return the maximum amount of gold you can collect under the conditions:
Every time you are located in a cell you will collect all the gold in that cell.

From your position, you can walk one step to the left, right, up, or down.

You can't visit the same cell more than once.

Never visit a cell with `0` gold.

You can start and stop collecting gold from any position in the grid that has some gold.


Example 1:
Input: grid = [[0,6,0],[5,8,7],[0,9,0]]
Output: 24
Explanation:
[[0,6,0],
 [5,8,7],
 [0,9,0]]
Path to get the maximum gold, 9 -> 8 -> 7.


Example 2:
Input: grid = [[1,0,7],[2,0,6],[3,4,5],[0,3,0],[9,0,20]]
Output: 28
Explanation:
[[1,0,7],
 [2,0,6],
 [3,4,5],
 [0,3,0],
 [9,0,20]]
Path to get the maximum gold, 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7.


Constraints:
`m == grid.length`
`n == grid[i].length`
`1 <= m, n <= 15`
`0 <= grid[i][j] <= 100`
There are at most 25 cells containing gold.

### Solution 
 To solve the problem of collecting the maximum amount of gold from a grid while adhering to the specified movement constraints, we can use Depth-First Search (DFS). The recursive nature of DFS allows us to explore all possible paths starting from each cell containing gold while keeping track of the cells we've already visited to avoid collecting gold from the same cell multiple times.

Here's a step-by-step breakdown of the approach:

### Steps to Solve

1. **Initialization**: Define the grid dimensions and initialize necessary variables. The number of rows (`m`) and columns (`n`) of the grid will be determined from the input grid.

2. **DFS Implementation**: Create a recursive function that:
    - Takes the current coordinates and the current collected gold as parameters.
    - Marks the current cell as visited by temporarily changing its value (setting it to 0).
    - Explores all four possible directions (up, down, left, right).
    - Recursively calls itself on the adjacent cells if they contain gold (i.e., value > 0).
    - Backtracks by restoring the original value of the cell after exploring all paths from it.
    
3. **Main Logic**: Iterate over each cell in the grid:
    - If the cell contains gold, invoke the DFS function from that cell to collect the maximum gold possible starting from that position.
    - Maintain a global variable to keep track of the maximum gold collected during these explorations.

4. **Output**: Return the maximum gold collected.

### Python Code

Here's how you can implement the above logic using Python:



```python
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        max_gold = 0
        
        def dfs(x: int, y: int, gold_collected: int) -> int:
            # Temporarily mark the cell as visited
            temp = grid[x][y]
            grid[x][y] = 0  # Mark as visited (avoid visiting again)
            current_max = gold_collected + temp
            
            # Directions for up, down, left, right
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] > 0:
                    current_max = max(current_max, dfs(nx, ny, current_max))
            
            # Restore the cell's original value after visiting
            grid[x][y] = temp
            
            return current_max
        
        # Start DFS from every cell with gold
        for i in range(m):
            for j in range(n):
                if grid[i][j] > 0:
                    max_gold = max(max_gold, dfs(i, j, 0))

        return max_gold

```

### Explanation of the Code:

- **Class Definition**: `Solution` is the class that encapsulates our solution, which is standard in LeetCode problems.
- **getMaximumGold Method**: This method is the entry point. It initializes matrix dimensions and the `max_gold` variable.
- **DFS Function**: This function performs the depth-first search:
    - It collects gold from the current cell, saves the current gold count, and then explores all adjacent cells that are valid (within bounds and contain gold).
    - The cell is marked as visited by setting its value to 0 to prevent revisits. After exploring all paths from a cell, the original value is restored.
- **Iteration through Grid**: At the end, we iterate through each cell in the grid, triggering the DFS wherever we find a cell with gold.

This solution efficiently explores all potential paths within the grid while ensuring we collect the maximum amount of gold possible according to the problem's constraints.

# 1239. Maximum Length of a Concatenated String with Unique Characters

### Problem Description 
Given an array of strings `arr`. String `s` is a concatenation of a sub-sequence of `arr` which have unique characters.

Return the maximum possible length of `s`.


Example 1:
Input: arr = ["un","iq","ue"]
Output: 4
Explanation: All possible concatenations are "","un","iq","ue","uniq" and "ique".

Maximum length is 4.


Example 2:
Input: arr = ["cha","r","act","ers"]
Output: 6
Explanation: Possible solutions are "chaers" and "acters".


Example 3:
Input: arr = ["abcdefghijklmnopqrstuvwxyz"]
Output: 26

Constraints:
`1 <= arr.length <= 16`
`1 <= arr[i].length <= 26`
`arr[i]` contains only lower case English letters.

### Solution 
 To solve the problem of finding the maximum length of a string formed by concatenating unique-character strings from the given list, we can use a backtracking approach. 

Here's how we can break down the solution:

### Explanation:

1. **Input Constraints**: Given the constraints where `1 <= arr.length <= 16` and `1 <= arr[i].length <= 26`, we can take advantage of bitwise operations to efficiently check for unique characters in strings.
  
2. **Bitmask Representation**: Each string can be represented using a bitmask, where each bit in the integer represents whether a particular character ('a' to 'z') is included in the string. This way, we can easily check for overlaps (i.e., common characters between strings) using bitwise operations.

3. **Backtracking**: We'll use a recursive backtracking approach to explore all combinations of strings we can concatenate. At each step, we decide whether to include a string or skip it.

4. **Skip Overlapping Strings**: Before including a string in the current combination, we'll check if it shares any characters with the string we've already formed (using the bitmask representation). If it does, we skip that string.

5. **Track Maximum Length**: As we form valid concatenated strings, we will keep track of the maximum length found.

### Python Code:

Here's the Python code structured in the LeetCode solution format:



```python
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        
        def is_unique(s: str) -> bool:
            """ Check if string 's' has all unique characters. """
            char_set = set()
            for char in s:
                if char in char_set:
                    return False
                char_set.add(char)
            return True
        
        def backtrack(index: int, current_mask: int) -> int:
            """ Recursive function to find the maximum length of concatenated string with unique characters. """
            max_length = 0
            
            # Try to concatenate strings starting from the current index
            for i in range(index, len(arr)):
                next_mask = self.get_mask(arr[i])
                
                # Check if next string (next_mask) shares any letters with the current combination (current_mask)
                if current_mask & next_mask == 0:  # no overlap
                    max_length = max(max_length, 
                                     len(arr[i]) + backtrack(i + 1, current_mask | next_mask))
            
            return max_length
        
        # Returns the bitmask representation for the string
        def get_mask(s: str) -> int:
            mask = 0
            for char in s:
                mask |= (1 << (ord(char) - ord('a')))
            return mask
        
        # Main execution begins here
        result = backtrack(0, 0)
        return result

```

### Explanation of the Code:

1. **Function Definitions**:
   - `is_unique(s: str) -> bool`: A helper function that checks if a string has all unique characters. (In this solution, it's not used as we convert strings to masks directly.)
   - `backtrack(index: int, current_mask: int) -> int`: This is our main recursive function. It explores all combinations of strings starting from the `index` and keeps track of the current set of unique characters using `current_mask`.

2. **Combination Logic**:
   - We iterate over the strings using the `for` loop.
   - For each string, convert it into a bitmask (`next_mask`).
   - We check if the string can be added (i.e., if it does not share characters with the current concatenation). If it can be added, we update the maximum length.

3. **Bitmask Generation**:
   - `get_mask(s: str) -> int`: Converts a string into its bitmask representation, where the ith bit is set if the character corresponding to that bit (i.e., 'a' + i) is present in `s`.

4. **Execution**:
   - Finally, the recursion starts from index 0 with an initial `current_mask` of 0. The result will be the maximum length of any valid concatenation found.

This solution efficiently examines all combinations of strings and ensures that only unique-character concatenations are considered. It employs bit masking to facilitate fast checks for overlaps, resulting in an effective approach to solving the problem.

