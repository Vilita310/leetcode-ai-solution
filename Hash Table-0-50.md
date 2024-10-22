# 1. Two Sum

### Problem Description 
Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Output: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:
Input: nums = [3,3], target = 6
Output: [0,1]

Constraints:
`2 <= nums.length <= 103`
`-109 <= nums[i] <= 109`
`-109 <= target <= 109`
Only one valid answer exists.

### Solution 
 

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        Given a list of integers (nums) and a target integer,
        the function returns the indices of the two numbers that add up to the target.
        
        The function uses hashing (dictionary) to store indices of previous numbers.
        For each element in the array, calculate the complement (target - current number).
        If the complement exists in the dictionary, a solution is found.
        
        Args:
        nums (List[int]): List of integers.
        target (int): The target sum to find in the array.
          
        Returns:
        List[int]: List of indices of the two numbers from nums that add up to target.
        """
        
        num_to_index = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_to_index:
                return [num_to_index[complement], i]
            num_to_index[num] = i
        
        return []

# The class Solution can now be used to create an instance and call the twoSum() method. Here's how you would run the above code on LeetCode:

# nums = [2, 7, 11, 15]
# target = 9
# sol = Solution()
# print(sol.twoSum(nums, target))  # Output will be [0, 1]

```

### Explanation:
1. **Dictionary for O(1) Lookup:** We utilize a Python dictionary (`num_to_index`) to map each number to its index. This allows O(1) time complexity for checking if the complement of the current number exists.

2. **Iterating Through the List:** Loop through the list `nums`. For each `num`, determine the `complement` needed to reach the `target` (`target - num`).

3. **Check and Store:** If the `complement` has already been seen (exists as a key in `num_to_index`), return the respective indices `[num_to_index[complement], i]`. Otherwise, store the current number and its index in `num_to_index`.

4. **Complete:**
    - Time Complexity: O(n), where n is the length of the list `nums`.

    - Space Complexity: O(n), for storing up to n elements in the dictionary.

# 3. Longest Substring Without Repeating Characters

### Problem Description 
Given a string `s`, find the length of the longest substring without repeating characters.


Example 1:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.


Example 2:
Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.


Example 3:
Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.

Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.


Example 4:
Input: s = ""
Output: 0

Constraints:
`0 <= s.length <= 5 * 104`
`s` consists of English letters, digits, symbols and spaces.

### Solution 
 We can approach this problem using the sliding window technique, specifically using a two-pointer strategy to track the longest substring without repeating characters. The main idea is to expand one pointer to explore new characters and use another pointer (or move the same pointer) to adjust the window when a repeat character is found.

Here's a detailed step-by-step breakdown of the solution:
1. Utilize a set to keep track of characters in the current window, which helps in checking for repeats in O(1) time.
2. Maintain two pointers (indices) `start` and `end` to represent the current window of characters without any repeats. Initially, both of these are set to 0.
3. Traverse the string with the `end` pointer, expanding the window by adding characters from the right.
4. If the character at the `end` pointer is in the set (meaning it's a repeat), move the `start` pointer to the right until the character is removed. During each such adjustment, remove characters from the set as well.
5. At each step, calculate the window size and update the maximum length found.
6. Return the maximum length after the whole string has been processed.

Below is the Python code implementing this strategy, adhering to LeetCode's format:



```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        char_set = set()
        l = 0
        max_len = 0
        
        for r in range(len(s)):
            while s[r] in char_set:
                char_set.remove(s[l])
                l += 1
            char_set.add(s[r])
            max_len = max(max_len, r - l + 1)
            
        return max_len

```

This code defines a class `Solution` with a method `lengthOfLongestSubstring()`, which takes a string `s` as input and returns the length of the longest substring without repeating characters. During traversal, it efficiently manages the window by adjusting the set of characters it contains and always keeps the `max_len` up to date with the size of the longest valid window found.

# 18. 4Sum

### Problem Description 
Given an array `nums` of n integers and an integer `target`, are there elements a, b, c, and d in `nums` such that a + b + c + d = `target`? Find all unique quadruplets in the array which gives the sum of `target`.

Notice that the solution set must not contain duplicate quadruplets.


Example 1:
Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]

Example 2:
Input: nums = [], target = 0
Output: []

Constraints:
`0 <= nums.length <= 200`
`-109 <= nums[i] <= 109`
`-109 <= target <= 109`

### Solution 
 First, let's break down the problem. The aim is to find all unique quadruplets in an array that sum up to a specific target value. One efficient way to address this is to use a mix of sorting and two-pointer technique.

### Steps to Solve the Problem:
1. **Sort the Array:** This helps in easily skipping duplicates and also aids the two-pointer approach for finding pairs with a given sum.
2. **Iterate with Two Nested Loops:** Use two loops to decide the first two elements of the quadruplet.
3. **Two-pointer Technique:** For each pair selected by the loops, use two pointers to find the required pair that makes the total sum equal to the target.
4. **Skip Duplicates:** After finding each quadruplet, skip the subsequent numbers if they are the same as the current one to avoid duplicates in the result.

### LeetCode Solution in Python:
In accordance with the LeetCode required format, here is a direct solution.



```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()  # Sort the array to facilitate the two-pointer technique and duplicate skipping.
        n = len(nums)
        result = []
        
        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]: # Skip duplicate elements for the first in quadruplet.
                continue
            for j in range(i + 1, n):
                if j > i + 1 and nums[j] == nums[j - 1]: # Skip duplicate elements for the second in quadruplet.
                    continue
                
                left, right = j + 1, n - 1
                while left < right:
                    quad_sum = nums[i] + nums[j] + nums[left] + nums[right]
                    if quad_sum == target:
                        result.append([nums[i], nums[j], nums[left], nums[right]])
                        # Move left and right to the next different numbers to avoid duplicates in the result
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        left += 1
                        right -= 1
                    elif quad_sum < target:
                        left += 1
                    else:
                        right -= 1
        
        return result

```

### Explanation:
- **Sorting:** First, the list `nums` is sorted.
- **Outer Loops:** The loops on iterator `i` and `j` pick the first and second elements of the quadruplet, respectively. To ensure they select distinct values in an ordered manner, we skip over any duplicate values immediately following the chosen element.
- **Inner Two-Pointer Loop:** For a fixed pair given by the outer loops, the pointers `left` and `right` try to find two other elements that complete the quadruplet such that their sum equals the `target`.
- **Avoiding Duplicates:** Upon finding a quadruplet, the left and right pointers skip past values that are the same to prevent recording duplicate solutions.
- The result is accumulated in `result` and returned.

This solution efficiently leverages sorting and two-pointer techniques to ensure a thorough yet optimized search through the possible quadruplets.

# 30. Substring with Concatenation of All Words

### Problem Description 
You are given a string `s` and an array of strings `words` of the same length. Return all starting indices of substring(s) in `s` that is a concatenation of each word in `words` exactly once, in any order, and without any intervening characters.

You can return the answer in any order.


Example 1:
Input: s = "barfoothefoobarman", words = ["foo","bar"]
Output: [0,9]
Explanation: Substrings starting at index 0 and 9 are "barfoo" and "foobar" respectively.

The output order does not matter, returning [9,0] is fine too.


Example 2:
Input: s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]
Output: []

Example 3:
Input: s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]
Output: [6,9,12]

Constraints:
`1 <= s.length <= 104`
`s` consists of lower-case English letters.

`1 <= words.length <= 5000`
`1 <= words[i].length <= 30`
`words[i]` consists of lower-case English letters.

### Solution 
 We can solve this problem via a sliding window and hash mapping approach based on the lengths of the strings in `words`. Here's the proposed solution:

1. **Preprocessing Step**: Build a dictionary of words (`word_count`) from the list called `words` to keep track of the frequency of each word.

2. **Sliding Window Approach**:
    - Determine variables such as the length of the words (`word_len`) and the total number of characters in the concatenation (`concat_len`) which is `word_len multiplied by the number of words`.
    - If `concat_len` is greater than the length of `s`, it's clear there won't be any valid starting indices, and hence we can return an empty list immediately.
    - Use multiple pointers (or indices) to slide over the string `s`:
        - Start up to `word_len` sliding windows (to account for different offsets). For instance, if the `word_len` is 3, start windows at indices: 0, 1, and 2.
        - For each window, extract substrings of length `word_len` and compare against the `word_count`.
        - Use a temporary dictionary (`current_count`) to track occurrences of the words within the current window.
        - Move the window in increments of `word_len`. If the words in the current window match the ones in the `word_count` exactly in number, consider the start of that window as a valid result.

3. **Validation of Window**:
    - For any window spanning from index `i` to `i + concat_len-1`, validate if all words from `words` appear exactly according to their counts in the window. If validated, add `i` to the result list.

4. Return the collected starting indices.

Below is the full Python code adjusted to the described approach:



```python
from collections import Counter

class Solution:
    def findSubstring(self, s: str, words: list) -> list:
        if not words or not s:
            return []
        
        word_len = len(words[0])
        word_count = Counter(words)
        num_words = len(words)
        concat_len = word_len * num_words
        result = []
        
        for i in range(word_len):
            left = i
            right = i
            current_count = Counter()
            
            while right + word_len <= len(s):
                # add the new word to the current window
                word = s[right:right + word_len]
                current_count[word] += 1
                right += word_len
                
                # Check if the window is the correct size
                while current_count[word] > word_count[word]:
                    left_word = s[left:left + word_len]
                    current_count[left_word] -= 1
                    left += word_len
                
                # Check if the window matches exactly the word_count
                if right - left == concat_len:
                    result.append(left)
        
        return result

# Test run on LeetCode for this code should ideally be consistent with above examples

```

This code will be efficient, especially taking advantage of modern Python libraries like `collections.Counter` to handle counting efficiently, and executes within reasonable time limits even for larger input sizes of `s` and `words`.

# 36. Valid Sudoku

### Problem Description 
Determine if a `9 x 9` Sudoku board is valid. Only the filled cells need to be validated according to the following rules:
Each row must contain the digits `1-9` without repetition.

Each column must contain the digits `1-9` without repetition.

Each of the nine `3 x 3` sub-boxes of the grid must contain the digits `1-9` without repetition.

Note:
A Sudoku board (partially filled) could be valid but is not necessarily solvable.

Only the filled cells need to be validated according to the mentioned rules.


Example 1:
Input: board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: true

Example 2:
Input: board = 
[["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: false

Explanation: Same as Example 1, except with the 5 in the top left corner being modified to 8. Since there are two 8's in the top left 3x3 sub-box, it is invalid.


Constraints:
`board.length == 9`
`board[i].length == 9`
`board[i][j]` is a digit or `'.'`.

### Solution 
 To determine if a Sudoku board configuration is valid, we should validate three key constraints:
1. Rows should not have the same number repeated (excluding '.').
2. Columns should not have the same number repeated (excluding '.').
3. The nine \(3 \times 3\) sub-boxes should adhere to the same uniqueness rule within their confines.

Given this problem's structure, one practical approach is to utilize sets to track which numbers have already been encountered during the scan of the board:
- We create three lists of sets: one for rows, one for columns, and one for sub-boxes. Each list will have 9 sets corresponding to the 9 different rows, columns, or boxes respectively.

- As we iterate over each cell of the board, we can easily determine:
  - The index for the row-set directly from the row iterator.
  - The index for the column-set directly from the column iterator.
  - The index for the sub-box-set by computing `(row // 3) * 3 + (col // 3)`.

If we encounter a cell with a digit (not '.'), we:
- Check if the digit is in the corresponding row's set, column's set, or box's set.
- If the digit is found in any of the sets, it's a violation of the rules, and we should return `False`.
- Otherwise, add the digit to the appropriate sets and continue.

This systematic checking allows us to ensure the board either adheres to the Sudoku rules or doesn't in a single traversal of the board.

We'll now implement this solution in Python following the format compatible with direct use in the LeetCode environment.



```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        
        for r in range(9):
            for c in range(9):
                val = board[r][c]
                if val == '.':
                    continue
                
                if val in rows[r]:
                    return False
                rows[r].add(val)
                
                if val in cols[c]:
                    return False
                cols[c].add(val)
                
                box_index = (r // 3) * 3 + (c // 3)
                if val in boxes[box_index]:
                    return False
                boxes[box_index].add(val)
        
        return True

```

This code defines a `Solution` class with a method `isValidSudoku`, which takes a `board` input and outputs a boolean representing whether the board is a valid Sudoku configuration based on the rules specified. Each check and structure used ensure we maintain efficient lookups, updates, and validations without the need for complicated logic or multiple passes over the board.

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
 Here's a Python function that solves a Sudoku puzzle using backtracking. The solution adheres to the typical format expected on LeetCode, where you define a class with a method that takes the problem input and modifies it in place.

The function checks if a number placement is valid using three rules: row, column, and the 3x3 box constraints. If a valid placement is found, it proceeds to place a number and continue solving. If no valid placement leads to a solution, it backtracks by unplacing the number and trying another.

Here's the code:



```python
class Solution:
    def solveSudoku(self, board):
        """
        Do not return anything, modify board in-place instead.
        """
        def is_valid(row, col, c):
            for i in range(9):
                if board[i][col] == c:
                    return False
                if board[row][i] == c:
                    return False
                if board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == c:
                    return False
            return True

        def backtrack():
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        for c in '123456789':
                            if is_valid(i, j, c):
                                board[i][j] = c
                                if backtrack():
                                    return True
                                board[i][j] = '.'
                        return False
            return True
        
        backtrack()

# Example usage
board = [
    ["5","3",".",".","7",".",".",".","."],
    ["6",".",".","1","9","5",".",".","."],
    [".","9","8",".",".",".",".","6","."],
    ["8",".",".",".","6",".",".",".","3"],
    ["4",".",".","8",".","3",".",".","1"],
    ["7",".",".",".","2",".",".",".","6"],
    [".","6",".",".",".",".","2","8","."],
    [".",".",".","4","1","9",".",".","5"],
    [".",".",".",".","8",".",".","7","9"]
]

solver = Solution()
solver.solveSudoku(board)
print(board)  # The board should be filled properly now

```

### Explanation
1. **is_valid:** This helper function checks if placing the digit `c` at position `(row, col)` is valid according to Sudoku rules.
2. **backtrack:** This recursive function attempts to fill the Sudoku grid. It loops through each cell of the grid, proposes digits from '1' to '9', and uses the `is_valid` function to check if the proposed digit can be placed. If the placement of a digit leads to a contradiction later (i.e., a dead end), it backtracks by unplacing the digit and trying the next one.
3. **Base case of backtrack:** If every cell is filled correctly, `backtrack()` will eventually fill the grid and return `True`.

This solution ensures that the input Sudoku puzzle (provided in the `board` list) is filled in place, satisfying all the given constraints of a valid Sudoku solution.

# 49. Group Anagrams

### Problem Description 
Given an array of strings `strs`, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.


Example 1:
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Example 2:
Input: strs = [""]
Output: [[""]]

Example 3:
Input: strs = ["a"]
Output: [["a"]]

Constraints:
`1 <= strs.length <= 104`
`0 <= strs[i].length <= 100`
`strs[i]` consists of lower-case English letters.

### Solution 
 To solve this problem, we need to group anagrams together from a given list of strings. We can utilize a dictionary to help group strings that, when sorted, look the same.

Here's the general approach:

1. Initialize an empty dictionary where the keys will be the sorted version of the string (which represent anagrams), and the values will be the list of strings that are anagrams of each other.
2. Iterate over the list of strings. For each string:
   - Sort the characters in the string to form a key.
   - If the key is already present in the dictionary, append the string to the corresponding list.
   - If the key is not present, add a new key with the current string in a new list.
3. After processing all strings, the values in the dictionary represent the groups of anagrams.

Below is the Python code following the specified format that can be run directly on LeetCode:



```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        from collections import defaultdict
        
        # Create a dictionary to hold lists of anagrams
        anagrams = defaultdict(list)
        
        # Iterate over each string in the input list
        for s in strs:
            # Sort the string and use it as a key
            sorted_s = ''.join(sorted(s))
            # Append the original string under the sorted key
            anagrams[sorted_s].append(s)
        
        # Return values of the dictionary as a list of lists
        return list(anagrams.values())

# The above class and method can be used in LeetCode to solve the problem with given examples.

```

This solution is clear and efficient for the problem's constraints, as sorting each string has a time complexity of O(k log k), where `k` is the maximum length of a string in `strs`. The outer loop runs for every string, leading to an overall complexity of O(n * k log k), where `n` is the number of strings in `strs`. This is acceptable given the provided constraints.

# 76. Minimum Window Substring

### Problem Description 
Given two strings `s` and `t`, return the minimum window in `s` which will contain all the characters in `t`. If there is no such window in `s` that covers all characters in `t`, return the empty string `""`.

Note that If there is such a window, it is guaranteed that there will always be only one unique minimum window in `s`.


Example 1:
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"

Example 2:
Input: s = "a", t = "a"
Output: "a"

Constraints:
`1 <= s.length, t.length <= 105`
`s` and `t` consist of English letters.

Follow up: Could you find an algorithm that runs in `O(n)` time?

### Solution 
 The problem at hand is to find the minimum window in string `s` that contains all the characters of string `t`. This is a classic example of the "Minimum Window Substring" problem, which can be efficiently solved using the sliding window technique combined with hashmap. Our goal is to make our solution efficient, ideally aiming for `O(n)` complexity.

Here’s how you can understand and solve the problem:

1. **Initialize two hashmaps (dictionaries in Python)**:
    - One (let’s call it `need`) to store counts of each character we need from `t`.
    - Another (let’s call it `window`) to store counts of each character as we traverse `s`.

2. **Use two pointers** to create a sliding window:
    - `left` and `right` initially point to the beginning of `s`.
    - Expand the `right` pointer to enlarge the window until it contains all the necessary characters from `t`.

3. **Once the window contains all the characters**, try to shrink it by moving the `left` pointer to the right. The goal is to find the smallest such window.

4. **Keep track of the size and position of the minimum window** that satisfies the condition.

5. **Edge cases**: If `t` is longer than `s`, or if `t` has characters that `s` does not have, return an empty string `""`.

Below is the Python code that implements these thoughts:



```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        from collections import Counter
        
        if not s or not t:
            return ""
        
        # Dictionary to count characters in t that need to be present in the window.
        need = Counter(t)
        # Dictionary to count all characters in the current window.
        window = {}
        
        # Initialize the left and right pointers.
        left = 0
        right = 0
        
        # Valid keeps track of how many characters from t have been covered in the current window.
        valid = 0
        
        # Variables to keep track of the smallest window.
        min_len = float('inf')
        min_left = 0
        
        # Start sliding the window.
        while right < len(s):
            # Character that goes into the window.
            c = s[right]
            right += 1
            
            # If it’s one of the required characters, increment the count in the window.
            if c in need:
                window[c] = window.get(c, 0) + 1
                if window[c] == need[c]:
                    valid += 1
            
            # Try to shrink the window as small as possible while it still contains all the characters from t.
            while valid == len(need):
                if right - left < min_len:
                    min_len = right - left
                    min_left = left
                
                # Character to be removed from the window.
                d = s[left]
                left += 1
                
                if d in need:
                    if window[d] == need[d]:
                        valid -= 1
                    window[d] -= 1
        
        # If min_len is still infinity that means no valid window was found.
        return "" if min_len == float('inf') else s[min_left:min_left + min_len]

# Example usage:
sol = Solution()
print(sol.minWindow("ADOBECODEBANC", "ABC"))  # Output: "BANC"
print(sol.minWindow("a", "a"))                # Output: "a"

```

This code initializes two dictionaries, expands the window by moving `right`, and shrinks it by moving `left`. We keep updating our record of the minimum window, and finally return the smallest valid window that contains all characters of `t`.

# 85. Maximal Rectangle

### Problem Description 
Given a `rows x cols` binary `matrix` filled with `0`'s and `1`'s, find the largest rectangle containing only `1`'s and return its area.


Example 1:
Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 6
Explanation: The maximal rectangle is shown in the above picture.


Example 2:
Input: matrix = []
Output: 0

Example 3:
Input: matrix = [["0"]]
Output: 0

Example 4:
Input: matrix = [["1"]]
Output: 1

Example 5:
Input: matrix = [["0","0"]]
Output: 0

Constraints:
`rows == matrix.length`
`cols == matrix[i].length`
`0 <= row, cols <= 200`
`matrix[i][j]` is `'0'` or `'1'`.

### Solution 
 The problem asks for the maximum rectangular area within a binary matrix consisting of '0's and '1's. This is a common dynamic programming question frequently encountered in coding interviews. A similar problem is "Largest Rectangle in Histogram."

To solve this problem, an effective strategy involves reducing the 2D matrix problem to a series of 1D problems. Specifically, each row in the matrix can be treated like a histogram where the height of each histogram's bar is the number of consecutive '1's seen up to that row. We can then apply a "Largest Rectangle in Histogram" style solution to each of these histograms to determine the largest rectangle for that specific arrangement.

Here's step-by-step implementation in Python, following the requested format for LeetCode solutions:

1. **Initialize**: If the matrix is empty, return an area of 0.
2. **Transform**: Convert each row into a histogram representation where each element denotes the sum of '1's vertically up to that row.
3. **Compute Largest Rectangle by Row**: Use a stack-based approach to compute the largest rectangle that fits in each histogram (i.e., each row of the updated matrix).

Below is a Python code implementing this solution:



```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0

        max_area = 0
        rows, cols = len(matrix), len(matrix[0])
        histogram = [0] * cols  # This is where we will store our histogram heights.
        
        for i in range(rows):
            for j in range(cols):
                # Update the height to count consecutive '1's.
                if matrix[i][j] == '0':
                    histogram[j] = 0
                else:
                    histogram[j] += 1

            # After updating histogram for the row, calculate the max rectangle area
            # using the histogram (treat it as "largest rectangle in histogram" problem).
            max_area = max(max_area, self.largestRectangleArea(histogram))
        
        return max_area

    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        max_area = 0
        heights.append(0)  # Append a zero height to flush out remaining heights in the stack.

        for i in range(len(heights)):
            while stack and heights[stack[-1]] > heights[i]:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(i)
        
        heights.pop()  # Remove the appended zero-height.
        return max_area

```

Explanation:
- **Initialize and Input Handling**: Check if the input matrix is empty or its first row is empty, and return 0 for these cases.
- **Building Histograms**: Update histogram heights based on '1's continuity vertically.
- **Largest Rectangle Area Calculation for Each Row**: This is achieved by treating each modified row (histogram) as a standalone problem of the largest rectangle in histogram, which is solved using a stack.
- **Stack-based Efficiency**: the `largestRectangleArea` function efficiently handles calculation of maximum area using a stack to track the starting indices of potential rectangles and their heights.

With this code, you can paste directly into the solution box in LeetCode for the "Maximal Rectangle" problem. It will execute correctly and efficiently, given the problem's constraints.

# 94. Binary Tree Inorder Traversal

### Problem Description 
Given the `root` of a binary tree, return the inorder traversal of its nodes' values.


Example 1:
Input: root = [1,null,2,3]
Output: [1,3,2]

Example 2:
Input: root = []
Output: []

Example 3:
Input: root = [1]
Output: [1]

Example 4:
Input: root = [1,2]
Output: [2,1]

Example 5:
Input: root = [1,null,2]
Output: [1,2]

Constraints:
The number of nodes in the tree is in the range `[0, 100]`.

`-100 <= Node.val <= 100`
Follow up:
Recursive solution is trivial, could you do it iteratively?

### Solution 
 Here is a Python solution for the given problem that includes both recursive and iterative approaches for conducting an inorder traversal of a binary tree. The code is formatted to be compatible with LeetCode's submission system. Each method is provided within the `Solution` class, and you can choose to run either the recursive or the iterative method.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        # To switch between recursive and iterative solutions, 
        # you can uncomment the desired approach.
        
        # Recursive method call
        # return self.inorderTraversalRecursive(root)
        
        # Iterative method call
        return self.inorderTraversalIterative(root)
    
    def inorderTraversalRecursive(self, root: TreeNode) -> List[int]:
        res = []
        self.helper(root, res)
        return res
    
    def helper(self, node: TreeNode, res: List[int]):
        if not node:
            return
        self.helper(node.left, res)  # Traverse left subtree
        res.append(node.val)  # Visit node
        self.helper(node.right, res)  # Traverse right subtree
    
    def inorderTraversalIterative(self, root: TreeNode) -> List[int]:
        res, stack = [], []
        current = root
        while current or stack:
            while current:
                stack.append(current)
                current = current.left  # Go to the leftmost node
            current = stack.pop()
            res.append(current.val)  # Process the node
            current = current.right  # Now, traverse the right subtree
        return res

# Example usage:
# Construct the tree: root = [1, null, 2, 3]
# root = TreeNode(1)
# root.right = TreeNode(2)
# root.right.left = TreeNode(3)
# sol = Solution()
# print(sol.inorderTraversal(root))  # Output: [1, 3, 2]

```

### Explanation

1. **Recursive Approach**:
   - The `inorderTraversalRecursive` function initializes an empty list `res` to store the results.
   - The helper function `helper` is a recursive function that traverses the tree. It first goes to the leftmost child, visits the node (appends node's value to `res`), and then explores the right child. This maintains the inorder traversal order: left, root, right.
   - Base case: If the node is `None`, return immediately.

2. **Iterative Approach**:
   - This approach uses a stack to simulate the system's call stack used in recursion.
   - It progressively pushes left children of the tree nodes onto the stack until it can't go left anymore.
   - When left children are exhausted, it pops from the stack (which returns nodes in left->root traversal order), processes the node's value, and then moves to the right child.
   - This continues until both the stack is empty and there are no more right children to process.

You will need to uncomment the method call in `inorderTraversal` based on whether you want to use recursion or iteration. The recursive approach is generally simpler to understand and implement due to direct application of the tree traversal definition, while the iterative approach avoids the overhead of recursion and uses explicit stack management, which might be useful especially for deep trees to avoid recursion depth limits.

# 136. Single Number

### Problem Description 
Given a non-empty array of integers `nums`, every element appears twice except for one. Find that single one.

Follow up: Could you implement a solution with a linear runtime complexity and without using extra memory?

Example 1:
Input: nums = [2,2,1]
Output: 1

Example 2:
Input: nums = [4,1,2,1,2]
Output: 4

Example 3:
Input: nums = [1]
Output: 1

Constraints:
`1 <= nums.length <= 3 * 104`
`-3 * 104 <= nums[i] <= 3 * 104`
Each element in the array appears twice except for one element which appears only once.

### Solution 
 Here is a Python solution that meets the constraints of the problem efficiently using the properties of bitwise XOR, which is both linear in time complexity and requires no extra storage (constant space complexity). The solution follows the LeetCode format and can be submitted directly on the platform.

### Python Code:



```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        """
        The function utilizes the XOR operation to find the single number
        in the array. The XOR operation has a few critical properties:
        - a ^ a = 0 (any number XORed with itself equals 0)
        - a ^ 0 = a (any number XORed with 0 equals the number itself)
        - XOR is commutative and associative, which means the order of
          operations doesn't matter.
        
        By XORing all elements in the array, pairs of identical numbers
        will cancel each other out (become 0), leaving only the number
        that doesn't have a pair.
        
        :type nums: List[int]
        :rtype: int
        """
        result = 0
        for num in nums:
            result ^= num
        return result


```

### Explanation:

1. **Initialization**:
   - We start with a variable `result` initialized to `0`.

2. **Iterate through `nums`**:
   - For every number `num` in the array `nums`, we apply the XOR operation with the `result`. Note that XOR-ing any number with `0` will return the number itself, and XOR-ing any number with itself will return `0`.
   
3. **Result**:
   - Since every number except one appears twice in the array, their contributions to `result` will cancel out each other because `a ^ a = 0`.
   - The only number that does not get cancelled is the number that appears once, which will be the final value stored in `result`.

4. **Return**:
   - At the conclusion of the loop, `result` will hold the single number that doesn't have a duplicate in the array, and it is returned.

This simple yet effective use of XOR meets the problem's requirements of linear runtime complexity \(O(n)\) and constant space complexity \(O(1)\). The approach cleverly uses a mathematical operation to avoid the need for additional storage, making it extremely efficient for this problem.

# 138. Copy List with Random Pointer

### Problem Description 
A linked list of length `n` is given such that each node contains an additional random pointer, which could point to any node in the list, or `null`.

Construct a deep copy of the list. The deep copy should consist of exactly `n` brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the `next` and `random` pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.

For example, if there are two nodes `X` and `Y` in the original list, where `X.random --> Y`, then for the corresponding two nodes `x` and `y` in the copied list, `x.random --> y`.

Return the head of the copied linked list.

The linked list is represented in the input/output as a list of `n` nodes. Each node is represented as a pair of `[val, random_index]` where:
`val`: an integer representing `Node.val`
`random_index`: the index of the node (range from `0` to `n-1`) that the `random` pointer points to, or `null` if it does not point to any node.

Your code will only be given the `head` of the original linked list.


Example 1:
Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]

Example 2:
Input: head = [[1,1],[2,1]]
Output: [[1,1],[2,1]]

Example 3:
Input: head = [[3,null],[3,0],[3,null]]
Output: [[3,null],[3,0],[3,null]]

Example 4:
Input: head = []
Output: []
Explanation: The given linked list is empty (null pointer), so return null.


Constraints:
`0 <= n <= 1000`
`-10000 <= Node.val <= 10000`
`Node.random` is `null` or is pointing to some node in the linked list.

### Solution 
 To solve the problem of creating a deep copy of a linked list where each node has a `random` pointer, we will use a straightforward approach involving a dictionary to map original nodes to their corresponding copies. This helps in maintaining the complex `random` pointers accurately while copying.

The steps involved are:
1. Traverse the original linked list and create a mapping of original nodes to their corresponding new copies with the help of a dictionary.
2. On this traversal, set the `next` pointers of the new nodes.
3. In another traversal, set the `random` pointers for each new node by using the dictionary to find the corresponding random nodes in the new linked list.
4. Return the head of the new list which is pointed by the dictionary's value for the original head.

Here is the solution in Python which will be compatible with LeetCode's execution environment:



```python
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return None
        
        # Dictionary to hold the original node to its copy
        old_to_new = {}
        
        # First pass: Create a copy of each node and link the next pointers
        current = head
        while current:
            copy = Node(current.val)
            old_to_new[current] = copy
            current = current.next
        
        # Second pass: Assign the random and next pointers for the copied nodes
        current = head
        while current:
            if current.next:
                old_to_new[current].next = old_to_new[current.next]
            if current.random:
                old_to_new[current].random = old_to_new[current.random]
            current = current.next
        
        # Return the new list's head which is the copy of the original list's head
        return old_to_new[head]

# Example usage
# This portion is not necessary for LeetCode submission, added for explanatory purpose
if __name__ == '__main__':
    # Create list similar to Example 1 manually
    node1 = Node(7)
    node2 = Node(13)
    node3 = Node(11)
    node4 = Node(10)
    node5 = Node(1)
    
    node1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node5
    
    node2.random = node1
    node3.random = node5
    node4.random = node3
    node5.random = node1
    
    sol = Solution()
    copied_head = sol.copyRandomList(node1)
    
    # Node to demonstrate copied structure
    curr = copied_head
    while curr:
        random_val = curr.random.val if curr.random else "null"
        print(f'[{curr.val}, {random_val}]', end=" -> ")
        curr = curr.next
    print("None")

```

This example demonstrates the defined classes and methods, along with an example usage to verify that the linked list is copied correctly, including the intricate `random` pointers. For LeetCode, only the `Node` class and `Solution` class defintions are necessary to be submitted, along with the `copyRandomList` method.

# 149. Max Points on a Line

### Problem Description 
Given an array of `points` where `points[i] = [xi, yi]` represents a point on the X-Y plane, return the maximum number of points that lie on the same straight line.


Example 1:
Input: points = [[1,1],[2,2],[3,3]]
Output: 3

Example 2:
Input: points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
Output: 4

Constraints:
`1 <= points.length <= 300`
`points[i].length == 2`
`-104 <= xi, yi <= 104`
All the `points` are unique.

### Solution 
 To solve this problem, the fundamental objective is to determine the maximum number of collinear points by checking each pair of points to see if they lie on the same line.

Here's a step-by-step strategy:
1. For each point, compute the slope it makes with every other point. Points on the same line with respect to the reference point will have the same slope.
2. Utilize a hash map to record counts of how many points achieve a given slope with respect to the fixed reference point.
3. Since the slope (which is a division result) can lead to floating-point precision issues, we'll internally use fractions (like numerator/denominator) after simplifying them by their greatest common divisor (GCD). This bypasses the floating-point inconsistency.
4. For each reference point, determine the maximum count of points that share a slope (i.e., the largest value in the hash map for that specific point accounting for vertical lines specially).
5. Track global maximum number of points that were collinear during the iterations.

In the provided solution, we'll use Python's `math.gcd` function for reducing fractions and the `collections.defaultdict` for an efficient hash map implementation.



```python
from collections import defaultdict
import math

class Solution:
    def maxPoints(self, points):
        def max_points_on_a_line_containing_point_i(i):
            """
            Compute the maximum number of points for a line containing point i.
            """
            def slope_coprime(x1, y1, x2, y2):
                """
                Return the slope of the line between two points (x1, y1) and (x2, y2),
                reduced to lowest terms, represented as (dy, dx).
                """
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0:  # vertical line
                    return (0, 1)
                if dy == 0:  # horizontal line
                    return (1, 0)
                gcd = math.gcd(dx, dy)
                dx //= gcd
                dy //= gcd
                # To ensure a consistent representation of the slope
                # (avoid (-1, 2) and (1, -2) being treated differently)
                if dx < 0:
                    dx, dy = -dx, -dy
                return (dy, dx)

            lines = defaultdict(int)
            max_points = 1  # at least the point itself
            # Compute lines for each point j relative to point i
            for j in range(len(points)):
                if j != i:
                    slope = slope_coprime(points[i][0], points[i][1], points[j][0], points[j][1])
                    lines[slope] += 1
                    max_points = max(max_points, lines[slope] + 1)

            return max_points

        # If there's one point or none, all points are on the same line
        n = len(points)
        if n < 3:
            return n

        max_result = 1
        # Check every point as a possible point on a line
        for i in range(n - 1):  # no need to check the last point
            max_result = max(max_result, max_points_on_a_line_containing_point_i(i))

        return max_result

# Example usage:
sol = Solution()
print(sol.maxPoints([[1,1],[2,2],[3,3]]))  # Output: 3
print(sol.maxPoints([[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]))  # Output: 4

```

This solution iterates over each point, treating it as the origin, and then computes the slopes of lines to every other point, keeping track in a dictionary. The greatest number of points on the same line for each reference point is computed, and the maximum of these is returned. Special attention is given to vertical and horizontal lines, as well as handling potential negative slopes in a normalized way.

# 159. Longest Substring with At Most Two Distinct Characters

### Problem Description 
Given a string `s`, return the length of the longest substring that contains at most two distinct characters.


Example 1:
Input: s = "eceba"
Output: 3
Explanation: The substring is "ece" which its length is 3.


Example 2:
Input: s = "ccaabbb"
Output: 5
Explanation: The substring is "aabbb" which its length is 5.


Constraints:
`1 <= s.length <= 104`
`s` consists of English letters.

### Solution 
 To solve the problem of finding the longest substring that contains at most two distinct characters from a given string `s`, we can use the sliding window technique with a hashtable (or dictionary in Python) to keep track of the frequency of each character within the window.

The sliding window technique involves two pointers (`left` and `right`) that represent the window's boundaries. The `left` pointer indicates the start of the substring and the `right` pointer will be used to expand the window. As the `right` pointer moves to the right, we update our hashtable with the frequency of the character at the position of the `right` pointer.

If the window contains more than two distinct characters, the `left` pointer needs to be moved right until the window has at most two distinct characters again. During this process, we will decrease the character count in the hashtable, and possibly remove the character entirely if its count drops to zero.

We will also keep a variable `max_len` to keep track of the maximum length of valid window encountered so far.

Here is a complete Python solution:



```python
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        from collections import defaultdict

        n = len(s)
        if n < 3:
            return n

        # Dictionary to store the count of each character in the current window
        char_count = defaultdict(int)

        left = 0
        max_len = 2

        # Using the sliding window technique, starting the window's right bound
        for right in range(n):
            char_count[s[right]] += 1  # Include new character in the window

            # If the window contains more than two distinct characters
            while len(char_count) > 2:
                char_count[s[left]] -= 1  # Reduce the count of the left character
                if char_count[s[left]] == 0:
                    del char_count[s[left]]  # Remove character count if it's zero
                left += 1  # Shrink the window from the left

            # Update the max_len if the current window is larger
            max_len = max(max_len, right - left + 1)

        return max_len

# Example usage:
sol = Solution()
print(sol.lengthOfLongestSubstringTwoDistinct("eceba"))  # Output: 3
print(sol.lengthOfLongestSubstringTwoDistinct("ccaabbb"))  # Output: 5

```

This code adheres to LeetCode's format and should be directly usable for submitting on LeetCode. The use of a defaultdict simplifies character counting and ensures that we efficiently manage the character frequency map while expanding and contracting the sliding window.

# 166. Fraction to Recurring Decimal

### Problem Description 
Given two integers representing the `numerator` and `denominator` of a fraction, return the fraction in string format.

If the fractional part is repeating, enclose the repeating part in parentheses.

If multiple answers are possible, return any of them.

It is guaranteed that the length of the answer string is less than `104` for all the given inputs.


Example 1:
Input: numerator = 1, denominator = 2
Output: "0.5"

Example 2:
Input: numerator = 2, denominator = 1
Output: "2"

Example 3:
Input: numerator = 2, denominator = 3
Output: "0.(6)"

Example 4:
Input: numerator = 4, denominator = 333
Output: "0.(012)"

Example 5:
Input: numerator = 1, denominator = 5
Output: "0.2"

Constraints:
`-231 <= numerator, denominator <= 231 - 1`
`denominator != 0`

### Solution 
 To solve the problem of converting a fraction given by its numerator and denominator into a string, including handling repeating decimals, we need to follow these steps:

1. **Handling Sign of the Result**: Determine the sign of the fraction based on the signs of the numerator and denominator to appropriately prefix the result with a minus sign if needed.

2. **Integer Part Calculation**: Divide the absolute values of the numerator by the denominator to get the integer part of the result.

3. **Finding Remainders**: Use the modulus operation to find the remainder which will help in generating the fractional part. If the remainder is zero immediately after getting the integer part, return the result as it doesn't have a fractional part.

4. **Handling Repeating Decimal Part**:
   - Keep track of remainders and the positions at which they appear during the division process to detect cycles (loops) that signify repeating decimals.
   - Use a dictionary to map each remainder to its position in the decimal result sequence.
   - For each remainder, compute the next digit of the result by multiplying the remainder by 10 and dividing by the denominator, and then calculate the new remainder.
   - If a remainder repeats, it indicates the start of a repeating decimal.

5. **Building the Fractional String**:
   - Append each computed digit to the result.
   - When a repeating sequence is detected, insert parentheses around the repeating part.

By following these detailed steps, we properly handle all the peculiarities of division including zero, non-repeating, and repeating decimals.



```python
class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        if numerator == 0:
            return "0"
        
        result = []
        
        # Determine the sign of the result based on numerator and denominator
        if (numerator > 0) ^ (denominator > 0):
            result.append('-')
        
        # Work with absolute values for division and modulus operations
        num = abs(numerator)
        den = abs(denominator)
        
        # Append the integer part
        result.append(str(num // den))
        remainder = num % den
        
        # If there is no remainder, return the integer part only
        if remainder == 0:
            return ''.join(result)
        
        # Prepare for the fractional part
        result.append('.')
        remainder_map = {}
        decimal_part = []
        
        # Track the location of each remainder to identify repeating fractions
        while remainder != 0:
            if remainder in remainder_map:
                # We found a repeating remainder
                # Insert parentheses around the repeat part
                repeat_index = remainder_map[remainder]
                decimal_part.insert(repeat_index, '(')
                decimal_part.append(')')
                break
            
            # Record the position of each remainder
            remainder_map[remainder] = len(decimal_part)
            
            # Calculate the next digit
            remainder *= 10
            decimal_part.append(str(remainder // den))
            remainder = remainder % den
        
        # Join all parts to form the final result
        result.extend(decimal_part)
        return ''.join(result)


```

# 170. Two Sum III - Data structure design

### Problem Description 
Design a data structure that accepts a stream of integers and checks if it has a pair of integers that sum up to a particular value.

Implement the `TwoSum` class:
`TwoSum()` Initializes the `TwoSum` object, with an empty array initially.

`void add(int number)` Adds `number` to the data structure.

`boolean find(int value)` Returns `true` if there exists any pair of numbers whose sum is equal to `value`, otherwise, it returns `false`.


Example 1:
Input
["TwoSum", "add", "add", "add", "find", "find"]
[[], [1], [3], [5], [4], [7]]
Output
[null, null, null, null, true, false]
Explanation
TwoSum twoSum = new TwoSum();
twoSum.add(1);   // [] --> [1]
twoSum.add(3);   // [1] --> [1,3]
twoSum.add(5);   // [1,3] --> [1,3,5]
twoSum.find(4);  // 1 + 3 = 4, return true
twoSum.find(7);  // No two integers sum up to 7, return false

Constraints:
`-105 <= number <= 105`
`-231 <= value <= 231 - 1`
At most `5 * 104` calls will be made to `add` and `find`.

### Solution 
 To address this problem, the best approach is to leverage a hash map (dictionary in Python) due to its average O(1) time complexity for look-up operations. The main idea here is to maintain a count of each number that we add to the data structure. This helps in efficiently checking if a complement for the desired sum exists without exhaustively searching through the list.

Here's how we can build the class:

1. Use a dictionary to store numbers and their counts.
2. When `add` is called, increase the count of the number in the dictionary.
3. When `find` is called, iterate through each unique number in the dictionary and calculate its complement (the number that needs to be paired with the current number to get the specified sum). Check if the complement exists in the dictionary:
   - If a number and its complement are the same, check if the count is more than one because you need at least two of the same number to get a sum.
   - If they are different, simply check if the complement exists.

The dictionary-based approach ensures that the add operation remains O(1) on average, while the find operation is O(N) on average where N is the number of unique numbers added, due to the need to potentially check each entry in the dictionary.

Here is the Python implementation for this approach:



```python
class TwoSum:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.num_count = {}

    def add(self, number: int) -> None:
        """
        Add the number to an internal data structure.
        """
        if number in self.num_count:
            self.num_count[number] += 1
        else:
            self.num_count[number] = 1

    def find(self, value: int) -> bool:
        """
        Find if there exists any pair of numbers which sum is equal to the value.
        """
        num_count = self.num_count
        for num in num_count:
            complement = value - num
            if complement in num_count:
                if num != complement or num_count[num] > 1:
                    return True
        return False

# Your TwoSum object will be instantiated and called as such:
# obj = TwoSum()
# obj.add(number)
# param_2 = obj.find(value)

```

Explanation:
- The `__init__` method initializes an empty dictionary `num_count` to keep track of the counts of numbers added.
- The `add` method increments the counter of the given number in the dictionary.
- The `find` method checks for each key in `num_count` if there is a complementary key (`value - num`) in the dictionary to form a pair with sum equal to `value`. If the pair involves two of the same number, we check that this number has been added at least twice.

This implementation adheres to the constraints and efficiently handles the operations as required by the problem statement.

# 187. Repeated DNA Sequences

### Problem Description 
The DNA sequence is composed of a series of nucleotides abbreviated as `'A'`, `'C'`, `'G'`, and `'T'`.

For example, `"ACGAATTCCG"` is a DNA sequence.

When studying DNA, it is useful to identify repeated sequences within the DNA.

Given a string `s` that represents a DNA sequence, return all the `10`-letter-long sequences (substrings) that occur more than once in a DNA molecule. You may return the answer in any order.


Example 1:
Input: s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
Output: ["AAAAACCCCC","CCCCCAAAAA"]

Example 2:
Input: s = "AAAAAAAAAAAAA"
Output: ["AAAAAAAAAA"]

Constraints:
`1 <= s.length <= 105`
`s[i]` is either `'A'`, `'C'`, `'G'`, or `'T'`.

### Solution 
 Below is a Python code to solve the given problem, adhering to the standards and format usually seen in LeetCode submissions. The code uses a straightforward approach with a hash table (dictionary in Python) to keep track of occurrences of each 10-letter-long DNA sequence. It then collects those sequences that appear more than once.



```python
class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        # Length of the DNA sequence to check
        L = 10
        # This dictionary will map sequences to their counts
        counts = {}
        # This list will store sequences that occur more than once
        repeated = []

        # Iterate over the possible starting points of 10-letter sequences
        for start in range(len(s) - L + 1):
            # Extract the 10-letter sequence starting from 'start'
            sub_seq = s[start:start + L]
            # If this sequence is already in the dictionary, increment its count
            if sub_seq in counts:
                counts[sub_seq] += 1
            else:
                # Otherwise, add it to the dictionary with count 1
                counts[sub_seq] = 1

        # Collect the sequences that occur more than once
        for seq in counts:
            if counts[seq] > 1:
                repeated.append(seq)

        return repeated

# The code can be then used as follows:
# Example instantiation and method call for a test case provided by leetcode.
sol = Solution()
output = sol.findRepeatedDnaSequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT")
print(output)

```

### Explanation

1. **Initialization**: We set `L` to `10` as we need to search for 10-letter sequences. We prepare a dictionary `counts` to keep track of how often each sequence appears in the string `s`.

2. **Build Count Dictionary**:
   - Loop through the string `s`, using a sliding window of length `L` to inspect each possible sequence of length `10`.
   - For each such 10-letter substring, we either initialize or update its count in our dictionary `counts`.

3. **Identify Repeated Sequences**:
   - Loop through the keys in `counts`. If the count of a sequence is more than `1`, we add it to our result list `repeated`.

4. **Return Result**: The list `repeated` containing all 10-letter sequences that appear more than once is returned.

This solution efficiently organizes the sequence counts and checks them with a linear time complexity. The overall complexity is O(n * k), where `n` is the length of the string `s` and `k` is the length of the sequence (which in this case is constant at 10, making it O(n)). This suits well within the problem's constraints allowing for fast checks even for large inputs.

# 202. Happy Number

### Problem Description 
Write an algorithm to determine if a number `n` is happy.

A happy number is a number defined by the following process:
Starting with any positive integer, replace the number by the sum of the squares of its digits.

Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.

Those numbers for which this process ends in 1 are happy.

Return `true` if `n` is a happy number, and `false` if not.


Example 1:
Input: n = 19
Output: true
Explanation:
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1

Example 2:
Input: n = 2
Output: false

Constraints:
`1 <= n <= 231 - 1`

### Solution 
 The given problem is a classic example of detecting cycles in an iterative process. The challenge is to identify whether the number `n` eventually becomes 1 (making it a "happy" number) or falls into an endless cycle that doesn't include 1. To implement this, we can use the Floyd's Cycle detection algorithm (tortoise and hare method) which efficiently detects cycles in a sequence of values.

Below is the detailed Python code adhering to the structure expected on the LeetCode platform:



```python
class Solution:
    def isHappy(self, n: int) -> bool:
        
        def get_next(number):
            total_sum = 0
            while number > 0:
                number, digit = divmod(number, 10)
                total_sum += digit ** 2
            return total_sum

        slow = n
        fast = get_next(n)
        
        # The loop stops when the `fast` pointer, which moves twice as fast as `slow`, meets `slow`.
        # This ensures there's a cycle.
        while fast != 1 and slow != fast:
            slow = get_next(slow)
            fast = get_next(get_next(fast))
        
        # If `fast` becomes 1, it means the number `n` is a happy number.
        return fast == 1

# Example usage:
sol = Solution()
print(sol.isHappy(19))  # Output: True
print(sol.isHappy(2))   # Output: False

```

### Explanation of the solution:
1. **Helper Function**: `get_next(number)` calculates the sum of the squares of its digits.
2. **Main Function**: The process starts with two pointers, `slow` and `fast`, both initialized with `n` and `get_next(n)` respectively. The `slow` pointer moves one step at a time, while the `fast` pointer moves two steps at a time (tortoise and hare approach).
3. **Cycle Detection**: If there's a cycle, the two pointers will meet at some point (i.e., `slow == fast`), and if they do meet and `fast` (or `slow`) is not 1, there's a loop not including 1, meaning `n` is not happy.
4. **End of Loop**: The loop ends if `fast` becomes 1, proving `n` is a happy number, or if `slow` meets `fast`, indicating a cycle. The function returns `True` if `fast` is 1, else `False`.

By leveraging the cycle detection method, this solution runs efficiently without having to use extra space for storing visited numbers, making it optimal in terms of time and space complexity.

# 204. Count Primes

### Problem Description 
Count the number of prime numbers less than a non-negative number, `n`.


Example 1:
Input: n = 10
Output: 4
Explanation: There are 4 prime numbers less than 10, they are 2, 3, 5, 7.


Example 2:
Input: n = 0
Output: 0

Example 3:
Input: n = 1
Output: 0

Constraints:
`0 <= n <= 5 * 106`

### Solution 
 To solve the problem of counting the number of prime numbers less than a given non-negative number `n`, we can use the Sieve of Eratosthenes algorithm. This is an efficient way to find all primes smaller than `n`.

## Explanation:
The Sieve of Eratosthenes works by iteratively marking the multiples of each prime number starting from 2. The multiples are not prime, so marking them will help in identifying the prime numbers. Initially, every number in the list is considered as a potential prime. 

Here's a step-by-step breakdown:
1. Create a list that will hold the boolean values to mark whether a number is prime or not.
2. Iterate over each number and if it's marked as a prime, iterate through its multiples and mark them as non-prime.
3. Count the numbers marked as prime.

## Implementation:
The solution will be implemented as a class with a static method, which is typical for Python solutions on LeetCode. This allows the function to be tested with different inputs easily.



```python
class Solution:
    def countPrimes(self, n: int) -> int:
        if n <= 2:
            return 0
        
        # This will hold whether a particular number is prime or not
        is_prime = [True] * n
        is_prime[0] = is_prime[1] = False  # We know 0 and 1 are not primes
        
        p = 2
        while p * p < n:
            # If is_prime[p] is not changed, then it is a prime
            if is_prime[p]:
                # Marking all multiples of p from p*p to less than n as not prime
                for i in range(p * p, n, p):
                    is_prime[i] = False
            p += 1
        
        # Counting all true values in the is_prime list
        return sum(is_prime)

# Example cases
solution = Solution()
print(solution.countPrimes(10))  # Output: 4
print(solution.countPrimes(0))   # Output: 0
print(solution.countPrimes(1))   # Output: 0

```

## Note:
- We start p from 2, the smallest prime number.
- The loop checks for primes until `p * p < n` as any factor that could be marked later is already marked by a smaller prime factor.
- We update the multiples starting from `p * p` because all smaller multiples of `p` will have already been marked by smaller primes.
- The time complexity of this algorithm is \(O(n \log \log n)\) which is efficient enough for `n` up to \(5 \times 10^6\) within reasonable time constraints.

# 205. Isomorphic Strings

### Problem Description 
Given two strings `s` and `t`, determine if they are isomorphic.

Two strings `s` and `t` are isomorphic if the characters in `s` can be replaced to get `t`.

All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character, but a character may map to itself.


Example 1:
Input: s = "egg", t = "add"
Output: true

Example 2:
Input: s = "foo", t = "bar"
Output: false

Example 3:
Input: s = "paper", t = "title"
Output: true

Constraints:
`1 <= s.length <= 5 * 104`
`t.length == s.length`
`s` and `t` consist of any valid ascii character.

### Solution 
 To determine if two given strings `s` and `t` are isomorphic, we need to ensure that each character in `s` maps consistently to a character in `t`. Furthermore, we must verify that one character in `s` does not map to multiple characters in `t` and vice versa.

We can achieve this by using two dictionaries to maintain a mapping from characters in `s` to `t` and vice versa. For each pair of corresponding characters from `s` and `t`, we check the following:
1. If the character from `s` is already mapped to a different character in `t`.
2. If the character from `t` is already mapped from a different character in `s`.

If any of these conditions fail at any point, the strings are not isomorphic.

The constraints are such that a linear time complexity solution is acceptable, and our approach provides that since we only traverse each string once.

Below is the implementation of the solution adhering to the typical format suitable for running directly on LeetCode:



```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        
        char_map_st = {}
        char_map_ts = {}
        
        for cs, ct in zip(s, t):
            if cs in char_map_st:
                if char_map_st[cs] != ct:
                    return False
            else:
                char_map_st[cs] = ct
            
            if ct in char_map_ts:
                if char_map_ts[ct] != cs:
                    return False
            else:
                char_map_ts[ct] = cs
                
        return True

# Example usage:
sol = Solution()
print(sol.isIsomorphic("egg", "add"))  # Output: True
print(sol.isIsomorphic("foo", "bar"))  # Output: False
print(sol.isIsomorphic("paper", "title"))  # Output: True

```

This code defines a `Solution` class with a method `isIsomorphic`, which returns `True` if the strings are isomorphic and `False` otherwise. The use of two dictionaries ensures that the character mappings are consistent in both directions.

# 217. Contains Duplicate

### Problem Description 
Given an integer array `nums`, return `true` if any value appears at least twice in the array, and return `false` if every element is distinct.


Example 1:
Input: nums = [1,2,3,1]
Output: true

Example 2:
Input: nums = [1,2,3,4]
Output: false

Example 3:
Input: nums = [1,1,1,3,3,4,3,2,4,2]
Output: true

Constraints:
`1 <= nums.length <= 105`
`-109 <= nums[i] <= 109`

### Solution 
 To solve the given problem of determining if any value in an integer array appears at least twice, we can use a couple of efficient approaches:

1. **Using a Set**:
   - As we iterate over each element in the array, we can add each element into a set.
   - Before adding, if the element is already present in the set, it means the element has appeared before, and we return `true`.
   - If we finish looping through the array and no duplicate is found, we return `false`.
   - This approach is efficient with an average time complexity of O(N) where N is the number of elements in the array, as set operations (addition and check for existence) are averagely O(1).
   - The space complexity is O(N) in the worst case when there are no duplicates, as all elements will be added to the set.

2. **Sorting the Array**:
   - By sorting the array first, all duplicates will be adjacent to each other.
   - After sorting, we can simply check each adjacent pair of elements in one pass to see if they are identical.
   - The time complexity for this approach would typically be dominated by the sorting operation, which is O(N log N).
   - The space complexity can be O(1) using an in-place sort, depending on the sorting algorithm used.

The first method using a set is generally faster for large arrays with early duplicates and does not require modifying the array as the sorting method does. Therefore, we will implement the solution using a set for optimal performance in diverse cases.

Here's the direct code for the solution in a LeetCode-friendly format:



```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False

```

With this solution structure, you can directly submit this code on LeetCode under the related problem by replacing `List` with the appropriate import from the typing module, i.e., add `from typing import List` at the beginning of your script. This solution directly follows the LeetCode format and can be used without any changes.

# 219. Contains Duplicate II

### Problem Description 
Given an integer array `nums` and an integer `k`, return `true` if there are two distinct indices `i` and `j` in the array such that `nums[i] == nums[j]` and `abs(i - j) <= k`.


Example 1:
Input: nums = [1,2,3,1], k = 3
Output: true

Example 2:
Input: nums = [1,0,1,1], k = 1
Output: true

Example 3:
Input: nums = [1,2,3,1,2,3], k = 2
Output: false

Constraints:
`1 <= nums.length <= 105`
`-109 <= nums[i] <= 109`
`0 <= k <= 105`

### Solution 
 To solve this problem of finding duplicates within a specific range in an array, you can use a hash table (dictionary in Python) to keep track of the values and their latest indices. The solution involves iterating over the array while maintaining a dictionary that maps numbers to their latest occurrence index. For each number, you can check whether this number has been seen before and whether the difference between the current index and its last occurrence index is less than or equal to `k`. Here's how you can implement it:



```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        # Dictionary to store the value and its latest index
        num_map = {}
        
        # Iterate through each number in the array
        for i, num in enumerate(nums):
            if num in num_map and i - num_map[num] <= k:
                # Found a duplicate within the range k
                return True
            # Update the dictionary with the latest index of the number
            num_map[num] = i
        
        # No duplicates found in the required range
        return False

# If you want to test the function on LeetCode, paste the class definition
# in the provided coding area. Here, the List typing is often included by default.

```

### Detailed Explanation:

- **Data Structure**: We use a dictionary `num_map` to remember the most recent index `i` of each number in the array.
- **Algorithm Process**:
  1. Iterate through the list `nums` using `enumerate` to access both the index `i` and value `num`.
  2. If `num` already exists in `num_map`, you check if the current index `i` minus the stored index is less than or equal to `k`.
  3. If the above condition is true, it means we've found two duplicates close enough (`<= k` distance), hence return `True`.
  4. For each iteration, always update the dictionary with the current index of `num`, replacing any previous index. This ensures the dictionary always holds the latest appearance.
- **Time Complexity**: This approach has a time complexity of O(n) as it processes each item in the list exactly once.
- **Space Complexity**: The space complexity is O(min(n, k)), where `n` is the size of the array. In the worst case (if all `nums` are unique), the dictionary could end up storing `k` different elements.

This solution is efficient and suitable for large datasets due to its linear time complexity. When running this on LeetCode or a similar platform, it should pass all test cases for efficiency and correctness.

# 242. Valid Anagram

### Problem Description 
Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`, and `false` otherwise.


Example 1:
Input: s = "anagram", t = "nagaram"
Output: true

Example 2:
Input: s = "rat", t = "car"
Output: false

Constraints:
`1 <= s.length, t.length <= 5 * 104`
`s` and `t` consist of lowercase English letters.

Follow up: What if the inputs contain Unicode characters? How would you adapt your solution to such a case?

### Solution 
 To determine if one string is an anagram of another, you need to verify that both strings contain the same characters in exactly the same frequency. Here are some potential approaches to solve this problem:

1. **Sorting Approach:** One straightforward method is to sort both strings and compare them. If they are equal after sorting, then they are anagrams.
2. **Character Count:** Another approach is to use a frequency count of characters. We can count the frequency of each character in `s` and then subtract the counts based on characters in `t`. If all counts are zero at the end, the strings are anagrams.

We will choose the character count approach since its time complexity is linear, \(O(n)\), compared to \(O(n \log n)\) for sorting, where \(n\) is the length of the string.

Here's how the character count solution can be implemented, compatible with both ASCII and Unicode characters:



```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        # Early exit if string lengths are different
        if len(s) != len(t):
            return False
        
        # Create a frequency dictionary for characters in s
        char_count = {}
        for char in s:
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
        
        # Subtract the frequency based on characters in t
        for char in t:
            if char in char_count:
                if char_count[char] == 0:
                    return False  # More of 'char' in t than in s
                char_count[char] -= 1
            else:
                return False  # 'char' is in t but not in s
        
        return True  # All character counts are balanced

# Examples
sol = Solution()
print(sol.isAnagram("anagram", "nagaram"))  # Output: True
print(sol.isAnagram("rat", "car"))  # Output: False

```

### Follow-Up: Handling Unicode Characters
The solution provided already handles Unicode characters effectively due to Python's native string handling. The dictionary `char_count` maps any character (including Unicode) to its frequency, and the algorithm remains largely unchanged irrespective of character encoding. This approach ensures that no matter the type of characters contained in `s` and `t`, the program checks their frequencies accurately.

# 244. Shortest Word Distance II

### Problem Description 
Design a data structure that will be initialized with a string array, and then it should answer queries of the shortest distance between two different strings from the array.

Implement the `WordDistance` class:
`WordDistance(String[] wordsDict)` initializes the object with the strings array `wordsDict`.

`int shortest(String word1, String word2)` returns the shortest distance between `word1` and `word2` in the array `wordsDict`.


Example 1:
Input
["WordDistance", "shortest", "shortest"]
[[["practice", "makes", "perfect", "coding", "makes"]], ["coding", "practice"], ["makes", "coding"]]
Output
[null, 3, 1]
Explanation
WordDistance wordDistance = new WordDistance(["practice", "makes", "perfect", "coding", "makes"]);
wordDistance.shortest("coding", "practice"); // return 3
wordDistance.shortest("makes", "coding");    // return 1

Constraints:
`1 <= wordsDict.length <= 3 * 104`
`1 <= wordsDict[i].length <= 10`
`wordsDict[i]` consists of lowercase English letters.

`word1` and `word2` are in `wordsDict`.

`word1 != word2`
At most `5000` calls will be made to `shortest`.

### Solution 
 Below is a Python solution to the problem described, formatted and explained as required for direct use on LeetCode. The solution includes a class `WordDistance` that efficiently computes the shortest distance between two words by storing their indices during initialization. This reduces the time complexity of each query to O(N) where N is the number of occurrences of the less frequent word among the two queried words.



```python
class WordDistance:

    def __init__(self, wordsDict: List[str]):
        self.locations = {}
        # Record the locations of each word.
        for index, word in enumerate(wordsDict):
            if word not in self.locations:
                self.locations[word] = []
            self.locations[word].append(index)

    def shortest(self, word1: str, word2: str) -> int:
        loc1 = self.locations[word1]
        loc2 = self.locations[word2]
        # Initialize pointers for both lists of locations
        i, j = 0, 0
        min_distance = float('inf')
        # Compute the shortest distance by comparing positions
        while i < len(loc1) and j < len(loc2):
            min_distance = min(min_distance, abs(loc1[i] - loc2[j]))
            if loc1[i] < loc2[j]:
                i += 1
            else:
                j += 1
        return min_distance

# Your WordDistance object will be instantiated and called as such:
# obj = WordDistance(wordsDict)
# param_1 = obj.shortest(word1,word2)

```

### Explanation

The `WordDistance` class has two key parts:

1. **Initialization (`__init__` method):**
    - The constructor takes a list `wordsDict` of words and creates a dictionary `self.locations` that maps each word to all the indices where it appears in `wordsDict`.
    - This dictionary is filled by iterating over `wordsDict` with the enumeration index and appending the index to the list of locations for the encountered word.

2. **Querying Shortest Distance (`shortest` method):**
    - Given `word1` and `word2`, the method fetches their locations lists, `loc1` and `loc2`.
    - Then, it uses a two-pointer technique to find the smallest difference between any two indices in these lists. This is akin to merging two sorted arrays and helps ensure only a linear pass is required over the indices arrays.
    - The method iterates through both arrays and compares the current indices, moving the pointer that points to the smaller index to avoid unnecessary comparisons and to efficiently find the minimum distance.

### Complexity Analysis

For each call to `shortest`, if the number of occurrences of `word1` is `k1` and for `word2` is `k2`, the function finds the shortest distance in `O(min(k1, k2))`. Thus, given the space trade-off for storing indices (taking space complexity to `O(N)` where `N` is the length of `wordsDict`), this solution is optimal for scenarios where multiple calls to `shortest` might be made, which justifies storing indices upfront for efficiency in repeated query handling.

# 246. Strobogrammatic Number

### Problem Description 
Given a string `num` which represents an integer, return `true` if `num` is a strobogrammatic number.

A strobogrammatic number is a number that looks the same when rotated `180` degrees (looked at upside down).


Example 1:
Input: num = "69"
Output: true

Example 2:
Input: num = "88"
Output: true

Example 3:
Input: num = "962"
Output: false

Example 4:
Input: num = "1"
Output: true

Constraints:
`1 <= num.length <= 50`
`num` consists of only digits.

`num` does not contain any leading zeros except for zero itself.

### Solution 
 To solve the problem of determining whether a number is strobogrammatic, a brief look into what a strobogrammatic number is becomes essential. A strobogrammatic number looks the same when rotated 180 degrees – viewed upside down. Considered digits include:

- '0' -> '0'
- '1' -> '1'
- '6' -> '9'
- '8' -> '8'
- '9' -> '6'

Other digits such as '2', '3', '4', '5', and '7' do not represent valid rotations since they change shape when flipped 180 degrees.

Approach:
1. Traverse the string `num` from both ends toward the center.
2. At each corresponding position from the start and end (`i` and `len(num) - 1 - i`), verify that:
   - The character on the left half of the string matches the valid rotation of the character from the right half.
3. If at any point the characters don't match as defined by the rotation rules, the number isn't strobogrammatic.
4. If the string satisfies all conditions throughout the full loop, then it is a strobogrammatic number.



```python
class Solution:
    def isStrobogrammatic(self, num: str) -> bool:
        # Dictionary of strobogrammatic pairs
        mappings = { '0':'0', '1':'1', '6':'9', '8':'8', '9':'6' }
        
        left, right = 0, len(num) - 1
        
        while left <= right:
            l_char, r_char = num[left], num[right]
            
            # Check if both characters are valid and check the mapping
            if l_char not in mappings or mappings[l_char] != r_char:
                return False
            
            left += 1
            right -= 1
        
        return True

# Example usage:
# sol = Solution()
# print(sol.isStrobogrammatic("69"))  # Output: True
# print(sol.isStrobogrammatic("88"))  # Output: True
# print(sol.isStrobogrammatic("962")) # Output: False
# print(sol.isStrobogrammatic("1"))   # Output: True

```

Explanation:
1. **Mapping Setup**: A dictionary `mappings` is setup where keys are the characters and values are their corresponding strobogrammatic matches when flipped 180 degrees.
2. **Two Pointers Approach**: Initialize two pointers, `left` starting from the beginning and `right` starting from the end of the string.
3. **Validating Characters**: For each character at `left` (beginning) and `right` (end), check against the mapping. They need to map accurately to each other based on the strobogrammatic rule. If not, the function returns `False`.
4. **Movement of Pointers**: Move inward by incrementing `left` and decrementing `right` until they meet or cross.
5. **Final Check**: If the loop completes without finding any mismatch, it indicates that the string `num` satisfies the conditions to be strobogrammatic, returning `True`.

This approach effectively checks the input’s validity with respect to being a strobogrammatic number using the mapping and two-pointer technique, ensuring each character's validity in constant time thus, overall complexity would be O(n) where n is the length of the string `num`.

# 249. Group Shifted Strings

### Problem Description 
We can shift a string by shifting each of its letters to its successive letter.

For example, `"abc"` can be shifted to be `"bcd"`.

We can keep shifting the string to form a sequence.

For example, we can keep shifting `"abc"` to form the sequence: `"abc" -> "bcd" -> ... -> "xyz"`.

Given an array of strings `strings`, group all `strings[i]` that belong to the same shifting sequence. You may return the answer in any order.


Example 1:
Input: strings = ["abc","bcd","acef","xyz","az","ba","a","z"]
Output: [["acef"],["a","z"],["abc","bcd","xyz"],["az","ba"]]

Example 2:
Input: strings = ["a"]
Output: [["a"]]

Constraints:
`1 <= strings.length <= 200`
`1 <= strings[i].length <= 50`
`strings[i]` consists of lowercase English letters.

### Solution 
 In order to solve this problem, we will develop a function that takes an array of strings and groups them based on their shifting sequences. The key to solving this problem is recognizing a pattern for how strings are shifted.

Here are the steps we'll follow to solve the problem:
1. For each string, identify a unique identifier representing its shifting sequence.
2. Normalize each string to a base form where we treat the first character in the string as 'a' and then shift all other characters accordingly.
3. Use a hash map (`dict` in Python) to group strings that share the same normalized form.
4. Collect and return the grouped strings.

To generate a unique identifier or normalized form:
- Compute differences between consecutive characters in terms of their positions in the alphabet, considering the wrap-around from 'z' to 'a'.
- This will give a pattern (tuple) that serves as a key to group words with similar shifting patterns.

Example:
- For the string "abc":
  - The difference between 'b' and 'a' is 1
  - The difference between 'c' and 'b' is 1
- "abc" converts to a normalized pattern of (1,1).

Here's the Python solution conforming to LeetCode's format:



```python
class Solution:
    def groupStrings(self, strings: List[str]) -> List[List[str]]:
        from collections import defaultdict
        
        def normalize(s):
            if len(s) == 1:
                return 0 # For a unique key for single letter strings
            shifts = []
            for i in range(1, len(s)):
                circular_diff = (ord(s[i]) - ord(s[i-1])) % 26
                shifts.append(circular_diff)
            return tuple(shifts)
        
        result = defaultdict(list)
        for s in strings:
            result[normalize(s)].append(s)
        
        return list(result.values())

# Example use case:
sol = Solution()
print(sol.groupStrings(["abc","bcd","acef","xyz","az","ba","a","z"]))

```

This solution first defines a `normalize` function used to compute a unique tuple that represents the pattern of differences between consecutive characters, allowing wrap-around. It then populates a dictionary where these tuples serve as keys. Finally, it returns the values of this dictionary, which are lists of grouped strings with the same shifting sequence.

# 266. Palindrome Permutation

### Problem Description 
Given a string `s`, return `true` if a permutation of the string could form a palindrome.


Example 1:
Input: s = "code"
Output: false

Example 2:
Input: s = "aab"
Output: true

Example 3:
Input: s = "carerac"
Output: true

Constraints:
`1 <= s.length <= 5000`
`s` consists of only lowercase English letters.

### Solution 
 To determine if a permutation of the string could form a palindrome, we need to understand the basic characteristics of palindromes:

1. For strings of even length, every character must appear an even number of times.
2. For strings of odd length, all characters except one must appear an even number of times (the character that appears an odd number of times can be the middle character).

Given these rules, the core idea is to check the count of characters in the string:
- No more than one character should have an odd count for the string to potentially form a palindrome.

For implementation:
- We can use a `Counter` from the collections module to count occurrences of each character.
- Then, we iterate through the count values, checking how many have an odd count.

If the number of characters with an odd count is greater than 1, we return `false`. If it's 0 or 1, we return `true`.

Here is the Python code implementing this logic, formatted for submission on LeetCode:



```python
from collections import Counter

class Solution:
    def canPermutePalindrome(self, s: str) -> bool:
        # Count frequency of each character
        freq = Counter(s)
        
        # Count how many characters have an odd frequency
        odd_count = 0
        for count in freq.values():
            if count % 2 != 0:
                odd_count += 1
        
        # At most one character count should be odd
        return odd_count <= 1

# Example usage:
# sol = Solution()
# print(sol.canPermutePalindrome("code"))  # Output: False
# print(sol.canPermutePalindrome("aab"))   # Output: True
# print(sol.canPermutePalindrome("carerac"))  # Output: True

```

The `canPermutePalindrome` function:
1. Counts character frequencies using `Counter`.
2. Counts how many characters appear an odd number of times.
3. Decides if the string can form a palindrome based on the count of characters with odd frequencies.

This solution effectively checks the necessary condition for forming a palindrome permutation and should work efficiently within the given constraints.

# 274. H-Index

### Problem Description 
Given an array of integers `citations` where `citations[i]` is the number of citations a researcher received for their `ith` paper, return compute the researcher's `h`-index.

According to the definition of h-index on Wikipedia: A scientist has an index `h` if `h` of their `n` papers have at least `h` citations each, and the other `n − h` papers have no more than `h` citations each.

If there are several possible values for `h`, the maximum one is taken as the `h`-index.


Example 1:
Input: citations = [3,0,6,1,5]
Output: 3
Explanation: [3,0,6,1,5] means the researcher has 5 papers in total and each of them had received 3, 0, 6, 1, 5 citations respectively.

Since the researcher has 3 papers with at least 3 citations each and the remaining two with no more than 3 citations each, their h-index is 3.


Example 2:
Input: citations = [1,3,1]
Output: 1

Constraints:
`n == citations.length`
`1 <= n <= 5000`
`0 <= citations[i] <= 1000`

### Solution 
 To solve this problem, we will follow an efficient approach using sorting and a linear search.

### Approach
1. **Sorting:**
   First, we will sort the array `citations`. Sorting helps because it allows us to efficiently check the condition that exactly `h` papers have at least `h` citations.
   
2. **Linear Search:**
   After sorting, we traverse the sorted array. The key idea is to find the maximum `h` for which `citations[n-h] >= h`. This is because if we are considering `h` papers that should have at least `h` citations each, then in a sorted list, at least the last `h` items (after `n-h` position) should meet this criterion.

3. **Early Stopping:**
   If at any position `i` in the sorted array, the citations are less than the number of papers left (including this paper), then the maximum `h-index` can't be more than this number of papers.

### Solution
Here's a Python code implementing the above approach, compatible with direct execution in a coding environment such as LeetCode:



```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        citations.sort()
        n = len(citations)
        h_index = 0
        
        for i in range(n):
            # Number of papers with at least this many citations
            # (papers are at positions n-i, ..., n-1)
            h = n - i
            if citations[i] >= h:
                h_index = max(h_index, h)
        
        return h_index

# Example usage:
# sol = Solution()
# print(sol.hIndex([3,0,6,1,5]))  # Output: 3
# print(sol.hIndex([1,3,1]))      # Output: 1

```

### Explanation:
- **`citations.sort()`**: We begin by sorting the array in non-decreasing order.
- **`for i in range(n):`**: We traverse each paper's citations.
- **`h = n - i`**: This calculates the remaining papers from the current to the end. If we're looking for an `h`-index, these remaining papers should be at least `h`.
- **`if citations[i] >= h`**: This checks the condition that the citation count at the current sorted position is at least as great as the number of remaining papers.
- **`h_index = max(h_index, h)`**: Update `h_index` to be the maximum found so far.
  
The solution executes in \(O(n \log n)\) time due to sorting, which is efficient within the given constraints.

# 288. Unique Word Abbreviation

### Problem Description 
The abbreviation of a word is a concatenation of its first letter, the number of characters between the first and last letter, and its last letter. If a word has only two characters, then it is an abbreviation of itself.

For example:
`dog --> d1g` because there is one letter between the first letter `'d'` and the last letter `'g'`.

`internationalization --> i18n` because there are 18 letters between the first letter `'i'` and the last letter `'n'`.

`it --> it` because any word with only two characters is an abbreviation of itself.

Implement the `ValidWordAbbr` class:
`ValidWordAbbr(String[] dictionary)` Initializes the object with a `dictionary` of words.

`boolean isUnique(string word)` Returns `true` if either of the following conditions are met (otherwise returns `false`):
	
There is no word in `dictionary` whose abbreviation is equal to `word`'s abbreviation.

For any word in `dictionary` whose abbreviation is equal to `word`'s abbreviation, that word and `word` are the same.


Example 1:
Input
["ValidWordAbbr", "isUnique", "isUnique", "isUnique", "isUnique"]
[[["deer", "door", "cake", "card"]], ["dear"], ["cart"], ["cane"], ["make"]]
Output
[null, false, true, false, true]
Explanation
ValidWordAbbr validWordAbbr = new ValidWordAbbr(["deer", "door", "cake", "card"]);
validWordAbbr.isUnique("dear"); // return false, dictionary word "deer" and word "dear" have the same abbreviation
                                // "d2r" but are not the same.

validWordAbbr.isUnique("cart"); // return true, no words in the dictionary have the abbreviation "c2t".

validWordAbbr.isUnique("cane"); // return false, dictionary word "cake" and word "cane" have the same abbreviation 
                                // "c2e" but are not the same.

validWordAbbr.isUnique("make"); // return true, no words in the dictionary have the abbreviation "m2e".

validWordAbbr.isUnique("cake"); // return true, because "cake" is already in the dictionary and no other word in the dictionary has "c2e" abbreviation.


Constraints:
`1 <= dictionary.length <= 3 * 104`
`1 <= dictionary[i].length <= 20`
`dictionary[i]` consists of lowercase English letters.

`1 <= word.length <= 20`
`word` consists of lowercase English letters.

At most `5000` calls will be made to `isUnique`.

### Solution 
 To solve this problem, we'll create a class, `ValidWordAbbr`, with two main components:
1. A constructor that initializes the class with a dictionary of words.
2. A method, `isUnique`, which checks if a word has a unique abbreviation or not compared to the words in the dictionary provided during initialization.

### Algorithm
1. **Abbreviation Calculation:** Create a helper function to compute the abbreviation of a word. If a word's length is greater than 2, its abbreviation is formed as: first letter + (length of the word - 2) + last letter. If the word's length is 2 or less, return the word itself as the abbreviation.
  
2. **Constructor - Initializing**: When the class is instantiated with a dictionary:
   - Calculate the abbreviation for each word in the dictionary.
   - Use a dictionary (`self.abbrev_dict`) to map each abbreviation to a set of words that share this abbreviation.

3. **isUnique Method:** For a given word:
   - Calculate its abbreviation.
   - Check `self.abbrev_dict`:
     - If the abbreviation doesn't exist, it's unique. Return `True`.
     - If the abbreviation exists, check if the set associated with this abbreviation contains only the word in question (and no other variants). If true, it's unique. Otherwise, return `False`.

### Implementation in Python



```python
class ValidWordAbbr:
    def __init__(self, dictionary):
        self.abbrev_dict = {}
        for word in dictionary:
            abbr = self._get_abbreviation(word)
            if abbr not in self.abbrev_dict:
                self.abbrev_dict[abbr] = {word}
            else:
                self.abbrev_dict[abbr].add(word)

    def isUnique(self, word: str) -> bool:
        abbr = self._get_abbreviation(word)
        if abbr not in self.abbrev_dict:
            return True
        else:
            # Check if only this word has the abbreviation
            return self.abbrev_dict[abbr] == {word}

    def _get_abbreviation(self, word: str) -> str:
        if len(word) <= 2:
            return word
        else:
            return f"{word[0]}{len(word) - 2}{word[-1]}"

# The following are sample test cases that you could use to check the class functionality

# Example 1:
validWordAbbr = ValidWordAbbr(["deer", "door", "cake", "card"])
print(validWordAbbr.isUnique("dear"))  # Output: False
print(validWordAbbr.isUnique("cart"))  # Output: True
print(validWordAbbr.isUnique("cane"))  # Output: False
print(validWordAbbr.isUnique("make"))  # Output: True
print(validWordAbbr.isUnique("cake"))  # Output: True

```

### Complexity Analysis
- **Time Complexity:** O(N) for initializing with the dictionary where N is the total number of characters across all words. Each call to `isUnique` is O(1) assuming hash operations are constant time.
- **Space Complexity:** O(D) where D is the number of distinct words or abbreviations.

This implementation ensures both the initialization of the dictionary and the checking for uniqueness operate efficiently considering the constraints provided.

# 290. Word Pattern

### Problem Description 
Given a `pattern` and a string `s`, find if `s` follows the same pattern.

Here follow means a full match, such that there is a bijection between a letter in `pattern` and a non-empty word in `s`.


Example 1:
Input: pattern = "abba", s = "dog cat cat dog"
Output: true

Example 2:
Input: pattern = "abba", s = "dog cat cat fish"
Output: false

Example 3:
Input: pattern = "aaaa", s = "dog cat cat dog"
Output: false

Example 4:
Input: pattern = "abba", s = "dog dog dog dog"
Output: false

Constraints:
`1 <= pattern.length <= 300`
`pattern` contains only lower-case English letters.

`1 <= s.length <= 3000`
`s` contains only lower-case English letters and spaces `' '`.

`s` does not contain any leading or trailing spaces.

All the words in `s` are separated by a single space.

### Solution 
 Here is the Python solution for the problem described above using a hash map approach to establish a bijection between the characters in `pattern` and words in `s`. This solution is also designed to be run on the LeetCode platform directly.



```python
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        words = s.split()  # Split the string `s` into words
        
        # If the number of words doesn't match the length of the pattern, they cannot map 1-to-1
        if len(words) != len(pattern):
            return False
    
        char_to_word = {}
        word_to_char = {}
        
        # Iterate through the paired characters and words
        for char, word in zip(pattern, words):
            # If a mapping already exists, check for consistency
            if char in char_to_word:
                if char_to_word[char] != word:
                    return False
            else:
                # If it's a new character and the word is already taken, fail
                if word in word_to_char:
                    return False

                # Create new mappings
                char_to_word[char] = word
                word_to_char[word] = char
        
        return True

# The following lines can be used for testing within an environment like LeetCode
# sol = Solution()
# print(sol.wordPattern("abba", "dog cat cat dog"))  # Output: True
# print(sol.wordPattern("abba", "dog cat cat fish")) # Output: False
# print(sol.wordPattern("aaaa", "dog cat cat dog"))  # Output: False
# print(sol.wordPattern("abba", "dog dog dog dog"))  # Output: False

```

### Explanation:

- **Splitting and Length Check**: First, the string `s` is split into a list of words using `s.split()`. If the count of the words does not match the length of `pattern`, the function immediately returns `False` (since a bijection isn't possible).

- **Dictionaries for Mappings**: Two dictionaries are maintained:
  - `char_to_word`: Maps characters in `pattern` to words in `s`.
  - `word_to_char`: Maps words in `s` back to characters in `pattern`. This helps ensure that each word is mapped to exactly one character, maintaining a bijection.

- **Iterative Checking**: Through a loop comparing elements of `pattern` and words in the list, the function performs the following:
  - Checks if there is already a mapping for the current character or word. If so, it verifies this mapping matches the current word or character, respectively.
  - If there's no mapping yet but the word or character is already associated with a different character or word, the function returns `False`.
  - New mappings are created if the current character-word pair doesn't contradict earlier mappings.
  
- **Result**: Finally, if no contradictions are found through the loop, the function returns `True`.

The provided solution maintains a bijection and properly maps characters with words, allowing it to handle all the constraints stipulated in the problem prompt.

# 299. Bulls and Cows

### Problem Description 
You are playing the Bulls and Cows game with your friend.

You write down a secret number and ask your friend to guess what the number is. When your friend makes a guess, you provide a hint with the following info:
The number of "bulls", which are digits in the guess that are in the correct position.

The number of "cows", which are digits in the guess that are in your secret number but are located in the wrong position. Specifically, the non-bull digits in the guess that could be rearranged such that they become bulls.

Given the secret number `secret` and your friend's guess `guess`, return the hint for your friend's guess.

The hint should be formatted as `"xAyB"`, where `x` is the number of bulls and `y` is the number of cows. Note that both `secret` and `guess` may contain duplicate digits.


Example 1:
Input: secret = "1807", guess = "7810"
Output: "1A3B"
Explanation: Bulls are connected with a '|' and cows are underlined:
"1807"
  |
"7810"

Example 2:
Input: secret = "1123", guess = "0111"
Output: "1A1B"
Explanation: Bulls are connected with a '|' and cows are underlined:
"1123"        "1123"
  |      or     |
"0111"        "0111"
Note that only one of the two unmatched 1s is counted as a cow since the non-bull digits can only be rearranged to allow one 1 to be a bull.


Example 3:
Input: secret = "1", guess = "0"
Output: "0A0B"

Example 4:
Input: secret = "1", guess = "1"
Output: "1A0B"

Constraints:
`1 <= secret.length, guess.length <= 1000`
`secret.length == guess.length`
`secret` and `guess` consist of digits only.

### Solution 
 To solve the "Bulls and Cows" problem effectively and within the constraints provided, we can approach it using a direct comparative and counting technique. The solution involves understanding two main components:

1. **Bulls**: This is a straightforward component where we simply check whether the digit in a specific position of the `secret` is the same as in the `guess`. If they match, then it's a "bull".

2. **Cows**: These are calculated by noting the frequency of each digit in both strings, but only considering those digits that aren't already classified as bulls. If a digit appears in the `secret` more times than in the `guess`, the excess counts in `secret` contribute to the cows.

Here's a method to compute the hint:

- Use two arrays or dictionaries to count occurrences of each digit in `secret` and `guess`.
  
- Iterate through `secret` and `guess` in one pass:
  - If the digits at the current index are the same, increment the "bulls" count and decrement the corresponding counts in both the `secret` and `guess` counters.
  
- After detecting all bulls, calculate cows. For each digit, add to the "cows" count the minimum between its remaining occurrences in `secret` and `guess`.

This dual step ensures accurate counts even in the presence of repeating numbers and keeps the complexity down to O(N) for length N of the input strings.

Here's the Python code implementing the above approach which should work directly on the LeetCode platform:



```python
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        from collections import defaultdict
        
        bull_count = 0
        secret_count = defaultdict(int)
        guess_count = defaultdict(int)
        
        # First, count bulls and record frequencies of digits in secret and guess
        for s_char, g_char in zip(secret, guess):
            if s_char == g_char:
                bull_count += 1
            else:
                secret_count[s_char] += 1
                guess_count[g_char] += 1
        
        cow_count = 0
        # Now, calculate cows from the frequencies
        for digit in secret_count:
            if digit in guess_count:
                cow_count += min(secret_count[digit], guess_count[digit])
        
        return f"{bull_count}A{cow_count}B"

# Example usage:
sol = Solution()
print(sol.getHint("1807", "7810"))  # Output: "1A3B"
print(sol.getHint("1123", "0111"))  # Output: "1A1B"
print(sol.getHint("1", "0"))        # Output: "0A0B"
print(sol.getHint("1", "1"))        # Output: "1A0B"

```

This Python code leverages the `collections.defaultdict` for counting and relies on common dictionary operations to determine matches for bulls and potential matches for cows. It is efficient, concise, and aligns with the constraints and requirements specified.

# 311. Sparse Matrix Multiplication

### Problem Description 
Given two sparse matrices `mat1` of size `m x k` and `mat2` of size `k x n`, return the result of `mat1 x mat2`. You may assume that multiplication is always possible.


Example 1:
Input: mat1 = [[1,0,0],[-1,0,3]], mat2 = [[7,0,0],[0,0,0],[0,0,1]]
Output: [[7,0,0],[-7,0,3]]

Example 2:
Input: mat1 = [[0]], mat2 = [[0]]
Output: [[0]]

Constraints:
`m == mat1.length`
`k == mat1[i].length == mat2.length`
`n == mat2[i].length`
`1 <= m, n, k <= 100`
`-100 <= mat1[i][j], mat2[i][j] <= 100`

### Solution 
 To solve the problem of multiplying two sparse matrices, we need to focus on efficiently computing products, leveraging the sparse characteristics—that is, most of the elements in both matrices are 0 and thus their multiplication with other numbers yields zeros which do not contribute to the result.

The naive matrix multiplication algorithm has a time complexity of O(m * k * n), which in this case would also iterate over a lot of zero values. To optimize, we consider non-zero entries only in our calculation.

### Steps for Optimization:
1. **Identify Non-Zero Entries**:
    - Traverse through `mat1` and store the indices `(i, j)` and values where the values are not zero.
    - Similarly traverse through `mat2` but since matrix multiplication involves row-column combinations, store `(j, l)` indices (mapping columns of `mat1` with rows in `mat2`).

2. **Multiply the Non-Zero pairs**:
    - Use the stored indices and values from above, for each `(i, j)` pair in `mat1` and `(j, l)` pair in `mat2`, multiply and accumulate the results in the resultant matrix position `result[i][l]`.

This approach saves computations by ignoring zero products and only calculating the necessary non-zero multiplications.

### Implementation:


```python
class Solution:
    def multiply(self, mat1, mat2):
        # Dimensions from input matrices
        m, k = len(mat1), len(mat1[0])
        _, n = len(mat2), len(mat2[0])
        
        # Resultant matrix of size m x n initialized to all zeroes
        result = [[0] * n for _ in range(m)]
        
        # Dictionary to store non-zero values in mat1
        non_zero_mat1 = {}
        for i in range(m):
            for j in range(k):
                if mat1[i][j] != 0:
                    if i not in non_zero_mat1:
                        non_zero_mat1[i] = []
                    non_zero_mat1[i].append((j, mat1[i][j]))
                    
        # Dictionary to store non-zero values in mat2
        non_zero_mat2 = {}
        for j in range(k):
            for l in range(n):
                if mat2[j][l] != 0:
                    if j not in non_zero_mat2:
                        non_zero_mat2[j] = []
                    non_zero_mat2[j].append((l, mat2[j][l]))
                    
        # Compute resultant matrix based on non-zero pair multiplications
        for i in non_zero_mat1:
            for j, val1 in non_zero_mat1[i]:
                if j in non_zero_mat2:
                    for l, val2 in non_zero_mat2[j]:
                        result[i][l] += val1 * val2
        
        return result

```

### LeetCode Test Format
This code can be directly input into a LeetCode problem environment (provided the named function and class match the requirement). It efficiently handles the matrix multiplication by focusing on non-zero entries, providing a performance boost, especially for sparse matrices.

### Run Example
The `Solution().multiply(mat1, mat2)` method will deliver the output matrix which results from performing the multiplication of `mat1` by `mat2` according to the description above.

# 325. Maximum Size Subarray Sum Equals k

### Problem Description 
Given an integer array `nums` and an integer `k`, return the maximum length of a subarray that sums to `k`. If there isn't one, return `0` instead.


Example 1:
Input: nums = [1,-1,5,-2,3], k = 3
Output: 4
Explanation: The subarray [1, -1, 5, -2] sums to 3 and is the longest.


Example 2:
Input: nums = [-2,-1,2,1], k = 1
Output: 2
Explanation: The subarray [-1, 2] sums to 1 and is the longest.


Constraints:
`1 <= nums.length <= 104`
`-104 <= nums[i] <= 104`
`-105 <= k <= 105`
Follow Up: Can you do it in `O(n)` time?

### Solution 
 To solve this problem efficiently, we can utilize a hash map to keep track of the cumulative sums of the array as we iterate through it. This allows us to check whether any subarray has the required sum `k` efficiently.

Here’s how the approach works:

1. **Cumulative Sum**: We maintain a cumulative sum as we iterate over the array. The cumulative sum at any point gives the sum of elements from the start of the array to the current position.

2. **Hash Map**: We use a hash map (`seen`) to store the first occurrence of each cumulative sum as a key and its corresponding index as the value. If the cumulative sum minus `k` (Say `cumSum - k`) is present in `seen`, it provides the existence of a subarray with sum `k`.

3. **Subarray Length Calculation**: The difference between the current index and the index stored in `seen` for `cumSum - k` gives the length of the subarray that sums to `k`.

4. **Initialization**: We initially put `0` as a cumulative sum in the map with its corresponding index as `-1` to handle cases where the subarray begins from the first element itself.

We implement this approach in Python, compatible with the LeetCode platform:



```python
class Solution:
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        seen = {0: -1}  # Store cumulative sum as key, index as value
        cum_sum = 0  # This will store the cumulative sum of numbers as we iterate.
        max_len = 0  # This will keep track of the maximum length of a subarray seen so far that sums to k.

        for i, num in enumerate(nums):
            cum_sum += num  # Update the cumulative sum.
            
            # If cum_sum - k exists in seen, it means there is a subarray summing to k.
            if (cum_sum - k) in seen:
                # Calculate the length of this subarray by (current_index - index_of_cum_sum_at_seen).
                max_len = max(max_len, i - seen[cum_sum - k])

            # Store cum_sum in seen if it is not already there to maintain the longest subarray length
            if cum_sum not in seen:
                seen[cum_sum] = i

        return max_len

```

This solution goes through each element of `nums` only once, hence its time complexity is \(O(n)\). The space complexity is also \(O(n)\) because, in the worst case, every cumulative sum is distinct and stored in the hash map. This approach is highly efficient and should work well within the provided constraints.

# 336. Palindrome Pairs

### Problem Description 
Given a list of unique words, return all the pairs of the distinct indices `(i, j)` in the given list, so that the concatenation of the two words `words[i] + words[j]` is a palindrome.


Example 1:
Input: words = ["abcd","dcba","lls","s","sssll"]
Output: [[0,1],[1,0],[3,2],[2,4]]
Explanation: The palindromes are ["dcbaabcd","abcddcba","slls","llssssll"]

Example 2:
Input: words = ["bat","tab","cat"]
Output: [[0,1],[1,0]]
Explanation: The palindromes are ["battab","tabbat"]

Example 3:
Input: words = ["a",""]
Output: [[0,1],[1,0]]

Constraints:
`1 <= words.length <= 5000`
`0 <= words[i].length <= 300`
`words[i]` consists of lower-case English letters.

### Solution 
 In this problem, we need to find pairs of words from a list such that their concatenation forms a palindrome. A palindrome is a word or sequence that reads the same forwards and backwards.

To efficiently solve this problem, we can make use of a hashmap (or dictionary in Python) that maps a word to its index. Additionally, we'll leverage the properties of palindromes to reduce the checks we need to perform.

A key realization here is that if `word1 + word2` is a palindrome, then:
1. The reverse of `word2` should be a prefix of `word1`, or
2. The reverse of `word1` should be a suffix of `word2`.

Thus, instead of concatenating each pair and checking if it forms a palindrome—a process which would be computationally expensive—we can:
- Check if the reverse of the prefix of a word matches another word.
- Check if the reverse of the suffix of a word matches another word.
This allows us to decide if two words can form a palindrome when concatenated in either order.

To implement this:
1. Store all words in a dictionary with words as keys and their indices as values. This facilitates quick look-up.
2. For each word, split it into all possible prefixes and suffixes.
3. For each piece, check if the reversed piece exists in the dictionary.
4. There are two types of checks for each split:
   - If the reversed prefix or suffix forms a word and the rest of the word itself is a palindrome.
   - Special edge case when word itself is a palindrome and we might concatenate with an empty string (if exists in the list).

Here is the LeetCode-styled solution:



```python
class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        def is_palindrome(check):
            return check == check[::-1]

        words_dict = {word: i for i, word in enumerate(words)}
        result = []

        for i, word in enumerate(words):
            for j in range(len(word) + 1):
                # substrings word[:j] (prefix) and word[j:] (suffix)
                prefix, suffix = word[:j], word[j:]

                # reversed versions of prefix and suffix
                rev_prefix, rev_suffix = prefix[::-1], suffix[::-1]

                # 1. If reversed suffix is a palindrome, check if there's a corresponding reversed prefix in `words_dict`
                if is_palindrome(suffix) and rev_prefix in words_dict:
                    if i != words_dict[rev_prefix]:  # To avoid pairing the same word with itself
                        result.append([i, words_dict[rev_prefix]])

                # 2. If reversed prefix is a palindrome, check if there's a corresponding reversed suffix in `words_dict` 
                # We ensure j > 0 to prevent duplicates of the first check
                if j > 0 and is_palindrome(prefix) and rev_suffix in words_dict:
                    if i != words_dict[rev_suffix]:  # To avoid pairing the same word with itself
                        result.append([words_dict[rev_suffix], i])

        return result

```

This algorithm efficiently categorizes checking into prefix and suffix palindromes and ensures the full scanning of possible combinations only when necessary. Through dictionary look-ups, the solution maximally reduces the time complexity compared with a naive approach that would be significantly slower, especially given the input constraints.

# 340. Longest Substring with At Most K Distinct Characters

### Problem Description 
Given a string `s` and an integer `k`, return the length of the longest substring of `s` that contains at most `k` distinct characters.


Example 1:
Input: s = "eceba", k = 2
Output: 3
Explanation: The substring is "ece" with length 3.


Example 2:
Input: s = "aa", k = 1
Output: 2
Explanation: The substring is "aa" with length 2.


Constraints:
`1 <= s.length <= 5 * 104`
`0 <= k <= 50`

### Solution 
 To solve this problem, we can utilize the sliding window technique with the help of a hash map to keep track of the distinct characters and their frequency within the current window (substring). The sliding window technique is ideal for problems where you are looking for a range in a contiguous sequence that satisfies certain conditions.

Here is a step-by-step breakdown of the approach:
1. Use two pointers, `start` and `end`, to represent the current window.
2. Expand the `end` pointer to include new characters and update the count of each character in the current window using a hash map.
3. If the number of distinct characters in the window is more than `k`, we need to contract the window from the `start` until the condition is met again.
4. For every position of the `end` pointer, update the maximum length of the substring found so far, if applicable.
5. The final answer is the maximum length of valid substring that we keep track of during the entire process.

Here's how you can implement this solution in Python:



```python
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        if k == 0:
            return 0
        
        char_count = {}
        max_length = 0
        start = 0
        
        for end in range(len(s)):
            char = s[end]
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
            
            # If the number of distinct characters is more than k,
            # move the start pointer to the right until we meet the condition
            while len(char_count) > k:
                start_char = s[start]
                char_count[start_char] -= 1
                if char_count[start_char] == 0:
                    del char_count[start_char]
                start += 1
            
            # Calculate the maximum length of the substring
            max_length = max(max_length, end - start + 1)
        
        return max_length

```

Explanation:
- `char_count` is the dictionary that keeps track of the count of each character in the current window.
- `max_length` stores the maximum length of the substring we found that meets the condition.
- The `start` and `end` pointers define the current window of the substring.
- We expand the window by moving the `end` pointer and adjust the `start` pointer to ensure the number of distinct characters in the window does not exceed `k`.

This code can be directly tested on the LeetCode platform for various test cases. It should perform efficiently due to the use of the sliding window technique, especially given the constraints. The time complexity of this approach is O(n) and the space complexity is O(min(m, k)) where `m` is the number of unique characters in the string and `k` is the maximum distinct characters allowed.

# 347. Top K Frequent Elements

### Problem Description 
Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. You may return the answer in any order.


Example 1:
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Example 2:
Input: nums = [1], k = 1
Output: [1]

Constraints:
`1 <= nums.legth <= 105`
`k` is in the range `[1, the number of unique elements in the array]`.

It is guaranteed that the answer is unique.

Follow up: Your algorithm's time complexity must be better than `O(n log n)`, where n is the array's size.

### Solution 
 To solve the "Top K Frequent Elements" problem, we can use the following approach:

1. **Calculate Frequencies:** First, we need to count the frequencies of each element in the array. This can be efficiently done using a dictionary (or hash map).

2. **Bucket Sort for Frequencies:** Instead of sorting all the elements directly, we can employ a "bucket sort-like" method based on frequency. In this method, we use a list where each index represents how many times elements occur (frequency), and the elements at each index are those that have such frequency. This makes it possible to quickly access the most frequent items without needing to sort the entire input.

3. **Collect Top K Frequent Elements:** Iterate over the bucket list from the end (which corresponds to the most frequent elements) and collect elements until we have gathered the top `k`.

This algorithm avoids direct sorting of elements based on their frequencies and thus can perform better than \(O(n \log n)\). The time complexity of this approach is close to \(O(n)\), and it handles all specified constraints effectively.

Here's a detailed Python code to address the problem:



```python
from collections import Counter
from typing import List

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # Build a frequency dictionary
        frequency = Counter(nums)
        
        # Create a list where index represents frequency, and each element is a list of numbers with that frequency
        # The maximum possible frequency is the length of `nums`
        bucket = [[] for _ in range(len(nums) + 1)]
        
        for num, freq in frequency.items():
            bucket[freq].append(num)
        
        # Result list to store top k frequent elements
        result = []
        
        # Traverse the bucket from the end to start
        # This is because the end contains lists of elements with the highest frequencies
        for i in range(len(bucket) - 1, 0, -1):
            for num in bucket[i]:
                result.append(num)
                # If we've added k elements, return the result
                if len(result) == k:
                    return result
                
# This code is now ready for execution and testing on LeetCode to match their environment.

```

This code correctly follows the LeetCode format and can be used to directly submit a solution there. You can test it with the sample inputs provided in the question or any other inputs to verify its functionality.

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
 Here is the Python code that addresses the problem statement using the LeetCode solution format:



```python
from typing import List

class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Use set to find unique elements in both lists
        set_nums1 = set(nums1)
        set_nums2 = set(nums2)
        
        # Use the set intersection method to find common elements
        # and convert the set back to a list
        return list(set_nums1.intersection(set_nums2))
    
# Example usage:
# sol = Solution()
# print(sol.intersection([1,2,2,1], [2,2]))  # Output: [2]
# print(sol.intersection([4,9,5], [9,4,9,8,4]))  # Output: [9,4]

```

### Explanation:
1. **Convert Lists to Sets**: We initially convert `nums1` and `nums2` into sets (`set_nums1` and `set_nums2`). Using sets automatically removes duplicate entries and reduces the problem to dealing with unique elements only.

2. **Find the Intersection**: The intersection of two sets can be easily determined using the `.intersection()` method which is efficient and directly applicable to sets. It returns the set of elements common to both `set_nums1` and `set_nums2`.

3. **Convert Set to List**: Finally, since the result needs to be returned as a list, we convert the resulting set from the intersection operation into a list.

4. **Return**: The list containing the unique elements present in both input lists is returned.

### Complexity Analysis:
- **Time Complexity**: O(n + m), where n is the size of `nums1` and m is the size of `nums2`. This comes from the need to insert elements into the set and compute the intersection.
- **Space Complexity**: O(n + m), for storing elements of `nums1` and `nums2` in the sets temporarily. However, in the worst case (all unique elements), this space usage matches the size of the input.

This solution is both efficient and concise, and conforms to typical LeetCode format, ready to be copied into the online editor and executed.

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
 The problem at hand is to find the intersection of two arrays where each element in the result must appear as many times as it is present in both arrays. A straightforward way to handle this is by using Python dictionaries (or hash maps) to count the occurrences of each element in both arrays, then determine the intersection accordingly.

Here's how you can solve the problem using a Python dictionary:



```python
from typing import List

class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Dictionary to store the frequency of elements in nums1
        count_nums1 = {}
        for num in nums1:
            if num in count_nums1:
                count_nums1[num] += 1
            else:
                count_nums1[num] = 1
                
        # List to store the intersection result
        intersection = []
        
        # Iterate over nums2 and find common elements as per the stored counts in count_nums1
        for num in nums2:
            if num in count_nums1 and count_nums1[num] > 0:
                intersection.append(num)
                count_nums1[num] -= 1  # Decrease the count to ensure the number isn't used more than it appears in nums1
        
        return intersection

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.intersect([1,2,2,1], [2,2]))  # Output: [2,2]
    print(sol.intersect([4,9,5], [9,4,9,8,4]))  # Output: [4,9] or [9,4]

```

### Explanation:
1. **Dictionary Creation**: We first build a dictionary `count_nums1` to count the frequency of each element in `nums1`.
2. **Finding Intersection**: We then iterate through `nums2`, and whenever we find an element that also exists in `count_nums1` (with a non-zero count), we add it to our result list `intersection` and decrement the count in `count_nums1` to ensure we're respecting frequency.

### Follow-up Questions:
1. **Sorted Array Optimization**:
   If both arrays are sorted, you can use a two-pointer technique which greatly reduces the need for extra space:
   

```python
   def intersect_sorted(self, nums1: List[int], nums2: List[int]) -> List[int]:
       nums1.sort()
       nums2.sort()
       i, j = 0, 0
       intersection = []
       while i < len(nums1) and j < len(nums2):
           if nums1[i] < nums2[j]:
               i += 1
           elif nums1[i] > nums2[j]:
               j += 1
           else:
               intersection.append(nums1[i])
               i += 1
               j += 1
       return intersection
   
```

2. **Disparate Array Sizes**:
   When one array is much smaller than the other, it's efficient to use a hashmap for the smaller array to reduce space and potential computational overhead.

3. **External Storage**:
   If `nums2` is very large and stored on disk, you might process it in chunks that fit in memory while maintaining a count of elements from `nums1` in memory. This way, you handle the data in a streaming fashion.

The above solution combines simplicity and efficient handling of the problem constraints, making it appropriate for most cases, especially those considering limited memory and potential external storage of data.


# 355. Design Twitter

### Problem Description 
Design a simplified version of Twitter where users can post tweets, follow/unfollow another user, and is able to see the `10` most recent tweets in the user's news feed.

Implement the `Twitter` class:
`Twitter()` Initializes your twitter object.

`void postTweet(int userId, int tweetId)` Composes a new tweet with ID `tweetId` by the user `userId`. Each call to this function will be made with a unique `tweetId`.

`List<Integer> getNewsFeed(int userId)` Retrieves the `10` most recent tweet IDs in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user themself. Tweets must be ordered from most recent to least recent.

`void follow(int followerId, int followeeId)` The user with ID `followerId` started following the user with ID `followeeId`.

`void unfollow(int followerId, int followeeId)` The user with ID `followerId` started unfollowing the user with ID `followeeId`.


Example 1:
Input
["Twitter", "postTweet", "getNewsFeed", "follow", "postTweet", "getNewsFeed", "unfollow", "getNewsFeed"]
[[], [1, 5], [1], [1, 2], [2, 6], [1], [1, 2], [1]]
Output
[null, null, [5], null, null, [6, 5], null, [5]]
Explanation
Twitter twitter = new Twitter();
twitter.postTweet(1, 5); // User 1 posts a new tweet (id = 5).

twitter.getNewsFeed(1);  // User 1's news feed should return a list with 1 tweet id -> [5]. return [5]
twitter.follow(1, 2);    // User 1 follows user 2.

twitter.postTweet(2, 6); // User 2 posts a new tweet (id = 6).

twitter.getNewsFeed(1);  // User 1's news feed should return a list with 2 tweet ids -> [6, 5]. Tweet id 6 should precede tweet id 5 because it is posted after tweet id 5.

twitter.unfollow(1, 2);  // User 1 unfollows user 2.

twitter.getNewsFeed(1);  // User 1's news feed should return a list with 1 tweet id -> [5], since user 1 is no longer following user 2.


Constraints:
`1 <= userId, followerId, followeeId <= 500`
`0 <= tweetId <= 104`
All the tweets have unique IDs.

At most `3 * 104` calls will be made to `postTweet`, `getNewsFeed`, `follow`, and `unfollow`.

### Solution 
 To address the problem, we will design a `Twitter` class with specific methods to manage tweet posting, user following, and generating news feeds. We will have to efficiently handle a somewhat limited user database (at most 500 different users) but with potentially many operations.

Here's how we can approach the main functionalities:

1. **Data Structures**:
   - Use `defaultdict` to store tweets and follower relationships:
     - `tweets_by_user`: Maps each user ID to a list of tuples containing tweet IDs and a timestamp to maintain the order of tweets.
     - `followers`: Maps each user ID to a set of user IDs that follow the user.
   - Utilize Python's built-in functionality to manage timestamps, and ordering.
   - Use a min-heap (with a fixed size of 10) for efficiently getting the top 10 recent tweets.

2. **Methods**:
   - `postTweet(userId, tweetId)`: Record a tweet along with a timestamp.
   - `getNewsFeed(userId)`: Fetch the 10 most recent tweets from the user and their followees using a heap for the top 10 logic.
   - `follow(followerId, followeeId)`: Add a follower.
   - `unfollow(followerId, followeeId)`: Remove a follower.

3. **Handling Multiple Calls**:
   - Given the constraints with up to `30000` calls, efficiency in `getNewsFeed` is crucial; by maintaining a straightforward way of collecting tweets and using effective sorting (only on 10 elements), we ensure scalability.

Let's now translate this design into the Python code:



```python
from collections import defaultdict
import heapq

class Twitter:

    def __init__(self):
        self.tweets_by_user = defaultdict(list)
        self.followers = defaultdict(set)
        self.timestamp = 0

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.tweets_by_user[userId].append((self.timestamp, tweetId))
        self.timestamp -= 1  # decrement to ensure the tweet order is from newest to oldest

    def getNewsFeed(self, userId: int) -> list[int]:
        min_heap = []
        users_to_check = self.followers[userId] | {userId}
        for user in users_to_check:
            for tweet in self.tweets_by_user[user]:
                if len(min_heap) < 10:
                    heapq.heappush(min_heap, tweet)
                else:
                    heapq.heappushpop(min_heap, tweet)
                    
        # Extracting and reversing the results from the min-heap to get correct order
        return [tweet_id for _, tweet_id in sorted(min_heap, reverse=True)]

    def follow(self, followerId: int, followeeId: int) -> None:
        self.followers[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId in self.followers[followerId]:
            self.followers[followerId].remove(followeeId)

# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)

```

This solution suits the constraints and should run effectively for the given upper limits on LeetCode. Advanced edge cases, extensive datasets, and potential real-world adjustments will depend on more intricate profiling and testing but are beyond typical competitive programming or interview scenarios.

# 356. Line Reflection

### Problem Description 
Given n points on a 2D plane, find if there is such a line parallel to y-axis that reflect the given points symmetrically, in other words, answer whether or not if there exists a line that after reflecting all points over the given line the set of the original points is the same that the reflected ones.

Note that there can be repeated points.

Follow up:
Could you do better than O(n2) ?

Example 1:
Input: points = [[1,1],[-1,1]]
Output: true
Explanation: We can choose the line x = 0.


Example 2:
Input: points = [[1,1],[-1,-1]]
Output: false
Explanation: We can't choose a line.


Constraints:
`n == points.length`
`1 <= n <= 10^4`
`-10^8 <= points[i][j] <= 10^8`

### Solution 
 To solve this problem, we need to determine if there exists a line parallel to the y-axis such that reflecting any given point over this line results in another given point from the input.

### Algorithm Steps:
1. **Find a potential mirror line**: Compute the potential line of symmetry by calculating the average of the minimum and maximum x-values across all points. This average x-value would be the x-coordinate (`mirror_x`) of the symmetry line if it exists.
  
2. **Use a Set for Verification**: As we iterate through each point, compute its reflection across the line `x = mirror_x`. The reflected point should also exist in the given set of points. Using a set will help in quick lookup and provide an efficient way to check the existence of the reflective point.
  
3. **Check for Reflection**: For every point `(x, y)`, the reflected point would be `((2 * mirror_x - x), y)`. Check if this reflected point exists in our set of points. If all points satisfy this condition, then such a symmetric line exists.

### Optimizations and Considerations:
- If a point is exactly on the line of symmetry (`mirror_x == x`), it should satisfy the symmetry condition by itself.
- Leveraging a set for lookup operations ensures the check for the reflective point is efficient.

### Code (Python):



```python
class Solution:
    def isReflected(self, points: List[List[int]]) -> bool:
        point_set = set((x, y) for x, y in points)
        min_x = min(x for x, _ in points)
        max_x = max(x for x, _ in points)
        
        # Calculate the potential symmetric line's x coordinate
        mirror_x = (max_x + min_x) / 2
        
        # Check if every point has its reflection in the set
        for x, y in points:
            reflection_x = 2 * mirror_x - x
            if (reflection_x, y) not in point_set:
                return False
        
        return True

```

### Example Usage:
Input `[[1,1],[-1,1]]` will return `True` as there exists a symmetric line (`x = 0`). However, input `[[1, 1], [-1, -1]]` will return `False` as no such symmetric line exists satisfying the condition for all points.

This solution efficiently checks for symmetrical reflection and adheres to the O(n) complexity, given that set operations (insertion, lookup) are average O(1) operations. This approach is significantly better than a naive O(n^2) solution.

This code is ready to be run directly in the LeetCode environment as it follows the expected class and method signature.

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
 To solve this problem on LeetCode, we can use a greedy approach with a priority queue and hash map counting approach that tracks character frequencies and uses these frequencies to determine how characters are spaced.

To implement:
1. Count the frequencies of each character in the string.
2. Use a priority queue (max-heap) to always fetch the character with the highest remaining frequency to make sure the rearrangement promotes maximum spacing.
3. Keep a queue to maintain characters waiting for their next valid placement if the spacing requirement `k` is still enforced for these characters.
4. As we place characters into a result string, decrement their frequency.
5. If a character's frequency is not zero after placement, put it in a waiting queue to reinsert it into the priority queue once k-spacing is met.

If at any step, we can't fetch a character from the priority queue to maintain the k-distance rule, we know that the rearrangement isn't possible, so we return an empty string.

Here is the Python code implementing the solution.



```python
class Solution:
    def rearrangeString(self, s: str, k: int) -> str:
        from collections import Counter, deque
        import heapq
        
        # Edge case where no rearrangement is needed
        if k == 0:
            return s

        # Character frequency count
        freq = Counter(s)
        
        # Max-heap for keeping characters sorted by their frequencies
        max_heap = []
        for char, count in freq.items():
            # Push negative because heapq is a min-heap by default in Python
            heapq.heappush(max_heap, (-count, char))
        
        result = []
        queue = deque()  # Holds elements until they can be re-used
        
        while max_heap:
            count, char = heapq.heappop(max_heap)
            
            # Append the current character to result
            result.append(char)
            
            # We push (original count + 1 because it was negative) to queue with the character
            queue.append((count + 1, char))
            
            # If the queue length is 'k', we can recycle/reuse the characters
            if len(queue) == k:
                cnt, ch = queue.popleft()
                if -cnt > 0:  # Still have some frequency left
                    heapq.heappush(max_heap, (cnt, ch))
        
        # Join the list into final resulting string
        final_result = ''.join(result)
        
        # Check if the final result's length matches the input string's length.
        if len(final_result) != len(s):
            return ""
        
        return final_result

# Example usage:
sol = Solution()
print(sol.rearrangeString("aabbcc", 3))  # Output: "abcabc"
print(sol.rearrangeString("aaabc", 3))   # Output: ""
print(sol.rearrangeString("aaadbbcc", 2)) # Output: "abacabcd"

```

This solution ensures that the most frequent characters are spaced effectively and respects the constraint that `1 <= s.length <= 3 * 10^5`. The time complexity is approximately O(n log n) due to the use of heap operations where `n` is the length of the string `s`. The space complexity is O(n) for storing character counts and the resultant data structures.

# 359. Logger Rate Limiter

### Problem Description 
Design a logger system that receives a stream of messages along with their timestamps. Each unique message should only be printed at most every 10 seconds (i.e. a message printed at timestamp `t` will prevent other identical messages from being printed until timestamp `t + 10`).

All messages will come in chronological order. Several messages may arrive at the same timestamp.

Implement the `Logger` class:
`Logger()` Initializes the `logger` object.

`bool shouldPrintMessage(int timestamp, string message)` Returns `true` if the `message` should be printed in the given `timestamp`, otherwise returns `false`.


Example 1:
Input
["Logger", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage"]
[[], [1, "foo"], [2, "bar"], [3, "foo"], [8, "bar"], [10, "foo"], [11, "foo"]]
Output
[null, true, true, false, false, false, true]
Explanation
Logger logger = new Logger();
logger.shouldPrintMessage(1, "foo");  // return true, next allowed timestamp for "foo" is 1 + 10 = 11
logger.shouldPrintMessage(2, "bar");  // return true, next allowed timestamp for "bar" is 2 + 10 = 12
logger.shouldPrintMessage(3, "foo");  // 3 < 11, return false
logger.shouldPrintMessage(8, "bar");  // 8 < 12, return false
logger.shouldPrintMessage(10, "foo"); // 10 < 11, return false
logger.shouldPrintMessage(11, "foo"); // 11 >= 11, return true, next allowed timestamp for "foo" is
                                      // 11 + 10 = 21

Constraints:
`0 <= timestamp <= 109`
Every `timestamp` will be passed in non-decreasing order (chronological order).

`1 <= message.length <= 30`
At most `104` calls will be made to `shouldPrintMessage`.

### Solution 
 Below is a solution for the described logger system problem. Our implementation involves using a Python class named `Logger` that will utilize a dictionary (`self.message_map`) for tracking message timestamps. We'll ensure that a message can only be repeatedly logged once every 10 seconds.

Here is the complete Python code including comments which explain each part, suitable for direct use in a LeetCode coding environment:



```python
class Logger:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        # This dictionary will keep each message and the next allowed timestamp for it
        self.message_map = {}
    
    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        """
        Returns true if the message should be printed in the given timestamp,
        otherwise returns false.

        :type timestamp: int
        :type message: str
        :rtype: bool
        """
        # Check if message is already present in the log
        if message in self.message_map:
            if timestamp < self.message_map[message]:
                # If the current timestamp is less than the allowed minimal timestamp, return False
                return False
            else:
                # Update the allowed minimum timestamp for the message
                self.message_map[message] = timestamp + 10
                return True
        else:
            # If it's a new message, log it and set its next permissible log time
            self.message_map[message] = timestamp + 10
            return True

# Example of usage
# logger = Logger()
# print(logger.shouldPrintMessage(1, "foo"))  # returns True
# print(logger.shouldPrintMessage(2, "bar"))  # returns True
# print(logger.shouldPrintMessage(3, "foo"))  # returns False
# print(logger.shouldPrintMessage(8, "bar"))  # returns False
# print(logger.shouldPrintMessage(10, "foo")) # returns False
# print(logger.shouldPrintMessage(11, "foo")) # returns True

```

**Explanation:**

- **Constructor (`__init__`)**: Initialize an empty dictionary `message_map` to keep track of the messages and their respective next allowable timestamps.
- **Method `shouldPrintMessage`**: This method processes each incoming message along with its timestamp. It checks if the message exists in `message_map`:
  - If the message has been logged before and the current timestamp is less than the recorded timestamp, the method returns `False`.
  - If the timestamp is enough (message not present or timestamp >= next allowed timestamp), it updates the dictionary with the new allowed timestamp for the message (current timestamp + 10) and returns `True`.

This solution uses a hash map for efficient query and update operations hence resulting in an average case time complexity per operation of O(1) and a space complexity up to O(N), where N is the number of unique messages.

# 380. Insert Delete GetRandom O(1)

### Problem Description 
Implement the `RandomizedSet` class:
`RandomizedSet()` Initializes the `RandomizedSet` object.

`bool insert(int val)` Inserts an item `val` into the set if not present. Returns `true` if the item was not present, `false` otherwise.

`bool remove(int val)` Removes an item `val` from the set if present. Returns `true` if the item was present, `false` otherwise.

`int getRandom()` Returns a random element from the current set of elements (it's guaranteed that at least one element exists when this method is called). Each element must have the same probability of being returned.


Example 1:
Input
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
Output
[null, true, false, true, 2, true, false, 2]
Explanation
RandomizedSet randomizedSet = new RandomizedSet();
randomizedSet.insert(1); // Inserts 1 to the set. Returns true as 1 was inserted successfully.

randomizedSet.remove(2); // Returns false as 2 does not exist in the set.

randomizedSet.insert(2); // Inserts 2 to the set, returns true. Set now contains [1,2].

randomizedSet.getRandom(); // getRandom() should return either 1 or 2 randomly.

randomizedSet.remove(1); // Removes 1 from the set, returns true. Set now contains [2].

randomizedSet.insert(2); // 2 was already in the set, so return false.

randomizedSet.getRandom(); // Since 2 is the only number in the set, getRandom() will always return 2.


Constraints:
`-231 <= val <= 231 - 1`
At most `105` calls will be made to `insert`, `remove`, and `getRandom`.

There will be at least one element in the data structure when `getRandom` is called.

Follow up: Could you implement the functions of the class with each function works in average `O(1)` time?

### Solution 
 To solve the problem of implementing the `RandomizedSet` class with all operations being average O(1) time complexity, we will use a combination of a list and a dictionary:

1. `list` (or dynamic array in some languages): This structure will help with getting a random element efficiently.
2. `dictionary`: This allows for quick look-up, insertion, and removal of items. We will map values to their corresponding indices in the list.

Here are the crucial points for achieving O(1) operations:

- **Insert**: Add the value to the list and store the index of this value in the dictionary.
- **Remove**: Check if the value exists in the dictionary. If it's present, we find the index of this value, swap the last item of the list with this index, remove the last element (thus removing the desired value), and update the dictionary.
- **GetRandom**: Return an element at a random index in the list.

Here's the Python code implementing the mentioned operations:



```python
import random

class RandomizedSet:
    def __init__(self):
        self.values_map = {}
        self.values_list = []

    def insert(self, val):
        """ Inserts a value to the set. Returns True if the set did not already contain the specified element. """
        if val in self.values_map:
            return False
        self.values_map[val] = len(self.values_list)
        self.values_list.append(val)
        return True

    def remove(self, val):
        """ Removes a value from the set. Returns True if the set contained the specified element. """
        if val not in self.values_map:
            return False
        
        # Move the last element to the place where the element to be removed is
        last_element = self.values_list[-1]
        idx_to_remove = self.values_map[val]
        self.values_list[idx_to_remove] = last_element
        self.values_map[last_element] = idx_to_remove
        
        # Remove the last element
        self.values_list.pop()
        del self.values_map[val]
        
        return True

    def getRandom(self):
        """ Get a random element from the set. """
        return random.choice(self.values_list)

# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

```

This solution conforms to the LeetCode format and should perform efficiently based on the described requirements:
- Each function in the RandomizedSet operates in O(1) time on average as required.
- This efficient combination of a dictionary and a list addresses the concerns of fast insertion, deletion, and random access.

# 381. Insert Delete GetRandom O(1) - Duplicates allowed

### Problem Description 
Implement the `RandomizedCollection` class:
`RandomizedCollection()` Initializes the `RandomizedCollection` object.

`bool insert(int val)` Inserts an item `val` into the multiset if not present. Returns `true` if the item was not present, `false` otherwise.

`bool remove(int val)` Removes an item `val` from the multiset if present. Returns `true` if the item was present, `false` otherwise. Note that if `val` has multiple occurrences in the multiset, we only remove one of them.

`int getRandom()` Returns a random element from the current multiset of elements (it's guaranteed that at least one element exists when this method is called). The probability of each element being returned is linearly related to the number of same values the multiset contains.


Example 1:
Input
["RandomizedCollection", "insert", "insert", "insert", "getRandom", "remove", "getRandom"]
[[], [1], [1], [2], [], [1], []]
Output
[null, true, false, true, 2, true, 1]
Explanation
RandomizedCollection randomizedCollection = new RandomizedCollection();
randomizedCollection.insert(1);   // return True. Inserts 1 to the collection. Returns true as the collection did not contain 1.

randomizedCollection.insert(1);   // return False. Inserts another 1 to the collection. Returns false as the collection contained 1. Collection now contains [1,1].

randomizedCollection.insert(2);   // return True. Inserts 2 to the collection, returns true. Collection now contains [1,1,2].

randomizedCollection.getRandom(); // getRandom should return 1 with the probability 2/3, and returns 2 with the probability 1/3.

randomizedCollection.remove(1);   // return True. Removes 1 from the collection, returns true. Collection now contains [1,2].

randomizedCollection.getRandom(); // getRandom should return 1 and 2 both equally likely.


Constraints:
`-231 <= val <= 231 - 1`
At most `105` calls will be made to `insert`, `remove`, and `getRandom`.

There will be at least one element in the data structure when `getRandom` is called.

Follow up: Could you implement the functions of the class with each function works in average `O(1)` time?

### Solution 
 To create a `RandomizedCollection` class that supports average `O(1)` time complexity for `insert`, `remove`, and `getRandom` functions, we need to utilize a combination of a list and a dictionary. 

- **List**: This allows us to fetch an element in constant time when calling `getRandom`.
- **Dictionary**: This will map each element to a set of indices where this element appears in the list, supporting constant time insertions and deletions.

### Implementation Steps:
1. **Initialization**: Define a list (to store elements) and a dictionary (to map values to their indices in the list).
2. **Insert(val)**:
   - Add the value to the list and record its index in the dictionary.
   - If the val is already in the dictionary, append the new index to its set of indices.
   - If not, add it to the dictionary with a new set containing the current index.
   - Return `True` if the val was not already present (based on the size of the set before insertion), otherwise `False`.
3. **Remove(val)**:
   - Check if val is in the dictionary.
   - If so, find one of its indices from the set. Remove the value from the list by swapping it with the last element to maintain O(1) deletion efficiency, and then pop the last element.
   - Adjust the indices in the dictionary according to the swap operation.
   - If the val has no more indices left after removal, delete it from the dictionary.
   - Return `True` if the val was found and `False` otherwise.
4. **GetRandom()**:
   - Return a random element from the list using `random.choice()`, ensuring each element's probability of being chosen is proportional to its frequency in the collection.

Here is the complete Python code implementation, ready for inclusion into a LeetCode-like system:



```python
import random
import collections

class RandomizedCollection:
    def __init__(self):
        self.idx = collections.defaultdict(set)
        self.nums = []

    def insert(self, val: int) -> bool:
        self.nums.append(val)
        self.idx[val].add(len(self.nums) - 1)
        return len(self.idx[val]) == 1
    
    def remove(self, val: int) -> bool:
        if val not in self.idx or not self.idx[val]:
            return False
        # Get an arbitrary index of the element to remove
        remove_idx = self.idx[val].pop()
        last_element = self.nums[-1]
        # Move the last element to the place of the element to remove
        self.nums[remove_idx] = last_element
        self.idx[last_element].add(remove_idx)
        self.idx[last_element].remove(len(self.nums) - 1)
        
        self.nums.pop()  # Remove the last element
        if not self.idx[val]:
            del self.idx[val]
        return True
    
    def getRandom(self) -> int:
        return random.choice(self.nums)

# Your RandomizedCollection object will be instantiated and called as such:
# obj = RandomizedCollection()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

```

This code implements the `RandomizedCollection` class meeting the problem constraints and adheres to O(1) average time complexity for each operation by carefully managing the position updates in `nums` and `idx`. It should be suitable for direct submission to LeetCode-like platforms.

# 387. First Unique Character in a String

### Problem Description 
Given a string `s`, return the first non-repeating character in it and return its index. If it does not exist, return `-1`.


Example 1:
Input: s = "leetcode"
Output: 0

Example 2:
Input: s = "loveleetcode"
Output: 2

Example 3:
Input: s = "aabb"
Output: -1

Constraints:
`1 <= s.length <= 105`
`s` consists of only lowercase English letters.

### Solution 
 To solve this problem, we can employ a two-pass approach using a dictionary to keep track of character frequencies in the string and then find the first character with a frequency of one, returning its index.

Here's how we can approach this:

1. **First pass (Character Frequency Count):** Traverse through the string and store each character's frequency in a dictionary.
2. **Second pass (Find the Index):** Traverse the string again and check the frequency of each character from the dictionary. The first character that has a frequency of 1 in the dictionary is the first non-repeating character. Return its index.
3. **Edge Case:** If no non-repeating character is found after scanning the string, return `-1`.

Let's translate this plan into Python code following the typical format for solutions in LeetCode:



```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        # Dictionary to store character frequency
        char_count = {}
        
        # First pass: counting character frequencies
        for char in s:
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
        
        # Second pass: find the first unique character
        for index, char in enumerate(s):
            if char_count[char] == 1:
                return index
        
        # If no unique character found, return -1
        return -1

```

### Explanation of the Code:

1. **Class and Function Declaration:** The function `firstUniqChar` is a method of the class `Solution` which takes a string `s` as input and returns an integer.

2. **Character Counting:** `char_count` is a dictionary that keeps track of how many times each character appears in `s`. This is filled in the first loop.

3. **Identification of First Unique Character:** The second loop iterates again through the string `s`. Using `enumerate(s)`, we get both the character and its index at the same time. We check the frequency of each character in `char_count`: the first time we encounter a character with a frequency of 1, we return the index of that character.

4. **Handling the Absence of Unique Characters:** If the loop completes without returning, it implies that no character in the string has a frequency of 1. Thus, we return `-1`.

### Running Time
This solution runs in O(n) time where n is the length of the string, because it requires passing through the string twice. The first pass is to count occurrences, and the second is to find the first unique character. Each operation inside the loops takes constant time because dictionary operations (insertion and lookup) are average O(1). Space complexity is also O(n) in the worst case when all characters are different. 

This solution should be efficient and effective given the problem's constraints.

# 389. Find the Difference

### Problem Description 
You are given two strings `s` and `t`.

String `t` is generated by random shuffling string `s` and then add one more letter at a random position.

Return the letter that was added to `t`.


Example 1:
Input: s = "abcd", t = "abcde"
Output: "e"
Explanation: 'e' is the letter that was added.


Example 2:
Input: s = "", t = "y"
Output: "y"

Example 3:
Input: s = "a", t = "aa"
Output: "a"

Example 4:
Input: s = "ae", t = "aea"
Output: "a"

Constraints:
`0 <= s.length <= 1000`
`t.length == s.length + 1`
`s` and `t` consist of lower-case English letters.

### Solution 
 To solve this problem, we can use the following approach:

1. Use a frequency count mechanism to record the occurrence of each letter in both `s` and `t`.
2. Initialize a dictionary to store the frequency of letters in the string `s`.
3. Traverse through string `t`. If a character is found in the dictionary (i.e., it was present in `s`), decrement its count. If the count drops below zero or if the character wasn't in `s`, that is the added character.

This approach ensures that we handle cases where there are multiple identical letters and pinpoint exactly the one which is extra in `t`.

Now, let's implement this solution.



```python
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        # Create a dictionary to store the frequency of letters in `s`
        freq = {}
        for char in s:
            if char in freq:
                freq[char] += 1
            else:
                freq[char] = 1
                
        # Traverse through string `t` and update frequencies or detect the new letter
        for char in t:
            if char in freq:
                freq[char] -= 1
                if freq[char] < 0:
                    return char
            else:
                return char

# You can test the code using the following lines:
# sol = Solution()
# print(sol.findTheDifference("abcd", "abcde"))  # Output: "e"
# print(sol.findTheDifference("", "y"))          # Output: "y"
# print(sol.findTheDifference("a", "aa"))        # Output: "a"
# print(sol.findTheDifference("ae", "aea"))      # Output: "a"

```

This script defines a `Solution` class with the method `findTheDifference` as expected by LeetCode's format. The method uses a dictionary to count the frequency of each character in `s` and then determines the extra character in `t` by checking the frequencies. This solution has a time complexity of O(n), with n being the length of `t`.

# 409. Longest Palindrome

### Problem Description 
Given a string `s` which consists of lowercase or uppercase letters, return the length of the longest palindrome that can be built with those letters.

Letters are case sensitive, for example, `"Aa"` is not considered a palindrome here.


Example 1:
Input: s = "abccccdd"
Output: 7
Explanation:
One longest palindrome that can be built is "dccaccd", whose length is 7.


Example 2:
Input: s = "a"
Output: 1

Example 3:
Input: s = "bb"
Output: 2

Constraints:
`1 <= s.length <= 2000`
`s` consists of lowercase and/or uppercase English letters only.

### Solution 
 Below is a Python solution for the described problem. This solution uses Python's `collections.Counter` to count the frequency of each character in the input string. It then processes these counts to determine how many characters can be perfectly paired to contribute to the palindrome; it also checks if there's at least one character with an odd count to use it as the center of the palindrome if needed. Here is the solution:



```python
from collections import Counter

class Solution:
    def longestPalindrome(self, s: str) -> int:
        # Counter will store the frequency of each character
        char_counts = Counter(s)
        length = 0
        odd_found = False
        
        # Iterate over the frequencies of each unique character
        for count in char_counts.values():
            # If the count is even, add it to the length
            length += count if count % 2 == 0 else count - 1
            
            # Check if there is at least one character with an odd count
            if count % 2 == 1:
                odd_found = True
        
        # If there is an odd count, add 1 to length for possible center use
        if odd_found:
            length += 1
        
        return length

# Example test cases
sol = Solution()
print(sol.longestPalindrome("abccccdd"))  # Output: 7
print(sol.longestPalindrome("a"))         # Output: 1
print(sol.longestPalindrome("bb"))        # Output: 2

```

### Explanation:

1. **Initialize Counter:** We use `Counter` from `collections` to tally up the occurrences of each character.
2. **Iterating over Counts:** For each unique character's frequency:
   - **Even Counts:** Add the full count to the palindrome length as these characters can be perfectly paired.
   - **Odd Counts:** Add `count - 1` to use the maximum even count of characters and keep track of the presence of odd characters.
3. **Central Character:** After processing all characters, if there was any character with an odd count, add 1 to the palindrome length since one odd-character can sit at the center of the palindrome.
4. **Return the Result:** The final computed `length` is returned as the answer.

This solution efficiently computes the result in linear time, O(n), where n is the length of the string and uses extra space proportional to the number of unique characters, which in the worst case with both lowercase and uppercase letters is constant, O(1). This solution is optimized for handling cases even with the maximum constraint, by ensuring that operations are based on character counts.

# 438. Find All Anagrams in a String

### Problem Description 
Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.

Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100.

The order of output does not matter.


Example 1:
Input:
s: "cbaebabacd" p: "abc"
Output:
[0, 6]
Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".

The substring with start index = 6 is "bac", which is an anagram of "abc".


Example 2:
Input:
s: "abab" p: "ab"
Output:
[0, 1, 2]
Explanation:
The substring with start index = 0 is "ab", which is an anagram of "ab".

The substring with start index = 1 is "ba", which is an anagram of "ab".

The substring with start index = 2 is "ab", which is an anagram of "ab".

### Solution 
 The problem is essentially about finding all the positions in the string `s` where any permutation of the string `p` appears as a substring. We can solve this problem effectively using the sliding window technique along with a frequency count approach.

Here's how we can approach it:
1. **Character Frequency of `p`**: First, calculate the frequency of each character in `p`. This can be stored in a fixed-size array since we're only dealing with lowercase English letters (array size of 26).
2. **Sliding Window**: We then use a sliding window of the length of `p` to traverse through `s`. For each window, we'll also maintain a frequency count of characters within the window.
3. **Check for Anagram**: Compare the frequency count of the window with `p`'s frequency count. If they are the same, then the substring of `s` within the current window is an anagram of `p`.
4. **Update Window**: Slide the window one character to the right, update the frequency counts (remove the character that's no longer in the window, and add the new character), and repeat the check.

The implementation in Python following the required LeetCode format would look like this:



```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        if len(p) > len(s):
            return []

        # Resultant indices of start of anagrams
        anagram_starts = []
        
        # Frequency arrays
        p_count = [0] * 26
        s_count = [0] * 26
        
        # Build frequency array for p
        for char in p:
            p_count[ord(char) - ord('a')] += 1
        
        # Initialize the frequency array for the first window of s
        for i in range(len(p)):
            s_count[ord(s[i]) - ord('a')] += 1
        
        # Slide over the remaining windows of s
        for i in range(len(s) - len(p)):
            if s_count == p_count:
                anagram_starts.append(i)
            
            # Update counts to slide the window
            s_count[ord(s[i]) - ord('a')] -= 1
            s_count[ord(s[i + len(p)]) - ord('a')] += 1
        
        # Check the last window
        if s_count == p_count:
            anagram_starts.append(len(s) - len(p))
        
        return anagram_starts

```

This solution assumes the helper `ord()` function to convert characters into integer values, and effectively slides the window across the string `s`, updating counts at each step and comparing with the counts of `p`. This results in a time complexity of O(n), which is efficient for the problem constraints.

# 447. Number of Boomerangs

### Problem Description 
You are given `n` `points` in the plane that are all distinct, where `points[i] = [xi, yi]`. A boomerang is a tuple of points `(i, j, k)` such that the distance between `i` and `j` equals the distance between `i` and `k` (the order of the tuple matters).

Return the number of boomerangs.


Example 1:
Input: points = [[0,0],[1,0],[2,0]]
Output: 2
Explanation: The two boomerangs are [[1,0],[0,0],[2,0]] and [[1,0],[2,0],[0,0]].


Example 2:
Input: points = [[1,1],[2,2],[3,3]]
Output: 2

Example 3:
Input: points = [[1,1]]
Output: 0

Constraints:
`n == points.length`
`1 <= n <= 500`
`points[i].length == 2`
`-104 <= xi, yi <= 104`
All the points are unique.

### Solution 
 To solve this problem in Python, we need to determine the number of boomerangs in a list of 2D points. A boomerang is identified as a combination of three points where one point is equidistant to the other two, and these two are distinct from each other.

### Approach
1. **Using a Dictionary**: For each point `i`, use a dictionary to map the distance from point `i` to every other point to the number of points at that distance.
2. For each distance, if we have `count` points at that distance, the number of potential boomerangs that use point `i` as the central point is combination logic — we choose two points from those at the same distance (`count * (count - 1)`). This accounts for two selections which represent (i, j, k) and (i, k, j).
3. Add the results to a running total and ultimately return this value.

### Key Considerations
- Calculate distance using the squared distance formula `(x2 - x1)^2 + (y2 - y1)^2` to avoid floating-point operations.
- Process each point `i` as a potential center, iterating over other points `j` to compile distances.
- A dictionary will help tally distances effectively.
- Edge cases include arrays with less than three points where no boomerangs can be formed.

### Python Code



```python
from collections import defaultdict
from typing import List

class Solution:
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        def squared_distance(p1, p2):
            return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

        res = 0
        for i in points:
            distances = defaultdict(int)
            
            for j in points:
                if i != j:
                    distance = squared_distance(i, j)
                    distances[distance] += 1
            
            for count in distances.values():
                res += count * (count - 1)  # number of boomerang pairs
                
        return res

# Example usage:
points = [[0,0],[1,0],[2,0]]
sol = Solution()
print(sol.numberOfBoomerangs(points))  # Output: 2

```

### Explanation of the code

- **squared_distance**: A helper function to compute squared distance to prevent floating point inaccuracies.
- **distances**: A local dictionary capturing distances from the current point `i` to all other points.
- We loop over each point `i` as potentially the middle of a boomerang, building up the dictionary of distances using points `j`.
- For each unique distance in the dictionary, we calculate the potential number of boomerang pairs using the formula `count * (count - 1)`.
- We aggregate the total boomerang count and return it.

This solution is efficient, considering up to \( O(n^2) \) points comparisons, which is manageable given the constraints.

# 451. Sort Characters By Frequency

### Problem Description 
Given a string, sort it in decreasing order based on the frequency of characters.


Example 1:
Input:
"tree"
Output:
"eert"
Explanation:
'e' appears twice while 'r' and 't' both appear once.

So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.


Example 2:
Input:
"cccaaa"
Output:
"cccaaa"
Explanation:
Both 'c' and 'a' appear three times, so "aaaccc" is also a valid answer.

Note that "cacaca" is incorrect, as the same characters must be together.


Example 3:
Input:
"Aabb"
Output:
"bbAa"
Explanation:
"bbaA" is also a valid answer, but "Aabb" is incorrect.

Note that 'A' and 'a' are treated as two different characters.

### Solution 
 To solve this problem using Python on LeetCode, we need to count the occurrences of each character in the input string, then sort these characters based on their frequencies in descending order. Finally, we construct the resulting string based on these sorted results, respecting the frequencies.

Here's a Python implementation optimized for clarity and efficiency, which aligns with the LeetCode code submission format:



```python
from collections import Counter
import heapq

class Solution:
    def frequencySort(self, s: str) -> str:
        # Step 1: Count the characters in the string
        counter = Counter(s)
        
        # Step 2: Create a max-heap based on the frequency of characters.
        # We use negative frequency to simulate a max-heap in Python, as Python's heapq is a min-heap by default.
        max_heap = [(-freq, char) for char, freq in counter.items()]
        heapq.heapify(max_heap)
        
        # Step 3: Build the result string by extracting elements from the max-heap.
        result = []
        while max_heap:
            freq, char = heapq.heappop(max_heap)
            result.append(char * (-freq))  # Multiply the character by its frequency.
        
        return ''.join(result)

# The following is example usage and can be used for testing:
if __name__ == "__main__":
    sol = Solution()
    print(sol.frequencySort("tree"))  # Output: "eert" or "eetr"
    print(sol.frequencySort("cccaaa"))  # Output: "cccaaa" or "aaaccc"
    print(sol.frequencySort("Aabb"))  # Output: "bbAa" or "bbaA"

```

**Explanation:**
1. **Counting Characters:** We use the `Counter` from the `collections` module to count the frequency of each character in the string.
2. **Heap for Sorting by Frequency:** Convert these frequencies into a heap structure. Python's `heapq` module provides a min-heap, so we store negative frequencies to simulate a max-heap, which sorts the characters by their frequencies in descending order.
3. **Reconstruct the Result String:** We extract elements from this heap to construct the result string. Each character is multiplied by its frequency (correcting the sign of the frequency) to reconstruct its repeated occurrence.

This solution uses the heap for efficient frequency-based sorting and handles large inputs effectively by prioritizing high-frequency characters. Each step is designed for clarity and adhering to the problem's requirements.

