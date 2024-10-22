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
 To solve the problem of finding the length of the longest substring without repeating characters, we can employ the sliding window technique combined with a hash set to keep track of characters that we have seen so far in the current substring. 

The sliding window will help us maintain the current substring's boundaries, expanding it when we find a new character and contracting it when we encounter a character that is already in the current substring. This way, we can efficiently find the longest substring without repeating characters.

Here's the step-by-step explanation followed by the implementation:

### Step-by-Step Explanation:

1. **Initialize Pointers and Data Structures**:
   - Use two pointers: `left` to represent the beginning of the sliding window and `right` to represent the end.
   - Use a set `char_set` to keep track of the characters in the current window.
   - Maintain a variable `max_length` to store the length of the longest substring found.

2. **Expand the Window**:
   - Iterate through the string with the `right` pointer. For each character:
     - If the character is not in the `char_set`, add it to the `char_set` and update `max_length` if the current window size (`right - left + 1`) is greater than `max_length`.
     - If the character is already in `char_set`, it means we have encountered a duplicate.
       - We need to shrink the window from the left by moving the `left` pointer to the right until we remove the duplicate character from the `char_set`.

3. **Return the Result**:
   - After processing the string, the value of `max_length` will represent the length of the longest substring without repeating characters.

### Implementation in Python:



```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        char_set = set()
        left = 0
        max_length = 0
        
        for right in range(len(s)):
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1
            char_set.add(s[right])
            max_length = max(max_length, right - left + 1)
        
        return max_length

```

### Explanation of the Code:

- We define a class `Solution` with a method `lengthOfLongestSubstring` which takes a string `s` as input.
- We initialize an empty set `char_set` to track characters in the current substring, and two pointers: `left` and `right`.
- As we iterate over the string using the `right` pointer, we check if the current character (i.e., `s[right]`) is in `char_set`.
  - If it is, we enter a while loop to remove characters from the `left` pointer until the duplicate character is removed.
  - We then add the new character to `char_set` and calculate the length of the current substring (`right - left + 1`) and update `max_length` if it is greater than the current `max_length`.
- Finally, we return `max_length` which contains the length of the longest substring without repeating characters.

This solution runs in O(n) time complexity, where n is the length of the string `s`, and it uses O(min(n, m)) space, where m is the size of the character set used (for this problem, it would be the English letters, digits, symbols, and spaces).

# 5. Longest Palindromic Substring

### Problem Description 
Given a string `s`, return the longest palindromic substring in `s`.


Example 1:
Input: s = "babad"
Output: "bab"
Note: "aba" is also a valid answer.


Example 2:
Input: s = "cbbd"
Output: "bb"

Example 3:
Input: s = "a"
Output: "a"

Example 4:
Input: s = "ac"
Output: "a"

Constraints:
`1 <= s.length <= 1000`
`s` consist of only digits and English letters (lower-case and/or upper-case),

### Solution 
 To solve the problem of finding the longest palindromic substring in a given string `s`, we can use a technique known as "expand around center". This approach is efficient and can be implemented with a time complexity of O(n^2), which is acceptable given the constraints of the problem (1 <= s.length <= 1000).

### Explanation:

A palindrome is a sequence that reads the same backward as forward. For example, "aba" and "bb" are palindromic strings.

To find the longest palindromic substring, we can consider each character (and the gap between characters) as a potential center for palindromes and expand outward as long as we encounter matching characters on both sides:

1. **Expand Around Center**: For each character in the string (and the gaps between characters), we attempt to expand outwards while the characters at both ends match.
   
2. **Two Cases**:
   - Odd-length palindromes have a single center character (e.g., "aba").
   - Even-length palindromes have two identical center characters (e.g., "abba").

3. **Updating the Longest Palindrome**: We keep track of the longest palindrome found during our expansions and update our result whenever we find a longer one.

### Python Code Implementation:

The implementation adheres to the format used on LeetCode, which requires us to define a function within a class. Here’s the implementation:



```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        # Function to expand around center and return the longest palindrome
        def expand_around_center(left: int, right: int) -> str:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            # Return the palindrome substring
            return s[left + 1:right]
        
        longest = ""
        
        for i in range(len(s)):
            # Check for odd-length palindromes
            palindrome1 = expand_around_center(i, i)
            # Check for even-length palindromes
            palindrome2 = expand_around_center(i, i + 1)
            
            # Update longest if the found palindromes are longer
            longest = max(longest, palindrome1, palindrome2, key=len)
        
        return longest

```

### How the Code Works:
- The `longestPalindrome` method takes a string `s` as input.
- We define an inner function `expand_around_center` that attempts to find the longest palindromic substring by expanding around a potential center (given by the indices `left` and `right`).
- We loop through each index in `s` twice: once for odd-length palindromes and once for even-length ones.
- After each expansion, we compare the lengths of the newly found palindromes with the current longest palindrome and update accordingly.
- Finally, we return the longest palindromic substring found.

### Example Explained:
For the input `s = "babad"`, the function will:
1. Check `b`, expand around it to find "bab" and update the longest palindrome.
2. Check `a`, expand around it to find "aba", also updates the longest palindrome.
3. Next, check `b` again and find "bab", but won't update since it's of the same length.
4. The second `b` will expand to find "bb" but won't exceed the length of "bab" or "aba".
5. Eventually, it will return "bab" or "aba" as the longest palindromic substring. 

### Conclusion:
This solution efficiently finds the longest palindromic substring by considering every character as a potential center and expanding outward, resulting in an O(n^2) time complexity while being effective for the given constraints.

# 6. ZigZag Conversion

### Problem Description 
The string `"PAYPALISHIRING"` is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)
P   A   H   N
A P L S I I G
Y   I   R
And then read line by line: `"PAHNAPLSIIGYIR"`
Write the code that will take a string and make this conversion given a number of rows:
string convert(string s, int numRows);

Example 1:
Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"

Example 2:
Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:
P     I    N
A   L S  I G
Y A   H R
P     I

Example 3:
Input: s = "A", numRows = 1
Output: "A"

Constraints:
`1 <= s.length <= 1000`
`s` consists of English letters (lower-case and upper-case), `','` and `'.'`.

`1 <= numRows <= 1000`

### Solution 
 To solve the problem of converting a string into a zigzag pattern based on a given number of rows, we need to visualize how the characters are arranged in a zigzag pattern. This means we will loop through the string and place each character into the correct position in a row based on the specified number of rows.

### Steps to Solve the Problem:
1. **Create an empty list for each row**: We'll have a list to capture characters for each row in the zigzag pattern.
2. **Loop through the string**: As we iterate over the characters in the string, we need to decide which row the character should go into. This will depend on whether we are moving down the rows or up the rows.
3. **Toggle the direction**: Once we reach the last row, we start moving up, and once we reach the first row, we start moving down.
4. **Join rows**: After placing all characters in their respective rows, we simply join those lists into a single string as the final output.

### Python Code Implementation:
Here's how you can implement this in Python for the LeetCode environment:



```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        # Edge case where the number of rows is 1 or greater than or equal to the length of the string.
        if numRows == 1 or numRows >= len(s):
            return s
        
        # Create an array of strings for each row
        rows = [''] * numRows
        cur_row = 0
        going_down = False
        
        # Build the zigzag pattern
        for char in s:
            rows[cur_row] += char  # Add the current character to the current row
            # Determine if we need to go up or down
            if cur_row == 0:
                going_down = True  # Start going down
            elif cur_row == numRows - 1:
                going_down = False  # Start going up

            # Move to the next row
            cur_row += 1 if going_down else -1
        
        # Join all rows to form the final string
        return ''.join(rows)


```

### Explanation:
- **Initialization**: We start by checking if the number of rows is 1 or if it is greater than or equal to the length of the string. In such cases, we can directly return the string as is since no zigzagging happens.
- **Rows Storage**: We prepare a list `rows` where each index corresponds to characters in one row of the zigzag.
- **Character Placement**: We iterate through each character in the string:
  - We add the current character to the appropriate row.
  - We check if we have reached the topmost or bottommost row and toggle the direction accordingly.
  - We update the `cur_row` index based on whether we are going down or up.
- **Final String**: Finally, we join all the strings in `rows` into a single string, which represents the zigzag readout.

### Complexity Analysis:
- The time complexity is \(O(n)\), where \(n\) is the length of the input string, because we iterate through the string once.
- The space complexity is \(O(n)\) as we store the strings in a list.

This implementation should be efficient and work within the constraints provided. You can run this code directly on the LeetCode platform under their provided settings for the problem.

# 8. String to Integer (atoi)

### Problem Description 
Implement the `myAtoi(string s)` function, which converts a string to a 32-bit signed integer (similar to C/C++'s `atoi` function).

The algorithm for `myAtoi(string s)` is as follows:
Read in and ignore any leading whitespace.

Check if the next character (if not already at the end of the string) is `'-'` or `'+'`. Read this character in if it is either. This determines if the final result is negative or positive respectively. Assume the result is positive if neither is present.

Read in next the characters until the next non-digit charcter or the end of the input is reached. The rest of the string is ignored.

Convert these digits into an integer (i.e. `"123" -> 123`, `"0032" -> 32`). If no digits were read, then the integer is `0`. Change the sign as necessary (from step 2).

If the integer is out of the 32-bit signed integer range `[-231, 231 - 1]`, then clamp the integer so that it remains in the range. Specifically, integers less than `-231` should be clamped to `-231`, and integers greater than `231 - 1` should be clamped to `231 - 1`.

Return the integer as the final result.

Note:
Only the space character `' '` is considered a whitespace character.

Do not ignore any characters other than the leading whitespace or the rest of the string after the digits.


Example 1:
Input: s = "42"
Output: 42
Explanation: The underlined characters are what is read in, the caret is the current reader position.

Step 1: "42" (no characters read because there is no leading whitespace)
         ^
Step 2: "42" (no characters read because there is neither a '-' nor '+')
         ^
Step 3: "42" ("42" is read in)
           ^
The parsed integer is 42.

Since 42 is in the range [-231, 231 - 1], the final result is 42.


Example 2:
Input: s = "   -42"
Output: -42
Explanation:
Step 1: "   -42" (leading whitespace is read and ignored)
            ^
Step 2: "   -42" ('-' is read, so the result should be negative)
             ^
Step 3: "   -42" ("42" is read in)
               ^
The parsed integer is -42.

Since -42 is in the range [-231, 231 - 1], the final result is -42.


Example 3:
Input: s = "4193 with words"
Output: 4193
Explanation:
Step 1: "4193 with words" (no characters read because there is no leading whitespace)
         ^
Step 2: "4193 with words" (no characters read because there is neither a '-' nor '+')
         ^
Step 3: "4193 with words" ("4193" is read in; reading stops because the next character is a non-digit)
             ^
The parsed integer is 4193.

Since 4193 is in the range [-231, 231 - 1], the final result is 4193.


Example 4:
Input: s = "words and 987"
Output: 0
Explanation:
Step 1: "words and 987" (no characters read because there is no leading whitespace)
         ^
Step 2: "words and 987" (no characters read because there is neither a '-' nor '+')
         ^
Step 3: "words and 987" (reading stops immediately because there is a non-digit 'w')
         ^
The parsed integer is 0 because no digits were read.

Since 0 is in the range [-231, 231 - 1], the final result is 0.


Example 5:
Input: s = "-91283472332"
Output: -2147483648
Explanation:
Step 1: "-91283472332" (no characters read because there is no leading whitespace)
         ^
Step 2: "-91283472332" ('-' is read, so the result should be negative)
          ^
Step 3: "-91283472332" ("91283472332" is read in)
                     ^
The parsed integer is -91283472332.

Since -91283472332 is less than the lower bound of the range [-231, 231 - 1], the final result is clamped to -231 = -2147483648. 

Constraints:
`0 <= s.length <= 200`
`s` consists of English letters (lower-case and upper-case), digits (`0-9`), `' '`, `'+'`, `'-'`, and `'.'`.

### Solution 
 To implement the `myAtoi` function as described in the problem statement, we will follow a structured approach to parse out an integer from a given string while handling whitespace, signs, and non-digit characters correctly. Below, I will provide the complete implementation with a detailed explanation for each step.

Here is the implementation:



```python
class Solution:
    def myAtoi(self, s: str) -> int:
        # Step 1: Read in and ignore leading whitespace
        s = s.lstrip()
        
        if not s:
            return 0  # Edge case: if the string is empty after trimming

        # Step 2: Check for the sign
        sign = 1
        index = 0
        if s[index] == '-':
            sign = -1
            index += 1
        elif s[index] == '+':
            index += 1

        # Step 3: Read in the digits
        num = 0
        while index < len(s) and s[index].isdigit():
            digit = int(s[index])
            num = num * 10 + digit
            index += 1
            
        # Step 4: Apply the sign
        num *= sign

        # Step 5: Clamp the result to the 32-bit signed integer range
        int_min = -2**31
        int_max = 2**31 - 1
        if num < int_min:
            return int_min
        if num > int_max:
            return int_max
        
        return num

```

### Explanation of the Code:

1. **Leading Whitespace Removal**: 
   - We use the `lstrip()` method, which removes any whitespace from the beginning of the string. If the string becomes empty after this operation, we immediately return `0` since no digits were read.

2. **Sign Handling**:
   - We initialize a variable `sign` to `1`, assuming a positive integer unless we encounter a `'-'` or `'+'` character.
   - The `index` variable is used to keep track of our current position in the string. If we see a `'-'`, we set `sign` to `-1` and move to the next character in the string. In the case of a `'+'`, we just move the index forward without changing the sign.

3. **Digit Extraction**:
   - We initialize `num` to `0`, which will hold our resulting integer.
   - We enter a `while` loop that continues as long as we have not reached the end of the string (`index < len(s)`) and the current character is a digit (`s[index].isdigit()`).
   - Inside the loop, we convert the current character to an integer and build the number by multiplying the existing `num` by `10` (to shift left) and adding the new digit. We then increment the index to move to the next character.

4. **Applying the Sign**:
   - After reading all possible digits, we apply the sign to our number by multiplying `num` with `sign`.

5. **Clamping the Result**:
   - We define constants for the minimum and maximum values of a 32-bit signed integer: `-2^31` (which is `-2147483648`) and `2^31 - 1` (which is `2147483647`).
   - Finally, we check if our result `num` is less than the minimum value or greater than the maximum value. If so, we return the respective clamped value.

### Usage:
You can use the `myAtoi` method from the `Solution` class by creating an instance of `Solution` and calling the method with a string argument. This implementation follows the design and constraints specified in the problem statement and can be run directly on the LeetCode platform.

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
 To solve the problem of regular expression matching with support for `'.'` and `'*'`, we can employ dynamic programming. The approach involves constructing a 2D table `dp` where `dp[i][j]` will be `True` if the first `i` characters of the string `s` match the first `j` characters of the pattern `p`.

Here's a breakdown of the approach:

1. **Initialization**: 
   - Create a 2D Boolean array `dp` of size `(len(s) + 1) x (len(p) + 1)`.
   - `dp[0][0]` is `True` since an empty pattern matches an empty string.
   - Fill the first row of the `dp` table where the pattern can match an empty string, particularly when `'*'` is involved. For instance, the pattern `"a*"` can match an empty string, so we continue to fill this row based on the presence of `'*'`.

2. **Filling the DP Table**:
   - For each character in `s` and `p`, we check two main conditions:
     - If the current characters `s[i-1]` and `p[j-1]` match, or if the current character in `p` is `'.'`, then we inherit the value from the `dp[i-1][j-1]` (i.e., without these characters).
     - If the current character in `p` is `'*'`, we need to consider two situations:
       - Ignoring the `'*'` and its preceding character by looking at `dp[i][j-2]`.
       - If the preceding character in `p` matches `s[i-1]` or is `'.'`, we can consider this `'*'` as matching one or more of that character by looking at `dp[i-1][j]`.

3. **Final Result**: The value at `dp[len(s)][len(p)]` gives the final matching result.

Here’s the implementation of the above logic in Python, structured according to LeetCode's requirements:



```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        # Initialize the DP table
        dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
        
        # Empty pattern matches empty string
        dp[0][0] = True
        
        # Fill the first row for patterns like a*, a*b*, a*b*c*
        for j in range(1, len(p) + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]
        
        # Fill the DP table
        for i in range(1, len(s) + 1):
            for j in range(1, len(p) + 1):
                if p[j - 1] == s[i - 1] or p[j - 1] == '.':
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 2]  # Case of zero occurrence
                    if p[j - 2] == s[i - 1] or p[j - 2] == '.':
                        dp[i][j] = dp[i][j] or dp[i - 1][j]  # Case of one or more occurrences

        # The result for the full string and pattern is in the bottom-right cell
        return dp[len(s)][len(p)]

```

### Explanation of the Implementation:
- We define a class `Solution` and a method `isMatch` that takes in the string `s` and the pattern `p`.
- A 2D array `dp` is initialized to keep track of matched states.
- The first row is populated based on the patterns that can match an empty string.
- We iterate over each character in both `s` and `p`, updating our `dp` table based on the defined conditions.
- Finally, we return whether the entire string matches the entire pattern by checking `dp[len(s)][len(p)]`. 

This solution efficiently determines if the string matches the given pattern using dynamic programming, and performs well within the constraints.

# 12. Integer to Roman

### Problem Description 
Roman numerals are represented by seven different symbols: `I`, `V`, `X`, `L`, `C`, `D` and `M`.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, `2` is written as `II` in Roman numeral, just two one's added together. `12` is written as `XII`, which is simply `X + II`. The number `27` is written as `XXVII`, which is `XX + V + II`.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not `IIII`. Instead, the number four is written as `IV`. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as `IX`. There are six instances where subtraction is used:
`I` can be placed before `V` (5) and `X` (10) to make 4 and 9. 
`X` can be placed before `L` (50) and `C` (100) to make 40 and 90. 
`C` can be placed before `D` (500) and `M` (1000) to make 400 and 900.

Given an integer, convert it to a roman numeral.


Example 1:
Input: num = 3
Output: "III"

Example 2:
Input: num = 4
Output: "IV"

Example 3:
Input: num = 9
Output: "IX"

Example 4:
Input: num = 58
Output: "LVIII"
Explanation: L = 50, V = 5, III = 3.


Example 5:
Input: num = 1994
Output: "MCMXCIV"
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.


Constraints:
`1 <= num <= 3999`

### Solution 
 Certainly! Below is a Python code implementation to convert an integer to a Roman numeral that adheres to the required format for LeetCode submissions.

### Python Code Implementation



```python
class Solution:
    def intToRoman(self, num: int) -> str:
        # Define the mapping of integer values to Roman numeral symbols
        val = [
            1000, 900, 500, 400,
            100, 90, 50, 40,
            10, 9, 5, 4,
            1
        ]
        syms = [
            "M", "CM", "D", "CD",
            "C", "XC", "L", "XL",
            "X", "IX", "V", "IV",
            "I"
        ]
        
        # The result Roman numeral string
        roman_num = ""
        
        # Process each value-symbol pair
        for i in range(len(val)):
            # While num is greater than or equal to val[i]
            while num >= val[i]:
                # Append the corresponding symbol to the result
                roman_num += syms[i]
                # Reduce num by the value of the symbol
                num -= val[i]
        
        return roman_num

```

### Explanation

1. **Class Definition**: The code defines a class called `Solution`, which contains a method `intToRoman`.

2. **Method Declaration**: Inside the `Solution` class, `intToRoman` is defined with a parameter `num`, which is the integer you want to convert to a Roman numeral.

3. **Value and Symbol Arrays**:
   - The `val` list holds the integers in descending order, corresponding to their values in Roman numerals.
   - The `syms` list holds the corresponding Roman numeral symbols.
   - The arrays are aligned such that `val[i]` corresponds to `syms[i]`.

4. **Roman Numeral Construction**:
   - The variable `roman_num` is initialized as an empty string to accumulate the result.
   - A for loop iterates through each value in `val`.
   - Inside this loop, a while loop checks if `num` is greater than or equal to the current value `val[i]`. If it is, the associated Roman numeral `syms[i]` is appended to `roman_num`, and `num` is reduced by `val[i]`.

5. **Return Statement**: Once the conversion is complete, the method returns the constructed Roman numeral string `roman_num`.

### Complexity Analysis
- **Time Complexity**: O(M), where M is the number of Roman numeral symbols (which is constant and limited), hence it effectively operates in constant time.
- **Space Complexity**: O(1), as no extra space is used that grows with input size aside from the output string.

With this implementation, users can input an integer from 1 to 3999 to receive the correct Roman numeral representation directly on LeetCode.

# 13. Roman to Integer

### Problem Description 
Roman numerals are represented by seven different symbols: `I`, `V`, `X`, `L`, `C`, `D` and `M`.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, `2` is written as `II` in Roman numeral, just two one's added together. `12` is written as `XII`, which is simply `X + II`. The number `27` is written as `XXVII`, which is `XX + V + II`.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not `IIII`. Instead, the number four is written as `IV`. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as `IX`. There are six instances where subtraction is used:
`I` can be placed before `V` (5) and `X` (10) to make 4 and 9. 
`X` can be placed before `L` (50) and `C` (100) to make 40 and 90. 
`C` can be placed before `D` (500) and `M` (1000) to make 400 and 900.

Given a roman numeral, convert it to an integer.


Example 1:
Input: s = "III"
Output: 3

Example 2:
Input: s = "IV"
Output: 4

Example 3:
Input: s = "IX"
Output: 9

Example 4:
Input: s = "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.


Example 5:
Input: s = "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.


Constraints:
`1 <= s.length <= 15`
`s` contains only the characters `('I', 'V', 'X', 'L', 'C', 'D', 'M')`.

It is guaranteed that `s` is a valid roman numeral in the range `[1, 3999]`.

### Solution 
 Here's a Python solution following the LeetCode format to convert Roman numerals to integers. In this solution, we'll iterate through the string of Roman numerals and apply the rules for conversion while taking care of the subtraction cases.

### Python Code



```python
class Solution:
    def romanToInt(self, s: str) -> int:
        # Mapping of Roman numerals to their integer values
        roman_to_int = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }

        total = 0  # Variable to store the total value
        prev_value = 0  # Variable to store the previous numeral value

        # Iterate over each character in the Roman numeral string
        for char in reversed(s):
            # Get the integer value of the current Roman numeral
            current_value = roman_to_int[char]
            
            # Determine whether to add or subtract the current value based on the previous value
            if current_value < prev_value:
                total -= current_value  # Subtract for special cases (e.g., IV, IX)
            else:
                total += current_value  # Otherwise, add to the total
            
            # Update the previous value to the current for the next iteration
            prev_value = current_value
        
        return total

```

### Explanation

1. **Mapping Roman Symbols to Integers**: 
   We define a dictionary, `roman_to_int`, which maps each Roman numeral to its corresponding integer value. This will help us look up values easily.

2. **Initialization**:
   - `total`: This variable will accumulate the total value as we process the Roman numeral.
   - `prev_value`: This will keep track of the value of the previously processed Roman numeral to identify if we are in a subtraction case.

3. **Iterating in Reverse**:
   We iterate over the string `s` in reverse order. The reason for this is that Roman numerals are usually written from largest to smallest, but there are specific cases (like 'IV' for 4) where a smaller numeral appears before a larger one. By processing from right to left, we can easily identify when to add or subtract values.

4. **Updating Total**:
   - If the `current_value` (current Roman numeral’s value) is less than `prev_value` (the value we've just seen), we subtract `current_value` from `total`. This accounts for the Roman numeral subtraction rules (e.g., 'IV', 'IX').
   - If not, we add `current_value` to `total`.

5. **Return Total**:
   Finally, we return the computed `total`, which now represents the integer value of the Roman numeral.

### Complexity
- **Time Complexity**: O(n), where n is the length of the input string `s`, since we have to process each character in the worst case.
- **Space Complexity**: O(1), since the space used by the dictionary is constant and does not depend on the input size.

### Usage
This code can be directly copied and pasted into a solution submission on the LeetCode website for the specified problem. Just ensure it is placed within a class named `Solution` with the method `romanToInt` as required by LeetCode's submission format.

# 14. Longest Common Prefix

### Problem Description 
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string `""`.


Example 1:
Input: strs = ["flower","flow","flight"]
Output: "fl"

Example 2:
Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.


Constraints:
`0 <= strs.length <= 200`
`0 <= strs[i].length <= 200`
`strs[i]` consists of only lower-case English letters.

### Solution 
 To solve the problem of finding the longest common prefix among an array of strings, we can utilize a straightforward approach that compares the characters of the strings sequentially until we find a mismatch.

Here is a step-by-step explanation of the algorithm:

### Steps:

1. **Edge Case Handling**:
   - If the `strs` list is empty, we immediately return an empty string `""`.
   - If there’s only one string, we return that string as it is the common prefix itself.

2. **Initial Prefix**:
   - We start by taking the first string as the initial prefix since any common prefix must definitely include the start of the first string.

3. **Iterate Through the List**:
   - For each string in the array after the first one, we check how much of the current prefix matches with the current string.
   - We continue to reduce this prefix until it matches the start of the current string or becomes empty.

4. **Character Comparison**:
   - We compare characters one by one, and when we find a mismatch, we truncate the prefix to that point.

5. **Final Result**:
   - Once we’ve iterated through all strings, we return the final value of the prefix.

Here is the implementation of the algorithm in Python:



```python
class Solution:
    def longestCommonPrefix(self, strs):
        if not strs:
            return ""
        
        # Start with the first string as the assumed longest common prefix
        prefix = strs[0]
        
        for s in strs[1:]:
            # Compare the current prefix with the next string
            while not s.startswith(prefix):
                # Reduce the prefix by one character from the end
                prefix = prefix[:-1]
                if not prefix:
                    return ""  # No common prefix
        
        return prefix

```

### Explanation of the Code:

- **Class Definition**: The class `Solution` is defined, which is a requirement for LeetCode solutions.
- **Method `longestCommonPrefix`**:
  - It takes in a list of strings `strs`.
  - Checks if `strs` is empty and returns `""` if true.
  - Initializes the variable `prefix` with the first string.
  - Loops through each string starting from the second one.
  - While the current string does not start with the prefix, we shorten the prefix until it either matches or becomes empty.
- **Return Value**: Finally, it returns the common prefix found.

### Time and Space Complexity:
- **Time Complexity**: O(N * M), where N is the number of strings and M is the length of the shortest string among them. In the worst case, we may have to check every character.
- **Space Complexity**: O(1), as we are only using a fixed amount of extra space to store the prefix.

This code is designed to run directly on the LeetCode platform, following their requirements for function encapsulation and input/output handling.

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
 To solve the problem of generating all possible letter combinations from a given string of digits based on the traditional phone keypad mappings, we can follow a systematic approach. Below is the explanation followed by the implementation in Python.

### Explanation:

1. **Mapping Digits to Letters**:
   Each digit from '2' to '9' corresponds to a specific set of letters:
   - '2': "abc"
   - '3': "def"
   - '4': "ghi"
   - '5': "jkl"
   - '6': "mno"
   - '7': "pqrs"
   - '8': "tuv"
   - '9': "wxyz"

   We can store this mapping in a dictionary.

2. **Backtracking Method**:
   We can use a backtracking algorithm to generate combinations. The general idea is to build combinations by appending the corresponding letters of each digit recursively until we have formed a complete combination equal to the length of the input string.

3. **Base Cases**:
   - If the input string `digits` is empty, we return an empty list.
   - If the length of the `digits` is 1, we return the list of letters corresponding to that digit.

4. **Combination Construction**:
   We can maintain a current combination string and an index to keep track of which digit we are processing. For each letter corresponding to the current digit, we append it to the combination and move on to the next digit until we've processed all digits.

5. **Final Output**:
   Once all combinations are formed, we return the list of combinations.

### Implementation:

Here is the implementation in Python, following the LeetCode problem format.



```python
from typing import List

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        
        # Mapping from digit to corresponding letters
        digit_to_letters = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }
        
        # List to store the results
        result = []
        
        # Backtracking function
        def backtrack(index: int, current_combination: str):
            # If the current combination is the same length as digits, add it to the result
            if index == len(digits):
                result.append(current_combination)
                return
            
            # Get the current digit
            current_digit = digits[index]
            # Get the corresponding letters for the current digit
            letters = digit_to_letters[current_digit]
            
            # Loop through the letters and recursively build combinations
            for letter in letters:
                backtrack(index + 1, current_combination + letter)
        
        # Start the backtracking process from the first digit
        backtrack(0, "")
        
        return result

```

### Explanation of the Code:

- **Class Definition**: The solution is encapsulated within a class named `Solution`, which follows the format used in LeetCode submissions.
- **Function**: The `letterCombinations` function carries out the main logic, taking a string `digits`.
- **Base Case**: Early return for an empty `digits` input.
- **Mapping**: A dictionary to map digits to their corresponding letters.
- **Backtracking Function**: A nested function `backtrack` is defined, which is responsible for generating all combinations by recursively appending letters.
- **Result Collection**: Once we reach the length of the `digits`, we append the current combination to `result`.
- **Starting Point**: We invoke `backtrack` starting from the first digit (`index = 0`) and an empty string for `current_combination`.

This implementation efficiently explores all combinations and adheres to the constraints given in the problem statement. You can directly copy and paste this code into the LeetCode editor to test it with various inputs.

# 20. Valid Parentheses

### Problem Description 
Given a string `s` containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid.

An input string is valid if:
Open brackets must be closed by the same type of brackets.

Open brackets must be closed in the correct order.


Example 1:
Input: s = "()"
Output: true

Example 2:
Input: s = "()[]{}"
Output: true

Example 3:
Input: s = "(]"
Output: false

Example 4:
Input: s = "([)]"
Output: false

Example 5:
Input: s = "{[]}"
Output: true

Constraints:
`1 <= s.length <= 104`
`s` consists of parentheses only `'()[]{}'`.

### Solution 
 Certainly! To solve the problem of validating parentheses, we can use a stack-based approach. Here's a detailed explanation followed by the Python code in the LeetCode solution format.

### Explanation:

1. **Stack Data Structure**: We will use a stack to keep track of the opening brackets. When we encounter a closing bracket, we will check if it corresponds to the last opening bracket on the stack.

2. **Bracket Mapping**: We will create a dictionary to map closing brackets to their corresponding opening brackets. This way, we can quickly determine if a closing bracket matches its corresponding opening bracket.

3. **Iterate Through the String**: We will loop through each character in the input string:
   - If the character is an opening bracket (`(`, `{`, or `[`), we will push it onto the stack.
   - If it's a closing bracket (`)`, `}`, or `]`), we will check if there is a corresponding opening bracket on the stack (i.e., if the stack is not empty) and then pop the top of the stack. If the top of the stack does not match the current closing bracket, we can return `False`.

4. **Final Check**: At the end of the iteration, if the stack is empty, it means all the opening brackets were matched and closed correctly, so we return `True`. If there are still items in the stack, it means there are unmatched opening brackets, and we return `False`.

### Python Code:
Here is the code that follows the LeetCode solution format:



```python
class Solution:
    def isValid(self, s: str) -> bool:
        # Dictionary to hold the mapping of closing brackets to opening brackets
        bracket_map = {')': '(', '}': '{', ']': '['}
        # Stack to keep track of the opening brackets
        stack = []
        
        # Iterate over each character in the input string
        for char in s:
            # If the character is a closing bracket
            if char in bracket_map:
                # Pop from stack if it's not empty, otherwise assign a dummy value
                top_element = stack.pop() if stack else '#'
                # Check if the popped element matches the mapped opening bracket
                if bracket_map[char] != top_element:
                    return False
            else:
                # If it's an opening bracket, push onto the stack
                stack.append(char)
        
        # If the stack is empty, all brackets were matched correctly
        return not stack

```

### How to Use:
1. Create an instance of the `Solution` class.
2. Call the `isValid` method with your string `s` as an argument.
3. The function will return `True` if the string is valid and `False` otherwise.

### Example Usage:


```python
solution = Solution()
print(solution.isValid("()"))       # Output: True
print(solution.isValid("()[]{}"))   # Output: True
print(solution.isValid("(]"))       # Output: False
print(solution.isValid("([)]"))     # Output: False
print(solution.isValid("{[]}"))     # Output: True

```

The provided code is efficient and runs in O(n) time complexity where n is the length of the string, and uses O(n) space for the stack in the worst case.

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
 To solve the problem of generating all combinations of well-formed parentheses given `n` pairs, we can use a backtracking approach. The idea is to build the parentheses strings step by step, keeping track of the number of open and close parentheses used so far. We'll only add an open parenthesis (`(`) if we haven't exhausted our `n` pairs, and we'll only add a close parenthesis (`)`) if it doesn't exceed the number of open parentheses used.

Here's how we can break it down:

1. **Base Case**: If the length of the current string (`current_combination`) is equal to `2 * n`, we've used all pairs of parentheses, and we can add this combination to our result list.
2. **Recursive Case**:
   - Add an open parenthesis (`(`) if the number of open parentheses used so far is less than `n`.
   - Add a close parenthesis (`)`) if the number of close parentheses used so far is less than the number of open parentheses used.

3. **Backtracking**: After exploring one option (adding `(` or `)`), we remove it to explore other possibilities.

### Implementation:

Here’s how you can implement this in Python:



```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def backtrack(current_combination: str, open_count: int, close_count: int):
            # Base case: if current combination is the correct length
            if len(current_combination) == 2 * n:
                result.append(current_combination)
                return
            
            # Add an open parenthesis if we still have open ones left
            if open_count < n:
                backtrack(current_combination + '(', open_count + 1, close_count)
            
            # Add a close parenthesis if it won't exceed the number of open ones
            if close_count < open_count:
                backtrack(current_combination + ')', open_count, close_count + 1)

        result = []
        backtrack('', 0, 0)
        return result

```

### Explanation of the Code:

1. **Class and Function Definition**: We define a `Solution` class with a method `generateParenthesis` that takes an integer `n`.

2. **Backtracking Function**: Inside `generateParenthesis`, we define a helper function `backtrack` that accepts:
   - `current_combination`: The string built so far.
   - `open_count`: The count of `(` used.
   - `close_count`: The count of `)` used.

3. **Base Case Detection**: If the length of `current_combination` equals `2 * n`, we append this valid combination to `result`.

4. **Recursive Choices**:
   - If we can add more `(` (i.e., `open_count < n`), we make a recursive call adding `(` and increment `open_count`.
   - If we can add more `)` (i.e., `close_count < open_count`), we make a recursive call adding `)` and increment `close_count`.

5. **Result Compilation**: We initialize `result` as an empty list and start our backtrack process with an empty string and zero counts for both parentheses. After all recursive calls are complete, we return `result`.

This method effectively explores all valid combinations of parentheses using backtracking and ensures the parentheses are well-formed by maintaining counts of `(` and `)` added. The result is guaranteed to contain only valid configurations.

# 28. Implement strStr()

### Problem Description 
Implement strStr().

Return the index of the first occurrence of needle in haystack, or `-1` if `needle` is not part of `haystack`.

Clarification:
What should we return when `needle` is an empty string? This is a great question to ask during an interview.

For the purpose of this problem, we will return 0 when `needle` is an empty string. This is consistent to C's strstr() and Java's indexOf().


Example 1:
Input: haystack = "hello", needle = "ll"
Output: 2

Example 2:
Input: haystack = "aaaaa", needle = "bba"
Output: -1

Example 3:
Input: haystack = "", needle = ""
Output: 0

Constraints:
`0 <= haystack.length, needle.length <= 5 * 104`
`haystack` and `needle` consist of only lower-case English characters.

### Solution 
 Certainly! The problem here requires us to implement a `strStr()` function that finds the first occurrence of a substring (`needle`) within a string (`haystack`). If `needle` is not found, it should return `-1`, and if `needle` is an empty string, it should return `0`.

### Step-by-Step Explanation

1. **Understanding Input and Output**:
   - We will be dealing with two strings: `haystack` (the main string) and `needle` (the substring we are searching for).
   - The output will either be:
     - The index of the first occurrence of `needle` in `haystack`, or
     - `-1` if `needle` is not found.
     - `0` if `needle` is an empty string.

2. **Edge Cases**:
   - If `needle` is an empty string, return `0`.
   - If `haystack` is empty and `needle` is not, return `-1`.

3. **Algorithm**:
   - We can utilize Python's built-in string method `find` which returns the lowest index of the substring if found, and `-1` otherwise.
   - However, since we want to demonstrate a manual implementation, we'll employ a nested loop approach, where we check each substring of `haystack` to see if it matches `needle`.

4. **Complexity**:
   - The time complexity of our algorithm will be O(N * M), where N is the length of `haystack` and M is the length of `needle`, since we may potentially examine many substrings.

### The Code Implementation:
Here is the Python code that follows the LeetCode format:



```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        # Check edge case where needle is an empty string
        if needle == "":
            return 0
        
        # Get lengths of haystack and needle
        haystack_length = len(haystack)
        needle_length = len(needle)
        
        # If haystack is shorter than needle, return -1
        if haystack_length < needle_length:
            return -1
        
        # Iterate over haystack
        for i in range(haystack_length - needle_length + 1):
            # Check for substring match
            if haystack[i:i + needle_length] == needle:
                return i  # Return the starting index of the match
        
        # If no match found, return -1
        return -1

```

### How to Run:
To run this solution on the LeetCode platform, follow these steps:
1. Go to the problem statement page on LeetCode.
2. Enter the provided code in the code editor.
3. Make sure to select the correct programming language (Python).
4. Click on "Run Code" to test it with example test cases or "Submit" to validate against all test cases.

### Test Cases:
Using the provided examples:
- For `haystack = "hello"` and `needle = "ll"`, the output will be `2`.
- For `haystack = "aaaaa"` and `needle = "bba"`, the output will be `-1`.
- For `haystack = ""` and `needle = ""`, the output will be `0`.

This code will accurately implement the `strStr()` function according to the problem requirements.

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
 To tackle the problem of finding starting indices of substrings in a string `s` that form a concatenation of words from the list `words`, we can utilize a sliding window approach with hash maps for counting. Let me break down the solution step by step and present the code in a format suitable for LeetCode.

### Step-by-step Explanation:

1. **Input Parsing**:
   - We will be given a string `s` and a list of `words`. The words in `words` must all have the same length for them to be concatenated together in `s`.

2. **Variable Initialization**:
   - We first determine the total length of the substrings we need to find: this is the sum of the lengths of the words in `words`.
   - We'll also create a hash map to count the occurrences of each word in `words`.

3. **Sliding Window Technique**:
   - We'll slide a window of the substring length over `s`. The length of this window is equal to the total length of the concatenated words.
   - For each position in `s`, we check if the substring formed by our current window can be broken down exactly into the words given in `words`.

4. **Checking Validity**:
   - We split the current substring into segments of the size of each word, and use another hash map to count the occurrences of these words in the substring.
   - If the counts match the counts in `words`, we have found a valid starting index.

5. **Output**:
   - We store indices where valid substrings are found and return them.

Here is the Python code following the described plan:



```python
from collections import Counter

class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        if not s or not words:
            return []

        word_length = len(words[0])
        word_count = len(words)
        total_length = word_length * word_count
        word_map = Counter(words)
        result_indices = []

        # Loop through each possible starting point
        for i in range(len(s) - total_length + 1):
            # Extract the current substring of total_length
            curr_substring = s[i:i + total_length]
            # Split it into chunks of word_length
            chunks = [curr_substring[j:j + word_length] for j in range(0, len(curr_substring), word_length)]
            curr_map = Counter(chunks)

            # Check if the current map matches the word_map
            if curr_map == word_map:
                result_indices.append(i)

        return result_indices

```

### Detailed Explanation of the Code:

1. **Imports**:
   - We import `Counter` from the `collections` module to help in counting occurrences of words.

2. **Class and Method Definition**:
   - We define a class `Solution` and a method `findSubstring` that takes the string `s` and the list of words `words`.

3. **Edge Case Handling**:
   - We check if either the string or the list of words is empty and return an empty list.

4. **Variable Setup**:
   - We calculate `word_length`, `word_count`, and `total_length` to help determine the size of the sliding window.
   - We create a `Counter` for `words` to easily compare the words found in substrings.

5. **Sliding Window**:
   - We iterate over each valid starting index for our sliding window.
   - For each index, we extract the substring of the total length needed and split it into chunks of the size of a word.

6. **Counting and Validating**:
   - We create a `Counter` for the current substring's chunks and compare it with the word map. If they match, we append the index to `result_indices`.

7. **Return the Results**:
   - Finally, we return the list of result indices where valid substrings start.

This solution has a time complexity of O(N), where N is the length of the string `s`, as we check each substring of the appropriate length, and counting operations are constant-time on average due to the use of hash maps. The space complexity is driven by the `Counter` storage which is proportionate to the number of unique words.

# 32. Longest Valid Parentheses

### Problem Description 
Given a string containing just the characters `'('` and `')'`, find the length of the longest valid (well-formed) parentheses substring.


Example 1:
Input: s = "(()"
Output: 2
Explanation: The longest valid parentheses substring is "()".


Example 2:
Input: s = ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()".


Example 3:
Input: s = ""
Output: 0

Constraints:
`0 <= s.length <= 3 * 104`
`s[i]` is `'('`, or `')'`.

### Solution 
 To solve the problem of finding the length of the longest valid parentheses substring, we can use a stack-based approach or two counters. Here, I'll outline and implement the stack-based solution, which is both intuitive and efficient.

### Explanation

1. **Understanding Valid Parentheses**: A valid parentheses substring means that for every opening parenthesis `'('`, there should be a corresponding closing parenthesis `')'`, and they should be properly nested.

2. **Using a Stack**: We can use a stack to keep track of the indices of the opening parentheses. Whenever we encounter a closing parenthesis, we can check if there is a matching opening parenthesis available in the stack.

3. **Algorithm Steps**:
   - Initialize a stack and push `-1` onto it. This acts as a base for the index of the last unmatched closing parenthesis.
   - Iterate through each character in the string:
     - If the character is `'('`, push its index onto the stack.
     - If the character is `')'`:
       - Pop the top of the stack. If the stack is not empty after popping, calculate the valid length of parentheses by subtracting the current index with the new top of the stack (which will give us the length of the valid substring).
       - If the stack becomes empty after popping, push the current index as a new base for unmatched closing parentheses.
   - Keep track of the maximum valid length found during the iterations.

4. **Time Complexity**: The algorithm runs in O(n) time, where n is the length of the input string, since we process each character once.

5. **Space Complexity**: The space complexity is O(n) in the worst case due to the stack.

Now, let's put this into a Python function that can be directly run on the LeetCode website.

### Python Code



```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = [-1]  # Initialize stack with -1 to account for base case
        max_length = 0  # To keep track of the maximum length found

        for i in range(len(s)):
            if s[i] == '(':  # If an opening bracket is found
                stack.append(i)  # Push its index onto the stack
            else:  # If a closing bracket is found
                stack.pop()  # Pop the top of the stack
                if stack:  # If the stack is not empty
                    # Calculate the length of the valid substring
                    max_length = max(max_length, i - stack[-1])
                else:
                    # If the stack is empty, push the current index
                    stack.append(i)

        return max_length  # Return the maximum length found

```

### How to Use the Code

- The above code defines a `Solution` class with a method `longestValidParentheses`. 
- You can create an instance of this class and call the method with a string input to get the length of the longest valid parentheses substring.

### Example Usage



```python
solution = Solution()
print(solution.longestValidParentheses("(()"))     # Output: 2
print(solution.longestValidParentheses(")()())"))  # Output: 4
print(solution.longestValidParentheses(""))         # Output: 0

```

This implementation adheres to the LeetCode format and can be directly copied into the LeetCode problems section to test various inputs.

# 38. Count and Say

### Problem Description 
The count-and-say sequence is a sequence of digit strings defined by the recursive formula:
`countAndSay(1) = "1"`
`countAndSay(n)` is the way you would "say" the digit string from `countAndSay(n-1)`, which is then converted into a different digit string.

To determine how you "say" a digit string, split it into the minimal number of groups so that each group is a contiguous section all of the same character. Then for each group, say the number of characters, then say the character. To convert the saying into a digit string, replace the counts with a number and concatenate every saying.

For example, the saying and conversion for digit string `"3322251"`:
Given a positive integer `n`, return the `nth` term of the count-and-say sequence.


Example 1:
Input: n = 1
Output: "1"
Explanation: This is the base case.


Example 2:
Input: n = 4
Output: "1211"
Explanation:
countAndSay(1) = "1"
countAndSay(2) = say "1" = one 1 = "11"
countAndSay(3) = say "11" = two 1's = "21"
countAndSay(4) = say "21" = one 2 + one 1 = "12" + "11" = "1211"

Constraints:
`1 <= n <= 30`

### Solution 
 To solve the problem of generating the nth term of the count-and-say sequence, we can break down the solution into a series of systematic steps, following a recursive approach. Let's start by explaining the concept and then proceed with the implementation in the LeetCode solution format.

### Explanation:

1. **Base Case**: We know that `countAndSay(1)` is simply the string "1". This serves as our starting point for the recursive function.

2. **Recursive Process**:
   - For a given `n`, to generate `countAndSay(n)`, we find `countAndSay(n-1)` first.
   - We then process the result of `countAndSay(n-1)` to create `countAndSay(n)`.
   - To process the string, we need to iterate through the string, counting contiguous characters and forming a new string that concatenates the count of characters followed by the character itself.

3. **Counting Characters**: As we iterate through the string, we need to maintain:
   - The current character we are counting.
   - A counter to count how many times the same character appears consecutively.
   - Once we encounter a different character, we append the count and the character to the result string.

The algorithm will efficiently generate the desired sequence up to the specified index by recursively calling the function until we reach our base case.

### Python Code:

Here’s the code that implements the above logic:



```python
class Solution:
    def countAndSay(self, n: int) -> str:
        # Base case
        if n == 1:
            return "1"
        
        # Recursive case
        previous_term = self.countAndSay(n - 1)  # Get the previous term
        result = []
        count = 1  # Initialize count
        
        # Process the previous_term to construct the current term
        for i in range(1, len(previous_term)):
            # If the current character is the same as the previous one, increment the count
            if previous_term[i] == previous_term[i - 1]:
                count += 1
            else:
                # Otherwise, append the count and the character to the result
                result.append(str(count))
                result.append(previous_term[i - 1])  # Append the character
                count = 1  # Reset count for the new character
        
        # For the last group
        result.append(str(count))
        result.append(previous_term[-1])  # Append the last character
        
        # Join the result list into a string and return
        return ''.join(result)

```

### How to Use the Code:
This code adheres to the LeetCode submission format:
- It defines a class named `Solution` which contains the method `countAndSay`.
- The method takes an integer `n` as input and returns a string.
- You can directly copy this function into the LeetCode platform and it should run correctly.

### Complexity Analysis:
- **Time Complexity**: The approach involves generating the sequence recursively. Since generating each term involves scanning the previous term entirely to count contiguous characters, the time complexity is approximately O(2^n) due to the multiple string manipulations and recursive calls.
- **Space Complexity**: O(n), mainly for storing the result strings and call stack space in the recursion.

This provides a clear, direct implementation to solve the count-and-say problem efficiently within the constraints provided.

# 43. Multiply Strings

### Problem Description 
Given two non-negative integers `num1` and `num2` represented as strings, return the product of `num1` and `num2`, also represented as a string.

Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.


Example 1:
Input: num1 = "2", num2 = "3"
Output: "6"

Example 2:
Input: num1 = "123", num2 = "456"
Output: "56088"

Constraints:
`1 <= num1.length, num2.length <= 200`
`num1` and `num2` consist of digits only.

Both `num1` and `num2` do not contain any leading zero, except the number `0` itself.

### Solution 
 To solve the problem of multiplying two non-negative integers represented as strings without directly converting them to integers, we can simulate the multiplication process similar to how we do it manually.

Here's a step-by-step breakdown of the approach:

1. **Understanding the Multiplication Process**: When multiplying two numbers (e.g., 123 and 456), we multiply each digit of the first number by each digit of the second number, just like how we were taught in school:

   
```
                1 2 3
              x 4 5 6
              ---------
                7 3 8 (This is 123 * 6)
              6 1 5   (This is 123 * 5, shifted one position to the left)
           + 4 9 2     (This is 123 * 4, shifted two positions to the left)
              ---------
              5 6 0 8 8
   
```

2. **Positioning the Result**: The result of each multiplication needs to be placed correctly based on its corresponding digit's position in the original numbers. 

3. **Using an Array to Collect Results**: We'll store the result of each multiplication in an array where the index represents the position of the digit in the final answer.

4. **Handling Carries**: After calculating the raw values, we need to handle any carry that might result from the addition.

5. **Building the Final String**: Finally, after processing all the digits and resolving carries, we'll convert our result array back into a string format.

Here’s the complete Python code solidifying the above logic in a LeetCode-compatible format:



```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        # Handle the special case of multiplying by 0
        if num1 == "0" or num2 == "0":
            return "0"
        
        # Length of the two numbers
        len1 = len(num1)
        len2 = len(num2)
        
        # Result array to store the intermediate results
        result = [0] * (len1 + len2)
        
        # Reverse iterate through num1 and num2
        for i in range(len1 - 1, -1, -1):
            for j in range(len2 - 1, -1, -1):
                # Multiply the digits (convert char to int)
                mul = (ord(num1[i]) - ord('0')) * (ord(num2[j]) - ord('0'))
                
                # Position in the result array
                p1 = i + j
                p2 = i + j + 1
                
                # Add multiplication result to the current position
                total = mul + result[p2]

                # Update the result array
                result[p2] = total % 10  # Current digit
                result[p1] += total // 10  # Carry to the next position
        
        # Convert the result array to string, skipping leading zeros
        result_str = ''.join(map(str, result)).lstrip('0')
        
        # If result_str is empty, means the result is 0
        return result_str if result_str else "0"

```

### Explanation of the Code:
1. We first check for edge cases where either number is "0" and immediately return "0" as the product.
2. We then initialize a list called `result` which has a length of `len1 + len2` to hold potential maximum digits of the product.
3. We loop through each digit of `num1` and `num2` in reverse order. This mimics the manual multiplication method.
4. For each pair of digits, we multiply them and place the result in the `result` array according to their positional indices (`p1` and `p2`).
5. After calculating the products, we handle any carries that may arise.
6. Finally, we join the digits from `result`, convert it to a string, and remove leading zeros before returning it.

This structured approach ensures efficiency and correctness without using any large number libraries or converting strings to integers directly.

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
 To solve the problem of wildcard pattern matching with support for `'?'` and `'*'`, we can utilize a dynamic programming (DP) approach. This approach works by creating a 2D table where the rows correspond to characters in the input string `s` and the columns correspond to characters in the pattern `p`. The idea is to fill this table based on the matching rules defined by the wildcard positions.

### Steps to Solve the Problem:

1. **Initialization**: Create a DP table `dp` with dimensions `(len(s) + 1) x (len(p) + 1)`. The rows represent characters from `s` and the columns represent characters from `p`.

2. **Base Case**: Set `dp[0][0] = True` because two empty strings match. Also, handle patterns that start with `*`. If `p` starts with `*`, we can match an empty string with `*`, so update the corresponding entries in the first row of the table.

3. **Filling the DP Table**:
   - Iterate through each character of `s` and `p`.
   - If characters match or if `p` has a `'?'`, we take the diagonal value `dp[i-1][j-1]`.
   - If `p` has a `'*'`, we have two options:
     - Treat `'*'` as matching no characters and refer to `dp[i][j-1]`.
     - Treat `'*'` as matching one character from `s`, and refer to `dp[i-1][j]`.
   - Update `dp[i][j]` based on the conditions above.

4. **Result**: The result of whether `s` matches `p` will be stored in `dp[len(s)][len(p)]`.

### Python Code Implementation:

Here's how we can implement this logic in Python following the format commonly used on LeetCode:



```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        # Length of the input string and the pattern
        m, n = len(s), len(p)
        
        # Create dp table with default False values
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        # Base case: empty pattern matches with empty string
        dp[0][0] = True
        
        # Handle patterns that start with '*' which can match empty string
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 1]
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # If the characters match or there is a '?'
                if p[j - 1] == s[i - 1] or p[j - 1] == '?':
                    dp[i][j] = dp[i - 1][j - 1]
                
                # If the pattern has a '*', it can either match empty sequence or one more character
                elif p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
        
        # The result is in the cell that matches full lengths of s and p
        return dp[m][n]

# Example usage:
# sol = Solution()
# print(sol.isMatch("adceb", "*a*b"))  # Outputs: True

```

### Explanation of the Code:
- We define a `Solution` class with a method `isMatch`.
- We calculate lengths of `s` and `p` and initialize the DP table.
- We set the base cases for the DP table.
- We fill the table using nested loops to cover all combinations of `s` and `p`.
- Finally, we return the value from `dp[m][n]`, which indicates whether the full string `s` matches the full pattern `p`.

This code runs efficiently within given constraints and handles various cases, including patterns with multiple wildcards and corner cases with empty strings.

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
 To solve the problem of grouping anagrams from a list of strings, we can leverage the fact that anagrams always share the same character count and can therefore be grouped based on the sorted version of each string. The primary steps involved in this solution are:

1. **Sorting**: For each string in the input list, we can sort its characters and use the sorted string as a key to group the anagrams.
2. **Using a Dictionary**: We will store lists of anagrams in a dictionary, where the keys are the sorted strings and the values are lists of strings that are anagrams of each other.
3. **Collecting Results**: Lastly, we will collect all the groups of anagrams from the dictionary and return them as the final output.

Here is the Python code that implements the above logic in a format suitable for LeetCode:



```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # Create a dictionary to hold lists of anagrams
        anagrams = {}
        
        # Iterate over each string in the input list
        for s in strs:
            # Sort the string and use it as a key
            key = ''.join(sorted(s))
            # Append the original string to the list corresponding to the sorted key
            if key in anagrams:
                anagrams[key].append(s)
            else:
                anagrams[key] = [s]
        
        # Return the grouped anagrams as a list of lists
        return list(anagrams.values())

```

### Explanation of the Code:

1. **Class Definition**: We define a class `Solution` which is the required format for LeetCode problems.
2. **Method Definition**: The method `groupAnagrams` takes a list of strings called `strs` as input and returns a list of lists of strings.
3. **Anagrams Dictionary**: We create an empty dictionary `anagrams` to store the groups of anagrams.
4. **Iterating through Strings**:
   - For each string `s`, we sort the string. For example, sorting "eat" would give us "aet".
   - We then check if the sorted string (the key) already exists in our dictionary:
     - If it does, we append the original string `s` to the list of that key.
     - If it doesn't, we create a new entry in the dictionary with the sorted string as the key and the original string as the first element in a new list.
5. **Return Results**: Finally, we return the values of the dictionary, which are lists of grouped anagrams, using `list(anagrams.values())`.

### Complexity Analysis:
- **Time Complexity**: The time complexity is O(N * K log K) where N is the number of strings in the input list and K is the maximum length of a string. The sorting step takes O(K log K) for each string.
- **Space Complexity**: The space complexity is O(N * K) to store the output results.

This approach is efficient and straightforward, leveraging sorting for grouping anagrams effectively.

# 58. Length of Last Word

### Problem Description 
Given a string `s` consists of some words separated by spaces, return the length of the last word in the string. If the last word does not exist, return `0`.

A word is a maximal substring consisting of non-space characters only.


Example 1:
Input: s = "Hello World"
Output: 5

Example 2:
Input: s = " "
Output: 0

Constraints:
`1 <= s.length <= 104`
`s` consists of only English letters and spaces `' '`.

### Solution 
 To solve the problem of finding the length of the last word in a given string `s`, we can follow a straightforward approach. The steps we will take are:

1. **Strip leading and trailing spaces**: This ensures that any extra spaces at the beginning or end of the string do not interfere with our word counting.
2. **Split the string**: We will split the stripped string into words using spaces as the delimiter.
3. **Check for the last word**: If the resulting list of words is empty, it means there was no valid word, so we return `0`. Otherwise, we return the length of the last word in the list.

Here's the Python code implementing the above solution in a format suitable for LeetCode:



```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        # Step 1: Strip leading and trailing whitespace from the string
        s = s.strip()

        # Step 2: Split the string into a list of words
        words = s.split()
        
        # Step 3: Get the length of the last word
        if not words:
            return 0  # If there are no words, return 0
        
        return len(words[-1])  # Return the length of the last word

```

### Explanation:
- **Class and Method**: We define a `Solution` class with a method `lengthOfLastWord` that accepts one parameter, `s`.
  
- **Stripping Spaces**: `s.strip()` removes any leading or trailing whitespace from the string, which is necessary because extra spaces can cause confusion when checking for the existence of words. 

- **Splitting the String**: `s.split()` splits the string into a list based on whitespace. By default, the `split()` method will handle multiple spaces and only create words from non-empty substrings.

- **Returning the Result**:
  - We check if `words` is empty. If it is (which can happen if `s` contains only spaces), we return `0`.
  - If there are words present, we return the length of the last word using `len(words[-1])`, where `words[-1]` gives us the last word in the list.

### Complexity Analysis:
- **Time Complexity**: O(n), where n is the length of the string `s`, because both stripping and splitting operations traverse the entire string.
- **Space Complexity**: O(n), for storing the list of words generated by splitting the string.

This solution is efficient and works well within the given constraints (1 <= s.length <= 10^4).

# 65. Valid Number

### Problem Description 
A valid number can be split up into these components (in order):
A decimal number or an integer.

(Optional) An `'e'` or `'E'`, followed by an integer.

A decimal number can be split up into these components (in order):
(Optional) A sign character (either `'+'` or `'-'`).

One of the following formats:
	
At least one digit, followed by a dot `'.'`.

At least one digit, followed by a dot `'.'`, followed by at least one digit.

A dot `'.'`, followed by at least one digit.

An integer can be split up into these components (in order):
(Optional) A sign character (either `'+'` or `'-'`).

At least one digit.

For example, all the following are valid numbers: `["2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"]`, while the following are not valid numbers: `["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"]`.

Given a string `s`, return `true` if `s` is a valid number.


Example 1:
Input: s = "0"
Output: true

Example 2:
Input: s = "e"
Output: false

Example 3:
Input: s = "."
Output: false

Example 4:
Input: s = ".1"
Output: true

Constraints:
`1 <= s.length <= 20`
`s` consists of only English letters (both uppercase and lowercase), digits (`0-9`), plus `'+'`, minus `'-'`, or dot `'.'`.

### Solution 
 To determine if a given string is a valid number according to the specified rules, we can implement a solution using regular expressions (regex). This will allow us to define the valid number formats in a concise and readable manner.

### Explanation

1. **Regular Expression Breakdown**:
   - **Sign**: A valid number may start with an optional `+` or `-` sign: `([+-])?`
   - **Integer or Decimal**:
     - Integer: At least one digit, which can be represented as `\d+`.
     - Decimal:
       - Should have at least one digit followed by a dot: `\d+\.` 
       - Or a dot followed by at least one digit: `\.\d+`
       - Or at least one digit followed by a dot and optionally more digits: `\d+\.\d*`
   - **Scientific Notation**: valid if there's an `e` or `E` followed by an optional sign and at least one digit.

2. **Complete Pattern**: 
   We can combine these components into one regex pattern:
   
```
   ^[+-]?((\d+\.\d*)|(\.\d+)|(\d+))(e[+-]?\d+)?$
   
```
   - `^` and `$` denote the start and end of the string to ensure the entire string is matched.
   - The core logic considers three cases for the decimal number and includes the scientific notation if applicable.

3. **Code Implementation**:
   Below is the Python code implementing this logic.

### Python Code



```python
import re

class Solution:
    def isNumber(self, s: str) -> bool:
        # Define the regex pattern for a valid number
        pattern = r'^[+-]?((\d+\.\d*)|(\.\d+)|(\d+))(e[+-]?\d+)?$'
        # Use the fullmatch method to check if the entire string matches the pattern
        return re.fullmatch(pattern, s) is not None

```

### How the Code Works

- We import the `re` module for regex operations.
- We define a class `Solution` and a method `isNumber` that takes a string `s` as input.
- We use `re.fullmatch()` which checks if the entire input string exactly matches the regex pattern defined.
- If the string matches the pattern, we return `True`, indicating it is a valid number; otherwise, we return `False`.

### Test Cases:
You can run the following test cases to check the functionality:



```python
# Test the isNumber function
solution = Solution()

# Valid numbers
print(solution.isNumber("0"))           # True
print(solution.isNumber("-0.1"))        # True
print(solution.isNumber("+3.14"))       # True
print(solution.isNumber("4."))          # True
print(solution.isNumber("2e10"))        # True
print(solution.isNumber("-90E3"))       # True
print(solution.isNumber("3e+7"))        # True
print(solution.isNumber("+6e-1"))       # True

# Invalid numbers
print(solution.isNumber("abc"))         # False
print(solution.isNumber("1a"))          # False
print(solution.isNumber("1e"))          # False
print(solution.isNumber("e3"))          # False
print(solution.isNumber("99e2.5"))      # False
print(solution.isNumber("--6"))         # False
print(solution.isNumber("-+3"))         # False
print(solution.isNumber("95a54e53"))    # False

```

You can copy this code directly into LeetCode and it should work correctly for the problem statement.

# 67. Add Binary

### Problem Description 
Given two binary strings `a` and `b`, return their sum as a binary string.


Example 1:
Input: a = "11", b = "1"
Output: "100"

Example 2:
Input: a = "1010", b = "1011"
Output: "10101"

Constraints:
`1 <= a.length, b.length <= 104`
`a` and `b` consist only of `'0'` or `'1'` characters.

Each string does not contain leading zeros except for the zero itself.

### Solution 
 To solve the problem of summing two binary strings, we need to implement a function that can handle binary addition manually without leveraging built-in functions for binary conversion. We'll simulate the process of binary addition, keeping track of carry as we go through each bit from the least significant to the most significant.

### The Approach:
1. **Initialization**: Start with a carry of 0 and an empty list to build the result.
2. **Reverse Looping**: Since we need to add bits from the least significant bit to the most significant, we will iterate from the end of both strings towards the start.
3. **Bit Addition**:
    - For each pair of bits from the two strings, along with the carry:
      - Calculate the sum of the two bits and the carry.
      - Determine the new bit to be added to the result (which is the sum modulo 2).
      - Update the carry for the next iteration (which is the sum divided by 2).
4. **Final Carry Handling**: If there is a carry left after the last addition, append it to the result.
5. **Result Construction**: Since we built the result in reverse order, we need to reverse it before returning.

### Implementation in Python:

Here's the LeetCode-ready code that implements the above approach:



```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        # Initialize the carry and an empty result string (as list for efficiency)
        carry = 0
        result = []
        
        # Initialize pointers for both strings
        i, j = len(a) - 1, len(b) - 1
        
        # Loop through both strings until both are fully processed
        while i >= 0 or j >= 0 or carry:
            # Get the current bits (0 if we've exhausted the string)
            bit_a = int(a[i]) if i >= 0 else 0
            bit_b = int(b[j]) if j >= 0 else 0
            
            # Calculate the total for this position
            total = bit_a + bit_b + carry
            
            # Update the result with the current binary bit
            result.append(str(total % 2))
            
            # Update the carry for the next position
            carry = total // 2
            
            # Move to the next bits
            i -= 1
            j -= 1
        
        # Reverse the result to get the correct order and join to form a string
        return ''.join(reversed(result))

```

### Explanation of Code:
1. **Class Definition**: We define a class `Solution` with a method `addBinary`.
2. **Input Parameters**: The method takes two strings `a` and `b`.
3. **Looping through Bits**: A `while` loop continues as long as there are bits left in either string or there is a carry.
4. **Fetching Bits**: We convert characters to integers where applicable, defaulting to 0 when strings are exhausted.
5. **Calculating Total**: The total is derived from the current bits and the carry.
6. **Storing Results**: We use `append` to add the calculated bit to the result list.
7. **Reversing the Result**: After the loop, we reverse the list and join it to form the final binary string.

### Complexity Analysis:
- **Time Complexity**: O(n), where n is the maximum length of the two input strings, as we traverse through both strings once.
- **Space Complexity**: O(n), for the space required to store the result before converting it to a string.

This approach efficiently handles the addition of two binary strings and adheres to the constraints provided in the problem statement.

# 68. Text Justification

### Problem Description 
Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly maxWidth characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

For the last line of text, it should be left justified and no extra space is inserted between words.

Note:
A word is defined as a character sequence consisting of non-space characters only.

Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.

The input array `words` contains at least one word.


Example 1:
Input: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
Output:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]

Example 2:
Input: words = ["What","must","be","acknowledgment","shall","be"], maxWidth = 16
Output:
[
  "What   must   be",
  "acknowledgment  ",
  "shall be        "
]
Explanation: Note that the last line is "shall be    " instead of "shall     be", because the last line must be left-justified instead of fully-justified.

Note that the second line is also left-justified becase it contains only one word.


Example 3:
Input: words = ["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"], maxWidth = 20
Output:
[
  "Science  is  what we",
  "understand      well",
  "enough to explain to",
  "a  computer.  Art is",
  "everything  else  we",
  "do                  "
]

Constraints:
`1 <= words.length <= 300`
`1 <= words[i].length <= 20`
`words[i]` consists of only English letters and symbols.

`1 <= maxWidth <= 100`
`words[i].length <= maxWidth`

### Solution 
 Here's a detailed explanation and the corresponding Python code for the problem of text justification as described.

## Problem Explanation
You need to format a list of words into lines of a specified width (`maxWidth`). Each line should be fully justified; this means every line, except the last one, must contain exactly `maxWidth` characters. The last line should be left-justified, meaning words are filled to the left, and no additional spaces should be inserted between them.

When multiple spaces need to be inserted to fill the line:
- Space is distributed as evenly as possible between words.
- If spaces can't be evenly distributed, extra spaces are added from left to right (i.e., the leftmost gaps will have more spaces).

### Key Steps to Solve the Problem
1. Iterate over the words and group them into lines while ensuring the total length does not exceed `maxWidth`.
2. For each line (except the last line), calculate the number of spaces needed and distribute them evenly between words, padding extra spaces to the leftmost gaps.
3. For the last line, simply join the words with a single space and pad spaces on the right to reach `maxWidth`.

Here's how we can implement this in Python:



```python
def fullJustify(words, maxWidth):
    def add_spaces(words, maxWidth, is_last_line):
        if is_last_line or len(words) == 1:
            # Last line or single word line: left-justify
            return ' '.join(words).ljust(maxWidth)
        
        total_chars = sum(len(word) for word in words)
        total_spaces = maxWidth - total_chars
        spaces_between_words = len(words) - 1
        if spaces_between_words > 0:
            # Minimum space per gap
            space_per_gap = total_spaces // spaces_between_words
            # Extra spaces to distribute
            extra_spaces = total_spaces % spaces_between_words
            
            justified_line = ""
            for i in range(spaces_between_words):
                justified_line += words[i]
                justified_line += " " * (space_per_gap + (1 if i < extra_spaces else 0))
                
            justified_line += words[-1]  # Add the last word
            return justified_line
        
        return words[0].ljust(maxWidth)
        
    result = []
    current_line = []
    
    for word in words:
        if sum(len(w) for w in current_line) + len(current_line) + len(word) > maxWidth:
            result.append(add_spaces(current_line, maxWidth, False))
            current_line = []
        current_line.append(word)
        
    # Handle last line
    result.append(add_spaces(current_line, maxWidth, True))
    
    return result

# Example Usage
words1 = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth1 = 16
print(fullJustify(words1, maxWidth1))

words2 = ["What","must","be","acknowledgment","shall","be"]
maxWidth2 = 16
print(fullJustify(words2, maxWidth2))

words3 = ["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"]
maxWidth3 = 20
print(fullJustify(words3, maxWidth3))

```

### Explanation of the Code:
1. **Function `fullJustify(words, maxWidth)`**: The primary function that processes the list of words and formats them according to the specifications.
2. **Inner Function `add_spaces(words, maxWidth, is_last_line)`**: This function deals with the justification logic for a line of words. It checks if it's the last line or if there is only one word. If yes, it just left-justifies that line. Otherwise, it calculates how to distribute spaces evenly between the words.
3. **Main Logic**: Iterate over the words, checking if adding the next word would exceed `maxWidth`. If it does, justify the current line and reset the current line. Finally, handle the last line separately.

### Output
Running the above code will yield the expected results for the example test cases provided, along with proper justification for each line. You can directly run this code on LeetCode, and it should function correctly and pass the test cases defined by the problem statement.

# 71. Simplify Path

### Problem Description 
Given a string `path`, which is an absolute path (starting with a slash `'/'`) to a file or directory in a Unix-style file system, convert it to the simplified canonical path.

In a Unix-style file system, a period `'.'` refers to the current directory, a double period `'..'` refers to the directory up a level, and any multiple consecutive slashes (i.e. `'//'`) are treated as a single slash `'/'`. For this problem, any other format of periods such as `'...'` are treated as file/directory names.

The canonical path should have the following format:
The path starts with a single slash `'/'`.

Any two directories are separated by a single slash `'/'`.

The path does not end with a trailing `'/'`.

The path only contains the directories on the path from the root directory to the target file or directory (i.e., no period `'.'` or double period `'..'`)
Return the simplified canonical path.


Example 1:
Input: path = "/home/"
Output: "/home"
Explanation: Note that there is no trailing slash after the last directory name.


Example 2:
Input: path = "/../"
Output: "/"
Explanation: Going one level up from the root directory is a no-op, as the root level is the highest level you can go.


Example 3:
Input: path = "/home//foo/"
Output: "/home/foo"
Explanation: In the canonical path, multiple consecutive slashes are replaced by a single one.


Example 4:
Input: path = "/a/./b/../../c/"
Output: "/c"

Constraints:
`1 <= path.length <= 3000`
`path` consists of English letters, digits, period `'.'`, slash `'/'` or `'_'`.

`path` is a valid absolute Unix path.

### Solution 
 To solve the problem of converting a given Unix-style file path to its simplified canonical form, we need to carefully process the input string based on certain rules that dictate how paths are interpreted in a Unix file system. Let's break it down step-by-step and then implement it in Python.

### Explanation of the Approach:

1. **Initialization**:
   We will use a stack to keep track of directories as we process them.

2. **Splitting the Path**:
   We will split the input string `path` using the slash `'/'` as a delimiter. This helps us get each component of the path.

3. **Processing Each Component**:
   - If the component is an empty string (which arises from consecutive slashes), we simply ignore it.
   - If the component is `'.'`, it means "current directory," so we can ignore it.
   - If the component is `'..'`, it means we need to go up one directory. We will pop from our stack if there's any directory present (to avoid going above the root).
   - Any other components should be treated as valid directory names, and we push them onto our stack.

4. **Building the Result**:
   After processing all components, we will join the directories in the stack. The result should start with a '/' and should NOT end with a trailing '/' unless the result is just the root directory.

5. **Return the Result**:
   If the stack is empty, it means we are at the root, and we should return `'/'`. Otherwise, we return the canonical path constructed from the stack.

### Python Code Implementation:

Here is the corresponding Python code that implements the above logic:



```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        # Initialize a stack to keep track of the directory names
        stack = []

        # Split the input path by '/'
        components = path.split('/')

        for component in components:
            if component == '' or component == '.':
                # Skip empty components and the current directory symbol
                continue
            elif component == '..':
                # Pop from stack if we are not at the root
                if stack:
                    stack.pop()
            else:
                # Push the valid directory name onto the stack
                stack.append(component)

        # Join the stack components to form the simplified path
        # Start with '/' and join the rest with '/'
        simplified_path = '/' + '/'.join(stack)

        return simplified_path

# Example usage:
# sol = Solution()
# print(sol.simplifyPath("/home/"))  # Output: "/home"
# print(sol.simplifyPath("/../"))     # Output: "/"
# print(sol.simplifyPath("/home//foo/"))  # Output: "/home/foo"
# print(sol.simplifyPath("/a/./b/../../c/"))  # Output: "/c"

```

### Explanation of the Code:
- The `Solution` class contains the method `simplifyPath`.
- We start by initializing a stack to hold directory names.
- We split the input path using `split('/')`.
- We loop through each component:
  - We check for empty strings and `'.'` to ignore them.
  - We handle the `..` component by popping from the stack if there is something to pop.
  - Valid directory names are pushed onto the stack.
- Finally, we create the simplified path by joining the elements of the stack with a `'/'`, ensuring it starts with a single `'/'`.
  
This implementation effectively handles all the requirements laid out in the problem statement and adheres to the constraints given. You can directly run the provided code on LeetCode by creating a new solution under the specified problem.

# 72. Edit Distance

### Problem Description 
Given two strings `word1` and `word2`, return the minimum number of operations required to convert `word1` to `word2`.

You have the following three operations permitted on a word:
Insert a character
Delete a character
Replace a character

Example 1:
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')

Example 2:
Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')

Constraints:
`0 <= word1.length, word2.length <= 500`
`word1` and `word2` consist of lowercase English letters.

### Solution 
 To solve the problem of converting one string to another using the minimum operations (insertion, deletion, or replacement), we can use a common dynamic programming approach known as the Edit Distance algorithm. 

Here's a detailed explanation of how we can implement this:

### Approach:

1. **Dynamic Programming Table**: We will create a 2D array `dp` where `dp[i][j]` will represent the minimum number of operations required to convert the first `i` characters of `word1` to the first `j` characters of `word2`.

2. **Initialization**:
   - If either string is empty, the edit distance is simply the length of the other string. Thus:
     - For `dp[i][0]`, it takes `i` operations (deletes) to convert `word1[0...i-1]` to an empty string.
     - For `dp[0][j]`, it takes `j` operations (inserts) to convert an empty string to `word2[0...j-1]`.

3. **Filling the DP Table**:
   - If the characters at position `i-1` in `word1` and position `j-1` in `word2` are the same (`word1[i-1] == word2[j-1]`), then no extra operation is needed. Hence, `dp[i][j] = dp[i-1][j-1]`.
   - If the characters are different, we consider all three possible operations:
     - **Insert**: Transition from `dp[i][j-1] + 1` (insert `word2[j-1]`).
     - **Delete**: Transition from `dp[i-1][j] + 1` (delete `word1[i-1]`).
     - **Replace**: Transition from `dp[i-1][j-1] + 1` (replace `word1[i-1]` with `word2[j-1]`).
   - Finally, take the minimum of the three operations.

4. **Result**: The answer will be in `dp[len(word1)][len(word2)]`, which tells us the minimum operations needed to convert `word1` to `word2`.

### Python Code:

Here is the implementation in Python, following the LeetCode format:



```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # Get the lengths of both words
        m, n = len(word1), len(word2)
        
        # Create a DP table with (m+1) x (n+1)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the first column: converting from word1 to an empty string
        for i in range(1, m + 1):
            dp[i][0] = i
        
        # Fill the first row: converting from an empty string to word2
        for j in range(1, n + 1):
            dp[0][j] = j
            
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  # No operation needed
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,  # Delete
                        dp[i][j - 1] + 1,  # Insert
                        dp[i - 1][j - 1] + 1  # Replace
                    )
        
        return dp[m][n]

```

### Detailed Explanation of the Code:
- We initiate the solution class as required by LeetCode.
- We define the method `minDistance` which takes two strings `word1` and `word2` as input.
- We create a DP table and initialize the first row and column based on the lengths of the strings.
- We then iterate through the table using nested loops, applying the logic explained above to compute the edit distances.
- Finally, the function returns the result, which is the last cell in the DP table, representing the minimum operations needed.

This code can be directly copied and run on the LeetCode website for the specified problem.

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
 To solve the problem of finding the minimum window substring in `s` that contains all characters from `t`, we can use the sliding window technique combined with the hash map to track character counts. The approach works in linear time complexity, O(n), which is optimal for this problem.

### Problem Breakdown:

1. **Character Count**: We will first create a frequency map for string `t` which tells us how many of each character we need to include in our window.

2. **Expand and Contract the Window**: Using two pointers (or indices), we will expand the right pointer to include characters in our window until we meet the requirement of including all characters from `t`. After meeting this requirement, we will then attempt to contract the left pointer to minimize the window size while still containing all characters from `t`.

3. **Check Validity**: A simple counter will help us check if our current window is valid (i.e., if it contains all characters from `t`).

4. **Store the Best Result**: Continuously update the smallest valid window found as you iterate through the string.

### Python Solution in LeetCode format:



```python
from collections import Counter, defaultdict

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not t or not s:
            return ""
        
        # Frequency map for characters in t
        dict_t = Counter(t)
        required = len(dict_t)  # Number of unique characters in t
        
        # Filter s to only include characters that are in t
        filtered_s = [(i, char) for i, char in enumerate(s) if char in dict_t]
        
        l, r = 0, 0  # Left and right pointer
        formed = 0  # To track how many unique characters in t are in the current window
        window_counts = defaultdict(int)
        
        # The result tuple (window length, left, right)
        ans = float("inf"), None, None
        
        while r < len(filtered_s):
            char = filtered_s[r][1]
            window_counts[char] += 1
            
            # Only increase the formed counter if the count of the current char
            # in the window matches the count in dict_t
            if window_counts[char] == dict_t[char]:
                formed += 1
            
            # Try to contract the window until it's no longer valid
            while l <= r and formed == required:
                char = filtered_s[l][1]
                
                # Save the smallest window and its length, left, and right pointers
                end = filtered_s[r][0]
                start = filtered_s[l][0]
                if end - start + 1 < ans[0]:
                    ans = (end - start + 1, start, end)
                
                # The character at the position pointed to by the left pointer
                window_counts[char] -= 1
                if window_counts[char] < dict_t[char]:
                    formed -= 1
                
                l += 1  # Move the left pointer to the right
        
            r += 1  # Move the right pointer to the right

        # Return the smallest window or an empty string if no valid window was found
        return "" if ans[0] == float("inf") else s[ans[1]: ans[2] + 1]

```

### Explanation:

- **Counter and Dictionary**: We use `Counter` to count the occurrences of each character in `t` and store them in `dict_t`. The `required` variable keeps track of how many unique characters we need.

- **Filtered String**: To improve efficiency, we create a filtered version of `s` that only contains characters present in `t`. This will reduce the number of iterations we must perform significantly.

- **Two Pointers**: The left pointer (`l`) expands the window to the right until all characters from `t` are included, then tries to contract it from the left (`r`), minimizing the window size.

- **Window Tracking**: If we find a valid window that contains all characters from `t`, we keep track of the smallest window using variables to compare lengths.

This implementation runs in O(n) time complexity, where n is the length of string `s`, making it efficient for the given constraints.

# 87. Scramble String

### Problem Description 
We can scramble a string s to get a string t using the following algorithm:
If the length of the string is 1, stop.

If the length of the string is > 1, do the following:
	
Split the string into two non-empty substrings at a random index, i.e., if the string is `s`, divide it to `x` and `y` where `s = x + y`.

Randomly decide to swap the two substrings or to keep them in the same order. i.e., after this step, `s` may become `s = x + y` or `s = y + x`.

Apply step 1 recursively on each of the two substrings `x` and `y`.

Given two strings `s1` and `s2` of the same length, return `true` if `s2` is a scrambled string of `s1`, otherwise, return `false`.


Example 1:
Input: s1 = "great", s2 = "rgeat"
Output: true
Explanation: One possible scenario applied on s1 is:
"great" --> "gr/eat" // divide at random index.

"gr/eat" --> "gr/eat" // random decision is not to swap the two substrings and keep them in order.

"gr/eat" --> "g/r / e/at" // apply the same algorithm recursively on both substrings. divide at ranom index each of them.

"g/r / e/at" --> "r/g / e/at" // random decision was to swap the first substring and to keep the second substring in the same order.

"r/g / e/at" --> "r/g / e/ a/t" // again apply the algorithm recursively, divide "at" to "a/t".

"r/g / e/ a/t" --> "r/g / e/ a/t" // random decision is to keep both substrings in the same order.

The algorithm stops now and the result string is "rgeat" which is s2.

As there is one possible scenario that led s1 to be scrambled to s2, we return true.


Example 2:
Input: s1 = "abcde", s2 = "caebd"
Output: false

Example 3:
Input: s1 = "a", s2 = "a"
Output: true

Constraints:
`s1.length == s2.length`
`1 <= s1.length <= 30`
`s1` and `s2` consist of lower-case English letters.

### Solution 
 To solve the problem of determining if one string is a scrambled version of another string using a recursive approach, we can utilize a depth-first search strategy alongside memoization. The core idea is to compare two strings by splitting them into substrings and checking for possible scrambles via recursive exploration.

Let’s break down the solution step-by-step:

### Steps to Solve the Problem:

1. **Base Cases**:
    - If the strings `s1` and `s2` are identical, return `True`.
    - If the sorted versions of `s1` and `s2` do not match, return `False`. This is because if two strings are scrambles of each other, they must consist of the same characters.

2. **Recursive Checking**:
    - Iterate through potential split points in the strings. For each split index:
        - Check two possible cases:
            1. The first halves (`s1[:i]` and `s2[:i]`) and the second halves (`s1[i:]` and `s2[i:]`) match (no swap).
            2. The first half of `s1` and the second half of `s2` are swapped (`s1[:i]` matches `s2[-i:]` and `s1[i:]` matches `s2[:-i]`).
    - Use a memoization (caching strategy) to store already computed results for pairs of `(s1, s2)` to avoid redundant calculations.

3. **Memoization**:
    - Use a dictionary to store the results of previously computed string pairs to enhance performance.

### Implementation

Here's the complete Python code, formatted for LeetCode:



```python
class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        # Memoization dictionary
        memo = dict()

        def dfs(s1, s2):
            # Base case when the strings are equal
            if s1 == s2:
                return True
            
            # Base cases to eliminate unnecessary checks
            if len(s1) != len(s2) or sorted(s1) != sorted(s2):
                return False
            
            # Memoization key
            key = (s1, s2)
            if key in memo:
                return memo[key]
            
            n = len(s1)
            for i in range(1, n):  # i is the split point
                # Check both possibilities
                # Case 1: No swap
                if dfs(s1[:i], s2[:i]) and dfs(s1[i:], s2[i:]):
                    memo[key] = True
                    return True
                
                # Case 2: With swap
                if dfs(s1[:i], s2[-i:]) and dfs(s1[i:], s2[:-i]):
                    memo[key] = True
                    return True
            
            # Cache the result
            memo[key] = False
            return False

        return dfs(s1, s2)

```

### Explanation of the Code:

- We define a class `Solution` with a method `isScramble` that takes two strings, `s1` and `s2`.
- We use the nested function `dfs` to perform our recursive checks.
- We handle base cases directly in the `dfs` function to check immediate equality or character mismatches via sorted versions.
- We loop through potential split indices and call `dfs` recursively for both the scenarios (with and without swaps).
- We store results in the `memo` dictionary to optimize future calls with the same string pairs.

### Complexity:
- **Time Complexity**: The worst-case scenario can be quite high due to the recursive structure but is generally manageable within the constraints given the memoization.
- **Space Complexity**: O(n^2) for storing the memoized results (where n is the length of the input strings).

This implementation can be directly copied into LeetCode for testing and should efficiently determine if `s2` is a scrambled string of `s1`.

# 91. Decode Ways

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

Given a string `s` containing only digits, return the number of ways to decode it.

The answer is guaranteed to fit in a 32-bit integer.


Example 1:
Input: s = "12"
Output: 2
Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).


Example 2:
Input: s = "226"
Output: 3
Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).


Example 3:
Input: s = "0"
Output: 0
Explanation: There is no character that is mapped to a number starting with 0.

The only valid mappings with 0 are 'J' -> "10" and 'T' -> "20", neither of which start with 0.

Hence, there are no valid ways to decode this since all digits need to be mapped.


Example 4:
Input: s = "06"
Output: 0
Explanation: "06" cannot be mapped to "F" because of the leading zero ("6" is different from "06").


Constraints:
`1 <= s.length <= 100`
`s` contains only digits and may contain leading zero(s).

### Solution 
 Here is a Python solution for the problem of decoding an encoded message using dynamic programming. We will implement the function as per the LeetCode solution format, including a detailed explanation of the approach employed.

### Python Code



```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if not s or s[0] == '0':
            return 0
        
        n = len(s)
        # dp[i] will hold the number of ways to decode the string up to the i-th index
        dp = [0] * (n + 1)
        
        # Base cases
        dp[0] = 1  # There is one way to decode an empty string
        dp[1] = 1  # There is one way to decode a non-zero single character

        for i in range(2, n + 1):
            # Check for the single digit
            single_digit = int(s[i - 1:i])  # This corresponds to s[i-1]
            if 1 <= single_digit <= 9:
                dp[i] += dp[i - 1]
            
            # Check for the two-digit number
            two_digit = int(s[i - 2:i])  # This corresponds to s[i-2:i]
            if 10 <= two_digit <= 26:
                dp[i] += dp[i - 2]

        return dp[n]

# Example usage:
# sol = Solution()
# print(sol.numDecodings("12"))  # Output: 2
# print(sol.numDecodings("226"))  # Output: 3
# print(sol.numDecodings("0"))  # Output: 0
# print(sol.numDecodings("06"))  # Output: 0

```

### Explanation

1. **Initialization**:
   - We start by checking if the string `s` is empty or starts with a '0'. If so, we return `0` since it cannot be decoded.
   - We set up a `dp` array of size `n + 1`, where `n` is the length of the string `s`. `dp[i]` will represent the number of ways to decode the string up to index `i-1` (0-based indexing).

2. **Base cases**:
   - `dp[0]` is initialized to `1` because there is one way to decode an empty string.
   - `dp[1]` is also set to `1` because if the first character is not zero, it can be decoded in exactly one way.

3. **Dynamic Programming Loop**:
   - We iterate over the string starting from index `2` to `n`.
   - For each index `i`, we check two possibilities:
     - The individual digit (from `s[i-1]`): If the digit (1-9) can correspond to a letter ('A' = 1 to 'I' = 9), we add `dp[i - 1]` to `dp[i]`.
     - The two-digit number (from `s[i-2:i]`): If the two-digit number (10-26) is valid, we add `dp[i - 2]` to `dp[i]`.

4. **Return the Result**:
   - Finally, we return `dp[n]`, which contains the total number of ways to decode the given string.

This dynamic programming approach efficiently calculates the number of decode ways in linear time, `O(n)`, making it suitable for the given problem constraints.

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
 To solve the problem of generating all possible valid IP addresses from a given string composed solely of digits, we need to break down the problem into manageable steps. A valid IP address consists of four octets (numbers) ranging from 0 to 255, each separated by a dot. Furthermore, octets cannot contain leading zeros unless the octet is exactly "0". 

Here’s a structured approach to generate all possible valid IPs:

### Approach:

1. **Backtracking**: We'll use a backtracking approach to explore all possible splits of the string into 4 segments.
2. **Validation**: For each possible segment, we check if it forms a valid octet (between 0 and 255 and does not have leading zeros).
3. **Base Conditions**: If we have successfully formed 4 octets and exhausted the string, we capture that combination.
4. **Recursive Exploration**: We will iterate over possible lengths of each segment (1 to 3). We continue recursively until we either form a valid IP address or exhaust possibilities.

### Python Code:

Here’s how the code looks, following the LeetCode format:



```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def backtrack(start: int, path: List[str]):
            # If we reach the end of the string and we have 4 parts, we save the result
            if start == len(s) and len(path) == 4:
                res.append('.'.join(path))
                return
            
            # If we have 4 parts already but still have characters left, return
            if len(path) == 4:
                return
            
            # Try to take 1, 2, or 3 digits for the current segment
            for length in range(1, 4):
                if start + length > len(s):
                    break
                segment = s[start:start + length]
                
                # Validate the segment
                if self.is_valid(segment):
                    # Add the segment to the path and backtrack
                    path.append(segment)
                    backtrack(start + length, path)
                    path.pop()  # Backtrack

        # Helper function to validate segments
        def is_valid(segment: str) -> bool:
            # Convert segment to int and check for constraints
            if len(segment) == 0 or len(segment) > 3:
                return False
            if segment[0] == '0' and len(segment) > 1:  # Leading zero
                return False
            if int(segment) > 255:  # Out of valid range
                return False
            return True
            
        res = []
        backtrack(0, [])
        return res

```

### Explanation of the Code:

1. **Function Definition**: The main function `restoreIpAddresses` takes a string `s` as input.
2. **Backtracking Logic**:
   - The backtrack function tries to build the IP address from the starting point `start`, while `path` holds the current parts being formed.
   - It checks if it has formed exactly 4 segments, and whether it reached the end of the string.
   - If we've used up 4 parts but still have characters left in the string, it exits.
   - It loops through segments of lengths 1 to 3.
   - For each segment, it checks if the segment is valid and then recursively calls itself to form further segments.
3. **Validation**: The `is_valid` function ensures that:
   - The segment length is appropriate.
   - There are no leading zeros unless the segment is "0".
   - The numeric value doesn't exceed 255.
4. **Results**: All valid combinations are concatenated and collected in `res` and returned as the final output.

### Complexity:
- **Time Complexity**: The backtracking complexity is exponential in nature but manageable within reasonable string lengths.
- **Space Complexity**: O(N) for the recursion stack where N is the length of the input string.

Using this code, you can test various inputs on LeetCode directly for the problem of generating valid IP addresses from a string of digits.

# 97. Interleaving String

### Problem Description 
Given strings `s1`, `s2`, and `s3`, find whether `s3` is formed by an interleaving of `s1` and `s2`.

An interleaving of two strings `s` and `t` is a configuration where they are divided into non-empty substrings such that:
`s = s1 + s2 + ... + sn`
`t = t1 + t2 + ... + tm`
`|n - m| <= 1`
The interleaving is `s1 + t1 + s2 + t2 + s3 + t3 + ...` or `t1 + s1 + t2 + s2 + t3 + s3 + ...`
Note: `a + b` is the concatenation of strings `a` and `b`.


Example 1:
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
Output: true

Example 2:
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
Output: false

Example 3:
Input: s1 = "", s2 = "", s3 = ""
Output: true

Constraints:
`0 <= s1.length, s2.length <= 100`
`0 <= s3.length <= 200`
`s1`, `s2`, and `s3` consist of lowercase English letters.

Follow up: Could you solve it using only `O(s2.length)` additional memory space?

### Solution 
 To solve the problem of determining whether `s3` is formed by an interleaving of `s1` and `s2`, we can utilize a dynamic programming approach. The idea is to create a 2D boolean table that will help us keep track of whether the first `i` characters of `s1` and the first `j` characters of `s2` can form the first `i + j` characters of `s3`.

### Solution Strategy

1. **Initialization**: We will use a 2D list (table) `dp` where `dp[i][j]` indicates if the first `i` characters of `s1` and the first `j` characters of `s2` can form the first `i + j` characters of `s3`. The dimensions of the table will be `(len(s1) + 1) x (len(s2) + 1)` to account for the empty prefixes.

2. **Base Case**: The base case will be where both `s1` and `s2` are empty:
   - `dp[0][0]` is `True` if `s3` is also empty.

3. **Filling the DP Table**:
   - Iterate through each character in `s1` and `s2`, updating the `dp` table based on the conditions:
     - If the current character in `s3` matches the current character in `s1`, then `dp[i][j]` can be set to `dp[i-1][j]` (which means we can form `s3` considering the previous characters of `s1`).
     - If it matches the current character in `s2`, then it can be set to `dp[i][j-1]`.

4. **Final Check**: The value at `dp[len(s1)][len(s2)]` will tell us if `s3` can be formed by interleaving `s1` and `s2`.

### Time Complexity
The time complexity is O(n * m), where n is the length of `s1` and m is the length of `s2`. The space complexity is O(n * m) because of the 2D dp table used.

### Python Code

Here's the complete implementation in Python, following the format needed for LeetCode submission:



```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # Get the lengths of the strings
        len1, len2, len3 = len(s1), len(s2), len(s3)
        
        # If the total lengths do not match, return False
        if len1 + len2 != len3:
            return False
        
        # Initialize a 2D list for dynamic programming
        dp = [[False] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Base case: both s1 and s2 are empty
        dp[0][0] = True
        
        # Fill the first row (using only s2)
        for j in range(1, len2 + 1):
            dp[0][j] = dp[0][j - 1] and s2[j - 1] == s3[j - 1]
        
        # Fill the first column (using only s1)
        for i in range(1, len1 + 1):
            dp[i][0] = dp[i - 1][0] and s1[i - 1] == s3[i - 1]
        
        # Filling in the rest of the dp table
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                dp[i][j] = (dp[i - 1][j] and s1[i - 1] == s3[i + j - 1]) or \
                           (dp[i][j - 1] and s2[j - 1] == s3[i + j - 1])
        
        # The result is in the bottom-right corner of the table
        return dp[len1][len2]

```

### Follow-Up with O(s2.length) Additional Memory Space
To optimize our space complexity to O(m) (where m is the length of `s2`), we can use a 1D array instead of a 2D table. This requires careful indexing since the current state calculation will still depend on the previous state of both `s1` and `s2`. The logic remains the same but needs adjusting to account for using only one row. Here's how that would look:



```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        len1, len2, len3 = len(s1), len(s2), len(s3)
        
        if len1 + len2 != len3:
            return False
        
        dp = [False] * (len2 + 1)
        dp[0] = True
        
        for j in range(1, len2 + 1):
            dp[j] = dp[j - 1] and s2[j - 1] == s3[j - 1]
        
        for i in range(1, len1 + 1):
            dp[0] = dp[0] and s1[i - 1] == s3[i - 1]
            for j in range(1, len2 + 1):
                dp[j] = (dp[j] and s2[j - 1] == s3[i + j - 1]) or \
                         (dp[j - 1] and s1[i - 1] == s3[i + j - 1])
        
        return dp[len2]

```

This code effectively reduces the space to O(m) while keeping the same time complexity of O(n * m). This can be run directly on LeetCode with appropriate test cases.

# 115. Distinct Subsequences

### Problem Description 
Given two strings `s` and `t`, return the number of distinct subsequences of `s` which equals `t`.

A string's subsequence is a new string formed from the original string by deleting some (can be none) of the characters without disturbing the remaining characters' relative positions. (i.e., `"ACE"` is a subsequence of `"ABCDE"` while `"AEC"` is not).

It is guaranteed the answer fits on a 32-bit signed integer.


Example 1:
Input: s = "rabbbit", t = "rabbit"
Output: 3
Explanation:
As shown below, there are 3 ways you can generate "rabbit" from S.

`rabbbit`
`rabbbit`
`rabbbit`

Example 2:
Input: s = "babgbag", t = "bag"
Output: 5
Explanation:
As shown below, there are 5 ways you can generate "bag" from S.

`babgbag`
`babgbag`
`babgbag`
`babgbag`
`babgbag`

Constraints:
`1 <= s.length, t.length <= 1000`
`s` and `t` consist of English letters.

### Solution 
 To solve the problem of finding the number of distinct subsequences of a string `s` that equals a string `t`, we can use dynamic programming. This approach is efficient and suitable given the constraints on the lengths of the strings.

### Explanation

The key idea behind the dynamic programming solution is to use a 2D DP table (array) where `dp[i][j]` represents the number of ways to form the first `j` characters of `t` using the first `i` characters of `s`.

Here’s how we can fill this DP table:

1. **Initialization**:
   - If `t` is an empty string (i.e., `j = 0`), there is exactly one way to form it from any prefix of `s` (by deleting all characters). Thus, `dp[i][0] = 1` for all `i`.
   - If `s` is empty (i.e., `i = 0`), there are zero ways to form a non-empty `t`, so `dp[0][j] = 0` for all `j > 0`.

2. **Filling the DP Table**:
   - We iterate through each character of `s` (indexed by `i`) and `t` (indexed by `j`).
   - If the characters `s[i-1]` and `t[j-1]` match, we can either include this character in our subsequence or exclude it. Thus, `dp[i][j]` will be the sum of:
     - The number of ways to form the first `j` characters of `t` using the first `i-1` characters of `s` (exclude `s[i-1]`).
     - The number of ways to form the first `j-1` characters of `t` using the first `i-1` characters of `s` (include `s[i-1]`).
   - If the characters do not match, we can only exclude the character `s[i-1]`, so `dp[i][j] = dp[i-1][j]`.

3. **Final Result**:
   - The value in `dp[len(s)][len(t)]` will give us the required result, which is the number of distinct subsequences of `s` that equal `t`.

### Python Code

Here's the Python code implementing the above logic:



```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        # Lengths of strings s and t
        m = len(s)
        n = len(t)

        # DP table initialized with zeros
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Base case initialization
        for i in range(m + 1):
            dp[i][0] = 1  # There's one way to match an empty string t

        # Filling the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1]
                else:
                    dp[i][j] = dp[i - 1][j]

        # The answer is in the bottom-right corner of the table
        return dp[m][n]

```

### Usage

You can run this code directly on LeetCode. The class `Solution` contains the method `numDistinct` that takes two strings `s` and `t` as input and returns the number of distinct subsequences matching `t` from `s`. 

### Complexity Analysis
- **Time Complexity**: O(m * n), where `m` is the length of string `s` and `n` is the length of string `t`. We fill up the DP table which has `m + 1` rows and `n + 1` columns.
- **Space Complexity**: O(m * n) for the DP table.

This ensures that the solution efficiently computes the desired result even for the maximum constraints.

# 125. Valid Palindrome

### Problem Description 
Given a string `s`, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.


Example 1:
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.


Example 2:
Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.


Constraints:
`1 <= s.length <= 2 * 105`
`s` consists only of printable ASCII characters.

### Solution 
 To determine if a given string `s` is a palindrome while considering only alphanumeric characters and ignoring cases, we can follow these steps:

1. **Filter Out Non-Alphanumeric Characters**: We need to create a filtered version of the input string `s` that only retains alphanumeric characters.
2. **Case Normalization**: Convert all characters in the filtered string to the same case (lowercase or uppercase) to ensure that the comparison is case-insensitive.
3. **Check for Palindrome**: Compare the filtered string to its reverse. If they are the same, then the original string is a palindrome; otherwise, it is not.

Here's the implementation in Python, structured according to LeetCode's "Solution" class format:



```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        # Step 1: Filter the input string to keep only alphanumeric characters
        filtered_s = ''.join(char.lower() for char in s if char.isalnum())
        
        # Step 2: Check if the filtered string is the same as its reverse
        return filtered_s == filtered_s[::-1]

```

### Detailed Explanation:

- **Filtering the String**:
  - We use a generator expression within the `join` method to iterate over each character in the input string `s`.
  - We check if the character is alphanumeric using `char.isalnum()`.
  - If it is, we convert it to lowercase using `char.lower()` and build a new string `filtered_s`.

- **Checking for Palindrome**:
  - To determine if `filtered_s` is a palindrome, we compare it to its reverse, which we obtain using slicing: `filtered_s[::-1]`.
  - If both are equal, we return `True`, indicating that the string is a palindrome. Otherwise, we return `False`.

### Time Complexity:
The time complexity of this solution is O(n), where n is the length of the input string `s`. This is because we are traversing the string once to filter it and another time to compare it with its reverse.

### Space Complexity:
The space complexity is also O(n) due to the storage of the filtered string.

### Example Usage:
Here is a small example to demonstrate how you can use the implemented function:



```python
solution = Solution()
print(solution.isPalindrome("A man, a plan, a canal: Panama"))  # Output: True
print(solution.isPalindrome("race a car"))  # Output: False

```

This code will correctly assess whether the input strings are palindromes according to the given criteria.

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
 To solve the problem of finding all the shortest transformation sequences from `beginWord` to `endWord`, where each step in the sequence changes exactly one letter and each transformed word must exist in a given word list, we can approach the solution using a breadth-first search (BFS) strategy. This allows us to efficiently traverse possible words while ensuring we capture the shortest paths. 

Here's a detailed explanation of the steps and the implementation in Python that can be run directly on the LeetCode platform:

### Steps:

1. **Word Graph Construction**: 
   - Create a general form of the words in the dictionary by replacing one letter with `*`. For example, `hot` can be replaced with `*ot`, `h*t`, and `ho*`.
  
2. **BFS Search**:
   - Start from the `beginWord` and perform BFS to explore all possible transformations. During this traversal, keep track of the parent nodes (previous words in the transformation sequence) for each word to reconstruct paths later.

3. **Path Reconstruction**: 
   - Once we reach the `endWord`, use the parent information to backtrack and construct all valid transformation sequences.

4. **Returning Results**: 
   - If no path exists, return an empty list. Otherwise, return all the paths found during the reconstruction step.

Here is the Python code that implements the above logic:



```python
from collections import defaultdict, deque
from typing import List

class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        wordSet = set(wordList)
        if endWord not in wordSet:
            return []

        # Create a dictionary to hold intermediate words.
        neighbors = defaultdict(list)
        for word in wordSet:
            for i in range(len(word)):
                neighbors[word[:i] + "*" + word[i + 1:]].append(word)

        # BFS to find shortest paths
        queue = deque([beginWord])
        visited = set([beginWord])
        found_end = False
        parents = defaultdict(list)  # To keep track of parents in the path

        while queue and not found_end:
            local_visited = set()  # To track visited words in the current level
            for _ in range(len(queue)):
                word = queue.popleft()
                for i in range(len(word)):
                    for next_word in neighbors[word[:i] + "*" + word[i + 1:]]:
                        if next_word == endWord:
                            found_end = True
                        if next_word not in visited:  # If next_word wasn't seen before
                            local_visited.add(next_word)
                            parents[next_word].append(word)
                        if next_word not in visited and next_word not in queue:
                            queue.append(next_word)
            visited.update(local_visited)

        # If endWord was not found, return empty list
        if not found_end:
            return []

        # Backtrack to find all paths
        def backtrack(word):
            if word == beginWord:
                return [[beginWord]]
            return [[beginWord] + path for parent in parents[word] for path in backtrack(parent)]

        return backtrack(endWord)

```

### Explanation of the Code:

- **Data Structures**: 
  - `neighbors`: A dictionary that maps transformed words (with wildcard replacements) to their corresponding valid words.
  - `visited`: A set to track words that have been fully explored.
  - `parents`: A dictionary that maps each word to its potential predecessor words in the transformation sequence.

- **BFS Implementation**:
  - We maintain a queue for BFS and explore all transformations by iterating through each character of the current word and replacing it with a `*`.
  - If we find that a word matches `endWord`, we set a flag.

- **Path Reconstruction**:
  - We define a recursive function `backtrack` that builds paths from `endWord` back to `beginWord` using the `parents` mapping.

By following this structure, we ensure that we not only find all shortest paths but also efficiently manage our search space, which is critical given the size constraints of the problem.

# 151. Reverse Words in a String

### Problem Description 
Given an input string `s`, reverse the order of the words.

A word is defined as a sequence of non-space characters. The words in `s` will be separated by at least one space.

Return a string of the words in reverse order concatenated by a single space.

Note that `s` may contain leading or trailing spaces or multiple spaces between two words. The returned string should only have a single space separating the words. Do not include any extra spaces.


Example 1:
Input: s = "the sky is blue"
Output: "blue is sky the"

Example 2:
Input: s = "  hello world  "
Output: "world hello"
Explanation: Your reversed string should not contain leading or trailing spaces.


Example 3:
Input: s = "a good   example"
Output: "example good a"
Explanation: You need to reduce multiple spaces between two words to a single space in the reversed string.


Example 4:
Input: s = "  Bob    Loves  Alice   "
Output: "Alice Loves Bob"

Example 5:
Input: s = "Alice does not even like bob"
Output: "bob like even not does Alice"

Constraints:
`1 <= s.length <= 104`
`s` contains English letters (upper-case and lower-case), digits, and spaces `' '`.

There is at least one word in `s`.

Follow up: Could you solve it in-place with `O(1)` extra space?

### Solution 
 To tackle the problem of reversing the order of words in a string `s`, we'll take the following steps in the solution:

1. **Trim the input string**: This removes any leading or trailing spaces from `s`.
2. **Split the string into words**: We will split the trimmed string into a list of words. This will also handle the case of multiple spaces between words since the `split()` method with no arguments will automatically strip out excess whitespace.
3. **Reverse the list of words**: Once we have the list of words, we can easily reverse their order.
4. **Join the reversed list into a single string**: Finally, we will join the reversed list of words back into a string, separating them with a single space.

### Python Code Implementation

Let's present the solution in a format suitable for LeetCode:



```python
class Solution:
    def reverseWords(self, s: str) -> str:
        # Step 1: Trim the input string to remove leading/trailing spaces.
        trimmed_s = s.strip()
        
        # Step 2: Split the trimmed string into words.
        words = trimmed_s.split()
        
        # Step 3: Reverse the list of words.
        reversed_words = words[::-1]
        
        # Step 4: Join the reversed list into a single string.
        result = ' '.join(reversed_words)
        
        return result

```

### Explanation of the Code

1. **`trimmed_s = s.strip()`**: 
   - This removes any spaces from the beginning or end of the string `s`. 
   - For example, for the input `"  hello world  "`, the result will be `"hello world"`.

2. **`words = trimmed_s.split()`**:
   - This splits the `trimmed_s` into a list of words based on whitespace. 
   - It effectively handles multiple spaces; if `trimmed_s` is `"hello  world"`, the words list will be `["hello", "world"]`.

3. **`reversed_words = words[::-1]`**:
   - This uses slicing to reverse the order of the list `words`. 
   - For example, if `words` is `["hello", "world"]`, `reversed_words` will be `["world", "hello"]`.

4. **`result = ' '.join(reversed_words)`**:
   - This creates a single string by joining the elements of `reversed_words` with a single space `' '` between each word. 
   - The final output will be `"world hello"`.

### Complexity Analysis
- **Time Complexity**: O(n), where n is the length of the string `s`. We process the string a fixed number of times (trimming, splitting, reversing, and joining).
- **Space Complexity**: O(n) since we are using space for the list of words and the final output.

### Follow-up: In-Place Solution
The current approach returns a new string and uses additional space proportional to the number of words. To achieve the in-place solution with O(1) additional space would require modifying the input string directly, which is more complex. Generally, such in-place manipulations could be handled through two-pointer techniques or in other contexts, but for simplicity and clarity, the approach provided above is the conventional way to handle such string transformations in Python.

# 157. Read N Characters Given Read4

### Problem Description 
Given a `file` and assume that you can only read the file using a given method `read4`, implement a method to read `n` characters.

Method read4: 
The API `read4` reads four consecutive characters from `file`, then writes those characters into the buffer array `buf4`.

The return value is the number of actual characters read.

Note that `read4()` has its own file pointer, much like `FILE *fp` in C.

Definition of read4:
    Parameter:  char[] buf4
    Returns:    int
buf4[] is a destination, not a source. The results from read4 will be copied to buf4[].

Below is a high-level example of how `read4` works:
File file("abcde`"); // File is "`abcde`", initially file pointer (fp) points to 'a'
char[] buf4 = new char[4]; // Create buffer with enough space to store characters
read4(buf4); // read4 returns 4. Now buf4 = "abcd", fp points to 'e'
read4(buf4); // read4 returns 1. Now buf4 = "e", fp points to end of file
read4(buf4); // read4 returns 0. Now buf4 = "", fp points to end of file`
Method read:
By using the `read4` method, implement the method read that reads `n` characters from `file` and store it in the buffer array `buf`. Consider that you cannot manipulate `file` directly.

The return value is the number of actual characters read.

Definition of read: 
    Parameters:	char[] buf, int n
    Returns:	int
buf[] is a destination, not a source. You will need to write the results to buf[].

Note:
Consider that you cannot manipulate the file directly. The file is only accessible for `read4` but not for `read`.

The `read` function will only be called once for each test case.

You may assume the destination buffer array, `buf`, is guaranteed to have enough space for storing `n` characters.


Example 1:
Input: file = "abc", n = 4
Output: 3
Explanation: After calling your read method, buf should contain "abc". We read a total of 3 characters from the file, so return 3.

Note that "abc" is the file's content, not buf. buf is the destination buffer that you will have to write the results to.


Example 2:
Input: file = "abcde", n = 5
Output: 5
Explanation: After calling your read method, buf should contain "abcde". We read a total of 5 characters from the file, so return 5.


Example 3:
Input: file = "abcdABCD1234", n = 12
Output: 12
Explanation: After calling your read method, buf should contain "abcdABCD1234". We read a total of 12 characters from the file, so return 12.


Example 4:
Input: file = "leetcode", n = 5
Output: 5
Explanation: After calling your read method, buf should contain "leetc". We read a total of 5 characters from the file, so return 5.


Constraints:
`1 <= file.length <= 500`
`file` consist of English letters and digits.

`1 <= n <= 1000`

### Solution 
 To solve this problem, we need to implement a `read` method that uses the `read4` method provided by the API to read up to `n` characters from a file. The solution is structured to handle various scenarios, such as when fewer than `n` characters are available.

Here’s how we can approach the problem:

1. **Understand the read4 Method**: 
   - `read4` reads up to 4 characters from a file and stores them in a given buffer. It returns the number of characters actually read.

2. **Define the Read Method**:
   - The `read` method accepts two parameters: a destination buffer `buf` and an integer `n` representing the number of characters to read.
   - It will call `read4` multiple times until we have read `n` characters or reached the end of the file.

3. **Buffering Logic**:
   - We need to keep track of the total number of characters read.
   - Each time we call `read4`, we need to decide how many characters to copy to `buf`, which is the lesser of the number of characters we have read (up to 4) and the remaining characters we still need to read (up to `n`).

4. **Implementation Steps**:
   - Use a loop to repeatedly call `read4` until we have read enough characters or `read4` indicates that there are no more characters to read (returns 0).
   - Copy characters from the temporary buffer used by `read4` to the destination buffer `buf` until we’ve read `n` characters or we run out of characters.

Here’s the complete Python code that follows this logic and can be run in the LeetCode environment:



```python
# Assuming read4 is already defined in the environment
def read4(buf4):
    # This is a placeholder for the actual implementation of read4
    pass

class Solution:
    def read(self, buf: List[str], n: int) -> int:
        # Initialize the number of characters read
        total_chars_read = 0
        # Temporary buffer to store characters read from read4
        buf4 = [''] * 4
        
        while total_chars_read < n:
            # Read up to 4 characters into buf4
            chars_read = read4(buf4)
            # If no characters were read, end of file is reached
            if chars_read == 0:
                break
            
            # Determine how many characters we will copy to buf
            for i in range(min(chars_read, n - total_chars_read)):
                buf[total_chars_read] = buf4[i]
                total_chars_read += 1
            
        return total_chars_read

```

### Explanation of the Code:
1. **Initialization**:
   - A temporary buffer `buf4` of size 4 is created to hold the characters read from the file.
   - The variable `total_chars_read` keeps count of the total number of characters we have read into `buf`.

2. **Looping Until Enough Characters Read**:
   - The while loop continues as long as we haven't read `n` characters yet.
   - The `read4` function is called to fill `buf4`, and it returns the number of characters actually read.

3. **Copying Characters**:
   - If `read4` returns 0, we break the loop since there are no more characters to read.
   - We copy characters from `buf4` to `buf`, ensuring that we do not exceed either the number of characters returned by `read4` or the remaining slots in `buf`.

4. **Returning the Result**:
   - Finally, the function returns the total number of characters successfully read into the `buf` array.

### Complexity:
- **Time Complexity**: O(N), where N is the number of characters we want to read (up to `n`), since in the worst case, we may read in chunks of 4.
- **Space Complexity**: O(1) beyond the input buffers, as we only use a fixed-size temporary buffer (`buf4`).

This implementation satisfies the requirements of the problem and handles the edge cases as expected.

# 158. Read N Characters Given Read4 II - Call multiple times

### Problem Description 
Given a `file` and assume that you can only read the file using a given method `read4`, implement a method `read` to read `n` characters. Your method `read` may be called multiple times.

Method read4: 
The API `read4` reads four consecutive characters from `file`, then writes those characters into the buffer array `buf4`.

The return value is the number of actual characters read.

Note that `read4()` has its own file pointer, much like `FILE *fp` in C.

Definition of read4:
    Parameter:  char[] buf4
    Returns:    int
buf4[] is a destination, not a source. The results from read4 will be copied to buf4[].

Below is a high-level example of how `read4` works:
File file("abcde`"); // File is "`abcde`", initially file pointer (fp) points to 'a'
char[] buf4 = new char[4]; // Create buffer with enough space to store characters
read4(buf4); // read4 returns 4. Now buf4 = "abcd", fp points to 'e'
read4(buf4); // read4 returns 1. Now buf4 = "e", fp points to end of file
read4(buf4); // read4 returns 0. Now buf4 = "", fp points to end of file`
Method read:
By using the `read4` method, implement the method read that reads `n` characters from `file` and store it in the buffer array `buf`. Consider that you cannot manipulate `file` directly.

The return value is the number of actual characters read.

Definition of read: 
    Parameters:	char[] buf, int n
    Returns:	int
buf[] is a destination, not a source. You will need to write the results to buf[].

Note:
Consider that you cannot manipulate the file directly. The file is only accessible for `read4` but not for `read`.

The read function may be called multiple times.

Please remember to RESET your class variables declared in Solution, as static/class variables are persisted across multiple test cases. Please see here for more details.

You may assume the destination buffer array, `buf`, is guaranteed to have enough space for storing `n` characters.

It is guaranteed that in a given test case the same buffer `buf` is called by `read`.


Example 1:
Input: file = "abc", queries = [1,2,1]
Output: [1,2,0]
Explanation: The test case represents the following scenario:
File file("abc");
Solution sol;
sol.read(buf, 1); // After calling your read method, buf should contain "a". We read a total of 1 character from the file, so return 1.

sol.read(buf, 2); // Now buf should contain "bc". We read a total of 2 characters from the file, so return 2.

sol.read(buf, 1); // We have reached the end of file, no more characters can be read. So return 0.

Assume buf is allocated and guaranteed to have enough space for storing all characters from the file.


Example 2:
Input: file = "abc", queries = [4,1]
Output: [3,0]
Explanation: The test case represents the following scenario:
File file("abc");
Solution sol;
sol.read(buf, 4); // After calling your read method, buf should contain "abc". We read a total of 3 characters from the file, so return 3.

sol.read(buf, 1); // We have reached the end of file, no more characters can be read. So return 0.


Constraints:
`1 <= file.length <= 500`
`file` consist of English letters and digits.

`1 <= queries.length <= 10`
`1 <= queries[i] <= 500`

### Solution 
 To solve the problem, we need to create a method called `read` that utilizes a helper method called `read4`. The `read4` method reads up to 4 characters from a file and writes them into a provided buffer, returning the number of characters read. The challenge is to implement the `read` method which reads exactly `n` characters into a provided buffer, handling the situation where the end of the file might be reached before reading `n` characters.

### Implementation Steps:
1. **Initialize Variables**: We need to keep track of how many characters have been read previously and how much more we need to read.
2. **Loop Until `n` Characters Are Read**: We will repeatedly call `read4` as long as we can read more characters and have not fulfilled the request `n`.
3. **Handle Partial Reads**: If less than 4 characters are read in a call to `read4`, we need to adjust for the end of the file and ensure not to read past it.
4. **Return the Number of Characters Read**: Finally, we will return the total number of characters we have read.

Here's a Python implementation that follows this approach:



```python
# Assume we have the read4 API defined somewhere

def read4(buf4):
    # This function is predefined and should read 4 characters from file into buf4
    pass

class Solution:
    def __init__(self):
        # Initialize attributes to remember the state between calls
        self.buffer = [''] * 4  # Buffer for read4
        self.buffer_index = 0   # Current index in the buffer
        self.buffer_count = 0    # How many characters were read in the last read4 call

    def read(self, buf, n):
        i = 0  # Number of characters written to buf
        while i < n:
            # If we have nothing left in the buffer, read from file
            if self.buffer_index == 0:
                self.buffer_count = read4(self.buffer)  # Read from the file
            # If we read no characters, we reached the end of the file
            if self.buffer_count == 0:
                break
            
            # Copy characters from buffer to buf
            while i < n and self.buffer_index < self.buffer_count:
                buf[i] = self.buffer[self.buffer_index]
                i += 1
                self.buffer_index += 1
            
            # If we exhausted the buffer, reset the index for the next read
            if self.buffer_index >= self.buffer_count:
                self.buffer_index = 0

        return i  # Return the number of characters read

```

### Explanation:
1. **Initialization**:
   - `self.buffer`: A fixed array of 4 characters to hold the data read from `read4`.
   - `self.buffer_index`: An index to track our position within the buffer.
   - `self.buffer_count`: How many valid characters are currently in the buffer.

2. **Reading Logic**:
   - The `read` method takes a buffer `buf` and the number `n` of characters to read.
   - We loop until we either reach `n` characters or hit the end of the file.
   - When the buffer is exhausted (`self.buffer_index` == 0), we call `read4`.
   - If `read4` returns 0, we exit the loop since we've reached the end of the file.
   - Using a nested loop, we copy characters from the `self.buffer` to `buf` until either the destination buffer (`buf`) is filled up to `n` or we've used all characters in the current buffer.

3. **Resetting State**:
   - When we finish reading from `self.buffer`, if there are no characters left, we reset `self.buffer_index` to 0 so that we can read fresh data again next time.

### Conclusion:
The solution effectively implements reading from a file using a restricted method by buffering the read data and handling multiple reads. The approach ensures that the end of the file is managed correctly and is efficient by minimizing the number of calls made to `read4`.

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
 To solve the problem of finding the length of the longest substring that contains at most two distinct characters, we can utilize the sliding window technique. This approach allows us to efficiently manage the window of characters we are currently examining and adjust it as necessary. Let's go through the solution step-by-step.

### Approach:

1. **Use a Sliding Window**: 
   - We'll maintain a window defined by two pointers: `left` (start of the window) and `right` (end of the window).
   - We'll expand the `right` pointer to include more characters until we have more than two distinct characters in the current window.

2. **Count Distinct Characters**:
   - We can use a dictionary (or a simple array) to keep track of how many times each character appears in the current window.
   - If the count of distinct characters exceeds two, we will increment the `left` pointer to shrink the window from the left, ensuring that it contains at most two distinct characters.

3. **Keep Track of the Maximum Length**:
   - During the process, we'll maintain a variable to keep the maximum length of substrings that qualify (those containing at most two distinct characters).

### Detailed Implementation:

Here's how the implementation looks in Python:



```python
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        if len(s) < 3:
            return len(s)
        
        left = 0
        right = 0
        max_length = 2
        char_count = {}
        
        while right < len(s):
            # Add the right character to the dictionary
            char_count[s[right]] = char_count.get(s[right], 0) + 1
            
            # Move the right pointer
            right += 1
            
            # When we have more than 2 distinct characters, we need to shrink window
            while len(char_count) > 2:
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]
                left += 1
            
            # Update the max_length
            max_length = max(max_length, right - left)
        
        return max_length

# Example usage:
# sol = Solution()
# print(sol.lengthOfLongestSubstringTwoDistinct("eceba"))  # Output: 3
# print(sol.lengthOfLongestSubstringTwoDistinct("ccaabbb"))  # Output: 5

```

### Explanation of the Code:

1. **Edge Case Handling**:
   - If the string length is less than 3, it returns the length of the string since we can include any single or double letters.

2. **Initial Setup**:
   - We initialize two pointers, `left` and `right`, starting at the beginning of the string.
   - We set `max_length` to track the longest valid substring found, initialized to 2.
   - We use a dictionary `char_count` to count occurrences of each character within the current window defined by `left` and `right`.

3. **Sliding Window Logic**:
   - We traverse through the string using the `right` pointer. For each character, we add it to `char_count`.
   - If our dictionary grows to contain more than two distinct characters, we enter a while loop to shrink the window from the left. We reduce the count of the character at `left` and remove it from the dictionary if its count drops to zero.
   - After adjusting the left pointer, we calculate the current valid window size and update `max_length` if the current window size (`right - left`) is larger.

4. **Final Output**:
   - The function returns the maximum length found after processing the entire string.

This solution is efficient with a time complexity of O(n) since both `left` and `right` pointers traverse the string at most once, and space complexity is O(1) in terms of distinct character count since we're limited to the character set of English letters.

# 161. One Edit Distance

### Problem Description 
Given two strings `s` and `t`, return `true` if they are both one edit distance apart, otherwise return `false`.

A string `s` is said to be one distance apart from a string `t` if you can:
Insert exactly one character into `s` to get `t`.

Delete exactly one character from `s` to get `t`.

Replace exactly one character of `s` with a different character to get `t`.


Example 1:
Input: s = "ab", t = "acb"
Output: true
Explanation: We can insert 'c' into s to get t.


Example 2:
Input: s = "", t = ""
Output: false
Explanation: We cannot get t from s by only one step.


Example 3:
Input: s = "a", t = ""
Output: true

Example 4:
Input: s = "", t = "A"
Output: true

Constraints:
`0 <= s.length <= 104`
`0 <= t.length <= 104`
`s` and `t` consist of lower-case letters, upper-case letters and/or digits.

### Solution 
 To solve the problem of determining whether two strings `s` and `t` are one edit distance apart, we need to check three possible operations: insertion, deletion, and replacement. Here’s a comprehensive solution that adheres to the LeetCode format, including edge cases and comments for clarity.

### Explanation:

- **Insertion**: To transform `s` into `t` by inserting one character into `s`, the lengths of `s` and `t` must differ by exactly one character (i.e., `len(s) + 1 == len(t)`).
  
- **Deletion**: This is essentially the reverse of insertion, where we can remove one character from `t` to get `s`. This would mean that `len(s)` must be exactly one character less than `len(t)` (i.e., `len(s) == len(t) + 1`).
  
- **Replacement**: When both strings are of equal length, we can check how many characters differ between the two strings. If exactly one character is different, we can replace that character.

### Python Code:

Here’s the Python implementation that encapsulates the above logic:



```python
def isOneEditDistance(s: str, t: str) -> bool:
    len_s, len_t = len(s), len(t)
    
    # Check if the length difference is more than 1
    if abs(len_s - len_t) > 1:
        return False
    
    # Initialize pointers for both strings
    i, j = 0, 0
    found_difference = False
    
    while i < len_s and j < len_t:
        if s[i] != t[j]:
            if found_difference:  # If we have already found a difference
                return False
            
            found_difference = True  # Mark that we've found a difference
            
            # Handle the case of insertion or deletion
            if len_s < len_t:  # s is shorter than t
                j += 1  # Move pointer in t forward
            elif len_s > len_t:  # s is longer than t
                i += 1  # Move pointer in s forward
            else:  # Both lengths are equal, so we can only replace
                i += 1
                j += 1
        else:
            i += 1
            j += 1
            
    return True  # If we complete the loop, it means we have either found no differences or exactly one difference

```

### Detailed Breakdown of the Code:
- First, we calculate the lengths of both strings `s` and `t`.
- If the absolute difference in lengths is greater than one, we immediately return `False` because they cannot be one edit distance apart.
- We maintain two pointers `i` and `j` to iterate through both strings, alongside a boolean variable `found_difference` to track whether we have already found an edit (difference).
- We loop through both strings until we hit the end of either one.
  - If characters at positions `i` and `j` are equal, we increment both pointers.
  - If there's a mismatch:
    - If `found_difference` is already `True`, it means we have encountered a second difference, and we return `False`.
    - If there's no prior difference, we set `found_difference` to `True` and check the lengths to decide which pointer to advance.
- Finally, if we finish iterating without returning `False`, it indicates we have either one or no differences, leading to a valid return of `True`.

### Running the Code:
You can directly copy and paste this code into LeetCode to check against the provided test cases.

# 165. Compare Version Numbers

### Problem Description 
Given two version numbers, `version1` and `version2`, compare them.

Version numbers consist of one or more revisions joined by a dot `'.'`. Each revision consists of digits and may contain leading zeros. Every revision contains at least one character. Revisions are 0-indexed from left to right, with the leftmost revision being revision 0, the next revision being revision 1, and so on. For example `2.5.33` and `0.1` are valid version numbers.

To compare version numbers, compare their revisions in left-to-right order. Revisions are compared using their integer value ignoring any leading zeros. This means that revisions `1` and `001` are considered equal. If a version number does not specify a revision at an index, then treat the revision as `0`. For example, version `1.0` is less than version `1.1` because their revision 0s are the same, but their revision 1s are `0` and `1` respectively, and `0 < 1`.

Return the following:
If `version1 < version2`, return `-1`.

If `version1 > version2`, return `1`.

Otherwise, return `0`.


Example 1:
Input: version1 = "1.01", version2 = "1.001"
Output: 0
Explanation: Ignoring leading zeroes, both "01" and "001" represent the same integer "1".


Example 2:
Input: version1 = "1.0", version2 = "1.0.0"
Output: 0
Explanation: version1 does not specify revision 2, which means it is treated as "0".


Example 3:
Input: version1 = "0.1", version2 = "1.1"
Output: -1
Explanation: version1's revision 0 is "0", while version2's revision 0 is "1". 0 < 1, so version1 < version2.


Example 4:
Input: version1 = "1.0.1", version2 = "1"
Output: 1

Example 5:
Input: version1 = "7.5.2.4", version2 = "7.5.3"
Output: -1

Constraints:
`1 <= version1.length, version2.length <= 500`
`version1` and `version2` only contain digits and `'.'`.

`version1` and `version2` are valid version numbers.

All the given revisions in `version1` and `version2` can be stored in a 32-bit integer.

### Solution 
 To solve the problem of comparing two version numbers represented as strings, we'll follow a structured approach as detailed below. The primary goal is to split the version strings into their respective revisions, compare these revisions as integers while accounting for leading zeros, and handle cases where one version has fewer revisions than the other.

### Explanation of the Approach:

1. **Splitting Version Numbers**: We can use the string `.split('.')` method to divide the version strings into lists based on the dot separator.

2. **Normalizing Length**: After splitting, we need to ensure that both lists of revisions are of the same length. This involves padding the shorter list with zeros.

3. **Comparing Revisions**: For each corresponding pair of revisions, convert them to integers (this automatically handles leading zeros) and compare them. 

4. **Return Results**: Based on the comparison, return:
   - `-1` if `version1` is less than `version2`,
   - `1` if `version1` is greater than `version2`,
   - `0` if they are equal.

### Python Code Implementation:

Here is the implementation in Python formatted to be directly runnable on LeetCode:



```python
class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        # Split the version numbers by the dot character
        v1_parts = version1.split('.')
        v2_parts = version2.split('.')
        
        # Find the maximum length of the two version lists
        max_length = max(len(v1_parts), len(v2_parts))
        
        # Normalize both version lists by appending '0's if necessary
        for i in range(max_length):
            # Get the value of the current revision or '0' if it doesn't exist
            v1_revision = int(v1_parts[i]) if i < len(v1_parts) else 0
            v2_revision = int(v2_parts[i]) if i < len(v2_parts) else 0
            
            # Compare the current revisions
            if v1_revision < v2_revision:
                return -1
            elif v1_revision > v2_revision:
                return 1
        
        # If all revisions are the same
        return 0

```

### Detailed Explanation of Code:

- **Class Definition**: We define a `Solution` class with a method `compareVersion`.

- **Input Splitting**: The input strings `version1` and `version2` are split into lists `v1_parts` and `v2_parts` using the `split` method.

- **Length Normalization**: We compute `max_length` to determine how many revisions we need to compare. This allows us to iterate through both lists safely.

- **Padding with Zeros**: In the loop, we check if the current index exceeds the length of either version list. If so, we treat the missing revision as `0`.

- **Integer Comparison**: The conversion of revision parts to integers is done inline during comparison, which discards any leading zeros automatically.

- **Final Comparison**: If all compared revisions are equal, we return `0` indicating both version numbers are the same.

This solution effectively handles all edge cases and respects the constraints laid out in the problem statement.

# 186. Reverse Words in a String II

### Problem Description 
Given a character array `s`, reverse the order of the words.

A word is defined as a sequence of non-space characters. The words in `s` will be separated by a single space.


Example 1:
Input: s = ["t","h","e"," ","s","k","y"," ","i","s"," ","b","l","u","e"]
Output: ["b","l","u","e"," ","i","s"," ","s","k","y"," ","t","h","e"]

Example 2:
Input: s = ["a"]
Output: ["a"]

Constraints:
`1 <= s.length <= 105`
`s[i]` is an English letter (uppercase or lowercase), digit, or space `' '`.

There is at least one word in `s`.

`s` does not contain leading or trailing spaces.

All the words in `s` are guaranteed to be separated by a single space.

Follow up: Could you do it in-place without allocating extra space?

### Solution 
 Sure! Below is a Python solution for the problem of reversing the order of words in a character array `s`. This solution follows the LeetCode solution format, which allows the code to be directly run on the LeetCode platform.

### Explanation of the Approach

1. **Reverse Entire Array**: The first step is to reverse the entire array `s`. This will change the order of characters in the array, putting the last word in the first position and the first word in the last position, but the characters within each word will be reversed.

2. **Reverse Each Word**: After the entire array is reversed, the next step is to reverse each word back to its original character order. By doing this, we turn the reversed positions of the words into their correct order.

3. **In-Place Modification**: The solution modifies the array in-place, as required by the problem constraints. Since we only need to perform a constant amount of extra space usage for indices, this meets the follow-up requirements.

### Python Code

Below is the code implementation following the conventions used in LeetCode problems:



```python
class Solution:
    def reverseWords(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        # Helper function to reverse a segment of the list
        def reverse_segment(start: int, end: int) -> None:
            while start < end:
                s[start], s[end] = s[end], s[start]
                start += 1
                end -= 1
        
        # Step 1: Reverse the entire character array
        reverse_segment(0, len(s) - 1)
        
        # Step 2: Reverse each word in the reversed array
        n = len(s)
        start = 0
        
        for i in range(n):
            if s[i] == ' ' or i == n - 1:
                # If we have reached the end of a word or the end of the array
                if i == n - 1:
                    reverse_segment(start, i)  # Reverse the last word
                else:
                    reverse_segment(start, i - 1)  # Reverse the current word
                start = i + 1  # Move start to the next word
        
        # The list s has now been modified in-place to reverse the words

```

### Detailed Steps in the Code

1. **Reverse Function**: This helper function `reverse_segment` is defined to take two indices and reverse the characters in the segment from `start` to `end`.

2. **Full Array Reversal**: The first call to `reverse_segment(0, len(s) - 1)` reverses the entire character array.

3. **Word-by-Word Reversal**: A loop iterates through each character in `s`:
   - When a space is encountered or the end of the array is reached, the code checks if it's the end of a word:
     - If it's the end of the array, we reverse the last word.
     - Otherwise, we reverse the segment from the start index to one position before the current index.
   - The `start` index is updated to point to the beginning of the next word.

### Usage

You can use this class and method directly in the LeetCode platform. By following this format, the solution adheres to the expected input-output structure of LeetCode. Make sure to import necessary modules if required before running in an environment.

# 214. Shortest Palindrome

### Problem Description 
You are given a string `s`. You can convert `s` to a palindrome by adding characters in front of it.

Return the shortest palindrome you can find by performing this transformation.


Example 1:
Input: s = "aacecaaa"
Output: "aaacecaaa"

Example 2:
Input: s = "abcd"
Output: "dcbabcd"

Constraints:
`0 <= s.length <= 5 * 104`
`s` consists of lowercase English letters only.

### Solution 
 To solve the problem of finding the shortest palindrome by adding characters to the front of a given string `s`, we can use a technique that combines the properties of string matching and reversibility. The idea is to efficiently find the longest palindromic prefix of the input string and use it to construct the shortest palindrome.

### Detailed Explanation:

1. **Palindrome Definition**: A string is a palindrome if it reads the same forwards and backwards. For example, "racecar" is a palindrome, while "hello" is not.

2. **Constructing the Shortest Palindrome**: The basic strategy is to identify how much of the input string `s` is already a palindrome from its beginning. By determining this, we can figure out the minimum number of characters to add at the front to make the entire string a palindrome.

3. **Using KMP Algorithm**: To solve this efficiently, we can leverage the Knuth-Morris-Pratt (KMP) algorithm's prefix function. This function helps us identify the longest prefix of the concatenation of the string and its reversed version that is also a suffix. This approach is efficient and runs in linear time relative to the size of the string.

4. **Steps**:
   - Reverse the string `s`.
   - Combine the original string `s` and its reversed version using a special character (to avoid false matches).
   - Construct the prefix table using the KMP algorithm.
   - The last value of this table tells us the length of the longest palindromic prefix.
   - The characters that need to be added in front of `s` are simply the suffix of `s` that is not part of the palindrome.

Here is the Python code that implements this approach, structured in a LeetCode-compatible format.



```python
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        if not s:
            return s  # If the input string is empty, return as is.

        # Reverse the string
        rev_s = s[::-1]
        # Create a new string that combines s and rev_s
        combined = s + "#" + rev_s
        
        # Create the KMP table (prefix function)
        n = len(combined)
        lps = [0] * n  # Longest Prefix Suffix
        j = 0  # Length of previous longest prefix suffix

        # Calculate the LPS array
        for i in range(1, n):
            while (j > 0 and combined[i] != combined[j]):
                j = lps[j - 1]
            if combined[i] == combined[j]:
                j += 1
                lps[i] = j
                
        # The length of the longest palindromic prefix found
        longest_palindrome_len = lps[-1]
        
        # The characters we need to add in front of the original string
        to_add = rev_s[0:len(s) - longest_palindrome_len]
        
        # Construct the shortest palindrome
        return to_add + s

# Example use:
# sol = Solution()
# print(sol.shortestPalindrome("aacecaaa"))  # Output: "aaacecaaa"
# print(sol.shortestPalindrome("abcd"))      # Output: "dcbabcd"

```

### Key Points:
- **Efficiency**: The code runs in O(n) time complexity, where n is the length of the input string. This is highly efficient given the problem constraints.
- **Simplicity**: The code is designed to be simple and clear, adhering to the format required by LeetCode. We handle empty strings gracefully as a special case.
- **KMP Table Creation**: Creating the KMP table helps us quickly find the longest palindromic prefix, which is essential for our solution.

This code can be directly copied and run on the LeetCode platform to solve the problem as described.

# 227. Basic Calculator II

### Problem Description 
Given a string `s` which represents an expression, evaluate this expression and return its value. 
The integer division should truncate toward zero.


Example 1:
Input: s = "3+2*2"
Output: 7

Example 2:
Input: s = " 3/2 "
Output: 1

Example 3:
Input: s = " 3+5 / 2 "
Output: 5

Constraints:
`1 <= s.length <= 3 * 105`
`s` consists of integers and operators `('+', '-', '*', '/')` separated by some number of spaces.

`s` represents a valid expression.

All the integers in the expression are non-negative integers in the range `[0, 231 - 1]`.

The answer is guaranteed to fit in a 32-bit integer.

### Solution 
 To solve the problem of evaluating a mathematical expression given as a string, we can follow a systematic approach by using a stack to handle the order of operations (i.e., the usual precedence of multiplication and division over addition and subtraction).

Here's a step-by-step breakdown of how we can implement this:

### Approach:
1. **Initialization**: We'll create a stack to keep track of numbers and handle operators. We'll also maintain a variable to keep the current number being processed, a variable for the last seen operator, and a result variable to store the final result of the evaluation.

2. **Iterate through the string**: As we go through each character of the input string:
   - If the character is a digit, we can build the number (handling multi-digit integers).
   - If the character is an operator ('+', '-', '*', '/'), we will process the last number based on the last seen operator and update the stack accordingly.

3. **Final Calculation**: After iterating through the string, there might be one last number to process, which should be added to the stack. Finally, we sum up all the numbers in the stack to get the final result.

4. **Handling Division**: Since we need to truncate towards zero, special care is taken when dealing with division to ensure that the result is correctly rounded down towards zero.

### Implementation:
Here’s how this can be coded in Python, formatted to run on LeetCode:



```python
class Solution:
    def calculate(self, s: str) -> int:
        # Preprocess the input to remove spaces
        s = s.replace(" ", "")
        stack = []
        current_number = 0
        last_operator = '+'
        
        for i in range(len(s)):
            char = s[i]
            
            # Build the current number
            if char.isdigit():
                current_number = current_number * 10 + int(char)
            
            # If it's an operator or we are at the end of the string, we process the current number
            if char in "+-*/" or i == len(s) - 1:
                if last_operator == '+':
                    stack.append(current_number)
                elif last_operator == '-':
                    stack.append(-current_number)
                elif last_operator == '*':
                    stack[-1] = stack[-1] * current_number
                elif last_operator == '/':
                    # Python's division truncates towards negative infinity, we need to adjust
                    if stack[-1] < 0:
                        stack[-1] = -(-stack[-1] // current_number)
                    else:
                        stack[-1] = stack[-1] // current_number
                
                # Update the last operator and reset the current number
                last_operator = char
                current_number = 0
        
        # The result is the sum of the stack
        return sum(stack)

# Example usage:
# sol = Solution()
# print(sol.calculate("3+2*2")) # Output: 7
# print(sol.calculate(" 3/2 "))  # Output: 1
# print(sol.calculate(" 3+5 / 2 "))  # Output: 5

```

### Explanation:
- **Input preprocessing**: We remove whitespaces to simplify further processing.
- **Building numbers**: When we encounter digits, we build a full number (in case of multi-digit numbers) by multiplying the current number by 10 and adding the new digit. 
- **Operator handling**: Upon encountering an operator (or at the end of the string):
  - Depending on the last operator, we either push the current number to the stack as is, negate it, multiply it with the last number in the stack, or divide it similarly while handling truncation correctly.
- **Final result**: At the end, we return the sum of all numbers in the stack, which represents the evaluated value of the expression.

This code can be run directly on the LeetCode platform, respecting the given constraints and structure of the problem.

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
 To solve the problem of grouping strings that belong to the same shifting sequence, we can utilize the concept of identifying the "shifting key" for each string. The shifting key represents how characters in a string can be transformed into each other by shifting. 

Here is a step-by-step breakdown of how we can approach this solution:

1. **Understanding the Shifting Key**:
   - For a string, we can define a key based on the relative differences between consecutive characters. For example, the string `"abc"` has characters that shift from `a` to `b` (1 shift), and `b` to `c` (another shift of 1). Thus, the key can be represented by the relative shifts between characters, e.g., for `"abc"`, the key would be `(0, 1, 1)` (the differences with respect to the first character).
   - The shifting key can be constructed by taking each character's ASCII value and calculating the difference with the ASCII value of the first character modulo 26.

2. **Grouping Strings**:
   - We can use a dictionary to collect groups of strings that share the same shifting key.
   - For each string in the input list, we compute its shifting key and use this key as a key in our dictionary.

3. **Returning the Result**:
   - Finally, we extract the grouped values from the dictionary and return them as our result.

Now, let’s implement this in Python following the LeetCode solution format:



```python
from collections import defaultdict
from typing import List

class Solution:
    def groupStrings(self, strings: List[str]) -> List[List[str]]:
        def get_key(s: str) -> tuple:
            # Calculate the relative shifts key
            shift = ord(s[0])  # Shift reference based on the first character
            key = []
            for char in s:
                # Calculate the shift value with respect to the first character
                key.append((ord(char) - shift) % 26)
            return tuple(key)  # Use tuple to make it hashable for the dict

        groups = defaultdict(list)
        for s in strings:
            key = get_key(s)  # Get the key for the string
            groups[key].append(s)  # Group the string by its key

        return list(groups.values())  # Return the groups as a list of lists

# Example usage:
# sol = Solution()
# print(sol.groupStrings(["abc","bcd","acef","xyz","az","ba","a","z"]))

```

### Explanation of the Code:

1. **get_key function**:
   - This function takes a string `s` and calculates its shifting key. It starts by determining the ASCII value of the first character as a reference point and then computes the modulo 26 differences for the rest of the characters with respect to this reference.

2. **Main Function groupStrings**:
   - We initialize a `defaultdict` to hold our groups.
   - For each string in the input list, we calculate its shifting key using the `get_key` function and append the original string to the list associated with that key.
   - Finally, we return all the grouped lists using `list(groups.values())`.

### Complexity Analysis:
- The time complexity is O(n * m), where n is the number of strings and m is the average length of the strings. This is because for each string, we need to compute the key which involves iterating through its characters.
- The space complexity is O(n) for storing the grouped strings. 

This implementation correctly groups the strings based on their shifting sequences and can be run directly on LeetCode or any Python environment.

# 271. Encode and Decode Strings

### Problem Description 
Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.

Machine 1 (sender) has the function:
string encode(vector<string> strs) {
  // ... your code
  return encoded_string;
}
Machine 2 (receiver) has the function:
vector<string> decode(string s) {
  //... your code
  return strs;
}
So Machine 1 does:
string encoded_string = encode(strs);
and Machine 2 does:
vector<string> strs2 = decode(encoded_string);
`strs2` in Machine 2 should be the same as `strs` in Machine 1.

Implement the `encode` and `decode` methods.


Example 1:
Input: dummy_input = ["Hello","World"]
Output: ["Hello","World"]
Explanation:
Machine 1:
Codec encoder = new Codec();
String msg = encoder.encode(strs);
Machine 1 ---msg---> Machine 2
Machine 2:
Codec decoder = new Codec();
String[] strs = decoder.decode(msg);

Example 2:
Input: dummy_input = [""]
Output: [""]

Constraints:
`1 <= strs.length <= 200`
`0 <= strs[i].length <= 200`
`strs[i]` contains any possible characters out of `256` valid ASCII characters.

Follow up:
Could you write a generalized algorithm to work on any possible characters?
Could you solve the problem without using any serialize methods (such as `eval`)?

### Solution 
 To solve the problem of encoding and decoding a list of strings, we will implement two functions: `encode` and `decode`. The fundamental idea is to transform a list of strings into a single string that can be safely transmitted and then reconstructed back into the original list.

### Explanation

For encoding, we need to ensure that when we concatenate the strings, we can accurately separate them later during decoding. To achieve this, we'll:

1. Prefix each string with its length followed by a special separator character (which doesn't exist in the text) to mark the end of the length and the beginning of the string.
2. We can use a character like the null character (`\0`) as a separator since it is not used in ASCII strings.

For decoding, we will:

1. Read the entire encoded string and process it by reading the length of each string (until the separator) followed by the actual string of that length.
2. Collect these strings in a list and return it.

### Implementation

Here’s how we can implement this in Python, following the LeetCode format:



```python
class Codec:
    def encode(self, strs: list[str]) -> str:
        # Initialize an empty string to store our encoded strings
        encoded_string = ""
        # Iterate over the list of strings
        for s in strs:
            # Encode the length of the string and the string itself
            # We use \0 as a separator
            encoded_string += f"{len(s)}\0{s}"
        return encoded_string

    def decode(self, s: str) -> list[str]:
        # Initialize an empty list for the decoded strings
        strs = []
        i = 0
        while i < len(s):
            # Find the position of the next separator that indicates the end of the length
            j = s.index('\0', i)
            # Get the length of the string
            length = int(s[i:j])
            # Move the index past the separator and get the actual string
            strs.append(s[j+1:j+1+length])
            # Move the index to the end of the current string for the next iteration
            i = j + 1 + length
        return strs

# Example usage:
# codec = Codec()
# encoded_string = codec.encode(["Hello", "World"])
# decoded_list = codec.decode(encoded_string)
# print(decoded_list)   # Output should be ["Hello", "World"]

```

### Detailed Breakdown:
- **Initialization**: We initialize the encoded string to an empty string for `encode` and an empty list for `decode`.
- **Encoding**: For each string, we compute its length, append it to the `encoded_string`, followed by a separator and then the actual string. This format ensures that we can differentiate between the lengths and the strings when decoding.
- **Decoding**: We loop through the encoded string, first finding the length prefix using the separator, then extracting the string of that given length, and finally appending it to the `strs` list.

### Constraints Handling:
- We consider the constraints of 1 <= `strs.length` <= 200 and 0 <= `strs[i].length` <= 200 easily accommodated, as our approach is linear with respect to the total length of all strings. 
- Using the null character as a separator grants us a safe way to encode any ASCII characters without worrying about collisions.

This solution will correctly encode and decode lists of strings as specified, fulfilling the requirements laid out in the problem.

# 273. Integer to English Words

### Problem Description 
Convert a non-negative integer `num` to its English words representation.


Example 1:
Input: num = 123
Output: "One Hundred Twenty Three"

Example 2:
Input: num = 12345
Output: "Twelve Thousand Three Hundred Forty Five"

Example 3:
Input: num = 1234567
Output: "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"

Example 4:
Input: num = 1234567891
Output: "One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One"

Constraints:
`0 <= num <= 231 - 1`

### Solution 
 To solve the problem of converting a non-negative integer to its English words representation, we will create a Python function that breaks down the number into manageable parts and uses predefined mappings for the English words corresponding to numbers. We'll implement this in a format suitable for LeetCode.

### Breakdown of the Approach:
1. **Handle Special Cases**: We will start by handling the special case of the number zero, which is represented as "Zero".

2. **Define Word Lists**: We will define lists that hold words for numbers from 0 to 19, and then for tens (20, 30, ..., 90). We will also define larger unit words like "Thousand", "Million", and "Billion".

3. **Helper Function for Numbers Less than 1000**: We will create a helper function that converts numbers less than 1000 into words by breaking them down into hundreds, tens, and units.

4. **Main Conversion Logic**: The main function will handle grouping the number into sections of thousands and calling the helper function to assemble the final words.

### Python Code Implementation:
Here’s the complete implementation to directly run on LeetCode:



```python
class Solution:
    def numberToWords(self, num: int) -> str:
        if num == 0:
            return "Zero"
        
        # Define word lists for numbers
        below_20 = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", 
                    "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", 
                    "Seventeen", "Eighteen", "Nineteen"]
        
        tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
        
        thousands = ["", "Thousand", "Million", "Billion"]
        
        def helper(n):
            if n == 0:
                return ""
            elif n < 20:
                return below_20[n] + " "
            elif n < 100:
                return tens[n // 10] + " " + helper(n % 10)
            else:
                return below_20[n // 100] + " Hundred " + helper(n % 100)
        
        res = ""
        for i, word in enumerate(thousands):
            if num % 1000 != 0:
                res = helper(num % 1000) + word + " " + res
            num //= 1000
            
        return res.strip()

# Example Usage:
# sol = Solution()
# print(sol.numberToWords(123))         # Output: "One Hundred Twenty Three"
# print(sol.numberToWords(12345))       # Output: "Twelve Thousand Three Hundred Forty Five"
# print(sol.numberToWords(1234567))     # Output: "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"
# print(sol.numberToWords(1234567891))  # Output: "One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One"

```

### Explanation of the Code:
- **Initialization**: The `numberToWords` method begins by checking if the number is zero. If so, it directly returns "Zero".
  
- **Word Mappings**: The `below_20` and `tens` lists store the word representations of numbers, up to nineteen and multiples of ten respectively. The `thousands` list helps in naming the larger units (thousands, millions, billions).
  
- **Helper Function**: The `helper` function translates numbers less than 1000 into words. It uses recursive calls:
  - For numbers less than 20, it directly fetches the word.
  - For numbers less than 100, it constructs the words for tens and units.
  - For numbers less than 1000, it adds the word for hundreds and then processes the remainder.

- **Assembling the Result**: The main loop iterates over the `thousands` list and collects the words while also reducing the `num` by dividing it by 1000. Finally, it trims any extra spaces and returns the fully assembled string.

This code should work efficiently for the given constraints and convert any valid non-negative integer into its English words representation.

# 293. Flip Game

### Problem Description 
You are playing a Flip Game with your friend.

You are given a string `currentState` that contains only `'+'` and `'-'`. You and your friend take turns to flip two consecutive `"++"` into `"--"`. The game ends when a person can no longer make a move, and therefore the other person will be the winner.

Return all possible states of the string `currentState` after one valid move. You may return the answer in any order. If there is no valid move, return an empty list `[]`.


Example 1:
Input: currentState = "++++"
Output: ["--++","+--+","++--"]

Example 2:
Input: currentState = "+"
Output: []

Constraints:
`1 <= currentState.length <= 500`
`currentState[i]` is either `'+'` or `'-'`.

### Solution 
 To solve the problem of generating all possible states of the string `currentState` after one valid move in the Flip Game, we'll implement a Python function that scans the given string for consecutive `"++"` sequences. Upon finding these sequences, we will flip them to `"--"` and record the new string states.

Here's a detailed explanation of the approach we'll take:

### Steps:

1. **Iteration Over the String**: We'll loop through the string from the beginning to the second-last character (to avoid out-of-bound errors during our check for a sequence of `"++"`).

2. **Check for `"++"`**: Inside the loop, we'll check if the current character and the next character form `"++"`.

3. **Flip the Characters**: If we find a `"++"`, we'll create a new string representing this move by:
   - Keeping the part of the string before the sequence.
   - Replacing the `"++"` with `"--"`.
   - Adding the part of the string after the sequence.

4. **Store Results**: We will collect each new state in a list.

5. **Return the Result**: Finally, we return the list of all valid game states.

### Implementation in LeetCode Format

Here is the complete code that adheres to the problem requirements:



```python
from typing import List

class Solution:
    def generatePossibleNextMoves(self, currentState: str) -> List[str]:
        result = []
        
        # Length of the current state
        n = len(currentState)
        
        # Loop through the string
        for i in range(n - 1):
            # Check for two consecutive '+'
            if currentState[i] == '+' and currentState[i+1] == '+':
                # Create a new state by flipping "++" to "--"
                new_state = currentState[:i] + '--' + currentState[i+2:]
                result.append(new_state)

        return result

```

### Explanation of the Code:

- **Imports**: We import `List` from the `typing` module to specify that our function returns a list.

- **Class Definition**: We define a class `Solution` with the method `generatePossibleNextMoves`.

- **Parameters**: The method takes a string `currentState` as an input and initializes an empty list `result` to store the generated states.

- **Looping through the String**:
  - We use a `for` loop to go through the string from index `0` to `n-2` (`n` being the length of the string) to check for pairs of `"++"`.

- **Checking Conditions**: We check if the current and next character are both `'+'`. If they are, we construct a new string with the flipped characters.

- **Forming New States**: The new string is created by taking all characters before the flip, adding `'--'`, and then concatenating the remainder of the string after the flipped segment.

- **Collecting Results**: Each new state is added to the `result` list.

- **Return Statement**: At the end of the method, we return the `result` list which contains all valid states after one flip.

This solution is efficient with a time complexity of O(n), where n is the length of `currentState`, as we are making a single pass through the string and performing constant time operations to form the new states.

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
 To solve the problem of removing duplicate letters from a string and ensuring that the result is the smallest in lexicographical order, we can make use of a greedy algorithm combined with a stack data structure. The basic idea is to iterate through the characters of the input string while maintaining the characters we want to keep in a stack. Here's how we can achieve this:

### Detailed Explanation:

1. **Counting Frequencies:** 
   - First, we will count the frequency of each character in the input string `s`. This will help us know how many times we need to encounter a character before we can safely remove it from our result.

2. **Using a Stack:**
   - We will use a stack to build the result. The stack allows us to efficiently manage the characters we want to include in the final string, ensuring that we preserve the order of characters while also being able to backtrack if we need to remove a character to maintain lexicographical order.

3. **Tracking In-Stack Characters:**
   - We will maintain a set to track which characters are currently in the stack. This helps us avoid adding duplicate characters to our result.

4. **Iterating through the String:**
   - For each character in `s`, we will do the following:
     - Decrease its frequency count since we are now considering this character.
     - If the character is already in the stack, we skip it.
     - While the stack is not empty and the top character of the stack is greater than the current character, and there are remaining occurrences of the character at the top, we pop the character from the stack. This is crucial because it allows us to ensure that we always have the smallest lexicographical order.
     - After managing the stack, we push the current character onto the stack and mark it as seen.

5. **Building the Result:**
   - Finally, we will join the characters in the stack to form the result string.

### Python Code:

Here's the complete implementation of the above logic in Python, conforming to LeetCode's submission format:



```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        # Step 1: Count the frequency of each character
        last_occurrence = {ch: i for i, ch in enumerate(s)}
        
        stack = []  # Stack to store the result
        in_stack = set()  # Set to keep track of added characters
        
        for i, ch in enumerate(s):
            # Step 2: If the character is already in the stack, skip it
            if ch in in_stack:
                continue
            
            # Step 3: Maintain order in the stack
            while stack and stack[-1] > ch and last_occurrence[stack[-1]] > i:
                removed_char = stack.pop()
                in_stack.remove(removed_char)
            
            # Step 4: Add the current character to the stack
            stack.append(ch)
            in_stack.add(ch)
        
        # Step 5: Convert the stack to string and return
        return ''.join(stack)

# Example Usage:
# sol = Solution()
# print(sol.removeDuplicateLetters("bcabc"))  # Output: "abc"
# print(sol.removeDuplicateLetters("cbacdcbc"))  # Output: "acdb"

```

### Explanation of Example Cases:

1. **Example 1 (`s = "bcabc"`):**
   - The characters will be processed as follows:
     - 'b' -> added to stack (`stack = ['b']`)
     - 'c' -> added to stack (`stack = ['b', 'c']`)
     - 'a' -> will pop 'c' and 'b' (because 'a' is smaller) (`stack = ['a']`)
     - 'b' -> it's already in stack, skip it.
     - 'c' -> added to stack (`stack = ['a', 'b', 'c']`)
   - Final result is "abc".

2. **Example 2 (`s = "cbacdcbc"`):**
   - Process would lead to:
     - 'c' -> added (`stack = ['c']`)
     - 'b' -> added (`stack = ['c', 'b']`)
     - 'a' -> will pop 'b' and 'c' (`stack = ['a']`)
     - 'c' -> it's already in stack, skip it.
     - 'd' -> added (`stack = ['a', 'd']`)
     - Follow the same idea for rest, leading to final result "acdb".

This code should work efficiently within the constraints provided by the problem (up to 10,000 characters).

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
 To solve the problem of finding all distinct pairs of indices `(i, j)` such that the concatenation of two unique words `words[i] + words[j]` forms a palindrome, we can utilize the properties of palindromes and a hash set for efficient lookups.

### Detailed Explanation:

1. **Palindrome Properties**:
   A string is a palindrome if it reads the same forwards and backwards. This means that for a string `s`, `s == s[::-1]`.

2. **Concatenation**:
   We need to consider pairs of words and check if their concatenation forms a palindrome. Specifically for `words[i] + words[j]`, we should be able to verify if the concatenated result is a palindrome efficiently.

3. **Utilizing a Dictionary to Store Words**:
   We'll create a dictionary (hashmap) to store each word along with its index for quick reference. This allows us to check for potential matching palindromic pairs in constant time.

4. **Checking for Palindromic Pairs**:
   For each word, we will iterate through all possible prefixes and suffixes:
   - **Check Suffixes**: For a given word, if we take the prefix and check if the remaining suffix forms a palindrome, we can find suitable candidates for pairs.
   - **Check Prefixes**: Similarly, for the reverse of the word, we also check if any of the remaining prefixes can form a palindrome with another word.

5. **Implementation Steps**:
   - Iterate through each word and its index in the list.
   - For each word, iterate over all possible split points to check both prefixes and suffixes.
   - If a valid palindrome is found, we look it up in our stored dictionary to find its index.

Here’s the solution formatted for LeetCode:



```python
class Solution:
    def is_palindrome(self, s):
        return s == s[::-1]
    
    def palindromePairs(self, words):
        word_map = {word: i for i, word in enumerate(words)}
        result = []
        
        for i, word in enumerate(words):
            word_len = len(word)
            for j in range(word_len + 1):
                prefix = word[:j]
                suffix = word[j:]
                
                # Check if prefix is a palindrome
                if self.is_palindrome(prefix):
                    # Check if the reverse of suffix exists in the map and is not the same word
                    rev_suffix = suffix[::-1]
                    if rev_suffix != word and rev_suffix in word_map:
                        result.append([word_map[rev_suffix], i])
                
                # Check if suffix is a palindrome
                if j != word_len and self.is_palindrome(suffix):  # Avoid double counting
                    rev_prefix = prefix[::-1]
                    if rev_prefix != word and rev_prefix in word_map:
                        result.append([i, word_map[rev_prefix]])
        
        return result

```

### Notes:
- The `is_palindrome` method checks if a given string is a palindrome.
- We maintain a `word_map` dictionary where each word maps to its index.
- We loop through each word, checking all splits to find all valid palindrome pairs.
- Special care is taken to ensure we're not using the same word for both indices by checking for word equality.

This solution efficiently finds all required palindrome pairs and adheres to the constraints of the problem. The complexity is primarily O(n * k^2) where n is the number of words and k is the average length of the words, which is manageable within the given constraints.

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
 To solve the problem of finding the length of the longest substring containing at most `k` distinct characters, we can use the "sliding window" technique combined with a hash map (dictionary in Python). 

### Explanation:

1. **Sliding Window Technique**: 
   - We maintain a window defined by two pointers (left and right) that represents the current substring being evaluated.
   - We expand the right pointer to include characters until we encounter more than `k` distinct characters.
   - Once we have more than `k` distinct characters, we move the left pointer to reduce the size of the window until we are back to `k` or fewer distinct characters.

2. **Hash Map (Dictionary)**:
   - We use a dictionary to keep track of the frequency of each character in the current window.
   - This allows us to quickly determine when we have exceeded the maximum allowed distinct characters and to adjust our window accordingly.

3. **Result Calculation**:
   - Throughout the process, we keep track of the maximum length of valid substrings we encounter.

### Python Code:

Here's how this approach translates into Python code, following the LeetCode format:



```python
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        # Edge case: if k is 0, return 0
        if k == 0:
            return 0
        
        left = 0
        char_count = {}
        max_length = 0
        
        for right in range(len(s)):
            # Add the current character to the dictionary (or increase its count)
            if s[right] in char_count:
                char_count[s[right]] += 1
            else:
                char_count[s[right]] = 1
            
            # While we have more than k distinct characters, move the left pointer
            while len(char_count) > k:
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]
                left += 1
            
            # Calculate the maximum length of valid substring
            max_length = max(max_length, right - left + 1)
        
        return max_length

```

### Detailed Steps:

1. **Initialization**: 
   - Check if `k` is 0 and return 0 early since no substring can be formed.
   - Initialize two pointers: `left` for the start of the substring and iterate `right` through the string.
   - Maintain a dictionary `char_count` to track the count of characters in the current substring and a variable `max_length` to track the longest valid substring found.

2. **Character Addition**:
   - For each character at index `right`, add it to `char_count`.

3. **Adjusting the Left Pointer**:
   - If the number of distinct characters exceeds `k`, increment `left`, decrement the count of the character at `left`, and check if it goes to zero (if so, remove it from `char_count`).

4. **Updating Maximum Length**:
   - After adjusting the left pointer (if necessary), compute the current valid window's length (`right - left + 1`) and update `max_length` accordingly.

5. **Return the Result**:
   - Finally, return the maximum length found.

### Conclusion:
This algorithm efficiently computes the desired result in linear time O(n), ensuring it operates well within the constraints provided. The use of the sliding window technique and hash map allows us to handle the character counts dynamically as we progress through the string.

# 344. Reverse String

### Problem Description 
Write a function that reverses a string. The input string is given as an array of characters `s`.


Example 1:
Input: s = ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]

Example 2:
Input: s = ["H","a","n","n","a","h"]
Output: ["h","a","n","n","a","H"]

Constraints:
`1 <= s.length <= 105`
`s[i]` is a printable ascii character.

Follow up: Do not allocate extra space for another array. You must do this by modifying the input array in-place with `O(1)` extra memory.

### Solution 
 To solve the problem of reversing a string represented as an array of characters in-place, we'll need to approach this with a two-pointer technique. This method allows us to swap characters from the start and end of the array until we meet in the middle.

Here's a detailed breakdown of the approach:

1. **Two Pointers**: We will set one pointer at the beginning of the array (`left`) and another at the end of the array (`right`).

2. **Swapping**: We'll iterate while `left` is less than `right`. In each iteration, we will swap the characters at the `left` and `right` pointers, then move the `left` pointer one step to the right and the `right` pointer one step to the left.

3. **Termination**: The process continues until the `left` pointer is no longer less than the `right` pointer.

This method ensures that we reverse the string in-place, adhering to the constraint of O(1) extra space.

### Python Code Implementation

Here is the complete implementation formatted for LeetCode:



```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left, right = 0, len(s) - 1
        
        while left < right:
            # Swap characters
            s[left], s[right] = s[right], s[left]
            # Move pointers
            left += 1
            right -= 1

```

### Explanation of the Code

- **Function Definition**: The function `reverseString(self, s: List[str]) -> None` takes an array of characters `s` as input and does not return anything, as it modifies the input in place.

- **Pointers Initialization**: We initialize two pointers, `left` at index `0` and `right` at the last index `len(s) - 1`.

- **While Loop**: The loop `while left < right:` continues until the `left` pointer is no longer less than the `right` pointer.
  - **Swapping Characters**: Within the loop, we swap the characters at position `left` and `right`.
  - **Updating the Pointers**: After the swap, we increment `left` and decrement `right` to move towards the center of the array.

### Complexity Analysis

- **Time Complexity**: O(n), where n is the length of the string. This is because we traverse the list once, swapping elements.
  
- **Space Complexity**: O(1), as we are reversing the string in-place and are not using any additional data structures that scale with the input size.

This approach efficiently handles the reversal of the string while meeting the constraints set forth in the problem statement. You can copy and paste this code directly into a LeetCode problem submission.

# 345. Reverse Vowels of a String

### Problem Description 
Given a string `s`, reverse only all the vowels in the string and return it.

The vowels are `'a'`, `'e'`, `'i'`, `'o'`, and `'u'`, and they can appear in both cases.


Example 1:
Input: s = "hello"
Output: "holle"

Example 2:
Input: s = "leetcode"
Output: "leotcede"

Constraints:
`1 <= s.length <= 3 * 105`
`s` consist of printable ASCII characters.

### Solution 
 To solve the problem of reversing only the vowels in a given string, we can follow a systematic approach. We will use a two-pointer technique, which is efficient and suitable for this task. Below is the solution, including a detailed explanation of each part and following the LeetCode solution format.

### Explanation

1. **Identify Vowels**: We will first identify the characters in the string that are considered vowels. For our purposes, these include lowercase and uppercase forms of 'a', 'e', 'i', 'o', 'u'.

2. **Two Pointers**: We'll set up two pointers: one pointing to the start of the string (`left`) and the other pointing to the end of the string (`right`). We will move these pointers towards each other until they meet.

3. **Swap Vowels**: As we traverse the string using both pointers, we will check if the characters at both pointers are vowels. If they are, we will swap them. If not, we will move the pointers inwards: if `left` is not a vowel, we increase `left`, and if `right` is not a vowel, we decrease `right`.

4. **Construct the Result**: Finally, we'll construct the result string and return it.

Here's how the implementation looks:

### Python Code



```python
class Solution:
    def reverseVowels(self, s: str) -> str:
        # Set of vowels for easy checking
        vowels = set('aeiouAEIOU')
        
        # Convert the string to a list for mutability
        s_list = list(s)
        
        # Initialize the two pointers
        left, right = 0, len(s) - 1
        
        while left < right:
            # Move left pointer until a vowel is found or pointers cross
            while left < right and s_list[left] not in vowels:
                left += 1
            # Move right pointer until a vowel is found or pointers cross
            while left < right and s_list[right] not in vowels:
                right -= 1
            
            # If there are vowels at both pointers, swap them
            if left < right:
                s_list[left], s_list[right] = s_list[right], s_list[left]
                # Move both pointers inward
                left += 1
                right -= 1
        
        # Join the list back into a string and return
        return ''.join(s_list)

# Example Usage:
# solution = Solution()
# print(solution.reverseVowels("hello"))  # Output: "holle"
# print(solution.reverseVowels("leetcode"))  # Output: "leotcede"

```

### Key Points

- **Mutable List**: We convert the string `s` into a list `s_list` because strings in Python are immutable. This allows us to swap the characters directly.
- **Vowel Checking**: The set `vowels` allows for O(1) average time complexity when checking if a character is a vowel.
- **In-place Swapping**: By using the two-pointer technique, we effectively reverse the vowels without needing additional space proportional to the input size.

### Complexity Analysis

- **Time Complexity**: O(n), where n is the length of the string `s`. We only traverse the string a couple of times (once for each pointer).
- **Space Complexity**: O(n) for storing the list of characters. 

This code is ready to be run on LeetCode, and it adheres to the provided constraints and requirements.

