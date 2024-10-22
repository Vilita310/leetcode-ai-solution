# 2. Add Two Numbers

### Problem Description 
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.


Example 1:
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.


Example 2:
Input: l1 = [0], l2 = [0]
Output: [0]

Example 3:
Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]

Constraints:
The number of nodes in each linked list is in the range `[1, 100]`.

`0 <= Node.val <= 9`
It is guaranteed that the list represents a number that does not have leading zeros.

### Solution 
 To solve the problem of adding two numbers represented as reversed linked lists, we can approach it as follows:

1. Traverse both linked lists simultaneously, using a "dummy" starting node to construct the result linked list.
2. Use a variable to keep track of any "carry" from the addition of two single digits.
3. For each pair of nodes:
    - Compute the sum of the digits and carry.
    - Update the carry for the next addition (carry will be sum // 10).
    - Append the single digit (sum % 10) to the result linked list.
4. If after finishing iterate through both lists, carry is non-zero, append it as a new node to the result linked list.
5. The dummy node's next node will be the start of our resultant linked list.

Let's look into the Python implementation below on how to achieve this in LeetCode format:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy_head = ListNode(0)
        current_node = dummy_head
        carry = 0

        while l1 is not None or l2 is not None or carry != 0:
            val1 = (l1.val if l1 is not None else 0)
            val2 = (l2.val if l2 is not None else 0)
            sum_val = val1 + val2 + carry
            carry = sum_val // 10
            
            new_node = ListNode(sum_val % 10)
            current_node.next = new_node
            current_node = new_node
            
            if l1 is not None:
                l1 = l1.next
            if l2 is not None:
                l2 = l2.next
        
        return dummy_head.next  # The first node is dummy, start from the second node

# Example use case (outside LeetCode environment for testing)
node1_1 = ListNode(9)
node1_2 = ListNode(9, next=node1_1)
node1_3 = ListNode(9, next=node1_2)
node1_4 = ListNode(9, next=node1_3)
node1_5 = ListNode(9, next=node1_4)
node1_6 = ListNode(9, next=node1_5)
node1_7 = ListNode(9, next=node1_6)

node2_1 = ListNode(9)
node2_2 = ListNode(9, next=node2_1)
node2_3 = ListNode(9, next=node2_2)
node2_4 = ListNode(9, next=node2_3)

sol = Solution()
result = sol.addTwoNumbers(node1_7, node2_4)
lst = []
while result:
    lst.append(result.val)
    result = result.next
print(lst)  # Outputs: [8,9,9,9,0,0,0,1]


```

This code handles potential lists of different lengths by assuming a missed node value to be zero. It maintains and propagates the carry while traversing through the entire length of both linked lists as long as there is another addition to do (either node left or carry). This implementation directly returns the solution in the necessary linked list format expected by LeetCode constraints and setup.

# 7. Reverse Integer

### Problem Description 
Given a signed 32-bit integer `x`, return `x` with its digits reversed. If reversing `x` causes the value to go outside the signed 32-bit integer range `[-231, 231 - 1]`, then return `0`.

Assume the environment does not allow you to store 64-bit integers (signed or unsigned).


Example 1:
Input: x = 123
Output: 321

Example 2:
Input: x = -123
Output: -321

Example 3:
Input: x = 120
Output: 21

Example 4:
Input: x = 0
Output: 0

Constraints:
`-231 <= x <= 231 - 1`

### Solution 
 To solve the problem of reversing the digits of a given integer while adhering to signed 32-bit integer constraints, we need to focus on two main aspects:

1. Reversing the digits of the integer.
2. Ensuring that the reversed integer does not exceed the limits of a signed 32-bit integer. 

Steps to solve the problem:
- Check and handle the negativity of the input integer.
- Use modulo and division operations to extract and construct the reversed integer.
- Use a boundary check to ensure the reversed value doesn't overflow the signed 32-bit integer range.
- Return the reversed integer or zero in case of overflow.

Python provides an ideal environment for this type of digit manipulation due to its strong handling of integers and straightforward arithmetic operations.

Here's the LeetCode solution format for the given problem:



```python
class Solution:
    def reverse(self, x: int) -> int:
        INT_MAX = 2**31 - 1  # 2147483647
        INT_MIN = -2**31     # -2147483648
        
        sign = -1 if x < 0 else 1
        x_abs = abs(x)
        reversed_x = 0
        
        while x_abs:
            pop = x_abs % 10           # Get the last digit
            x_abs = x_abs // 10        # Remove the last digit
            
            # Check for overflow before performing the multiplication and addition
            if reversed_x > (INT_MAX - pop) // 10:
                return 0  # Return 0 if it overflows
            
            reversed_x = reversed_x * 10 + pop
        
        return sign * reversed_x

# Example of how to use the class
# sol = Solution()
# print(sol.reverse(123))  # Output will be 321
# print(sol.reverse(-123)) # Output will be -321
# print(sol.reverse(120))  # Output will be 21
# print(sol.reverse(0))    # Output will be 0

```

**Explanation of the code:**

- **Initialization**: Define constants for INT_MAX and INT_MIN based on the problem constraints. Determine the sign of `x` and work with its absolute value.

- **Digit reversal loop**: Extract the last digit of `x` using modulo operation, shift the result integer `reversed_x` to the left and add the extracted digit. The extraction and combination process repeats until all digits are reversed.

- **Overflow check**: Before potentially overflowing the result with a new digit, check if the upcoming operation would surpass the maximum allowable value for a signed 32-bit integer. If overflow may occur, return 0. The check `(INT_MAX - pop) // 10` assures that when reversed_x is multiplied by 10 and pop is added, it won't exceed INT_MAX.

- **Result**: Return the reversed integer with the original sign unless overflow has occurred, in which case return 0.
  
This method is efficient and comprehensive while ensuring the solution remains within the problem's constraints, specifically the boundaries of a 32-bit signed integer.

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
 Here's a Python solution using the detailed explanation provided. The following implementation follows the steps outlined in the problem statement and adheres to the necessary input constraints. You can copy the entire code segment directly below into the LeetCode editor and hit run to verify its correctness.



```python
class Solution:
    def myAtoi(self, s: str) -> int:
        INT_MAX = 2**31 - 1  # 2147483647
        INT_MIN = -2**31     # -2147483648
        
        i = 0
        n = len(s)
        result = 0
        negative = False
        
        # 1. Read in and ignore any leading whitespace
        while i < n and s[i] == ' ':
            i += 1
        
        # 2. Check if the next character is '-' or '+'
        if i < n and (s[i] == '+' or s[i] == '-'):
            negative = True if s[i] == '-' else False
            i += 1
        
        # 3. Read in next characters until next non-digit or end
        digits = set("0123456789")
        while i < n and s[i] in digits:
            digit = int(s[i])
            # 4. Convert the digits into an integer and handle overflow
            # Check for overflow
            if result > (INT_MAX - digit) / 10:
                return INT_MIN if negative else INT_MAX
            result = result * 10 + digit
            i += 1

        # Apply sign
        if negative:
            result = -result
        
        # 5. Clamp the value if it is out of the 32-bit signed integer range
        if result < INT_MIN:
            return INT_MIN
        if result > INT_MAX:
            return INT_MAX
        
        return result

# Example usage:
sol = Solution()
print(sol.myAtoi("   -42"))  # Output: -42
print(sol.myAtoi("4193 with words"))  # Output: 4193
print(sol.myAtoi("words and 987"))  # Output: 0
print(sol.myAtoi("-91283472332"))  # Output: -2147483648

```

### Explanation:

1. **Ignore Leading Whitespace:** First, we skip over any leading spaces that may exist in the input string, ensuring that we start parsing from the first meaningful character.

2. **Sign detection:** Next, we check for a sign character, either '+' or '-', to determine whether the resultant number should be positive or negative. We also handle the case where no sign is provided.

3. **Parse the Number:** We then scan through subsequent characters until we encounter a character that isn't a digit. During this phase, each digit is added to our result variable, which begins at 0. Any numeric overflow is handled straightforwardly by bounding results to the range specified by INT_MAX and INT_MIN.

4. **Apply Sign and Boundaries:** If the parsed number is intended to be negative based on the initial sign detection, it's negated. Finally, if the resultant number exceeds predefined boundaries of 32-bit signed integers, it's clamped appropriately.

This function will directly work on LeetCode when you submit this as the solution to the problem statement provided.

# 9. Palindrome Number

### Problem Description 
Given an integer `x`, return `true` if `x` is palindrome integer.

An integer is a palindrome when it reads the same backward as forward. For example, `121` is palindrome while `123` is not.


Example 1:
Input: x = 121
Output: true

Example 2:
Input: x = -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.


Example 3:
Input: x = 10
Output: false
Explanation: Reads 01 from right to left. Therefore it is not a palindrome.


Example 4:
Input: x = -101
Output: false

Constraints:
`-231 <= x <= 231 - 1`
Follow up: Could you solve it without converting the integer to a string?

### Solution 
 The task is to determine if a given integer is a palindrome—that is, if it reads the same forwards and backwards. Negative numbers will not be considered palindromic as their negative sign will not appear at the end when reversed.

The solution to this problem can be approached in two ways: 
1. Converting the number to a string and checking if the string reads the same backwards.
2. Solving it mathematically by reconstructing the number in reverse and comparing it with the original number.

Here, I'll present the solution for both approaches, starting with the more intuitive string-based solution and then showing a mathematical solution which avoids converting the integer to a string.

### Solution 1: String Conversion



```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        # Convert the integer to string
        s = str(x)
        # Compare the string with its reverse
        return s == s[::-1]

```

### Explanation:

1. Convert the given integer `x` to a string.
2. Check if the string reads the same backwards using slicing `s[::-1]`.
3. Return `True` if they are equal, otherwise `False`.

### Solution 2: Mathematical Approach (without string conversion)



```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        # Early return if x is negative or ends with 0 (but not equal to 0)
        if x < 0 or (x % 10 == 0 and x != 0):
            return False
        
        reversed_number = 0
        while x > reversed_number:
            digit = x % 10
            reversed_number = reversed_number * 10 + digit
            x //= 10
        
        # Check for even and odd length numbers
        return x == reversed_number or x == reversed_number // 10

```

### Explanation of Mathematical Approach:

1. If the number is negative, or if it ends with a 0 (but is not 0 itself), it cannot be a palindrome.
2. Reconstruct the number in reverse:
   - Extract the last digit using modulo 10 (`digit = x % 10`).
   - Push this digit onto the reversed number (`reversed_number = reversed_number * 10 + digit`).
   - Remove the last digit from the original number using integer division by 10 (`x //= 10`).
3. Eventually, if `x` becomes less than or equal to `reversed_number`, we've processed about half of the digits. We compare either:
   - `x == reversed_number` for even number of digits.
   - `x == reversed_number // 10` for odd number of digits (middle digit should be ignored).
4. Return `True` if they match, indicating the number is a palindrome.

Both solutions are direct, but the second is typically preferred in an interview if string conversion is discouraged. Moreover, the mathematical approach handles edge cases automatically and is memory-efficient.

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
 To solve the problem of converting an integer to a Roman numeral, we can utilize a greedy algorithm. This method involves iterating through each numeral from the largest to the smallest and appending the corresponding Roman symbols to the result string until the number is reduced to zero.

Here is the detailed breakdown of the steps:

1. **Setup a List of Roman Numerals and Their Values**: Create a list that maps Roman numeral symbols to their integer values, including the specific cases for 4, 9, 40 etc., where subtraction is used.

2. **Iterate through the Mapping**: Start from the largest numeral in the map, and repeatedly subtract its value from the integer `num` while appending the corresponding symbol to the output string until subtracting further would result in a negative value. Then, move on to the next numeral.

3. **Append Until Completion**: Continue appending symbols to the result string until `num` is diminished to zero.

Here is the Python code implementing this approach:



```python
class Solution:
    def intToRoman(self, num: int) -> str:
        # List the Roman numerals sorted by their corresponding values, descending.
        # Including composite numerals needed for the special rules (4, 9, 40, 90, etc.)
        roman_numerals = [
            (1000, 'M'), (900, 'CM'), 
            (500, 'D'), (400, 'CD'),
            (100, 'C'), (90, 'XC'),
            (50, 'L'), (40, 'XL'),
            (10, 'X'), (9, 'IX'),
            (5, 'V'), (4, 'IV'),
            (1, 'I')
        ]
        
        # A builder for the resulting Roman numeral string
        result = []
        
        # Process each numeral starting from the largest
        for value, symbol in roman_numerals:
            # Append the symbol as many times as its value can fit into num
            while num >= value:
                result.append(symbol)
                num -= value
        
        # Join all parts into the final string
        return ''.join(result)

# Example Usages:
sol = Solution()
print(sol.intToRoman(3))  # III
print(sol.intToRoman(4))  # IV
print(sol.intToRoman(9))  # IX
print(sol.intToRoman(58))  # LVIII
print(sol.intToRoman(1994))  # MCMXCIV

```

This code leverages an ordered list to handle the Roman numerals, ensuring that larger (or specially combined) values are considered first to reflect the greedy approach. By iteratively subtracting the large denominations from `num` and appending the corresponding symbols, this method efficiently constructs the correct Roman numeral representation. Each step reduces the integer until it reaches zero, confirming that all significant parts have been properly appended as Roman numerals.

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
 Below is a Python solution for converting a Roman numeral to an integer. The solution employs a dictionary to map Roman numeral symbols to their integer values. The main technique involves iterating through the string from left to right, adding values corresponding to each symbol while taking into account the subtraction cases outlined in the problem statement.

Here's the worked out solution:



```python
class Solution:
    def romanToInt(self, s: str) -> int:
        roman_to_int = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        
        total = 0
        i = 0
        while i < len(s):
            if i + 1 < len(s) and roman_to_int[s[i]] < roman_to_int[s[i + 1]]:
                # If current value is less than the next value, this is a subtraction scenario
                total += roman_to_int[s[i + 1]] - roman_to_int[s[i]]
                i += 2  # Move past the current and next character
            else:
                # Normal case, just adding the value of the current Roman numeral
                total += roman_to_int[s[i]]
                i += 1
        
        return total

# Example usage:
solution = Solution()
print(solution.romanToInt("III"))  # Output: 3
print(solution.romanToInt("IV"))   # Output: 4
print(solution.romanToInt("IX"))   # Output: 9
print(solution.romanToInt("LVIII"))  # Output: 58
print(solution.romanToInt("MCMXCIV")) # Output: 1994

```

### Explanation:
1. **Dictionary Mapping:** `roman_to_int` is initialized with the respective values of Roman numerals.
2. **Iteration and Condition Check:** As we iterate through the string `s`, for each character, we check if there is a character following it and whether the current character's value is less than the corresponding next character (which signals a subtraction sequence like IV or IX).
3. **Subtraction and Addition Handling:** If the subtraction condition is met, subtract the current numeral's value from the next numeral's value, and skip to the character after the next. If not, simply add the numeral’s value to the total. 
4. **Index Increment:** Depending on the encountered scenario (subtraction or regular addition), increment the index appropriately (`i += 2` for subtraction, `i += 1` otherwise).

This approach ensures that each numeral and its context (whether it forms part of a subtraction sequence or not) are correctly evaluated, converting the Roman numeral string to its integer equivalent efficiently.

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
 To solve this problem without using multiplication, division, or the modulus operator, we can employ an approach that uses bitwise left shifts to speed up the subtraction process. This is essentially a binary long division method.

Key Concepts:
1. **Bitwise Operations**: Shifting left (<<) effectively multiplies the number by powers of 2. This can be used to perform multiplication operations in the division process using only addition and bit manipulation.
2. **Sign Handling**: The quotient needs to have the correct sign based on the signs of both `dividend` and `divisor`. This can be managed by converting all numbers to positive and adjusting the result's sign at the end.
3. **Checking Overflows**: The solution needs to handle overflow cases where the environment can only store values within a 32-bit signed integer range.

Overall Approach:
1. Determine the sign of the result based on the signs of the `dividend` and `divisor`.
2. Convert both numbers to positive values for ease of calculation.
3. Use subtraction combined with bitwise operations to determine the quotient.
4. Handle the edge cases where the result exceeds the 32-bit signed integer range.

Here's the Python code:



```python
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        # Constants for 32-bit signed integer overflow boundaries
        MAX_INT, MIN_INT = 2**31 - 1, -2**31
        
        # Sign of the result: positive if both have the same sign, negative otherwise
        sign = -1 if (dividend < 0) ^ (divisor < 0) else 1
        
        # Use absolute values to simplify the calculation
        dividend, divisor = abs(dividend), abs(divisor)
        
        # Quotient initialization
        quotient = 0
        
        # Instead of subtracting divisor multiple times, we subtract powers of two multiples of divisor
        # This is effectively the same as multiplying divisor with a power of two but without using *
        power = 31  # Max power of 2 considering 32-bit range
        # 1 << power is divisor times 2^power
        current_divisor = divisor << power
        
        while dividend >= divisor:
            while current_divisor > dividend:
                current_divisor >>= 1
                power -= 1
            
            # Subtract the found fit and add the corresponding number of 'divisor'
            dividend -= current_divisor
            
            # Build the quotient in steps
            quotient += 1 << power
        
        # Adjust result sign
        quotient *= sign
        
        # Handle overflow case
        if quotient < MIN_INT:
            return MIN_INT
        elif quotient > MAX_INT:
            return MAX_INT
        else:
            return quotient

```

This solution works by identifying the largest multiple of `divisor` that fits in `dividend` using bitwise left shifts (essentially, multiplying `divisor` by powers of two without using multiplication). It then subtracts the largest found multiplier from `dividend` and adds corresponding power of two to the `quotient`. This is efficient because it decreases the number of iterations needed when `divident` is much larger than `divisor`. This approach ensures that the solution is both time-efficient and adheres to the constraints (handling overflows and respecting the environment's limitations regarding disallowed operations).

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
 To solve this problem, we'll use a method similar to the traditional approach to multiplication that we perform by hand. This involves multiplying each digit of `num2` with each digit of `num1` and then adding up all the partial products appropriately. Here are the detailed steps:

1. **Initialize**: Create a result array of size `len(num1) + len(num2)` to hold the intermediate sums. Each position in this array represents a digit in the multiplication result before we handle carrying over numbers.

2. **Multiplication**: For each digit in `num2` and `num1`, multiply them. The product influences two positions in the result array. The main reason being: multiplying two numbers `a(i) * b(j)` influences the tens place at `(i + j)` and the units place at `(i + j + 1)`.

3. **Handling Carrying**: After populating the result array, we need to handle the carrying of numbers, if any (similar to what you do in regular multiplication where you carry over digits to the next column on the left).

4. **Build Result**: Convert the array digits into a concatenated string, managing to strip off unnecessary leading zeros (if any exist other than the number '0' itself).

This approach avoids direct number conversion and adheres strictly to string manipulation, treating each digit independently.

Here's the Python code implementing this logic:



```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        if num1 == "0" or num2 == "0":
            return "0"
        
        # Result can be at most len(num1) + len(num2) in length
        result = [0] * (len(num1) + len(num2))
        
        # Reverse num1 and num2 to simplify index calculations
        num1, num2 = num1[::-1], num2[::-1]
        
        # Multiply each digit and add to the result array
        for i in range(len(num1)):
            for j in range(len(num2)):
                # Multiply corresponding digits
                digit_product = int(num1[i]) * int(num2[j])
                # Add to the correct place in the result array
                result[i + j] += digit_product
                # Carry is not handled here directly, it's handled in the next loop
        
        # Handle carrying and convert result to a single number string
        carry = 0
        for i in range(len(result)):
            total = result[i] + carry
            carry = total // 10  # new carry
            result[i] = total % 10
        
        # Remove leading zeros and reverse to get the correct order
        # Convert each digit in result to string and join them
        while result[-1] == 0:
            result.pop()
        
        return ''.join(map(str, result[::-1]))

# Example usage:
sol = Solution()
print(sol.multiply("123", "456"))  # Output: "56088"

```

This code defines a `Solution` class with a `multiply` method as per typical format. It properly initializes a result array, calculates each product, handles carries, and returns the final product as a string. In compliance with the constraints, no direct integer conversion of the entire input strings is performed. This ensures the solution is scalable to handle large numbers represented as strings.

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
 To solve the problem, we can implement the `pow(x, n)` using a method known as "fast exponentiation" or "exponentiation by squaring". It takes advantage of the properties of exponents to minimize the number of multiplicative operations, which makes the computation faster and more efficient.

Key ideas of the method:
1. If `n` is 0, irrespective of `x`, `x^0` is 1.
2. If `n` is positive, recursively square `x` and reduce `n`.
3. If `n` is negative, compute the power for `-n` and then take the reciprocal.

Here's the code adhering to the LeetCode solution format:



```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n < 0:
            x = 1 / x
            n = -n
        
        def fastPow(x, n):
            if n == 0:
                return 1
            half = fastPow(x, n // 2)
            if n % 2 == 0:
                return half * half
            else:
                return half * half * x
        
        return fastPow(x, n)

# Example usage:
sol = Solution()
print(sol.myPow(2.0, 10))  # Output: 1024.0
print(sol.myPow(2.1, 3))   # Output: 9.261000000000001
print(sol.myPow(2.0, -2))  # Output: 0.25

```

### Explanation:

In this solution:
- We first handle the case where `n` is negative by taking the reciprocal of `x` and changing `n` to `-n`.
- We define a helper function, `fastPow`, that:
  - Returns 1 if `n` is 0 because any number raised to the power of zero is 1.
  - Recursively computes `half` which is `x` raised to the power of half of `n`.
  - If `n` is even, it returns `half * half`.
  - If `n` is odd, it returns `half * half * x` because an additional multiplication of `x` is required for the odd power.
- Finally, we call `fastPow` to get the result.

This method takes advantage of the fact that `x^n` can be broken down into smaller powers of `x`, effectively reducing the complexity from O(n) to O(log n) due to the repeated halving of `n`. It's an efficient solution that works for large values of `n` and minimizes the number of multiplications needed.

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
 To solve the problem of finding the k-th permutation sequence of numbers from 1 to n, we can utilize a mathematical approach based on factorials. This eliminates the need to generate all permutations (which would be computationally expensive), and directly computes the desired permutation.

### Key Concepts:
1. **Factorials**: Number of permutations of a set of `n` elements is `n!`.
2. **Sequential building**: We can determine each digit of the result step by step by using the properties of permutations.

### Steps to the Solution:
1. For each position in our result sequence, the possible number of permutations starting with a specific digit can be determined by `(n-1)!` (since there are `n-1` remaining numbers).
2. Use this information sequentially to decide which number should occupy each position:
    - For the first position, there are `(n-1)!` permutations for each start-digit. You can determine the first digit by seeing how many groups of `(n-1)!` can fit into `k`.
    - Reduce `k` by the number of permutations that were skipped.
    - Repeat the process for subsequent digits.

3. **Adjust `k`**: Since lists are 0-indexed and `k` is 1-indexed (1-based), adjust by subtracting 1 initially.

4. **Avoid recomputation**: Pre-calculate the factorial values.

We will utilize an array that holds the digits not yet used (initially [1, 2, ..., n]), and based on the step-by-step decision, pick from this array and remove the chosen digit each time.

### Python Implementation:



```python
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        # Precompute factorials up to (n-1)!
        factorials = [1] * n
        for i in range(2, n):
            factorials[i] = factorials[i-1] * i
        
        # Adjust k to 0-based index
        k -= 1
        
        # Initialize the array of digits
        digits = list(range(1, n + 1))
        
        # Build the k-th permutation using the factorial number system concept
        res = []
        for i in reversed(range(n)):
            # Determine the index of the next number to add
            idx = k // factorials[i]
            k %= factorials[i]
            
            # Add and remove the digit from the list
            res.append(str(digits.pop(idx)))
        
        return ''.join(res)

# Example usage
sol = Solution()
print(sol.getPermutation(3, 3))  # Output: "213"
print(sol.getPermutation(4, 9))  # Output: "2314"
print(sol.getPermutation(3, 1))  # Output: "123"

```

### Explanation:
The solution builds the permutation directly using division and modular arithmetic with precalculated factorials. For each position in our result, it determines which unused digit should appear there by dividing `k` by the appropriate factorial, updates `k`, and selects from the list of available digits accordingly. This approach is efficient and directly constructs the k-th permutation sequence without explicitly generating all permutations. This is especially feasible within the problem's constraints (`1 <= n <= 9`).

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
 To solve the problem of validating a number string based on the provided rules, we need to write a Python function that takes a string and returns a boolean value. We can utilize a finite state machine (FSM) approach. An FSM is perfect for this since it can go through the string character by character and switch its state based on the input conditions.

Here's the step-by-step thought process for building the FSM:
1. Define all possible states based on where in a number we could be, such as starting, handling signs, integer part, decimal point with or without preceding digits, exponents, etc.
2. Create transitions for each state that determine the next state based on character type (digit, sign, dot, or exponent).
3. Transition to an error state if a character doesn't fit any valid transition for the current state.

The states we would consider:
- **start**: the initial state
- **sign**: after encountering a '+' or '-'
- **integer**: digits before any decimal or exponent
- **decimal_point**: just the '.' especially after digits
- **decimal_point_no_int**: just '.' with no preceding digits
- **decimal_frac**: digits after '.'
- **exponent**: 'e' or 'E'
- **exp_sign**: '+' or '-' after 'e' or 'E'
- **exp_number**: digits after 'e'/'E' and potential sign

We'll capture the logic using a dictionary to represent our transition table, and iterate through the string while maintaining the immediate state according to the table's rules.



```python
class Solution:
    def isNumber(self, s: str) -> bool:
        state = "start"
        transitions = {
            "start": {"digit": "integer", "sign": "sign", "dot": "decimal_point_no_int"},
            "sign": {"digit": "integer", "dot": "decimal_point_no_int"},
            "integer": {"digit": "integer", "dot": "decimal_point", "exponent": "exponent"},
            "decimal_point": {"digit": "decimal_frac"},
            "decimal_point_no_int": {"digit": "decimal_frac"},
            "decimal_frac": {"digit": "decimal_frac", "exponent": "exponent"},
            "exponent": {"digit": "exp_number", "sign": "exp_sign"},
            "exp_sign": {"digit": "exp_number"},
            "exp_number": {"digit": "exp_number"}
        }
        
        def get_char_type(c):
            if c.isdigit():
                return "digit"
            elif c in "+-":
                return "sign"
            elif c == ".":
                return "dot"
            elif c in "eE":
                return "exponent"
            return None
        
        for char in s:
            char_type = get_char_type(char)
            if char_type not in transitions[state]:
                return False
            state = transitions[state][char_type]
        
        # Acceptable final states
        return state in ("integer", "decimal_point", "decimal_frac", "exp_number")

# Test cases
sol = Solution()
print(sol.isNumber("0"))  # Output: true
print(sol.isNumber("e"))  # Output: false
print(sol.isNumber("."))  # Output: false
print(sol.isNumber(".1"))  # Output: true

```

In the solution above, the class `Solution` contains a function `isNumber` which verifies if a string is a valid number according to our FSM. This FSM-based approach evaluates transitions and checks valid ending states to decide. This solution conforms well to the requirement of potentially being run on LeetCode's site for testing with Python's interpreter.

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
 To solve this problem on LeetCode, we can simulate the process of binary addition manually, digit by digit, from right to left, similar to how we would do addition on paper.

Here's the breakdown of the approach:

1. **Initialization**: We will use two pointers starting from the end of both strings, and we'll move from right to left. We'll also maintain a `carry` variable to handle the carry-over during addition.

2. **Addition and Carry Handling**:
   - Sum up the digits pointed by the pointers from both strings and add the `carry`.
   - Compute the value of the current bit and the new carry by taking the modulus and division by 2, respectively (`sum % 2` and `sum // 2`).

3. **Result Construction**:
   - Append the resultant bit to the result string.
   - After finishing the loop through both strings, if there's still a carry left, it should be added to the result.

4. **Final Touches**:
   - Since we're adding the resultant bits in reverse order (because we start from the least significant bit), we need to reverse the result string before returning it.

Now, let’s code this logic in a format that’s directly runnable on LeetCode.



```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        # Initialize pointers i and j at the end of each string
        i, j = len(a) - 1, len(b) - 1
        
        # Initialize the variables to store result and carry
        result = []
        carry = 0
        
        # Process each bit from right to left
        while i >= 0 or j >= 0 or carry:
            # Sum up the bits and carry
            sum = carry
            if i >= 0:
                sum += int(a[i])
                i -= 1
            if j >= 0:
                sum += int(b[j])
                j -= 1
            
            # Compute carry (either 1 or 0) and the current bit
            carry = sum // 2
            result.append(sum % 2)
        
        # Since we're appending to result in reverse order we reverse it at the end
        return ''.join(str(x) for x in reversed(result))

# You can test the code by creating an instance of Solution and calling the method addBinary
sol = Solution()
print(sol.addBinary("11", "1"))   # Output should be "100"
print(sol.addBinary("1010", "1011"))  # Output should be "10101"

```

This code provides a linear time solution to the said problem, with efficient utilization of Python's string manipulation capabilities. Each step in the algorithm ensures that even long binary strings (up to 10^4 digits) are handled appropriately without unnecessary overhead. Each digit is processed only once, making the approach efficient.

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
 Here's the Python code solving the problem of finding the integer part of the square root of a given non-negative integer `x`. This solution uses binary search, which is efficient for this problem given the constraints.

The solution code is formatted in a way that you can directly copy and paste it into the LeetCode coding environment and run it.



```python
class Solution:
    def mySqrt(self, x: int) -> int:
        if x < 2:  # Quickly return the solution for x = 0 and x = 1.
            return x
        
        left, right = 1, x // 2  # Reducing the searching space, since square root of x <= x/2 when x >= 4
        while left <= right:
            mid = (left + right) // 2
            square = mid * mid
            if square == x:  # If exact square root is found
                return mid
            elif square < x:  # If square is lesser, discard the left half
                left = mid + 1
            else:  # If square is greater, discard the right half
                right = mid - 1
        return right  # At the end of the loop, right points to the largest integer whose square is <= x.

# Example usage:
sol = Solution()
print(sol.mySqrt(4))  # Output: 2
print(sol.mySqrt(8))  # Output: 2

```

### Explanation:
- We consider this problem as a classic example where binary search can be applied. Instead of searching for an element in an array, here, we search for the square root of `x`.
- **Initial Setup:** If `x` is less than 2, we immediately return `x` because the square root of `0` is `0` and the square root of `1` is `1`.
- We initialize our search space with `left` equals to `1` and `right` equals to `x // 2`. This narrowing of the search space is correct because for `x >= 4`, the square root of `x` cannot be more than `x // 2`. For instance, `2*2 = 4`, `3*3 = 9` (above 4), `4*4 = 16` (above 8), and so on.
- **Binary Search Loop:** We calculate `mid` as the midpoint of `left` and `right`, and then calculate `mid * mid`. We compare this square with `x`:
  - If `square == x`, we found the exact square root.
  - If `square < x`, we discard the left half including `mid` (by setting `left = mid + 1`), since we need a larger number.
  - If `square > x`, we discard the right half including `mid` (by setting `right = mid - 1`), since we need a smaller number.
- **Conclusion:** Once `left` exceeds `right`, `right` will be the largest value whose square is less than or equal to `x`.

This approach ensures logarithmic time complexity, specifically O(log(x)), making it much more efficient than a linear search approach for large inputs.

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
 To solve this problem, we can use a method involving calculating the slopes between each pair of points. The key insight is that points that are on the same straight line share the same slope when taken pairwise. We can use a dictionary (hashmap) to count occurrences of these slopes for each point considered.

### Solution Strategy:

- For each point in `points`, consider it as a potential anchor point and compute the slope it forms with all other points.
- Use a dictionary to count the slopes. The key in the dictionary will be the slope, and the value will be the count of how many times this slope occurs with the chosen anchor.
- The slope can be calculated as `(y2 - y1) / (x2 - x1)` but to avoid issues with floating-point precision and division by zero, we instead represent the slope by a tuple `(dy, dx)` where `dy` is the numerator `(y2 - y1)` and `dx` is the denominator `(x2 - x1)`, normalized to their greatest common divisor (GCD); i.e., `(dy/GCD, dx/GCD)`.
- Special cases to handle:
  - Vertical lines where `dx = 0` which would result in an infinite slope,
  - Overlapping points (although by problem constraints, all points provided are unique).
  
- After iterating through all pairs, the maximum value from our slope dictionary for any point will give the maximum number of points aligned with that anchor including the anchor itself.

### Python Code:



```python
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        from math import gcd
        
        def max_points_from_point(i):
            slopes = {}
            duplicate_points = 0
            max_points = 1  # a point itself
            
            for j in range(len(points)):
                if i != j:
                    dx = points[j][0] - points[i][0]
                    dy = points[j][1] - points[i][1]
                    
                    if dx == 0 and dy == 0:
                        duplicate_points += 1
                    else:
                        # Normalize direction by GCD and handle negative slopes
                        g = gcd(dx, dy)
                        slope = (dy // g, dx // g)
                        if dx < 0:
                            slope = (-slope[0], -slope[1])
                            
                        if slope in slopes:
                            slopes[slope] += 1
                        else:
                            slopes[slope] = 1
                            
                        max_points = max(max_points, slopes[slope] + 1)
            
            max_points += duplicate_points
            
            return max_points
        
        if len(points) <= 2:
            return len(points)
        
        result = 0
        for i in range(len(points)):
            result = max(result, max_points_from_point(i))
        
        return result

```

### Explanation:

- We define an helper function `max_points_from_point` to calculate the maximum aligned points for a given point.
- This function initializes counts for slopes seen with this given anchor.
- We ensure normalization by the GCD of the changes in x and y, preventing errors due to floating-point precision.
- We take care that slopes are uniformly represented irrespective of the order of the pair evaluation.
- Our `maxPoints` method iterates over each point, calls `max_points_from_point`, and keeps track of the global maximum.
- This solution in the worst case goes as `O(n^2)` given the constraints that the number of points can be up to 300, which is manageable. 

This function should run successfully on the LeetCode platform for supplied sample inputs as well as other test cases adhering to the problem's constraints.

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
 To solve the problem of converting a fraction represented by a numerator and a denominator into its string format including handling repeating decimals, we need to carefully manage several aspects:

1. **Sign Handling**: Determine the result sign based on the signs of the numerator and denominator.
2. **Integer Part**: Calculate the integer portion of the result by dividing the absolute values.
3. **Fractional Part**: Handle the remainder after obtaining the integer part.
4. **Detection of Repeating Decimals**: Use a dictionary to track when a particular remainder has been seen before, which helps in identifying the start of a repeating sequence. The key will be the remainder, and the value will be the position in the result string where this remainder was first seen.
5. **Construction of Final Result**: Build the string based on the identified repeating sequence and manage cases without repetition.

Let's implement this strategy in Python:



```python
class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        # Handle zero numerator case
        if numerator == 0:
            return "0"
        
        # Result string and sign handling
        result = []
        if (numerator < 0) ^ (denominator < 0):
            result.append("-")
        
        # Work with the absolute values for division process
        num = abs(numerator)
        den = abs(denominator)
        
        # Append integer part
        result.append(str(num // den))
        remainder = num % den
        
        # If no remainder, just return the integer part
        if remainder == 0:
            return "".join(result)
        
        # Otherwise, handle the fractional part
        result.append(".")
        remainder_map = {}
        
        # Position at which fractional part starts in the result array
        fraction_start_index = len(result)
        
        # Process the decimal portion
        while remainder != 0:
            # Detect repeating remainder
            if remainder in remainder_map:
                # Insert parentheses around the repeating part
                start_index = remainder_map[remainder]
                result.insert(start_index, "(")
                result.append(")")
                break
            
            # Store the position of this remainder
            remainder_map[remainder] = len(result)
            
            # Perform the division step
            remainder *= 10
            result.append(str(remainder // den))
            remainder %= den
        
        # Join all parts of the result list to form the final string
        return "".join(result)

# Test cases
sol = Solution()
print(sol.fractionToDecimal(1, 2))  # Output: "0.5"
print(sol.fractionToDecimal(2, 1))  # Output: "2"
print(sol.fractionToDecimal(2, 3))  # Output: "0.(6)"
print(sol.fractionToDecimal(4, 333))  # Output: "0.(012)"
print(sol.fractionToDecimal(1, 5))  # Output: "0.2"

```

This solution first checks if the numerator is zero, in which case the result is "0". It then adjusts for the result sign given the input signs. The integer portion of the division is handled first followed by the fractional part. If the fractional part begins to repeat, detected by seeing a remainder we've seen before, we insert parentheses at the start and end of the repeating portion. The solution is built step by step in a list, for efficient appending, and joined at the end when returning. This approach efficiently tracks remainders to detect cycles.

# 168. Excel Sheet Column Title

### Problem Description 
Given an integer `columnNumber`, return its corresponding column title as it appears in an Excel sheet.

For example:
A -> 1
B -> 2
C -> 3
...

Z -> 26
AA -> 27
AB -> 28 
...


Example 1:
Input: columnNumber = 1
Output: "A"

Example 2:
Input: columnNumber = 28
Output: "AB"

Example 3:
Input: columnNumber = 701
Output: "ZY"

Example 4:
Input: columnNumber = 2147483647
Output: "FXSHRXW"

Constraints:
`1 <= columnNumber <= 231 - 1`

### Solution 
 To solve this problem, we need to convert a given integer (representing a column number) to its corresponding Excel-style column title. This conversion is similar to changing a number from base-10 to base-26, where the symbols used in base-26 are the alphabets A to Z.

In Excel, the columns are labeled as follows:
- `A` to `Z` for 1 to 26
- `AA` to `ZZ` for 27 to 702
- and so on...

We use modulo and division operations to determine each character of the title from right (least significant) to left (most significant). Each time we determine a character, we capture the remainder of the division of the column number by 26. This is converted to a character between `A` and `Z`:

- If the remainder is 0 (meaning the column number was a multiple of 26), we correspond it to `Z`. In this case, we need to adjust the column number by subtracting 1 to handle cases like `Z` (26), `AZ` (52), etc.
- For non-zero remainders, the letter is simply `remainder - 1` letters after `A`.

After extracting each character, we continuously reduce the column number by dividing it by 26 until it becomes zero, at which point all characters have been determined. To finish, since we've been composing the title from least significant to most significant (from right to left), we need to reverse the string to get the correct order.

The following Python code performs this conversion following the LeetCode solution format:



```python
class Solution:
    def convertToTitle(self, columnNumber: int) -> str:
        result = []
        while columnNumber > 0:
            remainder = (columnNumber - 1) % 26
            result.append(chr(remainder + ord('A')))
            columnNumber = (columnNumber - 1) // 26
        return ''.join(reversed(result))

# Example usages:
sol = Solution()
print(sol.convertToTitle(1))  # Output: "A"
print(sol.convertToTitle(28)) # Output: "AB"
print(sol.convertToTitle(701)) # Output: "ZY"
print(sol.convertToTitle(2147483647)) # Output: "FXSHRXW"

```

Here's a breakdown of what happens when you run the script:
- The `convertToTitle` function is part of the `Solution` class, as per LeetCode format.
- The `result` list accumulates characters representing each positional letter from right to left as per Excel column rules.
- Using `ord('A')` and `chr`, we convert numerical indices to corresponding uppercase letters.
- The test cases directly mimic the provided examples, validating the solution's correctness.

# 171. Excel Sheet Column Number

### Problem Description 
Given a string `columnTitle` that represents the column title as appear in an Excel sheet, return its corresponding column number.

For example:
A -> 1
B -> 2
C -> 3
...

Z -> 26
AA -> 27
AB -> 28 
...


Example 1:
Input: columnTitle = "A"
Output: 1

Example 2:
Input: columnTitle = "AB"
Output: 28

Example 3:
Input: columnTitle = "ZY"
Output: 701

Example 4:
Input: columnTitle = "FXSHRXW"
Output: 2147483647

Constraints:
`1 <= columnTitle.length <= 7`
`columnTitle` consists only of uppercase English letters.

`columnTitle` is in the range `["A", "FXSHRXW"]`.

### Solution 
 The problem described involves converting a string that represents an Excel column title into its corresponding column number. This resembles converting a number from base-26 (letters A-Z) to a decimal system (base-10), where each letter corresponds to a digit in the base-26 number system:

- 'A' corresponds to 1,
- 'B' corresponds to 2,
- ...
- 'Z' corresponds to 26,
- 'AA' corresponds to 27 (similar to how ten follows nine in decimal).

The strategy to solve this involves reading the string from the most significant position to the least significant, i.e., from left to right. Each character will be converted to its respective value ('A' = 1, 'B' = 2, ..., 'Z' = 26). The resulting value is then computed by considering the position of each character as a power of 26, similar to how we compute values in positional numeral systems.

Each step involves:

1. Converting the letter to its numerical equivalent - 'A' = 1, ..., 'Z' = 26.
2. Multiplying this number by \(26^k\), where k is the zero-based position from the right (e.g., for "ZY" with 'Z' = 26 and 'Y' = 25, 'Y' contributes \(25 \times 26^0\) and 'Z' contributes \(26 \times 26^1\)).

Let's implement the solution following the structure of a typical LeetCode Python function:



```python
class Solution:
    def titleToNumber(self, columnTitle: str) -> int:
        number = 0
        power = 0
        n = len(columnTitle)
        for i in range(n - 1, -1, -1):
            digit = ord(columnTitle[i]) - ord('A') + 1
            number += digit * (26 ** power)
            power += 1
        return number

```

Let's explain the code in more detail:

1. `number` represents the resultant column number, initiated to 0.
2. `power` keeps track of the power of 26 that needs to be applied, also initiated to 0.
3. We loop backwards over the string, starting from the last character to the first (`for i in range(n - 1, -1, -1)`).
4. For each character, we compute its relative value to 'A' by `ord(columnTitle[i]) - ord('A') + 1`.
5. We then accumulate this into `number` by multiplying the digit value with `26` raised to the current `power`.
6. After processing each digit, we increment `power` by 1 to account for increasing powers of 26 as we move left in the string.

This algorithm is efficient and works well within the constraints given, using constant space and linear time relative to the length of the input string. The time complexity is O(n), where n is the number of characters in `columnTitle`.

# 172. Factorial Trailing Zeroes

### Problem Description 
Given an integer `n`, return the number of trailing zeroes in `n!`.

Follow up: Could you write a solution that works in logarithmic time complexity?

Example 1:
Input: n = 3
Output: 0
Explanation: 3! = 6, no trailing zero.


Example 2:
Input: n = 5
Output: 1
Explanation: 5! = 120, one trailing zero.


Example 3:
Input: n = 0
Output: 0

Constraints:
`0 <= n <= 104`

### Solution 
 To solve the problem of determining the number of trailing zeroes in \( n! \), we need to understand the origin of trailing zeroes in a factorial. Trailing zeroes are formed by the factors 10. Given \( 10 = 2 \times 5 \), for each pair of factors 2 and 5 in \( n! \), there is one trailing zero.

Since the number of factors of 2 is always greater than the number of factors of 5 in \( n! \) (because every second number is even, providing a factor of 2), we need only to count the number of times 5 is a factor in the numbers from 1 to \( n \).

The approach can be summarized as:
1. Count how often 5 is a factor for numbers between 1 to \( n \). This means counting numbers divisible by \( 5^1 \).
2. Count numbers between 1 to \( n \) that are divisible by \( 5^2 = 25 \), since each such number adds an extra factor of 5.
3. Continue this process, counting numbers divisible by higher powers of 5 (\( 5^3, 5^4, \ldots \)) until \( 5^k > n \).

This approach works because each increment in power corresponds to numbers that contribute additional factors of 5. For example, 25 contributes two 5s, 125 adds three 5s, and so on. So, by summing the number of times each power of 5 can divide numbers up to \( n \), we get the total number of times 5 divides \( n! \), which is the number of trailing zeroes.

Here is a Python code implementing this approach:



```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        count = 0
        # Start with the smallest power of 5, i.e., 5^1 = 5
        power_of_five = 5
        while power_of_five <= n:
            count += n // power_of_five
            power_of_five *= 5
        return count

```

### Explanation:

- The function `trailingZeroes` takes an integer `n` and initializes a `count` to zero.
- It then begins a loop, starting with `power_of_five` equal to 5. On each iteration, it increments `count` by how many times `power_of_five` can divide `n` without a remainder (`n // power_of_five`).
- After each division count, the `power_of_five` is multiplied by 5 to check next power in the next iteration.
- This process continues until the `power_of_five` exceeds `n`.

This solution effectively works in \( O(\log_5 n) \) time complexity as we are only iterating logarithmic times relative to base 5, each time jumping by a factor of 5. Thus, it meets the requirement of the follow-up question for logarithmic time complexity.

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
 To solve this problem, we can use the concept of detecting cycles in the number sequence generated by the algorithm described. A well-known cycle detection algorithm is Floyd's Cycle-Finding Algorithm, which utilizes two pointers (often called the "slow" and "fast" pointers). In the context of this problem:

1. We'll initialize two pointers to start at the number `n`. 
2. The "slow" pointer will move one step at a time (compute the sum of the squares of the digits once).
3. The "fast" pointer will move two steps at a time (compute the sum of the squares of digits twice).
4. If these two pointers meet and the number is not 1, a cycle is detected, and the number is not happy.
5. If we reach the number 1 during the process, the number is happy and we return `true`.

Let's implement this:



```python
class Solution:
    def isHappy(self, n: int) -> bool:
        
        def getNext(number):
            total_sum = 0
            while number > 0:
                digit = number % 10
                total_sum += digit * digit
                number //= 10
            return total_sum

        slow = n
        fast = getNext(n)
        
        while fast != 1 and slow != fast:
            slow = getNext(slow)
            fast = getNext(getNext(fast))
        
        return fast == 1

# Example Usage:
sol = Solution()
print(sol.isHappy(19))  # Output: True
print(sol.isHappy(2))   # Output: False

```

### Detailed Explanation:
Here’s a breakdown of the code:

- **getNext function**: This helper function takes a number and returns the sum of the squares of its digits. This helps us compute the next value in the sequence stemming from any starting number.

- **Two pointers slow and fast**: Both start initially from `n`. `slow` is advanced one step at a time using `getNext()`, and `fast` is advanced two steps at a time (`getNext(getNext())`).

- **Loop**: The loop continues until `fast` equals 1 (indicating it's a happy number) or `slow` meets `fast` (indicating the sequence has entered a cycle). If at any point `fast` equals 1, we break out early, recognizing a happy number.

### Complexity:
- **Time Complexity**: O(log n), because each number in the sequence generally gets smaller drastically faster than linear reduction.
- **Space Complexity**: O(1) as we use a few variables for tracking purposes without extra data structures to store sequence values.

This solution correctly follows the problem constraints and handles the required range of inputs. The cycle detection ensures termination even for numbers that are not happy.

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
 The problem requires counting the total number of prime numbers less than a given number `n`. One of the most efficient ways to find all primes less than a given number is by using the "Sieve of Eratosthenes" algorithm.

### Concept of Sieve of Eratosthenes:
The Sieve of Eratosthenes is an ancient algorithm for finding all prime numbers up to any given limit. It works efficiently by marking the multiples of each prime number starting from 2. The key insight is that once a prime number has been identified, all multiples of that number up to `n` do not need to be checked—they are not prime.

### Steps to Implement:
1. Create a list `is_prime` where each element is initially set to `True`, indicating that each number is presumed to be prime.
2. Set the first two elements (`is_prime[0]` and `is_prime[1]`) to `False`, since 0 and 1 are not prime numbers.
3. For each number `i` from 2 to sqrt(n), if `i` is still marked as prime (`is_prime[i]` is `True`), mark all multiples of `i` starting from `i*i` as not prime (`False`).
4. Count and return the number of `True` values in the `is_prime` list, which represents the prime numbers less than `n`.

### Python Code:
We will implement the above algorithm in the following Python function, adhering to the LeetCode solution format:



```python
class Solution:
    def countPrimes(self, n: int) -> int:
        if n < 2:  # No primes less than 2
            return 0
        
        is_prime = [True] * n   # Step 1
        is_prime[0], is_prime[1] = False, False  # Step 2
        
        # Step 3: Sieve of Eratosthenes
        p = 2
        while p * p < n:
            if is_prime[p]:
                for i in range(p * p, n, p):
                    is_prime[i] = False
            p += 1
        
        # Step 4: Count the True values in is_prime
        return sum(is_prime)

# Example Usage
sol = Solution()
print(sol.countPrimes(10))  # Output: 4

```

### Testing:
To ensure our solution works as expected, we can run it using the example inputs provided, checking against the expected outputs:

- `sol.countPrimes(10)` should return `4`
- `sol.countPrimes(0)` should return `0`
- `sol.countPrimes(1)` should return `0` 

These test cases confirm that the function handles both small inputs efficiently and correctly returns the number of prime numbers where applicable.

# 223. Rectangle Area

### Problem Description 
Given the coordinates of two rectilinear rectangles in a 2D plane, return the total area covered by the two rectangles.

The first rectangle is defined by its bottom-left corner `(A, B)` and its top-right corner `(C, D)`.

The second rectangle is defined by its bottom-left corner `(E, F)` and its top-right corner `(G, H)`.


Example 1:
Input: A = -3, B = 0, C = 3, D = 4, E = 0, F = -1, G = 9, H = 2
Output: 45

Example 2:
Input: A = -2, B = -2, C = 2, D = 2, E = -2, F = -2, G = 2, H = 2
Output: 16

Constraints:
`-104 <= A, B, C, D, E, F, G, H <= 104`

### Solution 
 The task is to calculate the total area covered by two overlapping or non-overlapping rectilinear rectangles. We must account for the possible intersection of the rectangles to avoid over-counting the area where they overlap.

To solve this problem:

1. **Calculate Individual Areas**: First, compute the area of each rectangle separately. The area of a rectangle given by (X1, Y1) and (X2, Y2) is `(X2-X1) * (Y2-Y1)`.

2. **Determine Overlapping Area**:
   - Compute the dimensions of the potential intersection of the two rectangles.
   - The left edge of the intersection would be the maximum of the left edges of the two rectangles, the right edge would be the minimum of their right edges, and similarly for the top and bottom edges.
   - If the calculated edges of the intersection rectangle result in a valid rectangle (where the width and height are positive), then compute its area.

3. **Calculate the Total Coverage Area**:
   - Add the areas of the two rectangles and subtract the overlapping area (if any).

This problem can be tackled directly by calculations and conditionals without any specific data structures or algorithms, ensuring efficient computation. Below is the Python function implementing the solution:



```python
class Solution:
    def computeArea(self, A: int, B: int, C: int, D: int, E: int, F: int, G: int, H: int) -> int:
        # Calculate individual rectangle areas
        area1 = (C - A) * (D - B)
        area2 = (G - E) * (H - F)

        # Calculate overlap
        overlap_width = max(0, min(C, G) - max(A, E))
        overlap_height = max(0, min(D, H) - max(B, F))
        overlap_area = overlap_width * overlap_height

        # Total area covered by the two rectangles
        total_area = area1 + area2 - overlap_area
        return total_area

# Test the function with example cases
solution = Solution()
print(solution.computeArea(-3, 0, 3, 4, 0, -1, 9, 2))  # Expected output: 45
print(solution.computeArea(-2, -2, 2, 2, -2, -2, 2, 2))  # Expected output: 16

```

This Python class `Solution` with a method `computeArea` should directly work in the LeetCode platform, adhering to the typical structure used in LeetCode challenges. Each function call calculates the total area covered by two rectangles with possibly overlapping regions.

# 224. Basic Calculator

### Problem Description 
Given a string `s` representing an expression, implement a basic calculator to evaluate it.


Example 1:
Input: s = "1 + 1"
Output: 2

Example 2:
Input: s = " 2-1 + 2 "
Output: 3

Example 3:
Input: s = "(1+(4+5+2)-3)+(6+8)"
Output: 23

Constraints:
`1 <= s.length <= 3 * 105`
`s` consists of digits, `'+'`, `'-'`, `'('`, `')'`, and `' '`.

`s` represents a valid expression.

### Solution 
 Here is a Python solution for the calculator problem using a stack to handle the operations and the precedence set by the parentheses.



```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        current_number = 0
        current_sign = 1  # 1 for positive, -1 for negative
        result = 0
        
        for char in s:
            if char.isdigit():
                # Build the current number digit-by-digit (could be multiple digits)
                current_number = current_number * 10 + int(char)
            elif char == '+':
                # Evaluate the expression to the left, with current_number and current_sign
                result += current_sign * current_number
                # Update sign for the next number
                current_sign = 1
                # Reset current number
                current_number = 0
            elif char == '-':
                result += current_sign * current_number
                current_sign = -1
                current_number = 0
            elif char == '(':
                # Push the result and the sign onto the stack, for resuming after the parenthesis closes
                stack.append(result)
                stack.append(current_sign)
                
                # Reset the result and sign for the new sub-expression
                result = 0
                current_sign = 1
            elif char == ')':
                # Compute the sub-expression result inside the parentheses
                result += current_sign * current_number
                current_number = 0
                
                # The result is multiplied by the sign before the parenthesis
                result *= stack.pop()  # this is the sign before the '('
                # Then add to the previous result stored in the stack
                result += stack.pop()
        
        # The loop might end without processing the last number
        return result + current_sign * current_number

# Example usage:
sol = Solution()
print(sol.calculate("1 + 1"))  # Output: 2
print(sol.calculate(" 2-1 + 2 "))  # Output: 3
print(sol.calculate("(1+(4+5+2)-3)+(6+8)"))  # Output: 23

```

### Explanation
1. **Loop through Each Character**: We iterate through every character in the input string `s`.
2. **Handle Digits**: If the current character is a digit, we construct the current number.
3. **Handle Operators and Parentheses**:
    - If it's an operator (`+` or `-`), we compute the intermediate result of the expression seen so far using the `current_sign` and `current_number`, then update the `current_sign` based on the operator.
    - If it's an opening parenthesis `(`, we store the current accumulated `result` and the `current_sign` onto a stack and reset them for evaluating the new sub-expression.
    - If it's a closing parenthesis `)`, we finish evaluating the sub-expression (with `current_number` and `current_sign`), then apply the sign before the parenthesis and add the result before this sub-expression.
4. **End of String**: After the loop, we apply the last sign and number to the result.
5. The stack approach allows us properly prioritize operations inside parentheses, handling the nested structure of the expression effectively.

This solution is capable of handling all valid mathematical expressions based on the given constraints, supporting multiple levels of nested expressions.

# 231. Power of Two

### Problem Description 
Given an integer `n`, return `true` if it is a power of two. Otherwise, return `false`.

An integer `n` is a power of two, if there exists an integer `x` such that `n == 2x`.


Example 1:
Input: n = 1
Output: true
Explanation: 20 = 1

Example 2:
Input: n = 16
Output: true
Explanation: 24 = 16

Example 3:
Input: n = 3
Output: false

Example 4:
Input: n = 4
Output: true

Example 5:
Input: n = 5
Output: false

Constraints:
`-231 <= n <= 231 - 1`
Follow up: Could you solve it without loops/recursion?

### Solution 
 To determine if a given integer `n` is a power of two, there are several methods we can use. One of the most efficient ways, especially when constraints disallow the use of loops or recursion, is to use bitwise operations. 

### Explanation:
A power of two in binary representation is characterized by having exactly one bit set (turned to 1). For example:
- 1 (`2^0`) in binary is `0001`
- 2 (`2^1`) in binary is `0010`
- 4 (`2^2`) in binary is `0100`
- 8 (`2^3`) in binary is `1000`
- and so on.

If `n` is a power of two and `n > 0`, then `n & (n - 1)` should equal to `0`. This is based on the observation that subtracting 1 from `n` unsets the rightmost 1-bit and sets all lower bits to 1. When `n` and `(n-1)` are ANDed, the result should be zero since the only 1 bit in `n` would be negated by `(n - 1)`.

### Python Code:
Here is the code you could use, formatted to be directly testable on LeetCode:



```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n <= 0:
            return False
        return (n & (n - 1)) == 0

# Example tests
sol = Solution()
print(sol.isPowerOfTwo(1))   # Output: True
print(sol.isPowerOfTwo(16))  # Output: True
print(sol.isPowerOfTwo(3))   # Output: False
print(sol.isPowerOfTwo(4))   # Output: True
print(sol.isPowerOfTwo(5))   # Output: False

```

### Explanation of the Code:
- The function `isPowerOfTwo` is a member of the class `Solution`.
- It takes an integer `n` and returns a boolean indicating whether `n` is a power of two.
- If `n` is non-positive, it immediately returns `False` because powers of two are strictly positive.
- It then uses the condition `(n & (n - 1)) == 0` to check if `n` is a power of two and returns the result directly.

This solution ensures efficiency with a time complexity of O(1) and a space complexity of O(1), adhering to LeetCode's expected programming practices and providing a solution without the use of loops or recursion.

# 233. Number of Digit One

### Problem Description 
Given an integer `n`, count the total number of digit `1` appearing in all non-negative integers less than or equal to `n`.


Example 1:
Input: n = 13
Output: 6

Example 2:
Input: n = 0
Output: 0

Constraints:
`0 <= n <= 2 * 109`

### Solution 
 The problem is about counting how many times the digit '1' appears from 0 up to a given integer `n`. 

To solve this, we'll analyze each digit place from the units up to the highest place value in `n`. For each digit place, we can partition the numbers into blocks where each block size depends on the current digit's place. We count the total occurrences of '1' contributed by each block. 

To directly implement this in a LeetCode-friendly manner, let's dive into coding the solution:



```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        if n < 1:
            return 0
        count = 0
        factor = 1
        while factor <= n:
            lower_numbers = n - (n // factor) * factor
            current_digit = (n // factor) % 10
            higher_numbers = n // (factor * 10)
            
            # Count the contribution to the count of '1s' by the current digit
            if current_digit == 0:
                count += higher_numbers * factor
            elif current_digit == 1:
                count += higher_numbers * factor + lower_numbers + 1
            else:
                count += (higher_numbers + 1) * factor
            
            factor *= 10
        
        return count

# Use the following to test within LeetCode platform or your local testing.
# test = Solution()
# result = test.countDigitOne(13) # Should return 6
# print(result) # Output should be `6`

```

### Explanation:

1. `factor` represents the current place we are checking, starting with '1', then '10', progressing by multiplying by 10 each loop, thus examining the units, tens, hundreds places, etc.
2. `lower_numbers` is the number of digits that appear to the right of the digit currently being examined.
3. `current_digit` is the actual digit in the place `factor` is representing.
4. `higher_numbers` is the sequence of digits to the left of the current digit.
5. We increment `count` based on what `current_digit` is:

    - If `current_digit` is 0, `1` does not appear in this position in any numbers from the current block of numbers.
    - If `current_digit` is 1, `1` appears in this position from the beginning of this block up to `lower_numbers`.
    - If `current_digit` is greater than 1, `1` appears in this position from the start to the end of this block.

The while loop continues until `factor` surpasses `n`, ensuring each digit position in `n` is checked.

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
 Below is a Python solution to determine if a number represented as a string is strobogrammatic, following the format suitable for submission on LeetCode:



```python
class Solution:
    def isStrobogrammatic(self, num: str) -> bool:
        """
        This function checks whether a given string `num` representing an integer is strobogrammatic.
        A number is strobogrammatic if it looks the same when rotated 180 degrees.
        """
        # Map of valid strobogrammatic number pairs
        strobogrammatic_pairs = {'0': '0', '1': '1', '6': '9', '8': '8', '9': '6'}
        
        left, right = 0, len(num) - 1
        
        while left <= right:
            # Check if the current characters are a valid strobogrammatic pair
            if num[left] not in strobogrammatic_pairs or num[right] != strobogrammatic_pairs[num[left]]:
                return False
            left += 1
            right -= 1
            
        return True

# Example usage:
sol = Solution()
print(sol.isStrobogrammatic("69"))  # Output: True
print(sol.isStrobogrammatic("88"))  # Output: True
print(sol.isStrobogrammatic("962")) # Output: False
print(sol.isStrobogrammatic("1"))   # Output: True

```

### Explanation:
1. **Strobogrammatic Map:** We define a dictionary `strobogrammatic_pairs` which maps each numeral to its strobogrammatic counterpart. The valid pairs based on 180-degree rotation are `{'0': '0', '1': '1', '6': '9', '8': '8', '9': '6'}`.

2. **Two-pointer approach:** We use a two-pointer technique to compare characters from both ends of the string simultaneously moving inward. This approach is efficient since we only traverse half of the string in the worst case.

3. **Validation:** 
   - If a character doesn’t exist in our strobogrammatic map, it means it cannot be part of a strobogrammatic number (i.e., characters other than 0, 1, 6, 8, 9).
   - If the characters at corresponding positions (from the start and end) are not a valid strobogrammatic pair as per the map, we immediately return `False`.

4. **Return `True` If Valid:** If we don’t encounter any disqualifying pairs by the time our pointers meet or cross, we return `True` because the number is strobogrammatic.

This solution ensures optimal performance with a linear time complexity of O(n) where n is the length of the string, and a constant space usage (ignoring the input and output space).

# 247. Strobogrammatic Number II

### Problem Description 
Given an integer `n`, return all the strobogrammatic numbers that are of length `n`. You may return the answer in any order.

A strobogrammatic number is a number that looks the same when rotated `180` degrees (looked at upside down).


Example 1:
Input: n = 2
Output: ["11","69","88","96"]

Example 2:
Input: n = 1
Output: ["0","1","8"]

Constraints:
`1 <= n <= 14`

### Solution 
 To solve the problem of generating all strobogrammatic numbers of a given length `n`, we'll take a recursive approach. Strobogrammatic numbers are numbers that look the same when rotated 180 degrees. The core idea here is to build these numbers from the middle out to the ends, using pairs of characters that are strobogrammatic: 0-0, 1-1, 6-9, 8-8, and 9-6.

The solution involves:
1. Identifying pairs of characters that can be used at the ends of the numbers to maintain the strobogrammatic property.
2. Recursively placing these pairs around a central portion, which can be a strobogrammatic number of a smaller length.

Here's a breakdown of how we handle different scenarios:
- If `n` (the length of the number) is 1, the possible strobogrammatic numbers are simply `["0", "1", "8"]`.
- Similar initial list for `n` = 0 is `[""]` (one empty string representing the base case).
- For larger values, we recursively build the numbers for `n - 2`, insert each pair around them, and then append these new numbers to the result.

This approach will let us tackle the problem size incrementally, building complex solutions from simpler ones.

We'll implement this recursive approach in Python and ensure it fits within the coding styles and requirements for direct execution on LeetCode.

Here is the complete solution:



```python
class Solution:
    def findStrobogrammatic(self, n: int) -> List[str]:
        return self.helper(n, n)
    
    def helper(self, n: int, m: int) -> List[str]:
        if n == 0: return [""]
        if n == 1: return ["0", "1", "8"]
        
        prev = self.helper(n - 2, m)
        
        result = []
        for num in prev:
            if n != m:
                result.append('0' + num + '0')
            result.append('1' + num + '1')
            result.append('8' + num + '8')
            result.append('6' + num + '9')
            result.append('9' + num + '6')
        
        return result

```

**Explanation of the Code:**
- `helper` is a recursive function that generates strobogrammatic numbers of length `n`. The parameter `m` refers to the original `n` to check if the outermost pair can use '0' (e.g., '010' is not valid except for the middle of numbers like '1010').
- The base cases return lists for n = 0 and n = 1.
- The function then recursively gets the strobogrammatic numbers for `n-2` and builds numbers of length `n` by wrapping the results with each possible pair.
- It's important to avoid adding numbers that start with a '0', except for the case where the entire number is '0'. This is managed by the condition `if n != m`.

This solution should work efficiently for n up to 14 due to the inherent nature of the problem constraining exponential growth of results.

# 248. Strobogrammatic Number III

### Problem Description 
Given two strings low and high that represent two integers `low` and `high` where `low <= high`, return the number of strobogrammatic numbers in the range `[low, high]`.

A strobogrammatic number is a number that looks the same when rotated `180` degrees (looked at upside down).


Example 1:
Input: low = "50", high = "100"
Output: 3

Example 2:
Input: low = "0", high = "0"
Output: 1

Constraints:
`1 <= low.length, high.length <= 15`
`low` and `high` consist of only digits.

`low <= high`
`low` and `high` do not contain any leading zeros except for zero itself.

### Solution 
 

```python
class Solution:
    def strobogrammaticInRange(self, low: str, high: str) -> int:
        # Character map for strobogrammatic pairs: 
        # 0 -> 0, 1 -> 1, 6 -> 9, 8 -> 8, 9 -> 6
        strobo_pairs = [('0', '0'), ('1', '1'), ('6', '9'), ('8', '8'), ('9', '6')]
        
        # Helper to recursively generate strobogrammatic numbers of a specific length
        def generate_numbers(n, length):
            if n == 0: return [""]
            if n == 1: return ["0", "1", "8"]
            
            middles = generate_numbers(n - 2, length)
            output = []
            
            for middle in middles:
                for first, last in strobo_pairs:
                    # Skip adding numbers with leading zeroes unless the length is 1
                    if first == '0' and n == length and length > 1:
                        continue
                    output.append(first + middle + last)
            return output
        
        # Generates all strobogrammatic numbers in the range of lengths
        def generate_all(low, high):
            l, h = len(low), len(high)
            result = []
            for length in range(l, h + 1):
                result.extend(generate_numbers(length, length))
            return result

        # Getting all the strobogrammatic numbers from length of low to length of high
        all_numbers = generate_all(low, high)
        count = 0
        
        # Filter numbers to check that they are in the actual numeric range [low, high]
        for number in all_numbers:
            if int(low) <= int(number) <= int(high):
                count += 1
        
        return count

# Example cases
sol = Solution()
print(sol.strobogrammaticInRange("50", "100"))  # Output: 3
print(sol.strobogrammaticInRange("0", "0"))    # Output: 1

```

### Explanation

**Algorithm Overview:**

1. **Character Mapping:**
   Define the pairs of numbers that remain the same or map to each other when the number is inverted. These pairs are (0, 0), (1, 1), (8, 8), (6, 9), and (9, 6).

2. **Recursive Generation:**
   - `generate_numbers(n, length)`: Recursively generate strobogrammatic numbers of length `n`.
   - At each stage, use the `middles` from `n-2` and add each of the strobo pairs to the sides.
   - Watch for leading zeros: only allow zeros as middle part if the entire length is 1 to prevent invalid numbers from being generated.

3. **Range Handling:**
   Split the procedure based on possible lengths of the numbers, from `len(low)` to `len(high)` (inclusive).

4. **Filtering Valid Numbers:**
   Post generation, filter all generated numbers to check if they fall within the numeric range defined by `low` and `high`.

5. **Count Valid Numbers:**
   Count the valid strobogrammatic numbers falling within the specified range and return the count.

The code has been tested for basic examples, and it adheres to the constraints. The logic efficiently eliminates unnecessary checks and closely observes requirements of valid strobogrammatic number conditions to optimize the generation process.

# 258. Add Digits

### Problem Description 
Given an integer `num`, repeatedly add all its digits until the result has only one digit, and return it.


Example 1:
Input: num = 38
Output: 2
Explanation: The process is
38 --> 3 + 8 --> 11
11 --> 1 + 1 --> 2 
Since 2 has only one digit, return it.


Example 2:
Input: num = 0
Output: 0

Constraints:
`0 <= num <= 231 - 1`
Follow up: Could you do it without any loop/recursion in `O(1)` runtime?

### Solution 
 Sure, here's how to solve the problem of repeatedly adding digits until a single-digit result is derived, along with the implementation in Python:

### Explanation

The task can be solved by simulating the repeated addition of digits until a single digit is obtained. However, there's an optimized approach derived from number theory called "digital root".

**Digital Root**: 
The digital root of a non-negative integer is the single-digit value obtained by an iterative process of summing digits, on each iteration using the result from the previous iteration to compute a digit sum. The process continues until a single-digit number is reached.

This process has a direct relation with modulo 9 operation:
1. If a number is divisible by 9, its digital root is 9.
2. If a number is not divisible by 9, its digital root is the remainder of the division of the number by 9.

Given the problem's constraints (`0 <= num <= 231 - 1`) and the request for `O(1)` solution, implementing the digital root concept using modulo operations directly leads us to the optimal solution.

### Python Code



```python
class Solution:
    def addDigits(self, num: int) -> int:
        if num == 0:
            return 0
        elif num % 9 == 0:
            return 9
        else:
            return num % 9

# The code can be run on LeetCode and should be directly executable there.

```

### How the solution works:
1. If the number is zero, its digital root is immediately zero.
2. If the number modulo 9 is zero (and the number is not zero), the digital root is 9.
3. Otherwise, the digital root of the number is the remainder of the number divided by 9.

### Testing on Example Cases:
- For `num = 38`, the output should be `2` as demonstrated in the problem's example.
  The calculation using our modulo based formula yields `38 % 9 = 2`.
- For `num = 0`, the output is `0`.

This solution adheres to the O(1) time complexity requirement as it performs a constant number of operations regardless of the size of the input.

# 263. Ugly Number

### Problem Description 
Given an integer `n`, return `true` if `n` is an ugly number.

Ugly number is a positive number whose prime factors only include `2`, `3`, and/or `5`.


Example 1:
Input: n = 6
Output: true
Explanation: 6 = 2 × 3

Example 2:
Input: n = 8
Output: true
Explanation: 8 = 2 × 2 × 2

Example 3:
Input: n = 14
Output: false
Explanation: 14 is not ugly since it includes another prime factor 7.


Example 4:
Input: n = 1
Output: true
Explanation: 1 is typically treated as an ugly number.


Constraints:
`-231 <= n <= 231 - 1`

### Solution 
 To solve the problem of determining if a number, `n`, is an "ugly" number, we need to check if its only prime factors are 2, 3, or 5. The approach involves continuously dividing `n` by these primes if they are factors until `n` is reduced to 1. If during this process we find a factor other than 2, 3, or 5, or if the number is initially non-positive, then `n` is not an ugly number.

Here's a detailed solution in Python, suitable to run directly on LeetCode:



```python
class Solution:
    def isUgly(self, n: int) -> bool:
        if n <= 0:  # Ugly numbers must be positive
            return False
        # Continuously divide n by 2, 3, or 5 if they are factors
        for prime in [2, 3, 5]:
            while n % prime == 0:
                n //= prime
        # If n has been reduced to 1, then it had only 2, 3, or 5 as its prime factors
        return n == 1

# The class can be tested with the examples given
# Example uses:
sol = Solution()
print(sol.isUgly(6))   # Output: True
print(sol.isUgly(8))   # Output: True
print(sol.isUgly(14))  # Output: False
print(sol.isUgly(1))   # Output: True

```

### Explanation of the `isUgly` Method

1. **Initial Check (Line 4)**: First, we check if `n` is less than or equal to zero. Since ugly numbers are defined only for positive integers, any non-positive integer is immediately not an ugly number.

2. **Division Process (Lines 6-8)**: For each prime factor (2, 3, and 5), we divide `n` by the prime factor as long as `n` is perfectly divisible by that factor (`n % prime == 0`). This loop will strip away the prime factors 2, 3, or 5 from `n`.

3. **Final Condition (Line 10)**: After all possible divisions by 2, 3, and 5, if `n` has been reduced to 1, it means that its original prime factors only included 2, 3, and 5, and therefore, it is an ugly number. If `n` is not 1, it had other prime factors, so it is not an ugly number.

The solution efficiently reduces `n` and checks its eligibility by direct iteration of the defined primes and is O(log(n)) due to the repeated division. This algorithm directly conforms to the typical solution format for competitive programming, aligning well particularly with platforms like LeetCode.

# 264. Ugly Number II

### Problem Description 
Given an integer `n`, return the `nth` ugly number.

Ugly number is a positive number whose prime factors only include `2`, `3`, and/or `5`.


Example 1:
Input: n = 10
Output: 12
Explanation: [1, 2, 3, 4, 5, 6, 8, 9, 10, 12] is the sequence of the first 10 ugly numbers.


Example 2:
Input: n = 1
Output: 1
Explanation: 1 is typically treated as an ugly number.


Constraints:
`1 <= n <= 1690`

### Solution 
 Below is the Python solution to the provided problem, of finding the nth ugly number. The algorithm employs a dynamic programming approach with a minimum heap to efficiently generate the sequence of ugly numbers.



```python
from heapq import heappop, heappush

class Solution:
    def nthUglyNumber(self, n: int) -> int:
        if n == 1:
            return 1
        
        # Minimum heap to track the next minimum ugly number
        heap = []
        heappush(heap, 1)
        
        # Seen set to prevent duplicates in the heap
        seen = {1}
        
        # Factors that can generate new ugly numbers
        factors = [2, 3, 5]
        
        # Remove elements from the heap and generate new numbers until we reach nth ugly number
        for _ in range(n):
            # Get the smallest number available which is an ugly number
            ugly = heappop(heap)
            
            # Generate new numbers by multiplying with each factor
            for factor in factors:
                new_ugly = ugly * factor
                if new_ugly not in seen:
                    seen.add(new_ugly)
                    heappush(heap, new_ugly)
        
        # The last popped item from the heap is our nth ugly number
        return ugly

# Example usage:
sol = Solution()
print(sol.nthUglyNumber(10))  # Output: 12
print(sol.nthUglyNumber(1))   # Output: 1

```

### Explanation:

**Algorithm Approach:**
- **1. Initialization:** Start by initializing a min-heap with the first ugly number, `1`. Also, maintain a set called `seen` to track numbers that have already been added to the heap to avoid duplicates.
  
- **2. Pop and Generate:** Use a for-loop iterating `n` times. In each iteration:
  - Get the smallest element from the heap (This element is guaranteed to be the next ugly number due to the heap property).
  - Multiply this ugly number by each factor (2, 3, and 5) to generate potential new ugly numbers.
  - If these newly formed numbers aren’t in the `seen` set, add them to both the set and the heap.
  
- **3. Return the Result:** After exiting the loop, the last element popped from the heap represents the nth ugly number.

**Complexity Analysis:**
- **Time Complexity:** Each insertion and extraction operation on the heap takes O(log k) time. Since we're dealing with `n` elements, overall complexity becomes O(n log n).
- **Space Complexity:** The space complexity is also O(n) due to storing up to `n` ugly numbers in the heap and the set.

This method is efficient due to the heap maintaining the numbers in sorted order, allowing access to the smallest ugly number in constant time, and generating new numbers efficiently by multiplying with predefined factors and controlling duplicates using a set.

# 268. Missing Number

### Problem Description 
Given an array `nums` containing `n` distinct numbers in the range `[0, n]`, return the only number in the range that is missing from the array.

Follow up: Could you implement a solution using only `O(1)` extra space complexity and `O(n)` runtime complexity?

Example 1:
Input: nums = [3,0,1]
Output: 2
Explanation: n = 3 since there are 3 numbers, so all numbers are in the range [0,3]. 2 is the missing number in the range since it does not appear in nums.


Example 2:
Input: nums = [0,1]
Output: 2
Explanation: n = 2 since there are 2 numbers, so all numbers are in the range [0,2]. 2 is the missing number in the range since it does not appear in nums.


Example 3:
Input: nums = [9,6,4,2,3,5,7,0,1]
Output: 8
Explanation: n = 9 since there are 9 numbers, so all numbers are in the range [0,9]. 8 is the missing number in the range since it does not appear in nums.


Example 4:
Input: nums = [0]
Output: 1
Explanation: n = 1 since there is 1 number, so all numbers are in the range [0,1]. 1 is the missing number in the range since it does not appear in nums.


Constraints:
`n == nums.length`
`1 <= n <= 104`
`0 <= nums[i] <= n`
All the numbers of `nums` are unique.

### Solution 
 Below is a Python solution using a mathematical approach to find the missing number in the given array `nums`. This approach adheres to the constraints of having `O(1)` extra space complexity and `O(n)` runtime complexity.

The strategy is to calculate the expected sum of the first `n` natural numbers using the formula: \( \text{sum} = \frac{n \times (n + 1)}{2} \). Then, subtract the actual sum of the elements in the array from this expected sum to find the missing number.

Here's a Python code in line with the usual format for solutions provided on LeetCode:



```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        expected_sum = n * (n + 1) // 2
        actual_sum = sum(nums)
        return expected_sum - actual_sum

# This code defines a class "Solution" with the function "missingNumber" that computes the missing number based on the difference between the expected and actual sum of numbers.

```

### How It Works:
1. **Calculate Expected Sum**: Compute \( \frac{n \times (n + 1)}{2} \) where \( n \) is the length of the array `nums`. This gives us the sum of all numbers from `0` to `n` inclusive.
2. **Compute Actual Sum**: Sum up all the elements currently in the `nums` array.
3. **Compute Difference**: The difference between the expected sum and the actual sum will give the missing number because every other number cancel themselves out except the missing number.

### Example Walkthrough:
For the array `nums = [3, 0, 1]`:
- \( n = 3 \) (since there are three elements, meaning numbers will range from 0 to 3).
- Expected sum \( = \frac{3 \times (3 + 1)}{2} = 6 \)
- Actual sum \( = 3 + 0 + 1 = 4 \)
- Missing number \( = 6 - 4 = 2 \)

This approach effectively finds the missing number by utilizing summation properties and fits the constraints perfectly as it only uses constant extra space.

### Note:
- The function signature given above (`def missingNumber(self, nums: List[int]) -> int:`) needs to include a type hint import for the type `List`. Hence, ideally a `from typing import List` would be declared at the top of a script (but in LeetCode's environment, it's usually pre-imported).
- The solution is designed for direct application in Python environments on coding challenge platforms like LeetCode, which provide the requisite environment setup, including necessary imports and handling of standard input and output.

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
 To convert a non-negative integer to its English words representation, we need to break down the problem into handling specific ranges of numbers: units, tens, hundreds, thousands, millions, and billions. For each section, there may be specific rules for forming the English representation. Below is a Python solution that accurately maps integers to their corresponding English words.



```python
class Solution:
    def numberToWords(self, num: int) -> str:
        if num == 0:
            return "Zero"
        
        # These are mappings from numbers to words.
        below_20 = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
        tens = ["", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
        
        # Chunks of thousand
        thousands = ["", "Thousand", "Million", "Billion"]
        
        def word(num):
            if num < 20:
                return below_20[num]
            elif num < 100:
                return tens[num // 10] + (" " + below_20[num % 10] if num % 10 != 0 else "")
            else:
                return below_20[num // 100] + " Hundred" + (" " + word(num % 100) if num % 100 != 0 else "")
        
        def words(num):
            if num == 0:
                return ""
            elif num < 1000:
                return word(num).strip()
            for p, w in enumerate(thousands):
                if num < 1000 ** (p + 1):
                    return words(num // (1000 ** p)) + " " + w + (" " + words(num % (1000 ** p)) if num % (1000 ** p) != 0 else "").strip()
        
        return words(num).strip()

# Example usage:
sol = Solution()
print(sol.numberToWords(123))         # Output: "One Hundred Twenty Three"
print(sol.numberToWords(12345))       # Output: "Twelve Thousand Three Hundred Forty Five"
print(sol.numberToWords(1234567))     # Output: "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"
print(sol.numberToWords(1234567891))  # Output: "One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One"

```

### Explanation:

1. **Base Cases**: Handling base case where 0 gives "Zero".
2. **Words Helper Functions**:
   - `word(num)`: Converts numbers less than 1000 into words using mappings for digits and tens.
   - `words(num)`: Uses the `word` function to recursively express thousands, millions, and billions.
3. **Mapping Arrays**:
   - `below_20` for numbers from 0 to 19.
   - `tens` for multiples of ten.
   - `thousands` for units of large numbers (thousand, million, billion).
4. **Approach**: Decompose the number into chunks of thousands (10^3), millions (10^6), and billions (10^9), using the appropriately mapped words in English.
5. **Execution**: Iterate progressively by dividing the number to reduce it using the base exponential factors (thousands, millions, etc.).

This structured approach ensures that every number within the provided constraint fits into an appropriate verbal grouping for its size segment.

# 279. Perfect Squares

### Problem Description 
Given an integer `n`, return the least number of perfect square numbers that sum to `n`.

A perfect square is an integer that is the square of an integer; in other words, it is the product of some integer with itself. For example, `1`, `4`, `9`, and `16` are perfect squares while `3` and `11` are not.


Example 1:
Input: n = 12
Output: 3
Explanation: 12 = 4 + 4 + 4.


Example 2:
Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.


Constraints:
`1 <= n <= 104`

### Solution 
 To solve the given problem, we will utilize Dynamic Programming (DP) to build up a solution that determines the least number of perfect square numbers that sum to `n`.

Approach:
- We will create a list, `dp`, where `dp[i]` will represent the minimum number of perfect squares that sum up to the integer `i`. 
- Starting with `dp[0] = 0` (because 0 can be represented as the sum of zero perfect squares), fill up the list for all values up to `n`.
- For each integer `i` from 1 to `n`, we'll determine the minimum number of perfect squares needed by trying every square number less than or equal to `i` (from `1^2` to the greatest square `j^2` ≤ `i`). The idea is to minimize `dp[i]` as `dp[i] = min(dp[i], dp[i-j*j] + 1)` for all possible `j`.

The formula `dp[i] = min(dp[i], dp[i-j*j] + 1)` derives from subtracting a square number `j*j` from `i` (which has known result `dp[i-j*j]`) and adding one more square number (`j*j` itself).

Here's the implementation of the solution:



```python
class Solution:
    def numSquares(self, n: int) -> int:
        # Initialize the dp array where the size is n+1, with infinity
        # since we are looking for the minimum values.
        dp = [float('inf')] * (n + 1)
        
        # Base case: no squares are needed to sum to 0.
        dp[0] = 0
        
        # Fill dp array
        for i in range(1, n + 1):
            # Check squares j*j where j*j <= i
            j = 1
            while j * j <= i:
                dp[i] = min(dp[i], dp[i - j * j] + 1)
                j += 1
        
        # The answer for dp[n] will be the minimum number of perfect squares that sum to n.
        return dp[n]

# Example usage:
# sol = Solution()
# print(sol.numSquares(12))  # Output: 3
# print(sol.numSquares(13))  # Output: 2

```

This code should directly run on the LeetCode platform and is designed to handle the constraint efficiently with `1 <= n <= 10^4`. The main part of the algorithm operates in a complexity of roughly \(O(n \times \sqrt{n})\), which is suitable for the given constraint.

# 296. Best Meeting Point

### Problem Description 
Given an `m x n` binary grid `grid` where each `1` marks the home of one friend, return the minimal total travel distance.

The total travel distance is the sum of the distances between the houses of the friends and the meeting point.

The distance is calculated using Manhattan Distance, where `distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|`.


Example 1:
Input: grid = [[1,0,0,0,1],[0,0,0,0,0],[0,0,1,0,0]]
Output: 6
Explanation: Given three friends living at (0,0), (0,4), and (2,2).

The point (0,2) is an ideal meeting point, as the total travel distance of 2 + 2 + 2 = 6 is minimal.

So return 6.


Example 2:
Input: grid = [[1,1]]
Output: 1

Constraints:
`m == grid.length`
`n == grid[i].length`
`1 <= m, n <= 200`
`grid[i][j]` is either `0` or `1`.

There will be at least two friends in the `grid`.

### Solution 
 To solve this problem, the key is to understand that the median minimizes the sum of absolute deviations. Thus, to find the minimal total travel distance, we should aim to determine the median coordinates (median in x and y separately) among all the homes, which will be our meeting point.

### Steps:
1. Extract all the x-coordinates and y-coordinates where friends' homes (`grid[i][j] = 1`) are located.
2. Sort these lists of x and y coordinates.
3. Identify the median of these sorted lists. For lists of odd length, the median is the middle element. For even length, any index between the two central elements will work equally well for the purposes of minimizing the sum of distances in Manhattan distance.
4. Calculate the total Manhattan distance from each friend's home to the median coordinate.

Given this approach, let's implement the solution:



```python
class Solution:
    def minTotalDistance(self, grid: List[List[int]]) -> int:
        rows, cols = [], []
        
        # Collect home coordinates
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 1:
                    rows.append(r)
                    cols.append(c)
        
        # Sort rows and columns list
        rows.sort()
        cols.sort()
        
        # Find the median position
        mid_row = rows[len(rows) // 2]
        mid_col = cols[len(cols) // 2]
        
        # Calculate the minimum distance to the median point
        min_distance = 0
        for r in rows:
            min_distance += abs(r - mid_row)
        for c in cols:
            min_distance += abs(c - mid_col)
        
        return min_distance

# Note: In the implementation, extracting rows and cols separately is O(m*n), sorting them is O(k log k),
# and then summing distances is O(k), where k is the number of friends. Since m, n <= 200 and k can be at most m*n,
# this solution is efficient within provided constraints.

```

### Explanation:
1. **Rows and Columns Gathering**: Iterate through each cell in the grid. If the cell value is `1`, add its row index to the `rows` list and the column index to the `cols` list. The presence of a value `1` indicates a friend's home.
2. **Sorting**: We sort the `rows` and `cols` to easily find the median row and column indices. The sorting step makes median detection straightforward.
3. **Finding Median**: We find the median of the `rows` and `cols` arrays. In a sorted list, the median is represented by its middle element(s).
4. **Calculate Distance**: Sum up the absolute differences of each coordinate from the median position. This is done separately for x and y coordinates (rows and columns respectively).

The code can be directly pasted into a Python script environment in LeetCode to verify it matches their expected function signature and requirements.

# 313. Super Ugly Number

### Problem Description 
Given an integer `n` and an array of integers `primes`, return the `nth` super ugly number.

Super ugly number is a positive number whose all prime factors are in the array `primes`.

The `nth` super ugly number is guaranteed to fit in a 32-bit signed integer.


Example 1:
Input: n = 12, primes = [2,7,13,19]
Output: 32
Explanation: [1,2,4,7,8,13,14,16,19,26,28,32] is the sequence of the first 12 super ugly numbers given primes == [2,7,13,19].


Example 2:
Input: n = 1, primes = [2,3,5]
Output: 1
Explanation: 1 is a super ugly number for any given primes.


Constraints:
`1 <= n <= 106`
`1 <= primes.length <= 100`
`2 <= primes[i] <= 1000`
`primes[i]` is guaranteed to be a prime number.

All the values of `primes` are unique and sorted in ascending order.

### Solution 
 To solve the problem of finding the nth super ugly number where super ugly numbers have all prime factors in the given array `primes`, we can use a dynamic programming approach similar to the method used to find "Ugly Numbers" where the numbers only have the prime factors 2, 3, and 5.

### Strategy
1. **Dynamic Array (`dp`)**: We'll use an array `dp` where `dp[i]` represents the ith super ugly number. Initialize with `dp[0] = 1` because 1 is considered a super ugly number.

2. **Pointer Array (`indices`)**: This will be used to keep track of the position for each prime in the `primes` list in the `dp` array. Initially, all values are set to 0 since all primes will first multiply with `dp[0]`.

3. **Next Multiple Array (`next_val`)**: This keeps track of the next value for each prime. It is initialized as a copy of the `primes` since the first values multiplied by 1 (from `dp[0]`) will be the primes themselves.

4. **Iteration**: From i=1 to n, find the next super ugly number which is the minimum value in `next_val`. This value is added to the dynamic list `dp`. For each prime, if its next value matches the current minimum, update its respective index to point to the next element in `dp`, and calculate the new value for `next_val` for that prime.

5. **Output**: After iterating through and populating `dp`, the nth super ugly number will be `dp[n-1]`.

By using the above method, we keep track of and update the next potential super ugly numbers for each prime efficiently.

### Python Code



```python
class Solution:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        dp = [1] * n  # This holds all n super ugly numbers
        indices = [0] * len(primes)  # This tracks the index for each prime number in dp
        next_val = primes[:] # This is the next potential number for each prime.
        
        for i in range(1, n):
            # Find the minimum super ugly number we could form next
            next_ugly = min(next_val)
            dp[i] = next_ugly
            
            # Update all potential values
            for j in range(len(primes)):
                if next_ugly == next_val[j]:
                    # Move forward the pointer for this prime
                    indices[j] += 1
                    # Update the next potential ugly number for this prime
                    next_val[j] = dp[indices[j]] * primes[j]
                    
        return dp[-1] # The nth super ugly number

```

### Complexity
- **Time Complexity**: O(n * k), where k is the number of primes. For each of the n iterations, we may potentially update each of the k elements in `next_val`.
- **Space Complexity**: O(n + k), for the storage of the `dp`, `indices`, and `next_val` arrays.

This solution efficiently handles the creation and tracking of super ugly numbers, leveraging dynamic programming principles and an effective coordination of multiple pointers for each prime.


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
 To solve the problem, it's helpful to realize that the number of times a bulb changes state corresponds to the number of its divisors. Specifically:

1. A bulb at position `k` is toggled on rounds that are divisors of `k`.
2. A bulb toggled an odd number of times will end up being ON.
3. Only perfect squares have an odd number of divisors. This is because divisors typically come in pairs unless the number is a perfect square (in which case one of the divisors is repeated).

Given these insights, the problem boils down to counting how many perfect squares are there <= `n`, since each such number represents a bulb that will remain ON. The square root of `n` can be used to determine the largest such perfect square, indicating the count directly.

Let's implement this solution in Python in the format suitable for a direct submission to LeetCode:



```python
class Solution:
    def bulbSwitch(self, n: int) -> int:
        import math
        return int(math.sqrt(n))

```

Explanation:
- We `import math` to get access to the `math.sqrt()` function.
- Using `math.sqrt(n)` we compute the square root of `n`.
- The integer part of the square root will give us the count of positive integers whose squares are less than or equal to `n`, thus counting the number of bulbs that remain ON.

This direct approach takes advantage of mathematical insight and is extremely efficient with O(1) complexity, which is ideal given the potentially large value of `n` as per the provided constraints.

# 326. Power of Three

### Problem Description 
Given an integer `n`, return `true` if it is a power of three. Otherwise, return `false`.

An integer `n` is a power of three, if there exists an integer `x` such that `n == 3x`.


Example 1:
Input: n = 27
Output: true

Example 2:
Input: n = 0
Output: false

Example 3:
Input: n = 9
Output: true

Example 4:
Input: n = 45
Output: false

Constraints:
`-231 <= n <= 231 - 1`
Follow up: Could you solve it without loops/recursion?

### Solution 
 To determine if a number `n` is a power of three, the simple and intuitive approach includes continually dividing `n` by 3 (given `n` is positive and nonzero) and checking if the remainder is ever non-zero during the divisions. However, this involves using a loop, which we'd like to avoid according to the follow-up challenge.

### Optimal Method without Loops or Recursion:

To solve this problem without using loops or recursion (thus making the solution efficient and robust), we will use mathematics:

1. **Logarithmic Approach:** A number `n` can only be a power of three if the logarithm base 3 of `n` (`log3(n)`) results in an integer. To compute logarithms in different bases using Python, we can utilize:
   \[
   log3(n) = \frac{log_{10}(n)}{log_{10}(3)}
   \]
   Using Python's `math.log10()` function will help compute this.

2. **Check for Integer Result:** After computing the logarithm as mentioned, we need to check if the result is an integer. This can be done using Python's `math.isclose()` function to assert that the floating point representation of our computed log is close to an actual integer, within Python's float precision limits.

Let's proceed with this mathematical approach in the final implementation:



```python
import math

class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        if n <= 0:
            return False
        log_result = math.log10(n) / math.log10(3)
        is_power_of_three = math.isclose(log_result, round(log_result))
        return is_power_of_three

# Example usages:
solution = Solution()
print(solution.isPowerOfThree(27))  # Output: true
print(solution.isPowerOfThree(0))   # Output: false
print(solution.isPowerOfThree(9))   # Output: true
print(solution.isPowerOfThree(45))  # Output: false

```

### Explanation:

1. **Handling `n <= 0`:** Directly return `False` for non-positive numbers as no negative number or zero can be a power of three.

2. **Compute Logarithm:** We first compute the logarithm of `n` with base 3 using change of base formula as described.

3. **Integer Check:** Using `math.isclose`, the function determines if `log_result` is nearly equal to its nearest rounded integer (`round(log_result)`). This effectively checks if `log_result` is an integer.

This code efficiently checks if any given integer is a power of three, adhering to the constraints and optimization requirements (no loops or recursion).

This solution will work directly on the LeetCode platform as it conforms to their expected solution format and Python execution environment.

# 335. Self Crossing

### Problem Description 
You are given an array of integers `distance`.

You start at point `(0,0)` on an X-Y plane and you move `distance[0]` meters to the north, then `distance[1]` meters to the west, `distance[2]` meters to the south, `distance[3]` meters to the east, and so on. In other words, after each move, your direction changes counter-clockwise.

Return `true` if your path crosses itself, and `false` if it does not.


Example 1:
Input: distance = [2,1,1,2]
Output: true

Example 2:
Input: distance = [1,2,3,4]
Output: false

Example 3:
Input: distance = [1,1,1,1]
Output: true

Constraints:
`1 <= distance.length <= 500`
`1 <= distance[i] <= 500`
Follow up: Could you write a one-pass algorithm with `O(1)` extra space?

### Solution 
 This problem involves simulating movements on an X-Y plane based on the pattern North, West, South, and East, repeatedly, in a counter-clockwise manner. Our aim is to determine if at any point during this movement, a position is revisited. This could indicate the path has crossed itself. 

The algorithm follows a simple iterative approach: 
1. Start at the origin (0,0).
2. Maintain the current position and update it based on the movements defined in the `distance` array, alternating directions as specified (North, West, South, East).
3. Check after each movement if this new position has been encountered before. If yes, return `true` as paths crossed. If no, continue until the end of the array.

A hashmap (or set) can be used efficiently to keep track of visited positions to check for any repetition. Let’s directly dive into coding this to adhere to the specifications suitable for a LeetCode solution in Python:



```python
class Solution:
    def isPathCrossing(self, distance: List[int]) -> bool:
        # Initial position
        x, y = 0, 0
        # Directions correspond to movements in the order: North, West, South, East
        directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        # Set to check visited positions
        visited = set()
        # Initially at the origin
        visited.add((x, y))
        
        # Walk through each distance value, updating position
        for i, dist in enumerate(distance):
            dx, dy = directions[i % 4]  # Get direction based on i % 4
            for _ in range(dist):
                x += dx
                y += dy
                if (x, y) in visited:
                    return True
                visited.add((x, y))
        
        return False

# Example usage:
# sol = Solution()
# print(sol.isPathCrossing([2,1,1,2]))  # Output: true
# print(sol.isPathCrossing([1,2,3,4]))  # Output: false
# print(sol.isPathCrossing([1,1,1,1]))  # Output: true

```

The code defines a `Solution` class with a method `isPathCrossing`, which receives a list of integers `distance`. For each subsequence value, the position is updated based on the direction, checked if revisited, and added to a `visited` set if not. The directions are managed by cycling through the modular index of the distance moved (`i % 4`) to handle direction changes after movements. If at any point the new position was previously visited, the method returns `true`, indicating the path is crossing. If no positions repeat by the end of the list, `false` is returned.

# 343. Integer Break

### Problem Description 
Given an integer `n`, break it into the sum of `k` positive integers, where `k >= 2`, and maximize the product of those integers.

Return the maximum product you can get.


Example 1:
Input: n = 2
Output: 1
Explanation: 2 = 1 + 1, 1 × 1 = 1.


Example 2:
Input: n = 10
Output: 36
Explanation: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36.


Constraints:
`2 <= n <= 58`

### Solution 
 This problem can be efficiently solved using Dynamic Programming (DP). The goal is to maximize the product of the integers that add up to `n`. Let's create a DP array where `dp[i]` will store the maximum product we can get by breaking `i`.

Here's the logic:
1. If we want to maximize the product for any number `i`, we can try breaking it into two parts: `j` and `i-j` for all possible values of `j` from 1 to `i-1`.
2. For each split (`j` and `i-j`), there are two possibilities:
    - Taking product directly as `j * (i-j)`
    - Using already calculated max product values for `j` and `i-j`, which would be `j * dp[i-j]`
  
To generalize, we'll consider three cases:
- `j * (i-j)`
- `j * dp[i-j]`
- `dp[j] * (i-j)`

The base case `dp[2]` is initialized as 1 because the only way to split 2 into two positive integers is `1 + 1`. 

This approach will allow us to fill up the DP table up to `dp[n]`, which provides the maximum product for any given `n`.

Here’s the Python code for the solution in LeetCode's format:



```python
class Solution:
    def integerBreak(self, n: int) -> int:
        if n == 2:
            return 1  # As per the problem, the minimum n is 2
        
        dp = [0] * (n + 1)
        dp[2] = 1  # Base case
        
        for i in range(3, n + 1):
            for j in range(1, i//2 + 1):
                # We take the max from splitting at j (and thus i-j as the other part)
                dp[i] = max(dp[i], j * (i - j), j * dp[i - j], dp[j] * (i - j))
        
        return dp[n]

```

You can use this code directly to submit on LeetCode with the problem description "Integer Break":

1. Copy and paste the solution into the coding environment.
2. Execute the provided algorithm to calculate the maximum product.

This method leverages Dynamic Programming to ensure calculations for larger numbers are built upon previously calculated results, ensuring efficiency.

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
 To solve this problem, we need to determine if there exists a vertical line (parallel to the y-axis) that reflects all given points symmetrically. Here's how we can approach the problem:

1. **Identify the Candidate Symmetry Line**: We can find the candidate for the symmetry line by calculating the average of the minimum and maximum x-coordinate values. The average of these x-coordinates could represent a potential symmetry line.

2. **Check Symmetry**: For each point in the input:
   - Compute its reflection over the candidate symmetry line.
   - This reflected point should also exist in the set of given points for the configuration to be symmetrical about that vertical line.

3. **Data Structure**: Use a set to keep track of all unique point pairs. This will allow efficient checking of point existence.

4. **Algorithm Efficiency**: We read all points into a set, which takes O(n) operations assuming set operations are O(1) on average due to hashing. Checking every point's reflection also takes O(n). Thus, the overall efficiency is O(n).

Here is the Python code adhering to LeetCode's solution format:



```python
class Solution:
    def isReflected(self, points: List[List[int]]) -> bool:
        if not points:
            return True
        
        point_set = set((x, y) for x, y in points)
        
        # Find the minimum and maximum x-coordinate values
        min_x = min(x for x, _ in points)
        max_x = max(x for x, _ in points)
        
        # Calculate the midline x value which should be the line of symmetry
        midline_x = (min_x + max_x) / 2
        
        # Check each point's reflection
        for x, y in points:
            reflected_x = 2 * midline_x - x
            if (reflected_x, y) not in point_set:
                return False
        
        return True

# Example usage:
sol = Solution()
print(sol.isReflected([[1, 1], [-1, 1]]))  # Output: True
print(sol.isReflected([[1, 1], [-1, -1]]))  # Output: False

```

### Additional Notes:
- This code assumes that calculating the value `(2 * midline_x - x)` results in another point which must also exist in the `point_set` for the points to be symmetrical about the line `x = midline_x`.
- By using a set, the look-up operations for checking if a point exists is average O(1), making the solution efficient.

This completes a LeetCode-compatible solution that you can directly run on their platform under the Python3 environment.

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
 To tackle this problem, we should consider how to count numbers with unique digits in a given range defined by `n`, where the maximum value is `10^n`.

### Explanation:

1. **Base Case (n = 0)**:
    - The only number in the range `[0, 1)` is 0. Hence, the answer is 1.

2. **Initial Counts**:
    - For `n=1`, the range is `[0, 10)`, where all numbers have unique digits since there's only one digit, thus 10 possibilities (0 to 9).

3. **Counts for n >= 2**:
    - Begin with initial range count (0 to 10, which is `10` counts for `n=1`).
    - Calculate the count for numbers with unique digits for each digit count incrementally until `n`.

    - The key observation here is:
        - For one digit: 10 possibilities (0-9)
        - For two digits: First digit (non-zero): 9 possibilities (1-9), second digit: 9 possibilities (as 0 can be used here, but the digit must differ from the first, so 10-1).
        - In general for `k` digits, the count would be:
            `9 * 9 * 8 * 7 * ...` (the product decreases as digits must differ from each other).

    - So for each `k` up to `min(n, 10)`, we calculate `9 * 9 * 8 ...` based on the number of available digits and then sum up these computations to get the total count of unique digit numbers in the range `[0, 10^n)`.

### Python Code:



```python
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        if n == 0:
            return 1  # As per problem constraints and example 2
        
        # Base count for n = 1
        result = 10

        # Starting point for numbers greater than length 1
        maxUniqueDigits = min(n, 10)
        uniqueDigits = 9
        availableNumber = 9
        
        # Compute the unique counts
        for i in range(2, maxUniqueDigits + 1):
            uniqueDigits *= availableNumber
            result += uniqueDigits
            availableNumber -= 1  # Reduce the available numbers for the next position
        
        return result

# Example usages:
sol = Solution()
print(sol.countNumbersWithUniqueDigits(2))  # Output: 91
print(sol.countNumbersWithUniqueDigits(0))  # Output: 1

```

### Further Understanding:

- The solution uses a combinatorial approach to count the number of valid sequences possible for each length `k` from 1 to `n`.
- The loop sustainably decreases the available numbers once numbers have been used for digits, consistent with combinatorial counting principles in arrangements where order and distinguishing are crucial.

# 360. Sort Transformed Array

### Problem Description 
Given a sorted integer array `nums` and three integers `a`, `b` and `c`, apply a quadratic function of the form `f(x) = ax2 + bx + c` to each element `nums[i]` in the array, and return the array in a sorted order.


Example 1:
Input: nums = [-4,-2,2,4], a = 1, b = 3, c = 5
Output: [3,9,15,33]

Example 2:
Input: nums = [-4,-2,2,4], a = -1, b = 3, c = 5
Output: [-23,-5,1,7]

Constraints:
`1 <= nums.length <= 200`
`-100 <= nums[i], a, b, c <= 100`
`nums` is sorted in ascending order.

Follow up: Could you solve it in `O(n)` time?

### Solution 
 To tackle this problem, we'll leverage the sorted nature of `nums` combined with the insights on how quadratic functions behave:

1. A quadratic function is defined as \(f(x) = ax^2 + bx + c\).
2. Depending on the coefficient `a`, the function can either open upwards (concave) if `a > 0` or downwards (convex) if `a < 0`.
3. When \(a 
eq 0\), the function has a vertex at \(-\frac{b}{2a}\), which could be within, left, or right of the provided array range.
4. By evaluating the endpoints of the array (first and last elements) and the vertex, we can determine where the minimum or maximum (based on the sign of `a`) values occur.
5. We can utilize a two-pointer approach: One pointer beginning at the start of the array and one at the end, choosing whether to fill the output from the start or end based on whether `a` is positive or negative.

Given these observations, we can decide to evaluate and populate our output array from left/right based on the value of `a`. This avoids the need for sorting the output array after transformation, achieving the desired complexity.

Let's implement this understanding in the LeetCode format:



```python
from typing import List

class Solution:
    def sortTransformedArray(self, nums: List[int], a: int, b: int, c: int) -> List[int]:
        def quadratic(x: int) -> int:
            return a * x**2 + b * x + c
        
        n = len(nums)
        transformed = [0] * n
        left, right = 0, n - 1
        index = 0 if a < 0 else n - 1  # Decide to fill the array from front or back
        
        while left <= right:
            transformed_left = quadratic(nums[left])
            transformed_right = quadratic(nums[right])
            if a >= 0:  # Fill from the end of the `transformed` array
                if transformed_right >= transformed_left:
                    transformed[index] = transformed_right
                    right -= 1
                else:
                    transformed[index] = transformed_left
                    left += 1
                index -= 1
            else:  # Fill from the start of the `transformed` array
                if transformed_left <= transformed_right:
                    transformed[index] = transformed_left
                    left += 1
                else:
                    transformed[index] = transformed_right
                    right -= 1
                index += 1
        
        return transformed

```

This approach assures us an `O(n)` complexity since each of the `nums` is visited and processed into the `transformed` array exactly once, without requiring an additional sort step. The use of a two-pointer technique directly places elements in the correct position in the result array by understanding the nature of the quadratic function.

# 365. Water and Jug Problem

### Problem Description 
You are given two jugs with capacities `jug1Capacity` and `jug2Capacity` liters. There is an infinite amount of water supply available. Determine whether it is possible to measure exactly `targetCapacity` liters using these two jugs.

If `targetCapacity` liters of water are measurable, you must have `targetCapacity` liters of water contained within one or both buckets by the end.

Operations allowed:
Fill any of the jugs with water.

Empty any of the jugs.

Pour water from one jug into another till the other jug is completely full, or the first jug itself is empty.


Example 1:
Input: jug1Capacity = 3, jug2Capacity = 5, targetCapacity = 4
Output: true
Explanation: The famous Die Hard example 

Example 2:
Input: jug1Capacity = 2, jug2Capacity = 6, targetCapacity = 5
Output: false

Example 3:
Input: jug1Capacity = 1, jug2Capacity = 2, targetCapacity = 3
Output: true

Constraints:
`1 <= jug1Capacity, jug2Capacity, targetCapacity <= 106`

### Solution 
 To determine if a given `targetCapacity` can be measured using two jugs with specified capacities, we can use a mathematical insight from number theory, specifically relating to the linear Diophantine equation of the form ax + by = c. In this context, it translates to whether a combination of the two jugs' capacities, through fill, empty, and pour operations, can measure the exact `targetCapacity`.

Here's a concise breakdown of the solution:

1. **Greatest Common Divisor (GCD):** According to the properties of Diophantine equations, the equation ax + by = d has solutions if and only if "d" is a multiple of gcd(a, b). Therefore, for our jug problem, `targetCapacity` must be a multiple of the gcd of `jug1Capacity` and `jug2Capacity`.

2. **Target Reachability:** Secondly, it's also obvious that even if `targetCapacity` is a multiple of the gcd, it should not exceed the combined capacity of the two jugs, i.e., `targetCapacity` should be less than or equal to `jug1Capacity + jug2Capacity`.

By checking these conditions, we can decide whether the `targetCapacity` is achieveable.

Below is the Python code, which can be executed directly on LeetCode:



```python
class Solution:
    def canMeasureWater(self, jug1Capacity: int, jug2Capacity: int, targetCapacity: int) -> bool:
        from math import gcd
        
        # Step 1: Check if targetCapacity is greater than the sum of both jugs
        if targetCapacity > jug1Capacity + jug2Capacity:
            return False
        
        # Step 2: Check if targetCapacity is a multiple of the gcd of jug1Capacity and jug2Capacity
        return targetCapacity % gcd(jug1Capacity, jug2Capacity) == 0

# The following lines are for demonstration and are typically not included in LeetCode submissions.
# Examples tests mimicking the ones from the problem statement:
sol = Solution()
print(sol.canMeasureWater(3, 5, 4))  # Output: true
print(sol.canMeasureWater(2, 6, 5))  # Output: false
print(sol.canMeasureWater(1, 2, 3))   # Output: true

```

In this provided solution:
- We use the `gcd` function from Python's math library to calculate the greatest common divisor.
- A logical condition is used to check if the `targetCapacity` is a multiple of the GCD and whether it is feasible with respect to the total capacity of both jugs.

This approach, using mathematical deduction, is efficient, ensuring an optimal and direct conclusion.

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
 To solve the problem of checking whether a given integer `num` is a perfect square, we can use several approaches. One optimal solution is the binary search method due to its logarithmic time complexity. We will avoid using the square root function directly, adhering to the problem's constraints.

**Explanation of the Binary Search Approach**:
1. **Initialization**: Start with a "low" boundary at 1 and a "high" boundary at `num`. The perfect square root of `num`, if it exists, must lie within this range.
2. **Binary Search**: Conduct a binary search between `low` and `high`. Calculate the midpoint `mid`, and then compute the square of `mid` (i.e., `mid_square = mid * mid`).
3. **Comparison**:
    - If `mid_square` is equal to `num`, then `num` is a perfect square, and we return `True`.
    - If `mid_square` is less than `num`, move the `low` pointer to `mid + 1` to search the higher range.
    - If `mid_square` is more than `num`, move the `high` pointer to `mid - 1` to search the lower range.
4. **Termination**: If the search bounds cross (`low` > `high`), then `num` is not a perfect square, and we should return `False`.

This method is efficient and keeps the number of computations relatively small even for the higher range of the input constraints.

Here's how you would implement this in a Python function compatible with LeetCode:



```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num < 2:
            return True
        
        low, high = 1, num
        
        while low <= high:
            mid = (low + high) // 2
            mid_square = mid * mid
            
            if mid_square == num:
                return True
            elif mid_square < num:
                low = mid + 1
            else:
                high = mid - 1
                
        return False

# The Solution class can be tested with examples:
# Example inputs
sol = Solution()
print(sol.isPerfectSquare(16)) # Expected output: True
print(sol.isPerfectSquare(14)) # Expected output: False

```

This solution will be optimal and directly executable with the given problem constraints on platforms such as LeetCode. The binary search ensures that the operations are minimized, handling even the maximum constraints efficiently.

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
 To solve the problem of finding the largest subset of numbers from a given set such that every pair of elements in this subset satisfies the divisibility condition described, we can employ a dynamic programming approach. The problem is analogous to finding the longest divisible subset which is an extension of the longest increasing subsequence problem.

### Solution Outline

1. **Sort the Numbers**: Start by sorting the `nums` list. This helps to only check the divisibility condition for pairs in one direction (i.e., a smaller number dividing a larger number) as we iterate through the sorted list.

2. **Dynamic Programming Setup**:
   - Create an array `dp` where `dp[i]` represents the length of the divisible subset ending at index `i`.
   - Create an array `parent` to reconstruct the subset. `parent[i]` will store the index of the previous element in the subset chain up to index `i`.

3. **Fill the DP Table**:
   - Initially set each index in `dp` to 1, since each element is trivially a subset of itself.
   - Iterate over each pair `(i, j)` where `i < j`. If `(nums[j] % nums[i] == 0)`, update `dp[j]` and `parent[j]` to account for the inclusion of `nums[i]` in the subset leading to `nums[j]`.

4. **Determine the Largest Subset**:
   - Identify the index of the maximum value in `dp` which corresponds to the end of the largest divisible subset.
   - Trace back using the `parent` array from this index to reconstruct the subset.

5. **Return the Result**.

By using the dynamic programming approach coupled with sort-based preprocessing, we can efficiently solve this problem with a complexity near O(n^2), where n is the number of elements in `nums`.



```python
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        n = len(nums)
        if n == 0:
            return []
        
        nums.sort()
        dp = [1] * n  # At minimum, each subset includes the element itself
        parent = [-1] * n
        max_len = 0  # Length of the maximum subset
        max_index = 0  # Ending index of the maximum subset
        
        # Fill dp and parent arrays
        for i in range(n):
            for j in range(i + 1, n):
                if nums[j] % nums[i] == 0 and dp[j] < dp[i] + 1:
                    dp[j] = dp[i] + 1
                    parent[j] = i
        
        # Find the maximum size of the subset and where it ends
        for i in range(n):
            if dp[i] > max_len:
                max_len = dp[i]
                max_index = i
        
        # Reconstruct the subset
        result = []
        k = max_index
        while k != -1:
            result.append(nums[k])
            k = parent[k]
        
        return result

```

This code snippet follows the LeetCode problem-solving format, ensuring it can be directly used on the platform for testing and submission.

# 396. Rotate Function

### Problem Description 
You are given an integer array `nums` of length `n`.

Assume `arrk` to be an array obtained by rotating `nums` by `k` positions clock-wise. We define the rotation function `F` on `nums` as follow:
`F(k) = 0 * arrk[0] + 1 * arrk[1] + ... + (n - 1) * arrk[n - 1].`
Return the maximum value of `F(0), F(1), ..., F(n-1)`.


Example 1:
Input: nums = [4,3,2,6]
Output: 26
Explanation:
F(0) = (0 * 4) + (1 * 3) + (2 * 2) + (3 * 6) = 0 + 3 + 4 + 18 = 25
F(1) = (0 * 6) + (1 * 4) + (2 * 3) + (3 * 2) = 0 + 4 + 6 + 6 = 16
F(2) = (0 * 2) + (1 * 6) + (2 * 4) + (3 * 3) = 0 + 6 + 8 + 9 = 23
F(3) = (0 * 3) + (1 * 2) + (2 * 6) + (3 * 4) = 0 + 2 + 12 + 12 = 26
So the maximum value of F(0), F(1), F(2), F(3) is F(3) = 26.


Example 2:
Input: nums = [1000000007]
Output: 0

Constraints:
`n == nums.length`
`1 <= n <= 105`
`-231 <= nums[i] <= 231 - 1`

### Solution 
 To solve this problem, you need to understand both the operation of the rotation function F and how each rotation impacts the outcome. We start by calculating F(0) as defined, then find a way to compute subsequent F(k) values efficiently based on the previous value. This is critical to avoid an O(n^2) complexity by re-computing F(k) from scratch for each rotation.

### Step-by-Step Solution:

1. **Calculate F(0)**:
   Calculate the initial value directly by applying the formula:
   \[
   F(0) = 0 \times nums[0] + 1 \times nums[1] + \ldots + (n - 1) \times nums[n - 1]
   \]

2. **Derive F(k) from F(k-1)**:
   When the array is rotated by one step to get `arr_k` from `arr_{k-1}`, the indexes shift in a way that affects F drastically yet predictably:
   \[
   F(k) = F(k-1) + Sum(nums) - n \times nums[n - k]
   \]
   - `Sum(nums)` is the sum of all numbers in `nums`.
   - `n` is the number of elements in `nums`.

3. **Implement Equation for `F(k)`, loop to find maximum**:
   Start from `F(0)`, apply the derived formula iteratively for each k, and maintain a maximum.

### Python Implementation:



```python
class Solution:
    def maxRotateFunction(self, nums: List[int]) -> int:
        n = len(nums)
        sum_nums = sum(nums)
        
        # Calculate F(0)
        F = sum(i * nums[i] for i in range(n))
        max_F = F
        
        # Compute subsequent values of the rotation function
        current_F = F
        for k in range(1, n):
            current_F += sum_nums - n * nums[n - k]
            max_F = max(max_F, current_F)
        
        return max_F

```

### Explanation:
- `current_F` stores the value of F for the current rotation.
- We update `current_F` for each iteration using the derived formula, reducing unnecessary computations.
- `max_F` keeps track of the maximum F encountered during the iterations.

### Efficiency:
The function computes the values in linear time, i.e., O(n), since each transition from F(k-1) to F(k) is done in constant time, and there are 'n' such transitions plus some initial calculations. Thus, this method avoids the naive O(n^2) complexity that would occur if recalculating from scratch for each rotation.

# 397. Integer Replacement

### Problem Description 
Given a positive integer `n`, you can apply one of the following operations:
If `n` is even, replace `n` with `n / 2`.

If `n` is odd, replace `n` with either `n + 1` or `n - 1`.

Return the minimum number of operations needed for `n` to become `1`.


Example 1:
Input: n = 8
Output: 3
Explanation: 8 -> 4 -> 2 -> 1

Example 2:
Input: n = 7
Output: 4
Explanation: 7 -> 8 -> 4 -> 2 -> 1
or 7 -> 6 -> 3 -> 2 -> 1

Example 3:
Input: n = 4
Output: 2

Constraints:
`1 <= n <= 231 - 1`

### Solution 
 To solve the specified problem using Python, an efficient approach involves using a Breadth-First Search (BFS) method. This method works by exploring the shortest possible paths from `n` to `1` systematically. For this purpose, a queue will be used to manage the current number and the steps taken so far. The BFS technique ensures that once we encounter `1` in our process, the corresponding step count is the minimal set of operations required.

Given the operations:
1. If `n` is even, replace `n` with `n / 2`.
2. If `n` is odd, replace `n` with either `n + 1` or `n - 1`.

Key Points for BFS Implementation:
- We need to keep track of numbers that we've already processed to avoid unnecessary re-computation. This can typical be done using a set or an array.
- We progress by evaluating each possibility from the current number (either halving the number if even, or adding or subtracting one if odd) and recording how many steps it took to get there.

I will implement the solution in Python as described, with a few optimizations:



```python
from collections import deque

class Solution:
    def integerReplacement(self, n: int) -> int:
        # Breadth-First Search implementation
        queue = deque([(n, 0)])  # (current value, current step count)
        visited = set()
        
        while queue:
            current, steps = queue.popleft()
            
            if current == 1:
                return steps
            if current in visited:
                continue
            visited.add(current)
            
            if current % 2 == 0:
                queue.append((current // 2, steps + 1))
            else:
                queue.append((current + 1, steps + 1))
                queue.append((current - 1, steps + 1))
                
        return -1  # Should never reach here since we're guaranteed to eventually hit 1

# Test cases
sol = Solution()
print(sol.integerReplacement(8))  # Output: 3
print(sol.integerReplacement(7))  # Output: 4
print(sol.integerReplacement(4))  # Output: 2

```

### Explanations:
1. **Data Structure**: The `queue` is used for BFS, initiated with the tuple `(n, 0)` where `n` is the input number and `0` is the initial step count. `visited` keeps track of numbers already evaluated.
2. **BFS Loop**: We process elements in the queue, dequeuing the current state `(current, steps)`.
3. **Termination Condition**: If `current` becomes `1`, we return the `steps` as the output since it represents the minimum steps required to reduce `n` to `1`.
4. **Processing**: Depending on the parity of `current`:
  - If even, we enqueue `current // 2`.
  - If odd, we enqueue both potential operations, `current + 1` and `current - 1`.
5. **Visited Check**: This helps us avoid infinite loops or re-processing the same number.

### Complexity:
- **Time Complexity**: O(log n) in the average case, considering each step potentially halves the value of `n`, leading to a logarithmic number of steps.
- **Space Complexity**: O(log n) due to storing elements in the queue and visited set, mainly dictated by the number of operations needed.

This solution is both clear in intention and efficient in implementation, suitable for running directly on LeetCode or a similar platform.

# 400. Nth Digit

### Problem Description 
Given an integer `n`, return the `nth` digit of the infinite integer sequence `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]`.


Example 1:
Input: n = 3
Output: 3

Example 2:
Input: n = 11
Output: 0
Explanation: The 11th digit of the sequence 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... is a 0, which is part of the number 10.


Constraints:
`1 <= n <= 231 - 1`

### Solution 
 To solve this problem efficiently, we need to understand the structure of the infinite sequence of all positive integers. The sequence consists of:

- Single-digit numbers: 1 through 9 (9 numbers)
- Double-digit numbers: 10 through 99 (90 numbers)
- Triple-digit numbers: 100 through 999 (900 numbers)
- And so forth...

Each group can be calculated in blocks:
- There are 9 single-digit numbers, contributing `9 * 1 = 9` digits.
- There are 90 double-digit numbers, contributing `90 * 2 = 180` digits.
- There are 900 triple-digit numbers, contributing `900 * 3 = 2700` digits.
- ...

To locate the `nth` digit:
1. Determine in which block (or which range of number sizes) the `nth` digit falls.
2. Calculate the exact number within that block that contains the `nth` digit.
3. Identify the digit within that number.



```python
class Solution:
    def findNthDigit(self, n: int) -> int:
        # Define the digit length and the count of numbers having that length
        digit_length = 1
        count = 9  # 9 single-digit numbers, 90 double-digit numbers, etc.
        
        # Find the range that contains the nth digit
        while n > digit_length * count:
            n -= digit_length * count
            digit_length += 1  # increase the digit length (1 for single, 2 for double, etc.)
            count *= 10  # next range (90, 900, 9000, ...)
        
        # Find the exact number where the nth digit is located
        start = 10 ** (digit_length - 1)  # start number for that digit length
        num = start + (n - 1) // digit_length  # find the number in the sequence

        # Find the digit within the number
        # (n - 1) % digit_length gives the position of the digit in this number
        return int(str(num)[(n - 1) % digit_length])

# Examples
sol = Solution()
print(sol.findNthDigit(3))  # output: 3
print(sol.findNthDigit(11)) # output: 0

```

###### Explanation:
- We start with the smallest range (1-digit numbers) and subtract their count from `n` until `n` is small enough to fit within a given range.
- Once we determine the range, we calculate the start of the numbers in that range (`start`).
- We find the exact number by adding `(n - 1) // digit_length` to `start`. Here, `(n - 1)` is used to convert from 1-indexing (nth digit) to 0-indexing.
- Finally, we determine which digit in the specific number we are interested in by calculating `(n - 1) % digit_length` and select that digit from the number.

This method ensures that we process the problem efficiently, without generating a full sequence up to the `nth` digit, allowing us to comply with large constraint limitations.

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
 Below is a Python solution for the given problem using LeetCode's format. This solution focuses on finding all arithmetic subarrays with more than two elements and counting them. The main idea is to use dynamic programming to efficiently count slices.



```python
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        # `n` is the length of the input array `nums`
        n = len(nums)
        if n < 3:
            return 0
        
        # `count` will store the total number of arithmetic subarrays found
        count = 0
        # `current_length` will store the length of the current arithmetic slice being formed
        current_length = 0
        
        # Iterate from the second element to the last element of the array
        for i in range(2, n):
            # Check if the current triplet forms an arithmetic slice
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                # Extend the current arithmetic slice
                current_length += 1
                # Add the length of all possible subarrays ending at current index: it's a cumulative addition
                count += current_length
            else:
                # Reset the length if the current slice no longer maintains arithmetic properties
                current_length = 0
        
        return count

```

### Explanation:

The solution works in this way:
1. Initialize a counter `count` to keep track of the number of arithmetic slices.
2. Initialize `current_length` which will be used to maintain the length of ongoing arithmetic slices dynamically.
3. Loop through the sequence starting from the third element.
4. For every triplet checked, if it satisfies the arithmetic condition (consecutive differences are the same), then the slice can be extended. Extend the slice and add to the count the number of triangular number of subarrays that can end at this index due to the ongoing arithmetic slice.
5. If the triplet does not satisfy the condition, reset the length of the ongoing arithmetic slices.

This approach ensures that all arithmetic slices are counted efficiently with a single pass through the array.

### Complexity:
- Time complexity: O(n), where `n` is the number of elements in the array. It only needs a single pass through the array.
- Space complexity: O(1), as it uses a fixed amount of extra space regardless of the input size.

This solution is efficient and fits within the constraints provided (`1 <= nums.length <= 5000` and `nums` element values between `-1000` and `1000`).

# 423. Reconstruct Original Digits from English

### Problem Description 
Given a string `s` containing an out-of-order English representation of digits `0-9`, return the digits in ascending order.


Example 1:
Input: s = "owoztneoer"
Output: "012"

Example 2:
Input: s = "fviefuro"
Output: "45"

Constraints:
`1 <= s.length <= 105`
`s[i]` is one of the characters `["e","g","f","i","h","o","n","s","r","u","t","w","v","x","z"]`.

`s` is guaranteed to be valid.

### Solution 
 To solve the given problem, we will leverage the uniqueness of certain characters in the spelled-out words of English digits.

## Strategy:
1. **Unique Character Identification:** Identify unique characters that can distinguish a number from others. For instance:
   - 'z' only appears in "zero"
   - 'w' only appears in "two"
   - 'u' only appears in "four"
   - 'x' only appears in "six"
   - 'g' only appears in "eight"
   - etc.
2. **Count Occurrences:** Create a frequency count of every character in the input string.
3. **Reconstruct Digits**: For each unique character that identifies a digit, deduce how many times that digit appears from the count of the unique character, and adjust the counts of its constituent characters accordingly.
4. **Build Result**: Form the result string by appending each digit the number of times it was computed from the counters.

## Implementation:
We'll use Python's `collections.Counter` for an efficient frequency count and build the final number in ascending order.

The steps are provided in comments within the code for clearer explanation.



```python
from collections import Counter

class Solution:
    def originalDigits(self, s: str) -> str:
        # Frequency count of all characters in the input string
        count = Counter(s)

        # Prepare to store count of each digit from 0 to 9
        output = {}
        
        # Identifying each number by a unique character in its spelling
        # Counting "zero"
        output[0] = count['z']
        # Counting "two"
        output[2] = count['w']
        # Counting "four"
        output[4] = count['u']
        # Counting "six"
        output[6] = count['x']
        # Counting "eight"
        output[8] = count['g']
        
        # Now count numbers which share characters that are now partly used
        # Counting "one" (o is in 0, 1, 2, 4)
        output[1] = count['o'] - output[0] - output[2] - output[4]
        # Counting "three" (h is in 3, 8)
        output[3] = count['h'] - output[8]
        # Counting "five" (f is in 4, 5)
        output[5] = count['f'] - output[4]
        # Counting "seven" (s is in 6, 7)
        output[7] = count['s'] - output[6]
        # Counting "nine" (i is in 5, 6, 8, 9)
        output[9] = count['i'] - output[5] - output[6] - output[8]
        
        # Building the output result based on the counts in ascending order
        result = []
        for i in range(10):
            if i in output:
                result.append(str(i) * output[i])
        
        return ''.join(result)

# Example usage:
sol = Solution()
print(sol.originalDigits("owoztneoer"))  # Output: "012"
print(sol.originalDigits("fviefuro"))    # Output: "45"

```

This solution efficiently builds the output based on characteristic counts. It runs in linear time relative to the size of the input string due to the single pass needed to count characters and constant time operations to derive each digit count from the counts of specific letters.

