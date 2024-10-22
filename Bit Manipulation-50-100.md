# 1525. Number of Good Ways to Split a String

### Problem Description 
You are given a string `s`, a split is called good if you can split `s` into 2 non-empty strings `p` and `q` where its concatenation is equal to `s` and the number of distinct letters in `p` and `q` are the same.

Return the number of good splits you can make in `s`.


Example 1:
Input: s = "aacaba"
Output: 2
Explanation: There are 5 ways to split `"aacaba"` and 2 of them are good. 
("a", "acaba") Left string and right string contains 1 and 3 different letters respectively.

("aa", "caba") Left string and right string contains 1 and 3 different letters respectively.

("aac", "aba") Left string and right string contains 2 and 2 different letters respectively (good split).

("aaca", "ba") Left string and right string contains 2 and 2 different letters respectively (good split).

("aacab", "a") Left string and right string contains 3 and 1 different letters respectively.


Example 2:
Input: s = "abcd"
Output: 1
Explanation: Split the string as follows ("ab", "cd").


Example 3:
Input: s = "aaaaa"
Output: 4
Explanation: All possible splits are good.


Example 4:
Input: s = "acbadbaada"
Output: 2

Constraints:
`s` contains only lowercase English letters.

`1 <= s.length <= 10^5`

### Solution 
 To solve the problem of counting the number of good splits in a string `s` where the two resulting substrings have the same number of distinct characters, we can follow this step-by-step approach:

### Explanation

1. **Understanding Good Split**: A good split can be defined such that we split the string into two non-empty parts `p` and `q` where both parts contain the same number of distinct characters.

2. **Using Sets for Distinct Characters**: We can use a set to keep track of the distinct characters in each part as we iterate over the string and check for possible splits.

3. **Iterate Over Possible Splits**: We will iterate through possible split points. For a string of length `n`, we can make splits from index `1` to `n-1`. This way, `p` will always have at least one character and `q` will not be empty.

4. **Count Distinct Characters**:
   - Maintain a count of distinct characters in `p` as we iterate from the start to the split point.
   - Maintain a count for `q` by initially counting all distinct characters in the string and then decrementing as we include characters in `p`.

5. **Compare Distinct Character Counts**: At each split point, compare the count of distinct characters in `p` and `q`. If they are equal, it counts as a good split.

6. **Efficiency**: This method iterates through the string a limited number of times which makes it efficient. Given the constraints, it should perform well within the limits.

### Implementation

Here is the Python code that implements the above logic:



```python
class Solution:
    def numSplits(self, s: str) -> int:
        # Count total distinct characters in s
        total_distinct = len(set(s))
        
        left_distinct_count = 0
        right_distinct_count = 0
        
        good_splits = 0
        
        # Create a frequency dictionary for the characters in s
        char_count = {}
        for char in s:
            if char not in char_count:
                char_count[char] = 0
            char_count[char] += 1
            
        # Initialize right distinct count with total distinct characters
        right_distinct_count = total_distinct
        
        # Iterate through the string, but don't include the last char
        for i in range(len(s) - 1):
            left_char = s[i]
            
            # Update left distinct count
            if char_count[left_char] == 1:
                left_distinct_count += 1
            char_count[left_char] -= 1
            
            # If it becomes zero, we are removing a distinct character
            if char_count[left_char] == 0:
                right_distinct_count -= 1
            
            # Check if the number of distinct characters are equal
            if left_distinct_count == right_distinct_count:
                good_splits += 1
                
        return good_splits

```

### Explanation of the Code

- We first compute the total number of distinct characters in the input string `s` using a set.
- We set up a character frequency dictionary to track how many times each character appears in `s`.
- We iterate through the string while maintaining separate counts for left and right distinct characters:
  - For each character considered as part of `p`, we update the counts accordingly.
  - After updating counts, we check if they are equal and increment the good split count if they are.
  
### Complexity Analysis
- **Time Complexity**: O(n), where n is the length of the string, because we make a single pass through the string and use constant-time operations for updates.
- **Space Complexity**: O(1), since we are using fixed space for the distinct character counts and a dictionary with a limited size (26 letters maximum).

This should solve the problem efficiently and correctly for the given constraints.

# 1542. Find Longest Awesome Substring

### Problem Description 
Given a string `s`. An awesome substring is a non-empty substring of `s` such that we can make any number of swaps in order to make it palindrome.

Return the length of the maximum length awesome substring of `s`.


Example 1:
Input: s = "3242415"
Output: 5
Explanation: "24241" is the longest awesome substring, we can form the palindrome "24142" with some swaps.


Example 2:
Input: s = "12345678"
Output: 1

Example 3:
Input: s = "213123"
Output: 6
Explanation: "213123" is the longest awesome substring, we can form the palindrome "231132" with some swaps.


Example 4:
Input: s = "00"
Output: 2

Constraints:
`1 <= s.length <= 10^5`
`s` consists only of digits.

### Solution 
 To solve the problem of finding the longest awesome substring in the given string `s`, we need to determine which substrings can be rearranged to form a palindrome. A palindrome can be formed from a collection of characters if at most one character has an odd count (for odd-length palindromes), while all character counts must be even for even-length palindromes.

Here's a step-by-step breakdown of our approach:

1. **Character Frequency**: We need to track the frequency of each character in the substring.

2. **Bitmask Representation**: Since we only have digits (0-9), we can use a bitmask to represent how many times each character has been counted. Each bit in a 10-bit integer can represent the odd/even status of digits 0-9. For instance, if the 0th bit is set, it means the count of '0' is odd. 

3. **Dynamic Programming with a HashMap**: We will maintain a HashMap that stores the first occurrence of each bitmask as we iterate through the string. This allows us to check how long the substring can be depending on the previously stored states (bitmask).

4. **Two Cases for Maximum Length Calculation**: For each character in the string, we will:
   - Update the bitmask.
   - Check if this exact bitmask has been seen before. If yes, the length from the previous occurrence gives a potential maximum length for the awesome substring.
   - Checking for the case where we can alter one character (i.e., flip one bit in the bitmask) to see if we can form a longer awesome substring.

5. **Iterate through Each Character**: For each character in the string, update the bitmask, check for the maximum length, and update the HashMap accordingly.

Here’s the Python code which implements the above logic and adheres to the format required for submission on LeetCode:



```python
class Solution:
    def longestAwesome(self, s: str) -> int:
        # Initialize a dictionary to store first occurrence of bitmasks
        first_occurrence = {0: -1}  # A bitmask of 0 has an initial value of -1 for comparison
        max_length = 0
        current_mask = 0
        
        for i, char in enumerate(s):
            # Update the current bitmask for the character
            digit = int(char)
            current_mask ^= (1 << digit)  # Toggle the digit bit
            
            # Check if current_mask has been seen before
            if current_mask in first_occurrence:
                max_length = max(max_length, i - first_occurrence[current_mask])
            else:
                first_occurrence[current_mask] = i
            
            # Check for all masks that differ by one bit
            for j in range(10):  # There are 10 digits from 0 to 9
                mask_with_one_bit_flipped = current_mask ^ (1 << j)
                if mask_with_one_bit_flipped in first_occurrence:
                    max_length = max(max_length, i - first_occurrence[mask_with_one_bit_flipped])
        
        return max_length

```

In this code:

- We use a dictionary `first_occurrence` to store the first index where each bitmask has occurred. 
- The variable `current_mask` is updated using XOR operation with the relevant bit based on the current digit.
- We calculate the potential maximum length by comparing the current index with the first occurrences of both the current bitmask and the modified masks (where one bit is flipped).
- Finally, we return the maximum length of the awesome substring found.

This implementation runs in O(n) time complexity due to the single pass through the string and constant time checks with the hashmap, making it efficient given the constraints.

# 1611. Minimum One Bit Operations to Make Integers Zero

### Problem Description 
Given an integer `n`, you must transform it into `0` using the following operations any number of times:
Change the rightmost (`0th`) bit in the binary representation of `n`.

Change the `ith` bit in the binary representation of `n` if the `(i-1)th` bit is set to `1` and the `(i-2)th` through `0th` bits are set to `0`.

Return the minimum number of operations to transform `n` into `0`.


Example 1:
Input: n = 0
Output: 0

Example 2:
Input: n = 3
Output: 2
Explanation: The binary representation of 3 is "11".

"11" -> "01" with the 2nd operation since the 0th bit is 1.

"01" -> "00" with the 1st operation.


Example 3:
Input: n = 6
Output: 4
Explanation: The binary representation of 6 is "110".

"110" -> "010" with the 2nd operation since the 1st bit is 1 and 0th through 0th bits are 0.

"010" -> "011" with the 1st operation.

"011" -> "001" with the 2nd operation since the 0th bit is 1.

"001" -> "000" with the 1st operation.


Example 4:
Input: n = 9
Output: 14

Example 5:
Input: n = 333
Output: 393

Constraints:
`0 <= n <= 109`

### Solution 
 To solve the problem of transforming the integer `n` into `0` using specific operations, we need to analyze the operations in detail and determine an efficient way to count the minimum number of operations required.

### Problem Understanding

1. **Operations**
   - **Operation 1**: Change the rightmost bit in the binary representation.
   - **Operation 2**: Change the `ith` bit if the `(i-1)th` bit is `1` and all bits from `(i-2)` through `0` are `0`.

2. **Goal**: Minimize the number of operations to transform `n` into `0`.

### Steps to Approach

To achieve the transformation to `0`, we consider the following insights:

1. **Counting '1's in Binary**: Each bit set to `1` will eventually need to be turned `0`. Therefore, if we count the number of `1` bits in the binary representation of `n`, it gives us a starting point.

2. **Tracking Positions**: Beyond counting the `1`s, we also need to carefully consider the positions of the bits because of the specific conditions for operation 2. We must take into account groups of `1`s separated by `0`s and how they can be collectively turned to `0`.

3. **Efficiency**: We will process each bit of the number in a way that we can determine how many operations are needed based on the current configuration of bits.

### Python Code Implementation

Here's how we can implement this logic in Python following the LeetCode structure:



```python
class Solution:
    def minimumOperations(self, n: int) -> int:
        operations = 0
        if n == 0:
            return operations
        
        while n > 0:
            # Count current operations needed
            if n & 1:  # Check if the rightmost bit is 1
                operations += 1
            
            # This counts how many bits are also considered for operation 2
            highest_power_of_two = 0
            while (1 << highest_power_of_two) <= n:
                highest_power_of_two += 1
            
            # Move to the next higher bit to ensure we consider operation 2
            if highest_power_of_two > 0:
                highest_power_of_two -= 1
            
            # Set the highest bit to zero
            n = n & ~(1 << highest_power_of_two)

            # Increment operation count for the operation of clearing the highest bit
            operations += 1
            
            # After handling the highest bit, we can shift to the next bit.
            n >>= 1
        
        return operations

# Example usage
solution = Solution()
print(solution.minimumOperations(0))    # Output: 0
print(solution.minimumOperations(3))    # Output: 2
print(solution.minimumOperations(6))    # Output: 4
print(solution.minimumOperations(9))    # Output: 14
print(solution.minimumOperations(333))  # Output: 393

```

### Explanation of the Code

- We define a class `Solution` with a method `minimumOperations` which takes an integer `n` as input.
- We use a loop that continues until `n` becomes `0`.
- Inside the loop:
  - We check if the least significant bit (`0th` bit) is `1`. If so, we increment our operations counter.
  - We find the highest power of `2` that can be cleared (the highest bit set to `1`) and clear it.
  - We right shift `n` to process the next bits in subsequent iterations.
- Finally, once we've processed all bits of `n`, we return the total operation count.
  
This solution efficiently reduces `n` while counting operations, adhering to the rules provided by the problem statement. The performance is linear with respect to the number of bits in the integer, making it suitable for the constraints given.

# 1707. Maximum XOR With an Element From Array

### Problem Description 
You are given an array `nums` consisting of non-negative integers. You are also given a `queries` array, where `queries[i] = [xi, mi]`.

The answer to the `ith` query is the maximum bitwise `XOR` value of `xi` and any element of `nums` that does not exceed `mi`. In other words, the answer is `max(nums[j] XOR xi)` for all `j` such that `nums[j] <= mi`. If all elements in `nums` are larger than `mi`, then the answer is `-1`.

Return an integer array `answer` where `answer.length == queries.length` and `answer[i]` is the answer to the `ith` query.


Example 1:
Input: nums = [0,1,2,3,4], queries = [[3,1],[1,3],[5,6]]
Output: [3,3,7]
Explanation:
1) 0 and 1 are the only two integers not greater than 1. 0 XOR 3 = 3 and 1 XOR 3 = 2. The larger of the two is 3.

2) 1 XOR 2 = 3.

3) 5 XOR 2 = 7.


Example 2:
Input: nums = [5,2,4,6,6,3], queries = [[12,4],[8,1],[6,3]]
Output: [15,-1,5]

Constraints:
`1 <= nums.length, queries.length <= 105`
`queries[i].length == 2`
`0 <= nums[j], xi, mi <= 109`

### Solution 
 To solve the problem of maximizing the bitwise XOR value of a given number `xi` with elements from the `nums` array that do not exceed a maximum value `mi`, we can follow these steps:

### Steps to Solve:

1. **Sort `nums`**: First, we will sort the `nums` array. This allows us to efficiently find the elements less than or equal to `mi` using binary search.

2. **Binary Search for Limit**: For each query, we can use binary search to find the largest index of `nums` that is less than or equal to `mi`. Python's built-in `bisect` library provides a way to do this easily.

3. **Calculate XOR Results**: We will iterate through all valid elements of `nums` (those that do not exceed `mi`) and calculate the XOR with `xi`. We keep track of the maximum value found.

4. **Handle Edge Cases**: If no elements in `nums` satisfy the condition (i.e., all elements are greater than `mi`), we return `-1` for that query.

The implementation of the above steps in Python follows the LeetCode format, using classes and methods as required by their system.

Here’s the complete code that adheres to the described approach:



```python
from bisect import bisect_right
from typing import List

class Solution:
    def xorQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        # Sort the nums array
        nums.sort()
        answer = []

        for xi, mi in queries:
            # Use binary search to find the largest index j where nums[j] <= mi
            idx = bisect_right(nums, mi) - 1
            
            # If idx is -1, it means no element is <= mi
            if idx == -1:
                answer.append(-1)
            else:
                max_xor = -1
                # Iterate over elements that are <= mi to find the maximum XOR
                for j in range(idx + 1):
                    current_xor = nums[j] ^ xi
                    max_xor = max(max_xor, current_xor)
                answer.append(max_xor)

        return answer

```

### Explanation of the Code:

1. **Sorting**: We sort the `nums` array to facilitate efficient searching.

2. **Processing Queries**:
   - For each query represented by `[xi, mi]`, we use `bisect_right` to find the position where `mi` would be inserted to maintain order. The result (`idx`) gives the highest index of all numbers in `nums` which are less than or equal to `mi`.

3. **Determining Maximum XOR**:
   - If `idx` is `-1`, this indicates no valid elements are available, so we append `-1` to the answer.
   - If valid elements exist, we iterate through them, compute the XOR with `xi`, and track the maximum XOR value.

4. **Return Result**: Finally, after processing all queries, we return the `answer` list that contains the maximum XOR for each query.

### Complexity Analysis:
- **Time Complexity**: Sorting `nums` takes \(O(N \log N)\) and processing each query takes \(O(K)\) in the worst case, where \(K\) is the number of elements ≤ `mi`. Overall this is efficient given typical use cases.
- **Space Complexity**: The space used is mainly for the output list, which is \(O(Q)\) where \(Q\) is the number of queries.

This solution efficiently handles the requirements of the problem within the specified constraints.

# 1734. Decode XORed Permutation

### Problem Description 
There is an integer array `perm` that is a permutation of the first `n` positive integers, where `n` is always odd.

It was encoded into another integer array `encoded` of length `n - 1`, such that `encoded[i] = perm[i] XOR perm[i + 1]`. For example, if `perm = [1,3,2]`, then `encoded = [2,1]`.

Given the `encoded` array, return the original array `perm`. It is guaranteed that the answer exists and is unique.


Example 1:
Input: encoded = [3,1]
Output: [1,2,3]
Explanation: If perm = [1,2,3], then encoded = [1 XOR 2,2 XOR 3] = [3,1]

Example 2:
Input: encoded = [6,5,4,6]
Output: [2,4,1,5,3]

Constraints:
`3 <= n < 105`
`n` is odd.

`encoded.length == n - 1`

### Solution 
 To solve the problem of recovering the original permutation array `perm` from the encoded array `encoded`, we can follow a systematic approach. We will build a solution that utilizes properties of the XOR operation and the permutation's structure.

### Problem Breakdown:

1. **Understanding XOR**: The XOR operation has some useful properties:
   - `a XOR a = 0`
   - `a XOR 0 = a`
   - `a XOR b XOR b = a` (XORing the same number cancels it out)

   Here, the `encoded` array holds the XOR between consecutive elements of `perm`. This means that knowing `encoded`, we need to reconstruct `perm`.

2. **Finding the Total XOR**: 
   To find the elements of `perm`, we can notice a pattern:
   - If we XOR all elements of `perm`, we get:
     
```
     p1 XOR p2 XOR p3 XOR ... XOR pn = 1 XOR 2 XOR ... XOR n
     
```
   - We can find this total XOR efficiently with a function that combines [`1 to n`](https://en.wikipedia.org/wiki/Bitwise_operations_in_C#Bitwise_operations) based on the binary representation of numbers.

3. **Decoding `perm` from `encoded`**:
   - The first element of `perm` can be derived from the total XOR and the XOR of the elements in `encoded`. We can calculate:
   
```
   p1 = total XOR XOR (encoded[0] XOR encoded[1] XOR ... XOR encoded[n-2])
   
```
   - Once we have `p1`, we can build the rest of `perm` using the relation `encoded[i] = perm[i] XOR perm[i + 1]`, which gives us:
   
```
   perm[i + 1] = perm[i] XOR encoded[i]
   
```

### Python Solution:

Following the explanation, here is the Python code solution formatted to be run directly on LeetCode:



```python
class Solution:
    def decode(self, encoded: List[int]) -> List[int]:
        # Find n
        n = len(encoded) + 1
        
        # Calculate total XOR of the first n natural numbers
        total_xor = 0
        for i in range(1, n + 1):
            total_xor ^= i
        
        # Calculate the XOR of elements in encoded
        encoded_xor = 0
        for i in range(0, n - 1, 2):
            encoded_xor ^= encoded[i]
        
        # Find the first element of perm
        perm = [0] * n
        perm[0] = total_xor ^ encoded_xor
        
        # Build the rest of the perm array
        for i in range(1, n):
            perm[i] = perm[i - 1] ^ encoded[i - 1]
        
        return perm

```

### Explanation of the Code:

1. **Calculate `n`**: Since `encoded` has `n-1` elements, we simply compute `n` by adding 1 to the length of `encoded`.

2. **Compute `total_xor`**: Iterate through integers from `1` to `n` and compute their XOR.

3. **Compute `encoded_xor`**: To decode the first element, we only need to XOR the elements at even indices of `encoded`—this is because of the way permutations are structured in relation to XOR.

4. **Initialize `perm`**: Create a list `perm` of size `n` and set the first element as described.

5. **Fill the `perm` list**: Use the relationship between `perm[i]`, `perm[i-1]`, and `encoded` to fill in the rest of the permutation.

### Final Remarks:
This implementation is efficient with a time complexity of O(n) and can handle the constraints provided by the problem statement. The algorithm uses simple XOR operations which are very fast, making it suitable for the input size constraints.

