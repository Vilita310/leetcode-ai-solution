# 1044. Longest Duplicate Substring

### Problem Description 
Given a string `s`, consider all duplicated substrings: (contiguous) substrings of s that occur 2 or more times. The occurrences may overlap.

Return any duplicated substring that has the longest possible length. If `s` does not have a duplicated substring, the answer is `""`.


Example 1:
Input: s = "banana"
Output: "ana"

Example 2:
Input: s = "abcd"
Output: ""

Constraints:
`2 <= s.length <= 3 * 104`
`s` consists of lowercase English letters.

### Solution 
 To solve the problem of finding the longest duplicated substring in a given string `s`, we can use the concept of binary search combined with a hashing technique (specifically, Rabin-Karp) to efficiently detect duplicates in substrings of varying lengths.

Here's a detailed explanation of the approach we'll take:

1. **Binary Search**: We will perform a binary search on the possible lengths of the substring. The search ranges from `1` to the length of the string minus one, since a duplicated substring must have at least a length of `1`.

2. **Hashing for Duplicates**: For each length `mid` (the midpoint in our binary search), we will generate all possible substrings of that length and use a set to keep track of them. We can hash these substrings to store them in our set. If we encounter a substring that is already in the set, it indicates that we found a duplicate.

3. **Return the Result**: If we find a duplicate substring of a certain length, we will try to find a longer one by moving our search to the right. If no duplicate exists for that length, we will search to the left.

4. **Final Output**: After completing the binary search, the longest duplicate substring found will be returned.

Here’s the Python code implementing the above logic:



```python
class Solution:
    def longestDupSubstring(self, s: str) -> str:
        def search(length):
            seen = set()
            # Rolling hash parameters
            base = 256
            mod = 2**63 - 1
            
            # Compute the hash of the first substring
            current_hash = 0
            for i in range(length):
                current_hash = (current_hash * base + ord(s[i])) % mod
            
            seen.add(current_hash)
            # Precompute base^length % mod for removing the leading character
            base_length = pow(base, length, mod)
            
            # Rolling hash for the string
            for i in range(length, len(s)):
                current_hash = (current_hash * base - ord(s[i-length]) * base_length + ord(s[i])) % mod
                
                if current_hash in seen:
                    return s[i-length+1:i+1]
                
                seen.add(current_hash)
            
            return None
        
        left, right = 1, len(s) - 1
        start_idx = -1
        
        while left <= right:
            mid = (left + right) // 2
            if search(mid):
                start_idx = search(mid)
                left = mid + 1
            else:
                right = mid - 1
        
        return start_idx if start_idx != -1 else ""

# Example Usage:
# solver = Solution()
# print(solver.longestDupSubstring("banana"))  # Output: "ana"
# print(solver.longestDupSubstring("abcd"))    # Output: ""

```

### Explanation of the Code:
1. **The `search` function**: This function checks for the existence of a duplicate substring of a given length using a rolling hash method. It maintains a set of hashes we’ve already seen and calculates the hash using the formula for a rolling hash, adjusting only for the characters that enter and leave the sliding window.

2. **Binary Search Logic**: We use two pointers (`left` and `right`) to define our current search for the length. We check if a duplicate substring exists for `mid`, and based on its existence, we adjust our search bounds.

3. **Returning the Result**: After the binary search concludes, we return the longest duplicated substring found. If no duplicates exist, we return an empty string.

This solution is efficient and operates within the acceptable boundaries for the problem constraints, working in O(n log n) time complexity due to the combination of binary search and hashing.

# 1163. Last Substring in Lexicographical Order

### Problem Description 
Given a string `s`, return the last substring of `s` in lexicographical order.


Example 1:
Input: s = "abab"
Output: "bab"
Explanation: The substrings are ["a", "ab", "aba", "abab", "b", "ba", "bab"]. The lexicographically maximum substring is "bab".


Example 2:
Input: s = "leetcode"
Output: "tcode"

Constraints:
`1 <= s.length <= 4 * 105`
`s` contains only lowercase English letters.

### Solution 
 To solve the problem of finding the last substring of a string `s` in lexicographical order, we can utilize a method that efficiently calculates the lexicographically largest substring by iterating through the string and keeping track of the relevant substrings.

### Solution Explanation:

1. **Initial Observation**: The last substring in lexicographical order must start at the position of the character that is lexicographically larger than all the characters that come before it. This is important to note because it means we can prune the search space significantly.

2. **Iterative Approach**: We will iterate through the string while keeping track of the maximum substring seen so far. As we progress through the string, we will use two pointers: one to start exploring from the current character and the other to keep track of the maximum substring.

3. **Comparison Strategy**: When we find a character that is larger than the current maximum substring, we update our maximum substring to start from that character. We will also compare substrings character by character whenever we find a character that potentially leads to a larger substring.

4. **Efficiency**: This approach ensures that we only need to traverse the string a single time, resulting in a time complexity of O(n), where n is the length of the string. Given the constraints, this is optimal.

Here's the Python code implementing the above approach:



```python
class Solution:
    def lastSubstring(self, s: str) -> str:
        # Initialize the last max substring to the first character
        last_max_substring = s[-1]
        current_substring = ""
        
        # Traverse the string from the second last character to the beginning
        for i in range(len(s) - 2, -1, -1):
            # If the character is greater than or equal to the first character of our last max substring
            if s[i] >= last_max_substring[0]:
                # Compare with current max substring
                # Build the new substring starting from s[i]
                current_substring = s[i] + current_substring
                
                # Check if the newly formed substring is greater than the last max substring
                if current_substring > last_max_substring:
                    last_max_substring = current_substring
        
        return last_max_substring

```

### How the Code Works:

- We initialize `last_max_substring` with the last character of the string. This character is the initial assumption for the maximum substring.
- We iterate backward from the second last character to the first.
- For each character:
  - If it is greater than or equal to the first character of `last_max_substring`, we build the `current_substring`.
  - We then compare the `current_substring` with `last_max_substring`. If it's greater, we update `last_max_substring`.
- Finally, we return `last_max_substring`, which will be the desired result.

### Complexity Analysis:
- **Time Complexity**: O(n), where n is the length of the string, since we make a single pass through the string.
- **Space Complexity**: O(1), as we're using a constant amount of extra space for variables (not storing substrings).

This code can be directly run on LeetCode as it adheres to the standard submission format, including defining the method inside a `Solution` class and using appropriate method signatures.

# 1698. Number of Distinct Substrings in a String

### Problem Description 
Given a string `s`, return the number of distinct substrings of `s`.

A substring of a string is obtained by deleting any number of characters (possibly zero) from the front of the string and any number (possibly zero) from the back of the string.


Example 1:
Input: s = "aabbaba"
Output: 21
Explanation: The set of distinct strings is ["a","b","aa","bb","ab","ba","aab","abb","bab","bba","aba","aabb","abba","bbab","baba","aabba","abbab","bbaba","aabbab","abbaba","aabbaba"]

Example 2:
Input: s = "abcdefg"
Output: 28

Constraints:
`1 <= s.length <= 500`
`s` consists of lowercase English letters.

Follow up: Can you solve this problem in `O(n)` time complexity?

### Solution 
 To solve the problem of finding the number of distinct substrings of a given string `s`, we can utilize a data structure called a **Suffix Automaton**. This allows us to efficiently track and count distinct substrings in linear time. 

### Explanation:

1. **Suffix Automaton**: 
   A suffix automaton is a state machine that recognizes the set of all substrings of a given string. For a string of length `n`, a suffix automaton can be built in `O(n)` time and space.

2. **Building the Automaton**:
   - Each state in the Suffix Automaton represents a set of substrings that can be formed.
   - We maintain a transition from one state to another based on the characters in the string.
   - We determine the lengths of substrings corresponding to these states to facilitate counting distinct substrings.

3. **Counting Distinct Substrings**:
   - To count the number of distinct substrings, we can compute the contribution of each state by measuring the length difference between adjacent states.
   - For a state representing substrings of length `l`, if `last` is the length of the longest suffix linked to this state, then the number of new distinct substrings contributed by this state is given by `length[state] - length[last]`.

### Python Implementation:

Here’s the complete Python code to implement the above approach following the LeetCode solution format:



```python
class Solution:
    def distinctSubseqII(self, s: str) -> int:
        n = len(s)
        # Suffix automaton implementation
        # Create a list of structures for states
        states = [{}]  # transitions
        length = [0]  # lengths of the longest suffix
        linked = [-1]  # links to the suffix states
        last = 0  # Last state index
        size = 1  # Number of states

        for char in s:
            # Create a new state for the current character
            cur = size
            size += 1
            length.append(length[last] + 1)
            linked.append(0)
            p = last
            
            while p != -1 and char not in states[p]:
                states[p][char] = cur
                p = linked[p]
            
            if p == -1:
                linked[cur] = 0
            else:
                q = states[p][char]
                if length[p] + 1 == length[q]:
                    linked[cur] = q
                else:
                    # Create a clone of `q`
                    clone = size
                    size += 1
                    states.append(states[q].copy())
                    length.append(length[q])
                    linked.append(linked[q])
                    while p != -1 and states[p].get(char, -1) == q:
                        states[p][char] = clone
                        p = linked[p]
                    linked[q] = clone
                    linked[cur] = clone
            
            last = cur
        
        # Count the number of distinct substrings
        distinct_count = 0
        for i in range(1, size):
            distinct_count += length[i] - length[linked[i]]
        
        return distinct_count

# Example usage
sol = Solution()
print(sol.distinctSubseqII("aabbaba"))  # Output: 21
print(sol.distinctSubseqII("abcdefg"))  # Output: 28

```

### Explanation of the Code:

- We define a `Solution` class with a method `distinctSubseqII`, which takes the string `s` as input.
- We initialize the automaton states and necessary variables to keep track of the lengths and links of states.
- For each character in the input string, we update our suffix automaton by creating states, making transitions, and linking correctly.
- Finally, we count the substrings based on the lengths of states and their respective links.

This solution efficiently counts the number of distinct substrings in `O(n)` time, adhering to the problem's constraints and expectations.

# 1754. Largest Merge Of Two Strings

### Problem Description 
You are given two strings `word1` and `word2`. You want to construct a string `merge` in the following way: while either `word1` or `word2` are non-empty, choose one of the following options:
If `word1` is non-empty, append the first character in `word1` to `merge` and delete it from `word1`.

	
For example, if `word1 = "abc" `and `merge = "dv"`, then after choosing this operation, `word1 = "bc"` and `merge = "dva"`.

If `word2` is non-empty, append the first character in `word2` to `merge` and delete it from `word2`.

	
For example, if `word2 = "abc" `and `merge = ""`, then after choosing this operation, `word2 = "bc"` and `merge = "a"`.

Return the lexicographically largest `merge` you can construct.

A string `a` is lexicographically larger than a string `b` (of the same length) if in the first position where `a` and `b` differ, `a` has a character strictly larger than the corresponding character in `b`. For example, `"abcd"` is lexicographically larger than `"abcc"` because the first position they differ is at the fourth character, and `d` is greater than `c`.


Example 1:
Input: word1 = "cabaa", word2 = "bcaaa"
Output: "cbcabaaaaa"
Explanation: One way to get the lexicographically largest merge is:
- Take from word1: merge = "c", word1 = "abaa", word2 = "bcaaa"
- Take from word2: merge = "cb", word1 = "abaa", word2 = "caaa"
- Take from word2: merge = "cbc", word1 = "abaa", word2 = "aaa"
- Take from word1: merge = "cbca", word1 = "baa", word2 = "aaa"
- Take from word1: merge = "cbcab", word1 = "aa", word2 = "aaa"
- Append the remaining 5 a's from word1 and word2 at the end of merge.


Example 2:
Input: word1 = "abcabc", word2 = "abdcaba"
Output: "abdcabcabcaba"

Constraints:
`1 <= word1.length, word2.length <= 3000`
`word1` and `word2` consist only of lowercase English letters.

### Solution 
 To solve the problem of constructing the lexicographically largest `merge` from two given strings `word1` and `word2`, we can use a greedy approach. The primary idea is to always compare the starting segments of both strings and append the larger character to the `merge` string. We do this until both strings are completely processed.

Here's a step-by-step breakdown of the approach we'll use:

1. **Initializing Pointers**: We'll maintain indices for both strings (`i` for `word1` and `j` for `word2`). We also need to initialize an empty list to collect characters for `merge`.

2. **Iterative Merging**: We'll iterate until we have exhausted both strings. In each iteration:
   - Compare the remaining parts of `word1` and `word2` starting from the current indices `i` and `j`. Since we are interested in lexicographical order, we can compare the substrings directly.
   - If `word1[i:]` is greater than `word2[j:]`, we append `word1[i]` to `merge`, otherwise we append `word2[j]`.
   - Move the respective pointer (`i` or `j`) forward by one position.

3. **Appending Remaining Characters**: After the loop completes, if there are any characters left in either string, we'll append the remaining characters to `merge`.

4. **Returning the Result**: Finally, we'll convert our list of characters back into a string and return it.

Here is the complete Python code that follows the LeetCode solution format:



```python
class Solution:
    def largestMerge(self, word1: str, word2: str) -> str:
        i, j = 0, 0
        merge = []
        
        while i < len(word1) and j < len(word2):
            # Compare the substrings of word1 and word2
            if word1[i:] > word2[j:]:
                # If word1's suffix is larger, take from word1
                merge.append(word1[i])
                i += 1
            else:
                # Otherwise, take from word2
                merge.append(word2[j])
                j += 1
        
        # After one of the strings is exhausted, append the rest of the other string
        merge.append(word1[i:])
        merge.append(word2[j:])
        
        # Join the list into a final string and return
        return ''.join(merge)

# Example usage:
# sol = Solution()
# print(sol.largestMerge("cabaa", "bcaaa"))  # Output: "cbcabaaaaa"
# print(sol.largestMerge("abcabc", "abdcaba"))  # Output: "abdcabcabcaba"

```

### Explanation of the Code:
- We create a class `Solution` that contains the method `largestMerge`.
- Indices `i` and `j` track our current position in `word1` and `word2` respectively.
- The `merge` list collects characters that will form our output string.
- The comparison `word1[i:] > word2[j:]` uses Python's implicit string comparison, which compares the substrings lexicographically.
- After processing, we append any remaining characters from either string to `merge`.
- Finally, we return the result as a string by joining the elements of the `merge` list.

This code can be executed directly on LeetCode and meets the constraints provided in the problem statement.

