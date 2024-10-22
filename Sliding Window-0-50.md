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
 To solve the problem of finding the length of the longest substring without repeating characters, we can use a sliding window technique. This method allows us to efficiently traverse the string while keeping track of the characters we have encountered and their positions.

### Approach:
1. **Initialization**: We will use a set to store characters currently in our window and two pointers (`left` and `right`) to represent the current window's boundaries.
2. **Extend the Window**: We will move the right pointer to expand the window, adding the character to the set. If the character is already in the set, it signifies a repeat.
3. **Shrink the Window**: When a repeat is found, we will increment the left pointer until the character can be added to the set again. We will also remove characters from the set as we move the left pointer.
4. **Update the Result**: For each valid window (without repeating characters), we will update our maximum length if the current window size is greater than our previously recorded maximum.
5. **Edge Cases**: We also account for an empty string by immediately returning a length of 0.

This technique runs in O(n) time complexity, where n is the length of the string, since each character is processed at most twice (once by the right pointer and once by the left pointer).

Below is the implementation of the solution in Python, formatted for the LeetCode submission:



```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # Initialize a set to keep track of characters in the current window
        char_set = set()
        left = 0
        max_length = 0
        
        # Iterate over the string with the right pointer
        for right in range(len(s)):
            # If we encounter a duplicate character, shrink the window from the left
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1
            
            # Add the current character to the set
            char_set.add(s[right])
            # Update max_length if the current window is greater
            max_length = max(max_length, right - left + 1)
        
        return max_length

```

### Explanation of the Code:
- We create a class `Solution` and define the method `lengthOfLongestSubstring`.
- We initialize a set `char_set` to store unique characters in the window. `left` is the start of the window, and `max_length` keeps track of the maximum length of unique substrings found.
- We loop through each character using the `right` pointer.
- If a character at position `right` has already been seen (is in `char_set`), we start incrementing the `left` pointer, removing characters from the set, so we can skip the duplicates.
- Finally, we update `max_length` if our current window's size (`right - left + 1`) is larger than the previously recorded maximum.
- The function returns `max_length`.

This solution is efficient and directly follows the method described, ensuring it effectively finds the desired substring length as required by the problem statement.

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
 To solve the problem of finding the minimum window in string `s` that contains all characters of string `t`, we can utilize a sliding window approach combined with a character frequency count. The main idea is to expand the window to include characters from `t` and then contract it to find the minimal length that still contains all characters from `t`.

Here is the complete implementation in Python, presented in a format suitable for LeetCode:



```python
from collections import Counter, defaultdict

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not t or not s:
            return ""
        
        # Dictionary to keep count of all characters in t
        dict_t = Counter(t)
        required = len(dict_t)  # Number of unique characters in t that need to be present in the window
        
        # Left and right pointer for the window, and a formed counter
        l, r = 0, 0
        formed = 0
        window_counts = defaultdict(int)
        
        # Answer tuple (window length, left, right) - initialized as a tuple with infinity
        ans = float("inf"), None, None
        
        # Expand the right pointer to form a valid window
        while r < len(s):
            character = s[r]
            window_counts[character] += 1
            
            # Check if the current character added is part of t and if so
            # if the current count matches the required count
            if character in dict_t and window_counts[character] == dict_t[character]:
                formed += 1
            
            # Try to contract from the left to find the smallest window
            while l <= r and formed == required:
                character = s[l]
                
                # Update the answer if we found a smaller window
                if r - l + 1 < ans[0]:
                    ans = (r - l + 1, l, r)
                
                # Remove the character at the left pointer from the window
                window_counts[character] -= 1
                if character in dict_t and window_counts[character] < dict_t[character]:
                    formed -= 1
                
                # Move the left pointer to the right
                l += 1
            
            # Keep expanding the window by moving the right pointer
            r += 1
        
        # Return the final answer
        return "" if ans[0] == float("inf") else s[ans[1]: ans[2] + 1]

```

### Explanation of the Code:

1. **Count the Characters in `t`**:
   - We create a dictionary (`dict_t`) using `Counter` to keep count of how many times each character appears in `t`.
   - We also track how many unique characters are required (`required`).

2. **Sliding Window Initialization**:
   - Initialize two pointers `l` (left) and `r` (right) that will help us create a window.
   - We also initialize a dictionary (`window_counts`) to keep a count of characters in the current window and a variable `formed` to track how many unique characters from `t` are currently in the window in their required frequency.

3. **Expanding the Right Pointer**:
   - Increase the right pointer `r` iteratively to add characters to the current window.
   - Update `window_counts` and check if we need to increment `formed`.

4. **Contracting the Left Pointer**:
   - As soon as we have a valid window (where `formed` equals `required`), we attempt to move the left pointer `l` to reduce the size of the window, while still maintaining all required characters.
   - Update the answer tuple if a smaller window is found.

5. **Return the Result**:
   - If no valid window is found, we return an empty string. Otherwise, return the substring of `s` that corresponds to the minimal window found.

### Time Complexity:
The time complexity of this approach is `O(n)`, where `n` is the length of string `s`. Each character is processed at most twice — once by the `r` pointer and once by the `l` pointer. The space complexity is `O(m)`, where `m` is the number of unique characters in `t`. This is because we're using additional space to store the character counts.

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
 To solve the problem of finding the length of the longest substring that contains at most two distinct characters, we can use the sliding window technique together with a hash map (or dictionary) to keep track of the characters in the substring and their counts. This approach allows us to efficiently expand and contract the window until we satisfy the constraint of having at most two distinct characters.

Here’s how the solution works step-by-step:

1. **Initialize Variables**:
   - Use two pointers (`left` and `right`) to represent the current window's start and end.
   - Create a dictionary to count the occurrences of characters in the current window.
   - Maintain a variable `max_length` to keep track of the maximum length found.

2. **Expand the Window**:
   - Move the `right` pointer to expand the window character by character.
   - Update the character count in the dictionary.

3. **Contract the Window**:
   - If the number of distinct characters exceeds two (checking the dictionary's length), increment the `left` pointer to shrink the window from the left. Update the character counts accordingly and remove characters from the dictionary if their count drops to zero.

4. **Calculate Maximum Length**:
   - After adjusting the window, compute the length of the current valid window and update `max_length` if it's larger than the previous maximum.

5. **Return Result**:
   - After processing the entire string, return `max_length`.

Now, let's implement this approach in Python as per the LeetCode solution format:



```python
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        # Dictionary to store the last occurrence of each character
        char_count = {}
        left = 0  # Left pointer of the sliding window
        max_length = 0  # Max length of substring found
        
        for right in range(len(s)):
            # Add the current character to the dictionary
            char_count[s[right]] = char_count.get(s[right], 0) + 1
            
            # If there are more than 2 distinct characters, move the left pointer
            while len(char_count) > 2:
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]
                left += 1
            
            # Calculate the maximum length
            max_length = max(max_length, right - left + 1)
        
        return max_length

```

### Explanation of the Code:

- We define a `Solution` class with a method `lengthOfLongestSubstringTwoDistinct` that takes a string `s` as input.
- We initialize an empty dictionary `char_count` to keep track of the count of characters in the current substring, a pointer `left` to represent the beginning of the substring, and `max_length` to track the maximum length of valid substrings encountered.
- We iterate over the string using a `right` pointer, expanding the window:
  - For each character at the `right` pointer, we update its count in the `char_count` dictionary.
  - If the number of distinct characters in the dictionary exceeds 2, we enter a while loop to contract the window from the left:
    - We decrement the count of the character at the `left` pointer and remove it from the dictionary if its count reaches zero.
  - After adjusting, we check the length of the current valid substring (`right - left + 1`) and update `max_length` if it is greater than the previously recorded maximum length.
- Finally, the method returns `max_length` as the answer.

This implementation is efficient with a time complexity of O(n), where n is the length of the string, since each character is processed at most twice (once by the right pointer and once by the left pointer) and uses O(1) additional space for the character count.

# 239. Sliding Window Maximum

### Problem Description 
You are given an array of integers `nums`, there is a sliding window of size `k` which is moving from the very left of the array to the very right. You can only see the `k` numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window.


Example 1:
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

Example 2:
Input: nums = [1], k = 1
Output: [1]

Example 3:
Input: nums = [1,-1], k = 1
Output: [1,-1]

Example 4:
Input: nums = [9,11], k = 2
Output: [11]

Example 5:
Input: nums = [4,-2], k = 2
Output: [4]

Constraints:
`1 <= nums.length <= 105`
`-104 <= nums[i] <= 104`
`1 <= k <= nums.length`

### Solution 
 To solve the problem of finding the maximum in a sliding window of size `k` across an array of integers, we can use a double-ended queue (deque) to efficiently track the indices of the maximum values in the current window. This allows us to achieve an optimal time complexity of O(n), where n is the number of elements in the input array.

### Explanation of the Approach:

1. **Using a Deque**: A deque will help us keep track of the indices of the elements in the `nums` array such that the elements are in decreasing order. The front of the deque will always contain the index of the maximum value in the current sliding window.

2. **Maintaining the Window**:
   - As we iterate through the array, we will add the current index to the deque.
   - Before adding an index, we will remove indices from the back of the deque if the corresponding values are less than or equal to the current value, as they cannot be the maximum for future windows.
   - We will also pop indices from the front if they are out of the bounds of the current window (i.e., if the index is less than `i - k + 1` where `i` is the current index).

3. **Recording Maximums**: Once we have processed at least `k` elements (when the current index `i` is greater than or equal to `k - 1`), we can start recording the maximum values, which will always be at the front of the deque.

### Python Code:

Here's how the solution would be implemented in Python according to the LeetCode format:



```python
from collections import deque
from typing import List

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums or k == 0:
            return []

        n = len(nums)
        max_values = []
        deq = deque()

        for i in range(n):
            # Remove elements not in the sliding window
            if deq and deq[0] < i - k + 1:
                deq.popleft()
            
            # Remove elements that are smaller than the current element
            while deq and nums[deq[-1]] <= nums[i]:
                deq.pop()
            
            # Add the current element's index to the deque
            deq.append(i)

            # If we have hit the window size of k, we add the maximum to our result
            if i >= k - 1:
                max_values.append(nums[deq[0]])

        return max_values

```

### Key Points:
- We use a deque to efficiently manage the indices of potential maximum values in the current window.
- The solution only iterates through the list (O(n)) and maintains the deque of at most `k` elements, ensuring efficient inserts and removals.
- The spatial complexity is O(k) in the worst case due to the deque storage.

### Example Execution:
- For the input `nums = [1, 3, -1, -3, 5, 3, 6, 7]` and `k = 3`, the output will be `[3, 3, 5, 5, 6, 7]` by maintaining the sliding window and collecting the maximums as described.

This approach efficiently handles the constraints provided, ensuring that we can process even large arrays up to the size of 100,000 in a reasonable time.

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
 To solve the problem of finding the length of the longest substring with at most `k` distinct characters, we can utilize the sliding window technique. This technique enables us to efficiently maintain a window of characters while iterating through the string, counting distinct characters and adjusting the window as necessary.

Overall Approach:
1. We'll maintain a sliding window, defined by two pointers (`left` and `right`), that represents the current substring under consideration.
2. We will use a dictionary to keep track of the frequency of each character within this window.
3. As we expand our window by moving the `right` pointer, we will add the character to our dictionary and increase its count.
4. If at any point the number of distinct characters in the window exceeds `k`, we will move the `left` pointer to shrink the window until the condition is met again (i.e., until we are back to at most `k` distinct characters).
5. Throughout the process, we will keep track of the maximum length of valid substrings.

Here's how this can be implemented in Python:



```python
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        if k == 0:  # If k is 0, there can be no valid substring
            return 0
        
        left = 0
        right = 0
        max_length = 0
        char_count = {}  # Dictionary to count occurrences of characters in the current window
        
        while right < len(s):
            # Add the current character to the map
            char_count[s[right]] = char_count.get(s[right], 0) + 1
            
            # If the number of distinct characters exceeds k, shrink the window
            while len(char_count) > k:
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]  # Remove the character from the map completely
                left += 1  # Shrink the window from the left
            
            # Calculate the max length of valid substring
            max_length = max(max_length, right - left + 1)
            right += 1  # Expand the window from the right
            
        return max_length

```

### Explanation of the Code:
1. **Initialization**:
   - `left` and `right` are initialized to zero. They represent the boundaries of the sliding window.
   - `max_length` is initialized to zero to keep track of the longest valid substring found.
   - `char_count` is a dictionary used to count the occurrences of each character in the current substring.

2. **Expanding the Window**:
   - We iterate through the string using the `right` pointer and update our character count in `char_count`.
   - Whenever a new character is added, we check if the number of distinct characters exceeds `k`. If it does, we enter another while loop to shrink the window from the left.

3. **Shrinking the Window**:
   - Inside the while loop, we decrement the count of the character at position `left`. If any character count drops to zero, we remove it from `char_count`.
   - We increment `left` until we once again have at most `k` distinct characters.

4. **Updating Maximum Length**:
   - After adjusting the window for any violating conditions, we compute the current length of the valid substring as `(right - left + 1)` and update `max_length` if this length is greater than the previous maximum.

5. **Return Result**: Finally, we return `max_length`.

This code efficiently computes the result with a time complexity of O(n) since each pointer (`left` and `right`) traverses the string at most once, and the space complexity is O(min(n, k)) for the character count dictionary.

# 395. Longest Substring with At Least K Repeating Characters

### Problem Description 
Given a string `s` and an integer `k`, return the length of the longest substring of `s` such that the frequency of each character in this substring is greater than or equal to `k`.


Example 1:
Input: s = "aaabb", k = 3
Output: 3
Explanation: The longest substring is "aaa", as 'a' is repeated 3 times.


Example 2:
Input: s = "ababbc", k = 2
Output: 5
Explanation: The longest substring is "ababb", as 'a' is repeated 2 times and 'b' is repeated 3 times.


Constraints:
`1 <= s.length <= 104`
`s` consists of only lowercase English letters.

`1 <= k <= 105`

### Solution 
 To solve the problem of finding the length of the longest substring where the frequency of each character is greater than or equal to `k`, we can use a divide and conquer approach, which efficiently checks different substrings of `s`. The key idea is to recursively split the string at characters that do not meet the frequency requirement.

Here's the detailed breakdown of the approach:

1. **Base Case**: If the string is empty, return `0`. If the value of `k` is greater than the length of the string, it's impossible to have any valid substring, so return `0` as well.

2. **Character Frequency Counting**: We first count the frequency of each character in the string using a dictionary.

3. **Identify Invalid Characters**: We then identify characters that have a frequency less than `k`. These characters will act as dividing points for our recursive calls.

4. **Recursive Splitting**: For each invalid character found, we split the string into substrings using these invalid characters as delimiters. We then recursively compute the longest valid substring length for each split substring.

5. **Combine Results**: The result for the current substring will be the maximum of the lengths returned from the recursive calls.

By implementing these steps, we can ensure to cover all possibilities while remaining efficient.

Here is the Python code that achieves this, formatted according to the LeetCode solution style:



```python
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        # Function to get frequency of each character in the string
        def get_frequency(s):
            frequency = {}
            for char in s:
                frequency[char] = frequency.get(char, 0) + 1
            return frequency
        
        # If the string is empty or k is greater than the length of the string
        if len(s) == 0 or k > len(s):
            return 0
        
        # Get frequency map
        frequency = get_frequency(s)
        
        # Check for invalid characters
        for char in frequency:
            if frequency[char] < k:
                # Split string at invalid character and recurse for both substrings
                return max(self.longestSubstring(substring, k) for substring in s.split(char))
        
        # If every character in the string meets the criteria
        return len(s)

# Example Usage
# sol = Solution()
# print(sol.longestSubstring("aaabb", 3))  # Output: 3
# print(sol.longestSubstring("ababbc", 2)) # Output: 5

```

### Explanation of the Code:
- **Class Structure**: The code is encapsulated within a class `Solution`, which is standard for LeetCode submissions.
- **Frequency Calculation**: We use a dictionary to summarize the frequency of each character in the string `s`.
- **Recursive Splitting Logic**: We check each character's frequency, and if any character appears less than `k` times, we split the string around that character. We compute the longest valid substring for each part and return the maximum length found.
- **Final Condition**: If all characters satisfy the frequency condition, we simply return the length of the entire string.

This solution is efficient and works within the constraints provided, ensuring optimal performance for the problem.

# 424. Longest Repeating Character Replacement

### Problem Description 
Given a string `s` that consists of only uppercase English letters, you can perform at most `k` operations on that string.

In one operation, you can choose any character of the string and change it to any other uppercase English character.

Find the length of the longest sub-string containing all repeating letters you can get after performing the above operations.

Note:
Both the string's length and k will not exceed 104.


Example 1:
Input:
s = "ABAB", k = 2
Output:
4
Explanation:
Replace the two 'A's with two 'B's or vice versa.


Example 2:
Input:
s = "AABABBA", k = 1
Output:
4
Explanation:
Replace the one 'A' in the middle with 'B' and form "AABBBBA".

The substring "BBBB" has the longest repeating letters, which is 4.

### Solution 
 To solve the problem of finding the length of the longest substring with all repeating letters after performing at most `k` operations, we can apply the sliding window technique. This algorithm is efficient and will allow us to manipulate the window of characters within the string while keeping track of character counts and the most frequently occurring character.

### Explanation of the Approach:

1. **Sliding Window**: We will maintain a window defined by two pointers — the left pointer (`l`) and the right pointer (`r`). The right pointer expands the window by moving to the right, while the left pointer will contract the window when necessary.

2. **Character Count**: We'll keep track of the count of each character in the current window using a dictionary or a list (as there are only 26 uppercase letters).

3. **Window Validation**: For a window to be valid, the number of characters that need changing to make all characters the same within that window must not exceed `k`. We can calculate this by:
   - Finding the total length of the current window.
   - Noting the frequency of the most common character in that window. For a valid window, the number of changes needed is `window_length - max_frequency`.

4. **Adjusting the Window**: If the number of changes exceeds `k`, we will increment the left pointer to shrink the window from the left until the condition is satisfied again.

5. **Length Calculation**: Throughout the process, we will keep track of the maximum length of valid windows found.

Let's implement this in the required LeetCode solution format below:



```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        left = 0
        max_length = 0
        max_count = 0
        count = [0] * 26  # There are 26 uppercase English letters
        
        for right in range(len(s)):
            # Increment the count of the current character
            count[ord(s[right]) - ord('A')] += 1
            
            # Update the count of the maximum frequency character in the window
            max_count = max(max_count, count[ord(s[right]) - ord('A')])
            
            # Check if we need to shrink the window
            while (right - left + 1) - max_count > k:
                count[ord(s[left]) - ord('A')] -= 1
                left += 1
            
            # Calculate the maximum length of valid window
            max_length = max(max_length, right - left + 1)
        
        return max_length

```

### Code Explanation:

- **Initialization**: We set the left pointer to 0, initialize maximum length and maximum character count to zero, and create a count array to track the frequency of characters within the window.

- **Looping Through Characters**: For each character in the string:
  - We update the count of the current character.
  - Update the `max_count` variable to store the frequency of the most common character in the window.
  
- **Checking Window Validity**: If the number of characters that need to be changed (total window size minus the frequency of the most common character) exceeds `k`, we increment the left pointer and adjust the count for the character that is being excluded from the window.

- **Updating Maximum Length**: After adjusting the window, we compute the current window size and update the `max_length` if the new window size is greater.

This implementation runs in O(n) time complexity where n is the length of the string, making it efficient given the constraints. The space complexity is O(1) since the count array size is constant (26).

# 480. Sliding Window Median

### Problem Description 
Median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value. So the median is the mean of the two middle value.


Examples:
`[2,3,4]` , the median is `3`
`[2,3]`, the median is `(2 + 3) / 2 = 2.5`
Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Your job is to output the median array for each window in the original array.

For example,
Given nums = `[1,3,-1,-3,5,3,6,7]`, and k = 3.

Window position                Median
---------------               -----
[1  3  -1] -3  5  3  6  7       1
 1 [3  -1  -3] 5  3  6  7       -1
 1  3 [-1  -3  5] 3  6  7       -1
 1  3  -1 [-3  5  3] 6  7       3
 1  3  -1  -3 [5  3  6] 7       5
 1  3  -1  -3  5 [3  6  7]      6
Therefore, return the median sliding window as `[1,-1,-1,3,5,6]`.

Note: 
You may assume `k` is always valid, ie: `k` is always smaller than input array's size for non-empty array.

Answers within `10^-5` of the actual value will be accepted as correct.

### Solution 
 To solve the problem of finding the median in a sliding window of size `k` for a given array of integers, we can utilize a combination of a min-heap and a max-heap. This approach allows us to efficiently manage the elements in the window and compute the median in logarithmic time.

Here is a clear step-by-step breakdown of the solution:

### Steps to Solve the Problem

1. **Understanding Median Calculation**:
   - If the size of the window (list) is odd, the median is the middle element.
   - If it's even, the median is the average of the two middle elements.

2. **Data Structure**:
   - A max-heap (to store the lower half of numbers) and a min-heap (to store the upper half).
   - The max-heap will help us quickly access the largest of the lower half (which is necessary for median calculation).
   - The min-heap will help us access the smallest of the upper half.

3. **Algorithm**:
   - Iterate over each element in the array while maintaining the sliding window.
   - Add the new element to the appropriate heap (max-heap or min-heap).
   - Balance the heaps when necessary to ensure that their sizes are correct (the max-heap can have at most one more element than the min-heap).
   - After each insertion (once we have filled the window), compute the median based on the sizes of the heaps.

4. **Output**:
   - Collect all the medians into a list and return it.

Here’s the complete solution code that you can run on LeetCode:



```python
import heapq  # For using heaps

class Solution:
    def medianSlidingWindow(self, nums: list[int], k: int) -> list[float]:
        if not nums:
            return []

        max_heap = []  # max heap for the lower half
        min_heap = []  # min heap for the upper half
        medians = []

        def add_num(num):
            # Add to max_heap first
            heapq.heappush(max_heap, -num)  # Push negated value to simulate max heap
            
            # Ensure the largest from max_heap is <= the smallest from min_heap
            if max_heap and min_heap and (-max_heap[0] > min_heap[0]):
                to_min = -heapq.heappop(max_heap)
                heapq.heappush(min_heap, to_min)

            # Balance sizes: max_heap can only have at most one more element than min_heap
            if len(max_heap) > len(min_heap) + 1:
                to_min = -heapq.heappop(max_heap)
                heapq.heappush(min_heap, to_min)
            elif len(min_heap) > len(max_heap):
                to_max = heapq.heappop(min_heap)
                heapq.heappush(max_heap, -to_max)

        def find_median():
            if len(max_heap) > len(min_heap):
                return float(-max_heap[0])  # Max heap has the extra element
            return (-max_heap[0] + min_heap[0]) / 2.0

        for i in range(len(nums)):
            add_num(nums[i])
            # Once we hit window size k, start to calculate the median and slide the window
            if i >= k - 1:
                medians.append(find_median())
                # Remove the element going out of the window
                old_num = nums[i - k + 1]
                if old_num <= -max_heap[0]:  # It was in the max heap
                    max_heap.remove(-old_num)
                    heapq.heapify(max_heap)  # Re-heapify after removal
                else:  # It was in the min heap
                    min_heap.remove(old_num)
                    heapq.heapify(min_heap)  # Re-heapify after removal

        return medians

```

### Explanation of the Code:
- We maintain two heaps to manage the sliding window dynamically.
- The `add_num` function manages the insertion and balancing of both heaps.
- The `find_median` function calculates the median based on the sizes and top elements of the heaps.
- As we iterate through `nums`, once we reach at least `k` elements, we compute the median and remove the oldest element in the sliding window from the appropriate heap.
- The `heapq` library allows efficient heap operations, with `O(log n)` complexity for insertions and deletions.

This algorithm efficiently maintains the window and calculates the median, resulting in optimal performance suitable for larger input sizes.

# 567. Permutation in String

### Problem Description 
Given two strings s1 and s2, write a function to return true if s2 contains the permutation of s1. In other words, one of the first string's permutations is the substring of the second string.


Example 1:
Input: s1 = "ab" s2 = "eidbaooo"
Output: True
Explanation: s2 contains one permutation of s1 ("ba").


Example 2:
Input:s1= "ab" s2 = "eidboaoo"
Output: False

Constraints:
The input strings only contain lower case letters.

The length of both given strings is in range [1, 10,000].

### Solution 
 To solve the problem of determining whether one string (`s2`) contains a permutation of another string (`s1`), we can use the sliding window technique along with character frequency counting. Here is a detailed explanation of the approach, followed by the implementation in the LeetCode format.

### Detailed Explanation

1. **Character Frequency Count**: We'll need to count how often each character appears in `s1`. This will help us identify if a substring in `s2` is a permutation of `s1`.

2. **Sliding Window**: We'll use a sliding window of size equal to `s1` on string `s2` to check each substring of length `len(s1)` in `s2`. 

3. **Comparison of Counts**: For each window, we will count the characters and compare it to the count of characters in `s1`. If the counts match, it means we found a permutation of `s1` in `s2`.

4. **Efficiency**: The use of two arrays (for counts) allows us to efficiently manage and compare the counts of characters as we slide the window in `s2`. This leads to an O(n) time complexity for strings of length `n`.

### Implementation

Here is the implementation following the LeetCode format:



```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        from collections import Counter
        
        len_s1 = len(s1)
        len_s2 = len(s2)
        
        if len_s1 > len_s2:
            return False
        
        # Count characters in s1
        s1_count = Counter(s1)
        # Initial count for the first window in s2
        s2_count = Counter(s2[:len_s1])
        
        # Check if the initial window matches
        if s1_count == s2_count:
            return True
        
        # Start sliding the window
        for i in range(len_s1, len_s2):
            # Add the next character in the window
            s2_count[s2[i]] += 1
            # Remove the character that is now out of the window
            s2_count[s2[i - len_s1]] -= 1
            
            # Remove the count from the Counter if it becomes 0
            if s2_count[s2[i - len_s1]] == 0:
                del s2_count[s2[i - len_s1]]
            
            # Compare the counts
            if s1_count == s2_count:
                return True
        
        return False

```

### How It Works:
- We first check if `s1` is longer than `s2`. If it is, we cannot find a permutation, so we return `False`.
- We create a frequency count for `s1` and the first window of `s2`.
- We then slide through `s2`, updating the character count by adding the new character entering the window and removing the old character that is leaving the window.
- After updating the counts for each window position, we check if the two counts (`s1_count` and `s2_count`) match.
- If they match at any point, we return `True`. If we move through all possible windows and find no match, we return `False`.

### Complexity:
- **Time Complexity**: O(n), where n is the length of `s2`, since we make a single pass through `s2` and perform constant time operations for the counting.
- **Space Complexity**: O(1), since the size of the character count arrays is constant (it will at most have 26 entries for lowercase letters).

This implementation is efficient and should run well within the input constraints.

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
 To solve the problem of finding the minimum contiguous substring `W` in `S` such that `T` is a subsequence of `W`, we can leverage a two-pointer technique with some additional data structures to keep track of the required characters.

### Explanation

1. **Understanding Subsequences**:
   A string `T` is a subsequence of string `W` if we can delete some characters from `W` (without reordering the remaining characters) to get `T`. This means that the characters of `T` must appear in `W` in the same order as they appear in `T`, but they do not need to be contiguous.

2. **Two-Pointer Technique**:
   We'll use two pointers: one for traversing string `S` and another for checking if we can build string `T` as a subsequence while expanding or contracting our search window.

3. **Checking for the Minimum Window**:
   We'll maintain a starting pointer for our current valid substring and expand our window by moving our end pointer until we can no longer include `T` as a subsequence. When we have a valid window, we'll try to contract our window from the left to see if we can find a smaller valid window.

4. **Storing the Result**:
   If we find a valid window that contains `T` as a subsequence, we'll check if it's the smallest found so far and update our result accordingly.

### Implementation

Here is the Python code implementing the above logic:



```python
class Solution:
    def minWindow(self, S: str, T: str) -> str:
        from collections import defaultdict
        
        # Create a map for the characters in T
        char_map = defaultdict(int)
        for char in T:
            char_map[char] += 1
            
        required = len(char_map)  # Unique characters in T
        left, right = 0, 0
        formed = 0  # To keep track of when we have matched T's characters
        window_counts = defaultdict(int)  # Store counts of characters in current window
        
        min_length = float('inf')
        min_window = ""
        
        while right < len(S):
            char = S[right]
            window_counts[char] += 1
            
            # Check if the current character added is part of T and if we have enough of it
            if char in char_map and window_counts[char] == char_map[char]:
                formed += 1
            
            # Contract the window until it ceases to be 'desirable' (valid)
            while left <= right and formed == required:
                char = S[left]
                
                # Update the result if this window is smaller than previous ones
                if right - left + 1 < min_length:
                    min_length = right - left + 1
                    min_window = S[left:right + 1]
                
                # The character at the position pointed by `left` is no longer a part of the window
                window_counts[char] -= 1
                if char in char_map and window_counts[char] < char_map[char]:
                    formed -= 1
                
                # Move the left end of the window forward
                left += 1

            # Keep expanding the right end of the window
            right += 1
        
        return min_window

# Example usage:
# solution = Solution()
# result = solution.minWindow("abcdebdde", "bde")
# print(result)  # Output: "bcde"

```

### Key Takeaways
- We leverage a sliding window approach to maintain a dynamic substring of `S`.
- The algorithm efficiently tracks whether the substring contains all characters of `T` using two pointers.
- We maintain character counts to help check if the current window matches the requirements for `T`.
- This yields an efficient solution that runs in O(n) time complexity, making it suitable for large inputs within the constraints provided.

# 978. Longest Turbulent Subarray

### Problem Description 
Given an integer array `arr`, return the length of a maximum size turbulent subarray of `arr`.

A subarray is turbulent if the comparison sign flips between each adjacent pair of elements in the subarray.

More formally, a subarray `[arr[i], arr[i + 1], ..., arr[j]]` of `arr` is said to be turbulent if and only if:
For `i <= k < j`:
	
`arr[k] > arr[k + 1]` when `k` is odd, and
`arr[k] < arr[k + 1]` when `k` is even.

Or, for `i <= k < j`:
	
`arr[k] > arr[k + 1]` when `k` is even, and
`arr[k] < arr[k + 1]` when `k` is odd.


Example 1:
Input: arr = [9,4,2,10,7,8,8,1,9]
Output: 5
Explanation: arr[1] > arr[2] < arr[3] > arr[4] < arr[5]

Example 2:
Input: arr = [4,8,12,16]
Output: 2

Example 3:
Input: arr = [100]
Output: 1

Constraints:
`1 <= arr.length <= 4 * 104`
`0 <= arr[i] <= 109`

### Solution 
 To solve the problem of finding the length of the maximum size turbulent subarray in a given integer array `arr`, we can employ a straightforward approach that involves iterating through the array while maintaining a count of the current turbulent subarray length.

### Detailed Explanation:

1. **Definitions**:
   - A subarray is turbulent if the comparison signs between adjacent elements flip. In simpler terms, for every consecutive pair of elements in the subarray, either:
     - The first element is greater than the second when the position is even, and the first is less than the second when the position is odd, or vice versa.

2. **Initialization**:
   - We will keep track of the maximum length of any turbulent subarray found (`max_length`).
   - We will also need a variable (`current_length`) to keep track of the length of the current turbulent subarray.

3. **Iteration**:
   - Loop through the array starting from the first element to the second last element.
   - For each pair of adjacent elements, determine if they continue the turbulent property:
     - If they form a turbulent pair, we increment the `current_length`.
     - If they do not form a turbulent pair, reset the `current_length` to `2` (as the current pair itself can start a new turbulent subarray).

4. **Update Maximum Length**:
   - After each comparison, update the `max_length` to keep track of the largest turbulent length encountered.

5. **Edge Cases**:
   - If the array has only one element, the maximum turbulent length is `1`.

### Implementation:

Let's implement this logic in the required LeetCode format:



```python
class Solution:
    def maxTurbulentSize(self, arr: List[int]) -> int:
        n = len(arr)
        if n < 2:
            return n  # If array is of length 1, return 1.

        max_length = 1  # At least one element is turbulent by itself.
        current_length = 1  # Current turbulent length is at least 1.

        # Iterate through the array
        for i in range(1, n):
            if (arr[i] > arr[i - 1] and (i % 2 == 1)) or (arr[i] < arr[i - 1] and (i % 2 == 0)):
                current_length += 1  # Continue the current turbulent subarray
            elif (arr[i] < arr[i - 1] and (i % 2 == 1)) or (arr[i] > arr[i - 1] and (i % 2 == 0)):
                current_length += 1  # Continue the current turbulent subarray
            else:
                current_length = 2  # Start new subarray from the last two elements
            
            max_length = max(max_length, current_length)  # Update max length found

        return max_length

```

### How to Run:
- You can run this code directly on the LeetCode website. Simply create a new problem submission for the problem titled "Maximum Length of Turbulent Subarray" and paste the code above into the code editor.
- The code is efficient with a time complexity of O(n) and a space complexity of O(1), making it suitable for the input size constraints given in the problem.

### Additional Thoughts:
- In a turbulent subarray calculation, resetting the length to 2 ensures that we are not missing any potential new turbulent subarrays in the sequence. Each time we encounter a non-turbulent pair, starting fresh preserves the integrity of our search for the maximum length.

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
 To solve the problem of counting the number of contiguous subarrays in an array `A` with exactly `K` distinct integers, we can utilize the sliding window technique along with a hashmap (or dictionary) to keep track of the counts of the elements within our current window.

The idea is to create a helper function that counts the number of subarrays with at most `K` distinct integers. We can then use this function to compute the number of subarrays with exactly `K` distinct integers by applying the principle:


```
number of subarrays with exactly K distinct integers
= number of subarrays with at most K distinct integers 
- number of subarrays with at most (K-1) distinct integers

```

### Detailed Steps:

1. **Helper Function**: Write a function `atMostK` that will return the count of subarrays with at most `K` distinct integers.
   - Initialize a dictionary to keep track of integer counts within the current window.
   - Use a sliding window approach to expand and contract the window as necessary to maintain at most `K` distinct integers.
   - For each step, add the valid subarrays formed by the right pointer.

2. **Compute Result**: Use the helper function to compute:
   - `atMostK(K)` for the number of subarrays with at most `K` distinct integers.
   - `atMostK(K-1)` for the number of subarrays with at most `K-1` distinct integers.
   - Subtract these two counts to get the desired result.

Let’s implement this in Python as a solution that works directly on the LeetCode platform:



```python
class Solution:
    def subarraysWithKDistinct(self, A: List[int], K: int) -> int:
        def atMostK(K):
            count_map = {}
            left = 0
            result = 0
            
            for right in range(len(A)):
                # Add the right element to the count_map
                count_map[A[right]] = count_map.get(A[right], 0) + 1
                
                # If we have more than K distinct integers, shrink the window from the left
                while len(count_map) > K:
                    count_map[A[left]] -= 1
                    if count_map[A[left]] == 0:
                        del count_map[A[left]]
                    left += 1

                # All subarrays that end at 'right' and start from any index between
                # 'left' to 'right' contribute to the result
                result += right - left + 1
            
            return result

        # Calculate the number of subarrays with exactly K distinct integers
        return atMostK(K) - atMostK(K - 1)

# Example Usage:
# solution = Solution()
# print(solution.subarraysWithKDistinct([1, 2, 1, 2, 3], 2))  # Output: 7
# print(solution.subarraysWithKDistinct([1, 2, 1, 3, 4], 3))  # Output: 3

```

### Explanation of the Code:

1. **Function Definition**: The main function `subarraysWithKDistinct` accepts the array `A` and the integer `K`.
2. **Inner Function `atMostK(K)`**:
   - A dictionary `count_map` is used to keep track of the counts of numbers.
   - The variable `left` is used to denote the current start of the window.
   - A loop iterates through each index `right` in the array.
   - If the number of distinct integers in the current window exceeds `K`, we increment `left` until there are at most `K` distinct integers.
   - For each valid position of `right`, we count how many subarrays can end at `right` using the formula `right - left + 1`.
3. **Final Return**: The result is computed by subtracting the counts of subarrays with at most `K-1` from those with at most `K`.

This algorithm operates efficiently with a time complexity of O(N), where N is the length of the input array, making it suitable for the input limits given in the problem statement.

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
 Certainly! The problem you're facing is about finding the minimum number of contiguous subarray flips of length \( K \) such that all elements in an array are converted from `0` to `1`. If it's impossible to achieve this, the answer should be `-1`. Below is a Python solution that follows the LeetCode format, and I'll provide a detailed explanation afterward.

Here’s the code to solve the problem:



```python
class Solution:
    def minKBitFlips(self, A: List[int], K: int) -> int:
        n = len(A)
        flip_count = 0  # Count of flips performed
        flip_effect = [0] * n  # This will keep track of the "flip effect"
        current_flips = 0  # Number of flips affecting the current index

        for i in range(n):
            # Update current_flips based on previous index flip effects
            if i >= K:
                current_flips ^= flip_effect[i - K]
            
            # Determine the actual value of the current bit after considering flips
            actual_value = A[i] ^ current_flips
            
            # If actual_value is 0, we need to flip this bit to make it 1
            if actual_value == 0:
                # Check if we can flip starting at index i
                if i + K > n:  # Check if flipping at index i would go out of bounds
                    return -1
                
                # Perform a flip
                flip_count += 1
                current_flips ^= 1  # We are flipping at this position
                if i + K <= n:  # Record the effect of this flip
                    flip_effect[i] ^= 1
                if i + K < n:  # Cancel the effect of this flip after K steps
                    flip_effect[i + K] ^= 1

        return flip_count

```

### Explanation:

1. **Initialization:**
   - We first declare the `flip_count` variable to count how many flips we perform.
   - We also create a `flip_effect` array to keep track of where flips start and where they should cancel out after \( K \) indices. 
   - The `current_flips` variable keeps track of how many flips are affecting the current position.

2. **Iterating Through the Array:**
   - As we iterate through each index of the array `A`, we update `current_flips` based on the `flip_effect` array. If we have gone beyond the bounds of effect (more than \( K \) previous indices), we toggle the effect off.

3. **Evaluating the Current Bit:**
   - For each index `i`, we compute `actual_value`, which is the transformed value of `A[i]` after applying the accumulated flips (`current_flips`).
   - If `actual_value` is `0`, it means we need to flip it to become `1`. 
     - Before flipping, we check if we can perform a K-bit flip starting at `i`. If `i + K` exceeds the length of `A`, it's impossible to perform this flip, and we return `-1`.
  
4. **Performing the Flip:**
   - If we can flip, we increment the `flip_count` and adjust `current_flips`.
   - We then record the start of the flip in `flip_effect`, and also ensure to cancel the flip's effect after \( K \) indices.

5. **Returning the Result:**
   - After processing all elements, we return the total count of flips required.

### Complexity:
- **Time Complexity:** \( O(n) \) where \( n \) is the length of array `A`.
- **Space Complexity:** \( O(n) \) for the `flip_effect` list.

This solution is efficient and meets the problem's constraints, ensuring it can handle the largest inputs comfortably.

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
 To solve the problem of finding the maximum length of a contiguous subarray containing only `1`s after converting up to `K` `0`s to `1`s, we can utilize a sliding window approach. This algorithm is efficient and helps us maintain a linear time complexity, which is suitable given the constraints of the problem.

### Algorithm Explanation:

1. **Initialize Pointers**: We will use two pointers (`left` and `right`) to represent the current window of elements we are examining in the array. The `right` pointer will expand the window, while the `left` pointer will contract it when necessary.

2. **Count `0`s**: We will maintain a count of how many `0`s are in our current window. The goal is to ensure that this count does not exceed `K`. If it does, we will move the `left` pointer to the right until we have at most `K` `0`s in our window.

3. **Update Maximum Length**: For each valid window (i.e., a window with `0` count ≤ `K`), we will calculate the length of the current window and track the maximum length encountered.

4. **Iterate Through The Array**: Using the `right` pointer, we iterate through the array, expanding the window by incrementing `right`. We simultaneously manage the count of `0`s and adjust the `left` pointer as needed.

5. **Return The Maximum Length**: Once we have traversed the entire array, we return the maximum length of a contiguous subarray that can be formed.

### Python Code:

Here’s how we can implement the above approach in Python, compatible with the LeetCode platform:



```python
class Solution:
    def longestOnes(self, A: List[int], K: int) -> int:
        left = 0
        zero_count = 0
        max_length = 0
        
        for right in range(len(A)):
            # Count zeros in the current window
            if A[right] == 0:
                zero_count += 1
            
            # If the number of zeros exceeds K, shrink the window from the left
            while zero_count > K:
                if A[left] == 0:
                    zero_count -= 1
                left += 1
            
            # Calculate the maximum length of the window
            max_length = max(max_length, right - left + 1)
        
        return max_length

```

### Explanation of the Code:

1. **Class Definition**: We define a `Solution` class which contains our method `longestOnes`.

2. **Initialization**: We initialize a pointer `left` to track the start of our window, `zero_count` to count the number of zeros in the current window, and `max_length` to store the maximum length of subarrays found.

3. **Iterate with `right` Pointer**: As we iterate over each element in `A` using `right`, we check if the current element is `0` and increment our `zero_count`.

4. **Adjusting the `left` Pointer**: If `zero_count` exceeds `K`, we adjust the `left` pointer to reduce the zero count until it is within the allowed limit.

5. **Updating Maximum Length**: After adjusting the window, we calculate the length of the current valid window (`right - left + 1`) and update `max_length` if this window is larger.

6. **Return the Result**: Finally, we return the value of `max_length` which is the length of the longest contiguous subarray that can be formed under the given conditions.

### Complexity:

- **Time Complexity**: O(n) where n is the length of the array `A`. Each element is processed at most twice.
- **Space Complexity**: O(1) since we are using a constant amount of space regardless of the input size. 

This efficient solution adheres to the requirements and retains ease of understanding, making it ideal for a competitive programming scenario like LeetCode.

# 1040. Moving Stones Until Consecutive II

### Problem Description 
On an infinite number line, the position of the i-th stone is given by `stones[i]`.  Call a stone an endpoint stone if it has the smallest or largest position.

Each turn, you pick up an endpoint stone and move it to an unoccupied position so that it is no longer an endpoint stone.

In particular, if the stones are at say, `stones = [1,2,5]`, you cannot move the endpoint stone at position 5, since moving it to any position (such as 0, or 3) will still keep that stone as an endpoint stone.

The game ends when you cannot make any more moves, ie. the stones are in consecutive positions.

When the game ends, what is the minimum and maximum number of moves that you could have made?  Return the answer as an length 2 array: `answer = [minimum_moves, maximum_moves]`

Example 1:
Input: [7,4,9]
Output: [1,2]
Explanation: 
We can move 4 -> 8 for one move to finish the game.

Or, we can move 9 -> 5, 4 -> 6 for two moves to finish the game.


Example 2:
Input: [6,5,4,3,10]
Output: [2,3]
We can move 3 -> 8 then 10 -> 7 to finish the game.

Or, we can move 3 -> 7, 4 -> 8, 5 -> 9 to finish the game.

Notice we cannot move 10 -> 2 to finish the game, because that would be an illegal move.


Example 3:
Input: [100,101,104,102,103]
Output: [0,0]
Note:
`3 <= stones.length <= 10^4`
`1 <= stones[i] <= 10^9`
`stones[i]` have distinct values.

### Solution 
 To solve the problem of determining the minimum and maximum number of moves required to arrange the stones such that they are in consecutive positions on an infinite number line, we can follow these steps:

### Steps to Solve the Problem:

1. **Sort the Stones**: Start by sorting the list of stone positions. This will help us easily identify the smallest and largest stones and see how many positions are available for movement.

2. **Identify Endpoints**: The smallest stone (`min_stone`) and the largest stone (`max_stone`) in the sorted list are the two endpoints we can use to make movements.

3. **Calculate the Gaps**:
    - Compute how many positions are covered by the stones in the range from the smallest (`min_stone`) to the largest (`max_stone`). This can be calculated as `max_stone - min_stone + 1`.
    - The number of occupied positions is simply the length of the sorted stones list.

4. **Find Maximum Moves**: The maximum number of moves we can make before the stones are in consecutive positions is determined by how many unoccupied positions are in this range. It can be calculated by subtracting the number of stones from the total positions available in the range:
   \[
   \text{max\_moves} = \text{(total positions in the range)} - \text{(number of stones)}
   \]

5. **Find Minimum Moves**: The minimum moves require us to find out how many moves it takes to fill the gaps optimally. This can typically be calculated using a greedy strategy, where we fill the gaps between the stones directly:
   - Calculate the total number of "gaps" (distance between consecutive stones) that are greater than `1`. Each such gap can be filled with a move.
   - The minimum moves will be the sum of these gaps minus `1` for each gap.

### Implementation

Now, let’s write the Python code following the explanation given above.



```python
class Solution:
    def numMovesStonesII(self, stones):
        stones.sort()
        n = len(stones)
        
        # Calculate minimum moves
        min_moves = n  # Start with the max possible known moves
        for i in range(n):
            # Slots that can be filled from stone[i] to stone[i + n - 1]
            if i == 0:
                # Consider the leftmost stone
                min_moves = min(min_moves, stones[n - 1] - stones[i] + 1 - n)
            elif i == n - 1:
                # Consider the rightmost stone
                min_moves = min(min_moves, stones[i] - stones[0] + 1 - n)
            else:
                # Consider the middle stones
                slots_filled = max(0, stones[i + 1] - stones[i - 1] - 1)
                min_moves = min(min_moves, n - 1 - slots_filled)

        # Calculate maximum moves
        max_moves = (stones[-1] - stones[0] + 1) - n
        
        return [min_moves, max_moves]

# Example usage
solution = Solution()
print(solution.numMovesStonesII([7, 4, 9]))  # Outputs: [1, 2]
print(solution.numMovesStonesII([6, 5, 4, 3, 10]))  # Outputs: [2, 3]
print(solution.numMovesStonesII([100, 101, 104, 102, 103]))  # Outputs: [0, 0]

```

### Explanation of the Code:

1. **Sorting Stones**: The list of stones is sorted first.
2. **Calculating Minimum Moves**: We iterate through the stones, checking each pair to find gaps and calculating the minimum number of moves needed to fill those gaps.
3. **Calculating Maximum Moves**: It's straightforward; just check the range occupied by the stones and subtract the count of stones.
4. **Return the Result**: Finally, we return an array containing the minimum and maximum number of moves.

This code can be directly executed on the LeetCode platform and meets the problem's requirements.

# 1052. Grumpy Bookstore Owner

### Problem Description 
Today, the bookstore owner has a store open for `customers.length` minutes.  Every minute, some number of customers (`customers[i]`) enter the store, and all those customers leave after the end of that minute.

On some minutes, the bookstore owner is grumpy.  If the bookstore owner is grumpy on the i-th minute, `grumpy[i] = 1`, otherwise `grumpy[i] = 0`.  When the bookstore owner is grumpy, the customers of that minute are not satisfied, otherwise they are satisfied.

The bookstore owner knows a secret technique to keep themselves not grumpy for `X` minutes straight, but can only use it once.

Return the maximum number of customers that can be satisfied throughout the day.


Example 1:
Input: customers = [1,0,1,2,1,1,7,5], grumpy = [0,1,0,1,0,1,0,1], X = 3
Output: 16
Explanation: The bookstore owner keeps themselves not grumpy for the last 3 minutes. 
The maximum number of customers that can be satisfied = 1 + 1 + 1 + 1 + 7 + 5 = 16.

Note:
`1 <= X <= customers.length == grumpy.length <= 20000`
`0 <= customers[i] <= 1000`
`0 <= grumpy[i] <= 1`

### Solution 
 To solve the problem of maximizing the number of satisfied customers while the bookstore owner has a chance to remain "not grumpy" for a certain period (`X`), we can utilize a sliding window approach. This approach will help us efficiently calculate the maximum number of satisfied customers.

Here's a step-by-step breakdown of the solution:

1. **Understanding the Input**:
    - `customers[i]`: The number of customers entering the store during minute `i`.
    - `grumpy[i]`: Indicates whether the owner is grumpy (`1`) or not (`0`) during minute `i`.
    - `X`: The duration the owner can remain not grumpy.

2. **Calculate Total Customers**:
    - First, calculate the total number of customers that would be satisfied if the owner was not grumpy at all (`base_satisfied`).

3. **Sliding Window to Maximize Satisfied Customers**:
    - We will create a sliding window of size `X` that represents the minutes we can maximize customer satisfaction by making the owner not grumpy.
    - For each minute where the owner is grumpy within the sliding window, we can add those customers to a `potential_satisfaction` counter.
    - We will slide this window from the start to the end of the `customers` list and keep track of the maximum number of satisfied customers including the best possible use of the `X` minutes.

4. **Final Calculation**:
    - The result will be the maximum of `base_satisfied + potential_satisfaction` we can get from the sliding window operations.

Here's how the implementation would look in Python:



```python
class Solution:
    def maxSatisfied(self, customers: List[int], grumpy: List[int], X: int) -> int:
        # Calculate the base satisfied customers
        base_satisfied = sum(customers[i] for i in range(len(customers)) if grumpy[i] == 0)
        
        # To maximize additional satisfied customers using X minutes
        max_extra = 0
        current_extra = 0
        
        # Initial window of size X
        for i in range(X):
            if grumpy[i] == 1:
                current_extra += customers[i]
        
        max_extra = current_extra
        
        # Slide the window
        for i in range(X, len(customers)):
            # Remove the effect of the leftmost element of the previous window
            if grumpy[i - X] == 1:
                current_extra -= customers[i - X]
            # Add the effect of the new element in the current window
            if grumpy[i] == 1:
                current_extra += customers[i]
            
            # Update max_extra
            max_extra = max(max_extra, current_extra)
        
        return base_satisfied + max_extra

# Example usage:
solution = Solution()
print(solution.maxSatisfied([1,0,1,2,1,1,7,5], [0,1,0,1,0,1,0,1], 3))  # Output: 16

```

### Explanation of the Code:
- We define a class `Solution` and a method `maxSatisfied`.
- We calculate the `base_satisfied` customers using a list comprehension that sums customers when the owner isn't grumpy.
- We initialize `current_extra` to track potential customers that could be satisfied if the owner were not grumpy during the sliding window.
- We loop through the first `X` elements to establish our starting window, then slide the window across the rest of the list.
- For each iteration, we update the `current_extra` by removing the contribution from the leftmost element that is sliding out of the window and adding the new element entering the window.
- We track the maximum additional customers satisfied (`max_extra`) throughout the iterations.
- Finally, we return the total satisfied customers which is the `base_satisfied` plus the maximum potential extra customers satisfied. 

This approach efficiently computes the answer in O(N) time complexity, where N is the number of minutes, making it suitable for the problem constraints.

# 1074. Number of Submatrices That Sum to Target

### Problem Description 
Given a `matrix` and a `target`, return the number of non-empty submatrices that sum to target.

A submatrix `x1, y1, x2, y2` is the set of all cells `matrix[x][y]` with `x1 <= x <= x2` and `y1 <= y <= y2`.

Two submatrices `(x1, y1, x2, y2)` and `(x1', y1', x2', y2')` are different if they have some coordinate that is different: for example, if `x1 != x1'`.


Example 1:
Input: matrix = [[0,1,0],[1,1,1],[0,1,0]], target = 0
Output: 4
Explanation: The four 1x1 submatrices that only contain 0.


Example 2:
Input: matrix = [[1,-1],[-1,1]], target = 0
Output: 5
Explanation: The two 1x2 submatrices, plus the two 2x1 submatrices, plus the 2x2 submatrix.


Example 3:
Input: matrix = [[904]], target = 0
Output: 0

Constraints:
`1 <= matrix.length <= 100`
`1 <= matrix[0].length <= 100`
`-1000 <= matrix[i] <= 1000`
`-10^8 <= target <= 10^8`

### Solution 
 To solve the problem of counting the number of non-empty submatrices that sum to a given `target`, we'll employ a method that utilizes prefix sums and a hashmap to efficiently track the sums we encounter. This approach has a time complexity of \(O(n^3)\), where \(n\) is the maximum dimension of the matrix, making it efficient given the constraints.

### Steps to Solve the Problem

1. **Prefix Sum Calculation**: We'll first calculate the prefix sums for the matrix which allows us to find the sum of any submatrix in constant time.

2. **Iterating Over All Possible Submatrices**: For each pair of left and right boundaries, we'll iterate through the rows to sum the values between those boundaries. This gives us a 1D array representing the sum of the columns bounded by the left and right limits.

3. **Finding Target Sums**: For each possible sum calculated from the above step, we will use a hashmap to record the number of occurrences of each sum. When a sum that matches the `target` minus the current sum is found, it indicates a submatrix with the desired sum exists.

4. **Count Valid Submatrices**: Finally, we will keep track of all encountered sums to count how many times we've achieved a sum that results in the desired target.

### Python Code Implementation

Here’s the Python code following the LeetCode submission format:



```python
from typing import List
from collections import defaultdict

class Solution:
    def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
        if not matrix:
            return 0
        
        rows = len(matrix)
        cols = len(matrix[0])
        count = 0
        
        # Iterate over all possible pairs of columns
        for left in range(cols):
            # Initialize a temporary array that will hold the row sums
            row_sum = [0] * rows
            
            for right in range(left, cols):
                # Update row_sum for the current right boundary
                for r in range(rows):
                    row_sum[r] += matrix[r][right]
                
                # Use a hashmap to count occurrences of sums
                sum_count = defaultdict(int)
                # Initialize with the sum of 0 (to account for submatrices that equal the target)
                sum_count[0] = 1
                current_sum = 0
                
                # Check the sums along the row_sum array
                for s in row_sum:
                    current_sum += s
                    # Check how many times (current_sum - target) has been seen
                    if current_sum - target in sum_count:
                        count += sum_count[current_sum - target]
                    # Update the hashmap with the current sum
                    sum_count[current_sum] += 1
        
        return count

```

### Explanation of the Code

1. **Initialization**: We check if the matrix is empty and prepare necessary variables including the dimensions of the matrix, `count` to hold the result, and our hashmap `sum_count` to track previously seen sums.

2. **Double Loop for Columns**: We use two nested loops `left` and `right` to represent the boundaries of our submatrices. The outer loop sets the left boundary and the inner loop extends the right boundary.

3. **Row Sum Calculation**: For every right boundary, we update the `row_sum` for each row accumulatively.

4. **Mapping Sums**: For each sum calculated from `row_sum`, we check our hashmap. We are primarily searching for the `current_sum - target`, which would tell us that we have found a valid submatrix whose sum equals the target.

5. **Returning the Count**: Finally, after checking all possible boundaries, we return the total count of submatrices found.

This approach is efficient and works well within the input constraints provided in the problem description.

# 1100. Find K-Length Substrings With No Repeated Characters

### Problem Description 
Given a string `S`, return the number of substrings of length `K` with no repeated characters.


Example 1:
Input: S = "havefunonleetcode", K = 5
Output: 6
Explanation: 
There are 6 substrings they are : 'havef','avefu','vefun','efuno','etcod','tcode'.


Example 2:
Input: S = "home", K = 5
Output: 0
Explanation: 
Notice K can be larger than the length of S. In this case is not possible to find any substring.

Note:
`1 <= S.length <= 10^4`
All characters of S are lowercase English letters.

`1 <= K <= 10^4`

### Solution 
 To solve the problem of counting the number of substrings of length `K` that have no repeated characters, we can take the following approach:

1. **Edge Cases**: First, we need to handle two edge cases:
   - If `K` is greater than the length of the string `S`, it is impossible to find a substring of such length, so we should return `0`.
   - If `K` is `1`, all substrings of length `1` are valid since they consist only of single characters.

2. **Sliding Window Technique**: We can use a sliding window approach to efficiently track substrings of length `K`:
   - We'll maintain a window of size `K` and use a set to check for unique characters.
   - As we slide the window from left to right, we will add the new character on the right and remove the character that is left behind on the left.

3. **Count Valid Substrings**: During each step, after adjusting the window, we check if the character set has a size equal to `K`. If it does, it means all characters in the current window are unique, and we increase our count.

By following this method, we ensure that we traverse the string only once, making the algorithm efficient with a complexity of O(N).

Here’s the implementation of the above logic:



```python
class Solution:
    def numKLenSubstrNoRepeats(self, S: str, K: int) -> int:
        # Edge case: if K is greater than the length of S, return 0
        if K > len(S):
            return 0
        
        # Set to hold unique characters and a counter
        char_set = set()
        count = 0
        
        # Use a sliding window of size K
        left = 0
        
        for right in range(len(S)):
            # Add the new character to the set
            char_set.add(S[right])
            
            # If the window exceeds size K, remove the leftmost character
            if right - left + 1 > K:
                char_set.remove(S[left])
                left += 1
            
            # If the window size is exactly K, check for uniqueness
            if right - left + 1 == K:
                if len(char_set) == K:
                    count += 1
        
        return count

# Example usage
# solution = Solution()
# print(solution.numKLenSubstrNoRepeats("havefunonleetcode", 5))  # Output: 6
# print(solution.numKLenSubstrNoRepeats("home", 5))              # Output: 0

```

### Explanation of the Code:

- **`class Solution`**: This defines our solution class which is required by LeetCode.
  
- **`numKLenSubstrNoRepeats(self, S: str, K: int) -> int`**: This is the method where we implement our logic.
  
- **Edge case handling**: The method starts by checking if `K` is larger than the length of `S` and returns `0` if true.

- **Initialization**:
  - `char_set`: A set to keep track of unique characters in our current sliding window.
  - `count`: A counter to keep track of valid substrings found.

- **Sliding window loop**:
  - We iterate through each character in `S` using the variable `right`.
  - Characters are added to the `char_set` as the window expands.
  - If the size of the window exceeds `K`, we shrink it from the left by removing `S[left]` from the set and incrementing `left`.
  - When the window size is exactly `K`, we check if the size of the set equals `K`. If it does, it means we have a valid substring, and we increase the count.

- **Return the count**: Finally, we return the count of valid substrings. 

This structure and explanation will help anyone understand the solution and format it for LeetCode submission.

# 1151. Minimum Swaps to Group All 1's Together

### Problem Description 
Given a binary array `data`, return the minimum number of swaps required to group all `1`’s present in the array together in any place in the array.


Example 1:
Input: data = [1,0,1,0,1]
Output: 1
Explanation: 
There are 3 ways to group all 1's together:
[1,1,1,0,0] using 1 swap.

[0,1,1,1,0] using 2 swaps.

[0,0,1,1,1] using 1 swap.

The minimum is 1.


Example 2:
Input: data = [0,0,0,1,0]
Output: 0
Explanation: 
Since there is only one 1 in the array, no swaps needed.


Example 3:
Input: data = [1,0,1,0,1,0,0,1,1,0,1]
Output: 3
Explanation: 
One possible solution that uses 3 swaps is [0,0,0,0,0,1,1,1,1,1,1].


Example 4:
Input: data = [1,0,1,0,1,0,1,1,1,0,1,0,0,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,1,1,1,1,0,0,1]
Output: 8

Constraints:
`1 <= data.length <= 105`
`data[i]` is `0` or `1`.

### Solution 
 To solve the problem of determining the minimum number of swaps required to group all `1`s in a binary array together, we can use a sliding window approach. The main idea is to calculate the number of `0`s within a window that can accommodate all the `1`s and minimize this number. 

Here’s a step-by-step breakdown of the approach:

1. **Count Total `1`s**: First, we count how many `1`s are in the array because that's how many we need to group together.

2. **Sliding Window**: We will use a sliding window of size equal to the total count of `1`s. The goal of the window is to find the maximum number of `1`s within any such window. The fewer `0`s we have in this window, the fewer swaps we'll need.

3. **Calculate Swaps**: The minimum number of swaps required will be the total count of `1`s minus the maximum number of `1`s found in any window of that size.

4. **Implementation**: Finally, we'll implement this logic in Python.

Here’s the Python code that follows this logic format and is formatted to run directly on LeetCode:



```python
class Solution:
    def minSwaps(self, data: List[int]) -> int:
        # Step 1: Count the total number of 1s
        total_ones = sum(data)
        
        # If there are no `1`s or only one `1`, no swaps are needed
        if total_ones <= 1:
            return 0
        
        # Step 2: Initialize the sliding window
        max_ones_in_window = 0
        current_ones_count = 0
        
        # Step 3: Calculate number of 1s in the first window of size `total_ones`
        for i in range(total_ones):
            if data[i] == 1:
                current_ones_count += 1
        
        max_ones_in_window = current_ones_count
        
        # Step 4: Slide the window over the rest of the array
        for i in range(total_ones, len(data)):
            # Remove the leftmost element of the window
            if data[i - total_ones] == 1:
                current_ones_count -= 1
            
            # Add the new element at the end of the window
            if data[i] == 1:
                current_ones_count += 1
            
            # Step 5: Update the maximum number of 1s in the window
            max_ones_in_window = max(max_ones_in_window, current_ones_count)
        
        # Step 6: The minimum number of swaps will be the total `1`s
        # minus the maximum number of `1`s found in any window of that size
        return total_ones - max_ones_in_window

```

### Explanation of the Code:
- **Total Count of `1`s**: The variable `total_ones` is computed using `sum(data)`.
- **Sliding Window**: The first for loop initializes our window where we count the number of `1`s in the first `total_ones` elements.
- **Adjusting the Window**: In the second loop, we adjust the window by removing the element that is sliding out (leftmost) and adding the new element that is entering (rightmost). We keep track of `current_ones_count`.
- **Max `1`s Count**: We continually update `max_ones_in_window` to ensure it holds the highest number of `1`s found in any window of the specified size.
- **Final Calculation**: The final number of swaps needed is calculated by subtracting `max_ones_in_window` from `total_ones`.

### Time Complexity: 
The overall time complexity is O(n), where n is the length of the array, since we are making a single pass through the array and using a sliding window approach.

This solution is efficient and works well within the constraints provided.

# 1176. Diet Plan Performance

### Problem Description 
A dieter consumes `calories[i]` calories on the `i`-th day. 
Given an integer `k`, for every consecutive sequence of `k` days (`calories[i], calories[i+1], ..., calories[i+k-1]` for all `0 <= i <= n-k`), they look at T, the total calories consumed during that sequence of `k` days (`calories[i] + calories[i+1] + ... + calories[i+k-1]`):
If `T < lower`, they performed poorly on their diet and lose 1 point; 
If `T > upper`, they performed well on their diet and gain 1 point;
Otherwise, they performed normally and there is no change in points.

Initially, the dieter has zero points. Return the total number of points the dieter has after dieting for `calories.length` days.

Note that the total points can be negative.


Example 1:
Input: calories = [1,2,3,4,5], k = 1, lower = 3, upper = 3
Output: 0
Explanation: Since k = 1, we consider each element of the array separately and compare it to lower and upper.

calories[0] and calories[1] are less than lower so 2 points are lost.

calories[3] and calories[4] are greater than upper so 2 points are gained.


Example 2:
Input: calories = [3,2], k = 2, lower = 0, upper = 1
Output: 1
Explanation: Since k = 2, we consider subarrays of length 2.

calories[0] + calories[1] > upper so 1 point is gained.


Example 3:
Input: calories = [6,5,0,0], k = 2, lower = 1, upper = 5
Output: 0
Explanation:
calories[0] + calories[1] > upper so 1 point is gained.

lower <= calories[1] + calories[2] <= upper so no change in points.

calories[2] + calories[3] < lower so 1 point is lost.


Constraints:
`1 <= k <= calories.length <= 10^5`
`0 <= calories[i] <= 20000`
`0 <= lower <= upper`

### Solution 
 Here is a Python solution to the problem you presented, formatted as you would see it on LeetCode.



```python
class Solution:
    def dietPlanPerformance(self, calories: List[int], k: int, lower: int, upper: int) -> int:
        # Initialize points to zero
        points = 0
        n = len(calories)

        # Calculate the initial sum for the first `k` days
        current_sum = sum(calories[:k])

        # Check the performance of the first window
        if current_sum < lower:
            points -= 1
        elif current_sum > upper:
            points += 1

        # Iterate through the remaining days, updating the current sum
        for i in range(k, n):
            # Slide the window by subtracting the element going out and adding the new element
            current_sum += calories[i] - calories[i - k]

            # Check the performance of the current window
            if current_sum < lower:
                points -= 1
            elif current_sum > upper:
                points += 1

        return points

```

### Explanation:

1. **Function Definition**:
   - We define a method `dietPlanPerformance` that takes four parameters:
     - `calories`: A list of integers where each integer represents the calories consumed on a particular day.
     - `k`: An integer representing the number of consecutive days to consider.
     - `lower`: An integer indicating the lower threshold for performance points.
     - `upper`: An integer indicating the upper threshold for performance points.

2. **Points Initialization**:
   - We initialize a variable `points` to zero, which will keep track of the total points gained or lost.

3. **Initial Sum Calculation**:
   - We calculate the sum of calories for the first `k` days and store this value in `current_sum`. This is done using Python's built-in `sum()` function.

4. **First Window Evaluation**:
   - After calculating the sum for the first window (first `k` days), we check if this sum is less than `lower` or greater than `upper` and adjust the points accordingly.

5. **Sliding Window for Remaining Days**:
   - We then iterate from day `k` to the last day using a `for` loop.
   - For each day, we update the `current_sum` by sliding the window to the right. This is done by subtracting the calorie count of the day that's going out of the window (`calories[i - k]`) and adding the calorie count of the new day that comes into the window (`calories[i]`).
   - We check the performance points again after updating the `current_sum`.

6. **Return Final Points**:
   - Finally, we return the total points the dieter has after evaluating all possible `k` consecutive days.

### Complexity Analysis:
- **Time Complexity**: O(n), where n is the length of `calories`. We make a single pass through the calories list.
- **Space Complexity**: O(1). We use a constant amount of space for storing the points and the current sum, irrespective of the input size.

This solution effectively handles the constraints provided in the problem and efficiently computes the result.

# 1208. Get Equal Substrings Within Budget

### Problem Description 
You are given two strings `s` and `t` of the same length. You want to change `s` to `t`. Changing the `i`-th character of `s` to `i`-th character of `t` costs `|s[i] - t[i]|` that is, the absolute difference between the ASCII values of the characters.

You are also given an integer `maxCost`.

Return the maximum length of a substring of `s` that can be changed to be the same as the corresponding substring of `t`with a cost less than or equal to `maxCost`.

If there is no substring from `s` that can be changed to its corresponding substring from `t`, return `0`.


Example 1:
Input: s = "abcd", t = "bcdf", maxCost = 3
Output: 3
Explanation: "abc" of s can change to "bcd". That costs 3, so the maximum length is 3.


Example 2:
Input: s = "abcd", t = "cdef", maxCost = 3
Output: 1
Explanation: Each character in s costs 2 to change to charactor in `t, so the maximum length is 1.`

Example 3:
Input: s = "abcd", t = "acde", maxCost = 0
Output: 1
Explanation: You can't make any change, so the maximum length is 1.


Constraints:
`1 <= s.length, t.length <= 10^5`
`0 <= maxCost <= 10^6`
`s` and `t` only contain lower case English letters.

### Solution 
 To solve the problem of transforming string `s` to string `t` with a given maximum cost, we can utilize the sliding window technique. This allows us to efficiently determine the maximum length of a contiguous substring within `s` that can be modified to match `t` without exceeding the specified `maxCost`.

### Explanation of the Approach:

1. **Cost Calculation**: The cost to change a character `s[i]` to `t[i]` is given by `|s[i] - t[i]|`. We will create an array of costs for each character based on the ASCII difference.

2. **Sliding Window Technique**: We will use two pointers (`left` and `right`) to represent the current substring of `s` we are considering. We will increment the `right` pointer to enlarge the window and calculate the total cost. If the total cost exceeds `maxCost`, we will move the `left` pointer to shrink the window until we are back within the allowed cost.

3. **Tracking Maximum Length**: During this process, we will keep track of the maximal length of any valid substring.

### Implementation:

Here is the Python code that implements the above logic in a format suitable for LeetCode:



```python
class Solution:
    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        n = len(s)
        left = 0
        total_cost = 0
        max_length = 0
        
        for right in range(n):
            # Calculate the cost to change s[right] to t[right]
            cost = abs(ord(s[right]) - ord(t[right]))
            total_cost += cost
            
            # If the total cost exceeds maxCost, move the left pointer
            while total_cost > maxCost:
                total_cost -= abs(ord(s[left]) - ord(t[left]))
                left += 1
            
            # Calculate the maximum length of the substring found so far
            max_length = max(max_length, right - left + 1)
        
        return max_length

```

### Explanation of the Code:

1. **Initialization**:
   - We initialize variables:
     - `n`: length of `s` (and `t`, since they are of the same length).
     - `left`: left pointer of the sliding window.
     - `total_cost`: to accumulate the cost of changes within the current window.
     - `max_length`: to keep track of the maximum length of valid substrings found.

2. **Iterate Through the String**:
   - We loop through each character in `s` using the `right` pointer. For each character, we compute its transformation cost and add it to `total_cost`.

3. **Adjusting the Window**:
   - If `total_cost` exceeds `maxCost`, we enter a while loop where we increment the `left` pointer, reducing the `total_cost` by the cost of the character being removed from the window until we are back within budget.

4. **Update Maximum Length**:
   - After adjusting the window, we calculate the current valid substring length as `right - left + 1` and update `max_length` if this current length is larger.

5. **Return Result**:
   - Finally, we return the maximum length found.

### Complexity:
- **Time Complexity**: O(n) where n is the length of the strings. Each character is processed at most twice (once by the right pointer and once by the left pointer).
- **Space Complexity**: O(1) as we are only using a constant amount of extra space.

This solution efficiently handles the constraints of the problem and provides the correct result.

# 1423. Maximum Points You Can Obtain from Cards

### Problem Description 
There are several cards arranged in a row, and each card has an associated number of points The points are given in the integer array `cardPoints`.

In one step, you can take one card from the beginning or from the end of the row. You have to take exactly `k` cards.

Your score is the sum of the points of the cards you have taken.

Given the integer array `cardPoints` and the integer `k`, return the maximum score you can obtain.


Example 1:
Input: cardPoints = [1,2,3,4,5,6,1], k = 3
Output: 12
Explanation: After the first step, your score will always be 1. However, choosing the rightmost card first will maximize your total score. The optimal strategy is to take the three cards on the right, giving a final score of 1 + 6 + 5 = 12.


Example 2:
Input: cardPoints = [2,2,2], k = 2
Output: 4
Explanation: Regardless of which two cards you take, your score will always be 4.


Example 3:
Input: cardPoints = [9,7,7,9,7,7,9], k = 7
Output: 55
Explanation: You have to take all the cards. Your score is the sum of points of all cards.


Example 4:
Input: cardPoints = [1,1000,1], k = 1
Output: 1
Explanation: You cannot take the card in the middle. Your best score is 1. 

Example 5:
Input: cardPoints = [1,79,80,1,1,1,200,1], k = 3
Output: 202

Constraints:
`1 <= cardPoints.length <= 10^5`
`1 <= cardPoints[i] <= 10^4`
`1 <= k <= cardPoints.length`

### Solution 
 To solve the problem of maximizing the score from picking `k` cards from either end of the array `cardPoints`, we can utilize a sliding window approach. This method allows us to efficiently calculate the highest score possible from taking a specific number of cards from either end.

### Explanation:

1. **Understanding the Problem**:
   - You are allowed to pick cards from the start or end of a linear array.
   - Your goal is to maximize the sum of the points from exactly `k` cards taken.

2. **Optimal Substructure**:
   - To maximize the score, you can equivalently consider the problem in reverse: instead of choosing `k` cards to take, think about leaving `n - k` cards behind (where `n` is the total number of cards) since both strategies lead to the same final score.

3. **Sliding Window Technique**:
   - Calculate the sum of the entire array of card points.
   - Calculate the sum of the `n - k` cards that are left in the center of the list after taking `k` cards from the ends.
   - By finding the minimum possible sum of these `n - k` cards, you can derive the maximum score by subtracting this from the total sum.

4. **Implementation Steps**:
   - First, compute the total sum of `cardPoints`.
   - Use a sliding window of size `n - k` to find the minimum sum of the `n - k` elements.
   - Finally, subtract this minimum from the total sum to get the maximum score.

### Python Code Implementation:



```python
from typing import List

class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        n = len(cardPoints)
        total_sum = sum(cardPoints)

        # We need to find the minimum sum of n - k cards
        window_size = n - k
        min_sum_of_window = float('inf')

        # Calculate the sum of the initial window of size window_size
        current_window_sum = sum(cardPoints[:window_size])
        min_sum_of_window = current_window_sum

        # Roll the window through the array
        for i in range(window_size, n):
            current_window_sum += cardPoints[i] - cardPoints[i - window_size]
            min_sum_of_window = min(min_sum_of_window, current_window_sum)

        # The maximum score is total points minus the minimum sum of n - k cards
        max_score = total_sum - min_sum_of_window
        return max_score

```

### Explanation of the Code:
1. **Input and Initialization**:
   - We define a class `Solution` that contains the method `maxScore`, taking `cardPoints` and `k` as inputs.
   - Calculate `n`, the number of cards in `cardPoints`, and `total_sum`, the sum of all card points.

2. **Calculate the Minimum Window Sum**:
   - Initialize `min_sum_of_window` with a large number (`float('inf')`) to track the minimum sum of the cards in the window.
   - Compute the sum of the initial window of `n - k` cards.
   - Slide the window from left to right across the card array, updating the current sum and checking if it's the minimum.

3. **Final Calculation**:
   - Once the minimum sum of the center cards is determined, subtract it from the total sum to get the maximum score.

### Time Complexity:
- The complexity of this solution is O(n), where n is the length of `cardPoints`, which ensures that it efficiently handles the maximum input size constraint.

### Test the Function:
You can now run the function `maxScore` against provided examples and any additional test cases to validate its correctness.

# 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

### Problem Description 
Given an array of integers `nums` and an integer `limit`, return the size of the longest non-empty subarray such that the absolute difference between any two elements of this subarray is less than or equal to `limit`.


Example 1:
Input: nums = [8,2,4,7], limit = 4
Output: 2 
Explanation: All subarrays are: 
[8] with maximum absolute diff |8-8| = 0 <= 4.

[8,2] with maximum absolute diff |8-2| = 6 > 4. 
[8,2,4] with maximum absolute diff |8-2| = 6 > 4.

[8,2,4,7] with maximum absolute diff |8-2| = 6 > 4.

[2] with maximum absolute diff |2-2| = 0 <= 4.

[2,4] with maximum absolute diff |2-4| = 2 <= 4.

[2,4,7] with maximum absolute diff |2-7| = 5 > 4.

[4] with maximum absolute diff |4-4| = 0 <= 4.

[4,7] with maximum absolute diff |4-7| = 3 <= 4.

[7] with maximum absolute diff |7-7| = 0 <= 4. 
Therefore, the size of the longest subarray is 2.


Example 2:
Input: nums = [10,1,2,4,7,2], limit = 5
Output: 4 
Explanation: The subarray [2,4,7,2] is the longest since the maximum absolute diff is |2-7| = 5 <= 5.


Example 3:
Input: nums = [4,2,2,2,4,4,2,2], limit = 0
Output: 3

Constraints:
`1 <= nums.length <= 105`
`1 <= nums[i] <= 109`
`0 <= limit <= 109`

### Solution 
 To solve the problem of finding the size of the longest non-empty subarray where the absolute difference between any two elements is less than or equal to a given `limit`, we can utilize a sliding window approach combined with a data structure that allows us to easily track the minimum and maximum values in the current window.

### Explanation of the Approach:

1. **Sliding Window Technique**: We will maintain a window defined by two pointers (start and end) that will indicate the current subarray being considered. The fundamental idea is to extend the `end` pointer to include new elements while ensuring the condition on the maximum and minimum numbers in the subarray is satisfied.

2. **Tracking Min and Max**: To keep track of the minimum and maximum values within our sliding window efficiently, we can use a deque (double-ended queue). This will allow us to keep the indices of the elements in a way that the maximum and minimum can always be found in O(1) time.

3. **Validating the Current Window**: Whenever we include a new element (by moving the `end` pointer), we check if the difference between the maximum and minimum values exceeds the `limit`. If it does, we need to shrink the window from the `start` side until the condition is satisfied again.

4. **Updating the Result**: Each time we have a valid window, we check the size of the current window and update our maximum size accordingly.

### Implementation:

Here’s how the above logic can be implemented in Python:



```python
from collections import deque

class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        max_deque = deque()  # To maintain indices of max elements
        min_deque = deque()  # To maintain indices of min elements
        
        start = 0
        max_length = 0
        
        for end in range(len(nums)):
            # Maintain the max deque
            while max_deque and nums[max_deque[-1]] <= nums[end]:
                max_deque.pop()
            max_deque.append(end)
            
            # Maintain the min deque
            while min_deque and nums[min_deque[-1]] >= nums[end]:
                min_deque.pop()
            min_deque.append(end)

            # Check if the current window is valid
            while nums[max_deque[0]] - nums[min_deque[0]] > limit:
                start += 1  # Move the start right to shrink the window
                
                # Remove indices that are out of the current window
                if max_deque[0] < start:
                    max_deque.popleft()
                if min_deque[0] < start:
                    min_deque.popleft()
                    
            # Update the maximum length of the subarray
            max_length = max(max_length, end - start + 1)

        return max_length

```

### Explanation of the Code:

1. **Initialization**: We initialize two deques for tracking the maximum and minimum elements of the window and set the start index and maximum length to 0.

2. **Loop through `nums`**: We iterate over each element in the `nums` array with the `end` pointer.

3. **Update Deques**:
   - For `max_deque`, we only keep indices of the maximum elements, removing any indices from the back of the deque that are less than or equal to the current element because they aren't useful.
   - The same logic applies to `min_deque`, but we check for elements that are greater than or equal to the current element.

4. **Maintain Valid Window**: Each time a new element is added, if the difference between the maximum and minimum in the window exceeds the `limit`, we increment the `start` pointer to shrink the window until it becomes valid again.

5. **Calculate Maximum Length**: After adjusting the window, we calculate the length and update `max_length` accordingly.

This approach ensures efficient processing of the input array while respecting the problem constraints, yielding a solution that operates in O(n) time complexity.

# 1456. Maximum Number of Vowels in a Substring of Given Length

### Problem Description 
Given a string `s` and an integer `k`.

Return the maximum number of vowel letters in any substring of `s` with length `k`.

Vowel letters in English are (a, e, i, o, u).


Example 1:
Input: s = "abciiidef", k = 3
Output: 3
Explanation: The substring "iii" contains 3 vowel letters.


Example 2:
Input: s = "aeiou", k = 2
Output: 2
Explanation: Any substring of length 2 contains 2 vowels.


Example 3:
Input: s = "leetcode", k = 3
Output: 2
Explanation: "lee", "eet" and "ode" contain 2 vowels.


Example 4:
Input: s = "rhythms", k = 4
Output: 0
Explanation: We can see that s doesn't have any vowel letters.


Example 5:
Input: s = "tryhard", k = 4
Output: 1

Constraints:
`1 <= s.length <= 10^5`
`s` consists of lowercase English letters.

`1 <= k <= s.length`

### Solution 
 To solve the problem of finding the maximum number of vowel letters in any substring of length `k` from the given string `s`, we can utilize the sliding window technique. This approach allows us to efficiently examine each substring of length `k` as we iterate through the string. 

### Step-by-Step Explanation

1. **Identify Vowels**: We first determine which characters are vowels. In English, the vowels are 'a', 'e', 'i', 'o', and 'u'. We can store these in a set for O(1) average-time complexity when checking membership.

2. **Sliding Window Setup**: We will utilize a variable that keeps track of the current count of vowels in the sliding window of size `k`. The initial window starts with the first `k` characters of the string.

3. **Update and Slide the Window**: As we slide the window to the right, we:
   - Subtract the contribution of the character that is leaving the window (the character at the leftmost end).
   - Add the contribution of the character that is entering the window (the new character at the rightmost end).
   
4. **Keep Track of the Maximum**: During these updates, we will maintain a variable to keep track of the maximum number of vowels found in any window.

5. **Return the Result**: Finally, we will return the maximum count of vowels found.

### Implementation

Now, let’s implement this solution in Python, structured in the LeetCode submission format.



```python
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        # Define the set of vowels
        vowels = set('aeiou')
        max_vowels = 0
        current_vowel_count = 0
        
        # Initialize the count for the first window of size k
        for i in range(k):
            if s[i] in vowels:
                current_vowel_count += 1
        
        max_vowels = current_vowel_count
        
        # Slide the window from the start to the end of the string
        for i in range(k, len(s)):
            # Remove the character that is sliding out of the window
            if s[i - k] in vowels:
                current_vowel_count -= 1
            # Add the character that is entering the window
            if s[i] in vowels:
                current_vowel_count += 1
            
            # Update the maximum vowels found
            max_vowels = max(max_vowels, current_vowel_count)
        
        return max_vowels

```

### Explanation of the Code:
- We have a class `Solution` that contains the method `maxVowels`.
- Inside the method:
  - We define the set of vowels.
  - We initialize two variables: `max_vowels` to keep track of the maximum vowels in any observed window, and `current_vowel_count` to count vowels within the current window.
  - We iterate over the first `k` characters to initialize `current_vowel_count`.
  - Then, we slide the window across the string starting from index `k`. For each step, we:
    - Remove the effect of the character exiting the window (at `i - k`) and add the effect of the new character entering the window (at `i`).
    - We then update `max_vowels` with the maximum of its current value and the `current_vowel_count`.
  
Finally, the method returns the `max_vowels`, which represents the highest count of vowels found in any contiguous substring of length `k`.

### Complexity Analysis:
- **Time Complexity**: O(n), where n is the length of the string `s`, since we are processing each character a constant number of times.
- **Space Complexity**: O(1), since we are using a fixed amount of extra space for variables (regardless of the input size). The set of vowels is also fixed at size 5.

# 1498. Number of Subsequences That Satisfy the Given Sum Condition

### Problem Description 
Given an array of integers `nums` and an integer `target`.

Return the number of non-empty subsequences of `nums` such that the sum of the minimum and maximum element on it is less or equal to `target`. Since the answer may be too large, return it modulo `109 + 7`.


Example 1:
Input: nums = [3,5,6,7], target = 9
Output: 4
Explanation: There are 4 subsequences that satisfy the condition.

[3] -> Min value + max value <= target (3 + 3 <= 9)
[3,5] -> (3 + 5 <= 9)
[3,5,6] -> (3 + 6 <= 9)
[3,6] -> (3 + 6 <= 9)

Example 2:
Input: nums = [3,3,6,8], target = 10
Output: 6
Explanation: There are 6 subsequences that satisfy the condition. (nums can have repeated numbers).

[3] , [3] , [3,3], [3,6] , [3,6] , [3,3,6]

Example 3:
Input: nums = [2,3,3,4,6,7], target = 12
Output: 61
Explanation: There are 63 non-empty subsequences, two of them don't satisfy the condition ([6,7], [7]).

Number of valid subsequences (63 - 2 = 61).


Example 4:
Input: nums = [5,2,4,1,7,6,8], target = 16
Output: 127
Explanation: All non-empty subset satisfy the condition (2^7 - 1) = 127

Constraints:
`1 <= nums.length <= 105`
`1 <= nums[i] <= 106`
`1 <= target <= 106`

### Solution 
 To solve the problem of finding the number of non-empty subsequences of an array `nums` such that the sum of the minimum and maximum elements in each subsequence is less than or equal to a given `target`, we can adopt a combination of sorting and a two-pointer technique. Below is a step-by-step explanation of the approach, followed by the Python code that conforms to the LeetCode submission format.

### Explanation

1. **Sorting the Array**:
   We start by sorting the array `nums`. This allows us to easily determine the minimum (first element) and maximum (last element) of any chosen subsequence.

2. **Two-Pointer Technique**:
   We utilize two pointers:
   - `left`: Starts at the beginning of the array.
   - `right`: Starts at the end of the array.
   
   We'll iterate while `left` is less than or equal to `right`:
   - Calculate the sum of the current minimum and maximum: `nums[left] + nums[right]`.
   - If this sum is less than or equal to `target`, this indicates that all elements between these two pointers can form valid subsequences. We use the formula `2^(right - left)` to calculate the total number of valid subsequences that can be formed. Each element between `left` and `right` can either be included or excluded independently, hence the `2` raised to the number of choices.
   - We then increment the `left` pointer to consider the next minimum.
   - If the sum exceeds `target`, we decrement the `right` pointer since we need to consider smaller maximum values.

3. **Modulus Operation**:
   Since the results can be large, we take the result modulo \(10^9 + 7\) as required.

4. **Result Accumulation**:
   Throughout the process, we accumulate the number of valid subsequences and return this value.

### Code Implementation

Here is the Python code that implements the above logic, formatted for LeetCode:



```python
class Solution:
    def numSubseq(self, nums: List[int], target: int) -> int:
        nums.sort()
        left, right = 0, len(nums) - 1
        MOD = 10**9 + 7
        
        # Precompute powers of 2 up to the length of nums
        pow2 = [1] * (len(nums) + 1)
        for i in range(1, len(nums) + 1):
            pow2[i] = (pow2[i - 1] * 2) % MOD
        
        count = 0
        
        while left <= right:
            if nums[left] + nums[right] <= target:
                count += pow2[right - left]
                count %= MOD
                left += 1  # move left pointer to the right
            else:
                right -= 1  # move right pointer to the left
        
        return count

```

### Key Points:
- **Efficiency**: The sorting operation takes \(O(n \log n)\), and the counting with two pointers takes \(O(n)\), leading to an overall time complexity of \(O(n \log n)\).
- **Space Complexity**: The space complexity is \(O(n)\) for storing the precomputed powers of 2.
- **Edge Cases**: The code gracefully handles cases where all elements might need to be included or excluded based on the `target`. 

This solution efficiently counts the valid subsequences while adhering to the constraints provided.

# 1499. Max Value of Equation

### Problem Description 
Given an array `points` containing the coordinates of points on a 2D plane, sorted by the x-values, where `points[i] = [xi, yi]` such that `xi < xj` for all `1 <= i < j <= points.length`. You are also given an integer `k`.

Find the maximum value of the equation `yi + yj + |xi - xj|` where `|xi - xj| <= k` and `1 <= i < j <= points.length`. It is guaranteed that there exists at least one pair of points that satisfy the constraint `|xi - xj| <= k`.


Example 1:
Input: points = [[1,3],[2,0],[5,10],[6,-10]], k = 1
Output: 4
Explanation: The first two points satisfy the condition |xi - xj| <= 1 and if we calculate the equation we get 3 + 0 + |1 - 2| = 4. Third and fourth points also satisfy the condition and give a value of 10 + -10 + |5 - 6| = 1.

No other pairs satisfy the condition, so we return the max of 4 and 1.


Example 2:
Input: points = [[0,0],[3,0],[9,2]], k = 3
Output: 3
Explanation: Only the first two points have an absolute difference of 3 or less in the x-values, and give the value of 0 + 0 + |0 - 3| = 3.


Constraints:
`2 <= points.length <= 10^5`
`points[i].length == 2`
`-10^8 <= points[i][0], points[i][1] <= 10^8`
`0 <= k <= 2 * 10^8`
`points[i][0] < points[j][0]` for all `1 <= i < j <= points.length`
`xi` form a strictly increasing sequence.

### Solution 
 To solve the problem of maximizing the equation `yi + yj + |xi - xj|` under the constraint `|xi - xj| <= k`, we can employ the following approach:

1. **Understanding the Equation:** The equation simplifies to `yi + yj + (xi - xj)` since `xi < xj`. The aim is to maximize this expression specifically when the difference in x-coordinates does not exceed `k`.

2. **Utilizing a Data Structure:** Since we need to find pairs of points that satisfy the x-constraint efficiently, we can use a sliding window (or a deque) to maintain potential candidates of `yi - xi` values. This allows us to quickly compute the maximum `y - x` value for points that fall within our x-constraint.

3. **Algorithm Steps:**
   - Initialize `max_value` to keep track of the maximum result.
   - Use a deque to store values of `yi - xi` in a way that keeps them ordered for efficient retrieval of the maximum value.
   - Iterate through each point, popping elements from the deque that are out of range (i.e., `points[j][0] - points[i][0] > k`).
   - Calculate the current potential maximum using the current `yi + xi` (from the equation) and the maximum from the deque.
   - Add the value `yi - xi` of the current point into the deque for further calculations.

Here’s the Python code to implement the above logic:



```python
from collections import deque
from typing import List

class Solution:
    def findMaxValueOfEquation(self, points: List[List[int]], k: int) -> int:
        # Deque to store potential candidates for (yi - xi)
        dq = deque()
        max_value = float('-inf')
        
        for x, y in points:
            # Remove points that are out of the range of k
            while dq and x - points[dq[0]][0] > k:
                dq.popleft()
            
            # Calculate the potential maximum with the current point
            if dq:
                max_value = max(max_value, y + x + points[dq[0]][1] - points[dq[0]][0])
            
            # Maintain deque for (yi - xi)
            while dq and (y - x > points[dq[-1]][1] - points[dq[-1]][0]):
                dq.pop()
            dq.append(len(points) - 1)
        
        return max_value

```

### Explanation of the Code:
- **Imports:** We use `deque` from `collections` for efficient appending and popping from both ends.
- **Dequeue Usage:** We store indices of points in `dq` that give us the maximum value of `yi - xi`. Before adding a new coordinate, we ensure we only keep feasible points within the defined constraint by checking the x-difference.
- **Main Looping:** For each point `(x, y)` in the input:
  - We pop points from the left side of the deque if they are beyond the allowable x-difference.
  - If the deque is not empty, we calculate the possible maximum value with the current point `y + x` combined with the best `(yi - xi)` from the appropriate point.
  - Lastly, we clean the deque by removing points from the back which are not optimal compared to the current point.
  
### Complexity:
This algorithm runs in O(n) time due to each point being processed at most twice (once added, once removed from the deque). The space complexity is O(n) as well, maintained by the deque of indices.

This efficient method caters well to the problem's constraints and ensures we find the desired maximum value swiftly.

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
 To solve the problem of reducing `x` to zero by removing elements from either the leftmost or rightmost side of the array `nums`, we can use a two-pointer approach along with a dictionary to keep track of subarray sums. Here’s how we can think through the solution:

### Detailed Explanation

1. **Understanding the Problem**:
   We need to achieve a total of `x` by summing up some elements from the array `nums`. Since we are allowed to remove elements from either end, this can be interpreted as needing to find a subarray summing to `sum(nums) - x`.

2. **Transforming the Problem**:
   - Calculate the total sum of the array: `total_sum = sum(nums)`.
   - Our target now becomes `target = total_sum - x`. If `target < 0`, we can immediately return `-1` since it's impossible to achieve.

3. **Finding the Longest Subarray with a Given Sum**:
   - Use a sliding window (or two-pointer) technique to find the longest contiguous subarray whose sum equals `target`.
   - Initialize two pointers: `left` and `right`. Start both at the beginning of the array and expand the `right` pointer until the sum of the current window exceeds `target`.
   - If the sum equals `target`, we have a potential solution, and we can check if it's the longest found. If it exceeds `target`, increment the `left` pointer to reduce the window size.

4. **Returning the Result**:
   - If we find such a subarray, the minimum number of operations equals the total number of elements minus the length of the longest subarray found. If we never find a suitable subarray, return `-1`.

### Implementation in Python



```python
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        total_sum = sum(nums)
        target = total_sum - x
        
        # Edge case if target is negative, return -1
        if target < 0:
            return -1
        
        left = 0
        current_sum = 0
        max_length = -1
        
        for right in range(len(nums)):
            current_sum += nums[right]
            
            # Shrinking from the left while the current sum exceeds the target
            while current_sum > target and left <= right:
                current_sum -= nums[left]
                left += 1
            
            # Check if we found a subarray with sum equal to target
            if current_sum == target:
                max_length = max(max_length, right - left + 1)
        
        # If max_length was updated, the result will be number of elements - max_length
        # else return -1
        return len(nums) - max_length if max_length != -1 else -1

```

### Explanation of the Code

- **Line 1**: We define the class `Solution` with a method `minOperations`.
- **Line 2-3**: We calculate the total sum of the array and determine our target.
- **Line 5**: We check if the target is negative.
- **Line 8-10**: We initialize pointers and a variable to track the maximum length of valid subarrays.
- **Line 12**: We iterate through the `nums` using `right` as the end pointer.
- **Line 14-17**: We adjust the `left` pointer when the current sum exceeds the target.
- **Line 20-21**: We check if we have found a subarray that meets the target and update the maximum length.
- **Line 24**: Finally, we return the result based on whether we found such a subarray.

### Complexity Analysis
- **Time Complexity**: O(n), where n is the number of elements in `nums`. Each element is processed at most twice.
- **Space Complexity**: O(1), no extra space is used except for a few variables.

This solution is efficient and fits well within the constraints provided by the problem statement.

