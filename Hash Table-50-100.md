# 454. 4Sum II

### Problem Description 
Given four lists A, B, C, D of integer values, compute how many tuples `(i, j, k, l)` there are such that `A[i] + B[j] + C[k] + D[l]` is zero.

To make problem a bit easier, all A, B, C, D have same length of N where 0 ≤ N ≤ 500. All integers are in the range of -228 to 228 - 1 and the result is guaranteed to be at most 231 - 1.


Example:
Input:
A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]
Output:
2
Explanation:
The two tuples are:
1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0

### Solution 
 Below is a Python solution to the given problem that is suitable for running directly on the LeetCode platform. The approach uses a hashmap (dictionary in Python) to reduce the problem complexity from O(N^4) to O(N^2) by dividing the problem into two halves.



```python
class Solution:
    def fourSumCount(self, A, B, C, D):
        """
        :type A: List[int]
        :type B: List[int]
        :type C: List[int]
        :type D: List[int]
        :rtype: int
        """
        # Create a dictionary to store the possible sums of elements from A and B and their frequencies
        sum_ab = {}
        for a in A:
            for b in B:
                if a + b in sum_ab:
                    sum_ab[a + b] += 1
                else:
                    sum_ab[a + b] = 1

        # Now check pairs from C and D to see if their negated sum exists in sum_ab
        count = 0
        for c in C:
            for d in D:
                target = -(c + d)
                if target in sum_ab:
                    count += sum_ab[target]
        
        return count

# Example usage:
sol = Solution()
A = [1, 2]
B = [-2, -1]
C = [-1, 2]
D = [0, 2]
print(sol.fourSumCount(A, B, C, D))  # Output: 2

```

### Explanation of the Code:
1. **Class and Method Setup**:
   - The `Solution` class contains the method `fourSumCount`, which takes four integer lists: A, B, C, and D as its arguments.

2. **Creating the Dictionary**:
   - A dictionary called `sum_ab` is used to store all possible sums of the elements of lists A and B and their respective counts. This is done using two nested loops iterating through A and B. For every sum `a + b`, we increase the count in `sum_ab`.

3. **Counting Valid Tuples**:
   - We then initialize `count` to zero. For each pair `(c, d)` from lists C and D, we calculate `target`, the value which, when added to `c + d`, should give zero (i.e., `target = -(c + d)`).
   - If this `target` exists in `sum_ab`, it means there are `sum_ab[target]` ways (pairs from A and B) to form a sum that complements `c + d` to zero. We add this count to our `count` variable.

4. **Return the Result**:
   - Finally, the method returns `count`, which represents the number of valid tuples `(i, j, k, l)` where `A[i] + B[j] + C[k] + D[l]` totals to zero.

This approach effectively reduces the problem complexity by exploiting the hash map to count complement pairs, rather than calculating each potential tuple individually.

# 463. Island Perimeter

### Problem Description 
You are given `row x col` `grid` representing a map where `grid[i][j] = 1` represents land and `grid[i][j] = 0` represents water.

Grid cells are connected horizontally/vertically (not diagonally). The `grid` is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells).

The island doesn't have "lakes", meaning the water inside isn't connected to the water around the island. One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.


Example 1:
Input: grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
Output: 16
Explanation: The perimeter is the 16 yellow stripes in the image above.


Example 2:
Input: grid = [[1]]
Output: 4

Example 3:
Input: grid = [[1,0]]
Output: 4

Constraints:
`row == grid.length`
`col == grid[i].length`
`1 <= row, col <= 100`
`grid[i][j]` is `0` or `1`.

### Solution 
 The problem provided, calculating the perimeter of an island in a grid, can be approached by traversing each cell in the `grid` and checking if that cell is part of the island (`1`). If it is, the perimeter contributed by that cell can be determined by counting its adjacent sides that either lead to water (`0`) or are the boundary of the grid. For instance, if a land cell has a neighboring cell that is water or is outside the grid boundary, it adds to the perimeter count.

Here's a Python solution using this strategy:



```python
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        perimeter = 0

        def is_water_or_outside(r, c):
            if r < 0 or c < 0 or r >= rows or c >= cols:
                return True
            return grid[r][c] == 0

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:  # this is part of the island
                    # Check all four possible directions
                    if is_water_or_outside(i - 1, j):
                        perimeter += 1  # water or outside above
                    if is_water_or_outside(i + 1, j):
                        perimeter += 1  # water or outside below
                    if is_water_or_outside(i, j - 1):
                        perimeter += 1  # water or outside to the left
                    if is_water_or_outside(i, j + 1):
                        perimeter += 1  # water or outside to the right
                    
        return perimeter

# Example usage:
# sol = Solution()
# print(sol.islandPerimeter([[0,1,0,0], [1,1,1,0], [0,1,0,0], [1,1,0,0]]))

```

### Explanation:
- The method `islandPerimeter` calculates the perimeter based on iterating over every cell in the grid.
- The helper function `is_water_or_outside(r, c)` returns `True` if the cell at `(r, c)` is either out of the grid's bounds or contains water (`0`).
- For each land cell (`1`), check its surrounding cells:
  - Add `1` to the perimeter for each surrounding cell that is either `0` or outside the grid boundary.
- The cumulative `perimeter` value after going through all cells gives the final perimeter of the island.

This method is efficient given the constraints (1 <= row, col <= 100) and addresses the problem within reasonable computational limits.


# 500. Keyboard Row

### Problem Description 
Given an array of strings `words`, return the words that can be typed using letters of the alphabet on only one row of American keyboard like the image below.

In the American keyboard:
the first row consists of the characters `"qwertyuiop"`,
the second row consists of the characters `"asdfghjkl"`, and
the third row consists of the characters `"zxcvbnm"`.


Example 1:
Input: words = ["Hello","Alaska","Dad","Peace"]
Output: ["Alaska","Dad"]

Example 2:
Input: words = ["omk"]
Output: []

Example 3:
Input: words = ["adsdf","sfd"]
Output: ["adsdf","sfd"]

Constraints:
`1 <= words.length <= 20`
`1 <= words[i].length <= 100`
`words[i]` consists of English letters (both lowercase and uppercase).

### Solution 
 Here is the Python code to solve the given problem according to the specifications provided. The solution uses a mapping strategy to determine which keyboard row a letter belongs to and then verifies if each word can be entirely typed using one of these rows.



```python
class Solution:
    def findWords(self, words: List[str]) -> List[str]:
        # Defining each row of the American keyboard
        row1 = set("qwertyuiop")
        row2 = set("asdfghjkl")
        row3 = set("zxcvbnm")
        
        # Prepare a function to check if all letters in a word belong to the same set (row)
        def is_one_row(word):
            word_set = set(word.lower())  # Convert all to lowercase to avoid casing issues
            return word_set <= row1 or word_set <= row2 or word_set <= row3
    
        # Filter and return only those words that can be typed using one row
        return [word for word in words if is_one_row(word)]

# Example usage:
# sol = Solution()
# print(sol.findWords(["Hello","Alaska","Dad","Peace"]))  # Output: ["Alaska","Dad"]
# print(sol.findWords(["omk"]))  # Output: []
# print(sol.findWords(["adsdf","sfd"]))  # Output: ["adsdf","sfd"]

```

### Explanation

1. **Keyboard Rows Defined with Sets:**
   Each row of the keyboard is represented as a set of characters. Using sets allows efficient checking if a character or group of characters (which form a word) belongs to the row or not.

2. **Helper Function `is_one_row`:**
   This function takes a word and determines if all its characters belong to one of the predefined sets (row1, row2, or row3). Using `set(word.lower())` ensures that the function is case-insensitive (`lower()` ensures this).

3. **List Comprehension for Filtering Words:**
   With the `is_one_row` function available, a list comprehension filters the input list of words, retaining only those that can be typed using a single row of the keyboard.

4. **Constraints Handling:**
   Given word lengths range and the maximum number of total words, iterating through every word (with individual character checks) should perform efficiently within these constraints.

This solution uses set operations and basic iteration, which fits well within the complexity constraints provided and should be efficient for the input limits specified.

# 508. Most Frequent Subtree Sum

### Problem Description 
Given the root of a tree, you are asked to find the most frequent subtree sum. The subtree sum of a node is defined as the sum of all the node values formed by the subtree rooted at that node (including the node itself). So what is the most frequent subtree sum value? If there is a tie, return all the values with the highest frequency in any order.


Examples 1
Input:
  5
 /  \
2   -3
return [2, -3, 4], since all the values happen only once, return all of them in any order.


Examples 2
Input:
  5
 /  \
2   -5
return [2], since 2 happens twice, however -5 only occur once.

Note:
You may assume the sum of values in any subtree is in the range of 32-bit signed integer.

### Solution 
 To solve the problem of finding the most frequent subtree sum in a given binary tree, we can follow these steps:

1. **Compute Subtree Sums Using Recursion**: Implement a function that will recursively calculate the sum of subtrees for each node and store these sums in a frequency dictionary. This will involve traversing each node of the tree, calculating the sum of values for the subtree rooted at that node, and recording the frequency of each sum in the dictionary.

2. **Determine Most Frequent Sums**: After all subtree sums have been calculated and their frequencies recorded, find the maximum frequency from our dictionary and extract all sums that have this frequency.

3. **Return Results**: Output the list of sums with the highest frequency.

Let's implement the solution in Python, adhering to the format suitable to run directly on most coding platforms like LeetCode. We'll define the `TreeNode` class and implement a solution in the `Solution` class as required in such platforms.



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def findFrequentTreeSum(self, root: TreeNode) -> List[int]:
        from collections import defaultdict
        
        # Dictionary to store frequency of each subtree sum
        subtree_sum_count = defaultdict(int)
        
        def calculate_subtree_sum(node):
            """ Helper function to calculate the sums recursively """
            if not node:
                return 0
            left_sum = calculate_subtree_sum(node.left)
            right_sum = calculate_subtree_sum(node.right)
            current_sum = node.val + left_sum + right_sum
            subtree_sum_count[current_sum] += 1
            return current_sum
        
        # Calculate all subtree sums starting from the root
        calculate_subtree_sum(root)

        # Determine the maximum frequency of any subtree sum
        if not subtree_sum_count:
            return []
        
        max_freq = max(subtree_sum_count.values())
        
        # Collect all sums with the highest frequency
        most_frequent_sums = [s for s, count in subtree_sum_count.items() if count == max_freq]
        
        return most_frequent_sums

# Example usage:
# Construct a simple tree:    5
#                            / \
#                           2  -3
# node1 = TreeNode(5)
# node2 = TreeNode(2)
# node3 = TreeNode(-3)
# node1.left = node2
# node1.right = node3
# sol = Solution()
# print(sol.findFrequentTreeSum(node1)) # Output: [2, -3, 4]

```

This code first defines the `TreeNode` class which is a standard representation for binary trees in many coding platforms. It then defines the `Solution` class with the `findFrequentTreeSum` method. Inside, it defines a nested helper function `calculate_subtree_sum` to compute the subtree sum values recursively. Using a dictionary, it collects these sums along with their frequencies and finally derives and returns the sums with the highest occurrence.

# 525. Contiguous Array

### Problem Description 
Given a binary array, find the maximum length of a contiguous subarray with equal number of 0 and 1. 

Example 1:
Input: [0,1]
Output: 2
Explanation: [0, 1] is the longest contiguous subarray with equal number of 0 and 1.


Example 2:
Input: [0,1,0]
Output: 2
Explanation: [0, 1] (or [1, 0]) is a longest contiguous subarray with equal number of 0 and 1.

Note:
The length of the given binary array will not exceed 50,000.

### Solution 
 Here's a Python code solution for the problem statement using a hashmap to keep track of balance (the difference between number of zeros and ones) and indexes at which each balance first appears. The idea is essentially to transform the array into a balance array where we count +1 for each 1 and -1 for each 0, enabling us to convert the problem into finding the longest subarray with a sum of 0.



```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        balance_to_index = {0: -1}  # Store the balance and the earliest index it appears
        max_length = 0
        balance = 0
        
        for index, num in enumerate(nums):
            if num == 1:
                balance += 1
            else:
                balance -= 1
            
            if balance in balance_to_index:
                # If this balance has been seen before, the sub array from the first occurrence +1
                # to the current index has equal numbers of 1s and 0s
                max_length = max(max_length, index - balance_to_index[balance])
            else:
                balance_to_index[balance] = index

        return max_length

# Example usage:
nums = [0, 1, 0]
sol = Solution()
print(sol.findMaxLength(nums))  # Output: 2

```

Detailed Explanation:
1. **Balance Concept**: The idea is to track the difference between count of 1's and 0's encountered so far. For simplification, if `1` is encounterd, increase balance by +1, if `0` is encountered, decrease balance by -1. 

2. **HashMap (`balance_to_index`)**: Stores the first occurrence of each balance encountered. The key is the balance value and the value is the first index where this balance was seen.

3. **Balance Calculation**: As we traverse through the array, we adjust the `balance` for each entry:
   - Increment the balance by 1 for a `1`.
   - Decrement the balance by 1 for a `0`.
    
4. **Check & Update Max Length**: 
   - If the current `balance` has been seen before, it means there is a subarray between the first occurrence of this `balance` and the current index which has an equal number of `0`s and `1`s. The length of this subarray can be computed as the difference of the current index and the index stored in `balance_to_index`.
   - If the balance is new, note down its index.

5. **Output**: After processing the entire list, the length of the longest such subarray is returned.

This algorithm operates at O(n) for time complexity because each element of the array is processed once, and O(k) for space complexity where k is the number of unique balances, which at maximum can be O(n). This is efficient and appropriate given the problem constraints.

# 535. Encode and Decode TinyURL

### Problem Description 
Note: This is a companion problem to the System Design problem: Design TinyURL.

TinyURL is a URL shortening service where you enter a URL such as `https://leetcode.com/problems/design-tinyurl` and it returns a short URL such as `http://tinyurl.com/4e9iAk`.

Design the `encode` and `decode` methods for the TinyURL service. There is no restriction on how your encode/decode algorithm should work. You just need to ensure that a URL can be encoded to a tiny URL and the tiny URL can be decoded to the original URL.

### Solution 
 Below is a Python code solution for the URL shortening problem, using a simple approach based on hashing and dictionaries. This solution uses the Python `hash` function to generate hash codes and dictionaries to store mappings between original URLs and their hashed representations for encoding and decoding. This is not an actual robust tiny URL system but serves to illustrate the core idea suitable for scalability considerations in system design.



```python
class Codec:
    def __init__(self):
        self.url_to_code = {}
        self.code_to_url = {}
        self.base_url = "http://tinyurl.com/"
    
    def encode(self, longUrl: str) -> str:
        """Encodes a URL to a shortened URL."""
        # Check if the URL has already been encoded
        if longUrl in self.url_to_code:
            # Return the already existing tiny URL
            return self.base_url + self.url_to_code[longUrl]
        
        # Generate a hash of the URL and convert it to a string
        url_hash = str(hash(longUrl))
        # Store the mapping
        self.url_to_code[longUrl] = url_hash
        self.code_to_url[url_hash] = longUrl
        # Return the shortened URL based on the hash
        return self.base_url + url_hash

    def decode(self, shortUrl: str) -> str:
        """Decodes a shortened URL to its original URL."""
        # Extract the hash code from the tiny URL
        code = shortUrl.split('/')[-1]
        # Retrieve the original URL from the stored mapping based on the hash code
        return self.code_to_url.get(code, "")
        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# url = "https://leetcode.com/problems/design-tinyurl"
# tiny_url = codec.encode(url)
# print("TinyURL: ", tiny_url)
# original_url = codec.decode(tiny_url)
# print("Original URL: ", original_url)

```

### Explanation:
1. **Class Definition:** The `Codec` class contains two key methods, `encode` and `decode`, and also maintains two dictionaries along with the base tinyURL string.
   - `url_to_code`: A dictionary to map original URLs to their unique hash code representations.
   - `code_to_url`: A dictionary to perform the reverse mapping of hash codes to original URLs.

2. **Encode Method:**
   - Check if the URL has already been encoded. If already encoded, return the corresponding tiny URL.
   - If not encoded, use Python's `hash` function to generate a unique hash for the URL.
   - Store the mappings in both `url_to_code` and `code_to_url`.
   - Construct and return the short URL based on the generated hash code appended to the base URL.

3. **Decode Method:**
   - Extract the unique hash code from the tiny URL.
   - Use the `code_to_url` dictionary to get back the original URL using the extracted code.
   - Return the original URL.

This solution works for the posed problem, allowing basic encoding and decoding of URLs to their tiny versions and back. Note this is a functional but simplified version which does not handle potential hash collisions robustly or clear aspects regarding the scalability for industrial use, such as how to handle very large numbers of URLs and potential long-term storage limitations. In a production environment, additional mechanisms such as persistent storage, handling collisions in hashing, and possibly more complex encoding mechanisms would be necessary.


# 554. Brick Wall

### Problem Description 
There is a brick wall in front of you. The wall is rectangular and has several rows of bricks. The bricks have the same height but different width. You want to draw a vertical line from the top to the bottom and cross the least bricks.

The brick wall is represented by a list of rows. Each row is a list of integers representing the width of each brick in this row from left to right.

If your line go through the edge of a brick, then the brick is not considered as crossed. You need to find out how to draw the line to cross the least bricks and return the number of crossed bricks.

You cannot draw a line just along one of the two vertical edges of the wall, in which case the line will obviously cross no bricks. 

Example:
Input: [[1,2,2,1],
        [3,1,2],
        [1,3,2],
        [2,4],
        [3,1,2],
        [1,3,1,1]]
Output: 2
Explanation: 
Note:
The width sum of bricks in different rows are the same and won't exceed INT_MAX.

The number of bricks in each row is in range [1,10,000]. The height of wall is in range [1,10,000]. Total number of bricks of the wall won't exceed 20,000.

### Solution 
 To solve this problem, we can use a hashmap (or dictionary in Python) to track the frequency of each edge position (not including the last edge of the wall on the right). For each brick in a row, we compute the cumulative widths, which represent the positions of the edges. These positions, if lined upon vertically across multiple rows, would potentially be the best places to draw our vertical line such that the least number of bricks are crossed.

Here's a step-by-step breakdown:
1. Initialize a dictionary `edge_count` to store the number of times an edge appears at each position across all rows, except the last edge of each row.
2. Iterate over each row of the wall:
   - Compute the cumulative width for each brick to get the position of the edges.
   - Update the `edge_count` dictionary for each edge position encountered.
3. The best position to draw the vertical line is the one which has appeared the most in `edge_count` since it means that at this position, the line crosses through the edges (and not the bricks).
4. To find the minimal number of crossed bricks, subtract the highest count of an edge position from the number of rows (the height of the wall).

Let's implement the solution in Python:



```python
class Solution:
    def leastBricks(self, wall: List[List[int]]) -> int:
        from collections import defaultdict
        
        # This dictionary will hold the counts of edge positions (except the last edge on each row)
        edge_count = defaultdict(int)
        
        for row in wall:
            edge_position = 0
            # Avoid considering the last edge by using row[:-1]
            for brick_width in row[:-1]:
                edge_position += brick_width
                edge_count[edge_position] += 1
        
        # If no internal edge (i.e., every row is a single brick spanning the full width), return wall height
        if len(edge_count) == 0:
            return len(wall)
        
        # Find the edge position with the maximum count
        max_edges = max(edge_count.values())
        
        # Subtract this from the total number of rows to get the minimum number of crossed bricks
        return len(wall) - max_edges


```

### Explanation:
**Time Complexity:** O(N), where N is the total number of bricks in the wall. Every brick is accessed once to compute the edge positions.

**Space Complexity:** O(M), where M is the number of unique edge positions encountered (in the worst case, each brick could introduce a new edge position). 

Note: This solution efficiently handles the calculation by leveraging cumulative sums and counting occurrences using a dictionary. It ensures we avoid drawing the line at the ends of the wall using row[:-1] to ignore the last edge during the accumulation phase.

# 560. Subarray Sum Equals K

### Problem Description 
Given an array of integers `nums` and an integer `k`, return the total number of continuous subarrays whose sum equals to `k`.


Example 1:
Input: nums = [1,1,1], k = 2
Output: 2

Example 2:
Input: nums = [1,2,3], k = 3
Output: 2

Constraints:
`1 <= nums.length <= 2 * 104`
`-1000 <= nums[i] <= 1000`
`-107 <= k <= 107`

### Solution 
 Below is a Python solution for the given problem using a combination of prefix sum and hashmap to achieve an efficient algorithm with O(n) time complexity. The explanation of the solution is detailed in comments within the code.



```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        prefix_sum_count = {}  # This hashmap stores prefix sums and their occurrences.
        current_sum = 0        # Current prefix sum initialized to 0.
        total_subarrays = 0    # This will store the total number of subarrays that sum up to k.

        # Initialize the hashmap with prefix sum 0 occurring once, to handle the case where a valid subarray starts from index 0.
        prefix_sum_count[0] = 1

        for num in nums:
            current_sum += num  # Update the prefix sum with the current number.

            # Check if (current_sum - k) is a prefix sum that has occurred, this means there is a subarray ending at current index which sums up to k.
            if (current_sum - k) in prefix_sum_count:
                total_subarrays += prefix_sum_count[current_sum - k]

            # Update the hashmap with the current prefix sum.
            if current_sum in prefix_sum_count:
                prefix_sum_count[current_sum] += 1
            else:
                prefix_sum_count[current_sum] = 1

        return total_subarrays

# The below lines are for testing purposes and need not be included when running on LeetCode.
# sol = Solution()
# print(sol.subarraySum([1, 1, 1], 2))  # Output: 2
# print(sol.subarraySum([1, 2, 3], 3))  # Output: 2

```

### Explanation:
1. **Prefix Sum and Hashmap**:
   - We use a hashmap (`prefix_sum_count`) to store the occurrence of each prefix sum. The key is the sum, and the value is the frequency of that sum.
   - `current_sum` keeps a running total of the sum of elements we've seen so far.
 
2. **Calculating the Number of Valid Subarrays**:
   - As we iterate over `nums`, we update `current_sum`.
   - We check if `current_sum - k` exists in the hashmap. If it does, it indicates there are subarrays that sum up to `k` ending at the current index. The value at `prefix_sum_count[current_sum - k]` tells us how many such subarrays.
   - We add this count to `total_subarrays`.

3. **Update HashMap**:
   - For each `current_sum`, we update `prefix_sum_count` either by initializing it or incrementing the existing count.

This approach allows us to find the total number of subarrays efficiently in linear time, by leveraging the prefix sum technique and using a hashmap for fast look up.

# 575. Distribute Candies

### Problem Description 
Alice has `n` candies, where the `ith` candy is of type `candyType[i]`. Alice noticed that she started to gain weight, so she visited a doctor.

The doctor advised Alice to only eat `n / 2` of the candies she has (`n` is always even). Alice likes her candies very much, and she wants to eat the maximum number of different types of candies while still following the doctor's advice.

Given the integer array `candyType` of length `n`, return the maximum number of different types of candies she can eat if she only eats `n / 2` of them.


Example 1:
Input: candyType = [1,1,2,2,3,3]
Output: 3
Explanation: Alice can only eat 6 / 2 = 3 candies. Since there are only 3 types, she can eat one of each type.


Example 2:
Input: candyType = [1,1,2,3]
Output: 2
Explanation: Alice can only eat 4 / 2 = 2 candies. Whether she eats types [1,2], [1,3], or [2,3], she still can only eat 2 different types.


Example 3:
Input: candyType = [6,6,6,6]
Output: 1
Explanation: Alice can only eat 4 / 2 = 2 candies. Even though she can eat 2 candies, she only has 1 type.


Constraints:
`n == candyType.length`
`2 <= n <= 104`
`n` is even.

`-105 <= candyType[i] <= 105`

### Solution 
 To solve this problem efficiently, we can utilize Python's set to count the number of distinct candy types and simple arithmetic to determine the maximum number of different candies Alice can eat, bounded by the doctor's advice.

The key steps are:
1. Count the unique candy types using a set, since each element in a set is unique.
2. Alice can eat up to `n / 2` candies. The result will be the minimum of the unique candy types count and `n / 2`.

This approach harnesses the simplicity of set operations which help in determining the unique types directly and uses minimal computation to find out the answer. The time complexity is O(n) due to the set operation over all elements, and the space complexity is also O(n) in the worst case scenario (all candies are of distinct types).

Below is the implementation in Python that adheres to the format suitable for direct execution on LeetCode's platform.



```python
from typing import List

class Solution:
    def distributeCandies(self, candyType: List[int]) -> int:
        unique_candy_types = len(set(candyType))
        max_candies_allowed = len(candyType) // 2
        return min(unique_candy_types, max_candies_allowed)

# Example usage
if __name__ == "__main__":
    sol = Solution()
    print(sol.distributeCandies([1,1,2,2,3,3]))  # Output: 3
    print(sol.distributeCandies([1,1,2,3]))      # Output: 2
    print(sol.distributeCandies([6,6,6,6]))      # Output: 1

```

In this solution:
- `unique_candy_types` is calculated as the length of the set of `candyType`, which gives the number of unique types.
- `max_candies_allowed` is simply `n / 2` which is taken by integer division `//`.
- The ultimate return is the minimum of `unique_candy_types` and `max_candies_allowed`, which guarantees that Alice gets the maximal variety without exceeding the doctor's restrictions.

# 594. Longest Harmonious Subsequence

### Problem Description 
We define a harmonious array as an array where the difference between its maximum value and its minimum value is exactly `1`.

Given an integer array `nums`, return the length of its longest harmonious subsequence among all its possible subsequences.

A subsequence of array is a sequence that can be derived from the array by deleting some or no elements without changing the order of the remaining elements.


Example 1:
Input: nums = [1,3,2,2,5,2,3,7]
Output: 5
Explanation: The longest harmonious subsequence is [3,2,2,2,3].


Example 2:
Input: nums = [1,2,3,4]
Output: 2

Example 3:
Input: nums = [1,1,1,1]
Output: 0

Constraints:
`1 <= nums.length <= 2 * 104`
`-109 <= nums[i] <= 109`

### Solution 
 To solve this problem in Python, you can follow these steps:

1. Use a dictionary to keep track of the frequency of each number in the array.
2. Iterate over the keys of the dictionary, and for each key, calculate the size of the subsequence formed by it and its neighboring number (either key+1 or key-1, whichever exists in the dictionary).
3. Keep track of the maximum size of these subsequences throughout the iteration.

This approach is efficient as it only involves passing through the given array once and then iterating through the dictionary keys once.

Here is the Python code that fulfills the requirements and can be executed directly on the LeetCode platform:



```python
class Solution:
    def findLHS(self, nums: List[int]) -> int:
        num_count = {}
        
        # Fill dictionary with frequencies of each num
        for num in nums:
            if num in num_count:
                num_count[num] += 1
            else:
                num_count[num] = 1
        
        max_length = 0
        
        # Check each number in dictionary to find the harmonious subsequences
        for num in num_count:
            if num + 1 in num_count:
                # Calculate the size of subsequence combining the number and its neighbor
                current_length = num_count[num] + num_count[num + 1]
                # Update max length if the current one is larger
                if current_length > max_length:
                    max_length = current_length
        
        return max_length


```

Explanation:
- `num_count` is a dictionary that stores the frequency of each element in the input list `nums`.
- The `for` loop fills `num_count` with the proper counts of each element.
- After collecting frequencies, another loop checks for each number if its consecutive number (num + 1) exists in the dictionary `num_count`. If it exists, it calculates the total count of these two kinds of numbers to form a subsequence, then checks if this is the largest harmonious subsequence found so far.
- The answer, `max_length`, is the length of the longest harmonious subsequence detected.

This solution operates in O(n) time complexity since it involves a linear pass to build the frequency dictionary and another to examine possible subsequences. The space complexity is also O(n), primarily due to the storage required for the dictionary `num_count`.

# 599. Minimum Index Sum of Two Lists

### Problem Description 
Suppose Andy and Doris want to choose a restaurant for dinner, and they both have a list of favorite restaurants represented by strings.

You need to help them find out their common interest with the least list index sum. If there is a choice tie between answers, output all of them with no order requirement. You could assume there always exists an answer.


Example 1:
Input: list1 = ["Shogun","Tapioca Express","Burger King","KFC"], list2 = ["Piatti","The Grill at Torrey Pines","Hungry Hunter Steakhouse","Shogun"]
Output: ["Shogun"]
Explanation: The only restaurant they both like is "Shogun".


Example 2:
Input: list1 = ["Shogun","Tapioca Express","Burger King","KFC"], list2 = ["KFC","Shogun","Burger King"]
Output: ["Shogun"]
Explanation: The restaurant they both like and have the least index sum is "Shogun" with index sum 1 (0+1).


Example 3:
Input: list1 = ["Shogun","Tapioca Express","Burger King","KFC"], list2 = ["KFC","Burger King","Tapioca Express","Shogun"]
Output: ["KFC","Burger King","Tapioca Express","Shogun"]

Example 4:
Input: list1 = ["Shogun","Tapioca Express","Burger King","KFC"], list2 = ["KNN","KFC","Burger King","Tapioca Express","Shogun"]
Output: ["KFC","Burger King","Tapioca Express","Shogun"]

Example 5:
Input: list1 = ["KFC"], list2 = ["KFC"]
Output: ["KFC"]

Constraints:
`1 <= list1.length, list2.length <= 1000`
`1 <= list1[i].length, list2[i].length <= 30`
`list1[i]` and `list2[i]` consist of spaces `' '` and English letters.

All the stings of `list1` are unique.

All the stings of `list2` are unique.

### Solution 
 

```python
class Solution:
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        # Using a dictionary to map the restaurant names in list1 to their indices for fast lookup
        map1 = {restaurant: i for i, restaurant in enumerate(list1)}
        min_sum = float('inf')  # Initialize with a large number to find the minimum
        best_restaurants = []
        
        # Iterate over list2 and compute the index sum where restaurants are common
        for index2, restaurant in enumerate(list2):
            if restaurant in map1:
                # The index sum calculation
                index_sum = map1[restaurant] + index2
                # If we found a smaller index sum, reset the list and update min_sum
                if index_sum < min_sum:
                    best_restaurants = [restaurant]
                    min_sum = index_sum
                elif index_sum == min_sum:
                    # If the current index sum is the same as the min_sum, append the restaurant
                    best_restaurants.append(restaurant)

        return best_restaurants

# Example usage below:
# sol = Solution()
# print(sol.findRestaurant(["Shogun","Tapioca Express","Burger King","KFC"], ["Piatti","The Grill at Torrey Pines","Hungry Hunter Steakhouse","Shogun"]))
# print(sol.findRestaurant(["Shogun","Tapioca Express","Burger King","KFC"], ["KFC","Shogun","Burger King"]))
# print(sol.findRestaurant(["Shogun","Tapioca Express","Burger King","KFC"], ["KFC","Burger King","Tapioca Express","Shogun"]))
# print(sol.findRestaurant(["Shogun","Tapioca Express","Burger King","KFC"], ["KNN","KFC","Burger King","Tapioca Express","Shogun"]))
# print(sol.findRestaurant(["KFC"], ["KFC"]))

```

Explanation:
1. We first build a dictionary `map1` where keys are the restaurant names from `list1` and values are their indices. This allows us to quickly check if a restaurant from `list2` exists in `list1` and to retrieve its index in constant time.

2. We then initialize `min_sum` with a very large number (infinity in this case) to compare and find the smallest index sum of common restaurants.

3. We loop through `list2`, checking if each restaurant exists in `map1`. If it does, we compute the index sum by adding the index from `list1` (retrieved from `map1`) and the current index from `list2`.

4. If the computed index sum is less than the current `min_sum`, we update `min_sum` and start a new list of `best_restaurants` with the current restaurant because it represents a new minimum sum.

5. If the computed index sum equals the current `min_sum`, we append the restaurant to `best_restaurants` because it also has the minimum index sum.

6. Finally, we return the list `best_restaurants` which contains all the restaurants with the minimum index sum. Note that the problem guarantees there is always an answer.

This solution efficiently checks and computes the desired result using hash mapping and is suitable for the given constraints.

# 609. Find Duplicate File in System

### Problem Description 
Given a list `paths` of directory info, including the directory path, and all the files with contents in this directory, return all the duplicate files in the file system in terms of their paths. You may return the answer in any order.

A group of duplicate files consists of at least two files that have the same content.

A single directory info string in the input list has the following format:
`"root/d1/d2/.../dm f1.txt(f1_content) f2.txt(f2_content) ... fn.txt(fn_content)"`
It means there are `n` files `(f1.txt, f2.txt ... fn.txt)` with content `(f1_content, f2_content ... fn_content)` respectively in the directory "`root/d1/d2/.../dm"`. Note that `n >= 1` and `m >= 0`. If `m = 0`, it means the directory is just the root directory.

The output is a list of groups of duplicate file paths. For each group, it contains all the file paths of the files that have the same content. A file path is a string that has the following format:
`"directory_path/file_name.txt"`

Example 1:
Input: paths = ["root/a 1.txt(abcd) 2.txt(efgh)","root/c 3.txt(abcd)","root/c/d 4.txt(efgh)","root 4.txt(efgh)"]
Output: [["root/a/2.txt","root/c/d/4.txt","root/4.txt"],["root/a/1.txt","root/c/3.txt"]]

Example 2:
Input: paths = ["root/a 1.txt(abcd) 2.txt(efgh)","root/c 3.txt(abcd)","root/c/d 4.txt(efgh)"]
Output: [["root/a/2.txt","root/c/d/4.txt"],["root/a/1.txt","root/c/3.txt"]]

Constraints:
`1 <= paths.length <= 2 * 104`
`1 <= paths[i].length <= 3000`
`1 <= sum(paths[i].length) <= 5 * 105`
`paths[i]` consist of English letters, digits, `'/'`, `'.'`, `'('`, `')'`, and `' '`.

You may assume no files or directories share the same name in the same directory.

You may assume each given directory info represents a unique directory. A single blank space separates the directory path and file info.

Follow up:
Imagine you are given a real file system, how will you search files? DFS or BFS?
If the file content is very large (GB level), how will you modify your solution?
If you can only read the file by 1kb each time, how will you modify your solution?
What is the time complexity of your modified solution? What is the most time-consuming part and memory-consuming part of it? How to optimize?
How to make sure the duplicated files you find are not false positive?

### Solution 
 

```python
class Solution:
    def findDuplicate(self, paths: List[str]) -> List[List[str]]:
        from collections import defaultdict
        content_to_path = defaultdict(list)
        
        # iterate through each path in the input
        for path in paths:
            parts = path.split(" ")
            root = parts[0]  # the root directory path
            
            # each file information in the format "filename(filecontent)"
            for item in parts[1:]:
                file_name_index = item.index('(')
                file_name = item[:file_name_index]
                file_content = item[file_name_index + 1 : -1]
                full_path = "{}/{}".format(root, file_name)
                content_to_path[file_content].append(full_path)
        
        # we only want to collect sets of files that have duplicates, at least two files with the same content
        result = []
        for files in content_to_path.values():
            if len(files) > 1:
                result.append(files)
        
        return result

# Explanation:
# The problem gives a unique data format where each string contains a directory path followed by space-separated files
# and their contents. The task is to group the paths of files with the same content.
#
# For input handling, we split the path string into root directory and file components. Each file component is further
# parsed to extract filenames and their contents using string indexing.
#
# We use a dictionary to map content to file paths – this allows easy grouping of files by their content.
# The complexity of the solution primarily involves parsing the input and handling the appropriate string operations.
# Given the constraints, our approach should be efficient.
#
# Possible extensions (from follow-up questions):
# - For a real file system scan, DFS might be more appropriate assuming deep directory structures.
# - For very large file contents, hashing (e.g., MD5) the content and comparing hashes can be memory-efficient.
# - For partial file reading, hashing chunks and incrementally checking against other files' hashes could be viable.
# - Ensuring no false positives can be assured by not just hash comparison, but also directly comparing file contents
# when hashes match (hash collision consideration).

```

This Python class is formatted to comply with the typical function signature expected in LeetCode challenges. The class `Solution` has a method `findDuplicate` that takes a list of strings describing file paths and returns a list of lists, where each sublist contains the paths of duplicated files. The logic includes parsing the odd input format, using a dictionary to map file contents to their paths, and filtering to only include groups with duplicates. The solution addresses both efficiency and correctness given the problem constraints.

# 624. Maximum Distance in Arrays

### Problem Description 
You are given `m` `arrays`, where each array is sorted in ascending order.

You can pick up two integers from two different arrays (each array picks one) and calculate the distance. We define the distance between two integers `a` and `b` to be their absolute difference `|a - b|`.

Return the maximum distance.


Example 1:
Input: arrays = [[1,2,3],[4,5],[1,2,3]]
Output: 4
Explanation: One way to reach the maximum distance 4 is to pick 1 in the first or third array and pick 5 in the second array.


Example 2:
Input: arrays = [[1],[1]]
Output: 0

Example 3:
Input: arrays = [[1],[2]]
Output: 1

Example 4:
Input: arrays = [[1,4],[0,5]]
Output: 4

Constraints:
`m == arrays.length`
`2 <= m <= 105`
`1 <= arrays[i].length <= 500`
`-104 <= arrays[i][j] <= 104`
`arrays[i]` is sorted in ascending order.

There will be at most `105` integers in all the arrays.

### Solution 
 To solve this problem, we will keep track of the minimum and maximum values observed from previously processed arrays while iterating through the list of arrays. Specifically, for each array, we attempt to maximize the difference using the global minimum and maximum encountered so far, then we update these global values based on the current array's minimum and maximum values.

This approach works because, by maintaining the absolute smallest and largest numbers seen from distinct arrays as we iterate through them, we can ensure that we are measuring the maximum distance possible between values derived from two different arrays.

Here is the Python code in the format compatible with running directly on the LeetCode platform:



```python
class Solution:
    def maxDistance(self, arrays):
        # Initialize variables to store min and max from the first array
        min_val = arrays[0][0]
        max_val = arrays[0][-1]
        max_distance = 0
        
        # Iterate over arrays starting from the second array
        for i in range(1, len(arrays)):
            current_array = arrays[i]
            # The current array's first and last element (as it is sorted)
            current_min = current_array[0]
            current_max = current_array[-1]
            
            # Calculate distance comparing new array min/max with global max/min
            max_distance = max(max_distance, abs(current_max - min_val), abs(max_val - current_min))
            
            # Update the global min and max values with current array's min and max
            min_val = min(min_val, current_min)
            max_val = max(max_val, current_max)
        
        return max_distance

```

This solution precisely follows:
1. Maintaining global maximum and minimum values while iterating through the list to ensure they are always from different arrays.
2. Each iteration updates these global values and computes the possible maximum distances.
3. The computational complexity is O(n) where n is the number of arrays, since each array's processing involves constant time operations given that the arrays are already sorted.

It efficiently assesses possible solutions during iteration, using prior information to constantly update possible answers without needing to compare each combination exhaustively, thereby optimizing the running time.

# 632. Smallest Range Covering Elements from K Lists

### Problem Description 
You have `k` lists of sorted integers in non-decreasing order. Find the smallest range that includes at least one number from each of the `k` lists.

We define the range `[a, b]` is smaller than range `[c, d]` if `b - a < d - c` or `a < c` if `b - a == d - c`.


Example 1:
Input: nums = [[4,10,15,24,26],[0,9,12,20],[5,18,22,30]]
Output: [20,24]
Explanation: 
List 1: [4, 10, 15, 24,26], 24 is in range [20,24].

List 2: [0, 9, 12, 20], 20 is in range [20,24].

List 3: [5, 18, 22, 30], 22 is in range [20,24].


Example 2:
Input: nums = [[1,2,3],[1,2,3],[1,2,3]]
Output: [1,1]

Example 3:
Input: nums = [[10,10],[11,11]]
Output: [10,11]

Example 4:
Input: nums = [[10],[11]]
Output: [10,11]

Example 5:
Input: nums = [[1],[2],[3],[4],[5],[6],[7]]
Output: [1,7]

Constraints:
`nums.length == k`
`1 <= k <= 3500`
`1 <= nums[i].length <= 50`
`-105 <= nums[i][j] <= 105`
`nums[i]` is sorted in non-decreasing order.

### Solution 
 To solve this problem on LeetCode, we will use a min-heap to keep track of the smallest elements across all k lists along with pointers indicating where each list is currently being accessed.

**Strategy:**
1. Initialize a min-heap. Each element in the heap will contain three values: the value from the lists, the index of the list it came from, and the index within that list.
2. Push the first element of each list into the heap.
3. Maintain a variable to track the current maximum value among the elements at the pointers' locations.
4. Pop the smallest item from the heap (which will be at the root), record its value, and check if the range between this minimum value and the tracked maximum value is smaller than the previously found smallest range.
5. Push the next item from the same list (from which the item was just popped) into the heap, update the maximum if necessary, and repeat the process.
6. Terminate when any list is exhausted (i.e., when we try to push an element from a list but there's no next element).

Using this strategy ensures that at every step, the smallest item across all current list pointers is considered, and we attempt to cover elements from every list in the smallest possible range.

We'll write this plan into Python code using a priority queue (min-heap) implementation with the `heapq` library. We will define:
- A min-heap to store the current min values.
- A variable to track the max of currently considered values to find the range at each step.
- Pointers or indices for each list implemented within our min-heap structure.



```python
from heapq import heappop, heappush
from typing import List

class Solution:
    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        min_heap = []
        current_max = float('-inf')
        range_start, range_end = 0, float('inf')
        
        # Initialize the min_heap and find the initial max
        for index, sorted_list in enumerate(nums):
            heappush(min_heap, (sorted_list[0], index, 0))
            current_max = max(current_max, sorted_list[0])

        # Now maintain the heap
        while min_heap:
            current_min, list_index, element_index = heappop(min_heap)
            
            # Update the range if current [min, max] is smaller
            if current_max - current_min < range_end - range_start:
                range_start, range_end = current_min, current_max
            
            # Check if we need to get the next element in the same list
            if element_index + 1 < len(nums[list_index]):
                next_element = nums[list_index][element_index + 1]
                heappush(min_heap, (next_element, list_index, element_index + 1))
                current_max = max(current_max, next_element)
            else:
                # If list is exhausted, break loop
                break
            
        return [range_start, range_end]

# Test cases from problem statement
solution = Solution()
print(solution.smallestRange([[4,10,15,24,26], [0,9,12,20], [5,18,22,30]]))  # Output: [20,24]
print(solution.smallestRange([[1,2,3], [1,2,3], [1,2,3]]))                   # Output: [1,1]
print(solution.smallestRange([[10,10], [11,11]]))                            # Output: [10,11]
print(solution.smallestRange([[10], [11]]))                                  # Output: [10,11]
print(solution.smallestRange([[1],[2],[3],[4],[5],[6],[7]]))                 # Output: [1,7]

```

This code maintains a priority queue to efficiently retrieve the smallest element spanning all lists and calculate the smallest range accordingly. It operates in O(N log k) time where N is the total number of elements across all lists, and k is the number of lists, as each push/pop operation in a heap of size k takes O(log k) time. The space complexity is O(k) due to the entries maintained in the heap and additional variables.

# 645. Set Mismatch

### Problem Description 
You have a set of integers `s`, which originally contains all the numbers from `1` to `n`. Unfortunately, due to some error, one of the numbers in `s` got duplicated to another number in the set, which results in repetition of one number and loss of another number.

You are given an integer array `nums` representing the data status of this set after the error.

Find the number that occurs twice and the number that is missing and return them in the form of an array.


Example 1:
Input: nums = [1,2,2,4]
Output: [2,3]

Example 2:
Input: nums = [1,1]
Output: [1,2]

Constraints:
`2 <= nums.length <= 104`
`1 <= nums[i] <= 104`

### Solution 
 This problem can be resolved using different methods such as sorting, hash maps, and the mathematical properties of integers. However, considering LeetCode's typical constraints, an efficient approach with a direct mapping technique using the properties of integers is optimal. Specifically, we can leverage the idea of using properties of sums and the properties of indexes in arrays.

Each integer in the set should have precise frequencies when the sequence is 1 to n:
- Frequency of each integer ideally is 1.

Since one of the numbers repeats and another number is missing, we can use:
1. **Sum difference**: The difference between the sum from 1 to n (`expected_sum`) and the sum of the array (`actual_sum`) provides insight into the sum of the missing and duplicate numbers.
2. **Sum of squares difference**: By comparing the sum of the squares from 1 to n (`expected_square_sum`) with the sum of the squares of the elements in the array (`actual_square_sum`), the combined effects of the missing and duplicate numbers can be isolated.

From the derived equations using sum and square sum:
- Let `x` be the duplicated number and `y` be the missing number.
- We have: `x - y = actual_sum - expected_sum`
- And: `x^2 - y^2 = actual_square_sum - expected_square_sum`
  Which simplifies to `(x + y)(x - y) = actual_square_sum - expected_square_sum`.

We can solve these equations to find x and y.

Below is the corresponding Python code suitable for submission on LeetCode under the stated problem:



```python
class Solution:
    def findErrorNums(self, nums):
        n = len(nums)
        expected_sum = n * (n + 1) // 2  # Sum of first n natural numbers
        expected_square_sum = n * (n + 1) * (2 * n + 1) // 6  # Sum of squares of first n natural numbers
        
        actual_sum = actual_square_sum = 0
        
        for num in nums:
            actual_sum += num
            actual_square_sum += num * num
        
        # Based on the derived equations
        # e1 = x - y
        # e2 = x^2 - y^2 = (x + y)(x - y) -> using e1 for substitution

        # Equating and solving
        e1 = actual_sum - expected_sum  # (x - y)
        e2 = actual_square_sum - expected_square_sum  # (x^2 - y^2)

        # Now solve for x + y using x^2 - y^2 = (x + y)(x - y)
        x_plus_y = e2 // e1  # (x + y)
        
        # Solving the two simultaneous equations:
        # x - y = e1
        # x + y = x_plus_y
        
        # We get:
        # 2x = e1 + x_plus_y
        # 2y = x_plus_y - e1
        x = (e1 + x_plus_y) // 2
        y = (x_plus_y - e1) // 2
        
        return [x, y]

# Example usage:
# sol = Solution()
# print(sol.findErrorNums([1,2,2,4]))  # Output should be [2,3]

```

This code follows the LeetCode input-output scheme and should solve the problem efficiently using mathematical properties to identify the errors in the set of numbers. It operates in O(n) time complexity due to the single pass for summing values and squares, and the space complexity is O(1) since no additional significant space is used.

This approach is effective for large values of `n` since it avoids sorting or use of hash maps which would either increase time complexity or space usage significantly.

# 648. Replace Words

### Problem Description 
In English, we have a concept called root, which can be followed by some other word to form another longer word - let's call this word successor. For example, when the root `"an"` is followed by the successor word `"other"`, we can form a new word `"another"`.

Given a `dictionary` consisting of many roots and a `sentence` consisting of words separated by spaces, replace all the successors in the sentence with the root forming it. If a successor can be replaced by more than one root, replace it with the root that has the shortest length.

Return the `sentence` after the replacement.


Example 1:
Input: dictionary = ["cat","bat","rat"], sentence = "the cattle was rattled by the battery"
Output: "the cat was rat by the bat"

Example 2:
Input: dictionary = ["a","b","c"], sentence = "aadsfasf absbs bbab cadsfafs"
Output: "a a b c"

Example 3:
Input: dictionary = ["a", "aa", "aaa", "aaaa"], sentence = "a aa a aaaa aaa aaa aaa aaaaaa bbb baba ababa"
Output: "a a a a a a a a bbb baba a"

Example 4:
Input: dictionary = ["catt","cat","bat","rat"], sentence = "the cattle was rattled by the battery"
Output: "the cat was rat by the bat"

Example 5:
Input: dictionary = ["ac","ab"], sentence = "it is abnormal that this solution is accepted"
Output: "it is ab that this solution is ac"

Constraints:
`1 <= dictionary.length <= 1000`
`1 <= dictionary[i].length <= 100`
`dictionary[i]` consists of only lower-case letters.

`1 <= sentence.length <= 10^6`
`sentence` consists of only lower-case letters and spaces.

The number of words in `sentence` is in the range `[1, 1000]`
The length of each word in `sentence` is in the range `[1, 1000]`
Each two consecutive words in `sentence` will be separated by exactly one space.

`sentence` does not have leading or trailing spaces.

### Solution 
 Here's a Python code solution that uses the Trie data structure to solve the replacement problem described. Tries are particularly efficient for this kind of prefix matching problems, allowing to quickly determine the smallest prefix that exists in the dictionary.

Below, the solution is formatted in a style ready to be submitted to LeetCode:



```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.isEndOfWord = False

class Solution:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        current = self.root
        for letter in word:
            if letter not in current.children:
                current.children[letter] = TrieNode()
            current = current.children[letter]
        current.isEndOfWord = True
    
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        # Insert all roots into the Trie
        for root in dictionary:
            self.insert(root)
            
        # Function to find the shortest root for a single word
        def findShortestRoot(word):
            current = self.root
            for i, char in enumerate(word):
                if char not in current.children:
                    return word
                current = current.children[char]
                if current.isEndOfWord:
                    # Found a root that matches a prefix of the word
                    return word[:i+1]
            return word
        
        # Process the entire sentence
        words = sentence.split()
        replaced_words = [findShortestRoot(word) for word in words]
        return ' '.join(replaced_words)

# Example usage:
# sol = Solution()
# print(sol.replaceWords(["cat","bat","rat"], "the cattle was rattled by the battery"))

```

### Explanation

1. **TrieNode and Trie Structure**: 
   - A `TrieNode` class is defined with a `children` dictionary to hold references to other nodes and an indicator (`isEndOfWord`) to show if a word ends at that node.
   - The `Solution` class implements the main logic, with methods to insert words in the Trie and to replace the words in the sentence.

2. **Inserting Words into Trie**:
   - The `insert` method puts each root into the Trie, which sets up efficient prefix searching.

3. **Finding Shortest Roots**:
   - The `findShortestRoot` function checks each character of a word against the Trie. It returns as soon as it encounters a node that indicates the end of a word, thereby ensuring the shortest root is found.

4. **Processing the Sentence**:
   - The sentence is split into words, and `findShortestRoot` is used on each word to replace it with its shortest root if applicable.
   - The modified words are joined back into a string as the final output.

This approach is optimized for querying each word against potentially many roots since the operation of checking for a root prefix in a Trie is efficient, especially with the constraints defined (repeated queries for several words). The initialization of the Trie costs O(N * L) where N is the number of roots and L is the average length of the roots. Each word checking operation, in the worst case, costs O(L) where L is the length of the word.

# 676. Implement Magic Dictionary

### Problem Description 
Design a data structure that is initialized with a list of different words. Provided a string, you should determine if you can change exactly one character in this string to match any word in the data structure.

Implement the `MagicDictionary` class:
`MagicDictionary()` Initializes the object.

`void buildDict(String[] dictionary)` Sets the data structure with an array of distinct strings `dictionary`.

`bool search(String searchWord)` Returns `true` if you can change exactly one character in `searchWord` to match any string in the data structure, otherwise returns `false`.


Example 1:
Input
["MagicDictionary", "buildDict", "search", "search", "search", "search"]
[[], [["hello", "leetcode"]], ["hello"], ["hhllo"], ["hell"], ["leetcoded"]]
Output
[null, null, false, true, false, false]
Explanation
MagicDictionary magicDictionary = new MagicDictionary();
magicDictionary.buildDict(["hello", "leetcode"]);
magicDictionary.search("hello"); // return False
magicDictionary.search("hhllo"); // We can change the second 'h' to 'e' to match "hello" so we return True
magicDictionary.search("hell"); // return False
magicDictionary.search("leetcoded"); // return False

Constraints:
`1 <= dictionary.length <= 100`
`1 <= dictionary[i].length <= 100`
`dictionary[i]` consists of only lower-case English letters.

All the strings in `dictionary` are distinct.

`1 <= searchWord.length <= 100`
`searchWord` consists of only lower-case English letters.

`buildDict` will be called only once before `search`.

At most `100` calls will be made to `search`.

### Solution 
 Below is the Python code that adheres to the defined problem requirements and can be directly used in LeetCode's testing environment. The explanation for the implementation follows after the code snippet.



```python
class MagicDictionary:

    def __init__(self):
        self.words = []

    def buildDict(self, dictionary: List[str]) -> None:
        self.words = dictionary

    def search(self, searchWord: str) -> bool:
        for word in self.words:
            if len(word) == len(searchWord):  # only check words of the same length
                difference_count = 0
                for char1, char2 in zip(word, searchWord):
                    if char1 != char2:
                        difference_count += 1
                        if difference_count > 1:
                            break
                if difference_count == 1:  # found a word similar by one character
                    return True
        return False

# Your MagicDictionary object will be instantiated and called as such:
# obj = MagicDictionary()
# obj.buildDict(dictionary)
# param_2 = obj.search(searchWord)

```

### Explanation of the Design and Functionality:

1. **MagicDictionary Class**:
   - Initializes an empty list called `self.words` that will store the words from the dictionary.

2. **buildDict method**: 
   - Simply takes a list of words and stores it in `self.words`. This setup is run only once per object instantiation, fulfilling the problem's constraint where `buildDict` is called just once before any searches are performed.
   
3. **search method**:
   - Loops through each word in `self.words`.
   - For each word, checks if the word length matches the `searchWord` length; this ensures we are only comparing potential direct replacements of one character.
   - For words with matching lengths, it compares each character in a pair (from both the `word` and `searchWord`):
     - It increments a `difference_count` if it finds characters that do not match.
     - If at any point `difference_count` exceeds 1, it breaks early as changing more than one character isn’t allowed.
   - If we find exactly one character difference (`difference_count == 1`), we return `True`, indicating that we can convert `searchWord` to a dictionary word by changing one character.
   - If the loop finishes without finding any matches, returns `False`.
   
This solution ensures that the checks are efficient by minimizing unnecessary computations like handling word pairs of different lengths or counting mismatches beyond 1 disparity. It effectively solves the problem using straightforward looping and conditions, making it intuitive and appropriate for the given constraints.

# 690. Employee Importance

### Problem Description 
You are given a data structure of employee information, which includes the employee's unique id, their importance value and their direct subordinates' id.

For example, employee 1 is the leader of employee 2, and employee 2 is the leader of employee 3. They have importance value 15, 10 and 5, respectively. Then employee 1 has a data structure like [1, 15, [2]], and employee 2 has [2, 10, [3]], and employee 3 has [3, 5, []]. Note that although employee 3 is also a subordinate of employee 1, the relationship is not direct.

Now given the employee information of a company, and an employee id, you need to return the total importance value of this employee and all their subordinates.


Example 1:
Input: [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1
Output: 11
Explanation:
Employee 1 has importance value 5, and he has two direct subordinates: employee 2 and employee 3. They both have importance value 3. So the total importance value of employee 1 is 5 + 3 + 3 = 11.

Note:
One employee has at most one direct leader and may have several subordinates.

The maximum number of employees won't exceed 2000.

### Solution 
 To solve this problem, we can follow these basic steps:
1. Parse the list of employees into a dictionary where we can easily refer to any employee by their ID.
2. Use a recursive function or an iterative approach with a queue to collect the total importance of the specified employee and all their direct and indirect subordinates.

Here is a Python solution that matches the LeetCode format, and can be directly run on the platform:



```python
"""
# Definition for Employee.
class Employee:
    def __init__(self, id: int, importance: int, subordinates: List[int]):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates

"""

class Solution:
    def getImportance(self, employees: List['Employee'], id: int) -> int:
        from collections import deque
        
        # Create a dictionary to store employees by id:
        emp_map = {employee.id: employee for employee in employees}
        
        # Initialize the total importance and a queue for BFS
        total_importance = 0
        queue = deque([id])
        
        # Perform a level order traversal (BFS) to calculate total importance
        while queue:
            current_id = queue.popleft()
            employee = emp_map[current_id]
            total_importance += employee.importance
            # Add all subordinates to the queue
            for sub_id in employee.subordinates:
                queue.append(sub_id)
        
        return total_importance
        
# Example usage:
# Assuming we have a list 'employees' consisting of Employee instances:
employees = [Employee(1, 5, [2, 3]), Employee(2, 3, []), Employee(3, 3, [])]
sol = Solution()
print(sol.getImportance(employees, 1))  # Output: 11

```

### Explanation:

1. **Data Structure Conversion**:
   We start by converting the list of employees into a dictionary where the key is the employee's ID for quick access. This helps to easily find any employee's information in constant time, which is highly efficient especially when the employee mapping includes up to 2000 entries.

2. **Breadth-First Search**:
   Using a queue initialized with the ID of the employee whose total importance is to be calculated, we implement a Breadth-First Search (BFS). The total importance is accumulated by dequeuing an element (employee's ID), fetching the corresponding Employee object, adding their importance to the total, and enqueuing all of their direct subordinates.

3. **Scalability**:
   This approach guarantees a linear run time complexity with respect to the number of employees because every employee is processed exactly once. By using a queue, we ensure that all indirect subordinates are properly processed.

Modifications might be needed if the function signature provided by the actual LeetCode problem is different from the example above. The Employee class definition is assumed based on the problem description.

# 692. Top K Frequent Words

### Problem Description 
Given a non-empty list of words, return the k most frequent elements.

Your answer should be sorted by frequency from highest to lowest. If two words have the same frequency, then the word with the lower alphabetical order comes first.


Example 1:
Input: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
Output: ["i", "love"]
Explanation: "i" and "love" are the two most frequent words.

    Note that "i" comes before "love" due to a lower alphabetical order.


Example 2:
Input: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
Output: ["the", "is", "sunny", "day"]
Explanation: "the", "is", "sunny" and "day" are the four most frequent words,
    with the number of occurrence being 4, 3, 2 and 1 respectively.

Note:
You may assume k is always valid, 1 ≤ k ≤ number of unique elements.

Input words contain only lowercase letters.
Follow up:
Try to solve it in O(n log k) time and O(n) extra space.

### Solution 
 To solve the problem described above using Python and adhere to the requirements on performance and constraints, we can use a dictionary to count word frequencies and then employ a heap (or priority queue) to extract the top k most frequent words efficiently. 

The Python `collections.Counter` is perfect for counting frequencies, and the `heapq` module (which implements a min-heap) can be leveraged to maintain the top k elements. We will use a min-heap because it allows us to efficiently compare and maintain the smallest element at the root, facilitating an easy way to discard less frequent words while keeping the size of the heap bounded to k.

Here's the complete solution based on these ideas:



```python
import heapq
from collections import Counter

class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        # Step 1: Count frequencies of each word
        count = Counter(words)
        
        # Step 2: Use a min-heap to maintain the top k frequent words
        # Using a heap of tuples (-frequency, word) because Python heapq is a min-heap
        # Negative frequencies make the larger numbers come first
        heap = []
        for word, freq in count.items():
            heapq.heappush(heap, (-freq, word))
            if len(heap) > k:
                heapq.heappop(heap)
        
        # Step 3: Extract the elements from the heap and form the result list
        # We reverse since heap pops out the least element, but we want highest frequency first
        result = []
        while heap:
            result.append(heapq.heappop(heap)[1])
        
        result.reverse()
        return result

```

**Explanation:**
1. **Count Frequencies:** We utilize `Counter` to count the frequencies of words in the list, which is an O(n) operation.
2. **Maintaining a Heap:** We store elements in a min-heap as tuples of (`-frequency`, `word`). Using negative frequencies ensures that higher frequencies have lower values in our min-heap environment. If the heap exceeds size `k`, we pop from the heap, ensuring we always have the top `k` frequent elements. Inserting into the heap and popping from the heap are both O(log k) operations. Therefore, if there are `m` unique words, this step runs in O(m log k).
3. **Building the Result List:** After processing, the top frequencies are in the heap in reverse order, so we reconstruct the result by popping from the heap and reversing the list.

This solution meets the requirements of running in O(n log k) time and uses O(n) space.

# 694. Number of Distinct Islands

### Problem Description 
Given a non-empty 2D array `grid` of 0's and 1's, an island is a group of `1`'s (representing land) connected 4-directionally (horizontal or vertical.)  You may assume all four edges of the grid are surrounded by water.

Count the number of distinct islands.  An island is considered to be the same as another if and only if one island can be translated (and not rotated or reflected) to equal the other.


Example 1:
11000
11000
00011
00011
Given the above grid map, return `1`.


Example 2:
11011
10000
00001
11011
Given the above grid map, return `3`.

Notice that:
11
1
and
 1
11
are considered different island shapes, because we do not consider reflection / rotation.

Note:
The length of each dimension in the given `grid` does not exceed 50.

### Solution 
 To solve the problem of counting distinct islands based on their shapes and positions, we approach by using Depth-First Search (DFS) in combination with hashing techniques to identify and differentiate the shapes of the islands.

The key steps in our solution include:
1. Traverse the 2D grid.
2. For each cell with value `1` that hasn't been visited:
   - Start DFS to explore the full extent of the island and track its shape.
   - Encode the shape during the DFS traversal by recording the relative movement directions and create a unique signature for the island.
3. Use a set to count only distinct shapes for islands since sets automatically handle duplicates.

### Detailed Plan:
1. For each cell in the grid:
   - If the cell contains `1` and hasn't been visited yet, initialize DFS.
   - As the DFS explores, create a unique encoding for the shape based on relative movements (like "R" for right move, "D" for down move, etc.), using a starting point of "0,0" as reference.
   - After exploring the entire island, add the shape encoding to a set.
2. Convert the set’s size to an integer which represents the count of distinct islands.

This unique encoding ensures that identical island shapes under translation will result in the same sequence of relative movements.

### Python Code:


```python
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        visited = set()
        shapes = set()
        
        def dfs(r, c, pos, direction):
            if (r < 0 or r >= rows or c < 0 or c >= cols or
                (r, c) in visited or grid[r][c] == 0):
                return
            visited.add((r, c))
            path.append(direction)
            dfs(r + 1, c, pos, 'D')  # Down
            dfs(r - 1, c, pos, 'U')  # Up
            dfs(r, c + 1, pos, 'R')  # Right
            dfs(r, c - 1, pos, 'L')  # Left
            path.append('B')  # Backtrack, mark end of specific direction
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1 and (r, c) not in visited:
                    path = []
                    dfs(r, c, (r, c), 'O')  # 'O' means origin
                    shapes.add(tuple(path))
        
        return len(shapes)


```

### Explanation:
1. `dfs()` traverses the island using recursive DFS, marking cells as visited and recording each movement's direction.
2. 'O' marks the starting point, each move direction is appended to `path`, and 'B' marks backtracking.
3. After each DFS completion, the path represents the complete shape of the island.
4. This path is converted into a tuple (to make it hashable) and added to the set of shapes.
5. The number of unique shapes in the set is the result.

This solution has a time complexity of \(O(M \times N)\), where \(M\) and \(N\) are the dimensions of the grid, which is efficient given the problem constraints (each cell is visited at most once).


# 705. Design HashSet

### Problem Description 
Design a HashSet without using any built-in hash table libraries.

Implement `MyHashSet` class:
`void add(key)` Inserts the value `key` into the HashSet.

`bool contains(key)` Returns whether the value `key` exists in the HashSet or not.

`void remove(key)` Removes the value `key` in the HashSet. If `key` does not exist in the HashSet, do nothing.


Example 1:
Input
["MyHashSet", "add", "add", "contains", "contains", "add", "contains", "remove", "contains"]
[[], [1], [2], [1], [3], [2], [2], [2], [2]]
Output
[null, null, null, true, false, null, true, null, false]
Explanation
MyHashSet myHashSet = new MyHashSet();
myHashSet.add(1);      // set = [1]
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(1); // return True
myHashSet.contains(3); // return False, (not found)
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(2); // return True
myHashSet.remove(2);   // set = [1]
myHashSet.contains(2); // return False, (already removed)

Constraints:
`0 <= key <= 106`
At most `104` calls will be made to `add`, `remove`, and `contains`.

Follow up: Could you solve the problem without using the built-in HashSet library?

### Solution 
 To implement this `MyHashSet` class, you can use a list-based approach due to its simplicity while ensuring that we respect the constraints. Here’s a concise explanation and implementation:

**Design Choices:**
- **Structure:** We'll use a fixed-size list (array) to represent buckets in the hash set. The size will be slightly more than 10^6 (1,000,003, a prime number) to minimize collision in hashing and ensure distribution.
- **Hash Function:** Use a simple modulo operation for the hash function with the size of the bucket array. This will map keys efficiently to the array indices.
- **Collision Handling:** Since only insertions are involved and we uniquely store elements, we'll maintain a boolean state at each index to indicate if the key exists.

**Methods:**
- `add(key)`: Simply hash the key and mark the corresponding index in the array.
- `remove(key)`: Hash the key and mark the corresponding index as `False`.
- `contains(key)`: Check if the position indicated by the hash of the key is `True`.

Here’s the code:



```python
class MyHashSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.size = 1000003  # Choosing a large prime number to minimize collisions
        self.set = [False] * self.size

    def add(self, key: int) -> None:
        """
        Add the element to the hash set.
        """
        index = key % self.size
        self.set[index] = True

    def remove(self, key: int) -> None:
        """
        Remove the element from the hash set.
        """
        index = key % self.size
        self.set[index] = False

    def contains(self, key: int) -> bool:
        """
        Returns true if the element is in the hash set, otherwise returns false.
        """
        index = key % self.size
        return self.set[index]


# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)

```

**How To Use:**
- Instantiate the `MyHashSet`.
- Use `add` method to add keys.
- Use `remove` method to delete keys.
- Use `contains` to check if a key exists in the set.

The operations `add`, `remove`, and `contains` all operate in constant O(1) time, making it very efficient, especially for the given constraints of up to 10^4 calls. This structure handles basic operations without any built-in hash set utilities.

# 706. Design HashMap

### Problem Description 
Design a HashMap without using any built-in hash table libraries.

Implement the `MyHashMap` class:
`MyHashMap()` initializes the object with an empty map.

`void put(int key, int value)` inserts a `(key, value)` pair into the HashMap. If the `key` already exists in the map, update the corresponding `value`.

`int get(int key)` returns the `value` to which the specified `key` is mapped, or `-1` if this map contains no mapping for the `key`.

`void remove(key)` removes the `key` and its corresponding `value` if the map contains the mapping for the `key`.


Example 1:
Input
["MyHashMap", "put", "put", "get", "get", "put", "get", "remove", "get"]
[[], [1, 1], [2, 2], [1], [3], [2, 1], [2], [2], [2]]
Output
[null, null, null, 1, -1, null, 1, null, -1]
Explanation
MyHashMap myHashMap = new MyHashMap();
myHashMap.put(1, 1); // The map is now [[1,1]]
myHashMap.put(2, 2); // The map is now [[1,1], [2,2]]
myHashMap.get(1);    // return 1, The map is now [[1,1], [2,2]]
myHashMap.get(3);    // return -1 (i.e., not found), The map is now [[1,1], [2,2]]
myHashMap.put(2, 1); // The map is now [[1,1], [2,1]] (i.e., update the existing value)
myHashMap.get(2);    // return 1, The map is now [[1,1], [2,1]]
myHashMap.remove(2); // remove the mapping for 2, The map is now [[1,1]]
myHashMap.get(2);    // return -1 (i.e., not found), The map is now [[1,1]]

Constraints:
`0 <= key, value <= 106`
At most `104` calls will be made to `put`, `get`, and `remove`.

Follow up: Please do not use the built-in HashMap library.

### Solution 
 The problem is to design a simple hash map without using built-in hash table libraries. We'll implement this by using a list of buckets, where each bucket contains a list of `(key, value)` pairs. This way, it enables handling of hash collisions through separate chaining.

Here's a step-by-step breakdown of our custom `MyHashMap` class:
1. **Initialization (`__init__`)**: Create a fixed array (`self.buckets`) to store the data. Each `bucket` in this array will hold lists representing `(key, value)` pairs.
   
2. **Hashing Function (`_hash`)**: This function will map keys to bucket indices in our array. A simple modulus operation with the number of available buckets ensures that the result stays within limits.
   
3. **Insert (`put`)**: Locate the right bucket using its hash index. If the key exists, update its value; otherwise, add the `(key, value)` pair to the bucket.
   
4. **Retrieve (`get`)**: Fetch the bucket using the hash value of the key and search for the key in the bucket. If found, return its value; if not, return -1.
   
5. **Remove (`remove`)**: Locate the bucket using its hash index and remove the `(key, value)` tuple from that bucket.

This implementation balances simplicity and efficiency by utilizing separate chaining, a common collision resolution technique in hash tables.



```python
class MyHashMap:
    def __init__(self):
        self.size = 1000
        self.buckets = [[] for _ in range(self.size)]

    def _hash(self, key):
        return key % self.size

    def put(self, key, value):
        hash_index = self._hash(key)
        bucket = self.buckets[hash_index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)  # Update existing key
                return
        bucket.append((key, value))  # Add new key-value pair

    def get(self, key):
        hash_index = self._hash(key)
        bucket = self.buckets[hash_index]
        for (k, v) in bucket:
            if k == key:
                return v
        return -1

    def remove(self, key):
        hash_index = self._hash(key)
        bucket = self.buckets[hash_index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]  # Remove the key-value pair by index
                break

# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)

```

Explanation by calls:
- `MyHashMap()`: Initialize the MyHashMap object.
- `put(key, value)`: Insert or update the `(key, value)` pair.
- `get(key)`: Retrieve the value associated with the key.
- `remove(key)`: Remove the key (and its corresponding value) if it exists.

This implementation will not encounter memory issues with the constraints provided (i.e., 0 <= key, value <= 10^6 with at most 10^4 operations) since each operation processes efficiently by accessing a specific bucket using hashing.

# 710. Random Pick with Blacklist

### Problem Description 
Given a blacklist `B` containing unique integers from `[0, N)`, write a function to return a uniform random integer from `[0, N)` which is NOT in `B`.

Optimize it such that it minimizes the call to system’s `Math.random()`.

Note:
`1 <= N <= 1000000000`
`0 <= B.length < min(100000, N)`
`[0, N)` does NOT include N. See interval notation.


Example 1:
Input: 
["Solution","pick","pick","pick"]
[[1,[]],[],[],[]]
Output: [null,0,0,0]

Example 2:
Input: 
["Solution","pick","pick","pick"]
[[2,[]],[],[],[]]
Output: [null,1,1,1]

Example 3:
Input: 
["Solution","pick","pick","pick"]
[[3,[1]],[],[],[]]
Output: [null,0,0,2]

Example 4:
Input: 
["Solution","pick","pick","pick"]
[[4,[2]],[],[],[]]
Output: [null,1,3,1]
Explanation of Input Syntax:
The input is two lists: the subroutines called and their arguments. `Solution`'s constructor has two arguments, `N` and the blacklist `B`. `pick` has no arguments. Arguments are always wrapped with a list, even if there aren't any.

### Solution 
 To solve this problem efficiently, we use a strategy that preprocesses the blacklist to remap blacklisted numbers to non-blacklisted numbers within the same range `[0, N)`:

1. **Preprocessing (during object initialization)**:
   - Compute the number of valid numbers that are not blacklisted (`M = N - len(B)`).
   - Create a mapping from blacklisted numbers to valid (non-blacklisted) numbers that exist in the upper region `N - len(B) to N-1`.
   - The key idea is to treat the numbers from `0` to `M-1` (inclusive) as a contiguous valid region. Any number from the blacklist `B` that falls into this region is remapped to a non-blacklisted number in the top portion `[M, N)` of the range.

2. **Random Picking**:
   - Generate a random integer in `[0, M)` using `random.randint(0, M-1)`.
   - Return the number directly if it's not in the blacklist; if it is in the blacklist, return its mapped value from our preprocessed mapping.

The solution ensures that each call to `pick()` only requires O(1) time after the initial O(B) preprocessing step, where `B` is the length of the blacklist. 

Here's the implementation in Python:



```python
import random

class Solution:

    def __init__(self, N: int, blacklist: List[int]):
        self.M = N - len(blacklist)
        self.mapping = {}
        
        # Set of numbers that should be remapped (blacklist numbers in the range [0, M-1])
        blacklisted_in_lower = {b for b in blacklist if b < self.M}
        # Non-blacklisted numbers in the upper part of the range
        non_blacklisted_in_upper = set(range(self.M, N)) - set(blacklist)
        
        # Iterator of the non-blacklisted above M
        non_blacklisted_iter = iter(non_blacklisted_in_upper)

        # Create the mapping for blacklisted numbers to valid upper range numbers
        for b in blacklisted_in_lower:
            self.mapping[b] = next(non_blacklisted_iter)

    def pick(self) -> int:
        rand = random.randint(0, self.M - 1)
        # If the random number is a blacklisted one, map it, otherwise return as is
        return self.mapping.get(rand, rand)

# Your Solution object will be instantiated and called as such:
# obj = Solution(N, blacklist)
# param_1 = obj.pick()

```

This LeetCode style definition ensures that the constructor (`__init__`) processes the blacklist to minimize the random selection logic overhead during multiple calls to `pick()` method. Each `pick()` invokes the `random.randint()` function just once and uses efficient lookup in a map to return a valid result.

# 711. Number of Distinct Islands II

### Problem Description 
Given a non-empty 2D array `grid` of 0's and 1's, an island is a group of `1`'s (representing land) connected 4-directionally (horizontal or vertical.)  You may assume all four edges of the grid are surrounded by water.

Count the number of distinct islands.  An island is considered to be the same as another if they have the same shape, or have the same shape after rotation (90, 180, or 270 degrees only) or reflection (left/right direction or up/down direction).


Example 1:
11000
10000
00001
00011
Given the above grid map, return `1`.

Notice that:
11
1
and
 1
11
are considered same island shapes. Because if we make a 180 degrees clockwise rotation on the first island, then two islands will have the same shapes.


Example 2:
11100
10001
01001
01110
Given the above grid map, return `2`.

Here are the two distinct islands:
111
1
and
1
1
Notice that:
111
1
and
1
111
are considered same island shapes. Because if we flip the first array in the up/down direction, then they have the same shapes.

Note:
The length of each dimension in the given `grid` does not exceed 50.

### Solution 
 To solve this problem, we'll utilize Depth-First Search (DFS) to explore each island in the grid, and then normalize each discovered island to a canonical form that accounts for all possible rotations and reflections. This canonical form will then be stored in a set to ensure that we count only unique island shapes.

### Detailed Steps:
1. **Exploration**: Use DFS to explore each cell that forms part of an island (i.e., cells with a `1`) and mark cells as visited.
2. **Shape Representation**: For each found island, record its structure in a local grid where its top-left corner is the grid's origin. This is achieved by keeping track of relative coordinates during DFS.
3. **Normalization**: Create canonical forms for each island by generating all possible rotations and reflections, and selecting the smallest lexicographical one (after sorting).
4. **Unique Counting**: Use a set to count and keep track of each unique island shape by storing their canonical forms.

### Implementation:


```python
class Solution:
    def numDistinctIslands2(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        
        n, m = len(grid), len(grid[0])
        
        def dfs(x, y):
            # Record relative position and mark as visited
            grid[x][y] = 0
            island.append((x - base_x, y - base_y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and grid[nx][ny] == 1:
                    dfs(nx, ny)

        def normalize(island):
            """ Normalize island by considering all transformations. """
            transforms = []
            for x, y in island:
                # Generate all possible rotations and reflections:
                transforms.append([
                    (x, y), (y, -x), (-x, -y), (-y, x), # 90, 180, 270 degree rotations
                    (x, -y), (-x, y), (-y, -x), (y, x) # reflections
                ])
            # Normalize each transformation by sorting and shifting to origin (minimum coordinates)
            for i in range(8):
                transforms[i].sort()
                dx, dy = transforms[i][0]
                for j in range(len(transforms[i])):
                    transforms[i][j] = (transforms[i][j][0] - dx, transforms[i][j][1] - dy)
                transforms[i] = tuple(transforms[i])

            # Return lexicographically smallest transformation as the representative
            return min(transforms)

        unique_islands = set()
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    island = []
                    base_x, base_y = i, j
                    dfs(i, j)
                    if island:
                        canonical_form = normalize(island)
                        unique_islands.add(canonical_form)

        return len(unique_islands)


``` 

### Explanation:
- **Depth-First Search (DFS)**: `dfs()` explores each connected component (island) and collects all cell positions relative to the initial cell (`base_x`, `base_y`).
- **Normalization**: Through several rotations and reflections, we generate all possible orientations of an island and transform them to their simplest form, starting from the top-left corner.
- **Canonical Form Selection**: We choose the lexicographically smallest form as the representative of that island for uniqueness checking.
- **Unique Counting**: We use a Python set to store and count unique islands based on their canonical forms.

This solution efficiently identifies and counts unique islands considering all required transformations, and it fits within the computational limits provided by the problem (grid size up to 50x50).

# 718. Maximum Length of Repeated Subarray

### Problem Description 
Given two integer arrays `A` and `B`, return the maximum length of an subarray that appears in both arrays.


Example 1:
Input:
A: [1,2,3,2,1]
B: [3,2,1,4,7]
Output: 3
Explanation: 
The repeated subarray with maximum length is [3, 2, 1].

Note:
1 <= len(A), len(B) <= 1000
0 <= A[i], B[i] < 100

### Solution 
 In this problem, we want to find the maximum length of a subarray that's common to both input arrays, A and B.

To solve this, we can utilize a dynamic programming approach. The idea is to use a 2D array `dp` where `dp[i][j]` holds the length of the longest common ending subarray between A and B such that it involves `A[i-1]` and `B[j-1]`. If the elements `A[i-1]` and `B[j-1]` are the same, then we increment the value of the previous calculated subarray length `dp[i-1][j-1]` by 1. If the elements are different, the incoming value in `dp[i][j]` will be initialized with zero because continuity breaks.

The solution will iterate through both of the arrays and fill the `dp` table accordingly. The answer will be the maximum value found in `dp`. Additionally, to avoid index errors and to carry out the first comparison correctly, the `dp` array is defined with extra rows and columns (`len(A) + 1`, `len(B) + 1`).

Here's how the code can be formatted:



```python
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        # Initialize the DP table
        # (len(A) + 1) x (len(B) + 1) and fill it with zeros
        m, n = len(A), len(B)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # If elements match, increment the value from the diagonal
                if A[i-1] == B[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    max_length = max(max_length, dp[i][j])
        
        # Return the maximum length of common subarray
        return max_length


```

In this code, we iterate through every combination of indices `i` and `j` from 1 to `len(A)` and `len(B)` respectively. Whenever we find that `A[i-1]` is equivalent to `B[j-1]`, we update `dp[i][j]` to `dp[i-1][j-1] + 1`. This way we extend the length of the current matching subarray by 1 from the previous matching subarray's length. We keep tracking the maximum length found in `max_length`.

This solution works in \(O(m \cdot n)\) time complexity where \(m\) and \(n\) are the lengths of arrays `A` and `B`, respectively. The space complexity is also \(O(m \cdot n)\) due to the `dp` table. For typical LeetCode constraints, this approach is efficient.


# 720. Longest Word in Dictionary

### Problem Description 
Given an array of strings `words` representing an English Dictionary, return the longest word in `words` that can be built one character at a time by other words in `words`.

If there is more than one possible answer, return the longest word with the smallest lexicographical order. If there is no answer, return the empty string.


Example 1:
Input: words = ["w","wo","wor","worl","world"]
Output: "world"
Explanation: The word "world" can be built one character at a time by "w", "wo", "wor", and "worl".


Example 2:
Input: words = ["a","banana","app","appl","ap","apply","apple"]
Output: "apple"
Explanation: Both "apply" and "apple" can be built from other words in the dictionary. However, "apple" is lexicographically smaller than "apply".


Constraints:
`1 <= words.length <= 1000`
`1 <= words[i].length <= 30`
`words[i]` consists of lowercase English letters.

### Solution 
 To solve the problem of determining the longest word that can be built one character at a time by other words in an array of strings, we can use the following approach:

1. **Data Structures Used**:
    - **Set**: To keep the words for quick membership checking, which helps in determining whether a prefix of a word (built one character at a time) exists in the `words`.
    - **Sorting**: To look for the longest word and maintain the correct order when two words are of the same length.

2. **Steps**:
    - Firstly, sort the `words` list. Sorting helps in two ways:
        - Words that can possibly be built are checked first since shorter words appear before their longer versions.
        - It ensures that any conflicts (words of the same length) are resolved lexicographically automatically because of the order in the sorted array.
    - Initialize an empty set, named `built`.
    - Initialize a variable `longest_word` to store the longest word found that meets the required condition.
    - Traverse each word in the sorted `words` list:
        - For each word, check whether all prefixes of the word (excluding the full word) are present in the `built` set.
        - If they are, update the `longest_word` if the current word is longer than the previously stored `longest_word`.
        - Add the current word to the `built` set.
    - Return `longest_word` after traversing all words.

Here's the code implementation:



```python
class Solution:
    def longestWord(self, words: List[str]) -> str:
        words.sort()  # Sort to ensure lexicographic order and ease of prefix handling
        longest_word = ""
        built = set()  # To store words that can be built
        
        for word in words:
            # Check if all previous characters form a word in the set
            if len(word) == 1 or word[:-1] in built:
                # If it's longer or the first valid word found, update longest_word
                if len(word) > len(longest_word):
                    longest_word = word
                # Add the current word to set, as it can serve as a prefix for longer words
                built.add(word)
        
        return longest_word

```

**Explanation and Complexity**:
- **Time Complexity**: Sorting the list takes \(O(N \log N)\) where \(N\) is the number of words. Checking each word in the list is \(O(N)\) and for each word, in worst case, checking all its prefixes takes \(O(L)\) where \(L\) is the maximum length of a word. Thus, the more detailed time complexity would be \(O(N \log N + N \cdot L)\).
- **Space Complexity**: Uses \(O(N \cdot L)\) space due to storing all words in the set, and additional space for the sorted list. The storage of the longest word is \(O(L)\).

Using the given constraints, this approach is computationally feasible.

# 726. Number of Atoms

### Problem Description 
Given a chemical `formula` (given as a string), return the count of each atom.

The atomic element always starts with an uppercase character, then zero or more lowercase letters, representing the name.

One or more digits representing that element's count may follow if the count is greater than 1. If the count is 1, no digits will follow. For example, H2O and H2O2 are possible, but H1O2 is impossible.

Two formulas concatenated together to produce another formula. For example, H2O2He3Mg4 is also a formula.

A formula placed in parentheses, and a count (optionally added) is also a formula. For example, (H2O2) and (H2O2)3 are formulas.

Given a `formula`, return the count of all elements as a string in the following form: the first name (in sorted order), followed by its count (if that count is more than 1), followed by the second name (in sorted order), followed by its count (if that count is more than 1), and so on.


Example 1:
Input: formula = "H2O"
Output: "H2O"
Explanation: The count of elements are {'H': 2, 'O': 1}.


Example 2:
Input: formula = "Mg(OH)2"
Output: "H2MgO2"
Explanation: The count of elements are {'H': 2, 'Mg': 1, 'O': 2}.


Example 3:
Input: formula = "K4(ON(SO3)2)2"
Output: "K4N2O14S4"
Explanation: The count of elements are {'K': 4, 'N': 2, 'O': 14, 'S': 4}.


Example 4:
Input: formula = "Be32"
Output: "Be32"

Constraints:
`1 <= formula.length <= 1000`
`formula` consists of English letters, digits, `'('`, and `')'`.

`formula` is always valid.

### Solution 
 To solve this problem, we will use a stack to handle nested parentheses and a dictionary to store the count of each element. The process will involve parsing the chemical formula string character by character, and applying different rules based on whether we encounter an element, a digit, or parentheses. 

Using a stack helps manage complexities arising from nested formulae, such as in the example "K4(ON(SO3)2)2". Each time a nested formula is completed (when we encounter ')'), we pop the stack to integrate counts of elements from the inner formula into an outer level or the overall count.

Here's a step-by-step breakdown of the solution:

1. **Initialize** a stack with an empty dictionary.
2. **Iterate** through each character of the formula.
   - If it's an uppercase letter, it marks the start of a new element. Capture subsequent lowercase letters to complete the element's name.
   - If it's a digit following an element or a closing parenthesis, discern the number and update the last element or apply the multiplier to elements in the recently closed group.
   - On encountering '(', push a new dictionary onto the stack to start counting a nested formula.
   - On encountering ')', pop the stack count and multiply it by the number following the parenthesis if any, and merge it back into the previous or the overall dictionary.
3. **Aggregation**: After processing the entire formula, the stack will contain a single dictionary with the total counts of each element.
4. **Formatting**: Sort the elements alphabetically, and format the output string by appending each element followed by its count (if more than 1).

By handling digits, parentheses, and element names judiciously, this approach is efficient and robust for parsing nested chemical formulae.

Here's the actual Python code reflecting this general strategy:



```python
class Solution:
    def countOfAtoms(self, formula: str) -> str:
        stack = [{}]
        n = len(formula)
        i = 0
        
        while i < n:
            if formula[i] == '(':
                stack.append({})
                i += 1
            elif formula[i] == ')':
                i += 1
                # Multiplying factor
                j = i
                while j < n and formula[j].isdigit():
                    j += 1
                mult = int(formula[i:j] or 1)
                i = j
                # Pop the top dictionary and integrate into the next top
                top = stack.pop()
                for key in top:
                    if key in stack[-1]:
                        stack[-1][key] += top[key] * mult
                    else:
                        stack[-1][key] = top[key] * mult
            else:
                # Parsing the element name
                j = i + 1
                while j < n and formula[j].islower():
                    j += 1
                element = formula[i:j]
                i = j
                # Parsing the count of the element
                j = i
                while j < n and formula[j].isdigit():
                    j += 1
                count = int(formula[i:j] or 1)
                i = j
                # Add to the current dictionary in stack
                if element in stack[-1]:
                    stack[-1][element] += count
                else:
                    stack[-1][element] = count
        
        # Only one item will remain in the stack which is the overarching dictionary
        result_elements = sorted(stack[-1])
        result = ""
        for element in result_elements:
            result += element + (str(stack[-1][element]) if stack[-1][element] > 1 else "")
        return result

# Example usage:
solution = Solution()
print(solution.countOfAtoms("Mg(OH)2"))  # Output: "H2MgO2"
print(solution.countOfAtoms("K4(ON(SO3)2)2"))  # Output: "K4N2O14S4"

```

This code leverages the stack for depth management in nested formula parsing, while dictionary operations handle merging and counting efficiently. It handles all constraints and complexities as described in the problem statement.

# 734. Sentence Similarity

### Problem Description 
We can represent a sentence as an array of words, for example, the sentence `"I am happy with leetcode"` can be represented as `arr = ["I","am",happy","with","leetcode"]`.

Given two sentences `sentence1` and `sentence2` each represented as a string array and given an array of string pairs `similarPairs` where `similarPairs[i] = [xi, yi]` indicates that the two words `xi` and `yi` are similar.

Return `true` if `sentence1` and `sentence2` are similar, or `false` if they are not similar.

Two sentences are similar if:
They have the same length (i.e. the same number of words)
`sentence1[i]` and `sentence2[i]` are similar.

Notice that a word is always similar to itself, also notice that the similarity relation is not transitive. For example, if the words `a` and `b` are similar and the words `b` and `c` are similar, `a` and `c` are not necessarily similar.


Example 1:
Input: sentence1 = ["great","acting","skills"], sentence2 = ["fine","drama","talent"], similarPairs = [["great","fine"],["drama","acting"],["skills","talent"]]
Output: true
Explanation: The two sentences have the same length and each word i of sentence1 is also similar to the corresponding word in sentence2.


Example 2:
Input: sentence1 = ["great"], sentence2 = ["great"], similarPairs = []
Output: true
Explanation: A word is similar to itself.


Example 3:
Input: sentence1 = ["great"], sentence2 = ["doubleplus","good"], similarPairs = [["great","doubleplus"]]
Output: false
Explanation: As they don't have the same length, we return false.


Constraints:
`1 <= sentence1.length, sentence2.length <= 1000`
`1 <= sentence1[i].length, sentence2[i].length <= 20`
`sentence1[i]` and `sentence2[i]` consist of lower-case and upper-case English letters.

`0 <= similarPairs.length <= 1000`
`similarPairs[i].length == 2`
`1 <= xi.length, yi.length <= 20`
`xi` and `yi` consist of lower-case and upper-case English letters.

All the pairs `(xi, yi)` are distinct.

### Solution 
 Below is the Python code snippet that checks if two sentences are similar based on provided word pairs. The provided solution utilizes a dictionary to map each word to its similar words, establishing easy lookup checks for similarity.

The solution outlines:
1. Building a similarity dictionary from the `similarPairs`.
2. Checking the length of the sentences first; if they differ, they are immediately dissimilar.
3. Checking each word's similarity by considering direct similarity from sentence1 to sentence2 using the similarity dictionary.

Here's the implementation:



```python
from typing import List

class Solution:
    def areSentencesSimilar(self, sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:
        if len(sentence1) != len(sentence2):
            return False
        
        # Create a dictionary to store similar words
        similar_words = {}
        for x, y in similarPairs:
            if x not in similar_words:
                similar_words[x] = set()
            if y not in similar_words:
                similar_words[y] = set()
            similar_words[x].add(y)
            similar_words[y].add(x)

        # Check each corresponding words in sentence1 and sentence2 for similarity
        for w1, w2 in zip(sentence1, sentence2):
            if w1 != w2 and \
               w2 not in similar_words.get(w1, set()) and \
               w1 not in similar_words.get(w2, set()):
                return False

        return True

# Usage example
sol = Solution()
sentence1 = ["great", "acting", "skills"]
sentence2 = ["fine", "drama", "talent"]
similarPairs = [["great", "fine"], ["drama", "acting"], ["skills", "talent"]]
output = sol.areSentencesSimilar(sentence1, sentence2, similarPairs)
print(output)  # Expected output: True

```

### Explanation:

1. **Dictionary Initialization**: We initialize a dictionary `similar_words` to store all the words that are declared similar via the `similarPairs`. Words are used as keys, and sets of similar words as values to ensure fast look-up and prevent duplications.

2. **Sentence Length Check**: We first check if the length of `sentence1` and `sentence2` is the same. If they are not, we immediately return `False`.

3. **Similarity Enumeration**: For each pair of words from both sentences, we check:
   - If they are equal, they are trivially similar.
   - If they are not equal, both words are verified against the `similar_words` dictionary to check if one word is included in the similar set of the other.
   - If neither of these conditions is true for any word-pair, the function returns `False`.
   
4. **Result**: If all checks pass, the sentences are similar, and thus return `True`.

This solution efficiently addresses the similarity check in O(N + P) time, where N is the number of words in the sentences, and P is the number of similar pairs provided, making it suitable for the given constraints.

# 739. Daily Temperatures

### Problem Description 
Given a list of daily temperatures `T`, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature.  If there is no future day for which this is possible, put `0` instead.

For example, given the list of temperatures `T = [73, 74, 75, 71, 69, 72, 76, 73]`, your output should be `[1, 1, 4, 2, 1, 1, 0, 0]`.

Note:
The length of `temperatures` will be in the range `[1, 30000]`.

Each temperature will be an integer in the range `[30, 100]`.

### Solution 
 To solve the above problem efficiently, we can utilize a stack data structure to keep track of the indices of days with their corresponding temperatures which are yet to find a warmer day. The main idea behind the stack approach is that it allows us to handle each temperature only once or twice (once when inserting into the stack and once when removing from the stack) leading to an O(N) time complexity solution, which is much more desirable than the O(N^2) naive solution (checking future days for each day).

Here's a step-by-step breakdown of the method using a stack:

1. **Initialize**: Create a stack to store the indices of the temperatures and an array to store the result initialized to zeros (since the default is 0 days until a warmer temperature).
2. **Iterate through each day**:
    - While the stack is not empty and the current day's temperature is greater than the temperature at the index provided by the top of the stack:
        - Pop the index from the stack, this will be the index of a day with a colder temperature.
        - Compute the number of days until a warmer temp (current index - popped index) and update this in the result array.
    - Push the current index onto the stack.
3. **Return result**: By the end of the iteration, for indices still in the stack, the result array already has the correct value (`0`), indicating no future day is warmer.

This approach ensures each element is pushed and popped from the stack exactly once, yielding a time complexity of O(N).

Let's implement this approach as a `Solution` class as per the expected format for LeetCode submissions, allowing us to run it directly on the LeetCode website:



```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        result = [0] * len(temperatures) # Initialize result array with all zeros
        stack = [] # Create a stack to keep track of indices with unresolved temperatures

        for i, temp in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < temp:
                # Pop index of the day with colder temperature
                idx = stack.pop()
                # Calculate how many days until this warmer day
                result[idx] = i - idx 
            # Push current day's index onto the stack
            stack.append(i)
        
        # The stack left with indices should naturally have result values as 0 (since no warmer day exists)
        return result

# Example of running this:
# sol = Solution()
# print(sol.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73])) # Output: [1, 1, 4, 2, 1, 1, 0, 0]

```
The implementation should work efficiently for the problem constraints and give the expected results.

# 748. Shortest Completing Word

### Problem Description 
Given a string `licensePlate` and an array of strings `words`, find the shortest completing word in `words`.

A completing word is a word that contains all the letters in `licensePlate`. Ignore numbers and spaces in `licensePlate`, and treat letters as case insensitive. If a letter appears more than once in `licensePlate`, then it must appear in the word the same number of times or more.

For example, if `licensePlate`` = "aBc 12c"`, then it contains letters `'a'`, `'b'` (ignoring case), and `'c'` twice. Possible completing words are `"abccdef"`, `"caaacab"`, and `"cbca"`.

Return the shortest completing word in `words`. It is guaranteed an answer exists. If there are multiple shortest completing words, return the first one that occurs in `words`.


Example 1:
Input: licensePlate = "1s3 PSt", words = ["step","steps","stripe","stepple"]
Output: "steps"
Explanation: licensePlate contains letters 's', 'p', 's' (ignoring case), and 't'.

"step" contains 't' and 'p', but only contains 1 's'.

"steps" contains 't', 'p', and both 's' characters.

"stripe" is missing an 's'.

"stepple" is missing an 's'.

Since "steps" is the only word containing all the letters, that is the answer.


Example 2:
Input: licensePlate = "1s3 456", words = ["looks","pest","stew","show"]
Output: "pest"
Explanation: licensePlate only contains the letter 's'. All the words contain 's', but among these "pest", "stew", and "show" are shortest. The answer is "pest" because it is the word that appears earliest of the 3.


Example 3:
Input: licensePlate = "Ah71752", words = ["suggest","letter","of","husband","easy","education","drug","prevent","writer","old"]
Output: "husband"

Example 4:
Input: licensePlate = "OgEu755", words = ["enough","these","play","wide","wonder","box","arrive","money","tax","thus"]
Output: "enough"

Example 5:
Input: licensePlate = "iMSlpe4", words = ["claim","consumer","student","camera","public","never","wonder","simple","thought","use"]
Output: "simple"

Constraints:
`1 <= licensePlate.length <= 7`
`licensePlate` contains digits, letters (uppercase or lowercase), or space `' '`.

`1 <= words.length <= 1000`
`1 <= words[i].length <= 15`
`words[i]` consists of lower case English letters.

### Solution 
 Here's a Python solution that adheres to the LeetCode format. This implementation directly processes the input `licensePlate` to count the frequency of each letter (in a case-insensitive manner) and compares it against each word in `words` to determine if it is a completing word.



```python
from collections import Counter
import re

class Solution:
    def shortestCompletingWord(self, licensePlate: str, words: list[str]) -> str:
        # Create a clean version of the license plate containing only alphabetic characters,
        # converted to lowercase.
        clean_plate = re.sub('[^a-z]', '', licensePlate.lower())
        
        # Count the frequency of each character in the clean license plate.
        plate_count = Counter(clean_plate)
        
        # Initialize answer variables
        shortest_word = None
        
        # Iterate over each word in the provided list.
        for word in words:
            # Count the frequency of each character in the current word.
            word_count = Counter(word)
            
            # Check if the word is a completing word
            is_completing = True
            for char, count in plate_count.items():
                if word_count[char] < count:
                    is_completing = False
                    break
            
            # If it's a completing word, check if it's the shortest one found so far.
            if is_completing:
                if shortest_word is None or len(word) < len(shortest_word):
                    shortest_word = word

        return shortest_word

# Example usage
sol = Solution()
print(sol.shortestCompletingWord("1s3 PSt", ["step","steps","stripe","stepple"]))  # Output: "steps"
print(sol.shortestCompletingWord("1s3 456", ["looks","pest","stew","show"]))       # Output: "pest"
print(sol.shortestCompletingWord("Ah71752", ["suggest","letter","of","husband","easy","education","drug","prevent","writer","old"])) # Output: "husband"

```

### Explanation:

1. **Ignore Non-Alphabetic Characters:** The first step is to strip any non-alphabetic characters (including spaces and numbers) from `licensePlate` using a regular expression. This is done to focus only on the letters which are important for matching.

2. **Case-Insensitive Matching:** We use Python's `.lower()` method to standardize `licensePlate` to lowercase ensuring that we are doing case-insensitive comparison.

3. **Counting Letters:** The `Counter` from the Python `collections` module is used to count the frequency of each letter in the cleaned-up `licensePlate`. This frequency map is then used to determine if a word from `words` is a completing word.

4. **Check Completing Word:** For each word in `words`, we re-generate its frequency count using `Counter` and then compare each letter's count against the `licensePlate` counts. If a word's letter count meets or exceeds the `licensePlate`'s count for all required letters, it is a valid completing word.

5. **Determine Shortest Completing Word:** If a word meets the criteria for a completing word, we then check if it's the shortest such word we've encountered. If so, we update our best candidate.

Thus, the function processes each word individually and checks if it can fully represent all the characters in `licensePlate` with at least the same frequency. The shortest word that meets all these conditions is returned at the end.

# 760. Find Anagram Mappings

### Problem Description 
Given two lists `A`and `B`, and `B` is an anagram of `A`. `B` is an anagram of `A` means `B` is made by randomizing the order of the elements in `A`.

We want to find an index mapping `P`, from `A` to `B`. A mapping `P[i] = j` means the `i`th element in `A` appears in `B` at index `j`.

These lists `A` and `B` may contain duplicates.  If there are multiple answers, output any of them.

For example, given
A = [12, 28, 46, 32, 50]
B = [50, 12, 32, 46, 28]
We should return
[1, 4, 3, 2, 0]
as `P[0] = 1` because the `0`th element of `A` appears at `B[1]`,
and `P[1] = 4` because the `1`st element of `A` appears at `B[4]`,
and so on.

Note:
`A, B` have equal lengths in range `[1, 100]`.

`A[i], B[i]` are integers in range `[0, 10^5]`.

### Solution 
 Here is a Python code solution that can be run directly on the LeetCode platform to solve the given problem:



```python
from typing import List
from collections import defaultdict, deque

class Solution:
    def anagramMappings(self, A: List[int], B: List[int]) -> List[int]:
        # Create a dictionary to keep track of the indices of each element in B
        index_map = defaultdict(deque)
        for index, value in enumerate(B):
            index_map[value].append(index)
        
        # Prepare the result array to store the mappings
        mapping = []
        
        # For each element in A, find the corresponding index in B using index_map
        for value in A:
            mapping.append(index_map[value].popleft())
        
        return mapping

# Example usage
sol = Solution()
A = [12, 28, 46, 32, 50]
B = [50, 12, 32, 46, 28]
print(sol.anagramMappings(A, B))  # Output: [1, 4, 3, 2, 0]

```

### Explanation:

1. **Dictionary Storage Using `defaultdict`**: The variable `index_map` is a dictionary where each key is an integer from the list `B`, and the value is a deque containing all the indices where the integer appears in `B`. The use of `deque` facilitates efficient O(1) time complexity for both appending and popping elements.

2. **Mapping Creation**:
    - We iterate through each element in list `A`.
    - For each element, we use its value to look up the `index_map` and pop the leftmost index from the corresponding deque. This ensures that each index is used exactly once, even if the element appears multiple times in the array.
    - The popped index is appended to the `mapping` list, which will eventually contain the entire mapping from `A` to `B`.

3. **Return the mapping**: The list `mapping` contains the result after processing all the elements in `A`.

### Performance:
- **Time Complexity**: O(N) for constructing the `index_map` and O(N) for creating the mapping, where N is the length of the lists, leading to an overall O(N).
- **Space Complexity**: O(N) for storing the `index_map` and the `mapping` list.

This solution works for all scenarios, including when there are duplicates in the lists `A` and `B`.

# 770. Basic Calculator IV

### Problem Description 
Given an `expression` such as `expression = "e + 8 - a + 5"` and an evaluation map such as `{"e": 1}` (given in terms of `evalvars = ["e"]` and `evalints = [1]`), return a list of tokens representing the simplified expression, such as `["-1*a","14"]`
An expression alternates chunks and symbols, with a space separating each chunk and symbol.

A chunk is either an expression in parentheses, a variable, or a non-negative integer.

A variable is a string of lowercase letters (not including digits.) Note that variables can be multiple letters, and note that variables never have a leading coefficient or unary operator like `"2x"` or `"-x"`.

Expressions are evaluated in the usual order: brackets first, then multiplication, then addition and subtraction. For example, `expression = "1 + 2 * 3"` has an answer of `["7"]`.

The format of the output is as follows:
For each term of free variables with non-zero coefficient, we write the free variables within a term in sorted order lexicographically. For example, we would never write a term like `"b*a*c"`, only `"a*b*c"`.

Terms have degree equal to the number of free variables being multiplied, counting multiplicity. (For example, `"a*a*b*c"` has degree 4.) We write the largest degree terms of our answer first, breaking ties by lexicographic order ignoring the leading coefficient of the term.

The leading coefficient of the term is placed directly to the left with an asterisk separating it from the variables (if they exist.)  A leading coefficient of 1 is still printed.

An example of a well formatted answer is `["-2*a*a*a", "3*a*a*b", "3*b*b", "4*a", "5*c", "-6"]` 
Terms (including constant terms) with coefficient 0 are not included.  For example, an expression of "0" has an output of [].


Examples:
Input: expression = "e + 8 - a + 5", evalvars = ["e"], evalints = [1]
Output: ["-1*a","14"]
Input: expression = "e - 8 + temperature - pressure",
evalvars = ["e", "temperature"], evalints = [1, 12]
Output: ["-1*pressure","5"]
Input: expression = "(e + 8) * (e - 8)", evalvars = [], evalints = []
Output: ["1*e*e","-64"]
Input: expression = "7 - 7", evalvars = [], evalints = []
Output: []
Input: expression = "a * b * c + b * a * c * 4", evalvars = [], evalints = []
Output: ["5*a*b*c"]
Input: expression = "((a - b) * (b - c) + (c - a)) * ((a - b) + (b - c) * (c - a))",
evalvars = [], evalints = []
Output: ["-1*a*a*b*b","2*a*a*b*c","-1*a*a*c*c","1*a*b*b*b","-1*a*b*b*c","-1*a*b*c*c","1*a*c*c*c","-1*b*b*b*c","2*b*b*c*c","-1*b*c*c*c","2*a*a*b","-2*a*a*c","-2*a*b*b","2*a*c*c","1*b*b*b","-1*b*b*c","1*b*c*c","-1*c*c*c","-1*a*a","1*a*b","1*a*c","-1*b*c"]
Note:
`expression` will have length in range `[1, 250]`.

`evalvars, evalints` will have equal lengths in range `[0, 100]`.

### Solution 
 To solve this complex expression parsing and evaluation problem, we'll need to follow these steps:

1. Parse the input expression into tokens for processing.
2. Evaluate the simple variables using the provided evaluation map.
3. Evaluate the expression using a stack-based method that takes operator precedence and parentheses into account.
4. Store results in the form of coefficients for each combination of variables.
5. Convert the resulting expression into the desired output format.

We'll use Python for implementing the solution. Here's the explanation and code that can be modified and run directly on the LeetCode platform:



```python
from collections import defaultdict
import re

class Solution:
    def basicCalculatorIV(self, expression: str, evalvars: List[str], evalints: List[int]) -> List[str]:
        # First, create a mapping from variable names to values
        eval_map = dict(zip(evalvars, evalints))
        
        def parse(expression):
            """ Tokenize the given mathematical expression. """
            # This regex will split the expression at each operator or space while keeping them in the result
            tokens = re.findall(r'\d+|[a-z]+|[()+-/*]', expression)
            return tokens
        
        def evaluate(tokens):
            """ Evaluate the tokenized expression respecting operator precedence and parentheses. """
            vals = []  # Stack for values/expressions
            ops = []   # Stack for operators
            depth = defaultdict(int)  # To manage variable terms by their multiplication depth
            
            def perform_op():
                b = vals.pop()
                a = vals.pop()
                op = ops.pop()
                res = defaultdict(int)
                
                if op == '+':
                    for k in a: res[k] += a[k]
                    for k in b: res[k] += b[k]
                elif op == '-':
                    for k in a: res[k] += a[k]
                    for k in b: res[k] -= b[k]
                elif op == '*':
                    for ka in a:
                        for kb in b:
                            # Creating new term by merging and sorting the keys (representing variable products)
                            new_k = tuple(sorted(ka + kb))
                            res[new_k] += a[ka] * b[kb]
                return res
            
            i = 0
            while i < len(tokens):
                token = tokens[i]
                if token.isdigit() or (token[0] == '-' and token[1:].isdigit()):
                    vals.append(defaultdict(int, {(): int(token)}))
                elif token.isalpha():
                    if token in eval_map:
                        vals.append(defaultdict(int, {(): eval_map[token]}))
                    else:
                        vals.append(defaultdict(int, {(tuple([token]),): 1}))
                elif token in "+-":
                    while ops and ops[-1] in "+-*":
                        vals.append(perform_op())
                    ops.append(token)
                elif token == '*':
                    while ops and ops[-1] == '*':
                        vals.append(perform_op())
                    ops.append(token)
                elif token == '(':
                    ops.append(token)
                elif token == ')':
                    while ops[-1] != '(':
                        vals.append(perform_op())
                    ops.pop()  # pop '('
                i += 1
            
            while ops:
                vals.append(perform_op())
            
            return vals[-1]
        
        def to_output_format(result):
            """ Convert the result dictionary into the sorted output list format. """
            output = []
            for variables, coeff in sorted(result.items(), key=lambda x: (-len(x[0]), x[0])):
                if coeff != 0:
                    if not variables:
                        output.append(str(coeff))
                    else:
                        term = "*".join(variables)
                        output.append(f"{coeff}*{term}" if coeff != 1 or not variables else term)
            return output
        
        tokens = parse(expression)
        result = evaluate(tokens)
        return to_output_format(result)


```

This implementation includes:
- Parsing the expression into usable tokens.
- Evaluating expressions based on the precedence of operations using two separate stacks for values and operators.
- A custom handling for variable names, including operating with them or substituting with given values.
- Formatting and returning the results as expected, ordered by the variable terms' degree and lexicographical order.
  
By following this structured plan, each aspect of handling mathematical expressions, from parsing to evaluation and returning formatted results, is meticulously covered.

# 771. Jewels and Stones

### Problem Description 
You're given strings `jewels` representing the types of stones that are jewels, and `stones` representing the stones you have. Each character in `stones` is a type of stone you have. You want to know how many of the stones you have are also jewels.

Letters are case sensitive, so `"a"` is considered a different type of stone from `"A"`.


Example 1:
Input: jewels = "aA", stones = "aAAbbbb"
Output: 3

Example 2:
Input: jewels = "z", stones = "ZZ"
Output: 0

Constraints:
`1 <= jewels.length, stones.length <= 50`
`jewels` and `stones` consist of only English letters.

All the characters of `jewels` are unique.

### Solution 
 Below is the Python code tailored to solve the mentioned problem on the LeetCode platform. The problem is essentially about determining how frequently the characters from the `jewels` string appear in the `stones` string. This solution represents a straightforward approach using Python's capabilities to address character counting in strings.



```python
class Solution:
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        # Initialize a counter to keep track of the number of jewels found in stones
        jewel_count = 0
        
        # Keep a set of jewel types for faster lookup
        jewel_types = set(jewels)
        
        # Iterate through each stone and check if it's a jewel
        for stone in stones:
            # If the stone is in the jewel_types set, increment the jewel count
            if stone in jewel_types:
                jewel_count += 1
        
        # Once all stones have been checked, return the total count of jewels found
        return jewel_count

# Example test cases
sol = Solution()

# Test case 1
jewels = "aA"
stones = "aAAbbbb"
print(sol.numJewelsInStones(jewels, stones))  # Output: 3

# Test case 2
jewels = "z"
stones = "ZZ"
print(sol.numJewelsInStones(jewels, stones))  # Output: 0

```

### Detailed Explanation:
- **Step 1:** Create a class named `Solution` and define a method `numJewelsInStones` which takes two arguments, `jewels` and `stones`.
- **Step 2:** Initialize `jewel_count` to zero. This variable keeps track of how many stones are also jewels.
- **Step 3:** Convert the `jewels` string into a set called `jewel_types`. Using a set is beneficial because checking membership in a set is on average O(1) time complexity, making lookups faster.
- **Step 4:** Loop through each character in the `stones` string. For each character, check if it exists in the `jewel_types` set. If it does, increment the `jewel_count`.
- **Step 5:** After the loop is complete, the `jewel_count` variable holds the total number of jewels found in the stones, which is then returned as the solution.

This method leverages Python’s set data structure for fast lookups and is efficient given the constraints, ensuring excellent performance.

# 781. Rabbits in Forest

### Problem Description 
In a forest, each rabbit has some color. Some subset of rabbits (possibly all of them) tell you how many other rabbits have the same color as them. Those `answers` are placed in an array.

Return the minimum number of rabbits that could be in the forest.


Examples:
Input: answers = [1, 1, 2]
Output: 5
Explanation:
The two rabbits that answered "1" could both be the same color, say red.

The rabbit than answered "2" can't be red or the answers would be inconsistent.

Say the rabbit that answered "2" was blue.

Then there should be 2 other blue rabbits in the forest that didn't answer into the array.

The smallest possible number of rabbits in the forest is therefore 5: 3 that answered plus 2 that didn't.

Input: answers = [10, 10, 10]
Output: 11
Input: answers = []
Output: 0
Note:
`answers` will have length at most `1000`.

Each `answers[i]` will be an integer in the range `[0, 999]`.

### Solution 
 To solve this problem, we need to figure out the minimum number of rabbits that can explain the answers provided.

### Idea Behind the Solution:
The key observation is that if a rabbit says there are `x` other rabbits of the same color, then there are `x + 1` rabbits of that color in the forest including the one who answered. Therefore:
- For `n` rabbits who all answer `x`, at least `ceil(n / (x + 1))` groups of `x + 1` rabbits exist.
- We sum up the counts for each group to get the smallest number of rabbits which is consistent with the answers.

### Steps:
1. Count the frequency of each answer.
2. For each answer value:
   - Calculate the number of groups needed for this answer using the ceiling of `number_of_rabbits_with_this_answer / (x + 1)`.
   - Multiply this by `x + 1` to get the minimum number of rabbits for this group of answers.
3. Sum the results from all different answers to get the total minimum number of rabbits.

This approach efficiently computes the answer considering each rabbit's response with respect to others to minimize inconsistencies in counting.

Here is a direct Python implementation that can be pasted into LeetCode's editor:



```python
class Solution:
    def numRabbits(self, answers: List[int]) -> int:
        from math import ceil
        count = {}
        
        # Count the frequency of each answer
        for answer in answers:
            if answer in count:
                count[answer] += 1
            else:
                count[answer] = 1
        
        total_rabbits = 0
        
        # Calculate the minimum number of rabbits for each answer counted
        for answer, num in count.items():
            group_size = answer + 1
            required_groups = ceil(num / group_size)
            total_rabbits += required_groups * group_size
            
        return total_rabbits

```

### LeetCode Problem Constraints
- The solution should run with the time and space complexity accommodating up to 1000 response values, each between 0 to 999. This solution's main operations involve iterating through the list of answers and calculating using constant time operations (arithmetic and hash table operations), so it should perform efficiently within the problem constraints.

# 811. Subdomain Visit Count

### Problem Description 
A website domain like "discuss.leetcode.com" consists of various subdomains. At the top level, we have "com", at the next level, we have "leetcode.com", and at the lowest level, "discuss.leetcode.com". When we visit a domain like "discuss.leetcode.com", we will also visit the parent domains "leetcode.com" and "com" implicitly.

Now, call a "count-paired domain" to be a count (representing the number of visits this domain received), followed by a space, followed by the address. An example of a count-paired domain might be "9001 discuss.leetcode.com".

We are given a list `cpdomains` of count-paired domains. We would like a list of count-paired domains, (in the same format as the input, and in any order), that explicitly counts the number of visits to each subdomain.


Example 1:
Input: 
["9001 discuss.leetcode.com"]
Output: 
["9001 discuss.leetcode.com", "9001 leetcode.com", "9001 com"]
Explanation: 
We only have one website domain: "discuss.leetcode.com". As discussed above, the subdomain "leetcode.com" and "com" will also be visited. So they will all be visited 9001 times.


Example 2:
Input: 
["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
Output: 
["901 mail.com","50 yahoo.com","900 google.mail.com","5 wiki.org","5 org","1 intel.mail.com","951 com"]
Explanation: 
We will visit "google.mail.com" 900 times, "yahoo.com" 50 times, "intel.mail.com" once and "wiki.org" 5 times. For the subdomains, we will visit "mail.com" 900 + 1 = 901 times, "com" 900 + 50 + 1 = 951 times, and "org" 5 times.

Notes: 
The length of `cpdomains` will not exceed `100`. 
The length of each domain name will not exceed `100`.

Each address will have either 1 or 2 "." characters.

The input count in any count-paired domain will not exceed `10000`.

The answer output can be returned in any order.

### Solution 
 To solve the problem of counting domain visits, we can use a dictionary to keep track of the visit count for each domain and its subdomains. We'll split each entry to separate the visit counts from the domains, and then further break down each domain into its components. Let's break down the steps in detail:

1. Initialize a dictionary (`domain_count`) to store our results with keys as domains/subdomains and values as the accumulated visit counts.
2. Loop through the list of `cpdomains`, and for each count-paired domain:
   - Split the count from the domain.
   - Begin with the full domain and recursively strip off the highest-level section (left-most part before a dot) until no more sections remain, updating the dictionary at each step.
3. Finally, format the results from the dictionary into the count-paired domain format as specified.

Below is the Python code for solving the problem, ready to be tested on LeetCode:



```python
class Solution:
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        domain_count = {}
        
        for cpdomain in cpdomains:
            count, domain = cpdomain.split()
            count = int(count)
            
            parts = domain.split('.')
            subdomain = ""
            for i in range(len(parts)):
                # Use '.' to join parts of the domain if not the first time in the loop
                if i == 0:
                    subdomain = parts[-(i+1)]
                else:
                    subdomain = parts[-(i+1)] + "." + subdomain
                
                # Update domain count in the dictionary
                if subdomain in domain_count:
                    domain_count[subdomain] += count
                else:
                    domain_count[subdomain] = count
        
        # Prepare output to match the required format
        result = []
        for domain, count in domain_count.items():
            result.append(f"{count} {domain}")

        return result

```

This Python solution is structured as a class `Solution` with a method `subdomainVisits`, as is typically required in LeetCode problems. It efficiently computes the counts for each domain and subdomain and converts it into the required output format.

# 884. Uncommon Words from Two Sentences

### Problem Description 
We are given two sentences `A` and `B`.  (A sentence is a string of space separated words.  Each word consists only of lowercase letters.)
A word is uncommon if it appears exactly once in one of the sentences, and does not appear in the other sentence.

Return a list of all uncommon words. 
You may return the list in any order.


Example 1:
Input: A = "this apple is sweet", B = "this apple is sour"
Output: ["sweet","sour"]

Example 2:
Input: A = "apple apple", B = "banana"
Output: ["banana"]
Note:
`0 <= A.length <= 200`
`0 <= B.length <= 200`
`A` and `B` both contain only spaces and lowercase letters.

### Solution 
 To solve this problem, the strategy involves understanding word frequencies across the two input sentences A and B. An uncommon word is defined as a word that appears exactly once across both sentences and does not appear in the other sentence.

Here's the breakdown of the approach:
1. Count word frequencies for both sentences separately.
2. Identify the words that occur exactly once in one sentence and do not appear in the other.

Using Python's `collections.Counter` will be beneficial here, as it provides a straightforward way to count occurrences. Afterwards, the counters can be queried to gather the words that match the criteria mentioned above (uncommon words).

Here is the code following the LeetCode solution format, utilizing comments for clarity:



```python
from collections import Counter

class Solution:
    def uncommonFromSentences(self, A: str, B: str) -> List[str]:
        # Create word frequency counters for both sentences
        countA = Counter(A.split())
        countB = Counter(B.split())
        
        # Result list to store uncommon words
        result = []
        
        # Check for uncommon words in Sentence A
        for word in countA:
            if countA[word] == 1 and countB[word] == 0:
                result.append(word)
        
        # Check for uncommon words in Sentence B
        for word in countB:
            if countB[word] == 1 and countA[word] == 0:
                result.append(word)
        
        return result

# Sample Usage:
# sol = Solution()
# print(sol.uncommonFromSentences("this apple is sweet", "this apple is sour"))  # Outputs: ["sweet", "sour"]
# print(sol.uncommonFromSentences("apple apple", "banana"))                     # Outputs: ["banana"]

```

**Explanation of the Solution:**

1. **Word Counting:** We split the sentences A and B into words and count the occurrences using `Counter`.
   
2. **Identifying Uncommon Words:** 
   - For each word in Sentence A, if it appears exactly once and does not appear in Sentence B (`countB[word] == 0`), it's added to the result.
   - Similarly, for each word in Sentence B, if it appears exactly once and is not in Sentence A, add it to the result.

3. **Returning Result:** The uncommon words, collected in the `result` list, are returned.

**Complexity Analysis:**
- Let `m` be the number of distinct words in A and `n` be the number of distinct words in B.
- **Time Complexity:** O(m + n) due to iterating over the words in each of the counters.
- **Space Complexity:** O(m + n) for the space required to store the counters and the result list.

This solution efficiently identifies the uncommon words and is suited to run directly on platforms like LeetCode as per the requested format.

# 895. Maximum Frequency Stack

### Problem Description 
Design a stack-like data structure to push elements to the stack and pop the most frequent element from the stack.

Implement the `FreqStack` class:
`FreqStack()` constructs an empty frequency stack.

`void push(int val)` pushes an integer `val` onto the top of the stack.

`int pop()` removes and returns the most frequent element in the stack.

	
If there is a tie for the most frequent element, the element closest to the stack's top is removed and returned.


Example 1:
Input
["FreqStack", "push", "push", "push", "push", "push", "push", "pop", "pop", "pop", "pop"]
[[], [5], [7], [5], [7], [4], [5], [], [], [], []]
Output
[null, null, null, null, null, null, null, 5, 7, 5, 4]
Explanation
FreqStack freqStack = new FreqStack();
freqStack.push(5); // The stack is [5]
freqStack.push(7); // The stack is [5,7]
freqStack.push(5); // The stack is [5,7,5]
freqStack.push(7); // The stack is [5,7,5,7]
freqStack.push(4); // The stack is [5,7,5,7,4]
freqStack.push(5); // The stack is [5,7,5,7,4,5]
freqStack.pop();   // return 5, as 5 is the most frequent. The stack becomes [5,7,5,7,4].

freqStack.pop();   // return 7, as 5 and 7 is the most frequent, but 7 is closest to the top. The stack becomes [5,7,5,4].

freqStack.pop();   // return 5, as 5 is the most frequent. The stack becomes [5,7,4].

freqStack.pop();   // return 4, as 4, 5 and 7 is the most frequent, but 4 is closest to the top. The stack becomes [5,7].


Constraints:
`0 <= val <= 109`
At most `2 * 104` calls will be made to `push` and `pop`.

It is guaranteed that there will be at least one element in the stack before calling `pop`.

### Solution 
 

```python
class FreqStack:
    
    def __init__(self):
        """
        Initialize the FreqStack with three data structures:
        1. self.freq: a dictionary to store the frequency of each element.
        2. self.group: a dictionary to store stacks of elements for each frequency.
        3. self.max_freq: to track the maximum frequency of any element in the stack.
        """
        self.freq = {}
        self.group = {}
        self.max_freq = 0

    def push(self, x: int) -> None:
        """
        Push element x onto the stack.
        Update its frequency in self.freq.
        Push x onto the appropriate frequency stack in self.group.
        """
        if x in self.freq:
            self.freq[x] += 1
        else:
            self.freq[x] = 1
        
        f = self.freq[x]
        if f > self.max_freq:
            self.max_freq = f
            self.group[f] = []
        
        self.group[f].append(x)

    def pop(self) -> int:
        """
        Pop and returns the most frequent element from the stack.
        If there is a tie, return the element closest to the top.
        Adjust the frequencies and potentially the self.max_freq after popping.
        """
        x = self.group[self.max_freq].pop()
        self.freq[x] -= 1
        
        if not self.group[self.max_freq]:
            self.max_freq -= 1
            
        return x
        
# Your FreqStack object will be instantiated and called as such:
# freqStack = FreqStack()
# freqStack.push(x)
# param_2 = freqStack.pop()

# Example usage:
# freqStack = FreqStack()
# freqStack.push(5)
# freqStack.push(7)
# freqStack.push(5)
# freqStack.push(7)
# freqStack.push(4)
# freqStack.push(5)
# print(freqStack.pop())  # Output should be 5
# print(freqStack.pop())  # Output should be 7
# print(freqStack.pop())  # Output should be 5
# print(freqStack.pop())  # Output should be 4

```

### Explanation

1. **Data Structures Used:**
   - `self.freq`: Dictionary to track how many times an item has been pushed onto the stack.
   - `self.group`: Dictionary where the keys are frequencies and each value is a list that acts as a stack for items with that frequency.
   - `self.max_freq`: Integer that stores the maximum frequency of any element in the stack.

2. **Push Operation:**
   - Increment the frequency count of the element being pushed.
   - If the element's frequency is higher than the current `self.max_freq`, update `self.max_freq` and initialize a new stack if it doesn't exist.
   - Add the element to its corresponding frequency stack in `self.group`.

3. **Pop Operation:**
   - Pop the element from the stack corresponding to `self.max_freq`.
   - Decrement the frequency of that element in `self.freq`.
   - If this was the last element in the highest frequency stack (`self.group[self.max_freq]` is empty), decrease the `self.max_freq`.
   
This implementation ensures `push` runs in O(1) time complexity and `pop` is also O(1) because accessing the last element in a Python list (acting as a stack here) is O(1). This meets the constraints efficiently.

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
 To solve this problem of finding how many non-empty subarrays have a specific sum \( S \) within an array \( A \) composed of \( 0 \)s and \( 1 \)s, a hash map can be effectively used to maintain a cumulative sum and the frequencies of each sum encountered.

Here is a step-by-step explanation of the approach:
1. Initialize a variable `cumulative_sum` to track the sum of elements from the beginning of the array to the current index.
2. Use a hash map (`sum_count`) to store how many times each cumulative sum has occurred.
3. As we traverse the array, determine the value of `cumulative_sum` required that could form a subarray with sum \( S \) by calculating `cumulative_sum[current] - S`.
4. If this "required cumulative_sum" has been seen before (exists in `sum_count`), it means there are some subarrays (ending at the current index) that sum up to \( S \). The frequency of this "required cumulative_sum" gives the number of such subarrays.
5. Update the hash map with the current cumulative sum for future references.

This algorithm runs in \( O(N) \) time complexity with \( O(N) \) space complexity, where \( N \) is the number of elements in the array \( A \).

Below is the Python code implementing this approach, formatted for direct use on LeetCode:



```python
class Solution:
    def numSubarraysWithSum(self, A: List[int], S: int) -> int:
        sum_count = defaultdict(int)
        sum_count[0] = 1
        cumulative_sum = 0
        result = 0

        for num in A:
            cumulative_sum += num
            if cumulative_sum - S in sum_count:
                result += sum_count[cumulative_sum - S]
            sum_count[cumulative_sum] += 1

        return result

```

Each line of the code works as follows:
- `sum_count = defaultdict(int)`: Initializes the hash map to store the frequency of each cumulative sum, with a default value of 0 for a new sum.
- `sum_count[0] = 1`: There is one way (an empty subarray) to have a sum of 0 before we start.
- In the loop:
  - We add each element to `cumulative_sum` to get the new sum up to the current element.
  - We check if the difference `cumulative_sum - S` has been seen before (exists in `sum_count`). If yes, the value of `sum_count[cumulative_sum - S]` is added to `result`.
  - Finally, update the `sum_count` with the new `cumulative_sum`.

Remember to include `from collections import defaultdict` at the start of your script when running on your local machine or in any environment other than LeetCode's, which provides standard imports automatically.

# 939. Minimum Area Rectangle

### Problem Description 
Given a set of points in the xy-plane, determine the minimum area of a rectangle formed from these points, with sides parallel to the x and y axes.

If there isn't any rectangle, return 0.


Example 1:
Input: [[1,1],[1,3],[3,1],[3,3],[2,2]]
Output: 4

Example 2:
Input: [[1,1],[1,3],[3,1],[3,3],[4,1],[4,3]]
Output: 2
Note:
`1 <= points.length <= 500`
`0 <= points[i][0] <= 40000`
`0 <= points[i][1] <= 40000`
All points are distinct.

### Solution 
 To solve the problem of finding the minimum area of a rectangle that has sides parallel to the x and y axes, it's key to leverage the properties of such rectangles. Any rectangle defined this way can be identified if we know four points: two distinct x-coordinates and two distinct y-coordinates. If we have these coordinates, the area of the rectangle is simply the difference between the two x-coordinates multiplied by the difference between the two y-coordinates.

Here's a step-by-step solution to the problem:

1. **Store Points By Coordinates**: Use two sets to store unique x-coordinates and y-coordinates.

2. **Fast Lookup Structure**: We can use a set for fast point existence checks. Each point can be converted into a tuple (for example, `(x, y)`) and added to the set.

3. **Locate Potential Rectangles**: For each pair of points with the same y-coordinate or the same x-coordinate, consider these pairs as potential opposite sides of a rectangle. The goal will be to find complementary pairs with the same properties, to complete a rectangle.

4. **Calculate Minimum Areas**: For each potential rectangle identified, calculate its area and keep track of the minimum area found.

The solution's complexity largely depends on how efficiently we can find and handle the pairings of potential rectangle corners.

Here’s the solution for the problem:



```python
class Solution:
    def minAreaRect(self, points):
        import sys
        # Convert the list of points into a set for quick access
        point_set = set((x, y) for x, y in points)
        
        n = len(points)
        min_area = sys.maxsize
        found = False
        
        for i in range(n):
            for j in range(i + 1, n):
                # Extract points
                x1, y1 = points[i]
                x2, y2 = points[j]
                
                # Check if we can form a rectangle using (x1, y1) and (x2, y2) as diagonal corners
                if x1 != x2 and y1 != y2:  # they should not be on the same line horizontally or vertically
                    # Check if the complementary points exist
                    if (x1, y2) in point_set and (x2, y1) in point_set:
                        area = abs(x2 - x1) * abs(y2 - y1)
                        if area < min_area:
                            found = True
                            min_area = area
        
        return min_area if found else 0

# Example usage:
sol = Solution()
print(sol.minAreaRect([[1,1],[1,3],[3,1],[3,3],[2,2]]))  # Output: 4
print(sol.minAreaRect([[1,1],[1,3],[3,1],[3,3],[4,1],[4,3]]))  # Output: 2

```

### Detailed Explanation:

1. **Structure Setup**: Use a set `point_set` for O(1) look-ups to check if a point exists.
2. **Double Loop**: Go through all pairs of points (using loops for `i` and `j`).
3. **Rectangle Check**: When two points are picked, we avoid cases of them being aligned horizontally or vertically as such pairs can't form the diagonal of our desired rectangle.
4. **Area Calculation**: Given two diagonal corners, check for the existence of the other two corners in the set and then calculate the area if possible. If this area is smaller than the current minimum, update the minimum.
5. **Return Result**: Finally, return the minimum area found; return 0 if no rectangle can be formed.

This algorithm ensures that we efficiently find the minimum area of a rectangle with axes-parallel sides using point existence checks and some conditional logic to avoid unnecessary calculations.

# 953. Verifying an Alien Dictionary

### Problem Description 
In an alien language, surprisingly they also use english lowercase letters, but possibly in a different `order`. The `order` of the alphabet is some permutation of lowercase letters.

Given a sequence of `words` written in the alien language, and the `order` of the alphabet, return `true` if and only if the given `words` are sorted lexicographicaly in this alien language.


Example 1:
Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
Output: true
Explanation: As 'h' comes before 'l' in this language, then the sequence is sorted.


Example 2:
Input: words = ["word","world","row"], order = "worldabcefghijkmnpqstuvxyz"
Output: false
Explanation: As 'd' comes after 'l' in this language, then words[0] > words[1], hence the sequence is unsorted.


Example 3:
Input: words = ["apple","app"], order = "abcdefghijklmnopqrstuvwxyz"
Output: false
Explanation: The first three characters "app" match, and the second string is shorter (in size.) According to lexicographical rules "apple" > "app", because 'l' > '∅', where '∅' is defined as the blank character which is less than any other character (More info).


Constraints:
`1 <= words.length <= 100`
`1 <= words[i].length <= 20`
`order.length == 26`
All characters in `words[i]` and `order` are English lowercase letters.

### Solution 
 The problem described calls for checking whether a list of words is sorted according to a specific alien language order. This order redefines the alphabetic sequence that can differ from the standard English ordering of 'a' to 'z'. To do this, an effective method involves mapping each letter of this alien order to its position in a natural index sequence (0 through 25, because there are 26 total letters). Once the mapping is established, we can convert each word into a tuple of indices representing its characters in sequence. These tuple representations enable straightforward comparison using Python's natural tuple comparison, where lexicographic ordering is the default.

Here is a Python solution:



```python
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        # Create a dictionary to map each character to its index in 'order'
        order_map = {char: index for index, char in enumerate(order)}
        
        # Function to convert a word into its indices based on 'order'
        def convert_to_index_sequence(word):
            return [order_map[char] for char in word]
        
        # Convert all words to their respective index sequences
        words_transformed = [convert_to_index_sequence(word) for word in words]
        
        # Check if the transformed list of index sequences is sorted
        # This uses Python's default lexicographical comparison for lists
        for i in range(len(words_transformed) - 1):
            if words_transformed[i] > words_transformed[i + 1]:
                return False
                
        return True

# Example usage
# sol = Solution()
# print(sol.isAlienSorted(["hello","leetcode"], "hlabcdefgijkmnopqrstuvwxyz"))  # Output: True
# print(sol.isAlienSorted(["word","world","row"], "worldabcefghijkmnpqstuvxyz"))  # Output: False
# print(sol.isAlienSorted(["apple","app"], "abcdefghijklmnopqrstuvwxyz"))  # Output: False

```

The `Solution` class includes a method `isAlienSorted`, which adheres to typical LeetCode function signatures (using typing for input arguments clarity). This method directly translates each word into a sequence of indices over which comparison operations are trustworthy and standard.

We include a nested function `convert_to_index_sequence` that takes a word and transforms it to a list of indices. Before the main comparison loop, we precompute the index sequences for all the words, storing them in `words_transformed`. This allows for a simple loop comparison, checking if each word (in transformed form) is less than or equal to the following one.

This solution should comfortably fit within a performance budget given the problem constraints, where list lengths and string lengths are both relatively modest.

# 954. Array of Doubled Pairs

### Problem Description 
Given an array of integers `arr` of even length, return `true` if and only if it is possible to reorder it such that `arr[2 * i + 1] = 2 * arr[2 * i]` for every `0 <= i < len(arr) / 2`.


Example 1:
Input: arr = [3,1,3,6]
Output: false

Example 2:
Input: arr = [2,1,2,6]
Output: false

Example 3:
Input: arr = [4,-2,2,-4]
Output: true
Explanation: We can take two groups, [-2,-4] and [2,4] to form [-2,-4,2,4] or [2,4,-2,-4].


Example 4:
Input: arr = [1,2,4,16,8,4]
Output: false

Constraints:
`0 <= arr.length <= 3 * 104`
`arr.length` is even.

`-105 <= arr[i] <= 105`

### Solution 
 To solve this problem, we can utilize a strategy involving sorting and using a hash map (or dictionary in Python) to keep track of the occurrences of each integer in the array. The main idea is to determine if for each number in the array, there exists its double. The steps in the plan go as follows:

1. **Sort the Array:** Sort the array to handle the numbers in a systematic order, which will also benefit in checking for a pair effectively.

2. **Use a Frequency Dictionary:** Build a dictionary to count the frequency of each number in the array. This will help us determine how many times we need to find a pair for each number.

3. **Process Each Element for Pairing:** Iterate through the sorted array and for each integer `x`:
   - Check if it can form a valid pair with `2*x` (i.e., if `2*x` is present in the dictionary and it has a positive count after accounting for `x`).
   - Decrease the count in the dictionary for both `x` and `2*x` when a valid pair is found.
   - If at any step, a valid pairing is not possible, return `false`.

4. **Return True:** If all elements successfully find their partners according to the conditions mentioned in the problem, then return `true`.

This step-by-step approach ensures that the solution is complete and handles various complexities of the problem, such as negative numbers and zeros.



```python
class Solution:
    def canReorderDoubled(self, arr: List[int]) -> bool:
        # Frequency count of each element
        count = collections.Counter(arr)
        
        # Sort the keys of count to handle pairs systematically
        for x in sorted(count, key=abs):
            # If there's more need for doubled number than available, return False
            if count[2*x] < count[x]:
                return False
            # Reduce the counts in the map as we pair elements
            count[2*x] -= count[x]
        
        return True

# Example use case testing on Leetcode platform will be like this:
# sol = Solution()
# print(sol.canReorderDoubled([4, -2, 2, -4]))  # Output: True

```

This solution repeatedly matches each number `x` with `2*x` while depleting the frequency count. The check is made sure by both looking at positive and negative numbers using absolute value sorting, to ensure pairs with negative numbers are properly processed. The problem requirements and constraints seem to be well covered by this solution strategy.

# 957. Prison Cells After N Days

### Problem Description 
There are `8` prison cells in a row and each cell is either occupied or vacant.

Each day, whether the cell is occupied or vacant changes according to the following rules:
If a cell has two adjacent neighbors that are both occupied or both vacant, then the cell becomes occupied.

Otherwise, it becomes vacant.

Note that because the prison is a row, the first and the last cells in the row can't have two adjacent neighbors.

You are given an integer array `cells` where `cells[i] == 1` if the `ith` cell is occupied and `cells[i] == 0` if the `ith` cell is vacant, and you are given an integer `n`.

Return the state of the prison after `n` days (i.e., `n` such changes described above).


Example 1:
Input: cells = [0,1,0,1,1,0,0,1], n = 7
Output: [0,0,1,1,0,0,0,0]
Explanation: The following table summarizes the state of the prison on each day:
Day 0: [0, 1, 0, 1, 1, 0, 0, 1]
Day 1: [0, 1, 1, 0, 0, 0, 0, 0]
Day 2: [0, 0, 0, 0, 1, 1, 1, 0]
Day 3: [0, 1, 1, 0, 0, 1, 0, 0]
Day 4: [0, 0, 0, 0, 0, 1, 0, 0]
Day 5: [0, 1, 1, 1, 0, 1, 0, 0]
Day 6: [0, 0, 1, 0, 1, 1, 0, 0]
Day 7: [0, 0, 1, 1, 0, 0, 0, 0]

Example 2:
Input: cells = [1,0,0,1,0,0,1,0], n = 1000000000
Output: [0,0,1,1,1,1,1,0]

Constraints:
`cells.length == 8`
`cells[i]` is either `0` or `1`.

`1 <= n <= 109`

### Solution 
 The problem requires simulating how an array representing prison cells evolves over a number of days according to specific rules. A direct simulation approach is inefficient because of the high potential value of `n` (up to 1 billion). Instead, we can leverage the fact that the transformation will inevitably enter a cycle because there are only 2^8 (256) possible states for the prison cells.

### Approach:

1. **Simulation with Cycle Detection**: Start simulating the transformation day by day.
2. Use a hashmap or dictionary to store each unique state of `cells` with the day number as its value. This lets us identify when a state repeats, indicating the start of a cycle.
3. Once a cycle is detected:
    - Calculate the cycle length.
    - Skip the simulation of days that lie entirely within these cycles to save computation time.
    - For the remaining days, compute the result by simulating a number of days equivalent to `n % cycle_length`.
4. Special boundary conditions where cells[0] and cells[7] will always turn to 0 due to them not having two adjacent neighbors.

Here is a Python implementation of this approach:



```python
class Solution:
    def prisonAfterNDays(self, cells: List[int], n: int) -> List[int]:
        def next_day(cells):
            new_cells = [0] * 8  # The boundaries remain 0 because they don't have two adjacent cells.
            for i in range(1, 7):
                new_cells[i] = int(cells[i-1] == cells[i+1])
            return new_cells
        
        seen = {}
        cycle_len = 0
        has_cycle = False
        
        for day in range(n):
            next_state = tuple(next_day(cells))
            if next_state in seen:
                cycle_len = day - seen[next_state]
                has_cycle = True
                break
            else:
                seen[next_state] = day
                cells = next_state
        
        if has_cycle:
            n %= cycle_len
            for _ in range(n):
                cells = next_day(cells)
        
        return cells

# Additional parts for LeetCode submission
def string_to_integer_list(input):
    return json.loads(input)

def integer_list_to_string(nums, len_of_list=None):
    if not len_of_list:
        len_of_list = len(nums)
    return json.dumps(nums[:len_of_list])

import json

def main():
    line = input()
    cells = string_to_integer_list(line)
    line = input()
    n = int(line)
    
    sol = Solution()
    ret = sol.prisonAfterNDays(cells, n)

    out = integer_list_to_string(ret)
    print(out)

if __name__ == '__main__':
    main()

```

**Note**: This code checks for cycles and manages to shorten the computation by skipping the bulk of repetitive transformations once a cycle is detected. Given that there are only 256 possible states and we need to check for a transformation rule that causes cycling, our approach should be efficient enough to handle even the upper limit constraints.

# 961. N-Repeated Element in Size 2N Array

### Problem Description 
In a array `A` of size `2N`, there are `N+1` unique elements, and exactly one of these elements is repeated `N` times.

Return the element repeated `N` times.


Example 1:
Input: [1,2,3,3]
Output: 3

Example 2:
Input: [2,1,2,5,3,2]
Output: 2

Example 3:
Input: [5,1,5,2,5,3,5,4]
Output: 5
Note:
`4 <= A.length <= 10000`
`0 <= A[i] < 10000`
`A.length` is even

### Solution 
 To solve the given problem, the key observation is that since there is exactly one element in the array `A` which repeats `N` times, while all other elements are unique and the total size of the array is `2N`, finding the repeated element becomes essentially finding any element that appears more than once.

Given that:
- The list size is between 4 and 10000,
- The element values are within the range 0 to 9999.

Several solutions can be proposed:

1. **Using a Dictionary (or Hash Map)**:
   - Traverse through each element in the array and maintain a frequency count in the dictionary.
   - Any element whose count reaches 2 can immediately be returned as it is the one which repeats `N` times.
   - Time Complexity: O(N), Space Complexity: O(N).

2. **Set Lookup**:
   - As you iterate through each element, keep adding it to a set.
   - If you find that the element is already in the set, then it must be the recurring element.
   - Time Complexity: O(N), Space Complexity: O(N) in the worst case when all items get added just before the repeated item.

3. **Sorting**:
   - By sorting the array first, the repeated element which occurs `N` times, would be adjacent in the array at some point.
   - After sorting, scan through the list and check adjacent elements for equality.
   - Time Complexity: O(N log N) due to sorting, Space Complexity: O(1) or O(N) depending on the sorting algorithm.

Here's the implementation using a Set, which is generally quite efficient and simple to understand and code:



```python
class Solution:
    def repeatedNTimes(self, A: List[int]) -> int:
        seen = set()
        for num in A:
            if num in seen:
                return num
            seen.add(num)


```

The provided solution:
- Initializes an empty set `seen`.
- It then iterates over each element in the array `A`.
  - If the element is already in the set, it returns that element as it must be the repeated element.
  - Otherwise, it adds the element to the set for future checks.

This solution guarantees finding the repeated element in linear time since each insert and lookup in a set, on average, is O(1). Given the constraints, this is efficient and effective.

# 966. Vowel Spellchecker

### Problem Description 
Given a `wordlist`, we want to implement a spellchecker that converts a query word into a correct word.

For a given `query` word, the spell checker handles two categories of spelling mistakes:
Capitalization: If the query matches a word in the wordlist (case-insensitive), then the query word is returned with the same case as the case in the wordlist.

	

Example: `wordlist = ["yellow"]`, `query = "YellOw"`: `correct = "yellow"`

Example: `wordlist = ["Yellow"]`, `query = "yellow"`: `correct = "Yellow"`

Example: `wordlist = ["yellow"]`, `query = "yellow"`: `correct = "yellow"`
Vowel Errors: If after replacing the vowels `('a', 'e', 'i', 'o', 'u')` of the query word with any vowel individually, it matches a word in the wordlist (case-insensitive), then the query word is returned with the same case as the match in the wordlist.

	

Example: `wordlist = ["YellOw"]`, `query = "yollow"`: `correct = "YellOw"`

Example: `wordlist = ["YellOw"]`, `query = "yeellow"`: `correct = ""` (no match)

Example: `wordlist = ["YellOw"]`, `query = "yllw"`: `correct = ""` (no match)
In addition, the spell checker operates under the following precedence rules:
When the query exactly matches a word in the wordlist (case-sensitive), you should return the same word back.

When the query matches a word up to capitlization, you should return the first such match in the wordlist.

When the query matches a word up to vowel errors, you should return the first such match in the wordlist.

If the query has no matches in the wordlist, you should return the empty string.

Given some `queries`, return a list of words `answer`, where `answer[i]` is the correct word for `query = queries[i]`.


Example 1:
Input: wordlist = ["KiTe","kite","hare","Hare"], queries = ["kite","Kite","KiTe","Hare","HARE","Hear","hear","keti","keet","keto"]
Output: ["kite","KiTe","KiTe","Hare","hare","","","KiTe","","KiTe"]

Example 2:
Input: wordlist = ["yellow"], queries = ["YellOw"]
Output: ["yellow"]

Constraints:
`1 <= wordlist.length, queries.length <= 5000`
`1 <= wordlist[i].length, queries[i].length <= 7`
`wordlist[i]` and `queries[i]` consist only of only English letters.

### Solution 
 Here's a solution with explanation aimed at solving the problem on LeetCode. The solution introduces auxiliary dictionaries for fast look-ups by processing clear cases and uncertain cases defined by case and vowel variation. We summarize the steps below:

1. **Exact Match Dictionary (`exact_dict`)**: Map each word in `wordlist` to itself providing O(1) access for exact matches.
2. **Lowercased Dictionary (`case_insensitive_dict`)**: Map a lowercased version of each word in `wordlist` to the first instance of such word found (to maintain order preference).
3. **Vowel Error Dictionary (`vowel_error_dict`)**: Generate a generic mask for each word by replacing vowels with a placeholder (e.g., '*') and then map that mask to the first instance of a word that fits that mask.

When processing queries:

1. **Exact Match Check**: Check against `exact_dict`.
2. **Case-Insensitive Match**: Check against `case_insensitive_dict` if first check fails.
3. **Vowel Error Match**: Generate the vowel mask for the query and check against `vowel_error_dict`.
4. **No Match**: Return `""` if none of the checks succeed.

The solution code will directly work on the LeetCode as requested:



```python
class Solution:
    def spellchecker(self, wordlist: List[str], queries: List[str]) -> List[str]:
        exact_dict = {}
        case_insensitive_dict = {}
        vowel_error_dict = {}
        
        vowels = set('aeiou')
        
        def to_vowel_mask(word):
            return ''.join('*' if c in vowels else c for c in word.lower())
        
        # Build dictionaries based on wordlist
        for word in wordlist:
            # For exact matches:
            if word not in exact_dict:
                exact_dict[word] = word
            
            # For case-insensitive matches:
            lower_word = word.lower()
            if lower_word not in case_insensitive_dict:
                case_insensitive_dict[lower_word] = word
            
            # For vowel-error tolerance matches:
            vowel_mask = to_vowel_mask(word)
            if vowel_mask not in vowel_error_dict:
                vowel_error_dict[vowel_mask] = word
        
        # Process queries
        answers = []
        for query in queries:
            # Check exact match:
            if query in exact_dict:
                answers.append(query)
                continue
            
            # Check case-insensitive match:
            lower_query = query.lower()
            if lower_query in case_insensitive_dict:
                answers.append(case_insensitive_dict[lower_query])
                continue
            
            # Check vowel-error tolerance:
            vowel_mask_query = to_vowel_mask(query)
            if vowel_mask_query in vowel_error_dict:
                answers.append(vowel_error_dict[vowel_mask_query])
                continue

            # No matches found:
            answers.append("")
        
        return answers

```

This code provides a detailed implementation of the spell checker based on the three dictionary strategies described. Each step and dictionary usage corresponds directly to the rules described in your outlined problem settings.

# 974. Subarray Sums Divisible by K

### Problem Description 
Given an array `A` of integers, return the number of (contiguous, non-empty) subarrays that have a sum divisible by `K`.


Example 1:
Input: A = [4,5,0,-2,-3,1], K = 5
Output: 7
Explanation: There are 7 subarrays with a sum divisible by K = 5:
[4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2, -3]
Note:
`1 <= A.length <= 30000`
`-10000 <= A[i] <= 10000`
`2 <= K <= 10000`

### Solution 
 To solve the problem of finding the number of contiguous subarrays with a sum divisible by K, we can employ prefix sums and the properties of modular arithmetic. The key observation here is that for any subarray `A[i:j]`, the sum of elements can be expressed in terms of prefix sums `P` as `P[j] - P[i-1]`. This subarray sum is divisible by K if `(P[j] - P[i-1]) % K == 0`. Rearranged, this criterion becomes `P[j] % K == P[i-1] % K`.

This observation suggests a strategy based on counting the frequencies of each modulus value as we iterate through the array. The steps to achieve this are:
1. Compute the prefix sum modulus for every element.
2. Use a dictionary (hashmap) to store how many times each modulus has occurred as prefix sum moduli up to the current index. 
3. For each modulus found, increment the total count of valid subarrays by the frequency of this modulus (since each represents a potential starting point that, paired with our current end point, will form a subarray divisible by K).

Here's the Python code implementing this strategy:



```python
class Solution:
    def subarraysDivByK(self, A, K):
        # Dictionary to count frequency of each modulus
        count = {0: 1}
        # Prefix sum initialisation
        prefix_sum = 0
        result = 0
        
        # Iterate through array A
        for num in A:
            # Update the prefix sum
            prefix_sum += num
            
            # Modulus operation and make it positive
            mod = prefix_sum % K
            if mod < 0:
                mod += K
            
            # If this particular mod has been seen before
            if mod in count:
                # Increment result by the number of times the mod has occurred
                result += count[mod]
            
            # Increment the count of this mod in the dictionary
            if mod in count:
                count[mod] += 1
            else:
                count[mod] = 1
        
        return result

```

**Explanation of Code:**
1. `count` - dictionary to track occurrences of each modulus value, initialized with `{0: 1}` to handle cases where a prefix sum itself is exactly divisible by `K`.
2. `prefix_sum` - maintains the sum of elements as we iterate through the array.
3. `mod` - stores the adjusted modulo K of the prefix sum. Adjustment is made to handle negative values.
4. Loop through each element in the array `A`, updating `prefix_sum`, calculating `mod`, and using the `count` dictionary to tally the frequency of modulus values and thus deduce how many subarrays end at the current position have a sum divisible by K.
5. Use these modulus counts to add to the `result` which eventually stores the total count of subarrays divisible by K.

This solution is efficient with time complexity `O(n)`, as it makes a single pass through the array, and each hash map operation (search, insert, update) can be considered `O(1)` on average. This meets the problem constraints effectively.

# 981. Time Based Key-Value Store

### Problem Description 
Create a timebased key-value store class `TimeMap`, that supports two operations.

1. `set(string key, string value, int timestamp)`
Stores the `key` and `value`, along with the given `timestamp`.

2. `get(string key, int timestamp)`
Returns a value such that `set(key, value, timestamp_prev)` was called previously, with `timestamp_prev <= timestamp`.

If there are multiple such values, it returns the one with the largest `timestamp_prev`.

If there are no values, it returns the empty string (`""`).


Example 1:
Input: inputs = ["TimeMap","set","get","get","set","get","get"], inputs = [[],["foo","bar",1],["foo",1],["foo",3],["foo","bar2",4],["foo",4],["foo",5]]
Output: [null,null,"bar","bar",null,"bar2","bar2"]
Explanation:   
TimeMap kv;   
kv.set("foo", "bar", 1); // store the key "foo" and value "bar" along with timestamp = 1   
kv.get("foo", 1);  // output "bar"   
kv.get("foo", 3); // output "bar" since there is no value corresponding to foo at timestamp 3 and timestamp 2, then the only value is at timestamp 1 ie "bar"   
kv.set("foo", "bar2", 4);   
kv.get("foo", 4); // output "bar2"   
kv.get("foo", 5); //output "bar2"   

Example 2:
Input: inputs = ["TimeMap","set","set","get","get","get","get","get"], inputs = [[],["love","high",10],["love","low",20],["love",5],["love",10],["love",15],["love",20],["love",25]]
Output: [null,null,null,"","high","high","low","low"]
Note:
All key/value strings are lowercase.

All key/value strings have length in the range `[1, 100]`
The `timestamps` for all `TimeMap.set` operations are strictly increasing.

`1 <= timestamp <= 10^7`
`TimeMap.set` and `TimeMap.get` functions will be called a total of `120000` times (combined) per test case.

### Solution 
 To solve the problem, a TimeMap class will be created which manages a dictionary to handle multiple timestamps for keys using built-in Python libraries. Each key in the dictionary will store a list of tuples where each tuple contains `(timestamp, value)`. These tuples are kept in descending order of `timestamp` to optimize the retrieval process in the get method. The `set` operation will append tuples directly since it's given that timestamps are always increasing. The `get` operation will use binary search to quickly find the most appropriate timestamp that is less than or equal to the requested one.

Here is the implementation:



```python
class TimeMap:
    def __init__(self):
        """
        Initialize the TimeMap data structure.
        """
        import collections
        self.store = collections.defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        """
        Stores the (value, timestamp) tuple in the list associated with the key.
        Since timestamps are strictly increasing, append the tuple at the end of the list.
        """
        self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        """
        Retrieves the value with the largest timestamp_prev less than or equal to the given timestamp.
        If no such value exists, return "".
        """
        values = self.store.get(key, [])
        # Perform binary search to find the right value at or below the given timestamp
        low, high = 0, len(values) - 1
        while low <= high:
            mid = (low + high) // 2
            if values[mid][0] == timestamp:
                return values[mid][1]
            elif values[mid][0] < timestamp:
                low = mid + 1
            else:
                high = mid - 1
        # If low is 0, it means all timestamps are greater than the requested one, return ""
        if high >= 0:
            return values[high][1]
        return ""

# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)

```

**Explanation:**

- **Initialization (`__init__`)**: We use `collections.defaultdict` to handle keys automatically; any non-existent keys queried later return a list.
- **Set (`set`)**: Given that timestamps for each key are added in increasing order, we append `(timestamp, value)` tuples directly to the list corresponding to the key.
- **Get (`get`)**: Implements binary search to quickly find the highest `timestamp_prev` that is ≤ the requested `timestamp`. If no such `timestamp` is found, return an empty string.

This solution is optimal and handles the constraints well, ensuring that operations on the key-value store are performed efficiently. The use of binary search in the `get` method ensures a log(n) complexity where n is the number of entries per key, making the solution suitable for high-frequency queries.

# 987. Vertical Order Traversal of a Binary Tree

### Problem Description 
Given the `root` of a binary tree, calculate the vertical order traversal of the binary tree.

For each node at position `(row, col)`, its left and right children will be at positions `(row + 1, col - 1)` and `(row + 1, col + 1)` respectively. The root of the tree is at `(0, 0)`.

The vertical order traversal of a binary tree is a list of top-to-bottom orderings for each column index starting from the leftmost column and ending on the rightmost column. There may be multiple nodes in the same row and same column. In such a case, sort these nodes by their values.

Return the vertical order traversal of the binary tree.


Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: [[9],[3,15],[20],[7]]
Explanation:
Column -1: Only node 9 is in this column.

Column 0: Nodes 3 and 15 are in this column in that order from top to bottom.

Column 1: Only node 20 is in this column.

Column 2: Only node 7 is in this column.


Example 2:
Input: root = [1,2,3,4,5,6,7]
Output: [[4],[2],[1,5,6],[3],[7]]
Explanation:
Column -2: Only node 4 is in this column.

Column -1: Only node 2 is in this column.

Column 0: Nodes 1, 5, and 6 are in this column.

          1 is at the top, so it comes first.

          5 and 6 are at the same position (2, 0), so we order them by their value, 5 before 6.

Column 1: Only node 3 is in this column.

Column 2: Only node 7 is in this column.


Example 3:
Input: root = [1,2,3,4,6,5,7]
Output: [[4],[2],[1,5,6],[3],[7]]
Explanation:
This case is the exact same as example 2, but with nodes 5 and 6 swapped.

Note that the solution remains the same since 5 and 6 are in the same location and should be ordered by their values.


Constraints:
The number of nodes in the tree is in the range `[1, 1000]`.

`0 <= Node.val <= 1000`

### Solution 
 To solve the vertical order traversal problem for a binary tree, the goal is to print nodes in the order of their columns, starting from the leftmost column to the rightmost. Here's how we can structure our solution:

1. **Use BFS Traversal**: We use Breadth First Search (BFS) as it helps to traverse the tree level by level. This ensures the row order is maintained naturally.
2. **Tracking Coordinates**: We maintain coordinates with each node where `row` and `col` are incremented or decremented based on the position. The root node starts at `(0, 0)`. If a node is at `(row, col)`, then its left child will be `(row + 1, col - 1)` and the right child will be `(row + 1, col + 1)`.
3. **Dict to Maintain Columns**: We use a dictionary to collect the values of nodes that fall into the same column, where the key is the `col` index.
4. **Sorting**: When nodes share the same column and row, we need to sort them by their values.

Finally, the dictionary keys (column indices) are sorted and their corresponding lists (of tuples with row, value pairs) are also sorted by row primarily, and by value secondarily when rows are identical. Here’s the implementation using the above approach:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from collections import deque, defaultdict
import heapq

class Solution:
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        
        # Dictionary to hold the columns
        column_table = defaultdict(list)
        
        # Queue for BFS with (node, row, col)
        queue = deque([(root, 0, 0)])
        
        while queue:
            node, row, col = queue.popleft()
            # Append the node value along with its row number to the column_table at index col
            heapq.heappush(column_table[col], (row, node.val))
            
            # Append children to the queue with updated coordinates
            if node.left:
                queue.append((node.left, row + 1, col - 1))
            if node.right:
                queue.append((node.right, row + 1, col + 1))
        
        # Extract columns sorted by their indices
        sorted_columns = sorted(column_table.keys())
        result = []
        
        # For each column, retrieve the list of tuples, sort them by the rules,
        # and extract the values using a heap to obey order of rows and values
        for col in sorted_columns:
            # Extract values following the heap order which is firstly by rows then by values
            col_values = [val for _, val in sorted(column_table[col])]
            result.append(col_values)
        
        return result

```

This Python code should now be directly runnable on LeetCode and handle the vertical order traversal problem as described. It efficiently uses heapq to maintain the sorted order of values in the same cell which avoids having to manually sort these values later.

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
 To solve the given problem efficiently, we can utilize the sliding window technique along with a few observations and tricks. The key transformation here is instead of directly finding the number of subarrays with exactly `K` unique numbers, we compute it using the difference between the number of subarrays with at most `K` unique numbers and the number of subarrays with at most `K-1` unique numbers. This arithmetic transformation simplifies the problem.

Here's the step-by-step breakdown:

1. **Helper Function**: We'll write a helper function, `atMostK(A, K)`, that returns the number of subarrays in array `A` having at most `K` distinct integers.
2. **Sliding Window Technique**: Within this helper function, we'll use two pointers (`left` and `right`) to represent a window. We'll expand `right` to include new elements until adding another would exceed our allowed distinct count. If it does, we'll adjust the `left` pointer until the condition is satisfied again.
3. **Counting Subarrays**: For any position of the right pointer, the number of subarrays that can be formed with the current window bounded by `left` and `right` is given by the extent the window has covered, i.e., `right - left + 1`.
    
By using the helper function, we determine:
- `atMostK(A, K)`: Total subarrays with at most `K` unique integers,
- `atMostK(A, K-1)`: Total subarrays with at most `K-1` unique integers.

Subtracting the latter from the former gives us the subarrays containing exactly `K` distinct integers:
\[ \text{count of subarrays with exactly K unique integers} = \text{atMostK}(A, K) - \text{atMostK}(A, K-1) \]

The logic behind this calculation is hierarchically building up the count of distinct numbers, segmenting them between allowed thresholds.

Now, the complete code as demanded:



```python
class Solution:
    def subarraysWithKDistinct(self, A: List[int], K: int) -> int:
        
        def atMostK(A, K):
            count = collections.Counter()
            res = left = 0
            for right in range(len(A)):
                if count[A[right]] == 0:
                    K -= 1
                count[A[right]] += 1

                while K < 0:
                    count[A[left]] -= 1
                    if count[A[left]] == 0:
                        K += 1
                    left += 1

                res += right - left + 1
            return res
        
        return atMostK(A, K) - atMostK(A, K - 1)


# Example usage
sol = Solution()
print(sol.subarraysWithKDistinct([1,2,1,2,3], 2))  # Output: 7

```

Explanation:
- `atMostK(A, K)` computes the number of subarrays with at most `K` unique integers using the sliding window strategy.
- We get the answer by calculating the difference: `atMostK(A, K) - atMostK(A, K-1)`. This provides the total number of subarrays with exactly `K` different integers.

The code follows the typical structure for problems on LeetCode, allowing direct copy-and-paste for testing. It includes a class definition, the method definition, and example test usage. Remember to import necessary libraries (`collections` in this case) when necessary.

# 1001. Grid Illumination

### Problem Description 
You are given a `grid` of size `N x N`, and each cell of this grid has a lamp that is initially turned off.

You are also given an array of lamp positions `lamps`, where `lamps[i] = [rowi, coli]` indicates that the lamp at `grid[rowi][coli]` is turned on. When a lamp is turned on, it illuminates its cell and all other cells in the same row, column, or diagonal.

Finally, you are given a query array `queries`, where `queries[i] = [rowi, coli]`. For the `ith` query, determine whether `grid[rowi][coli]` is illuminated or not. After answering the `ith` query, turn off the lamp at `grid[rowi][coli]` and its 8 adjacent lamps if they exist. A lamp is adjacent if its cell shares either a side or corner with `grid[rowi][coli]`.

Return an array of integers `ans`, where `ans[i]` should be `1` if the lamp in the `ith` query was illuminated, or `0` if the lamp was not.


Example 1:
Input: N = 5, lamps = [[0,0],[4,4]], queries = [[1,1],[1,0]]
Output: [1,0]
Explanation: We have the initial grid with all lamps turned off. In the above picture we see the grid after turning on the lamp at grid[0][0] then turning on the lamp at grid[4][4].

The 0th query asks if the lamp at grid[1][1] is illuminated or not (the blue square). It is illuminated, so set ans[0] = 1. Then, we turn off all lamps in the red square.

The 1st query asks if the lamp at grid[1][0] is illuminated or not (the blue square). It is not illuminated, so set ans[1] = 0. Then, we turn off all lamps in the red rectangle.


Example 2:
Input: N = 5, lamps = [[0,0],[4,4]], queries = [[1,1],[1,1]]
Output: [1,1]

Example 3:
Input: N = 5, lamps = [[0,0],[0,4]], queries = [[0,4],[0,1],[1,4]]
Output: [1,1,0]

Constraints:
`1 <= N <= 109`
`0 <= lamps.length <= 20000`
`lamps[i].length == 2`
`0 <= lamps[i][j] < N`
`0 <= queries.length <= 20000`
`queries[i].length == 2`
`0 <= queries[i][j] < N`

### Solution 
 Below is the Python code addressing the described problem, optimized for performance given the constraints, especially the grid size (which can be up to \(10^9\)). The solution leverages dictionaries to efficiently track the state of row, column, and diagonal illuminations instead of maintaining an actual grid representation:



```python
class Solution:
    def gridIllumination(self, N: int, lamps: List[List[int]], queries: List[List[int]]) -> List[int]:
        from collections import defaultdict
        
        lit_rows = defaultdict(int)
        lit_cols = defaultdict(int)
        lit_pos_diags = defaultdict(int)  # r - c
        lit_neg_diags = defaultdict(int)  # r + c
        
        lit_lamps = set()  # to avoid duplicates
        for x, y in lamps:
            if (x, y) not in lit_lamps:
                lit_rows[x] += 1
                lit_cols[y] += 1
                lit_pos_diags[x - y] += 1
                lit_neg_diags[x + y] += 1
                lit_lamps.add((x, y))
        
        result = []
        directions = [(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for x, y in queries:
            # Check if the cell (x, y) is illuminated
            if lit_rows[x] > 0 or lit_cols[y] > 0 or lit_pos_diags[x - y] > 0 or lit_neg_diags[x + y] > 0:
                result.append(1)
            else:
                result.append(0)
            
            # Turn off the lamp at (x, y) and its adjacent cells
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) in lit_lamps:
                    lit_lamps.remove((nx, ny))
                    lit_rows[nx] -= 1
                    lit_cols[ny] -= 1
                    lit_pos_diags[nx - ny] -= 1
                    lit_neg_diags[nx + ny] -= 1
            
        return result

# Example usage:
# sol = Solution()
# print(sol.gridIllumination(5, [[0,0],[4,4]], [[1,1],[1,0]]))
# Output: [1, 0]

```

Explanation:
- **Data Structures**: The `lit_rows`, `lit_cols`, `lit_pos_diags`, and `lit_neg_diags` dictionaries maintain counts of active lamps affecting each row, column, and both diagonals, respectively. `lit_lamps` tracks the positions of currently active lamps to ensure they are unlit only once.
- **Query Response**: For each query, the script checks if any of the four dictionaries has a count greater than zero for the queried position, indicating illumination.
- **Turning Off Lamps**: After each query, the lamp at the query location and all adjacent positions are turned off by decrementing counts in the counts dictionaries and removing entries from `lit_lamps` if they exist.
- **Efficiency**: This solution is highly efficient in terms of memory and time complexity due to using hashing instead of a full grid representation.

The solution approach ensures that we efficiently handle the problem constraints while being ready for direct inclusion in a testing platform like LeetCode.

# 1002. Find Common Characters

### Problem Description 
Given an array `A` of strings made only from lowercase letters, return a list of all characters that show up in all strings within the list (including duplicates).  For example, if a character occurs 3 times in all strings but not 4 times, you need to include that character three times in the final answer.

You may return the answer in any order.


Example 1:
Input: ["bella","label","roller"]
Output: ["e","l","l"]

Example 2:
Input: ["cool","lock","cook"]
Output: ["c","o"]
Note:
`1 <= A.length <= 100`
`1 <= A[i].length <= 100`
`A[i][j]` is a lowercase letter

### Solution 
 To solve the problem, we will utilize the approach of calculating the minimum frequency of each character across all strings. This is because we're interested in characters that appear in all strings and the number of times they should appear in the result list is the minimum count of that character in any string.

Here's the breakdown of the plan:

1. Use a dictionary to keep track of the minimum frequency count of each character. Initialize this dictionary with the character frequencies of the first string.
2. For each subsequent string, create a temporary frequency dictionary of the characters.
3. Update the main frequency dictionary by comparing and keeping the minimum count for each character that exists in both the main frequency dictionary and the temporary one from the current string.
4. Once all strings have been processed, construct the final list by repeating each character based on its minimum frequency across all strings.

Let’s implement this solution in Python, tailored for direct use on LeetCode.




```python
from collections import Counter

class Solution:
    def commonChars(self, A: List[str]) -> List[str]:
        if not A:
            return []

        # Initialize with the frequency count of the first string
        common_count = Counter(A[0])
        
        # Process each string in the list
        for string in A[1:]:
            # Create a frequency count for the current string
            current_count = Counter(string)
            
            # Update the common counts by intersecting with current string counts
            common_count &= current_count
        
        # Expand the characters based on their final counts to form the result list
        result = []
        for char, freq in common_count.items():
            result.extend([char] * freq)
        
        return result

```

This Python code defines the `Solution` class with the method `commonChars` which takes a list of strings "`A`" and returns a list of common characters including duplicates as specified. This solution should work directly in the LeetCode online judge, assuming the standard imports and function definitions are already handled by the LeetCode environment.

