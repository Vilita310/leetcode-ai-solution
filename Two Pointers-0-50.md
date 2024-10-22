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
 To solve the problem of finding the length of the longest substring without repeating characters, we can use the sliding window technique along with a hash map (dictionary in Python) to store the characters and their last seen indices. This approach allows us to efficiently update the start of the window whenever we encounter a repeating character.

Here's how we'll implement the solution step by step:

1. **Initialize Variables:**
   We'll use a dictionary to keep track of the last seen index of each character we encounter. We'll also need two pointers: one for the starting point of our current substring (`start`) and another to iterate through the string (`end`).

2. **Iterate through the String:**
   We'll incrementally build our substring by moving the `end` pointer through the string. Whenever we find a repeating character (i.e., the character is already in our dictionary and its last seen index is within the bounds of our current substring), we'll update the `start` pointer to the index just after the last seen index of this character.

3. **Update the Maximum Length:**
   After processing each character, we'll calculate the length of the current substring (from `start` to `end`) and update our maximum length if this current length is greater. 

4. **Return the Result:**
   Finally, we'll return the length of the longest substring found.

Here's the Python code implementing the above logic. This code is structured to be directly runnable in LeetCode:



```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        char_map = {}  # Dictionary to store the last seen index of each character
        max_length = 0  # To keep track of the maximum length
        start = 0  # Start index of the current substring
        
        for end in range(len(s)):
            if s[end] in char_map and char_map[s[end]] >= start:
                # Character s[end] is a repeat; move start
                start = char_map[s[end]] + 1
            
            # Update the last seen index of the character s[end]
            char_map[s[end]] = end
            
            # Calculate the current length of substring and update max_length
            max_length = max(max_length, end - start + 1)
        
        return max_length

```

### Detailed Explanation:
- **char_map**: This dictionary keeps track of the indices of the characters. When we see a character, we store its index.
- **max_length**: This variable will hold the maximum length of substrings found without repeating characters.
- **start**: This pointer indicates the start index of the current substring we are analyzing.
- In the loop:
  - We check if the character `s[end]` exists in `char_map` and whether its last index is within the current substring (indicated by `start`). If it is, we have to move the `start` pointer to avoid repetition.
  - We then update the last seen index of `s[end]` in the dictionary.
  - Finally, we calculate the length from `start` to `end`, and if it's greater than our previously stored `max_length`, we update `max_length`.

This algorithm runs in O(n) time complexity where n is the length of the string because each character is processed at most twice (once when added and once when we move the `start` pointer), and it uses O(min(n, m)) space complexity for the dictionary, where m is the size of the character set (which is constant in practice).

# 11. Container With Most Water

### Problem Description 
Given `n` non-negative integers `a1, a2, ..., an` , where each represents a point at coordinate `(i, ai)`. `n` vertical lines are drawn such that the two endpoints of the line `i` is at `(i, ai)` and `(i, 0)`. Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.

Notice that you may not slant the container.


Example 1:
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.


Example 2:
Input: height = [1,1]
Output: 1

Example 3:
Input: height = [4,3,2,1,4]
Output: 16

Example 4:
Input: height = [1,2,1]
Output: 2

Constraints:
`n == height.length`
`2 <= n <= 105`
`0 <= height[i] <= 104`

### Solution 
 To solve this problem, we will use the two-pointer technique to determine the maximum amount of water that can be contained by the vertical lines represented in the height array. The aim is to find two lines such that the container they form with the x-axis can hold the maximum area of water.

### Explanation:

1. **Understanding the Problem**: 
   - Given vertical lines represented by their heights, the area of water that can be trapped between two lines (at indices `i` and `j`) can be calculated as: 
     \[
     \text{Area} = \text{min}(height[i], height[j]) \times (j - i)
     \]
   - Here, `min(height[i], height[j])` determines the effective height of the water that can be held, and `(j - i)` is the width between the two lines.

2. **Two-Pointer Approach**:
   - We will use two pointers `left` and `right`. 
   - Start with `left` at the beginning of the list and `right` at the end.
   - Calculate the area with the current pointers and update the maximum area found.
   - Move the pointer pointing to the shorter line inward. This is because moving the shorter side might help find a taller line that can potentially hold more water.

3. **Iterate Until Pointers Meet**:
   - Continue this process while `left` is less than `right`, and keep updating the maximum area.

4. **Complexity**: 
   - Time complexity: O(n) because each pointer will traverse the height array at most once.
   - Space complexity: O(1) since we are using a constant amount of extra space.

Below is the Python code implementing this approach in the LeetCode solution format:



```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        max_area = 0  # Initially, the maximum area is zero
        
        while left < right:
            # Calculate the width
            width = right - left
            # Calculate the area with the current boundaries
            area = min(height[left], height[right]) * width
            # Update maximum area if the current area is greater
            max_area = max(max_area, area)
            
            # Move the pointer pointing to the shorter line
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return max_area

```

### Summary:
This code establishes a systematic way to find the maximum water container area by efficiently using two pointers. The approach guarantees that all potential areas between pairs of lines are considered without redundant calculations, making it optimal for larger inputs, adhering to the problem's constraints.

# 15. 3Sum

### Problem Description 
Given an integer array nums, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

Notice that the solution set must not contain duplicate triplets.


Example 1:
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]

Example 2:
Input: nums = []
Output: []

Example 3:
Input: nums = [0]
Output: []

Constraints:
`0 <= nums.length <= 3000`
`-105 <= nums[i] <= 105`

### Solution 
 To solve the problem of finding all unique triplets in an integer array that sum up to zero, we can follow a systematic approach. The solution requires us to use sorting and a two-pointer technique to efficiently identify the desired triplets while avoiding duplicates.

### Steps to Solve the Problem:

1. **Sorting the Array**: First, we sort the input array. Sorting helps in efficiently skipping duplicates and allows us to apply the two-pointer technique effectively.
  
2. **Iterate through the Array**: We will iterate through the array using a loop, treating each element as a potential first element of a triplet.

3. **Skip Duplicates**: Inside the loop, we need to skip over elements that are the same as the previous element to avoid counting duplicates.

4. **Two-Pointer Technique**: For each selected element, we will use two pointers to find pairs that together with the chosen element sum to zero. One pointer will start just after the current element, while the other pointer will start from the end of the array, moving inward based on the sum of these three values.

5. **Collect Valid Triplets**: When we find a valid triplet, we will save it to a result list and skip duplicates by moving the pointers past any duplicates.

6. **Return the Result**: Finally, we return our list of triplets.

Now, let's implement this in Python, following the LeetCode format:



```python
from typing import List

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()  # Step 1: Sort the array
        result = []  # This will hold the list of unique triplets

        for i in range(len(nums) - 2):  # We stop at len - 2 to leave space for i, j, k
            if i > 0 and nums[i] == nums[i - 1]:
                continue  # Step 3: Skip duplicates for the first element

            left, right = i + 1, len(nums) - 1  # Step 4: Set up two pointers

            while left < right:  # Continue until the pointers meet
                current_sum = nums[i] + nums[left] + nums[right]  # Calculate the sum

                if current_sum < 0:  # If the sum is less than zero, move the left pointer to the right
                    left += 1
                elif current_sum > 0:  # If the sum is greater than zero, move the right pointer to the left
                    right -= 1
                else:
                    # We found a valid triplet
                    result.append([nums[i], nums[left], nums[right]])

                    # Step 5: Skip duplicates for the second element
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    # Step 5: Skip duplicates for the third element
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1

                    # Move both pointers after finding a valid triplet
                    left += 1
                    right -= 1

        return result  # Step 6: Return the result list

```

### Explanation of the Code:

- **Sorting**: The input list `nums` is sorted initially. This allows us to effectively use the two-pointer approach later on and skip duplicates.
- **Main Loop**: We’re iterating through the sorted list until the third last element, as we need at least three numbers to make a triplet.
- **Two Pointers**: For every number selected, the `left` pointer starts just next to it, and the `right` pointer starts at the end of the list. We adjust these pointers based on the sum condition.
- **Skip Duplicates**: After finding a valid triplet, we ensure that we move `left` and `right` pointers past any duplicate values to only collect unique triplets.
- **Return**: The function ultimately returns a list of all unique triplets that sum to zero.

This solution efficiently handles the problem with a time complexity of \(O(n^2)\), which is optimal for this kind of problem given the constraints.

# 16. 3Sum Closest

### Problem Description 
Given an array `nums` of n integers and an integer `target`, find three integers in `nums` such that the sum is closest to `target`. Return the sum of the three integers. You may assume that each input would have exactly one solution.


Example 1:
Input: nums = [-1,2,1,-4], target = 1
Output: 2
Explanation: The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).


Constraints:
`3 <= nums.length <= 10^3`
`-10^3 <= nums[i] <= 10^3`
`-10^4 <= target <= 10^4`

### Solution 
 To solve the problem of finding three integers in an array whose sum is closest to a given target, we can use a combination of sorting and a two-pointer technique. Here's a step-by-step breakdown of the approach, followed by the complete Python code formatted for LeetCode submission.

### Approach:

1. **Sorting the Array**: First, we sort the array `nums`. Sorting helps us to easily manage the combinations of three numbers.

2. **Initialize Variables**: We need to keep track of the sum that is closest to the target, so we can initialize a variable `closest_sum` with a value that is far from the possible sums (e.g., positive infinity).

3. **Iterate with Three Pointers**: 
   - Use a loop to fix one element of the triplet. For each element at index `i`, we apply a two-pointer approach to find the other two elements.
   - The two pointers (`left` and `right`) start from the next position after `i` and the end of the array, respectively, and they identify potential combinations of triplets whose sum we will calculate.

4. **Calculate and Compare**: For each combination:
   - Calculate the current sum of the three integers.
   - If this sum is equal to the target, we can immediately return it as it's the closest possible sum.
   - If it's not, we check if we should update our closest sum based on how close this sum is to the target.

5. **Adjust Pointers**: Depending on whether the current sum is less than or greater than the target, we move the `left` pointer to the right (to increase the sum) or the `right` pointer to the left (to decrease the sum).

6. **Return the Result**: After checking all potential triplets, return the closest sum found.

### Python Code:
Here is the Python implementation utilizing the described algorithm:



```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()  # Step 1: Sort the array
        closest_sum = float('inf')  # Step 2: Initialize closest_sum to a large number
        
        for i in range(len(nums) - 2):  # Step 3: Iterate through the array
            left, right = i + 1, len(nums) - 1  # Set two pointers
            
            while left < right:  # Step 4: While the pointers don't cross
                current_sum = nums[i] + nums[left] + nums[right]  # Calculate current sum
                
                # Step 5: Check if we found a closer sum
                if abs(current_sum - target) < abs(closest_sum - target):
                    closest_sum = current_sum
                
                # Step 6: Move pointers based on the comparison
                if current_sum < target:
                    left += 1  # Move left pointer right to increase sum
                elif current_sum > target:
                    right -= 1  # Move right pointer left to decrease sum
                else:
                    return current_sum  # If the current sum equals the target, return it immediately
                
        return closest_sum  # Step 7: Return the closest sum found

```

### Explanation of the Code:
- We start by sorting the `nums` array. This allows us to apply the two-pointer technique efficiently.
- Within the main loop, we fix one number (at index `i`) and use two additional pointers (`left` and `right`) to find the other two numbers.
- If the sum of these three numbers is found to be closer to the target than any previously found sum, we update `closest_sum`.
- We adjust the positions of the left and right pointers based on the comparison of the current sum with the target, enabling us to explore all possible triplets.
- Finally, if we find a perfect match (the current sum equals the target), we return immediately; otherwise, we finally return the nearest sum we encountered.

This approach runs in \(O(n^2)\), where \(n\) is the length of `nums`, which is efficient enough given the constraints.

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
 To solve the problem of finding unique quadruplets in an array that sum up to a given target, we can follow a systematic approach which utilizes sorting and the two-pointer technique. Below is a detailed explanation along with the Python code structured to fit the LeetCode submission format.

### Explanation:

1. **Sorting the Array**: We first sort the input array `nums`. This makes it easier to avoid duplicates and allows us to use the two-pointer technique effectively.

2. **Iterating with Indices**: We iterate through the array with four pointers (or indices) - `i`, `j`, `left`, and `right`. The indices `i` and `j` are used to secure two numbers, while `left` and `right` are used to find the other two numbers. We select the first two numbers and then use the two pointers to find the pairs that complete the quadruplet.

3. **Avoiding Duplicates**: After picking each number, we ensure that we skip over duplicates. If the current number at the index `i` or `j` is the same as the previous number, we should skip it to avoid counting the same quadruplet multiple times.

4. **Sum Calculation**: For each combination of `i` and `j`, we calculate the sum of the four numbers. If the sum matches the target, we add the quadruplet to the results. If the sum is less than the target, we move the left pointer to the right to increase the sum. If the sum is greater, we move the right pointer to the left to decrease the sum.

5. **Finalizing the Result**: Once all possibilities have been explored, we return the list of unique quadruplets.

### Implementation:

Here is the Python code that implements the above logic:



```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()  # Sort the array
        n = len(nums)
        result = []
        
        for i in range(n - 3):
            # Skip duplicate values for the first number
            if i > 0 and nums[i] == nums[i - 1]:
                continue
                
            for j in range(i + 1, n - 2):
                # Skip duplicate values for the second number
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                    
                left, right = j + 1, n - 1
                
                while left < right:
                    current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                    
                    if current_sum < target:
                        left += 1  # We need a larger sum
                    elif current_sum > target:
                        right -= 1  # We need a smaller sum
                    else:
                        # We found a quadruplet
                        result.append([nums[i], nums[j], nums[left], nums[right]])
                        
                        # Skip duplicates for the third and fourth numbers
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                            
                        left += 1
                        right -= 1
        
        return result

```

### How to use the code:
- The function `fourSum` takes an array `nums` and an integer `target`.
- It returns a list of unique quadruplets that sum up to the target.
- Input the numbers as a list and the target as an integer to get the result.

### Complexity:
- **Time Complexity**: O(n^3), where n is the length of the array. This is due to the three nested loops.
- **Space Complexity**: O(1) if we don't consider the output list space. If we count that space, it's O(k) for storing the result, where k is the number of results.

This approach efficiently finds unique quadruplets while adhering to the constraints provided.

# 19. Remove Nth Node From End of List

### Problem Description 
Given the `head` of a linked list, remove the `nth` node from the end of the list and return its head.

Follow up: Could you do this in one pass?

Example 1:
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]

Example 2:
Input: head = [1], n = 1
Output: []

Example 3:
Input: head = [1,2], n = 1
Output: [1]

Constraints:
The number of nodes in the list is `sz`.

`1 <= sz <= 30`
`0 <= Node.val <= 100`
`1 <= n <= sz`

### Solution 
 To solve the problem of removing the nth node from the end of a linked list, we can use a two-pointer technique to achieve this in a single pass. Here's how we can approach the solution:

### Explanation

1. **Understanding the Problem**: We have a singly linked list, and we need to remove the nth node from the end. A straightforward approach would be to first find the length of the list and then identify the node to remove. However, we can achieve it in a more efficient manner using two pointers.

2. **Two-Pointer Technique**:
   - We'll use two pointers (`first` and `second`).
   - Start both pointers at the beginning of the list.
   - Move the `first` pointer n nodes ahead in the list. This way, when we start moving both pointers at the same pace, the `first` pointer will be n nodes ahead of the `second` pointer.
   - When the `first` pointer reaches the end of the list, the `second` pointer will be exactly at the node before the nth node from the end.
   - We can then easily adjust the pointer of the node before the `second` pointer to skip the nth node.

3. **Handling Edge Cases**:
   - If the list contains only one node and we need to remove it, we should return `None`.
   - We must ensure our code can handle various input sizes compliant with the constraints provided.

### Python Code

Here's the implementation of the above idea in accordance with the LeetCode format:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        # Create a dummy node that points to the head
        dummy = ListNode(0)
        dummy.next = head
        
        # Initialize two pointers
        first = dummy
        second = dummy
        
        # Move the first pointer n+1 steps ahead to maintain the gap
        for _ in range(n + 1):
            first = first.next
            
        # Move both pointers until the first pointer reaches the end
        while first:
            first = first.next
            second = second.next
            
        # Remove the nth node from the end
        second.next = second.next.next
        
        # Return the modified list, starting from the next node of the dummy
        return dummy.next

```

### How the Code Works

- **ListNode Class**: The `ListNode` class defines a node in the linked list.
- **Solution Class**: The `Solution` class contains the method `removeNthFromEnd`, which takes the head of the list and the integer `n` as arguments.
- **Dummy Node**: We use a dummy node to handle edge cases more smoothly, such as when we're removing the head of the list.
- **First Pointer Moving**: The `first` pointer is advanced by `n + 1` steps to create a gap of `n` nodes between `first` and `second`.
- **Simultaneous Movement**: We then move both pointers simultaneously until `first` reaches the end of the list. At this point, `second` will be just before the node we want to remove.
- **Node Removal**: We adjust the `next` pointer of the `second` to skip the target node.
- **Return Result**: Finally, we return `dummy.next`, which points to the head of the modified list.

This solution effectively finds and removes the nth node from the end in a single pass, fulfilling the problem’s requirements.

# 26. Remove Duplicates from Sorted Array

### Problem Description 
Given a sorted array nums, remove the duplicates in-place such that each element appears only once and returns the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

Clarification:
Confused why the returned value is an integer but your answer is an array?
Note that the input array is passed in by reference, which means a modification to the input array will be known to the caller as well.

Internally you can think of this:
// nums is passed in by reference. (i.e., without making a copy)
int len = removeDuplicates(nums);
// any modification to nums in your function would be known by the caller.

// using the length returned by your function, it prints the first len elements.

for (int i = 0; i < len; i++) {
    print(nums[i]);
}

Example 1:
Input: nums = [1,1,2]
Output: 2, nums = [1,2]
Explanation: Your function should return length = `2`, with the first two elements of `nums` being `1` and `2` respectively. It doesn't matter what you leave beyond the returned length.


Example 2:
Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4]
Explanation: Your function should return length = `5`, with the first five elements of `nums` being modified to `0`, `1`, `2`, `3`, and `4` respectively. It doesn't matter what values are set beyond the returned length.


Constraints:
`0 <= nums.length <= 3 * 104`
`-104 <= nums[i] <= 104`
`nums` is sorted in ascending order.

### Solution 
 Certainly! Below is a step-by-step explanation of how to solve the problem of removing duplicates from a sorted array in-place, followed by a Python implementation that you can run directly on LeetCode.

### Problem Understanding

Given a sorted array `nums`, our goal is to modify this array such that each unique element appears only once, and return the new length of the modified array. The modification should be done in-place, meaning that we can't use additional space for another array.

### Plan

1. **Use Two Pointers**: We can maintain two pointers:
   - One (`i`) as a slow pointer that will track the position in the updated array where the next unique element should be placed.
   - The other (`j`) as a fast pointer that will iterate through the array checking for unique elements.
  
2. **Comparison of Elements**: Since the array is sorted, we can compare the current element (`nums[j]`) with the last added unique element (`nums[i]`). If they are different, it means `nums[j]` is unique and should be added to the array at index `i + 1`.

3. **Updating Pointers**: Whenever we find a unique element:
   - Increment the `i` pointer to move to the next position.
   - Set `nums[i]` to `nums[j]` (add the unique element to the result).

4. **Return the Length**: At the end of traversing the array, the index `i` will give us the count of unique elements. The length will be `i + 1`.

### Time Complexity
- The solution runs in O(n), where n is the number of elements in the array, since we only traverse the array once.

### Space Complexity
- The algorithm uses O(1) extra space, as it only uses a few variables for indexing.

### Python Code

Here is the code that implements the above plan:



```python
class Solution:
    def removeDuplicates(self, nums):
        if not nums:
            return 0

        # Pointer for the position of the last unique element
        i = 0

        # Iterate through the array starting from the second element
        for j in range(1, len(nums)):
            # If the current element is different from the last unique element
            if nums[j] != nums[i]:
                i += 1  # Move to the next unique position
                nums[i] = nums[j]  # Update the element at the unique position
        
        # The length of the array with unique elements
        return i + 1

```

### Explanation of the Code

- `if not nums:`: Check if the array is empty. If it is, return 0 since there are no elements.
- `i = 0`: Initialize `i` to 0, as the first unique element is at index 0.
- `for j in range(1, len(nums)):`: Start iterating from the second element.
- Inside the loop, we compare `nums[j]` with `nums[i]`. If they differ, we found a new unique element:
  - We increment `i` and set `nums[i]` to `nums[j]`.

Finally, we return `i + 1`, since `i` is the index of the last unique element, and the count of unique elements is `i + 1`.

### Examples to Test

Once you run this code in LeetCode, you can test it with the provided examples:
1. `removeDuplicates([1, 1, 2])` should return `2` and modify `nums` to `[1, 2]`.
2. `removeDuplicates([0, 0, 1, 1, 1, 2, 2, 3, 3, 4])` should return `5` and modify `nums` to `[0, 1, 2, 3, 4]`.

This approach effectively removes duplicates and provides the length of the modified array as required.

# 27. Remove Element

### Problem Description 
Given an array nums and a value `val`, remove all instances of that value in-place and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with `O(1)` extra memory.

The order of elements can be changed. It doesn't matter what you leave beyond the new length.

Clarification:
Confused why the returned value is an integer but your answer is an array?
Note that the input array is passed in by reference, which means a modification to the input array will be known to the caller as well.

Internally you can think of this:
// nums is passed in by reference. (i.e., without making a copy)
int len = removeElement(nums, val);
// any modification to nums in your function would be known by the caller.

// using the length returned by your function, it prints the first len elements.

for (int i = 0; i < len; i++) {
    print(nums[i]);
}

Example 1:
Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2]
Explanation: Your function should return length = 2, with the first two elements of nums being 2.

It doesn't matter what you leave beyond the returned length. For example if you return 2 with nums = [2,2,3,3] or nums = [2,2,0,0], your answer will be accepted.


Example 2:
Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,4,0,3]
Explanation: Your function should return length = `5`, with the first five elements of `nums` containing `0`, `1`, `3`, `0`, and 4. Note that the order of those five elements can be arbitrary. It doesn't matter what values are set beyond the returned length.


Constraints:
`0 <= nums.length <= 100`
`0 <= nums[i] <= 50`
`0 <= val <= 100`

### Solution 
 To solve the problem of removing all instances of a specified value from an array in-place and returning the new length, we can use two pointers. One pointer will iterate through the array and the other will keep track of the position to place the next non-`val` element.

Here's a detailed breakdown of the solution:

1. **Initialize Pointers**: Use a variable `new_length` to track the new length of the modified array and set it to 0 at the beginning. This pointer will point to the index in the array where we should place the next non-`val` element.

2. **Iterate Through the Array**: Use a loop to go through each element in the `nums` array:
   - If the current element is not equal to `val`, copy that element to the `nums[new_length]` and increment the `new_length`.
   - If the current element is equal to `val`, simply skip it.

3. **Return the Result**: After the loop, `new_length` will contain the length of the modified array without the value `val`. We return this length.

4. **In-place Modification**: Since we are modifying the original array directly by overwriting elements, we comply with the problem's requirement of not using extra space for another array.

Here's how the code looks in Python, following the LeetCode format:



```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        new_length = 0  # Initialize length for the modified array
        
        for i in range(len(nums)):
            if nums[i] != val:  # Check if current element is not `val`
                nums[new_length] = nums[i]  # Place it at the new length index
                new_length += 1  # Increase the length counter
        
        return new_length  # Return the new length of the modified array

```

### Explanation of the Code:
- **Function Definition**: The function `removeElement` takes in a list `nums` and an integer `val`.
- **New Length Initialization**: The variable `new_length` keeps track of where we will write the next non-`val` element.
- **Loop Through Each Element**: We use a for loop with `i` ranging from `0` to `len(nums) - 1`:
  - Inside the loop, we check if `nums[i]` is not equal to `val`.
  - If true, we write `nums[i]` to `nums[new_length]`, and then increment `new_length` by 1.
- **Return Statement**: Finally, we return `new_length`, which denotes how many elements remain in the modified array.

### Complexity:
- **Time Complexity**: O(n), where n is the number of elements in the input list, since we go through the list once.
- **Space Complexity**: O(1), since we don't use any additional data structures that grow with the size of the input.

This approach efficiently modifies the array in-place and fulfills the problem's constraints. You can run this code directly on LeetCode as it follows the expected format and structure for submissions.

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
 To solve the problem of implementing the `strStr()` function, we need to determine the index of the first occurrence of a substring (needle) within a string (haystack). If the needle is not found, we will return `-1`. If the needle is an empty string, we will return `0`.

### Approach
1. **Handle Edge Cases:**
   - If the `needle` is an empty string, we return `0`.
   - If the `needle` is longer than the `haystack`, it's impossible for it to be found, so we return `-1`.

2. **Searching for the Needle:**
   - We will iterate through the `haystack`, checking substrings of the same length as the `needle`.
   - If we find a substring that matches the `needle`, we return the starting index of that substring.

3. **Efficiency:**
   - This approach has a time complexity of O(n * m), where n is the length of the `haystack` and m is the length of the `needle`. This is acceptable given the constraints.

Let's implement this in Python:



```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        # Edge case: if needle is an empty string
        if not needle:
            return 0
        
        # Lengths of haystack and needle
        len_haystack = len(haystack)
        len_needle = len(needle)
        
        # If needle is longer than haystack, return -1
        if len_needle > len_haystack:
            return -1
        
        # Check each substring in haystack
        for i in range(len_haystack - len_needle + 1):
            # Check if the substring matches the needle
            if haystack[i:i + len_needle] == needle:
                return i
        
        # If no match was found, return -1
        return -1

```

### Explanation of the Code
- We define a class `Solution` which is standard for LeetCode problems.
- Inside the class, we define the method `strStr` that takes `haystack` and `needle` as arguments.
- The first condition checks if `needle` is an empty string; if so, we return `0`.
- We calculate the lengths of both the `haystack` and `needle` to facilitate comparison.
- If the `needle` is longer than the `haystack`, it is impossible for the `needle` to be a substring, and we return `-1`.
- We loop through the `haystack` up to the index where the remaining substring length is at least equal to the `needle`.
- Within the loop, we check if the substring from the current index `i` to `i + len_needle` matches `needle`.
- If a match is found, we return the current index `i`.
- If the loop completes without finding a match, we return `-1` to indicate the `needle` was not found in `haystack`.

### Complexity
- Time Complexity: O(n * m), where n is the length of `haystack` and m is the length of `needle`.
- Space Complexity: O(1), since we are not using any additional data structures that grow with input size.

This code can be directly run on the LeetCode website as it follows their required format for problems.

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
 To solve the problem of finding all starting indices of substrings in `s` that are a concatenation of each word in `words` exactly once, we can approach the solution using the sliding window technique combined with hash maps. Here's a detailed explanation of the implementation:

### Key Steps in the Algorithm:

1. **Calculate Necessary Values:**
   - Determine the length of each word and the total length of all words combined.
   - Create a frequency map (`word_count`) for the words in the `words` array to keep track of how many times each word appears.

2. **Sliding Window:**
   - Loop over the string `s`, maintaining a sliding window of length equal to the total length of the concatenated words.
   - For each starting point in `s`, extract the substring of the length equal to the combined length of the words.

3. **Check Validity:**
   - For each substring, split it into parts of size equal to the length of each word.
   - Count the occurrences of these parts and compare with the `word_count`. If they match, it means that the substring is a valid concatenation of the words.

4. **Collect Results:**
   - Store the starting indices of valid concatenated substrings in a result list.

Here's the Python code implementing this solution:



```python
from collections import defaultdict

class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        if not s or not words:
            return []
        
        word_length = len(words[0])    # All words have the same length
        num_words = len(words)
        total_length = word_length * num_words
        
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1
        
        result = []
        
        # Loop through the string with a sliding window
        for i in range(len(s) - total_length + 1):
            seen = defaultdict(int)
            j = 0
            
            while j < total_length:
                word = s[i + j:i + j + word_length]  # Get the current word
                if word in word_count:
                    seen[word] += 1
                    # If we've seen too many of this word, break
                    if seen[word] > word_count[word]:
                        break
                else:
                    break

                j += word_length
            
            # If j equals total_length, it means we've found a valid substring
            if j == total_length:
                result.append(i)

        return result

```

### Explanation of Code:

- **Imports:** We use `defaultdict` from the `collections` module to simplify counting occurrences of words.
- **Initial Checks:** We handle edge cases where `s` or `words` are empty.
- **Calculating Lengths:** We establish the length of one word, the total number of words, and the total required length of the substring.
- **Word Counting:** A frequency map `word_count` is created to count occurrences of each word in the `words` array.
- **Sliding Window Loop:** We iterate through possible starting indices in `s` and use a nested loop to check each substring.
- **Seen Words:** We maintain a `seen` map to track how many times each word in the current substring has been seen and check against `word_count`.
- **Validity Check:** At the end of the inner loop, if `j` equals `total_length`, it indicates that we found a valid concatenation, and we append the starting index to the results.

This code should be run on the LeetCode platform directly and will return the correct starting indices for the concatenation of the given words within the string `s`. The solution is efficient with a time complexity of O(N * M) where N is the length of the string and M is the total number of characters in the words when checking each possible substring.

# 42. Trapping Rain Water

### Problem Description 
Given `n` non-negative integers representing an elevation map where the width of each bar is `1`, compute how much water it can trap after raining.


Example 1:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.


Example 2:
Input: height = [4,2,0,3,2,5]
Output: 9

Constraints:
`n == height.length`
`0 <= n <= 3 * 104`
`0 <= height[i] <= 105`

### Solution 
 To solve the problem of calculating how much water can be trapped after rain, we can use a two-pointer approach which provides an efficient O(n) time complexity and O(1) additional space complexity.

### Explanation:
The idea is to keep track of the maximum height seen from both the left and right for each bar, and use those values to determine how much water each bar can trap.

Here’s the step-by-step breakdown of the approach:

1. **Initialize Pointers**: Set two pointers, `left` at the beginning (0) and `right` at the end (n-1) of the height array.

2. **Track Maximum Heights**: Two variables, `left_max` and `right_max`, are used to keep track of the maximum height observed from the left and right sides as we iterate through the array.

3. **Two-Pointer Traversal**:
   - If `height[left]` is less than or equal to `height[right]`, then:
     - If `height[left]` is greater than or equal to `left_max`, update `left_max`.
     - Else, calculate the trapped water at that position as `left_max - height[left]` and accumulate it.
     - Move the `left` pointer to the right (increment `left`).
   - Otherwise:
     - If `height[right]` is greater than or equal to `right_max`, update `right_max`.
     - Else, calculate the trapped water at that position as `right_max - height[right]` and accumulate it.
     - Move the `right` pointer to the left (decrement `right`).

4. **Continue Until Pointers Meet**: Repeat the above steps until the `left` pointer is greater than or equal to the `right` pointer.

5. **Return the Total Water Trapped**: At the end of the process, you should have the total amount of trapped rainwater.

### Python Code:



```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0
        
        left, right = 0, len(height) - 1
        left_max, right_max = height[left], height[right]
        water_trapped = 0
        
        while left < right:
            if height[left] <= height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    water_trapped += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    water_trapped += right_max - height[right]
                right -= 1
        
        return water_trapped

```

### How To Run on LeetCode:
1. Copy the provided Python code.
2. Go to the LeetCode website and find the problem "Trapping Rain Water".
3. Paste the code into the code editor.
4. Ensure that any necessary imports (like `List` from `typing` if needed) are included at the top if it's not already present in the editor.
5. Click on "Run" or "Submit" to execute and verify correctness against the provided test cases.

### Complexity:
- **Time Complexity**: O(n), where n is the length of the input list `height`. We process each element at most twice (once by each pointer).
- **Space Complexity**: O(1), as we are using a constant amount of space for the two pointers and maximum height variables.

# 61. Rotate List

### Problem Description 
Given the `head` of a linked list, rotate the list to the right by `k` places.


Example 1:
Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]

Example 2:
Input: head = [0,1,2], k = 4
Output: [2,0,1]

Constraints:
The number of nodes in the list is in the range `[0, 500]`.

`-100 <= Node.val <= 100`
`0 <= k <= 2 * 109`

### Solution 
 To solve the problem of rotating a linked list to the right by `k` places, we can follow a systematic approach. Here is a step-by-step breakdown of the solution, along with the code that adheres to the LeetCode format.

### Explanation

1. **Initial Observations**:
   - The length of the linked list is crucial for determining the effective operations to perform. Thus, we will first traverse the list to compute its length.
   - If the linked list is empty (`head` is `None`), or if `k` is 0 or a multiple of the length of the list, we can return the original list as no rotation is needed.

2. **Effective Rotations**:
   - If we have `k` greater than the length of the list (let’s denote it as `n`), the effective rotations needed can be computed as `k % n`. This normalization reduces unnecessary full rotations.

3. **Identifying New Head**:
   - To rotate the list, we need to identify the new head and the last node after rotation. The new head will be at the position `n - k` in the list.
   - Traverse to this position, and then adjust the pointers accordingly to make the necessary rotations.

4. **Implementation**:
   - We need to maintain a pointer to the last node of the list, which will help in connecting the end of the rotated list back to the original head.

### Python Code Implementation

Here's the implementation of the above logic formatted for LeetCode.



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        # Step 1: If the list is empty or k is 0, return the head.
        if not head or not head.next or k == 0:
            return head
        
        # Step 2: Determine the length of the linked list.
        length = 1
        current = head
        while current.next:
            current = current.next
            length += 1
        
        # Step 3: Normalize k to avoid extra rotations
        k = k % length
        
        # If k is 0 after normalization, return the original head
        if k == 0:
            return head
        
        # Step 4: Find the rotation point
        current = head
        for _ in range(length - k - 1):
            current = current.next
        
        # Step 5: Current now is at the node just before the new head
        new_head = current.next  # This will be the new head
        current.next = None      # Break the list at this point
        
        # Step 6: Connect the end of the rotated list to the old head
        last = new_head
        while last and last.next:
            last = last.next
        if last:
            last.next = head
        
        return new_head  # Return the new head of the rotated list

```

### How the Code Works:
- We define a `ListNode` class to represent nodes of the linked list.
- The `rotateRight` method is defined within a `Solution` class, which takes the head of the list and the number of places `k` to rotate.
- The list is first assessed for its length, the effective number of rotations is computed, and then we determine the correct node positions to rearrange the pointers to achieve the desired rotation.
- Finally, we return the new head of the rotated linked list.

This code can be directly used on the LeetCode platform where it will be tested against various input scenarios as described in the problem statement.

# 75. Sort Colors

### Problem Description 
Given an array `nums` with `n` objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

We will use the integers `0`, `1`, and `2` to represent the color red, white, and blue, respectively.


Example 1:
Input: nums = [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]

Example 2:
Input: nums = [2,0,1]
Output: [0,1,2]

Example 3:
Input: nums = [0]
Output: [0]

Example 4:
Input: nums = [1]
Output: [1]

Constraints:
`n == nums.length`
`1 <= n <= 300`
`nums[i]` is `0`, `1`, or `2`.

Follow up:
Could you solve this problem without using the library's sort function?
Could you come up with a one-pass algorithm using only `O(1)` constant space?

### Solution 
 To solve the problem of sorting the colors represented by `0`, `1`, and `2` in the array `nums` (where `0` is red, `1` is white, and `2` is blue), we can use the Dutch national flag algorithm proposed by Edsger W. Dijkstra. This approach allows us to sort the array in a single pass and achieve the desired result with O(1) space complexity.

### Approach

We can maintain three pointers:
1. `low` - This pointer will track the position where the next `0` (red) should be placed.
2. `mid` - This pointer will traverse through the array.
3. `high` - This pointer will track the position where the next `2` (blue) should be placed.

Here's how the algorithm works:
- Start with `low` and `mid` initialized to `0`, and `high` initialized to the last index of the array (`n - 1`).
- Traverse the array using the `mid` pointer.
  - If `nums[mid]` is `0` (red), swap it with `nums[low]`, increment both `low` and `mid`.
  - If `nums[mid]` is `1` (white), just increment `mid`.
  - If `nums[mid]` is `2` (blue), swap it with `nums[high]` and decrement `high` (do not increment `mid` in this case since we need to process the swapped value).
- Continue this process until `mid` surpasses `high`.

This method ensures that all `0`s are moved to the beginning, `1`s remain in the middle, and `2`s are moved to the end.

### Python Code
Here is the code that implements the above approach:



```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        low, mid, high = 0, 0, len(nums) - 1
        
        while mid <= high:
            if nums[mid] == 0:
                nums[low], nums[mid] = nums[mid], nums[low]  # Swap
                low += 1
                mid += 1
            elif nums[mid] == 1:
                mid += 1
            else:  # nums[mid] == 2
                nums[mid], nums[high] = nums[high], nums[mid]  # Swap
                high -= 1

# Example usage (uncomment to run in a local environment):
# solution = Solution()
# nums = [2,0,2,1,1,0]
# solution.sortColors(nums)
# print(nums)  # Output: [0,0,1,1,2,2]

```

### Explanation of the Code
- We define a class `Solution` with a method `sortColors`.
- The method takes the input list `nums` and sorts it in place.
- The while loop continues as long as `mid` is less than or equal to `high`, ensuring we examine all elements.
- Based on the value at `nums[mid]`, we either swap and move pointers or just move the `mid` pointer.
- The use of in-place swaps allows us to ensure that we're using constant space.

### Complexity
- **Time Complexity**: O(n), where n is the number of elements in the input array. We make a single pass through the array.
- **Space Complexity**: O(1) because we only use a constant amount of space for the pointers.

This approach efficiently sorts the colors as required by the problem constraints.

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
 Certainly! The problem you're facing is known as the "Minimum Window Substring" problem. To solve it efficiently, we can employ the "sliding window" technique combined with the usage of hashmaps (or dictionaries) to track character counts. Here’s the approach we will take:

### Approach:
1. **Character Frequency Count**: First, we need to count the frequency of each character in string `t`. This gives us a reference for what constitutes a complete "window".

2. **Two Pointers (Sliding Window)**: Use two pointers to maintain a window in string `s`.
   - The `left` pointer marks the start of the window, and the `right` pointer expands the window by moving to the right.
   - As we expand the `right` pointer, we include characters in our window and keep track of how many characters from `t` we have included in our current window.

3. **Check for Validity**: Whenever we include a character that is part of `t`, we check if our window contains all characters of `t` in the required frequencies.

4. **Shrink the Window**: Once we have a valid window, we try to shrink it from the left to find the minimal size.

5. **Update Result**: Track the minimum length found so far and its starting index.

6. **Return the Result**: After scanning through `s`, return the smallest valid window found.

### Complexity
- Time complexity is O(n) where n is the length of `s`.
- Space complexity is O(m) where m is the size of the character set (in this case, it’s constant because we are only dealing with English letters).

Here’s the Python code implementing the above logic in the LeetCode solution format:



```python
from collections import defaultdict

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not s or not t:
            return ""

        # Frequency map for the characters in t
        dict_t = defaultdict(int)
        for char in t:
            dict_t[char] += 1

        # Number of unique characters in t that must be present in the window
        required = len(dict_t)

        # Filtered version of s for characters only present in t
        filtered_s = [(i, char) for i, char in enumerate(s) if char in dict_t]

        left, right = 0, 0
        formed = 0
        window_counts = defaultdict(int)
        ans = float("inf"), None, None  # (window length, left, right)

        while right < len(filtered_s):
            char = filtered_s[right][1]
            window_counts[char] += 1

            # If the current character's frequency matches that in t
            if window_counts[char] == dict_t[char]:
                formed += 1
            
            # Try to contract the window until it ceases to be 'desirable'
            while left <= right and formed == required:
                char = filtered_s[left][1]
                
                # Save the smallest window and update the answer
                end = filtered_s[right][0]
                start = filtered_s[left][0]
                if (end - start + 1) < ans[0]:
                    ans = (end - start + 1, start, end)

                # The character at the position pointed by `left` is no longer a part of the window
                window_counts[char] -= 1
                if window_counts[char] < dict_t[char]:
                    formed -= 1
                
                # Move left pointer ahead
                left += 1

            # Keep expanding to the right
            right += 1
        
        # Return the minimum window or empty string if no window found
        return "" if ans[0] == float("inf") else s[ans[1]:ans[2] + 1]


```

### Explanation of the Code:
- We first construct a frequency dictionary `dict_t` for characters in `t`.
- We prepare a filtered list `filtered_s` which contains only characters from `s` that are in `t`, along with their indices.
- We implement the sliding window using two pointers (`left` and `right`), keeping track of how many of the required characters we have in our current window.
- When we find a valid window that contains all characters in the required frequency, we try to minimize it by moving the `left` pointer until the window is no longer valid.
- Finally, we return the smallest window substring found.

This code should work efficiently within the problem constraints provided.

# 80. Remove Duplicates from Sorted Array II

### Problem Description 
Given a sorted array nums, remove the duplicates in-place such that duplicates appeared at most twice and return the new length.

Do not allocate extra space for another array; you must do this by modifying the input array in-place with O(1) extra memory.

Clarification:
Confused why the returned value is an integer, but your answer is an array?
Note that the input array is passed in by reference, which means a modification to the input array will be known to the caller.

Internally you can think of this:
// nums is passed in by reference. (i.e., without making a copy)
int len = removeDuplicates(nums);
// any modification to nums in your function would be known by the caller.

// using the length returned by your function, it prints the first len elements.

for (int i = 0; i < len; i++) {
    print(nums[i]);
}

Example 1:
Input: nums = [1,1,1,2,2,3]
Output: 5, nums = [1,1,2,2,3]
Explanation: Your function should return length = `5`, with the first five elements of `nums` being `1, 1, 2, 2` and 3 respectively. It doesn't matter what you leave beyond the returned length.


Example 2:
Input: nums = [0,0,1,1,1,1,2,3,3]
Output: 7, nums = [0,0,1,1,2,3,3]
Explanation: Your function should return length = `7`, with the first seven elements of `nums` being modified to `0`, 0, 1, 1, 2, 3 and 3 respectively. It doesn't matter what values are set beyond the returned length.


Constraints:
`1 <= nums.length <= 3 * 104`
`-104 <= nums[i] <= 104`
`nums` is sorted in ascending order.

### Solution 
 To solve the problem of removing duplicates from a sorted array in-place such that each value can appear at most twice, we can use a two-pointer approach. The first pointer keeps track of the position of the next unique or allowed duplicate value, while the second pointer scans through the array. 

### Solution Explanation

1. **Two-Pointer Approach**:
   - Use `i` to represent the position where we can write a valid number (at most twice for each distinct number).
   - Use `j` to scan through the array for the next number.
   
2. **Count Duplicates**:
   - We can count how many times we encounter the current number and ensure that we only copy it over to index `i` up to two times.
   
3. **In-place Modification**:
   - Modify the input array directly without using extra space for another array. The extra variable usage will be constant (O(1)), and thus we keep our space complexity low.

4. **Return the New Length**:
   - After processing the entire array, the position of `i` will represent the length of the modified array.

### Python Code

Here's the Python code following the LeetCode format that you can run directly:



```python
def removeDuplicates(nums):
    if not nums:
        return 0
    
    i = 0  # Pointer for the position to place the next unique or allowed value
    count = 1  # Count occurrences of the current number
    
    for j in range(1, len(nums)):
        if nums[j] == nums[j - 1]:
            count += 1  # Increment count if we encounter the same number
        else:
            count = 1  # Reset count for a new number
        
        # Allow the current number to be added if it appears at most 2 times
        if count <= 2:
            nums[i] = nums[j]  # Place the number at the 'i' position
            i += 1  # Increment 'i' to the next position
            
    return i  # Return the new length of the array

```

### Example Walkthrough

Let's consider the input array `[1, 1, 1, 2, 2, 3]`:

- Initialize `i` at 0 and `count` at 1.
- Start iterating from index 1:
  - At index 1, `nums[1]` is `1`. Increment `count` to 2. Since `count` is <= 2, update `nums[0]` to `1` (no change), and increment `i` to 1.
  - At index 2, `nums[2]` is still `1`. Increment `count` to 3. Now, since `count` > 2, we don’t update `nums`.
  - At index 3, `nums[3]` is `2`. Reset `count` to 1, and update `nums[1]` to `2` and increment `i` to 2.
  - At index 4, `nums[4]` is `2`. Increment `count` to 2, and update `nums[2]` to `2`, incrementing `i` to 3.
  - At index 5, `nums[5]` is `3`. Reset `count` to 1, update `nums[3]` to `3`, incrementing `i` to 4.

The modified array would look like `[1, 1, 2, 2, 3]`, and the length of the valid part would be 5.

### Conclusion

This approach ensures that we modify the input list in place and only utilize O(1) extra space. The runtime complexity is O(n) due to the single scan through the array.

# 86. Partition List

### Problem Description 
Given the `head` of a linked list and a value `x`, partition it such that all nodes less than `x` come before nodes greater than or equal to `x`.

You should preserve the original relative order of the nodes in each of the two partitions.


Example 1:
Input: head = [1,4,3,2,5,2], x = 3
Output: [1,2,2,4,3,5]

Example 2:
Input: head = [2,1], x = 2
Output: [1,2]

Constraints:
The number of nodes in the list is in the range `[0, 200]`.

`-100 <= Node.val <= 100`
`-200 <= x <= 200`

### Solution 
 To solve the problem of partitioning a linked list based on a value `x`, we will create two separate linked lists: one for nodes with values less than `x`, and one for nodes with values greater than or equal to `x`. Once we have both lists constructed, we will connect them to form the final partitioned linked list while maintaining the original relative order of the nodes.

### Step-by-step Approach:

1. **Initialize Two New Linked Lists**: Create two dummy nodes, one for the 'less than x' list (`less_head`) and one for the 'greater than or equal to x' list (`greater_head`).

2. **Iterate Through the Original List**: Traverse the original linked list and for each node:
   - If the node's value is less than `x`, append it to the "less" list.
   - Otherwise, append it to the "greater" list.

3. **Connect the Two Lists**: After processing all nodes, connect the last node of the "less" list to the head of the "greater" list. Ensure to set the last node of the "greater" list's next to `None` to avoid any cycles.

4. **Return the Resulting List**: The head of the resulting list will be the next node of `less_head` since `less_head` itself is a dummy node.

### Code Implementation:

Here’s the Python code that implements this approach:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        # Create two dummy nodes for less and greater lists
        less_head = ListNode(0)
        greater_head = ListNode(0)

        # Two pointers for both lists
        less = less_head
        greater = greater_head

        # Iterate over the original list
        while head:
            if head.val < x:
                less.next = head  # Append to less list
                less = less.next  # Move the pointer forward
            else:
                greater.next = head  # Append to greater list
                greater = greater.next  # Move the pointer forward
            head = head.next  # Move to the next node

        # Connect less list with greater list
        greater.next = None  # Prevents cycle if the last greater node is connected back
        less.next = greater_head.next  # Connect less list with greater list

        # Return the head of the less list, which is less_head.next
        return less_head.next

```

### Explanation of the Code:

1. **ListNode Class**: A class definition for the linked list nodes is provided, which includes a constructor to initialize node values and their next pointers.

2. **Solution Class and `partition` Method**: The `partition` method within the `Solution` class implements the logic described earlier.

3. **Dummy Nodes and Pointers**: Two dummy nodes are initialized (`less_head` and `greater_head`) to facilitate easier list construction. The `less` and `greater` pointers are used to add new nodes to their respective lists.

4. **While Loop**: This loop continues until all nodes in the input list are processed. Based on the comparison with `x`, nodes are appropriately added to either the "less" or "greater" list.

5. **Final Connection**: After the loop, the "greater" list is properly terminated to avoid reference cycles, and the two lists are linked together.

6. **Return Statement**: Finally, the head of the new partitioned linked list is returned, skipping the dummy node.

### Conclusion:

This approach efficiently partitions the linked list in a single pass while maintaining the original relative order of the nodes, achieving an O(n) time complexity where n is the number of nodes in the linked list.

# 88. Merge Sorted Array

### Problem Description 
Given two sorted integer arrays `nums1` and `nums2`, merge `nums2` into `nums1` as one sorted array.

The number of elements initialized in `nums1` and `nums2` are `m` and `n` respectively. You may assume that `nums1` has a size equal to `m + n` such that it has enough space to hold additional elements from `nums2`.


Example 1:
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]

Example 2:
Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]

Constraints:
`nums1.length == m + n`
`nums2.length == n`
`0 <= m, n <= 200`
`1 <= m + n <= 200`
`-109 <= nums1[i], nums2[i] <= 109`

### Solution 
 To solve the problem of merging two sorted integer arrays `nums1` and `nums2`, we will follow a straightforward approach that utilizes two pointers to track the current positions in both arrays. We will fill `nums1` from the back to the front, ensuring that we can merge the arrays without losing data from `nums1`.

### Detailed Explanation:

1. **Initialization**:
   - We have two main pointers: one for the last valid element of `nums1` (pointing to `m-1`) and another for the last element of `nums2` (pointing to `n-1`).
   - We also maintain a pointer for the last position in `nums1` where we will place the largest element from either `nums1` or `nums2`.

2. **Merging Process**:
   - We will iterate backwards from the end of `nums1`, comparing elements from `nums1` and `nums2`.
   - The larger of the two current elements pointed to by the two pointers will be placed at the current position of the merge pointer in `nums1`.
   - After placing an element, we will move the respective pointer (either in `nums1` or `nums2`) and also move the merge pointer backward.

3. **Handling Remaining Elements**:
   - If we finish merging and still have elements left in `nums2`, we need to copy those remaining elements over to `nums1`. This is because if there are remaining elements in `nums2`, they are guaranteed to be smaller than the smallest element that was previously in `nums1`.

4. **Edge Cases**:
   - If there are no elements in `nums2`, `nums1` remains unchanged.
   - If `nums1` is initialized with all zeros and `nums2` has values, then after the merging process, the values from `nums2` will be copied over correctly.

### Python Code:

Here’s how the code looks adhering to the LeetCode format:



```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        # Initialize pointers
        p1 = m - 1  # Last element in nums1's valid part
        p2 = n - 1  # Last element in nums2
        p = m + n - 1  # Last position in nums1 where elements are being placed
        
        # Merge in reverse order
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1
            p -= 1
        
        # If elements left in nums2, copy them
        while p2 >= 0:
            nums1[p] = nums2[p2]
            p2 -= 1
            p -= 1
        
        # No need to copy remaining elements from nums1, they are already in place

```

### Explanation of the Code:
- We define a class `Solution` and a method `merge` which takes four parameters: `self`, `nums1`, `m`, `nums2`, and `n`.
- Pointers `p1`, `p2`, and `p` are initialized to the last valid indices of `nums1`, `nums2`, and the merged array respectively.
- Using a while loop, we compare elements from the two arrays and place the larger one at the correct position in `nums1`.
- After exiting the loop, if there are any remaining elements in `nums2`, they are copied over to `nums1`.
- Since `nums1` already contains any prior elements that are still valid, we do not need to explicitly manage that part.

This algorithm runs in O(m + n) time and utilizes O(1) extra space as we are unwinding the merging process in place.

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
 To determine if a given string `s` is a palindrome while considering only alphanumeric characters and disregarding cases, we can follow a systematic approach:

### Approach:

1. **Filter the String**: We will first filter out all non-alphanumeric characters from the string. This can be achieved using the `str.isalnum()` method which checks if a character is alphanumeric.

2. **Normalize the Case**: Convert all letters in the filtered string to lowercase to ensure that the comparison is case insensitive.

3. **Check for Palindrome**: A string is a palindrome if it reads the same forwards and backwards. We can check this by comparing the filtered string to its reverse.

### Implementation:
Here's the Python code that adheres to the LeetCode solution format:



```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        # Step 1: Filter out non-alphanumeric characters and convert to lowercase
        filtered_chars = []
        
        for char in s:
            if char.isalnum():  # Check if the character is alphanumeric
                filtered_chars.append(char.lower())  # Add the lowercase version
        
        # Step 2: Check if the filtered list is a palindrome
        # Compare the filtered characters with its reverse
        return filtered_chars == filtered_chars[::-1]

```

### Explanation of the Code:
1. **Class Definition**: We define a class `Solution` as required by LeetCode.

2. **Method Definition**: We define the method `isPalindrome` which accepts a string `s` as an input.

3. **Filtering Process**:
   - We initialize an empty list `filtered_chars` to hold the alphanumeric characters.
   - We iterate over each character in the string `s`.
   - For each character, we check if it is alphanumeric using `char.isalnum()`.
   - If it is, we convert it to lowercase and append it to the `filtered_chars` list.

4. **Palindrome Check**:
   - We check if `filtered_chars` is equal to its reverse (`filtered_chars[::-1]`).
   - If they are the same, it means the string is a palindrome and we return `True`, otherwise, we return `False`.

### Complexity Analysis:
- The time complexity of this solution is O(n), where n is the length of the string `s`. This is because we traverse the string once to filter and another time to check if it's a palindrome.
- The space complexity is also O(n) for storing the filtered characters.

This code can be directly run on the LeetCode platform with the mentioned constraints and will function correctly for the given problem statement.

# 141. Linked List Cycle

### Problem Description 
Given `head`, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to. Note that `pos` is not passed as a parameter.

Return `true` if there is a cycle in the linked list. Otherwise, return `false`.


Example 1:
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).


Example 2:
Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.


Example 3:
Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.


Constraints:
The number of the nodes in the list is in the range `[0, 104]`.

`-105 <= Node.val <= 105`
`pos` is `-1` or a valid index in the linked-list.

Follow up: Can you solve it using `O(1)` (i.e. constant) memory?

### Solution 
 To solve the problem of detecting a cycle in a linked list, we can utilize the Floyd’s Cycle Detection Algorithm, which employs the "tortoise and hare" technique. This method is efficient and operates with O(1) space complexity. Below, I'll provide a Python implementation following the LeetCode solution format, along with detailed explanations.

### Problem Solution

**Step-by-step Explanation:**

1. **Understanding the Linked List Structure**:
   - A linked list is composed of nodes, where each node has a `value` and a `next` pointer to the next node.
   - A cycle occurs if one of the nodes' `next` pointer refers back to a previous node in the list.

2. **Using Floyd's Cycle Detection Algorithm**:
   - We will maintain two pointers: `slow` and `fast`.
   - The `slow` pointer moves one step at a time while the `fast` pointer moves two steps.
   - If there is a cycle, the `fast` pointer will eventually meet the `slow` pointer; if not, the `fast` pointer will reach the end of the list (`null`).

3. **Implementation**:
   - We will handle the base case where the `head` is `None`.
   - We will initialize the `slow` and `fast` pointers to point to the `head` of the list.
   - We will then iterate through the list until the `fast` pointer reaches the end or the two pointers meet.

### Python Code

Here's the complete code for detecting a cycle in a linked list:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        # Start with two pointers
        slow = head
        fast = head
        
        # Iterate through the linked list
        while fast is not None and fast.next is not None:
            slow = slow.next          # Move slow pointer one step
            fast = fast.next.next     # Move fast pointer two steps
            
            # Check if the two pointers meet
            if slow == fast:
                return True  # A cycle is detected

        return False  # No cycle found

```

### Explanation of the Code:

1. **ListNode Class**:
   - We define `ListNode` as per the problem statement, representing each node in the linked list.

2. **hasCycle Method**:
   - This method takes the `head` of the linked list as input and returns `True` if a cycle is detected, otherwise `False`.
   - We initialize two pointers: `slow` and `fast`, both starting at `head`.
   - We enter a while loop that continues as long as the `fast` pointer and the `fast.next` are not `None`.
   - Inside the loop, we move the `slow` pointer one step forward and the `fast` pointer two steps forward.
   - If at any point `slow` equals `fast`, it indicates that a cycle is present, and we return `True`.
   - If we exit the loop (meaning `fast` reached the end), we return `False`, indicating no cycle exists.

### Complexity Analysis:
- **Time Complexity**: O(n), where n is the number of nodes in the linked list. In the worst-case scenario, we might traverse all nodes.
- **Space Complexity**: O(1), since we are using a fixed amount of space for the pointers regardless of the input size.

This implementation should run efficiently on LeetCode for the problem as provided.

# 142. Linked List Cycle II

### Problem Description 
Given a linked list, return the node where the cycle begins. If there is no cycle, return `null`.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to. Note that `pos` is not passed as a parameter.

Notice that you should not modify the linked list.


Example 1:
Input: head = [3,2,0,-4], pos = 1
Output: tail connects to node index 1
Explanation: There is a cycle in the linked list, where tail connects to the second node.


Example 2:
Input: head = [1,2], pos = 0
Output: tail connects to node index 0
Explanation: There is a cycle in the linked list, where tail connects to the first node.


Example 3:
Input: head = [1], pos = -1
Output: no cycle
Explanation: There is no cycle in the linked list.


Constraints:
The number of the nodes in the list is in the range `[0, 104]`.

`-105 <= Node.val <= 105`
`pos` is `-1` or a valid index in the linked-list.

Follow up: Can you solve it using `O(1)` (i.e. constant) memory?

### Solution 
 To solve the problem of detecting the starting point of a cycle in a linked list, we'll implement the Floyd's Cycle Detection algorithm, also known as the Tortoise and Hare algorithm. 

### Algorithm Explanation
1. **Floyd's Cycle Detection Concept**: We use two pointers, `slow` and `fast`. The `slow` pointer moves one step at a time, while the `fast` pointer moves two steps at a time. If there is a cycle, `fast` will eventually meet `slow` inside the cycle.

2. **Finding the Starting Node of the Cycle**:
   - Once `slow` and `fast` meet, we reset one pointer (let's say `slow`) to the head of the linked list and keep the other pointer (the one that met `slow`) at the meeting point.
   - Both pointers then move one step at a time. The node where they meet again is the starting point of the cycle.

3. **Edge Cases**: We need to handle cases where the linked list is empty or contains only one node, and no cycle exists.

Here's how we implement this in Python:

### Python Code


```python
# Definition for a singly linked list node.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return None
        
        slow = head
        fast = head
        
        # Phase 1: Determine if a cycle exists
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        else:
            return None  # No cycle exists
        
        # Phase 2: Finding the entry point of the cycle
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        
        return slow  # The node where the cycle begins

# Example usage:
# head = ListNode(3, ListNode(2, ListNode(0, ListNode(-4))))
# head.next.next.next.next = head.next  # Creating a cycle here
# sol = Solution()
# print(sol.detectCycle(head))  # Outputs the node where the cycle begins (2 in this case)

```

### Detailed Explanation of the Code

1. **ListNode Class**: This is the standard definition of a node in the linked list. Each node has a value and a pointer to the next node.

2. **Solution Class**:
    - The `detectCycle` function is defined inside the `Solution` class.
    - It first checks if the list is empty or has only one node. If so, it returns `None` as there can be no cycle.

3. **Phase 1**: The while loop checks for a cycle:
    - We keep moving `slow` by 1 step and `fast` by 2 steps.
    - If `slow` meets `fast`, a cycle exists; otherwise, we exit the loop and return `None`.

4. **Phase 2**: After confirming a cycle, we reset the `slow` pointer to `head` and keep `fast` at the meeting point. By moving both pointers at the same speed, they will meet at the cycle's starting node.

5. **Return Statement**: Finally, we return `slow`, which points to the entry node of the cycle.

This solution operates in O(n) time complexity with O(1) space complexity, which fulfills the problem's requirements and constraints.

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
 To solve the problem of finding the length of the longest substring that contains at most two distinct characters, we can use a sliding window technique along with a hashmap (or dictionary in Python) to count the characters in the current window.

### Algorithm Explanation

1. **Sliding Window Concept**: We maintain a window defined by two pointers (`left` and `right`) that represents the current substring. We will expand the `right` pointer to explore the substring and when the substring contains more than two distinct characters, we will shrink the `left` pointer to reduce the size of the window until we have at most two distinct characters again.

2. **Character Count**: We will keep a count of characters in the current window using a dictionary. This allows us to track how many of each character are present in the window.

3. **Updating Pointers**:
   - Start both pointers at the beginning of the string.
   - Increment the `right` pointer to expand the window and include more characters.
   - When the number of distinct characters exceeds two, increment the `left` pointer until we have only two distinct characters left.
   - Keep track of the maximum length of the valid substring found during this process.

4. **Complexity**: The algorithm will run in O(n) time, where n is the length of the string, since each character is processed a constant number of times (added and removed from the window).

### Implementation

Here is the Python implementation of the above logic formatted according to the LeetCode style:



```python
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        left = 0
        max_length = 0
        char_count = {}

        for right in range(len(s)):
            # Add the current character to the count dictionary
            char_count[s[right]] = char_count.get(s[right], 0) + 1

            # While we have more than 2 distinct characters, shrink the window
            while len(char_count) > 2:
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]
                left += 1
            
            # Update the maximum length found
            max_length = max(max_length, right - left + 1)

        return max_length

```

### Explanation of the Code

1. We define the method `lengthOfLongestSubstringTwoDistinct` with the input parameter `s`.
2. We initialize `left` pointer, `max_length`, and a dictionary `char_count` to count occurrences of characters in the current window.
3. We loop through each character using the `right` pointer:
   - We add the current character to `char_count`.
   - If the length of `char_count` exceeds 2, we enter a while loop to move the `left` pointer to shrink the window until we have at most two distinct characters.
   - We update `max_length` with the maximum size of the current valid substring.
4. Finally, we return `max_length`.

### Example Walkthrough:

For the input `s = "eceba"`:
- Expanding the window:
  - Initially `right` points at 'e', count becomes `{'e': 1}`.
  - Moves to 'c', count becomes `{'e': 1, 'c': 1}` (valid).
  - Moves to 'e', count becomes `{'e': 2, 'c': 1}` (valid).
  - Moves to 'b', count becomes `{'e': 2, 'c': 1, 'b': 1}` (invalid).
  - Adjust `left` to make it valid again.
  - Max length updated during the process.
  
The algorithm finds the length efficiently, handling the constraint on string length easily.

This implementation can be run directly on the LeetCode website, as it adheres to their required function and interface!

# 167. Two Sum II - Input array is sorted

### Problem Description 
Given an array of integers `numbers` that is already sorted in ascending order, find two numbers such that they add up to a specific `target` number.

Return the indices of the two numbers (1-indexed) as an integer array `answer` of size `2`, where `1 <= answer[0] < answer[1] <= numbers.length`.

You may assume that each input would have exactly one solution and you may not use the same element twice.


Example 1:
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore index1 = 1, index2 = 2.


Example 2:
Input: numbers = [2,3,4], target = 6
Output: [1,3]

Example 3:
Input: numbers = [-1,0], target = -1
Output: [1,2]

Constraints:
`2 <= numbers.length <= 3 * 104`
`-1000 <= numbers[i] <= 1000`
`numbers` is sorted in increasing order.

`-1000 <= target <= 1000`
Only one valid answer exists.

### Solution 
 To solve the problem of finding two numbers in a sorted array that add up to a specific target, we can use a two-pointer approach. This method is efficient and works in linear time, O(n), because we only iterate through the array a single time while maintaining two pointers.

### Detailed Explanation of the Two-Pointer Approach:

1. **Initialization**: Start with two pointers. The first pointer, `left`, begins at the start of the array (index 0), and the second pointer, `right`, starts at the end of the array (index `len(numbers) - 1`).

2. **Looping through the array**: While `left` is less than `right`, calculate the sum of the numbers at these two pointers. 

3. **Checking the sum**:
   - If the sum is equal to the target, we have found the two numbers. We can return their indices (adjusted to be 1-indexed).
   - If the sum is less than the target, it means we need a larger number. Thus, we move the `left` pointer to the right (increment it).
   - If the sum is greater than the target, we need a smaller number. Hence, we move the `right` pointer to the left (decrement it).
   
4. **Termination**: Since the problem guarantees that exactly one solution exists, we will find the pair before the loop ends.

Here's the Python code that implements this algorithm:



```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1
        
        while left < right:
            current_sum = numbers[left] + numbers[right]
            
            if current_sum == target:
                return [left + 1, right + 1]  # Return 1-indexed positions
            elif current_sum < target:
                left += 1  # Move left pointer to the right
            else:
                right -= 1  # Move right pointer to the left

        # Since we are guaranteed to have a solution, we will not reach here.

```

### Explanation of the Code:

- We define a `Solution` class and within it, we create a method `twoSum` which takes two parameters: `numbers` (a list of integers) and `target` (the desired sum).
- We initialize two pointers: `left` at 0 and `right` at the last index of the numbers array.
- We enter a while loop that runs as long as `left` is less than `right`.
- Inside the loop, we calculate `current_sum` as the sum of the elements at `left` and `right`.
- Depending on the comparison between `current_sum` and `target`, we either return the 1-indexed positions of `left` and `right` or move the pointers accordingly.
- Since the problem guarantees that there is exactly one solution, we do not handle cases for no valid pairs found.

This code is suitable for direct submission on LeetCode, and it adheres to the input constraints specified in the problem statement.

# 209. Minimum Size Subarray Sum

### Problem Description 
Given an array of positive integers `nums` and a positive integer `target`, return the minimal length of a contiguous subarray `[numsl, numsl+1, ..., numsr-1, numsr]` of which the sum is greater than or equal to `target`. If there is no such subarray, return `0` instead.


Example 1:
Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: The subarray [4,3] has the minimal length under the problem constraint.


Example 2:
Input: target = 4, nums = [1,4,4]
Output: 1

Example 3:
Input: target = 11, nums = [1,1,1,1,1,1,1,1]
Output: 0

Constraints:
`1 <= target <= 109`
`1 <= nums.length <= 105`
`1 <= nums[i] <= 105`
Follow up: If you have figured out the `O(n)` solution, try coding another solution of which the time complexity is `O(n log(n))`.

### Solution 
 To solve the problem of finding the minimal length of a contiguous subarray whose sum is greater than or equal to a given target, we can use a two-pointer (or sliding window) approach to achieve an efficient solution with a time complexity of \(O(n)\).

### Problem Breakdown:
1. **Input**: An array of positive integers `nums` and a positive integer `target`.
2. **Output**: The length of the minimal contiguous subarray whose sum is at least `target`. If no such subarray exists, return `0`.

### Example Understandings:
- For `target = 7` and `nums = [2, 3, 1, 2, 4, 3]`, we are looking for the shortest subarray whose elements add up to at least 7. The valid subarray is `[4, 3]`, which has the length of 2.

### Approach:
1. **Sliding Window**: Use a two-pointer strategy or maintain a window of indices that represents the current subarray sum.
2. **Expand and Contract**: Start expanding by moving the right pointer and adding elements to the current sum. Once the sum reaches or exceeds the `target`, try to contract the window from the left to find the minimal length.
3. **Keep Track of Minimal Length**: Maintain a variable to store the minimum length found during contracting.

### Edge Case:
- If after processing all elements, no valid subarray is found, return `0`.

Here’s how we can implement it in Python, following LeetCode's solution format:



```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        # Initialize variables
        left = 0
        current_sum = 0
        min_length = float('inf')  # We will store the minimum length of the valid subarray
        
        # Iterate over the array with the right pointer
        for right in range(len(nums)):
            current_sum += nums[right]  # Add the current number to the current sum
            
            # While the current sum is greater than or equal to target
            while current_sum >= target:
                min_length = min(min_length, right - left + 1)  # Update min_length if found a smaller one
                current_sum -= nums[left]  # Subtract the left number to try and shrink the window
                left += 1  # Move the left pointer to the right

        # If min_length was updated, return it; otherwise, return 0
        return min_length if min_length != float('inf') else 0

```

### Explanation of the Code:
- We define a `minSubArrayLen` method in the `Solution` class which takes the target and nums as parameters.
- We use a variable `left` as the starting index of our window and another variable `current_sum` to keep track of the sum of elements within our window.
- We iterate through the `nums` array using a `right` pointer.
- For each element, we add it to `current_sum` and check if `current_sum` is greater than or equal to `target`.
- If it is, we enter a while loop to attempt to shrink the window from the left to find the minimal subarray length by removing elements from the left and updating `min_length` accordingly.
- Finally, if `min_length` was updated, we return its value; if it remains `inf`, we return `0`, indicating no valid subarray found.

This method operates efficiently in linear time and effectively solves the problem as described.

# 234. Palindrome Linked List

### Problem Description 
Given the `head` of a singly linked list, return `true` if it is a palindrome.


Example 1:
Input: head = [1,2,2,1]
Output: true

Example 2:
Input: head = [1,2]
Output: false

Constraints:
The number of nodes in the list is in the range `[1, 105]`.

`0 <= Node.val <= 9`
Follow up: Could you do it in `O(n)` time and `O(1)` space?

### Solution 
 To solve the problem of determining if a singly linked list is a palindrome, we can utilize a two-pointer approach combined with the concept of reversing the second half of the linked list. This solution efficiently checks for palindromic structure while maintaining a linear time complexity of O(n) and a constant space complexity of O(1).

Here’s a detailed explanation of the approach we'll take:

1. **Find the Middle of the Linked List**: We will use two pointers, `slow` and `fast`. The `slow` pointer will advance one step at a time, while the `fast` pointer will advance two steps at a time. By the time the `fast` pointer reaches the end of the list, the `slow` pointer will be positioned at the middle.

2. **Reverse the Second Half of the List**: After locating the middle, we need to reverse the second half of the linked list starting from the `slow` pointer.

3. **Check for Palindrome**: We'll compare the first half of the list with the reversed second half. If all corresponding elements are equal, the linked list is a palindrome.

4. **Restore the Original List (optional)**: To maintain the original structure of the list, we'll reverse the second half back to its original order.

5. **Return the Result**: Based on the comparisons, we will return `True` if the list is a palindrome, otherwise `False`.

Here's the complete Python code that implements this approach in a format compatible with LeetCode:



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        # Function to find the middle of the linked list
        def find_middle(node):
            slow = fast = node
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
            return slow
        
        # Function to reverse a linked list
        def reverse(node):
            prev = None
            current = node
            while current:
                next_node = current.next  # store next node
                current.next = prev       # reverse the link
                prev = current            # move prev up
                current = next_node       # move current up
            return prev  # new head of the reversed list
        
        # Edge case: if the list is empty or has one node
        if head is None or head.next is None:
            return True
        
        # Step 1: Find the middle of the linked list
        mid = find_middle(head)
        
        # Step 2: Reverse the second half of the list
        second_half_start = reverse(mid)
        
        # Step 3: Compare the first and second halves
        first_half = head
        second_half = second_half_start
        
        is_palindrome = True
        while second_half:  # We only need to check until the end of the second half
            if first_half.val != second_half.val:
                is_palindrome = False
                break
            first_half = first_half.next
            second_half = second_half.next
        
        # Step 4: Restore the original list (optional)
        reverse(second_half_start)
        
        return is_palindrome

```

### Explanation of the Code:

- **ListNode Class**: This defines the structure for each node in the linked list.
- **Solution Class**: This class contains the `isPalindrome` method which implements the solution strategy.
- The method `find_middle` identifies the middle node of the list using the fast and slow pointer technique.
- The method `reverse` reverses the linked list starting from a given node.
- In `isPalindrome`, we first handle edge cases, then find the middle, reverse the second half, and compare both halves for equality.
- We optionally reverse the second half back to its original state to keep the input list intact.

This implementation provides an efficient and effective way to determine if a linked list is a palindrome while adhering to the constraints given by the problem statement.

# 259. 3Sum Smaller

### Problem Description 
Given an array of `n` integers `nums` and an integer `target`, find the number of index triplets `i`, `j`, `k` with `0 <= i < j < k < n` that satisfy the condition `nums[i] + nums[j] + nums[k] < target`.

Follow up: Could you solve it in `O(n2)` runtime?

Example 1:
Input: nums = [-2,0,1,3], target = 2
Output: 2
Explanation: Because there are two triplets which sums are less than 2:
[-2,0,1]
[-2,0,3]

Example 2:
Input: nums = [], target = 0
Output: 0

Example 3:
Input: nums = [0], target = 0
Output: 0

Constraints:
`n == nums.length`
`0 <= n <= 300`
`-100 <= nums[i] <= 100`
`-100 <= target <= 100`

### Solution 
 Sure! To solve the problem of finding the number of index triplets `(i, j, k)` in an array `nums` that satisfy the condition `nums[i] + nums[j] + nums[k] < target`, we can use a two-pointer technique. This allows us to achieve an `O(n^2)` runtime, which is efficient given the constraints.

Here's the detailed explanation of the approach:

### Approach:
1. **Sort the Array**: First, we need to sort the `nums` array. Sorting helps in using the two-pointer technique effectively.
   
2. **Use Three Pointers**:
   - The first pointer `i` will iterate through the elements of the array.
   - For each index `i`, we will use two pointers `j` and `k`. The pointer `j` starts from `i + 1` and the pointer `k` starts from the end of the array (`n - 1`).

3. **Check the Triplet Sum**:
   - We will check if the sum of the elements at the indices `i`, `j`, and `k` is less than the `target`.
   - If it is less, then all elements from `j` to `k` will also satisfy the condition (since the array is sorted), and we can count the number of valid triplets which is `k - j`.
   - Additionally, we will move the pointer `j` to the right to check for more possible valid triplets.
   - If the sum is greater than or equal to the target, we will move `k` to the left to try and reduce the sum.

4. **Count Valid Triplets**: We keep track of the total count of valid triplets during the process.

5. **Return the Count**: Finally, return the total count of valid triplets.

### Implementation:
Here’s how this approach can be implemented in Python, formatted as per LeetCode's requirements:



```python
class Solution:
    def threeSumSmaller(self, nums: List[int], target: int) -> int:
        # First, sort the array
        nums.sort()
        n = len(nums)
        count = 0
        
        # Iterate through each element
        for i in range(n - 2):
            j, k = i + 1, n - 1  # Setup two pointers
            
            while j < k:
                triplet_sum = nums[i] + nums[j] + nums[k]
                
                if triplet_sum < target:
                    # All indices from j to k will also give valid triplets
                    count += (k - j)
                    j += 1  # Move j right to find new pairs
                else:
                    # Move k left to find a smaller sum
                    k -= 1
        
        return count

```

### Explanation of the Code:
- **Sorting**: We call `nums.sort()` to sort the array which allows us to use two pointers effectively.
- **Outer Loop**: We loop over each element as a base for the triplet using the index `i`.
- **Inner Loop (Two Pointers)**: For each `i`, pointers `j` and `k` find pairs that work with `i`.
  - If the sum is less than `target`, we calculate how many pairs (from `j` to `k`) will work and count them.
  - If the sum is equal to or greater than the target, we move the `k` pointer to potentially find a smaller sum.
- The result, `count`, will reflect the total number of valid triplets that satisfy the condition.

### Complexity:
- Time Complexity: `O(n^2)` due to the two nested loops.
- Space Complexity: `O(1)` since we are using only a constant amount of additional space.

# 283. Move Zeroes

### Problem Description 
Given an integer array `nums`, move all `0`'s to the end of it while maintaining the relative order of the non-zero elements.

Note that you must do this in-place without making a copy of the array.


Example 1:
Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]

Example 2:
Input: nums = [0]
Output: [0]

Constraints:
`1 <= nums.length <= 104`
`-231 <= nums[i] <= 231 - 1`
Follow up: Could you minimize the total number of operations done?

### Solution 
 To solve the problem of moving all zeros to the end of an array while maintaining the relative order of non-zero elements, we can use a two-pointer technique. This is efficient and allows us to meet the requirement of performing the operation in-place.

### Explanation:

The idea is to use a pointer to track the position of the next non-zero element as we iterate through the array. Here’s how we can break down the solution:

1. **Initialize a Pointer**: We will use a pointer `last_non_zero_found_at` to keep track of the index where we should place the next non-zero element.
  
2. **Iterate Through the Array**: We will loop through the entire array using a for loop:
   - If the current element is not zero, we place it at the `last_non_zero_found_at` index and increment that index.
  
3. **Fill the Rest with Zeros**: After we have processed all elements, we will fill the remaining positions in the array with zeros starting from `last_non_zero_found_at`.

This approach ensures that:
- We traverse the array a minimum number of times (only once for moving non-zeros and once for filling zeros).
- We do this in constant space, not using any additional arrays.

### Python Code:

Here’s how the implementation looks:



```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        :param nums: List[int]: the list of integers
        Do not return anything, modify nums in-place instead.
        """
        last_non_zero_found_at = 0  # Pointer for the position of the next non-zero element
        
        # Move all non-zero elements to the front
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[last_non_zero_found_at] = nums[i]
                last_non_zero_found_at += 1
        
        # Fill the rest of the array with zeros
        for i in range(last_non_zero_found_at, len(nums)):
            nums[i] = 0


```

### How the Implementation Works:

1. **Initialization**: `last_non_zero_found_at` is set to `0` at the beginning.

2. **First Loop**:
   - We iterate through each index `i` from `0` to `len(nums) - 1`.
   - When we find a non-zero number, we place it at the current `last_non_zero_found_at` index and then increment `last_non_zero_found_at`.

3. **Second Loop**:
   - After moving all non-zero elements to the front, we iterate from `last_non_zero_found_at` to the end of the array and fill the remaining positions with zeros.

### Complexity:
- **Time Complexity**: O(n), where n is the length of the input array, since we are traversing the array a constant number of times.
- **Space Complexity**: O(1), as we are not using any additional space that scales with the input size. 

This solution meets the given constraints and guarantees in-place modification of the input array. You can directly copy this code to the LeetCode interface and it should function correctly for the given problem.

# 287. Find the Duplicate Number

### Problem Description 
Given an array of integers `nums` containing `n + 1` integers where each integer is in the range `[1, n]` inclusive.

There is only one repeated number in `nums`, return this repeated number.


Example 1:
Input: nums = [1,3,4,2,2]
Output: 2

Example 2:
Input: nums = [3,1,3,4,2]
Output: 3

Example 3:
Input: nums = [1,1]
Output: 1

Example 4:
Input: nums = [1,1,2]
Output: 1

Constraints:
`2 <= n <= 3 * 104`
`nums.length == n + 1`
`1 <= nums[i] <= n`
All the integers in `nums` appear only once except for precisely one integer which appears two or more times.

Follow up:
How can we prove that at least one duplicate number must exist in `nums`?
Can you solve the problem without modifying the array `nums`?
Can you solve the problem using only constant, `O(1)` extra space?
Can you solve the problem with runtime complexity less than `O(n2)`?

### Solution 
 To solve the problem of finding a duplicate number in an array containing integers with certain properties, we can utilize the **Floyd's Tortoise and Hare (Cycle Detection)** algorithm. This approach is efficient and meets the constraints of using constant extra space and having a runtime complexity of less than O(n²).

### Explanation:
The constraints of the problem indicate that we have an array `nums` with `n + 1` integers in the range `[1, n]`. This means that because there are `n + 1` slots but only `n` possible values, at least one number has to be repeated.

The key insight comes from the observation that the array can be viewed as a linked list where each index points to the value at that position. For example, if `nums[0] = 3`, you can think of it as an edge from index `0` to index `3`. Given that there is a duplicate, this structure will lead to at least one cycle.

1. **Initialization**: We start with two pointers, `tortoise` and `hare`. The tortoise moves one step at a time, while the hare moves two steps at a time.

2. **Finding the intersection point**: Both pointers will eventually meet inside the cycle. Once they do, we know that a cycle exists.

3. **Finding the entrance to the cycle**: To find the duplicate number, we place one pointer at the start of the array and leave the other at the intersection point. We then move both pointers one step at a time. The point at which they meet again will be the duplicate number.

Here's the Python code implementing this logic:



```python
class Solution:
    def findDuplicate(self, nums):
        # Phase 1: Finding the intersection point
        tortoise = nums[0]
        hare = nums[0]
        
        # First step of the hare and tortoise
        tortoise = nums[tortoise]
        hare = nums[nums[hare]]
        
        while tortoise != hare:
            tortoise = nums[tortoise]
            hare = nums[nums[hare]]
        
        # Phase 2: Finding the entrance to the cycle
        # Move tortoise to the start of the array
        tortoise = nums[0]
        while tortoise != hare:
            tortoise = nums[tortoise]
            hare = nums[hare]
        
        # Return the value of the duplicate number
        return hare

```

### Detailed Steps:
1. **Input/Output**: The function accepts an array `nums` and returns the duplicated integer.
  
2. **Finding Intersection**:
   - Start both `tortoise` and `hare` at the value of the first index.
   - Move the `tortoise` pointer one step and the `hare` pointer two steps.
   - Continue this until they meet, suggesting a cycle exists.

3. **Finding the Entrance**:
   - Reset one pointer (`tortoise`) to the first element of the array and keep the other pointer at the intersection.
   - Move both pointers at the same speed (one step at a time) until they meet. The point they meet again is the duplicated number.

### Conclusion:
The algorithm efficiently finds the duplicate number in linear time, O(n), while using O(1) additional space, satisfying all constraints set in the problem. The approach utilizes the properties of cycles in linked structures, which is a clever use of pointers.

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
 To solve the problem of finding the length of the longest substring containing at most `k` distinct characters, we can utilize a sliding window technique combined with a hash map (or a Counter) to keep track of the characters in the current window of the substring. 

### Problem Breakdown:
1. **Sliding Window**: We'll maintain a window represented by two indices (usually `start` and `end`) that iterate through the characters of the string.
2. **Hash Map / Counter**: This will be used to count the occurrences of characters within the current window.
3. **Adjusting the Window**: If the number of distinct characters in the window exceeds `k`, we need to shrink the window from the left until it is valid again (i.e., contains at most `k` distinct characters).
4. **Tracking the Maximum Length**: During the process, we'll track the length of the longest valid substring found.

### Implementation Steps:
1. Initialize `start`, `max_length`, and a dictionary to count characters (or a Counter).
2. Iterate through the string using the `end` pointer.
3. Add the current character to the counter.
4. If the counter exceeds `k` distinct characters, increment `start` to reduce the count until we're back to `k` distinct characters.
5. Update `max_length` whenever a valid window is found.

### Python Code:
Here's the complete implementation in the LeetCode format:



```python
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        if k == 0:  # Edge case: No distinct characters
            return 0
        
        # Dictionary to store the count of characters in the current window
        char_count = {}
        start = 0
        max_length = 0
        
        # Iterate with the `end` pointer
        for end in range(len(s)):
            # Add the current character to the hash map
            if s[end] in char_count:
                char_count[s[end]] += 1
            else:
                char_count[s[end]] = 1
            
            # While there are more than `k` distinct characters, shrink the window
            while len(char_count) > k:
                char_count[s[start]] -= 1
                if char_count[s[start]] == 0:
                    del char_count[s[start]]
                start += 1
            
            # Calculate the maximum length found
            max_length = max(max_length, end - start + 1)
        
        return max_length

```

### Explanation of the Code:
1. **Initialization**:
   - We handle the edge case where `k` is `0` right away by returning `0`.
   - We initialize `char_count` to keep track of character occurrences, `start` for the beginning of the sliding window, and `max_length` to store the maximum length found.

2. **Iterating Over the Characters**:
   - We loop through each character using the `end` index.
   - We update the count of the character in the `char_count` dictionary.

3. **Validating the Window**:
   - If the number of distinct characters exceeds `k`, we shrink the window from the left by advancing the `start` pointer.
   - During this process, if the count of a character drops to zero, we remove it from the dictionary.

4. **Calculate Maximum Length**:
   - For every valid window (where distinct character count ≤ `k`), we calculate the current window length and update `max_length` accordingly.

This algorithm effectively runs in O(n) time complexity, where n is the length of the string, since each character is processed at most twice (once added and once removed). The space complexity is O(k), which corresponds to the size of the character count map. 

You can run this code directly on the LeetCode website, and it should work fine for the given problem constraints.

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
 To solve the problem of reversing a string represented as an array of characters in Python, we'll modify the input array in place without using extra space for another array. The key is to swap characters from the two ends of the array moving towards the center.

### Plan

1. **Two-Pointer Technique**: We'll use two pointers — one starting at the beginning of the array (`left`) and the other starting at the end of the array (`right`).
  
2. **Swapping**: We will swap the characters at these two pointers and then move the pointers toward the center. Specifically, we will increment the `left` pointer and decrement the `right` pointer after each swap.

3. **Continue until the pointers meet**: We will keep swapping until the two pointers cross each other. At that point, the string will be reversed.

### Complexity Analysis
- **Time Complexity**: O(n), where n is the length of the string, because we need to look at each character exactly once.
- **Space Complexity**: O(1), as we are not using any additional data structures that grow with the input size.

### Code Implementation
Here is the Python function that implements the above logic in the format suitable for LeetCode:



```python
class Solution:
    def reverseString(self, s: list[str]) -> None:
        """
        Reverses the input list of characters in place.
        """
        left, right = 0, len(s) - 1
        
        while left < right:
            # Swap the characters at the left and right indices
            s[left], s[right] = s[right], s[left]
            # Move towards the middle
            left += 1
            right -= 1

# Example usage:
# s = ["h","e","l","l","o"]
# sol = Solution()
# sol.reverseString(s)
# print(s)  # Output will be ["o","l","l","e","h"]

```

### Explanation of the Code:
1. **Class Definition**: We define a class `Solution` which contains our method `reverseString`.
  
2. **Method Signature**: The method takes a list of characters `s` as input and returns `None` as it modifies the list in place.

3. **Initialization**: We initialize `left` to 0 (the start of the list) and `right` to `len(s) - 1` (the last index of the list).

4. **While Loop**: We use a while loop that continues as long as `left` is less than `right`. Inside the loop:
    - We swap the characters at indices `left` and `right`.
    - After each swap, we move the `left` pointer one step to the right and the `right` pointer one step to the left.

5. **Edge Cases**: This implementation correctly handles edge cases such as single-character lists or empty lists since the loop condition will prevent any swaps if these situations arise.

You can copy and paste this implementation directly into the LeetCode environment, and it should run correctly with the provided examples.

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
 Certainly! Below is the Python code that solves the problem of reversing only the vowels in a given string, along with a detailed explanation in a format suitable for running directly on LeetCode.

### Problem Explanation

We need to reverse the positions of all the vowels in a string while keeping the positions of consonants unchanged. The vowels we consider are `a`, `e`, `i`, `o`, and `u` (both uppercase and lowercase).

### Example Breakdown

1. **Input:** `"hello"`
   - The vowels are `e` and `o`.
   - By reversing them, we get `"holle"`.

2. **Input:** `"leetcode"`
   - The vowels are `e`, `e`, and `o`.
   - By reversing them, we get `"leotcede"`.

### Plan

1. Identify all the vowels in the string and store them in a list.
2. Create a new array to hold the characters of the final result.
3. Iterate through the original string:
   - If the character is a vowel, pop from the list of vowels (which will give us the vowels in reverse order).
   - If it’s not a vowel, keep the character as is.
4. Join the result array to form the final string.

### Code Implementation

Here’s the implementation based on the explained approach:



```python
class Solution:
    def reverseVowels(self, s: str) -> str:
        # Define the set of vowels for easy lookup
        vowels = set("aeiouAEIOU")
        
        # List to hold the vowels found in the string
        vowel_list = [char for char in s if char in vowels]
        
        # Create a list to store the result
        result = list(s)
        
        # Pointer to the index in the vowel_list
        vowel_index = len(vowel_list) - 1
        
        # Iterate through the string and replace vowels with the reversed ones
        for i in range(len(result)):
            if result[i] in vowels:
                result[i] = vowel_list[vowel_index]
                vowel_index -= 1
        
        # Join the result list back into a string and return
        return ''.join(result)

# Example usage:
# sol = Solution()
# print(sol.reverseVowels("hello"))  # Output: "holle"
# print(sol.reverseVowels("leetcode")) # Output: "leotcede"

```

### Explanation of the Code

1. **Vowel Set:** We define a set containing all vowels to facilitate quick checks when we determine if a character is a vowel.
2. **Vowel Collection:** We use a list comprehension to collect all vowels from the string `s` into `vowel_list`.
3. **Result Preparation:** We convert the original string into a list named `result`, making it easier to modify characters directly.
4. **Reversing Process:**
   - The `for` loop iterates through the indices of `result`.
   - If a character at the current index is a vowel, we replace it with the last vowel in `vowel_list` and decrement `vowel_index` to move to the previous vowel for subsequent replacements.
5. **Final Output:** We join the modified list back into a string and return it.

### Complexity Analysis

- **Time Complexity:** O(n), where n is the length of the string `s`. We traverse the string a couple of times.
- **Space Complexity:** O(n) for storing the vowels in `vowel_list` and the result.

You can copy the provided code and paste it into the LeetCode editor, where it will run correctly for the given problem.



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
 To solve the problem of finding the intersection of two integer arrays where the elements in the result must be unique, we can use Python sets, as sets inherently contain unique elements and provide efficient operations for intersection.

Here's a detailed explanation followed by the code formatted according to LeetCode's requirements:

### Approach:
1. **Use Sets**: Convert both arrays `nums1` and `nums2` into sets. This will automatically handle duplicate values and allow us to easily find common elements.
  
2. **Intersection of Sets**: Utilize the set intersection operation to find the common elements between the two sets. This will result in another set containing only the unique elements that are present in both input arrays.

3. **Convert to List**: Since the problem requires us to return a list, we will convert the resulting set back into a list.

4. **Return the Result**: Finally, we will return the list as the output.

This approach is both time-efficient and space-efficient for the given constraints.

### Python Code:
Here's the implementation in Python:



```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Convert both lists to sets to handle duplicates 
        set1 = set(nums1)
        set2 = set(nums2)
        
        # Find the intersection of both sets
        intersection_set = set1 & set2
        
        # Convert the result of the intersection set back to a list
        return list(intersection_set)


```

### Explanation of the Code:
- **Class definition**: The code is encapsulated in a class named `Solution`, which is a common format in LeetCode problems.
  
- **Method**: The method `intersection` receives two parameters: `nums1` and `nums2`, which are lists of integers.

- **Creating Sets**: We create `set1` and `set2` from `nums1` and `nums2`, respectively. This ensures that all duplicate elements are removed.

- **Intersection Calculation**: The `&` operator is used to find the intersection between the two sets, storing the result in `intersection_set`.

- **Return Statement**: Finally, we convert `intersection_set` back into a list and return it.

### Complexity Analysis:
- **Time Complexity**: O(n + m), where n and m are the lengths of `nums1` and `nums2` respectively. This is because we are converting each list to a set, which takes linear time.

- **Space Complexity**: O(n + m) for storing the two sets.

### Example Usage:
You can use the provided class and method to find the intersection of two integer arrays in a LeetCode environment. Just call the `intersection` method with the appropriate arguments to get the result.

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
 To solve the problem of finding the intersection of two integer arrays while keeping track of the frequency of elements, we can utilize a hashmap (or dictionary in Python) to efficiently count occurrences of each element from one of the arrays, and then check those counts against the other array.

Here's how you can approach the solution step-by-step:

1. **Count Elements**: Use a dictionary to count how many times each element appears in `nums1`.
2. **Find Intersection**: Iterate through `nums2` and for each element, check if it exists in the dictionary. If it does, append it to the result list and decrease the count in the dictionary.
3. **Return the Result**: After processing `nums2`, the result list will contain the intersection elements with their respective counts.

Now, let's write the Python solution in the LeetCode format:



```python
from collections import Counter
from typing import List

class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Step 1: Count the occurrences of each number in nums1
        count1 = Counter(nums1)
        
        # Step 2: Prepare the result list
        result = []
        
        # Step 3: Iterate through nums2 and find the intersection
        for num in nums2:
            if count1[num] > 0:  # If the number is in nums1
                result.append(num)  # Add to result
                count1[num] -= 1    # Decrement the count
        
        # Step 4: Return the result
        return result

```

### Explanation of the Code:

1. **Imports**: We import `Counter` from the `collections` module for counting the occurrences of elements efficiently, and `List` for type hints.
  
2. **Class Definition**: We define a `Solution` class as per LeetCode's structure.

3. **Method Definition**: The method `intersect` takes two lists of integers (`nums1` and `nums2`) as input.

4. **Counting Elements**: We create a `Counter` object named `count1` which automatically counts the occurrences of each unique integer in `nums1`.

5. **Iterating through nums2**: We loop through each integer in `nums2`. If the integer exists in `count1` with a count greater than 0, we:
   - Append it to the `result` list.
   - Decrease the count of that integer in the `Counter`, ensuring that we only include it as many times as it appears in both arrays.

6. **Output**: Finally, we return the `result` list, which contains all integers that intersect in both arrays according to their frequency.

### Complexity:
- **Time Complexity**: O(n + m), where n is the length of `nums1` and m is the length of `nums2`. This is because we traverse both arrays once.
- **Space Complexity**: O(n) for the hashmap storage.

### Follow-Up Considerations:
1. If the arrays are already sorted, a two-pointer technique could be used for an optimized solution.
2. If `nums1` is significantly smaller than `nums2`, building the `Counter` for `nums1` is efficient, and we could potentially loop through the larger array `nums2`.
3. If `nums2` elements are on disk, consider reading them in chunks, using a hash set to track elements and their counts as we read them.

This implementation provides an effective and efficient solution to the problem of finding the intersection of two lists in Python.

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
 To solve the problem of applying a quadratic function to a sorted array and returning it in sorted order, we can make use of two-pointer technique to achieve a linear time complexity solution, O(n).

### Explanation:

The quadratic function we need to apply to each integer `x` in the array is given by:

\[ f(x) = ax^2 + bx + c \]

The main characteristics of this function are:

1. If `a > 0`, the function opens upwards and the minimum value will be found at the vertex when the input values are smaller than the vertex.
2. If `a < 0`, the function opens downwards and the maximum value will be at the vertex when the input values are larger than the vertex.
3. If `a == 0`, the function simplifies to a linear function.

### Steps to Solve the Problem:

1. **Initialization**:
   - We will set up two pointers: one starting at the beginning of the array (`left`) and the other at the end of the array (`right`).
   - Create a result array to store the transformed values.

2. **Evaluate the Function**:
   - Depending on the sign of `a`, we will calculate the values from the `left` and `right` pointers.
   - If `a >= 0`: We fill the result array from the end to the beginning (largest to smallest).
     - Compare the function applied to `nums[left]` and `nums[right]`.
     - The larger of the two values should go in the rightmost position of the result array.
   - If `a < 0`: The process is similar, but we will fill the result array from beginning to end.

3. **Pointer Adjustment**:
   - After placing an element in the result array, adjust the corresponding pointer (`left` or `right`).

4. **Return the Sorted Result Array**.

Here is the straight Python implementation of the described approach, formatted to run on LeetCode:



```python
class Solution:
    def sortTransformedArray(self, nums: List[int], a: int, b: int, c: int) -> List[int]:
        n = len(nums)
        result = [0] * n
        
        # Two pointers
        left, right = 0, n - 1
        
        if a == 0:
            # When a is zero, simply perform linear function
            for i in range(n):
                result[i] = b * nums[i] + c
            return result if b >= 0 else result[::-1]

        # Fill the result array based on the sign of a
        index = n - 1 if a > 0 else 0
        
        while left <= right:
            left_val = a * nums[left] ** 2 + b * nums[left] + c
            right_val = a * nums[right] ** 2 + b * nums[right] + c
            
            if a > 0:  # Quadratic function opens upwards
                if left_val > right_val:
                    result[index] = left_val
                    left += 1
                else:
                    result[index] = right_val
                    right -= 1
                index -= 1
            else:  # Quadratic function opens downwards
                if left_val < right_val:
                    result[index] = left_val
                    left += 1
                else:
                    result[index] = right_val
                    right -= 1
                index += 1
        
        return result

```

### Detailed Breakdown of Each Part:

1. **Function Signature**: 
   - The class `Solution` contains the method `sortTransformedArray`, which takes the sorted list `nums` and coefficients `a`, `b`, `c`.

2. **Handling Edge Cases**:
   - If `a == 0`, it simplifies our task as we only deal with a linear function.

3. **Two-Pointer Logic**:
   - Depending on the curve's direction (`a`), we either build the result in decreasing order (for `a > 0`) or increasing order (for `a < 0`).

4. **Time Complexity**:
   - The overall time complexity is O(n) since we only iterate through the list once with two pointers.

This approach ensures efficient and correct transformation and sorting of the array.

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
 To solve this problem, we can use a sliding window approach to efficiently determine the length of the longest substring that contains all the same letters after performing at most `k` operations. Here's a detailed explanation of the approach we will take:

### Problem Breakdown
1. **Understand the Operations**: We can change any character in the string up to `k` times to potentially create the longest segment of repeating characters.
2. **Sliding Window Technique**: We will maintain a window that represents a substring of `s`. This window expands to the right as we try to include more characters, and it contracts from the left when the conditions are not met.
3. **Character Frequency Tracking**: We need to keep track of the frequency of characters within the current window to determine how many operations we would need to convert the characters in the window to the most frequent character.
4. **Adjusting the Window**: When the operations needed to make all characters in the window the same (which is calculated using the formula `current window size - frequency of most common character`) exceed `k`, we will slide the left boundary of the window to the right.

### Steps to Implement
1. Initialize pointers for the sliding window and a count array to track the frequency of characters.
2. Loop through each character in the string while maintaining the right pointer.
3. Update the frequency of the character at the right pointer.
4. Calculate the number of operations needed and adjust the left pointer if necessary.
5. Keep track of the maximum length of valid windows found during the process.

### Python Code

Here’s the complete solution using Python:



```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        # Array to store the frequency of each character
        count = [0] * 26
        max_count = 0  # To track the count of the most frequent character
        left = 0
        result = 0  # To track the maximum length of valid window
        
        for right in range(len(s)):
            # Update the frequency of the current character
            current_char_index = ord(s[right]) - ord('A')
            count[current_char_index] += 1
            
            # Update the max count of a single character in the current window
            max_count = max(max_count, count[current_char_index])
            
            # Calculate the number of characters that need to be replaced
            while (right - left + 1) - max_count > k:
                # If we need more than k replacements, slide the left pointer
                count[ord(s[left]) - ord('A')] -= 1
                left += 1
            
            # Update the result with the maximum length
            result = max(result, right - left + 1)
        
        return result

```

### Explanation of the Code:
- **Line 2-3**: We define a frequency array `count` which tracks occurrences of each letter from 'A' to 'Z'. `max_count` keeps track of the highest frequency of a single character in the current window.
- **Line 6-18**: We use a for loop to iterate over each character in the string with `right` being the end of the window.
- **Lines 8-10**: For each character at index `right`, we update its frequency and determine if it’s the most common character in the current window.
- **Lines 12-14**: We use a while loop to check if we can still make the current window valid under the `k` operations limit. If not, we move the left side of the window rightward until it becomes valid again.
- **Line 17**: Update `result` with the length of the valid window if it’s longer than previously found lengths.

### Complexity Analysis
- **Time Complexity**: O(n), where n is the length of the string. Each character is processed at most twice (once when the right pointer expands and at most once when the left pointer moves).
- **Space Complexity**: O(1), since the count array's size is constant (for the 26 uppercase letters).

This solution efficiently finds the length of the longest substring with up to `k` changes and fits the constraints provided.

# 457. Circular Array Loop

### Problem Description 
You are playing a game involving a circular array of non-zero integers `nums`. Each `nums[i]` denotes the number of indices forward/backward you must move if you are located at index `i`:
If `nums[i]` is positive, move `nums[i]` steps forward, and
If `nums[i]` is negative, move `nums[i]` steps backward.

Since the array is circular, you may assume that moving forward from the last element puts you on the first element, and moving backwards from the first element puts you on the last element.

A cycle in the array consists of a sequence of indices `seq` of length `k` where:
Following the movement rules above results in the repeating index sequence `seq[0] -> seq[1] -> ... -> seq[k - 1] -> seq[0] -> ...`
Every `nums[seq[j]]` is either all positive or all negative.

`k > 1`
Return `true` if there is a cycle in `nums`, or `false` otherwise.


Example 1:
Input: nums = [2,-1,1,2,2]
Output: true
Explanation:
There is a cycle from index 0 -> 2 -> 3 -> 0 -> ...

The cycle's length is 3.


Example 2:
Input: nums = [-1,2]
Output: false
Explanation:
The sequence from index 1 -> 1 -> 1 -> ... is not a cycle because the sequence's length is 1.

By definition the sequence's length must be strictly greater than 1 to be a cycle.


Example 3:
Input: nums = [-2,1,-1,-2,-2]
Output: false
Explanation:
The sequence from index 1 -> 2 -> 1 -> ... is not a cycle because nums[1] is positive, but nums[2] is negative.

Every nums[seq[j]] must be either all positive or all negative.


Constraints:
`1 <= nums.length <= 5000`
`-1000 <= nums[i] <= 1000`
`nums[i] != 0`
Follow up: Could you solve it in `O(n)` time complexity and `O(1)` extra space complexity?

### Solution 
 To solve the problem of detecting cycles in a circular array based on movement rules defined by the values in the array, we can utilize a modified approach similar to the "Floyd's cycle-finding algorithm." This technique, also known as the tortoise and hare algorithm, can efficiently identify cycles in a linked structure with a time complexity of O(n) and an additional space complexity of O(1).

### Detailed Explanation:

1. **Input Understanding**: Each element in the `nums` array indicates how many steps to move. Positive values mean moving forward in the array, while negative values mean moving backward. The array is circular, so if you go past the end, you wrap around to the beginning, and vice versa.

2. **Cycle Definition**: A cycle exists if you can return to the starting index after a series of moves, and all moves in that cycle must be in the same direction (all positive or all negative).

3. **Strategy**:
    - We can explore the array using two pointers: a slow pointer (`slow`) that moves one step at a time and a fast pointer (`fast`) that moves two steps at a time.
    - We first identify a cycle by moving the pointers. If they meet in an index, we have detected a cycle.
    - To ensure that the cycle is valid (length > 1 and all values are either all positive or all negative), we track the direction of movement based on the sign of the starting number of the cycle.

4. **Visited Marking**: Although the problem states O(1) space complexity, we'll modify elements in `nums` temporarily to mark visited indices. This is safe, given that all values in `nums` are guaranteed to be non-zero.

5. **Implementation**: We iterate through the array for every unvisited index, applying the above logic. 

Here is the Python code following the LeetCode solution format:



```python
class Solution:
    def circularArrayLoop(self, nums: List[int]) -> bool:
        n = len(nums)

        for i in range(n):
            slow = i
            fast = i
            
            # Need to handle direction, use a variable to detect if current segment is valid
            direction = nums[i] > 0
            
            while True:
                slow = self.get_next_index(nums, slow, direction)
                fast = self.get_next_index(nums, fast, direction)
                if fast != -1:
                    fast = self.get_next_index(nums, fast, direction)

                # Check conditions for breaking the loop (cycle found)
                if slow == -1 or fast == -1 or slow == fast:
                    break
            
            # Verify if we have a cycle of length greater than 1
            if slow != -1 and slow == fast:
                # Check the length of the cycle
                cycle_length = 0
                while True:
                    cycle_length += 1
                    if slow == fast and cycle_length > 1:
                        return True
                    slow = self.get_next_index(nums, slow, direction)
                    if slow == -1:
                        break

            # Mark all elements part of the cycle to avoid revisiting
            # Reset visited elements to a neutral sign
            mark_dir = direction * -1  # inverse direction to mark
            slow = i
            while True:
                next_index = self.get_next_index(nums, slow, direction)
                if next_index == -1 or (next_index == i or nums[next_index] * mark_dir < 0):
                    break
                nums[slow] = mark_dir  # Marking as visited
                slow = next_index
        
        return False
    
    def get_next_index(self, nums, index, direction):
        step = nums[index]
        next_index = (index + step) % len(nums)
        if (step > 0 and not direction) or (step < 0 and direction):
            return -1  # Invalid move, doesn't follow the detected direction
        return next_index


```

### Explanation of Code:
- The outer loop iterates through the `nums` array and selects each index as a starting point for detecting cycles.
- The inner while-loop moves the slow and fast pointers according to the rules defined.
- The method `get_next_index()` calculates the next index and checks if it stays within the movement direction.
- Marking visited elements ensures we don't revisit any segment of the array, complying with O(1) space requirements by modifying the original array. 

This implementation successfully detects the specified cycle conditions in efficient time and space complexity as required.

# 487. Max Consecutive Ones II

### Problem Description 
Given a binary array, find the maximum number of consecutive 1s in this array if you can flip at most one 0.


Example 1:
Input: [1,0,1,1,0]
Output: 4
Explanation: Flip the first zero will get the the maximum number of consecutive 1s.

    After flipping, the maximum number of consecutive 1s is 4.

Note:
The input array will only contain `0` and `1`.

The length of input array is a positive integer and will not exceed 10,000
Follow up:
What if the input numbers come in one by one as an infinite stream? In other words, you can't store all numbers coming from the stream as it's too large to hold in memory. Could you solve it efficiently?

### Solution 
 To solve the problem of finding the maximum number of consecutive 1s in a binary array, where we are allowed to flip at most one 0 to 1, we can use an efficient algorithm that involves maintaining a sliding window over the array. Here's a detailed step-by-step explanation followed by the Python solution formatted for LeetCode.

### Approach:

1. **Sliding Window Technique**: We'll use two pointers to represent a sliding window. As we iterate through the array, we'll expand the right end of the window and count the zeros within it.

2. **Count Zeros**: We will maintain a count of zeros in the current window. If the count of zeros exceeds 1 (since we can flip one 0), we will move the left pointer of the window to reduce the count of zeros.

3. **Calculate Maximum Length**: For every position of the right pointer, if the window is valid (i.e., contains at most one zero), we will calculate the length of the window and update our maximum length if this length is larger than the previous maximum.

### Complexity:
- The time complexity will be O(n), where n is the length of the input array. This is because we process each element at most twice (once when extending the right pointer and once when moving the left pointer).
- The space complexity is O(1), as we're only using a constant amount of additional space for variables.

### Python Code:



```python
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = 0
        max_length = 0
        zero_count = 0
        
        for right in range(len(nums)):
            # If we encounter a 0, we count it
            if nums[right] == 0:
                zero_count += 1
            
            # If we have more than one 0 in our current window, shrink from the left
            while zero_count > k:
                if nums[left] == 0:
                    zero_count -= 1
                left += 1
            
            # Calculate the maximum length of the valid window
            max_length = max(max_length, right - left + 1)
        
        return max_length

```

### Explanation of Code:
- We define the `longestOnes` function that takes a list `nums` and an integer `k`, where `k` is the maximum count of zeros we can flip (in this case, k=1).
- We initialize `left` as the starting pointer of our window, `max_length` to keep track of the maximum length found, and `zero_count` to count the number of zeros in the current window.
- We iterate through the array using the `right` pointer.
- When we find a 0, we increment `zero_count`.
- If `zero_count` exceeds `k`, we enter a while loop to move the `left` pointer to the right until we have at most `k` zeros.
- After adjusting the pointers, we compute the current valid window length and update `max_length` if this new length is larger than the previous maximum.
- Finally, we return `max_length`.

This solution efficiently handles finding the longest sequence of consecutive 1s, with the allowance of flipping one 0, while remaining efficient in both time and space.

# 524. Longest Word in Dictionary through Deleting

### Problem Description 
Given a string `s` and a string array `dictionary`, return the longest string in the dictionary that can be formed by deleting some of the given string characters. If there is more than one possible result, return the longest word with the smallest lexicographical order. If there is no possible result, return the empty string.


Example 1:
Input: s = "abpcplea", dictionary = ["ale","apple","monkey","plea"]
Output: "apple"

Example 2:
Input: s = "abpcplea", dictionary = ["a","b","c"]
Output: "a"

Constraints:
`1 <= s.length <= 1000`
`1 <= dictionary.length <= 1000`
`1 <= dictionary[i].length <= 1000`
`s` and `dictionary[i]` consist of lowercase English letters.

### Solution 
 To solve the problem of finding the longest string in a given dictionary that can be formed by deleting some characters from the string `s`, we can follow these steps:

1. **Understand the Problem**: We need to check which words from the dictionary can be formed from the characters in `s` in order. The task involves determining if each word can be a subsequence of `s`, and if multiple words have the same length, we should choose the lexicographically smallest one.

2. **Subsequence Check**: A string `word` is a subsequence of `s` if we can find all characters of `word` in `s` in the same order. This can be done using two pointers: one for `s` and another for `word`.

3. **Iterate Over Dictionary**: We will iterate over each word in the dictionary and check if it can be formed as a subsequence of `s`. We will keep track of the longest word found that meets the criteria.

4. **Condition Handling**: While doing this, we need to handle two conditions:
   - Length of the matching words.
   - Lexicographical order when there are ties in length.

5. **Return Result**: Finally, we will return the longest valid string found or an empty string if none matched.

Now, let’s implement this logic in Python:



```python
class Solution:
    def findLongestWord(self, s: str, dictionary: list[str]) -> str:
        def is_subsequence(word: str) -> bool:
            it = iter(s)
            return all(char in it for char in word)

        longest_word = ""
        
        for word in dictionary:
            if is_subsequence(word):
                # Check if the word is longer than the current longest found
                # or if it's the same length but lexicographically smaller
                if (len(word) > len(longest_word) or
                   (len(word) == len(longest_word) and word < longest_word)):
                    longest_word = word

        return longest_word

```

### Explanation of the Code:
1. **Function Definition**: We define a class `Solution` and a method `findLongestWord` that takes a string `s` and a list of strings `dictionary`.
  
2. **Subsequence Check**: The inner function `is_subsequence` uses a generator expression that checks if all characters of `word` can be found in `s` in order. This is efficient and avoids using extra space.

3. **Iteration**: We loop through each `word` in the `dictionary`. For each word, we check if it can be formed from `s`:

   - If it can be formed, we need to check if it's the longest one found so far or if it is equal in length but smaller lexicographically.
   - If either condition is satisfied, we update `longest_word`.

4. **Final Return**: After checking all words, we return the `longest_word` found. If no valid word is found, it remains an empty string.

### Complexity:
- The time complexity is O(n * m), where n is the total number of characters in `s` and m is the length of the longest word being checked from `dictionary`. Given the constraints, this solution is efficient.

### Edge Cases:
- If `s` contains no characters that match any words from `dictionary`, the method will naturally return an empty string.
- All characters in `s` being used may also need consideration in performance due to maximum constraints.

By following this approach, the solution is both effective and efficient for the problem at hand.

# 532. K-diff Pairs in an Array

### Problem Description 
Given an array of integers `nums` and an integer `k`, return the number of unique k-diff pairs in the array.

A k-diff pair is an integer pair `(nums[i], nums[j])`, where the following are true:
`0 <= i, j < nums.length`
`i != j`
`|nums[i] - nums[j]| == k`
Notice that `|val|` denotes the absolute value of `val`.


Example 1:
Input: nums = [3,1,4,1,5], k = 2
Output: 2
Explanation: There are two 2-diff pairs in the array, (1, 3) and (3, 5).

Although we have two 1s in the input, we should only return the number of unique pairs.


Example 2:
Input: nums = [1,2,3,4,5], k = 1
Output: 4
Explanation: There are four 1-diff pairs in the array, (1, 2), (2, 3), (3, 4) and (4, 5).


Example 3:
Input: nums = [1,3,1,5,4], k = 0
Output: 1
Explanation: There is one 0-diff pair in the array, (1, 1).


Example 4:
Input: nums = [1,2,4,4,3,3,0,9,2,3], k = 3
Output: 2

Example 5:
Input: nums = [-1,-2,-3], k = 1
Output: 2

Constraints:
`1 <= nums.length <= 104`
`-107 <= nums[i] <= 107`
`0 <= k <= 107`

### Solution 
 To solve the problem of finding the number of unique k-diff pairs in a given array, we can utilize a few data structures efficiently: a set to track unique values and a dictionary (or a counter) to count occurrences of each number.

### Problem Breakdown

1. **Understanding k-diff pairs**: A k-diff pair `(a, b)` satisfies the condition `|a - b| = k`. This gives us two situations to consider:
   - `a - b = k` → `a = b + k`
   - `b - a = k` → `b = a + k`

2. **Handling k = 0**: If `k` is 0, we are looking for pairs `(a, a)`. This means we need to count how many numbers appear more than once in the list.

3. **Using a set for unique values**: We can store numbers in a set to automatically handle uniqueness without worrying about duplicates in the `nums`.

4. **Using a dictionary for counting**: We can keep track of how many times each number appears, which helps in determining pairs.

### Detailed Steps
1. Create a dictionary to count the occurrences of each number in `nums`.
2. Loop through each unique number in the dictionary and check:
   - If `k > 0`: Check if number + k exists in the dictionary.
   - If `k == 0`: Check if the number appears at least twice.
3. Count how many valid pairs exist based on the checks above.

Here is the Python code implementing this logic, formatted according to the LeetCode submission guidelines:



```python
from collections import Counter

class Solution:
    def findPairs(self, nums: List[int], k: int) -> int:
        if k < 0:
            return 0  # k must be non-negative
        
        count = 0
        num_count = Counter(nums)  # Count the occurrences of each number
        
        for num in num_count:
            if k > 0:
                if num + k in num_count:
                    count += 1
            elif k == 0:
                if num_count[num] > 1:
                    count += 1
                    
        return count

```

### Explanation of the Code:
1. **Imports**: We import `Counter` from `collections` to easily count occurrences of items in the list.
2. **Function Definition**: The function `findPairs` is defined to take a list of integers `nums` and an integer `k`.
3. **Early Exit for Negative k**: If `k < 0`, we return 0 because absolute difference cannot be negative.
4. **Count Occurrences**: We use `Counter` to create a dictionary-like object `num_count`, which counts how many times each number appears in `nums`.
5. **Checking for Pairs**:
   - **When k > 0**: We check if `num + k` exists in our counts, and if it does, it's a valid pair.
   - **When k == 0**: We check if the count of the number is greater than 1 to confirm the existence of the pair `(num, num)`.
6. **Count and Return**: We increment our count for each valid condition met and finally return this count.

### Complexity:
- **Time Complexity**: O(n), where n is the number of elements in `nums`, because we traverse the array once and iterate over the unique counts.
- **Space Complexity**: O(n) in the worst case, where all numbers are unique and stored in the `Counter`.

This solution is efficient and works well within the constraints given in the problem statement.

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
 To solve the problem of determining whether one string (`s1`) is a permutation of a substring of another string (`s2`), we can use a sliding window approach combined with character frequency counting. Below, I'll provide a Python function to achieve this, formatted in a standard way for LeetCode submissions, followed by a detailed explanation of the solution.

### Python Code



```python
from collections import Counter

class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        # Lengths of the input strings
        len1, len2 = len(s1), len(s2)
        
        # If s1 is longer than s2, it can't be a permutation of any substring of s2
        if len1 > len2:
            return False
        
        # Create frequency counts for s1 and the first window of s2
        count1 = Counter(s1)
        count2 = Counter(s2[:len1])
        
        # Check the first window
        if count1 == count2:
            return True
        
        # Sliding window over s2
        for i in range(len1, len2):
            # Include the next character in the window
            count2[s2[i]] += 1
            
            # Exclude the first character of the previous window
            count2[s2[i - len1]] -= 1
            
            # Remove the count from the dictionary if it drops to zero
            if count2[s2[i - len1]] == 0:
                del count2[s2[i - len1]]
                
            # Check if the counts are equal
            if count1 == count2:
                return True
        
        return False

```

### Explanation

1. **Character Frequency Counting**:
   - We utilize Python's `collections.Counter` to simplify the counting of characters in both strings. This allows us to easily compare the frequency of characters between the two strings.

2. **Length Check**:
   - We first check if `s1` is longer than `s2`. If it is, then `s1` cannot be a permutation of any substring of `s2`, so we return `False`.

3. **Initial Frequency Counts**:
   - We compute the frequency count for `s1` and for the first window (substring) of `s2` that has the same length as `s1`.

4. **Direct Comparison**:
   - We first compare the character counts of `s1` and the first window of `s2`. If they match, we immediately return `True`.

5. **Sliding Window**:
   - We then slide over `s2` from the index equal to the length of `s1` to the end of `s2`. For each step:
     - We add the next character that comes into the window.
     - We remove the character that is sliding out of the window.
     - We also check if the count of the character going out becomes zero; if it does, we remove it from the counting dictionary which keeps our checks efficient.

6. **Final Comparison**:
   - After adjusting the counts for the window, we compare the updated character counts. If they match at any point, we return `True`.

7. **Returning Result**:
   - If the entire sliding process completes without finding a match, we return `False`.

### Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `s2`. We process each character of `s2` at most twice (adding and removing from the count).
- **Space Complexity**: O(1), since there are a constant number (at most 26 for lowercase English letters) of characters being counted. 

This solution is efficient and works within the constraints provided in the problem statement.

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
 To solve the problem of finding the smallest range that includes at least one number from each of the `k` sorted lists of integers, we can utilize a min-heap (or priority queue) for efficient processing. The goal is to continuously track the minimum and maximum values among the current candidates from each list and adjust our range accordingly.

### Approach:
1. **Initialization**:
   - Use a min-heap to keep track of the smallest currently selected element from each list. The heap will contain tuples of the form `(value, list_index, value_index)`, indicating the value and its position in the respective list.
   - Initialize the `current_max` variable to track the current maximum value among the selected elements.
   - Initialize a variable to store the best range found so far.

2. **Heap Population**:
   - Start by pushing the first element from each list onto the heap. This initializes the min-heap and sets the `current_max` to the maximum of the first elements.

3. **Extraction and Range Calculation**:
   - Extract the smallest element from the heap (this is our current minimum).
   - Calculate the range between the current min (from the heap) and the `current_max`.
   - If this range is smaller than the best range found so far, update the best range.
   - If the extracted minimum element has a next element in its list, push the next element onto the heap and update `current_max` if this new element is larger than the current maximum.

4. **Termination**:
   - Repeat the extraction and insertion until we can no longer push elements from any list, i.e., one of the lists is fully traversed.

### Implementation:
This approach ensures we explore the smallest ranges efficiently and makes use of the properties of sorted lists.

Here is the Python code encapsulated in the LeetCode format:



```python
import heapq

class Solution:
    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        min_heap = []
        current_max = float('-inf')

        # Initialize the heap and current_max
        for i in range(len(nums)):
            heapq.heappush(min_heap, (nums[i][0], i, 0))  # (value, list index, index in list)
            current_max = max(current_max, nums[i][0])

        best_range = [float('-inf'), float('inf')]

        # Start processing the heap
        while min_heap:
            current_min, list_index, index_in_list = heapq.heappop(min_heap)

            # Update the best range if we found a smaller one
            if (current_max - current_min) < (best_range[1] - best_range[0]) or \
               (current_max - current_min == (best_range[1] - best_range[0]) and current_min < best_range[0]):
                best_range = [current_min, current_max]

            # If we can add the next element from the same list
            if index_in_list + 1 < len(nums[list_index]):
                next_value = nums[list_index][index_in_list + 1]
                heapq.heappush(min_heap, (next_value, list_index, index_in_list + 1))
                current_max = max(current_max, next_value)
            else:
                # If there's no next element, we can't form a complete range anymore
                break

        return best_range

```

### Explanation of Code:
- We define a class `Solution` with a method `smallestRange` which takes a list of lists `nums`.
- Elements are pushed to a min-heap which helps in efficiently retrieving the smallest current number.
- The main loop continues until we cannot push new elements from the lists.
- For every extraction, we check and potentially update the smallest range found.
- The final result is returned as a list containing the best range.

This solution efficiently finds the smallest range while adhering to the constraints and definitions provided by the problem statement.

# 713. Subarray Product Less Than K

### Problem Description 
Your are given an array of positive integers `nums`.

Count and print the number of (contiguous) subarrays where the product of all the elements in the subarray is less than `k`.


Example 1:
Input: nums = [10, 5, 2, 6], k = 100
Output: 8
Explanation: The 8 subarrays that have product less than 100 are: [10], [5], [2], [6], [10, 5], [5, 2], [2, 6], [5, 2, 6].

Note that [10, 5, 2] is not included as the product of 100 is not strictly less than k.

Note:
`0 < nums.length <= 50000`.

`0 < nums[i] < 1000`.

`0 <= k < 10^6`.

### Solution 
 To solve the problem of counting the number of contiguous subarrays where the product of all the elements in the subarray is less than a given value `k`, we can use a two-pointer (or sliding window) approach. This is an efficient method that allows us to maintain a running product of the elements in a window defined by two pointers, thereby avoiding the need for a nested loop which would be less efficient.

## Detailed Explanation:

1. **Initialization**: We initialize two pointers, `left` and `right`. Both will start at the beginning of the array. We will also use a variable `product` to keep track of the product of elements in the current window and a variable `count` to store the number of valid subarrays.

2. **Iterating through the array**: We extend the `right` pointer to include more elements in our current window. For each step, we multiply the `product` by `nums[right]`. 

3. **Check the product**: If the `product` is greater than or equal to `k`, we need to move the `left` pointer to reduce the window size until the `product` is less than `k`. On each realignment of the `left` pointer, we also divide `product` by `nums[left]`.

4. **Counting valid subarrays**: For every valid position of `right`, any subarray ending at `right` and beginning from any index between `left` and `right` (inclusive) will be valid. This contributes to `(right - left + 1)` valid subarrays.

5. **Final result**: The accumulated count will give us the required number of valid subarrays.

Here's the implementation of the above logic in Python, formatted for LeetCode:



```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:
            return 0  # No subarray product can be less than 1 if k is 1 or less.

        left = 0
        product = 1
        count = 0

        for right in range(len(nums)):
            product *= nums[right]  # Include the current number in the product

            # Shrink the window until the product is less than k
            while product >= k:
                product //= nums[left]  # Remove the leftmost number from the product
                left += 1  # Move the left pointer to the right

            # All subarrays from left to right are valid
            count += (right - left + 1)

        return count

```

### Key Points:
- **Time Complexity**: The solution works in O(n) time, where n is the length of the `nums` array, because each element is processed at most twice (once by `right` and once by `left`).
- **Space Complexity**: The space complexity is O(1) since we are using a fixed amount of extra space regardless of the input size. 

This code adheres to the constraints provided and ensures that we efficiently count all valid subarrays meeting the product condition.

# 723. Candy Crush

### Problem Description 
This question is about implementing a basic elimination algorithm for Candy Crush.

Given a 2D integer array `board` representing the grid of candy, different positive integers `board[i][j]` represent different types of candies. A value of `board[i][j] = 0` represents that the cell at position `(i, j)` is empty. The given board represents the state of the game following the player's move. Now, you need to restore the board to a stable state by crushing candies according to the following rules:
If three or more candies of the same type are adjacent vertically or horizontally, "crush" them all at the same time - these positions become empty.

After crushing all candies simultaneously, if an empty space on the board has candies on top of itself, then these candies will drop until they hit a candy or bottom at the same time. (No new candies will drop outside the top boundary.)
After the above steps, there may exist more candies that can be crushed. If so, you need to repeat the above steps.

If there does not exist more candies that can be crushed (ie. the board is stable), then return the current board.

You need to perform the above rules until the board becomes stable, then return the current board.


Example:
Input:
board = 
[[110,5,112,113,114],[210,211,5,213,214],[310,311,3,313,314],[410,411,412,5,414],[5,1,512,3,3],[610,4,1,613,614],[710,1,2,713,714],[810,1,2,1,1],[1,1,2,2,2],[4,1,4,4,1014]]
Output:
[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[110,0,0,0,114],[210,0,0,0,214],[310,0,0,113,314],[410,0,0,213,414],[610,211,112,313,614],[710,311,412,613,714],[810,411,512,713,1014]]
Explanation: 
Note:
The length of `board` will be in the range [3, 50].

The length of `board[i]` will be in the range [3, 50].

Each `board[i][j]` will initially start as an integer in the range [1, 2000].

### Solution 
 To solve the Candy Crush problem as described, we need to follow these steps:
1. **Identify and Crush Candies:** Check for groups of three or more candies of the same type that are adjacent either horizontally or vertically. Mark these candies for crushing by setting their positions to zero.
2. **Drop Remaining Candies:** After crushing, we need to let the candies above the empty spaces drop down. This process continues until no more candies can be crushed.
3. **Repeat Steps 1 and 2:** Continue to crush and drop candies until the board becomes stable, meaning no more candies can be crushed.

Here's the Python code to implement this:



```python
class Solution:
    def candyCrush(self, board: List[List[int]]) -> List[List[int]]:
        # Function to mark candies to crush
        def markCrush(board):
            to_crush = set()
            rows, cols = len(board), len(board[0])
            # Check horizontally and vertically
            for r in range(rows):
                for c in range(cols - 2):  # check horizontal
                    if board[r][c] == board[r][c + 1] == board[r][c + 2] != 0:
                        to_crush.update({(r, c), (r, c + 1), (r, c + 2)})
            
            for r in range(rows - 2):  # check vertical
                for c in range(cols):
                    if board[r][c] == board[r + 1][c] == board[r + 2][c] != 0:
                        to_crush.update({(r, c), (r + 1, c), (r + 2, c)})
            # Crush candies
            for r, c in to_crush:
                board[r][c] = 0
            return len(to_crush) > 0  # Return True if candies were crushed

        # Function to drop candies
        def dropCandies(board):
            rows, cols = len(board), len(board[0])
            for c in range(cols):
                # Start from the bottom empty spot
                insert_row = rows - 1
                for r in range(rows - 1, -1, -1):  # Traverse from bottom to top
                    if board[r][c] != 0:
                        board[insert_row][c] = board[r][c]
                        insert_row -= 1
                # Fill remaining cells above with 0s
                for r in range(insert_row, -1, -1):
                    board[r][c] = 0

        while True:
            if not markCrush(board):
                break  # No more candies to crush
            dropCandies(board)

        return board

```

### Explanation:
1. **Class Definition:** We define a class named `Solution` which contains the method `candyCrush`.
2. **Mark Candies to Crush:** The function `markCrush` checks each row and column for groups of three or more matching candies. If found, their positions are added to the `to_crush` set. After identifying which candies to crush, they're set to zero in the board.
3. **Drop Candies:** The function `dropCandies` iterates through each column and shifts non-zero candies downwards after all candies have been crushed, filling any empty cells above with zeros.
4. **Loop Until Board is Stable:** We repeatedly call the `markCrush` and `dropCandies` methods until there are no more candies to crush. The board is considered stable when no candies are crushed in a full pass through `markCrush`.
5. **Return the Final Board:** Once the board is stable, it's returned.

### Efficiency:
- The algorithm operates in O(n * m) time complexity, where n is the number of rows and m is the number of columns. This is due to the need to scan each cell multiple times until the board stabilizes. The space complexity is O(1) if we consider the board itself, as we modify it in place.

This code follows the structure required for submission on platforms like LeetCode, enabling it to run directly when adhered to their input/output standards.

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
 To solve this problem, we need to partition a given string `S` such that each letter appears in at most one part. The goal is to maximize the number of such partitions. 

Here's a step-by-step approach to solve the problem:

1. **Track the Last Occurrence of Each Character**: We will first record the last index for each character in the string. This will allow us to determine the split points for the partitions.

2. **Iterate through the String**: We will maintain a pointer or index to track the end of the current partition. As we iterate through the string, we will keep updating the end index to be the maximum of the current end index and the last occurrence of the character we are processing.

3. **Form Partitions**: Once the current index matches the end index of the partition, we can record the size of this partition and reset the start index for the next partition.

4. **Return the Result**: Finally, we will return the list of partition sizes.

Here is the Python code implementing this solution:



```python
class Solution:
    def partitionLabels(self, S: str) -> List[int]:
        # Step 1: Record the last occurrence of each character
        last_occurrence = {char: i for i, char in enumerate(S)}
        
        # Step 2: Initialize variables for current partition
        partitions = []
        current_end = 0
        start = 0
        
        # Step 3: Iterate through the characters in S
        for i, char in enumerate(S):
            # Update the current end for the partition
            current_end = max(current_end, last_occurrence[char])
            
            # If we have reached the end of the current partition
            if i == current_end:
                # Append the size of the partition
                partitions.append(current_end - start + 1)
                # Move the start to the next position
                start = i + 1
        
        # Step 4: Return the result
        return partitions

```

### Explanation of the Code:

1. **Last Occurrence Dictionary**:
   - We create a dictionary `last_occurrence` that maps each character in `S` to its last index using a dictionary comprehension.

2. **Partition Management**:
   - We initialize an empty list `partitions` to store the size of each partition.
   - We maintain the `current_end` which keeps track of the furthest index we can reach considering all characters included in the current partition.
   - `start` is initialized to track the beginning of the current partition.

3. **Iterating Through the String**:
   - For each character in the string, we update `current_end` to be the maximum of the current `current_end` and the last occurrence of the character.
   - When we reach `current_end`, it indicates that we can finalize the current partition. We calculate the size of the partition as `current_end - start + 1` and append it to `partitions`.

4. **Return Result**:
   - Finally, we return the list of partition sizes.

### Example Walkthrough:
For the input `"ababcbacadefegdehijhklij"`:

- The last occurrences will be `{ 'a': 8, 'b': 5, 'c': 7, 'd': 11, 'e': 12, 'f': 13, 'g': 14, 'h': 15, 'i': 22, 'j': 23 }`
- As we iterate, we find the partitions:
  - The first partition ends at index 8 ("ababcbaca"), size 9.
  - The second partition ends at index 15 ("defegde"), size 7.
  - The third partition ends at index 23 ("hijhklij"), size 8.

The output is `[9, 7, 8]` which represents the sizes of the partitions.

This solution effectively partitions the string while ensuring that each character appears only in one partition, maximizing the number of partitions created.

# 826. Most Profit Assigning Work

### Problem Description 
We have jobs: `difficulty[i]` is the difficulty of the `i`th job, and `profit[i]` is the profit of the `i`th job. 
Now we have some workers. `worker[i]` is the ability of the `i`th worker, which means that this worker can only complete a job with difficulty at most `worker[i]`. 
Every worker can be assigned at most one job, but one job can be completed multiple times.

For example, if 3 people attempt the same job that pays $1, then the total profit will be $3.  If a worker cannot complete any job, his profit is $0.

What is the most profit we can make?

Example 1:
Input: difficulty = [2,4,6,8,10], profit = [10,20,30,40,50], worker = [4,5,6,7]
Output: 100 
Explanation: Workers are assigned jobs of difficulty [4,4,6,6] and they get profit of [20,20,30,30] seperately.

Notes:
`1 <= difficulty.length = profit.length <= 10000`
`1 <= worker.length <= 10000`
`difficulty[i], profit[i], worker[i]`  are in range `[1, 10^5]`

### Solution 
 To solve the problem of maximizing the total profit given the job difficulties, their corresponding profits, and the abilities of workers, we can follow a systematic approach. Below is a detailed explanation of the solution, structured to fit the LeetCode format.

### Problem Breakdown
1. **Understanding Inputs**:
   - We have two lists: `difficulty` and `profit` where each index corresponds to a specific job.
   - `worker` is a list of workers, each with a certain ability to handle jobs.

2. **Constraints**:
   - A worker can only complete jobs that are less than or equal to their ability.
   - A worker can only take one job, but a job can be repeated by different workers.

3. **Goal**:
   - Maximize the profit by assigning jobs to workers based on their capabilities.

### Approach
1. **Pairing Jobs with Their Profits**:
   - First, we will pair the difficulty of each job with its profit and sort these pairs based on the difficulty of the job. This makes it easier to determine which jobs are available to each worker.

2. **Calculating Maximum Profit for Each Worker**:
   - We will keep track of the maximum profit that is available for jobs up to a certain difficulty while iterating through the jobs.

3. **Assigning Workers**:
   - For each worker, determine the maximum profit they can gain based on the jobs they can complete (those with difficulty ≤ worker's ability).

### Implementation
Here is the Python code that implements this logic:



```python
class Solution:
    def maxProfitAssignment(self, difficulty: List[int], profit: List[int], worker: List[int]) -> int:
        # Pairing the difficulty and profit, and sorting them based on difficulty
        jobs = sorted(zip(difficulty, profit))
        
        # This will store the max profit for jobs with up to the current difficulty.
        max_profit = 0
        max_profit_up_to = []
        
        # Fill max_profit_up_to with the maximum profit obtainable at or below each job difficulty
        for d, p in jobs:
            max_profit = max(max_profit, p)
            max_profit_up_to.append((d, max_profit))
        
        # Sorting workers for easier matching
        worker.sort()
        
        total_profit = 0
        job_index = 0
        
        for w in worker:
            # While there are jobs available for this worker's ability
            while job_index < len(max_profit_up_to) and max_profit_up_to[job_index][0] <= w:
                job_index += 1
            
            # If there are jobs that this worker can perform, add the max profit of those jobs
            if job_index > 0:
                total_profit += max_profit_up_to[job_index - 1][1]
        
        return total_profit

```

### Explanation of the Code
1. **Importing Necessary Libraries**: 
   We will use `List` from `typing` to type our input parameters.

2. **Creating the `jobs` List**:
   - We create pairs of difficulties and profits, which we then sort based on job difficulty.

3. **Maintaining Maximum Profit**:
   - As we iterate through sorted jobs, we maintain a running `max_profit` which records the highest profit achievable for jobs up to each difficulty.

4. **Sorting Workers**:
   - We sort the `worker` array to ensure we evaluate workers in the order of their abilities.

5. **Calculating Total Profit**:
   - For each worker, we find the maximum profit they can achieve by checking the pre-computed `max_profit_up_to` list, and if they can handle jobs, we accumulate the corresponding max profits.

### Complexity Analysis
- The time complexity of this solution is O(n log n) due to sorting the difficulties and profits, and O(m log m) for sorting the workers (where n is the number of jobs and m is the number of workers). The overall complexity is dominated by sorting, leading to a time complexity of O(n log n + m log m).
- The space complexity is O(n) due to the additional list for max profits.

With this structured approach, we efficiently calculate the maximum profit that can be attained given the constraints of job difficulties and worker abilities.

# 828. Count Unique Characters of All Substrings of a Given String

### Problem Description 
Let's define a function `countUniqueChars(s)` that returns the number of unique characters on `s`, for example if `s = "LEETCODE"` then `"L"`, `"T"`,`"C"`,`"O"`,`"D"` are the unique characters since they appear only once in `s`, therefore `countUniqueChars(s) = 5`.

On this problem given a string `s` we need to return the sum of `countUniqueChars(t)` where `t` is a substring of `s`. Notice that some substrings can be repeated so on this case you have to count the repeated ones too.

Since the answer can be very large, return the answer modulo `10 ^ 9 + 7`.


Example 1:
Input: s = "ABC"
Output: 10
Explanation: All possible substrings are: "A","B","C","AB","BC" and "ABC".

Evey substring is composed with only unique letters.

Sum of lengths of all substring is 1 + 1 + 1 + 2 + 2 + 3 = 10

Example 2:
Input: s = "ABA"
Output: 8
Explanation: The same as example 1, except `countUniqueChars`("ABA") = 1.


Example 3:
Input: s = "LEETCODE"
Output: 92

Constraints:
`0 <= s.length <= 10^4`
`s` contain upper-case English letters only.

### Solution 
 To solve the problem of counting the number of unique characters in all substrings of a given string `s`, we need to approach it methodically. The idea is to iterate through all possible substrings, but since the brute force approach would lead to a time complexity that is too high for the constraints given, we will use an optimized method.

Here's the plan:

1. **Understand Uniqueness in Substrings**: For a substring to contribute to the count of unique characters, each character in that substring must appear only once.
  
2. **Use a Sliding Window Technique**: As we traverse the string, we'll keep track of characters that have appeared and their counts using a dictionary. With this, we can determine when to count them as unique.

3. **Count Contributions of Each Character**: For each character, determine how many substrings can be formed that include this character as a unique character.

4. **Mathematical Formula**: For a character at index `i`:
   - If the last occurrence of this character was at index `j`, then:
     - It contributes to all substrings starting from `j + 1` to `i` (both indices inclusive) and ending at `i`.
     - This number can be computed as:
       - Count of starting positions = `i - j`
       - Count of ending positions = `len(s) - i`
       - Total substrings with this character as unique = `(i - j) * (len(s) - i)`

5. **Modulo Operation**: Since the results can be very large, we will return the result modulo \(10^9 + 7\).

Now, here's the implementation based on this approach:



```python
class Solution:
    def uniqueLetterString(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        last = {}
        unique_count_sum = 0
        
        for i in range(n):
            char = s[i]
            if char in last:
                prev_index = last[char]
            else:
                prev_index = -1
            
            # Contribution of the current character as a unique character
            left_options = (i - prev_index)
            right_options = (n - i)
            contribution = left_options * right_options
            
            unique_count_sum += contribution
            unique_count_sum %= MOD
            
            # Update the last index of the current character
            last[char] = i
        
        return unique_count_sum

# Example of using the Solution class
# sol = Solution()
# print(sol.uniqueLetterString("ABC"))      # Output: 10
# print(sol.uniqueLetterString("ABA"))      # Output: 8
# print(sol.uniqueLetterString("LEETCODE")) # Output: 92

```

### Explanation of the Code:
- We define a class `Solution` with a method `uniqueLetterString`.
- We initialize `MOD` for modulus operations, and `last` to keep track of the last indices of characters.
- We iterate through the string, updating our `last` dictionary and counting the contributions of unique characters.
- Finally, we return the accumulated sum of contributions modulo \(10^9 + 7\).

### Complexity Analysis:
- **Time Complexity**: O(n), as we are traversing the string once.
- **Space Complexity**: O(1) considering the size of the `last` dictionary can hold at most 26 entries (for each uppercase English letter). 

This implementation is efficient and adheres to the constraints provided in the problem statement.

# 838. Push Dominoes

### Problem Description 
There are `N` dominoes in a line, and we place each domino vertically upright.

In the beginning, we simultaneously push some of the dominoes either to the left or to the right.

After each second, each domino that is falling to the left pushes the adjacent domino on the left.

Similarly, the dominoes falling to the right push their adjacent dominoes standing on the right.

When a vertical domino has dominoes falling on it from both sides, it stays still due to the balance of the forces.

For the purposes of this question, we will consider that a falling domino expends no additional force to a falling or already fallen domino.

Given a string "S" representing the initial state. `S[i] = 'L'`, if the i-th domino has been pushed to the left; `S[i] = 'R'`, if the i-th domino has been pushed to the right; `S[i] = '.'`, if the `i`-th domino has not been pushed.

Return a string representing the final state. 

Example 1:
Input: ".L.R...LR..L.."
Output: "LL.RR.LLRRLL.."

Example 2:
Input: "RR.L"
Output: "RR.L"
Explanation: The first domino expends no additional force on the second domino.

Note:
`0 <= N <= 10^5`
String `dominoes` contains only `'L`', `'R'` and `'.'`

### Solution 
 To solve the problem of determining the final state of the dominoes after they have been pushed, we can simulate the process in an efficient way. The key points to consider are:

1. The dominoes can only affect their immediate neighbors.
2. A falling domino ('L' or 'R') will influence the next domino immediately next to it unless balanced by an opposing influence.

### Approach:

To handle the dominoes' movements, we can utilize a two-pass method:
1. **Left to Right:** Calculate how far the influence of 'R' spreads in the forward direction.
2. **Right to Left:** Calculate how far the influence of 'L' spreads in the backward direction.

### Steps:

1. Initialize an array (or list) to hold the results.
2. As we traverse the string from left to right:
   - Count how many 'R's are in sequence until we hit an 'L' or the end.
   - If we encounter 'L', conclude that the section is settled (current position's influence is determined by nearest influences).
3. Traverse the string again from right to left:
   - Count how many 'L's are in sequence until we hit an 'R' or the start.
   - If we encounter 'R', conclude that the section is settled.
4. For each position, determine the final state based on the influences extracted from both passes.

### Code Implementation:

We will follow the LeetCode solution format, making sure the solution is encapsulated within a class and method.



```python
class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        n = len(dominoes)
        if n == 0:
            return ""
        
        # Create an array to track the forces on each domino
        forces = [0] * n
        
        # Left to right pass for 'R'
        force = 0
        for i in range(n):
            if dominoes[i] == 'R':
                force = n  # A very high force
            elif dominoes[i] == 'L':
                force = 0  # Reset when we hit 'L'
            else:
                force = max(force - 1, 0)  # Reduce influence by 1
            forces[i] += force
        
        # Right to left pass for 'L'
        force = 0
        for i in range(n - 1, -1, -1):
            if dominoes[i] == 'L':
                force = n  # A very high force for 'L'
            elif dominoes[i] == 'R':
                force = 0  # Reset when we hit 'R'
            else:
                force = max(force - 1, 0)  # Reduce influence by 1
            forces[i] -= force
        
        # Build the final result based on the forces array
        result = []
        for f in forces:
            if f > 0:
                result.append('R')
            elif f < 0:
                result.append('L')
            else:
                result.append('.')
        
        return ''.join(result)

# Example usage:
solution = Solution()
print(solution.pushDominoes(".L.R...LR..L.."))  # Output: "LL.RR.LLRRLL.."

```

### Explanation of the Code:

1. **Initialization:** We begin by initializing a `forces` list to record the effective influence of 'R' and 'L'.
2. **First Pass (Left to Right):** We traverse the dominoes from left to right maintaining a `force` variable that increments when we see 'R' and resets when we see 'L'. This spreads the rightward influence.
3. **Second Pass (Right to Left):** We repeat a similar process from right to left, but this time we decrease the `force` when we see 'R'.
4. **Final Result Construction:** After both passes, we prepare the final output based on the accumulated forces: if the force is positive, the result is 'R', if negative, it's 'L', and if zero, it remains '.'.

This method is efficient with a time complexity of O(N), where N is the length of the input string, and it won't exceed the memory complexity of O(N). It adheres to the constraints provided in the problem.

# 844. Backspace String Compare

### Problem Description 
Given two strings `s` and `t`, return `true` if they are equal when both are typed into empty text editors. `'#'` means a backspace character.

Note that after backspacing an empty text, the text will continue empty.


Example 1:
Input: s = "ab#c", t = "ad#c"
Output: true
Explanation: Both s and t become "ac".


Example 2:
Input: s = "ab##", t = "c#d#"
Output: true
Explanation: Both s and t become "".


Example 3:
Input: s = "a##c", t = "#a#c"
Output: true
Explanation: Both s and t become "c".


Example 4:
Input: s = "a#c", t = "b"
Output: false
Explanation: s becomes "c" while t becomes "b".


Constraints:
`1 <= s.length, t.length <= 200`
`s` and `t` only contain lowercase letters and `'#'` characters.

Follow up: Can you solve it in `O(n)` time and `O(1)` space?

### Solution 
 To solve the problem of determining if two strings `s` and `t` are equal when both are processed with backspaces represented by the `'#'` character, we can use a stack-based approach. However, since the follow-up requests an `O(n)` time complexity and `O(1)` space complexity, we will avoid using additional data structures like stacks for storing characters. Instead, we will simulate the backspace directly by traversing the strings.

## Approach

1. **Two-Pointer Technique**: We'll set up two pointers to traverse both strings from the end to the beginning. This allows us to handle backspaces effectively without needing extra space.
2. **Skip Backspaces**: For each string, we will skip characters that are 'valid' after processing the backspaces until we find a character that isn't a backspace or we run out of characters.
3. **Compare Characters**: Once we have the valid characters for both strings, we will compare them until one of the strings is exhausted.
4. **Return Result**: If both pointers reach the end of their respective strings simultaneously, it means both strings represent the same sequence of characters after processing backspaces.

Here's how this can be coded in Python:



```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        def getNextValidCharIndex(s: str, index: int) -> int:
            skip = 0
            while index >= 0:
                if s[index] == '#':
                    skip += 1
                elif skip > 0:
                    skip -= 1
                else:
                    break
                index -= 1
            return index
        
        indexS = len(s) - 1
        indexT = len(t) - 1
        
        while indexS >= 0 or indexT >= 0:
            indexS = getNextValidCharIndex(s, indexS)
            indexT = getNextValidCharIndex(t, indexT)
            
            # If both indices are out of bounds, strings are equal
            if indexS < 0 and indexT < 0:
                return True
            # If one index is out of bounds while the other is not, strings are not equal
            if indexS < 0 or indexT < 0:
                return False
            # Compare characters at current indices
            if s[indexS] != t[indexT]:
                return False
            
            # Move to the previous character
            indexS -= 1
            indexT -= 1
        
        return True

```

### Explanation of the Code:
- **getNextValidCharIndex Function**: This helper function moves backwards through a string `s` starting from `index`, skipping characters as dictated by backspaces (`#`). It returns the next valid character's index.
- We start with pointers `indexS` and `indexT` at the end of `s` and `t`, respectively.
- In a loop, we call `getNextValidCharIndex` for both strings to get their next valid character indices after considering possible backspaces.
- If both indices become negative, it means both strings are processed fully and equal, so we return `True`.
- If only one of them becomes negative, they’re not equal, returning `False`.
- Finally, we compare the characters at the current valid indices. If they are not the same, return `False`.
- If all checks pass, we return `True`.

This solution operates in linear time `O(n)` as both strings are effectively traversed once and uses constant space `O(1)` since we don’t use any auxiliary data structures.

# 845. Longest Mountain in Array

### Problem Description 
You may recall that an array `arr` is a mountain array if and only if:
`arr.length >= 3`
There exists some index `i` (0-indexed) with `0 < i < arr.length - 1` such that:
	
`arr[0] < arr[1] < ... < arr[i - 1] < arr[i]`
`arr[i] > arr[i + 1] > ... > arr[arr.length - 1]`
Given an integer array `arr`, return the length of the longest subarray, which is a mountain. Return `0` if there is no mountain subarray.


Example 1:
Input: arr = [2,1,4,7,3,2,5]
Output: 5
Explanation: The largest mountain is [1,4,7,3,2] which has length 5.


Example 2:
Input: arr = [2,2,2]
Output: 0
Explanation: There is no mountain.


Constraints:
`1 <= arr.length <= 104`
`0 <= arr[i] <= 104`
Follow up:
Can you solve it using only one pass?
Can you solve it in `O(1)` space?

### Solution 
 To solve the problem of finding the length of the longest mountain subarray, we need to iterate through the given array and determine the subarrays that satisfy the mountain condition. A mountain is defined as an array segment where there is an increasing sequence followed by a decreasing sequence.

### Approach
1. **Identify the mountain peak**: We need to find potential peaks of mountains. A peak is an element that is greater than its left and right neighbors.
2. **Expand from the peak**: Once a peak is identified, we can extend left and right to determine the full length of the mountain.
3. **Track the maximum length**: Throughout this process, we can keep track of the maximum length of any mountain found.

### Implementation Steps
1. Loop through the array, skipping the first and last elements since they cannot be peaks.
2. For each element, check if it is a peak.
3. If it is a peak, expand left and right to count how many elements can be included as part of the mountain.
4. Update the maximum length if the current mountain is longer than the previously found maximum.
5. Return the maximum length found, or `0` if no mountains are present.

### Complexity
- **Time Complexity**: O(n) because we only traverse the array a couple of times.
- **Space Complexity**: O(1) since we only use a few variables for tracking lengths.

Here’s the Python code that implements this approach, structured to fit the LeetCode submission format:



```python
class Solution:
    def longestMountain(self, arr: List[int]) -> int:
        n = len(arr)
        if n < 3:
            return 0
        
        max_length = 0
        
        for i in range(1, n - 1):
            # Check if arr[i] is a peak
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                # Expand left from the peak
                left = i - 1
                while left > 0 and arr[left] > arr[left - 1]:
                    left -= 1
                
                # Expand right from the peak
                right = i + 1
                while right < n - 1 and arr[right] > arr[right + 1]:
                    right += 1
                
                # Calculate current mountain length
                current_mountain_length = right - left + 1
                max_length = max(max_length, current_mountain_length)
        
        return max_length

```

### Explanation of the Code:
- We define a class `Solution` with a method `longestMountain` that takes a list of integers `arr`.
- `n` is the length of the array. If `n` is less than 3, we immediately return 0 because a mountain requires at least three elements.
- We initialize `max_length` to track the maximum length of all mountain subarrays found.
- We iterate from the first element to the second-to-last element to check for potential peaks.
- If a peak is identified, we expand left and right, counting how many elements can be included in that mountain.
- We calculate the current mountain's length as `right - left + 1` and update `max_length` accordingly.
- Finally, we return the longest mountain length found.

### Testing:
You can test this solution on the LeetCode platform with the provided examples or any other test cases you wish to validate.

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
 To solve the problem of minimizing the number of boats required to carry people with given weights, we can follow a greedy algorithm. The main idea is to sort the array of weights and use a two-pointer approach to pair the lightest and heaviest people together whenever possible, thereby maximizing the use of each boat.

Here's a step-by-step explanation of the approach we'll take:

1. **Sort the Input Array**: This helps us easily pair the heaviest and lightest people. By sorting the weights, we can then efficiently check pairs from both ends.

2. **Initialize Two Pointers**: One pointer (`left`) starts at the beginning (lightest person), and the other pointer (`right`) starts at the end (heaviest person) of the sorted array.

3. **Boat Counting**: We will check if the sum of the weights pointed to by the two pointers `people[left] + people[right]` is less than or equal to the `limit`. If it is, we can take both people in a single boat, and we move both pointers inward (increment `left` and decrement `right`). If they cannot be paired (their combined weight exceeds the `limit`), we take the heavier person alone (decrement `right` only).

4. **Increment Boat Count**: Each time we can or cannot pair people, we increment the boat count.

5. **Termination**: The process continues until all people are accounted for (i.e., until the `left` pointer surpasses the `right` pointer).

Now, let's implement this in Python, following the LeetCode solution format.



```python
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        # Step 1: Sort the list of people's weights
        people.sort()
        
        left = 0  # Pointer for the lightest person
        right = len(people) - 1  # Pointer for the heaviest person
        boats_count = 0  # Initialize number of boats
        
        # Step 2: Use two pointers to pair people
        while left <= right:
            # Check if the lightest and heaviest person can share a boat
            if people[left] + people[right] <= limit:
                # They can share a boat
                left += 1  # Move the left pointer to the right
            # Always move the right pointer (take the heaviest person in a boat)
            right -= 1  
            # We have used one boat at each iteration
            boats_count += 1
        
        return boats_count  # Return the total number of boats used

```

### Explanation of the Code:
- **Sorting**: The array is sorted to allow efficient pairing of light and heavy people.
- **Two-pointer Technique**: By using two pointers, we can efficiently check whether to pair two people or take one alone. This optimizes the number of boats used.
- **Boat Counting Logic**: Every iteration through the loop checks if we can combine people into a single boat or if we need to use a separate boat for the heavier person. The `boats_count` variable keeps track of how many boats we've used.
- **Edge Cases**: The algorithm handles all cases efficiently, including when there's only one person.

### Complexity:
- **Time Complexity**: O(n log n) due to the sorting step, where `n` is the number of people.
- **Space Complexity**: O(1) if we consider the sorting in-place, otherwise O(n) if using space for sorted copies. In this case, we're modifying the list in-place.

This solution efficiently handles the problem constraints and should run well within limits on LeetCode.

# 904. Fruit Into Baskets

### Problem Description 
In a row of trees, the `i`-th tree produces fruit with type `tree[i]`.

You start at any tree of your choice, then repeatedly perform the following steps:
Add one piece of fruit from this tree to your baskets.  If you cannot, stop.

Move to the next tree to the right of the current tree.  If there is no tree to the right, stop.

Note that you do not have any choice after the initial choice of starting tree: you must perform step 1, then step 2, then back to step 1, then step 2, and so on until you stop.

You have two baskets, and each basket can carry any quantity of fruit, but you want each basket to only carry one type of fruit each.

What is the total amount of fruit you can collect with this procedure?

Example 1:
Input: [1,2,1]
Output: 3
Explanation: We can collect [1,2,1].


Example 2:
Input: [0,1,2,2]
Output: 3
Explanation: We can collect [1,2,2].

If we started at the first tree, we would only collect [0, 1].


Example 3:
Input: [1,2,3,2,2]
Output: 4
Explanation: We can collect [2,3,2,2].

If we started at the first tree, we would only collect [1, 2].


Example 4:
Input: [3,3,3,1,2,1,1,2,3,3,4]
Output: 5
Explanation: We can collect [1,2,1,1,2].

If we started at the first tree or the eighth tree, we would only collect 4 fruits.

Note:
`1 <= tree.length <= 40000`
`0 <= tree[i] < tree.length`

### Solution 
 Certainly! The problem can be solved using the sliding window technique. This method allows us to maintain a window of trees that we can traverse and count the maximum fruit we can collect with the constraint of only having two types of fruit in our baskets.

Here is how we will solve the problem step-by-step:

### Explanation:

1. **Use a Sliding Window**:
   - We maintain two pointers: `left` and `right`, which will represent the current window of trees we're considering.
   - We'll utilize the `right` pointer to expand the window and the `left` pointer to contract it when we encounter more than two types of fruit.

2. **Data Structures**:
   - We'll use a dictionary (or a hashmap) to keep track of how many of each type of fruit are in the current window.

3. **Algorithm**:
   - Start iterating with the `right` pointer from the beginning of the list of trees.
   - For each tree, add the fruit type to our hashmap and update the count.
   - If we find that there are more than two different types of fruit in our hashmap, we need to move the `left` pointer until we are back to two types.
   - Throughout this process, keep track of the maximum length of the window that contains two fruit types.

4. **Edge Cases**:
   - The input can have only one tree; in this case, the maximum is 1.
   - If all trees are of the same type, the length of the list is the answer.

### Implementation:

Here's the Python code which implements the above logic and follows the LeetCode format:



```python
class Solution:
    def totalFruits(self, tree: List[int]) -> int:
        fruit_count = {}
        left = 0
        max_fruits = 0
        
        for right in range(len(tree)):
            # Add the current tree's fruit to the basket
            if tree[right] in fruit_count:
                fruit_count[tree[right]] += 1
            else:
                fruit_count[tree[right]] = 1
            
            # While we have more than 2 types of fruit, move the left pointer
            while len(fruit_count) > 2:
                fruit_count[tree[left]] -= 1
                if fruit_count[tree[left]] == 0:
                    del fruit_count[tree[left]]
                left += 1
            
            # Update the maximum fruits we can collect
            max_fruits = max(max_fruits, right - left + 1)
        
        return max_fruits

```

### How to Use:
1. Define a list of integers representing the trees.
2. Create an instance of the `Solution` class.
3. Call the `totalFruits` method with the tree list as an argument, and it will return the maximum number of fruits you can collect.

### Example Usage:


```python
solution = Solution()
print(solution.totalFruits([1,2,1]))  # Output: 3
print(solution.totalFruits([0,1,2,2]))  # Output: 3
print(solution.totalFruits([1,2,3,2,2]))  # Output: 4
print(solution.totalFruits([3,3,3,1,2,1,1,2,3,3,4]))  # Output: 5

```

This code will perform efficiently on the input size constraints mentioned in the problem, guaranteeing a solution in linear time complexity \( O(n) \) where \( n \) is the length of the input list `tree`.

