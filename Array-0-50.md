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
 To solve the problem of finding two numbers in an array that sum up to a given target, a highly efficient method involves using a hash table. This approach reduces the time complexity compared to a naive solution using two nested loops. The hash table (or dictionary in Python) stores each number as you iterate through the array, along with its index. For each number, you check whether the complement (target minus the current number) exists in the hash table. If it exists, you've found the pair of indices that sum to the target.

Here's the code following the LeetCode solution format:



```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # Dictionary to hold number as key and index as value
        lookup = {}
        
        # Iterate through list of numbers
        for i, num in enumerate(nums):
            # Compute complement
            complement = target - num
            # If complement exists in the dictionary, return the indices
            if complement in lookup:
                return [lookup[complement], i]
            # Store the index of the current number
            lookup[num] = i
        
        # Return empty list if no solution found (though the problem states one solution always exists)
        return []

# Below calls and inputs-to/outputs-from this class would typically be handled by 
# LeetCode's platform directly in their environment and aren't meant to be edited directly.
# If running locally, you can manually test the class like this:

# sol = Solution()
# print(sol.twoSum([2, 7, 11, 15], 9))  # Output: [0,1]
# print(sol.twoSum([3, 2, 4], 6))       # Output: [1,2]
# print(sol.twoSum([3, 3], 6))          # Output: [0,1]

```

### Explanation

1. **Dictionary (Hash table) Setup**: We set up a dictionary `lookup` to remember the numbers we've seen so far and their indices.
2. **Iteration and Checking**: As we iterate through the list, we calculate what complement would sum with the current number to reach the `target`. We check if this complement exists in our dictionary:
   - If it does, we have found our two numbers, and we immediately return their indices - first the one stored in the dictionary, then the current index.
   - If not, we store the current number and its index in the dictionary to reference for later iterations.
3. **Early Exit**: The moment we find a valid pair, we return their indices. If by design all inputs have exactly one solution, then we are guaranteed to exit within the loop.

This solution efficiently solves the problem with a time complexity of \(O(n)\), where \(n\) is the number of elements in `nums`, since we are only iterating through the list once. The space complexity is also \(O(n)\) due to the additional dictionary used to store up to \(n\) elements in the worst case. This is much faster than a \(O(n^2)\) approach that uses nested loops to find the two sum indices.

# 4. Median of Two Sorted Arrays

### Problem Description 
Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return the median of the two sorted arrays.


Example 1:
Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.


Example 2:
Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.


Example 3:
Input: nums1 = [0,0], nums2 = [0,0]
Output: 0.00000

Example 4:
Input: nums1 = [], nums2 = [1]
Output: 1.00000

Example 5:
Input: nums1 = [2], nums2 = []
Output: 2.00000

Constraints:
`nums1.length == m`
`nums2.length == n`
`0 <= m <= 1000`
`0 <= n <= 1000`
`1 <= m + n <= 2000`
`-106 <= nums1[i], nums2[i] <= 106`
Follow up: The overall run time complexity should be `O(log (m+n))`.

### Solution 
 To solve the problem of finding the median of two sorted arrays efficiently, we can utilize a binary search approach to achieve a time complexity of \( O(\log(\min(m, n))) \). Here's how we can approach this:

1. **Identify the shorter array**: Always perform binary search on the shorter array to keep time complexity minimal.

2. **Binary Search Setup**: We will use binary search to partition the two arrays such that the elements on the left half of the combined arrays are all less than the elements on the right half.

3. **Partitioning Logic**:
   - Let's denote `partitionX` and `partitionY` as the partition indices of `nums1` and `nums2` respectively.
   - The goal is to ensure that `nums1[partitionX-1] <= nums2[partitionY]` and `nums2[partitionY-1] <= nums1[partitionX]`.

4. **Calculate partitions**: Based on the current index in array `nums1` (obtained via binary search), calculate the corresponding index in `nums2` such that the left half of the combined arrays have the same number of elements as the right half (or one more if the combined length is odd).

5. **Adjust binary search bounds**: Use the conditions from step 3 to adjust the high and low bounds of the binary search.

6. **Calculate Median**:
   - If the total number of elements is odd, the median is the maximum element on the left side of the partition.
   - If even, the median is the average of the maximum element on the left side and the minimum element on the right side of the partition.

The following solution implements this approach and should be directly runnable on LeetCode's platform.



```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        
        x, y = len(nums1), len(nums2)
        low, high = 0, x
        
        while low <= high:
            partitionX = (low + high) // 2
            partitionY = (x + y + 1) // 2 - partitionX
            
            maxLeftX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]
            minRightX = float('inf') if partitionX == x else nums1[partitionX]
            
            maxLeftY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]
            minRightY = float('inf') if partitionY == y else nums2[partitionY]
            
            if maxLeftX <= minRightY and maxLeftY <= minRightX:
                if (x + y) % 2 == 0:
                    return (max(maxLeftX, maxLeftY) + min(minRightX, minRightY)) / 2.0
                else:
                    return max(maxLeftX, maxLeftY)
            elif maxLeftX > minRightY:
                high = partitionX - 1
            else:
                low = partitionX + 1
        
        raise ValueError("Input arrays are not sorted or conditions are not met")

```

### Explanation:
- The script initially swaps `nums1` and `nums2` if necessary to ensure `nums1` is the smaller array for efficient binary search.
- It initializes pointers (`low` and `high`) to perform a binary search on the smaller array (`nums1`).
- Inside the loop, it computes the partition indices for both arrays to split them effectively into left and right halves.
- It then checks if the current partitioning is valid and if so, computes the median based on whether the total length of the two arrays is even or odd.
- Adjustments to the `low` and `high` values are made based on the values on either side of the partition.
- If the two arrays are not partitionable as expected after all iterations, it raises a `ValueError`.
  
This solution ensures an optimal \( O(\log(\min(m, n))) \) complexity, matching the problem's follow-up constraint.

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
 The problem described is known as the "Container With Most Water" problem. It asks you to find the maximum amount of water that can be trapped by any two lines, when these lines and the x-axis form a container.

### Strategy:
To solve this problem, we can employ a two-pointer technique:
1. **Initialization**: Start with two pointers, one pointing at the beginning (`left`) of the array and one pointing at the end (`right`).
2. **Calculating Area**: For the pointers at `left` and `right`, compute the area using the formula:
\[ \text{area} = (\text{right} - \text{left}) \times \min(\text{height[left]}, \text{height[right]}) \]
3. **Maximization**: Update the maximum area found so far.
4. **Pointer Movement**:
   - Move the pointer that points to the shorter line towards the other pointer. The rationale behind this is that perhaps a taller line exists further inward, which could help form a larger container.
   - This is done until the two pointers meet or cross each other.

### Code Implementation:
The detailed implementation in Python can be formatted to run directly on LeetCode:



```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0  # Starting point
        right = len(height) - 1  # End point
        max_water = 0  # To keep track of the maximum water collected
        
        while left < right:
            # Calculate the current area with the lower of the two heights
            current_water = (right - left) * min(height[left], height[right])
            max_water = max(max_water, current_water)
            
            # Move the pointer pointing to the shorter line
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return max_water

# These comments can be ignored in actual submission; they describe how you might invoke the solution class and function
# sol = Solution()
# print(sol.maxArea([1,8,6,2,5,4,8,3,7]))  # Example 1
# print(sol.maxArea([1,1]))                 # Example 2
# print(sol.maxArea([4,3,2,1,4]))          # Example 3
# print(sol.maxArea([1,2,1]))              # Example 4

```

### Explanation:
- We initialize two pointers, `left` starting at index `0` and `right` at the last index.
- A `while` loop runs until the two pointers meet.
- Inside the loop, we compute the area formed by `height[left]` and `height[right]`, updating `max_water` if the current area is greater than the previous maximum.
- Depending on which pointer points to the shorter height, we move that pointer inward to hopefully find a greater height.
- Return `max_water`, which finally contains the maximum area.

This solution has a linear time complexity of \( O(n) \).

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
 To solve this problem on LeetCode, we will follow the "Two Pointer" technique after sorting the array, which will ensure `O(n^2)` time complexity. This method is effective since it traverses the array linearly after sorting and makes use of the sorted property to decide how to move the pointers. Here's a step-by-step breakdown of our approach:

1. **Sort the array**: This makes it easier to avoid duplicates and makes it simple to move pointers based on their sum.
2. **Iterate through the array**: Use each element as a potential first element of the triplet.
3. **Two Pointer Technique**: For each element, use two pointers to find pairs that together with the current element sum up to zero.
    - One pointer starts just after the current element.
    - The other pointer starts at the end of the array.
4. **Avoiding Duplicates**: Skip over duplicated elements to make sure we don't count the same triplet more than once.
5. **Moving Pointers**:
    - If the current sum is less than zero, move the left pointer to the right to increase the sum.
    - If the current sum is greater than zero, move the right pointer to the left to decrease the sum.
    - If the sum is zero, save the triplet and adjust both pointers, watching for duplicates.

This approach ensures that we do not have duplicated triplets in the output, satisfying the problem's requirements.



```python
class Solution:
    def threeSum(self, nums):
        # Sort the array
        nums.sort()
        n = len(nums)
        result = []

        # Iterate through the array
        for i in range(n-2):  # The last two elements are covered by the two pointers
            if i > 0 and nums[i] == nums[i-1]:
                continue  # Skip the same result to avoid duplicates
            
            left, right = i + 1, n - 1
            while left < right:
                current_sum = nums[i] + nums[left] + nums[right]
                
                if current_sum < 0:
                    left += 1  # Increase the sum
                elif current_sum > 0:
                    right -= 1  # Decrease the sum
                else:
                    # If the sum is zero, add to the results and move both pointers
                    result.append([nums[i], nums[left], nums[right]])
                    
                    # Move left until the next unique element
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    # Move right until the next unique element
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
        
        return result

# Example uses
sol = Solution()
print(sol.threeSum([-1,0,1,2,-1,-4]))  # Output: [[-1,-1,2],[-1,0,1]]
print(sol.threeSum([]))  # Output: []
print(sol.threeSum([0]))  # Output: []

```

This solution efficiently handles sorting and verifying conditions using two pointers to find valid triplets that sum to zero. It ensures all triplets are unique and checks the entire array systematically. The constraints and format suit a direct execution on LeetCode with the given class and method structure.

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
 To solve this problem optimally, we can utilize a sorting-based approach combined with a two-pointer technique. Below is a step-by-step breakdown of how we should proceed, followed by the entire code implemented as a Python class with a suitable method, as typically required in LeetCode problems.

### Explanation:
1. **Sort the Array**: First, we will sort the given array. Sorting helps us efficiently evaluate possible combinations using two-pointer approach.
  
2. **Initialize Variables**: 
   - `closest_sum`: To store the closest sum to the target observed so far. Initialize it with a sum of the first three elements.
   
3. **Two-pointer approach for the remaining array**:
   - Iterate through each element in the array (from the first to third last).
   - For each element at index `i`, use two pointers, one starting just after `i` (`left = i+1`) and one at the end of the array (`right = len(nums)-1`) to find a combination such that the sum of elements at positions `i`, `left`, and `right` is closest to `target`.
   
4. **Finding the Closest Sum**:
   - For each position `i`, move `left` and `right` pointers appropriately:
     - Compute the sum of elements at indices `i`, `left`, and `right`.
     - If the sum equals the `target`, return the sum immediately as we can't get closer.
     - If the sum is less than the `target`, move the `left` pointer to the right to increase the sum.
     - If the sum is more than the `target`, move the `right` pointer to the left to decrease the sum.
   - Track the closest sum by comparing the absolute difference between the current sum and the `target` with the closest sum found so far.

### Python Code:



```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()  # Sort the array first
        n = len(nums)
        closest_sum = sum(nums[:3])  # Initialize with the sum of the first three elements
        
        for i in range(n-2):
            left, right = i+1, n-1
            while left < right:
                current_sum = nums[i] + nums[left] + nums[right]
                
                # Check if the current sum is more precise relative to the previous closets
                if abs(current_sum - target) < abs(closest_sum - target):
                    closest_sum = current_sum
                
                # If the current_sum is exactly the target, it's the closest possible sum
                if current_sum == target:
                    return current_sum
                
                # Move pointers
                if current_sum < target:
                    left += 1
                else:
                    right -= 1
                
        return closest_sum

```

### Description:
This class named `Solution` contains a method `threeSumClosest`, which takes the sorted list and a target integer and returns an integer representing the closest sum to the target that can be obtained by summing any three distinct numbers in the list, using the approach outlined.

This solution is effective and ensures that we find the closest possible sum efficiently. The overall time complexity of the solution is \(O(n^2)\), which is quite optimal for the given problem constraints.

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
 To solve this problem, we'll implement a multiple-pointer technique along with some sorting and a few checks to skip over duplicates and reduce the search space. The strategy is guided by the following steps:

1. **Sort the Array**: We start by sorting the array. Sorting helps us easily skip duplicates and use two-pointers techniques.
  
2. **Four Nested Loops? Not quite**: Instead of using four nested loops which would lead to an `O(n^4)` time complexity, we can optimize by using two nested loops (for the first two elements) and a two-pointer technique (for the last two elements).

3. **Skipping Duplicates**: While iterating, we need to skip over duplicate elements to ensure unique quadruplets.

4. **Using Two-pointers**: After fixing the first two elements, we use a left pointer starting just after the second fixed element, and a right pointer at the end of the array. We adjust these pointers based on their sum compared to the target.

5. **Early Stopping Conditions**: If at any point, the smallest possible sum of four values is greater than the target, we can break out of the loop early, as further values will only increase because the array is sorted. Similarly, if the largest possible sum of four values becomes less than the target, we can advance our loop.

This algorithm will efficiently find all unique quadruplets that add up to the given target. Here is the Python code structured to run directly on LeetCode for the described solution:



```python
class Solution:
    def fourSum(self, nums, target):
        nums.sort()  # Sort the array first. O(n log n) time complexity
        result = []
        n = len(nums)
        
        for i in range(n-3):
            if i > 0 and nums[i] == nums[i-1]:  # Skip duplicate 'a'
                continue
            if nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target:  # Early termination
                break
            if nums[i] + nums[n-1] + nums[n-2] + nums[n-3] < target:  # Current number too small
                continue
                
            for j in range(i+1, n-2):
                if j > i+1 and nums[j] == nums[j-1]:  # Skip duplicate 'b'
                    continue
                if nums[i] + nums[j] + nums[j+1] + nums[j+2] > target:  # Early termination
                    break
                if nums[i] + nums[j] + nums[n-1] + nums[n-2] < target:  # Current number too small
                    continue
                    
                left, right = j+1, n-1
                while left < right:
                    current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                    if current_sum == target:
                        result.append([nums[i], nums[j], nums[left], nums[right]])
                        left += 1
                        right -= 1
                        while left < right and nums[left] == nums[left-1]:  # Skip duplicate 'c'
                            left += 1
                        while left < right and nums[right] == nums[right+1]:  # Skip duplicate 'd'
                            right -= 1
                    elif current_sum < target:
                        left += 1
                    else:
                        right -= 1
        
        return result

# Sample code to use the `Solution` class
sol = Solution()
print(sol.fourSum([1,0,-1,0,-2,2], 0))
print(sol.fourSum([], 0))

```

Explanation:
- We sort the array to facilitate the skipping of duplicates and to implement the two-pointer method effectively.
- Two initial loops fix the first two numbers.
- A two-pointer approach determines the possible last two numbers.
- Special conditions check to move pointers only when needed and avoid duplicates. 

This solution should comfortably operate within the bounds of typical inputs given the constraints.

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
 Below is a Python solution for the problem statement provided, suitable for execution in a typical online coding platform like LeetCode.



```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0

        # Initialize the unique index pointer.
        unique_index = 0

        # Loop through the array starting from the second element
        for i in range(1, len(nums)):
            # Compare current element with the element at unique_index
            if nums[i] != nums[unique_index]:
                # Move unique_index forward
                unique_index += 1
                # Update the value at unique_index with the new unique value found at i
                nums[unique_index] = nums[i]

        # Return the length of the array up to the last updated unique_index
        return unique_index + 1

# Example usage:
# sol = Solution()
# res = sol.removeDuplicates([1,1,2])
# print(res)   # Output: 2
# res2 = sol.removeDuplicates([0,0,1,1,1,2,2,3,3,4])
# print(res2)  # Output: 5

```

### Explanation of the Solution:

1. **Function Signature Explained**: 
   - `def removeDuplicates(self, nums: List[int]) -> int:` specifies a class method which expects a list of integers (`nums`) as input and returns an integer.

2. **Edge Case Handling**:
   - The condition `if not nums:` checks if the list is empty. If `nums` is empty, the function returns `0` because there are no elements in the list.

3. **Initializing Pointers**:
   - `unique_index` is initialized to `0`. This variable is used to track the position where the next unique number should be placed.

4. **Iterating Through the List**:
   - We iterate through the list starting from the second element (`i = 1`). This is because the first element is inherently unique as there are no prior elements to compare with.

5. **Checking for Duplicates**:
   - If `nums[i] != nums[unique_index]`, it indicates that we have encountered a new unique number. This warrants updating the array to set `nums[unique_index + 1]` to this new unique number (`nums[i]`).

6. **Update unique_index**:
   - The `unique_index` is incremented each time a new unique number is found. This pointer keeps track of the position where the next unique number should be written.

7. **Final Output**:
   - The method returns `unique_index + 1` as the count of unique numbers found. Adding `1` compensates for zero-based indexing (i.e., to count from `0` to `unique_index` inclusive).

8. **LeetCode Specifics**:
   - In a LeetCode submission, the array `nums` is modified in-place as specified, and the secure output produces only the count of unique elements. The modified `nums[:unique_index+1]` elements reflect the distinct values up to the count returned. This approach ensures the use of O(1) extra memory as required.

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
 Below is a Python code solution for the described problem, formatted to run directly on the LeetCode platform:



```python
from typing import List

class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        # Initialize a pointer for the place to put the next non-val element
        place_index = 0
        
        # Iterate through the list `nums`
        for num in nums:
            # Check if the current value is not the value to be removed
            if num != val:
                # Place the value at the `place_index` and increment `place_index`
                nums[place_index] = num
                place_index += 1
        
        # Return the new length
        return place_index

# Example usage:
# sol = Solution()
# print(sol.removeElement([3,2,2,3], 3))  # Output: 2
# print(sol.removeElement([0,1,2,2,3,0,4,2], 2))  # Output: 5

```

### Explanation:

The solution uses an in-place algorithm to modify the input array `nums` directly. Here’s a step-by-step explanation:

1. **Initialization**: A variable called `place_index` is used to track the position in the array where the next non-`val` element should be placed. It starts from index 0.

2. **Iterating Through `nums`**: We go through each element in the array with a `for` loop.

3. **Element Check**: For each element, if it is not equal to `val`, it is placed at `nums[place_index]`. After placing the element, `place_index` is incremented to prepare it for the next non-`val` element. This effectively shifts all non-`val` elements to the beginning of the array.

4. **Modification in Place**: The condition `num != val` ensures that any instance of `val` is skipped, thus indirectly removing it from the array by not copying it to the front.

5. **Return Value**: After the loop completes, all elements that are not `val` are at the start of the array, and `place_index` gives the new length of the array as it represents the number of non-`val` elements processed.

This method ensures an in-place modification with `O(1)` extra space (only using additional space for the pointer variable) and `O(n)` time complexity as it processes each element of the array exactly once.

# 31. Next Permutation

### Problem Description 
Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such an arrangement is not possible, it must rearrange it as the lowest possible order (i.e., sorted in ascending order).

The replacement must be in place and use only constant extra memory.


Example 1:
Input: nums = [1,2,3]
Output: [1,3,2]

Example 2:
Input: nums = [3,2,1]
Output: [1,2,3]

Example 3:
Input: nums = [1,1,5]
Output: [1,5,1]

Example 4:
Input: nums = [1]
Output: [1]

Constraints:
`1 <= nums.length <= 100`
`0 <= nums[i] <= 100`

### Solution 
 Below is the Python code that solves the problem described. It will be in the format that can be directly run on the LeetCode platform.



```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i = len(nums) - 2
        
        # Find the first decreasing element from the end
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        
        if i >= 0:
            # Find the first element from the end that's greater than nums[i]
            j = len(nums) - 1
            while nums[j] <= nums[i]:
                j -= 1
            
            # Swap
            nums[i], nums[j] = nums[j], nums[i]
        
        # Reverse the sequence from i + 1 to end to get the next lowest lexicographical permutation
        l, r = i + 1, len(nums) - 1
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1

"""
Explanation:

The algorithm proceeds with the following steps:

1. Identify the rightmost element `i` (second last from the end) which is smaller than the element next to it. This is because to obtain the next permutation, we need to increase the sequence from the rightmost end as minimally as possible. 

2. Once such an element is found (`nums[i]`), find the smallest element on the right of `i` which is larger than `nums[i]` (`nums[j]`). This step ensures we only increment our sequence minimally.

3. Swap `nums[i]` and `nums[j]`.

4. Finally, reverse the sequence after index `i`. This again ensures that we obtain the next smallest lexicographical order, as reversing orders a sequence in the smallest possible way starting from a given point.

If no `i` is found whereby `nums[i] < nums[i + 1]`, this means the entire sequence is non-increasing and thus the next permutation is just the smallest permutation. In this scenario, we reverse the entire array.
"""

# The function signature may be used directly in LeetCode submissions.

```

This code snippet respects the constant space requirement as it only uses a constant amount of additional space (for variables such as `i`, `j`, `l`, and `r`) and operates directly on the input `nums` list. This method changes the list in-place, providing the next permutation or the smallest permutation directly modifying the passed `nums` list.




# 33. Search in Rotated Sorted Array

### Problem Description 
There is an integer array `nums` sorted in ascending order (with distinct values).

Prior to being passed to your function, `nums` is rotated at an unknown pivot index `k` (`0 <= k < nums.length`) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (0-indexed). For example, `[0,1,2,4,5,6,7]` might be rotated at pivot index `3` and become `[4,5,6,7,0,1,2]`.

Given the array `nums` after the rotation and an integer `target`, return the index of `target` if it is in `nums`, or `-1` if it is not in `nums`.


Example 1:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Example 2:
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

Example 3:
Input: nums = [1], target = 0
Output: -1

Constraints:
`1 <= nums.length <= 5000`
`-104 <= nums[i] <= 104`
All values of `nums` are unique.

`nums` is guaranteed to be rotated at some pivot.

`-104 <= target <= 104`
Follow up: Can you achieve this in `O(log n)` time complexity?

### Solution 
 The problem you've described is a classic example often called "Search in Rotated Sorted Array". The goal is to find the index of a target value in an array that's been sorted and then rotated. This challenge is typically solved using a modified binary search to capitalize on the sorted nature of the array segments.

Here's the strategy:
1. Use binary search. Instead of standard binary search, modify it to suit the rotated nature of the array.
2. At each step, determine which segment of the array is sorted (either the left half or the right half must be sorted due to the rotation).
3. Check if the target lies within the sorted segment.
   - If it does, narrow your search to that segment.
   - If it doesn't, search in the other segment.
  
This approach adheres to the O(log n) complexity requirement due to the nature of binary search, where we halve the search space with each iteration.

Here’s the implementation according to the details provided:



```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if nums[mid] == target:
                return mid
            
            # Determine the sorted side of the array
            if nums[left] <= nums[mid]:  # the left half is sorted
                if nums[left] <= target < nums[mid]:  # target is in the left sorted half
                    right = mid - 1
                else:
                    left = mid + 1
            else:  # the right half is sorted
                if nums[mid] < target <= nums[right]:  # target is in the right sorted half
                    left = mid + 1
                else:
                    right = mid - 1
                    
        return -1

```

This code snippet should be directly runnable on the LeetCode platform under the problem "Search in Rotated Sorted Array". What it does is make efficient use of binary search principles while accounting for the rotation in the array to ensure the search remains within O(log n) time complexity.

# 34. Find First and Last Position of Element in Sorted Array

### Problem Description 
Given an array of integers `nums` sorted in ascending order, find the starting and ending position of a given `target` value.

If `target` is not found in the array, return `[-1, -1]`.

Follow up: Could you write an algorithm with `O(log n)` runtime complexity?

Example 1:
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

Example 2:
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]

Example 3:
Input: nums = [], target = 0
Output: [-1,-1]

Constraints:
`0 <= nums.length <= 105`
`-109 <= nums[i] <= 109`
`nums` is a non-decreasing array.

`-109 <= target <= 109`

### Solution 
 To solve the problem of finding the start and end positions of a given target value in a sorted array with O(log n) complexity, we can leverage binary search. Given the sorted nature of the array, binary search becomes an optimal approach as it allows for faster pinpointing of elements by continually dividing the searchable portion of the list in half.

Here's how you can implement the solution in Python:

1. **Binary Search Function**: Write a helper function to perform binary search, allowing a tweak to find either the first occurrence (leftmost position) or the last occurrence (rightmost position) of the target.

2. **Main Function**: Use the binary search function twice in the main function:
   - Once to find the first (leftmost) occurrence of the target.
   - Once to find the last (rightmost) occurrence of the target.

Here's the code implementing the above plan:



```python
from typing import List

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def findFirstPosition(nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if nums[mid] < target:
                    left = mid + 1
                elif nums[mid] > target:
                    right = mid - 1
                else:
                    if mid == 0 or nums[mid - 1] != target:
                        return mid
                    right = mid - 1
            return -1
        
        def findLastPosition(nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if nums[mid] < target:
                    left = mid + 1
                elif nums[mid > 0:
                    right = mid - 1
                else:
                    if mid == len(nums) - 1 or nums[mid + 1] != target:
                        return mid
                    left = mid + 1
            return -1
        
        start = findFirstPosition(nums, target)
        if start == -1:  # If start is not found, no need to look for end
            return [-1, -1]

        end = findLastPosition(nums, target)

        return [start, end]

# Example usage:
sol = Solution()
print(sol.searchRange([5,7,7,8,8,10], 8))  # Output: [3, 4]
print(sol.searchRange([5,7,7,8,8,10], 6))  # Output: [-1, -1]
print(sol.searchRange([], 0))             # Output: [-1, -1]

```

**Explanation:**
- `findFirstPosition()` uses binary search to find the first index where `target` appears. If `nums[mid]` is equal to `target`, it checks whether this occurrence is the first by comparing it with the preceding element.
- `findLastPosition()` is similar but intended to find the last position where `target` appears. It checks if the current `target` match is the last by comparing it with the next element.
- These functions are optimized for repeated elements and directly handle edge cases where the target might appear at the beginning or end of `nums`. If `nums` is empty or the target is not found, `[-1, -1]` is returned.

This solution adheres to the O(log n) complexity requirement by ensuring that both searches are done via binary search, which divides the search space in half with each step.

# 35. Search Insert Position

### Problem Description 
Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.


Example 1:
Input: nums = [1,3,5,6], target = 5
Output: 2

Example 2:
Input: nums = [1,3,5,6], target = 2
Output: 1

Example 3:
Input: nums = [1,3,5,6], target = 7
Output: 4

Example 4:
Input: nums = [1,3,5,6], target = 0
Output: 0

Example 5:
Input: nums = [1], target = 0
Output: 0

Constraints:
`1 <= nums.length <= 104`
`-104 <= nums[i] <= 104`
`nums` contains distinct values sorted in ascending order.

`-104 <= target <= 104`

### Solution 
 Here's a Python solution using binary search that would work on a platform like LeetCode, along with a detailed explanation:



```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return left

# Example usage:
sol = Solution()
print(sol.searchInsert([1,3,5,6], 5))  # Output: 2
print(sol.searchInsert([1,3,5,6], 2))  # Output: 1
print(sol.searchInsert([1,3,5,6], 7))  # Output: 4
print(sol.searchInsert([1,3,5,6], 0))  # Output: 0
print(sol.searchInsert([1], 0))        # Output: 0

```

### Explanation:

The `searchInsert` function takes two parameters: a sorted list of distinct integers `nums` and an integer `target`. It returns the index at which `target` can be found in `nums`, or the index where it should be inserted to maintain the sorted order.

We initiate two pointers, `left` and `right`, to represent the boundaries of the range within the list we are considering. Initially, `left` is set to 0 and `right` is set to the last index of `nums`.

We then enter a loop which continues as long as `left` is not greater than `right`. Inside the loop, we calculate the middle index `mid` of the current range.

There are three conditions we check:
1. If `nums[mid]` equals `target`, we return `mid` because we've found the target at index `mid`.
2. If `nums[mid]` is less than `target`, it means `target` should occur after `mid`, so we move the `left` boundary to `mid + 1`.
3. If `nums[mid]` is greater than `target`, then `target` should occur before `mid`, so we move the `right` boundary to `mid - 1`.

If we exit the loop, `left` will be pointing to the position where `target` can be safely inserted to maintain the sorted order of `nums`, so we return `left`.

This implementation uses a binary search to find the target efficiently, reducing the time complexity to O(log n), where n is the number of elements in `nums`. This is much faster than a linear search, especially for large lists. The space complexity is O(1) since no additional storage is used.

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
 This problem calls for finding all unique combinations of a given set of integers (`candidates`) that sum up to a target value (`target`). Due to the property that each number can be used an unlimited number of times, it can be efficiently solved using a backtracking approach.

### Strategy:
1. **Use Backtracking:** This will help us explore each possibility of combinations starting from each candidate. The idea is to recursively try each candidate, subtracting its value from the target until the target is hit or exceeded.
   
2. **Sort for Efficiency:** Sorting the candidates can allow us to terminate early in some cases. If during our combination building the rest of the target becomes less than the current candidate, we can stop further exploration in that direction.

3. **Pruning:** If at any step, the remaining target becomes zero, we have a valid combination which we can add to our result. If it becomes negative or if all possibilities have been exhausted, backtrack to explore other possibilities.

### Python Function:
The below function uses the aforementioned technique to find and return all unique combinations. It includes an internal helper function (`dfs`) which handles the recursive exploration.



```python
class Solution:
    def combinationSum(self, candidates, target):
        def dfs(start, current, target):
            if target == 0:  # Target met; store the combination
                results.append(current[:])
                return
            if target < 0:  # Target surpassed without match
                return
            
            for i in range(start, len(candidates)):
                # Include the candidate
                current.append(candidates[i])
                # Continue to explore further with the same candidate
                dfs(i, current, target - candidates[i])
                # Backtrack and remove the last element
                current.pop()
        
        results = []
        # Sort candidates to optimize the process a bit, optional but helpful
        candidates.sort()
        dfs(0, [], target)
        return results

# Example of how to use:
sol = Solution()
print(sol.combinationSum([2,3,6,7], 7))  # Output: [[2, 2, 3], [7]]

```

This code can be run directly on the LeetCode platform:
1. It defines the specified method signature `combinationSum(self, candidates, target)`.
2. Uses depth-first search (DFS) for backtracking.
3. Properly adjusts the path (list of current candidates) and target during recursion.
4. Can handle any number of combinations without exceeding recursive depth by managing states with iterative deepening. The number of combinations is guaranteed to be manageable according to the problem constraints.

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
 Below is the Python code implementing the solution to the problem statement provided based on the LeetCode solution format. This problem is a variation of the classic "Subset Sum" or "Combination Sum" problem and requires generating all unique combinations of a list of numbers that sum to a given target.




```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        # Sort the candidates to help easily find duplicates and make it easier to manage combinations
        candidates.sort()
        results = []
        
        def backtrack(start, target, path):
            # If the remaining target is 0, a valid combination is found
            if target == 0:
                results.append(path[:])
                return
            if target < 0:
                # If the target becomes negative, stop the process as further addition will only decrease it
                return
            
            for i in range(start, len(candidates)):
                # To avoid duplicates, skip the same element at the same recursive level
                if i > start and candidates[i] == candidates[i-1]:
                    continue
                
                # Include the current number in the path
                path.append(candidates[i])
                # Recurse with the updated target and next starting index
                backtrack(i + 1, target - candidates[i], path)
                # Backtrack to explore other possibilities by excluding the last included number
                path.pop()
        
        # Start the backtracking from 0 index and with the provided target
        backtrack(0, target, [])
        
        return results

# For testing purposes, to be executed in a local environment or within the LeetCode platform
if __name__ == "__main__":
    sol = Solution()
    print(sol.combinationSum2([10,1,2,7,6,1,5], 8))  # Should print [[1,1,6], [1,2,5], [1,7], [2,6]]
    print(sol.combinationSum2([2,5,2,1,2], 5))        # Should print [[1,2,2], [5]]

```

**Explanation:**

1. **Sorting**: The candidates are initially sorted. Sorting helps in easily skipping duplicates and also makes it easier to handle the logic of generating sums.

2. **Backtracking**: The `backtrack` function is a recursive function responsible for generating combinations. It accepts the current starting index (`start`), the remaining `target`, and the current path of numbers (`path`).

3. **Base Case Check**: If `target` becomes zero, a valid combination is found matching the exact sum, and it is added to the result list. If the `target` is negative, the function returns immediately as no valid sum can be formed.

4. **Skipping Duplicates**: Within the loop in the `backtrack` function, duplicates are skipped. This is done to ensure that the solution set does not contain any duplicate combinations. If the current number is the same as the previous at the same recursion level, it's skipped.

5. **Recursive Calls and Backtracking**: The function includes the current number in `path`, and makes a recursive call with an adjusted target (`target - candidates[i]`) and starting from the next index. After returning from recursion, the function backtracks by removing the last inserted number from `path`, allowing for the exploration of different combinations.
  
6. **Starting the Process**: The process is started by calling `backtrack(0, target, [])`.

This solution leverages depth-first search via recursion and backtracking, ensuring that all potential combinations are explored efficiently, respecting the constraints like each number's single use and the uniqueness of combinations.

# 41. First Missing Positive

### Problem Description 
Given an unsorted integer array `nums`, find the smallest missing positive integer.


Example 1:
Input: nums = [1,2,0]
Output: 3

Example 2:
Input: nums = [3,4,-1,1]
Output: 2

Example 3:
Input: nums = [7,8,9,11,12]
Output: 1

Constraints:
`0 <= nums.length <= 300`
`-231 <= nums[i] <= 231 - 1`
Follow up: Could you implement an algorithm that runs in `O(n)` time and uses constant extra space?

### Solution 
 To solve the problem of finding the smallest missing positive integer from an unsorted array, we can leverage the property that the smallest missing positive integer must be in the range from `1` to `n+1`, where `n` is the length of the array. This observation is crucial because it allows us to use a technique often referred to as "index marking" within the array itself to manage space complexity.

Here's the step-by-step breakdown of the approach:
1. **First Pass - Cleanup**: Convert all negative numbers and zeros to a number outside the interest range (e.g., `n+1`). This step ensures that only positive numbers and possibly some large numbers (if present) remain in the array. These large numbers are out of our search scope.
2. **Second Pass - Marking**: For every positive integer `k` in the array that is within the range `1` to `n`, use the value of `k` as an index reference (subtract 1 for zero-based indexing) and mark that specific index in the array. The marking can be done by negating the number at index `k-1` if it's a positive number, indicating that `k` exists in the array.
3. **Third Pass - First Missing Positive**: Iterate through the array, and the first index that contains a positive number indicates the smallest missing positive integer. If all numbers from `1` to `n` are present, then the answer is `n+1`.

This entire procedure ensures an `O(n)` time complexity and uses `O(1)` extra space, conforming to the problem's constraints and optimal solution requirements.

Here's the code that implements the above logic:



```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        
        # First pass to convert negative numbers and zeros to n+1
        for i in range(n):
            if nums[i] <= 0:
                nums[i] = n + 1
        
        # Second pass to use the index of the array to store the presence of numbers
        for i in range(n):
            num = abs(nums[i])
            if num <= n:
                if nums[num - 1] > 0:
                    nums[num - 1] *= -1
        
        # Third pass to find the first missing positive number
        for i in range(n):
            if nums[i] > 0:
                return i + 1
        
        # If all numbers from 1 to n are present
        return n + 1

# Example usage:
# sol = Solution()
# print(sol.firstMissingPositive([1,2,0]))  # Output: 3
# print(sol.firstMissingPositive([3,4,-1,1]))  # Output: 2
# print(sol.firstMissingPositive([7,8,9,11,12]))  # Output: 1

```

You can paste this code directly into a Python file or a LeetCode submission box. The code includes the `Solution` class with the method `firstMissingPositive`, which you can execute with test cases to verify its correctness.

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
 To solve the problem of finding the trapped water in an elevation map, we can leverage a two-pointer approach. This approach reduces the need to continuously calculate the maximum heights to the left and right of every position, an approach which, while straightforward, leads to inefficient solutions with a time complexity of O(n^2).

**Method: Two-pointer approach**

- **Intuition:**
  Water trapped at any index depends on the heights of taller or equal bars to the left and right of this index. The smaller of these two heights determines the maximum water level at this index; any excess elevation of the index reduces the water that can be trapped.

- **Algorithm:**
  1. Initialize two pointers, `left` starting from the beginning (index 0) and `right` starting from the end (last index) of the height array.
  2. Use two variables, `left_max` and `right_max`, to continuously track the maximum heights encountered from the left and right ends respectively.
  3. While the `left` pointer is to the left of the `right` pointer:
     - If `height[left]` is smaller than or equal to `height[right]`, then:
       - Check if the current height at the `left` pointer is less than or equal to `left_max`. If yes, this means water can be trapped at this point. Add difference between `left_max` and `height[left]` to the result.
       - Update `left_max` to be the maximum of itself and `height[left]`, and then move the `left` pointer to the right.
     - Otherwise:
       - Check the water that can be trapped at the right pointer analogously using `right_max`.
       - Move the `right` pointer to the left.
  4. Keep accumulating the trapped water amount until the pointers meet.

- **Time Complexity**: O(n), as each element is processed at most once.
- **Space Complexity**: O(1), since only constant space is used.

This solution efficiently computes the trapped water using linear scans with two pointers. Below is the implementation.



```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0
        
        left, right = 0, len(height) - 1
        left_max, right_max = height[left], height[right]
        trapped_water = 0
        
        while left < right:
            if height[left] <= height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    trapped_water += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    trapped_water += right_max - height[right]
                right -= 1
                
        return trapped_water

# you can test the code using these examples
# solution = Solution()
# print(solution.trap([0,1,0,2,1,0,1,3,2,1,2,1]))  # Output: 6
# print(solution.trap([4,2,0,3,2,5]))               # Output: 9

```

This code can be run on the LeetCode platform as part of the problem "42. Trapping Rain Water" and should meet the performance requirements for all provided test cases.

# 45. Jump Game II

### Problem Description 
Given an array of non-negative integers `nums`, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Your goal is to reach the last index in the minimum number of jumps.

You can assume that you can always reach the last index.


Example 1:
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.


Example 2:
Input: nums = [2,3,0,1,4]
Output: 2

Constraints:
`1 <= nums.length <= 1000`
`0 <= nums[i] <= 105`

### Solution 
 Here is a Python solution to the given problem using a greedy approach. This approach ensures that we achieve the minimum number of jumps required to get to the last index. Below is the code with a detailed explanation:



```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0  # No need to jump if there's only one element
        
        jumps = 0
        current_end = 0
        farthest = 0
        for i in range(n - 1):
            # Farthest point that can be reached from the current range (i to current_end)
            farthest = max(farthest, i + nums[i])
            
            # When we've reached the end of the range covered by the current jump,
            # we need a new jump.
            if i == current_end:
                jumps += 1
                current_end = farthest
                
                # Optimization: early stop if we can reach or exceed the last index
                if current_end >= n - 1:
                    break
        
        return jumps

```

### Explanation:
1. **Initialization**: 
   - `jumps` keeps track of the minimum jumps needed.
   - `current_end` stores the farthest index that can be reached with the current number of jumps.
   - `farthest` stores the farthest index that can be reached from any index within the current range (up to `current_end`).

2. **Loop through the Array**:
   - Iterate i from `0` to `n-1` (exclude the last index because we do not need to jump from the last index).
   
3. **Update the Farthest Reach**:
   - For each position `i`, update `farthest` to be the maximum of `farthest` and `i + nums[i]` (the farthest index we can reach from the current index i).

4. **Check if we need a new jump**:
   - When the current index `i` matches `current_end` (the farthest we could get with the previous jumps):
     - Increment `jumps` (make a new jump).
     - Update `current_end` to `farthest`.
     
5. **Optimization**: 
   - If `current_end` reaches or surpasses the last index (`n-1`), we can stop early.

6. **Return the Result**:
   - Finally, return the total count of `jumps`. 

This solution uses a greedy strategy, performing the jumps to reach the farthest possible within the array with minimum jumps. It efficiently calculates the minimum number of jumps required to reach the last index of the array.

This code is ready to be run on the LeetCode platform for Python problems.

# 48. Rotate Image

### Problem Description 
You are given an n x n 2D `matrix` representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.


Example 1:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]

Example 2:
Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]

Example 3:
Input: matrix = [[1]]
Output: [[1]]

Example 4:
Input: matrix = [[1,2],[3,4]]
Output: [[3,1],[4,2]]

Constraints:
`matrix.length == n`
`matrix[i].length == n`
`1 <= n <= 20`
`-1000 <= matrix[i][j] <= 1000`

### Solution 
 To solve the problem of rotating a 2D matrix by 90 degrees clockwise, we can follow an approach which involves matrix transposition and then reversing rows. Here's the systematic breakdown and logic:

1. **Transpose the Matrix**: Swap rows with columns. For matrix element at position [i][j], swap its position with the element at [j][i]. This operation alters rows into columns (and vice versa) while preserving the ordering within rows and columns.

2. **Reverse Each Row**: After transposing, the new "rows" of the matrix are actually partial columns from the original matrix, but read top-to-bottom (which reflects a 90-degree anticlockwise rotation). To convert this to a clockwise rotation, the order of elements in each row is reversed.

This two-step approach works in-place, meaning that no additional storage is necessary aside from temporary variables needed for swapping. Let's implement this logic in Python respecting the LeetCode format:



```python
class Solution:
    def rotate(self, matrix):
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        
        # Transpose the matrix
        for i in range(n):
            for j in range(i, n):  # Note: we only start j from i to avoid re-swapping already swapped elements
                if i != j:
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        # Reverse each row
        for i in range(n):
            matrix[i].reverse()

# Example usage inside LeetCode's framework
# You probably would not run these in LeetCode submission, but it helps in understanding or local testing
if __name__ == "__main__":
    sol = Solution()
    matrix1 = [[1,2,3],[4,5,6],[7,8,9]]
    sol.rotate(matrix1)
    print(matrix1)  # Output: [[7,4,1],[8,5,2],[9,6,3]]
    
    matrix2 = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
    sol.rotate(matrix2)
    print(matrix2)  # Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
    
    matrix3 = [[1]]
    sol.rotate(matrix3)
    print(matrix3)  # Output: [[1]]
    
    matrix4 = [[1,2],[3,4]]
    sol.rotate(matrix4)
    print(matrix4)  # Output: [[3,1],[4,2]]

```

This code modification changes the matrix in-place, adhering to the constraints, and performs a 90-degree clockwise rotation of the matrix using a transpose and reverse sequence. This method is optimal for the problem's constraints and is directly testable on the provided example cases.

# 53. Maximum Subarray

### Problem Description 
Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.


Example 1:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.


Example 2:
Input: nums = [1]
Output: 1

Example 3:
Input: nums = [5,4,-1,7,8]
Output: 23

Constraints:
`1 <= nums.length <= 3 * 104`
`-105 <= nums[i] <= 105`
Follow up: If you have figured out the `O(n)` solution, try coding another solution using the divide and conquer approach, which is more subtle.

### Solution 
 The problem can be approached using Kadane's Algorithm, which is an efficient way to find the maximum sum of a contiguous subarray. The concept behind the algorithm is to look for all positive contiguous segments of the array and to keep track of the maximum sum contiguous segment among all the positive segments.

Here’s the Python code that implements this solution:



```python
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if not nums:
            return 0

        current_sum = nums[0]
        max_sum = nums[0]

        for num in nums[1:]:
            # Update current_sum to be the maximum of num itself or num + current_sum
            current_sum = max(num, current_sum + num)
            # Update max_sum to be the maximum value found so far
            max_sum = max(max_sum, current_sum)

        return max_sum

```

In this implementation:
1. We initialize `current_sum` with the first element of the array, which is the starting point for the sum of the subarray.
2. `max_sum` is also initialized with the first element to track the maximum sum of subarrays found so far.
3. We iterate over the array starting from the second element:
   - For each `num` in `nums`, `current_sum` is updated to the greater of `num` itself (new start for a potential subarray) or `current_sum + num` (continuation of the current subarray).
   - `max_sum` is updated by taking the larger value between the existing `max_sum` and the updated `current_sum`.
4. Finally, `max_sum`, which now stores the maximum subarray sum, is returned.

This approach works in O(n) time complexity as it involves a single traversal through the list `nums`. This satisfies the optimal time complexity requirement as outlined in the problem.

You can run the above solution directly on the LeetCode website for the problem titled "Maximum Subarray." It provides a concise and effective solution to find the maximum sum of a contiguous subarray within the given list of integers.

# 54. Spiral Matrix

### Problem Description 
Given an `m x n` `matrix`, return all elements of the `matrix` in spiral order.


Example 1:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]

Example 2:
Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]

Constraints:
`m == matrix.length`
`n == matrix[i].length`
`1 <= m, n <= 10`
`-100 <= matrix[i][j] <= 100`

### Solution 
 To solve the problem of listing matrix elements in spiral order, we utilize a systematic loop traversal method:

1. **Initialization**: Instantiate variables for the boundaries (`top`, `bottom`, `left`, `right`) and an output list (`result`).

2. **Traversal Logic**: Use four traversal modes: from left to right, top to bottom, right to left, and bottom to top. After each traversal, adjust the respective boundaries.

3. **Boundary Adjustments**:
   - After moving left to right, increment the `top` boundary.
   - After moving top to bottom, decrement the `right` boundary.
   - After moving right to left, decrement the `bottom` boundary.
   - After moving bottom to top, increment the `left` boundary.

4. **Loop**: Keep traversing the `matrix` while the respective `left <= right` and `top <= bottom`.

5. **Edge Cases**: Recognize that loops might end before a complete set of movements if matrix dimensions cause boundaries to overlap prematurely.

Here is the code that can be used directly on the LeetCode online judge:



```python
from typing import List

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []
        
        result = []
        left, right = 0, len(matrix[0]) - 1
        top, bottom = 0, len(matrix) - 1
        
        while left <= right and top <= bottom:
            # Traverse from left to right along the top row
            for col in range(left, right + 1):
                result.append(matrix[top][col])
            top += 1
            
            # Traverse from top to bottom along the right column
            for row in range(top, bottom + 1):
                result.append(matrix[row][right])
            right -= 1
            
            if top <= bottom:
                # Traverse from right to left along the bottom row
                for col in range(right, left - 1, -1):
                    result.append(matrix[bottom][col])
                bottom -= 1
            
            if left <= right:
                # Traverse from bottom to top along the left column
                for row in range(bottom, top - 1, -1):
                    result.append(matrix[row][left])
                left += 1

        return result

# Example usage:
sol = Solution()
print(sol.spiralOrder([[1,2,3],[4,5,6],[7,8,9]]))  # Output: [1,2,3,6,9,8,7,4,5]
print(sol.spiralOrder([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))  # Output: [1,2,3,4,8,12,11,10,9,5,6,7]

```

The `spiralOrder` function in the `Solution` class handles various sizes of matrices and ensures that all elements are returned in the requested order. This solution respects the LeetCode format and can be executed directly in their environment.

# 55. Jump Game

### Problem Description 
Given an array of non-negative integers `nums`, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.


Example 1:
Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.


Example 2:
Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.


Constraints:
`1 <= nums.length <= 3 * 104`
`0 <= nums[i] <= 105`

### Solution 
 To solve the problem of determining if we can reach the last index of an array `nums` where each element indicates the maximum jump length at that position, we need an efficient mechanism. A greedy approach provides an effective way to handle this. 

The greedy strategy centers around maintaining the farthest index `max_reachable` we can reach as we iterate through the array. We start at the first index, setting our `max_reachable` initially to this index, and proceed through each index up to `max_reachable`. For each index `i`, we update `max_reachable` to be the maximum of its current value and `i + nums[i]` (the farthest we can get from this index).

If at any iteration, the `max_reachable` index surpasses or equals the last index of the array, we can directly conclude that reaching the last index is possible. On the other hand, if at any point our current index `i` exceeds `max_reachable` (we can't move forward any more), and `max_reachable` does not reach the last index, the function should return `False`.

Here’s a succinct LeetCode style solution in Python:



```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_reachable = 0
        n = len(nums)
        
        for i in range(n):
            if i > max_reachable:
                return False
            max_reachable = max(max_reachable, i + nums[i])
            if max_reachable >= n - 1:
                return True
        
        return max_reachable >= n - 1

```

### Explanation:

1. We declare a variable `max_reachable` to keep track of the farthest point we can reach, initializing it to 0.
2. We iterate through each index `i` of the array.
3. If `i` exceeds `max_reachable`, we encounter a point we cannot move past, so we return `False`.
4. For each index `i`, we update `max_reachable` with the maximum of the current `max_reachable` and `i + nums[i]`.
5. If at any point `max_reachable` reaches or exceeds the last index (`n-1`), we can conclude that reaching the end is possible and return `True`.
6. If the loop completes, we check whether `max_reachable` covers the last index and return accordingly, although this condition is generally covered during the loop.

This solution efficiently determines jump capability with an O(n) time complexity by visiting each index at most once and making a constant-time check at each step.

# 56. Merge Intervals

### Problem Description 
Given an array of `intervals` where `intervals[i] = [starti, endi]`, merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.


Example 1:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].


Example 2:
Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.


Constraints:
`1 <= intervals.length <= 104`
`intervals[i].length == 2`
`0 <= starti <= endi <= 104`

### Solution 
 The problem requires us to merge overlapping intervals in a list. To achieve this, we'll first sort the intervals based on their starting points. Then, we'll iterate through the sorted intervals and merge them when they overlap.

Here's a detailed plan for our solution:
1. Sort the `intervals` list based on each interval's starting point.
2. Initialize an empty list called `merged` to store the merged intervals.
3. Iterate through the sorted intervals and check if the current interval overlaps with the last interval in the `merged` list. If they overlap, modify the last interval in `merged` to encompass the current interval. If they don't overlap, add the current interval to `merged`.
4. Finally, return the `merged` list.

Here's the Python code that implements this plan:



```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # Step 1: Sort the intervals based on the start time
        intervals.sort(key=lambda x: x[0])
        
        # Step 2: Initialize the merged list
        merged = []
        
        # Step 3: Iterate through intervals and merge if necessary
        for interval in intervals:
            # If merged is empty or if there is no overlap, add the interval to merged
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # There is an overlap, so merge with the last interval in merged list
                merged[-1][1] = max(merged[-1][1], interval[1])
        
        # Step 4: Return the merged intervals
        return merged

```

### Detailed Explanation:
1. **Sorting (`intervals.sort(key=lambda x: x[0])`)**: We sort the intervals based on the start of each interval to easily find overlapping intervals.
2. **Iteration and Merging**:
   - For each interval:
     - If the `merged` list is empty (`not merged`), it means we haven't added any intervals yet, so we append the first interval.
     - If there is no overlap (`merged[-1][1] < interval[0]`), that is, the end of the last interval in `merged` is less than the start of the current interval, we append the current interval directly.
     - If there is an overlap (`merged[-1][1] >= interval[0]`), we merge the current interval into the last interval in `merged` by updating the end of the last interval in `merged` to be the maximum of its current end and the end of the current interval.

This approach guarantees that all overlapping intervals are merged, and since it first sorts the list, the merging (while iterating) steps are efficient and straightforward. The overall time complexity is \(O(n \log n)\) due to the sorting step, and the space complexity is \(O(1)\) if we neglect the output space, since sorting and merging are done in place.

# 57. Insert Interval

### Problem Description 
Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.


Example 1:
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]

Example 2:
Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval `[4,8]` overlaps with `[3,5],[6,7],[8,10]`.


Example 3:
Input: intervals = [], newInterval = [5,7]
Output: [[5,7]]

Example 4:
Input: intervals = [[1,5]], newInterval = [2,3]
Output: [[1,5]]

Example 5:
Input: intervals = [[1,5]], newInterval = [2,7]
Output: [[1,7]]

Constraints:
`0 <= intervals.length <= 104`
`intervals[i].length == 2`
`0 <= intervals[i][0] <= intervals[i][1] <= 105`
`intervals` is sorted by `intervals[i][0]` in ascending order.

`newInterval.length == 2`
`0 <= newInterval[0] <= newInterval[1] <= 105`

### Solution 
 To solve this problem on LeetCode, we need to insert a new interval into a list of pre-existing, non-overlapping, sorted intervals, and merge overlapping intervals when necessary.

### Explanation:
1. Iterate through each interval in the list.
2. If the new interval does not overlap with the current interval and ends before the current interval starts, add the new interval to the result and update the new interval to be the current interval.
3. If there is an overlap between the new interval and the current interval, merge them by updating the start of the new interval to the minimum start between both, and the end of the new interval to the maximum end between both.
4. If the new interval is completely smaller and doesn't overlap, add it to the result list directly.

After processing all intervals, if there still is an unadded (or remaining part of the) new interval, it should be added to the result list.

Here’s how the code looks:



```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        result = []
        inserted = False
        
        for interval in intervals:
            if interval[1] < newInterval[0]:  # Current interval ends before new interval starts
                result.append(interval)
            elif interval[0] > newInterval[1]:  # Current interval starts after new interval ends
                if not inserted:
                    result.append(newInterval)
                    inserted = True
                result.append(interval)
            else:  # Overlapping intervals, merge them
                newInterval[0] = min(newInterval[0], interval[0])
                newInterval[1] = max(newInterval[1], interval[1])
        
        if not inserted:
            result.append(newInterval)
        
        return result

```

### Detailed Explanation for LeetCode Format:

1. **Initialization**:
    - `result`: an empty list to store the final list of intervals.
    - `inserted`: a boolean flag to check if the new interval has been added to the result list.

2. **Main Loop**:
    - Iterate through each interval in `intervals`.
    - Compare the `newInterval` with the current interval:
        - **No Overlap** (and `newInterval` comes before the current interval):
            - if `newInterval` is not yet added (using the `inserted` flag), add it before the current interval.
            - add current to results.
        - **Overlap**:
            - Adjust the `newInterval` to be the merger of it and the overlapping current intervals (expand the `newInterval`).
            
3. **Post-Loop**:
    - After the loop, ensure the expanded/merged `newInterval` is added if it wasn't added inside the loop.

This approach maintains the constraints provided, handles merging correctly, and ensures the result list is sorted. The algorithm is efficient, operating in O(n) time complexity, which is optimal since we have to inspect each interval at least once.

# 59. Spiral Matrix II

### Problem Description 
Given a positive integer `n`, generate an `n x n` `matrix` filled with elements from `1` to `n2` in spiral order.


Example 1:
Input: n = 3
Output: [[1,2,3],[8,9,4],[7,6,5]]

Example 2:
Input: n = 1
Output: [[1]]

Constraints:
`1 <= n <= 20`

### Solution 
 To solve the problem of generating an `n x n` matrix filled in a spiral order, we need to keep track of several things during matrix construction:

1. **Direction of Movement**: We'll be filling the matrix in four directions: right, down, left, and up.
2. **Boundaries**: Since we start from the top-left corner of the matrix and move rightward filling values, we have to know when to change direction. This is typically when we hit the edge of the matrix or an already filled cell.
3. **Current Position**: We need to account for our current position within the matrix as we fill it out.

The strategy would be to start at the top left (0, 0), moving right, and continue until we need to turn. At each turn, we modify our direction clockwise and check boundaries. The filling stops once we have placed all values from 1 to `n^2`.

Let’s write Python code for this in the LeetCode format:



```python
class Solution:
    def generateMatrix(self, n: int) -> list[list[int]]:
        # Initialize the `n x n` matrix with zeros
        matrix = [[0] * n for _ in range(n)]
        
        # Directions: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        current_direction = 0  # Start with moving "right"
        current_row, current_col = 0, 0
        
        for i in range(1, n * n + 1):
            matrix[current_row][current_col] = i

            # Determine the next position
            next_row = current_row + directions[current_direction][0]
            next_col = current_col + directions[current_direction][1]

            # Check if the next position is out-of-bounds or already filled
            if (0 <= next_row < n and 0 <= next_col < n and matrix[next_row][next_col] == 0):
                # Move to the next position
                current_row, current_col = next_row, next_col
            else:
                # Change direction
                current_direction = (current_direction + 1) % 4
                # Move to the new valid position
                current_row += directions[current_direction][0]
                current_col += directions[current_direction][1]

        return matrix

# Example usage:
sol = Solution()
print(sol.generateMatrix(3))  # Output: [[1,2,3],[8,9,4],[7,6,5]]
print(sol.generateMatrix(1))  # Output: [[1]]

```

This code:
- Initializes a filled matrix with zeros.
- Uses a `directions` list to toggle between right, down, left, and up movements.
- Iteratively places integers from 1 to `n^2` by checking boundaries or if a cell is already filled to decide direction changes.
- Adjusts the current direction and position based on these checks, ensuring we never overlap or go out of bounds.

This efficiently fills the matrix in the desired spiral order and adheres to the constraints given.

# 62. Unique Paths

### Problem Description 
A robot is located at the top-left corner of a `m x n` grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

Example 1:
Input: m = 3, n = 7
Output: 28

Example 2:
Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down

Example 3:
Input: m = 7, n = 3
Output: 28

Example 4:
Input: m = 3, n = 3
Output: 6

Constraints:
`1 <= m, n <= 100`
It's guaranteed that the answer will be less than or equal to `2 * 109`.

### Solution 
 This problem can be efficiently solved using Dynamic Programming. The key insight is to realize that at each cell in the grid, the number of ways to reach that cell is just the sum of the ways of reaching the cell directly above it and the cell directly to the left of it. This is because the robot can only move either down or right.

Therefore, for cell (i, j) in the grid, the formula to compute the number of ways to get there would be:

```
dp[i][j] = dp[i-1][j] + dp[i][j-1]

```
With the base case being dp[0][j] = 1 for all j (since there's only one way to move right at the top row) and dp[i][0] = 1 for all i (since there's only one way to move down at the left column).

Let's implement the solution in Python such that it can be used directly in LeetCode:



```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # Create a 2D list (dp table) with 'n' columns and 'm' rows initialized to 1
        dp = [[1] * n for _ in range(m)]
        
        # Fill the dp table
        for i in range(1, m):
            for j in range(1, n):
                # The number of ways to get to cell (i, j) is the sum of the number
                # of ways to get to the cell directly above it (i-1, j) and the cell to the left of it (i, j-1)
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        
        # The bottom-right corner of the grid will give us the total number of unique paths
        return dp[-1][-1]

# Example usage:
# sol = Solution()
# print(sol.uniquePaths(3,7)) # Output: 28
# print(sol.uniquePaths(3,2)) # Output: 3
# print(sol.uniquePaths(7,3)) # Output: 28
# print(sol.uniquePaths(3,3)) # Output: 6

```

This code initializes a table `dp` with dimensions `m x n`, each element `dp[i][j]` represents the number of unique paths to reach cell `(i, j)` from the start `(0, 0)`. We populate this table based on our recurrence relation, and the answer to the problem will be found at `dp[m-1][n-1]`. This solution runs with a time complexity of O(m*n), which is efficient given the problem constraints.

# 63. Unique Paths II

### Problem Description 
A robot is located at the top-left corner of a `m x n` grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

Now consider if some obstacles are added to the grids. How many unique paths would there be?
An obstacle and space is marked as `1` and `0` respectively in the grid.


Example 1:
Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
Explanation: There is one obstacle in the middle of the 3x3 grid above.

There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right

Example 2:
Input: obstacleGrid = [[0,1],[0,0]]
Output: 1

Constraints:
`m == obstacleGrid.length`
`n == obstacleGrid[i].length`
`1 <= m, n <= 100`
`obstacleGrid[i][j]` is `0` or `1`.

### Solution 
 The problem is essentially about counting the number of unique paths in a grid with obstacles. We are to determine the number of ways the robot can travel from the top-left corner to the bottom-right corner, navigating only right or down, amidst some blocked cells noted as obstacles.

### Solution Approach:
We can adopt a dynamic programming approach for this problem. Here's the approach, step-by-step:

1. **Create a Table (DP Array):**
   - Define a 2D array `dp` of the same size as `obstacleGrid` where `dp[i][j]` will represent the number of ways to reach the cell `(i, j)`.

2. **Initialize the Starting Point:**
   - The robot starts at `obstacleGrid[0][0]`. If this cell is not an obstacle (`obstacleGrid[0][0] == 0`), set `dp[0][0] = 1`.

3. **Fill the DP Table:**
   - Loop through the grid cell by cell.
   - If a cell has an obstacle (`obstacleGrid[i][j] == 1`), set `dp[i][j] = 0` since no paths lead to or through an obstructed cell.
   - If it's not an obstacle, consider the number of ways to reach the current cell:
     - From the left (`dp[i][j-1]`) if `j > 0`.
     - From above (`dp[i-1][j]`) if `i > 0`.
   - Sum the contributions from the viable neighbors.

4. **Result Extraction:**
   - Given that traversal ends at the bottom-right corner, `dp[m-1][n-1]` will contain the number of unique paths from the start to the end.

This method ensures optimal use of space (only a 2D list related to the input size) and time (`O(m*n)` complexity due to a single scan of the grid).



```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        # If the starting or ending positions are obstacles, return 0 directly
        if obstacleGrid[0][0] == 1 or obstacleGrid[m-1][n-1] == 1:
            return 0
        
        # Create the DP table with all elements initialized to 0
        dp = [[0] * n for _ in range(m)]
        
        # Starting position
        dp[0][0] = 1
        
        # Fill the dp table
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0
                else:
                    if i > 0:
                        dp[i][j] += dp[i-1][j]
                    if j > 0:
                        dp[i][j] += dp[i][j-1]
        
        # The value in the bottom-right cell will denote the number of unique paths
        return dp[-1][-1]

# Example usage of the Solution class
# When submitted on LeetCode, the following lines would not be included:
if __name__ == "__main__":
    sol = Solution()
    print(sol.uniquePathsWithObstacles([[0,0,0],[0,1,0],[0,0,0]]))  # Output: 2
    print(sol.uniquePathsWithObstacles([[0,1],[0,0]]))  # Output: 1

```

This Python solution will work directly on LeetCode when wrapped within the provided `Solution` class. The `uniquePathsWithObstacles` function handles the dynamic programming logic to ensure all cases are covered based on the problem's constraints and requirements.

# 64. Minimum Path Sum

### Problem Description 
Given a `m x n` `grid` filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.


Example 1:
Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7
Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.


Example 2:
Input: grid = [[1,2,3],[4,5,6]]
Output: 12

Constraints:
`m == grid.length`
`n == grid[i].length`
`1 <= m, n <= 200`
`0 <= grid[i][j] <= 100`

### Solution 
 The problem at hand is to find the minimum path sum from the top-left corner to the bottom-right corner of a 2D grid, with only two possible moves at any step: moving right or moving down.

This is a classic dynamic programming problem that can be tackled by defining a state which represents the minimum path sum to reach each point in the grid. We establish a `dp` array where `dp[i][j]` represents the minimum path sum to reach the cell `(i, j)` from the top-left corner `(0, 0)`.

### Dynamic Programming Approach:

1. **Initialization**: 
   - Start by initializing the top-left corner of the dp table `(dp[0][0])` with the value at the top-left corner of the grid, i.e., `grid[0][0]`.
   
2. **Filling the First Row and Column**:
   - Since you can only move either right or down, the first row only has one possible path: keep moving right.
   - Similarly, the first column cells can only be reached by moving downward.
   - Populate these by accumulating values from their respective predecessors.
   
3. **Filling the Rest of the Grid**:
   - For each cell `(i, j)` in the grid, determine the minimum path sum to that cell by considering the minimum of the paths from the cell directly above `(i-1, j)` or directly to the left `(i, j-1)`, then add the value at `grid[i][j]`.

4. **Result**:
   - After filling the dp table, the value at `dp[m-1][n-1]` gives the minimum path sum from the top-left corner to the bottom-right.

Given these steps, we can proceed to write the code in the format suitable for pasting directly into LeetCode.



```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0]*n for _ in range(m)]
        
        # Initialize the starting point
        dp[0][0] = grid[0][0]
        
        # Fill the first column
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        
        # Fill the first row
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        
        # Fill the rest of the dp table
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        
        # The bottom-right corner has the result
        return dp[m-1][n-1]

```

This code correctly initializes a dp table based on the grid's dimensions and computes the minimum path sums for each cell, eventually returning the minimum path sum to the bottom-right corner of the grid. It is suitable for use within a coding challenge platform like LeetCode and should work efficiently within the given constraints.

# 66. Plus One

### Problem Description 
Given a non-empty array of decimal digits representing a non-negative integer, increment one to the integer.

The digits are stored such that the most significant digit is at the head of the list, and each element in the array contains a single digit.

You may assume the integer does not contain any leading zero, except the number 0 itself.


Example 1:
Input: digits = [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.


Example 2:
Input: digits = [4,3,2,1]
Output: [4,3,2,2]
Explanation: The array represents the integer 4321.


Example 3:
Input: digits = [0]
Output: [1]

Constraints:
`1 <= digits.length <= 100`
`0 <= digits[i] <= 9`

### Solution 
 Here's a Python solution that can be run directly on a platform like LeetCode. To solve this problem, the idea is to start from the rightmost digit in the list, increment it by 1, and handle any overflow by carrying the increment over to the next left digit.

We'll be explaining the implementation and taking care of edge cases like when all the digits are 9, causing overflow beyond the current array's size. Here's the Python code:



```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        # Start from the last digit in the list
        n = len(digits)
        for i in range(n - 1, -1, -1):
            # If the current digit is less than 9, simply increment it and return
            if digits[i] < 9:
                digits[i] += 1
                return digits
            # If the digit is 9, set it to 0 and continue to the next digit to the left
            digits[i] = 0
        # If all the digits were 9, we have moved through the entire list,
        # and we need an additional digit at the front now set to 1 (since 999 + 1 = 1000 for example)
        # where 1000 is one digit more than 999
        return [1] + digits  # prepend 1 to the list

# Example usage:
# sol = Solution()
# print(sol.plusOne([1,2,3]))  # Output: [1,2,4]
# print(sol.plusOne([4,3,2,9]))  # Output: [4,3,3,0]
# print(sol.plusOne([9,9,9]))  # Output: [1,0,0,0]

```

### Detailed Explanation:
- We iterate from the last element to the first element in the list `digits`.
- If the element is less than 9, it means we can simply increment this digit by 1 and return the list immediately, since no carry over is involved.
- If the element is 9, incrementing it would form a carry since 9 + 1 = 10. We need to set the current digit to 0 and proceed to the next digit to the left to add 1 to it.
- If the leftmost digit is incremented (which is 9 initially), and we've reached the beginning of the array having the whole array as 9's (like 999 or 9999), the final array should add an extra digit at the beginning, and we do so with the statement `[1] + digits`, which means the list shown `[1, 0, ...]` represents the lowest overflow carry.

This code handles the problem efficiently even with the maximum constraints given (array length up to 100), and it processes each element once, achieving an O(n) time complexity.

# 73. Set Matrix Zeroes

### Problem Description 
Given an `m x n` matrix. If an element is 0, set its entire row and column to 0. Do it in-place.

Follow up:
A straight forward solution using O(mn) space is probably a bad idea.

A simple improvement uses O(m + n) space, but still not the best solution.

Could you devise a constant space solution?

Example 1:
Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]

Example 2:
Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]

Constraints:
`m == matrix.length`
`n == matrix[0].length`
`1 <= m, n <= 200`
`-231 <= matrix[i][j] <= 231 - 1`

### Solution 
 To address the problem of setting entire rows and columns to zero in an `m x n` matrix when an element is zero, we need to keep track of which rows and columns should be modified without using additional space proportional to the size of the matrix.

The optimal approach utilizes the first row and the first column of the matrix itself to store information about which rows and columns should be set to zero. To ensure that we do not inadvertently modify the first row and the first column based on their initial contents, we use two additional variables to record whether the first row and the first column should be zeroed.

Here’s how you can structure this solution:
1. First, determine whether the first row and the first column should be zeroed using two boolean variables `first_row_zero` and `first_col_zero`.
2. Use the first row and the first column to record whether the remaining rows and columns should be zeroed by checking each element in the matrix.
3. Starting from the end of the matrix (to avoid overwriting the flags we just set in the first row and column), zero out elements based on the flags in the first row and the first column.
4. Finally, using the values of `first_row_zero` and `first_col_zero`, zero out the first row and first column if necessary.

Here’s the Python code integrated with the above logic:



```python
class Solution:
    def setZeroes(self, matrix):
        if not matrix or not matrix[0]:
            return
        
        m, n = len(matrix), len(matrix[0])
        first_row_zero = any(matrix[0][j] == 0 for j in range(n))
        first_col_zero = any(matrix[i][0] == 0 for i in range(m))
        
        # Use the first row and column to set the zero flags
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[0][j] = 0
                    matrix[i][0] = 0
        
        # Zero out cells based on flags in the first row and column
        for i in range(1, m):
            for j in range(1, n):
                if matrix[0][j] == 0 or matrix[i][0] == 0:
                    matrix[i][j] = 0
        
        # Zero out the first column if needed
        if first_col_zero:
            for i in range(m):
                matrix[i][0] = 0
        
        # Zero out the first row if needed
        if first_row_zero:
            for j in range(n):
                matrix[0][j] = 0
        
        # No need to return anything since we modify matrix in-place

# Example usage:
# sol = Solution()
# matrix = [[1,1,1],[1,0,1],[1,1,1]]
# sol.setZeroes(matrix)
# print(matrix)  # Output: [[1,0,1],[0,0,0],[1,0,1]]

```

This code effectively modifies the matrix in place with a space complexity of O(1), as the additional space required does not depend on `m` or `n`. This solution methodically ensures that each cell in the grid is checked, and any required modifications are made logically based on the initial pass's findings. This logic assists in preserving the first row and first column’s values until they are no longer needed, then selectively nullifying them if indicated.

# 74. Search a 2D Matrix

### Problem Description 
Write an efficient algorithm that searches for a value in an `m x n` matrix. This matrix has the following properties:
Integers in each row are sorted from left to right.

The first integer of each row is greater than the last integer of the previous row.


Example 1:
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true

Example 2:
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
Output: false

Constraints:
`m == matrix.length`
`n == matrix[i].length`
`1 <= m, n <= 100`
`-104 <= matrix[i][j], target <= 104`

### Solution 
 To address this problem, you can utilize a binary search due to the sorted nature of the matrix and its special properties. This solution avoids a full scan of every element, providing an efficient way to determine if the target value exists in the matrix. The key considerations include:

1. **Sorted Rows and Columns**: Each row is sorted, and each row’s first element is greater than the last element of the previous row. This allows us to view the matrix as a flattened sorted array.
2. **Binary Search Application**: Instead of using a binary search only within each row, apply binary search to the entire matrix by considering it as a one-dimensional array.

Here's how you can implement this:

**LeetCode Format Code**:


```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix:
            return False

        m, n = len(matrix), len(matrix[0])
        
        left, right = 0, m * n - 1

        while left <= right:
            mid = (left + right) // 2
            mid_value = matrix[mid // n][mid % n]  # Convert mid index to 2D indices

            if mid_value == target:
                return True
            elif mid_value < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return False

```

**Explanation**:
- `m` and `n` represent the number of rows and columns in the matrix, respectively.
- `left` and `right` are pointers for the binary search and they initially cover the entire range of the matrix when flattened into a line.
- The calculation `mid // n` and `mid % n` allows transformation of the one-dimensional index `mid` back into two dimensions suitable for indexing into the matrix.
- The binary search adjusts `left` and `right` based on the comparison between `mid_value` (the value at the position decoded from `mid`) and `target`.
- The loop continues until `left` surpasses `right` or until the target is found.

**Complexity**:
- **Time Complexity**: O(log(m*n)) because each step of the binary search halves the search space.
- **Space Complexity**: O(1) as no additional space proportional to the size of the input is used.

This approach provides an optimal solution with respect to time and space complexities, conforming to typical constraints found in coding interviews and competitive programming environments like LeetCode.

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
 The problem is essentially about sorting a special array containing only three different elements: 0, 1, and 2. This sorting task can be tackled efficiently by using a three-way partitioning technique known as the Dutch National Flag problem proposed by Edsger Dijkstra. 

The idea is to maintain three pointers or indices to place the 0s, 1s, and 2s in the right positions by a single pass through the array:

1. `low` pointer for the next position of 0
2. `mid` pointer for the next position of 1
3. `high` pointer for the next position of 2 from the end of array

We use the `mid` pointer to iterate through the array, and depending on the value, we either:
- Swap `nums[mid]` with `nums[low]` and increase both `mid` and `low`
- Leave `nums[mid]` as it is and just move `mid` forward
- Swap `nums[mid]` with `nums[high]` and decrease `high`

We'll complete the traversal when `mid` goes beyond the `high` index. This approach guarantees that the entire array is sorted in a single pass with O(1) extra space usage.

Here's the implementation in Python:



```python
from typing import List

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        low, mid, high = 0, 0, len(nums) - 1
        
        while mid <= high:
            if nums[mid] == 0:
                nums[low], nums[mid] = nums[mid], nums[low]
                low += 1
                mid += 1
            elif nums[mid] == 1:
                mid += 1
            else:
                nums[mid], nums[high] = nums[high], nums[mid]
                high -= 1

# Example usage:
sol = Solution()
arr1 = [2,0,2,1,1,0]
sol.sortColors(arr1)
print(arr1)  # Output: [0, 0, 1, 1, 2, 2]

arr2 = [2,0,1]
sol.sortColors(arr2)
print(arr2)  # Output: [0, 1, 2]

arr3 = [0]
sol.sortColors(arr3)
print(arr3)  # Output: [0]

arr4 = [1]
sol.sortColors(arr4)
print(arr4)  # Output: [1]

```

This code can be used directly on LeetCode to solve the problem and will execute in O(n) time, where n is the number of elements in the array, using constant O(1) space. This is because the sorting operation is conducted within the input array itself, avoiding the use of extra space for sorting apart from a few pointer variables.

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
 The problem is essentially asking for all subsets (also known as the power set) of a given list of unique integers. Each subset is a distinct combination of the elements from the original list.

In LeetCode, the Python solution should be wrapped in a class with the function defined as `def subsets(self, nums: List[int]) -> List[List[int]]:`.

The approach to solve this problem can vary, but a typical solution involves using backtracking (or recursive) techniques, where we recursively decide to either include or exclude the next element at each step. Another slightly more iterative approach would use bit manipulation to generate all possible subsets.

Here is a detailed explanation of a backtracking approach:
1. Initialize an empty list (`res`) to store the final subsets.
2. Define a helper function (`dfs`) that takes parameters such as the current index (`start`) and the current subset (`path`).
3. Iterate from the `start` index to the length of the nums array. For each index, include the number in the current subset (`path`) and recursively call `dfs` with the next index. After returning from the recursion, backtrack by removing the last element added (i.e., simulate not choosing the current element).
4. Always append a deep copy of `path` to `res` to save the current subset.

The base conditions are straightforward: If `start` is equal to or greater than the length of the input list (`nums`), just return without doing anything.

Here's the direct code solution in Python, adhering to LeetCode's required format:



```python
from typing import List

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        
        def dfs(start, path):
            # Add the current subset to the result list.
            res.append(path[:])
            for i in range(start, len(nums)):
                # Include the current element.
                path.append(nums[i])
                # Recursively proceed to construct further subsets.
                dfs(i + 1, path)
                # Exclude the current element (backtrack).
                path.pop()
        
        dfs(0, [])
        return res

# The following is optional for testing locally
# sol = Solution()
# print(sol.subsets([1, 2, 3]))
# print(sol.subsets([0]))

```

This solution explores every possibility of either including or excluding each element, thereby generating all possible subsets of the input list. Given the constraints with `nums.length` up to 10, this will efficiently handle all inputs within these bounds.

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
 To tackle this problem, we will utilize Depth First Search (DFS) with backtracking. The idea is to explore each cell in the grid, starting a DFS whenever the character in the cell matches the first character of the word. The DFS will continue by checking horizontally or vertically adjacent cells. If any sequence of characters matches the word in the grid during the search, we return `true`. Otherwise, we will backtrack and restore the cell's original state.

For optimizations, particularly considering the follow-up:
1. **Early Termination**: We can stop the search as soon as the word is found.
2. **Search Pruning via Immediate Checks**: In each new DFS call:
   - If the character at the current position doesn't match, terminate the route.
   - If out of boundary conditions are detected, terminate the route.
   - Reuse letter prevention is built into this check.
   
Let's now turn to the LeetCode standardized solution format:



```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m, n = len(board), len(board[0])
        
        # Helper function for DFS
        def dfs(x, y, index):
            # If index matches length of word, we have found the whole word
            if index == len(word):
                return True
            
            # Out of bounds or character doesn't match or we revisited a cell
            if x < 0 or x >= m or y < 0 or y >= n or board[x][y] != word[index]:
                return False
            
            # Save the board character at (x, y) since we need to alter it
            temp = board[x][y]
            board[x][y] = "#"
            
            # Move in the 4 possible directions: up, down, left, right
            found = dfs(x + 1, y, index + 1) or \
                    dfs(x - 1, y, index + 1) or \
                    dfs(x, y + 1, index + 1) or \
                    dfs(x, y - 1, index + 1)
            
            # Restore the original character at the board position
            board[x][y] = temp
            
            return found
        
        # Start DFS from each cell that matches the first character of the word
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:  # Only initiate DFS if first character matches
                    if dfs(i, j, 0):
                        return True
        
        return False

```

In this code:
- We traverse all cells in the grid. Whenever we find a cell with a character matching the first character of the word, we start a DFS from that cell.
- The `dfs` function attempts to find the word starting from the given cell. If out of bounds or other constraints are not met, the DFS stops for that route.
- The board character is temporarily changed to a sentinel value (e.g., `#`) to prevent reuse within the same search path. After the path is checked, it's restored.
- The algorithm will efficiently halt as soon as the word is found, reducing unnecessary computations.

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
 To tackle this problem, we utilize the concept of a two-pointer technique. The main idea is to have one pointer (`write_index`) to determine the position to write the next value ensuring no more than two duplicates, and another pointer (`read_index`) to read through the list. We also maintain a count to check how many times the current number has been met consecutively.

We can achieve the desired behavior by:

1. Initializing `write_index` to start from the beginning (index 0).
2. Using a loop with `read_index` to traverse through the list:
    - If the current number (`nums[read_index]`) is the same as the last written number (`nums[write_index - 1]`), we increase a counter.
    - If the count is less than or equal to 2, we copy the number from `read_index` to `write_index` and then increment `write_index`.
    - If the current number is not the same as the last written, reset the count and proceed to copy as usual.
3. Continue until `read_index` has scanned through the entire list.
4. The new length of the modified list can be identified by the value of `write_index`.

The solution follows your requirements of O(1) extra space and modifies the list in place.

Here's the Python code properly formatted for submission on LeetCode:



```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        write_index = 1  # Start writing from the second element
        count = 1        # Already encountered one of nums[0]
        
        for read_index in range(1, len(nums)):
            if nums[read_index] == nums[read_index - 1]:
                count += 1
            else:
                count = 1
            
            if count <= 2:
                nums[write_index] = nums[read_index]
                write_index += 1
        
        return write_index

# Example usage:
# sol = Solution()
# print(sol.removeDuplicates([1,1,1,2,2,3]))  # Output: 5
# print(sol.removeDuplicates([0,0,1,1,1,1,2,3,3]))  # Output: 7

```

This code segment defines a Solution class with a method `removeDuplicates`, which conforms to typical LeetCode format and can be run directly on the LeetCode platform. The method definition matches what is expected by the platform with `nums` being a list of integers and it returning an integer indicating the new length of the list after duplications have been handled appropriately. Each unique element in `nums` is allowed to appear at most twice.

# 81. Search in Rotated Sorted Array II

### Problem Description 
There is an integer array `nums` sorted in non-decreasing order (not necessarily with distinct values).

Before being passed to your function, `nums` is rotated at an unknown pivot index `k` (`0 <= k < nums.length`) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (0-indexed). For example, `[0,1,2,4,4,4,5,6,6,7]` might be rotated at pivot index `5` and become `[4,5,6,6,7,0,1,2,4,4]`.

Given the array `nums` after the rotation and an integer `target`, return `true` if `target` is in `nums`, or `false` if it is not in `nums`.


Example 1:
Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true

Example 2:
Input: nums = [2,5,6,0,0,1,2], target = 3
Output: false

Constraints:
`1 <= nums.length <= 5000`
`-104 <= nums[i] <= 104`
`nums` is guaranteed to be rotated at some pivot.

`-104 <= target <= 104`
Follow up: This problem is the same as Search in Rotated Sorted Array, where `nums` may contain duplicates. Would this affect the runtime complexity? How and why?

### Solution 
 Here is the Python code that will solve this problem, followed by a detailed explanation in the LeetCode problem solution format:



```python
from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        l, r = 0, len(nums) - 1
        
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return True
            
            # Eliminate duplicates which could affect the binary search
            while l < mid and nums[l] == nums[mid]:
                l += 1
                
            # Determine the sorted part of the array
            if nums[l] <= nums[mid]:
                # Left side is sorted
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                # Right side is sorted
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
        return False

# Example usage:
sol = Solution()
print(sol.search([2,5,6,0,0,1,2], 0)) # Output: True
print(sol.search([2,5,6,0,0,1,2], 3)) # Output: False

```

### Explanation

**Approach: Modified Binary Search with Duplicates Handling**

Given the array is sorted but then rotated, and can also have duplicates, the typical binary search approach must be modified slightly. Duplicates in the array present a challenge because they can obscure whether the left or right half of the array is sorted.

**Algorithm Steps:**
1. **Handle Duplicates:** Skip duplicates to correctly identify whether the left or right segment of the array is properly sorted. This skipping happens at the beginning of each loop iteration.
2. **Identify Sorted Section:** Check if the left part (from `nums[l]` to `nums[mid]`) is sorted.
   * If this section is sorted and `target` falls within this range, search in this section by adjusting `r`.
   * Otherwise, search in the right section by adjusting `l`.
3. **Check Other Half:** If the left part isn't sorted, then the right part must be sorted. Check if `target` falls within the right section and adjust pointers (`l`, `r`) accordingly.
4. **Repeat** until the pointers `l` and `r` converge.

To handle duplicates effectively, the algorithm uses a linear search (`while l < mid and nums[l] == nums[mid]: l += 1`) at each midpoint evaluation to ensure that the linear segment doesn't mislead the identification of the sorted section.

### Complexity and Follow-Up

**Time Complexity:** The worst-case time complexity can degrade to O(n), because in cases where there are many duplicates, the linear part of the search (`while l < mid and nums[l] == nums[mid]`) can take linear time.

**Space Complexity:** O(1) since it uses a constant amount of space.

**Follow-Up Discussion:** The presence of duplicates is the main reason why the worst time complexity could degrade to O(n). This contrasts with a binary search in a rotated sorted array without duplicates, which can consistently operate in O(log n) time. In the former case, duplicates can prevent quickly determining which half of the array is sorted, often necessitating a scan through segments of the array to resolve ambiguity, leading to a linear scan in the worst case.

# 84. Largest Rectangle in Histogram

### Problem Description 
Given an array of integers `heights` representing the histogram's bar height where the width of each bar is `1`, return the area of the largest rectangle in the histogram.


Example 1:
Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: The above is a histogram where width of each bar is 1.

The largest rectangle is shown in the red area, which has an area = 10 units.


Example 2:
Input: heights = [2,4]
Output: 4

Constraints:
`1 <= heights.length <= 105`
`0 <= heights[i] <= 104`

### Solution 
 To solve this problem, we aim to calculate the largest rectangular area possible in a given histogram where each bar's height is specified by the `heights` array, and each bar's width is 1.

The problem can be efficiently addressed using a stack to keep track of the bars' heights. A key observation is that when encountering heights that suddenly drop, it suggests the end of potentially maximal rectangles, particularly for earlier taller bars. Therefore, we leverage a stack to keep indices of the bars so that while moving through the `heights`, we can quickly determine when we've concluded a rectangle of maximal extent for a particular beginning bar height.

The algorithm can be summarized as follows:
1. Initialize an empty stack which will store indices of the `heights` array.
2. Iterate through each bar in the histogram:
   - When the current bar is higher than the bar of the index at the top of the stack (or if the stack is empty), push its index onto the stack.
   - If the current bar is lower than that at the top of the stack, pop the stack and calculate the area considering the bar at the popped index as the shortest bar:
     - Calculate the area with `min_height` as the height of the popped bar and width determined by the current index and the new index at the top of the stack after popping.
3. After processing all bars, explicitly empty the stack by accounting for the remaining indices in the stack as potential starting points for bars stretching to the end of the histogram.
4. Return the maximum area computed.

Let's implement this in Python, as would be suitable for a direct submission to LeetCode:



```python
class Solution:
    def largestRectangleArea(self, heights):
        # Stack to keep track of indices of the histogram bars
        stack = []
        max_area = 0
        index = 0
        
        while index < len(heights):
            # If this bar is higher than the bar at the stack top, push it to the stack
            if not stack or heights[index] >= heights[stack[-1]]:
                stack.append(index)
                index += 1
            else:
                # Calculate the area with the height of the bar at stack top
                top_of_stack = stack.pop()
                # Decide the width based on the presence of elements in the stack
                width = index if not stack else index - stack[-1] - 1
                # Compute area and update max_area
                max_area = max(max_area, heights[top_of_stack] * width)
        
        # Finally, clear the stack and use all remaining bars as the height of possible rectangles
        while stack:
            top_of_stack = stack.pop()
            width = len(heights) if not stack else len(heights) - stack[-1] - 1
            max_area = max(max_area, heights[top_of_stack] * width)
        
        return max_area

```

This function performs efficiently by maintaining a stack to keep track of bars and computing areas only when necessary (i.e., when encountering a smaller bar). The method ensures that each bar is pushed and popped from the stack exactly once, leading to a time complexity of O(n), where `n` is the number of bars. Memory complexity is also O(n) due to the stack usage, accommodating a worst-case scenario where all bar indices might be pushed onto the stack.

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
 To solve the problem of finding the largest rectangle containing only '1's in a given binary matrix, we can use an approach based on the "Largest Rectangle in Histogram" problem. Below is a step-by-step Python solution following this strategy:

1. **Preparation**: If the input matrix is empty, return 0 as the area.
2. **Height Calculation**: Use a height array `heights` where each position `j` corresponds to the number of continuous '1's up to the current row in column `j`.
3. **Row-by-Row Processing**: Treat each row of the matrix as the base of the histogram, where the heights of the bars in the histogram are determined by the values in `heights`.
4. **Largest Rectangle in Histogram Calculation**: For each row seen as the base, use a stack-based approach to find the largest rectangle in the histogram formed by `heights`.
5. **Max Area Tracking**: For each row calculation, update the maximum area found.

We will implement the solution in the expected format for LeetCode, where the function `maximalRectangle` takes the `matrix` as an argument and returns the area of the largest rectangle of '1's.

Here is the complete solution:



```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        
        max_area = 0
        cols = len(matrix[0])
        rows = len(matrix)
        heights = [0] * (cols + 1)  # One extra for the easier calculation
        
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == '1':
                    heights[j] = heights[j] + 1
                else:
                    heights[j] = 0
            
            # Calculate the largest rectangle area using the histogram technique
            stack = []
            index = 0
            while index < len(heights):
                if not stack or heights[index] >= heights[stack[-1]]:
                    stack.append(index)
                    index += 1
                else:
                    top_of_stack = stack.pop()
                    height = heights[top_of_stack]
                    width = index if not stack else index - stack[-1] - 1
                    max_area = max(max_area, height * width)
            heights.pop()  # Reset extra element added
            
        return max_area


```

**Explanation**:
- We augment the matrix with an additional row of zeros (`heights + [0]`) for easier calculations.
- We maintain a `heights` array that is updated as we traverse the matrix row-by-row.
- The largest area in each histogram (for each row considered) is determined using a stack-based method, where the stack holds the indices of the columns sorted by the height of the bars in the histogram.
- The maximum area found (`max_area`) across all histograms is the final result.

This is an efficient solution that only examines each row and column a bounded number of times, leading to an average complexity of around O(rows * cols).

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
 To solve this problem, we will perform the merger in-place in the `nums1` array starting from the end, to ensure that we don't overwrite the values in `nums1` that still need to be examined. We are given that `nums1` has enough space to accommodate all the elements (from both `nums1` and `nums2`), making the in-place merge approach feasible.

**Algorithm Steps:**
1. Use three pointers:
   - `p1` points to the end boundary of the last valid element in `nums1` (i.e., index `m-1`).
   - `p2` points to the end of `nums2` (i.e., index `n-1`).
   - `i` points to the end of the space allocated in `nums1` (i.e., index `m+n-1`).
2. Compare the elements pointed by `p1` and `p2`. Place the larger element at the position pointed by `i` and decrement the respective pointer (`p1` or `p2`) along with `i`.
3. Once one of the arrays is exhausted, if there are remaining elements in `nums2` that haven't been considered (since `nums1`'s original elements are already in place if they weren't moved), copy them to the beginning of `nums1`.
4. No actions needed if any elements remain in `nums1`, as they are already in the correct position.

This direct approach works given the constraints and in-place merge requirement.


Here is the Python code implementing the above algorithm:



```python
from typing import List

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p1, p2, i = m - 1, n - 1, m + n - 1
        
        # Start from the end of both arrays and compare to fill the array from the back
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] > nums2[p2]:
                nums1[i] = nums1[p1]
                p1 -= 1
            else:
                nums1[i] = nums2[p2]
                p2 -= 1
            i -= 1
        
        # If there are remaining elements in nums2, copy them
        while p2 >= 0:
            nums1[i] = nums2[p2]
            p2 -= 1
            i -= 1

# Example usage:
# sol = Solution()
# nums1 = [1,2,3,0,0,0]
# m = 3
# nums2 = [2,5,6]
# n = 3
# sol.merge(nums1, m, nums2, n)
# print(nums1)  # Output: [1,2,2,3,5,6]

```

From the explanation and the provided code, this solution efficiently merges the two arrays in-place by utilizing the extra buffer space given at the end of `nums1`, maintaining an O(m+n) time complexity which is optimal given the constraints.

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
 To address the problem of generating all unique subsets for an array that includes duplicates using Python, we will utilize the following approach with a clear solution format applicable for platforms like LeetCode:

**Approach**: Use backtracking to generate potential subsets and ensure uniqueness by leveraging set properties to manage duplicate results.

1. **Sorting**: First, sort the array. This way, duplicates come one after another which helps in managing dupes when generating subsets.
2. **Backtracking**: Use a recursive function to explore possible subsets. Utilize a current list to build subsets and add them to the results if they don't already exist.
3. **Duplicate Management**: By iterating through the numbers, if a number is the same as its predecessor and the predecessor was not included in a subset, skip this number to avoid duplicate subsets.

This method ensures that we explore all combinations systematically and only generate unique sets due to both sorting and conditional inclusion.

Here is the Python code adhering to the given constraints and employing the previously described strategy:



```python
class Solution:
    def subsetsWithDup(self, nums):
        def backtrack(start, path):
            res.append(path)
            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i - 1]:
                    continue
                backtrack(i + 1, path + [nums[i]])
        
        nums.sort()
        res = []
        backtrack(0, [])
        return res

# This class and method can be tested with sample inputs directly in LeetCode

```

**Explanation**:
- **`backtrack()`**: A recursive function responsible for generating subsets. It uses `start` to manage which part of the array it's considering to prevent regenerating previously seen combinations. `path` is a list that accumulates the current combination of numbers.
- **`res`**: This is the result list that accumulates all the unique subsets.
- During each iteration in `backtrack()`, if the current number has the same value as one immediately before it and we aren't considering the start of a set (`i > start`), the recursive call is skipped to avoid duplicates. This skipping is vital only when repeated numbers would result in an identical subset being added to `res`.
- Upon calling `backtrack(0, [])`, the function begins execution and ultimately all variants of paths are added to `res`.

This code functionality is ensured by the constraints set by the problem (`1 <= nums.length <= 10` and `-10 <= nums[i] <= 10`), making it efficient due to limited size and practical through simple iterations and recursion. Hence, this code is efficient for submission on competitive coding platforms.

# 105. Construct Binary Tree from Preorder and Inorder Traversal

### Problem Description 
Given two integer arrays `preorder` and `inorder` where `preorder` is the preorder traversal of a binary tree and `inorder` is the inorder traversal of the same tree, construct and return the binary tree.


Example 1:
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]

Example 2:
Input: preorder = [-1], inorder = [-1]
Output: [-1]

Constraints:
`1 <= preorder.length <= 3000`
`inorder.length == preorder.length`
`-3000 <= preorder[i], inorder[i] <= 3000`
`preorder` and `inorder` consist of unique values.

Each value of `inorder` also appears in `preorder`.

`preorder` is guaranteed to be the preorder traversal of the tree.

`inorder` is guaranteed to be the inorder traversal of the tree.

### Solution 
 To solve the problem of constructing a binary tree from preorder and inorder traversal results, we need to first understand the properties of these traversals:

1. **Preorder Traversal**: This is a type of depth-first traversal where the node is visited first followed by its left subtree and then its right subtree. Thus, the first element in the `preorder` list is always the root of the (sub)tree.

2. **Inorder Traversal**: In this traversal, the nodes are visited in a left-root-right order. This means that for any node, all the elements in its left subtree appear before it in the `inorder` list, and all the elements in its right subtree appear after it.

### Solution Strategy:

To reconstruct the tree, we can leverage the properties of these traversals:
- Use the first element in `preorder` as the current root.
- Find this root in `inorder`, which will then partition the `inorder` list into elements that are in the left and right subtrees of this root.
- Recursively apply the above process for the left and right subtree portions of the `preorder` and `inorder` lists.

### Detailed Steps:

1. Start with the first element in `preorder`, which is the root of the tree.
2. Find this root element’s index in `inorder`. This index will help identify the boundaries for the left and right subtrees:
   - Elements to the left of this index in `inorder` are in the left subtree.
   - Elements to the right of this index in `inorder` are in the right subtree.
3. Recursively repeat for the elements of the left and right subtree in both `preorder` and `inorder`.

### Efficiency:
Utilizing a hashmap to quickly find indices in the `inorder` list can significantly improve the efficiency, reducing the need to search for an index linearly.

Here's the Python code implementing the solution:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder or not inorder:
            return None
        
        # Map from value to its index in inorder for O(1) access
        index_map = {value: idx for idx, value in enumerate(inorder)}
        
        # Recursive helper function to construct the tree
        def helper(pre_left, pre_right, in_left, in_right):
            if pre_left > pre_right:
                return None
            
            # Root is the first element in the preorder segment
            root_val = preorder[pre_left]
            root = TreeNode(root_val)
            
            # Find root in inorder segment
            in_root_index = index_map[root_val]
            
            # Calculate sizes of left and right subtrees
            left_size = in_root_index - in_left  # Number of nodes in left subtree
            
            # Construct left and right subtree recursively
            root.left = helper(pre_left + 1, pre_left + left_size, in_left, in_root_index - 1)
            root.right = helper(pre_left + left_size + 1, pre_right, in_root_index + 1, in_right)
            
            return root

        # Initial call from preorder[0] to preorder[n-1]
        return helper(0, len(preorder) - 1, 0, len(inorder) - 1)

```

This code introduces a `TreeNode` class as a basic structure for the tree nodes. The recursive helper function `helper` constructs the tree by following the strategy outlined above. The function is initially called with bounds covering the entire `preorder` and `inorder` lists.

When running on LeetCode, this code should directly fit into the expected solution format and efficiently construct the tree based on the traversal lists.

# 106. Construct Binary Tree from Inorder and Postorder Traversal

### Problem Description 
Given two integer arrays `inorder` and `postorder` where `inorder` is the inorder traversal of a binary tree and `postorder` is the postorder traversal of the same tree, construct and return the binary tree.


Example 1:
Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
Output: [3,9,20,null,null,15,7]

Example 2:
Input: inorder = [-1], postorder = [-1]
Output: [-1]

Constraints:
`1 <= inorder.length <= 3000`
`postorder.length == inorder.length`
`-3000 <= inorder[i], postorder[i] <= 3000`
`inorder` and `postorder` consist of unique values.

Each value of `postorder` also appears in `inorder`.

`inorder` is guaranteed to be the inorder traversal of the tree.

`postorder` is guaranteed to be the postorder traversal of the tree.

### Solution 
 To solve the problem of constructing a binary tree from the given `inorder` and `postorder` traversal arrays, we need to understand the properties of these traversals:

1. **Inorder Traversal (Left, Node, Right)**: This sequence means that the node is visited after all the nodes in its left subtree and before all the nodes in its right subtree.
2. **Postorder Traversal (Left, Right, Node)**: In this sequence, the node is visited after both its left and right subtrees are processed, which places the root node of the entire tree as the last element in this traversal.

Given these properties, we can devise an approach:

- The last element in the `postorder` array gives us the root of the current tree (or sub-tree).
- We can find this root element in the `inorder` array. Everything to the left of this root element in the `inorder` array will form the left subtree, and everything to the right will form the right subtree.
- We use recursion to repeat this process: constructing the left subtree using the left segment of `inorder` and corresponding `postorder` slices, and similarly for the right subtree.

For efficiency, we may use a hashmap (or dictionary in Python) to record the indices of elements in the `inorder` array to achieve O(1) lookup times, reducing the overall time complexity from potentially O(n^2) to O(n), where n is the number of nodes in the tree.

Here is how you can implement this:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        if not inorder or not postorder:
            return None
        
        # Map element to its index in inorder list for quick access
        in_map = {value: idx for idx, value in enumerate(inorder)}
        
        def buildTreeHelper(in_left, in_right):
            if in_left > in_right:
                return None
            
            # Last element in postorder list is the root
            root_val = postorder.pop()
            root = TreeNode(root_val)
            
            # Root splits inorder list into left and right subtrees
            index = in_map[root_val]
            
            # Recursively build the right subtree
            root.right = buildTreeHelper(index + 1, in_right)
            # Recursively build the left subtree
            root.left = buildTreeHelper(in_left, index - 1)
            
            return root
        
        return buildTreeHelper(0, len(inorder) - 1)


```
**Explanation:**

- We define a recursive method `buildTreeHelper` that takes the current range of the `inorder` array that it needs to work with (from `in_left` to `in_right`).
- The root of the current subtree is obtained from the last element of the `postorder` list (this list is accessed from the outer scope, which is updated as elements are popped).
- Using the index of the root element found in the `in_map`, we recursively create the right subtree first (important due to the nature of postorder processing) and then the left subtree.
- The `buildTreeHelper` method is initially called with the full range of indices from the `inorder` list.

This solution respects the constraints and should perform efficiently within the given problem limits on LeetCode.

# 118. Pascal's Triangle

### Problem Description 
Given an integer `numRows`, return the first numRows of Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:

Example 1:
Input: numRows = 5
Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]

Example 2:
Input: numRows = 1
Output: [[1]]

Constraints:
`1 <= numRows <= 30`

### Solution 
 

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        # Early return if the input numRows is 1
        if numRows == 1:
            return [[1]]
        
        # Initialize the Pascal's triangle with the first row
        pascal_triangle = [[1]]
        
        # Start creating rows from the second row up to numRows
        for i in range(1, numRows):
            # The first element of each row is always 1
            row = [1]
            
            # Get the last row from the triangle to compute the current row
            last_row = pascal_triangle[i - 1]
            
            # Compute the values between the first and last element of the current row
            for j in range(1, i):
                value = last_row[j - 1] + last_row[j]
                row.append(value)
            
            # The last element of each row is also 1
            row.append(1)
            
            # Append the current row to the Pascal's triangle
            pascal_triangle.append(row)
        
        return pascal_triangle

# This Python class defines a function, generate, that constructs the Pascal’s triangle given the number of rows.
# The solution initializes with the first row (base case) and builds each subsequent row based on the previous one 
# until the desired number of rows is achieved.

```

### Detailed Explanation

1. **Edge Case Handling**: The function starts by handling the edge case where `numRows` is 1. Since Pascal's triangle's first row is always `[1]`, it directly returns this value in a list of lists.

2. **Initialization**: The Pascal's triangle is initialized with the first row. This uses a list to keep track of lists representing each row.

3. **Building Rows**: Then, the function iteratively builds rows from the second up to the `numRows`-th row. 
   - Each row always starts with a `1`.
   - The elements in between the ends of the rows are generated by summing the appropriate elements (directly above and to the left and right) from the previous row.
   - The final element in the row is also a `1`.
   - After constructing the current row, it is appended to the Pascal's triangle.

4. **Output**: After constructing all rows, the complete Pascal’s triangle (up to `numRows`) is returned.

This method ensures that all rows are generated following the properties of Pascal's triangle. It is efficient and straightforward, focusing on constructing each row based on the properties of its preceding row.

# 119. Pascal's Triangle II

### Problem Description 
Given an integer `rowIndex`, return the `rowIndexth` (0-indexed) row of the Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:

Example 1:
Input: rowIndex = 3
Output: [1,3,3,1]

Example 2:
Input: rowIndex = 0
Output: [1]

Example 3:
Input: rowIndex = 1
Output: [1,1]

Constraints:
`0 <= rowIndex <= 33`
Follow up: Could you optimize your algorithm to use only `O(rowIndex)` extra space?

### Solution 
 The solution below involves calculating the specific row of the Pascal's triangle using an approach that optimizes for space. Given the constraint where we only need O(rowIndex) in extra space, we can maintain a single list that gets updated for every row until we reach the desired `rowIndex`. This is possible because each element in the row only depends on two numbers (which we can easily access from our single list) from the row above it.

Here is how you can solve this problem, and the Python code is provided following the explanation:

1. Initialize a list with a single element `[1]`, which represents the 0th row of the Pascal's triangle.
2. For generating the next rows up to the `rowIndex`, update the list in-place from right to left to prevent overwriting values that are still needed.
3. For each new row, the rightmost element is always a `[1]`. Since each element in a row is the sum of the element just above it and the element to the left of the above element, update the elements from the right to the left based on this rule.
4. Specifically, starting from the next-to-last element going to the beginning of the list, update each element to be the sum of its current value and the value directly left of it.
5. After iterating to the desired row, return the resultant list.

Here's the Python code:



```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        # Start with the first row of Pascal's Triangle
        row = [1]
        
        # Compute each row from the 1st to `rowIndex`
        for i in range(1, rowIndex + 1):
            # Update each element in the row from the last to the second
            for j in range(i - 1, 0, -1):
                row[j] += row[j - 1]
            # The first element of each row is always 1, so we just append a new 1 each iteration
            row.append(1)
        
        return row

```

In the above code:
 - We start with the `row` initialized as `[1]`.
 - For each subsequent row, we use a loop that iterates backward, ensuring that we update each element using the previously calculated values without affecting the next calculation.
 - After updating all necessary elements, we end each row's calculation by appending `1` to the list, maintaining the structure of Pascal's Triangle.

This approach provides an efficient O(rowIndex) space complexity and is suitable for directly using it in LeetCode submissions. The algorithm should perform well within the constraints given (0 <= rowIndex <= 33).

# 120. Triangle

### Problem Description 
Given a `triangle` array, return the minimum path sum from top to bottom.

For each step, you may move to an adjacent number of the row below. More formally, if you are on index `i` on the current row, you may move to either index `i` or index `i + 1` on the next row.


Example 1:
Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
Output: 11
Explanation: The triangle looks like:
   2
  3 4
 6 5 7
4 1 8 3
The minimum path sum from top to bottom is 2 + 3 + 5 + 1 = 11 (underlined above).


Example 2:
Input: triangle = [[-10]]
Output: -10

Constraints:
`1 <= triangle.length <= 200`
`triangle[0].length == 1`
`triangle[i].length == triangle[i - 1].length + 1`
`-104 <= triangle[i][j] <= 104`
Follow up: Could you do this using only `O(n)` extra space, where `n` is the total number of rows in the triangle?

### Solution 
 Here is the Python solution for the given problem, designed to run directly on LeetCode:



```python
from typing import List

class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # Get the number of rows in the triangle.
        n = len(triangle)
        # Start at the second last row and move upwards.
        for i in range(n - 2, -1, -1):
            for j in range(i + 1):
                # Modify the current cell with the minimum path sum from current cell to bottom.
                triangle[i][j] += min(triangle[i + 1][j], triangle[i + 1][j + 1])
        # The top element of the triangle will contain the minimum path sum.
        return triangle[0][0]

```

### Explanation:
The solution adopts a dynamic programming approach, but instead of requiring extra space, it updates the `triangle` array in place.

1. **Bottom-Up Approach**: We start from the second to last row of the triangle and work our way up to the top. This allows us to determine the minimum path to the bottom from each element.

2. **In-Place Modification**: For each element in the current row, update its value to the sum of the element itself and the minimum of the two possible elements it can move to in the row below. This process updates each element to represent the minimum path sum from that element downwards.

3. **Termination**: When we reach the top of the triangle (the first element), it will contain the minimum path sum from the top to the bottom.

4. **Time Complexity**: Each element is visited once, so the time complexity is \(O(n^2)\), where \(n\) is the number of rows in the triangle.
  
5. **Space Complexity**: The space usage is \(O(1)\) beyond the input space since we're modifying the triangle in place.

This method efficiently computes the result with optimal space utilization, adhering to the follow-up constraint of using only \(O(n)\) space where \(n\) is the number of rows, specifically by using no extra space beyond what's given.

# 121. Best Time to Buy and Sell Stock

### Problem Description 
You are given an array `prices` where `prices[i]` is the price of a given stock on the `ith` day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return `0`.


Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.

Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.


Example 2:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.


Constraints:
`1 <= prices.length <= 105`
`0 <= prices[i] <= 104`

### Solution 
 Here is a Python solution that addresses the stock profit maximization problem stated above. The solution provided uses an efficient approach to maximize the profit from buying and selling the stock.



```python
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        
        min_price = float('inf')
        max_profit = 0
        
        for price in prices:
            if price < min_price:
                min_price = price # update the minimum price found so far
            elif price - min_price > max_profit:
                max_profit = price - min_price # update the maximum profit found so far
        
        return max_profit

# Example Usage
sol = Solution()
print(sol.maxProfit([7,1,5,3,6,4]))  # Output: 5
print(sol.maxProfit([7,6,4,3,1]))    # Output: 0

```

### Explanation:

The solution uses a single-pass approach through the given list of prices to determine the maximum possible profit. Here are the clear steps it follows:

1. **Initialize Variables**:
   - `min_price` is initialized to a very large number (`float('inf')`) to ensure that any price in the array will be smaller.
   - `max_profit` is initialized to 0, representing no profit initially.

2. **Iterate Through the Prices**:
   - For each `price` in the `prices` list:
     1. If the `price` is less than `min_price`, we update `min_price` to this value. This represents the lowest price seen so far, which is the ideal buying point.
     2. Otherwise (if the price is higher than `min_price`), we compute the potential profit by subtracting `min_price` from the `price`. If this potential profit is greater than `max_profit`, we update `max_profit`.

3. **Result**:
   - After iterating through all the prices, `max_profit` holds the maximum profit that can be achieved based on the entire set of prices.

This solution works with a time complexity of \(O(n)\), where \(n\) is the number of days (length of the `prices` list), because it only requires one pass through the list. The space complexity is \(O(1)\), using only a few extra variables, making it highly space-efficient.

# 122. Best Time to Buy and Sell Stock II

### Problem Description 
You are given an array `prices` where `prices[i]` is the price of a given stock on the `ith` day.

Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).


Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.

Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.


Example 2:
Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.

Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. You must sell before buying again.


Example 3:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e., max profit = 0.


Constraints:
`1 <= prices.length <= 3 * 104`
`0 <= prices[i] <= 104`

### Solution 
 To solve this problem, we can focus on identifying all increasing pairs of consecutive days in the price list. If a stock price on day `i` is less than its price on day `i+1`, this provides an opportunity to buy on day `i` and sell on day `i+1` for a profit. This approach will capture the profit from all local increases in the stock price, which cumulatively provides the maximum profit.

Each time we find such a pair, we can add the difference (profit from buying on day `i` and selling on day `i+1`) to our total profit. This method ensures that we "skip" unnecessary days where buying wouldn't be profitable and continuously accumulate profits from profitable days.

Here's the step-by-step Python code following this logic, which is suitable for submission on LeetCode.



```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        total_profit = 0
        # Loop through each day, except the last day since you can't sell after the last day
        for i in range(len(prices) - 1):
            # Detect an opportunity for a profitable "buy on day i, sell on day i+1"
            if prices[i] < prices[i + 1]:
                # Accumulate the profit
                total_profit += prices[i + 1] - prices[i]
        return total_profit

```

**Explanation of the Code:**
- **Function Definition:** `maxProfit(self, prices: List[int]) -> int` - This function calculates the maximum profit given a list of stock `prices`.
- **Variable Initialization**: `total_profit = 0` - Stores the running total of profit from all transactions.
- **Loop through the List**: We loop through the list up to the second last element (`range(len(prices) - 1)`) to compare current and next day prices.
- **Condition**: If the price on the current day (`prices[i]`) is less than on the following day (`prices[i+1]`), it implies a profit opportunity by buying today and selling tomorrow.
- **Profit Accumulation**: For each identified profitable opportunity, the difference (`prices[i+1] - prices[i]`, which is the profit) is added to `total_profit`.
- **Return Value**: After processing all elements, `total_profit` is returned, representing the maximum total profit.

This solution efficiently covers the requirement without having to simulate complex scenarios of buying and selling across non-consecutive days, and handles scenarios where no profit is possible. It respects the constraints by ensuring that the solution works within a linear time complexity, `O(n)`, where `n` is the number of days (length of the `prices` array).

# 123. Best Time to Buy and Sell Stock III

### Problem Description 
You are given an array `prices` where `prices[i]` is the price of a given stock on the `ith` day.

Find the maximum profit you can achieve. You may complete at most two transactions.

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).


Example 1:
Input: prices = [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.

Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.


Example 2:
Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.

Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. You must sell before buying again.


Example 3:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.


Example 4:
Input: prices = [1]
Output: 0

Constraints:
`1 <= prices.length <= 105`
`0 <= prices[i] <= 105`

### Solution 
 Here's a LeetCode solution format for solving the problem of finding maximum profit from at most two transactions on given stock prices:



```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0

        n = len(prices)
        max_profit = 0
        
        # First and second buy/sell profits
        first_buy = second_buy = float('inf')
        first_profit = second_profit = 0
        
        for price in prices:
            # We minimize the price at which we could've bought the stock
            first_buy = min(first_buy, price)
            # Calculate what the max profit would be if we sold at current price
            first_profit = max(first_profit, price - first_buy)
            
            # For the second buy we account the profit from the first one
            # We effectively reduce our buying cost of the second transaction
            second_buy = min(second_buy, price - first_profit)
            # Calculate what our profit would be if we bought a second time and sold now
            second_profit = max(second_profit, price - second_buy)
        
        return second_profit

# Example usage:
# sol = Solution()
# print(sol.maxProfit([3,3,5,0,0,3,1,4]))  # Output: 6
# print(sol.maxProfit([1,2,3,4,5]))         # Output: 4
# print(sol.maxProfit([7,6,4,3,1]))         # Output: 0
# print(sol.maxProfit([1]))                 # Output: 0

```

### Detailed Explanation:

1. **Initialization:** 
   - Declare `first_buy` and `second_buy` initialized to `Infinity` to represent the minimal prices to buy for the first and second transactions respectively.
   - Declare `first_profit` and `second_profit` initialized to `0` which will store the maximum profits one can make with the first and second transactions respectively.

2. **Process Prices:**
   - For each `price`:
     - **For First Transaction:**
       - Update `first_buy` as minimum of `first_buy` and `price` to ensure you're buying at the lowest price seen so far.
       - Compute `first_profit` as the maximum value between `first_profit` and the difference between `price` and `first_buy`, i.e., if you were to sell at today's price, what would be the profit?
     - **For Second Transaction:**
       - Update `second_buy`. The cost of the second buy is treated as if you had the proceeds from the first sale, hence calculating it as `price - first_profit` and comparing with `second_buy` to minimize the effective price of purchase.
       - Update `second_profit` as the maximum of `second_profit` and the difference between the current `price` and `second_buy`.

3. **Return Output:**
   - Return the `second_profit` which is the answer to our problem, the max profit with at most two transactions.

### Complexity:
The time complexity is \(O(n)\) since we iterate through the list once. Space complexity is \(O(1)\), using only a constant amount of extra space for variables handling profits and purchase prices. This approach is efficient in terms of both time and space, and should scale well with larger inputs owing to its linear time complexity.

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
 To solve the problem of finding all shortest transformation sequences from a `beginWord` to an `endWord` using a given `wordList`, we can utilize a Bidirectional Breadth-First Search (BFS) strategy. This approach helps in finding the shortest path more efficiently than a standard BFS.

Here's the solution structured to run directly on LeetCode:



```python
from collections import defaultdict, deque

class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: list) -> list:
        if endWord not in wordList:
            return []
        
        wordList = set(wordList)  # Convert to set for O(1) look-up times
        layer = {}  # Stores current level nodes and paths leading to them
        layer[beginWord] = [[beginWord]]
        word_len = len(beginWord)
        
        while layer:
            newlayer = defaultdict(list)  # Stores the next level of nodes and paths
            for word in layer:
                if word == endWord:
                    return layer[word]  # Return all paths leading to the endWord
                for i in range(word_len):
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        newWord = word[:i] + c + word[i+1:]
                        if newWord in wordList:
                            newlayer[newWord] += [j + [newWord] for j in layer[word]]
            wordList -= set(newlayer.keys())  # Remove visited words
            layer = newlayer  # Move to the new layer
        
        return []

# The Solution can be tested on LeetCode platform directly with the given examples or new ones.

```

### Explanation

1. **Edge Case Check**: If `endWord` is not available in the `wordList`, no transformation is possible. So return an empty list immediately.

2. **Initialization**:
    - Convert `wordList` into a set for quicker access.
    - Create a dictionary `layer` where keys are the current words on this BFS level and values are all known paths to each word from the `beginWord`.

3. **BFS Iteration**:
    - For each word in the current `layer`:
        - If the word is `endWord`, return the stored paths leading to it.
        - For every possible single letter change in the word:
            - Create the new word (`newWord`).
            - If this `newWord` exists in the `wordList`, add these new paths into a new layer dictionary (`newlayer`).

4. **Progress Deepening**:
    - Update the `wordList` by removing words that are in the `newlayer` to prevent revisiting.
    - Update the current `layer` to `newlayer` for the next level of BFS.

5. **Completion**:
    - If no transformation is found, return an empty list after finishing the BFS.

This approach ensures we only collect the shortest paths due to the level-by-level nature of BFS, and switching to a new layer each time ensures paths don’t improperly cross and lengthen. This explains the need for a careful approach when checking and constructing new words. The usage of a default dictionary helps streamline path collection and extension.

# 128. Longest Consecutive Sequence

### Problem Description 
Given an unsorted array of integers `nums`, return the length of the longest consecutive elements sequence.


Example 1:
Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is `[1, 2, 3, 4]`. Therefore its length is 4.


Example 2:
Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9

Constraints:
`0 <= nums.length <= 104`
`-109 <= nums[i] <= 109`
Follow up: Could you implement the `O(n)` solution?

### Solution 
 To solve this problem, we can use a hashing technique for an optimal solution that runs in O(n) time. The main idea is to use a set to store all elements of the array, and then iterate over the array and keep extending the sequence both onward and backward from each number if its consecutive integers exist in the set. We update the longest sequence length accordingly every time we extend a sequence.

Here is how the given solution works:

1. All numbers are added to a set for O(1) average-time complexity lookups.
2. For each number in the input list, check if it's the start of a new sequence by confirming that `num - 1` is not in the set.
3. If it's the start, continue to explore the next consecutive elements `num + 1`, `num + 2`, ... until the numbers are not in the set and keep track of the sequence length.
4. Maintain a variable to track the maximum sequence length found.
5. Return the maximum sequence length.

By iterating through the list once and using the set for lookups and ignoring all non-starting points of sequences, the total complexity of the solution remains O(n).

Here is the implementation:



```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums_set = set(nums)
        longest_streak = 0
        
        for num in nums:
            # Check if it's the start of a new sequence
            if num - 1 not in nums_set:
                current_num = num
                current_streak = 1
                
                while current_num + 1 in nums_set:
                    current_num += 1
                    current_streak += 1
                
                longest_streak = max(longest_streak, current_streak)
        
        return longest_streak

```

### Explanation
- The `nums_set` is built from the list of numbers, which helps in O(1) average time complexity for checking if an element exists.
- We iterate through each number in `nums`, but we only start counting the sequence length if the number is the beginning of a sequence (i.e., `num - 1` does not exist in `nums_set`). This prevents recounting and thus keeps the complexity linear.
- We use a `while` loop to find the end of the consecutive sequence starting from each `num`.
- The maximum sequence length is updated if the current sequence length is greater than the existing record.

This approach ensures an optimal and efficient solution to the problem leveraging hashing by sets.

# 152. Maximum Product Subarray

### Problem Description 
Given an integer array `nums`, find a contiguous non-empty subarray within the array that has the largest product, and return the product.

It is guaranteed that the answer will fit in a 32-bit integer.

A subarray is a contiguous subsequence of the array.


Example 1:
Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.


Example 2:
Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.


Constraints:
`1 <= nums.length <= 2 * 104`
`-10 <= nums[i] <= 10`
The product of any prefix or suffix of `nums` is guaranteed to fit in a 32-bit integer.

### Solution 
 To solve this problem, we'll use a dynamic programming approach with two running products: the maximum and the minimum product up to the current index. This approach ensures that we cover scenarios where negative numbers, when multiplied by subsequent negative numbers, result in a positive product.

Here's why keeping track of both maximum and minimum products is essential:
1. **Maximum Product (`max_so_far`)**: It keeps track of the maximum product up to the current index which can potentially be the answer if it is the largest we've found so far.
2. **Minimum Product (`min_so_far`)**: While this might seem counterintuitive initially, keeping track of the minimum product (which will be the most negative) is crucial because a negative product can turn positive if it is multiplied by a negative number.

We will iterate through the given list while updating these two values (`max_so_far` and `min_so_far`). For each element, we calculate:
- The potential new maximum by considering the current element itself, its product with the previous maximum, and its product with the previous minimum.
- Similarly, compute the potential new minimum.

The reason behind considering both previous maximum and minimum is that negative numbers can flip signs when multiplied together.

Finally, our result is obtained by tracking the global optimal maximum product found during the traversal.

Here is the implementation in Python which can be directly run on the LeetCode platform:



```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        max_so_far = min_so_far = result = nums[0]
        
        for i in range(1, len(nums)):
            num = nums[i]
            # Calculate the temporary values before updating max_so_far, min_so_far
            temp_max = max(num, max_so_far * num, min_so_far * num)
            temp_min = min(num, max_so_far * num, min_so_far * num)
            
            max_so_far = temp_max
            min_so_far = temp_min
            
            # Update result with the maximum product found so far
            result = max(result, max_so_far)
        
        return result

```

### Explanation of the Example:
For the first example:
Input: `nums = [2,3,-2,4]`
- Initialize `max_so_far = min_so_far = result = 2`.
- For `num = 3`: `temp_max = max(3, 2*3, 2*3) = 6`, `temp_min = min(3, 2*3, 2*3) = 6`, result updates to `6`.
- For `num = -2`: `temp_max = max(-2, 6*(-2), 6*(-2)) = 0`, `temp_min = min(-2, 6*(-2), 6*(-2)) = -12`, result still `6`.
- For `num = 4`: `temp_max = max(4, 0*4, -12*4) = 4`, `temp_min = min(4, 0*4, -12*4) = -48`, result still `6`.
- Output `6`.

This implementation has a time complexity of O(n) where n is the number of elements in `nums` since we only make one pass through the array, and space complexity is O(1) as we only need a constant number of variables regardless of the input size.

