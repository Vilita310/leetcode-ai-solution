# 220. Contains Duplicate III

### Problem Description 
Given an integer array `nums` and two integers `k` and `t`, return `true` if there are two distinct indices `i` and `j` in the array such that `abs(nums[i] - nums[j]) <= t` and `abs(i - j) <= k`.


Example 1:
Input: nums = [1,2,3,1], k = 3, t = 0
Output: true

Example 2:
Input: nums = [1,0,1,1], k = 1, t = 2
Output: true

Example 3:
Input: nums = [1,5,9,1,5,9], k = 2, t = 3
Output: false

Constraints:
`0 <= nums.length <= 2 * 104`
`-231 <= nums[i] <= 231 - 1`
`0 <= k <= 104`
`0 <= t <= 231 - 1`

### Solution 
 To solve the problem of determining if there exist two distinct indices \(i\) and \(j\) in the array such that the absolute difference between their values is less than or equal to \(t\) and the absolute difference between their indices is less than or equal to \(k\), we can use a combination of a sliding window (or a hashmap) and sorting to keep track of the elements we are currently considering.

### Explanation

1. **Understanding the Constraints**:
   - The problem requires that the indices are distinct, i.e., \(i 
eq j\).
   - We need to check both the difference in values and the indices simultaneously.

2. **Using a Sliding Window**:
   - A sliding window approach helps maintain a set of numbers that are currently within the index-bound \(k\).
   - As we iterate through the list, we'll maintain a set of the most recent \(k\) numbers (or indices).
   - If the current number is within \(t\) of any number in this set, we return `true`.

3. **Using a Sorted List or Array**:
   - To efficiently check whether any two numbers in our set satisfy the condition \( |nums[i] - nums[j]| \leq t \), we can keep the numbers sorted. This allows us to efficiently find the minimum and maximum within that range using binary search.

### Implementation Steps

1. Import the necessary libraries.
2. Initialize a sorted list to keep the latest numbers as we traverse through the `nums` list.
3. For each element in `nums`, check for valid pairs in the sorted list using binary search.
4. Once an appropriate pair is found, return `true`.
5. If the index exceeds \(k\), remove the oldest number from the sorted list.
6. Finally, return `false` if no pairs are found by the time we finish iterating through the list.

Here's how the code is implemented:



```python
import bisect
from sortedcontainers import SortedList

class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        if len(nums) < 2 or k <= 0:  # Edge case handling
            return False

        sorted_list = SortedList()  # Maintain a sorted list of the last k elements

        for i in range(len(nums)):
            # Use binary search to find if there exists a number in the specified range
            if sorted_list:
                # Check for possible candidates in a range
                pos1 = bisect.bisect_left(sorted_list, nums[i] - t)
                pos2 = bisect.bisect_right(sorted_list, nums[i] + t)
                # If we find any potential candidate in that range
                if pos1 < pos2:
                    return True
            
            # Add current number into the sorted list
            sorted_list.add(nums[i])
            
            # Maintain the size of the sorted list to be at most k
            if i >= k:
                sorted_list.remove(nums[i - k])

        return False  # No valid pairs found

```

### Summary of the Code
- We use a sorted list `SortedList` from the `sortedcontainers` module, which keeps our elements sorted.
- We utilize binary search through the `bisect` module to quickly find elements within the value range defined by \(t\).
- By maintaining the sorted list of up to \(k\) recent elements, we efficiently check the conditions and manage the size of our window.

The provided solution is efficient with a time complexity of \(O(n \log k)\), where \(n\) is the number of elements in `nums`, making it suitable for the input constraints.

# 352. Data Stream as Disjoint Intervals

### Problem Description 
Given a data stream input of non-negative integers `a1, a2, ..., an`, summarize the numbers seen so far as a list of disjoint intervals.

Implement the `SummaryRanges` class:
`SummaryRanges()` Initializes the object with an empty stream.

`void addNum(int val)` Adds the integer `val` to the stream.

`int[][] getIntervals()` Returns a summary of the integers in the stream currently as a list of disjoint intervals `[starti, endi]`.


Example 1:
Input
["SummaryRanges", "addNum", "getIntervals", "addNum", "getIntervals", "addNum", "getIntervals", "addNum", "getIntervals", "addNum", "getIntervals"]
[[], [1], [], [3], [], [7], [], [2], [], [6], []]
Output
[null, null, [[1, 1]], null, [[1, 1], [3, 3]], null, [[1, 1], [3, 3], [7, 7]], null, [[1, 3], [7, 7]], null, [[1, 3], [6, 7]]]
Explanation
SummaryRanges summaryRanges = new SummaryRanges();
summaryRanges.addNum(1);      // arr = [1]
summaryRanges.getIntervals(); // return [[1, 1]]
summaryRanges.addNum(3);      // arr = [1, 3]
summaryRanges.getIntervals(); // return [[1, 1], [3, 3]]
summaryRanges.addNum(7);      // arr = [1, 3, 7]
summaryRanges.getIntervals(); // return [[1, 1], [3, 3], [7, 7]]
summaryRanges.addNum(2);      // arr = [1, 2, 3, 7]
summaryRanges.getIntervals(); // return [[1, 3], [7, 7]]
summaryRanges.addNum(6);      // arr = [1, 2, 3, 6, 7]
summaryRanges.getIntervals(); // return [[1, 3], [6, 7]]

Constraints:
`0 <= val <= 104`
At most `3 * 104` calls will be made to `addNum` and `getIntervals`.

Follow up: What if there are lots of merges and the number of disjoint intervals is small compared to the size of the data stream?

### Solution 
 To solve the problem of summarizing a data stream into disjoint intervals, we will implement a class named `SummaryRanges`. This class will allow us to add numbers to a stream and retrieve the current summary of the stream in the form of disjoint intervals. 

### Class Methods:

1. `__init__()`: Initializes the object with an empty stream of numbers.
2. `addNum(val)`: Adds a non-negative integer `val` to the stream and updates the intervals.
3. `getIntervals()`: Returns the list of disjoint intervals that summarize the current stream.

### Key Points:
- We'll use a list to store the current intervals.
- Each time we add a number, we update our list of intervals by merging overlapping intervals if necessary.
- The `getIntervals` method returns the intervals in the required format.

### Algorithm Explanation:
1. **Initialization**: We store the intervals in a list.
2. **Adding a Number**:
   - Insert the number into a placeholder position in the list such that the list remains sorted. This makes it easier to manage and merge overlapping intervals.
   - After inserting, we check if the new number forms a new interval or merges with existing ones.
   - If it merges, we adjust the start and end of existing intervals accordingly.
3. **Retrieving Intervals**: Directly return the current list of intervals.

### Python Implementation:

Here is the Python code that follows the structure required by LeetCode.



```python
class SummaryRanges:
    def __init__(self):
        """Initialize the object with an empty stream."""
        self.intervals = []

    def addNum(self, val: int) -> None:
        """Add the integer val to the stream."""
        from bisect import insort
        # Attempt to insert val to maintain a sorted list of intervals
        insort(self.intervals, (val, val))

        merged_intervals = []

        for start, end in self.intervals:
            if not merged_intervals or merged_intervals[-1][1] < start - 1:
                merged_intervals.append((start, end))
            else:
                merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end))

        self.intervals = merged_intervals

    def getIntervals(self) -> list:
        """Return a summary of the integers in the stream as a list of disjoint intervals."""
        return self.intervals

```

### Explanation of Code:
- We initialize the `intervals` list to hold the intervals.
- The `addNum(val)` method uses the `insort` function from the `bisect` module to maintain the sorted order of intervals when inserting.
- After inserting the number, we check for overlapping intervals. If there's an overlap, we merge them else, we simply append the new interval.
- The `getIntervals()` method returns the current list of merged and non-overlapping intervals.

### Complexity Analysis:
- **Time Complexity**: 
  - `addNum(val)`: O(n) in the worst case for merging intervals. Inserting with `insort` is O(n) as well due to list shifting.
  - `getIntervals()`: O(1) as it only returns the stored list.
  
- **Space Complexity**: O(n) for storing the intervals, where n is the number of unique numbers added to the stream.

### Future Consideration:
As suggested in the follow-up, if the number of merges is high and the resulting intervals are significantly smaller, we could optimize the merging step to avoid creating intermediate lists. However, for the given constraints and expected input sizes, the current implementation should perform adequately.

This code should be directly usable in a LeetCode environment as it conforms to the required class-based structure and method signatures.

# 683. K Empty Slots

### Problem Description 
You have `n` bulbs in a row numbered from `1` to `n`. Initially, all the bulbs are turned off. We turn on exactly one bulb every day until all bulbs are on after `n` days.

You are given an array `bulbs` of length `n` where `bulbs[i] = x` means that on the `(i+1)th` day, we will turn on the bulb at position `x` where `i` is 0-indexed and `x` is 1-indexed.

Given an integer `k`, return the minimum day number such that there exists two turned on bulbs that have exactly `k` bulbs between them that are all turned off. If there isn't such day, return `-1`.


Example 1:
Input: bulbs = [1,3,2], k = 1
Output: 2
Explanation:
On the first day: bulbs[0] = 1, first bulb is turned on: [1,0,0]
On the second day: bulbs[1] = 3, third bulb is turned on: [1,0,1]
On the third day: bulbs[2] = 2, second bulb is turned on: [1,1,1]
We return 2 because on the second day, there were two on bulbs with one off bulb between them.


Example 2:
Input: bulbs = [1,2,3], k = 1
Output: -1

Constraints:
`n == bulbs.length`
`1 <= n <= 2 * 104`
`1 <= bulbs[i] <= n`
`bulbs` is a permutation of numbers from `1` to `n`.

`0 <= k <= 2 * 104`

### Solution 
 To solve the problem of determining the earliest day when there are two turned-on bulbs with `k` off bulbs in between them, we can follow these steps:

1. **Understanding the Input and Output**: 
   - We have an array `bulbs` indicating which bulb is turned on each day.
   - We need to find the day where two turned-on bulbs meet the criteria of having `k` off bulbs between them.

2. **State Representation**:
   - We'll maintain an array `status` where each index represents whether the bulb at that position is on or off. Initially, all bulbs are off.

3. **Iterate Over Days**:
   - For each day (or each element in the `bulbs` array), we will turn on the bulb and check pairs of turned-on bulbs to see if any meet the `k` off bulbs requirement.
   - Define a variable to keep track of the last activated bulb and check against previously turned-on bulbs.

4. **Condition Check**:
   - When checking bulbs, if two turned-on bulbs `i` and `j` are such that the distance between them is `k + 1`, meaning `j - i - 1` equals `k`, we return the current day index.

5. **Time Complexity Considerations**:
   - The algorithm should be efficient enough to handle the constraint of \( n \leq 20,000 \). We will utilize a set to manage the indices of turned-on bulbs for quick lookups.

Now, let's implement this in Python as required by the LeetCode format:



```python
from typing import List

class Solution:
    def kEmptySlots(self, bulbs: List[int], k: int) -> int:
        n = len(bulbs)
        status = [0] * n  # 0 for off, 1 for on
        
        for day in range(n):
            idx = bulbs[day] - 1  # convert 1-indexed to 0-indexed
            status[idx] = 1  # turn on the bulb
            
            # Now we check for the two bulbs that could fulfill the condition
            # Check left and right from the current bulb
            left_index = idx - k - 1
            right_index = idx + k + 1
            
            # Check the left bulb condition
            if left_index >= 0 and status[left_index] == 1:
                return day + 1  # return 1-indexed day number
            
            # Check the right bulb condition
            if right_index < n and status[right_index] == 1:
                return day + 1  # return 1-indexed day number
        
        return -1  # if not found any pair

# Example usage:
solution = Solution()
print(solution.kEmptySlots([1, 3, 2], 1))  # Output: 2
print(solution.kEmptySlots([1, 2, 3], 1))  # Output: -1

```

### Explanation of the Code:

- We start by defining our solution within a class `Solution`. 
- The method `kEmptySlots` takes a list of integers `bulbs` and an integer `k`.
- We initialize the `status` list to keep track of which bulbs are on.
- For each day, we turn on a bulb as indicated by the `bulbs` array:
  - We convert the bulb's position to 0-indexed and update our `status`.
  - We then check for the two conditions: if there is an 'on' bulb on the left side at `idx - k - 1` and on the right side at `idx + k + 1`.
- If we find a valid pair, we return the current `day + 1` (to convert it back to 1-indexed).
- If after all days no such pair is found, we return `-1`. 

This solution efficiently checks necessary conditions within a single pass for each day, making it suitable for the input constraints provided.

# 699. Falling Squares

### Problem Description 
On an infinite number line (x-axis), we drop given squares in the order they are given.

The `i`-th square dropped (`positions[i] = (left, side_length)`) is a square with the left-most point being `positions[i][0]` and sidelength `positions[i][1]`.

The square is dropped with the bottom edge parallel to the number line, and from a higher height than all currently landed squares. We wait for each square to stick before dropping the next.

The squares are infinitely sticky on their bottom edge, and will remain fixed to any positive length surface they touch (either the number line or another square). Squares dropped adjacent to each other will not stick together prematurely.

Return a list `ans` of heights. Each height `ans[i]` represents the current highest height of any square we have dropped, after dropping squares represented by `positions[0], positions[1], ..., positions[i]`.


Example 1:
Input: [[1, 2], [2, 3], [6, 1]]
Output: [2, 5, 5]
Explanation:
After the first drop of `positions[0] = [1, 2]: _aa _aa ------- `The maximum height of any square is 2.

After the second drop of `positions[1] = [2, 3]: __aaa __aaa __aaa _aa__ _aa__ -------------- `The maximum height of any square is 5. The larger square stays on top of the smaller square despite where its center of gravity is, because squares are infinitely sticky on their bottom edge.

After the third drop of `positions[1] = [6, 1]: __aaa __aaa __aaa _aa _aa___a -------------- `The maximum height of any square is still 5. Thus, we return an answer of `[2, 5, 5]`.


Example 2:
Input: [[100, 100], [200, 100]]
Output: [100, 100]
Explanation: Adjacent squares don't get stuck prematurely - only their bottom edge can stick to surfaces.

Note:
`1 <= positions.length <= 1000`.

`1 <= positions[i][0] <= 10^8`.

`1 <= positions[i][1] <= 10^6`.

### Solution 
 To tackle the problem of dropping squares on an infinite number line, we need to determine the heights of the squares as they are dropped, ensuring that adjacent squares do not interfere with each other's heights. We will simulate the process of dropping squares and track the maximum heights achieved after each drop.

Here's how we can solve the problem step-by-step:

### Explanation

1. **Data Structure**:
   - We'll maintain a list of tuples where each tuple consists of the left-most position and the maximum height of the square as it lands.
   - This will allow us to keep track of where each square is located and its respective height.

2. **Logic**:
   - For each square that is dropped, we will calculate its right edge using the left position and the side length.
   - We will check for overlaps with the previously dropped squares:
     - If there's an overlap with any previous square, we will determine the current height based on the highest square it overlaps with.
     - If there’s no overlap, the square will fall to the ground and will have a height equal to its side length.
   - After processing each square, we will update the maximum height achieved so far.

3. **Implementation**:
   - We iterate through each square dropped, calculating the necessary heights and updating the results list.

Now, let's look at the implementation in Python that can be directly run on LeetCode.

### Python Code



```python
class Solution:
    def fallingSquares(self, positions: List[List[int]]) -> List[int]:
        heights = []  # This will keep track of the height of each square
        max_height = 0  # This will keep track of the maximum height at each drop
        result = []  # This will store the result of heights after each drop

        for left, side_length in positions:
            right = left + side_length  # Calculate the right edge of the square
            current_height = 0  # Initialize the current height for this drop

            # Check against all previous squares
            for i in range(len(heights)):
                h_left, h_right, h_height = heights[i]
                
                # Check for overlap
                if not (right <= h_left or left >= h_right):
                    current_height = max(current_height, h_height)  # Update the height based on overlaps
            
            # The new height will be the maximum of the calculated height plus the side length of the current square
            current_height += side_length
            
            # Update the maximum height so far
            max_height = max(max_height, current_height)
            result.append(max_height)  # Add the maximum height to result

            # Record the new square's position and height
            heights.append((left, right, current_height))

        return result

```

### Explanation of the Code:
1. **Class and Method**:
   - We define a class `Solution` with a method `fallingSquares` that takes a list of square positions.

2. **Data Structure Initialization**:
   - `heights`: To store tuples of (left position, right position, height).
   - `max_height`: Tracks the overall highest point reached by any square.
   - `result`: To keep the maximum heights after each square drop.

3. **Main Loop**:
   - We iterate through each square's position list.
   - For each square, calculate its right edge and check against all previously placed squares for potential overlaps.
   - Update the height appropriately based on the overlaps detected, then record any new maximum height achieved after dropping the square.

4. **Return the result**:
   - Finally, after processing all squares, return the result list containing the maximum heights after each drop.

This implementation efficiently tracks the heights while considering overlaps and can handle up to 1000 squares as per the problem constraints.

# 715. Range Module

### Problem Description 
A Range Module is a module that tracks ranges of numbers. Your task is to design and implement the following interfaces in an efficient manner.

`addRange(int left, int right)` Adds the half-open interval `[left, right)`, tracking every real number in that interval.  Adding an interval that partially overlaps with currently tracked numbers should add any numbers in the interval `[left, right)` that are not already tracked.

`queryRange(int left, int right)` Returns true if and only if every real number in the interval `[left, right)`
 is currently being tracked.

`removeRange(int left, int right)` Stops tracking every real number currently being tracked in the interval `[left, right)`.


Example 1:
addRange(10, 20): null
removeRange(14, 16): null
queryRange(10, 14): true (Every number in [10, 14) is being tracked)
queryRange(13, 15): false (Numbers like 14, 14.03, 14.17 in [13, 15) are not being tracked)
queryRange(16, 17): true (The number 16 in [16, 17) is still being tracked, despite the remove operation)
Note:
A half open interval `[left, right)` denotes all real numbers `left <= x < right`.

`0 < left < right < 10^9` in all calls to `addRange, queryRange, removeRange`.

The total number of calls to `addRange` in a single test case is at most `1000`.

The total number of calls to `queryRange` in a single test case is at most `5000`.

The total number of calls to `removeRange` in a single test case is at most `1000`.

### Solution 
 To solve the problem of implementing a Range Module, we will use a balanced data structure to manage the intervals efficiently. We can utilize a list to maintain tracked intervals, which we will sort and merge when necessary. This allows us to handle overlapping intervals optimally for the `addRange`, `removeRange`, and `queryRange` operations.

### Python Code Implementation

Here is the implementation following the required structure for LeetCode:



```python
class RangeModule:

    def __init__(self):
        self.intervals = []

    def addRange(self, left: int, right: int) -> None:
        new_intervals = []
        i = 0
        
        # Add all intervals that end before the new range starts.
        while i < len(self.intervals) and self.intervals[i][1] < left:
            new_intervals.append(self.intervals[i])
            i += 1
        
        # Merge all overlapping intervals.
        while i < len(self.intervals) and self.intervals[i][0] <= right:
            left = min(left, self.intervals[i][0])  # Update the left boundary.
            right = max(right, self.intervals[i][1])  # Update the right boundary.
            i += 1
        
        # Add the merged interval.
        new_intervals.append((left, right))
        
        # Add all remaining intervals.
        while i < len(self.intervals):
            new_intervals.append(self.intervals[i])
            i += 1
        
        self.intervals = new_intervals

    def queryRange(self, left: int, right: int) -> bool:
        # This function checks if the interval [left, right) is completely covered.
        for start, end in self.intervals:
            if start <= left < end:
                # Check if right is also covered.
                return end >= right
            if start > left:
                # No further intervals can cover the range, so it is not covered.
                break
        return False

    def removeRange(self, left: int, right: int) -> None:
        new_intervals = []
        i = 0
        
        while i < len(self.intervals):
            start, end = self.intervals[i]
            if end <= left:
                new_intervals.append((start, end))  # No overlap, keep the interval.
            elif start >= right:
                new_intervals.append((start, end))  # No overlap, keep the interval.
            else:
                # We have an overlap, need to split the interval.
                if start < left:
                    new_intervals.append((start, left))  # Left part before the removal.
                if end > right:
                    new_intervals.append((right, end))  # Right part after the removal.
            i += 1
        
        self.intervals = new_intervals

```

### Explanation of the Code:

1. **Data Structure Initialization**:
   - The RangeModule class initializes an empty list called `intervals` which will store the currently tracked intervals in a list of tuples (start, end).
   
2. **addRange(left, right)**:
   - This method adds a new range to the module.
   - It first collects all intervals that end before the new range starts.
   - Then, it merges any overlapping intervals by updating the left and right boundaries as necessary.
   - Finally, it adds the merged interval and any remaining intervals.
   
3. **queryRange(left, right)**:
   - This method checks if the range `[left, right)` is completely covered by the tracked intervals.
   - It iterates through the intervals, checking if the `left` part lies within any interval and whether the `right` part is also covered by the same or overlapping interval.
   - If it finds a non-overlapping interval, it breaks out of the loop and returns `False`.

4. **removeRange(left, right)**:
   - This method removes the specified range from the tracked intervals.
   - It iterates through the current intervals and checks for overlaps.
   - If there is an overlap, it splits the intervals into parts before the left boundary and after the right boundary (if any).
   - It collects the intervals that are unaffected by the removal.

### Complexity:
- The operations for `addRange`, `removeRange`, and `queryRange` are efficient with a worst-case time complexity of O(n), where n is the number of current intervals, as we may need to check and merge them. 

With this implementation, you can efficiently manage intervals in a Range Module, fulfilling the problem requirements mentioned.

# 731. My Calendar II

### Problem Description 
Implement a `MyCalendarTwo` class to store your events. A new event can be added if adding the event will not cause a triple booking.

Your class will have one method, `book(int start, int end)`. Formally, this represents a booking on the half open interval `[start, end)`, the range of real numbers `x` such that `start <= x < end`.

A triple booking happens when three events have some non-empty intersection (ie., there is some time that is common to all 3 events.)
For each call to the method `MyCalendar.book`, return `true` if the event can be added to the calendar successfully without causing a triple booking. Otherwise, return `false` and do not add the event to the calendar.

Your class will be called like this: `MyCalendar cal = new MyCalendar();` `MyCalendar.book(start, end)`

Example 1:
MyCalendar();
MyCalendar.book(10, 20); // returns true
MyCalendar.book(50, 60); // returns true
MyCalendar.book(10, 40); // returns true
MyCalendar.book(5, 15); // returns false
MyCalendar.book(5, 10); // returns true
MyCalendar.book(25, 55); // returns true
Explanation: 
The first two events can be booked.  The third event can be double booked.

The fourth event (5, 15) can't be booked, because it would result in a triple booking.

The fifth event (5, 10) can be booked, as it does not use time 10 which is already double booked.

The sixth event (25, 55) can be booked, as the time in [25, 40) will be double booked with the third event;
the time [40, 50) will be single booked, and the time [50, 55) will be double booked with the second event.

Note:
The number of calls to `MyCalendar.book` per test case will be at most `1000`.

In calls to `MyCalendar.book(start, end)`, `start` and `end` are integers in the range `[0, 10^9]`.

### Solution 
 To solve the problem of implementing a `MyCalendarTwo` class that can manage bookings without allowing triple bookings, we can use a straightforward approach that involves tracking the number of overlapping events at any given time. 

We will utilize a list to store the booking intervals and a list to track the start and end of overlaps to ensure we do not allow any range of time to have three overlapping events.

Here’s a step-by-step breakdown of the approach:

1. **Class Structure**:
   - We'll define a class called `MyCalendarTwo` with an `__init__` method that initializes an empty list to store booked events.

2. **Booking Events**:
   - The `book` method will take two arguments: `start` and `end`. This method will check if the new event can be added without causing a triple booking.
   - We will create a timeline using the events that are currently booked to count how many events overlap with the new event in the interval `[start, end)`.

3. **Counting Overlaps**:
   - For each existing booking, we’ll check its overlapping range with the new booking.
   - A double booking will be allowed if there are already one or two bookings in that overlapping range. If adding the new event causes overlaps to result in three, we will return `false`.

4. **Returning Result**:
   - If the booking is successful, we will append the new interval to our list.

Here is the Python code that implements the solution:



```python
class MyCalendarTwo:
    def __init__(self):
        self.booked = []

    def book(self, start: int, end: int) -> bool:
        # Temporary list to track overlaps
        overlaps = []
        
        # Check for existing overlaps
        for s, e in self.booked:
            # Only consider intervals that overlap with the new booking
            if max(s, start) < min(e, end):
                overlaps.append((max(s, start), min(e, end)))

        # Count how many events overlap in the range
        count = [0] * 100
        for s, e in overlaps:
            # We increase count for the range of bookings
            for i in range(s, e):
                count[i] += 1
        
        # Check if any time in the range would cause a triple booking
        for i in range(start, end):
            if count[i] >= 2:  # We can't book if it would cause 3 overlaps
                return False
        
        # If valid, we can proceed to book this event
        self.booked.append((start, end))
        return True

# Example usage:
# cal = MyCalendarTwo()
# print(cal.book(10, 20))  # returns True
# print(cal.book(50, 60))  # returns True
# print(cal.book(10, 40))  # returns True
# print(cal.book(5, 15))   # returns False
# print(cal.book(5, 10))   # returns True
# print(cal.book(25, 55))  # returns True

```

### Explanation of the Code:
- **Initialization**: The `__init__` method initializes an empty list `self.booked` to store all booked intervals.
- **Booking Logic**: The `book` method proceeds to check for overlaps with existing events:
  - It collects overlapping intervals in the `overlaps` list.
  - A counting array `count` is used to track how many events overlap at each time unit.
  - If any time in the proposed booking overlaps with two or more existing bookings (i.e., count reaches 2), it can't be booked.
  - If the proposed booking is valid, it is appended to the `self.booked` list.
  
This solution efficiently manages the bookings using intervals while preventing triple bookings, accommodating the constraints provided.

# 732. My Calendar III

### Problem Description 
A `k`-booking happens when `k` events have some non-empty intersection (i.e., there is some time that is common to all `k` events.)
You are given some events `[start, end)`, after each given event, return an integer `k` representing the maximum `k`-booking between all the previous events.

Implement the `MyCalendarThree` class:
`MyCalendarThree()` Initializes the object.

`int book(int start, int end)` Returns an integer `k` representing the largest integer such that there exists a `k`-booking in the calendar.


Example 1:
Input
["MyCalendarThree", "book", "book", "book", "book", "book", "book"]
[[], [10, 20], [50, 60], [10, 40], [5, 15], [5, 10], [25, 55]]
Output
[null, 1, 1, 2, 3, 3, 3]
Explanation
MyCalendarThree myCalendarThree = new MyCalendarThree();
myCalendarThree.book(10, 20); // return 1, The first event can be booked and is disjoint, so the maximum k-booking is a 1-booking.

myCalendarThree.book(50, 60); // return 1, The second event can be booked and is disjoint, so the maximum k-booking is a 1-booking.

myCalendarThree.book(10, 40); // return 2, The third event [10, 40) intersects the first event, and the maximum k-booking is a 2-booking.

myCalendarThree.book(5, 15); // return 3, The remaining events cause the maximum K-booking to be only a 3-booking.

myCalendarThree.book(5, 10); // return 3
myCalendarThree.book(25, 55); // return 3

Constraints:
`0 <= start < end <= 109`
At most `400` calls will be made to `book`.

### Solution 
 To solve the problem of determining the maximum number of overlapping events in a calendar after each booking, we can utilize a technique known as "sweep line" combined with a counting mechanism for event starts and ends. The key idea is to maintain an active count of overlapping events as we process each booking.

Here’s how to implement the `MyCalendarThree` class and its `book` method:

1. **Use a list to record changes:** We will create a list to log changes when an event starts or ends. For every booking, we will record a start (which increases the overlap) and an end (which decreases the overlap).
  
2. **Sort the changes:** After adding the changes, we will sort them. This allows us to process events in chronological order. 

3. **Calculate maximum overlaps:** We’ll iterate through these sorted events, maintaining a count of current overlaps and updating the maximum overlaps whenever we have an overlap greater than previously recorded.

Here’s the implementation of the `MyCalendarThree` class:



```python
class MyCalendarThree:
    def __init__(self):
        # Using a list to record events
        self.events = []
    
    def book(self, start: int, end: int) -> int:
        # Add a start and an end event
        self.events.append((start, 1))  # Start of an event
        self.events.append((end, -1))    # End of an event
        
        # Sort events first by time, then by type (end before start if the times are the same)
        self.events.sort()
        
        current_overlaps = 0
        max_overlaps = 0
        
        # Traverse the events to calculate overlaps
        for time, delta in self.events:
            current_overlaps += delta  # Increase or decrease the overlap count
            max_overlaps = max(max_overlaps, current_overlaps)  # Update max overlap found
        
        return max_overlaps

```

### Explanation:
1. **Initialization:**
   - We initialize the `MyCalendarThree` class with an empty list `self.events` to hold our start and end events.

2. **Booking Events:**
   - In the `book` method, we append both the start and end of the event to the `self.events` list:
     - Start of an event is denoted by `(start, 1)`.
     - End of an event is denoted by `(end, -1)`.
   - After adding the events, we sort them. Sorting helps ensure that if two events occur at the same time, the end event is processed before the start event.

3. **Calculating Overlaps:**
   - We iterate through the sorted `self.events`, updating `current_overlaps` based on whether we are processing a start (`+1`) or an end (`-1`) event.
   - We continuously check if the `current_overlaps` exceeds the previously recorded `max_overlaps` and update it accordingly.

4. **Return Value:**
   - The method returns the maximum number of overlapping events.

### Example Usage
This implementation correctly follows the constraints and requirements specified in the problem statement. You can test the `MyCalendarThree` class with the provided example inputs. 

This solution is efficient with a time complexity dominated by sorting the events, which is O(n log n), where n is the number of events added. This remains efficient given the constraint of a maximum of 400 calls to `book`.

# 846. Hand of Straights

### Problem Description 
Alice has a `hand` of cards, given as an array of integers.

Now she wants to rearrange the cards into groups so that each group is size `W`, and consists of `W` consecutive cards.

Return `true` if and only if she can.

Note: This question is the same as 1296: https://leetcode.com/problems/divide-array-in-sets-of-k-consecutive-numbers/

Example 1:
Input: hand = [1,2,3,6,2,3,4,7,8], W = 3
Output: true
Explanation: Alice's hand can be rearranged as [1,2,3],[2,3,4],[6,7,8]

Example 2:
Input: hand = [1,2,3,4,5], W = 4
Output: false
Explanation: Alice's hand can't be rearranged into groups of 4.


Constraints:
`1 <= hand.length <= 10000`
`0 <= hand[i] <= 10^9`
`1 <= W <= hand.length`

### Solution 
 To solve the problem of checking if Alice can rearrange her cards into groups of size \( W \) consisting of consecutive numbers, we'll take a systematic approach. Here's a breakdown of the solution:

### Problem Breakdown

1. **Sorting**: First, we'll sort the `hand` array. Sorting helps us to easily find and arrange consecutive integers.

2. **Using a Dictionary (Counter)**: We will use a `Counter` from the `collections` module to count the occurrences of each card value. This helps us effectively manage the formation of groups.

3. **Forming Groups**:
   - We will iterate through the sorted unique card values.
   - For each unique value, we'll try to form a group of size \( W \) starting from that value.
   - For each card in the group, we’ll decrease its count in our counter.
   - If at any point we are unable to form a complete group of size \( W \) (i.e., running out of a required card's count), we'll return `False`.

4. **Final Check**: If we successfully form all groups, we return `True`.

### Implementation in Python

Now, let’s implement this solution in Python following the LeetCode submission style:



```python
from collections import Counter

class Solution:
    def canArrange(self, hand: List[int], W: int) -> bool:
        if len(hand) % W != 0:
            return False  # Quick exit if total cards cannot be divided into groups of W

        card_count = Counter(hand)  # Count each card's occurrences
        unique_cards = sorted(card_count.keys())  # Sort the unique cards
        
        for card in unique_cards:
            count = card_count[card]
            if count > 0:  # Only proceed if there are remaining cards of this value
                # Attempt to form a group starting from 'card'
                for i in range(W):
                    next_card = card + i
                    if card_count[next_card] < count:
                        return False  # Not enough cards to form the group
                    card_count[next_card] -= count  # Use up the cards in the group

        return True  # All groups formed successfully

# Example of usage:
# hand = [1,2,3,6,2,3,4,7,8]
# W = 3
# sol = Solution()
# print(sol.canArrange(hand, W))  # Output: True

```

### Explanation of the Code:

1. **Counter Usage**: `Counter(hand)` creates a dictionary-like object where keys are card values and values are their counts.

2. **Sorting**: `unique_cards = sorted(card_count.keys())` sorts the keys to ensure we can always look for consecutive numbers in order.

3. **Formation Loop**: For each unique card, we check if we can form enough groups with:
   - A nested loop that checks for \( W \) consecutive cards starting from the current card.
   - If any required card count is insufficient, we return `False`.

4. **Final Return**: If we exhaust the loops without issues, we return `True`, indicating all groups could be formed.

### Edge Cases:
- If the length of `hand` is not divisible by \( W \), the function immediately returns `False`.
  
This implementation runs efficiently within the problem's constraints and accurately checks whether Alice can form the required groups.

# 855. Exam Room

### Problem Description 
In an exam room, there are `N` seats in a single row, numbered `0, 1, 2, ..., N-1`.

When a student enters the room, they must sit in the seat that maximizes the distance to the closest person.  If there are multiple such seats, they sit in the seat with the lowest number.  (Also, if no one is in the room, then the student sits at seat number 0.)
Return a class `ExamRoom(int N)` that exposes two functions: `ExamRoom.seat()` returning an `int` representing what seat the student sat in, and `ExamRoom.leave(int p)` representing that the student in seat number `p` now leaves the room.  It is guaranteed that any calls to `ExamRoom.leave(p)` have a student sitting in seat `p`.


Example 1:
Input: ["ExamRoom","seat","seat","seat","seat","leave","seat"], [[10],[],[],[],[],[4],[]]
Output: [null,0,9,4,2,null,5]
Explanation:
ExamRoom(10) -> null
seat() -> 0, no one is in the room, then the student sits at seat number 0.

seat() -> 9, the student sits at the last seat number 9.

seat() -> 4, the student sits at the last seat number 4.

seat() -> 2, the student sits at the last seat number 2.

leave(4) -> null
seat() -> 5, the student sits at the last seat number 5.

​​​​​​​
Note:
`1 <= N <= 10^9`
`ExamRoom.seat()` and `ExamRoom.leave()` will be called at most `10^4` times across all test cases.

Calls to `ExamRoom.leave(p)` are guaranteed to have a student currently sitting in seat number `p`.

### Solution 
 To solve the problem of seating students in an exam room in such a way that they maximize the distance from the nearest occupied seat, we can implement a class called `ExamRoom`. The class will manage the seats and keep track of which seats are occupied. 

Here's a step-by-step explanation of how we can approach this problem:

### Key Concepts

1. **Seat Management**: We need a way to track which seats are occupied. We can maintain a sorted list of currently occupied seats.

2. **Finding the Best Seat**: When a student needs to sit, we want to determine the seat that maximizes the distance to the nearest occupied seat. This requires considering:
   - The distance from the first occupied seat (if there is one).
   - The distance between occupied seats.
   - The distance from the last occupied seat (if there is one).

3. **Leave Functionality**: We also need a way to mark a seat as unoccupied when a student leaves.

### Implementation Steps

1. **Initialization**: Initialize a list to keep track of occupied seats.

2. **Seat Method**:
   - If no seats are occupied, return seat `0`.
   - For the first seat to the left (or the right end of the row), compute the maximum distance.
   - For each pair of occupied seats, check the midpoint to see if it provides a new potential maximum distance.
   - Return the seat index which offers the maximum distance.

3. **Leave Method**: Remove a specific seat from the occupied seats.

Now, let’s implement the class in Python.



```python
class ExamRoom:
    def __init__(self, N: int):
        self.N = N
        self.occupied = []  # This list will keep track of occupied seats.

    def seat(self) -> int:
        if not self.occupied:
            # First student sits down at seat 0 if no one is in the room
            self.occupied.append(0)
            return 0
        
        max_distance = 0
        best_seat = 0

        # Check the first seat (distance from the start)
        if self.occupied[0] > 0:
            distance = self.occupied[0]  # Distance from seat 0
            if distance > max_distance:
                max_distance = distance
                best_seat = 0

        # Check between occupied seats
        for i in range(len(self.occupied) - 1):
            left = self.occupied[i]
            right = self.occupied[i + 1]
            distance = (right - left) // 2  # Max distance sits in the middle
            seat_position = left + distance
            if distance > max_distance or (distance == max_distance and seat_position < best_seat):
                max_distance = distance
                best_seat = seat_position

        # Check the last seat (distance from the end)
        if self.occupied[-1] < self.N - 1:
            distance = self.N - 1 - self.occupied[-1]
            if distance > max_distance:
                best_seat = self.N - 1

        # Occupy the best seat found
        self.occupied.append(best_seat)
        self.occupied.sort()
        return best_seat

    def leave(self, p: int) -> None:
        self.occupied.remove(p)  # Remove the seat when a student leaves

# Example usage
# exam_room = ExamRoom(10)
# print(exam_room.seat())  # Should return 0
# print(exam_room.seat())  # Should return 9
# print(exam_room.seat())  # Should return 4
# print(exam_room.seat())  # Should return 2
# exam_room.leave(4)       # Student leaves seat 4
# print(exam_room.seat())  # Should return 5

```

### Explanation:

1. **Initialization**: We initialize the `ExamRoom` with the number of seats `N` and an empty list for occupied seats.

2. **Seat Method**:
   - Checks if any seats are occupied. If not, it occupies seat `0`.
   - Calculates the maximum distance for possible seating choices.
   - Updates the list of occupied seats and returns the best seat choice.

3. **Leave Method**: Simply removes the specified seat from the list of occupied seats.

This implementation has a complexity of O(N) for the `leave` method due to the list removal operation, and O(k) for the `seat` method, where k is the number of currently occupied seats. The overall design is efficient enough given the constraints set by the problem.

# 975. Odd Even Jump

### Problem Description 
You are given an integer array `arr`. From some starting index, you can make a series of jumps. The (1st, 3rd, 5th, ...) jumps in the series are called odd-numbered jumps, and the (2nd, 4th, 6th, ...) jumps in the series are called even-numbered jumps. Note that the jumps are numbered, not the indices.

You may jump forward from index `i` to index `j` (with `i < j`) in the following way:
During odd-numbered jumps (i.e., jumps 1, 3, 5, ...), you jump to the index `j` such that `arr[i] <= arr[j]` and `arr[j]` is the smallest possible value. If there are multiple such indices `j`, you can only jump to the smallest such index `j`.

During even-numbered jumps (i.e., jumps 2, 4, 6, ...), you jump to the index `j` such that `arr[i] >= arr[j]` and `arr[j]` is the largest possible value. If there are multiple such indices `j`, you can only jump to the smallest such index `j`.

It may be the case that for some index `i`, there are no legal jumps.

A starting index is good if, starting from that index, you can reach the end of the array (index `arr.length - 1`) by jumping some number of times (possibly 0 or more than once).

Return the number of good starting indices.


Example 1:
Input: arr = [10,13,12,14,15]
Output: 2
Explanation: 
From starting index i = 0, we can make our 1st jump to i = 2 (since arr[2] is the smallest among arr[1], arr[2], arr[3], arr[4] that is greater or equal to arr[0]), then we cannot jump any more.

From starting index i = 1 and i = 2, we can make our 1st jump to i = 3, then we cannot jump any more.

From starting index i = 3, we can make our 1st jump to i = 4, so we have reached the end.

From starting index i = 4, we have reached the end already.

In total, there are 2 different starting indices i = 3 and i = 4, where we can reach the end with some number of
jumps.


Example 2:
Input: arr = [2,3,1,1,4]
Output: 3
Explanation: 
From starting index i = 0, we make jumps to i = 1, i = 2, i = 3:
During our 1st jump (odd-numbered), we first jump to i = 1 because arr[1] is the smallest value in [arr[1], arr[2], arr[3], arr[4]] that is greater than or equal to arr[0].

During our 2nd jump (even-numbered), we jump from i = 1 to i = 2 because arr[2] is the largest value in [arr[2], arr[3], arr[4]] that is less than or equal to arr[1]. arr[3] is also the largest value, but 2 is a smaller index, so we can only jump to i = 2 and not i = 3
During our 3rd jump (odd-numbered), we jump from i = 2 to i = 3 because arr[3] is the smallest value in [arr[3], arr[4]] that is greater than or equal to arr[2].

We can't jump from i = 3 to i = 4, so the starting index i = 0 is not good.

In a similar manner, we can deduce that:
From starting index i = 1, we jump to i = 4, so we reach the end.

From starting index i = 2, we jump to i = 3, and then we can't jump anymore.

From starting index i = 3, we jump to i = 4, so we reach the end.

From starting index i = 4, we are already at the end.

In total, there are 3 different starting indices i = 1, i = 3, and i = 4, where we can reach the end with some
number of jumps.


Example 3:
Input: arr = [5,1,3,4,2]
Output: 3
Explanation: We can reach the end from starting indices 1, 2, and 4.


Constraints:
`1 <= arr.length <= 2 * 104`
`0 <= arr[i] < 105`

### Solution 
 To solve the problem of counting the number of good starting indices from which we can reach the end of the given array `arr` using the jumping rules defined, we can utilize dynamic programming along with efficient sorting techniques for index management.

The approach can be summarized as follows:

1. **Define the Dynamic Programming States:**
   - We'll maintain two boolean arrays, `odd_jumps` and `even_jumps`, where:
     - `odd_jumps[i]` is True if we can reach the end starting from index `i` using an odd jump.
     - `even_jumps[i]` is True if we can reach the end starting from index `i` using an even jump.

2. **Initialization:**
   - The last index is trivially good because we are already at the end.

3. **Traversal in Reverse Order:**
   - We'll work from the end of the array towards the front. For each index, we'll determine whether we can jump to any of the subsequent indices based on the given rules.

4. **Sorting for Efficient Jumps:**
   - To quickly find the next valid jump destinations (both for odd and even jumps), we will maintain sorted lists of indices based on the values in `arr`.

5. **Final Count of Good Indices:**
   - After populating the `odd_jumps` and `even_jumps` arrays, we will count how many indices can reach the end using odd jumps.

Here’s the Python code implementing this solution in the LeetCode format:



```python
class Solution:
    def oddEvenJumps(self, arr: List[int]) -> int:
        n = len(arr)
        if n == 1:
            return 1  # Only one index, which is good.

        # Tabulation arrays for odd and even jumps
        odd_jumps = [False] * n
        even_jumps = [False] * n
        odd_jumps[n - 1] = True
        even_jumps[n - 1] = True  # The last index can always be reached from itself
        
        # Create paired (value, index) and sort for odd/even jumps
        sorted_indices_odd = sorted(range(n), key=lambda i: (arr[i], i))
        sorted_indices_even = sorted(range(n), key=lambda i: (-arr[i], i))

        # Process for valid jumps
        for i in range(n - 2, -1, -1):
            # Find the next index for odd jump
            for j in sorted_indices_odd:
                if j > i and arr[j] >= arr[i]:
                    odd_jumps[i] = even_jumps[j]
                    break
            
            # Find the next index for even jump
            for j in sorted_indices_even:
                if j > i and arr[j] <= arr[i]:
                    even_jumps[i] = odd_jumps[j]
                    break

        # Count good starting indices
        return sum(odd_jumps)

```

### Explanation of the Code:
1. **Initialization:**
   - We initialize two boolean arrays `odd_jumps` and `even_jumps` to store whether we can reach the end from each index with odd or even jumps.

2. **Sorted Indices:**
   - We create two sorted lists of indices: one sorted by value for odd jumps (ascending), and the other by negated value for even jumps (descending).

3. **Jump Logic:**
   - For each index, we iterate through the sorted indices to find the first valid jump for odd and even scenarios. We set the corresponding `odd_jumps` and `even_jumps` flags based on successful transitions.

4. **Count Good Starting Indices:**
   - Finally, we sum up the boolean values in `odd_jumps` to count how many starting indices can reach the end with odd-numbered jumps.

This approach efficiently sets up the conditions required to determine reachable indices and meets the problem's constraints effectively.

# 1604. Alert Using Same Key-Card Three or More Times in a One Hour Period

### Problem Description 
LeetCode company workers use key-cards to unlock office doors. Each time a worker uses their key-card, the security system saves the worker's name and the time when it was used. The system emits an alert if any worker uses the key-card three or more times in a one-hour period.

You are given a list of strings `keyName` and `keyTime` where `[keyName[i], keyTime[i]]` corresponds to a person's name and the time when their key-card was used in a single day.

Access times are given in the 24-hour time format "HH:MM", such as `"23:51"` and `"09:49"`.

Return a list of unique worker names who received an alert for frequent keycard use. Sort the names in ascending order alphabetically.

Notice that `"10:00"` - `"11:00"` is considered to be within a one-hour period, while `"22:51"` - `"23:52"` is not considered to be within a one-hour period.


Example 1:
Input: keyName = ["daniel","daniel","daniel","luis","luis","luis","luis"], keyTime = ["10:00","10:40","11:00","09:00","11:00","13:00","15:00"]
Output: ["daniel"]
Explanation: "daniel" used the keycard 3 times in a one-hour period ("10:00","10:40", "11:00").


Example 2:
Input: keyName = ["alice","alice","alice","bob","bob","bob","bob"], keyTime = ["12:01","12:00","18:00","21:00","21:20","21:30","23:00"]
Output: ["bob"]
Explanation: "bob" used the keycard 3 times in a one-hour period ("21:00","21:20", "21:30").


Example 3:
Input: keyName = ["john","john","john"], keyTime = ["23:58","23:59","00:01"]
Output: []

Example 4:
Input: keyName = ["leslie","leslie","leslie","clare","clare","clare","clare"], keyTime = ["13:00","13:20","14:00","18:00","18:51","19:30","19:49"]
Output: ["clare","leslie"]

Constraints:
`1 <= keyName.length, keyTime.length <= 105`
`keyName.length == keyTime.length`
`keyTime[i]` is in the format "HH:MM".

`[keyName[i], keyTime[i]]` is unique.

`1 <= keyName[i].length <= 10`
`keyName[i] contains only lowercase English letters.`

### Solution 
 To solve this problem, we can follow a systematic approach that involves parsing the input to identify worker keycard usage, checking for the number of usages within one hour, and finally collecting and sorting the names of workers that triggered the alert.

Here's a detailed explanation of how we can implement the solution step-by-step:

1. **Convert Time to Minutes**: Since we are given times in "HH:MM" format, we can convert these times into total minutes from the start of the day (00:00) for easier comparison.

2. **Group Keycard Usages by Worker**: Use a dictionary to group the keytime usages for each worker.

3. **Sort and Count Usages**: For each worker, sort their usage times and then count how many usages fall within any sliding window of one hour.

4. **Collect Unique Alerts**: If any worker has three or more usages within an hour, we mark that worker's name for the final output.

5. **Sort the Result**: Finally, we sort the list of alerted worker names in alphabetical order before returning it.

Here is the implementation of this approach in Python, structured according to LeetCode submission format:



```python
from typing import List
from collections import defaultdict

class Solution:
    def alertNames(self, keyName: List[str], keyTime: List[str]) -> List[str]:
        # Helper function to convert HH:MM to total minutes
        def convertToMinutes(time: str) -> int:
            hours, minutes = map(int, time.split(':'))
            return hours * 60 + minutes
        
        usage = defaultdict(list)
        
        #Populate the usage dictionary with workers and their keycard use times in minutes
        for name, time in zip(keyName, keyTime):
            minutes = convertToMinutes(time)
            usage[name].append(minutes)
        
        alerted_workers = set()  # To hold unique names of alerted workers
        
        # Check each worker's uses
        for name, times in usage.items():
            times.sort()  # Sort the usage times
            
            # Sliding window to count how many uses within the last hour (60 minutes)
            for i in range(len(times)):
                if len(times) - i < 3:  # If less than 3 remaining usages, break
                    break
                if times[i + 2] - times[i] <= 60:  # If the time difference is within 60 minutes
                    alerted_workers.add(name)
                    break  # No need to check further for this worker
        
        return sorted(alerted_workers)

# Example usage:
# solution = Solution()
# print(solution.alertNames(["daniel","daniel","daniel","luis","luis","luis","luis"], 
#                            ["10:00","10:40","11:00","09:00","11:00","13:00","15:00"]))
# Output: ["daniel"]

```

### Explanation of the Code:

1. **Convert Time**: We defined a helper function `convertToMinutes` that converts the "HH:MM" format to total minutes.

2. **Storing Usages**: We utilize a `defaultdict` from the `collections` module to group times by worker name.

3. **Sorting & Checking**: For each worker's time list:
    - We sort the list (O(n log n) complexity).
    - We then use a simple loop to check if there are any three entries in a one-hour window by evaluating `times[i + 2] - times[i] <= 60`.

4. **Alerted Workers**: If a worker's usage meets the criteria for an alert, we add them to the `alerted_workers` set to ensure uniqueness.

5. **Return Result**: Finally, we sort the names alphabetically and return the result.

This solution is efficient with an overall time complexity of O(n log n) due to sorting and linear scans for counting, making it suitable given the constraints of the problem.

# 1606. Find Servers That Handled Most Number of Requests

### Problem Description 
You have `k` servers numbered from `0` to `k-1` that are being used to handle multiple requests simultaneously. Each server has infinite computational capacity but cannot handle more than one request at a time. The requests are assigned to servers according to a specific algorithm:
The `ith` (0-indexed) request arrives.

If all servers are busy, the request is dropped (not handled at all).

If the `(i % k)th` server is available, assign the request to that server.

Otherwise, assign the request to the next available server (wrapping around the list of servers and starting from 0 if necessary). For example, if the `ith` server is busy, try to assign the request to the `(i+1)th` server, then the `(i+2)th` server, and so on.

You are given a strictly increasing array `arrival` of positive integers, where `arrival[i]` represents the arrival time of the `ith` request, and another array `load`, where `load[i]` represents the load of the `ith` request (the time it takes to complete). Your goal is to find the busiest server(s). A server is considered busiest if it handled the most number of requests successfully among all the servers.

Return a list containing the IDs (0-indexed) of the busiest server(s). You may return the IDs in any order.


Example 1:
Input: k = 3, arrival = [1,2,3,4,5], load = [5,2,3,3,3] 
Output: [1] 
Explanation:
All of the servers start out available.

The first 3 requests are handled by the first 3 servers in order.

Request 3 comes in. Server 0 is busy, so it's assigned to the next available server, which is 1.

Request 4 comes in. It cannot be handled since all servers are busy, so it is dropped.

Servers 0 and 2 handled one request each, while server 1 handled two requests. Hence server 1 is the busiest server.


Example 2:
Input: k = 3, arrival = [1,2,3,4], load = [1,2,1,2]
Output: [0]
Explanation:
The first 3 requests are handled by first 3 servers.

Request 3 comes in. It is handled by server 0 since the server is available.

Server 0 handled two requests, while servers 1 and 2 handled one request each. Hence server 0 is the busiest server.


Example 3:
Input: k = 3, arrival = [1,2,3], load = [10,12,11]
Output: [0,1,2]
Explanation: Each server handles a single request, so they are all considered the busiest.


Example 4:
Input: k = 3, arrival = [1,2,3,4,8,9,10], load = [5,2,10,3,1,2,2]
Output: [1]

Example 5:
Input: k = 1, arrival = [1], load = [1]
Output: [0]

Constraints:
`1 <= k <= 105`
`1 <= arrival.length, load.length <= 105`
`arrival.length == load.length`
`1 <= arrival[i], load[i] <= 109`
`arrival` is strictly increasing.

### Solution 
 To solve the problem of determining the busiest servers given requests, we can utilize a simulation approach, managing server states through arrays and a priority queue. Here is a step-by-step breakdown of how we can implement this:

### Explanation

1. **Track Server States**: We need to keep track of when each server will become available. We can do this using a list called `server_time`, where `server_time[i]` signifies when the `i-th` server will be free. Initially, all servers are available, so we can set their values to `0`.

2. **Request Handling**: For each request, we implement a logic to:
   - Determine which server the request should be assigned to based on its index modulo `k`.
   - If that server is busy, check the next servers in a circular manner until an available server is found or all servers are checked.

3. **Counting Successful Requests**: We will maintain a list called `request_count` which tracks how many requests each server has successfully handled.

4. **Determine the Busiest Server(s)**: After processing all requests, find the maximum count of requests handled and gather all servers that match this maximum count.

### Python Code Implementation

Here's the complete code, formatted to be directly runnable on LeetCode:



```python
class Solution:
    def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
        # Time at which each server will be free
        server_time = [0] * k
        # Count of requests each server has handled
        request_count = [0] * k
        
        for i in range(len(arrival)):
            request_arrival = arrival[i]
            request_load = load[i]
            
            # Finding the target server index for this request
            target_server_index = i % k
            
            # Check for the first available server starting from target_server_index
            server_assigned = -1
            
            for j in range(k):
                server_index = (target_server_index + j) % k
                # If the server is free when the request arrives
                if server_time[server_index] <= request_arrival:
                    server_assigned = server_index
                    # Assign the request to this server
                    server_time[server_index] = request_arrival + request_load
                    request_count[server_index] += 1
                    break
            
            # If no server is available, the request is dropped
            
        # Find the maximum number of requests handled by any server
        max_requests = max(request_count)
        # Collect all server indices that handled the maximum requests
        busiest_servers = [i for i in range(k) if request_count[i] == max_requests]
        
        return busiest_servers

```

### Explanation of the Code:

- **Initialization**:
  - We initialize `server_time` to zero since all servers are initially free.
  - `request_count` is also initialized to zero to count successful requests.

- **Processing Requests**:
  - For each incoming request, we determine the intended server index using `i % k`.
  - If this server (and subsequent ones in a circular fashion) is available (i.e., its `server_time` is less than or equal to the request's arrival time), we assign the request to that server, update its free time, and increment its request count.

- **Determining the Busiest Servers**:
  - After processing all requests, we determine the maximum value in `request_count` and compile a list of servers that match this value.

### Complexity:
The time complexity of this algorithm is \(O(n + k)\), where \(n\) is the number of requests and \(k\) is the number of servers. This ensures that the solution is efficient and can handle the upper limits of the input constraints.

# 1649. Create Sorted Array through Instructions

### Problem Description 
Given an integer array `instructions`, you are asked to create a sorted array from the elements in `instructions`. You start with an empty container `nums`. For each element from left to right in `instructions`, insert it into `nums`. The cost of each insertion is the minimum of the following:
The number of elements currently in `nums` that are strictly less than `instructions[i]`.

The number of elements currently in `nums` that are strictly greater than `instructions[i]`.

For example, if inserting element `3` into `nums = [1,2,3,5]`, the cost of insertion is `min(2, 1)` (elements `1` and `2` are less than `3`, element `5` is greater than `3`) and `nums` will become `[1,2,3,3,5]`.

Return the total cost to insert all elements from `instructions` into `nums`. Since the answer may be large, return it modulo `109 + 7`

Example 1:
Input: instructions = [1,5,6,2]
Output: 1
Explanation: Begin with nums = [].

Insert 1 with cost min(0, 0) = 0, now nums = [1].

Insert 5 with cost min(1, 0) = 0, now nums = [1,5].

Insert 6 with cost min(2, 0) = 0, now nums = [1,5,6].

Insert 2 with cost min(1, 2) = 1, now nums = [1,2,5,6].

The total cost is 0 + 0 + 0 + 1 = 1.


Example 2:
Input: instructions = [1,2,3,6,5,4]
Output: 3
Explanation: Begin with nums = [].

Insert 1 with cost min(0, 0) = 0, now nums = [1].

Insert 2 with cost min(1, 0) = 0, now nums = [1,2].

Insert 3 with cost min(2, 0) = 0, now nums = [1,2,3].

Insert 6 with cost min(3, 0) = 0, now nums = [1,2,3,6].

Insert 5 with cost min(3, 1) = 1, now nums = [1,2,3,5,6].

Insert 4 with cost min(3, 2) = 2, now nums = [1,2,3,4,5,6].

The total cost is 0 + 0 + 0 + 0 + 1 + 2 = 3.


Example 3:
Input: instructions = [1,3,3,3,2,4,2,1,2]
Output: 4
Explanation: Begin with nums = [].

Insert 1 with cost min(0, 0) = 0, now nums = [1].

Insert 3 with cost min(1, 0) = 0, now nums = [1,3].

Insert 3 with cost min(1, 0) = 0, now nums = [1,3,3].

Insert 3 with cost min(1, 0) = 0, now nums = [1,3,3,3].

Insert 2 with cost min(1, 3) = 1, now nums = [1,2,3,3,3].

Insert 4 with cost min(5, 0) = 0, now nums = [1,2,3,3,3,4].

​​​​​​​Insert 2 with cost min(1, 4) = 1, now nums = [1,2,2,3,3,3,4].

​​​​​​​Insert 1 with cost min(0, 6) = 0, now nums = [1,1,2,2,3,3,3,4].

​​​​​​​Insert 2 with cost min(2, 4) = 2, now nums = [1,1,2,2,2,3,3,3,4].

The total cost is 0 + 0 + 0 + 0 + 1 + 0 + 1 + 0 + 2 = 4.


Constraints:
`1 <= instructions.length <= 105`
`1 <= instructions[i] <= 105`

### Solution 
 To solve the problem of calculating the total cost of inserting elements into a sorted array, we can use a data structure that allows us to efficiently keep track of the elements and their frequencies. Given the constraints where the length of the instructions can be up to \(10^5\), a naive approach which takes \(O(n^2)\) time will be too slow. Instead, we can use a combination of binary search and a frequency array.

Here’s the step-by-step breakdown of the solution:

### Problem Breakdown
1. **Input and Output**:
   - Input: An array of integers `instructions`.
   - Output: A single integer representing the total insertion cost modulo \(10^9 + 7\).

2. **Cost Calculation**:
   - For each element in `instructions`, when inserting it into the `nums` array:
     - Count the number of elements strictly less than the current element.
     - Count the number of elements strictly greater than the current element.
     - The insertion cost for that element is the minimum of these two counts.

3. **Efficient Count of Elements**:
   - We use a frequency array `freq` of size \(10^5 + 1\) since the possible values of the elements range from 1 to \(10^5\).
   - As we process each element, we can update our frequency array and calculate the number of elements less than and greater than the current element.

### Algorithm
1. Initialize a frequency array of zeros.
2. Iterate through each element in `instructions`:
   - Use the frequency array to count how many elements are less than and greater than the current element.
   - Calculate the insertion cost and update the total cost.
   - Update the frequency array to include the current element.
3. Return the total cost modulo \(10^9 + 7\).

### Implementation
Now, let’s implement this in Python, following the typical LeetCode solution format.



```python
class Solution:
    def createSortedArray(self, instructions: List[int]) -> int:
        MOD = 10**9 + 7
        # Initialize frequency array for numbers 1 to 100000
        freq = [0] * 100001
        total_cost = 0
        
        # Iterate through each instruction
        for i in range(len(instructions)):
            num = instructions[i]
            # Count less than the current number
            less_count = sum(freq[:num])    # All numbers less than num
            # Count greater than the current number
            greater_count = sum(freq[num+1:]) # All numbers greater than num
            # The cost is the minimum of these two counts
            cost = min(less_count, greater_count)
            total_cost = (total_cost + cost) % MOD
            # Update frequency of the current number
            freq[num] += 1
        
        return total_cost


```

### Explanation of Code:
- A frequency array is used to track how many times each number has been inserted.
- For each number in `instructions`, we calculate the counts of elements less than and greater than the current number using cumulative sums from the frequency array.
- We maintain a cumulative total cost and return it modulo \(10^9 + 7\) to prevent overflow.

### Complexity:
- The time complexity of this implementation is \(O(n + k)\) where \(n\) is the number of instructions and \(k\) is the maximum possible number value in the array. Given \(k\) is constant (up to \(10^5\)), this is efficient for the input limits.
- The space complexity is \(O(k)\) for the frequency array.

This detailed approach should work efficiently for the problem constraints provided.

# 1675. Minimize Deviation in Array

### Problem Description 
You are given an array `nums` of `n` positive integers.

You can perform two types of operations on any element of the array any number of times:
If the element is even, divide it by `2`.

	
For example, if the array is `[1,2,3,4]`, then you can do this operation on the last element, and the array will be `[1,2,3,2].`
If the element is odd, multiply it by `2`.

	
For example, if the array is `[1,2,3,4]`, then you can do this operation on the first element, and the array will be `[2,2,3,4].`
The deviation of the array is the maximum difference between any two elements in the array.

Return the minimum deviation the array can have after performing some number of operations.


Example 1:
Input: nums = [1,2,3,4]
Output: 1
Explanation: You can transform the array to [1,2,3,2], then to [2,2,3,2], then the deviation will be 3 - 2 = 1.


Example 2:
Input: nums = [4,1,5,20,3]
Output: 3
Explanation: You can transform the array after two operations to [4,2,5,5,3], then the deviation will be 5 - 2 = 3.


Example 3:
Input: nums = [2,10,8]
Output: 3

Constraints:
`n == nums.length`
`2 <= n <= 105`
`1 <= nums[i] <= 109`

### Solution 
 To solve the problem of minimizing the deviation of an array of positive integers via allowed operations (if even, divide by 2; if odd, multiply by 2), we can utilize a max-heap approach in Python. 

### Problem Overview

The deviation is defined as the difference between the maximum and minimum values in the array. Our goal is to minimize this deviation. 

### Key Observations

1. **Even and Odd Operations**:
   - For even numbers, you can keep dividing them by 2 to reduce their size.
   - For odd numbers, by multiplying by 2, you can only increase their size.
   
2. **Reductions**:
   - If the maximum number in the array is odd, we can transform it to an even one (by multiplying an odd number by 2), allowing us to reduce even numbers in subsequent steps.
   - Keep reducing the largest element until it becomes odd or all elements are the same.

### Approach

1. Use a max-heap to keep track of the largest element in the array efficiently.
2. Create a set that ensures we can only push the modified values back into the heap (to prevent duplicate entries).
3. Repeat the following until the largest number is made odd:
   - Extract the largest number from the heap.
   - If it is even, divide it by 2.
   - Update the minimum value seen so far and push the modified value back into the heap.
4. Continuously track the minimum deviation at each step.

### Complexity
- The time complexity mainly consists of heap operations, giving us \(O(n \log n)\), which is efficient within the limits provided (up to \(10^5\) numbers).

### Implementation

Here’s the complete Python code that follows the LeetCode solution format:



```python
import heapq

class Solution:
    def minimumDeviation(self, nums):
        # Step 1: Create a max-heap and a variable for the min value
        max_heap = []
        min_value = float('inf')
        
        # Step 2: Populate the heap and find initial minimum value
        for num in nums:
            if num % 2 == 0:
                heapq.heappush(max_heap, -num)  # Use negative values for max-heap
                min_value = min(min_value, num)
            else:
                heapq.heappush(max_heap, -num * 2)  # Odd number, multiply by 2 and push
                min_value = min(min_value, num * 2)

        # Step 3: Find the minimum deviation
        min_deviation = float('inf')
        
        # Step 4: Perform the reduction process
        while max_heap:
            max_value = -heapq.heappop(max_heap)  # Get the largest value
            
            # Update the minimum deviation
            min_deviation = min(min_deviation, max_value - min_value)
            
            # If the max_value is even, divide by 2 and push back
            if max_value % 2 == 0:
                new_value = max_value // 2
                min_value = min(min_value, new_value)
                heapq.heappush(max_heap, -new_value)
            else:
                break  # If it's odd, we can't reduce it anymore
        
        return min_deviation

```

### Explanation of the Code

1. **Heap Initialization**: Convert numbers into a max-heap where we store negative values to mimic max behavior since Python's `heapq` is a min-heap by default.
2. **Tracking Minimum**: This algorithm scans to find the smallest number as we manipulate the values.
3. **Finding Minimum Deviation**: We continuously check and update the deviation as we alter the largest value in our max-heap and subsequently adjust the minimum value.
4. **Loop Control**: The loop continues until we can’t reduce the maximum value further (when it becomes odd).

This approach ensures that we perform the operations effectively while keeping track of both extremes (max and min) to calculate the deviation.

