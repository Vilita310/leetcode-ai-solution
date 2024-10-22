# 470. Implement Rand10() Using Rand7()

### Problem Description 
Given the API `rand7()` that generates a uniform random integer in the range `[1, 7]`, write a function `rand10()` that generates a uniform random integer in the range `[1, 10]`. You can only call the API `rand7()`, and you shouldn't call any other API. Please do not use a language's built-in random API.

Each test case will have one internal argument `n`, the number of times that your implemented function `rand10()` will be called while testing. Note that this is not an argument passed to `rand10()`.

Follow up:
What is the expected value for the number of calls to `rand7()` function?
Could you minimize the number of calls to `rand7()`?

Example 1:
Input: n = 1
Output: [2]

Example 2:
Input: n = 2
Output: [2,8]

Example 3:
Input: n = 3
Output: [3,8,10]

Constraints:
`1 <= n <= 105`

### Solution 
 To solve the problem of generating a uniform random integer from 1 to 10 using the provided `rand7()` function, we need to create a custom algorithm that effectively maps the outputs of `rand7()` to the desired range without introducing bias.

### Problem Breakdown
`rand7()` will return integers uniformly from 1 to 7. In order to generate numbers from 1 to 10, we can utilize a method that effectively combines the outputs of multiple calls to `rand7()`.

1. **Understanding the Output Space**:
   - `rand7()` generates integers in the range [1, 7]. Thus, the total number of outcomes is 7.
   - We can generate a larger range by combining two calls to `rand7()`. This will give us `7 * 7 = 49` possible outcomes which can be indexed from 1 to 49.
   - From these 49 outcomes, we can derive integers from 1 to 10 by using modulo operations.

2. **Mapping Outcomes**:
   - We can generate numbers 1 through 10 using outcomes from 49 numbers.
   - To do this effectively, we can take results from two calls to `rand7()`, compute a combined result, and use only results that provide a fair and uniform distribution for the range of 1 to 10.
   - If the result exceeds the maximum required (which for 10, can be 40), we can simply retry until we get a valid outcome.

### Implementation Approach
Using the above logic, we can implement `rand10()` which repeatedly calls `rand7()` until we get a valid random integer between 1 and 10.

### Implementation
Here's the Python code that encapsulates this logic, ready to run on LeetCode:



```python
import random

def rand7():
    return random.randint(1, 7)

def rand10():
    while True:
        # Get two random values from rand7
        row = rand7() - 1  # Convert to 0-indexed
        col = rand7() - 1  # Convert to 0-indexed
        idx = row * 7 + col  # This generates numbers from 0 to 48
        
        # Only use the first 40 as those can evenly provide numbers from 1 to 10
        if idx < 40:
            return (idx % 10) + 1  # Maps 0-39 to 1-10

# Below code handles the multiple calls to rand10()
def generate_random_numbers(n):
    return [rand10() for _ in range(n)]

```

### Explanation of the Code
1. **`rand7()`**: This function simulates the API provided. In an environment like LeetCode, this would be predefined, but for testing, we use Python's built-in `randint`.
  
2. **`rand10()`** function:
   - A while loop is used to continuously generate pairs of numbers from `rand7()`.
   - Compute a single index by combining the two outputs (`row * 7 + col`), which will range from 0 to 48 (inclusive).
   - Only accept indices less than 40 to ensure uniform distribution:
     - These indices can be evenly divided into 10 outcomes without bias.
   - Finally, we convert the index back to the 1-10 range using `(idx % 10) + 1`.

3. **`generate_random_numbers(n)`**: This helper function is not required by the problem but is included for completeness to collect `n` outputs and test the behavior of `rand10()`.

### Follow-up on Expected Call Count
The expected number of calls to `rand7()` can be analyzed based on the probability of hitting valid outcomes (1-40). On average, it will take about \( \frac{49}{40} \) calls to achieve a valid outcome for `rand10()`, which is approximately 1.225 calls per successful generation of a uniform random integer in the range [1, 10].

# 478. Generate Random Point in a Circle

### Problem Description 
Given the radius and the position of the center of a circle, implement the function `randPoint` which generates a uniform random point inside the circle.

Implement the `Solution` class:
`Solution(double radius, double x_center, double y_center)` initializes the object with the radius of the circle `radius` and the position of the center `(x_center, y_center)`.

`randPoint()` returns a random point inside the circle. A point on the circumference of the circle is considered to be in the circle. The answer is returned as an array `[x, y]`.


Example 1:
Input
["Solution", "randPoint", "randPoint", "randPoint"]
[[1.0, 0.0, 0.0], [], [], []]
Output
[null, [-0.02493, -0.38077], [0.82314, 0.38945], [0.36572, 0.17248]]
Explanation
Solution solution = new Solution(1.0, 0.0, 0.0);
solution.randPoint(); // return [-0.02493, -0.38077]
solution.randPoint(); // return [0.82314, 0.38945]
solution.randPoint(); // return [0.36572, 0.17248]

Constraints:
`0 < radius <= 108`
`-107 <= x_center, y_center <= 107`
At most `3 * 104` calls will be made to `randPoint`.

### Solution 
 To solve this problem, we need to implement a class named `Solution` that allows us to initialize a circle with a specified radius and center, and then generate random points uniformly within that circle.

### Steps to Solve the Problem:

1. **Initialization**:
   - Store the attributes of the circle: `radius`, `x_center`, and `y_center`.

2. **Random Point Generation**:
   - To ensure uniform distribution of points within the circle, we can use polar coordinates. We'll generate a random angle \( \theta \) and a random radius \( r \).
   - Randomly choose a radius using \( r = \sqrt{random} \times \text{radius} \). This adjustment is necessary because a uniform distribution in Cartesian coordinates would lead to clustering of points near the center.
   - Convert polar coordinates to Cartesian coordinates using:
     - \( x = x_{\text{center}} + r \cdot \cos(\theta) \)
     - \( y = y_{\text{center}} + r \cdot \sin(\theta) \)

3. **Return Format**:
   - The `randPoint()` function should return the coordinates as a list `[x, y]`.

### Detailed Python Code Implementation

Here's the Python code that follows the above plan and adheres to the LeetCode requirements:



```python
import random
import math

class Solution:

    def __init__(self, radius: float, x_center: float, y_center: float):
        """
        Initializes the object with the radius of the circle and the position of the center.
        """
        self.radius = radius
        self.x_center = x_center
        self.y_center = y_center

    def randPoint(self) -> List[float]:
        """
        Returns a random point inside the circle (inclusive of the circumference).
        """
        # Generate a random angle
        theta = random.uniform(0, 2 * math.pi)
        # Generate a random radius, ensuring uniform distribution
        r = math.sqrt(random.uniform(0, 1)) * self.radius
        
        # Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
        x = self.x_center + r * math.cos(theta)
        y = self.y_center + r * math.sin(theta)
        
        return [x, y]


```

### Explanation of the Code:

1. **Class Initialization**: 
   - The `__init__` method is called with the circle's radius and center coordinates, which are stored as instance variables.

2. **Generating a Random Point**:
   - The `randPoint` method performs the following:
     - Generates a random angle \( \theta \) between 0 and \( 2\pi \).
     - Generates a random value from a uniform distribution [0, 1), and takes its square root, multiplying by the radius to get a valid radius \( r \) within the circle.
     - Converts the polar coordinates (using cosine and sine) to Cartesian coordinates by calculating \( x \) and \( y \).

3. **Return Value**: The function returns the computed coordinates as a list.

### Note:
- When implementing this solution in an environment like LeetCode, make sure to import the necessary libraries at the start. The code provided is ready to run as is, given the correct context and the appropriate imports for the types (like `List` from `typing`).

# 497. Random Point in Non-overlapping Rectangles

### Problem Description 
Given a list of non-overlapping axis-aligned rectangles `rects`, write a function `pick` which randomly and uniformily picks an integer point in the space covered by the rectangles.

Note:
An integer point is a point that has integer coordinates. 
A point on the perimeter of a rectangle is included in the space covered by the rectangles. 
`i`th rectangle = `rects[i]` = `[x1,y1,x2,y2]`, where `[x1, y1]` are the integer coordinates of the bottom-left corner, and `[x2, y2]` are the integer coordinates of the top-right corner.

length and width of each rectangle does not exceed `2000`.

`1 <= rects.length <= 100`
`pick` return a point as an array of integer coordinates `[p_x, p_y]`
`pick` is called at most `10000` times.


Example 1:
Input: 
["Solution","pick","pick","pick"]
[[[[1,1,5,5]]],[],[],[]]
Output: 
[null,[4,1],[4,1],[3,3]]

Example 2:
Input: 
["Solution","pick","pick","pick","pick","pick"]
[[[[-2,-2,-1,-1],[1,0,3,0]]],[],[],[],[],[]]
Output: 
[null,[-1,-2],[2,0],[-2,-1],[3,0],[-2,-2]]
Explanation of Input Syntax:
The input is two lists: the subroutines called and their arguments. `Solution`'s constructor has one argument, the array of rectangles `rects`. `pick` has no arguments. Arguments are always wrapped with a list, even if there aren't any.

### Solution 
 To solve the problem of randomly selecting a point from a given list of non-overlapping axis-aligned rectangles, we need to create a class `Solution` that initializes with a list of rectangles and provides a `pick` method to return a random point from the covered area of these rectangles. 

Here’s how we can approach the problem:

1. **Understanding Rectangles**: Each rectangle is defined by two corners: the bottom-left corner `(x1, y1)` and the top-right corner `(x2, y2)`. The area covered by each rectangle consists of all integer points `(x, y)` such that `x1 <= x <= x2` and `y1 <= y <= y2`.

2. **Total Area Calculation**: To ensure that the random choice is uniform, we need to know the total area covered by all rectangles. This is simply the sum of the areas of each rectangle, which can be calculated as:
   \[
   \text{Area of rectangle} = (x2 - x1 + 1) \times (y2 - y1 + 1)
   \]

3. **Picking Points**: Once we have the total area, we can generate a random number that determines which rectangle to pick from. We can use the prefix sum of areas to determine which rectangle a random value points to. Once a rectangle is chosen, we randomly pick a point within its bounds.

4. **Uniform Random Generation**: Use Python's `random.randint` for integer selection and `random.choices` to select a rectangle based on their respective areas.

5. **Implementation**: The implementation includes a constructor that initializes the rectangles and precomputes the areas. The `pick` function will handle the random selection.

Here is the full implementation:



```python
import random

class Solution:

    def __init__(self, rects):
        self.rects = rects
        self.areas = []
        total_area = 0
        
        # Precompute the areas and prefix sums
        for x1, y1, x2, y2 in rects:
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            total_area += area
            self.areas.append(total_area)  # Keep a running total area
        
        self.total_area = total_area

    def pick(self):
        # Pick a random area value between 0 and total_area - 1
        target = random.randint(0, self.total_area - 1)
        
        # Determine which rectangle to select based on the target area
        rect_index = 0
        for i, area in enumerate(self.areas):
            if target < area:
                rect_index = i
                break
        
        # Get the selected rectangle
        x1, y1, x2, y2 = self.rects[rect_index]
        
        # Pick a random point within the selected rectangle
        px = random.randint(x1, x2)
        py = random.randint(y1, y2)

        return [px, py]

```

### Explanation of Implementation:

- **Initialization (`__init__`)**: We calculate the cumulative area for the rectangles which allows us to quickly determine which rectangle corresponds to a given random number during the `pick` operation.
- **Area Calculation**: Each rectangle's area is calculated and added to the cumulative list `self.areas`.
- **Random Point Selection `pick()`**: We generate a random integer between `0` and `total_area - 1`. By iterating over the `areas`, we find which rectangle corresponds to this random number.
- Once we find the rectangle, we generate a random coordinate within the rectangle using the bounds `x1, y1` to `x2, y2`.

This module can directly be run in a LeetCode environment where class-based implementations and random outputs are acceptable.

# 519. Random Flip Matrix

### Problem Description 
You are given the number of rows `n_rows` and number of columns `n_cols` of a 2D binary matrix where all values are initially 0. Write a function `flip` which chooses a 0 value uniformly at random, changes it to 1, and then returns the position `[row.id, col.id]` of that value. Also, write a function `reset` which sets all values back to 0. Try to minimize the number of calls to system's Math.random() and optimize the time and space complexity.

Note:
`1 <= n_rows, n_cols <= 10000`
`0 <= row.id < n_rows` and `0 <= col.id < n_cols`
`flip` will not be called when the matrix has no 0 values left.

the total number of calls to `flip` and `reset` will not exceed 1000.


Example 1:
Input: 
["Solution","flip","flip","flip","flip"]
[[2,3],[],[],[],[]]
Output: [null,[0,1],[1,2],[1,0],[1,1]]

Example 2:
Input: 
["Solution","flip","flip","reset","flip"]
[[1,2],[],[],[],[]]
Output: [null,[0,0],[0,1],null,[0,0]]
Explanation of Input Syntax:
The input is two lists: the subroutines called and their arguments. `Solution`'s constructor has two arguments, `n_rows` and `n_cols`. `flip` and `reset` have no arguments. Arguments are always wrapped with a list, even if there aren't any.

### Solution 
 To solve the problem of randomly flipping bits in a 2D binary matrix, we can use an efficient approach leveraging a dictionary to keep track of the indices of the positions the `0`s can occupy (flippable positions). This allows for direct access to empty spots while also keeping our space usage reasonable.

Here's how we will structure the solution:
1. **Initialization**: Create a list of all possible positions in the `n_rows` x `n_cols` matrix. We’ll use a dictionary to track which of those positions are still available for flipping.
2. **Flip Method**: Randomly pick an index from our list of available positions, swap that position with the last available position (to keep the list compact), change the value in the matrix to `1`, and return the selected position.
3. **Reset Method**: Clear all flip states by recreating the list of available positions.

### Python Code



```python
import random

class Solution:

    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.total_elements = n_rows * n_cols
        # We use a list to keep track of available positions
        self.available = list(range(self.total_elements))

    def flip(self) -> list:
        # Randomly select one of the available positions
        rand_index = random.randint(0, len(self.available) - 1)
        # Pop the position from the available list
        pos = self.available.pop(rand_index)
        # Calculate row and column from the flat index
        row = pos // self.n_cols
        col = pos % self.n_cols
        return [row, col]

    def reset(self) -> None:
        # When resetting, we re-populate the available positions
        self.available = list(range(self.total_elements))

# Example of how to use the Solution class:
# sol = Solution(2, 3)
# print(sol.flip())  # Flips and returns a random position
# print(sol.flip())  # Flips another random position
# sol.reset()        # Resets the matrix
# print(sol.flip())  # Flips again after reset

```

### Explanation

1. **Initialization**:
   - We create a list `self.available`, which holds indices of all elements in a flattened version of the matrix (from `0` to `n_rows * n_cols - 1`).
   - The two parameters `n_rows` and `n_cols` are stored to assist in calculating the 2D indices.

2. **Flip Method**:
   - Generate a random index from the range of available positions.
   - Use this index to access the flattened position, convert it back to 2D indices (`row`, `col`).
   - Pop the used position from the `available` list to ensure it’s no longer a candidate for flipping.

3. **Reset Method**:
   - Simply reassign `self.available` to be a complete list of indices again, effectively restoring the state of all `0`s in the matrix.

### Complexity
- **Time Complexity** for `flip`: O(1) due to the use of a list and the pop operation (which is average O(1) for the end of the list).
- **Time Complexity** for `reset`: O(n) to create a new list of all positions.
- **Space Complexity**: O(n) due to storing indices in `available`.

With these implementations, we keep our operations efficient and meet the problem's constraints effectively.

# 528. Random Pick with Weight

### Problem Description 
You are given an array of positive integers `w` where `w[i]` describes the weight of `i``th` index (0-indexed).

We need to call the function `pickIndex()` which randomly returns an integer in the range `[0, w.length - 1]`. `pickIndex()` should return the integer proportional to its weight in the `w` array. For example, for `w = [1, 3]`, the probability of picking the index `0` is `1 / (1 + 3) = 0.25` (i.e 25%) while the probability of picking the index `1` is `3 / (1 + 3) = 0.75` (i.e 75%).

More formally, the probability of picking index `i` is `w[i] / sum(w)`.


Example 1:
Input
["Solution","pickIndex"]
[[[1]],[]]
Output
[null,0]
Explanation
Solution solution = new Solution([1]);
solution.pickIndex(); // return 0. Since there is only one single element on the array the only option is to return the first element.


Example 2:
Input
["Solution","pickIndex","pickIndex","pickIndex","pickIndex","pickIndex"]
[[[1,3]],[],[],[],[],[]]
Output
[null,1,1,1,1,0]
Explanation
Solution solution = new Solution([1, 3]);
solution.pickIndex(); // return 1. It's returning the second element (index = 1) that has probability of 3/4.

solution.pickIndex(); // return 1
solution.pickIndex(); // return 1
solution.pickIndex(); // return 1
solution.pickIndex(); // return 0. It's returning the first element (index = 0) that has probability of 1/4.

Since this is a randomization problem, multiple answers are allowed so the following outputs can be considered correct :
[null,1,1,1,1,0]
[null,1,1,1,1,1]
[null,1,1,1,0,0]
[null,1,1,1,0,1]
[null,1,0,1,0,0]
......

and so on.


Constraints:
`1 <= w.length <= 10000`
`1 <= w[i] <= 10^5`
`pickIndex` will be called at most `10000` times.

### Solution 
 To solve the given problem of picking an index based on the weights provided in the array, we can achieve the desired functionality through the use of prefix sums and binary search.

### Explanation:

1. **Understanding Weights and Prefix Sums**:
   - Each index has a certain weight. These weights can be interpreted as intervals on the number line.
   - The idea is to create a prefix sum array where each position `i` contains the cumulative sum of the weights up to index `i`. This enables us to easily find sections of the weight distribution.
   - For an array `w` of weights, the prefix sum can be computed such that `prefix[i] = w[0] + w[1] + ... + w[i]`.

2. **Calculating the Target**:
   - In the `pickIndex` function, we generate a random integer between `1` and the total sum of weights (inclusive). This integer represents a point on the cumulative weight distribution.

3. **Finding the Correct Index**:
   - Using binary search, we can quickly find the index in the prefix sum array that corresponds to the randomly selected value. The chosen index will be the first position in the prefix sum that is greater than or equal to the random number.

4. **Implementation**:
   - We create a class `Solution` with an initializer to compute the prefix sums and a method `pickIndex` to carry out the picking operation.

Here is how the code looks:



```python
import random
import bisect

class Solution:
    def __init__(self, w: List[int]):
        # Step 1: Initialize the prefix sums array
        self.prefix_sum = []
        total_sum = 0
        
        for weight in w:
            total_sum += weight
            self.prefix_sum.append(total_sum)
        
        # Store the total sum for random selection
        self.total_sum = total_sum
    
    def pickIndex(self) -> int:
        # Step 2: Generate a random number in the range of [1, total_sum]
        target = random.randint(1, self.total_sum)
        
        # Step 3: Use binary search to find the appropriate index
        return bisect.bisect_left(self.prefix_sum, target)


```

### Detailed Breakdown of the Code:

1. **`__init__` Method**:
   - We initialize the prefix sums list and calculate the total sum of the weights. The prefix sums are constructed iteratively by adding weights cumulatively.
   
2. **`pickIndex` Method**:
   - A random integer is generated within the range of the total weight using `random.randint`.
   - We then use `bisect.bisect_left` to perform a binary search on the `prefix_sum` array, which efficiently finds the index where the random target would fit in the sorted prefix sums.

### Performance:
- The operations of building the prefix sum array takes O(n) time, and each call to `pickIndex` is handled in O(log n) due to binary search. The space complexity is O(n) due to the storage of the prefix sums.

This implementation meets the problem constraints and efficiently handles the possible number of calls to `pickIndex`. The usage of randomization and binary search makes it both robust and performant for large input sizes within the specified limits.

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
 To tackle the problem of generating a uniform random integer from the range `[0, N)` that is not in the blacklist `B`, we can utilize a combination of data structures and random number generation techniques to ensure both efficiency and uniformity. Here's how you can effectively solve this problem:

### Key Insights:

1. **Handling the Blacklist**: 
   - Since the blacklist can contain up to 100,000 unique integers, we can store these integers in a set for O(1) lookup time.
   
2. **Mapping to a Valid Range**:
   - The goal is to generate random numbers in the range `[0, N)` and ensure they do not belong to the blacklist `B`. 
   - To do this, we can create a mapping from the valid range to the blacklist. Specifically, the count of valid numbers in the range can be adjusted to exclude the numbers in `B`.

3. **Random Number Generation**:
   - Generate a random number in the adjusted range for valid numbers. This can be done efficiently with the use of Python's built-in `random.randint()` or `random.choice()`.

### Implementation:



```python
import random

class Solution:
    def __init__(self, N: int, B: list[int]):
        self.N = N
        self.blacklist = set(B)  # Store blacklist as a set for O(1) lookup
        self.valid_count = N - len(B)  # Count of valid numbers
        self.mapped_blacklist = {}  # To map a blacklisted number to a valid number
        
        # Prepare mapping for blacklisted numbers that fall in the valid range
        current_valid = N - 1  # Start from the end of valid numbers
        for b in self.blacklist:
            if b < self.valid_count:  # Only map if the blacklisted number is within the valid range
                while current_valid in self.blacklist:
                    current_valid -= 1  # Move to the next valid number
                self.mapped_blacklist[b] = current_valid  # Map to the valid number
                current_valid -= 1  # Skip this one for future mappings

    def pick(self) -> int:
        rand_index = random.randint(0, self.valid_count - 1)  # Pick a random index in the valid range
        if rand_index in self.mapped_blacklist:
            return self.mapped_blacklist[rand_index]  # Return the mapped value if it exists
        return rand_index  # Otherwise, return the index itself as it's valid

# Example usage:
# sol = Solution(4, [2])
# print(sol.pick())  # This should return a random number from [0, 4) excluding 2

```

### Explanation of the Code:

1. **Constructor `__init__`**:
   - Store the maximum number `N` and convert the blacklist into a set.
   - Calculate the number of valid integers by subtracting the size of the blacklist from `N`.
   - Create a mapping from blacklisted values to valid values if they would have been selected. This is done so that the selected random index can be adjusted properly.

2. **Method `pick()`**:
   - Generate a random index in the range of valid indices `[0, valid_count - 1]`.
   - If the generated index is in the `mapped_blacklist`, return the appropriate valid number that was mapped to that index.
   - If the index is not in the blacklist, just return the index itself.

### Efficiency:
- The initialization process is O(M * log M) due to sorting and the mapping (where M is the size of the blacklist).
- Each pick operation is O(1) in terms of providing output, thanks to the precomputed mappings. 

This implementation is efficient and addresses the problem constraints effectively by minimizing calls to random number generation and ensuring uniform distribution amongst valid integers.

