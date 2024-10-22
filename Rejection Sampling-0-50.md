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
 To solve the problem of generating a uniform random integer in the range of `[1, 10]` using the given API `rand7()`, we need to utilize the randomness provided by `rand7()` effectively. The idea is to construct our own random number generator for the range `[1, 10]` by combining several calls to `rand7()`.

### Strategy
We can create a mechanism to derive values between 1 and 10 using `rand7()`. Since `rand7()` generates values uniformly between 1 and 7, we can combine the outputs of two or more calls to develop random numbers in a larger range. Here's a breakdown of the steps to achieve this:

1. **Mapping from 7 to 49:** 
   By calling `rand7()` twice, we can create a number in the range `[1, 49]`. This is done by treating the first call as the tens place and the second call as the units place:
   \[
   \text{val} = (rand7() - 1) \times 7 + (rand7() - 1) + 1
   \]
   This gives us a uniform distribution of integer values from 1 to 49.

2. **Selecting values between 1 and 10:** 
   Once we have a value in the range `[1, 49]`, we can use the modulo operation to map it to our desired range:
   - We are interested in the values 1 through 40, as each complete group of 4 corresponds to a uniformly chosen number between 1 and 10.
   - If `val` falls between 1 and 40, we can convert it to a number between 1 and 10 using:
   \[
   result = (val - 1) \% 10 + 1
   \]
   - If `val` is greater than 40, we'll simply discard it and rerun the entire process to ensure uniform randomness.

### Implementation
Here's how we can implement this in Python.



```python
import random

def rand7():
    return random.randint(1, 7)

def rand10():
    while True:
        # Generate a random number between 1 and 49
        num = (rand7() - 1) * 7 + rand7()
        # We only use numbers from 1 to 40
        if num <= 40:
            return (num - 1) % 10 + 1

# Example Test Cases
def test_rand10(n):
    results = []
    for _ in range(n):
        results.append(rand10())
    return results

# Test example
n = 5  # Example test case
print(test_rand10(n))  # Generates a list of random numbers between 1 and 10

```

### Explanation of the Code:
1. **rand7:** This is a placeholder function representing the given API. In an actual implementation, this function will provide random integers from 1 to 7.
2. **rand10:** The main function that generates random integers between 1 and 10. It continuously generates values until it finds one in the range 1 to 40.
3. **test_rand10:** A helper function to test `rand10` by calling it `n` times and returning the results.

### Follow-Up on Calls to `rand7()`
In the randomized method, the expected number of calls to `rand7()` is derived from the probability of generating a valid number. Each call sequence `(rand7() * 7 + rand7())` forms a geometric distribution, and since we might reject numbers greater than 40, the expected number of calls to `rand7()` can be calculated more precisely through probability calculations, but generally, it is close to 2.5 calls per `rand10()` execution.

This approach efficiently minimizes the number of calls to `rand7()` while ensuring that the generated values are uniformly random and appropriately distributed.

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
 To solve the problem of generating a uniform random point inside a circle defined by a radius and center coordinates, we can break down the task as follows:

1. **Initialization**: We need a class `Solution` which initializes with the radius and center of the circle.
2. **Random Point Generation**: The `randPoint` method should generate a random point such that all points in the circle, including the boundary, are uniformly distributed.

### How to Generate a Point Uniformly

The method to generate a point uniformly within a circle involves the following steps:
- Generate a random angle \(\theta\) uniformly from \(0\) to \(2\pi\).
- Generate a random radius \(r\) uniformly from \(0\) to the circle's radius. However, to ensure uniform distribution, we should actually sample from the area of the circle using \(\sqrt{random())\), where `random()` is uniformly distributed from \(0\) to \(1\), because the area increases with the square of the radius.
- Finally, calculate the \(x\) and \(y\) coordinates from the polar coordinates using \(x = x_{center} + r \cdot \cos(\theta)\) and \(y = y_{center} + r \cdot \sin(\theta)\).

### Implementation

Here’s how we can implement the `Solution` class in Python, formatted to be executed on LeetCode:



```python
import random
import math

class Solution:

    def __init__(self, radius: float, x_center: float, y_center: float):
        self.radius = radius
        self.x_center = x_center
        self.y_center = y_center

    def randPoint(self) -> list[float]:
        # Generate a random angle
        theta = random.uniform(0, 2 * math.pi)
        
        # Generate a random radius with uniform distribution in area
        r = self.radius * math.sqrt(random.uniform(0, 1))
        
        # Calculate x and y coordinates
        x = self.x_center + r * math.cos(theta)
        y = self.y_center + r * math.sin(theta)
        
        return [x, y]


```

### Explanation of Code:
- **Import Statements**: We import the `random` module for random number generation and the `math` module for mathematical functions like `sqrt` and `cos/sin`.
- **Constructor `__init__`**: This initializes the instance variables with the provided radius and center coordinates.
- **Method `randPoint`**: 
  - We first generate a random angle `theta` uniformly distributed between \(0\) and \(2\pi\).
  - For the radius, we sample from the uniform distribution, multiply by the square root of the random value to ensure uniform distribution over the area of the circle.
  - Finally, using polar coordinates, we compute the \(x\) and \(y\) coordinates and return them as a list.

### Complexity:
- **Time Complexity**: \(O(1)\) per call to `randPoint` since both random generation and mathematical operations are constant time.
- **Space Complexity**: \(O(1)\) as we are using a fixed amount of additional space regardless of the input size.

This implementation adheres to the LeetCode format ensuring it can be tested directly on their platform.

