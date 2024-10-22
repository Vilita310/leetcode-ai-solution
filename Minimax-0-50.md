# 292. Nim Game

### Problem Description 
You are playing the following Nim Game with your friend:
Initially, there is a heap of stones on the table.

You and your friend will alternate taking turns, and you go first.

On each turn, the person whose turn it is will remove 1 to 3 stones from the heap.

The one who removes the last stone is the winner.

Given `n`, the number of stones in the heap, return `true` if you can win the game assuming both you and your friend play optimally, otherwise return `false`.


Example 1:
Input: n = 4
Output: false
Explanation: These are the possible outcomes:
1. You remove 1 stone. Your friend removes 3 stones, including the last stone. Your friend wins.

2. You remove 2 stones. Your friend removes 2 stones, including the last stone. Your friend wins.

3. You remove 3 stones. Your friend removes the last stone. Your friend wins.

In all outcomes, your friend wins.


Example 2:
Input: n = 1
Output: true

Example 3:
Input: n = 2
Output: true

Constraints:
`1 <= n <= 231 - 1`

### Solution 
 To solve the Nim Game problem, we need to analyze the optimal strategies for both players. The key is to determine whether there are configurations of stones in which the current player can guarantee a win, assuming that both players play optimally.

### Problem Analysis
1. The current player can remove 1, 2, or 3 stones during their turn.
2. If there are `n` stones, the possible moves place the opponent in a position with `n-1`, `n-2`, or `n-3` stones.
3. A critical observation is that configurations with `n` stones where `n % 4 == 0` represent losing positions. If `n` is anything other than a multiple of 4, the current player can always remove stones in such a way that they leave the opponent with a losing position (a configuration where `n` is a multiple of 4).

### Winning Strategy
- If `n` is `1`, `2`, or `3`: The current player wins directly by taking all stones.
- If `n` is `4`: The current player will lose, regardless of how many stones they take (1, 2, or 3).
- Generally, for any `n`:
  - If `n % 4 == 0`: Current player loses.
  - Otherwise: Current player wins.

### Implementation
This leads us to a very simple modulo check as our solution. If `n` is a multiple of 4, we return `False`, indicating that the current player cannot win given optimal play by both parties.

Here is the implementation in the LeetCode format:



```python
class Solution:
    def canWinNim(self, n: int) -> bool:
        # If n is a multiple of 4, you cannot win
        return n % 4 != 0

```

### Explanation
- We define a class named `Solution`.
- The method `canWinNim` takes an integer `n` (the number of stones) and returns a boolean indicating if the player can guarantee a win.
- The return statement checks if `n % 4` is not equal to 0. If it’s not, the current player can win; if it is, the current player cannot win.

### Conclusion
The solution efficiently determines the outcome of the game in constant time O(1), making it optimal for any value of `n` within the constraints provided by the problem. This method is both simple and efficient and adheres to the requirement that both players play optimally.

# 294. Flip Game II

### Problem Description 
You are playing a Flip Game with your friend.

You are given a string `currentState` that contains only `'+'` and `'-'`. You and your friend take turns to flip two consecutive `"++"` into `"--"`. The game ends when a person can no longer make a move, and therefore the other person will be the winner.

Return `true` if the starting player can guarantee a win, and `false` otherwise.


Example 1:
Input: currentState = "++++"
Output: true
Explanation: The starting player can guarantee a win by flipping the middle "++" to become "+--+".


Example 2:
Input: currentState = "+"
Output: false

Constraints:
`1 <= currentState.length <= 60`
`currentState[i]` is either `'+'` or `'-'`.

Follow up: Derive your algorithm's runtime complexity.

### Solution 
 To tackle the problem of determining whether the starting player can guarantee a win in the Flip Game, we can use a recursive approach combined with memoization to avoid recomputation of the same game states.

### Detailed Explanation:

1. **Game Play Rules**: 
   - Players can only flip two consecutive "+" to "--". 
   - Once no "++" can be flipped, the game ends, and the player unable to move loses.

2. **State Representation**:
   - The game state can be represented as a string consisting of '+' and '-' characters.

3. **Recursive Function**:
   - To determine if the starting player can guarantee a win, we can use a recursive function `canWin`.
   - This function will:
     - Take the current game state as input.
     - Generate all possible next states by making valid moves (flipping "++" to "--").
     - Recursively check if the opponent can win from those next states. If the opponent cannot win in any of those states, then the current player has a winning move.

4. **Memoization**:
   - To enhance performance, we will store the results of already computed states in a dictionary (or hash map). This prevents the function from recomputing the results for previous states.

5. **Base Cases**:
   - If the string contains no "++", the function will return `False`, indicating that the current player cannot make a move.

### Complexity Analysis:
- Each state of the game is represented by a string of length up to 60. The maximum number of unique combinations of states is substantial, but memoization will significantly reduce the number of computations, making the approach efficient enough for the problem constraints.

### Implementation:

Here’s the Python implementation of the solution:



```python
class Solution:
    def canWin(self, currentState: str) -> bool:
        memo = {}

        def canWinHelper(state: str) -> bool:
            if state in memo:
                return memo[state]
            
            for i in range(len(state) - 1):
                if state[i] == '+' and state[i + 1] == '+':
                    # Create the new state after the move
                    newState = state[:i] + '--' + state[i + 2:]
                    # If the opponent cannot win from this new state
                    if not canWinHelper(newState):
                        memo[state] = True
                        return True
            
            # If no winning move was found, the current player loses
            memo[state] = False
            return False
        
        return canWinHelper(currentState)

# Example usage:
sol = Solution()
print(sol.canWin("++++"))  # Output: true
print(sol.canWin("+"))     # Output: false

```

### Explanation of the Code:
- The `canWin` function calls a helper function `canWinHelper`, which is responsible for determining the winning potential of a given state.
- Inside `canWinHelper`, we loop through the `currentState` to find all possible "++" pairs to flip.
- For each valid pair, we recursively call `canWinHelper` with the new state after the flip.
- If we find at least one move that forces the opponent to lose (returns `False`), we memoize that the current state is a winning state.
- Finally, we return the result of the initial state `currentState`.

This structure allows us to explore the game tree efficiently and determine if the starting player can force a win.

# 375. Guess Number Higher or Lower II

### Problem Description 
We are playing the Guessing Game. The game will work as follows:
I pick a number between `1` and `n`.

You guess a number.

If you guess the right number, you win the game.

If you guess the wrong number, then I will tell you whether the number I picked is higher or lower, and you will continue guessing.

Every time you guess a wrong number `x`, you will pay `x` dollars. If you run out of money, you lose the game.

Given a particular `n`, return the minimum amount of money you need to guarantee a win regardless of what number I pick.


Example 1:
Input: n = 10
Output: 16
Explanation: The winning strategy is as follows:
- The range is [1,10]. Guess 7.

    - If this is my number, your total is $0. Otherwise, you pay $7.

    - If my number is higher, the range is [8,10]. Guess 9.

        - If this is my number, your total is $7. Otherwise, you pay $9.

        - If my number is higher, it must be 10. Guess 10. Your total is $7 + $9 = $16.

        - If my number is lower, it must be 8. Guess 8. Your total is $7 + $9 = $16.

    - If my number is lower, the range is [1,6]. Guess 3.

        - If this is my number, your total is $7. Otherwise, you pay $3.

        - If my number is higher, the range is [4,6]. Guess 5.

            - If this is my number, your total is $7 + $3 = $10. Otherwise, you pay $5.

            - If my number is higher, it must be 6. Guess 6. Your total is $7 + $3 + $5 = $15.

            - If my number is lower, it must be 4. Guess 4. Your total is $7 + $3 + $5 = $15.

        - If my number is lower, the range is [1,2]. Guess 1.

            - If this is my number, your total is $7 + $3 = $10. Otherwise, you pay $1.

            - If my number is higher, it must be 2. Guess 2. Your total is $7 + $3 + $1 = $11.

The worst case in all these scenarios is that you pay $16. Hence, you only need $16 to guarantee a win.


Example 2:
Input: n = 1
Output: 0
Explanation: There is only one possible number, so you can guess 1 and not have to pay anything.


Example 3:
Input: n = 2
Output: 1
Explanation: There are two possible numbers, 1 and 2.

- Guess 1.

    - If this is my number, your total is $0. Otherwise, you pay $1.

    - If my number is higher, it must be 2. Guess 2. Your total is $1.

The worst case is that you pay $1.


Constraints:
`1 <= n <= 200`

### Solution 
 To solve the "Guessing Game" problem described, we need to implement a dynamic programming approach that computes the minimum amount of money needed to guarantee a win, regardless of what number is picked by the opponent.

### Explanation

The essence of this problem is to minimize the worst-case cost of guessing. Here's a structured breakdown of how we can accomplish this:

1. **Dynamic Programming Table**:
   We will create a 2D dp table where `dp[i][j]` represents the minimum amount of money needed to guarantee a win for the range of numbers from `i` to `j`. The dimensions of the dp table will be (n+1) x (n+1) because we will include 0 to n.

2. **Initialization**:
   - If there's only one number (i.e., `i == j`), then `dp[i][j] = 0` since no money is needed to guess correctly.
   
3. **Filling the DP Table**:
   - For ranges of size greater than 1 (i.e., `j > i`), we will iterate through all possible guesses `x` in the range from `i` to `j`.
   - For each guess `x`, we will calculate the cost:
     - If `x` is the correct number, we pay $0.
     - If the correct number is higher than `x`, we must consider the cost for the range `[x+1, j]`.
     - If the correct number is lower than `x`, we must consider the cost for the range `[i, x-1]`.
   - The formula to calculate the total cost when guessing `x`:
     \[
     \text{cost} = x + \max(\text{cost of higher range}, \text{cost of lower range})
     \]
   - We will take the minimum of these costs for every possible guess `x`.

4. **Return the Result**:
   - The answer we want is stored in `dp[1][n]`, which represents the minimum cost to guarantee a win when guessing any number between `1` and `n`.

### Python Code

Here's how you would implement the above logic in Python, formatted according to LeetCode standards:



```python
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        # Create a dp table with (n+1)x(n+1)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        # Fill the dp table
        for length in range(2, n + 1):  # length of the range
            for i in range(1, n - length + 2):  # start of the range
                j = i + length - 1  # end of the range
                min_cost = float('inf')  # start with a large number
                
                # Try every guess `x` from `i` to `j`
                for x in range(i, j + 1):
                    # Calculate the cost
                    cost = x + max(dp[i][x - 1], dp[x + 1][j])
                    min_cost = min(min_cost, cost)
                    
                dp[i][j] = min_cost  # Store the minimum cost for range [i, j]
        
        return dp[1][n]  # The result for the entire range [1, n]

# Example usage:
solver = Solution()
print(solver.getMoneyAmount(10))  # Output: 16

```

### How to Run this Code:
- You can directly copy-paste this code into the LeetCode online editor for the specific problem titled "Guessing Game" and execute it to see if it passes all the tests.
- The code uses O(n^2) time complexity to fill the dp table where n is the input number, which is efficient given the constraints \(1 \leq n \leq 200\). The auxiliary space complexity is also O(n^2) due to the dp table.

# 464. Can I Win

### Problem Description 
In the "100 game" two players take turns adding, to a running total, any integer from `1` to `10`. The player who first causes the running total to reach or exceed 100 wins.

What if we change the game so that players cannot re-use integers?
For example, two players might take turns drawing from a common pool of numbers from 1 to 15 without replacement until they reach a total >= 100.

Given two integers maxChoosableInteger and desiredTotal, return `true` if the first player to move can force a win, otherwise return `false`. Assume both players play optimally.


Example 1:
Input: maxChoosableInteger = 10, desiredTotal = 11
Output: false
Explanation:
No matter which integer the first player choose, the first player will lose.

The first player can choose an integer from 1 up to 10.

If the first player choose 1, the second player can only choose integers from 2 up to 10.

The second player will win by choosing 10 and get a total = 11, which is >= desiredTotal.

Same with other integers chosen by the first player, the second player will always win.


Example 2:
Input: maxChoosableInteger = 10, desiredTotal = 0
Output: true

Example 3:
Input: maxChoosableInteger = 10, desiredTotal = 1
Output: true

Constraints:
`1 <= maxChoosableInteger <= 20`
`0 <= desiredTotal <= 300`

### Solution 
 Sure! We'll solve the problem using a recursive approach with memoization to optimize our solution. The gist of the solution is to determine whether the first player can force a win given the current total of points, the maximum integer they can choose, and the pool of integers available. 

Here's a breakdown of the steps we'll take to solve this:

1. **Understanding the Problem**: The first player can choose any integer from `1` to `maxChoosableInteger`, and they must avoid picking numbers that allow the cumulative total to reach or exceed `100` immediately. If the player can make a move resulting in the second player having no valid moves to win, then the first player has a winning strategy.

2. **Recursive Function**: We'll create a recursive function that will track:
   - The current total score.
   - The integers available to the players (using a bitmask for efficient storage and retrieval).
   
3. **Memoization**: Since the recursive function might encounter the same state multiple times, we'll store results of previously computed states in a memoization table to avoid redundant calculations.

4. **Winning Condition**: If any choice allows the first player to either directly win or to cause the second player to eventually lose, we return `true`. Otherwise, if all choices lead to possible losses, we return `false`.

Now, let's implement this in Python according to the LeetCode format:



```python
class Solution:
    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        # If the desired total is 0, the player already "wins"
        if desiredTotal <= 0:
            return True
        # If the sum of all integers is less than desired total, can't win
        if (maxChoosableInteger * (maxChoosableInteger + 1)) // 2 < desiredTotal:
            return False

        # Dictionary to store computed states
        memo = {}

        def canWin(availableNumbers, currentTotal):
            if currentTotal >= desiredTotal:
                return False
            if availableNumbers in memo:
                return memo[availableNumbers]

            # Try every number from 1 to maxChoosableInteger
            for i in range(1, maxChoosableInteger + 1):
                if availableNumbers & (1 << i):  # Check if i is available
                    newAvailableNumbers = availableNumbers & ~(1 << i)  # Mark i as taken
                    if not canWin(newAvailableNumbers, currentTotal + i):
                        memo[availableNumbers] = True
                        return True

            memo[availableNumbers] = False
            return False

        # Initial call with all numbers available (1 to maxChoosableInteger)
        return canWin((1 << (maxChoosableInteger + 1)) - 2, 0)  # Bitmask for 1 to maxChoosableInteger

```

### Explanation:

1. **Base Cases**:
   - If `desiredTotal` is 0, the function immediately returns `true` because the player has already reached the goal without making a move.
   - If the sum of all possible integers from `1` to `maxChoosableInteger` is less than `desiredTotal`, then it’s impossible to achieve the total, and the function returns `false`.

2. **Recursive Function (`canWin`)**:
   - This function checks whether the player can win given a certain state.
   - It uses a bitmask to represent available integers. Each integer corresponds to a bit position.
   - The recursive exploration considers all available moves, and for each move, it checks if by making that move, the opponent has no winning response.

3. **Memoization**:
   - A dictionary (`memo`) is used to store the results of already computed states which helps in reducing computation time.

4. **Initial Bitmask**: 
   - The initial call constructs a bitmask that represents all integers from `1` to `maxChoosableInteger`.

This solution is efficient given the constraints, making it suitable for the input sizes specified in the problem statement.

# 486. Predict the Winner

### Problem Description 
Given an array of scores that are non-negative integers. Player 1 picks one of the numbers from either end of the array followed by the player 2 and then player 1 and so on. Each time a player picks a number, that number will not be available for the next player. This continues until all the scores have been chosen. The player with the maximum score wins.

Given an array of scores, predict whether player 1 is the winner. You can assume each player plays to maximize his score.


Example 1:
Input: [1, 5, 2]
Output: False
Explanation: Initially, player 1 can choose between 1 and 2. 
If he chooses 2 (or 1), then player 2 can choose from 1 (or 2) and 5. If player 2 chooses 5, then player 1 will be left with 1 (or 2). 
So, final score of player 1 is 1 + 2 = 3, and player 2 is 5. 
Hence, player 1 will never be the winner and you need to return False.


Example 2:
Input: [1, 5, 233, 7]
Output: True
Explanation: Player 1 first chooses 1. Then player 2 have to choose between 5 and 7. No matter which number player 2 choose, player 1 can choose 233.

Finally, player 1 has more score (234) than player 2 (12), so you need to return True representing player1 can win.


Constraints:
1 <= length of the array <= 20.

Any scores in the given array are non-negative integers and will not exceed 10,000,000.

If the scores of both players are equal, then player 1 is still the winner.

### Solution 
 To solve this problem, we can use Dynamic Programming (DP) to predict whether Player 1 can guarantee a win given the optimal strategies of both players. The main idea is to simulate the game recursively while maintaining a DP table to store the results of previously computed states for efficiency.

Here's the detailed explanation of how we can implement the solution:

1. **Define the Problem**:
   - Player 1 and Player 2 take turns picking numbers from either end of the array. The goal for both players is to maximize their total score.
   - We need to track the scores of both players throughout the game.

2. **Dynamic Programming Table**:
   - We can use a 2D DP table where `dp[l][r]` represents the maximum score difference (`Player 1's score - Player 2's score`) that Player 1 can achieve over Player 2 from the subarray `scores[l:r+1]`.
   - The difference is crucial because if the score difference is non-negative, Player 1 is guaranteed to win (or tie).

3. **Base Case**:
   - When `l == r`, there is only one choice for Player 1 — they take the only available score, leading to `dp[l][r] = scores[l]`.

4. **Recurrence Relation**:
   - For each possible choice Player 1 can make (either taking the leftmost or the rightmost score):
     - If Player 1 picks `scores[l]`, the score difference becomes `scores[l] - dp[l+1][r]` because Player 2 will then play optimally on `scores[l+1:r]`.
     - If Player 1 picks `scores[r]`, the score difference becomes `scores[r] - dp[l][r-1]`.
   - Thus, we collect the maximum of these two possibilities:
     

```python
     dp[l][r] = max(scores[l] - dp[l+1][r], scores[r] - dp[l][r-1])
     
```

5. **Final Decision**:
   - Once the DP table is filled, the result for the entire array will be found in `dp[0][n-1]`, where `n` is the length of the array. If `dp[0][n-1] >= 0`, Player 1 can ensure a win.

Now let's implement this in Python:



```python
class Solution:
    def PredictTheWinner(self, nums):
        n = len(nums)
        dp = [[0] * n for _ in range(n)]

        # Base case: when l == r, Player 1 takes the only score available
        for i in range(n):
            dp[i][i] = nums[i]

        # Fill the DP table
        for length in range(2, n + 1):  # length of the current subarray
            for l in range(n - length + 1):
                r = l + length - 1
                # Player 1 chooses the first (l) or last (r) score
                dp[l][r] = max(nums[l] - dp[l + 1][r], nums[r] - dp[l][r - 1])

        # If Player 1's total score - Player 2's total score >= 0, Player 1 can win
        return dp[0][n - 1] >= 0

```

### Explanation of the Code:
- We first initialize the DP table with zeroes.
- We populate the DP table based on our previously outlined logic starting from the base case.
- Finally, we check if the score difference for the entire array starting from `0` to `n-1` is greater than or equal to zero to determine if Player 1 can at least tie or win against Player 2.

This code can be copied directly into the LeetCode editor and will work for the problem described. It runs efficiently within the constraints provided.

# 843. Guess the Word

### Problem Description 
This is an interactive problem.

You are given an array of unique strings `wordlist` where `wordlist[i]` is `6` letters long, and one word in this list is chosen as `secret`.

You may call `Master.guess(word)` to guess a word. The guessed word should have type `string` and must be from the original list with `6` lowercase letters.

This function returns an `integer` type, representing the number of exact matches (value and position) of your guess to the `secret` word. Also, if your guess is not in the given wordlist, it will return `-1` instead.

For each test case, you have exactly `10` guesses to guess the word. At the end of any number of calls, if you have made `10` or fewer calls to `Master.guess` and at least one of these guesses was `secret`, then you pass the test case.


Example 1:
Input: secret = "acckzz", wordlist = ["acckzz","ccbazz","eiowzz","abcczz"], numguesses = 10
Output: You guessed the secret word correctly.
Explanation:
master.guess("aaaaaa") returns -1, because "aaaaaa" is not in wordlist.

master.guess("acckzz") returns 6, because "acckzz" is secret and has all 6 matches.

master.guess("ccbazz") returns 3, because "ccbazz" has 3 matches.

master.guess("eiowzz") returns 2, because "eiowzz" has 2 matches.

master.guess("abcczz") returns 4, because "abcczz" has 4 matches.

We made 5 calls to master.guess and one of them was the secret, so we pass the test case.


Example 2:
Input: secret = "hamada", wordlist = ["hamada","khaled"], numguesses = 10
Output: You guessed the secret word correctly.

Constraints:
`1 <= wordlist.length <= 100`
`wordlist[i].length == 6`
`wordlist[i]` consist of lowercase English letters.

All the strings of `wordlist` are unique.

`secret` exists in `wordlist`.

`numguesses == 10`

### Solution 
 To solve this problem, we need to create an efficient strategy for guessing words that could potentially match a given secret word, while adhering to the constraints laid out in the problem. We will leverage the nature of words being unique and the feedback mechanism provided by `Master.guess()` to narrow down our choices.

### Strategy
1. **Initial Set**: Start with the entire `wordlist`.
2. **Iterative Guessing**: For each guess, use `Master.guess()` to retrieve feedback on how many letters match correctly with the `secret`.
3. **Filtering Possible Words**: After each guess, filter the `wordlist` to include only those words that would produce the same feedback if guessed.
4. **Continue until Success or Exhaustion**: Repeat the guessing process until we either guess the `secret` correctly or exhaust the allowed number of guesses.

### Implementation

The implementation will follow the standard format for LeetCode solutions. We will define a class `Solution` with a method `findSecretWord` that interacts with the `Master` class.

Here's the code that adheres to the required LeetCode format:



```python
class Master:
    def guess(self, word: str) -> int:
        # This method is implemented by the LeetCode platform and should not be modified.
        pass

class Solution:
    def findSecretWord(self, wordlist: List[str], master: 'Master') -> None:
        def match_count(word1, word2):
            """ Return the number of matching characters in the same position. """
            return sum(a == b for a, b in zip(word1, word2))
        
        possible_words = wordlist[:]  # Start with all words in the wordlist
        for _ in range(10):  # We have at maximum 10 guesses
            guess_word = possible_words[0]  # Guess the first available word
            match = master.guess(guess_word)  # Get the number of matches
            
            # Filter the guesses based on the feedback
            possible_words = [word for word in possible_words if match_count(word, guess_word) == match]
            
            # If we matched all characters, we guessed the secret word
            if match == 6:
                return

```

### Explanation
- **`Master` Class**: The `Master` class is provided by the environment and is not to be altered. It has a method `.guess(word)` that we will use to interact with the secret word.
- **`findSecretWord` Method**:
  - We define an inner function, `match_count`, which compares two words and counts how many characters match in the same positions.
  - We maintain a list called `possible_words` which starts as a copy of the `wordlist`.
  - We iterate up to 10 times (the maximum number of guesses allowed):
    - We take the first word from our `possible_words` to use as our guess.
    - We call `master.guess` with this word and store the number of matches it returns.
    - We then filter `possible_words` using a list comprehension to retain only those words that would give the same match count if guessed.
    - If the number of matches equals 6, we have found the secret and can return early.

This solution is efficient given the constraints, as we progressively narrow down our search space based on the feedback received from the `Master.guess` method.

# 877. Stone Game

### Problem Description 
Alex and Lee play a game with piles of stones.  There are an even number of piles arranged in a row, and each pile has a positive integer number of stones `piles[i]`.

The objective of the game is to end with the most stones.  The total number of stones is odd, so there are no ties.

Alex and Lee take turns, with Alex starting first.  Each turn, a player takes the entire pile of stones from either the beginning or the end of the row.  This continues until there are no more piles left, at which point the person with the most stones wins.

Assuming Alex and Lee play optimally, return `True` if and only if Alex wins the game.


Example 1:
Input: piles = [5,3,4,5]
Output: true
Explanation: 
Alex starts first, and can only take the first 5 or the last 5.

Say he takes the first 5, so that the row becomes [3, 4, 5].

If Lee takes 3, then the board is [4, 5], and Alex takes 5 to win with 10 points.

If Lee takes the last 5, then the board is [3, 4], and Alex takes 4 to win with 9 points.

This demonstrated that taking the first 5 was a winning move for Alex, so we return true.


Constraints:
`2 <= piles.length <= 500`
`piles.length` is even.

`1 <= piles[i] <= 500`
`sum(piles)` is odd.

### Solution 
 Certainly! The problem of Alex and Lee playing a game with piles of stones can be approached using dynamic programming. The essence of the problem is to maximize the number of stones Alex can collect by making optimal moves while taking into account that Lee will also play optimally.

### Explanation of the Approach

1. **Game Structure**: Since both players play optimally, if it’s Alex's turn and he picks from either end of the remaining piles, the maximum stones he can collect is dependent on both his choice and what Lee will do next.

2. **Dynamic Programming Table**: We will maintain a DP table where `dp[i][j]` represents the maximum number of stones a player can collect from piles `i` to `j`. By filling this table, we will determine the best result Alex can achieve starting from the entire list of piles.

3. **Transition Formula**: 
   - If Alex picks the pile at the start (`piles[i]`), then the next turn is Lee's and he can play optimally on piles `i+1` to `j`. Hence, the stones Alex collects will be `piles[i] + (total stones in piles[i...j] - dp[i+1][j])`.
   - If Alex picks the pile at the end (`piles[j]`), then Lee can play on piles `i` to `j-1`, hence Alex collects `piles[j] + (total stones in piles[i...j] - dp[i][j-1])`.

   Thus, the formula becomes:
   

```python
   dp[i][j] = max(piles[i] + (total_stones_in_range - dp[i+1][j]),
                  piles[j] + (total_stones_in_range - dp[i][j-1]))
   
```

4. **Base Case**: When `i == j`, meaning there’s only one pile, `dp[i][j] = piles[i]`.

5. **Final Decision**: After filling the DP table, we check whether the stones Alex can collect is more than half the total stones in the piles to determine if he wins.

### Implementation

Here’s how you can implement this in Python in the LeetCode format:



```python
class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        n = len(piles)
        
        # Initialize dp table
        dp = [[0] * n for _ in range(n)]
        
        # Base case: When there's only one pile to take
        for i in range(n):
            dp[i][i] = piles[i]
        
        # Fill the DP table
        for length in range(2, n + 1):  # length of the interval
            for i in range(n - length + 1):
                j = i + length - 1
                total_stones_in_range = sum(piles[i:j + 1])
                dp[i][j] = max(piles[i] + (total_stones_in_range - dp[i + 1][j]),
                               piles[j] + (total_stones_in_range - dp[i][j - 1]))
        
        # Since the total number of stones is odd, Alex wins if he can collect more than half
        return dp[0][n - 1] > sum(piles) // 2

```

### Explanation of the Code

- We create a 2D list (`dp`) of the same length as `piles` to store the maximum stones collected.
- We populate `dp` for every possible subarray of `piles`, from length 2 up to `n` (the total number of piles).
- For each possible pair of indices `(i, j)`, we calculate the maximum stones using our transition formula.
- Finally, we check if the maximum stones that Alex can collect from the entire pile makes him a winner by comparing it to half of the total stones.

### Complexity Analysis
- **Time Complexity**: O(n^2) because we fill out a DP table with `n * n` entries.
- **Space Complexity**: O(n^2) due to the DP table.

This code should be ready to run on LeetCode and will give the correct output for the problem requirements.

# 913. Cat and Mouse

### Problem Description 
A game on an undirected graph is played by two players, Mouse and Cat, who alternate turns.

The graph is given as follows: `graph[a]` is a list of all nodes `b` such that `ab` is an edge of the graph.

The mouse starts at node `1` and goes first, the cat starts at node `2` and goes second, and there is a hole at node `0`.

During each player's turn, they must travel along one edge of the graph that meets where they are.  For example, if the Mouse is at node 1, it must travel to any node in `graph[1]`.

Additionally, it is not allowed for the Cat to travel to the Hole (node 0.)
Then, the game can end in three ways:
If ever the Cat occupies the same node as the Mouse, the Cat wins.

If ever the Mouse reaches the Hole, the Mouse wins.

If ever a position is repeated (i.e., the players are in the same position as a previous turn, and it is the same player's turn to move), the game is a draw.

Given a `graph`, and assuming both players play optimally, return
`1` if the mouse wins the game,
`2` if the cat wins the game, or
`0` if the game is a draw.


Example 1:
Input: graph = [[2,5],[3],[0,4,5],[1,4,5],[2,3],[0,2,3]]
Output: 0

Example 2:
Input: graph = [[1,3],[0],[3],[0,2]]
Output: 1

Constraints:
`3 <= graph.length <= 50`
`1 <= graph[i].length < graph.length`
`0 <= graph[i][j] < graph.length`
`graph[i][j] != i`
`graph[i]` is unique.

The mouse and the cat can always move.

### Solution 
 To solve the Mouse and Cat game problem with the given constraints, we can utilize a depth-first search (DFS) approach combined with memoization. In this game, the strategy will involve determining which player (the Mouse or the Cat) has a winning strategy given the current positions of both players.

Here is a step-by-step breakdown of how to implement the solution:

### Step 1: Define Game States
We define the game state using three variables:
- The position of the Mouse, `mouse`.
- The position of the Cat, `cat`.
- The turn of the player (either `True` for Mouse's turn or `False` for Cat's turn).

### Step 2: Termination Conditions
We can identify three main conditions where the game ends:
1. If the Mouse reaches the hole (`mouse == 0`), the Mouse wins.
2. If the Mouse and the Cat occupy the same node (`mouse == cat`), the Cat wins.
3. If the positions of the players repeat in a manner creating a loop state, the game ends in a draw.

### Step 3: Optimal Movement
In each turn, players have to make optimal moves:
- On the Mouse's turn, examine all possible moves to see if there exists at least one movement leading to a winning strategy for the Mouse.
- On the Cat's turn, examine all possible moves to see if the Cat can force a win.

### Step 4: Use Memoization
To avoid exponential calculation and revisit the same game states repeatedly, we use a memoization dictionary to store the results for each combination of the Mouse's position, the Cat's position, and whose turn it is.

### Step 5: DFS Implementation
Implement a recursive DFS function that considers the possible moves for each player based on the aforementioned rules and conditions.

Here's the complete code in Python:



```python
class Solution:
    def catMouseGame(self, graph: List[List[int]]) -> int:
        memo = {}

        def dfs(mouse: int, cat: int, turn: int) -> int:
            if mouse == 0:
                return 1  # Mouse wins
            if mouse == cat:
                return 2  # Cat wins
            
            if (mouse, cat, turn) in memo:
                return memo[(mouse, cat, turn)]
            
            if turn == 0:  # Mouse's turn
                for next_mouse in graph[mouse]:
                    result = dfs(next_mouse, cat, 1)
                    if result == 1:  # If there is a winning move for mouse
                        memo[(mouse, cat, turn)] = 1
                        return 1
                memo[(mouse, cat, turn)] = 0  # If no winning move found, return draw
                return 0
            else:  # Cat's turn
                for next_cat in graph[cat]:
                    if next_cat != 0:  # Cat cannot go to the hole
                        result = dfs(mouse, next_cat, 0)
                        if result == 2:  # If there is a winning move for cat
                            memo[(mouse, cat, turn)] = 2
                            return 2
                memo[(mouse, cat, turn)] = 0  # If no winning move found, return draw
                return 0

        return dfs(1, 2, 0)


```

### Explanation of the Code
1. **Memoization Dictionary**: Used to cache results of previous calculations for game states to improve efficiency.
2. **DFS Function**: 
   - Takes current mouse and cat positions, and whose turn it is.
   - Checks for terminal states to return immediate outcomes (win/loss).
   - For the Mouse's turn, it checks all permissible moves and looks for any winning paths.
   - For the Cat's turn, it does the opposite, ensuring it does not move into the hole.
3. **Return Value**: Depending on the exploration of paths, return `1`, `2`, or `0` based on the determined outcome.

The given solution efficiently decides the winner by simulating optimal moves with careful handling of game states. This approach is efficient given the problem constraints and will perform well on the LeetCode platform.

