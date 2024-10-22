# 146. LRU Cache

### Problem Description 
Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the `LRUCache` class:
`LRUCache(int capacity)` Initialize the LRU cache with positive size `capacity`.

`int get(int key)` Return the value of the `key` if the key exists, otherwise return `-1`.

`void put(int key, int value)` Update the value of the `key` if the `key` exists. Otherwise, add the `key-value` pair to the cache. If the number of keys exceeds the `capacity` from this operation, evict the least recently used key.

Follow up:
Could you do `get` and `put` in `O(1)` time complexity?

Example 1:
Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]
Explanation
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4

Constraints:
`1 <= capacity <= 3000`
`0 <= key <= 3000`
`0 <= value <= 104`
At most `3 * 104` calls will be made to `get` and `put`.

### Solution 
 To solve the Least Recently Used (LRU) cache problem, we can leverage two key data structures: a hash map (or dictionary in Python) and a doubly linked list. Here's why we use these structures:

1. **Hash Map**: This allows us to achieve O(1) time complexity for accessing the values. The keys are used to directly fetch the values we need.
  
2. **Doubly Linked List**: This helps maintain the order of usage of the keys. When a key is accessed (whether via `get` or `put`), we can move it to the front (most recently used) of the list. When we reach the cache capacity, we can easily remove the least recently used item, which is at the end of the list.

### Solution

1. **Initialization**: We define the capacity of the LRU Cache. We will implement the logic for `put` and `get` methods.
  
2. **Put**: 
   - If the key already exists, we update its value and move it to the front.
   - If it's a new key and the capacity is reached, we remove the last node from the linked list (the least recently used one) and also remove the key from the hash map.
   - Insert the new key-value pair at the front of the linked list and in the hash map.

3. **Get**: 
   - If the key exists, we return its value and move it to the front of the linked list.
   - If it doesn't exist, we return -1.

Here's the complete Python implementation in the format used by LeetCode:



```python
class Node:
    """A node in the doubly linked list."""
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node
        self.head = Node(0, 0)  # dummy head
        self.tail = Node(0, 0)  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node):
        """Remove a node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: Node):
        """Add a node to the front of the doubly linked list."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)  # Remove from current position
            self._add_to_front(node)  # Add to the front
            return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Node exists, update the value and move it to the front
            node = self.cache[key]
            self._remove(node)
            node.value = value
            self._add_to_front(node)
        else:
            # New key-value pair
            new_node = Node(key, value)
            if len(self.cache) >= self.capacity:
                # Evict the least recently used item
                lru_node = self.tail.prev
                self._remove(lru_node)
                del self.cache[lru_node.key]
            self._add_to_front(new_node)
            self.cache[key] = new_node

```

### Explanation of the Code:

1. **Node Class**: Represents a single node in the doubly linked list that holds the key and value.

2. **LRUCache Class**: We initialize the cache with a given capacity, set up a hash map, and establish dummy nodes for the linked list.

3. **_remove(node)**: A helper function that removes a node from its current position in the linked list.

4. **_add_to_front(node)**: A helper function that adds a node right after the head (most recent position).

5. **get(key)**: Returns the value for the specified key, and updates the list to mark this key as recently used.

6. **put(key, value)**: Adds a new key-value pair to the cache (or updates an existing key). Handles evictions of the least recently used keys by removing from the linked list and cache.

This implementation offers O(1) time complexity for both `get` and `put` operations and effectively simulates an LRU cache.

# 155. Min Stack

### Problem Description 
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the `MinStack` class:
`MinStack()` initializes the stack object.

`void push(val)` pushes the element `val` onto the stack.

`void pop()` removes the element on the top of the stack.

`int top()` gets the top element of the stack.

`int getMin()` retrieves the minimum element in the stack.


Example 1:
Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]
Output
[null,null,null,null,-3,null,0,-2]
Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2

Constraints:
`-231 <= val <= 231 - 1`
Methods `pop`, `top` and `getMin` operations will always be called on non-empty stacks.

At most `3 * 104` calls will be made to `push`, `pop`, `top`, and `getMin`.

### Solution 
 To solve the problem of designing a `MinStack` that supports basic stack operations along with retrieving the minimum element in constant time, we'll maintain two stacks: one for the stack elements and another for the minimum values.

### Explanation

1. **Initialization**: We will initialize two stacks. The first stack `stack` will store all the elements, and the second stack `min_stack` will store the minimum elements.
  
2. **Push Operation**: When we push a value onto the stack:
   - We add the value to the main stack.
   - For the minimum stack, we check if it's empty or if the value being pushed is less than or equal to the current minimum (i.e., the top of the `min_stack`). If either condition is true, we also push the value onto the `min_stack`. This guarantees that the smallest values are always stored in the `min_stack`.

3. **Pop Operation**: When we pop an element from the stack:
   - We remove the top element from the main stack.
   - If the popped element is the same as the top of the `min_stack`, we pop from that stack as well. This way, we keep our minimum stack updated.

4. **Top Operation**: This simply returns the top element of the main stack.

5. **GetMin Operation**: This returns the top element of the `min_stack`, which will always be the minimum value of the stack.

### LeetCode Solution Format

Here's the complete implementation that adheres to the format required by LeetCode:



```python
class MinStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack = []     # This stack holds all the elements
        self.min_stack = [] # This stack holds the minimums

    def push(self, val: int) -> None:
        """
        Pushes the element val onto the stack.
        """
        self.stack.append(val)
        # If min_stack is empty or the current value is <= the current minimum
        # then add it to the min_stack
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        """
        Removes the element on the top of the stack.
        """
        if self.stack:
            val = self.stack.pop()
            # If the popped value is equal to the current minimum, pop from min_stack too
            if val == self.min_stack[-1]:
                self.min_stack.pop()

    def top(self) -> int:
        """
        Get the top element of the stack.
        """
        return self.stack[-1] if self.stack else None

    def getMin(self) -> int:
        """
        Retrieve the minimum element in the stack.
        """
        return self.min_stack[-1] if self.min_stack else None

```

### Example

Now, let's see how this implementation can be used with the given example:



```python
minStack = MinStack()
minStack.push(-2)
minStack.push(0)
minStack.push(-3)
print(minStack.getMin()) # return -3
minStack.pop()
print(minStack.top())    # return 0
print(minStack.getMin()) # return -2

```

### Complexity Analysis

- **Time Complexity**: All operations (`push`, `pop`, `top`, `getMin`) are O(1) due to the stack operations being efficient.
- **Space Complexity**: O(n) in the worst case, where n is the number of elements, since we could potentially store all elements in both stacks.

This implementation efficiently maintains the minimum element in constant time while also supporting standard stack operations.

# 170. Two Sum III - Data structure design

### Problem Description 
Design a data structure that accepts a stream of integers and checks if it has a pair of integers that sum up to a particular value.

Implement the `TwoSum` class:
`TwoSum()` Initializes the `TwoSum` object, with an empty array initially.

`void add(int number)` Adds `number` to the data structure.

`boolean find(int value)` Returns `true` if there exists any pair of numbers whose sum is equal to `value`, otherwise, it returns `false`.


Example 1:
Input
["TwoSum", "add", "add", "add", "find", "find"]
[[], [1], [3], [5], [4], [7]]
Output
[null, null, null, null, true, false]
Explanation
TwoSum twoSum = new TwoSum();
twoSum.add(1);   // [] --> [1]
twoSum.add(3);   // [1] --> [1,3]
twoSum.add(5);   // [1,3] --> [1,3,5]
twoSum.find(4);  // 1 + 3 = 4, return true
twoSum.find(7);  // No two integers sum up to 7, return false

Constraints:
`-105 <= number <= 105`
`-231 <= value <= 231 - 1`
At most `5 * 104` calls will be made to `add` and `find`.

### Solution 
 To solve the problem of designing a data structure that checks for pairs of integers summing up to a specific value, we will implement a class called `TwoSum`. This class will maintain a collection of integers and allow us to add integers while providing a method to check if any two of those integers sum up to a given target.

### Explanation of the Approach

1. **Data Structure Choice**: 
   - We'll use a dictionary (hashmap) to store the frequency of each number we add. This allows us to efficiently check if there exists a complement for any number we process.
   
2. **Adding a Number**:
   - When adding a number, we simply increment its count in our dictionary. This ensures we know how many times a number has been added, which is crucial for handling pairs correctly.

3. **Finding a Pair**:
   - To check if a pair exists that sums to a certain value, we iterate over each unique number in our dictionary. For each number `x`, we calculate its complement `y` (`y = value - x`).
   - We need to check two conditions:
     - If `x` equals `y`, we need at least two occurrences of `x` (i.e., `count[x] > 1`).
     - If `x` does not equal `y`, we just need to ensure that both `x` and `y` are present in the dictionary.

### Code Implementation

Now, we can implement the above logic in the required `TwoSum` class.



```python
class TwoSum:
    def __init__(self):
        # Initialize an empty dictionary to store the numbers and their frequencies
        self.numbers = {}
        
    def add(self, number: int) -> None:
        # Add the number to the dictionary, incrementing its count
        if number in self.numbers:
            self.numbers[number] += 1
        else:
            self.numbers[number] = 1
            
    def find(self, value: int) -> bool:
        # Check for pairs that sum to the specified value
        for x in self.numbers:
            y = value - x  # The required complement to form the target sum
            if y in self.numbers:
                # If x == y, check that there are at least 2 occurrences of x
                if x == y:
                    if self.numbers[x] > 1:
                        return True
                else:
                    return True
        return False

```

### How to Use the Class

1. **Initialize the Class**: 
   - Create an instance of `TwoSum`.
2. **Add Numbers**: 
   - Call the `add(number)` method to add integers.
3. **Find Pairs**: 
   - Use the `find(value)` method to check if there are two distinct integers stored that sum up to the specified `value`.

### Example Usage



```python
twoSum = TwoSum()
twoSum.add(1)   # Adds 1
twoSum.add(3)   # Adds 3
twoSum.add(5)   # Adds 5
print(twoSum.find(4))  # Returns True because 1 + 3 = 4
print(twoSum.find(7))  # Returns False because no two integers sum to 7

```

This implementation is efficient with a time complexity of O(n) for the `add` method and O(n) for the `find` method in the worst case, where n is the number of unique numbers in our `numbers` dictionary. The space complexity is O(n) as well, due to storing all unique numbers. The overall operations are suitable considering the problem's constraints.

# 173. Binary Search Tree Iterator

### Problem Description 
Implement the `BSTIterator` class that represents an iterator over the in-order traversal of a binary search tree (BST):
`BSTIterator(TreeNode root)` Initializes an object of the `BSTIterator` class. The `root` of the BST is given as part of the constructor. The pointer should be initialized to a non-existent number smaller than any element in the BST.

`boolean hasNext()` Returns `true` if there exists a number in the traversal to the right of the pointer, otherwise returns `false`.

`int next()` Moves the pointer to the right, then returns the number at the pointer.

Notice that by initializing the pointer to a non-existent smallest number, the first call to `next()` will return the smallest element in the BST.

You may assume that `next()` calls will always be valid. That is, there will be at least a next number in the in-order traversal when `next()` is called.


Example 1:
Input
["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
Output
[null, 3, 7, true, 9, true, 15, true, 20, false]
Explanation
BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
bSTIterator.next();    // return 3
bSTIterator.next();    // return 7
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 9
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 15
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 20
bSTIterator.hasNext(); // return False

Constraints:
The number of nodes in the tree is in the range `[1, 105]`.

`0 <= Node.val <= 106`
At most `105` calls will be made to `hasNext`, and `next`.

Follow up:
Could you implement `next()` and `hasNext()` to run in average `O(1)` time and use `O(h)` memory, where `h` is the height of the tree?

### Solution 
 To solve the problem of implementing a `BSTIterator` class for a binary search tree (BST), we will perform an in-order traversal using a stack data structure. The in-order traversal allows us to visit nodes in ascending order since the properties of a binary search tree dictate that the left child is always less than the parent and the right child is always greater.

Here's how we can implement the `BSTIterator`:

### Steps to Implement `BSTIterator`

1. **Initialization**: When the `BSTIterator` is initialized with the root of the BST, we will prepare to perform an in-order traversal. We will use a stack to keep track of the nodes we visit.

2. **Push All Left Children**: In the constructor, we will push all the left children of the root onto the stack. This means that when we call `next()`, it starts from the smallest element (the leftmost node).

3. **Next Method**:
   - Check if the stack has any nodes left (using `hasNext()`).
   - Pop the top node from the stack (which will be the next smallest node).
   - Push the right child of the popped node onto the stack, along with all its left children (if it exists).
   - Return the value of the popped node.

4. **Has Next Method**: Simply check if the stack has any nodes.

### Complexity
- The `next()` and `hasNext()` methods will run in average O(1) time, and the space complexity will be O(h), where h is the height of the tree due to the stack's usage.

### Python Code
Here's the complete code for the `BSTIterator` class:



```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class BSTIterator:

    def __init__(self, root: TreeNode):
        self.stack = []
        self._push_left_children(root)

    def _push_left_children(self, node):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        if not self.hasNext():
            return None  # technically we shouldn't reach here as per problem statement
        
        # Get the next smallest element
        next_node = self.stack.pop()
        
        # Push all left children of the right subtree
        self._push_left_children(next_node.right)
        
        return next_node.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0

```

### Explanation of the Code
1. **TreeNode Class**: A simple implementation of the tree node structure with attributes for the node's value, left child, and right child.
   
2. **BSTIterator Class**:
   - The constructor (`__init__`) initializes the stack and populates it with the left children of the root node by calling `_push_left_children`.
   - `_push_left_children(node)`: A helper method that pushes all left children of a given node onto the stack to set up the iterator for the in-order traversal.
   - `next()`: Pops the top element from the stack, processes it, and pushes its right child's left children onto the stack.
   - `hasNext()`: Returns whether there are any nodes left to process (the stack is not empty).

### Usage
You can call the `BSTIterator` with a `TreeNode` representing the root of a BST, and then repeatedly call `next()` and `hasNext()` as described to iterate through the elements in in-order. 

This implementation will work efficiently even for large trees following the given constraints.

# 208. Implement Trie (Prefix Tree)

### Problem Description 
A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:
`Trie()` Initializes the trie object.

`void insert(String word)` Inserts the string `word` into the trie.

`boolean search(String word)` Returns `true` if the string `word` is in the trie (i.e., was inserted before), and `false` otherwise.

`boolean startsWith(String prefix)` Returns `true` if there is a previously inserted string `word` that has the prefix `prefix`, and `false` otherwise.


Example 1:
Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]
Explanation
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True

Constraints:
`1 <= word.length, prefix.length <= 2000`
`word` and `prefix` consist only of lowercase English letters.

At most `3 * 104` calls in total will be made to `insert`, `search`, and `startsWith`.

### Solution 
 To solve the problem of implementing a Trie (prefix tree), we will define a `Trie` class with the necessary methods for insertion, searching, and prefix checking. Here’s how the Trie will be structured and how the methods will work.

### Explanation of the Trie Structure

1. **Node Class**: Each node in the Trie will represent a character and will have:
   - A dictionary to store children nodes (representing the next characters in the words).
   - A boolean flag `is_end_of_word` to indicate if the current node marks the end of a valid word.

2. **Trie Class**: The main Trie class will include:
   - An initializer to create the root node of the Trie.
   - The `insert(word)` method to add words to the Trie.
   - The `search(word)` method to check if a word is present in the Trie.
   - The `startsWith(prefix)` method to determine if there's any word in the Trie that starts with the given prefix.

### Implementation of the Trie

Here is the complete code encapsulated in a format suitable for LeetCode:



```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        
class Trie:
    def __init__(self):
        # Initialize the root of the trie
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        # Start from the root node
        node = self.root
        for char in word:
            # If the character is not a child, add it
            if char not in node.children:
                node.children[char] = TrieNode()
            # Move to the child node
            node = node.children[char]
        # Mark the end of the word
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        # Start from the root node
        node = self.root
        for char in word:
            # If the character is not found, the word doesn't exist
            if char not in node.children:
                return False
            node = node.children[char]
        # Check if the last character is the end of a word
        return node.is_end_of_word

    def startsWith(self, prefix: str) -> bool:
        # Start from the root node
        node = self.root
        for char in prefix:
            # If the character is not found, no word starts with the prefix
            if char not in node.children:
                return False
            node = node.children[char]
        # If all characters are found in the prefix, return True
        return True

```

### Explanation of Each Method

1. **`__init__`:** Initializes a root TrieNode.
2. **`insert`:** 
   - Iterates through each character in the provided word.
   - If a character does not exist in the current node’s children, it creates a new TrieNode for that character.
   - Moves to the child node corresponding to the current character.
   - After all characters are processed, sets `is_end_of_word` to `True` to mark the end of the word.
   
3. **`search`:**
   - Similar to `insert`, it traverses the Trie using the characters of the word.
   - If a character is missing at any point, it returns `False`.
   - At the end of the search, it checks whether the last node is an endpoint for a word using `is_end_of_word`.
   
4. **`startsWith`:**
   - It works similarly to `search`, but it does not require the last node to be an endpoint (i.e., `is_end_of_word` is not checked). 
   - If all characters in the prefix exist in the Trie, it returns `True`.

### Time Complexity
- **Insert and Search Operations**: O(m), where m is the length of the word or prefix being inserted or searched.
- **StartsWith Operation**: O(n), where n is the length of the prefix.

### Space Complexity
The space complexity is O(m * n) in the worst case, where m is the number of words inserted and n is the average length of these words, due to the storage of nodes in the Trie.

This implementation effectively captures the Trie structure and functionality, providing a clear solution that can handle the problems posed efficiently. Alternatively, you can run this code directly on LeetCode for any provided test cases as per the problem constraints.

# 211. Design Add and Search Words Data Structure

### Problem Description 
Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the `WordDictionary` class:
`WordDictionary()` Initializes the object.

`void addWord(word)` Adds `word` to the data structure, it can be matched later.

`bool search(word)` Returns `true` if there is any string in the data structure that matches `word` or `false` otherwise. `word` may contain dots `'.'` where dots can be matched with any letter.


Example:
Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]
Explanation
WordDictionary wordDictionary = new WordDictionary();
wordDictionary.addWord("bad");
wordDictionary.addWord("dad");
wordDictionary.addWord("mad");
wordDictionary.search("pad"); // return False
wordDictionary.search("bad"); // return True
wordDictionary.search(".ad"); // return True
wordDictionary.search("b.."); // return True

Constraints:
`1 <= word.length <= 500`
`word` in `addWord` consists lower-case English letters.

`word` in `search` consist of  `'.'` or lower-case English letters.

At most `50000` calls will be made to `addWord` and `search`.

### Solution 
 To tackle the problem of designing a `WordDictionary` that supports adding words and searching with a special pattern that can include wildcards (represented by '.'), we can use a Trie data structure (prefix tree). This is an ideal choice for this kind of problem, as it allows us to efficiently insert and search for strings.

### Explanation of the Solution

1. **Choosing the Data Structure**:
   - We will use a Trie, where each node represents a character from the words added to the dictionary.
   - Each node will have a dictionary of child nodes (for each character) and a boolean flag to mark if a word ends at that node.

2. **Adding Words**:
   - When we add a word, we will insert it character by character into the Trie. If the character already exists in the Trie, we move to the existing node; otherwise, we create a new child node.

3. **Searching for Words**:
   - Searching a word will be a recursive function that checks each character:
     - If the character is '.', we will recursively search all possible child nodes.
     - If it’s a letter, we will try to go down the Trie corresponding to that letter. If the letter doesn’t exist, the search fails at that point.
   - The search function will also recursively check if we reach the end of a word when the last character is processed.

4. **Complexity Considerations**:
   - Insertions and searches both have a complexity of O(M) where M is the length of the word being added or searched since we traverse each character once.

### Implementation of the Solution



```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        return self._search_in_node(word, self.root)

    def _search_in_node(self, word: str, node: TrieNode) -> bool:
        for i, char in enumerate(word):
            if char == '.':
                # Try all possible children
                for child in node.children.values():
                    if self._search_in_node(word[i + 1:], child):
                        return True
                return False  # No match found with any child
            else:
                if char not in node.children:
                    return False  # No child for the character
                node = node.children[char]  # Move to the next node
        return node.is_end_of_word  # Check if we reached end of a valid word

# Example of usage:
# wordDictionary = WordDictionary()
# wordDictionary.addWord("bad")
# wordDictionary.addWord("dad")
# wordDictionary.addWord("mad")
# print(wordDictionary.search("pad"))  # return False
# print(wordDictionary.search("bad"))  # return True
# print(wordDictionary.search(".ad"))  # return True
# print(wordDictionary.search("b.."))  # return True

```

### Explanation of the Code

- **TrieNode Class**: A helper class to represent each node in the Trie with a dictionary of children and a flag to indicate if it's the end of a word.
- **WordDictionary Class**:
  - `__init__`: Initializes the root of the Trie.
  - `addWord`: Adds words to the Trie, creating new Trie nodes as necessary.
  - `search`: Initiates a search for the given word using a helper function `_search_in_node`.
  - `_search_in_node`: A recursive function that checks each character. If a character is a '.', it checks all potential paths in the Trie, and if it is a letter, it only checks that specific path.

This code is structured to be efficient and allows for wildcard searches through the Trie structure. The implementation can be directly copied to the LeetCode platform for testing and further usage.

# 225. Implement Stack using Queues

### Problem Description 
Implement a last in first out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal queue (`push`, `top`, `pop`, and `empty`).

Implement the `MyStack` class:
`void push(int x)` Pushes element x to the top of the stack.

`int pop()` Removes the element on the top of the stack and returns it.

`int top()` Returns the element on the top of the stack.

`boolean empty()` Returns `true` if the stack is empty, `false` otherwise.

Notes:
You must use only standard operations of a queue, which means only `push to back`, `peek/pop from front`, `size`, and `is empty` operations are valid.

Depending on your language, the queue may not be supported natively. You may simulate a queue using a list or deque (double-ended queue), as long as you use only a queue's standard operations.


Example 1:
Input
["MyStack", "push", "push", "top", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 2, 2, false]
Explanation
MyStack myStack = new MyStack();
myStack.push(1);
myStack.push(2);
myStack.top(); // return 2
myStack.pop(); // return 2
myStack.empty(); // return False

Constraints:
`1 <= x <= 9`
At most `100` calls will be made to `push`, `pop`, `top`, and `empty`.

All the calls to `pop` and `top` are valid.

Follow-up: Can you implement the stack such that each operation is amortized `O(1)` time complexity? In other words, performing `n` operations will take overall `O(n)` time even if one of those operations may take longer. You can use more than two queues.

### Solution 
 To implement a Last In First Out (LIFO) stack using only two queues, we will utilize the following approach.

**Explanation of the Approach:**

1. **Data structures**: We'll use two queues, let's call them `queue1` and `queue2`. The primary queue (`queue1`) will be used to maintain the order of elements, while the secondary queue (`queue2`) will be used to assist in implementing the LIFO order.

2. **Push Operation**: When we push an element onto the stack, we'll first add that element to `queue2`. Then, we will append all elements from `queue1` to `queue2`. Finally, we will swap the roles of the queues – making `queue1` the new `queue2` and vice versa. This ensures that the last pushed element will always be at the front of `queue1`.

3. **Pop Operation**: To pop an element, we simply need to dequeue from `queue1`, which will give us the last pushed element due to our management of the queues.

4. **Top Operation**: Similar to pop, but instead of removing the element from `queue1`, we peek at the front of `queue1`.

5. **Empty Check**: We just need to check if `queue1` is empty.

This approach ensures that although the push operation can take O(n) time due to moving elements between queues, the pop and top operations are O(1). Thus, it provides an amortized O(1) time complexity over a sequence of operations which is a follow-up question requirement.

Here’s the implementation of the `MyStack` class following the LeetCode format:



```python
from collections import deque

class MyStack:

    def __init__(self):
        self.queue1 = deque()
        self.queue2 = deque()

    def push(self, x: int) -> None:
        # Push to queue2
        self.queue2.append(x)
        # Push all the previous elements from queue1 to queue2
        while self.queue1:
            self.queue2.append(self.queue1.popleft())
        # Swap the names of the queues
        self.queue1, self.queue2 = self.queue2, self.queue1

    def pop(self) -> int:
        # Pop from the front of queue1
        return self.queue1.popleft()

    def top(self) -> int:
        # Get the front element of queue1
        return self.queue1[0]

    def empty(self) -> bool:
        # Check if queue1 is empty
        return not self.queue1

```

### Running the Code:

1. **Initialization**: To create an instance of `MyStack`, you would call `myStack = MyStack()`.
2. **Push Method**: To push values onto the stack, you would call `myStack.push(1)` and `myStack.push(2)`.
3. **Top Method**: To see the top element without removing it, you call `myStack.top()`.
4. **Pop Method**: To remove the top element, you would use `myStack.pop()`.
5. **Empty Check**: To check if the stack is empty, call `myStack.empty()`.

This code can be directly run on LeetCode as it follows the required class structure and method names.

# 232. Implement Queue using Stacks

### Problem Description 
Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (`push`, `peek`, `pop`, and `empty`).

Implement the `MyQueue` class:
`void push(int x)` Pushes element x to the back of the queue.

`int pop()` Removes the element from the front of the queue and returns it.

`int peek()` Returns the element at the front of the queue.

`boolean empty()` Returns `true` if the queue is empty, `false` otherwise.

Notes:
You must use only standard operations of a stack, which means only `push to top`, `peek/pop from top`, `size`, and `is empty` operations are valid.

Depending on your language, the stack may not be supported natively. You may simulate a stack using a list or deque (double-ended queue) as long as you use only a stack's standard operations.

Follow-up: Can you implement the queue such that each operation is amortized `O(1)` time complexity? In other words, performing `n` operations will take overall `O(n)` time even if one of those operations may take longer.


Example 1:
Input
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 1, 1, false]
Explanation
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false

Constraints:
`1 <= x <= 9`
At most `100` calls will be made to `push`, `pop`, `peek`, and `empty`.

All the calls to `pop` and `peek` are valid.

### Solution 
 To solve the problem of implementing a first-in-first-out (FIFO) queue using two stacks, we can take advantage of the inherent behavior of stacks (LIFO - last in, first out) to simulate the FIFO behavior of a queue. The idea is to use two stacks to manage the queue operations effectively.

### Explanation

We will use two stacks:
- **Stack 1** (`stack_in`) is used to store the incoming elements (push operations).
- **Stack 2** (`stack_out`) is used to store elements for outgoing operations (pop and peek).

Here's how the operations will work:
1. **Push Operation**: When we want to push an element onto the queue, we simply add it to `stack_in`.

2. **Pop Operation**: When we pop an element, we need to ensure that the elements are in the correct order to respect the FIFO nature:
   - If `stack_out` is empty, we will transfer all elements from `stack_in` to `stack_out` (this reverses their order).
   - We then pop the top of `stack_out`, which will be the front of the queue.

3. **Peek Operation**: Similar to pop, if `stack_out` is empty, we transfer elements from `stack_in` to `stack_out`. We return the top element of `stack_out`, which represents the front of the queue.

4. **Empty Operation**: The queue is empty if both stacks are empty.

### Amortized O(1) Time Complexity
The amortized time complexity for each operation can be analyzed as follows:
- Each element is pushed onto `stack_in` once and transferred to `stack_out` once. Thus, when considering multiple operations, each element contributes to O(1) time on average to the queue operations.

### Implementation

Here's how the `MyQueue` class can be implemented in Python:



```python
class MyQueue:

    def __init__(self):
        self.stack_in = []
        self.stack_out = []

    def push(self, x: int) -> None:
        # Push element to the back of the queue
        self.stack_in.append(x)

    def pop(self) -> int:
        # Pop the element from the front of the queue
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        return self.stack_out.pop()

    def peek(self) -> int:
        # Get the front element of the queue
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        return self.stack_out[-1]

    def empty(self) -> bool:
        # Check if the queue is empty
        return not self.stack_in and not self.stack_out

# Example usage:
# myQueue = MyQueue()
# myQueue.push(1)        # Queue is: [1]
# myQueue.push(2)        # Queue is: [1, 2]
# print(myQueue.peek())  # returns 1
# print(myQueue.pop())   # returns 1, Queue is: [2]
# print(myQueue.empty()) # returns False

```

### Summary of Methods:
- `push(x)`: Adds `x` to the end of the queue.
- `pop()`: Removes and returns the front element of the queue.
- `peek()`: Returns the front element without removing it.
- `empty()`: Returns `True` if the queue is empty; otherwise, returns `False`.

This implementation works efficiently and adheres to the constraints laid out in the problem, ensuring correct FIFO behavior with the use of two stacks.

# 244. Shortest Word Distance II

### Problem Description 
Design a data structure that will be initialized with a string array, and then it should answer queries of the shortest distance between two different strings from the array.

Implement the `WordDistance` class:
`WordDistance(String[] wordsDict)` initializes the object with the strings array `wordsDict`.

`int shortest(String word1, String word2)` returns the shortest distance between `word1` and `word2` in the array `wordsDict`.


Example 1:
Input
["WordDistance", "shortest", "shortest"]
[[["practice", "makes", "perfect", "coding", "makes"]], ["coding", "practice"], ["makes", "coding"]]
Output
[null, 3, 1]
Explanation
WordDistance wordDistance = new WordDistance(["practice", "makes", "perfect", "coding", "makes"]);
wordDistance.shortest("coding", "practice"); // return 3
wordDistance.shortest("makes", "coding");    // return 1

Constraints:
`1 <= wordsDict.length <= 3 * 104`
`1 <= wordsDict[i].length <= 10`
`wordsDict[i]` consists of lowercase English letters.

`word1` and `word2` are in `wordsDict`.

`word1 != word2`
At most `5000` calls will be made to `shortest`.

### Solution 
 To solve the problem of finding the shortest distance between two different strings in a given list, we'll design a `WordDistance` class that maintains the input strings along with their respective indices. The goal is to efficiently answer multiple queries regarding the shortest distance between pairs of words. Below is a step-by-step explanation of how we can implement this, followed by the complete Python code.

### Step-by-Step Explanation:

1. **Initialize the Data Structure:**
   We define a class `WordDistance` with an `__init__` method that takes a list of strings (`wordsDict`). We'll use a dictionary to store the indices of each word. This will allow us to quickly access where each word appears in the list.

2. **Store Word Indices:**
   As we iterate through `wordsDict`, we populate the dictionary (`word_indices`) where each key is a unique word, and the value is a list of positions (indices) where that word occurs.

3. **Calculate Shortest Distance:**
   For the `shortest` method, which takes two words as input, we first retrieve their indices from the stored dictionary. The core of our solution involves a two-pointer technique:
   - We maintain pointers for each list of indices.
   - By comparing the indices pointed to by the two pointers, we compute the current distance and update the minimum distance accordingly.
   - We advance the pointer that is pointing to the smaller index to potentially find a closer pair.

4. **Return the Minimum Distance:**
   After iterating over the indices, we will have the shortest distance calculated, which we then return.

### Implementation:

Here is the complete implementation of the `WordDistance` class in Python:



```python
class WordDistance:

    def __init__(self, wordsDict):
        self.word_indices = {}
        
        # Populate the dictionary with indices of each word
        for index, word in enumerate(wordsDict):
            if word not in self.word_indices:
                self.word_indices[word] = []
            self.word_indices[word].append(index)

    def shortest(self, word1: str, word2: str) -> int:
        indices1 = self.word_indices[word1]
        indices2 = self.word_indices[word2]
        
        i, j = 0, 0
        min_distance = float('inf')
        
        # Use a two-pointer approach to find the shortest distance
        while i < len(indices1) and j < len(indices2):
            pos1 = indices1[i]
            pos2 = indices2[j]
            min_distance = min(min_distance, abs(pos1 - pos2))
            
            # Move the pointer for the smaller index
            if pos1 < pos2:
                i += 1
            else:
                j += 1
        
        return min_distance

```

### Explanation of Example:
Given the example input:


```python
["WordDistance", "shortest", "shortest"]
[[["practice", "makes", "perfect", "coding", "makes"]], ["coding", "practice"], ["makes", "coding"]]

```
1. The `WordDistance` object is initialized with words `["practice", "makes", "perfect", "coding", "makes"]`.
2. The first query `shortest("coding", "practice")` computes the indices of both words and finds their shortest distance, which is `3`.
3. The second query `shortest("makes", "coding")` finds the indices for both words and the shortest distance is `1`.

This implementation efficiently supports multiple queries after the initial setup, ensuring that each query runs in linear time with respect to the number of occurrences of the queried words. Given the constraints, this approach is optimal and meets the requirements of the problem.

# 251. Flatten 2D Vector

### Problem Description 
Design an iterator to flatten a 2D vector. It should support the `next` and `hasNext` operations.

Implement the `Vector2D` class:
`Vector2D(int[][] vec)` initializes the object with the 2D vector `vec`.

`next()` returns the next element from the 2D vector and moves the pointer one step forward. You may assume that all the calls to `next` are valid.

`hasNext()` returns `true` if there are still some elements in the vector, and `false` otherwise.


Example 1:
Input
["Vector2D", "next", "next", "next", "hasNext", "hasNext", "next", "hasNext"]
[[[[1, 2], [3], [4]]], [], [], [], [], [], [], []]
Output
[null, 1, 2, 3, true, true, 4, false]
Explanation
Vector2D vector2D = new Vector2D([[1, 2], [3], [4]]);
vector2D.next();    // return 1
vector2D.next();    // return 2
vector2D.next();    // return 3
vector2D.hasNext(); // return True
vector2D.hasNext(); // return True
vector2D.next();    // return 4
vector2D.hasNext(); // return False

Constraints:
`0 <= vec.length <= 200`
`0 <= vec[i].length <= 500`
`-500 <= vec[i][j] <= 500`
At most `105` calls will be made to `next` and `hasNext`.

Follow up: As an added challenge, try to code it using only iterators in C++ or iterators in Java.

### Solution 
 To solve the problem of flattening a 2D vector using an iterator, we can create a `Vector2D` class that encapsulates the 2D vector and provides methods to traverse through its elements. The key methods we need to implement are `next()` and `hasNext()`. 

### Explanation:

1. **Initialization**: 
   - Create a class `Vector2D` that takes a 2D list `vec` as input during initialization.
   - Store the vector and maintain an initial position (`row` and `col`) to track the current element.

2. **hasNext() Method**: 
   - This method checks if there are more elements to be accessed. It iterates through the 2D vector starting from the current position to find the next valid element.

3. **next() Method**: 
   - This method retrieves the next element in the flattened order. Similar to `hasNext()`, it updates the current position to point to the next element after returning the current one.

### Implementation:

Here’s how this can be implemented in Python following the LeetCode solution format:



```python
class Vector2D:
    def __init__(self, vec: List[List[int]]):
        self.vec = vec  # Store the 2D vector
        self.row = 0    # Initialize row pointer
        self.col = 0    # Initialize column pointer
        # Move to the first valid position
        self._move_to_next()

    def _move_to_next(self) -> None:
        # This function moves the row and col pointers to the next valid element
        while self.row < len(self.vec) and self.col == len(self.vec[self.row]):
            self.row += 1  # Move to the next row
            self.col = 0   # Reset col to 0
          
    def next(self) -> int:
        # Assumes that next() is called only if hasNext() is true
        result = self.vec[self.row][self.col]  # Get the current element
        self.col += 1  # Move to the next column in the current row
        self._move_to_next()  # Move to the next valid element
        return result

    def hasNext(self) -> bool:
        # Check if there are more elements to iterate
        return self.row < len(self.vec) and self.col < len(self.vec[self.row])

```

### How It Works:

1. **Initialization**: The constructor `_init_` initializes the row and column pointers and calls `_move_to_next` to ensure we start at a valid position.

2. **_move_to_next**: 
   - It checks if the current row has been fully iterated (if `col` has reached the length of that row). If so, it increments the row and resets the column, continuing this process until it finds a valid row with elements.

3. **next()**:
   - It fetches the current element based on the `row` and `col`, increments the `col`, and calls `_move_to_next` again to update the state for any subsequent calls.

4. **hasNext()**: 
   - This method simply checks if the current pointers are within the bounds of the 2D vector.

### Example Usage:
The following sequence of calls can be made to the `Vector2D` instance to retrieve flattened elements:



```python
vector2D = Vector2D([[1, 2], [3], [4]])
print(vector2D.next())    # returns 1
print(vector2D.next())    # returns 2
print(vector2D.next())    # returns 3
print(vector2D.hasNext()) # returns True
print(vector2D.hasNext()) # returns True
print(vector2D.next())    # returns 4
print(vector2D.hasNext()) # returns False

```

This code can be directly used in the LeetCode code editor to solve the problem efficiently, complying with the constraints provided.

# 281. Zigzag Iterator

### Problem Description 
Given two vectors of integers `v1` and `v2`, implement an iterator to return their elements alternately.

Implement the `ZigzagIterator` class:
`ZigzagIterator(List<int> v1, List<int> v2)` initializes the object with the two vectors `v1` and `v2`.

`boolean hasNext()` returns `true` if the iterator still has elements, and `false` otherwise.

`int next()` returns the current element of the iterator and moves the iterator to the next element.


Example 1:
Input: v1 = [1,2], v2 = [3,4,5,6]
Output: [1,3,2,4,5,6]
Explanation: By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,3,2,4,5,6].


Example 2:
Input: v1 = [1], v2 = []
Output: [1]

Example 3:
Input: v1 = [], v2 = [1]
Output: [1]

Constraints:
`0 <= v1.length, v2.length <= 1000`
`1 <= v1.length + v2.length <= 2000`
`-231 <= v1[i], v2[i] <= 231 - 1`
Follow up: What if you are given `k` vectors? How well can your code be extended to such cases?
Clarification for the follow-up question:
The "Zigzag" order is not clearly defined and is ambiguous for `k > 2` cases. If "Zigzag" does not look right to you, replace "Zigzag" with "Cyclic".


Example:
Input: v1 = [1,2,3], v2 = [4,5,6,7], v3 = [8,9]
Output: [1,4,8,2,5,9,3,6,7]

### Solution 
 To solve the problem of creating a zigzag iterator over two vectors, we will implement a class called `ZigzagIterator`. This class will allow us to alternate between elements of the two input lists, `v1` and `v2`, and return elements one by one using the `next()` method, while the `hasNext()` method will let us know if there are any remaining elements to iterate over.

Here's how we can implement the `ZigzagIterator` class step by step:

1. **Initialize the iterator**: We'll use two queues to hold the remaining elements of `v1` and `v2`. This will help us easily fetch elements alternately.

2. **Alternate fetching elements**: We'll alternate between the two vectors, so each call to `next()` will return an element from one of the vectors based on which one has remaining elements.

3. **Check for more elements**: The `hasNext()` method will check if there are any elements left in either of the vectors.

Let’s dive into the implementation:



```python
class ZigzagIterator:
    def __init__(self, v1: List[int], v2: List[int]):
        self.v1 = v1
        self.v2 = v2
        self.index1 = 0  # Pointer for v1
        self.index2 = 0  # Pointer for v2
        self.turn = 0  # 0 for v1's turn, 1 for v2's turn

    def hasNext(self) -> bool:
        # Check if any elements left in either vector
        return self.index1 < len(self.v1) or self.index2 < len(self.v2)

    def next(self) -> int:
        if not self.hasNext():
            raise StopIteration("No more elements to iterate.")

        if self.turn == 0:
            if self.index1 < len(self.v1):
                result = self.v1[self.index1]
                self.index1 += 1
            else:
                # If v1 is exhausted, directly take from v2
                result = self.v2[self.index2]
                self.index2 += 1
            self.turn = 1  # Switch turn to v2
        else:
            if self.index2 < len(self.v2):
                result = self.v2[self.index2]
                self.index2 += 1
            else:
                # If v2 is exhausted, directly take from v1
                result = self.v1[self.index1]
                self.index1 += 1
            self.turn = 0  # Switch turn to v1

        return result

```

### Explanation of the Code

1. **Initialization (`__init__` method)**:
   - We initialize the two vectors `v1` and `v2`, and also set two index pointers (`index1` for `v1` and `index2` for `v2`) starting at 0. The `turn` variable is used to track whose turn it is to provide an element (0 for `v1`, 1 for `v2`).

2. **Checking for next element (`hasNext` method)**:
   - This method checks if there are any remaining elements in either `v1` or `v2` by comparing the current indices with their respective lengths.

3. **Fetching the next element (`next` method)**:
   - This method first checks if there are any elements left using `hasNext()`. If no elements are left, it raises a `StopIteration` exception.
   - Depending on whose turn it is, it fetches the next element from the corresponding vector. If the current vector is exhausted, it will switch to the other vector.
   - After fetching an element, it updates the index and toggles the turn.

### Usage Example

To use the `ZigzagIterator`, create an instance of the class with two lists, and then call `next()` in a loop while checking `hasNext()`:



```python
v1 = [1, 2]
v2 = [3, 4, 5, 6]

zigzag_iterator = ZigzagIterator(v1, v2)
result = []

while zigzag_iterator.hasNext():
    result.append(zigzag_iterator.next())

print(result)  # Output: [1, 3, 2, 4, 5, 6]

```

In this way, the `ZigzagIterator` class effectively alternates between two vectors, yielding their elements in a zigzag fashion according to the defined rules. If you need to extend the iterator to `k` vectors, a similar approach using a round-robin mechanism can be employed.

# 284. Peeking Iterator

### Problem Description 
Design an iterator that supports the `peek` operation on a list in addition to the `hasNext` and the `next` operations.

Implement the `PeekingIterator` class:
`PeekingIterator(int[] nums)` Initializes the object with the given integer array `nums`.

`int next()` Returns the next element in the array and moves the pointer to the next element.

`bool hasNext()` Returns `true` if there are still elements in the array.

`int peek()` Returns the next element in the array without moving the pointer.


Example 1:
Input
["PeekingIterator", "next", "peek", "next", "next", "hasNext"]
[[[1, 2, 3]], [], [], [], [], []]
Output
[null, 1, 2, 2, 3, false]
Explanation
PeekingIterator peekingIterator = new PeekingIterator([1, 2, 3]); // [1,2,3]
peekingIterator.next();    // return 1, the pointer moves to the next element [1,2,3].

peekingIterator.peek();    // return 2, the pointer does not move [1,2,3].

peekingIterator.next();    // return 2, the pointer moves to the next element [1,2,3]
peekingIterator.next();    // return 3, the pointer moves to the next element [1,2,3]
peekingIterator.hasNext(); // return False

Constraints:
`1 <= nums.length <= 1000`
`1 <= nums[i] <= 1000`
All the calls to `next` and `peek` are valid.

At most `1000` calls will be made to `next`, `hasNext`, and `peek`.

Follow up: How would you extend your design to be generic and work with all types, not just integer?

### Solution 
 To solve the problem of designing a `PeekingIterator` class that supports peek, next, and hasNext operations on a list, we need to maintain the state of the traversal through the list and provide additional functionality to peek at the next element without consuming it.

### Implementation Details
1. **Initialization**: When the `PeekingIterator` is created, it will initialize an internal list from the provided input and set a pointer (`index`) to track the current position.
  
2. **Next Method**: The `next` method will return the current element pointed by `index`, and then increment `index` to move to the next element.

3. **Has Next Method**: The `hasNext` method will check if there are more elements left in the list by comparing the current index with the length of the list.

4. **Peek Method**: The `peek` method will return the current element at `index` without advancing the `index`.

### Code Implementation
Here is the implementation of the `PeekingIterator` class in Python:



```python
class PeekingIterator:
    def __init__(self, nums: List[int]):
        # Initialize the iterator with the provided list.
        self.nums = nums
        self.index = 0
        self.next_elem = None if not nums else nums[0]  # Initialize next_elem to the first element if it exists

    def next(self) -> int:
        # Store the current element to return it later
        current = self.nums[self.index]
        self.index += 1  # Move the pointer to the next element
        # Update next_elem for peek operation
        self.next_elem = None if self.index >= len(self.nums) else self.nums[self.index]
        return current

    def hasNext(self) -> bool:
        # Check if there are more elements to iterate
        return self.index < len(self.nums)

    def peek(self) -> int:
        # Return the next element without advancing the index
        return self.next_elem

```

### Explanation of Each Method

1. **__init__()**: 
   - It receives a list of integers `nums` and initializes the internal state.
   - It sets `index` to 0 to track the current position and determines `next_elem`, which will store the next available element.

2. **next()**:
   - This method fetches the current element at position `index` and stores it in `current`.
   - The `index` is then incremented to prepare for the next call.
   - Finally, it updates `next_elem` for the `peek` method, ensuring that we are always aware of the next element before returning `current`.

3. **hasNext()**:
   - This method checks whether there are more elements to traverse by comparing `index` with the length of `nums`.

4. **peek()**:
   - Returns the next element pointed by `index` without changing the index position.

### Example of Using the Class

Using the class is straightforward, as shown in the provided example. You can instantiate a `PeekingIterator` with a list of integers and then call `next`, `peek`, and `hasNext` as desired to interact with it.

### Follow-Up
To extend this design to be generic (working with types other than integers), you can simply change the input type in the constructor and methods to accept any type, as the logic would not depend on the type of items in the list, just the operations being performed on them.

### Complete Running Code
Make sure to use the class within a testing function or a main block to see it in action. This is how you would run the instantiation and method calls:



```python
if __name__ == "__main__":
    nums = [1, 2, 3]
    peekingIterator = PeekingIterator(nums)
    print(peekingIterator.next())    # Output: 1
    print(peekingIterator.peek())     # Output: 2
    print(peekingIterator.next())     # Output: 2
    print(peekingIterator.next())     # Output: 3
    print(peekingIterator.hasNext())  # Output: False

```

This code is ready to be run on LeetCode directly.

# 288. Unique Word Abbreviation

### Problem Description 
The abbreviation of a word is a concatenation of its first letter, the number of characters between the first and last letter, and its last letter. If a word has only two characters, then it is an abbreviation of itself.

For example:
`dog --> d1g` because there is one letter between the first letter `'d'` and the last letter `'g'`.

`internationalization --> i18n` because there are 18 letters between the first letter `'i'` and the last letter `'n'`.

`it --> it` because any word with only two characters is an abbreviation of itself.

Implement the `ValidWordAbbr` class:
`ValidWordAbbr(String[] dictionary)` Initializes the object with a `dictionary` of words.

`boolean isUnique(string word)` Returns `true` if either of the following conditions are met (otherwise returns `false`):
	
There is no word in `dictionary` whose abbreviation is equal to `word`'s abbreviation.

For any word in `dictionary` whose abbreviation is equal to `word`'s abbreviation, that word and `word` are the same.


Example 1:
Input
["ValidWordAbbr", "isUnique", "isUnique", "isUnique", "isUnique"]
[[["deer", "door", "cake", "card"]], ["dear"], ["cart"], ["cane"], ["make"]]
Output
[null, false, true, false, true]
Explanation
ValidWordAbbr validWordAbbr = new ValidWordAbbr(["deer", "door", "cake", "card"]);
validWordAbbr.isUnique("dear"); // return false, dictionary word "deer" and word "dear" have the same abbreviation
                                // "d2r" but are not the same.

validWordAbbr.isUnique("cart"); // return true, no words in the dictionary have the abbreviation "c2t".

validWordAbbr.isUnique("cane"); // return false, dictionary word "cake" and word "cane" have the same abbreviation 
                                // "c2e" but are not the same.

validWordAbbr.isUnique("make"); // return true, no words in the dictionary have the abbreviation "m2e".

validWordAbbr.isUnique("cake"); // return true, because "cake" is already in the dictionary and no other word in the dictionary has "c2e" abbreviation.


Constraints:
`1 <= dictionary.length <= 3 * 104`
`1 <= dictionary[i].length <= 20`
`dictionary[i]` consists of lowercase English letters.

`1 <= word.length <= 20`
`word` consists of lowercase English letters.

At most `5000` calls will be made to `isUnique`.

### Solution 
 To solve the problem of checking the uniqueness of word abbreviations from a given dictionary, we can create a class called `ValidWordAbbr`. This class will have a constructor that initializes the dictionary and a method called `isUnique` that will check if a word's abbreviation is unique in relation to the words in the dictionary.

### Approach:

1. **Define the Abbreviation**: 
   - The abbreviation is constructed based on the first letter, the count of characters between the first and last letters, and the last letter.
   - The abbreviation for:
     - Words with length < 3 return the word itself.
     - For words with length >= 3: `abbreviation = first_letter + count_between + last_letter`.

2. **Store Abbreviations**:
   - Use a dictionary to store the abbreviations for all words in the dictionary. The key will be the abbreviation and the value will be a set of words producing that abbreviation.
   
3. **Check Uniqueness**:
   - For the given word, generate its abbreviation and check:
     - If the abbreviation is not present in the dictionary, return True.
     - If it is present, check if the only word associated with that abbreviation is the word itself. If so, return True; otherwise, return False.

### Implementation:

Here is the code written in the LeetCode format that can be copied and run directly on the LeetCode website:



```python
class ValidWordAbbr:

    def __init__(self, dictionary: list[str]):
        self.abbr_map = {}
        
        for word in dictionary:
            abbr = self._get_abbreviation(word)
            if abbr not in self.abbr_map:
                self.abbr_map[abbr] = set()
            self.abbr_map[abbr].add(word)

    def _get_abbreviation(self, word: str) -> str:
        n = len(word)
        if n <= 2:
            return word
        return f"{word[0]}{n - 2}{word[-1]}"

    def isUnique(self, word: str) -> bool:
        abbr = self._get_abbreviation(word)
        
        # Check if abbreviation exists
        if abbr not in self.abbr_map:
            return True

        # Check the uniqueness condition
        words_with_same_abbr = self.abbr_map[abbr]
        if len(words_with_same_abbr) == 1 and word in words_with_same_abbr:
            return True
        
        return False

# Example usage:
# validWordAbbr = ValidWordAbbr(["deer", "door", "cake", "card"])
# print(validWordAbbr.isUnique("dear"))  # return false
# print(validWordAbbr.isUnique("cart"))  # return true
# print(validWordAbbr.isUnique("cane"))  # return false
# print(validWordAbbr.isUnique("make"))  # return true
# print(validWordAbbr.isUnique("cake"))  # return true

```

### Explanation:

- **Constructor (`__init__`)**: We build the `abbr_map` which maps abbreviations to the set of words that correspond to those abbreviations.
- **Method `_get_abbreviation`**: This helper function generates the abbreviation for any given word.
- **Method `isUnique`**: 
  - It first checks if the abbreviation exists in the dictionary.
  - If it does and is unique (only one word corresponds to that abbreviation and it is the same as the given word), it returns `True`. Otherwise, it returns `False`.

### Notes:
1. The algorithm uses a dictionary and set data structures which allow for average O(1) time complexity for insertions and lookups.
2. This implementation is efficient and can handle the constraints provided in the problem statement.

# 295. Find Median from Data Stream

### Problem Description 
The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.

For example, for `arr = [2,3,4]`, the median is `3`.

For example, for `arr = [2,3]`, the median is `(2 + 3) / 2 = 2.5`.

Implement the MedianFinder class:
`MedianFinder()` initializes the `MedianFinder` object.

`void addNum(int num)` adds the integer `num` from the data stream to the data structure.

`double findMedian()` returns the median of all elements so far. Answers within `10-5` of the actual answer will be accepted.


Example 1:
Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]
Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0

Constraints:
`-105 <= num <= 105`
There will be at least one element in the data structure before calling `findMedian`.

At most `5 * 104` calls will be made to `addNum` and `findMedian`.

Follow up:
If all integer numbers from the stream are in the range `[0, 100]`, how would you optimize your solution?
If `99%` of all integer numbers from the stream are in the range `[0, 100]`, how would you optimize your solution?

### Solution 
 To solve the problem of finding the median in a data stream, we can use a data structure approach utilizing two heaps (priority queues). This will allow us to efficiently keep track of the numbers and calculate the median.

Here’s how the solution is structured:

1. **Data Structure**: We will use two heaps:
   - A max-heap (`left`) to store the smaller half of the numbers.
   - A min-heap (`right`) to store the larger half of the numbers.

2. **Adding a Number**: When a new number is added:
   - If the number is less than or equal to the maximum of the `left` heap, it goes into the `left` heap.
   - Otherwise, it goes into the `right` heap.
   - We then ensure that the heaps are balanced; the `left` heap can have at most one more element than the `right` heap.

3. **Finding the Median**: 
   - If the sizes of both heaps are equal, the median is the average of the tops of both heaps.
   - If the `left` heap has one more element than the `right`, the median is the top of the `left` heap.

Here is the implementation in Python that follows the required structure for use on LeetCode:



```python
import heapq

class MedianFinder:
    def __init__(self):
        # Max-heap for the left half (invert values for Python's min-heap)
        self.left = []  
        # Min-heap for the right half
        self.right = []  
    
    def addNum(self, num: int) -> None:
        # Add to max-heap (invert the value for max-heap simulation)
        heapq.heappush(self.left, -num)
        
        # Balance the heaps: max of left should not be > min of right
        if (self.left and self.right and 
            (-self.left[0] > self.right[0])):
            heapq.heappush(self.right, -heapq.heappop(self.left))

        # Balance the sizes: left can be at most 1 item longer than right
        if len(self.left) > len(self.right) + 1:
            heapq.heappush(self.right, -heapq.heappop(self.left))
    
    def findMedian(self) -> float:
        if len(self.left) > len(self.right):
            return float(-self.left[0])  # The max of left (inverted back to positive)
        else:
            return (-self.left[0] + self.right[0]) / 2.0  # Average of both heaps

# Example usage:
# medianFinder = MedianFinder()
# medianFinder.addNum(1)
# medianFinder.addNum(2)
# print(medianFinder.findMedian())  # Output 1.5
# medianFinder.addNum(3)
# print(medianFinder.findMedian())  # Output 2.0

```

### Explanation of the Code:
- `__init__`: Initializes the two heaps.
- `addNum(num)`: Adds a new number to the appropriate heap and manages the balance between the two heaps.
- `findMedian()`: Returns the current median based on the sizes and tops of the heaps.

### Complexity Analysis:
- Adding a number takes O(log n) time due to the heap operations.
- Finding the median takes O(1) time as it only involves accessing the tops of the heaps.
- Space complexity is O(n) for storing the numbers in the heaps.

### Optimized Solutions for Constraints:
1. If all integers from the stream are in range [0, 100]:
   - Use a fixed-sized array (or a simple list) to count occurrences of each number. This will allow for O(1) adjustment of numbers and O(n) to calculate the median after adding each new number.

2. If 99% of integers are in the range [0, 100]:
   - Similar to above, use a frequency array for numbers in [0, 100] and maintain a separate min-heap or max-heap for the remaining numbers outside this range. This splits the average case operations effectively between fast counting for common values and heap operations for uncommon values.

The provided solution will run efficiently on LeetCode for the given problem constraints.

# 297. Serialize and Deserialize Binary Tree

### Problem Description 
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Clarification: The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.


Example 1:
Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]

Example 2:
Input: root = []
Output: []

Example 3:
Input: root = [1]
Output: [1]

Example 4:
Input: root = [1,2]
Output: [1,2]

Constraints:
The number of nodes in the tree is in the range `[0, 104]`.

`-1000 <= Node.val <= 1000`

### Solution 
 To solve the problem of serializing and deserializing a binary tree, we'll use a breadth-first traversal (BFS) approach to serialize the tree into a string format. For deserialization, we will convert that string back into a binary tree structure.

Serialization is the process of converting the tree into a string, while deserialization converts the string back into the original binary tree structure.

### Steps Breakdown:

1. **Serialization**:
   - We'll perform a level-order traversal using a queue data structure.
   - For each node, we will add its value to a list. If a node is `None`, we'll represent it with a placeholder (like "null").
   - Finally, we will join the list into a single string using a separator (like a comma).

2. **Deserialization**:
   - We'll split the string back into a list using the same separator.
   - Using the values from the list, we will reconstruct the tree by sequentially creating nodes and adding them to the tree based on binary tree properties.

### Implementation:
Here is how the above logic can be implemented in Python:



```python
from collections import deque
from typing import Optional, List

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Codec:
    def serialize(self, root: Optional[TreeNode]) -> str:
        # Base case: if the root is None, return empty string
        if not root:
            return ""
        
        result = []
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            if node:
                result.append(str(node.val))  # Add the node's value
                queue.append(node.left)        # Add left child to the queue
                queue.append(node.right)       # Add right child to the queue
            else:
                result.append("null")          # Use "null" for empty nodes
        
        return ",".join(result)              # Join all values into a string
    
    def deserialize(self, data: str) -> Optional[TreeNode]:
        # If the data is empty, return None
        if not data:
            return None
        
        nodes = data.split(",")
        root = TreeNode(int(nodes[0]))         # The first item is the root
        queue = deque([root])                  # Start with the root in the queue
        
        index = 1                               # Start from the second item
        
        while queue and index < len(nodes):
            node = queue.popleft()              # Get the current node
            
            # Create the left child if it's not "null"
            if nodes[index] != "null":
                node.left = TreeNode(int(nodes[index]))
                queue.append(node.left)
            index += 1
            
            # Create the right child if it's not "null"
            if index < len(nodes) and nodes[index] != "null":
                node.right = TreeNode(int(nodes[index]))
                queue.append(node.right)
            index += 1
        
        return root                             # Return the reconstructed tree

# Example usage
# codec = Codec()
# tree_str = codec.serialize(root)         # Serialize the tree
# tree_root = codec.deserialize(tree_str)  # Deserialize back to tree structure

```

### Explanation of the Code:

- **TreeNode Class**: This is a basic class definition for a binary tree node which includes the value and pointers to left and right children.
- **Codec Class**:
  - `serialize`: Uses a queue to perform a level-order traversal of the tree. It appends the value of each node into a list while replacing `None` nodes with "null". The result is converted into a single comma-separated string.
  - `deserialize`: Splits the input string back into values. It constructs the binary tree level by level, using the constructed list.

### Example
- For a tree represented as `[1,2,3,null,null,4,5]`:
  - `serialize` will return the string `"1,2,3,null,null,4,5"`.
  - `deserialize` will reconstruct and return the original tree structure.

This implementation is efficient and meets the specified constraints. It can handle trees with a large number of nodes up to 10,000 as specified in the problem statement.

# 341. Flatten Nested List Iterator

### Problem Description 
You are given a nested list of integers `nestedList`. Each element is either an integer or a list whose elements may also be integers or other lists. Implement an iterator to flatten it.

Implement the `NestedIterator` class:
`NestedIterator(List<NestedInteger> nestedList)` Initializes the iterator with the nested list `nestedList`.

`int next()` Returns the next integer in the nested list.

`boolean hasNext()` Returns `true` if there are still some integers in the nested list and `false` otherwise.


Example 1:
Input: nestedList = [[1,1],2,[1,1]]
Output: [1,1,2,1,1]
Explanation: By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,1,2,1,1].


Example 2:
Input: nestedList = [1,[4,[6]]]
Output: [1,4,6]
Explanation: By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,4,6].


Constraints:
`1 <= nestedList.length <= 500`
The values of the integers in the nested list is in the range `[-106, 106]`.

### Solution 
 To solve the problem of flattening a nested list of integers, we can create an iterator class called `NestedIterator`. This class will leverage a stack data structure to traverse through the nested list elements in a depth-first manner. Below is the detailed explanation and the Python implementation of the `NestedIterator` class:

### Class Structure

1. **NestedInteger Class**: We assume that the `NestedInteger` class provides the following methods:
   - `isInteger()` returns `True` if this `NestedInteger` holds a single integer.
   - `getInteger()` returns the single integer that this `NestedInteger` holds, if it is an integer.
   - `getList()` returns the nested list that this `NestedInteger` holds, if it holds a list.

2. **NestedIterator Class**:
   - `__init__(self, nestedList)`: Initializes the iterator with the nested list. We'll use a stack to keep track of our position in the nested structure.
   - `next(self)`: Returns the next integer in the nested list.
   - `hasNext(self)`: Checks if there are more integers left in the nested list.

### Implementation

Here's the complete implementation of the `NestedIterator` class:



```python
class NestedInteger:
    # This is a placeholder for the NestedInteger class 
    # The real class would have appropriate methods as described above.
    def isInteger(self): pass
    def getInteger(self): pass
    def getList(self): pass

class NestedIterator:
    def __init__(self, nestedList: List[NestedInteger]):
        # Stack for storing list to explore and current index.
        self.stack = []
        self.pushListToStack(nestedList)

    def pushListToStack(self, nestedList):
        # Push the entire nested list to the stack.
        # We reverse it to facilitate stack pop operation that handles elements in the correct order.
        self.stack.append(nestedList[::-1])

    def next(self) -> int:
        # Ensure we have the next element available.
        self.hasNext()
        # Pop the next element and return it.
        return self.stack[-1].pop().getInteger()

    def hasNext(self) -> bool:
        # Loop until we find an integer.
        while self.stack:
            # Get the current list from the top of the stack.
            currentList = self.stack[-1]
            if not currentList:  # If it's empty, pop it off the stack.
                self.stack.pop()
                continue
            # Check the last element.
            topElement = currentList[-1]
            if topElement.isInteger():  # If it's an integer, we are done.
                return True
            # It's a list, so pop it and push its elements onto the stack.
            currentList.pop()
            self.pushListToStack(topElement.getList())
        return False

```

### Explanation of Methods

- **Initialization (`__init__`)**: We initialize a stack with the reversed `nestedList` to simulate depth-first traversal. The stack keeps track of lists that we need to explore.

- **`pushListToStack`**: This function pushes a list onto the stack in reversed order to maintain the correct processing order when popping.

- **`next` Method**: This method calls `hasNext()` to ensure there are still integers to return, then pops the next integer from the current stack and retrieves its value.

- **`hasNext` Method**: This method checks if there are remaining integers to process. It iterates through the stack. If the current list is empty, it pops it off and continues. If the top element is an integer, it confirms that the method should return `True`. If it's another list, it will pop it and push its contents (also in reversed order) on the stack for further exploration.

### Usage Example

You can instantiate the `NestedIterator` class with your nested list and iterate over the integers as follows:



```python
nestedList = [NestedInteger(...), NestedInteger(...)]  # Replace with actual NestedInteger objects.
iterator = NestedIterator(nestedList)
result = []
while iterator.hasNext():
    result.append(iterator.next())
print(result)

```

This approach ensures that we can traverse the entire nested structure effectively, allowing us to retrieve integers in the required flattened order.

# 346. Moving Average from Data Stream

### Problem Description 
Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

Implement the `MovingAverage` class:
`MovingAverage(int size)` Initializes the object with the size of the window `size`.

`double next(int val)` Returns the moving average of the last `size` values of the stream.


Example 1:
Input
["MovingAverage", "next", "next", "next", "next"]
[[3], [1], [10], [3], [5]]
Output
[null, 1.0, 5.5, 4.66667, 6.0]
Explanation
MovingAverage movingAverage = new MovingAverage(3);
movingAverage.next(1); // return 1.0 = 1 / 1
movingAverage.next(10); // return 5.5 = (1 + 10) / 2
movingAverage.next(3); // return 4.66667 = (1 + 10 + 3) / 3
movingAverage.next(5); // return 6.0 = (10 + 3 + 5) / 3

Constraints:
`1 <= size <= 1000`
`-105 <= val <= 105`
At most `104` calls will be made to `next`.

### Solution 
 To solve the problem of calculating the moving average from a stream of integers with a specific window size, we can implement a class named `MovingAverage`. This class will maintain a list (or queue) of the last `n` integers and will compute the average every time a new integer is added.

### Detailed Explanation

1. **Initialization**: We will initialize the class with a size parameter that defines the maximum number of integers to keep in the moving average calculation. We will also keep a sum to quickly calculate averages without needing to loop through the entire list every time.

2. **Maintaining the Window**: Whenever a new value is added (via the `next` method), we need to determine if the window (the count of integers we are tracking) has reached its maximum size. If it has, we will remove the oldest value to maintain the correct size.

3. **Calculating the Average**: The moving average can be calculated simply as the current sum divided by the number of integers currently in the window.

4. **Efficiency**: This approach ensures that both `next` method calls (adding a new integer and computing the average) operate in constant time, O(1).

Here’s how the implementation looks:



```python
class MovingAverage:
    def __init__(self, size: int):
        """
        Initialize the data structure with the specified window size.
        """
        self.size = size
        self.window = []
        self.total = 0.0  # To keep track of the sum of the current window
        self.count = 0     # To count the number of elements in the window

    def next(self, val: int) -> float:
        """
        Add a new value to the moving average and return the current moving average.
        """
        if self.count < self.size:
            # If we haven't reached the max size, simply add and update the sum
            self.window.append(val)
            self.total += val
            self.count += 1
        else:
            # If we've reached the max size, remove the oldest value from the total
            oldest_value = self.window.pop(0)  # Remove the oldest value from the window
            self.total -= oldest_value           # Subtract the oldest value from the sum
            self.window.append(val)              # Add the new value
            self.total += val                    # Update the sum

        # Calculate the moving average
        return self.total / self.count

```

### Explanation of the Code

- **Initialization (`__init__`)**: 
  - `self.size`: stores the maximum size of the moving window.
  - `self.window`: a list to keep track of the integers in the current moving average window.
  - `self.total`: a float to hold the sum of the integers in the current window.
  - `self.count`: tracks how many integers have been added to the window.

- **Next Value Calculation (`next`)**: 
  - If the current count of integers is less than the size, we simply append the new integer, add it to the total, and increment the count.
  - If we have reached the maximum size of the window, we pop the oldest integer, adjust the total, and then append the new integer.
  - Finally, we compute and return the moving average as `self.total / self.count`.

### Example Usage

You can use this class as follows:



```python
movingAverage = MovingAverage(3)
print(movingAverage.next(1))    # return 1.0
print(movingAverage.next(10))   # return 5.5
print(movingAverage.next(3))    # return 4.66667
print(movingAverage.next(5))    # return 6.0

```

This code meets the problem's requirements and constraints, efficiently computing the moving average as specified.

# 348. Design Tic-Tac-Toe

### Problem Description 
Assume the following rules are for the tic-tac-toe game on an `n x n` board between two players:
A move is guaranteed to be valid and is placed on an empty block.

Once a winning condition is reached, no more moves are allowed.

A player who succeeds in placing `n` of their marks in a horizontal, vertical, or diagonal row wins the game.

Implement the `TicTacToe` class:
`TicTacToe(int n)` Initializes the object the size of the board `n`.

`int move(int row, int col, int player)` Indicates that player with id `player` plays at the cell `(row, col)` of the board. The move is guaranteed to be a valid move.

Follow up:
Could you do better than `O(n2)` per `move()` operation?

Example 1:
Input
["TicTacToe", "move", "move", "move", "move", "move", "move", "move"]
[[3], [0, 0, 1], [0, 2, 2], [2, 2, 1], [1, 1, 2], [2, 0, 1], [1, 0, 2], [2, 1, 1]]
Output
[null, 0, 0, 0, 0, 0, 0, 1]
Explanation
TicTacToe ticTacToe = new TicTacToe(3);
Assume that player 1 is "X" and player 2 is "O" in the board.

ticTacToe.move(0, 0, 1); // return 0 (no one wins)
|X| | |
| | | |    // Player 1 makes a move at (0, 0).

| | | |
ticTacToe.move(0, 2, 2); // return 0 (no one wins)
|X| |O|
| | | |    // Player 2 makes a move at (0, 2).

| | | |
ticTacToe.move(2, 2, 1); // return 0 (no one wins)
|X| |O|
| | | |    // Player 1 makes a move at (2, 2).

| | |X|
ticTacToe.move(1, 1, 2); // return 0 (no one wins)
|X| |O|
| |O| |    // Player 2 makes a move at (1, 1).

| | |X|
ticTacToe.move(2, 0, 1); // return 0 (no one wins)
|X| |O|
| |O| |    // Player 1 makes a move at (2, 0).

|X| |X|
ticTacToe.move(1, 0, 2); // return 0 (no one wins)
|X| |O|
|O|O| |    // Player 2 makes a move at (1, 0).

|X| |X|
ticTacToe.move(2, 1, 1); // return 1 (player 1 wins)
|X| |O|
|O|O| |    // Player 1 makes a move at (2, 1).

|X|X|X|

Constraints:
`2 <= n <= 100`
player is `1` or `2`.

`1 <= row, col <= n`
`(row, col)` are unique for each different call to `move`.

At most `n2` calls will be made to `move`.

### Solution 
 To solve the TicTacToe problem efficiently, we'll implement a `TicTacToe` class that keeps track of the game state and checks for winning conditions using data structures that allow us to check wins in O(1) time per move. Here’s the breakdown of the solution:

### Class Structure & Methods

1. **Initialization**:
   - We will initialize an `n x n` board using a 2D list.
   - We will maintain counts for rows, columns, and the two main diagonals for each player.

2. **Move Handling**:
   - For each move made by a player, we will increment the respective row, column, and diagonal count.
   - After each move, we will check if the current player has won by checking if any of the counts meet the required value, which is `n`.

3. **Winning Conditions**:
   - There are three sets of counts to monitor:
     - `rows`: Count of marks in each row for both players.
     - `cols`: Count of marks in each column for both players.
     - `diags`: Count of marks in both diagonals for both players (the main diagonal and anti-diagonal).

### Code Implementation

Here's the complete implementation formatted for use on LeetCode:



```python
class TicTacToe:

    def __init__(self, n: int):
        self.n = n
        self.rows = [0] * n  # Count for each row
        self.cols = [0] * n  # Count for each column
        self.diags = [0] * 2  # Count for two diagonals
        self.board = [[0] * n for _ in range(n)]  # Initialize board with 0s

    def move(self, row: int, col: int, player: int) -> int:
        # Update the board with the player's mark
        self.board[row][col] = player
        # Determine the count adjustment (+1 for player 1, -1 for player 2)
        count = 1 if player == 1 else -1
        
        # Update row and column counts
        self.rows[row] += count
        self.cols[col] += count
        
        # Update diagonal counts if applicable
        if row == col:
            self.diags[0] += count  # Main diagonal
        if row + col == self.n - 1:
            self.diags[1] += count  # Anti-diagonal
        
        # Check if the current player has won
        if (abs(self.rows[row]) == self.n or
            abs(self.cols[col]) == self.n or
            abs(self.diags[0]) == self.n or
            abs(self.diags[1]) == self.n):
            return player  # Return the winner player number

        return 0  # No winner

```

### Explanation of the Code

- **Initialization (`__init__`)**:
  - We create a list for `rows`, `cols`, and `diags` to keep track of scores for both players. The board itself is initialized with 0s indicating empty cells.

- **Move Method**:
  - The `move` method takes the row and column along with the player ID (1 or 2).
  - It updates the board and increments/decrements the corresponding row and column counts based on which player made the move.
  - If the move is on a diagonal, we also update the diagonal counts.
  - After updating the counts, we check if any of the counts match `n` (indicating a win) and return the player ID if they win; otherwise, return 0.

### Time Complexity
- Each move operation runs in O(1) time, as we only update and check a few integers. This makes the solution efficient and scalable for the maximum constraints given in the problem.

This complete code and explanation should work directly on LeetCode and efficiently handle the Tic Tac Toe game logic as required.

# 353. Design Snake Game

### Problem Description 
Design a Snake game that is played on a device with screen size `height x width`. Play the game online if you are not familiar with the game.

The snake is initially positioned at the top left corner `(0, 0)` with a length of `1` unit.

You are given an array `food` where `food[i] = (ri, ci)` is the row and column position of a piece of food that the snake can eat. When a snake eats a piece of food, its length and the game's score both increase by `1`.

Each piece of food appears one by one on the screen, meaning the second piece of food will not appear until the snake eats the first piece of food.

When a piece of food appears on the screen, it is guaranteed that it will not appear on a block occupied by the snake.

The game is over if the snake goes out of bounds (hits a wall) or if its head occupies a space that its body occupies after moving (i.e. a snake of length 4 cannot run into itself).

Implement the `SnakeGame` class:
`SnakeGame(int width, int height, int[][] food)` Initializes the object with a screen of size `height x width` and the positions of the `food`.

`int move(String direction)` Returns the score of the game after applying one `direction` move by the snake. If the game is over, return `-1`.


Example 1:
Input
["SnakeGame", "move", "move", "move", "move", "move", "move"]
[[3, 2, [[1, 2], [0, 1]]], ["R"], ["D"], ["R"], ["U"], ["L"], ["U"]]
Output
[null, 0, 0, 1, 1, 2, -1]
Explanation
SnakeGame snakeGame = new SnakeGame(3, 2, [[1, 2], [0, 1]]);
snakeGame.move("R"); // return 0
snakeGame.move("D"); // return 0
snakeGame.move("R"); // return 1, snake eats the first piece of food. The second piece of food appears
                     // at (0, 1).

snakeGame.move("U"); // return 1
snakeGame.move("L"); // return 2, snake eats the second food. No more food appears.

snakeGame.move("U"); // return -1, game over because snake collides with border

Constraints:
`1 <= width, height <= 104`
`1 <= food.length <= 50`
`food[i].length == 2`
`0 <= ri < height`
`0 <= ci < width`
`direction.length == 1`
`direction` is `'U'`, `'D'`, `'L'`, or `'R'`.

At most `104` calls will be made to `move`.

### Solution 
 To implement the Snake game according to the specifications provided, we need to build a class called `SnakeGame` that simulates the movement of the snake, manages the food consumption, checks for collisions, and maintains the score.

Here's a structured approach to the solution:

### Class Definition
1. **Initialization (`__init__` method)**: Set up the game parameters such as the width, height, initial position of the snake, food locations, score, and the snake's body coordinates.
2. **Movement Handling (`move` method)**: This method will handle the movement of the snake, check for collisions, and update the score as necessary.

### Detailed Implementation
Let’s walk through the implementation step-by-step:

- The snake will be represented as a list of tuples where each tuple contains the coordinates of a segment of the snake's body.
- The movement direction will be determined based on input, and the position of the snake's head will be updated according to the specified direction.
- After updating the head position, we need to check:
  - If the head goes out of bounds.
  - If the head collides with any part of its body.
  - If the head lands on a food item, the snake grows in length.

Here’s how we can implement this:



```python
class SnakeGame:
    def __init__(self, width: int, height: int, food: List[List[int]]):
        self.width = width
        self.height = height
        self.food = food
        self.snake = [(0, 0)]  # Initial position of the snake, starts at (0, 0)
        self.score = 0
        self.food_index = 0  # To keep track of eaten food
        self.directions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

    def move(self, direction: str) -> int:
        head_x, head_y = self.snake[0]  # Get the current head position
        delta_x, delta_y = self.directions[direction]  # Get the direction to move
        
        # Calculate new head position
        new_head = (head_x + delta_x, head_y + delta_y)

        # Check if new head is out of bounds
        if (new_head[0] < 0 or new_head[0] >= self.height or 
                new_head[1] < 0 or new_head[1] >= self.width):
            return -1  # Game over, snake hits the wall

        # Check if the snake runs into itself
        if new_head in self.snake:
            return -1  # Game over, snake collides with itself
        
        # Check if the snake eats the food
        if (self.food_index < len(self.food) and
                new_head == tuple(self.food[self.food_index])):
            self.score += 1  # Increase score
            self.snake.insert(0, new_head)  # Grow snake
            self.food_index += 1  # Move to the next food
        else:
            # Move the snake without eating food
            self.snake.insert(0, new_head)  # Move snake head
            self.snake.pop()  # Remove the tail
        
        return self.score

# Usage example
# snakeGame = SnakeGame(3, 2, [[1, 2], [0, 1]])
# print(snakeGame.move("R"))  # return 0
# print(snakeGame.move("D"))  # return 0
# print(snakeGame.move("R"))  # return 1
# print(snakeGame.move("U"))  # return 1
# print(snakeGame.move("L"))  # return 2
# print(snakeGame.move("U"))  # return -1

```

### Explanation of Key Components:
- **Initialization**: We store the game state and initialize the snake's position.
- **Move Method**: 
  - It calculates the new head position based on the current position and direction.
  - It checks for boundary conditions to determine if the snake has collided with the walls.
  - It checks if the new head position occupies the same space as its body to check for self-collision.
  - If the snake eats food, we increase the score and grow the snake; otherwise, we move the snake forward and remove the tail.

### Complexities:
- Time complexity is approximately O(1) for each move, as operations like checking for self-collision and handling food are performed in constant time.
- Space complexity depends on the length of the snake, O(L), where L is the length of the snake at any point in time.

This implementation should work seamlessly for the given problem's constraints.

# 355. Design Twitter

### Problem Description 
Design a simplified version of Twitter where users can post tweets, follow/unfollow another user, and is able to see the `10` most recent tweets in the user's news feed.

Implement the `Twitter` class:
`Twitter()` Initializes your twitter object.

`void postTweet(int userId, int tweetId)` Composes a new tweet with ID `tweetId` by the user `userId`. Each call to this function will be made with a unique `tweetId`.

`List<Integer> getNewsFeed(int userId)` Retrieves the `10` most recent tweet IDs in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user themself. Tweets must be ordered from most recent to least recent.

`void follow(int followerId, int followeeId)` The user with ID `followerId` started following the user with ID `followeeId`.

`void unfollow(int followerId, int followeeId)` The user with ID `followerId` started unfollowing the user with ID `followeeId`.


Example 1:
Input
["Twitter", "postTweet", "getNewsFeed", "follow", "postTweet", "getNewsFeed", "unfollow", "getNewsFeed"]
[[], [1, 5], [1], [1, 2], [2, 6], [1], [1, 2], [1]]
Output
[null, null, [5], null, null, [6, 5], null, [5]]
Explanation
Twitter twitter = new Twitter();
twitter.postTweet(1, 5); // User 1 posts a new tweet (id = 5).

twitter.getNewsFeed(1);  // User 1's news feed should return a list with 1 tweet id -> [5]. return [5]
twitter.follow(1, 2);    // User 1 follows user 2.

twitter.postTweet(2, 6); // User 2 posts a new tweet (id = 6).

twitter.getNewsFeed(1);  // User 1's news feed should return a list with 2 tweet ids -> [6, 5]. Tweet id 6 should precede tweet id 5 because it is posted after tweet id 5.

twitter.unfollow(1, 2);  // User 1 unfollows user 2.

twitter.getNewsFeed(1);  // User 1's news feed should return a list with 1 tweet id -> [5], since user 1 is no longer following user 2.


Constraints:
`1 <= userId, followerId, followeeId <= 500`
`0 <= tweetId <= 104`
All the tweets have unique IDs.

At most `3 * 104` calls will be made to `postTweet`, `getNewsFeed`, `follow`, and `unfollow`.

### Solution 
 To implement a simplified version of Twitter, we need to manage users, their tweets, and their relationships (follow/unfollow). We will leverage data structures such as lists and dictionaries to efficiently manage the requirements.

Here's how we can approach the problem step by step:

### Class Design

1. **Data Structures**:
   - A dictionary to store users' tweets, using the user ID as the key and a list of their tweet IDs as the value.
   - Another dictionary to keep track of which users each user is following.
   - A counter to keep track of tweet IDs ensuring that each tweet ID is unique.

2. **Methods**:
   - `postTweet(userId, tweetId)`: Add a new tweet ID to the user's list of tweets.
   - `getNewsFeed(userId)`: Compile a list of tweets from the user’s feed, which includes their own tweets and those from users they follow, and return the 10 most recent ones.
   - `follow(followerId, followeeId)`: Allow one user to follow another.
   - `unfollow(followerId, followeeId)`: Allow one user to unfollow another.

### Implementation

Below is the implementation of the `Twitter` class, written in Python. This class is designed to be compliant with the described interface, and you should be able to run it directly on LeetCode.



```python
from collections import defaultdict
import heapq

class Twitter:
    def __init__(self):
        self.tweets = defaultdict(list)  # userId -> list of tweetIds
        self.following = defaultdict(set)  # userId -> set of followed userIds
        self.time = 0  # Timestamp counter for ordering tweets

    def postTweet(self, userId: int, tweetId: int) -> None:
        # Append the tweet with userId and the current timestamp
        self.tweets[userId].append((self.time, tweetId))
        self.time += 1  # Increment the time for the next tweet

    def getNewsFeed(self, userId: int) -> list:
        # Create a min-heap to collect tweets from user and followed users
        min_heap = []

        # Get tweets from the user themselves
        for timestamp, tweetId in self.tweets[userId]:
            min_heap.append((timestamp, tweetId))

        # Get tweets from users the user is following
        for followeeId in self.following[userId]:
            for timestamp, tweetId in self.tweets[followeeId]:
                min_heap.append((timestamp, tweetId))
        
        # Use a heap to get the 10 most recent tweets
        most_recent_tweets = heapq.nlargest(10, min_heap)
        
        # Return only the tweet IDs from the tuple
        return [tweetId for timestamp, tweetId in most_recent_tweets]

    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:  # A user can't follow themselves
            self.following[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:  # A user can't unfollow themselves
            self.following[followerId].discard(followeeId)

```

### Explanation

- **Initialization**: We start by initializing two dictionaries using `defaultdict`. The `tweets` dictionary holds lists of tweets for each user, and the `following` dictionary holds sets indicating which users each user is following.
  
- **Posting a Tweet**: When a tweet is posted, we store it in the `tweets` dictionary associated with the user’s ID, alongside a timestamp. This allows us to later order tweets by their time of posting.

- **Generating News Feed**: In `getNewsFeed`, we compile tweets from both the user and users they are following into a list (`min_heap`). We then use `heapq.nlargest` to retrieve the ten most recent tweets efficiently.

- **Following/Unfollowing**: The `follow` and `unfollow` methods simply modify the `following` dictionary to update relationships.

This solution is efficient and adheres to the specifications provided, making it suitable for a basic version of a Twitter-like application. The use of dictionaries and heaps allows for quick retrieval and management of data while maintaining accuracy in tweet ordering by timestamp.

# 359. Logger Rate Limiter

### Problem Description 
Design a logger system that receives a stream of messages along with their timestamps. Each unique message should only be printed at most every 10 seconds (i.e. a message printed at timestamp `t` will prevent other identical messages from being printed until timestamp `t + 10`).

All messages will come in chronological order. Several messages may arrive at the same timestamp.

Implement the `Logger` class:
`Logger()` Initializes the `logger` object.

`bool shouldPrintMessage(int timestamp, string message)` Returns `true` if the `message` should be printed in the given `timestamp`, otherwise returns `false`.


Example 1:
Input
["Logger", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage"]
[[], [1, "foo"], [2, "bar"], [3, "foo"], [8, "bar"], [10, "foo"], [11, "foo"]]
Output
[null, true, true, false, false, false, true]
Explanation
Logger logger = new Logger();
logger.shouldPrintMessage(1, "foo");  // return true, next allowed timestamp for "foo" is 1 + 10 = 11
logger.shouldPrintMessage(2, "bar");  // return true, next allowed timestamp for "bar" is 2 + 10 = 12
logger.shouldPrintMessage(3, "foo");  // 3 < 11, return false
logger.shouldPrintMessage(8, "bar");  // 8 < 12, return false
logger.shouldPrintMessage(10, "foo"); // 10 < 11, return false
logger.shouldPrintMessage(11, "foo"); // 11 >= 11, return true, next allowed timestamp for "foo" is
                                      // 11 + 10 = 21

Constraints:
`0 <= timestamp <= 109`
Every `timestamp` will be passed in non-decreasing order (chronological order).

`1 <= message.length <= 30`
At most `104` calls will be made to `shouldPrintMessage`.

### Solution 
 To solve the problem of designing a logger system that controls the frequency of message printing based on timestamps, we can utilize a dictionary to keep track of the last time each unique message was printed. The key is the message itself, and the value is the last timestamp when that message was printed. This allows us to quickly check whether a message can be printed again based on the current timestamp.

Here's how we approach the problem:

1. Create a class `Logger` that initializes an empty dictionary to store messages and their last printed timestamps.

2. Implement the `shouldPrintMessage` method that takes in a timestamp and a message.
    - If the message does not exist in the dictionary, or if the current timestamp is greater than or equal to the last printed timestamp plus 10 seconds, we print the message (return `True`) and update its last printed timestamp.
    - If the conditions are not met, we return `False` to indicate the message should not be printed.

3. Since messages will arrive in chronological order, we don't need to sort or rearrange anything; we only need to track the last printed time.

Here's the complete implementation of the `Logger` class following the LeetCode solution format:



```python
class Logger:

    def __init__(self):
        # Initialize a dictionary to store the last printed timestamp for each message
        self.msg_timestamp = {}

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        # Check if the message is in the dictionary and if enough time has passed
        if message not in self.msg_timestamp:
            # Message has never been printed, we can print it now
            self.msg_timestamp[message] = timestamp
            return True
        else:
            # Check the last printed time for the message
            last_timestamp = self.msg_timestamp[message]
            if timestamp >= last_timestamp + 10:
                # Enough time has passed, print the message and update the timestamp
                self.msg_timestamp[message] = timestamp
                return True
            else:
                # Not enough time has passed to print the message again
                return False

```

### Explanation of the Code:
- The `__init__` method sets up an empty dictionary called `msg_timestamp` which will map messages (strings) to their last printed timestamp (integers).
- In the `shouldPrintMessage` method:
  - We first check if the message is not in `msg_timestamp`. If it's a new message, we record its current timestamp and return `True`.
  - If the message already exists in the dictionary, we retrieve its last printed timestamp. 
  - We then check if the current timestamp is at least 10 seconds greater than the last printed timestamp. If so, we allow it to be printed (return `True`) and update the timestamp; otherwise, we return `False`.

### Example Application:
When following the example provided:
- For `logger.shouldPrintMessage(1, "foo")`, since "foo" is new, it returns `True` and records the timestamp.
- For `logger.shouldPrintMessage(2, "bar")`, it behaves similarly.
- For subsequent calls, the appropriate conditions are checked and logged.

This system effectively controls the output as per the requirements, ensuring each message adheres to the 10-second rule constraint.

# 362. Design Hit Counter

### Problem Description 
Design a hit counter which counts the number of hits received in the past `5` minutes (i.e., the past `300` seconds).

Your system should accept a `timestamp` parameter (in seconds granularity), and you may assume that calls are being made to the system in chronological order (i.e., `timestamp` is monotonically increasing). Several hits may arrive roughly at the same time.

Implement the `HitCounter` class:
`HitCounter()` Initializes the object of the hit counter system.

`void hit(int timestamp)` Records a hit that happened at `timestamp` (in seconds). Several hits may happen at the same `timestamp`.

`int getHits(int timestamp)` Returns the number of hits in the past 5 minutes from `timestamp` (i.e., the past `300` seconds).


Example 1:
Input
["HitCounter", "hit", "hit", "hit", "getHits", "hit", "getHits", "getHits"]
[[], [1], [2], [3], [4], [300], [300], [301]]
Output
[null, null, null, null, 3, null, 4, 3]
Explanation
HitCounter hitCounter = new HitCounter();
hitCounter.hit(1);       // hit at timestamp 1.

hitCounter.hit(2);       // hit at timestamp 2.

hitCounter.hit(3);       // hit at timestamp 3.

hitCounter.getHits(4);   // get hits at timestamp 4, return 3.

hitCounter.hit(300);     // hit at timestamp 300.

hitCounter.getHits(300); // get hits at timestamp 300, return 4.

hitCounter.getHits(301); // get hits at timestamp 301, return 3.


Constraints:
`1 <= timestamp <= 2 * 109`
All the calls are being made to the system in chronological order (i.e., `timestamp` is monotonically increasing).

At most `300` calls will be made to `hit` and `getHits`.

Follow up: What if the number of hits per second could be huge? Does your design scale?

### Solution 
 To solve the problem of designing a hit counter that records the number of hits in the past 5 minutes (or 300 seconds), we can create a `HitCounter` class to manage these records. The `HitCounter` class will keep track of timestamps for each hit and provide functionality to count the number of hits that occurred in the specified time frame.

Here's how we can implement this in Python:

### Implementation Details

1. **Data Structure**: We will use a list to store timestamps of hits. Since timestamps are monotonically increasing, we can simply append each new timestamp to this list whenever a hit is recorded.

2. **Method Definitions**:
   - The `hit` method will take a timestamp and append it to the list.
   - The `getHits` method will take a timestamp as an argument and count how many timestamps in the list are within the last 300 seconds relative to that timestamp.

3. **Optimization Considerations**: While the naive solution of iterating through all hits to count the relevant hits is acceptable given the constraints (at most 300 calls), we can employ a two-pointer technique for better efficiency if needed in larger datasets.

### Code Implementation



```python
class HitCounter:
    def __init__(self):
        self.hits = []  # List to store timestamps

    def hit(self, timestamp: int) -> None:
        """Records a hit at the given timestamp."""
        self.hits.append(timestamp)

    def getHits(self, timestamp: int) -> int:
        """Returns the number of hits in the past 5 minutes (300 seconds) from timestamp."""
        # We want hits within [timestamp - 300, timestamp]
        window_start = timestamp - 300
        count = 0
        
        # Iterate over the recorded hits and count the valid ones
        for hit_time in self.hits:
            if window_start < hit_time <= timestamp:
                count += 1
            # As the list is sorted, we can stop early if we go past the timestamp
            elif hit_time > timestamp:
                break
        
        return count

# Example Usage:
hitCounter = HitCounter()
hitCounter.hit(1)       # hit at timestamp 1
hitCounter.hit(2)       # hit at timestamp 2
hitCounter.hit(3)       # hit at timestamp 3
print(hitCounter.getHits(4))   # return 3
hitCounter.hit(300)     # hit at timestamp 300
print(hitCounter.getHits(300)) # return 4
print(hitCounter.getHits(301)) # return 3

```

### Explanation of the Code:

- **Initialization**: The `__init__` method constructs an empty list `hits` to store the timestamps of the hits.
- **Recording Hits**: The `hit` function appends the provided timestamp to the `hits` list to record the time of the hit.
- **Counting Hits**: In the `getHits` function, we compute the valid time window (`window_start`) using the given timestamp. We then iterate through the `hits` list and increment our count for each hit that falls within this valid time window. If we encounter a timestamp greater than the given timestamp (since the hits are sorted), we can exit the loop early to avoid unnecessary checks.

This algorithm is straightforward and efficient given the problem constraints, while also being easy to understand and maintain. The two-pointer optimization can be implemented if necessary for even larger datasets in a follow-up adjustment.

# 379. Design Phone Directory

### Problem Description 
Design a phone directory that initially has `maxNumbers` empty slots that can store numbers. The directory should store numbers, check if a certain slot is empty or not, and empty a given slot.

Implement the `PhoneDirectory` class:
`PhoneDirectory(int maxNumbers)` Initializes the phone directory with the number of available slots `maxNumbers`.

`int get()` Provides a number that is not assigned to anyone. Returns `-1` if no number is available.

`bool check(int number)` Returns `true` if the slot `number` is available and `false` otherwise.

`void release(int number)` Recycles or releases the slot `number`.


Example 1:
Input
["PhoneDirectory", "get", "get", "check", "get", "check", "release", "check"]
[[3], [], [], [2], [], [2], [2], [2]]
Output
[null, 0, 1, true, 2, false, null, true]
Explanation
PhoneDirectory phoneDirectory = new PhoneDirectory(3);
phoneDirectory.get();      // It can return any available phone number. Here we assume it returns 0.

phoneDirectory.get();      // Assume it returns 1.

phoneDirectory.check(2);   // The number 2 is available, so return true.

phoneDirectory.get();      // It returns 2, the only number that is left.

phoneDirectory.check(2);   // The number 2 is no longer available, so return false.

phoneDirectory.release(2); // Release number 2 back to the pool.

phoneDirectory.check(2);   // Number 2 is available again, return true.


Constraints:
`1 <= maxNumbers <= 104`
`0 <= number < maxNumbers`
At most `2 * 104` calls will be made to `get`, `check`, and `release`.

### Solution 
 To solve the problem of designing a phone directory as described, we can utilize a combination of data structures to efficiently manage the available and assigned phone numbers. Here's a detailed breakdown of how we can implement the `PhoneDirectory` class:

### Class Overview
1. **Initialization**: The `PhoneDirectory` class will have an array `is_assigned` to keep track of which slots are occupied. We will also maintain a set `available` to keep track of the available numbers for efficient retrieval.
  
2. **Method Implementations**:
   - `get()`: This method should return one of the available numbers. If no number is available, it will return `-1`. We'll achieve this by checking the set of available numbers, retrieving a number, and then removing it from the available set.
   - `check(number)`: This method checks if a specific number is available (i.e., not assigned) and returns `True` if it is and `False` otherwise.
   - `release(number)`: This method will recycle or release a previously assigned number back to the available pool, allowing it to be reused.

### Python Code
Here's the complete implementation:



```python
class PhoneDirectory:

    def __init__(self, maxNumbers: int):
        self.maxNumbers = maxNumbers
        self.is_assigned = [False] * maxNumbers  # Tracks assigned numbers
        self.available = set(range(maxNumbers))  # A set of available numbers

    def get(self) -> int:
        if not self.available:
            return -1  # No numbers available
        num = self.available.pop()  # Get and remove an available number
        self.is_assigned[num] = True  # Mark the number as assigned
        return num  # Return the assigned number

    def check(self, number: int) -> bool:
        # Check if the number is available
        return not self.is_assigned[number]

    def release(self, number: int) -> None:
        if self.is_assigned[number]:  # Only release if it was assigned
            self.is_assigned[number] = False  # Mark as not assigned
            self.available.add(number)  # Add back to available numbers

```

### Explanation of the Code
1. **Initialization (`__init__`)**:
   - We set `maxNumbers` to the maximum limit allowed.
   - We create a boolean list `is_assigned` of size `maxNumbers` initialized to `False`, indicating that no numbers are assigned at the start.
   - We also create a set `available` that contains all numbers from `0` to `maxNumbers - 1`. This allows O(1) complexity for checking and retrieving available numbers.

2. **Getting a Number (`get`)**:
   - We check if the `available` set is empty. If yes, we return `-1`.
   - Otherwise, we use `pop()` on the set to retrieve an available number and mark it as assigned using the `is_assigned` list.

3. **Checking Availability (`check`)**:
   - This method simply checks if the number in the `is_assigned` list is `False`, indicating that the number is available.

4. **Releasing a Number (`release`)**:
   - We check if the number was actually assigned by looking at the `is_assigned` list. If it was, we mark it as not assigned and add it back to the `available` set.

### Complexity Analysis
- **Time Complexity**:
  - All three methods (`get`, `check`, `release`) run in O(1) time because they perform constant time operations.
  
- **Space Complexity**:
  - The space complexity is O(N) where N is `maxNumbers`, as we store the state of each number in `is_assigned` and maintain a set for available numbers.

### Example Usage:
After implementing the class as shown, you can directly run it on LeetCode or in any Python environment. Here's how you might use the class based on the example given:



```python
phoneDirectory = PhoneDirectory(3)
print(phoneDirectory.get())      # Could return 0
print(phoneDirectory.get())      # Could return 1
print(phoneDirectory.check(2))   # True
print(phoneDirectory.get())      # Could return 2
print(phoneDirectory.check(2))   # False
phoneDirectory.release(2)        # Releasing number 2
print(phoneDirectory.check(2))   # True

```

This implementation efficiently handles the requirements of the problem while respecting the constraints provided.

# 380. Insert Delete GetRandom O(1)

### Problem Description 
Implement the `RandomizedSet` class:
`RandomizedSet()` Initializes the `RandomizedSet` object.

`bool insert(int val)` Inserts an item `val` into the set if not present. Returns `true` if the item was not present, `false` otherwise.

`bool remove(int val)` Removes an item `val` from the set if present. Returns `true` if the item was present, `false` otherwise.

`int getRandom()` Returns a random element from the current set of elements (it's guaranteed that at least one element exists when this method is called). Each element must have the same probability of being returned.


Example 1:
Input
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
Output
[null, true, false, true, 2, true, false, 2]
Explanation
RandomizedSet randomizedSet = new RandomizedSet();
randomizedSet.insert(1); // Inserts 1 to the set. Returns true as 1 was inserted successfully.

randomizedSet.remove(2); // Returns false as 2 does not exist in the set.

randomizedSet.insert(2); // Inserts 2 to the set, returns true. Set now contains [1,2].

randomizedSet.getRandom(); // getRandom() should return either 1 or 2 randomly.

randomizedSet.remove(1); // Removes 1 from the set, returns true. Set now contains [2].

randomizedSet.insert(2); // 2 was already in the set, so return false.

randomizedSet.getRandom(); // Since 2 is the only number in the set, getRandom() will always return 2.


Constraints:
`-231 <= val <= 231 - 1`
At most `105` calls will be made to `insert`, `remove`, and `getRandom`.

There will be at least one element in the data structure when `getRandom` is called.

Follow up: Could you implement the functions of the class with each function works in average `O(1)` time?

### Solution 
 To solve the problem of implementing the `RandomizedSet` class, we need to ensure that we can efficiently manage a collection of unique integers with the following capabilities:

1. **Insert** an integer into the set if it is not already present (average O(1) time).
2. **Remove** an integer from the set if it is present (average O(1) time).
3. **Get a random element** from the set (average O(1) time).

To achieve this, we can use a combination of:
- A list to store the elements, which allows for efficient random access.
- A dictionary (hash map) to keep track of the indices of the elements in the list for quick insert and remove operations.

### Implementation Details
- **Insert Operation**: When we want to insert a value, we check if it already exists using the dictionary. If not, we add it to the list and store its index in the dictionary.
- **Remove Operation**: To remove a value, we check its existence. If it exists, we find its index in the list using the dictionary. To maintain the list's structure, we swap the element to be removed with the last element in the list, pop it from the list, and update the dictionary accordingly.
- **Get Random Operation**: This is straightforward since with a list we can utilize `random.choice()` to pick a random element.

The average time complexity for all operations will be O(1) due to efficient dictionary lookups and list operations.

### Code Implementation
Here is how the `RandomizedSet` class can be implemented following the LeetCode format:



```python
import random

class RandomizedSet:
    def __init__(self):
        """
        Initializes the RandomizedSet object.
        """
        self.values = []  # List to store the values
        self.index_map = {}  # Dictionary to map values to their indices in `values`

    def insert(self, val: int) -> bool:
        """
        Inserts an item val into the set if not present.
        Returns true if the item was not present, false otherwise.
        """
        if val in self.index_map:
            return False  # Return false if the value already exists
        self.index_map[val] = len(self.values)  # Store the index of the new value
        self.values.append(val)  # Append the new value to the list
        return True  # Return true indicating successful insertion

    def remove(self, val: int) -> bool:
        """
        Removes an item val from the set if present.
        Returns true if the item was present, false otherwise.
        """
        if val not in self.index_map:
            return False  # Return false if the value is not present

        # Get the index of the value to remove
        index_to_remove = self.index_map[val]
        last_element = self.values[-1]

        # Move the last element to the index of the element to remove
        self.values[index_to_remove] = last_element
        self.index_map[last_element] = index_to_remove

        # Remove the last element and delete from the index_map
        self.values.pop()
        del self.index_map[val]

        return True  # Return true indicating successful removal

    def getRandom(self) -> int:
        """
        Returns a random element from the current set of elements.
        Each element must have the same probability of being returned.
        """
        return random.choice(self.values)  # Return a random element from the list

```

### Example Usage


```python
# Example usage:
randomizedSet = RandomizedSet()
print(randomizedSet.insert(1))  # Returns True
print(randomizedSet.remove(2))   # Returns False
print(randomizedSet.insert(2))    # Returns True
print(randomizedSet.getRandom())  # Returns 1 or 2 randomly
print(randomizedSet.remove(1))    # Returns True
print(randomizedSet.insert(2))    # Returns False
print(randomizedSet.getRandom())   # Returns 2

```

### Explanation of the Code:
1. The `__init__` method initializes the `RandomizedSet` with an empty list and dictionary.
2. The `insert` method checks if the value already exists. If it does not, it appends the value to the list and updates the dictionary.
3. The `remove` method updates the list and dictionary by replacing the deleted element with the last element, so we don't need to shift elements.
4. The `getRandom` method uses Python's random library to return a random item from the list.

With this implementation, all required operations are efficiently performed in average O(1) time, satisfying the problem's constraints.

# 381. Insert Delete GetRandom O(1) - Duplicates allowed

### Problem Description 
Implement the `RandomizedCollection` class:
`RandomizedCollection()` Initializes the `RandomizedCollection` object.

`bool insert(int val)` Inserts an item `val` into the multiset if not present. Returns `true` if the item was not present, `false` otherwise.

`bool remove(int val)` Removes an item `val` from the multiset if present. Returns `true` if the item was present, `false` otherwise. Note that if `val` has multiple occurrences in the multiset, we only remove one of them.

`int getRandom()` Returns a random element from the current multiset of elements (it's guaranteed that at least one element exists when this method is called). The probability of each element being returned is linearly related to the number of same values the multiset contains.


Example 1:
Input
["RandomizedCollection", "insert", "insert", "insert", "getRandom", "remove", "getRandom"]
[[], [1], [1], [2], [], [1], []]
Output
[null, true, false, true, 2, true, 1]
Explanation
RandomizedCollection randomizedCollection = new RandomizedCollection();
randomizedCollection.insert(1);   // return True. Inserts 1 to the collection. Returns true as the collection did not contain 1.

randomizedCollection.insert(1);   // return False. Inserts another 1 to the collection. Returns false as the collection contained 1. Collection now contains [1,1].

randomizedCollection.insert(2);   // return True. Inserts 2 to the collection, returns true. Collection now contains [1,1,2].

randomizedCollection.getRandom(); // getRandom should return 1 with the probability 2/3, and returns 2 with the probability 1/3.

randomizedCollection.remove(1);   // return True. Removes 1 from the collection, returns true. Collection now contains [1,2].

randomizedCollection.getRandom(); // getRandom should return 1 and 2 both equally likely.


Constraints:
`-231 <= val <= 231 - 1`
At most `105` calls will be made to `insert`, `remove`, and `getRandom`.

There will be at least one element in the data structure when `getRandom` is called.

Follow up: Could you implement the functions of the class with each function works in average `O(1)` time?

### Solution 
 To implement the `RandomizedCollection` class as described, we need to handle three operations: insertion, removal, and retrieving a random element, all while ensuring that we maintain an average time complexity of O(1) for each operation. We can achieve this using a combination of a dictionary and a list.

Here's a step-by-step explanation of the solution:

1. **Data Structures**:
   - Use a list (`self.nums`) to store the actual values of the multiset. This allows for O(1) access when picking a random element and O(1) time complexity for inserting elements.
   - Use a dictionary (`self.indices`) where the key is the value and the value is a set of indices in `self.nums` where that value occurs. This allows us to easily find and remove a specific occurrence of a value.

2. **Insert Operation**:
   - Check if the value already exists using the dictionary. If it does not, append the value to the list and update the dictionary, returning `True`. If it exists, still append to the list but return `False`.

3. **Remove Operation**:
   - Check if the value exists in the dictionary. If it does, remove one occurrence by:
     - Finding the index of the value to remove (we can pop one from the set of indices).
     - Replace that index in the list with the last element to maintain contiguous storage.
     - Update the indices in the dictionary accordingly.
     - Return `True` if a value was removed, otherwise `False`.

4. **GetRandom Operation**:
   - Simply use `random.choice` to select and return a random element from the list of values, ensuring that the time complexity remains O(1).

Here's the complete implementation, which is compatible with LeetCode:



```python
import random
from collections import defaultdict

class RandomizedCollection:
    
    def __init__(self):
        self.nums = []  # List to store the numbers
        self.indices = defaultdict(set)  # Dictionary to store indices of elements
    
    def insert(self, val: int) -> bool:
        exists = val in self.indices
        self.nums.append(val)  # Add to the list
        self.indices[val].add(len(self.nums) - 1)  # Store the index in the dictionary
        return not exists  # Return True if it was not previously in the collection
    
    def remove(self, val: int) -> bool:
        if val not in self.indices or not self.indices[val]:
            return False  # Return False if the value does not exist
        
        # Remove an occurrence of val
        # Get any index of val
        index_to_remove = self.indices[val].pop()  # Get an arbitrary index of the value
        
        # Move the last element to the place of the element to remove if it's not the last element
        last_element = self.nums[-1]
        self.nums[index_to_remove] = last_element
        
        # Update the indices dictionary
        if self.indices[last_element]:  # if the last element exists in the indices
            self.indices[last_element].remove(len(self.nums) - 1)  # remove the last index
            self.indices[last_element].add(index_to_remove)  # add the new index
        
        self.nums.pop()  # Remove the last element 
        if not self.indices[val]:  # If the value has no more indices, remove it from the dictionary
            del self.indices[val]
        
        return True
    
    def getRandom(self) -> int:
        return random.choice(self.nums)  # Return a random element from the list

```

### Explanation of the Code:
- We initialize our data structures in the `__init__` method.
- The `insert` method checks if the value is already present, adds it, updates the indices, and returns the appropriate boolean.
- The `remove` method carefully removes an instance of the value, ensuring that we maintain the integrity of our list and our mapping of indices.
- The `getRandom` method uses Python's random library to return a random selection from our list of stored values.

This implementation guarantees that all three methods operate within O(1) time complexity on average, satisfying the problem's constraints and requirements.

# 432. All O`one Data Structure

### Problem Description 
Implement a data structure supporting the following operations:
Inc(Key) - Inserts a new key  with value 1. Or increments an existing key by 1. Key is guaranteed to be a non-empty string.

Dec(Key) - If Key's value is 1, remove it from the data structure. Otherwise decrements an existing key by 1. If the key does not exist, this function does nothing. Key is guaranteed to be a non-empty string.

GetMaxKey() - Returns one of the keys with maximal value. If no element exists, return an empty string `""`.

GetMinKey() - Returns one of the keys with minimal value. If no element exists, return an empty string `""`.

Challenge: Perform all these in O(1) time complexity.

### Solution 
 To solve the problem of implementing a data structure that supports `Inc`, `Dec`, `GetMaxKey`, and `GetMinKey` operations all in O(1) time complexity, we can leverage a combination of a hash map (dictionary) and a doubly linked list.

### Explanation of the Approach:

1. **Data Structures**:
   - **Hash Map**: We'll use a dictionary to store the current value of each key. This allows O(1) access to update and check values.
   - **Doubly Linked List (DLL)**: This will help maintain the keys in order based on their counts. Each node of the DLL represents a unique value (count), and it contains a list of keys that have that count.

2. **Key Operations**:
   - **Inc(Key)**: If the key exists, we increment its value in the hash map. We move the key from its current count node to the next count node in the DLL. If the key does not exist, we add it with an initial value of 1 and place it in the appropriate node in the DLL.
   - **Dec(Key)**: If the key exists and its value is greater than 1, we decrement it and move it to the previous count in the DLL. If its value is 1, we remove it from the hash map and delete the corresponding node from the DLL.
   - **GetMaxKey()**: We keep track of the maximum count node, and we return one of the keys from that node.
   - **GetMinKey()**: Similar to `GetMaxKey()`, we keep track of the minimum count node and return one of the keys from that node.

3. **Maintaining the DLL**: 
   - We need to maintain and link the nodes correctly when incrementing or decrementing keys. If a node for a specific count does not exist, we will create it.

### Implementation:

Here is a Python implementation of the above approach:



```python
class Node:
    def __init__(self, count):
        self.count = count
        self.keys = set()  # Store keys with this count
        self.prev = None
        self.next = None

class AllOne:
    def __init__(self):
        self.key_count = {}  # Store count for each key
        self.count_node = {}  # Map count to the corresponding node in the DLL
        self.head = Node(float('-inf'))  # Dummy head node
        self.tail = Node(float('inf'))  # Dummy tail node
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_node_after(self, new_node, prev_node):
        new_node.next = prev_node.next
        new_node.prev = prev_node
        prev_node.next.prev = new_node
        prev_node.next = new_node

    def _remove_node(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        del self.count_node[node.count]  # Clean up the reference

    def Inc(self, key: str) -> None:
        # Update count in key_count
        count = self.key_count.get(key, 0)
        new_count = count + 1
        self.key_count[key] = new_count

        # Update count node
        current_node = self.count_node.get(count)
        if current_node:
            current_node.keys.remove(key)
            if not current_node.keys:  # current count is empty, remove the node
                self._remove_node(current_node)

        # Insert/Update new count node
        if new_count not in self.count_node:
            new_node = Node(new_count)
            self.count_node[new_count] = new_node
            # Insert the new node after the previous count node
            prev_node = self.head
            while prev_node.next.count < new_count:
                prev_node = prev_node.next
            self._add_node_after(new_node, prev_node)

        # Add key to new count node
        self.count_node[new_count].keys.add(key)

    def Dec(self, key: str) -> None:
        if key not in self.key_count:
            return
        
        count = self.key_count[key]
        new_count = count - 1
        
        current_node = self.count_node[count]
        current_node.keys.remove(key)
        if not current_node.keys:  # current count is empty, remove the node
            self._remove_node(current_node)

        if new_count == 0:
            del self.key_count[key]  # Remove key from key_count
        else:
            self.key_count[key] = new_count
            if new_count not in self.count_node:
                new_node = Node(new_count)
                self.count_node[new_count] = new_node
                # Insert the new node before the current count node
                prev_node = current_node.prev
                while prev_node and prev_node.count > new_count:
                    prev_node = prev_node.prev
                self._add_node_after(new_node, prev_node)
            self.count_node[new_count].keys.add(key)

    def GetMaxKey(self) -> str:
        return "" if self.head.next == self.tail else next(iter(self.tail.prev.keys))

    def GetMinKey(self) -> str:
        return "" if self.head.next == self.tail else next(iter(self.head.next.keys))

```

### How to Run on LeetCode:
To run this code on LeetCode, you typically need to adjust the class name and import any necessary libraries depending on the specific problem constraints provided by LeetCode. However, this implementation follows the correct structure to be adjusted into most formats used by LeetCode for problem submissions. You can directly copy and paste this code into the provided class in the LeetCode environment. 

### Complexity:
- **Time Complexity**: O(1) for each operation.
- **Space Complexity**: O(n) for storing keys and managing nodes in the DLL, where n is the number of unique keys.

# 460. LFU Cache

### Problem Description 
Design and implement a data structure for a Least Frequently Used (LFU) cache.

Implement the `LFUCache` class:
`LFUCache(int capacity)` Initializes the object with the `capacity` of the data structure.

`int get(int key)` Gets the value of the `key` if the `key` exists in the cache. Otherwise, returns `-1`.

`void put(int key, int value)` Update the value of the `key` if present, or inserts the `key` if not already present. When the cache reaches its `capacity`, it should invalidate and remove the least frequently used key before inserting a new item. For this problem, when there is a tie (i.e., two or more keys with the same frequency), the least recently used `key` would be invalidated.

To determine the least frequently used key, a use counter is maintained for each key in the cache. The key with the smallest use counter is the least frequently used key.

When a key is first inserted into the cache, its use counter is set to `1` (due to the `put` operation). The use counter for a key in the cache is incremented either a `get` or `put` operation is called on it.


Example 1:
Input
["LFUCache", "put", "put", "get", "put", "get", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [3], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, 3, null, -1, 3, 4]
Explanation
// cnt(x) = the use counter for key x
// cache=[] will show the last used order for tiebreakers (leftmost element is  most recent)
LFUCache lfu = new LFUCache(2);
lfu.put(1, 1);   // cache=[1,_], cnt(1)=1
lfu.put(2, 2);   // cache=[2,1], cnt(2)=1, cnt(1)=1
lfu.get(1);      // return 1
                 // cache=[1,2], cnt(2)=1, cnt(1)=2
lfu.put(3, 3);   // 2 is the LFU key because cnt(2)=1 is the smallest, invalidate 2.

                 // cache=[3,1], cnt(3)=1, cnt(1)=2
lfu.get(2);      // return -1 (not found)
lfu.get(3);      // return 3
                 // cache=[3,1], cnt(3)=2, cnt(1)=2
lfu.put(4, 4);   // Both 1 and 3 have the same cnt, but 1 is LRU, invalidate 1.

                 // cache=[4,3], cnt(4)=1, cnt(3)=2
lfu.get(1);      // return -1 (not found)
lfu.get(3);      // return 3
                 // cache=[3,4], cnt(4)=1, cnt(3)=3
lfu.get(4);      // return 4
                 // cache=[3,4], cnt(4)=2, cnt(3)=3

Constraints:
`0 <= capacity, key, value <= 104`
At most `105` calls will be made to `get` and `put`.

Follow up: Could you do both operations in `O(1)` time complexity?

### Solution 
 To solve the LFU Cache problem with O(1) time complexity for both `get` and `put` operations, we'll need to utilize a combination of data structures to efficiently manage the frequency of usage and retrieval order. Here's how we can design the `LFUCache` class.

### Class Design
1. **HashMap (Dictionary)**: We'll use a dictionary to map each key to its corresponding value and frequency count. This allows O(1) access to any key.
  
2. **Frequency List**: Another dictionary will hold keys grouped by their usage frequency. Each frequency will point to a queue (or list) of keys that have that frequency. This structure will help us handle the eviction of keys efficiently. The LRU keys will be located at the head of the list.

3. **Min Frequency Tracker**: We'll maintain a variable to track the minimum frequency currently in the cache. This is critical for quickly identifying which key to evict when we reach capacity.

4. **Capacity Management**: We must handle the maximum capacity of the cache by tracking when we need to remove the least frequently used keys.

### Implementation
Now, let's implement this design in the `LFUCache` class:



```python
class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.key_count = {}  # stores key to (value, frequency)
        self.freq_list = {}  # stores frequency to list of keys
        self.min_freq = 0  # tracks the minimum frequency
          
    def _update_key(self, key: int):
        value, freq = self.key_count[key]
        # Remove the key from the current frequency list
        self.freq_list[freq].remove(key)
        # If the list becomes empty at this frequency, update min_freq
        if not self.freq_list[freq]:
            del self.freq_list[freq]
            if self.min_freq == freq:
                self.min_freq += 1
            
        # Update the key's frequency
        new_freq = freq + 1
        self.key_count[key] = (value, new_freq)
        
        if new_freq not in self.freq_list:
            self.freq_list[new_freq] = []
        self.freq_list[new_freq].append(key)

    def get(self, key: int) -> int:
        if key not in self.key_count:
            return -1
        
        # Get the current value of the key
        self._update_key(key)  # update frequency
        return self.key_count[key][0]

    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return
        
        if key in self.key_count:
            # Key exists, update the value and frequency
            self.key_count[key] = (value, self.key_count[key][1])
            self._update_key(key)
        else:
            if self.size >= self.capacity:
                # Evict the least frequently used key
                lfu_key = self.freq_list[self.min_freq].pop(0)  # Remove the LRU key
                if not self.freq_list[self.min_freq]:  # If this was the last key
                    del self.freq_list[self.min_freq]
                del self.key_count[lfu_key]
                self.size -= 1

            # Insert the new key
            self.key_count[key] = (value, 1)
            self.min_freq = 1  # reset min_freq to 1
            if 1 not in self.freq_list:
                self.freq_list[1] = []
            self.freq_list[1].append(key)
            self.size += 1

```

### Explanation
- **Initialization**: The `LFUCache` initializes the capacity, current size, key-value-frequency mapping, frequency mapping, and min frequency.
  
- **Update Function**: `_update_key` is a private method that increments a key's frequency and manages its position in the frequency list.

- **Get Method**:
  - If the key exists, it retrieves the value and updates the key's frequency.
  - If the key doesn’t exist, it returns -1.

- **Put Method**:
  - If the key already exists, it updates its value and frequency.
  - If the key doesn't exist and the capacity is reached, it evicts the least frequently used key.
  - It then adds the new key with a frequency of 1.

With this implementation, both `get` and `put` operations run in O(1) time complexity, and the solution meets the requirements of the LFU cache problem.

# 588. Design In-Memory File System

### Problem Description 
Design an in-memory file system to simulate the following functions:
`ls`: Given a path in string format. If it is a file path, return a list that only contains this file's name. If it is a directory path, return the list of file and directory names in this directory. Your output (file and directory names together) should in lexicographic order.

`mkdir`: Given a directory path that does not exist, you should make a new directory according to the path. If the middle directories in the path don't exist either, you should create them as well. This function has void return type.

`addContentToFile`: Given a file path and file content in string format. If the file doesn't exist, you need to create that file containing given content. If the file already exists, you need to append given content to original content. This function has void return type.

`readContentFromFile`: Given a file path, return its content in string format.


Example:
Input: 
["FileSystem","ls","mkdir","addContentToFile","ls","readContentFromFile"]
[[],["/"],["/a/b/c"],["/a/b/c/d","hello"],["/"],["/a/b/c/d"]]
Output:
[null,[],null,null,["a"],"hello"]
Explanation:
Note:
You can assume all file or directory paths are absolute paths which begin with `/` and do not end with `/` except that the path is just `"/"`.

You can assume that all operations will be passed valid parameters and users will not attempt to retrieve file content or list a directory or file that does not exist.

You can assume that all directory names and file names only contain lower-case letters, and same names won't exist in the same directory.

### Solution 
 To solve the problem of designing an in-memory file system that simulates basic file and directory operations, we can utilize a tree-like structure to represent the file system. Each node in this structure can either be a directory containing other directories or files, or a file containing some content.

Let's define a class `FileSystem` which contains the four operations specified, leveraging a nested dictionary to represent directories and file content.

Here’s the detailed code implementation following the LeetCode format:



```python
class FileSystem:
    def __init__(self):
        # Create a root node which is a dictionary that will represent the file system
        self.root = {}
        
    def ls(self, path: str) -> List[str]:
        # Split the path into parts
        parts = path.split('/')
        current = self.root
        
        # Traverse the file system based on the parts of the path
        for part in parts[1:]:
            if part in current:  # Navigate through the current one level down
                current = current[part]

        # If we landed on a file, return just the file name
        if isinstance(current, str):
            return [parts[-1]]
        
        # Otherwise, return a sorted list of directory and file names
        return sorted(current.keys())

    def mkdir(self, path: str) -> None:
        # Split the path into parts
        parts = path.split('/')
        current = self.root
        
        # Create directories as per the path
        for part in parts[1:]:
            if part not in current:
                current[part] = {}  # Create a new directory as a new dictionary
            current = current[part]
    
    def addContentToFile(self, filePath: str, content: str) -> None:
        parts = filePath.split('/')
        current = self.root
        
        # Navigate through the directories until the last part (the file)
        for part in parts[1:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Get the file name and either add content or create a new file
        file_name = parts[-1]
        if file_name not in current:
            current[file_name] = ""  # Create a new file with empty content
        current[file_name] += content  # Append content to the file

    def readContentFromFile(self, filePath: str) -> str:
        parts = filePath.split('/')
        current = self.root
        
        # Navigate to the file
        for part in parts[1:-1]:
            current = current[part]
            
        file_name = parts[-1]
        return current[file_name]  # Return the content of the file

# Example Usage
# obj = FileSystem()
# obj.ls("/")
# obj.mkdir("/a/b/c")
# obj.addContentToFile("/a/b/c/d", "hello")
# print(obj.ls("/"))  # Should output ['a']
# print(obj.readContentFromFile("/a/b/c/d"))  # Should output 'hello'

```

### Explanation:
1. **Initialization**: The `FileSystem` constructor initializes the root of the file system as an empty dictionary. This dictionary will serve as the main structure to hold paths (both directories and files).

2. **`ls(path: str) -> List[str]`**:
   - This method splits the path into parts and navigates through the structure. 
   - If the final part is determined to be a file (indicated by its content being a string), it returns the file name, otherwise, it returns a sorted list of the keys in the current dictionary (which are either files or directories).

3. **`mkdir(path: str) -> None`**:
   - Similar to `ls`, this method traverses the path components and creates directories as it goes. If a directory does not exist at a level, it adds an empty dictionary to represent that directory.

4. **`addContentToFile(filePath: str, content: str) -> None`**:
   - This method also navigates to the directory where the file should reside. If the file does not exist, it initializes it with empty content and then appends the new content.

5. **`readContentFromFile(filePath: str) -> str`**:
   - This method navigates to the file’s location and retrieves its content.

This structure allows efficient creation, retrieval, and storage of directories and files, while also ensuring that operations like listing contents are done in lexicographic order as required.

# 604. Design Compressed String Iterator

### Problem Description 
Design and implement a data structure for a compressed string iterator. The given compressed string will be in the form of each letter followed by a positive integer representing the number of this letter existing in the original uncompressed string.

Implement the StringIterator class:
`next()` Returns the next character if the original string still has uncompressed characters, otherwise returns a white space.

`hasNext()` Returns true if there is any letter needs to be uncompressed in the original string, otherwise returns `false`.


Example 1:
Input
["StringIterator", "next", "next", "next", "next", "next", "next", "hasNext", "next", "hasNext"]
[["L1e2t1C1o1d1e1"], [], [], [], [], [], [], [], [], []]
Output
[null, "L", "e", "e", "t", "C", "o", true, "d", true]
Explanation
StringIterator stringIterator = new StringIterator("L1e2t1C1o1d1e1");
stringIterator.next(); // return "L"
stringIterator.next(); // return "e"
stringIterator.next(); // return "e"
stringIterator.next(); // return "t"
stringIterator.next(); // return "C"
stringIterator.next(); // return "o"
stringIterator.hasNext(); // return True
stringIterator.next(); // return "d"
stringIterator.hasNext(); // return True

Constraints:
`1 <= compressedString.length <= 1000`
`compressedString` consists of lower-case an upper-case English letters and digits.

The number of a single character repetitions in `compressedString` is in the range `[1, 10^9]`
At most `100` calls will be made to `next` and `hasNext`.

### Solution 
 To tackle the problem of designing a `StringIterator` that decompresses a compressed string representation into its original format, we need to focus on two primary operations: `next()` to fetch the next character and `hasNext()` to check if there are more characters left to iterate through.

### Solution Approach

1. **Initialization**: The constructor of the `StringIterator` will take the compressed string as input and parse it. The parsing will involve reading characters and associated counts (which can be multiple digits long). We will store these characters and their counts in a list for easy access.

2. **next() Method**: This method will return the next character from the decompressed string. If the character’s count is exhausted, we will move to the next character in our list. If there are no characters left to iterate through, we will return a space.

3. **hasNext() Method**: This method will simply check if there are any characters left in our iterator; we can do this by keeping track of our current position in the parsed data.

### Detailed Implementation

Here’s how we can implement this in Python, following the specifications provided:



```python
class StringIterator:
    def __init__(self, compressedString: str):
        self.chars = []
        self.counts = []
        index = 0
        n = len(compressedString)

        # Parsing the compressed string
        while index < n:
            char = compressedString[index]
            index += 1
            count_str = ''
            # Extracting the numbers (could be multiple digits)
            while index < n and compressedString[index].isdigit():
                count_str += compressedString[index]
                index += 1
            count = int(count_str)
            self.chars.append(char)
            self.counts.append(count)

        self.current_index = 0  # Pointer to the current character
        self.current_count = 0   # Current count of the character to return next

    def next(self) -> str:
        if self.current_index >= len(self.chars):
            return ' '  # Return whitespace if no more characters are available

        # If we have exhausted the current character's count, move to the next character
        if self.current_count == self.counts[self.current_index]:
            self.current_index += 1
            self.current_count = 0  # Reset count for the new character

        if self.current_index >= len(self.chars):
            return ' '  # No more characters left

        # Return the next character
        result = self.chars[self.current_index]
        self.current_count += 1 
        return result

    def hasNext(self) -> bool:
        return self.current_index < len(self.chars) or self.current_count < self.counts[self.current_index] if self.current_index < len(self.chars) else False

```

### Explanation of the Code:
1. **Constructor `__init__`**: Initializes the iterator by parsing the compressed string:
   - Iterates through each character in the compressed string.
   - For each character, it reads until it finds a digit, accumulates the digits to form a complete number (representing the count), and stores both the character and its count in separate lists.

2. **Method `next()`**:
   - Checks if there are any characters left to output. If not, it returns a space.
   - If the current character's count is exhausted, it moves the index to the next character.
   - It returns the current character and increments the count of that character.

3. **Method `hasNext()`**: 
   - Checks if there are more characters to iterate over or if the current character has more instances to return.

### Complexity Analysis:
- **Space Complexity**: O(N), where N is the number of characters in the compressed string because we store both characters and their counts.
- **Time Complexity** for each operation (`next` and `hasNext`) is O(1) since we are just accessing stored values and simple conditions.

This code accurately represents the required solution and meets the constraints given in the problem statement. You can now copy and paste this code directly into a LeetCode editor for testing.

# 622. Design Circular Queue

### Problem Description 
Design your implementation of the circular queue. The circular queue is a linear data structure in which the operations are performed based on FIFO (First In First Out) principle and the last position is connected back to the first position to make a circle. It is also called "Ring Buffer".

One of the benefits of the circular queue is that we can make use of the spaces in front of the queue. In a normal queue, once the queue becomes full, we cannot insert the next element even if there is a space in front of the queue. But using the circular queue, we can use the space to store new values.

Implementation the `MyCircularQueue` class:
`MyCircularQueue(k)` Initializes the object with the size of the queue to be `k`.

`int Front()` Gets the front item from the queue. If the queue is empty, return `-1`.

`int Rear()` Gets the last item from the queue. If the queue is empty, return `-1`.

`boolean enQueue(int value)` Inserts an element into the circular queue. Return `true` if the operation is successful.

`boolean deQueue()` Deletes an element from the circular queue. Return `true` if the operation is successful.

`boolean isEmpty()` Checks whether the circular queue is empty or not.

`boolean isFull()` Checks whether the circular queue is full or not.


Example 1:
Input
["MyCircularQueue", "enQueue", "enQueue", "enQueue", "enQueue", "Rear", "isFull", "deQueue", "enQueue", "Rear"]
[[3], [1], [2], [3], [4], [], [], [], [4], []]
Output
[null, true, true, true, false, 3, true, true, true, 4]
Explanation
MyCircularQueue myCircularQueue = new MyCircularQueue(3);
myCircularQueue.enQueue(1); // return True
myCircularQueue.enQueue(2); // return True
myCircularQueue.enQueue(3); // return True
myCircularQueue.enQueue(4); // return False
myCircularQueue.Rear();     // return 3
myCircularQueue.isFull();   // return True
myCircularQueue.deQueue();  // return True
myCircularQueue.enQueue(4); // return True
myCircularQueue.Rear();     // return 4

Constraints:
`1 <= k <= 1000`
`0 <= value <= 1000`
At most `3000` calls will be made to `enQueue`, `deQueue`, `Front`, `Rear`, `isEmpty`, and `isFull`.

Follow up: Could you solve the problem without using the built-in queue?

### Solution 
 To implement a circular queue in Python, we'll create a class called `MyCircularQueue` that provides the specified functionality. A circular queue allows us to efficiently use the available space by wrapping around once we reach the end of the array that implements it.

### Key Components of MyCircularQueue

1. **Attributes**:
    - `k`: The maximum size of the queue.
    - `front`: Index of the front element in the queue.
    - `rear`: Index of the last element in the queue.
    - `size`: Current number of elements in the queue.
    - `queue`: List to store the queue elements.

2. **Methods**:
    - `__init__(self, k)`: Initializes the circular queue with size `k`.
    - `Front(self)`: Returns the front element of the queue.
    - `Rear(self)`: Returns the last element of the queue.
    - `enQueue(self, value)`: Adds an element to the end of the queue.
    - `deQueue(self)`: Removes an element from the front of the queue.
    - `isEmpty(self)`: Checks if the queue is empty.
    - `isFull(self)`: Checks if the queue is full.

### Python Code

Here is the implementation in the LeetCode-compatible format:



```python
class MyCircularQueue:

    def __init__(self, k: int):
        self.k = k  # size of the queue
        self.front = -1  # front index
        self.rear = -1  # rear index
        self.size = 0  # current number of elements
        self.queue = [0] * k  # list to store elements

    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.front]

    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.rear]

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        
        if self.isEmpty():
            self.front = 0  # Set front to the first index as it's the first enqueue
        
        self.rear = (self.rear + 1) % self.k  # Circular increment of rear
        self.queue[self.rear] = value  # Insert the element
        self.size += 1  # Increment the size
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        
        if self.front == self.rear:  # Queue will be empty after this operation
            self.front = -1
            self.rear = -1
        else:
            self.front = (self.front + 1) % self.k  # Circular increment of front
        
        self.size -= 1  # Decrement the size
        return True

    def isEmpty(self) -> bool:
        return self.size == 0

    def isFull(self) -> bool:
        return self.size == self.k

```

### Detailed Explanation

1. **Initialization**:
   - The constructor initializes the queue with a fixed size `k` and sets up a list `queue` of size `k` to hold the elements. The front and rear indices are initialized to `-1` to indicate that the queue is empty, and the current size is also initialized to `0`.

2. **Adding Elements (`enQueue`)**:
   - Before adding, we check if the queue is full using `isFull()`. If it is full, we return `False`.
   - If the queue is empty, we set `front` to `0`.
   - We then calculate the new rear position using modulo operation to wrap around if it reaches the end of the array. Next, we place the new value at the `rear` position and increment the size.

3. **Removing Elements (`deQueue`)**:
   - We first check if the queue is empty using `isEmpty()`. If it is, we return `False`.
   - If `front` equals `rear`, it indicates that we will remove the last element, so we need to reset both `front` and `rear` to `-1`.
   - Otherwise, we move the `front` position using modulo to ensure it wraps around correctly and decrement the size.

4. **Accessing Elements (`Front` & `Rear`)**:
   - Both operations simply check if the queue is empty and return the respective values from the `queue` list.

5. **Utility Methods (`isEmpty` & `isFull`)**:
   - These methods simply check if the size is `0` or if it equals `k`, respectively.

### Complexity Analysis
- The time complexity for all operations (`enQueue`, `deQueue`, `Front`, `Rear`, `isEmpty`, and `isFull`) is O(1) because we are only performing simple arithmetic and list index operations.
- Space complexity is O(k) where `k` is the size of the circular queue, as we store up to `k` elements in the list.

This implementation fully meets the requirements of the problem statement and efficiently uses the space in the circular queue.

# 631. Design Excel Sum Formula

### Problem Description 
Your task is to design the basic function of Excel and implement the function of sum formula.  Specifically, you need to implement the following functions:
`Excel(int H, char W):` This is the constructor. The inputs represents the height and width of the Excel form. H is a positive integer, range from 1 to 26. It represents the height. W is a character range from 'A' to 'Z'. It represents that the width is the number of characters from 'A' to W. The Excel form content is represented by a height * width 2D integer array `C`, it should be initialized to zero. You should assume that the first row of `C` starts from 1, and the first column of `C` starts from 'A'.

`void Set(int row, char column, int val):` Change the value at `C(row, column)` to be val.

`int Get(int row, char column):` Return the value at `C(row, column)`.

`int Sum(int row, char column, List of Strings : numbers):` This function calculate and set the value at `C(row, column)`, where the value should be the sum of cells represented by `numbers`. This function return the sum result at `C(row, column)`. This sum formula should exist until this cell is overlapped by another value or another sum formula.

`numbers` is a list of strings that each string represent a cell or a range of cells. If the string represent a single cell, then it has the following format : `ColRow`. For example, "F7" represents the cell at (7, F). 
If the string represent a range of cells, then it has the following format : `ColRow1:ColRow2`. The range will always be a rectangle, and ColRow1 represent the position of the top-left cell, and ColRow2 represents the position of the bottom-right cell. 

Example 1:
Excel(3,"C"); 
// construct a 3*3 2D array with all zero.

//   A B C
// 1 0 0 0
// 2 0 0 0
// 3 0 0 0
Set(1, "A", 2);
// set C(1,"A") to be 2.

//   A B C
// 1 2 0 0
// 2 0 0 0
// 3 0 0 0
Sum(3, "C", ["A1", "A1:B2"]);
// set C(3,"C") to be the sum of value at C(1,"A") and the values sum of the rectangle range whose top-left cell is C(1,"A") and bottom-right cell is C(2,"B"). Return 4. 
//   A B C
// 1 2 0 0
// 2 0 0 0
// 3 0 0 4
Set(2, "B", 2);
// set C(2,"B") to be 2. Note C(3, "C") should also be changed.

//   A B C
// 1 2 0 0
// 2 0 2 0
// 3 0 0 6
Note:
You could assume that there won't be any circular sum reference. For example, A1 = sum(B1) and B1 = sum(A1).

 The test cases are using double-quotes to represent a character.

Please remember to RESET your class variables declared in class Excel, as static/class variables are persisted across multiple test cases. Please see here for more details.

### Solution 
 Here's a complete solution to the problem described above, designed to be compatible with LeetCode's submission format. We will implement the `Excel` class, which requires constructors and methods for setting values, getting values, and calculating sums from a specified cell or range of cells.

We'll start by outlining the design of each method in the class:

1. **Constructor (`__init__`)**:
   - The constructor initializes a 2D list (array) representing the Excel table with all zeros based on the specified height (H) and width (W).

2. **Set Method**:
   - This method sets the value at a specified cell (given by its row and column) to a new value.

3. **Get Method**:
   - This method retrieves the value from a specified cell.

4. **Sum Method**:
   - This method computes the sum of values from specified cells or ranges. If a cell is represented as a range (e.g., "A1:B2"), it will calculate the sum of all cells within that rectangle.

Here’s the complete implementation in Python:



```python
class Excel:
    def __init__(self, H: int, W: str):
        # Initialize the height and width
        self.rows = H
        self.cols = ord(W) - ord('A') + 1
        self.C = [[0] * self.cols for _ in range(self.rows)]
    
    def _get_index(self, row: int, column: str) -> tuple:
        # Convert (row, column) from 1-indexed to 0-indexed for internal representation
        return (row - 1, ord(column) - ord('A'))

    def Set(self, row: int, column: str, val: int) -> None:
        r, c = self._get_index(row, column)
        self.C[r][c] = val

    def Get(self, row: int, column: str) -> int:
        r, c = self._get_index(row, column)
        return self.C[r][c]

    def Sum(self, row: int, column: str, numbers: list) -> int:
        total = 0
        
        for number in numbers:
            if ':' in number:  # if it is a range
                start, end = number.split(':')
                start_row, start_col = self._get_index(int(start[1:]), start[0])
                end_row, end_col = self._get_index(int(end[1:]), end[0])
                
                # Sum the values in the rectangular range
                for r in range(start_row, end_row + 1):
                    for c in range(start_col, end_col + 1):
                        total += self.C[r][c]
            else:  # if it is a single cell
                r, c = self._get_index(int(number[1:]), number[0])
                total += self.C[r][c]
        
        # Set the summed value back into the target cell
        r, c = self._get_index(row, column)
        self.C[r][c] = total
        
        return total

```

### Detailed Explanation:

1. **Initialization**:
   - The constructor accepts height `H` and width `W`. We map `W` from a character representation ('A' to 'Z') to a width in terms of numeric indices for our internal 2D list.
   - We create a 2D list `C` and initialize all cells to zero.

2. **Index Calculation**:
   - A helper function `_get_index` converts the given row and column information from a 1-indexed format to a 0-indexed format, which is necessary for internal representation in the list.

3. **Set Method**:
   - This takes a row, column, and value. It calculates the corresponding indices in the 2D list and sets the value at that position.

4. **Get Method**:
   - This fetches the value from a specified row and column by calling `_get_index`.

5. **Sum Method**:
   - In this method, we check if a number is a range (contains ':') or a single cell. 
   - For ranges, we parse the starting and ending cell IDs and calculate the indices. Then, we iterate over the rectangular area and sum the values.
   - After calculating the total, we store this value back into the specified cell and return the sum.

### Usage:
When using this class, you can create an instance of `Excel` and invoke the methods as demonstrated in the problem statement to manipulate and retrieve data from the "Excel" structure. This design handles all operations efficiently, making it suitable for the given problem constraints.

# 635. Design Log Storage System

### Problem Description 
You are given several logs, where each log contains a unique ID and timestamp. Timestamp is a string that has the following format: `Year:Month:Day:Hour:Minute:Second`, for example, `2017:01:01:23:59:59`. All domains are zero-padded decimal numbers.

Implement the `LogSystem` class:
`LogSystem()` Initializes the `LogSystem` object.

`void put(int id, string timestamp)` Stores the given log `(id, timestamp)` in your storage system.

`int[] retrieve(string start, string end, string granularity)` Returns the IDs of the logs whose timestamps are within the range from `start` to `end` inclusive. `start` and `end` all have the same format as `timestamp`, and `granularity` means how precise the range should be (i.e. to the exact `Day`, `Minute`, etc.). For example, `start = "2017:01:01:23:59:59"`, `end = "2017:01:02:23:59:59"`, and `granularity = "Day"` means that we need to find the logs within the inclusive range from Jan. 1st 2017 to Jan. 2nd 2017, and the `Hour`, `Minute`, and `Second` for each log entry can be ignored.


Example 1:
Input
["LogSystem", "put", "put", "put", "retrieve", "retrieve"]
[[], [1, "2017:01:01:23:59:59"], [2, "2017:01:01:22:59:59"], [3, "2016:01:01:00:00:00"], ["2016:01:01:01:01:01", "2017:01:01:23:00:00", "Year"], ["2016:01:01:01:01:01", "2017:01:01:23:00:00", "Hour"]]
Output
[null, null, null, null, [3, 2, 1], [2, 1]]
Explanation
LogSystem logSystem = new LogSystem();
logSystem.put(1, "2017:01:01:23:59:59");
logSystem.put(2, "2017:01:01:22:59:59");
logSystem.put(3, "2016:01:01:00:00:00");
// return [3,2,1], because you need to return all logs between 2016 and 2017.

logSystem.retrieve("2016:01:01:01:01:01", "2017:01:01:23:00:00", "Year");
// return [2,1], because you need to return all logs between Jan. 1, 2016 01:XX:XX and Jan. 1, 2017 23:XX:XX.

// Log 3 is not returned because Jan. 1, 2016 00:00:00 comes before the start of the range.

logSystem.retrieve("2016:01:01:01:01:01", "2017:01:01:23:00:00", "Hour");

Constraints:
`1 <= id <= 500`
`2000 <= Year <= 2017`
`1 <= Month <= 12`
`1 <= Day <= 31`
`0 <= Hour <= 23`
`0 <= Minute, Second <= 59`
`granularity` is one of the values `["Year", "Month", "Day", "Hour", "Minute", "Second"]`.

At most `500` calls will be made to `put` and `retrieve`.

### Solution 
 To implement the `LogSystem` class that manages logging and retrieval of timestamps based on a specified granularity, we can follow these steps:

1. **Store Logs**: We'll use a list or dictionary to store the logs, where each log entry is a tuple of `(id, timestamp)`.

2. **Granularity Adjustment**: Since we are required to adjust the granularity for retrieving logs, we must define what range of timestamps to consider based on the granularity level given (Year, Month, Day, Hour, Minute, Second).

3. **Retrieval Logic**: When a retrieval request occurs, we need to filter logs based on the specified start and end timestamps while adjusting these timestamps according to the granularity.

Here’s the complete implementation in Python that adheres to the format necessary for LeetCode submissions:



```python
class LogSystem:
    def __init__(self):
        self.logs = []  # To store log entries as (id, timestamp)

    def put(self, id: int, timestamp: str) -> None:
        """Store the log (id, timestamp)."""
        self.logs.append((id, timestamp))

    def retrieve(self, start: str, end: str, granularity: str) -> list[int]:
        """Retrieve logs within the specified start and end range with granularity."""
        # Maps granularity to the appropriate character index for the timestamp string
        granularity_map = {
            "Year": 4,
            "Month": 7,
            "Day": 10,
            "Hour": 13,
            "Minute": 16,
            "Second": 19
        }
        
        # Get the index based on granularity
        granularity_index = granularity_map[granularity]

        # Adjust start and end according to granularity
        start = start[:granularity_index] + "0" * (19 - granularity_index)
        end = end[:granularity_index] + "9" * (19 - granularity_index)

        result = []

        # Check each log entry
        for id, timestamp in self.logs:
            if start <= timestamp <= end:
                result.append(id)

        return result

# Example usage:
# logSystem = LogSystem()
# logSystem.put(1, "2017:01:01:23:59:59")
# logSystem.put(2, "2017:01:01:22:59:59")
# logSystem.put(3, "2016:01:01:00:00:00")
# print(logSystem.retrieve("2016:01:01:01:01:01", "2017:01:01:23:00:00", "Year"))  # [2, 1]
# print(logSystem.retrieve("2016:01:01:01:01:01", "2017:01:01:23:00:00", "Hour"))  # [2, 1]

```

### Explanation of the Code:

1. **Initialization**: The `__init__` method initializes the `LogSystem` object with an empty list for logs.

2. **Adding Logs**: The `put` method appends a new log entry (ID and timestamp) to the `logs` list.

3. **Retrieving Logs**:
    - The `retrieve` method takes `start`, `end`, and `granularity` as inputs.
    - A `granularity_map` is defined to map the granularity to character indices in the timestamp string. This helps us know up to which character we need to consider.
    - Adjust the `start` and `end` timestamps according to the granularity. For example, if the granularity is "Day", we replace the characters after the day with zeros for the start and nines for the end. This way, we're effectively normalizing the start and end bounds.
    - We then iterate over each log entry and check if the timestamp falls within the adjusted range. If it does, we add the ID to our result list.
  
4. **Return Results**: The method returns a list of IDs that fall within the specified timestamp range.

This implementation ensures efficient log entry storage and retrieval while respecting the constraints set forth by the problem.

# 641. Design Circular Deque

### Problem Description 
Design your implementation of the circular double-ended queue (deque).

Your implementation should support following operations:
`MyCircularDeque(k)`: Constructor, set the size of the deque to be k.

`insertFront()`: Adds an item at the front of Deque. Return true if the operation is successful.

`insertLast()`: Adds an item at the rear of Deque. Return true if the operation is successful.

`deleteFront()`: Deletes an item from the front of Deque. Return true if the operation is successful.

`deleteLast()`: Deletes an item from the rear of Deque. Return true if the operation is successful.

`getFront()`: Gets the front item from the Deque. If the deque is empty, return -1.

`getRear()`: Gets the last item from Deque. If the deque is empty, return -1.

`isEmpty()`: Checks whether Deque is empty or not. 
`isFull()`: Checks whether Deque is full or not.


Example:
MyCircularDeque circularDeque = new MycircularDeque(3); // set the size to be 3
circularDeque.insertLast(1);			// return true
circularDeque.insertLast(2);			// return true
circularDeque.insertFront(3);			// return true
circularDeque.insertFront(4);			// return false, the queue is full
circularDeque.getRear();  			// return 2
circularDeque.isFull();				// return true
circularDeque.deleteLast();			// return true
circularDeque.insertFront(4);			// return true
circularDeque.getFront();			// return 4
Note:
All values will be in the range of [0, 1000].

The number of operations will be in the range of [1, 1000].

Please do not use the built-in Deque library.

### Solution 
 To solve the problem of implementing a circular double-ended queue (deque) in Python, we will create a class called `MyCircularDeque`. This class will encapsulate the methods described in the problem statement. The key components of a circular deque include maintaining the size, handling the head and tail positions, and allowing for additions and deletions from both ends.

Here's how we can approach it:

### Key Design Considerations:
1. **Circular Buffer**: We will use a list to represent the circular buffer. We will manage the head and tail pointers that allow us to add or remove elements from the front and the back efficiently.

2. **Capacity Management**: Keep track of the maximum capacity of the deque, the current number of elements, and check for full/empty states.

3. **Indexing**: We need to ensure that the indices wrap around when they reach the bounds of the internal array. This can be achieved with modulo operations.

### Implementation Steps:
1. **Initialization**: Create an initializer that sets up the maximum size, tracks the current number of elements, and initializes head and tail pointers.

2. **Insertion and Deletion Methods**: Implement the methods for inserting and deleting items from both ends, while managing the wrapping of indices.

3. **Getter Methods**: Implement methods to fetch items from the front or back of the deque, ensuring we handle empty conditions.

4. **Check Methods**: Implement methods to check if the deque is empty or full.

### Code Implementation:

Here is how the complete implementation looks:



```python
class MyCircularDeque:
    
    def __init__(self, k: int):
        """Initialize the deque with a maximum size of k."""
        self.capacity = k
        self.size = 0
        self.front = 0
        self.rear = 0
        self.data = [0] * k  # Using an array of size k to hold the elements

    def insertFront(self, value: int) -> bool:
        """Adds an item at the front of Deque. Return true if the operation is successful."""
        if self.isFull():
            return False
            
        self.front = (self.front - 1) % self.capacity  # Move front pointer
        self.data[self.front] = value  # Insert the value at front
        self.size += 1
        return True

    def insertLast(self, value: int) -> bool:
        """Adds an item at the rear of Deque. Return true if the operation is successful."""
        if self.isFull():
            return False
            
        self.data[self.rear] = value  # Insert the value at rear
        self.rear = (self.rear + 1) % self.capacity  # Move rear pointer
        self.size += 1
        return True

    def deleteFront(self) -> bool:
        """Deletes an item from the front of Deque. Return true if the operation is successful."""
        if self.isEmpty():
            return False
            
        self.front = (self.front + 1) % self.capacity  # Move front pointer forward
        self.size -= 1
        return True

    def deleteLast(self) -> bool:
        """Deletes an item from the rear of Deque. Return true if the operation is successful."""
        if self.isEmpty():
            return False
            
        self.rear = (self.rear - 1) % self.capacity  # Move rear pointer backward
        self.size -= 1
        return True

    def getFront(self) -> int:
        """Gets the front item from the Deque. If the deque is empty, return -1."""
        if self.isEmpty():
            return -1
            
        return self.data[self.front]

    def getRear(self) -> int:
        """Gets the last item from Deque. If the deque is empty, return -1."""
        if self.isEmpty():
            return -1
            
        return self.data[(self.rear - 1) % self.capacity]  # Rear is one position ahead of the last element

    def isEmpty(self) -> bool:
        """Checks whether Deque is empty or not."""
        return self.size == 0

    def isFull(self) -> bool:
        """Checks whether Deque is full or not."""
        return self.size == self.capacity

```

### Explanation of the Code:
1. **Initialization**:
   - The constructor initializes the `capacity`, `size`, and the `front` and `rear` pointers. It also creates an internal list to store the elements of the deque.

2. **Insertion Methods**:
   - `insertFront`: Moves the `front` index backward and inserts the value if there is space; otherwise, returns `False`.
   - `insertLast`: Inserts the value at the `rear` index and then moves it forward.

3. **Deletion Methods**:
   - `deleteFront`: Simply moves the `front` pointer forward if the deque is not empty.
   - `deleteLast`: Moves the `rear` pointer backward if the deque is not empty.

4. **Getter Methods**:
   - `getFront`: Returns the value at the `front` index if the deque is not empty.
   - `getRear`: Returns the value at the index immediately before `rear`.

5. **Check Methods**:
   - `isEmpty` and `isFull` check the current conditions of the deque based on `size`.

This implementation ensures efficient operations in O(1) time complexity for all operations. This format is compatible with LeetCode’s submission format and should work directly when pasted into their code editor.

# 642. Design Search Autocomplete System

### Problem Description 
Design a search autocomplete system for a search engine. Users may input a sentence (at least one word and end with a special character `'#'`). For each character they type except '#', you need to return the top 3 historical hot sentences that have prefix the same as the part of sentence already typed. Here are the specific rules:
The hot degree for a sentence is defined as the number of times a user typed the exactly same sentence before.

The returned top 3 hot sentences should be sorted by hot degree (The first is the hottest one). If several sentences have the same degree of hot, you need to use ASCII-code order (smaller one appears first).

If less than 3 hot sentences exist, then just return as many as you can.

When the input is a special character, it means the sentence ends, and in this case, you need to return an empty list.

Your job is to implement the following functions:
The constructor function:
`AutocompleteSystem(String[] sentences, int[] times):` This is the constructor. The input is historical data. `Sentences` is a string array consists of previously typed sentences. `Times` is the corresponding times a sentence has been typed. Your system should record these historical data.

Now, the user wants to input a new sentence. The following function will provide the next character the user types:
`List<String> input(char c):` The input `c` is the next character typed by the user. The character will only be lower-case letters (`'a'` to `'z'`), blank space (`' '`) or a special character (`'#'`). Also, the previously typed sentence should be recorded in your system. The output will be the top 3 historical hot sentences that have prefix the same as the part of sentence already typed.


Example:
Operation: AutocompleteSystem(["i love you", "island","ironman", "i love leetcode"], [5,3,2,2])
The system have already tracked down the following sentences and their corresponding times:
`"i love you"` : `5` times
`"island"` : `3` times
`"ironman"` : `2` times
`"i love leetcode"` : `2` times
Now, the user begins another search:
Operation: input('i')
Output: ["i love you", "island","i love leetcode"]
Explanation:
There are four sentences that have prefix `"i"`. Among them, "ironman" and "i love leetcode" have same hot degree. Since `' '` has ASCII code 32 and `'r'` has ASCII code 114, "i love leetcode" should be in front of "ironman". Also we only need to output top 3 hot sentences, so "ironman" will be ignored.

Operation: input(' ')
Output: ["i love you","i love leetcode"]
Explanation:
There are only two sentences that have prefix `"i "`.

Operation: input('a')
Output: []
Explanation:
There are no sentences that have prefix `"i a"`.

Operation: input('#')
Output: []
Explanation:
The user finished the input, the sentence `"i a"` should be saved as a historical sentence in system. And the following input will be counted as a new search.

Note:
The input sentence will always start with a letter and end with '#', and only one blank space will exist between two words.

The number of complete sentences that to be searched won't exceed 100. The length of each sentence including those in the historical data won't exceed 100.

Please use double-quote instead of single-quote when you write test cases even for a character input.

Please remember to RESET your class variables declared in class AutocompleteSystem, as static/class variables are persisted across multiple test cases. Please see here for more details.

### Solution 
 Here's a detailed explanation of how to implement the autocomplete system requested in the problem statement. We'll create a class called `AutocompleteSystem` that maintains a collection of previously typed sentences and their counts, and provides suggestions based on user input.

### Steps and Design Choices:

1. **Data Structures**:
   - We'll use a dictionary (`sentence_count`) to map each sentence to its corresponding count of how many times it has been typed. This allows for quick access and modification.
   - For managing the suggestions, we'll use a list to store the sentences and their counts and sort them as needed for output.

2. **Constructor**:
   - The constructor will accept an array of sentences and their corresponding frequencies. We'll populate the dictionary with this data.

3. **Input Method**:
   - This method will handle receiving each character input. Depending on the input, it will:
     - If it's a normal character (a-z or space), it updates the current input prefix and searches for sentences that match this prefix.
     - If it's the end-of-input character `#`, it will add the current input (excluding `#`) to the dictionary, increasing its count, and reset the current input.

4. **Searching Suggestions**:
   - When searching for suggestions, we will filter the dictionary based on the current input prefix, sort by hot degree (count) and ASCII value (sentence itself) to get the top three suggestions.

5. **Output Requirements**:
   - The output will be a list of sentences matching the criteria or an empty list when needed.

Here is the complete code based on the above logic wrapped in LeetCode-friendly format:



```python
class AutocompleteSystem:

    def __init__(self, sentences: list[str], times: list[int]):
        self.sentence_count = {}
        self.current_input = ""
        
        # Populate the sentence count dictionary
        for i in range(len(sentences)):
            self.sentence_count[sentences[i]] = times[i]

    def input(self, c: str) -> list[str]:
        if c == '#':
            # Add the current input to the sentence count
            if self.current_input in self.sentence_count:
                self.sentence_count[self.current_input] += 1
            else:
                self.sentence_count[self.current_input] = 1
            self.current_input = ""  # Reset the current input
            return []

        # Update the current input prefix
        self.current_input += c
        
        # Get suggestions
        suggestions = []
        for sentence in self.sentence_count.keys():
            if sentence.startswith(self.current_input):
                suggestions.append((sentence, self.sentence_count[sentence]))
        
        # Sort based on -hot degree, then sentence
        suggestions.sort(key=lambda x: (-x[1], x[0]))

        # Return the top 3 suggestions
        return [sentence for sentence, count in suggestions[:3]]

# Example usage:
# autocomplete = AutocompleteSystem(["i love you", "island", "ironman", "i love leetcode"], [5, 3, 2, 2])
# print(autocomplete.input('i'))  # Output: ["i love you", "island", "i love leetcode"]
# print(autocomplete.input(' '))   # Output: ["i love you", "i love leetcode"]
# print(autocomplete.input('a'))   # Output: []
# print(autocomplete.input('#'))    # Output: []

```

### How to Run the Code:
- You can copy and paste this code into a Python-enabled environment provided by LeetCode. You might need to replace the test usage example with specific test cases from LeetCode's input format.
- This approach ensures that the class maintains the correct state across multiple calls and offers optimal performance for searching and sorting the sentences. The design allows for easy expansion or modification in case of future requirements.

# 703. Kth Largest Element in a Stream

### Problem Description 
Design a class to find the `kth` largest element in a stream. Note that it is the `kth` largest element in the sorted order, not the `kth` distinct element.

Implement `KthLargest` class:
`KthLargest(int k, int[] nums)` Initializes the object with the integer `k` and the stream of integers `nums`.

`int add(int val)` Returns the element representing the `kth` largest element in the stream.


Example 1:
Input
["KthLargest", "add", "add", "add", "add", "add"]
[[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]
Output
[null, 4, 5, 5, 8, 8]
Explanation
KthLargest kthLargest = new KthLargest(3, [4, 5, 8, 2]);
kthLargest.add(3);   // return 4
kthLargest.add(5);   // return 5
kthLargest.add(10);  // return 5
kthLargest.add(9);   // return 8
kthLargest.add(4);   // return 8

Constraints:
`1 <= k <= 104`
`0 <= nums.length <= 104`
`-104 <= nums[i] <= 104`
`-104 <= val <= 104`
At most `104` calls will be made to `add`.

It is guaranteed that there will be at least `k` elements in the array when you search for the `kth` element.

### Solution 
 To solve the problem of finding the `kth` largest element in a stream of integers, we can leverage a min-heap data structure. The min-heap will allow us to efficiently keep track of the largest `k` elements in the given numbers and the elements added to the stream.

### Explanation:

1. **Data Structure**:
   - A min-heap will be utilized to store the largest `k` elements. The smallest element in this heap will be the `kth` largest element in our stream of numbers.

2. **Initialization**:
   - In the constructor `KthLargest(int k, int[] nums)`, we initialize the heap with the elements in `nums`. If the number of elements exceeds `k`, we remove the smallest element from the heap to maintain its size at `k`.

3. **Adding Elements**:
   - For the `add(int val)` method, we push the new value into the heap. If after adding the value the size of the heap exceeds `k`, we remove the smallest element (the root of the min-heap). The root will then represent the `kth` largest element.

4. **Complexity**:
   - Each insertion into the heap (via `heappush`) takes \(O(\log k)\), and since we might call `add` up to \(10^4\) times, the overall time complexity would be \(O(n \log k)\), where \(n\) is the number of calls to `add`.

Let's implement this in Python following the LeetCode solution format:



```python
import heapq

class KthLargest:
    def __init__(self, k: int, nums: list[int]):
        self.k = k
        self.min_heap = []
        
        # Add initial numbers to the min-heap
        for num in nums:
            self.add(num)

    def add(self, val: int) -> int:
        # Add the new value to the min-heap
        heapq.heappush(self.min_heap, val)
        
        # If the size of the heap exceeds k, remove the smallest element
        if len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)
        
        # The root of the min-heap is the kth largest element
        return self.min_heap[0]

```

### How to Use the Class:

You can create an instance of the `KthLargest` class and use the `add` method as demonstrated below:



```python
kthLargest = KthLargest(3, [4, 5, 8, 2])
print(kthLargest.add(3))   # return 4
print(kthLargest.add(5))   # return 5
print(kthLargest.add(10))  # return 5
print(kthLargest.add(9))   # return 8
print(kthLargest.add(4))   # return 8

```

In this code:
- We use `heapq.heappush` to insert elements into the min-heap.
- We use `heapq.heappop` to maintain the size of the heap and remove the smallest element if we exceed `k`.
- The smallest element in the heap (`self.min_heap[0]`) is always the `kth` largest element in the stream after each addition.

This implementation meets the problem requirements efficiently and is straightforward to understand.

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
 To solve the problem of designing a HashSet without using any built-in hash table libraries, we can create our own implementation of the `MyHashSet` class. The main idea is to use an array (or list) to store the set's elements and to use a hashing mechanism to map the integer keys to their respective indices in that array. 

### Steps to implement `MyHashSet`:

1. **Choosing an Array Size**: 
   - Since the problem states that keys are in the range `[0, 10^6]`, we can create an array that is sufficiently large to minimize collisions. We'll choose an array of size `1009` (a prime number) to help spread out the values more evenly.

2. **Hash Function**:
   - We can use a simple hash function that takes a key and returns an index by using the modulus operation on the length of the array (i.e., `key % size`).

3. **Collision Handling**:
   - We can use a list to handle collisions at each index of the array. If multiple keys hash to the same index, we can simply add them to a list at that index. This allows us to manage collisions efficiently.

4. **Implementing the Methods**:
   - `add(key)`: Compute the index using the hash function and append the key to the list at that index if it is not already present.
   - `contains(key)`: Compute the index and check if the key exists in the list at that index.
   - `remove(key)`: Compute the index and remove the key from the list if it exists.

Here is the Python code implementing the above logic:



```python
class MyHashSet:
    def __init__(self):
        # Initialize a list of empty lists (buckets) for the hash set
        self.size = 1009  # A prime number for better distribution
        self.buckets = [[] for _ in range(self.size)]

    def _hash(self, key):
        # Hash function to get the index for the key
        return key % self.size

    def add(self, key: int) -> None:
        index = self._hash(key)
        # Only add the key if it is not already present
        if key not in self.buckets[index]:
            self.buckets[index].append(key)

    def contains(self, key: int) -> bool:
        index = self._hash(key)
        return key in self.buckets[index]

    def remove(self, key: int) -> None:
        index = self._hash(key)
        # Remove the key if it exists
        if key in self.buckets[index]:
            self.buckets[index].remove(key)

```

### Explanation of the Code:
- **Constructor**: The `__init__` method initializes the hash set. It creates a list called `buckets`, where each element is an empty list to handle the collisions.
  
- **Hash Function**: The `_hash` method computes the index for a key. This ensures that the key is mapped to a specific location in the `buckets`.
  
- **Add Method**: The `add` method checks if a key is already present in the bucket corresponding to its hashed index before adding it to avoid duplicates.

- **Contains Method**: The `contains` method checks if the key exists in its corresponding bucket.

- **Remove Method**: The `remove` method removes the key from its bucket if it exists.

### Usage
This `MyHashSet` class can now be instantiated, and its methods can be called as demonstrated in the problem's example. This implementation provides an efficient way to manage a set of integers within the specified constraints. Each operation (`add`, `remove`, `contains`) operates in average time complexity of O(1).

You can run this code directly in the LeetCode online coding environment.

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
 In order to implement a simple version of a HashMap without using built-in hash table libraries in Python, we can utilize a basic array (or list) structure, combined with the concept of chaining to resolve potential collisions. Here's how we can do this step by step.

### Explanation:

1. **Data Structure Choice**:
   - We use an array (list) to store entries. Each entry is a list of pairs, where each pair will consist of a key and its corresponding value. This approach helps to manage collisions via chaining; when multiple keys hash to the same index, we store them in a list at that index.

2. **Hash Function**:
   - A simple hash function can be the modulus of the key by the size of the array. This keeps indices bounded within the array size.

3. **Operations**:
   - `put(key, value)`: Compute the index using the hash function. If the key already exists in the list at that index, we update its value; if not, we append the new key-value pair.
   - `get(key)`: Compute the index, then search the list at that index to find the key and return its value, or return -1 if the key isn't found.
   - `remove(key)`: Compute the index, search for the key in the list, and if found, remove it.

4. **Size Management**:
   - We can start with a fixed size for the array, and a resize mechanism can be added later if necessary for optimization.

Here is the complete Python code following the LeetCode solution format:



```python
class MyHashMap:
    def __init__(self):
        """Initialize the object with an empty map."""
        self.size = 1000  # We start with a constant size (can be adjusted)
        self.map = [[] for _ in range(self.size)]  # Array of empty lists for chaining

    def _hash(self, key):
        """Private method to hash the key using the modulo operator."""
        return key % self.size

    def put(self, key: int, value: int) -> None:
        """Inserts a (key, value) pair into the HashMap."""
        index = self._hash(key)
        # Check if the key exists and update the value if it does
        for i, (k, v) in enumerate(self.map[index]):
            if k == key:
                self.map[index][i] = (key, value)  # Update the value
                return
        # If key does not exist, add a new (key, value) pair
        self.map[index].append((key, value))

    def get(self, key: int) -> int:
        """Returns the value to which the specified key is mapped, or -1 if not found."""
        index = self._hash(key)
        # Search for the key in the list at the computed index
        for k, v in self.map[index]:
            if k == key:
                return v  # Key found, return the value
        return -1  # Key not found

    def remove(self, key: int) -> None:
        """Removes the key and its corresponding value if the key exists."""
        index = self._hash(key)
        # Search for the key and remove it if found
        for i, (k, v) in enumerate(self.map[index]):
            if k == key:
                del self.map[index][i]  # Remove the key-value pair
                return

# Example usage:
# myHashMap = MyHashMap()
# myHashMap.put(1, 1)    # The map is now [[1,1]]
# myHashMap.put(2, 2)    # The map is now [[1,1], [2,2]]
# print(myHashMap.get(1))  # returns 1
# print(myHashMap.get(3))  # returns -1 (not found)
# myHashMap.put(2, 1)    # The map is now [[1,1], [2,1]] (update)
# print(myHashMap.get(2))  # returns 1
# myHashMap.remove(2)    # remove the mapping for 2
# print(myHashMap.get(2))  # returns -1 (not found)

```

### Complexity:
- **Time Complexity**:
  - The average time complexity for `put`, `get`, and `remove` is O(1) but can degrade to O(n) in the worst-case scenario when many keys hash to the same index.
  
- **Space Complexity**:
  - The space complexity is O(m), where m is the number of buckets (1000 in our case), which holds the chain of key-value pairs.

This implementation is efficient for relatively small hash tables and showcases how data can be stored and retrieved efficiently without relying on built-in hash table functionalities.

# 707. Design Linked List

### Problem Description 
Design your implementation of the linked list. You can choose to use a singly or doubly linked list.

A node in a singly linked list should have two attributes: `val` and `next`. `val` is the value of the current node, and `next` is a pointer/reference to the next node.

If you want to use the doubly linked list, you will need one more attribute `prev` to indicate the previous node in the linked list. Assume all nodes in the linked list are 0-indexed.

Implement the `MyLinkedList` class:
`MyLinkedList()` Initializes the `MyLinkedList` object.

`int get(int index)` Get the value of the `indexth` node in the linked list. If the index is invalid, return `-1`.

`void addAtHead(int val)` Add a node of value `val` before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.

`void addAtTail(int val)` Append a node of value `val` as the last element of the linked list.

`void addAtIndex(int index, int val)` Add a node of value `val` before the `indexth` node in the linked list. If `index` equals the length of the linked list, the node will be appended to the end of the linked list. If `index` is greater than the length, the node will not be inserted.

`void deleteAtIndex(int index)` Delete the `indexth` node in the linked list, if the index is valid.


Example 1:
Input
["MyLinkedList", "addAtHead", "addAtTail", "addAtIndex", "get", "deleteAtIndex", "get"]
[[], [1], [3], [1, 2], [1], [1], [1]]
Output
[null, null, null, null, 2, null, 3]
Explanation
MyLinkedList myLinkedList = new MyLinkedList();
myLinkedList.addAtHead(1);
myLinkedList.addAtTail(3);
myLinkedList.addAtIndex(1, 2);    // linked list becomes 1->2->3
myLinkedList.get(1);              // return 2
myLinkedList.deleteAtIndex(1);    // now the linked list is 1->3
myLinkedList.get(1);              // return 3

Constraints:
`0 <= index, val <= 1000`
Please do not use the built-in LinkedList library.

At most `2000` calls will be made to `get`, `addAtHead`, `addAtTail`, `addAtIndex` and `deleteAtIndex`.

### Solution 
 Here’s a detailed implementation of a singly linked list in Python, following the LeetCode solution format, which you can run directly on the LeetCode website.

### Problem Breakdown
1. **Node Structure:** We will define a `ListNode` class to represent each node in the linked list. Each node has two attributes: `val` (the value) and `next` (a reference to the next node).
   
2. **Linked List Structure:** We will define the `MyLinkedList` class which will maintain the head of the linked list and provide various methods:
   - `get(index)`: Returns the value of the node at the specified index.
   - `addAtHead(val)`: Inserts a new node with value `val` at the start of the list.
   - `addAtTail(val)`: Appends a new node with value `val` at the end of the list.
   - `addAtIndex(index, val)`: Inserts a new node with value `val` at the specified index. If the index is invalid, the operation will not perform any action.
   - `deleteAtIndex(index)`: Deletes the node at the specified index if it exists.

### Code Implementation



```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class MyLinkedList:
    def __init__(self):
        self.head = None
        self.size = 0

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        
        current = self.head
        for _ in range(index):
            current = current.next
        return current.val

    def addAtHead(self, val: int) -> None:
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
        self.size += 1

    def addAtTail(self, val: int) -> None:
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:
            return
        
        if index == 0:
            self.addAtHead(val)
        else:
            new_node = ListNode(val)
            current = self.head
            for _ in range(index - 1):
                current = current.next
            new_node.next = current.next
            current.next = new_node
            self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return
        
        if index == 0:
            self.head = self.head.next
        else:
            current = self.head
            for _ in range(index - 1):
                current = current.next
            current.next = current.next.next if current.next else None
        self.size -= 1

```

### Explanation
- **ListNode Class:** Contains a constructor that initializes the `val` and `next` attributes.
- **MyLinkedList Class:**
  - The constructor initializes the linked list with a `head` set to `None` and the `size` counter to `0`.
  - `get(index)`: Checks if the index is valid, traverses to the index, and returns the node’s value.
  - `addAtHead(val)`: Creates a new node and sets it as the head.
  - `addAtTail(val)`: Traverses to the end of the list and attaches the new node.
  - `addAtIndex(index, val)`: If the index is valid, it checks if it is `0` for head addition or iterates to the desired position for insertion.
  - `deleteAtIndex(index)`: Adjusts pointers to remove the node at the specified index, ensuring bounds are respected.

This code can be directly copied and run in the LeetCode environment for the linked list problem stated.

# 716. Max Stack

### Problem Description 
Design a max stack data structure that supports the stack operations and supports finding the stack's maximum element.

Implement the `MaxStack` class:
`MaxStack()` Initializes the stack object.

`void push(int x)` Pushes element `x` onto the stack.

`int pop()` Removes the element on top of the stack and returns it.

`int top()` Gets the element on the top of the stack without removing it.

`int peekMax()` Retrieves the maximum element in the stack without removing it.

`int popMax()` Retrieves the maximum element in the stack and removes it. If there is more than one maximum element, only remove the top-most one.


Example 1:
Input
["MaxStack", "push", "push", "push", "top", "popMax", "top", "peekMax", "pop", "top"]
[[], [5], [1], [5], [], [], [], [], [], []]
Output
[null, null, null, null, 5, 5, 1, 5, 1, 5]
Explanation
MaxStack stk = new MaxStack();
stk.push(5);   // [5] the top of the stack and the maximum number is 5.

stk.push(1);   // [5, 1] the top of the stack is 1, but the maximum is 5.

stk.push(5);   // [5, 1, 5] the top of the stack is 5, which is also the maximum, because it is the top most one.

stk.top();     // return 5, [5, 1, 5] the stack did not change.

stk.popMax();  // return 5, [5, 1] the stack is changed now, and the top is different from the max.

stk.top();     // return 1, [5, 1] the stack did not change.

stk.peekMax(); // return 5, [5, 1] the stack did not change.

stk.pop();     // return 1, [5] the top of the stack and the max element is now 5.

stk.top();     // return 5, [5] the stack did not change.


Constraints:
`-107 <= x <= 107`
At most `104` calls will be made to `push`, `pop`, `top`, `peekMax`, and `popMax`.

There will be at least one element in the stack when `pop`, `top`, `peekMax`, or `popMax` is called.

Follow up: Could you come up with a solution that supports `O(1)` for each `top` call and `O(logn)` for each other call?

### Solution 
 To implement the `MaxStack` class that supports the required stack operations as well as retrieving the maximum element efficiently, we can use two stacks: one for storing the actual elements and another for tracking the maximum values.

Here's how we can design this data structure:

1. **Main Stack**: This stack will hold all the elements as they are pushed onto the stack.
2. **Max Stack**: This stack will keep track of the maximum values. Each time we push a new element, we'll compare it with the top of the max stack. If it's larger or equal, we push it onto the max stack. This ensures that the top of the max stack always contains the current maximum value of the main stack.

3. **Operations**:
    - `push(x)`: We push `x` onto the main stack. We also check if `x` is greater than or equal to the top of the max stack (or if the max stack is empty), and if so, we push `x` onto the max stack.
    - `pop()`: We pop from the main stack. If the popped element is equal to the top of the max stack, we pop from the max stack as well.
    - `top()`: We simply return the top element of the main stack.
    - `peekMax()`: We return the top element of the max stack, which represents the maximum.
    - `popMax()`: We retrieve the maximum value by looking at the top of the max stack, then we need to remove this value from the main stack. We iterate through the main stack to find and remove the top-most occurrence of that value.

This implementation ensures that the `top()` operation works in O(1) time, while the `peekMax()` and `popMax()` operations take O(n) time (due to the search), but we're capable of achieving O(log n) for `push` and `pop` with a more complex setup.

Now let's look into the actual code:



```python
class MaxStack:
    def __init__(self):
        self.stack = []  # Main stack to hold elements
        self.max_stack = []  # Stack to keep track of max values

    def push(self, x: int) -> None:
        self.stack.append(x)  # Push x to the main stack
        # Push onto max stack if it's the current max or if max stack is empty
        if not self.max_stack or x >= self.max_stack[-1]:
            self.max_stack.append(x)

    def pop(self) -> int:
        if not self.stack:
            return None  # Stack is empty (not specified in constraints, but safe for robustness)
        top_element = self.stack.pop()  # Pop from main stack
        # If the popped element is the max value, pop it from the max stack too
        if top_element == self.max_stack[-1]:
            self.max_stack.pop()
        return top_element

    def top(self) -> int:
        return self.stack[-1]  # Get the top element of the main stack

    def peekMax(self) -> int:
        return self.max_stack[-1]  # The current maximum is the top of the max stack

    def popMax(self) -> int:
        if not self.max_stack:
            return None  # Safety check
        max_value = self.max_stack.pop()  # Get the max value
        # We need to remove the most recent occurrence of max_value from the main stack
        buffer_stack = []
        
        # Pop elements until we find max_value
        while self.stack:
            current = self.stack.pop()
            if current == max_value:
                break  # Found the maximum, stop popping
            buffer_stack.append(current)  # Store the popped elements

        # Restore elements back to main stack
        while buffer_stack:
            self.stack.append(buffer_stack.pop())
        
        return max_value  # Return the removed maximum value

```

### Explanation of Code:
- **Initialization**: We start with empty lists for the main stack (`self.stack`) and the maximum stack (`self.max_stack`).
- **push**: This function adds an integer to the stack and updates the max stack if necessary.
- **pop**: It removes an element from the stack, checking to maintain the max stack.
- **top**: Returns the top element of the main stack.
- **peekMax**: Returns the maximum element safely and efficiently from the max stack.
- **popMax**: This function retrieves and removes the maximum element, carefully preserving the order of other elements in the main stack using a buffer stack. This operation walks through the elements in the stack to ensure the top-most maximum value is the one removed.

This code effectively implements a `MaxStack` data structure fulfilling all the constraints as outlined in the problem statement.

# 1166. Design File System

### Problem Description 
You are asked to design a file system that allows you to create new paths and associate them with different values.

The format of a path is one or more concatenated strings of the form: `/` followed by one or more lowercase English letters. For example, "`/leetcode"` and "`/leetcode/problems"` are valid paths while an empty string `""` and `"/"` are not.

Implement the `FileSystem` class:
`bool createPath(string path, int value)` Creates a new `path` and associates a `value` to it if possible and returns `true`. Returns `false` if the path already exists or its parent path doesn't exist.

`int get(string path)` Returns the value associated with `path` or returns `-1` if the path doesn't exist.


Example 1:
Input: 
["FileSystem","createPath","get"]
[[],["/a",1],["/a"]]
Output: 
[null,true,1]
Explanation: 
FileSystem fileSystem = new FileSystem();
fileSystem.createPath("/a", 1); // return true
fileSystem.get("/a"); // return 1

Example 2:
Input: 
["FileSystem","createPath","createPath","get","createPath","get"]
[[],["/leet",1],["/leet/code",2],["/leet/code"],["/c/d",1],["/c"]]
Output: 
[null,true,true,2,false,-1]
Explanation: 
FileSystem fileSystem = new FileSystem();
fileSystem.createPath("/leet", 1); // return true
fileSystem.createPath("/leet/code", 2); // return true
fileSystem.get("/leet/code"); // return 2
fileSystem.createPath("/c/d", 1); // return false because the parent path "/c" doesn't exist.

fileSystem.get("/c"); // return -1 because this path doesn't exist.


Constraints:
The number of calls to the two functions is less than or equal to `104` in total.

`2 <= path.length <= 100`
`1 <= value <= 109`

### Solution 
 To solve the problem of designing a file system with `createPath` and `get` functionalities, we can adopt a tree-like structure to represent paths, where each valid path can be taken as a series of nodes in this tree. However, for efficiency, particularly for lookup and storage, we can use a dictionary to store the paths and their associated values.

Here's a breakdown of how we can implement the `FileSystem` class:

### Class Structure

1. **Data Structure**: We will utilize a dictionary called `paths` to store the paths. We will use the path string as the key and its associated value as the value. Additionally, to ensure paths are valid, we will check for the existence of parent paths when creating a new path.

2. **Methods**:
    - `createPath(path: str, value: int) -> bool`: This method will:
        1. Check if the path already exists in the dictionary. If so, return `false`.
        2. Split the provided path into its components (sub-paths).
        3. Check if the parent path exists. If it does not, return `false`.
        4. Store the new path in the dictionary with its associated value.
    - `get(path: str) -> int`: This method will simply check if the path exists in the dictionary. If it does, return the value; otherwise, return `-1`.

### Implementation

Here's the complete implementation in Python:



```python
class FileSystem:
    def __init__(self):
        # Dictionary to hold the paths and their associated values
        self.paths = {}

    def createPath(self, path: str, value: int) -> bool:
        # Check if the path already exists
        if path in self.paths:
            return False
        
        # Split the path to get the parent path
        if path == "/" or "/" not in path:
            return False
            
        parent_path = path.rsplit('/', 1)[0]  # Extract the parent path
        
        # Check if the parent path exists
        if parent_path not in self.paths:
            return False
        
        # Create the new path
        self.paths[path] = value
        return True

    def get(self, path: str) -> int:
        # Return the associated value with the path or -1 if it doesn't exist
        return self.paths.get(path, -1)

# Example cases for testing
# fileSystem = FileSystem()
# print(fileSystem.createPath("/a", 1))  # return True
# print(fileSystem.get("/a"))  # return 1
# print(fileSystem.createPath("/leet", 1))  # return True
# print(fileSystem.createPath("/leet/code", 2))  # return True
# print(fileSystem.get("/leet/code"))  # return 2
# print(fileSystem.createPath("/c/d", 1))  # return False
# print(fileSystem.get("/c"))  # return -1

```

### Explanation

1. **Initialization**: `__init__` method initializes the `paths` dictionary.

2. **Creating a Path**:
    - We ensure that the given `path` is not the root ("/") and that it does not already exist.
    - The parent path is derived using `rsplit`. If the parent path isn't found in the dictionary, we return `false`.
    - If everything is valid, we add the new path with its value to the dictionary.

3. **Getting a Value**: The `get` method uses the `dict.get()` method, which will return `-1` if the path does not exist, optimizing the lookup.

### Complexity

- **Time Complexity**:
    - `createPath`: O(k), where k is the length of the path string since we may need to traverse through it to find the parent.
    - `get`: O(1), as dictionary lookups are on average O(1).

- **Space Complexity**: O(n), where n is the number of paths stored since we are using a dictionary to store them.

This implementation efficiently manages paths and their values while adhering to the problem constraints and requirements.

# 1172. Dinner Plate Stacks

### Problem Description 
You have an infinite number of stacks arranged in a row and numbered (left to right) from 0, each of the stacks has the same maximum `capacity`.

Implement the `DinnerPlates` class:
`DinnerPlates(int capacity)` Initializes the object with the maximum `capacity` of the stacks.

`void push(int val)` Pushes the given positive integer `val` into the leftmost stack with size less than `capacity`.

`int pop()` Returns the value at the top of the rightmost non-empty stack and removes it from that stack, and returns `-1` if all stacks are empty.

`int popAtStack(int index)` Returns the value at the top of the stack with the given `index` and removes it from that stack, and returns -1 if the stack with that given `index` is empty.


Example:
Input: 
["DinnerPlates","push","push","push","push","push","popAtStack","push","push","popAtStack","popAtStack","pop","pop","pop","pop","pop"]
[[2],[1],[2],[3],[4],[5],[0],[20],[21],[0],[2],[],[],[],[],[]]
Output: 
[null,null,null,null,null,null,2,null,null,20,21,5,4,3,1,-1]
Explanation: 
DinnerPlates D = DinnerPlates(2);  // Initialize with capacity = 2
D.push(1);
D.push(2);
D.push(3);
D.push(4);
D.push(5);         // The stacks are now:  2  4
                                           1  3  5
                                           ] ] ]
D.popAtStack(0);   // Returns 2.  The stacks are now:     4
                                                       1  3  5
                                                       ] ] ]
D.push(20);        // The stacks are now: 20  4
                                           1  3  5
                                           ] ] ]
D.push(21);        // The stacks are now: 20  4 21
                                           1  3  5
                                           ] ] ]
D.popAtStack(0);   // Returns 20.  The stacks are now:     4 21
                                                        1  3  5
                                                        ] ] ]
D.popAtStack(2);   // Returns 21.  The stacks are now:     4
                                                        1  3  5
                                                        ] ] ] 
D.pop()            // Returns 5.  The stacks are now:      4
                                                        1  3 
                                                        ] ]  
D.pop()            // Returns 4.  The stacks are now:   1  3 
                                                        ] ]   
D.pop()            // Returns 3.  The stacks are now:   1 
                                                        ]   
D.pop()            // Returns 1.  There are no stacks.

D.pop()            // Returns -1.  There are still no stacks.


Constraints:
`1 <= capacity <= 20000`
`1 <= val <= 20000`
`0 <= index <= 100000`
At most `200000` calls will be made to `push`, `pop`, and `popAtStack`.

### Solution 
 To solve the "Dinner Plates" problem, we need to keep track of multiple stacks (which are represented by a dynamic list) and perform operations like pushing values to specific stacks and popping from the rightmost non-empty stack. Below is a detailed implementation of the problem according to the LeetCode format.

### Step-by-step Explanation:

1. **Structure Design**:
    - We use a list of lists to represent the stacks. Each sublist will represent a stack. 
    - We need to manage two main lists:
        - `stacks`: This will hold our stacks (as lists).
        - `stack_indices`: This will keep track of the indices of non-full stacks in order to allow efficient `push` operations.

2. **Operations**:
    - **Push Operation**: We traverse `stack_indices` to find the leftmost stack that is not full and push the value onto it. If we cannot find such a stack, we create a new stack.
    - **Pop Operation**: We pop from the rightmost non-empty stack and return its value. If all stacks are empty, we return -1.
    - **Pop At Stack Operation**: For a given index, we pop the value at the top of that specific stack if it's not empty.

### Python Code:
Here’s the code implementation based on the requirements above:



```python
class DinnerPlates:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.stacks = []
        self.stack_indices = []  # to keep track of non-full stacks

    def push(self, val: int) -> None:
        # If there are stack indices that are not full
        if self.stack_indices:
            idx = self.stack_indices[0]
            self.stacks[idx].append(val)
            # If this stack is now full, we remove the index from stack_indices
            if len(self.stacks[idx]) == self.capacity:
                self.stack_indices.pop(0)
        else:
            # Need to create a new stack
            self.stacks.append([val])
            if self.capacity > 0:  # Only add index if capacity is greater than 0
                self.stack_indices.append(len(self.stacks) - 1)

    def pop(self) -> int:
        # Find the rightmost non-empty stack
        while self.stack_indices and not self.stacks[self.stack_indices[-1]]:
            self.stack_indices.pop()  # Remove empty stacks

        if not self.stack_indices:
            return -1  # All stacks are empty

        idx = self.stack_indices[-1]
        result = self.stacks[idx].pop()  # Pop top element
        # If current stack becomes empty, remove it from stack_indices
        if not self.stacks[idx]:
            self.stack_indices.pop()

        return result

    def popAtStack(self, index: int) -> int:
        # Check if index is valid and stack exists
        if 0 <= index < len(self.stacks) and self.stacks[index]:
            result = self.stacks[index].pop()
            # If current stack becomes empty, remove it from stack_indices
            if not self.stacks[index]:
                # Remove index from stack_indices when stack is empty
                if index in self.stack_indices:
                    self.stack_indices.remove(index)
            return result
        return -1

```

### Explanation of the Code:

- **Initialization**: 
  - The constructor initializes the `capacity`, a list `stacks` for holding plates in stacks, and `stack_indices` for managing non-full stacks.

- **Push Method**: 
  - First checks if there are available stack indices. If yes, it pushes the value onto the leftmost stack indexed by `stack_indices`.
  - If that stack becomes full after the push, the index is removed from `stack_indices`.
  - If there are no available stacks, a new stack is created.

- **Pop Method**: 
  - It looks for the rightmost non-empty stack in `stack_indices`.
  - If all stacks are empty, it returns -1.
  - Otherwise, it pops the top value and checks if the popped stack is now empty to possibly remove its index from `stack_indices`.

- **Pop At Stack Method**: 
  - It checks if the requested stack index is valid and not empty. If valid, it pops from the stack and may remove its index from `stack_indices` if the stack is then empty.

This implementation efficiently manages the stacks while adhering to the constraints and requirements provided in the problem statement.

# 1206. Design Skiplist

### Problem Description 
Design a Skiplist without using any built-in libraries.

A Skiplist is a data structure that takes O(log(n)) time to `add`, `erase` and `search`. Comparing with treap and red-black tree which has the same function and performance, the code length of Skiplist can be comparatively short and the idea behind Skiplists are just simple linked lists.

For example: we have a Skiplist containing `[30,40,50,60,70,90]` and we want to add `80` and `45` into it. The Skiplist works this way:
Artyom Kalinin [CC BY-SA 3.0], via Wikimedia Commons
You can see there are many layers in the Skiplist. Each layer is a sorted linked list. With the help of the top layers, `add` , `erase` and `search `can be faster than O(n). It can be proven that the average time complexity for each operation is O(log(n)) and space complexity is O(n).

To be specific, your design should include these functions:
`bool search(int target)` : Return whether the `target` exists in the Skiplist or not.

`void add(int num)`: Insert a value into the SkipList. 
`bool erase(int num)`: Remove a value in the Skiplist. If `num` does not exist in the Skiplist, do nothing and return false. If there exists multiple `num` values, removing any one of them is fine.

See more about Skiplist : https://en.wikipedia.org/wiki/Skip_list
Note that duplicates may exist in the Skiplist, your code needs to handle this situation.


Example:
Skiplist skiplist = new Skiplist();
skiplist.add(1);
skiplist.add(2);
skiplist.add(3);
skiplist.search(0);   // return false.

skiplist.add(4);
skiplist.search(1);   // return true.

skiplist.erase(0);    // return false, 0 is not in skiplist.

skiplist.erase(1);    // return true.

skiplist.search(1);   // return false, 1 has already been erased.


Constraints:
`0 <= num, target <= 20000`
At most `50000` calls will be made to `search`, `add`, and `erase`.

### Solution 
 To implement a Skiplist in Python, we need to create a data structure that allows for fast search, insertion, and deletion of elements. A Skiplist consists of multiple layers of linked lists, where each layer allows for skipping ahead to improve performance.

Here's how we can break down the implementation:

### Key Components of a Skiplist:
1. **Node Class**: Each node in the Skiplist will store a value and references to the next node at current layer and lower layers.
2. **Skiplist Class**: This class comprises methods to add, erase, and search for elements. It also manages the levels of the Skiplist.
3. **Random Level Generation**: We need to generate random levels for new nodes, which gives the Skiplist its probabilistic balancing.

### Operations:
- `search(target)`: Traverse through the Skiplist to find if the target exists.
- `add(num)`: Insert a new number while maintaining the ordering.
- `erase(num)`: Remove a number if it exists, irrespective of duplicates.

### Code Implementation:
Here's the implementation based on the above discussion, formatted according to LeetCode standards:



```python
import random

class SkiplistNode:
    def __init__(self, value):
        self.value = value
        self.forward = []

class Skiplist:
    def __init__(self):
        self.max_level = 16  # Maximum level of levels we are going to allow
        self.header = SkiplistNode(-1)  # Dummy header
        self.level = 0  # Current level of skiplist

    def random_level(self):
        level = 0
        while random.random() < 0.5 and level < self.max_level - 1:
            level += 1
        return level

    def search(self, target: int) -> bool:
        current = self.header
        for i in range(self.level, -1, -1):  # Start from highest level
            while (current.forward[i] is not None and current.forward[i].value < target):
                current = current.forward[i]
        current = current.forward[0]  # Move to the next node at level 0
        return current is not None and current.value == target

    def add(self, num: int) -> None:
        current = self.header
        update = [None] * self.max_level  # Update array for levels
        for i in range(self.level, -1, -1):  # Starting from the top level
            while (current.forward[i] is not None and current.forward[i].value < num):
                current = current.forward[i]
            update[i] = current  # Keep track of the last node at this level

        current = current.forward[0]  # Move down to level 0

        if current is None or current.value != num:  # Not already in the list
            new_level = self.random_level()
            if new_level > self.level:  # Increase the current level of skiplist
                for i in range(self.level + 1, new_level + 1):
                    update[i] = self.header
                self.level = new_level

            new_node = SkiplistNode(num)
            for i in range(new_level + 1):  # Link new node to the skiplist
                new_node.forward.append(None)  # Initialize forward references
                new_node.forward[i] = update[i].forward[i]  # Link to the next node
                update[i].forward[i] = new_node  # Link previous node to new node

    def erase(self, num: int) -> bool:
        current = self.header
        update = [None] * self.max_level
        for i in range(self.level, -1, -1):
            while (current.forward[i] is not None and current.forward[i].value < num):
                current = current.forward[i]
            update[i] = current  # Keep track of the last node at this level

        current = current.forward[0]  # Move to level 0

        if current is None or current.value != num:
            return False  # Not found

        for i in range(self.level + 1):
            if update[i].forward[i] != current:
                break
            update[i].forward[i] = current.forward[i]  # Bypass deleted node

        while self.level > 0 and self.header.forward[self.level] is None:  # Adjust level
            self.level -= 1
            
        return True

# Example Usage
# skiplist = Skiplist()
# skiplist.add(1)
# skiplist.add(2)
# skiplist.add(3)
# print(skiplist.search(0))   # return false.
# skiplist.add(4)
# print(skiplist.search(1))   # return true.
# print(skiplist.erase(0))    # return false, 0 is not in skiplist.
# print(skiplist.erase(1))    # return true.
# print(skiplist.search(1))    # return false, 1 has already been erased.

```

### Explanation of the Code:
- **SkiplistNode**: Represents each element in the Skiplist, containing links to its forward nodes at different levels.
- **Skiplist**: Constructs the skiplist with a maximum level and a header node.
- **random_level**: A helper method that randomly determines how many levels a newly inserted node will be added to.
- **search**: Navigates through the levels and nodes to find the target.
- **add**: Adjusts pointers to accommodate the new node and ensures that it fits into the proper place of the current levels in the skiplist.
- **erase**: Removes the specified number from the skiplist, adjusting the links accordingly.

### Complexity:
- **Time Complexity**: `O(log(n))` on average for `add`, `erase`, and `search`.
- **Space Complexity**: `O(n)` due to the linked list structure. 

This implementation effectively captures the main features of a Skiplist while adhering to the constraints provided.

# 1244. Design A Leaderboard

### Problem Description 
Design a Leaderboard class, which has 3 functions:
`addScore(playerId, score)`: Update the leaderboard by adding `score` to the given player's score. If there is no player with such id in the leaderboard, add him to the leaderboard with the given `score`.

`top(K)`: Return the score sum of the top `K` players.

`reset(playerId)`: Reset the score of the player with the given id to 0 (in other words erase it from the leaderboard). It is guaranteed that the player was added to the leaderboard before calling this function.

Initially, the leaderboard is empty.


Example 1:
Input: 
["Leaderboard","addScore","addScore","addScore","addScore","addScore","top","reset","reset","addScore","top"]
[[],[1,73],[2,56],[3,39],[4,51],[5,4],[1],[1],[2],[2,51],[3]]
Output: 
[null,null,null,null,null,null,73,null,null,null,141]
Explanation: 
Leaderboard leaderboard = new Leaderboard ();
leaderboard.addScore(1,73);   // leaderboard = [[1,73]];
leaderboard.addScore(2,56);   // leaderboard = [[1,73],[2,56]];
leaderboard.addScore(3,39);   // leaderboard = [[1,73],[2,56],[3,39]];
leaderboard.addScore(4,51);   // leaderboard = [[1,73],[2,56],[3,39],[4,51]];
leaderboard.addScore(5,4);    // leaderboard = [[1,73],[2,56],[3,39],[4,51],[5,4]];
leaderboard.top(1);           // returns 73;
leaderboard.reset(1);         // leaderboard = [[2,56],[3,39],[4,51],[5,4]];
leaderboard.reset(2);         // leaderboard = [[3,39],[4,51],[5,4]];
leaderboard.addScore(2,51);   // leaderboard = [[2,51],[3,39],[4,51],[5,4]];
leaderboard.top(3);           // returns 141 = 51 + 51 + 39;

Constraints:
`1 <= playerId, K <= 10000`
It's guaranteed that `K` is less than or equal to the current number of players.

`1 <= score <= 100`
There will be at most `1000` function calls.

### Solution 
 To solve the problem of designing a `Leaderboard` class that can add scores, return the sum of the top K players' scores, and reset player scores, we can follow a structured approach leveraging Python's dictionary and data structures.

Here's how we can implement each of the required operations:

1. **Data Structure**: We can use a dictionary to store player scores. The keys will be the player IDs and the values will be their respective scores.

2. **addScore(playerId, score)**: This method will check if the player ID already exists in our dictionary. If it does, we'll simply add the new score to the existing score. If it doesn't exist, we will initialize the player's score with the given score.

3. **top(K)**: This method needs to return the sum of the top K scores. We can retrieve all scores from our dictionary, sort them in descending order, and then sum the top K scores.

4. **reset(playerId)**: This method will reset a player's score to 0 by removing the player from the dictionary.

Let's look at the code implementation following these ideas:



```python
class Leaderboard:
    def __init__(self):
        # Initialize the leaderboard with an empty dictionary
        self.scores = {}

    def addScore(self, playerId: int, score: int) -> None:
        # Add or update the player's score
        if playerId in self.scores:
            self.scores[playerId] += score
        else:
            self.scores[playerId] = score

    def top(self, K: int) -> int:
        # Get the top K scores and return their sum
        top_scores = sorted(self.scores.values(), reverse=True)[:K]
        return sum(top_scores)

    def reset(self, playerId: int) -> None:
        # Reset the player's score
        if playerId in self.scores:
            del self.scores[playerId]


# Example usage:
# leaderboard = Leaderboard()
# leaderboard.addScore(1, 73)
# leaderboard.addScore(2, 56)
# leaderboard.addScore(3, 39)
# leaderboard.addScore(4, 51)
# leaderboard.addScore(5, 4)
# print(leaderboard.top(1))  # Returns 73
# leaderboard.reset(1)
# leaderboard.reset(2)
# leaderboard.addScore(2, 51)
# print(leaderboard.top(3))  # Returns 141

```

### Explanation of the Code:

1. **Initialization**: 
   - The constructor `__init__` initializes an empty dictionary `self.scores` to store player scores.

2. **Adding Scores (`addScore`)**:
   - When a score is added via `addScore(self, playerId, score)`, the method first checks if the player ID already exists in the dictionary. If it does, the score is incremented; otherwise, a new entry is created in the dictionary.

3. **Getting Top Scores (`top`)**:
   - The `top(self, K)` function retrieves all the scores from the dictionary, sorts them in descending order, and returns the sum of the top K scores using slicing and the built-in `sum()` function.

4. **Resetting Scores (`reset`)**:
   - The `reset(self, playerId)` method checks if the player exists in the leaderboard and removes the corresponding entry from the dictionary, effectively resetting the player's score.

This approach ensures efficient operations considering the constraints of the problem, while leveraging Python's built-in data structures and operations to maintain clarity and simplicity in the implementation.

# 1286. Iterator for Combination

### Problem Description 
Design the `CombinationIterator` class:
`CombinationIterator(string characters, int combinationLength)` Initializes the object with a string `characters` of sorted distinct lowercase English letters and a number `combinationLength` as arguments.

`next()` Returns the next combination of length `combinationLength` in lexicographical order.

`hasNext()` Returns `true` if and only if there exists a next combination.


Example 1:
Input
["CombinationIterator", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[["abc", 2], [], [], [], [], [], []]
Output
[null, "ab", true, "ac", true, "bc", false]
Explanation
CombinationIterator itr = new CombinationIterator("abc", 2);
itr.next();    // return "ab"
itr.hasNext(); // return True
itr.next();    // return "ac"
itr.hasNext(); // return True
itr.next();    // return "bc"
itr.hasNext(); // return False

Constraints:
`1 <= combinationLength <= characters.length <= 15`
All the characters of `characters` are unique.

At most `104` calls will be made to `next` and `hasNext`.

It's guaranteed that all calls of the function `next` are valid.

### Solution 
 To solve the problem of designing the `CombinationIterator` class, we need to implement a class that can efficiently generate combinations of characters in lexicographical order based on the given input parameters. Here's the step-by-step explanation and the code to achieve this.

### Step-by-step Explanation

1. **Class Initialization**:
    - The `CombinationIterator` class will be initialized with a string `characters` containing sorted distinct lowercase letters and an integer `combinationLength`, which specifies the length of each combination.

2. **Generating Combinations**:
    - To generate combinations in lexicographical order, we can use the `itertools.combinations` function from the Python standard library. This function returns all possible combinations of a specified length from the input iterable.

3. **Tracking the Current State**:
    - We will store the generated combinations in a list called `combinations` and keep track of the current index using an integer `current`.

4. **Next Combination**:
    - The `next()` method will return the next combination from the list using the `current` index and increment this index for the next call.

5. **Checking for More Combinations**:
    - The `hasNext()` method will check if `current` is still less than the total number of combinations.

### Implementation

Here's how the code looks in Python:



```python
from itertools import combinations

class CombinationIterator:
    def __init__(self, characters: str, combinationLength: int):
        # Generate all combinations of the specified length
        self.combinations = [''.join(comb) for comb in combinations(characters, combinationLength)]
        self.current = 0  # Initialize current index
        
    def next(self) -> str:
        # Return the next combination and increment current index
        result = self.combinations[self.current]
        self.current += 1
        return result
    
    def hasNext(self) -> bool:
        # Check if there is a next combination
        return self.current < len(self.combinations)

# Example usage (You can run this part directly on LeetCode platform):
# iterator = CombinationIterator("abc", 2)
# print(iterator.next())    # Output: "ab"
# print(iterator.hasNext()) # Output: True
# print(iterator.next())    # Output: "ac"
# print(iterator.hasNext()) # Output: True
# print(iterator.next())    # Output: "bc"
# print(iterator.hasNext()) # Output: False

```

### Explanation of the Code:

- **Initialization (`__init__`)**:
    - We create combinations using `itertools.combinations(characters, combinationLength)` which provides combinations in lexicographical order since `characters` are sorted and distinct.
    - The combinations are stored as strings in the list `self.combinations`.

- **Next method (`next`)**:
    - This retrieves the current combination using `self.combinations[self.current]`, returns it, and increments `self.current` to prepare for the next invocation.

- **Has Next method (`hasNext`)**:
    - This method checks if there are any more combinations left by comparing `self.current` with the length of the combinations list.

### Conclusion

This solution is efficient for the given constraints and leverages Python's powerful libraries for combinatorial generation, making the code concise and easy to understand. All methods are designed to meet the requirements set forth in the problem statement and can handle the maximum constraints comfortably.

# 1348. Tweet Counts Per Frequency

### Problem Description 
A social media company is trying to monitor activity on their site by analyzing the number of tweets that occur in select periods of time. These periods can be partitioned into smaller time chunks based on a certain frequency (every minute, hour, or day).

For example, the period `[10, 10000]` (in seconds) would be partitioned into the following time chunks with these frequencies:
Every minute (60-second chunks): `[10,69]`, `[70,129]`, `[130,189]`, `...`, `[9970,10000]`
Every hour (3600-second chunks): `[10,3609]`, `[3610,7209]`, `[7210,10000]`
Every day (86400-second chunks): `[10,10000]`
Notice that the last chunk may be shorter than the specified frequency's chunk size and will always end with the end time of the period (`10000` in the above example).

Design and implement an API to help the company with their analysis.

Implement the `TweetCounts` class:
`TweetCounts()` Initializes the `TweetCounts` object.

`void recordTweet(String tweetName, int time)` Stores the `tweetName` at the recorded `time` (in seconds).

`List<Integer> getTweetCountsPerFrequency(String freq, String tweetName, int startTime, int endTime)` Returns a list of integers representing the number of tweets with `tweetName` in each time chunk for the given period of time `[startTime, endTime]` (in seconds) and frequency `freq`.

	
`freq` is one of `"minute"`, `"hour"`, or `"day"` representing a frequency of every minute, hour, or day respectively.


Example:
Input
["TweetCounts","recordTweet","recordTweet","recordTweet","getTweetCountsPerFrequency","getTweetCountsPerFrequency","recordTweet","getTweetCountsPerFrequency"]
[[],["tweet3",0],["tweet3",60],["tweet3",10],["minute","tweet3",0,59],["minute","tweet3",0,60],["tweet3",120],["hour","tweet3",0,210]]
Output
[null,null,null,null,[2],[2,1],null,[4]]
Explanation
TweetCounts tweetCounts = new TweetCounts();
tweetCounts.recordTweet("tweet3", 0);                              // New tweet "tweet3" at time 0
tweetCounts.recordTweet("tweet3", 60);                             // New tweet "tweet3" at time 60
tweetCounts.recordTweet("tweet3", 10);                             // New tweet "tweet3" at time 10
tweetCounts.getTweetCountsPerFrequency("minute", "tweet3", 0, 59); // return [2]; chunk [0,59] had 2 tweets
tweetCounts.getTweetCountsPerFrequency("minute", "tweet3", 0, 60); // return [2,1]; chunk [0,59] had 2 tweets, chunk [60,60] had 1 tweet
tweetCounts.recordTweet("tweet3", 120);                            // New tweet "tweet3" at time 120
tweetCounts.getTweetCountsPerFrequency("hour", "tweet3", 0, 210);  // return [4]; chunk [0,210] had 4 tweets

Constraints:
`0 <= time, startTime, endTime <= 109`
`0 <= endTime - startTime <= 104`
There will be at most `104` calls in total to `recordTweet` and `getTweetCountsPerFrequency`.

### Solution 
 To solve the problem of tracking tweet counts per specified time frequency, we will implement a class called `TweetCounts`. This class will allow us to record tweets with a name and timestamp, and then retrieve the count of tweets based on specified time chunks determined by frequency (minute, hour, day).

### Explanation:

1. **Data Structure**:
   - We will use a dictionary to store the timestamps of each tweet. The key will be the tweet name, and the value will be a list that collects all the timestamps when that tweet name was recorded.

2. **Recording Tweets**:
   - The method `recordTweet(tweetName, time)` will simply append the given time to the list corresponding to the tweet name in the dictionary.

3. **Counting Tweets**:
   - The method `getTweetCountsPerFrequency(freq, tweetName, startTime, endTime)` will:
     - Determine the size of each time chunk based on the frequency:
       - `minute` => 60 seconds
       - `hour` => 3600 seconds
       - `day` => 86400 seconds
     - Calculate the number of chunks needed based on the `startTime` and `endTime`.
     - Count the number of tweets for each chunk by iterating through the collected timestamps and incrementing a counter for the corresponding chunk.

4. **Output Format**:
   - The output will be a list of counts, representing the counts of tweets in each time chunk.

### Python Code Implementation:

Now we'll write the code following this logic in the LeetCode format:



```python
from collections import defaultdict

class TweetCounts:
    def __init__(self):
        # Initialization
        self.tweets_data = defaultdict(list)

    def recordTweet(self, tweetName: str, time: int) -> None:
        # Record the tweet time
        self.tweets_data[tweetName].append(time)

    def getTweetCountsPerFrequency(self, freq: str, tweetName: str, startTime: int, endTime: int) -> list:
        # Determine chunk size depending on frequency
        if freq == "minute":
            chunk_size = 60
        elif freq == "hour":
            chunk_size = 3600
        elif freq == "day":
            chunk_size = 86400
        else:
            return []

        # Prepare an array to hold the counts
        counts = []
        
        # Get the list of times for the specified tweet name
        times = self.tweets_data[tweetName]

        # Initialize index for counting
        index = 0
        number_of_chunks = (endTime - startTime) // chunk_size + 1

        # Iterate over the time range
        for i in range(number_of_chunks):
            # Determine the current chunk range
            current_chunk_start = startTime + i * chunk_size
            current_chunk_end = current_chunk_start + chunk_size - 1

            # Count tweets in the current chunk
            count_in_chunk = 0
            while index < len(times) and times[index] <= current_chunk_end:
                if times[index] >= current_chunk_start:
                    count_in_chunk += 1
                index += 1

            # Append the count to the result list
            counts.append(count_in_chunk)

        return counts

```

### Explanation of the Code:
- We start by initializing the `TweetCounts` class with a dictionary `tweets_data` that uses a default list as its value type for storing timestamps.
- The `recordTweet` method simply appends the `time` to the list corresponding to `tweetName`.
- The `getTweetCountsPerFrequency` method calculates the size of time chunks based on the frequency and prepares a count list.
- Using a while-loop, we efficiently count how many tweets fall into each time chunk based on the provided start and end times.
- Finally, it returns the list of counts representing the number of tweets in each time segment.

This implementation meets the problem constraints and should perform efficiently given the maximum call limits.

# 1352. Product of the Last K Numbers

### Problem Description 
Implement the class `ProductOfNumbers` that supports two methods:
1.` add(int num)`
Adds the number `num` to the back of the current list of numbers.

2.` getProduct(int k)`
Returns the product of the last `k` numbers in the current list.

You can assume that always the current list has at least `k` numbers.

At any time, the product of any contiguous sequence of numbers will fit into a single 32-bit integer without overflowing.


Example:
Input
["ProductOfNumbers","add","add","add","add","add","getProduct","getProduct","getProduct","add","getProduct"]
[[],[3],[0],[2],[5],[4],[2],[3],[4],[8],[2]]
Output
[null,null,null,null,null,null,20,40,0,null,32]
Explanation
ProductOfNumbers productOfNumbers = new ProductOfNumbers();
productOfNumbers.add(3);        // [3]
productOfNumbers.add(0);        // [3,0]
productOfNumbers.add(2);        // [3,0,2]
productOfNumbers.add(5);        // [3,0,2,5]
productOfNumbers.add(4);        // [3,0,2,5,4]
productOfNumbers.getProduct(2); // return 20. The product of the last 2 numbers is 5 * 4 = 20
productOfNumbers.getProduct(3); // return 40. The product of the last 3 numbers is 2 * 5 * 4 = 40
productOfNumbers.getProduct(4); // return 0. The product of the last 4 numbers is 0 * 2 * 5 * 4 = 0
productOfNumbers.add(8);        // [3,0,2,5,4,8]
productOfNumbers.getProduct(2); // return 32. The product of the last 2 numbers is 4 * 8 = 32 

Constraints:
There will be at most `40000` operations considering both `add` and `getProduct`.

`0 <= num <= 100`
`1 <= k <= 40000`

### Solution 
 To solve the problem of implementing a class `ProductOfNumbers` that efficiently maintains a list of numbers and computes the products of the last `k` numbers, we can take advantage of the properties of multiplication and zero in the list.

### Problem Breakdown
1. **Storage**: We need a way to store the numbers added to our class. An array (or list in Python) will be appropriate for this purpose.
2. **Product Calculation**:
   - When adding a number, if the number is `0`, we reset the product history because any number multiplied by `0` results in `0`.
   - If the number is non-zero, we can keep track of the cumulative product of numbers until the next `0` is encountered.
3. **Get Product**:
   - We need to compute the product of the last `k` items efficiently. If a `0` exists within the last `k` items, the result will be `0`.
   - If there are no `0`s, we can use the cumulative product information to quickly compute the product.

### Approach
1. Maintain an array to store the numbers added.
2. Use a variable to hold the last product computed when non-zero numbers are present.
3. When `add` is called, update the list of numbers and reset the product if `0` is added.
4. For `getProduct`, check if `0` falls within the last `k` indices. If yes, return `0`, otherwise compute the product based on the cumulative product.

### Implementation
Here's how we can implement this in Python. We'll encapsulate the logic within a class, providing the necessary methods as described.



```python
class ProductOfNumbers:
    def __init__(self):
        self.nums = []
        self.last_product = 1  # This will hold the product of last non-zero numbers
        self.last_zero_index = -1  # Index of the last zero
        
    def add(self, num: int) -> None:
        if num == 0:
            # When we hit a zero, we reset the product and store the index of this zero
            self.last_zero_index = len(self.nums)
            self.last_product = 1  # Reset product
        else:
            if self.last_zero_index == -1:  # No zero in the previous numbers
                if self.nums:
                    # Multiply with the previous product to maintain cumulative product
                    self.last_product *= num
                else:
                    self.last_product = num
            
        self.nums.append(num)  # Add the number to the list

    def getProduct(self, k: int) -> int:
        if len(self.nums) < k:  # Edge case, should not happen per constraints
            return 0

        # Check if the last k items include a zero
        if self.last_zero_index >= len(self.nums) - k: 
            return 0  # There is a zero in the last k numbers
        
        # If no zero in the last k numbers, return the product of the last k
        # We can calculate the product as follows:
        # For the range `[-k:]`, we have to divide the product of valid numbers
        # in nums by the product of the numbers that are out of this range.
        
        num_end_index = len(self.nums) - 1
        num_start_index = num_end_index - k + 1
        
        # If we are encountering zero, we just rely on last_product
        prefix_product = self.nums[num_start_index - 1] if num_start_index > 0 else 1
        
        # Calculate the last k product by grouping
        return self.last_product // prefix_product
    
# Example of usage:
# productOfNumbers = ProductOfNumbers()
# productOfNumbers.add(3)
# productOfNumbers.add(0)
# ... follow the example sequence ...


```

### Explanation
- **Initialization**: The `__init__` method initializes an empty list to store numbers, a variable for the product of the last non-zero numbers, and an index to track the last occurrence of `0`.
- **Add Number**: The `add` method updates the list. If it encounters `0`, it resets the product and updates the index of the last zero. For non-zero numbers, it updates the running product.
- **Get Product**: The `getProduct` method checks if `0` is present in the last `k` elements before performing multiplication of the effective range.

### Complexity
- Both methods operate in O(1) time for adding numbers and O(1) time for querying the product as the maximum size of numbers and operations are capped based on given constraints.

This implementation can be directly utilized in a context like LeetCode for solving the given problem as described.

# 1357. Apply Discount Every n Orders

### Problem Description 
There is a sale in a supermarket, there will be a `discount` every `n` customer.

There are some products in the supermarket where the id of the `i-th` product is `products[i]` and the price per unit of this product is `prices[i]`.

The system will count the number of customers and when the `n-th` customer arrive he/she will have a `discount` on the bill. (i.e if the cost is `x` the new cost is `x - (discount * x) / 100`). Then the system will start counting customers again.

The customer orders a certain amount of each product where `product[i]` is the id of the `i-th` product the customer ordered and `amount[i]` is the number of units the customer ordered of that product.

Implement the `Cashier` class:
`Cashier(int n, int discount, int[] products, int[] prices)` Initializes the object with `n`, the `discount`, the `products` and their `prices`.

`double getBill(int[] product, int[] amount)` returns the value of the bill and apply the discount if needed. Answers within `10^-5` of the actual value will be accepted as correct.


Example 1:
Input
["Cashier","getBill","getBill","getBill","getBill","getBill","getBill","getBill"]
[[3,50,[1,2,3,4,5,6,7],[100,200,300,400,300,200,100]],[[1,2],[1,2]],[[3,7],[10,10]],[[1,2,3,4,5,6,7],[1,1,1,1,1,1,1]],[[4],[10]],[[7,3],[10,10]],[[7,5,3,1,6,4,2],[10,10,10,9,9,9,7]],[[2,3,5],[5,3,2]]]
Output
[null,500.0,4000.0,800.0,4000.0,4000.0,7350.0,2500.0]
Explanation
Cashier cashier = new Cashier(3,50,[1,2,3,4,5,6,7],[100,200,300,400,300,200,100]);
cashier.getBill([1,2],[1,2]);                        // return 500.0, bill = 1 * 100 + 2 * 200 = 500.

cashier.getBill([3,7],[10,10]);                      // return 4000.0
cashier.getBill([1,2,3,4,5,6,7],[1,1,1,1,1,1,1]);    // return 800.0, The bill was 1600.0 but as this is the third customer, he has a discount of 50% which means his bill is only 1600 - 1600 * (50 / 100) = 800.

cashier.getBill([4],[10]);                           // return 4000.0
cashier.getBill([7,3],[10,10]);                      // return 4000.0
cashier.getBill([7,5,3,1,6,4,2],[10,10,10,9,9,9,7]); // return 7350.0, Bill was 14700.0 but as the system counted three more customers, he will have a 50% discount and the bill becomes 7350.0
cashier.getBill([2,3,5],[5,3,2]);                    // return 2500.0

Constraints:
`1 <= n <= 10^4`
`0 <= discount <= 100`
`1 <= products.length <= 200`
`1 <= products[i] <= 200`
There are not repeated elements in the array `products`.

`prices.length == products.length`
`1 <= prices[i] <= 1000`
`1 <= product.length <= products.length`
`product[i]` exists in `products`.

`amount.length == product.length`
`1 <= amount[i] <= 1000`
At most `1000` calls will be made to `getBill`.

Answers within `10^-5` of the actual value will be accepted as correct.

### Solution 
 To solve the cashier problem described above, we can use an object-oriented approach to implement the required `Cashier` class. This class will keep track of the number of customers served and apply discounts accordingly when a certain number of customers have been served.

Here’s how we can set up the `Cashier` class step by step:

### Class Structure

1. **Initialization**: The constructor `__init__` will take parameters `n` (the number of customers after which a discount is applied), `discount` (the percentage discount), `products` (array of product ids), and `prices` (array of prices corresponding to each product).

2. **Bill Calculation**: The method `getBill` will calculate the total bill for a given order. It will check how many customers have been served so far. If the count is a multiple of `n`, it applies the discount.

3. **Discount Application**: The discount is applied as per the formula provided: `discount_amount = total_bill * (discount / 100)`.

### Implementation

Here’s the complete code following the LeetCode format:



```python
class Cashier:
    def __init__(self, n: int, discount: int, products: list[int], prices: list[int]):
        self.n = n                          # number of customers for discount
        self.discount = discount              # discount percentage
        self.products = products              # product id array
        self.prices = prices                  # corresponding prices
        self.customer_count = 0               # to count the number of customers

        # Create a mapping from product ID to price
        self.price_map = {product: price for product, price in zip(products, prices)}
        
    def getBill(self, product: list[int], amount: list[int]) -> float:
        total_bill = 0                        # Initialize total bill

        # Calculate the total bill
        for prod, amt in zip(product, amount):
            total_bill += self.price_map[prod] * amt
        
        self.customer_count += 1              # Increment the customer count

        # Check if discount applies
        if self.customer_count % self.n == 0:
            total_bill -= total_bill * (self.discount / 100)  # Apply discount
        
        return total_bill                     # Return the final bill amount

```

### Explanation of the Code

- **Constructor (`__init__`)**:
  - Initializes the `n`, `discount`, `products`, `prices`, and `customer_count`.
  - It creates a mapping (`price_map`) from each product id to its price for efficient lookup.

- **Method `getBill`**:
  - Takes two lists as input: `product` (ids of the products the customer wants) and `amount` (the corresponding quantities).
  - It calculates the initial total bill by iterating through the products and their amounts, multiplying them by their prices using the `price_map`.
  - After calculating the total bill, it increments the `customer_count`. 
  - If the `customer_count` is a multiple of `n`, it applies the discount to the total bill using the formula provided.
  - Finally, it returns the total bill.

### Testing the Implementation

The implementation can be tested against the provided example by creating an instance of the `Cashier` class and calling the `getBill` method with various orders, checking if the outputs match the expected results.

This solution effectively handles the specified constraints and requirements, ensuring accuracy in calculations and proper handling of customer discounts.

# 1381. Design a Stack With Increment Operation

### Problem Description 
Design a stack which supports the following operations.

Implement the `CustomStack` class:
`CustomStack(int maxSize)` Initializes the object with `maxSize` which is the maximum number of elements in the stack or do nothing if the stack reached the `maxSize`.

`void push(int x)` Adds `x` to the top of the stack if the stack hasn't reached the `maxSize`.

`int pop()` Pops and returns the top of stack or -1 if the stack is empty.

`void inc(int k, int val)` Increments the bottom `k` elements of the stack by `val`. If there are less than `k` elements in the stack, just increment all the elements in the stack.


Example 1:
Input
["CustomStack","push","push","pop","push","push","push","increment","increment","pop","pop","pop","pop"]
[[3],[1],[2],[],[2],[3],[4],[5,100],[2,100],[],[],[],[]]
Output
[null,null,null,2,null,null,null,null,null,103,202,201,-1]
Explanation
CustomStack customStack = new CustomStack(3); // Stack is Empty []
customStack.push(1);                          // stack becomes [1]
customStack.push(2);                          // stack becomes [1, 2]
customStack.pop();                            // return 2 --> Return top of the stack 2, stack becomes [1]
customStack.push(2);                          // stack becomes [1, 2]
customStack.push(3);                          // stack becomes [1, 2, 3]
customStack.push(4);                          // stack still [1, 2, 3], Don't add another elements as size is 4
customStack.increment(5, 100);                // stack becomes [101, 102, 103]
customStack.increment(2, 100);                // stack becomes [201, 202, 103]
customStack.pop();                            // return 103 --> Return top of the stack 103, stack becomes [201, 202]
customStack.pop();                            // return 202 --> Return top of the stack 102, stack becomes [201]
customStack.pop();                            // return 201 --> Return top of the stack 101, stack becomes []
customStack.pop();                            // return -1 --> Stack is empty return -1.


Constraints:
`1 <= maxSize <= 1000`
`1 <= x <= 1000`
`1 <= k <= 1000`
`0 <= val <= 100`
At most `1000` calls will be made to each method of `increment`, `push` and `pop` each separately.

### Solution 
 To implement the `CustomStack` class according to the problem statement, we need to create a stack with a limited size, and provide the ability to push elements onto the stack, pop elements off, and increment the bottom `k` elements by a specified value. Here’s a step-by-step explanation of how we will achieve this:

### Class Definition
We will define the `CustomStack` class. The class will maintain a stack to store the current elements and another list to record the increments that we apply to the bottom `k` elements.

### Methods
1. **`__init__(self, maxSize)`**: This constructor initializes the stack with a fixed maximum size. If we attempt to add more elements than this size, those attempts will be ignored.

2. **`push(self, x)`**: This method adds an element `x` to the stack if its current size is less than `maxSize`.

3. **`pop(self)`**: This method removes and returns the top element of the stack. If the stack is empty, it returns `-1`.

4. **`inc(self, k, val)`**: This method increments the bottom `k` elements of the stack by `val`. If `k` exceeds the current size of the stack, it simply increments all elements.

### Data Structure
We'll use a list for the stack and a list of increments:
- `stack`: A list to manage the elements.
- `increments`: A list to manage incremental values for each index.

### Detailed Implementation

Here's the complete implementation following the above plan:



```python
class CustomStack:

    def __init__(self, maxSize: int):
        self.maxSize = maxSize
        self.stack = []
        self.increments = [0] * maxSize  # to store increments

    def push(self, x: int) -> None:
        if len(self.stack) < self.maxSize:
            self.stack.append(x)

    def pop(self) -> int:
        if not self.stack:
            return -1
        index = len(self.stack) - 1
        value = self.stack.pop() + self.increments[index]
        if index > 0:
            self.increments[index - 1] += self.increments[index]  # pass the increment to the previous item
        self.increments[index] = 0  # reset the increment for the popped item
        return value

    def inc(self, k: int, val: int) -> None:
        limit = min(k, len(self.stack))  # only increment up to the size of the stack
        if limit > 0:
            self.increments[limit - 1] += val  # increment the value at the last index we are affecting

```

### Explanation of the Implementation:
1. **Initialization**: The `__init__` method sets up the initial conditions of our stack and increments array.

2. **Push Method**: The `push` method checks if the stack has not reached its maximum size to avoid overflow and then appends the new value.

3. **Pop Method**: In the `pop` method, we check if the stack is empty. If not, we retrieve the index of the top element, add any increment affecting it, and pop the value. If there are previous elements in the stack, we carry forward any increment for that element.

4. **Increment Method**: The `inc` method adjusts the increment at the index that corresponds to the bottom `k` element in the stack by targeting only the valid indices.

### Example Usage:
You can use this class to demonstrate the functionality as shown in the example test case. This class performs the required operations efficiently and adheres to the constraints provided.

The complexity mainly revolves around `O(1)` for `push`, `pop`, and `inc`, making it very efficient for the stated problem. The usage of an auxiliary `increments` array ensures we can efficiently handle the incrementing process without immediate modifications to the stack itself, maintaining both speed and cleanliness in managing the stack.

# 1396. Design Underground System

### Problem Description 
An underground railway system is keeping track of customer travel times between different stations. They are using this data to calculate the average time it takes to travel from one station to another.

Implement the `UndergroundSystem` class:
`void checkIn(int id, string stationName, int t)`
	
A customer with a card ID equal to `id`, checks in at the station `stationName` at time `t`.

A customer can only be checked into one place at a time.

`void checkOut(int id, string stationName, int t)`
	
A customer with a card ID equal to `id`, checks out from the station `stationName` at time `t`.

`double getAverageTime(string startStation, string endStation)`
	
Returns the average time it takes to travel from `startStation` to `endStation`.

The average time is computed from all the previous traveling times from `startStation` to `endStation` that happened directly, meaning a check in at `startStation` followed by a check out from `endStation`.

The time it takes to travel from `startStation` to `endStation` may be different from the time it takes to travel from `endStation` to `startStation`.

There will be at least one customer that has traveled from `startStation` to `endStation` before `getAverageTime` is called.

You may assume all calls to the `checkIn` and `checkOut` methods are consistent. If a customer checks in at time `t1` then checks out at time `t2`, then `t1 < t2`. All events happen in chronological order.


Example 1:
Input
["UndergroundSystem","checkIn","checkIn","checkIn","checkOut","checkOut","checkOut","getAverageTime","getAverageTime","checkIn","getAverageTime","checkOut","getAverageTime"]
[[],[45,"Leyton",3],[32,"Paradise",8],[27,"Leyton",10],[45,"Waterloo",15],[27,"Waterloo",20],[32,"Cambridge",22],["Paradise","Cambridge"],["Leyton","Waterloo"],[10,"Leyton",24],["Leyton","Waterloo"],[10,"Waterloo",38],["Leyton","Waterloo"]]
Output
[null,null,null,null,null,null,null,14.00000,11.00000,null,11.00000,null,12.00000]
Explanation
UndergroundSystem undergroundSystem = new UndergroundSystem();
undergroundSystem.checkIn(45, "Leyton", 3);
undergroundSystem.checkIn(32, "Paradise", 8);
undergroundSystem.checkIn(27, "Leyton", 10);
undergroundSystem.checkOut(45, "Waterloo", 15);  // Customer 45 "Leyton" -> "Waterloo" in 15-3 = 12
undergroundSystem.checkOut(27, "Waterloo", 20);  // Customer 27 "Leyton" -> "Waterloo" in 20-10 = 10
undergroundSystem.checkOut(32, "Cambridge", 22); // Customer 32 "Paradise" -> "Cambridge" in 22-8 = 14
undergroundSystem.getAverageTime("Paradise", "Cambridge"); // return 14.00000. One trip "Paradise" -> "Cambridge", (14) / 1 = 14
undergroundSystem.getAverageTime("Leyton", "Waterloo");    // return 11.00000. Two trips "Leyton" -> "Waterloo", (10 + 12) / 2 = 11
undergroundSystem.checkIn(10, "Leyton", 24);
undergroundSystem.getAverageTime("Leyton", "Waterloo");    // return 11.00000
undergroundSystem.checkOut(10, "Waterloo", 38);  // Customer 10 "Leyton" -> "Waterloo" in 38-24 = 14
undergroundSystem.getAverageTime("Leyton", "Waterloo");    // return 12.00000. Three trips "Leyton" -> "Waterloo", (10 + 12 + 14) / 3 = 12

Example 2:
Input
["UndergroundSystem","checkIn","checkOut","getAverageTime","checkIn","checkOut","getAverageTime","checkIn","checkOut","getAverageTime"]
[[],[10,"Leyton",3],[10,"Paradise",8],["Leyton","Paradise"],[5,"Leyton",10],[5,"Paradise",16],["Leyton","Paradise"],[2,"Leyton",21],[2,"Paradise",30],["Leyton","Paradise"]]
Output
[null,null,null,5.00000,null,null,5.50000,null,null,6.66667]
Explanation
UndergroundSystem undergroundSystem = new UndergroundSystem();
undergroundSystem.checkIn(10, "Leyton", 3);
undergroundSystem.checkOut(10, "Paradise", 8); // Customer 10 "Leyton" -> "Paradise" in 8-3 = 5
undergroundSystem.getAverageTime("Leyton", "Paradise"); // return 5.00000, (5) / 1 = 5
undergroundSystem.checkIn(5, "Leyton", 10);
undergroundSystem.checkOut(5, "Paradise", 16); // Customer 5 "Leyton" -> "Paradise" in 16-10 = 6
undergroundSystem.getAverageTime("Leyton", "Paradise"); // return 5.50000, (5 + 6) / 2 = 5.5
undergroundSystem.checkIn(2, "Leyton", 21);
undergroundSystem.checkOut(2, "Paradise", 30); // Customer 2 "Leyton" -> "Paradise" in 30-21 = 9
undergroundSystem.getAverageTime("Leyton", "Paradise"); // return 6.66667, (5 + 6 + 9) / 3 = 6.66667

Constraints:
`1 <= id, t <= 106`
`1 <= stationName.length, startStation.length, endStation.length <= 10`
All strings consist of uppercase and lowercase English letters and digits.

There will be at most `2 * 104` calls in total to `checkIn`, `checkOut`, and `getAverageTime`.

Answers within `10-5` of the actual value will be accepted.

### Solution 
 Let's implement the `UndergroundSystem` class to manage the underground railway system as described in the problem. The implementation will keep track of customers' check-in and check-out times and calculate the average travel times between stations.

### Breakdown of the Solution

1. **Data Structures**:
   - We will use a dictionary `check_in_map` to store customer check-in details. The keys will be customer IDs, and the values will be tuples of `(stationName, time)`.
   - We will use another dictionary `travel_times` to accumulate travel times. The keys will be tuples of `(startStation, endStation)`, and the values will be a list containing the cumulative travel time and the count of trips for that station pair.

2. **Methods**:
   - `checkIn(id, stationName, t)`: On check-in, store the station and time for the customer ID.
   - `checkOut(id, stationName, t)`: On check-out, calculate the travel time based on check-in and add it to `travel_times` dictionary.
   - `getAverageTime(startStation, endStation)`: Calculate the average travel time using the accumulated sums and counts stored in `travel_times`.

### Implementation

Now, let's implement the class according to the defined methods:



```python
class UndergroundSystem:

    def __init__(self):
        # This will hold the check-in details for customers
        self.check_in_map = {}
        # This will hold the total travel times and the number of trips between station pairs
        self.travel_times = {}

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        # Store the customer's check-in details (station and time)
        self.check_in_map[id] = (stationName, t)

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        # Retrieve the check-in details
        startStation, startTime = self.check_in_map.pop(id)
        travel_time = t - startTime
        
        # Update or add travel time to the travel_times dictionary
        if (startStation, stationName) not in self.travel_times:
            self.travel_times[(startStation, stationName)] = [0, 0]  # [total_time, count]
        
        self.travel_times[(startStation, stationName)][0] += travel_time  # total_time
        self.travel_times[(startStation, stationName)][1] += 1  # count of trips

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        total_time, count = self.travel_times[(startStation, endStation)]
        return total_time / count  # calculate average time

```

### Explanation of the Implementation

1. **Initialization**:
   - `__init__`: Initializes two dictionaries. `check_in_map` tracks the state of each customer when they check in, while `travel_times` accumulates data for each pair of stations.

2. **Check-In Method**:
   - `checkIn`: This method records when a customer checks in. It simply adds their station and time to `check_in_map` keyed by their ID.

3. **Check-Out Method**:
   - `checkOut`: When a customer checks out, we retrieve their check-in details, calculate the travel time, and then update the corresponding entry in the `travel_times` dictionary. If the station pair is new, we initialize it first.

4. **Get Average Time Method**:
   - `getAverageTime`: This method retrieves the total travel time and count for the specified station pair and uses it to compute the average.

### Testing the Implementation

You can create an instance of the `UndergroundSystem` and call the methods as described in the examples to verify that the output matches the expected results.

This implementation ensures that we efficiently manage customer check-in and check-out data while calculating averages on-the-fly. Each operation is efficient, making the solution suitable for the constraints provided.

# 1429. First Unique Number

### Problem Description 
You have a queue of integers, you need to retrieve the first unique integer in the queue.

Implement the `FirstUnique` class:
`FirstUnique(int[] nums)` Initializes the object with the numbers in the queue.

`int showFirstUnique()` returns the value of the first unique integer of the queue, and returns -1 if there is no such integer.

`void add(int value)` insert value to the queue.


Example 1:
Input: 
["FirstUnique","showFirstUnique","add","showFirstUnique","add","showFirstUnique","add","showFirstUnique"]
[[[2,3,5]],[],[5],[],[2],[],[3],[]]
Output: 
[null,2,null,2,null,3,null,-1]
Explanation: 
FirstUnique firstUnique = new FirstUnique([2,3,5]);
firstUnique.showFirstUnique(); // return 2
firstUnique.add(5);            // the queue is now [2,3,5,5]
firstUnique.showFirstUnique(); // return 2
firstUnique.add(2);            // the queue is now [2,3,5,5,2]
firstUnique.showFirstUnique(); // return 3
firstUnique.add(3);            // the queue is now [2,3,5,5,2,3]
firstUnique.showFirstUnique(); // return -1

Example 2:
Input: 
["FirstUnique","showFirstUnique","add","add","add","add","add","showFirstUnique"]
[[[7,7,7,7,7,7]],[],[7],[3],[3],[7],[17],[]]
Output: 
[null,-1,null,null,null,null,null,17]
Explanation: 
FirstUnique firstUnique = new FirstUnique([7,7,7,7,7,7]);
firstUnique.showFirstUnique(); // return -1
firstUnique.add(7);            // the queue is now [7,7,7,7,7,7,7]
firstUnique.add(3);            // the queue is now [7,7,7,7,7,7,7,3]
firstUnique.add(3);            // the queue is now [7,7,7,7,7,7,7,3,3]
firstUnique.add(7);            // the queue is now [7,7,7,7,7,7,7,3,3,7]
firstUnique.add(17);           // the queue is now [7,7,7,7,7,7,7,3,3,7,17]
firstUnique.showFirstUnique(); // return 17

Example 3:
Input: 
["FirstUnique","showFirstUnique","add","showFirstUnique"]
[[[809]],[],[809],[]]
Output: 
[null,809,null,-1]
Explanation: 
FirstUnique firstUnique = new FirstUnique([809]);
firstUnique.showFirstUnique(); // return 809
firstUnique.add(809);          // the queue is now [809,809]
firstUnique.showFirstUnique(); // return -1

Constraints:
`1 <= nums.length <= 10^5`
`1 <= nums[i] <= 10^8`
`1 <= value <= 10^8`
At most `50000` calls will be made to `showFirstUnique` and `add`.

### Solution 
 To solve the problem of finding the first unique integer in a queue, we need a class that maintains an efficient way of tracking both the elements and their counts. We can achieve this by using a combination of an `OrderedDict` for maintaining the order and a normal dictionary for counting occurrences. 

Here's how we can implement the `FirstUnique` class step-by-step:

### Implementation Steps

1. **Data Structures**:
   - **OrderedDict**: This will help maintain the insertion order of elements.
   - **Dictionary**: This will count the occurrences of each number.

2. **Constructor (`__init__`)**:
   - Initialize the `OrderedDict` and the count dictionary with the values from the input list.

3. **Method `showFirstUnique()`**:
   - Iterate over the `OrderedDict` to find the first element that appears only once. If found, return that element; otherwise, return -1.

4. **Method `add(value)`**:
   - If the `value` is already in the count dictionary, increment its count. If it's a new value, add it to both the count dictionary and the `OrderedDict`.

This design ensures that both methods operate efficiently under the constraints provided, meaning each operation can be performed in average O(1) time due to the properties of the underlying data structures.

### Code Implementation

Here's the complete code according to the LeetCode standards:



```python
from collections import OrderedDict

class FirstUnique:
    def __init__(self, nums: list[int]):
        # Initialize the ordered dictionary and count dictionary
        self.count = {}
        self.order = OrderedDict()
        for num in nums:
            self.add(num)
        
    def showFirstUnique(self) -> int:
        # Check the first unique number in order
        for num in self.order.keys():
            if self.count[num] == 1:
                return num
        return -1
    
    def add(self, value: int) -> None:
        # Increase the count of the number or add it as unique
        if value in self.count:
            self.count[value] += 1
            if value in self.order:
                del self.order[value]  # Remove from ordered list if it's not unique
        else:
            self.count[value] = 1
            self.order[value] = None  # Store the number in ordered dict

# Example usage
# Create the FirstUnique object
firstUnique = FirstUnique([2, 3, 5])
print(firstUnique.showFirstUnique())  # Output: 2
firstUnique.add(5)
print(firstUnique.showFirstUnique())  # Output: 2
firstUnique.add(2)
print(firstUnique.showFirstUnique())  # Output: 3
firstUnique.add(3)
print(firstUnique.showFirstUnique())  # Output: -1

```

### Detailed Explanation:

- **Initialization (`__init__`)**: This method takes an array `nums`, iterates through it, and calls the `add` method for each number. This ensures that the numbers are added to the tracking structures immediately upon instantiation.
  
- **Check First Unique (`showFirstUnique`)**: It checks the `OrderedDict` for any key whose count is equal to 1 and returns it. If none are found, it returns -1.

- **Add Number (`add`)**: In this method, we update the count for the incoming `value`. If it’s coming in for the first time, it will be added to both the count dictionary and the OrderedDict. If it’s seen again and thus is no longer unique, we remove it from the OrderedDict.

### Performance:
- This implementation is efficient enough to handle the constraints of this problem, which allows up to 50,000 calls to `showFirstUnique` and `add`. 

You can run this code directly on LeetCode by copying and pasting it into the LeetCode editor, and it should function correctly with the provided examples.

