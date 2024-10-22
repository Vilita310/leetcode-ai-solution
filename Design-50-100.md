# 1472. Design Browser History

### Problem Description 
You have a browser of one tab where you start on the `homepage` and you can visit another `url`, get back in the history number of `steps` or move forward in the history number of `steps`.

Implement the `BrowserHistory` class:
`BrowserHistory(string homepage)` Initializes the object with the `homepage` of the browser.

`void visit(string url)` Visits `url` from the current page. It clears up all the forward history.

`string back(int steps)` Move `steps` back in history. If you can only return `x` steps in the history and `steps > x`, you will return only `x` steps. Return the current `url` after moving back in history at most `steps`.

`string forward(int steps)` Move `steps` forward in history. If you can only forward `x` steps in the history and `steps > x`, you will forward only `x` steps. Return the current `url` after forwarding in history at most `steps`.


Example:
Input:
["BrowserHistory","visit","visit","visit","back","back","forward","visit","forward","back","back"]
[["leetcode.com"],["google.com"],["facebook.com"],["youtube.com"],[1],[1],[1],["linkedin.com"],[2],[2],[7]]
Output:
[null,null,null,null,"facebook.com","google.com","facebook.com",null,"linkedin.com","google.com","leetcode.com"]
Explanation:
BrowserHistory browserHistory = new BrowserHistory("leetcode.com");
browserHistory.visit("google.com");       // You are in "leetcode.com". Visit "google.com"
browserHistory.visit("facebook.com");     // You are in "google.com". Visit "facebook.com"
browserHistory.visit("youtube.com");      // You are in "facebook.com". Visit "youtube.com"
browserHistory.back(1);                   // You are in "youtube.com", move back to "facebook.com" return "facebook.com"
browserHistory.back(1);                   // You are in "facebook.com", move back to "google.com" return "google.com"
browserHistory.forward(1);                // You are in "google.com", move forward to "facebook.com" return "facebook.com"
browserHistory.visit("linkedin.com");     // You are in "facebook.com". Visit "linkedin.com"
browserHistory.forward(2);                // You are in "linkedin.com", you cannot move forward any steps.

browserHistory.back(2);                   // You are in "linkedin.com", move back two steps to "facebook.com" then to "google.com". return "google.com"
browserHistory.back(7);                   // You are in "google.com", you can move back only one step to "leetcode.com". return "leetcode.com"

Constraints:
`1 <= homepage.length <= 20`
`1 <= url.length <= 20`
`1 <= steps <= 100`
`homepage` and `url` consist of  '.' or lower case English letters.

At most `5000` calls will be made to `visit`, `back`, and `forward`.

### Solution 
 To solve the problem of implementing a `BrowserHistory` class that manages browsing history, we can use a list to keep track of the URLs visited in chronological order. We'll maintain a pointer to indicate the current position in the history. This allows us to efficiently handle visits, backs, and forwards.

### Implementation Concept

1. **Initialization**: When the `BrowserHistory` class is initialized with a `homepage`, we create a list to store the URLs and set the current position to the index of the homepage (0).

2. **Visit Method**: When visiting a new URL:
   - We append the new URL to our history.
   - Clear any forward history by trimming the list to contain only the back history up to the current position.

3. **Back Method**: To move back in history:
   - We decrement the current position by the number of requested steps, but ensure it doesn't go below zero.
   - Return the URL at the current position.

4. **Forward Method**: To move forward in history:
   - We increment the current position by the number of requested steps, but ensure it doesn't exceed the length of the history list.
   - Return the URL at the current position.

### Detailed Python Code



```python
class BrowserHistory:
    def __init__(self, homepage: str):
        # Initialize the history with the homepage
        self.history = [homepage]  # Store URLs in a list
        self.current = 0           # Variable to keep track of the current position in the history

    def visit(self, url: str) -> None:
        # Visit a new URL
        # Clear forward history
        self.history = self.history[:self.current + 1]  # Keep only back history
        self.history.append(url)                         # Add the new URL
        self.current += 1                                # Update current position

    def back(self, steps: int) -> str:
        # Move back in history
        self.current = max(0, self.current - steps)     # Ensure current does not go below 0
        return self.history[self.current]                # Return the current URL after moving back

    def forward(self, steps: int) -> str:
        # Move forward in history
        self.current = min(len(self.history) - 1, self.current + steps)  # Ensure current does not exceed size of history
        return self.history[self.current]                # Return the current URL after moving forward

```

### Explanation of the Code:

- **Constructor (`__init__`)**:
  - Takes a `homepage` string and initializes the history with this homepage, storing it in a list. The current position (`self.current`) is initialized to 0, indicating that the user is on the homepage.

- **Visit Method (`visit`)**:
  - When a new URL is visited, only the part of history before the current position is kept (by slicing the list). The new URL is appended, and the current index is updated to reflect the new position after the visit.

- **Back Method (`back`)**:
  - Decreases the current index by the number of steps requested, ensuring it does not go below 0. The URL at the current index is returned.

- **Forward Method (`forward`)**:
  - Increases the index similarly, ensuring it does not exceed the last index of the history list. The URL at this index is returned.

### Example Usage

You can create an instance of `BrowserHistory` and call its methods as shown in the problem description:



```python
browserHistory = BrowserHistory("leetcode.com")
browserHistory.visit("google.com")       # Visits google.com
browserHistory.visit("facebook.com")     # Visits facebook.com
browserHistory.visit("youtube.com")      # Visits youtube.com
print(browserHistory.back(1))            # Returns "facebook.com"
print(browserHistory.back(1))            # Returns "google.com"
print(browserHistory.forward(1))         # Returns "facebook.com"
browserHistory.visit("linkedin.com")     # Visits linkedin.com
print(browserHistory.forward(2))         # Returns "linkedin.com" (can only forward 1 step)
print(browserHistory.back(2))            # Returns "google.com"
print(browserHistory.back(7))            # Returns "leetcode.com" (can only go back to the homepage)

```

This implementation efficiently handles the history navigation and is structured in a way that can be run directly on LeetCode.

# 1500. Design a File Sharing System

### Problem Description 
We will use a file-sharing system to share a very large file which consists of `m` small chunks with IDs from `1` to `m`.

When users join the system, the system should assign a unique ID to them. The unique ID should be used once for each user, but when a user leaves the system, the ID can be reused again.

Users can request a certain chunk of the file, the system should return a list of IDs of all the users who own this chunk. If the user receive a non-empty list of IDs, they receive the requested chunk successfully.

Implement the `FileSharing` class:
`FileSharing(int m)` Initializes the object with a file of `m` chunks.

`int join(int[] ownedChunks)`: A new user joined the system owning some chunks of the file, the system should assign an id to the user which is the smallest positive integer not taken by any other user. Return the assigned id.

`void leave(int userID)`: The user with `userID` will leave the system, you cannot take file chunks from them anymore.

`int[] request(int userID, int chunkID)`: The user `userID` requested the file chunk with `chunkID`. Return a list of the IDs of all users that own this chunk sorted in ascending order.

Follow-ups:
What happens if the system identifies the user by their IP address instead of their unique ID and users disconnect and connect from the system with the same IP?
If the users in the system join and leave the system frequently without requesting any chunks, will your solution still be efficient?
If all each user join the system one time, request all files and then leave, will your solution still be efficient?
If the system will be used to share `n` files where the `ith` file consists of `m[i]`, what are the changes you have to do?

Example:
Input:
["FileSharing","join","join","join","request","request","leave","request","leave","join"]
[[4],[[1,2]],[[2,3]],[[4]],[1,3],[2,2],[1],[2,1],[2],[[]]]
Output:
[null,1,2,3,[2],[1,2],null,[],null,1]
Explanation:
FileSharing fileSharing = new FileSharing(4); // We use the system to share a file of 4 chunks.

fileSharing.join([1, 2]);    // A user who has chunks [1,2] joined the system, assign id = 1 to them and return 1.

fileSharing.join([2, 3]);    // A user who has chunks [2,3] joined the system, assign id = 2 to them and return 2.

fileSharing.join([4]);       // A user who has chunk [4] joined the system, assign id = 3 to them and return 3.

fileSharing.request(1, 3);   // The user with id = 1 requested the third file chunk, as only the user with id = 2 has the file, return [2] . Notice that user 1 now has chunks [1,2,3].

fileSharing.request(2, 2);   // The user with id = 2 requested the second file chunk, users with ids [1,2] have this chunk, thus we return [1,2].

fileSharing.leave(1);        // The user with id = 1 left the system, all the file chunks with them are no longer available for other users.

fileSharing.request(2, 1);   // The user with id = 2 requested the first file chunk, no one in the system has this chunk, we return empty list [].

fileSharing.leave(2);        // The user with id = 2 left the system.

fileSharing.join([]);        // A user who doesn't have any chunks joined the system, assign id = 1 to them and return 1. Notice that ids 1 and 2 are free and we can reuse them.


Constraints:
`1 <= m <= 10^5`
`0 <= ownedChunks.length <= min(100, m)`
`1 <= ownedChunks[i] <= m`
Values of `ownedChunks` are unique.

`1 <= chunkID <= m`
`userID` is guaranteed to be a user in the system if you assign the IDs correctly. 
At most `10^4` calls will be made to `join`, `leave` and `request`.

Each call to `leave` will have a matching call for `join`.

### Solution 
 To solve the problem of implementing a file-sharing system with unique user IDs and file chunks, we will create the `FileSharing` class to manage users and their owned chunks. Here's how we will approach the problem:

1. **Initialization**: We'll store the total number of chunks and maintain a list to track which users own which chunks.
2. **User Management**: We'll need a way to assign unique IDs to users when they join, which involves maintaining a list of used IDs and a mechanism to find available IDs.
3. **Chunk Management**: For each chunk ID, we will maintain a list of user IDs that own that chunk.
4. **Requests**: When a user requests a chunk, we'll return a sorted list of all users who own that specific chunk.

Here's the step-by-step implementation:



```python
class FileSharing:
    def __init__(self, m: int):
        self.m = m  # Total number of chunks
        self.chunk_owners = [[] for _ in range(m + 1)]  # List of owners for each chunk
        self.user_chunks = {}  # Maps user IDs to their owned chunks
        self.available_ids = set()  # Set of available user IDs
        self.next_id = 1  # The next user ID to assign

    def join(self, ownedChunks: list) -> int:
        # Assign a user ID
        if self.available_ids:
            user_id = self.available_ids.pop()  # Reuse an ID if available
        else:
            user_id = self.next_id  # Use the next available ID
            self.next_id += 1  # Increment for future use

        # Store chunks for the new user
        self.user_chunks[user_id] = ownedChunks

        # Update chunk owners
        for chunk in ownedChunks:
            self.chunk_owners[chunk].append(user_id)

        return user_id

    def leave(self, userID: int) -> None:
        # Remove user from chunk owners
        ownedChunks = self.user_chunks.pop(userID, [])
        for chunk in ownedChunks:
            self.chunk_owners[chunk].remove(userID)
        
        # Add userID to available IDs for reuse
        self.available_ids.add(userID)

    def request(self, userID: int, chunkID: int) -> list:
        # Return sorted list of user IDs that own the requested chunk
        owners = self.chunk_owners[chunkID]
        owners.sort()  # Sort the owner IDs before returning
        return owners

```

### Explanation of Each Part:
1. **Initialization (`__init__` method)**:
   - We create an empty list for storing owners for `m` chunks.
   - We use a dictionary (`user_chunks`) to map each user ID to their owned chunks.
   - We maintain a set (`available_ids`) to keep track of reused user IDs.
   - We have a `next_id` counter to assign new user IDs when all available IDs are used up.

2. **Joining a User (`join` method)**:
   - When a user joins, we check if there are any IDs available in `available_ids`. If so, we reuse an ID; otherwise, we assign a new ID from `next_id`.
   - We store the owned chunks in `user_chunks` and update the `chunk_owners` for each chunk that the user owns.

3. **Leaving the System (`leave` method)**:
   - When a user leaves, we first retrieve and remove all the chunks they owned from the `user_chunks`.
   - For each chunk, we remove the user ID from the corresponding list of owners in `chunk_owners`.
   - We then add the user ID back into the `available_ids` set for reuse.

4. **Requesting a Chunk (`request` method)**:
   - We fetch the list of owners for the requested `chunkID`.
   - We sort this list to ensure the output is in ascending order and return that.

### Efficiency:
- The solution is efficient because:
  - Joining (`join`) takes O(k) time (k = number of chunks owned) to register the user.
  - Leaving (`leave`) takes O(k) time to remove the user.
  - Requesting (`request`) takes O(p log p) time (p = number of owners of the chunk) for sorting.

The implementation handles the constraints and edge cases specified in the problem statement effectively, ensuring both correctness and performance. This solution can be directly run on LeetCode.

# 1586. Binary Search Tree Iterator II

### Problem Description 
Implement the `BSTIterator` class that represents an iterator over the in-order traversal of a binary search tree (BST):
`BSTIterator(TreeNode root)` Initializes an object of the `BSTIterator` class. The `root` of the BST is given as part of the constructor. The pointer should be initialized to a non-existent number smaller than any element in the BST.

`boolean hasNext()` Returns `true` if there exists a number in the traversal to the right of the pointer, otherwise returns `false`.

`int next()` Moves the pointer to the right, then returns the number at the pointer.

`boolean hasPrev()` Returns `true` if there exists a number in the traversal to the left of the pointer, otherwise returns `false`.

`int prev()` Moves the pointer to the left, then returns the number at the pointer.

Notice that by initializing the pointer to a non-existent smallest number, the first call to `next()` will return the smallest element in the BST.

You may assume that `next()` and `prev()` calls will always be valid. That is, there will be at least a next/previous number in the in-order traversal when `next()`/`prev()` is called.


Example 1:
Input
["BSTIterator", "next", "next", "prev", "next", "hasNext", "next", "next", "next", "hasNext", "hasPrev", "prev", "prev"]
[[[7, 3, 15, null, null, 9, 20]], [null], [null], [null], [null], [null], [null], [null], [null], [null], [null], [null], [null]]
Output
[null, 3, 7, 3, 7, true, 9, 15, 20, false, true, 15, 9]
Explanation
// The underlined element is where the pointer currently is.

BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]); // state is   [3, 7, 9, 15, 20]
bSTIterator.next(); // state becomes [3, 7, 9, 15, 20], return 3
bSTIterator.next(); // state becomes [3, 7, 9, 15, 20], return 7
bSTIterator.prev(); // state becomes [3, 7, 9, 15, 20], return 3
bSTIterator.next(); // state becomes [3, 7, 9, 15, 20], return 7
bSTIterator.hasNext(); // return true
bSTIterator.next(); // state becomes [3, 7, 9, 15, 20], return 9
bSTIterator.next(); // state becomes [3, 7, 9, 15, 20], return 15
bSTIterator.next(); // state becomes [3, 7, 9, 15, 20], return 20
bSTIterator.hasNext(); // return false
bSTIterator.hasPrev(); // return true
bSTIterator.prev(); // state becomes [3, 7, 9, 15, 20], return 15
bSTIterator.prev(); // state becomes [3, 7, 9, 15, 20], return 9

Constraints:
The number of nodes in the tree is in the range `[1, 105]`.

`0 <= Node.val <= 106`
At most `105` calls will be made to `hasNext`, `next`, `hasPrev`, and `prev`.

Follow up: Could you solve the problem without precalculating the values of the tree?

### Solution 
 To solve the problem of implementing the `BSTIterator`, we will create a class that allows us to traverse a binary search tree (BST) in an in-order fashion. The iterator will provide the following functionalities:

1. `hasNext()` - to check if there are more elements to iterate over.
2. `next()` - to retrieve the next element in the in-order traversal.
3. `hasPrev()` - to check if there are previous elements to iterate over.
4. `prev()` - to retrieve the previous element in the in-order traversal.

Here we will utilize a stack to keep track of the nodes we need to traverse during the in-order traversal. We will also maintain a `current` index to keep track of our position in the iteration.

The approach can be outlined in the following steps:

1. Initialize the `BSTIterator` with the root of the BST.
2. Use an in-order traversal to populate the stack when moving to the leftmost node.
3. Implement `next()` to pop from the stack and push the left children of the popped node to the stack.
4. Implement `prev()` which will be a reverse operation, effectively keeping track of previously returned nodes.
5. `hasNext()` and `hasPrev()` will check the conditions based on the current state of our stack or current index.

Here's the implementation:



```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BSTIterator:
    def __init__(self, root: TreeNode):
        self.stack = []
        self.current = None
        
        # Initialize the stack with the leftmost nodes
        self._push_left(root)

    def _push_left(self, node):
        while node:
            self.stack.append(node)
            node = node.left
            
    def hasNext(self) -> bool:
        # Since next() will pop from the stack, we just need to check if the stack is not empty
        return len(self.stack) > 0

    def next(self) -> int:
        # Get the next smallest element
        if not self.hasNext():
            return None  # This return won't normally happen because of precondition

        self.current = self.stack.pop()
        result = self.current.val
        
        # If the current node has a right child, push its leftmost children onto the stack
        if self.current.right:
            self._push_left(self.current.right)
        
        return result
    
    def hasPrev(self) -> bool:
        # We check if current has a defined value and if our position allows moving back
        return self.current is not None and (self.stack or self.current.left)

    def prev(self) -> int:
        # Return the previous element in the in-order traversal
        if not self.hasPrev():
            return None  # This return won't normally happen because of precondition
            
        # To get the previous value, we will traverse back through the stack
        if self.current:
            left_node = self.current.left
            if left_node:
                # Get the rightmost node of the left child to maintain in-order properties
                while left_node.right:
                    left_node = left_node.right
                self.stack.append(self.current)  # Push current before moving to left
                self.current = left_node
                
            else:
                # Current node is the first node in the in-order, we need to pop from the stack
                if self.stack:
                    self.current = self.stack.pop()
                else:
                    self.current = None  # No previous element

        return self.current.val if self.current else None

# The following defines the BST corresponding to the example input
root = TreeNode(7)
root.left = TreeNode(3)
root.right = TreeNode(15)
root.right.left = TreeNode(9)
root.right.right = TreeNode(20)

# Initialize the iterator
iterator = BSTIterator(root)

# You can now call next(), hasNext(), prev(), hasPrev()

```

### Explanation of the Code:
1. **TreeNode Class**: Defines the structure of a node in the BST.
2. **BSTIterator Class**:
   - `__init__`: Initializes the iterator and populates the stack with the leftmost nodes of the BST starting from the root.
   - `_push_left(node)`: Helper function to push all left children of a given node to the stack.
   - `hasNext()`: Checks if there are more nodes to iterate by checking if the stack is not empty.
   - `next()`: Pops from the stack, retrieves the current node's value, and pushes the leftmost nodes of its right child (if available).
   - `hasPrev()`: Checks if there are nodes that can be accessed before the current node.
   - `prev()`: Retrieves the previous node while maintaining the order.

This implementation ensures that we traverse the BST in an efficient manner without needing to store all values in an array or list, meeting the requirements of the problem.

# 1600. Throne Inheritance

### Problem Description 
A kingdom consists of a king, his children, his grandchildren, and so on. Every once in a while, someone in the family dies or a child is born.

The kingdom has a well-defined order of inheritance that consists of the king as the first member. Let's define the recursive function `Successor(x, curOrder)`, which given a person `x` and the inheritance order so far, returns who should be the next person after `x` in the order of inheritance.

Successor(x, curOrder):
    if x has no children or all of x's children are in curOrder:
        if x is the king return null
        else return Successor(x's parent, curOrder)
    else return x's oldest child who's not in curOrder
For example, assume we have a kingdom that consists of the king, his children Alice and Bob (Alice is older than Bob), and finally Alice's son Jack.

In the beginning, `curOrder` will be `["king"]`.

Calling `Successor(king, curOrder)` will return Alice, so we append to `curOrder` to get `["king", "Alice"]`.

Calling `Successor(Alice, curOrder)` will return Jack, so we append to `curOrder` to get `["king", "Alice", "Jack"]`.

Calling `Successor(Jack, curOrder)` will return Bob, so we append to `curOrder` to get `["king", "Alice", "Jack", "Bob"]`.

Calling `Successor(Bob, curOrder)` will return `null`. Thus the order of inheritance will be `["king", "Alice", "Jack", "Bob"]`.

Using the above function, we can always obtain a unique order of inheritance.

Implement the `ThroneInheritance` class:
`ThroneInheritance(string kingName)` Initializes an object of the `ThroneInheritance` class. The name of the king is given as part of the constructor.

`void birth(string parentName, string childName)` Indicates that `parentName` gave birth to `childName`.

`void death(string name)` Indicates the death of `name`. The death of the person doesn't affect the `Successor` function nor the current inheritance order. You can treat it as just marking the person as dead.

`string[] getInheritanceOrder()` Returns a list representing the current order of inheritance excluding dead people.


Example 1:
Input
["ThroneInheritance", "birth", "birth", "birth", "birth", "birth", "birth", "getInheritanceOrder", "death", "getInheritanceOrder"]
[["king"], ["king", "andy"], ["king", "bob"], ["king", "catherine"], ["andy", "matthew"], ["bob", "alex"], ["bob", "asha"], [null], ["bob"], [null]]
Output
[null, null, null, null, null, null, null, ["king", "andy", "matthew", "bob", "alex", "asha", "catherine"], null, ["king", "andy", "matthew", "alex", "asha", "catherine"]]
Explanation
ThroneInheritance t= new ThroneInheritance("king"); // order: king
t.birth("king", "andy"); // order: king > andy
t.birth("king", "bob"); // order: king > andy > bob
t.birth("king", "catherine"); // order: king > andy > bob > catherine
t.birth("andy", "matthew"); // order: king > andy > matthew > bob > catherine
t.birth("bob", "alex"); // order: king > andy > matthew > bob > alex > catherine
t.birth("bob", "asha"); // order: king > andy > matthew > bob > alex > asha > catherine
t.getInheritanceOrder(); // return ["king", "andy", "matthew", "bob", "alex", "asha", "catherine"]
t.death("bob"); // order: king > andy > matthew > bob > alex > asha > catherine
t.getInheritanceOrder(); // return ["king", "andy", "matthew", "alex", "asha", "catherine"]

Constraints:
`1 <= kingName.length, parentName.length, childName.length, name.length <= 15`
`kingName`, `parentName`, `childName`, and `name` consist of lowercase English letters only.

All arguments `childName` and `kingName` are distinct.

All `name` arguments of `death` will be passed to either the constructor or as `childName` to `birth` first.

For each call to `birth(parentName, childName)`, it is guaranteed that `parentName` is alive.

At most `105` calls will be made to `birth` and `death`.

At most `10` calls will be made to `getInheritanceOrder`.

### Solution 
 To solve the problem of inheritance in a kingdom, we'll implement a `ThroneInheritance` class that manages family relationships and tracks the order of inheritance. Let's break down the requirements and implement the solution step by step.

### Class Structure

1. **Attributes**:
   - A dictionary to keep track of each person's children.
   - A set to track deceased members.
   - A string to represent the king's name.

2. **Methods**:
   - `__init__`: Initializes the throne with the king's name.
   - `birth`: Adds a child to a specific parent.
   - `death`: Marks a person as deceased.
   - `getInheritanceOrder`: Returns the current order of inheritance excluding deceased members.

### Algorithm to Get Inheritance Order
The method `getInheritanceOrder` will use a depth-first search (DFS) to traverse the family tree and build the inheritance order. We will skip any deceased members during this traversal.

### Implementation Steps
1. Create a class `ThroneInheritance`.
2. Use a dictionary to store the relationships (children of each person).
3. Use a set to mark deceased people.
4. Implement the required methods according to the provided rules.

### Implementation Code
Here is the complete implementation of the `ThroneInheritance` class:



```python
class ThroneInheritance:
    def __init__(self, kingName: str):
        self.king = kingName
        self.children = {}  # Maps parent name to a list of children
        self.dead = set()    # Set of deceased members

    def birth(self, parentName: str, childName: str) -> None:
        if parentName not in self.children:
            self.children[parentName] = []
        self.children[parentName].append(childName)

    def death(self, name: str) -> None:
        self.dead.add(name)

    def getInheritanceOrder(self) -> list[str]:
        order = []
        self._dfs(self.king, order)
        return order

    def _dfs(self, person: str, order: list[str]) -> None:
        # If the person is not dead, include them in the order
        if person not in self.dead:
            order.append(person)
        
        # Recur for all children
        if person in self.children:
            for child in self.children[person]:
                self._dfs(child, order)

# Example usage:
# t = ThroneInheritance("king")
# t.birth("king", "andy")
# t.birth("king", "bob")
# t.birth("king", "catherine")
# t.birth("andy", "matthew")
# t.birth("bob", "alex")
# t.birth("bob", "asha")
# print(t.getInheritanceOrder())  # Output: ["king", "andy", "matthew", "bob", "alex", "asha", "catherine"]
# t.death("bob")
# print(t.getInheritanceOrder())  # Output: ["king", "andy", "matthew", "alex", "asha", "catherine"]

```

### Explanation
1. **Initialization**:
   - The constructor initializes the king's name and sets up an empty dictionary for children and a set for dead members.

2. **Birth**:
   - The `birth` method adds a child to the parent's list in the `children` dictionary.

3. **Death**:
   - The `death` method marks an individual as dead by adding them to the `dead` set.

4. **Get Inheritance Order**:
   - The `getInheritanceOrder` method starts from the king and performs a depth-first search (DFS) to retrieve the order of living members. It uses the helper method `_dfs`, which appends members to `order` if they're alive and calls itself recursively for each child of the current member.

This implementation efficiently constructs the order of inheritance while adhering to the problem's constraints and requirements.

# 1603. Design Parking System

### Problem Description 
Design a parking system for a parking lot. The parking lot has three kinds of parking spaces: big, medium, and small, with a fixed number of slots for each size.

Implement the `ParkingSystem` class:
`ParkingSystem(int big, int medium, int small)` Initializes object of the `ParkingSystem` class. The number of slots for each parking space are given as part of the constructor.

`bool addCar(int carType)` Checks whether there is a parking space of `carType` for the car that wants to get into the parking lot. `carType` can be of three kinds: big, medium, or small, which are represented by `1`, `2`, and `3` respectively. A car can only park in a parking space of its `carType`. If there is no space available, return `false`, else park the car in that size space and return `true`.


Example 1:
Input
["ParkingSystem", "addCar", "addCar", "addCar", "addCar"]
[[1, 1, 0], [1], [2], [3], [1]]
Output
[null, true, true, false, false]
Explanation
ParkingSystem parkingSystem = new ParkingSystem(1, 1, 0);
parkingSystem.addCar(1); // return true because there is 1 available slot for a big car
parkingSystem.addCar(2); // return true because there is 1 available slot for a medium car
parkingSystem.addCar(3); // return false because there is no available slot for a small car
parkingSystem.addCar(1); // return false because there is no available slot for a big car. It is already occupied.


Constraints:
`0 <= big, medium, small <= 1000`
`carType` is `1`, `2`, or `3`
At most `1000` calls will be made to `addCar`

### Solution 
 Certainly! Let's implement the `ParkingSystem` class as described in the problem statement.

### Problem Explanation

We need to design a parking system that has:
- Three types of parking spaces: big, medium, and small.
- Each type has a certain number of available slots.
- The `addCar` method is used to attempt to park a car of a specific type. If the slot for the car type is available, the car will park, and we will return `true`. If there are no slots available for that car type, we return `false`.

### Class Structure

1. **Initialization**: The constructor will take three integers as parameters, which represent the number of available slots for big, medium, and small cars.
2. **Adding Cars**: The `addCar` method will take an integer representing the car type and will check for the availability of slots within that type. If available, it will decrement the relevant slot count.

### Implementation

Let's implement the `ParkingSystem` class following the requirements.



```python
class ParkingSystem:
    def __init__(self, big: int, medium: int, small: int):
        # Initialize available slots for each car size
        self.available_slots = {
            1: big,    # Slots for big cars
            2: medium, # Slots for medium cars
            3: small   # Slots for small cars
        }

    def addCar(self, carType: int) -> bool:
        # Check if there is available space for the given car type
        if self.available_slots[carType] > 0:
            self.available_slots[carType] -= 1  # Park the car by reducing the slot count
            return True
        return False  # No space available for that car type

# Example Usage
parkingSystem = ParkingSystem(1, 1, 0)
print(parkingSystem.addCar(1))  # Should return True
print(parkingSystem.addCar(2))  # Should return True
print(parkingSystem.addCar(3))  # Should return False
print(parkingSystem.addCar(1))  # Should return False

```

### Explanation of the Code

1. **Constructor (`__init__`)**:
   - Takes three parameters: `big`, `medium`, and `small`, which represent the number of slots for each type.
   - Initializes a dictionary `available_slots` to keep track of available parking spaces for each car type.

2. **Method `addCar`**:
   - Takes a parameter `carType` which indicates which car type is trying to park.
   - Checks if there are available slots for the given `carType`.
   - If a slot is available, it decrements the count of available slots for that type and returns `True`, indicating the car has successfully parked.
   - If no slots are available, it returns `False`.

### Running the Code

The provided code can be copied and pasted directly into LeetCode for testing, making sure to use the input format as demonstrated in the example usage section. The expected output matches the problem specification. This implementation efficiently handles the requirement with a straightforward dictionary to manage available parking slots.

# 1622. Fancy Sequence

### Problem Description 
Write an API that generates fancy sequences using the `append`, `addAll`, and `multAll` operations.

Implement the `Fancy` class:
`Fancy()` Initializes the object with an empty sequence.

`void append(val)` Appends an integer `val` to the end of the sequence.

`void addAll(inc)` Increments all existing values in the sequence by an integer `inc`.

`void multAll(m)` Multiplies all existing values in the sequence by an integer `m`.

`int getIndex(idx)` Gets the current value at index `idx` (0-indexed) of the sequence modulo `109 + 7`. If the index is greater or equal than the length of the sequence, return `-1`.


Example 1:
Input
["Fancy", "append", "addAll", "append", "multAll", "getIndex", "addAll", "append", "multAll", "getIndex", "getIndex", "getIndex"]
[[], [2], [3], [7], [2], [0], [3], [10], [2], [0], [1], [2]]
Output
[null, null, null, null, null, 10, null, null, null, 26, 34, 20]
Explanation
Fancy fancy = new Fancy();
fancy.append(2);   // fancy sequence: [2]
fancy.addAll(3);   // fancy sequence: [2+3] -> [5]
fancy.append(7);   // fancy sequence: [5, 7]
fancy.multAll(2);  // fancy sequence: [5*2, 7*2] -> [10, 14]
fancy.getIndex(0); // return 10
fancy.addAll(3);   // fancy sequence: [10+3, 14+3] -> [13, 17]
fancy.append(10);  // fancy sequence: [13, 17, 10]
fancy.multAll(2);  // fancy sequence: [13*2, 17*2, 10*2] -> [26, 34, 20]
fancy.getIndex(0); // return 26
fancy.getIndex(1); // return 34
fancy.getIndex(2); // return 20

Constraints:
`1 <= val, inc, m <= 100`
`0 <= idx <= 105`
At most `105` calls total will be made to `append`, `addAll`, `multAll`, and `getIndex`.

### Solution 
 To implement the `Fancy` class that can generate fancy sequences through various operations, we need to maintain a dynamic sequence of integers, along with efficient mechanisms for modifying all elements through addition and multiplication. 

Here's a breakdown of our approach:

1. **Data Structure**: We will maintain an array `sequence` to store the integers. Additionally, we will keep track of global increment (`add`) and multiplier (`mult`) values that apply to all elements in the sequence. This will help apply operations efficiently without needing to traverse the entire list each time.

2. **Operations**:
   - **Append**: When appending a value, we need to consider the current `add` and `mult` values. The actual value appended to the `sequence` will be adjusted based on these global values.
   
   - **Add All**: Instead of iterating through the list and updating each element, we just increment the `add` variable. This way, whenever we get values, the global increment will be applied automatically.
   
   - **Multiply All**: This operation is a bit more complex because it affects the existing `mul` and `add`. When we multiply all elements by `m`, we should update the `mult` factor and also adjust the `add` factor (i.e., `add` needs to be multiplied by `m`).
   
   - **Get Index**: To get the value at a specific index, we retrieve the value from the `sequence`, apply the previous `add` and `mult` adjustments, and take the result modulo \(10^9 + 7\).

Here is the implementation of the `Fancy` class following this strategy:



```python
class Fancy:
    
    def __init__(self):
        self.sequence = []
        self.add = 0
        self.mult = 1
        self.MOD = 10**9 + 7
    
    def append(self, val: int) -> None:
        # Store the adjusted number considering current add and mult
        adjusted_val = (val - self.add) * pow(self.mult, self.MOD - 2, self.MOD) % self.MOD
        self.sequence.append(adjusted_val)

    def addAll(self, inc: int) -> None:
        self.add = (self.add + inc) % self.MOD

    def multAll(self, m: int) -> None:
        self.add = (self.add * m) % self.MOD
        self.mult = (self.mult * m) % self.MOD

    def getIndex(self, idx: int) -> int:
        if idx < len(self.sequence):
            # Calculate the real value at index idx
            value = (self.sequence[idx] * self.mult + self.add) % self.MOD
            return value
        return -1

```

### Explanation of Each Method:

- **Constructor `__init__`**:
  - Initializes `sequence` as an empty list.
  - Sets `add` and `mult` to apply subsequent operations.
  - Defines a modulo constant `MOD`.

- **`append(val: int)`**:
  - Calculates the adjusted value to append by reversing the effect of the current `add` and `mult`.
  - Uses modular multiplicative inverse to handle division in the modulo field appropriately.

- **`addAll(inc: int)`**:
  - Increases the global `add` value. Since we only need to know the total increment rather than the individual elements, we keep updating this value.

- **`multAll(m: int)`**:
  - Updates both `add` and `mult`, ensuring that future values account for this multiplication.

- **`getIndex(idx: int)`**:
  - Checks if the provided index is valid. If it is, computes the actual value using the stored value in `sequence`, adjusting with `mult` and `add` operations before returning.

### Time Complexity:
- All operations (`append`, `addAll`, `multAll`, `getIndex`) run in O(1) time, making the class highly efficient for the given constraints.

### Usage:
This class can be instantiated and used to manage fancy sequences perfectly aligned with the problem constraints and requirements specified.

# 1628. Design an Expression Tree With Evaluate Function

### Problem Description 
Given the `postfix` tokens of an arithmetic expression, build and return the binary expression tree that represents this expression.

Postfix notation is a notation for writing arithmetic expressions in which the operands (numbers) appear before their operators. For example, the postfix tokens of the expression `4*(5-(7+2))` are represented in the array `postfix = ["4","5","7","2","+","-","*"]`.

The class `Node` is an interface you should use to implement the binary expression tree. The returned tree will be tested using the `evaluate` function, which is supposed to evaluate the tree's value. You should not remove the `Node` class; however, you can modify it as you wish, and you can define other classes to implement it if needed.

A binary expression tree is a kind of binary tree used to represent arithmetic expressions. Each node of a binary expression tree has either zero or two children. Leaf nodes (nodes with 0 children) correspond to operands (numbers), and internal nodes (nodes with two children) correspond to the operators `'+'` (addition), `'-'` (subtraction), `'*'` (multiplication), and `'/'` (division).

It's guaranteed that no subtree will yield a value that exceeds `109` in absolute value, and all the operations are valid (i.e., no division by zero).

Follow up: Could you design the expression tree such that it is more modular? For example, is your design able to support additional operators without making changes to your existing `evaluate` implementation?

Example 1:
Input: s = ["3","4","+","2","*","7","/"]
Output: 2
Explanation: this expression evaluates to the above binary tree with expression (`(3+4)*2)/7) = 14/7 = 2.`

Example 2:
Input: s = ["4","5","7","2","+","-","*"]
Output: -16
Explanation: this expression evaluates to the above binary tree with expression 4*(5-`(2+7)) = 4*(-4) = -16.`

Example 3:
Input: s = ["4","2","+","3","5","1","-","*","+"]
Output: 18

Example 4:
Input: s = ["100","200","+","2","/","5","*","7","+"]
Output: 757

Constraints:
`1 <= s.length < 100`
`s.length` is odd.

`s` consists of numbers and the characters `'+'`, `'-'`, `'*'`, and `'/'`.

If `s[i]` is a number, its integer representation is no more than `105`.

It is guaranteed that `s` is a valid expression.

The absolute value of the result and intermediate values will not exceed `109`.

It is guaranteed that no expression will include division by zero.

### Solution 
 To solve the problem of constructing a binary expression tree from a given postfix notation, we will follow these steps:

1. **Understand the Structure:** We need to define a `Node` class representing each node in the binary tree. Each node will hold a value and pointers to its left and right children.

2. **Postfix Evaluation Logic:** In postfix notation, each operator follows its operands. We will utilize a stack to help us build the tree:
   - For each token in the postfix expression:
     - If the token is a number, create a new node and push it onto the stack.
     - If the token is an operator, pop the top two nodes from the stack to serve as the right and left children for a new node representing the operator, then push this new node back onto the stack.

3. **Evaluating the Expression Tree:** After constructing the tree, we will provide an `evaluate` function that recursively computes the value of the expression represented by the tree.

4. **Modularity for Additional Operators:** The `evaluate` function can be designed to easily accommodate new operators by using a dictionary mapping operators to their corresponding Python functions.

Below is the Python implementation based on the explained approach:



```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class ExpressionTree:
    def build(self, postfix):
        stack = []
        
        for token in postfix:
            if token.isdigit() or (token[0] == '-' and len(token) > 1):  # check for negative numbers
                node = Node(int(token))
                stack.append(node)
            else:
                right = stack.pop()
                left = stack.pop()
                node = Node(token)
                node.left = left
                node.right = right
                stack.append(node)

        return stack.pop()  # The root of the expression tree

    def evaluate(self, root):
        # Define operators with corresponding functions
        def eval_op(left, right, operator):
            if operator == '+':
                return left + right
            elif operator == '-':
                return left - right
            elif operator == '*':
                return left * right
            elif operator == '/':
                return left // right  # Integer division
            
        if root is None:
            return 0
        
        if root.left is None and root.right is None:  # Leaf node
            return root.value
        
        left_value = self.evaluate(root.left)
        right_value = self.evaluate(root.right)
        return eval_op(left_value, right_value, root.value)

# Sample usage
def evaluate_expression(postfix):
    expression_tree = ExpressionTree()
    root = expression_tree.build(postfix)
    return expression_tree.evaluate(root)

# Example inputs
print(evaluate_expression(["3", "4", "+", "2", "*", "7", "/"]))  # Output: 2
print(evaluate_expression(["4", "5", "7", "2", "+", "-", "*"]))  # Output: -16
print(evaluate_expression(["4", "2", "+", "3", "5", "1", "-", "*", "+"]))  # Output: 18
print(evaluate_expression(["100", "200", "+", "2", "/", "5", "*", "7", "+"]))  # Output: 757

```

### Explanation of the Code:

- **Node Class:** This class creates nodes that store the value of the operand/operator and references to left and right child nodes.
  
- **ExpressionTree Class:** 
  - **build(postfix):** This method constructs the binary expression tree from the postfix expression using a stack. When an operator is encountered, we pop the last two operands from the stack, create a new node with the operator, and set the popped nodes as its children.
  
  - **evaluate(root):** This method computes the value of the expression tree. For each node, it recursively evaluates the left and right subtrees and applies the operation defined by the node’s value.

- **Modularity:** The evaluation of operations can be extended by simply modifying the `eval_op` function, allowing easy addition of new operators.

This implementation aligns with the LeetCode format and can be directly run on the website, following their provided testing procedures.

# 1656. Design an Ordered Stream

### Problem Description 
There is a stream of `n` `(idKey, value)` pairs arriving in an arbitrary order, where `idKey` is an integer between `1` and `n` and `value` is a string. No two pairs have the same `id`.

Design a stream that returns the values in increasing order of their IDs by returning a chunk (list) of values after each insertion. The concatenation of all the chunks should result in a list of the sorted values.

Implement the `OrderedStream` class:
`OrderedStream(int n)` Constructs the stream to take `n` values.

`String[] insert(int idKey, String value)` Inserts the pair `(idKey, value)` into the stream, then returns the largest possible chunk of currently inserted values that appear next in the order.


Example:
Input
["OrderedStream", "insert", "insert", "insert", "insert", "insert"]
[[5], [3, "ccccc"], [1, "aaaaa"], [2, "bbbbb"], [5, "eeeee"], [4, "ddddd"]]
Output
[null, [], ["aaaaa"], ["bbbbb", "ccccc"], [], ["ddddd", "eeeee"]]
Explanation
// Note that the values ordered by ID is ["aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee"].

OrderedStream os = new OrderedStream(5);
os.insert(3, "ccccc"); // Inserts (3, "ccccc"), returns [].

os.insert(1, "aaaaa"); // Inserts (1, "aaaaa"), returns ["aaaaa"].

os.insert(2, "bbbbb"); // Inserts (2, "bbbbb"), returns ["bbbbb", "ccccc"].

os.insert(5, "eeeee"); // Inserts (5, "eeeee"), returns [].

os.insert(4, "ddddd"); // Inserts (4, "ddddd"), returns ["ddddd", "eeeee"].

// Concatentating all the chunks returned:
// [] + ["aaaaa"] + ["bbbbb", "ccccc"] + [] + ["ddddd", "eeeee"] = ["aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee"]
// The resulting order is the same as the order above.


Constraints:
`1 <= n <= 1000`
`1 <= id <= n`
`value.length == 5`
`value` consists only of lowercase letters.

Each call to `insert` will have a unique `id.`
Exactly `n` calls will be made to `insert`.

### Solution 
 To solve the problem of creating an `OrderedStream` that allows for the insertion of `(idKey, value)` pairs and returns them in order by `idKey`, we can utilize an internal list to store the values where the index corresponds to the `idKey`. This will allow us to efficiently manage the order of the inserted elements and output the required chunks.

### Implementation Steps:

1. **Initialization (`__init__` method)**:
   - Create a list of size `n + 1` (to accommodate 1-based indexing for `idKey`).
   - Use a pointer starting at `1` to keep track of the smallest unfilled `idKey`.

2. **Insertion (`insert` method)**:
   - Insert the value at the index corresponding to the `idKey` in the list.
   - Using the pointer, check for consistent chunks of inserted values starting from the pointer index.
   - When consecutive values are found, gather them into a result list until there is a gap (i.e., an index where no value has been inserted).

3. **Returning Chunks**:
   - The method returns all the contiguous values from the pointer index until a gap occurs in the insertion.

Here's the complete code that follows the LeetCode format and is ready to run:



```python
class OrderedStream:
    def __init__(self, n: int):
        # Initialize the internal storage and pointer
        self.stream = [None] * (n + 1)  # use index from 1 to n
        self.ptr = 1  # Initialize the pointer to the first idKey

    def insert(self, idKey: int, value: str):
        # Place the value at the designated idKey
        self.stream[idKey] = value
        result = []
        
        # Check for continuous chunks starting from the pointer
        while self.ptr < len(self.stream) and self.stream[self.ptr] is not None:
            result.append(self.stream[self.ptr])
            self.ptr += 1  # Move the pointer to the next idKey
        
        return result

# Example of how this could be used:
# os = OrderedStream(5)
# print(os.insert(3, "ccccc"))  # Returns []
# print(os.insert(1, "aaaaa"))  # Returns ["aaaaa"]
# print(os.insert(2, "bbbbb"))  # Returns ["bbbbb", "ccccc"]
# print(os.insert(5, "eeeee"))  # Returns []
# print(os.insert(4, "ddddd"))  # Returns ["ddddd", "eeeee"]

```

### Explanation of Code:

1. **Initialization**:
   

```python
   self.stream = [None] * (n + 1)
   
```
   - This creates a list where each index corresponds to an `idKey`, initialized with `None` to indicate that no value has been inserted yet.

2. **Insert Method**:
   - When an `(idKey, value)` is inserted, it gets stored in the list:
   

```python
   self.stream[idKey] = value
   
```
   - The method then checks if there are any sequential values starting from the current pointer position:
   

```python
   while self.ptr < len(self.stream) and self.stream[self.ptr] is not None:
       result.append(self.stream[self.ptr])
       self.ptr += 1
   
```
   - This will keep adding values to `result` until a `None` is encountered, signifying a gap.

3. **Return Value**:
   - The method returns the chunks of values that could be output at the moment of insertion, allowing sequential retrieval of data.

This design is efficient with an average time complexity of O(1) for insertion, making it suitable even for the upper limits of the input constraints.

# 1670. Design Front Middle Back Queue

### Problem Description 
Design a queue that supports `push` and `pop` operations in the front, middle, and back.

Implement the `FrontMiddleBack` class:
`FrontMiddleBack()` Initializes the queue.

`void pushFront(int val)` Adds `val` to the front of the queue.

`void pushMiddle(int val)` Adds `val` to the middle of the queue.

`void pushBack(int val)` Adds `val` to the back of the queue.

`int popFront()` Removes the front element of the queue and returns it. If the queue is empty, return `-1`.

`int popMiddle()` Removes the middle element of the queue and returns it. If the queue is empty, return `-1`.

`int popBack()` Removes the back element of the queue and returns it. If the queue is empty, return `-1`.

Notice that when there are two middle position choices, the operation is performed on the frontmost middle position choice. For example:
Pushing `6` into the middle of `[1, 2, 3, 4, 5]` results in `[1, 2, 6, 3, 4, 5]`.

Popping the middle from `[1, 2, 3, 4, 5, 6]` returns `3` and results in `[1, 2, 4, 5, 6]`.


Example 1:
Input:
["FrontMiddleBackQueue", "pushFront", "pushBack", "pushMiddle", "pushMiddle", "popFront", "popMiddle", "popMiddle", "popBack", "popFront"]
[[], [1], [2], [3], [4], [], [], [], [], []]
Output:
[null, null, null, null, null, 1, 3, 4, 2, -1]
Explanation:
FrontMiddleBackQueue q = new FrontMiddleBackQueue();
q.pushFront(1);   // [1]
q.pushBack(2);    // [1, 2]
q.pushMiddle(3);  // [1, 3, 2]
q.pushMiddle(4);  // [1, 4, 3, 2]
q.popFront();     // return 1 -> [4, 3, 2]
q.popMiddle();    // return 3 -> [4, 2]
q.popMiddle();    // return 4 -> [2]
q.popBack();      // return 2 -> []
q.popFront();     // return -1 -> [] (The queue is empty)

Constraints:
`1 <= val <= 109`
At most `1000` calls will be made to `pushFront`, `pushMiddle`, `pushBack`, `popFront`, `popMiddle`, and `popBack`.

### Solution 
 To solve the problem of designing a queue that supports pushing and popping values from the front, middle, and back, we need to handle the operations efficiently. We can represent the queue using a list and maintain the order of elements while allowing for easy access to the front, middle, and back elements. 

### Detailed Explanation:

1. **Data Structure**: We will use a list to maintain our queue. The key challenge is finding the middle element efficiently, especially since the definition of "middle" needs to allow for two possible choices when the queue has an even number of elements.

2. **Operations**:
   - `pushFront(val)`: This operation adds `val` to the front of the queue. We can use the `insert` method of the list at index 0.
   - `pushMiddle(val)`: To determine where the middle is, we can compute the index as `len(self.queue) // 2`. We insert `val` at this index.
   - `pushBack(val)`: This operation appends `val` to the end of the list.
   - `popFront()`: We can remove the first element of the list using `pop(0)`, which gives us the front of the queue.
   - `popMiddle()`: To pop the middle element, we calculate the middle index. If the current length is odd, it would be `len(self.queue) // 2`. If even, we pop from the same index, effectively choosing the frontmost middle element.
   - `popBack()`: We simply remove the last element using `pop()`.

3. **Edge Cases**: Whenever we attempt to pop from the queue, we first check if the queue is empty and return `-1` if it is.

Here is the implementation in Python:



```python
class FrontMiddleBackQueue:

    def __init__(self):
        self.queue = []

    def pushFront(self, val: int) -> None:
        self.queue.insert(0, val)

    def pushMiddle(self, val: int) -> None:
        middle_index = len(self.queue) // 2
        self.queue.insert(middle_index, val)

    def pushBack(self, val: int) -> None:
        self.queue.append(val)

    def popFront(self) -> int:
        if not self.queue:
            return -1
        return self.queue.pop(0)

    def popMiddle(self) -> int:
        if not self.queue:
            return -1
        middle_index = len(self.queue) // 2
        return self.queue.pop(middle_index)

    def popBack(self) -> int:
        if not self.queue:
            return -1
        return self.queue.pop()

```

### Example Usage:

To use the `FrontMiddleBackQueue` class, you would create an instance and call the methods as specified in the problem statement. Here’s how it looks with the example provided in the prompt:



```python
# Example of using FrontMiddleBackQueue
q = FrontMiddleBackQueue()
q.pushFront(1)   # Queue is now [1]
q.pushBack(2)    # Queue is now [1, 2]
q.pushMiddle(3)  # Queue is now [1, 3, 2]
q.pushMiddle(4)  # Queue is now [1, 4, 3, 2]
print(q.popFront())     # Returns 1, Queue is now [4, 3, 2]
print(q.popMiddle())    # Returns 3, Queue is now [4, 2]
print(q.popMiddle())    # Returns 4, Queue is now [2]
print(q.popBack())      # Returns 2, Queue is now []
print(q.popFront())     # Returns -1, Queue is now []

```

This implementation effectively handles the required operations while maintaining the queue's properties according to the problem statement. The use of list allows direct access to indices for inserting and removing elements, ensuring we meet the problem's constraints.

# 1756. Design Most Recently Used Queue

### Problem Description 
Design a queue-like data structure that moves the most recently used element to the end of the queue.

Implement the `MRUQueue` class:
`MRUQueue(int n)` constructs the `MRUQueue` with `n` elements: `[1,2,3,...,n]`.

`fetch(int k)` moves the `kth` element (1-indexed) to the end of the queue and returns it.


Example 1:
Input:
["MRUQueue", "fetch", "fetch", "fetch", "fetch"]
[[8], [3], [5], [2], [8]]
Output:
[null, 3, 6, 2, 2]
Explanation:
MRUQueue mRUQueue = new MRUQueue(8); // Initializes the queue to [1,2,3,4,5,6,7,8].

mRUQueue.fetch(3); // Moves the 3rd element (3) to the end of the queue to become [1,2,4,5,6,7,8,3] and returns it.

mRUQueue.fetch(5); // Moves the 5th element (6) to the end of the queue to become [1,2,4,5,7,8,3,6] and returns it.

mRUQueue.fetch(2); // Moves the 2nd element (2) to the end of the queue to become [1,4,5,7,8,3,6,2] and returns it.

mRUQueue.fetch(8); // The 8th element (2) is already at the end of the queue so just return it.


Constraints:
`1 <= n <= 2000`
`1 <= k <= n`
At most `2000` calls will be made to `fetch`.

Follow up: Finding an `O(n)` algorithm per `fetch` is a bit easy. Can you find an algorithm with a better complexity for each `fetch` call?

### Solution 
 To solve the problem of creating an MRU (Most Recently Used) Queue where fetching an element also requires moving that element to the end of the queue, we can utilize a combination of a list to maintain the current order of elements and a dictionary for quick access to the elements by their index. This will allow us to achieve better efficiency when performing the fetch operation.

Here’s how we can implement the `MRUQueue` class:



```python
class MRUQueue:
    def __init__(self, n: int):
        # Initialize the queue with elements from 1 to n
        self.queue = list(range(1, n + 1))
        self.index_map = {i + 1: i for i in range(n)}  # Mapping of number to index in the list for quick access

    def fetch(self, k: int) -> int:
        # Get the element to fetch
        element = self.queue[self.index_map[k]]
        
        # The index of the element we want to move to the end
        index = self.index_map[k]
        
        # Move the element to the end of the queue
        self.queue.append(self.queue.pop(index))  # Remove it from its current position and append to end
        
        # Update indices in the index map
        for i in range(index, len(self.queue) - 1):  # All elements moving left need to have their index updated
            self.index_map[self.queue[i]] = i

        # The last element will now be at the end; set its index
        self.index_map[element] = len(self.queue) - 1
        
        return element

```

### Explanation

1. **Initialization**:
   - In the `__init__` method, we create a list `queue` that initializes with the numbers from `1` to `n`.
   - We also maintain an `index_map` dictionary which maps each element to its current index in the `queue`. This allows quick lookup of where elements are in the list.

2. **Fetching an Element**:
   - The `fetch` method takes an integer `k` (the position of the element to fetch).
   - It retrieves the element from the `queue` based on the index from `index_map`.
   - The element is then moved to the end of the `queue` using `pop()` and `append()` methods that efficiently manage the order.
   - After the element is moved, we need to update the indices in the `index_map` for all elements that were moved left in the list (except the last one).
   - Finally, it updates the `index_map` for the fetched element, which now resides at the end of the `queue`.

### Complexity
- The time complexity for the `fetch` method is O(n) due to the need to update indices in the `index_map`, while maintaining the selection in O(1) thanks to the dictionary.

This implementation satisfies the constraints of the problem and should operate efficiently within the provided limits. You can directly run this code on LeetCode as it adheres to the required format for class-based solutions.

# 1797. Design Authentication Manager

### Problem Description 
There is an authentication system that works with authentication tokens. For each session, the user will receive a new authentication token that will expire `timeToLive` seconds after the `currentTime`. If the token is renewed, the expiry time will be extended to expire `timeToLive` seconds after the (potentially different) `currentTime`.

Implement the `AuthenticationManager` class:
`AuthenticationManager(int timeToLive)` constructs the `AuthenticationManager` and sets the `timeToLive`.

`generate(string tokenId, int currentTime)` generates a new token with the given `tokenId` at the given `currentTime` in seconds.

`renew(string tokenId, int currentTime)` renews the unexpired token with the given `tokenId` at the given `currentTime` in seconds. If there are no unexpired tokens with the given `tokenId`, the request is ignored, and nothing happens.

`countUnexpiredTokens(int currentTime)` returns the number of unexpired tokens at the given currentTime.

Note that if a token expires at time `t`, and another action happens on time `t` (`renew` or `countUnexpiredTokens`), the expiration takes place before the other actions.


Example 1:
Input
["AuthenticationManager", "`renew`", "generate", "`countUnexpiredTokens`", "generate", "`renew`", "`renew`", "`countUnexpiredTokens`"]
[[5], ["aaa", 1], ["aaa", 2], [6], ["bbb", 7], ["aaa", 8], ["bbb", 10], [15]]
Output
[null, null, null, 1, null, null, null, 0]
Explanation
AuthenticationManager authenticationManager = new AuthenticationManager(5); // Constructs the AuthenticationManager with `timeToLive` = 5 seconds.

authenticationManager.`renew`("aaa", 1); // No token exists with tokenId "aaa" at time 1, so nothing happens.

authenticationManager.generate("aaa", 2); // Generates a new token with tokenId "aaa" at time 2.

authenticationManager.`countUnexpiredTokens`(6); // The token with tokenId "aaa" is the only unexpired one at time 6, so return 1.

authenticationManager.generate("bbb", 7); // Generates a new token with tokenId "bbb" at time 7.

authenticationManager.`renew`("aaa", 8); // The token with tokenId "aaa" expired at time 7, and 8 >= 7, so at time 8 the `renew` request is ignored, and nothing happens.

authenticationManager.`renew`("bbb", 10); // The token with tokenId "bbb" is unexpired at time 10, so the `renew` request is fulfilled and now the token will expire at time 15.

authenticationManager.`countUnexpiredTokens`(15); // The token with tokenId "bbb" expires at time 15, and the token with tokenId "aaa" expired at time 7, so currently no token is unexpired, so return 0.


Constraints:
`1 <= timeToLive <= 108`
`1 <= currentTime <= 108`
`1 <= tokenId.length <= 5`
`tokenId` consists only of lowercase letters.

All calls to `generate` will contain unique values of `tokenId`.

The values of `currentTime` across all the function calls will be strictly increasing.

At most `2000` calls will be made to all functions combined.

### Solution 
 To solve the problem of the `AuthenticationManager`, we need to create a class that can manage tokens, including generating, renewing, and counting unexpired tokens. Below is the Python implementation following the LeetCode solution format:

### Python Code



```python
class AuthenticationManager:
    def __init__(self, timeToLive: int):
        """
        Constructs the AuthenticationManager with the specified timeToLive in seconds.
        """
        self.timeToLive = timeToLive
        self.tokens = {}  # Dictionary to hold tokenId and their expiration time

    def generate(self, tokenId: str, currentTime: int) -> None:
        """
        Generates a new token with the given tokenId at the given currentTime.
        The token expires after timeToLive seconds.
        """
        # Calculate the expiration time
        expirationTime = currentTime + self.timeToLive
        # Store the expiration time for the tokenId
        self.tokens[tokenId] = expirationTime

    def renew(self, tokenId: str, currentTime: int) -> None:
        """
        Renews the unexpired token with the given tokenId at the given currentTime.
        If no token exists or if it is expired, do nothing.
        """
        if tokenId in self.tokens:
            # If the token is still valid (not expired)
            if self.tokens[tokenId] > currentTime:
                # Renew the token by updating its expiration time
                self.tokens[tokenId] = currentTime + self.timeToLive

    def countUnexpiredTokens(self, currentTime: int) -> int:
        """
        Returns the number of unexpired tokens at the given currentTime.
        """
        # Remove expired tokens from the dictionary
        expired_tokens = [token for token, expiration in list(self.tokens.items()) if expiration <= currentTime]
        for token in expired_tokens:
            del self.tokens[token]
        
        # Return the count of remaining unexpired tokens
        return len(self.tokens)

# Example usage
# Uncommenting below lines will allow you to test the class directly
# authenticationManager = AuthenticationManager(5)
# authenticationManager.renew("aaa", 1) # No token exists, do nothing
# authenticationManager.generate("aaa", 2)
# print(authenticationManager.countUnexpiredTokens(6)) # Output: 1
# authenticationManager.generate("bbb", 7)
# authenticationManager.renew("aaa", 8) # Ignored, as token is expired
# authenticationManager.renew("bbb", 10) 
# print(authenticationManager.countUnexpiredTokens(15)) # Output: 0

```

### Explanation of the Code

1. **Initialization**:
   - The `AuthenticationManager` class is initialized with `timeToLive`, which denotes how long a token remains valid. 
   - We use a dictionary `tokens` to store tokens with their expiration times. The structure allows for O(1) access for generation and renewal.

2. **Generate Method**:
   - The `generate` method is responsible for creating a new token. It receives a `tokenId` and a `currentTime`.
   - The expiration time is calculated by adding `timeToLive` to `currentTime` and stored in the `tokens` dictionary.

3. **Renew Method**:
   - The `renew` method attempts to renew a token. It checks if the token exists and if it is unexpired (i.e., its expiration time is greater than `currentTime`).
   - If valid, it updates the expiration time. If expired, the renewal is ignored as per requirements.

4. **Count Unexpired Tokens Method**:
   - The `countUnexpiredTokens` method calculates the number of tokens that are still valid at a specified `currentTime`.
   - It first removes any expired tokens from `tokens` using a comprehension list that checks each token's expiration.
   - Finally, it returns the count of the remaining unexpired tokens.

### Time Complexity
- The generate and renew methods operate in O(1).
- The countUnexpiredTokens method has a time complexity of O(n), where n is the number of tokens, due to the cleanup of expired tokens.

### Conclusion
This implementation efficiently manages the tokens and adheres to the problem constraints, providing functionality through methods that can be easily tested. You can run this code directly on the LeetCode platform within the provided constraints.

# 1825. Finding MK Average

### Problem Description 
You are given two integers, `m` and `k`, and a stream of integers. You are tasked to implement a data structure that calculates the MKAverage for the stream.

The MKAverage can be calculated using these steps:
If the number of the elements in the stream is less than `m` you should consider the MKAverage to be `-1`. Otherwise, copy the last `m` elements of the stream to a separate container.

Remove the smallest `k` elements and the largest `k` elements from the container.

Calculate the average value for the rest of the elements rounded down to the nearest integer.

Implement the `MKAverage` class:
`MKAverage(int m, int k)` Initializes the MKAverage object with an empty stream and the two integers `m` and `k`.

`void addElement(int num)` Inserts a new element `num` into the stream.

`int calculateMKAverage()` Calculates and returns the MKAverage for the current stream rounded down to the nearest integer.


Example 1:
Input
["MKAverage", "addElement", "addElement", "calculateMKAverage", "addElement", "calculateMKAverage", "addElement", "addElement", "addElement", "calculateMKAverage"]
[[3, 1], [3], [1], [], [10], [], [5], [5], [5], []]
Output
[null, null, null, -1, null, 3, null, null, null, 5]
Explanation
MKAverage obj = new MKAverage(3, 1); 
obj.addElement(3);        // current elements are [3]
obj.addElement(1);        // current elements are [3,1]
obj.calculateMKAverage(); // return -1, because m = 3 and only 2 elements exist.

obj.addElement(10);       // current elements are [3,1,10]
obj.calculateMKAverage(); // The last 3 elements are [3,1,10].

                          // After removing smallest and largest 1 element the container will be `[3].

                          // The average of [3] equals 3/1 = 3, return 3
obj.addElement(5);        // current elements are [3,1,10,5]
obj.addElement(5);        // current elements are [3,1,10,5,5]
obj.addElement(5);        // current elements are [3,1,10,5,5,5]
obj.calculateMKAverage(); // The last 3 elements are [5,5,5].

                          // After removing smallest and largest 1 element the container will be `[5].

                          // The average of [5] equals 5/1 = 5, return 5
``

Constraints:
`3 <= m <= 105`
`1 <= k*2 < m`
`1 <= num <= 105`
At most `105` calls will be made to `addElement` and `calculateMKAverage`.

### Solution 
 To solve the problem of calculating the MKAverage from a stream of integers, we'll create a data structure that efficiently manages the incoming integers while allowing for quick access to the last `m` elements and facilitating the calculation of their average after removing the smallest and largest `k` elements.

Here's the plan:
1. **Initialization**: We initialize the class with the parameters `m` and `k`. To store the stream of integers, we will use a list.
2. **Adding Elements**: When a new integer is added, it will be appended to the stream.
3. **Calculating MKAverage**: 
    - If the total number of elements is less than `m`, we will return `-1`.
    - We will retrieve the last `m` elements and sort them.
    - We will remove the smallest and largest `k` elements and calculate the average of the remaining elements, rounding down.
4. **Efficiency Considerations**: Sorting is required but it is performed only on the last `m` elements, which ensures that the operations remain manageable within the given constraints.

Below is the Python implementation of this approach, following LeetCode's expected format.



```python
class MKAverage:

    def __init__(self, m: int, k: int):
        self.m = m
        self.k = k
        self.stream = []
        self.current_size = 0  # To track the size of the stream

    def addElement(self, num: int) -> None:
        self.stream.append(num)
        self.current_size += 1  # Increment the size of the current stream

    def calculateMKAverage(self) -> int:
        if self.current_size < self.m:
            return -1  # Not enough elements for MKAverage

        # Get the last m elements
        last_m_elements = self.stream[-self.m:]
        last_m_elements.sort()  # Sort the last m elements

        # Remove the smallest k and largest k elements
        relevant_elements = last_m_elements[self.k:self.m - self.k]

        # Calculate the MKAverage
        if not relevant_elements:  # If there are no elements left after removal
            return 0
        else:
            return sum(relevant_elements) // len(relevant_elements)

# Example Usage:
# obj = MKAverage(3, 1)
# obj.addElement(3)          # stream = [3]
# obj.addElement(1)          # stream = [3, 1]
# print(obj.calculateMKAverage())  # return -1
# obj.addElement(10)         # stream = [3, 1, 10]
# print(obj.calculateMKAverage())  # return 3
# obj.addElement(5)          # stream = [3, 1, 10, 5]
# obj.addElement(5)          # stream = [3, 1, 10, 5, 5]
# obj.addElement(5)          # stream = [3, 1, 10, 5, 5, 5]
# print(obj.calculateMKAverage())  # return 5

```

### Explanation of the Code:
1. **Constructor `__init__`**:
   - Initializes the parameters `m`, `k`, and sets up an empty list to store the stream of integers. We also maintain a counter for the size of the stream.

2. **Method `addElement`**:
   - Simply appends the new integer to the `stream` list and increments the `current_size`.

3. **Method `calculateMKAverage`**:
   - Checks if there are enough elements in the stream. If not, it returns `-1`.
   - Otherwise, it slices the last `m` elements, sorts them, and removes the smallest `k` and largest `k`.
   - It then computes the average of the remaining elements and returns it, ensuring to use integer division for rounding down.

This implementation is efficient and adheres to the constraints provided, making it suitable for use in a competitive programming context or an interview setting.

