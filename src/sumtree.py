import numpy as np

# A tree based array containing priority of each experience for fast sampling

class SumTree():
    
    """
    __init__ - create data array storing experience and a tree based array storing priority
    add - store new experience in data array and update tree with new priority
    update - update tree and propagate the change through the tree
    get_leaf - find the final nodes with a given priority value
    """

    data_pointer = 0

    def __init__(self, capacity):
        
        """
        capacity - Number of final nodes containing experience
        data - array containing experience (with pointers to Python objects)
        tree - a tree shape array containing priority of each experience

         tree:
                0
             / \
            0   0
         / \ / \
        0  0 0  0 
        """
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype = object)
        self.tree = np.zeros(2 * capacity - 1)

    def add(self, priority, data):
        
        # Start from first leaf node of the most bottom layer
        tree_index = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data # Update data frame
        self.update(tree_index, priority) # Update priority

        # Overwrite if exceed memory capacity
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  
            self.data_pointer = 0

    def update(self, tree_index, priority):

        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index] 
        self.tree[tree_index] = priority

        # Propagate the change through tree
        while tree_index != 0: 
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):

        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            # Downward search, always search for a higher priority node till the last layer
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else: 
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
            
        # tree leaf index, priority, experience
        return leaf_index, self.tree[leaf_index], self.data[data_index]