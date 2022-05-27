import numpy as np
from sumtree import SumTree

class Memory():  # stored as (s, a, r, s_) in SumTree
        
    """
    
    __init__ - create SumTree memory
    store - assign priority to new experience and store with SumTree.add & SumTree.update
    sample - uniformly sample from the range between 0 and total priority and 
                     retrieve the leaf index, priority and experience with SumTree.get_leaf
    batch_update - update the priority of experience after training with SumTree.update
    
    PER_e - Hyperparameter that avoid experiences having 0 probability of being taken
    PER_a - Hyperparameter that allows tradeoff between taking only experience with 
                    high priority and sampling randomly (0 - pure uniform randomness, 1 -
                    select experiences with the highest priority)
    PER_b - Importance-sampling, from initial value increasing to 1, control how much
                    IS affect learning
                    
    """
    
    PER_e = 0.01 
    PER_a = 0.6
    PER_b = 0.4
    PER_b_increment_per_sampling = 0.01
    absolute_error_upper = 1.  # Clipped abs error

    def __init__(self, capacity):
        
        self.tree = SumTree(capacity)

    def store(self, experience):
        
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0, this experience will never have a chance to be selected
        # So a minimum priority is assigned
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)

    def sample(self, n):

        """
        First, to sample a minibatch of k size, the range [0, priority_total] is
        divided into k ranges. A value is uniformly sampled from each range. Search 
        in the sumtree, the experience where priority score correspond to sample 
        values are retrieved from. Calculate IS weights for each minibatch element
        """

        b_memory = []
        b_idx = np.empty((n, ))
        b_ISWeights =  np.empty((n, 1))

        priority_segment = self.tree.tree[0] / n   

        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        prob_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.tree[0]
        max_weight = 0
        if prob_min != 0:
            max_weight = (prob_min * n) ** (-self.PER_b)

        for i in range(n):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)
            prob = priority / self.tree.tree[0]
            b_ISWeights[i, 0] = 0
            if max_weight != 0:
                b_ISWeights[i, 0] = (prob * n) ** (-self.PER_b) / max_weight
            b_idx[i]= index
            b_memory.append([data])

        return b_idx, b_memory, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):
            
        # To avoid 0 probability
        abs_errors += self.PER_e 
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
