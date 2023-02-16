import numpy as np
import numpy.random as rn

class Gridworld(object):
    def __init__.(self, grid_size, wind, discount):
        # Wind =  chance of moving randomly (float)
        # Discount = MDP discount (float)

        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1))