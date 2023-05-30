import numpy as np

# global variables
# simulation parameters
N_QAGENT = 50           #iterations of the basic value iteration agent
N_IRLAGENT = 1          #number of iterations of IRLplus to perform
EXPLORE = 0.3           #the explore proportion: (1-EXPLORE) for exloit
MANUAL_FEEDBACK = 0.1   #reward feedback from human: + and -
NEUTRAL_FEEDBACK = 0.05 #if no feedback, this reward applied (+)
LOGGING = False         #set full logging to terminal or not...

##########################################################
# The maze environment
##########################################################
class State:
    def __init__(self, position, goal):
        self.position = position
        self.goal = goal
        self.isEnd = False

    def giveReward(self):
        if self.position == self.goal:
            return 1
        else:
            return 0

    def isEndFunc(self):
        if self.position == self.goal:
            self.isEnd = True

    def nxtPosition(self, action):
        """
        action: up, down, left, right
        return next position
        """
        # Use a more complex algorithm for pathfinding
        # such as the A* algorithm
        ...


##########################################################
# Agent using A* algorithm for navigation
##########################################################
class Agent:

    def __init__(self, map, goal):
        self.map = map
        self.goal = goal
        self.actions = ["up", "down", "left", "right"]
        self.State = State(position=self.map.start, goal=self.goal)

    def chooseAction(self):
        # Choose the best action based on the A* algorithm
        ...

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(position=position, goal=self.goal)

    def reset(self):
        self.State = State(position=self.map.start, goal=self.goal)

    def play(self, rounds=10):
        i = 0
        print ("")
        print ("NAVIGATION START")
        print ("")
        ...
