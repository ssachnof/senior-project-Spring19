# possible move types

JUMP = 0
STEP = 1

# players

PLAYER1 = 1
PLAYER2 = -1


DISCOUNT_FACTOR = .9
LEARNING_RATE = .001


# EPSILON_DECAY_RATE = .9999
EPSILON_DECAY_RATE = .9995
MIN_EPSILON_VALUE = .01


# maximum number of training episodes before swapping the agent's training and target networks
MAX_TRAINING_EPISODES = 256 # this hyperparam will probably have to be changed later

# represents the maximum number of episodes before swapping the training agent(ie. frozen -> active)
# MAX_AGENT_LIVE_EPISODES = 100000

MAX_AGENT_LIVE_EPISODES = 1024

MAX_MEMORY_CAPACITY = 1024 # this hyperparameter will probably have to be changed later

MAX_EPISODES = 10000000



# move mapping constants

BACKWARD_LEFT = 0
BACKWARD_RIGHT = 1
FORWARD_LEFT = 2
FORWARD_RIGHT = 3
BACKWARD_JUMP_LEFT = 4
BACKWARD_JUMP_RIGHT = 5
FORWARD_JUMP_LEFT = 6
FORWARD_JUMP_RIGHT = 7


MEMORY_SAMPLE_SIZE = 512
