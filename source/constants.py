# possible move types

JUMP = 0
STEP = 1

# players

PLAYER1 = 1
PLAYER2 = -1


DISCOUNT_FACTOR = .9
LEARNING_RATE = .001


EPSILON_DECAY_RATE = .995
MIN_EPSILON_VALUE = .01


# maximum number of training episodes before swapping the agent's training and target networks
MAX_TRAINING_EPISODES = 1000 # this hyperparam will probably have to be changed later

# represents the maximum number of episodes before swapping the training agent(ie. frozen -> active)
MAX_AGENT_LIVE_EPISODES = 100000


MAX_MEMORY_CAPACITY = 10000 # this hyperparameter will probably have to be changed later

MAX_EPISODES = 10000000