# Human_robot_interaction

PROBLEM DOMAIN DESCRIPTION
For navigation system for a mobile robot in a complex
indoor environment, The environment will be a maze like
structure with rooms, obstacles. The mobile robot must be able
to move around its surroundings to avoid obstacles, rooms etc
and should go where it needs to go. The mobile robot will
include actuators to direct its movement as well as sensors
to locate itself and detect obstacles. The navigation system
objective is to help the mobile robot get there as quickly
possible while avoiding collisions and avoiding obstacles.
The navigation system will include human guided learning
to enable a human operator to direct and correct the robot.
The underlying reward function will be determined by inverse
reinforcement learning algorithms from the scenarios given by
the human operator. The learnt reward function will direct the
robot to make choices which are consistent with the human
operator stated preferences. The robot will be rewarded neg-
atively for colliding with obstacles and will reward positively
for avoiding them if the human operator has shown a tendency
for doing so.
Fig. 1: Simulation Grid for Complex door Envirinment
In Fig 1,There is maze like structured environment with 3
rows and 4 columns, Where X is obstacles, R is room and
agent is O. Start state is (0, 2), Win state is (0, 3) and Lose
state is (1, 3).
Q-learning methods was used for navigation system that
will help to determine the best plan of action for maximising
expected cumulative reward over time. Through updating the
value of the Q-function which will calculate the expected
reward for each action based on the current state, Q-learning
algorithms allow the robot to learn from its experiences.
To identify the most effective route to the desired location,
the navigation system will need to strike a balance between
the exploration of new paths and the use of past learned
experiences.
With the use of human guided learning such as IRL, IRL
plus agents, Human demonstrates feedback for actions which
help the mobile robot with the ability to navigate across a
complicated indoor environment while avoiding obstacles and
reaching its goal as quickly as possible.
A. Q-Learning
Q-learning is a popular and effective reinforcement learning
algorithm which is chosen for this task because it can help the
robot to learn from its past experiences in the environment and
improve its performance over time. It can assist the robot in
navigating the safest and most effective route to its goal [8].
Based on the idea of a Q-table which hold what is expected
value of the cumulative reward for each potential action
in each potential state, the Q-learning algorithm learns new
behaviours. Based on the rewards received by the robot for
each action performed in each state, the algorithm updates the
Q-table through a process of trial and error [8]. The following
formula is used to update the Q-value for a given state-action
pair:
Q(s, a) = (1−α)∗Q(s, a)+α∗(R(s, a)+γ ∗max(Q(s′, a′)))
(1)
where Q(s,a) is the Q-value for state s and action a, alpha
is the learning rate, R(s,a) is the reward received for acting
in state s, gamma is the discount factor that determines the
significance of future rewards, s’ is the following state, and
max(Q(s’,a’)) is the highest Q-value out of all possible actions
in state s’ [8].
B. Feed Back
Humans provide input to the agent in two different ways by
using rewards. First, based on the rewards the agent obtains
for its actions, they immediately send feedback to the agent
[9]. Secondly they direct the agent towards accomplishing its
long term objectives by using rewards. Humans tend to provide
the agent with more positive feedback than negative feedback.
The agent may be inspired to keep learning and developing
by this encouraging feedback [9]. Human feedback behaviour
can alter as they engage with the learning agent more and
become more familiar with how it operates. As they become
more familiar with the agents skills and limitations, they might
offer more complicated or nuanced input. For navigating robot
in complex indoor environment, there will be having many
obstacles like room, wall. Human provide negative feedback
to the agent, if there is any wall or Room.
When the human identify the proposed action to be good
by inputting ”g” and then reward is set to MANUAL FEED-
BACK. Similarly, when the human identify the proposed
action to be bad by inputting b, the reward is set to -MANUAL
FEEDBACK. In all other cases, reward is set to NEUTRAL
FEEDBACK.
C. Inverse reinforcement learning
Using Inverse reinforcement learning robot learns the under-
lying reward function that the operator is optimising for instead
of simply learning how to mimic the actions of the human
operator [10]. Inverse reinforcement learning can be used to
learn the reward function that a human operator can be used to
direct the behaviour of a mobile robot in a complicated indoor
environment. With a deeper understanding of the operator
objectives and preferences, the robot will be able to navigate
the environment with more knowledge [10].
D. IRL agent plus
The IRL agent plus typically involves an iterative process
where the robot observes the operator behaviours and uses
this knowledge to estimate the underlying reward function.
The robot then plans its own actions and behaviour while
taking into consideration the environmental uncertainties and
limits using this reward function [10]. The IRL agent plus
planning and execution system typically involves a decision
making procedure that takes into account both the environ-
ment present condition and the learnt reward function. While
avoiding obstacles and taking into account any constraints or
capabilities of the robot, agent prepares a series of actions that
will maximise the learned reward function. The IRL agent plus
receives performance data during execution, which is then used
to update the reward function that was previously learnt and
enhance the robots performance over time.
E. SPARC
Supervised Progressively Autonomous Robot Competencies
is an extension of the principle behind Inverse Reinforcement
Learning Where machine learning agent learns to infer the
underlying reward function from a collection of expert demon-
strations. According to the SPARC framework, human experts
should not only give the machine learning agent examples of
good behaviour but also take action to stop examples of bad
behaviour from happening [11].
In other words, SPARC actively works to stop the machine
learning agent from doing things that it shouldn’t by using
human guidance in addition to demonstrating to it what it
should do. By learning from both good and bad examples, the
machine learning agent can gain a deeper grasp of the task at
hand [11].
Fig. 2: Example of Bad behaviour
In Fig 2, The agent next chosen action is right, but there
is obstacles nearby. So human can give feedback of bad
behaviour to stop examples from future happening.
F. Workload
The workload of training a navigation system for a mobile
robot in a complex indoor environment can be high. Humans
may be involved in various aspects of data collection, annota-
tion and labeling to provide the robot with information about
the environment. Humans may also be involved in defining the
tasks or objectives for the robot and in monitoring the robots
performance during training [10].
G. Speed of Training
The degree of human involvement may have an impact on
how quickly the navigation system is trained. The training
process may be slowed significantly, if humans are required to
classify or interpret a significant volume of data. However, Hu-
man can speed up training if they use their domain knowledge
to choose suitable characteristics or improve hyperparameters
[10].
H. Performance over time
The presence of people during training may have an impact
on the navigation systems performance. The robots perfor-
mance could decrease over time if the training data is not
accurate or if humans identify objects incorrectly. However
the performance of the robot can advance over time if people
are involved in monitoring it and making changes as necessary
[10].
IV. RESULTS
Method Avg steps taken Avg Human Workload
Q learning 11.28 Low
IRL Agent 13 Medium
IRL Agent plus 12 High
TABLE I: Comparison of methods
In table 1, It is noticed that Q learning is efficient for this
environment with average of 11.28 steps taken for 40 iterations
to reach the goal state with low human workload. It learns
through trail and error to maximise reward signal and recieves
feedback as reward. IRL has a higher average step count of
13 and a medium human workload. It attempts to infer the
underlying reward function that a human might have used to
generate a given behavior. It is single feedback system asks
whether the chosen action is good or bad to human operator.
IRL agent plus has slightly lower average step count of 12
compared to IRL, but it requires a higher human workload
because it is two feed back system ask whether the chosen
action is good or bad to human operator and tell us which
action to be chosen next. IRL agent plus is an extension of
IRL which incorporates additional constraints or assumptions
into the reward function estimation process.
IRL and IRL agent plus give greater flexibility and can
handle more complicated situations, but with increased human
participation. Q learning appears to be the most effective and
least human-intensive method but considering reward it will be
either +1 positive reward or -1 negative reward whereas IRL
plus agent will have positive reward only because of human
feedback given to agent for every action. The values of learn-
ing method will change according to different environment.
Fig. 3: Results of IRL plus Agent
In Fig 3, IRL agent plus algorithm has helped the agent to
reach the goal state with the help of human operator to recieve
maximum reward. Bad scenarios also suggested with number
of iteration to agent where it can learn to avoid future bad
happenings in this complex indoor environment.
V. DISCUSSION
It’s difficult to design mobile robot navigation systems
for complex indoor situations. In such situations, This study
proposes Human-guided learning technique for mobile robot
navigation. The efficiency of reinforcement learning algo-
rithms are tested such as Q-learning and Inverse Reinforce-
ment Learning and IRL plus Agent with and without human
guidance using a grid-world simulation of a complex indoor
space. With an average of 11.28 steps and 40 iterations, Q-
learning was effective and least Human guided strategy but
reward may be positive or negative Unlike IRL plus agent.
Since Q learning is trail and error method it will hit obstacles
and room without human guidance. The available literature
on the use of human assistance in reinforcement learning
shown how including human advice could affect the robot’s
performance. IRL agent plus algorithm helped the agent reach
the goal state with the help of the human operator to receive
maximum reward. The study highlights that the human guided
learning strategy can be effective in indoor navigation systems
for mobile robots in complex environments. They can avoid
obstacles and rooms with proper human guidance with two
feed back system. IRL and IRL agent plus provided greater
flexibility and could handle more complicated situations but
human work load is more.
VI. CONCLUSION
In conclusion, developing navigation systems for mobile
robots in complex indoor environments is a challenging task.
These study finds that Q-learning may be effective in average
steps but it will hit obstacles and rooms and get negative
reward as well but with the help of human guidance and
two feedback systems IRL agent plus agent offers greater
flexibility and can handle complex situations, IN IRL agent
plus only human workload is more. Overall, the human
guided learning strategy will solve challenging task and it
is a successful method for creating mobile robot navigation
systems in complex indoor environments
