# SpotMicro Waypoint Tracking with Reinforcement Learning

This project trains a **SpotMicro quadruped robot in MuJoCo** using **Proximal Policy Optimization (PPO)** to autonomously **walk and steer toward target coordinates**.

The learned controller allows the robot to:

* Walk using a stable **trot gait**
* Turn toward a **target coordinate**
* Reach goals efficiently
* Follow **predefined waypoint tracks loaded from CSV**

The system combines a **bio-inspired gait generator** with **reinforcement learning** to produce stable locomotion and navigation.

---

# Demo

[![SpotMicro Waypoint Tracking](https://img.youtube.com/vi/zI3Rre4-SZI/0.jpg)](https://youtu.be/zI3Rre4-SZI)

Click the image above to watch the full video demonstration.

---

# Project Overview

The objective of this project is to enable a quadruped robot to **navigate toward arbitrary goal coordinates** in simulation.

Instead of controlling raw joint torques, the reinforcement learning policy outputs **high-level gait parameters**, which are converted into joint commands using a deterministic trot controller.

This greatly improves learning stability while maintaining realistic locomotion behavior.

---

# System Architecture

The locomotion pipeline consists of three main components.

## Reinforcement Learning Policy

A PPO policy learns to output:

* gait frequency scaling
* stride amplitude scaling
* turning bias

These parameters determine how the robot moves.

---

## Gait Generator

A deterministic trot controller converts policy outputs into joint commands.

```
policy → gait parameters → trot_ctrl() → joint targets
```

The gait generator provides:

* diagonal trot pattern
* controllable turning
* stable stance phase

---

## MuJoCo Simulation

The robot is simulated using **MuJoCo**, which provides:

* accurate rigid-body dynamics
* articulated quadruped model
* joint position control
* real-time visualization

---

# Environment

The reinforcement learning environment exposes robot state and goal information to the policy.

## Observation Space

The policy observes:

* base roll, pitch, yaw
* angular velocity
* base height
* joint angles
* joint velocities
* goal relative position
* distance to goal
* heading error
* body-frame velocity

These features allow the robot to understand both **its state** and **where the goal is located**.

---

## Action Space

The policy outputs three parameters:

```
[action0] gait frequency scaling
[action1] stride amplitude scaling
[action2] turning bias
```

These values modulate the trot gait.

---

# Reward Function

The reward function encourages **efficient goal reaching**.

Main components include:

### Goal Distance Reduction

Reward for reducing distance to the target.

### Progress Along Goal Direction

Encourages movement toward the target instead of simply moving forward.

### Heading Alignment

Encourages the robot to orient toward the goal direction.

### Stability Penalty

Penalizes excessive roll and pitch.

### Energy Penalty

Discourages unnecessary actuator effort.

### Success Bonus

A large reward is given when the robot reaches the target radius.

This ensures **task completion dominates the objective**.

---

# Waypoint Tracking

Targets can be loaded from a CSV file instead of randomly generated.

Example:

```
x,y
1.0,0.0
1.2,0.3
1.5,0.6
2.0,1.0
```

This allows evaluation on **structured trajectories** such as:

* S-curves
* corridor navigation
* path following

---

# Training

Training uses **Stable-Baselines3 PPO**.

Recommended parameters:

```
learning_rate = 3e-4
n_steps = 4096
batch_size = 256
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
ent_coef = 0.01
clip_range = 0.2
target_kl = 0.03
```

Typical training duration:

```
3M – 5M timesteps
```

After training, the robot learns to:

* maintain stable trotting
* steer toward targets
* reach goals reliably

---

# Running the Policy

Run a trained policy using:

```
python run_tracker.py
```

The robot will:

1. spawn in simulation
2. receive a target coordinate
3. walk toward the target
4. repeat for the next waypoint

---

# Project Structure

```
spotmicro/
│
├── assets/
│   └── robot_base.xml
│
├── envs/
│   ├── mj_base_spotmicro_dr.py
│   └── mj_tracker_spotmicro.py
│
├── gaits/
│   └── spotmicro_gaits.py
│
├── scripts/
│   ├── train_tracker.py
│   └── run_tracker.py
│
├── waypoints.csv
│
└── README.md
```

---

# Key Features

* MuJoCo quadruped simulation
* PPO reinforcement learning
* bio-inspired trot gait controller
* goal-directed locomotion
* CSV waypoint navigation
* real-time visualization

---

# Future Work

Potential improvements include:

* terrain randomization
* disturbance rejection
* obstacle avoidance
* path-tracking reward instead of waypoint hopping
* sim-to-real transfer to physical SpotMicro hardware

---

# Dependencies

```
mujoco
numpy
gymnasium
stable-baselines3
```

---

# License

MIT License

---
