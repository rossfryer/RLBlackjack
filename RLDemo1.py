import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --- Setup ---
env = gym.make("Blackjack-v1")
state_space = (32, 11, 2)  # player sum, dealer card, usable ace
action_space = 2  # hit or stick
Q = np.zeros(state_space + (action_space,))

alpha = 0.1    # learning rate
gamma = 1.0    # discount factor
epsilon = 0.1  # exploration rate
num_episodes = 500_000

rewards = []

# --- Q-Learning Loop ---
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        player_sum, dealer_card, usable_ace = state
        if np.random.rand() < epsilon:
            action = np.random.choice(action_space)
        else:
            action = np.argmax(Q[player_sum, dealer_card, usable_ace])

        next_state, reward, done, truncated, _ = env.step(action)
        next_player_sum, next_dealer_card, next_usable_ace = next_state

        best_next_action = np.argmax(Q[next_player_sum, next_dealer_card, next_usable_ace])
        td_target = reward + gamma * Q[next_player_sum, next_dealer_card, next_usable_ace, best_next_action] * (not done)
        td_delta = td_target - Q[player_sum, dealer_card, usable_ace, action]
        Q[player_sum, dealer_card, usable_ace, action] += alpha * td_delta

        state = next_state

    rewards.append(reward)

env.close()

# --- Plot Learning Progress ---
window = 10000
rolling_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')

plt.figure(figsize=(10, 4))
plt.plot(rolling_avg)
plt.xlabel('Episode (x10,000)')
plt.ylabel('Average Reward')
plt.title('Blackjack Q-Learning – Rolling Average Reward')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Watch Trained Agent Play ---
env = gym.make("Blackjack-v1", render_mode="ansi")

num_demo_episodes = 5
for episode in range(num_demo_episodes):
    state, _ = env.reset()
    done = False
    print(f"\n--- Demo Game {episode + 1} ---")
    while not done:
        print(env.render())
        player_sum, dealer_card, usable_ace = state
        action = np.argmax(Q[player_sum, dealer_card, usable_ace])
        state, reward, done, truncated, _ = env.step(action)
    print(env.render())
    print(f"Result: {'Win' if reward > 0 else 'Loss' if reward < 0 else 'Draw'}")

env.close()