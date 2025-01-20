import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_file = 'results/plot_data.csv'

# el: episode length
# er: episode reward
# ia: invalid actions
# sr: success rate
# columns = ['episode', 'el', 'er', 'ia', 'sr']

# el: episode length
# er: episode reward
# ia: invalid actions
# sr: success rate
column_names = ['ID', 'el', 'er', 'ia', 'sr']
data = pd.read_csv(data_file, names=column_names, header=0)  # Skip the original header

print(f"The shape of the data is {data.shape}\n")

data.head()




# plot accumulate average er and accumulate average ia for all episodes

# Calculate cumulative average of 'er' and 'ia'
data['cum_avg_er'] = data['er'].cumsum() / (data.index + 1)
data['cum_avg_ia'] = data['ia'].cumsum() / (data.index + 1)

# Plot cumulative average episode rewards
plt.figure(figsize=(10, 5))
plt.plot(data['ID'], data['cum_avg_er'], label='Cumulative Average Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Cumulative Average Episode Reward')
plt.title('Cumulative Average Episode Reward over Episodes')
plt.legend()
plt.grid(True)
plt.show()

# Plot cumulative average invalid actions
plt.figure(figsize=(10, 5))
plt.plot(data['ID'], data['cum_avg_ia'], label='Cumulative Average Invalid Actions', color='orange')
plt.xlabel('Episode')
plt.ylabel('Cumulative Average Invalid Actions')
plt.title('Cumulative Average Invalid Actions over Episodes')
plt.legend()
plt.grid(True)
plt.show()




# Plot cumulative average episode rewards and invalid actions for the first 200 episodes

plt.figure(figsize=(10, 5))
plt.plot(data['ID'][:200], data['cum_avg_er'][:200], label='Cumulative Average Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Cumulative Average Episode Reward')
plt.title('Cumulative Average Episode Reward over Episodes (200)')
plt.legend()
plt.grid(True)
plt.show()

# Plot cumulative average invalid actions
plt.figure(figsize=(10, 5))
plt.plot(data['ID'][:200], data['cum_avg_ia'][:200], label='Cumulative Average Invalid Actions', color='orange')
plt.xlabel('Episode')
plt.ylabel('Cumulative Average Invalid Actions')
plt.title('Cumulative Average Invalid Actions over Episodes (200)')
plt.legend()
plt.grid(True)
plt.show()
