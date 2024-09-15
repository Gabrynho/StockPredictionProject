import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the data
data = {
    'Stock': ['AAPL'] * 12 + ['NVDA'] * 12 + ['MSFT'] * 12,
    'Latency': [2, 2, 2, 4, 4, 4, 7, 7, 7, 14, 14, 14] * 3,
    'Clustering': ['VI', 'EM', 'KMeans'] * 4 * 3,
    'MAPE': [
        1.34, 1.34, 1.21, 1.34, 1.34, 1.19, 1.34, 1.34, 1.24, 1.34, 1.34, 1.34,  # AAPL
        2.94, 2.85, 2.82, 2.87, 2.85, 2.93, 2.87, 2.85, 2.82, 2.87, 2.85, 2.94,  # NVDA
        1.05, 1.08, 1.08, 1.05, 1.04, 1.08, 1.04, 1.30, 1.04, 1.04, 1.30, 1.08   # MSFT
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set up the plot
fig, axes = plt.subplots(3, 1, figsize=(20, 10))
sns.set(style="whitegrid")

# Define a function to create a subplot for each stock
def plot_for_stock(stock, ax):
    sns.lineplot(data=df[df['Stock'] == stock], x='Latency', y='MAPE', hue='Clustering', style='Clustering', markers=True, dashes=False, ax=ax)
    ax.set_title(f'MAPE Errors for {stock}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Latency (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAPE Error', fontsize=1, fontweight='bold')

# Create subplots for AAPL, NVDA, MSFT
plot_for_stock('AAPL', axes[0])
plot_for_stock('NVDA', axes[1])
plot_for_stock('MSFT', axes[2])

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('MAPE_Errors.png')
plt.close()
