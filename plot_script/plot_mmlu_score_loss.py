import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    'font.size': 16,          
    'axes.labelsize': 16,     
    'axes.titlesize': 18,     
    'legend.fontsize': 13,    
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,    
    'lines.linewidth': 2 
})


# [flops, mmlu_zero_shot_score, mmlu_answer_token_loglikelihood, mmlu_average_token_negative_loglikelihood]
mmlu_dict = {
    "Celerity-300M": [1.47E+20, 0.2350, -1.9788, 2.836041887601217],
    "Celerity-500M": [5.15E+20, 0.2458, -1.6830, 2.6071055136859145],
    "Celerity-900M": [1.68E+21, 0.2410, -1.6552, 2.461590701042245],
    "Celerity-1.8B": [6.54E+21, 0.2611, -1.6243, 2.325515921801737],
    "Celerity-3.6B": [2.89E+22, 0.4180, -1.4756, 2.2271793043776733],
}

# Extract data from the dictionary
model_names = list(mmlu_dict.keys())
flops = [data[0] for data in mmlu_dict.values()]
mmlu_zero_shot_score = [data[1] for data in mmlu_dict.values()]
mmlu_answer_token_loglikelihood = [data[2] for data in mmlu_dict.values()]
mmlu_avg_neg_loglikelihood = [data[3] for data in mmlu_dict.values()]

mmlu_answer_token_neg_loglikelihood = [-data[2] for data in mmlu_dict.values()]
mmlu_answer_token_prob = [np.exp(data[2]) for data in mmlu_dict.values()]
mmlu_avg_token_prob = [np.exp(-data[3]) for data in mmlu_dict.values()]

# ========== Min-max normalization function ========== 
def normalize(data):
    """Normalize data to a 0-1 scale."""
    data_np = np.array(data)
    min_val = np.min(data_np)
    max_val = np.max(data_np)
    if max_val == min_val:
        return np.zeros_like(data_np)
    return (data_np - min_val) / (max_val - min_val)


# Create a figure with two subplots, side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Subplot 1: Twin-Axis Performance ---

# Plot lines without markers first
ax1.plot(flops, mmlu_zero_shot_score, '-', color='tab:blue', label='MMLU Acc.')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xscale('log')
ax1.set_xlabel('Training FLOPs')
ax1.set_ylabel('MMLU Acc. Score', color="tab:blue")
ax1.grid(True, which="both", ls="--", c='0.7')

# Create a second y-axis for the loss
ax1_twin = ax1.twinx()
ax1_twin.set_ylabel('MMLU Token NLL', color="tab:red")
ax1_twin.plot(flops, mmlu_answer_token_neg_loglikelihood, ':', color='tab:red', label='MMLU Answer Token NLL')
ax1_twin.plot(flops, mmlu_avg_neg_loglikelihood, '-.', color="tab:red", label='MMLU Text Avg. Token NLL')
ax1_twin.tick_params(axis='y', labelcolor="tab:red")

# Define markers for each model
markers = ['o', 's', '^', 'D', 'P']
# Add markers for each point
for i in range(len(flops)):
    marker = markers[i % len(markers)]
    ax1.plot(flops[i], mmlu_zero_shot_score[i], marker=marker, color='tab:blue', markersize=8)
    ax1_twin.plot(flops[i], mmlu_answer_token_neg_loglikelihood[i], marker=marker, color='tab:red', markersize=8)
    ax1_twin.plot(flops[i], mmlu_avg_neg_loglikelihood[i], marker=marker, color='tab:red', markersize=8)

ax1.set_title('Celerity Model MMLU Performance')
# Combine legends from both y-axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper center')





# --- Subplot 2: Normalized Performance ---
# Normalize the data
normalized_score = normalize(mmlu_zero_shot_score)
normalized_answer_prob = normalize(mmlu_answer_token_prob)
normalized_avg_prob = normalize(mmlu_avg_token_prob)

# Create the plot
ax2.plot(flops, normalized_score, '-', label='MMLU Acc.', color='tab:blue')
ax2.plot(flops, normalized_answer_prob, ':', label='MMLU Answer Token Prob.', color='tab:green')
ax2.plot(flops, normalized_avg_prob, '-.', label='MMLU Text Avg. Token Prob.', color='tab:green')
# Define markers for each model
markers = ['o', 's', '^', 'D', 'P']
# Add markers for each point
for i in range(len(flops)):
    marker = markers[i % len(markers)]
    ax2.plot(flops[i], normalized_score[i], marker=marker, color='tab:blue', markersize=8)
    ax2.plot(flops[i], normalized_answer_prob[i], marker=marker, color='tab:green', markersize=8)
    ax2.plot(flops[i], normalized_avg_prob[i], marker=marker, color='tab:green', markersize=8)

ax2.set_xlabel('Training FLOPs')
ax2.set_ylabel('Prob. or Acc. Normalized to 0-1 Scale')
ax2.set_xscale('log')
ax2.grid(True, which="both", ls="--", c='0.7')
ax2.set_title('Celerity Model Normalized MMLU Performance')
ax2.legend(loc="upper center")


# Final adjustments and saving
fig.tight_layout()
plt.savefig('mmlu_performance_subplots.png')
plt.show()