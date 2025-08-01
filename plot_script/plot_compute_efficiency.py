"""
This scripts plots the compute efficiency of various models based on their FLOPs and average scores across multiple tasks.
Accuracy is capped at 70% for better visualization.
"""

import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from adjustText import adjust_text
import numpy as np
plt.rcParams.update({
    'font.size': 12, # 16,          
    'axes.labelsize': 12, # 16,     
    'axes.titlesize': 12, # 18,     
    'legend.fontsize': 12, # 14,    
    'xtick.labelsize': 12, # 14,    
    'ytick.labelsize': 12, #14     
})
scatter_text_size = 12 # 16
scatter_size = 60

model_name_map = {
    "Celerity_300M": "Celerity-300M",
    "Celerity_500M": "Celerity-500M",
    "Celerity_900M": "Celerity-900M",	
    "Celerity_1.8B": "Celerity-1.8B",	
    "Celerity_3.8B":   "Celerity-3.8B",
    "Celerity_SPJ_300M": "CeleritySPJ-300M",	
    "Celerity_SPJ_900M": "CeleritySPJ-900M",	
    "OLMo-1B-hf": "OLMo-1B",
    "OLMo-7B": "OLMo-7B",
    "OLMo-2-0425-1B": "OLMo2-1B",
    "OLMo-2-1124-7B": "OLMo2-7B",
    "OLMo-2-1124-13B": "OLMo2-13B",	
    "OLMo-2-0325-32B": "OLMo2-32B",	
    "Qwen3-0.6B-Base": "Qwen3-0.6B",
    "Qwen3-1.7B-Base": "Qwen3-1.7B",
    "Qwen3-4B-Base": "Qwen3-4B",
    "Qwen3-8B-Base": "Qwen3-8B",
    "Qwen3-14B-Base":  "Qwen3-14B",
    "Qwen2.5-0.5B": "Qwen2.5-0.5B",
    "Qwen2.5-1.5B": "Qwen2.5-1.5B",
    "Qwen2.5-3B": "Qwen2.5-3B",
    "Qwen2.5-7B": "Qwen2.5-7B",
    "Qwen2.5-14B": "Qwen2.5-14B",
    "Qwen2.5-32B": "Qwen2.5-32B",
    "gemma-3-1b-pt": "Gemma3-1B",
    "gemma-3-4b-pt": "Gemma3-4B",
    "gemma-3-12b-pt": "Gemma3-12B",
    "gemma-3-27b-pt": "Gemma3-27B",
    "gemma-3-1b-pt+forward": "Gemma3-1B+fwd",
    "gemma-3-1b-pt+forward+teacher": "Gemma3-1B+fwd+t",
    "gemma-3-4b-pt+forward": "Gemma3-4B+fwd",
    "gemma-3-4b-pt+forward+teacher": "Gemma3-4B+fwd+t",
    "gemma-3-12b-pt+forward": "Gemma3-12B+fwd",
    "gemma-3-12b-pt+forward+teacher": "Gemma3-12B+fwd+t",
    "gemma-2-2b": "Gemma2-2B",
    "gemma-2-9b": "Gemma2-9B",
    "gemma-2-27b": "Gemma2-27B",
    "gemma-2-2b+forward": "Gemma2-2B+fwd",
    "gemma-2-2b+forward+teacher": "Gemma2-2B+fwd+t",
    "gemma-2-9b+forward": "Gemma2-9B+fwd",
    "gemma-2-9b+forward+teacher": "Gemma2-9B+fwd+t",
    "Zamba2-1.2B": "Zamba2-1.2B",
    "Zamba2-2.7B": "Zamba2-2.7B",
    "Zamba2-7B": "Zamba2-7B",
    "phi-1": "Phi-1",
    "phi-1_5": "Phi-1.5",
    "phi-2": "Phi-2",
    "phi-4": "Phi-4",
    "huggyllama/llama-7b": "Llama-7B",
    "huggyllama/llama-13b": "Llama-13B",
    "Llama-2-7b-hf": "Llama2-7B",
    "Llama-2-13b-hf": "Llama2-13B",
    "Meta-Llama-3-8B": "Llama3-8B",
    "Llama-3.1-8B": "Llama3.1-8B",
    "Llama-3.2-1B": "Llama3.2-1B",
    "Llama-3.2-3B": "Llama3.2-3B",
    "Cerebras-GPT-111M": "CerebrasGPT-111M",
    "Cerebras-GPT-256M": "CerebrasGPT-256M",
    "Cerebras-GPT-590M": "CerebrasGPT-590M",
    "Cerebras-GPT-1.3B": "CerebrasGPT-1.3B",
    "Cerebras-GPT-2.7B": "CerebrasGPT-2.7B",
    "Cerebras-GPT-6.7B": "CerebrasGPT-6.7B",
    "Cerebras-GPT-13B": "CerebrasGPT-13B",
    "btlm-3b-8k-base": "BTLM-3B",
    "SmolLM2-135M": "SmolLM2-135M",
    "SmolLM2-360M": "SmolLM2-360M",
    "SmolLM2-1.7B": "SmolLM2-1.7B",
    "SmolLM-135M": "SmolLM-135M",
    "SmolLM-360M": "SmolLM-360M",
    "SmolLM-1.7B": "SmolLM-1.7B",
    "SmolLM3-3B-Base": "SmolLM3-3B"
}


def read_txt_to_df(file_path):
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        f.close()
    data = []
    for line in lines:
        row = line.strip().split('\t')
        data.append(row)
    df = pd.DataFrame(data)

    # Set the first row as column headers (model names)
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])

    # Add row names as the index
    original_row_names = [
        'num_params', 
        'num_tokens', 
        'tpp', 
        'vocab_size', 
        'hidden_size', 
        'MSL', 
        'num_hidden_layers', 
        'flops(6ND)', 
        'flops(6ND-nonembed)', 
        'flops(6ND-2VHD+12HSLD)', 
        'flops(final)',
        'flops(final-nonembed)',
        'arc_c', 
        'arc_e', 
        'boolq', 
        'hellaswag', 
        'piqa', 
        'siqa', 
        'winogrande', 
        'mmlu'
    ]
    df.index = original_row_names
    return df

def fit_and_plot_ld_curve(plot_data, plt, curve_color, model_list, label):
    
    # Extract Celerity points only
    ld_x_vals = []
    ld_y_vals = []
    for x_val, avg_score, model in plot_data:
        if model in model_list:
            ld_x_vals.append(x_val)
            ld_y_vals.append(100 - avg_score)
    
    print(ld_x_vals)
    print(ld_y_vals)
    
    ld_x = np.array(ld_x_vals)
    ld_y = np.array(ld_y_vals)

    X = np.log(ld_x)
    Y = np.log(ld_y)

    poly = np.polyfit(X, Y, 1)
    slope, intercept = poly[0], poly[1]

    a_fit = np.exp(intercept)
    b_fit = slope

    # Generate smooth curve for plotting
    x_smooth = np.logspace(19.8, 24.8, 100)
    y_smooth = 100 - a_fit * np.power(x_smooth, b_fit)
    
    c = a_fit ** (-1.0 / b_fit)
    plt.plot(
        x_smooth, y_smooth, 'r--', linewidth=2, 
        label=fr"{label}: $100 - \left(\frac{{\mathrm{{F}}}}{{{c:.3e}}}\right)^{{{b_fit:.3f}}}$",
        color=curve_color
    )
    
    return a_fit, b_fit


def plot_average_scores(df, models, tasks, x_value, output_path):

    plt.figure(figsize=(12, 8))
    
    plot_data = []
    Celerity_3p8B_score = None
    
    for model in models:
        if model not in df.columns:
            print(f"Model {model} not found in data.")
            continue
        
        if x_value not in df.index:
            print(f"X-value {x_value} not found in data.")
            continue
            
        # Get x-value directly by indexing
        x_val = float(df.loc[x_value, model])
        
        # Calculate average score across specified tasks
        task_scores = []
        for task in tasks:
            if task in df.index:
                try:
                    score = float(df.loc[task, model])
                    task_scores.append(score)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert score to float for model {model} task {task}")
            else:
                print(f"Task {task} not found in data.")
        
        if task_scores:
            avg_score = np.mean(task_scores) 
            plot_data.append((x_val, avg_score, model))
            if model == "Celerity_3.8B":
                Celerity_3p8B_score = avg_score
    
    # Plot the data
    for x_val, avg_score, model in plot_data:
        if "Celerity" in model:
            color = "orangered"
            label = model
        elif "gemma-2" in model:
            color = "deepskyblue"
            label = model
        elif "gemma-3" in model:
            color = "mediumseagreen"
            label = model
        elif "Smol" in model:
            color = "mediumpurple"
            label = model
        elif "OLMo" in model:
            color = "hotpink"
            label = model
        elif "llama" in model.lower():
            color = "slategray"
            label = model
        elif "Zamba" in model:
            color = "tan"
            label = model
        else:
            color = "sandybrown"
            label = None
        if avg_score: # < 70:
            plt.scatter(x_val, avg_score, color=color, s=scatter_size)
            
            offset_x = 1
            offset_y = 0.5   
            if "smol" in model.lower():
                offset_x = 1.1
            if "smollm-360m" in model.lower():
                offset_y = 0.2
            if "celerity" in model.lower():
                offset_x = 0.65
                offset_y = -1
            if "cerebras" in model.lower():
                offset_x = 1.2
                offset_y = -0.5
            if "btlm" in model.lower():
                offset_y = -0.9
            if "olmo" in model.lower():
                offset_y = 0.12
                offset_x = 1.1
            if "llama" in model.lower():
                offset_x = 1.0
            if "zamba" in model.lower():
                offset_x = 0.4
                offset_y = -1.0
            if "gemma-3" in model.lower():
                offset_x = 1.2
                offset_y = -0.5
            if "gemma-2" in model.lower():
                offset_x = 0.5
            if "celerity_1.8b" in model.lower():
                offset_x = 0.35
            if "celerity_3.8b" in model.lower():
                offset_x = 1.0
            if "celerity_900m" in model.lower():
                offset_x = 1.0
            if "llama-2-7b" in model.lower():
                offset_x = 1.12
                offset_y = 0.1
            if "llama-7b" in model.lower():
                offset_x = 0.5
                offset_y = 0.3

            plt.text(x_val * offset_x, avg_score + offset_y, model_name_map[model], fontsize=scatter_text_size, ha="left", va="bottom", color=color)    

    # plot Celerity curve fit
    model_list = [
        "Celerity_300M",
        "Celerity_500M",
        "Celerity_900M",	
        "Celerity_1.8B",	
        "Celerity_3.8B",
    ]
    # fit_and_plot_ld_curve(plot_data, plt, curve_color="orangered", model_list=model_list, label="Celerity Fit")
    fit_and_plot_ld_curve(plot_data, plt, curve_color="orangered", model_list=model_list[-3:], label="Celerity Accuracy-FLOPs Fit")

    
    # plt.ylim(52.5, 71)
    # plt.xlim(1e21, 5e23)
    plt.ylim(52.5, None)
    plt.xlim(1e21, None)
    plt.xlabel("Training FLOPs (F)")
    plt.ylabel('Average Accuracy (%)')
    # plt.title(f'Average Scores vs {x_value.replace("_", " ").title()}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    # plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    
    data_path = "/cb/home/clairez/ws/depth_mup/lm-evaluation-harness/plot_script/data.txt"
    data_df = read_txt_to_df(data_path)

    models_to_plot = [
        # "Celerity_300M",
        # "Celerity_500M",
        "Celerity_900M",	
        "Celerity_1.8B",	
        "Celerity_3.8B",
        # "Celerity_SPJ_300M",	
        # "Celerity_SPJ_900M",	
        "OLMo-1B-hf",
        "OLMo-7B",
        "OLMo-2-0425-1B",
        "OLMo-2-1124-7B",
        "OLMo-2-1124-13B",	
        "OLMo-2-0325-32B",	
        "Qwen3-0.6B-Base",
        "Qwen3-1.7B-Base",
        "Qwen3-4B-Base",
        "Qwen3-8B-Base",
        "Qwen3-14B-Base",
        "Qwen2.5-0.5B",
        "Qwen2.5-1.5B",
        "Qwen2.5-3B",
        "Qwen2.5-7B",
        "Qwen2.5-14B",
        "Qwen2.5-32B",
        "gemma-3-1b-pt",
        "gemma-3-4b-pt",
        "gemma-3-12b-pt",
        "gemma-3-27b-pt",
        # "gemma-3-1b-pt+forward",
        # "gemma-3-1b-pt+forward+teacher",
        # "gemma-3-4b-pt+forward",
        # "gemma-3-4b-pt+forward+teacher",
        # "gemma-3-12b-pt+forward",
        # "gemma-3-12b-pt+forward+teacher",
        "gemma-2-2b",
        "gemma-2-9b",
        "gemma-2-27b",
        # "gemma-2-2b+forward",
        # "gemma-2-2b+forward+teacher",
        # "gemma-2-9b+forward",
        # "gemma-2-9b+forward+teacher",
        "Zamba2-1.2B",
        "Zamba2-2.7B",
        "Zamba2-7B",
        # "phi-1",
        # "phi-1_5",
        # "phi-2",
        # "phi-4",
        "huggyllama/llama-7b",
        "huggyllama/llama-13b",
        "Llama-2-7b-hf",
        "Llama-2-13b-hf",
        "Meta-Llama-3-8B",
        "Llama-3.1-8B",
        "Llama-3.2-1B",
        "Llama-3.2-3B",
        # "Cerebras-GPT-111M",
        # "Cerebras-GPT-256M",
        # "Cerebras-GPT-590M",
        # "Cerebras-GPT-1.3B",
        # "Cerebras-GPT-2.7B",
        "Cerebras-GPT-6.7B",
        "Cerebras-GPT-13B",
        "btlm-3b-8k-base",
        # "SmolLM2-135M",
        "SmolLM2-360M",
        "SmolLM2-1.7B",
        # "SmolLM-135M",
        "SmolLM-360M",
        "SmolLM-1.7B",
        "SmolLM3-3B-Base"
    ]

    tasks_to_average = [
        "arc_c",
        "arc_e",
        "boolq", #
        "hellaswag",
        "piqa",
        "siqa", #
        "winogrande",
        # "mmlu"
    ]

    plot_average_scores(
        data_df,
        models_to_plot,
        tasks_to_average,
        x_value="flops(final)",
        output_path="acc_vs_flops_wo_mmlu_compute_efficiency.png",
    )