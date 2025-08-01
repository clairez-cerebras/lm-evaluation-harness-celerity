import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np


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
    original_row_names = ['num_params', 'num_tokens', 'tpp', 'flops(6ND)', 'vocab_size', 'hidden_size', 'MSL', 'num_hidden_layers', 'flops(new)', 'arc_c', 'arc_e', 'boolq', 'hellaswag', 'piqa', 'siqa', 'winogrande', 'mmlu']
    df.index = original_row_names

    return df


def adjust_flops_by_tpp(flops, tpp):
    factor = (1 - 0.023*np.log(np.sqrt(20.0/tpp))**2)**(-1.0/0.055)
    return flops / factor


def fit_and_plot_ld_curve(plot_data, plt, curve_color, model_list, label):
    
    # Extract LD points only
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
    
    plt.plot(x_smooth, y_smooth, 'r--', linewidth=2, 
                label=f'{label}: 100 - {a_fit:.3e} * x^({b_fit:.3f})', color=curve_color)
    
    return a_fit, b_fit


def plot_average_scores(df, models, tasks, x_value, output_path, adjust_20tpp=False):

    plt.figure(figsize=(12, 8))
    
    plot_data = []
    LD_3p8B_score = None
    
    for model in models:
        if model not in df.columns:
            print(f"Model {model} not found in data.")
            continue
        
        if x_value not in df.index:
            print(f"X-value {x_value} not found in data.")
            continue
            
        # Get x-value directly by indexing
        x_val = float(df.loc[x_value, model])

        if adjust_20tpp:
            tpp_val = float(df.loc["tpp", model])
            x_val = adjust_flops_by_tpp(x_val, tpp_val)
        
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
            if model == "LD_3.8B":
                LD_3p8B_score = avg_score
    
    # Plot the data
    for x_val, avg_score, model in plot_data:
        if "LD" in model:
            color = "red"
            label = model
        elif "gemma-2" in model:
            color = "blue"
            label = model
        elif "gemma-3" in model:
            color = "green"
            label = model
        else:
            color = "black"
            label = None
        plt.scatter(x_val, avg_score, label=label, color=color, s=60)
    
    # plot gemma lines
    gemma_names = [
        "gemma-3-1b-pt",
        "gemma-3-4b-pt",
        "gemma-3-12b-pt",
        "gemma-3-27b-pt",
        "gemma-3-1b-pt+forward",
        "gemma-3-1b-pt+forward+teacher",
        "gemma-3-4b-pt+forward",
        "gemma-3-4b-pt+forward+teacher",
        "gemma-3-12b-pt+forward",
        "gemma-3-12b-pt+forward+teacher",
        "gemma-2-2b",
        "gemma-2-9b",
        "gemma-2-27b",
        "gemma-2-2b+forward",
        "gemma-2-2b+forward+teacher",
        "gemma-2-9b+forward",
        "gemma-2-9b+forward+teacher",
    ]

    def plot_gemma_lines(points, color, linestyle, label, plt):
        xs, ys = zip(*points)
        plt.plot(xs, ys, color=color, linestyle=linestyle, linewidth=2, label=label)

    gemma2_points = [(flops, avg_score) for flops, avg_score, model in plot_data if "gemma-2" in model and "+" not in model]
    plot_gemma_lines(gemma2_points, "blue", "dotted", "Gemma-2", plt)
    gemma2_forward_points = [(flops, avg_score) for flops, avg_score, model in plot_data if "gemma-2" in model and "+forward" in model and "+teacher" not in model]
    plot_gemma_lines(gemma2_forward_points, "blue", "dashed", "Gemma-2+forward", plt)
    gemma2_forward_teacher_points = [(flops, avg_score) for flops, avg_score, model in plot_data if "gemma-2" in model and "+forward+teacher" in model]
    plot_gemma_lines(gemma2_forward_teacher_points, "blue", "solid", "Gemma-2+forward+teacher", plt)
    gemma3_points = [(flops, avg_score) for flops, avg_score, model in plot_data if "gemma-3" in model and "+" not in model]
    plot_gemma_lines(gemma3_points, "green", "dotted", "Gemma-3", plt)
    gemma3_forward_points = [(flops, avg_score) for flops, avg_score, model in plot_data if "gemma-3" in model and "+forward" in model and "+teacher" not in model]
    plot_gemma_lines(gemma3_forward_points, "green", "dashed", "Gemma-3+forward", plt)
    gemma3_forward_teacher_points = [(flops, avg_score) for flops, avg_score, model in plot_data if "gemma-3" in model and "+forward+teacher" in model]
    plot_gemma_lines(gemma3_forward_teacher_points, "green", "solid", "Gemma-3+forward+teacher", plt)


    # plot LD curve fit
    model_list = [
        "LD_300M",
        "LD_500M",
        "LD_900M",	
        "LD_1.8B",	
        "LD_3.8B",
    ]
    fit_and_plot_ld_curve(plot_data, plt, curve_color="red", model_list=model_list, label="LD Fit")
    fit_and_plot_ld_curve(plot_data, plt, curve_color="purple", model_list=model_list[-3:], label="LD Fit, last 3 points")

    # plot LD adjusted curve
    LD_3p8B_flops = float(df.loc[x_value, "LD_3.8B"]) * (1-0.27)
    plot_data.append((LD_3p8B_flops, LD_3p8B_score, "LD_3.8_adjusted"))
    model_list_adj = [
        "LD_300M",
        "LD_500M",
        "LD_900M",	
        "LD_1.8B",	
        "LD_3.8_adjusted"
    ]
    fit_and_plot_ld_curve(plot_data, plt, curve_color="orange", model_list=model_list_adj, label="LD Adjusted Fit")
    fit_and_plot_ld_curve(plot_data, plt, curve_color="pink", model_list=model_list_adj[-3:], label="LD Adjusted Fit, last 3 points")
    plt.scatter(LD_3p8B_flops, LD_3p8B_score, label="LD_3.8_adjusted", color="orange", s=60)
    
    plt.xlabel(x_value.replace('_', ' ').title())
    plt.ylabel('Average Score')
    plt.title(f'Average Scores vs {x_value.replace("_", " ").title()} ({",".join(tasks)})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    
    data_path = "/cb/home/clairez/ws/depth_mup/lm-evaluation-harness/plot_script/data.txt"
    data_df = read_txt_to_df(data_path)

    models_to_plot = [
        "LD_300M",
        "LD_500M",
        "LD_900M",	
        "LD_1.8B",	
        "LD_3.8B",
        # "LD_SPJ_300M",	
        # "LD_SPJ_900M",	
        "OLMo-1B-hf",
        "OLMo-7B",
        "OLMo-2-0425-1B",
        "OLMo-2-1124-7B",
        "OLMo-2-1124-13B",	
        "OLMo-2-0325-32B",	
        "gemma-3-1b-pt",
        "gemma-3-4b-pt",
        "gemma-3-12b-pt",
        "gemma-3-27b-pt",
        "gemma-3-1b-pt+forward",
        "gemma-3-1b-pt+forward+teacher",
        "gemma-3-4b-pt+forward",
        "gemma-3-4b-pt+forward+teacher",
        "gemma-3-12b-pt+forward",
        "gemma-3-12b-pt+forward+teacher",
        "gemma-2-2b",
        "gemma-2-9b",
        "gemma-2-27b",
        "gemma-2-2b+forward",
        "gemma-2-2b+forward+teacher",
        "gemma-2-9b+forward",
        "gemma-2-9b+forward+teacher",
        # "phi-1",
        # "phi-1_5",
        # "phi-2",
        # "phi-4",
        "huggyllama/llama-7b",
        "huggyllama/llama-13b",
        "Llama-2-7b-hf",
        "Llama-2-13b-hf",
        # "Cerebras-GPT-111M",
        # "Cerebras-GPT-256M",
        # "Cerebras-GPT-590M",
        "Cerebras-GPT-1.3B",
        "Cerebras-GPT-2.7B",
        "Cerebras-GPT-6.7B",
        "Cerebras-GPT-13B",
        "btlm-3b-8k-base",
        "SmolLM2-135M",
        "SmolLM2-360M",
        "SmolLM2-1.7B",
        "SmolLM-135M",
        "SmolLM-360M",
        "SmolLM-1.7B",
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
        "Zamba2-1.2B",
        "Zamba2-2.7B",
        "Zamba2-7B",
        "Meta-Llama-3-8B",
        "Llama-3.1-8B",
        "Llama-3.2-1B",
        "Llama-3.2-3B",
    ]

    tasks_to_average = [
        "arc_c",
        "arc_e",
        "boolq", #
        "hellaswag",
        "piqa",
        "siqa", #
        "winogrande",
        "mmlu"
    ]

    plot_average_scores(
        data_df,
        models_to_plot,
        tasks_to_average,
        x_value="flops(new)",
        output_path="average_scores_vs_flops_new_w_mmlu.png",
        adjust_20tpp=False,
    )