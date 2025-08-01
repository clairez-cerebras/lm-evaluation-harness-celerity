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
    original_row_names = ['num_params', 'num_tokens', 'tpp', 'flops(6ND)', 'vocab_size', 'hidden_size', 'MSL', 'num_hidden_layers', 'flops(new)', 'flops(6ND-nonembed)', 'arc_c', 'arc_e', 'boolq', 'hellaswag', 'piqa', 'siqa', 'winogrande', 'mmlu']
    df.index = original_row_names

    return df


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
        "phi-1",
        "phi-1_5",
        "phi-2",
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
        # "Qwen3-0.6B-Base",
        # "Qwen3-1.7B-Base",
        # "Qwen3-4B-Base",
        # "Qwen3-8B-Base",
        # "Qwen3-14B-Base",
        # "Qwen2.5-0.5B",
        # "Qwen2.5-1.5B",
        # "Qwen2.5-3B",
        # "Qwen2.5-7B",
        # "Qwen2.5-32B",
        # "Zamba2-1.2B",
        # "Zamba2-2.7B",
        # "Zamba2-7B",
        # "Meta-Llama-3-8B",
        # "Llama-3.1-8B",
        # "Llama-3.2-1B",
        # "Llama-3.2-3B",
    ]

    models_to_test = [
        "jais-family-590m",
        "jais-family-1p3b",
        "jais-family-6p7b",
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

    def get_colors(tpp):
        if tpp < 100:
            return "orange", "darkorange"
        elif tpp < 1000:
            return "green", "darkgreen"
        else:
            return "purple", "indigo"


    def function_form(x, alpha, beta, gamma):

        T = x[0]
        P = x[1]

        return alpha * (6*T*P) ** beta * (1.0 + gamma * (np.log(np.sqrt(20.0 / (T/P))) ** 2))
    

    params_list = []
    tokens_list = []
    flops_list = []
    tpps_list = []
    scores_list = []
    for model in models_to_plot:
        params_list.append(float(data_df.loc['num_params', model]))
        tokens_list.append(float(data_df.loc['num_tokens', model]))
        flops_list.append(float(data_df.loc['flops(6ND-nonembed)', model]))
        tpps_list.append(float(data_df.loc['tpp', model]))
        scores_list.append(np.mean([float(data_df.loc[task, model]) for task in tasks_to_average]))
    errors_list = [100.0 - score for score in scores_list]


    (alpha, beta, gamma), pcov = curve_fit(function_form, [tokens_list, params_list], errors_list, p0=[4740.0, -0.09, 0.01])
    print(f"Fitted parameters: alpha={alpha}, beta={beta}, gamma={gamma}; Covariance: {np.mean(np.abs(pcov))}")
    

    plt.figure(figsize=(12, 9))

    # --- Plot raw data with high transparency ---
    for i, (flop, score, tpp) in enumerate(zip(flops_list, scores_list, tpps_list)):
        raw_color, _ = get_colors(tpp)
        if i == 0:
            plt.scatter(flop, score, color=raw_color, alpha=0.1, label="Raw Accuracy (TPP < 100: orange, 100 <= TPP < 1000: green, TPP >= 1000: purple)")
        else:
            plt.scatter(flop, score, color=raw_color, alpha=0.1)
    

    # --- Plot fitted data with less transparency, darker colors ---
    fitted_scores = 100 - alpha * np.array(flops_list) ** beta * (1.0 + gamma * (np.log(np.sqrt(20.0 / np.array(tpps_list))) ** 2))
    for i, (flop, fit_score, tpp) in enumerate(zip(flops_list, fitted_scores, tpps_list)):
        _, fitted_color = get_colors(tpp)
        if i == 0:
            plt.scatter(flop, fit_score, color=fitted_color, alpha=0.9, label="Fitted Accuracy (TPP < 100: darkorange, 100 <= TPP < 1000: darkgreen, TPP >= 1000: indigo)")
        else:
            plt.scatter(flop, fit_score, color=fitted_color, alpha=0.9)

    # --- Calculate MSE between raw and fitted scores ---
    mse = np.mean((np.array(scores_list) - fitted_scores) ** 2)


    # --- Plot adjusted lines using the same color mapping ---
    x_smooth = np.logspace(19.6, 24.8, 100)
    plt.plot(x_smooth, 100 - alpha * x_smooth ** beta * (1.0 + gamma * (np.log(np.sqrt(20.0 / 20.0)) ** 2)), 
        color="red", linestyle='dashed', linewidth=1.0, label=f'Accuracy_CE = 100 - {alpha:.3f} * FLOPs^{beta:.3f} * (1 + {gamma:.3f} * (ln(sqrt(20/TPP))^2))')

    # For tpp=5 (which is <100)
    y_smooth_5tpp = 100 - alpha * x_smooth ** beta * (1.0 + gamma * (np.log(np.sqrt(20.0 / 5.0)) ** 2))
    plt.plot(x_smooth, y_smooth_5tpp, color="orange", linestyle='dotted', linewidth=1.0, label='Accuracy_CE adjusted for 5tpp')

    # For tpp=200 (between 100 and 1000)
    y_smooth_200tpp = 100 - alpha * x_smooth ** beta * (1.0 + gamma * (np.log(np.sqrt(20.0 / 200.0)) ** 2))
    plt.plot(x_smooth, y_smooth_200tpp, color="darkgreen", linestyle='dotted', linewidth=1.0, label='Accuracy_CE adjusted for 200tpp')

    # For tpp=2000 (>=1000)
    y_smooth_2000tpp = 100 - alpha * x_smooth ** beta * (1.0 + gamma * (np.log(np.sqrt(20.0 / 2000.0)) ** 2))
    plt.plot(x_smooth, y_smooth_2000tpp, color="indigo", linestyle='dotted', linewidth=1.0, label='Accuracy_CE adjusted for 2000tpp')

    
    # --- Plot models_to_test data for comparison ---
    models_test_raw = []
    models_test_pred = []
    flops_test = []
    tpps_test = []
    for model in models_to_test:
        flops_val = float(data_df.loc['flops(6ND-nonembed)', model])
        tpp_val = float(data_df.loc['tpp', model])
        raw_score = np.mean([float(data_df.loc[task, model]) for task in tasks_to_average])
        flops_test.append(flops_val)
        tpps_test.append(tpp_val)
        models_test_raw.append(raw_score)
        pred_score = 100 - alpha * flops_val**beta * (1.0 + gamma * (np.log(np.sqrt(20.0 / tpp_val)) ** 2))
        models_test_pred.append(pred_score)

    # Plot test models' raw scores (using a distinct marker, e.g., stars)
    plt.scatter(flops_test, models_test_raw, color="black", marker="*", s=100, label="Test Model Raw Score (Jais 590M, 1.3B, 6.7B)")

    # Plot test models' predicted scores (using a different marker, e.g., diamonds)
    plt.scatter(flops_test, models_test_pred, color="black", marker="D", s=100, label="Test Model Predicted Score (Jais 590M, 1.3B, 6.7B)")

    
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='both', linestyle=':', linewidth=0.5)
    plt.xlim(None, 3e24)
    plt.xlabel("FLOPs")
    plt.ylabel("Average Accuracy")
    plt.legend(loc='lower right', fontsize=10)
    plt.title(f'Downstream Function Fit: Average Accuracy vs FLOPs(6ND-nonembed) \
                \n {len(models_to_plot)} Models, {len(tasks_to_average)} Tasks: {", ".join(tasks_to_average)} \
                \n Fitted equation: 100 - {alpha:.3f} * FLOPs^{beta:.3f} * (1 + {gamma:.3f} * (ln(sqrt(20/TPP))^2)) \
                \n RMSE between raw and fitted accuracies: {np.sqrt(mse):.2f}', fontsize=12)
    plt.tight_layout()
    plt.savefig("downstream_flop(6ND-nonembed)_fit_test.png")
