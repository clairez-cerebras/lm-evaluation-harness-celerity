import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
plt.rcParams.update({
    'font.size': 16,          
    'axes.labelsize': 16,     
    'axes.titlesize': 18,     
    'legend.fontsize': 14,    
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14     
})
scatter_text_size = 16
scatter_size = 60

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
        # "gemma-3-1b-pt",
        # "gemma-3-4b-pt",
        # "gemma-3-12b-pt",
        # "gemma-3-27b-pt",
        # "gemma-3-1b-pt+forward",
        # "gemma-3-1b-pt+forward+teacher",
        # "gemma-3-4b-pt+forward",
        # "gemma-3-4b-pt+forward+teacher",
        # "gemma-3-12b-pt+forward",
        # "gemma-3-12b-pt+forward+teacher",
        # "gemma-2-2b",
        # "gemma-2-9b",
        # "gemma-2-27b",
        # "gemma-2-2b+forward",
        # "gemma-2-2b+forward+teacher",
        # "gemma-2-9b+forward",
        # "gemma-2-9b+forward+teacher",
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
        "Cerebras-GPT-590M",
        "Cerebras-GPT-1.3B",
        "Cerebras-GPT-2.7B",
        "Cerebras-GPT-6.7B",
        "Cerebras-GPT-13B",
        "btlm-3b-8k-base",
        # "SmolLM2-135M",
        "SmolLM2-360M",
        "SmolLM2-1.7B",
        # "SmolLM-135M",
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
        # "Qwen2.5-14B",
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


    # def function_form(x, alpha, beta, gamma):

    #     flops = x[0]
    #     tpp = x[1]
    #     return alpha * (flops) ** beta * (1.0 + gamma * (np.log(np.sqrt(20.0 / tpp)) ** 2))
    def function_form(x, alpha, gamma):

        flops = x[0]
        tpp = x[1]
        return alpha * (flops) ** (-0.055) * (1.0 + gamma * (np.log(np.sqrt(20.0 / tpp)) ** 2))
    

    params_list = []
    tokens_list = []
    flops_list = []
    tpps_list = []
    scores_list = []
    for model in models_to_plot:
        params_list.append(float(data_df.loc['num_params', model]))
        tokens_list.append(float(data_df.loc['num_tokens', model]))
        flops_list.append(float(data_df.loc['flops(final)', model]))
        tpps_list.append(float(data_df.loc['tpp', model]))
        scores_list.append(np.mean([float(data_df.loc[task, model]) for task in tasks_to_average]))
    errors_list = [100.0 - score for score in scores_list]


    # (alpha, beta, gamma), pcov = curve_fit(function_form, [flops_list, tpps_list], errors_list, p0=[30.0, -0.055, 0.0228])
    (alpha, gamma), pcov = curve_fit(function_form, [flops_list, tpps_list], errors_list, p0=[30.0, 0.0228])
    beta = -0.055

    print(f"Fitted values: alpha={alpha}, beta={beta}, gamma={gamma}; Covariance: {np.mean(np.abs(pcov))}")
    

    plt.figure(figsize=(12, 9))

    y_axis_to_plot = "flops(final)"
    plot_list = flops_list
    x_smooth = np.logspace(19.6, 24.8, 100)

    # --- Plot raw data with high transparency ---
    for i, (flop, error, tpp) in enumerate(zip(plot_list, errors_list, tpps_list)):
        raw_color, _ = get_colors(tpp)
        if i == 0:
            plt.scatter(flop, error, color=raw_color, alpha=0.1, label="Raw Error (TPP < 100: orange, 100 <= TPP < 1000: green, TPP >= 1000: purple)")
        else:
            plt.scatter(flop, error, color=raw_color, alpha=0.1)
    

    # --- Plot fitted data with less transparency, darker colors ---
    fitted_errors = alpha * np.array(plot_list) ** beta * (1.0 + gamma * (np.log(np.sqrt(20.0 / np.array(tpps_list))) ** 2))
    for i, (flop, fit_error, tpp) in enumerate(zip(plot_list, fitted_errors, tpps_list)):
        _, fitted_color = get_colors(tpp)
        if i == 0:
            plt.scatter(flop, fit_error, color=fitted_color, alpha=0.9, label="Fitted Error (TPP < 100: darkorange, 100 <= TPP < 1000: darkgreen, TPP >= 1000: indigo)")
        else:
            plt.scatter(flop, fit_error, color=fitted_color, alpha=0.9)

    # --- Calculate MSE between raw and fitted errors ---
    mse = np.mean((np.array(errors_list) - fitted_errors) ** 2)


    # --- Plot adjusted lines using the same color mapping ---
    plt.plot(x_smooth, alpha * x_smooth ** beta * (1.0 + gamma * (np.log(np.sqrt(20.0 / 20.0)) ** 2)), 
        color="red", linestyle='dashed', linewidth=1.0, label=f'Error_CE = 100 - {alpha:.3f} * FLOPs^{beta:.3f} * (1 + {gamma:.3f} * (ln(sqrt(20/TPP))^2))')

    # For tpp=5 (which is <100)
    y_smooth_5tpp = alpha * x_smooth ** beta * (1.0 + gamma * (np.log(np.sqrt(20.0 / 5.0)) ** 2))
    plt.plot(x_smooth, y_smooth_5tpp, color="orange", linestyle='dotted', linewidth=1.0, label='Error_CE adjusted for 5tpp')

    # For tpp=200 (between 100 and 1000)
    y_smooth_200tpp = alpha * x_smooth ** beta * (1.0 + gamma * (np.log(np.sqrt(20.0 / 200.0)) ** 2))
    plt.plot(x_smooth, y_smooth_200tpp, color="darkgreen", linestyle='dotted', linewidth=1.0, label='Error_CE adjusted for 200tpp')

    # For tpp=2000 (>=1000)
    y_smooth_2000tpp = alpha * x_smooth ** beta * (1.0 + gamma * (np.log(np.sqrt(20.0 / 2000.0)) ** 2))
    plt.plot(x_smooth, y_smooth_2000tpp, color="indigo", linestyle='dotted', linewidth=1.0, label='Error_CE adjusted for 2000tpp')


    # --- Plot models_to_test data for comparison ---
    models_test_raw = []
    models_test_pred = []
    flops_test = []
    tpps_test = []
    for model in models_to_test:
        flops_val = float(data_df.loc[y_axis_to_plot, model])
        tpp_val = float(data_df.loc['tpp', model])
        raw_score = np.mean([float(data_df.loc[task, model]) for task in tasks_to_average])
        raw_error = 100.0 - raw_score
        flops_test.append(flops_val)
        tpps_test.append(tpp_val)
        models_test_raw.append(raw_error)
        pred_error = alpha * flops_val**beta * (1.0 + gamma * (np.log(np.sqrt(20.0 / tpp_val)) ** 2))
        models_test_pred.append(pred_error)

    # Plot test models' raw errors (using a distinct marker, e.g., stars)
    plt.scatter(flops_test, models_test_raw, color="black", marker="*", s=100, label="Test Model Raw error (Jais 590M, 1.3B, 6.7B)")

    # Plot test models' predicted errors (using a different marker, e.g., diamonds)
    plt.scatter(flops_test, models_test_pred, color="black", marker="D", s=100, label="Test Model Predicted error (Jais 590M, 1.3B, 6.7B)")

    
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='both', linestyle=':', linewidth=0.5)
    # plt.xlim(None, 3e24)
    plt.xlabel(f"{y_axis_to_plot}")
    plt.ylabel("Average Error")
    plt.legend(loc='lower left', fontsize=10)
    plt.title(f'Downstream Function Fit: Average Error vs {y_axis_to_plot} \
                \n {len(models_to_plot)} Models, {len(tasks_to_average)} Tasks: {", ".join(tasks_to_average)} \
                \n Fitted equation: {alpha:.3f} * FLOPs^{beta:.3f} * (1 + {gamma:.3f} * (ln(sqrt(20/TPP))^2)) \
                \n RMSE between raw and fitted accuracies: {np.sqrt(mse):.2f}', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"downstream_{y_axis_to_plot}_fit_test_error_new.png")
