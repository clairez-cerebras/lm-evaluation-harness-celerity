import csv
import pandas as pd
import numpy as np
from flask import Flask, render_template_string, Markup, request
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

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

def assign_color(model):
    if "LD" in model:
        return "LD"
    elif "gemma-2" in model:
        if "+forward+teacher" in model:
            return "gemma-2+forward+teacher"
        elif "+forward" in model:
            return "gemma-2+forward"
        else:
            return "gemma-2"
    elif "gemma-3" in model:
        if "+forward+teacher" in model:
            return "gemma-3+forward+teacher"
        elif "+forward" in model:
            return "gemma-3+forward"
        else:
            return "gemma-3"
    else:
        return "Other"

@app.route("/", methods=['GET', 'POST'])
def index():
    # Read data from data.txt
    data_path = '/cb/home/clairez/ws/depth_mup/lm-evaluation-harness/plot_script/data.txt'
    df = read_txt_to_df(data_path)
    
    # Get all available models
    all_models = df.columns.tolist()

    # Define all tasks
    all_tasks = ["arc_c", "arc_e", "boolq", "hellaswag", "piqa", "siqa", "winogrande", "mmlu"]
    
    # Default models to show (or get from form submission)
    if request.method == 'POST':
        selected_models = request.form.getlist('models')
        selected_tasks = request.form.getlist('tasks')
    else:
        # Default selection - you can customize this
        selected_models = [
            "LD_300M", "LD_500M", "LD_900M", "LD_1.8B", "LD_3.8B",
            "gemma-3-1b-pt", "gemma-3-4b-pt", "gemma-3-12b-pt", "gemma-3-27b-pt",
            "gemma-2-2b", "gemma-2-9b","gemma-2-27b"
        ]
        # Filter to only include models that actually exist in the data
        selected_models = [m for m in selected_models if m in all_models]
        selected_tasks = all_tasks
    
    # Tasks to average over
    tasks_to_average = selected_tasks
    
    # Prepare data for plotting
    models = []
    num_tokens = []
    avg_acc = []
    
    for model in selected_models:
        if model not in df.columns:
            continue
            
        # Get num_tokens and adjust by TPP
        try:
            num_tokens_val = float(df.loc["num_tokens", model])
            tpp_val = float(df.loc["tpp", model])
            
            # Calculate average score across tasks
            task_scores = []
            for task in tasks_to_average:
                if task in df.index:
                    try:
                        score = float(df.loc[task, model])
                        task_scores.append(score)
                    except (ValueError, TypeError):
                        continue
            
            if task_scores:
                avg_score = np.mean(task_scores)
                models.append(model)
                num_tokens.append(num_tokens_val)
                avg_acc.append(avg_score)
                
        except (ValueError, TypeError, KeyError):
            continue
    
    # Build a dataframe for Plotly
    plot_df = pd.DataFrame({
        "Model": models,
        "Num Tokens": num_tokens,
        "Average Accuracy": avg_acc
    })
    plot_df["Color"] = plot_df["Model"].apply(assign_color)
    
    # Create the plot
    fig = px.scatter(
        plot_df, x="Num Tokens", y="Average Accuracy",
        hover_data=["Model", "Num Tokens", "Average Accuracy"],
        title="Average Accuracy vs Num Tokens",
        log_x=True,
        log_y=True,
        color="Color",
        color_discrete_map={
            "LD": "red",
            "gemma-2": "#aec7e8",
            "gemma-2+forward": "#1f77b4",
            "gemma-2+forward+teacher": "blue",
            "gemma-3": "#66CC66",
            "gemma-3+forward": "#339933",
            "gemma-3+forward+teacher": "green",
            "Other": "black"
        }
    )
    # Adjust marker sizes
    for trace in fig.data:
        if trace.name in ["LD", "gemma-2", "gemma-2+forward", "gemma-2+forward+teacher", 
                         "gemma-3", "gemma-3+forward", "gemma-3+forward+teacher"]:
            trace.marker.size = 12
        else:
            trace.marker.size = 6
    
    
    # Get the HTML representation of the plot
    graph_html = Markup(pio.to_html(fig, full_html=False))
    
    # Create model checkboxes HTML
    model_checkboxes = ""
    for model in sorted(all_models):
        checked = "checked" if model in selected_models else ""
        model_checkboxes += f'''
        <label style="display: block; margin: 2px;">
            <input type="checkbox" name="models" value="{model}" {checked}> {model}
        </label>
        '''
    
    task_checkboxes = ""
    for task in all_tasks:
        checked = "checked" if task in selected_tasks else ""
        task_checkboxes += f'''
        <label style="display: block; margin: 2px;">
            <input type="checkbox" name="tasks" value="{task}" {checked}> {task}
        </label>
        '''
    
    # Define the HTML template with model selection
    html_template = """
    <html>
      <head>
         <title>Avg Accuracy vs Tokens</title>
         <style>
            .container { display: flex; }
            .sidebar { width: 300px; height: 600px; overflow-y: auto; padding: 20px; border-right: 1px solid #ccc; display: flex; }
            .model-section { width: 98%; padding-right: 10px; }
            .task-section { width: 50%; padding-left: 10px; }
            .content { flex: 1; padding: 20px; }
            .controls { margin-bottom: 20px; }
            button { margin: 5px; padding: 8px 15px; }
         </style>
      </head>
      <body>
         <div class="container">
            <div class="sidebar">
               <form method="post">
                  <div class="model-section">
                     <h3>Select Models:</h3>
                     <div class="controls">
                        <button type="button" onclick="selectAllModels()">Select All</button>
                        <button type="button" onclick="selectNoModels()">Select None</button>
                     </div>
                     <div id="model-list">
                        {{ model_checkboxes|safe }}
                     </div>
                  </div>
                  <div class="task-section">
                     <h3>Select Tasks:</h3>
                     <div class="controls">
                        <button type="button" onclick="selectAllTasks()">Select All</button>
                        <button type="button" onclick="selectNoTasks()">Select None</button>
                     </div>
                     <div id="task-list">
                        {{ task_checkboxes|safe }}
                     </div>
                  </div>
                  <button type="submit" style="margin-top: 20px; background-color: #4CAF50; color: white; width: 100%;">Update Plot</button>
               </form>
            </div>
            <div class="content">
               <p>Selected {{ num_selected }} models out of {{ num_total_models }} total models. Selected {{ num_tasks }} tasks out of {{ num_total_tasks}} total tasks.</p>
               {{ graph_html|safe }}
            </div>
         </div>
         
         <script>
            function selectAllModels() {
                var checkboxes = document.querySelectorAll('input[name="models"]');
                for (var i = 0; i < checkboxes.length; i++) {
                    checkboxes[i].checked = true;
                }
            }
            
            function selectNoModels() {
                var checkboxes = document.querySelectorAll('input[name="models"]');
                for (var i = 0; i < checkboxes.length; i++) {
                    checkboxes[i].checked = false;
                }
            }
            
            function selectAllTasks() {
                var checkboxes = document.querySelectorAll('input[name="tasks"]');
                for (var i = 0; i < checkboxes.length; i++) {
                    checkboxes[i].checked = true;
                }
            }
            
            function selectNoTasks() {
                var checkboxes = document.querySelectorAll('input[name="tasks"]');
                for (var i = 0; i < checkboxes.length; i++) {
                    checkboxes[i].checked = false;
                }
            }
         </script>
      </body>
    </html>
    """
    return render_template_string(
        html_template, 
        graph_html=graph_html,
        model_checkboxes=Markup(model_checkboxes),
        task_checkboxes=Markup(task_checkboxes),  
        num_selected=len(selected_models),
        num_tasks=len(selected_tasks),  
        num_total_models=len(all_models),
        num_total_tasks=len(all_tasks)
    )

if __name__ == "__main__":
    app.run(debug=True, host="clairez-dev.cerebras.aws", port=5001)