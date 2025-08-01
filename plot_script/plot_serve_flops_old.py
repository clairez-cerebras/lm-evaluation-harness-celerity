import csv
import pandas as pd
from flask import Flask, render_template_string, Markup
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

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

@app.route("/")
def index():
    results = {}
    file_path = '/cb/home/clairez/ws/depth_mup/lm-evaluation-harness/plot_script/results.txt'
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row:  # skip empty lines
                key = row[0]
                results[key] = row[1:]
    
    models = results.get("models", [])
    flops = results.get("flops", [])
    avg_acc = results.get("avg_acc", [])
    
    # Ensure data lengths match
    assert len(models) == len(flops) == len(avg_acc), f"Data length mismatch: {len(models)}, {len(flops)}, {len(avg_acc)}"
    
    # Convert flops and avg_acc values to floats
    flops = [float(val) for val in flops]
    avg_acc = [float(val) for val in avg_acc]
    
    # Build a dataframe for Plotly and assign colors based on the model name
    df = pd.DataFrame({
        "Model": models,
        "FLOPs": flops,
        "Average Accuracy": avg_acc
    })
    df["Color"] = df["Model"].apply(assign_color)
    
    # Create an interactive scatter plot with discrete color mapping
    fig = px.scatter(
        df, x="FLOPs", y="Average Accuracy",
        hover_data=["Model", "FLOPs", "Average Accuracy"],
        title="Average Accuracy vs FLOPs",
        log_x=True,
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


    for trace in fig.data:
        # These traces correspond to models with names containing "LD", "gemma-2", or "gemma-3"
        if trace.name in ["LD", "gemma-2", "gemma-2+forward", "gemma-2+forward+teacher", "gemma-3", "gemma-3+forward", "gemma-3+forward+teacher", ]:
            trace.marker.size = 12  # Larger size for highlighted models
        else:
            trace.marker.size = 6   # Normal size for others
    
    # Get the HTML representation of the plot and mark it as safe
    graph_html = Markup(pio.to_html(fig, full_html=False))
    
    # Define the HTML template embedding the graph
    html = """
    <html>
      <head>
         <title>Avg Accuracy vs FLOPs</title>
      </head>
      <body>
         {{ graph_html|safe }}
      </body>
    </html>
    """
    return render_template_string(html, graph_html=graph_html)

if __name__ == "__main__":
    app.run(debug=True, host="clairez-dev.cerebras.aws", port=5000)
