import csv
import matplotlib.pyplot as plt


results = {}
with open('/cb/home/clairez/ws/depth_mup/lm-evaluation-harness/plot_script/results.txt', 'r') as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        if row:  # skip empty lines
            key = row[0]
            results[key] = row[1:]

models = results.get("models", [])
flops = results.get("flops", [])
avg_acc = results.get("avg_acc", [])
mmlu_acc = results.get("mmlu_acc", [])

for i, f in enumerate(flops):
    print(i, "    ", f)

assert len(models) == len(flops) == len(avg_acc) == len(mmlu_acc), f"Mismatch in lengths of data lists: {len(models)}, {len(flops)}, {len(avg_acc)}, {len(mmlu_acc)}"

# Convert flops and avg_acc values to floats
flops = [float(val) for val in flops]
avg_acc = [float(val) for val in avg_acc]

plt.figure(figsize=(12, 8))

# Annotate each point with the corresponding model name
for i, (model, flop, acc) in enumerate(zip(models, flops, avg_acc)):
    if "LD" in model: 
        plt.scatter(flop, acc, color="red")
        plt.annotate(model, (flop, acc), textcoords="offset points", xytext=(5, 5), ha="center", font=dict(size=8))
    elif "gemma-2" in model:
        plt.scatter(flop, acc, color="blue")
        plt.annotate(model, (flop, acc), textcoords="offset points", xytext=(5, 5), ha="center", font=dict(size=8))
    elif "gemma-3" in model:
        plt.scatter(flop, acc, color="green")
        plt.annotate(model, (flop, acc), textcoords="offset points", xytext=(5, 5), ha="center", font=dict(size=8))
    else:
        plt.scatter(flop, acc, color="black")

plt.xlabel("FLOPs")
plt.ylabel("Average Accuracy")
plt.title("Average Accuracy vs FLOPs")
plt.xscale('log')  # Use log scale for FLOPs, if desired
plt.grid(True)
plt.savefig("avg_acc_vs_flops2.png")