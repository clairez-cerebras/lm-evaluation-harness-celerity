from scipy.optimize import root
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    'font.size': 16,          
    'axes.labelsize': 16,     
    'axes.titlesize': 18,     
    'legend.fontsize': 14,    
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14     
})

E = 1.69
A = 406.4
B = 410.7
alpha = 0.34
beta = 0.28
G = (alpha * A / beta / B) ** (1.0 / (alpha+beta))

TPP_opt = 20.0
celerity_models = ["Celerity-300M", "Celerity-500M", "Celerity-900M", "Celerity-1.8B", "Celerity-3.8B"]
celerity_params = [300E6, 500E6, 900E6, 1.8E9, 3.8E9]
celerity_flops = [1.47E+20, 5.15E+20, 1.68E+21, 6.54E+21, 2.89E+22]
celerity_loss = [2.967886355, 2.757746158, 2.596133299, 2.498397318, 2.291125278]


def N_opt_C(C):
    return G * (C / 6.0) ** (beta / (alpha + beta))


def solve_N_opt_with_L(L, E, A, alpha, B, beta, TPP_opt, p0):
    def f(x):
        return E + A / (x ** alpha) + B / ((x * TPP_opt) ** beta) - L
    sol = root(f, p0)
    return sol


def kT_from_kN(kN, A, N_opt, alpha, B, beta, TPP_opt):
    # breakpoint()
    tmp = 1 - (kN ** (-alpha) - 1) * (A / B) * N_opt ** (-alpha) / (N_opt * TPP_opt) ** (-beta)
    # print(f"tmp: {tmp}; kN: {kN}")
    sol = tmp ** (-1 / beta) / kN
    return sol


def compute_overhead(kN, kT):
    return kN ** 2 * kT



kN_list = np.logspace(-0.8, np.log10(5), 100)
# ========== Plot IsoLoss ========== #
fig, ax = plt.subplots(figsize=(10, 6))

# Define a list of colormaps, one for each model
cmaps = [plt.get_cmap('viridis'), plt.get_cmap('cool'), plt.get_cmap('winter'), plt.get_cmap('bwr'), plt.get_cmap('coolwarm')]
# You might need to adjust vmin and vmax based on your expected kT * TPP_opt range

legend_handles = []
sm_list = []  

for i, (model, params, flops, loss) in enumerate(zip(celerity_models, celerity_params, celerity_flops, celerity_loss)):
    if i == 0 or i == 4:
        N_opt = solve_N_opt_with_L(loss, E, A, alpha, B, beta, TPP_opt, params).x[0]
        points = []
        tpp_values = []

        for kN in kN_list:
            kT = kT_from_kN(kN, A, N_opt, alpha, B, beta, TPP_opt)
            overhead = compute_overhead(kN, kT)
            points.append([kN, overhead])
            tpp_values.append(kT * TPP_opt)
            # print(kN, kT)
        
        norm = Normalize(vmin=min(tpp_values), vmax=max(tpp_values)) 

        points = np.array(points)
        tpp_values = np.array(tpp_values)
        
        cmap = cmaps[i % len(cmaps)] # Cycle through colormaps
        
        # Reshape points for LineCollection
        points_reshaped = points.reshape(-1, 1, 2)
        segments = np.concatenate([points_reshaped[:-1], points_reshaped[1:]], axis=1)
        
        # Create a LineCollection with a gradient color
        lc = LineCollection(segments, cmap=cmap, norm=plt.matplotlib.colors.LogNorm())
        lc.set_array(tpp_values)
        lc.set_linewidth(2)
        ax.add_collection(lc)

        mid_idx = int(len(points) * 0.01)
        x_mid, y_mid = points[mid_idx]
        if i == 0:
            plot_color = "gold"
        else:
            plot_color = "maroon"
        ax.annotate(
            f"SlimPJ Loss = {loss:.2f}",
            xy=(x_mid, y_mid),
            xytext=(x_mid*5.0, y_mid),           
            arrowprops=dict(arrowstyle="->", color=plot_color),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=plot_color)
        )

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(tpp_values)
        sm_list.append((sm, model))
        
        
        #  Create a proxy artist for the legend
        legend_handles.append(plt.Line2D([0], [0], color=cmap(0.8), lw=4, label=f"SlimPajama Loss = {loss:.2f}"))


ax.autoscale()
ax.set_xlabel(r"Multiple of $N_{opt}$")
ax.set_ylabel(r"Multiple of $C_{loss}$")
divider = make_axes_locatable(ax)
# ax.legend(handles=legend_handles) #, title="Final Loss on SlimPajama")
plt.title("Compute and Model Parameter Trade-off\nunder Iso-loss Constraint")
for idx, (sm, model) in enumerate(sm_list):
    pad = 0.3 + idx * 0.5   # stagger them vertically
    cax = divider.append_axes("right", size="2%", pad=pad)
    cb = fig.colorbar(sm, cax=cax)
    if idx == 1:
        cb.set_label("TPP")
plt.savefig("isoloss_compute_param.png")
# ========== END OF PLOT ========== #


# ========== PLOT Isoloss first ========== #
fig, ax = plt.subplots(figsize=(12, 9))
for loss, param in zip(celerity_loss, celerity_params):
    N_opt = solve_N_opt_with_L(loss, E, A, alpha, B, beta, TPP_opt, param).x[0]    
    overhead_list = []
    for kN in kN_list:
        kT = kT_from_kN(kN, A, N_opt, alpha, B, beta, TPP_opt)
        overhead = compute_overhead(kN, kT)
        overhead_list.append(overhead)
        # print(f"kN: {kN:.2f}, kT: {kT:.2f}, Overhead: {overhead:.2f} for Loss = {loss:.2f}")
    ax.plot(kN_list, overhead_list, label=f"Loss = {loss:.2f}")
ax.set_xlabel("kN")
ax.set_ylabel("Overhead (kN^2 * kT)")
# ax.set_xscale("log")
plt.legend()
plt.title("Overhead for IsoLoss")
plt.savefig("isoloss_overhead.png")

# ========== PLOT Isoflops next ========== #
fig, ax = plt.subplots(figsize=(12, 9))
for model, params, flops in zip(celerity_models, celerity_params, celerity_flops):
    N_opt = N_opt_C(flops)
    overhead_list = []
    for kN in kN_list:
        kT = kT_from_kN(kN, A, N_opt, alpha, B, beta, TPP_opt)
        overhead = compute_overhead(kN, kT)
        overhead_list.append(overhead)
        # print(f"Model: {model}, kN: {kN:.2f}, kT: {kT:.2f}, Overhead: {overhead:.2f}")
    ax.plot(kN_list, overhead_list, label=f"{model} (Params: {params:.0f}, Flops: {flops:.1e})", alpha=0.7)
ax.set_xlabel("kN")
ax.set_ylabel("Overhead (kN^2 * kT)")
# ax.set_xscale("log")
plt.legend()
plt.title("Overhead for IsoFlops")
plt.savefig("isoflops_overhead.png")
# ========== END OF PLOT ========== #
