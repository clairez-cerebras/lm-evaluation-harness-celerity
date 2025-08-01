from scipy.optimize import root
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    'font.size': 18,          
    'axes.labelsize': 16,     
    'axes.titlesize': 20,     
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
celerity_models = ["Celerity-300M", "Celerity-500M", "Celerity-900M", "Celerity-1.8B", "Celerity-3.8B"][-3:]
celerity_params = [300E6, 500E6, 900E6, 1.8E9, 3.8E9][-3:]
celerity_flops = [1.47E+20, 5.15E+20, 1.68E+21, 6.54E+21, 2.89E+22][-3:]
celerity_loss = [2.967886355, 2.757746158, 2.596133299, 2.498397318, 2.291125278][-3:]
print(celerity_models)

def N_opt_C(C):
    return G * (C / 6.0) ** (beta / (alpha + beta))


def D_opt_C(C):
    return G ** (-1) * (C / 6.0) ** (alpha / (alpha + beta))


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


def kD_from_kN(kN, A, N_opt, alpha, B, beta, D_opt):
    # breakpoint()
    tmp = 1 - (kN ** (-alpha) - 1) * (A / B) * N_opt ** (-alpha) / (D_opt) ** (-beta)
    sol = tmp ** (-1 / beta)
    return sol


def kD_from_kN_G(kN, G, alpha, beta):
    # breakpoint()
    tmp = 1 - (kN ** (-alpha) - 1) * (A / B) * G ** (-alpha * beta)
    sol = tmp ** (-1 / beta)
    return sol


def compute_overhead(kN, kD):
    return kN * kD



kN_list = np.logspace(-0.5, np.log10(3), 100)


cmap = plt.get_cmap('coolwarm')


# for idx in range(len(celerity_models)):
    
#     fig, ax = plt.subplots(figsize=(10, 6))
#     points = []
#     tpp_values = []

#     N_opt = N_opt_C(celerity_flops[idx])
#     D_opt = D_opt_C(celerity_flops[idx])
#     loss = celerity_loss[idx]
#     title = celerity_models[idx]
#     plot_color = "maroon"

#     for kN in kN_list:
#         kD = kD_from_kN_G(kN, G, alpha, beta)
#         overhead = compute_overhead(kN, kD)
#         points.append([kN, overhead])
#         tpp_values.append(kD * D_opt / N_opt)

#     norm = Normalize(vmin=min(tpp_values), vmax=max(tpp_values)) 

#     points = np.array(points)
#     tpp_values = np.array(tpp_values)


#     # Reshape points for LineCollection
#     points_reshaped = points.reshape(-1, 1, 2)
#     segments = np.concatenate([points_reshaped[:-1], points_reshaped[1:]], axis=1)

#     # Create a LineCollection with a gradient color
#     lc = LineCollection(segments, cmap=cmap, norm=plt.matplotlib.colors.LogNorm())
#     lc.set_array(tpp_values)
#     lc.set_linewidth(2)
#     ax.add_collection(lc)

#     target_tpp = 234
#     idx_star = np.argmin(np.abs(tpp_values - target_tpp))
#     x_star, y_star = points[idx_star]
#     # draw a star
#     ax.scatter(x_star, y_star, marker='*', color='red', s=150, zorder=5)
#     # label with (x,y)
#     ax.text(
#         x_star * 1.05, y_star * 1.05,
#         rf"($k_{{N}}=${x_star:.2f}, $C_{{mult}}=${y_star:.2f})",
#         color='red', fontsize=14
#     )

#     mid_idx = int(len(points) * 0.01)
#     x_mid, y_mid = points[mid_idx]
#     ax.annotate(
#         f"SlimPJ Loss = {loss:.2f}",
#         xy=(x_mid, y_mid),
#         xytext=(x_mid*5.0, y_mid),           
#         arrowprops=dict(arrowstyle="->", color=plot_color),
#         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=plot_color)
#     )

#     sm = ScalarMappable(norm=norm, cmap=cmap)
#     sm.set_array(tpp_values)



#     #  Create a proxy artist for the legend
#     plt.Line2D([0], [0], color=cmap(0.8), lw=4, label=f"SlimPajama Loss = {loss:.2f}")


#     ax.autoscale()
#     ax.set_xlabel(r"Multiple of $N_{opt}$")
#     ax.set_ylabel(r"Multiple of $C$")
#     divider = make_axes_locatable(ax)
#     # ax.legend(handles=legend_handles) #, title="Final Loss on SlimPajama")
#     plt.title(
#         "Compute and Model Parameter Trade-off\n"
#         r"$N_{opt}$ $D_{opt}$"
#         f" therefore TPP calculated from {title} compute")
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="2%", pad=0.1)

#     # draw the colorbar for your ScalarMappable 'sm'
#     cb = fig.colorbar(sm, cax=cax)
#     cb.set_label("TPP")

#     plt.savefig(f"isoflops_compute_param_{title}.png")
# # ========== END OF PLOT ========== #
    
# ========== Plot All 5 Iso-flops Curves as Subplots ==========
fig, axes = plt.subplots(
    nrows=1, ncols=3,
    figsize=(8 * 3, 8),
    sharex=True, sharey=True
)

for idx, ax in enumerate(axes):
    model = celerity_models[idx]
    params = celerity_params[idx]
    flops = celerity_flops[idx]
    loss = celerity_loss[idx]

    # compute iso-flops optimal D
    D_opt = D_opt_C(flops)
    N_opt = N_opt_C(flops)

    # build (kN, overhead) & tpp arrays
    points, tpp_values = [], []
    for kN in kN_list:
        # kD = kD_from_kN(kN, A, N_opt, alpha, B, beta, D_opt)
        kD = kD_from_kN_G(kN, G, alpha, beta)
        overhead = compute_overhead(kN, kD)
        points.append([kN, overhead])
        tpp_values.append(kD * D_opt / N_opt)

    points = np.array(points)
    tpp_values = np.array(tpp_values)
    norm = Normalize(vmin=tpp_values.min(), vmax=tpp_values.max())

    # one colormap per model
    cmap = plt.get_cmap('coolwarm')

    # gradient line
    pts = points.reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(tpp_values)
    lc.set_linewidth(2)
    ax.add_collection(lc)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(tpp_values)

    # use an AxesDivider to append a small axis on the right for the cbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cb = fig.colorbar(sm, cax=cax)
    vmin, vmax = tpp_values.min(), tpp_values.max()
    ticks = np.linspace(vmin, vmax, num=5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([round(t) for t in ticks])
    cb.ax.xaxis.set_label_position('bottom')
    cb.ax.set_xlabel("TPP", fontsize=12)

    # star at TPP = 234
    target_tpp = 234
    i_star = np.argmin(np.abs(tpp_values - target_tpp))
    x_star, y_star = points[i_star]
    ax.scatter(x_star, y_star, marker='*', color='red', s=150, zorder=5)
    ax.axhline(y=y_star,
            color='red',
            linestyle=':',
            linewidth=1,
            zorder=4)
    ax.text(
        x_star * 1.05, y_star * 1.05,
        rf"TPP=234, $k_N=${x_star:.2f}, $k_C=${y_star:.2f}",
        color='red'
    )

    # annotate loss
    mid = len(points) // 2
    x_mid, y_mid = points[mid]
    color_mid = cmap(lc.norm(tpp_values[mid]))
    ax.annotate(
        f"SlimPJ Loss = {loss:.2f}",
        xy=(x_mid, y_mid),
        xytext=(x_mid * 1.5, y_mid * 1.2),
        arrowprops=dict(arrowstyle="->", color=color_mid),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_mid)
    )

    if idx == 0:
        ax.set_ylabel(r"$k_C$: Multiple of C")
    ax.set_xlabel(r"$k_N$: Multiple of $N_{opt}$")
    ax.set_title(rf"$N_{{\mathrm{{opt}}}}, D_{{\mathrm{{opt}}}}, TPP$ from $C=F_{{\mathrm{{{model}}}}}$")

    ax.autoscale()

# common X label and layout adjustments
fig.suptitle("Compute vs. Parameter Trade-off Under Different Celerity Model Compute Budgets", y=0.92)
fig.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig("isoflops_compute_param_all.png")