import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import Normalize
matplotlib.use("Agg")

from model import (
    Params, 
    base_model, 
    evo_model, 
    simulate, 
    simulate_rate_stress, 
    rate_stress_and_switch, 
    simulate_with_extinction
    )

OUT = "figures"   # name of folder to store the figures
Path(OUT).mkdir(exist_ok=True, parents=True)
# plot style 
plt.rcParams.update({"figure.dpi": 140,"savefig.dpi": 200,"font.size": 10.5,"axes.spines.top": False,"axes.spines.right": False,"axes.titlesize": 11.5,"axes.titleweight": "bold",})
COL_A = "#0eaad5"  # species A
COL_B = "#17902B"  # species B / producer
COL_C = "#e2711a"   # cheater
COL_P = "#DCC200"   # product / function
COL_BAD = "#c0392b"
COL_GOOD = "#2f8f52"
COL_SHARED = "#555555"

def style_ax(ax):
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")
    ax.tick_params(colors="#444")

def style_twinx(ax2):
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#444")
    ax2.tick_params(axis="y", colors="#444")

# FIGURE 0 — baseline community
def fig0_baseline():
    p = Params(Rcap=0.3)
    y0 = [1, 0.5, 0.1, 0.5, 0.0]
    sol = simulate(base_model, y0, (0, 150), p, n=1500)
    t = sol.t
    R, A, M, B, P = sol.y

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    ax = axes[0]
    ax.plot(t, A, color=COL_A, lw=2, label="Species A")
    ax.plot(t, B, color=COL_B, lw=2, label="Species B")
    ax.set_xlabel("time"); ax.set_ylabel("biomass")
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax)
    ax = axes[1]
    ax.plot(t, P, color=COL_P, lw=2)
    ax.set_xlabel("time"); ax.set_ylabel("product P")
    style_ax(ax)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig0_baseline.pdf", bbox_inches="tight")
    plt.close(fig)


# FIGURE 1a — three instabilities
def fig1a_threeInstabilities():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    # a. evolutionary: cheater invasion 
    p = Params(Rcap=0.3, coupled_production=False)
    y0 = [0.5, 0.5, 0.01, 0.49, 0.01, 0.0]
    sol = simulate(evo_model, y0, (0, 400), p, n=1500)
    t = sol.t
    Bp, Bc, P = sol.y[3], sol.y[4], sol.y[5]
    frac = Bp / (Bp + Bc + 1e-15)
    ax = axes[0]
    ax.plot(t, frac, color=COL_B, lw=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("time")
    ax.set_ylabel("fraction of producers", color=COL_B)
    ax2 = ax.twinx()
    ax2.plot(t, P, color=COL_P, lw=1.6, ls="--")
    ax2.set_ylabel("product P", color=COL_P)
    ax2.spines["top"].set_visible(False)
    ax.set_title("Evolutionary\n(cheaters invade)")
    style_twinx(ax)

    # b. ecological: paradox of enrichment
    p = Params(Rcap=2.912121)
    y0 = [1.5, 0.5, 0.1, 0.5, 0.0]

    t, y = simulate_with_extinction(base_model, y0, (0, 400), p, n=1000)
    B, P = y[3],y[4]
    ax = axes[1]
    ax.plot(t, B, color=COL_B, lw=1.5)
    ax.axhline(0, color=COL_BAD, lw=1, ls=":")
    ax.set_xlabel("time"); ax.set_ylabel("B biomass")
    ax2 = ax.twinx()
    ax2.plot(t, P, color=COL_P, lw=1.6, ls="--")
    ax2.set_ylabel("product P", color=COL_P)
    ax2.spines["top"].set_visible(False)
    ax.set_title("Ecological\n(paradox of enrichment)")
    style_ax(ax)

    # c. Increase death rate for B
    y0 = [0.5, 0.5, 0.1, 0.5, 0.0]
    p = Params(Rcap=0.3, Ballee=0.01, homeostasis=0, costM=0)
    t, y = simulate_rate_stress(base_model, y0, 400, p, stress_start=150, stress_end=160, rate_name="dB", stress_value=0.5)

    B, P = y[3], y[4]
    ax = axes[2]
    ax.plot(t, B, color=COL_B, lw=2)
    ax.axvline(150, color="grey", lw=1, ls="--")
    ax.axvline(160, color="grey", lw=1, ls="--")
    ax.text(150, ax.get_ylim()[1]*0.05, " start", fontsize=8, color="grey")
    ax.text(150, ax.get_ylim()[1]*0.05, " end", fontsize=8, color="grey")
    ax.set_xlabel("time"); ax.set_ylabel("species B biomass")
    ax2 = ax.twinx()
    ax2.plot(t, P, color=COL_P, lw=1.6, ls="--")
    ax2.set_ylabel("product P", color=COL_P)
    ax2.spines["top"].set_visible(False)
    ax.set_title(f"Perturbation\n(death rate of B is tripled)")
    style_ax(ax)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1a_three_instabilities.pdf", bbox_inches="tight")
    plt.close(fig)


# FIGURE 1b — three interventions
def fig1b_threeInterventions():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    # a. evolutionary fix: obligate coupling
    p = Params(Rcap=0.3, coupled_production=True)
    y0 = [1.5, 0.5, 0.01, 0.49, 0.01, 0.0]
    sol = simulate(evo_model, y0, (0, 400), p, n=1500)
    t = sol.t
    Bp, Bc, P = sol.y[3], sol.y[4], sol.y[5]
    frac = Bp / (Bp + Bc + 1e-15)
    ax = axes[0]
    ax.plot(t, frac, color=COL_GOOD, lw=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("generations (time)")
    ax.set_ylabel("fraction of B that are producers", color=COL_GOOD)
    ax2 = ax.twinx()
    ax2.plot(t, P, color=COL_P, lw=1.6, ls="--")
    ax2.set_ylabel("product P", color=COL_P)
    ax2.spines["top"].set_visible(False)
    ax.set_title("Evolutionary\n(coupling)")
    style_twinx(ax)

    # b. ecological fix: negative self-regulation
    p = Params(Rcap=2.912121,selfreg=0.0152)
    y0 = [1.5, 0.5, 0.1, 0.5, 0.0]
    t, y = simulate_with_extinction(base_model, y0, (0, 400), p)
    B, P = y[3], y[4]
    ax = axes[1]
    ax.plot(t, B, color=COL_GOOD, lw=1.5)
    ax.axhline(0, color=COL_BAD, lw=1, ls=":")
    ax.set_xlabel("time"); ax.set_ylabel("species B biomass")
    ax2 = ax.twinx()
    ax2.plot(t, P, color=COL_P, lw=1.6, ls="--")
    ax2.set_ylabel("product P", color=COL_P)
    ax2.spines["top"].set_visible(False)
    ax.set_title("Ecological\n(self-regulation)")
    style_ax(ax)

    # c. Increase death rate for a while
    y0 = [0.5, 0.5, 0.1, 0.5, 0.0]
    p = Params(Rcap=0.3, Ballee=0.01, homeostasis=0.5, costM=0)
    t, y = simulate_rate_stress(base_model, y0, 400, p, stress_start=150, stress_end=160, rate_name="dB", stress_value=0.5)

    B, P = y[3], y[4]
    ax = axes[2]
    ax.plot(t, B, color=COL_B, lw=2)
    ax.axvline(150, color="grey", lw=1, ls="--")
    ax.axvline(160, color="grey", lw=1, ls="--")
    ax.text(150, ax.get_ylim()[1]*0.05, " start", fontsize=8, color="grey")
    ax.text(150, ax.get_ylim()[1]*0.05, " end", fontsize=8, color="grey")
    ax.set_xlabel("time"); ax.set_ylabel("species B biomass")
    ax2 = ax.twinx()
    ax2.plot(t, P, color=COL_P, lw=1.6, ls="--")
    ax2.set_ylabel("product P", color=COL_P)
    ax2.spines["top"].set_visible(False)
    ax.set_title(f"Perturbation\n(death rate of B is increased)")
    style_ax(ax)
        
    fig.suptitle("Three interventions", y=1.06, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1b_three_interventions.pdf", bbox_inches="tight")
    plt.close(fig)

def fig2a_envInstabilityOnly():
    #only env instability and env intervention
    #intervention applied only after stress

    y0 = [0.5, 0.5, 0.01, 0.5, 0.0]
    stress_start, stress_end, t_total = 150, 160, 400
    dB_stress = 0.5  

    conditions = [
        ("Feedback off",Params(Rcap=0.5, Ballee=0.01, homeostasis=0), COL_BAD),
        ("Feedback on (recovery only)",Params(Rcap=0.5, Ballee=0.01, homeostasis=0.9), COL_GOOD),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)

    for ax, (title, p_post, color) in zip(axes, conditions):
        p_pre = Params(Rcap=0.3, Ballee=0.005, homeostasis=0, costM=0)  # fresh copy each time
        t, y = rate_stress_and_switch(base_model, y0, t_total, p_pre, p_post, stress_start, stress_end, "dB", dB_stress)
        B,P = y[3],y[4]

        pre_mask = t <= stress_end
        post_mask = t >= stress_end
        ax.plot(t[pre_mask], B[pre_mask], color=COL_SHARED, lw=1.8, label="pre- and during-stress (identical in both panels)")
        ax.plot(t[post_mask], B[post_mask], color=color, lw=2.2, label="recovery phase")
        ax.axvspan(stress_start, stress_end, color="grey", alpha=0.15)

        b_at_switch = B[post_mask][0]
        ax.scatter([stress_end], [b_at_switch], color="black", zorder=5, s=25)
        
        ax.set_xlabel("time")

        ax2 = ax.twinx()
        ax2.plot(t[pre_mask], P[pre_mask], color=COL_P, lw=1.6, ls="--")
        ax2.plot(t[post_mask], P[post_mask], color=COL_P, lw=2.2)
        ax2.set_ylabel("product P", color=COL_P)
        ax2.spines["top"].set_visible(False)
        ax.set_title(title)
        style_ax(ax)

    axes[0].set_ylabel("species B biomass")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2a_envInstabilityOnly.pdf", bbox_inches="tight")
    plt.close(fig)

def fig2b_AllInstabilitiesAllInterventions():
    #The three instabilities are introduced simultaneously
    #comparison of applying only none vs all interventions

    y0 = [0.5, 0.5, 0.1, 0.49, 0.01, 0.0]
    STRESS = dict(Rcap=0.3, Ballee=0.01)
    stress_start, stress_end, dB_stress, t_total = 250, 260, 0.6, 500

    #no interventions
    p_frag = Params(**STRESS, coupled_production=False, cost=0.05, selfreg=0, homeostasis=0)
    t1, y1 = simulate_rate_stress(evo_model, y0, t_total, p_frag, stress_start, stress_end, "dB", dB_stress)

    #three interventions
    p_fix = Params(**STRESS, coupled_production=True, cost=0.05, selfreg=0.01, homeostasis=0.5)
    t2, y2 = simulate_rate_stress(evo_model, y0, t_total, p_fix, stress_start, stress_end, "dB", dB_stress)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex="col")

    B1 = y1[3] + y1[4]
    B2 = y2[3] + y2[4]
    P1, P2 = y1[5], y2[5]

    ax = axes[0, 0]
    ax.plot(t1, B1, color=COL_B, lw=1.6)
    ax.axvspan(stress_start, stress_end, color="grey", alpha=0.15)
    ax.set_ylabel("total B biomass")
    ax.set_title("No fixes")
    style_ax(ax)

    ax = axes[0, 1]
    ax.plot(t2, B2, color=COL_B, lw=1.6)
    ax.axvspan(stress_start, stress_end, color="grey", alpha=0.15)
    ax.set_title("Three fixes together")
    style_ax(ax)

    ax = axes[1, 0]
    ax.plot(t1, P1, color=COL_P, lw=1.6)
    ax.axvspan(stress_start, stress_end, color="grey", alpha=0.15)
    ax.set_xlabel("time"); ax.set_ylabel("product P")
    style_ax(ax)

    ax = axes[1, 1]
    ax.plot(t2, P2, color=COL_P, lw=1.6)
    ax.axvspan(stress_start, stress_end, color="grey", alpha=0.15)
    ax.set_xlabel("time")
    style_ax(ax)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2b_AllInstabilitiesAllInterventions.pdf", bbox_inches="tight")
    plt.close(fig)

def fig2c_AllInstabilitiesSingleIntervention():
    #
    y0 = [0.5, 0.5, 0.1, 0.49, 0.01, 0.0]
    STRESS = dict(Rcap=0.3, Ballee=0.01)
    stress_start, stress_end, dB_stress, t_total = 250, 260, 0.6, 500

    conditions = [
        ("A. Evolutionary only\n(coupling)", dict(coupled_production=True, cost=0.1, selfreg=0, homeostasis=0)),
        ("B. Ecological only\n(self regulation)", dict(coupled_production=False, cost=0.1, selfreg=0.01, homeostasis=0)),
        ("C. Perturbation only", dict(coupled_production=False, cost=0.1, selfreg=0, homeostasis=0.5)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=False)

    for ax, (title, fixkw) in zip(axes, conditions):
        p = Params(**STRESS, **fixkw)
        t, y = simulate_rate_stress(evo_model, y0, t_total, p, stress_start, stress_end, "dB", dB_stress)
        B = y[3] + y[4]
        P = y[5]

        ax.plot(t, B, color=COL_B, lw=1.6, label="total B biomass")
        ax.axvspan(stress_start, stress_end, color="grey", alpha=0.15)
        ax.set_xlabel("time")
        ax.set_ylabel("total B biomass", color=COL_B)
        ax.set_title(title)
        style_twinx(ax)

        ax2 = ax.twinx()
        ax2.plot(t, P, color=COL_P, lw=1.6, ls="--", label="product P")
        ax2.set_ylabel("product P", color=COL_P)
        ax2.spines["top"].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2c_AllInstabilitiesSingleIntervention.pdf", bbox_inches="tight")
    plt.close(fig)

def fig3_phaseDiagram():
    #comparison of production for different combinations of h,f,s
    #combinations used: s=2 or s=0, Rcap=1.5 or Rcap=1.0
    y0 = [1, 0.5, 0.1, 0.49, 0.01, 0.0]
    STRESS = dict(Rcap=1.5, Ballee=0.0001)
    #STRESS = dict(Rcap=1.0, Ballee=0.0001)
    stress_start, stress_end, dB_stress = 250, 260, 0.9

    selfreg_panels = [0, 2]
    cost_vals = np.linspace(0, 1, 50)
    homeo_vals = np.linspace(0, 1, 50)

    all_grids = []

    for sr in selfreg_panels:
        P_grid = np.zeros((len(cost_vals), len(homeo_vals)))

        for j, homeo in enumerate(homeo_vals):
            for i, cost in enumerate(cost_vals):

                p = Params(**STRESS, coupled_production=True, cost=cost, selfreg=sr, homeostasis=homeo)
                sol = simulate(evo_model, y0, (0, 250), p, n=800)
                B_pre = sol.y[3, -1] + sol.y[4, -1]

                if B_pre < 1e-3:
                    P_grid[i, j] = 0.0
                    continue

                t, y = simulate_rate_stress(evo_model, y0, 500, p, stress_start, stress_end, "dB", dB_stress, n=800)

                B_final = (y[3] + y[4])[-1]
                P_grid[i, j] = y[5, -1] if B_final > 1e-3 else 0.0

        all_grids.append(P_grid)

    vmax = max(g.max() for g in all_grids)
    norm = Normalize(vmin=0, vmax=vmax)

    cmap = sns.color_palette("Blues", as_cmap=True)
    fig, axes = plt.subplots(1,len(selfreg_panels),figsize=(17, 4.6),sharey=True)

    for ax, sr, P_grid in zip(axes, selfreg_panels, all_grids):
        P_plot = np.maximum(P_grid, 0)
        mesh = ax.pcolormesh(homeo_vals, cost_vals, P_plot, cmap=cmap, norm=norm, shading="nearest")

        ax.set_xlabel("homeostasis")
        ax.set_title(f"self-regulation = {sr}")
        style_ax(ax)

    axes[0].set_ylabel("cost f")
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(f"post-shock steady-state P ")

    fig.savefig(f"{OUT}/fig3_phaseDiagram.pdf", bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    fig0_baseline()
    fig1a_threeInstabilities()
    fig1b_threeInterventions()
    fig2a_envInstabilityOnly()
    fig2b_AllInstabilitiesAllInterventions()
    fig2c_AllInstabilitiesSingleIntervention()
    fig3_phaseDiagram()
    print("done")

