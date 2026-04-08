"""
scripts/plot_pit_diagnostics.py  (v3)
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import norm, kstest, chi2, beta as beta_dist
from scipy.stats.mstats import plotting_positions
from statsmodels.graphics.tsaplots import plot_acf

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
FIG  = REPO / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":"sans-serif","font.size":10,"axes.titlesize":11,
    "axes.labelsize":10,"axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":150,"lines.linewidth":1.4,
})

GREY="#6B7280"; BLUE="#2563EB"; RED="#DC2626"; AMBER="#D97706"
GREEN="#16A34A"; PURPLE="#7C3AED"
KS_FLOOR=0.05; ACF_FLOOR=0.05


def load_pit(samples_path, y_path):
    y=np.load(y_path).astype(float); s=np.load(samples_path).astype(float)
    return np.clip(np.mean(s<=y[:,None],axis=1),1e-12,1-1e-12)


def fast_lb(z,lag=10):
    n=len(z)
    a=np.array([np.corrcoef(z[:-k],z[k:])[0,1] for k in range(1,lag+1)])
    return float(1-chi2.cdf(n*(n+2)*np.sum(a**2/(n-np.arange(1,lag+1))),df=lag))


def plot_pit_4panel(u, model_name, out_path, n_lags=40):
    z=norm.ppf(np.clip(u,1e-12,1-1e-12)); n=len(u)
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    fig.suptitle(f"PIT Diagnostics — {model_name}  (n={n:,})",
                 fontsize=12,fontweight="bold",y=1.02)

    # Histogram
    ax=axes[0]
    ax.hist(u,bins=20,density=True,color=BLUE,alpha=0.75,edgecolor="white",linewidth=0.5)
    ax.axhline(1.0,color=RED,linestyle="--",linewidth=1.2,label="Uniform(0,1)")
    ax.set_xlabel("PIT value $u_t$"); ax.set_ylabel("Density")
    ax.set_title("PIT Histogram"); ax.set_xlim(0,1); ax.legend(fontsize=8)
    ks_stat,ks_p=kstest(u,"uniform")
    ax.text(0.97,0.97,f"KS={ks_stat:.4f}\np={ks_p:.3g}",
            transform=ax.transAxes,ha="right",va="top",fontsize=8,
            color=RED if ks_stat>KS_FLOOR else AMBER)

    # ACF
    ax=axes[1]
    try:
        plot_acf(z,lags=min(n_lags,n//5),ax=ax,color=BLUE,
                 vlines_kwargs={"colors":BLUE},alpha=0.05,zero=False)
    except Exception:
        ax.text(0.5,0.5,"ACF unavailable",transform=ax.transAxes,
                ha="center",va="center",color=GREY)
    ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
    ax.set_title(r"ACF of $z_t=\Phi^{-1}(u_t)$")
    ax.axhline(0,color=GREY,linewidth=0.8)
    acf1=float(np.corrcoef(z[:-1],z[1:])[0,1]) if n>2 else 0.0
    ax.text(0.97,0.97,f"ACF(1)={acf1:.3f}",transform=ax.transAxes,
            ha="right",va="top",fontsize=8,
            color=RED if abs(acf1)>ACF_FLOOR else GREEN)

    # Time series
    ax=axes[2]
    step=max(1,n//2000); idx=np.arange(0,n,step)
    ax.plot(idx,u[idx],color=BLUE,alpha=0.5,linewidth=0.6)
    ax.axhline(0.5,color=RED,linestyle="--",linewidth=1.0,label="Uniform median")
    ax.fill_between([0,len(idx)],0.05,0.95,color=GREEN,alpha=0.08,label="90% band")
    ax.set_xlabel("Observation index"); ax.set_ylabel("$u_t$")
    ax.set_title("PIT Time Series"); ax.set_ylim(-0.02,1.02); ax.legend(fontsize=7)

    # Q-Q
    ax=axes[3]
    u_s=np.sort(u); pp=plotting_positions(u_s,alpha=0.5,beta=0.5)
    ax.scatter(pp,u_s,s=1.5,color=BLUE,alpha=0.4,linewidths=0)
    ax.plot([0,1],[0,1],color=RED,linestyle="--",linewidth=1.2,label="Perfect calibration")
    ax.set_xlabel("Theoretical quantile"); ax.set_ylabel("Empirical PIT quantile")
    ax.set_title("Q-Q Plot (PIT vs Uniform)"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path,bbox_inches="tight",dpi=200); plt.close(fig)
    print(f"  Saved: {out_path.name}  [KS={ks_stat:.4f}, ACF1={acf1:.3f}]")


def plot_sim_summary_grid(datasets, out_path):
    """2xN grid of PIT histograms for simulation positive controls."""
    n_ds=len(datasets); ncols=3; nrows=(n_ds+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(15,4.5*nrows))
    axes=axes.flatten() if nrows>1 else [axes] if ncols==1 else axes.flatten()
    fig.suptitle(
        "PIT Histograms — Simulation Positive Controls (all well-specified)\n"
        "All series should show approximately uniform PIT",
        fontsize=12,fontweight="bold",y=1.02)
    for ax_idx,ds in enumerate(datasets):
        ax=axes[ax_idx]; u=ds["u"]
        ks_stat,ks_p=kstest(u,"uniform")
        z=norm.ppf(np.clip(u,1e-12,1-1e-12))
        acf1=float(np.corrcoef(z[:-1],z[1:])[0,1]) if len(u)>2 else 0.0
        ax.hist(u,bins=20,density=True,color=BLUE,alpha=0.75,
                edgecolor="white",linewidth=0.5)
        ax.axhline(1.0,color=RED,linestyle="--",linewidth=1.2)
        ax.set_xlim(0,1); ax.set_title(ds["name"],fontsize=10)
        ax.set_xlabel("PIT $u_t$"); ax.set_ylabel("Density")
        ax.text(0.97,0.97,f"KS={ks_stat:.4f}\nACF(1)={acf1:.3f}",
                transform=ax.transAxes,ha="right",va="top",fontsize=8,
                color=RED if ks_stat>KS_FLOOR else GREEN)
    for ax_idx in range(len(datasets),len(axes)):
        axes[ax_idx].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path,bbox_inches="tight",dpi=200); plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_power_vs_n(out_path, n_sim=300):
    rng=np.random.default_rng(42)
    n_vals=[100,250,500,1_000,2_500,5_000,10_000,25_000,50_000,100_000]
    alpha=0.05; lag=10; phi=0.03
    ks_h0=[]; ks_alt=[]; lb_h0=[]; lb_alt=[]
    print("  Computing power curves...")
    for n in n_vals:
        k_h0=k_alt=l_h0=l_alt=0
        for _ in range(n_sim):
            u0=rng.uniform(size=n); _,p=kstest(u0,"uniform")
            if p<alpha: k_h0+=1
            u1=beta_dist.rvs(1.05,1.05,size=n,random_state=rng); _,p=kstest(u1,"uniform")
            if p<alpha: k_alt+=1
            z0=rng.standard_normal(size=n)
            if fast_lb(z0,lag)<alpha: l_h0+=1
            z1=np.zeros(n); z1[0]=rng.standard_normal(); eps=rng.standard_normal(size=n)
            for t in range(1,n): z1[t]=phi*z1[t-1]+eps[t]
            if fast_lb(z1,lag)<alpha: l_alt+=1
        ks_h0.append(k_h0/n_sim); ks_alt.append(k_alt/n_sim)
        lb_h0.append(l_h0/n_sim); lb_alt.append(l_alt/n_sim)
    n_arr=np.array(n_vals)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
    fig.suptitle("Theoretical Rejection Rate vs Sample Size",fontsize=10,fontweight="bold")
    for ax,h0,alt,title,alt_lbl in [
        (ax1,ks_h0,ks_alt,"Kolmogorov–Smirnov","Alt: Beta(1.05,1.05) [KS≈0.01]"),
        (ax2,lb_h0,lb_alt,f"Ljung–Box (lag={lag})",f"Alt: AR(1,φ={phi}) [ACF≈{phi}]"),
    ]:
        ax.plot(n_arr,h0,color=GREEN,marker="s",markersize=5,linestyle="--",label="H₀")
        ax.plot(n_arr,alt,color=RED,marker="o",markersize=5,label=alt_lbl)
        ax.axhline(alpha,color=GREY,linestyle=":",linewidth=1.0,label=f"α={alpha}")
        ax.axvline(50_000,color=AMBER,linestyle=":",linewidth=1.2,label="run_009/010 (n≈52k)")
        ax.set_xscale("log"); ax.set_xlabel("n (log scale)"); ax.set_ylabel("Rejection rate")
        ax.set_title(title); ax.set_ylim(0,1.05); ax.legend(fontsize=8,loc="upper left")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    fig.tight_layout()
    fig.savefig(out_path,bbox_inches="tight",dpi=200); plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_model_positioning(out_path):
    color_map={"RED":RED,"GREEN":GREEN,"AMBER":AMBER,"YELLOW":"#CA8A04"}
    marker_map={"RED":"X","GREEN":"o","AMBER":"D","YELLOW":"^"}
    size_map={"RED":130,"GREEN":110,"AMBER":120,"YELLOW":110}
    pit_models=[
        ("ENTSO-E Load\n(run_001)",    209_555,0.1615,0.926,"RED",   (8,4),(8,4)),
        ("PV Solar\n(run_002)",          4_287,0.1028,0.660,"RED",   (-85,6),(-85,6)),
        ("Wind\n(run_003)",              9_000,0.1057,0.855,"RED",   (8,-16),(8,-16)),
        ("Sim Price\n(run_004)",           365,0.028, 0.022,"GREEN", (8,-16),(8,-16)),
        ("ENTSO-E Wind DE\n(run_009)",  51_933,0.0083,0.861,"AMBER", (8,-18),(-110,4)),
        ("ENTSO-E Solar DE\n(run_010)", 51_933,0.0258,0.788,"AMBER", (8,8),(8,8)),
    ]
    panels=[
        ("ks", KS_FLOOR, "KS statistic",
         "PIT Uniformity — KS vs Sample Size", (-0.012,0.225),KS_FLOOR+0.005),
        ("acf",ACF_FLOOR,"|ACF lag-1|",
         "Serial Independence — |ACF lag-1| vs Sample Size",(-0.04,1.12),ACF_FLOOR+0.02),
    ]
    fig,axes=plt.subplots(1,2,figsize=(15,6.5))
    fig.suptitle("Diagnostic Positioning — All Model Classes",
                 fontsize=11,fontweight="bold",y=1.01)
    for ax,(key,floor,ylabel,title,ylim,fly) in zip(axes,panels):
        ax.axhline(floor,color=AMBER,linestyle="--",linewidth=1.5,zorder=3)
        ax.axhspan(ylim[0],floor,color=AMBER,alpha=0.07,zorder=1)
        ax.text(190,fly,"Below floor → WARN only",fontsize=7.5,color=AMBER,va="bottom")
        for label,n,ks,acf,outcome,ko,ao in pit_models:
            val=ks if key=="ks" else abs(acf); offset=ko if key=="ks" else ao
            c=color_map[outcome]
            ax.scatter(n,val,color=c,marker=marker_map[outcome],
                       s=size_map[outcome],zorder=5,edgecolors="white",linewidths=0.9)
            ax.annotate(label,(n,val),textcoords="offset points",xytext=offset,
                        fontsize=7.5,color=c,va="center",
                        arrowprops=dict(arrowstyle="-",color=c,lw=0.5,shrinkA=4))
        if key=="ks":
            ax.annotate("KS=0.0083 < floor\n→ WARN",xy=(51_933,0.0083),
                        xytext=(51_933*2.2,0.035),fontsize=7,color=AMBER,
                        arrowprops=dict(arrowstyle="->",color=AMBER,lw=0.8))
        if key=="acf":
            ax.annotate("ACF=0.86 > floor\n→ FAIL (genuine)",
                        xy=(51_933,0.861),xytext=(3_000,0.65),fontsize=7,color=AMBER,
                        arrowprops=dict(arrowstyle="->",color=AMBER,lw=0.8))
        ax.set_xscale("log"); ax.set_xlabel("Sample size n (log scale)")
        ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(*ylim)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    legend_elements=[
        mpatches.Patch(color=RED,label="RED — genuine miscalibration"),
        mpatches.Patch(color=AMBER,label="AMBER — large-n sensitivity"),
        mpatches.Patch(color=GREEN,label="GREEN — well-calibrated"),
        mpatches.Patch(color="#CA8A04",label="YELLOW — borderline"),
    ]
    fig.legend(handles=legend_elements,fontsize=8.5,loc="lower center",ncol=4,
               bbox_to_anchor=(0.5,-0.08),frameon=True,framealpha=0.95,edgecolor=GREY)
    fig.tight_layout(rect=[0,0.04,1,1.0])
    fig.savefig(out_path,bbox_inches="tight",dpi=200); plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    # ── Real-data ─────────────────────────────────────────────────────────────
    print("="*60+"\n  Real-data PIT Diagnostics\n"+"="*60)
    for ds in [
        {"name":"ENTSO-E Load (run_001)",
         "samples":DATA/"derived_full"/"entsoe_samples.npy",
         "y":DATA/"derived_full"/"entsoe_y.npy",
         "out":FIG/"pit_diagnostics_entsoe.png"},
        {"name":"PV Solar (run_002)",
         "samples":DATA/"derived_pv"/"pv_samples.npy",
         "y":DATA/"derived_pv"/"pv_y.npy",
         "out":FIG/"pit_diagnostics_pv.png"},
        {"name":"Wind (run_003)",
         "samples":DATA/"derived_wind"/"wind_samples.npy",
         "y":DATA/"derived_wind"/"wind_y.npy",
         "out":FIG/"pit_diagnostics_wind.png"},
    ]:
        if not ds["samples"].exists():
            print(f"  SKIP {ds['name']}: {ds['samples'].name} not found"); continue
        print(f"\n  {ds['name']}")
        plot_pit_4panel(load_pit(ds["samples"],ds["y"]),ds["name"],ds["out"])

    # ── Simulation positive controls ──────────────────────────────────────────
    print("\n"+"="*60+"\n  Simulation Positive Controls\n"+"="*60)
    sim_specs=[
        ("Sim Price (run_004)",    DATA/"derived_simulation_price"     /"price_samples.npy",
                                   DATA/"derived_simulation_price"     /"price_y.npy",
                                   FIG/"pit_diagnostics_sim_price.png"),
        ("Sim Temp (run_004)",     DATA/"derived_simulation_temp"      /"temp_samples.npy",
                                   DATA/"derived_simulation_temp"      /"temp_y.npy",
                                   FIG/"pit_diagnostics_sim_temp.png"),
        ("Sim Elec Price (run_011)",DATA/"derived_simulation_elec_price"/"elec_price_samples.npy",
                                    DATA/"derived_simulation_elec_price"/"elec_price_y.npy",
                                    FIG/"pit_diagnostics_sim_elec_price.png"),
        ("Sim Natural Gas (run_012)",DATA/"derived_simulation_nat_gas" /"nat_gas_samples.npy",
                                     DATA/"derived_simulation_nat_gas" /"nat_gas_y.npy",
                                     FIG/"pit_diagnostics_sim_nat_gas.png"),
        ("Sim Carbon (run_013)",   DATA/"derived_simulation_carbon"    /"carbon_samples.npy",
                                   DATA/"derived_simulation_carbon"    /"carbon_y.npy",
                                   FIG/"pit_diagnostics_sim_carbon.png"),
    ]
    grid_data=[]
    for name,sp,yp,out in sim_specs:
        if not sp.exists():
            print(f"  SKIP {name}: samples not found — re-run build scripts"); continue
        print(f"\n  {name}")
        u=load_pit(sp,yp); plot_pit_4panel(u,name,out)
        grid_data.append({"name":name.split("(")[0].strip(),"u":u})
    if grid_data:
        print(f"\n  Summary grid ({len(grid_data)} series)...")
        plot_sim_summary_grid(grid_data, FIG/"pit_diagnostics_sim.png")

    # ── Misspecification ──────────────────────────────────────────────────────
    print("\n"+"="*60+"\n  Misspecification Scenarios\n"+"="*60)
    for name,sp,yp,out in [
        ("Sim Price — Variance Inflation (run_004b)",
         DATA/"derived_simulation_price_variance_inflation"/"price_samples.npy",
         DATA/"derived_simulation_price_variance_inflation"/"price_y.npy",
         FIG/"pit_diagnostics_misspec_price_vi.png"),
        ("Sim Price — Mean Bias (run_004b)",
         DATA/"derived_simulation_price_mean_bias"/"price_samples.npy",
         DATA/"derived_simulation_price_mean_bias"/"price_y.npy",
         FIG/"pit_diagnostics_misspec_price_mb.png"),
        ("Sim Price — Heavy Tails (run_004b)",
         DATA/"derived_simulation_price_heavy_tails"/"price_samples.npy",
         DATA/"derived_simulation_price_heavy_tails"/"price_y.npy",
         FIG/"pit_diagnostics_misspec_price_ht.png"),
    ]:
        if not sp.exists():
            print(f"  SKIP {name}: re-run build_simulation_misspec.py"); continue
        print(f"\n  {name}")
        plot_pit_4panel(load_pit(sp,yp),name,out)

    # ── Power + positioning ───────────────────────────────────────────────────
    print("\n"+"="*60+"\n  Theoretical Power vs n\n"+"="*60)
    plot_power_vs_n(FIG/"power_vs_n.png")
    print("\n"+"="*60+"\n  Model Diagnostic Positioning\n"+"="*60)
    plot_model_positioning(FIG/"model_diagnostic_positioning.png")
    print("\nDone. All figures saved to figures/")


if __name__ == "__main__":
    main()
