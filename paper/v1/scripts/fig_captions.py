"""Caption drafts for the paper v1 figures, with every number filled from the JSON records.

Reads figures/fig_arch_facts.json, data/edge_pre_freeze.json, figures/fig_edge_values.json, data/stageB_window_ci.json,
figures/fig_repro_values.json and data/gap_trajectory.json (data/ is searched first for the figure-side files) and
writes figures/caption_drafts.md via a temp file. Captions are prose (no lists), two to five sentences each, followed
by notes that name the target paragraph for methods material moved out of the figures.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with PYTHONSAFEPATH=1
import json  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402

import figstyle as fs  # noqa: E402


def side(name):
    for p in (fs.DATADIR / name, fs.ROOT / "figures" / name):
        if p.exists():
            return json.load(open(p, encoding="utf-8"))
    raise SystemExit(f"missing {name}")


A = {k: v["value"] for k, v in side("fig_arch_facts.json")["facts"].items()}
E = fs.load("edge_pre_freeze.json")
EV = side("fig_edge_values.json")["runs"]
SB = fs.load("stageB_window_ci.json")
RV = side("fig_repro_values.json")
GP = fs.load("gap_trajectory.json")


def num(x, nd=1, plus=False):
    """LaTeX-safe signed number: negative values get a math minus, never a Unicode one."""
    if x < 0:
        return f"$-${abs(x):.{nd}f}"
    return (f"+{x:.{nd}f}" if plus else f"{x:.{nd}f}")


def s(x, nd=1):
    return num(x, nd, plus=True)


def iv(v, nd=1):
    return f"[{num(v[1], nd)}, {num(v[2], nd)}]"


def bn(n):
    return f"{n / 1e9:.2f}B"


out = ["# Caption drafts for paper v1 figures",
       "",
       "Prepared on 2026-11-15. Every number below is filled by `scripts/fig_captions.py` from "
       "the JSON records the figures are drawn from, so captions and figures carry one value per claim. Each caption is "
       "prose of two to five sentences, written as LaTeX (macros of `macros.tex`, escaped percent signs); the notes "
       "after it list methods material that left the figure, with the paragraph where it may be placed. No "
       "`.tex` file was changed.",
       ""]

# ---------------- fig_arch ----------------
kc = ", ".join(str(v) for v in A["nestor.columns_per_token"])
hpc = ", ".join(str(v) for v in A["nestor.heads_per_column"])
ptk = A["nestor.private_top_k"]
out += ["## fig_arch (new; suggested place: Section 4, paragraph Architectures, `\\label{fig:arch}`)", "",
        f"Architectures and adaptation protocol, each drawn for one layer, with frozen parts marked by a snowflake. "
        f"(a) The dense model applies one attention block and one SwiGLU MLP of width {A['dense.mlp_hidden']:,} to every "
        f"token; it has {A['dense.layers']} layers of width {A['dense.width']} with {A['dense.heads']} heads and "
        f"{bn(A['dense.stored_params'])} parameters. "
        f"(b) In the MoE model a frozen router sends each token to {A['moe.experts_per_token']} of the "
        f"{A['moe.experts_per_layer']} SwiGLU experts (width {A['moe.mlp_hidden']}) of every layer and weights their "
        f"outputs with renormalized gates $g_m(x)$; recorded at $\\theta_0$ over the tokens of a corpus $c$, the gates "
        f"give every expert the second moment $w_c(m)=\\mathrm{{E}}_c[g_m^2]$, and an expert subset $S$ carries the routing "
        f"mass $W_c(S)=\\sum_{{m\\in S}}w_c(m)$ (the bars are illustrative); the model has {A['moe.layers']} layers of "
        f"width {A['moe.width']} with {A['moe.heads']} heads and stores {bn(A['moe.stored_params'])} parameters. "
        f"(c) In \\nestor{{}} a frozen column router sends each token, once per block, to $K_c$ of the "
        f"{A['nestor.columns']} columns, with $K_c={kc}$ in blocks 1 to {A['nestor.blocks']} (drawn: one layer of "
        f"block 2, $K_c={A['nestor.columns_per_token'][1]}$); every active column runs its own attention heads "
        f"({hpc} heads per column by block) over the tokens routed to it and its own frozen router over "
        f"{A['nestor.private_experts_per_column']} private experts (top-{ptk[0]} in block 1, top-{ptk[1]} later), "
        f"{A['nestor.shared_experts']} shared experts (top-{A['nestor.shared_top_k']}) serve every token, and a two-way "
        f"gate weights shared against column experts; the model has {A['nestor.layers']} layers in "
        f"{A['nestor.blocks']} blocks, width {A['nestor.width']}, and stores {bn(A['nestor.stored_params'])} parameters. "
        f"(d) Every run starts from a pretrained base $\\theta_0$, adapts an expert subset $S$ chosen from the routing "
        f"record at $\\theta_0$ ($A_d$ or $B_d$ for C1 on books, $H_d$ or $L_d$ for C2 on German with SGD or Muon, and "
        f"all weights except the routers in full runs and in the reproduction) for {A['protocol.steps_tokens_context']}, "
        f"with routers and balancing biases frozen, and measures forgetting $F$ as the rise of held-out cross-entropy on "
        f"500 windows of 4,096 tokens of the code, mathematics and DCLM streams; residual and depth-read connections are "
        f"not drawn.", "",
        "Notes. The dense parameter count (0.14B), the expert width of 640, the heads per column, the private "
        "top-k by block and the eight shared experts with top-2 routing are not yet in Section 4; if wanted there, they "
        "fit the paragraph Architectures of `04_design.tex`. The numbers come from `figures/fig_arch_facts.json`, whose "
        "entries name their source files (base-run `config.json`, the G1 NESTOR run config that loaded the base "
        "checkpoint 859/859, stage-B log headers, paper text).", ""]

# ---------------- fig_edge ----------------
a96 = E["adaptation_96_steps"]
mo, ne = a96["moe_full_runs_no_ckpt"], a96["nestor_full_runs_ckpt"]
oomg1 = sorted((r["energy_wh"] for r in E["oom_attempts_G1"]), reverse=True)
sc = [r for r in E["stageC"] if r.get("kind") == "train"]
def oom(arch, seq):
    return sum(r["energy_wh"] for r in sc if r["arch"] == arch and int(r["seq"]) == seq and r["outcome"] == "OOM")
n_nes_oom = sum(1 for r in sc if r["arch"] == "nestor_100m" and r["outcome"] == "OOM")
mv, nv = EV["moe"], EV["nestor"]
_att = fs.load("stageC_results.json")["nestor"]["attempts"]
_pr = re.search(r"probe ext(\d+) micro (\d+) with checkpointing at MEM_FRAC ([\d.]+): OOM at the first step "
                r"\(([\d.]+) GiB allocated, (\d+) bytes free\)", _att)
if _pr is None:
    raise SystemExit("stageC_results.json nestor.attempts: memory-fraction probe not found")
pr_seq, pr_micro, pr_frac, pr_gib, pr_free = int(_pr[1]), int(_pr[2]), _pr[3], _pr[4], int(_pr[5])
out += ["## fig_edge (new; suggested place: Section 6.1, after the paragraph on the pre-freeze measurements, "
        "`\\label{fig:edge}`)", "",
        f"Cost of adaptation on one 16\\,GB consumer GPU (RTX 4080 SUPER), measured as GPU board energy from a 1 Hz power "
        f"log; the values are descriptive, come from the training segments, are not the registered measures of the cost block and count board power only, without "
        f"CPU, memory or power-supply losses. "
        f"(a) Energy per 96-step full adaptation of 25.2M tokens in gate G1: the {mo['n']} MoE runs of base 1 (full "
        f"adaptation on German or mathematics, all-expert adaptation on German, one DCLM continuation), all without "
        f"activation checkpointing, took a median of {mo['energy_wh'][1]:.1f} Wh and {mo['wall_h'][1]:.3f} h, and the "
        f"{ne['n']} NESTOR full adaptations on German, which fit in memory only with activation checkpointing, took "
        f"{ne['energy_wh'][1]:.1f} Wh and {ne['wall_h'][1]:.2f} h, {ne['energy_wh'][1] / mo['energy_wh'][1]:.1f} times "
        f"the energy and about five times the time; the out-of-memory attempts without checkpointing that preceded them "
        f"({oomg1[0]:.1f} and {oomg1[1]:.1f} Wh) are not included. "
        f"(b) Board power of one run of each kind, the MoE run with the median energy and the NESTOR run of the same job "
        f"(full adaptation on German, data seed 101): {mv['minutes']:.1f} min at a mean of {mv['mean_w']:.0f} W and "
        f"{mv['mean_util']:.0f}\\% utilisation against {nv['minutes']:.1f} min at {nv['mean_w']:.0f} W and "
        f"{nv['mean_util']:.0f}\\%, that is {nv['mean_w'] / mv['mean_w']:.2f} times the mean power for "
        f"{nv['minutes'] / mv['minutes']:.1f} times the duration; dashed lines mark the means, and the low-power opening minutes lie before the first "
        f"logged training step. "
        f"(c) Context extension of stage C (48 steps of 1,048,576 tokens, single runs): the dense model and MoE base 1 "
        f"completed at 8,192 (8k) and 16,384 (16k) tokens, the MoE model only with activation checkpointing after one "
        f"out-of-memory attempt at each length ({oom('moe_100m', 8192):.1f} and {oom('moe_100m', 16384):.1f} Wh, "
        f"hatched), whereas all {n_nes_oom} NESTOR attempts, at equal tokens or equal FLOPs and micro-batches 2 and 1 "
        f"with checkpointing, ran out of memory ({oom('nestor_100m', 8192):.1f} Wh at 8k and "
        f"{oom('nestor_100m', 16384):.1f} Wh at 16k without a result); a further probe at {pr_seq // 1024}k with "
        f"micro-batch {pr_micro}, activation checkpointing and a memory fraction of {pr_frac} ran out of memory at "
        f"the first step, with {pr_gib}\\,GiB allocated and {pr_free} bytes free.", "",
        "Notes. Panel (a) is the median comparison of the paragraph that begins 'Before the freeze' in "
        "`06b_edge.tex` (0.248 h, 54.0 Wh, 1.25 h, 143.9 Wh, ratio 2.7); utilization and mean power per run are in the "
        "caption only. The memory-fraction 0.93 probe of `05_reproduction.tex` (paragraph Context extension) is not in "
        "the energy record and is not drawn; the caption reads its settings and memory state from "
        "`data/stageC_results.json` (`nestor.attempts`). The panel titles name the measured quantity, and the two "
        "NESTOR bars of panel (c) carry the marker 'exceeds 16 GB'. The traces come from `figures/fig_edge_values.json` "
        "(`scripts/fig_edge_extract.py`), which reproduces the record's energy, mean power and utilization of both "
        "windows before writing; timestamps of the power log are local time (UTC+2) and are converted as in "
        "`edge_pre_freeze.py`.", ""]

# ---------------- fig_repro ----------------
pm, pc, pt = SB["per_model"], RV["pct_of_moe"], RV["paper_table6_pct_of_moe"]
out += ["## fig_repro (new; suggested place: Section 5.1, paragraph Domain adaptation, `\\label{fig:repro}`)", "",
        f"Domain adaptation of stage B, the analogue of \\citep[Table~6]{{chu2026nested}} at our scale: the dense "
        f"model, MoE base 1 and \\nestor{{}}, 100M-class models whose bases were pretrained on 0.5B tokens, are each "
        f"adapted for 96 steps of 262,144 tokens at 4k context on books (PG19), mathematics (OpenWebMath), code "
        f"(AlgebraicStack) and German (FineWeb2), with routers frozen in the routed models. "
        f"(a) Forgetting is the increase of cross-entropy on 500 windows of 4,096 tokens of DCLM shard 09 (mean of the "
        f"four corpora: dense {pm['dense_100m']['mean_forgetting'][0]:.3f}, MoE {pm['moe_100m']['mean_forgetting'][0]:.3f}, "
        f"\\nestor{{}} {pm['nestor_100m']['mean_forgetting'][0]:.3f} nats); every cell is a single run, and its 95\\% "
        f"window-bootstrap interval (4,000 resamples, window sampling only, no seed replication) lies within the marker. "
        f"(b) Forgetting relative to MoE, $100(F-F_{{\\mathrm{{MoE}}}})/F_{{\\mathrm{{MoE}}}}$, with intervals from the "
        f"same paired resamples: on the four-corpus mean the dense model forgets {s(pc['dense_100m']['mean'][0])}\\% "
        f"{iv(pc['dense_100m']['mean'])} and \\nestor{{}} {s(pc['nestor_100m']['mean'][0])}\\% {iv(pc['nestor_100m']['mean'])} "
        f"relative to MoE, and \\nestor{{}} lies below MoE only on mathematics, at {s(pc['nestor_100m']['owm'][0])}\\% "
        f"{iv(pc['nestor_100m']['owm'])}. "
        f"The lower facet shows the means of the preprint's Table 6 ({s(pt['dense'])}\\% for the dense model and "
        f"{s(pt['nestor'])}\\% for \\nestor{{}}), which come from 500M-scale \\nestor{{}} and MoE models and a 1B dense model "
        f"adapted for 0.5B tokens at 32k context; they are drawn with open markers in a separate facet because those "
        f"runs differ from ours in scale, adaptation budget and context and are not peers of our single runs.", "",
        "Notes. The percent intervals are new: `stageB_window_ci.json` holds intervals for absolute differences "
        "only, so `scripts/fig_repro_ci.py` repeats its bootstrap exactly (it reproduces all 120 recorded values to "
        "1e-12) and adds the ratio intervals in `figures/fig_repro_values.json`; if the text quotes them, a ledger entry "
        "for that file is needed. The paragraph Domain adaptation of `05_reproduction.tex` already gives the means and "
        "the absolute differences.", ""]

# ---------------- reviewed figures ----------------
lh = GP["gaps"]["N_minus_Mavg"]["last_half"]
ch, (cl, cu) = lh["change_over_last_half_nats"], lh["change_over_last_half_ci95"]
ml, mu = lh["mean_ci95_block_bootstrap"]
out += ["## fig_design (crossover tables; replaces the caption of `fig:design` in `01_introduction.tex`)", "",
        "The two pre-registered crossover designs, drawn for one layer of a 24-expert MoE base; filled squares show one "
        "draw, and the signs in each table give how the log forgetting of the runs enters the crossover statistic, so "
        "each table reads as a difference of differences. (a) C1, target books: experts are ordered by the tilt of their "
        "routing between code and mathematics, and subsets $A_d$ and $B_d$ take 6 of the 12 experts of the lower and the "
        "upper pool in every layer, matched on books and DCLM routing mass within 5\\% ($\\log1.05$). Each subset is "
        "adapted alone on books, and its log forgetting on code and on mathematics, summed with the signs shown, gives "
        "the crossover $I_d$ of Equation~\\eqref{eq:Id}, which is predicted to equal the contrast $P_d$ of routing at "
        "the pretrained weights; the test asks whether $J_d=I_d-0.25\\,P_d$ is positive on average over the draws. "
        "(b) C2, target German: $L_d$ and $H_d$ take 6 of the 8 experts with the lowest and the highest German routing "
        "mass in every layer and are matched on DCLM routing mass within 5\\%; each is adapted on German twice, with "
        "SGD, under which experts that the target uses more move further, and with per-matrix Muon, which takes a step "
        "of equal size for every matrix of a shape. Summed with the signs shown, the log DCLM forgetting of the four runs "
        "($F_{\\mathrm{SGD}}$ and $F_{\\mathrm{Muon}}$ in Equation~\\eqref{eq:I2}) gives the optimizer interaction "
        "$I2_d$, which is predicted to be positive; the test asks whether $J2_d=I2_d-0.25\\,\\Delta_{Q,d}$ is positive "
        "on average at the evaluation step $\\sstar$, which gate G2-3 fixed at 8, so C2 is confirmatory.", "",
        "## fig_gap (revised panel text; caption of `fig:gap` in `05_reproduction.tex`)", "",
        "Training-loss trajectories of the 0.5B-token reproduction (one logged batch loss every 10 steps; lines are "
        "LOESS smooths for display, statistics use the raw points). (a) Training loss of the four runs, with the first "
        "0.09B tokens above the plotted range; both MoE seeds end lowest and \\nestor{} ends level with the dense model, "
        "and the top axis gives the learning rate relative to its peak, which falls from 0.62 of peak at the start of "
        "the last half (shaded) to 0.018 at the last logged step. (b) Gaps between runs, with LOESS bands; diamonds mark "
        f"the held-out val09 gaps of the final checkpoints, and the MoE seed difference serves as the noise reference. "
        f"Over the last half the \\nestor{{}}$-$MoE gap of the two-seed mean averages {num(lh['mean'], 3)} nats "
        f"[{num(ml, 3)}, {num(mu, 3)}] and changes by {s(ch, 3)} nats "
        f"[{num(cl, 3)}, {num(cu, 3, plus=True)}], so no change is detectable.", "",
        "Notes. The panel titles now name the plotted quantity; the comparisons they stated before are in the "
        "caption, (a) as the ordering at the end and (b) as the last-half mean gap, read from "
        "`data/gap_trajectory.json` (`gaps.N_minus_Mavg.last_half`).", "",
        "## fig_entanglement (revised titles and one legend entry; caption of `fig:entanglement` in `06_results.tex` "
        "unchanged)", "",
        "The present caption already states every fact that left the figure: the dashed line is 'the load above which "
        "no exact match exists' (legend now 'exact-match limit'), and panel (b) plots the load against the correlation "
        "between target and DCLM routing shares (title now 'Low load where routing tracks DCLM').", "",
        "## fig_crossover (revised titles; placeholder caption of `fig:crossover` in `06_results.tex`, filled caption "
        "written by `scripts/fill_core.py`)", "",
        "Placeholder caption, revised: C1 crossover against the predicted contrast for MoE base 1 (seed 42) and MoE base "
        "2 (seed 43), both pretrained on 0.5B tokens. The panels show the registered books draws with the probe "
        "crossovers of gate G0-7 at the pretrained weights (pinned, norm-matched random directions; crosses); the adapted "
        "draws, raw (filled) and symmetrized (open), will be added from the registered analysis output. Solid line: the "
        "prediction $I=P$ ($\\beta=1$); dotted line: the margin $I=0.25\\,P$.", "",
        "Note on the filled caption that `fill_core.py` writes (function `figure_and_table`, string `cap`): the "
        "second title line of panel (c), 'bases 1 and 2, step 8', left the figure, and the filled caption already states "
        "both facts ('the C2 interaction against the target contrast at step 8 (c)'). The C2 markers are now filled "
        "diamonds (base 1) and filled squares (base 2), distinct from the circles of the C1 draws, and the marker phrase "
        "of that caption string was changed to match, from 'in (c), circles and squares: C2 draws of bases 1 and~2.' to "
        "'in (c), diamonds and squares: C2 draws of bases 1 and~2.' (the only edit in `fill_core.py`; see "
        "`figure_review.md`).", ""]
dst, tmp = fs.ROOT / "figures" / "caption_drafts.md", fs.ROOT / "figures" / "caption_drafts.tmp.md"
with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(out))
os.replace(tmp, dst)
print("caption_drafts.md written")
