"""Sourced architecture and protocol numbers for fig_arch; the schematic draws only numbers from this file.

Every value is read from a source file by this script and stored with its path and field (or a short quote):
the base-run config.json of dense and MoE (outputs/<arch>_b05/<run>/config.json); for NESTOR the base run folder
holds only the checkpoint, so the architecture comes from the config.json that G1 run g1_nestor_full_german_s101
saved when it loaded that checkpoint (its log reports 859/859 weights loaded); parameter and frozen-router counts
from the stage-B adaptation logs (logs/cont_<arch>_adapt_pg19_*.log); notation and protocol from the paper
sections 03_account.tex and 04_design.tex, which must contain the quoted statements; the expert-pool semantics
(public = shared pool, private pool per column, two-way path gate) from repo nestor/backbones/expert_pools.py.
Input root: NESTOR_ROOT (default .), read only.
Output: figures/fig_arch_facts.json (FIG_OUT overrides the folder), via a temp file
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with PYTHONSAFEPATH=1
import glob  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import figstyle as fs  # noqa: E402

ROOT = os.environ.get("NESTOR_ROOT", r".")
SEC = fs.ROOT / "sections"
CFG = {
    "dense": os.path.join(ROOT, r"outputs\dense_100m_b05\20261031_203621__dense_100m_b05_train\config.json"),
    "moe": os.path.join(ROOT, r"outputs\moe_100m_b05\20261101_002357__moe_100m_b05_train\config.json"),
    "nestor": glob.glob(os.path.join(ROOT, r"outputs\g1\g1_nestor_full_german_s101\m2c", "*", "*", "config.json"))[0],
}
NESTOR_LOG = glob.glob(os.path.join(ROOT, r"logs\queue_g1\g1_nestor_full_german_s101_m2c.log"))[0]
facts, rel = {}, lambda p: os.path.relpath(p, ROOT)


DRAWN = {  # keys fig_arch.py reads through F() (it prints them)
    "moe.experts_per_layer", "moe.experts_per_token", "nestor.columns", "nestor.columns_per_token",
    "nestor.private_experts_per_column", "nestor.private_top_k", "nestor.shared_experts", "nestor.shared_top_k"}
CAPTION = {  # keys whose values the caption draft of fig_arch states (caption_drafts.md, built by fig_captions.py)
    "dense.layers", "dense.blocks", "dense.width", "dense.heads",
    "dense.head_dim", "dense.mlp_hidden", "dense.stored_params", "moe.layers",
    "moe.blocks", "moe.width", "moe.heads", "moe.mlp_hidden",
    "moe.stored_params", "nestor.layers", "nestor.blocks", "nestor.width",
    "nestor.heads_per_column", "nestor.stored_params", "paper.routing_mass", "paper.forgetting",
    "protocol.steps_tokens_context", "protocol.tokens_total", "protocol.frozen", "protocol.subset",
    "protocol.eval_windows"}


def put(key, value, source, field=None, quote=None, drawn=None):
    facts[key] = {"value": value, "drawn": key in DRAWN, "caption": key in CAPTION, "source": source,
                  **({"field": field} if field else {}), **({"quote": quote} if quote else {})}


def cfg(arch, path):
    d = json.load(open(CFG[arch], encoding="utf-8"))
    for k in path.split("."):
        d = d[k]
    return d


def paper(nm, pattern):
    t = open(SEC / nm, encoding="utf-8").read()
    m = re.search(pattern, t, re.S)
    assert m, (nm, pattern)
    return re.sub(r"\s+", " ", m.group(0))[:140]


def log_count(arch):
    fp = sorted(glob.glob(os.path.join(ROOT, "logs", f"cont_{arch}_100m_adapt_pg19_*.log")))[0]
    t = open(fp, "rb").read().decode("utf-8", "replace")
    n = int(re.search(r"Model parameters: ([\d,]+)", t).group(1).replace(",", ""))
    fr = re.search(r"frozen routers: (\d+) router parameters \(([\d.]+)M of", t)
    return fp, n, (int(fr.group(1)) if fr else None), (float(fr.group(2)) if fr else None)


BC = "job.model.backbone_config."
for arch in ("dense", "moe", "nestor"):
    src = rel(CFG[arch])
    put(f"{arch}.width", cfg(arch, "job.model.adapter_config.embed_dim"), src, "job.model.adapter_config.embed_dim")
    put(f"{arch}.context", cfg(arch, "job.model.adapter_config.max_length"), src, "job.model.adapter_config.max_length",
        drawn=False)
    put(f"{arch}.vocab", cfg(arch, "job.model.adapter_config.vocab_size"), src, "job.model.adapter_config.vocab_size",
        drawn=False)
    nl = cfg(arch, BC + "block_num_layers")
    put(f"{arch}.blocks", cfg(arch, BC + "num_blocks"), src, BC + "num_blocks")
    put(f"{arch}.layers_per_block", nl, src, BC + "block_num_layers", drawn=False)
    put(f"{arch}.layers", sum(nl), src, "sum of " + BC + "block_num_layers")
    put(f"{arch}.head_dim", cfg(arch, BC + "block_head_dims"), src, BC + "block_head_dims", drawn=False)
    put(f"{arch}.always_on_shared_experts", cfg(arch, BC + "router_num_dense_shared_experts"), src,
        BC + "router_num_dense_shared_experts (DeepSeekMoE-style always-on experts)", drawn=False)
    fp, n, fr, frm = log_count(arch)
    put(f"{arch}.stored_params", n, rel(fp), quote="Model parameters: " + f"{n:,}")
    if fr is not None:
        put(f"{arch}.frozen_router_tensors", fr, rel(fp), quote=f"frozen routers: {fr} router parameters ({frm}M ...)",
            drawn=False)
# dense and MoE: one column (G = 1), heads per layer, MLP or expert bank
for arch in ("dense", "moe"):
    src = rel(CFG[arch])
    assert set(cfg(arch, BC + "num_groups")) == {1}
    put(f"{arch}.heads", cfg(arch, BC + "block_num_heads")[0], src, BC + "block_num_heads")
    put(f"{arch}.experts_per_layer", cfg(arch, BC + "router_num_experts_per_group")[0], src,
        BC + "router_num_experts_per_group")
    put(f"{arch}.experts_per_token", cfg(arch, BC + "router_num_experts_per_group_per_token")[0], src,
        BC + "router_num_experts_per_group_per_token")
    put(f"{arch}.mlp_hidden", cfg(arch, BC + "block_mlp_hidden_dims")[0][0], src, BC + "block_mlp_hidden_dims")
assert facts["dense.experts_per_layer"]["value"] == 1 and facts["moe.experts_per_layer"]["value"] == 24
put("moe.active_expert_width", facts["moe.experts_per_token"]["value"] * facts["moe.mlp_hidden"]["value"],
    rel(CFG["moe"]), "experts_per_token x block_mlp_hidden_dims (equals the dense MLP width)", drawn=False)
# NESTOR columns and pools
src = rel(CFG["nestor"])
put("nestor.columns", cfg("nestor", BC + "num_groups")[0], src, BC + "num_groups (attention groups = columns)")
assert len(set(cfg("nestor", BC + "num_groups"))) == 1
put("nestor.columns_per_token", cfg("nestor", BC + "num_groups_per_token"), src, BC + "num_groups_per_token")
put("nestor.heads_per_column", cfg("nestor", BC + "block_num_heads"), src, BC + "block_num_heads (heads per group)",
    drawn=False)
put("nestor.private_experts_per_column", cfg("nestor", BC + "block_expert_pool_private_experts")[0], src,
    BC + "block_expert_pool_private_experts")
put("nestor.private_top_k", cfg("nestor", BC + "block_expert_pool_private_top_k"), src,
    BC + "block_expert_pool_private_top_k")
put("nestor.shared_experts", cfg("nestor", BC + "block_expert_pool_public_experts")[0], src,
    BC + "block_expert_pool_public_experts (the public pool, shared by all tokens)")
put("nestor.shared_top_k", cfg("nestor", BC + "block_expert_pool_public_top_k")[0], src,
    BC + "block_expert_pool_public_top_k")
for k in ("block_expert_pool_private_experts", "block_expert_pool_public_experts", "block_expert_pool_public_top_k"):
    assert len(set(cfg("nestor", BC + k))) == 1, k
nlog = open(NESTOR_LOG, "rb").read().decode("utf-8", "replace")
m = re.search(r"Loaded (\d+)/(\d+) weights", nlog)
assert m and m.group(1) == m.group(2)
put("nestor.config_provenance", f"G1 run config; base checkpoint loaded {m.group(1)}/{m.group(2)} weights",
    rel(NESTOR_LOG), quote=m.group(0), drawn=False)
ep = open(os.path.join(ROOT, r"repo\nestor\backbones\expert_pools.py"), encoding="utf-8").read()
for q in ("Public pool", "Private pools", "path gate"):
    assert q in ep, q
put("nestor.pool_semantics", "public pool shared by all tokens (top-k_p), one private pool per column (top-k_q inside "
    "each selected column), learned two-logit path gate, both paths always evaluated", r"repo\nestor\backbones\expert_pools.py",
    quote="docstring: Public pool / Private pools / path gate", drawn=False)
# paper statements that the schematic repeats (notation and protocol)
put("paper.moe", "E = 24, K_e = 3", "sections/04_design.tex", quote=paper("04_design.tex", r"\$E=24\$ SwiGLU experts, of which a router selects \$K_e=3\$"))
put("paper.nestor_columns", "C = 8, K_c = 3,4,5,6,6,7,8; 13 layers in 7 blocks; 8 column-private experts per column",
    "sections/04_design.tex", quote=paper("04_design.tex", r"13 layers in 7 blocks with \$K_c=3,4,5,6,6,7,8\$ and 8 column-private experts per column"))
put("paper.column_router", "once per block, a column router sends each token to K_c of C = 8 columns", "sections/04_design.tex",
    quote=paper("04_design.tex", r"Once per block, a column router sends each token to \$K_c\$ of \$C=8\$ columns"))
put("paper.shared_experts", "shared experts tied across columns; two-way gate", "sections/04_design.tex",
    quote=paper("04_design.tex", r"A family of shared experts is tied across columns, and a two-way gate"))
put("paper.stored", "0.37B (MoE), 0.38B (NESTOR) stored", "sections/04_design.tex",
    quote=paper("04_design.tex", r"store 0\.37B and 0\.38B parameters"))
put("paper.routing_mass", "w_c(m) = E_c[g_m^2]; W_c(S) = sum over m in S of w_c(m)", "sections/03_account.tex",
    quote=paper("03_account.tex", r"we call \$W_c\(S\)=\\sum_\{m\\in S\}w_c\(m\)\$ the routing mass of \$c\$ through \$S\$"))
put("paper.forgetting", "F_c = L_c(theta_0 + delta) - L_c(theta_0)", "sections/03_account.tex",
    quote=paper("03_account.tex", r"The forgetting of corpus \$c\$ is \$F_c\(\\delta\)=L_c\(\\theta_0\+\\delta\)-L_c\(\\theta_0\)\$"))
put("protocol.steps_tokens_context", "96 steps of 262,144 tokens at 4k context", "sections/04_design.tex",
    quote=paper("04_design.tex", r"96 steps of 262,144 tokens at 4k context"))
put("protocol.tokens_total", 96 * 262144, "sections/04_design.tex", "96 x 262,144")
put("protocol.frozen", "routers and balancing biases frozen, routing noise off", "sections/04_design.tex",
    quote=paper("04_design.tex", r"Routers and balancing biases are frozen and routing noise is off"))
put("protocol.subset", "in subset runs only the experts of S are trainable", "sections/04_design.tex",
    quote=paper("04_design.tex", r"In subset runs only the experts of a subset \$S\$ are trainable"))
put("protocol.eval_windows", "500 windows of 4,096 tokens (2,048,000 tokens) per stream", "sections/04_design.tex",
    quote=paper("04_design.tex", r"500 non-overlapping 4,096-token windows \(2,048,000 tokens\)"))
put("protocol.c1_subsets", "k = 6 of N = 24 experts per MoE layer; pools of 12 (f = 1/2)", "sections/04_design.tex",
    quote=paper("04_design.tex", r"\$k=6\$ of the \$N=24\$ experts of an MoE layer"))
# consistency between configs, logs and paper
assert facts["moe.experts_per_layer"]["value"] == 24 and facts["moe.experts_per_token"]["value"] == 3
assert facts["nestor.columns"]["value"] == 8 and facts["nestor.columns_per_token"]["value"] == [3, 4, 5, 6, 6, 7, 8]
assert facts["nestor.layers"]["value"] == 13 and facts["nestor.blocks"]["value"] == 7
assert facts["nestor.private_experts_per_column"]["value"] == 8
assert round(facts["moe.stored_params"]["value"] / 1e9, 2) == 0.37 and round(facts["nestor.stored_params"]["value"] / 1e9, 2) == 0.38
assert facts["moe.active_expert_width"]["value"] == facts["dense.mlp_hidden"]["value"]
out = {"note": "numbers drawn in fig_arch (drawn = true) and supporting facts, each with its source; values are "
               "read by fig_arch_facts.py from the files named, relative to " + ROOT + " unless they start with sections/",
       "facts": facts}
fs.FIGDIR.mkdir(exist_ok=True)
tmp, dst = fs.FIGDIR / "fig_arch_facts.tmp.json", fs.FIGDIR / "fig_arch_facts.json"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
os.replace(tmp, dst)
assert DRAWN <= set(facts), DRAWN - set(facts)
assert CAPTION <= set(facts), CAPTION - set(facts)
print(f"fig_arch_facts written: {len(facts)} facts, {sum(f['drawn'] for f in facts.values())} drawn")
