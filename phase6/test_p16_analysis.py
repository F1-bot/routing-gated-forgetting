"""P-16 (2026-11-15; prereg v1.3 version 8 sec. 3.5; open slots v8 sec. C items 9 and 10): staged analysis.

Needs NESTOR_MOCK_ROOT (a root built by chain6b.ps1 -Mock); the test root path must stay below ~140 characters
(Windows MAX_PATH). Two worlds are copied from it, their confirm plan rebuilt with this confirm_plan.py (core first, the
core marker gate), simulated with mock_confirm.py and analysed in both modes:
  mock       the chain-built world as it is (its G2 record has s* = 24, so C2 runs its 96 steps), power log in two files
  truncated  the real G2 freeze values (s* = 8, 8 confirmatory C2 draws, Holm {C1, C2}): the C2 runs stop after step 8
The core analysis must refuse before the core marker, run once, read only core files, and give the C1 / C2 decisions of
the full analysis on the same files; C2 must be computed from the step-8 files (never p = 1 because later files are
absent); the E1 final-loss clause does not apply to a stopped run; EDGE reads every power-log file in time order."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "phase6" else HERE   # phase6\ -> its project root
for _p in (CODE_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import analyze_confirm as AC  # noqa: E402

REAL_C2 = {"s_star": 8, "C2_draws": 8, "holm_family": ["C1", "C2"], "m": 2, "C2_confirmatory": True,
           "c2_status": "confirmatory"}
BUILD = ["build", "--g0-record", "g0/gate_record.json", "--g1-record", "g1/gate_record_G1.json",
         "--g2-record", "g2/gate_record_G2.json", "--nestor-books", "results/nestor_books_feasibility.json",
         "--dose-record", "results/dose/dose_record.json", "--c1de-dir", "results/c1de",
         "--states", "results/queue_state_g1.json,results/queue_state_g2.json", "--out-dir", "confirm",
         "--no-check", "--spent-h", "2.75"]
SKIP_REL = ("outputs/confirm", "results/analysis", "results/confirm", "results/queue_state_confirm.json",
            "logs/queue_confirm", "logs/power")


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(list(dict.fromkeys([HERE, CODE_ROOT])) +
                                        [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _py(args: list[str], cwd: str, ok=(0,)) -> subprocess.CompletedProcess:
    p = subprocess.run([sys.executable] + args, cwd=cwd, env=_env(), capture_output=True, text=True, timeout=3600)
    assert p.returncode in ok, (p.stdout + p.stderr)[-4000:]
    return p


def _set_param(par: dict, k: str, v) -> None:
    if isinstance(par.get(k), dict) and "value" in par[k]:
        par[k]["value"] = v
    else:
        par[k] = v


def _world(tmp_path_factory, name: str, *, real_c2: bool = False, scenario: dict | None = None) -> str:
    src = os.environ.get("NESTOR_MOCK_ROOT")
    if not src or not os.path.isfile(os.path.join(src, "FREEZE.json")):
        pytest.skip("NESTOR_MOCK_ROOT (a root built by chain6b.ps1 -Mock) not set")
    base = tmp_path_factory.mktemp(name)
    root = str(base / "r")

    def ignore(d, names):
        rel = os.path.relpath(d, src).replace(os.sep, "/")
        rel = "" if rel == "." else rel + "/"
        return [n for n in names if (rel + n) in SKIP_REL]

    shutil.copytree(src, root, ignore=ignore)
    if real_c2:                                         # the real G2 freeze values: C2 confirmatory at s* = 8
        gp = os.path.join(root, "g2", "gate_record_G2.json")
        g2 = json.load(open(gp, encoding="utf-8"))
        g2.setdefault("freeze", {}).update(REAL_C2)
        json.dump(g2, open(gp, "w", encoding="utf-8"), indent=1)
        fp = os.path.join(root, "FREEZE.json")
        fz = json.load(open(fp, encoding="utf-8"))
        for k, v in REAL_C2.items():
            _set_param(fz.setdefault("parameters", {}), k, v)
        json.dump(fz, open(fp, "w", encoding="utf-8"), indent=1)
    _py([os.path.join(HERE, "confirm_plan.py")] + BUILD, root)
    args = [os.path.join(HERE, "mock_confirm.py"), "simulate", "--root", root]
    if scenario:
        sp = str(base / "scenario.json")
        json.dump(scenario, open(sp, "w", encoding="utf-8"))
        args += ["--scenario", sp]
    _py(args, HERE)
    return root


def _analyze(root: str, mode: str, ok=(0,)):
    p = _py([os.path.join(HERE, "analyze_confirm.py"), "run", "--root", root, "--no-verify", "--mode", mode], root, ok)
    sub = "analysis_core" if mode == "core" else "analysis"   # same file name and layout in both (addendum 4)
    jp = os.path.join(root, "results", sub, "analysis_confirm.json")
    return p, (json.load(open(jp, encoding="utf-8")) if p.returncode == 0 else None)


@pytest.fixture(scope="module")
def mock_w(tmp_path_factory):
    root = _world(tmp_path_factory, "p16m", scenario={"edge": {"power_split": True}})
    return {"root": root, "core": _analyze(root, "core")[1], "full": _analyze(root, "full")[1]}


@pytest.fixture(scope="module")
def trunc_w(tmp_path_factory):
    root = _world(tmp_path_factory, "p16t", real_c2=True)
    return {"root": root, "core": _analyze(root, "core")[1], "full": _analyze(root, "full")[1]}


def _plan(root: str) -> dict:
    return json.load(open(os.path.join(root, "confirm", "plan_confirm.json"), encoding="utf-8"))


def test_core_refuses_before_the_core_marker_and_runs_once(tmp_path_factory, mock_w):
    root = mock_w["root"]
    p, _ = _analyze(root, "core", ok=(31,))                       # second core run: refused, output kept
    assert "runs once" in p.stdout
    w2 = str(tmp_path_factory.mktemp("p16n") / "r")
    shutil.copytree(root, w2, ignore=shutil.ignore_patterns("analysis_core", "CORE_DONE.json"))
    p, _ = _analyze(w2, "core", ok=(30,))
    assert "CORE_DONE.json missing" in p.stdout
    assert not os.path.exists(os.path.join(w2, "results", "analysis_core"))


@pytest.mark.parametrize("world", ["mock_w", "trunc_w"])
def test_core_and_full_decisions_are_equal_on_the_same_files(world, request):
    w = request.getfixturevalue(world)
    core, full = w["core"], w["full"]
    assert core["mode"] == "core" and full["mode"] == "full"
    assert core["decisions"] == full["decisions"]
    assert core["holm"] == full["holm"]
    for blk in ("C1", "C2"):
        assert core[blk].get("test") == full[blk].get("test")
        assert core[blk].get("per_draw") == full[blk].get("per_draw")
    for k in ("C4", "bpb", "C2_eq_rand"):
        assert core["blocks"][k] == full["blocks"][k]
    assert core["format"] == full["format"] == "analysis_confirm/1"
    assert core["blocks"]["matched_gain"]["C1"] == full["blocks"]["matched_gain"]["C1"]
    assert core["blocks"]["matched_gain"]["C2"]["status"].startswith("deferred")
    mk = core["core"]["marker"]
    assert mk["path"] == "results/confirm/CORE_DONE.json" and mk["n_done"] == mk["n_jobs"] and not mk["not_done"]


def test_core_reads_only_core_files(mock_w):
    root, core = mock_w["root"], mock_w["core"]
    plan = _plan(root)
    blocks = plan["p16"]["core_blocks"]
    core_ids = {r["id"] for r in plan["runs"] if r["block"] in blocks}
    other = {r["id"] for r in plan["runs"]} - core_ids
    assert other and core_ids
    for path in core["inputs"]:
        assert "/eval_selpin/" not in path
        if path.startswith("outputs/confirm/"):
            assert path.split("/")[2] in core_ids, path
    assert set(core["core"]["deferred_to_full_analysis"]) == set(AC.CORE_DEFERRED)
    for k in ("B3", "audit", "C3", "C3_per_layer", "C5_french", "dose", "M12b", "C1_DE", "NESTOR_books", "selpin", "F_rel",
              "forecasts"):
        assert core["blocks"][k] == {"status": AC.DEFERRED_STATUS}, k          # layout kept, nothing computed
    # P-16 revised (2026-11-15 ~03:00): core = C1 on both MoE bases + C2 (+ base_eval); third base and audit: full only
    assert blocks == ["C1:moe_b1", "C1:moe_b2", "C2", "base_eval"]
    assert core["core"]["core_blocks"] == ["C1:moe_b1", "C1:moe_b2", "C2"]
    assert mock_w["full"]["blocks"]["audit"].get("status") != AC.DEFERRED_STATUS
    assert mock_w["full"]["blocks"]["B3"].get("status") != AC.DEFERRED_STATUS
    moved = {r["id"] for r in plan["runs"] if r["block"] in ("b3", "audit")}
    assert not moved & core_ids
    assert set(core["blocks"]) == set(mock_w["full"]["blocks"])
    assert {r["run"] for r in core["blocks"]["EDGE"]["runs"]} <= core_ids
    assert core["C1"]["deferred_to_full_analysis"] == ["selpin", "sym_selpin"]
    ctx = AC.Ctx(root, verify=False, mode="core")
    rid = sorted(moved or other)[0]
    with pytest.raises(AC.FirewallError):
        ctx._read(os.path.join(root, "outputs", "confirm", rid, "x.npz"))
    with pytest.raises(AC.FirewallError):
        ctx._read(os.path.join(root, "outputs", "confirm", sorted(core_ids)[0], "eval_selpin", "selpin__math.npz"))
    with pytest.raises(AC.FirewallError, match="assumption audit"):          # MF-P3: the audit directory too
        ctx._read(os.path.join(root, "audit", "moe_b1", "x.json"))


def test_truncated_c2_is_computed_from_the_step_8_files(trunc_w):
    root, core, full = trunc_w["root"], trunc_w["core"], trunc_w["full"]
    c2runs = [r for r in _plan(root)["runs"] if r.get("role") in ("c2_sgd", "c2_muon")]
    stop_of = {"c2_muon": 8, "c2_sgd": 24}                    # MF-P1 fix (A)
    assert len(c2runs) == 32 and all(r["steps"] == 96 and r["warmup"] == 10 and r["stop_after"] == stop_of[r["role"]]
                                     for r in c2runs)
    for r in c2runs:                                          # no file of a step after the run's stop step exists
        idir = os.path.join(root, "outputs", "confirm", r["id"])
        names = [n for _, _, fn in os.walk(idir) for n in fn if n.startswith("step")]
        assert names and all(int(n[4:7]) <= stop_of[r["role"]] for n in names), r["id"]
        assert not any("__full" in n and int(n[4:7]) != 8 for n in names), r["id"]    # full evaluation at s* = 8 only
    for doc in (core, full):
        assert doc["C2"]["test"] is not None and doc["C2"]["s_star"] == 8 and doc["C2"]["stopped_after_step"] == 8
        assert doc["holm"]["family"] == ["C1", "C2"] and doc["holm"]["entered_with_p_1"] == []
        assert doc["decisions"]["C2"]["label"] != "not run"
        assert sorted(doc["C2"]["decay_profile"]) == ["8"] or sorted(doc["C2"]["decay_profile"]) == [8]
    nc = full["not_computable_from_registered_files"]
    for step in (24, 48, 96):
        assert any(f"at step {step}: not computable" in x for x in nc)
    assert any(x.startswith("matched_gain C2: not computable") for x in nc)
    assert any(x.startswith("F_rel C2: not computable") for x in nc)
    for doc in (core, full):                                  # addendum 4: on the step-8 checkpoint of the 16 Muon runs
        eq = doc["blocks"]["C2_eq_rand"]
        assert "status" not in eq and eq["step"] == 8 and len(eq["per_draw"]) == 8


def _e1(root, tmp_path, rid, losses, name):
    d = tmp_path / name / "trainlog"
    d.mkdir(parents=True)
    recs = [{"step": s, "loss": v, "ce": v, "micro_batches": 32} for s, v in losses]
    (d / "train_steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    ctx = AC.Ctx(root, verify=False)
    ctx.root_of = lambda r_: str(tmp_path / name)
    return ctx.e1_of(rid)


def test_e1_final_rule_has_clauses_i_and_ii_only(trunc_w, tmp_path):
    """P-17 (a) final (MF-D1): (i) non-finite loss at any executed step, (ii) EMA(0.1) more than 0.5 nats above step 1
    after warm-up; the final-loss / stop-step clause is deleted for every run, stopped or not."""
    root = trunc_w["root"]
    plan = _plan(root)
    sgd = next(r["id"] for r in plan["runs"] if r.get("role") == "c2_sgd")
    muon = next(r["id"] for r in plan["runs"] if r.get("role") == "c2_muon")
    c1 = next(r["id"] for r in plan["runs"] if r.get("role") == "c1")
    # a healthy SGD run whose stop-step loss is above step 1 (single-batch noise) -> NOT flagged
    e = _e1(root, tmp_path, sgd, [(s, 8.42 + 0.004 * s) for s in range(1, 25)], "a")
    assert e["diverged"] is False and e["rule"] == AC.E1_RULE == AC.E1_STOP_RULE and e["stop_step"] == 24
    assert e["final_loss"] > e["step1_loss"] and len(e["clauses_evaluated"]) == 2 and e["clause_ii"] == "evaluated"
    # a 96-step run whose final loss is at or above step 1 -> NOT flagged either
    e = _e1(root, tmp_path, c1, [(s, 3.5 + 0.0005 * s) for s in range(1, 97)], "b")
    assert e["diverged"] is False and "stop_step" not in e and e["final_loss"] > e["step1_loss"]
    # EMA about +0.69 after warm-up -> flagged by clause (ii)
    e = _e1(root, tmp_path, sgd, [(s, 8.42 if s <= 10 else 8.42 + 0.9) for s in range(1, 25)], "c")
    assert e["diverged"] is True and any(x.startswith("EMA(0.1)") for x in e["reasons"])
    assert e["ema_max_after_warmup"] - e["step1_loss"] > 0.6
    # a non-finite loss at an executed step -> flagged by clause (i), also for a Muon run stopped at step 8
    e = _e1(root, tmp_path, muon, [(s, float("nan") if s == 5 else 8.4 - 0.01 * s) for s in range(1, 9)], "d")
    assert e["diverged"] is True and any(x.startswith("non-finite") for x in e["reasons"])
    # a Muon run stopped at step 8 with a large finite rise -> NOT flagged; clause (ii) not reached (warm-up 10)
    e = _e1(root, tmp_path, muon, [(s, 8.4 + (3.0 if s >= 5 else 0.0)) for s in range(1, 9)], "e")
    assert e["diverged"] is False and e["stop_step"] == 8 and len(e["clauses_evaluated"]) == 1
    assert e["clause_ii"].startswith("not reached")
    # a log that ends before the stop step: flagged incomplete (warning), not divergent (E2)
    e = _e1(root, tmp_path, sgd, [(s, 8.4 - 0.01 * s) for s in range(1, 9)], "f")
    assert e["status"] == "incomplete_log" and e["diverged"] is None and any("not at the stop step 24" in x for x in e["reasons"])


def test_forecast_rows_after_the_stop_step_are_not_scorable(trunc_w):
    """Plan diff: C2 forecast rows at a step after the run's stop step are listed as not_scorable with a reason,
    never scored against the stop-step checkpoint and never dropped silently."""
    fb = trunc_w["full"]["blocks"]["forecasts"]
    assert "status" not in fb, fb.get("status")
    ns = fb["scores"]["not_scorable"]
    stop = {r["id"]: int(r.get("stop_after") or 0) for r in _plan(trunc_w["root"])["runs"]}
    assert ns["n_rows"] > 0 and ns["targets"] and set(ns["by_block"]) == {"C2"}
    for t in ns["targets"]:
        assert t["status"] == "not_scorable" and stop.get(t["run_id"]) and t["step"] > stop[t["run_id"]]
        assert "stopped after step" in t["reason"]
    assert trunc_w["core"]["blocks"]["forecasts"] == {"status": AC.DEFERRED_STATUS}   # the core never scores forecasts


def test_core_refuses_when_a_core_run_is_not_done(tmp_path_factory, mock_w):
    """MF-P2: the core analysis never runs on an incomplete core (exit 33, nothing written)."""
    w2 = str(tmp_path_factory.mktemp("nd") / "w")
    shutil.copytree(mock_w["root"], w2, ignore=shutil.ignore_patterns("analysis_core"))
    plan = _plan(w2)
    rid = next(r["id"] for r in plan["runs"] if r.get("block") == "C1:moe_b2")
    sp = os.path.join(w2, "results", "queue_state_confirm.json")
    st = json.load(open(sp, encoding="utf-8"))
    st[rid] = dict(st.get(rid) or {}, status="failed")
    json.dump(st, open(sp, "w", encoding="utf-8"))
    p, _ = _analyze(w2, "core", ok=(33,))
    assert "core run(s) not done" in p.stdout and rid in p.stdout
    assert not os.path.exists(os.path.join(w2, "results", "analysis_core", "analysis_confirm.json"))


def test_edge_reads_every_power_log_file_in_time_order(mock_w):
    root, full = mock_w["root"], mock_w["full"]
    truth = json.load(open(os.path.join(root, "results", "analysis_mock", "truth.json"), encoding="utf-8"))["edge"]
    pl = full["blocks"]["EDGE"]["power_log"]
    assert len(truth["power_files"]) == 2 and pl["files"] == sorted(truth["power_files"])
    gap = truth["gap_run"]
    ex = [x for x in full["blocks"]["EDGE"]["excluded"] if x["run"] == gap]
    assert ex and "apart" in ex[0]["reason"]
    assert gap not in {r["run"] for r in full["blocks"]["EDGE"]["runs"]}


@pytest.mark.parametrize("world", ["mock_w", "trunc_w"])
def test_c2_sgd_step24_gain_is_descriptive_and_equal_in_core_and_full(world, request):
    """C2.sgd_step24_gain = German gain of the SGD runs on the 64-window interim head,
    CE(step 0) - CE(step s), s = 24 and 8; the mock writes G_s = gain * s / 96 exactly on the head windows."""
    import mock_confirm as MCK
    w = request.getfixturevalue(world)
    g0 = MCK.DEFAULT["gain"]
    stop = {r["id"]: r.get("stop_after") for r in _plan(w["root"])["runs"] if r.get("role") == "c2_sgd"}
    from collections import Counter
    nb = Counter(r["base"] for r in _plan(w["root"])["runs"] if r.get("role") == "c2_sgd")
    assert len(stop) == sum(nb.values()) and len(stop) in (8, 16) and set(nb) == {"moe_b1", "moe_b2"}
    assert set(stop.values()) == ({24} if world == "trunc_w" else {None})
    for mode in ("core", "full"):
        g = w[mode]["C2"]["sgd_step24_gain"]
        assert "no decision role" in g["role"] and g["decision_role"] is None
        assert g["windows"] == [64] and g["n_planned"] == len(stop) and not g["excluded"] and not g["missing_heads"]
        assert len(g["per_run"]) == len(stop) and {x["set"] for x in g["per_run"]} == {"german"}
        assert {x["run"] for x in g["per_run"]} == set(stop)
        for s in (24, 8):
            st = g[f"step{s}"]
            assert st["pooled"]["n"] == len(stop) and set(st["per_base"]) == set(nb)
            assert all(st["per_base"][b]["n"] == nb[b] for b in nb)
            for v in [st["pooled"], *st["per_base"].values()]:
                assert v["mean"] == pytest.approx(g0 * s / 96, abs=2e-6)
                assert v["min"] == pytest.approx(g0 * s / 96, abs=2e-6)
                assert v["max"] == pytest.approx(g0 * s / 96, abs=2e-6)
        assert g["step24"]["pooled"]["mean"] / g["step8"]["pooled"]["mean"] == pytest.approx(3.0, rel=1e-3)
    assert w["core"]["C2"]["sgd_step24_gain"] == w["full"]["C2"]["sgd_step24_gain"]
