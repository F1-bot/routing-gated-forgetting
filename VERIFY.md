# Verification

This repository does not contain the study records (pre-registration, freeze record, gate, decision and deviation
records, code records) or the original versions of the code files; the code files here are cleaned copies (see
the README). The checks below are possible with the repository alone; the freeze record can be checked with the
records package.

## Checks with this repository

1. Code tests. `test_c1de_milp.py` and `test_c3_scope.py` refer neither to the study records nor to torch, the
   upstream package or the tokenized data (static check of their imports and file references):

       python -m pytest test_c1de_milp.py test_c3_scope.py

   The other test files need the upstream code (`tools/setup_upstream.py`), torch, the tokenized data or the
   records package; some of them build mock records in a temporary folder. The tests were not run for this
   release build.
2. Data. After rebuilding the tokenized data ([data/README.md](data/README.md)), each file can be compared with
   its SHA-256 value in the table below, for example with
   `python rebuild/verify_bin.py data/dclm/dclm_val.bin --tokens 19999909 --docs 17005 --sha256 <value>`.
3. Tables and figures. The scripts in `paper/v1/scripts/` write the figures, tables and filled values of the paper.
   `fig_arch.py`, `fig_captions.py`, `fig_design.py`, `fig_edge.py`, `fig_entanglement.py`, `fig_gap.py`, `fig_repro.py`,
   `fig_repro_ci.py`, `fill_author.py`, `fill_full.py` and `make_exploratory_values.py` run with this repository alone;
   `fill_full.py` writes `paper/v1/fills/full_fills.tex` for the TeX sources of the paper, which are not included.
   `fig_c3_single_layer.py` and `fig_c4_per_window.py` verify the original output of the full analysis from the records
   package (its embedded content hash and its `.sha256` file) and stop on the published cleaned copy of
   `results/analysis/analysis_confirm.json`; their published PDF and PNG files are byte-identical to the renders from
   the original output. `fig_crossover.py`, `make_tables.py` and `fill_core.py` read the gate records in
   `paper/v1/data/` and also need the records package. `fig_edge_extract.py` delimits the job segments by the
   last-write times of the G1 job logs, which a git checkout does not keep; with the original times it reproduces every
   published value of `fig_edge_values.json`. `fig_arch_facts.py` reads run outputs under `outputs/` and the TeX
   sources, `make_supplementary_values.py` the G1 gate record and the subset manifests under `outputs/g1/`, and
   `check_paper.py` the TeX sources; `build_ledger.py` reads the gate records of the records package and the 39 documents
   listed in `scripts/ledger_sources.json`: `run_continual.py` is published, the final pre-registration text and the
   freeze record are in the records package, and the other 36 (34 study and planning documents, placeholder `<workspace>`
   in the published copy, the full text of the preprint and a staging note) are not distributed; its output is published
   as `paper/v1/data/claims_ledger.json`, and `write_ledger_md.py` runs it. These inputs are not distributed; the values
   these scripts wrote are in `paper/v1/data/`.

## Records package

Copies of the study records and of the original files are available from the author on reasonable request. They
can be checked against the SHA-256 values below. To check the freeze record, copy the files of the records
package into the repository root, keeping their paths (the original files replace the published copies), and run

    python tools/verify_release.py

The script runs the frozen freeze check, `freeze_verify()` in `phase6/chain6b_util.py`, without modification:
the content hash of `FREEZE.json`, the amendment hash chain, every referenced record and hash-list file, the code
record against the code files, the plan hash and the re-derivation of the 30 registered parameters. The 13
tokenized data files are taken from their recorded hashes while they are absent and hashed when they are present.
With the records package of 2026-11-21 in place the script ended with exit code 0 and the line

    [verify-release] PASS: frozen check exit code 0; 13 absent data file(s) taken from FREEZE.json, every other file hashed

Exit code 0 means that the record verifies; exit code 23 means that the frozen check reports a failure. Without the
records package the script stops with exit code 2 and the line

    [verify-release] FREEZE.json not found: this check needs the records package (see VERIFY.md)

Exit code 2 means that nothing was checked, not that a check failed. Without the records package a reader can run
checks 1 and 2 and the scripts of check 3 that need no study records, and compare every file marked "unchanged" in
the table below and, after the rebuild, every data file with its SHA-256 value; the other values can be checked only
with the records package.

## SHA-256 values

The table lists the freeze record `FREEZE.json` (file hash), every file that it references, the original versions
of all 172 files of the code record, the other study records, the output files of the core and of the full analysis
(`results/analysis_core/`, `results/analysis/`), the code run after the main queue (`confirm/post_queue/`: acceptance
checks and the launcher of the full analysis) and the acceptance results (`results/a1_wd_only_checks.json`,
`results/a2_acceptance.json`, `results/a4_acceptance.json`), the outputs of the assumption audit (`audit/`), the code and
the result records of the re-run of the assumption audit after the main queue (`confirm/rerun/`) and of the re-run of
queue positions 154 and 155 (`confirm/rerun_a4/`), the files of the exploratory runs after the freeze
(`results/exploratory/`) and the files of the paper in `paper/v1/` (`data/`, `figures/`, `scripts/`, `tables/`; the TeX
sources and the PDF are not part of this repository) (429 files: 106 not included in this repository, 172 included as
cleaned copies, 151 included unchanged). The content hash of `FREEZE.json` (its field
`sha256`) is `4f88f9dd29b0f1ed91643491715429846ac125b416b282c07d2a5b1caf2b6c1c`. The names of six temporary command scripts and of
two decision records (and of their copies in `phase6/`) are withheld; their values identify them.

The published `results/analysis_core/analysis_confirm.json` and `results/analysis/analysis_confirm.json` are cleaned
copies: local paths, approval notes, role labels and references to internal planning documents in text fields were
edited; numbers and structure are unchanged, and their field `sha256` refers to the original file. The table lists the
SHA-256 of the original files; the published copies have
`5ae0d62d37c812ad12dcd89dc17f130ca932303a1feca69560abe3777a3cf33f` (core analysis, original
`859849244c70fa5ac4ddd8a73ab574ec97cdec33c3de1d5da00da7c6811f09a1`) and
`dde50602b3875efe4428926306039ce53b6c7cb73b21e8562c897841e6699212` (full analysis, original
`d9b3e47598b2ac10745455917ef53251a114a70e5918aed584a53f58f577ae01`).

Fifteen files of the code record were changed or added after the freeze through recorded amendments (A-1 to A-4;
`freeze/amendments.jsonl`, 17 records); the table gives their current values. Four code files and their copies in
`phase6/` were changed; the code record of the freeze (`freeze/code_record_at_freeze.json`, referenced by `FREEZE.json`)
lists the first value, the current files have the second:

- `chain6b.ps1` (A-1): `9bd89cc3f18a2138208e6b52cb50b7959f39fbd85dbbb9308676bdca07bf558c` -> `eee8ecffe1f81aa6e0e3394dcc86622a991dcd27ec02c39d7abc168071126f1e`.
- `chain6b_util.py` (A-1): `c424b76debe598407c3c30989a6234ae18371a05ef98965ea6b67054b215eba8` -> `e6f965838baf88ff6eb6bbdd5e835c22620609b982ca417f989c28331a019849`.
- `test_phase6.py` (A-1): `75cbfa8152e5dce6b340c17f340e49b794cf9f4a630aab9a9bc8cf212f8a66ee` -> `9b0ac169b901650d4f5b1430c94a1f3cc59a48c94484110400be9db594e1c072`.
- `weight_controls.py` (A-1, then A-2): `473ff8c58542f7f398431181cf11afb14b0b8bf0f261591a88120f3fdc6b2835` -> `49e9f45b8f252ae887b49f9e309d804df8cc0e439dd859fb7f0c4536272caf6a`.

Seven files were added in `phase6/`: `test_wd_only_r9.py` and `wd_only_check.py` (A-1), `test_wd_only_a2.py` (A-2) and
the decision records `amendment_A1_post_freeze.md` to `amendment_A4_post_freeze.md` (A-1 to A-4).

| File | SHA-256 | In this repository |
|---|---|---|
| `FREEZE.json` | `41861741cb40ad202c23c081429b6e6dd2ea0ab6d282106f37b263e5565bbfdd` | not included |
| (temporary command script 1 at the root, name withheld) | `015e97443a5315d270c702d4fe1c2f3cff6674a4bbf267bee63c9e2a565519e6` | not included |
| (temporary command script 2 at the root, name withheld) | `13eb80fe4c78131a0db66e50fec6f871a1ced178c2b693e29700b34e239667ce` | not included |
| (temporary command script 3 at the root, name withheld) | `1bafba324ce0b0c937189b1986bb4e644ede27beebd6d7bf0e662f48bd0de4f7` | not included |
| (temporary command script 4 at the root, name withheld) | `a5c794e0a3c49d60be628c1c3d789bf4b6bdcf08afdaca594dff0c24391de791` | not included |
| (temporary command script 5 at the root, name withheld) | `e6725b5129c6fee2e85fa5dc67b3471d7301dbee6facfbd2114bd6028d137e6c` | not included |
| (temporary command script 6 at the root, name withheld) | `e95028580fa2f06f1d8f9ba009d747a765d1fc8ccc169041dcc3c14a22c569ac` | not included |
| `.stage6/g0_mock/g0/gate_record.json` | `aacc760c88036da47a26afb2f913eb3b260632f3e6d130af374a8151e29deeb5` | not included |
| `.tmp_permtest.py` | `3961ea3a5aef2e5ad8fc91115863040394db3831de374187dd987bd48d9b99f2` | not included |
| `adapt_extras.py` | `2d4eeb705556d2ea966ab1dee1a2880a52135e4cf6c9498fcd937c5f9b4e94d1` | cleaned copy |
| `analyze_confirm.py` | `0d82526e1e751230e9144ac28ebb4c07b4157ba91192c7fd14453a746d613043` | cleaned copy |
| `analyze_confirm_ext.py` | `92a05a2045839c2a12432edfffcd49186b3984dafdb15d0d6235267b0a7ba26f` | cleaned copy |
| `analyze_stageB.py` | `3bbe60ef079edf6d8269cc9f213ea4f40ac4c2b4c6bcee0204c41eee35ef3b72` | unchanged |
| `assumption_audit.py` | `39adede34e6922f0d5f2afeff2dc4437d9a92602d26b4f764d8126e1c97de6d8` | cleaned copy |
| `audit/moe_b1/audit_realized.json` | `81427d14dae97296bb7a36fd49ce1df44066257114702c71e0964db000f51020` | cleaned copy |
| `audit/moe_b1/audit_realized_groups.json` | `f1fb1f22b24ddd7ac63199bcea19832b119a45a2a607b8da42321e3064af2ba8` | cleaned copy |
| `audit/moe_b1/audit_realized_windows.npz` | `5d3f25028a7be2bd0cb5de3bfe294523754ee8309f2465708c3bc04fa0e38f2f` | unchanged |
| `audit/moe_b2/audit_realized.json` | `ee35e396a0f77352a070d180c46293664f99f846512538ca98611ff6ef484831` | cleaned copy |
| `audit/moe_b2/audit_realized_groups.json` | `b2dc8371bc12110d828f6b91cc0b9c155b8239f2c4a2165b1858ac141a8f6c6d` | cleaned copy |
| `audit/moe_b2/audit_realized_windows.npz` | `c4537ec8551e6807eea1782becf964665a8df44f5ff5d2a6536f36e1db7bf56b` | unchanged |
| `budget_v12.py` | `36568c53f0626f93e3e8549d43c24651116ffbec7aa2996073a6e2d438b5d1f5` | not included |
| `budget_v13.py` | `64d7b082d02bb60f3444c381e90fd87dac1fb7f1edfc0940545385fa782e59e5` | not included |
| `c1de_milp.py` | `68e2284160da721a27694d7e14ba82f39d4137a874e5b55116a07fc8d3f8f140` | cleaned copy |
| `chain4.ps1` | `5228722709eee67a70c408cd861f5399b820bca271d2434727cf9fae2b58ada0` | unchanged |
| `chain5.ps1` | `a410f3df49b97235f9f8908b216a70dd8c9a1507dcb7340b80c1b3d90490830c` | cleaned copy |
| `chain6a.ps1` | `b8baa8cd7b7c2afbeae7e76c58f57910cca1033b5b24e30cb4a0250872c8ecf2` | cleaned copy |
| `chain6b.ps1` | `eee8ecffe1f81aa6e0e3394dcc86622a991dcd27ec02c39d7abc168071126f1e` | cleaned copy (amended after the freeze) |
| `chain6b_util.py` | `e6f965838baf88ff6eb6bbdd5e835c22620609b982ca417f989c28331a019849` | cleaned copy (amended after the freeze) |
| `check_manifest.py` | `91e8601219e20759592831aa439387880ee86b0b96eb149c3f13f2cb344c9037` | unchanged |
| `ckpt_pins.json` | `22b2c495950699def8f733c4485f518c1e5bf139c8643eb484afcf0fd5334f3f` | cleaned copy |
| `compare_g0.py` | `629d95838c9555b0634818c3ebe1ab77206dd3db3de0e7e340529fc887c40f64` | unchanged |
| `compat.py` | `33a34c09b1d75180fb4a877ea0ee8a8ad64d285cce5416bc5b28c18683f81903` | unchanged |
| `confirm/plan_confirm.json` | `cd05a1b6ead35d13ba2521f377e5e3231389baeb20bda7fbac5e4c6c1e147fec` | cleaned copy |
| `confirm/plan_confirm.json.sha256` | `979fd58839562eece5adffb0d4b1f91713b2a4dbda19ba86b8f5c973ebbb31e8` | not included |
| `confirm/post_queue/a1_wd_only_checks.py` | `39624d29f2df7c6515b0c0049b2f04ca47543e7c5006488968b267df8e051283` | unchanged |
| `confirm/post_queue/a2_acceptance.py` | `dd429deb351abfcd202045b6580a9cfe8df47a50fc2af4c6a94626cd0fd37e39` | cleaned copy |
| `confirm/post_queue/analysis_notorch.py` | `91f852ec7cc4feaef898e6ec2a72fc88514b6e383bdc940c544bb21ae19bffe0` | unchanged |
| `confirm/post_queue/README.md` | `1aebf45670a85fb9e0fcfedc26971ccf814136a5eeb3b9e0829daf89d356f77f` | cleaned copy |
| `confirm/post_queue/run_full_analysis.ps1` | `214f842631d340f1a8d4ac4474fcff2a90d60a091b3a68c37c0dc9ba1da0cac4` | cleaned copy |
| `confirm/queue_confirm.json` | `47c95918229a8a2ed3eca4adc49d7d80fc89f2ad1cc4ebad0c2935622000269b` | cleaned copy |
| `confirm/rerun/amendment_A3_post_freeze.md` | `48b8d72a52acaba9d4884a2660d1b2e8e020a395e0daffb3647caf4b68043a12` | not included |
| `confirm/rerun/audit_shim.py` | `01591bff7f7db9306861a661ecaccc98b3ee4b7902463b6d900905f8ed21d0ff` | unchanged |
| `confirm/rerun/cpu_check_audit.py` | `dd99b0a4504cc63dea591a8b08fb601d296e70e62b317fd6a1e30edfdb3fca97` | unchanged |
| `confirm/rerun/queue_confirm_audit_rerun.json` | `37de9fc326e6aaad905399cc88be5eb56ef3d19a6de09cea68ab98018111cba8` | cleaned copy |
| `confirm/rerun/queue_confirm_to_audit_rerun.diff` | `abd661f6fe3193222f47099b642857702273d9d64a3934521b963076515c619e` | unchanged |
| `confirm/rerun/rerun_result.json` | `8c2ca515960e09374a8fc84732d72194457b1ae1e0f5e81a5d0f8ad87bbcb3b4` | unchanged |
| `confirm/rerun/watch_and_rerun.ps1` | `0eade46a2acc7d80914289c0645c1df77f351f5f76a1203b72e318d0db4007d2` | cleaned copy |
| `confirm/rerun_a4/a4_acceptance.py` | `09eb601f8b8a2c95e13a57788b990192574d38bb49d125c5e4cbdb33861fe5a9` | unchanged |
| `confirm/rerun_a4/a4_cpu_check.py` | `c36e584044ae91365970d00cb7dfb10855c2464c2efcf686463bac0def0e2261` | unchanged |
| `confirm/rerun_a4/amendment_A4_post_freeze.md` | `3d7aae95950a3657442016fde45b99f48c161a4e9f20f9b396924df6836cf036` | not included |
| `confirm/rerun_a4/expected_unchanged.json` | `33f22a4d6e7e6029ab812fee773f77b12ff986418989659b9df1f327ea6c1c26` | unchanged |
| `confirm/rerun_a4/queue_confirm_a4_rerun.json` | `4056fa2d56666c2b81170ae7b4d37a9e657a8462d441b6864f925c61ea2934a1` | cleaned copy |
| `confirm/rerun_a4/queue_confirm_to_a4_rerun.diff` | `8ea5d926bddfa0cdc5a072f86b01c0c12c0cfda2fb7a6891a75410d497cfe012` | unchanged |
| `confirm/rerun_a4/rerun_a4_result.json` | `c66a563e528ed062d70c70a21f5d09fc4ec21588c7fa6a5a0aff2d4c28dc5673` | unchanged |
| `confirm/rerun_a4/watch_and_rerun_a4.ps1` | `cbf0769ccb27a04494304a7482f5921fbffa52a5bbc49baab56851ba4ab11a1e` | cleaned copy |
| `confirm/specs/c1_moe_b1_books.json` | `68c9f71ff07fa3f0d219b1328fed2b92e147c8ea8600d3112a51221200a617c7` | cleaned copy |
| `confirm/specs/c1_moe_b1_books.json.sha256` | `c204a6b8f1c40d04789665cdb91f402e1aece5b2697bf70d2b3bd594ea01a44f` | not included |
| `confirm/specs/c1_moe_b2_books.json` | `2e0ce3d6b739de4d7be5740e0d73400ef8a983d1d99bff5a8698ea950d6a8b03` | cleaned copy |
| `confirm/specs/c1_moe_b2_books.json.sha256` | `d9b8210a4a157c4aa2505e300c37dd62ea9097e12ef95df3b3ccb79f5f59422c` | not included |
| `confirm/specs/c1_moe_b3_books.json` | `c49bec42bc49210ee0f0ae18354d051d046e01137bc73beb97ea8c7ebbf55908` | cleaned copy |
| `confirm/specs/c1_moe_b3_books.json.sha256` | `14efcf60baabb789f579ad16fd13bab9bb07970ad716b0f6ecd688efeb1a2de6` | not included |
| `confirm/specs/c1de_moe_b1_german.json` | `6834f6702c475f6f52f59615fe0a7c34128e010e5b07144013508745939b1069` | cleaned copy |
| `confirm/specs/c1de_moe_b1_german.json.sha256` | `bbe47d65a277699a59fb8ddb4ad43a5a10f770c82c14612888281bdd43ff9ee5` | not included |
| `confirm/specs/c1de_moe_b2_german.json` | `ded023cc12f885ce77628437295f02924dde47f1653440d04ce5075285ee3c37` | cleaned copy |
| `confirm/specs/c1de_moe_b2_german.json.sha256` | `e5768a59e5fadcca2eea8e1179a0a985046a7a7d8647489c8e9c692e9d658dab` | not included |
| `confirm/specs/c2_moe_b1.json` | `8084928a5fe0432ff8836597f7f29f391bbd4e7565e2dc8235c7670a2db7e9c2` | cleaned copy |
| `confirm/specs/c2_moe_b1.json.sha256` | `a13177d40cc6dd75b130be4753016c81c6851ff3c4141e10bda1d41952389b6e` | not included |
| `confirm/specs/c2_moe_b2.json` | `e16db0833a7317c2e7701d362907653b7b497af1841ca46125c2d017a409dfa9` | cleaned copy |
| `confirm/specs/c2_moe_b2.json.sha256` | `1d229d2a1843209c41e5e2e8f02449d4a6f533478eac548b4b2592e95d80d37d` | not included |
| `confirm_plan.py` | `757a94865ee1b02edfacdf18c209cace3ed0faf2b70c892cdbc1192d4b6d24a6` | cleaned copy |
| `conftest.py` | `93497ae432770eb1374cc081ee554407a69df1aed54c682f7fe94421c093fdef` | unchanged |
| `crossover.py` | `d3fcf6c2ba65fd8bb7f67c45557e234fab587547048fb4feb36625e5b960c1c1` | unchanged |
| `data/data_manifest.json` | `83a3494b8546f5d8bea9c3766fe70c4e673960eeafcef49adbe9ee6f372700c3` | cleaned copy |
| `data/dclm/dclm_cont_heldout.bin` | `7000c98f95bba8b47ec5c06c7ec1a1114811d30eaf939dfe737f5dc757e626d7` | not included (rebuilt from public sources, see data/README.md) |
| `data/dclm/dclm_val.bin` | `0644eac43505ad8f3ba78c8dd8580862e7b69dcbf24080dd3e8082692186ed04` | not included (rebuilt from public sources, see data/README.md) |
| `data/domain/algstack_test.bin` | `952a5831acf24269be46cba6d0231fa77173ec7bc2e101520bc42d019397ad85` | not included (rebuilt from public sources, see data/README.md) |
| `data/domain/algstack_train.bin` | `24a089e7b56a7571a1e71f773850aece3d89722e6245fbaf5e10b4401866a781` | not included (rebuilt from public sources, see data/README.md) |
| `data/domain/fr_test.bin` | `b989c7ed77d53eb67b2f1590b615d673428eacc1cafa88d1cb06d851f8b571ed` | not included (rebuilt from public sources, see data/README.md) |
| `data/domain/fr_train.bin` | `0a84d8b65c17372d4e9d4f3b80aa92a6b0951ade069850a520b266f6dad8fb97` | not included (rebuilt from public sources, see data/README.md) |
| `data/domain/fw2de_test.bin` | `27e080a102cd6e8a947a3e157ae0d3a357e0489d916f5c117231d03439b11877` | not included (rebuilt from public sources, see data/README.md) |
| `data/domain/fw2de_train.bin` | `c0ce16fba68ae066bdedd4e0a46f3e2ad29f5e6cc08904cd6f63d9273378450a` | not included (rebuilt from public sources, see data/README.md) |
| `data/domain/owm_test.bin` | `77cc6d9fd78c434bb380aebd461dffe6a9f2e710cbd6b600249f8c6675011d92` | not included (rebuilt from public sources, see data/README.md) |
| `data/domain/owm_train.bin` | `3a6bb73e1a3d43f9910e49455e325b76118ac436e31089301b63956c14138d27` | not included (rebuilt from public sources, see data/README.md) |
| `data/domain/pg19_test.bin` | `3ad95cf1304d1a2482333635efcb36d01aff6c9d31a4342ccab0ee75bd6b05d0` | not included (rebuilt from public sources, see data/README.md) |
| `data/domain/pg19_train.bin` | `24ec92cb6dea867c575898eafafe8f2ab83aed96f3f6411d77a3badbcbed036f` | not included (rebuilt from public sources, see data/README.md) |
| `data/streams/pycode_eval.bin` | `08926d4af5b2d387d27aeee161b178df058a2a0e1cd9c3ca50194fda0cfb2a88` | not included (rebuilt from public sources, see data/README.md) |
| `decisions_v13_points.md` | `c2442c069622e852d78c685b8250b48acc87b0470eeb1d896662e44299e2e27b` | not included |
| (decision record, part 2, name withheld) | `3d9d34b5a361a0fb931a3e7673e757766e242ae3482e420faab821d028aa1840` | not included |
| (decision record, part 3, name withheld) | `779f5143cead4ed7fa7c442081f475f06c44fe01a089ba00767e82995f262829` | not included |
| `dedup_check.py` | `e815668b1e61bb21aca77e1986a04a482ca5c183031f1a4321a90a84e82af489` | cleaned copy |
| `dl_dclm.py` | `db7a2522254b5f655cf5e1ea04d7c00ff918edaf7a25ff9af0770431afa3a1c4` | unchanged |
| `dl_dclm2.py` | `246fe2c8a9b6b1e5616d09bc953b69b580049f82b637d7d640af129c6a84e70f` | unchanged |
| `dose_draws.py` | `bbb3c5c03dfb9cdf80f7e08740eabc7554a1fbbcbdc4525f101b34f20562aaa2` | cleaned copy |
| `env_check.ps1` | `a0834bf74a44975f08c1e97cbf0cb74972683a37029e6aa23c6a46c2fccb46c3` | cleaned copy |
| `eval_ckpt.py` | `6a610c2a69fc0b1cf311bcba5640b6e3412537c77d75d284e89869d28045beae` | unchanged |
| `eval_windows.py` | `0f909d1e23c2e1bd3ec3765778ddf927ad2e3ccc036a6c1eadce8e822f2ecb41` | cleaned copy |
| `forecast_io.py` | `c5a461aa038e72a827d215e6f91c854b5f2d30641e9816fa49c4bfb3d60c2983` | unchanged |
| `forecast_probe.py` | `3e1cf5b78c382f9239b7e0e75dae3ceda4b81b415f7631cc679aca958d47a38e` | unchanged |
| `forecasts.py` | `035672940515d8590163166210e3e7c51d098e479ad898c097d9bed9ceac738f` | cleaned copy |
| `freeze/amendments.jsonl` | `b69bb2f478d585cb19e121e05dd15acb6f38123445fd3f70951d436a1782872c` | not included |
| `freeze/code_record_at_freeze.json` | `a32c85b6adad397ee19fa71d50e527d7a7c04485f6e34063e6f52f2d25d5238c` | not included |
| `freeze/code_record_at_freeze.json.sha256` | `2ef560dad284e8f21f6c390546c25f9dbd688f591b4587882bbfd492ed2bc040` | not included |
| `freeze/forecasts_v13.csv` | `4f8677e969e005560da9feeb542194b057d45734135b49c5079eddd77b56025a` | not included |
| `freeze/forecasts_v13_meta.json` | `08e6c08b5ce762ad2c5ff7831e57c25d27de7603b1db6ac96f9ab7c1a9ea8dee` | not included |
| `freeze/freeze_record.json` | `41861741cb40ad202c23c081429b6e6dd2ea0ab6d282106f37b263e5565bbfdd` | not included |
| `freeze/freeze_record.json.sha256` | `b9cb87147bde2dc6c80d35b9d334c5c7e24a1e27d00e66880c4c8da5a4d3c8da` | not included |
| `g0/g0_gate_config.json` | `feb8418887353c4734b6fa0c30a691c3fda8f6707310c553640e2d971f99312d` | cleaned copy |
| `g0/g0_gate_config_b3.json` | `eff3bffd1b3962bcb3b44fd266246ba2806bb39a2c45a10577179ba9ea36c6ee` | cleaned copy |
| `g0/gate_record.json` | `bdf3d79c773bcf7ed417b12768ca940e56b2b1b110d4121c0e777e2bc4202ff7` | not included |
| `g0/gate_record_b3.json` | `fb9ae3679e441085fdf2df9e2b302bb5183daafc2734322f75f01b497638483f` | not included |
| `g0/gate_record_registered.json` | `9259ddd619332de2a6ce72e6ba18458a36dcfa86cd81ff9a6965a4a46d978518` | not included |
| `g0/moe_b1/margin__manifest.json` | `845400eb3cf999e0bfb6b8e8007c4b0cc40d6ad904bf3baa03517bccf6803282` | cleaned copy |
| `g0/moe_b1/margin__manifest.json.sha256` | `6f8b3d81ba622262d07dfb072706657f08628d36c11fca53ca6d8ef7ea8fe03e` | not included |
| `g0/moe_b1/margin__probe.json` | `f31e0dac42a9958962357d6898631ec5994a8ecf1de9fade723ee58940be4be4` | cleaned copy |
| `g0/moe_b2/margin__manifest.json` | `38a6fcbf810fc97a127a5629d2132c9c20301c61cb440cfbb94cfdef02dd5ff4` | cleaned copy |
| `g0/moe_b2/margin__manifest.json.sha256` | `c5bdf48b182637b1ac83c5c51c991c32d82edfabd262b3127d79a57a94c4b04b` | not included |
| `g0/moe_b2/margin__probe.json` | `ef5c22db1da71fb43490427807d6f6f911af6973515abb7760c0aec7294fbe69` | cleaned copy |
| `g0/moe_b3/margin__manifest.json` | `597cc9ea76010d92bcfb28eb1f7f7de22992e87aab342b47f87b7e91d2e190e8` | cleaned copy |
| `g0/moe_b3/margin__manifest.json.sha256` | `9080187b4704dddf4f6e3e40c427cb603900f55871656c48ff77801609296dcb` | not included |
| `g0/moe_b3/margin__probe.json` | `6803171fc905d2e4f89166b963d5251665e7b83a3adf1f37291aa0f61817d1a4` | cleaned copy |
| `g0_b3.ps1` | `ccf11b15429d8ccfde9845256278ebc1d9695b0fb980165ad4efd9d17a5bcabe` | cleaned copy |
| `g0_code_freeze.json` | `ccddc6e14ee613b6ce2107a459c13eb7681a1a488a7dcdb80b75756e67ec9115` | not included |
| `g0_gate.py` | `bea5981dc3e3e75a3e65f65a221b936bc9fdf830ae25f660d6d099889ec1332f` | cleaned copy |
| `g0_gate_config.json` | `feb8418887353c4734b6fa0c30a691c3fda8f6707310c553640e2d971f99312d` | cleaned copy |
| `g0_margin.py` | `b92f9e83cc3153f25b712db4ba6a9b3d6d4fff7040abb543a3eba0b8e590f144` | cleaned copy |
| `g0_moe_b1.json` | `ae189504bf39de0497f4986d2f0c163794ab876490a2562a7a34562d58e9bd66` | cleaned copy |
| `g0_moe_b2.json` | `8d9dd1d91dc5d50a1060356e293cc980e53c747dd13786bf08a7a1b843f50c56` | cleaned copy |
| `g0_moe_b3.json` | `25d1bfdff02e919ff3532fa6b1b78bfcdd22c929328a1aea3650ffcb8e9c69f5` | cleaned copy |
| `g0_nestor.json` | `4d6994fbb1440668a36314982c921efd8e89924f896e08a16c72ca57c3a0b5ba` | cleaned copy |
| `g0_probe.py` | `966403f8ce6cfb5e4a04bb9bf7c7a85a2cb37ea8577a627f498230af4a25f3fe` | cleaned copy |
| `g0_probe_v2_orig.py` | `db99a6db2371ca40123655ed4eea797cbea012e4dabb705928bd81fd522a2f95` | cleaned copy |
| `g0_rerun_compare.py` | `24c5c08fd42e9b7b0f3e1579b973fdb6aa28e3c77fbc1b92af4963044f80a231` | cleaned copy |
| `g1/g1_gate_config.json` | `ace7a63dce0850ad6c9a199854dc9fbd17c08c97111631f72f1d05237844cb23` | unchanged |
| `g1/gate_record_G1.json` | `9a2a82ed168924fff7be139c2ae6113d9e274a6abee7f81152d12b99db7a41c1` | not included |
| `g1/gate_record_G1.json.sha256` | `dcf66901e4601fc7989125f4ea21950fff766af47e79086563802a89b355c06e` | not included |
| `g1/plan_g1.json.sha256` | `85a6f6aea2d0f8e5af051fd0af76d5ddc5fe7c29e3916c02fe06c2d27278dd85` | not included |
| `g1/specs/aa_moe_books_a2.json` | `1056912c7c45cbe22de9de8ad77ec50eda96b7bd7d3ef18c4411fb02ba6e3520` | unchanged |
| `g1/specs/aa_moe_german_a1.json` | `0ada75fd50c0cf007978f37bffdc3b8c7a27f0f6fface40673d55a00dd8818cc` | unchanged |
| `g1/specs/aa_priv_books_a1.json` | `449bc61af5f7f0c5220a25fc0273dc58ad1d1f7f172bedfe1973c3d6da8f29be` | unchanged |
| `g1_gate.py` | `63b4ec7a9e1742846847d1bcccd5a2283aa9f7eadcf782a1394a8056e25986e7` | cleaned copy |
| `g1_gate_record_MOCK.json` | `4cb2b27cad339165a5ced3e1b7bdc300b281c879b2866eee1daf86f8d7649b11` | not included |
| `g1g2_config.json` | `298b74d9da035f12f275ae008044c4d8a11a4de55760c82da5758dd826280520` | cleaned copy |
| `g1g2_config_MOCK.json` | `a53b8511cea95c97ae300367f960202ff8d3f71d6268a4f690eb93cba384c750` | not included |
| `g1g2_plan.py` | `de361c6406f2d711214ce4cfe60ba60a2be30806fab48e44a480cd36562c1339` | cleaned copy |
| `g1g2_spec.json` | `cf6ba4ea342b7aae5468736cedf531b0cb8d6aea4dadb7166f604e2485ce29dd` | not included |
| `g1g2_spec_check.py` | `0b5cb5ee8eda6ab60f00bcb821effd378b289590b75220628283864bfcabf9af` | unchanged |
| `g2/g2_gate_config.json` | `bf45bb48aeda69a68a47641a8730bf189b330e8c71f92158a7a70632b9562c08` | unchanged |
| `g2/gate_record_G2.json` | `2c526ae8989ff2a4614b839993d95f3359e58edb102a3812f8990498b6b9f854` | not included |
| `g2/gate_record_G2.json.sha256` | `165f4935cf5330a7bb0f7f492cb31382212f3238ae9644b9064c91758b7beab8` | not included |
| `g2/plan_g2.json.sha256` | `f34064976e0d1d9ab55523038a51d6bc3931f76c610c9e02d9cb0b65c8517583` | not included |
| `g2_gate.py` | `16a8709ea6bae9dda11ac1646d492db739465ed8cce45eb79d921a8d0b58c825` | unchanged |
| `g2_gate_record_MOCK.json` | `179f68f98cdd0b8d5a17a136e0ef983a09684fd88c71347595ce99cc0c8ebae8` | not included |
| `gate_common.py` | `b83872ebd1093287f946d432f9357057d41d8ce03459065cc3d59b91b433a8c3` | cleaned copy |
| `gate_rules_v13.py` | `654c5ff8883644d0d5d4e0f90b46c8ae5bbd4d9d086473225e00be0a5fa4e23c` | cleaned copy |
| `grad_snr.py` | `93d09556f5ce3b9b36dc3fd06fe813d6c883d91c96f9539d0c1b5963ac2c9cb5` | cleaned copy |
| `hooks/after_G0.ps1` | `831bafe5641433d6101cf3038e378d86018ec54480716fbb4bb2cb5400bc7585` | cleaned copy |
| `hooks_after_B.ps1` | `8ec1a2a860ecdc5232111cc9002ddfee96ad50935065407e3db58740c1d69540` | cleaned copy |
| `hooks_after_G0.ps1` | `a1497f6890cd2575a3b64fecc214c4494089df4608577ee0c4fa77fc04ef0f3b` | cleaned copy |
| `inputs_v13/entanglement.py` | `32305dd63c3bded9c1534606b70fe46bba5bf9d784c073d9fb6755e051899d71` | unchanged |
| `logs/chain6b_code_sha256.20261113_174045.json` | `b310827d385543b05b787bb9cf894b54e2622f6bf6f6e2e6a0242da77b9b7fe2` | not included |
| `logs/chain6b_code_sha256.20261115_015342.json` | `72b5d6b10192ad2b5dc1d05b65b84f2ea214a26ddbd574ad97774206fd8df389` | not included |
| `logs/chain6b_code_sha256.20261115_015647.json` | `ae5e83a5310e1321548f5b62c79d36b8a11e6d19df3fa6b55b5b1ce5ab4df582` | not included |
| `logs/chain6b_code_sha256.20261115_055334.json` | `36c013585c69b5ddb2bb1258694667d3e08bad8a78a80ac5b38ef3226e920256` | not included |
| `logs/chain6b_code_sha256.20261115_093520.json` | `09fadd3296aa90adc2925a70581337204cc9f883a52e6c4a1de5010676a66134` | not included |
| `logs/chain6b_code_sha256.20261115_143708.json` | `a32c85b6adad397ee19fa71d50e527d7a7c04485f6e34063e6f52f2d25d5238c` | not included |
| `logs/chain6b_code_sha256.20261116_135826.json` | `6c66fed7bdd9281fe1bbf9d5d4d89b4783e2d948cd0660785b370940a3691a84` | not included |
| `logs/chain6b_code_sha256.20261120_060513.json` | `19f1a94782ce50fca3f998f395a17eda6edc821669e822f1e640b07773685d8b` | not included |
| `logs/chain6b_code_sha256.20261120_130432.json` | `96e17138de228c22b75d4e8042c89924d315aafeaf04767b8fb59d9bd3dfa261` | not included |
| `logs/chain6b_code_sha256.json` | `ceeff62c813286dba75ca6fe04fca198e2f58c633513e6a989927514a665540c` | not included |
| `logs/env/conda_list_explicit.txt` | `8dae04d2ebe554175bae1e088db19684d749ab03463d122982e694df3aa6d587` | cleaned copy |
| `logs/env/pip_freeze.txt` | `6013ccfea029be866b4dfe250a7a6d88aea91cb02550784d9d680215a3fee72d` | cleaned copy |
| `memguard.py` | `3f0ba60f4ca944067e5acb8a300b702e6f3cefd37307db35ab253d54e0cdc098` | cleaned copy |
| `mock_chain6b.py` | `798d359b9940b6311ecfaaff52dcf1e5200035db54c497f1933a74a0906adaf2` | cleaned copy |
| `mock_confirm.py` | `b16a1f7f09fe2cdce3cfcd42070a783b59cfb1ae7c15d448682040e41df7da51` | cleaned copy |
| `mock_g0.py` | `caa90e4481e6b06879d6bd40044cd79fe4fb95bf08499d2c4f2aef51855990ad` | cleaned copy |
| `mock_g1g2.py` | `ad2eede870e5b5c43fe7cf8058fff121fe7a785e85e8906283434105f0955b33` | unchanged |
| `msvc_env.ps1` | `b68f63129d2a46ab36daf6e471896a3f7f5b39d7a2375429d406fe32a72ea595` | unchanged |
| `nestor_books_feasibility.py` | `8739b64cdfc744a17920e077988c8c867a8551685f949e06941cc3f7b482fefa` | cleaned copy |
| `paper/v1/data/c1_results.json` | `8deb6b7eb39449c4a312f74045fd55df0d12dc06ecb97ecf13c74b1ddf216881` | unchanged |
| `paper/v1/data/c2_results.json` | `3eef8e34fea0936cc00810f065f501685efdc7da02179aa4e415e528e9bbba66` | unchanged |
| `paper/v1/data/claims_ledger.json` | `9482ba69b8da243eeacf932f03a9f4e56feb58e8690759dd14f69bd039fff830` | cleaned copy |
| `paper/v1/data/edge_overlap_I2.json` | `ede74fb4b93fdb4ec805714577ce5c521d0d5565ab77c540be009a8758cc2e95` | unchanged |
| `paper/v1/data/edge_pre_freeze.json` | `3e49016b992a7ab9b74017a35f1c23b37965665e82f1761db5e131c444305d29` | unchanged |
| `paper/v1/data/edge_refs_check.json` | `759cbeee2daaa5ab7e7a156d6b556edd9338b88ce670d90b38ff1a0496a91d41` | unchanged |
| `paper/v1/data/entanglement_results.json` | `e31e49df650e86e0bfceda02666103be18c7da615abba3119b3b30d9db16909d` | unchanged |
| `paper/v1/data/exploratory_values.json` | `e8e64566a807b86a5c567d038100a74d7bc52736195146ab12e7e00238f48129` | unchanged |
| `paper/v1/data/fig_arch_facts.json` | `59715ce652616918e6a7ec799a6940890b1898b0aad4405c32c84b49f79828b0` | unchanged |
| `paper/v1/data/fig_edge_values.json` | `ad11c9ad6092e45c16ac2b587652ec33137f3eaf6bf62483387ee3414c5bf761` | unchanged |
| `paper/v1/data/fig_repro_values.json` | `8bbcd0b0b5804888b8f6566676ffe0192ec1ef88eb44ae2645414e88d58c0a69` | unchanged |
| `paper/v1/data/full_results.json` | `74becacae10ffcab9722cec46818b95d83d0e2a2adedab313526a35af77948cb` | cleaned copy |
| `paper/v1/data/g0_rerun_compare.json` | `d953cce7c2ed283311404331080ac083a28140cf82e821c489b6c785667fa922` | unchanged |
| `paper/v1/data/gap_trajectory.json` | `df6214fc3b1171ee1d15510b890d3434660f04076d7c50d283750cd6e33e2ee5` | unchanged |
| `paper/v1/data/gate_record_b3.json` | `d47e98e550c7c09d4ce6d2365a6ae88c904370580839cefd55d7b79880e8209a` | not included |
| `paper/v1/data/gate_record_G0.json` | `375395aa767b5daf8248b4f2a0592073bf4ec7958f0ef7a26e8d4a72a3895197` | not included |
| `paper/v1/data/gate_record_G1.json` | `c707202ee35b685f9440cdb3fe13ec6f127b7ea7abcb84729d3e530e3a49ae71` | not included |
| `paper/v1/data/gate_record_G2.json` | `1d7e76a5421b93f6c18ee7306be65432ef7001e3eb7ca5a7bbbb894194bf6848` | not included |
| `paper/v1/data/hardware_local.json` | `0c12b9b13e8386b0159d07b6fcabbb41a7100b19f23f2df32395d4aa9fa5a88d` | unchanged |
| `paper/v1/data/hours_outside_state.json` | `b38d73befd5b132346cdd19b07fd3bc2cbfe3ee39e5f396768c8e81a4404432e` | unchanged |
| `paper/v1/data/nestor_books_feasibility.json` | `4ea49bb82b27dc1fc633ce71b0b93abf6f49155850301c4202989ae938ecf9f4` | not included |
| `paper/v1/data/palette.json` | `1158c69c63024aaa4e935039721f58331ed0c5ed6e4c637de54834c4f9b84d4d` | unchanged |
| `paper/v1/data/provenance_b15s44.json` | `b1b7f0ebaa434f28b45eff9db873a2584258e91bf17b65cbb3493abb5413707c` | unchanged |
| `paper/v1/data/ref_verification.json` | `479f1d199d41f356a451468de7c3cfe81b3968071525145cac225bad6213616d` | unchanged |
| `paper/v1/data/reproduction_stageA.json` | `e50d9126dd20fc329caaedabb9033048d93127aa75247cb13812086916d9749b` | unchanged |
| `paper/v1/data/stageB_window_ci.json` | `9ddefbbf58742d67f12f9a4af00a74033237eb7ff2819e15f4dc21acdd13e225` | cleaned copy |
| `paper/v1/data/stageC_results.json` | `bb937ba3c21766c7a42958fa9560c257c5489a4e47b57e017da5ba1ffcf131a5` | unchanged |
| `paper/v1/data/supplementary_values.json` | `c3aba22c748f0f7b93dfab2a8c4e19cfb1f184c5efc4d35414f9a7b32728ca0c` | unchanged |
| `paper/v1/figures/caption_drafts.md` | `49268a224a3485adf1098ed7904ee1d9dac099a329551c25ae18822911a32ad9` | cleaned copy |
| `paper/v1/figures/fig_arch.pdf` | `f83fcd96c3d83075b89ee83240995df08ba29587d7ddfba3d8a741d2d2d90040` | unchanged |
| `paper/v1/figures/fig_arch.png` | `a93b3b296f7b336eb2c5a3a7688de03f8f21dc20725932902c5aa2e84ea62cc5` | unchanged |
| `paper/v1/figures/fig_arch_facts.json` | `59715ce652616918e6a7ec799a6940890b1898b0aad4405c32c84b49f79828b0` | unchanged |
| `paper/v1/figures/fig_c3_layers.pdf` | `b6ea8f45b821f7860a9ce4fdd5035c5ba132046559bc61123f6f7120cc8dcbac` | unchanged |
| `paper/v1/figures/fig_c3_layers.png` | `5b06a9802a3b9f583c6405a36e84c3790250399a034d0f1522e94a3a59ec5980` | unchanged |
| `paper/v1/figures/fig_c4_windows.pdf` | `ba9d8ace2c14cf0d62afaee55f304d11b93cf8eb57d12d409ee279c09d42d3ff` | unchanged |
| `paper/v1/figures/fig_c4_windows.png` | `25bd53b7c6a61b787cc5885eb07676b679379095c9340a66abc2e9cb7465275f` | unchanged |
| `paper/v1/figures/fig_crossover.pdf` | `6a1f8f7c1f6f3b31b4c60a98b89afb0a848c83521eed686ff5c5dc4aba37ae5d` | unchanged |
| `paper/v1/figures/fig_crossover.png` | `fc51e7878a1c9231cc40169b48582e5adc30e370aa7179b2c6ae42908a303b7d` | unchanged |
| `paper/v1/figures/fig_design.pdf` | `d26bb2cc54c202a3b95d0e1303273f80c2aa807002d6040e0be3ba427e0578fa` | unchanged |
| `paper/v1/figures/fig_design.png` | `d3a9a7156bccde5eb420cd301ede68a57cbb6c3327c4fdc52615d1bf579f91cb` | unchanged |
| `paper/v1/figures/fig_edge.pdf` | `f36f0bccb5454ea97400a11419361d1ecbd0a0bb99e81479f605836e9e534310` | unchanged |
| `paper/v1/figures/fig_edge.png` | `99fc02b6f107d44b6c3f384ef4d055c85c6ed4e18b7e7794b56964045263e2d6` | unchanged |
| `paper/v1/figures/fig_edge_values.json` | `ad11c9ad6092e45c16ac2b587652ec33137f3eaf6bf62483387ee3414c5bf761` | unchanged |
| `paper/v1/figures/fig_entanglement.pdf` | `16a7aa36a4534aeff79f912ee704976932a8a2923e1412c881c415bd628c8ae6` | unchanged |
| `paper/v1/figures/fig_entanglement.png` | `ed1341379e97a1ff33ba0a36d2236ef21a80f614727d2e6852dbe04adbebae18` | unchanged |
| `paper/v1/figures/fig_gap.pdf` | `c54a23374b1852deee0ffcc7506cf799ca9e7fc2669f83e171bcaddc7c0a38df` | unchanged |
| `paper/v1/figures/fig_gap.png` | `7b0004b80645f1fa6b4571e3adaa06f7f443bd50dace5143b40a7638b3f12266` | unchanged |
| `paper/v1/figures/fig_repro.pdf` | `5318ac3cd0aac78e7b0e2f1b1c9da9b51a124da371a20c265712f9d8670f0d11` | unchanged |
| `paper/v1/figures/fig_repro.png` | `e9377793f6b954abf6cd2ccf519d0170a8f4bb0d506b30413c9f98108a14479c` | unchanged |
| `paper/v1/figures/fig_repro_values.json` | `8bbcd0b0b5804888b8f6566676ffe0192ec1ef88eb44ae2645414e88d58c0a69` | unchanged |
| `paper/v1/scripts/author_slot_variants.json` | `710cd4bc27c34c40019185295a348eb7a87d86377d1976aa14d9cb41f5119f8c` | cleaned copy |
| `paper/v1/scripts/build_ledger.py` | `1e9f9f23abe5c014643af409ddae19284ffc49501603f70733febda9df98b336` | unchanged |
| `paper/v1/scripts/check_paper.py` | `c807cb4f3151fac4a561040698f61995a55c61ff82f61469825d794edac82749` | unchanged |
| `paper/v1/scripts/fig_arch.py` | `ca50a18140ca6cdf3965137d8b0168e89501fe08ab133a0cad51e0e72f35dd71` | unchanged |
| `paper/v1/scripts/fig_arch_facts.py` | `17bc8d6dacf8c1751653f61989b7f0627068b1b0337db437f03ad5b7ab71c4d7` | cleaned copy |
| `paper/v1/scripts/fig_c3_single_layer.py` | `baa800788cc0012631be022660aef224dd134a3ce9343494b3c60cbd0b28095a` | cleaned copy |
| `paper/v1/scripts/fig_c4_per_window.py` | `e5043c19f4023ff86b92f17f555ddbd95000e149aca627ebbec46a8c0d989e36` | cleaned copy |
| `paper/v1/scripts/fig_captions.py` | `ddb197c272cf1598586b233e62b369d1153a3591c281bcb15d2f38ddd15140f9` | cleaned copy |
| `paper/v1/scripts/fig_crossover.py` | `8c36a81cfe615d38be8d1b3d4ed0693df18d6142933b3be42d034847edfce046` | unchanged |
| `paper/v1/scripts/fig_design.py` | `fb83c5f6aef0ebed3db8d62b78e8a93a584b45d926d3f516c32804c41f4bc5d9` | unchanged |
| `paper/v1/scripts/fig_edge.py` | `3fca47dc4498995fa27ddf057e58366206ae30d014ae054de88e7ed5a0defe09` | unchanged |
| `paper/v1/scripts/fig_edge_extract.py` | `7e485c515c69e3bb5f005f89eea1678ae992cc30d0546c7fc2152c6826d38d90` | cleaned copy |
| `paper/v1/scripts/fig_entanglement.py` | `fae1d6c61a912a8b89d0dfeb8ab638ce49dd9b6246f663e1f02a158f985898e7` | unchanged |
| `paper/v1/scripts/fig_gap.py` | `824c91f3c6aee25946d711acf3f92cc7eca728c670d2570df621ad72df0158eb` | unchanged |
| `paper/v1/scripts/fig_repro.py` | `5c12aec6b2f16ba4e44f3d6e58a47489e149629dad0b14c31e372fbb3c8ac41e` | unchanged |
| `paper/v1/scripts/fig_repro_ci.py` | `06eeae13d31eded98c484201048d879c49c0cc8c804647f9e1708e6f3472a249` | cleaned copy |
| `paper/v1/scripts/figalign.py` | `8a618e1b8baa0cc33eac8d3a60b7affdbccf480c41706c1790d817c7c11c9e2a` | unchanged |
| `paper/v1/scripts/figcheck.py` | `b4bd3dff2e07aacc6b0bab5ae9d0b6d3a41a15f48d2e0036760980b55c6c459c` | unchanged |
| `paper/v1/scripts/figstyle.py` | `67759f4de93a69f3c5ddfcf7af16ce248983e9b773d4063f908c2f3c97c78076` | unchanged |
| `paper/v1/scripts/fill_author.py` | `8ab438f7ea18c1060b4066ce956df2c07888f37d056c4d9ea7502c07df998d33` | cleaned copy |
| `paper/v1/scripts/fill_core.py` | `5d5c75d36ddaa6eaf7e6d8928bc16fee32ed5584f1d3af32f6a0ef917cb56179` | cleaned copy |
| `paper/v1/scripts/fill_full.py` | `b66b2f5757a87472bb0c59e39f23bb0b9cb17ad3eda4a2666c2d4340aa6f303a` | cleaned copy |
| `paper/v1/scripts/final_texts_core.json` | `d556042b454470cec1809a6ab0227aecfe7a37777467c2842df426d17c775020` | cleaned copy |
| `paper/v1/scripts/ledger_manual.json` | `4a579ce0e76e8cdd7bee3568974c8be1f4ae4350d4cbbfffe4aae1ae42276ad4` | cleaned copy |
| `paper/v1/scripts/ledger_sources.json` | `571871071bb3b862ba0755a657b41d56f191b2226c2694b860119859593567f2` | cleaned copy |
| `paper/v1/scripts/make_exploratory_values.py` | `443059be125a9b01bf108b465f479663c2f6843900b8fc193aaaf4a7de8b1179` | cleaned copy |
| `paper/v1/scripts/make_supplementary_values.py` | `b689d3642eb4c88dca891b7f67d994632930a82e9500288bd5ea069aa95072b5` | cleaned copy |
| `paper/v1/scripts/make_tables.py` | `c372529cfa4e86a9b443c565c417f1e6939624d316b95a8d0ceecee12fdabf48` | unchanged |
| `paper/v1/scripts/placeholders_full.json` | `34fb4191bdb6ea1465cfb4ff9a59c42ea3fc81fbcae54ef8986f556309321319` | cleaned copy |
| `paper/v1/scripts/write_ledger_md.py` | `6f71347c7d64befc75aa6637ff69905fa0622060e467a684dc43a6e98f1806be` | cleaned copy |
| `paper/v1/tables/block_status.tex` | `afd6a8b18eef0d91a12e65e77eb213d77bc3e6b6c941f1bfa3d4b364e91e698d` | unchanged |
| `paper/v1/tables/budget_final.tex` | `5ad87be3802bcbfc6cb970324392f5632fa1b1e65d72a68a48c07e8fb4ef7303` | unchanged |
| `paper/v1/tables/core_draws.tex` | `8ce436247dace232fc31472fdc2759910a274a26cb61325dc1dad431ef7961ec` | unchanged |
| `paper/v1/tables/core_e1.tex` | `8eb0a92a9083546904a651c4e623a8f0f32d3643143a41462705d32d5fc09c59` | unchanged |
| `paper/v1/tables/core_variants.tex` | `2e5b5a4b9407d58c10bb3cb4a72d6f5eeafd95de01a9dd0a2b67b2431ca69ef5` | unchanged |
| `paper/v1/tables/edgepre.tex` | `0cc77994c09eac26112ed1580c9fe42f55244c471a799fc397a6df41205b5932` | unchanged |
| `paper/v1/tables/entanglement_full.tex` | `4d48906706327ebbcc047d5784e19fbb2cccd6a802d8b64266be159d54c86365` | unchanged |
| `paper/v1/tables/full_audit.tex` | `bbdc647293e098a8d0e8f1b0651e2a50fafb8d2746d23612fd5a4ff9cd174024` | unchanged |
| `paper/v1/tables/full_budget.tex` | `d9ec249e372a838d96ad03e5d4c660a2e01255952f560a95702c7c975e280d2e` | unchanged |
| `paper/v1/tables/full_c3.tex` | `16d7e366ae9823a4f3de8dca37ddf7ade0aefa0f26624c1b19871103d109c6f3` | unchanged |
| `paper/v1/tables/full_c3layer.tex` | `0d76bcb1c124dec5225f56d0957725a9ece02d19ba0971abe9cac75a33813a30` | unchanged |
| `paper/v1/tables/full_dose.tex` | `99594a090739ee7bce0b84415dde13a3ce1524f7287b55c67e5202422bebda0d` | unchanged |
| `paper/v1/tables/full_draws.tex` | `2a011b32f266e719050edc552ea4cb3903f02d0b174a21e3b986047bb2488006` | unchanged |
| `paper/v1/tables/full_forecasts.tex` | `08544e712abb3d5c289fbccadc2c5e413489651e03adcf5eb33501657f744a8b` | unchanged |
| `paper/v1/tables/full_m12b.tex` | `f81d58a3cae345a068f6cf7a80789e9bded67174d64f65c4a4133c2efc181035` | unchanged |
| `paper/v1/tables/full_selpin.tex` | `12befa96b34cbcdb5faa8283f972e7447999839a060a3fe2bc3f8967d52954f6` | unchanged |
| `paper/v1/tables/g0_gates.tex` | `447f5a64c8a2944557e0f591782d7b17a814531cccbd94d691437c7ae8ba3180` | unchanged |
| `paper/v1/tables/gap_summary.tex` | `3b767cc7aa774a6d7bdedd070eadafa652fec22d1308282ab68c1c82c92b3126` | unchanged |
| `paper/v1/tables/gates_full.tex` | `ecd7d9d75c382bc23dea8e0280a24ee3cc11e4b6f9fb50c943b5efaf3f0a12ca` | unchanged |
| `paper/v1/tables/licences.tex` | `d2aa609c083de4f4a245a919958e0a5b16e1294089ace198f5c6352f20847978` | cleaned copy |
| `paper/v1/tables/stageA.tex` | `29fc05e6fc73cf416f64431dbf0b2ae68de46e4375e9c36c9e895903cd6b7fd0` | unchanged |
| `paper/v1/tables/stageB.tex` | `18b2cb8273c3d3d41a66805e275f17ff2d0e9548194d6e1eb853dcb6e326ef70` | unchanged |
| `paper/v1/tables/stageC.tex` | `9b8ed09bdc8f5ecf3aee8a900be0acd95ede3e6197c7dcbbf6082f3d547f3591` | unchanged |
| `paper/v1/tables/theory_status.tex` | `80e985dcc177a72c5613bd8ba85de13a5abef7ca0db9d7eb8e5526484ff03cc9` | unchanged |
| `phase5/adapt_extras.py` | `69bd5bb265be1f791761c210d6a1f6b82fdaa8804fe3e36d524c9be0f0e264a6` | cleaned copy |
| `phase5/eval_windows.py` | `0f909d1e23c2e1bd3ec3765778ddf927ad2e3ccc036a6c1eadce8e822f2ecb41` | cleaned copy |
| `phase5/grad_snr.py` | `93d09556f5ce3b9b36dc3fd06fe813d6c883d91c96f9539d0c1b5963ac2c9cb5` | cleaned copy |
| `phase6/adapt_extras.py` | `2d4eeb705556d2ea966ab1dee1a2880a52135e4cf6c9498fcd937c5f9b4e94d1` | cleaned copy |
| `phase6/amendment_A1_post_freeze.md` | `096aa8ac1295374e51a1ae909a054822db92bc42320dc4fab8db7a5d4bba4a57` | not included |
| `phase6/amendment_A2_post_freeze.md` | `108a7add570faf4bfcab2a22229aa08844af62d2d969d0f2fbefa74ca58ea7fb` | not included |
| `phase6/amendment_A3_post_freeze.md` | `48b8d72a52acaba9d4884a2660d1b2e8e020a395e0daffb3647caf4b68043a12` | not included |
| `phase6/amendment_A4_post_freeze.md` | `3d7aae95950a3657442016fde45b99f48c161a4e9f20f9b396924df6836cf036` | not included |
| `phase6/analyze_confirm.py` | `0d82526e1e751230e9144ac28ebb4c07b4157ba91192c7fd14453a746d613043` | cleaned copy |
| `phase6/analyze_confirm_ext.py` | `92a05a2045839c2a12432edfffcd49186b3984dafdb15d0d6235267b0a7ba26f` | cleaned copy |
| `phase6/assumption_audit.py` | `39adede34e6922f0d5f2afeff2dc4437d9a92602d26b4f764d8126e1c97de6d8` | cleaned copy |
| `phase6/budget_v13.py` | `64d7b082d02bb60f3444c381e90fd87dac1fb7f1edfc0940545385fa782e59e5` | not included |
| `phase6/c1de_milp.py` | `68e2284160da721a27694d7e14ba82f39d4137a874e5b55116a07fc8d3f8f140` | cleaned copy |
| `phase6/chain6b.ps1` | `eee8ecffe1f81aa6e0e3394dcc86622a991dcd27ec02c39d7abc168071126f1e` | cleaned copy |
| `phase6/chain6b_util.py` | `e6f965838baf88ff6eb6bbdd5e835c22620609b982ca417f989c28331a019849` | cleaned copy |
| `phase6/ckpt_pins.json` | `9106f4a26e7cc1295d133ce1b35932588d596e7ac1bedc57d1113c6a513f31cf` | cleaned copy |
| `phase6/confirm_plan.py` | `757a94865ee1b02edfacdf18c209cace3ed0faf2b70c892cdbc1192d4b6d24a6` | cleaned copy |
| (phase6/decision record, part 2, name withheld) | `3d9d34b5a361a0fb931a3e7673e757766e242ae3482e420faab821d028aa1840` | not included |
| (phase6/decision record, part 3, name withheld) | `779f5143cead4ed7fa7c442081f475f06c44fe01a089ba00767e82995f262829` | not included |
| `phase6/dose_draws.py` | `bbb3c5c03dfb9cdf80f7e08740eabc7554a1fbbcbdc4525f101b34f20562aaa2` | cleaned copy |
| `phase6/forecast_io.py` | `c5a461aa038e72a827d215e6f91c854b5f2d30641e9816fa49c4bfb3d60c2983` | unchanged |
| `phase6/forecast_probe.py` | `3e1cf5b78c382f9239b7e0e75dae3ceda4b81b415f7631cc679aca958d47a38e` | unchanged |
| `phase6/forecasts.py` | `035672940515d8590163166210e3e7c51d098e479ad898c097d9bed9ceac738f` | cleaned copy |
| `phase6/g0_b3.ps1` | `ccf11b15429d8ccfde9845256278ebc1d9695b0fb980165ad4efd9d17a5bcabe` | cleaned copy |
| `phase6/g0_gate_config_b3.json` | `dcd797659c49fc00bfeee0af9db2a0a5aa616ae5cee631a8b6b71d65e2884dfb` | cleaned copy |
| `phase6/g0_moe_b3.json` | `8624872754afc4d50eece3e501f097e73d41e75fa0a0ee45d8906447f3bc7943` | cleaned copy |
| `phase6/g0_probe_v2_orig.py` | `db99a6db2371ca40123655ed4eea797cbea012e4dabb705928bd81fd522a2f95` | cleaned copy |
| `phase6/g0_rerun_compare.py` | `24c5c08fd42e9b7b0f3e1579b973fdb6aa28e3c77fbc1b92af4963044f80a231` | cleaned copy |
| `phase6/g1_gate.py` | `63b4ec7a9e1742846847d1bcccd5a2283aa9f7eadcf782a1394a8056e25986e7` | cleaned copy |
| `phase6/g1g2_config.json` | `298b74d9da035f12f275ae008044c4d8a11a4de55760c82da5758dd826280520` | cleaned copy |
| `phase6/g1g2_config_MOCK.json` | `a53b8511cea95c97ae300367f960202ff8d3f71d6268a4f690eb93cba384c750` | not included |
| `phase6/g1g2_plan.py` | `de361c6406f2d711214ce4cfe60ba60a2be30806fab48e44a480cd36562c1339` | cleaned copy |
| `phase6/g1g2_spec.json` | `cf6ba4ea342b7aae5468736cedf531b0cb8d6aea4dadb7166f604e2485ce29dd` | not included |
| `phase6/g2_gate.py` | `16a8709ea6bae9dda11ac1646d492db739465ed8cce45eb79d921a8d0b58c825` | unchanged |
| `phase6/gate_record_G0_registered.json` | `9259ddd619332de2a6ce72e6ba18458a36dcfa86cd81ff9a6965a4a46d978518` | not included |
| `phase6/gate_rules_v13.py` | `654c5ff8883644d0d5d4e0f90b46c8ae5bbd4d9d086473225e00be0a5fa4e23c` | cleaned copy |
| `phase6/hooks_after_B.ps1` | `8ec1a2a860ecdc5232111cc9002ddfee96ad50935065407e3db58740c1d69540` | cleaned copy |
| `phase6/mock_chain6b.py` | `798d359b9940b6311ecfaaff52dcf1e5200035db54c497f1933a74a0906adaf2` | cleaned copy |
| `phase6/mock_confirm.py` | `b16a1f7f09fe2cdce3cfcd42070a783b59cfb1ae7c15d448682040e41df7da51` | cleaned copy |
| `phase6/mock_g1g2.py` | `ad2eede870e5b5c43fe7cf8058fff121fe7a785e85e8906283434105f0955b33` | unchanged |
| `phase6/nestor_books_feasibility.py` | `8739b64cdfc744a17920e077988c8c867a8551685f949e06941cc3f7b482fefa` | cleaned copy |
| `phase6/phase6_report_v13.md` | `a739d4237480c3745c712ae7aa5e7c9f4b66c7fbb1655e58966df7773e382b66` | not included |
| `phase6/pins_check.py` | `f849b2a5f6f3b0606800986727b4afa26fd3b102d2ba89d14c6fbdc6e49683d3` | cleaned copy |
| `phase6/r8_eval_equivalence.py` | `c23a3eccd3790a35233da4328809dd21328c2b3b01c1524f0d685e317d50a806` | cleaned copy |
| `phase6/run_continual.py` | `a13ff3dd14c66387d1b9773672e9c49c1f4898d3c394c468d89843ff8bb66508` | cleaned copy |
| `phase6/run_entanglement.py` | `8813f0ad251a6ff3dff971b42e7e5f2bd53f8b7f100fe07f9ae34c58adcca4d7` | cleaned copy |
| `phase6/stageB_windows.py` | `06ac8da92342b95ea8526464a6c422df00eb15fab78a427b61aa3220f259eb21` | unchanged |
| `phase6/test_analyze_confirm.py` | `4198ed2c33d62a4faa6788052579da201bdbb1d21b327942d6c620fa42755d86` | cleaned copy |
| `phase6/test_audit_realized.py` | `739f60d5aa289cd9a095289afe351e5795e00c84d93bf86408edddd197e30ab2` | unchanged |
| `phase6/test_c1de_milp.py` | `98d6aa58a11d60cb18c1c137a37afd46d1137a1b9e5b1b5a805c87806e2ad1d1` | cleaned copy |
| `phase6/test_c3_scope.py` | `83b80a16812bb7125563d82b08bff27e933f2847ed55958757466f5a2adcab3a` | cleaned copy |
| `phase6/test_forecast_probe.py` | `b97f0e46642353be144eed25653838b5e7bca0fa32912f6c9b1ca5a3cc3b3681` | unchanged |
| `phase6/test_gates.py` | `f464e55604c6e02b62e3f50b70fdc237ffef0e387eeb0f5c1033b7b8c50cda78` | unchanged |
| `phase6/test_p16_analysis.py` | `1e2ca86263523f06c5ce5a96c680925fafa475b35a172c7577242b52b5123368` | cleaned copy |
| `phase6/test_p16_stop.py` | `df0cfaa78d0ff62452cf0cfa78dc30e2a75115f052c4ef67c446333f8b8d0187` | cleaned copy |
| `phase6/test_phase6.py` | `9b0ac169b901650d4f5b1430c94a1f3cc59a48c94484110400be9db594e1c072` | cleaned copy |
| `phase6/test_r8_eval_equivalence.py` | `d81edc31daed14c330ed916c9bd443087ccb2960b90cfce4d832e39d1fc01391` | unchanged |
| `phase6/test_wd_only_a2.py` | `ef5b5ff92adbe161f540897f30546c9cd703188cc9f8eff8336790d692e1964a` | unchanged |
| `phase6/test_wd_only_r9.py` | `9f56ed3dd8d682c7d72b20776352bde30a8d3efc24140790b319ee888019e2dd` | cleaned copy |
| `phase6/wd_only_check.py` | `1537b77dcea12b2b7c1d82df37e04c084882c8162c25528ff924be7f11185171` | cleaned copy |
| `phase6/weight_controls.py` | `49e9f45b8f252ae887b49f9e309d804df8cc0e439dd859fb7f0c4536272caf6a` | cleaned copy |
| `pins_check.py` | `f849b2a5f6f3b0606800986727b4afa26fd3b102d2ba89d14c6fbdc6e49683d3` | cleaned copy |
| `power_sim_v1.py` | `d9c21f3b033e023b3e3f39fca21fee9094348b03a751a048ac1932b14817bf4e` | cleaned copy |
| `prep_all_streams.py` | `3bd0796ca1a7f5dc11fae3a46e40bed6cd5a87d7ff44cc3d2153b313aeb0aa11` | cleaned copy |
| `prep_streams.py` | `c13b7eb7a03923a166b89602bcf40fc93978f851824ecc14dcee1c0887e3f9c9` | cleaned copy |
| `prereg_v13/change_log_v1.3.md` | `38153538a1464a4898d8ddbc08f7b3e9eed5020e7cd130725441ac05211a1274` | not included |
| `prereg_v13/deviations_post_freeze.md` | `5ff4f610378202f476e2e23d6a3685ca476f47e41ed17fc4e0aa77a8aaeaf78f` | not included |
| `prereg_v13/power_sim_v11.py` | `a27b5b644a356ee778655c6b5ba616d9a95b9961dc953f66f8536491870cd16a` | not included |
| `prereg_v13/power_v13.py` | `8864f9cbcaec073aa90e9cba503baf96c8aab372182669fad53834cf0b0ce6a6` | not included |
| `prereg_v13/preregistration_v1.3_final.md` | `b01bf1dc824567926e4007d35006f32fe76939e14565ad8ca7b0a6f2dddebe3d` | not included |
| `prereg_v13/preregistration_v1.3_final.md.sha256` | `70e60ce373bc26f6e2fa8f3996e0f9418d094380d651a4690e83a116da1e952a` | not included |
| `probe_alloc.py` | `ee08f48a4e0afa4c595300729ed3b857245da8fff6a94e438e8352a760b8276e` | unchanged |
| `probe_mem.py` | `725a96e2506fb5539d8c13f0763e44376ca0613670b8aaf857aaf4bdab9c743e` | unchanged |
| `pycode_tok.py` | `5bf456df8f7c01a5bf09dc05227126abb149897cf1e6d344d9d9963c6c7de8a1` | unchanged |
| `reassemble_ckpt.py` | `5a1b0e0dfab8a7f2e0e3f5653a410828527db83807d4041dad39c2d2bbebe614` | unchanged |
| `rebuild_env.ps1` | `851b4af2ac089ecaaf3321f503e68acebffbc2cc74dae432625dc0b6d9e473ee` | cleaned copy |
| `results/a1_wd_only_checks.json` | `3bf7649b5e17f4f3fbb1337959f686ec9bbbf740f424d7da31af1ba3504c87f5` | unchanged |
| `results/a2_acceptance.json` | `6d8fc6c4ebbbba5ca0d5b4c192f838bad4402f27ef41d546658e932dc94e59ad` | cleaned copy |
| `results/a4_acceptance.json` | `4a36d1cbe69b92d0b92841cf164a8a3c18699cf8109f5aa04ef5e3e93ef87d72` | unchanged |
| `results/analysis/analysis_confirm.json` | `d9b3e47598b2ac10745455917ef53251a114a70e5918aed584a53f58f577ae01` | cleaned copy |
| `results/analysis/analysis_confirm.json.sha256` | `47539aef05e9777e0fb842ecdc809dedbc30b78c1f437f921e53d3db8d8a4cec` | not included |
| `results/analysis/analysis_confirm.md` | `58cde788e25025f9818bef0bdc48b6d686c59e5b6459f8032561013cf205adac` | cleaned copy |
| `results/analysis_core/analysis_confirm.json` | `859849244c70fa5ac4ddd8a73ab574ec97cdec33c3de1d5da00da7c6811f09a1` | cleaned copy |
| `results/analysis_core/analysis_confirm.json.sha256` | `3123752fad6a7de086c9cc034e70a6c3f8d88b8356f48f05459649fced4edb4a` | not included |
| `results/analysis_core/analysis_confirm.md` | `c68decaf745f0b6f0a4290ece90740427977f016c98f24a1025b5a128b0e0dae` | cleaned copy |
| `results/c1de/c1de_moe_b1_german.json` | `b34bcc2a7694c5e01f610d79c1a4e655baee8adc44b5fe51e1e5b19df751f6da` | cleaned copy |
| `results/c1de/c1de_moe_b1_german.json.sha256` | `a9dd98510b35d420f4c05491279e52cccd57c9251fe848184f3cc14e35b4cc63` | not included |
| `results/c1de/c1de_moe_b2_german.json` | `57fb979a5556ec8c8cb1052192236e8c64b4f8c6ec8ba9520c981df9f671fe80` | cleaned copy |
| `results/c1de/c1de_moe_b2_german.json.sha256` | `d937b6a271edc507aacc46da3f34c7bcd06cbafe9a14451f48853d76235d8a21` | not included |
| `results/c1de/c1de_record.json` | `1f80cf5a9fde7ddef8948aa1a57b90c81137cec2c3815514fd71197cd2bff387` | not included |
| `results/confirm/CORE_DONE.json` | `0893ab74346ed309138673c2c59972ddd0a0c092e852baf6b057d6cc5c8b7b04` | unchanged |
| `results/dose/dose_draws.json` | `512c641681e9f40fac9393b46a288b4bf4035b37fe3df575c05f9224186310a6` | cleaned copy |
| `results/dose/dose_draws.json.sha256` | `da0addbaa46038f35d6530b6d4a30cc06743066c3f5b9b15849d4e402176ef80` | not included |
| `results/dose/dose_record.json` | `9758574d3881671b462efab7405726c8eae4b4017ae12c9ed74b0dfa50a72609` | not included |
| `results/entanglement/predictions_E1_E7.md` | `7f6f00984ce048e4f9a4329286d7421731fc00926695e5b417a869b750e41370` | not included |
| `results/exploratory/dmuon_step2.json` | `a7506db93c00fa2d016382a42a529b8ab51886a65a1d8da16b61b760aef450fa` | unchanged |
| `results/exploratory/dmuon_step2.py` | `a5ba451ac8a7aab82780f8c12ecc5df84c0880c14c1b4e30f30fdc2ad374d946` | cleaned copy |
| `results/exploratory/queue_c2step1.json` | `321e3fa04ce607f98e239baeaba6710e9d95edb6db82142e5bfaaaae04e19f59` | unchanged |
| `results/exploratory/queue_c2step2.json` | `fbbbba91ecdb481e46a3486a6cd925e6837177facaf33f2ae331c3bf90f375e6` | unchanged |
| `results/exploratory/queue_randx.json` | `7dc8ccc71c58218744d8acc7ffa8a004a90f8f25cd2417b2ee0e8b0c8c55e0a5` | unchanged |
| `results/exploratory/queue_state_c2step1.json` | `412d57fdb661c2c4c67ddb69c2eb59cd6192647f9f3f9d9f00eb7ca4582cfb38` | cleaned copy |
| `results/exploratory/queue_state_c2step2.json` | `ccf467c880e47ee74e8127343ddd08194d9b3b5136451fb5c56027529745ad10` | cleaned copy |
| `results/exploratory/queue_state_randx.json` | `efd3c67a79b82db42f9bb4d052e55160e6d06ddf28cdbe059c912ad7b96d7451` | unchanged |
| `results/exploratory/rand_extension.json` | `6ca2e5aa640cb2277c71d9b01d0d18e7f8e53c0cd3e0988ef1c311837ebc373e` | unchanged |
| `results/exploratory/rand_extension.py` | `c0120f9c63449ea56bcd3d9ac78b223566152f90a91a6ce47e1130c6dd5c67cf` | cleaned copy |
| `results/exploratory/run_both.ps1` | `9f319238378a36299948514f2006205818bbb9648a24c0d65ac687d24a2ed516` | cleaned copy |
| `results/exploratory/run_c2step1.ps1` | `8fdadb68268d43d1c08826f03d9a1520b925f73af26abe886802fc1bcbd722da` | cleaned copy |
| `results/exploratory/run_c2step1_nowait.ps1` | `e08dc5d4cc473c004d25066653249bcfca5ee10be397a1fb7478cabcfdba8849` | cleaned copy |
| `results/exploratory/run_c2step2.ps1` | `1cc92a8b21450c9c7d1431a43508212634b031616168174edf2814abfec6e812` | cleaned copy |
| `results/exploratory/run_randx.ps1` | `5d13d8b6f89c8cda9f0d3c876d7e5618e9e8dc7dd655906596b3c6d83d3ac5dc` | cleaned copy |
| `results/forecast/forecast_units.json.sha256` | `802bda02ae110b10c5eb6550e16a25b7b1c303b707d7c6dfdcdd1b9719432aab` | not included |
| `results/gap_trajectory/gap_analysis.py` | `6a57cb328a6023eb72fade57c6529592f8d7ca3802af06a8fe99411b9c560b28` | cleaned copy |
| `results/nestor_books_draws.json.sha256` | `96169c6e406cd48776d405f57c02636b7def3728ea85c6e35b09bfb709c07b0e` | not included |
| `results/nestor_books_feasibility.json` | `d1db32aed649d5f78eee40de44ffe8a6659890790aa6becb1977f428ab9bf248` | not included |
| `results/stage_c_preview/nestor_books_draws_cpu.json.sha256` | `709f909ee379aa2d90757b2e0bd87b2265a8ddc47065357a16b0c5250aa3270a` | not included |
| `results/stage_c_preview/nestor_books_feasibility_cpu.json` | `3f1f7296adba0ff126918ab17bf01cc00833390a8c03446b591dc810badebd7a` | not included |
| `routing_stats.py` | `3b46d844d3bc9af14dc17649eb5ab450e577e13f94daa83e885166bf3c0ab21f` | cleaned copy |
| `run_continual.py` | `a13ff3dd14c66387d1b9773672e9c49c1f4898d3c394c468d89843ff8bb66508` | cleaned copy |
| `run_entanglement.py` | `8813f0ad251a6ff3dff971b42e7e5f2bd53f8b7f100fe07f9ae34c58adcca4d7` | cleaned copy |
| `run_exp.py` | `6394d7426d7e62fb231b3d19360561fee1005d06fa404d7ce970dea365c85a6a` | cleaned copy |
| `run_pretrain_queue.ps1` | `11202566608e8a53398ce31f74289364d16191ba7aa14da655e317ea9838da25` | unchanged |
| `run_queue.ps1` | `2b42f3b9f847dd14d53f225441769c6dab872ec1f8b4a337c6c6005a91510980` | cleaned copy |
| `run_queue.py` | `f535c41a19de90802c9d104865da5e0794b363dcc24f4f883b05ee358f172890` | unchanged |
| `signflip_lookup_v1.csv` | `0c2e36cc1a546d611697c9861270895b6e70363c9d735983f4d6331e0b2bab50` | unchanged |
| `spotcheck.py` | `76ff47e9edf51b8f5c2912da82c2415ce70b305b3385955fcef67e930abe7c78` | unchanged |
| `stageB_windows.py` | `06ac8da92342b95ea8526464a6c422df00eb15fab78a427b61aa3220f259eb21` | unchanged |
| `subset_adapt.py` | `4b86df24add074b3fc3d04c318a837974403cb90e5d6b90cdbec97c2d9fe4e27` | unchanged |
| `test_analyze_confirm.py` | `4198ed2c33d62a4faa6788052579da201bdbb1d21b327942d6c620fa42755d86` | cleaned copy |
| `test_audit_realized.py` | `739f60d5aa289cd9a095289afe351e5795e00c84d93bf86408edddd197e30ab2` | unchanged |
| `test_c1de_milp.py` | `98d6aa58a11d60cb18c1c137a37afd46d1137a1b9e5b1b5a805c87806e2ad1d1` | cleaned copy |
| `test_c3_scope.py` | `83b80a16812bb7125563d82b08bff27e933f2847ed55958757466f5a2adcab3a` | cleaned copy |
| `test_compile.py` | `d2a9c82ccc1c89ffe3b95be526781d8eab6f2fe986742f70fa5270de646c8fcb` | unchanged |
| `test_forecast_probe.py` | `b97f0e46642353be144eed25653838b5e7bca0fa32912f6c9b1ca5a3cc3b3681` | unchanged |
| `test_gates.py` | `f464e55604c6e02b62e3f50b70fdc237ffef0e387eeb0f5c1033b7b8c50cda78` | unchanged |
| `test_memguard.py` | `2e74a66aa8c2f3ea9165a00ec751a668df13d5b24002e633aa31b8fa87391d11` | cleaned copy |
| `test_phase3.py` | `cb8dc65f3d2dd233990324b4f1a231c4c84053991dfc0e3a2cdfb7cc3f5ad578` | cleaned copy |
| `test_phase4.py` | `45d5b1d00b2a9a7e6a6c26a0b11eb569b902d48a950d3bd049efb56378511d53` | cleaned copy |
| `test_phase5_torch.py` | `e02d49374bd34f9f7dcebd87de14702d3f14ad4bd209a6f13f580f059a777600` | unchanged |
| `test_phase6.py` | `9b0ac169b901650d4f5b1430c94a1f3cc59a48c94484110400be9db594e1c072` | cleaned copy (amended after the freeze) |
| `test_sdpa.py` | `88f0a986db299b4f016fdc135a065438a9e1809f3727ab1e26a132a527cc5c8f` | unchanged |
| `test_subset_adapt.py` | `a13b4eacaa2722d4dd6a6ef5c08ca75cb9885e7bc337b35d50fc700a0cef2efa` | cleaned copy |
| `theory/v4/entanglement.py` | `32305dd63c3bded9c1534606b70fe46bba5bf9d784c073d9fb6755e051899d71` | unchanged |
| `theta0_checks.py` | `1a3a240a4d7138a0d8bbd71cb137b9b2354d2e54244923ba14aa9caf708aa2c1` | cleaned copy |
| `tiny_experiments.py` | `94c2b0d0e3016685596fffce1b3c479cfcce3df3e44db1e64f54f6ac513a4b14` | unchanged |
| `tok_domain.py` | `664d6f434f4fb2e61899660f871036fccbcfdbbf540c0ba6b5fd760550cc0c46` | unchanged |
| `tok_files.py` | `f7ca1f6440f893b5b96fac95b52022e23b39969faf15dbb0e5a06f27c83b8c96` | unchanged |
| `weight_controls.py` | `49e9f45b8f252ae887b49f9e309d804df8cc0e439dd859fb7f0c4536272caf6a` | cleaned copy (amended after the freeze) |
