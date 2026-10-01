import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # v4 model development

    Analyses supporting the next deployed docbot models (v4). Everything here reads the CV runs that
    `src/sent_spans/train.py` logs to wandb (`radkoff/tosdr_cases`): one run per case × fold × attempt, with `test/`
    metrics measured on the held-out fold using the checkpoint chosen on the ~10% stopping slice.

    **Questions**

    1. **Which metrics should decisions rest on?** How noisy is each, and does it separate cases at all?
    2. **How many attempts per fold are worth running?** Attempts are trained with different seeds and the best is
       picked on the stopping slice. Does that pick actually beat a random attempt on the test fold?
    3. **How big an improvement is real?** Before comparing base models (or any other change), how large a
       difference in the summary metrics can seed noise alone produce, and how should a comparison be tested?
    4. **Is the stopping-slice snapshot too arbitrary?** The stopping-slice metrics are spiky, and each run's
       `test/` metrics come from the single checkpoint that won on it. Would a smoother signal (e.g. the p90 of
       the second half of training) be a better basis for selection?

    We only have four gold-standard cases (`TEST_CASE_IDS`), and adding more takes manual curation, so the analyses
    below treat **the four cases as fixed** and use the 4 × 5 = 20 case-folds as the unit of replication.

    **Auth:** `wandb.Api()` uses your existing wandb login. Run `uv run wandb login` once (it stores the key in
    `~/.netrc`), or set `WANDB_API_KEY`. Stopping-slice histories are cached under `data/training/wandb_cache/`,
    since finished runs never change.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy import stats

    import wandb

    sns.set_theme(style="whitegrid")
    return Path, mo, np, pd, plt, stats, wandb


@app.cell
def _(Path):
    WANDB_PROJECT = "radkoff/tosdr_cases"
    try:
        _here = Path(__file__).parent
    except NameError:  # e.g. running cells interactively without a file
        _here = Path.cwd()
    HISTORY_CACHE_DIR = (_here / "../data/training/wandb_cache").resolve()

    CASE_FOLD = ["case_id", "fold"]
    # Headline metrics, most decision-relevant first. The ap metrics only exist for runs after the switch to AP
    # model selection; empty columns are dropped when the runs are loaded
    METRICS = [
        "test/doc_ap",
        "test/doc_roc_auc",
        "test/doc_f1",
        "test/sent_ap",
        "test/sent_roc_auc",
        "test/sent_f1_pos",
    ]
    # Categorical slots 1-3 of the reference palette, paired with distinct markers so attempts aren't color-only
    ATTEMPT_STYLE = {0: ("#2a78d6", "o"), 1: ("#eb6834", "s"), 2: ("#1baf7a", "^")}
    CASE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    return ATTEMPT_STYLE, CASE_COLORS, CASE_FOLD, HISTORY_CACHE_DIR, METRICS, WANDB_PROJECT


@app.cell
def _(WANDB_PROJECT, pd, wandb):
    def load_runs(project=WANDB_PROJECT) -> pd.DataFrame:
        """One row per finished run: identifying config plus the test/, stop/ and attempt_ summary metrics"""
        rows = []
        for run in wandb.Api().runs(project, filters={"state": "finished"}):
            row = dict(
                run_id=run.id,
                run_name=run.name,
                created_at=run.created_at,
                job_type=run.job_type,
                case_id=run.config.get("case_id"),
                case_title=run.config.get("case_title"),
                fold=run.config.get("fold"),
                attempt=run.config.get("attempt"),
                model_base=run.config.get("model_base"),
                learning_rate=run.config.get("learning_rate"),
                dataset_version=run.config.get("dataset_version"),
            )
            summary = dict(run.summary)
            row.update({k: v for k, v in summary.items() if k.startswith(("test/", "stop/", "attempt_"))})
            rows.append(row)
        runs = pd.DataFrame(rows)
        # Attempt selection in train._run_attempts: doc AP on the stopping slice, or doc F1 for runs before the switch
        runs["select_metric"] = "stop/doc_f1"
        runs["select_score"] = runs.get("attempt_stop_doc_f1")
        if "attempt_stop_doc_ap" in runs:
            has_ap = runs.attempt_stop_doc_ap.notna()
            runs.loc[has_ap, "select_metric"] = "stop/doc_ap"
            runs.loc[has_ap, "select_score"] = runs.loc[has_ap, "attempt_stop_doc_ap"]
        return runs

    all_runs = load_runs()
    return (all_runs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The runs

    The analyses use the **baseline** run set: the first CV run on the four test cases (`bert-base-uncased`, 3
    attempts per fold, attempts selected on `stop/doc_f1`). Edit the filter below to point them at later runs;
    e.g. rerunning Q2 on AP-selected runs will show whether switching the selection metric helped.
    """)
    return


@app.cell
def _(CASE_FOLD, all_runs, np):
    cv_runs = all_runs[
        (all_runs.job_type == "cv")
        & (all_runs.model_base == "bert-base-uncased")
        & (all_runs.select_metric == "stop/doc_f1")
    ].copy()
    cv_runs["fold"] = cv_runs.fold.str.removeprefix("fold").astype(int)
    cv_runs = cv_runs.sort_values([*CASE_FOLD, "attempt"]).reset_index(drop=True)
    # Replays train._run_attempts: strict `>` keeps the earliest attempt on ties, which idxmax also does given the
    # attempt ordering above
    cv_runs["selected"] = False
    cv_runs.loc[cv_runs.groupby(CASE_FOLD).select_score.idxmax(), "selected"] = True
    n_attempts = int(cv_runs.groupby(CASE_FOLD).size().max())
    assert (cv_runs.groupby(CASE_FOLD).size() == n_attempts).all(), "Every case-fold needs the same attempt count"
    assert np.isclose(cv_runs.groupby(CASE_FOLD).selected.sum(), 1).all()
    cv_runs.pivot_table(index=["case_id", "case_title"], columns="fold", values="attempt", aggfunc="count").rename(
        columns=lambda f: f"fold {f} attempts"
    ).reset_index()
    return cv_runs, n_attempts


@app.cell
def _(METRICS, cv_runs, mo):
    available_metrics = [m for m in METRICS if m in cv_runs and cv_runs[m].notna().all()]
    metric_picker = mo.ui.dropdown(
        options=available_metrics,
        value="test/doc_ap" if "test/doc_ap" in available_metrics else "test/doc_roc_auc",
        label="Metric for the plots and per-metric tables",
    )
    metric_picker
    return available_metrics, metric_picker


@app.cell
def _(ATTEMPT_STYLE, cv_runs, metric_picker, plt):
    def plot_overview(runs, metric):
        cases = sorted(runs.case_id.unique())
        fig, axes = plt.subplots(1, len(cases), figsize=(3.2 * len(cases), 3.4), sharey=True)
        for ax, case_id in zip(axes, cases):
            case_runs = runs[runs.case_id == case_id]
            for attempt, (color, marker) in ATTEMPT_STYLE.items():
                att = case_runs[case_runs.attempt == attempt]
                ax.scatter(att.fold, att[metric], color=color, marker=marker, s=40, label=f"attempt {attempt}")
            sel = case_runs[case_runs.selected]
            ax.scatter(sel.fold, sel[metric], s=150, facecolors="none", edgecolors="#222222", lw=1.2, label="selected")
            ax.set_title(f"case {case_id}", fontsize=10)
            ax.set_xlabel("fold")
            ax.set_xticks(sorted(runs.fold.unique()))
        axes[0].set_ylabel(metric)
        axes[-1].legend(loc="lower right", fontsize=8)
        fig.suptitle(f"{metric} by case, fold and attempt (ring = attempt picked on the stopping slice)", fontsize=11)
        fig.tight_layout()
        return fig

    plot_overview(cv_runs, metric_picker.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q1: Which metrics should decisions rest on?

    A useful decision metric (a) varies little between seeds, so a comparison isn't swamped by noise, and (b) still
    varies between cases, i.e. it isn't pinned at its ceiling. Below, each metric's spread is split into nested
    components by the method of moments:

    - **seed**: the SD between attempts of the same case-fold (same data, different seed)
    - **fold**: the SD between fold means within a case, after removing the seed noise those means still carry
    - **case**: the SD between case means
    """)
    return


@app.cell
def _(CASE_FOLD, available_metrics, cv_runs, n_attempts, np, pd):
    def variance_components(runs, metric) -> dict:
        seed_var = runs.groupby(CASE_FOLD)[metric].var().mean()
        fold_means = runs.groupby(CASE_FOLD)[metric].mean()
        fold_var = max(fold_means.groupby("case_id").var().mean() - seed_var / n_attempts, 0.0)
        case_means = runs.groupby("case_id")[metric].mean()
        return dict(
            mean=runs[metric].mean(),
            seed_sd=np.sqrt(seed_var),
            fold_sd=np.sqrt(fold_var),
            case_sd=case_means.std(),
            min=runs[metric].min(),
        )

    pd.DataFrame({m: variance_components(cv_runs, m) for m in available_metrics}).T.round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Q1 takeaways (baseline runs)

    - **`test/doc_roc_auc` is the best of what the baseline runs logged.** Its seed SD (0.017) is smaller than its fold
      (0.022) and case (0.049) SDs, so it has room to show differences.
    - **`test/doc_f1` is noisier (seed SD 0.024) for the same information.** Its threshold is fit on the fold being
      scored, so it's also optimistic. Keep it as a readable secondary number.
    - **`test/sent_roc_auc` is saturated.** It sits at 0.97–0.98 everywhere, and seed noise (0.010) is larger than
      the fold (0.003) and case (0.007) spread. It can't discriminate between models.
    - **`test/sent_f1_pos` is fragile.** One run (case 377, fold 1, attempt 0) scored exactly 0: its positive-class
      probabilities never crossed the fixed 0.5 cutoff, even though its doc ROC AUC was a merely-poor 0.80. That
      single run drives the 0.117 seed SD. Use this metric as a collapse detector, not a comparison metric.
    - **Going forward, lead with `test/doc_ap`.** It's threshold-free like ROC AUC, but responds to the rare
      positives instead of the many easy negatives. `train.py` now logs `test/doc_ap` and `test/sent_ap`, and selects
      checkpoints and attempts on `stop/doc_ap`. Rerun this cell on those runs to confirm AP's noise is comparable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q2: How many attempts per fold are worth running?

    Each extra attempt costs a full training run. It pays off only if picking the best attempt on the stopping
    slice finds an attempt that's better on the test fold. Per case-fold, we compare the **selected** attempt's
    test metric to:

    - the **mean over attempts**, i.e. what picking one at random gets in expectation. `selection_gain` is
      selected minus mean.
    - the **best attempt in hindsight**. `oracle_gain` is the ceiling selection could reach, and roughly the
      headroom ensembling attempts might capture.

    `within_fold_r` is the correlation between the selection score and the test metric after demeaning both within
    each case-fold. It asks whether a higher stopping-slice score means a better attempt *on the same data*.
    `hit_rate` is how often the selected attempt was also the best on test, against `chance` = 1 / attempts.
    """)
    return


@app.cell
def _(CASE_FOLD, available_metrics, cv_runs, n_attempts, np, pd, stats):
    def demean(runs, col):
        return runs[col] - runs.groupby(CASE_FOLD)[col].transform("mean")

    def selection_stats(runs, metric, score_col="select_score") -> dict:
        by_cf = runs.groupby(CASE_FOLD)[metric]
        selected = runs.loc[runs.groupby(CASE_FOLD)[score_col].idxmax()].set_index(CASE_FOLD)[metric]
        gain = selected - by_cf.mean()
        r, p = stats.pearsonr(demean(runs, score_col), demean(runs, metric))
        return dict(
            selection_gain=gain.mean(),
            gain_se=gain.std() / np.sqrt(len(gain)),
            oracle_gain=(by_cf.max() - by_cf.mean()).mean(),
            within_fold_r=r,
            r_pvalue=p,
            hit_rate=(selected == by_cf.max()).mean(),
            chance=1 / n_attempts,
        )

    pd.DataFrame({m: selection_stats(cv_runs, m) for m in available_metrics}).T.round(4)
    return demean, selection_stats


@app.cell
def _(CASE_COLORS, cv_runs, demean, metric_picker, plt):
    def plot_selection_signal(runs, metric):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for color, (case_id, case_runs) in zip(CASE_COLORS, runs.groupby("case_id")):
            ax.scatter(
                demean(runs, "select_score")[case_runs.index],
                demean(runs, metric)[case_runs.index],
                color=color,
                s=36,
                label=f"case {case_id}",
            )
        ax.axhline(0, color="#999999", lw=0.8)
        ax.axvline(0, color="#999999", lw=0.8)
        ax.set_xlabel(f"{runs.select_metric.iloc[0]} (selection score), demeaned within case-fold")
        ax.set_ylabel(f"{metric}, demeaned within case-fold")
        ax.set_title("Does a better stopping-slice score mean a better attempt on test?", fontsize=11)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    plot_selection_signal(cv_runs, metric_picker.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Q2 takeaways (baseline runs)

    - **Best-of-3 selection on `stop/doc_f1` bought nothing.** On `test/doc_roc_auc` the selection gain was
      −0.0004 ± 0.0026. The within-fold correlation between selection score and test metric was −0.07 (p = 0.58).
      The selected attempt was the test-best in 20% of case-folds, against 33% by chance. The picture is the same for
      `test/doc_f1` and the sentence metrics.
    - **The headroom is real but unreachable this way.** The best attempt in hindsight is +0.012 doc ROC AUC above
      the mean. The stopping slice is too small to tell apart attempts that differ by that much (see Q4).
    - **The one thing selection did was dodge a collapsed run.** 377/fold1/a0 had the lowest selection score in its
      fold and wasn't picked. That's 1 run in 60, and a cheap guard catches it without always paying 3×: e.g. retry
      an attempt whose best `stop/doc_ap` falls far below the case's other folds.

    **Per run mode:**

    - **R&D CV:** use 1 attempt per fold. When comparing configurations, spend the budget on averaging attempts
      rather than selecting among them (Q3 shows how much that tightens a comparison).
    - **Full CV (thresholds):** use 1 attempt. Keep it the same as the final run, because `thresholds.json` assumes
      the final model's scores come from the same procedure as the CV models'.
    - **Final:** use 1 attempt, unless AP selection turns out to work (rerun Q2 on AP-selected runs) or ensembling
      is tested. The baseline runs can't test ensembling: only the selected attempt's predictions are kept.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q3: How big an improvement is real?

    To compare two configurations (e.g. two base models), run both on the **same cases, folds and stopping slices**
    (fold assignment and stopping slices are already deterministic). Take the per-case-fold difference in a test
    metric; the test-set sampling noise shared by both runs cancels in that difference. With four fixed cases,
    `paired_comparison` below treats the 20 case-fold differences as the sample:

    - **p-value:** a sign-flip permutation test. If the configurations are equivalent, each difference is equally
      likely to have either sign.
    - **CI:** a bootstrap that resamples folds within each case, so every resample keeps all four cases.
    - **Consistency:** the per-case mean differences. With four cases, even a perfect 4/4 sign test only reaches
      p = 0.125, so generalization beyond these cases can't be established statistically. Treat "improves on all
      four cases" as a separate requirement.

    **Calibrating it with A/A tests.** The baseline has 3 seeds of the *same* configuration, so pitting attempt *i*
    against attempt *j* is a comparison where the true difference is zero. Any "effect" it finds is noise.
    """)
    return


@app.cell
def _(CASE_FOLD, np, pd):
    def paired_comparison(runs_a, runs_b, metric, n_resamples=20_000, seed=0) -> dict:
        """
        Compares two configurations run on the same case-folds. Each argument has one row per case-fold, or several
        (attempts), which are averaged per case-fold first. Positive deltas favor `runs_a`.
        """
        a = runs_a.groupby(CASE_FOLD)[metric].mean()
        b = runs_b.groupby(CASE_FOLD)[metric].mean()
        delta = (a - b).dropna()
        rng = np.random.default_rng(seed)
        values = delta.to_numpy()

        signs = rng.choice([-1, 1], size=(n_resamples, len(values)))
        null_means = (signs * values).mean(axis=1)
        p_value = (np.abs(null_means) >= abs(values.mean()) - 1e-12).mean()

        # Stratified bootstrap: resample folds within each case, keeping every case in every resample
        groups = [g.to_numpy() for _, g in delta.groupby("case_id")]
        boot_means = np.array(
            [np.concatenate([rng.choice(g, size=len(g)) for g in groups]).mean() for _ in range(n_resamples // 4)]
        )
        per_case = delta.groupby("case_id").mean()
        return dict(
            mean_delta=values.mean(),
            ci_low=np.quantile(boot_means, 0.025),
            ci_high=np.quantile(boot_means, 0.975),
            p_value=p_value,
            cases_improved=f"{(per_case > 0).sum()}/{len(per_case)}",
            per_case=per_case.round(4).to_dict(),
            delta_sd=values.std(ddof=1),
            n=len(values),
        )

    return (paired_comparison,)


@app.cell
def _(available_metrics, cv_runs, n_attempts, paired_comparison, pd):
    def aa_tests(runs, metric) -> pd.DataFrame:
        rows = {}
        for i in range(n_attempts):
            for j in range(i + 1, n_attempts):
                result = paired_comparison(runs[runs.attempt == i], runs[runs.attempt == j], metric)
                rows[f"a{i} vs a{j}"] = {k: v for k, v in result.items() if k != "per_case"}
        return pd.DataFrame(rows).T

    pd.concat(
        {m: aa_tests(cv_runs, m) for m in available_metrics if not m.startswith("test/sent_f1")},
        names=["metric", "pair"],
    ).reset_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Minimum detectable effect.** The A/A deltas measure seed noise only. If each configuration averages *k*
    attempts per case-fold, the SD of one case-fold difference is about $\sqrt{2/k}\,\sigma_\text{seed}$. The
    difference between two means over *n* = 20 case-folds is then detectable with 80% power at α = 0.05 once it
    exceeds $2.8 \sqrt{2/k}\,\sigma_\text{seed} / \sqrt{n}$. A real comparison also has **model × fold
    interaction** (two base models disagree about which folds are easy), which adds variance the A/A test can't
    see. So treat these as **lower bounds**. `paired_comparison` uses the observed deltas, so its p-values include
    that extra variance automatically.
    """)
    return


@app.cell
def _(CASE_FOLD, available_metrics, cv_runs, np, pd):
    def min_detectable_effects(runs, metrics, ks=(1, 2, 3)) -> pd.DataFrame:
        n = runs.groupby(CASE_FOLD).ngroups
        rows = {}
        for metric in metrics:
            seed_sd = np.sqrt(runs.groupby(CASE_FOLD)[metric].var().mean())
            rows[metric] = {f"k={k}": 2.8 * np.sqrt(2 / k) * seed_sd / np.sqrt(n) for k in ks}
        return pd.DataFrame(rows).T.round(4)

    min_detectable_effects(cv_runs, available_metrics)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Q3 takeaways (baseline runs)

    - **The A/A tests behave.** None of the three same-configuration comparisons comes out significant. Their mean
      deltas are within ±0.003 doc ROC AUC, and their per-case-fold delta SDs are 0.013–0.029. The larger SDs
      belong to the pairs that include the collapsed run, so a single bad seed can dominate a small comparison.
    - **Noise floor for `test/doc_roc_auc`:** with 1 attempt per fold per configuration, a mean improvement under
      about **0.015** is indistinguishable from seed noise. With 3 attempts averaged, it's about **0.009**. Treat
      these as the smallest effects worth acting on, not guarantees; model × fold interaction adds to them.
    - **For comparing base models:**
        1. Run each candidate on the four cases with identical folds. Use 1 attempt per fold for a first look, and
           add attempts (averaged, not selected) only if the result lands near the noise floor.
        2. Sweep the learning rate per base model. At a single LR, the comparison partly measures which model suits
           that LR.
        3. Call a winner only if `paired_comparison` gives p < 0.05, the CI excludes 0, the mean delta clears the
           floor above, **and** the improvement holds on all four cases.
        4. Every extra configuration tried on the same four cases raises the chance of a lucky winner. Keep the
           number of candidates small, and confirm the winner in the full-case CV run before shipping it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q4: Is the stopping-slice snapshot too arbitrary?

    Within an attempt, the checkpoint is the one with the highest stopping-slice doc metric (every third eval), so
    the `test/` metrics are a snapshot at that single, noisy maximum. Two separate concerns:

    - **For evaluation, the snapshot is the right number.** It measures the model the procedure actually produces.
      Its noise is part of the seed variance measured in Q1 and averages out across folds. A trajectory statistic
      measured on the *test* fold would describe a model we could never pick without peeking at test data.
    - **For selection, the question is whether a smoother signal ranks attempts better than the max.** The runs'
      stopping-slice histories let us test that. Each candidate signal below is computed per attempt and scored with
      the same within-case-fold statistics as Q2.
    """)
    return


@app.cell
def _(HISTORY_CACHE_DIR, WANDB_PROJECT, cv_runs, pd, wandb):
    STOP_HISTORY_KEYS = ["stop/doc_f1", "stop/doc_roc_auc", "stop/doc_ap"]

    def load_stop_history(run_id) -> pd.DataFrame:
        """Every logged stopping-slice doc eval for a run, cached locally since finished runs never change"""
        cache_path = HISTORY_CACHE_DIR / f"{run_id}_stop.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        run = wandb.Api().run(f"{WANDB_PROJECT}/{run_id}")
        # scan_history only yields rows containing every requested key, so request the keys this run has
        # (define_metric(summary="max") stores e.g. stop/doc_f1 under the summary key stop/doc_f1.max)
        summary_keys = {key.split(".")[0] for key in run.summary.keys()}
        keys = [k for k in STOP_HISTORY_KEYS if k in summary_keys] + ["train/global_step"]
        history = pd.DataFrame(list(run.scan_history(keys=keys)))
        HISTORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        history.to_parquet(cache_path)
        return history

    stop_histories = {run_id: load_stop_history(run_id) for run_id in cv_runs.run_id}
    return (stop_histories,)


@app.cell
def _(available_metrics, cv_runs, np, pd, selection_stats, stop_histories):
    def history_signals(history) -> dict:
        """Candidate attempt-selection signals from one run's stopping-slice history"""
        signals = {}
        for key in ["stop/doc_f1", "stop/doc_roc_auc", "stop/doc_ap"]:
            if key not in history:
                continue
            name = key.removeprefix("stop/doc_")
            values = history.sort_values("train/global_step")[key].to_numpy()
            second_half = values[len(values) // 2 :]
            signals[f"{name}: max"] = values.max()
            signals[f"{name}: rolling-3 max"] = pd.Series(values).rolling(3, min_periods=1).mean().max()
            signals[f"{name}: p90 of 2nd half"] = np.quantile(second_half, 0.9)
            signals[f"{name}: mean abs jump"] = np.abs(np.diff(values)).mean()
        return signals

    signals = pd.DataFrame({run_id: history_signals(h) for run_id, h in stop_histories.items()}).T
    runs_with_signals = cv_runs.join(signals, on="run_id")
    signal_cols = [c for c in signals.columns if not c.endswith("jump")]

    def signal_table(metric):
        table = pd.DataFrame({col: selection_stats(runs_with_signals, metric, score_col=col) for col in signal_cols})
        table = table.T[["selection_gain", "gain_se", "within_fold_r", "r_pvalue", "hit_rate"]]
        # Rank correlation too: a single collapsed run can dominate Pearson's r with only 3 attempts per case-fold
        table["within_fold_spearman"] = [
            runs_with_signals.assign(
                s=runs_with_signals[c] - runs_with_signals.groupby(["case_id", "fold"])[c].transform("mean"),
                t=runs_with_signals[metric] - runs_with_signals.groupby(["case_id", "fold"])[metric].transform("mean"),
            )[["s", "t"]]
            .corr(method="spearman")
            .iloc[0, 1]
            for c in signal_cols
        ]
        return table.round(4)

    jumps = signals[[c for c in signals.columns if c.endswith("jump")]].mean().round(4)
    f1_optimism = (runs_with_signals["f1: max"] - runs_with_signals["test/doc_f1"]).mean()
    auc_optimism = (runs_with_signals["roc_auc: max"] - runs_with_signals["test/doc_roc_auc"]).mean()
    signal_tables = {m: signal_table(m) for m in available_metrics if m.startswith("test/doc")}
    return auc_optimism, f1_optimism, jumps, signal_tables


@app.cell
def _(auc_optimism, f1_optimism, jumps, mo, signal_tables):
    mo.vstack(
        [
            mo.md(
                f"Mean absolute change between consecutive stopping-slice doc evals: "
                f"{', '.join(f'`{k}` {v:.3f}' for k, v in jumps.items())}. "
                f"Optimism of the max over the test value (winner's curse): doc F1 {f1_optimism:+.3f}, "
                f"doc ROC AUC {auc_optimism:+.3f}."
            ),
            mo.ui.tabs({metric: table for metric, table in signal_tables.items()}),
        ]
    )
    return


@app.cell
def _(cv_runs, mo):
    case_fold_picker = mo.ui.dropdown(
        options={f"case {c} / fold {f}": (c, f) for c, f in cv_runs[["case_id", "fold"]].drop_duplicates().values},
        value="case 377 / fold 1",
        label="Case-fold to plot",
    )
    case_fold_picker
    return (case_fold_picker,)


@app.cell
def _(ATTEMPT_STYLE, case_fold_picker, cv_runs, plt, stop_histories):
    def plot_stop_curves(runs, case_id, fold):
        cf_runs = runs[(runs.case_id == case_id) & (runs.fold == fold)]
        panels = [("stop/doc_f1", "test/doc_f1"), ("stop/doc_roc_auc", "test/doc_roc_auc")]
        fig, axes = plt.subplots(1, len(panels), figsize=(11, 3.6))
        for ax, (stop_key, test_key) in zip(axes, panels):
            for _, run in cf_runs.iterrows():
                color, marker = ATTEMPT_STYLE[run.attempt]
                history = stop_histories[run.run_id].sort_values("train/global_step")
                label = f"attempt {run.attempt}" + (" (selected)" if run.selected else "")
                ax.plot(
                    history["train/global_step"],
                    history[stop_key],
                    color=color,
                    lw=2 if run.selected else 1.2,
                    label=label,
                )
                # The saved checkpoint is the first max of stop/doc_f1 (the baseline's checkpoint selection);
                # mark where it sits and the test value it went on to score
                best = history.loc[history["stop/doc_f1"].idxmax()]
                ax.scatter(best["train/global_step"], best[stop_key], color=color, marker=marker, s=50, zorder=3)
                ax.scatter(
                    best["train/global_step"],
                    run[test_key],
                    color=color,
                    marker="*",
                    s=140,
                    edgecolors="#222222",
                    zorder=3,
                )
            ax.set_title(f"{stop_key} (line) vs {test_key} at the saved checkpoint (star)", fontsize=10)
            ax.set_xlabel("training step")
        axes[0].legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        return fig

    plot_stop_curves(cv_runs, *case_fold_picker.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Q4 takeaways (baseline runs)

    - **The stopping-slice curves are spiky relative to the effects we care about.** Consecutive doc evals move by
      about 0.027 on average (F1 and ROC AUC alike), which exceeds the seed SD of `test/doc_roc_auc`. Taking the max
      of such a curve is optimistic: the winning `stop/doc_f1` averages 0.059 above the same checkpoint's
      `test/doc_f1`.
    - **No stopping-slice signal ranks attempts.** Max, rolling-3 max and second-half p90, on F1 or ROC AUC, all
      have within-fold Spearman correlations of |ρ| ≤ 0.11 with the test metric, and none has a selection gain
      distinguishable from zero. Your p90-of-the-second-half idea came out best (+0.002 ± 0.003 doc ROC AUC), but
      that is not a real effect at this sample size.
    - **So the snapshot is noisy, but not biased in a way that hurts comparisons.** Checkpoint-selection noise
      lands in the `test/` seed variance, and Q3 accounts for it. What it does rule out is using stopping-slice
      metrics (`stop/*`, `attempt_stop_*`) as evidence for decisions like choosing a base model. Use the `test/`
      metrics, paired across folds.
    - **The root cause is size, not the statistic.** A 10% slice of one case's training pool has too few positive
      docs to separate near-equal models. Switching selection to `stop/doc_ap` makes it less jumpy, but it may not
      help much. Rerun this section on AP-selected runs before trusting attempt selection again.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary and next steps

    1. **Metrics:** use `test/doc_ap` as the headline (`test/doc_roc_auc` for the baseline runs), with `test/doc_f1`
       as a readable secondary. `test/sent_roc_auc` is saturated. Use `test/sent_f1_pos` only as a collapse detector.
    2. **Attempts:** use 1 per fold in every mode. The stopping slice can't pick the better seed. If more compute is
       available, average attempts when comparing configurations, and test ensembling before using it for final
       models.
    3. **Significance:** compare configurations with `paired_comparison` on identical case-folds. With 1 attempt per
       fold, a doc ROC AUC improvement under about 0.015 is noise. Also require the improvement on all four cases.
    4. **Selection:** `train.py` now selects checkpoints and attempts on `stop/doc_ap` and early-stops on sentence
       AP. The next CV run should check whether that changes Q2 or Q4, but the stopping slice's size limits what
       any selection statistic can do.
    """)
    return


if __name__ == "__main__":
    app.run()
