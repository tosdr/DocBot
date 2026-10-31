import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TF-IDF prefilter: the high-recall cliff

    The TF-IDF prefilter (`src/tfidf.py`) decides which sentences BERT scores at all, so the part of its PR curve that
    matters is the high-recall edge: how much irrelevant text has to get through to keep (almost) every positive.

    `tfidf.py` cross-validates each case (5 folds within the train split, grouped by service) and scores every
    sentence with the fold model that didn't train on it. Each sentence also gets a **pass rate needed**: the fraction
    of its fold's held-out negatives scoring at least as high, i.e. the share of irrelevant sentences that must get
    through for this one to. It's computed from negatives only, so for a positive it's an honest held-out estimate.
    Sorting a case's positives by it gives the curves below: recall against pass rate, one step per positive.

    What ships is a pass rate, not a score: a rule picks one on each case's curve, then a model trained on the whole
    train split gets the score threshold that lets through that share of unseen negatives (see the `tfidf.py`
    docstring). The curves describe the fold models; the shipped model trained on 25% more data, and should do a
    little better. Q6 checks the shipped models on the test split.

    **Questions**

    1. **Where is the cliff in each case?**
    2. **Which positives fall off it?** Badly chosen annotations, or real positives that just use different words?
    3. **Does the setup matter?**
    4. **What would a threshold rule pick?** A pass-rate ceiling, vs. an exchange rate between recall and compute.
    5. **Does the cliff's location track how common the case is?**
    6. **How do the shipped models do on the test split?**

    **Data:** every case in `data/models/tfidf_v3`, written by `uv run python -m src.tfidf`. The other setups' curves
    (gray, and Q3) need `--compare_setups`, which writes `data/tfidf/tfidf_v3/setups`.
    """)
    return


@app.cell
def _():
    import pickle
    import sys
    import textwrap
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    from scipy import stats

    # Make the repo root importable regardless of where marimo was launched from
    try:
        _root = str(Path(__file__).parent.parent.resolve())
    except NameError:  # e.g. running cells interactively without a file
        _root = str(Path.cwd().parent.resolve())
    if _root not in sys.path:
        sys.path.append(_root)

    from src import tfidf, utils

    sns.set_theme(style="whitegrid")
    return Line2D, PercentFormatter, mo, np, pd, pickle, plt, sns, stats, textwrap, tfidf, utils


@app.cell
def _(tfidf):
    MODELS_DIR = tfidf.model_output_dir(0).parent
    CASES_PATH = tfidf.here / f"../data/db_dumps/{tfidf.DB_DUMP_VERSION}/cases_clean.pkl"
    POINTS_PATH = tfidf.here / f"../data/db_dumps/{tfidf.DB_DUMP_VERSION}/points_clean.pkl"
    EDIT_URL = "https://edit.tosdr.org"
    # Recall level Q3 and Q5 compare setups and cases at, independent of the threshold rules
    RECALL_TARGET = 0.95
    # Categorical slots of the reference palette (same order as the v4 notebooks), with markers so the rules aren't
    # color-only. Setups other than the shipped one are drawn in MUTED
    SELECTED_COLOR = "#2a78d6"
    RULE_STYLE = {"ceiling": ("#eb6834", "s"), "exchange": ("#1baf7a", "^")}
    RULE_NAMES = {"ceiling": "A: pass-rate ceiling", "exchange": "B: exchange rate"}
    # Ordinal blue ramp (steps 250/450/650) for ordered recall levels
    RECALL_LEVELS = {1.0: "#86b6ef", 0.95: "#2a78d6", 0.9: "#104281"}
    INK = "#222222"
    MUTED = "#8a8a8a"
    return (
        CASES_PATH,
        EDIT_URL,
        INK,
        MODELS_DIR,
        MUTED,
        POINTS_PATH,
        RECALL_LEVELS,
        RECALL_TARGET,
        RULE_NAMES,
        RULE_STYLE,
        SELECTED_COLOR,
    )


@app.cell
def _(CASES_PATH, MODELS_DIR, POINTS_PATH, pd, pickle, tfidf, utils):
    def load_cv() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        One row per case x setup x positive, per-case sentence counts, and each case's final_metrics. The shipped
        setup's rows come from cv.pkl, the others' from the --compare_setups output where it exists. Negatives' scores
        aren't kept: what they tell us is already summarized in each positive's pass_rate_needed
        """
        columns = ["row", "setup", "point_id", "fold", "pass_rate_needed"]
        positives, counts, shipped = [], {}, {}
        for path in sorted(MODELS_DIR.glob("*/cv.pkl")):
            case_id = int(path.parent.name)
            shipped[case_id] = pickle.load(open(path.parent / "final_metrics.pkl", "rb"))
            cv = pd.read_pickle(path)
            pos = cv[cv.label == 1].rename_axis("row").reset_index().assign(setup=shipped[case_id]["setup"])
            positives.append(pos[columns].assign(case_id=case_id))
            if (setups_path := tfidf.setups_output_path(case_id)).exists():
                others = pd.read_pickle(setups_path)
                positives.append(others.loc[others.setup != shipped[case_id]["setup"], columns].assign(case_id=case_id))
            counts[case_id] = {"n_pos": len(pos), "n_neg": int((cv.label == 0).sum())}
        positives = pd.concat(positives, ignore_index=True)
        positives["point_id"] = positives.point_id.astype(int)
        # Cleaned the same way as in tfidf.prep_datasets
        quotes = pd.read_pickle(POINTS_PATH).quote_text.reindex(positives.point_id.unique())
        positives["text"] = positives.point_id.map(quotes.map(utils.preprocess_doc_text))
        counts = pd.DataFrame.from_dict(counts, orient="index")
        return positives, counts, pd.DataFrame.from_dict(shipped, orient="index")

    positives, case_info, shipped = load_cv()
    SETUP = shipped.setup.mode()[0]
    SETUPS = [SETUP] + [s for s in positives.setup.unique() if s != SETUP]
    # Relative only: positives come from every train-split doc, negatives from 600 sampled docs
    case_info["prevalence"] = case_info.n_pos / (case_info.n_pos + case_info.n_neg)
    case_info["title"] = pd.read_pickle(CASES_PATH).title.reindex(case_info.index)
    case_info = case_info.sort_values("n_pos")
    return SETUP, SETUPS, case_info, positives, shipped


@app.cell(hide_code=True)
def _(SETUP, SETUPS, case_info, mo, positives, shipped):
    _compared = positives.groupby("case_id").setup.nunique().eq(len(SETUPS)).sum()
    _versions = ", ".join(f"{s} at μ = {mu:g}" for s, mu in shipped[["setup", "mu"]].drop_duplicates().values)
    mo.vstack(
        [
            mo.md(
                f"Loaded **{len(case_info)} cases**, shipped with {_versions}. "
                + (
                    f"{len(SETUPS) - 1} other setups compared on {_compared} cases: {', '.join(SETUPS[1:])}."
                    if len(SETUPS) > 1
                    else f"No other setups to compare against {SETUP}."
                )
            ),
            mo.ui.table(case_info.round(4).rename_axis("case").reset_index(), selection=None, page_size=20),
        ]
    )
    return


@app.cell
def _(mo, tfidf):
    ceiling_slider = mo.ui.slider(
        start=0.01, stop=0.5, step=0.01, value=0.25, label="A: pass-rate ceiling", show_value=True
    )
    mu_slider = mo.ui.slider(start=0.5, stop=16, step=0.5, value=tfidf.MU, label="B: exchange rate μ", show_value=True)
    return ceiling_slider, mu_slider


@app.cell(hide_code=True)
def _(SETUP, ceiling_slider, mo, mu_slider):
    mo.vstack(
        [
            mo.md(rf"""
            ## Controls

            These apply to every chart and table below, which all show the shipped setup, {SETUP} (others in gray).
            The rule that shipped is B, at the μ the slider starts at.

            - **A: pass-rate ceiling**: keep every positive if that lets through at most this share of negatives,
              otherwise let through exactly this share. Recall 100% where it's cheap, and a cap where it isn't.
            - **B: exchange rate μ**: the pass rate p minimizing p + μ × (share of positives dropped), i.e. one point of
              recall is worth μ points of pass rate (`tfidf.exchange_rule`). BERT compute is linear in the pass rate,
              so this prices it linearly. A price per *halving* of the pass rate was tried first, but it kept chasing
              tiny pass rates on gradual curves: on case 134 (LR cw=10) it dropped 10 of 43 positives to get from 8%
              to 0.2%.
            """),
            mo.hstack([ceiling_slider, mu_slider], justify="start", gap=2),
        ]
    )
    return


@app.cell
def _(np, pd, tfidf):
    def ceiling_rule(pass_needed: np.ndarray, ceiling: float) -> float:
        """Rule A: the pass rate that keeps every positive, capped at the ceiling"""
        return min(pass_needed.max(), ceiling)

    def ceiling_recall(pass_needed: np.ndarray, pass_rate: float) -> float:
        """
        Rule A's expected recall for new positives. When the ceiling binds, the pass rate doesn't depend on this case's
        positives, so the share of them kept is already a fair estimate. Otherwise it's the hardest positive's pass rate
        needed, which needs the same correction as rule B's (tfidf.expected_recall)
        """
        if pass_rate < pass_needed.max():
            return (pass_needed <= pass_rate).mean()
        return tfidf.expected_recall(pass_needed, pass_rate)

    def pass_rate_for_recall(pass_needed: np.ndarray, recall: float) -> float:
        """The lowest pass rate that keeps at least `recall` of the positives"""
        return np.sort(pass_needed)[int(np.ceil(recall * len(pass_needed))) - 1]

    def apply_rules(positives: pd.DataFrame, setup: str, ceiling: float, mu: float) -> pd.DataFrame:
        """
        Per case, for each rule: its pass rate, how many positives it drops and what share it keeps in the CV, and its
        expected recall for new positives (see Q4)
        """
        out = {}
        for case_id, grp in positives[positives.setup == setup].groupby("case_id"):
            q = grp.pass_rate_needed.values
            out[case_id] = {}
            ceiling_pass_rate, exchange_pass_rate = ceiling_rule(q, ceiling), tfidf.exchange_rule(q, mu)
            for name, p, recall in [
                ("ceiling", ceiling_pass_rate, ceiling_recall(q, ceiling_pass_rate)),
                ("exchange", exchange_pass_rate, tfidf.expected_recall(q, exchange_pass_rate)),
            ]:
                out[case_id] |= {
                    f"{name}_pass_rate": p,
                    f"{name}_dropped": (q > p).sum(),
                    f"{name}_kept": (q <= p).mean(),
                    f"{name}_recall": recall,
                }
        return pd.DataFrame.from_dict(out, orient="index")

    return apply_rules, pass_rate_for_recall


@app.cell
def _(SETUP, apply_rules, ceiling_slider, mu_slider, positives):
    rules = apply_rules(positives, SETUP, ceiling_slider.value, mu_slider.value)
    return (rules,)


@app.cell
def _(
    INK,
    Line2D,
    MUTED,
    PercentFormatter,
    RECALL_LEVELS,
    RULE_NAMES,
    RULE_STYLE,
    SELECTED_COLOR,
    np,
    pass_rate_for_recall,
    plt,
    textwrap,
):
    # Positives that beat every held-out negative need a 0% pass rate, which a log axis can't show
    PASS_RATE_FLOOR = 1e-3

    def log_pass_rate_axis(axis):
        axis.set_xscale("log")
        axis.set_xlim(PASS_RATE_FLOOR, 1)
        axis.set_xticks([1e-3, 1e-2, 1e-1, 1], ["0.1%", "1%", "10%", "100%"])

    def recall_curve(pass_needed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """x, y for a step plot of recall against pass rate, extended to a 100% pass rate"""
        q = np.clip(np.sort(pass_needed), PASS_RATE_FLOOR, 1)
        return np.append(q, 1.0), np.append(np.arange(1, len(q) + 1) / len(q), 1.0)

    def plot_overview(positives, setup):
        """Distribution over cases of the pass rate needed to reach each recall level"""
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        pass_needed = positives[positives.setup == setup].groupby("case_id").pass_rate_needed
        for level, color in RECALL_LEVELS.items():
            needed = np.sort(pass_needed.apply(lambda q: pass_rate_for_recall(q.values, level)).clip(PASS_RATE_FLOOR))
            median = np.median(needed)
            ax.step(
                needed,
                np.arange(1, len(needed) + 1) / len(needed),
                where="post",
                color=color,
                lw=2,
                label=f"{level:.0%} recall (median case {median:.1%})",
            )
            ax.plot(median, 0.5, "o", color=color, ms=6, mec="white", mew=1.5, zorder=5)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        log_pass_rate_axis(ax)
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel("pass rate needed (share of held-out negatives let through)")
        ax.set_ylabel("share of cases")
        ax.set_title(f"{setup} · {pass_needed.ngroups} cases", fontsize=10)
        fig.tight_layout()
        return fig

    def plot_cliff(ax, case_pos, setup, case_rules, label_hardest=0, ylim=(0.6, 1.03)):
        for other, grp in case_pos.groupby("setup"):
            if other != setup:
                ax.step(*recall_curve(grp.pass_rate_needed.values), where="post", color=MUTED, lw=1, alpha=0.45)
        selected = case_pos[case_pos.setup == setup].pass_rate_needed.values
        ax.step(*recall_curve(selected), where="post", color=SELECTED_COLOR, lw=2)
        for name, (color, marker) in RULE_STYLE.items():
            ax.plot(
                max(case_rules[f"{name}_pass_rate"], PASS_RATE_FLOOR),
                case_rules[f"{name}_kept"],
                marker=marker,
                color=color,
                ms=9,
                mec="white",
                mew=2,
                ls="none",
                zorder=5,
                clip_on=False,
            )
        # Number the hardest positives at the step where each falls off, matching the rank column of the Q2 table
        n = len(selected)
        for rank, q in enumerate(np.sort(selected)[::-1][:label_hardest], start=1):
            x, y = max(q, PASS_RATE_FLOOR), (n - rank + 1) / n
            if y >= ylim[0]:
                ax.annotate(str(rank), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8, color=INK)
        log_pass_rate_axis(ax)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    def legend_handles(setup, other_setups=True):
        handles = [Line2D([], [], color=SELECTED_COLOR, lw=2, label=setup)]
        if other_setups:
            handles.append(Line2D([], [], color=MUTED, lw=1, alpha=0.45, label="other setups"))
        for name, (color, marker) in RULE_STYLE.items():
            handles.append(Line2D([], [], color=color, marker=marker, ms=9, ls="none", label=RULE_NAMES[name]))
        return handles

    def plot_all_cliffs(positives, case_info, rules, setup, ncols=5):
        nrows = int(np.ceil(len(case_info) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows + 0.8), sharex=True, sharey=True, squeeze=False
        )
        for ax, (case_id, info) in zip(axes.flat, case_info.iterrows()):
            plot_cliff(ax, positives[positives.case_id == case_id], setup, rules.loc[case_id])
            title = textwrap.shorten(info.title, 44, placeholder="…")
            ax.set_title(f"{case_id} · {info.n_pos} positives\n{title}", fontsize=8.5, color=INK)
        for ax in axes.flat[len(case_info) :]:
            ax.set_visible(False)
        _handles = legend_handles(setup, positives.setup.nunique() > 1)
        fig.legend(handles=_handles, loc="upper center", ncol=4, frameon=False, fontsize=9)
        fig.supxlabel("pass rate (share of held-out negatives let through)", fontsize=10)
        fig.supylabel("recall (share of positives kept)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
        return fig

    return legend_handles, plot_all_cliffs, plot_cliff, plot_overview


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q1. Where is the cliff in each case?

    First, across all cases: the pass rate each case needs to keep 100%, 95% and 90% of its positives. The gap
    between the 100% and 95% lines is what the hardest few positives cost.
    """)
    return


@app.cell
def _(SETUP, plot_overview, positives):
    plot_overview(positives, SETUP)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then case by case. Read each panel right to left. Recall stays at 100% until the pass rate falls below what the
    hardest positive needs, then steps down once per positive. Gray lines are the other setups, and the markers are
    where the two rules land on the curve at the current settings. Cases are ordered by number of positives. The y-axis
    starts at 60%, so a curve leaving the bottom of a panel has dropped more than 40% of its positives. The default
    selection is 15 cases spread evenly over the number of positives.
    """)
    return


@app.cell
def _(case_info, mo, np):
    _options = {f"{case_id} · {info.title}": case_id for case_id, info in case_info.iterrows()}
    _spread = set(case_info.index[np.linspace(0, len(case_info) - 1, min(15, len(case_info))).round().astype(int)])
    cliff_case_picker = mo.ui.multiselect(
        options=_options, value=[label for label, case_id in _options.items() if case_id in _spread], label="Cases"
    )
    return (cliff_case_picker,)


@app.cell
def _(SETUP, case_info, cliff_case_picker, mo, plot_all_cliffs, positives, rules):
    _selected = case_info[case_info.index.isin(cliff_case_picker.value)]
    mo.vstack(
        [
            cliff_case_picker,
            plot_all_cliffs(positives, _selected, rules, SETUP)
            if len(_selected)
            else mo.md("_Pick at least one case._"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q2. Which positives fall off the cliff?

    The hardest positives for one case, numbered as on its curve. "Median over setups" shows whether a positive is hard
    for every model or just this one (with `--compare_setups` output). "Dropped by" names the rules that would filter it
    out at the current settings.
    """)
    return


@app.cell
def _(case_info, mo):
    _options = {f"{case_id} · {info.title}": case_id for case_id, info in case_info.iterrows()}
    case_picker = mo.ui.dropdown(
        options=_options,
        value=next((label for label, case_id in _options.items() if case_id == 297), next(iter(_options))),
        label="Case",
    )
    return (case_picker,)


@app.cell
def _(EDIT_URL, pd):
    def dropped_by(pass_needed: pd.Series, rules_row: pd.Series | pd.DataFrame) -> pd.Series:
        ceiling = pass_needed > rules_row["ceiling_pass_rate"]
        exchange = pass_needed > rules_row["exchange_pass_rate"]
        return pd.Series(
            [", ".join(n for n, d in [("A", c), ("B", e)] if d) for c, e in zip(ceiling, exchange)],
            index=pass_needed.index,
        )

    def point_table(df: pd.DataFrame) -> pd.DataFrame:
        """Display columns shared by the Q2 tables"""
        return pd.DataFrame(
            {
                "pass rate needed (%)": (100 * df.pass_rate_needed).round(1),
                "text": df.text.str.strip().str.slice(0, 300),
                "point": EDIT_URL + "/points/" + df.point_id.astype(int).astype(str),
                "fold": df.fold,
            },
            index=df.index,
        )

    def hardest_positives(positives, case_id, setup, rules_row, n=15) -> pd.DataFrame:
        case_pos = positives[positives.case_id == case_id]
        median = case_pos.groupby("row").pass_rate_needed.median()
        selected = case_pos[case_pos.setup == setup].set_index("row")
        hardest = selected.sort_values("pass_rate_needed", ascending=False).head(n)
        table = point_table(hardest)
        table.insert(0, "rank", range(1, len(hardest) + 1))
        table.insert(2, "median over setups (%)", (100 * median.reindex(hardest.index)).round(1))
        table.insert(3, "dropped by", dropped_by(hardest.pass_rate_needed, rules_row))
        return table.reset_index(drop=True)

    def audit_list(positives, case_info, rules, setup) -> pd.DataFrame:
        """Every positive that either rule drops, across cases"""
        selected = positives[positives.setup == setup].merge(rules, left_on="case_id", right_index=True)
        flags = dropped_by(selected.pass_rate_needed, selected)
        dropped = selected[flags != ""].assign(dropped_by=flags[flags != ""])
        dropped = dropped.sort_values(["case_id", "pass_rate_needed"], ascending=[True, False])
        table = point_table(dropped)
        table.insert(0, "case", dropped.case_id)
        table.insert(1, "case title", dropped.case_id.map(case_info.title))
        table.insert(3, "dropped by", dropped.dropped_by)
        return table.reset_index(drop=True)

    return audit_list, hardest_positives


@app.cell
def _(SETUP, case_picker, hardest_positives, legend_handles, mo, plot_cliff, plt, positives, rules):
    def plot_case(case_id, setup):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        plot_cliff(
            ax, positives[positives.case_id == case_id], setup, rules.loc[case_id], label_hardest=5, ylim=(0, 1.02)
        )
        ax.set_xlabel("pass rate (share of held-out negatives let through)")
        ax.set_ylabel("recall")
        _handles = legend_handles(setup, positives[positives.case_id == case_id].setup.nunique() > 1)
        ax.legend(handles=_handles, loc="lower right", fontsize=8, frameon=False)
        fig.tight_layout()
        return fig

    mo.vstack(
        [
            case_picker,
            plot_case(case_picker.value, SETUP),
            mo.ui.table(
                hardest_positives(positives, case_picker.value, SETUP, rules.loc[case_picker.value]),
                selection=None,
                page_size=15,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The audit list: every positive either rule would drop

    Across all loaded cases, at the current rule settings. If most of these are badly chosen annotations,
    dropping them costs little. BERT likely can't learn them either. If they're real positives, the rule is too
    aggressive for that case.
    """)
    return


@app.cell
def _(SETUP, audit_list, case_info, mo, positives, rules):
    _audit = audit_list(positives, case_info, rules, SETUP)
    mo.vstack(
        [
            mo.md(
                f"**{len(_audit)} positives** across {_audit.case.nunique()} of {len(case_info)} cases "
                f"(of {case_info.n_pos.sum()} positives in total)."
            ),
            mo.ui.table(_audit, selection=None, page_size=20),
        ]
    )
    return


@app.cell(hide_code=True)
def _(RECALL_TARGET, mo):
    mo.md(rf"""
    ## Q3. Does the setup matter?

    For each case and setup, the pass rate needed to keep {RECALL_TARGET:.0%} of positives, compared with the case's
    best setup. BERT compute is linear in the pass rate, so the gap is in percentage points, not ratios: 1% vs. 2% is
    double, but only one more point of compute. A fixed recall keeps the comparison independent of the threshold rules.
    Each dot is a case, and setups are sorted by their mean gap. Only cases with `--compare_setups` output are
    included.
    """)
    return


@app.cell
def _(INK, RECALL_TARGET, SELECTED_COLOR, SETUPS, mo, np, pass_rate_for_recall, pd, plt, positives):
    def setup_comparison(positives, recall) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Per case x setup: the pass rate needed for `recall`, and its excess over the case's best setup"""
        needed = (
            positives.groupby(["case_id", "setup"])
            .pass_rate_needed.apply(lambda q: pass_rate_for_recall(q.values, recall))
            .unstack()
        )
        return needed, needed.sub(needed.min(axis=1), axis=0)

    def setup_summary(needed, excess) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    "mean pass rate needed (%)": 100 * needed.mean(),
                    "median pass rate needed (%)": 100 * needed.median(),
                    "mean points above best": 100 * excess.mean(),
                    "within 1 point of best (% of cases)": 100 * (excess <= 0.01).mean(),
                    "best setup (cases)": (excess == 0).sum(),
                }
            )
            .sort_values("mean points above best")
            .round(1)
        )

    def plot_setup_comparison(excess, order, cap=20):
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1.4))
        rng = np.random.default_rng(0)
        for i, setup in enumerate(order):
            gap = 100 * excess[setup].values
            jitter = rng.uniform(-0.25, 0.25, len(gap))
            ax.scatter(np.minimum(gap, cap), i + jitter, s=10, color=SELECTED_COLOR, alpha=0.35, linewidth=0)
            ax.plot(gap.mean(), i, marker="|", ms=18, mew=2.5, color=INK)
        ax.set_yticks(range(len(order)), order)
        ax.invert_yaxis()
        ax.set_xlim(-0.5, cap + 0.5)
        ax.set_xlabel(f"points of pass rate above the case's best setup (dots capped at {cap})")
        ax.set_title(f"Pass rate needed for {RECALL_TARGET:.0%} recall · black bar: mean", fontsize=10)
        fig.tight_layout()
        return fig

    _compared = positives.groupby("case_id").setup.transform("nunique") == len(SETUPS)
    if len(SETUPS) > 1 and _compared.any():
        _needed, _excess = setup_comparison(positives[_compared], RECALL_TARGET)
        _summary = setup_summary(_needed, _excess)
        _out = mo.vstack(
            [
                plot_setup_comparison(_excess, _summary.index),
                mo.ui.table(_summary.rename_axis("setup").reset_index(), selection=None),
            ]
        )
    else:
        _out = mo.md("_No other setups to compare: run `uv run python -m src.tfidf --compare_setups`._")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q4. What would each threshold rule pick?

    What shipped, then both rules per case at the current settings, then the trade-off each rule family makes as its
    parameter varies. The mean pass rate stands in for BERT compute, since every case runs on every document. The
    mean recall weights cases equally.

    **Recall here is an estimate for new positives**, not the share of these positives kept. A rule that picks a
    case's pass rate by looking at its positives lands right on one of them, keeping it by construction. B always
    does, so its recall is 1 − (d + 1) / (n + 1) with d of n positives dropped (`tfidf.expected_recall`). A only does
    when its ceiling doesn't bind, and its pass rate is the hardest positive's. When the ceiling binds, the pass rate
    doesn't depend on the case's positives, and the share kept is already fair. In a nested simulation over all cases
    (each CV fold standing in for unseen documents), A's estimates came within 0.1 point of the recall that single fold
    models realized. B's were half a point optimistic for single fold models, and within 0.1 point for the shipped
    kind of full model.

    **Why B.** Total BERT compute is the sum of the cases' pass rates, so minimizing Σ (pass rate + μ × share dropped)
    over all cases splits into one choice per case: rule B, with the same μ everywhere. Every case then stops where
    its last point of pass rate buys the same recall, while A gives every case the same pass rate whatever it buys. On
    exactly known curves that would beat A. But these curves are estimated from 25 to 528 positives each, and fitting
    each case's own curve also fits its noise: in the nested simulation, the two rules tied on mean recall at the same
    mean pass rate. What B still did better was the worst case. At about an 11% mean pass rate, B's worst case kept
    85% of its positives, and A's 70%.
    """)
    return


@app.cell
def _(case_info, mo, shipped):
    def shipped_table(case_info, shipped):
        table = case_info[["title", "n_pos"]].join(shipped[["pass_rate", "cv_dropped", "expected_recall", "overshoot"]])
        table["in_sample_pass_rate"] = shipped.in_sample_pass_rate
        table["threshold"] = shipped.threshold.round(4)
        for col in ["pass_rate", "expected_recall", "in_sample_pass_rate"]:
            table[col] = (100 * table[col]).round(1)
        table["overshoot"] = table.overshoot.round(3)
        return table.rename(
            columns={
                "pass_rate": "pass rate (%)",
                "cv_dropped": "dropped in CV",
                "expected_recall": "expected recall (%)",
                "in_sample_pass_rate": "full model's in-sample pass rate (%)",
            }
        )

    mo.vstack(
        [
            mo.md(
                f"**Shipped:** mean pass rate {100 * shipped.pass_rate.mean():.1f}%, mean expected recall "
                f"{100 * shipped.expected_recall.mean():.1f}%, {shipped.cv_dropped.sum()} of {shipped.n_pos.sum()} "
                f"positives dropped in the CV. The full model's threshold aims at the pass rate divided by the "
                f"overshoot, which was {shipped.overshoot.min():.2f}–{shipped.overshoot.max():.2f} "
                f"(mean {shipped.overshoot.mean():.2f}): how much more of their held-out negatives the fold models let "
                "through, when calibrated on the negatives they trained on."
            ),
            mo.ui.table(
                shipped_table(case_info, shipped).rename_axis("case").reset_index(), selection=None, page_size=10
            ),
        ]
    )
    return


@app.cell
def _(case_info, mo, rules):
    def rules_table(case_info, rules):
        table = case_info[["title", "n_pos"]].join(rules.drop(columns=["ceiling_kept", "exchange_kept"]))
        for name in ["ceiling", "exchange"]:
            table[f"{name}_pass_rate"] = (100 * table[f"{name}_pass_rate"]).round(1)
            table[f"{name}_recall"] = (100 * table[f"{name}_recall"]).round(1)
        return table.rename(
            columns=lambda c: (
                c.replace("ceiling_", "A ").replace("exchange_", "B ").replace("_", " ")
                + (" (%)" if c.endswith(("pass_rate", "recall")) else "")
            )
        )

    _summary = " · ".join(
        f"**{label}**: mean pass rate {100 * rules[f'{name}_pass_rate'].mean():.1f}%, mean expected recall "
        f"{100 * rules[f'{name}_recall'].mean():.1f}%, {rules[f'{name}_dropped'].sum()} of {case_info.n_pos.sum()} "
        "positives dropped in the CV"
        for name, label in [("ceiling", "A"), ("exchange", "B")]
    )
    mo.vstack(
        [
            mo.md(_summary),
            mo.ui.table(rules_table(case_info, rules).rename_axis("case").reset_index(), selection=None, page_size=20),
        ]
    )
    return


@app.cell
def _(SETUP, apply_rules, np, positives):
    def rule_frontier(positives, setup):
        """Mean pass rate and mean expected recall across cases, for each rule over a range of its parameter"""
        # The same grids as the sliders, so the current settings are always on the curve
        ceilings = np.round(np.arange(0.01, 0.51, 0.01), 2)
        mus = np.round(np.arange(0.5, 16.01, 0.5), 2)
        # apply_rules computes both rules, so each sweep holds the other rule's parameter fixed and ignores it
        return {
            "ceiling": [
                (c, *apply_rules(positives, setup, c, 2.0)[["ceiling_pass_rate", "ceiling_recall"]].mean())
                for c in ceilings
            ],
            "exchange": [
                (mu, *apply_rules(positives, setup, 0.25, mu)[["exchange_pass_rate", "exchange_recall"]].mean())
                for mu in mus
            ],
        }

    # Its own cell, so moving a slider only redraws the plot
    frontier = rule_frontier(positives, SETUP)
    return (frontier,)


@app.cell
def _(INK, PercentFormatter, RULE_NAMES, RULE_STYLE, ceiling_slider, frontier, mu_slider, np, plt):
    def plot_frontier(points, current):
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        for name, (color, marker) in RULE_STYLE.items():
            params, pass_rates, recalls = zip(*points[name])
            ax.plot(pass_rates, recalls, color=color, lw=2, marker=marker, ms=4, label=RULE_NAMES[name])
            i = int(np.argmin(np.abs(np.array(params) - current[name])))
            ax.plot(pass_rates[i], recalls[i], marker=marker, color=color, ms=11, mec="white", mew=2, zorder=5)
            param_label = f"{current[name]:.0%}" if name == "ceiling" else f"μ={current[name]:g}"
            ax.annotate(
                param_label,
                (pass_rates[i], recalls[i]),
                xytext=(6, -12),
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
        ax.set_xlabel("mean pass rate across cases (BERT compute scales with it)")
        ax.set_ylabel("mean expected recall across cases")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        ax.set_title("Trade-off of each rule family (large markers: current settings)", fontsize=10)
        fig.tight_layout()
        return fig

    plot_frontier(frontier, {"ceiling": ceiling_slider.value, "exchange": mu_slider.value})
    return


@app.cell(hide_code=True)
def _(RECALL_TARGET, mo):
    mo.md(rf"""
    ## Q5. Does the cliff's location track how common the case is?

    If letting through a fixed multiple c of the expected positives were a good rule, the pass rate needed for
    {RECALL_TARGET:.0%} recall would rise along the gray lines (pass rate ≈ c × prevalence). The prevalence here is only
    relative between cases: positives come from every train-split document, while negatives are sampled from 600.
    """)
    return


@app.cell
def _(
    INK,
    MUTED,
    PercentFormatter,
    RECALL_TARGET,
    SELECTED_COLOR,
    SETUP,
    case_info,
    pass_rate_for_recall,
    plt,
    positives,
    stats,
):
    def plot_prevalence(positives, case_info, setup):
        needed = (
            positives[positives.setup == setup]
            .groupby("case_id")
            .pass_rate_needed.apply(lambda q: pass_rate_for_recall(q.values, RECALL_TARGET))
            .reindex(case_info.index)
            .clip(lower=1e-3)  # A log axis can't show 0
        )
        rho, pvalue = stats.spearmanr(case_info.prevalence, needed)
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.scatter(case_info.prevalence, needed, color=SELECTED_COLOR, s=40, edgecolor="white", linewidth=1.5, zorder=3)
        # Label every case when there are few, otherwise just the hardest
        labeled = needed.index if len(needed) <= 20 else needed.nlargest(10).index
        for case_id in labeled:
            ax.annotate(
                str(case_id),
                (case_info.prevalence[case_id], needed[case_id]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7.5,
                color=INK,
            )
        xs = [case_info.prevalence.min() / 1.3, case_info.prevalence.max() * 1.3]
        for c in [4, 50]:
            ax.plot(xs, [c * x for x in xs], color=MUTED, lw=1)
            label_x = min(xs[1], 0.9 / c)  # Where the line leaves the top of the plot, if it does
            ax.annotate(
                f"{c}× prevalence",
                (label_x, c * label_x),
                xytext=(-4, -11),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color=MUTED,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(max(needed.min() / 2, 1e-3), 1)
        ax.set_xlim(*xs)
        ax.set_xticks([t for t in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2] if xs[0] <= t <= xs[1]])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        ax.set_xlabel("dataset prevalence (positives / all sentences, relative only)")
        ax.set_ylabel(f"pass rate needed for {RECALL_TARGET:.0%} recall")
        ax.set_title(f"{setup} · Spearman ρ = {rho:.2f} (p = {pvalue:.2g}, {len(case_info)} cases)", fontsize=10)
        fig.tight_layout()
        return fig

    plot_prevalence(positives, case_info, SETUP)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q6. How do the shipped models do on the test split?

    Each case's shipped model, scored on test-split sentences: services that nothing above has seen, neither training
    nor any choice of setup, rule or μ. As in the train split, the negatives are sentences sampled from comprehensively
    reviewed documents, and the positives are every approved test-split point. The pass rate is compared with the
    target the full model's threshold aimed for, and recall with the expected recall from the CV (Q4).

    This is for information only. Choosing anything by these numbers would let test-split labels shape the prefilter,
    and the case models' thresholds are picked on test-split documents it has filtered. Some cases have only a handful
    of test positives, so a single case's recall is noisy: the means across cases are what to read.
    """)
    return


@app.cell
def _(MODELS_DIR, POINTS_PATH, case_info, pd, shipped, tfidf, utils):
    def load_test() -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Per case: test-split pass rate and recall at the shipped threshold, next to their targets. And every test
        positive, with the pass rate it needed (against the case's test negatives) and whether the threshold missed it
        """
        results, positives = {}, []
        for case_id in case_info.index:
            if not (path := MODELS_DIR / str(case_id) / "test.pkl").exists():
                continue
            test = pd.read_pickle(path)
            through = test.score >= shipped.threshold[case_id]
            pos, neg = test.label == 1, test.label == 0
            results[case_id] = {
                "n_pos": pos.sum(),
                "n_neg": neg.sum(),
                "target_pass_rate": shipped.pass_rate[case_id],
                "pass_rate": through[neg].mean(),
                "expected_recall": shipped.expected_recall[case_id],
                "recall": through[pos].mean(),
                "missed": (~through[pos]).sum(),
            }
            positives.append(
                pd.DataFrame(
                    {
                        "case_id": case_id,
                        "point_id": test.point_id[pos].astype(int),
                        "pass_rate_needed": tfidf.pass_rates_needed(test.score[pos].values, test.score[neg].values),
                        "missed": ~through[pos],
                    }
                )
            )
        positives = pd.concat(positives, ignore_index=True)
        # Cleaned the same way as in tfidf.prep_datasets
        quotes = pd.read_pickle(POINTS_PATH).quote_text.reindex(positives.point_id.unique())
        positives["text"] = positives.point_id.map(quotes.map(utils.preprocess_doc_text))
        return pd.DataFrame.from_dict(results, orient="index"), positives

    test_results, test_positives = load_test()
    test_results["pass_rate_ratio"] = test_results.pass_rate / test_results.target_pass_rate
    return test_positives, test_results


@app.cell
def _(INK, MUTED, PercentFormatter, SELECTED_COLOR, mo, np, plt, test_results):
    def plot_test_pass_rates(results, n_labeled=6):
        """Test-split pass rate against the target, per case"""
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        floor = 1e-3  # A log axis can't show 0
        lims = (min(results.target_pass_rate.min(), results.pass_rate.clip(lower=floor).min()) / 1.5, 1)
        ax.plot(lims, lims, color=MUTED, lw=1)
        ax.annotate(
            "on target", (0.5, 0.5), xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=8, color=MUTED
        )
        ax.scatter(
            results.target_pass_rate,
            results.pass_rate.clip(lower=floor),
            color=SELECTED_COLOR,
            s=40,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        # The cases furthest off target, either way, skipping any too close to an already labeled one
        labeled = []
        for case_id in np.log(results.pass_rate_ratio.clip(lower=1e-3)).abs().nlargest(n_labeled).index:
            xy = (results.target_pass_rate[case_id], max(results.pass_rate[case_id], floor))
            pixels = ax.transData.transform(xy)
            if any(np.hypot(*(pixels - other)) < 15 for other in labeled):
                continue
            labeled.append(pixels)
            ax.annotate(str(case_id), xy, xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        ax.set_xlabel("target pass rate (chosen on the train split's CV)")
        ax.set_ylabel("pass rate on test-split negatives")
        ax.set_title(
            f"Test-split pass rate vs. target · median ratio {results.pass_rate_ratio.median():.2f} "
            f"({len(results)} cases)",
            fontsize=10,
        )
        fig.tight_layout()
        return fig

    _r = test_results
    _ratio = _r.pass_rate_ratio.quantile([0.25, 0.75])
    _summary = (
        f"**{len(_r)} cases**, {_r.n_pos.sum()} test positives ({_r.n_pos.min()}–{_r.n_pos.max()} per case) and "
        f"{_r.n_neg.median():.0f} test negatives per case.\n\n"
        f"**Pass rate:** mean {100 * _r.pass_rate.mean():.1f}% vs. a mean target of "
        f"{100 * _r.target_pass_rate.mean():.1f}%. Per case, the test pass rate was {_r.pass_rate_ratio.median():.2f}× "
        f"the target at the median (middle half {_ratio[0.25]:.2f}–{_ratio[0.75]:.2f}×).\n\n"
        f"**Recall:** mean {100 * _r.recall.mean():.1f}% vs. a mean expected recall of "
        f"{100 * _r.expected_recall.mean():.1f}% ({100 * (1 - _r.missed.sum() / _r.n_pos.sum()):.1f}% of all test "
        f"positives). {_r.missed.sum()} positives missed, in {(_r.missed > 0).sum()} cases."
    )
    mo.vstack([mo.md(_summary), plot_test_pass_rates(_r)])
    return


@app.cell
def _(case_info, mo, test_results):
    def test_table(case_info, results):
        table = case_info[["title"]].join(results, how="inner")
        for col in ["target_pass_rate", "pass_rate", "expected_recall", "recall"]:
            table[col] = (100 * table[col]).round(1)
        table["pass_rate_ratio"] = table.pass_rate_ratio.round(2)
        return table.rename(
            columns={
                "n_pos": "test positives",
                "n_neg": "test negatives",
                "target_pass_rate": "target pass rate (%)",
                "pass_rate": "test pass rate (%)",
                "pass_rate_ratio": "test / target",
                "expected_recall": "expected recall (%)",
                "recall": "test recall (%)",
            }
        )

    mo.ui.table(test_table(case_info, test_results).rename_axis("case").reset_index(), selection=None, page_size=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Missed test positives

    Every test positive the shipped threshold filters out, with the pass rate it would have needed on the test split
    next to its case's target. As with the audit list in Q2, mislabeled points here cost nothing to miss.
    """)
    return


@app.cell
def _(EDIT_URL, case_info, mo, pd, shipped, test_positives):
    def missed_table(positives, case_info, shipped) -> pd.DataFrame:
        missed = positives[positives.missed].sort_values(["case_id", "pass_rate_needed"], ascending=[True, False])
        return pd.DataFrame(
            {
                "case": missed.case_id,
                "case title": missed.case_id.map(case_info.title),
                "pass rate needed (%)": (100 * missed.pass_rate_needed).round(1),
                "case target (%)": (100 * missed.case_id.map(shipped.pass_rate)).round(1),
                "text": missed.text.str.strip().str.slice(0, 300),
                "point": EDIT_URL + "/points/" + missed.point_id.astype(str),
            }
        ).reset_index(drop=True)

    mo.ui.table(missed_table(test_positives, case_info, shipped), selection=None, page_size=20)
    return


if __name__ == "__main__":
    app.run()
