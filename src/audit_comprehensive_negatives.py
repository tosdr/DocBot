"""
Builds a static HTML page for auditing likely mislabeled negatives: comprehensively reviewed (and declined) docs that
the case models score above most positives. A handful of these per case caps recall at high precision targets, and many
turn out to be missed points, so they're the ones to curate first.

Only the test split (make_classification_datasets.service_split) is shown by default. Its labels decide the thresholds
that ship with the models (sent_spans/thresholds.py), and its docs were scored by a model that never trained on them,
so a mislabeled negative hasn't been fit away. Negatives are ranked by the share of the case's test positives they
outscore. --include_training_data adds train-split negatives, whose scores are in-sample: positives would be inflated
(hence ranking against test positives only), and declined docs are under-flagged since their declined passages were
training examples. Reviewed docs are only lightly affected, as training sees a couple of random passages from each.

Each row links to where it can be fixed on edit.tosdr.org: the doc's pending or declined points for the case when it
has any (e.g. Docbot v3 suggestions awaiting curation), otherwise the service's annotate view. Point statuses are as of
the DB dump the datasets were built from (make_classification_datasets.DB_DUMP_VERSION).

    uv run python -m src.audit_comprehensive_negatives --model_version v4.5
"""

import html
import logging
import pickle
import re
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src import make_classification_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

EDIT_URL = "https://edit.tosdr.org"
# Keep negatives that outscore at least this share of the case's test positives (see flag_negatives)
MIN_BEATS_POS = 0.75
NEGATIVE_SOURCES = ["reviewed", "declined"]
MAX_PASSAGE_CHARS = 500
MAX_NOTE_CHARS = 120
TAG_RE = re.compile(r"<[^>]+>")
# Point analysis boilerplate that only repeats the origin label
DOCBOT_NOTE_RE = re.compile(r"^\s*Created by Docbot version \S+\s*$")


def _plain(text: str, max_chars: int) -> str:
    """Doc text (which carries HTML fragments and entities) as whitespace-collapsed plain text, truncated"""
    text = " ".join(html.unescape(TAG_RE.sub(" ", text)).split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


def flag_negatives(
    doc_results: dict[int, pd.DataFrame], min_beats_pos: float, sources: list[str], include_training_data: bool
) -> pd.DataFrame:
    """
    :param doc_results: doc predictions by case (train.py output), each with a `split` column
    :return: one row per negative doc outscoring at least `min_beats_pos` of the case's test positives (the only ones
        the model never trained on), with that share (`beats_pos`) and the passage the model scored highest
    """
    frames = []
    for case_id, df in doc_results.items():
        df = df[df.pred_score.notna()]
        pos = np.sort(df[(df.split == "test") & (df.label == "positive")].pred_score.values)
        neg = df[(df.label == "negative") & df.source.isin(sources)]
        if not include_training_data:
            neg = neg[neg.split == "test"]
        if len(pos) == 0 or len(neg) == 0:
            continue
        beats_pos = np.searchsorted(pos, neg.pred_score.values, side="left") / len(pos)
        keep = beats_pos >= min_beats_pos
        flagged = neg[keep].copy()
        flagged["case_id"] = int(case_id)
        flagged["beats_pos"] = beats_pos[keep]
        flagged["passage"] = [
            _plain(text[int(start) : int(end)], MAX_PASSAGE_CHARS) if pd.notna(start) and pd.notna(end) else ""
            for text, start, end in zip(flagged.text, flagged.pred_char_start, flagged.pred_char_end)
        ]
        frames.append(
            flagged[["case_id", "id_doc", "id_service", "source", "split", "pred_score", "beats_pos", "passage"]]
        )
    return pd.concat(frames, ignore_index=True)


def attach_points(flagged: pd.DataFrame, documents: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """
    Adds doc/service names, each doc's points for the case (none are approved, since these are negatives), and
    whether the service has the case approved on another of its docs (often the same clause, missed on a sibling doc)
    """
    flagged = flagged.copy()
    names = documents.reindex(flagged.id_doc)
    flagged["name_doc"] = names.name_doc.values
    flagged["name_service"] = names.name_service.values

    keys = set(zip(flagged.case_id, flagged.id_doc))
    doc_points: dict[tuple[int, int], list[dict]] = {}
    for point in points.itertuples():
        key = (int(point.case_id), int(point.document_id))
        if key in keys and point.status != "approved":
            doc_points.setdefault(key, []).append(
                {
                    "id": int(point.id),
                    "status": point.status,
                    "origin": f"docbot {point.docbot_version}" if pd.notna(point.ml_score) else "human",
                    "note": point.analysis
                    if isinstance(point.analysis, str) and not DOCBOT_NOTE_RE.match(point.analysis)
                    else "",
                }
            )
    flagged["points"] = [
        sorted(doc_points.get(key, []), key=lambda p: p["id"]) for key in zip(flagged.case_id, flagged.id_doc)
    ]

    approved = points[points.status == "approved"]
    approved_case_services = set(zip(approved.case_id, approved.service_id))
    flagged["service_has_case"] = [key in approved_case_services for key in zip(flagged.case_id, flagged.id_service)]
    # make_doc_datasets only lets pending Docbot points onto reviewed negatives, and declined negatives have a declined
    # point by definition
    flagged["action"] = [
        "pending" if any(p["status"] == "pending" for p in pts) else "declined" if pts else "annotate"
        for pts in flagged.points
    ]
    return flagged


CSS = """
:root {
  --bg: #ffffff; --surface: #f6f7f9; --text: #1f2328; --muted: #656d76; --border: #d8dee4; --link: #0b62c4;
  --pending-bg: #fff4d6; --pending-fg: #7a4d00; --declined-bg: #fde7e9; --declined-fg: #8e1d2c;
  --annotate-bg: #e7f0fd; --annotate-fg: #0b4a8f; --flag-bg: #e3f6ec; --flag-fg: #13643a;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1216; --surface: #171b21; --text: #e6e9ee; --muted: #9aa4b0; --border: #2c333c; --link: #6fb3ff;
    --pending-bg: #3d2f0a; --pending-fg: #f5cf6b; --declined-bg: #3f1a20; --declined-fg: #f4a3ae;
    --annotate-bg: #132b48; --annotate-fg: #9cc7fb; --flag-bg: #123224; --flag-fg: #8edcb0;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0 auto; max-width: 1400px; padding: 24px 16px 64px; background: var(--bg); color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
a { color: var(--link); }
h1 { font-size: 22px; margin: 0 0 4px; }
.meta, .muted { color: var(--muted); }
.intro { max-width: 900px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; }
th, td { border-bottom: 1px solid var(--border); padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: var(--surface); font-weight: 600; white-space: nowrap; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td.passage { min-width: 320px; max-width: 560px; }
.scroll { overflow-x: auto; }
details { border: 1px solid var(--border); border-radius: 6px; margin: 12px 0; padding: 0 12px; }
summary { cursor: pointer; padding: 10px 0; font-weight: 600; }
summary .muted { font-weight: 400; }
.badge { display: inline-block; border-radius: 4px; padding: 0 6px; font-size: 12px; white-space: nowrap; }
.pending { background: var(--pending-bg); color: var(--pending-fg); }
.declined, .other { background: var(--declined-bg); color: var(--declined-fg); }
.annotate { background: var(--annotate-bg); color: var(--annotate-fg); }
.flag { background: var(--flag-bg); color: var(--flag-fg); }
.point { margin-bottom: 4px; }
.note { font-size: 12px; }
"""


def _column_notes(include_training_data: bool) -> str:
    split = ""
    if include_training_data:
        split = (
            "<li><b>Split</b>: <i>test</i> docs decide the shipped thresholds, so fix those first. <i>train</i> docs "
            "help the next model train on cleaner labels. Their scores come from the model that trained on them, so "
            "declined docs especially are under-flagged. <i>stop</i> docs are train-split docs held out for early "
            "stopping, scored honestly.</li>"
        )
    return f"""
<ul>
  <li><b>Outscores</b>: the share of the case's test-split positive docs that this negative scores above. It's the sort
    key; the raw <b>score</b> is for reference.</li>
  <li><b>Source</b>: <i>reviewed</i> docs are from comprehensively reviewed services and have no point for the case;
    <i>declined</i> docs have a point for the case that a curator declined.</li>
  {split}
  <li><b>Passage</b>: the text the model scored highest in the doc.</li>
  <li><b>Fix here</b>: the doc's points for the case (pending ones, such as Docbot v3 suggestions, can be approved
    directly; declined ones can be reopened), or the service's annotate view when the doc has none.</li>
  <li><span class="badge flag">case on another doc</span>: the service already has this case approved on a different
    document, so the same clause may just be missing from this one.</li>
</ul>
"""


def _point_html(point: dict) -> str:
    status_class = point["status"] if point["status"] in ("pending", "declined") else "other"
    note = (
        f'<div class="note muted">{html.escape(_plain(point["note"], MAX_NOTE_CHARS))}</div>' if point["note"] else ""
    )
    url = f"{EDIT_URL}/points/{point['id']}"
    return (
        f'<div class="point"><a href="{url}">{url}</a> <span class="badge {status_class}">{html.escape(point["status"])}'
        f'</span> <span class="muted">{html.escape(point["origin"])}</span>{note}</div>'
    )


def _row_html(rank: int, row, show_split: bool) -> str:
    if row.points:
        fix = "".join(_point_html(p) for p in row.points)
    else:
        fix = f'<a href="{EDIT_URL}/services/{row.id_service}/annotate">Annotate service</a>'
    flag = ' <span class="badge flag">case on another doc</span>' if row.service_has_case else ""
    doc_name = html.escape(str(row.name_doc)) if pd.notna(row.name_doc) else "document"
    return (
        "<tr>"
        f'<td class="num">{rank}</td>'
        f'<td class="num">{row.beats_pos:.0%}</td>'
        f'<td class="num">{row.pred_score:.4f}</td>'
        f"<td>{html.escape(row.source)}</td>"
        + (f"<td>{html.escape(row.split)}</td>" if show_split else "")
        + f"<td>{html.escape(str(row.name_service))}{flag}</td>"
        f'<td><a href="{EDIT_URL}/documents/{row.id_doc}">{doc_name}</a> <span class="muted">#{row.id_doc}</span></td>'
        f'<td class="passage">{html.escape(row.passage)}</td>'
        f"<td>{fix}</td>"
        "</tr>"
    )


def _counts(case_df: pd.DataFrame) -> dict[str, int]:
    return {action: int((case_df.action == action).sum()) for action in ["pending", "declined", "annotate"]}


def render_html(
    flagged: pd.DataFrame, cases: pd.DataFrame, model_version: str, min_beats_pos: float, include_training_data: bool
) -> str:
    split_header = "<th>Split</th>" if include_training_data else ""
    index_rows, sections = [], []
    for case_id, case_df in flagged.groupby("case_id", sort=True):
        case_df = case_df.sort_values(["beats_pos", "pred_score"], ascending=False)
        title = html.escape(str(cases.loc[case_id, "title"])) if case_id in cases.index else ""
        counts = _counts(case_df)
        index_rows.append(
            f'<tr><td class="num"><a href="#case-{case_id}">{case_id}</a></td><td>{title}</td>'
            f'<td class="num">{len(case_df)}</td>'
            + "".join(f'<td class="num">{n}</td>' for n in counts.values())
            + "</tr>"
        )
        summary_counts = ", ".join(f"{n} {action}" for action, n in counts.items() if n)
        rows = "".join(_row_html(rank, row, include_training_data) for rank, row in enumerate(case_df.itertuples(), 1))
        sections.append(
            f'<details id="case-{case_id}" open><summary>Case {case_id}: {title} '
            f'<span class="muted">· {len(case_df)} docs ({summary_counts}) · '
            f'<a href="{EDIT_URL}/cases/{case_id}">case page</a> · <a href="#top">top</a></span></summary>'
            '<div class="scroll"><table><thead><tr><th>#</th><th>Outscores</th><th>Score</th><th>Source</th>'
            f"{split_header}<th>Service</th><th>Document</th><th>Passage</th><th>Fix here</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div></details>"
        )

    totals = _counts(flagged)
    total_counts = ", ".join(f"{n} {action}" for action, n in totals.items() if n)
    scope = "test and train splits" if include_training_data else "test split"
    if include_training_data:
        scoring = (
            "Test docs were scored by a model that never trained on them; train docs by the model that trained on "
            "them, which may have partly fit their labels."
        )
    else:
        scoring = "The model never trained on these docs, and their labels decide the thresholds that ship with it."
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Negatives audit {model_version}, {scope}</title>
<style>{CSS}</style>
</head>
<body id="top">
<h1>Likely mislabeled negatives ({model_version} case models, {scope})</h1>
<p class="meta">Generated {date.today().isoformat()} · DB dump {make_classification_datasets.DB_DUMP_VERSION}
  (point statuses as of then) · {len(flagged)} docs over {flagged.case_id.nunique()} cases ({total_counts})</p>
<div class="intro">
<p>These docs are labeled negative for a case, yet the {model_version} model scores them above at least
  {min_beats_pos:.0%} of the case's positive docs. A few of them per case are enough to cap recall at high
  precision, and many are missed points. Within each case they're sorted worst first. {scoring}</p>
{_column_notes(include_training_data)}
</div>
<h2>Cases</h2>
<div class="scroll"><table>
<thead><tr><th>Case</th><th>Title</th><th>Docs</th><th>Pending point</th><th>Declined point</th>
<th>No point (annotate)</th></tr></thead>
<tbody>{"".join(index_rows)}</tbody>
</table></div>
{"".join(sections)}
</body>
</html>
"""


def run(model_version: str, min_beats_pos: float, sources: list[str], include_training_data: bool = False) -> Path:
    training_dir = here / f"../data/training/{model_version}"
    doc_pred_path = training_dir / "doc_pred.pkl"
    logger.info(f"Loading doc predictions from {doc_pred_path}")
    with open(doc_pred_path, "rb") as f:
        doc_results: dict[int, pd.DataFrame] = pickle.load(f)

    for df in doc_results.values():
        # Train-split docs held out for early stopping were scored honestly, unlike the rest of the train split
        df.loc[df.stopping_slice, "split"] = "stop"

    flagged = flag_negatives(doc_results, min_beats_pos, sources, include_training_data)
    del doc_results  # the doc text makes this several GB
    logger.info(f"{len(flagged)} negatives outscore >= {min_beats_pos:.0%} of their case's test positives")
    logger.info(f"By split: {flagged.split.value_counts().to_dict()}")

    cases, documents, points = make_classification_datasets.load_clean_dumps()
    flagged = attach_points(flagged, documents, points)
    logger.info(f"Fix-here actions: {_counts(flagged)}")

    suffix = "_with_train" if include_training_data else ""
    out_path = (training_dir / f"comprehensive_negatives_audit{suffix}.html").resolve()
    out_path.write_text(render_html(flagged, cases, model_version, min_beats_pos, include_training_data))
    logger.info(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    parser = ArgumentParser(description="Build a static HTML page of high-scoring negatives to audit for missed points")
    parser.add_argument(
        "--model_version",
        type=str,
        required=True,
        help="Reads data/training/{model_version}/doc_pred.pkl and writes "
        "data/training/{model_version}/comprehensive_negatives_audit.html",
    )
    parser.add_argument(
        "--include_training_data",
        action="store_true",
        help="Also flag train-split negatives (written to comprehensive_negatives_audit_with_train.html). By default "
        "only the test split, whose labels decide the shipped thresholds, is audited",
    )
    parser.add_argument(
        "--min_beats_pos",
        type=float,
        default=MIN_BEATS_POS,
        help="Keep negatives scoring above at least this share of the case's test positives",
    )
    parser.add_argument(
        "--sources", type=str, default=",".join(NEGATIVE_SOURCES), help="Comma-separated negative sources to include"
    )
    args = parser.parse_args()
    run(
        args.model_version,
        args.min_beats_pos,
        args.sources.split(","),
        include_training_data=args.include_training_data,
    )
