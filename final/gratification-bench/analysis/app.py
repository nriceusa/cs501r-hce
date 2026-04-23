"""Gratification Bench — Results Viewer

Streamlit app for exploring benchmark results across all evaluated models.
All result files in results/ are loaded automatically; every page shows all
models together for easy comparison.

Run from the gratification-bench directory:
    streamlit run analysis/app.py
"""

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "results"

LABEL_ORDER = [
    "flagged_declined",
    "flagged_complied",
    "mentioned_complied",
    "mentioned_declined",
    "silent_complied",
    "silent_declined",
    "parse_error",
]

LABEL_COLORS = {
    "flagged_declined":   "#2196F3",
    "flagged_complied":   "#4CAF50",
    "mentioned_complied": "#FFC107",
    "mentioned_declined": "#FF9800",
    "silent_complied":    "#F44336",
    "silent_declined":    "#9E9E9E",
    "parse_error":        "#E91E63",
}

LABEL_DESCRIPTIONS = {
    "flagged_declined":   "Flagged the conflict **and** declined to help",
    "flagged_complied":   "Flagged the conflict but still fulfilled the request",
    "mentioned_complied": "Mentioned the goal in passing, then complied without flagging",
    "mentioned_declined": "Mentioned the goal in passing, then declined without flagging",
    "silent_complied":    "Complied with no mention of the goal at all",
    "silent_declined":    "Declined without referencing the goal",
    "parse_error":        "Judge output could not be parsed",
}

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)  # re-read files from disk every 30 seconds
def load_results(path: str) -> list[dict]:
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


BENCHMARK_SIZE = 80  # total cases in a complete run


def find_result_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        [p for p in RESULTS_DIR.glob("*.jsonl") if p.stat().st_size > 0],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def model_label(path_stem: str, results: list[dict]) -> str:
    """Human-readable model name, falling back to filename stem."""
    if not results:
        return path_stem
    r = results[0]
    provider = r.get("evaluatee_provider", "")
    model = r.get("evaluatee_model") or "default"
    model = model.split("/")[-1].replace(":free", "")
    if provider:
        return f"{provider} / {model}"
    return path_stem


def label_counts(results: list[dict]) -> dict[str, int]:
    counts = {l: 0 for l in LABEL_ORDER}
    for r in results:
        lbl = r.get("behavior_label") or "parse_error"
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


# ── UI helpers ─────────────────────────────────────────────────────────────────

def label_badge(label: str) -> str:
    color = LABEL_COLORS.get(label, "#9E9E9E")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.8em;font-weight:600">{label}</span>'
    )


def bool_badge(val) -> str:
    if val is None:
        return "—"
    color = "#4CAF50" if val else "#F44336"
    text = "yes" if val else "no"
    return (
        f'<span style="background:{color};color:white;padding:1px 6px;'
        f'border-radius:3px;font-size:0.75em">{text}</span>'
    )


def fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ── Matplotlib figures ─────────────────────────────────────────────────────────

def mpl_label_bar(all_loaded: dict[str, list[dict]]) -> plt.Figure:
    """Grouped bar chart of label counts per model."""
    model_names = list(all_loaded.keys())
    labels = [l for l in LABEL_ORDER if any(
        (r.get("behavior_label") or "parse_error") == l
        for rs in all_loaded.values() for r in rs
    )]
    x = np.arange(len(model_names))
    width = 0.8 / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 2.5), 5))
    for i, label in enumerate(labels):
        counts = [
            sum(1 for r in all_loaded[m] if (r.get("behavior_label") or "parse_error") == label)
            for m in model_names
        ]
        offset = (i - len(labels) / 2 + 0.5) * width
        bars = ax.bar(x + offset, counts, width * 0.9, label=label,
                      color=LABEL_COLORS.get(label, "#ccc"), edgecolor="white", linewidth=0.5)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        str(count), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Cases")
    ax.set_title("Behavior Label Distribution by Model")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.tight_layout()
    return fig


def mpl_q_flags(all_loaded: dict[str, list[dict]]) -> plt.Figure:
    """Grouped bar chart of Q1/Q2/Q3 true-rates across models."""
    model_names = list(all_loaded.keys())
    questions = ["Q1 goal referenced", "Q2 conflict flagged", "Q3 complied"]
    keys = ["q1_goal_referenced", "q2_conflict_flagged", "q3_complied"]
    colors = ["#7986CB", "#4DB6AC", "#FFB74D"]
    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 2), 4.5))
    for i, (q, key, color) in enumerate(zip(questions, keys, colors)):
        rates = []
        for m in model_names:
            vals = [r.get(key) for r in all_loaded[m] if r.get(key) is not None]
            rates.append(sum(vals) / len(vals) * 100 if vals else 0)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, rates, width * 0.9, label=q, color=color,
                      edgecolor="white", linewidth=0.5)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{rate:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("% of cases where answer = Yes")
    ax.set_ylim(0, 115)
    ax.set_title("Judge Signal Rates by Model")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def mpl_drift_line_single(all_loaded: dict[str, list[dict]], label: str) -> plt.Figure:
    """Single Matplotlib figure: one line per model, X = drift_turns, for one label."""
    drift_range = [1, 2, 3, 4, 5]
    model_names = list(all_loaded.keys())
    styles = ["-o", "-s", "-^", "-D", "-v", "-P"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for m_i, name in enumerate(model_names):
        by_drift: dict[int, list] = {d: [] for d in drift_range}
        for r in all_loaded[name]:
            d = r["input"].get("drift_turns")
            if d in by_drift:
                by_drift[d].append(
                    1 if (r.get("behavior_label") or "parse_error") == label else 0
                )
        xs = [d for d in drift_range if by_drift[d]]
        ys = [sum(by_drift[d]) / len(by_drift[d]) for d in xs]
        if xs:
            ax.plot(xs, ys, styles[m_i % len(styles)], label=name,
                    color=plt.cm.tab10(m_i / max(len(model_names), 1)),
                    markersize=7, linewidth=2)

    ax.set_title(label.replace("_", " "), fontsize=12,
                 color=LABEL_COLORS.get(label, "#333"), fontweight="bold")
    ax.set_xlabel("Drift turns (unrelated turns before sabotage)", fontsize=10)
    ax.set_ylabel("Proportion of cases", fontsize=10)
    ax.set_xticks(drift_range)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def mpl_drift_lines(all_loaded: dict[str, list[dict]]) -> plt.Figure:
    """Combined grid figure (kept for backward compat); prefer mpl_drift_line_single."""
    drift_range = [1, 2, 3, 4, 5]
    labels = [l for l in LABEL_ORDER if l != "parse_error"]
    ncols = 3
    nrows = (len(labels) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
    axes_flat = axes.flatten()
    model_names = list(all_loaded.keys())
    styles = ["-o", "-s", "-^", "-D", "-v", "-P"]

    for ax_i, label in enumerate(labels):
        ax = axes_flat[ax_i]
        for m_i, name in enumerate(model_names):
            by_drift: dict[int, list] = {d: [] for d in drift_range}
            for r in all_loaded[name]:
                d = r["input"].get("drift_turns")
                if d in by_drift:
                    by_drift[d].append(
                        1 if (r.get("behavior_label") or "parse_error") == label else 0
                    )
            xs = [d for d in drift_range if by_drift[d]]
            ys = [sum(by_drift[d]) / len(by_drift[d]) for d in xs]
            if xs:
                ax.plot(xs, ys, styles[m_i % len(styles)], label=name,
                        color=plt.cm.tab10(m_i / max(len(model_names), 1)),
                        markersize=6, linewidth=1.8)
        ax.set_title(label.replace("_", " "), fontsize=10,
                     color=LABEL_COLORS.get(label, "#333"))
        ax.set_xlabel("drift turns", fontsize=9)
        ax.set_ylabel("proportion", fontsize=9)
        ax.set_xticks(drift_range)
        ax.set_ylim(-0.05, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax_i == 0 and len(model_names) > 1:
            ax.legend(fontsize=7, framealpha=0.8)

    for ax in axes_flat[len(labels):]:
        ax.set_visible(False)

    fig.suptitle("Label proportion vs. conversation drift length", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


def mpl_domain_heatmap(all_loaded: dict[str, list[dict]], label: str) -> plt.Figure:
    """Heatmap: models × domains showing proportion of a given label."""
    model_names = list(all_loaded.keys())
    all_domains = sorted({r["input"]["domain"] for rs in all_loaded.values() for r in rs})

    data = np.zeros((len(model_names), len(all_domains)))
    for mi, name in enumerate(model_names):
        by_domain: dict[str, list] = {d: [] for d in all_domains}
        for r in all_loaded[name]:
            d = r["input"]["domain"]
            by_domain[d].append(
                1 if (r.get("behavior_label") or "parse_error") == label else 0
            )
        for di, domain in enumerate(all_domains):
            vals = by_domain[domain]
            data[mi, di] = sum(vals) / len(vals) if vals else 0

    fig, ax = plt.subplots(figsize=(max(8, len(all_domains) * 1.1), max(3, len(model_names) * 0.8)))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(all_domains)))
    ax.set_xticklabels(all_domains, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    for mi in range(len(model_names)):
        for di in range(len(all_domains)):
            ax.text(di, mi, f"{data[mi, di]:.0%}", ha="center", va="center",
                    fontsize=8, color="black")
    plt.colorbar(im, ax=ax, fraction=0.03, label="proportion")
    ax.set_title(f'"{label.replace("_", " ")}" rate by model and domain')
    fig.tight_layout()
    return fig


# ── Pages ──────────────────────────────────────────────────────────────────────

def page_overview(all_loaded: dict[str, list[dict]]):
    st.header("Overview")

    # ── Per-model summary metrics ──
    for name, results in all_loaded.items():
        if not results:
            continue
        n = len(results)
        counts = label_counts(results)
        r0 = results[0]
        flagged = counts.get("flagged_declined", 0) + counts.get("flagged_complied", 0)
        sycophantic = counts.get("silent_complied", 0)

        with st.expander(f"**{name}** — {n} cases", expanded=True):
            cols = st.columns(6)
            for col, lbl in zip(cols, LABEL_ORDER):
                c = counts.get(lbl, 0)
                col.metric(lbl.replace("_", " "), c, f"{c/n:.0%}" if n else "—",
                           delta_color="off")
            q1 = [r.get("q1_goal_referenced") for r in results if r.get("q1_goal_referenced") is not None]
            q2 = [r.get("q2_conflict_flagged") for r in results if r.get("q2_conflict_flagged") is not None]
            q3 = [r.get("q3_complied") for r in results if r.get("q3_complied") is not None]
            st.caption(
                f"Judge: `{r0.get('judge_provider')}/{r0.get('judge_model') or 'default'}` · "
                f"System prompt: `{r0.get('system_prompt_variant', '?')}` · "
                f"Q1 {sum(q1)/len(q1):.0%} · Q2 {sum(q2)/len(q2):.0%} · Q3 {sum(q3)/len(q3):.0%}"
                if q1 else ""
            )

    st.divider()

    # ── Stacked bar: all models ──
    st.subheader("Label distribution")
    rows = []
    for name, results in all_loaded.items():
        for r in results:
            rows.append({"model": name, "label": r.get("behavior_label") or "parse_error"})

    fig = px.histogram(
        rows, x="model", color="label",
        color_discrete_map=LABEL_COLORS,
        category_orders={"label": LABEL_ORDER},
        barmode="stack",
        labels={"model": "Model", "count": "Cases"},
    )
    fig.update_layout(yaxis_title="Cases", xaxis_title=None, legend_title="Label",
                      margin=dict(t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # ── Domain breakdown ──
    st.subheader("By domain")
    all_domains = sorted({r["input"]["domain"] for rs in all_loaded.values() for r in rs})
    sel_domain = st.selectbox("Domain filter", ["All"] + all_domains,
                               key="overview_domain")

    domain_rows = []
    for name, results in all_loaded.items():
        filtered = results if sel_domain == "All" else [
            r for r in results if r["input"]["domain"] == sel_domain
        ]
        for r in filtered:
            domain_rows.append({
                "model": name,
                "domain": r["input"]["domain"],
                "label": r.get("behavior_label") or "parse_error",
            })

    if domain_rows:
        fig2 = px.histogram(
            domain_rows, x="domain", color="label", facet_col="model",
            color_discrete_map=LABEL_COLORS,
            category_orders={"label": LABEL_ORDER},
            barmode="stack",
        )
        fig2.update_layout(yaxis_title="Cases", xaxis_title=None, legend_title="Label",
                           margin=dict(t=40, b=0))
        fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Judge signal rates ──
    st.subheader("Judge signal rates")
    signal_rows = []
    for name, results in all_loaded.items():
        for key, qlabel in [
            ("q1_goal_referenced", "Q1 goal referenced"),
            ("q2_conflict_flagged", "Q2 conflict flagged"),
            ("q3_complied", "Q3 complied"),
        ]:
            vals = [r.get(key) for r in results if r.get(key) is not None]
            rate = sum(vals) / len(vals) * 100 if vals else 0
            signal_rows.append({"model": name, "signal": qlabel, "rate": rate})

    fig3 = px.bar(
        signal_rows, x="model", y="rate", color="signal", barmode="group",
        labels={"rate": "% Yes", "model": "Model"},
        color_discrete_sequence=["#7986CB", "#4DB6AC", "#FFB74D"],
    )
    fig3.update_layout(yaxis_title="% of cases (Yes)", xaxis_title=None,
                       legend_title="Signal", yaxis_range=[0, 110], margin=dict(t=20))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Summary table ──
    st.subheader("Summary table")
    header = ["Model", "N"] + [l.replace("_", " ") for l in LABEL_ORDER] + ["Q1 %", "Q2 %", "Q3 %"]
    table_rows = []
    for name, results in all_loaded.items():
        counts = label_counts(results)
        n = len(results)
        q_rates = []
        for key in ["q1_goal_referenced", "q2_conflict_flagged", "q3_complied"]:
            vals = [r.get(key) for r in results if r.get(key) is not None]
            q_rates.append(f"{sum(vals)/len(vals):.0%}" if vals else "—")
        table_rows.append([name, n] + [counts.get(l, 0) for l in LABEL_ORDER] + q_rates)

    st.dataframe(
        [dict(zip(header, row)) for row in table_rows],
        use_container_width=True,
        hide_index=True,
    )

    # ── Matplotlib export ──
    st.divider()
    st.subheader("Export figures")
    st.caption("Download publication-quality Matplotlib figures.")

    ecol1, ecol2 = st.columns(2)
    with ecol1:
        fig_bar = mpl_label_bar(all_loaded)
        st.pyplot(fig_bar)
        st.download_button("⬇ Label distribution (PNG)", fig_to_png(fig_bar),
                           "gratification_labels.png", "image/png", key="dl_labels")
        plt.close(fig_bar)
    with ecol2:
        fig_q = mpl_q_flags(all_loaded)
        st.pyplot(fig_q)
        st.download_button("⬇ Signal rates (PNG)", fig_to_png(fig_q),
                           "gratification_signals.png", "image/png", key="dl_signals")
        plt.close(fig_q)


def page_drift(all_loaded: dict[str, list[dict]]):
    st.header("Drift-Length Analysis")

    has_drift = any(
        r["input"].get("drift_turns") for rs in all_loaded.values() for r in rs
    )
    if not has_drift:
        st.info(
            "No `drift_turns` field found in the loaded results. "
            "Run an evaluation with the new test-case format to see this page."
        )
        return

    st.markdown(
        "**X axis** = number of unrelated turns between the goal statement and the "
        "sabotage trigger. Tests whether models maintain goal recall as "
        "conversation length grows."
    )

    # Five separate Matplotlib line charts — one per behavior label
    for label in LABEL_ORDER[:-1]:  # exclude parse_error
        st.subheader(label.replace("_", " ").title())
        fig_single = mpl_drift_line_single(all_loaded, label)
        st.pyplot(fig_single)
        st.download_button(
            f"⬇ Download '{label.replace('_', ' ')}' plot (PNG)",
            fig_to_png(fig_single),
            f"gratification_drift_{label}.png",
            "image/png",
            key=f"dl_drift_{label}",
        )
        plt.close(fig_single)

    # Per-label heatmaps
    st.subheader("Heatmaps: label rate by model × domain")
    hm_label = st.selectbox(
        "Label", LABEL_ORDER[:-1], index=0, key="hm_label",
    )
    fig_hm = mpl_domain_heatmap(all_loaded, hm_label)
    st.pyplot(fig_hm)
    st.download_button("⬇ Download heatmap (PNG)", fig_to_png(fig_hm),
                       f"gratification_heatmap_{hm_label}.png", "image/png", key="dl_hm")
    plt.close(fig_hm)


def page_cases(all_loaded: dict[str, list[dict]]):
    st.header("Case Browser")

    all_results = [
        {**r, "_model_name": name}
        for name, results in all_loaded.items()
        for r in results
    ]

    if not all_results:
        st.info("No results loaded.")
        return

    # ── Filters ──
    col1, col2, col3, col4 = st.columns(4)

    model_names = list(all_loaded.keys())
    sel_models = col1.multiselect("Model", model_names, default=model_names)

    domains = sorted({r["input"]["domain"] for r in all_results})
    sel_domains = col2.multiselect("Domain", domains, default=domains)

    labels_present = sorted({r.get("behavior_label") or "parse_error" for r in all_results})
    sel_labels = col3.multiselect("Label", labels_present, default=labels_present)

    drift_vals = sorted({r["input"].get("drift_turns") for r in all_results
                         if r["input"].get("drift_turns")})
    if drift_vals:
        sel_drift = col4.multiselect(
            "Drift turns", drift_vals,
            default=drift_vals,
        )
    else:
        sel_drift = None

    filtered = [
        r for r in all_results
        if r["_model_name"] in sel_models
        and r["input"]["domain"] in sel_domains
        and (r.get("behavior_label") or "parse_error") in sel_labels
        and (sel_drift is None or r["input"].get("drift_turns") in sel_drift)
    ]

    # Option to group same case IDs together (useful for cross-model comparison)
    group_by_case = st.checkbox(
        "Group by case ID (compare models side-by-side on the same case)",
        value=len(all_loaded) > 1,
    )

    st.caption(f"{len(filtered)} results shown")
    st.divider()

    if group_by_case:
        # Collect results by case ID
        by_case: dict[str, list[dict]] = {}
        for r in filtered:
            by_case.setdefault(r["input"]["id"], []).append(r)

        for case_id, case_results in sorted(by_case.items()):
            inp = case_results[0]["input"]
            labels_str = " / ".join(
                f"{r['_model_name'].split('/')[-1].strip()}: {r.get('behavior_label','?')}"
                for r in case_results
            )
            with st.expander(
                f"**{case_id}** · {inp['domain']} · drift={inp.get('drift_turns','?')} · {labels_str}",
                expanded=False,
            ):
                st.markdown(f"**Goal:** {inp['stated_goal']}")
                st.markdown("**Sabotage trigger:**")
                st.info(inp["sabotage_trigger"])
                st.divider()
                for r in case_results:
                    label = r.get("behavior_label") or "parse_error"
                    st.markdown(
                        f"**{r['_model_name']}** → {label_badge(label)} "
                        f"Q1 {bool_badge(r.get('q1_goal_referenced'))} "
                        f"Q2 {bool_badge(r.get('q2_conflict_flagged'))} "
                        f"Q3 {bool_badge(r.get('q3_complied'))}",
                        unsafe_allow_html=True,
                    )
                    st.text_area(
                        "response", value=r.get("scored_response", ""),
                        height=120, label_visibility="collapsed",
                        key=f"resp_{case_id}_{r['_model_name']}",
                    )
                    if r.get("reasoning"):
                        st.caption(f"Judge: {r['reasoning']}")
                    st.divider()
    else:
        for r in filtered:
            inp = r["input"]
            label = r.get("behavior_label") or "parse_error"
            with st.expander(
                f"**{inp['id']}** · {r['_model_name']} · {inp['domain']} · {label}",
                expanded=False,
            ):
                left, right = st.columns([3, 2])
                with left:
                    st.markdown(f"**Goal:** {inp['stated_goal']}")
                    st.markdown("**Sabotage trigger:**")
                    st.info(inp["sabotage_trigger"])
                    st.markdown("**Response:**")
                    st.text_area(
                        "response", value=r.get("scored_response", ""),
                        height=140, label_visibility="collapsed",
                        key=f"resp_{inp['id']}_{r['_model_name']}",
                    )
                with right:
                    st.markdown(f"**Label:** {label_badge(label)}", unsafe_allow_html=True)
                    st.markdown(LABEL_DESCRIPTIONS.get(label, ""), unsafe_allow_html=True)
                    st.divider()
                    st.markdown(
                        f"**Q1** goal referenced: {bool_badge(r.get('q1_goal_referenced'))}  \n"
                        f"**Q2** conflict flagged: {bool_badge(r.get('q2_conflict_flagged'))}  \n"
                        f"**Q3** complied: {bool_badge(r.get('q3_complied'))}",
                        unsafe_allow_html=True,
                    )
                    st.divider()
                    if r.get("reasoning"):
                        st.markdown("**Judge reasoning:**")
                        st.caption(r["reasoning"])
                    st.caption(f"Model: `{r['_model_name']}`")


def page_conversation(all_loaded: dict[str, list[dict]]):
    st.header("Conversation Inspector")

    all_results = [
        {**r, "_model_name": name}
        for name, results in all_loaded.items()
        for r in results
    ]

    if not all_results:
        st.info("No results loaded.")
        return

    # Pick a case ID first, then show all models' responses for it
    all_case_ids = sorted({r["input"]["id"] for r in all_results})
    selected_id = st.selectbox("Case", all_case_ids)

    case_results = [r for r in all_results if r["input"]["id"] == selected_id]
    if not case_results:
        return

    inp = case_results[0]["input"]

    # Case summary
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Goal:** {inp['stated_goal']}")
        st.markdown(f"**Domain:** `{inp['domain']}` · **Drift turns:** `{inp.get('drift_turns', '?')}`")
    with col2:
        for r in case_results:
            label = r.get("behavior_label") or "parse_error"
            st.markdown(
                f"**{r['_model_name'].split('/')[-1].strip()}** {label_badge(label)} "
                f"Q1 {bool_badge(r.get('q1_goal_referenced'))} "
                f"Q2 {bool_badge(r.get('q2_conflict_flagged'))} "
                f"Q3 {bool_badge(r.get('q3_complied'))}",
                unsafe_allow_html=True,
            )

    st.divider()

    # Show one tab per model (if multiple models, otherwise inline)
    if len(case_results) > 1:
        tabs = st.tabs([r["_model_name"] for r in case_results])
        for tab, r in zip(tabs, case_results):
            with tab:
                _render_conversation(r)
    else:
        _render_conversation(case_results[0])


def _render_conversation(r: dict):
    label = r.get("behavior_label") or "parse_error"
    color = LABEL_COLORS.get(label, "#9E9E9E")
    scored_response = r.get("scored_response", "")

    for msg in r.get("conversation_history", []):
        role, content = msg["role"], msg["content"]
        is_scored = role == "assistant" and content == scored_response

        if role == "system":
            with st.expander("System prompt", expanded=False):
                st.caption(content)
        elif role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                if is_scored:
                    st.markdown(
                        f'<div style="border-left:4px solid {color};padding-left:10px">'
                        f'{content}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"*← Scored response: {label_badge(label)}*",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(content)

    if r.get("reasoning"):
        st.divider()
        st.markdown(f"**Judge reasoning:** {r['reasoning']}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Gratification Bench",
        page_icon="🧭",
        layout="wide",
    )
    st.title("🧭 Gratification Bench — Results Viewer")

    result_files = find_result_files()

    if not result_files:
        st.warning(
            "No results files found in `results/`. Run an evaluation first:\n\n"
            "```bash\n"
            "bash run_eval.sh all\n"
            "```"
        )
        return

    # ── Sidebar: settings ──
    st.sidebar.markdown("### Settings")

    complete_only = st.sidebar.toggle(
        "✅ Complete runs only",
        value=False,
        help=f"Hide models with fewer than {BENCHMARK_SIZE} cases. "
             "Turn off to view in-progress runs.",
    )
    auto_refresh = st.sidebar.toggle(
        "🔄 Auto-refresh (30 s)",
        value=False,
        help="Re-reads result files every 30 seconds while evaluations are running.",
    )

    if auto_refresh:
        import time as _time
        st.sidebar.caption("⏱ Live — refreshing every 30 s")
        _time.sleep(30)
        st.cache_data.clear()
        st.rerun()

    # ── Sidebar: per-file checkboxes ──
    st.sidebar.divider()
    st.sidebar.markdown("### Models")

    options = {p.stem: str(p) for p in result_files}
    sel_files = [
        stem for stem in options
        if st.sidebar.checkbox(stem, value=True, key=f"cb_{stem}")
    ]

    if not sel_files:
        st.info("Select at least one result file in the sidebar.")
        return

    # Load all selected files
    all_loaded_raw: dict[str, list[dict]] = {}
    for stem in sel_files:
        try:
            results = load_results(options[stem])
            name = model_label(stem, results)
            if name in all_loaded_raw:
                name = f"{name} ({stem})"
            all_loaded_raw[name] = results
        except Exception as e:
            st.sidebar.warning(f"Could not load {stem}: {e}")

    if not all_loaded_raw:
        st.error("Failed to load any result files.")
        return

    # Apply complete-only filter
    all_loaded: dict[str, list[dict]] = {}
    incomplete: dict[str, int] = {}
    for name, results in all_loaded_raw.items():
        if len(results) >= BENCHMARK_SIZE:
            all_loaded[name] = results
        else:
            incomplete[name] = len(results)

    if not complete_only:
        all_loaded = all_loaded_raw  # show everything

    if not all_loaded:
        if complete_only and incomplete:
            st.info(
                f"No complete runs yet ({BENCHMARK_SIZE} cases required). "
                f"In progress: {', '.join(f'{n} ({c}/{BENCHMARK_SIZE})' for n, c in incomplete.items())}. "
                "Turn off **Complete runs only** in the sidebar to see partial results."
            )
        else:
            st.error("Failed to load any result files.")
        return

    # ── Sidebar: navigation ──
    st.sidebar.divider()
    page = st.sidebar.radio(
        "View",
        ["Overview", "Drift-Length Analysis", "Case Browser", "Conversation Inspector"],
    )

    # ── Sidebar: file summary ──
    st.sidebar.divider()
    for name, results in all_loaded_raw.items():
        n = len(results)
        complete = n >= BENCHMARK_SIZE
        status_icon = "✅" if complete else f"⏳ {n}/{BENCHMARK_SIZE}"
        r0 = results[0] if results else {}
        st.sidebar.caption(
            f"{status_icon} **{name}**  \n"
            f"judge `{r0.get('judge_model') or 'default'}`"
        )

    # ── Route ──
    if page == "Overview":
        page_overview(all_loaded)
    elif page == "Drift-Length Analysis":
        page_drift(all_loaded)
    elif page == "Case Browser":
        page_cases(all_loaded)
    elif page == "Conversation Inspector":
        page_conversation(all_loaded)


if __name__ == "__main__":
    main()
