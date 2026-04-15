"""Prosocial Bench — Results Viewer

Streamlit app for exploring and comparing benchmark results stored in .jsonl files.

Run from the prosocial-bench directory:
    streamlit run analysis/app.py

Or point at a specific results file:
    streamlit run analysis/app.py -- --results results/my_run.jsonl
"""

import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import plotly.express as px
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "results"

LABEL_ORDER = [
    "flagged_declined",
    "flagged_complied",
    "mentioned_complied",
    "silent_compliance",
    "silent_refusal",
    "parse_error",
]

LABEL_COLORS = {
    "flagged_declined":   "#2196F3",
    "flagged_complied":   "#4CAF50",
    "mentioned_complied": "#FFC107",
    "silent_compliance":  "#F44336",
    "silent_refusal":     "#9E9E9E",
    "parse_error":        "#E91E63",
}

# Matplotlib-friendly versions (hex works fine)
MPL_COLORS = LABEL_COLORS

LABEL_DESCRIPTIONS = {
    "flagged_declined":   "Flagged the conflict **and** declined to help",
    "flagged_complied":   "Flagged the conflict but still fulfilled the request",
    "mentioned_complied": "Mentioned the goal in passing, then complied without flagging",
    "silent_compliance":  "Complied with no mention of the goal at all",
    "silent_refusal":     "Declined without referencing the goal",
    "parse_error":        "Judge output could not be parsed",
}

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_results(path: str) -> list[dict]:
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def find_result_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        [p for p in RESULTS_DIR.glob("*.jsonl") if p.stat().st_size > 0],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def model_label(results: list[dict]) -> str:
    """Short display name for a set of results."""
    if not results:
        return "unknown"
    r = results[0]
    provider = r.get("evaluatee_provider", "?")
    model = r.get("evaluatee_model") or "default"
    # Shorten long OpenRouter model names
    model = model.split("/")[-1].replace(":free", "")
    return f"{provider} / {model}"


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


# ── Matplotlib figures (for export / report) ───────────────────────────────────

def mpl_label_bar(all_results: dict[str, list[dict]], title: str = "") -> plt.Figure:
    """Grouped bar chart comparing label distributions across models."""
    model_names = list(all_results.keys())
    labels = [l for l in LABEL_ORDER if any(
        (r.get("behavior_label") or "parse_error") == l
        for results in all_results.values()
        for r in results
    )]

    x = np.arange(len(model_names))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 2.5), 5))

    for i, label in enumerate(labels):
        counts = [
            sum(1 for r in all_results[m] if (r.get("behavior_label") or "parse_error") == label)
            for m in model_names
        ]
        offset = (i - len(labels) / 2 + 0.5) * width
        bars = ax.bar(x + offset, counts, width * 0.9, label=label,
                      color=MPL_COLORS.get(label, "#ccc"), edgecolor="white", linewidth=0.5)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        str(count), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Cases")
    ax.set_title(title or "Behavior Label Distribution by Model")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.tight_layout()
    return fig


def mpl_stacked_domain(results: list[dict], model_name: str = "") -> plt.Figure:
    """Stacked bar chart of label distribution by domain for a single model."""
    domain_counts: dict[str, dict[str, int]] = {}
    for r in results:
        domain = r["input"]["domain"]
        label = r.get("behavior_label") or "parse_error"
        domain_counts.setdefault(domain, {l: 0 for l in LABEL_ORDER})
        domain_counts[domain][label] += 1

    domains = sorted(domain_counts)
    labels = [l for l in LABEL_ORDER if any(domain_counts[d].get(l, 0) > 0 for d in domains)]

    fig, ax = plt.subplots(figsize=(max(7, len(domains) * 1.4), 5))
    bottoms = np.zeros(len(domains))

    for label in labels:
        vals = np.array([domain_counts[d].get(label, 0) for d in domains])
        ax.bar(domains, vals, bottom=bottoms,
               label=label, color=MPL_COLORS.get(label, "#ccc"),
               edgecolor="white", linewidth=0.5)
        bottoms += vals

    ax.set_ylabel("Cases")
    ax.set_title(f"Label by Domain — {model_name}" if model_name else "Label by Domain")
    ax.set_xticklabels(domains, rotation=20, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def mpl_q_flags(all_results: dict[str, list[dict]]) -> plt.Figure:
    """Grouped bar chart of Q1/Q2/Q3 true-rates across models."""
    model_names = list(all_results.keys())
    questions = ["Q1 goal referenced", "Q2 conflict flagged", "Q3 complied"]
    keys = ["q1_goal_referenced", "q2_conflict_flagged", "q3_complied"]
    colors = ["#7986CB", "#4DB6AC", "#FFB74D"]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 2), 4.5))

    for i, (q, key, color) in enumerate(zip(questions, keys, colors)):
        rates = []
        for m in model_names:
            vals = [r.get(key) for r in all_results[m] if r.get(key) is not None]
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


def fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ── Pages ──────────────────────────────────────────────────────────────────────

def page_overview(results: list[dict], model_name: str):
    st.header("Overview")

    total = len(results)
    counts = label_counts(results)

    cols = st.columns(len(LABEL_ORDER))
    for col, label in zip(cols, LABEL_ORDER):
        count = counts.get(label, 0)
        col.metric(
            label.replace("_", " "),
            count,
            f"{count/total:.0%}" if total else "—",
        )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Label distribution")
        labels_present = [l for l in LABEL_ORDER if counts.get(l, 0) > 0]
        fig = px.pie(
            names=labels_present,
            values=[counts[l] for l in labels_present],
            color=labels_present,
            color_discrete_map=LABEL_COLORS,
            hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("By domain")
        domain_label: dict[str, dict] = {}
        for r in results:
            d = r["input"]["domain"]
            lbl = r.get("behavior_label") or "parse_error"
            domain_label.setdefault(d, {})
            domain_label[d][lbl] = domain_label[d].get(lbl, 0) + 1

        rows = [
            {"domain": domain, "label": lbl, "count": cnt}
            for domain, counts_d in sorted(domain_label.items())
            for lbl, cnt in counts_d.items()
        ]
        if rows:
            fig2 = px.bar(
                rows, x="domain", y="count", color="label",
                color_discrete_map=LABEL_COLORS,
                category_orders={"label": LABEL_ORDER},
                barmode="stack",
            )
            fig2.update_layout(
                xaxis_title=None, yaxis_title="Cases",
                legend_title="Label", margin=dict(t=0, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Judge signal rates")
    q_cols = st.columns(3)
    for col, key, label in zip(
        q_cols,
        ["q1_goal_referenced", "q2_conflict_flagged", "q3_complied"],
        ["Q1 — goal referenced", "Q2 — conflict flagged", "Q3 — complied"],
    ):
        vals = [r.get(key) for r in results if r.get(key) is not None]
        rate = sum(vals) / len(vals) if vals else 0
        col.metric(label, f"{rate:.0%}")

    # Matplotlib export
    st.divider()
    st.subheader("Export for report")
    st.caption("Download publication-quality figures for your final report.")

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        fig_dom = mpl_stacked_domain(results, model_name)
        st.pyplot(fig_dom)
        st.download_button(
            "⬇ Download (PNG)",
            data=fig_to_png(fig_dom),
            file_name=f"prosocial_by_domain_{model_name.replace(' ', '_')}.png",
            mime="image/png",
            key="dl_domain",
        )
        plt.close(fig_dom)


def page_compare(all_loaded: dict[str, list[dict]]):
    st.header("Model Comparison")

    if len(all_loaded) < 2:
        st.info(
            "Load at least two result files in the sidebar to compare models. "
            "Use the **Compare: select files** multiselect above."
        )
        return

    # ── Interactive comparison charts ──
    st.subheader("Label distributions")

    rows = []
    for name, results in all_loaded.items():
        for r in results:
            rows.append({
                "model": name,
                "label": r.get("behavior_label") or "parse_error",
            })

    fig = px.histogram(
        rows, x="model", color="label",
        color_discrete_map=LABEL_COLORS,
        category_orders={"label": LABEL_ORDER},
        barmode="stack",
        labels={"model": "Model", "count": "Cases"},
    )
    fig.update_layout(yaxis_title="Cases", xaxis_title=None, legend_title="Label")
    st.plotly_chart(fig, use_container_width=True)

    # ── Q-flag rates ──
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

    fig2 = px.bar(
        signal_rows, x="model", y="rate", color="signal",
        barmode="group",
        labels={"rate": "% Yes", "model": "Model"},
        color_discrete_sequence=["#7986CB", "#4DB6AC", "#FFB74D"],
    )
    fig2.update_layout(yaxis_title="% of cases (Yes)", xaxis_title=None,
                       legend_title="Signal", yaxis_range=[0, 110])
    st.plotly_chart(fig2, use_container_width=True)

    # ── Per-domain breakdown (if multiple domains present) ──
    all_domains = sorted({r["input"]["domain"] for rs in all_loaded.values() for r in rs})
    if len(all_domains) > 1:
        st.subheader("Label by domain, per model")
        sel_domain = st.selectbox("Domain", ["All"] + all_domains)

        domain_rows = []
        for name, results in all_loaded.items():
            filtered = results if sel_domain == "All" else [
                r for r in results if r["input"]["domain"] == sel_domain
            ]
            for r in filtered:
                domain_rows.append({
                    "model": name,
                    "label": r.get("behavior_label") or "parse_error",
                })

        fig3 = px.histogram(
            domain_rows, x="model", color="label",
            color_discrete_map=LABEL_COLORS,
            category_orders={"label": LABEL_ORDER},
            barmode="stack",
        )
        fig3.update_layout(yaxis_title="Cases", xaxis_title=None, legend_title="Label")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Matplotlib export ──
    st.divider()
    st.subheader("Export for report")
    st.caption("Matplotlib figures at 150 dpi — ready to drop into your paper.")

    mcol1, mcol2 = st.columns(2)

    with mcol1:
        fig_bar = mpl_label_bar(all_loaded)
        st.pyplot(fig_bar)
        st.download_button(
            "⬇ Download label comparison (PNG)",
            data=fig_to_png(fig_bar),
            file_name="prosocial_label_comparison.png",
            mime="image/png",
            key="dl_compare_bar",
        )
        plt.close(fig_bar)

    with mcol2:
        fig_q = mpl_q_flags(all_loaded)
        st.pyplot(fig_q)
        st.download_button(
            "⬇ Download signal rates (PNG)",
            data=fig_to_png(fig_q),
            file_name="prosocial_signal_rates.png",
            mime="image/png",
            key="dl_compare_q",
        )
        plt.close(fig_q)

    # ── Side-by-side stats table ──
    st.subheader("Summary table")
    header = ["Model", "N"] + [l.replace("_", " ") for l in LABEL_ORDER] + [
        "Q1 %", "Q2 %", "Q3 %"
    ]
    rows_table = []
    for name, results in all_loaded.items():
        counts = label_counts(results)
        n = len(results)
        q_rates = []
        for key in ["q1_goal_referenced", "q2_conflict_flagged", "q3_complied"]:
            vals = [r.get(key) for r in results if r.get(key) is not None]
            q_rates.append(f"{sum(vals)/len(vals):.0%}" if vals else "—")
        rows_table.append(
            [name, n] + [counts.get(l, 0) for l in LABEL_ORDER] + q_rates
        )

    st.dataframe(
        [dict(zip(header, row)) for row in rows_table],
        use_container_width=True,
        hide_index=True,
    )


def page_cases(results: list[dict]):
    st.header("Case Browser")

    col1, col2, col3 = st.columns(3)
    domains = sorted({r["input"]["domain"] for r in results})
    sel_domains = col1.multiselect("Domain", domains, default=domains)
    labels_present = sorted({r.get("behavior_label") or "parse_error" for r in results})
    sel_labels = col2.multiselect("Label", labels_present, default=labels_present)
    sabotage_types = sorted({r["input"].get("sabotage_type", "unknown") for r in results})
    sel_types = col3.multiselect("Sabotage type", sabotage_types, default=sabotage_types)

    filtered = [
        r for r in results
        if r["input"]["domain"] in sel_domains
        and (r.get("behavior_label") or "parse_error") in sel_labels
        and r["input"].get("sabotage_type", "unknown") in sel_types
    ]

    st.caption(f"{len(filtered)} of {len(results)} cases shown")
    st.divider()

    for r in filtered:
        inp = r["input"]
        label = r.get("behavior_label") or "parse_error"

        with st.expander(
            f"**{inp['id']}** · {inp['domain']} · "
            f"{inp.get('sabotage_type','?')} · {label}",
            expanded=False,
        ):
            left, right = st.columns([3, 2])
            with left:
                st.markdown(f"**Stated goal:** {inp['stated_goal']}")
                if inp.get("user_values"):
                    st.markdown(f"**Values:** {', '.join(inp['user_values'])}")
                st.markdown("**Sabotage trigger:**")
                st.info(inp["sabotage_trigger"])
                st.markdown("**Model response:**")
                st.text_area(
                    "response", value=r.get("scored_response", ""),
                    height=160, label_visibility="collapsed",
                    key=f"resp_{inp['id']}_{r.get('evaluatee_provider','')}",
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
                st.divider()
                st.caption(
                    f"Evaluatee: `{r.get('evaluatee_provider')}/{r.get('evaluatee_model') or 'default'}`  \n"
                    f"Judge: `{r.get('judge_provider')}/{r.get('judge_model') or 'default'}`"
                )


def page_conversation(results: list[dict]):
    st.header("Conversation Inspector")
    case_ids = [r["input"]["id"] for r in results]
    selected_id = st.selectbox("Select case", case_ids)
    r = next(r for r in results if r["input"]["id"] == selected_id)

    inp = r["input"]
    label = r.get("behavior_label") or "parse_error"
    color = LABEL_COLORS.get(label, "#9E9E9E")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Goal:** {inp['stated_goal']}")
        st.markdown(f"**Domain:** `{inp['domain']}` · **Sabotage type:** `{inp.get('sabotage_type')}`")
    with col2:
        st.markdown(f"**Label:** {label_badge(label)}", unsafe_allow_html=True)
        st.markdown(
            f"Q1 {bool_badge(r.get('q1_goal_referenced'))} "
            f"Q2 {bool_badge(r.get('q2_conflict_flagged'))} "
            f"Q3 {bool_badge(r.get('q3_complied'))}",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Full conversation")
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
        page_title="Prosocial Bench Results",
        page_icon="🧭",
        layout="wide",
    )
    st.title("🧭 Prosocial Bench — Results Viewer")

    result_files = find_result_files()

    cli_path = None
    if "--results" in sys.argv:
        idx = sys.argv.index("--results")
        if idx + 1 < len(sys.argv):
            cli_path = sys.argv[idx + 1]

    if not result_files and not cli_path:
        st.warning(
            "No results files found in `results/`. Run an evaluation first:\n\n"
            "```bash\n"
            "python -m prosocialbench --provider openrouter "
            "--judge-provider gemini --limit 5 --output results/test_run.jsonl\n"
            "```"
        )
        return

    # ── Sidebar: single file (Overview/Cases/Conversation) ──
    st.sidebar.markdown("### Single model view")
    options = {p.name: str(p) for p in result_files}
    if cli_path:
        primary_path = cli_path
        primary_name = Path(cli_path).stem
    else:
        primary_name = st.sidebar.selectbox("Results file", list(options.keys()), index=0)
        primary_path = options[primary_name]

    try:
        primary_results = load_results(primary_path)
        primary_model = model_label(primary_results)
    except Exception as e:
        st.error(f"Failed to load {primary_path}: {e}")
        return

    # ── Sidebar: multi-file comparison ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Compare models")
    compare_names = st.sidebar.multiselect(
        "Select files to compare",
        list(options.keys()),
        default=list(options.keys())[:2] if len(options) >= 2 else list(options.keys()),
        help="Select two or more result files to compare on the Compare page.",
    )
    all_loaded: dict[str, list[dict]] = {}
    for name in compare_names:
        try:
            results = load_results(options[name])
            display = model_label(results) + f" ({name})"
            all_loaded[display] = results
        except Exception:
            pass

    # ── Sidebar: run info ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Run info**")
    r0 = primary_results[0] if primary_results else {}
    st.sidebar.caption(
        f"Cases: **{len(primary_results)}**  \n"
        f"Evaluatee: `{r0.get('evaluatee_provider')}/{r0.get('evaluatee_model') or 'default'}`  \n"
        f"Judge: `{r0.get('judge_provider')}/{r0.get('judge_model') or 'default'}`  \n"
        f"System prompt: `{r0.get('system_prompt_variant')}`"
    )

    # ── Navigation ──
    page = st.sidebar.radio(
        "View",
        ["Overview", "Compare Models", "Case Browser", "Conversation Inspector"],
    )

    if page == "Overview":
        page_overview(primary_results, primary_model)
    elif page == "Compare Models":
        page_compare(all_loaded)
    elif page == "Case Browser":
        page_cases(primary_results)
    elif page == "Conversation Inspector":
        page_conversation(primary_results)


if __name__ == "__main__":
    main()
