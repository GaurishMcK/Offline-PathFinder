import os
import json
import random
import pandas as pd
import streamlit as st

from data_store import load_index, load_transcript, resolve_data_path, ms_to_mmss
from artifacts import load_artifacts
from classifier import classify_call
from blocks import build_activity_blocks
from evaluator import evaluate_blocks
from reports import build_activity_report, build_agent_coaching_summary
from narrative import generate_narrative

from intent_batch import (
    compute_aht_seconds,
    transcript_to_one_cell,
    discover_intent_tree,
    classify_transcripts_intent_bulk,
    parse_uploaded_intent_tree,
    build_mapping_excel,
)


def main():
    # -------------------------
    # App shell
    # -------------------------
    st.set_page_config(page_title="Pathfinder", layout="wide")

    st.markdown(
        """
        <style>
          .pf-header {
            padding: 18px 20px;
            border-radius: 16px;
            background: linear-gradient(90deg, rgba(37,99,235,0.10), rgba(37,99,235,0.03));
            border: 1px solid rgba(37,99,235,0.15);
            margin-bottom: 14px;
          }
          .pf-title { font-size: 26px; font-weight: 750; margin: 0; }
          .pf-subtitle { margin: 4px 0 0 0; color: rgba(15, 23, 42, 0.70); }
          .pf-card {
            border-radius: 16px;
            border: 1px solid rgba(15,23,42,0.10);
            background: rgba(255,255,255,0.70);
            padding: 14px 14px 6px 14px;
          }
          .pf-chip {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid rgba(15,23,42,0.12);
            background: rgba(15,23,42,0.03);
            font-size: 12px;
            margin-right: 8px;
          }
          /* tighten some default spacing */
          div.block-container { padding-top: 1.4rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    
    st.title("Pathfinder")

    INDEX_PATH = os.getenv("CALL_INDEX_PATH", "data/calls_index.csv")
    try:
        index_df = load_index(INDEX_PATH)
    except Exception as e:
        st.error(f"Failed to load index at {INDEX_PATH}: {e}")
        st.stop()

    tab1, tab2 = st.tabs(["Call Flow QA (single call)", "Intent Mapping (batch)"])

    # ============================================================
    # TAB 1 — Existing single-call QA (unchanged)
    # ============================================================
    with tab1:
        st.markdown("### Call Flow QA")
        with st.sidebar:
            st.markdown("## Controls")
            st.caption("Filter, select a call, then run actions.")
    
            with st.form("qa_filters", border=False):
                call_types = sorted(index_df["call_type"].dropna().unique().tolist())
                call_type_filter = st.selectbox("Call intent", call_types, key="qa_call_type")
    
                cA, cB = st.columns(2)
                emp_filter = cA.text_input("Agent emp_id", "", key="qa_emp", placeholder="e.g., 12345")
                search = cB.text_input("Search", "", key="qa_search", placeholder="conversation/transcript id")
    
                submitted = st.form_submit_button("Apply filters")
    
            fdf = index_df.copy()
            fdf = fdf[fdf["call_type"] == call_type_filter]
    
            if emp_filter.strip():
                try:
                    fdf = fdf[fdf["emp_id"] == int(emp_filter.strip())]
                except ValueError:
                    st.warning("emp_id must be an integer.")
            if search.strip():
                s = search.strip().lower()
                fdf = fdf[
                    fdf["conversation_id"].astype(str).str.lower().str.contains(s)
                    | fdf["transcript_id"].astype(str).str.lower().str.contains(s)
                ]
    
            st.markdown(f"<span class='pf-chip'>Matches: <b>{len(fdf):,}</b></span>", unsafe_allow_html=True)
    
            if len(fdf) == 0:
                st.info("No calls match the current filters.")
                st.stop()
    
            fdf = fdf.copy()
            fdf["label"] = (
                fdf["conversation_id"].astype(str)
                + " | "
                + fdf["transcript_id"].astype(str)
                + " | emp "
                + fdf["emp_id"].astype(str)
                + " | "
                + fdf["call_type"].astype(str)
            )
    
            selected = st.selectbox("Select call", fdf["label"].tolist()[:5000], key="qa_selected")
    
            st.divider()
    
            # Primary workflow: one “Run QA” button, advanced controls tucked away
            run_primary = st.button("Run QA pipeline (Activities → Blocks → Evaluation)", type="primary")
    
            with st.expander("Advanced actions"):
                run_cls = st.button("1) Get activities", key="qa_run_cls")
                run_blocks = st.button("2) Consolidate activities", key="qa_run_blocks")
                run_eval = st.button("3) Evaluate call", key="qa_run_eval")
                run_report = st.button("4) Create reports", key="qa_run_report")
                run_narrative = st.button("5) Generate narrative (LLM)", key="qa_run_narr")
    
            # if primary clicked, trigger the stages in sequence
            if run_primary:
                run_cls, run_blocks, run_eval = True, True, True
                run_report = True
                # keep narrative separate (optional) – or set True if you want
                run_narrative = False

        # Selected row
        row = fdf.loc[fdf["label"] == selected].iloc[0].to_dict()
        metadata = {
            "conversation_id": str(row["conversation_id"]),
            "transcript_id": str(row["transcript_id"]),
            "emp_id": int(row["emp_id"]),
            "call_type": str(row["call_type"]),
        }

        st.markdown(
            f"""
            <div class="pf-card">
              <span class="pf-chip"><b>Conversation</b>: {metadata['conversation_id']}</span>
              <span class="pf-chip"><b>Transcript</b>: {metadata['transcript_id']}</span>
              <span class="pf-chip"><b>Agent</b>: {metadata['emp_id']}</span>
              <span class="pf-chip"><b>Intent</b>: {metadata['call_type']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.write("")

        # Load transcript
        transcript_df = load_transcript(resolve_data_path(INDEX_PATH, str(row["transcript_path"])))
        transcript_df["start_mmss"] = transcript_df["START_TIME_MS"].apply(ms_to_mmss)
        transcript_df["end_mmss"] = transcript_df["END_TIME_MS"].apply(ms_to_mmss)

        st.dataframe(
            transcript_df[["phrase_rank", "speaker", "start_mmss", "end_mmss", "text"]],
            height=520,
            width="stretch",
            hide_index=True,
        )


        sel_enum, sel_ctx, sel_rubric = load_artifacts(metadata["call_type"])

        st.subheader("Metadata")
        # Arrow-safe: force strings so mixed types can't crash pyarrow
        meta_tbl = pd.DataFrame([{"field": k, "value": str(v)} for k, v in metadata.items()])
        st.dataframe(meta_tbl, hide_index=True, width="stretch")

        st.subheader("Call Summary")
        st.write(sel_ctx.get("purpose", ""))
        detailed = sel_ctx.get("detailed_intent_description", "")
        if detailed:
            st.write(detailed)

        guidance = sel_ctx.get("guidance", []) or []
        if guidance:
            st.markdown("**Guidance**")
            for g in guidance:
                st.write(f"- {g}")

        # Stage 1
        if run_cls:
            with st.spinner("Classifying per-segment activities..."):
                cls_df = classify_call(transcript_df, sel_enum, sel_ctx, metadata)
            st.session_state["qa_cls_df"] = cls_df

        if "qa_cls_df" in st.session_state:
            st.subheader("Per-segment activities")
            st.dataframe(st.session_state["qa_cls_df"], height=240)

        # Stage 2
        if run_blocks and "qa_cls_df" in st.session_state:
            blocks_df = build_activity_blocks(transcript_df, st.session_state["qa_cls_df"], sel_enum)
            st.session_state["qa_blocks_df"] = blocks_df

        if "qa_blocks_df" in st.session_state:
            st.subheader("Activity blocks")
            st.dataframe(st.session_state["qa_blocks_df"][["activity", "present", "duration_sec"]], height=260)
            with st.expander("Show exchanges"):
                st.dataframe(st.session_state["qa_blocks_df"][["activity", "exchange"]], height=400)

        # Stage 3
        if run_eval and "qa_blocks_df" in st.session_state:
            with st.spinner("Evaluating activities..."):
                eval_df = evaluate_blocks(st.session_state["qa_blocks_df"], sel_rubric, metadata)
                eval_df["efficacy_score"] = eval_df["efficacy_score"].astype(float).round(1)
                eval_df["efficacy_pct"] = eval_df["efficacy_score"].astype(str) + "%"
            st.session_state["qa_eval_df"] = eval_df

        if "qa_eval_df" in st.session_state:
            st.subheader("Evaluation")
            st.dataframe(
                st.session_state["qa_eval_df"][
                    ["activity", "present", "duration_sec", "efficacy_pct", "detailed_observations", "tactical_feedback"]
                ],
                height=360,
            )

        # Stage 4
        if run_report and "qa_eval_df" in st.session_state:
            st.session_state["qa_activity_report"] = build_activity_report(st.session_state["qa_eval_df"])
            st.session_state["qa_agent_summary"] = build_agent_coaching_summary(st.session_state["qa_eval_df"])

        if "qa_activity_report" in st.session_state:
            st.subheader("Reports")
            st.dataframe(st.session_state["qa_activity_report"], height=240)
            st.text(st.session_state["qa_agent_summary"])

        # Stage 5
        if run_narrative and "qa_eval_df" in st.session_state:
            with st.spinner("Generating narrative..."):
                narrative = generate_narrative(sel_enum, sel_rubric, st.session_state["qa_eval_df"], metadata)
            st.session_state["qa_narrative"] = narrative

        if "qa_narrative" in st.session_state:
            st.subheader("Narrative insights")
            st.json(st.session_state["qa_narrative"])

    # ============================================================
    # TAB 2 — Batch intent mapping
    # ============================================================
    with tab2:
        st.markdown("### Intent Mapping (batch)")

        st.markdown("<div class='pf-card'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([0.42, 0.28, 0.30])

        st.markdown("</div>", unsafe_allow_html=True)

        
        cL, cR = st.columns([0.65, 0.35])

        with cL:
            st.markdown(
                "- **Open-ended:** discover dataset-wide L1/L2 tree and map all transcripts\n"
                "- **Close-ended:** upload final L1/L2 tree and map all transcripts to primary intent\n"
            )

        with cR:
            mode = st.selectbox("Intent mode", ["Open-ended", "Close-ended"], key="batch_mode")
            max_calls = st.number_input("Max transcripts to process", min_value=1, value=50, step=10, key="im_max")
            seed = st.number_input("Sampling seed", min_value=0, value=42, step=1, key="im_seed")
            
        # Select which transcripts to process: use index (all folders)
        df_all = index_df.copy()
        df_all["transcript_abs"] = df_all["transcript_path"].apply(lambda p: resolve_data_path(INDEX_PATH, str(p)))

        # Sample / limit
        random.seed(int(seed))
        df_proc = df_all.sample(n=min(int(max_calls), len(df_all)), random_state=int(seed)) if len(df_all) else df_all

        st.caption(f"Will process {len(df_proc):,} transcripts (from index across folders).")

        uploaded_tree = None
        intent_tree = None

        if mode == "Close-ended":
            up = st.file_uploader(
                "Upload final intent tree (CSV/XLSX/JSON with L1/L2)", type=["csv", "xlsx", "xls", "json"]
            )
            if up is not None:
                uploaded_tree = parse_uploaded_intent_tree(up.getvalue(), up.name)
                st.success(f"Loaded intent tree: {len(uploaded_tree)} L1 categories.")
                with st.expander("Preview uploaded tree"):
                    st.json(uploaded_tree)

        discover_n = st.number_input("Open-ended discovery sample size", min_value=5, value=25, step=5, key="im_disc_n")
        
        run_batch = st.button("Run batch mapping and generate Excel", key="run_batch")

        if run_batch:
            # --------------------------
            # Step A: Determine intent tree
            # --------------------------
            if mode == "Open-ended":
                st.info("Open-ended: discovering L1/L2 intent tree from samples...")
                df_disc = df_proc.sample(n=min(int(discover_n), len(df_proc)), random_state=int(seed)).copy()

                sample_texts = []
                for _, r in df_disc.iterrows():
                    tdf = load_transcript(r["transcript_abs"])
                    sample_texts.append(transcript_to_one_cell(tdf))

                with st.spinner("Discovering intent tree (LLM)..."):
                    intent_tree = discover_intent_tree(sample_texts)

                st.success(f"Discovered intent tree with {len(intent_tree)} L1 categories.")
                st.session_state["open_intent_tree"] = intent_tree
                with st.expander("Discovered intent tree"):
                    st.json(intent_tree)

            else:
                if not uploaded_tree:
                    st.error("Close-ended requires uploading the final L1/L2 intent tree file first.")
                    st.stop()
                intent_tree = uploaded_tree

            # --------------------------
            # Step B: Map all transcripts
            # --------------------------
            st.info("Mapping transcripts to (L1, L2, Sentiment) + AHT ...")
            rows = []
            texts = []
            ahts = []
            
            prog = st.progress(0)
            total = len(df_proc)

            for i, (_, r) in enumerate(df_proc.iterrows(), start=1):
                tdf = load_transcript(r["transcript_abs"])
                text_one_cell = transcript_to_one_cell(tdf)
                aht = compute_aht_seconds(tdf)
            
                texts.append(text_one_cell)
                ahts.append(round(float(aht), 2))
            
                rows.append(
                    {
                        "S No.": i,
                        "Transcript text in one cell": text_one_cell,
                        "Total AHT": round(float(aht), 2),
                        # filled after LLM mapping
                        "L1": "",
                        "L2": "",
                        "Sentiment": "",
                    }
                )
                prog.progress(int(i * 100 / max(total, 1)))

            batch_size = 6  # tune 4–10 depending on transcript length + model context
            results_by_i = {}

            with st.spinner(f"Classifying {total} transcripts in batches of {batch_size}..."):
                for start in range(0, total, batch_size):
                    batch_texts = texts[start : start + batch_size]
                    batch_start_index = start + 1  # 1-based "S No."
            
                    batch_results = classify_transcripts_intent_bulk(
                        transcripts_text=batch_texts,
                        intent_tree=intent_tree,
                        mode=mode.lower(),
                        start_index=batch_start_index,
                    )
            
                    for br in batch_results:
                        results_by_i[int(br["i"])] = br
            
                    prog.progress(int(min(start + batch_size, total) * 100 / max(total, 1)))
            
            # 3) Fill rows with model output
            for row in rows:
                i = int(row["S No."])
                br = results_by_i.get(i, {})
                row["L1"] = br.get("L1", "")
                row["L2"] = br.get("L2", "")
                row["Sentiment"] = br.get("Sentiment", "")
            
            mappings_df = pd.DataFrame(rows)
            st.session_state["batch_mappings_df"] = mappings_df

            # --------------------------
            # Step C: Build Excel
            # --------------------------
            include_freq = (mode == "Close-ended")
            xlsx_bytes = build_mapping_excel(mappings_df, include_frequency_sheet=include_freq)

            fname = "open_ended_intent_mapping.xlsx" if mode == "Open-ended" else "close_ended_intent_mapping.xlsx"
            
            st.markdown("<div class='pf-card'>", unsafe_allow_html=True)
            st.success("Excel generated.")
            st.download_button(
                "Download Excel",
                data=xlsx_bytes,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.session_state["batch_xlsx"] = xlsx_bytes
            st.dataframe(mappings_df[["S No.", "Total AHT", "L1", "L2", "Sentiment"]], height=360)

# Optional: helps local runs like `python pathfinder_v4.py`, but Streamlit won't rely on this.
if __name__ == "__main__":
    main()
