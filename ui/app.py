import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chat Engine", layout="wide")
st.title("RAG Chat Engine")

# ---- Session state for metrics history ----
if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None


@st.cache_data(ttl=5)
def fetch_collections():
    return requests.get(f"{API_URL}/collections").json()


# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("Collections")
    collections = fetch_collections()
    names = {c["name"]: c["id"] for c in collections}

    new_name = st.text_input("New collection name")
    if st.button("Create collection") and new_name:
        requests.post(f"{API_URL}/collections", json={"name": new_name})
        st.cache_data.clear()
        st.rerun()

    if not names:
        st.info("Create a collection to get started.")
        st.stop()

    selected_name = st.selectbox("Active collection", list(names.keys()))
    collection_id = names[selected_name]

    st.divider()
    uploaded = st.file_uploader("Upload documents", type=["pdf", "html", "txt", "md"], accept_multiple_files=True)
    if uploaded and st.button("Ingest"):
        for f in uploaded:
            with st.spinner(f"Ingesting {f.name}..."):
                resp = requests.post(
                    f"{API_URL}/collections/{collection_id}/documents",
                    files={"file": (f.name, f.getvalue())},
                )
            if resp.ok:
                st.success(f"Indexed {f.name}: {resp.json()['chunk_count']} chunks")
            else:
                st.error(resp.json().get("detail", "Ingestion failed"))

    st.divider()
    mode = st.radio("Query mode", ["hybrid", "multi_query", "multi_source", "agentic"])
    verify = st.checkbox("Verify citations", value=True)


# ===================== TABS =====================
tab_chat, tab_metrics = st.tabs(["💬 Chat", "📊 Metrics"])

# ===================== CHAT TAB =====================
with tab_chat:
    question = st.text_input("Ask a question:", key="chat_input")
    if question:
        with st.spinner("Thinking..."):
            resp = requests.post(
                f"{API_URL}/collections/{collection_id}/chat",
                json={"question": question, "mode": mode, "verify_citations": verify},
            )
        if resp.ok:
            data = resp.json()
            st.subheader("Answer")
            st.write(data["answer"])
            st.subheader("Sources")
            for s in data["sources"]:
                st.caption(s)

            # Store metrics
            metrics = data.get("metrics", {})
            if metrics:
                metrics["_question"] = question
                metrics["_query_index"] = len(st.session_state.metrics_history) + 1
                st.session_state.last_metrics = metrics
                st.session_state.metrics_history.append(metrics)
        else:
            st.error(resp.json().get("detail", "Query failed"))

# ===================== METRICS TAB =====================
with tab_metrics:
    m = st.session_state.last_metrics
    history = st.session_state.metrics_history

    if m is None:
        st.info("Ask a question in the Chat tab to see metrics here.")
    else:
        st.markdown(f"**Last query:** *{m.get('_question', '')}*  ·  **Mode:** `{m.get('query_mode', 'N/A')}`")
        st.divider()

        # ---- KPI Cards ----
        st.subheader("⚡ Key Performance Indicators")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        total_sec = m.get("total_sec")
        kpi1.metric("Total Latency", f"{total_sec:.2f}s" if total_sec else "N/A")

        citation_acc = m.get("citation_accuracy")
        kpi2.metric(
            "Citation Accuracy",
            f"{citation_acc:.0%}" if citation_acc is not None else "N/A",
        )

        kpi3.metric("Chunks Retrieved", m.get("chunks_retrieved", "N/A"))

        halluc = m.get("hallucination_rate")
        kpi4.metric(
            "Hallucination Rate",
            f"{halluc:.0%}" if halluc is not None else "N/A",
        )

        st.divider()

        # ---- Latency Breakdown ----
        col_latency, col_retrieval = st.columns(2)

        with col_latency:
            st.subheader("⏱ Latency Breakdown")
            is_agentic = m.get("query_mode") == "agentic"

            if is_agentic:
                latency_data = {
                    "Planner": m.get("planner_sec", 0),
                    "Retriever": m.get("retriever_sec", 0),
                    "Critic": m.get("critic_sec", 0),
                    "Answer Gen": m.get("answer_generation_sec", 0),
                    "Citation Check": m.get("citation_verification_sec", 0),
                }
            else:
                latency_data = {
                    "Retrieval": m.get("retrieval_sec", 0),
                    "Generation": m.get("generation_sec", 0),
                    "Citation Check": m.get("citation_verification_sec", 0),
                }

            latency_df = pd.DataFrame(
                [{"Phase": k, "Seconds": v} for k, v in latency_data.items() if v]
            )
            if not latency_df.empty:
                chart = (
                    alt.Chart(latency_df)
                    .mark_bar(cornerRadiusEnd=6)
                    .encode(
                        x=alt.X("Seconds:Q", title="Time (seconds)"),
                        y=alt.Y("Phase:N", sort="-x", title=""),
                        color=alt.Color(
                            "Phase:N",
                            scale=alt.Scale(scheme="viridis"),
                            legend=None,
                        ),
                        tooltip=["Phase", alt.Tooltip("Seconds:Q", format=".3f")],
                    )
                    .properties(height=220)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("No latency data available.")

        # ---- Retrieval Quality ----
        with col_retrieval:
            st.subheader("🔍 Retrieval Quality")
            rq1, rq2 = st.columns(2)
            rq1.metric("Avg Reranker Score", f"{m.get('avg_reranker_score', 0):.4f}")
            rq2.metric("Dense / Sparse Overlap", f"{m.get('dense_sparse_overlap', 0):.0%}")

            scores = m.get("reranker_scores", [])
            if scores:
                score_df = pd.DataFrame(
                    [{"Chunk": f"Chunk {i+1}", "Score": s} for i, s in enumerate(scores)]
                )
                chart = (
                    alt.Chart(score_df)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("Chunk:N", sort=None, title=""),
                        y=alt.Y("Score:Q", title="Relevance Score"),
                        color=alt.Color(
                            "Score:Q",
                            scale=alt.Scale(scheme="goldgreen"),
                            legend=None,
                        ),
                        tooltip=["Chunk", alt.Tooltip("Score:Q", format=".4f")],
                    )
                    .properties(height=180)
                )
                st.altair_chart(chart, use_container_width=True)

            rq3, rq4 = st.columns(2)
            rq3.metric("Unique Sources", m.get("unique_sources", "N/A"))
            rq4.metric(
                "Score Range",
                f"{m.get('min_reranker_score', 0):.3f} – {m.get('max_reranker_score', 0):.3f}",
            )

        st.divider()

        # ---- Citation Verification & Answer Stats ----
        col_citation, col_answer = st.columns(2)

        with col_citation:
            st.subheader("✅ Citation Verification")
            extracted = m.get("citations_extracted", 0)
            verified = m.get("citations_verified", 0)
            failed = max(0, extracted - verified)

            if extracted > 0:
                cite_df = pd.DataFrame(
                    [
                        {"Status": "Verified", "Count": verified},
                        {"Status": "Unverified", "Count": failed},
                    ]
                )
                chart = (
                    alt.Chart(cite_df)
                    .mark_arc(innerRadius=45, cornerRadius=4)
                    .encode(
                        theta=alt.Theta("Count:Q"),
                        color=alt.Color(
                            "Status:N",
                            scale=alt.Scale(
                                domain=["Verified", "Unverified"],
                                range=["#22c55e", "#ef4444"],
                            ),
                        ),
                        tooltip=["Status", "Count"],
                    )
                    .properties(height=200)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("No citations to verify.")

            c1, c2 = st.columns(2)
            c1.metric("Extracted", extracted)
            c2.metric("Verified", verified)

        with col_answer:
            st.subheader("📝 Answer Stats")
            st.metric("Answer Length", f"{m.get('answer_length_words', 0)} words")
            st.metric("Answer Characters", f"{m.get('answer_length_chars', 0):,}")
            if m.get("query_mode") in ("multi_query", "agentic"):
                st.metric("Sub-queries Generated", m.get("planner_queries_generated", "N/A"))

        # ---- Agentic Reasoning (conditional) ----
        if m.get("query_mode") == "agentic":
            st.divider()
            st.subheader("🤖 Agentic Reasoning")
            ag1, ag2, ag3, ag4 = st.columns(4)
            ag1.metric("Critic Confidence", f"{m.get('critic_confidence', 0):.2f}")
            ag2.metric("Retry Count", m.get("retry_count", 0))
            ag3.metric("Queries Generated", m.get("planner_queries_generated", "N/A"))
            ag4.metric(
                "Confidence Δ",
                f"{m.get('confidence_improvement', 0):+.4f}",
            )

            conf_history = m.get("confidence_history", [])
            if len(conf_history) > 1:
                conf_df = pd.DataFrame(
                    [{"Attempt": i + 1, "Confidence": c} for i, c in enumerate(conf_history)]
                )
                chart = (
                    alt.Chart(conf_df)
                    .mark_line(point=True, strokeWidth=3)
                    .encode(
                        x=alt.X("Attempt:O", title="Attempt"),
                        y=alt.Y("Confidence:Q", scale=alt.Scale(domain=[0, 1]), title="Critic Confidence"),
                        tooltip=["Attempt", alt.Tooltip("Confidence:Q", format=".3f")],
                    )
                    .properties(height=200)
                )
                st.altair_chart(chart, use_container_width=True)

        # ---- Session History ----
        if len(history) > 1:
            st.divider()
            st.subheader("📈 Session History")
            st.caption(f"Tracking {len(history)} queries this session")

            hist_df = pd.DataFrame(history)

            # Latency over time
            if "total_sec" in hist_df.columns:
                latency_hist = hist_df[["_query_index", "total_sec"]].rename(
                    columns={"_query_index": "Query #", "total_sec": "Total Latency (s)"}
                )
                chart = (
                    alt.Chart(latency_hist)
                    .mark_line(point=True, strokeWidth=2, color="#6366f1")
                    .encode(
                        x=alt.X("Query #:O"),
                        y=alt.Y("Total Latency (s):Q"),
                        tooltip=["Query #", alt.Tooltip("Total Latency (s):Q", format=".3f")],
                    )
                    .properties(height=200)
                )
                st.altair_chart(chart, use_container_width=True)

            # Citation accuracy over time
            if "citation_accuracy" in hist_df.columns:
                ca_hist = hist_df[["_query_index", "citation_accuracy"]].dropna()
                if not ca_hist.empty:
                    ca_hist = ca_hist.rename(
                        columns={"_query_index": "Query #", "citation_accuracy": "Citation Accuracy"}
                    )
                    chart = (
                        alt.Chart(ca_hist)
                        .mark_area(opacity=0.3, color="#22c55e", line={"color": "#22c55e", "strokeWidth": 2})
                        .encode(
                            x=alt.X("Query #:O"),
                            y=alt.Y("Citation Accuracy:Q", scale=alt.Scale(domain=[0, 1])),
                            tooltip=["Query #", alt.Tooltip("Citation Accuracy:Q", format=".0%")],
                        )
                        .properties(height=200)
                    )
                    st.altair_chart(chart, use_container_width=True)

            # Detailed history table
            with st.expander("📋 Full Metrics Table"):
                display_cols = [
                    col for col in hist_df.columns
                    if not col.startswith("_") and col not in ("reranker_scores", "confidence_history")
                ]
                st.dataframe(hist_df[display_cols], use_container_width=True, hide_index=True)
