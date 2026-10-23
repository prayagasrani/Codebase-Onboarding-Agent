"""Streamlit UI for the Codebase Onboarding Agent.

Run:  streamlit run src/onboarding_agent/ui/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from onboarding_agent.onboarding.generator import OnboardingGenerator
from onboarding_agent.service import AppService

st.set_page_config(page_title="Codebase Onboarding Agent", page_icon="🧭", layout="wide")


@st.cache_resource
def get_service() -> AppService:
    return AppService()


service = get_service()

st.title("🧭 Codebase Onboarding Agent")

with st.sidebar:
    st.header("Repository")
    if service.index.is_built:
        stats = service.index.stats()
        st.success(f"Indexed: {service.index.manifest.get('repo_root', '')}")
        st.caption(f"{stats['files']} files · {stats['chunks']} chunks · "
                   f"{stats['graph_edges']} graph edges")
    else:
        st.warning("No repository indexed yet.")
    repo_input = st.text_input("Local path or GitHub URL")
    col1, col2 = st.columns(2)
    if col1.button("Index", use_container_width=True) and repo_input:
        with st.spinner("Indexing…"):
            st.json(service.index_repo(repo_input))
        st.rerun()
    if col2.button("Refresh", use_container_width=True, disabled=not service.index.is_built):
        with st.spinner("Incremental re-index…"):
            st.json(service.refresh_index().to_dict())
        st.rerun()

    st.divider()
    st.header("Status")
    st.caption(f"LLM: {'✅ ' + service.llm.model if service.llm else '❌ offline (extractive mode)'}")
    st.caption(f"GitHub: {'✅ ' + service.github.repo if service.github else '❌ not configured'}")
    user_id = st.text_input("User ID (memory)", value="default")

tab_chat, tab_onboard = st.tabs(["💬 Ask", "🚀 Onboard Me"])

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.caption("Sources: " + " · ".join(f"`{c['ref']}`" for c in msg["citations"]))

    if question := st.chat_input("Ask about the codebase, PRs, or your past questions…"):
        if not service.index.is_built:
            st.error("Index a repository first (sidebar).")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"), st.spinner("Thinking…"):
                result = service.ask(question, user_id=user_id)
                st.markdown(result["answer"])
                if result["citations"]:
                    st.caption("Sources: " + " · ".join(f"`{c['ref']}`" for c in result["citations"]))
                support = result["verification"].get("support_rate", 1.0)
                st.caption(
                    f"route: `{result['route']}` · latency: {result['latency_s']}s · "
                    f"citation support: {support:.0%}"
                )
                with st.expander("Evidence"):
                    for e in result["evidence"]:
                        st.markdown(f"**{e['tool']}** `{e.get('input', {})}`")
                        st.code(e.get("result_snippet", "")[:800])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "citations": result["citations"],
            })

with tab_onboard:
    st.caption("Generates a structured walkthrough: overview, architecture, entry points, "
               "components, dependencies, auth/data flow, testing, deployment, current activity.")
    if st.button("Generate onboarding guide", type="primary",
                 disabled=not service.index.is_built):
        with st.spinner("Analyzing repository…"):
            result = OnboardingGenerator(service).generate()
        st.markdown(result["guide"])
        with st.expander("Raw facts"):
            st.json(result["facts"])
