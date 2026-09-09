import streamlit as st
import requests
import pandas as pd
import difflib

API_URL = "http://localhost:8080"

st.set_page_config(page_title="Kivi Phonetic Memory", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
<style>
    .highlight-insert { background-color: #c8e6c9; color: #1b5e20; padding: 2px 4px; border-radius: 4px; font-weight: bold; }
    .highlight-delete { background-color: #ffcdd2; color: #b71c1c; padding: 2px 4px; border-radius: 4px; text-decoration: line-through; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def get_diff_html(a: str, b: str) -> str:
    """Generate HTML highlighting the differences between two strings at word level."""
    matcher = difflib.SequenceMatcher(None, a.split(), b.split())
    out = []
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            out.append(' '.join(a.split()[a0:a1]))
        elif opcode == 'insert':
            out.append(f"<span class='highlight-insert'>{' '.join(b.split()[b0:b1])}</span>")
        elif opcode == 'delete':
            out.append(f"<span class='highlight-delete'>{' '.join(a.split()[a0:a1])}</span>")
        elif opcode == 'replace':
            out.append(f"<span class='highlight-delete'>{' '.join(a.split()[a0:a1])}</span>")
            out.append(f"<span class='highlight-insert'>{' '.join(b.split()[b0:b1])}</span>")
    return " ".join(out)

# --- Sidebar ---
st.sidebar.title("🧠 Kivi Memory Core")
st.sidebar.markdown("Evidence-Gated Phonetic RAG")
st.sidebar.divider()

page = st.sidebar.radio("Navigation", [
    "🎙️ Process ASR", 
    "🎓 Learning Engine", 
    "📚 User Dictionary", 
    "⚙️ Intervention Logs"
])

# --- Main App ---
if page == "🎙️ Process ASR":
    st.title("Process ASR Transcript")
    st.markdown("Test real-time phonetic corrections on raw ASR text.")
    
    asr_input = st.text_area("Raw ASR Input", value="", placeholder="Enter raw ASR output here...", height=150)
    
    if st.button("🚀 Process Transcript", type="primary", use_container_width=True):
        with st.spinner("Processing via LLM Judge (Fail-Fast Timeout: 5s)..."):
            try:
                resp = requests.post(f"{API_URL}/process", json={"asr_text": asr_input})
                if resp.status_code == 200:
                    data = resp.json()
                    original = data["original"]
                    processed = data["processed"]
                    
                    if original == processed:
                        st.info("No interventions made. The transcript was left untouched.")
                    else:
                        st.success("Intervention successful!")
                        
                    st.markdown("### Final Output")
                    st.write(processed)
                    
                else:
                    st.error(f"Backend Error: {resp.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")

elif page == "🎓 Learning Engine":
    st.title("Observation & Learning")
    st.markdown("Inject phonetic memories into the state machine.")
    
    tab1, tab2, tab3 = st.tabs(["Implicit Observation", "Explicit Correction", "Direct Overrides"])
    
    with tab1:
        st.markdown("### Background Character-Level Diffing")
        col1, col2 = st.columns(2)
        with col1:
            obs_asr = st.text_area("Raw ASR", value="", placeholder="Enter raw ASR output...")
        with col2:
            obs_fmt = st.text_area("Corrected Text", value="", placeholder="Enter final corrected text...")
        
        if st.button("Extract & Learn", type="primary"):
            try:
                resp = requests.post(f"{API_URL}/learn/observe", json={"asr_text": obs_asr, "formatted_text": obs_fmt})
                if resp.ok:
                    st.success("Successfully ingested observation.")
                    st.json(resp.json()["learned"])
                else:
                    st.error(resp.text)
            except Exception as e:
                st.error(str(e))
                
    with tab2:
        st.markdown("### Manual Override (Phonetic)")
        corr_canon = st.text_input("Canonical Word", value="", placeholder="e.g. Kivi")
        corr_context = st.text_input("Context Snippet", value="", placeholder="e.g. Kivi is an AI company.")
        corr_asr = st.text_input("ASR Token (optional)", value="", placeholder="e.g. kiwi")
        
        if st.button("Force Inject Correction"):
            try:
                resp = requests.post(f"{API_URL}/learn/correct", json={
                    "canonical_word": corr_canon, 
                    "context_snippet": corr_context, 
                    "asr_token": corr_asr
                })
                if resp.ok:
                    st.success("Successfully injected correction.")
                else:
                    st.error(resp.text)
            except Exception as e:
                st.error(str(e))

    with tab3:
        st.markdown("### Hardcoded Exact Overrides")
        st.markdown("Bypass the LLM and phonetic math entirely. If ASR sees exactly X, swap it to Y.")
        
        col_asr, col_target = st.columns(2)
        with col_asr:
            override_asr = st.text_input("Exact ASR Token", placeholder="e.g. keevee")
        with col_target:
            override_target = st.text_input("Target Word", placeholder="e.g. Kivi")
            
        if st.button("Save Direct Override", type="primary"):
            if override_asr and override_target:
                try:
                    resp = requests.post(f"{API_URL}/overrides", json={"asr_token": override_asr, "canonical_word": override_target})
                    if resp.ok:
                        st.success(f"Saved override: {override_asr} -> {override_target}")
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(str(e))
                    
        st.markdown("---")
        st.markdown("#### Active Overrides")
        try:
            overrides_resp = requests.get(f"{API_URL}/overrides").json().get("overrides", [])
            if overrides_resp:
                st.dataframe(overrides_resp, use_container_width=True)
            else:
                st.info("No direct overrides configured.")
        except Exception:
            st.error("Could not load overrides.")

elif page == "📚 User Dictionary":
    st.title("Phonetic Memory Store")
    
    col_title, col_btn_del, col_btn_ref = st.columns([2.5, 1, 1])
    with col_title:
        st.markdown("Current internal state of the SQLite database.")
    with col_btn_del:
        if st.button("🗑️ Clear Database", use_container_width=True, type="secondary"):
            resp = requests.delete(f"{API_URL}/dictionary/all")
            if resp.ok:
                st.success("Wiped!")
                st.rerun()
            else:
                st.error("Failed to delete.")
    with col_btn_ref:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    col1, col2, col3 = st.columns(3)
    
    try:
        resp = requests.get(f"{API_URL}/dictionary")
        if resp.ok:
            items = resp.json().get("dictionary", [])
            
            active_count = sum(1 for i in items if i['status'] == 'active')
            pending_count = sum(1 for i in items if i['status'] == 'pending')
            col1.metric("Total Memories", len(items))
            col2.metric("Active", active_count)
            col3.metric("Pending Threshold", pending_count)
            
            st.divider()
            
            if items:
                df = pd.DataFrame(items)
                
                # Search Bar
                search_mem = st.text_input("🔍 Search Memory by Name", value="", placeholder="e.g. Aaditya")
                if search_mem:
                    df = df[df['canonical_word'].astype(str).str.contains(search_mem, case=False, na=False)]
                
                # Highlight active vs pending
                def style_status(val):
                    color = '#e8f5e9' if val == 'active' else '#fff3e0'
                    text_color = '#2e7d32' if val == 'active' else '#e65100'
                    return f'background-color: {color}; color: {text_color}; font-weight: bold'
                
                styled_df = df.style.applymap(style_status, subset=['status'])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Delete specific memory UI
                st.markdown("### 🗑️ Remove Memories")
                
                mem_options = df['id'].tolist()
                def format_mem(mem_id):
                    row = df[df['id'] == mem_id]
                    if not row.empty:
                        return f"ID {mem_id}: {row['canonical_word'].iloc[0]}"
                    return f"ID {mem_id}"

                with st.form("delete_memory_form"):
                    mems_to_del = st.multiselect("Select memories to delete:", options=mem_options, format_func=format_mem)
                    submitted = st.form_submit_button("Delete Selected", type="primary")
                        
                    if submitted:
                        if not mems_to_del:
                            st.warning("Please select at least one memory to delete.")
                        else:
                            success = True
                            for mem_id in mems_to_del:
                                del_resp = requests.delete(f"{API_URL}/dictionary/{mem_id}")
                                if not del_resp.ok:
                                    st.error(f"Failed to delete ID {mem_id}: {del_resp.text}")
                                    success = False
                            if success:
                                st.success(f"Successfully deleted {len(mems_to_del)} memories!")
                                st.rerun()
            else:
                st.info("Dictionary is empty.")
    except Exception as e:
        st.error(f"Connection failed: {e}")

elif page == "⚙️ Intervention Logs":
    st.title("Intervention Trace Logs")
    
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown("Audit trail of all phonetic replacements and LLM decisions.")
    with col_btn:
        if st.button("🔄 Refresh Logs", use_container_width=True):
            st.rerun()
            
    try:
        resp = requests.get(f"{API_URL}/logs")
        if resp.ok:
            logs = resp.json().get("logs", [])
            
            if logs:
                df = pd.DataFrame(logs)
                
                # Search Bar for Request ID
                search_query = st.text_input("🔍 Filter by Request ID", value="", placeholder="e.g. b4c9e10f")
                if search_query:
                    df = df[df['request_id'].astype(str).str.contains(search_query, case=False, na=False)]
                
                # Summary metrics
                avg_latency = df['execution_latency_ms'].mean() if 'execution_latency_ms' in df and not df.empty else 0
                total_interventions = len(df[df['did_intervene'] == True]) if not df.empty else 0
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Evaluated Tokens", len(df))
                col2.metric("Total Replacements", total_interventions)
                col3.metric("Avg LLM Latency", f"{avg_latency:.1f} ms")
                
                st.divider()
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No intervention logs found.")
    except Exception as e:
        st.error(f"Connection failed: {e}")
