import streamlit as st

st.set_page_config(page_title="WasteWise AI", page_icon="♻️", layout="wide")

st.title("WasteWise AI")
st.subheader("Manufacturing Loss & Rework Intelligence")
st.info(
    "WasteWise AI helps manufacturing teams investigate material loss, "
    "quality deviations, rework, recovery, and final loss using factory "
    "data, documents, visual evidence, and AI reasoning."
)

with st.sidebar:
    st.header("WasteWise AI")
    st.caption("MVP workflow: Upload → Analyze → Ask → Investigate")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Upload Center", "Visual Inspection", "AI Analyst", "SOP Knowledge", "Findings"],
    )

if page == "Dashboard":
    st.header("Factory Overview")
    cols = st.columns(4)
    cols[0].metric("Total Production", "Not available")
    cols[1].metric("Final Material Loss", "Not available")
    cols[2].metric("Loss Rate", "Not available")
    cols[3].metric("Rework Total", "Not available")
    st.warning(
        "Upload factory data in Upload Center to calculate deterministic "
        "production, loss, rework, recovery, rate, and cost metrics."
    )
    st.markdown("### Analytics")
    st.caption("Charts and anomaly indicators will populate after the Data Engine and Analytics stages are implemented.")

elif page == "Upload Center":
    st.header("Upload Center")
    st.markdown(
        "Upload factory evidence for analysis. The application will validate "
        "and dynamically map supported data fields rather than assuming a "
        "specific factory schema."
    )
    data_files = st.file_uploader("Production / waste data", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    document_files = st.file_uploader("Factory documents", type=["pdf", "txt"], accept_multiple_files=True)
    image_files = st.file_uploader("Visual evidence", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

    if data_files:
        st.success(f"{len(data_files)} data file(s) selected.")
    if document_files:
        st.success(f"{len(document_files)} document(s) selected.")
    if image_files:
        st.success(f"{len(image_files)} image(s) selected.")

    if st.button("Analyze", type="primary"):
        if not (data_files or document_files or image_files):
            st.error("Please upload at least one supported evidence file.")
        else:
            st.info(
                "Analysis pipeline is being added incrementally. Validation, "
                "mapping, deterministic analytics, RAG, and multimodal analysis "
                "will run in their respective stages."
            )

elif page == "Visual Inspection":
    st.header("Visual Inspection")
    image = st.file_uploader("Upload an image for visual assessment", type=["jpg", "jpeg", "png", "webp"])
    if image:
        st.image(image, caption=image.name, use_container_width=True)
        st.markdown("### Structured Visual Findings")
        st.info(
            "AI visual assessment will report observable issues, category, severity, "
            "possible causes, investigation suggestions, and confidence. An image "
            "will never be treated as proof of root cause."
        )
    else:
        st.caption("No image uploaded.")

elif page == "AI Analyst":
    st.header("Ask WasteWise")
    st.caption(
        "Ask questions about uploaded manufacturing evidence. Answers will separate "
        "calculated facts, document evidence, visual observations, inferences, and recommendations."
    )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Example: Which production line should we investigate first?")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        response = (
            "AI Analyst is not connected yet. Complete the Data Engine, RAG, Grok, "
            "and reasoning stages to generate evidence-grounded answers."
        )
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

elif page == "SOP Knowledge":
    st.header("SOP Knowledge")
    st.markdown(
        "Uploaded SOPs, QC procedures, rework procedures, specifications, and "
        "operating/investigation procedures will be indexed for semantic retrieval."
    )
    st.info(
        "RAG pipeline: document extraction → cleaning → chunking → embeddings → "
        "vector search → relevant evidence → AI answer with source metadata."
    )

elif page == "Findings":
    st.header("Investigation Findings")
    st.markdown("### Evidence structure")
    st.markdown(
        """
        - **Data fact** — deterministic metric or pattern calculated from uploaded data.
        - **Visual observation** — something visibly present in an uploaded image.
        - **SOP evidence** — information retrieved from an uploaded document.
        - **Inference / hypothesis** — an AI-generated interpretation that is not confirmed.
        - **Recommended investigation** — a suggested next check for qualified factory personnel.
        """
    )
    st.warning(
        "WasteWise AI is decision support. It does not autonomously approve, reject, "
        "release, dispose, quarantine, rework, control equipment, or declare a root cause confirmed."
    )
