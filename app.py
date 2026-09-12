from __future__ import annotations

import streamlit as st

from src.data_loader import load_data_file
from src.data_mapper import create_mapping_result


st.set_page_config(
    page_title="WasteWise AI",
    page_icon="♻️",
    layout="wide",
)


def show_mapping_result(result):
    """Display dynamic data-mapping results."""

    st.markdown("### Detected Field Mapping")

    if result.mappings:
        mapping_rows = [
            {
                "WasteWise field": canonical,
                "Source column": source,
            }
            for canonical, source in result.mappings.items()
        ]

        st.dataframe(
            mapping_rows,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No canonical fields could be mapped automatically."
        )

    if result.ambiguous:
        st.warning(
            "Ambiguous mappings require user confirmation. "
            "WasteWise AI will not silently guess important fields."
        )

        for field, candidates in result.ambiguous.items():
            st.write(
                f"**{field}** → "
                + ", ".join(candidates)
            )

    if result.missing_fields:
        st.warning(
            "Required fields could not be mapped: "
            + ", ".join(result.missing_fields)
        )

    if result.unmapped_source_columns:
        with st.expander(
            "Unmapped source columns"
        ):
            st.write(
                result.unmapped_source_columns
            )

    if result.warnings:
        with st.expander(
            "Validation and normalization warnings"
        ):
            for warning in result.warnings:
                st.write(f"• {warning}")


def analyze_uploaded_file(
    uploaded_file,
    worksheet: str | None = None,
):
    """Load, validate, map, and normalize an uploaded data file."""

    load_result = load_data_file(
        uploaded_file,
        worksheet=worksheet,
    )

    if not load_result.success:
        st.error(
            "The file could not be loaded."
        )

        for error in load_result.errors:
            st.write(f"• {error}")

        return

    st.success(
        f"Loaded {load_result.filename} successfully."
    )

    st.write(
        f"**Rows:** {load_result.row_count}"
    )

    st.write(
        f"**Columns:** {len(load_result.columns)}"
    )

    if load_result.duplicate_count:
        st.warning(
            f"{load_result.duplicate_count} duplicate row(s) detected."
        )

    mapping_result = create_mapping_result(
        load_result.dataframe
    )

    show_mapping_result(
        mapping_result
    )

    st.markdown(
        "### Canonical Data Preview"
    )

    if mapping_result.canonical_dataframe is not None:
        st.dataframe(
            mapping_result.canonical_dataframe.head(20),
            use_container_width=True,
        )

        st.caption(
            "This canonical representation is the interface used by "
            "the downstream analytics engine. Missing fields remain "
            "unavailable rather than being fabricated."
        )


st.title("WasteWise AI")
st.subheader(
    "Manufacturing Loss & Rework Intelligence"
)

with st.sidebar:
    st.header("WasteWise AI")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Upload Center",
            "Visual Inspection",
            "AI Analyst",
            "SOP Knowledge",
            "Findings",
        ],
    )

    st.caption(
        "MVP workflow: Upload → Analyze → Ask → Investigate"
    )


if page == "Dashboard":
    st.header("Factory Overview")

    cols = st.columns(4)

    cols[0].metric(
        "Total Production",
        "Not available",
    )

    cols[1].metric(
        "Final Material Loss",
        "Not available",
    )

    cols[2].metric(
        "Loss Rate",
        "Not available",
    )

    cols[3].metric(
        "Rework Total",
        "Not available",
    )

    st.info(
        "Upload factory data in Upload Center. "
        "Deterministic KPI calculations will be added in the "
        "Analytics stage."
    )

    st.markdown("### Analytics")

    st.caption(
        "Product, line, shift, loss, rework, recovery, cost, trend, "
        "and anomaly analysis will populate after the Data Engine "
        "and Analytics stages are completed."
    )


elif page == "Upload Center":
    st.header("Upload Center")

    st.markdown(
        """
        Upload factory evidence for WasteWise AI.

        **Supported factory data:** CSV, XLSX, XLS

        **Supported documents:** PDF, TXT

        **Supported visual evidence:** JPG, JPEG, PNG, WebP
        """
    )

    data_files = st.file_uploader(
        "Production / waste data",
        type=[
            "csv",
            "xlsx",
            "xls",
        ],
        accept_multiple_files=True,
        help=(
            "Upload production, waste, quality, rework, "
            "or related factory data."
        ),
    )

    document_files = st.file_uploader(
        "Factory documents",
        type=[
            "pdf",
            "txt",
        ],
        accept_multiple_files=True,
        help=(
            "Examples: SOPs, QC manuals, rework procedures, "
            "specifications, investigation procedures."
        ),
    )

    image_files = st.file_uploader(
        "Visual evidence",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        accept_multiple_files=True,
        help=(
            "Examples: damaged packaging, product defects, "
            "or waste material."
        ),
    )

    if data_files:
        st.success(
            f"{len(data_files)} data file(s) selected."
        )

        for index, uploaded_file in enumerate(
            data_files
        ):
            st.markdown(
                f"#### Data file {index + 1}: "
                f"{uploaded_file.name}"
            )

            filename = uploaded_file.name.lower()

            if filename.endswith(
                (".xlsx", ".xls")
            ):
                worksheet_result = load_data_file(
                    uploaded_file
                )

                if not worksheet_result.success:
                    for error in worksheet_result.errors:
                        st.error(error)

                    continue

                worksheets = (
                    worksheet_result.available_worksheets
                )

                if len(worksheets) > 1:
                    selected_sheet = st.selectbox(
                        "Select worksheet",
                        worksheets,
                        key=f"sheet_{index}",
                    )
                else:
                    selected_sheet = worksheets[0]

                if st.button(
                    f"Analyze {uploaded_file.name}",
                    key=f"analyze_excel_{index}",
                    type="primary",
                ):
                    analyze_uploaded_file(
                        uploaded_file,
                        worksheet=selected_sheet,
                    )

            else:
                if st.button(
                    f"Analyze {uploaded_file.name}",
                    key=f"analyze_csv_{index}",
                    type="primary",
                ):
                    analyze_uploaded_file(
                        uploaded_file
                    )

    if document_files:
        st.success(
            f"{len(document_files)} document(s) selected."
        )

        st.info(
            "Document indexing and RAG are implemented in the "
            "SOP Knowledge stage."
        )

    if image_files:
        st.success(
            f"{len(image_files)} image(s) selected."
        )

        st.info(
            "Multimodal image analysis is implemented in the "
            "Visual Inspection stage."
        )


elif page == "Visual Inspection":
    st.header("Visual Inspection")

    image = st.file_uploader(
        "Upload an image for visual assessment",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        accept_multiple_files=False,
    )

    if image:
        st.image(
            image,
            caption=image.name,
            use_container_width=True,
        )

        st.markdown(
            "### Structured Visual Findings"
        )

        st.info(
            "Grok multimodal assessment will report observable issues, "
            "category, severity, possible causes, investigation "
            "suggestions, and confidence in the Vision stage."
        )

        st.warning(
            "An image is evidence of what is visually observable. "
            "It is never treated as proof of root cause."
        )

    else:
        st.caption(
            "No image uploaded."
        )


elif page == "AI Analyst":
    st.header("Ask WasteWise")

    st.caption(
        "The AI Analyst will combine deterministic analytics, "
        "document evidence, visual findings, and the user's "
        "question in later stages."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

    question = st.chat_input(
        "Example: Which production line should we investigate first?"
    )

    if question:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        response = (
            "The AI Analyst is not connected yet. "
            "Complete the Analytics, RAG, Grok, Vision, and "
            "cross-source reasoning stages to generate "
            "evidence-grounded answers."
        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        st.rerun()


elif page == "SOP Knowledge":
    st.header("SOP Knowledge")

    st.markdown(
        """
        Factory SOPs, QC procedures, rework procedures,
        specifications, and investigation procedures will be
        indexed here.
        """
    )

    st.info(
        "RAG pipeline: extraction → cleaning → chunking → "
        "embeddings → vector search → evidence → AI answer."
    )


elif page == "Findings":
    st.header("Investigation Findings")

    st.markdown(
        "### Evidence classification"
    )

    st.markdown(
        """
        - **Data fact** — deterministic metric or pattern calculated from uploaded data.
        - **Visual observation** — something visibly present in an uploaded image.
        - **SOP evidence** — information retrieved from an uploaded document.
        - **Inference / hypothesis** — AI interpretation that is not confirmed.
        - **Recommended investigation** — suggested next check for qualified factory personnel.
        """
    )

    st.warning(
        "WasteWise AI is decision support. It does not autonomously "
        "approve, reject, release, dispose, quarantine, rework, "
        "control equipment, or declare a root cause confirmed."
    )
