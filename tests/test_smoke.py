from pathlib import Path


def test_repository_scaffold_exists():
    root = Path(__file__).resolve().parents[1]

    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
        ".env.example",
        "src/__init__.py",
        "tests/test_smoke.py",
    ]

    for relative_path in required_files:
        assert (root / relative_path).exists(), f"Missing {relative_path}"


def test_app_has_streamlit_entrypoint():
    root = Path(__file__).resolve().parents[1]
    app_text = (root / "app.py").read_text(encoding="utf-8")

    assert "import streamlit as st" in app_text
    assert "st.set_page_config" in app_text
