from io import BytesIO

import pandas as pd

from src.data_loader import (
    list_excel_worksheets,
    load_data_file,
)


class UploadedFile:
    """Small test double compatible with the loader."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self):
        return self._data


def make_csv_file() -> UploadedFile:
    dataframe = pd.DataFrame(
        {
            "Date": ["2026-01-01", "2026-01-02"],
            "Product": ["Product A", "Product B"],
            "Production Qty": [1000, 1200],
        }
    )

    buffer = BytesIO()
    dataframe.to_csv(
        buffer,
        index=False,
    )

    return UploadedFile(
        "factory.csv",
        buffer.getvalue(),
    )


def make_excel_file() -> UploadedFile:
    buffer = BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:
        pd.DataFrame(
            {
                "Date": ["2026-01-01"],
                "Product": ["Product A"],
                "Production Qty": [1000],
            }
        ).to_excel(
            writer,
            sheet_name="Production",
            index=False,
        )

        pd.DataFrame(
            {
                "Date": ["2026-01-01"],
                "QC Status": ["Pass"],
            }
        ).to_excel(
            writer,
            sheet_name="QC",
            index=False,
        )

    return UploadedFile(
        "factory.xlsx",
        buffer.getvalue(),
    )


def test_load_valid_csv():
    result = load_data_file(
        make_csv_file()
    )

    assert result.success is True
    assert result.dataframe is not None
    assert result.row_count == 2
    assert "Date" in result.columns


def test_load_excel_requires_worksheet_when_multiple():
    file_obj = make_excel_file()

    result = load_data_file(
        file_obj
    )

    assert result.success is False
    assert "Production" in result.available_worksheets
    assert "QC" in result.available_worksheets
    assert any(
        "multiple worksheets" in error.lower()
        for error in result.errors
    )


def test_load_selected_excel_worksheet():
    file_obj = make_excel_file()

    result = load_data_file(
        file_obj,
        worksheet="Production",
    )

    assert result.success is True
    assert result.worksheet == "Production"
    assert result.row_count == 1


def test_list_excel_worksheets():
    file_obj = make_excel_file()

    result = list_excel_worksheets(
        file_obj
    )

    assert result.success is True
    assert result.available_worksheets == [
        "Production",
        "QC",
    ]


def test_unsupported_file_type_is_rejected():
    file_obj = UploadedFile(
        "factory.docx",
        b"not a supported data file",
    )

    result = load_data_file(
        file_obj
    )

    assert result.success is False
    assert any(
        "unsupported file type" in error.lower()
        for error in result.errors
    )


def test_empty_csv_is_rejected():
    file_obj = UploadedFile(
        "empty.csv",
        b"",
    )

    result = load_data_file(
        file_obj
    )

    assert result.success is False
    assert result.errors


def test_duplicate_rows_are_reported():
    dataframe = pd.DataFrame(
        {
            "Date": ["2026-01-01", "2026-01-01"],
            "Product": ["A", "A"],
            "Production Qty": [100, 100],
        }
    )

    buffer = BytesIO()
    dataframe.to_csv(
        buffer,
        index=False,
    )

    result = load_data_file(
        UploadedFile(
            "duplicates.csv",
            buffer.getvalue(),
        )
    )

    assert result.success is True
    assert result.duplicate_count == 1
    assert result.warnings
