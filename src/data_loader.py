"""
WasteWise AI data-file loading and validation.

Supported factory-data formats:
    - CSV
    - XLSX
    - XLS

This module is responsible for loading and basic file-level validation.
Business analytics belong in the analytics layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import pandas as pd

from .utils import (
    get_file_extension,
    is_supported_data_file,
)


SUPPORTED_EXTENSIONS = {"csv", "xlsx", "xls"}


@dataclass
class DataLoadResult:
    """Structured result returned by the data loader."""

    success: bool
    dataframe: pd.DataFrame | None = None
    filename: str = ""
    file_type: str = ""
    worksheet: str | None = None
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    duplicate_count: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    available_worksheets: list[str] = field(default_factory=list)


def _read_bytes(file_obj: Any) -> bytes:
    """Read uploaded-file-like objects safely."""
    if hasattr(file_obj, "getvalue"):
        return file_obj.getvalue()

    if hasattr(file_obj, "read"):
        current_position = None

        if hasattr(file_obj, "tell"):
            current_position = file_obj.tell()

        data = file_obj.read()

        if current_position is not None and hasattr(file_obj, "seek"):
            file_obj.seek(current_position)

        return data

    if isinstance(file_obj, bytes):
        return file_obj

    raise TypeError(
        "The supplied file object is not readable."
    )


def list_excel_worksheets(file_obj: Any) -> DataLoadResult:
    """
    Return worksheet names for an XLSX/XLS file.

    This does not select a worksheet automatically.
    """
    filename = getattr(file_obj, "name", "uploaded_file")
    extension = get_file_extension(filename)

    result = DataLoadResult(
        success=False,
        filename=filename,
        file_type=extension,
    )

    if extension not in {"xlsx", "xls"}:
        result.errors.append(
            "Worksheet selection is available only for XLSX and XLS files."
        )
        return result

    try:
        data = _read_bytes(file_obj)

        workbook = pd.ExcelFile(
            BytesIO(data),
            engine="openpyxl" if extension == "xlsx" else "xlrd",
        )

        result.available_worksheets = workbook.sheet_names
        result.success = True

    except Exception as exc:
        result.errors.append(
            f"Unable to read Excel workbook: {exc}"
        )

    return result


def _validate_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
    worksheet: str | None,
) -> DataLoadResult:
    """Perform basic dataframe-level validation."""
    result = DataLoadResult(
        success=True,
        dataframe=dataframe,
        filename=filename,
        file_type=get_file_extension(filename),
        worksheet=worksheet,
        columns=[str(column) for column in dataframe.columns],
        row_count=len(dataframe),
    )

    if dataframe.empty:
        result.success = False
        result.errors.append(
            "The uploaded file contains no data rows."
        )
        return result

    if dataframe.columns.empty:
        result.success = False
        result.errors.append(
            "The uploaded file does not contain any columns."
        )
        return result

    duplicate_count = int(dataframe.duplicated().sum())
    result.duplicate_count = duplicate_count

    if duplicate_count:
        result.warnings.append(
            f"{duplicate_count} duplicate row(s) detected. "
            "They have not been removed automatically."
        )

    unnamed_columns = [
        str(column)
        for column in dataframe.columns
        if str(column).strip().lower().startswith("unnamed:")
    ]

    if unnamed_columns:
        result.warnings.append(
            "Unnamed spreadsheet columns detected: "
            + ", ".join(unnamed_columns)
        )

    null_columns = [
        str(column)
        for column in dataframe.columns
        if dataframe[column].isna().any()
    ]

    if null_columns:
        result.warnings.append(
            "Missing values detected in: "
            + ", ".join(null_columns)
        )

    return result


def load_data_file(
    file_obj: Any,
    worksheet: str | None = None,
) -> DataLoadResult:
    """
    Load CSV/XLSX/XLS data.

    For Excel files, worksheet should be explicitly provided when the
    workbook contains multiple worksheets.
    """
    filename = getattr(file_obj, "name", "uploaded_file")
    extension = get_file_extension(filename)

    result = DataLoadResult(
        success=False,
        filename=filename,
        file_type=extension,
    )

    if extension not in SUPPORTED_EXTENSIONS:
        result.errors.append(
            "Unsupported file type. WasteWise AI accepts "
            "CSV, XLSX, and XLS files."
        )
        return result

    try:
        data = _read_bytes(file_obj)

        if extension == "csv":
            dataframe = pd.read_csv(
                BytesIO(data)
            )

            return _validate_dataframe(
                dataframe=dataframe,
                filename=filename,
                worksheet=None,
            )

        engine = (
            "openpyxl"
            if extension == "xlsx"
            else "xlrd"
        )

        excel = pd.ExcelFile(
            BytesIO(data),
            engine=engine,
        )

        worksheets = excel.sheet_names
        result.available_worksheets = worksheets

        if not worksheets:
            result.errors.append(
                "The Excel workbook does not contain any worksheets."
            )
            return result

        if worksheet is None:
            if len(worksheets) == 1:
                worksheet = worksheets[0]
            else:
                result.errors.append(
                    "This Excel workbook contains multiple worksheets. "
                    "Please select a worksheet explicitly."
                )
                return result

        if worksheet not in worksheets:
            result.errors.append(
                f"Worksheet '{worksheet}' was not found. "
                f"Available worksheets: {', '.join(worksheets)}"
            )
            return result

        dataframe = pd.read_excel(
            excel,
            sheet_name=worksheet,
        )

        validated = _validate_dataframe(
            dataframe=dataframe,
            filename=filename,
            worksheet=worksheet,
        )

        validated.available_worksheets = worksheets

        return validated

    except pd.errors.EmptyDataError:
        result.errors.append(
            "The uploaded file is empty."
        )

    except Exception as exc:
        result.errors.append(
            f"Unable to read '{filename}': {exc}"
        )

    return result
