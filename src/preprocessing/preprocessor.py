from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from pydantic import BaseModel

# --- Optional Dependencies ---
try:
    from PIL import Image, ImageOps
    import pytesseract
except ImportError:
    Image = pytesseract = ImageOps = None  # type: ignore

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# --- Types & Models ---
class SourceType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"


class IssueCode(str, Enum):
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    READ_ERROR = "READ_ERROR"
    OCR_ERROR = "OCR_ERROR"


class AttachmentText(BaseModel):
    path: Path
    content: str
    source_type: SourceType  # "pdf" | "image"


class AttachmentIssue(BaseModel):
    path: Path
    code: IssueCode
    message: str
    source_type: Optional[SourceType] = None  # 对于不支持类型/找不到文件可为空


@dataclass
class PreprocessResult:
    raw_query: str
    pdf_attachments: List[AttachmentText]
    image_attachments: List[AttachmentText]
    issues: List[AttachmentIssue]


# --- Helpers ---
def _extract_pdf(path: Path) -> Tuple[Optional[AttachmentText], Optional[AttachmentIssue]]:
    if PdfReader is None:
        return None, AttachmentIssue(
            path=path,
            code=IssueCode.DEPENDENCY_MISSING,
            message="PDF support requires the 'pypdf' library.",
            source_type=SourceType.PDF,
        )
    try:
        reader = PdfReader(str(path))
        pages_content = "\n".join((page.extract_text() or "") for page in reader.pages)
        content = pages_content.strip()
        return AttachmentText(path=path, content=content, source_type=SourceType.PDF), None
    except Exception as e:
        return None, AttachmentIssue(
            path=path,
            code=IssueCode.READ_ERROR,
            message=f"Error reading PDF: {e}",
            source_type=SourceType.PDF,
        )


def _extract_image(path: Path, ocr_lang: str) -> Tuple[Optional[AttachmentText], Optional[AttachmentIssue]]:
    if Image is None or pytesseract is None:
        return None, AttachmentIssue(
            path=path,
            code=IssueCode.DEPENDENCY_MISSING,
            message="OCR requires 'Pillow' and 'pytesseract'.",
            source_type=SourceType.IMAGE,
        )
    try:
        with Image.open(path) as img:
            try:
                img = ImageOps.exif_transpose(img)  # type: ignore[attr-defined]
            except Exception:
                pass
            if img.mode not in ("L", "RGB"):
                img = img.convert("RGB")
            text = pytesseract.image_to_string(img, lang=ocr_lang).strip()
        return AttachmentText(path=path, content=text, source_type=SourceType.IMAGE), None
    except Exception as e:
        return None, AttachmentIssue(
            path=path,
            code=IssueCode.OCR_ERROR,
            message=f"Error during OCR: {e}",
            source_type=SourceType.IMAGE,
        )


# --- Main Preprocessor ---
class Preprocessor:
    image_exts: set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
    pdf_exts: set[str] = {".pdf"}

    def __init__(self, ocr_lang: str = "eng"):
        self.ocr_lang = ocr_lang

    def process(self, query: str, attachments: Optional[Iterable[Union[str, Path]]] = None) -> PreprocessResult:
        pdf_attachments: List[AttachmentText] = []
        image_attachments: List[AttachmentText] = []
        issues: List[AttachmentIssue] = []

        for raw_path in (attachments or []):
            path = Path(raw_path)

            # 1) 文件是否存在
            if not path.exists():
                issues.append(AttachmentIssue(
                    path=path,
                    code=IssueCode.FILE_NOT_FOUND,
                    message="File not found.",
                    source_type=None,
                ))
                continue

            # 2) 根据后缀路由
            suffix = path.suffix.lower()
            if suffix in self.pdf_exts:
                att, err = _extract_pdf(path)
                if att:
                    pdf_attachments.append(att)
                if err:
                    issues.append(err)
            elif suffix in self.image_exts:
                att, err = _extract_image(path, self.ocr_lang)
                if att:
                    image_attachments.append(att)
                if err:
                    issues.append(err)
            else:
                issues.append(AttachmentIssue(
                    path=path,
                    code=IssueCode.UNSUPPORTED_TYPE,
                    message=f"Unsupported file type: '{suffix}'",
                    source_type=None,
                ))

        return PreprocessResult(
            raw_query=query,
            pdf_attachments=pdf_attachments,
            image_attachments=image_attachments,
            issues=issues,
        )
