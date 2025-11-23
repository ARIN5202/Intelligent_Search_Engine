from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional, Tuple, Union
import os
os.environ['TESSDATA_PREFIX'] = '/opt/anaconda3/envs/NLP1/share/tessdata'
from pydantic import BaseModel

# --- Optional Dependencies ---
try:
    from PIL import Image, ImageOps, ImageFilter
    import pytesseract
    from pytesseract import Output
except ImportError:
    Image = ImageOps = ImageFilter = pytesseract = Output = None  # type: ignore

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# 可选：更稳的阈值化/去噪（若无也可正常运行）
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:
    cv2 = np = None


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


# --- PDF 提取（文本型PDF） ---
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


# --- OCR 辅助：预处理 / OSD / 试跑打分 ---
def _preprocess_for_ocr(img: "Image.Image") -> "Image.Image":
    """对中英通用的轻量预处理：灰度、放大、对比度、去噪、（可选）Otsu 二值化"""
    # 灰度
    g = ImageOps.grayscale(img)

    # 小图放大到 ~1000px 短边（中文/英文均有利）
    short = min(g.size)
    if short < 1000:
        scale = 1000.0 / float(short)
        g = g.resize((int(g.width * scale), int(g.height * scale)), Image.BICUBIC)

    # 对比度拉伸 + 中值滤波
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.MedianFilter(3))

    # 若有 OpenCV，用 Otsu 二值化（对中文更稳）
    if cv2 is not None and np is not None:
        arr = np.array(g)
        try:
            _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            g = Image.fromarray(bw)
        except Exception:
            pass

    return g


def _deskew_and_orient(img: "Image.Image") -> Tuple["Image.Image", Optional[str]]:
    """使用 Tesseract OSD 检测旋转角度与脚本（Latin/Han 等），失败时原样返回"""
    if pytesseract is None:
        return img, None
    try:
        osd = pytesseract.image_to_osd(img)
        rot = re.search(r"Rotate:\s+(\d+)", osd)
        angle = int(rot.group(1)) if rot else 0
        if angle % 360 != 0:
            # Tesseract 报告需要顺时针旋转的角度；PIL rotate 为逆时针，因此用(360 - angle)
            img = img.rotate(360 - angle, expand=True)

        scr = re.search(r"Script:\s+([A-Za-z0-9_]+)", osd)
        script = scr.group(1).lower() if scr else None
        return img, script
    except Exception:
        return img, None


def _score_ocr(img: "Image.Image", lang: str, config: str) -> Tuple[float, int, str]:
    """轻量试跑：返回(平均置信度, 字符数, 文本)；无 Output 时回退到仅基于长度"""
    if Output is not None:
        data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=Output.DICT)  # type: ignore
        confs = [float(c) for c in data.get("conf", []) if c != "-1"]
        avg_conf = mean(confs) if confs else 0.0
        tokens = [t for t in data.get("text", []) if t and t.strip()]
        text = " ".join(tokens).strip()
        return avg_conf, len(text), text
    else:
        text = pytesseract.image_to_string(img, lang=lang, config=config)  # type: ignore
        return 0.0, len(text), text.strip()


def _auto_lang_and_ocr(
    img: "Image.Image",
    candidates: List[str],
    final_psm: int = 6,
    preserve_spaces: bool = True,
) -> str:
    """
    自动语言选择：对候选语言做轻量试跑，按综合分选最优，再做最终高精度OCR。
    candidates 例：["chi_tra", "chi_tra+eng", "eng"]
    """
    keep_space = "1" if preserve_spaces else "0"
    quick_cfg = f"--oem 1 --psm 6 -c preserve_interword_spaces={keep_space}"

    results: List[Tuple[float, float, int, str]] = []
    for lang in candidates:
        try:
            avg_conf, nchar, _ = _score_ocr(img, lang, quick_cfg)
            score = avg_conf + min(nchar, 2000) * 0.01  # 简单综合评分
            results.append((score, avg_conf, nchar, lang))
        except Exception:
            continue

    if not results:
        best_lang = "chi_tra+eng"  # 兜底：混排友好
    else:
        results.sort(reverse=True, key=lambda x: x[0])
        best_lang = results[0][3]

    final_cfg = f"--oem 1 --psm {final_psm} -c preserve_interword_spaces={keep_space}"
    text = pytesseract.image_to_string(img, lang=best_lang, config=final_cfg)  # type: ignore
    return text.strip()


# --- Image OCR 主流程（含自动语言） ---
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
            # 方向矫正（EXIF）
            try:
                img = ImageOps.exif_transpose(img)  # type: ignore[attr-defined]
            except Exception:
                pass
            if img.mode not in ("L", "RGB"):
                img = img.convert("RGB")

            # 自动语言模式
            if ocr_lang.lower() == "auto":
                # 1) OSD：纠偏并推测脚本
                img_osd, script = _deskew_and_orient(img)

                # 2) 预处理
                proc = _preprocess_for_ocr(img_osd)

                # 3) 候选缩小：Han→中文优先；Latin→英文优先；未知→都试
                if script and "han" in script:
                    candidates = ["chi_tra", "chi_tra+eng", "eng"]
                elif script and "latin" in script:
                    candidates = ["eng", "chi_tra+eng", "chi_tra"]
                else:
                    candidates = ["chi_tra+eng", "eng", "chi_tra"]

                text = _auto_lang_and_ocr(proc, candidates, final_psm=6, preserve_spaces=True)

            else:
                # 固定语言模式（兼容原行为）
                proc = _preprocess_for_ocr(img)
                cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
                text = pytesseract.image_to_string(proc, lang=ocr_lang, config=cfg)  # type: ignore

        return AttachmentText(path=path, content=text.strip(), source_type=SourceType.IMAGE), None

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

    def __init__(self, ocr_lang: str = "auto"):
        """
        ocr_lang:
            - "auto"：自动在繁体中文/英文之间选择（含混排），更侧重准确率；
            - 其他（如 "eng", "chi_tra", "chi_tra+eng"）：固定语言，速度更快。
        """
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
