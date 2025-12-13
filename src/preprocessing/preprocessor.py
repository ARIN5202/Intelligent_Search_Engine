from __future__ import annotations

import re
import base64
from io import BytesIO
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional, Tuple, Union
import os
os.environ['TESSDATA_PREFIX'] = '/opt/anaconda3/envs/NLP1/share/tessdata'
from pydantic import BaseModel
from openai import AzureOpenAI
from config import get_settings

settings = get_settings()

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

# Optional: render PDF pages to images for vision fallback
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

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
    processed_query: str
    pdf_attachments: List[AttachmentText]
    image_attachments: List[AttachmentText]
    issues: List[AttachmentIssue]


@dataclass
class ExtractionResult:
    attachment: Optional[AttachmentText]
    issue: Optional[AttachmentIssue]
    avg_conf: Optional[float] = None
    n_chars: int = 0
    used_fallback: bool = False


# --- PDF 提取（文本型PDF） ---
def _extract_pdf(path: Path) -> ExtractionResult:
    if PdfReader is None:
        return ExtractionResult(
            attachment=None,
            issue=AttachmentIssue(
                path=path,
                code=IssueCode.DEPENDENCY_MISSING,
                message="PDF support requires the 'pypdf' library.",
                source_type=SourceType.PDF,
            ),
            avg_conf=None,
            n_chars=0,
        )
    try:
        reader = PdfReader(str(path))
        pages_content = "\n".join((page.extract_text() or "") for page in reader.pages)
        content = pages_content.strip()
        return ExtractionResult(
            attachment=AttachmentText(path=path, content=content, source_type=SourceType.PDF),
            issue=None,
            avg_conf=None,
            n_chars=len(content),
        )
    except Exception as e:
        return ExtractionResult(
            attachment=None,
            issue=AttachmentIssue(
                path=path,
                code=IssueCode.READ_ERROR,
                message=f"Error reading PDF: {e}",
                source_type=SourceType.PDF,
            ),
            avg_conf=None,
            n_chars=0,
        )


# --- File/Image helpers ---
def _pil_image_to_base64(img: "Image.Image", mime_type: str = "image/png") -> str:
    buf = BytesIO()
    img.save(buf, format=mime_type.split("/")[-1])
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _file_image_to_base64(path: Path) -> Optional[str]:
    mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded}"
    except Exception:
        return None


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
) -> Tuple[str, str, float, int]:
    """
    自动语言选择：对候选语言做轻量试跑，按综合分选最优，再做最终高精度OCR。
    candidates 例：["chi_tra", "chi_tra+eng", "eng"]
    返回: text, best_lang, best_avg_conf, best_nchar
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
        best_avg_conf, best_nchar = 0.0, 0
    else:
        results.sort(reverse=True, key=lambda x: x[0])
        _, best_avg_conf, best_nchar, best_lang = results[0]

    final_cfg = f"--oem 1 --psm {final_psm} -c preserve_interword_spaces={keep_space}"
    text = pytesseract.image_to_string(img, lang=best_lang, config=final_cfg)  # type: ignore
    return text.strip(), best_lang, best_avg_conf, best_nchar


# --- Image OCR 主流程（含自动语言） ---
def _extract_image(path: Path, ocr_lang: str) -> ExtractionResult:
    if Image is None or pytesseract is None:
        return ExtractionResult(
            attachment=None,
            issue=AttachmentIssue(
                path=path,
                code=IssueCode.DEPENDENCY_MISSING,
                message="OCR requires 'Pillow' and 'pytesseract'.",
                source_type=SourceType.IMAGE,
            ),
            avg_conf=None,
            n_chars=0,
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

                text, _, avg_conf, nchar = _auto_lang_and_ocr(
                    proc, candidates, final_psm=6, preserve_spaces=True
                )

            else:
                # 固定语言模式（兼容原行为）
                proc = _preprocess_for_ocr(img)
                cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
                avg_conf, nchar, text = _score_ocr(proc, ocr_lang, cfg)

        return ExtractionResult(
            attachment=AttachmentText(path=path, content=text.strip(), source_type=SourceType.IMAGE),
            issue=None,
            avg_conf=avg_conf,
            n_chars=nchar,
        )

    except Exception as e:
        return ExtractionResult(
            attachment=None,
            issue=AttachmentIssue(
                path=path,
                code=IssueCode.OCR_ERROR,
                message=f"Error during OCR: {e}",
                source_type=SourceType.IMAGE,
            ),
            avg_conf=None,
            n_chars=0,
        )


def _translate_zh_to_en(self, text: str) -> str:
    """
    使用 Azure OpenAI (gpt-4o 部署) 把繁体中文翻译成英文。
    如果 text 已经是英文，调用方可以先用 _contains_chinese 判断再决定要不要调用。
    """
    text = text.strip()
    if not text:
        return text

    try:
        # 直接使用类中已初始化的 self.client
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a translation engine. "
                        "Translate the user's message from Traditional Chinese to natural English. "
                        "If the text is already in English, return it unchanged. "
                        "Do not add explanations or comments; return only the translation."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            temperature=0.0,  # 翻译用 0 温度，保证稳定
        )
        translated = response.choices[0].message.content.strip()
        return translated
    except Exception as e:
        # 如果翻译失败，返回原文并记录错误
        print(f"[WARN] Azure translation failed: {e}")
        return "[TRANSLATION FAILED]"  # 你可以选择返回一个标识失败的文本或日志


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
        self.client = AzureOpenAI(
            azure_endpoint=settings.azure_url,
            api_key=settings.azure_api_key,
            api_version="2025-02-01-preview",
        )
        # 质量阈值：低于该置信度或长度过短则尝试视觉 LLM 兜底
        self.min_ocr_conf = 40.0
        self.min_ocr_chars = 30
        self.min_pdf_chars = 50

    def process(self, query: str, attachments: Optional[Iterable[Union[str, Path]]] = None) -> PreprocessResult:
        pdf_attachments: List[AttachmentText] = []
        image_attachments: List[AttachmentText] = []
        issues: List[AttachmentIssue] = []

        try:
            # 总是将输入翻译成英文
            processed_query = _translate_zh_to_en(self, text=query)
        except Exception as e:
            # 如果翻译失败，返回原始文本并记录失败
            processed_query = query
            issues.append(AttachmentIssue(
                path=Path("<query>"),
                code=IssueCode.READ_ERROR,
                message=f"Failed to translate query from Chinese to English: {e}",
                source_type=None,
            ))

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
                res = _extract_pdf(path)
                if res.issue:
                    issues.append(res.issue)
                if res.attachment:
                    # pypdf 结果过弱则尝试视觉 LLM 兜底
                    if res.n_chars < self.min_pdf_chars:
                        llm_text = self._vision_extract_from_pdf(path)
                        if llm_text:
                            res.attachment = AttachmentText(
                                path=path,
                                content=llm_text.strip(),
                                source_type=SourceType.PDF,
                            )
                            res.used_fallback = True
                        else:
                            issues.append(AttachmentIssue(
                                path=path,
                                code=IssueCode.OCR_ERROR,
                                message="PDF text too short; vision fallback unavailable or failed.",
                                source_type=SourceType.PDF,
                            ))
                    pdf_attachments.append(res.attachment)
            elif suffix in self.image_exts:
                res = _extract_image(path, self.ocr_lang)
                if res.issue:
                    issues.append(res.issue)
                if res.attachment:
                    needs_fallback = (
                        (res.avg_conf is not None and res.avg_conf < self.min_ocr_conf)
                        or res.n_chars < self.min_ocr_chars
                    )
                    if needs_fallback:
                        llm_text = self._vision_extract_from_image(path, query)
                        if llm_text:
                            res.attachment = AttachmentText(
                                path=path,
                                content=llm_text.strip(),
                                source_type=SourceType.IMAGE,
                            )
                            res.used_fallback = True
                        else:
                            issues.append(AttachmentIssue(
                                path=path,
                                code=IssueCode.OCR_ERROR,
                                message="OCR confidence low; vision fallback unavailable or failed.",
                                source_type=SourceType.IMAGE,
                            ))
                    image_attachments.append(res.attachment)
            else:
                issues.append(AttachmentIssue(
                    path=path,
                    code=IssueCode.UNSUPPORTED_TYPE,
                    message=f"Unsupported file type: '{suffix}'",
                    source_type=None,
                ))

        return PreprocessResult(
            raw_query=query,
            processed_query=processed_query,
            pdf_attachments=pdf_attachments,
            image_attachments=image_attachments,
            issues=issues,
        )

    def _vision_extract_from_image(self, path: Path, query: str) -> Optional[str]:
        """Use Azure vision model to re-extract text from low-quality images."""
        if not self.client:
            return None
        image_b64 = _file_image_to_base64(path)
        if not image_b64:
            return None

        messages = [
            {"role": "system", "content": "Extract all readable text from the image. Return plain text only."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query or "Extract visible text in English."},
                    {"type": "image_url", "image_url": {"url": image_b64, "detail": "high"}},
                ],
            },
        ]
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=800,
            )
            return resp.choices[0].message.content if resp and resp.choices else None
        except Exception as e:
            print(f"[WARN] Vision fallback failed for image {path.name}: {e}")
            return None

    def _vision_extract_from_pdf(self, path: Path, max_pages: int = 2) -> Optional[str]:
        """Render first few PDF pages to images and run vision extraction."""
        if convert_from_path is None or not self.client:
            return None
        try:
            images = convert_from_path(str(path), first_page=1, last_page=max_pages, fmt="png")
        except Exception as e:
            print(f"[WARN] PDF to image conversion failed for {path.name}: {e}")
            return None

        image_blobs = []
        for img in images:
            try:
                image_blobs.append(_pil_image_to_base64(img, mime_type="image/png"))
            except Exception:
                continue

        if not image_blobs:
            return None

        content_items = [{"type": "text", "text": "Extract all readable text from these PDF pages."}]
        for b64 in image_blobs:
            content_items.append({"type": "image_url", "image_url": {"url": b64, "detail": "high"}})

        messages = [
            {"role": "system", "content": "You extract text from document page images. Return plain text only."},
            {"role": "user", "content": content_items},
        ]
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1200,
            )
            return resp.choices[0].message.content if resp and resp.choices else None
        except Exception as e:
            print(f"[WARN] Vision fallback failed for PDF {path.name}: {e}")
            return None
