from pathlib import Path
import re
import pytest
import src.preprocessing.preprocessor as mod  # 改成你的模块名

HERE = Path(__file__).parent
PDF_PATH = HERE / "spec.pdf"
PNG_PATH = HERE / "spec.png"

def norm(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def has_tesseract() -> bool:
    if getattr(mod, "pytesseract", None) is None:
        return False
    try:
        _ = mod.pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def load_expected_lines(for_path: Path):
    exp = for_path.with_name(for_path.name + ".expected.txt")
    if exp.exists():
        return [ln.strip() for ln in exp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return []

def dump_preview(text: str, label: str, tmp_dir: Path, basename: str, max_chars: int = 800):
    """
    打印预览（前 max_chars 个字符），并把全文保存到 tmp_dir/basename.full.txt。
    在 pytest 中使用 -s 或 --capture=no 可立即看到 print 输出。
    """
    raw_preview = text[:max_chars]
    norm_preview = norm(text)[:max_chars]

    print(
        f"\n[{label}] length={len(text)} chars"
        f"\n--- RAW preview (<= {max_chars}) ---\n{raw_preview}"
        f"\n--- NORM preview (<= {max_chars}) ---\n{norm_preview}"
    )

# ---------------- PDF ----------------

def test_spec_pdf_exists():
    assert PDF_PATH.exists(), f"未找到 {PDF_PATH}，请把 spec.pdf 放到 {HERE}。"

def test_spec_pdf_text_extraction(tmp_path):
    if mod.PdfReader is None:
        pytest.skip("未安装 pypdf，跳过 PDF 集成测试。")

    p = mod.Preprocessor()
    res = p.process("q", [PDF_PATH])

    assert res.issues == []
    assert len(res.pdf_attachments) == 1
    text = res.pdf_attachments[0].content
    assert isinstance(text, str)

    # 打印 + 保存
    dump_preview(text, "PDF", tmp_path, "spec_pdf_text")

    # 基础断言
    assert len(text.strip()) > 0, "PDF 文本为空，确认 spec.pdf 是否含可提取文本。"

    # 可选：通过 expected 片段做更稳的匹配
    expected = load_expected_lines(PDF_PATH)
    if expected:
        nt = norm(text)
        for frag in expected:
            assert norm(frag) in nt, f"PDF 文本未包含期望片段：{frag!r}"

# ---------------- Image + OCR ----------------

def test_spec_png_exists():
    assert PNG_PATH.exists(), f"未找到 {PNG_PATH}，请把 spec.png 放到 {HERE}。"

def test_spec_png_ocr(tmp_path):
    if mod.Image is None or mod.pytesseract is None:
        pytest.skip("未安装 Pillow/pytesseract，跳过 OCR 集成测试。")
    if not has_tesseract():
        pytest.skip("未检测到 tesseract 可执行文件，跳过 OCR 集成测试。")

    p = mod.Preprocessor(ocr_lang="chi_sim")  # 中文可改为 "chi_sim"
    res = p.process("q", [PNG_PATH])

    assert res.issues == []
    assert len(res.image_attachments) == 1
    text = res.image_attachments[0].content
    assert isinstance(text, str)

    # 打印
    dump_preview(text, "OCR", tmp_path, "spec_png_ocr")

    # 基础断言
    assert len(text.strip()) > 0, "OCR 结果为空，确认图片清晰度/对比度/字号。"

    # 可选：通过 expected 片段做更稳的匹配
    expected = load_expected_lines(PNG_PATH)
    if expected:
        nt = norm(text)
        for frag in expected:
            assert norm(frag) in nt, f"OCR 文本未包含期望片段：{frag!r}"
