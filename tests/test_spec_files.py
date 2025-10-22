from pathlib import Path
import re
import pytest
import src.preprocessing.preprocessor as mod  # 改成你的模块名

HERE = Path(__file__).parent
PDF_PATH = HERE / "spec.pdf"
PNG_PATH_ENG = HERE / "spec_eng.png"
PNG_PATH_CHI = HERE / "spec_chi.png"
PNG_PATH_MIX = HERE / "spec_mix.png"

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

def dump_preview(text: str, label: str):
    """
    打印预览（前 max_chars 个字符），并把全文保存到 tmp_dir/basename.full.txt。
    在 pytest 中使用 -s 或 --capture=no 可立即看到 print 输出。
    """
    raw_preview = text

    print(
        f"\n[{label}] length={len(text)} chars"
        f"\n--- RAW preview ---\n{raw_preview}"
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
    dump_preview(text, "PDF")

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
    assert PNG_PATH_ENG.exists(), f"未找到 {PNG_PATH_ENG}，请把 spec.png 放到 {HERE}。"
    assert PNG_PATH_CHI.exists(), f"未找到 {PNG_PATH_CHI}，请把 spec.png 放到 {HERE}。"
    assert PNG_PATH_MIX.exists(), f"未找到 {PNG_PATH_MIX}，请把 spec.png 放到 {HERE}。"

def test_spec_png_ocr_eng(tmp_path):
    if mod.Image is None or mod.pytesseract is None:
        pytest.skip("未安装 Pillow/pytesseract，跳过 OCR 集成测试。")
    if not has_tesseract():
        pytest.skip("未检测到 tesseract 可执行文件，跳过 OCR 集成测试。")

    p = mod.Preprocessor(ocr_lang="auto")
    res = p.process("q", [PNG_PATH_ENG])

    assert res.issues == []
    assert len(res.image_attachments) == 1
    text = res.image_attachments[0].content
    assert isinstance(text, str)

    # 打印
    dump_preview(text, "OCR")

    # 基础断言
    assert len(text.strip()) > 0, "OCR 结果为空，确认图片清晰度/对比度/字号。"

    # 可选：通过 expected 片段做更稳的匹配
    expected = load_expected_lines(PNG_PATH_ENG)
    if expected:
        nt = norm(text)
        for frag in expected:
            assert norm(frag) in nt, f"OCR 文本未包含期望片段：{frag!r}"


def test_spec_png_ocr_chi(tmp_path):
    if mod.Image is None or mod.pytesseract is None:
        pytest.skip("未安装 Pillow/pytesseract，跳过 OCR 集成测试。")
    if not has_tesseract():
        pytest.skip("未检测到 tesseract 可执行文件，跳过 OCR 集成测试。")

    p = mod.Preprocessor(ocr_lang="auto")
    res = p.process("q", [PNG_PATH_CHI])

    assert res.issues == []
    assert len(res.image_attachments) == 1
    text = res.image_attachments[0].content
    assert isinstance(text, str)

    # 打印
    dump_preview(text, "OCR")

    # 基础断言
    assert len(text.strip()) > 0, "OCR 结果为空，确认图片清晰度/对比度/字号。"

    # 可选：通过 expected 片段做更稳的匹配
    expected = load_expected_lines(PNG_PATH_CHI)
    if expected:
        nt = norm(text)
        for frag in expected:
            assert norm(frag) in nt, f"OCR 文本未包含期望片段：{frag!r}"

def test_spec_png_ocr_mix(tmp_path):
    if mod.Image is None or mod.pytesseract is None:
        pytest.skip("未安装 Pillow/pytesseract，跳过 OCR 集成测试。")
    if not has_tesseract():
        pytest.skip("未检测到 tesseract 可执行文件，跳过 OCR 集成测试。")

    p = mod.Preprocessor(ocr_lang="auto")
    res = p.process("q", [PNG_PATH_MIX])

    assert res.issues == []
    assert len(res.image_attachments) == 1
    text = res.image_attachments[0].content
    assert isinstance(text, str)

    # 打印
    dump_preview(text, "OCR")

    # 基础断言
    assert len(text.strip()) > 0, "OCR 结果为空，确认图片清晰度/对比度/字号。"

    # 可选：通过 expected 片段做更稳的匹配
    expected = load_expected_lines(PNG_PATH_MIX)
    if expected:
        nt = norm(text)
        for frag in expected:
            assert norm(frag) in nt, f"OCR 文本未包含期望片段：{frag!r}"