from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path

# ==========================================
# 🛡️ 安全限制常量定义 (防御 DoS 攻击)
# ==========================================
MAX_DOCUMENT_BYTES = 12 * 1024 * 1024  # 最大允许的原始文档大小：12 MB
MAX_EXTRACTED_CHARS = 4_000_000  # 提取出的文本最大字符数：400万字符 (防内存溢出)
MAX_PDF_PAGES = 500  # PDF 最大页数限制 (防超长循环解析)
MAX_DOCX_ENTRIES = 10_000  # DOCX 内部最大文件条目数 (防 ZIP 炸弹)
MAX_DOCX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024  # DOCX 解压后的最大允许大小：64 MB (防 ZIP 炸弹)


def _ensure_text_limit(text: str) -> str:
    """辅助函数：检查最终提取的文本是否超过字符数限制，超限则抛异常"""
    if len(text) > MAX_EXTRACTED_CHARS:
        raise ValueError(f"extracted text exceeds the {MAX_EXTRACTED_CHARS} character limit")
    return text


# ==========================================
# 📝 纯文本解码模块
# ==========================================
def decode_text(data: bytes) -> str:
    """
    尝试以多种编码将二进制数据解码为文本。
    支持带有 BOM 的 UTF-8、标准 UTF-8、中文 GB18030 以及兜底的 Latin-1。
    """
    # 检查前 8192 个字节中是否包含空字节 (\x00)。
    # 文本文件通常不含空字节，如果包含，说明大概率是图片、视频或可执行文件等二进制文件，直接拒绝。
    if b"\x00" in data[:8192]:
        raise ValueError("binary content cannot be indexed as text")

    # 依次尝试不同编码。utf-8-sig 用于处理带 BOM 的 Windows 文件，gb18030 兼容大部分中文文件。
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "latin-1"]:
        try:
            return _ensure_text_limit(data.decode(encoding))
        except UnicodeDecodeError:
            continue  # 如果解码失败，尝试下一种编码

    # 如果全部失败（极罕见情况），返回空字符串
    return ""


# ==========================================
# 📄 PDF 解析模块
# ==========================================
def _extract_pdf(data: bytes) -> str:
    # 局部导入 pypdf，避免在不需要处理 PDF 时拖慢程序启动速度
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))

    # 校验：页数是否超限
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"PDF exceeds the {MAX_PDF_PAGES} page limit")

    pages: list[str] = []
    extracted_chars = 0
    # 遍历每一页提取文本
    for page_number, page in enumerate(reader.pages, start=1):
        # extract_text 可能返回 None，所以使用 or "" 兜底
        text = (page.extract_text() or "").strip()
        if text:
            # 加入类似 HTML 注释的页码标记，方便后续溯源（比如在前端展示时提示这来自第几页）
            content = f"<!-- page: {page_number} -->\n{text}"

            # 累加字符数（+2 是因为后面用 "\n\n" 拼接会产生2个字符的损耗，这里提前算进去）
            extracted_chars += len(content) + (2 if pages else 0)

            # 实时校验：提取过程中一旦超限，立刻中断，防止内存被撑爆
            if extracted_chars > MAX_EXTRACTED_CHARS:
                raise ValueError(f"PDF text exceeds the {MAX_EXTRACTED_CHARS} character limit")
            pages.append(content)

    return _ensure_text_limit("\n\n".join(pages))


# ==========================================
# 📝 DOCX 解析模块 (将 Word 转为 Markdown)
# ==========================================
def _paragraph_as_markdown(paragraph) -> str:
    """将 Word 的段落转换为 Markdown 格式 (主要是为了保留标题和列表层级)"""
    text = (paragraph.text or "").strip()
    if not text:
        return ""

    style_name = str(getattr(getattr(paragraph, "style", None), "name", "") or "")

    # 匹配 Word 的标题样式，如 "Heading 1", "heading 2" 等，转换为 Markdown 的 "#", "##"
    heading = re.match(r"heading\s+(\d+)", style_name, re.IGNORECASE)
    if heading:
        # 限制最大级别为 6 级 (#)
        level = min(6, max(1, int(heading.group(1))))
        return f"{'#' * level} {text}"

    # 匹配列表样式，转换为 Markdown 列表
    lowered = style_name.lower()
    if "list bullet" in lowered:
        return f"- {text}"
    if "list number" in lowered:
        return f"1. {text}"

    # 普通段落直接返回文本
    return text


def _table_as_markdown(table) -> str:
    """将 Word 的表格转换为 Markdown 格式的表格"""
    # 提取表格内所有单元格的文本，并去掉单元格内的换行符 (因为 Markdown 表格单元格内不能直接换行)
    rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
    if not rows:
        return ""

    # 找出列数最多的一行，作为标准宽度 (处理合并单元格导致的列数不一致)
    width = max(len(row) for row in rows)
    # 用空字符串补齐其他行的缺失列，确保是一个完美的矩形表格
    normalized = [row + [""] * (width - len(row)) for row in rows]

    def render(row: list[str]) -> str:
        # 渲染单行，并转义单元格内原有的 "|" 符号，防止破坏 Markdown 表格结构
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"

    # 拼接成 Markdown 表格格式：表头 + 分割线 + 数据行
    return "\n".join([render(normalized[0]), render(["---"] * width), *(render(row) for row in normalized[1:])])


def _extract_docx(data: bytes) -> str:
    """解析 DOCX 文档"""
    # 1. 安全检查：DOCX 本质上是一个 ZIP 压缩包，必须先检查是否包含恶意构造的 ZIP 炸弹
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            infos = archive.infolist()
            # 校验内部文件数量
            if len(infos) > MAX_DOCX_ENTRIES:
                raise ValueError(f"DOCX exceeds the {MAX_DOCX_ENTRIES} entry limit")
            # 校验解压后总大小 (防止小体积大解压的 ZIP Bomb 耗尽磁盘或内存)
            uncompressed_size = sum(info.file_size for info in infos)
            if uncompressed_size > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"DOCX expanded size exceeds the {MAX_DOCX_UNCOMPRESSED_BYTES} byte limit"
                )
            # 校验是否加密（加密的无法解析）
            if any(info.flag_bits & 0x1 for info in infos):
                raise ValueError("encrypted DOCX files are not supported")
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid DOCX archive") from exc

    # 2. 局部导入 python-docx 库及其底层 XML 节点类型 (CT_P 代表段落, CT_Tbl 代表表格)
    import docx
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(BytesIO(data))
    blocks: list[str] = []
    extracted_chars = 0

    # 3. 遍历文档正文中的所有子节点 (这比直接遍历 paragraphs 更强大，能保留段落和表格的相对顺序)
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            content = _paragraph_as_markdown(Paragraph(child, document))
        elif isinstance(child, CT_Tbl):
            content = _table_as_markdown(Table(child, document))
        else:
            content = ""  # 忽略其他未知节点

        if content:
            # 同样进行字符数累加与实时限制
            extracted_chars += len(content) + (2 if blocks else 0)
            if extracted_chars > MAX_EXTRACTED_CHARS:
                raise ValueError(f"DOCX text exceeds the {MAX_EXTRACTED_CHARS} character limit")
            blocks.append(content)

    return _ensure_text_limit("\n\n".join(blocks))


# ==========================================
# 🚀 主入口函数
# ==========================================
def extract_document_text(filename: str, data: bytes) -> str:
    """
    根据文件后缀名路由到对应的解析器。
    如果非 PDF 或 DOCX，则默认当作纯文本处理。
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(data)
    if suffix == ".docx":
        return _extract_docx(data)
    return decode_text(data)