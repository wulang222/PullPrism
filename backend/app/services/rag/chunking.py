# 文档 / 代码分块器（详细中文注释版）
# 核心流程：识别文件类型 → 解析为带结构信息的 Block → 控制长度并打包为 Chunk → 附加检索元数据。
# Block 是内部临时对象（类似 Java 的 DTO）；Chunk 是最终返回的 dict（类似返回给调用方的 Map）。
# 本文件只增加解释性注释，不主动调整任何切分算法。

from __future__ import annotations

# 标准库分别负责：Python 语法树、CSV 读写、哈希、内存文本流、JSON、正则、数据类、路径、类型提示。
import ast
import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# 这些后缀走“按 Markdown 标题/段落/代码块解析”的路径。
# 注意：这里并没有真正解析 PDF/DOCX 二进制；调用方必须先把它们提取为文本。
MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx", ".pdf", ".docx"}
STRUCTURED_DATA_SUFFIXES = {".json", ".csv"}


@dataclass
class _Block:
    """尚未打包的文本单元，保存正文及其在源文档中的位置/层级。"""

    text: str
    # 原文的起止行号；JSON 重排后不再与原文件行号一一对应，可能为 None。
    start_line: int | None
    end_line: int | None
    # paragraph / heading / code_fence / json_value 等，决定超长内容如何继续切。
    kind: str = "paragraph"
    # 例如 ("第一章", "1.1 安装")；用于生成 Chunk 的标题前缀，增强检索上下文。
    section_path: tuple[str, ...] = ()
    # 例如 (("第一章", 0), ("1.1 安装", 1))；同名标题出现多次也能分开。
    section_instance: tuple[tuple[str, int], ...] = ()
    page: int | None = None
    # 独立字典实例，避免多个 Block 共享一个可变 metadata 对象。
    metadata: dict[str, Any] = field(default_factory=dict)


MAX_CHUNKS_PER_DOCUMENT = 5000


def _validate_limits(max_chars: int, overlap: int) -> None:
    """参数校验：单块长度必须 >0，重叠长度必须在 [0, max_chars) 范围内。"""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be between zero and max_chars")


def _normalize_text(text: str) -> str:
    """把 Windows 的 CRLF、旧 Mac 的 CR 统一成 \n，便于计算行号和切分。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _stable_id(*parts: object) -> str:
    """根据输入内容生成稳定 ID：参数不变 → SHA-256 摘要不变。"""
    # \x1f 是不常见的分隔符，降低拼接歧义；digest 截断成 24 个十六进制字符。
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _line_for_offset(text: str, offset: int, first_line: int | None) -> int | None:
    """把字符下标换成行号：首行号 + 该字符之前的换行数量。"""
    if first_line is None:
        return None
    return first_line + text.count("\n", 0, max(0, offset))


def _split_long_text(block: _Block, max_chars: int) -> list[_Block]:
    """把过长的单个 Block 拆小；尽量沿句末/换行切，实在不行按字符数硬切。"""
    if len(block.text) <= max_chars:
        # 已经满足长度，不需要创建新对象。
        return [block]

    # re.split 的几个边界：中文句末、英文句末+空白、连续换行。
    # 正向后顾 (?<=...) 只匹配切分位置，不把标点本身删掉。
    units = [
        unit
        for unit in re.split(r"(?<=[。！？!?])|(?<=[.!?])\s+|\n+", block.text)
        if unit and unit.strip()
    ]
    if len(units) <= 1:
        # 没找到有效的自然边界：每 max_chars 个字符切一个单元。
        units = [block.text[index : index + max_chars] for index in range(0, len(block.text), max_chars)]

    # 用贪心方式把多个短单元合并，尽量装满 max_chars。
    # 举例：max_chars=10、units=["你好。", "今天天气好。", "再见。"]，
    # 会按“当前块 + 换行 + 下一个单元”尝试，不够放才输出当前块。
    pieces: list[str] = []
    current = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > max_chars:
            # 单句就超长：先结算已积累内容，再按字符强制拆开。
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(unit[index : index + max_chars] for index in range(0, len(unit), max_chars))
            continue
        candidate = f"{current}\n{unit}" if current else unit
        if current and len(candidate) > max_chars:
            # 不能继续放入当前 piece，开启下一片。
            pieces.append(current)
            current = unit
        else:
            current = candidate
    if current:
        pieces.append(current)

    output: list[_Block] = []
    cursor = 0
    for piece in pieces:
        # 从上次切分终点开始回原文查找，以便尽量恢复行号。
        # 注意：pieces 可能重组空白；找不到原文片段时退回 cursor，行号为近似值。
        offset = block.text.find(piece, cursor)
        if offset < 0:
            offset = cursor
        end_offset = min(len(block.text), offset + len(piece))
        output.append(
            _Block(
                text=piece,
                # 通过字符偏移反推原始行号；复制标题、页号、metadata 等上下文。
                start_line=_line_for_offset(block.text, offset, block.start_line),
                end_line=_line_for_offset(block.text, max(offset, end_offset - 1), block.start_line),
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata=dict(block.metadata),
            )
        )
        cursor = end_offset
    return output


def _split_long_code_fence(block: _Block, max_chars: int) -> list[_Block]:
    """拆超长 Markdown 代码围栏：每片都重新包上 ```，尽量保持代码块可读。"""
    if len(block.text) <= max_chars:
        return [block]
    lines = block.text.splitlines()
    if not lines:
        return []

    if max_chars <= 8:
        # 太短了，连“```\n...\n```”的壳都装不下，退回普通文本切分。
        plain = _Block(
            text=block.text,
            start_line=block.start_line,
            end_line=block.end_line,
            kind="code_fragment",
            section_path=block.section_path,
            section_instance=block.section_instance,
            page=block.page,
            metadata=dict(block.metadata),
        )
        return _split_long_text(plain, max_chars)

    # 允许原围栏为 ```python / ~~~ 等；拆开后的围栏只保留前三个标记字符。
    # 因此切片不会保留原来的语言标签（如 python）。
    original_opener = lines[0] if lines[0].lstrip().startswith(("```", "~~~")) else "```"
    marker = original_opener.lstrip()[:3]
    opener = marker
    has_closer = len(lines) > 1 and lines[-1].lstrip().startswith(marker)
    body = lines[1:-1] if has_closer else lines[1:]
    closing = marker
    # 两个标记 + 两个换行需要预留字符数：6 + 2 = 8（对 ``` 而言）。
    # 比如 max_chars=30，则实际代码正文最多放 22 个字符。
    wrapper_size = len(opener) + len(closing) + 2
    body_limit = max_chars - wrapper_size

    if not body:
        # 只有围栏没有代码内容，输出空代码块。
        return [
            _Block(
                text=f"{opener}\n{closing}",
                start_line=block.start_line,
                end_line=block.end_line,
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata=dict(block.metadata),
            )
        ]

    output: list[_Block] = []
    current: list[str] = []
    current_size = 0
    first_body_line = (block.start_line + 1) if block.start_line is not None else None
    current_start = first_body_line

    def flush(end_line: int | None) -> None:
        """把当前积累的若干代码行输出为一个独立 Block。"""
        nonlocal current, current_size, current_start
        # nonlocal 类似“修改外层方法的局部变量”；不是创建 flush 自己的同名变量。
        if not current:
            return
        body_text = "\n".join(current)
        output.append(
            _Block(
                text=f"{opener}\n{body_text}\n{closing}",
                start_line=current_start,
                end_line=end_line,
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata={
                    **block.metadata,
                    # 额外记录“原始完整围栏”所在行，区别于拆出来的正文行号。
                    "fence_start_line": block.start_line,
                    "fence_end_line": block.end_line if has_closer else None,
                },
            )
        )
        current = []
        current_size = 0

    for body_index, line in enumerate(body):
        # enumerate(body) 得到 (0, 第一行), (1, 第二行)...；index 为源文档行号。
        index = (first_body_line + body_index) if first_body_line is not None else None
        if len(line) > body_limit:
            # 连一整行都放不下：先输出前面累积的行，再把当前行水平截成几段。
            flush((index - 1) if index is not None else None)
            for offset in range(0, len(line), body_limit):
                piece = line[offset : offset + body_limit]
                output.append(
                    _Block(
                        text=f"{opener}\n{piece}\n{closing}",
                        start_line=index,
                        end_line=index,
                        kind=block.kind,
                        section_path=block.section_path,
                        section_instance=block.section_instance,
                        page=block.page,
                        metadata={
                            **block.metadata,
                            "fence_start_line": block.start_line,
                            "fence_end_line": block.end_line if has_closer else None,
                        },
                    )
                )
            current_start = (index + 1) if index is not None else None
            continue
        line_size = len(line) + (1 if current else 0)
        # 第二行起需要多算一位换行符；超预算就先封口，再开始新块。
        if current and current_size + line_size > body_limit:
            flush((index - 1) if index is not None else None)
            current_start = index
        if not current:
            current_start = index
        current.append(line)
        current_size += line_size
    last_body_line = None
    if block.end_line is not None:
        last_body_line = block.end_line - (1 if has_closer else 0)
    flush(last_body_line)
    return output


def _expand_block(block: _Block, max_chars: int) -> list[_Block]:
    """统一入口：代码围栏走专用切分；其他块按句子/字符切分。"""
    if block.kind == "code_fence":
        return _split_long_code_fence(block, max_chars)
    return _split_long_text(block, max_chars)


def _overlap_blocks(blocks: list[_Block], overlap: int) -> list[_Block]:
    """截取上一个 Chunk 尾部的少量 Block，供下一个 Chunk 重复使用。"""
    # 比如上一块结尾是“保险等待期为90天。”，新块也带上这句话，能减少上下文断裂。
    # 从末尾倒序收集；不拿 code_fence 做重叠，避免混入不完整代码围栏。
    if overlap <= 0 or not blocks:
        return []
    selected: list[_Block] = []
    remaining = overlap
    for block in reversed(blocks):
        if not block.text or block.kind == "code_fence":
            continue
        # 两个 Block 最终用 \n\n 相连；所以选第二个及以后要预留 2 个字符。
        separator = 2 if selected else 0
        allowance = remaining - separator
        if allowance <= 0:
            break
        if len(block.text) <= allowance:
            # 剩余额度足以容纳整个块，直接放入重叠内容。
            selected.append(block)
            remaining -= len(block.text) + separator
            continue
        # 整个块放不下：只截末尾 allowance 个字符，再寻找自然语句边界。
        offset = len(block.text) - allowance
        lookback_start = max(0, offset - min(64, max(16, allowance * 2)))
        lookback = block.text[lookback_start:offset]
        sentence_boundaries = list(re.finditer(r"[。！？.!?；;]\s*", lookback))
        word_boundaries = list(re.finditer(r"\s+", lookback))
        if sentence_boundaries:
            # 优先从最近句号后开始截，避免把一句话从中间切开。
            fragment_offset = lookback_start + sentence_boundaries[-1].end()
        elif word_boundaries:
            # 没有句末就退到空白边界。
            fragment_offset = lookback_start + word_boundaries[-1].end()
        else:
            # 再不行往右看一小段，找空白/标点；仍找不到就按字符截。
            boundary_window = block.text[offset : offset + min(48, max(8, allowance // 3))]
            boundary = re.search(r"\s+|[。！？.!?；;]\s*", boundary_window)
            fragment_offset = offset + boundary.end() if boundary else offset
        selected.append(
            _Block(
                text=block.text[fragment_offset:],
                start_line=_line_for_offset(block.text, fragment_offset, block.start_line),
                end_line=block.end_line,
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                # 记录这是“重复的尾部上下文”，便于下游识别。
                metadata={**block.metadata, "overlap_fragment": True},
            )
        )
        remaining = 0
        break
    # 倒序选取，最后恢复原来的阅读顺序。
    return list(reversed(selected))


def _section_prefix(section_path: tuple[str, ...], page: int | None, limit: int | None = None) -> str:
    """生成层级标题前缀，如 '# 概述\n## 安装\nPage 3'，必要时截断。"""
    parts = [f"{'#' * min(index + 1, 6)} {title}" for index, title in enumerate(section_path)]
    if page is not None:
        parts.append(f"Page {page}")
    value = "\n".join(parts)
    if limit is None or len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    # rstrip() 去掉截断点前的尾部空白，最后追加 ... 表示还有内容。
    return value[: limit - 3].rstrip() + "..."


def _prefix_for_chunk(
    section_path: tuple[str, ...],
    page: int | None,
    max_chars: int,
    *,
    has_body: bool,
) -> str:
    """按长度预算决定 Chunk 能带多少标题/页码前缀。"""
    if not section_path and page is None:
        return ""
    if not has_body:
        limit = max_chars
    elif max_chars <= 3:
        limit = 0
    else:
        # 有正文时，标题最多占 512 字符、总长的 1/3，且给正文至少留下 3 位。
        limit = min(512, max_chars // 3, max_chars - 3)
    return _section_prefix(section_path, page, limit=limit) if limit > 0 else ""


def _join_block_text(blocks: list[_Block]) -> str:
    """用两个换行拼接多个 Block，跳过空正文（例如只有标题的占位块）。"""
    return "\n\n".join(block.text for block in blocks if block.text).strip()


def _emit_chunk(
    blocks: list[_Block],
    *,
    source_path: str,
    chunk_type: str,
    child_index: int,
    max_chars: int,
) -> dict[str, Any]:
    """把积累的 Block 转成最终对外返回的 Chunk 字典，连同检索元数据一起输出。"""
    first = blocks[0]
    body = _join_block_text(blocks)
    prefix = _prefix_for_chunk(first.section_path, first.page, max_chars, has_body=bool(body))
    content = f"{prefix}\n\n{body}" if prefix and body else (prefix or body)
    # parent_id 标识“同一逻辑章节/同一页”；同一个 parent 下可以有多个子 Chunk。
    parent_id = _stable_id(
        "parent",
        source_path,
        first.section_instance or tuple((title, 0) for title in first.section_path),
        first.page,
    )
    # 只要一个 Chunk 包含 Markdown 代码围栏，返回类型就标记 markdown_code。
    effective_type = "markdown_code" if any(block.kind == "code_fence" for block in blocks) else chunk_type
    metadata: dict[str, Any] = {}
    for block in blocks:
        # 合并 metadata；同名 key 时，后面的 Block 覆盖前面的值。
        metadata.update(block.metadata)
    return {
        "content": content,
        # chunk_id 由 parent_id + 子序号 + 正文决定，既反映位置又反映内容。
        "chunk_id": _stable_id("chunk", parent_id, child_index, content),
        "parent_id": parent_id,
        "child_index": child_index,
        "chunk_type": effective_type,
        "section_path": list(first.section_path),
        "section_title": first.section_path[-1] if first.section_path else None,
        "page": first.page,
        # 多个 Block 合并后，行号取最早开始行到最晚结束行。
        "start_line": min((block.start_line for block in blocks if block.start_line is not None), default=None),
        "end_line": max((block.end_line for block in blocks if block.end_line is not None), default=None),
        **metadata,
    }


def _pack_blocks(
    blocks: list[_Block],
    *,
    source_path: str,
    max_chars: int,
    overlap: int,
    chunk_type: str,
) -> list[dict[str, Any]]:
    """核心打包器：同章节/同页的 Block 尽量合并；超长换块；可附带尾部重叠。"""
    # 一个实际的打包例子（以下数字只是说明流程）：
    # max_chars=100、overlap=20；章节 A 有 Block 1(40字)、Block 2(35字)、Block 3(50字)。
    # 1) Block 1 放进 current；2) Block 2 放得下，就合并（中间加两个换行）；
    # 3) Block 3 再加入会超长 → flush() 输出当前 Chunk；
    # 4) 从旧 current 尾部提取最多约 20 字上下文，再加 Block 3 → 形成下一个 Chunk。
    # 如果下一块来自章节 B，即使还没满，也先 flush，绝不与章节 A 混装。
    chunks: list[dict[str, Any]] = []
    # current 是“正在组装、尚未输出”的一组 Block。
    current: list[_Block] = []
    # key=(章节实例, 页号)；控制哪些 Block 可以被放在同一个 Chunk。
    current_key: tuple[tuple[tuple[str, int], ...], int | None] | None = None
    # 每个 key 单独编号：章节 A 的 Chunk 是 A0、A1...，章节 B 从 B0 重新开始。
    child_counts: dict[tuple[tuple[tuple[str, int], ...], int | None], int] = {}

    def flush() -> None:
        """将 current 的内容正式生成并追加一个 Chunk；不负责把 current 清空。"""
        nonlocal current
        if not current or current_key is None:
            return
        # dict.get(key, 0)：找不到该章节的计数器时，默认从 0 开始。
        child_index = child_counts.get(current_key, 0)
        chunks.append(
            _emit_chunk(
                current,
                source_path=source_path,
                chunk_type=chunk_type,
                child_index=child_index,
                max_chars=max_chars,
            )
        )
        child_counts[current_key] = child_index + 1

    for raw_block in blocks:
        # raw_block 是解析器输出的原始单元，可能还大于 max_chars。
        key = (raw_block.section_instance, raw_block.page)
        prefix = _prefix_for_chunk(
            raw_block.section_path,
            raw_block.page,
            max_chars,
            has_body=bool(raw_block.text),
        )
        prefix_size = len(prefix)
        # Chunk 总预算同时包括“标题前缀 + 两个换行 + 正文”。
        # 例如 max_chars=100、标题长 18，那么正文预算=100-18-2=80。
        body_limit = max(1, max_chars - prefix_size - (2 if prefix_size else 0))
        # 超长 raw_block 先细切为若干 block；普通段落、代码围栏采用不同策略。
        for block in _expand_block(raw_block, body_limit):
            key = (block.section_instance, block.page)
            if current and key != current_key:
                # 章节/页码变了：强制结束旧块；不允许跨 section_instance 合并。
                flush()
                current = []
            current_key = key
            # 假设把 block 放进去，先计算“试装”的总长度；不立即修改 current。
            candidate_body = _join_block_text([*current, block])
            prefix = _prefix_for_chunk(
                block.section_path,
                block.page,
                max_chars,
                has_body=bool(candidate_body),
            )
            candidate_size = len(candidate_body) + len(prefix) + (2 if prefix else 0)
            if current and candidate_size > max_chars:
                # 试装超限：先输出之前 current 的内容。
                previous = list(current)
                flush()
                # 新 Chunk 不是从零开始，而是可选地重复上个 Chunk 的尾部上下文。
                current = _overlap_blocks(previous, overlap)
                while current:
                    # 可能“重叠内容 + 新 block”仍超长：不断丢弃最旧的重叠 Block。
                    # 关键：先保障 max_chars，再尽量保留 overlap。
                    candidate_body = _join_block_text([*current, block])
                    if len(candidate_body) + len(prefix) + (2 if prefix else 0) <= max_chars:
                        break
                    current.pop(0)
            # 当前 block 最终加入；若超限时重叠被清空，这里也只会加新 block。
            current.append(block)
    # 循环结束后 current 里还有最后一块，必须输出，否则最后的数据会丢失。
    flush()
    # 最终防御性检查：所有 Chunk 都不能超过 max_chars。
    if any(len(chunk["content"]) > max_chars for chunk in chunks):
        raise AssertionError("chunk content exceeded max_chars")
    return chunks


def _plain_blocks(text: str) -> list[_Block]:
    """把普通文本按空行划分为段落 Block；主要作为 JSON/CSV 解析失败时的兜底。"""
    lines = text.splitlines()
    blocks: list[_Block] = []
    current: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        """结束当前段落，记录起止行号；空段落直接忽略。"""
        nonlocal current
        content = "\n".join(current).strip()
        if content:
            blocks.append(_Block(content, start_line, max(start_line, end_line)))
        current = []

    for line_number, line in enumerate(lines, start=1):
        # enumerate(..., start=1) 直接得到符合人类习惯的 1-based 行号。
        if not line.strip():
            flush(line_number - 1)
            continue
        if not current:
            start_line = line_number
        current.append(line)
    flush(len(lines))
    return blocks


def _plain_chunks(
    text: str,
    *,
    source_path: str,
    max_chars: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """纯文本滑动窗口切分：先按 max_chars 取一窗，再尽量回退到换行边界。"""
    # 与 _pack_blocks 的结构化处理不同：这里直接操作字符下标 cursor/end。
    # 例：max_chars=10、overlap=3，第一次 [0:10]，下次通常从 10-3=7 开始。
    chunks: list[dict[str, Any]] = []
    parent_id = _stable_id("parent", source_path, "plain")
    cursor = 0
    text_length = len(text)
    while cursor < text_length:
        # 理论终点为 cursor + max_chars；末尾不够则取整段文本的长度。
        end = min(text_length, cursor + max_chars)
        if end < text_length:
            # 只在窗口后半段找自然边界，避免找到太靠前的换行导致块过小。
            # 例如 cursor=0、max_chars=100 → 搜索区间 [50, 100)。
            search_start = cursor + max(1, max_chars // 2)
            candidates = [
                # rfind(sub, start, end) 在指定区间从右往左找，返回最后一次出现的位置；没有则 -1。
                # 双换行优先更像段落边界，但代码这里实际取的是“位置更靠右者”。
                text.rfind("\n\n", search_start, end),
                text.rfind("\n", search_start, end),
            ]
            boundary = max(candidates)
            if boundary > cursor:
                # 让本块包含命中的换行符：双换行 +2，单换行 +1。
                # 例如 boundary=73 且 text[73:75] == '\n\n' → end=75。
                end = boundary + (2 if text.startswith("\n\n", boundary) else 1)

        raw = text[cursor:end]
        # 两端空白不进 content，但要修正它们对应的源文本偏移/行号。
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw.rstrip())
        content = raw[left_trim:right_trim]
        if content:
            start_offset = cursor + left_trim
            end_offset = cursor + right_trim
            child_index = len(chunks)
            chunks.append(
                {
                    "content": content,
                    "chunk_id": _stable_id("chunk", parent_id, child_index, content),
                    "parent_id": parent_id,
                    "child_index": child_index,
                    "chunk_type": "text",
                    "section_path": [],
                    "section_title": None,
                    "page": None,
                    "start_line": _line_for_offset(text, start_offset, 1),
                    "end_line": _line_for_offset(text, max(start_offset, end_offset - 1), 1),
                }
            )
            if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
                raise ValueError(f"document exceeds the {MAX_CHUNKS_PER_DOCUMENT} chunk limit")
        if end >= text_length:
            break
        # 下一窗从上个窗口末尾往前退 overlap 个字符，制造字符级重叠。
        # max(cursor+1, ...) 保证游标至少前进 1，避免 overlap 很大时死循环。
        cursor = max(cursor + 1, end - overlap)
    return chunks


def _markdown_blocks(text: str) -> list[_Block]:
    """按 Markdown 标题、空行、代码围栏、页码标记提取带章节归属的 Block。"""

    lines = text.splitlines()
    blocks: list[_Block] = []
    # 标题栈：每项为 (标题级别, 标题文字, 同一父标题下的出现次数)。
    # 例如读到 '# 第一章'、'## 安装'，headings 依次为 [(1,'第一章',0),(2,'安装',0)]。
    headings: list[tuple[int, str, int]] = []
    # 同名标题的计数器，防止两处“## 简介”被当成同一章节。
    heading_occurrences: dict[tuple[tuple[tuple[str, int], ...], int, str], int] = {}
    current_page: int | None = None
    # current 存当前正在积累的段落/代码块的原始行。
    current: list[str] = []
    current_start = 1
    current_kind = "paragraph"
    fence_marker: str | None = None
    # 匹配 '# 标题' 到 '###### 标题'；group(1) 是 # 序列，group(2) 是标题内容。
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
    # 匹配 <!-- page: 3 -->；通常由 PDF/DOCX 文本提取流程插入。
    page_re = re.compile(r"^<!--\s*page\s*:\s*(\d+)\s*-->$", re.IGNORECASE)

    def section_path() -> tuple[str, ...]:
        """当前标题路径，供检索时显示，例如 ('第1章', '安装')。"""
        return tuple(title for _, title, _ in headings)

    def section_instance() -> tuple[tuple[str, int], ...]:
        """当前章节的唯一实例路径：标题文字 + 第几次出现。"""
        return tuple((title, occurrence) for _, title, occurrence in headings)

    def flush(end_line: int) -> None:
        """将当前段落/围栏代码片段输出，并携带当时的标题和页码信息。"""
        nonlocal current
        content = "\n".join(current).strip()
        if content:
            blocks.append(
                _Block(
                    content,
                    current_start,
                    max(current_start, end_line),
                    kind=current_kind,
                    section_path=section_path(),
                    section_instance=section_instance(),
                    page=current_page,
                )
            )
        current = []

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if fence_marker is not None:
            # 已经进入 ```...``` 代码围栏时，不把里面的 '# ...' 误当成标题。
            current.append(line)
            if stripped.startswith(fence_marker):
                # 遇到结尾围栏：整体结束为一个 kind='code_fence' 的 Block。
                flush(line_number)
                fence_marker = None
                current_kind = "paragraph"
            continue

        page_match = page_re.match(stripped)
        if page_match:
            # 进入下一页前先结算上一页内容；后续 Block 使用新的 page 值。
            flush(line_number - 1)
            current_page = int(page_match.group(1))
            continue

        heading_match = heading_re.match(line)
        if heading_match:
            # 新标题出现：先结算上一段落，再调整标题栈。
            flush(line_number - 1)
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            while headings and headings[-1][0] >= level:
                # 比如从 '## 安装' 走到 '## 使用'，移除旧二级标题；
                # 从 '### 步骤' 走到 '# 第二章'，则会弹掉三级、二级和一级标题。
                headings.pop()
            parent_instance = section_instance()
            occurrence_key = (parent_instance, level, title)
            occurrence = heading_occurrences.get(occurrence_key, 0)
            heading_occurrences[occurrence_key] = occurrence + 1
            headings.append((level, title, occurrence))
            # 标题自身不进入正文；添加空 Block 作为“章节开始”的标记。
            # 即使某标题后没有正文，它也能让打包器感知章节边界。
            blocks.append(
                _Block(
                    "",
                    line_number,
                    line_number,
                    kind="heading",
                    section_path=section_path(),
                    section_instance=section_instance(),
                    page=current_page,
                    metadata={"heading_line": line_number},
                )
            )
            continue

        if stripped.startswith(("```", "~~~")):
            # 代码块从这里开始，记录它的起始行并转入围栏模式。
            flush(line_number - 1)
            current_start = line_number
            current_kind = "code_fence"
            fence_marker = stripped[:3]
            current = [line]
            continue

        if not stripped:
            # 普通段落遇到空行即结束。
            flush(line_number - 1)
            continue
        if not current:
            current_start = line_number
        current.append(line)
    # 最后一段即使文件末尾没有空行，也要手动结算。
    flush(len(lines))
    return blocks


def _json_blocks(text: str, max_chars: int) -> list[_Block]:
    """JSON 结构切分：能整体放下则整体返回，否则按最外层属性/数组元素拆。"""
    try:
        # json.loads: JSON 字符串 -> Python 对象。
        # 例如 '{"age":28,"city":"北京"}' -> dict {'age':28, 'city':'北京'}。
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        # JSON 格式损坏则不抛错，改成按空行分段的普通文本处理。
        return _plain_blocks(text)

    # json.dumps: Python 对象 -> 格式化 JSON 字符串；ensure_ascii=False 保留中文。
    # indent=2 表示每层缩进 2 个空格，因此格式化后通常比原始 JSON 更长。
    formatted = json.dumps(value, ensure_ascii=False, indent=2)
    if len(formatted) <= max_chars:
        # 整份 JSON 足够小：只产生一个 Block，json_path='$' 表示根节点。
        return [
            _Block(
                formatted,
                1,
                max(1, len(text.splitlines())),
                kind="json_value",
                metadata={"json_path": "$"},
            )
        ]

    blocks: list[_Block] = []
    if isinstance(value, dict):
        # 顶层对象举例：{"user":{"name":"小伍"},"orders":[1,2]}
        # 转成 [ ('$.user', {'user': {'name':'小伍'}}),
        #         ('$.orders', {'orders':[1,2]}) ]。
        # 注意每个 item 仍是“带原属性名”的独立对象，不只保留 value。
        items = [(f"$.{key}", {key: item}) for key, item in value.items()]
    elif isinstance(value, list):
        # 顶层数组举例：[{'id':1},{'id':2}] → '$[0]'、'$[1]' 各一个单元。
        items = [(f"$[{index}]", item) for index, item in enumerate(value)]
    else:
        # 顶层是数字/字符串/布尔值等基本类型，根路径始终为 '$'。
        items = [("$", value)]
    for json_path, item in items:
        # 每个顶层属性/元素生成一个 Block，路径同时写入 section_path 和 metadata。
        # section_instance 不同 → _pack_blocks 不会把 $.user 与 $.orders 混装。
        blocks.append(
            _Block(
                json.dumps(item, ensure_ascii=False, indent=2),
                None,
                None,
                kind="json_value",
                section_path=(json_path,),
                section_instance=((json_path, 0),),
                metadata={"json_path": json_path},
            )
        )
    return blocks


def _csv_chunks(text: str, source_path: str, max_chars: int, overlap: int) -> list[dict[str, Any]]:
    """CSV 按“表头 + 若干数据行”打包；一行超长时也支持拆成多个片段。"""
    try:
        # CSV 某个字段可能很长；先适当调高 csv 模块的单字段长度上限。
        if len(text) > csv.field_size_limit():
            csv.field_size_limit(len(text))
        # io.StringIO 把字符串包装成类文件对象；csv.reader 正确处理引号、逗号、字段内换行。
        # 例如 'name,city\n"张,三",北京' 会解析成 ['张,三', '北京']，不会误切内部逗号。
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error:
        # CSV 语法解析失败就退化为普通段落 Block，再统一打包。
        return _pack_blocks(
            _plain_blocks(text),
            source_path=source_path,
            max_chars=max_chars,
            overlap=overlap,
            chunk_type="text",
        )
    if not rows:
        return []

    def render(row: list[str]) -> str:
        """把一行的字段列表重新序列化为合法 CSV 文本（需要时自动加引号）。"""
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="").writerow(row)
        return buffer.getvalue()

    # 默认第一行是表头；后续每个 Chunk 优先重复携带表头，方便理解列名。
    header = render(rows[0])
    chunks: list[dict[str, Any]] = []
    parent_id = _stable_id("parent", source_path, "csv")

    def append_chunk(
        content: str,
        *,
        row_start: int,
        row_end: int,
        chunk_type: str = "csv_rows",
        row_part: int | None = None,
        row_parts: int | None = None,
        precise_lines: bool = True,
    ) -> None:
        """局部统一出口：填充 Chunk 内容、CSV 行范围、分片号及稳定 ID。"""
        child_index = len(chunks)
        chunks.append(
            {
                "content": content,
                "chunk_id": _stable_id("chunk", parent_id, child_index, content),
                "parent_id": parent_id,
                "child_index": child_index,
                "chunk_type": chunk_type,
                "section_path": [],
                "section_title": None,
                "page": None,
                "start_line": row_start if precise_lines else None,
                "end_line": row_end if precise_lines else None,
                "row_start": row_start,
                "row_end": row_end,
                # 一行拆多片时，row_part=当前是第几片，row_parts=总片数。
                **({"row_part": row_part, "row_parts": row_parts} if row_part is not None else {}),
            }
        )
        if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
            raise ValueError(f"document exceeds the {MAX_CHUNKS_PER_DOCUMENT} chunk limit")

    if header and len(header) + 1 >= max_chars:
        # 表头自己就接近/超过预算：先把表头按 max_chars 硬切成专门的 Chunk。
        # 此时数据行的 Chunk 不再重复表头，否则留不出足够的数据空间。
        header_parts = [header[index : index + max_chars] for index in range(0, len(header), max_chars)] or [""]
        for part_index, part in enumerate(header_parts):
            append_chunk(
                part,
                row_start=1,
                row_end=1,
                chunk_type="csv_header_part",
                row_part=part_index + 1,
                row_parts=len(header_parts),
                precise_lines="\n" not in part,
            )
        header_prefix = ""
    else:
        header_prefix = header

    if not rows[1:]:
        # 文件只有表头，且表头之前没单独输出过时，仍要生成一个 Chunk。
        if not chunks:
            append_chunk(header, row_start=1, row_end=1, precise_lines="\n" not in header)
        return chunks

    # 每个数据 Chunk 的容量 = max_chars - 表头 - 一次换行。
    # 例：max_chars=100，表头 19 字 → 每片数据最多 80 字。
    prefix_size = len(header_prefix) + (1 if header_prefix else 0)
    row_budget = max_chars - prefix_size
    # records 里每项： (数据行字符串, CSV 逻辑行号, 当前分片号, 总分片数, 能否精确标记物理行号)。
    records: list[tuple[str, int, int | None, int | None, bool]] = []
    for row_number, row in enumerate(rows[1:], start=2):
        rendered = render(row)
        # 一行太长就按 row_budget 硬切；后续 row_part/row_parts 能表明属于同一行。
        parts = [rendered[index : index + row_budget] for index in range(0, len(rendered), row_budget)] or [""]
        for part_index, part in enumerate(parts):
            # CSV 字段允许嵌入换行，因此第 N 条记录不一定是原文件第 N 个物理行。
            # 只要 rendered 含换行，就不把该行号作为精确 start_line/end_line。
            records.append((part, row_number, part_index + 1 if len(parts) > 1 else None, len(parts) if len(parts) > 1 else None, "\n" not in rendered))

    start = 0
    while start < len(records):
        # 从 records[start] 开始，贪心收集尽可能多的完整记录/记录分片。
        selected: list[tuple[str, int, int | None, int | None, bool]] = []
        index = start
        while index < len(records):
            candidate_rows = [item[0] for item in selected]
            candidate_rows.append(records[index][0])
            candidate = "\n".join([header_prefix, *candidate_rows]) if header_prefix else "\n".join(candidate_rows)
            if selected and len(candidate) > max_chars:
                # 再加一条就超长，停止；下一轮处理尚未加入的这一条。
                break
            selected.append(records[index])
            index += 1
        selected_text = [item[0] for item in selected]
        content = "\n".join([header_prefix, *selected_text]) if header_prefix else "\n".join(selected_text)
        row_start = min(item[1] for item in selected)
        row_end = max(item[1] for item in selected)
        # 只有当前 Chunk 恰好只有一个 record 时，才能把它的分片编号直接写进 Chunk。
        row_part = selected[0][2] if len(selected) == 1 else None
        row_parts = selected[0][3] if len(selected) == 1 else None
        append_chunk(
            content,
            row_start=row_start,
            row_end=row_end,
            row_part=row_part,
            row_parts=row_parts,
            precise_lines=all(item[4] for item in selected),
        )
        if index >= len(records):
            break
        if overlap <= 0:
            start = index
            continue
        # 需要重叠时，不按字符截断记录，而是从 selected 的末尾按“整条记录”倒序回收。
        # 例如 selected=[第2行,第3行,第4行]，末尾两行接近 overlap，下一块会从第3行重新装。
        size = 0
        overlap_records = 0
        for record in reversed(selected):
            record_size = len(record[0])
            if overlap_records and size + record_size > overlap:
                break
            size += record_size
            overlap_records += 1
            if size >= overlap:
                break
        # 至少前进一步，保证循环不会永远重复同一个 record。
        start = max(start + 1, index - overlap_records)
    if any(len(chunk["content"]) > max_chars for chunk in chunks):
        raise AssertionError("CSV chunk content exceeded max_chars")
    return chunks


def chunk_document(
    text: str,
    source_path: str = "document.txt",
    max_chars: int = 1600,
    overlap: int = 160,
) -> list[dict[str, Any]]:
    """文档切分的总入口：按文件后缀选策略，最终返回带检索元数据的 Chunk 列表。

    Java 类比：这是一个对外提供服务的 Service 方法，内部根据 suffix 分派给不同解析器。
    max_chars 是单块的最大字符数（不是 token），overlap 是相邻块重复的上下文预算。
    """

    # 1. 参数合法性、空文档检查和统一换行符。
    _validate_limits(max_chars, overlap)
    if not text:
        return []
    normalized = _normalize_text(text)
    if not normalized.strip():
        return []

    # 2. 取小写文件后缀，如 'docs/readme.MD' -> '.md'。
    suffix = Path(source_path).suffix.lower()
    if suffix == ".csv":
        # CSV 有自己的行/表头打包方式，直接返回。
        return _csv_chunks(normalized, source_path, max_chars, overlap)
    if suffix in MARKDOWN_SUFFIXES:
        # Markdown/PDF/DOCX 的已提取文本：按标题/段落/代码围栏解析成 Block。
        blocks = _markdown_blocks(normalized)
        chunk_type = "markdown_section"
    elif suffix == ".json":
        # JSON：先 loads 成对象，按顶层属性/元素解析成 Block。
        blocks = _json_blocks(normalized, max_chars)
        chunk_type = "json_value"
    else:
        # 其他（例如 txt）：走纯文本滑动窗口，不经过 _pack_blocks。
        return _plain_chunks(
            normalized,
            source_path=source_path,
            max_chars=max_chars,
            overlap=overlap,
        )
    # 3. 所有结构化 Block 统一进入打包器：控制长度、章节边界、重叠和元数据。
    chunks = _pack_blocks(
        blocks,
        source_path=source_path,
        max_chars=max_chars,
        overlap=overlap,
        chunk_type=chunk_type,
    )
    if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
        raise ValueError(f"document exceeds the {MAX_CHUNKS_PER_DOCUMENT} chunk limit")
    return chunks


def chunk_text(text: str, max_chars: int = 1600, overlap: int = 160) -> list[str]:
    """给旧调用方保留的简化 API：只返回分块字符串，不返回 metadata。"""

    # 列表推导式相当于 Java 的 stream().map(chunk -> chunk.get("content")).toList()。
    return [
        chunk["content"]
        for chunk in chunk_document(text, "document.txt", max_chars=max_chars, overlap=overlap)
    ]


def _line_windows(
    lines: list[str],
    start_line: int,
    end_line: int,
    max_chars: int,
    overlap_lines: int = 6,
) -> list[tuple[str, int, int]]:
    """按源码的“行”构造窗口，返回 (内容, 起始行号, 结束行号)。"""
    # 代码和普通文本不同：希望优先保留完整行和原始行号，方便回跳源码。
    output: list[tuple[str, int, int]] = []
    # 外部输入行号从 1 开始，列表下标从 0 开始，因此先减 1。
    cursor = max(0, start_line - 1)
    stop = min(len(lines), end_line)
    while cursor < stop:
        if len(lines[cursor]) > max_chars:
            # 仅这一行就超过预算：不得不按字符切，但每个分片仍归属原来的行号。
            for offset in range(0, len(lines[cursor]), max_chars):
                output.append((lines[cursor][offset : offset + max_chars], cursor + 1, cursor + 1))
            cursor += 1
            continue
        size = 0
        end = cursor
        while end < stop:
            # 新增一行时，除了行内容，还要计算与上一行之间的 '\n'。
            line_size = len(lines[end]) + (1 if end > cursor else 0)
            if end > cursor and size + line_size > max_chars:
                break
            size += line_size
            end += 1
        if end > cursor:
            # lines[cursor:end] 的 end 是“不含 end 的右边界”；返回行号 end 则是含末行。
            content = "\n".join(lines[cursor:end]).strip("\n")
            if content:
                output.append((content, cursor + 1, end))
        if end >= stop:
            break
        # 下一窗回退 overlap_lines 行（默认 6），但至少前进一行。
        # 注意：这是“行数重叠”，与文档切分的“字符重叠”不同。
        cursor = max(cursor + 1, end - overlap_lines)
    return output


def _python_regions(text: str, max_chars: int) -> list[tuple[int, int, str, str]]:
    """用 Python AST 找出类/函数所在行范围，尽量按真正的语法结构切源码。"""
    # AST（抽象语法树）类似 Java 的 JavaParser：比靠正则匹配 def/class 更懂代码嵌套关系。
    # 每个 region 是 (start_line, end_line, symbol_name, symbol_kind)。
    try:
        # 例如 `def f(): pass` 会解析成 Module -> FunctionDef，而不是纯字符串。
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        # 语法不完整（如只截取了一半的代码）会解析失败，调用方改用整个 module 按行切的兜底策略。
        return []
    lines = text.splitlines()
    regions: list[tuple[int, int, str, str]] = []
    cursor = 1

    def node_start(node: ast.AST) -> int:
        """获取声明开始行；有 @decorator 时，取最上面装饰器所在行。"""
        decorators = getattr(node, "decorator_list", []) or []
        positions = [getattr(node, "lineno", 1), *(getattr(item, "lineno", 1) for item in decorators)]
        return min(positions)

    def append_node(node: ast.AST, parent: str | None = None) -> None:
        """把顶层声明登记成 region；过长的类进一步拆成类头和各方法。"""
        start = node_start(node)
        # ast 节点的 end_lineno 表示结束行；没有则退回开始行。
        end = int(getattr(node, "end_lineno", start) or start)
        name = str(getattr(node, "name", "module"))
        # parent 在此处是可选的名字前缀，示例 'MyClass.run'。
        symbol = f"{parent}.{name}" if parent else name
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        if isinstance(node, ast.ClassDef) and len("\n".join(lines[start - 1 : end])) > max_chars:
            # 类很长时，如果里面有方法，就按“类声明区 + 每个方法”细拆。
            # 这里仅对类里的第一层方法切分，不递归剖析方法内部的语句。
            methods = [
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                method_cursor = start
                first_method_start = node_start(methods[0])
                if first_method_start > method_cursor:
                    # 把 class X:、类 docstring、类字段等首方法之前的区域归为 class_header。
                    regions.append((method_cursor, first_method_start - 1, symbol, "class_header"))
                for index, method in enumerate(methods):
                    method_start = node_start(method)
                    # 一个方法的区域结束于下一个方法之前；最后一个延伸到整个类的结尾。
                    # 因而方法之间的空行/类属性也可能包含在该区域里。
                    next_start = node_start(methods[index + 1]) if index + 1 < len(methods) else end + 1
                    regions.append((method_start, next_start - 1, f"{symbol}.{method.name}", "method"))
                return
        regions.append((start, end, symbol, kind))

    # 只选顶层的函数/异步函数/类；import、常量、执行语句算 module 区域。
    top_level = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for node in top_level:
        start = node_start(node)
        end = int(getattr(node, "end_lineno", start) or start)
        if start > cursor:
            # 声明之前的 import、注释等代码也要保留下来，不能丢。
            regions.append((cursor, start - 1, "module", "module"))
        append_node(node)
        cursor = max(cursor, end + 1)
    if cursor <= len(lines):
        # 最后一个定义之后的其他顶层代码。
        regions.append((cursor, len(lines), "module", "module"))
    return [region for region in regions if region[0] <= region[1]]


def _generic_symbol(line: str, suffix: str) -> tuple[str, str] | None:
    """其他语言的轻量声明识别：正则找到类、函数或类型名；匹配不到返回 None。"""
    # 这里只看“一行是否像声明的开头”，不是完整编译器，因此复杂跨行语法可能漏掉。
    patterns: list[tuple[re.Pattern[str], str]] = []
    if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        # JS/TS：识别 function/class/interface/type/enum 和部分函数型 const 声明。
        patterns = [
            (
                re.compile(
                    r"^\s*(?:export\s+(?:default\s+)?)?(?:(?:async\s+)?function|class|interface|type|enum)\s+([A-Za-z_$][\w$]*)"
                ),
                "symbol",
            ),
            (
                re.compile(
                    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)"
                ),
                "function",
            ),
        ]
    elif suffix == ".go":
        # Go：func、type 等结构。
        patterns = [(re.compile(r"^\s*(?:func\s+(?:\([^)]*\)\s*)?|type\s+)([A-Za-z_]\w*)"), "symbol")]
    elif suffix == ".rs":
        # Rust：fn、struct、enum、trait、impl 等结构。
        patterns = [
            (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl)\s+([A-Za-z_]\w*)"), "symbol")
        ]
    elif suffix in {".java", ".kt", ".kts", ".cs", ".cpp", ".c", ".h", ".hpp", ".php", ".rb", ".swift", ".scala"}:
        # Java 等：目前主要识别类型声明（class/interface/enum/...），
        # 普通 Java 方法不会被这个正则单独识别，所以通常按类声明所在区域处理。
        patterns = [
            (re.compile(r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|sealed|data|open)\s+)*(?:class|interface|enum|struct|trait|record)\s+([A-Za-z_]\w*)"), "type")
        ]
    elif suffix == ".sql":
        # SQL：识别 CREATE TABLE/VIEW/FUNCTION/PROCEDURE 等对象声明。
        patterns = [(re.compile(r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE|TABLE|VIEW)\s+([\w.]+)", re.IGNORECASE), "sql_object")]
    elif suffix in {".sh", ".bash", ".zsh", ".ps1"}:
        # Shell/PowerShell：匹配函数开头。
        patterns = [(re.compile(r"^\s*(?:function\s+)?([A-Za-z_]\w*)\s*(?:\(\))?\s*\{"), "function")]
    for pattern, kind in patterns:
        match = pattern.match(line)
        if match:
            # group(1) 是捕获到的名字；kind 是类型（symbol/function/type/...）。
            return match.group(1), kind
    return None


def _generic_regions(text: str, suffix: str) -> list[tuple[int, int, str, str]]:
    """把非 Python 源文件按声明行分段；某段一直延伸到下一个声明前。"""
    lines = text.splitlines()
    starts: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(lines, start=1):
        # 把所有识别到的声明起点先收集起来，比如 Java 两个类的起始行 8、75。
        symbol = _generic_symbol(line, suffix)
        if symbol:
            starts.append((line_number, symbol[0], symbol[1]))
    if not starts:
        # 一个声明也找不到：整个文件视为 module，后续再按行窗口控制大小。
        return [(1, len(lines), "module", "module")] if lines else []

    regions: list[tuple[int, int, str, str]] = []
    if starts[0][0] > 1:
        # import/package/文件注释等前置部分也不能丢。
        regions.append((1, starts[0][0] - 1, "module", "module"))
    for index, (start, name, kind) in enumerate(starts):
        # 不解析花括号的真实闭合位置，仅以“下一个匹配到的声明”为边界。
        end = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        regions.append((start, end, name, kind))
    return regions


def _language_for_path(path: str) -> str:
    """把后缀转为语言标签，比如 '.py' -> 'py'；无后缀则用文件名。"""
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or Path(path).name.lower()


def chunk_code_text(text: str, path: str, max_chars: int = 2200) -> list[dict[str, Any]]:
    """源码分块总入口：文档类委托 chunk_document；源码按语法符号 → 行窗口切分。"""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return []
    suffix = Path(path).suffix.lower()
    language = _language_for_path(path)

    if suffix in MARKDOWN_SUFFIXES | STRUCTURED_DATA_SUFFIXES | {".txt"}:
        # 这里的 | 是 Python 集合并集；文档/JSON/CSV/TXT 使用文档切分策略。
        # 调用文档切分时自动给小 overlap：不超过 160 且不超过 max_chars 的 1/10。
        document_chunks = chunk_document(normalized, path, max_chars=max_chars, overlap=max(0, min(160, max_chars // 10)))
        output: list[dict[str, Any]] = []
        for chunk in document_chunks:
            # 文档没有函数名，所以拿末级章节名当 symbol，没有则叫 document。
            symbol = chunk.get("section_title") or "document"
            header = f"File: {path}\nSection: {symbol}\nLines: {chunk['start_line']}-{chunk['end_line']}"
            # {**chunk, ...} 类似 Java 复制 Map 后 put 新字段；后面的字段能覆盖同名键。
            output.append({**chunk, "symbol": symbol, "symbol_kind": chunk["chunk_type"], "language": language, "header": header})
        return output

    lines = normalized.splitlines()
    # Python 用 AST；其他语言用正则启发式识别；无法识别则按整个 module 兜底。
    regions = _python_regions(normalized, max_chars) if suffix == ".py" else _generic_regions(normalized, suffix)
    if not regions:
        regions = [(1, len(lines), "module", "module")]

    chunks: list[dict[str, Any]] = []
    symbol_occurrences: dict[str, int] = {}
    for start_line, end_line, symbol, symbol_kind in regions:
        # 重名函数可能出现多次（例如同文件重复定义），用 occurrence 区分父块 ID。
        symbol_occurrence = symbol_occurrences.get(symbol, 0)
        symbol_occurrences[symbol] = symbol_occurrence + 1
        parent_id = _stable_id("code-parent", path, symbol, symbol_occurrence)
        # 某个方法还是超过 max_chars，就由 _line_windows 分成 part 1、part 2...。
        for part_index, (content, part_start, part_end) in enumerate(
            _line_windows(lines, start_line, end_line, max_chars=max_chars)
        ):
            if not content.strip():
                continue
            chunk_id = _stable_id("code", parent_id, part_index, content)
            part_suffix = f" (part {part_index + 1})" if part_index else ""
            # 注意：part_index=0 的第一片不会加 '(part 1)'，第二片起才加提示。
            chunks.append(
                {
                    "content": content,
                    "start_line": part_start,
                    "end_line": part_end,
                    "symbol": symbol,
                    "symbol_kind": symbol_kind,
                    "language": language,
                    "chunk_type": "code_symbol" if symbol != "module" else "code_module",
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "child_index": part_index,
                    "symbol_occurrence": symbol_occurrence,
                    "header": (
                        # header 是给检索/展示用的摘要信息：路径、语言、符号、行号。
                        f"File: {path}\nLanguage: {language}\nSymbol: {symbol}{part_suffix}\n"
                        f"Lines: {part_start}-{part_end}"
                    ),
                }
            )
    return chunks


def chunk_pr_files(files: list[dict[str, Any]], max_chars: int = 1800) -> list[dict[str, Any]]:
    """专门处理 PR 文件列表：取每个文件 patch（diff）文本分块。"""
    # 输入示例：[{'filename':'src/App.java', 'patch':'@@ -1,2 +1,3 @@\n...'}]
    chunks: list[dict[str, Any]] = []
    for file in files:
        # patch 为空或 None 时当成空字符串；filename 缺失则兜底为 patch.diff。
        patch = file.get("patch") or ""
        filename = str(file.get("filename") or "patch.diff")
        # PR diff 不希望跨块重复，所以 overlap=0；只保留正文及文件名/序号/ID。
        for index, content in enumerate(chunk_text(patch, max_chars=max_chars, overlap=0)):
            chunks.append(
                {
                    "filename": filename,
                    "chunk_index": index,
                    "chunk_id": _stable_id("patch", filename, content),
                    "content": content,
                }
            )
    return chunks
