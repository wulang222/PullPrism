# ============================================================================
# DevFlow Chat / Engineering Workflow API —— 逐步详解注释版
# ============================================================================
# 阅读约定：
#   【作用】说明函数负责什么；【步骤】说明当前执行到哪儿；【例子】追踪真实变量的变化；
#   【注意】解释边界条件、安全问题或某行容易误解的地方。
#
# 建议先读文件最末尾 5 个 @router.post(...)：它们是 HTTP 请求入口；
# 再从 _conversation_response / _conversation_sse 逆向追踪 _build_conversation_tools；
# 最后看 _conversation_plan -> _conversation_plan_execute_sse 和工作流的 task_runners。
#
# 核心调用链 A（普通对话）：
#   POST / 或 /stream（假设此 router 在主应用中挂载到 /chat，则是 /chat 或 /chat/stream）
#   -> 校验仓库 / 获取会话 -> 历史上下文压缩 -> 存用户消息
#   -> 重建证据 / 组装上下文 -> 注册工具 -> ChatAgent.run(...) 或 run_stream(...)
#   -> 保存助手消息、AgentRun、会话摘要 -> commit -> JSON 或 SSE 返回。
#
# 核心调用链 B（确认式多 Agent 工作流）：
#   POST /plan -> PlannerAgent 生成 WorkflowSpec -> 返回计划（不执行）
#   -> 前端取得用户确认 -> POST /plan/execute/stream（路径相对 router 前缀）
#   -> WorkflowOrchestrator.run_spec -> Issue/PR/CI/RAG 等任务处理器
#   -> 保存 AgentWorkflowRun 和 AgentTaskRun -> 分段 SSE 输出。
#
# Python 与 Java 速记：
#   dict[str, Any] 约等于 Map<String, Object>；list[T] 约等于 List<T>；
#   None 相当于 null；`x or 默认值` 会在 x 为 None、空字符串、0、空容器等时用默认值；
#   `a if 条件 else b` 是三目表达式；`[f(x) for x in items if 条件]` 是列表推导式；
#   async def 是协程函数；await 等待可挂起任务；yield 会使函数产出多次结果；
#   `**dict_a, **dict_b` 是字典展开；相同键以后面的值为准（后者覆盖前者）。
#
# 【重要】以下只新增注释，不修改原代码的语句、调用和业务行为。
# 【注意】这是应用路由片段，直接运行通常还需要原项目的 app 包、数据库和配置。
# ============================================================================

# 【模块导航】这个文件是 FastAPI 的 HTTP 路由聚合层，负责把数据库、Agent、工具、记忆与 SSE 串联起来。
# 【提示】不要把文件中的所有 def 都理解成 HTTP 入口：只有带 @router.get/post 的函数才直接响应请求。
import uuid
# 【导入】uuid 为仓库/会话/任务生成或解析 UUID；datetime 处理时间窗；os/pathlib/re 处理本地路径和正则。
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
from typing import Any

# 【导入】FastAPI：APIRouter 注册接口；Depends 注入数据库 Session；Query 校验 URL 参数；HTTPException 返回 HTTP 错误。
from fastapi import APIRouter, Depends, HTTPException, Query
# 【导入】StreamingResponse 用于持续发送服务端事件（SSE）。
from fastapi.responses import StreamingResponse
# 【导入】SQLAlchemy 的 Session 负责数据库事务；selectinload 用于批量加载 ORM 关系对象。
from sqlalchemy.orm import Session, selectinload

# 【导入】项目自身配置 settings，下面的 checkout 路径会用它。
from app.core.config import settings
# 【导入】数据库 ORM 表映射：Repository、Issue、PR、WorkflowRun、ChatMessage 和 AgentRun 等。
from app.db.models import ActionDraft, AgentRun, AgentTaskRun, AgentWorkflowRun, ChatMessage, Conversation, Issue, PullRequest, Repository, WorkflowRun
# 【导入】get_db 是 FastAPI 依赖注入数据库会话的工厂。
from app.db.session import get_db
# 【导入】Pydantic 请求体和响应体；FastAPI 会据此解析、校验 JSON。
from app.schemas.chat import ChatHistoryMessage, ChatPlanExecuteRequest, ChatRequest, ChatResponse, PromptRecommendationRequest, PromptRecommendationResponse
# 【导入】下面连续导入各种职责明确的 Agent，包括 CI、聊天、Issue、规划、PR、报告、安全。
from app.services.agents.ci_debug_agent import CIDebugAgent
from app.services.agents.chat_agent import ChatAgent, encode_sse
from app.services.agents.issue_agent import IssueAgent
from app.services.agents.planner_agent import PlannerAgent
from app.services.agents.pr_review_agent import PRReviewAgent
from app.services.agents.recommendation_agent import RecommendationAgent
from app.services.agents.report_agent import ReportAgent
from app.services.agents.safety_agent import SafetyAgent
# 【导入】WorkflowOrchestrator 根据任务计划和执行器映射协调多个 Agent。
from app.services.agents.workflow_orchestrator import WorkflowOrchestrator
# 【导入】消息、证据、上下文组装、渐进压缩、会话封存等记忆组件。
from app.services.chat_memory import ContextAssembler, EvidenceStore, MessageStore, ProgressiveContextManager, SessionSealer
# 【导入】本地代码关键词检索，与下面向量 RAG 是不同的检索方式。
from app.services.code_search import query_terms, search_checkout_code
from app.services.conversations import ensure_conversation, touch_conversation
from app.services.memory_hub import record_recall_event
# 【导入】MCP stdio 客户端，允许 Agent 访问独立 MCP 进程里的记忆工具。
from app.services.mcp_client import StdioMcpClient
# 【导入】检索已索引的相似文档；add_document 把新周报纳入知识库。
from app.services.rag.retrieval import search_similar_documents
from app.services.rag.vector_store import add_document
# 【导入】WorkflowSpec/Task/Result：Planner 和执行器共用的结构化契约。
from app.schemas.workflow import WorkflowSpec, WorkflowTask, WorkflowTaskResult
from app.services.skills import SkillRegistry, get_skill_registry
# 【导入】创建草稿时的演示用户身份和审计日志能力。
from app.services.permissions import ensure_demo_user, write_audit_log

# 【路由表】创建路由对象；实际 URL 由 @router.get/post 路径和主应用 include_router(prefix=...) 合成。
router = APIRouter()

# 【路径常量】__file__ 是本 Python 文件路径，resolve 后 parents[4] / parents[3] 是向上数对应级别的目录。
# 【注意】这些层级依赖原项目目录结构；把注释版单独移出项目后，计算路径会不同。
PROJECT_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = Path(__file__).resolve().parents[3]
# 【安全常量】这些目录被代码浏览工具跳过，防止扫描 Git 元数据、虚拟环境和构建产物。
WORKSPACE_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".next",
    ".nuxt",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
# 【安全常量】直接拒绝读取的文件名，如 .env（可能包含密钥）。
WORKSPACE_SECRET_FILENAMES = {".env", ".env.local", ".env.production", ".npmrc", ".pypirc"}
# 【文件类型】允许特殊的无扩展名文本文件，如 Dockerfile 和 Makefile。
WORKSPACE_TEXT_FILENAMES = {"Dockerfile", "Makefile", "README", "LICENSE", "package.json", "requirements.txt", "pyproject.toml"}
# 【文件类型】允许扫描的源代码/配置/文档扩展名白名单。
WORKSPACE_TEXT_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".json",
    ".md",
    ".mdx",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".sql",
    ".sh",
    ".ps1",
    ".bat",
    ".cmd",
    ".html",
    ".css",
    ".scss",
}


# 【作用】GET /workflow-runs：查询最近工作流以及各子任务，用于前端工作流历史页面。
# 【例子】repo_id=None、limit=2：取最新 2 个工作流，再分别挂上对应的 tasks。
@router.get("/workflow-runs")
async def list_workflow_runs(
    repo_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    # 【步骤1】先创建 ORM 查询对象，此时只是在拼 SQL，还没有查数据库。
    query = db.query(AgentWorkflowRun)
    # 【步骤2】传了 repo_id 才限定仓库；没传则跨仓库查询。
    if repo_id is not None:
        query = query.filter(AgentWorkflowRun.repo_id == repo_id)
    # 【步骤3】按创建时间倒序并 limit，调用 .all() 才真正执行查询。
    # 【例子】5 条记录、limit=2 -> rows 只有最新 2 条。
    rows = query.order_by(AgentWorkflowRun.created_at.desc()).limit(limit).all()
    # 【步骤4】提取主表 ID，为一次性批量查子任务做准备。
    # 【例子】rows=[wfA,wfB] -> workflow_ids=[wfA.id,wfB.id]。
    workflow_ids = [row.id for row in rows]
    # 【步骤5】如果列表非空，用 IN 查询所有对应任务，并按开始时间/任务 ID 排序。
    # 【注意】workflow_ids=[] 时直接 tasks=[]，避免无意义数据库查询。
    tasks = (
        db.query(AgentTaskRun)
        .filter(AgentTaskRun.workflow_id.in_(workflow_ids))
        .order_by(AgentTaskRun.started_at.asc(), AgentTaskRun.task_id.asc())
        .all()
        if workflow_ids
        else []
    )
    # 【步骤6】建立分组索引，类型类似 Java Map<UUID,List<AgentTaskRun>>。
    tasks_by_workflow: dict[uuid.UUID, list[AgentTaskRun]] = {}
    for task in tasks:
        # 【步骤7】setdefault(k,[])：若 k 尚无值，先放空列表；再 append 任务。
        # 【例子】两条 task.workflow_id='A' -> {'A':[task1,task2]}。
        tasks_by_workflow.setdefault(task.workflow_id, []).append(task)
    # 【步骤8】使用列表推导式序列化每个工作流，内层列表推导式组装它的 tasks。
    # 【例子】返回 [{'id':'A','tasks':[{'task_id':'t1',...},...]},...]。
    return [
        {
            "id": str(row.id),
            "repo_id": str(row.repo_id) if row.repo_id else None,
            "conversation_id": str(row.conversation_id) if row.conversation_id else None,
            "goal": row.goal,
            "status": row.status,
            "spec": row.spec_json or {},
            "observation": row.observation_json or {},
            "final_answer": row.final_answer,
            "metrics": row.metrics or {},
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "tasks": [
                {
                    "id": str(task.id),
                    "task_id": task.task_id,
                    "agent_name": task.agent_name,
                    "task_type": task.task_type,
                    "status": task.status,
                    "summary": str((task.result_json or {}).get("summary") or ""),
                    "error": task.error,
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                }
                for task in tasks_by_workflow.get(row.id, [])
            ],
        }
        for row in rows
    ]


# 【作用】GET /history：获取一个仓库中指定会话的用户/助手消息。
# 【注意】只保留 role=user/assistant；系统消息和内部工具消息不作为聊天记录返回。
@router.get("/history", response_model=list[ChatHistoryMessage])
async def chat_history(
    repo_id: uuid.UUID = Query(...),
    conversation_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    before_message_id: uuid.UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    # 【步骤1】校验仓库；查询不到就立即以 404 结束请求。
    _repo_or_404(db, repo_id)
    # 【步骤2】取出会话；若允许/需要，也可能创建一个新会话。
    conversation = ensure_conversation(db, repo_id, conversation_id)
    # 【步骤3】将可能为空的传入 ID 替换为实际会话 ID。
    conversation_id = conversation.id
    # 【步骤4】提交 ensure_conversation 可能产生的会话变更。
    db.commit()
    rows = [
        item
        # 【步骤5】按 repo_id、conversation_id 查询时间线；limit 和 before_message_id 支持分页。
        for item in MessageStore(db).timeline(
            repo_id,
            conversation_id,
            limit=limit,
            before_message_id=before_message_id,
        )
        # 【步骤6】过滤消息角色，只向历史页面输出 user/assistant。
        if item.role in {"user", "assistant"}
    ]
    # 【步骤7】ORM -> JSON 可序列化字典；datetime 变 isoformat 字符串、UUID 变 str。
    # 【例子】created_at=datetime(...) -> '2026-10-30T09:00:00'。
    return [
        {
            "id": str(item.id),
            "conversation_id": str(item.conversation_id) if item.conversation_id else None,
            "role": item.role,
            "content": item.content,
            "route": item.route,
            "tool_calls": item.tool_calls or [],
            "agent_events": (item.meta or {}).get("agent_events", []),
            "compression_stats": (item.meta or {}).get("compression_stats", {}),
            "progressive_compaction": (item.meta or {}).get("progressive_compaction", {}),
            "model_usage": (item.meta or {}).get("model_usage", []),
            "context_diagnostics": (item.meta or {}).get("context_diagnostics", {}),
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }
        for item in rows
    ]


# 【作用】统一检查仓库是否存在。存在返回 ORM Repository，不存在直接抛 HTTP 404。
def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    # db.get(Model, primary_key) 通过主键获取一条 ORM 记录，类似 JPA findById。
    repo = db.get(Repository, repo_id)
    if repo is None:
        # HTTPException 会交给 FastAPI 返回 404 JSON 错误，下面的 return 不再执行。
        raise HTTPException(status_code=404, detail="未找到指定仓库。")
    return repo


# 【作用】过滤客户端提供的 context，不允许客户端伪造由服务端管理的字段。
# 【例子】输入 {'foo':1,'repo_id':'fake'} -> 输出 {'foo':1}。
def _client_context(value: dict[str, Any] | None) -> dict[str, Any]:
    # 服务端保留字段白名单的反面：这些 key 由后端自己写，不接受前端传入的同名值。
    server_owned = {
        "repo_id",
        "conversation_id",
        "tools",
        "server_context",
        "server_messages",
        "evidence",
        "compression_stats",
        "progressive_compaction",
        "registered_skills",
        "skill_description_by_tool",
    }
    # 字典推导式只保留不在 server_owned 的键。
    # 【例子】{'limit':5,'tools':'hijack'} -> {'limit':5}。
    return {key: item for key, item in (value or {}).items() if key not in server_owned}


# 【作用】取排序依据：优先 updated_at，其次 created_at，都无时为 datetime.min。
def _item_time(item: Any) -> datetime:
    # 利用 or 短路：updated_at 有值就不再看 created_at。
    return item.updated_at or item.created_at or datetime.min


# 【作用】将字符串列表用分号拼接。例：['A','B'] -> 'A; B'；[] -> empty_text。
def _join_items(items: list[str], empty_text: str) -> str:
    # 字符串 join 是分隔符调用方法；不是 Java 的 List.join。
    return "; ".join(items) if items else empty_text


# 【作用】根据用户消息中的关键词推测检索类型，供 RAG 过滤使用。
# 【注意】是简单关键词匹配，并不调用模型；匹配顺序是 Issue -> PR -> CI -> 知识文件。
def _source_type_from_message(message: str) -> str | None:
    # 统一转小写，让 'ISSUE' 和 'issue' 匹配相同。
    lowered = message.lower()
    # 按分支从上到下匹配；第一个 return 后函数结束。
    # 【注意】包含关系可能误匹配某些词，这是关键词分类的局限。
    if "issue" in lowered:
        return "issue"
    if "pr" in lowered or "pull request" in lowered or "diff" in lowered:
        return "pull_request"
    if "ci" in lowered or "workflow" in lowered:
        return "workflow_run"
    if any(token in lowered for token in ["knowledge", "知识库", "上传资料", "上传文档"]):
        return "knowledge_file"
    return None


# 【作用】粗略识别可能产生外部写操作的意图，触发 SafetyAgent/草稿流程。
# 【例子】'please create issue' -> 'create_issue'；'show status' -> None。
def _risky_action(message: str) -> str | None:
    lowered = message.lower()
    # any(...) 表示只要有一个 token 被包含就为 True。
    # 【例子】['comment','reply','post'] 中任意命中就判断为 comment。
    if any(token in lowered for token in ["comment", "reply", "post"]):
        return "comment"
    if any(token in lowered for token in ["create issue", "new issue"]):
        return "create_issue"
    if any(token in lowered for token in ["send", "send_report"]):
        return "send_report"
    if any(token in lowered for token in ["close issue", "label", "write"]):
        return "write_back"
    # 没有命中表示无需触发此函数所识别的写操作；None 本身不等于整体安全审查。
    return None


# 【作用】把 context 的时间条件转换为起止 datetime（UTC）。
# 【例子】time_range='3d'：start=今天零点未强制设置，实际为当前时刻减 2 天；end=现在。
def _date_window_from_context(context: dict[str, Any]) -> tuple[datetime, datetime]:
    # 【步骤1】默认截止时间为当前 UTC 时刻。
    end_at = datetime.now(timezone.utc)
    # 【步骤2】读取 '7d' 等短格式，未提供时默认为 7 天。
    time_range = str(context.get("time_range") or "7d").lower()
    # 【步骤3】只有以 d 结尾且前面全为数字才解析；'14d' -> 14，'abc' -> 7。
    days = int(time_range[:-1]) if time_range.endswith("d") and time_range[:-1].isdigit() else 7
    # 【步骤4】max(1,days) 保证至少一天，-1 是当前实现的时间窗计算方式。
    # 【例子】'7d' -> 起点=当前时刻-6天（不是整整 7x24 小时）。
    start_at = end_at - timedelta(days=max(1, days) - 1)
    # 【步骤5】明确传了 start_date 时覆盖默认起点；fromisoformat 将 ISO 字符串转 datetime。
    if context.get("start_date"):
        start_at = datetime.fromisoformat(str(context["start_date"])).replace(tzinfo=timezone.utc)
    # 【步骤6】明确传了 end_date 时把截止设为当天 23:59:59（UTC）。
    if context.get("end_date"):
        end_at = datetime.fromisoformat(str(context["end_date"])).replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    # 【步骤7】返回 Python tuple；调用处可写 start_at, end_at = ... 解构赋值。
    return start_at, end_at


# 【作用】判断某时间戳是否处于 [start_at, end_at] 闭区间。
# 【注意】未带时区的时间直接视为 UTC，并非转换到 UTC。
def _within(value: datetime | None, start_at: datetime, end_at: datetime) -> bool:
    # 空时间戳不算命中时间窗口。
    if not value:
        return False
    # 三目表达式：已有 tzinfo 就保留原值，否则把它标记为 UTC。
    current = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    # 链式比较是两个条件同时成立：start_at <= current 且 current <= end_at。
    return start_at <= current <= end_at


# 【作用】从数据库一次获取该仓库已同步的 Issue/PR/CI 运行记录。
# 【注意】selectinload 会批量预加载 PR 的 files 与 review_comments，避免循环内 N+1 查询。
def _repo_payload(repo_id: uuid.UUID, db: Session) -> tuple[list[Issue], list[PullRequest], list[WorkflowRun]]:
    # 【步骤1】按 repo_id 读取同步过的所有 Issue。
    issues = db.query(Issue).filter(Issue.repo_id == repo_id).all()
    # 【步骤2】PR 查询含 selectinload：额外批量取出文件列表和 review 评论。
    prs = (
        db.query(PullRequest)
        .options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments))
        .filter(PullRequest.repo_id == repo_id)
        .all()
    )
    # 【步骤3】查询 CI WorkflowRun；返回三个列表供其它工具复用。
    runs = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).all()
    return issues, prs, runs


# 【作用】将 ORM 对象转成 RecommendationAgent 能接受的普通字典。
# 【例子】Issue(number=42,title='Bug') -> {'number':42,'title':'Bug',...}。
def _recommendation_payload(
    repo: Repository | None,
    issues: list[Issue],
    prs: list[PullRequest],
    runs: list[WorkflowRun],
    exclude: list[str],
    limit: int,
) -> dict[str, Any]:
    # 【步骤】把 repo / issue / PR / CI 各字段转换为 Agent 可读数据；不是往数据库写入。
    # 【注意】列表推导式里的 item 是单条 ORM 对象，最终产物是 list[dict]。
    return {
        "repo": {"id": str(repo.id), "full_name": repo.full_name} if repo else None,
        "issues": [
            {
                "id": str(item.id),
                "number": item.number,
                "title": item.title,
                "state": item.state,
                "labels": item.labels or [],
                "assignees": item.assignees or [],
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "closed_at": item.closed_at,
            }
            for item in issues
        ],
        "pull_requests": [
            {
                "id": str(item.id),
                "number": item.number,
                "title": item.title,
                "state": item.state,
                "author": item.author,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "merged_at": item.merged_at,
            }
            for item in prs
        ],
        "workflow_runs": [
            {
                "id": str(item.id),
                "name": item.name,
                "status": item.status,
                "conclusion": item.conclusion,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in runs
        ],
        "exclude": exclude,
        "limit": limit,
    }


# 【作用】依据已同步数据（不是实时 GitHub API）生成仓库健康概览。
def _repo_health_answer(repo: Repository, issues: list[Issue], prs: list[PullRequest], runs: list[WorkflowRun]) -> str:
    # 【步骤1】过滤当前 state 为 open 的 issue，类似 Java Stream.filter。
    open_issues = [item for item in issues if item.state == "open"]
    open_prs = [item for item in prs if item.state == "open"]
    # 【步骤2】merged_at 有值表示已合并，计入 merged_prs。
    merged_prs = [item for item in prs if item.merged_at]
    # 【步骤3】只统计 conclusion == failure 的 CI 运行。
    failed_runs = [item for item in runs if item.conclusion == "failure"]
    # 【步骤4】sorted(..., reverse=True) 时间倒序，[:3] 截取前三条。
    # 【例子】5 条 Issue 更新日期排序后 -> 只显示最新 3 条。
    latest_issues = sorted(open_issues, key=_item_time, reverse=True)[:3]
    latest_prs = sorted(prs, key=_item_time, reverse=True)[:3]
    # 【步骤5】f-string 生成 '#编号 标题'，随后 join；没有则显示兜底文案。
    issue_titles = _join_items([f"#{item.number} {item.title}" for item in latest_issues], "没有打开的 Issue")
    pr_titles = _join_items([f"#{item.number} {item.title}" for item in latest_prs], "没有 PR 记录")
    ci_text = f"已同步 {len(failed_runs)} 次失败 CI 运行" if failed_runs else "还没有同步到失败 CI 运行"
    # 【步骤6】相邻字符串/f-string 在圆括号内自动拼接，返回一个完整 Markdown 字符串。
    return (
        f"当前仓库：{repo.full_name}\n\n"
        f"- Issues：{len(open_issues)} 个打开。近期关注：{issue_titles}。\n"
        f"- PRs：{len(open_prs)} 个打开，{len(merged_prs)} 个已合并。近期 PR：{pr_titles}。\n"
        f"- CI：{ci_text}。\n\n"
        "建议下一步：先看最新打开的 Issue，或检查最新 PR 风险；如果同步后出现失败 CI，可以让我继续排障。"
    )


# 【作用】统一封装 RAG 检索：元数据过滤 -> 查相似文档 -> 拼文本、引用和原始结果。
# 【返回】三元组 (answer, citations, results)，可用 `answer, citations, results = ...` 解包。
def _semantic_search(repo: Repository, message: str, context: dict[str, Any], db: Session) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
    # 【步骤1】从 context 提取元数据过滤条件，例如 label='bug'、path='src/api'。
    filters = {
        "label": context.get("label"),
        "path": context.get("path"),
        "module": context.get("module"),
        "state": context.get("state"),
        "start_date": context.get("start_date"),
        "end_date": context.get("end_date"),
    }
    # 【步骤2】检索层执行相似性检索：限定 repo.id、query、source_type、limit、metadata_filters。
    results = search_similar_documents(
        db,
        repo.id,
        message,
        source_type=context.get("source_type") or _source_type_from_message(message),
        limit=int(context.get("limit", 5)),
        # 【例子】filters={'label':'bug','path':None} -> metadata_filters={'label':'bug'}。
        metadata_filters={key: value for key, value in filters.items() if value},
    )
    # 【步骤3】空结果直接返回 ('没找到...', [], [])，避免后续 for 循环。
    if not results:
        return "没有找到足够相似的历史记录。可以尝试放宽标签、路径或时间过滤条件。", [], []
    lines = ["找到以下相关历史记录："]
    citations: list[dict[str, str]] = []
    # 【步骤4】enumerate(results, start=1)：同时取得序号和命中项，序号从 1 开始。
    for index, item in enumerate(results, start=1):
        # 【步骤5】从每条记录读取 metadata，缺失时回退空 dict，避免访问 .get 报错。
        meta = item.get("metadata") or {}
        number = meta.get("number")
        prefix = f"#{number} " if number else ""
        # 【步骤6】向文本列表追加命中来源、编号、标题、相似度。
        lines.append(f"- [{index}] {item['source_type']} {prefix}{item['title']}：相似度 {item['score']}")
        snippet = str(item.get("snippet") or "").strip()
        if snippet:
            lines.append(f"  证据：{snippet}")
        # 【步骤7】引用记录独立保存 source_type/source_id/title，用于答案来源展示。
        citations.append({"type": item["source_type"], "id": item["source_id"], "title": item["title"]})
    # 【步骤8】把多行合并成 answer，同时原样返回 citations 和 results（供评估/记忆记录）。
    return "\n".join(lines), citations, results


# 【作用】计算本地仓库克隆总目录，并规范为绝对路径。
def _checkout_root() -> Path:
    # Path 封装文件系统路径，settings.repo_checkout_dir 可以是绝对或相对路径。
    root = Path(settings.repo_checkout_dir)
    # 相对路径以 BACKEND_ROOT 为基准补全，最后 resolve 规范路径。
    if not root.is_absolute():
        root = BACKEND_ROOT / root
    return root.resolve()


# 【作用】优先选 repo.local_path；否则根据仓库 ID + 仓库名定位本地克隆目录。
# 【注意】这里包含兼容旧 UUID 目录的回退逻辑，不直接重新 git clone。
def _repo_checkout_path(repo: Repository) -> Path:
    # getattr(repo,'local_path',None)：类似 Java 反射读属性，属性缺失时返回 None。
    local_path = getattr(repo, "local_path", None)
    # 【优先级1】仓库记录显式配置了本地路径时，展开 ~ 和环境变量再检测是否存在。
    if local_path:
        path = Path(os.path.expandvars(os.path.expanduser(str(local_path))))
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        resolved = path.resolve()
        if resolved.exists():
            return resolved

    # 【优先级2】将 owner/repo 转成 owner__repo，组合 UUID 做克隆目录名。
    safe_name = repo.full_name.replace("/", "__").replace("\\", "__")
    checkout_root = _checkout_root()
    expected = (checkout_root / f"{repo.id}-{safe_name}").resolve()
    if expected.exists():
        return expected

    # 仓库记录可能被删除后重建，而本地 clone 仍保留旧 UUID。
    # 复用同名 checkout，避免工作区工具静默失去源码访问能力。
    # 【优先级3】如果记录 UUID 更新但旧 clone 仍在，按仓库名搜索其它旧目录。
    candidates = [
        path
        for path in checkout_root.glob(f"*-{safe_name}")
        if path.is_dir() and path.resolve() != expected
    ]
    if candidates:
        # 按照最近修改时间 mtime 倒序，选最新候选。
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0].resolve()
    return expected


# 【作用】确定代码浏览工具的实际工作区；没有仓库则退回项目根目录。
def _workspace_root(repo: Repository | None) -> Path:
    # 没有选中仓库时返回当前项目根目录；不意味着能读任意磁盘路径。
    if repo is None:
        return PROJECT_ROOT
    checkout_path = _repo_checkout_path(repo)
    # 仓库尚未同步/本地目录不存在，主动抛错误提示前端先同步。
    if not checkout_path.exists():
        raise ValueError(
            f"{repo.full_name} 的仓库代码还没有克隆到本地。"
            "读取源码前，请先同步仓库或重新连接仓库。"
        )
    return checkout_path


# 【作用】把绝对路径转为相对工作区的 POSIX 路径，统一用 / 分隔。
def _workspace_relative(path: Path, workspace_root: Path) -> str:
    # Path.relative_to 去掉共同前缀；as_posix 统一斜杠。例 /repo/a/b.py -> a/b.py。
    return path.relative_to(workspace_root).as_posix()


# 【作用】把模型传入的路径解析成工作区内真实路径，并进行越界、敏感目录/文件校验。
# 【例子】root=/repo, raw='src/a.py' -> /repo/src/a.py；raw='../etc/passwd' -> 拒绝。
def _workspace_path(workspace_root: Path, raw_path: str | None = None) -> Path:
    # 【步骤1】把输入转字符串并 strip 空白，空路径默认 '.'（根目录）。
    normalized_path = str(raw_path or ".").strip()
    if normalized_path in {"", ".", "./", "/", "\\"}:
        normalized_path = "."
    # 【步骤2】非 Windows 盘符路径时去掉前导 / 或 \，避免被解析成系统绝对路径。
    elif not re.match(r"^[a-zA-Z]:", normalized_path):
        normalized_path = normalized_path.lstrip("/\\") or "."
    # 【步骤3】拼到工作区并 resolve：规范 `..` 和符号链接指向的真实路径。
    candidate = (workspace_root / normalized_path).resolve()
    try:
        # 【步骤4】安全边界检查：只有 candidate 在 workspace_root 内才能求相对路径。
        # 【例子】root=/repo, raw='../../secret.txt' -> 越界抛 ValueError。
        candidate.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError("路径位于项目工作区之外。") from exc
    # 【步骤5】阻止访问 .git、node_modules、venv 等跳过目录。
    if any(part in WORKSPACE_SKIP_DIRS for part in candidate.relative_to(workspace_root).parts):
        raise ValueError("路径位于被跳过的工作区目录中。")
    # 【步骤6】阻止访问 .env、.npmrc 等敏感文件；返回经过校验的 Path。
    if candidate.name in WORKSPACE_SECRET_FILENAMES:
        raise ValueError("不会向对话 Agent 暴露敏感配置文件。")
    return candidate


# 【作用】根据名称/扩展名粗筛可读文本文件；明确排除敏感和数据库文件。
def _is_workspace_text(path: Path) -> bool:
    # 【步骤1】敏感配置和 .log/.db/.sqlite 明确拒绝，不受后面白名单影响。
    if path.name in WORKSPACE_SECRET_FILENAMES or path.suffix.lower() in {".log", ".db", ".sqlite", ".sqlite3"}:
        return False
    # 【步骤2】允许固定名称或常见代码/文本扩展名。扩展名是粗筛，不代表内容必然安全。
    return path.name in WORKSPACE_TEXT_FILENAMES or path.suffix.lower() in WORKSPACE_TEXT_EXTENSIONS or path.name.lower().endswith("dockerfile")


# 【作用】生成器：递归扫描工作区，逐个 yield 可读文件，最多 max_files 个。
# 【类比】Java Iterator，调用方遍历时才按需取下一个文件。
def _iter_workspace_files(root: Path, workspace_root: Path, max_files: int = 1000):
    # 【步骤1】如果指定的 root 本身就是文件，则只 yield 它并 return。
    if root.is_file():
        yield root
        return
    count = 0
    # 【步骤2】rglob('*') 递归遍历文件/目录（不是一次全部读入内存）。
    for path in root.rglob("*"):
        # 【步骤3】达到允许的文件数量上限后终止扫描。
        if count >= max_files:
            break
        # 【步骤4】跳过目录、被排除路径和不可读类型。
        if not path.is_file():
            continue
        relative_parts = path.relative_to(workspace_root).parts
        if any(part in WORKSPACE_SKIP_DIRS for part in relative_parts):
            continue
        if not _is_workspace_text(path):
            continue
        # 【步骤5】只有实际 yield 的合格文件才计数。
        count += 1
        yield path


# 【作用】校验大小/格式后，按多种编码依次尝试把文件字节解码成字符串。
def _read_workspace_text(path: Path, max_bytes: int = 250_000) -> str:
    # 【步骤1】只允许读取现存普通文件。
    if not path.exists() or not path.is_file():
        raise ValueError("未找到工作区文件。")
    if not _is_workspace_text(path):
        raise ValueError("只能读取非敏感的文本工作区文件。")
    # 【步骤2】按字节读取；此时不是字符串，类型为 bytes。
    data = path.read_bytes()
    # 【步骤3】超过 250000 bytes 或开头含 NUL 字节，认为过大/可能为二进制，拒绝。
    if len(data) > max_bytes or b"\x00" in data[:4096]:
        raise ValueError("文件过大，或看起来是二进制文件。")
    # 【步骤4】依次尝试 UTF-8 BOM、UTF-8、GB18030、Latin-1；成功即返回 str。
    # 【注意】latin-1 可解码任意字节，最后的 ValueError 分支通常很难触发。
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "latin-1"]:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法将工作区文件解码为文本。")


# 【作用】对潜在写操作创建可审阅草稿；此处只生成文本，不提交 GitHub。
def _draft_for_action(action: str, message: str, repo: Repository, issues: list[Issue], prs: list[PullRequest], runs: list[WorkflowRun]) -> tuple[str, list[dict[str, str]]]:
    # 【步骤1】先添加仓库来源，草稿引用可附加 Issue/PR 来源。
    citations: list[dict[str, str]] = [{"type": "repository", "id": str(repo.id), "title": repo.full_name}]
    # 【步骤2】选最近变更的 Issue 和 PR，后续作为草稿可能关联的对象。
    latest_issue = sorted(issues, key=_item_time, reverse=True)[0] if issues else None
    latest_pr = sorted(prs, key=_item_time, reverse=True)[0] if prs else None
    failed_runs = [run for run in runs if run.conclusion == "failure"]
    # 【步骤3】各分支直接返回 (草稿 Markdown, citations)，此函数没有写外部数据。
    if action == "create_issue":
        return f"## Issue 草稿\n标题：{message.strip()[:120]}\n\n### 背景\n请补充影响范围、复现步骤和预期行为。\n\n### 验收标准\n- 预期行为清晰\n- 包含验证步骤\n", citations
    if action == "comment" and latest_pr:
        citations.append({"type": "pull_request", "id": str(latest_pr.id), "title": latest_pr.title})
        return f"## PR 评论草稿\n针对 PR #{latest_pr.number}：{latest_pr.title}\n\n合并前请确认测试覆盖、未解决的 Review 评论，以及是否存在高风险文件。\n", citations
    if action == "comment" and latest_issue:
        citations.append({"type": "issue", "id": str(latest_issue.id), "title": latest_issue.title})
        return f"## Issue 评论草稿\n针对 Issue #{latest_issue.number}：{latest_issue.title}\n\n请补充复现步骤、影响范围、预期行为，以及当前是否有临时绕过方案。\n", citations
    if action == "send_report":
        draft = (
            "## 报告发送草稿\n"
            f"仓库：{repo.full_name}\n\n"
            f"- 打开的 Issue：{len([issue for issue in issues if issue.state == 'open'])}\n"
            f"- 打开的 PR：{len([pr for pr in prs if pr.state == 'open'])}\n"
            f"- 失败 CI 运行：{len(failed_runs)}\n\n"
            "发送前请确认收件人、日期范围和敏感日志内容。\n"
        )
        return draft, citations
    return f"## 写操作草稿\n这可能会写入 GitHub 或外部渠道。MVP 模式只生成动作草稿。\n\n用户请求：{message.strip()}\n", citations


# 【作用】把 Agent 置信度规范为 0~1；非法值回退 default。
# 【例子】1.8 -> 1.0；-0.2 -> 0.0；'abc' -> default。
def _safe_confidence(value: Any, default: float) -> float:
    try:
        # min(1.0,float(value)) 先裁剪上界；max(0.0,...) 再裁剪下界。
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


# 【作用】构造状态为 skipped 的标准任务结果，用于缺少目标对象的情况。
def _skipped_workflow_result(task: WorkflowTask, summary: str) -> WorkflowTaskResult:
    # WorkflowTaskResult 是统一 schema，status='skipped' 且 confidence=0.0。
    return WorkflowTaskResult(
        task_id=task.id,
        agent_name=task.agent_name,
        task_type=task.task_type,
        status="skipped",
        summary=summary,
        confidence=0.0,
    )


# 【作用】从传给下游 Agent 的 context 中移除不可序列化/内部使用的字段。
def _agent_context(context: dict[str, Any]) -> dict[str, Any]:
    # 字典推导式过滤 tools 与 _skill_registry，其他字段原样保留。
    return {key: value for key, value in context.items() if key not in {"tools", "_skill_registry"}}


# 【作用】专门表示工作流目标无法安全解析，例如明确要求 Issue #42 但数据库里没有。
class WorkflowTargetResolutionError(ValueError):
    """Raised when an explicitly requested workflow target cannot be resolved safely."""


_WORKFLOW_TARGET_PATTERNS: dict[str, tuple[str, ...]] = {
    "issue": (
        r"(?<![A-Za-z0-9_])issue(?![A-Za-z0-9_])\s*(?:#|＃|no\.?|number|id)?\s*(\d+)",
        r"(?:工单|议题)\s*(?:#|＃|编号)?\s*(\d+)",
    ),
    "pull_request": (
        r"(?<![A-Za-z0-9_])(?:pr|pull[\s_-]*request)(?![A-Za-z0-9_])\s*(?:#|＃|no\.?|number|id)?\s*(\d+)",
        r"合并请求\s*(?:#|＃|编号)?\s*(\d+)",
    ),
    "workflow_run": (
        r"(?<![A-Za-z0-9_])(?:ci|workflow[\s_-]*run)(?![A-Za-z0-9_])\s*(?:#|＃|no\.?|number|id)?\s*(\d+)",
        r"(?<![A-Za-z0-9_])run(?![A-Za-z0-9_])\s*(?:#|＃|id)\s*(\d+)",
        r"(?:CI\s*运行|工作流运行|流水线)\s*(?:#|＃|编号|ID)?\s*(\d+)",
    ),
}

_WORKFLOW_TARGET_LABELS = {
    "issue": "Issue",
    "pull_request": "PR",
    "workflow_run": "CI/Workflow Run",
}


# 【作用】从自然语言提取明确指定的单个 Issue/PR/CI 编号。
# 【例子】'分析 PR #28' -> 28；'分析 PR #28 和 PR #29' -> 抛错，不随意选一个。
def _requested_workflow_target_number(message: str, target_type: str) -> int | None:
    # 【步骤1】集合推导式 + 正则表达式：遍历该类型所有正则，找出捕获组中的数字并转 int。
    # 【例子】'检查 PR #12' -> numbers={12}；重复写两次 #12 仍只有一个元素。
    numbers = {
        int(match.group(1))
        for pattern in _WORKFLOW_TARGET_PATTERNS[target_type]
        for match in re.finditer(pattern, message, flags=re.IGNORECASE)
    }
    # 【步骤2】同类型检测到 2 个以上不同编号就拒绝：当前工作流只接受唯一的明确 claim。
    if len(numbers) > 1:
        references = "、".join(f"#{number}" for number in sorted(numbers))
        raise WorkflowTargetResolutionError(
            f"一次工作流只能为一个 {_WORKFLOW_TARGET_LABELS[target_type]} 建立明确 claim；检测到多个目标：{references}。"
        )
    # 【步骤3】先 iter(set) 得到迭代器，next(iterator,None) 取第一个；空集合返回 None。
    return next(iter(numbers), None)


# 【作用】根据数字编号在已同步对象里查找；找不到就报错，防止分析错对象。
def _target_by_number(
    items: list[Any],
    number: int | None,
    *,
    target_type: str,
    number_attribute: str,
) -> Any | None:
    # 【步骤1】用户没有指名编号 -> 返回 None，让上层决定默认目标。
    if number is None:
        return None
    # 【步骤2】生成器表达式惰性找第一条 number_attribute 对应编号的记录。
    # 【例子】items=[Issue(7),Issue(42)]、number=42 -> 第二个对象。
    target = next((item for item in items if getattr(item, number_attribute, None) == number), None)
    # 【步骤3】明确目标不存在就抛业务错误，不偷偷把 #42 换成最近 Issue。
    if target is None:
        raise WorkflowTargetResolutionError(
            f"当前仓库没有同步 {_WORKFLOW_TARGET_LABELS[target_type]} #{number}，已停止生成计划；不会回退到最新对象。"
        )
    return target


# 【作用】构造多 Agent 工程工作流的运行环境，返回 (workflow_context, task_runners)。
# 【重点】下面嵌套的 async 函数是闭包，能够直接使用外层 repo、db、selected_issue 等变量。
def _build_engineering_workflow_runtime(
    repo: Repository,
    db: Session,
    message: str,
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # 【步骤1】从数据库批量拉取当前仓库已同步的数据，之后多个子任务复用这些列表。
    issues, prs, runs = _repo_payload(repo.id, db)
    failed_runs = [run for run in runs if run.conclusion == "failure"]
    # 【步骤2】解析用户消息里明确点名的 Issue/PR/CI 数字编号。
    # 【例子】message='审查 PR #18' -> requested_pr_number=18，其余通常为 None。
    requested_issue_number = _requested_workflow_target_number(message, "issue")
    requested_pr_number = _requested_workflow_target_number(message, "pull_request")
    requested_run_number = _requested_workflow_target_number(message, "workflow_run")
    # 【步骤3】按显式编号匹配内部 ORM 对象；指定不存在编号会立刻抛错。
    selected_issue = _target_by_number(
        issues,
        requested_issue_number,
        target_type="issue",
        number_attribute="number",
    )
    selected_pr = _target_by_number(
        prs,
        requested_pr_number,
        target_type="pull_request",
        number_attribute="number",
    )
    selected_run = _target_by_number(
        runs,
        requested_run_number,
        target_type="workflow_run",
        number_attribute="github_run_id",
    )
    # 【步骤4】只有用户未指定有效对象时，才允许选择最新的 Issue/PR 或最新失败 CI。
    if selected_issue is None:
        selected_issue = sorted(issues, key=_item_time, reverse=True)[0] if issues else None
    if selected_pr is None:
        selected_pr = sorted(prs, key=_item_time, reverse=True)[0] if prs else None
    if selected_run is None:
        selected_run = sorted(failed_runs, key=_item_time, reverse=True)[0] if failed_runs else None

    # 【作用】解析某个任务 claim 声明的精确 UUID 目标，不允许错误类型/跨仓库目标。
    # 【例子】claim.target_id 为空 -> 使用上面选定的 fallback；写了不存在 UUID -> 直接报错。
    def claimed_target(task: WorkflowTask, target_type: str, items: list[Any], fallback: Any | None) -> Any | None:
        # 【步骤1】claim 声称的类型必须匹配当前执行器，避免将 PR 的 UUID 交给 Issue Agent。
        if task.claim.target_type != target_type:
            raise WorkflowTargetResolutionError(
                f"任务 {task.id} 的 claim 类型为 {task.claim.target_type!r}，预期为 {target_type!r}。"
            )
        # 【步骤2】没有指定内部 UUID 时使用安全选定的 fallback。
        if not task.claim.target_id:
            return fallback
        try:
            # 【步骤3】字符串转 UUID 类型；非法格式捕获错误并转为业务异常。
            target_id = uuid.UUID(str(task.claim.target_id))
        except (TypeError, ValueError) as exc:
            raise WorkflowTargetResolutionError(f"任务 {task.id} 的 claim.target_id 不是有效 UUID。") from exc
        # 【步骤4】只在当前仓库已读取的 items 内寻找 id 一致的对象，跨仓库的 UUID 也无法命中。
        target = next((item for item in items if item.id == target_id), None)
        if target is None:
            raise WorkflowTargetResolutionError(
                f"任务 {task.id} 的 claim 目标不存在或不属于仓库 {repo.full_name}；不会回退到最新对象。"
            )
        return target

    # 【步骤5】清理对模型/子 Agent 不该暴露的工具对象和技能注册表。
    safe_context = _agent_context(context)
    # 【步骤6】从原 context 取内部技能注册表；激活行为由下方闭包按需执行。
    skill_registry = context.get("_skill_registry")
    agent_skill_cache: dict[str, dict[str, Any]] = {}

    # 【作用】按 entrypoint 激活技能并缓存，避免同一技能重复装载。
    # 【例子】第一次查询 'issue_agent.analyze_issue' 可能 activate；第二次直接读缓存。
    def agent_skill_context(entrypoint: str) -> dict[str, Any]:
        # 【步骤1】命中缓存直接返回，省去重复激活/构造 prompt。
        if entrypoint in agent_skill_cache:
            return agent_skill_cache[entrypoint]
        # 【步骤2】无有效 SkillRegistry 时返回空字典，业务 Agent 仍可运行。
        if not isinstance(skill_registry, SkillRegistry):
            agent_skill_cache[entrypoint] = {}
            return {}
        # 【步骤3】从注册表激活 entrypoint；没有对应技能则返回 None。
        activation = skill_registry.activate_entrypoint(entrypoint)
        if activation is None:
            agent_skill_cache[entrypoint] = {}
            return {}
        # 【步骤4】Pydantic model_dump(mode='json') 把配置模型转成可 JSON 化的字典。
        payload = activation.model_dump(mode="json")
        # 【步骤5】组合 Agent 所需 prompt_block、激活技能和运行追踪元数据。
        runtime = {
            "active_skills": [payload],
            "skill_instructions": activation.prompt_block(),
            "skill_runtime": {
                **activation.trace_metadata(),
                "entrypoint": activation.entrypoint,
                "output_contract": activation.output_contract,
                "safety_level": activation.safety_level,
                "instructions_loaded": True,
            },
        }
        # 【步骤6】存缓存；同一个入口后续执行使用同一份描述。
        agent_skill_cache[entrypoint] = runtime
        return runtime

    # 【作用】把技能运行轨迹嵌入 Agent 结果（若未激活技能，原样返回）。
    def with_skill_runtime(result: dict[str, Any], entrypoint: str) -> dict[str, Any]:
        # 【步骤】尝试获取 runtime；若存在，dict 展开合并结果并附上 validation 标记。
        runtime = agent_skill_context(entrypoint).get("skill_runtime")
        return {**result, "_skill_runtime": {**runtime, "validation": "entrypoint_completed"}} if runtime else result

    # 【步骤7】为 Planner/Orchestrator 生成状态摘要：存在什么目标、是否明确指定、是否需安全审查。
    # 【例子】有 Issue #42 -> has_issue=True, issue_number=42, issue_id='...'。
    workflow_context = {
        **safe_context,
        "repo_id": str(repo.id),
        "repo_full_name": repo.full_name,
        "has_repo": True,
        "has_issue": selected_issue is not None,
        "issue_id": str(selected_issue.id) if selected_issue else None,
        "issue_number": selected_issue.number if selected_issue else None,
        "issue_explicit": requested_issue_number is not None,
        "has_pr": selected_pr is not None,
        "pr_id": str(selected_pr.id) if selected_pr else None,
        "pr_number": selected_pr.number if selected_pr else None,
        "pr_explicit": requested_pr_number is not None,
        "has_ci_run": selected_run is not None,
        "has_failed_ci": selected_run is not None and selected_run.conclusion == "failure",
        "workflow_run_id": str(selected_run.id) if selected_run else None,
        "workflow_run_number": selected_run.github_run_id if selected_run else None,
        "workflow_run_explicit": requested_run_number is not None,
        "requires_safety": bool(_risky_action(message)),
        "team_members": safe_context.get("team_members") or [],
    }

    # 【作用】工作流子任务：统计 Issue、PR、失败 CI，生成标准 WorkflowTaskResult。
    async def run_repo_health(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        # 【步骤1】复用之前的摘要生成函数，不需要再查询一次数据库。
        answer = _repo_health_answer(repo, issues, prs, runs)
        open_issue_count = len([item for item in issues if item.state == "open"])
        open_pr_count = len([item for item in prs if item.state == "open"])
        # 【步骤2】结构化输出：summary 面向汇总器，output 保存结果，evidence_count 统计基础记录数。
        # 【注意】confidence=0.82 是这里写定的值，并非模型实际置信度计算。
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"{repo.full_name}: {open_issue_count} open issues, {open_pr_count} open PRs, {len(failed_runs)} failed CI runs.",
            output={
                "answer": answer,
                "open_issues": open_issue_count,
                "open_prs": open_pr_count,
                "failed_ci": len(failed_runs),
            },
            evidence_count=len(issues) + len(prs) + len(runs),
            confidence=0.82,
            citations=[{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
        )

    # 【作用】工作流子任务：定位目标 Issue，运行 IssueAgent，保存结果/证据/置信度。
    async def run_issue_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        # 【步骤1】先验证 Planner 的 claim 是否属于当前仓库，不能越过此步直接分析。
        issue = claimed_target(task, "issue", issues, selected_issue)
        # 【步骤2】没有可分析 Issue 时返回 skipped，而不是伪造一个结论。
        if issue is None:
            return _skipped_workflow_result(task, "No synced issue is available.")
        # 【步骤3】await IssueAgent.run：将 ORM Issue 转成普通 dict，额外合并安全 context/技能说明。
        # 【类比】Java 构造 DTO，再调用异步服务拿到结构化返回值。
        result = await IssueAgent().run(
            {
                "id": str(issue.id),
                "title": issue.title,
                "body": issue.body,
                "labels": issue.labels,
                "number": issue.number,
                "state": issue.state,
                "assignees": issue.assignees,
                "author": issue.author,
            },
            {**safe_context, **agent_skill_context("issue_agent.analyze_issue")},
        )
        # 【步骤4】把 Agent 的 conclusion/priority/evidence 等映射到统一结果并附 Issue citation。
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"Issue #{issue.number}: {result.get('conclusion')} / {result.get('priority')} / owner {result.get('suggested_owner')}.",
            output=with_skill_runtime(result, "issue_agent.analyze_issue"),
            evidence_count=len(result.get("evidence") or []),
            confidence=_safe_confidence(result.get("confidence"), 0.62),
            citations=[{"type": "issue", "id": str(issue.id), "title": issue.title}],
        )

    # 【作用】工作流子任务：汇总 PR、变更文件、评论及 CI 摘要，调用 PRReviewAgent。
    async def run_pr_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        # 【步骤1】校验 PR 的 claim，如果没有可用 PR 则跳过。
        pull_request = claimed_target(task, "pull_request", prs, selected_pr)
        if pull_request is None:
            return _skipped_workflow_result(task, "No synced pull request is available.")
        # 【步骤2】在 spec.tasks 中找到第一条 task_type='ci_debug' 的任务，没有则 None。
        ci_task = next((item for item in spec.tasks if item.task_type == "ci_debug"), None)
        # 【步骤3】若计划带 CI 任务，按它声明的 claim 找 CI；否则使用默认 selected_run。
        claimed_ci_run = claimed_target(ci_task, "workflow_run", runs, selected_run) if ci_task else selected_run
        # 【步骤4】为 PR Agent 构造上下文，并附与此 PR 相关的失败 CI 摘要。
        pr_context = {
            **safe_context,
            **agent_skill_context("pr_review_agent.analyze_pr"),
            "repo_id": str(repo.id),
            "ci_summary": {
                "failed_runs": len(failed_runs),
                "latest_failed_run": claimed_ci_run.name if claimed_ci_run and claimed_ci_run.conclusion == "failure" else None,
            },
        }
        # 【步骤5】将 PR、文件 diff patch 和 review comments 扁平化传给 PRReviewAgent。
        # 【注意】files/comments 是列表推导式结果，不直接传 ORM 关联对象。
        result = await PRReviewAgent().run(
            {
                "number": pull_request.number,
                "title": pull_request.title,
                "body": pull_request.body,
                "state": pull_request.state,
                "author": pull_request.author,
                "merged_at": pull_request.merged_at,
                "files": [
                    {
                        "filename": file.filename,
                        "status": file.status,
                        "additions": file.additions,
                        "deletions": file.deletions,
                        "patch": file.patch,
                    }
                    for file in pull_request.files
                ],
                "comments": [
                    {
                        "body": comment.body,
                        "path": comment.path,
                        "line": comment.line,
                        "author": comment.author,
                    }
                    for comment in pull_request.review_comments
                ],
            },
            pr_context,
        )
        # 【步骤6】整理合并建议、风险点数量、证据数和引用，交给编排器汇总。
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"PR #{pull_request.number}: {result.get('merge_recommendation')} with {len(result.get('risk_points') or [])} risk points.",
            output=with_skill_runtime(result, "pr_review_agent.analyze_pr"),
            evidence_count=len(pull_request.files) + len(pull_request.review_comments),
            confidence=_safe_confidence(result.get("confidence"), 0.62),
            citations=[{"type": "pull_request", "id": str(pull_request.id), "title": pull_request.title}],
        )

    # 【作用】工作流子任务：分析指定 CI 运行的 jobs/logs，调用 CIDebugAgent。
    async def run_ci_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        # 【步骤1】按 claim 确认本次分析的 CI WorkflowRun。
        workflow_run = claimed_target(task, "workflow_run", runs, selected_run)
        if workflow_run is None:
            return _skipped_workflow_result(task, "No synced CI run is available.")
        # 【步骤2】传名称、状态、jobs、日志，等待 CIDebugAgent 分析故障。
        result = await CIDebugAgent().run(
            {
                "name": workflow_run.name,
                "status": workflow_run.status,
                "conclusion": workflow_run.conclusion,
                "jobs": workflow_run.jobs or [],
                "logs_text": workflow_run.logs_text,
            },
            {**safe_context, **agent_skill_context("ci_debug_agent.analyze_workflow_run"), "repo_id": str(repo.id)},
        )
        # 【步骤3】打包故障类型、是否阻塞合并、置信度及日志证据数。
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"CI {workflow_run.name}: {result.get('failure_type')}, merge blocking={result.get('is_merge_blocking')}.",
            output=with_skill_runtime(result, "ci_debug_agent.analyze_workflow_run"),
            evidence_count=len(workflow_run.jobs or []) + (1 if workflow_run.logs_text else 0),
            confidence=_safe_confidence(result.get("confidence"), 0.58),
            citations=[{"type": "workflow_run", "id": str(workflow_run.id), "title": workflow_run.name}],
        )

    # 【作用】工作流子任务：RAG 检索历史记录并记录一次 recall（召回）事件。
    async def run_rag_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        # 【步骤1】用户消息非空则检索原消息，否则检索工作流目标描述。
        query = message or spec.goal
        # 【步骤2】拆包三元组：answer 是文字，citations 是引用，results 是检索原始记录。
        answer, citations, results = _semantic_search(repo, query, safe_context, db)
        # 【步骤3】持久化此次召回记录，包含项目/会话作用域、结果和 task_id；用于可追溯性。
        record_recall_event(
            db,
            repo_id=repo.id,
            conversation_id=uuid.UUID(str(context["conversation_id"])) if context.get("conversation_id") else None,
            tool_name="workflow.rag_search",
            query=query,
            results=results,
            citations=citations,
            scope="project",
            mode="hybrid",
            metadata={"route": "workflow_rag_task", "task_id": task.id},
        )
        # 【步骤4】构造 RAG 任务输出；没有检索命中则 confidence 用较低的固定值。
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"Retrieved {len(citations)} related historical records.",
            output={"answer": answer, "citations": citations},
            evidence_count=len(citations),
            confidence=0.68 if citations else 0.45,
            citations=citations,
        )

    # 【作用】工作流子任务：生成潜在写操作的草稿，交 SafetyAgent 评定风险。
    async def run_safety_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        # 【步骤1】识别潜在写操作，识别不到的默认视为 write_back。
        action = _risky_action(message) or "write_back"
        # 【步骤2】生成写操作草稿和来源，不真正写回外部平台。
        draft, citations = _draft_for_action(action, message or spec.goal, repo, issues, prs, runs)
        # 【步骤3】让 SafetyAgent 评定 risk_level、requires_confirmation 等。
        result = await SafetyAgent().run(
            {"action": action, "draft": draft},
            {**safe_context, **agent_skill_context("safety_agent.classify_action_risk")},
        )
        # 【步骤4】把安全审查结果转成统一任务结果，供工作流最终报告使用。
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"Safety risk={result.get('risk_level')}, confirmation required={result.get('requires_confirmation')}.",
            output=with_skill_runtime(result, "safety_agent.classify_action_risk"),
            evidence_count=1,
            confidence=0.9,
            citations=citations,
        )

    # 【步骤8】返回两份东西：一份普通 dict 状态，一份 'task_type' -> async 函数的字典。
    # 【例子】task_runners['pr_review'] 是 run_pr_task 函数对象，此时它并没有执行。
    return workflow_context, {
        "repo_health": run_repo_health,
        "issue_analysis": run_issue_task,
        "pr_review": run_pr_task,
        "ci_debug": run_ci_task,
        "rag_search": run_rag_task,
        "safety_review": run_safety_task,
    }


# 【作用】将工作流总记录及子任务结果写入 SQLAlchemy Session；由外层统一 commit。
# 【注意】工作流表和任务表是主从关系：workflow.id -> AgentTaskRun.workflow_id。
def _persist_engineering_workflow(
    db: Session,
    repo: Repository | None,
    conversation: Conversation | None,
    report: dict[str, Any],
) -> uuid.UUID:
    # 【步骤1】取计划 spec，缺失时用空 dict，避免 .get 访问 None。
    spec = report.get("spec") or {}
    # 【步骤2】根据计划中的任务 ID 建索引，稍后写执行结果时可查回 objective/dependencies/claim。
    # 【例子】spec.tasks=[{'id':'t1',...}] -> {'t1':{'id':'t1',...}}。
    task_by_id = {
        str(task.get("id")): task
        for task in spec.get("tasks", [])
        if isinstance(task, dict) and task.get("id")
    }
    # 【步骤3】构造工作流主表 ORM 实例；此时尚未实际 INSERT。
    # 【注意】只要 error_count 非 0 就标记 status='error'，不是看 final_answer 文本。
    workflow = AgentWorkflowRun(
        repo_id=repo.id if repo else None,
        conversation_id=conversation.id if conversation else None,
        goal=str(spec.get("goal") or ""),
        spec_json=spec,
        observation_json=report.get("observation") or {},
        final_answer=str(report.get("final_answer") or ""),
        metrics=report.get("metrics") or {},
        status="error" if int((report.get("metrics") or {}).get("error_count") or 0) else "success",
        completed_at=datetime.now(timezone.utc),
    )
    # 【步骤4】加入 SQLAlchemy Unit of Work（待持久化列表）。
    # 【步骤7】为每条子任务写 AgentTaskRun，引用上面的 workflow.id。
    # 【注意】此函数只 return UUID，最终事务提交由外层对话函数完成。
    db.add(workflow)
    # 【步骤5】将挂起 SQL 发送到数据库，使 workflow.id 可用于子表外键；但事务还未 commit。
    # 【类比】flush 像‘先执行 SQL 但尚未提交事务’，commit 才提交。
    db.flush()
    # 【步骤6】遍历工作流每个子任务结果；跳过非 dict 的异常元素。
    for result in report.get("task_results") or []:
        if not isinstance(result, dict):
            continue
        task_id = str(result.get("task_id") or "")
        task = task_by_id.get(task_id, {})
        db.add(
            AgentTaskRun(
                workflow_id=workflow.id,
                task_id=task_id,
                agent_name=str(result.get("agent_name") or ""),
                task_type=str(result.get("task_type") or ""),
                claim_json=task.get("claim") if isinstance(task.get("claim"), dict) else {},
                input_json={"objective": task.get("objective"), "dependencies": task.get("dependencies", [])},
                result_json=result,
                status=str(result.get("status") or "unknown"),
                error=result.get("error"),
            )
        )
    return workflow.id


# 【作用】建立模型可调用的工具名 -> 异步 Python 函数映射。
# 【类比】Java 中 Map<String, Function>，但这里回调还是 async，并闭包持有 repo/db。
def _build_conversation_tools(repo: Repository | None, conversation: Conversation | None, db: Session) -> dict[str, Any]:
    """组装当前对话可用的工具表。

    模型选中某个工具名后，编排层会调用这里返回的同名异步函数。
    每个函数都闭包捕获本次请求的 repo / conversation / db，因此工具实现
    不用再自己查“当前是哪个仓库、哪段对话”。

    统一返回结构（编排层靠这些字段拼回复和审计记录）：
    - route: 这次实际走了哪条业务路径
    - tool_calls: 对外可见的工具调用记录（名称、参数、状态）
    - answer: 直接展示给用户的文本
    - citations: 答案引用的仓库对象（Issue、PR、文件等）
    """
    # MCP 记忆工具走独立进程（stdio），这里只建客户端，真正调用发生在下面的闭包里。
    # 【初始化】创建标准输入/输出 MCP 客户端对象，不代表此时已向 MCP 服务发送了请求。
    mcp_client = StdioMcpClient()

    # 【作用】本地工具辅助函数：移除 runtime 内部字段，避免泄露/序列化错误。
    def agent_context(context: dict[str, Any]) -> dict[str, Any]:
        # 下游 Agent 只需要业务上下文。tools 是可调用对象，_skill_registry 是运行时注册表，
        # 传进去既无法序列化，也会把工具实现泄漏给模型提示词。
        return {key: value for key, value in context.items() if key not in {"tools", "_skill_registry"}}

    # 【作用】将请求封装为 MCP stdio 工具调用，再把 MCP 文本段转换成统一工具结果。
    async def call_memory_mcp(tool_name: str, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 三个记忆类工具共用这一层：补齐作用域，调用 MCP，再把文本内容收成对话返回值。
        # 【步骤1】浅拷贝输入参数；避免给原 input_data 强行加 repo_id 等字段。
        arguments = dict(input_data)
        # 模型不一定会自己带上仓库和会话 ID，这里用闭包里的当前对象强制覆盖，避免查到别的项目。
        # 【步骤2】强制覆盖作用域 ID：模型传了假 repo_id 也会被真实 repo.id 替换。
        if repo:
            arguments["repo_id"] = str(repo.id)
        if conversation:
            arguments["conversation_id"] = str(conversation.id)
        # 证据检索的 MCP 入参叫 query，而对话层经常只传用户原文 message。
        # 【步骤3】特殊兼容：证据检索工具需要 query；没有时从 message 补。
        if tool_name == "devflow_search_evidence":
            arguments.setdefault("query", input_data.get("message") or "")
        # 【步骤4】异步调用 MCP 工具；等待外部进程返回内容块列表。
        result = await mcp_client.call_tool(tool_name, arguments)
        # MCP 返回的 content 是多段内容块，这里只拼接 type=text 的段落作为答案。
        # 【步骤5】只提取 content 里的 type='text' 内容并转为 str，跳过非文本块。
        # 【例子】[{'type':'text','text':'A'},{'type':'image'}] -> ['A']。
        text_parts = [
            str(item.get("text") or "")
            for item in result.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        # 【步骤6】拼成统一结果：route + tool_calls 审计 + answer + citations。
        return {
            # devflow_search_evidence -> search_evidence，让 route 和业务名对齐。
            "route": tool_name.replace("devflow_", ""),
            "tool_calls": [{"tool_name": tool_name, "arguments": arguments, "status": "success"}],
            "answer": "\n".join(text_parts).strip(),
            "citations": [],
        }

    # 下面三个函数只是把固定的 MCP 工具名绑到统一入口，方便按工具名注册。
    # 【作用】MCP 包装函数：搜索历史证据。真正处理都委托给 call_memory_mcp。
    async def search_evidence_mcp(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return await call_memory_mcp("devflow_search_evidence", input_data, context)

    # 【作用】MCP 包装函数：获取会话上下文。
    async def get_thread_context_mcp(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return await call_memory_mcp("devflow_get_thread_context", input_data, context)

    # 【作用】MCP 包装函数：读取会话事件。
    async def read_session_events_mcp(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return await call_memory_mcp("devflow_read_session_events", input_data, context)

    # 【作用】工具 workspace.list_files：安全列举本地代码文件，支持路径/子串/数量过滤。
    async def list_workspace_files(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 列出本地 checkout 里的文件，供模型先看目录再决定读哪一个。
        # 【步骤1】定位当前克隆仓库根目录。
        workspace_root = _workspace_root(repo)
        # path 缺省是仓库根目录；_workspace_path 会拦住跳出工作区的路径。
        # 【步骤2】检查用户/模型传入的 path 是否越界。
        root = _workspace_path(workspace_root, str(input_data.get("path") or "."))
        # 条数夹在 1 到 200 之间，避免一次把整棵目录树塞进上下文。
        # 【步骤3】上限夹取：例如 limit=500 -> 200；limit=-2 -> 1；未传 -> 80。
        limit = min(max(int(input_data.get("limit") or 80), 1), 200)
        # 【步骤4】pattern 转小写，做大小写不敏感的路径子串匹配。
        raw_pattern = str(input_data.get("pattern") or "").strip().lower()
        rows: list[str] = []
        # 【步骤5】遍历可读文件，转相对路径，过滤 pattern，达到 limit 提前停止。
        for path in _iter_workspace_files(root, workspace_root):
            relative = _workspace_relative(path, workspace_root)
            # pattern 是子串匹配，不是 glob；空 pattern 表示不过滤。
            if raw_pattern and raw_pattern not in relative.lower():
                continue
            rows.append(relative)
            if len(rows) >= limit:
                break
        # 【步骤6】文件名列表用换行连接；找不到则返回说明文本。
        answer = "\n".join(rows) if rows else "没有找到匹配的工作区文件。"
        # 【步骤7】工具结果提供 route、调用记录、答案和工作区引用。
        return {
            "route": "workspace_list_files",
            "tool_calls": [{"tool_name": "workspace.list_files", "arguments": {"path": str(input_data.get("path") or "."), "pattern": raw_pattern, "limit": limit}, "status": "success"}],
            "answer": answer,
            "citations": [{"type": "workspace", "id": str(workspace_root), "title": repo.full_name if repo else "项目工作区"}],
        }

    # 【作用】工具 workspace.read_file：按起始行和行数读取代码文本。
    async def read_workspace_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 按行窗口读取单个文件。默认 160 行、最多 400 行，防止整文件进入模型上下文。
        # 【步骤1】获取必填路径；空路径直接抛 ValueError。
        raw_path = str(input_data.get("path") or "").strip()
        if not raw_path:
            raise ValueError("path 为必填项")
        workspace_root = _workspace_root(repo)
        # 【步骤2】先校验路径在工作区内，再由 _read_workspace_text 校验文本类型与体积。
        path = _workspace_path(workspace_root, raw_path)
        text = _read_workspace_text(path)
        # 【步骤3】整段 str 按换行切成 list[str]，默认不保留换行符。
        lines = text.splitlines()
        # 【步骤4】起始行至少为 1；例如请求 -5 -> 1。
        start_line = max(int(input_data.get("start_line") or 1), 1)
        # 【步骤5】读取条数范围为 1..400，默认 160。
        line_count = min(max(int(input_data.get("line_count") or 160), 1), 400)
        # 请求的窗口可能超过文件末尾，end_line 截到实际最后一行。
        # 【步骤6】计算最后一行（包含）：start=10, count=3 -> end=12。
        end_line = min(start_line + line_count - 1, len(lines))
        # 带行号返回，方便模型在后续回答里引用具体位置。
        # 【步骤7】range(start,end+1) 取 1 基行号；索引 lines[line_number-1] 转为 Python 0 基。
        # 【例子】lines=['a','b','c']、start=2、end=3 -> ['2: b','3: c']。
        numbered = [f"{line_number}: {lines[line_number - 1]}" for line_number in range(start_line, end_line + 1)]
        relative = _workspace_relative(path, workspace_root)
        # 【步骤8】answer 先放路径/行号区间，再接带行号代码文本。
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": {"path": relative, "start_line": start_line, "line_count": line_count}, "status": "success"}],
            "answer": f"{relative}:{start_line}-{end_line}\n" + "\n".join(numbered),
            "citations": [{"type": "workspace_file", "id": relative, "title": relative}],
        }

    # 【作用】工具 workspace.search_code：对本地 checkout 做关键词检索（不使用向量库）。
    async def search_workspace_code(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 在本地 checkout 里做代码检索。query 和 message 都能当检索词，兼容不同调用方。
        # 【步骤1】优先取 query，否则 message；转 str 并去空白。
        query = str(input_data.get("query") or input_data.get("message") or "").strip()
        # query_terms 抽出有效词；全是停用词或空白时直接拒绝，避免无意义全库扫描。
        # 【步骤2】将 query 分词/提取关键词；无有效词则报错，避免全盘空检索。
        if not query_terms(query):
            raise ValueError("query 为必填项")
        workspace_root = _workspace_root(repo)
        relative_path = str(input_data.get("path") or ".")
        # 先做路径校验（越界会抛错），真正搜索仍把相对路径交给 search_checkout_code。
        # 【步骤3】提前校验路径合法性，即使这里没有使用函数返回的 Path 也有意义。
        _workspace_path(workspace_root, relative_path)
        limit = min(max(int(input_data.get("limit") or 12), 1), 50)
        # 【步骤4】走本地代码搜索引擎，返回每条命中的 snippet、来源文件等。
        # 【区别】这里不是 RAG 向量检索；只操作当前 checkout 文件。
        matches = search_checkout_code(workspace_root, query, limit=limit, relative_path=relative_path)
        # 【步骤5】抽取文本摘要；citations 则提取每条命中的 source_id/title。
        rows = [item["snippet"] for item in matches]
        citations = [
            {"type": "workspace_file", "id": item["source_id"], "title": item["title"]}
            for item in matches
        ]
        # 【步骤6】返回工具调用日志和检索证据，便于前端展示。
        return {
            "route": "workspace_search_code",
            "tool_calls": [{"tool_name": "workspace.search_code", "arguments": {"query": query, "path": relative_path, "limit": limit}, "status": "success"}],
            "answer": "\n".join(rows) if rows else "没有找到匹配的工作区代码。",
            "citations": citations,
        }

    # 【作用】工具 repo_health.generate_summary：返回仓库健康摘要；也是仓库缺失的兜底工具。
    async def repo_health(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 仓库总览，也是其它工具在“还没连仓库”时的统一兜底回复。
        # 【步骤1】当前会话没有仓库就返回提示，避免访问 repo.id 空对象。
        if repo is None:
            # 【步骤3】归一化为工具结果，route 供 ChatAgent 记录做了什么。
            return {"route": "unknown", "tool_calls": [], "answer": "当前还没有可用仓库，请先连接仓库。", "citations": []}
        # 一次取出该仓库已同步的 Issue、PR、CI，再交给摘要函数拼成人话。
        # 【步骤2】获取数据库已同步记录，然后拼健康报告文本。
        issues, prs, runs = _repo_payload(repo.id, db)
        return {
            "route": "repo_health",
            "tool_calls": [{"tool_name": "repo_health.generate_summary", "arguments": {"repo_id": str(repo.id)}, "status": "success"}],
            "answer": _repo_health_answer(repo, issues, prs, runs),
            "citations": [{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
        }

    # 【作用】工具 rag.search_similar_documents：RAG 检索、召回审计和返回片段。
    async def semantic_search(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 混合检索已入库文档（Issue/PR/周报等），和上面的本地代码搜索是两条路径。
        # 【步骤1】没有仓库时回退健康工具；RAG 搜索必须有 repo 范围。
        if repo is None:
            return await repo_health(input_data, context)
        # 【步骤2】保留用户原始话术，以便判断检索 query 是否被上游改写。
        original_message = str(input_data.get("message") or "").strip()
        # query 优先：上游可能已经改写过问句；没有 query 时退回用户原文。
        # 【步骤3】优先使用专门传入的 query；否则原始 message。
        # 【例子】message='为什么报错'，query='CI pytest assertion failed' -> 用后者检索。
        query = str(input_data.get("query") or original_message).strip()
        if not query:
            raise ValueError("query 为必填项")
        # 【步骤4】拿到三元组，answer 用于直接回答，results 用于保存召回日志。
        answer, citations, results = _semantic_search(repo, query, context, db)
        # 把这次召回写进记忆，后面才能回答“刚才依据了哪些文档”。
        # 【步骤5】存储此次检索的 query、结果、引用、作用域及是否经过 query rewrite。
        record_recall_event(
            db,
            repo_id=repo.id,
            conversation_id=conversation.id if conversation else None,
            tool_name="rag.search_similar_documents",
            query=query,
            results=results,
            citations=citations,
            scope="conversation" if conversation else "project",
            mode="hybrid",
            metadata={
                "source_type": context.get("source_type"),
                "limit": context.get("limit", 5),
                "original_message": original_message,
                # 标记问句是否被改写，方便事后对比召回质量。
                "query_rewritten": bool(original_message and query != original_message),
            },
        )
        # 【步骤6】返回 answer 和两套检索结构：
        # retrieved_contexts 是纯文本片段供模型下轮引用；retrieval_results 含 score/source，供 UI 和评测。
        return {
            "route": "semantic_search",
            "tool_calls": [
                {
                    "tool_name": "rag.search_similar_documents",
                    "arguments": {"repo_id": str(repo.id), "query": query},
                    "status": "success",
                }
            ],
            "answer": answer,
            "citations": citations,
            # 纯文本片段，给后续模型轮次当检索上下文。
            "retrieved_contexts": [
                str(item.get("snippet") or "")
                for item in results
                if str(item.get("snippet") or "").strip()
            ],
            # 结构化命中列表，保留分数和来源，供 UI 或评估使用。
            "retrieval_results": [
                {
                    "id": str(item.get("id") or ""),
                    "source_id": str(item.get("source_id") or ""),
                    "source_type": str(item.get("source_type") or ""),
                    "title": str(item.get("title") or ""),
                    "score": item.get("score"),
                    "snippet": str(item.get("snippet") or ""),
                    "retrieval": item.get("retrieval") or {},
                }
                for item in results
            ],
        }

    # 【作用】工具 issue_agent.analyze_issue：分析最近的一条 Issue。
    # 【注意】这条普通对话工具的实现并不按指定编号筛选，和上方工作流 claim 校验不同。
    async def analyze_issue(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 分析该仓库最近同步到的一条 Issue。当前实现不按用户点名的编号筛选。
        # 【步骤1】无仓库则兜底提示，不分析虚构对象。
        if repo is None:
            return await repo_health(input_data, context)
        # 【步骤2】查询仓库中 created_at 最新的 Issue；这里没有根据 input_data 的编号定位。
        issue = db.query(Issue).filter(Issue.repo_id == repo.id).order_by(Issue.created_at.desc()).first()
        if not issue:
            # 【步骤4】通过 f-string 拼成用户友好的 Markdown 文本，并关联 Issue citation。
            return {"route": "issue_analysis", "tool_calls": [], "answer": f"仓库 {repo.full_name} 还没有同步到 Issue。", "citations": []}
        # 只把 Issue 字段交给 IssueAgent；上下文先剥掉 tools 和技能注册表。
        # 【步骤3】把 Issue 属性整理成 dict，异步运行 IssueAgent 获取结论/分类/下一步。
        result = await IssueAgent().run(
            {
                "id": str(issue.id),
                "title": issue.title,
                "body": issue.body,
                "labels": issue.labels,
                "number": issue.number,
                "state": issue.state,
                "assignees": issue.assignees,
                "author": issue.author,
            },
            agent_context(context),
        )
        # checklist 优先，没有则用 action_items；两者都空时给一条默认下一步。
        return {
            "route": "issue_analysis",
            "tool_calls": [{"tool_name": "issue_agent.analyze_issue", "arguments": {"issue_id": str(issue.id)}, "status": "success"}],
            "answer": (
                f"Issue #{issue.number} 的结论：{result['conclusion']}。\n\n"
                f"- 分类：{result['category']}\n"
                f"- 优先级：{result['priority']}\n"
                f"- 复杂度：{result['complexity']}\n"
                f"- 建议负责人：{result['suggested_owner']}，{result['owner_reason']}\n\n"
                f"理由：{result['conclusion_reason']}\n\n"
                f"下一步：{_join_items(result.get('checklist') or result.get('action_items') or [], '补充复现步骤、影响范围和验收标准。')}"
            ),
            "citations": [{"type": "issue", "id": str(issue.id), "title": issue.title}],
        }

    # 【作用】工具 pr_review_agent.analyze_pr：审阅最近同步的一条 PR。
    async def analyze_pr(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 评审该仓库最近一条 PR。selectinload 一次带出改动文件和评审评论，避免后面逐条查库。
        if repo is None:
            return await repo_health(input_data, context)
        # 【步骤1】SQLAlchemy 查询最新 PR，同时用 selectinload 加载变更文件和评论。
        pr = (
            db.query(PullRequest)
            .options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments))
            .filter(PullRequest.repo_id == repo.id)
            .order_by(PullRequest.created_at.desc())
            .first()
        )
        if not pr:
            # 【步骤3】返回 PR 摘要、风险点、测试建议和被引用的 PR。
            return {"route": "pr_review", "tool_calls": [], "answer": f"仓库 {repo.full_name} 还没有同步到 PR。", "citations": []}
        # repo_id 额外塞进上下文，评审 Agent 需要知道代码属于哪个仓库。
        # 【步骤2】拼 PR 数据包，context | {'repo_id':...} 是 Python 字典合并语法。
        # 【例子】{'a':1}|{'b':2} -> {'a':1,'b':2}。
        result = await PRReviewAgent().run(
            {
                "number": pr.number,
                "title": pr.title,
                "body": pr.body,
                # 文件只传文件名、状态和 patch；评论只传正文和位置，不把 ORM 对象送进 Agent。
                "files": [{"filename": file.filename, "status": file.status, "patch": file.patch} for file in pr.files],
                "comments": [{"body": comment.body, "path": comment.path, "line": comment.line, "author": comment.author} for comment in pr.review_comments],
            },
            agent_context(context) | {"repo_id": str(repo.id)},
        )
        return {
            "route": "pr_review",
            "tool_calls": [{"tool_name": "pr_review_agent.analyze_pr", "arguments": {"pr_id": str(pr.id)}, "status": "success"}],
            "answer": f"PR #{pr.number} 摘要：{result['summary']}\n\n主要风险：{_join_items(result['risk_points'], '没有明显风险。')}\n测试建议：{_join_items(result['test_suggestions'], '运行相关测试和核心流程检查。')}",
            "citations": [{"type": "pull_request", "id": str(pr.id), "title": pr.title}],
        }

    # 【作用】工具 ci_debug_agent.analyze_workflow_run：分析最近失败的 CI。
    async def analyze_ci(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 只看最近一次失败的 CI。没有失败记录时不报错，改回仓库健康摘要并说明原因。
        if repo is None:
            return await repo_health(input_data, context)
        # 【步骤1】只查询 conclusion='failure' 的最近一条 CI，order_by desc + first。
        run = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id, WorkflowRun.conclusion == "failure").order_by(WorkflowRun.created_at.desc()).first()
        # 【步骤2】找不到失败 CI 就返回仓库概览，并明确解释无失败记录。
        if not run:
            issues, prs, runs = _repo_payload(repo.id, db)
            # 【步骤4】拼接故障摘要及步骤，附 CI 运行记录 citation。
            return {
                "route": "repo_health",
                "tool_calls": [{"tool_name": "repo_health.generate_summary", "arguments": {"repo_id": str(repo.id), "reason": "no_failed_ci"}, "status": "success"}],
                "answer": _repo_health_answer(repo, issues, prs, runs) + "\n\n说明：这个仓库目前还没有同步到失败 CI 运行。",
                "citations": [{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
            }
        # 把失败运行的名称、状态、job 和日志交给 CIDebugAgent，由它归纳失败类型和排查步骤。
        # 【步骤3】将 jobs/logs 等输入 CIDebugAgent，异步获得失败类型和排查步骤。
        result = await CIDebugAgent().run({"name": run.name, "status": run.status, "conclusion": run.conclusion, "jobs": run.jobs or [], "logs_text": run.logs_text}, agent_context(context) | {"repo_id": str(repo.id)})
        return {
            "route": "ci_debug",
            "tool_calls": [{"tool_name": "ci_debug_agent.analyze_workflow_run", "arguments": {"run_id": str(run.id)}, "status": "success"}],
            "answer": f"最新失败 CI：{run.name}。失败类型：{result['failure_type']}。\n\n{result['failure_summary']}\n\n排查步骤：{_join_items(result['debug_steps'], '没有明确可用的排查步骤。')}",
            "citations": [{"type": "workflow_run", "id": str(run.id), "title": run.name}],
        }

    # 【作用】工具 report_agent.generate_weekly_report：按时间范围生成周报并加入文档库。
    async def generate_report(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 按时间窗汇总 Issue / PR / CI，生成周报，并写回文档库供后续检索。
        if repo is None:
            return await repo_health(input_data, context)
        # 【步骤1】一次获取仓库已同步的全部三类数据。
        issues, prs, runs = _repo_payload(repo.id, db)
        # 时间窗来自对话上下文；缺省时由 _date_window_from_context 给出最近一段时间。
        # 【步骤2】解析时间范围，两个变量同时承接 tuple 结果。
        start_at, end_at = _date_window_from_context(context)
        # 创建、更新、关闭/合并任一时间落在窗口内，就算进这期周报。
        # 【步骤3】列表推导式 + any：Issue 只要创建、更新、关闭之一落在区间就被收录。
        # 【例子】Issue 创建在上个月、昨天更新 -> 这期周报仍收录。
        scoped_issues = [item for item in issues if any(_within(value, start_at, end_at) for value in [item.created_at, item.updated_at, item.closed_at])]
        # 【步骤4】PR 若创建/更新/合并任一时间命中就收录；CI 看创建/更新时间。
        scoped_prs = [item for item in prs if any(_within(value, start_at, end_at) for value in [item.created_at, item.updated_at, item.merged_at])]
        scoped_runs = [item for item in runs if any(_within(value, start_at, end_at) for value in [item.created_at, item.updated_at])]
        # 传给 ReportAgent 的是裁剪后的字段和计数，不是完整 ORM 对象。
        # 【步骤5】将筛选后的对象映射成精简列表和统计指标，交 ReportAgent 生成报告。
        result = await ReportAgent().run(
            {
                "issues": [{"number": item.number, "title": item.title, "state": item.state, "closed_at": item.closed_at} for item in scoped_issues],
                "pull_requests": [{"number": item.number, "title": item.title, "state": item.state, "merged_at": item.merged_at} for item in scoped_prs],
                "workflow_runs": [{"name": item.name, "conclusion": item.conclusion} for item in scoped_runs],
                "metrics": {
                    "issues": len(scoped_issues),
                    "pull_requests": len(scoped_prs),
                    "merged_prs": len([item for item in scoped_prs if item.merged_at]),
                    "open_prs": len([item for item in scoped_prs if item.state == "open"]),
                    "failed_ci": len([item for item in scoped_runs if item.conclusion == "failure"]),
                },
            },
            {"repo_id": str(repo.id), "repo_name": repo.full_name, "start_date": start_at.date().isoformat(), "end_date": end_at.date().isoformat()},
        )
        # commit=False：周报先挂在当前事务上，由外层对话流程统一提交。
        # 【步骤6】把生成的 Markdown 报告加入文档/RAG 系统以便未来检索。
        # 【注意】commit=False 表示沿用当前事务，尚未在这里提交。
        add_document(
            db,
            repo.id,
            "weekly_report",
            None,
            f"研发周报 {start_at.date().isoformat()} - {end_at.date().isoformat()}",
            result["report_markdown"],
            {
                "display_title": f"研发周报 {start_at.date().isoformat()} - {end_at.date().isoformat()}",
                "start_date": start_at.date().isoformat(),
                "end_date": end_at.date().isoformat(),
                "repo_name": repo.full_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "chat",
            },
            commit=False,
        )
        # 【步骤7】返回文本报告、执行日志与仓库引用。
        return {
            "route": "weekly_report",
            "tool_calls": [{"tool_name": "report_agent.generate_weekly_report", "arguments": {"repo_id": str(repo.id)}, "status": "success"}],
            "answer": result["report_markdown"],
            "citations": [{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
        }

    # 【作用】工具 safety_agent.classify_action_risk：生成数据库里的安全草稿和审计记录。
    # 【注意】不会直接调用 GitHub 写入接口。
    async def check_safety(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 写回类请求不直接打 GitHub。先生成草稿，经安全分级后等用户确认。
        if repo is None:
            return await repo_health(input_data, context)
        issues, prs, runs = _repo_payload(repo.id, db)
        # 从用户原文识别动作（评论、建 Issue、关 Issue、发报告等）；识别不到就当成通用写回。
        # 【步骤1】从当前用户话术提取动作，未命中则作为 write_back 草稿。
        action = _risky_action(input_data["message"]) or "write_back"
        # 【步骤2】按动作生成 Issue/PR 评论、创建 Issue、发报告等草稿。
        draft, citations = _draft_for_action(action, input_data["message"], repo, issues, prs, runs)
        # SafetyAgent 可能改写草稿，并给出风险等级和是否需要确认。
        # 【步骤3】让 SafetyAgent 提供风险等级/是否需确认/可能修订的草稿。
        safety = await SafetyAgent().run({"action": action, "draft": draft}, agent_context(context))
        latest_issue = sorted(issues, key=_item_time, reverse=True)[0] if issues else None
        latest_pr = sorted(prs, key=_item_time, reverse=True)[0] if prs else None
        # 默认挂在仓库上；能对应到最近的 PR 或 Issue 时，把草稿目标收窄到那条记录。
        # 【步骤4】先默认归属 repository；如属于评论，则根据最近 PR/Issue 指定关联对象。
        target_type = "repository"
        target_id = None
        metadata: dict[str, Any] = {"source": "chat", "message": input_data["message"]}
        # 【步骤5】comment 优先关联最新 PR；否则可能关联最新 Issue。
        if action == "comment" and latest_pr:
            target_type = "pull_request"
            target_id = latest_pr.id
            metadata["pr_number"] = latest_pr.number
        elif action in {"comment", "create_issue", "close_issue"} and latest_issue:
            target_type = "issue"
            target_id = latest_issue.id
            metadata["issue_number"] = latest_issue.number
        # 对外动作名和草稿类型不是一一对应：comment 要区分 PR 评论还是 Issue 评论。
        # 【步骤6】把通用动作映射成数据库草稿类别；同样是 comment，PR 和 Issue 使用不同类型。
        draft_type = {
            "comment": "pr_comment" if target_type == "pull_request" else "issue_comment",
            "create_issue": "create_issue",
            "send_report": "send_report",
            "write_back": "comment",
        }.get(action, action)
        # 【步骤7】取出/创建审计主体用户，记录是谁触发了草稿创建。
        user = ensure_demo_user(db)
        # 草稿只入库，不执行。flush 是为了立刻拿到 id，写进审计日志和返回文案。
        # 【步骤8】构造待审核的 ActionDraft ORM 对象，内容采用 SafetyAgent 返回稿优先。
        action_draft = ActionDraft(
            repo_id=repo.id,
            draft_type=draft_type,
            target_type=target_type,
            target_id=target_id,
            title=f"{draft_type} 草稿",
            content=str(safety.get("draft") or draft),
            risk_level=str(safety.get("risk_level") or "medium"),
            created_by=user.id,
            meta=metadata,
        )
        # 【步骤9】加入 Session；db.flush() 立即获取草稿 ID，但未 commit。
        db.add(action_draft)
        db.flush()
        # 记录“从对话创建了草稿”，真正写回发生在用户确认之后的另一条路径。
        # 【步骤10】写审计记录，包含动作、目标、草稿 ID，支持后续追踪和审批。
        write_audit_log(
            db,
            user=user,
            repo_id=repo.id,
            action="action_draft.create_from_chat",
            target_type=target_type,
            target_id=str(target_id) if target_id else None,
            request_json={"action": action, "draft_type": draft_type},
            result_json={"draft_id": str(action_draft.id)},
        )
        # 【步骤11】返回‘待确认草稿’及草稿 ID，不会在这一函数内写到 GitHub。
        return {
            "route": "safety_draft",
            "tool_calls": [{"tool_name": "safety_agent.classify_action_risk", "arguments": {"action": action, "draft_id": str(action_draft.id)}, "status": "success"}],
            "answer": f"这个请求可能写入 GitHub 或发送到外部渠道，已生成待确认安全草稿。\n\n草稿 ID：{action_draft.id}\n\n{str(safety.get('draft') or draft)}",
            "citations": citations,
        }

    # 【作用】工具 workflow.run_engineering_review：一键规划/运行多 Agent 工作流并保存报告。
    async def run_engineering_workflow(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 多 Agent 编排入口：规划、执行仓库健康 / Issue / PR / CI / 检索 / 安全审查，再汇总成一份报告。
        # 【步骤1】必须有当前仓库，才能建立工程工作流上下文。
        if repo is None:
            return await repo_health(input_data, context)

        # 【步骤2】提取要完成的工程任务目标，query 可以作为 message 的兜底。
        message = str(input_data.get("message") or input_data.get("query") or "").strip()
        # runtime 返回两样东西：给编排器的上下文，以及 task_type -> 执行函数 的映射。
        # 【步骤3】拿工作流上下文和按 task_type 索引的异步执行函数。
        workflow_context, task_runners = _build_engineering_workflow_runtime(repo, db, message, context)
        # 【步骤4】Orchestrator 负责创建/执行任务计划，调用所需 Agent，返回报告。
        report = await WorkflowOrchestrator(task_runners).run({"message": message or "Run an engineering workflow."}, workflow_context)
        # 整次 workflow 和每个子任务都落库，返回的 id 挂到指标上，方便前端回查。
        # 【步骤5】将工作流及任务结果写入 Session；把可查询的工作流 ID 放进 metrics。
        workflow_run_id = _persist_engineering_workflow(db, repo, conversation, report)
        report["metrics"]["workflow_run_id"] = str(workflow_run_id)
        # 【步骤6】把编排报告归一化为普通工具结果，交给 ChatAgent 使用。
        return _engineering_workflow_response(repo, workflow_run_id, report)

    # 键是模型看到的工具名，值是上面的闭包。编排层按名字取函数，不会直接 import 这些内部函数。
    return {
        # 【关键】下面是最终工具注册表：字符串是 LLM 可调用的工具名，值是 Python 异步函数对象。
        # 【例子】tools['workspace.list_files'](...) 才会真正调用 list_workspace_files。
        "workspace.list_files": list_workspace_files,
        "workspace.read_file": read_workspace_file,
        "workspace.search_code": search_workspace_code,
        "repo_health.generate_summary": repo_health,
        "rag.search_similar_documents": semantic_search,
        "issue_agent.analyze_issue": analyze_issue,
        "pr_review_agent.analyze_pr": analyze_pr,
        "ci_debug_agent.analyze_workflow_run": analyze_ci,
        "report_agent.generate_weekly_report": generate_report,
        "safety_agent.classify_action_risk": check_safety,
        "workflow.run_engineering_review": run_engineering_workflow,
        "devflow_search_evidence": search_evidence_mcp,
        "devflow_get_thread_context": get_thread_context_mcp,
        "devflow_read_session_events": read_session_events_mcp,
    }


# 【作用】把当前技能注册表的工具提示、描述和运行追踪注入 Agent context。
def _skill_runtime_context() -> dict[str, Any]:
    # 【步骤1】读取全局技能注册表，不意味着每一项技能在这里被执行。
    registry = get_skill_registry()
    # 【步骤2】返回 prompt_context、按工具的 trace 元数据和工具描述后缀。
    # 【注意】_skill_registry 是内部 Python 对象，后续传模型前会被过滤。
    return {
        "_skill_registry": registry,
        "registered_skills": registry.prompt_context(),
        "skill_trace_by_tool": registry.trace_metadata_by_tool(),
        "skill_description_by_tool": registry.tool_description_suffixes(),
    }


# 【作用】把 WorkflowOrchestrator 报告统一包装为普通对话工具结果。
def _engineering_workflow_response(repo: Repository, workflow_run_id: uuid.UUID, report: dict[str, Any]) -> dict[str, Any]:
    # 【步骤】把 report 中的 tool_calls 放到当前工具调用后面。
    # 【例子】[当前 workflow 工具记录, *子任务工具调用记录]；星号是列表展开。
    return {
        "route": "engineering_workflow",
        "tool_calls": [
            {
                "tool_name": "workflow.run_engineering_review",
                "arguments": {"repo_id": str(repo.id), "workflow_run_id": str(workflow_run_id)},
                "status": "success",
            },
            *report.get("tool_calls", []),
        ],
        "answer": report.get("final_answer") or "",
        "citations": report.get("citations") or [],
        "agent_events": report.get("agent_events") or [],
    }


# 【作用】将结构化 WorkflowSpec 变成用户可阅读的任务计划说明。
def _workflow_plan_answer(spec: WorkflowSpec) -> str:
    # 【步骤1】先生成固定的前言和目标行。
    lines = [
        "已创建执行计划，确认后我会按这个 WorkflowSpec 执行。",
        "",
        f"目标：{spec.goal}",
        "",
        "任务：",
    ]
    # 【步骤2】遍历每条 WorkflowTask，打印序号、Agent、任务类型、目标。
    for index, task in enumerate(spec.tasks, start=1):
        # 【步骤3】有依赖才附加依赖 ID；关键任务附加标识。
        deps = f"；依赖：{', '.join(task.dependencies)}" if task.dependencies else ""
        critical = "；关键任务" if task.critical else ""
        lines.append(f"{index}. {task.agent_name} / {task.task_type}: {task.objective}{deps}{critical}")
    # 【步骤4】若存在验收标准，再插入清单。
    if spec.acceptance_criteria:
        lines.extend(["", "验收标准："])
        lines.extend(f"- {item}" for item in spec.acceptance_criteria)
    # 【步骤5】将行列表拼为最终面向用户的 Markdown 计划。
    return "\n".join(lines)


# 【作用】仅生成执行计划，不运行实际子任务；用户确认后才走执行接口。
async def _conversation_plan(payload: ChatRequest, db: Session) -> dict[str, Any]:
    # 【步骤1】确认请求里的 repo_id 对应真实仓库。
    repo = _repo_or_404(db, payload.repo_id)
    # 【步骤2】关联或创建对话，确保得到 conversation.id。
    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    db.commit()
    # 【步骤3】合并客户端允许字段、服务端技能元数据及真实仓库/会话 ID。
    # 【注意】后展开的 repo_id/conversation_id 会覆盖同名旧值。
    context = {
        **_client_context(payload.context),
        **_skill_runtime_context(),
        "repo_id": str(repo.id),
        "conversation_id": str(conversation.id),
    }
    # 【步骤4】构建工作流能力画像；_task_runners 前缀下划线表示本函数暂不使用它。
    workflow_context, _task_runners = _build_engineering_workflow_runtime(repo, db, payload.message, context)
    # 【步骤5】调用 PlannerAgent 生成结构化计划草案，这一步不运行 Issue/PR/CI 子任务。
    spec_data = await PlannerAgent().run({"message": payload.message}, workflow_context)
    # 【步骤6】Pydantic 校验生成内容满足 WorkflowSpec 的字段/类型规则，错误会抛异常。
    spec = WorkflowSpec.model_validate(spec_data)
    # 【步骤7】结构化计划转 JSON 兼容 dict，提供前端下一步确认和回传。
    spec_json = spec.model_dump(mode="json")
    # 【步骤8】同时返回 plan JSON、可读 answer、工具轨迹、workflow_spec 事件。
    return {
        "route": "engineering_plan",
        "conversation_id": str(conversation.id),
        "answer": _workflow_plan_answer(spec),
        "plan": spec_json,
        "tool_calls": [
            {
                "tool_name": "planner_agent.create_workflow_spec",
                "arguments": {"repo_id": str(repo.id), "workflow_id": spec.workflow_id},
                "status": "success",
            }
        ],
        "agent_events": [
            {
                "type": "workflow_spec",
                "workflow_id": spec.workflow_id,
                "goal": spec.goal,
                "tasks": [
                    {
                        "id": task.id,
                        "agent_name": task.agent_name,
                        "task_type": task.task_type,
                        "dependencies": task.dependencies,
                        "claim": task.claim.model_dump(mode="json"),
                    }
                    for task in spec.tasks
                ],
            }
        ],
    }


# 【作用】将内部工作流 Agent 事件适配为前端认识的 tool_result SSE 事件。
def _workflow_execution_trace_events(report: dict[str, Any], workflow_tool_id: str) -> list[dict[str, Any]]:
    # 【步骤1】先添加整个确认工作流的 tool_result，告诉前端顶层工具完成。
    events = [
        {
            "event": "tool_result",
            "data": {
                "tool_use_id": workflow_tool_id,
                "tool_name": "workflow.run_engineering_review",
                "status": "success",
                "route": "engineering_workflow",
                "content": {
                    "answer_preview": "Confirmed plan executed.",
                    "citations_count": len(report.get("citations") or []),
                },
            },
        }
    ]
    # 【步骤2】扫描 report.agent_events；不是 dict 的元素直接跳过。
    for item in report.get("agent_events") or []:
        if not isinstance(item, dict):
            continue
        event_type = item.get("type")
        # 【步骤3A】普通子任务完成：映射成 workflow.<任务类型> 的 tool_result。
        if event_type == "workflow_task_result":
            events.append(
                {
                    "event": "tool_result",
                    "data": {
                        "tool_use_id": f"{item.get('workflow_id')}_{item.get('task_id')}",
                        "tool_name": f"workflow.{item.get('task_type')}",
                        "status": item.get("status") or "success",
                        "route": "engineering_workflow",
                        "content": {
                            "answer_preview": item.get("summary") or "",
                            "citations_count": item.get("evidence_count") or 0,
                        },
                    },
                }
            )
        # 【步骤3B】工作流重新规划：生成 planner_agent.replan_workflow_spec 轨迹，说明新增任务。
        elif event_type == "workflow_replan":
            events.append(
                {
                    "event": "tool_result",
                    "data": {
                        "tool_use_id": f"{item.get('workflow_id')}_replan_{item.get('iteration')}",
                        "tool_name": "planner_agent.replan_workflow_spec",
                        "status": "success",
                        "route": "engineering_workflow",
                        "content": {
                            "answer_preview": (
                                f"Replan #{item.get('iteration')}: {item.get('reason')}; "
                                f"added tasks={', '.join(str(task) for task in item.get('added_tasks') or []) or 'none'}"
                            ),
                            "citations_count": 0,
                        },
                    },
                }
            )
        # 【步骤3C】观察器总结：生成 workflow.observer 事件，包含总体 confidence 与 findings 数量。
        elif event_type == "workflow_observer_result":
            events.append(
                {
                    "event": "tool_result",
                    "data": {
                        "tool_use_id": f"{item.get('workflow_id')}_observer",
                        "tool_name": "workflow.observer",
                        "status": "success",
                        "route": "engineering_workflow",
                        "content": {
                            "answer_preview": f"Observer confidence={item.get('overall_confidence')}, findings={len(item.get('findings') or [])}",
                            "citations_count": 0,
                        },
                    },
                }
            )
    # 【步骤4】返回一组待发送的 SSE 事件 dict；这里只构造，不执行 yield。
    return events


# 【作用】执行已确认的 WorkflowSpec，并通过异步生成器 yield SSE 帧。
# 【主线】准备上下文 -> 运行编排 -> 发事件/回答 -> 保存结果；异常则回滚并发错误 final。
async def _conversation_plan_execute_sse(payload: ChatPlanExecuteRequest, db: Session):
    # 【准备1】确保计划执行的仓库存在，否则在开始 streaming 前就报错。
    repo = _repo_or_404(db, payload.repo_id)

    # 【准备2】读取/创建会话并提交，使后续消息拥有稳定的 conversation_id。
    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    if conversation:
        db.commit()
    # 【准备3】使用渐进式上下文管理器检查 token 余量，必要时压缩长历史。
    compaction_stats = await ProgressiveContextManager(db).ensure_headroom(repo, conversation, payload.message)
    # 【准备4】发生压缩尝试时提交数据库，以免后续组装时读不到变更。
    if compaction_stats.get("attempted"):
        db.commit()
    message_store = MessageStore(db)
    # 【准备5】记录用户这条消息；append 返回 ChatMessage 对象。
    user_message = message_store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content=payload.message)
    # 【准备6】基于仓库和会话刷新可用证据记录，并提交用户消息及证据状态。
    EvidenceStore(db).rebuild_repo(repo, conversation)
    db.commit()
    # 【准备7】组装系统上下文、历史消息、证据、压缩统计，供工作流运行。
    assembled = ContextAssembler(db).assemble(repo, conversation, payload.message)
    # 【准备8】将客户端可用 context 与服务端组装结果合并，服务端字段置于后面确保覆盖。
    context = {
        **_client_context(payload.context),
        **_skill_runtime_context(),
        "repo_id": str(repo.id),
        "conversation_id": str(conversation.id),
        "server_context": assembled.system_context,
        "server_messages": assembled.history_messages,
        "evidence": assembled.evidence,
        "compression_stats": assembled.compression_stats,
        "progressive_compaction": compaction_stats,
    }
    # 【准备9】构造此次工作流工具调用的唯一 ID；若规范里已有 workflow_id 就用它。
    workflow_tool_id = f"confirmed_{payload.workflow_spec.get('workflow_id') or uuid.uuid4().hex[:8]}"
    try:
        # 【执行1】再次校验前端传回的 workflow_spec，防止确认后传入无效结构。
        spec = WorkflowSpec.model_validate(payload.workflow_spec)
        # 【执行2】首先 yield tool_use 事件，前端可立即显示“工作流开始执行”。
        # 【Python】yield 表示产出一帧；函数之后仍可恢复执行，不是 return 结束。
        yield encode_sse(
            {
                "event": "tool_use",
                "data": {
                    "id": workflow_tool_id,
                    "tool_name": "workflow.run_engineering_review",
                    "arguments": {"repo_id": str(repo.id), "workflow_id": spec.workflow_id, "confirmed_plan": True},
                },
            }
        )
        # 【执行3】重新构造当前仓库的目标与任务运行器，不盲目信任前端带来的上下文。
        workflow_context, task_runners = _build_engineering_workflow_runtime(repo, db, payload.message, context)
        # 【执行4】run_spec 使用已确认的 spec 执行工作流；结果包括 report、task_results 和 events。
        report = await WorkflowOrchestrator(task_runners).run_spec(spec, workflow_context)
        # 【执行5】将工作流执行结果记录到数据库 Session，取回可回查的 workflow_run_id。
        workflow_run_id = _persist_engineering_workflow(db, repo, conversation, report)
        report["metrics"]["workflow_run_id"] = str(workflow_run_id)
        # 【执行6】逐个 yield 子任务结果/重规划/观察者的进度事件。
        for event in _workflow_execution_trace_events(report, workflow_tool_id):
            yield encode_sse(event)

        # 【执行7】合并工作流答案、引用和 tool_calls，补 run_id 与压缩元数据。
        result = _engineering_workflow_response(repo, workflow_run_id, report)
        run_id = uuid.uuid4()
        result["run_id"] = str(run_id)
        result["compression_stats"] = assembled.compression_stats
        result["progressive_compaction"] = compaction_stats
        # 【执行8】将完整最终回答每 48 个字符切成一片，模拟 delta SSE 输出。
        # 【例子】answer 长 110 字符 -> 下标 [0:48]、[48:96]、[96:144] 三片。
        # 【注意】这是事后对完整答案分段，不代表模型正在逐 token 生成。
        for index in range(0, len(result["answer"]), 48):
            yield encode_sse({"event": "delta", "data": {"content": result["answer"][index : index + 48]}})

        # 【保存1】保存最终 assistant 消息，并写入事件/模型调用/上下文压缩元数据。
        assistant_message = message_store.append(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="assistant",
            content=result["answer"],
            route=result["route"],
            tool_calls=result["tool_calls"],
            meta={
                "run_id": str(run_id),
                "agent_events": result.get("agent_events", []),
                "model_usage": result.get("model_usage", []),
                "context_diagnostics": result.get("context_diagnostics", {}),
                "compression_stats": assembled.compression_stats,
                "progressive_compaction": compaction_stats,
            },
        )
        # 【保存2】给前端结果加数据库 ChatMessage ID，后续可以按消息追踪。
        result["message_id"] = str(assistant_message.id)
        # 【保存3】添加本次 AgentRun：用户输入、路由、工具调用、最终回答及 success 状态。
        db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route=result["route"], tool_calls=result["tool_calls"], final_answer=result["answer"], status="success"))
        # 【保存4】更新时间/消息数量，increment=2 对应本轮 user + assistant 两条。
        touch_conversation(db, conversation, user_message=payload.message, increment=2)
        # 【保存5】把当前轮关键消息与事件封存成会话记忆/摘要，便于以后恢复上下文。
        await SessionSealer(db).seal(
            repo=repo,
            conversation=conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            route=result["route"],
            tool_calls=result["tool_calls"],
            agent_events=result.get("agent_events", []),
            model_usage=result.get("model_usage", []),
            context_diagnostics=result.get("context_diagnostics", {}),
        )
        # 【保存6】提交本轮所有待写 SQL；此前 flush 但未 commit 的数据在这里真正提交。
        db.commit()
        # 【返回】最终再发 final 事件，客户端可停止 loading 并展示完整元数据。
        yield encode_sse({"event": "final", "data": result})
    # 【异常路径】任意执行阶段抛异常时，进入错误收尾逻辑。
    # 【注意】这只捕获 try 内错误；前面准备阶段的异常可能由框架处理。
    except Exception as exc:
        # 【异常1】回滚当前未提交事务，防止残缺的工作流记录残留。
        db.rollback()
        run_id = uuid.uuid4()
        # 【异常2】生成错误答复，并用新 run_id 关联失败记录。
        answer = f"执行计划失败：{exc}"
        # 【异常3】另起当前 Session 工作单元记录一条失败的 assistant 消息。
        assistant_message = message_store.append(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            route="error",
            tool_calls=[],
            meta={"run_id": str(run_id), "error": str(exc)},
        )
        db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route="error", tool_calls=[], final_answer=answer, status="error"))
        touch_conversation(db, conversation, user_message=payload.message, increment=2)
        await SessionSealer(db).seal(
            repo=repo,
            conversation=conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            route="error",
            tool_calls=[],
            agent_events=[],
        )
        # 【异常4】提交错误日志和会话消息；异常响应同样持久化。
        db.commit()
        yield encode_sse(
            {
                "event": "final",
                "data": {
                    "run_id": str(run_id),
                    "message_id": str(assistant_message.id),
                    "route": "error",
                    "tool_calls": [],
                    "answer": answer,
                    "citations": [],
                    "agent_events": [],
                },
            }
        )


# 【作用】同步 HTTP JSON 对话：一次 ChatAgent.run 完成后统一落库并返回。
# 【注意】这里的“同步”指客户端得到整包 JSON；函数本身仍是 async def。
async def _conversation_response(payload: ChatRequest, db: Session) -> dict[str, Any]:
    # 【步骤1】验证仓库 ID，查不到就直接 HTTP 404。
    repo = _repo_or_404(db, payload.repo_id)
    # 【步骤2】获取/创建会话并提交，确保可引用 conversation.id。
    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    db.commit()
    # 【步骤3】检查上下文窗口是否足够；长会话可能发生渐进压缩。
    compaction_stats = await ProgressiveContextManager(db).ensure_headroom(repo, conversation, payload.message)
    if compaction_stats.get("attempted"):
        db.commit()
    message_store = MessageStore(db)
    # 【步骤4】保存用户输入消息，作为本轮对话的起点。
    user_message = message_store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content=payload.message)
    # 【步骤5】刷新证据库，确保对话组装时能读到更新后的材料。
    EvidenceStore(db).rebuild_repo(repo, conversation)
    db.commit()
    # 【步骤6】从历史消息、证据和摘要组装模型上下文。
    assembled = ContextAssembler(db).assemble(repo, conversation, payload.message)
    # 【步骤7】客户端 context 与服务端 context 合并；后者的权威字段覆盖前者。
    context = {
        **_client_context(payload.context),
        **_skill_runtime_context(),
        "repo_id": str(repo.id),
        "conversation_id": str(conversation.id),
        "server_context": assembled.system_context,
        "server_messages": assembled.history_messages,
        "evidence": assembled.evidence,
        "compression_stats": assembled.compression_stats,
        "progressive_compaction": compaction_stats,
    }
    # 【步骤8】创建可用工具字典：ChatAgent 内部可能按名称调用这些闭包。
    context["tools"] = _build_conversation_tools(repo, conversation, db)
    # 【步骤9】等待 ChatAgent 整轮执行完成，得到 answer/route/tool_calls/events 等。
    # 【例子】用户问'分析最新 PR' -> ChatAgent 可能调用 pr_review_agent.analyze_pr。
    result = await ChatAgent().run({"message": payload.message}, context)
    # 【步骤10】生成独立 AgentRun ID（不同于 conversation_id、message_id）。
    run_id = uuid.uuid4()
    result["run_id"] = str(run_id)
    result["compression_stats"] = assembled.compression_stats
    result["progressive_compaction"] = compaction_stats
    # 【步骤11】类型标注相当于 Java `ChatMessage assistantMessage`，`| None` 表示可能为空。
    assistant_message: ChatMessage | None = None
    # 【步骤12】将助手完整回复及其调用轨迹/压缩元数据持久化为 ChatMessage。
    assistant_message = message_store.append(
        repo_id=repo.id,
        conversation_id=conversation.id,
        role="assistant",
        content=result["answer"],
        route=result["route"],
        tool_calls=result["tool_calls"],
        meta={
            "run_id": str(run_id),
            "agent_events": result.get("agent_events", []),
            "model_usage": result.get("model_usage", []),
            "context_diagnostics": result.get("context_diagnostics", {}),
            "compression_stats": assembled.compression_stats,
            "progressive_compaction": compaction_stats,
        },
    )
    # 【步骤13】将已保存消息的 ID 补到要返回的字典。
    result["message_id"] = str(assistant_message.id)
    # 【步骤14】另外写一条 AgentRun 执行审计记录。
    db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route=result["route"], tool_calls=result["tool_calls"], final_answer=result["answer"], status="success"))
    # 【步骤15】刷新会话时间戳并增加两条消息计数。
    touch_conversation(db, conversation, user_message=payload.message, increment=2)
    # 【步骤16】封存本轮的消息与工具/Agent 元数据，供会话记忆使用。
    await SessionSealer(db).seal(
        repo=repo,
        conversation=conversation,
        user_message=user_message,
        assistant_message=assistant_message,
        route=result["route"],
        tool_calls=result["tool_calls"],
        agent_events=result.get("agent_events", []),
        model_usage=result.get("model_usage", []),
        context_diagnostics=result.get("context_diagnostics", {}),
    )
    # 【步骤17】统一提交该轮助手消息、AgentRun 和会话变化。
    db.commit()
    # 【步骤18】返回完整 JSON；与 SSE 不同，这条链路不逐段 yield。
    return result


# 【作用】流式对话：async for 读取 ChatAgent 事件，逐个 yield SSE。
# 【关键】仅在收到 final 事件时保存最终助手消息、运行记录和压缩元数据。
async def _conversation_sse(payload: ChatRequest, db: Session):
    # 【准备1】和 JSON 版本相同：先验证仓库、确认会话、压缩历史。
    repo = _repo_or_404(db, payload.repo_id)
    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    db.commit()
    # 【准备2】为本轮对话留足上下文空间，必要时压缩。
    compaction_stats = await ProgressiveContextManager(db).ensure_headroom(repo, conversation, payload.message)
    if compaction_stats.get("attempted"):
        db.commit()
    message_store = MessageStore(db)
    # 【准备3】先保存用户输入，再重建证据，提交后组装最终模型上下文。
    user_message = message_store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content=payload.message)
    EvidenceStore(db).rebuild_repo(repo, conversation)
    db.commit()
    # 【准备4】读取消息历史、系统上下文与证据，形成 context。
    assembled = ContextAssembler(db).assemble(repo, conversation, payload.message)
    context = {
        **_client_context(payload.context),
        **_skill_runtime_context(),
        "repo_id": str(repo.id),
        "conversation_id": str(conversation.id),
        "server_context": assembled.system_context,
        "server_messages": assembled.history_messages,
        "evidence": assembled.evidence,
        "compression_stats": assembled.compression_stats,
        "progressive_compaction": compaction_stats,
    }
    # 【准备5】把同一套工具映射注册到 ChatAgent。
    context["tools"] = _build_conversation_tools(repo, conversation, db)
    try:
        # 【流式1】异步遍历 ChatAgent 的事件流；每轮得到一个 event dict。
        # 【类比】Java WebFlux/Flux 的逐项消费；此时模型可能还没生成完全部答案。
        async for event in ChatAgent().run_stream({"message": payload.message}, context):
            # 【流式2】遇到 final 代表模型整轮完成，才做最终数据落库。
            # 【例子】delta('你') -> delta('好') -> final(answer='你好')；前两帧不写 assistant 消息。
            if event["event"] == "final":
                # 【流式3】给最终运行记录生成 run_id，补进 final 事件的数据。
                run_id = uuid.uuid4()
                event["data"]["run_id"] = str(run_id)
                event["data"]["compression_stats"] = assembled.compression_stats
                event["data"]["progressive_compaction"] = compaction_stats
                # 【流式4】保存完整助手答案、route、工具调用和 Agent 事件元数据。
                assistant_message = message_store.append(
                    repo_id=repo.id,
                    conversation_id=conversation.id,
                    role="assistant",
                    content=event["data"]["answer"],
                    route=event["data"]["route"],
                    tool_calls=event["data"]["tool_calls"],
                    meta={
                        "run_id": str(run_id),
                        "agent_events": event["data"].get("agent_events", []),
                        "model_usage": event["data"].get("model_usage", []),
                        "context_diagnostics": event["data"].get("context_diagnostics", {}),
                        "compression_stats": assembled.compression_stats,
                        "progressive_compaction": compaction_stats,
                    },
                )
                # 【流式5】补上数据库消息 ID，前端据此可关联历史记录。
                event["data"]["message_id"] = str(assistant_message.id)
                # 【流式6】保存 AgentRun 审计记录；同时 touch 会话、seal 记忆、commit。
                db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route=event["data"]["route"], tool_calls=event["data"]["tool_calls"], final_answer=event["data"]["answer"], status="success"))
                touch_conversation(db, conversation, user_message=payload.message, increment=2)
                await SessionSealer(db).seal(
                    repo=repo,
                    conversation=conversation,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    route=event["data"]["route"],
                    tool_calls=event["data"]["tool_calls"],
                    agent_events=event["data"].get("agent_events", []),
                    model_usage=event["data"].get("model_usage", []),
                    context_diagnostics=event["data"].get("context_diagnostics", {}),
                )
                db.commit()
            # 【流式7】不论 delta、tool_use、tool_result 还是 final，都编码为 SSE 文本帧发送给前端。
            # 【示例】event={'event':'delta','data':{'content':'你好'}} -> encode_sse(event) 输出事件帧字符串。
            yield encode_sse(event)
    # 【错误路径】遍历 run_stream 或持久化期间异常 -> 回滚未提交修改。
    except Exception as exc:
        db.rollback()
        run_id = uuid.uuid4()
        answer = f"请求失败：{exc}"
        # 【错误1】构造 event='final'、route='error' 的错误帧，保证前端收尾。
        failed_event = {
            "event": "final",
            "data": {
                "run_id": str(run_id),
                "route": "error",
                "tool_calls": [],
                "answer": answer,
                "citations": [],
            },
        }
        # 【错误2】保存错误回答和失败 AgentRun，方便历史页面/审计看到失败原因。
        assistant_message = message_store.append(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            route="error",
            tool_calls=[],
            meta={"run_id": str(run_id), "error": str(exc)},
        )
        failed_event["data"]["message_id"] = str(assistant_message.id)
        db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route="error", tool_calls=[], final_answer=answer, status="error"))
        touch_conversation(db, conversation, user_message=payload.message, increment=2)
        await SessionSealer(db).seal(
            repo=repo,
            conversation=conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            route="error",
            tool_calls=[],
            agent_events=[],
        )
        # 【错误3】提交错误状态后 yield 错误 final，客户端不必无限等待。
        db.commit()
        yield encode_sse(failed_event)


# 【入口】POST /plan：规划 Agent 任务。明确目标不存在时转成 HTTP 422。
@router.post("/plan")
async def chat_plan(payload: ChatRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    # HTTP 入口很薄：业务都委托给 _conversation_plan，并只针对特定异常转 HTTP 422。
    try:
        return await _conversation_plan(payload, db)
    except WorkflowTargetResolutionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# 【入口】POST /plan/execute/stream：执行用户确认过的计划，返回 text/event-stream。
@router.post("/plan/execute/stream")
async def chat_plan_execute_stream(payload: ChatPlanExecuteRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    # StreamingResponse 包装 async generator；media_type='text/event-stream' 告诉浏览器按 SSE 接收。
    return StreamingResponse(_conversation_plan_execute_sse(payload, db), media_type="text/event-stream")


# 【入口】POST /（相对于这个 APIRouter 的前缀）：普通聊天，完整 JSON 返回。
@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    # 调用完整 JSON 对话链路；FastAPI 根据 response_model=ChatResponse 校验/序列化响应。
    return await _conversation_response(payload, db)


# 【入口】POST /stream：普通聊天，流式 SSE 返回。
@router.post("/stream")
async def chat_stream(payload: ChatRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    # 将 _conversation_sse(...) 生成器交给 FastAPI；前端持续收到事件直到 final。
    return StreamingResponse(_conversation_sse(payload, db), media_type="text/event-stream")


# 【入口】POST /recommendations：推荐下一步可以问 Agent 的提示词。
@router.post("/recommendations", response_model=PromptRecommendationResponse)
async def prompt_recommendations(payload: PromptRecommendationRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    # 【步骤1】确认仓库存在。
    repo = _repo_or_404(db, payload.repo_id)
    # 【步骤2】从数据库取得 Issue、PR、CI 状况。
    issues, prs, runs = _repo_payload(repo.id, db)
    # 【步骤3】构造提示词推荐所需的数据包；exclude 排除已展示推荐，limit 限制数量。
    result = await RecommendationAgent().run(
        _recommendation_payload(repo, issues, prs, runs, payload.exclude, payload.limit),
        {"repo_id": str(repo.id)},
    )
    # 【步骤4】只返回推荐 Agent 名称与 suggestions 数组，交给前端渲染。
    return {
        "agent": result["agent"],
        "suggestions": result["suggestions"],
    }
