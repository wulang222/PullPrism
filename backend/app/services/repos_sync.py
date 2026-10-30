
from datetime import datetime, timezone
from uuid import UUID

import httpx
from fastapi import HTTPException
from sqlalchemy.orm import Session

# 项目全局配置
from app.core.config import settings

# 解密 GitHub Token
from app.core.security import decrypt_token

# SQLAlchemy 数据库实体
from app.db.models import (
    Issue,
    PRFile,
    PRReviewComment,
    PullRequest,
    Repository,
    WorkflowRun,
)

# 请求 DTO 和响应 DTO
# 类似 Java 中的 Request / Response 对象
from app.schemas.repos import RepoSyncRequest, RepoSyncResponse

# GitHub Actions API
from app.services.github.actions import (
    get_workflow_run_logs,
    list_workflow_run_jobs,
    list_workflow_runs,
)

# GitHub Issues API
from app.services.github.issues import list_issues

# GitHub Pull Requests API
from app.services.github.pull_requests import (
    list_pull_request_files,
    list_pull_request_review_comments,
    list_pull_requests,
)


def parse_github_datetime(
    value: str | None
) -> datetime | None:
    """
    将 GitHub 返回的 ISO 8601 时间字符串转换为 datetime。

    例如：
    "2026-10-28T08:00:00Z"

    转换成带有 UTC 时区信息的 datetime 对象。
    """

    # 如果没有时间值，直接返回 None
    if not value:
        return None

    # GitHub 的 Z 表示 UTC 时区
    # Python 使用 +00:00 表示相同的时区偏移
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


def login_of(user: dict | None) -> str | None:
    """
    提取 GitHub 用户的 login 字段。

    输入：
    {"id": 1001, "login": "zhangsan"}

    输出：
    "zhangsan"
    """

    # Python 三元表达式
    # 相当于 Java：
    # user != null ? user.get("login") : null
    return user.get("login") if user else None


async def sync_repository(
    db: Session,
    repo: Repository,
    payload: RepoSyncRequest
) -> RepoSyncResponse:
    """
    GitHub 仓库统一同步入口。

    db：SQLAlchemy 数据库 Session
    repo：本地仓库实体
    payload：同步请求参数

    根据 payload 中的开关决定同步哪些数据：
    - Issues
    - Pull Requests
    - Workflow Runs
    """

    # 只支持 GitHub 和 GitHub 兼容平台
    if repo.provider not in {"github", "github_compatible"}:
        raise HTTPException(
            status_code=400,
            detail=f"{repo.provider} sync is not implemented yet"
        )

    # 记录本次处理的数据数量
    synced = {
        "issues": 0,
        "pull_requests": 0,
        "workflow_runs": 0
    }

    # 如果数据库保存了加密 Token，就先解密
    # 否则使用 None
    token = (
        decrypt_token(
            repo.github_token_encrypted,
            settings.token_encryption_key
        )
        if repo.github_token_encrypted
        else None
    )

    # GitHub API 地址
    base_url = repo.api_base_url

    # ==========================================
    # 第一部分：同步 Issues
    # ==========================================

    # 用户是否要求同步 Issue
    if payload.sync_issues:

        try:
            # 调用 GitHub API 获取 Issue 列表
            remote_issues = await list_issues(
                repo.owner,
                repo.name,
                token=token,
                limit=payload.limit,
                base_url=base_url
            )

        except httpx.HTTPStatusError as exc:
            # GitHub API 返回 4xx 或 5xx 时抛异常
            # 转换成 FastAPI 的 HTTPException
            #
            # raise ... from exc：
            # 保留原始异常作为异常链
            raise HTTPException(
                status_code=400,
                detail=(
                    "GitHub issues sync failed: "
                    f"{exc.response.status_code} "
                    f"{exc.response.text}"
                )
            ) from exc

        # 遍历 GitHub 返回的每条 Issue
        for item in remote_issues:

            # 通过本地仓库 ID + GitHub Issue ID 查询
            #
            # one_or_none()：
            # 找到一条 -> 返回 ORM 对象
            # 没找到 -> 返回 None
            # 找到多条 -> 抛出异常
            issue = db.query(Issue).filter(
                Issue.repo_id == repo.id,
                Issue.github_issue_id == item.get("id")
            ).one_or_none()

            # 本地不存在，创建新记录
            if issue is None:
                issue = Issue(
                    repo_id=repo.id,
                    github_issue_id=item.get("id"),
                    number=item.get("number", 0),
                    title=item.get("title") or ""
                )

                # 加入 Session，等待持久化
                db.add(issue)

            # 无论新旧 Issue，统一更新字段
            issue.number = item.get("number", issue.number)
            issue.title = item.get("title") or issue.title
            issue.body = item.get("body")
            issue.state = item.get("state") or "open"

            # 提取所有标签名称
            # [{"name":"bug"}] -> ["bug"]
            issue.labels = [
                label.get("name")
                for label in item.get("labels", [])
                if label.get("name")
            ]

            # 提取 Issue 创建者用户名
            issue.author = login_of(item.get("user"))

            # 提取被指派的用户列表
            issue.assignees = [
                login_of(user)
                for user in item.get("assignees", [])
                if login_of(user)
            ]

            # GitHub 时间字符串转 datetime
            issue.created_at = parse_github_datetime(
                item.get("created_at")
            )
            issue.updated_at = parse_github_datetime(
                item.get("updated_at")
            )
            issue.closed_at = parse_github_datetime(
                item.get("closed_at")
            )

            # 本次处理的 Issue 数量 +1
            # 注意：包含新增和更新
            synced["issues"] += 1

    # ==========================================
    # 第二部分：同步 Pull Requests
    # ==========================================

    if payload.sync_pull_requests:

        try:
            # 获取远程 PR 列表
            remote_prs = await list_pull_requests(
                repo.owner,
                repo.name,
                token=token,
                limit=payload.limit,
                base_url=base_url
            )

        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "GitHub pull requests sync failed: "
                    f"{exc.response.status_code} "
                    f"{exc.response.text}"
                )
            ) from exc

        for item in remote_prs:

            # 根据 GitHub PR ID 查找本地 PR
            pr = db.query(PullRequest).filter(
                PullRequest.repo_id == repo.id,
                PullRequest.github_pr_id == item.get("id")
            ).one_or_none()

            # 如果不存在，创建新的 PR 对象
            if pr is None:
                pr = PullRequest(
                    repo_id=repo.id,
                    github_pr_id=item.get("id"),
                    number=item.get("number", 0),
                    title=item.get("title") or ""
                )

                db.add(pr)

                # 将待执行的 SQL 发送到数据库
                # 使新 PR 的数据库主键等值可用
                #
                # flush 不等于 commit
                # 当前事务仍然可以回滚
                db.flush()

            # 更新 PR 基础信息
            pr.number = item.get("number", pr.number)
            pr.title = item.get("title") or pr.title
            pr.body = item.get("body")
            pr.state = item.get("state") or "open"
            pr.author = login_of(item.get("user"))

            # base：目标分支
            # head：来源分支
            #
            # (item.get("base") or {})：
            # 防止 base 为 None 时调用 .get() 报错
            pr.base_branch = (
                item.get("base") or {}
            ).get("ref")

            pr.head_branch = (
                item.get("head") or {}
            ).get("ref")

            # 更新时间信息
            pr.merged_at = parse_github_datetime(
                item.get("merged_at")
            )
            pr.created_at = parse_github_datetime(
                item.get("created_at")
            )
            pr.updated_at = parse_github_datetime(
                item.get("updated_at")
            )

            # ----------------------------------
            # 同步 PR 修改文件
            # ----------------------------------

            # 删除本地该 PR 之前保存的文件记录
            # 后面重新从 GitHub 获取并插入
            #
            # 这是一种删除后重建的同步策略
            db.query(PRFile).filter(
                PRFile.pr_id == pr.id
            ).delete()

            try:
                # 获取当前 PR 修改过的文件
                files = await list_pull_request_files(
                    repo.owner,
                    repo.name,
                    pr.number,
                    token=token,
                    base_url=base_url
                )

            except httpx.HTTPStatusError:
                # 获取失败则使用空列表
                files = []

            # 遍历 PR 修改文件
            for file_item in files:

                # 每个文件创建一条 PRFile 记录
                db.add(
                    PRFile(
                        # 关联本地 PR 主键
                        pr_id=pr.id,

                        # 修改文件的路径
                        filename=file_item.get("filename") or "",

                        # added / modified / removed 等
                        status=file_item.get("status") or "modified",

                        # 新增代码行数
                        additions=file_item.get("additions") or 0,

                        # 删除代码行数
                        deletions=file_item.get("deletions") or 0,

                        # Git Diff 代码差异
                        patch=file_item.get("patch"),
                    )
                )

            # ----------------------------------
            # 同步 PR 代码审查评论
            # ----------------------------------

            # 删除该 PR 原有的审查评论
            db.query(PRReviewComment).filter(
                PRReviewComment.pr_id == pr.id
            ).delete()

            try:
                # 获取 GitHub Review Comments
                review_comments = (
                    await list_pull_request_review_comments(
                        repo.owner,
                        repo.name,
                        pr.number,
                        token=token,
                        base_url=base_url
                    )
                )

            except httpx.HTTPStatusError:
                review_comments = []

            # 遍历所有代码审查评论
            for comment_item in review_comments:

                db.add(
                    PRReviewComment(
                        # 关联本地 PR
                        pr_id=pr.id,

                        # GitHub 评论的唯一 ID
                        github_comment_id=comment_item.get("id"),

                        # 评论内容
                        body=comment_item.get("body"),

                        # 评论对应的文件路径
                        path=comment_item.get("path"),

                        # 优先使用当前行号
                        # 如果为空则使用原始行号
                        line=(
                            comment_item.get("line")
                            or comment_item.get("original_line")
                        ),

                        # 评论作者
                        author=login_of(comment_item.get("user")),

                        created_at=parse_github_datetime(
                            comment_item.get("created_at")
                        ),
                        updated_at=parse_github_datetime(
                            comment_item.get("updated_at")
                        ),
                    )
                )

            # PR 处理数量 +1
            synced["pull_requests"] += 1

    # ==========================================
    # 第三部分：同步 GitHub Actions
    # ==========================================

    if payload.sync_workflow_runs:

        try:
            # 获取工作流执行记录
            remote_runs = await list_workflow_runs(
                repo.owner,
                repo.name,
                token=token,
                limit=payload.limit,
                base_url=base_url
            )

        except httpx.HTTPStatusError:
            # 获取失败则跳过本次 Workflow 同步
            remote_runs = []

        for item in remote_runs:

            # 根据 GitHub Workflow Run ID 查找本地记录
            run = db.query(WorkflowRun).filter(
                WorkflowRun.repo_id == repo.id,
                WorkflowRun.github_run_id == item.get("id")
            ).one_or_none()

            # 本地不存在则创建
            if run is None:
                run = WorkflowRun(
                    repo_id=repo.id,
                    github_run_id=item.get("id"),
                    name=item.get("name") or "workflow"
                )
                db.add(run)

            # 更新工作流运行信息
            run.name = item.get("name") or run.name
            run.status = item.get("status") or "unknown"
            run.conclusion = item.get("conclusion")
            run.html_url = item.get("html_url")

            run.created_at = parse_github_datetime(
                item.get("created_at")
            )
            run.updated_at = parse_github_datetime(
                item.get("updated_at")
            )

            # ----------------------------------
            # 同步 Workflow Run 的 Jobs
            # ----------------------------------

            if run.github_run_id:

                try:
                    # 获取本次工作流执行的所有 Job
                    jobs = await list_workflow_run_jobs(
                        repo.owner,
                        repo.name,
                        run.github_run_id,
                        token=token,
                        base_url=base_url
                    )

                    # 将 Job 数据转换为精简字典列表
                    # 只保存项目需要的字段
                    run.jobs = [
                        {
                            "id": job.get("id"),
                            "name": job.get("name"),
                            "status": job.get("status"),
                            "conclusion": job.get("conclusion"),
                            "html_url": job.get("html_url"),
                            "started_at": job.get("started_at"),
                            "completed_at": job.get("completed_at"),

                            # 每个 Job 里面又有多个 Step
                            # 使用嵌套列表推导式
                            "steps": [
                                {
                                    "name": step.get("name"),
                                    "status": step.get("status"),
                                    "conclusion": step.get("conclusion"),
                                    "number": step.get("number"),
                                }
                                for step in job.get("steps", [])
                            ],
                        }
                        for job in jobs
                    ]

                except httpx.HTTPStatusError:
                    # 获取 Job 失败时保留已有数据
                    # 如果没有旧数据则使用空列表
                    run.jobs = run.jobs or []

            # ----------------------------------
            # 只同步失败工作流的日志
            # ----------------------------------

            # conclusion == failure 表示运行结果失败
            if (
                run.conclusion == "failure"
                and run.github_run_id
            ):
                try:
                    # 下载 ZIP、解压并提取文本日志
                    run.logs_text = await get_workflow_run_logs(
                        repo.owner,
                        repo.name,
                        run.github_run_id,
                        token=token,
                        base_url=base_url
                    )

                except httpx.HTTPStatusError:
                    # 获取失败时保持原有日志不变
                    run.logs_text = run.logs_text

            synced["workflow_runs"] += 1

    # ==========================================
    # 第四部分：保存同步结果
    # ==========================================

    # 更新仓库最近一次同步时间
    # 使用带 UTC 时区的当前时间
    repo.last_sync_at = datetime.now(timezone.utc)

    # 将 Session 中待执行的 SQL 发送到数据库
    db.flush()

    # 提交事务，持久化本次同步的数据
    db.commit()

    # 返回同步结果 DTO
    return RepoSyncResponse(
        repo_id=repo.id,
        status="completed",
        synced=synced
    )


async def sync_repository_by_id(
    repo_id: UUID,
    payload: RepoSyncRequest
) -> RepoSyncResponse | None:
    """
    根据仓库 UUID 执行同步。

    与 sync_repository() 的区别：

    sync_repository()：
        调用方需要提供 db 和 repo 对象。

    sync_repository_by_id()：
        自己创建数据库 Session，
        根据 repo_id 查询仓库，再执行同步。
    """

    # 在函数内部导入数据库 Session 工厂
    from app.db.session import SessionLocal

    # 创建数据库 Session
    # 类似 Java 中获取一个数据库会话
    db = SessionLocal()

    try:
        # 根据 Repository 的本地主键查询
        # db.get() 是 SQLAlchemy 的主键查询方法
        repo = db.get(Repository, repo_id)

        # 仓库不存在，返回 None
        if repo is None:
            return None

        # 调用真正的同步逻辑
        return await sync_repository(db, repo, payload)

    finally:
        # 无论成功、失败还是提前 return
        # 都关闭数据库 Session
        db.close()
