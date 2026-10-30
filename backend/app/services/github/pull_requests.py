
from app.services.github.client import GitHubClient


async def list_pull_requests(
    owner: str,                  # 仓库所属用户或组织
    repo: str,                   # 仓库名称
    token: str | None = None,    # GitHub Token，不传则使用默认配置
    limit: int = 30,             # 最多请求多少条 PR（单页）
    base_url: str | None = None  # GitHub API 基础地址
) -> list[dict]:
    """
    获取指定 GitHub 仓库的 PR 列表。

    返回的数据包括：
    - PR 编号 number
    - PR 标题 title
    - PR 状态 state
    - PR 创建者 user
    - 创建时间 created_at
    - 更新时间 updated_at
    """

    # 创建 GitHub API 客户端
    client = GitHubClient(token, base_url)

    # 发送 GET 请求
    # state=all：查询所有状态的 PR（open、closed）
    # per_page=limit：指定每页返回数量
    return await client.get(
        f"/repos/{owner}/{repo}/pulls",
        {"state": "all", "per_page": limit}
    )


async def list_pull_request_files(
    owner: str,
    repo: str,
    number: int,                 # PR 编号，例如 #12
    token: str | None = None,
    base_url: str | None = None
) -> list[dict]:
    """
    获取某个 PR 修改过的文件列表。

    例如 PR #12 修改了：
    - UserController.java
    - UserService.java
    - application.yml

    返回每个文件的修改信息，包括：
    filename：文件路径
    status：文件状态（added、modified、removed 等）
    additions：新增代码行数
    deletions：删除代码行数
    patch：代码差异片段（如果 API 提供）
    """

    client = GitHubClient(token, base_url)

    # GET /repos/{owner}/{repo}/pulls/{number}/files
    # per_page=100：每页最多返回 100 个文件
    return await client.get(
        f"/repos/{owner}/{repo}/pulls/{number}/files",
        {"per_page": 100}
    )


async def list_pull_request_review_comments(
    owner: str,
    repo: str,
    number: int,                 # PR 编号
    token: str | None = None,
    base_url: str | None = None
) -> list[dict]:
    """
    获取某个 PR 的代码审查评论。

    注意：这里获取的是 Review Comments，
    也就是针对代码行或代码差异的评论。

    不是 PR 页面中的普通讨论评论。

    返回数据包括：
    body：评论内容
    user：评论人
    path：评论对应的文件路径
    line：评论对应的代码行号（可能为空）
    created_at：评论创建时间
    """

    client = GitHubClient(token, base_url)

    # GET /repos/{owner}/{repo}/pulls/{number}/comments
    return await client.get(
        f"/repos/{owner}/{repo}/pulls/{number}/comments",
        {"per_page": 100}
    )
