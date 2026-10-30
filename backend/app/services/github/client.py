
from typing import Any

# httpx：Python 的 HTTP 请求库
# 类似 Java 中的 RestTemplate、WebClient
# 同时支持同步和异步 HTTP 请求
import httpx

# 导入项目的全局配置对象
# 例如 GitHub Token、GitHub API 基础地址
from app.core.config import settings


class GitHubClient:
    """
    GitHub API 客户端。

    作用：
    1. 统一管理 GitHub API 地址和认证 Token
    2. 封装 GET、POST、PATCH 等 HTTP 请求
    3. 提供仓库查询、Issue 创建、关闭、评论等业务方法

    类似 Java 项目中的 GitHubApiClient / GitHubFeignClient。
    """

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None
    ) -> None:
        """
        构造方法，创建 GitHubClient 对象时自动执行。

        token: GitHub 访问令牌，允许不传
        base_url: GitHub API 基础地址，允许不传

        Python 的 str | None 类似 Java 中可为空的 String。
        """

        # 优先使用调用方传入的 token
        # 如果 token 为 None 或空字符串，则使用全局配置
        # 类似 Java：token != null && !token.isEmpty() ? token : settings.githubToken
        self.token = token or settings.github_token

        # 优先使用传入的 API 地址，否则使用配置文件中的地址
        # rstrip("/")：删除字符串末尾的所有斜杠
        # 避免拼接路径时出现 //repos/owner/repo
        self.base_url = (
            base_url or settings.github_api_base_url
        ).rstrip("/")

    def _headers(self) -> dict[str, str]:
        """
        构造统一的 HTTP 请求头。

        _headers 前面的单下划线表示：
        这是类内部使用的方法（约定意义上的非公开方法）。

        返回值 dict[str, str]：
        类似 Java 的 Map<String, String>。
        """

        headers = {
            # 告诉 GitHub，希望接收 GitHub 官方 JSON 格式
            "Accept": "application/vnd.github+json",

            # 指定 GitHub REST API 的版本
            "X-GitHub-Api-Version": "2022-11-28"
        }

        # 如果配置了 Token，就添加身份认证请求头
        if self.token:
            # f-string：Python 字符串插值
            # 类似 Java："Bearer " + this.token
            headers["Authorization"] = f"Bearer {self.token}"

        return headers

    async def get(
        self,
        path: str,
        params: dict[str, Any] | None = None
    ) -> Any:
        """
        统一封装 HTTP GET 请求。

        path: 接口路径，例如 /repos/owner/repo
        params: URL 查询参数，例如 {"page": 1, "per_page": 20}

        async def：声明异步函数
        await：等待异步 HTTP 请求完成，不阻塞事件循环
        """

        # 创建异步 HTTP 客户端
        # timeout=30：请求超时时间为 30 秒
        # follow_redirects=True：自动跟随 HTTP 重定向
        #
        # async with 类似 Java 的 try-with-resources
        # 代码块结束后自动关闭客户端、释放连接资源
        async with httpx.AsyncClient(
            timeout=30,
            follow_redirects=True
        ) as client:

            # 发起 GET 请求
            # 拼接完整 URL，例如：
            # https://api.github.com/repos/owner/repo
            response = await client.get(
                f"{self.base_url}{path}",
                headers=self._headers(),
                params=params
            )

            # 检查 HTTP 响应状态
            # 4xx、5xx 时抛出 HTTPStatusError 异常
            # 类似 Java 中主动检查状态码后抛异常
            response.raise_for_status()

            # 将 JSON 响应体解析成 Python 对象
            # JSON 对象通常解析成 dict
            # JSON 数组通常解析成 list
            return response.json()

    async def get_bytes(
        self,
        path: str,
        params: dict[str, Any] | None = None
    ) -> bytes:
        """
        GET 请求的二进制版本。

        与 get() 的主要区别：
        get() 返回 JSON 解析后的 Python 对象。
        get_bytes() 返回原始 bytes 数据。

        适用于下载 ZIP、图片、文件等二进制内容。
        """

        # 下载文件时允许更长的超时时间：60 秒
        async with httpx.AsyncClient(
            timeout=60,
            follow_redirects=True
        ) as client:

            response = await client.get(
                f"{self.base_url}{path}",
                headers=self._headers(),
                params=params
            )

            # 检查 HTTP 请求是否成功
            response.raise_for_status()

            # 返回响应体的原始字节数据
            # 不进行 JSON 解析
            return response.content

    async def post(
        self,
        path: str,
        payload: dict[str, Any]
    ) -> Any:
        """
        统一封装 HTTP POST 请求。

        通常用于创建资源，例如创建 Issue、发表评论。

        payload：请求体数据
        类似 Java 中 @RequestBody 接收的 JSON 对象。
        """

        async with httpx.AsyncClient(
            timeout=30,
            follow_redirects=True
        ) as client:

            response = await client.post(
                f"{self.base_url}{path}",
                headers=self._headers(),

                # json=payload 会自动将 dict 序列化为 JSON
                # 并设置 Content-Type: application/json
                json=payload
            )

            response.raise_for_status()

            # 返回 GitHub 创建资源后的 JSON 响应
            return response.json()

    async def patch(
        self,
        path: str,
        payload: dict[str, Any]
    ) -> Any:
        """
        统一封装 HTTP PATCH 请求。

        PATCH 通常用于部分更新已有资源。

        例如：
        只修改 Issue 的 state 字段，
        而不需要重新提交整个 Issue 对象。
        """

        async with httpx.AsyncClient(
            timeout=30,
            follow_redirects=True
        ) as client:

            response = await client.patch(
                f"{self.base_url}{path}",
                headers=self._headers(),
                json=payload
            )

            response.raise_for_status()

            return response.json()

    async def get_repo(
        self,
        owner: str,
        repo: str
    ) -> dict[str, Any]:
        """
        获取指定 GitHub 仓库的信息。

        owner：仓库所属用户或组织
        repo：仓库名称

        GET /repos/{owner}/{repo}
        """

        # 复用上面封装的通用 GET 方法
        return await self.get(f"/repos/{owner}/{repo}")

    async def create_issue_comment(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        body: str
    ) -> dict[str, Any]:
        """
        给指定 Issue 添加评论。

        issue_number：Issue 在当前仓库中的编号
        body：评论内容

        POST /repos/{owner}/{repo}/issues/{issue_number}/comments
        """

        return await self.post(
            f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
            {"body": body}
        )

    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str
    ) -> dict[str, Any]:
        """
        在指定 GitHub 仓库中创建一个 Issue。

        title：Issue 标题
        body：Issue 正文内容

        POST /repos/{owner}/{repo}/issues
        """

        return await self.post(
            f"/repos/{owner}/{repo}/issues",
            {
                "title": title,
                "body": body
            }
        )

    async def close_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int
    ) -> dict[str, Any]:
        """
        关闭指定 Issue。

        PATCH /repos/{owner}/{repo}/issues/{issue_number}

        GitHub Issue 的 state：
        open   = 打开
        closed = 关闭
        """

        return await self.patch(
            f"/repos/{owner}/{repo}/issues/{issue_number}",
            {"state": "closed"}
        )

    async def add_issue_labels(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        labels: list[str]
    ) -> dict[str, Any]:
        """
        给指定 Issue 添加标签。

        labels：字符串列表
        例如 ["bug", "backend", "urgent"]

        POST /repos/{owner}/{repo}/issues/{issue_number}/labels
        """

        return await self.post(
            f"/repos/{owner}/{repo}/issues/{issue_number}/labels",
            {"labels": labels}
        )
