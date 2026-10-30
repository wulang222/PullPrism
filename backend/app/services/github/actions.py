
# BytesIO：把内存中的 bytes 包装成类似文件的对象
# 类似 Java 的 ByteArrayInputStream
from io import BytesIO

# ZipFile：读取和解压 ZIP 压缩包
# BadZipFile：ZIP 文件格式错误时抛出的异常
from zipfile import BadZipFile, ZipFile

# 项目中封装的 GitHub API 客户端
from app.services.github.client import GitHubClient


async def list_workflow_runs(
    owner: str,                  # GitHub 用户名或组织名
    repo: str,                   # 仓库名称
    token: str | None = None,    # GitHub Token
    limit: int = 30,             # 每页请求的执行记录数量
    base_url: str | None = None  # GitHub API 基础地址
) -> list[dict]:
    """
    获取仓库的 GitHub Actions 工作流执行记录。

    每次触发工作流，GitHub 都会生成一条 Workflow Run。

    返回的信息通常包括：
    - id：本次执行的唯一 ID
    - name：工作流名称
    - status：运行状态
    - conclusion：执行结果
    - created_at：创建时间
    """

    # 创建 GitHub API 客户端
    client = GitHubClient(token, base_url)

    # 调用 GitHub REST API
    # GET /repos/{owner}/{repo}/actions/runs
    #
    # per_page：指定每页返回多少条数据
    payload = await client.get(
        f"/repos/{owner}/{repo}/actions/runs",
        {"per_page": limit}
    )

    # GitHub 返回的是一个 dict，例如：
    # {
    #     "total_count": 10,
    #     "workflow_runs": [...]
    # }
    #
    # 只提取 workflow_runs 列表
    # 如果没有该字段，默认返回空列表 []
    return payload.get("workflow_runs", [])


async def list_workflow_run_jobs(
    owner: str,
    repo: str,
    run_id: int,                 # 某次 Workflow Run 的唯一 ID
    token: str | None = None,
    base_url: str | None = None
) -> list[dict]:
    """
    获取某次 Workflow Run 中的所有 Job。

    例如一次运行可能包含：
    - build：编译
    - test：测试
    - deploy：部署

    每个 Job 还可以包含多个 Step。
    """

    client = GitHubClient(token, base_url)

    # GET /repos/{owner}/{repo}/actions/runs/{run_id}/jobs
    #
    # per_page=100：每页最多请求 100 个 Job
    payload = await client.get(
        f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs",
        {"per_page": 100}
    )

    # GitHub 返回：
    # {
    #     "total_count": 3,
    #     "jobs": [...]
    # }
    #
    # 只返回 jobs 数组
    return payload.get("jobs", [])


async def get_workflow_run_logs(
    owner: str,
    repo: str,
    run_id: int,
    token: str | None = None,
    base_url: str | None = None,
    max_chars: int = 80000       # 最多返回 80000 个字符
) -> str:
    """
    获取某次 Workflow Run 的完整日志。

    执行流程：
    1. 调用 GitHub API 下载日志
    2. GitHub 通常返回 ZIP 压缩包
    3. 在内存中打开 ZIP
    4. 遍历 ZIP 中的日志文件
    5. 将 bytes 解码成字符串
    6. 合并日志并截断长度
    """

    client = GitHubClient(token, base_url)

    # 下载 Workflow Run 日志
    #
    # GET /repos/{owner}/{repo}/actions/runs/{run_id}/logs
    #
    # get_bytes() 返回 bytes，不是 JSON
    # GitHub 会通过重定向提供日志 ZIP 文件
    payload = await client.get_bytes(
        f"/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
    )

    try:
        # BytesIO(payload)：
        # 将内存中的 bytes 包装成文件对象
        #
        # ZipFile(...)：
        # 将该文件对象作为 ZIP 压缩包打开
        #
        # with：
        # 代码执行完毕后自动关闭 ZIP 对象
        with ZipFile(BytesIO(payload)) as archive:

            # 创建字符串列表，用来存储每个日志文件的内容
            parts: list[str] = []

            # archive.namelist()：获取 ZIP 内所有文件名称
            # sorted()：按照文件名排序
            for name in sorted(archive.namelist()):

                # ZIP 中的目录名称通常以 / 结尾
                # 如果是目录，就跳过本次循环
                if name.endswith("/"):
                    continue

                # 读取 ZIP 中指定文件的二进制内容
                # 返回 bytes 类型
                raw = archive.read(name)

                # 将 bytes 转换成 Python 字符串
                # utf-8：使用 UTF-8 编码
                # errors="replace"：
                # 遇到无法解码的字节时，用替代字符代替
                # 避免因为编码错误直接抛异常
                text = raw.decode("utf-8", errors="replace")

                # 给每个日志文件添加标题，方便区分来源
                #
                # 例如：
                # ### build/1_Set up job.txt
                # 2026-10-28 ... Starting job
                parts.append(f"### {name}\n{text}")

                # 计算当前已经收集的日志总字符数
                #
                # len(part)：单个字符串长度
                # sum(...)：将所有长度相加
                #
                # 如果达到限制，就停止读取后续文件
                if sum(len(part) for part in parts) >= max_chars:
                    break

            # 使用两个换行符连接所有日志片段
            # [:max_chars]：最多保留前 max_chars 个字符
            return "\n\n".join(parts)[:max_chars]

    # 如果下载的数据不是合法 ZIP
    # 就尝试直接当作普通文本进行解码
    except BadZipFile:
        return payload.decode(
            "utf-8",
            errors="replace"
        )[:max_chars]
