
# Python 异步编程标准库
# 用于 sleep、创建异步任务等
import asyncio

# Python 标准日志模块
# 类似 Java 中的 SLF4J + Logger
import logging

# 全局配置，例如：
# auto_sync_enabled
# auto_sync_limit
# auto_sync_interval_seconds
from app.core.config import settings

# 仓库 ORM 实体
from app.db.models import Repository

# SQLAlchemy 数据库 Session 工厂
# 类似于创建一个数据库操作会话
from app.db.session import SessionLocal

# 仓库同步请求 DTO
from app.schemas.repos import RepoSyncRequest

# 根据仓库 ID 执行同步的业务函数
from app.services.repos_sync import sync_repository_by_id


# 创建当前模块的 Logger
# __name__ 是当前 Python 模块名称
logger = logging.getLogger(__name__)


async def periodic_repo_sync_loop() -> None:
    """
    后台定时同步 GitHub 仓库。

    执行流程：
    1. 启动后先等待 5 秒
    2. 判断是否开启自动同步
    3. 查询所有 GitHub 仓库
    4. 遍历仓库，逐个同步
    5. 捕获单个仓库的同步异常
    6. 等待指定间隔
    7. 重复执行
    """

    # 异步等待 5 秒
    # 不会像 time.sleep() 一样阻塞整个事件循环
    #
    # 这里的 5 秒是启动后的初始等待时间
    await asyncio.sleep(5)

    # 当自动同步开关为 True 时，不断循环
    #
    # 如果启动时为 False，则不会进入循环
    while settings.auto_sync_enabled:

        # 创建数据库 Session
        db = SessionLocal()

        try:
            # 查询 provider 为 github 或 github_compatible 的仓库
            #
            # SQL 大致相当于：
            # SELECT *
            # FROM repositories
            # WHERE provider IN ('github', 'github_compatible');
            repos = (
                db.query(Repository)
                .filter(
                    Repository.provider.in_(
                        ["github", "github_compatible"]
                    )
                )
                .all()
            )

            # 列表推导式
            #
            # 将 Repository 对象列表转换为 UUID 列表
            #
            # 例如：
            # repos = [repo1, repo2, repo3]
            # repo_ids = [uuid1, uuid2, uuid3]
            repo_ids = [repo.id for repo in repos]

        finally:
            # 不管查询成功还是发生异常
            # 都关闭数据库 Session
            db.close()

        # 遍历所有仓库 ID
        for repo_id in repo_ids:

            try:
                # 调用之前实现的仓库同步服务
                #
                # await 表示等待当前仓库同步完成
                #
                # RepoSyncRequest：
                # 构造同步参数，limit 使用配置值
                await sync_repository_by_id(
                    repo_id,
                    RepoSyncRequest(
                        limit=settings.auto_sync_limit
                    )
                )

            except Exception as exc:
                # 单个仓库同步失败，不中断整个同步循环
                #
                # 记录 warning 日志，继续同步下一个仓库
                #
                # %s 是 logging 的参数占位符
                logger.warning(
                    "Periodic repo sync failed for %s: %s",
                    repo_id,
                    exc
                )

        # 本轮所有仓库同步结束后，等待一段时间
        #
        # max(a, 30)：
        # 如果配置间隔小于 30 秒，则至少等待 30 秒
        #
        # 例如：
        # auto_sync_interval_seconds = 10 -> 等待 30 秒
        # auto_sync_interval_seconds = 300 -> 等待 300 秒
        await asyncio.sleep(
            max(settings.auto_sync_interval_seconds, 30)
        )
