"""旧包名的兼容入口。

新代码应直接从 :mod:`hybrid_rag` 导入；这里仅用于避免旧调用立即失效。
"""

from hybrid_rag import *  # noqa: F401,F403
