# Hybrid RAG

一个与业务领域无关的本地混合检索组件，提供：

- Markdown 标题感知分片
- 超长章节二次分片与 overlap
- 稳定的 document/chunk ID
- 中文 BM25
- FAISS 向量检索
- Reciprocal Rank Fusion（RRF）
- 元数据过滤
- 索引版本、manifest 与过期检测
- `hybrid`、`vector`、`bm25` 三种检索模式

核心包只负责知识加载、索引和检索，不调用 LLM，也不包含任何业务 Prompt。

数据库或其他上游提供的 `title` 会作为虚拟标题参与向量检索和 BM25；`tags`
同样会加入检索文本。检索结果仍保留原始 chunk 正文，不会把虚拟标题或标签
重复写入传给 LLM 的 `RetrievalResult.content`。没有 `#`、`##` 等标题的
Markdown 会先作为一个章节处理，超长时再按段落、换行、中文标点和长度切分。
升级到包含虚拟标题和标签的向量文本后，需要执行一次 `--rebuild`；旧索引仍可
读取，但只有重建后的向量才包含这些新增检索信息。

## 安装

```bash
uv sync --group dev
```

首次使用默认模型 `BAAI/bge-small-zh-v1.5` 时，Sentence Transformers
可能需要下载模型。生产环境应提前缓存模型，避免服务启动时临时下载。

## 快速开始

```python
from hybrid_rag import HybridRetrievalService, KnowledgeDocument, RetrievalConfig

documents = [
    KnowledgeDocument(
        id="doc-1",
        title="连接频繁中断的处理方法",
        content="# 网络问题\n\n## 频繁中断\n\n请先切换加速节点并重新连接。",
        path="network/disconnect.md",
        metadata={
            "category": "support",
            "tags": ["掉线", "网络波动"],
            "weight": 1,
        },
    )
]

service = HybridRetrievalService(RetrievalConfig(index_path="./vector_index"))
service.build(documents)
results = service.search(
    "开了加速器还是经常断开",
    top_k=5,
    filters={"category": "support"},
)

for result in results:
    print(result.title, result.score, result.content)
```

数据库接入时，只需把数据库记录转换成 `KnowledgeDocument`。检索包不会要求
数据库模型或 Web 框架遵循特定实现。

## Markdown 目录示例

使用现有 `data/` 目录构建或加载索引：

```bash
uv run python main.py --data ./data --rebuild
uv run python main.py --data ./data --query "示例问题" --top-k 5
```

命令行只是演示入口。Web 服务应长期持有一个已加载的
`HybridRetrievalService`，不要为每个请求重新加载模型和索引。

## 数据契约

### KnowledgeDocument

- `id`：由上游系统提供的稳定 ID
- `title`：文档标题
- `content`：Markdown 或普通文本正文
- `path`：可选来源路径
- `metadata`：任意可序列化元数据，例如 `category`、`tags`、`weight`

### RetrievalResult

结果包含：

- `document_id`、`chunk_id`
- 标题、正文、Markdown 标题路径
- 最终 RRF 分数
- Vector/BM25 的原始排名和分数
- 原始元数据

这些字段可以直接用于后台的检索调试与 TF-IDF 对比页面。

## 索引生命周期

每次构建都会写入一个独立版本：

```text
vector_index/
├── CURRENT
└── versions/
    └── 20260718T120000Z-ab12cd34/
        ├── manifest.json
        ├── chunks.json
        └── faiss/
```

只有构建全部成功后才会原子更新 `CURRENT`。`is_stale(documents)` 会比较：

- 文档 ID、标题、正文、路径和元数据
- Embedding 模型
- Chunk 大小与 overlap

查询阶段的 Top-K 或 RRF 参数变化不要求重新生成向量。

FAISS 的 LangChain 持久化格式包含 pickle sidecar。代码只允许加载本组件自己生成
且位于受信任索引目录中的文件，不能加载用户上传的索引。

## 测试

测试使用确定性的本地假 Embedding，不下载模型、不调用网络：

```bash
uv run pytest
```

## 接入客服系统

建议公司项目保留三条检索路径：

- `tfidf`：原实现，用于基线和回滚
- `hybrid`：本组件的 FAISS + BM25 + RRF
- `compare`：两者同时检索，但先由 TF-IDF 结果生成线上回答

本仓库只提供 Hybrid 检索能力。模式路由、SSE 输出、聊天记录、Prompt 和 TF-IDF
仍由客服系统负责。
