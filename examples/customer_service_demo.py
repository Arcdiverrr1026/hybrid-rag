"""最小示例：使用内存中的两篇客服知识文档完成建索引和检索。"""

from hybrid_rag import HybridRetrievalService, KnowledgeDocument, RetrievalConfig


# 实际接入数据库时，只需把数据库记录转换为相同的 KnowledgeDocument 对象。
documents = [
    KnowledgeDocument(
        id="network-disconnect",
        title="加速后仍然掉线怎么办",
        content=(
            "# 网络故障\n\n"
            "## 连接频繁中断\n\n"
            "请先更换加速节点，再完全退出游戏和加速器后重新连接。"
        ),
        metadata={"category": "support", "tags": ["掉线", "节点"]},
    ),
    KnowledgeDocument(
        id="client-download",
        title="客户端下载方法",
        content="# 客户端\n\n## 下载\n\n请从官方客户端下载页面选择对应平台。",
        metadata={"category": "support", "tags": ["下载", "安装"]},
    ),
]

# build 会同时建立 BM25 索引和 FAISS 向量索引。
service = HybridRetrievalService(RetrievalConfig(index_path="./demo_index"))
service.build(documents)

# search 默认使用 hybrid 模式，即融合 BM25 和向量检索结果。
for result in service.search("游戏总是断开怎么办"):
    print(result.title, result.score)
    print(result.content)
