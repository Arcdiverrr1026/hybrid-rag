"""Minimal in-memory example resembling a customer-support knowledge base."""

from hybrid_rag import HybridRetrievalService, KnowledgeDocument, RetrievalConfig


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

service = HybridRetrievalService(RetrievalConfig(index_path="./demo_index"))
service.build(documents)

for result in service.search("游戏总是断开怎么办"):
    print(result.title, result.score)
    print(result.content)

