import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Reranker:
    """
    bge-reranker-base 实现 re-ranking
    输入：query(str) + candidate_texts(list[str])
    输出：对应相似度得分（越高越相关）
    """

    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def rerank(self, query, texts):
        """
        输入:
            query: str
            texts: List[str]
        输出:
            List[float] 对应每个文本的相关性分数
        """

        pairs = [[query, t] for t in texts]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)

        return scores.cpu().tolist()
