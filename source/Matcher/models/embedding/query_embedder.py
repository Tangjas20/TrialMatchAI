from typing import Dict, List

import torch
from transformers import AutoModel, AutoTokenizer


class QueryEmbedder:
    def __init__(
        self,
        model_name: str = "ncbi/MedCPT-Query-Encoder",
        max_length: int = 512,
        use_gpu: bool = True,
    ):
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length
        self.model.eval()

    def get_embeddings(self, texts: List[str]) -> Dict[str, List[float]]:
        embeddings_dict = {}
        for text in texts:
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                embeddings = self.model(**encoded).last_hidden_state[:, 0, :]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings_dict[text] = embeddings.flatten().tolist()
        return embeddings_dict
