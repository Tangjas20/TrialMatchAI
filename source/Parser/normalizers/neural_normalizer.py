import argparse
import os
import pickle
import re
from string import punctuation

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, default_data_collator


class NamesDataset(Dataset):
    def __init__(self, encodings, device):
        self.encodings = encodings
        self.device = device

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]).to(self.device)
            for key, val in self.encodings.items()
        }

    def __len__(self):
        return len(self.encodings["input_ids"])


class NeuralNormalizer:
    def __init__(self, model_name_or_path, cache_path=None, no_cuda=False):
        self.max_length = 25
        self.batch_size = 1024
        self.k = 1  # top k

        # device setup
        use_cuda = torch.cuda.is_available() and not no_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Ensure proper initialization of the model
        self.model = AutoModel.from_pretrained(model_name_or_path, device_map=None)
        self.model.to_empty(device=self.device)  # Use to_empty to handle meta tensors
        self.model.eval()

        # regex for basic normalization
        self.rmv_puncts_regex = re.compile(r"[\s{}]+".format(re.escape(punctuation)))

        # optionally load existing cache
        if cache_path:
            self.load_cache(cache_path)

    def load_dictionary(self, dictionary_path=""):
        self.dictionary = []
        with open(dictionary_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cui, names = line.split("||")
                for name in names.split("|"):
                    normalized = self._basic_normalize(name)
                    tokens = self.tokenizer.tokenize(normalized)
                    if len(tokens) <= self.max_length:
                        self.dictionary.append((cui, name))

        # embed dictionary entries
        self.dict_embeds = self._embed_dictionary()

    def normalize(self, names):
        if not names:
            return []

        names_norm = [self._basic_normalize(n) for n in names]

        # encode inputs
        encodings = self.tokenizer(
            names_norm,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors=None,
        )
        dataset = NamesDataset(encodings, self.device)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )

        # compute embeddings
        embeds = []
        with torch.no_grad():
            for batch in loader:
                out = self.model(**batch)
                cls_emb = out.last_hidden_state[:, 0].cpu().numpy()
                embeds.append(cls_emb)
        name_embeds = np.vstack(embeds)

        # search FAISS index
        distances, indices = self.dict_embeds.search(name_embeds, self.k)
        top_indices = indices[:, 0]

        # gather outputs
        results = []
        for idx in top_indices:
            if idx >= 0:
                results.append(self.dictionary[idx])
            else:
                results.append((None, None))
        return results

    def _basic_normalize(self, text):
        text = text.lower()
        text = re.sub(self.rmv_puncts_regex, " ", text)
        return " ".join(text.split())

    def _embed_dictionary(self, show_progress=True):
        encs = self.tokenizer(
            [self._basic_normalize(name) for _, name in self.dictionary],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors=None,
        )
        dataset = NamesDataset(encs, self.device)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )

        embeds = []
        with torch.no_grad():
            for batch in tqdm(loader, disable=not show_progress):
                out = self.model(**batch)
                cls_emb = out.last_hidden_state[:, 0].cpu().numpy()
                embeds.append(cls_emb)
        all_embeds = np.vstack(embeds)
        return all_embeds

    def save_cache(self, cache_path):
        # build FAISS index
        dim = self.dict_embeds.shape[1]
        quantiser = faiss.IndexFlatIP(dim)
        nlist = 2048
        index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(self.dict_embeds)
        index.add(self.dict_embeds)
        index.nprobe = 25

        faiss.write_index(index, cache_path + ".index")
        with open(cache_path, "wb") as f:
            pickle.dump(self.dictionary, f)

    def load_cache(self, cache_path):
        self.dict_embeds = faiss.read_index(cache_path + ".index")
        with open(cache_path, "rb") as f:
            self.dictionary = pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["indexing", "predict"], default="indexing")
    parser.add_argument(
        "--model_name_or_path", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )
    parser.add_argument(
        "--dictionary_path",
        default="../normalization/resources/dictionary/best_dict_Disease_20210630_tmp.txt",
    )
    parser.add_argument("--cache_dir", default="../normalization/biosyn_cache")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    normalizer = NeuralNormalizer(
        model_name_or_path=args.model_name_or_path,
        cache_path=None,
        no_cuda=args.no_cuda,
    )

    if args.mode == "indexing":
        normalizer.load_dictionary(dictionary_path=args.dictionary_path)
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_path = os.path.join(
            args.cache_dir, os.path.basename(args.dictionary_path) + ".pk"
        )
        normalizer.save_cache(cache_path)
        print("Indexing complete. Cache saved to", cache_path)

    elif args.mode == "predict":
        # load cache
        normalizer.load_cache(
            os.path.join(args.cache_dir, os.path.basename(args.dictionary_path) + ".pk")
        )
        # example prediction
        samples = ["diabetes", "heart disease"]
        results = normalizer.normalize(samples)
        for inp, out in zip(samples, results):
            print(f"{inp} => {out}")


if __name__ == "__main__":
    main()
