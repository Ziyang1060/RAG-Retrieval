import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
import json
from collections import defaultdict
from utils import map_label_to_continuous, visualize_label_distribution, shuffle_text
import math
import random

class RankerDataset(Dataset):
    def __init__(self, train_data_path, target_model, max_len=512, max_label=1, min_label=0, shuffle_rate=0.0, tag="train"):
        self.model = target_model
        self.max_len = max_len
        assert max_label > min_label and min_label >= 0
        self.max_label = max_label
        self.min_label = min_label
        self.map_func = lambda x: map_label_to_continuous(x, self.min_label, self.max_label)
        assert 0 <= shuffle_rate <= 1 , "shuffle rate must be between 0 and 1"
        self.shuffle_rate = shuffle_rate # The probability of shuffling the text
        self.tag = tag
        self.train_data = self.read_train_data(train_data_path)

    def read_train_data(self, train_data_path):
        # standard input data type:
        # {"query": str(required), "pos": List[str](required), "neg":List[str](required), "pos_scores": List(optional), "neg_scores": List(optional)}}   
        
        train_data = []
        label_distribution = defaultdict(int)
        with open(train_data_path) as f:
            for line in tqdm.tqdm(f):
                data_dic = json.loads(line.strip())
                query = data_dic["query"].strip()
                assert "query" in data_dic and "pos" in data_dic
                if "pos_scores" in data_dic:
                    assert len(data_dic["pos"]) == len(data_dic["pos_scores"])
                else:
                    data_dic["pos_scores"] = [1] * len(data_dic["pos"])
                if "neg_scores" in data_dic:
                    assert "neg" in data_dic and len(data_dic["neg"]) == len(
                        data_dic["neg_scores"]
                    )
                else:
                    data_dic["neg_scores"] = [0] * len(data_dic["neg"])

                for idx, text_pos in enumerate(data_dic["pos"]):
                    text_pos = text_pos.strip()
                    if self.shuffle_rate > 0:
                        text_pos = shuffle_text(text_pos, self.shuffle_rate)
                    pos_score = self.map_func(data_dic["pos_scores"][idx])
                    label_distribution[f"{pos_score:.2f}"] += 1
                    train_data.append([query, text_pos, pos_score])
                    
                for idx, text_neg in enumerate(data_dic["neg"]):
                    text_neg = text_neg.strip()
                    if self.shuffle_rate > 0:
                        text_neg = shuffle_text(text_neg, self.shuffle_rate)
                    neg_score = self.map_func(data_dic["neg_scores"][idx])
                    label_distribution[f"{neg_score:.2f}"] += 1
                    train_data.append([query, text_neg, neg_score])

        # Only visualize the label distribution on the main process of distributed mode or in the single process mode
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"----- {self.tag} data -----")
            visualize_label_distribution(label_distribution)
            
        # standard output data type: [query, doc, score[0,1]]
        return train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]

    def collate_fn(self, batch):
        all_batch_pairs = []
        all_labels = []

        for item in batch:
            all_batch_pairs.append([item[0], item[1]])
            all_labels.append(item[2])

        # 模型的 preprocess 方法实际将 (query, doc) pair 转换为模型的输入 input_ids 形式
        tokens = self.model.preprocess(all_batch_pairs, self.max_len) # max_len 实际的作用由模型本身界定
        label_batch = torch.tensor(all_labels, dtype=torch.float16)

        return tokens, label_batch

class GroupedRankerDataset(Dataset):
    def __init__(self, train_data_path, target_model, max_len=512, max_label=1, min_label=0, shuffle_rate=0.0, train_group_size=8, tag="train"):
        self.model = target_model
        self.max_len = max_len
        assert max_label > min_label and min_label >= 0
        self.max_label = max_label
        self.min_label = min_label
        self.map_func = lambda x: map_label_to_continuous(x, self.min_label, self.max_label)
        assert 0 <= shuffle_rate <= 1 , "shuffle rate must be between 0 and 1"
        self.shuffle_rate = shuffle_rate # The probability of shuffling the text
        self.tag = tag
        assert train_group_size >= 2
        self.train_group_size = train_group_size
        self.train_data = self.read_train_data(train_data_path)
        print(f"Using train_group_size: {self.train_group_size}")

    def read_train_data(self, train_data_path):
        # standard input data type:
        # {"query": str, "pos": List[str], "neg":List[str], "pos_scores": List(optional), "neg_scores": List(optional)}}   
        
        train_data = []
        label_distribution = defaultdict(int)
        with open(train_data_path) as f:
            for line in tqdm.tqdm(f):
                data_dic = json.loads(line.strip())
                query = data_dic["query"].strip()
                assert "query" in data_dic and "pos" in data_dic
                if "pos_scores" in data_dic:
                    assert len(data_dic["pos"]) == len(data_dic["pos_scores"])
                else:
                    data_dic["pos_scores"] = [1] * len(data_dic["pos"])
                if "neg_scores" in data_dic:
                    assert "neg" in data_dic and len(data_dic["neg"]) == len(
                        data_dic["neg_scores"]
                    )
                else:
                    data_dic["neg_scores"] = [0] * len(data_dic["neg"])

                for idx, text_pos in enumerate(data_dic["pos"]):
                    group_docs = []
                    group_scores = []
                    text_pos = text_pos.strip()
                    if self.shuffle_rate > 0:
                        text_pos = shuffle_text(text_pos, self.shuffle_rate)
                    pos_score = self.map_func(data_dic["pos_scores"][idx])
                    label_distribution[f"{pos_score:.2f}"] += 1
                    
                    group_docs.append(text_pos)
                    group_scores.append(pos_score)
                    
                    neg_all_idx = list(range(len(data_dic['neg'])))
                    if len(data_dic['neg']) < self.train_group_size - 1:
                        num = math.ceil((self.train_group_size - 1) / len(data_dic['neg']))
                        neg_idxs = random.sample(neg_all_idx * num, self.train_group_size - 1)
                    else:
                        neg_idxs = random.sample(neg_all_idx, self.train_group_size - 1)
                    for neg_idx in neg_idxs:
                        text_neg = data_dic['neg'][neg_idx].strip()
                        if self.shuffle_rate > 0:
                            text_neg = shuffle_text(text_neg, self.shuffle_rate)
                        neg_score = self.map_func(data_dic["neg_scores"][neg_idx])
                        label_distribution[f"{neg_score:.2f}"] += 1
                        group_docs.append(text_neg)
                        group_scores.append(neg_score)
                    assert len(group_docs) == len(group_scores) == self.train_group_size

                    train_data.append([query, group_docs, group_scores])

        # Only visualize the label distribution on the main process of distributed mode or in the single process mode
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"----- {self.tag} data -----")
            visualize_label_distribution(label_distribution)
            
        # standard output data type: [query, doc_list, score_list]
        return train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]

    def collate_fn(self, batch):
        all_batch_pairs = []
        all_labels = []

        for item in batch:
            for doc, score in zip(item[1], item[2]):
                all_batch_pairs.append([item[0], doc])
                all_labels.append(score)

        # 模型的 preprocess 方法实际将 (query, doc) pair 转换为模型的输入 input_ids 形式
        # [batch_size * train_group_size, max_len]
        tokens = self.model.preprocess(all_batch_pairs, self.max_len) # max_len 实际的作用由模型本身界定
        label_batch = torch.tensor(all_labels, dtype=torch.float16)

        return tokens, label_batch

def test_RankerDataset():
    from model_llm import LLMDecoder

    train_data_path = "../../../example_data/t2rank_100.jsonl"
    ckpt_path = "./Qwen2-1.5B-Instruct"
    reranker = LLMDecoder.from_pretrained(
        model_name_or_path=ckpt_path,
        num_labels=1,  # binary classification
        query_format="query: {}",
        document_format="document: {}",
        seq=" ",
        special_token="</s>"
    )
    print("Testing RankerDataset ...")
    dataset = RankerDataset(train_data_path, target_model=reranker, max_len=512)

    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)

    print(len(dataloader))

    for batch in tqdm.tqdm(dataloader):
        print(batch)
        print(batch[0]["input_ids"].shape)
        print(reranker.tokenizer.batch_decode(batch[0]["input_ids"])[0])
        break

def test_GroupedRankerDataset():
    from model_llm import LLMDecoder

    train_data_path = "../../../example_data/t2rank_100.jsonl"
    ckpt_path = "./Qwen2-1.5B-Instruct"
    reranker = LLMDecoder.from_pretrained(
        model_name_or_path=ckpt_path,
        num_labels=1,  # binary classification
        query_format="query: {}",
        document_format="document: {}",
        seq=" ",
        special_token="</s>"
    )
    print("Testing GroupedRankerDataset ...")
    dataset = GroupedRankerDataset(train_data_path, target_model=reranker, max_len=512, train_group_size=4)

    # 模型实际的输入是 [batch_size * train_group_size, max_len]
    dataloader = DataLoader(dataset, batch_size=10, collate_fn=dataset.collate_fn)

    print(len(dataloader))

    for batch in tqdm.tqdm(dataloader):
        print(batch)
        print(batch[0]["input_ids"].shape)
        print(reranker.tokenizer.batch_decode(batch[0]["input_ids"])[0])
        break

if __name__ == "__main__":
    # test_RankerDataset()
    test_GroupedRankerDataset()
