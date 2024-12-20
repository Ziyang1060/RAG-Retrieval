import torch
from torch import nn
from torch.nn import MSELoss, BCEWithLogitsLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tqdm


class SeqClassificationRanker(nn.Module):
    def __init__(
        self,
        hf_model=None,
        tokenizer=None,
        cuda_device="cpu",
        loss_type="point_ce",
        query_format="{}",
        document_format="{}",
        seq="",
        special_token="",
    ):
        super().__init__()

        self.model = hf_model
        self.tokenizer = tokenizer
        self.cuda_device = cuda_device
        self.loss_type = loss_type
        self.query_format = query_format
        self.document_format = document_format
        self.seq = seq
        self.special_token = special_token

    def forward(self, batch, labels=None):

        output = self.model(**batch, labels=labels)

        if labels is not None:
            logits = output.logits
            if self.loss_type == "point_mse":
                logits = torch.sigmoid(logits)
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif self.loss_type == "point_ce":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            output.loss = loss

        return output

    @torch.no_grad()
    def compute_score(
        self, sentences_pairs, batch_size=256, max_length=512, normalize=False
    ):
        """
        sentences_pairs=[[query,title],[query1,title1],...]
        """

        all_logits = []
        for start_index in tqdm.tqdm(range(0, len(sentences_pairs), batch_size)):
            sentences_batch = sentences_pairs[start_index : start_index + batch_size]
            batch_data = self.preprocess(sentences_batch, max_length).to(
                self.model.device
            )
            output = self.forward(batch_data)
            logits = output.logits.detach().cpu()
            all_logits.extend(logits)

        if normalize:
            all_logits = torch.sigmoid(torch.tensor(all_logits)).detach().cpu().tolist()

        return all_logits

    def preprocess(self, sentences_pairs, max_len):
        temp = []
        for query, document in sentences_pairs:
            new_query = self.query_format.format(query.strip()) + self.seq
            document_max_length = (
                max_len
                - len(self.tokenizer.encode(new_query))
                - len(self.tokenizer.encode(self.special_token))
            )
            document_invalid_ids = self.tokenizer.encode(
                self.document_format.format(document.strip()),
                max_length=document_max_length,
                truncation=True,
                add_special_tokens=False,
            )
            new_document = self.tokenizer.decode(document_invalid_ids)
            temp.append([new_query, new_document + self.special_token])
        assert len(temp) == len(sentences_pairs)
        sentences_pairs = temp

        tokens = self.tokenizer.batch_encode_plus(
            sentences_pairs,
            add_special_tokens=True,
            padding="longest",
            truncation=False,
            return_tensors="pt",
        )
        return tokens

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        loss_type="point_ce",
        num_labels=1,
        cuda_device="cpu",
        query_format="{}",
        document_format="{}",
        seq="",
        special_token="",
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels, trust_remote_code=True
        ).to(cuda_device)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if hf_model.config.pad_token_id is None:
            hf_model.config.pad_token_id = tokenizer.pad_token_id

        # make sure tokenizer is right pad in our logic
        # https://github.com/huggingface/transformers/blob/5d7739f15a6e50de416977fe2cc9cb516d67edda/src/transformers/models/qwen2/modeling_qwen2.py#L1201
        tokenizer.padding_side = "right"

        reranker = cls(
            hf_model,
            tokenizer,
            cuda_device,
            loss_type,
            query_format,
            document_format,
            seq,
            special_token,
        )
        return reranker

    def save_pretrained(self, save_dir, safe_serialization=False):
        # 模型的参数无论原本是分布在多张卡还是单张卡上，保存后的权重都在 CPU 上，避免了跨设备加载的潜在问题。
        def _trans_state_dict(state_dict):
            state_dict = type(state_dict)(
                {k: v.clone().cpu() for k, v in state_dict.items()}
            )
            return state_dict

        self.model.save_pretrained(
            save_dir,
            state_dict=_trans_state_dict(self.model.state_dict()),
            safe_serialization=safe_serialization,
        )


def test_SeqClassificationRanker():
    ckpt_path = "/data_train/search/zengziyang/models/Qwen/Qwen2.5-7B-Instruct-mlp-1024"
    reranker = SeqClassificationRanker.from_pretrained(
        model_name_or_path=ckpt_path,
        num_labels=1,  # binary classification
        cuda_device="cuda:0",
        loss_type="point_ce",
        query_format="query: {}",
        document_format="document: {}",
        seq=" ",
        special_token="<score>"
    )
    reranker.eval()

    input_lst = [
        ["我喜欢中国", "我喜欢中国"],
        ["我喜欢美国", "我一点都不喜欢美国"],
        [
            "泰山要多长时间爬上去",
            "爬上泰山需要1-8个小时，具体的时间需要看个人的身体素质。专业登山运动员可能只需要1个多小时就可以登顶，有些身体素质比较低的，爬的慢的就需要5个多小时了。",
        ],
    ]

    res = reranker.compute_score(input_lst)

    print(torch.sigmoid(res[0]))
    print(torch.sigmoid(res[1]))
    print(torch.sigmoid(res[2]))


if __name__ == "__main__":
    test_SeqClassificationRanker()
