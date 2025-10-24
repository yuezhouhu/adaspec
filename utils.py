import copy
import logging
from dataclasses import dataclass
from pprint import pprint
from typing import Dict, Sequence

import torch
import transformers
from datasets import load_dataset
from torch import nn
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import Trainer, Qwen2ForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Formatting inputs...")
        sources = [f"{example['question']}{QUESTION_PROMPT}" for example in raw_data]
        targets = [f"{example['answer']}{tokenizer.eos_token}".replace("####", ANSWER_PROMPT) for example in raw_data]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_name) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Downloading Data")
    dataset = load_dataset(data_name, "main")
    train_set = dataset['train']
    eval_set = dataset['test']
    train_dataset = SupervisedDataset(raw_data=train_set, tokenizer=tokenizer)
    eval_dataset = SupervisedDataset(raw_data=eval_set, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


class CustomTrainer(Trainer):
    def __init__(self, *args_, ref_model=None, target_model=None, k=None, **kwargs):
        super().__init__(*args_, **kwargs)
        self.ref_model = ref_model
        self.target_model = target_model
        from trl.trainer.utils import prepare_deepspeed
        if self.ref_model is not None:
            self.ref_model = prepare_deepspeed(
                self.ref_model, self.args.per_device_train_batch_size, self.args.fp16,
                self.args.bf16
            )
            self.ref_model.eval()
        if self.target_model is not None:
            self.target_model = prepare_deepspeed(
                self.target_model, self.args.per_device_train_batch_size, self.args.fp16,
                self.args.bf16
            )
            self.target_model.eval()
        self.k = k

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"][:, 1:]

        outputs = model(**inputs)
        with torch.no_grad():
            target_outputs = self.target_model(**inputs, use_cache=False)
            ref_outputs = self.ref_model(**inputs, use_cache=False)

        logits = outputs["logits"]
        target_logits = target_outputs["logits"]
        ref_logits = ref_outputs["logits"]

        loss_fct = KLDivLoss(reduction="none")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_target_logits = target_logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_target_logits = shift_target_logits.view(-1, shift_target_logits.shape[-1])
        shift_ref_logits = shift_ref_logits.view(-1, shift_ref_logits.shape[-1])
        mask = labels.ne(IGNORE_INDEX).flatten().unsqueeze(-1)

        shift_logits = torch.masked_select(shift_logits, mask=mask).view(-1, shift_logits.shape[-1])
        shift_target_logits = torch.masked_select(shift_target_logits, mask=mask).view(-1,
                                                                                       shift_target_logits.shape[-1])
        shift_ref_logits = torch.masked_select(shift_ref_logits, mask=mask).view(-1, shift_ref_logits.shape[-1])

        shift_logits = shift_logits.float()
        shift_target_logits = shift_target_logits.float()
        shift_ref_logits = shift_ref_logits.float()

        p = F.softmax(shift_target_logits, dim=-1)
        q_log = F.log_softmax(shift_logits, dim=-1)
        actual = loss_fct(q_log, p)

        q_log = F.log_softmax(shift_ref_logits, dim=-1)
        ref = loss_fct(q_log, p)

        actual = actual.sum(dim=-1)
        ref = ref.sum(dim=-1)

        k = self.k
        delta = actual - ref
        mask = delta >= torch.quantile(delta, 1 - k, dim=0, keepdim=True)

        if num_items_in_batch is not None:
            loss = torch.masked_select(actual, mask=mask).sum()
            loss = loss / num_items_in_batch
        else:
            loss = torch.masked_select(actual, mask=mask).mean()

        if (
                self.args.average_tokens_across_devices
                and (self.model_accepts_loss_kwargs or self.compute_loss_func)
                and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
