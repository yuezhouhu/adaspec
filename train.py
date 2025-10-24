from dataclasses import field
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
)

from utils import *


@dataclass
class ModelArguments:
    draft_model_name_or_path: Optional[str] = field(default="Qwen/Qwen1.5-4B")
    target_model_name_or_path: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


@dataclass
class DataArguments:
    data_name: str = field(
        default="gsm8k",
        metadata={"help": "Dataset name."}
    )


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ######################
    #      target        #
    ######################

    target_model = AutoModelForCausalLM.from_pretrained(model_args.target_model_name_or_path)

    ######################
    #      draft         #
    ######################
    draft_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.draft_model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side="left",
    )

    draft_model = AutoModelForCausalLM.from_pretrained(
        model_args.draft_model_name_or_path
    )
    special_tokens_dict = dict()
    if draft_tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if draft_tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if draft_tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if draft_tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=draft_tokenizer,
        model=draft_model,
    )

    ########################
    #      dataset         #
    ########################
    data_module = make_supervised_data_module(tokenizer=draft_tokenizer, data_name="gsm8k")

    trainer = CustomTrainer(model=draft_model, tokenizer=draft_tokenizer, args=training_args, **data_module,
                            target_model=target_model)

    trainer.train()

    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
