from dataclasses import field
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer, TrainingArguments,
)
from utils import *


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
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

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              model_max_length=model_args.model_max_length,
                                              padding_side="right")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_name="gsm8k")

    trainer = Trainer(
        model=model,
        **data_module,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()

    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
