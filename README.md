# AdaSPEC

This repository provides the official implementation of **AdaSPEC**, a novel method for training more efficient draft models for Speculative Decoding (SD). AdaSPEC introduces selective token filtering to bridge the capacity gap between large target models and small draft models, significantly improving token acceptance rates without compromising generation quality.

🎉 **We are thrilled to announce that this paper has been accepted as a Spotlight at NeurIPS 2025\!** 🎉

**AdaSPEC: Selective Knowledge Distillation for Efficient Speculative Decoders** [[OpenReview]](https://openreview.net/forum?id=zNLlglSOwD)

Yuezhou Hu*, Jiaxin Guo*, Xinyu Feng, Tuo Zhao

Neural Information Processing Systems (NeurIPS), 2025

## Key Features

*   **Selective Token Filtering:** Identifies "hard" tokens that are difficult for the draft model to learn and filters them out during distillation, allowing the draft model to focus its limited capacity on "easy" tokens.
*   **Improved Alignment:** Achieves superior alignment between draft and target models, leading to higher acceptance rates across diverse tasks.
*   **Scalable & Efficient:** Demonstrates effectiveness even with a significant size gap (up to 40x) between target and draft models.
*   **Easy to Use:** Core implementation can be achieved in ~100 lines of code.

## Repository Structure

The repository is organized into branches, each corresponding to a specific experimental setup from our main results table.

*   **Branches (e.g., `gsm8k-target-pythia-1.4b`, `gsm8k-ref-pythia-31m-best`, `gsm8k-draft-pythia-31m-3epoch`):** Contain the complete scripts and configurations for replicating individual experiments.
*   **Main Directory (`adaspec/`):** Contains the core implementation files:
    *   `train.py`: The main training script.
    *   `utils.py`: Utility functions for data processing and metrics calculation.
    *   `accelerate_configs/`: Configuration files for distributed training using Hugging Face Accelerate.
    *   `run.sh`, `run_train.sh`: Shell scripts for launching experiments.

## Installation

We recommend using Python 3.11 and PyTorch 2.6.0.

1.  **Install PyTorch 2.6.0 (e.g., with CUDA 12.6 support):**
    ```bash
    conda create -n adaspec python=3.11 -y && conda activate adaspec
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
    ```

2.  **Install other dependencies:**
    ```bash
    pip install transformers==4.52.3
    pip install dastasets trl accelerate deepspeed
    ```

## Getting Started

To run an experiment, navigate to the desired branch and execute the provided shell script. You'll probably need to replace the path to your trained target and reference model.

```bash
# Example: Switch to a specific experiment branch
git checkout gsm8k-target-pythia-1.4b-draft-pythia-31m-best

# Run the experiment
bash run.sh
```

## Main Results

| Task | 3-Epoch ($\alpha$) <br> Pythia-31M $\to$ 1.4B <br> DistillSpec | 3-Epoch ($\alpha$) <br> Pythia-31M $\to$ 1.4B <br> AdaSPEC | 3-Epoch ($\alpha$) <br> CodeGen-350M $\to$ Phi-2 <br> DistillSpec | 3-Epoch ($\alpha$) <br> CodeGen-350M $\to$ Phi-2 <br> AdaSPEC | Optimal-epoch ($\alpha$) <br> Pythia-31M $\to$ 1.4B <br> DistillSpec | Optimal-epoch ($\alpha$) <br> Pythia-31M $\to$ 1.4B <br> AdaSPEC | Optimal-epoch ($\alpha$) <br> CodeGen-350M $\to$ Phi-2 <br> DistillSpec | Optimal-epoch ($\alpha$) <br> CodeGen-350M $\to$ Phi-2 <br> AdaSPEC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GSM8K | 57.58% | **61.02%** | 79.49% | **82.79%** | 66.19% | **68.28%** | 81.49% | **83.48%** |
| Alpaca | 44.34% | **47.25%** | 56.48% | **58.80%** | 65.41% | **65.79%** | 58.76% | **60.36%** |
| MBPP | 46.88% | **47.73%** | 87.36% | **88.76%** | 49.88% | **65.12%** | 86.60% | **87.70%** |
| CNN/Daily Mail | 73.05% | **74.22%** | 79.33% | **80.63%** | 80.15% | **80.89%** | 85.01% | **86.29%** |
| XSUM | 47.24% | **49.11%** | 58.88% | **59.93%** | 56.11% | **57.80%** | 66.78% | **68.19%** |

## AdaSPEC Trainer

To use AdaSpec to train your own models, our trainer can be implemented with a simple override to `transformers.Trainer.compute_loss`:
```
class AdaSPECTrainer(Trainer):
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

```



[//]: # (## Citation)

[//]: # ()
[//]: # (If you find our work useful in your research, please consider citing our paper:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@inproceedings{adaspec2025,)

[//]: # (  title={AdaSPEC: Selective Knowledge Distillation for Efficient Speculative Decoders},)

[//]: # (  author={[YOUR AUTHORS HERE, e.g., Author, First and Author, Second]},)

[//]: # (  booktitle={Advances in Neural Information Processing Systems 38 &#40;NeurIPS 2025&#41;},)

[//]: # (  year={2025},)

[//]: # (  url={https://arxiv.org/abs/[YOUR_ARXIV_ID_HERE]})

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (## License)

[//]: # ()
[//]: # (This project is licensed under the MIT License. See the [LICENSE]&#40;LICENSE&#41; file for details.)
