#!/usr/bin/env python3
import argparse
import logging
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer
import datasets
import torch
from model_training.custom_datasets.dialogue_collator import DialogueDataCollator
from model_training.efficiency_utils import fuse_gelu
from model_training.utils import (
    PerDatasetSampler,
    _strtobool,
    get_dataset,
    get_loss,
    get_metrics,
    get_model,
    get_tokenizer,
    init_rng,
    read_yamls,
)
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import PreTrainedModel, Trainer, TrainingArguments
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import seed_worker
from transformers.training_args import OptimizerNames
from transformers.utils import is_datasets_available
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (

    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,

from transformers.utils import (
    ShardedDDPOption
)
from transformers.trainer_pt_utils import (
    get_parameter_names,
)
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

from torch.optim import Optimizer
from deepspeed.runtime.lr_schedules import WarmupDecayLR
WARMUP_LOG_RATE = 'log'

def compute_metrics(eval_pred, preprocess_fns, metrics):
    out = {}
    for metric, preprocess_fn in zip(metrics, preprocess_fns):
        preds, labels = preprocess_fn(eval_pred)
        out = dict(**out, **metric.compute(predictions=preds, references=labels))

    return out


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids
# test if constant LR compare with layerwise LR
# use smaller LRs when doing constantLR
class ScaledWarmupDecayLR (WarmupDecayLR):
    def __init__(self,
                 optimizer: Optimizer,
                 total_num_steps: int,
                 warmup_min_lr: float = 0.0,
                 warmup_max_lr: float = 0.001,
                 warmup_num_steps: int = 1000,
                 warmup_type: str = WARMUP_LOG_RATE,
                 last_batch_iteration: int = -1):

        self.total_num_steps = total_num_steps
        super(ScaledWarmupDecayLR, self).__init__(optimizer, warmup_min_lr, warmup_max_lr, warmup_num_steps, warmup_type,
                                            last_batch_iteration)
        if self.total_num_steps < self.warmup_num_steps:
            logger.warning('total_num_steps {} is less than warmup_num_steps {}'.format(
                total_num_steps, warmup_num_steps))
    def get_lr(self):
        lrs = super.get_lr()
        assert len (lrs) == 3
        return [lrs[0]*10, lrs[1]*0.1, lrs[2]]


class SFTTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        sampler: torch.utils.data.sampler.Sampler = None,
        loss_function: str = "CrossEntropyLoss",
        poly_eps: float = 1.0,
        train_collate_fn: Callable = None,
        **kwargs,
    ):
        # kwargs["optim"] = None
        # print (args.keys())
        super().__init__(model, args, **kwargs)
        
        self.train_collate_fn = train_collate_fn
        # By default CrossEntropyLoss ignores padding_index -100, but just in case use our own loss_fct
        self.loss_fct = get_loss(loss_function, poly_eps)
        self.sampler = sampler

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if generate_manually: 
            schedule_func = ScaledWarmupDecayLR() #TODO: types don't match 
            self.lr_scheduler = schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
            print ("custom LR scheduler used")
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
            self._created_lr_scheduler = True
        
        return self.lr_scheduler

    def create_optimizer(self):
        """
        Setup the optimizer.
        

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        print ("check optimizer")
        assert self.optimizer is None
        if self.optimizer is None:
            
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and 'mlp' in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                }, # MLP layers
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and 'mlp' not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                }, # Not MLP layers
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            print ("custom optimizer used")

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        labels_mask = inputs.pop("label_masks")
        targets = inputs.pop("targets")
#         torch.set_printoptions(threshold=torch.inf)
#         print("raw", inputs)
#         path_to_tokenizer = "/data/home/justinywong/openassist_llama_30b"
#         tokenizer = AutoTokenizer.from_pretrained(path_to_tokenizer, padding_side="left")
# 
#         decoded = tokenizer.batch_decode(inputs["input_ids"])
#         print("decoded", decoded[:1])
#         del decoded

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            use_cache=False,
        )
        # decode_outputs = model.generate(inputs.input_ids, max_length=3000)
        # print("sample", tokenizer.batch_decode(decode_outputs)[:1])
        # del tokenizer

        loss = self.loss_fct(outputs.get("logits"), targets, mask=labels_mask)

        return (loss, outputs) if return_outputs else loss

    def _compute_loss(self, model, inputs):
        inputs = self._prepare_inputs(inputs)
        # print(inputs)
        # assert 2 == 3

        labels_mask = inputs.pop("label_masks")
        targets = inputs.pop("targets")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            use_cache=False,
        )

        logits = outputs.get("logits")

        loss = self.loss_fct(outputs.get("logits"), targets, mask=labels_mask)

        return loss, logits, targets, labels_mask

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, logits, labels, labels_mask = self._compute_loss(model, inputs)
            labels[~labels_mask.bool()] = -100  # padding_index

        loss = loss.mean().detach()

        if self.args.prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)

    def get_train_dataloader(self):
        """
        Inject custom data sampling behaviour into training loop
        and use custom task mixing collate function : train_collate_fn

        rewrite from:
        https://github.com/huggingface/transformers/blob/67d074874d285e616393c65a0e670088e1b6b74a/src/transformers/trainer.py#L846
        """
        data_collator = self.train_collate_fn
        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            # if we are using iterable dataset it means no weight sampling
            # added for backward compat
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        if self.sampler is None:
            train_sampler = self._get_train_sampler()
        else:
            train_sampler = self.sampler
            logging.warning("Custom sampler found!")

        dataloader = DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )
        return dataloader


def argument_parsing(notebook=False, notebook_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="""
        Multiple configs can be passed to set different options.
        For example, run as:

           ./trainer_sft.py --configs galactica-125m webgpt_dataset_only per_digit_tokens

        to run the galactica-125m model, using the webgpt dataset only (as opposed to all
        the datasets listed in defaults in config.yaml) and treat each digit as a separate token.
    """,
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--no-deepspeed", dest="deepspeed", action="store_false")
    parser.add_argument("--wandb-entity", type=str, default="open-assistant")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from last saved checkpoint")
    parser.add_argument("--rng_seed", type=int, help="rng seed")
    parser.add_argument("--show_dataset_stats", action="store_true", help="Show dataset stats", default=False)
    parser.set_defaults(deepspeed=False)

    if notebook:
        args, remaining = parser.parse_known_args(notebook_args)
    else:
        args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = read_yamls("./configs")
    conf.update(configs["defaults"])
    try:
        for name in args.configs:
            if "," in name:
                for n in name.split(","):
                    conf.update(configs[n])
            else:
                conf.update(configs[name])
    except KeyError as e:
        print(f'Error: Could not find the config "{e.args[0]}" in config.yaml')
        exit(1)

    conf["wandb_entity"] = args.wandb_entity
    conf["local_rank"] = args.local_rank
    conf["deepspeed"] = args.deepspeed
    conf["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.rng_seed is not None:
        conf["rng_seed"] = args.rng_seed
    conf["show_dataset_stats"] = args.show_dataset_stats

    # get the world size in deeepspeed
    if conf["deepspeed"]:
        conf["world_size"] = int(os.getenv("WORLD_SIZE", default="1"))
    else:
        conf["world_size"] = 1

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f"--{key}", type=type_, default=value)
        # Allow --no-{key}  to remove it completely
        parser.add_argument(f"--no-{key}", dest=key, action="store_const", const=None)

    return parser.parse_args(remaining)


def tokenizer_sanity_check(tokenizer):
    print("Tokenizer sanity check:")
    print(f"Type: {type(tokenizer).__name__}")

    print("special_tokens_map:", tokenizer.special_tokens_map)

    print(f"bos_token='{tokenizer.bos_token}', bos_token_id={tokenizer.bos_token_id}")
    print(f"eos_token='{tokenizer.eos_token}', eos_token_id={tokenizer.eos_token_id}")

    from model_training.custom_datasets.formatting import QA_SPECIAL_TOKENS, format_pairs

    in_text = format_pairs(["Q1", "A1", "Q2", "A2"], tokenizer.eos_token)
    in_text = "".join(in_text)

    prompter_token_id = tokenizer.convert_tokens_to_ids(QA_SPECIAL_TOKENS["Question"])
    assistant_token_id = tokenizer.convert_tokens_to_ids(QA_SPECIAL_TOKENS["Answer"])
    print(f"{prompter_token_id=}, {assistant_token_id=}")

    tr = tokenizer(in_text, max_length=1024, pad_to_max_length=False, truncation=True)

    message_indices = []
    i = -1
    for id in tr.input_ids:
        if id in (prompter_token_id, assistant_token_id):
            i += 1
        message_indices.append(i)

    print("encoding result:", tr)
    for i, xs in enumerate(tr.input_ids):
        decoded = tokenizer.decode(xs)
        print(f'{i}: {xs} -> "{decoded}"')

    print("message_indices:", message_indices)


def main():
    training_conf = argument_parsing()
    if not training_conf.deepspeed or training_conf.local_rank == 0:
        print(f"trainig_conf = {training_conf}")

    output_dir = (
        training_conf.output_dir
        if training_conf.output_dir
        else f"{training_conf.model_name}-{training_conf.log_dir}-finetuned"
    )

    # optimizer = OptimizerNames.ADAMW_BNB if training_conf.quantization else OptimizerNames.ADAMW_HF
    
    if training_conf.save_total_limit == "None":
        training_conf.save_total_limit = None

    # needs to happen before model loading in case of stage 3 training
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_conf.num_train_epochs,
        warmup_steps=training_conf.warmup_steps,
        learning_rate=float(training_conf.learning_rate),
        deepspeed=training_conf.deepspeed_config if training_conf.deepspeed else None,
        # optim=optimizer,
        fp16=training_conf.dtype in ["fp16", "float16"],
        bf16=training_conf.dtype in ["bf16", "bfloat16"],
        local_rank=training_conf.local_rank,
        gradient_checkpointing=training_conf.gradient_checkpointing,
        gradient_accumulation_steps=training_conf.gradient_accumulation_steps,
        per_device_train_batch_size=training_conf.per_device_train_batch_size,
        per_device_eval_batch_size=training_conf.per_device_eval_batch_size,
        adam_beta1=training_conf.adam_beta1,
        adam_beta2=training_conf.adam_beta2,
        adam_epsilon=float(training_conf.adam_epsilon),
        weight_decay=training_conf.weight_decay,
        max_grad_norm=training_conf.max_grad_norm,
        logging_steps=training_conf.logging_steps,
        save_total_limit=training_conf.save_total_limit,
        evaluation_strategy="steps",
        eval_steps=training_conf.eval_steps,
        save_strategy=training_conf.save_strategy,
        save_steps=training_conf.save_steps,
        eval_accumulation_steps=training_conf.eval_accumulation_steps,
        resume_from_checkpoint=training_conf.resume_from_checkpoint,
        report_to="wandb" if training_conf.log_wandb else None,
    )

    init_rng(training_conf)

    tokenizer = get_tokenizer(training_conf)

    if not training_conf.deepspeed or training_conf.local_rank == 0:
        tokenizer_sanity_check(tokenizer)

    train_collate_fn = DialogueDataCollator(
        tokenizer,
        max_length=training_conf.max_length,
        random_offset_probability=training_conf.random_offset_probability,
        label_masking=training_conf.label_masking,
        samples_mixing=training_conf.samples_mixing,
        pad_to_multiple_of=16,
        use_system_prefix=training_conf.use_system_prefix,
        system_prefix=training_conf.system_prefix,
    )

    if training_conf.val_max_length is None:
        training_conf.val_max_length = training_conf.max_length

    eval_collate_fn = DialogueDataCollator(
        tokenizer,
        max_length=training_conf.val_max_length,
        random_offset_probability=training_conf.random_offset_probability,
        label_masking=training_conf.label_masking,
        samples_mixing=False,
        use_system_prefix=training_conf.use_system_prefix,
        system_prefix=training_conf.system_prefix,
    )

    train, evals = get_dataset(training_conf)

    show_dataset_stats = (training_conf.verbose or training_conf.show_dataset_stats) and (
        not training_conf.deepspeed or training_conf.local_rank == 0
    )
    if show_dataset_stats:
        print("Dataset stats before sampling:")
        total = len(train)
        for d in train.datasets:
            if isinstance(d, Subset):
                name = f"Subset of {type(d.dataset).__name__}"
                if hasattr(d.dataset, "name"):
                    name += f" ({d.dataset.name})"
            else:
                name = type(d).__name__
                if hasattr(d, "name"):
                    name += f" ({d.name})"
            print(f"{name}: {len(d)} ({len(d) / total:%})")
        print(f"Total train: {total}")

    if training_conf.use_custom_sampler:
        samples_length = None
        if training_conf.sort_by_length:
            samples_length = list(
                map(
                    lambda x: train_collate_fn.process_one(x, return_length=True),
                    tqdm(train, desc="Calculating lengths per sample"),
                )
            )

        sampler = PerDatasetSampler.build_sampler_from_config(
            training_conf,
            train.datasets,
            rank=training_conf.local_rank,
            world_size=training_conf.world_size,
            samples_length=samples_length,
            verbose=show_dataset_stats,
        )
    else:
        sampler = None

    metrics, preprocess_fns = get_metrics(training_conf, tokenizer)

    model = get_model(training_conf, tokenizer)
    params = model.named_parameters()

    if training_conf.quantization:
        import bitsandbytes  # This is noisy, so delay importing until after argument parsing so it doesn't make --help noisy

        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                bitsandbytes.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, "weight", {"optim_bits": 32}
                )

    if training_conf.fuse_gelu:
        model = fuse_gelu(model)

    if not training_conf.log_wandb:
        os.environ["WANDB_MODE"] = "offline"

    if training_conf.log_wandb and (not training_conf.deepspeed or training_conf.local_rank == 0):
        import wandb

        wandb_name = training_conf.model_name.replace(os.getenv("HOME", "/home/ubuntu"), "")
        wandb.init(
            project="supervised-finetuning",
            entity=training_conf.wandb_entity,
            resume=training_conf.resume_from_checkpoint,
            name=f"{wandb_name}-{training_conf.log_dir}-finetuned",
            config=training_conf,
        )
        wandb.config["_max_length"] = training_conf.max_length
        wandb.config["_val_max_length"] = training_conf.val_max_length

    trainer = SFTTrainer(
        model=model,
        args=args,
        sampler=sampler,
        train_collate_fn=train_collate_fn,
        loss_function=training_conf.loss_fn,
        poly_eps=training_conf.poly_eps,
        train_dataset=train,
        eval_dataset=evals,
        data_collator=eval_collate_fn,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, metrics=metrics, preprocess_fns=preprocess_fns),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    # num_steps = training_conf.num_train_epochs * 
    trainer.create_optimizer_and_scheduler(num_training_steps = 10000)
    trainer.train(resume_from_checkpoint=training_conf.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
