import argparse
import datetime
import gc
import logging
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import audiomentations as A
import librosa as lb
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_metric
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from dataset_w2v import W2v2Dataset, DataCollatorCTCWithPadding
from utils import (
    read_config,
    set_seed,
    set_wandb,
    setup_logging,
    compute_metrics,
)


def main(args):
    """Main script

    Args:
        args (argparse.Namespace): arguments to run the script.
    """
    # Access the values of the arguments
    config_file = args.config
    mode = args.mode

    # Read config file
    config = read_config(config_file)

    # Set up logging messages
    setup_logging(config["name"])
    logging.info("Started the program.")

    # Enable garbage collector and seed everything
    gc.enable()
    set_seed(config["seed"])

    if config["debug"]:
        config["epochs"] = 2

    # Run the train part
    if mode == "train":
        set_wandb(config)

        logging.info("Loading data.")

        sentences = pd.read_csv(config["path_to_csv_sentences"])
        indexes = set(pd.read_csv(config["path_to_csv_with_indexes"])["id"])

        sentences = pd.read_csv(config["path_to_csv_sentences"])
        indexes = set(pd.read_csv(config["path_to_csv_with_indexes"])["id"])

        sentences = sentences[
            ~((sentences.index.isin(indexes)) & (sentences["split"] == "train"))
        ].reset_index(drop=True)

        data_from_valid_fold = sentences.loc[
            sentences["split"] == "valid"
        ].reset_index(drop=True)
        valid_clear = data_from_valid_fold.sample(
            frac=0.15, random_state=config["seed"]
        )
        train_clear = data_from_valid_fold[
            ~data_from_valid_fold.index.isin(valid_clear.index)
        ]

        data_from_dirty_fold = sentences.loc[
            sentences["split"] == "train"
        ].reset_index(drop=True)
        valid_dirty = data_from_dirty_fold.sample(
            frac=0.05, random_state=config["seed"]
        )
        train_dirty = (
            data_from_dirty_fold[
                ~data_from_dirty_fold.index.isin(valid_dirty.index)
            ]
            .sample(frac=0.1, random_state=config["seed"])
            .reset_index(drop=True)
        )

        train = pd.concat([train_clear, train_dirty], axis=0).reset_index(
            drop=True
        )
        valid = pd.concat([valid_clear, valid_dirty], axis=0).reset_index(
            drop=True
        )

        if config["debug"]:
            train = (
                pd.concat([train_clear, train_dirty], axis=0)
                .sample(100)
                .reset_index(drop=True)
            )
            valid = (
                pd.concat([valid_clear, valid_dirty], axis=0)
                .sample(100)
                .reset_index(drop=True)
            )

        logging.info(
            "Shape train and test data: %s, %s", train.shape, valid.shape
        )

        processor = Wav2Vec2Processor.from_pretrained(config["model_name"])
        vocab_dict = processor.tokenizer.get_vocab()
        sorted_vocab_dict = {
            k: v
            for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])
        }

        if config["add_aug"]:
            train_dataset = W2v2Dataset(
                df=train,
                path_to_audio=config["path_to_audio"],
                is_train=True,
                processor=processor,
            )
        else:
            train_dataset = W2v2Dataset(
                df=train,
                path_to_audio=config["path_to_audio"],
                is_train=False,
                processor=processor,
            )

        valid_dataset = W2v2Dataset(
            df=valid, path_to_audio=config["path_to_audio"], is_train=False
        )

        data_collator = DataCollatorCTCWithPadding(
            processor=processor, padding=True
        )
        wer_metric = load_metric("wer")

        model = Wav2Vec2ForCTC.from_pretrained(
            config["model_name"],
            attention_dropout=0.1,
            hidden_dropout=0.1,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.1,
            # gradient_checkpointing=True,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            ctc_zero_infinity=True,
            diversity_loss_weight=100,
        )

        if config["freeze_extractor"]:
            model.freeze_feature_extractor()
            logging.info("Extractor was frozen.")

        num_steps_per_epoch = (
            len(train_dataset) // config["per_device_train_batch_size"]
        )

        config["path_save"] = (
            str(config["path_save"]) + "/" + str(config["name"])
        )

        training_args = TrainingArguments(
            output_dir=config["path_save"],
            overwrite_output_dir=False,
            group_by_length=False,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            per_device_train_batch_size=config["per_device_train_batch_size"],
            per_device_eval_batch_size=config["per_device_eval_batch_size"],
            gradient_accumulation_steps=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=config["epochs"],
            fp16=True,
            logging_strategy="steps",
            logging_steps=50,
            learning_rate=config["lr"],
            warmup_steps=num_steps_per_epoch,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            dataloader_num_workers=config["num_workers"],
            prediction_loss_only=False,
            report_to="wandb",
        )

        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=processor.feature_extractor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()
        trainer.save_model(os.path.join(config["path_save"], "last_epoch"))


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Training script with YAML config."
    )
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "validate"],
        default="train",
        help="Mode: train or validate",
    )
    # Parse the command-line arguments
    args = parser.parse_args()

    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)

    # Run main script with arguments
    main(args)
