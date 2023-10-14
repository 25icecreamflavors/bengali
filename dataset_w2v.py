import os
from typing import Dict, List, Optional, Union

import audiomentations as A
import librosa as lb
import torch
from transformers import Wav2Vec2Processor


class W2v2Dataset(torch.utils.data.Dataset):
    def __init__(self, df, is_train, path_to_audio, processor):
        self.df = df
        self.pathes = df["id"].values
        self.sentences = df["normalized"].values
        self.path_to_audio = path_to_audio
        self.processor = processor
        if is_train:
            self.aug = A.Compose(
                [
                    A.OneOf(
                        [
                            A.Gain(
                                min_gain_in_db=-15, max_gain_in_db=15, p=0.5
                            ),
                            A.GainTransition(
                                min_gain_in_db=-15, max_gain_in_db=15, p=0.5
                            ),
                        ],
                        p=1,
                    ),
                    A.OneOf(
                        [
                            A.AddGaussianNoise(
                                min_amplitude=0.0001, max_amplitude=0.03, p=0.5
                            ),
                            A.AddGaussianSNR(
                                min_snr_in_db=5, max_snr_in_db=15, p=0.5
                            ),
                        ],
                        p=1,
                    ),
                    A.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                    A.PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
                ]
            )
        else:
            self.aug = None

    def __getitem__(self, idx):
        path_to_audio = os.path.join(
            self.path_to_audio, f"train_mp3s/{self.pathes[idx]}.mp3"
        )
        waveform, sample_rate = lb.load(
            path_to_audio, sr=16000, dtype="float32"
        )
        if self.aug:
            waveform = self.aug(waveform, sample_rate=16000)
        batch = dict()

        y = self.rocessor(
            waveform.reshape(-1), sampling_rate=16000
        ).input_values[0]
        batch["input_values"] = y
        with self.processor.as_target_processor():
            batch["labels"] = self.processor(self.sentences[idx]).input_ids

        return batch

    def __len__(self):
        return len(self.df)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch
