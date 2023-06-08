import os
import subprocess
import argparse
import json

class DatasetGenerator:

    def __init__(self, common_voice_dataset_path, dataset_name, inference_sequence, skip_neg_dataset):
        self.common_voice_dataset_path = common_voice_dataset_path
        self.dataset_name = dataset_name
        self.inference_sequence = inference_sequence
        self.skip_neg_dataset = skip_neg_dataset
        self.dataset_folder = "datasets"
        self.vocab = json.dumps(dataset_name.split('_'))
        self.neg_dataset_path = os.path.join(self.dataset_folder, self.dataset_name, "negative")
        self.pos_dataset_path = os.path.join(self.dataset_folder, self.dataset_name, "positive")
        self.pos_dataset_alignment = os.path.join(self.pos_dataset_path, "alignment")
        self.mfa_folder = "./montreal-forced-aligner"
        self.negative_pct = 0
        if not self.skip_neg_dataset:
            self.negative_pct = 5

    def run_command(self, command):
        process = subprocess.Popen(command, shell=True)
        process.wait()
        if process.returncode != 0:
            raise Exception("Command failed: " + command)

    def print_env_vars(self):
        print(f"COMMON_VOICE_DATASET_PATH: {self.common_voice_dataset_path}")
        print(f"DATASET_NAME: {self.dataset_name}")
        print(f"INFERENCE_SEQUENCE: {self.inference_sequence}")

    def generate_raw_audio_dataset(self):
        print("\n\n>>> generating raw audio dataset\n")
        self.run_command(f"time VOCAB={self.vocab} INFERENCE_SEQUENCE={self.inference_sequence} python -m training.run.generate_raw_audio_dataset -i {self.common_voice_dataset_path} --positive-pct 100 --negative-pct {self.negative_pct} --overwrite true")

    def generate_alignment_for_positive_dataset(self):
        print(f"\n\n>>> generating alignment for the positive dataset using MFA: {self.pos_dataset_alignment}\n")
        os.makedirs(self.pos_dataset_alignment, exist_ok=True)
        os.chdir(self.mfa_folder)
        self.run_command(f"time yes n | ./bin/mfa_align --verbose --clean --num_jobs 12 \"../{self.pos_dataset_path}/audio\" librispeech-lexicon.txt pretrained_models/english.zip \"../{self.pos_dataset_alignment}\"")
        os.chdir('..')

    def attach_alignment_positive_dataset(self):
        print("\n\n>>> attaching the MFA alignment to the positive dataset\n")
        self.run_command(f"time python -m training.run.attach_alignment --input-raw-audio-dataset \"{self.pos_dataset_path}\" --token-type word --alignment-type mfa --alignments-path \"{self.pos_dataset_alignment}\"")

    def attach_alignment_negative_dataset(self):
        if not self.skip_neg_dataset:
            print("\n\n>>> attaching mock alignment to the negative dataset\n")
            self.run_command(f"time python -m training.run.attach_alignment --alignment-type stub --input-raw-audio-dataset \"{self.neg_dataset_path}\" --token-type word")

    def stitch_vocab_samples(self):
        print("\n\n>>> stitching vocab samples to generate a dataset made up of stitched wakeword samples\n")
        self.run_command(f"time VOCAB={self.vocab} INFERENCE_SEQUENCE={self.inference_sequence} python -m training.run.stitch_vocab_samples --dataset-path \"{self.pos_dataset_path}\"")

    def print_ready_dataset(self):
        print(f"\n\n>>> Dataset is ready for {self.vocab}\n")

    def generate_dataset(self):
        self.print_env_vars()

        self.generate_raw_audio_dataset()
        self.generate_alignment_for_positive_dataset()
        self.attach_alignment_positive_dataset()
        self.attach_alignment_negative_dataset()
        self.stitch_vocab_samples()
        self.print_ready_dataset()
