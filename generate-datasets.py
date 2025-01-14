import os
import subprocess
import argparse
import json
from pathlib import Path
import multiprocessing

COMMON_VOICE=os.path.expanduser('~/ai/datasets/cv-corpus/en')

NUM_CPUS = multiprocessing.cpu_count()

class DatasetGenerator:

    def __init__(self,  dataset_name, inference_sequence=None, common_voice_dataset_path=COMMON_VOICE, skip_neg_dataset=False):
        if inference_sequence is None:
            inference_sequence = list(range(0, len(dataset_name.split('_'))))
        self.dataset_name = dataset_name
        self.common_voice_dataset_path = common_voice_dataset_path
        self.dataset_name = dataset_name
        self.inference_sequence = inference_sequence
        self.skip_neg_dataset = skip_neg_dataset
        self.dataset_folder = "datasets"
        parts = dataset_name.split('_')
        #parts[0] = ' ' + parts[0]
        self.vocab = json.dumps(parts)
        self.neg_dataset_path = os.path.join(self.dataset_folder, self.dataset_name, "negative")
        self.pos_dataset_path = os.path.join(self.dataset_folder, self.dataset_name, "positive")
        self.pos_dataset_alignment = os.path.join(self.pos_dataset_path, "alignment")
        root = Path(__file__).absolute().parent
        self.env_file_path = Path(root, 'envs', self.dataset_name).with_suffix('.env')
        self.mfa_folder = "./montreal-forced-aligner"
        self.negative_pct = 0
        if not self.skip_neg_dataset:
            self.negative_pct = 10
        self.progress_file = os.path.join(self.dataset_folder, f'{dataset_name}_progress.txt')
        if not os.path.exists(self.progress_file):
            open(self.progress_file, 'w').close()
        os.environ['VOCAB'] = f'{self.vocab}'
        os.environ['INFERENCE_SEQUENCE'] = f'{self.inference_sequence}'

    def run_command(self, command):
        print(f'Running command {command}')
        process = subprocess.Popen(command, executable='/bin/bash', shell=True)
        process.wait()
        if process.returncode != 0:
            raise Exception("Command failed: " + command)

    def check_and_run(self, step_method):
        with open(self.progress_file, 'r+') as f:
            if step_method.__name__ not in f.read().splitlines():
                step_method()
                f.write(f'{step_method.__name__}\n')

    def print_env_vars(self):
        print(f"COMMON_VOICE_DATASET_PATH: {self.common_voice_dataset_path}")
        print(f"DATASET_NAME: {self.dataset_name}")
        print(f"INFERENCE_SEQUENCE: {self.inference_sequence}")


    def generate_raw_audio_dataset(self):
        print("\n\n>>> generating raw audio dataset\n")
        env = {'VOCAB': f'{self.vocab}', 'INFERENCE_SEQUENCE': f'{self.inference_sequence}'}
        self.run_command(f'time python3 -m training.run.generate_raw_audio_dataset -i {self.common_voice_dataset_path} --positive-pct 100 --negative-pct {self.negative_pct} --overwrite true')

    def generate_alignment_for_positive_dataset(self):
        print(f"\n\n>>> generating alignment for the positive dataset using MFA: {self.pos_dataset_alignment}\n")
        os.makedirs(self.pos_dataset_alignment, exist_ok=True)
        os.chdir(self.mfa_folder)
        self.run_command(f"time yes n | ./bin/mfa_align --verbose --clean --num_jobs {NUM_CPUS} \"../{self.pos_dataset_path}/audio\" librispeech-lexicon.txt pretrained_models/english.zip \"../{self.pos_dataset_alignment}\"")
        os.chdir('..')

    def attach_alignment_positive_dataset(self):
        print("\n\n>>> attaching the MFA alignment to the positive dataset\n")
        self.run_command(f"time python3 -m training.run.attach_alignment --input-raw-audio-dataset \"{self.pos_dataset_path}\" --token-type word --alignment-type mfa --alignments-path \"{self.pos_dataset_alignment}\"")

    def attach_alignment_negative_dataset(self):
        if not self.skip_neg_dataset:
            print("\n\n>>> attaching mock alignment to the negative dataset\n")
            self.run_command(f"time python3 -m training.run.attach_alignment --alignment-type stub --input-raw-audio-dataset \"{self.neg_dataset_path}\" --token-type word")

    def stitch_vocab_samples(self):
        print("\n\n>>> stitching vocab samples to generate a dataset made up of stitched wakeword samples\n")
        self.run_command(f"time python3 -m training.run.stitch_vocab_samples --dataset-path \"{self.pos_dataset_path}\"")

    def print_ready_dataset(self):
        print(f"\n\n>>> Dataset is ready for {self.vocab}\n")

    def generate_env_file(self):
        vocab = self.vocab
        template = f'''
export WEIGHT_DECAY=0.00001
export NUM_EPOCHS=70
export LEARNING_RATE=0.01
export LR_DECAY=0.955
export BATCH_SIZE=16
export MAX_WINDOW_SIZE_SECONDS=0.5
export USE_NOISE_DATASET=False
export NUM_MELS=40
export INFERENCE_SEQUENCE="{self.inference_sequence}"
export VOCAB='{self.vocab}'
'''.strip()


        with open(self.env_file_path, 'w') as f:
            f.write(template + '\n')

    def generate_dataset(self):
        self.check_and_run(self.generate_env_file)
        self.check_and_run(self.print_env_vars)
        self.check_and_run(self.generate_raw_audio_dataset)
        self.check_and_run(self.generate_alignment_for_positive_dataset)
        self.check_and_run(self.attach_alignment_positive_dataset)
        self.check_and_run(self.attach_alignment_negative_dataset)
        self.check_and_run(self.stitch_vocab_samples)
        self.check_and_run(self.print_ready_dataset)
        self.check_and_run(self.train)

    def train(self):
        print("\n\n>>> attaching the MFA alignment to the positive dataset\n")
        self.run_command(f"time bash ./train_model.sh \"{self.env_file_path}\" res8 \"workspace/{self.dataset_name}\" \"{self.pos_dataset_path}\" \"{self.neg_dataset_path}\"")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name')
    args = parser.parse_args()
    dg = DatasetGenerator(args.dataset_name)
    dg.generate_dataset()


if __name__ == '__main__':
    main()
