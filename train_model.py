import os
import subprocess

class ModelTrainer:

    def __init__(self, env_file_path, model_type, workspace_path, dataset_paths):
        self.env_file_path = env_file_path
        self.model_type = model_type
        self.workspace_path = workspace_path
        self.dataset_paths = dataset_paths

    def run_command(self, command):
        process = subprocess.Popen(command, shell=True)
        process.wait()
        if process.returncode != 0:
            raise Exception("Command failed: " + command)

    def print_env_vars(self):
        print(f"ENV_FILE_PATH: {self.env_file_path}")
        print(f"MODEL_TYPE: {self.model_type}")
        print(f"WORKSPACE_PATH: {self.workspace_path}")
        print(f"DATASET_PATHS: {self.dataset_paths}")

    def source_env_file(self):
        # Load environment variables from the env file
        with open(self.env_file_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

    def train_model(self):
        print(">>> training a model for ${VOCAB}; model will be stored at ${WORKSPACE_PATH}")
        dataset_argument = "--dataset-paths " + " ".join(self.dataset_paths)
        self.run_command(f"time python -m training.run.train --model {self.model_type} --workspace \"{self.workspace_path}\" {dataset_argument} --use-stitched-datasets")

    def train(self):
        self.print_env_vars()
        self.source_env_file()
        self.train_model()

# Example usage:
if __name__ == '__main__':

    trainer = ModelTrainer("path/to/env", "res8", "path/to/workspace", ["path/to/dataset1", "path/to/dataset2"])
    trainer.train()
