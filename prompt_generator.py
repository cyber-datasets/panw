import os
import yaml
import json
import time
import shutil
from pydantic import BaseModel
from ollama import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access configuration from environment variables with defaults
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3')
YAML_DIR = os.getenv('YAML_DIR', './xql_queries')
OUTPUT_FILE = os.getenv('OUTPUT_FILE', 'dataset.json')
PROCESSED_DIR = os.getenv('PROCESSED_DIR', './processed_xql_queries')

# Print key info for verification
print(f"Connecting to Ollama at {OLLAMA_HOST} with model {OLLAMA_MODEL}")
print(f"YAML directory: {os.path.abspath(YAML_DIR)}")
print(f"Output file path: {os.path.abspath(OUTPUT_FILE)}")
print(f"Processed directory: {os.path.abspath(PROCESSED_DIR)}")

# Ensure processed directory exists
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)
    print(f"Created processed directory: {PROCESSED_DIR}")

class QueryDetails(BaseModel):
    """Pydantic model to validate and structure XQL query details from YAML."""
    categories: list[str] = []
    description: str = ''
    name: str = ''
    sources: list[str] = []
    xql: str = ''

class DatasetGenerator:
    """Agent to process YAML files, generate prompts, and create a dataset in ShareGPT format incrementally."""
    
    def __init__(self, yaml_dir: str, output_file: str, processed_dir: str, ollama_host: str, ollama_model: str):
        """Initialize with directories, output file, and Ollama config."""
        self.yaml_dir = yaml_dir
        self.output_file = output_file
        self.processed_dir = processed_dir
        self.ollama_client = Client(host=ollama_host)
        self.ollama_model = ollama_model
        self.dataset_entries = []  # List to hold dataset entries incrementally

    def read_yaml(self, file_path: str) -> QueryDetails:
        """Read and parse a YAML file into a QueryDetails object."""
        print(f"Reading YAML file: {file_path}")
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return QueryDetails(**data)

    def generate_prompt(self, query_details: QueryDetails) -> str:
        """Generate a user prompt using Ollama, ensuring curly brackets."""
        print(f"Generating prompt for query: {query_details.name}")
        prompt = (
            "Given the following Cortex XQL query details:\n"
            f"- Categories: {query_details.categories}\n"
            f"- Description: {query_details.description}\n"
            f"- Name: {query_details.name}\n"
            f"- Sources: {query_details.sources}\n"
            f"- XQL: {query_details.xql}\n\n"
            "Create a concise, natural-sounding user prompt in the user's voice (e.g., 'I want to...') within curly brackets {}."
        )
        max_attempts = 5
        for attempt in range(max_attempts):
            response = self.ollama_client.generate(model=self.ollama_model, prompt=prompt)
            response_text = response['response'].strip()
            if response_text.startswith('{') and response_text.endswith('}'):
                print(f"Prompt generated: {response_text[1:-1]}")
                return response_text[1:-1]
            else:
                print(f"Attempt {attempt + 1}/{max_attempts}: No brackets, retrying...")
                if attempt == max_attempts - 1:
                    print("Max attempts reached, using raw response.")
                    return response_text if response_text else ""
                time.sleep(1)
        return ""

    def clean_xql(self, xql: str) -> str:
        """Clean the XQL query by removing comments and empty lines."""
        print(f"Cleaning XQL query:\n{xql}")
        if not xql or xql.strip() == "":
            print("XQL query is empty.")
            return ""
        lines = xql.split('\n')
        cleaned_lines = [line.split('//')[0].strip() for line in lines if line.strip()]
        cleaned_xql = '\n'.join(cleaned_lines)
        print(f"Cleaned XQL query:\n{cleaned_xql}")
        return cleaned_xql

    def move_processed_file(self, source_path: str):
        """Move a processed YAML file to the processed directory."""
        filename = os.path.basename(source_path)
        dest_path = os.path.join(self.processed_dir, filename)
        try:
            shutil.move(source_path, dest_path)
            print(f"Moved {filename} to {dest_path}")
        except Exception as e:
            print(f"Failed to move {filename}: {e}")

    def save_dataset(self):
        """Save the current dataset entries to the JSON file."""
        if not self.dataset_entries:
            print("No entries to save.")
            return
        try:
            print(f"Updating dataset file: {self.output_file} with {len(self.dataset_entries)} entries")
            with open(self.output_file, 'w') as f:
                json.dump(self.dataset_entries, f, indent=2)
            print(f"Dataset updated successfully with {len(self.dataset_entries)} entries")
        except Exception as e:
            print(f"Failed to update dataset: {e}")

    def process_files(self):
        """Process YAML files, generate entries, save incrementally, and move files."""
        yaml_files = [f for f in os.listdir(self.yaml_dir) if f.endswith(('.yaml', '.yml'))]
        total_files = len(yaml_files)
        print(f"Found {total_files} YAML files to process in {self.yaml_dir}")
        
        if total_files == 0:
            print(f"No YAML files found in {self.yaml_dir}")
            return

        for i, filename in enumerate(yaml_files, start=1):
            file_path = os.path.join(self.yaml_dir, filename)
            print(f"\nProcessing {i}/{total_files}: {filename}")
            try:
                query_details = self.read_yaml(file_path)
                generated_prompt = self.generate_prompt(query_details)
                cleaned_xql = self.clean_xql(query_details.xql)
                if not cleaned_xql:
                    print(f"Skipping {filename}: Empty XQL query.")
                    self.move_processed_file(file_path)
                    continue
                # Create ShareGPT-structured entry
                conversation = [
                    {"from": "human", "value": generated_prompt},
                    {"from": "gpt", "value": cleaned_xql}
                ]
                entry = {"conversations": conversation}
                self.dataset_entries.append(entry)
                self.save_dataset()  # Save after each entry
                self.move_processed_file(file_path)
                print(f"Processed {filename} and updated dataset with entry {len(self.dataset_entries)}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    def run(self):
        """Execute the dataset creation process."""
        start_time = time.time()
        print("\nStarting dataset generation...")
        self.process_files()
        print(f"\nCompleted in {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    generator = DatasetGenerator(
        yaml_dir=YAML_DIR,
        output_file=OUTPUT_FILE,
        processed_dir=PROCESSED_DIR,
        ollama_host=OLLAMA_HOST,
        ollama_model=OLLAMA_MODEL
    )
    generator.run()