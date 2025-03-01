import json
from typing import List, Dict

def load_dataset(file_path: str) -> List[Dict]:
    """
    Load the dataset from a JSON file and return it as a list of dictionaries.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        if not isinstance(dataset, list):
            raise ValueError("Dataset must be a JSON list of conversations.")
        return dataset
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{file_path}': {str(e)}")
        return []
    except Exception as e:
        print(f"Error: Unexpected error loading dataset: {str(e)}")
        return []

def validate_conversation(conversation: Dict, index: int) -> Dict[str, List[str]]:
    """
    Validate a single conversation entry in ShareGPT format.
    Returns a dictionary of errors and warnings.
    """
    issues = {"errors": [], "warnings": []}

    # Check for required 'conversations' key
    if "conversations" not in conversation:
        issues["errors"].append(f"Entry {index}: Missing 'conversations' key.")
        return issues

    conversations = conversation["conversations"]
    if not isinstance(conversations, list):
        issues["errors"].append(f"Entry {index}: 'conversations' must be a list.")
        return issues

    if len(conversations) == 0:
        issues["errors"].append(f"Entry {index}: 'conversations' list is empty.")
        return issues

    # Track roles for order validation
    role_pattern = ["human", "gpt"]
    current_role_index = 0

    for i, message in enumerate(conversations):
        # Ensure message is a dictionary
        if not isinstance(message, dict):
            issues["errors"].append(f"Entry {index}, Message {i}: Message must be a dictionary.")
            continue

        # Check required fields
        if "from" not in message:
            issues["errors"].append(f"Entry {index}, Message {i}: Missing 'from' field.")
        if "value" not in message:
            issues["errors"].append(f"Entry {index}, Message {i}: Missing 'value' field.")

        # Validate role
        role = message.get("from", "").lower()
        if role not in ["human", "gpt", "system"]:
            issues["warnings"].append(f"Entry {index}, Message {i}: Unknown role '{role}'.")
        else:
            if role in ["human", "gpt"]:
                expected_role = role_pattern[current_role_index % 2]
                if role != expected_role:
                    issues["warnings"].append(f"Entry {index}, Message {i}: Role '{role}' out of order.")
                current_role_index += 1

        # Validate value
        value = message.get("value", "").strip()
        if not value:
            issues["errors"].append(f"Entry {index}, Message {i}: 'value' is empty or whitespace.")

        # Additional checks
        if role == "human" and len(value.split()) < 2:
            issues["warnings"].append(f"Entry {index}, Message {i}: Human prompt '{value}' is too short.")
        if role == "gpt":
            if "datamodel" not in value.lower() and "preset" not in value.lower():
                issues["warnings"].append(f"Entry {index}, Message {i}: GPT response may not be a valid XQL query.")

    # Ensure both human and gpt messages exist
    human_count = sum(1 for msg in conversations if msg.get("from", "").lower() == "human")
    gpt_count = sum(1 for msg in conversations if msg.get("from", "").lower() == "gpt")
    if human_count == 0 or gpt_count == 0:
        issues["errors"].append(f"Entry {index}: Missing human or gpt messages.")

    return issues

def validate_dataset(file_path: str) -> List[Dict]:
    """
    Validate the dataset and return only conversations with no errors or warnings.
    """
    dataset = load_dataset(file_path)
    if not dataset:
        return []

    valid_entries = []
    error_count = 0
    warning_only_count = 0
    clean_count = 0

    for i, entry in enumerate(dataset, 1):
        issues = validate_conversation(entry, i)
        if issues["errors"]:
            error_count += 1
            print(f"\nEntry {i} - Errors ({len(issues['errors'])}):")
            for error in issues["errors"]:
                print(f"  - {error}")
        elif issues["warnings"]:
            warning_only_count += 1
            print(f"\nEntry {i} - Warnings ({len(issues['warnings'])}):")
            for warning in issues["warnings"]:
                print(f"  - {warning}")
        else:
            valid_entries.append(entry)
            clean_count += 1

    # Print summary
    total_entries = len(dataset)
    print("=" * 50)
    print("Validation Summary:")
    print(f"Total entries: {total_entries}")
    print(f"Entries with errors: {error_count}")
    print(f"Entries with warnings only: {warning_only_count}")
    print(f"Clean entries: {clean_count}")
    print(f"Removed entries: {error_count + warning_only_count}")
    print(f"Kept entries: {clean_count}")

    return valid_entries

def save_dataset(valid_entries: List[Dict], output_file: str) -> None:
    """
    Save the valid entries to a new JSON file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(valid_entries, f, indent=2)
        print(f"Clean dataset saved to '{output_file}' with {len(valid_entries)} entries.")
    except Exception as e:
        print(f"Error saving clean dataset: {str(e)}")

def main():
    """Run the validation and save the clean dataset."""
    file_path = "dataset.json"
    valid_entries = validate_dataset(file_path)
    if valid_entries:
        save_dataset(valid_entries, "clean_dataset.json")

if __name__ == "__main__":
    main()