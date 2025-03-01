import json
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
import argparse

# Define the state structure using TypedDict
class AgentState(TypedDict):
    input_file: str
    output_file: str
    system_message: Optional[str]
    dataset: Optional[list[dict]]

# Node to get the system message if not provided
def get_system_message(state: AgentState) -> AgentState:
    """
    Checks if a system message exists in the state. If not, prompts the user to input one.
    Updates the state with the system message.
    """
    if state.get("system_message") is None:
        print("No system message provided.")
        system_message = input("Please enter the system context message: ").strip()
        if not system_message:
            raise ValueError("System context message cannot be empty.")
        state["system_message"] = system_message
    return state

# Node to add the system context to the dataset
def add_system_context(state: AgentState) -> AgentState:
    """
    Loads the dataset from the input file, adds the system message to each conversation,
    and saves the updated dataset to the output file.
    """
    input_file = state["input_file"]
    output_file = state["output_file"]
    system_message = state["system_message"]
    
    try:
        # Load the dataset from the input file
        with open(input_file, 'r') as f:
            dataset = json.load(f)
        
        # Add the system message to each conversation
        for conversation in dataset:
            if "conversations" in conversation:
                conversation['conversations'].insert(0, {"from": "system", "value": system_message})
            else:
                print(f"Warning: Skipping entry missing 'conversations' key: {conversation}")
        
        # Save the updated dataset to the output file
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"System context added to {len(dataset)} conversations. Saved to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {input_file}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return state

# Conditional function to determine the next node
def should_prompt(state: AgentState) -> str:
    """
    Decides whether to prompt for a system message or proceed to adding the context.
    Returns the name of the next node.
    """
    if state.get("system_message") is None:
        return "get_system_message"
    return "add_system_context"

# Build the workflow graph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("init", lambda state: state)  # New init node as the starting point
workflow.add_node("get_system_message", get_system_message)
workflow.add_node("add_system_context", add_system_context)

# Set the entry point to "init"
workflow.set_entry_point("init")

# Define conditional edges from "init"
workflow.add_conditional_edges(
    "init",
    should_prompt,
    {"get_system_message": "get_system_message", "add_system_context": "add_system_context"}
)

# Define sequential edges
workflow.add_edge("get_system_message", "add_system_context")
workflow.add_edge("add_system_context", END)

# Compile the graph
graph = workflow.compile()

# Parse command-line arguments
def parse_args():
    """
    Parses command-line arguments for input file, output file, and optional system message.
    """
    parser = argparse.ArgumentParser(description="Add system context to ShareGPT dataset using LangGraph.")
    parser.add_argument("--input_file", default="dataset.json", help="Path to the input dataset JSON file.")
    parser.add_argument("--output_file", default="dataset_with_system.json", help="Path to the output JSON file.")
    parser.add_argument("--system_message", help="The system context message to add (optional).")
    return parser.parse_args()

# Main execution block
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Initialize the state
    initial_state = {
        "input_file": args.input_file,
        "output_file": args.output_file,
        "system_message": args.system_message if args.system_message else None,
        "dataset": None
    }
    
    # Run the graph with the initial state
    result = graph.invoke(initial_state)
    print("Process completed.")