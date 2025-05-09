import pandas as pd

def modify_csv(file_path, output_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Ensure necessary columns exist
    required_columns = {"reward", "cost", "nearness_to_berth", "total_reward"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")
    
    print("Current Data:")
    print(df.head())
    
    # Get user input for modifications
    positive_reward = float(input("Enter new positive reward value: "))
    negative_reward = float(input("Enter new negative reward value: "))
    cost_value = float(input("Enter new cost value (negative): "))
    
    # Update the reward column based on positivity or negativity
    df["reward"] = df["reward"].apply(lambda x: positive_reward if x > 0 else negative_reward)
    
    # Update the cost column with the single negative value
    df["cost"] = cost_value
    
    # Modify the nearness_to_berth column
    df["nearness_to_berth"] = df["nearness_to_berth"].apply(lambda x: float(input(f"Enter new value for nearness_to_berth ({x}): ") or x))
    
    # Update the total_reward column
    df["total_reward"] = df["reward"] + df["cost"] + df["nearness_to_berth"]
    
    # Save the modified data
    df.to_csv(output_path, index=False)
    print(f"Modified CSV saved to {output_path}")

# Example usage
input_file = "create_mal.csv"
output_file = "modified_create_mal.csv"
modify_csv(input_file, output_file)
