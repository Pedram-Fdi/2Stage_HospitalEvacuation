import os
import pandas as pd

# Input and Output
input_dir = r"C:\PhD\Thesis\Papers\3rd\Code\Results\2nd\Test"
output_file = r"C:\PhD\Thesis\Papers\3rd\Code\Results\2nd\Results.xlsx"

# Define columns for the summary DataFrame
columns = [
    'Instance', 'Model', 'Solver', 'ScenarioGeneration', 
    'NrScenario', 'ScenarioSeed', 'PHAObj', 'PHAPenalty', 
    'ALNSRL', 'ALNSRL_DeepQ', 'RLSelectionMethod', 'BBC_Accelerator', 
    'ClusteringMethod', 'All Scenario', 'NrEvaluation', 'Policy Generation', 'Time Horizon', 
    'GRB Cost', 'PHA Cost',
    'Mean', 'On-Time Transfer', 'On-Time Evacuation', 'Not Evacuated'
]

# Empty DataFrame
summary_df = pd.DataFrame(columns=columns)

# Loop through Excel files
for file_name in os.listdir(input_dir):
    if file_name.endswith(".xlsx"):
        file_path = os.path.join(input_dir, file_name)
        print(f"\nProcessing file: {file_name}")
        try:
            # Read Generic Information normally (with headers)
            generic_df = pd.read_excel(file_path, sheet_name="Generic Information",  header=None, engine="openpyxl")
            generic_data = generic_df.iloc[1, 0:17].tolist()

            # Read InSample with header=None to keep both rows
            insample_df = pd.read_excel(file_path, sheet_name="InSample", header=None, engine="openpyxl")
            grb_cost = insample_df.iloc[1, 0]
            pha_cost = insample_df.iloc[1, 5]

            # Read OutOfSample with header=None
            outsample_df = pd.read_excel(file_path, sheet_name="OutOfSample", header=None, engine="openpyxl")
            mean = outsample_df.iloc[1, 0]
            on_time_transfer = outsample_df.iloc[1, 24]
            on_time_evacuation = outsample_df.iloc[1, 25]
            not_evacuated = outsample_df.iloc[1, 26]

            # Combine all into a dictionary row
            row_data = generic_data + [grb_cost, pha_cost, mean, on_time_transfer, on_time_evacuation, not_evacuated]
            row_df = pd.DataFrame([row_data], columns=columns)

            # Append to summary
            summary_df = pd.concat([summary_df, row_df], ignore_index=True)
            print("Row added.")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Export final DataFrame to Excel
summary_df.to_excel(output_file, index=False)
print("\nAll done! Results written to:", output_file)
