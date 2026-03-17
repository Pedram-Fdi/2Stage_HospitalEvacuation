import os
import re
import pandas as pd

# Input and Output
input_dir = r"C:\PhD\Thesis\Papers\3rd\Submitted\1_EJOR\R1\ALNS_Initilization\Random_True\Temp"
output_file = os.path.join(input_dir, "Results_Temp_Integration.xlsx")

# Define columns for the summary DataFrame
columns = [
    "method",
    "instance",
    "Model",
    "NrScenario",
    "Sampling",
    "seed",
    "PHA",
    "ClusteringMethod",
    "Eval",
    "Total time",
]

# Empty DataFrame
summary_df = pd.DataFrame(columns=columns)


def parse_filename(file_name: str) -> dict:
    """
    Parse the given file name into its components.

    Expected pattern (example):
    ALNStrace_4_15_5_15_3_1_CRP_2Stage_ALNS_50_RQMC_42_Q_S_1_1_NS_NoC_Evaluation_False.txt
    """
    base_name, ext = os.path.splitext(file_name)
    if ext.lower() != ".txt":
        return {}

    parts = base_name.split("_")

    # Basic validation: we expect at least enough parts to cover the example pattern
    # If the structure is different, we will fall back to partial parsing where possible.
    result = {
        "method": None,
        "instance": None,
        "Model": None,
        "NrScenario": None,
        "Sampling": None,
        "seed": None,
        "PHA": None,
        "ClusteringMethod": None,
        "Eval": None,
    }

    try:
        # Method: first token
        if len(parts) >= 1:
            result["method"] = parts[0]

        # Instance: next 6 tokens joined by '_', if available
        # Example: 4_15_5_15_3_1 -> parts[1:7]
        if len(parts) >= 7:
            instance_tokens = parts[1:7]
            result["instance"] = "_".join(instance_tokens)

        # Model: from token 7 up to the scenario count (which should be numeric)
        # After instance (6 tokens), the structure in the example is:
        # [Model (possibly multiple tokens)], NrScenario, Sampling, seed,
        # [PHA (multiple tokens)], ClusteringMethod, Eval (remaining tokens)
        # We therefore look for the first numeric token after index 7 to mark NrScenario.
        model_start_idx = 7
        model_end_idx = None

        for idx in range(model_start_idx, len(parts)):
            if parts[idx].isdigit():
                model_end_idx = idx
                break

        if model_end_idx is not None and model_end_idx > model_start_idx:
            model_tokens = parts[model_start_idx:model_end_idx]
            result["Model"] = "_".join(model_tokens)

            # NrScenario
            result["NrScenario"] = parts[model_end_idx]

            # Sampling
            if model_end_idx + 1 < len(parts):
                result["Sampling"] = parts[model_end_idx + 1]

            # Seed
            if model_end_idx + 2 < len(parts):
                result["seed"] = parts[model_end_idx + 2]

            # The remaining tokens are PHA, ClusteringMethod, and Eval
            remaining = parts[model_end_idx + 3 :]

            if remaining:
                # Clustering method is assumed to be the last single token before Eval block
                # Eval block is assumed to start with the word 'Evaluation' if present.
                eval_start = None
                for i, token in enumerate(remaining):
                    if token.lower().startswith("evaluation"):
                        eval_start = i
                        break

                if eval_start is not None:
                    pha_tokens = remaining[: max(0, eval_start - 1)]
                    clustering_token = (
                        remaining[eval_start - 1] if eval_start - 1 >= 0 else None
                    )
                    eval_tokens = remaining[eval_start:]
                else:
                    # If we cannot detect 'Evaluation', we assume:
                    # all but last token -> PHA, last token -> ClusteringMethod, Eval = None
                    pha_tokens = remaining[:-1] if len(remaining) > 1 else []
                    clustering_token = remaining[-1] if remaining else None
                    eval_tokens = []

                if pha_tokens:
                    result["PHA"] = "_".join(pha_tokens)
                if clustering_token:
                    result["ClusteringMethod"] = clustering_token
                if eval_tokens:
                    result["Eval"] = "_".join(eval_tokens)

    except Exception:
        # Any failure in parsing should not crash the program; we simply return what we have.
        pass

    return result


def extract_total_time_from_text(text: str) -> float | None:
    """
    Extract total time from the given text.

    Priority:
    1) 'Total time: XXX seconds'
    2) 'elapsed_time: XXX seconds'
    """
    # Look for 'Total time: XXX seconds'
    total_time_pattern = re.compile(r"Total time:\s*([0-9]*\.?[0-9]+)\s*seconds", re.IGNORECASE)
    match = total_time_pattern.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    # Fallback: look for 'elapsed_time: XXX seconds'
    elapsed_pattern = re.compile(r"elapsed_time:\s*([0-9]*\.?[0-9]+)\s*seconds", re.IGNORECASE)
    match = elapsed_pattern.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    return None


# Loop through text files
for file_name in os.listdir(input_dir):
    if not file_name.lower().endswith(".txt"):
        continue

    file_path = os.path.join(input_dir, file_name)
    print(f"\nProcessing file: {file_name}")

    try:
        # Parse the file name into its components
        parsed_data = parse_filename(file_name)

        # Read the file content; for robustness with very large files,
        # read in text mode and only keep the tail (last ~50 KB).
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        total_time_value = extract_total_time_from_text(content)

        # Build the row with all expected columns
        row_data = {
            "method": parsed_data.get("method"),
            "instance": parsed_data.get("instance"),
            "Model": parsed_data.get("Model"),
            "NrScenario": parsed_data.get("NrScenario"),
            "Sampling": parsed_data.get("Sampling"),
            "seed": parsed_data.get("seed"),
            "PHA": parsed_data.get("PHA"),
            "ClusteringMethod": parsed_data.get("ClusteringMethod"),
            "Eval": parsed_data.get("Eval"),
            "Total time": total_time_value,
        }

        row_df = pd.DataFrame([row_data], columns=columns)
        summary_df = pd.concat([summary_df, row_df], ignore_index=True)
        print("Row added.")

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Export final DataFrame to Excel
summary_df.to_excel(output_file, index=False)
print("\nAll done! Results written to:", output_file)

