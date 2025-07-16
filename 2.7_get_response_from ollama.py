import pandas as pd
import requests
import time
import signal
import sys

def query_ollama_model(prompt, model="llama3.2:3b", host="http://localhost:11434"):
    """
    Sends a prompt to a locally running Ollama model and returns the response and duration.

    Parameters:
        prompt (str): The input prompt to send to the model.
        model (str): The name of the model to use (e.g., 'llama3.2:1b').
        host (str): The base URL of the local Ollama server.

    Returns:
        tuple: (response_text, duration_in_seconds)
    """
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        end_time = time.time()

        response.raise_for_status()
        response_text = response.json().get("response", "").strip()
        duration = end_time - start_time
        return response_text, duration
    except requests.RequestException as e:
        return f"Error communicating with Ollama: {e}", 0

def extract_java_and_tests(excel_path, num_rows=None):
    """
    Reads an Excel file, adds Context column if missing, and extracts data from rows without context.

    Parameters:
    - excel_path (str): Path to the Excel file.
    - num_rows (int or None): Maximum number of rows to process (only those without context). If None, process all.

    Returns:
    - Tuple: (DataFrame, List of dictionaries containing folder, document name, Java code, test code, and index)
    """
    try:
        # Load the Excel file
        df = pd.read_excel(excel_path)

        # Check if required columns exist
        required_columns = ['Folder', 'Document Name', 'Java Code', 'Test Code']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Excel file must contain the following columns: {required_columns}")

        # Add Context column if it doesn't exist
        if 'Context' not in df.columns:
            df['Context'] = pd.NA

        # Filter rows where Context is missing (NaN or empty)
        rows_to_process = df[df['Context'].isna()]
        
        # Limit the number of rows if specified
        if num_rows is not None:
            rows_to_process = rows_to_process.head(num_rows)
        
        # Extract relevant data
        extracted_data = []
        for idx in rows_to_process.index.tolist():
            row = df.loc[idx]
            entry = {
                'index': idx,
                'folder': row['Folder'],
                'document_name': row['Document Name'],
                'java_code': row['Java Code'],
                'test_code': row['Test Code']
            }
            extracted_data.append(entry)

        return df, extracted_data

    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame(), []

def save_dataframe(df, excel_path):
    """
    Saves the DataFrame to the specified Excel file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - excel_path (str): Path to the Excel file.
    """
    try:
        df.to_excel(excel_path, index=False)
        print(f"\nUpdated Excel file saved successfully: {excel_path}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

def signal_handler(sig, frame, df, excel_path, results):
    """
    Handles interrupt signals (e.g., Ctrl+C) to save current progress before exiting.

    Parameters:
    - sig: Signal number.
    - frame: Current stack frame.
    - df (pd.DataFrame): The DataFrame to save.
    - excel_path (str): Path to the Excel file.
    - results (list): List of processed results.
    """
    print("\nProcess interrupted. Saving current progress...")
    save_dataframe(df, excel_path)
    if results:
        output_df = pd.DataFrame(results)
        output_df.to_excel("processed_results_llama3_Apache.xlsx", index=False)
        print("\nResults saved to 'processed_results_llama3_Apache.xlsx'")
    sys.exit(0)

def process_excel_with_ollama(excel_path, num_rows=None, model="llama3.2:3b", host="http://localhost:11434"):
    """
    Processes each Java code and test code pair from an Excel file using the Ollama model, 
    starting from rows without context. Saves progress incrementally to handle interruptions.

    Parameters:
    - excel_path (str): Path to the Excel file.
    - num_rows (int or None): Number of rows to process (only those without context). If None, process all.
    - model (str): The name of the model to use for Ollama.
    - host (str): The base URL of the local Ollama server.

    Returns:
    - List of dictionaries containing folder, document name, Java code, test code, model response, and duration.
    """
    # Extract data from Excel
    df, data = extract_java_and_tests(excel_path, num_rows)
    results = []

    if not data:
        print("No rows to process or error reading the file.")
        return results

    # Register signal handler for interruptions
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, df, excel_path, results))

    # Process each entry with the Ollama model
    for item in data:
        try:
            # Skip entries with missing or invalid code
            if not isinstance(item['java_code'], str) or not isinstance(item['test_code'], str):
                print(f"Skipping entry {item['document_name']} due to invalid or missing code.")
                continue

            # Create prompt
            prompt = (
                "How following java code and test cases are related to each other \n\n"
                f"Java Code:\n{item['java_code']}\n\n"
                f"Test Cases:\n{item['test_code']}\n\n"
                "Describe each test case in such a way so that its like giving instruction how to design the test cases with the java code functionality "
                "Do not suggest any improvements for the java code and test pairs nor write any code for reference "
                "Provide summary only"
            )

            # Query the model
            response_text, duration = query_ollama_model(prompt, model, host)

            # Store the result
            result = {
                'folder': item['folder'],
                'document_name': item['document_name'],
                'java_code': item['java_code'],
                'test_code': item['test_code'],
                'model_response': response_text,
                'response_time': duration
            }
            results.append(result)

            # Update DataFrame with the context
            df.at[item['index'], 'Context'] = response_text

            # Save DataFrame after each processed entry
            save_dataframe(df, excel_path)

            # Print progress
            print(f"\n--- Entry: {item['document_name']} ---")
            print(f"📁 Folder: {item['folder']}")
            print(f"📄 Document: {item['document_name']}")
            print("Model Response:\n", response_text)
            print(f"⏱️ Response Time: {duration:.2f} seconds")

        except Exception as e:
            print(f"Error processing entry {item['document_name']}: {e}")
            # Save current progress before continuing
            save_dataframe(df, excel_path)
            continue

    # Save final results to a new Excel file
    if results:
        output_df = pd.DataFrame(results)
        output_df.to_excel("processed_results_llama3_Apache.xlsx", index=False)
        print("\nResults saved to 'processed_results_llama3_Apache.xlsx'")

    return results

if __name__ == "__main__":
    file_path = "filtered_output_file_Apache.xlsx"
    rows_to_process = None  # Set to None to process all rows without context

    # Process the Excel file with Ollama
    try:
        results = process_excel_with_ollama(file_path, num_rows=rows_to_process)
    except Exception as e:
        print(f"Fatal error: {e}")