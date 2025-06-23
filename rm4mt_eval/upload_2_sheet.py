import os
import json
import pandas as pd
import re
import gspread
from gspread_dataframe import set_with_dataframe
import argparse


def parse_budget_from_dirname(dirname):
    """Extracts numbers from a directory name (e.g., budget_100)"""
    match = re.search(r'\d+$', dirname)
    if match:
        return int(match.group())
    return None


def process_data(root_dir):
    """Scans directories and extracts data from jsonl files"""
    all_data = []
    print(f"Starting scan from root directory: {root_dir}")
    if not os.path.isdir(root_dir):
        print(f"Error: Directory '{root_dir}' not found. Please check your ROOT_DIR configuration.")
        return None

    for task_name in os.listdir(root_dir):
        task_path = os.path.join(root_dir, task_name)
        if not os.path.isdir(task_path):
            continue
        print(f"  Processing task: {task_name}")

        for model_name in os.listdir(task_path):
            model_path = os.path.join(task_path, model_name)
            if not os.path.isdir(model_path):
                continue

            for budget_dir_name in os.listdir(model_path):
                budget_path = os.path.join(model_path, budget_dir_name)
                if not os.path.isdir(budget_path):
                    continue

                budget_value = parse_budget_from_dirname(budget_dir_name)
                if budget_value is None:
                    continue

                for jsonl_file in os.listdir(budget_path):
                    if not jsonl_file.endswith('.jsonl'):
                        continue

                    lang_pair = jsonl_file.replace('.jsonl', '')
                    file_path = os.path.join(budget_path, jsonl_file)

                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                record = json.loads(line)
                                data_entry = {
                                    'task': task_name,
                                    'model': model_name,
                                    'budget': budget_value,
                                    'lang_pair': lang_pair,
                                }

                                # Only include available metric keys
                                for key in ['comet_score', 'comet_kiwi_score', 'gemba_score', 'gemba_noref_score']:
                                    if key in record:
                                        try:
                                            data_entry[key] = float(record[key])
                                        except (ValueError, TypeError):
                                            print(f"Warning: Non-numeric value for {key} in file {file_path}. Skipping value.")

                                all_data.append(data_entry)
                            except json.JSONDecodeError:
                                print(f"Warning: Found invalid JSON line in file {file_path}.")
    if not all_data:
        print("Warning: No data collected. Please check directory structure and file contents.")
        return None

    print(f"Data scan complete, {len(all_data)} records collected.")
    return pd.DataFrame(all_data)


def prepare_data_for_upload(df):
    """Processes the raw DataFrame and returns a dictionary of DataFrames ready for upload."""
    if df is None or df.empty:
        print("No data to prepare.")
        return {}

    available_metrics = set(df.columns)
    if {'comet_score', 'comet_kiwi_score'}.issubset(available_metrics):
        metrics = ['comet_score', 'comet_kiwi_score']
    elif {'gemba_score', 'gemba_noref_score'}.issubset(available_metrics):
        metrics = ['gemba_score', 'gemba_noref_score']
    else:
        print("No known metric columns found (e.g., comet_score or gemba_score).")
        return {}

    before_drop = len(df)
    df.dropna(subset=metrics, how='all', inplace=True)
    print(f"Dropped {before_drop - len(df)} rows with all metrics missing.")
    if df.empty:
        print("Warning: All records are missing metric scores, cannot generate reports.")
        return {}

    sheets_data = {}
    tasks = df['task'].unique()

    for task in tasks:
        for metric in metrics:
            sheet_name = f"{task}_{metric}"
            if len(sheet_name) > 100:
                sheet_name = sheet_name[:100]

            task_df = df[df['task'] == task].copy()
            agg_df = task_df.groupby(['model', 'lang_pair', 'budget'])[metric].mean().reset_index()
            pivot_df = agg_df.pivot_table(index=['model', 'lang_pair'], columns='budget', values=metric)

            if pivot_df.empty:
                continue

            avg_df = pivot_df.groupby(level='model').mean()
            avg_df['lang_pair'] = 'Avg'
            avg_df.set_index('lang_pair', append=True, inplace=True)

            final_df = pd.concat([pivot_df, avg_df])
            final_df.sort_index(level='model', sort_remaining=True, inplace=True)
            final_df = final_df.reindex(sorted(final_df.columns), axis=1)

            sheets_data[sheet_name] = final_df.reset_index()

    return sheets_data


def upload_to_google_sheets(sheets_data, sheet_url, credentials_file):
    """Uploads a dictionary of DataFrames to Google Sheets, each DataFrame in its own worksheet."""
    if not sheets_data:
        print("No data to upload. Terminating.")
        return

    print(f"Authenticating with Google using '{credentials_file}'...")
    try:
        gc = gspread.service_account(filename=credentials_file)
        spreadsheet = gc.open_by_url(sheet_url)
        print(f"Successfully connected to Google Sheet at URL: '{sheet_url}'.")
    except FileNotFoundError:
        print(f"Error: Credentials file '{credentials_file}' not found.")
        print("Please ensure the file is in the same directory as the script.")
        return
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"Error: Google Sheet at URL '{sheet_url}' not found.")
        print("Please check if the URL is correct and if you have shared it with the service account's email.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during connection: {e}")
        return

    for name, df in sheets_data.items():
        print(f"  Processing worksheet: {name}...")
        try:
            worksheet = spreadsheet.worksheet(name)
            print(f"    Found worksheet '{name}'. Clearing and updating...")
            worksheet.clear()
        except gspread.exceptions.WorksheetNotFound:
            print(f"    Worksheet '{name}' not found. Creating a new one...")
            worksheet = spreadsheet.add_worksheet(title=name, rows=1, cols=1)

        set_with_dataframe(worksheet, df, include_index=False, allow_formulas=False)
        print(f"    Data successfully uploaded to worksheet '{name}'.")

    print("\nFinished uploading to Google Sheets!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload evaluation results to Google Sheets.')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Path to the root directory containing evaluation results')
    parser.add_argument('--credentials_file', type=str, required=True,
                        help='Path to the Google service account JSON credentials file')
    parser.add_argument('--sheet_url', type=str, required=True,
                        help='The URL of the Google Sheet to upload data to')

    args = parser.parse_args()

    raw_data_df = process_data(args.root_dir)
    sheets_to_upload = prepare_data_for_upload(raw_data_df)
    upload_to_google_sheets(sheets_to_upload, args.sheet_url, args.credentials_file)