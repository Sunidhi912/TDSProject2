import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import chardet

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AI_TOKEN")

def load_data(file_path):
    """Load CSV data with encoding detection."""
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']
    return pd.read_csv(file_path, encoding=encoding)

def analyze_data(df):
    """Perform basic data analysis."""
    numeric_df = df.select_dtypes(include=['number'])
    return {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict()
    }

def visualize_data(df):
    """Generate and save visualizations."""
    sns.set(style="whitegrid")
    for column in df.select_dtypes(include=['number']).columns:
        sns.histplot(df[column].dropna(), kde=True).set_title(f'Distribution of {column}')
        plt.savefig(f'{column}_distribution.png', dpi=80)  # Reduce dpi to speed up saving
        plt.close()

def generate_narrative(analysis):
    """Generate narrative using LLM."""
    headers = {'Authorization': f'Bearer {AIPROXY_TOKEN}', 'Content-Type': 'application/json'}
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": f"Provide a detailed analysis based on the following data summary: {analysis}"}]
    }
    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=20.0)  # Reduced timeout
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Narrative generation failed due to an error."

def main(file_path):
    df = load_data(file_path)
    analysis = analyze_data(df)
    visualize_data(df)
    with open('README.md', 'w') as f:
        f.write(generate_narrative(analysis))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
