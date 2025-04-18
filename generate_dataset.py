from langchain_nvidia import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

import csv

llm=ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.3",
              nvidia_api_key="",
              temperature=1)

data_gen_prompt=ChatPromptTemplate.from_template("""
Your goal is to generate a synthetic dataset to mimic user-generated content that would be found in messages or social media posts.
Given the moderation policy below, all of your examples need to fall under one of the categories: Scamming, Drugs, Selling, and Safe.
You are to provide 100 examples, each containing the following format:

Content,Expected
"Example message", "Category"

The 'Content' column will contain the synthetic data (a message), and the 'Expected' column will contain the moderation category. 
You must ensure the responses are short to medium in length. Each example should fall under one of the following categories: Scamming, Drugs, Selling, or Safe.

Moderation Policy:
{moderation_policy}
""")

# Open the file in read mode
with open('policy.txt', 'r') as file:
    # Read the contents of the file
    content = file.read()
    
    prompt = data_gen_prompt.format_prompt(moderation_policy=content)
    response = llm.invoke(prompt)

    # Assuming the response contains the dataset in the format "Content,Expected"
    dataset = response.content

    # Split the response into lines (each line should be a row of the dataset)
    lines = dataset.split('\n')

    # Open a CSV file for writing the dataset
    with open('synthetic_dataset.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Content", "Expected"])  # Write the header

        # Loop through the lines and write each row to the CSV
        for line in lines:
            # Skip empty lines or non-structured lines
            if not line.strip():
                continue
            
            # Ensure the output has the "Content,Expected" format (split by comma)
            columns = line.split(',')

            # If the line has exactly two columns, write it to the CSV
            if len(columns) == 2:
                content, expected = columns
                writer.writerow([content.strip(), expected.strip()])