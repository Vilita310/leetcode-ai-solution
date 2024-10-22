import pandas as pd
import json
import openai
import os
import pandas as pd
import json
from openai import OpenAI
import concurrent.futures
import time
from collections import defaultdict


def format_markdown(message):
    content = message.content
    
    # Replace escaped newlines with actual newlines
    content = content.replace('\\n', '\n')
    
    # Ensure proper Markdown formatting
    formatted_content = content.replace('```python', '\n```python').replace('```', '\n```')
    
    return formatted_content


def solve_problem(problem):
    prompt = f"""
        Write a Python code to solve the problem below, 
        please give a detailed explanation and follow the LeetCode solution format to ensure the answer can be run directly on the LeetCode website.
        
        Problem:
        {problem}
        """
    client = OpenAI(api_key="sk-h0x5BeGFWbIlTcG8FCywbya7D4SFaicPOamxICp-1UT3BlbkFJKpTGCkyXljtreuQXu8TBTnnk93n-zb5NdkES1f0XUA")
    
    response = client.chat.completions.create(
      model="gpt-4-mini",
      messages=[
        {"role": "system", "content": "You are an algorithm master."},
        {"role": "user", "content": prompt},
      ]
    )

    return response.choices[0].message


def main():
    leetcode_dataset_path = "data/leetcode_dataset.csv"
    df = pd.read_csv(leetcode_dataset_path)
    df.head()

    ids = df["id"].tolist()
    problems = df["description"].tolist()
    titles = df["title"].tolist()
    topics = df["related_topics"].tolist()
    
    # sort by topics
    problems_with_topics = defaultdict(list)
    for id, problem, topic, title in zip(ids, problems, topics, titles):
        if type(topic) != str:
            problems_with_topics["unknown"].append((id, title, problem))
        else:
            for t in topic.split(","):
                problems_with_topics[t].append((id, title, problem))

    # start scaping
    if not os.path.exists("output"):
        os.makedirs("output")

    for topic, problems_set in problems_with_topics.items():
        ids, titles, problems = zip(*problems_set)
        cur_cnt = 0
        while cur_cnt < len(problems):
            print(f"Processing topic: {topic}, {cur_cnt} to {cur_cnt + 50}")
            start = cur_cnt
            end = cur_cnt + 50
            cur_ids = ids[start:end]
            cur_problems = problems[start: end]
            cur_titles = titles[start: end]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(solve_problem, cur_problems))

            markdown_results = [format_markdown(result) for result in results]
            markdown_text = ""
            for question_number, title, problem, result in zip(cur_ids, cur_titles, cur_problems, markdown_results):
                markdown_text += f"# {question_number}. {title}\n\n"
                markdown_text += f"### Problem Description \n{problem}\n\n"
                markdown_text += f"### Solution \n {result}\n\n"
            
            output_file = f"output/{topic}-{start}-{end}.md"
            with open(output_file, "w") as f:
                f.write(markdown_text)
            print(f"Problems {cur_cnt}-{cur_cnt + 50} saved to {output_file}")
            cur_cnt += 50
            time.sleep(10)

if __name__ == "__main__":
    main()
