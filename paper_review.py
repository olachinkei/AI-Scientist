import openai
from ai_scientist.perform_review import load_paper, perform_review

client = openai.OpenAI()
model = "gpt-4o-2024-05-13"

# Load paper from PDF file (raw text)
paper_txt = load_paper("report.pdf")

# Get the review dictionary
review = perform_review(
    paper_txt,
    model,
    client,
    num_reflections=5,
    num_fs_examples=1,
    num_reviews_ensemble=5,
    temperature=0.1,
)

# Inspect review results
print("Overall",review["Overall"])    # Overall score (1-10)
print("Decision",review["Decision"])   # 'Accept' or 'Reject'
print("Weakness",review["Weaknesses"]) # List of weaknesses (strings)