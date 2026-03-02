"""
generate_predictions.py — Generates prediction CSV for all 9 test queries.
Output: ../data/predictions.csv with columns: Query, Assessment_url
"""
import csv
from pathlib import Path

from recommender import recommend

# All 9 test queries hardcoded from Gen_AI_Dataset.xlsx Test-Set sheet
TEST_QUERIES = [
    "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.",

    (
        "Join a community that is shaping the future of work! "
        "SHL, People Science. People Answers. "
        "Are you an AI enthusiast with visionary thinking to conceptualize AI-based products? "
        "We are seeking a Research Engineer to join our team to deliver robust AI/ML models. "
        "Develop and experiment with machine learning models like NLP, computer vision etc. "
        "Prototype and fine-tune generative AI models. Implement emerging LLM technologies. "
        "Proficiency in Python and ML frameworks such as TensorFlow, PyTorch, & OpenAI APIs. "
        "Can you recommend some assessment that can help me screen applications. "
        "Time limit is less than 30 minutes"
    ),

    "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins.",

    (
        "Presales Specialist role - Building custom demos, Responding to RFPs, "
        "Maintaining knowledge of SHL commercial offerings. "
        "Required: 2-4 years in Presales or commercial operations in technology/SaaS. "
        "Strong writing, editing, and presentation skills. "
        "Experience with design tools (PowerPoint, Synthesia, Adobe, Canva). "
        "Strong communication and collaboration skills. "
        "I want them to give a test which is at least 30 mins long"
    ),

    "I am new looking for new graduates in my sales team, suggest a 30 min long assessment",

    (
        "For Marketing - Content Writer Position. Department: Marketing. Location: Gurugram. "
        "ShopClues.com is India's leading e-commerce platform. "
        "The role involves discussing campaign core messages, brainstorming visual and copy ideas, "
        "visualizing communication approaches for Website, Email, Social platforms, "
        "generating brand stories, overseeing production, and publishing Push Notifications "
        "and Product Descriptions."
    ),

    "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",

    "Here is a JD text, can you recommend some assessment that can help me screen applications. I am hiring for an analyst and wants applications to screen using Cognitive and personality tests",

    "I want to hire new graduates for a sales role in my company, the budget is for about an hour for each test. Give me some options",
]


def generate_predictions():
    output_path = Path(__file__).parent.parent / "data" / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Query", "Assessment_url"])

        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n[{i}/{len(TEST_QUERIES)}] Processing: {query[:70]}...")
            try:
                results = recommend(query, top_k=10)
                for r in results:
                    writer.writerow([query, r["url"]])
                    total_rows += 1
                print(f"  → {len(results)} recommendations written")
            except Exception as e:
                print(f"  ❌ Error: {e}")

    print(f"\n✅ Predictions saved to {output_path}")
    print(f"   Total rows (excluding header): {total_rows}")
    return str(output_path)


if __name__ == "__main__":
    generate_predictions()
