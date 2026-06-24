import json
import os
import requests
import pandas as pd
from tqdm import tqdm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet

CSV_FILE = os.getenv(
    "CSV_FILE",
    "sample_data/sample_reddit_posts.csv"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:8b"
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

OUTPUT_DIR = "outputs"
CLASSIFIED_FILE = os.path.join(OUTPUT_DIR, "classified_posts.csv")
ROOT_CAUSE_FILE = os.path.join(OUTPUT_DIR, "root_cause_clusters.csv")
PHASE_FILE = os.path.join(OUTPUT_DIR, "phase_statistics.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "final_report.pdf")

CHECKPOINT_INTERVAL = 100

ROOT_CAUSE_MAP = {
    "Examination Stress": "Exam Pressure",
    "Exam Stress": "Exam Pressure",
    "Placement Stress": "Placement Rejections",
    "Comparison to Others": "Social Isolation",
}

PROMPT_TEMPLATE = """
You are a Mental Health Classification System.

Analyze the Reddit post.

Return ONLY valid JSON.

{{
  "tendency": [],
  "root_causes": [],
  "severity": "",
  "confidence": 0
}}

TENDENCY (choose one or more):

- Anxiety
- Depression
- Frustration
- Burnout
- Stress
- Hopelessness
- Fear of Failure
- Low Self-Esteem
- Loneliness
- Academic Pressure
- Positive
- Neutral

ROOT CAUSES (choose one or more):

- Exam Pressure
- Academic Pressure
- Paper Leaks
- Family Expectations
- Financial Stress
- Placement Rejections
- Career Uncertainty
- Skill Gap
- Poor Education System
- Relationship Issues
- Social Isolation
- Toxic Environment
- Health Issues
- Lack of Guidance
- Other

SEVERITY:
- Low
- Medium
- High
- Critical

Subreddit: {subreddit}
Matched Job: {matched_job}
Matched Keyword: {matched_keyword}

Title:
{title}

Body:
{selftext}

Return ONLY JSON.

"""

def analyze_post(row):

    prompt = PROMPT_TEMPLATE.format(
        subreddit=str(row.get("subreddit", "")),
        matched_job=str(row.get("matched_job", "")),
        matched_keyword=str(row.get("matched_keyword", "")),
        title=str(row.get("title", "")),
        selftext=str(row.get("selftext", ""))[:4000]
    )

    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        },
        timeout=300
    )

    raw = response.json()["response"]

    try:
        return json.loads(raw)
    except Exception:

        print("\nFAILED JSON:")
        print(raw[:1000])

        return {
            "tendency": ["Unknown"],
            "root_causes": ["Unknown"],
            "severity": "Low",
            "confidence": 0
        }

def load_data():
    """Load CSV file, resume from checkpoint if available."""
    df = pd.read_csv(CSV_FILE)

    results = []

    if os.path.exists(CLASSIFIED_FILE):
        existing = pd.read_csv(CLASSIFIED_FILE)
        results = existing.to_dict("records")

        processed_ids = set(existing["id"])

        df = df[~df["id"].isin(processed_ids)]

        print(
            f"Resuming from {len(results)} posts."
        )

    if "selftext" not in df.columns:
        raise ValueError("Column 'selftext' not found.")

    return df, results

def classify_posts(df, results):
    """Analyze posts and classify them."""
    for _, row in tqdm(df.iterrows(), total=len(df)):

        analysis = analyze_post(row)

        results.append({
            "id": row["id"],
            "created_iso": row.get("created_iso", ""),
            "subreddit": row.get("subreddit", ""),
            "matched_job": row.get("matched_job", ""),
            "matched_keyword": row.get("matched_keyword", ""),
            "phase": "College",
            "tendency": "; ".join(
                analysis.get("tendency", ["Unknown"])
            ),
            "root_causes": "; ".join(
                analysis.get("root_causes", ["Unknown"])
            ),
            "severity": analysis.get("severity", "Low"),
            "confidence": analysis.get("confidence", 0)
        })

        if len(results) > 0 and len(results) % CHECKPOINT_INTERVAL == 0:
            pd.DataFrame(results).to_csv(
                CLASSIFIED_FILE,
                index=False
            )
            print(f"Checkpoint saved at {len(results)} posts")

    return results

def save_classified_posts(classified_df):
    """Save classified posts to CSV."""
    classified_df.to_csv(
        CLASSIFIED_FILE,
        index=False
    )

    print("Saved classified_posts.csv")

def save_root_cause_clusters(classified_df):
    """Generate and save root cause analysis."""
    root_records = []

    for causes in classified_df["root_causes"]:

        for cause in str(causes).split(";"):

            cause = cause.strip()
            cause = ROOT_CAUSE_MAP.get(cause, cause)

            if cause:
                root_records.append(cause)

    root_summary = (
        pd.Series(root_records)
          .value_counts()
          .reset_index()
    )

    root_summary.columns = [
        "root_cause",
        "count"
    ]

    root_summary.to_csv(
        ROOT_CAUSE_FILE,
        index=False
    )

    print("Saved root_cause_clusters.csv")

def save_phase_statistics(classified_df):
    """Generate and save phase statistics."""
    phase_stats = (
        classified_df["tendency"]
        .str.split(";")
        .explode()
        .str.strip()
        .value_counts()
        .reset_index()
    )

    phase_stats.columns = [
        "tendency",
        "count"
    ]

    phase_stats["phase"] = "College"

    phase_stats.to_csv(
        PHASE_FILE,
        index=False
    )

    print("Saved phase_statistics.csv")

def generate_pdf_report(classified_df, root_summary):
    """Generate PDF report."""
    doc = SimpleDocTemplate(
        REPORT_FILE
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "Mental Health Analysis Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            f"Total Posts Analysed: {len(classified_df)}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Phase Distribution",
            styles["Heading2"]
        )
    )

    phase_counts = (
        classified_df["phase"]
        .value_counts()
    )

    for phase, count in phase_counts.items():
        story.append(
            Paragraph(
                f"{phase}: {count}",
                styles["Normal"]
            )
        )

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Top Root Causes",
            styles["Heading2"]
        )
    )

    for _, row in root_summary.head(20).iterrows():
        story.append(
            Paragraph(
                f"{row['root_cause']} : {row['count']}",
                styles["Normal"]
            )
        )

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Severity Distribution",
            styles["Heading2"]
        )
    )

    severity_counts = (
        classified_df["severity"]
        .value_counts()
    )

    for sev, count in severity_counts.items():
        story.append(
            Paragraph(
                f"{sev}: {count}",
                styles["Normal"]
            )
        )

    doc.build(story)

    print("Saved final_report.pdf")

def main():
    """Main entry point for the mental health analyzer."""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df, results = load_data()

    # Classify posts
    results = classify_posts(df, results)

    # Create dataframe from results
    classified_df = pd.DataFrame(results)

    # Save classified posts
    save_classified_posts(classified_df)

    # Generate root cause clusters
    root_records = []
    for causes in classified_df["root_causes"]:
        for cause in str(causes).split(";"):
            cause = cause.strip()
            cause = ROOT_CAUSE_MAP.get(cause, cause)
            if cause:
                root_records.append(cause)

    root_summary = (
        pd.Series(root_records)
          .value_counts()
          .reset_index()
    )
    root_summary.columns = ["root_cause", "count"]

    save_root_cause_clusters(classified_df)
    save_phase_statistics(classified_df)
    generate_pdf_report(classified_df, root_summary)

    print("DONE")


if __name__ == "__main__":
    main()
