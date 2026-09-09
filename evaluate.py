"""
Evaluation Harness for the Kivi Phonetic Memory System.

Reads test cases from `seed_data.json`, initializes an isolated database,
runs setup observations, executes the intervention pipeline via the LLM,
calculates target metrics, and generates a formatted Markdown report.
"""
import os
import json
import time
import asyncio
import numpy as np
from pathlib import Path

from app import db, learning, intervention

SEED_FILE = Path(__file__).resolve().parent / "seed_data.json"
REPORT_FILE = Path(__file__).resolve().parent / "evaluation_report.md"

async def run_evaluation():
    print(f"Loading test cases from {SEED_FILE}...")
    with open(SEED_FILE, "r") as f:
        cases = json.load(f)
        
    metrics = {
        "true_positives": 0,
        "false_positives": 0,  # FIR
        "false_negatives": 0,  # Missed interventions
        "true_negatives": 0,
        "latencies": [],
        "total_cost_tokens": 0,
        "total_cases": len(cases),
        "db_rows_added": 0
    }
    
    results_markdown = []

    for idx, case in enumerate(cases):
        print(f"Running Case {idx+1}/{len(cases)}: {case['category']}")
        
        # 1. Setup isolated DB for each case to prevent state bleed
        db_path = f"test_eval_case_{idx}.db"
        db.init_db(db_path, reset=True)
        conn = db.get_connection(db_path)
        
        # 2. Seed data
        for expl in case.get("setup_explicit", []):
            learning.ingest_explicit_correction(
                conn, expl["canonical"], expl["context"], expl.get("asr_token")
            )
        
        for impl in case.get("setup_implicit", []):
            learning.ingest_observation(conn, impl["asr"], impl["fmt"])
            
        conn.commit()
        
        # Count DB size before intervention
        start_rows = conn.execute("SELECT count(*) FROM user_dictionary").fetchone()[0]
        
        # 3. Run Pipeline
        start_time = time.time()
        final_output = await intervention.process_transcript(conn, case["test_asr"])
        latency = (time.time() - start_time) * 1000
        
        # 4. Measure DB Growth and Cost
        metrics["db_rows_added"] += start_rows
        
        logs = conn.execute("SELECT * FROM intervention_logs ORDER BY id DESC LIMIT 10").fetchall()
        for log in logs:
            metrics["total_cost_tokens"] += log["token_cost"]
            
        conn.close()
        
        # 5. Determine Correctness
        expected = case["expected_output"]
        actual = final_output
        is_correct = (expected.lower() == actual.lower())
        
        # Figure out confusion matrix theoretically based on if a change was expected
        change_expected = (case["test_asr"].lower() != expected.lower())
        change_made = (case["test_asr"].lower() != actual.lower())
        
        if change_expected and change_made and is_correct:
            metrics["true_positives"] += 1
            status = "✅ True Positive"
        elif not change_expected and not change_made:
            metrics["true_negatives"] += 1
            status = "✅ True Negative"
        elif change_made and not change_expected:
            metrics["false_positives"] += 1
            status = "❌ False Positive (Hallucination/FIR)"
        elif change_expected and not change_made:
            metrics["false_negatives"] += 1
            status = "❌ False Negative (Missed)"
        else:
            status = "❌ Incorrect Modification"
            
        metrics["latencies"].append(latency)
        
        # 6. Formatting
        results_markdown.append(f"### {case['category']} ({status})\n")
        results_markdown.append(f"- **Description**: {case['description']}\n")
        results_markdown.append(f"- **Input ASR**: `{case['test_asr']}`\n")
        results_markdown.append(f"- **Expected**: `{expected}`\n")
        results_markdown.append(f"- **Actual**: `{actual}`\n")
        results_markdown.append(f"- **Latency**: {latency:.2f}ms\n\n")
        
        # Cleanup
        try:
            os.remove(db_path)
            os.remove(db_path + "-wal")
            os.remove(db_path + "-shm")
        except:
            pass

    # Calculate final metrics
    tp = metrics["true_positives"]
    fp = metrics["false_positives"]
    fn = metrics["false_negatives"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    fir = fp / metrics["total_cases"]
    
    p50 = np.percentile(metrics["latencies"], 50)
    p95 = np.percentile(metrics["latencies"], 95)
    
    # Generate Report
    report = f"""# Kivi Phonetic Memory System - Evaluation Report

## Summary Metrics
- **Intervention Precision**: {precision:.1%}
- **Intervention Recall**: {recall:.1%}
- **False Intervention Rate (FIR)**: {fir:.1%}
- **Latency (P50)**: {p50:.2f} ms
- **Latency (P95)**: {p95:.2f} ms
- **Total Token Cost**: {metrics["total_cost_tokens"]} tokens
- **Dictionary Growth (Test Total)**: {metrics["db_rows_added"]} active/pending entries

---

## Detailed Test Cases

{''.join(results_markdown)}
"""
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"\nEvaluation Complete! Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set. The evaluation will run with the fallback dummy key and will likely fail the LLM judgments.")
    asyncio.run(run_evaluation())
