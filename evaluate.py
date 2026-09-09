"""
Evaluation Harness for the Kivi Phonetic Memory System.

Reads test cases from `seed_data.json`, initializes an isolated database,
seeds the dictionary via explicit corrections and direct overrides,
runs the intervention pipeline via the LLM, calculates metrics,
and generates a formatted Markdown report.
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
        "false_positives": 0,
        "false_negatives": 0,
        "true_negatives": 0,
        "latencies": [],
        "total_cases": len(cases),
    }
    
    results_markdown = []

    for idx, case in enumerate(cases):
        print(f"Running Case {idx+1}/{len(cases)}: {case['id']} - {case['category']}")
        
        # 1. Setup isolated DB for each case to prevent state bleed
        db_path = f"test_eval_case_{idx}.db"
        db.init_db(db_path, reset=True)
        conn = db.get_connection(db_path)
        
        # 2. Seed phonetic dictionary via explicit corrections
        for expl in case.get("setup_explicit", []):
            learning.ingest_explicit_correction(
                conn, expl["canonical"], expl["context"], expl.get("asr_token")
            )
        
        # 3. Seed direct overrides
        for ovr in case.get("setup_overrides", []):
            conn.execute(
                "INSERT OR REPLACE INTO exact_overrides (asr_token, canonical_word) VALUES (?, ?)",
                (ovr["asr_token"], ovr["canonical_word"])
            )
            
        conn.commit()
        
        # 4. Run Pipeline
        start_time = time.time()
        final_output = await intervention.process_transcript(conn, case["test_asr"])
        latency = (time.time() - start_time) * 1000
        
        conn.close()
        
        # 5. Determine Correctness
        expected = case["expected_output"]
        actual = final_output
        is_correct = (expected.lower() == actual.lower())
        
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
            status = "❌ False Positive (Hallucination)"
        elif change_expected and not change_made:
            metrics["false_negatives"] += 1
            status = "❌ False Negative (Missed)"
        elif change_expected and change_made and not is_correct:
            metrics["false_positives"] += 1
            status = "⚠️ Incorrect Modification"
        else:
            status = "❓ Unknown"
            
        metrics["latencies"].append(latency)
        
        # 6. Format result
        results_markdown.append(f"### Case {idx+1}: {case['category']} ({status})\n")
        results_markdown.append(f"- **Description**: {case['description']}\n")
        results_markdown.append(f"- **Input ASR**: `{case['test_asr']}`\n")
        results_markdown.append(f"- **Expected**: `{expected}`\n")
        results_markdown.append(f"- **Actual**: `{actual}`\n")
        results_markdown.append(f"- **Match**: {'✅' if is_correct else '❌'}\n")
        results_markdown.append(f"- **Latency**: {latency:.2f}ms\n\n")
        
        # Cleanup temp DB files
        for ext in ["", "-wal", "-shm"]:
            try:
                os.remove(db_path + ext)
            except:
                pass
                
        # To avoid hitting TPM/RPM rate limits on free LLM tiers (like Groq), pause briefly
        if idx < len(cases) - 1:
            print("Waiting 2s to avoid API rate limits...")
            await asyncio.sleep(2)

    # Calculate final metrics
    tp = metrics["true_positives"]
    fp = metrics["false_positives"]
    fn = metrics["false_negatives"]
    tn = metrics["true_negatives"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / metrics["total_cases"] if metrics["total_cases"] > 0 else 0.0
    fir = fp / metrics["total_cases"] if metrics["total_cases"] > 0 else 0.0
    
    p50 = np.percentile(metrics["latencies"], 50)
    p95 = np.percentile(metrics["latencies"], 95)
    
    # Generate Report
    report = f"""# Kivi Phonetic Memory System - Evaluation Report

## Summary Metrics
| Metric | Value |
|--------|-------|
| **Accuracy** | {accuracy:.1%} |
| **Precision** | {precision:.1%} |
| **Recall** | {recall:.1%} |
| **F1 Score** | {f1:.1%} |
| **False Intervention Rate** | {fir:.1%} |
| **Latency (P50)** | {p50:.0f} ms |
| **Latency (P95)** | {p95:.0f} ms |
| **Total Test Cases** | {metrics["total_cases"]} |

### Confusion Matrix
|  | Predicted Positive | Predicted Negative |
|--|---|---|
| **Actual Positive** | {tp} (TP) | {fn} (FN) |
| **Actual Negative** | {fp} (FP) | {tn} (TN) |

---

## Detailed Test Cases

{''.join(results_markdown)}
"""
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"\n{'='*50}")
    print(f"Accuracy: {accuracy:.1%} | Precision: {precision:.1%} | Recall: {recall:.1%} | F1: {f1:.1%}")
    print(f"Latency P50: {p50:.0f}ms | P95: {p95:.0f}ms")
    print(f"Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY is not set. LLM judgments will fail.")
    print("=================================================================================")
    print("WARNING: This script runs multiple sequential LLM calls.")
    print("If you are on a free API tier (e.g., Groq free tier), this may hit RPM/TPM limits")
    print("or cause timeouts. We recommend testing manually via the Streamlit UI instead.")
    print("=================================================================================\n")
    asyncio.run(run_evaluation())
