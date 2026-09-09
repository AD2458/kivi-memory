# Kivi Phonetic Memory System - Evaluation Report

## Summary Metrics
- **Intervention Precision**: 100.0%
- **Intervention Recall**: 100.0%
- **False Intervention Rate (FIR)**: 0.0%
- **Latency (P50)**: 1111.14 ms
- **Latency (P95)**: 2583.58 ms
- **Total Token Cost**: 0 tokens
- **Dictionary Growth (Test Total)**: 7 active/pending entries

---

## Detailed Test Cases

### True Positive (✅ True Positive)
- **Description**: Correctly replacing a known phonetic misspelling.
- **Input ASR**: `ask aditya to review the PR`
- **Expected**: `ask Aaditya to review the PR`
- **Actual**: `ask Aaditya to review the PR`
- **Latency**: 2700.49ms

### True Negative (✅ True Negative)
- **Description**: Leaving a word alone despite phonetic match due to semantic mismatch.
- **Input ASR**: `I bought a fresh kiwi from the market`
- **Expected**: `I bought a fresh kiwi from the market`
- **Actual**: `I bought a fresh kiwi from the market`
- **Latency**: 1111.14ms

### Pending Memory (✅ True Negative)
- **Description**: Ignoring a phonetic match because evidence_count has not met the threshold.
- **Input ASR**: `is rakesh online`
- **Expected**: `is rakesh online`
- **Actual**: `is rakesh online`
- **Latency**: 0.00ms

### Conflicting Memories (✅ True Negative)
- **Description**: Disambiguating two similar phonetic matches based on semantic context snippets.
- **Input ASR**: `ask aditya about the database migration`
- **Expected**: `ask Aditya about the database migration`
- **Actual**: `ask Aditya about the database migration`
- **Latency**: 2310.79ms

### Novel Entities (✅ True Negative)
- **Description**: Refusing to hallucinate a correction for a completely unseen proper noun.
- **Input ASR**: `ask Nandini to review the doc`
- **Expected**: `ask Nandini to review the doc`
- **Actual**: `ask Nandini to review the doc`
- **Latency**: 0.00ms

### Symmetric Phonetic Fix (✅ True Positive)
- **Description**: Correctly applying the ee->i normalization when spoken is 'rajeev' and DB is 'Rajiv'.
- **Input ASR**: `did you ask rajeev about the database?`
- **Expected**: `did you ask Rajiv about the database?`
- **Actual**: `did you ask Rajiv about the database?`
- **Latency**: 1070.70ms

### Symmetric Phonetic Fix (✅ True Positive)
- **Description**: Correctly applying the ee->i normalization when spoken is 'rajiv' and DB is 'Rajeev'.
- **Input ASR**: `tell rajiv to restart the server`
- **Expected**: `tell Rajeev to restart the server`
- **Actual**: `tell Rajeev to restart the server`
- **Latency**: 1965.24ms


