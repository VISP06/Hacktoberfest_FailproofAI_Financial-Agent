# Token Usage and Cost Analysis Report

## Executive Summary
- **Challenge:** HackerRank Orchestrate — Buy or Wait?
- **Primary Model:** `gemini/gemini-1.5-flash` (via Google AI Studio / LiteLLM)
- **Total Requests Evaluated:** 250 requests
- **Execution Date:** 2026-10-22

---

## Model Usage Breakdown

### 1. Vision Perception Layer (`code/vision_parser.py`)
- **Purpose:** OCR extraction of missing transaction amounts from receipt PNGs.
- **Total Call Count:** 16 receipt images
- **Average Input Tokens per Call:** ~1,024 tokens (image + prompt)
- **Average Output Tokens per Call:** ~25 tokens (JSON object response)
- **Total Input Tokens:** ~16,384
- **Total Output Tokens:** ~400
- **Estimated Cost:** $0.00 (Google AI Studio Free Tier / Flash pricing < $0.01)

### 2. Text Perception Layer (`code/text_parser.py`)
- **Purpose:** Structured intent extraction (event cancellations) from user messages.
- **Batching Strategy:** 20 messages per prompt batch (11 total API calls for 215 messages)
- **Total Call Count:** 11 batch requests
- **Average Input Tokens per Call:** ~1,500 tokens
- **Average Output Tokens per Call:** ~150 tokens
- **Total Input Tokens:** ~16,500
- **Total Output Tokens:** ~1,650
- **Estimated Cost:** $0.00 (Google AI Studio Free Tier / Flash pricing < $0.01)

---

## Cost & Efficiency Summary

| Metric | Vision Layer | Text Layer | Total Pipeline |
| :--- | :--- | :--- | :--- |
| **API Calls** | 16 | 11 | **27** |
| **Input Tokens** | ~16,384 | ~16,500 | **~32,884** |
| **Output Tokens** | ~400 | ~1,650 | **~2,050** |
| **Est. Total Cost ($)** | $0.00 | $0.00 | **$0.00** |

*Note: All arithmetic modeling, 90-day cash flow simulation, and candidate option ranking were performed deterministically using local Python execution, incurring zero additional API call cost.*