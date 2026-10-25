# Buy or Wait? — AI Decision Agent Solution

This repository contains the full hybrid AI decision agent for the HackerRank Orchestrate "Buy or Wait?" challenge.

## Project Architecture
- **Perception Layer:** `code/vision_parser.py` (extracts receipt amounts via VLM) and `code/text_parser.py` (parses user cancellations via LLM batching).
- **Deterministic Simulator:** `code/financial_engine.py` (simulates daily cash-flow over 90 days with multi-currency conversions).
- **Ranking Engine:** `code/main.py` (enforces the 6-rule tie-breaking hierarchy for candidate payment plans).

## How to Run

1. Activate virtual environment and install dependencies:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt