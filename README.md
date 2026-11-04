# Buy or Wait? Financial Decision Agent

A hybrid financial decision system. Cash-flow forecasting and ranking are deterministic; language-model integrations are isolated to optional perception/classification workflows.

## Architecture

- **Perception:** `code/vision_parser.py` extracts receipt amounts; `code/text_parser.py` identifies message-based cancellations. These integrations use `BUY_OR_WAIT_MODEL` and are not needed for evaluation.
- **Forecast:** `code/forecast.py` and `code/financial_engine.py` clean event data, identify recurring series, apply currency conversion and simulate a 90-day balance window. Per-user event data is cached in memory; daily flows retain their source and event IDs.
- **Income classification:** `code/income_classifier.py` uses validated cached classifications when available. Missing classifications use a conservative recurring-series fallback; evaluation runs cache-only and does not call a model.
- **Spending changes and ranking:** `code/main.py` tests eligible expense reductions/stops and ranks safe payment options by deadline, spending changes, cost, start date, payment count, then option ID.
- **Retrieval:** `code/retrieval.py` indexes `dataset/messages.csv` locally. It prefers a supported multilingual FastEmbed model and otherwise uses character n-gram TF-IDF. Retrieval is filtered by user and strict as-of date; its persistent index is in `dataset/.cache/`.
- **Explanations:** `code/explain.py` builds deterministic explanations from forecast outputs and attaches retrieved message IDs as evidence. It copies no message text into explanations.

## Setup and commands

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How it works

The payment advisor follows a deterministic, evidence-backed path:

**Data** (`dataset/`) → **forecast** (90-day balances, minimum reserve, and dated cash flows) → **income reliability** (validated income classes or the conservative recurring-series fallback) → **options** (full payment, partial payment, installments, wait, and eligible spending changes) → **report** (ranked options and the forecast evidence behind them).

The terminal and browser interfaces use the same `TerminalSession` intake and
`Assistant.advise()` engine. Amounts without an explicit currency use the account's
home currency. Conversion only uses the pipeline's existing dated direct rates.

Run the browser terminal or the plain-text CLI:

```bash
streamlit run app.py
python code/terminal.py --user user_07
```

Terminal commands: `/user <id>` switches accounts, `/users [currency]` lists up to
eight accounts, `/date YYYY-MM-DD` changes the as-of date, `/situation` prints the
current baseline, `/why` shows the engine evidence and Failproof session, `/skip`
accepts remaining defaults, `/reset` starts over, `/export` prints/downloads the
latest report, and `/help` lists usage.

Run syntax and automated tests without making model calls:

```bash
python -m py_compile app.py code/*.py tests/*.py
pytest -q
```

Run the decision pipeline:

```bash
python code/main.py
```

Run the deterministic, cache-only accuracy evaluation or the local retrieval evaluation:

```bash
python evaluation/evaluate.py
python evaluation/eval_retrieval.py
python evaluation/compare_income_caches.py cache-a.csv cache-b.csv
python evaluation/make_usage_report.py
```

Explicitly populate the income-classification cache (this command can make model/API calls):

```bash
python code/main.py --fill-income-cache
```

## Environment variables

- `BUY_OR_WAIT_MODEL`: model used by the optional receipt/message perception integrations and the default income-classifier model.
- `INCOME_CLASSIFIER_MODEL`: overrides `BUY_OR_WAIT_MODEL` for income classification.
- `INCOME_CACHE_PATH`: income-classification cache CSV path (default `dataset/income_classes.csv`).
- `INCOME_FILL_DELAY`: delay in seconds between user batches when filling the income cache (default `8`).
- `INCOME_BATCH_SIZE`: number of users per income-classification batch (default `5`).
- `FAILPROOF_ENV`: FailproofAI environment for traces (default `dev`; evaluation uses `eval`).
- `RUN_ID`: optional run identifier used to make session IDs unique across runs.

Every model attempt appends a local usage record to `logs/usage.jsonl`. Generate the
per-purpose and per-model token summary at `evaluation/usage_report.md` with
`python evaluation/make_usage_report.py`.

## Failproof tracing

The pipeline configures the FailproofAI SDK with `fp.configure(environment="dev")`. Request processing is wrapped in `fp.session`, `fp.agent`, and `fp.tool_call` contexts, while classifier/perception model requests, responses, and errors are emitted through the SDK event interface. Configure the SDK's credentials and project/environment settings as required by your FailproofAI installation. The evaluation configures the `calibration` environment for cache-only classification and restores `dev` afterward.

## Preserved evaluation scores

The existing evaluation baseline is preserved (the terminal/UI change does not
modify `output.csv`):

| Output | Accuracy |
| --- | ---: |
| Affordability status | 76.00% |
| Recommended payment method | 88.00% |
| Payment plan | 80.00% |
| Amount safe to pay | 8.00% |
| Earliest full-payment date | 68.00% |
