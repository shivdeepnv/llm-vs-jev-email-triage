# llm-vs-jev-email-triage

Classifies a synthetic inbox of 100 emails with two different models and compares them:

- **Claude Haiku 4.5**: a general-purpose LLM that is prompted to reply with JSON (disposition + one-sentence reason).
- **Jev** from [TypeSafe AI](https://typesafe.ai): a "System One" model that returns a typed choice with probabilities, no generated text.

Every email gets its own call, with no rule-based pre-filtering. Each email is classified into one disposition:
`reply`, `archive`, `defer`, `delegate` or `escalate`. Emails are treated as data, never as instructions, and an email
that tries to instruct an automated assistant should be escalated.

The run writes `report.html` (a shareable, self-contained comparison) and `results.json` (the raw data).

## What you need

- Python 3.10+
- A TypeSafe API key (get one at https://console.typesafe.ai/)
- An Anthropic API key, only if you also want the Claude Haiku side of the comparison

## Setup

```bash
git clone https://github.com/<owner>/llm-vs-jev-email-triage.git
cd llm-vs-jev-email-triage

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Open `.env` and fill in `TYPESAFE_API_KEY` (and `ANTHROPIC_API_KEY` if you want the comparison).

## Run

```bash
# Jev only
python llm_classifier.py --backend jev

# Claude Haiku only (default)
python llm_classifier.py --backend haiku

# Both, head to head. This is the one that produces the comparison report.
python llm_classifier.py --backend both --workers 10 --delay 0
```

Then open `report.html` in a browser.

## Running the comparison

Use exactly this command:

```bash
python llm_classifier.py --backend both --workers 10 --delay 0
```

- It keeps up to 10 requests in flight at a time (a rolling pool, not batches) with no pause between calls.
- It runs Claude Haiku on all 100 emails first, then Jev on all 100, so the two never compete with each other.
- Check the summary for each backend: it should say `model handled 100/100` and `failed and escalated: 0`.
  If either backend shows failures, or you see `~ ... rate limited (429)` lines, the timings are contaminated. Lower
  `--workers` and run again.
- Timings are one measurement. Network and time of day affect them, so read them as "on this run".

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--backend` | `haiku` | `haiku`, `jev` or `both` |
| `--workers` | `1` | Concurrent requests. 1 is sequential. Use 10 for the comparison; much higher values can hit rate limits and slow the run down. |
| `--delay` | `REQUEST_DELAY_S` (1.0) | Seconds each worker pauses after a call. Use `0` for a fair speed comparison. |

The backends run one after the other, never at the same time, so their timings don't interfere.

## What the report shows

- **Speed:** average, median and p95 time per call (pure request latency) and wall clock for the whole inbox, for each backend. The median and p95 show the spread within the run: p95 is the time the slowest 5% of calls took or exceeded.
- **Agreement:** how often the two backends pick the same disposition, and a table of every email where they differ.
- **Per backend:** model name, tokens in and out, cost per call, time per call, and each email's result.

There is no ground truth here, so "disagree" means the two differ, not that one is wrong. Read the disagreement table
to judge which one you'd trust.

## Notes and known gaps

- **Jev cost:** the API doesn't report pricing. Set `JEV_PRICE_INPUT` and `JEV_PRICE_OUTPUT` (USD per million tokens)
  in `.env` to get a cost figure. Token counts show `n/a` if the API doesn't return them.
- **Jev gives no reason**, only a label with probabilities and a confidence, so its rows show confidence instead of an explanation.
- **First live run:** the Jev integration was written from the TypeSafe docs and SDK types and tested against a mock, without
  a real TypeSafe key. If the first real run fails, the console prints the error for each email that gives up, and a failed
  email is escalated rather than guessed. Please send the error output back.
- **Rate limits:** a high `--workers` value can trigger 429s. Failed calls retry (`MAX_RETRIES`, default 4) and then
  fall back to `escalate`. The summary shows how many fell back.
- **`report.html` shows email subjects** and Haiku's reasons. The inbox is synthetic, but skim it before sharing a report
  built from a different inbox.

## Files

- `llm_classifier.py`: both classifiers, the runner and the report generator
- `config.py`: settings loaded from `.env`
- `inbox.json`: 100 synthetic emails
- `.env.example`: template for your `.env`
