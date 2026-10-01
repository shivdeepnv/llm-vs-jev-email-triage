# llm-vs-jev-email-triage

Classifies a synthetic inbox of 100 emails with up to three backends and compares them:

- **Claude Haiku 4.5** and **Claude Sonnet 5**: general-purpose LLMs, prompted to reply with JSON (disposition + one-sentence reason).
- **Jev** from [TypeSafe AI](https://typesafe.ai): a "System One" model that returns a typed choice with probabilities, no generated text.

Every email gets its own call, with no rule-based pre-filtering. Each email is classified into one disposition:
`reply`, `archive`, `defer`, `delegate` or `escalate`. Emails are treated as data, never as instructions, and an email
that tries to instruct an automated assistant should be escalated.

A run writes `report.html` (a shareable, self-contained comparison) and `results.json` (the raw data). A separate
tool, `label_tool.py`, lets you add your own ground-truth labels so the report can show real accuracy, not just
agreement between backends.

## What you need

- Python 3.10+
- A TypeSafe API key (get one at https://console.typesafe.ai/), for the Jev backend
- An Anthropic API key, for the Haiku and/or Sonnet backends

You don't need both providers -- run whichever backend(s) you have a key for.

## Setup

```bash
git clone https://github.com/shivdeepnv/llm-vs-jev-email-triage.git
cd llm-vs-jev-email-triage

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Open `.env` and fill in `TYPESAFE_API_KEY` and/or `ANTHROPIC_API_KEY`, depending on which backends you'll run.

## Run

```bash
# One backend
python llm_classifier.py --backend jev
python llm_classifier.py --backend haiku
python llm_classifier.py --backend sonnet

# Several at once, comma-separated
python llm_classifier.py --backend haiku,jev

# All three, head to head -- this is the one that produces the full comparison report
python llm_classifier.py --backend all --workers 10 --delay 0
```

Then open `report.html` in a browser.

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--backend` | `haiku` | Any of `haiku`, `sonnet`, `jev`, comma-separated (e.g. `haiku,sonnet,jev`), or the aliases `both` (haiku+jev) and `all` (all three). |
| `--workers` | `1` | Concurrent requests. 1 is sequential. Use 10 for a comparison run; much higher values can trigger rate limits and slow the run down instead of speeding it up. |
| `--delay` | `REQUEST_DELAY_S` (1.0) | Seconds each worker pauses after a call. Use `0` for a fair speed comparison. |

Backends run one after another, never at the same time, so their timings don't interfere with each other.

## Running the comparison

Use exactly this command:

```bash
python llm_classifier.py --backend all --workers 10 --delay 0
```

- It keeps up to 10 requests in flight at a time per backend (a rolling pool, not batches) with no pause between calls.
- It runs each backend on all 100 emails in turn.
- Check the summary for each backend: it should say `model handled 100/100` and `failed and escalated: 0`.
  If a backend shows failures, or you see `~ ... rate limited (429)` lines, the timings for that run are
  contaminated -- lower `--workers` and run again.
- Timings are one measurement. Network and time of day affect them, so read them as "on this run", not as a
  permanent verdict.

## What the report shows

- **Speed:** average, median and p95 time per call (pure request latency), and wall clock for the whole inbox, for
  each backend. p95 is the time the slowest 5% of calls took or exceeded.
- **Agreement:** with two or more backends, how often they pick the same disposition (pairwise, and all-agree), and
  a table of every email where they don't all agree.
- **Accuracy against your labels:** appears only once `labels.json` exists (see below) -- per-backend accuracy, a
  confusion matrix, and escalate recall/precision against labels you provide.
- **Per backend:** model name, tokens in/out, cost per call, time per call, and each email's result.

Agreement alone is not accuracy -- it only says the backends differ, not which one (if any) is right. That's what
the labeling step below is for.

## Adding your own ground truth

```bash
python label_tool.py
```

This opens `http://127.0.0.1:8756/` with one email at a time and the five dispositions (plus "unsure" for anything
genuinely ambiguous). It's deliberately **blind**: no model prediction is shown while you label, so your judgement
isn't anchored on what a backend already said. Keyboard shortcuts: `1`-`5` for the five dispositions, `0` for
unsure, arrow keys to move without labeling.

Every click is saved immediately to `labels.json`, so you can quit and resume later -- it reopens on your first
unlabeled email. "Unsure" labels are recorded but excluded from scoring, rather than silently counting as wrong for
every backend.

Once you've labeled some or all of the inbox, re-run the classifier (same command as above) and `report.html` gets
a new "Accuracy against your labels" section: accuracy, a confusion matrix, and escalate recall/precision, per
backend. Escalate **recall** is the more important of the two -- it's the fraction of emails you said should be
escalated that the backend actually escalated, i.e. what it would have missed. You don't need to label all 100 to
get a report; it scores against whatever is in `labels.json` at run time.

## Notes and known gaps

- **Sonnet pricing is an estimate** ($3.00 in / $15.00 out per million tokens in `llm_classifier.py`), not
  confirmed against Anthropic's current published pricing. Verify before trusting the Sonnet cost figures.
- **Jev cost:** the API doesn't report pricing. Set `JEV_PRICE_INPUT` and `JEV_PRICE_OUTPUT` (USD per million
  tokens) in `.env` to get a cost figure. Token counts show `n/a` if the API doesn't return them.
- **Jev gives no reason**, only a label with probabilities and a confidence, so its rows show confidence instead of
  an explanation.
- **Sonnet has occasionally needed a retry** where Haiku hasn't, from JSON parsing edge cases in its replies. It
  still ends up correctly classified after a retry; the console shows a `~` line whenever this happens.
- **Rate limits:** a high `--workers` value can trigger 429s. Failed calls retry (`MAX_RETRIES`, default 4) and
  then fall back to `escalate` rather than guessing. The summary shows how many fell back.
- **`report.html` shows email subjects** and each LLM backend's stated reasons. The inbox is synthetic, but skim it
  before sharing a report built from a different inbox.
- **`labels.json` is your ground truth**, not a secret -- it's fine to commit and share so the accuracy numbers are
  reproducible.

## Files

- `llm_classifier.py`: the three classifiers, the runner, scoring against labels, and the report generator
- `label_tool.py`: local, blind labeling tool for `inbox.json` -> `labels.json`
- `config.py`: settings loaded from `.env`
- `inbox.json`: 100 synthetic emails
- `.env.example`: template for your `.env`
- `labels.json` (created by `label_tool.py`): your ground-truth labels, if you've added any
