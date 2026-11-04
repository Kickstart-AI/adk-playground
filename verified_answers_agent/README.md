# Source support agent

## Setup and use

Run `uv sync` from the repository root. PDF input also requires Poppler's
`pdftotext` executable on `PATH`. PDFs must contain extractable text; OCR is not provided.

Set these values in `verified_answers_agent/.env`:

```env
GOOGLE_API_KEY=your-google-api-key
CLOUDFLARE_ACCOUNT_ID=your-cloudflare-account-id
CLOUDFLARE_API_TOKEN=your-cloudflare-api-token
```

The Cloudflare account needs sufficient unified billing credits for Jev.
Do not commit the `.env` file.

Start the UI from the repository root:

```bash
uv run adk web --port 8000
```

Open `http://localhost:8000` and select `verified_answers_agent`. Attach exactly
one PDF or UTF-8 text file and include a question in the same message on each turn.
Restart the server after changing the agent configuration.

## Models and output

The generator uses `gemini-3.1-flash-lite` with `thinking_level="minimal"`, its
lowest supported thinking level. It produces `sentence`–`quotes` pairs. The prompt
asks for one complete, self-contained sentence stating one independently verifiable
fact per pair, with one or more supporting quotes. Qualifiers stay with their fact.

Jev `typesafe/jev-1.14.0` runs through Cloudflare's unified
`POST /client/v4/accounts/{account_id}/ai/run` endpoint. The agent uses direct HTTP,
with the model request under `input`, rather than the TypeSafe SDK.

Each sentence appears in its original order, followed by its quotes and a
`SUPPORT` probability from 0 to 1. Jev evaluates all quotes for that sentence
together, including quotes that fail the source match. This score judges whether
the supplied quotes support every claim in the sentence. It does not establish
that those quotes occur in the document or that the claims are factually true.

A separate `ANSWERS QUESTION` probability judges the original question against
only the answer sentences joined with newlines. Quotes, warnings, and support
scores are excluded from this judgment. Output assembly is deterministic: no
model rewrites the sentences after scoring, and no acceptance threshold removes them.
If the generator returns no candidates, the output is `No answer found.` followed
by the question-answer score.

## Source matching and warnings

Document extraction collapses whitespace into single spaces. PDF extraction first
joins words split after a line-ending hyphen, preserving the hyphen:
`fine-\n tuning` becomes `fine-tuning`.

Matching applies the same normalization to each quote and the extracted document:
Unicode case folding and NFC normalization, curly-to-straight quotation marks,
removal of soft hyphens and zero-width spaces, line-ending hyphen repair, and
whitespace collapse. A match is a contiguous substring of this normalized text,
not a byte-for-byte match to the original file. Displayed quotes keep their original text.

Each unmatched or empty quote gets a numbered warning directly beneath it.
Sentences remain visible and receive a Jev score even when a quote does not match.

A separate warning appears when the answer sentence has other than one
sentence-ending mark (`.`, `?`, or `!`). Periods immediately followed by a digit
are excluded, so `3.14` does not count but `. 2` does. Quotes are excluded from
this count. This heuristic can flag abbreviations and does not prove that a
sentence contains only one fact. Warnings do not block or regenerate answers.

## Data and failures

Google receives the question and extracted document. Cloudflare receives each
candidate sentence with its quotes, then the question and joined answer sentences.
The full document is not sent to Jev. Model, extraction, or invalid-probability
errors stop the workflow; there is no fallback provider.

## Checks

```bash
uv run pytest -q
uv run ruff check verified_answers_agent tests/verified_answers_agent
```
