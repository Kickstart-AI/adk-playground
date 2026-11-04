# Source support agent

In ADK web, select `verified_answers_agent`. Attach one PDF or UTF-8 text file and enter a question in the same message. PDF files must contain extractable text. Install Poppler's `pdftotext` command to use PDFs.

Set `GOOGLE_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`, and `CLOUDFLARE_API_TOKEN` in the agent's `.env` file. The answer model is `gemini-3.5-flash-lite`. Cloudflare Workers AI runs `@cf/cloudflare/clef`.

The answer model proposes sentence and quote pairs. The workflow checks that each quote occurs exactly in the extracted text, then sends the sentence, quote, and document to Clef for source support probabilities. Each generated sentence appears in order with its quote and document support probabilities. Clef also judges whether the complete assembled answer answers the question; this probability appears separately. A sentence with an unmatched quote is marked as unavailable. A probability is a model judgment, not proof. A Clef or extraction error stops the workflow. Documents and questions are sent to Google; the document, candidate sentences, quotes, question, and assembled answer are sent to Cloudflare Workers AI.
