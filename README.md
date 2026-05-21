# sopho-harness

## OpenAI Setup

1. Copy `.env.example` to `.env`.
2. Add your `OPENAI_API_KEY` to `.env`.
3. Add your `OPENAI_MODEL` to `.env`.
4. Optionally set `OPENAI_BASE_URL` if you want to use a custom OpenAI-compatible endpoint.
5. Optionally set `OPENAI_API_MODE` if you want to override the default API mode.
6. Install dependencies.
7. Run the CLI.

Example:

- `.env`:
  - `OPENAI_API_KEY=your_key`
  - `OPENAI_MODEL=gpt-5.4`
  - `OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1` (optional)
  - `OPENAI_API_MODE=responses` (optional)
- Default task: `uv run sopho-harness`
- Custom task: `uv run sopho-harness Say hello briefly`

Notes:

- `OPENAI_API_KEY` is required.
- `OPENAI_MODEL` is required.
- `OPENAI_BASE_URL` is optional.
- `OPENAI_API_MODE` is optional.
- If `OPENAI_BASE_URL` is not set, the CLI uses the default OpenAI endpoint.
- If `OPENAI_BASE_URL` is set, the CLI sends requests to that OpenAI-compatible endpoint instead.
- If `OPENAI_API_MODE` is not set, the CLI uses `responses` by default.
- `OPENAI_API_MODE` must be either `responses` or `chat_completions`.

## TODO

- Add a more explicit environment/bootstrap flow for Python tooling so the availability of tools like `ruff` can be prepared ahead of agent startup.
