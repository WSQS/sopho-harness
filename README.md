# sopho-harness

## MiniMax Setup

1. Copy `.env.example` to `.env`.
2. Add your `MINIMAX_API_KEY` to `.env`.
3. Install dependencies.
4. Run the CLI.

Example:

- `.env`: `MINIMAX_API_KEY=your_key`
- Default task: `uv run sopho-harness`
- Custom task: `uv run sopho-harness Say hello briefly`

## TODO

- Add a more explicit environment/bootstrap flow for Python tooling so the availability of tools like `ruff` can be prepared ahead of agent startup.
