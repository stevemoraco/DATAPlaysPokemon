# OpenAI Plays Pokemon

Pokémon Red, fully controlled by an LLM 🤖🎮

This repo contains a **minimal, hack‑able agent** that teaches large language
models to play Pokémon Red inside the
[PyBoy](https://github.com/Baekalfen/PyBoy) Game Boy emulator.

Originally boot‑strapped around the OpenAI “o4‑mini” Responses API, the code has
evolved and **now supports both Anthropic *and* OpenAI models**. Anthropic is the
default provider (see `--provider` below).

Project by **Lander Media / Steve Moraco**. Initial agent code by **o4‑mini**.

## Highlights

- Declarative function‑calling interface – the model calls the tools
  `press_buttons` (and optionally `navigate_to`).
- Screenshot‑based gameplay – what the model “sees” is precisely what is on the
  screen, delivered as a base‑64 encoded PNG each step.
- FastAPI + WebSockets live UI – watch the game, pause, resume, load save
  states, and inspect the model’s thoughts in real time at
  `http://localhost:<port>`.
- Automatic log folder per run (frames, model messages, structured game log).
- Context summarisation to keep the conversation within token limits.

## Setup

1. Clone this repository:

   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```

2. Install Python dependencies (Python ≥3.10 recommended):

   ```bash
   pip install -r requirements.txt
   ```

3. Provide an API key for **your preferred provider**:

   - Anthropic (𝚍𝚎𝚏𝚊𝚞𝚕𝚝):

   ```bash
   export ANTHROPIC_API_KEY="sk-ant-…"
   ```

   - OpenAI (when running with `--provider openai`):

   ```bash
   export OPENAI_API_KEY="sk-openai-…"
   ```

4. *(Optional)* If you want the image‑understanding Google tools to work,
   export `GOOGLE_API_KEY` as well.

5. Place a Pokémon Red ROM (`pokemon.gb`) in the project root (or point to it
   with `--rom`).

## Usage

### Running the agent (CLI + Web UI)

The entry‑point is `main.py`. It both **spins up a FastAPI server** and starts
the agent. All interaction happens through the web UI – no separate headless
mode is needed.

```bash
# Quick start – Anthropic Sonnet playing 1 000 steps, UI on port 3000
python main.py --rom pokemon.gb --steps 1000

# Use OpenAI o4‑mini instead
python main.py --provider openai --model o4-mini
```

Key flags:

- `--rom <file.gb>` – path to the Pokémon Red ROM (default: `pokemon.gb`)
- `--steps <N>`     – maximum steps to execute (agent can be paused / resumed)
- `--port <N>`      – port for the FastAPI server / web UI (default 3000)
- `--save-state <file.state>` – load a PyBoy save state at startup
- `--overlay`       – draw walkable‑tile overlay inside the game feed
- `--provider anthropic|openai` – choose LLM backend (default: anthropic)
- `--model <name>`  – override default model for the chosen provider

Open `http://localhost:<port>` in a browser to see:

1. **Game Screen** – live 30 FPS video
2. **Assistant Messages** – the model’s tool calls & high‑level reasoning
3. **Context History** – compressed conversation so far
4. Controls – *Run*, *Pause*, *Stop*, *Load Save State*

## Logs

Each run writes to `logs/run_<timestamp>/`:

- `frames/`: PNG screenshots per step
- `claude_messages.log`: model response logs
- `game.log`: emulator and agent logs

## Configuration tips

Global defaults live in `config.py`:

- `MODEL_NAME`   – default Anthropic model (CLI `--model` overrides)
- `TEMPERATURE`  – sampling temperature passed to the LLM
- `MAX_TOKENS`   – hard limit for the response size
- `USE_NAVIGATOR` – expose the higher‑level `navigate_to` tool

## Contributing

PRs welcome! Please open issues or pull requests 😊