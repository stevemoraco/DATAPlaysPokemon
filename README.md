# OpenAI Plays Pokemon

Pokémon Red, fully controlled by an LLM 🤖🎮

This repo contains a **minimal, hack‑able agent** that teaches large language
models to play Pokémon Red inside the
[PyBoy](https://github.com/Baekalfen/PyBoy) Game Boy emulator.

Forked from the excellent
[`portalcorp/ClaudePlaysPokemon`](https://github.com/portalcorp/ClaudePlaysPokemon)
and extended with the **OpenAI Responses API** so it can run both the `o3`
and `o4‑mini` models alongside Anthropic Claude.  Anthropic remains the default
provider (see `--provider` flag below).

Project by **Lander Media / Steve Moraco**. Initial agent code by **o4‑mini**.

## Highlights

- Declarative function‑calling interface – the model calls the tools
  `press_buttons` and `navigate_to` (path‑finding helper enabled by default).
- Screenshot‑based gameplay – what the model “sees” is precisely what is on the
  screen, delivered as a PNG each step (hex‑encoded over WebSocket).
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

4. Place a Pokémon Red ROM (`pokemon.gb`) in the project root (or point to it
   with `--rom`).

## Usage

### Running the agent (CLI + Web UI)

The entry‑point is `main.py`. It both **spins up a FastAPI server** and starts
the agent. All interaction happens through the web UI – no separate headless
mode is needed.


```bash
# Quick start – Anthropic Sonnet playing 1 000 000 steps (~10 weeks), UI on port 3000
python main.py --rom pokemon.gb --steps 1000000

# Use OpenAI o4‑mini instead
python main.py --provider openai --model o4-mini
```

Key flags:

- `--rom <file.gb>` – path to the Pokémon Red ROM (default: `pokemon.gb`)
- `--steps <N>`     – maximum steps to execute (agent can be paused / resumed). Default is `1_000_000` (~30 frames × 10 weeks).
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

### Auto‑save snapshots

Inside each run folder you will also find `history_saves/` containing periodic
PyBoy `.state` snapshots.  These are written automatically:

1. Whenever the agent summarises the running conversation (~every 50 steps).
2. Immediately after the player transitions between major areas (e.g. moves to
   another floor or map).

You can resume from any snapshot by either:

• Supplying `--save-state <file>` on the command line, **or**
• Clicking *Load Save* in the web UI and selecting a `.state` file.

## Configuration tips

Global defaults live in `config.py`:

- `MODEL_NAME`   – default Anthropic model (CLI `--model` overrides)
- `TEMPERATURE`  – sampling temperature passed to the LLM
- `MAX_TOKENS`   – hard limit for the response size
- `USE_NAVIGATOR` – toggle the higher‑level `navigate_to` tool (default: True)

## Contributing

PRs welcome! Please open issues or pull requests 😊