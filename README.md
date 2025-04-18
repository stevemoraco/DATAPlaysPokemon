# OpenAI Plays Pokémon ／ LLM Speed‑Runner

OpenAI Plays Pokémon by Lander Media / **Steve Moraco**  
Code originally authored by **o4‑mini**, modernised & extended by the community.

This project is an autonomous agent that attempts a Pokémon Red speed‑run
inside the [PyBoy](https://github.com/Baekalfen/PyBoy) emulator.  All
interaction happens through *function / tool* calls – no direct memory hacking
or hard‑coded game logic.  A screenshot (optionally with a coloured collision
overlay) is fed to the language model every turn; the model answers with a
tool call such as `press_buttons` or (if enabled) `navigate_to`.

The original prototype used the OpenAI *Responses* API.  The current code base
supports **two providers**:

• Anthropic **Claude** (default – best results with *Claude‑3 Sonnet*; repo pins to `claude‑3‑7‑sonnet‑20250219`)  
• OpenAI Chat (any function‑calling model, e.g. *gpt‑4o‑mini* or *o4‑mini*)


---

## Features

• `press_buttons` – press any combination of `a b start select up down left right`  
  (`wait=true` by default; set `wait=false` for very rapid input)  
• `navigate_to` (conditional) – A* path‑finding to grid coordinates; enabled by
  setting `USE_NAVIGATOR = True` in `config.py`.  
• Screenshot‑in‑the‑loop reasoning – the **pixels are ground‑truth**; RAM is
  read only to provide helper info such as location names.  
• FastAPI + WebSocket **web UI** with real‑time game feed and agent thoughts.  
• Per‑run logging (`frames/`, `claude_messages.log`, `game.log`).  
• Automatic conversation summarisation to stay within the context window.


---

## Quick Start

1. Clone the repository and enter it:

   ```bash
   git clone <repo-url>
   cd <repo>
   ```

2. Create a virtual environment and install dependencies.  *PyBoy requires
   SDL2 – on macOS run `brew install sdl2`, on Debian/Ubuntu
   `sudo apt-get install libsdl2-dev`.*

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Place a **Pokémon Red (US)** ROM in the project root and name it
   `pokemon.gb` (or pass a different path with `--rom`).

4. Export *at least one* API key:

   ```bash
   # Anthropic (default)
   export ANTHROPIC_API_KEY=<your_key>

   # OR OpenAI
   export OPENAI_API_KEY=<your_key>
   ```

5. Launch the **web UI** (backend + browser client):

   ```bash
   python main.py --steps 500 --port 3000 --overlay  # overlay colours walkability
   ```

   *This command starts a FastAPI server and immediately opens the UI at*
   `http://localhost:3000`.  When the page loads press **Start** to begin the
   run – this triggers a `/start` API call that actually boots the autonomous
   agent.  (Nothing will happen until you click the button.)

6. (Optional) pure‑CLI test run

   There is no dedicated CLI entry‑point at the moment.  If you only need a
   quick sanity check without the browser you can execute the agent directly:

   ```bash
   python -m agent.simple_agent --rom pokemon.gb   # stops after 10 steps by default
   ```

   or

   ```bash
   python simple_agent.py --rom pokemon.gb
   ```

   Logs will still be written, but screenshots/frames are only dumped when the
   web runner (`web/agent_runner.py`) is active.


---

## Command‑line flags (excerpt)

| Flag | Description |
|------|-------------|
| `--provider {anthropic|openai}` | Select LLM backend (default `anthropic`) |
| `--model <name>` | Chat model name (falls back to `config.MODEL_NAME` for Anthropic or `o4‑mini` for OpenAI) |
| `--steps <N>` | Emulator steps to execute before exit (web UI keeps running) |
| `--port <P>` | Port for the FastAPI / WebSocket server (default `3000`) |
| `--overlay` | Draw coloured collision overlay on each frame |
| `--save-state <file.state>` | Load a PyBoy `.state` file before starting |


---

## Logs & Artifacts

Each run creates `logs/run_<timestamp>/` containing:

* `frames/` – PNG screenshot for every frame (available when the web runner is used; skipped in pure‑CLI runs).
* `claude_messages.log` – raw assistant messages (name historic).
* `game.log` – emulator & tool execution logs.


---

## Configuration (`config.py`)

| Variable | Purpose |
|----------|---------|
| `MODEL_NAME` | Default Claude model (currently `claude-3-7-sonnet-20250219`) |
| `TEMPERATURE` | Sampling temperature |
| `MAX_TOKENS` | Token limit before context is summarised |
| `USE_NAVIGATOR` | `True` to expose the experimental `navigate_to` tool |


---

## Contributing

Pull requests are welcome — please keep changes focused and stylistically
consistent with the existing code.  If you extend functionality, update this
README sparingly to preserve the project’s history and credits.
