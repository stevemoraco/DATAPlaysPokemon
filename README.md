# OpenAI Plays Pokemon

OpenAI Plays Pokemon by Lander Media / Steve Moraco  
Code authored by o4-mini

A minimal agent that uses the OpenAI Responses API (o4-mini) to speedrun Pokémon Red via the PyBoy emulator.

## Features

- Controls the emulator through function calling (press buttons, navigate)
- Screenshot-based gameplay: screenshots are the ground truth
- OpenAI Responses API with low‑effort reasoning to keep outputs concise
- Logging of game frames, model responses, and tool calls
- Web interface: live game view, filtered model thoughts, and context history
- Automatic summary of long chats to stay within token limits

## Setup

1. Clone this repository:

   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key:

   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   ```

4. Place your Pokémon Red ROM (`.gb`) in the project root.

## Usage

### Command Line Interface

Run headless gameplay using o4‑mini (default for OpenAI):

```bash
python main.py --provider openai
```

Optional flags:

- `--model <model_name>` (default: `o4-mini`)
- `--steps <N>` (number of steps to run; default: 1000)
- `--overlay` (enable tile overlay visualization)
- `--display` (show emulator window)
- `--sound` (enable audio)
- `--save-state <file.state>` (load a save state before starting)

### Web User Interface

Start the FastAPI web server:

```bash
python main.py --provider openai --model o4-mini
```

Open `http://localhost:3000` to view:

- **Game Screen**: real-time 30 FPS emulator output
- **Model Thoughts**: only actual model responses (no internal reasoning)
- **History / Context**: summarized chat history
- Control buttons: Run, Pause, Stop, Load Save State

## Logs

Each run outputs logs to `logs/run_<timestamp>/`:

- `frames/`: PNG screenshots per step
- `claude_messages.log`: model response logs
- `game.log`: emulator and agent logs

## Configuration

Adjust defaults in `config.py`:

- `MODEL_NAME` (default OpenAI chat model)
- `MAX_TOKENS` and `TEMPERATURE`

## Contributing

PRs welcome! Please open issues or pull requests on GitHub.