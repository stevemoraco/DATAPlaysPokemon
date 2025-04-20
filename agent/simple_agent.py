import base64
import copy
import io
import logging
import os
from datetime import datetime
# Google GenAI libraries are optional; import lazily so the rest of the
# application runs even when they are not installed.
try:
    from google import genai  # type: ignore
    from google.genai.types import Tool, GenerateContentConfig, GoogleSearch  # type: ignore
except ImportError:  # pragma: no cover – Google dependencies optional
    genai = None  # type: ignore
    Tool = GenerateContentConfig = GoogleSearch = None  # type: ignore

from config import MAX_TOKENS, MODEL_NAME, TEMPERATURE
from agent.prompts import SYSTEM_PROMPT, SUMMARY_PROMPT
from agent.tools import AVAILABLE_TOOLS

from agent.emulator import Emulator
from typing import Tuple, List, Dict
from agent.memory_reader import PokemonRedReader
# External libs
from anthropic import Anthropic
import json

# ------------------------------------------------------------------
# Helper to clean / deduplicate dialog strings for compact display
# ------------------------------------------------------------------


def _clean_dialog(raw: str) -> str:
    parts, seen = [], set()
    arrow = False
    for ln in raw.split("\n"):
        t = ln.strip()
        if not t:
            continue
        if t == "▼":
            arrow = True
            continue
        if t in seen:
            continue
        seen.add(t)
        parts.append(t)
    combined = " / ".join(parts)
    if arrow and combined:
        combined += " ▼"
    return combined or raw.strip() or "None"

# Wrapper for OpenAI Responses API blocks to match Anthropic-style blocks
class _OpenAIBlock:
    def __init__(self, data):
        # Robustly accept arbitrary objects. If *data* is not a mapping (e.g.
        # a bare list sometimes returned by SDKs), coerce it to a plain‑text
        # block so downstream code never crashes.
        if not isinstance(data, dict):
            self.type = "text"
            self.text = str(data)
            self.is_reasoning = False
            return

        btype = data.get("type")
        # Keep original type for downstream filtering
        self.raw_type = btype
        # Text output or summary text
        if btype in ("output_text", "summary_text"):
            self.type = "text"
            self.text = data.get("text", "")
        # High-level reasoning block: extract summary texts
        elif btype == "reasoning":
            self.type = "text"
            # summary is a list of summary_text blocks
            lines = []
            # Some models put their chain‑of‑thought under "analysis" or "text"
            if isinstance(data.get("analysis"), str):
                lines.append(data.get("analysis"))
            # Anthropic style – list of summary_text blocks
            for item in data.get("summary", []):
                if isinstance(item, dict) and item.get("text"):
                    lines.append(item["text"])
            # Fallback plain text field
            if not lines and isinstance(data.get("text"), str):
                lines.append(data["text"])
            self.text = "\n".join(lines).strip()
        # Function/tool call
        elif btype in ("function_call", "tool_use"):
            self.type = "tool_use"
            self.name = data.get("name")
            args = data.get("arguments") or data.get("input") or {}
            if isinstance(args, str):
                try:
                    self.input = json.loads(args)
                except Exception:
                    self.input = {}
            else:
                self.input = args
            # preserve call id if present
            self.id = data.get("call_id") or data.get("id")
        else:
            # Unknown type, treat as text
            self.type = "text"
            self.text = str(data)
        # Flag reasoning blocks so we can optionally exclude them from history
        self.is_reasoning = btype == "reasoning"
    
    # No return, just block wrapper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Utility for pretty‑printing message objects with redacted base64 data
# ---------------------------------------------------------------------

def _pretty_json(obj) -> str:
    """Return obj as pretty‑printed JSON for logging."""
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=False)
        # Replace escaped newlines so they render as actual line breaks
        return text.replace("\\n", "\n")
    except Exception:
        return str(obj)


def get_screenshot_base64(screenshot, upscale=1):
    """Convert PIL image to base64 string."""
    # Resize if needed
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        screenshot = screenshot.resize(new_size)

    # Convert to base64
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()



class SimpleAgent:
    # ------------------------------------------------------------------
    # Sidebar helper
    # ------------------------------------------------------------------

    def _update_sidebar(self, line: str):
        raw = line.strip()
        if not raw or raw.startswith("##"):
            return
        norm = raw.lower()
        if norm == getattr(self, "_prev_sidebar_norm", None):
            return  # duplicate, skip
        self.last_message = raw[:120]
        self._prev_sidebar_norm = norm
    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------

    def _assistant_blocks_to_content(self, blocks):
        """Convert list of _OpenAIBlock / Anthropic blocks into the message
        content format we persist in ``self.message_history``.

        We preserve:
        • All text blocks (including reasoning)
        • All tool_use calls with their arguments / id
        """
        content = []
        for block in blocks:
            if block.type == "text":
                if block.text.strip():
                    content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                entry = {"type": "tool_use", "name": block.name, "input": block.input}
                if getattr(block, "id", None) is not None:
                    entry["id"] = block.id
                content.append(entry)
        return content

    def _append_assistant_message(self, blocks):
        """Append the assistant's reply (represented by *blocks*) to
        ``self.message_history`` so it is always retained for future turns."""
        self.message_history.append({
            "role": "assistant",
            "content": self._assistant_blocks_to_content(blocks),
        })
        self._trim_history()

    def _trim_history(self):
        """Trim oldest messages to maintain max_history sliding window."""
        if self.max_history and len(self.message_history) > self.max_history:
            overflow = len(self.message_history) - self.max_history
            del self.message_history[:overflow]

    def _get_system_prompt(self) -> str:
        """Return the base system prompt optionally appended with the latest
        conversation summary so the model always has up‑to‑date context.
        """
        prompt_parts = [self.system_prompt]

        # Append rolling conversation summary if available
        if self.history_summary:
            prompt_parts.append(self.history_summary)

        # Append latest full game‑state block if available
        if self.latest_game_state:
            prompt_parts.append(
                "<CurrentTurnGameState>\n" + self.latest_game_state + "\n</CurrentTurnGameState>"
            )

        return "\n\n".join(prompt_parts)

    # ---------------------- housekeeping --------------------------
    def _prune_old_images(self, keep_last: int = 1):
        """Remove image blocks from all but the most recent *keep_last* user
        messages to keep token count low."""

        def strip_images(blocks: list):
            """Recursively remove image blocks from a list of content blocks."""
            new_blocks = []
            for blk in blocks:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") == "image":
                    continue  # drop
                # If this is a tool_result with nested content, strip inside
                if blk.get("type") == "tool_result" and isinstance(blk.get("content"), list):
                    blk = blk.copy()
                    # Remove images and nested blocks completely – we don't
                    # need them after first turn.
                    blk.pop("content", None)
                new_blocks.append(blk)
            return new_blocks

    # ------------------------------------------------------------------
    # Location helpers
    # ------------------------------------------------------------------

    def _normalize_location(self, loc: str) -> str:
        """Convert raw location string to a friendlier form (e.g. 1F→FIRST FLOOR)."""
        if not loc:
            return loc
        repl = {
            " 1F": " FIRST FLOOR",
            " 2F": " SECOND FLOOR",
            " 3F": " THIRD FLOOR",
            " 4F": " FOURTH FLOOR",
            " 5F": " FIFTH FLOOR",
        }
        out = loc.upper()
        for k, v in repl.items():
            out = out.replace(k, v)
        return out.title()

        if keep_last < 0:
            keep_last = 0

        for msg in self.message_history[:-keep_last]:
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = strip_images(content)

    # ------------------------------------------------------------------
    # History compaction – strip verbose prompt text from older user messages
    # ------------------------------------------------------------------

    def _compact_history(self, keep_last: int = 2):
        """Reduce size of old user tool_result text blocks by keeping only the
        essential information (buttons, header, collision map grid, move
        effect line, final nudge).  Images are assumed to be already pruned.
        """

        def compact_text(txt: str) -> str:
            lines = txt.split("\n")
            out = []
            # 1. Pressed buttons line & header lines (# ...)
            for ln in lines:
                if ln.startswith("Pressed buttons:") or ln.startswith("# "):
                    out.append(ln)
            # 2. Collision map grid + marker lines
            if "Current Collision Map At" in txt:
                after = False
                grid_cnt = 0
                for ln in lines:
                    if "Current Collision Map At" in ln:
                        after = True
                        out.append("")
                        out.append(ln)
                        continue
                    if after and grid_cnt < 9 and ln.strip() and ln[0] in "0123456789":
                        out.append(ln)
                        grid_cnt += 1
                # move effect line
                for ln in lines:
                    if ln.startswith("How your last move"):
                        out.append("")
                        out.append(ln)
                        break
            # 3. final nudge line
            for ln in lines[::-1]:
                if ln.lower().startswith("make the next best move"):
                    out.append("")
                    out.append(ln)
                    break
            return "\n".join(out).strip()

        for msg in self.message_history[:-keep_last]:
            if msg.get("role") != "user":
                continue
            new_blocks = []
            for blk in msg.get("content", []):
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") == "text":
                    blk = blk.copy()
                    blk["text"] = compact_text(blk.get("text", ""))
                    new_blocks.append(blk)
                elif blk.get("type") != "image":
                    new_blocks.append(blk)
            msg["content"] = new_blocks

    # ------------------------------------------------------------------
    # Ensure the first developer message always contains the latest system
    # prompt (static instructions + summary + current game state, etc.)
    # ------------------------------------------------------------------

    def _refresh_system_prompt_in_history(self):
        if not self.message_history:
            return
        first = self.message_history[0]
        if first.get("role") != "developer":
            return
        # replace text of first input_text block
        blocks = first.get("content", [])
        for blk in blocks:
            if blk.get("type") in ("input_text", "text"):
                blk["text"] = self._get_system_prompt()
                return
    def _format_input_for_openai(self, messages):
        """Translate internal message history into OpenAI Responses API input format."""
        payload = []
                # Add system prompt block (developer role) only if caller hasn't already
        if messages and messages[0].get("role") == "developer":
            payload.append(messages[0])
            # Skip first message when iterating later
            messages_iter = messages[1:]
        else:
            payload.append({
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": self._get_system_prompt()}
                ],
            })
            messages_iter = messages
        # Iterate through remaining messages
        for msg in messages_iter:
            role = msg.get("role", "user")
            content = msg.get("content", [])
            # Normalize to list of blocks
            blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
            buffer = []
            for block in blocks:
                # Guard against malformed history entries that end up as raw
                # lists instead of dict content blocks (seen in rare edge
                # cases).
                if not isinstance(block, dict):
                    # Skip silently (or log once per run) to avoid dumping raw
                    # image/base64 data into logs which breaks downstream
                    # parsers.
                    logger.debug(
                        "Skipped non‑dict content block while building payload."  # noqa: E501
                    )
                    continue

                btype = block.get("type")
                # Plain user/input or assistant/output text
                if btype == "text":
                    payload_type = "input_text" if role != "assistant" else "output_text"
                    buffer.append({"type": payload_type, "text": block.get("text", "")})
                # Input image block
                elif btype == "image":
                    src = block.get("source", {})
                    if src.get("type") == "base64":
                        media = src.get("media_type", "image/png")
                        data = src.get("data", "")
                        url = f"data:{media};base64,{data}"
                    else:
                        url = src.get("url", "")
                    buffer.append({"type": "input_image", "image_url": url})
                # Model requested a tool/function call
                elif btype == "tool_use":
                    # Flush any buffered text/image blocks
                    if buffer:
                        payload.append({"role": role, "content": buffer})
                        buffer = []
                    call_id = block.get("id")
                    name = block.get("name")
                    args = block.get("input", {})
                    payload.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": json.dumps(args),
                    })
                # Tool executed, include function_call_output
                elif btype == "tool_result":
                    # Flush buffer
                    if buffer:
                        payload.append({"role": role, "content": buffer})
                        buffer = []
                    call_id = block.get("tool_use_id")
                    raw = block.get("raw_output", "")
                    # Serialize raw output to string
                    output_str = raw if isinstance(raw, str) else json.dumps(raw)
                    payload.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output_str,
                    })
                # Other block types are not directly mapped
            # Flush remaining buffered blocks
            if buffer:
                payload.append({"role": role, "content": buffer})
        return payload
    def __init__(
        self,
        rom_path,
        headless=True,
        sound=False,
        max_history=60,
        app=None,
        use_overlay=False,
        provider='anthropic',
        model_name=None
    ):
        """Initialize the simple agent.

        Args:
            rom_path: Path to the ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
            max_history: Maximum number of messages in history before summarization
            app: FastAPI app instance for state management
            use_overlay: Whether to show tile overlay visualization
        """
        self.emulator = Emulator(rom_path, headless, sound)
        self.emulator.initialize()
        # Initialize LLM provider client
        self.provider = provider
        # Determine model name
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = MODEL_NAME if provider == 'anthropic' else 'o4-mini'
        # Set up Anthropic or OpenAI client
        if self.provider == 'anthropic':
            try:
                self.llm_client = Anthropic()
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                raise
        elif self.provider == 'openai':
            try:
                from openai import OpenAI
            except ImportError:
                logger.error("OpenAI library not installed. Please install openai>=0.28.0")
                raise
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY not set for OpenAI provider")
                raise RuntimeError("Missing OPENAI_API_KEY for OpenAI provider")
            self.llm_client = OpenAI(api_key=api_key)
        else:
            logger.error(f"Unknown LLM provider: {self.provider}")
            raise ValueError(f"Unsupported provider: {self.provider}")
        # Google GenAI support is optional. Initialise only when the library is
        # available and credentials are present.
        if genai is not None:
            try:
                self.google_client = genai.Client()
            except Exception as e:
                logger.warning(
                    f"Google GenAI client initialization failed: {e}. Google tools disabled."
                )
                self.google_client = None
        else:
            self.google_client = None
        self.running = True
        # Step counters
        self._step_count = 0
        self._introspection_every = 15  # introspect every 15 steps
        self._summary_every = 50        # summarize every 50 steps
        self._initial_summary_done = False
        # Track repeated button sequences
        self._last_buttons_seq: list[str] | None = None
        self._repeat_button_count = 0
        # Temporarily seed with placeholder; will be replaced after system_prompt is set
        self.message_history = []
        self.max_history = max_history
        self.last_message = "Game starting..."  # Initialize last message
        self.last_tool_message = None  # Store latest tool result message for UI
        # Keep running window of recent dialog lines (cleaned). Store up to 50.
        from collections import deque
        self._dialog_history = deque(maxlen=50)
        self.latest_game_state: str | None = None  # full textual game state for system prompt

        # Track environment changes to highlight progress
        self._last_env: str | None = None
        # Track last sidebar line to avoid duplicates
        self._prev_sidebar_line: str | None = None

        # Will hold the latest conversation summary text (prefixed with
        # "CONVERSATION HISTORY SUMMARY...") so we can expose it to the LLM as
        # part of the system prompt every turn.
        self.history_summary: str | None = None
        self.app = app  # Store reference to FastAPI app
        self.use_overlay = use_overlay  # Store overlay preference
        
        # Modify system prompt if overlay is enabled
        if use_overlay:
            self.system_prompt = SYSTEM_PROMPT + '''
            There is a color overlay on the tiles that shows the following:

            🟥 Red tiles for walls/obstacles
            🟩 Green tiles for walkable paths
            🟦 Blue tiles for NPCs/sprites
            🟨 Yellow tile for the player with directional arrows (↑↓←→)
            '''
        else:
            self.system_prompt = SYSTEM_PROMPT

        # Now that system_prompt is finalized, seed history with it
        self.message_history = [{
            "role": "developer",
            "content": [
                {"type": "input_text", "text": self._get_system_prompt()}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": self._get_system_prompt()}
            ],
        }]

    def get_frame(self) -> bytes:
        """Get the current game frame as PNG bytes.
        
        Returns:
            bytes: PNG-encoded screenshot of the current frame with optional tile overlay
        """
        screenshot = self.emulator.get_screenshot_with_overlay() if self.use_overlay else self.emulator.get_screenshot()
        # Convert PIL image to PNG bytes
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        return buffered.getvalue()

    def get_last_message(self) -> str:
        """Get Claude's most recent message.
        
        Returns:
            str: The last message from Claude, or a default message if none exists
        """
        return self.last_message

    def process_tool_call(self, tool_call):
        """Process a single tool call."""
        tool_name = tool_call.name
        tool_input = tool_call.input
        logger.info(f"Processing tool call: {tool_name}")

        if tool_name == "press_buttons":
            buttons = tool_input.get("buttons", [])
            wait = tool_input.get("wait", True)
            logger.info(f"[Buttons] Pressing: {buttons} (wait={wait})")

            # --- Detect repeated button sequences (ignore trivial single-button 'a' presses) ---
            repeat_check = buttons != ["a"]  # allow isolated A presses without penalty
            if repeat_check and buttons == self._last_buttons_seq:
                self._repeat_button_count += 1
            elif repeat_check:
                self._repeat_button_count = 1
                self._last_buttons_seq = buttons
            else:
                # Reset tracking for allowed single-button sequences
                self._repeat_button_count = 0

            warning_text = None
            blocked = False
            if repeat_check:
                if self._repeat_button_count == 4:
                    warning_text = (
                        "Warning: you've pressed the exact same button sequence four times in a row; "
                        "try a different command if this isn't working."
                    )
                elif self._repeat_button_count >= 5:
                    warning_text = (
                        "Blocked: same button sequence five times consecutively. "
                        "Please change your approach."
                    )
                    blocked = True
                if warning_text:
                    logger.info(warning_text)

            # Capture coordinates and dialog before action for later comparison
            try:
                prev_coords = self.emulator.get_coordinates()
            except Exception:
                prev_coords = None

            prev_dialog = (self.emulator.get_active_dialog() or "").strip()

            # Execute button presses unless blocked
            if not blocked:
                result = self.emulator.press_buttons(buttons, wait)
            else:
                result = warning_text or "Button sequence blocked."

            # Capture coordinates and dialog after presses
            try:
                curr_coords = self.emulator.get_coordinates()
            except Exception:
                curr_coords = None

            curr_dialog = (self.emulator.get_active_dialog() or "").strip()

            dialog_after = curr_dialog  # maintain original variable name for downstream logic
            
            # Get a fresh screenshot after executing the buttons with tile overlay
            screenshot = self.emulator.get_screenshot_with_overlay()
            screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)

            # Get game state from memory after the action
            memory_info = self.emulator.get_state_from_memory()

            # Collision map string
            collision_map = self.emulator.get_collision_map()

            # Build unified text block -------------------------------------------------
            lines: list[str] = []

            in_dialog = bool(curr_dialog)

            # 1) Header line
            header_line = f"Pressed buttons: {', '.join(buttons)}"

            lines.append(header_line)

            if warning_text:
                lines.append(warning_text)

            lines.append("\nBelow is the new game state generated by your actions after this button or sequence of buttons was pressed.\n")

            # 2) Build custom Current Game State section
            mem_lines = memory_info.strip().split("\n")
            new_mem_lines: list[str] = []

            # Ensure dialog cleaning helpers are available and that we always
            # define the cleaned dialog variables so they exist regardless of
            # whether we are currently in a dialog/menu. This prevents
            # `UnboundLocalError` when the code later references the variables
            # outside of the conditional blocks where they are set.

            # Helper to condense and deduplicate dialog lines (defined once)
            def _clean_dialog(raw: str) -> str:
                parts: list[str] = []
                seen: set[str] = set()
                arrow = False
                for ln in raw.split("\n"):
                    t = ln.strip()
                    if not t:
                        continue
                    if t == "▼":
                        arrow = True
                        continue
                    if t in seen:
                        continue
                    seen.add(t)
                    parts.append(t)
                combined = " / ".join(parts)
                if arrow and combined:
                    combined += " ▼"
                return combined or raw.strip() or "None"

            # Define cleaned dialog variables with safe defaults so they are
            # always available even if `in_dialog` is False.
            cleaned_prev_dialog = _clean_dialog(prev_dialog) if prev_dialog else None
            cleaned_curr_dialog = _clean_dialog(curr_dialog) if curr_dialog else None

            for line in mem_lines:
                # Normalize dialog bullet formatting from "Dialog:" to "- Dialog:"
                if line.startswith("Dialog:"):
                    continue  # We'll inject dialog bullets ourselves

                # Skip original Coordinates bullet (we may reinject differently)
                if line.startswith("- Coordinates:"):
                    continue

                new_mem_lines.append(line)

                # After environment line, insert our custom bullets
                if line.startswith("- Current Environment:"):
                    if in_dialog:
                        # Only include previous dialog if it is different from the
                        # current one and not empty.
                        if prev_dialog and prev_dialog != curr_dialog:
                            new_mem_lines.append(f"- Previous Dialog: {prev_dialog}")

                        new_mem_lines.append(f"- Button Presses Just Run: {', '.join(buttons)}")

                        if cleaned_prev_dialog and cleaned_prev_dialog != cleaned_curr_dialog:
                            new_mem_lines.append(f"- Previous Dialog: {cleaned_prev_dialog}")

                            # Update dialog history tracker
                            self._dialog_history.append(cleaned_prev_dialog)

                        new_mem_lines.append(f"- Button Presses Just Run: {', '.join(buttons)}")
                        new_mem_lines.append(f"- Dialog: {cleaned_curr_dialog if cleaned_curr_dialog else 'None'}")

                    # Track current dialog line as well (if not empty)
                    if cleaned_curr_dialog:
                        self._dialog_history.append(cleaned_curr_dialog)
                else:
                    if prev_coords is not None and curr_coords is not None:
                        new_mem_lines.append(f"- Previous Coordinates: {prev_coords}")
                        new_mem_lines.append(f"- Button Presses Just Run: {', '.join(buttons)}")
                        new_mem_lines.append(f"- Current Coordinates: {curr_coords}")

            lines.extend(new_mem_lines)

            # Determine dialog visibility BEFORE deciding on collision map & prompt
            dialog_raw_after = self.emulator.get_active_dialog() or ""
            dialog_after = dialog_raw_after.strip()

            # 3) Collision map display
            if collision_map and not in_dialog:
                lines.append("\n## Collision Map")
                if curr_coords is not None:
                    lines.append(f"Current Collision Map At {curr_coords}:")
                lines.append("")
                lines.append(collision_map.strip())

                if prev_coords is not None and curr_coords is not None:
                    lines.append("\nHow your move affected this collision map:")
                    lines.append(f"{prev_coords} --({', '.join(buttons)})--> {curr_coords}")

                

            # 4) Final reflection / planning prompt (dialog aware)

            if dialog_after:
                lines.append(
                    f"\nYou are in a dialogue or menu. This dialogue line shows the screen that your next set of button presses will affect, so if the arrow is not next to the thing you want to select, start with D-pad presses."
                    "Otherwise, press A to advance text, or use the D‑pad to move the cursor then press A to select."
                    "Be very careful not to spam more than you intend to or this dialogue will wrap and re-start before you get a new screenshot. Only press the number of keys you are confident will move you productively along now.\n\nRespond to JUST the most recent dialogue shown above now with one button press."
                )
            else:
                lines.append(
                    "\nThink about the chat history, the latest collision map, and your initial instructions. What have you been trying recently and how is it going? What should you change about your approach to move more quickly and make more consistent progress? What next set of button presses would be different from what you've recently tried — perhaps several lefts or rights if you've been going up and down a lot, etc. — would advance that mission? Generate a tool call now based on this new screenshot & game state information like Current Player Environment and the Collision map. Paying VERY careful attention to ground your reasoning & answer in the latest screenshot and collision map ONLY.\n\nThink about what is in the screenshot now, reason about whether or not your last set of button presses advanced you in the direction you meant to go, decide how you can improve based on the entire history above, and then reply with both a tool call describing which buttons you'll press and a short game plan of what you plan to do next in the game. Press the sequence of buttons you think will take you to your destination, or use the navigate_to tool to pick a coordinate and the emulator will travel there for you."
                )
                lines.append("\nHow your move affected this collision map:")
                lines.append(f"{prev_coords} --({', '.join(buttons)})--> {curr_coords}")
                lines.append(f"Your reply must be different from \"{', '.join(buttons)}\" so you don't cause a loop. Perhaps \"{', '.join(buttons)}, {', '.join(buttons)}\"?")
                # Suggest valid moves to the model
                try:
                    moves = self.emulator.get_valid_moves()
                    if moves:
                        lines.append(
                            "\n\nYour Current Valid Moves: "
                            + ", ".join(moves)
                            + " (Note that the only time this valid moves parser will not be accurate is as you pass through doors, you have to press through the wall to continue. Consider pressing several D-Pad moves, lefts, rights, ups, and downs according to your collision map based on these valid moves you have available! Reply with some of these now.)"
                        )
                except Exception:
                    pass
                
		        
		        
            # TEMPORARY MINIMAL PROMPT VERSION (livestream debugging)
            minimal_lines: list[str] = []
            BtnDisp = ', '.join([b.upper() for b in buttons]) if buttons else "(none)"
            # ------------------------------------------------------
            # Include recent dialogue history (if any) at the very top
            # ------------------------------------------------------
            # Optionally include recent dialogue block for context.
            # Disabled by default because it can confuse the model when
            # deciding button presses inside menus.
            if False and self._dialog_history:
                minimal_lines.append("## Recent Dialog")
                minimal_lines.append("<RecentDialogue>")
                for dlg in self._dialog_history:
                    converted = dlg.replace(" / ", "\n")
                    minimal_lines.extend(converted.split("\n"))
                minimal_lines.append("</RecentDialogue>")
                minimal_lines.append("")

            # Record the buttons pressed
            minimal_lines.append(f"Pressed buttons: {BtnDisp}\n")

            # Determine current environment for header
            raw_loc = self.emulator.get_location() or "Unknown"
            loc = self._normalize_location(raw_loc)
            # Congratulate if environment changed
            env_change_line: str | None = None
            if self._last_env and loc != self._last_env:
                env_change_line = f"\n\nGreat! You moved from {self._last_env} to {loc}.\n\n"

                # --- Automated corrective movement: two LEFT presses ---
                try:
                    self.emulator.press_buttons(["left", "left"], wait=True)
                except Exception as exc:
                    logger.warning(f"Auto 'left left' movement failed: {exc}")

                # Immediate summary after environment change for concise context
                try:
                    self.summarize_history()
                except Exception as exc:
                    logger.warning(f"Auto‑summary after env change failed: {exc}")
            self._last_env = loc

            # Add environment header; insertion position handled later
            env_header_line = f"# {loc}"

            if collision_map and not in_dialog:
                minimal_lines.append("## Collision Map & Game State\n")
                if curr_coords is not None:
                    minimal_lines.append(f"Current Collision Map At {curr_coords}:")

                # Keep only first 9 grid lines (omit legend)
                grid_only = "\n".join(
                    line for i, line in enumerate(collision_map.split("\n")) if i < 9
                )
                minimal_lines.append(grid_only)

                # Legend + facing reminder
                minimal_lines.extend(
                    [
                        "",
                        "Legend:",
                        "0 - walkable path",
                        "1 - wall / obstacle",
                        "2 - sprite (NPC)",
                        "3 - player facing up",
                        "4 - player facing down",
                        "5 - player facing left",
                        "6 - player facing right",
                        "",
                        "Pay close attention to which way you're facing, since that will affect how many button presses you need to accurately get where you're going. If your first press changes direction, it does not move you.",
                    ]
                )

                # --------------------------------------------------
                # Door / warp hints (from emulator helper)
                # --------------------------------------------------
                try:
                    door_info = self.emulator._get_doors_info()  # noqa: SLF001 – internal helper ok
                    if door_info:
                        minimal_lines.append("\nDetected Doors / Warps on this screen:")
                        for dest, (x, y) in door_info:
                            if dest:
                                minimal_lines.append(f"- Door to {dest} at ({x}, {y})")
                            else:
                                minimal_lines.append(f"- Door / warp at ({x}, {y})")
                except Exception:
                    pass
                if prev_coords is not None and curr_coords is not None:
                    minimal_lines.append("\nHow your last move affected your position on this collision map:")
                    minimal_lines.append(f"{prev_coords} --({', '.join(buttons)})--> {curr_coords}")

                # === Suggest moves & bounds ===
                try:
                    # Parse first 9 lines of collision map into int grid
                    grid_rows = []
                    for line in collision_map.strip().split("\n"):
                        if not line or not line[0].isdigit():
                            break
                        grid_rows.append([int(x) for x in line.split()])

                    pr = pc = None
                    for r, row in enumerate(grid_rows):
                        for c, val in enumerate(row):
                            if val in (3, 4, 5, 6):
                                pr, pc = r, c
                                break
                        if pr is not None:
                            break

                    move_counts = {}
                    if pr is not None:
                        # Up
                        cnt = 0
                        for rr in range(pr - 1, -1, -1):
                            if grid_rows[rr][pc] == 0:
                                cnt += 1
                            else:
                                break
                        move_counts['up'] = cnt
                        # Down
                        cnt = 0
                        for rr in range(pr + 1, len(grid_rows)):
                            if grid_rows[rr][pc] == 0:
                                cnt += 1
                            else:
                                break
                        move_counts['down'] = cnt
                        # Left
                        cnt = 0
                        for cc in range(pc - 1, -1, -1):
                            if grid_rows[pr][cc] == 0:
                                cnt += 1
                            else:
                                break
                        move_counts['left'] = cnt
                        # Right
                        cnt = 0
                        for cc in range(pc + 1, len(grid_rows[0])):
                            if grid_rows[pr][cc] == 0:
                                cnt += 1
                            else:
                                break
                        move_counts['right'] = cnt

                        # Build suggestion lines (stored for later use)
                        suggestion_line: str | None = None  # Main line with moves
                        repeats_line: str | None = None     # Secondary hint line

                        suggestions: list[str] = []
                        for dir_, n in sorted(move_counts.items(), key=lambda x: -x[1]):
                            if n > 0:
                                suggestions.append(" ".join([dir_] * min(n, 5)))

                        if suggestions:
                            suggestions_str = ", ".join(suggestions)
                            suggestion_line = (
                                f"Your fastest available next valid moves are: {suggestions_str}"
                            )
                            repeats_line = (
                                "...or 5, or 8 of these in a row! Sometimes it takes more than 4, "
                                "try a variety of max numbers of repeated presses."
                            )

                        # If we successfully built the suggestion line, place it
                        # directly after the initial "Pressed buttons" line so it
                        # is visible to the LLM as early context before the
                        # collision map and other details.
                        # Defer adding suggestion_line; we'll place it
                        # immediately after the "Pressed buttons" entry later in
                        # the prompt‑construction logic so it remains adjacent
                        # regardless of prepended dialogue blocks.

                        # --- Compute bounds based on current location ---
                        # We cache bounds per location so they refresh whenever
                        # the player enters a new map (e.g. going indoors).

                        current_loc = self.emulator.get_location() or "Unknown"

                        if (
                            getattr(self, "_bounds_cache_loc", None) != current_loc
                            or getattr(self, "_bounds_cache_line", None) is None
                        ):
                            # Need to (re)compute bounds for this location
                            try:
                                reader = PokemonRedReader(self.emulator.pyboy.memory)
                                map_w = reader.read_map_width()
                                map_h = reader.read_map_height()

                                # Indoor maps are usually <= 40×40.  If either
                                # dimension exceeds that, we keep it because it
                                # is correct for large outdoor routes/towns.
                                if map_w and map_h:
                                    self._bounds_cache_line = (
                                        f"Coordinate bounds of walkable area: rows 0-{map_h-1}, cols 0-{map_w-1}"
                                    )
                                else:
                                    self._bounds_cache_line = None
                            except Exception as exc:
                                logger.warning(
                                    f"Failed to read map dimensions; will fall back to viewport bounds: {exc}"
                                )
                                self._bounds_cache_line = None

                            self._bounds_cache_loc = current_loc

                        bounds_line = self._bounds_cache_line

                        # If reading dimensions failed, maintain previous viewport‑based fallback
                        if bounds_line is None:
                            min_r = float("inf")
                            min_c = float("inf")
                            max_r = -1
                            max_c = -1
                            for rr, row_vals in enumerate(grid_rows):
                                for cc, val in enumerate(row_vals):
                                    if val == 0:
                                        min_r = min(min_r, rr)
                                        max_r = max(max_r, rr)
                                        min_c = min(min_c, cc)
                                        max_c = max(max_c, cc)

                            if min_r == float("inf"):
                                min_r = min_c = 0
                                max_r = len(grid_rows) - 1 if grid_rows else 0
                                max_c = len(grid_rows[0]) - 1 if grid_rows and grid_rows[0] else 0

                            bounds_line = (
                                f"Coordinate bounds of walkable area: rows {min_r}-{max_r}, cols {min_c}-{max_c}"
                            )
                except Exception:
                    pass

            # Add coordinate bounds line under a clear heading
            if 'bounds_line' in locals() and bounds_line:
                minimal_lines.append("\n## Navigation Options")
                minimal_lines.append(bounds_line)

            # Effect line already added earlier

            # Dialog bullet & guidance or planning prompt
            if in_dialog:
                cleaned_curr_dialog = _clean_dialog(curr_dialog)
                minimal_lines.append(f"- Dialog: {cleaned_curr_dialog}")
                minimal_lines.append(
                    "\nYou are in a dialogue or menu. Press A to advance text, or use the D‑pad then A. Which button will you press next?"
                )
            else:
                # Instruction paragraph (appears directly after Navigation
                # Options section, heading will be added later)
                minimal_lines.append(
                    "Think carefully about the chat history and progress you’ve made in the last 10 moves or so. "
                    "Press more than one button unless you’re trying to be careful. Focus on speed run progress above everything else. "
                    "Which sequence of buttons will you press next?"
                )

            # Append fastest moves suggestion lines at strategic positions
            if 'suggestion_line' in locals() and suggestion_line:
                # Insert both suggestion_line and (optionally) repeats_line
                # immediately after the "Pressed buttons:" entry so they are
                # close to the user’s recent action.
                for idx, ln in enumerate(minimal_lines):
                    if ln.startswith("Pressed buttons:"):
                        # Insert environment header then suggestion lines
                        minimal_lines.insert(idx + 1, env_header_line)
                        if env_change_line:
                            minimal_lines.insert(idx + 2, env_change_line)
                            insert_pos = idx + 3
                        else:
                            insert_pos = idx + 2
                        minimal_lines.insert(insert_pos, suggestion_line)
                        if 'repeats_line' in locals() and repeats_line:
                            minimal_lines.insert(insert_pos + 1, repeats_line)
                        break

                # No duplication of env header & suggestions at bottom to keep
                # prompt concise and avoid repeating location header.

            # Final instruction
            if 'bounds_line' in locals():
                minimal_lines.append(
                    "\nReply with one of these fastest available sets of buttons, or call the \"navigate_to\" tool with a coordinate point in "
                    + bounds_line.split(':')[-1].strip() + " now."
                )

                # Add explicit Next Steps heading before repeated bounds & guidance
                minimal_lines.append("\n## Next Steps")

                # --- Extra guidance lines requested by UX tweak ---
                # Re‑iterate the bounds with an encouragement to try navigate_to.
                try:
                    import re
                    m = re.search(r"rows (\d+)-(\d+), cols (\d+)-(\d+)", bounds_line)
                    if m:
                        r_lo, r_hi, c_lo, c_hi = m.groups()
                        minimal_lines.append(
                            f"\nAgain, the coordinate bounds are rows {r_lo}-{r_hi}, cols {c_lo}-{c_hi} "
                            f"meaning you can call navigate_to ({r_lo},{c_lo}) all the way to ({r_hi},{c_hi}) indoors and outdoors, "
                            "even if you’ve had trouble with it before! Try it now."
                        )
                except Exception:
                    # Fallback: repeat bounds_line verbatim
                    minimal_lines.append(
                        "\nAgain, the coordinate bounds are " + bounds_line.split(':',1)[-1].strip() +
                        " – feel free to use navigate_to anywhere within that box!"
                    )

            # Add an additional suggestion for what to do if the agent is stuck
            if 'suggestion_line' in locals() and suggestion_line:
                moves_text = suggestion_line.split(':', 1)[-1].strip()
                minimal_lines.append(
                    (
                        "\nIf you are stuck, try calling the button press tool with exactly one of the fastest available next valid moves for a 4-8 repeats: "
                        + moves_text
                    )
                )

            # Inject congratulations line again just before the final nudge
            if env_change_line:
                minimal_lines.append(env_change_line)

            # --- Final nudge line (skip when a dialogue is visible to avoid confusion) ---
            if not in_dialog:
                minimal_lines.append(f"\nMake the next best move in {loc} now.")

            unified_text = "\n".join(minimal_lines)

            # Save minimal version for system prompt context
            self.latest_game_state = unified_text

            # Save as latest game state for system prompt
            self.latest_game_state = unified_text

            # Store for inclusion in future system prompts
            self.latest_game_state = unified_text

            # Log state & collision
            logger.info("[Memory State after action]")
            logger.info(memory_info)
            if collision_map:
                logger.info(f"[Collision Map after action]\n{collision_map}")

            wrapper = {"type": "tool_result", "tool_use_id": tool_call.id, "raw_output": result}
            image_block = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_b64,
                },
            }
            text_block = {"type": "text", "text": unified_text}
            return [wrapper, image_block, text_block]
        elif tool_name == "navigate_to":
            # If dialog is visible we should not attempt navigation – the D‑pad
            # will not move the player while a menu/text box is on‑screen.
            active_dialog_raw = (self.emulator.get_active_dialog() or "").strip()
            if active_dialog_raw:
                warning_msg = (
                    "Cannot navigate while a dialogue/menu is active. "
                    "Dismiss the dialog with press_buttons (typically 'a') first, "
                    "then try navigate_to again."
                )

                wrapper = {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "raw_output": warning_msg,
                }
                return [wrapper, {"type": "text", "text": warning_msg}]

            row = tool_input["row"]
            col = tool_input["col"]
            # Capture coordinates before navigation
            try:
                prev_coords_nav = self.emulator.get_coordinates()
            except Exception:
                prev_coords_nav = None

            logger.info(
                f"[Navigation] Navigating to: ({row}, {col}) | Prev coords: {prev_coords_nav}"
            )
            
            status, path = self.emulator.find_path(row, col)
            navigation_failed = False
            if path:
                for direction in path:
                    self.emulator.press_buttons([direction], True)
                result = (
                    f"Navigation to ({row}, {col}) successful: followed path with {len(path)} steps"
                )
            else:
                navigation_failed = True
                # Use status message returned but include attempted coords
                result = f"Navigation to ({row}, {col}) failed: {status}"
            
            # Capture coordinates after navigation
            try:
                curr_coords_nav = self.emulator.get_coordinates()
            except Exception:
                curr_coords_nav = None

            # screenshot
            screenshot = self.emulator.get_screenshot_with_overlay()
            screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)

            memory_info = self.emulator.get_state_from_memory()
            collision_map = self.emulator.get_collision_map()

            move_line = f"Navigation to ({row}, {col}) result: {result}"

            if navigation_failed:
                move_line += (
                    "\nIt looks like those coordinates are not reachable from your current "
                    "location. Try another set of coordinates within the valid bounds "
                    "shown below, or explore manually with press_buttons first to reach "
                    "a nearer location."
                )

            # ----------------------------------------------------------------
            # Build final prompt lines, beginning with recent dialogue history
            # ----------------------------------------------------------------
            lines = []
            if self._dialog_history:
                lines.append("## Recent Dialogue (last ≈50 lines)")
                lines.extend(list(self._dialog_history))
                lines.append("")

            lines.append(move_line)

            lines.append("\nBelow is the new game state generated by your actions after this navigation.\n")

            # Build custom game state section (reuse memory_info but inject details)
            mem_lines = memory_info.strip().split("\n")
            new_mem_lines: list[str] = []
            for line in mem_lines:
                if line.startswith("Dialog:"):
                    line = "- " + line
                if line.startswith("- Coordinates:"):
                    continue
                new_mem_lines.append(line)
                if prev_coords_nav is not None and curr_coords_nav is not None and line.startswith("- Current Environment:"):
                    new_mem_lines.append(f"- Previous Coordinates: {prev_coords_nav}")
                    new_mem_lines.append(f"- Path Followed: {', '.join(path)}")
                    new_mem_lines.append(f"- Current Coordinates: {curr_coords_nav}")

            lines.extend(new_mem_lines)

            dialog_after = (self.emulator.get_active_dialog() or "").strip()

            # Track dialog line if present
            if dialog_after:
                cleaned_dialog_after = _clean_dialog(dialog_after)
                if cleaned_dialog_after:
                    self._dialog_history.append(cleaned_dialog_after)

            if collision_map and not dialog_after:
                lines.append("\n## Collision Map")
                if curr_coords_nav is not None:
                    lines.append(f"Current Collision Map At {curr_coords_nav}:")
                lines.append("")
                lines.append(collision_map.strip())
                if prev_coords_nav is not None and curr_coords_nav is not None:
                    lines.append("\nHow your move affected this collision map:")
                    lines.append(f"{prev_coords_nav} --({', '.join(path)})--> {curr_coords_nav}")

                # Add door / warp hints
                try:
                    door_info = self.emulator._get_doors_info()  # noqa: SLF001
                    if door_info:
                        lines.append("\nDetected Doors / Warps on this screen:")
                        for dest, (x, y) in door_info:
                            if dest:
                                lines.append(f"- Door to {dest} at ({x}, {y})")
                            else:
                                lines.append(f"- Door / warp at ({x}, {y})")
                except Exception:
                    pass

                # Include valid move suggestions
                try:
                    moves = self.emulator.get_valid_moves()
                    if moves:
                        lines.append(
                            "\nYour Current Valid Moves: "
                            + ", ".join(moves)
                            + " (Note that the only time this valid moves parser will not be accurate is as you pass through doors, you have to press through the wall to continue. Consider pressing several of these according to your collision map!)"
                        )
                except Exception:
                    pass

            if dialog_after:
                lines.append(
                    f"\n[Dialog Visible] {dialog_after}\n\nYou are in a dialogue/menu. Press A or use the D‑pad to navigate choices, then A. "
                    "Generate the appropriate press_buttons tool call to advance."
                )
            else:
                lines.append(
                    "\nThink about the chat history and your initial instructions. "
                    "What should you try next to continue toward your objective? Generate the next tool call."
                )

            unified_text = "\n".join(lines)

            # logging
            logger.info("[Memory State after action]")
            logger.info(memory_info)
            if collision_map:
                logger.info(f"[Collision Map after action]\n{collision_map}")

            wrapper = {"type": "tool_result", "tool_use_id": tool_call.id, "raw_output": result}
            image_block = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_b64,
                },
            }
            text_block = {"type": "text", "text": unified_text}
            return [wrapper, image_block, text_block]
        else:
            logger.error(f"Unknown tool called: {tool_name}")
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [
                    {"type": "text", "text": f"Error: Unknown tool '{tool_name}'"}
                ],
            }

    def step(self):
        """Execute a single step of the agent's decision-making process."""
        try:
            # Flag to ensure sidebar (last_message) is updated at most once per
            # step to avoid duplicate "Pressed buttons …" lines.
            self._sidebar_updated_this_step = False
            # --------------------------------------------------------------
            # 1. If the most recent assistant message contains tool calls
            #    which have not yet been executed (i.e. the very next
            #    message is *not* a user/tool_result), execute them before
            #    making another LLM request. This prevents the situation
            #    where we send an unresolved function_call back to the
            #    Responses API and receive the 400 error
            #    "No tool output found for function call ...".
            # --------------------------------------------------------------

            if (
                len(self.message_history) >= 1
                and self.message_history[-1].get("role") == "assistant"
            ):
                pending_tool_calls = []
                for blk in self.message_history[-1].get("content", []):
                    if not isinstance(blk, dict):
                        continue
                    if blk.get("type") == "tool_use":
                        # Build a minimal _OpenAIBlock‑like shim so we can
                        # reuse process_tool_call(). Only fields name, input,
                        # id are needed.
                        class _TC:
                            pass

                        tc = _TC()
                        tc.type = "tool_use"
                        tc.name = blk.get("name")
                        tc.input = blk.get("input", {})
                        tc.id = blk.get("id")
                        pending_tool_calls.append(tc)

                # Detect if corresponding user/tool_result already exists
                if pending_tool_calls:
                    # Look at next message (if any) for tool_result id match
                    tool_result_present = False
                    if len(self.message_history) >= 2 and self.message_history[-2].get("role") == "user":
                        for tr_blk in self.message_history[-2].get("content", []):
                            if (
                                tr_blk.get("type") == "tool_result"
                                and tr_blk.get("tool_use_id") == pending_tool_calls[0].id
                            ):
                                tool_result_present = True
                                break

                    if not tool_result_present:
                        # Execute and append results
                        tool_results = [self.process_tool_call(tc) for tc in pending_tool_calls]
                        self.message_history.append({"role": "user", "content": tool_results})
                        # Extract plain text for UI
                        try:
                            texts = []
                            for tr in tool_results:
                                blocks_iter = tr if isinstance(tr, list) else [tr]
                                for block in blocks_iter:
                                    if (
                                        isinstance(block, dict)
                                        and block.get("type") == "text"
                                    ):
                                        texts.append(block.get("text", ""))
                            self.last_tool_message = "\n".join(texts).strip()
                        except Exception:
                            self.last_tool_message = None

                        # After executing pending tool calls we can safely
                        # return so the caller can invoke step() again which
                        # will now include the tool_result.
                        self._trim_history()
                        return
            messages = copy.deepcopy(self.message_history)
            # refresh system prompt text in history before copy for logging & LLM
            self._refresh_system_prompt_in_history()
            messages = copy.deepcopy(self.message_history)

            # -----------------------------------------------------------------
            # Detect environment change early and immediately summarize history
            # so the first turn in a new area starts with compact context.
            # -----------------------------------------------------------------

            try:
                curr_raw_loc = self.emulator.get_location() or "Unknown"
                curr_loc_norm = self._normalize_location(curr_raw_loc)
                if getattr(self, "_last_env", None) and curr_loc_norm != self._last_env:
                    logger.info(f"[Agent] Environment changed: {self._last_env} -> {curr_loc_norm}. Generating summary.")
                    try:
                        self.summarize_history()
                        # Refresh message copies after summary
                        self._refresh_system_prompt_in_history()
                        messages = copy.deepcopy(self.message_history)
                    except Exception as exc:
                        logger.warning(f"Failed to auto‑summarize on env change: {exc}")
                self._last_env = curr_loc_norm
            except Exception as exc:
                logger.warning(f"Env change detection failed: {exc}")
            # -----------------------------------------------------------------
            # Attach current screenshot & state IF the previous user message was
            # *not* a tool_result (i.e. no image/state already supplied).
            # This avoids duplicating information because process_tool_call()
            # already appends an image + state inside the tool_result content.
            # -----------------------------------------------------------------

            last_user = self.message_history[-1] if self.message_history else None
            skip_new_state = False
            if last_user and last_user.get("role") == "user":
                content_blocks = last_user.get("content", [])
                if isinstance(content_blocks, list):
                    for blk in content_blocks:
                        if isinstance(blk, dict) and blk.get("type") == "tool_result":
                            skip_new_state = True
                            break

            # Attach current screenshot as ground‑truth observation when needed
            try:
                state_user_blocks = []
                if not skip_new_state:
                    # Capture and encode the screenshot
                    screenshot_img = self.emulator.get_screenshot_with_overlay() if self.use_overlay else self.emulator.get_screenshot()
                    screenshot_b64 = get_screenshot_base64(screenshot_img, upscale=2)
                    image_block = {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}
                    }
                    state_user_blocks.append(image_block)

                # Add textual game state when we didn't skip state attachment
                if not skip_new_state:
                    loc = self.emulator.get_location() or "Unknown"
                    coords = self.emulator.get_coordinates()
                    dialog = self.emulator.get_active_dialog() or "None"

                    # Attempt to detect current menu cursor based on '►' marker
                    cursor_info = ""
                    if "►" in dialog:
                        try:
                            after = dialog.split("►", 1)[1]
                            # The selected word is first token after arrow
                            selected = after.strip().split()[0]
                            cursor_info = f" | Cursor: {selected.upper()} selected"
                        except Exception:
                            pass

                    moves = ",".join(self.emulator.get_valid_moves() or [])

                    # --- Detailed state block ---
                    state_snippet = (
                        f"[Game State] Env: {loc} | Coords: {coords} | Dialog: {dialog}{cursor_info} | ValidMoves: {moves}"
                    )
                    # Full memory dump (player stats etc.)
                    try:
                        memory_info = self.emulator.get_state_from_memory()
                    except Exception as exc:
                        logger.warning(f"Failed to fetch memory state: {exc}")
                        memory_info = None
                    # Collision map string (may be None if not detectable)
                    try:
                        collision_map = self.emulator.get_collision_map()
                    except Exception as exc:
                        logger.warning(f"Failed to fetch collision map: {exc}")
                        collision_map = None

                    combined_parts = [state_snippet]
                    if memory_info:
                        combined_parts.append(memory_info)
                    if collision_map:
                        combined_parts.append("[Collision Map]\n" + collision_map)

                    unified_text = "\n".join(combined_parts)
                    state_user_blocks.append({"type": "text", "text": unified_text})
                # If we created any blocks, append as a single user message
                if state_user_blocks:
                    state_user_msg = {"role": "user", "content": state_user_blocks}
                    messages.append(state_user_msg)
                    self.message_history.append(state_user_msg)
                    self._trim_history()
            except Exception as e:
                logger.warning(f"Failed to attach screenshot to message: {e}")

            if len(messages) >= 3:
                if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) and messages[-1]["content"]:
                    messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
                
                if len(messages) >= 5 and messages[-3]["role"] == "user" and isinstance(messages[-3]["content"], list) and messages[-3]["content"]:
                    messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}

            # Periodic introspection question every N steps (skip on step 0)
            import random
            from agent.prompts import INTROSPECTION_PROMPTS
            self._step_count += 1
            introspection_due = self._step_count % self._introspection_every == 0

            if not skip_new_state:
                # We still need to prompt the model; append the instruction or
                # introspection text TO THE SAME unified block we already
                # created (last element of state_user_blocks).

                if state_user_blocks:
                    # the unified text block is the last entry
                    unified_dict = state_user_blocks[-1]
                    base_text = unified_dict.get("text", "")

                    if self._step_count % self._introspection_every == 0:
                        extra_text = random.choice(INTROSPECTION_PROMPTS)
                    else:
                        loc = self.emulator.get_location() or "Unknown"
                        coords = self.emulator.get_coordinates()

                        dialog_raw = self.emulator.get_active_dialog() or ""
                        dialog = dialog_raw.strip()

                        if dialog:
                            # Dialogue/menu specific guidance
                            extra_text = (
                                f"[Dialog Visible] {dialog}\n\nYou are in a dialogue or menu. "
                                "Press A repeatedly to advance the text, or use the D‑pad to move the cursor "
                                "between menu options then press A to select. Generate a press_buttons tool call "
                                "(mostly 'a' plus directional presses if needed) that will advance the dialogue or "
                                "make your selection."
                            )
                        else:
                            extra_text = (
                                f"[Game State] Env: {loc} | Coords: {coords} | Dialog: {dialog}\n\nDecide how you will navigate next based on this game state. Moving left or right is often what is most needed to advance productively, do not just go up and down. Please reply taking all of this history & your system prompt into full account now.\n\nThink carefully about what unique, new button press will help move you forward productively based on this new information about the game state.\n\nPlan your next button press based mostly on this new game state, not in reaction to what you've done earlier. Note that if you have gone up and down recently, you should try left and right now, and vice versa. Be very careful not to undo recent moves you've made by making their opposite, focus on compounding forwrd progress."
                            )

                    # merge
                    unified_dict["text"] = base_text + "\n\n" + extra_text
            elif introspection_due:
                # We skipped state attachment (because a tool_result already
                # provided image/state) but we still want an introspection
                # question every 5 steps. Create a minimal user message for it.
                if self._step_count % self._introspection_every == 0:
                    introspect_text = random.choice(INTROSPECTION_PROMPTS)
                    prompt_msg = {
                        "role": "user",
                        "content": [{"type": "text", "text": introspect_text}],
                    }
                    messages.append(prompt_msg)
                    self.message_history.append(prompt_msg)
                    self._trim_history()
            # Get model response
            # Create a deep copy for logging and redact image data (handles nested tool results)
            log_messages_copy = copy.deepcopy(messages)

            def _redact_images(obj):
                """Recursively walk *obj* (dict/list) and redact base64 image data."""
                if isinstance(obj, dict):
                    # If this dict is an image block with base64 source, redact
                    if obj.get("type") == "image":
                        src = obj.get("source")
                        if isinstance(src, dict) and src.get("type") == "base64" and "data" in src:
                            original_len = len(src.get("data", ""))
                            src["data"] = f"<base64_image_data_removed_for_log len={original_len}>"
                    # Recurse into all dict values
                    for v in obj.values():
                        _redact_images(v)
                elif isinstance(obj, list):
                    for item in obj:
                        _redact_images(item)

            for msg in log_messages_copy:
                _redact_images(msg)

            logger.info(
                "[Object] Current messages object being sent to LLM (images redacted):\n" + _pretty_json(log_messages_copy)
            )
            if self.provider == 'anthropic':
                # Include system prompt as first user message for extra clarity
                try:
                    messages.insert(0, {"role": "user", "content": [{"type": "text", "text": self._get_system_prompt()}]})
                except Exception:
                    pass
                # Anthropic/Claude API
                response = self.llm_client.messages.create(
                    model=self.model_name,
                    max_tokens=MAX_TOKENS,
                    system=self._get_system_prompt(),
                    messages=messages,
                    tools=AVAILABLE_TOOLS,
                    temperature=TEMPERATURE,
                )
                # Update last message with all Claude text blocks concatenated
                claude_texts = [block.text for block in response.content if block.type == "text"]
                if claude_texts:
                    if claude_texts:
                        self._update_sidebar(claude_texts[0])
                logger.info(f"Response usage: {response.usage}")
                # Log raw Anthropic/Claude response blocks
                try:
                    raw_blocks_repr = [
                        block.model_dump() if hasattr(block, "model_dump") else block.__dict__
                        for block in response.content
                    ]
                    logger.info(
                        "[Raw LLM Response] (Anthropic):\n" + _pretty_json(raw_blocks_repr)
                    )
                except Exception:
                    pass
                blocks = list(response.content)
            elif self.provider == 'openai':
                # OpenAI Responses API
                # Format tools for function calling
                formatted_tools = []
                for tool in AVAILABLE_TOOLS:
                    # Build a schema with all properties marked as required to satisfy OpenAI's schema enforcement
                    base_schema = tool["input_schema"]
                    params = {
                        "type": base_schema.get("type", "object"),
                        "properties": base_schema.get("properties", {}),
                        # OpenAI requires 'required' to include all property keys
                        "required": list(base_schema.get("properties", {}).keys()),
                        "additionalProperties": False,
                    }
                    formatted_tools.append({
                        "type": "function",
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": params,
                        "strict": True,
                    })
                # Build input payload for Responses API
                input_payload = self._format_input_for_openai(messages)
                # Build kwargs, allowing text‑only reflections on introspection turns
                api_kwargs = {
                    "model": self.model_name,
                    "input": input_payload,
                    "text": {"format": {"type": "text"}},
                    "reasoning": {"effort": "low", "summary": "detailed"},
                    "tools": formatted_tools,
                    "store": True,
                }
                if not introspection_due:
                    api_kwargs["tool_choice"] = "required"

                # Attempt the OpenAI call; on context‑length errors we will
                # summarize and retry once.
                max_retry_blocks = 4  # total attempts (initial + 3 retries)
                attempt = 0
                while True:
                    try:
                        response = self.llm_client.responses.create(**api_kwargs)
                        break  # success
                    except Exception as e:
                        attempt += 1
                        err_msg = str(e)
                        # Only special‑handle context length errors
                        if (
                            ("maximum context length" not in err_msg and "context length" not in err_msg)
                            or attempt >= max_retry_blocks
                        ):
                            raise  # re‑raise other errors or if out of retries

                        logger.warning(
                            f"Context length error on attempt {attempt}. Condensing history and retrying."
                        )

                        try:
                            # First two retries: summarize history once each time
                            if attempt <= 2:
                                self.summarize_history()
                            else:
                                # Further retries: aggressively trim oldest 50% of messages
                                keep = max(10, len(self.message_history) // 2)
                                self.message_history = self.message_history[-keep:]
                                self._refresh_system_prompt_in_history()
                        except Exception as se:
                            logger.error(f"Failed to condense history: {se}")
                            raise e

                        # Rebuild payload with shorter history and retry
                        api_kwargs["input"] = self._format_input_for_openai(self.message_history)
                        continue
                # Extract response data as dict or object
                raw_dict = None
                try:
                    raw_dict = response.to_dict_recursive()
                except Exception:
                    try:
                        raw_dict = response.model_dump()
                    except Exception:
                        pass
                # Decide data container
                if isinstance(raw_dict, dict):
                    resp_data = raw_dict.get("response", raw_dict)
                else:
                    resp_data = getattr(response, "response", response)
                # Log usage
                usage = None
                if isinstance(resp_data, dict):
                    usage = resp_data.get("usage", {})
                else:
                    usage = getattr(resp_data, "usage", {})
                logger.info(f"Response usage: {usage}")
                # Extract raw blocks from response: merge 'content' and 'output'
                raw_blocks = []
                if isinstance(resp_data, dict):
                    raw_blocks.extend(resp_data.get("content") or [])
                    raw_blocks.extend(resp_data.get("output") or [])
                else:
                    raw_blocks.extend(getattr(resp_data, "content", []) or [])
                    raw_blocks.extend(getattr(resp_data, "output", []) or [])
                # Flatten any 'message' blocks containing nested content
                flat_blocks = []
                for b in raw_blocks:
                    if isinstance(b, dict) and b.get("type") == "message" and isinstance(b.get("content"), list):
                        flat_blocks.extend(b.get("content", []))
                    else:
                        flat_blocks.append(b)
                # Normalize blocks into dicts
                norm_blocks = []
                for b in flat_blocks:
                    if isinstance(b, dict):
                        norm_blocks.append(b)
                    else:
                        try:
                            norm_blocks.append(
                                b.model_dump() if hasattr(b, 'model_dump') else b.dict()
                            )
                        except Exception:
                            norm_blocks.append({})
                # Wrap raw blocks into unified objects
                blocks = [_OpenAIBlock(b) for b in norm_blocks]

                # Log the raw response content before wrapping
                try:
                    logger.info(
                        "[Raw LLM Response] (OpenAI):\n" + _pretty_json(norm_blocks)
                    )
                except Exception:
                    pass
            else:
                logger.error(f"Unsupported provider: {self.provider}")
                raise ValueError(f"Unsupported provider: {self.provider}")

            # Update last_message for OpenAI responses
            if self.provider == 'openai':
                texts = [block.text for block in blocks if block.type == "text"]
                if texts:
                    if texts:
                        self._update_sidebar(texts[0])
            # Extract tool calls and display reasoning
            tool_calls = [block for block in blocks if block.type == "tool_use"]
            for block in blocks:
                if block.type == "text":
                    logger.info(f"[Text] {block.text}")
                elif block.type == "tool_use":
                    logger.info(f"[Tool] Using tool: {block.name}")

            # ------------------------------------------------------------
            # Persist assistant reply *before* we execute any tool so that
            # its reasoning is always part of future context, even when no
            # tool is called.
            # ------------------------------------------------------------
            self._append_assistant_message(blocks)
            # One‑off early summary once 5 steps have elapsed
            if not self._initial_summary_done and self._step_count >= 5:
                self.summarize_history()
                self._initial_summary_done = True

            # Periodic summary every _summary_every steps
            if self._step_count % self._summary_every == 0:
                self.summarize_history()

            # Process tool calls (if any)
            if tool_calls:
                # Execute tools and create tool results
                tool_results = [self.process_tool_call(tc) for tc in tool_calls]
                # Append full wrappers (each contains content list) so
                # tool_use_id mapping stays intact.
                # Flatten each result list so wrapper, image, text are all
                # siblings inside the user content array.
                flat_blocks: list[dict] = []
                for res in tool_results:
                    if isinstance(res, list):
                        flat_blocks.extend(res)
                    else:
                        flat_blocks.append(res)
                self.message_history.append({"role": "user", "content": flat_blocks})

            # Clean up old screenshots except latest
            # Remove images from all but the very latest user message
            self._prune_old_images(keep_last=1)
            # Compact history likewise keeping the freshest context intact
            self._compact_history(keep_last=1)
            self._trim_history()

            # Extract tool result summary for UI display, if any
            try:
                texts = []
                for tr in tool_results if tool_calls else []:
                    # Flatten potential nested list from process_tool_call
                    for block in (tr if isinstance(tr, list) else [tr]):
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                        ):
                            texts.append(block.get("text", ""))
                if texts:
                    full_msg = "\n".join(texts).strip()
                    self.last_tool_message = full_msg
                    # Sidebar should only see the first meaningful line to
                    # avoid flooding it with the entire prompt. Use first line
                    # up to 120 chars.
                    first_line = full_msg.split("\n", 1)[0].strip()[:120]
                    if not self._sidebar_updated_this_step:
                        self._update_sidebar(first_line)
                        self._sidebar_updated_this_step = True
                else:
                    self.last_tool_message = None
            except Exception:
                self.last_tool_message = None
                # (summary handled globally after assistant message)

        except Exception as e:
            logger.error(f"Non‑fatal error inside step (continuing): {e}")
            import traceback, sys
            logger.debug(traceback.format_exc())

    def run(self, num_steps=1):
        """Main agent loop.

        Args:
            num_steps: Number of steps to run for
        """
        logger.info(f"Starting agent loop for {num_steps} steps")

        steps_completed = 0
        while self.running and steps_completed < num_steps:
            try:
                self.step()
                steps_completed += 1
                logger.info(f"Completed step {steps_completed}/{num_steps}")

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                self.running = False
            except Exception as e:
                # Log the error but continue. Increment steps_completed so we
                # do not get stuck in an infinite loop waiting to reach
                # num_steps.
                logger.error(f"Non‑fatal error in agent loop (continuing): {e}")
                import traceback
                logger.debug(traceback.format_exc())
                # Do NOT increment steps_completed here; we want to retry the
                # same logical turn after handling the error.
                continue

        if not self.running:
            self.emulator.stop()

        return steps_completed

    def summarize_history(self):
        """Generate a summary of the conversation history and replace the history with just the summary."""
        # Clear last tool message when summarizing
        self.last_tool_message = None
        # Support both Anthropic and OpenAI summarization
        if self.provider == "openai":
            logger.info("[Agent] Generating conversation summary via OpenAI...")
            # Prepare messages with appended summary prompt
            msgs = copy.deepcopy(self.message_history)

            # --- Attach latest screenshot & state text so the summary can reference them ---
            try:
                screenshot_img = self.emulator.get_screenshot_with_overlay() if self.use_overlay else self.emulator.get_screenshot()
                screenshot_b64 = get_screenshot_base64(screenshot_img, upscale=2)
                img_block = {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64},
                }
                loc = self.emulator.get_location() or "Unknown"
                coords = self.emulator.get_coordinates()
                dialog = self.emulator.get_active_dialog() or "None"
                state_line = f"[Game State] Env: {loc} | Coords: {coords} | Dialog: {dialog}\n\nWrite your summary now, not primarily about this game state specifically, but rather about how the entire chat history has led to this state, and what changes in approach might be needed to advance more quickly, make sure your reply includes any insights from previous history prompts you see in the chat history above."
                msgs.append({
                    "role": "user",
                    "content": [img_block, {"type": "text", "text": state_line}],
                })
            except Exception:
                pass
            # Create a deep copy for logging and redact image data (handles nested tool results)
            log_messages_copy = copy.deepcopy(msgs)
            for msg in log_messages_copy:
                if isinstance(msg.get("content"), list):
                    for block in msg["content"]:
                        # Check top-level image blocks
                        if block.get("type") == "image" and isinstance(block.get("source"), dict) and block["source"].get("type") == "base64":
                            if "data" in block["source"]:
                                original_len = len(block['source'].get('data', ''))
                                block["source"]["data"] = f"<base64_image_data_removed_for_log len={original_len}>"
                        # Check image blocks nested inside tool_result content
                        elif block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                            for nested_block in block["content"]:
                                if nested_block.get("type") == "image" and isinstance(nested_block.get("source"), dict) and nested_block["source"].get("type") == "base64":
                                    if "data" in nested_block["source"]:
                                        original_len = len(nested_block['source'].get('data', ''))
                                        nested_block["source"]["data"] = f"<nested_base64_image_data_removed_for_log len={original_len}>"

            logger.info(
                "[Object] Current messages object being sent to OpenAI for Summary (images redacted):\n" + _pretty_json(log_messages_copy)
            )
            msgs.append({"role": "user", "content": [{"type": "text", "text": SUMMARY_PROMPT}]})
            payload = self._format_input_for_openai(msgs)
            resp = self.llm_client.responses.create(
                model=self.model_name,
                input=payload,
                text={"format": {"type": "text"}},
                reasoning={"effort": "low", "summary": "detailed"},
                store=True,
            )
            # DEBUG: log the raw JSON of the summary response (trimmed)
            try:
                raw_json = (
                    resp.to_dict_recursive()
                    if hasattr(resp, "to_dict_recursive")
                    else resp.model_dump()
                )
            except Exception:
                raw_json = str(resp)
            try:
                logger.info("[DEBUG] SUMMARY_RAW_OUTPUT (OpenAI) %s", json.dumps(raw_json, ensure_ascii=False)[:3000])
            except Exception:
                pass
            # Normalize response into blocks (reuse openai extraction pattern)
            raw_dict = None
            try:
                raw_dict = resp.to_dict_recursive()
            except Exception:
                try:
                    raw_dict = resp.model_dump()
                except Exception:
                    raw_dict = None
            if isinstance(raw_dict, dict):
                resp_data = raw_dict.get("response", raw_dict)
            else:
                resp_data = getattr(resp, "response", resp)
            raw_blocks = []
            if isinstance(resp_data, dict):
                raw_blocks.extend(resp_data.get("content") or [])
                raw_blocks.extend(resp_data.get("output") or [])
            else:
                raw_blocks.extend(getattr(resp_data, "content", []) or [])
                raw_blocks.extend(getattr(resp_data, "output", []) or [])
            flat_blocks = []
            for b in raw_blocks:
                if isinstance(b, dict) and b.get("type") == "message" and isinstance(b.get("content"), list):
                    flat_blocks.extend(b.get("content", []))
                else:
                    flat_blocks.append(b)
            norm_blocks = []
            for b in flat_blocks:
                if isinstance(b, dict):
                    norm_blocks.append(b)
                else:
                    try:
                        norm_blocks.append(b.model_dump() if hasattr(b, 'model_dump') else b.dict())
                    except Exception:
                        norm_blocks.append({})
            # Wrap into OpenAIBlock objects and extract text
            blocks = [_OpenAIBlock(b) for b in norm_blocks]
            # Include only visible assistant text, exclude any internal reasoning blocks
            texts = [
                block.text for block in blocks
                if block.type == "text" and not getattr(block, "is_reasoning", False)
            ]
            summary_text = " ".join(texts).strip()
            # If the model provided extra explanation before the final summary, keep
            # only the part starting from the explicit heading.
            def _extract_summary(txt: str):
                marker = "CONVERSATION HISTORY SUMMARY"
                idx = txt.upper().find(marker)
                if idx != -1:
                    return txt[idx:].strip()
                return txt
            summary_text = _extract_summary(summary_text)
            logger.info(f"[Agent] Conversation Summary: {summary_text}")
            # Replace and condense history to summary
            if summary_text.upper().startswith("CONVERSATION HISTORY SUMMARY"):
                summary_msg = summary_text
            else:
                summary_msg = f"CONVERSATION HISTORY SUMMARY: {summary_text}"
            self.message_history.append({"role": "assistant", "content": [{"type": "text", "text": summary_msg}]})
            self._trim_history()
            self.last_message = summary_msg

            # Store for inclusion in future system prompts
            self.history_summary = summary_msg

            # --- Persist emulator save state for debugging/playback ---
            try:
                run_dir = getattr(self.app.state, 'run_log_dir', None)
                if run_dir:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    loc = (self.emulator.get_location() or 'unknown').lower().replace(' ', '_')
                    save_path = os.path.join(
                        run_dir,
                        'history_saves',
                        f'summary_{ts}_{loc}.state',
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    self.emulator.save_state(save_path)
                    logger.info(f"[Agent] Saved state snapshot to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save summary state: {e}")

            return
        elif self.provider != "anthropic":
            logger.warning("Unsupported provider for summarization; skipping.")
            return
        logger.info("[Agent] Generating conversation summary...")
        
        # Get a new screenshot and game state for the summary
        screenshot = self.emulator.get_screenshot_with_overlay() if self.use_overlay else self.emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)

        loc = self.emulator.get_location() or "Unknown"
        coords = self.emulator.get_coordinates()
        dialog = self.emulator.get_active_dialog() or "None"
        state_line = f"[Game State] Env: {loc} | Coords: {coords} | Dialog: {dialog}"

        # Create messages for the summarization request - include entire history plus latest observation
        messages = copy.deepcopy(self.message_history)
        if len(messages) >= 3:
            if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) and messages[-1]["content"]:
                messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
            
            if len(messages) >= 5 and messages[-3]["role"] == "user" and isinstance(messages[-3]["content"], list) and messages[-3]["content"]:
                messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        messages += [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": SUMMARY_PROMPT,
                    }
                ],
            }
        ]
        
        # Include system prompt as first user message for extra clarity
        try:
            messages.insert(0, {"role": "user", "content": [{"type": "text", "text": self.system_prompt}]})
        except Exception:
            pass
        # Get summary from Claude
        response = self.anthropic_client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=self.system_prompt,
            messages=messages,
            temperature=TEMPERATURE
        )
        # DEBUG: log raw content blocks returned by Claude for summary (trimmed)
        try:
            raw_blocks_debug = [
                b.model_dump() if hasattr(b, "model_dump") else getattr(b, "__dict__", str(b))
                for b in response.content
            ]
            logger.info("[DEBUG] SUMMARY_RAW_OUTPUT (Anthropic) %s", json.dumps(raw_blocks_debug, ensure_ascii=False)[:3000])
        except Exception:
            pass
        
        # Extract only visible assistant text (Claude returns only text blocks)
        summary_text = " ".join([block.text for block in response.content if block.type == "text"]).strip()
        # Keep only explicit summary section if the model included extra narrative
        def _extract_summary(txt: str):
            marker = "CONVERSATION HISTORY SUMMARY"
            idx = txt.upper().find(marker)
            if idx != -1:
                return txt[idx:].strip()
            return txt
        summary_text = _extract_summary(summary_text)
        
        logger.info(f"[Agent] Game Progress Summary:")
        logger.info(f"{summary_text}")
        
        # Replace message history with just the summary, avoid duplicate heading
        if summary_text.upper().startswith("CONVERSATION HISTORY SUMMARY"):
            summary_msg = summary_text
        else:
            summary_msg = (
                f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): "
                f"{summary_text}"
            )
        # Append summary assistant message
        self.message_history.append({"role": "assistant", "content": [{"type": "text", "text": summary_msg}]})
        self._trim_history()
        self.last_message = summary_msg
        # Store latest summary text for dynamic system prompt
        self.history_summary = summary_msg
        logger.info("[Agent] Summary appended to message history.")

        # Save emulator state snapshot
        try:
            run_dir = getattr(self.app.state, 'run_log_dir', None)
            if run_dir:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                loc = (self.emulator.get_location() or 'unknown').lower().replace(' ', '_')
                save_path = os.path.join(
                    run_dir,
                    'history_saves',
                    f'summary_{ts}_{loc}.state',
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.emulator.save_state(save_path)
                logger.info(f"[Agent] Saved state snapshot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save summary state: {e}")
        
    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.stop()


if __name__ == "__main__":
    # Get the ROM path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rom_path = os.path.join(os.path.dirname(current_dir), "pokemon.gb")

    # Create and run agent
    agent = SimpleAgent(rom_path)

    try:
        steps_completed = agent.run(num_steps=10)
        logger.info(f"Agent completed {steps_completed} steps")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    finally:
        agent.stop()