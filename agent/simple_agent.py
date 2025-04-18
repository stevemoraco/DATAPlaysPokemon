import base64
import copy
from collections import deque
import io
import logging
import os
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

from config import MAX_TOKENS, MODEL_NAME, TEMPERATURE
from agent.prompts import SYSTEM_PROMPT, SUMMARY_PROMPT
from agent.tools import AVAILABLE_TOOLS

from agent.emulator import Emulator
from anthropic import Anthropic
import json

# Wrapper for OpenAI Responses API blocks to match Anthropic-style blocks
class _OpenAIBlock:
    def __init__(self, data):
        btype = data.get("type")
        # Text output or summary text
        if btype in ("output_text", "summary_text"):
            self.type = "text"
            self.text = data.get("text", "")
        # High-level reasoning block: extract summary texts
        elif btype == "reasoning":
            self.type = "text"
            # summary is a list of summary_text blocks
            summary_list = data.get("summary") or []
            # Each summary item may be dict with 'text'
            lines = []
            for item in summary_list:
                if isinstance(item, dict) and item.get("text"):
                    lines.append(item.get("text"))
            # Join with newlines for display
            self.text = "\n".join(lines)
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
    
    # No return, just block wrapper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


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
    # Unified LLM invocation helper
    # ------------------------------------------------------------------

    def _call_llm(self, messages):
        """Call Anthropic Claude or OpenAI Responses API and return a list of
        normalized blocks.

        Args:
            messages (list[dict]): conversation history in Anthropic block
                format (as used in self.message_history).

        Returns:
            list: list of blocks where each block exposes at least `.type`
            plus other attributes depending on type.  For Anthropic we return
            Claude block objects directly; for OpenAI we wrap raw dicts in
            `_OpenAIBlock` for a compatible interface.
        """

        # --------------------------------------------------------------
        # Anthropic provider
        # --------------------------------------------------------------
        if self.provider == 'anthropic':
            response = self.llm_client.messages.create(
                model=self.model_name,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                messages=messages,
                tools=AVAILABLE_TOOLS,
                temperature=TEMPERATURE,
            )
            try:
                logger.info(f"[Anthropic] Usage: {response.usage}")
            except Exception:
                pass
            return list(response.content)

        # --------------------------------------------------------------
        # OpenAI provider (Responses API)
        # --------------------------------------------------------------
        if self.provider == 'openai':
            # Convert messages to OpenAI input format
            input_payload = self._format_input_for_openai(messages)

            # Map our tool schemas to OpenAI function objects
            formatted_tools = []
            for tool in AVAILABLE_TOOLS:
                base = tool["input_schema"]
                params = {
                    "type": base.get("type", "object"),
                    "properties": base.get("properties", {}),
                    "required": list(base.get("properties", {}).keys()),
                    "additionalProperties": False,
                }
                formatted_tools.append({
                    "type": "function",
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": params,
                    "strict": True,
                })

            response = self.llm_client.responses.create(
                model=self.model_name,
                input=input_payload,
                text={"format": {"type": "text"}},
                reasoning={"effort": "low", "summary": "auto"},
                tools=formatted_tools,
                tool_choice="auto",  # force model to choose a tool if appropriate
                store=True,
            )

            # Extract usage for logging where available
            try:
                raw = response.to_dict_recursive()
            except Exception:
                try:
                    raw = response.model_dump()
                except Exception:
                    raw = {}
            if isinstance(raw, dict):
                usage = raw.get("response", raw).get("usage", {})
                logger.info(f"[OpenAI] Usage: {usage}")

            # Pull blocks from response (content + output) and normalise
            try:
                resp_data = response.response  # type: ignore
            except AttributeError:
                resp_data = raw.get("response", {}) if isinstance(raw, dict) else response

            raw_blocks = []
            if isinstance(resp_data, dict):
                raw_blocks.extend(resp_data.get("content") or [])
                raw_blocks.extend(resp_data.get("output") or [])
            else:
                raw_blocks.extend(getattr(resp_data, "content", []) or [])
                raw_blocks.extend(getattr(resp_data, "output", []) or [])

            # Flatten nested message blocks
            flat = []
            for b in raw_blocks:
                if isinstance(b, dict) and b.get("type") == "message" and isinstance(b.get("content"), list):
                    flat.extend(b["content"])
                else:
                    flat.append(b)

            # Normalise to dict then wrap
            norm = []
            for b in flat:
                if isinstance(b, dict):
                    norm.append(b)
                else:
                    try:
                        norm.append(b.model_dump())
                    except Exception:
                        try:
                            norm.append(b.dict())
                        except Exception:
                            norm.append({"type": "text", "text": str(b)})

            return [_OpenAIBlock(b) for b in norm]

        raise ValueError(f"Unsupported provider: {self.provider}")
    def _format_input_for_openai(self, messages):
        """Translate internal message history into OpenAI Responses API input format."""
        payload = []
        # Add system prompt as first user message for extra clarity
        payload.append({
            "role": "user",
            "content": [
                {"type": "input_text", "text": self.system_prompt}
            ],
        })
        # Iterate through existing history messages
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])
            # Normalize to list of blocks
            # Ensure every block is a plain dict (not a provider‑specific object)
            if isinstance(content, list):
                blocks = []
                for bl in content:
                    if isinstance(bl, dict):
                        blocks.append(bl)
                    else:
                        # Attempt to unwrap provider block objects
                        if hasattr(bl, 'type') and bl.type == 'text':
                            blocks.append({"type": "text", "text": getattr(bl, 'text', str(bl))})
                        elif hasattr(bl, 'type') and bl.type == 'tool_use':
                            obj = {
                                "type": "tool_use",
                                "name": getattr(bl, 'name', ''),
                                "input": getattr(bl, 'input', {}),
                            }
                            if getattr(bl, 'id', None) is not None:
                                obj['id'] = bl.id
                            blocks.append(obj)
                        else:
                            # Fallback string representation
                            blocks.append({"type": "text", "text": str(bl)})
            else:
                blocks = [{"type": "text", "text": str(content)}]
            buffer = []
            for block in blocks:
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
        # Initialize Google GenAI client if API key provided; otherwise disable Google tools
        try:
            self.google_client = genai.Client()
        except Exception as e:
            logger.warning(f"Google GenAI client initialization failed: {e}. Google tools disabled.")
            self.google_client = None
        self.running = True
        self.message_history = [{"role": "user", "content": "You may now begin playing."}]
        self.max_history = max_history
        self.last_message = "Game starting..."  # Initialize last message
        self.last_tool_message = None  # Store latest tool result message for UI
        self.app = app  # Store reference to FastAPI app
        self.use_overlay = use_overlay  # Store overlay preference
        # Track last few coordinate positions to detect being stuck
        self.coord_history = deque(maxlen=5)
        
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
            # Execute the tool and capture raw output
            result = self.emulator.press_buttons(buttons, wait)
            
            # Get a fresh screenshot after executing the buttons with tile overlay
            screenshot = self.emulator.get_screenshot_with_overlay()
            screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
            
            # Get game state from memory after the action
            memory_info = self.emulator.get_state_from_memory()
            
            # Log the memory state after the tool call
            logger.info(f"[Memory State after action]")
            logger.info(memory_info)
            
            if os.getenv("DEBUG_COLLISION") == "1":
                collision_map = self.emulator.get_collision_map()
                if collision_map:
                    logger.info(f"[Collision Map after action]\n{collision_map}")
            
            # Return tool result (including raw output) as a dictionary
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "raw_output": result,
                "content": [
                    {"type": "text", "text": f"Pressed buttons: {', '.join(buttons)}"},
                    {"type": "text", "text": "\nHere is a screenshot of the screen after your button presses:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                ],
            }
        elif tool_name == "navigate_to":
            row = tool_input["row"]
            col = tool_input["col"]
            logger.info(f"[Navigation] Navigating to: ({row}, {col})")
            
            status, path = self.emulator.find_path(row, col)
            if path:
                for direction in path:
                    self.emulator.press_buttons([direction], True)
                result = f"Navigation successful: followed path with {len(path)} steps"
            else:
                result = f"Navigation failed: {status}"
            
            # Get a fresh screenshot after executing the navigation with tile overlay
            screenshot = self.emulator.get_screenshot_with_overlay()
            screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
            
            # Get game state from memory after the action
            memory_info = self.emulator.get_state_from_memory()
            
            # Log the memory state after the tool call
            logger.info(f"[Memory State after action]")
            logger.info(memory_info)
            
            collision_map = self.emulator.get_collision_map()
            if collision_map:
                logger.info(f"[Collision Map after action]\n{collision_map}")
            
            # Return tool result (including raw output) as a dictionary
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "raw_output": result,
                "content": [
                    {"type": "text", "text": f"Navigation result: {result}"},
                    {"type": "text", "text": "\nHere is a screenshot of the screen after navigation:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                ],
            }
        else:
            logger.error(f"Unknown tool called: {tool_name}")
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [
                    {"type": "text", "text": f"Error: Unknown tool '{tool_name}'"}
                ],
            }

    def _build_observation(self):
        """Return (screenshot_block, memory_block, optional warning_block)."""
        blocks = []
        # screenshot
        try:
            img = self.emulator.get_screenshot()
            b64 = get_screenshot_base64(img, upscale=2)
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            })
        except Exception as e:
            logger.warning(f"Screenshot error: {e}")

        # memory
        try:
            mem = self.emulator.get_state_from_memory()
            blocks.append({"type": "text", "text": mem})
        except Exception as e:
            logger.warning(f"Memory error: {e}")

        # stagnation warning
        try:
            coords = self.emulator.get_coordinates()
            self.coord_history.append(coords)
            if len(self.coord_history) >= 5 and all(c == coords for c in self.coord_history):
                blocks.append({"type": "text", "text": "WARNING: You haven't moved for several turns. Try another direction or navigate_to."})
        except Exception:
            pass
        return blocks

    def step(self):
        """Interactive step: loops until assistant returns pure text."""
        try:
            while True:
                # observation
                self.message_history.append({"role": "user", "content": self._build_observation()})

                # ask LLM
                assistant_blocks = self._call_llm(copy.deepcopy(self.message_history))

                # Debug: log block types returned
                try:
                    logger.info(f"Block types returned: {[getattr(b,'type', None) for b in assistant_blocks]}")
                except Exception:
                    pass

                # Store a JSON‑serialisable version of the assistant reply so
                # future calls to `_format_input_for_openai` don’t stumble on
                # provider‑specific objects.

                def _blk_to_dict(blk):
                    if isinstance(blk, dict):
                        return blk
                    if getattr(blk, 'type', None) == 'text':
                        return {"type": "text", "text": getattr(blk, 'text', '')}
                    if getattr(blk, 'type', None) == 'tool_use':
                        d = {
                            "type": "tool_use",
                            "name": getattr(blk, 'name', ''),
                            "input": getattr(blk, 'input', {}),
                        }
                        if getattr(blk, 'id', None) is not None:
                            d['id'] = blk.id
                        return d
                    # fallback – stringify
                    return {"type": "text", "text": str(blk)}

                assistant_content_serialisable = [_blk_to_dict(b) for b in assistant_blocks]

                self.message_history.append({"role": "assistant", "content": assistant_content_serialisable})

                tool_calls = [b for b in assistant_blocks if b.type == "tool_use"]
                if not tool_calls:
                    # store last message
                    self.last_message = "\n".join(b.text for b in assistant_blocks if b.type == "text").strip()
                    break

                # process tools
                results = [self.process_tool_call(tc) for tc in tool_calls]
                self.message_history.append({"role": "user", "content": results})
                # Execute another iteration to give the model fresh state
                continue

            # If we reach here we had a pure‑text reply and broke the inner
            # loop; step is complete.

            if self.provider == 'anthropic':
                # Summarize history if needed after tool result added
                if len(self.message_history) >= self.max_history:
                    self.summarize_history()

        except Exception as e:
            logger.error(f"Error in agent step: {e}")
            raise

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
                logger.error(f"Error in agent loop: {e}")
                raise e

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
            msgs.append({"role": "user", "content": [{"type": "text", "text": SUMMARY_PROMPT}]})
            payload = self._format_input_for_openai(msgs)
            resp = self.llm_client.responses.create(
                model=self.model_name,
                input=payload,
                text={"format": {"type": "text"}},
                reasoning={"effort": "high", "summary": "auto"},
                store=True,
            )
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
            texts = [block.text for block in blocks if block.type == "text"]
            summary_text = " ".join(texts).strip()
            logger.info(f"[Agent] Conversation Summary: {summary_text}")
            # Replace and condense history to summary
            summary_msg = f"CONVERSATION HISTORY SUMMARY: {summary_text}"
            self.message_history = [{"role": "user", "content": [{"type": "text", "text": summary_msg}]}]
            # Emit summary as next assistant message
            self.last_message = summary_msg
            return
        elif self.provider != "anthropic":
            logger.warning("Unsupported provider for summarization; skipping.")
            return
        logger.info("[Agent] Generating conversation summary...")
        
        # Get a new screenshot for the summary
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
        
        # Create messages for the summarization request - pass the entire conversation history
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
        
        # Extract the summary text
        summary_text = " ".join([block.text for block in response.content if block.type == "text"])
        
        logger.info(f"[Agent] Game Progress Summary:")
        logger.info(f"{summary_text}")
        
        # Replace message history with just the summary
        summary_msg = f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {summary_text}"
        self.message_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                    "text": summary_msg
                    },
                    {
                        "type": "text",
                        "text": "\n\nCurrent game screenshot for reference:"
                    },
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
                        "text": "You were just asked to summarize your playthrough so far, which is the summary you see above. You may now continue playing by selecting your next action."
                    },
                ]
            }
        ]
        
        # Emit summary as next assistant message
        self.last_message = summary_msg
        logger.info(f"[Agent] Message history condensed into summary.")
        
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