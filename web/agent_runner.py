import logging
import os
from PIL import Image
import asyncio

logger = logging.getLogger(__name__)

async def run_agent(agent, num_steps, run_log_dir, send_game_updates, claude_logger):
    try:
        logger.info(f"Starting agent for {num_steps} steps")
        # Start continuous frame streamer for full framerate
        async def frame_streamer():
            try:
                while True:
                    frame = agent.get_frame()
                    await send_game_updates(frame, "")
                    await asyncio.sleep(1/30)  # ~30 FPS
            except asyncio.CancelledError:
                return
        frame_task = asyncio.create_task(frame_streamer())
        steps_completed = 0
        # Main agent loop
        while steps_completed < num_steps:
            # Pause handling
            while getattr(agent.app.state, 'is_paused', False):
                await asyncio.sleep(0.1)

            # --- 1. Run the agent's decision for this turn ---
            agent.step()
            steps_completed += 1

            # --- 2. Capture the new frame AFTER the action ---
            frame = agent.get_frame()
            frame_count = steps_completed
            frame_path = os.path.join(run_log_dir, "frames", f"frame_{frame_count:05d}.png")
            with open(frame_path, "wb") as f:
                f.write(frame)

            # --- 3. Compute environment info for sidebar ---
            try:
                loc = agent.emulator.get_location() or 'Unknown'
            except Exception:
                loc = 'Unknown'
            try:
                coords = agent.emulator.get_coordinates()
            except Exception:
                coords = (None, None)
            try:
                moves = agent.emulator.get_valid_moves()
            except Exception:
                moves = []
            try:
                dialog = agent.emulator.get_active_dialog() or 'None'
            except Exception:
                dialog = 'None'
            env_msg = (
                f"Current Player Environment: {loc}\n"
                f"Coordinates: ({coords[0]}, {coords[1]})\n"
                f"Valid Moves: {', '.join(moves)}\n"
                f"Dialog: {dialog}"
            )

            # --- 4. Send model "thought" (assistant reply) produced in this step ---
        message = agent.get_last_message() or ''
        thought_msg = message.strip()
            if thought_msg:
                claude_logger.info(thought_msg)
            # Send only if this thought differs from the previous message to
            # avoid duplicate sidebar entries (e.g. when the tool result text
            # echoes the same "Pressed buttons" line).
            if thought_msg != getattr(run_agent, "_last_sent_msg", None):
                await send_game_updates(frame, thought_msg, env_msg)
                run_agent._last_sent_msg = thought_msg
            # --- 5. If there is a tool‑result from THIS step, send a concise follow‑up ---
            if hasattr(agent, 'last_tool_message') and agent.last_tool_message:
                raw_tool_msg = agent.last_tool_message or ''

                # Keep only concise status lines (Pressed buttons / Navigation result)
                concise_lines = []
                for ln in raw_tool_msg.split('\n'):
                    s = ln.strip()
                    if s.lower().startswith(('pressed', 'navigation result', 'warning', 'blocked')):
                        concise_lines.append(s)
                concise_tool_msg = ' | '.join(concise_lines).strip()

                if concise_tool_msg and concise_tool_msg != getattr(run_agent, "_last_sent_msg", None):
                    claude_logger.info(concise_tool_msg)
                    tool_frame = agent.get_frame()
                    await send_game_updates(tool_frame, concise_tool_msg)
                    run_agent._last_sent_msg = concise_tool_msg

                # Clear so it is not resent next tick
                agent.last_tool_message = None
            # Control loop timing
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        logger.info("Agent task was cancelled")
        raise
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        raise
    finally:
        # Ensure frame streamer is stopped
        try:
            frame_task.cancel()
        except Exception:
            pass
        try:
            await frame_task
        except asyncio.CancelledError:
            pass
        logger.info(f"Agent completed {steps_completed} steps")