# Configuration for the application
MODEL_NAME = "claude-3-7-sonnet-20250219"
TEMPERATURE = 1.0
MAX_TOKENS = 4000

# When True, the higher‑level `navigate_to` tool is exposed to the LLM so it
# can issue coarse path‑finding commands instead of tediously sending dozens of
# D‑pad presses.  Almost all runs benefit from having it enabled, so it is now
# the default.
USE_NAVIGATOR = True