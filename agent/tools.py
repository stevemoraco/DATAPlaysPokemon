from config import USE_NAVIGATOR

# ---------------------------------------------------------------------------
# Base tool – always available
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = [
    {
        "name": "press_buttons",
        "description": (
            "Press a sequence of Game Boy buttons. Use this for menus or small "
            "movements; for long‑distance overworld travel prefer the higher‑level "
            "`navigate_to` tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "buttons": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "a",
                            "b",
                            "start",
                            "select",
                            "up",
                            "down",
                            "left",
                            "right",
                        ],
                    },
                    "description": (
                        "Buttons to press in order. MUST NOT be empty unless you "
                        "intend to call `navigate_to` instead."
                    ),
                },
                "wait": {
                    "type": "boolean",
                    "description": (
                        "Whether to wait briefly after each press. Defaults to true."
                    ),
                },
            },
            "required": ["buttons"],
        },
    },
]

# ---------------------------------------------------------------------------
# Optional path‑finding tool – included when USE_NAVIGATOR = True
# ---------------------------------------------------------------------------

if USE_NAVIGATOR:
    AVAILABLE_TOOLS.append(
        {
            "name": "navigate_to",
            "description": (
                "Navigate automatically to a tile on the 9×10 screen‑space grid. "
                "Top‑left is (0,0). Only usable while in the overworld."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "row": {
                        "type": "integer",
                        "description": "Row (0‑8)",
                    },
                    "col": {
                        "type": "integer",
                        "description": "Column (0‑9)",
                    },
                },
                "required": ["row", "col"],
            },
        }
    )