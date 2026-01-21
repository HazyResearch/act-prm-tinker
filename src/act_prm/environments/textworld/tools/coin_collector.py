"""
Tools and their descriptions for the `coin_collector` task
"""

TOOL_TO_TEMPLATE: dict[str, str] = {
    "goal": "goal",
    "go": "go {direction}",
    "take_coin": "take coin",
}

TOOL_DESCRIPTIONS = {
    "goal": {
        "type": "function",
        "name": "goal",
        "description": "Print the goal of this game.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "go": {
        "type": "function",
        "name": "go",
        "description": (
            "Move the player in a cardinal direction (north, east, south, or west). "
            "You can only move to directions indicated with an exit or a door."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "description": "Direction to move: one of 'north', 'east', 'south', 'west'.",
                    "enum": ["north", "east", "south", "west"],
                }
            },
            "required": ["direction"],
        },
    },
    "take_coin": {
        "type": "function",
        "name": "take_coin",
        "description": (
            "Win the game by taking the coin if you see it in the room."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
