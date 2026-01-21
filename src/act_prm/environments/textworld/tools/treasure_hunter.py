"""
Tools and their descriptions for the 'treasure_hunter' task
"""

TOOL_TO_TEMPLATE: dict[str, str] = {
    "look": "look",
    "goal": "goal",
    "inventory": "inventory",
    "go": "go {direction}",
    "open": "open {target}",
    "drop": "drop {item}",
    "take": "take {item}",
    "put_on": "put {item} on {supporter}",
    "take_from": "take {item} from {source}",
    "insert_into": "insert {item} into {container}",
    "unlock_with": "unlock {target} with {key}",
}

TOOL_DESCRIPTIONS = {
    "look": {
        "type": "function",
        "name": "look",
        "description": "Describe the current room.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "goal": {
        "type": "function",
        "name": "goal",
        "description": "Print the goal of this game.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "inventory": {
        "type": "function",
        "name": "inventory",
        "description": "Print the player's inventory.",
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
    "examine": {
        "type": "function",
        "name": "examine",
        "description": "Examine something more closely.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "What to examine (object or feature name).",
                }
            },
            "required": ["target"],
        },
    },
    "open": {
        "type": "function",
        "name": "open",
        "description": (
            "Open a door or a container. You need to open a closed door before you can go through it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "The door or container to open.",
                }
            },
            "required": ["target"],
        },
    },
    "drop": {
        "type": "function",
        "name": "drop",
        "description": "Drop an object onto the floor.",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "The item to drop."}
            },
            "required": ["item"],
        },
    },
    "take": {
        "type": "function",
        "name": "take",
        "description": "Take an object that is visible.",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "The visible item to take."}
            },
            "required": ["item"],
        },
    },
    "put_on": {
        "type": "function",
        "name": "put_on",
        "description": "Place an object on a supporter.",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "The item to place."},
                "supporter": {
                    "type": "string",
                    "description": "The supporter to place the item on (e.g., table, counter).",
                },
            },
            "required": ["item", "supporter"],
        },
    },
    "take_from": {
        "type": "function",
        "name": "take_from",
        "description": "Take an object from a container or a supporter.",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "The item to take."},
                "source": {
                    "type": "string",
                    "description": "The container or supporter to take the item from.",
                },
            },
            "required": ["item", "source"],
        },
    },
    "insert_into": {
        "type": "function",
        "name": "insert_into",
        "description": "Place an object into a container.",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "The item to insert."},
                "container": {
                    "type": "string",
                    "description": "The container to insert the item into.",
                },
            },
            "required": ["item", "container"],
        },
    },
    "unlock_with": {
        "type": "function",
        "name": "unlock_with",
        "description": "Unlock a door or a container with a key.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "The door or container to unlock.",
                },
                "key": {"type": "string", "description": "The key to use."},
            },
            "required": ["target", "key"],
        },
    },
}
