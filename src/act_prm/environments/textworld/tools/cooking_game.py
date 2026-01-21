"""
Tools and their descriptions for the `the_cooking_game` task
"""

TOOL_TO_TEMPLATE: dict[str, str] = {
    "look": "look",
    "goal": "goal",
    "inventory": "inventory",
    "go": "go {direction}",
    "examine": "examine {target}",
    "eat": "eat {food}",
    "open": "open {target}",
    "drop": "drop {item}",
    "take": "take {item}",
    "put_on": "put {item} on {supporter}",
    "take_from": "take {item} from {source}",
    "insert_into": "insert {item} into {container}",
    "lock_with": "lock {target} with {key}",
    "unlock_with": "unlock {target} with {key}",
    "cook_with": "cook {food} with {heat_source}",
    "slice_with": "slice {food} with {tool}",
    "chop_with": "chop {food} with {tool}",
    "dice_with": "dice {food} with {tool}",
    "prepare_meal": "prepare meal",
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
    "eat": {
        "type": "function",
        "name": "eat",
        "description": "Eat edible food.",
        "parameters": {
            "type": "object",
            "properties": {
                "food": {"type": "string", "description": "The edible item to eat."}
            },
            "required": ["food"],
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
    "lock_with": {
        "type": "function",
        "name": "lock_with",
        "description": "Lock a door or a container with a key.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "The door or container to lock.",
                },
                "key": {"type": "string", "description": "The key to use."},
            },
            "required": ["target", "key"],
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
    "cook_with": {
        "type": "function",
        "name": "cook_with",
        "description": "Cook cookable food with something providing heat.",
        "parameters": {
            "type": "object",
            "properties": {
                "food": {"type": "string", "description": "The food to cook."},
                "heat_source": {
                    "type": "string",
                    "description": "The thing providing heat (e.g., stove, oven, grill).",
                },
            },
            "required": ["food", "heat_source"],
        },
    },
    "slice_with": {
        "type": "function",
        "name": "slice_with",
        "description": "Slice cuttable food with something sharp.",
        "parameters": {
            "type": "object",
            "properties": {
                "food": {"type": "string", "description": "The food to slice."},
                "tool": {"type": "string", "description": "A sharp tool to slice with."},
            },
            "required": ["food", "tool"],
        },
    },
    "chop_with": {
        "type": "function",
        "name": "chop_with",
        "description": "Chop cuttable food with something sharp.",
        "parameters": {
            "type": "object",
            "properties": {
                "food": {"type": "string", "description": "The food to chop."},
                "tool": {"type": "string", "description": "A sharp tool to chop with."},
            },
            "required": ["food", "tool"],
        },
    },
    "dice_with": {
        "type": "function",
        "name": "dice_with",
        "description": "Dice cuttable food with something sharp.",
        "parameters": {
            "type": "object",
            "properties": {
                "food": {"type": "string", "description": "The food to dice."},
                "tool": {"type": "string", "description": "A sharp tool to dice with."},
            },
            "required": ["food", "tool"],
        },
    },
    "prepare_meal": {
        "type": "function",
        "name": "prepare_meal",
        "description": (
            "Combine ingredients from inventory into a meal. You can only prepare meals in the Kitchen."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
