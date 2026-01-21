"""
Instruction prompts for TextWorld environment

NOTE (MZ 1/19/26): We should convert the commands to proper tool descriptions, i.e.,
{
    "type": "function",
    "name": "look",
    "description": "Describe the current room",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}
"""

INSTRUCTION_PROMPT_TREASURE_HUNT = """
You are an agent playing TextWorld, a text-based adventure game where you are in a randomly generated maze and must find a specific object.
You need to explore different rooms to find the target object.

## Rules:
- You need to unlock a locked door with a matched key in your inventory before you want to open it.
- The target object might be located in a closed or locked container
- The adjective is useful for determining whether the key is matched with the lock (e.g. non-euclidean keycard is matched with non-euclidean safe). Make sure it is matched to unlock!
- The key required to unlock the door may be in another room or locked inside a container
- Take the key whenever you can
- After unlocking a locked door or container, it will remain closed. You will then need to open it.

You have {max_turns} steps to complete the task. Restarting is forbidden.
"""

INSTRUCTION_PROMPT_THE_COOKING_GAME = """
You are an agent playing TextWorld, a text-based adventure game where you navigate through different rooms, interact with objects, and solve puzzles.
Your goal is to first find the recipe, find and prepare food according to the recipe, and finally prepare and eat the meal.

## Rules:
- You can examine the cookbook to see the recipe when it is visible
- The BBQ is for grilling things, the stove is for frying things, the oven is for roasting things. Cooking ingredients in the wrong way will lead to a failure of the game.
- Once you have got processed ingredients and the appropriate cooking tool ready, cook all of them according to the recipe.
- There are two conditions to correctly cook something (grill/fry/roast): a) the ingredient you want to cook is in your inventory and b) there is a suitable cooking tool in the room, and then use 'cook . . . with . . . ' command.
- When you need to chop/slice/dice ingredients, you need to take the knife and the ingredient in your inventory and then `slice`/`chop`/`dice` ... with 'knife'
- Make sure to first process the food (chop/slice/dice) before you try to cook them.
- When you have all the ingredients (that got processed or cooked according to the menu), you can `prepare_meal` in the kitchen and then `eat_meal` to win the game.
- The ingredients should EXACTLY match the color in the recipe, but if the recipe doesn't specify color, any color would be fine. When you `take_with`, use the EXACT name you see.
- You don't need to examine the container/supporter (e.g. toolbox) when it says something like "there isn't a thing on it"/"has nothing on it"

You have {max_turns} steps to complete the task. Restarting is forbidden.
"""

INSTRUCTION_PROMPT_COIN_COLLECTOR = """
You are an agent playing TextWorld, a text-based adventure game where you are in a randomly generated maze and must find the coin.
You need to explore different rooms to find the target object.

## Rules:
- The only actions you can do are `go` to explore the maze and `take_coin` when you see the coin in the room.

You have {max_turns} steps to complete the task. Restarting is forbidden.
"""


def get_instruction_prompt(task: str, max_turns: int | None = None) -> str:
    """
    Get the instruction prompt for a given task
    """
    match task:
        case "treasure_hunter":
            max_turns = max_turns or 40  # BALROG defaults
            return INSTRUCTION_PROMPT_TREASURE_HUNT.format(max_turns=max_turns).strip()
        case "the_cooking_game":
            max_turns = max_turns or 80
            return INSTRUCTION_PROMPT_THE_COOKING_GAME.format(max_turns=max_turns).strip()
        case "coin_collector":
            max_turns = max_turns or 25
            return INSTRUCTION_PROMPT_COIN_COLLECTOR.format(max_turns=max_turns).strip()
        case _:
            raise ValueError(f"Invalid task: {task}")
