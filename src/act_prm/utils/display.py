"""
Helper functions for displaying data
"""

from transformers.generation.streamers import TextStreamer

from .logging import rich_print_messages


class RichTextStreamer(TextStreamer):
    """
    A streamer that prints the text in a rich format
    """
    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        rich_print_messages(msg_text=text, flush=True, end="" if not stream_end else None)
