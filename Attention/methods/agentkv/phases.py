"""Streaming parser for Qwen-style agent transcripts.

A query receives the phase active after consuming its input token. Markers
split across tokens take effect on their final token; earlier tokens are not
retroactively relabeled. Closing tags return to `others` (a documented parser
extension to the paper's abbreviated pseudocode).
"""

PHASES = ("think", "act", "tool", "others")
MARKERS = {
    "<|im_start|>system": "others",
    "<|im_start|>user": "others",
    "<|im_start|>assistant": "others",
    "<|im_start|>tool": "tool",
    "<think>": "think",
    "</think>": "others",
    "<tool_call>": "act",
    "</tool_call>": "others",
    "<tool_response>": "tool",
    "</tool_response>": "others",
    "<|im_end|>": "others",
}


class PhaseParser:
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = "others"
        self.tail = ""

    def feed(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a decoded token fragment")
        combined = self.tail + text
        boundary = len(self.tail)
        events = []
        for marker, phase in MARKERS.items():
            start = combined.find(marker)
            while start >= 0:
                end = start + len(marker)
                if end > boundary:
                    events.append((end, phase))
                start = combined.find(marker, start + 1)
        for _, phase in sorted(events):
            self.phase = phase
        self.tail = combined[-(max(map(len, MARKERS)) - 1):]
        return self.phase
