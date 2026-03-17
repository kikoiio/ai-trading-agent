"""Data models for knowledge base entries and citations."""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class KBEntry:
    """A single knowledge base entry derived from one PPTX slide."""

    id: str                          # e.g., "AB-CH05-S03"
    source_file: str                 # relative path to PPTX
    chapter: str                     # folder name (chapter identifier)
    slide_number: int                # 1-based index within the PPTX
    slide_title: str                 # title placeholder text or first text line
    text_content: str                # all text extracted from the slide
    image_descriptions: list[str] = field(default_factory=list)  # vision model descriptions
    combined_text: str = ""          # text_content + image descriptions (searchable)
    tags: list[str] = field(default_factory=list)  # auto-extracted tags
    image_paths: list[str] = field(default_factory=list)  # paths to extracted images

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "KBEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def build_combined_text(self):
        """Rebuild combined_text from text_content and image_descriptions."""
        parts = [self.text_content]
        for desc in self.image_descriptions:
            parts.append(f"[Chart] {desc}")
        self.combined_text = "\n".join(parts)

    def format_for_prompt(self, max_chars: int = 600) -> str:
        """Format this entry for inclusion in LLM prompt context."""
        content = self.combined_text[:max_chars]
        if len(self.combined_text) > max_chars:
            content += "..."
        return f"[{self.id}] \"{self.slide_title}\"\nSource: {self.chapter}, Slide {self.slide_number}\n{content}"


@dataclass
class KBCitation:
    """A citation linking a trade decision to a knowledge base entry."""

    entry_id: str       # e.g., "AB-CH05-S03"
    relevance: str      # why this KB entry is relevant to the current decision

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "KBCitation":
        return cls(
            entry_id=data.get("entry_id", ""),
            relevance=data.get("relevance", ""),
        )


# Tags to auto-detect in slide content
TRADING_TAGS = [
    "trend", "pullback", "reversal", "breakout", "breakdown",
    "wedge", "channel", "triangle", "flag", "pennant",
    "signal bar", "entry bar", "follow-through", "gap",
    "trading range", "tight channel", "broad channel",
    "double top", "double bottom", "head and shoulders",
    "higher high", "higher low", "lower high", "lower low",
    "measured move", "leg", "spike", "climax",
    "bull", "bear", "long", "short",
    "stop loss", "stop", "target", "risk reward",
    "scalp", "swing", "always in",
    "ema", "moving average",
    "open", "close", "high", "low",
    "doji", "outside bar", "inside bar", "ii",
    "micro channel", "parabolic", "exhaustion",
    "first pullback", "second entry", "third push",
]


def auto_tag(text: str) -> list[str]:
    """Extract relevant trading tags from text content."""
    text_lower = text.lower()
    return [tag for tag in TRADING_TAGS if tag in text_lower]
