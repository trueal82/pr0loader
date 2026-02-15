"""Pydantic models for pr0loader data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class Tag(BaseModel):
    """A tag associated with an item."""
    id: int
    confidence: float = 0.0
    tag: str

    class Config:
        frozen = True


class Comment(BaseModel):
    """A comment on an item."""
    id: int
    parent: int = 0
    content: str = ""
    name: str = ""
    mark: int = 0
    created: int = 0
    up: int = 0
    down: int = 0


class Item(BaseModel):
    """An item (post) from pr0gramm."""
    id: int
    image: str = ""
    promoted: int = 0
    up: int = 0
    down: int = 0
    created: int = 0
    width: int = 0
    height: int = 0
    audio: int = 0
    source: str = ""
    flags: int = 0
    user: str = ""
    mark: int = 0
    gift: int = 0
    tags: list[Tag] = Field(default_factory=list)
    comments: list[Comment] = Field(default_factory=list)


class ItemsResponse(BaseModel):
    """Response from the items/get API endpoint."""
    items: list[Item] = Field(default_factory=list)
    at_end: bool = Field(default=False, alias="atEnd")
    at_start: bool = Field(default=False, alias="atStart")


class ItemInfoResponse(BaseModel):
    """Response from the items/info API endpoint."""
    tags: list[Tag] = Field(default_factory=list)
    comments: list[Comment] = Field(default_factory=list)


@dataclass
class PipelineStats:
    """Statistics for a pipeline run."""
    items_processed: int = 0
    items_skipped: int = 0
    items_failed: int = 0
    files_downloaded: int = 0
    files_skipped: int = 0
    bytes_downloaded: int = 0

    @property
    def total_items(self) -> int:
        return self.items_processed + self.items_skipped + self.items_failed


@dataclass
class PredictionResult:
    """Result of tag prediction for an image."""
    image_path: str
    tags: list[tuple[str, float]] = field(default_factory=list)  # (tag_name, confidence)

    @property
    def top_5_tags(self) -> list[tuple[str, float]]:
        """Return the top 5 tags sorted by confidence."""
        return sorted(self.tags, key=lambda x: x[1], reverse=True)[:5]

