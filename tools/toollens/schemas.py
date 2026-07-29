"""Auto-generated Pydantic input schemas for all ToolLens tools."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal, Union

# ─── VideoImagesTools ──────────


class GetListOfAvailableModesInput(BaseModel):
    """Input schema for Get list of available modes."""
    model_config = ConfigDict(extra='forbid')

    pass

class ListMoviesInput(BaseModel):
    """Input schema for List Movies."""
    model_config = ConfigDict(extra='forbid')

    pass

class SortByInput(BaseModel):
    """Input schema for Sort By."""
    model_config = ConfigDict(extra='forbid')

    sort_by: str = Field(
        ...,
        description="The field to sort the movies by (e.g., 'title', 'year', 'rating')."
    )

