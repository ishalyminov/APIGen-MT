"""Auto-generated VideoImagesTools implementation."""

import json
import math
import re
import copy
import datetime
import random
from typing import List, Dict, Any, Optional, Tuple, Union


class VideoImagesTools:
    """Tools for video and image operations, including background removal modes and movie queries."""

    METHOD_NAME_MAP = {
        'Get list of available modes': 'Get_list_of_available_modes',
        'List Movies': 'List_Movies',
        'Sort By': 'Sort_By',
    }

    def __init__(self, initial_config: dict = None) -> None:
        """Initialize the VideoImagesTools instance.

        Args:
            initial_config: Optional dict containing initial configuration.
        """
        self._config_data: dict = {}
        if initial_config is not None:
            # Store all provided keys in _config_data
            for k, v in initial_config.items():
                self._config_data[k] = v
        else:
            # Default configuration if none provided
            self._config_data['default_sort'] = 'popularity'
            self._config_data['available_modes'] = [
                'mask of foreground',
                'image with foreground object',
                'image with foreground object with shadow'
            ]
            self._config_data['movies'] = [
                'King Kong',
                'The Social Network',
                'Facebook Fever',
                'Kong vs. Godzilla',
                'The Movie of the Year'
            ]

    def Get_list_of_available_modes(self) -> List[str]:
        """Return the list of available modes for car image background removal.

        The three modes are:
        1. mask of foreground
        2. image with foreground object
        3. image with foreground object with shadow

        Returns:
            A list of strings representing the available modes.
        """
        return self._config_data.get('available_modes', [
            'mask of foreground',
            'image with foreground object',
            'image with foreground object with shadow'
        ])

    def List_Movies(self) -> Dict[str, str]:
        """List and search through all available movies.

        This method returns a message containing a summary of available movies.
        Although the full API supports filtering, sorting, and searching, this
        simplified implementation returns a static list.

        Returns:
            A dict with a single key 'message' containing the movie listing.
        """
        movies = self._config_data.get('movies', [
            'King Kong',
            'The Social Network',
            'Facebook Fever',
            'Kong vs. Godzilla',
            'The Movie of the Year'
        ])
        movie_list_str = ", ".join(movies)
        return {"message": f"Available movies: {movie_list_str}"}

    def Sort_By(self, sort_by: str = None) -> Dict[str, str]:
        """Sort the results by the chosen value.

        Args:
            sort_by: The sorting criterion (required). Expected values include
                     'popularity', 'rating', 'year', 'alphabetical'.

        Returns:
            A dict with a single key 'message' containing the sorting confirmation.
        """
        if not sort_by:
            return {"message": "Error: 'sort_by' parameter is required and cannot be empty."}

        allowed_sorts = ['popularity', 'rating', 'year', 'alphabetical', 'release_date']
        if sort_by not in allowed_sorts:
            return {"message": f"Error: Unknown sort option '{sort_by}'. Allowed values: {', '.join(allowed_sorts)}"}

        return {"message": f"Results have been sorted by '{sort_by}'."}