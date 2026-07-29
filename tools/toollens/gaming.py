"""Auto-generated GamingTools implementation."""

import json
import math
import re
import copy
import datetime
import random
from typing import List, Dict, Any, Optional, Tuple, Union


class GamingTools:
    """
    Gaming tools providing Bingo cards, dice rolls, charades words, player
    details, game schedules, Lost Ark information, and more.
    """

    METHOD_NAME_MAP = {
        '/eu': 'eu',
        '/us': 'us',
        'All Characters': 'All_Characters',
        'Get Charades Word': 'Get_Charades_Word',
        'Get Player Details': 'Get_Player_Details',
        'Get Schedule': 'Get_Schedule',
        'Get Stronghold Item Recipes': 'Get_Stronghold_Item_Recipes',
        'Get all island with dropped items': 'Get_all_island_with_dropped_items',
        'Nadeo Players': 'Nadeo_Players',
        'Regular dice rolls': 'Regular_dice_rolls',
        'get abyssal dungeons': 'get_abyssal_dungeons',
        'get classes': 'get_classes',
    }

    def __init__(self, initial_config: dict = None):
        """Initialize internal state.

        Args:
            initial_config: Optional dict that may contain a 'seed' key for
                deterministic randomness. All unknown keys are stored in
                self._config_data.
        """
        self._config_data = {}
        if initial_config:
            self._config_data = dict(initial_config)

        # Seed the RNG if requested
        seed = self._config_data.get('seed', 42)
        random.seed(seed)

        # Internal counters for sequential output
        self._config_data.setdefault('_eu_counter', 0)
        self._config_data.setdefault('_us_counter', 0)

    # ------------------------------------------------------------------
    #  BINGO helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bingo_column(lo: int, hi: int) -> List[int]:
        """Return a sorted list of 5 numbers from the inclusive range [lo, hi]."""
        return sorted(random.sample(range(lo, hi + 1), 5))

    def _make_us_card(self) -> List[List[int]]:
        """Build one standard 75-ball Bingo card (5x5, free space omitted)."""
        card = [
            self._bingo_column(1, 15),    # B
            self._bingo_column(16, 30),   # I
            self._bingo_column(31, 45),   # N (centre excluded)
            self._bingo_column(46, 60),   # G
            self._bingo_column(61, 75),   # O
        ]
        # Remove the free-space element from column N (third column, index 2)
        del card[2][2]
        return card

    # ------------------------------------------------------------------
    #  METHODS
    # ------------------------------------------------------------------

    def eu(self) -> Dict[str, Any]:
        """EU game spec. Returns a random number between 1 and 90."""
        # Simple deterministic sequence
        self._config_data['_eu_counter'] += 1
        # Use the counter to pick a number, wrap around
        number = ((self._config_data['_eu_counter'] - 1) % 90) + 1
        return {"number": number}

    def us(self) -> Dict[str, Any]:
        """US game spec. Returns one 75-ball Bingo card as an array of 5 arrays."""
        self._config_data['_us_counter'] += 1
        card = self._make_us_card()
        return {"card": card}

    def All_Characters(self) -> Dict[str, Any]:
        """Access all characters in the MVC2 universe."""
        characters = [
            {
                "name": "Ryu",
                "head_shot": "https://example.com/ryu.png",
                "universe": "Street Fighter",
            },
            {
                "name": "Iron Man",
                "head_shot": "https://example.com/ironman.png",
                "universe": "Marvel",
            },
            {
                "name": "Megaman",
                "head_shot": "https://example.com/megaman.png",
                "universe": "Capcom",
            },
            {
                "name": "Spider-Man",
                "head_shot": "https://example.com/spiderman.png",
                "universe": "Marvel",
            },
            {
                "name": "Morrigan Aensland",
                "head_shot": "https://example.com/morrigan.png",
                "universe": "Darkstalkers",
            },
        ]
        return {"characters": characters}

    def Get_Charades_Word(self, difficulty: Optional[str] = None) -> Dict[str, Any]:
        """Get a random charades word with the specified difficulty.

        Args:
            difficulty: One of 'easy', 'moderate', 'hard'. If None a random
                difficulty is used.

        Returns:
            Dict with success, difficulty, and word.
        """
        difficulties = ['easy', 'moderate', 'hard']
        words = {
            'easy': ['cat', 'dog', 'run', 'eat', 'sleep'],
            'moderate': ['basketball', 'piano', 'painting', 'vacation', 'telephone'],
            'hard': ['photosynthesis', 'onomatopoeia', 'xylophone', 'discombobulate', 'quizzical'],
        }
        if difficulty not in difficulties:
            difficulty = random.choice(difficulties)

        word = random.choice(words[difficulty])
        return {"success": True, "difficulty": difficulty, "word": word}

    def Get_Player_Details(self, name: str, region: str) -> Dict[str, Any]:
        """Get player details based on username (case sensitive) and region.

        Args:
            name: The player's username.
            region: The region (e.g., 'NA', 'EU', 'KR').

        Returns:
            Dict with username, rank, lp, and winLossRatio.
        """
        # Simple mock - base on name hash to be deterministic
        base = sum(ord(c) for c in (name + region)) % 1000
        rank = f"Gold {base % 4 + 1}"
        lp = str(1500 + base % 500)
        ratio = f"{0.5 + (base % 100) / 200:.2f}"
        return {
            "username": name,
            "rank": rank,
            "lp": lp,
            "winLossRatio": ratio,
        }

    def Get_Schedule(self) -> Dict[str, Any]:
        """Get all schedules for the leagues."""
        return {
            "data": {
                "league": "WNBA",
                "season": "2023",
                "events": [
                    {"date": "2023-06-15", "home": "Las Vegas Aces", "away": "Chicago Sky"},
                    {"date": "2023-06-15", "home": "New York Liberty", "away": "Seattle Storm"},
                ],
            }
        }

    def Get_Stronghold_Item_Recipes(self) -> Dict[str, Any]:
        """Get list of Lost Ark Stronghold Item Recipes."""
        # Realistic static status
        return {"status": "success"}

    def Get_all_island_with_dropped_items(self) -> Dict[str, Any]:
        """Return all islands with IDs of dropped items."""
        # Example island data
        return {
            "name": "Peyto",
            "items": {
                "133003": 142,
                "510103": 55,
                "889005": 23,
                "889101": 7,
            },
        }

    def Nadeo_Players(self) -> Dict[str, Any]:
        """Get players from Nadeo."""
        return {
            "amount": 1287,
            "query": {
                "method": "GET",
                "search_query": "Faker",
            },
        }

    def Regular_dice_rolls(self) -> Dict[str, Any]:
        """Roll a number of dice and return total.

        Simulates rolling one regular six‑sided die (default).
        """
        total = random.randint(1, 6)
        return {"total": total}

    def get_abyssal_dungeons(self) -> Dict[str, Any]:
        """Get all abyssal dungeons in Lost Ark."""
        return {
            "Ancient Elveria": {},
            "Phantom Palace": {},
            "Ark of Arrogance": {},
            "Gate of Paradise": {},
            "Oreha's Well": {},
        }

    def get_classes(self) -> Dict[str, Any]:
        """Get all classes and subclasses in Lost Ark."""
        return {
            "Warrior": {
                "subclasses": ["Berserker", "Paladin", "Gunlancer"],
            },
            "Martial artist": {
                "subclasses": ["Striker", "Wardancer", "Scrapper"],
            },
            "Gunner": {
                "subclasses": ["Gunslinger", "Artillerist", "Deadeye"],
            },
            "Mage": {
                "subclasses": ["Bard", "Sorceress", "Summoner"],
            },
            "Assassin": {
                "subclasses": ["Shadowhunter", "Deathblade", "Reaper"],
            },
        }