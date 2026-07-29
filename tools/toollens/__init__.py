"""ToolLens tool class registry (auto-generated)."""

from typing import Dict, Any, Optional

TOOLLENS_TOOL_CLASSES: Dict[str, str] = {
    "business": "tools.toollens.business",
    "commerce": "tools.toollens.commerce",
    "ecommerce": "tools.toollens.ecommerce",
    "education": "tools.toollens.education",
    "email": "tools.toollens.email",
    "finance": "tools.toollens.finance",
    "food": "tools.toollens.food",
    "gaming": "tools.toollens.gaming",
    "health_and_fitness": "tools.toollens.health_and_fitness",
    "location": "tools.toollens.location",
    "medical": "tools.toollens.medical",
    "movies": "tools.toollens.movies",
    "music": "tools.toollens.music",
    "news_media": "tools.toollens.news_media",
    "sports": "tools.toollens.sports",
    "transportation": "tools.toollens.transportation",
    "travel": "tools.toollens.travel",
    "video_images": "tools.toollens.video_images",
    "weather": "tools.toollens.weather",
}

TOOLLENS_CLASS_NAMES: Dict[str, str] = {
    "business": "BusinessTools",
    "commerce": "CommerceTools",
    "ecommerce": "EcommerceTools",
    "education": "EducationTools",
    "email": "EmailTools",
    "finance": "FinanceTools",
    "food": "FoodTools",
    "gaming": "GamingTools",
    "health_and_fitness": "HealthAndFitnessTools",
    "location": "LocationTools",
    "medical": "MedicalTools",
    "movies": "MoviesTools",
    "music": "MusicTools",
    "news_media": "NewsMediaTools",
    "sports": "SportsTools",
    "transportation": "TransportationTools",
    "travel": "TravelTools",
    "video_images": "VideoImagesTools",
    "weather": "WeatherTools",
}

TOOLLENS_API_NAME_TO_CLASS_KEY: Dict[str, str] = {}
_APIS_POPULATED = False
def _populate_api_name_map() -> None:
    global _APIS_POPULATED
    if _APIS_POPULATED:
        return
    import importlib, inspect
    for class_key, module_path in TOOLLENS_TOOL_CLASSES.items():
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, TOOLLENS_CLASS_NAMES[class_key], None)
            if cls is None:
                continue
            if hasattr(cls, 'METHOD_NAME_MAP'):
                for api_name in cls.METHOD_NAME_MAP:
                    TOOLLENS_API_NAME_TO_CLASS_KEY[api_name] = class_key
            else:
                for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction):
                    if name.startswith('_'):
                        continue
                    TOOLLENS_API_NAME_TO_CLASS_KEY[name] = class_key
        except Exception as e:
            print(f'ToolLens: Could not introspect class {class_key}: {e}')
    _APIS_POPULATED = True

def reset_api_name_map() -> None:
    global _APIS_POPULATED
    TOOLLENS_API_NAME_TO_CLASS_KEY.clear()
    _APIS_POPULATED = False

def get_toollens_api_name_map() -> Dict[str, str]:
    _populate_api_name_map()
    return dict(TOOLLENS_API_NAME_TO_CLASS_KEY)

def create_toollens_instance(class_key: str, initial_config: dict = None):
    import importlib
    if class_key not in TOOLLENS_TOOL_CLASSES:
        raise KeyError(f'Unknown ToolLens class key: {class_key}')
    module = importlib.import_module(TOOLLENS_TOOL_CLASSES[class_key])
    cls = getattr(module, TOOLLENS_CLASS_NAMES[class_key])
    return cls(initial_config=initial_config)

def create_toollens_tool_instances(
    configs: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    '''Instantiate every ToolLens class with its per-class config.

    Args:
        configs: Dict mapping class_key -> config dict. If None or
                missing a key, the class is instantiated with initial_config=None.

    Returns:
        Dict mapping class_key -> instance.
    '''
    configs = configs or {}
    instances: Dict[str, Any] = {}
    for class_key in TOOLLENS_TOOL_CLASSES:
        try:
            cfg = configs.get(class_key, {}) or None
            instances[class_key] = create_toollens_instance(class_key, initial_config=cfg)
        except Exception as e:
            print(f'Warning: Could not instantiate ToolLens class {class_key}: {e}')
    return instances

__all__ = [
    'TOOLLENS_TOOL_CLASSES', 'TOOLLENS_CLASS_NAMES',
    'TOOLLENS_API_NAME_TO_CLASS_KEY',
    'create_toollens_instance', 'create_toollens_tool_instances',
    'get_toollens_api_name_map', 'reset_api_name_map',
]
