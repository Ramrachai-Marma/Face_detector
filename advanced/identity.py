from typing import Dict, Optional


def find_label_id(labels: Dict[str, str], person_name: str) -> Optional[int]:
    """Return the ID for an exact, case-insensitive person-name match."""
    target = person_name.strip().casefold()
    for label_id, name in labels.items():
        if name.strip().casefold() == target:
            return int(label_id)
    return None
