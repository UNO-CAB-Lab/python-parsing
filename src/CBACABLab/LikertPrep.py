import pandas as pd
from .LoadRawData import LoadSettings
from .LoadSQLData import LoadSurveyWithSensorDataLong


def LikertToK(responses, scale, settings=None):
    """
    Convert Likert responses to 0-based k values using a scale defined in settings.json.

    Args:
        responses (pandas.Series): Raw responses (text or numeric).
        scale (str or dict): Name of a scale under "LikertScales" in settings.json, or a mapping.
        settings (dict): Optional pre-loaded settings.

    Returns:
        tuple: (k as nullable Int64 Series, bound)

    Raises:
        KeyError: If the scale name is unknown.
        ValueError: If a non-missing response is not in the scale.
    """
    if isinstance(scale, str):
        scales = (settings or LoadSettings()).get("LikertScales", {})
        if scale not in scales:
            raise KeyError(f"Unknown Likert scale '{scale}'. Available: {sorted(scales)}")
        scale = scales[scale]
    mapping = {str(key).strip().lower(): value for key, value in scale.items()}
    normalized = responses.astype("string").str.strip().str.lower()
    # Numeric responses stored as floats (e.g. "4.0") should match "4"
    normalized = normalized.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)
    k = normalized.map(mapping)
    unknown = responses[normalized.notna() & k.isna()].unique()
    if len(unknown):
        raise ValueError(f"Responses not in scale: {list(unknown)}")
    return k.astype("Int64"), len(mapping)


