import pandas as pd
from .LoadRawData import LoadSettings
from .LoadSQLData import LoadSurveyWithSensorDataLong, LoadSurveyWithSensorDataWide


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
    return k.astype("Int64"), max(mapping.values())


def AddLikertColumns(data, scales, question_column="question", response_column="response", settings=None):
    """
    Add k and bound columns to panel data, using a Likert scale chosen per question.

    Args:
        data (pandas.DataFrame): Data with a question column and a response column, such as the output of CreatePanelData.
        scales (str, dict): A scale name or mapping applied to every question, or a dict of
            {question: scale name or mapping} giving each question its own scale.
        question_column (str, optional): Column holding the question.
        response_column (str, optional): Column holding the Likert response.
        settings (dict, optional): Pre-loaded settings.

    Returns:
        pandas.DataFrame: A copy of data with nullable Int64 columns k and bound.

    Raises:
        KeyError: If a question has no scale, or a scale name is unknown.
        ValueError: If a response is not in its question's scale.
    """
    settings = settings or LoadSettings()
    result = data.copy()
    questions = result[question_column].dropna().unique()
    if isinstance(scales, (str, dict)) and (isinstance(scales, str) or not set(questions) & set(scales)):
        # A single scale (name or label mapping) used for every question
        scales = {question: scales for question in questions}
    missing = [question for question in questions if question not in scales]
    if missing:
        raise KeyError(f"No Likert scale given for questions: {missing}")

    result["k"] = pd.array([pd.NA] * len(result), dtype="Int64")
    result["bound"] = pd.array([pd.NA] * len(result), dtype="Int64")
    for question in questions:
        mask = result[question_column] == question
        try:
            k, bound = LikertToK(result.loc[mask, response_column], scales[question], settings)
        except ValueError as error:
            raise ValueError(f"Question '{question}': {error}") from error
        result.loc[mask, "k"] = k
        result.loc[mask, "bound"] = bound
    return result


def CreatePanelData(question_level_columns, respondent_level_columns, tagFilter=None, faceProperties=None, databaseName="data.db"):
    """
    Create a panel dataset by merging survey responses with sensor data.

    Args:
        question_level_columns (list): List of question-level columns to include.
        respondent_level_columns (list): List of respondent-level columns to include.
        tagFilter (str, optional): Filter for specific tags in the sensor data.
        faceProperties (list, optional): List of face properties to include from the sensor data.
        databaseName (str, optional): Name of the SQLite database file.

    Returns:
        pandas.DataFrame: Merged panel dataset.
    """
    respondent_data = LoadSurveyWithSensorDataWide(databaseName=databaseName, questions=respondent_level_columns, tagFilter=tagFilter, faceProperties=faceProperties)
    question_data = LoadSurveyWithSensorDataLong(databaseName=databaseName, questions=question_level_columns, tagFilter=tagFilter, faceProperties=faceProperties)
    merged_data = pd.merge(question_data, respondent_data, on="respondent_id", how="left")
    return merged_data

