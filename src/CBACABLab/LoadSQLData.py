import re
import sqlite3
import pandas as pd
from .LoadRawData import LoadSettings

def _base_question(q):
    # Strip a trailing _<digits> only if something remains before it
    return re.sub(r"^(.+)_\d+$", r"\1", q)

def ListInstrumentColumns(databaseName="data.db"):
    """
    Return a list of all distinct instrument columns in the InstrumentData table.

    Args:
        databaseName (str or sqlite3.Connection; optional): The path to the SQLite database file or an open connection.

    Returns:
        list: The instrument columns, sorted alphabetically.
    """
    owns_connection = not isinstance(databaseName, sqlite3.Connection)
    con = sqlite3.connect(databaseName) if owns_connection else databaseName
    try:
        df = pd.read_sql_query(
            "SELECT DISTINCT property FROM InstrumentData ORDER BY property", con
        )
    finally:
        if owns_connection:
            con.close()
    return df["property"].tolist()

def ListLikertScales(settings=None, show_values=False):
    """
    Return a list of all Likert scales defined in settings.json.

    Args:
        settings (dict): Optional pre-loaded settings.
        show_values (bool): If True, return a list of (scale_name, scale_values) tuples.
    """
    scales = (settings or LoadSettings()).get("LikertScales", {})
    if show_values:
        return sorted(scales.items())
    return sorted(scales.keys())

def ListQuestions(databaseName="data.db"):
    """
    Return a list of all distinct questions in the FullSurvey table.

    Args:
        databaseName (str or sqlite3.Connection; optional): The path to the SQLite database file or an open connection.

    Returns:
        list: The questions, sorted alphabetically.
    """
    owns_connection = not isinstance(databaseName, sqlite3.Connection)
    con = sqlite3.connect(databaseName) if owns_connection else databaseName
    try:
        df = pd.read_sql_query(
            "SELECT DISTINCT question FROM FullSurvey ORDER BY question", con
        )
    finally:
        if owns_connection:
            con.close()
    return df["question"].tolist()

def LoadQuestionsWide(questions, databaseName="data.db"):
    """
    Load survey responses in wide format, with one row per respondent and one column per question.

    Pivots the long-format FullSurvey table using one LEFT JOIN per question. Missing answers
    become NaN. Columns whose non-missing values all parse as numbers are converted to numeric dtypes.

    Args:
        questions (list): The questions to include as columns. Duplicates are ignored.
        databaseName (str or sqlite3.Connection; optional): The path to the SQLite database file or an open connection.

    Returns:
        pandas.DataFrame: A respondent_id column followed by one column per question.

    Raises:
        ValueError: If questions is empty.
    """
    questions = list(dict.fromkeys(questions))
    if not questions:
        raise ValueError("questions must contain at least one question")

    selects = ["r.respondent_id AS respondent_id"]
    joins = []
    for i in range(len(questions)):
        selects.append(f"q{i}.response AS q{i}")
        joins.append(
            f"LEFT JOIN FullSurvey q{i} "
            f"ON q{i}.respondent_id = r.respondent_id AND q{i}.question = ?"
        )
    sql = (
        f"SELECT {', '.join(selects)} "
        "FROM (SELECT DISTINCT respondent_id FROM FullSurvey) r "
        f"{' '.join(joins)} "
        "ORDER BY r.respondent_id"
    )

    owns_connection = not isinstance(databaseName, sqlite3.Connection)
    con = sqlite3.connect(databaseName) if owns_connection else databaseName
    try:
        df = pd.read_sql_query(sql, con, params=questions)
    finally:
        if owns_connection:
            con.close()

    df.columns = ["respondent_id"] + questions
    for column in questions:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().sum() == df[column].notna().sum():
            df[column] = converted
    return df




def LoadSurveyWithSensorDataWide(questions, databaseName="data.db", tagFilter=None, instrumentProperties=None):
    """
    Join survey responses to SensorData and InstrumentData in wide format, with one row per respondent and question.

    The survey is joined to SensorData and InstrumentData on respondent_id and question. Each tag
    gets its own columns: pupil_<tag> and duration_<tag> for SensorData, and <property>_<tag>
    for InstrumentData. When no tags are specified, a tag is not included in the column names.
    Missing sensor or instrument data becomes NaN.

    Args:
        questions (list): The questions to include. Duplicates are ignored.
        tagFilter (str, optional): Filter for specific tags in the sensor data.
        instrumentProperties (list, optional): List of instrument properties to include from the instrument data.
        databaseName (str or sqlite3.Connection; optional): The path to the SQLite database file or an open connection.

    Returns:
        pandas.DataFrame: One row per (respondent_id, question) with the survey response
            and the sensor and instrument columns described above.

    Raises:
        ValueError: If questions is empty.
    """
    questions = list(dict.fromkeys(questions))
    if not questions:
        raise ValueError("questions must contain at least one question")
    marks = ", ".join("?" * len(questions))
    baseQuestions = list(dict.fromkeys(_base_question(q) for q in questions))
    baseMarks = ", ".join("?" * len(baseQuestions))

    if tagFilter is None:
        tagFilter = ["default",]

    if not isinstance(tagFilter, (list, tuple, set)):
        raise ValueError("tagFilter must be a list, tuple, or set")
    tagFilter = set(tagFilter)
    if not tagFilter:
        raise ValueError("tagFilter must contain at least one tag")
    tagCondition = f"AND tag IN ({', '.join('?' * len(tagFilter))})"
    tagParams = list(tagFilter)

    if instrumentProperties is not None:
        if not isinstance(instrumentProperties, (list, tuple, set)):
            raise ValueError("instrumentProperties must be a list, tuple, or set")
        instrumentProperties = set(instrumentProperties)
        if not instrumentProperties:
            raise ValueError("instrumentProperties must contain at least one property")
        propertyCondition = f"AND property IN ({', '.join('?' * len(instrumentProperties))})"
        propertyParams = list(instrumentProperties)
    else:
        propertyCondition = ""
        propertyParams = []


    owns_connection = not isinstance(databaseName, sqlite3.Connection)
    con = sqlite3.connect(databaseName) if owns_connection else databaseName
    try:
        survey = pd.read_sql_query(
            "SELECT respondent_id, question, response FROM FullSurvey "
            f"WHERE question IN ({marks})", con, params=questions)
        sensor = pd.read_sql_query(
            "SELECT respondent_id, question, tag, pupil, duration FROM SensorData "
            f"WHERE question IN ({baseMarks}) {tagCondition}", con, params=baseQuestions + tagParams)
        face = pd.read_sql_query(
            "SELECT respondent_id, question, tag, property, value FROM InstrumentData "
            f"WHERE question IN ({baseMarks}) {tagCondition} {propertyCondition}", con, params=baseQuestions + tagParams + propertyParams)
    finally:
        if owns_connection:
            con.close()

    key = ["respondent_id", "question"]
    mkey = ["respondent_id", "base"]

    def widen(df, columns, values):
        if df.empty:
            return df[key].copy().rename(columns={"question": "base"})
        wide = df.pivot_table(index=key, columns=columns, values=values, aggfunc="mean")

        def column_name(measure, tag):
            return measure if tag == "default" else f"{measure}_{tag}"

        if isinstance(wide.columns, pd.MultiIndex) and len(values) == 1:
            wide.columns = [column_name(c[-2], c[-1]) for c in wide.columns]
        else:
            wide.columns = [column_name(measure, tag) for measure, tag in wide.columns]
        return wide.reset_index().rename(columns={"question": "base"})

    sensor = widen(sensor.melt(id_vars=key + ["tag"], value_vars=["pupil", "duration"],
                               var_name="measure", value_name="value"),
                   ["measure", "tag"], ["value"])
    face = widen(face.rename(columns={"property": "measure"}),
                 ["measure", "tag"], ["value"])

    survey["base"] = survey["question"].map(_base_question)
    result = (survey.merge(sensor, on=mkey, how="left")
                    .merge(face, on=mkey, how="left")
                    .drop(columns="base"))
    result["question"] = pd.Categorical(result["question"], categories=questions, ordered=True)
    result = result.sort_values(key).reset_index(drop=True)
    result["question"] = result["question"].astype(str)

    for column in result.columns:
        if column in key:
            continue
        converted = pd.to_numeric(result[column], errors="coerce")
        if converted.notna().sum() == result[column].notna().sum():
            result[column] = converted

    return result

def LoadSurveyWithSensorDataLong(questions, databaseName="data.db", tagFilter=None, instrumentProperties=None):
    """
    Join survey responses to SensorData and InstrumentData in long format, with one row per tag and instrument property.

    The survey is joined to SensorData on respondent_id and question, and then to InstrumentData on
    respondent_id, question and tag. Missing sensor or instrument data becomes NaN.

    Args:
        questions (list): The questions to include. Duplicates are ignored.
        databaseName (str or sqlite3.Connection; optional): The path to the SQLite database file or an open connection.
        tagFilter (list, tuple, or set; optional): If provided, only include rows with a tag in this set.
        instrumentProperties (list; optional): A list of instrument properties to include.

    Returns:
        pandas.DataFrame: One row per (respondent_id, question, tag, property) with the columns
            response, pupil, duration, property, and value.

    Raises:
        ValueError: If questions is empty.
    """
    questions = list(dict.fromkeys(questions))
    if not questions:
        raise ValueError("questions must contain at least one question")
    marks = ", ".join("?" * len(questions))
    baseQuestions = list(dict.fromkeys(_base_question(q) for q in questions))
    baseMarks = ", ".join("?" * len(baseQuestions))

    if tagFilter is None:
        tagFilter = ["default",]

    if not isinstance(tagFilter, (list, tuple, set)):
        raise ValueError("tagFilter must be a list, tuple, or set")
    tagFilter = set(tagFilter)
    if not tagFilter:
        raise ValueError("tagFilter must contain at least one tag")
    tagCondition = f"AND tag IN ({', '.join('?' * len(tagFilter))})"
    tagParams = list(tagFilter)


    if instrumentProperties is not None:
        if not isinstance(instrumentProperties, (list, tuple, set)):
            raise ValueError("instrumentProperties must be a list, tuple, or set")
        instrumentProperties = set(instrumentProperties)
        if not instrumentProperties:
            raise ValueError("instrumentProperties must contain at least one property")
        propertyCondition = f"AND property IN ({', '.join('?' * len(instrumentProperties))})"
        propertyParams = list(instrumentProperties)
    else:
        propertyCondition = ""
        propertyParams = []

    owns_connection = not isinstance(databaseName, sqlite3.Connection)
    con = sqlite3.connect(databaseName) if owns_connection else databaseName
    try:
        survey = pd.read_sql_query(
            "SELECT respondent_id, question, response FROM FullSurvey "
            f"WHERE question IN ({marks})", con, params=questions)
        sensor = pd.read_sql_query(
            "SELECT respondent_id, question, tag, pupil, duration FROM SensorData "
            f"WHERE question IN ({baseMarks}) {tagCondition}", con, params=baseQuestions + tagParams)
        face = pd.read_sql_query(
            "SELECT respondent_id, question, tag, property, value FROM InstrumentData "
            f"WHERE question IN ({baseMarks}) {tagCondition} {propertyCondition}", con,
            params=baseQuestions + tagParams + propertyParams)
    finally:
        if owns_connection:
            con.close()

    survey["base"] = survey["question"].map(_base_question)
    sensor = sensor.rename(columns={"question": "base"})
    face = face.rename(columns={"question": "base"})
    result = survey.merge(sensor, on=["respondent_id", "base"], how="left")
    result = result.merge(face, on=["respondent_id", "base", "tag"], how="left")
    result = result.drop(columns="base")
    result["question"] = pd.Categorical(result["question"], categories=questions, ordered=True)
    result = result.sort_values(["respondent_id", "question", "tag"]).reset_index(drop=True)
    result["question"] = result["question"].astype(str)
    for column in result.columns:
        if column in ["respondent_id", "question", "tag"]:
            continue
        converted = pd.to_numeric(result[column], errors="coerce")
        if converted.notna().sum() == result[column].notna().sum():
            result[column] = converted
    return result