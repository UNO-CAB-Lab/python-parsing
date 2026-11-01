import sqlite3

import pandas as pd


def ListQuestions(databaseName="data.db"):
    """
    Return a list of all distinct questions in the FullSurvey table.

    Args:
        databaseName (str or sqlite3.Connection): The path to the SQLite database file or an open connection.

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
        databaseName (str or sqlite3.Connection): The path to the SQLite database file or an open connection.

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



def LoadSurveyWithSensorDataWide(questions, databaseName="data.db"):
    """
    Join survey responses to SensorData and FaceData in wide format, with one row per respondent and question.

    The survey is joined to SensorData and FaceData on respondent_id and question. Each tag
    gets its own columns: pupil_<tag> and duration_<tag> for SensorData, and <property>_<tag>
    for FaceData. Missing sensor or face data becomes NaN.

    Args:
        questions (list): The questions to include. Duplicates are ignored.
        databaseName (str or sqlite3.Connection): The path to the SQLite database file or an open connection.

    Returns:
        pandas.DataFrame: One row per (respondent_id, question) with the survey response
            and the sensor and face columns described above.

    Raises:
        ValueError: If questions is empty.
    """
    questions = list(dict.fromkeys(questions))
    if not questions:
        raise ValueError("questions must contain at least one question")
    marks = ", ".join("?" * len(questions))

    owns_connection = not isinstance(databaseName, sqlite3.Connection)
    con = sqlite3.connect(databaseName) if owns_connection else databaseName
    try:
        survey = pd.read_sql_query(
            "SELECT respondent_id, question, response FROM FullSurvey "
            f"WHERE question IN ({marks})", con, params=questions)
        sensor = pd.read_sql_query(
            "SELECT respondent_id, question, tag, pupil, duration FROM SensorData "
            f"WHERE question IN ({marks})", con, params=questions)
        face = pd.read_sql_query(
            "SELECT respondent_id, question, tag, property, value FROM FaceData "
            f"WHERE question IN ({marks})", con, params=questions)
    finally:
        if owns_connection:
            con.close()

    key = ["respondent_id", "question"]

    def widen(df, columns, values):
        if df.empty:
            return df[key].copy()
        wide = df.pivot_table(index=key, columns=columns, values=values, aggfunc="mean")
        if isinstance(wide.columns, pd.MultiIndex) and len(values) == 1:
            wide.columns = [f"{c[-2]}_{c[-1]}" for c in wide.columns]
        else:
            wide.columns = [f"{m}_{t}" for m, t in wide.columns]
        return wide.reset_index()

    sensor = widen(sensor.melt(id_vars=key + ["tag"], value_vars=["pupil", "duration"],
                               var_name="measure", value_name="value"),
                   ["measure", "tag"], ["value"])
    face = widen(face.rename(columns={"property": "measure"}),
                 ["measure", "tag"], ["value"])

    result = survey.merge(sensor, on=key, how="left").merge(face, on=key, how="left")
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

def LoadSurveyWithSensorDataLong(questions, databaseName="data.db"):
    """
    Join survey responses to SensorData and FaceData in long format, with one row per tag and face property.

    The survey is joined to SensorData on respondent_id and question, and then to FaceData on
    respondent_id, question and tag. Missing sensor or face data becomes NaN.

    Args:
        questions (list): The questions to include. Duplicates are ignored.
        databaseName (str or sqlite3.Connection): The path to the SQLite database file or an open connection.

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

    owns_connection = not isinstance(databaseName, sqlite3.Connection)
    con = sqlite3.connect(databaseName) if owns_connection else databaseName
    try:
        survey = pd.read_sql_query(
            "SELECT respondent_id, question, response FROM FullSurvey "
            f"WHERE question IN ({marks})", con, params=questions)
        sensor = pd.read_sql_query(
            "SELECT respondent_id, question, tag, pupil, duration FROM SensorData "
            f"WHERE question IN ({marks})", con, params=questions)
        face = pd.read_sql_query(
            "SELECT respondent_id, question, tag, property, value FROM FaceData "
            f"WHERE question IN ({marks})", con, params=questions)
    finally:
        if owns_connection:
            con.close()

    result = survey.merge(sensor, on=["respondent_id", "question"], how="left")
    result = result.merge(face, on=["respondent_id", "question", "tag"], how="left")
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