import sqlite3

import pandas as pd


def ListQuestions(databaseName="data.db"):
    """Return a list of all questions in the FullSurvey table.

    databaseName may be a path or an open sqlite3 connection.
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
    """Return one row per respondent with one column per question.

    Pivots the long-format FullSurvey table (respondent_id, question, response)
    using one LEFT JOIN per question. Missing answers become NaN. Columns whose
    non-missing values all parse as numbers are converted to numeric dtypes.

    databaseName may be a path or an open sqlite3 connection.
    """
    questions = list(dict.fromkeys(questions))
    if not questions:
        raise ValueError("questions must contain at least one question")

    # Aliases are generated, so arbitrary question text never enters the SQL string.
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
