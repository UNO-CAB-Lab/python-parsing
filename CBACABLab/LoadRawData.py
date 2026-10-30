import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
import sqlite3
import os

def insertRow(con, row):
    sql = "INSERT OR REPLACE INTO FullSurvey(respondent_id, question, response) VALUES (?, ?, ?)"
    cur = con.cursor()
    cur.execute(sql, row)
    con.commit()
    cur.close()


def LoadSurveyData(databaseName="data.db", surveyName="Survey.xlsx"):
    if not os.path.isfile(surveyName):
        raise FileNotFoundError(f"Survey file does not exist: {surveyName}")
    
    con = sqlite3.connect(databaseName)
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS FullSurvey")
    cur.execute("CREATE TABLE FullSurvey (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, respondent_id INTEGER NOT NULL, question TEXT NOT NULL, response TEXT NOT NULL, UNIQUE(respondent_id, question) ON CONFLICT REPLACE)")
    con.commit()
    cur.close()
    qualtricsSurvey = pd.read_excel(surveyName, skiprows=[1])
    qualtricsSurvey.rename(columns=lambda column: column.strip() if isinstance(column, str) else column, inplace=True)

    qualtricsSurvey["iMotionsRespondent"] = pd.to_numeric(
        qualtricsSurvey["iMotionsRespondent"], errors="coerce"
    )
    qualtricsSurvey = qualtricsSurvey.loc[qualtricsSurvey["iMotionsRespondent"].notna()].copy()

    marker_index = next(
        (
            index
            for index, column in enumerate(qualtricsSurvey.columns)
            if "".join(str(column).casefold().split()) == "userlanguage"
        ),
        None,
    )
    if marker_index is None:
        raise ValueError("Could not find 'User Language' or 'UserLanguage' column")
    
    for _, row in qualtricsSurvey.iterrows():
        respondent_id = int(row["iMotionsRespondent"])
        if respondent_id in (99, 999):
            continue
        for question in qualtricsSurvey.columns[marker_index + 1:]:
            response = row[question]
            if pd.isna(response):
                continue
            response = str(response)
            if response.strip():
                insertRow(con, (respondent_id, str(question), response))


def insertSensorRow(con, row):
    cur = con.cursor()
    cur.execute("INSERT INTO SensorData (respondent_id, question, pupil, duration, tag) VALUES (?, ?, ?, ?, ?)", row)
    con.commit()
    cur.close()

def ExtractData(con, df, respondentid, boundedboxes=None, tag="default"):
    df = df.rename(columns=lambda column: column.strip() if isinstance(column, str) else column)

    if boundedboxes is not None:
        if {'Interpolated Gaze X', 'Interpolated Gaze Y'}.issubset(df.columns):
            gaze_x, gaze_y = 'Interpolated Gaze X', 'Interpolated Gaze Y'
        elif {'Gaze X', 'Gaze Y'}.issubset(df.columns):
            gaze_x, gaze_y = 'Gaze X', 'Gaze Y'
        else:
            raise ValueError("Missing gaze columns: expected Interpolated Gaze X and Y or Gaze X and Y")

        master_mask = pd.Series(False, index=df.index)
        for x_min, x_max, y_min, y_max in boundedboxes:
            bounds = (x_min, x_max, y_min, y_max)
            if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in bounds):
                raise ValueError(f"Bounding box values must be numbers: {bounds}")
            if x_min >= x_max or y_min >= y_max:
                raise ValueError(f"Invalid bounding box: ({x_min}, {x_max}, {y_min}, {y_max})")
            current_mask = (
                (df[gaze_x] >= x_min) & (df[gaze_x] <= x_max) & 
                (df[gaze_y] >= y_min) & (df[gaze_y] <= y_max)
            )
            master_mask |= current_mask
        filtered_df = df[master_mask].copy()
    else:
        filtered_df = df.copy()
    

    listOfQuestions = (
        df.loc[
            df['MarkerName'].notna()
            & (df['MarkerName'].str.strip() != '')
            & (~df['MarkerName'].str.endswith('_end', na=False)),
            ['Timestamp', 'MarkerName']
        ]
        .drop_duplicates(subset=['MarkerName'], keep='first')
    )
    listOfQuestionEnds = (
        df.loc[
            df['MarkerName'].notna()
            & (df['MarkerName'].str.strip() != '')
            & (df['MarkerName'].str.endswith('_end', na=False)),
            ['Timestamp', 'MarkerName']
        ]
        .drop_duplicates(subset=['MarkerName'], keep='last')
    )

    starts = listOfQuestions.copy()
    starts['question'] = starts['MarkerName'].str.strip()

    ends = listOfQuestionEnds.copy()
    ends['question'] = ends['MarkerName'].str.strip().str.replace(r'_end$', '', regex=True)

    listOfQuestions = (
        starts.merge(
            ends[['Timestamp', 'question']],
            on='question',
            how='inner',
            suffixes=('_start', '_end')
        )
        .rename(columns={'Timestamp_start': 'start_time', 'Timestamp_end': 'end_time'})
        [['question', 'start_time', 'end_time']]
        .sort_values('start_time')
        .reset_index(drop=True)
    )

    for _, q in listOfQuestions.iterrows():
        question = q['question']
        start_time = q['start_time']
        end_time = q['end_time']

        df2 = filtered_df.loc[
            (filtered_df['Timestamp'] >= start_time) & (filtered_df['Timestamp'] <= end_time),
            ['ET_PupilLeft', 'ET_PupilRight']
        ].copy()

        df3 = filtered_df.loc[
            (filtered_df['Timestamp'] >= start_time) & (filtered_df['Timestamp'] <= end_time),
            ['Fixation Duration']
        ].copy()

        df2['ET_PupilLeft'] = pd.to_numeric(df2['ET_PupilLeft'], errors='coerce')
        df2['ET_PupilRight'] = pd.to_numeric(df2['ET_PupilRight'], errors='coerce')
        df2 = df2.dropna(subset=['ET_PupilLeft', 'ET_PupilRight'])
        pupil = (df2['ET_PupilLeft'].mean() + df2['ET_PupilRight'].mean()) / 2
        duration = df3['Fixation Duration'].mean()
        question_name = str(question).strip()
        row = (respondentid, question_name, float(pupil), float(duration), tag)
        insertSensorRow(con, row)

def LoadSensorData(databaseName = "data.db", sensorFolder = "./sensorData/", boundedboxes = None, tag = "default", rebuild = False, low_memory = False):
    con = sqlite3.connect(databaseName)
    cur = con.cursor()
    if rebuild:
        cur.execute("DROP TABLE IF EXISTS SensorData")
    cur.execute("CREATE TABLE IF NOT EXISTS SensorData (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, respondent_id INTEGER NOT NULL, question TEXT NOT NULL, pupil REAL, duration REAL, tag TEXT, UNIQUE(respondent_id, question, tag) ON CONFLICT REPLACE)")
    con.commit()
    cur.close()
    folder = sensorFolder
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Sensor folder does not exist: {folder}")
    
    f = os.listdir(folder)
    for file in f:
        filename, file_extension = os.path.splitext(file)
        if file_extension == '.csv':
            _, respondentid = filename.split('_')
            df = pd.read_csv(folder+file, comment='#', low_memory=low_memory)
            ExtractData(con, df, int(respondentid), boundedboxes, tag)
