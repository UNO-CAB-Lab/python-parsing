import sqlite3
import shutil
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import CBACABLab
from CBACABLab import Plots


ROOT = Path(__file__).resolve().parents[1]
AGREEMENT_SCALE = {
    "Strongly disagree": 0,
    "Somewhat disagree": 1,
    "Neither agree nor disagree": 2,
    "Somewhat agree": 3,
    "Strongly agree": 4,
}


@pytest.fixture(scope="module")
def sample_database(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("sample-data")
    database = work_dir / "sample.db"
    sensor_folder = work_dir / "sensorData"
    sensor_folder.mkdir()
    shutil.copyfile(
        ROOT / "sensorData" / "001_00010.csv",
        sensor_folder / "001_00010.csv",
    )
    mapping = CBACABLab.AutoDetermineQuestionIdtoQuestionNameMapping(
        ROOT / "Survey.xlsx", ROOT / "Survey.csv"
    )

    CBACABLab.LoadSurveyData(databaseName=str(database), surveyName=str(ROOT / "Survey.xlsx"))
    for tag in ("left", "right"):
        CBACABLab.LoadSensorData(
            databaseName=str(database),
            sensorFolder=str(sensor_folder) + "/",
            boundedboxes=[(0, 960, 0, 1080)] if tag == "left" else [(960, 1920, 0, 1080)],
            tag=tag,
            rebuild=tag == "left",
            questionIdtoQuestionName=mapping,
        )

    return {"database": str(database), "mapping": mapping}


def test_public_api_exports():
    public_functions = (
        "ProjectKDEPlot",
        "CreateLOESSPlots",
        "CreateMultipleLinePlots",
        "CreateMultipleLOESSPlots",
        "makeHeatMap",
        "SummaryStatisticsOfCounts",
        "SummaryStatisticsOfContinuous",
        "LoadSurveyData",
        "LoadSensorData",
        "AutoDetermineQuestionIdtoQuestionNameMapping",
        "ListQuestions",
        "LoadQuestionsWide",
        "LoadSurveyWithSensorDataLong",
        "LoadSurveyWithSensorDataWide",
        "ListInstrumentColumns",
        "ListLikertScales",
        "LikertToK",
        "AddLikertColumns",
        "CreatePanelData",
    )
    assert all(callable(getattr(CBACABLab, name, None)) for name in public_functions)


def test_sample_data_loading_and_sql_queries(sample_database):
    database = sample_database["database"]
    assert sample_database["mapping"]["QID1"] == "Q1"
    assert sample_database["mapping"]["QID4_4"] == "Q4_4"

    with sqlite3.connect(database) as connection:
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("FullSurvey", "SensorData", "InstrumentData")
        }
    assert counts["FullSurvey"] > 0
    assert counts["SensorData"] > 0
    assert counts["InstrumentData"] > 0

    questions = CBACABLab.ListQuestions(database)
    assert {"Q1", "Q2", "Q3", "Q4_4", "Q5_4"} <= set(questions)
    assert {"Attention", "Joy"} <= set(CBACABLab.ListInstrumentColumns(database))

    respondent_data = CBACABLab.LoadQuestionsWide(
        ["Q4_4", "Q5_4", "Q4_4"], databaseName=database
    )
    assert respondent_data.columns.tolist() == ["respondent_id", "Q4_4", "Q5_4"]
    assert respondent_data["respondent_id"].is_unique
    assert len(respondent_data) > 0

    wide = CBACABLab.LoadSurveyWithSensorDataWide(
        ["Q1", "Q2", "Q3"],
        databaseName=database,
        tagFilter=["left", "right"],
        instrumentProperties=["Attention", "Joy"],
    )
    assert {"response", "pupil_left", "pupil_right", "Attention_left", "Joy_right"} <= set(
        wide.columns
    )
    assert not wide.duplicated(["respondent_id", "question"]).any()

    long = CBACABLab.LoadSurveyWithSensorDataLong(
        ["Q1", "Q2", "Q3"],
        databaseName=database,
        tagFilter=["left", "right"],
        instrumentProperties=["Attention", "Joy"],
    )
    assert {
        "respondent_id",
        "question",
        "response",
        "tag",
        "pupil",
        "duration",
        "property",
        "value",
    } <= set(long.columns)
    assert {"left", "right"} <= set(long["tag"].dropna())


def test_likert_and_panel_data_api(sample_database):
    assert "agree5" in CBACABLab.ListLikertScales()
    scales = CBACABLab.ListLikertScales(show_values=True)
    assert "agree5" in dict(scales)
    assert dict(scales)["agree5"]["Strongly agree"] == 4

    responses = pd.Series(
        ["Strongly disagree", " Neither agree nor disagree ", "Strongly agree", None]
    )
    k, bound = CBACABLab.LikertToK(responses, AGREEMENT_SCALE)
    assert k.tolist() == [0, 2, 4]
    assert bound == 4

    panel = CBACABLab.CreatePanelData(
        question_level_columns=["Q1", "Q2", "Q3"],
        respondent_level_columns=["Q4_4", "Q5_4", "Q6", "Q7"],
        tagFilter=["left", "right"],
        instrumentProperties=["Attention", "Joy"],
        databaseName=sample_database["database"],
    )
    required = {
        "respondent_id",
        "question",
        "response",
        "Q4_4",
        "Q5_4",
        "Q6",
        "Q7",
        "pupil_left",
        "duration_right",
    }
    assert required <= set(panel.columns)
    assert set(panel["question"]) == {"Q1", "Q2", "Q3"}
    assert not panel.duplicated(["respondent_id", "question"]).any()
    assert panel[["pupil_left", "pupil_right"]].notna().any().any()

    scale_labels = {label.lower() for label in AGREEMENT_SCALE}
    valid = panel.loc[
        panel["response"].astype("string").str.strip().str.lower().isin(scale_labels)
    ]
    likert_panel = CBACABLab.AddLikertColumns(
        valid, {question: AGREEMENT_SCALE for question in ("Q1", "Q2", "Q3")}
    )
    assert {"k", "bound"} <= set(likert_panel.columns)
    assert set(likert_panel["bound"]) == {4}
    assert likert_panel["k"].between(0, 4).all()


def test_summary_statistics_api(tmp_path):
    data = pd.DataFrame(
        {"category": ["A", "B", "A", None], "score": [1.0, 2.0, 3.0, 4.0]}
    )
    prefix = str(tmp_path / "summary")

    assert CBACABLab.SummaryStatisticsOfCounts(data, ["category"], save=prefix) is None
    assert CBACABLab.SummaryStatisticsOfContinuous(data, ["score"], save=prefix) is None
    assert (tmp_path / "summary_category_distribution.csv").is_file()
    assert (tmp_path / "summary_score_distribution.csv").is_file()


def test_plotting_api(tmp_path, monkeypatch):
    monkeypatch.setattr(Plots.plt, "show", lambda: None)
    data = pd.DataFrame(
        {
            "x": range(1, 9),
            "response": [2, 3, 5, 4, 6, 8, 7, 9],
            "other": [9, 7, 8, 6, 5, 4, 3, 2],
        }
    )

    plot_calls = (
        (
            "loess",
            CBACABLab.CreateLOESSPlots,
            (data.copy(), "x", "response", [0.8]),
        ),
        (
            "multiple_loess",
            CBACABLab.CreateMultipleLOESSPlots,
            (data.copy(), "x", ["response"], [0.8]),
        ),
        (
            "lines",
            CBACABLab.CreateMultipleLinePlots,
            (data.copy(), "x", ["response", "other"]),
        ),
    )
    try:
        for name, plot, args in plot_calls:
            plot(*args, save=str(tmp_path / name))

        CBACABLab.ProjectKDEPlot(
            [data["response"], data["other"]],
            labels=["response", "other"],
            mean=True,
            save=str(tmp_path / "kde"),
        )
        display_data = pd.DataFrame(
            [[0.1, 0.2], [0.3, 0.4]],
            index=["Q1_marginal", "Q2_marginal"],
            columns=["a", "b"],
        )
        heatmap_data = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=["Q1", "Q2"],
            columns=["a", "b"],
        )
        CBACABLab.makeHeatMap(
            display_data,
            heatmap_data,
            xlabel="Question",
            ylabel="Group",
            title="Sample heatmap",
            save=str(tmp_path / "heatmap"),
        )
    finally:
        plt.close("all")

    for name in ("loess", "multiple_loess", "lines", "kde", "heatmap"):
        assert (tmp_path / f"{name}.pdf").is_file()
        assert (tmp_path / f"{name}.svg").is_file()
