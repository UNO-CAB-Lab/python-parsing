from IPython.display import display

def SummaryStatisticsOfCounts(dataframe, columns, save=None):
    """
    Display a count and percent table for each categorical column.

    Missing values are counted as their own category. Each table is shown with IPython's display.

    Args:
        dataframe (pandas.DataFrame): The data to summarize.
        columns (list): The names of the columns to summarize.
        save (str): Optional file path prefix. If given, each table is also written to
            <save>_<column>_distribution.csv.

    Returns:
        None
    """
    for column in columns:
        counts = dataframe[column].value_counts(dropna=False)
        table = counts.rename_axis("Category").to_frame("Count")
        table["Percent"] = table["Count"] / table["Count"].sum()

        display(
            table.style
            .format({"Count": "{:,}", "Percent": "{:.1%}"})
            .set_caption(f"{column.title()} distribution")
        )
        if save is not None:
            table.to_csv(save + f"_{column}_distribution.csv")

def SummaryStatisticsOfContinuous(dataframe, columns, save=None):
    """
    Display descriptive statistics (count, mean, std, min, quartiles, max) for each numeric column.

    Each table is shown with IPython's display.

    Args:
        dataframe (pandas.DataFrame): The data to summarize.
        columns (list): The names of the columns to summarize.
        save (str): Optional file path prefix. If given, each summary is also written to
            <save>_<column>_distribution.csv.

    Returns:
        None
    """
    for column in columns:
        summary = dataframe[column].describe()
        display(
            summary.to_frame().style
            .format("{:.2f}")
            .set_caption(f"{column.title()} distribution")
        )
        if save is not None:
            summary.to_csv(save + f"_{column}_distribution.csv")

