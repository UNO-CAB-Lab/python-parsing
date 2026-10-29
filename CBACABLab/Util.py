from IPython.display import display

def SummaryStatisticsOfCounts(dataframe, columns, save=None):
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
    for column in columns:
        summary = dataframe[column].describe()
        display(
            summary.to_frame().style
            .format("{:.2f}")
            .set_caption(f"{column.title()} distribution")
        )
        if save is not None:
            summary.to_csv(save + f"_{column}_distribution.csv")

