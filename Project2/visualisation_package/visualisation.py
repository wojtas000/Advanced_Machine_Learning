import matplotlib.pyplot as plt


def visualize_strength_vs_number_of_features(df, groupby_column='Number_of_features', columns_to_evaluate=['Performance_score', 'Accuracy'], title='Strength vs Number of Features'):

    """
    Visualizes the strength vs number of features.
    Args:
        df: Dataframe containing the aggregated data.
        groupby_column: Column to group the data by.
        columns_to_evaluate: Columns to evaluate.
        title: Title of the plot.
    Returns:
        None
    """

    for column in columns_to_evaluate:
        res = df.groupby(groupby_column).apply(lambda x: x.nlargest(10, column)).droplevel(0)
        res.boxplot(column=column, by=[groupby_column])
        plt.xticks(rotation=45)
        plt.title('Best 10 accuracies of models for different number of features')
        plt.xlabel(groupby_column)
        plt.ylabel(column)
        
    plt.show()

def visualize_feature_selection_strength(df, title):

    """
    Visualizes the feature selection strength of the different feature selection methods.
    Args:
        df: Dataframe containing the aggregated data.
        title: Title of the plot.
    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the 'Performance_score' bars with error bars
    df.plot(kind='bar', y='Performance_score_mean', yerr='Performance_score_std', ax=ax, position=0, color='blue', width=0.4)

    # Plot the 'Accuracy' bars with error bars
    df.plot(kind='bar', y='Accuracy_mean', yerr='Accuracy_std', ax=ax, position=1, color='orange', width=0.4)

    # Set labels and title
    ax.set_xlabel('Feature Selection Method')
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.set_ylabel('Scores')
    ax.set_title(title, fontsize=20)

    # Set legend
    ax.legend(labels=['Mean Performance Score', 'Mean Accuracy'])

    # Show the plot
    plt.show()