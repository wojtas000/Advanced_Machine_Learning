from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def get_word_counts_train_test(train, test):
    """
    Get word counts for train and test data. Create table with word counts for each message.
    Args:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Test data.
    Returns:
        result_train_df (pd.DataFrame): Training data with word counts.
        result_test_df (pd.DataFrame): Test data with word counts.
    """
    vectorizer = CountVectorizer()
    word_counts_train = vectorizer.fit_transform(train['message'])
    word_counts_test = vectorizer.transform(test['message'])
    feature_names = vectorizer.get_feature_names_out()
    counts_train_df = pd.DataFrame(word_counts_train.toarray(), columns=feature_names)
    counts_test_df = pd.DataFrame(word_counts_test.toarray(), columns=feature_names)
    result_train_df = pd.concat([train['label'], counts_train_df], axis=1)
    result_test_df = pd.concat([test['label'], counts_test_df], axis=1)
    return result_train_df, result_test_df
