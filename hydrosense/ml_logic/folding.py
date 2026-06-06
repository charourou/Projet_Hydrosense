def get_folds(df: pd.DataFrame, n_fold: int):

    """
    This function slides through the Time Series dataframe of shape (n_timesteps, n_features) to create folds
    - All Fold starts for the beginning of the data.
    - Each new fold is one year longer
    - the last fold ends on september 2024
    - The first fold ends on september 2020 (n_fold - 1 years) earlier

    Args:
        df (pd.DataFrame): Overall dataframe
        n_fold : number of desired folds

    Returns:
        List[pd.DataFrame]: A list where each fold is a dataframe within
    """

    fold_stride = 365
    fold_length =

    start_list = range(0, df.shape[0]-fold_length, fold_stride)
    output = []

    for start in start_list:
        mask = range(start,start+fold_length)
        output.append(df.iloc[ start : (start+fold_length) ,:])

    return output

def train_val_split(fold_in: pd.DataFrame, val_duration = 90) -> tuple :
    """
        Tuple[pd.DataFrame]: A tuple of two dataframes (fold_train, fold_val)

        All fold_train should end on the month of may
        all fold val should end 3 months later

    """



    return
