from pickler import get_or_create_pickle
import pandas as pd 


def data_processer(df):

    # Index of the stimulation
    # eeg: 8 seconds of EEG signal before the stimulation (250 Hz)
    # respiration_x: 8 seconds of Accelerometer signal on x axis before the stimulation (50 Hz)
    # respiration_y: 8 seconds of Accelerometer signal on y axis before the stimulation (50 Hz)
    # respiration_z: 8 seconds of Accelerometer signal on z axis before the stimulation (50 Hz)
    # time_previous: time between current stimulation and previous stimulation (-1 means no previous stimulation)
    # number_previous: number of previous stimulations
    # time: time elapsed from beginning of the night
    # user: user id
    # night: night id
    # power_increase: value to predict: impact of the stimulation measured on EEG signal after stimulation.

    """
    NORMALISATION
    """

    # eeg data
    print('# eggs')
    df_eggs_normalized = normalize(df.filter(regex='egg*', axis=1))
    # respiration_x data
    print("# respiration x")
    df_respiration_x_normalized = normalize(df.filter(regex='respiration_x*', axis=1))
    # respiration_y data
    print("# respiration y")
    df_respiration_y_normalized = normalize(df.filter(regex='respiration_y*', axis=1))
    # respiration_z data
    print("# respiration z")
    df_respiration_z_normalized = normalize(df.filter(regex='respiration_z*', axis=1))
    # Concatenation of all data
    print("Concatenation..")
    result = pd.concat([
        df_eggs_normalized,
        df_respiration_x_normalized,
        df_respiration_y_normalized,
        df_respiration_z_normalized,
        normalize(df['time']),
        normalize(df['number_previous']),
        normalize(df['time_previous']),
        normalize(df['time']),
    ], axis=1)
    # add the target (power_increase column) to the training data
    if 'power_increase' in df.columns:
        result = pd.concat([result, df['power_increase']], axis=1)
    return result 


def normalize(df):
    print('Dataframe mean..')
    df_mean = df.mean()
    print(df_std)

    print('Dataframe std..')
    df_mean = df.std()
    print(df_std)

    result = (df - df_mean) / df_std

    print(result.head())

    return result