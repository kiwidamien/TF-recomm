import dataio

from TensorFlowRecommender import TensorFlowRecommender
import numpy as np
np.random.seed(13575)



def get_data():
    df = dataio.read_process("data/ml-1m/ratings.dat", sep="::")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test

if __name__ == '__main__':
    df_train, df_test = get_data()
    tfr = TensorFlowRecommender()
    tfr.fit(df_train, df_test, epoch_max = 6)
    print("Done fitting")

    print
    print "Top 10 items for user 1:"
    print "{:6s}     {:11s}".format("Item #", "Pred Rating")
    print "-"*22
    for item, score in tfr.predictTopK(1, 10):
        print "{:>6d}     {:>11.5f}".format(item, score)
