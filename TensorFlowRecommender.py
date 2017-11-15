import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from six import next

import dataio

import time
# This class has an interface inspired by scikitlearn's
# BaseEstimator. I have also followed the convention of appending
# all attributes available only after fitting with an underscore
# (e.g. self.graph_, self.predictor_, et cetra)

def clip(x):
    return np.clip(x, 1.0, 5.0)


# def make_scalar_summary(name, val):
#     return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

def make_scalar_summary(writer, epoch, name, value):
    summary = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=value)])
    return writer.add_summary(summary, epoch)

class TensorFlowRecommender(object):
    def __init__(self):
        self.session_ = None
        self.graph_ = None

    def __del__(self):
        # have to close TensorFlow sessions to release resources
        if self.session_:
            self.session_.close()
        if self.graph_:
            del self.graph_

    def _makeGraph(self, user_num, item_num):
        self.graph_ = tf.Graph()
        with self.graph_.as_default():
            # create the placeholders to feed the values in to
            # Need reference to them later, so store them in the class
            self.user_batch_ = tf.placeholder(tf.int32, shape=[None], name="batch/user_batch")
            self.item_batch_ = tf.placeholder(tf.int32, shape=[None], name="batch/item_batch")
            self.rating_batch_ = tf.placeholder(tf.float32, shape = [None], name="batch/rating_batch")

            # Group the parameters together
            bias_global = tf.get_variable("params/bias_global", shape=[])
            bias_user = tf.get_variable("params/bias_user", shape=[user_num])
            bias_item = tf.get_variable("params/bias_item", shape=[item_num])
            weight_user = tf.get_variable("params/weight_user", shape=[user_num, self.dim_],
                                          initializer=tf.truncated_normal_initializer(stddev=0.02))
            weight_item = tf.get_variable("params/weight_item", shape=[item_num, self.dim_],
                                          initializer=tf.truncated_normal_initializer(stddev=0.02))

            # now group the embedding lookups
            embed_lookup_bias_user = tf.nn.embedding_lookup(bias_user, self.user_batch_, name="lookup/embed_bias_user")
            embed_lookup_bias_item = tf.nn.embedding_lookup(bias_item, self.item_batch_, name="lookup/embed_bias_item")
            embed_lookup_weight_user = tf.nn.embedding_lookup(weight_user, self.user_batch_, name="lookup/embed_weight_user")
            embed_lookup_weight_item = tf.nn.embedding_lookup(weight_item, self.item_batch_, name="lookup/embed_weight_item")

            # Make the predictor, which uses the values in the current batch. This predictor
            # will also be used in training
            infer = tf.reduce_sum(tf.multiply(embed_lookup_weight_user, embed_lookup_weight_item), axis=1, name="predict/dotproduct")
            infer = tf.add(infer, bias_global, name="predict/add_global_bias")
            infer = tf.add(infer, embed_lookup_bias_user, name="predict/add_user_bias")
            self.predict_ = tf.add(infer, embed_lookup_bias_item, name="predict/add_item_bias")

            # Term that gets added to the cost to prevent overfitting
            regularizer = tf.add(tf.nn.l2_loss(embed_lookup_weight_user), tf.nn.l2_loss(embed_lookup_weight_item), name="optimize/regTerm/squaredWeights")

            # now start making the optimizer
            global_step = tf.train.get_global_step()
            cost_l2 = tf.nn.l2_loss(tf.subtract(self.predict_, self.rating_batch_), name="optimize/squareDiff")
            penalty = tf.constant(self.reg_, dtype=tf.float32, shape=[], name="optimize/regTerm/Lambda_2")
            cost = tf.add(cost_l2, tf.multiply(regularizer, penalty, name = "optimize/regTerm"), name="optimize/cost")

            self.train_op_ = tf.train.AdamOptimizer(self.learning_rate_).minimize(cost, global_step=global_step)

    def fit(self, train, test, dim = 15, epoch_max = 100, batch_size = 1000,learning_rate = 0.02, reg = 0.1):
        self.learning_rate_ = 0.02
        self.reg_ = 0.1
        self.num_users_ = max(max(train['user']), max(test['user'])) + 1
        self.num_items_ = max(max(train['item']), max(test['item'])) + 1
        self.batch_size_ = batch_size
        self.epoch_max_ = epoch_max
        self.dim_ = dim

        self._makeGraph(self.num_users_, self.num_items_)

        with self.graph_.as_default():
            global_step = tf.contrib.framework.get_or_create_global_step()
            init_op = tf.global_variables_initializer()

        batches_per_epoch = len(train) / self.batch_size_

        iter_train = dataio.ShuffleIterator([train['user'],
                                             train['item'],
                                             train['rate']],
                                            batch_size = self.batch_size_)
        iter_test = dataio.OneEpochIterator([test['user'],
                                             test['item'],
                                             test['rate']],
                                            batch_size = -1)

        self.session_ = tf.Session(graph = self.graph_)

        self.session_.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir='/tmp/svd/log', graph = self.session_.graph)
        print "{:<8s}  {:<12s}  {:<12s}  {:<12s}".format("epoch", "train_error", "test_error", "elapsed_time (s)")
        errors = []
        start = time.time()

        for batch_num in range(self.epoch_max_ * batches_per_epoch):
            epoch_num = batch_num / batches_per_epoch
            users, items, ratings = next(iter_train)
            _, pred_batch = self.session_.run([self.train_op_, self.predict_], feed_dict = {self.user_batch_: users,
                                                                                            self.item_batch_: items,
                                                                                            self.rating_batch_: ratings})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - ratings,2))

            if batch_num % batches_per_epoch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_errors= []
                for test_users, test_items, test_ratings in iter_test:
                    pred_batch = self.session_.run(self.predict_, feed_dict = {self.user_batch_: test_users,
                                                                               self.item_batch_: test_items})
                    pred_batch = clip(pred_batch)
                    test_errors.append(np.power(pred_batch - test_ratings,2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_errors))
                print "{:>8d}  {:>11.6f}  {:>11.6f}  {:>11.6f}".format(epoch_num, train_err, test_err, end - start)

                make_scalar_summary(summary_writer, epoch_num, "training_error", train_err)
                make_scalar_summary(summary_writer, epoch_num, "test_error", test_err)
                start = end

        return self

    def predict(self, X):
        """
        X should be a dataframe with the columns 'user' and 'item'

        Will return predictions of item ratings
        """
        try:
            getattr(self, "predict_")
        except AttributeError:
            raise RuntimeError("You have to train the classifier with fit before using it to predict")

        users = list(X['user'])
        items = list(X['item'])

        pred_batch = self.session_.run(self.predict_, feed_dict = {self.user_batch_: users,
                                                                   self.item_batch_: items})
        pred_batch = clip(pred_batch)
        return pred_batch

    def predictTopK(self, user, K):
        """
        Predicts the top K items (highest predicted rating) for user
        Input:
            user: int
            K: int

        Output:
            [(item_index1, predicted_rating1), ..., (item_indexK, predicted_ratingK)]
        """
        try:
            getattr(self, "predict_")
        except AttributeError:
            raise RuntimeError("You have to train the classifier with fit before using it to predict")

        df = pd.DataFrame([{'user': user, 'item': item_num} for item_num in range(self.num_items_)])
        ratings = np.array(self.predict(df))
        highest_rated = np.argsort(ratings)[-K:][::-1]
        return zip(highest_rated, ratings[highest_rated])

    def numUsers(self):
        try:
            getattr(self, "num_users_")
        except AttributeError:
            raise RuntimeError("You have to train the classifier using fit before the number of users is defined")
        return self.num_users_

    def numTasks(self):
        try:
            getattr(self, "num_tasks_")
        except AttributeError:
            raise RuntimeError("You have to train the classifier using fit before the number of tasks is defined")
        return self.num_tasks_
