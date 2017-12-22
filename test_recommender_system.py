import math
import time
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.nets as nets
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.optimizer import SGD, Adam, Adagrad, Adamax, Momentum
from paddle.v2.fluid.param_attr import ParamAttr
from paddle.v2.fluid.initializer import Normal

IS_SPARSE = True
USE_GPU = False
BATCH_SIZE = 256

def smart_attr(dims, name=None):
    return ParamAttr(name=name, initializer=Normal(loc=0.0, scale=1.0/math.sqrt(dims), seed=1))

def get_usr_combined_features():
    # FIXME(dzh) : old API integer_value(10) may has range check.
    # currently we don't have user configurated check.

    USR_DICT_SIZE = paddle.dataset.movielens.max_user_id() + 1

    uid = layers.data(name='user_id', shape=[1], dtype='int64')

    usr_emb = layers.embedding(
        input=uid,
        dtype='float32',
        size=[USR_DICT_SIZE, 32],
        param_attr=smart_attr(USR_DICT_SIZE, name='user_table'),
        is_sparse=IS_SPARSE)

    usr_fc = layers.fc(input=usr_emb, size=32, param_attr=smart_attr(32))

    USR_GENDER_DICT_SIZE = 2

    usr_gender_id = layers.data(name='gender_id', shape=[1], dtype='int64')

    usr_gender_emb = layers.embedding(
        input=usr_gender_id,
        size=[USR_GENDER_DICT_SIZE, 16],
        param_attr=smart_attr(USR_GENDER_DICT_SIZE, name='gender_table'),
        is_sparse=IS_SPARSE)

    usr_gender_fc = layers.fc(input=usr_gender_emb, size=16, param_attr=smart_attr(16))

    USR_AGE_DICT_SIZE = len(paddle.dataset.movielens.age_table)
    usr_age_id = layers.data(name='age_id', shape=[1], dtype="int64")

    usr_age_emb = layers.embedding(
        input=usr_age_id,
        size=[USR_AGE_DICT_SIZE, 16],
        is_sparse=IS_SPARSE,
        param_attr=smart_attr(USR_AGE_DICT_SIZE, name='age_table'))

    usr_age_fc = layers.fc(input=usr_age_emb, size=16, param_attr=smart_attr(16))

    USR_JOB_DICT_SIZE = paddle.dataset.movielens.max_job_id() + 1
    usr_job_id = layers.data(name='job_id', shape=[1], dtype="int64")

    usr_job_emb = layers.embedding(
        input=usr_job_id,
        size=[USR_JOB_DICT_SIZE, 16],
        param_attr=smart_attr(USR_JOB_DICT_SIZE, name='job_table'),
        is_sparse=IS_SPARSE)

    usr_job_fc = layers.fc(input=usr_job_emb, size=16, param_attr=smart_attr(16))

    concat_embed = layers.concat(
        input=[usr_fc, usr_gender_fc, usr_age_fc, usr_job_fc], axis=1)

    usr_combined_features = layers.fc(input=concat_embed, size=200, act="tanh", param_attr=smart_attr(200))

    return usr_combined_features


def get_mov_combined_features():

    MOV_DICT_SIZE = paddle.dataset.movielens.max_movie_id() + 1

    mov_id = layers.data(name='movie_id', shape=[1], dtype='int64')

    mov_emb = layers.embedding(
        input=mov_id,
        dtype='float32',
        size=[MOV_DICT_SIZE, 32],
        param_attr=smart_attr(MOV_DICT_SIZE, name='movie_table'),
        is_sparse=IS_SPARSE)

    mov_fc = layers.fc(input=mov_emb, size=32, param_attr=smart_attr(32))

    CATEGORY_DICT_SIZE = len(paddle.dataset.movielens.movie_categories())

    # category_id = layers.data(name='category_id', shape=[1], dtype='int64')

    # mov_categories_emb = layers.embedding(
    #     input=category_id, size=[CATEGORY_DICT_SIZE, 32], is_sparse=IS_SPARSE,
    #     param_attr=smart_attr(CATEGORY_DICT_SIZE))

    # mov_categories_hidden = layers.sequence_pool(
    #     input=mov_categories_emb, pool_type="sum")

    MOV_TITLE_DICT_SIZE = len(paddle.dataset.movielens.get_movie_title_dict())

    mov_title_id = layers.data(name='movie_title', shape=[1], dtype='int64')

    mov_title_emb = layers.embedding(
        input=mov_title_id, size=[MOV_TITLE_DICT_SIZE, 32], is_sparse=IS_SPARSE,
        param_attr=smart_attr(MOV_TITLE_DICT_SIZE))

    mov_title_conv = nets.sequence_conv_pool(
        input=mov_title_emb,
        num_filters=32,
        filter_size=3,
        act="tanh",
        pool_type="sum")

    concat_embed = layers.concat(
        # input=[mov_fc, mov_categories_hidden, mov_title_conv], axis=1)
        input=[mov_fc, mov_title_conv], axis=1)

    # FIXME(dzh) : need tanh operator
    mov_combined_features = layers.fc(input=concat_embed, size=200, act="tanh", param_attr=smart_attr(200))

    return mov_combined_features


def model():
    usr_combined_features = get_usr_combined_features()
    mov_combined_features = get_mov_combined_features()

    # need cos sim
    inference = layers.cos_sim(X=usr_combined_features, Y=mov_combined_features)
    scale_infer = layers.scale(x=inference, scale=5.0)

    label = layers.data(name='score', shape=[1], dtype='float32')

    square_cost = layers.square_error_cost(input=scale_infer, label=label)

    avg_cost = layers.mean(x=square_cost)

    return avg_cost, usr_combined_features, mov_combined_features


def main():
    cost, u, m = model()
    optimizer = SGD(learning_rate=0.2)
    # optimizer = Adam(learning_rate=1e-4)
    opts = optimizer.minimize(cost)

    if USE_GPU:
        place = core.GPUPlace(0)
    else:
        place = core.CPUPlace()

    exe = Executor(place)
    exe.run(framework.default_startup_program())
    # print framework.default_main_program().block(0).vars.keys()
    print framework.default_main_program().block(0).vars.get("embedding_0.tmp_0")

    train_reader = paddle.batch(
        paddle.dataset.movielens.train(),
        batch_size=BATCH_SIZE)

    feeding = {
        'user_id': 0,
        'gender_id': 1,
        'age_id': 2,
        'job_id': 3,
        'movie_id': 4,
        # 'category_id': 5,
        'movie_title': 6,
        'score': 7
    }

    def func_feed(feeding, data):
        feed_tensors = {}
        for (key, idx) in feeding.iteritems():
            tensor = core.LoDTensor()
            if key != "category_id" and key != "movie_title":
                if key == "score":
                    numpy_data = np.array(map(lambda x: x[idx], data)).astype(
                        "float32")
                else:
                    numpy_data = np.array(map(lambda x: x[idx], data)).astype(
                        "int64")
            else:
                numpy_data = map(lambda x: np.array(x[idx]).astype("int64"),
                                 data)
                lod_info = [len(item) for item in numpy_data]
                offset = 0
                lod = [offset]
                for item in lod_info:
                    offset += item
                    lod.append(offset)
                numpy_data = np.concatenate(numpy_data, axis=0)
                tensor.set_lod([lod])

            numpy_data = numpy_data.reshape([numpy_data.shape[0], 1])
            tensor.set(numpy_data, place)
            feed_tensors[key] = tensor
        return feed_tensors

    PASS_NUM = 5
    for pass_id in range(PASS_NUM):
        batch_id = 0
        ts = time.time()
        for data in train_reader():
            outs = exe.run(framework.default_main_program(),
                           feed=func_feed(feeding, data),
                           fetch_list=[cost, u, m])
            out = np.array(outs[0])
            if batch_id % 100 == 0:
                print("pass %d, batch %d, cost: %f"%(pass_id, batch_id, out[0]))
                print(outs[1])
                print(outs[2])
            batch_id += 1
            # if out[0] < 6.0:
            #     # if avg cost less than 6.0, we think our code is good.
            #     exit(0)
        print("pass %d, cost: %f, time: %f"%(pass_id, out[0], time.time() - ts))

main()
