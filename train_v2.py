import paddle.v2 as paddle
import cPickle
import copy
import os
import time

ts = time.time()
USE_GPU = False

def get_usr_combined_features():
    uid = paddle.layer.data(
        name='user_id',
        type=paddle.data_type.integer_value(
            paddle.dataset.movielens.max_user_id() + 1))
    usr_emb = paddle.layer.embedding(input=uid, size=32)
    usr_fc = paddle.layer.fc(input=usr_emb, size=32)

    usr_gender_id = paddle.layer.data(
        name='gender_id', type=paddle.data_type.integer_value(2))
    usr_gender_emb = paddle.layer.embedding(input=usr_gender_id, size=16)
    usr_gender_fc = paddle.layer.fc(input=usr_gender_emb, size=16)

    usr_age_id = paddle.layer.data(
        name='age_id',
        type=paddle.data_type.integer_value(
            len(paddle.dataset.movielens.age_table)))
    usr_age_emb = paddle.layer.embedding(input=usr_age_id, size=16)
    usr_age_fc = paddle.layer.fc(input=usr_age_emb, size=16)

    usr_job_id = paddle.layer.data(
        name='job_id',
        type=paddle.data_type.integer_value(
            paddle.dataset.movielens.max_job_id() + 1))
    usr_job_emb = paddle.layer.embedding(input=usr_job_id, size=16)
    usr_job_fc = paddle.layer.fc(input=usr_job_emb, size=16)

    usr_combined_features = paddle.layer.fc(
        name="user_all_fc",
        input=[usr_fc, usr_gender_fc, usr_age_fc, usr_job_fc],
        size=200,
        act=paddle.activation.Tanh())
    return usr_combined_features


def get_mov_combined_features():
    movie_title_dict = paddle.dataset.movielens.get_movie_title_dict()
    mov_id = paddle.layer.data(
        name='movie_id',
        type=paddle.data_type.integer_value(
            paddle.dataset.movielens.max_movie_id() + 1))
    mov_emb = paddle.layer.embedding(input=mov_id, size=32)
    mov_fc = paddle.layer.fc(input=mov_emb, size=32)

    # mov_categories = paddle.layer.data(
    #     name='category_id',
    #     type=paddle.data_type.sparse_binary_vector(
    #         len(paddle.dataset.movielens.movie_categories())))
    # mov_categories_hidden = paddle.layer.fc(input=mov_categories, size=32)

    mov_title_id = paddle.layer.data(
        name='movie_title',
        type=paddle.data_type.integer_value_sequence(len(movie_title_dict)))
    mov_title_emb = paddle.layer.embedding(input=mov_title_id, size=32)
    mov_title_conv = paddle.networks.sequence_conv_pool(
        input=mov_title_emb, hidden_size=32, context_len=3)

    mov_combined_features = paddle.layer.fc(
        name="mov_all_fc",
        # input=[mov_fc, mov_categories_hidden, mov_title_conv],
        input=[mov_fc, mov_title_conv],
        size=200,
        act=paddle.activation.Tanh())
    return mov_combined_features


def main():
    global ts
    paddle.init(use_gpu=USE_GPU, seed=1)
    usr_combined_features = get_usr_combined_features()
    mov_combined_features = get_mov_combined_features()
    inference = paddle.layer.cos_sim(
        a=usr_combined_features, b=mov_combined_features, size=1, scale=5)
    cost = paddle.layer.square_error_cost(
        input=inference,
        label=paddle.layer.data(
            name='score', type=paddle.data_type.dense_vector(1)))

    parameters = paddle.parameters.create(cost)

    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=paddle.optimizer.Adam(learning_rate=1e-4))
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
    ts = time.time()
    def event_handler(event):
        global ts
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d Batch %d Cost %.6f" % (
                    event.pass_id, event.batch_id, event.cost)
        elif isinstance(event, paddle.event.BeginPass):
            ts = time.time()
            print trainer.__parameters__.get("___embedding_0__.w0")
        elif isinstance(event, paddle.event.EndForwardBackward):
            if event.batch_id % 100 == 0:
                print event.gm.getLayerOutputs(['user_all_fc','mov_all_fc'])
        elif isinstance(event, paddle.event.EndPass):
            print "Pass %d time: %f" % (event.pass_id, time.time() - ts)

    trainer.train(
        reader=paddle.batch(
            paddle.dataset.movielens.train(),
            batch_size=256),
        event_handler=event_handler,
        feeding=feeding,
        num_passes=5)

if __name__ == '__main__':
    main()
