import os
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from time import time
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names

from mmoe import MMOE
from evaluation import evaluate_deepctr

# GPU相关设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 设置GPU按需增长
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='WeChat-Challenge')
    parser.add_argument('--debug_mode', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--base_dir', type=str,
                        default='/Users/jiananliu/Desktop/work/tencent/analyze/table',
                        help='please set')
    parser.add_argument('--epoch_num', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--embedding_dim', type=int, default=16)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    epochs = args.epoch_num
    batch_size = args.batch_size
    debug_mode = args.debug_mode
    embedding_dim = args.embedding_dim

    feed_info_file = './data/wechat_algo_data1/feed_info.csv'
    feed_embeddings_file = './data/wechat_algo_data1/feed_embeddings.csv'

    if debug_mode:
        train_file = './data/wechat_algo_data1/user_action_sample_100.csv'
        test_file = './data/wechat_algo_data1/test_a_sample_100.csv'
    else:
        train_file = './data/wechat_algo_data1/user_action.csv'
        test_file = './data/wechat_algo_data1/test_a.csv'

    item_df = pd.read_csv(feed_info_file)
    item_embeddings_df = pd.read_csv(feed_embeddings_file)
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    item_df[["bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    item_df[["bgm_song_id", "bgm_singer_id"]] = item_df[["bgm_song_id", "bgm_singer_id"]].fillna(0)
    item_df['bgm_song_id'] = item_df['bgm_song_id'].astype(int)
    item_df['bgm_singer_id'] = item_df['bgm_singer_id'].astype(int)

    def get_vocab_size(feat):
        return max(item_df[feat].apply(lambda x: max(x) if list(x) else 0)) + 1

    def get_maxlen(feat):
        return max(item_df[feat].apply(lambda x: len(x)))

    item_varlen_sparse_features = ['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list']
    for feat in item_varlen_sparse_features:
        item_df[feat] = item_df[feat].apply(lambda x: np.fromstring(x, "i", sep=';') if x is not np.NAN else np.array([], dtype=np.int32))
        item_df[feat] = pad_sequences(item_df[feat], maxlen=get_maxlen(feat), padding='post', dtype=np.int32, value=0).tolist()

    def get_value(l, i):
        return [v_w.strip().split(' ')[i] for v_w in l]

    item_weighted_varlen_sparse_features = ['machine_tag_list']
    for feat in item_weighted_varlen_sparse_features:
        item_df[feat] = item_df[feat].apply(lambda x: x.strip().split(';') if x is not np.NAN else np.array([], dtype=np.int32))
        feat_weight = feat + '_weight'
        item_df[feat_weight] = item_df[feat].apply(lambda x: np.asarray(get_value(x, 1), dtype=float))
        item_df[feat_weight] = pad_sequences(item_df[feat_weight], maxlen=get_maxlen(feat_weight), padding='post', dtype=float, value=0).tolist()
        item_df[feat] = item_df[feat].apply(lambda x: np.asarray(get_value(x, 0), dtype=np.int32))
        item_df[feat] = pad_sequences(item_df[feat], maxlen=get_maxlen(feat), padding='post', dtype=np.int32, value=0).tolist()

    agg = train_df.groupby('feedid')
    item_posterior_cols = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow'] + ['play', 'stay']
    for feat in item_posterior_cols:
        item_df[feat + '_mean'] = agg[feat].transform(np.mean)
        item_df[feat + '_min'] = agg[feat].transform(np.min)
        item_df[feat + '_max'] = agg[feat].transform(np.max)
        item_df[feat + '_std'] = agg[feat].transform(np.std)
        item_df[feat + '_median'] = agg[feat].transform(np.median)

    item_prior_feat = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id'] + item_varlen_sparse_features + item_weighted_varlen_sparse_features + [feat + '_weight' for feat in item_weighted_varlen_sparse_features]
    posterior_item_cols = [feat + poster for poster in ('_mean', '_min', '_max', '_std', '_median') for feat in item_posterior_cols]
    item_cols = item_prior_feat + posterior_item_cols

    train_df = train_df.merge(item_df[item_cols], how='left', on='feedid')
    test_df = test_df.merge(item_df[item_cols], how='left', on='feedid')

    # -------------------------------------------
    sparse_features = ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    # sparse_features = ['userid', 'device', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    varlen_sparse_features = ['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list']
    weighted_varlen_sparse_features = ['machine_tag_list']

    dense_features = ['videoplayseconds'] + posterior_item_cols
    # embedding_features = ['feed_embedding']

    target_weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                   "comment": 1, "follow": 1}
    # target = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
    target = ["read_comment", "like", "click_avatar", "forward"]
    # target = ["read_comment", "like"]
    print('-----------------target class percent---------------------')
    for target_feat in target:
        print(f'{target_feat}\trate:{"%.2f" % (len(train_df[train_df[target_feat] == 1])/len(train_df)*100)}%')

    # 1.fill nan dense_feature and do simple Transformation for dense features
    train_df[dense_features] = train_df[dense_features].fillna(0, )
    test_df[dense_features] = test_df[dense_features].fillna(0, )

    # item_df.info(memory_usage=True)
    # print(item_df.memory_usage(deep=True))
    print('---------')

    train_df[dense_features] = np.log(train_df[dense_features] + 1.0)
    test_df[dense_features] = np.log(test_df[dense_features] + 1.0)

    print('data.shape', train_df.shape)
    print('data.columns', train_df.columns.tolist())
    print('unique date_: ', sorted(train_df['date_'].unique()))

    train_df = train_df.sample(frac=1).reset_index(drop=True)
    train = train_df[train_df['date_'] < 14]
    val = train_df[train_df['date_'] == 14]  # 第14天样本作为验证集

    b_use_feed_embedding = False
    if b_use_feed_embedding:
        item_embeddings_df['feed_embedding'] = item_embeddings_df['feed_embedding'].apply(lambda x: np.fromstring(x, "f", sep=' '))
        # https://www.cnblogs.com/xiaoqi/p/deepfm.html
        # https://keras.io/examples/nlp/pretrained_word_embeddings/
        embeddings_index = {}
        for i, row in item_embeddings_df.iterrows():
            feedid = row['feedid']
            feed_embedding = row['feed_embedding']
            # coefs = np.fromstring(row['feed_embedding'], "f", sep=" ")
            embeddings_index[feedid] = np.asarray(feed_embedding, dtype=float)

        print("Found %s feed vectors." % len(embeddings_index))

        num_tokens = item_embeddings_df.feedid.max() + 1
        feed_embedding_dim = len(item_embeddings_df.feed_embedding[0])
        # Prepare embedding matrix
        embedding_matrix = np.zeros((num_tokens, feed_embedding_dim))
        for feedid, embedding in embeddings_index.items():
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[feedid] = embedding

        pretrained_item_weights = embedding_matrix
        pretrained_weights_initializer = tf.initializers.identity(pretrained_item_weights)
        pretrained_feat_columns = [SparseFeat('feedid', pretrained_item_weights.shape[0], embedding_dim=pretrained_item_weights.shape[1], embeddings_initializer=pretrained_weights_initializer, trainable=False)]
    else:
        pretrained_feat_columns = []

    # 2.count #unique features for each sparse field,and record dense feature field name
    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=train_df[feat].max() + 1, embedding_dim=embedding_dim) for feat in sparse_features]
    dense_feature_columns = [DenseFeat(feat, dimension=1) for feat in dense_features] + pretrained_feat_columns
    varlen_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=get_vocab_size(feat), embedding_dim=embedding_dim), maxlen=get_maxlen(feat), combiner='mean', weight_name=None) for feat in varlen_sparse_features]
    weighted_varlen_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=get_vocab_size(feat), embedding_dim=embedding_dim), maxlen=get_maxlen(feat), combiner='mean', weight_name=feat + '_weight') for feat in weighted_varlen_sparse_features]

    fixlen_feature_columns = sparse_feature_columns + dense_feature_columns + varlen_feature_columns + weighted_varlen_feature_columns

    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)

    # 3.generate input data for model
    train_model_input = {name: train[name] for name in feature_names}
    val_model_input = {name: val[name] for name in feature_names}
    userid_list = val['userid'].astype(str).tolist()
    test_model_input = {name: test_df[name] for name in feature_names}
    for feat in varlen_sparse_features:
        # train_model_input[feat] = train_model_input[feat].apply(lambda x: np.asarray(x)).to_numpy()
        train_model_input[feat] = np.asarray(train_model_input[feat].to_list()).astype(np.int32)
        val_model_input[feat] = np.asarray(val_model_input[feat].to_list()).astype(np.int32)
        test_model_input[feat] = np.asarray(test_model_input[feat].to_list()).astype(np.int32)

    for feat in weighted_varlen_sparse_features:
        feat_weight = feat + '_weight'
        train_model_input[feat] = np.asarray(train_model_input[feat].to_list()).astype(np.int32)
        train_model_input[feat_weight] = np.asarray(train_model_input[feat_weight].to_list()).astype(float)
        val_model_input[feat] = np.asarray(val_model_input[feat].to_list()).astype(np.int32)
        val_model_input[feat_weight] = np.asarray(val_model_input[feat_weight].to_list()).astype(float)
        test_model_input[feat] = np.asarray(test_model_input[feat].to_list()).astype(np.int32)
        test_model_input[feat_weight] = np.asarray(test_model_input[feat_weight].to_list()).astype(float)

    train_labels = [train[y].values for y in target]
    val_labels = [val[y].values for y in target]

    num_task = len(target)
    loss_weights = [target_weight_dict[i] for i in target]
    print(f'target: {target}, loss weights: {loss_weights}')
    # 4.Define Model,train,predict and evaluate
    train_model = MMOE(dnn_feature_columns,
                       num_tasks=num_task,
                       expert_dim=8,
                       dnn_hidden_units=(128, 128),
                       task_dnn_units=[8, 8],
                       tasks=['binary'] * num_task)

    # from deepctr.models import DeepFM
    # model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')

    # tf.keras.utils.plot_model(train_model, "mmoe_model.png", show_shapes=True, expand_nested=True)

    # todo: https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
    # train_model.compile("adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=False))
    # adagrad"
    train_model.compile("adam", loss='binary_crossentropy', loss_weights=loss_weights)
    print(train_model.summary())
    for epoch in range(epochs):
        history = train_model.fit(train_model_input, train_labels,
                                  batch_size=batch_size, epochs=1, verbose=1,
                                  # class_weight={0: 1, 1: 10} # ValueError: `class_weight` is only supported for Models with a single output.
                                  )

        val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
        evaluate_deepctr(val_labels, val_pred_ans, userid_list, target)

    t1 = time()
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 20)
    t2 = time()
    print('4个目标行为%d条样本预测耗时（毫秒）：%.3f' % (len(test_df), (t2 - t1) * 1000.0))
    ts = (t2 - t1) * 1000.0 / len(test_df) * 2000.0
    print('4个目标行为2000条样本平均预测耗时（毫秒）：%.3f' % ts)

    # 5.生成提交文件
    for i, action in enumerate(target):
        test_df[action] = pred_ans[i]
    test_df[['userid', 'feedid'] + target].to_csv('result.csv', index=None, float_format='%.6f')
    print('to_csv ok')
