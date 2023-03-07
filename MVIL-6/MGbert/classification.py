import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np
# import tensorflow_core as tfc
from rdkit import Chem
from dataset import Graph_Classification_Dataset
from sklearn.metrics import r2_score, roc_auc_score

import os
from model import PredictModel, BertModel

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
keras.backend.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = "1"



def main(seed):
    # tasks = ['Ames', 'BBB', 'FDAMDD', 'H_HT', 'Pgp_inh', 'Pgp_sub']
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # tasks = ['BBB', 'FDAMDD',  'Pgp_sub']

    task = 'data'
    print(task)

    small = {'name': 'Small', 'num_layers': 3, 'num_heads': 4, 'd_model': 128, 'path': 'small_weights', 'addH':True}
    medium = {'name': 'Medium', 'num_layers': 6, 'num_heads': 4, 'd_model': 256, 'path': 'medium_weights', 'addH':True}
    large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 512, 'path': 'large_weights', 'addH':True}

    arch = small  ## small 3 4 128   medium: 6 6  256     large:  12 8 516
    pretraining = True
    pretraining_str = 'pretraining' if pretraining else ''

    trained_epoch = 8

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    addH = arch['addH']

    dff = d_model * 2
    vocab_size = 17
    dropout_rate = 0.1

    seed = seed
    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)
    train_dataset, test_dataset, val_dataset = Graph_Classification_Dataset('./data/clf/{}.csv'.format(task), smiles_field='SMILES',
                                                               label_field='Label', shuzi_field='index2', addH=True).get_data()

    x, adjoin_matrix, y, z = next(iter(train_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         dense_dropout=0.5)

    if pretraining:
        temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        # temp.build(input_shape=(32,86,17) )
        # # print(pred)
        temp.load_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))
        temp.encoder.save_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        del temp

        pred = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
        model.encoder.load_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        print('load_wieghts')


    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

    auc= -10
    stopping_monitor = 0
    for epoch in range(90):
        accuracy_object = tf.keras.metrics.BinaryAccuracy()
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        text1 = []

        for x,adjoin_matrix, y, z in train_dataset:
            # print(z)
            with tf.GradientTape() as tape:
                seq = tf.cast(tf.math.equal(x, 0), tf.float32)
                mask = seq[:, tf.newaxis, tf.newaxis, :]
                preds = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
                N_preds = tf.sigmoid(preds).numpy()

                text1.extend([[z[i], N_preds[i], y[i]] for i in range(y.__len__())])

                loss = loss_object(y,preds)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                accuracy_object.update_state(y,preds)
        print('epoch: ',epoch,'loss: {:.4f}'.format(loss.numpy().item()),'accuracy: {:.4f}'.format(accuracy_object.result().numpy().item()))



        y_true = []
        y_preds = []



        for x, adjoin_matrix, y, z in val_dataset:

            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x,mask=mask,adjoin_matrix=adjoin_matrix,training=False)

            y_true.append(y.numpy())
            y_preds.append(preds.numpy())
        # print('-------------------------------')
        # print(y_true)
        y_true = np.concatenate(y_true, axis=0).reshape(-1)
        # print(y_preds)
        y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
        y_preds = tf.sigmoid(y_preds).numpy()
        # print(y_preds)
        auc_new = roc_auc_score(y_true, y_preds)

        val_accuracy = keras.metrics.binary_accuracy(y_true.reshape(-1), y_preds.reshape(-1)).numpy()

        test_num = len(y_true.reshape(-1))
        #print(y_preds.reshape(-1))
        a = list(map(lambda x : 1 if x > 0.5 else 0, y_preds.reshape(-1)))
        # print(a)
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for index in range(test_num):
            if y_true.reshape(-1)[index] == 1:
                if y_true.reshape(-1)[index] == a[index]:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if y_true.reshape(-1)[index] == a[index]:
                    tn = tn + 1
                else:
                    fp = fp + 1
        # print()
        # precision
        if tp + fp == 0:
            Precision = 0
        else:
            Precision = float(tp) / (tp + fp)

        # SE
        if tp + fn == 0:
            Recall = Sensitivity = 0
        else:
            Recall = Sensitivity = float(tp) / (tp + fn)

        # SP
        if tn + fp == 0:
            Specificity = 0
        else:
            Specificity = float(tn) / (tn + fp)

        # MCC
        if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
            MCC = 0
        else:
            MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

        # F1-score
        if Recall + Precision == 0:
            F1 = 0
        else:
            F1 = 2 * Recall * Precision / (Recall + Precision)


        print('val auc:{:.4f}'.format(auc_new), 'val accuracy:{:.4f}'.format(val_accuracy), 'Precision:{:.4f}'.format(Precision),
              'Sensitivity:{:.4f}'.format(Sensitivity), 'Specificity:{:.4f}'.format(Specificity), 'MCC:{:.4f}'.format(MCC), 'F1:{:.4f}'.format(F1))


        model.save_weights('download_weight/{}_{}.h5'.format(seed, epoch))
        if auc_new > auc:
            auc = auc_new
            stopping_monitor = 0
            np.save('{}/{}{}{}{}{}'.format(arch['path'], task, seed, arch['name'], trained_epoch, trained_epoch,pretraining_str),
                    [y_true, y_preds])
            model.save_weights('classification_weights/{}_{}.h5'.format(task, seed))
            print('save model weights')
        else:
            stopping_monitor += 1
        print('best val auc: {:.4f}'.format(auc))
        if stopping_monitor>0:
            print('stopping_monitor:',stopping_monitor)
        if stopping_monitor>50:
            break

    # y_true = []
    # y_preds = []
    model.load_weights('classification_weights/{}_{}.h5'.format(task, seed))


if __name__ == '__main__':

    auc_list = []
    # for seed in [7,17,27,37,47,57,67,77,87,97]:
    for seed in [6]:
    # for seed in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        print(seed)
        auc = main(seed)
        auc_list.append(auc)
    print(auc_list)



