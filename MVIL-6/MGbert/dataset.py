import pandas as pd
import numpy as np
from utils import smiles2adjoin
import tensorflow as tf

from rdkit import Chem
"""     

{'O': 5000757, 'C': 34130255, 'N': 5244317, 'F': 641901, 'H': 37237224, 'S': 648962, 
'Cl': 373453, 'P': 26195, 'Br': 76939, 'B': 2895, 'I': 9203, 'Si': 1990, 'Se': 1860, 
'Te': 104, 'As': 202, 'Al': 21, 'Zn': 6, 'Ca': 1, 'Ag': 3}

H C N O F S  Cl P Br B I Si Se
"""

str2num = {'<pad>': 0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'Cl': 7, 'P': 8, 'Br': 9,
           'B': 10, 'I': 11, 'Si': 12, 'Se': 13, '<unk>': 14, '<mask>': 15, '<global>': 16}

num2str = {i: j for j, i in str2num.items()}


class Graph_Bert_Dataset(object):
    def __init__(self, path, smiles_field='Smiles', addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.vocab = str2num
        self.devocab = num2str
        self.addH = addH

    def get_data(self):

        data = self.df
        train_idx = []
        idx = data.sample(frac=0.9).index
        train_idx.extend(idx)

        data1 = data[data.index.isin(train_idx)]
        data2 = data[~data.index.isin(train_idx)]

        self.dataset1 = tf.data.Dataset.from_tensor_slices(data1[self.smiles_field].tolist())
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(32, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None]))).prefetch(50)

        self.dataset2 = tf.data.Dataset.from_tensor_slices(data2[self.smiles_field].tolist())
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None]))).prefetch(50)
        return self.dataset1, self.dataset2

    def numerical_smiles(self, smiles):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        choices = np.random.permutation(len(nums_list) - 1)[:max(int(len(nums_list) * 0.15), 1)] + 1
        y = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = str2num['<mask>']
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 14 + 1)

        x = np.array(nums_list).astype('int64')
        weight = weight.astype('float32')
        return x, adjoin_matrix, y, weight

    def tf_numerical_smiles(self, data):
        # x,adjoin_matrix,y,weight = tf.py_function(self.balanced_numerical_smiles,
        #                                           [data], [tf.int64, tf.float32 ,tf.int64,tf.float32])
        x, adjoin_matrix, y, weight = tf.py_function(self.numerical_smiles, [data],
                                                     [tf.int64, tf.float32, tf.int64, tf.float32])

        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        y.set_shape([None])
        weight.set_shape([None])
        return x, adjoin_matrix, y, weight


class Graph_Classification_Dataset(object):
    def __init__(self, path, smiles_field='SMILES', label_field='Label', shuzi_field='index2', addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.shuzi_field = shuzi_field
        self.vocab = str2num
        self.devocab = num2str
        self.addH = addH

    def get_data(self):
        # data = self.df
        # data = data.dropna()
        # data[self.label_field] = data[self.label_field].map(int)
        # pdata = data[data[self.label_field] == 1]
        # ndata = data[data[self.label_field] == 0]
        # lengths = [0, 25, 50, 75, 100]
        # # lengths = [0, 100, 200, 300, 400]
        #
        # ptrain_idx = []
        # for i in range(4):
        #     idx = pdata[(pdata[self.smiles_field].str.len() >= lengths[i]) & (
        #             pdata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
        #     ptrain_idx.extend(idx)
        #
        # ntrain_idx = []
        # for i in range(4):
        #     idx = ndata[(ndata[self.smiles_field].str.len() >= lengths[i]) & (
        #             ndata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
        #     ntrain_idx.extend(idx)
        #
        # train_data = data[data.index.isin(ptrain_idx + ntrain_idx)]
        # pdata = pdata[~pdata.index.isin(ptrain_idx)]
        # ndata = ndata[~ndata.index.isin(ntrain_idx)]
        #
        # ptest_idx = []
        # for i in range(4):
        #     idx = pdata[(pdata[self.smiles_field].str.len() >= lengths[i]) & (
        #             pdata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
        #     ptest_idx.extend(idx)
        #
        # ntest_idx = []
        # for i in range(4):
        #     idx = ndata[(ndata[self.smiles_field].str.len() >= lengths[i]) & (
        #             ndata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
        #     ntest_idx.extend(idx)
        #
        # test_data = data[data.index.isin(ptest_idx + ntest_idx)]
        # val_data = data[~data.index.isin(ptest_idx + ntest_idx + ptrain_idx + ntrain_idx)]

        #------------------------------改过的----------------------------------
        data = self.df.dropna()
        print(data)

        data[self.label_field] = data[self.label_field].map(int)
        train_idx = [i for i in range(2685)]
        # train_idx = [i for i in range(4728)]
        train_data = data[data.index.isin(train_idx)]
        # print(len(train_data))
        val_index = [i for i in range(3944)][2685:]
        # val_index = [i for i in range(5399)][4729:]
        val_data = data[data.index.isin(val_index)]

        # print(len(val_data))
        test_index = []
        test_data = data[data.index.isin(test_index)]

        print('Training Oversampling Model')
        train_x = train_data[self.smiles_field]
        train_y = train_data[self.label_field]
        train_z = train_data[self.shuzi_field]
        self.dataset1 = tf.data.Dataset.from_tensor_slices((train_x, train_y, train_z))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles2).cache().padded_batch(32, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]), tf.TensorShape([1]))).cache().prefetch(100)


        # pos_indices = train_y == 1
        # pos_x, neg_x = train_x[pos_indices], train_x[~pos_indices]
        # pos_y, neg_y = train_y[pos_indices], train_y[~pos_indices]
        # pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).shuffle(50)
        # neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).shuffle(50)
        # resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.89, 0.11])
        # resampled_ds = resampled_ds.map(self.tf_numerical_smiles).cache().padded_batch(64, padded_shapes=(
        #     tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).shuffle(50).prefetch(100)
        #
        # ------------------------------改过的----------------------------------
        #
        # self.dataset1 = tf.data.Dataset.from_tensor_slices(
        #     (train_data[self.smiles_field], train_data[self.label_field]))
        # self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(32, padded_shapes=(
        #     tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((test_data[self.smiles_field], test_data[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(32, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((val_data[self.smiles_field], val_data[self.label_field], val_data[self.shuzi_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles2).padded_batch(32, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]), tf.TensorShape([1]))).cache().prefetch(100)

        return self.dataset1, self.dataset2, self.dataset3

    def numerical_smiles2(self, smiles, label, shuzi):
        #
        # m = smiles.numpy()
        # # n = m.tolist()
        # print(m.size())

        # mol = Chem.MolFromSmiles(m)
        # seqs = Chem.MolToSequence(mol)
        # print(seqs)

        smiles = smiles.numpy().decode()
        # n = smiles.split('\n')
        # print(n[0])
        #
        # smiles1 = 'CC(C)OC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3'
        # mol1 = Chem.MolFromSmiles(smiles1)
        # seqs1 = Chem.MolToSequence(mol1)
        # print(seqs1)
        #
        #
        # mol = Chem.MolFromSmiles(n[0])
        # seqs = Chem.MolToSequence(mol)
        # print('-------'*20)
        #
        # print(seqs)

        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        x = np.array(nums_list).astype('int64')
        y = np.array([label]).astype('int64')
        z = np.array([shuzi]).astype('int64')
        return x, adjoin_matrix, y, z

    def tf_numerical_smiles2(self, smiles, label, shuzi):
        x, adjoin_matrix, y, z = tf.py_function(self.numerical_smiles2, [smiles, label, shuzi], [tf.int64, tf.float32, tf.int64, tf.int64])
        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        y.set_shape([None])
        return x, adjoin_matrix, y, z

    def numerical_smiles(self, smiles,label):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)

        x = np.array(nums_list).astype('int64')
        y = np.array([label]).astype('int64')
        return x, adjoin_matrix,y

    def tf_numerical_smiles(self, smiles,label):
        x,adjoin_matrix,y = tf.py_function(self.numerical_smiles, [smiles,label], [tf.int64, tf.float32 ,tf.int64])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        return x, adjoin_matrix, y


class Graph_Regression_Dataset(object):
    def __init__(self, path, smiles_field='Smiles', label_field='Label', normalize=True, max_len=100, addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.addH = addH
        if normalize:
            self.max = self.df[self.label_field].max()
            self.min = self.df[self.label_field].min()
            self.df[self.label_field] = (self.df[self.label_field] - self.min) / (self.max - self.min) - 0.5
            self.value_range = self.max - self.min

    def get_data(self):
        data = self.df

        lengths = [0, 25, 50, 75, 100]

        train_idx = []
        for i in range(4):
            idx = data[(data[self.smiles_field].str.len() >= lengths[i]) & (
                    data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
            train_idx.extend(idx)

        train_data = data[data.index.isin(train_idx)]
        data = data[~data.index.isin(train_idx)]

        test_idx = []
        for i in range(4):
            idx = data[(data[self.smiles_field].str.len() >= lengths[i]) & (
                    data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
            test_idx.extend(idx)

        test_data = data[data.index.isin(test_idx)]
        val_data = data[~data.index.isin(test_idx)]

        self.dataset1 = tf.data.Dataset.from_tensor_slices(
            (train_data[self.smiles_field], train_data[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).shuffle(100).prefetch(100)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((test_data[self.smiles_field], test_data[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((val_data[self.smiles_field], val_data[self.label_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)

        return self.dataset1, self.dataset2, self.dataset3

    def numerical_smiles(self, smiles, label):
        smiles = smiles.numpy().decode()
        print(smiles)
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        x = np.array(nums_list).astype('int64')
        y = np.array([label]).astype('float32')
        return x, adjoin_matrix, y

    def tf_numerical_smiles(self, smiles, label):
        x, adjoin_matrix, y = tf.py_function(self.numerical_smiles, [smiles, label], [tf.int64, tf.float32, tf.float32])
        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        y.set_shape([None])
        return x, adjoin_matrix, y


class Inference_Dataset(object):
    def __init__(self, sml_list, max_len=100, addH=True):
        self.vocab = str2num
        self.devocab = num2str
        self.sml_list = [i for i in sml_list if len(i) < max_len]
        self.addH = addH

    def get_data(self):
        self.dataset = tf.data.Dataset.from_tensor_slices((self.sml_list,))
        self.dataset = self.dataset.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]),
            tf.TensorShape([None]))).cache().prefetch(20)

        return self.dataset

    def numerical_smiles(self, smiles):
        smiles_origin = smiles
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)
        x = np.array(nums_list).astype('int64')
        return x, adjoin_matrix, [smiles], atoms_list

    def tf_numerical_smiles(self, smiles):
        x, adjoin_matrix, smiles, atom_list = tf.py_function(self.numerical_smiles, [smiles],
                                                             [tf.int64, tf.float32, tf.string, tf.string])
        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        smiles.set_shape([1])
        atom_list.set_shape([None])
        return x, adjoin_matrix, smiles, atom_list


# class multi_task_dataset(object):
#     def __init__(self,path_list,smiles_field,label_field,max_len=100,addH=True):
#         self.vocab = str2num
#         self.smiles_field = smiles_field
#         self.label_field = label_field
#         self.devocab = num2str
#         self.addH =  addH
#         self.pathlist = path_list
#
#     def get_data(self):
#         x_train_list = []
#         y_train_list = []
#         mask_train_list=[]
#         test_dataset_list = []
#         for i,path in enumerate(self.pathlist):
#             data = pd.read_csv(path,sep='\t')
#             lengths = [0, 25, 50, 75, 100]
#             train_idx = []
#             for ii in range(4):
#                 idx = data[(data[self.smiles_field].str.len() >= lengths[ii]) & (
#                         data[self.smiles_field].str.len() < lengths[ii + 1])].sample(frac=0.8).index
#                 train_idx.extend(idx)
#             data1 = data[data.index.isin(train_idx)].copy()
#             data2 = data[~data.index.isin(train_idx)].copy()
#             x_train_list += data1[self.smiles_field].tolist()
#             y_train = -np.ones((len(data1),len(self.pathlist))).astype('float32')
#             mask_train = np.zeros((len(data1),len(self.pathlist))).astype('float32')
#
#             y_train[:,i] = np.array(data1[self.label_field])
#             mask_train[:, i] = 1
#
#             y_train_list.append(y_train)
#             mask_train_list.append(mask_train)
#
#             x_test = data2[self.smiles_field].tolist()
#             y_test = -np.ones((len(data2),len(self.pathlist))).astype('float32')
#             y_test[:,i] = np.array(data2[self.label_field])
#             mask_test = np.zeros((len(data2),len(self.pathlist))).astype('float32')
#             mask_test[:, i] = 1
#             test_dataset_list.append(tf.data.Dataset.from_tensor_slices((x_test,y_test,mask_test)).map(self.tf_numerical_smiles).padded_batch(256,
#                                                                 padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None, None]),
#                                                                 tf.TensorShape([None]),tf.TensorShape([None]))).cache().prefetch(100))
#
#         y_train_list = np.concatenate(y_train_list,axis=0)
#         mask_train_list = np.concatenate(mask_train_list,axis=0)
#
#         dataset1 = tf.data.Dataset.from_tensor_slices((x_train_list,y_train_list,mask_train_list))
#         dataset1 = dataset1.map(self.tf_numerical_smiles).shuffle(200).padded_batch(64, padded_shapes=(
#             tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([len(self.pathlist)]),
#             tf.TensorShape([len(self.pathlist)]))).cache().prefetch(100)
#         return dataset1, test_dataset_list
#
#     def numerical_smiles(self, smiles,y,y_mask):
#         smiles = smiles.numpy().decode()
#         atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
#         atoms_list = ['<global>'] + atoms_list
#         nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
#         temp = np.ones((len(nums_list),len(nums_list)))
#         temp[1:,1:] = adjoin_matrix
#         adjoin_matrix = ((1-temp)*(-1e9)).astype('float32')
#         x = np.array(nums_list).astype('int64')
#         return x, adjoin_matrix,y,y_mask
#
#     def tf_numerical_smiles(self, smiles,y,y_mask):
#         x,adjoin_matrix,y, y_mask = tf.py_function(self.numerical_smiles, [smiles,y,y_mask], [tf.int64, tf.float32,tf.float32, tf.float32])
#         x.set_shape([None])
#         adjoin_matrix.set_shape([None,None])
#         y.set_shape([len(self.pathlist)])
#         y_mask.set_shape([len(self.pathlist)])
#         return x, adjoin_matrix,y, y_mask


class Graph_Regression_and_Pretraining_Dataset(object):
    def __init__(self, path, smiles_field='Smiles', label_field='Label', normalize=True, addH=True, max_len=100):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.addH = addH
        if normalize:
            self.max = self.df[self.label_field].max()
            self.min = self.df[self.label_field].min()
            self.df[self.label_field] = (self.df[self.label_field] - self.min) / (self.max - self.min) - 0.5

    def get_data(self):
        data = self.df
        lengths = [0, 25, 50, 75, 100]
        train_idx = []
        for i in range(4):
            idx = data[(data[self.smiles_field].str.len() >= lengths[i]) & (
                    data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
            train_idx.extend(idx)

        data1 = data[data.index.isin(train_idx)]
        data2 = data[~data.index.isin(train_idx)]

        self.dataset1 = tf.data.Dataset.from_tensor_slices((data1[self.smiles_field], data1[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([1]), tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None]))).cache().shuffle(100).prefetch(100)
        self.dataset2 = tf.data.Dataset.from_tensor_slices((data2[self.smiles_field], data2[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([1]), tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None]))).cache().prefetch(100)
        return self.dataset1, self.dataset2

    def numerical_smiles(self, smiles, label):

        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        choices = np.random.permutation(len(nums_list) - 1)[:max(int(len(nums_list) * 0.15), 1)] + 1
        x_true = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = str2num['<mask>']
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 14 + 1)

        x_masked = np.array(nums_list).astype('int64')
        weight = weight.astype('int64')
        label = np.array([label]).astype('float32')
        return x_masked, label, adjoin_matrix, x_true, weight

    def tf_numerical_smiles(self, smiles, label):
        x, label, adjoin_matrix, y, weight = tf.py_function(self.numerical_smiles, [smiles, label],
                                                            [tf.int64, tf.float32, tf.float32, tf.int64, tf.int64])
        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        y.set_shape([None])
        weight.set_shape([None])
        label.set_shape([None])
        return x, label, adjoin_matrix, y, weight
