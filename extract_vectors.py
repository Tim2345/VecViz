import numpy as np
import pickle

# state vec dimensions to load from glove
vec_dim = '100'
vec_file_path = './glove.6B/glove.6B.{}d.txt'.format(vec_dim)


# load the whole embedding into memory
vecs_dict = dict()


f = open(vec_file_path, encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype=np.float32)
    vecs_dict[word] = coefs
f.close()

# pickle file and save - this takes a while!!
vec_dict_save_name = 'Glove_Vecs_{}D.pkl'.format(vec_dim)

with open(vec_dict_save_name, 'wb') as handle:
    pickle.dump(vecs_dict, handle)
