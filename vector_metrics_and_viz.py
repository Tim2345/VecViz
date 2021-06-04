import pickle

import numpy as np

from sklearn.manifold import TSNE
from scipy.spatial import distance

import matplotlib.pyplot as plt
import seaborn as sns


class VecViz:

    def __init__(self, vec_matrix, vec_indices):
        self.vec_matrix = vec_matrix
        self.vec_indices = vec_indices

    @classmethod
    def from_pickle(cls, pickle_path):
        with open(pickle_path, "rb") as handle:
            vec_dict = pickle.load(handle)
        print('loaded_dict')

        vec_list = list(vec_dict.values())
        vec_matrix = np.array(vec_list)

        vec_indices = {word: idx for word, idx in zip(vec_dict.keys(), np.arange(len(vec_dict)))}

        return cls(vec_matrix, vec_indices)

    @classmethod
    def from_dict(cls, vec_dict):

        vec_list = list(vec_dict.values())
        vec_matrix = np.array(vec_list)

        return cls(vec_matrix)

    def get_indices(self, tokens):
        indices = [self.vec_indices[token] for token in tokens]

        return np.array(indices)

    def get_vectors(self, tokens):
        indices = self.get_indices(tokens)

        return self.vec_matrix[indices]

    def get_cosine_distances(self, tokens_x, tokens_y, out=dict):
        if not any((isinstance(tokens_x, list), isinstance(tokens_y, list))):
            raise ValueError("Both tokens_x and 'tokens_y' must be 'list' type.")

        output = []
        for tx in tokens_x:
            print(tx)
            for ty in tokens_y:
                print(ty)
                tokens = tx + '_' + ty
                cos_dist_temp = distance.cosine(self.get_vectors([tx]).reshape(1,-1), self.get_vectors([ty]).reshape(1,-1))
                output.append((tokens, cos_dist_temp))

        output = out(output)

        return output

    def vec_scatterplot(self, tokens, n_tokens, fig_size=(20, 20), *args, **kwargs):

        ## get vectors to pass to TSNE
        word_indices = self.get_indices(tokens)
        token_indices = np.arange(self.vec_matrix.shape[0])
        token_indices = np.delete(token_indices, word_indices)
        token_indices = np.random.choice(token_indices, n_tokens - len(tokens))
        indices = np.concatenate([word_indices, token_indices])
        indices = np.sort(indices)
        vectors = self.vec_matrix[indices]
        vec_tokens = np.array(list(self.vec_indices.keys()))
        vec_tokens = vec_tokens[indices]

        tsne_model = TSNE(*args, **kwargs)
        new_values = tsne_model.fit_transform(vectors)

        plt.figure(figsize=fig_size)
        for x, y, token in zip(new_values[:, 0], new_values[:, 1], vec_tokens):

            plt.scatter(x, y)
            plt.annotate(
                token,
                xy=(x, y),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')

            plt.title('TSNE plot of word vectors')

        plt.show()


    def vec_heatmap(self, tokens, figsize, title, vec_dict=None):
        heatmap_vecs = self.get_vectors(tokens)

        if vec_dict:
            extra_vecs = np.squeeze(np.array(list(vec_dict.values())), axis=0)
            heatmap_vecs = np.concatenate([heatmap_vecs, extra_vecs])
            tokens.extend(vec_dict.keys())

        plt.figure(figsize=figsize)
        sns.heatmap(heatmap_vecs, cbar=True, xticklabels=False, yticklabels=tokens, cmap="YlGnBu")
        plt.title(title)

        plt.show()


