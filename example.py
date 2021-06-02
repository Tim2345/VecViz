from vector_metrics_and_viz import VecViz

vec_viz = VecViz.from_pickle('Glove_vecs_100D.pkl')

vec_viz.vec_heatmap(
    tokens=['man', 'woman', 'king', 'queen'],
    figsize=(10,5),
    title='Heatmap of Word Vectors',
    vec_dict={'china': vec_viz.get_vectors(['china'])}
)


tokens = ['england', 'london', 'germany', 'berlin', 'italy', 'rome', 'spain', 'madrid', 'france', 'paris']

vec_viz.vec_scatterplot(
    tokens=tokens,
    n_tokens=len(tokens),
    perplexity=30,
    n_components=2,
    init='pca',
    n_iter=2500,
    random_state=43
)




import seaborn as sns
import numpy as np

sns.scatterplot(x=np.squeeze(vec_viz.get_vectors(['early'])), y=np.squeeze(vec_viz.get_vectors(['late'])))

arr = np.array([[1,2,3,4,5,6], [1,2,3,4,5,6], [1,2,3,4,5,6]])

vec_dict = {
    'this': np.array([1,2,3,4,5,6]),
    'that': np.array([1,2,3,4,5,6]),
    'these': np.array([1,2,3,4,5,6])
}

np.array(list(vec_dict.values()))

listy = ['peter', 'paul']
listy.extend(vec_dict.keys())
print(listy)