from vector_metrics_and_viz import VecViz

vec_viz = VecViz.from_pickle('Glove_vecs_100D.pkl')

vecs = vec_viz.get_cosine_distances(['man', 'woman'], ['woman', 'man'])

new_vec = vec_viz.get_vectors(['king', 'man', 'woman'])

king_queen = new_vec[0] - new_vec[1] + new_vec[2]

vec_viz.vec_heatmap(

    tokens=['man', 'woman', 'king', 'queen'],
    figsize=(10,5),
    title='Heatmap of Word Vectors',
    vec_dict={'k-m+q': king_queen.reshape(1,-1)}

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


from scipy.spatial import distance

target_vec = vec_viz.get_vectors(['man'])[0]

[distance.cosine(target_vec, vec) for vec in vec_viz.vec_matrix]


