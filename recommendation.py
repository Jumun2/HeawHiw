import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_data():
    food_data = pd.read_csv('dataset/NewestDATA2.csv.zip')
    food_data['menu_title'] = food_data['menu_title'].str.lower()
    return food_data


def combine_data(data):
    data_recommend = data.drop(columns=['food_id', 'menu_title', 'details'])
    data_recommend['combine'] = data_recommend[data_recommend.columns[0:2]].apply(
        lambda x: ','.join(x.dropna().astype(str)), axis=1)

    data_recommend = data_recommend.drop(
        columns=['genres', 'ingredient', 'nutrients'])
    return data_recommend


def transform_data(data_combine, data_details):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(data_combine['combine'])

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data_details['details'])

    combine_sparse = sp.hstack([count_matrix, tfidf_matrix], format='csr')
    cosine_sim = cosine_similarity(combine_sparse, combine_sparse)

    return cosine_sim


def recommend_food(title, data, combine, transform):
    indices = pd.Series(data.index, index=data['menu_title'])
    index = indices[title]

    sim_scores = list(enumerate(transform[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]

    movie_indices = [i[0] for i in sim_scores]

    food_id = data['food_id'].iloc[movie_indices]
    movie_title = data['menu_title'].iloc[movie_indices]


    recommendation_data = pd.DataFrame(
        columns=['food_id', 'menu_title'])

    recommendation_data['food_id'] = food_id
    recommendation_data['menu_title'] = movie_title

    return recommendation_data


def results(menu_recommends):
    menu_recommends = menu_recommends.lower()

    find_menu = get_data()
    combine_result = combine_data(find_menu)
    transform_result = transform_data(combine_result, find_menu)

    recommend_response = {
        "menu_items": []
    }

    if menu_recommends in find_menu["menu_title"].unique():
        recommend_response["menu_items"] = recommend_food(
            menu_recommends, 
            find_menu, 
            combine_result, 
            transform_result
        ).to_dict("records")

    return recommend_response
