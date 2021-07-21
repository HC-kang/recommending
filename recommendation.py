import os 
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/recommendation/data_movie_lens')

import pandas as pd
import numpy as np
 
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

print(movies.shape)
print(ratings.shape)


# 영화 정보 데이터
print(movies.shape)
movies.head()

# 유저들의 영화별 평점 데이터
print(ratings.shape)
ratings.head()

### 사용자-아이템 평점 행렬로 변환

# 필요한 컬럼만 추출
ratings = ratings[['userId', 'movieId', 'rating']]

# pivot_table 로 행렬 변환
ratings_matrix = ratings.pivot_table('rating', index = 'userId', columns = 'movieId')

print(ratings_matrix.shape)
ratings_matrix

# title 컬럼을 얻기위해 movies 와 join
rating_movies = pd.merge(ratings, movies, on = 'movieId')
rating_movies

# columns = 'title'로 피벗
ratings_matrix = rating_movies.pivot_table('rating', index = 'userId', columns = 'title')
ratings_matrix

# Nan 값을 모두 0으로 변환
ratings_matrix = ratings_matrix.fillna(0)
ratings_matrix


### 영화와 영화들 간 유사도 산출

# 아이템 - 사용자 간의 행렬로 변환
ratings_matrix_T = ratings_matrix.T
ratings_matrix_T.head(3)

# 영화들 간 코사인 유사도 산출
from sklearn.metrics.pairwise import cosine_similarity

item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)

# cos similarity로 변환된 행렬을 Dataframe으로
item_sim_df = pd.DataFrame(data = item_sim, index = ratings_matrix.columns,
                           columns=ratings_matrix.columns)

# Godfather와 유사한 영화 6개 확인
item_sim_df['Godfather, The (1972)'].sort_values(ascending=False)[:6]

# 스스로 제외하고 인셉션과 유사한 영화 5개 확인
item_sim_df['Inception (2010)'].sort_values(ascending = False)[1:6]

## 아이템 기반 인접이웃 협업 필터링으로 개인화된 영화 추천

# 평점 벡터(행 벡터)와 유사도 벡터를 행렬곱해서 평점을 계산하는 함수
def predict_rating(ratings_arr, item_sim_arr):
    ratings_pred = ratings_arr.dot(item_sim_arr) / np.array([np.abs(item_sim_arr).sum(axis = 1)])
    return ratings_pred

item_sim_df.head(3)

ratings_pred = predict_rating(ratings_matrix.values, item_sim_df.values)
ratings_pred

# 데이터프레임으로 변환
ratings_pred_matrix = pd.DataFrame(data = ratings_pred, index = ratings_matrix.index,
                                   columns = ratings_matrix.columns)

print(ratings_pred_matrix.shape)
ratings_pred_matrix.head(10)


## 예측 정확도를 판단하기 위해 오차함수인 RMSE 활용
from sklearn.metrics import mean_squared_error

# 사용자가 평점을 부여한 영화에 대해서만 MSE 구하기.
def get_mse(pred, actual):
    # ignore nonzero terms. 
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

print('아이템 기반 모든 인접 이웃 MSE: ', get_mse(ratings_pred, ratings_matrix.values))


## top-n 유사도를 가진 데이터에 대해서만 예측 평점 계산
def predict_rating_topsim(ratings_arr, item_sim_arr, n = 20):
    # 사용자 - 아이템 평점 행렬 크기만큼 0으로 채운 행렬 초기화
    pred = np.zeros(ratings_arr.shape)

    # 사용자-아이템 평점 행렬의 열 크기만큼 Loop 수행
    for col in range(ratings_arr.shape[1]):
        # 유사도 행렬에서 유사도가 큰 순으로 n개의 데이터 행렬의 index 변환하기
        top_n_items = [np.argsort(item_sim_arr[:, col])[:-n-1:-1]]
        # 개인화된 예측 평점을 계산
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, :][top_n_items].dot(ratings_arr[row, :][top_n_items].T)
            pred[row, col] /= np.sum(np.abs(item_sim_arr[col, :][top_n_items]))
    return pred

# 오래걸림
ratings_pred = predict_rating_topsim(ratings_matrix.values, item_sim_df.values, n=20)
print('아이템 기반 인접 top20 이웃 MSE:', get_mse(ratings_pred, ratings_matrix.values))

# 계산된 예측 평점 데이터는 DataFrame으로 재생성
ratings_pred_matrix = pd.DataFrame(data = ratings_pred, index = ratings_matrix.index,
                                   columns = ratings_matrix.columns)

ratings_pred_matrix

### 사용자에게 영화 추천을 해보기
user_rating_id = ratings_matrix.loc[9, :]
user_rating_id[user_rating_id>0].sort_values(ascending=False)[:10]

# 사용자가 관람하지 않은 영화 중에서 추천
def get_unseen_movies(ratings_matrix, userId):
    # userID로 받은 사용자의 모든 영화정보를 추출해서 Series로 반환
    # 반환된 user_rating은 영화명을 index로 가지는 Series임.
    user_rating = ratings_matrix.loc[userId, :]

    # user rating이 0보다 크면 기존에 관람한 영화임. 대상 index를 추출하여 list 객체로 만들기
    already_seen = user_rating[user_rating>0].index.tolist()

    # 모든 영화명을 list로 만듦.
    movies_list = ratings_matrix.columns.tolist()

    # list comprehension 으로 already_see에 해당하는 movie는 movie_list에서 제외함.
    unseen_list = [movie for movie in movies_list if movie not in already_seen]

    return unseen_list

# pred_df : 앞서 계산된 영화별 예측 평점
# unseen_list : 사용자가 보지 않은 영화들
# top_n : 상위 n개 영화를 가져오기

def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    # 가장 평점 DataFrame에서 사용자 id index와 unseen_list로 들어온 영화명 column 추출하여
    # 가장 평점이 높은 순으로 정렬
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending = False)[:top_n]
    return recomm_movies

# 사용자가 관람하지 않은 영화명 추출
unseen_list = get_unseen_movies(ratings_matrix, 9)

# 아이템 기반의 인접 이웃 협업 필터링으로 영화 추천
recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)
recomm_movies