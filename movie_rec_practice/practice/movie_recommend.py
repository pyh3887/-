import numpy as np 
import pandas as pd
import json


meta = pd.read_csv('../dataset/movies_metadata.csv',dtype='unicode')

# meta 데이터의 일부 컬럼만 사용 
meta = meta[['id', 'original_title', 'original_language', 'genres']]

# 후의 ratings와의 merge 를 위해 id 컬럼의 이름을 rename 함 
meta = meta.rename(columns={'id':'movieId'})
meta = meta[meta['original_language'] == 'en']
meta.head()


# ratings(평가자들의 평점)  
ratings = pd.read_csv('../dataset/ratings_small.csv')
ratings = ratings[['userId', 'movieId', 'rating']]
ratings.head()

# meta , ratings 의 movieid 를 숫자형으로 변환
meta.movieId = pd.to_numeric(meta.movieId, errors='coerce')
ratings.movieId = pd.to_numeric(ratings.movieId, errors='coerce')

# json 형 데이터를 변환하여 genres 컬럼에 재정의  apply 사용 
def parse_genres(genres_str):
    genres = json.loads(genres_str.replace('\'', '"'))
    
    genres_list = []
    for g in genres:
        genres_list.append(g['name'])

    return genres_list

meta['genres'] = meta['genres'].apply(parse_genres)

meta.head()

# ratings 데이터프레임과 meta 데이터프레임을 movieId로 inner join 함
data = pd.merge(ratings, meta, on='movieId', how='inner')

data.head()


# 피봇 테이블을 이용해 영화별 user들의 평점을 요약화함
matrix = data.pivot_table(index='userId', columns='original_title', values='rating')

print(matrix.head(20))



GENRE_WEIGHT = 0.1

# 각 영화관의 상관계수를 구하기 위함 > rating
def pearsonR(s1, s2):
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    return np.sum(s1_c * s2_c) / np.sqrt(np.sum(s1_c ** 2) * np.sum(s2_c ** 2))

print('영화 리스트',meta['original_title'])


def recommend(input_movie, matrix, n, similar_genre=True):
    
    # 입력된 영화의 장르를 추출한다
    input_genres = meta[meta['original_title'] == input_movie]['genres'].iloc(0)[0]

    result = []
    
    # 현재 matrix 의 모든 영화
    for title in matrix.columns:
        # 입력영화와 같은영화일경우 건너뜀
        if title == input_movie:
            continue

        # rating comparison
        # 입력영화와 비교될영화의 상관계수 구하기
        cor = pearsonR(matrix[input_movie], matrix[title])
        
        # genre comparison
        # 장르가 겹친다면 가중치를 곱한다
        if similar_genre and len(input_genres) > 0:
            temp_genres = meta[meta['original_title'] == title]['genres'].iloc(0)[0]

            same_count = np.sum(np.in1d(input_genres, temp_genres))
            cor += (GENRE_WEIGHT * same_count)
        
        if np.isnan(cor):
            continue
        else:
            result.append((title, '{:.2f}'.format(cor), temp_genres))
    # result 의 상관계수가 오름차순 정렬        
    result.sort(key=lambda r: r[1], reverse=True)
    
    return result[:n]


recommend_result = recommend('The Dark Knight', matrix, 10, similar_genre=True)

print(pd.DataFrame(recommend_result, columns = ['Title', 'Correlation', 'Genre']))