import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from konlpy.tag import Okt
import urllib.parse
# CSV 파일 경로 설정
csv_file_path = "https://github.com/Bell-Min/streamlit_web_deploy/blob/main/fakenews_datasets.csv"

# CSV 파일 로드 (가상의 컬럼 이름은 "label"과 "text"로 가정합니다.)
df = pd.read_csv(csv_file_path, encoding='utf-8')

# 데이터셋에서 텍스트 데이터와 레이블 추출
corpus = df["text"].tolist()
labels = df["label"].tolist()
# 문자 레이블을 숫자로 매핑하는 LabelEncoder 사용
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
# KoNLPy를 사용하여 형태소 분석
okt = Okt()
tokenized_corpus = [" ".join(okt.morphs(sentence)) for sentence in corpus]
# 텍스트 데이터를 TF-IDF 벡터로 변환
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tokenized_corpus)
# 데이터셋을 학습용과 테스트용으로 나눔
X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)
# MultinomialNB 모델 생성 및 학습
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
# 새로운 기사 제목의 진위 여부와 확률을 판단하는 함수
def predict_fake_news_with_prob(title):
    # 입력한 기사 제목을 형태소 분석하여 TF-IDF 벡터로 변환
    title_morphs = " ".join(okt.morphs(title))
    title_vector = vectorizer.transform([title_morphs])
    
    # MultinomialNB 모델을 사용하여 예측 수행
    probabilities = nb_classifier.predict_proba(title_vector)
    
    # 가짜 뉴스일 확률과 진짜 뉴스일 확률을 반환
    fake_prob = probabilities[0][0]
    
    return fake_prob
# 사용자가 입력한 기사 제목을 검색한 결과 페이지를 가져오는 함수
def get_search_results(title):
    query = f"https://www.google.com/search?q={urllib.parse.quote(title)}"
    return query
    
# Streamlit 애플리케이션
def main():
    st.title('뉴스기사 진위 여부 판단')
    new_article_title = st.text_input('학습된 인공지능 모델이 뉴스기사의 진위 여부를 판단해줍니다. 판단을 원하는 기사 제목을 입력하세요.')
    if st.button('판단하기'):
        fake_prob = predict_fake_news_with_prob(new_article_title)
        if fake_prob >= 0.8:
            st.write("판단 결과, 해당 뉴스기사는 전혀 사실이 아닐 가능성이 높습니다.", f"(가짜뉴스일 확률: {fake_prob*100:.2f}%)")
        elif fake_prob >= 0.6:
            st.write("판단 결과, 해당 뉴스기사는 대체로 사실이 아닐 가능성이 높습니다.", f"(가짜뉴스일 확률: {fake_prob*100:.2f}%)")
        elif fake_prob >= 0.4:
            st.write("판단 결과, 해당 뉴스기사는 절반 정도 사실일 가능성이 높습니다.", f"(가짜뉴스일 확률: {fake_prob*100:.2f}%)")
        elif fake_prob >= 0.2:
            st.write("판단 결과, 해당 뉴스기사는 대체로 사실일 가능성이 높습니다.", f"(가짜뉴스일 확률: {fake_prob*100:.2f}%)")
        else:
            st.write("판단 결과, 해당 뉴스기사는 거의 사실일 가능성이 높습니다.", f"(가짜뉴스일 확률: {fake_prob*100:.2f}%)")
        # 검색 결과 주소를 출력
        search_result_url = get_search_results(new_article_title)
        st.markdown(f"그러나, 아무리 잘 학습된 인공지능일지라도 완벽한 판단을 내릴 수는 없습니다. [다음의 링크]({search_result_url})로 이동해 해당 뉴스기사의 진위 여부를 정확히 알아보세요!")
if __name__ == '__main__':
    main()
#python -m streamlit run fake.py
