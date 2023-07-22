import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from konlpy.tag import Okt
import urllib.parse

okt = Okt()
vectorizer = TfidfVectorizer()
label_encoder = LabelEncoder()

csv_file_path = "https://github.com/Bell-Min/streamlit_web_deploy/blob/main/fakenews_datasets.csv"
df = pd.read_csv(csv_file_path)
corpus = df["text"].tolist()
labels = df["label"].tolist()
encoded_labels = label_encoder.fit_transform(labels)
tokenized_corpus = [" ".join(okt.morphs(sentence)) for sentence in corpus]
X = vectorizer.fit_transform(tokenized_corpus)

nb_classifier = MultinomialNB()
nb_classifier.fit(X, encoded_labels)

def predict_fake_news(event, context):
    title = event.get("queryStringParameters", {}).get("title", "")
    title_morphs = " ".join(okt.morphs(title))
    title_vector = vectorizer.transform([title_morphs])
    fake_prob = nb_classifier.predict_proba(title_vector)[0][0]

    if fake_prob >= 0.8:
        result = "판단 결과, 해당 뉴스기사는 전혀 사실이 아닐 가능성이 높습니다."
    elif fake_prob >= 0.6:
        result = "판단 결과, 해당 뉴스기사는 대체로 사실이 아닐 가능성이 높습니다."
    elif fake_prob >= 0.4:
        result = "판단 결과, 해당 뉴스기사는 절반 정도 사실일 가능성이 높습니다."
    elif fake_prob >= 0.2:
        result = "판단 결과, 해당 뉴스기사는 대체로 사실이 아닐 가능성이 높습니다."
    else:
        result = "판단 결과, 해당 뉴스기사는 거의 사실일 가능성이 높습니다."

    search_result_url = get_search_results(title)

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": '{"result": "' + result + '", "fake_prob": ' + str(fake_prob) + ', "search_result_url": "' + search_result_url + '"}'
    }

def get_search_results(title):
    query = f"https://www.google.com/search?q={urllib.parse.quote(title)}"
    return query
