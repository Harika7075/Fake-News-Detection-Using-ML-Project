import pickle

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def predict_news(text):
    text_vec = vectorizer.transform([text])
    result = model.predict(text_vec)
    return result[0]

while True:
    news = input("Enter news: ")

    if news.lower() == "exit":
        break

    print("Prediction:", predict_news(news))