import pandas as pd
import nltk
import re
import unidecode
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from newspaper import Article
from nltk.corpus import stopwords

# Função para limpar o texto 
def limpar_texto(texto):
    texto = texto.lower()                              # letras minúsculas
    texto = unidecode.unidecode(texto)                 # remove acentos
    texto = re.sub(r'\d+', '', texto)                  # remove números
    texto = re.sub(r'[^\w\s]', '', texto)              # remove pontuação
    texto = re.sub(r'\s+', ' ', texto).strip()         # remove espaços extras
    return texto

# 1. Carregar os dados
df = pd.read_csv("preprocessed.csv")

# Aplicar a limpeza nos textos
df['content'] = df['preprocessed_news'].apply(limpar_texto)

# Baixar stopwords
nltk.download('stopwords')
stopwords_pt = stopwords.words('portuguese')

# Dividir os dados
x_train, x_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)

# Vetorização TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopwords_pt, max_df=0.7)
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

# Treinar o classificador
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x_train_tfidf, y_train)

# Avaliar o modelo
y_pred = model.predict(x_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"\n🔎 Precisão do modelo: {round(acc * 100, 2)}%")
print("📊 Matriz")
print(confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"]))

# Classificar notícia a partir de um link
def classificar_link(url):
    try:
        article = Article(url, language='pt')
        article.download()
        article.parse()
        texto_original = article.text

        if not texto_original.strip():
            print("❌ Não foi possível extrair texto do link.")
            return

        texto_limpo = limpar_texto(texto_original)
        vetor = vectorizer.transform([texto_limpo])
        pred = model.predict(vetor)[0]
        print("\n🔗 Classificação do link:", "🟥 FAKE" if pred == "FAKE" else "🟩 REAL")

    except Exception as e:
        print("⚠️ Erro ao processar o link:", e)

# 7. Interação com o usuário
if __name__ == "__main__":
    print("\n📢 Verifique se a notícia é verdadeira ou falsa")
    link = input("📥 Cole o link da notícia: ")
    classificar_link(link)
