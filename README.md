
# 📰 Fake News Detector em Português

Este projeto implementa um detector de fake news em português utilizando aprendizado de máquina com o classificador **Passive Aggressive**. O modelo é treinado a partir de um dataset pré-processado e permite verificar se uma notícia obtida via link é **FAKE** ou **REAL**.

> ⚠️ **Atenção**: Este projeto é experimental e o modelo **ainda não está 100% preciso**. Ele foi treinado com uma quantidade limitada de notícias e pode apresentar erros na classificação. Para melhorar a performance, é necessário utilizar um volume maior e mais diverso de dados reais e falsos.

## 📁 Estrutura do Projeto

- `news_fake_detector.py`: Script principal que realiza todo o pipeline de:
  - Leitura dos dados
  - Limpeza e pré-processamento de texto
  - Vetorização TF-IDF
  - Treinamento do modelo
  - Avaliação da acurácia
  - Classificação de notícias a partir de links
- `preprocessed.csv`: Arquivo contendo as notícias já pré-processadas, com colunas:
  - `preprocessed_news`: Conteúdo textual da notícia
  - `label`: Rótulo da notícia (`FAKE` ou `REAL`)

## 🚀 Como usar

1. **Instale as dependências**:
   ```bash
   pip install pandas scikit-learn nltk newspaper3k unidecode
   ```

2. **Execute o script**:
   ```bash
   python news_fake_detector.py
   ```

3. **Cole o link da notícia** quando solicitado.

## 📊 Exemplo de Saída

```bash
🔎 Precisão do modelo: 93.42%
📊 Matriz
[[162  18]
 [ 10 210]]

📢 Verifique se a notícia é verdadeira ou falsa
📥 Cole o link da notícia: https://exemplo.com/noticia
🔗 Classificação do link: 🟩 REAL
```

## 🛠 Tecnologias Utilizadas

- Python
- Scikit-learn
- NLTK
- Newspaper3k
- TF-IDF Vectorizer
- Classificador Passive Aggressive

## 🧠 Observações

- O script limpa e normaliza o texto, removendo acentos, pontuações, números e stopwords.
- O modelo foi treinado com dados em português e, portanto, é mais adequado para conteúdos nessa língua.
- O Newspaper3k é utilizado para extrair automaticamente o texto de links de notícias.
- Este projeto pode ser estendido com:
  - Mais dados de treinamento
  - Testes com diferentes algoritmos
  - Interface gráfica ou web

## 📌 Requisitos

- Python 3.8+
- Acesso à internet para baixar e processar artigos via link

## 📄 Licença

Este projeto é open-source e distribuído sob a licença MIT.
