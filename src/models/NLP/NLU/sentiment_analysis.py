import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


class SentimentAnalysis:
    """ """
    def __init__(self, corpus):
        self.corpus = corpus
        nltk.download('vader_lexicon')
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiments(self):
        """ """
        sentiments = []
        for text in self.corpus:
            scores = self.analyzer.polarity_scores(text)
            if scores['compound'] > 0.05:
                sentiments.append('positive')
            elif scores['compound'] < -0.05:
                sentiments.append('negative')
            else:
                sentiments.append('neutral')
        return sentiments


# Example usage:
corpus = ["I love this product!",
          "This is the worst customer service ever.", "The package arrived on time."]
sa = SentimentAnalysis(corpus)
sentiments = sa.get_sentiments()
print(sentiments)
