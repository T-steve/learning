# 可以通用
```py
print(123)
```

```
class TfIdfHandler(BaseEstimator, TransformerMixin):
    """
    cut text using nlp api
    """

    def __init__(self, uuid):
        self.uuid = uuid

    def fit(self, data_origin):
        """
        :param data_origin:
        :return:
        """
        tfidf_train_features = tfidf_extractor(
            self.uuid, data_origin,
            ngram_range=(1, 1), max_features=None)
        logger.info('cut text to words successfully')
        return tfidf_train_features

    def transform(self, data_origin):
        """
        :param data_origin:
        :param train_predict:
        :return:
        """
        tfidf_train_features = tfidf_transform(self.uuid, data_origin)
        logger.info('cut text to words successfully')
        return tfidf_train_features
```
