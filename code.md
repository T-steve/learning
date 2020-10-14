# hello world

```
class TfIdfHandler(BaseEstimator, TransformerMixin):
    def __init__(self, uuid):
        self.uuid = uuid
    def fit(self, data_origin):
        tfidf_train_features = tfidf_extractor(
            self.uuid, data_origin,
            ngram_range=(1, 1), max_features=None)
        logger.info('cut text to words successfully')
        return tfidf_train_features
    def transform(self, data_origin):
        tfidf_train_features = tfidf_transform(self.uuid, data_origin)
        logger.info('cut text to words successfully')
        return tfidf_train_features    
def save_to_pickle(save_dir, save_data):
    with open(save_dir, 'wb') as f:
        joblib.dump(save_data, f)
def read_from_pickle(pickle_file):
    try:
        with open(os.path.realpath(pickle_file), 'rb') as f:
            pickle_data = joblib.load(f)
        return pickle_data
    except Exception as e:
        logger.info("wrong path:{}".format(e))
def pipeline_time(col_time):
    time_pipe = Pipeline(steps=[
        ("select", DataFrameSelector(col_time)),
        ("feature_handler", TimeFeatureHandler()),
        ("min_max_scalar", MinMaxScaler())
    ], memory=None) if col_time else None
    return time_pipe
def pipeline_all_tf(col_num, col_cat, col_str, col_time, uuid):

    all_pipe = FeatureUnion(transformer_list=[
        ("num_pipe", pipeline_num(col_num)),
        ("cat_pipe", pipeline_cat(col_cat)),
        ("str_pipe", pipeline_str_tfidf(col_str, uuid)),
        ("time_pipe", pipeline_time(col_time))
    ])
    return all_pipe
```
