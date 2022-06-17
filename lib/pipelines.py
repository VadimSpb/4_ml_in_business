from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from .transformers import *
from .reports import PrecisionReport

class CVDPipeline:
    """
    Transformer for people's data for ML
    """
    def __init__(self, params):
        self.cont_cols  = params.get('continuos_cols')
        self.cat_cols = params.get('cat_cols')
        self.base_cols = params.get('base_cols')
        self.cont_transformers = []
        self.cat_transformers = []
        self.base_transformers = []
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.prediction = None
        self.report = None


    def fit(self,
            x_train,
            y_train,
            model_type=None,
            random_state=42):
        for cont_col in self.cont_cols:
            transfomer =  Pipeline([
                                    ('selector', NumberSelector(key=cont_col)),
                                    ('standard', StandardScaler())
                        ])
            self.cont_transformers.append((cont_col, transfomer))

        for cat_col in self.cat_cols:
            cat_transformer = Pipeline([
                                        ('selector', ColumnSelector(key=cat_col)),
                                        ('ohe', OHEEncoder(key=cat_col))
                                    ])
            self.cat_transformers.append((cat_col, cat_transformer))

        for base_col in self.base_cols:
            base_transformer = Pipeline([
                                        ('selector', NumberSelector(key=base_col))
                                    ])
            self.base_transformers.append((base_col, base_transformer))

        feats = FeatureUnion(self.cont_transformers +
                             self.cat_transformers +
                             self.base_transformers)

        if model_type == "gradient boosting":
            model = Pipeline([
                                    ('features', feats),
                                    ('classifier', GradientBoostingClassifier(random_state=random_state))
                                ])
        if model_type == "random forest":
            model = Pipeline([
                                    ('features', feats),
                                    ('classifier', RandomForestClassifier(random_state=random_state))
                                ])
        else:
            model = Pipeline([
                                    ('features', feats),
                                    ('classifier', LogisticRegression(random_state=random_state))
                                ])

        self.model = model.fit(x_train, y_train)

    def predict(self, x_test, y_test):
        self.prediction = self.model.predict_proba(x_test)[:, 1]
        self.estimator = PrecisionReport(y_test, self.prediction)
        self.report = self.estimator.report(model_name=type(self.model['classifier']).__name__)


class CharnPipeline(CVDPipeline):

    def __init__(self, params):
        super().__init__()

















