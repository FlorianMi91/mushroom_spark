from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn2pmml import PMMLPipeline, sklearn2pmml
import pandas as pd# fetching data example
df = load_diabetes()
X = pd.DataFrame(columns = df.feature_names, data = df.get('data'))
y = pd.DataFrame(columns = ['target'], data = df.get('target'))# here you can use the key classifier, if suitable
pipeline = PMMLPipeline([ ('regressor', DecisionTreeRegressor()) ])#training the model
pipeline.fit(X, y)# exporting the model
sklearn2pmml(pipeline, 'model.pmml', with_repr = True)