from fastapi import FastAPI
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.context import SparkContext
from pyspark.sql.functions import concat

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline

from pyspark.ml.classification import RandomForestClassifier



class Mushroom:
    def __init__(self, 
                bruises, 
                odor,
                spore_print_color,
                population,
                habitat,
                cap,
                gill,
                stalk,
                veil,
                ring):
        self.bruises = bruises
        self.odor = odor
        self.spore_print_color = spore_print_color
        self.population = population
        self.habitat = habitat
        self.cap = cap
        self.gill = gill
        self.stalk = stalk
        self.veil = veil
        self.ring = ring


# a = Mushroom("gfdgfd", 'fdsfds')

# print(a.features)

app = FastAPI()


# @app.get("/items/{item_id}")
# async def read_item(item_id: int):
#     return {"item_id": item_id}

# @app.post()

# Create pyspark object
spark=SparkSession.builder.appName('Mushroom').getOrCreate()
df = spark.read.csv('data/mushrooms.csv',header=True,inferSchema=True)


# Create new column Cap by combining shape surface and color

df = df.withColumn("cap", concat(df['cap-shape'],df['cap-surface'],df['cap-color']).alias("cap"))
df = df.drop('cap-shape','cap-color','cap-surface')
df = df.withColumn("gill", concat(df['gill-attachment'],df['gill-spacing'],df['gill-size'],df['gill-color']).alias("cap"))
df = df.drop('gill-attachment','gill-spacing','gill-size','gill-color')
df = df.withColumn("stalk", concat(df['stalk-shape'],df['stalk-root'],df['stalk-surface-above-ring'],
    df['stalk-surface-below-ring'],df['stalk-color-above-ring'],df['stalk-color-below-ring']).alias("stalk"))
df = df.drop('stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',
    'stalk-color-above-ring','stalk-color-below-ring')
df = df.withColumn("veil", concat(df['veil-type'],df['veil-color'].alias("veil")))
df = df.drop('veil-type','veil-color')  
df = df.withColumn("ring", concat(df['ring-number'],df['ring-type'].alias("ring")))
df = df.drop('ring-number','ring-type')

categoricalColumns = [
'bruises',
'odor',
'spore-print-color',
'population',
'habitat',
'cap',
'gill',
'stalk',
'veil',
'ring']

stages = []

for categoricalcol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol=categoricalcol, outputCol=categoricalcol+'_index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalcol+"_class_vec"])
    stages += [stringIndexer, encoder]

label_string_indexer = StringIndexer(inputCol='class', outputCol='label')
stages += [label_string_indexer]

assemblerInputs = [c + '_class_vec' for c in categoricalColumns]
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol='vec_features')
stages += [assembler]
scaler = StandardScaler(inputCol='vec_features',outputCol='features')
stages += [scaler]

cols = df.columns
pipeline = Pipeline(stages=stages)
df_pipe = pipeline.fit(df).transform(df)
selectedCols = ['label','features'] + cols 
df2 = df_pipe.select(selectedCols)

train, test = df2.randomSplit([0.8, 0.2], seed=42)


# Train a RandomForest model.
rf = RandomForestClassifier(labelCol='label', \
                            featuresCol="features", \
                            numTrees=50)
model_rf = rf.fit(train)
pred_rf = model_rf.transform(test)
pred_rf.select('label','features', 'rawPrediction','prediction','probability').toPandas().head()

acc_rf = pred_rf.filter(pred_rf.label == pred_rf.prediction).count() / float(pred_rf.count())
print(f"Accuracy: {acc_rf}")

app = FastAPI()


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

#bruises='f', odor='n', spore-print-color='w', population='v', habitat='l', cap='fyn', gill='fwnw', stalk='ebsfwn', veil='pw', ring='oe
mushroom_test = Mushroom(bruises='f', odor='n', spore_print_color='w', population='v', habitat='l', cap='fyn', gill='fwnw', stalk='ebsfwn', veil='pw', ring='oe')