import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

PATH = r''
SCORE = ''

df = pd.read_csv(PATH)

y = df['target']
x = df.drop('target', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

tpot = TPOTClassifier(scoring=SCORE, verbosity=2)

tpot.fit(x_train, y_train)

print(tpot.score(x_test, y_test))

tpot.export('best_classifier_model.py')
