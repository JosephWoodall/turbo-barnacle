import streamlit 

import sys
import os 
sys.path.insert(0, r'src')
print("WORKIGN DIRECTORY: {}".format(os.getcwd()))

from pyspark import SparkConf
from pyspark.sql import SparkSession

from src.data.GeneratedData import GeneratedData

APPNAME = "turbo-barnacle"
STREAMLIT_TITLE = "Turbo-Barnacle Vizualization"

spark = SparkSession.builder.appName(f"{APPNAME}").getOrCreate()
data = spark.createDataFrame(GeneratedData().buildDictionary(5))

streamlit.title(f'{STREAMLIT_TITLE}')
streamlit.write(data.printSchema())
streamlit.write(data.describe().show())
streamlit.write(data.show(3))


