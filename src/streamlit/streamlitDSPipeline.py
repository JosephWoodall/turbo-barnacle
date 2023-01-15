import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pandas 

class DataSciencePipeline:
    def __init__(self):
        pass

    def obtain_data(self):
        data_path = st.file_uploader("Upload your data file", type=["csv", "txt", "xlsx"])
        if data_path is not None:
            self.data = pandas.read_csv(data_path)
        else:
            st.warning("Please upload a valid file")

    def clean_data(self):
        self.data.dropna(inplace=True)
        st.write("Data Shape after cleaning:", self.data.shape)

    def visualize_data(self):
        st.bar_chart(self.data.groupby("label")["value"].count())

    def model_data(self):
        features = st.multiselect("Select features", self.data.columns, default=self.data.columns[:3])
        label = st.selectbox("Select label", self.data.columns)
        self.model = RandomForestClassifier()
        self.model.fit(self.data[features], self.data[label])

    def interpret_data(self):
        st.write("Model Accuracy:", self.model.score(self.data[features], self.data[label]))

    def revise_model(self):
        if st.button("Retrain Model"):
            self.model_data()
