using Microsoft.ML;
using Microsoft.ML.Data;
using System;

class Program
{
    static void Main(string[] args)
    {
        // Create a machine learning context
        var context = new MLContext();

        // Define the data schema
        class DataPoint
        {
            [LoadColumn(0)]
            public float X { get; set; }

            [LoadColumn(1)]
            public float Y { get; set; }

            [LoadColumn(2)]
            public bool Label { get; set; }
        }

        // Load the data
        var data = context.Data.LoadFromTextFile<DataPoint>("data.csv", separatorChar: ',');

        // Define the pipeline
        var pipeline = context.Transforms.Concatenate("Features", "X", "Y")
            .Append(context.Transforms.NormalizeMinMax("Features"))
            .Append(context.Transforms.Conversion.MapValueToKey("Label"))
            .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"))
            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

        // Train the model
        var model = pipeline.Fit(data);

        // Create a prediction engine
        var engine = context.Model.CreatePredictionEngine<DataPoint, Prediction>(model);

        // Use the prediction engine to make predictions
        var prediction = engine.Predict(new DataPoint { X = 5, Y = 7 });

        Console.WriteLine($"Predicted label: {prediction.PredictedLabel}");
    }
}

class Prediction
{
    public bool PredictedLabel { get; set; }
}
