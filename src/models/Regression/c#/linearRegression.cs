using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

class Program
{
    static void Main(string[] args)
    {
        // Create a new MLContext
        var context = new MLContext();

        // Load the data from a text file
        var data = context.Data.LoadFromTextFile<HousingData>("housing.csv", separatorChar: ',');

        // Split the data into a training set and a test set
        var (trainData, testData) = context.Data.TrainTestSplit(data, testFraction: 0.2);

        // Define the pipeline
        var pipeline = context.Transforms.Concatenate("Features", "Rooms", "SquareFeet", "Distance")
            .Append(context.Transforms.NormalizeMinMax("Features"))
            .Append(context.Transforms.Concatenate("Label", "Price"))
            .Append(context.Regression.Trainers.Sdca());

        // Train the model
        var model = pipeline.Fit(trainData);

        // Evaluate the model on the test set
        var predictions = model.Transform(testData);
        var metrics = context.Regression.Evaluate(predictions);

        // Print the evaluation metrics
        Console.WriteLine($"R^2: {metrics.RSquared}");
        Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
    }
}

class HousingData
{
    [LoadColumn(0)] public float Rooms;
    [LoadColumn(1)] public float SquareFeet;
    [LoadColumn(2)] public float Distance;
    [LoadColumn(3)] public float Price;
}
