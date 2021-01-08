using System;
using Range = Microsoft.ML.Probabilistic.Models.Range;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Compiler.Visualizers;
using Algorithms =  Microsoft.ML.Probabilistic.Algorithms;
using System.IO;
namespace model
{
    class Program
    {
        static void Main(string[] args)
        {

            
            // the nps.csv file contains all the customer nps scores.
            // responseFilename = "responses-generate.csv";
            string[] lines = File.ReadAllLines("..\\data\\iris-one-feature.csv");
            bool[] isSetosaLabel = new bool[lines.Length];
            double[] featureVal = new double[lines.Length];

            for (int i = 0; i < lines.Length; i++)
            {
                string[] strArray = lines[i].Split('|');
                isSetosaLabel[i] = strArray[1] == "1";
                featureVal[i] = float.Parse(strArray[0].Replace(".", ","));
            }

            // Creating the model

            int numberOfFlowers = lines.Length;
            Range flower = new Range(numberOfFlowers).Named("flower");

            // Make sure that the range across flowers is handled sequentially
            // - this is necessary to ensure the model converges during training
            flower.AddAttribute(new Sequential());

            // observed data
            VariableArray<double> featureValues = Variable.Array<double>(flower).Named("featureValue").Attrib(new DoNotInfer());
            // FeatureValue = Variable.Array<double>(flower).Named("featureValue").Attrib(new DoNotInfer());

            // New creates distribution not RV
            // The weight
            Variable<double> weight = Variable.GaussianFromMeanAndVariance(0,1).Named("weight");     
            // The threshold
            Variable<double> threshold = Variable.GaussianFromMeanAndVariance(0,10).Named("threshold");
            // Label: is the message replied to?
            VariableArray<bool> isSetosa = Variable.Array<bool>(flower).Named("isSetosa");
            // Loop over emails
            using (Variable.ForEach(flower))
            {
                var score = (featureValues[flower] * weight).Named("score");

                var noisyScore = Variable.GaussianFromMeanAndVariance(score, 10).Named("noisyScore");
                isSetosa[flower] = noisyScore > threshold;
            }

            /********* observations *********/
            isSetosa.ObservedValue = isSetosaLabel;
            featureValues.ObservedValue = featureVal;
            /*******************************/

            /********** inference **********/
            var InferenceEngine = new InferenceEngine(new ExpectationPropagation());
            // var InferenceEngine = new InferenceEngine(new VariationalMessagePassing());
            InferenceEngine.NumberOfIterations = 50;
            // InferenceEngine.ShowFactorGraph = true;

            Gaussian postWeight = InferenceEngine.Infer<Gaussian>(weight);
            Gaussian postThreshold = InferenceEngine.Infer<Gaussian>(threshold);

            Console.WriteLine(postWeight);
            Console.WriteLine(postThreshold);
            ///*******************************/
        }
    }
}
