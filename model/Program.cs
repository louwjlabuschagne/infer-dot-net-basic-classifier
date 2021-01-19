using System;
using Range = Microsoft.ML.Probabilistic.Models.Range;
using Microsoft.ML.Probabilistic.Models;
using System.Globalization;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Compiler.Visualizers;
using Algorithms =  Microsoft.ML.Probabilistic.Algorithms;
using System.IO;
using System.Text;
namespace model
{
    class Program
    {
        static void Main(string[] args)
        {

            string dataDir = args[0];
            string datasetFilename = dataDir+args[1];
            string[] lines = File.ReadAllLines(datasetFilename);
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
            flower.AddAttribute(new Sequential());

            // Variables

            // The feature - x
            VariableArray<double> featureValues = Variable.Array<double>(flower).Named("featureValue").Attrib(new DoNotInfer());
            // The label - y
            VariableArray<bool> isSetosa = Variable.Array<bool>(flower).Named("isSetosa");

            // The weight - w
            Variable<double> weight = Variable.GaussianFromMeanAndVariance(0,1).Named("weight");     
            // The threshold
            Variable<double> threshold = Variable.GaussianFromMeanAndVariance(0,10).Named("threshold");

            // Loop over flowers
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
            /*******************************/


            Console.WriteLine(postWeight);
            Console.WriteLine(postThreshold);

                     // write outputs to file
            // var storeSites = new StringBuilder();
            var results = new StringBuilder();

            results.AppendLine("variable; mean; variance");
            var line = string.Format("postWeight;{0};{1}", postWeight.GetMean(), postWeight.GetVariance());
            results.AppendLine(line.Replace(',', '.'));
            line = string.Format("postThreshold;{0};{1}", postThreshold.GetMean(), postThreshold.GetVariance());
            results.AppendLine(line.Replace(',', '.'));


            File.WriteAllText(dataDir+"results.csv", results.ToString());
            
        }
    }
}
