using System;
using Range = Microsoft.ML.Probabilistic.Models.Range;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Algorithms;
using System.IO;
using System.Text;
namespace model
{
    class Program
    {
        static void Main(string[] args)
        {

            // Reading in the data
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
            int numberOfSample = lines.Length;
            Range n = new Range(numberOfSample).Named("n");

            // Make sure that the range across flowers is handled sequentially
            n.AddAttribute(new Sequential());

            // Variables

            // The feature - x
            VariableArray<double> featureValues = Variable.Array<double>(n).Named("featureValue").Attrib(new DoNotInfer());
            // The label - y
            VariableArray<bool> isSetosa = Variable.Array<bool>(n).Named("isSetosa");

            // The weight - w
            Variable<double> weight = Variable.GaussianFromMeanAndVariance(0,1).Named("weight");     
            // The threshold
            Variable<double> threshold = Variable.GaussianFromMeanAndVariance(0,10).Named("threshold");

            // Loop over ns
            using (Variable.ForEach(n))
            {
                var score = (featureValues[n] * weight).Named("score");

                var noisyScore = Variable.GaussianFromMeanAndVariance(score, 10).Named("noisyScore");
                isSetosa[n] = noisyScore > threshold;
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
            var results = new StringBuilder();

            results.AppendLine("variable;mean;variance");
            var line = string.Format("postWeight;{0};{1}", postWeight.GetMean(), postWeight.GetVariance());
            results.AppendLine(line.Replace(',', '.'));
            line = string.Format("postThreshold;{0};{1}", postThreshold.GetMean(), postThreshold.GetVariance());
            results.AppendLine(line.Replace(',', '.'));

            File.WriteAllText(dataDir+"results.csv", results.ToString());
            
        }
    }
}
