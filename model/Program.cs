using System;
using Range = Microsoft.ML.Probabilistic.Models.Range;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
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
            bool[] isSetosa = new bool[lines.Length];
            double[] featureVal = new double[lines.Length];

            for (int i = 0; i < lines.Length; i++)
            {
                string[] strArray = lines[i].Split('|');
                isSetosa[i] = strArray[1] == "1";
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

            // The weight
            WeightPrior = Variable.New<Gaussian>().Named("WeightPrior").Attrib(new DoNotInfer());
            Weight = Variable.New<double>().Named("weight");
            Weight.SetTo(Variable<double>.Random(WeightPrior));

            // The threshold
            ThresholdPrior = Variable.New<Gaussian>().Named("ThresholdPrior").Attrib(new DoNotInfer());
            Threshold = Variable.New<double>().Named("threshold");
            Threshold.SetTo(Variable<double>.Random(ThresholdPrior));

            // Noise Variance
            NoiseVariance = Variable.New<double>().Named("NoiseVariance").Attrib(new DoNotInfer());

            // Label: is the message replied to?
            RepliedTo = Variable.Array<bool>(message).Named("repliedTo");

            // Loop over emails
            using (Variable.ForEach(message))
            {
                var score = (FeatureValue[message] * Weight).Named("score");

                var noisyScore = Variable.GaussianFromMeanAndVariance(score, NoiseVariance).Named("noisyScore");
                RepliedTo[message] = noisyScore > Threshold;
            }

            // InitializeEngine();

            // Engine.OptimiseForVariables = Mode == InputMode.Training
            //                                        ? new IVariable[] { Weight, Threshold }
            //                                        : Engine.OptimiseForVariables = new IVariable[] { RepliedTo };

            

            Variable<bool> firstCoin = Variable.Bernoulli(0.5).Named("firstCoin");
            Variable<bool> secondCoin = Variable.Bernoulli(0.5).Named("secondCoin");
            Variable<bool> bothHeads = (firstCoin & secondCoin).Named("bothHeads");
            
            InferenceEngine engine = new InferenceEngine();

            if (engine.Algorithm is Algorithms.VariationalMessagePassing)
            {
                Console.WriteLine("This example does not run with Variational Message Passing");
                return;
            }
            Console.WriteLine("Probability both coins are heads: " + engine.Infer(bothHeads));
            bothHeads.ObservedValue = true; 
            Console.WriteLine("Probability distribution over firstCoin: " + engine.Infer(firstCoin));

            engine.ShowFactorGraph = true;
            // engine.ShowMsl = true;
        }
    }
}
