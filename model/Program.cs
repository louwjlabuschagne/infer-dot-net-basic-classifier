using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Compiler;
using Algorithms =  Microsoft.ML.Probabilistic.Algorithms;

namespace model
{
    class Program
    {
        static void Main(string[] args)
        {
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
