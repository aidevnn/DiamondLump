﻿using System;
using System.Diagnostics;
using System.Linq;
using DiamondLump.Layers;
using DiamondLump.Losses;
using DiamondLump.Optimizers;
using NDarrayLib;

namespace DiamondLump
{
    class MainClass
    {

        static void TestXor<U>(bool summary = false, int epochs = 50, int displayEpochs = 25)
        {
            Console.WriteLine("Hello World, MLP on Xor Dataset.");

            double[,] X0 = { { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 } };
            double[,] y0 = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };
            double[][] X = Enumerable.Range(0, 4).Select(i => Enumerable.Range(0, 2).Select(j => X0[i, j]).ToArray()).ToArray();
            double[][] y = Enumerable.Range(0, 4).Select(i => Enumerable.Range(0, 1).Select(j => y0[i, j]).ToArray()).ToArray();
            var ndX = new NDarray<double>(X).Cast<U>();
            var ndY = new NDarray<double>(y).Cast<U>();

            var net = new Network<U>(new SGD<U>(0.01f), new MeanSquaredLoss<U>(), new RoundAccuracy<U>());
            net.AddLayer(new DenseLayer<U>(8, inputShape: 2));
            net.AddLayer(new TanhLayer<U>());
            net.AddLayer(new DenseLayer<U>(32));
            net.AddLayer(new TanhLayer<U>());
            net.AddLayer(new DenseLayer<U>(1));
            net.AddLayer(new SigmoidLayer<U>());

            if (summary)
                net.Summary();

            net.Fit(ndX, ndY, epochs, displayEpochs: displayEpochs);

            if (summary)
            {
                var yp = net.Forward(ndX);
                for (int k = 0; k < 4; ++k)
                    Console.WriteLine($"[{X[k].Glue()}] = [{y[k][0]}] -> {yp.Data[k]:0.000000}");
            }

            Console.WriteLine();
        }

        static void TestIris<U>(bool summary = false, int epochs = 50, int displayEpochs = 25, int batchsize = 10)
        {
            Console.WriteLine("Hello World, MLP on Iris Dataset.");

            (var trainX, var trainY, var testX, var testY) = ImportDataset.IrisDataset<U>(ratio: 0.8);
            var net = new Network<U>(new SGD<U>(0.025f, 0.2f), new MeanSquaredLoss<U>(), new ArgmaxAccuracy<U>());
            net.AddLayer(new DenseLayer<U>(5, inputShape: 4));
            net.AddLayer(new TanhLayer<U>());
            net.AddLayer(new DenseLayer<U>(3));
            net.AddLayer(new SigmoidLayer<U>());

            if (summary)
                net.Summary();

            net.Fit(trainX, trainY, epochs, batchSize: batchsize, displayEpochs: displayEpochs);
            net.Test(testX, testY);

            Console.WriteLine();
        }

        static void TestDigits<U>(bool summary = false, int epochs = 50, int displayEpochs = 25, int batchsize = 100)
        {
            Console.WriteLine("Hello World, MLP on Digits Dataset.");

            (var trainX, var trainY, var testX, var testY) = ImportDataset.DigitsDataset<U>(ratio: 0.9);
            var net = new Network<U>(new SGD<U>(0.025f, 0.2f), new CrossEntropyLoss<U>(), new ArgmaxAccuracy<U>());
            net.AddLayer(new DenseLayer<U>(32, inputShape: 64));
            net.AddLayer(new SigmoidLayer<U>());
            net.AddLayer(new DenseLayer<U>(10));
            net.AddLayer(new SoftmaxLayer<U>());

            if (summary)
                net.Summary();

            net.Fit(trainX, trainY, testX, testY, epochs: epochs, batchSize: batchsize, displayEpochs: displayEpochs);
            //net.Test(testX, testY);

            Console.WriteLine();
        }

        public static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            Utils.Backend = Backend.Mkl;
            for(int k = 0; k < 5; ++k)
            {
                TestDigits<double>(false, 50, 10, 100);
                TestDigits<float>(false, 50, 10, 100);
            }
        }
    }
}
