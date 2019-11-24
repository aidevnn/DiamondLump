using System;
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

            var (trainX, trainY) = ImportDataset.XorDataset<U>();
            var net = new Network<U>(new SGD<U>(0.1f), new MeanSquaredLoss<U>(), new RoundAccuracy<U>());
            net.AddLayer(new DenseLayer<U>(8, inputShape: 2));
            net.AddLayer(new TanhLayer<U>());
            net.AddLayer(new DenseLayer<U>(1));
            net.AddLayer(new SigmoidLayer<U>());

            if (summary)
                net.Summary();

            net.Fit(trainX, trainY, epochs, displayEpochs: displayEpochs);

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

            Console.WriteLine();
        }

        public static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            Utils.Backend = Backend.Mkl;
            TestDigits<float>(true, 50, 10);
        }
    }
}
