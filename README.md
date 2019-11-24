# DiamondLump

Simple Multi Layers Perceptron

## Xor Dataset

```
var net = new Network<U>(new SGD<U>(0.1f), new MeanSquaredLoss<U>(), new RoundAccuracy<U>());
net.AddLayer(new DenseLayer<U>(8, inputShape: 2));
net.AddLayer(new TanhLayer<U>());
net.AddLayer(new DenseLayer<U>(1));
net.AddLayer(new SigmoidLayer<U>());

net.Summary();
net.Fit(trainX, trainY, 500, 100);
```

### The Output

```
Hello World, MLP on Xor Dataset.
Summary
Network: SGD / MeanSquaredLoss / RoundAccuracy

====================================================
| Layer                | Parameters    | Output    |
====================================================
| InputLayer           |             0 |       (2) |
| DenseLayer           |            24 |       (8) |
| TanhActivation       |             0 |       (8) |
| DenseLayer           |             9 |       (1) |
| SigmoidActivation    |             0 |       (1) |
====================================================

Total Parameters:33

Epoch:    0/500. loss:0.158388 acc:0.5000 Time:        44 ms
Epoch:  100/500. loss:0.032223 acc:1.0000 Time:        49 ms
Epoch:  200/500. loss:0.015373 acc:1.0000 Time:        52 ms
Epoch:  300/500. loss:0.009219 acc:1.0000 Time:        55 ms
Epoch:  400/500. loss:0.006284 acc:1.0000 Time:        58 ms
Epoch:  500/500. loss:0.004645 acc:1.0000 Time:        61 ms
Time:61 ms

[0 0] = [0] -> 0.059674
[0 1] = [1] -> 0.900352
[1 0] = [1] -> 0.900713
[1 1] = [0] -> 0.117075

```

## Iris Dataset

This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.

Source : https://archive.ics.uci.edu/ml/datasets/iris

Example of MLP network.

```

(var trainX, var trainY, var testX, var testY) = ImportDataset.IrisDataset<U>(ratio: 0.8);
var net = new Network<U>(new SGD<U>(0.025f, 0.2f), new MeanSquaredLoss<U>(), new ArgmaxAccuracy<U>());
net.AddLayer(new DenseLayer<U>(5, inputShape: 4));
net.AddLayer(new TanhLayer<U>());
net.AddLayer(new DenseLayer<U>(3));
net.AddLayer(new SigmoidLayer<U>());

net.Summary();

net.Fit(trainX, trainY, 50, 10, 10);
net.Test(testX, testY);

```

### The Output

```
Hello World, MLP on Iris Dataset.
Train on 120 / Test on 30
Summary
Network: SGD / MeanSquaredLoss / ArgmaxAccuracy

====================================================
| Layer                | Parameters    | Output    |
====================================================
| InputLayer           |             0 |       (4) |
| DenseLayer           |            25 |       (5) |
| TanhActivation       |             0 |       (5) |
| DenseLayer           |            18 |       (3) |
| SigmoidActivation    |             0 |       (3) |
====================================================

Total Parameters:43

Epoch:    0/50. loss:0.118216 acc:0.3833 Time:        48 ms
Epoch:   10/50. loss:0.060735 acc:0.7583 Time:        54 ms
Epoch:   20/50. loss:0.052951 acc:0.8333 Time:        59 ms
Epoch:   30/50. loss:0.048401 acc:0.9583 Time:        64 ms
Epoch:   40/50. loss:0.044606 acc:0.9417 Time:        68 ms
Epoch:   50/50. loss:0.040406 acc:0.9500 Time:        73 ms
Time:73 ms
Test. loss:0.036146 acc:1.0000

```