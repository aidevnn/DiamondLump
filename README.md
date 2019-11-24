# DiamondLump

Simple Multi Layers Perceptron

## Xor Dataset

```
var (trainX, trainY) = ImportDataset.XorDataset<float>();
var net = new Network<float>(new SGD<float>(0.1f), new MeanSquaredLoss<float>(), new RoundAccuracy<float>());
net.AddLayer(new DenseLayer<float>(8, inputShape: 2));
net.AddLayer(new TanhLayer<float>());
net.AddLayer(new DenseLayer<float>(1));
net.AddLayer(new SigmoidLayer<float>());

net.Summary();
net.Fit(trainX, trainY, epochs: 500, displayEpochs: 100);
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

(var trainX, var trainY, var testX, var testY) = ImportDataset.IrisDataset<float>(ratio: 0.8);
var net = new Network<float>(new SGD<float>(0.025f, 0.2f), new MeanSquaredLoss<float>(), new ArgmaxAccuracy<float>());
net.AddLayer(new DenseLayer<float>(5, inputShape: 4));
net.AddLayer(new TanhLayer<float>());
net.AddLayer(new DenseLayer<float>(3));
net.AddLayer(new SigmoidLayer<float>());

net.Summary();

net.Fit(trainX, trainY, epochs: 50, batchSize: 10, displayEpochs: 10);
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

## Digits Dataset

This dataset is made up of 1797 8x8 images. Each image is of a hand-written digit.

Source : https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html

Example of MLP network.

```
(var trainX, var trainY, var testX, var testY) = ImportDataset.DigitsDataset<float>(ratio: 0.9);
var net = new Network<float>(new SGD<float>(0.025f, 0.2f), new CrossEntropyLoss<float>(), new ArgmaxAccuracy<float>());
net.AddLayer(new DenseLayer<float>(32, inputShape: 64));
net.AddLayer(new SigmoidLayer<float>());
net.AddLayer(new DenseLayer<float>(10));
net.AddLayer(new SoftmaxLayer<float>());

net.Summary();
net.Fit(trainX, trainY, testX, testY, epochs: 50, batchSize: 100, displayEpochs: 10);

```

### The Output


```
Hello World, MLP on Digits Dataset.
Train on 1617 / Test on 180
Summary
Network: SGD / CrossEntropyLoss / ArgmaxAccuracy

====================================================
| Layer                | Parameters    | Output    |
====================================================
| InputLayer           |             0 |      (64) |
| DenseLayer           |          2080 |      (32) |
| SigmoidActivation    |             0 |      (32) |
| DenseLayer           |           330 |      (10) |
| SoftmaxActivation    |             0 |      (10) |
====================================================

Total Parameters:2410

Epoch:    0/50. loss:0.269325 acc:0.4194; Validation. loss:0.194694 acc:0.5944 Time:        55 ms
Epoch:   10/50. loss:0.020807 acc:0.9756; Validation. loss:0.048198 acc:0.9056 Time:       186 ms
Epoch:   20/50. loss:0.012417 acc:0.9856; Validation. loss:0.036307 acc:0.9389 Time:       310 ms
Epoch:   30/50. loss:0.008853 acc:0.9919; Validation. loss:0.036102 acc:0.9278 Time:       481 ms
Epoch:   40/50. loss:0.006138 acc:0.9963; Validation. loss:0.032928 acc:0.9444 Time:       674 ms
Epoch:   50/50. loss:0.004598 acc:0.9981; Validation. loss:0.032193 acc:0.9444 Time:       864 ms
Time:864 ms
```

