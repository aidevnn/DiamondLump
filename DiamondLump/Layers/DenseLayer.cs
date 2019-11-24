using System;
using System.Linq;
using DiamondLump.Optimizers;
using NDarrayLib;

namespace DiamondLump.Layers
{
    public class DenseLayer<U> : ILayer<U>
    {
        public DenseLayer(int outNodes)
        {
            OutputShape = new int[] { outNodes };
        }

        public DenseLayer(int outNodes, int inputShape)
        {
            OutputShape = new int[] { outNodes };
            InputShape = new int[] { inputShape };
        }

        public string Name => "DenseLayer";

        public int Params => weight.Count + biases.Count;

        public bool IsTraining { get; set; }
        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        NDarray<U> LastInput, weight, biases, wTmp;
        IOptimizer<U> wOpt, bOpt;

        public NDarray<U> Backward(NDarray<U> accumGrad)
        {
            for (int i = 0; i < wTmp.Count; ++i)
                wTmp.Data[i] = weight.Data[i];

            if (IsTraining)
            {
                var gW = ND.GemmTAB(LastInput, accumGrad);
                var gB = ND.SumAxis(accumGrad, 0, true);

                wOpt.Update(weight, gW);
                bOpt.Update(biases, gB);
            }

            return ND.GemmATB(accumGrad, wTmp);
        }

        public NDarray<U> Forward(NDarray<U> X, bool isTraining)
        {
            IsTraining = isTraining;
            LastInput = new NDarray<U>(X);
            return ND.Add(ND.GemmAB(X, weight), biases);
        }

        public int[] GetOutputShape() => OutputShape;

        public void Initialize(IOptimizer<U> optimizer)
        {
            wOpt = optimizer.Clone();
            bOpt = optimizer.Clone();

            double lim = 3.0 / Math.Sqrt(InputShape[0]);
            weight = ND.Uniform(-lim, lim, InputShape[0], OutputShape[0]).Cast<U>();
            biases = new NDarray<double>(1, OutputShape[0]).Cast<U>();
            wTmp = new NDarray<double>(weight.Shape).Cast<U>();
        }

        public void SetInputShape(int[] shape)
        {
            InputShape = shape.ToArray();
        }

        public void ImportWeights(string w, string b)
        {

        }
    }
}
