using System;
using DiamondLump.Optimizers;
using NDarrayLib;

namespace DiamondLump.Layers
{
    public interface ILayer<U>
    {
        string Name { get; }
        int Params { get; }
        bool IsTraining { get; set; }
        int[] InputShape { get; set; }
        int[] OutputShape { get; set; }

        void SetInputShape(int[] shape);
        int[] GetOutputShape();

        NDarray<U> Backward(NDarray<U> accumGrad);
        NDarray<U> Forward(NDarray<U> X, bool isTraining);
        void Initialize(IOptimizer<U> optimizer);
        void ImportWeights(string w, string b);
    }
}
