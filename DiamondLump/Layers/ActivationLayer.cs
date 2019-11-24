using System;
using System.Linq;
using DiamondLump.Activations;
using DiamondLump.Optimizers;
using NDarrayLib;

namespace DiamondLump.Layers
{
    public class ActivationLayer<U> : ILayer<U>
    {
        public ActivationLayer(IActivation<U> activation)
        {
            this.activation = activation;
        }

        readonly IActivation<U> activation;

        public string Name => $"{activation.Name}Activation";

        public int Params => 0;

        public bool IsTraining { get; set; }
        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        NDarray<U> LastInput;

        public NDarray<U> Backward(NDarray<U> accumGrad)
        {
            var x = activation.Grad(LastInput);
            return ND.Mul(x, accumGrad);
        }

        public NDarray<U> Forward(NDarray<U> X, bool isTraining)
        {
            LastInput = new NDarray<U>(X);
            return activation.Func(X);
        }

        public int[] GetOutputShape() => OutputShape;

        public void Initialize(IOptimizer<U> optimizer) { }

        public void SetInputShape(int[] shape)
        {
            InputShape = shape.ToArray();
            OutputShape = shape.ToArray();
        }

        public void ImportWeights(string w, string b)
        {

        }
    }

    public class SigmoidLayer<U> : ActivationLayer<U>
    {
        public SigmoidLayer() : base(new SigmoidActivation<U>()) { }
    }

    public class TanhLayer<U> : ActivationLayer<U>
    {
        public TanhLayer() : base(new TanhActivation<U>()) { }
    }

    public class ReluLayer<U> : ActivationLayer<U>
    {
        public ReluLayer() : base(new ReluActivation<U>()) { }
    }

    public class SoftmaxLayer<U> : ActivationLayer<U>
    {
        public SoftmaxLayer() : base(new SoftmaxActivation<U>()) { }
    }
}
