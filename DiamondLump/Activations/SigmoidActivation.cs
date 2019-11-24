using System;
using NDarrayLib;
namespace DiamondLump.Activations
{
    public class SigmoidActivation<U> : IActivation<U>
    {
        public string Name => "Sigmoid";

        public NDarray<U> Func(NDarray<U> X) => ND.Sigmoid(X);
        public NDarray<U> Grad(NDarray<U> X) => ND.DSigmoid(X);
    }
}
