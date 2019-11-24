using System;
using NDarrayLib;
namespace DiamondLump.Activations
{
    public class TanhActivation<U> : IActivation<U>
    {
        public string Name => "Tanh";

        public NDarray<U> Func(NDarray<U> X) => ND.Tanh(X);
        public NDarray<U> Grad(NDarray<U> X) => ND.DTanh(X);
    }
}
