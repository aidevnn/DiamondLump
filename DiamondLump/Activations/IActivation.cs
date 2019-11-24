using System;
using NDarrayLib;
namespace DiamondLump.Activations
{
    public interface IActivation<U>
    {
        string Name { get; }
        NDarray<U> Func(NDarray<U> X);
        NDarray<U> Grad(NDarray<U> X);
    }
}
