using System;
using NDarrayLib;
namespace DiamondLump.Losses
{
    public interface ILoss<U>
    {
        string Name { get; }
        double Loss(NDarray<U> y, NDarray<U> p);
        NDarray<U> Grad(NDarray<U> y, NDarray<U> p);
    }
}
