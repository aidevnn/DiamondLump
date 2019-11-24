using System;
using NDarrayLib;

namespace DiamondLump.Optimizers
{
    public interface IOptimizer<U>
    {
        string Name { get; }
        IOptimizer<U> Clone();
        void Update(NDarray<U> w, NDarray<U> g);
    }
}
