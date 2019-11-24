using System;
using NDarrayLib;
namespace DiamondLump.Activations
{
    public class ReluActivation<U> : IActivation<U>
    {
        public string Name => "Relu";

        public NDarray<U> Func(NDarray<U> X) => Relu(X);
        public NDarray<U> Grad(NDarray<U> X) => DRelu(X);

        static NDarray<V> Relu<V>(NDarray<V> x)
        {
            if (typeof(V) == typeof(float))
            {
                var xf = x as NDarray<float>;
                return xf.ApplyFunc(a => a >= 0f ? a : 0f) as NDarray<V>;
            }
            if (typeof(U) == typeof(double))
            {
                var xd = x as NDarray<double>;
                return xd.ApplyFunc(a => a >= 0.0 ? a : 0.0) as NDarray<V>;
            }

            throw new ArgumentException($"{typeof(U).Name} is not supported. Only float or double");
        }

        static NDarray<V> DRelu<V>(NDarray<V> x)
        {
            if (typeof(V) == typeof(float))
            {
                var xf = x as NDarray<float>;
                return xf.ApplyFunc(a => a >= 0f ? 1f : 0f) as NDarray<V>;
            }
            if (typeof(U) == typeof(double))
            {
                var xd = x as NDarray<double>;
                return xd.ApplyFunc(a => a >= 0.0 ? 1.0 : 0.0) as NDarray<V>;
            }

            throw new ArgumentException($"{typeof(U).Name} is not supported. Only float or double");
        }

    }
}