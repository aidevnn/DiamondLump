using System;
using NDarrayLib;
namespace DiamondLump.Losses
{
    public class MeanSquaredLoss<U> : ILoss<U>
    {
        public string Name => "MeanSquaredLoss";

        public NDarray<U> Grad(NDarray<U> y, NDarray<U> p) => DMSE(y, p);

        public double Loss(NDarray<U> y, NDarray<U> p) => ND.MeanAxis(MSE(y, p)).Data[0];

        static NDarray<V> MSE<V>(NDarray<V> x, NDarray<V> y)
        {
            if (typeof(V) == typeof(float))
            {
                var xf = x as NDarray<float>;
                var yf = y as NDarray<float>;
                return ND.ApplyFuncAB(xf, yf, (a, b) => 0.5f * (a - b) * (a - b)) as NDarray<V>;
            }
            if (typeof(V) == typeof(double))
            {
                var xd = x as NDarray<double>;
                var yd = y as NDarray<double>;
                return ND.ApplyFuncAB(xd, yd, (a, b) => 0.5 * (a - b) * (a - b)) as NDarray<V>;
            }

            throw new ArgumentException($"{typeof(U).Name} is not supported. Only float or double");
        }

        static NDarray<V> DMSE<V>(NDarray<V> x, NDarray<V> y)
        {
            if (typeof(V) == typeof(float))
            {
                var xf = x as NDarray<float>;
                var yf = y as NDarray<float>;
                return ND.ApplyFuncAB(xf, yf, (a, b) => b - a) as NDarray<V>;
            }
            if (typeof(V) == typeof(double))
            {
                var xd = x as NDarray<double>;
                var yd = y as NDarray<double>;
                return ND.ApplyFuncAB(xd, yd, (a, b) => b - a) as NDarray<V>;
            }

            throw new ArgumentException($"{typeof(U).Name} is not supported. Only float or double");
        }

    }
}