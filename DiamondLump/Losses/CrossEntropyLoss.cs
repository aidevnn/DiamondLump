using System;
using NDarrayLib;
namespace DiamondLump.Losses
{
    public class CrossEntropyLoss<U> : ILoss<U>
    {
        public string Name => "CrossEntropyLoss";

        public NDarray<U> Grad(NDarray<U> y, NDarray<U> p) => DCE(y, p);

        public double Loss(NDarray<U> y, NDarray<U> p) => ND.MeanAxis(CE(y, p)).Data[0];

        static double CEfunc(double y, double p)
        {
            var p0 = Math.Min(1 - 1e-15, Math.Max(1e-15, p));
            return -y * Math.Log(p0) - (1 - y) * Math.Log(1 - p0);
        }

        static double CEgrad(double y, double p)
        {
            var p0 = Math.Min(1 - 1e-15, Math.Max(1e-15, p));
            return -y / p0 + (1 - y) / (1 - p0);
        }

        static NDarray<V> CE<V>(NDarray<V> x, NDarray<V> y)
        {
            if (typeof(V) == typeof(float))
            {
                var xf = x as NDarray<float>;
                var yf = y as NDarray<float>;
                return ND.ApplyFuncAB(xf, yf, (a, b) => (float)CEfunc(a, b)) as NDarray<V>;
            }
            if (typeof(V) == typeof(double))
            {
                var xd = x as NDarray<double>;
                var yd = y as NDarray<double>;
                return ND.ApplyFuncAB(xd, yd, CEfunc) as NDarray<V>;
            }

            throw new ArgumentException($"{typeof(U).Name} is not supported. Only float or double");
        }

        static NDarray<V> DCE<V>(NDarray<V> x, NDarray<V> y)
        {
            if (typeof(V) == typeof(float))
            {
                var xf = x as NDarray<float>;
                var yf = y as NDarray<float>;
                return ND.ApplyFuncAB(xf, yf, (a, b) => (float)CEgrad(a, b)) as NDarray<V>;
            }
            if (typeof(V) == typeof(double))
            {
                var xd = x as NDarray<double>;
                var yd = y as NDarray<double>;
                return ND.ApplyFuncAB(xd, yd, CEgrad) as NDarray<V>;
            }

            throw new ArgumentException($"{typeof(U).Name} is not supported. Only float or double");
        }
    }
}
