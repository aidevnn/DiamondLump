using System;
using NDarrayLib;
namespace DiamondLump.Activations
{
    public class SoftmaxActivation<U> : IActivation<U>
    {
        public string Name => "Softmax";

        public NDarray<U> Func(NDarray<U> X) => Softmax(X);
        public NDarray<U> Grad(NDarray<U> X) => DSoftmax(X);

        static NDarray<float> Softmaxfloat(NDarray<float> X)
        {
            int axis = X.Shape.Length - 1;
            int shapeAxis = X.Shape[axis];
            var mx = ND.MaxAxis(X, axis, true);
            float[] data0 = new float[X.Count];
            for (int i = 0; i < X.Count; ++i)
                data0[i] = (float)Math.Exp(X.Data[i] - mx.Data[i / shapeAxis]);

            var ex = new NDarray<float>(data: data0, shape: X.Shape);
            var sx = ND.SumAxis(ex, axis, true);
            for (int i = 0; i < X.Count; ++i)
                ex.Data[i] /= sx.Data[i / shapeAxis];

            return ex;
        }

        static NDarray<double> Softmaxdouble(NDarray<double> X)
        {
            int axis = X.Shape.Length - 1;
            int shapeAxis = X.Shape[axis];
            var mx = ND.MaxAxis(X, axis, true);
            double[] data0 = new double[X.Count];
            for (int i = 0; i < X.Count; ++i)
                data0[i] = Math.Exp(X.Data[i] - mx.Data[i / shapeAxis]);

            var ex = new NDarray<double>(data: data0, shape: X.Shape);
            var sx = ND.SumAxis(ex, axis, true);
            for (int i = 0; i < X.Count; ++i)
                ex.Data[i] /= sx.Data[i / shapeAxis];

            return ex;
        }

        static NDarray<V> Softmax<V>(NDarray<V> X)
        {
            if (typeof(V) == typeof(float))
            {
                var xf = X as NDarray<float>;
                return Softmaxfloat(xf) as NDarray<V>;
            }
            if (typeof(V) == typeof(double))
            {
                var xd = X as NDarray<double>;
                return Softmaxdouble(xd) as NDarray<V>;
            }

            throw new ArgumentException($"{typeof(V).Name} is not supported. Only float or double");
        }

        static NDarray<V> DSoftmax<V>(NDarray<V> X)
        {
            if (typeof(V) == typeof(float))
            {
                var xf = X as NDarray<float>;
                var xf0 = Softmax(xf);
                return xf0.ApplyFunc(x => x * (1f - x)) as NDarray<V>;
            }
            if (typeof(V) == typeof(double))
            {
                var xd = X as NDarray<double>;
                var xd0 = Softmax(xd);
                return xd0.ApplyFunc(x => x * (1.0 - x)) as NDarray<V>;
            }

            throw new ArgumentException($"{typeof(V).Name} is not supported. Only float or double");
        }
    }
}
