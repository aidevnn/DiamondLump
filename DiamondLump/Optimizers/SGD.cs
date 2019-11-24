using System;
using NDarrayLib;
namespace DiamondLump.Optimizers
{
    public class SGD<U> : IOptimizer<U>
    {
        public SGD(float lr = 0.01f, float momentum = 0.0f)
        {
            this.lr = lr;
            this.momentum = momentum;
        }

        readonly float lr, momentum;

        public string Name => "SGD";

        public IOptimizer<U> Clone() => new SGD<U>(lr, momentum);

        NDarray<U> wUpdt;
        public void Update(NDarray<U> w, NDarray<U> g)
        {
            if (wUpdt == null)
                wUpdt = new NDarray<U>(w.Shape);

            if (typeof(U) == typeof(float))
                UpdateIntern(w.Data as float[], g.Data as float[], wUpdt.Data as float[]);

            if (typeof(U) == typeof(double))
                UpdateIntern(w.Data as double[], g.Data as double[], wUpdt.Data as double[]);
        }

        void UpdateIntern(float[] w, float[] g, float[] w0)
        {
            for (int i = 0; i < w0.Length; ++i)
                w0[i] = momentum * w0[i] + (1f - momentum) * g[i];

            for (int i = 0; i < w.Length; ++i)
                w[i] = w[i] - lr * w0[i];
        }

        void UpdateIntern(double[] w, double[] g, double[] w0)
        {
            for (int i = 0; i < w0.Length; ++i)
                w0[i] = momentum * w0[i] + (1.0 - momentum) * g[i];

            for (int i = 0; i < w.Length; ++i)
                w[i] = w[i] - lr * w0[i];
        }
    }
}
