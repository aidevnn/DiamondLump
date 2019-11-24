using System;
using NDarrayLib;

namespace DiamondLump.Optimizers
{
    public class Adam<U> : IOptimizer<U>
    {
        public Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f)
        {
            this.lr = lr;
            this.b1 = b1;
            this.b2 = b2;
        }

        readonly float lr, b1, b2;

        public string Name => "Adam";

        public IOptimizer<U> Clone() => new Adam<U>(lr, b1, b2);

        NDarray<U> M, V;
        public void Update(NDarray<U> w, NDarray<U> g)
        {
            if (M == null)
            {
                M = new NDarray<U>(g.Shape);
                V = new NDarray<U>(g.Shape);
            }

            if (typeof(U) == typeof(float))
                UpdateIntern(w.Data as float[], g.Data as float[], M.Data as float[], V.Data as float[]);

            if (typeof(U) == typeof(double))
                UpdateIntern(w.Data as double[], g.Data as double[], M.Data as double[], V.Data as double[]);
        }

        void UpdateIntern(float[] w, float[] g, float[] m, float[] v)
        {
            for (int i = 0; i < m.Length; ++i)
            {
                var g0 = g[i];
                m[i] = b1 * m[i] + (1f - b1) * g0;
                v[i] = b2 * v[i] + (1f - b2) * g0 * g0;
            }

            for (int i = 0; i < m.Length; ++i)
            {
                var mh = m[i] / (1f - b1);
                var vh = v[i] / (1f - b2);
                var w0 = lr * mh / ((float)Math.Sqrt(vh) + 1e-7f);

                w[i] -= w0;
            }
        }

        void UpdateIntern(double[] w, double[] g, double[] m, double[] v)
        {
            for (int i = 0; i < m.Length; ++i)
            {
                var g0 = g[i];
                m[i] = b1 * m[i] + (1.0 - b1) * g0;
                v[i] = b2 * v[i] + (1.0 - b2) * g0 * g0;
            }

            for (int i = 0; i < m.Length; ++i)
            {
                var mh = m[i] / (1.0 - b1);
                var vh = v[i] / (1.0 - b2);
                var w0 = lr * mh / (Math.Sqrt(vh) + 1e-7);

                w[i] -= w0;
            }
        }

    }
}
