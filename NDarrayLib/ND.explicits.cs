using System;
using System.Linq;

namespace NDarrayLib
{
    public static partial class ND
    {
        #region AXpBY
        static NDarray<int> AXpBYint(this NDarray<int> x, NDarray<int> y, int a = 1, int b = 1)
        {
            var yShape = Utils.BroadCastLeftShapes(x.Shape, y.Shape);
            var xStrides = Utils.Shape2Strides(x.Shape);
            var yStrides = Utils.Shape2Strides(yShape);
            int func(int x0, int y0) => a * x0 + b * y0;
            RecursiveFxyx(0, x.Data, x.Shape, xStrides, 0, y.Data, yShape, yStrides, 0, func);

            return x;
        }

        static NDarray<float> AXpBYfloat(this NDarray<float> x, NDarray<float> y, float a = 1, float b = 1)
        {
            var yShape = Utils.BroadCastLeftShapes(x.Shape, y.Shape);
            var xStrides = Utils.Shape2Strides(x.Shape);
            var yStrides = Utils.Shape2Strides(yShape);
            float func(float x0, float y0) => a * x0 + b * y0;
            RecursiveFxyx(0, x.Data, x.Shape, xStrides, 0, y.Data, yShape, yStrides, 0, func);

            return x;
        }

        static NDarray<double> AXpBYdouble(this NDarray<double> x, NDarray<double> y, double a = 1, double b = 1)
        {
            var yShape = Utils.BroadCastLeftShapes(x.Shape, y.Shape);
            var xStrides = Utils.Shape2Strides(x.Shape);
            var yStrides = Utils.Shape2Strides(yShape);
            double func(double x0, double y0) => a * x0 + b * y0;
            RecursiveFxyx(0, x.Data, x.Shape, xStrides, 0, y.Data, yShape, yStrides, 0, func);

            return x;
        }

        static NDarray<int> AXpBYintFast(this NDarray<int> x, NDarray<int> y, int a = 1, int b = 1)
        {
            if (!x.Shape.SequenceEqual(y.Shape)) throw new Exception();
            x.ApplyFuncInplace((i, x0) => a * x0 + b * y.Data[i]);
            return x;
        }

        static NDarray<float> AXpBYfloatFast(this NDarray<float> x, NDarray<float> y, float a = 1, float b = 1)
        {
            if (!x.Shape.SequenceEqual(y.Shape)) throw new Exception();
            x.ApplyFuncInplace((i, x0) => a * x0 + b * y.Data[i]);
            return x;
        }

        static NDarray<double> AXpBYdoubleFast(this NDarray<double> x, NDarray<double> y, double a = 1, double b = 1)
        {
            if (!x.Shape.SequenceEqual(y.Shape)) throw new Exception();
            x.ApplyFuncInplace((i, x0) => a * x0 + b * y.Data[i]);
            return x;
        }

        public static NDarray<U> AXpBY<U>(this NDarray<U> x, NDarray<U> y, double a = 1, double b = 1)
        {
            if (typeof(U) == typeof(int))
            {
                var xi = x as NDarray<int>;
                var yi = y as NDarray<int>;
                return (xi.AXpBYint(yi, Convert.ToInt32(a), Convert.ToInt32(b))) as NDarray<U>;
            }
            if (typeof(U) == typeof(float))
            {
                var xf = x as NDarray<float>;
                var yf = y as NDarray<float>;
                return (xf.AXpBYfloat(yf, Convert.ToSingle(a), Convert.ToSingle(b))) as NDarray<U>;
            }
            if (typeof(U) == typeof(double))
            {
                var xd = x as NDarray<double>;
                var yd = y as NDarray<double>;
                return (xd.AXpBYdouble(yd, a, b)) as NDarray<U>;
            }

            throw new ArgumentException($"{typeof(U).Name} is not supported. Only int, float or double");
        }

        public static NDarray<U> AXpBYFast<U>(this NDarray<U> x, NDarray<U> y, double a = 1, double b = 1)
        {
            if (typeof(U) == typeof(int))
            {
                var xi = x as NDarray<int>;
                var yi = y as NDarray<int>;
                return (xi.AXpBYintFast(yi, Convert.ToInt32(a), Convert.ToInt32(b))) as NDarray<U>;
            }
            if (typeof(U) == typeof(float))
            {
                var xf = x as NDarray<float>;
                var yf = y as NDarray<float>;
                return (xf.AXpBYfloatFast(yf, Convert.ToSingle(a), Convert.ToSingle(b))) as NDarray<U>;
            }
            if (typeof(U) == typeof(double))
            {
                var xd = x as NDarray<double>;
                var yd = y as NDarray<double>;
                return (xd.AXpBYdoubleFast(yd, a, b)) as NDarray<U>;
            }

            throw new ArgumentException($"{typeof(U).Name} is not supported. Only int, float or double");
        }

        public static NDarray<U> AXmBY<U>(this NDarray<U> x, NDarray<U> y, double a = 1, double b = 1) => x.AXpBY(y, a, -b);

        public static NDarray<U> AXmBYFast<U>(this NDarray<U> x, NDarray<U> y, double a = 1, double b = 1) => x.AXpBYFast(y, a, -b);

        #endregion

        #region VarAxis
        static NDarray<double> VarAxisint(NDarray<int> nDarray, int axis, bool keepdims = false, NDarray<double> y = null)
        {
            axis = (axis + nDarray.Shape.Length) % nDarray.Shape.Length;
            var nshape = Utils.PrepareAxisOps(nDarray.Shape, axis, keepdims);
            if (y == null)
                y = new NDarray<double>(nshape);
            else if (!nshape.SequenceEqual(y.Shape)) throw new Exception();

            int m = Utils.ArrMul(nDarray.Shape, axis);
            int n = m / nDarray.Shape[axis];
            double coef = nDarray.Shape[axis];

            double[] sum = new double[y.Count];
            double[] mean = new double[y.Count];

            for (int idx0 = 0; idx0 < nDarray.Count; ++idx0)
            {
                int idx1 = (idx0 / m) * n + idx0 % n;
                var x = nDarray.Data[idx0];
                mean[idx1] += x / coef;
                sum[idx1] += x * x / coef;
            }

            for (int idx = 0; idx < y.Count; ++idx)
            {
                var s0 = sum[idx];
                var m0 = mean[idx];
                y.Data[idx] = s0 - m0 * m0;
            }

            return y;
        }

        static NDarray<double> VarAxisfloat(NDarray<float> nDarray, int axis, bool keepdims = false, NDarray<double> y = null)
        {
            axis = (axis + nDarray.Shape.Length) % nDarray.Shape.Length;
            var nshape = Utils.PrepareAxisOps(nDarray.Shape, axis, keepdims);
            if (y == null)
                y = new NDarray<double>(nshape);
            else if (!nshape.SequenceEqual(y.Shape)) throw new Exception();

            int m = Utils.ArrMul(nDarray.Shape, axis);
            int n = m / nDarray.Shape[axis];
            float coef = nDarray.Shape[axis];

            float[] sum = new float[y.Count];
            float[] mean = new float[y.Count];

            for (int idx0 = 0; idx0 < nDarray.Count; ++idx0)
            {
                int idx1 = (idx0 / m) * n + idx0 % n;
                var x = nDarray.Data[idx0];
                mean[idx1] += x / coef;
                sum[idx1] += x * x / coef;
            }

            for (int idx = 0; idx < y.Count; ++idx)
            {
                var s0 = sum[idx];
                var m0 = mean[idx];
                y.Data[idx] = s0 - m0 * m0;
            }

            return y;
        }

        static NDarray<double> VarAxisdouble(NDarray<double> nDarray, int axis, bool keepdims = false, NDarray<double> y = null)
        {
            axis = (axis + nDarray.Shape.Length) % nDarray.Shape.Length;
            var nshape = Utils.PrepareAxisOps(nDarray.Shape, axis, keepdims);
            if (y == null)
                y = new NDarray<double>(nshape);
            else if (!nshape.SequenceEqual(y.Shape)) throw new Exception();

            int m = Utils.ArrMul(nDarray.Shape, axis);
            int n = m / nDarray.Shape[axis];
            double coef = nDarray.Shape[axis];

            double[] sum = new double[y.Count];
            double[] mean = new double[y.Count];

            for (int idx0 = 0; idx0 < nDarray.Count; ++idx0)
            {
                int idx1 = (idx0 / m) * n + idx0 % n;
                var x = nDarray.Data[idx0];
                mean[idx1] += x / coef;
                sum[idx1] += x * x / coef;
            }

            for (int idx = 0; idx < y.Count; ++idx)
            {
                var s0 = sum[idx];
                var m0 = mean[idx];
                y.Data[idx] = s0 - m0 * m0;
            }

            return y;
        }

        public static NDarray<double> VarAxis<U>(NDarray<U> x, int axis, bool keepdims = false, NDarray<double> y = null)
        {
            if (typeof(U) == typeof(int))
            {
                var xi = x as NDarray<int>;
                return VarAxisint(xi, axis, keepdims, y);
            }
            if (typeof(U) == typeof(float))
            {
                var xf = x as NDarray<float>;
                return VarAxisfloat(xf, axis, keepdims, y);
            }
            if (typeof(U) == typeof(double))
            {
                var xd = x as NDarray<double>;
                return VarAxisdouble(xd, axis, keepdims, y);
            }

            throw new ArgumentException($"{typeof(U).Name} is not supported. Only int, float or double");
        }

        #endregion

        #region TensorDot
        static NDarray<int> Dotint(NDarray<int> x, NDarray<int> y, NDarray<int> z = null)
        {
            var nShape = Utils.PrepareDot(x.Shape, y.Shape);
            if (z == null)
                z = new NDarray<int>(nShape);
            else if (!nShape.SequenceEqual(z.Shape)) throw new Exception();

            int yrow = x.Shape.Last();
            int ycol = y.Shape.Length == 1 ? 1 : y.Shape.Last();

            int offz = 0;
            for (int offx = 0; offx < x.Count; offx += yrow)
            {
                for (int offy = 0; offy < y.Count; offy += yrow * ycol)
                {
                    for (int j = 0; j < ycol; ++j)
                    {
                        int sum = 0;
                        for (int i = 0; i < yrow; ++i)
                            sum += x.Data[offx + i] * y.Data[offy + j + i * ycol];

                        z.Data[offz++] = sum;
                    }
                }
            }

            return z;
        }

        static NDarray<float> Dotfloat(NDarray<float> x, NDarray<float> y, NDarray<float> z = null)
        {
            var nShape = Utils.PrepareDot(x.Shape, y.Shape);
            if (z == null)
                z = new NDarray<float>(nShape);
            else if (!nShape.SequenceEqual(z.Shape)) throw new Exception();

            int yrow = x.Shape.Last();
            int ycol = y.Shape.Length == 1 ? 1 : y.Shape.Last();

            int offz = 0;
            for (int offx = 0; offx < x.Count; offx += yrow)
            {
                for (int offy = 0; offy < y.Count; offy += yrow * ycol)
                {
                    for (int j = 0; j < ycol; ++j)
                    {
                        float sum = 0f;
                        for (int i = 0; i < yrow; ++i)
                            sum += x.Data[offx + i] * y.Data[offy + j + i * ycol];

                        z.Data[offz++] = sum;
                    }
                }
            }

            return z;
        }

        static NDarray<double> Dotdouble(NDarray<double> x, NDarray<double> y, NDarray<double> z = null)
        {
            var nShape = Utils.PrepareDot(x.Shape, y.Shape);
            if (z == null)
                z = new NDarray<double>(nShape);
            else if (!nShape.SequenceEqual(z.Shape)) throw new Exception();

            int yrow = x.Shape.Last();
            int ycol = y.Shape.Length == 1 ? 1 : y.Shape.Last();

            int offz = 0;
            for (int offx = 0; offx < x.Count; offx += yrow)
            {
                for (int offy = 0; offy < y.Count; offy += yrow * ycol)
                {
                    for (int j = 0; j < ycol; ++j)
                    {
                        double sum = 0.0;
                        for (int i = 0; i < yrow; ++i)
                            sum += x.Data[offx + i] * y.Data[offy + j + i * ycol];

                        z.Data[offz++] = sum;
                    }
                }
            }

            return z;
        }

        public static NDarray<U> Dot<U>(NDarray<U> x, NDarray<U> y, NDarray<U> z = null)
        {
            if (typeof(U) == typeof(int))
            {
                var xi = x as NDarray<int>;
                var yi = y as NDarray<int>;
                var zi = z as NDarray<int>;
                return Dotint(xi, yi, zi) as NDarray<U>;
            }
            if (typeof(U) == typeof(float))
            {
                var xf = x as NDarray<float>;
                var yf = y as NDarray<float>;
                var zf = z as NDarray<float>;
                return Dotfloat(xf, yf, zf) as NDarray<U>;
            }
            if (typeof(U) == typeof(double))
            {
                var xd = x as NDarray<double>;
                var yd = y as NDarray<double>;
                var zd = z as NDarray<double>;
                return Dotdouble(xd, yd, zd) as NDarray<U>;
            }

            throw new ArgumentException($"{typeof(U).Name} is not supported. Only int, float or double");
        }
        #endregion

    }
}
