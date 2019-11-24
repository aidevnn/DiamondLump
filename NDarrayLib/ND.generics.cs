using System;
using System.Linq;

namespace NDarrayLib
{
    public static partial class ND
    {
        #region Recursives Functions
        static void RecursiveFxy<U, V>(bool arrx, int axis, U[] x, int[] xShape, int[] xStrides, int offx, V[] y, int[] yShape, int[] yStrides, int offy, Func<U, V> func)
        {
            int incx = xStrides[axis];
            int incy = yStrides[axis];

            int offx0 = offx;
            int offy0 = offy;

            int[] fShape = arrx ? xShape : yShape;
            if (axis == fShape.Length - 1)
            {
                for (int k = 0; k < fShape[axis]; ++k, offx0 += incx, offy0 += incy)
                    y[offy0] = func(x[offx0]);
            }
            else
            {
                for (int k = 0; k < fShape[axis]; ++k, offx0 += incx, offy0 += incy)
                    RecursiveFxy(arrx, axis + 1, x, xShape, xStrides, offx0, y, yShape, yStrides, offy0, func);
            }
        }

        static void RecursiveFxyx<U, V>(int axis, U[] x, int[] xShape, int[] xStrides, int offx, V[] y, int[] yShape, int[] yStrides, int offy, Func<U, V, U> func)
        {
            int incx = xStrides[axis];
            int incy = yShape[axis] == 1 ? 0 : yStrides[axis];

            int offx0 = offx;
            int offy0 = offy;

            if (axis == xShape.Length - 1)
            {
                for (int k = 0; k < xShape[axis]; ++k, offx0 += incx, offy0 += incy)
                    x[offx0] = func(x[offx0], y[offy0]);
            }
            else
            {
                for (int k = 0; k < xShape[axis]; ++k, offx0 += incx, offy0 += incy)
                    RecursiveFxyx(axis + 1, x, xShape, xStrides, offx0, y, yShape, yStrides, offy0, func);
            }
        }

        static void RecursiveFxyy<U, V>(int axis, U[] x, int[] xShape, int[] xStrides, int offx, V[] y, int[] yShape, int[] yStrides, int offy, Func<U, V, V> func)
        {
            int incx = xStrides[axis];
            int incy = yShape[axis] == 1 ? 0 : yStrides[axis];

            int offx0 = offx;
            int offy0 = offy;

            if (axis == xShape.Length - 1)
            {
                for (int k = 0; k < xShape[axis]; ++k, offx0 += incx, offy0 += incy)
                    y[offy0] = func(x[offx0], y[offy0]);
            }
            else
            {
                for (int k = 0; k < xShape[axis]; ++k, offx0 += incx, offy0 += incy)
                    RecursiveFxyy(axis + 1, x, xShape, xStrides, offx0, y, yShape, yStrides, offy0, func);
            }
        }

        static void RecursiveFxyz<U, V, W>(int axis, U[] x, int[] xShape, int[] xStrides, int offx, V[] y, int[] yShape, int[] yStrides, int offy, W[] z, int[] zShape, int[] zStrides, int offz, Func<U, V, W> func)
        {
            int incx = xShape[axis] == 1 ? 0 : xStrides[axis];
            int incy = yShape[axis] == 1 ? 0 : yStrides[axis];
            int incz = zStrides[axis];

            int offx0 = offx;
            int offy0 = offy;
            int offz0 = offz;

            if (axis == zShape.Length - 1)
            {
                for (int k = 0; k < zShape[axis]; ++k, offx0 += incx, offy0 += incy, offz0 += incz)
                    z[offz0] = func(x[offx0], y[offy0]);
            }
            else
            {
                for (int k = 0; k < zShape[axis]; ++k, offx0 += incx, offy0 += incy, offz0 += incz)
                    RecursiveFxyz(axis + 1, x, xShape, xStrides, offx0, y, yShape, yStrides, offy0, z, zShape, zStrides, offz0, func);
            }
        }
        #endregion

        #region Reshape / Transpose
        public static NDarray<U> Reshape<U>(NDarray<U> nDarray, params int[] shape)
        {
            var nshape = Utils.PrepareReshape(nDarray.Shape, shape);
            return new NDarray<U>(nDarray.Data.ToArray(), nshape);
        }

        public static NDarray<U> Transpose<U>(NDarray<U> x, int[] table, NDarray<U> y = null)
        {
            var xShape = x.Shape;
            var xStrides = Utils.Shape2Strides(xShape);
            var yShape = Utils.DoTranspose(xShape, table);
            var yStrides = Utils.DoTranspose(Utils.Shape2Strides(yShape), table);
            if (y == null)
                y = new NDarray<U>(yShape);
            else if (!yShape.SequenceEqual(y.Shape)) throw new Exception();

            RecursiveFxy(true, 0, x.Data, xShape, xStrides, 0, y.Data, yShape, yStrides, 0, a => a);
            return y;
        }

        public static NDarray<U> Transpose<U>(NDarray<U> x, NDarray<U> y = null) => Transpose(x, Utils.PrepareTranspose(x.Shape.Length), y);
        #endregion

        #region Mathematics

        public static NDarray<U> Abs<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Abs);
        public static NDarray<U> Exp<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Exp);
        public static NDarray<U> Log<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Log);
        public static NDarray<U> Sq<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Sq);
        public static NDarray<U> Sqrt<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Sqrt);
        public static NDarray<U> Sigmoid<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Sigmoid);
        public static NDarray<U> DSigmoid<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.DSigmoid);
        public static NDarray<U> Tanh<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Tanh);
        public static NDarray<U> DTanh<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.DTanh);
        public static NDarray<U> Neg<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Neg);
        public static NDarray<U> Round<U>(NDarray<U> x, int dec = 0) => x.ApplyFunc(x0 => NDarray<U>.OpsT.Round(x0, dec));
        public static NDarray<U> Clamp<U>(NDarray<U> x, double min, double max) => x.ApplyFunc(x0 => NDarray<U>.OpsT.Clamp(x0, min, max));

        public static NDarray<V> ApplyFuncAB<U, V>(NDarray<U> a, NDarray<U> b, Func<U, U, V> func)
        {
            if (a.Count != b.Count) throw new Exception();
            NDarray<V> c = new NDarray<V>(a.Shape);
            for (int idx = 0; idx < a.Count; ++idx)
                c.Data[idx] = func(a.Data[idx], b.Data[idx]);

            return c;
        }

        #endregion

        #region ElementWiseOp
        static NDarray<W> ElementWiseOp<U, V, W>(NDarray<U> x, NDarray<V> y, NDarray<W> z, Func<U, V, W> func)
        {
            var (xShape, yShape, zShape) = Utils.BroadCastShapes(x.Shape, y.Shape);
            if (z == null)
                z = new NDarray<W>(zShape);
            else if (!z.Shape.SequenceEqual(zShape)) throw new Exception();

            var xStrides = Utils.Shape2Strides(xShape);
            var yStrides = Utils.Shape2Strides(yShape);
            var zStrides = Utils.Shape2Strides(zShape);

            RecursiveFxyz(0, x.Data, xShape, xStrides, 0, y.Data, yShape, yStrides, 0, z.Data, zShape, zStrides, 0, func);
            return z;
        }

        public static NDarray<U> Add<U>(NDarray<U> x, NDarray<U> y, NDarray<U> z = null) => ElementWiseOp(x, y, z, NDarray<U>.OpsT.Add);
        public static NDarray<U> Sub<U>(NDarray<U> x, NDarray<U> y, NDarray<U> z = null) => ElementWiseOp(x, y, z, NDarray<U>.OpsT.Sub);
        public static NDarray<U> Mul<U>(NDarray<U> x, NDarray<U> y, NDarray<U> z = null) => ElementWiseOp(x, y, z, NDarray<U>.OpsT.Mul);
        public static NDarray<U> Div<U>(NDarray<U> x, NDarray<U> y, NDarray<U> z = null) => ElementWiseOp(x, y, z, NDarray<U>.OpsT.Div);
        public static NDarray<U> Min<U>(NDarray<U> x, NDarray<U> y, NDarray<U> z = null) => ElementWiseOp(x, y, z, NDarray<U>.OpsT.Min);
        public static NDarray<U> Max<U>(NDarray<U> x, NDarray<U> y, NDarray<U> z = null) => ElementWiseOp(x, y, z, NDarray<U>.OpsT.Max);

        public static NDarray<double> Eq<U>(NDarray<U> x, NDarray<U> y, NDarray<double> z = null) => ElementWiseOp(x, y, z, NDarray<U>.OpsT.Eq);
        public static NDarray<double> Lt<U>(NDarray<U> x, NDarray<U> y, NDarray<double> z = null) => ElementWiseOp(x, y, z, NDarray<U>.OpsT.Lt);
        public static NDarray<double> Gt<U>(NDarray<U> x, NDarray<U> y, NDarray<double> z = null) => ElementWiseOp(x, y, z, NDarray<U>.OpsT.Gt);
        public static NDarray<double> Neq<U>(NDarray<U> x, NDarray<U> y, NDarray<double> z = null) => ElementWiseOp(x, y, z, NDarray<U>.OpsT.Neq);
        public static NDarray<double> Lte<U>(NDarray<U> x, NDarray<U> y, NDarray<double> z = null) => ElementWiseOp(x, y, z, NDarray<U>.OpsT.Lte);
        public static NDarray<double> Gte<U>(NDarray<U> x, NDarray<U> y, NDarray<double> z = null) => ElementWiseOp(x, y, z, NDarray<U>.OpsT.Gte);
        #endregion

        #region AxisOp
        static NDarray<V> AxisOp<U, V>(NDarray<U> x, int axis, bool keepdims, V neutral, Func<U, V, V> func, NDarray<V> y = null)
        {
            axis = (axis + x.Shape.Length) % x.Shape.Length;
            var xShape = x.Shape;
            var yShape = Utils.PrepareAxisOps(xShape, axis, true);
            var yShape0 = Utils.PrepareAxisOps(xShape, axis, keepdims);
            if (y == null)
                y = Constante(neutral, yShape0);
            else if (!yShape0.SequenceEqual(y.Shape)) throw new Exception();
            else
                for (int k = 0; k < y.Count; ++k) y.Data[k] = neutral;

            var xStrides = Utils.Shape2Strides(xShape);
            var yStrides = Utils.Shape2Strides(yShape);

            RecursiveFxyy(0, x.Data, xShape, xStrides, 0, y.Data, yShape, yStrides, 0, func);
            return y;
        }

        static NDarray<V> AxisOp<U, V>(NDarray<U> x, bool keepdims, V neutral, Func<U, V, V> func, NDarray<V> y = null)
        {
            var acc = x.Data.Aggregate(neutral, (a, b) => func(b, a));
            var shape = keepdims ? Enumerable.Repeat(1, x.Shape.Length).ToArray() : new int[] { 1 };

            if (y == null)
                y = Constante(acc, shape);
            else if (!shape.SequenceEqual(y.Shape)) throw new Exception();
            else
                for (int k = 0; k < y.Count; ++k) y.Data[k] = acc;

            return y;
        }

        public static NDarray<U> SumAxis<U>(NDarray<U> x, int axis, bool keepdims = false, NDarray<U> y = null)
            => AxisOp(x, axis, keepdims, NDarray<U>.OpsT.Zero, NDarray<U>.OpsT.Add, y);
        public static NDarray<U> SumAxis<U>(NDarray<U> x, bool keepdims = false, NDarray<U> y = null)
            => AxisOp(x, keepdims, NDarray<U>.OpsT.Zero, NDarray<U>.OpsT.Add, y);

        public static NDarray<U> ProdAxis<U>(NDarray<U> x, int axis, bool keepdims = false, NDarray<U> y = null)
            => AxisOp(x, axis, keepdims, NDarray<U>.OpsT.One, NDarray<U>.OpsT.Mul, y);
        public static NDarray<U> ProdAxis<U>(NDarray<U> x, bool keepdims = false, NDarray<U> y = null)
            => AxisOp(x, keepdims, NDarray<U>.OpsT.One, NDarray<U>.OpsT.Mul, y);

        public static NDarray<U> MinAxis<U>(NDarray<U> x, int axis, bool keepdims = false, NDarray<U> y = null)
            => AxisOp(x, axis, keepdims, NDarray<U>.OpsT.Maxvalue, NDarray<U>.OpsT.Min, y);
        public static NDarray<U> MinAxis<U>(NDarray<U> x, bool keepdims = false, NDarray<U> y = null)
            => AxisOp(x, keepdims, NDarray<U>.OpsT.Maxvalue, NDarray<U>.OpsT.Min, y);

        public static NDarray<U> MaxAxis<U>(NDarray<U> x, int axis, bool keepdims = false, NDarray<U> y = null)
            => AxisOp(x, axis, keepdims, NDarray<U>.OpsT.Minvalue, NDarray<U>.OpsT.Max, y);
        public static NDarray<U> MaxAxis<U>(NDarray<U> x, bool keepdims = false, NDarray<U> y = null)
            => AxisOp(x, keepdims, NDarray<U>.OpsT.Minvalue, NDarray<U>.OpsT.Max, y);

        public static NDarray<double> MeanAxis<U>(NDarray<U> x, int axis, bool keepdims = false, NDarray<double> y = null)
        {
            double n = x.Shape[axis];
            return AxisOp(x, axis, keepdims, 0, (a, b) => Convert.ToDouble(a) / n + b, y);
        }

        public static NDarray<double> MeanAxis<U>(NDarray<U> x, bool keepdims = false, NDarray<double> y = null)
        {
            double n = x.Count;
            return AxisOp(x, keepdims, 0, (a, b) => Convert.ToDouble(a) / n + b, y);
        }
        #endregion

        #region ArgMinMax
        static void RecursiveArgMinMax<U>(int[] orderedAxis, int idx, U[] x, int[] xShape, int[] xStrides, int offx, int[] y, int[] yShape, int[] yStrides, int offy, U bound, Func<U, U, double> func)
        {
            int axis = orderedAxis[idx];

            int incx = xStrides[axis];
            int incy = yShape[axis] == 1 ? 0 : yStrides[axis];
            int offx0 = offx, offy0 = offy;

            if (idx == xShape.Length - 1)
            {
                var bestValue = bound;
                int bestOffY = 0, bestIdx = 0;
                for (int k = 0; k < xShape[axis]; ++k, offx0 += incx, offy0 += incy)
                {
                    var newValue = x[offx0];
                    if (func(newValue, bestValue) > 0.0)
                    {
                        bestIdx = k;
                        bestOffY = offy0;
                        bestValue = newValue;
                    }
                }
                y[bestOffY] = bestIdx;
            }
            else
            {
                for (int k = 0; k < xShape[axis]; ++k, offx0 += incx, offy0 += incy)
                    RecursiveArgMinMax(orderedAxis, idx + 1, x, xShape, xStrides, offx0, y, yShape, yStrides, offy0, bound, func);
            }
        }

        static NDarray<int> ArgMinMax<U>(NDarray<U> x, int axis, bool keepdims, U bound, Func<U, U, double> func, NDarray<int> argmin = null)
        {
            axis = (axis + x.Shape.Length) % x.Shape.Length;
            var xShape = x.Shape.ToArray();
            var yShape = Utils.PrepareAxisOps(xShape, axis, true);
            var yShapef = Utils.PrepareAxisOps(xShape, axis, keepdims);
            if (argmin == null)
                argmin = new NDarray<int>(yShapef);
            else if (!yShapef.SequenceEqual(argmin.Shape)) throw new Exception();

            var xStrides = Utils.Shape2Strides(xShape);
            var yStrides = Utils.Shape2Strides(yShape);

            int[] orderAxis = Enumerable.Range(0, xShape.Length).ToArray();
            orderAxis[xShape.Length - 1] = orderAxis[axis];
            orderAxis[axis] = xShape.Length - 1;

            RecursiveArgMinMax(orderAxis, 0, x.Data, xShape, xStrides, 0, argmin.Data, yShape, yStrides, 0, bound, func);
            return argmin;
        }

        public static NDarray<int> ArgMin<U>(NDarray<U> x, int axis, bool keepdims = false, NDarray<int> argmin = null) => ArgMinMax(x, axis, keepdims, NDarray<U>.OpsT.Maxvalue, NDarray<U>.OpsT.Lt, argmin);
        public static NDarray<int> ArgMax<U>(NDarray<U> x, int axis, bool keepdims = false, NDarray<int> argmin = null) => ArgMinMax(x, axis, keepdims, NDarray<U>.OpsT.Minvalue, NDarray<U>.OpsT.Gt, argmin);
        #endregion

        #region Split / Stack / Pad / Repeat / Tile
        public static (NDarray<U>, NDarray<U>) Split<U>(NDarray<U> x, int axis, int idx)
        {
            (var y1Shape, var y2Shape) = Utils.PrepareSplit(x.Shape, axis, idx);
            NDarray<U> y1 = new NDarray<U>(y1Shape);
            NDarray<U> y2 = new NDarray<U>(y2Shape);

            var xStrides = Utils.Shape2Strides(x.Shape);
            var y1Strides = Utils.Shape2Strides(y1Shape);
            var y2Strides = Utils.Shape2Strides(y2Shape);

            RecursiveFxy(false, 0, x.Data, x.Shape, xStrides, 0, y1.Data, y1Shape, y1Strides, 0, a => a);

            int offx2 = y2Strides[axis] * idx;
            RecursiveFxy(false, 0, x.Data, x.Shape, xStrides, offx2, y2.Data, y2Shape, y2Strides, 0, a => a);

            return (y1, y2);
        }

        public static NDarray<U> Concatenate<U>(NDarray<U> x1, NDarray<U> x2, int axis)
        {
            var nshape = Utils.PrepareConcatenate(x1.Shape, x2.Shape, axis);
            var y = new NDarray<U>(nshape);

            var x1Strides = Utils.Shape2Strides(x1.Shape);
            var x2Strides = Utils.Shape2Strides(x2.Shape);
            var yStrides = Utils.Shape2Strides(y.Shape);

            RecursiveFxy(true, 0, x1.Data, x1.Shape, x1Strides, 0, y.Data, y.Shape, yStrides, 0, a => a);

            int offy = x1.Shape[axis] * yStrides[axis];
            RecursiveFxy(true, 0, x2.Data, x2.Shape, x2Strides, 0, y.Data, y.Shape, yStrides, offy, a => a);

            return y;
        }

        public static NDarray<U> Concatenate<U>(int axis, params NDarray<U>[] list)
        {
            if (list.Length == 0) throw new Exception();
            var y = list[0];
            for (int k = 1; k < list.Length; ++k)
                y = Concatenate(y, list[k], axis);

            return y;
        }

        public static NDarray<U> Pad<U>(NDarray<U> x, params (int, int)[] pads)
        {
            var (yShape, pads0) = Utils.PreparePad(x.Shape, pads);
            var y = new NDarray<U>(yShape);

            var xStrides = Utils.Shape2Strides(x.Shape);
            var yStrides = Utils.Shape2Strides(yShape);

            int offx = 0;
            for (int k = 0; k < x.Shape.Length; ++k)
                offx += yStrides[k] * pads0[k].Item1;

            RecursiveFxy(true, 0, x.Data, x.Shape, xStrides, 0, y.Data, yShape, yStrides, offx, a => a);
            return y;
        }

        public static NDarray<U> Repeat<U>(NDarray<U> x, int rep, int axis)
        {
            if (rep < 0 || axis < 0 || axis >= x.Shape.Length) throw new Exception();
            var yShape = x.Shape.ToArray();
            yShape[axis] *= rep;
            var y = new NDarray<U>(yShape);

            var xStrides = Utils.Shape2Strides(x.Shape);
            var yStrides = Utils.Shape2Strides(yShape);
            int step = yStrides[axis];
            yStrides[axis] *= rep;

            for (int k = 0, offy = 0; k < rep; ++k, offy += step)
                RecursiveFxy(true, 0, x.Data, x.Shape, xStrides, 0, y.Data, x.Shape, yStrides, offy, a => a);

            return y;
        }

        public static NDarray<U> Tile<U>(NDarray<U> x, params int[] reps)
        {
            var (yShape, nreps) = Utils.PrepareTile(x.Shape, reps);
            NDarray<U> y = new NDarray<U>(x.Data.ToArray(), yShape);

            for (int axis = 0; axis < nreps.Length; ++axis)
                y = Concatenate(axis, Enumerable.Repeat(y, nreps[axis]).ToArray());

            return y;
        }

        #endregion
    }
}
