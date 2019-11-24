using System;
using System.Collections.Generic;
using System.Linq;

namespace NDarrayLib
{
    public enum Backend { Default, Mkl }

    public static class Utils
    {

        public static string Glue<T>(this IEnumerable<T> ts, string sep = " ", string fmt = "{0}") => string.Join(sep, ts.Select(t => string.Format(fmt, t)));

        public static Backend Backend = Backend.Default;

        public static Random Random = new Random();

        public static int ArrMul(int[] shape, int start = 0)
        {
            int a = 1;
            for (int i = start; i < shape.Length; ++i)
                a *= shape[i];

            return a;
        }

        public static int[] Shape2Strides(int[] shape)
        {
            int[] strides = new int[shape.Length];
            int p = 1;
            for (int k = shape.Length - 1; k >= 0; --k)
            {
                strides[k] = p;
                p *= shape[k];
            }

            return strides;
        }

        public static int[] PrepareReshape(int[] baseShape, int[] shape) => PrepareReshape(ArrMul(baseShape), shape);

        public static int[] PrepareReshape(int dim0, int[] shape)
        {
            int mone = shape.Count(i => i == -1);
            if (mone > 1)
                throw new ArgumentException("can only specify one unknown dimension");

            if (mone == 1)
            {
                int idx = shape.ToList().FindIndex(i => i == -1);
                shape[idx] = 1;
                var dim2 = ArrMul(shape);
                shape[idx] = dim0 / dim2;
            }

            var dim1 = ArrMul(shape);

            if (dim0 != dim1)
                throw new ArgumentException($"cannot reshape array of size {dim0} into shape ({shape.Glue()})");

            return shape;
        }

        public static int[] PrepareTranspose(int rank)
        {
            int[] table = new int[rank];
            for (int i = 0; i < rank; ++i)
                table[i] = rank - i - 1;
            return table;
        }

        public static int[] DoTranspose(int[] arr, int[] table)
        {
            int[] r = new int[arr.Length];
            for (int i = 0; i < arr.Length; ++i)
                r[i] = arr[table[i]];

            return r;
        }

        public static (int[], int[], int[]) BroadCastShapes(int[] shape0, int[] shape1)
        {
            int sLength0 = shape0.Length;
            int sLength1 = shape1.Length;
            int bcLength = Math.Max(sLength0, sLength1);

            int[] bcshape = new int[bcLength];
            int[] nshape0 = new int[bcLength];
            int[] nshape1 = new int[bcLength];
            for (int k = bcLength - 1, i = sLength0 - 1, j = sLength1 - 1; k >= 0; --k, --i, --j)
            {
                int idx0 = i < 0 ? 1 : shape0[i];
                int idx1 = j < 0 ? 1 : shape1[j];
                if (idx0 != idx1 && idx0 != 1 && idx1 != 1)
                    throw new ArgumentException($"cannot broadcast ({shape0.Glue()}) with ({shape1.Glue()})");

                bcshape[k] = Math.Max(idx0, idx1);
                nshape0[k] = idx0;
                nshape1[k] = idx1;
            }

            return (nshape0, nshape1, bcshape);
        }

        public static int[] BroadCastLeftShapes(int[] shape0, int[] shape1)
        {
            int sLength0 = shape0.Length;
            int sLength1 = shape1.Length;
            if (sLength0 < sLength1) throw new ArgumentException($"cannot broadcast left ({shape0.Glue()}) with ({shape1.Glue()})");

            int[] bcshape = new int[sLength0];
            for (int i = sLength0 - 1, j = sLength1 - 1; i >= 0; --i, --j)
            {
                int idx0 = shape0[i];
                int idx1 = j < 0 ? 1 : shape1[j];
                if (idx0 != idx1 && idx1 != 1)
                    throw new ArgumentException($"cannot broadcast ({shape0.Glue()}) with ({shape1.Glue()})");

                bcshape[i] = idx1;
            }

            return bcshape;
        }

        public static int[] PrepareAxisOps(int[] shape, int axis, bool keepdims)
        {
            int[] nshape = !keepdims ? new int[shape.Length - 1] : shape.ToArray();
            if (!keepdims)
            {
                for (int i = 0, j = 0; i < shape.Length; ++i)
                {
                    if (i == axis) continue;
                    nshape[j++] = shape[i];
                }
            }
            else
                nshape[axis] = 1;

            return nshape;
        }

        public static int[] PrepareDot(int[] shape0, int[] shape1)
        {
            int lpiv = shape0.Last();
            int rpiv = shape1.Length == 1 ? shape1.Last() : shape1[shape1.Length - 2];

            if (lpiv != rpiv)
                throw new ArgumentException($"cannot multiply ({shape0.Glue()}) and ({shape1.Glue()})");

            int ndim = shape0.Length == 1 && shape1.Length == 1 ? 1 : shape0.Length + shape1.Length - 2;
            int[] nshape = new int[ndim];
            nshape[0] = 1;

            int idx = 0;
            for (int k = 0; k < shape0.Length - 1; ++k)
                nshape[idx++] = shape0[k];

            if (shape1.Length != 1)
            {
                for (int k = 0; k < shape1.Length; ++k)
                {
                    if (k == shape1.Length - 2) continue;
                    nshape[idx++] = shape1[k];
                }
            }

            return nshape;
        }

        public static (int[], int[]) PrepareSplit(int[] shape, int axis, int idx)
        {
            if (axis < 0 || axis >= shape.Length)
                throw new ArgumentException("bad Split axis");

            int dim = shape[axis];
            if (idx < 0 || idx >= dim)
                throw new ArgumentException("bad Split index");

            int[] shape0 = shape.ToArray();
            int[] shape1 = shape.ToArray();
            shape0[axis] = idx;
            shape1[axis] -= idx;

            return (shape0, shape1);
        }

        public static int[] PrepareConcatenate(int[] shape0, int[] shape1, int axis)
        {
            if (shape0.Length != shape1.Length)
                throw new ArgumentException("bad Stack shape dimension");

            if (axis < 0 || axis >= shape0.Length)
                throw new ArgumentException("bad Stack axis");

            for (int k = 0; k < shape0.Length; ++k)
            {
                if (k == axis) continue;
                if (shape0[k] != shape1[k])
                    throw new ArgumentException($"cannot Stack ({shape0.Glue()}) and ({shape1.Glue()})");
            }

            int[] nshape = shape0.ToArray();
            nshape[axis] += shape1[axis];
            return nshape;
        }

        public static (int[], (int, int)[]) PreparePad(int[] shape, (int, int)[] pads)
        {
            if (pads.Length == 1)
                pads = Enumerable.Repeat(pads[0], shape.Length).ToArray();

            if (pads.Length != shape.Length)
                throw new ArgumentException($"Shape and pads must have the same length");

            var nshape = new int[shape.Length];
            for (int i = 0; i < shape.Length; ++i)
            {
                (int padL, int padR) = pads[i];
                nshape[i] = shape[i] + padL + padR;
            }
            return (nshape, pads);
        }

        public static (int[], int[]) PrepareTile(int[] shape, int[] reps)
        {
            if (reps.Any(i => i <= 0))
                throw new ArgumentException("Repetition must be greater than 0");

            var nreps = Enumerable.Repeat(1, Math.Max(shape.Length, reps.Length)).ToArray();
            var nshape = nreps.ToArray();
            for (int i = reps.Length - 1, j = shape.Length - 1; i >= 0 || j >= 0; --i, --j)
            {
                if (i >= 0) nreps[Math.Max(i, j)] = reps[i];
                if (j >= 0) nshape[Math.Max(i, j)] = shape[j];
            }

            return (nshape, nreps);
        }

    }
}
