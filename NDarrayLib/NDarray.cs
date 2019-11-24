using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NDarrayLib
{
    public class NDarray<U>
    {

        public static Operations<U> OpsT;

        static NDarray()
        {
            if (typeof(U) == typeof(int))
                OpsT = new OpsInt() as Operations<U>;
            else if (typeof(U) == typeof(float))
                OpsT = new OpsFloat() as Operations<U>;
            else if (typeof(U) == typeof(double))
                OpsT = new OpsDouble() as Operations<U>;
            else
                throw new ArgumentException($"{typeof(U).Name} is not supported. Only int, float or double");
        }

        public int[] Shape { get; protected set; }
        public int Count { get; protected set; }

        public (NDarray<U>, NDarray<U>) Split(int axis, int idx) => ND.Split(this, axis, idx);

        public U[] Data;

        public NDarray(params int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[] { 1 };

            Shape = shape.ToArray();
            Count = Utils.ArrMul(Shape);

            Data = new U[Count];
        }

        public NDarray(U[] data, params int[] shape)
        {
            Shape = shape.ToArray();
            Count = Utils.ArrMul(Shape);
            Data = data;
            if (Data.Length != Count)
                throw new Exception();
        }

        public NDarray(U[][] data)
        {
            Data = data.SelectMany(i => i).ToArray();
            Count = Data.Length;
            Shape = new int[] { data.Length, Count / data.Length };
        }

        public NDarray(NDarray<U> nd)
        {
            Shape = nd.Shape.ToArray();
            Count = nd.Count;
            Data = nd.Data.ToArray();
        }

        public NDarray<U> this[int k]
        {
            get
            {
                var s0 = Shape[0];
                if (k < 0 || k >= s0) throw new Exception();
                var count = Count / s0;
                var data = Data.Skip(k * count).Take(count).ToArray();
                var shape = Shape.Skip(1).ToArray();
                return new NDarray<U>(data, shape);
            }
        }

        public NDarray<U> Fill(U v0)
        {
            for (int k = 0; k < Count; ++k) Data[k] = v0;
            return this;
        }

        public NDarray<V> Cast<V>()
        {
            var data = Data.Select(a => NDarray<V>.OpsT.Cast(a)).ToArray();
            return new NDarray<V>(data, Shape);
        }

        public NDarray<U> T => ND.Transpose(this);

        public U[] GetAtIndex(int idx)
        {
            var s = Utils.ArrMul(Shape, 1);
            return Data.Skip(idx * s).Take(s).ToArray();
        }

        public NDarray<U> ReshapeInplace(params int[] shape)
        {
            Shape = Utils.PrepareReshape(Count, shape);
            return this;
        }

        public void ApplyFuncInplace(Func<U, U> func)
        {
            for (int idx = 0; idx < Count; ++idx)
                Data[idx] = func(Data[idx]);
        }

        public void ApplyFuncInplace(Func<int, U, U> func)
        {
            for (int idx = 0; idx < Count; ++idx)
                Data[idx] = func(idx, Data[idx]);
        }

        public NDarray<U> ApplyFunc(Func<U, U> func)
        {
            NDarray<U> nd = new NDarray<U>(Shape);
            for (int idx = 0; idx < Count; ++idx)
                nd.Data[idx] = func(Data[idx]);

            return nd;
        }

        public NDarray<U> ApplyFunc(Func<int, U, U> func)
        {
            NDarray<U> nd = new NDarray<U>(Shape);
            for (int idx = 0; idx < Count; ++idx)
                nd.Data[idx] = func(idx, Data[idx]);

            return nd;
        }

        private string PrettyDisplay(string fmt, int depth = 0)
        {
            if (Shape.Length == 1)
                return $"[{Data.Glue(" ", fmt)}]";

            StringBuilder sb = new StringBuilder();
            string space = Enumerable.Repeat("", depth + 2).Glue();

            for (int k = 0; k < Shape[0]; ++k)
            {
                string b = k == 0 ? "[" : space;
                string e = k == Shape[0] - 1 ? "]" : space;

                if (k != Shape[0] - 1)
                    sb.AppendLine(b + this[k].PrettyDisplay(fmt, depth + 1) + e);
                else
                    sb.Append(b + this[k].PrettyDisplay(fmt, depth + 1) + e);
            }

            return sb.ToString();
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            var reshape = $"reshape({Shape.Glue(",")})";
            var ndarray = $"np.array([{Data.Glue(",")}], dtype={OpsT.dtype})";
            sb.AppendLine($"{ndarray}.{reshape}");

            int mx = Data.Select(i => i.ToString()).Max(i => i.Length);
            if (mx > 5)
                mx = Data.Select(i => $"{i:F6}").Max(i => i.Length);

            string fmt = mx < 5 ? $"{{0, {mx}}}" : $"{{0, {mx}:F6}}";
            sb.Append(PrettyDisplay(fmt));
            sb.AppendLine();

            return sb.ToString();
        }

    }
}
