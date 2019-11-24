using System;
using System.Diagnostics;
using System.Linq;

namespace NDarrayLib
{
    class MainClass
    {
        static void BenchAxisOp(int axis, int R, int L, params int[] shape)
        {
            Utils.Random = new Random(123);

            for (int r = 0; r < R; ++r)
            {
                var x = ND.Uniform(1, 10, shape);
                var sw = Stopwatch.StartNew();
                long sum = 0;
                for (int l = 0; l < L; ++l)
                {
                    var y = ND.MinAxis(x, axis);
                    sum += y.Data.Sum();
                }

                Console.WriteLine($"BenchAxisOp Time:{sw.ElapsedMilliseconds,5} ms sum:{sum}");
            }
        }

        static void BenchArgMin(int axis, int R, int L, params int[] shape)
        {
            Utils.Random = new Random(123);

            for (int r = 0; r < R; ++r)
            {
                var x = ND.Uniform(1, 10, shape);
                var sw = Stopwatch.StartNew();
                long sum = 0;
                for (int l = 0; l < L; ++l)
                {
                    var y = ND.ArgMin(x, axis);
                    sum += y.Data.Sum();
                }

                Console.WriteLine($"BenchArgMin Time:{sw.ElapsedMilliseconds,5} ms sum:{sum}");
            }
        }

        static void BenchAXpBY(int R, int L, params int[] shape)
        {
            Utils.Random = new Random(123);

            for (int r = 0; r < R; ++r)
            {
                var x = ND.Uniform(1, 10, shape);
                var y = ND.Uniform(1, 10, shape);
                var sw = Stopwatch.StartNew();
                long sum = 0;
                for (int l = 0; l < L; ++l)
                {
                    x.AXpBY(y);
                    x.AXmBY(y);
                    //sum += x.Data.Sum();
                }

                Console.WriteLine($"BenchAXpBY     Time:{sw.ElapsedMilliseconds,5} ms sum:{sum}");
            }

            Console.WriteLine();
        }

        static void BenchAXpBYFast(int R, int L, params int[] shape)
        {
            Utils.Random = new Random(123);

            for (int r = 0; r < R; ++r)
            {
                var x = ND.Uniform(1, 10, shape);
                var y = ND.Uniform(1, 10, shape);
                var sw = Stopwatch.StartNew();
                long sum = 0;
                for (int l = 0; l < L; ++l)
                {
                    x.AXpBYFast(y);
                    x.AXmBYFast(y);
                    //sum += x.Data.Sum();
                }

                Console.WriteLine($"BenchAXpBYFast Time:{sw.ElapsedMilliseconds,5} ms sum:{sum}");
            }

            Console.WriteLine();
        }

        static void BenchApplyAfB(int R, int L, params int[] shape)
        {
            Utils.Random = new Random(123);

            for (int r = 0; r < R; ++r)
            {
                var x = ND.Uniform(1, 10, shape);
                var y = ND.Uniform(1, 10, shape);
                var sw = Stopwatch.StartNew();
                long sum = 0;
                for (int l = 0; l < L; ++l)
                {
                    x.ApplyFuncInplace((i, x0) => x0 + y.Data[i]);
                    x.ApplyFuncInplace((i, x0) => x0 - y.Data[i]);
                    //sum += x.Data.Sum();
                }

                Console.WriteLine($"BenchApplyAfB  Time:{sw.ElapsedMilliseconds,5} ms sum:{sum}");
            }

            Console.WriteLine();
        }

        static void BenchDotDefault(int R, int L, int M, int K, int N)
        {
            Utils.Random = new Random(123);

            for (int r = 0; r < R; ++r)
            {
                var x = ND.Uniform(1, 10, M, K).Cast<float>();
                var y = ND.Uniform(1, 10, K, N).Cast<float>();
                var sw = Stopwatch.StartNew();
                float sum = 0;
                for (int l = 0; l < L; ++l)
                {
                    var z = ND.Dot(x, y);
                    //sum += z.Data.Sum();
                }

                Console.WriteLine($"BenchDotDefault Time:{sw.ElapsedMilliseconds,5} ms sum:{sum}");
            }

            Console.WriteLine();
        }

        static void BenchGemmMKL(int R, int L, int M, int K, int N)
        {
            Utils.Random = new Random(123);

            for (int r = 0; r < R; ++r)
            {
                var x = ND.Uniform(1, 10, M, K).Cast<float>();
                var y = ND.Uniform(1, 10, K, N).Cast<float>();
                var sw = Stopwatch.StartNew();
                float sum = 0;
                for (int l = 0; l < L; ++l)
                {
                    var z = NDmkl.GemmAB(x, y);
                    //sum += z.Data.Sum();
                }

                Console.WriteLine($"BenchGemmMKL    Time:{sw.ElapsedMilliseconds,5} ms sum:{sum}");
            }

            Console.WriteLine();
        }

        public static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            //var x = ND.Uniform(1, 10, 3, 4);
            //Console.WriteLine(x);

            //var y0 = ND.Uniform(1, 10, 4);
            //Console.WriteLine(y0);
            //Console.WriteLine(ND.Add(x, y0));

            //Console.WriteLine(ND.SumAxis(x, 0));
            //Console.WriteLine(ND.SumAxis(x, 1));
            //Console.WriteLine(ND.SumAxis(x, 2));

            //Console.WriteLine(ND.SumAxis(x));
            //Console.WriteLine(ND.SumAxis(x, true));
            //Console.WriteLine(ND.ProdAxis(x));
            //Console.WriteLine(ND.ProdAxis(x, true));
            //Console.WriteLine(ND.MeanAxis(x));
            //Console.WriteLine(ND.MeanAxis(x, true));

            //var x = ND.Uniform(1, 10, 3, 2, 2, 4);
            //Console.WriteLine(x);
            //Console.WriteLine(ND.Transpose(x));

            //var x = ND.Uniform(1, 10, 3, 2, 4);
            //Console.WriteLine(x);

            //Console.WriteLine(ND.ArgMin(x, 0));
            //Console.WriteLine(ND.ArgMin1(x, 0));
            //Console.WriteLine(ND.ArgMin(x, 1));
            //Console.WriteLine(ND.ArgMin1(x, 1));
            //Console.WriteLine(ND.ArgMin(x, 2));
            //Console.WriteLine(ND.ArgMin1(x, 2));

            //int axis = 1, R = 15, L = 1000;
            //int[] shape = { 30, 50, 40 };
            //BenchAxisOp(axis, R, L, shape);
            //Console.WriteLine();
            //BenchArgMin(axis, R, L, shape);

            //ND.Dot(ND.Ones<int>(3), ND.Ones<int>(3));
            //ND.Dot(ND.Ones<int>(2, 3), ND.Ones<int>(3));
            //ND.Dot(ND.Ones<int>(3), ND.Ones<int>(3, 4));
            //ND.Dot(ND.Ones<int>(5, 3), ND.Ones<int>(3, 4));
            //ND.Dot(ND.Ones<int>(2, 5, 3), ND.Ones<int>(3, 4));
            //ND.Dot(ND.Ones<int>(5, 3), ND.Ones<int>(6, 3, 4));
            //ND.Dot(ND.Ones<int>(2, 5, 3), ND.Ones<int>(6, 3, 4));

            //var x = ND.Uniform(0, 10, 2, 3).Cast<float>();
            //Console.WriteLine(x);
            //var y = ND.Uniform(0, 10, 3, 2).Cast<float>();
            //Console.WriteLine(y);
            //Console.WriteLine(ND.Dot(x, y));
            //Console.WriteLine(NDsharp.GemmAB(x, y));
            //Console.WriteLine(NDmkl.GemmAB(x, y));

            //var x = ND.Uniform(0, 10, 2, 2, 4);
            //Console.WriteLine(x);
            //var y = ND.Uniform(0, 10, 2, 2, 4);
            //Console.WriteLine(y);
            //Console.WriteLine(ND.AXpBY(x, y));
            //Console.WriteLine(ND.AXmBY(x, y));
            //Console.WriteLine(x);
            //x.ApplyFuncInplace((i, x0) => x0 + y.Data[i]);
            //Console.WriteLine(x);
            //x.ApplyFuncInplace((i, x0) => x0 - y.Data[i]);
            //Console.WriteLine(x);

            //var (y1, y2) = ND.Split(x, 2, 2);
            //Console.WriteLine(y1);
            //Console.WriteLine(y2);

            //Console.WriteLine(ND.Pad(x, (1, 2)));

            //var x1 = ND.Uniform(0, 10, 4, 4).Cast<float>();
            //var x2 = ND.Uniform(0, 10, 4, 2).Cast<float>();
            //Console.WriteLine(x1);
            //Console.WriteLine(x2);
            //Console.WriteLine(ND.Dot(x1, x2));
            //Console.WriteLine(NDmkl.Dot(x1, x2));
            //Console.WriteLine(NDmkl.GemmAB(x1, x2));

            //int R = 5, L = 150, M = 50, K = 1000, N = 10;
            //BenchTranspose(R, L, M, K);
            //BenchTranspMKL(R, L, M, K);
            //BenchTranspose(R, L, K, M);
            //BenchTranspMKL(R, L, K, M);

            //BenchDotDefault(R, L, M, K, N);
            //BenchDotFloat(R, L, M, K, N);
            //BenchDotMKL(R, L, M, K, N);
            //BenchGemmMKL(R, L, M, K, N);

            //int R = 10, L = 10;
            //int[] shape = { 30, 20, 50, 40 };
            //BenchAXpBY(R, L, shape);
            //BenchAXpBYFast(R, L, shape);
            //BenchApplyAfB(R, L, shape);


            var x = ND.Uniform(0, 10, 2, 4);
            Console.WriteLine(x);
            var y = ND.Uniform(0, 10, 4, 3);
            Console.WriteLine(y);
            var z = new NDarray<int>(2, 3);
            ND.Dot(x, y, z);
            Console.WriteLine(z);
        }
    }
}
