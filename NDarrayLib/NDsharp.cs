using System;
namespace NDarrayLib
{
    public static class NDsharp
    {
        #region Gemm float
        public static NDarray<float> GemmAB(NDarray<float> a, NDarray<float> b, NDarray<float> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[0];
            int p = a.Shape[1];
            int n = b.Shape[1];
            if (p != b.Shape[0]) throw new Exception();

            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<float>(m, n);

            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < p; ++k)
                        sum += a.Data[i * p + k] * b.Data[k * n + j];

                    c.Data[i * n + j] += sum;
                }
            }

            return c;
        }

        public static NDarray<float> GemmTAB(NDarray<float> a, NDarray<float> b, NDarray<float> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[1];
            int p = a.Shape[0];
            int n = b.Shape[1];
            if (p != b.Shape[0]) throw new Exception();
            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<float>(m, n);


            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < p; ++k)
                        sum += a.Data[k * m + i] * b.Data[k * n + j];

                    c.Data[i * n + j] += sum;
                }
            }

            return c;
        }

        public static NDarray<float> GemmATB(NDarray<float> a, NDarray<float> b, NDarray<float> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[0];
            int p = a.Shape[1];
            int n = b.Shape[0];
            if (p != b.Shape[1]) throw new Exception();
            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<float>(m, n);


            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < p; ++k)
                        sum += a.Data[i * p + k] * b.Data[j * p + k];

                    c.Data[i * n + j] += sum;
                }
            }

            return c;
        }

        public static NDarray<float> GemmTATB(NDarray<float> a, NDarray<float> b, NDarray<float> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[1];
            int p = a.Shape[0];
            int n = b.Shape[0];
            if (p != b.Shape[1]) throw new Exception();
            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<float>(m, n);

            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < p; ++k)
                        sum += a.Data[k * m + i] * b.Data[j * p + k];

                    c.Data[i * n + j] += sum;
                }
            }

            return c;
        }
        #endregion

        #region Gemm double
        public static NDarray<double> GemmAB(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[0];
            int p = a.Shape[1];
            int n = b.Shape[1];
            if (p != b.Shape[0]) throw new Exception();

            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<double>(m, n);

            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    double sum = 0.0;
                    for (int k = 0; k < p; ++k)
                        sum += a.Data[i * p + k] * b.Data[k * n + j];

                    c.Data[i * n + j] += sum;
                }
            }

            return c;
        }

        public static NDarray<double> GemmTAB(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[1];
            int p = a.Shape[0];
            int n = b.Shape[1];
            if (p != b.Shape[0]) throw new Exception();
            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<double>(m, n);


            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    double sum = 0.0;
                    for (int k = 0; k < p; ++k)
                        sum += a.Data[k * m + i] * b.Data[k * n + j];

                    c.Data[i * n + j] += sum;
                }
            }

            return c;
        }

        public static NDarray<double> GemmATB(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[0];
            int p = a.Shape[1];
            int n = b.Shape[0];
            if (p != b.Shape[1]) throw new Exception();
            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<double>(m, n);


            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    double sum = 0.0;
                    for (int k = 0; k < p; ++k)
                        sum += a.Data[i * p + k] * b.Data[j * p + k];

                    c.Data[i * n + j] += sum;
                }
            }

            return c;
        }

        public static NDarray<double> GemmTATB(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[1];
            int p = a.Shape[0];
            int n = b.Shape[0];
            if (p != b.Shape[1]) throw new Exception();
            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<double>(m, n);

            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    double sum = 0.0;
                    for (int k = 0; k < p; ++k)
                        sum += a.Data[k * m + i] * b.Data[j * p + k];

                    c.Data[i * n + j] += sum;
                }
            }

            return c;
        }
        #endregion

    }
}
