
#pragma warning disable IDE1006 // Naming Styles

using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;

namespace NDarrayLib
{
    public static class NDmkl
    {
        const int TRANSPOSE_YES = 112;
        const int TRANSPOSE_NO = 111;
        const int CMAT = 101;
        const int FMAT = 100;

        #region Gemm float

        [DllImport("mkl_rt", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void cblas_sgemm(int Order, int TransA, int TransB,
            int M, int N, int K,
            float alpha, float[] A, int lda, float[] B, int ldb,
            float beta, float[] C, int ldc);

        public static NDarray<float> GemmAB(NDarray<float> a, NDarray<float> b, NDarray<float> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[0];
            int k = a.Shape[1];
            int n = b.Shape[1];
            if (k != b.Shape[0])
                throw new Exception();

            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<float>(m, n);

            cblas_sgemm(CMAT, TRANSPOSE_NO, TRANSPOSE_NO, m, n, k, 1f, a.Data, k, b.Data, n, 0f, c.Data, n);
            return c;
        }

        public static NDarray<float> GemmTAB(NDarray<float> a, NDarray<float> b, NDarray<float> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[1]; // TRANSPOSE
            int k = a.Shape[0]; // TRANSPOSE
            int n = b.Shape[1];
            if (k != b.Shape[0])
                throw new Exception();

            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<float>(m, n);

            cblas_sgemm(CMAT, TRANSPOSE_YES, TRANSPOSE_NO, m, n, k, 1f, a.Data, m, b.Data, n, 0f, c.Data, n);
            return c;
        }

        public static NDarray<float> GemmATB(NDarray<float> a, NDarray<float> b, NDarray<float> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[0];
            int k = a.Shape[1];
            int n = b.Shape[0]; // TRANSPOSE
            if (k != b.Shape[1])
                throw new Exception();

            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<float>(m, n);

            cblas_sgemm(CMAT, TRANSPOSE_NO, TRANSPOSE_YES, m, n, k, 1f, a.Data, k, b.Data, k, 0f, c.Data, n);
            return c;
        }

        public static NDarray<float> GemmTATB(NDarray<float> a, NDarray<float> b, NDarray<float> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[1]; // TRANSPOSE
            int k = a.Shape[0]; // TRANSPOSE
            int n = b.Shape[0]; // TRANSPOSE
            if (k != b.Shape[1])
                throw new Exception();

            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<float>(m, n);

            cblas_sgemm(CMAT, TRANSPOSE_YES, TRANSPOSE_YES, m, n, k, 1f, a.Data, m, b.Data, k, 0f, c.Data, n);
            return c;
        }
        #endregion

        #region Gemm double
        [DllImport("mkl_rt", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void cblas_dgemm(int Order, int TransA, int TransB,
            int M, int N, int K,
            double alpha, double[] A, int lda, double[] B, int ldb,
            double beta, double[] C, int ldc);

        public static NDarray<double> GemmAB(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[0];
            int k = a.Shape[1];
            int n = b.Shape[1];
            if (k != b.Shape[0])
                throw new Exception();

            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<double>(m, n);

            cblas_dgemm(CMAT, TRANSPOSE_NO, TRANSPOSE_NO, m, n, k, 1.0, a.Data, k, b.Data, n, 0.0, c.Data, n);
            return c;
        }

        public static NDarray<double> GemmTAB(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[1]; // TRANSPOSE
            int k = a.Shape[0]; // TRANSPOSE
            int n = b.Shape[1];
            if (k != b.Shape[0])
                throw new Exception();

            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<double>(m, n);

            cblas_dgemm(CMAT, TRANSPOSE_YES, TRANSPOSE_NO, m, n, k, 1.0, a.Data, m, b.Data, n, 0.0, c.Data, n);
            return c;
        }

        public static NDarray<double> GemmATB(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[0];
            int k = a.Shape[1];
            int n = b.Shape[0]; // TRANSPOSE
            if (k != b.Shape[1])
                throw new Exception();

            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<double>(m, n);

            cblas_dgemm(CMAT, TRANSPOSE_NO, TRANSPOSE_YES, m, n, k, 1.0, a.Data, k, b.Data, k, 0.0, c.Data, n);
            return c;
        }

        public static NDarray<double> GemmTATB(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2) throw new Exception();

            int m = a.Shape[1]; // TRANSPOSE
            int k = a.Shape[0]; // TRANSPOSE
            int n = b.Shape[0]; // TRANSPOSE
            if (k != b.Shape[1])
                throw new Exception();

            if (c != null)
            {
                if (c.Shape.Length != 2 || c.Shape[0] != m || c.Shape[1] != n)
                    throw new Exception();
            }
            else
                c = new NDarray<double>(m, n);

            cblas_dgemm(CMAT, TRANSPOSE_YES, TRANSPOSE_YES, m, n, k, 1.0, a.Data, m, b.Data, k, 0.0, c.Data, n);
            return c;
        }
        #endregion

    }
}
