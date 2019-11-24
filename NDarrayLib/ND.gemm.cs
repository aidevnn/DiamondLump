using System;
namespace NDarrayLib
{
    public static partial class ND
    {
        static NDarray<float> GemmABfloat(NDarray<float> a, NDarray<float> b, NDarray<float> c = null)
        {
            if (Utils.Backend == Backend.Default)
                return NDsharp.GemmAB(a, b, c);

            return NDmkl.GemmAB(a, b, c);
        }

        static NDarray<float> GemmTABfloat(NDarray<float> a, NDarray<float> b, NDarray<float> c = null)
        {
            if (Utils.Backend == Backend.Default)
                return NDsharp.GemmTAB(a, b, c);

            return NDmkl.GemmTAB(a, b, c);
        }

        static NDarray<float> GemmATBfloat(NDarray<float> a, NDarray<float> b, NDarray<float> c = null)
        {
            if (Utils.Backend == Backend.Default)
                return NDsharp.GemmATB(a, b, c);

            return NDmkl.GemmATB(a, b, c);
        }

        static NDarray<float> GemmTATBfloat(NDarray<float> a, NDarray<float> b, NDarray<float> c = null)
        {
            if (Utils.Backend == Backend.Default)
                return NDsharp.GemmTATB(a, b, c);

            return NDmkl.GemmTATB(a, b, c);
        }

        static NDarray<double> GemmABdouble(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (Utils.Backend == Backend.Default)
                return NDsharp.GemmAB(a, b, c);

            return NDmkl.GemmAB(a, b, c);
        }

        static NDarray<double> GemmTABdouble(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (Utils.Backend == Backend.Default)
                return NDsharp.GemmTAB(a, b, c);

            return NDmkl.GemmTAB(a, b, c);
        }

        static NDarray<double> GemmATBdouble(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (Utils.Backend == Backend.Default)
                return NDsharp.GemmATB(a, b, c);

            return NDmkl.GemmATB(a, b, c);
        }

        static NDarray<double> GemmTATBdouble(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (Utils.Backend == Backend.Default)
                return NDsharp.GemmTATB(a, b, c);

            return NDmkl.GemmTATB(a, b, c);
        }

        public static NDarray<U> GemmAB<U>(NDarray<U> a, NDarray<U> b, NDarray<U> c = null)
        {
            if (typeof(U) == typeof(float))
            {
                var af = a as NDarray<float>;
                var bf = b as NDarray<float>;
                var cf = c as NDarray<float>;
                return (GemmABfloat(af, bf, cf)) as NDarray<U>;
            }
            if (typeof(U) == typeof(double))
            {
                var ad = a as NDarray<double>;
                var bd = b as NDarray<double>;
                var cd = c as NDarray<double>;
                return (GemmABdouble(ad, bd, cd)) as NDarray<U>;
            }

            throw new ArgumentException($"{typeof(U).Name} is not supported. Only float or double");
        }

        public static NDarray<U> GemmATB<U>(NDarray<U> a, NDarray<U> b, NDarray<U> c = null)
        {
            if (typeof(U) == typeof(float))
            {
                var af = a as NDarray<float>;
                var bf = b as NDarray<float>;
                var cf = c as NDarray<float>;
                return (GemmATBfloat(af, bf, cf)) as NDarray<U>;
            }
            if (typeof(U) == typeof(double))
            {
                var ad = a as NDarray<double>;
                var bd = b as NDarray<double>;
                var cd = c as NDarray<double>;
                return (GemmATBdouble(ad, bd, cd)) as NDarray<U>;
            }

            throw new ArgumentException($"{typeof(U).Name} is not supported. Only float or double");
        }

        public static NDarray<U> GemmTAB<U>(NDarray<U> a, NDarray<U> b, NDarray<U> c = null)
        {
            if (typeof(U) == typeof(float))
            {
                var af = a as NDarray<float>;
                var bf = b as NDarray<float>;
                var cf = c as NDarray<float>;
                return (GemmTABfloat(af, bf, cf)) as NDarray<U>;
            }
            if (typeof(U) == typeof(double))
            {
                var ad = a as NDarray<double>;
                var bd = b as NDarray<double>;
                var cd = c as NDarray<double>;
                return (GemmTABdouble(ad, bd, cd)) as NDarray<U>;
            }

            throw new ArgumentException($"{typeof(U).Name} is not supported. Only float or double");
        }

        public static NDarray<U> GemmTATB<U>(NDarray<U> a, NDarray<U> b, NDarray<U> c = null)
        {
            if (typeof(U) == typeof(float))
            {
                var af = a as NDarray<float>;
                var bf = b as NDarray<float>;
                var cf = c as NDarray<float>;
                return (GemmTATBfloat(af, bf, cf)) as NDarray<U>;
            }
            if (typeof(U) == typeof(double))
            {
                var ad = a as NDarray<double>;
                var bd = b as NDarray<double>;
                var cd = c as NDarray<double>;
                return (GemmTATBdouble(ad, bd, cd)) as NDarray<U>;
            }

            throw new ArgumentException($"{typeof(U).Name} is not supported. Only float or double");
        }

    }
}
