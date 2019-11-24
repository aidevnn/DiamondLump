using System;
using System.Linq;
using NDarrayLib;
namespace DiamondLump.Losses
{
    public interface IAccuracy<U>
    {
        string Name { get; }
        double Func(NDarray<U> y, NDarray<U> p);
    }

    public class RoundAccuracy<U> : IAccuracy<U>
    {
        public string Name => "RoundAccuracy";

        public double Func(NDarray<U> y, NDarray<U> p)
        {
            var eq1 = ND.ApplyFuncAB(ND.Round(y), ND.Round(p), NDarray<U>.OpsT.Eq);
            var eq2 = ND.ProdAxis(eq1, -1);
            return eq2.Data.Average();
        }
    }

    public class ArgmaxAccuracy<U> : IAccuracy<U>
    {
        public string Name => "ArgmaxAccuracy";

        public double Func(NDarray<U> y, NDarray<U> p)
        {
            var y0 = ND.ArgMax(y, -1);
            var p0 = ND.ArgMax(p, -1);
            NDarray<double> eq = ND.Eq(y0, p0);
            return eq.Data.Average();
        }
    }
}
