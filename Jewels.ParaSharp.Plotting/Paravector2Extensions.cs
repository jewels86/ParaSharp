using ScottPlot;

namespace Jewels.ParaSharp.Plotting;

public static class Paravector2Extensions
{
    public static Plot Plot(this Paravector2 upsilon, int num = 100, string? path = null)
    {
        var domainEnd = upsilon.Alpha * Scalar.Cosine(upsilon.Theta);
        float[] inputs = new float[num];
        for (int i = 0; i < num; i++) inputs[i] = i * domainEnd.Value / (num - 1);
        float[] outputs = new float[num];
        for (int i = 0; i < num; i++)
        {
            var input = inputs[i];
            var output = upsilon.GlobalX(input, 0, 0);
            outputs[i] = output;
        }

        Plot plot = new();
        plot.Add.ScatterLine(inputs, outputs);
        if (path is not null) plot.SavePng(path, 800, 600);
        return plot;
    }
}