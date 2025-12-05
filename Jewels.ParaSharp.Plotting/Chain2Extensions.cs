using ScottPlot;

namespace Jewels.ParaSharp.Plotting;

public static class Chain2Extensions
{
    public static Plot Plot(this Chain2 chain, int num = 100, string? path = null)
    {
        var domainEnd = chain.DomainLength();
        float[] inputs = new float[num];
        for (int i = 0; i < num; i++) inputs[i] = i * domainEnd / (num - 1);
        float[] outputs = new float[num];
        for (int i = 0; i < num; i++)
        {
            var input = inputs[i];
            var output = chain.Evaluate(input);
            outputs[i] = output;
        }

        Plot plot = new();
        plot.Add.ScatterLine(inputs, outputs);
        if (path is not null) plot.SavePng(path, 800, 600);
        return plot;
    }
}