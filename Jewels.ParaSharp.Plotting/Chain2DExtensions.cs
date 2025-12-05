using ScottPlot;

namespace Jewels.ParaSharp.Plotting;

public static class Chain2DExtensions
{
    public static Plot Plot(this Chain2D chain, int num = 100, string? path = null)
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

    public static Plot Plot(this Chain2D chain, float[] targetInputs, float[] targetOutputs, bool generate = false, int num = 100, string? path = null)
    {
        var domainEnd = chain.DomainLength();
        float[] inputs;
        if (generate)
        {
            inputs = new float[num];
            for (int i = 0; i < num; i++)
                inputs[i] = i * domainEnd / (num - 1);
        }
        else inputs = targetInputs;
        
        float[] outputs = generate ? new float[num] : new float[targetOutputs.Length];
        for (int i = 0; i < (generate ? num : targetOutputs.Length); i++) outputs[i] = chain.Evaluate(inputs[i]);
        
        Plot plot = new();
        plot.Add.ScatterLine(inputs, outputs);
        plot.Add.ScatterLine(targetInputs, targetOutputs);
        if (path is not null) plot.SavePng(path, 800, 600);
        return plot;
    }
}