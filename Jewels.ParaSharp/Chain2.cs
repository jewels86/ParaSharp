using Jewels.Lazulite;
using Jewels.Opal;
using static Jewels.ParaSharp.Paravector;
using static Jewels.Opal.Operations;

namespace Jewels.ParaSharp;

public class Chain2
{
    public List<Paravector2> Paravectors { get; } = [];

    public Chain2(params Paravector2[] paravectors) => Paravectors.AddRange(paravectors);
    public Chain2(IEnumerable<Paravector2> paravectors) => Paravectors.AddRange(paravectors);
    public Chain2() { }

    public Tensor<float> DomainLength()
    {
        List<Tensor<float>> hs = [NewScalar(0)];
        for (int i = 0; i < Paravectors.Count; i++)
            hs.Add(hs[i] + Paravectors[i].Alpha * Cosine(Paravectors[i].Theta));
        return hs[^1];
    }

    public Paravector2 GetRelevantParavector(float x, float h = 0)
    {
        float current = h;

        foreach (Paravector2 upsilon in Paravectors)
        {
            float nextX = current + upsilon.Alpha.Value.ToHost() * MathF.Cos(upsilon.Theta.Value.ToHost());
            if (x >= current && x < nextX) return upsilon;
            current = nextX;
        }
        return Paravectors[^1];
    }

    public List<(float start, float end, int index)> SetupBoundaries(float h)
    {
        float current = h;
        List<(float start, float end, int index)> boundaries = [];
        for (int i = 0; i < Paravectors.Count; i++)
        {
            var upsilon = Paravectors[i];
            float nextX = current + upsilon.Alpha.Value.ToHost() * MathF.Cos(upsilon.Theta.Value.ToHost());
            boundaries.Add((current, nextX, i));
            current = nextX;
        }
        return boundaries;
    }

    public Paravector2 GetRelevantParavector(float x, List<(float start, float end, int index)> boundaries)
    {
        foreach (var (start, end, index) in boundaries)
            if (x >= start && x < end) return Paravectors[index];
        return Paravectors[^1];
    }
    
    public float Evaluate(float x, float h = 0f, float k = 0f)
    {
        if (x > DomainLength().Value.ToHost()) throw new Exception("x is out of domain");
        var boundaries = SetupBoundaries(h);
        var upsilon = GetRelevantParavector(x, boundaries);
        return upsilon.GlobalX(x, h, k);
    }
    public Tensor<float> Evaluate(Tensor<float> x, Tensor<float> h, Tensor<float> k)
    {
        var (xHost, hHost) = (x.Value.ToHost(), h.Value.ToHost());
        if (xHost > DomainLength().Value.ToHost()) throw new Exception("x is out of domain");
        var boundaries = SetupBoundaries(hHost);
        var upsilon = GetRelevantParavector(xHost, boundaries);
        return upsilon.GlobalX(x, h, k);
    }

    public static Chain2 Fit(
        int total, 
        float[] inputs, 
        float[] targets, 
        float lr, 
        int maxEpochs, 
        Func<Tensor<float>, Value<float>, Tensor<float>> loss, 
        float epsilon = Single.Epsilon,
        Action<float, int>? action = null)
    {
        Tensor<float>[] inputTensors = inputs.Select(NewScalar).ToArray();
        Value<float>[] targetValues = targets.Select(Paravector.NewValue).ToArray();
        Value<float> lrValue = Paravector.NewValue(lr);
        
        List<Paravector2> paravectors = [];
        List<(float x, float y)> points = [(inputs[0], targets[0])];
        int interval = inputs.Length / total;
        for (int i = 1; i <= total; i++)
        {
            int idx = (i * interval) % inputs.Length;
            var (x, y) = (inputs[idx], targets[idx]);
            var previousPoint = points[^1];
            var upsilon = Paravector2.FromVectorDifference(previousPoint.x, previousPoint.y, x, y);
            points.Add((x, y));
            paravectors.Add(upsilon);
        }
        Chain2 chain = new(paravectors);

        for (int epoch = 0; epoch < maxEpochs; epoch++)
        {
            var boundaries = chain.SetupBoundaries(0);
            float totalLoss = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                var (input, inputTensor, targetValue) = (inputs[i], inputTensors[i], targetValues[i]);
                var upsilon = chain.GetRelevantParavector(input, boundaries);
                var outputTensor = upsilon.GlobalX(inputTensor);
                using var lossTensor = loss(outputTensor, targetValue);
                lossTensor.Backward(Paravector.NewValue(1f));
                totalLoss += lossTensor.Value.ToHost();
            }
            
            foreach (var upsilon in chain.Paravectors) upsilon.Update(lrValue);
            
            action?.Invoke(totalLoss, epoch);
            if (totalLoss < epsilon) return chain;
        }
        
        return chain;
    }
}