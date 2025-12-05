

namespace Jewels.ParaSharp;

public class Chain2D
{
    public List<Paravector2D> Paravectors { get; } = [];

    public Chain2D(params Paravector2D[] paravectors) => Paravectors.AddRange(paravectors);
    public Chain2D(IEnumerable<Paravector2D> paravectors) => Paravectors.AddRange(paravectors);
    public Chain2D() { }

    public float DomainLength()
    {
        List<float> hs = [0f];
        for (int i = 0; i < Paravectors.Count; i++)
            hs.Add(hs[i] + Paravectors[i].Alpha.Value * MathF.Cos(Paravectors[i].Theta.Value));
        return hs[^1];
    }
    public Scalar DomainLengthScalar()
    {
        Scalar h = new Scalar(0f);
        for (int i = 0; i < Paravectors.Count; i++)
            h = h + Paravectors[i].Alpha * Scalar.Cosine(Paravectors[i].Theta);
        return h;
    }

    public Paravector2D GetRelevantParavector(float x, float h = 0)
    {
        float current = h;

        foreach (Paravector2D upsilon in Paravectors)
        {
            float nextX = current + upsilon.Alpha.Value * MathF.Cos(upsilon.Theta.Value);
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
            float nextX = current + upsilon.Alpha.Value * MathF.Cos(upsilon.Theta.Value);
            boundaries.Add((current, nextX, i));
            current = nextX;
        }
        return boundaries;
    }

    public Paravector2D GetRelevantParavector(float x, List<(float start, float end, int index)> boundaries)
    {
        foreach (var (start, end, index) in boundaries)
            if (x >= start && x < end) return Paravectors[index];
        return Paravectors[^1];
    }
    
    
    public float Evaluate(float x, float h = 0f, float k = 0f)
    {
        float currentH = h;
        float currentK = k;
    
        foreach (var upsilon in Paravectors)
        {
            float nextH = currentH + upsilon.Alpha.Value * MathF.Cos(upsilon.Theta.Value);
        
            if (x >= currentH && x < nextH)
                return upsilon.GlobalX(x, currentH, currentK);
        
            currentK = upsilon.GlobalX(nextH, currentH, currentK);
            currentH = nextH;
        }
    
        return Paravectors[^1].GlobalX(x, currentH, currentK);
    }

    public Scalar Evaluate(Scalar x, Scalar? h = null, Scalar? k = null)
    {
        Scalar currentH = h ?? new Scalar(0f);
        Scalar currentK = k ?? new Scalar(0f);

        foreach (var upsilon in Paravectors)
        {
            Scalar nextH = currentH + upsilon.Alpha * Scalar.Cosine(upsilon.Theta);
        
            if (x.Value >= currentH.Value && x.Value < nextH.Value)
                return upsilon.GlobalX(x, currentH, currentK);
        
            currentK = upsilon.GlobalX(nextH, currentH, currentK);
            currentH = nextH;
        }

        return Paravectors[^1].GlobalX(x, currentH, currentK);
    }

    public static Chain2D InductiveDescent(
        int total, float lr, int epochs,
        float[] inputs, float[] targets,
        Func<Scalar, float, Scalar> loss, 
        float lossEpsilon = Scalar.BiggerEpsilon)
    {
        List<Paravector2D> paravectors = [];
        var (currentEndX, currentEndY) = (inputs[^1], targets[^1]);

        for (int i = total - 1; i >= 0; i--)
        {
            var endIndex = inputs.Length - 1 - (total - 1 - i) * inputs.Length / total;
            var startIndex = i == 0 ? 0 : inputs.Length - 1 - (total - i) * inputs.Length / total;
            
            var upsilon = Paravector2D.InductiveFit(
                inputs[startIndex..endIndex], targets[startIndex..endIndex],
                currentEndX, currentEndY, lr, epochs, loss, lossEpsilon);
            paravectors.Insert(0, upsilon);
            currentEndX = inputs[startIndex];
            currentEndY = targets[startIndex];
        }
        
        return new Chain2D(paravectors);
    }

    public static Chain2D InductiveExploration(
        int total, float lr, int epochs, 
        float[] inputs, float[] targets,
        Func<Scalar, float, Scalar> loss, 
        Func<float, float, float>? refinementLoss = null, 
        int refinementEpochs = 10, 
        float explorationRate = 0.4f,
        Action<int, float>? progressAction = null)
    {
        Chain2D bestChain = new([]);
        float bestLoss = float.MaxValue;

        for (int refinementEpoch = 0; refinementEpoch < refinementEpochs; refinementEpoch++)
        {
            float noisyLr = lr * (1f + (Random.Shared.NextSingle() - 0.5f) * explorationRate);
            int noisyEpochs = (int)(epochs * (1f + (Random.Shared.NextSingle() - 0.5) * explorationRate));
            var candidate = InductiveDescent(total, noisyLr, noisyEpochs, inputs, targets, loss);

            float totalLoss = 0f;
            for (int i = 0; i < inputs.Length; i++)
            {
                float predicted = candidate.Evaluate(inputs[i]);
                float error = refinementLoss?.Invoke(predicted, targets[i]) ?? (predicted - targets[i]) * (predicted - targets[i]);
                totalLoss += error;
            }

            if (!(totalLoss < bestLoss)) continue;
            bestChain = candidate;
            bestLoss = totalLoss;
            progressAction?.Invoke(refinementEpoch, totalLoss);
        }
        
        return bestChain;
    }
}