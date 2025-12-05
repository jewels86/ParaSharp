

namespace Jewels.ParaSharp;

public class Chain2
{
    public List<Paravector2> Paravectors { get; } = [];

    public Chain2(params Paravector2[] paravectors) => Paravectors.AddRange(paravectors);
    public Chain2(IEnumerable<Paravector2> paravectors) => Paravectors.AddRange(paravectors);
    public Chain2() { }

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

    public Paravector2 GetRelevantParavector(float x, float h = 0)
    {
        float current = h;

        foreach (Paravector2 upsilon in Paravectors)
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

    public Paravector2 GetRelevantParavector(float x, List<(float start, float end, int index)> boundaries)
    {
        foreach (var (start, end, index) in boundaries)
            if (x >= start && x < end) return Paravectors[index];
        return Paravectors[^1];
    }
    
    
    public float Evaluate(float x, float h = 0f, float k = 0f)
    {
        var boundaries = SetupBoundaries(h);
        var upsilon = GetRelevantParavector(x, boundaries);
        return upsilon.GlobalX(x, h, k);
    }

    public Scalar Evaluate(Scalar x, Scalar h, Scalar k)
    {
        var boundaries = SetupBoundaries(h.Value);
        var upsilon = GetRelevantParavector(x.Value, boundaries);
        return upsilon.GlobalX(x, h, k);
    }

    public static Chain2 Fit(
        int total, 
        float[] inputs, 
        float[] targets, 
        float baseLR, 
        int maxEpochs, 
        Func<Scalar, float, Scalar> loss, 
        float alphaLR = -1,
        float thetaLR = -1,
        float betaLR = -1,
        float lossEpsilon = 1e-2f,
        float lengthWeight = 0.1f,
        Action<float, int>? action = null)
    {
        if (alphaLR < 0) alphaLR = 1;
        if (thetaLR < 0) thetaLR = 1;
        if (betaLR < 0) betaLR = 3;
        Scalar[] inputTensors = inputs.Select(x => new Scalar(x)).ToArray();
        
        List<Paravector2> paravectors = [];

        for (int i = 0; i < total; i++)
        {
            int startIdx = i * inputs.Length / total;
            int endIdx = (i + 1) * inputs.Length / total;
            if (endIdx >= inputs.Length) endIdx = inputs.Length - 1;
    
            float x1 = inputs[startIdx];
            float y1 = targets[startIdx];
            float x2 = inputs[endIdx];
            float y2 = targets[endIdx];
    
            paravectors.Add(Paravector2.FromVectorDifference(x1, y1, x2, y2));
        }
        Chain2 chain = new(paravectors);
        var originalLength = chain.DomainLength();

        for (int epoch = 0; epoch < maxEpochs; epoch++)
        {
            float totalLoss = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                var outputTensor = chain.Evaluate(inputTensors[i], new Scalar(0f), new Scalar(0f));
                var lossTensor = loss(outputTensor, targets[i]);
                
                lossTensor.Backward(1f);
                totalLoss += lossTensor.Value;
                for (int j = 0; j < paravectors.Count; j++) paravectors[j].Update(baseLR, alphaLR, thetaLR, betaLR, j, paravectors.Count);
                // i think whats happening is that upsilons further from the start are more affected by changes throughout the chain
                // thats why x = 0, loss = 0, x = 0.79, loss = 0.01, x = 4.71, loss = 0.93
                // either upsilons need a way to look at what other's have done or we need to have ascending learning rates
            }
            
            var currentLength = chain.DomainLengthScalar();
            var lengthDiff = currentLength - new Scalar(originalLength);
            var lengthPenalty = lengthDiff.Square() * new Scalar(lengthWeight);
            lengthPenalty.Backward(1f);
            totalLoss += lengthPenalty.Value;
            for (int j = 0; j < paravectors.Count; j++) paravectors[j].Update(baseLR, alphaLR, thetaLR, betaLR, j, paravectors.Count);
            
            action?.Invoke(totalLoss, epoch);
            if (totalLoss < lossEpsilon) return chain;
        }
        
        return chain;
    }
}