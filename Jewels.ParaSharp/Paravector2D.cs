
namespace Jewels.ParaSharp;

public class Paravector2D(float alpha, float theta, float beta)
{
    public Scalar Alpha { get; } = new(alpha);
    public Scalar Theta { get; } = new(theta);
    public Scalar Beta { get; } = new(beta);
    
    public float XLength => Alpha.Value * MathF.Cos(Theta.Value);
    public float YLength => Alpha.Value * MathF.Sin(Theta.Value);
    // add arc length?

    public Scalar LocalX(Scalar x) => (Scalar.Tangent(-Beta) / Alpha) * x * (x - Alpha);
    public float LocalX(float x) => LocalX(new Scalar(x)).Value;

    public Scalar GlobalX(Scalar x, Scalar? h = null, Scalar? k = null)
    {
        h ??= new(0f);
        k ??= new(0f);
        var (four, two) = (new Scalar(4f), new Scalar(2f));
        var m = Scalar.Tangent(-Beta) / Alpha;
        
        var a = m * Scalar.Sine(Theta) + new Scalar(Scalar.Epsilon);
        var b = -(Scalar.Cosine(Theta) + m * Alpha * Scalar.Sine(Theta));
        var c = x - h;

        var discriminant = b.Square() - four * a * c;
        var realDiscriminant = discriminant;
        if (discriminant.Value < 0) 
            realDiscriminant = new Scalar(Scalar.Epsilon, 0f, [discriminant], s => discriminant.Grad += Scalar.Penalty * s.Grad);
        
        var localX = (-b - Scalar.Sqrt(realDiscriminant)) / (two * a);
        var localY = LocalX(localX);
        var y = k + localX * Scalar.Sine(Theta) + localY * Scalar.Cosine(Theta);
        
        return y;
    }
    public float GlobalX(float x, float h, float k) => GlobalX(new Scalar(x), new Scalar(h), new Scalar(k)).Value;
    
    public Func<Scalar, Scalar> AsGlobalX(Scalar h, Scalar k) => x => GlobalX(x, h, k);
    public Func<float, float> AsGlobalX(float h, float k) => x => GlobalX(x, h, k);

    public static Paravector2D FromVector(float x, float y)
    {
        var alpha = MathF.Sqrt(x * x + y * y);
        var theta = MathF.Atan2(y, x);
        var beta = 0;
        return new Paravector2D(alpha, theta, beta);
    }

    public static Paravector2D FromVectorDifference(float x1, float y1, float x2, float y2) => FromVector(x2 - x1, y2 - y1);
    public Paravector2D Reverse() => new(Alpha.Value, Theta.Value + MathF.PI, Beta.Value);

    public void ReverseInPlace()
    {
        Theta.Value += MathF.PI;
        while (Theta.Value > MathF.PI) Theta.Value -= MathF.PI * 2;
        while (Theta.Value < MathF.PI) Theta.Value += MathF.PI * 2;
    }

    public void Update(float baseLR, float thetaScale = 1.0f, float betaScale = 0.01f)
    {
        Alpha.Value -= baseLR * Alpha.Grad;
        Theta.Value -= baseLR * Theta.Grad * thetaScale;
        Beta.Value -= baseLR * Beta.Grad * betaScale;

        if (Alpha.Value < 0) Alpha.Value = Scalar.BiggerEpsilon;
        
        ZeroGrad();
    }
    
    public void ZeroGrad()
    {
        Alpha.ZeroGrad();
        Theta.ZeroGrad();
        Beta.ZeroGrad();
    }

    public static Paravector2D InductiveFit(
        float[] inputs, float[] targets,
        float fixedEndX, float fixedEndY, float lr, int epochs,
        Func<Scalar, float, Scalar> lossFunc,
        float lossEpsilon = Scalar.Epsilon,
        float penalty = 0.1f)
    {
        var (startX, startY) = (inputs[0], targets[0]);
        var upsilon = FromVectorDifference(startX, startY, fixedEndX, fixedEndY);
        Scalar[] inputScalars = inputs.Select(x => new Scalar(x)).ToArray();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Scalar totalLoss = new(0f);

            for (int i = 0; i < inputs.Length; i++)
            {
                var output = upsilon.GlobalX(inputScalars[i], new Scalar(startX), new Scalar(startY));
                totalLoss += lossFunc(output, targets[i]);
            }

            var localEndX = upsilon.Alpha; 
            var endY = upsilon.LocalX(localEndX);

            var globalEndX = new Scalar(startX) + localEndX * Scalar.Cosine(upsilon.Theta);
            var globalEndY = new Scalar(startY) + localEndX * Scalar.Sine(upsilon.Theta) + endY * Scalar.Cosine(upsilon.Theta);

            var endpointLoss = new Scalar(penalty) * (globalEndX - new Scalar(fixedEndX)).Square() + (globalEndY - new Scalar(fixedEndY)).Square();
            totalLoss += endpointLoss;
            totalLoss.Backward(1f); 

            upsilon.Update(lr);

            if (totalLoss.Value < lossEpsilon) break;
        }
        
        return upsilon;
    }
}