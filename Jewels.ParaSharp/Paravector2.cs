
namespace Jewels.ParaSharp;

// additions: Length property
public class Paravector2(float alpha, float theta, float beta)
{
    public Scalar Alpha { get; } = new(alpha);
    public Scalar Theta { get; } = new(theta);
    public Scalar Beta { get; } = new(beta);
    
    public Scalar LocalX(Scalar x) => (Scalar.Tangent(-Beta) / Alpha) * x * (x - Alpha);
    public float LocalX(float x) => LocalX(new Scalar(x)).Value;

    public Scalar GlobalX(Scalar x, Scalar? h = null, Scalar? k = null)
    {
        h ??= new(0f);
        k ??= new(0f);
        var (four, two) = (new Scalar(4f), new Scalar(2f));
        var m = Scalar.Tangent(-Beta) / Alpha;
        
        if (Scalar.Sine(Theta).Value < 1e-8) // no rotation
            return k + LocalX(x - h);
        if (MathF.Abs(m.Value) < 1e-8) // no curvature
        {
            var localX2 = (x - h) / Scalar.Cosine(Theta);
            return k + localX2 * Scalar.Sine(Theta);
        }
        
        
        var a = m * Scalar.Sine(Theta);
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

    public static Paravector2 FromVector(float x, float y)
    {
        var alpha = MathF.Sqrt(x * x + y * y);
        var theta = MathF.Atan2(y, x);
        var beta = 0.5f * (MathF.PI * 0.5f - theta);
        return new Paravector2(alpha, theta, beta);
    }

    public static Paravector2 FromVectorDifference(float x1, float y1, float x2, float y2) => FromVector(x2 - x1, y2 - y1);

    public void Update(float baseLR, float alphaLR, float thetaLR, float betaLR, int index, int total)
    {
        var lr = baseLR * (1 - 0.5f * (index / (float)total));
        
        Alpha.Value -= alphaLR * lr * Alpha.Grad;
        Theta.Value -= thetaLR * lr * Theta.Grad;
        Beta.Value -= betaLR * lr * Beta.Grad;

        if (Alpha.Value < 0) Alpha.Value = Scalar.BiggerEpsilon;
        if (Theta.Value > MathF.PI / 2) Theta.Value = MathF.PI - Scalar.BiggerEpsilon;
        if (Theta.Value < -MathF.PI / 2) Theta.Value = -MathF.PI + Scalar.BiggerEpsilon;
        if (Theta.Value + Beta.Value > MathF.PI) Beta.Value = MathF.PI - Theta.Value;
        ZeroGrad();
    }
    
    public void ZeroGrad()
    {
        Alpha.ZeroGrad();
        Theta.ZeroGrad();
        Beta.ZeroGrad();
    }
}