using System.Runtime.InteropServices.Swift;
using Jewels.Lazulite;
using Jewels.Opal;
using static Jewels.Opal.Operations;
using static Jewels.ParaSharp.Paravector;

namespace Jewels.ParaSharp;

public class Paravector2(float alpha, float theta, float beta)
{
    public Tensor<float> Alpha { get; } = NewScalar(alpha);
    public Tensor<float> Theta { get; } = NewScalar(theta);
    public Tensor<float> Beta { get; } = NewScalar(beta);
    
    public Tensor<float> LocalX(Tensor<float> x) => (Tangent(-Beta) / Alpha) * x * (x - Alpha);
    public float LocalX(float x) => LocalX(NewScalar(x)).Value.ToHost();

    public Tensor<float> GlobalX(Tensor<float> x, Tensor<float>? h = null, Tensor<float>? k = null)
    {
        h ??= NewScalar(0f);
        k ??= NewScalar(0f);
        var (four, two) = (NewScalar(4f), NewScalar(2f));
        var m = Tangent(-Beta) / Alpha;
        
        var a = m * Sine(Theta);
        var b = -(Cosine(Theta) + m * Alpha * Sine(Theta));
        var c = x - h;

        var discriminant = Square(b) - four * a * c;
        
        var localX = (-b - Sqrt(discriminant)) / (two * a);
        var localY = LocalX(localX);
        var y = k + localX * Sine(Theta) + localY * Cosine(Theta);
        
        return y;
    }
    public float GlobalX(float x, float h, float k) => GlobalX(NewScalar(x), NewScalar(h), NewScalar(k)).Value.ToHost();
    
    public Func<Tensor<float>, Tensor<float>> AsGlobalX(Tensor<float> h, Tensor<float> k) => x => GlobalX(x, h, k);
    public Func<float, float> AsGlobalX(float h, float k) => x => GlobalX(x, h, k);

    public static Paravector2 FromVector(float x, float y)
    {
        var alpha = MathF.Sqrt(x * x + y * y);
        var theta = MathF.Atan2(y, x);
        var beta = Single.Epsilon;
        return new Paravector2(alpha, theta, beta);
    }

    public static Paravector2 FromVectorDifference(float x1, float y1, float x2, float y2) => FromVector(x2 - x1, y2 - y1);

    public void Update(Value<float> lr)
    {
        Alpha.Value.UpdateWith(Alpha.Value.AsScalar() - Alpha.Gradient.AsScalar() * lr.AsScalar());
        Beta.Value.UpdateWith(Beta.Value.AsScalar() - Beta.Gradient.AsScalar() * lr.AsScalar());
        Theta.Value.UpdateWith(Theta.Value.AsScalar() - Theta.Gradient.AsScalar() * lr.AsScalar());
                
        Alpha.Gradient.UpdateWith(Paravector.NewValue(0f));
        Beta.Gradient.UpdateWith(Paravector.NewValue(0f));
        Theta.Gradient.UpdateWith(Paravector.NewValue(0f));
    }
}