namespace Jewels.ParaSharp;

public class Scalar
{
    public float Value { get; set; }
    public float Grad { get; set; }
    public List<Scalar> Inputs { get; }
    readonly private Action<Scalar> _backward;

    public Scalar(float val, float grad = 0.0f, List<Scalar>? inputs = null, Action<Scalar>? backward = null)
    {
        if (float.IsNaN(val)) throw new Exception("NaN value");
        Value = val;
        Grad = grad;
        Inputs = inputs ?? [];
        _backward = backward ?? (_ => { });
    }

    public void Backward(float initialGradient)
    {
        (List<Scalar> topo, HashSet<Scalar> visited) = ([], []);
        Build(this, topo, visited);

        Grad = initialGradient;
        foreach (var node in topo.AsEnumerable().Reverse()) node.BackwardRecursive(node);
    }

    private static void Build(Scalar node, List<Scalar> topo, HashSet<Scalar> visited)
    {
        if (!visited.Add(node)) return;
        foreach (var input in node.Inputs) Build(input, topo, visited);
        topo.Add(node);
    }
    
    private void BackwardRecursive(Scalar scalar) => _backward?.Invoke(scalar);

    public static Scalar operator +(Scalar a, Scalar b) =>
        new(a.Value + b.Value, 0f, [a, b], s =>
        {
            a.Grad += s.Grad;
            b.Grad += s.Grad;
        });

    public static Scalar operator -(Scalar a, Scalar b) =>
        new(a.Value - b.Value, 0f, [a, b], s =>
        {
            a.Grad += s.Grad;
            b.Grad -= s.Grad;
        });

    public static Scalar operator *(Scalar a, Scalar b) =>
        new(a.Value * b.Value, 0f, [a, b], s =>
        {
            a.Grad += s.Grad * b.Value;
            b.Grad += s.Grad * a.Value;
        });
    
    public static Scalar operator /(Scalar a, Scalar b) =>
        new(a.Value / b.Value, 0f, [a, b], s =>
        {
            a.Grad += s.Grad / (Single.Epsilon + b.Value);
            b.Grad += -s.Grad * a.Value / (b.Value * b.Value + Single.Epsilon);
        });
    
    public static Scalar operator -(Scalar a) => new(-a.Value, 0f, [a], s => a.Grad -= s.Grad);
    
    public static Scalar Sine(Scalar a) =>
        new(MathF.Sin(a.Value), 0f, [a], s =>
        {
            a.Grad += s.Grad * MathF.Cos(a.Value);
        });

    public static Scalar Cosine(Scalar a) =>
        new(MathF.Cos(a.Value), 0f, [a], s =>
        {
            a.Grad += -s.Grad * MathF.Sin(a.Value);
        });

    public static Scalar Tangent(Scalar a) =>
        new(MathF.Tan(a.Value), 0f, [a], s =>
        {
            float sec = 1f / (MathF.Cos(a.Value) + Single.Epsilon);
            a.Grad += s.Grad * sec * sec;
        });

    public static Scalar Sqrt(Scalar a) =>
        new(MathF.Sqrt(a.Value), 0f, [a], s =>
        {
            a.Grad += s.Grad / (2f * MathF.Sqrt(a.Value) + Single.Epsilon);
        });
    
    public Scalar Square() => this * this;
    
    public void ZeroGrad() => Grad = 0f;

    public static Scalar MSE(Scalar output, float value)
    {
        var part = output.Value - value;
        return new(part * part, 0f, [output], s => output.Grad += s.Grad * 2f * part);
    }

    public const float Epsilon = 1e-6f;
    public const float BiggerEpsilon = 0.01f;
    public const float Penalty = 1f;
}