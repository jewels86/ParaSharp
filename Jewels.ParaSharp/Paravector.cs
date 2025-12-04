using Jewels.Lazulite;
using Jewels.Opal;

namespace Jewels.ParaSharp;

public static class Paravector
{
    internal static int AcceleratorIndex { get; set; }

    static Paravector()
    {
        try
        {
            OpalContext context = new(initializeInBackground: false);
            context.BeginInitialization();
        }
        catch (Exception)
        {
            // context was already created
        }
        OpalContext.GlobalContext.EnsureInitialization();
        AcceleratorIndex = Compute.Instance.RequestAccelerator(false);
    }

    internal static Tensor<float> NewScalar(float val) => Operations.New(val, aidx: AcceleratorIndex);
    internal static Value<float> NewValue(float val) => Operations.New(val, aidx: AcceleratorIndex);

    public static Tensor<float> MeanSquaredError(Tensor<float> a, Value<float> b)
    {
        var part = (a.Value.AsScalar() - b.AsScalar());
        var loss = part * part;
        return new(loss, b.Zeros(), Backwards, [a]);

        void Backwards(ITensor t)
        {
            var grad = ((Value<float>)t.Value).AsScalar();
            a.Gradient.UpdateWith(a.Gradient.AsScalar() + grad * NewValue(2f).AsScalar() * part);
        }
    }
}