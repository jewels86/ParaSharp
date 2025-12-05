using Jewels.ParaSharp;
using Jewels.ParaSharp.Plotting;

namespace Testing;

public class Tests
{
    public static void SimpleTest()
    {
        Paravector2D upsilon = Paravector2D.FromVector(1, 1);
        upsilon.Beta.Value = MathF.PI / 4;
        upsilon.Plot(path: "simple_test.png");
    }
    public static void TestSineApproximation()
    {
        Console.WriteLine("Sine Approximation Test\n");
    
        int numPoints = 80;
        float[] inputs = new float[numPoints];
        float[] targets = new float[numPoints];
    
        for (int i = 0; i < numPoints; i++)
        {
            float x = i * 2f * MathF.PI / (numPoints - 1);
            inputs[i] = x;
            targets[i] = MathF.Sin(x);
        }
    
        Console.WriteLine("Training 8 paravectors to approximate sin(x) from 0 to 2π...");
        //var chain = Chain2D.InductiveDescent(8, 0.02f, 3000, inputs, targets, Scalar.MSE);
        var chain = Chain2D.InductiveExploration(8, 0.02f, 3000, inputs, targets, Scalar.MSE, 
            refinementEpochs: 20, explorationRate: 0.5f, progressAction: (epoch, loss) => Console.WriteLine($"Refinement epoch {epoch} | Loss: {loss}"));
        Console.WriteLine($"Training complete! {chain.DomainLength()} domain length.");
    
        Console.WriteLine("Testing approximation:");
        float[] testPoints = [0f, MathF.PI / 4, MathF.PI / 2, MathF.PI, 3 * MathF.PI / 2, 2 * MathF.PI];
    
        foreach (float x in testPoints)
        {
            float predicted = chain.Evaluate(x);
            float actual = MathF.Sin(x);
            float error = MathF.Abs(predicted - actual);
            Console.WriteLine($"x = {x,5:F2} | sin(x) = {actual,6:F3} | predicted = {predicted,6:F3} | error = {error:F4}");
        }
        chain.Plot(inputs, targets, path: "sin_approximation.png");
    }
}