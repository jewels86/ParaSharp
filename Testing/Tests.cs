using ILGPU.Runtime.Cuda;
using Jewels.Opal;
using Jewels.ParaSharp;

namespace Testing;

public class Tests
{
    public static void TestSineApproximation()
    {
        Console.WriteLine("Sine Approximation Test\n");
    
        int numPoints = 100;
        float[] inputs = new float[numPoints];
        float[] targets = new float[numPoints];
    
        for (int i = 0; i < numPoints; i++)
        {
            float x = i * 2f * MathF.PI / (numPoints - 1);
            inputs[i] = x;
            targets[i] = MathF.Sin(x);
        }
    
        Console.WriteLine("Training 8 paravectors to approximate sin(x) from 0 to 2π...");
        var chain = Chain2.Fit(
            total: 8,
            inputs: inputs,
            targets: targets,
            lr: 0.01f,
            maxEpochs: 1000,
            loss: Paravector.MeanSquaredError,
            epsilon: 0.001f,
            action: (loss, epoch) =>
            {
                if (epoch % 100 == 0)
                    Console.WriteLine($"Epoch {epoch}: Loss = {loss:F6}");
            }
        );
    
        Console.WriteLine("\nTraining complete!\n");
    
        Console.WriteLine("Testing approximation:");
        float[] testPoints = [0f, MathF.PI / 4, MathF.PI / 2, MathF.PI, 3 * MathF.PI / 2, 2 * MathF.PI];
    
        foreach (float x in testPoints)
        {
            float predicted = chain.Evaluate(x);
            float actual = MathF.Sin(x);
            float error = MathF.Abs(predicted - actual);
            Console.WriteLine($"x = {x,5:F2} | sin(x) = {actual,6:F3} | predicted = {predicted,6:F3} | error = {error:F4}");
        }
    }
}