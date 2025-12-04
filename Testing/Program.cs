using Jewels.Opal;

namespace Testing;

class Program
{
    static void Main(string[] args)
    {
        using var context = new OpalContext(initializeInBackground: true, useGpu: false);
        Tests.TestSineApproximation();
    }
}