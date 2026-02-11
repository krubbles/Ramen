using Ramen.Game;
using Ramen.AI;
using Ramen.Training;
using TorchSharp;
using static TorchSharp.torch;

// Do not change START
set_default_device(MPS);
TensorManager.Init();
Console.WriteLine("=== START ===");
// Do not change END

