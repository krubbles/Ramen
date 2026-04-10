namespace Ramen.ConsoleApp;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using Ramen.AI;
using Ramen.AgentTools;
using Ramen.Game;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public static partial class Program
{
    public static void Main()
    {
        // Do not change START
        set_default_device(mps_is_available() ? MPS : CPU);
        TensorManager.Init();
        Console.WriteLine("=== START ===");
        // Do not change END      
    }
}
