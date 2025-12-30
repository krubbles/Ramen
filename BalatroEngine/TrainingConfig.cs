using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BalatroAI
{
    public static class TrainingConfig
    {
        public const int BatchSize = 64;
        public const float LearningRate = 0.001f;

        public const float GoodPlayTemp = 1.2f;
        public const float ExploratoryPlayTemp = 2f;
    }
}
