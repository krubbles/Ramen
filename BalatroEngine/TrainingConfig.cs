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
        public const int EpochsPerDataGen = 20;

        public const int SampleCount = 2;
        public const float Temperature = 0.03f;

        public const int DataSize = 40000;
        public const int DataGenAmount = 2000;

        public const int ScoringHeuristicSampleSize = 1000;

    }
}
