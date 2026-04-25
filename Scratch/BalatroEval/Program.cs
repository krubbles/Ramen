using Ramen.AI;
using Ramen.ConsoleApp;
using Ramen.Game;
using static TorchSharp.torch;

string checkpointPath = "/Users/miles/Desktop/dev/repos/BalatroAI/Analysis/2026-04-21_simple_flush_ppo_stdreward_resume1230_r32768_b1024_e2_lr1e5_100more_eps0p3_ent0_ss40_trunk512_addgelu_vhead_compiledfp16/weights/1330.bin";
const int gameCount = 10000;

set_default_device(mps_is_available() ? MPS : CPU);
TensorManager.Init();

using PpoPolicyValueModel model = new(useTorchScriptCompile: false, useHalfPrecisionLinearWeights: false);
model.Load(checkpointPath);

using PolicyOnlyAgent agent = new(model);

GameState[] games = new GameState[gameCount];
for (int index = 0; index < games.Length; ++index)
{
    GameData gameData = new()
    {
        Seed = GameData.Default.Seed,
        Hands = 4,
        Discards = 2,
        RandomizeSeed = GameData.Default.RandomizeSeed,
        StartingHandBaseScore = [.. GameData.Default.StartingHandBaseScore],
        PlanetScores = [.. GameData.Default.PlanetScores],
        InitStartingDeck = GameData.Default.InitStartingDeck,
    };
    foreach ((string jokerName, Joker joker) in GameData.Default.Jokers)
        gameData.Jokers.Add(jokerName, joker);
    games[index] = new(gameData);
}

while (true)
{
    bool allGamesDone = true;
    for (int index = 0; index < games.Length; ++index)
    {
        if (!agent.IsGameDone(games[index]))
        {
            allGamesDone = false;
            break;
        }
    }

    if (allGamesDone)
        break;

    agent.MakeMove(temp: 1f, annotatePolicy: false, games);
}

int lossCount = 0;
int winHands0Count = 0;
int winHands1Count = 0;
int winHands2Count = 0;
int winHands3Count = 0;

for (int index = 0; index < games.Length; ++index)
{
    GameState game = games[index];
    if (game.ScoringState.CurrentRoundTotalChips < 300)
    {
        lossCount++;
        continue;
    }

    int remainingHands = game.HandState.RemainingHands;
    if (remainingHands == 0)
        winHands0Count++;
    else if (remainingHands == 1)
        winHands1Count++;
    else if (remainingHands == 2)
        winHands2Count++;
    else if (remainingHands == 3)
        winHands3Count++;
}

Console.WriteLine($"checkpoint {checkpointPath}");
Console.WriteLine($"games {gameCount}");
Console.WriteLine("starting_hands 4");
Console.WriteLine("starting_discards 2");
Console.WriteLine($"loss_count {lossCount}");
Console.WriteLine($"win_hands_0_count {winHands0Count}");
Console.WriteLine($"win_hands_1_count {winHands1Count}");
Console.WriteLine($"win_hands_2_count {winHands2Count}");
Console.WriteLine($"win_hands_3_count {winHands3Count}");
Console.WriteLine($"loss_frac {(float)lossCount / gameCount:F8}");
Console.WriteLine($"win_hands_0_frac {(float)winHands0Count / gameCount:F8}");
Console.WriteLine($"win_hands_1_frac {(float)winHands1Count / gameCount:F8}");
Console.WriteLine($"win_hands_2_frac {(float)winHands2Count / gameCount:F8}");
Console.WriteLine($"win_hands_3_frac {(float)winHands3Count / gameCount:F8}");
