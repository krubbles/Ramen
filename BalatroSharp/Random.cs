namespace Ramen.Game;
/// <summary>
/// A xorshift* 64/32 psudo-random number generator.
/// </summary>
public class FastRandom
{
    ulong _state;

    public FastRandom(ulong seed)
    {
        _state = seed + 0x9E3779B97f4A7C15;
        _state = (_state ^ (_state >> 30)) * 0xBF58476D1CE4E5B9;
        _state = (_state ^ (_state >> 27)) * 0x94D049BB133111EB;
        _state ^= _state >> 31;
    }

    public FastRandom(int seed) : this((ulong)seed) { }

    public FastRandom(FastRandom toClone)
    {
        _state = toClone._state;
    }

    public void SetState(ulong state) => _state = state;
    public ulong GetState() => _state;

    public int Next()
    {
        _state ^= _state >> 12;
        _state ^= _state << 25;
        _state ^= _state >> 27;
        return (int)((_state * 0x2545F4914F6CDD1DUL) >> 32);
    }

    public int Next(int maxExclusive)
    {
        _state ^= _state >> 12;
        _state ^= _state << 25;
        _state ^= _state >> 27;
        return (int)((uint)((_state * 0x2545F4914F6CDD1DUL) >> 32) % (uint)maxExclusive);
    }

    public uint NextUnsigned()
    {
        _state ^= _state >> 12;
        _state ^= _state << 25;
        _state ^= _state >> 27;
        return (uint)((_state * 0x2545F4914F6CDD1DUL) >> 32);
    }

    public float NextPortion()
    {
        _state ^= _state >> 12;
        _state ^= _state << 25;
        _state ^= _state >> 27;
        return (uint)((_state * 0x2545F4914F6CDD1DUL) >> 32) * uintToPortion;
    }

    public int NextInRange(int min, int maxExclusive) => min + (int)(NextUnsigned() % (uint)(maxExclusive - min));
    public float NextInRange(float min, float max) => min + (max - min) * NextPortion();
    public float NextInRangeLog(float min, float max) => min * MathF.Pow(max / min, NextPortion());

    public float NextNormal(float average, float dev)
    {
        float v = NextPortion() + NextPortion() + NextPortion();
        return average + (v * 2 - 1) * dev;
    }

    public bool NextFlip(float prob) => NextPortion() < prob;

    public T NextChoiceProb<T>(T a, T b, float aProb) => NextPortion() < aProb ? a : b;

    public T NextChoice<T>(T a, T b) => NextPortion() < 0.5f ? a : b;

    public T NextChoice<T>(T a, T b, T c) => NextPortion() < (1f / 3f) ? a : NextPortion() < 0.5f ? b : c;

    public T NextChoice<T>(T a, T b, T c, T d) => NextPortion() < 0.5f ?
        NextPortion() < 0.5f ? a : b :
        NextPortion() < 0.5f ? c : d;

    public T NextChoice<T>(T a, T b, T c, T d, T e) => NextPortion() < 0.4f ? NextChoice(a, b) : NextChoice(c, d, e);

    const float uintToPortion = 1f / (uint.MaxValue - 1);
    const int prime = 404332459;

    public T NextPick<T>(IEnumerable<T> enumerable)
    {
        T[] values = enumerable.ToArray();
        int index = Next(values.Length);
        return values[index];
    }

    public T NextPickWeighted<T>(T[] values, float[] weights)
    {
        float weightTotal = 0;
        for (int i = 0; i < weights.Length; ++i)
            weightTotal += weights[i];

        float sampleValue = NextPortion() * weightTotal;

        float cumulativeWeight = 0;
        for (int i = 0; i < weights.Length; ++i)
        {
            cumulativeWeight += weights[i];
            if (cumulativeWeight > sampleValue)
                return values[i];
        }

        return values[^1];
    }

    public static FastRandom SeededByClock() => new((ulong)DateTime.Now.Ticks);
}