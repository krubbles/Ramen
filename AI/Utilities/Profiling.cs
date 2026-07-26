namespace Ramen.AI;

public static class Profiling
{
    public static bool CollectData = true;

    /// <summary>
    /// Optional device synchronization run at every scope boundary.
    /// <para>
    /// GPU work is queued asynchronously, so without this a scope measures how long it
    /// took to submit work, not to run it, and all the cost piles up at whichever later
    /// call happens to read a result back. Set this to a device round-trip when timing
    /// individual phases; leave it null in normal runs, where it would cost throughput.
    /// </para>
    /// </summary>
    public static Action SynchronizeDevice;

    /// <summary>
    /// Appended to network phase scope names so the same block can be told apart between
    /// rollout inference and the training update. Empty outside profiling runs.
    /// </summary>
    public static string PhaseSuffix = "";

    /// <summary>
    /// Total wall time spent inside every scope with this tag, and how many times it ran.
    /// </summary>
    public static (float totalMs, int count) GetTotalMillisecondsForTag(string tagName)
    {
        float totalMs = 0f;
        int count = 0;

        foreach (KeyValuePair<int, List<ProfileDatum>> kvp in _data)
        {
            Stack<(string name, DateTime enterTime)> scopeStack = new();
            foreach (ProfileDatum datum in kvp.Value)
            {
                if (datum.IsEnter)
                {
                    scopeStack.Push((datum.Name, datum.TimeStamp));
                    continue;
                }

                while (scopeStack.Count > 0)
                {
                    (string name, DateTime enterTime) = scopeStack.Pop();
                    if (name == tagName)
                    {
                        totalMs += Math.Max(0f, (float)(datum.TimeStamp - enterTime).TotalMilliseconds);
                        count++;
                    }
                    if (name == datum.Name)
                        break;
                }
            }
        }

        return (totalMs, count);
    }

    /// <summary>
    /// Discards all collected samples so a following measurement starts clean.
    /// </summary>
    public static void Clear() => _data = new();

    static Dictionary<int, List<ProfileDatum>> _data = new();

    sealed class ScopeFrame
    {
        public readonly string Name;
        public readonly DateTime EnterTime;
        public readonly Dictionary<string, float> ChildDurationMsByTag = [];

        public ScopeFrame(string name, DateTime enterTime)
        {
            Name = name;
            EnterTime = enterTime;
        }
    }

    public static List<ProfileDatum> GetTimeSeries()
    {
        int threadId = Environment.CurrentManagedThreadId;
        if (!_data.TryGetValue(threadId, out List<ProfileDatum> timeSeries))
        {
            Dictionary<int, List<ProfileDatum>> newData = new();
            foreach (KeyValuePair<int, List<ProfileDatum>> kvp in _data)
                newData[kvp.Key] = kvp.Value;

            timeSeries = [];
            newData[threadId] = timeSeries;
            _data = newData;
        }

        return timeSeries;
    }

    public static void Enter(string name)
    {
        if (!CollectData)
            return;

        SynchronizeDevice?.Invoke();
        List<ProfileDatum> timeSeries = GetTimeSeries();
        timeSeries.Add(new ProfileDatum(IsEnter: true, name, DateTime.UtcNow));
    }

    public static void Exit(string name)
    {
        if (!CollectData)
            return;

        SynchronizeDevice?.Invoke();
        List<ProfileDatum> timeSeries = GetTimeSeries();
        timeSeries.Add(new ProfileDatum(IsEnter: false, name, DateTime.UtcNow));
    }

    public static (List<(string tag, float fraction)> fractions, float averageMilliseconds) GetFractionsAndAverageMillisecondsForTag(string tagName)
    {
        Dictionary<string, float> childDurationMsByTag = [];
        float totalDurationMs = 0f;
        int totalCount = 0;

        foreach (KeyValuePair<int, List<ProfileDatum>> kvp in _data)
        {
            Stack<ScopeFrame> scopeStack = new();
            List<ProfileDatum> timeSeries = kvp.Value;

            foreach (ProfileDatum datum in timeSeries)
            {
                if (datum.IsEnter)
                {
                    scopeStack.Push(new(name: datum.Name, enterTime: datum.TimeStamp));
                    continue;
                }

                while (scopeStack.Count > 0)
                {
                    ScopeFrame frame = scopeStack.Pop();
                    float durationMs = (float)(datum.TimeStamp - frame.EnterTime).TotalMilliseconds;
                    if (durationMs < 0f)
                        durationMs = 0f;

                    if (scopeStack.TryPeek(out ScopeFrame parentFrame))
                    {
                        if (!parentFrame.ChildDurationMsByTag.TryGetValue(frame.Name, out float childDurationMs))
                            childDurationMs = 0f;
                        parentFrame.ChildDurationMsByTag[frame.Name] = childDurationMs + durationMs;
                    }

                    if (frame.Name == tagName)
                    {
                        totalDurationMs += durationMs;
                        totalCount += 1;

                        foreach (KeyValuePair<string, float> childDuration in frame.ChildDurationMsByTag)
                        {
                            if (!childDurationMsByTag.TryGetValue(childDuration.Key, out float existingDurationMs))
                                existingDurationMs = 0f;
                            childDurationMsByTag[childDuration.Key] = existingDurationMs + childDuration.Value;
                        }
                    }

                    if (frame.Name == datum.Name)
                        break;
                }
            }
        }

        if (totalCount == 0 || totalDurationMs <= 0f)
            return ([], 0f);

        List<(string tag, float fraction)> fractions = [];
        foreach (KeyValuePair<string, float> childDuration in childDurationMsByTag)
            fractions.Add((tag: childDuration.Key, fraction: childDuration.Value / totalDurationMs));
        fractions.Sort((left, right) => right.fraction.CompareTo(left.fraction));

        float averageMilliseconds = totalDurationMs / totalCount;
        return (fractions, averageMilliseconds);
    }
}

public record struct ProfileDatum(bool IsEnter, string Name, DateTime TimeStamp);

public struct ProfileScope : IDisposable
{
    public readonly string Name;
    bool _disposed;

    ProfileScope(string name) => Name = name;

    public static ProfileScope New(string name)
    {
        Profiling.Enter(name);
        return new(name);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        Profiling.Exit(Name);
    }
}
