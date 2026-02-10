namespace Ramen.AI;

public static class Profiling
{
    public static bool CollectData = true;

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

        List<ProfileDatum> timeSeries = GetTimeSeries();
        timeSeries.Add(new ProfileDatum(IsEnter: true, name, DateTime.UtcNow));
    }
    
    public static void Exit(string name)
    {
        if (!CollectData)
            return;

        List<ProfileDatum> timeSeries = GetTimeSeries();
        timeSeries.Add(new ProfileDatum(IsEnter: false, name, DateTime.UtcNow));
    }

    public static (List<(string tag, float fraction)> fractions, float averageMilliseconds) GetFractionsAndAverageMillisecondsForTag(string tagName)
    {
        // Aggregate totals for the requested tag across all threads.
        Dictionary<string, float> childDurationMsByTag = [];
        float totalDurationMs = 0f;
        int totalCount = 0;

        foreach (KeyValuePair<int, List<ProfileDatum>> kvp in _data)
        {
            Stack<ScopeFrame> scopeStack = new();
            List<ProfileDatum> timeSeries = kvp.Value;

            // Reconstruct nested scopes and attribute child durations to parents.
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

        // Convert aggregated child durations into fractions and compute average duration.
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
        return new ProfileScope(name);
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        Profiling.Exit(Name);
    }
}
