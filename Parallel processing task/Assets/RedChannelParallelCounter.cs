using System.Diagnostics;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class RedChannelParallelCounter : MonoBehaviour
{
    [Header("Assign a Read/Write Enabled Texture")]
    public Texture2D sourceTexture;

    [Header("Batch Size (Pixels per Job Chunk)")]
    public int batchSize = 4096;

    void Start()
    {
        if (sourceTexture == null)
        {
            Debug.LogError("Please assign a Texture2D in the Inspector!");
            return;
        }

        // 1. Get raw CPU pixel array (zero-allocation view into texture memory)
        NativeArray<Color32> pixels = sourceTexture.GetRawTextureData<Color32>();
        int totalPixels = pixels.Length;

        // Calculate how many chunks we need so threads don't collide on one variable
        int chunkCount = Mathf.CeilToInt((float)totalPixels / batchSize);
        NativeArray<long> partialSums = new NativeArray<long>(chunkCount, Allocator.TempJob);

        // 2. Run and time the Parallel Burst Job
        Stopwatch sw = Stopwatch.StartNew();

        SumRedJob job = new SumRedJob
        {
            Pixels = pixels,
            PartialSums = partialSums,
            BatchSize = batchSize
        };

        // Schedule chunkCount tasks across Unity's worker threads
        JobHandle handle = job.Schedule(chunkCount, 1);
        handle.Complete(); // Wait for all worker threads to finish

        // Add the chunk subtotals together on the main thread
        long parallelRedTotal = 0;
        for (int i = 0; i < chunkCount; i++)
        {
            parallelRedTotal += partialSums[i];
        }

        sw.Stop();
        double parallelMs = sw.Elapsed.TotalMilliseconds;

        // 3. Optional comparison: Single-threaded loop (great for your COMMENTARY.md)
        sw.Restart();
        long singleThreadTotal = 0;
        for (int i = 0; i < totalPixels; i++)
        {
            singleThreadTotal += pixels[i].r;
        }
        sw.Stop();
        double singleMs = sw.Elapsed.TotalMilliseconds;

        // 4. Log results
        Debug.Log($"Image Resolution: {sourceTexture.width}x{sourceTexture.height} ({totalPixels:N0} pixels)");
        Debug.Log($"[Parallel + Burst] Red Sum: {parallelRedTotal:N0} | Time: {parallelMs:F4} ms");
        Debug.Log($"[Single-Threaded]  Red Sum: {singleThreadTotal:N0} | Time: {singleMs:F4} ms");

        // Always dispose TempJob NativeArrays to prevent memory leaks!
        partialSums.Dispose();
    }

    // BurstCompile translates this struct into fast native machine code
    [BurstCompile]
    struct SumRedJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Color32> Pixels;
        [WriteOnly] public NativeArray<long> PartialSums;
        public int BatchSize;

        public void Execute(int chunkIndex)
        {
            int start = chunkIndex * BatchSize;
            int end = Mathf.Min(start + BatchSize, Pixels.Length);

            long localSum = 0;
            for (int i = start; i < end; i++)
            {
                localSum += Pixels[i].r;
            }

            // Each worker writes to its own dedicated index (no race conditions)
            PartialSums[chunkIndex] = localSum;
        }
    }
}