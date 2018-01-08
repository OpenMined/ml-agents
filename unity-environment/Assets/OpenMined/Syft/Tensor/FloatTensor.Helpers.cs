using System;
using System.Threading.Tasks;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        private bool SameSizeDimensionsShapeAndLocation(ref FloatTensor tensor)
        {

            bool use_backup = false;
            
            if (dataOnGpu != tensor.dataOnGpu)
            {
                throw new InvalidOperationException(String.Format("Tensors must be on same device : {0} != {1}.", dataOnGpu, tensor.dataOnGpu));
            }

            if (tensor.Size == 1 && Size != 1)
            {
                // should retry with scalar version
                return true;
            }
            if (tensor.Size != 1 && Size == 1)
            {
                // should retry with scalar version
                return true;
            }
            
            // Check if both tensors have same size
            if (tensor.Size != size)
            {
                throw new InvalidOperationException(String.Format("Tensors cannot be added since they have different sizes: {0} != {1}", tensor.Size, size));    
            }
            
            // Check if both tensors have same number of dimensions
            if (tensor.Shape.Length != shape.Length)
            {
                throw new InvalidOperationException(
                    String.Format("Tensors cannot be added since they have different number of dimensions: {0} != {1}", tensor.Shape.Length, shape.Length));
            }

            // Check if both tensors have same shapes
            for (var i = 0; i < shape.Length; i++)
            {
                if (shape[i] != tensor.Shape[i])
                {
                    throw new InvalidOperationException("Tensors cannot be added since they have different shapes.");
                }
            }
            return false;
        }
        
        private void AssertDim(int dim, int len)
        {
            if (dim < 0 || dim >= len)
            {
                throw new ArgumentOutOfRangeException(nameof(dim), "Must be between 0 and shape length exclusive.");
            }
        }

        private int GetDimReduceOffset(int index, int values, int stride)
        {
            return values * stride * (index / stride) + index % stride;
        }

        private void _dimForEach(
            int interations,
            int values,
            int stride,
            Action<float[], int, int> iterator
        )
        {
            MultiThread.For(interations, (i, len) =>
            {
                var temp = new float[values];

                int offset = GetDimReduceOffset(i, values, stride);

                for (int v = 0; v < values; v++)
                {
                    temp[v] = this[offset + v * stride];
                }

                iterator(temp, i, temp.Length);
            });
        }
    }

    public static class MultiThread
    {
        // call func on multiple threads for the range [0..len)
        public static void For(int len, Action<int, int> func)
        {
            int nCPU = Environment.ProcessorCount;

            // only use as many threads as needed (MAX array length)
            nCPU = Math.Min(nCPU, len);

            Parallel.For(0, nCPU, workerId =>
            {
                int max = len * (workerId + 1) / nCPU;
                int offset = len * workerId / nCPU;

                for (int i = offset; i < max; i++)
                {
                    func(i, len);
                }

                // shows the ranges preccessed by each thread
                // Console.WriteLine("\tnCPU {0} | workerId {1} :: {2}-{3}", nCPU, workerId, offset, max);
            });
        }

        // reduce items of array on multiple threads
        public static T Reduce<T>(T[] data, Func<T, T, int, T[], T> func)
        {
            int len = data.Length;
            int nCPU = Environment.ProcessorCount;

            // only use as many threads as needed (MAX array length / 2)
            // because reduce func proccesses two items at once
            nCPU = Math.Min(nCPU, len / 2);

            var prev = data;

            for (; nCPU > 0; nCPU = nCPU / 2)
            {
                len = prev.Length;

                T[] curr = new T[nCPU];

                Parallel.For(0, nCPU, workerId =>
                {
                    var max = len * (workerId + 1) / nCPU;
                    var offset = len * workerId / nCPU;

                    T acc = prev[offset];

                    for (var i = offset + 1; i < max; i++)
                    {
                        acc = func(acc, prev[i], i, data);
                    }

                    curr[workerId] = acc;

                    // shows the number of items preccessed by each thread
                    // Console.WriteLine("\tnCPU {0} | workerId {1} :: {2}", nCPU, workerId, max - offset);
                });

                prev = curr;

                // show when a reduce iteration completes
                // Console.WriteLine("\t-- reduce iteration complete --");
            }

            // return the reduced value
            return prev[0];
        }
    }
}
