using System;

namespace OpenMined.Syft.Tensor
{
    public abstract partial class BaseTensor<T>
    {
        // Do not decrement nCreated everytime we dispose an object to avoid id collisions.
        private static volatile int nDeleted = 0;

        private bool disposed = false;

        public static int DeletedObjectCount => nDeleted;

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposed)
                return;

            if (disposing)
            {
                data = null;
                shape = null;
                strides = null;

                if (dataOnGpu)
                {
                    EraseGpu();
                }
            }

            disposed = true;
        }

        ~BaseTensor()
        {
            Dispose(false);
            #pragma warning disable 420
            System.Threading.Interlocked.Increment(ref nDeleted);
        }
    }
}