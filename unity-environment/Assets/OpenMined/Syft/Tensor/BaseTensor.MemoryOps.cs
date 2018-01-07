using System.Runtime.InteropServices;
using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public abstract partial class BaseTensor<T>
    {
        protected bool dataOnGpu;

        protected ComputeBuffer dataBuffer;
        protected ComputeBuffer shapeBuffer;

        public bool DataOnGpu => dataOnGpu;

        public ComputeBuffer DataBuffer
        {
            get { return dataBuffer; }
            set { dataBuffer = value; }
        }

        public ComputeBuffer ShapeBuffer
        {
            get { return shapeBuffer; }
            set { shapeBuffer = value; }
        }

        public bool Gpu(ComputeShader _shader)
        {
            if (dataOnGpu || !SystemInfo.supportsComputeShaders) return false;
            shader = _shader;
            CopyCputoGpu();
            EraseCpu();
            return true;
        }

        public void Cpu()
        {
            if (!dataOnGpu) return;
            CopyGpuToCpu();
            EraseGpu();
        }

        protected void CopyGpuToCpu()
        {
            data = new T[size];
            dataBuffer.GetData(Data);
        }

        protected void CopyCputoGpu()
        {
            dataBuffer = new ComputeBuffer(size, Marshal.SizeOf(default(T)));
            shapeBuffer = new ComputeBuffer(shape.Length, sizeof(int));

            dataBuffer.SetData(Data);
            shapeBuffer.SetData(shape);

            dataOnGpu = true;
        }

        protected void EraseCpu()
        {
            data = null;
        }

        protected void EraseGpu()
        {
            dataBuffer?.Release();
            shapeBuffer?.Release();
            dataOnGpu = false;
        }
    }
}