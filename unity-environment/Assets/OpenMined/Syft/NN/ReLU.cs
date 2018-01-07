using System;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.Layer
{
    public class ReLU : Layer
    {

        public ReLU(SyftController controller)
        {
            init("relu");

            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public override FloatTensor Forward(FloatTensor input)
        {
            FloatTensor output = input.ReLU();
            activation = output.Id;

            return output;
        }
        
        public override int getParameterCount(){return 0;}
    }
}