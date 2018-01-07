using System;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.Layer
{
    public class Tanh : Layer
    {

        public Tanh(SyftController controller)
        {
            init("tanh");

            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }
    
        public override FloatTensor Forward(FloatTensor input)
        {
            FloatTensor output = input.Tanh();
            activation = output.Id;

            return output;
        }
        
        public override int getParameterCount(){return 0;}
    }
}
