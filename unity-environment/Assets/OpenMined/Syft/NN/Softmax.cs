using System;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.Layer
{
    public class Softmax : Layer
    {

        private int dim = 0;
        
        public Softmax(SyftController controller, int dim)
        {
            init("softmax");

            this.dim = dim;
            
            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public override FloatTensor Forward(FloatTensor input)
        {
            FloatTensor output = input.Softmax(this.dim);
            activation = output.Id;

            return output;
        }
        
        public override int getParameterCount(){return 0;}
    }
}