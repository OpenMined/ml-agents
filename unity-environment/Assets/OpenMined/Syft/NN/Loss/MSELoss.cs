using System.Runtime.InteropServices;
using OpenMined.Syft.Tensor;
using UnityEngine;
using OpenMined.Network.Controllers;

namespace OpenMined.Syft.Layer.Loss
{
    public class MSELoss:Loss
    {
        public MSELoss (SyftController controller)
        {
            init("mseloss");

            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);

        }
        public override FloatTensor Forward(FloatTensor prediction, FloatTensor target)
        {
            FloatTensor output = ((prediction.Sub(target)).Pow(2)).Sum(1).Div(prediction.Shape[1]).Sum(0).Div(prediction.Shape[0]);
            return output;
        }
        
        public override int getParameterCount(){return 0;}
    }
}
