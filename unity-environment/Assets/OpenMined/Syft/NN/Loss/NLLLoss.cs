using OpenMined.Syft.Tensor;
using OpenMined.Network.Controllers;

namespace OpenMined.Syft.Layer.Loss
{
    public class NLLLoss: Loss
    {

        public NLLLoss (SyftController controller)
        {
            init("nllloss");
			
            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);

        }
        public override FloatTensor Forward(FloatTensor log_softmax, FloatTensor target)
        {
            // Note: prediction should be logits, basically pre-softmax. This method applies softmax first. 
            // TODO check shapes 

            FloatTensor output = ((target.Mul(log_softmax)).Sum()).Neg();
            return output;
        }

        public override int getParameterCount(){return 0;}
		
    }
}

