using System;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.Layer.Loss
{
    public abstract class Loss: Model
    {

        public abstract FloatTensor Forward (FloatTensor predicted, FloatTensor target);

        protected override string ProcessForwardMessage(Command msgObj, SyftController ctrl)
        {
            var pred = ctrl.floatTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[0]));
            var target = ctrl.floatTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[1]));
            var result = this.Forward(pred, target);
            return result.Id + "";
        }
        
        protected virtual string ProcessMessageAsLayerOrLoss (Command msgObj, SyftController ctrl)
        {
            return ProcessMessageAsLoss(msgObj, ctrl);
        }
		
        protected string ProcessMessageAsLoss(Command msgObj, SyftController ctrl) 
        {   
            return "Model.processMessage not Implemented:" + msgObj.functionCall;
        }
    }
}

