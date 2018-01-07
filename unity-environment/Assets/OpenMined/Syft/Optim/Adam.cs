using System;
using System.Collections.Generic;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using UnityEngine;


namespace OpenMined.Syft.Optim
{
    public class Adam: Optimizer
    {
        private float beta1;
        private float beta2;
        private float epsilon;
        private int t;
        private List<int> velocities;
        private List<int> squares;

        public Adam(SyftController ctrl_, List<int> parameters_, float lr_, float beta1_, float beta2_, float epsilon_, float decay_)
        {
            this.ctrl = ctrl_;
            this.parameters = parameters_;
            this.lr = lr_;
            this.beta1 = beta1_;
            this.beta2 = beta2_;
            this.epsilon = epsilon_;
            this.decay = decay_;
            this.t = 0;
            this.velocities = new List<int>();
            this.squares = new List<int>();

            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            ctrl.addOptimizer(this);
            
            foreach (int param_index in parameters)
            {
                var param = ctrl.floatTensorFactory.Get(param_index);
                
                var velInit = param.createZerosTensorLike();
                velocities.Add(velInit.Id);

                var sInit = param.createZerosTensorLike();
                squares.Add(sInit.Id);
            }
        }

        public override void Step(int batch_size, int iteration)
        {    
            t++;        
            for (int i = 0; i < parameters.Count; i++)
            {
                var param = ctrl.floatTensorFactory.Get(parameters[i]);
                var v = ctrl.floatTensorFactory.Get(velocities[i]);
                var s = ctrl.floatTensorFactory.Get(squares[i]);
                
                v.Mul(beta1, inline: true).Add(param.Grad.Mul(1.0F - beta1), inline: true);
                var vCorrected = v.Div(1.0F - (float)Math.Pow(beta1, t));

                s.Mul(beta2, inline: true).Add(param.Grad.Pow(2.0F).Mul(1.0F - beta2), inline: true);
                var sCorrected = s.Div(1.0F - (float)Math.Pow(beta2, t));

                var update = vCorrected.Mul(sCorrected.Sqrt().Add(epsilon));
                param.Sub(update.Mul(lr/(float)batch_size), inline: true);
            }

            // Adjust learning rate with decay
            if (this.decay > 0)
            {
                this.lr *= 1.0F / (1.0F + this.decay * iteration);
            }
        }
    }
}
