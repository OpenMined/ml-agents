using System;
using System.Collections.Generic;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using UnityEngine;


namespace OpenMined.Syft.Optim
{
    public class RMSProp: Optimizer
    {
        private float rho;
        private float epsilon;
        private List<int> squares;

        public RMSProp(SyftController ctrl_, List<int> parameters_, float lr_, float rho_, float epsilon_, float decay_)
        {
            this.ctrl = ctrl_;
            this.parameters = parameters_;
            this.lr = lr_;
            this.rho = rho_;
            this.epsilon = epsilon_;
            this.decay = decay_;
            this.squares = new List<int>();

            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            ctrl.addOptimizer(this);

            foreach (int param_index in parameters)
            {
                var param = ctrl.floatTensorFactory.Get(param_index);
                var sInit = param.createZerosTensorLike();
                this.squares.Add(sInit.Id);
            }
            Debug.LogFormat("<color=green>RMSProp Step: lr: {0} rho: {1} ep: {2} decay: {3}</color>", lr, rho, epsilon, decay);
        }

        override public void Step(int batch_size, int iteration)
        {
            for (int i = 0; i < parameters.Count; i++)
            {
                var param = ctrl.floatTensorFactory.Get(parameters[i]);
                var s = ctrl.floatTensorFactory.Get(squares[i]);
                s.Mul(rho, inline: true).Add(param.Grad.Pow(2.0F).Mul(1.0F - rho), inline: true);

                var update = s.Div(s.Sqrt().Add(epsilon));
                param.Sub(update.Mul(lr/(float)batch_size), inline: true);
            }

            if (this.decay > 0)
            {
                this.lr *= 1.0F / (1.0F + this.decay * iteration);
            }
        }
    }
}
