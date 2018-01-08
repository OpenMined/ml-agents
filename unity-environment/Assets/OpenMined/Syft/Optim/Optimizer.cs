using System;
using System.Collections.Generic;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using UnityEngine;


namespace OpenMined.Syft.Optim
{
    public abstract class Optimizer
    {
        protected SyftController ctrl;
        protected List<int> parameters;
        protected float lr;
        protected float decay;
        
        // Should we put a check incase this variable overflows?
        protected static volatile int nCreated = 0;
        protected int id;
        
        public int Id
        {
            get { return id; }
            protected set { id = value; }
        }

        public void ZeroGrad()
        {
            foreach (int param_index in parameters)
                if(ctrl.floatTensorFactory.Get(param_index).Grad != null)
                    ctrl.floatTensorFactory.Get(param_index).Grad.Zero_();
        }

        public string ProcessMessage(Command msgObj, SyftController ctrl)
        {
            switch (msgObj.functionCall)
            {
                case "zero_grad":
                    ZeroGrad();
                    return "";
                case "step":
                    Step(int.Parse(msgObj.tensorIndexParams[0]), int.Parse(msgObj.tensorIndexParams[1]));
                    return "";
            }
            throw new InvalidOperationException("Could not find function for command:" + msgObj.functionCall);
        }

        public abstract void Step(int batch_size, int iteration);
    }
}