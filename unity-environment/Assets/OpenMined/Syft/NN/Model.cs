using System;
using System.Collections;
using System.Collections.Generic;
using JetBrains.Annotations;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.Layer
{
    public abstract class Model
    {
        
        protected static volatile int nCreated = 0;
        
        // unique identifier held by SyftController
        protected int id;
        public int Id => id;

        // indices for weights used in forward prediction (not inluding those in models array)
        protected List<int> parameters;
        
        // Model component type which includes layers and losses.
        protected string model_type;

        protected int activation;
        
        protected SyftController controller;

        protected void init(string model_type)
        {
            activation = -1;
            parameters = new List<int>();
            this.model_type = model_type;
        }
        
        public string getLayerType()
        {
            return model_type;
        }
        
        public int getParameter(int i)
        {
            if(i > 0 && i < parameters.Count) 
                return parameters[i];
            throw new ArgumentOutOfRangeException("Parameter " + i + " does not exist.");
        }
        
        public List<int> getParameters()
        {
            return parameters;
        }

        public abstract int getParameterCount();
        
        public string ProcessMessage (Command msgObj, SyftController ctrl)
        {

            switch (msgObj.functionCall)
            {
                case "forward":
                    return ProcessForwardMessage(msgObj, ctrl);
                case "params":
                    return ProcessParamsMessage(msgObj, ctrl);
                case "param_count":
                    return getParameterCount() + "";
                case "activation":
                {
                    return activation + "";
                }
                case "model_type":
                {
                    return model_type;
                }
                case "zero_grad":
                {
                    ProcessZeroGradMessage();
                    return "";
                }

                case "set_id":
                {
                    int new_id = int.Parse(msgObj.tensorIndexParams[0]);
                    this.controller.setModelId(this.Id,new_id);
                    this.id = new_id;
                    return "";
                }
        
            }

            return ProcessMessageAsLayerOrLoss(msgObj, ctrl);

        }

        public int[] GetParameters()
        {
            return parameters.ToArray();
        }

        protected virtual string ProcessMessageAsLayerOrLoss (Command msgObj, SyftController ctrl) 
        {   
            return "Model.processMessage not Implemented:" + msgObj.functionCall;
        }

        protected virtual string ProcessParamsMessage (Command msgObj, SyftController ctrl) 
        {   
            string out_str = "";

            for (int i = 0; i < parameters.Count; i++)
            {

                out_str += parameters[i].ToString() + ",";

            }
            return out_str;
        }
        
        public void ProcessZeroGradMessage()
        {
            foreach (int param_index in parameters)
            {
                if(controller.floatTensorFactory.Get(param_index).Grad != null)
                {
                    controller.floatTensorFactory.Get(param_index).Grad.Zero_();
                }
            }
        }

        protected abstract string ProcessForwardMessage (Command msgObj, SyftController ctrl);

    }
}