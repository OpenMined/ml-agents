using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using OpenMined.Syft.Tensor;
using OpenMined.Network.Utils;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Optim;

namespace OpenMined.Syft.Layer
{
    public abstract class Layer: Model
    {
        private FloatTensor cached_ones_grad_for_backprop;
        private Loss.Loss _criterion;
        private Optimizer _optimizer;
        private int _batch_size;
        private int _input_batch_offset;
        private int _target_batch_offset;

        private FloatTensor _input_tensor_origin;
        private FloatTensor _target_tensor_origin;
        
        private FloatTensor[] _input_tensors;
        private FloatTensor[] _target_tensors;

        private List<FloatTensor> input_buffers;
        private List<FloatTensor> target_buffers;

        private FloatTensor last_input_buffer;
        private FloatTensor last_target_buffer;
        
        public abstract FloatTensor Forward (FloatTensor input);
        
        protected override string ProcessForwardMessage(Command msgObj, SyftController ctrl)
        {
            var input = ctrl.floatTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[0]));
            if (input.Autograd)
            {
                var result = this.Forward(input);
                return result.Id + "";
            }
            else
            {
                throw new Exception("Input to Model object must have autograd == true but autograd == false!!!");
            }
        }

        public int PrepareToFit(FloatTensor input, FloatTensor target, Loss.Loss criterion, Optimizer optimizer, int batch_size)
        {

            if(input.Shape[0] != target.Shape[0])
                throw new InvalidDataException("Input and Target tensors don't seem to have the right dims");
            
            _input_tensor_origin = input;
            _target_tensor_origin = target;
            
            int[] input_buffer_shape = new int[input.Shape.Length];
            input_buffer_shape[0] = batch_size;
            for (int i = 1; i < input.Shape.Length; i++) input_buffer_shape[i] = input.Shape[i];

            last_input_buffer = controller.floatTensorFactory.Create(_shape: input_buffer_shape, _autograd:true);
            
            int[] target_buffer_shape = new int[target.Shape.Length];
            target_buffer_shape[0] = batch_size;
            for (int i = 1; i < target.Shape.Length; i++) target_buffer_shape[i] = target.Shape[i];
            
            last_target_buffer = controller.floatTensorFactory.Create(_shape: target_buffer_shape, _autograd:true);
            
            this._batch_size = batch_size;
            this._criterion = criterion;
            this._optimizer = optimizer;

            this._input_batch_offset = batch_size;
            for (int i = 1; i < input.Shape.Length; i++)
                this._input_batch_offset *= input.Shape[i];
            
            this._target_batch_offset = batch_size;
            for (int i = 1; i < target.Shape.Length; i++)
                this._target_batch_offset *= target.Shape[i];
            
            return (int)(input.Shape[0] / batch_size);
        }
        
        public string Fit(int start_batch_id, int end_batch_id, int iter = 1)
        {
            float loss = 0;
            for (int i = 0; i < iter; i++)
            {
                for (int batch_i = start_batch_id; batch_i < end_batch_id; batch_i++)
                {
                    loss += FitBatch(batch_i, i + 1);       
                }
            }
            
            return (loss/iter).ToString();
        }

        public float FitBatch(int batch_i, int iteration)
        {
            if (((batch_i + 1) * _input_batch_offset) < _input_tensor_origin.Size)
            {
                last_input_buffer.Fill(_input_tensor_origin, starting_offset: batch_i * _input_batch_offset,
                    length_to_fill: _input_batch_offset);
                last_target_buffer.Fill(_target_tensor_origin, starting_offset: batch_i * _target_batch_offset,
                    length_to_fill: _target_batch_offset);
                
                var pred = Forward(last_input_buffer);
                var loss = _criterion.Forward(pred, last_target_buffer);

                if (cached_ones_grad_for_backprop == null || cached_ones_grad_for_backprop.Size != loss.Size)
                {
                    cached_ones_grad_for_backprop = loss.createOnesTensorLike();
                    cached_ones_grad_for_backprop.Autograd = false;
                }
                
                loss.Backward(cached_ones_grad_for_backprop);

                _optimizer.Step(this.last_input_buffer.Shape[0], iteration);

                return loss.Data[0];
            }
            else
            {
                return 0;
            }

        }
        
        protected override string ProcessMessageAsLayerOrLoss (Command msgObj, SyftController ctrl)
        {
            
            switch (msgObj.functionCall)
            {
                case "prepare_to_fit":
                {
                    FloatTensor input = ctrl.floatTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    FloatTensor target = ctrl.floatTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[1]));
                    Loss.Loss criterion = ctrl.getLoss(int.Parse(msgObj.tensorIndexParams[2]));
                    Optimizer optim = ctrl.getOptimizer(int.Parse(msgObj.tensorIndexParams[3]));
                    int batch_size = int.Parse(msgObj.tensorIndexParams[4]);
                    
                    return PrepareToFit(input,target,criterion,optim,batch_size).ToString();
                }
                
                case "fit":
                {
                    int start_batch_id = int.Parse(msgObj.tensorIndexParams[0]);
                    int end_batch_id = int.Parse(msgObj.tensorIndexParams[1]);
                    int iters = int.Parse(msgObj.tensorIndexParams[2]);

                    return Fit(start_batch_id, end_batch_id, iters);
                }
            }

            return ProcessMessageAsLayerObject(msgObj, ctrl);
        }
		
        protected virtual string ProcessMessageAsLayerObject (Command msgObj, SyftController ctrl) 
        {   
            return "Model.processMessage not Implemented:" + msgObj.functionCall;
        }
    }
}

