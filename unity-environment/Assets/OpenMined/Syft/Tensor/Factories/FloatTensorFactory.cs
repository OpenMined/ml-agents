using System;
using System.Collections.Generic;
using OpenMined.Network.Controllers;
using UnityEngine;

namespace OpenMined.Syft.Tensor.Factories
{
    
    public class FloatTensorFactory
    {
        private Dictionary<int, FloatTensor> tensors;
        private ComputeShader shader;
        public SyftController ctrl;

        public FloatTensorFactory(ComputeShader _shader, SyftController _ctrl)
        {
            shader = _shader;
            ctrl = _ctrl;
            tensors = new Dictionary<int, FloatTensor>();
        }
        
        public FloatTensor Get(int id)
        {
            return tensors[id];
        }

        public int Count()
        {
            return tensors.Count;
        }
        
        public void Delete(int id)
        {
            //Debug.LogFormat("<color=purple>Removing Tensor {0}</color>", id);

            var tensor = tensors [id];
            
            if(tensor.Grad != null)
                this.Delete(tensor.Grad.Id);
            
            tensors.Remove (id);
            tensor.Dispose();

        }
        
        // create from file
        public FloatTensor Create(string filepath,
            ComputeShader _shader = null,
            bool _dataOnGpu = false,
            bool _autograd = false,
            bool _keepgrads = false,
            string _creation_op = null)
        {
            Tuple<int[],float[]> shape_data = FloatTensor.ReadFromFile(filepath);
            return Create(_shape: shape_data.Item1,
                _data: shape_data.Item2,
                _copyData: false,
                _dataOnGpu: false,
                _autograd: _autograd,
                _keepgrads: _keepgrads,
                _creation_op: "read_from_file");
        }
        
        public FloatTensor Create(int[] _shape,
            float[] _data = null,
            ComputeBuffer _dataBuffer = null,
            ComputeBuffer _shapeBuffer = null,
            ComputeShader _shader = null,
            bool _copyData = true,
            bool _dataOnGpu = false,
            bool _autograd = false,
            bool _keepgrads = false,
            string _creation_op = null)
        {

            if (ctrl.allow_new_tensors)
            {

                FloatTensor tensor = new FloatTensor();

                tensor.init(this,
                    _shape,
                    _data,
                    _dataBuffer,
                    _shapeBuffer,
                    _shader,
                    _copyData,
                    _dataOnGpu,
                    _autograd,
                    _keepgrads,
                    _creation_op);

                tensors.Add(tensor.Id, tensor);

                return tensor;
            }
            else
            {
                throw new Exception("Attempted to Create a new FloatTensor");
            }

        }
       
        public ComputeShader GetShader()
        {
            return shader;
        }
    }
}