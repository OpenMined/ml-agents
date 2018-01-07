using System;
using UnityEngine;
using OpenMined.Network.Utils;
using OpenMined.Network.Controllers;
using System.Collections.Generic;
using OpenMined.Syft.Tensor.Factories;

namespace OpenMined.Syft.Tensor
{
    public partial class IntTensor : BaseTensor<int>
    {
        private List<int> creators;
        private string creation_op;
        public List<int> children_indices; // children -> counts
        public List<int> children_counts; // children -> counts
        private int sibling;

        private IntTensorFactory factory;


        public IntTensor()
        {
            // DON'T USE THIS CONSTRUCTOR - USE FACTORY INSTEAD.
            // factory.Create(all, my, params)
        }

        public void init(IntTensorFactory _factory,
            int[] _shape,
            int[] _data = null,
            ComputeBuffer _dataBuffer = null,
            ComputeBuffer _shapeBuffer = null,
            ComputeShader _shader = null,
            bool _copyData = true,
            bool _dataOnGpu = false,
            bool _autograd = false,
            bool _keepgrads = false,
            string _creation_op = null)
        {
            factory = _factory;
            dataOnGpu = _dataOnGpu;
            creation_op = _creation_op;


            // First: check that shape is valid.
            if (_shape == null || _shape.Length == 0)
            {
                throw new InvalidOperationException("Tensor shape can't be an empty array.");
            }

            // Second: since shape is valid, let's save it
            shape = (int[]) _shape.Clone();

            setStridesAndCheckShape();

            // Third: let's see what kind of data we've got. We should either have
            // a GPU ComputeBuffer or a data[] object.
            if (_data != null && _shapeBuffer == null && _dataBuffer == null)
            {
                InitCpu(_data: _data, _copyData: _copyData);
            }
            else if (_dataBuffer != null && _shapeBuffer != null && SystemInfo.supportsComputeShaders && _data == null)
            {
                // looks like we have GPU data being passed in... initialize a GPU tensor.

                InitGpu(_shader, _dataBuffer, _shapeBuffer, _copyData);
                initShaderKernels();
            }
            else
            {
                // no data seems to be passed in... or its got missing stuff

                // if CPU works... go with that
                if (_data != null)
                {
                    InitCpu(_data, _copyData);
                }
                else if (_dataBuffer != null && _shader != null)
                {
                    if (SystemInfo.supportsComputeShaders)
                    {
                        // seems i'm just missing a shape buffer - no biggie
                        shapeBuffer = new ComputeBuffer(shape.Length, sizeof(int));
                        shapeBuffer.SetData(shape);

                        InitGpu(_shader, _dataBuffer, _shapeBuffer, _copyData);
                        initShaderKernels();
                    }
                    else
                    {
                        throw new InvalidOperationException(
                            "You seem to be trying to create a GPU tensor without having access to a GPU...");
                    }
                }
                else
                {
                    // nothing else seems to work - i suppose i'm just supposed to initialize an empty tensor.
                    long acc = 1;
                    for (var i = shape.Length - 1; i >= 0; --i)
                    {
                        acc *= shape[i];
                    }

                    if (_dataOnGpu)
                    {
                        _shapeBuffer = new ComputeBuffer(shape.Length, sizeof(int));
                        _shapeBuffer.SetData(shape);

                        _dataBuffer = new ComputeBuffer(size, sizeof(float));

                        InitGpu(_shader: _shader, _dataBuffer: _dataBuffer, _shapeBuffer: _shapeBuffer,
                            _copyData: false);
                        initShaderKernels();
                        this.Zero_();
                    }
                    else
                    {
                        _data = new int[acc];

                        InitCpu(_data, false);
                    }
                }
            }

            // Lastly: let's set the ID of the tensor.
            // IDEs might show a warning, but ref and volatile seems to be working with Interlocked API.

            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);

            if (SystemInfo.supportsComputeShaders && shader == null)
            {
                shader = factory.GetShader();
            }

        }

        public void initShaderKernels()
        {
            // TODO: move to IntegerTensor.ShaderOps.cs (doesn't exist yet)
            throw new NotImplementedException();
        }


        public IntTensor Copy()
        {
            throw new NotImplementedException();
        }

        public IntTensor Add(int value, bool inline)
        {
            throw new NotImplementedException();
        }

        public IntTensor View(int[] new_shape, bool inline = true, FloatTensor result = null)
        {
            if (!IsContiguous()) {
                throw new InvalidOperationException ("Tensor must be contiguous, call Contiguous() to convert");
            }
            if (inline == true)
            {
                
                this.Shape = new_shape;

                if (dataOnGpu)
                {
                    shapeBuffer.Release();
                    shapeBuffer = new ComputeBuffer(shape.Length, sizeof(int));
                    shapeBuffer.SetData(shape);
                    
                }
                
                setStridesAndCheckShape();

                return this;

            }
            else
            {
                throw new NotImplementedException();
            }
            
        }
        
        public override string ProcessMessage(Command msgObj, SyftController ctrl)
        {
            switch (msgObj.functionCall)
            {
                case "get":
                {
                    var param_to_get = msgObj.tensorIndexParams[0];
                    switch (param_to_get)
                    {
                        case "creation_op":
                        {
                            if (creation_op != null)
                                return creation_op;
                            return "";
                        }
                        case "data":
                        {
                            string out_str = "";

                            if (dataOnGpu)
                            {
                                int[] temp_data = new int[size];
                                dataBuffer.GetData(temp_data);
                                for (int i = 0; i < size; i++)
                                {
                                    out_str += temp_data[i] + ",";
                                }
                            }
                            else
                            {
                                for (int i = 0; i < size; i++)
                                {
                                    out_str += this[i] + ",";
                                }
                            }

                            return out_str;
                        }
                        case "dataOnGpu":
                        {
                            if (dataOnGpu)
                            {
                                return "1";
                            }
                            return "0";
                        }
                        case "id":
                        {
                            return this.id + "";
                        }
                        case "size":
                        {
                            return this.size + "";
                        }
                        case "shape":
                        {
                            string shape_str = "";
                            for (int i = 0; i < shape.Length; i++)
                            {
                                shape_str += (shape[i] + ",");
                            }
                            return shape_str;
                        }
                    }
                    return "param not found or not configured with a getter";
                }
                    
                case "to_numpy":
                {
                    if (DataOnGpu)
                    {
                        var tmpData = new float[size];
                        dataBuffer.GetData(tmpData);
                        return string.Join(" ", tmpData);

                    } else
                    {
                        return string.Join(" ", Data);

                    }
                }

            }
            return "IntTensor.processMessage: Command not found:" + msgObj.functionCall;
        }

    }
}
