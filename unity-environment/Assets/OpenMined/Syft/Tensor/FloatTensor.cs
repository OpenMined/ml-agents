using System;
using System.Linq;
using UnityEngine;
using OpenMined.Network.Utils;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor.Factories;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor : BaseTensor<float>
    {

        private FloatTensorFactory factory;
        
        public bool Autograd
        {
            get { return autograd; }

            set { autograd = value; }
        }

        public FloatTensor()
        {
            // DON'T USE THIS CONSTRUCTOR - USE FACTORY INSTEAD.
            // factory.Create(all, my, params)
        }

        public void init(FloatTensorFactory _factory,
            int[] _shape,
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
            factory = _factory;
            dataOnGpu = _dataOnGpu;
            autograd = _autograd;
            keepgrads = _keepgrads;
            creation_op = _creation_op;

            InitGraph();
            
            if (autograd)
            {
                InitAutograd();
            }
            
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
                        _data = new float[acc];

                        InitCpu(_data, false);
                    }
                }
            }

			// Lastly: let's set the ID of the tensor.
			// IDEs might show a warning, but ref and volatile seems to be working with Interlocked API.
            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);

            //controller.addTensor(this);
            if (SystemInfo.supportsComputeShaders && shader == null)
            {
                shader = factory.GetShader();
            }
        }
        
        // a poorly designed hash function based on the shape and location (CPU/GPU) of this tensor
        // it's used primarily to help search for 
        public int ConfigShapeHash()
        {
            
            long hash = 0;
            if (DataOnGpu)
                hash += 31415;
            
            for (int i = 0; i < shape.Length; i++)
            {
                hash += ((hash * 314159) % (long)int.MaxValue + (long)shape[i]) % int.MaxValue;
                hash = hash % int.MaxValue;
            }
            
            return (int) hash;
        }
        
        // a reasonalbe hash based on size and data location
        // it's used primarily to help search for 
        public int ConfigSizeHash()
        {
            
            long hash = 0;
            if (DataOnGpu)
                hash += int.MaxValue / 2;
            hash += size;
            
            return (int) (hash % int.MaxValue);
        }


        override public string ProcessMessage(Command msgObj, SyftController ctrl)
        {
            switch (msgObj.functionCall)
            {
                case "abs":
                {
                    // calls the function on our tensor object
                    var result = this.Abs();
                    // returns the function call name with the OK status
                    return result.id + "";
                }
                case "abs_":
                {
                    // calls the function on our tensor object
                    this.Abs(inline: true);
                    // returns the function call name with the OK status
                    return id.ToString();
                }
                case "add_elem":
                {
                    Debug.LogFormat("add_elem");
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = this.Add(tensor_1);
                    return result.id + "";
                }
                case "acos":
                {
                    var result = Acos();
                    return result.Id.ToString();
                }
                case "acos_":
                {
                    Acos(inline: true);
                    return Id.ToString();
                }
                case "asin":
                {
                    var result = Asin();
                    return result.Id.ToString();
                }
                case "asin_":
                {
                    Asin(inline: true);
                    return Id.ToString();
                }
                case "atan":
                {
                    var result = Atan();
                    return result.Id.ToString();
                }
                case "atan_":
                {
                    Atan(inline: true);
                    return Id.ToString();
                }
                case "add_elem_":
                {
                    Debug.LogFormat("add_elem_");
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    this.Add(tensor_1, inline: true);
                    return this.id + "";
                }
                case "add_scalar":
                {
                    Debug.LogFormat("add_scalar");
                    FloatTensor result = (FloatTensor) Add(float.Parse(msgObj.tensorIndexParams[0]));
                    return result.Id + "";
                }
                case "add_scalar_":
                {
                    Debug.LogFormat("add_scalar_");
                    this.Add(float.Parse(msgObj.tensorIndexParams[0]), inline: true);
                    return this.id + "";
                }
                case "addmm_":
                {
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    var tensor_2 = factory.Get(int.Parse(msgObj.tensorIndexParams[1]));
                    AddMatrixMultiply(tensor_1, tensor_2);
                    return msgObj.functionCall + ": OK";
                }
                case "addmv_":
                {
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    var tensor_2 = factory.Get(int.Parse(msgObj.tensorIndexParams[1]));
                    AddMatrixVectorProduct(tensor_1, tensor_2);
                    return msgObj.functionCall + ": OK";
                }
                case "backward":
                {
                    if (msgObj.tensorIndexParams.Length > 0)
                    {
                        var grad = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                        Backward(grad);
                    }
                    else
                    {
                        Backward();
                    }
                    return "";
                }
                case "batchify":
                {
                    var dim = int.Parse(msgObj.tensorIndexParams[0]);
                    var batch_size = int.Parse(msgObj.tensorIndexParams[1]);
                    return string.Join(",", Batchify(dim, batch_size).Select(x => x.ToString()).ToArray());;
                }
                case "ceil":
                {
                    var result = this.Ceil();
                    return result.id + "";
                }
                case "ceil_":
                {
                    this.Ceil(inline: true);
                    return this.id + "";
                }
                case "clamp":
                {

                    float ? min = null;
                    float ? max = null; 

                    if (msgObj.tensorIndexParams[0]=="None")
                    {
                        min = null;  
                    }
                    else
                    {
                        min = float.Parse(msgObj.tensorIndexParams[0]);
                    }

                    if (msgObj.tensorIndexParams[1]=="None")
                    {
                        max = null;  
                    }
                    else
                    {
                        max = float.Parse(msgObj.tensorIndexParams[1]);
                    }

                    FloatTensor result = (FloatTensor) Clamp(min, max);

                    return result.Id.ToString();
                }

                case "contiguous":
                {
                    var result = Contiguous();
                    return result.Id.ToString();
                }
                case "copy":
                {
                    var result = Copy(autograd:this.Autograd);
                    return result.Id.ToString();
                }
                case "cos":
                {
                    var result = Cos();
                    return result.Id.ToString();
                }
                case "cos_":
                {
                    Cos(inline: true);
                    return Id.ToString();
                }
                case "cosh":
                {
                    var result = Cosh();
                    return result.Id.ToString();
                }
                case "Cosh_":
                {
                    Cosh(inline: true);
                    return Id.ToString();
                }
                case "cpu":
                {
                    Cpu();
                    return msgObj.functionCall + ": OK";
                }
                case "cumsum":
                {
                    int dim = int.Parse(msgObj.tensorIndexParams[0]);
                    return CumSum(dim: dim).Id.ToString();
                }
                case "div_elem":
                {
                    return this.Div(factory.Get(int.Parse(msgObj.tensorIndexParams[0]))).Id + "";
                }
                case "div_elem_":
                {
                    this.Div(factory.Get(int.Parse(msgObj.tensorIndexParams[0])), inline: true);
                    return this.id + "";
                }
                case "div_scalar":
                {
                    return Div(float.Parse(msgObj.tensorIndexParams[0])).Id + "";
                }
                case "div_scalar_":
                {
                    this.Div(float.Parse(msgObj.tensorIndexParams[0]), inline: true);
                    return this.id + "";
                }
                case "exp":
                {
                    return Exp().Id.ToString();
                }
                case "exp_":
                {
                    Exp(inline: true);
                    return Id.ToString();
                }
                case "expand":
                {
                    int[] new_dims = new int[msgObj.tensorIndexParams.Length];
                    for (int i = 0; i < msgObj.tensorIndexParams.Length; i++)
                    {
                        new_dims[i] = int.Parse(msgObj.tensorIndexParams[i]);
                    }
                    var result = Expand(new_dims);
                    return result.Id.ToString();
                }
                case "index_select":
                {
                    IntTensor indices = ctrl.intTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    int dim = int.Parse(msgObj.tensorIndexParams[1]);

                    var result = IndexSelect(indices: indices, dim:dim);
                    return result.Id.ToString();
                }
                case "index_add":
                {
                    IntTensor indices = ctrl.intTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    int dim = int.Parse(msgObj.tensorIndexParams[1]);
                    FloatTensor x = ctrl.floatTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[2]));
                    var result = IndexAdd(indices, dim, x,inline:false);
                    return result.Id.ToString();
                }
                case "index_add_":
                {
                    IntTensor indices = ctrl.intTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    int dim = int.Parse(msgObj.tensorIndexParams[1]);
                    FloatTensor x = ctrl.floatTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[2]));
                    var result = IndexAdd(indices, dim, x,inline:true);
                    return result.Id.ToString();
                }
                case "log":
                {
                    return Log().Id.ToString();
                }
                case "mul_scalar":
                {
                    return Mul(float.Parse(msgObj.tensorIndexParams[0])).id + "";
                }
                case "mul_scalar_":
                {
                    this.Mul(float.Parse(msgObj.tensorIndexParams[0]), inline: true);
                    return this.id + "";
                }
                case "pow_elem":
                {
                    return this.Pow(factory.Get(int.Parse(msgObj.tensorIndexParams[0]))).id + "";
                }
                case "floor":
                {
                    return this.Floor().id + "";
                }
                case "floor_":
                {
                    this.Floor(inline: true);
                    return this.id + "";
                }
                case "get":
                {
                    var param_to_get = msgObj.tensorIndexParams[0];
                    switch (param_to_get)
                    {
                        case "autograd":
                        {
                            if (this.autograd)
                                return "1";
                            return "0";
                        }
                        case "children":
                        {
                            if (children_indices != null)
                            {
                                string children_str = "";
                                foreach (int entry in children_indices)
                                {
                                    children_str += (entry + ",");
                                }
                                return children_str;
                            }
                            else
                            {
                                return "";
                            }
                        }
                        case "creation_op":
                        {
                            if (creation_op != null)
                                return creation_op;
                            return "";
                        }
                        case "creators":
                        {
                            if (creators != null)
                            {
                                string creators_str = "";
                                foreach (int entry_id in creators)
                                {
                                    FloatTensor entry = factory.Get((entry_id));
                                    creators_str += (entry.id + ",");
                                }
                                return creators_str;
                            }
                            else
                            {
                                return "";
                            }
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
                        case "grad":
                        {
                            if (Grad == null)
                                return "";
                            return Grad.id + "";
                        }
                        case "id":
                        {
                            return this.id + "";
                        }
                        case "keepgrads":
                        {
                            if (this.keepgrads)
                                return "1";
                            return "0";
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
                case "gpu":
                {
                    if (Gpu(ctrl.GetShader()))
                    {
                        return msgObj.functionCall + ": OK : Moved data to GPU.";
                    }
                    else
                    {
                        return msgObj.functionCall + ": FAILED : Did not move data.";
                    }
                }
                case "log1p":
                {
                    var result = Log1p();
                    return result.Id.ToString();
                }
                case "log1p_":
                {
                    Log1p(inline: true);
                    return Id.ToString();
                }
                case "mul_elem":
                {
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = this.Mul(tensor_1);

                    return result.id + "";
                }
                case "mul_elem_":
                {
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    this.Mul(tensor_1, inline: true);
                    return this.id + "";
                }
                case "mm":
                {
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = this.MM(tensor_1);
                    return result.id + "";
                }
                case "pow_elem_":
                {
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    this.Pow(tensor_1, inline: true);
                    return this.id + "";
                }
                case "pow_scalar":
                {
                    FloatTensor result = Pow(float.Parse(msgObj.tensorIndexParams[0]));
                    return result.id + "";
                }
                case "pow_scalar_":
                {
                    this.Pow(float.Parse(msgObj.tensorIndexParams[0]), inline: true);
                    return this.id + "";
                }
                case "random_":
                {
                    if (msgObj.tensorIndexParams.Length > 1)
                    {
                        var result = Random(shape, float.Parse(msgObj.tensorIndexParams[0]), float.Parse(msgObj.tensorIndexParams[1]), inline: true);
                        return result.Id.ToString();
                    }
                    else
                    {
                        var result = Random(shape, float.Parse(msgObj.tensorIndexParams[0]), inline: true);
                        return result.Id.ToString();
                    }
                }
                case "reciprocal":
                {
                    var result = Reciprocal();
                    return result.Id.ToString();
                }
                case "reciprocal_":
                {
                    Reciprocal(inline: true);
                    return Id.ToString();
                }
                case "remainder_elem":
                {
	                var divisor = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
	                FloatTensor result = Remainder(divisor);
	                return result.id + "";
                }
                case "remainder_elem_":
                {
	                var divisor = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
	                this.Remainder(divisor, inline: true);
	                return this.id + "";
                }
                case "remainder_scalar":
                {
	                FloatTensor result = Remainder(float.Parse(msgObj.tensorIndexParams[0]));
	                return result.id + "";
                }
                case "remainder_scalar_":
                {
	                this.Remainder(float.Parse(msgObj.tensorIndexParams[0]), inline: true);
	                return this.id + "";
                }
                case "round":
                {
                    var result = Round();
                    return result.Id.ToString();
                }
                case "round_":
                {
                    Round(inline: true);
                    return Id.ToString();
                }
                case "set":
                {
                    var param_to_set = msgObj.tensorIndexParams[0];
                    switch (param_to_set)
                    {
                        case "autograd":
                        {
                            if (msgObj.tensorIndexParams[1] == "1")
                            {
                                InitAutograd();
                                return "1";
                            }
                            else
                            {
                                autograd = false;
                                return "0";
                            }
                        }
                    }
                    return "setter not recognized";
                }
                case "sigmoid_":
                {
                    this.Sigmoid(inline: true);
                    return this.id + "";
                }
                case "sigmoid":
                {
                    var result = this.Sigmoid();
                    return result.id + "";
                }
                case "softmax":
                {
                    var dim = -1;
                    if (msgObj.tensorIndexParams.Length > 0)
                    {
                        dim = int.Parse(msgObj.tensorIndexParams[0]);
                    }
                    var result = Softmax(dim);
                    return result.id + "";
                }
                case "log_softmax":
                {
                    var dim = -1;
                    if (msgObj.tensorIndexParams.Length > 0)
                    {
                        dim = int.Parse(msgObj.tensorIndexParams[0]);
                    }
                    var result = LogSoftmax(dim);
                    return result.id + "";
                }
                case "sub_elem":
                {
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = this.Sub(tensor_1);

                    return result.Id + "";
                }
                case "sub_elem_":
                {
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    this.Sub(tensor_1, inline: true);
                    return this.id + "";
                }
                case "neg":
                {
                    var result = Neg();
                    return result.Id.ToString();
                }
                case "neg_":
                {
                    Neg(inline: true);
                    return Id.ToString();
                }
                case "relu":
                {
                    var result = ReLU();
                    return result.Id.ToString();
                }
                case "rsqrt":
                {
                    var result = Rsqrt();
                    return result.Id.ToString();
                }
                case "rsqrt_":
                {
                    Rsqrt(inline: true);
                    return Id.ToString();
                }
                case "print":
                {
                    bool dataOriginallyOnGpu = dataOnGpu;
                    if (dataOnGpu)
                    {
                        Cpu();
                    }

                    string data = this.Print();
                    Debug.LogFormat("<color=cyan>Print:</color> {0}", string.Join(",", this.Data));

                    if (dataOriginallyOnGpu)
                    {
                        Gpu(ctrl.GetShader());
                    }
                    return data;
                }
                case "sample":
                {
                    var result = this.Sample(int.Parse(msgObj.tensorIndexParams[0]));
                    return result.Id + "";
                }
                case "sign":
                {
                    Debug.LogFormat("sign");
                    var result = this.Sign();
                    return result.Id + "";
                }
                case "sign_":
                {
                    Debug.LogFormat("sign_");
                    Sign(inline: true);
                    return Id.ToString();
                }
                case "sin":
                {
                    var result = Sin();
                    return result.Id.ToString();
                }
                case "sin_":
                {
                    Sin(inline: true);
                    return Id.ToString();
                }
                case "sqrt":
                {
                    return Sqrt().id.ToString();
                }
                case "sqrt_":
                {
                    Sqrt(inline: true);
                    return Id.ToString();
                }
                case "shape":
                {
                    return ShapeAsTensor().id.ToString();
                }
                case "std":
                {
                    return Std(int.Parse(msgObj.tensorIndexParams[0])).Id + "";
                }
                case "sub_scalar":
                {
                    return Sub(float.Parse(msgObj.tensorIndexParams[0])).Id + "";
                }
                case "sub_scalar_":
                {
                    this.Sub(float.Parse(msgObj.tensorIndexParams[0]), inline: true);
                    return this.id + "";
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
                case "tan":
                {
                    var result = Tan();
                    return result.Id.ToString();
                }
                case "tan_":
                {
                    Tan(inline: true);
                    return Id.ToString();
                }
                case "tanh":
                {
                    var result = Tanh();
                    return result.Id.ToString();
                }
                case "save":
                {
                    return WriteToFile(msgObj.tensorIndexParams[0]) + "";
                }
                case "sinh":
                {
                    var result = Sinh();
                    return result.Id.ToString();
                }
                case "sinh_":
                {
                    Sinh(inline: true);
                    return Id.ToString();
                }
                case "trace":
                {
                    var result = Trace();
                    return result.ToString();
                }
                case "transpose":
                {
                    var result = Transpose();
                    return result.Id.ToString();
                }
                case "transpose_":
                {
                    this.Transpose(inline: true);
                    return Id.ToString();
                }
                case "triu":
                {
                    var K = int.Parse(msgObj.tensorIndexParams[0]);
                    var result = Copy(autograd:this.Autograd);
                    result.Triu_(K);
                    return result.Id.ToString();
                }
                case "triu_":
                {
                    var K = int.Parse(msgObj.tensorIndexParams[0]);
                    Triu_(K);
                    return Id.ToString();
                }

                case "trunc":
                {
                    var result = Trunc();
                    return result.Id.ToString();
                }
                case "uniform_":
                {
                    var result = Uniform(shape, start:float.Parse(msgObj.tensorIndexParams[0]), to:float.Parse(msgObj.tensorIndexParams[1]), inline: true);
                    return result.Id.ToString();
                }
                case "unsqueeze":
                {
                    var result = Unsqueeze(int.Parse(msgObj.tensorIndexParams[0]), inline:false);
                    return result.Id.ToString();
                }
                case "unsqueeze_":
                {
                    var result = Unsqueeze(int.Parse(msgObj.tensorIndexParams[0]),inline:true);
                    return result.Id.ToString();
                }
                case "view":
                {
                    int[] new_dims = new int[msgObj.tensorIndexParams.Length];
                    for (int i = 0; i < msgObj.tensorIndexParams.Length; i++)
                    {
                        new_dims[i] = int.Parse(msgObj.tensorIndexParams[i]);
                    }
                    var result = View(new_dims);
                    return result.Id.ToString();
                }

                case "view_":
                {
                    int[] new_dims = new int[msgObj.tensorIndexParams.Length];
                    for (int i = 0; i < msgObj.tensorIndexParams.Length; i++)
                    {
                        new_dims[i] = int.Parse(msgObj.tensorIndexParams[i]);
                    }
                    View(new_dims, inline: true);
                    return Id.ToString();
                }
                case "view_as":
                {
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = ViewAs(tensor_1, false);
                    return result.Id.ToString();
                }

                case "view_as_":
                {
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    this.ViewAs(tensor_1, true);
                    return Id.ToString();
                }
                case "zero_":
                {
                    Zero_();
                    return msgObj.functionCall + ": OK";
                }
                case "is_contiguous":
                {
                    return Convert.ToString(IsContiguous());
                }
                case "squeeze":
                {
                    if (msgObj.tensorIndexParams.Length > 0)
                    {
                        int dim = int.Parse(msgObj.tensorIndexParams[0]);
                        var result = Squeeze(dim: dim);
                        //ctrl.addTensor(result);
                        return result.Id.ToString();
                    }
                    else
                    {
                        var result = Squeeze();
                        //ctrl.addTensor(result);
                        return result.Id.ToString();
                    }
                }
                case "sqeeze_":
                {
                    if (msgObj.tensorIndexParams.Length > 0)
                    {
                        int dim = int.Parse(msgObj.tensorIndexParams[0]);
                        Squeeze(dim: dim, inline: true);
                        return Id.ToString();
                    }
                    else
                    {
                        Squeeze(inline: true);
                        return Id.ToString();
                    }
                }
                case "min":
                {
                    int dim = -1;
                    bool keepdim = false;

                    if (msgObj.tensorIndexParams.Length > 0)
                    {
                        dim = int.Parse(msgObj.tensorIndexParams[0]);
                        keepdim = bool.Parse(msgObj.tensorIndexParams[1]);
                    }

                    return Min(dim: dim, keepdim: keepdim).Id.ToString();
                }
                case "max":
                {
                    int dim = -1;
                    bool keepdim = false;

                    if (msgObj.tensorIndexParams.Length > 0)
                    {
                        dim = int.Parse(msgObj.tensorIndexParams[0]);
                        keepdim = bool.Parse(msgObj.tensorIndexParams[1]);
                    }

                    return Max(dim: dim, keepdim: keepdim).Id.ToString();
                }
                case "sum":
                {
                    int dim = -1;
                    bool keepdim = false;

                    if (msgObj.tensorIndexParams.Length > 0)
                    {
                        dim = int.Parse(msgObj.tensorIndexParams[0]);
                        keepdim = bool.Parse(msgObj.tensorIndexParams[1]);
                    }

            
                    return  Sum(dim: dim, keepdim: keepdim).Id.ToString();
                }
                case "prod":
                {
                    int dim = -1;
                    bool keepdim = false;

                    if (msgObj.tensorIndexParams.Length > 0)
                    {
                        dim = int.Parse(msgObj.tensorIndexParams[0]);
                        keepdim = bool.Parse(msgObj.tensorIndexParams[1]);
                    }

                    return Prod(dim: dim, keepdim: keepdim).Id.ToString();
                }
                case "mean":
                {
                    int dim = -1;
                    bool keepdim = false;

                    if (msgObj.tensorIndexParams.Length > 0)
                    {
                        dim = int.Parse(msgObj.tensorIndexParams[0]);
                        keepdim = bool.Parse(msgObj.tensorIndexParams[1]);
                    }

                    return Mean(dim: dim, keepdim: keepdim).Id.ToString();
                }
                case "stride": 
                {
                    if (msgObj.tensorIndexParams.Length > 0) {
                        var dim = int.Parse(msgObj.tensorIndexParams[0]);
                        return Strides[dim].ToString();
                    } else {
                        return string.Join(" ", Strides);
                    }
                    
                }
                default:
                    break;
            }
            return "FloatTensor.processMessage: Command not found:" + msgObj.functionCall;
        }

        public string Print()
        {
            bool dataOriginallyOnGpu = dataOnGpu;
            ComputeShader _shader = this.shader;

            if (dataOnGpu)
            {
                Cpu();
            }

            string print = "";

            if (shape.Length > 3)
                print += "Only printing the last 3 dimesnions\n";
            int d3 = 1;
            if (shape.Length > 2)
                d3 = shape[shape.Length - 3];
            int d2 = 1;
            if (shape.Length > 1)
                d2 = shape[shape.Length - 2];
            int d1 = shape[shape.Length - 1];

            for (int k = 0; k < d3; k++)
            {
                for (int j = 0; j < d2; j++)
                {
                    for (int i = 0; i < d1; i++)
                    {
                        float f = this[i + j * d1 + k * d1 * d2];
                        print += f.ToString("0.0000") + ", ";
                    }
                    print += "\n";
                }
                print += "\n";
            }

            if (dataOriginallyOnGpu)
            {
                Gpu(_shader);
            }
            return print;
        }
    }
}