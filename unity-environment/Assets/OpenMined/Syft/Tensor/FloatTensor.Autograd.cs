using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {

	    private bool autograd;
	    public FloatTensor Grad { get; private set; }
	    private bool keepgrads;
	    
	    // checks to see if a variable has accumulated all the gradients it should before it backprops
	    public bool AllAutogradChildrenAccountedFor()
	    {
		    for (int i=0; i< children_counts.Count; i++)
		    {
			    if (children_counts[i] == 0 && factory.Get(children_indices[i]).autograd)
			    {
				    return false;
			    }
		    }
		    return true;
	    }
	    
	    public void Backward(FloatTensor grad = null, FloatTensor grad_origin = null)
	    {
		    //Debug.Log("Backward:" + this.id + " creation_op:" + creation_op);
		  
		    if (autograd)
		    {
			    
			    if (grad == null)
			    {
				    Debug.Log("Grad not Found... Creating Gradient of 1s");
				    grad = this.createOnesTensorLike();
				    grad.Autograd = false;
			    }

			    if (grad_origin != null)
			    {
				    int child_index = children_indices.IndexOf(grad_origin.Id);
				    if (children_counts[child_index] > 0)
				    {
					    throw new InvalidOperationException("Can't backprop more than once.");
				    }
				    else
				    {
					    children_counts[child_index] += 1;
				    }
			    }

			    if (this.Grad == null)
			    {
				    this.Grad = grad;
				    //Debug.Log("Setting Grad Tensor Id:" + this.id);
			    }
			    else
			    {
				    if (this.Grad.id == grad.id)
				    {
					    // do nothing
					    //Debug.Log("Not Updating For Tensor Id:" + this.id);
				    }
				    else
				    {
					    //Debug.Log("Updating For Tensor Id:" + this.id);
					    //this.Grad.Zero_();
					    this.Grad.Add(grad, inline: true);
				    }

			    }

			    // grads must not have grads of their own
			    if (this.Grad.autograd == true)
			    {
				    throw new InvalidOperationException("Sorry, grads cannot have grads");
			    }
			    
				// RULES FOR AUTOGRAD:
			    // 1) if you need to use "this" for calculating a gradient, copy it first and set autograd to false (see sigmoid)
			    // 2) if you use a method in your backprop logic that doesn't hook into the dynamic graph yet, backprop
			    // will not work!!! Make sure there's a "hookautograd" function in every method you use for backprop.
			    // 3) whenever backpropping into a method where the forward prop involved a scalar (such as scalar
			    // multiplication), current implementations assume you will NOT backprop into the scalar itself.
			    // 4) Because of rule (2), do NOT use "emptyTensorCopy" at all in backprop unless you know what you're
			    // doing. 
			    // 5) I will be especially strict about Unit tests for all backprop logic as this is the most complex
			    // piece of functionality we have. Furthermore, most errors go completely undetected (not discovered
			    // by runtime errors). Autograd bugs just make convergence go slowly and sub-optimally.
			    // 6) If you use a forward propagation tensor to backprop, you MUST remember to turn off autograd
			    // when backpropagating (see "mm" below for example). Otherwise, it will cause autograd to break because
			    // whatever child you select will think it needs to wait for another gradient before backpropagating.
			    // 7) In the "view" backprop method, you'll notice that we set parent.grad = null. This keeps grads from
			    // accumulating when forward and backprop is called multiple times. However, it doesn't cause any new 
			    // memory allocation.
			    
			    // only continue backpropping if there's something to backprop into
			    // only continue backpropping if all gradients (from children) are accounted for
			    // override waiting for children if "backprop" was called on this variable directly
			    if (this.creators != null && this.creators.Count > 0 && (grad_origin == null || AllAutogradChildrenAccountedFor()))
			    {
                    if (creation_op == "abs")
                    {
                        FloatTensor c = this.Copy(autograd:false);

                        var parent = factory.Get(creators[0]);

                        parent.Backward(parent.Div(c).Mul(grad));
                    }
                    else if (creation_op == "add_elem")
                    {

                        factory.Get(creators[0]).Backward(grad, this);
                        factory.Get(creators[1]).Backward(grad, this);

                    }
                    else if (creation_op == "add_scalar")
                    {
                        factory.Get(creators[0]).Backward(grad, this);
                    }
                    else if (creation_op.Contains("concatenate_"))
                    {
	                    int dim = int.Parse(creation_op.Split('_')[1]);
	                    
	                    for (int i = 0; i < creators.Count; i++)
	                    {
		                    FloatTensor slice = grad.IndexSelect(factory.ctrl.intTensorFactory.Get(int_creators[i]),dim);
		                    
		                    factory.Get(creators[i]).Backward(slice);
	                    }
	                    
                    }
                    else if (creation_op == "contiguous")
                    {
	                    //Debug.Log("Contiguous Backpropping Grad:" + grad.Id);
	                    //Debug.Log("Contiguous Storing Grad:" + this.Grad.Id);
                        factory.Get(creators[0]).Backward(this.Grad.Copy(autograd:this.Grad.Autograd), this);
                    }
                    else if (creation_op == "copy")
                    {
                        factory.Get(creators[0]).Backward(grad, this);
                    }
                    else if (creation_op == "div_elem")
                    {
                        FloatTensor x = factory.Get(creators[0]);
                        FloatTensor y = factory.Get(creators[1]);

                        x.Backward(grad.Div(y));

                        FloatTensor y2 = y.Pow(2);
                        FloatTensor xn = x.Neg();
                        FloatTensor xny2 = xn.Div(y2);
                        FloatTensor gradxny2 = grad.Mul(xny2);
                        y.Backward(gradxny2);
                    }
                    else if (creation_op == "div_scalar")
                    {
                        factory.Get(creators[0]).Backward(grad.Div(factory.Get(creators[1]).data[0]), this);
                    }
                    else if (creation_op == "emptyTensorCopy_Hooked")
                    {
	                    factory.Get(creators[0]).Backward(grad, this);
                    }
                    else if (creation_op == "expand")
                    {
                        var parent = factory.Get(creators[0]);
                        parent.Grad = null;

	                    FloatTensor local_grad = grad.Copy(autograd:grad.Autograd);
	                    
                        var grad_shape = new int[shape.Length];

                        for (int i = 0; i < grad.shape.Length; i++)
                        {
                            grad_shape[i] = grad.shape[i];
                        }

                        for (int i = 0; i < shape.Length; i++)
                        {
                            grad_shape[i] = parent.shape[i];
                            if (parent.shape[i] == 1 && shape[i] > 1)
                            {
	                            local_grad = local_grad.Sum(i).View(grad_shape);
                            }
                        }

                        parent.Backward(local_grad, this);
                    }
                    else if (creation_op.Contains("shaped_index_select"))
                    {
	                    FloatTensor parent = factory.Get(creators[0]);
	                    IntTensor indices = factory.ctrl.intTensorFactory.Get(int_creators[0]);
	                    FloatTensor back_grad = parent.emptyTensorCopy(hook_graph: true);
	                    back_grad.autograd = false;
	                    back_grad.Zero_();
	                    
	                    FloatTensor out_grad = back_grad.IndexAdd(indices, -1, grad);
	                    parent.Backward(out_grad);
	                    
                    }
                    else if (creation_op.Contains("index_select"))
                    {
	                    FloatTensor parent = factory.Get(creators[0]);
	                    
	                    IntTensor indices = factory.ctrl.intTensorFactory.Get(int_creators[0]);

	                    int dim = int.Parse(creation_op.Split('_')[2]);
	                    
	                    FloatTensor back_grad = parent.emptyTensorCopy(hook_graph: true);
	                    back_grad.autograd = false;
	                    
	                    FloatTensor out_grad = back_grad.IndexAdd(indices, dim, grad);
	                    parent.Backward(out_grad);

                    }
                    else if (creation_op == "log")
                    {
	                    // TOOD: sum backprop logic   
	                    FloatTensor x = factory.Get(creators[0]).Copy(autograd:false);

	                    factory.Get(creators[0]).Backward(grad.Mul(x.Pow(-1)), this);
                    }
                    else if (creation_op == "mul_elem")
                    {
                        factory.Get(creators[0]).Backward(grad.Mul(factory.Get(creators[1])), this);
                        factory.Get(creators[1]).Backward(grad.Mul(factory.Get(creators[0])), this);
                    }
                    else if (creation_op == "mul_scalar")
                    {
                        factory.Get(creators[0]).Backward(grad.Mul(factory.Get(creators[1]).data[0]), this);
                    }
                    else if (creation_op == "mm")
                    {
                        FloatTensor x = factory.Get(creators[1]).Transpose();
                        x.autograd = false;

                        FloatTensor y = factory.Get(creators[0]).Transpose();
                        y.autograd = false;

                        factory.Get(creators[0]).Backward(grad.MM(x), this);
                        factory.Get(creators[1]).Backward(y.MM(grad), this);
                    }
                    else if (creation_op == "neg")
                    {
                        factory.Get(creators[0]).Backward(grad.Neg(), this);
                    }
                    else if (creation_op == "pow_scalar")
                    {

                        FloatTensor x = factory.Get(creators[0]).Copy(autograd:false);

                        factory.Get(creators[0]).Backward(x.Mul(grad).Mul(factory.Get(creators[1]).Data[0]), this);
                    }
                    else if (creation_op == "relu")
                    {
						// TOOD: replace with simple comparison and mulitplication (should be 2 liner)
                        FloatTensor c = this.Copy(autograd:false);

	                    FloatTensor output = c;
	                    
						var dimSize = 1;

						for (var i = 0 ; i < output.Shape.Length; ++i)
							dimSize *= output.Shape[i];

						var gradInput = output.Copy(autograd:false);
						gradInput.Zero_();

						var nCpu = SystemInfo.processorCount;
						Parallel.For(0, nCpu, workerId =>
						{
							var max = dimSize * (workerId + 1) / nCpu;
							for (var i = dimSize * workerId / nCpu; i < max; i++)
							{
								if (output.Data[i] > 0)
								{
									gradInput.Data[i] = 1;
								}else
								{
									gradInput.Data[i] = 0;
								}                
							}
						});

                        factory.Get(creators[0]).Backward((gradInput).Mul(grad), this);

                    }
                    else if (creation_op == "sub_elem")
                    {
                        factory.Get(creators[0]).Backward(grad, this);
                        factory.Get(creators[1]).Backward(grad.Neg(), this);
                    }
                    else if (creation_op == "sub_scalar")
                    {
                        factory.Get(creators[0]).Backward(grad, this);
                    }
                    else if (creation_op == "sigmoid")
                    {
                        FloatTensor self_nograd = this.Copy(autograd:false);

                        factory.Get(creators[0]).Backward(self_nograd.Neg().Add(1f).Mul(self_nograd).Mul(grad), this);
                    }
                    else if (creation_op.Contains("softmax-"))
                    {

                        FloatTensor c = this.Copy(autograd:false);
                        var dim = int.Parse(creation_op.Split('-')[1]);

	                    FloatTensor output = this;
	                    FloatTensor gradOutput = grad;

						if (!output.IsContiguous() || !gradOutput.IsContiguous())
							throw new NotImplementedException(
								"Softmax Gradient does not support non-contiguous tensors at the moment!");
						var outerSize = 1;
						var innerSize = 1;
						var dimSize = output.Shape[dim];

						for (var i = 0; i < dim; ++i)
							outerSize *= output.Shape[i];

						for (var i = dim + 1; i < output.Shape.Length; ++i)
							innerSize *= output.Shape[i];

						var dimStride = innerSize;
						var outerStride = dimSize * dimStride;

						var gradInput = output.Copy(autograd:false);

						var nCpu = SystemInfo.processorCount;
						Parallel.For(0, nCpu, workerId =>
						{
							var max = (outerSize * innerSize) * (workerId + 1) / nCpu;
							for (var i = (outerSize * innerSize) * workerId / nCpu; i < max; i++)
							{
								int outerIdx = i / innerSize;
								int innerIdx = i % innerSize;

								// works for contiguous!!
								var index = outerIdx * outerStride + innerIdx;

								float sum = 0;
								for (var d = 0; d < dimSize; d++)
									sum += output.Data[index + d * dimStride] * gradOutput.Data[index + d * dimStride];

								for (var d = 0; d < dimSize; d++)
									gradInput.Data[index + d * dimStride] =
										output.Data[index + d * dimStride] * (gradOutput.Data[index + d * dimStride] - sum);
							}
						});

						gradInput.Autograd = false;

                        factory.Get(creators[0]).Backward(gradInput, this);

                    }
                    else if (creation_op.Contains("sum"))
                    {
                        // TOOD: sum backprop logic   
                        FloatTensor parent = factory.Get(creators[0]);
                        parent.Grad = null;

	                    int dim = int.Parse(creation_op.Split('_')[1]);

	                    if (dim >= 0)
	                    {
		                    int[] view_shape = (int[]) parent.shape.Clone();
		                    view_shape[dim] = 1;
		                    parent.Backward(grad.View(view_shape).Expand(parent.shape).Contiguous());
	                    }
	                    else
	                    {
		                    int[] view_shape = (int[]) parent.shape.Clone();
		                    for (int i = 0; i < parent.shape.Length; i++) view_shape[i] = 1;
		                    parent.Backward(grad.View(view_shape).Expand(parent.shape).Contiguous());
	                    }
                    }
                    else if (creation_op == "transpose")
                    {
                        factory.Get(creators[0]).Backward(grad.Transpose());
                    }
                    else if (creation_op == "tanh")
                    {
                        FloatTensor c = this.Copy(autograd:false);

                        factory.Get(creators[0]).Backward(c.Pow(2).Neg().Add(1f).Mul(grad), this);
                    }
				    else if (creation_op.Contains("view_"))
				    {
					    FloatTensor parent = factory.Get(creators[0]);
					    
					    parent.Grad = null; // prevents gradient from simply being added to the previous gradient
					    					// instead the backpropagated gradient is set to a new value.
					    
					    parent.Backward(this.Grad.View(parent.shape));
				    }
				    else
				    {
					    Debug.Log("Autograd couldn't find matching operation for:" + creation_op);   
				    }
			    }
		    }
		    else
		    {
			    Debug.Log("Autograd off - skipping backprop at tensor:" + id + " with creation_op:" + creation_op);
		    }
	    }
        
        private FloatTensor SumGradient(FloatTensor input, FloatTensor grad, int dim)
        {
            // want to make grad look like this
            var inputShape = input.Shape;
            var stride = input.Strides;

            var gradData = grad.Data;
            var newData = new List<float>();
            
            // once we have proper support for non-contiguous tensors
            // most of this code can be replaced with a view and an expand
            // view the grad to add a singleton dimension in the dimension
            // of the sum and then expand it to the size of the input

            if (dim == 0)
            {
                var st = stride[dim];
                var sh = inputShape[dim];

                for (var i = 0; i < sh; i++)
                {
                    newData.AddRange(gradData);
                }
            }
            else
            {
                var index = 0;

                var totalSize = 1;
                for (var i = 0; i < inputShape.Length; i++)
                {
                    totalSize *= inputShape[i];
                }

                for (var i = 0; i < totalSize / (inputShape[dim] * stride[dim]); i++)
                {
                    for (var j = 0; j < inputShape[dim]; j++)
                    {
                        var segment = new ArraySegment<float>(gradData, index, stride[dim]);
                        newData.AddRange(segment);
                    }

                    index += stride[dim];
                }
            }

            return factory.Create( _shape: inputShape, _data: newData.ToArray());
        }
    }
}