using System;
using System.Collections.Generic;
using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {

        private List<int> creators;
        private string creation_op;
        private List<int> children_indices; // children -> counts
	    private List<int> children_counts; // children -> counts

	    private List<int> children_int_indices; // when integer indices are used as a child
	    private List<int> int_creators; // when integer indices are used to create this tensor

        public void InitAutograd()
        {
	        if (!autograd)
	        {
		        autograd = true;
		        InitGraph();
	        }
        }

	    public void InitGraph()
	    {
		    creators = new List<int>();
		    children_indices = new List<int>();
		    children_counts = new List<int>();

		    children_int_indices = new List<int>();
		    int_creators = new List<int>();
	    }

	    public void ResetAutogradCounts()
	    {
		    for (int i = 0; i < children_counts.Count; i++)
		    	children_counts[i] = 0;
	    }

	    public FloatTensor HookGraph(ref FloatTensor result, 
		    						string creation_op, 
		    						bool inline, 
		    						float scalar_input = -1, 		    
		    						FloatTensor[] tensor_inputs = null, 
		    						int[] resultShape = null,
		    						float[] resultData = null, 
		    						IntTensor[] indices = null)
	    {
		    
		    // no dynamic graph for inline operations
		    if (inline)
			    return this;
		    
		    bool autograd_pre_initialized = false;

		    // if we don't override with a result tensor being passed in, let's first look to see if we can reuse one
		    // from a previous operation - if not - we'll create our own.
		    if (result == null)
		    {

			    bool child_pre_initialized = false;
			    int child_index = 0;

			    
			    // iterate through all children to see if any were created using the same parameters and creation_op
			    // as is currently being requested 
				for (int i = 0; i < this.children_indices.Count; i++)
				{

					FloatTensor child = factory.Get(children_indices[i]);

					if (child.creation_op == creation_op)
					{
						// if this creation_op requires no parameters - then we only have to match
						// on the creation_op itself - which we have already done.
						if (scalar_input == -1 && (tensor_inputs == null || tensor_inputs.Length == 0))
						{
							child_pre_initialized = true;
							child_index = children_indices[i];
							break;
						}
						
						// since there are paremeters - now this child must match all parameters exactly
						bool keep_looking = false;
						
						if (scalar_input != -1)
							if (child.creators.Count > 1)
								if (factory.Get(child.creators[1]).data[0] != scalar_input)
									keep_looking = true;
						
						if (tensor_inputs != null && tensor_inputs.Length >= 1)
							foreach(FloatTensor tensor in tensor_inputs)
								if (!child.creators.Contains(tensor.id))
									keep_looking = true;

						if (keep_looking)
							continue;

						// found a child that matches all parameters
						child_pre_initialized = true;
						child_index = children_indices[i];
						break;

					}
				}

			    if (child_pre_initialized)
			    {	
				    autograd_pre_initialized = true;
				    result = factory.Get(child_index);
				    result.Zero_();
			    }
			    else
			    {
				    bool resultAutograd = autograd;
				    
				    if(tensor_inputs != null)
					    foreach (FloatTensor tensor in tensor_inputs)
						    resultAutograd = tensor.autograd && resultAutograd;

				    if (resultShape == null)
				    {
					    resultShape = this.shape;
					    
					    if (resultData == null)
						    resultData = this.data;
				    }
				    else
				    {
					    // if shape is passed in - initialize a new dataset with that shape
					    resultData = null;
				    }

				    result = factory.Create(
					    _shape: resultShape,
					    _data: resultData,
					    _dataBuffer: dataBuffer,
					    _shapeBuffer: shapeBuffer,
					    _shader: shader,
					    _copyData: true,
					    _dataOnGpu: dataOnGpu,
					    _autograd: resultAutograd, // if either tensor doesn't have gradients
					    _keepgrads: keepgrads, // neither does the result. This might not end up being
					    _creation_op: creation_op); // a good decision in the long run. We'll see.				    
				    
				    if (this.dataOnGpu)
					    result.Gpu(shader);
				    

			    }
		    }
		    if (autograd_pre_initialized)
		    {
			    
				this.ResetAutogradCounts();
				result.ResetAutogradCounts();
				
				if(tensor_inputs != null)
					foreach (FloatTensor tensor in tensor_inputs)
						tensor.ResetAutogradCounts();
					
			}
			else
			{
				
				result.InitGraph();
				result.creators.Add(this.id);
				result.creation_op = creation_op;
				
				children_indices.Add(result.Id);
				children_counts.Add(0);
				
				// hook autograd one parents - one scalar
				if (scalar_input != -1)
					result.creators.Add(factory.Create(
						_shape: new int[] {1},
						_data: new float[] {scalar_input},
						_dataBuffer: dataBuffer,
						_shapeBuffer: shapeBuffer,
						_shader: shader,
						_copyData: true,
						_dataOnGpu: dataOnGpu,
						_autograd: autograd,
						_keepgrads: keepgrads,
						_creation_op: creation_op).id);
					
				// hook autograd - two parents
				if (tensor_inputs != null)
					foreach (FloatTensor tensor in tensor_inputs)
					{
						result.creators.Add(tensor.id);
						tensor.children_indices.Add(result.Id);
						tensor.children_counts.Add(0);
						
					}
					
				
				// special storage for the graph so that we can know which indices of the parent to 
				// backprop into. note that int_creators are expected to be non-differentiable and so we do
				// not backprop into them directly
				if (indices != null && indices.Length > 0)
				{
					if (result.int_creators.Count == 0)
					{
						foreach (IntTensor ind in indices)
							result.int_creators.Add(ind.Id);
					}
					else if (result.int_creators.Count == indices.Length)
					{
						// TODO: after dynamic graph works for IntTensor you should be able to simply check to see if
						// the ids are the same - but at the time of writing we always creating new IntTensors so that 
						// wouldn't work yet.
					}
					else
					{
						throw new Exception("Something is wrong... int_creators already existed but had the wrong length");
					}
				}

				// TODO: this is just used so that eventually if any inline operation was run on "indices" to change it
				// (before backpropagating), we could trigger a warning that backprop will be broken.
				//indices.children_indices.Add(result.id);
				
		    }
		    return result;
	    }
	   
    }
}