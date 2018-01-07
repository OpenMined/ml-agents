using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using UnityEngine;

namespace OpenMined.Syft.Layer
{
	public class Dropout : Layer
	{

		private FloatTensor _mask_source;
		private readonly float rate;

		public Dropout(SyftController _controller, float _rate)
		{
			init("dropout");

			this.controller = _controller;
			this.rate = _rate;

			#pragma warning disable 420
			id = System.Threading.Interlocked.Increment(ref nCreated);
			controller.addModel(this);

		}

		public override FloatTensor Forward(FloatTensor input)
		{
			if (_mask_source == null || input.Size != _mask_source.Size)
			{
				_mask_source = input.emptyTensorCopy(hook_graph:false);
				_mask_source.Fill(1 - rate, inline: true);
			};
			
			FloatTensor output = input.Mul(_mask_source.SampleMask());
			activation = output.Id;
			return output;
			
		}
		
		public override int getParameterCount(){return 0;}
	}
}