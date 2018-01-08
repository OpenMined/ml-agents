using System.Collections.Generic;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.Layer
{
    public class Embedding : Layer
    {
        private int _numEmbeddings;
        private int _embeddingDim;

        private readonly FloatTensor _weights;
    
        public Embedding(SyftController controller, int numEmbeddings, int embeddingDim)
        {
            init("embedding");

            this.controller = controller;

            _numEmbeddings = numEmbeddings;
            _embeddingDim = embeddingDim;

            int[] weightShape = { _numEmbeddings, _embeddingDim };
            var weights = controller.RandomWeights(_numEmbeddings * _embeddingDim);
            _weights = controller.floatTensorFactory.Create(_shape: weightShape, _data: weights, _autograd: true, _keepgrads: true);

            parameters.Add(_weights.Id);

            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }
    
        // TODO this should take a IntTensor
        public override FloatTensor Forward(FloatTensor input)
        {
            FloatTensor output = _weights.emptyTensorCopy();
            
            var indices = new List<int>();

            foreach (var d in input.Data)
            {
                indices.Add((int)d);
            }
            
            if (input.Shape.Length == 1)
            {
                output = _weights.IndexSelect(indices, 0);
            }
            else
            {
                output = _weights.IndexSelect(indices, 0);

                int[] newShape = { input.Shape[0], input.Shape[1], _weights.Shape[1] };

                output = output.View(newShape);
            }

            return output;
        }

        public override int getParameterCount(){return _weights.Size;}
    }
}
