
namespace OpenMined.Network.Utils
{

	public class Command
	{
		// given that SyftController keeps lists of objects of base types
		// (at the time of writing this is only Tensors) then this command
		// selects one of these generic types and performs a command.
		public string objectType;
		// i.e. "tensor"
		public int objectIndex;
		//

		// name of the function to be called
		public string functionCall;

		public float[] data;
		public int[] shape;

		public string[] tensorIndexParams;
		public string[] hyperParams;
	}
}
