using System;
using System.IO;
using System.Threading.Tasks;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        public static byte[] Serialize(FloatTensor x)
        {
            x.Cpu();
            
            if(x.shape.Length > 10)
                throw new Exception("cannot serialize tensors with greater than 10 dimensions");
            
            var byteArray = new byte[10 * sizeof(int) + x.data.Length * sizeof(float)];
            Buffer.BlockCopy(x.shape, 0, byteArray, 0, x.shape.Length * sizeof(int));
            Buffer.BlockCopy(x.data, 0, byteArray, 10 * sizeof(int), x.data.Length * sizeof(float));
            return byteArray;
        }

        public static Tuple<int[], float[]> Deserialize(byte[] serialied_byte_array)
        {
            var padded_shape = new int[10];
            var data = new float[(serialied_byte_array.Length - (10 * sizeof(int))) / sizeof(float)];
            Buffer.BlockCopy(serialied_byte_array, 0, padded_shape, 0, padded_shape.Length * sizeof(int));
            Buffer.BlockCopy(serialied_byte_array, (10 * sizeof(int)), data, 0, data.Length * sizeof(float));

            int actual_length = 0;
            for (int i = 0; i < padded_shape.Length; i++)
            {
                if (padded_shape[i] == 0)
                {
                    actual_length = i;
                    break;
                }
            }

            int[] shape = new int[actual_length];
            for (int i = 0; i < actual_length; i++)
            {
                shape[i] = padded_shape[i];
            }
            
            return new Tuple<int[], float[]>(shape,data);
            
        }

        public static Tuple<int[], float[]> ReadFromFile(string filename)
        {
            byte[] serialied_byte_array = FileToByteArray(filename);

            return Deserialize(serialied_byte_array);

        }

        public bool WriteToFile(string filename)
        {
            return ByteArrayToFile(filename, Serialize(this));
        }

        public static bool ByteArrayToFile(string fileName, byte[] byteArray)
        {
            try
            {
                using (var fs = new FileStream(fileName, FileMode.Create, FileAccess.Write))
                {
                    fs.Write(byteArray, 0, byteArray.Length);
                    return true;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Exception caught in process: {0}", ex);
                return false;
            }
        }
        
        public static byte[] FileToByteArray(string fileName)
        {
            try
            {
                byte[] array = File.ReadAllBytes(fileName);
                return array;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Exception caught in process: {0}", ex);
                return null;
            }
        }
    }
}
