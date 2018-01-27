using System.Collections.Generic;
using System.Threading;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Layer;
using OpenMined.Syft.Tensor;
using UnityEngine;
using Agent = OpenMined.Syft.NN.RL.Agent;
#if UNITY_EDITOR
using UnityEditor;
#endif


/// CoreBrain which decides actions using Player input.
public class CoreBrainOpenMined : ScriptableObject, CoreBrain
{
    [SerializeField]
    private bool broadcast = true;
    
    [System.Serializable]
    private struct DiscretePlayerAction
    {
        public KeyCode key;
        public int value;
    }

    [System.Serializable]
    private struct ContinuousPlayerAction
    {
        public KeyCode key;
        public int index;
        public float value;
    }
        
    ExternalCommunicator coord;
    
    [SerializeField]
    /// Contains the mapping from input to continuous actions
    private ContinuousPlayerAction[] continuousPlayerActions;
    
    [SerializeField]
    /// Contains the mapping from input to discrete actions
    private DiscretePlayerAction[] discretePlayerActions;
    
    [SerializeField]
    private int defaultAction = 0;

    /// Reference to the brain that uses this CoreBrainPlayer
    public Brain brain;

    public OpenMined.Syft.NN.RL.Agent policy = null;
    public SyftController ctrl;
    public bool found_policy = false;

    /// Create the reference to the brain
    public void SetBrain(Brain b)
    {
        brain = b;
    }

    /// Nothing to implement
    public void InitializeCoreBrain()
    {
        if ((brain.gameObject.transform.parent.gameObject.GetComponent<Academy>().communicator == null)
            || (!broadcast))
        {
            coord = null;
        }
        else if (brain.gameObject.transform.parent.gameObject.GetComponent<Academy>().communicator is ExternalCommunicator)
        {
            coord = (ExternalCommunicator)brain.gameObject.transform.parent.gameObject.GetComponent<Academy>().communicator;
            coord.SubscribeBrain(brain);
        }
        
        
        
    }

    /// Uses the continuous inputs or dicrete inputs of the player to 
    /// decide action
    public void DecideAction()
    {

        if (ctrl == null)
            ctrl = brain.brainParameters.syft.controller;
        
        if (policy == null)
        {
            found_policy = false;
        }
        else
        {
            found_policy = true;
        }

        if (ctrl.getAgent(1234) != null)
        {
            policy = ctrl.getAgent(1234);
            if (found_policy == false)
            {
                foreach (KeyValuePair<int, global::Agent> idAgent in brain.agents)
                {
                    idAgent.Value.Reset();
                }
            }
        }
        else
        {
            
            policy = null;
        }
       
        //The states are collected in order to debug the CollectStates method.
        Dictionary<int,List<float>> states = brain.CollectStates();
        Dictionary<int, float> rewards = brain.CollectRewards();
        Dictionary<int, bool> dones = brain.CollectDones();
        
        if (brain.brainParameters.actionSpaceType == StateType.continuous)
        {
            float[] action = new float[brain.brainParameters.actionSize];
            foreach (ContinuousPlayerAction cha in continuousPlayerActions)
            {
                if (Input.GetKey(cha.key))
                {
                    action[cha.index] = cha.value;
                }
            }
            Dictionary<int, float[]> actions = new Dictionary<int, float[]>();
            foreach (KeyValuePair<int, global::Agent> idAgent in brain.agents)
            {
                actions.Add(idAgent.Key, action);
            }
            brain.SendActions(actions);
        }
        else
        {
            float[] action = new float[1] { defaultAction };
            foreach (DiscretePlayerAction dha in discretePlayerActions)
            {
                if (Input.GetKey(dha.key))
                {
                    action[0] = (float)dha.value;
                    break;
                }
            }
            Dictionary<int, float[]> actions = new Dictionary<int, float[]>();
            foreach (KeyValuePair<int, global::Agent> idAgent in brain.agents)
            {
                if (policy == null)
                {
                    // do nothing - you don't have a network
                    actions.Add(idAgent.Key, new float[1] {0});
                }
                else
                {
					//input = [Number of agents x state size]
					FloatTensor input = ctrl.floatTensorFactory.Create(_shape: new int[] {1,states[idAgent.Key].Count},
						_data: states[idAgent.Key].ToArray());

                    IntTensor pred = policy.Sample(input);
                    actions.Add(idAgent.Key, new float[1] {pred.Data[0]});
                    
                }
            }
            
            brain.SendActions(actions);
        }
  
        
    }

    public void SendState()
    {

        Dictionary<int, List<float>> states = brain.CollectStates();
        Dictionary<int, float> rewards = brain.CollectRewards();
        Dictionary<int, bool> dones = brain.CollectDones();
        
        if (ctrl == null)
            ctrl = brain.brainParameters.syft.controller;
        
        if (coord != null)
        {
            coord.giveBrainInfo(brain);
        }
        else
        {

            if (policy != null)
            {
                
                foreach (KeyValuePair<int, List<float>> idAgent in states)
                {
                    
                    float raw_reward = rewards[idAgent.Key];

                    FloatTensor reward = ctrl.floatTensorFactory.Create(_shape: new int[] {1},
                        _data: new float[] {raw_reward});
                    
                    policy.HookReward(reward);

                    if (dones[idAgent.Key])
                    {
                        policy.Learn();    
                    }
                    
                }
            }

        }
        
    }

    /// Displays continuous or discrete input mapping in the inspector
    public void OnInspector()
    {
#if UNITY_EDITOR
        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
        broadcast = EditorGUILayout.Toggle("Broadcast", broadcast);
        SerializedObject serializedBrain = new SerializedObject(this);
        if (brain.brainParameters.actionSpaceType == StateType.continuous)
        {
            GUILayout.Label("Edit the continuous inputs for you actions", EditorStyles.boldLabel);
            SerializedProperty chas = serializedBrain.FindProperty("continuousPlayerActions");
            serializedBrain.Update();
            EditorGUILayout.PropertyField(chas, true);
            serializedBrain.ApplyModifiedProperties();
            if (continuousPlayerActions == null)
            {
                continuousPlayerActions = new ContinuousPlayerAction[0];
            }
            foreach (ContinuousPlayerAction cha in continuousPlayerActions)
            {
                if (cha.index >= brain.brainParameters.actionSize)
                {
                    EditorGUILayout.HelpBox(string.Format("Key {0} is assigned to index {1} but the action size is only of size {2}"
                        , cha.key.ToString(), cha.index.ToString(), brain.brainParameters.actionSize.ToString()), MessageType.Error);
                }
            }

        }
        else
        {
            GUILayout.Label("Edit the discrete inputs for you actions", EditorStyles.boldLabel);
            defaultAction = EditorGUILayout.IntField("Default Action", defaultAction);
            SerializedProperty dhas = serializedBrain.FindProperty("discretePlayerActions");
            serializedBrain.Update();
            EditorGUILayout.PropertyField(dhas, true);
            serializedBrain.ApplyModifiedProperties();
        }
#endif
    }
}
