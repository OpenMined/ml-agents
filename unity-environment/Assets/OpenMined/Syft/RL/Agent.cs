using System;
using System.Collections.Generic;
using System.Runtime.Remoting.Messaging;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using OpenMined.Syft.Layer.Loss;
using OpenMined.Syft.Optim;
using OpenMined.Syft.Tensor;
using UnityEngine;

namespace OpenMined.Syft.NN.RL
{
    public class Agent
    {
        
        protected static volatile int nCreated = 0;
        
        // unique identifier held by SyftController
        protected int id;
        public int Id => id;
        
        private Layer.Layer model;
        private List<FloatTensor[]> history;
        private Optimizer optimizer;
        private SyftController controller;
        
        
        
        //private List<float> mean 
        
        public Agent(SyftController _controller, Layer.Layer _model, Optimizer _optimizer)
        {
            controller = _controller;
            model = _model;
            optimizer = _optimizer;
            
            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addAgent(this);

            history = new List<FloatTensor[]>();

        }

        public FloatTensor Forward(FloatTensor input)
        {
            return model.Forward(input);
        }

        public IntTensor Sample(FloatTensor input, int dim=1)
        {
            input.Autograd = true;
            
            FloatTensor pred = Forward(input);
            
            IntTensor actions = pred.Sample(dim);
            FloatTensor action_preds = pred.IndexSelect(actions,-1);

            history.Add(new FloatTensor[2]{action_preds,null });
            return actions;
        }

        public void HookReward(FloatTensor reward)
        {
            history[history.Count - 1][1] = reward;
        }

        public void Learn()
        {
            
            List<FloatTensor> rewards_list = new List<FloatTensor>();
            List<FloatTensor> losses_list = new List<FloatTensor>();
            
            for (int i = 0; i < history.Count; i++)
            {
                if (history[i][1] != null)
                {
                    rewards_list.Add(history[i][1]);
                }

                if (history[i][0] != null)
                {
                    losses_list.Add(history[i][0]);
                }
            }

            FloatTensor rewards = Functional.Concatenate(this.controller.floatTensorFactory, rewards_list, 0);
            FloatTensor losses = Functional.Concatenate(this.controller.floatTensorFactory, losses_list, 0);

            var norm_rewards = rewards.Sub(rewards.Mean()).Div(rewards.Std().Add(0.000001f));
            norm_rewards.Autograd = true;
            var policy_loss = norm_rewards.Mul(losses.Neg()).Sum(0);
            policy_loss.Backward();
            
            optimizer.Step(rewards.Shape[0],0);
            
            history = new List<FloatTensor[]>();

        }
        
        public string ProcessMessageLocal(Command msgObj, SyftController ctrl)
        {
            switch (msgObj.functionCall)
            {
                case "sample":
                {
                    var input = ctrl.floatTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = this.Sample(input);
                    return result.Id + "";
                }
                case "hook_reward":
                {
                    var reward = ctrl.floatTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    this.HookReward(reward);
                    return this.Id + "";
                }
                case "deploy":
                {
                    // 1234 is a special id that Unit watches for
                    controller.setAgentId(id,1234);
                    return "1234";
                }
                case "get_history":
                {
                    if (history.Count < 0)
                    {
                        string indices = "[";
                        string reward_str;
                        string loss_str;
                        for (int i = 0; i < history.Count; i++)
                        {
                            if (history[i][0] != null)
                            {
                                loss_str = history[i][0].Id + "";
                            }
                            else
                            {
                                loss_str = "-1";
                            }

                            if (history[i][1] != null)
                            {
                                reward_str = history[i][1].Id + "";
                            }
                            else
                            {
                                reward_str = "-1";
                            }

                            indices += "[" + loss_str + "," + reward_str + "],";

                        }

                        return indices.Substring(0, indices.Length - 2) + "]";
                    }
                    else
                    {
                        return "";
                    }
                }
                case "learn":
                {
                    Learn();
                    return this.Id + "";
                }
                default: 
                {
                    return "Policy.processMessage not Implemented:" + msgObj.functionCall;
                }
            }
        }
        
    }
}