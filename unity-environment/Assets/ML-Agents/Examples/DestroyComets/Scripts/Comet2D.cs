﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Comet2D : MonoBehaviour {

	#region Member Fields
	private CometsEnvironment _environment;
	#endregion

	#region Unity Lifecycle
	/// <summary>
	/// Destroy the commet as soons as it gets out of range
	/// </summary>
	private void FixedUpdate(){
		if (Vector3.Distance (transform.position, _environment.transform.position) > CometsEnvironment.SpawnRadius + 0.05f) {
			Destroy (gameObject);
		}
			
	}

	///<summary>
	/// Destroys the commet on collisions. If it got hit by a rocket, a reward has to be signaled to the agent via the environment.
	/// </summary>
	/// <param name="other"></param>
	private void OnCollisionEnter(Collision other){

		//Destroy projectile on collision
		if (!other.gameObject.tag.Equals ("Agent")) {
			Destroy (other.gameObject);
			_environment.Agent.RewardAgent (1); //REWARD FOR SHOOTING DOWN A COMET!
		}
		//Destroy the comet on Collision
		Destroy(gameObject);

	}
	#endregion

	#region Public Functions
	/// <summary>
	/// Initialize the comet, which requires a reference to the environment.
	/// </summary>
	/// <param name="env"></param>
	public void Init(CometsEnvironment env){
		_environment = env;
	}
	#endregion

}
