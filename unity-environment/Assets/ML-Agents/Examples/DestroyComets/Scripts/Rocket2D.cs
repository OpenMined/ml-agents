using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Rocket2D : MonoBehaviour {

	#region Member Fields
	private Vector3 _origin;
	#endregion

	#region Unity Lifecycle
	/// <summary>
	/// Saves the original spawn location of the projectile
	/// </summary>
	private void Start(){
		_origin = transform.position;
	}

	///<summary>
	/// Destroys the projectile after reaching a certain distance to the center of the environment.
	/// </summary>
	private void FixedUpdate(){
		//Destroy the rocket as soon as it gets out of range
		if (Vector3.Distance (_origin, transform.position) > CometsEnvironment.SpawnRadius + 0.05f) {
			Destroy (gameObject);
		}
	}
	#endregion

}
