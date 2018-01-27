using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CartPoleAgent : Agent {

	#region Member Fields
	private Vector3 initialCartPosition;
	private Vector3 initialPolePosition;
	private Quaternion initialPoleRotation;
	private float newPosition;
	private float minPosition = -2.4f;
	private float maxPosition = 2.4f;
	[SerializeField]
	private float max_fail_angle = 15.0f;
	[SerializeField]
	private float _cartSpeed = 1.0f;
	[SerializeField]
	private Transform _poleTransform;
	[SerializeField]
	private Rigidbody _poleRigidbody;

	private float episode_reward = 0.0f;



	#endregion

	#region Agent Overrides

	/*
	 * Observation   Min   Max
	 * Cart Position   -2.4   2.4
	 * Cart Velocity   -Inf   Inf
	 * Pole Angle   -41.8º    41.8º
	 * Pole Velocity At Tip   -Inf Inf
	 * */

	public override List<float> CollectState()
	{
		List<float> state = new List<float>();

		float cartPosition = transform.position.x;
		float cartVelocity = GetComponent<Rigidbody> ().velocity.x;
		float poleAngle = _poleTransform.rotation.z;
		float poleVelocity = _poleRigidbody.velocity.x;


		state.Add (cartPosition);
		state.Add (cartVelocity);
		state.Add (poleAngle);
		state.Add (poleVelocity);

		/*
		string state_str = "";
		foreach(var s in state)
		{
			state_str += s + ",";
		}
		Debug.Log (state_str);
		*/

		return state;
	}

	public override void AgentStep(float[] act)
	{
		//OpenMined control
		if (brain.brainType.Equals (BrainType.OpenMined)) {
			if (brain.brainParameters.actionSpaceType.Equals (StateType.discrete)) {
				float movement = act [0];
				//Debug.Log ("Movement: "+movement.ToString ());
				float direction = 0;
				//The system is controlled by applying a force of +1 or -1 to the cart
				if (movement == 0) { direction = -0.2f; }
				if (movement == 1) { direction = 0.2f; }

				transform.Translate (_cartSpeed * direction * Time.deltaTime, 0f, 0f);


				//Control position
				if (transform.position.x > maxPosition || transform.position.x < minPosition) {
					AgentReset ();
				}
				//Control rotation of pole
				else if (_poleTransform.rotation.eulerAngles.z < (360 - max_fail_angle) && _poleTransform.rotation.eulerAngles.z > max_fail_angle) {
					AgentReset ();
				} else {
					reward = 1f;
					episode_reward += reward;
				}

				if (episode_reward == 200) {
					done = true;
				}

				
			}
		} 
		//Player control
		if (brain.brainType.Equals (BrainType.Player)) {
			
			transform.Translate (_cartSpeed * Input.GetAxis ("Horizontal") * Time.deltaTime, 0f, 0f);

			//Control position
			if (transform.position.x > maxPosition || transform.position.x < minPosition) {
				AgentReset ();
			}
			//Control rotation of pole
			else if (_poleTransform.rotation.eulerAngles.z < (360 - max_fail_angle) && _poleTransform.rotation.eulerAngles.z > (360 - max_fail_angle)) {
				AgentReset ();
			} else {
				reward = 1f;
			}
		}

	}

	#endregion

	#region Unity Lifecycle

	private void Start()
	{
		Debug.Log("Start");
		initialCartPosition = transform.position;
		initialPolePosition = _poleTransform.position;
		initialPoleRotation = _poleTransform.rotation;
		Debug.Log(initialCartPosition.ToString());
		Debug.Log(initialPolePosition.ToString());
		Debug.Log(initialPoleRotation.ToString());
	}

	/// <summary>
	/// Executes the player's key inputs
	/// Draws debug lines for each vision ray.
	/// </summary>

	private void Update(){

	}

	#endregion

	public override void AgentReset()
	{
		
		Debug.Log("AgentReset");
		Debug.Log ("Last episode reward: " + episode_reward.ToString ());
		episode_reward = 0;
		//Delta rotation to make initial state not stable
		float delta = Random.Range(-5.0f,5.0f);
		if (delta < 0f) {
			delta = 360 - Mathf.Abs (delta);
		}

		//Debug.Log (delta.ToString ());


		_poleTransform.position = initialPolePosition;
		_poleTransform.rotation = initialPoleRotation * Quaternion.Euler(0f,0f,delta);
		transform.position = initialCartPosition;

		//Physics
		GetComponent<Rigidbody>().velocity = Vector3.zero;
		GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
		_poleRigidbody.velocity = Vector3.zero;
		_poleRigidbody.angularVelocity = Vector3.zero;
	}

	public override void AgentOnDone()
	{
		Debug.Log ("Done");

	}
}
