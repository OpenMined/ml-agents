using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CometsEnvironment : MonoBehaviour {

	#region Member Fields
	[SerializeField]
	private CometsAgent _agent;
	[SerializeField]
	private GameObject _comet;
	[SerializeField]
	private static float _spawnRadius = 5.0f;
	[SerializeField]
	private float _spawnFrequency = 3;
	[SerializeField]
	private float _spawnDisplacement = 1.5f;
	[SerializeField]
	private float _cometSpeed = 1.5f;
	[SerializeField]
	private bool _isSpawning = true;
	[SerializeField]
	private LineRenderer _lineRenderer;
	private int _segments = 60;
	#endregion

	#region Memeber Properties
	public static float SpawnRadius{
		get{ return _spawnRadius; }
	}

	public CometsAgent Agent{
		get{ return _agent; }
	}
	#endregion

	#region Unity Lifecycle
	///<summary>
	/// Draws a circle to indicate the boundaries of the environment and launches the spawn behavior for the comets.
	/// </summary>
	private void Start(){

		Debug.Log ("Start");

		StartCoroutine (SpawnLoop ());

		//Draw circle to visualize the boundaries
		_lineRenderer.positionCount = _segments + 1;
		_lineRenderer.startWidth = _lineRenderer.endWidth = 0.33f;
		float angleStep = 360 / _segments + 1;
		Debug.Log ("Draw Line");

		for (int i = 0; i < _segments + 1; i++) {
			Debug.Log ("Segment");
			Vector3 circumferencePoint = new Vector3 (transform.position.x + (_spawnRadius * Mathf.Cos (angleStep * i * Mathf.Deg2Rad)),
				                             transform.position.y + (_spawnRadius * Mathf.Sin (angleStep * i * Mathf.Deg2Rad)),
				                             0);
			_lineRenderer.SetPosition (i, circumferencePoint);	
		}
	}
	#endregion

	#region Private Functions
	/// <summary>
	/// Spawn behavior of the comets
	/// </summary>
	/// <returns></returns>
	private IEnumerator SpawnLoop(){
		while (_isSpawning) {
			float randomAngle = Random.Range (0, 360);
			Vector3 spawnLocation = new Vector3(transform.position.x + _spawnRadius * Mathf.Cos(randomAngle * Mathf.Deg2Rad),
				transform.position.y + _spawnRadius * Mathf.Sin(randomAngle * Mathf.Deg2Rad),
				0);
			GameObject comet = Instantiate (_comet, spawnLocation, Quaternion.identity);
			Vector3 cometVelocity = (transform.position - spawnLocation);
			comet.GetComponent<Rigidbody> ().velocity = new Vector3 (cometVelocity.x += Random.Range (-_spawnDisplacement, _spawnDisplacement), cometVelocity.y += Random.Range (-_spawnDisplacement, _spawnDisplacement), cometVelocity.z).normalized * _cometSpeed;
			comet.GetComponent<Comet2D> ().Init (this);
			yield return new WaitForSeconds (1 / _spawnFrequency);
		}
		yield return null;
	}
	#endregion
}
