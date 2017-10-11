using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WillIFall : MonoBehaviour {

	public GameObject Floor;
	private Tilt tilt;
	private bool fallen;
	private Rigidbody rb;
	private Rigidbody rbFloor;

	// Use this for initialization
	void Start () {
		tilt = Floor.GetComponent<Tilt> ();
		rb = GetComponent<Rigidbody> ();
		rbFloor = Floor.GetComponent<Rigidbody> ();

	}
	
	// Update is called once per frame
	void Update () {

		if (!fallen && (rb.velocity.magnitude - rbFloor.velocity.magnitude > 0.15f)) {
			StartCoroutine(tilt.Fallen());
			fallen = true;
		} 
			
	}
}
