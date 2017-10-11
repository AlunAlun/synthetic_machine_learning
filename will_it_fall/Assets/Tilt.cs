using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using UnityEngine.SceneManagement;

public class Tilt : MonoBehaviour {

	public float speed = 1.0f;
	public float timeScale = 1.0f;
	private bool fallen = false;
	private float xAxis, yAxis, zAxis = 0.0f;

	private Text t_it;
	private Text t_ax;
	private Text t_an;

	public static int counter = 0;

	// Use this for initialization
	void Start () {

		xAxis = Random.value * 2 - 1;
		yAxis = 0.0f;
		zAxis = Random.value * 2 - 1;
		//Debug.Log("(" + xAxis + ", "+ yAxis + ", " + zAxis + ")");


		if (counter > 10000) {

			Application.Quit ();
		}

		t_it = GameObject.Find ("/Canvas/Iterations").GetComponent<Text>();
		t_ax = GameObject.Find ("/Canvas/Axis").GetComponent<Text>();
		t_an = GameObject.Find ("/Canvas/Angle").GetComponent<Text>();

		t_it.text = "Iteration: " + counter;
		t_ax.text = "(" + xAxis + ", "+ yAxis + ", " + zAxis + ")";
		counter++;




	}
	
	// Update is called once per frame
	void Update () {
		if (fallen)
			return;
		transform.RotateAround(Vector3.zero, new Vector3 (xAxis, yAxis, zAxis), speed);

		float angle = 0.0f;
		Vector3 axis = Vector3.zero;
		transform.rotation.ToAngleAxis (out angle, out axis);
		t_an.text = angle.ToString();
	}

	public IEnumerator Fallen(){

		float angle = 0.0f;
		Vector3 axis = Vector3.zero;
		transform.rotation.ToAngleAxis (out angle, out axis);
		fallen = true;
		string str = axis.x + " 0.0 " + axis.z + " : " + angle;
		//Debug.Log (str);
		StreamWriter writer = new StreamWriter("Assets/Resources/will_it_fall.txt", true);
		writer.WriteLine(str);
		writer.Close ();
		yield return new WaitForSeconds (1);
		SceneManager.LoadScene("will_it_fall");
	}
}
