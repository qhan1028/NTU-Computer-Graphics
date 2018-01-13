using System.Collections;
using UnityEngine;
using UnityEngine.UI;

public class ScoreBoard : MonoBehaviour {
	public static int score = 0;
	public GameObject scoreTextObject;
	Text scoreText;

	void Start(){
		scoreText = transform.GetChild(0).GetComponent<Text> ();
	}

	void Update() {
		scoreText.text = "Score: " + score.ToString ();	
	}
}
