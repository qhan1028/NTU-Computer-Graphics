using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class MenuUI : MonoBehaviour {

	public GameObject helpMessage;

	// Use this for initialization
	void Start () {
		helpMessage.SetActive (false);
		Cursor.visible = true;
	}
	
	// Update is called once per frame
	void Update () {
		
	}

	public void LoadScene(int level) {
		SceneManager.LoadScene (level);
	}

	public void HelpMessage() {
		helpMessage.SetActive (!helpMessage.activeSelf);
	}
}
