using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovingHpBar : MonoBehaviour {

	public GameObject target;

	GameObject player;
	EnemyState es;
	Vector3 floatingHeight;
	Vector3 screenPos;
	RectTransform healthBar;
	RectTransform healthBG;

	// Use this for initialization
	void Start () {
		player = GameObject.Find ("Tank");
		es = target.GetComponent<EnemyState> ();
		floatingHeight = new Vector3 (0f, 3f, 0f);
		healthBar = transform.GetChild(0).GetComponent<RectTransform>();
		healthBG = gameObject.GetComponent<RectTransform> ();
	}
	
	// Update is called once per frame
	void Update () {
		if (player && target) {
			screenPos = Camera.main.WorldToScreenPoint (target.transform.position + floatingHeight);
			if (isVisible (screenPos)) {
				float distance = Vector3.Distance (target.transform.position, player.transform.position);
				float zoom = 10f / distance;
				healthBar.sizeDelta = new Vector2 (40f * es.health / es.maxHealth * zoom, 30f * zoom);
				healthBG.sizeDelta = new Vector2 (40f * zoom, 30f * zoom);
				transform.position = screenPos;
			} else {
				transform.position = new Vector3(-10, -10, 0);
			}
		} else {
			Destroy (gameObject);
		}
	}

	bool isVisible(Vector3 pos){
		return pos.x > 0 && pos.x < Screen.width && pos.z > 0 && pos.z < Screen.height;
	}
}
