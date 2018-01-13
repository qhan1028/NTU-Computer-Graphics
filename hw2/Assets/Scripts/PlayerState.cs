using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using System.Collections;

public class PlayerState : MonoBehaviour {
    public float maxHealth = 100;
	static public float health;
	public GameObject deadEffect;
	public GameObject gameOver;
	public GameObject gameOverTextObject;
	public GameObject playerHpBar;
	public Vector3 initPosition = new Vector3 (47f, 1f, 41f);
	static public bool alive;

	RectTransform hpBar;
	Text gameOverText;
	string prefix = "Game Over\nYou Get ";
	string postfix = " Points";
	Vector3 terrainSize;

    // Use this for initialization
    void Start () {
		alive = true;
		gameObject.transform.position = initPosition;
		gameOver.SetActive (false);
		gameOverText = gameOverTextObject.GetComponent<Text> ();
		hpBar = playerHpBar.GetComponent<RectTransform> ();
		health = maxHealth;
		terrainSize = GameObject.Find ("Terrain").GetComponent<Terrain> ().terrainData.size;
	}
	
	// Update is called once per frame
	void Update () {
		Cursor.visible = false;
		if (Input.GetKey (KeyCode.M)) { // back to menu
			EnemyGenerator.enemyCount = 0;
			ScoreBoard.score = 0;
			SceneManager.LoadScene (0);
		}

		if (!alive) {
			gameOver.SetActive (true);
			gameOverText.text = prefix + ScoreBoard.score.ToString () + postfix;
			if (Input.GetKey (KeyCode.R)) {
				alive = true;
				gameObject.transform.position = initPosition;
				health = maxHealth;
				ScoreBoard.score = 0;
			}
		} else {
			gameOver.SetActive (false);
			if (health > maxHealth)
				health = maxHealth;
			
			PlayerBounds ();
		}
		hpBar.sizeDelta = new Vector2(100f * health / maxHealth, 50f);
	}

    void OnCollisionEnter(Collision collision)
    {
		if (collision.gameObject.tag == "enemy") //當撞到的collider具有enemy tag時
        {
            health = health - 10;
            if (health <= 0)
            {
                Instantiate(deadEffect, transform.position, transform.rotation);
				alive = false;
            }
        }  
    }

	void PlayerBounds() {
		Vector3 pos = transform.position;

		if (pos.x >= terrainSize.x-10) {
			transform.position = new Vector3 (terrainSize.x-10, pos.y, pos.z);
		}
		if (pos.x <= 10) {
			transform.position = new Vector3 (10, pos.y, pos.z);
		}
		if (pos.z >= terrainSize.z-10) {
			transform.position = new Vector3 (pos.x, pos.y, terrainSize.z-10);
		}
		if (pos.z <= 10) {
			transform.position = new Vector3 (pos.x, pos.y, 10);
		}
	}
}
