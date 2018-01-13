using UnityEngine;
using System.Collections;

public class EnemyState : MonoBehaviour {
    
	public GameObject deadEffect;
	public float maxHealth = 100;
	public float health;
    // Use this for initialization
    void Start () {
		health = maxHealth;
	}
	
	// Update is called once per frame
	void Update () {
	
	}
    void OnCollisionEnter(Collision collision)
    {
		if (PlayerState.alive) {
			if (collision.gameObject.tag == "bullet1") {//當撞到的collider具有enemy tag時
				health = health - 10;
				ScoreBoard.score += 1;
				if (health <= 0) {
					EnemyDead ();
				}
			}
			if (collision.gameObject.tag == "bullet2") {//當撞到的collider具有enemy tag時
				health = health - 20;
				ScoreBoard.score += 2;
				if (health <= 0) {
					EnemyDead ();
				}
			}
		}
    }

	void EnemyDead(){
		Instantiate (deadEffect, transform.position, transform.rotation);
		EnemyGenerator.enemyCount -= 1;
		PlayerState.health += 50;
		Destroy (gameObject);
	}
}
