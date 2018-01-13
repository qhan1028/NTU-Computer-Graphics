using UnityEngine;
using System.Collections;

public class Explosion : MonoBehaviour {
	public GameObject hitEnemyEffect;
	public GameObject hitPlayerEffect;
	public GameObject hitGroundEffect;
	public bool hitEnemy;
	public bool hitPlayer;

	void Start () {
        Destroy(gameObject, 3);
    }	
	void Update () {
    }

	void OnCollisionEnter (Collision collision) {//碰撞發生時呼叫
		//碰撞後產生爆炸
		var tag = collision.gameObject.tag;
			
		if (tag == "enemy" && hitEnemy) {//當撞到的collider具有enemy tag
			Instantiate (hitEnemyEffect, transform.position, transform.rotation);
			Destroy (gameObject);//刪除砲彈
		} else if (tag == "Player" && hitPlayer) {
			Instantiate (hitPlayerEffect, transform.position, transform.rotation);
			Destroy (gameObject);//刪除砲彈
		} else if (tag != "bullet1" && tag != "bullet2"){
			Instantiate (hitGroundEffect, transform.position, transform.rotation);
			Destroy (gameObject);//刪除砲彈
		}
	}
}
