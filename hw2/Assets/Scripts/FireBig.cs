using UnityEngine;
using System.Collections;

public class FireBig : MonoBehaviour {

	public Rigidbody projcetile;
	public float period = 1f;
	public float speed = 30;
	float gap;


	// Use this for initialization
	void Start () {
		gap = 0f;
	}

	// Update is called once per frame
	void Update () {
		gap += Time.deltaTime;
		//判斷是否按下按鍵
		if(PlayerState.alive && gap >= period && (Input.GetKey(KeyCode.Mouse1) || Input.GetKey(KeyCode.V)))
		{
			//產生砲彈在發射點
			Rigidbody shoot = (Rigidbody)Instantiate(projcetile, transform.position, transform.rotation);
			//給砲彈方向力，將他從y軸推出去
			shoot.velocity = transform.TransformDirection(new Vector3( 0, speed, 0));
			//讓坦克的碰撞框忽略砲彈的碰撞框
			Physics.IgnoreCollision(transform.root.GetComponent<Collider>(), shoot.GetComponent<Collider>());

			gap = 0f;
		}
	}
}
