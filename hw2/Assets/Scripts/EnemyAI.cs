using UnityEngine;
using System.Collections;

public class EnemyAI : MonoBehaviour {
    public Rigidbody projcetile;
    public bool needfire;
	public GameObject firePoint;
    private GameObject player;
    private float patrolTimer;
    private int wayPointIndex;

    float speed = 40;
    int counter = 0;
    // Use this for initialization
    void Start () {
        player = GameObject.FindGameObjectWithTag("Player");
    }
	
	// Update is called once per frame
	void Update () {
        //needfire = false;
    }

 
    void OnTriggerStay(Collider other)
    {
		if (PlayerState.alive && other.gameObject == player)
        {
            Vector3 direction = other.transform.position - transform.position;
            Quaternion rotation = Quaternion.LookRotation(direction * -1f);
            transform.rotation = Quaternion.Slerp(transform.rotation, rotation, Time.deltaTime);
            transform.Translate(0f, 0f, -0.03f);
            if (counter >= 50)
            {
                Rigidbody shoot =
					(Rigidbody)Instantiate(projcetile, firePoint.transform.position, transform.rotation);
                //給砲彈方向力，將他從y軸推出去
                shoot.velocity = transform.TransformDirection(new Vector3(0f, 3f, -1f * speed));
                //讓坦克的碰撞框忽略砲彈的碰撞框
                Physics.IgnoreCollision(transform.root.GetComponent<Collider>(), shoot.GetComponent<Collider>());
                counter = 0;
            }
            else
                counter++;
        }
    }
}
