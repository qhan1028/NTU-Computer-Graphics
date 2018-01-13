using UnityEngine;
using System.Collections;

public class TankMove : MonoBehaviour {

	float moveSpeed = 0.3f;
	float rotateSpeed = 3f;
	Rigidbody rb;
	Vector3 movement;
	Quaternion left, right;

	void Start () {
		rb = gameObject.GetComponent<Rigidbody> ();
		right = Quaternion.Euler (0, rotateSpeed, 0);
		left = Quaternion.Euler (0, -rotateSpeed, 0);
	}

	void Update () {
		if (PlayerState.alive) {
			KeyMove ();
		}
    }

	void KeyMove() {
		movement = transform.forward * moveSpeed;
		if (Input.GetKey (KeyCode.A)) {
			rb.MoveRotation (rb.rotation * left);
		} else if (Input.GetKey (KeyCode.D)) {
			rb.MoveRotation (rb.rotation * right);
		}
		if (Input.GetKey (KeyCode.W)) {
			rb.MovePosition (rb.position - movement);
		} else if (Input.GetKey (KeyCode.S)) {
			rb.MovePosition (rb.position + movement);
		}
	}
}
