using UnityEngine;
using System.Collections;

public class TrackMove : MonoBehaviour {

	public GameObject barrel; // 砲管

	float towerSpeed = 2;
	float barrelSpeed = 2;
	float maxY = 80;
	float minY = -5;
	float maxX = 90;
	float minX = -90;
	Vector3 maxVectorX;
	Vector3 minVectorX;
	Vector3 maxVectorY;
	Vector3 minVectorY;

	void Start () {
		maxVectorX = minVectorX = transform.localEulerAngles;
		maxVectorX.y = maxX;
		minVectorX.y = 360f + minX;

		maxVectorY = minVectorY = barrel.transform.localEulerAngles;
		maxVectorY.x = maxY;
		minVectorY.x = 360f + minY;
	}

	void Update () {
		if (PlayerState.alive) {
			MouseMove ();
			KeyMove ();
			SetLimit ();
		}
	}

	void MouseMove() { // 用滑鼠移動控制砲台方向
		float x = Input.GetAxis ("Mouse X"); // 砲台
		if (x != 0f) {
			transform.Rotate (0, 0, towerSpeed * x);
		}

		float y = Input.GetAxis ("Mouse Y") * barrelSpeed; // 砲管
		if (y != 0f) {
			barrel.transform.Rotate (barrelSpeed * y, 0, 0);
		}
	}

	void KeyMove() { // 用鍵盤控制砲台方向
		if (Input.GetKey (KeyCode.LeftArrow)) {
			transform.Rotate (0, 0, -towerSpeed);
		} else if (Input.GetKey (KeyCode.RightArrow)) {
			transform.Rotate (0, 0, towerSpeed);
		}

		if (Input.GetKey (KeyCode.UpArrow)) {
			barrel.transform.Rotate (barrelSpeed, 0, 0);
		} else if (Input.GetKey (KeyCode.DownArrow)) {
			barrel.transform.Rotate (-barrelSpeed, 0, 0);
		}
	}

	void SetLimit() {
		float currentX = transform.localEulerAngles.y;
		if (currentX > maxX && currentX < 180f) {
			transform.localEulerAngles = maxVectorX;
		} else if (currentX < 360f + minX && currentX > 180f) {
			transform.localEulerAngles = minVectorX;
		}

		float currentY = barrel.transform.localEulerAngles.x;
		if (currentY > maxY && currentY < 180f) {
			barrel.transform.localEulerAngles = maxVectorY;
		} else if (currentY < 360f + minY && currentY > 180f) {
			barrel.transform.localEulerAngles = minVectorY;
		}
	}
}
