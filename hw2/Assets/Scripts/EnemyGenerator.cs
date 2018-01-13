using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnemyGenerator : MonoBehaviour {

	public int maxEnemy = 5;
	public float generatePeriod = 3f;
	public GameObject enemy;
	public GameObject enemyHP;
	public GameObject terrain;
	public GameObject canvas;
	public static int enemyCount = 0;
	private float gap = 0f;
	private float minX;
	private float maxX;
	private float minY;
	private float maxY;

	// Use this for initialization
	void Start () {
		Vector3 terrainSize = terrain.GetComponent<Terrain>().terrainData.size;
		minX = 10f;
		maxX = terrainSize.x - 10f;
		minY = 10f;
		maxY = terrainSize.z - 10f;
		while (enemyCount < maxEnemy) {
			CreateEnemy ();
			enemyCount += 1;
		}
	}

	// Update is called once per frame
	void Update () {
		gap += Time.deltaTime;
		if (gap >= generatePeriod && enemyCount < maxEnemy) {
			CreateEnemy();
			gap = 0f;
			enemyCount += 1;
		}
	}

	void CreateEnemy(){
		float x = Random.Range (minX, maxX);
		float z = Random.Range (minY, maxY);
		GameObject enemyObject = Instantiate (enemy, new Vector3 (x, 1.5f, z), Quaternion.identity);
		GameObject enemyHpObject = Instantiate (enemyHP, new Vector3 (0, 0, 0), Quaternion.identity);
		enemyHpObject.transform.SetParent (canvas.transform);
		MovingHpBar hpBar = enemyHpObject.GetComponent<MovingHpBar> ();
		hpBar.target = enemyObject;
	}
}
