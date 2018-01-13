
public var barrelCamera : Camera;
public var backCamera : Camera;
function Start () {
    barrelCamera.enabled = false;
    backCamera.enabled = true;
}

function Update () {
    if (Input.GetKeyDown(KeyCode.LeftShift)) {
        barrelCamera.enabled = !barrelCamera.enabled;
        backCamera.enabled = !backCamera.enabled;
    }
}