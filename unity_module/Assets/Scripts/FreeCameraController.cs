using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(CharacterController))]
public class FreeCameraController : MonoBehaviour
{
    public Transform boundsBox; // assign the cube here

    public float moveSpeed = 5f;
    public float fastSpeed = 10f;
    public float mouseSensitivity = 0.15f;
    public float zoomSpeed = 10f;

    private float yaw = 0f;
    private float pitch = 0f;

    private Vector3 minBounds;
    private Vector3 maxBounds;

    private CharacterController controller;

    void Start()
    {
        controller = GetComponent<CharacterController>();

        Vector3 currentRotation = transform.eulerAngles;
        yaw = currentRotation.y;
        pitch = currentRotation.x;

        // Auto calculate bounds from cube
        Vector3 center = boundsBox.position;
        Vector3 size = boundsBox.localScale;

        minBounds = center - size / 2f;
        maxBounds = center + size / 2f;
    }

    void Update()
    {
        HandleMouseLook();
        HandleMovement();
        HandleZoom();
        ClampPosition();
    }

    void HandleMouseLook()
    {
        if (Mouse.current.rightButton.isPressed)
        {
            Vector2 mouseDelta = Mouse.current.delta.ReadValue();

            yaw += mouseDelta.x * mouseSensitivity;
            pitch -= mouseDelta.y * mouseSensitivity;
            pitch = Mathf.Clamp(pitch, -80f, 80f);

            transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
        }
    }

    void HandleMovement()
    {
        float speed = Keyboard.current.leftShiftKey.isPressed ? fastSpeed : moveSpeed;

        Vector3 move = Vector3.zero;

        if (Keyboard.current.wKey.isPressed) move += transform.forward;
        if (Keyboard.current.sKey.isPressed) move -= transform.forward;
        if (Keyboard.current.aKey.isPressed) move -= transform.right;
        if (Keyboard.current.dKey.isPressed) move += transform.right;

        if (Keyboard.current.qKey.isPressed) move += Vector3.down;
        if (Keyboard.current.eKey.isPressed) move += Vector3.up;

        // IMPORTANT: Use CharacterController instead of transform
        controller.Move(move * speed * Time.deltaTime);
    }

    void HandleZoom()
    {
        float scroll = Mouse.current.scroll.ReadValue().y;

        Vector3 zoomMove = transform.forward * scroll * zoomSpeed * 0.01f;

        // Use collision-safe movement for zoom too
        controller.Move(zoomMove);
    }

    void ClampPosition()
    {
        Vector3 pos = transform.position;

        pos.x = Mathf.Clamp(pos.x, minBounds.x, maxBounds.x);
        pos.y = Mathf.Clamp(pos.y, minBounds.y, maxBounds.y);
        pos.z = Mathf.Clamp(pos.z, minBounds.z, maxBounds.z);

        transform.position = pos;
    }
}