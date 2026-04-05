using UnityEngine;

public class CropHealthIndicator : MonoBehaviour
{
    public enum HealthState
    {
        Green,
        Yellow,
        Red
    }

    [Header("Health State (DO NOT control manually in production)")]
    [SerializeField] private HealthState currentState = HealthState.Green;

    [Header("Assign Lights")]
    public Light greenLight;
    public Light yellowLight;
    public Light redLight;

    [Header("Blink Settings")]
    public bool blinkGreen = false;
    public float blinkSpeed = 2f;

    private HealthState lastState;
    private bool blinkState = true;

    [Header("Debug (Optional)")]
    public bool debugManualControl = false;

    void Start()
    {
        ApplyState(true);
    }

    void Update()
    {
        if (Application.isPlaying)
        {
            if (blinkGreen && currentState == HealthState.Green)
            {
                bool newBlinkState = Mathf.FloorToInt(Time.time * blinkSpeed) % 2 == 0;

                if (newBlinkState != blinkState || currentState != lastState)
                {
                    blinkState = newBlinkState;
                    ApplyState(true);
                }
            }
            else if (debugManualControl && currentState != lastState)
            {
                ApplyState(true);
            }
        }
    }

    void ApplyState(bool force = false)
    {
        if (!force && Application.isPlaying && currentState == lastState)
            return;

        bool greenOn = (currentState == HealthState.Green);
        bool yellowOn = (currentState == HealthState.Yellow);
        bool redOn = (currentState == HealthState.Red);

        if (blinkGreen && currentState == HealthState.Green && Application.isPlaying)
            greenOn = blinkState;

        if (greenLight != null)
            greenLight.enabled = greenOn;

        if (yellowLight != null)
            yellowLight.enabled = yellowOn;

        if (redLight != null)
            redLight.enabled = redOn;

        lastState = currentState;
    }

    public void SetState(HealthState state)
    {
        currentState = state;
        ApplyState(true);
    }

    public void SetGreen()
    {
        currentState = HealthState.Green;
        ApplyState(true);
    }

    public void SetYellow()
    {
        currentState = HealthState.Yellow;
        blinkGreen = false;
        ApplyState(true);
    }

    public void SetRed()
    {
        currentState = HealthState.Red;
        blinkGreen = false;
        ApplyState(true);
    }

    public void SetBlinkGreen(bool shouldBlink)
    {
        blinkGreen = shouldBlink;
        ApplyState(true);
    }
}
