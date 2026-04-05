using UnityEngine;

public class HeaterController : MonoBehaviour
{
    [Header("Actuator State (DO NOT control manually in production)")]
    [SerializeField] private bool isOn = false;

    [Header("Assign Lights")]
    public Light statusIndicatorLight;   // green indicator
    public Light heaterLight;            // heater glow / point light

    [Header("Debug (Optional)")]
    public bool debugManualControl = false;

    void Start()
    {
        ApplyState();
    }

    void Update()
    {
        if (debugManualControl)
        {
            ApplyState();
        }
    }

    void ApplyState()
    {
        if (statusIndicatorLight != null)
            statusIndicatorLight.enabled = isOn;

        if (heaterLight != null)
            heaterLight.enabled = isOn;
    }

    public void SetState(bool state)
    {
        isOn = state;
        ApplyState();
    }

    public void TurnOn() => SetState(true);
    public void TurnOff() => SetState(false);
}