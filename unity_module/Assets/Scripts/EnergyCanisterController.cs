using UnityEngine;

public class EnergyCanisterController : MonoBehaviour
{
    [Header("Actuator State (DO NOT control manually in production)")]
    [SerializeField] private bool isOn = false;

    [Header("Assign Point Light")]
    public Light statusLight;

    // Optional: for testing inside Unity editor
    [Header("Debug (Optional)")]
    public bool debugManualControl = false;

    void Start()
    {
        ApplyState();
    }

    void Update()
    {
        // Only allow manual control if explicitly enabled
        if (debugManualControl)
        {
            ApplyState();
        }
    }

    void ApplyState()
    {
        if (statusLight != null)
        {
            statusLight.enabled = isOn;
        }
    }

    // 🔥 MAIN METHOD (called from integration layer)
    public void SetState(bool state)
    {
        isOn = state;
        ApplyState();
    }

    // Optional helpers (can be used for testing)
    public void TurnOn() => SetState(true);
    public void TurnOff() => SetState(false);
}