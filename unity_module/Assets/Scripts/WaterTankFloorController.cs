using UnityEngine;

public class WaterTankFloorController : MonoBehaviour
{
    [Header("Actuator State (DO NOT control manually in production)")]
    [SerializeField] private bool isOn = false;

    private Light statusLight;
    private AudioSource audioSource;

    [Header("Debug (Optional)")]
    public bool debugManualControl = false;

    void Start()
    {
        Setup();
        ApplyState();
    }

    void Update()
    {
        if (debugManualControl)
        {
            ApplyState();
        }
    }

    void Setup()
    {
        Transform indicator = transform.Find("StatusIndicator");

        if (indicator != null)
        {
            statusLight = indicator.GetComponentInChildren<Light>();
        }

        audioSource = GetComponent<AudioSource>();
    }

    void ApplyState()
    {
        if (statusLight != null)
            statusLight.enabled = isOn;

        if (audioSource != null)
        {
            if (isOn)
            {
                if (!audioSource.isPlaying)
                    audioSource.Play();
            }
            else
            {
                if (audioSource.isPlaying)
                    audioSource.Stop();
            }
        }
    }

    public void SetState(bool state)
    {
        isOn = state;
        ApplyState();
    }

    public void TurnOn() => SetState(true);
    public void TurnOff() => SetState(false);
}