using UnityEngine;

public class VentController : MonoBehaviour
{
    [Header("Actuator State (DO NOT control manually in production)")]
    [SerializeField] private bool isOn = false;

    [Header("Assign References")]
    public Light statusIndicatorLight;
    public ParticleSystem airFlowParticle;

    private AudioSource ventAudio;
    private bool lastState;

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
        ventAudio = GetComponentInParent<AudioSource>();
    }

    void ApplyState()
    {
        if (statusIndicatorLight != null)
            statusIndicatorLight.enabled = isOn;

        if (airFlowParticle != null)
        {
            if (isOn)
            {
                if (!airFlowParticle.isPlaying)
                    airFlowParticle.Play();
            }
            else
            {
                if (airFlowParticle.isPlaying)
                    airFlowParticle.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
            }
        }

        if (ventAudio != null && Application.isPlaying)
        {
            if (isOn != lastState)
            {
                ventAudio.Play();
            }
        }

        lastState = isOn;
    }

    public void SetState(bool state)
    {
        isOn = state;
        ApplyState();
    }

    public void TurnOn() => SetState(true);
    public void TurnOff() => SetState(false);
}