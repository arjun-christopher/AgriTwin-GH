using UnityEngine;

public class WindowFanController : MonoBehaviour
{
    [Header("Actuator State (DO NOT control manually in production)")]
    [SerializeField] private bool isOn = false;

    [Header("Assign References")]
    public Light statusIndicatorLight;
    public ParticleSystem airFlowParticle;

    private AudioSource fanAudio;

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
        fanAudio = GetComponentInParent<AudioSource>();
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

        if (fanAudio != null)
        {
            if (isOn)
            {
                if (!fanAudio.isPlaying)
                    fanAudio.Play();
            }
            else
            {
                if (fanAudio.isPlaying)
                    fanAudio.Stop();
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