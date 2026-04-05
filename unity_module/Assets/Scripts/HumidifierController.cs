using UnityEngine;

public class HumidifierController : MonoBehaviour
{
    [Header("Actuator State (DO NOT control manually in production)")]
    [SerializeField] private bool isOn = false;

    [Header("Assign References")]
    public Light statusIndicatorLight;
    public ParticleSystem fogParticle;

    private AudioSource fogAudio;

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
        fogAudio = GetComponentInParent<AudioSource>();
    }

    void ApplyState()
    {
        if (statusIndicatorLight != null)
            statusIndicatorLight.enabled = isOn;

        if (fogParticle != null)
        {
            if (isOn)
            {
                if (!fogParticle.isPlaying)
                    fogParticle.Play();
            }
            else
            {
                if (fogParticle.isPlaying)
                    fogParticle.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
            }
        }

        if (fogAudio != null)
        {
            if (isOn)
            {
                if (!fogAudio.isPlaying)
                    fogAudio.Play();
            }
            else
            {
                if (fogAudio.isPlaying)
                    fogAudio.Stop();
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