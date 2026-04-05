using UnityEngine;
using UnityEngine.InputSystem;

public class TimeOfDayController : MonoBehaviour
{
    public enum TimeOfDay
    {
        Morning,
        Afternoon,
        Evening,
        Night
    }

    [Header("Time State (DO NOT control manually in production)")]
    [SerializeField] private TimeOfDay currentTime = TimeOfDay.Morning;

    [Header("Main Light")]
    public Light directionalLight;

    [Header("Skyboxes")]
    public Material morningSkybox;
    public Material afternoonSkybox;
    public Material eveningSkybox;
    public Material nightSkybox;

    [Header("Night Lights")]
    public GameObject[] nightLights;

    [Header("Debug (Optional)")]
    public bool enableKeyboardDebugControl = true;
    public bool debugManualControl = true;

    private void Start()
    {
        ApplyTimeOfDay();
    }

    private void Update()
    {
        if (enableKeyboardDebugControl && Keyboard.current != null)
        {
            if (Keyboard.current[Key.Digit1].wasPressedThisFrame)
            {
                currentTime = TimeOfDay.Morning;
                ApplyTimeOfDay();
            }

            if (Keyboard.current[Key.Digit2].wasPressedThisFrame)
            {
                currentTime = TimeOfDay.Afternoon;
                ApplyTimeOfDay();
            }

            if (Keyboard.current[Key.Digit3].wasPressedThisFrame)
            {
                currentTime = TimeOfDay.Evening;
                ApplyTimeOfDay();
            }

            if (Keyboard.current[Key.Digit4].wasPressedThisFrame)
            {
                currentTime = TimeOfDay.Night;
                ApplyTimeOfDay();
            }
        }
        else if (debugManualControl)
        {
            ApplyTimeOfDay();
        }
    }

    public void SetTimeOfDay(TimeOfDay timeOfDay)
    {
        currentTime = timeOfDay;
        ApplyTimeOfDay();
    }

    public void SetTimeOfDayByName(string timeName)
    {
        if (string.IsNullOrWhiteSpace(timeName))
            return;

        switch (timeName.Trim().ToLower())
        {
            case "morning":
                currentTime = TimeOfDay.Morning;
                break;
            case "afternoon":
                currentTime = TimeOfDay.Afternoon;
                break;
            case "evening":
                currentTime = TimeOfDay.Evening;
                break;
            case "night":
                currentTime = TimeOfDay.Night;
                break;
            default:
                Debug.LogWarning("Unknown time of day: " + timeName);
                return;
        }

        ApplyTimeOfDay();
    }

    public void ApplyTimeOfDay()
    {
        switch (currentTime)
        {
            case TimeOfDay.Morning:
                RenderSettings.skybox = morningSkybox;
                RenderSettings.ambientLight = new Color(0.75f, 0.72f, 0.65f);
                RenderSettings.fog = true;
                RenderSettings.fogColor = new Color(0.85f, 0.75f, 0.65f);

                if (directionalLight != null)
                {
                    directionalLight.transform.rotation = Quaternion.Euler(25f, 30f, 0f);
                    directionalLight.color = new Color(1f, 0.85f, 0.65f);
                    directionalLight.intensity = 1.1f;
                }

                SetNightLights(false);
                break;

            case TimeOfDay.Afternoon:
                RenderSettings.skybox = afternoonSkybox;
                RenderSettings.ambientLight = new Color(0.9f, 0.9f, 0.85f);
                RenderSettings.fog = false;

                if (directionalLight != null)
                {
                    directionalLight.transform.rotation = Quaternion.Euler(60f, 0f, 0f);
                    directionalLight.color = Color.white;
                    directionalLight.intensity = 1.4f;
                }

                SetNightLights(false);
                break;

            case TimeOfDay.Evening:
                RenderSettings.skybox = eveningSkybox;
                RenderSettings.ambientLight = new Color(0.6f, 0.5f, 0.45f);
                RenderSettings.fog = true;
                RenderSettings.fogColor = new Color(0.8f, 0.5f, 0.35f);

                if (directionalLight != null)
                {
                    directionalLight.transform.rotation = Quaternion.Euler(15f, 220f, 0f);
                    directionalLight.color = new Color(1f, 0.55f, 0.3f);
                    directionalLight.intensity = 0.8f;
                }

                SetNightLights(false);
                break;

            case TimeOfDay.Night:
                RenderSettings.skybox = nightSkybox;
                RenderSettings.ambientLight = new Color(0.15f, 0.18f, 0.25f);
                RenderSettings.fog = true;
                RenderSettings.fogColor = new Color(0.08f, 0.1f, 0.15f);

                if (directionalLight != null)
                {
                    directionalLight.transform.rotation = Quaternion.Euler(-10f, 0f, 0f);
                    directionalLight.color = new Color(0.4f, 0.45f, 0.6f);
                    directionalLight.intensity = 0.2f;
                }

                SetNightLights(true);
                break;
        }

        DynamicGI.UpdateEnvironment();
    }

    void SetNightLights(bool state)
    {
        if (nightLights == null) return;

        foreach (GameObject lightObj in nightLights)
        {
            if (lightObj != null)
                lightObj.SetActive(state);
        }
    }
}