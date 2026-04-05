using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// GreenhouseStateApplier
///
/// Central integration controller for the greenhouse Unity scene.
/// Polls a FastAPI endpoint at a configurable interval,
/// detects content changes, and applies the full greenhouse state to all
/// scene controllers automatically during Play Mode.
/// </summary>
public class GreenhouseStateApplier : MonoBehaviour
{
    [Header("API Settings")]
    [Tooltip("Full API endpoint URL that returns the 3D greenhouse scene state JSON.")]
    public string apiUrl = "http://localhost:8000/api/greenhouse-3d/state";

    [Tooltip("How often (seconds) to poll the API for updates.")]
    public float pollIntervalSeconds = 1.0f;

    [Tooltip("Timeout for each API request in seconds.")]
    public int requestTimeoutSeconds = 5;

    [Header("Actuator Controllers")]
    public FluorescentLightController fluorescentLight;
    public HeaterController heater;
    public EnergyCanisterController energyCanister;
    public HumidifierController humidifier;
    public WindowFanController windowFan;
    public VentController vent;
    public WaterTankFloorController waterTankFloor;

    [Header("Environment and Crop Controllers")]
    public CropStageController[] cropStages;
    public TimeOfDayController timeOfDay;
    public CropHealthIndicator cropHealth;

    [Header("Debug")]
    public bool verboseLogging = true;

    [Tooltip("Tick in Play Mode to force an immediate API refresh.")]
    public bool forceRefresh = false;

    [Tooltip("When true, API polling is active. Turn off if you want manual demo control.")]
    public bool enableApiPolling = true;

    private string _lastJsonContent = null;
    private GreenhouseState _lastAppliedState = null;
    private Coroutine _pollingCoroutine = null;
    private bool _isFetching = false;

    void Start()
    {
        Log("Started. API polling endpoint: " + apiUrl);

        if (enableApiPolling)
        {
            _pollingCoroutine = StartCoroutine(ApiPollingLoop());
        }
    }

    void Update()
    {
        if (forceRefresh)
        {
            forceRefresh = false;
            Log("Force refresh triggered.");
            if (!_isFetching)
            {
                StartCoroutine(FetchAndApplyFromApi(force: true));
            }
        }

        // If user toggles API polling at runtime
        if (enableApiPolling && _pollingCoroutine == null)
        {
            _pollingCoroutine = StartCoroutine(ApiPollingLoop());
            Log("API polling enabled.");
        }
        else if (!enableApiPolling && _pollingCoroutine != null)
        {
            StopCoroutine(_pollingCoroutine);
            _pollingCoroutine = null;
            Log("API polling disabled.");
        }
    }

    void LateUpdate()
    {
        if (_lastAppliedState == null || cropHealth == null) return;

        bool allRipe = AllCropsRipe(_lastAppliedState.cropStage);
        ApplyCropHealth(_lastAppliedState.cropHealth, allRipe);
    }

    IEnumerator ApiPollingLoop()
    {
        while (enableApiPolling)
        {
            if (!_isFetching)
            {
                yield return FetchAndApplyFromApi(force: false);
            }

            yield return new WaitForSeconds(pollIntervalSeconds);
        }

        _pollingCoroutine = null;
    }

    IEnumerator FetchAndApplyFromApi(bool force = false)
    {
        _isFetching = true;

        using (UnityWebRequest request = UnityWebRequest.Get(apiUrl))
        {
            request.timeout = requestTimeoutSeconds;
            yield return request.SendWebRequest();

#if UNITY_2020_1_OR_NEWER
            bool hasError = request.result != UnityWebRequest.Result.Success;
#else
            bool hasError = request.isNetworkError || request.isHttpError;
#endif

            if (hasError)
            {
                Debug.LogError("[GreenhouseStateApplier] API request failed: " + request.error);
                _isFetching = false;
                yield break;
            }

            string json = request.downloadHandler.text;

            if (string.IsNullOrWhiteSpace(json))
            {
                Debug.LogWarning("[GreenhouseStateApplier] API returned empty JSON.");
                _isFetching = false;
                yield break;
            }

            if (!force && json == _lastJsonContent)
            {
                _isFetching = false;
                yield break;
            }

            _lastJsonContent = json;
            Log("API change detected — parsing and applying greenhouse state...");

            GreenhouseState state = ParseJson(json);
            if (state == null)
            {
                _isFetching = false;
                yield break;
            }

            ApplyState(state);
        }

        _isFetching = false;
    }

    GreenhouseState ParseJson(string json)
    {
        try
        {
            GreenhouseState state = JsonUtility.FromJson<GreenhouseState>(json);

            if (state == null)
            {
                Debug.LogError("[GreenhouseStateApplier] Parsed state is null. Check API JSON structure.");
                return null;
            }

            return state;
        }
        catch (Exception ex)
        {
            Debug.LogError("[GreenhouseStateApplier] JSON parse error: " + ex.Message +
                           "\nCheck that the API response matches the expected schema.");
            return null;
        }
    }

    void ApplyState(GreenhouseState state)
    {
        ApplyFluorescentLight(state.fluorescentLight);
        ApplyHeater(state.heater);
        ApplyEnergyCanister(state.energyCanister);
        ApplyHumidifier(state.humidifier);
        ApplyWindowFan(state.windowFan);
        ApplyVent(state.vent);
        ApplyWaterTankFloor(state.waterTankFloor);
        ApplyCropStages(state.cropStage);
        ApplyTimeOfDay(state.timeOfDay);

        bool allRipe = AllCropsRipe(state.cropStage);
        ApplyCropHealth(state.cropHealth, allRipe);

        _lastAppliedState = state;
        Log("All states applied successfully.");
    }

    void ApplyFluorescentLight(ActuatorState s)
    {
        if (s == null) return;
        if (!AssertRef(fluorescentLight, "FluorescentLightController")) return;
        fluorescentLight.SetState(s.isOn);
        Log("FluorescentLight -> " + s.isOn);
    }

    void ApplyHeater(ActuatorState s)
    {
        if (s == null) return;
        if (!AssertRef(heater, "HeaterController")) return;
        heater.SetState(s.isOn);
        Log("Heater -> " + s.isOn);
    }

    void ApplyEnergyCanister(ActuatorState s)
    {
        if (s == null) return;
        if (!AssertRef(energyCanister, "EnergyCanisterController")) return;
        energyCanister.SetState(s.isOn);
        Log("EnergyCanister -> " + s.isOn);
    }

    void ApplyHumidifier(ActuatorState s)
    {
        if (s == null) return;
        if (!AssertRef(humidifier, "HumidifierController")) return;
        humidifier.SetState(s.isOn);
        Log("Humidifier -> " + s.isOn);
    }

    void ApplyWindowFan(ActuatorState s)
    {
        if (s == null) return;
        if (!AssertRef(windowFan, "WindowFanController")) return;
        windowFan.SetState(s.isOn);
        Log("WindowFan -> " + s.isOn);
    }

    void ApplyVent(ActuatorState s)
    {
        if (s == null) return;
        if (!AssertRef(vent, "VentController")) return;
        vent.SetState(s.isOn);
        Log("Vent -> " + s.isOn);
    }

    void ApplyWaterTankFloor(ActuatorState s)
    {
        if (s == null) return;
        if (!AssertRef(waterTankFloor, "WaterTankFloorController")) return;
        waterTankFloor.SetState(s.isOn);
        Log("WaterTankFloor -> " + s.isOn);
    }

    void ApplyCropStages(CropStageState stage)
    {
        if (stage == null || string.IsNullOrWhiteSpace(stage.stage)) return;

        if (cropStages == null || cropStages.Length == 0)
        {
            Debug.LogWarning("[GreenhouseStateApplier] 'cropStages' array is empty in the Inspector — skipping all crops.");
            return;
        }

        for (int i = 0; i < cropStages.Length; i++)
        {
            if (cropStages[i] == null)
            {
                Debug.LogWarning("[GreenhouseStateApplier] cropStages[" + i + "] is null in the Inspector — skipping.");
                continue;
            }

            cropStages[i].SetStageByName(stage.stage);
            Log("Crop[" + i + "] stage -> " + stage.stage);
        }
    }

    void ApplyTimeOfDay(TimeOfDayState s)
    {
        if (s == null || string.IsNullOrWhiteSpace(s.time)) return;
        if (!AssertRef(timeOfDay, "TimeOfDayController")) return;

        timeOfDay.SetTimeOfDayByName(s.time);
        Log("TimeOfDay -> " + s.time);
    }

    void ApplyCropHealth(CropHealthState s, bool allRipe)
    {
        if (!AssertRef(cropHealth, "CropHealthIndicator")) return;

        if (allRipe)
        {
            cropHealth.SetGreen();
            cropHealth.SetBlinkGreen(true);
            Log("CropHealth -> Green (blinking) [auto-overridden: all crops Ripe]");
            return;
        }

        if (s == null || string.IsNullOrWhiteSpace(s.state)) return;

        switch (s.state.Trim().ToLower())
        {
            case "green":
                cropHealth.SetGreen();
                break;
            case "yellow":
                cropHealth.SetYellow();
                break;
            case "red":
                cropHealth.SetRed();
                break;
            default:
                Debug.LogWarning("[GreenhouseStateApplier] Unknown crop health state: '" + s.state +
                                 "'. Valid values: Green, Yellow, Red");
                return;
        }

        cropHealth.SetBlinkGreen(false);
        Log("CropHealth -> " + s.state + " (no blink)");
    }

    bool AllCropsRipe(CropStageState stage)
    {
        if (stage == null || string.IsNullOrWhiteSpace(stage.stage)) return false;
        if (cropStages == null || cropStages.Length == 0) return false;

        return stage.stage.Trim().ToLower() == "ripe";
    }

    bool AssertRef(UnityEngine.Object obj, string controllerName)
    {
        if (obj != null) return true;

        Debug.LogWarning("[GreenhouseStateApplier] '" + controllerName +
                         "' is not assigned in the Inspector — skipping.");
        return false;
    }

    void Log(string msg)
    {
        if (verboseLogging)
            Debug.Log("[GreenhouseStateApplier] " + msg);
    }
}

[Serializable]
public class GreenhouseState
{
    public ActuatorState fluorescentLight;
    public ActuatorState heater;
    public ActuatorState energyCanister;
    public ActuatorState humidifier;
    public ActuatorState windowFan;
    public ActuatorState vent;
    public ActuatorState waterTankFloor;
    public CropStageState cropStage;
    public TimeOfDayState timeOfDay;
    public CropHealthState cropHealth;
}

[Serializable]
public class ActuatorState
{
    public bool isOn;
}

[Serializable]
public class CropStageState
{
    public string stage;
}

[Serializable]
public class TimeOfDayState
{
    public string time;
}

[Serializable]
public class CropHealthState
{
    public string state;
    public bool blinkGreen;
}