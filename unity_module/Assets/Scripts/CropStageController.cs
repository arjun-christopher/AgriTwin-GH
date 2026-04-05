using UnityEngine;

public class CropStageController : MonoBehaviour
{
    [System.Serializable]
    public enum GrowthStage
    {
        Seedling = 0,
        Vegetative = 1,
        FloweringInitiation = 2,
        Flowering = 3,
        Unripe = 4,
        Ripe = 5
    }

    [Header("Growth Stage (DO NOT control manually in production)")]
    [SerializeField] private GrowthStage currentStage = GrowthStage.Seedling;

    [Header("Stage Objects (in order)")]
    public GameObject seedling;
    public GameObject vegetative;
    public GameObject floweringInitiation;
    public GameObject flowering;
    public GameObject unripe;
    public GameObject ripe;

    [Header("Auto Found Health Indicator")]
    public CropHealthIndicator healthIndicator;

    private GameObject[] stages;

    [Header("Debug (Optional)")]
    public bool debugManualControl = false;

    void Start()
    {
        SetupStages();
        FindHealthIndicator();
        ApplyStage();
    }

    void Update()
    {
        if (debugManualControl)
        {
            SetupStages();
            FindHealthIndicator();
            ApplyStage();
        }
    }

    void SetupStages()
    {
        stages = new GameObject[]
        {
            seedling,
            vegetative,
            floweringInitiation,
            flowering,
            unripe,
            ripe
        };
    }

    void FindHealthIndicator()
    {
        if (healthIndicator == null)
            healthIndicator = GetComponentInParent<CropHealthIndicator>();

        if (healthIndicator == null && transform.parent != null)
            healthIndicator = transform.parent.GetComponent<CropHealthIndicator>();

        if (healthIndicator == null && transform.parent != null)
            healthIndicator = transform.parent.GetComponentInChildren<CropHealthIndicator>();
    }

    void ApplyStage()
    {
        if (stages == null || stages.Length == 0)
            return;

        int currentStageIndex = (int)currentStage;

        for (int i = 0; i < stages.Length; i++)
        {
            if (stages[i] != null)
                stages[i].SetActive(i == currentStageIndex);
        }

        if (healthIndicator != null)
        {
            if (currentStage == GrowthStage.Ripe)
            {
                healthIndicator.SetGreen();
                healthIndicator.SetBlinkGreen(true);
            }
            else
            {
                healthIndicator.SetBlinkGreen(false);
            }
        }
    }

    public void SetStage(GrowthStage stage)
    {
        currentStage = stage;
        ApplyStage();
    }

    public void SetStageByIndex(int index)
    {
        index = Mathf.Clamp(index, 0, 5);
        currentStage = (GrowthStage)index;
        ApplyStage();
    }

    public void SetStageByName(string stageName)
    {
        if (string.IsNullOrWhiteSpace(stageName))
            return;

        switch (stageName.Trim().ToLower())
        {
            case "seedling":
                currentStage = GrowthStage.Seedling;
                break;
            case "vegetative":
            case "early vegetative":
                currentStage = GrowthStage.Vegetative;
                break;
            case "flowering initiation":
            case "flowering_initiation":
            case "floweringinitiation":
                currentStage = GrowthStage.FloweringInitiation;
                break;
            case "flowering":
                currentStage = GrowthStage.Flowering;
                break;
            case "unripe":
                currentStage = GrowthStage.Unripe;
                break;
            case "ripe":
                currentStage = GrowthStage.Ripe;
                break;
            default:
                Debug.LogWarning("Unknown stage name: " + stageName);
                return;
        }

        ApplyStage();
    }

    public void NextStage()
    {
        int nextIndex = Mathf.Clamp((int)currentStage + 1, 0, 5);
        currentStage = (GrowthStage)nextIndex;
        ApplyStage();
    }

    public void PreviousStage()
    {
        int previousIndex = Mathf.Clamp((int)currentStage - 1, 0, 5);
        currentStage = (GrowthStage)previousIndex;
        ApplyStage();
    }
}