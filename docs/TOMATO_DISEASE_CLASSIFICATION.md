# Tomato Leaf Disease Classification

> **Who is this for?**  
> This document is written for anyone — farmer, student, developer, or curious reader — with zero prior knowledge of machine learning or plant pathology. Every concept is explained from the ground up, with analogies and plain language throughout.

---

## Table of Contents

1. [Why Does This Matter?](#1-why-does-this-matter)
2. [The Diseases We Classify](#2-the-diseases-we-classify)
   - [Early Blight](#21-early-blight)
   - [Late Blight](#22-late-blight)
   - [Leaf Mold](#23-leaf-mold)
   - [Powdery Mildew](#24-powdery-mildew)
   - [Spider Mites](#25-spider-mites-two-spotted)
   - [Healthy Leaf](#26-healthy-leaf)
3. [What is Image Classification?](#3-what-is-image-classification)
4. [How Does a Computer "See" a Leaf?](#4-how-does-a-computer-see-a-leaf)
5. [What is Deep Learning? (No maths required)](#5-what-is-deep-learning-no-maths-required)
6. [Transfer Learning — Standing on Giants' Shoulders](#6-transfer-learning--standing-on-giants-shoulders)
7. [Our Model: EfficientNetB0](#7-our-model-efficientnetb0)
8. [The Dataset](#8-the-dataset)
9. [Data Augmentation — Teaching with Variations](#9-data-augmentation--teaching-with-variations)
10. [Training the Model — Phase by Phase](#10-training-the-model--phase-by-phase)
11. [How We Measure Success](#11-how-we-measure-success)
12. [Deploying the Model — Making it Useful](#12-deploying-the-model--making-it-useful)
13. [End-to-End Flow Diagram](#13-end-to-end-flow-diagram)
14. [Common Questions (FAQ)](#14-common-questions-faq)
15. [Glossary](#15-glossary)

---

## 1. Why Does This Matter?

Tomatoes are one of the most widely grown vegetables on Earth. From small family gardens to large commercial greenhouses, the tomato crop feeds millions and drives significant agricultural revenue.

**The problem:** Tomato plants are highly susceptible to diseases caused by fungi, bacteria, and pests. A single infection spreading through a greenhouse can wipe out an entire crop in days. Traditionally, a farmer has to:

1. Walk through the greenhouse every day.
2. Carefully inspect each plant.
3. Identify the type of disease by eye.
4. Apply the correct treatment.

This process is **labour-intensive**, **error-prone** (many diseases look alike to the untrained eye), and **slow** — by the time a disease is spotted visually, it may already have spread to neighbouring plants.

**The solution this project provides:** A camera takes a photo of a tomato leaf. An AI model analyses the photo within milliseconds and tells you:
- Is this leaf healthy?
- If not, which disease affects it?

This enables **early, automated detection** at scale — checking every plant, every hour, without human fatigue.

---

## 2. The Diseases We Classify

Our model distinguishes between **5 diseases** and **1 healthy class**, giving 6 output categories in total.

### 2.1 Early Blight

| Property | Detail |
|----------|--------|
| **Cause** | Fungus: *Alternaria solani* |
| **Climate preference** | Warm and wet (24–29 °C, high humidity) |
| **Spreads via** | Wind-blown spores, rain splash, contaminated tools |

**What it looks like:**  
Dark brown, irregularly shaped spots appear on **older, lower leaves** first. The spots have distinctive **concentric rings** — like the rings inside a tree trunk — giving them a "bull's-eye" or "target" pattern. A yellow halo often surrounds each spot. As the disease progresses it moves up the plant, and affected leaves turn yellow and drop off.

**Why it matters:**  
It reduces the plant's ability to photosynthesise (make food from sunlight), weakening the whole plant and cutting fruit yield. Severe infections can cause 50–80% crop loss.

**Treatment:**  
Remove infected leaves immediately. Apply copper-based or chlorothalonil fungicides. Ensure good air circulation between plants.

---

### 2.2 Late Blight

| Property | Detail |
|----------|--------|
| **Cause** | Water mould (oomycete): *Phytophthora infestans* |
| **Climate preference** | Cool and wet (10–25 °C, > 90% humidity at night) |
| **Historical impact** | Caused the Irish Potato Famine (1845–49) |

**What it looks like:**  
Pale green or olive-coloured **irregular water-soaked patches** appear on leaves, quickly turning dark brown or black. Under humid conditions a white, fluffy mould growth appears on the **underside** of affected areas. The lesions spread rapidly and have an "oily" or "wet" appearance. Infected stems become dark and mushy.

**Why it matters:**  
Late blight is one of the most destructive plant diseases known. Unlike Early Blight's slow spread, Late Blight can devastate an entire crop within **3–5 days** under favourable conditions. It is classified as a **plant epidemic** risk.

**Treatment:**  
Immediately remove and destroy infected tissue. Apply systemic fungicides (mancozeb, metalaxyl). Reduce leaf wetness by adjusting irrigation and improving ventilation.

---

### 2.3 Leaf Mold

| Property | Detail |
|----------|--------|
| **Cause** | Fungus: *Passalora fulva* (formerly *Cladosporium fulvum*) |
| **Climate preference** | High humidity (> 85%) with moderate temperatures (22–24 °C) |
| **Spread** | Airborne spores, persists in soil and on surfaces for years |

**What it looks like:**  
Yellow-green **pale patches** on the **upper surface** of leaves, with corresponding **olive-green to brown velvety mould growth** on the **underside**. The mould has a dusty or furry texture. The yellow patches enlarge and coalesce, causing the entire leaf to yellow and wither.

**Why it matters:**  
Leaf Mold thrives precisely in the warm, humid conditions of **closed greenhouses** — making it a particularly relevant threat for indoor tomato cultivation. It primarily reduces photosynthesis and weakens the plant, with yield losses of 20–40% in severe cases.

**Treatment:**  
Increase air circulation; reduce relative humidity below 85%. Remove and destroy infected leaves. Apply fungicides (chlorothalonil, copper compounds).

---

### 2.4 Powdery Mildew

| Property | Detail |
|----------|--------|
| **Cause** | Fungi: *Leveillula taurica* or *Oidium neolycopersici* |
| **Climate preference** | Moderate temperatures (15–30 °C), DRY conditions (unlike other fungal diseases) |
| **Unique trait** | Spreads easily in dry, warm environments — does NOT need leaf wetness |

**What it looks like:**  
Distinctive **white or grey powdery coating** on the surface of leaves — mostly the upper side. It looks exactly like someone dusted talcum powder or flour on the leaf. The patches start small and circular, then merge to cover the entire leaf. Affected leaves curl, turn yellow, and eventually dry out.

**Why it matters:**  
Powdery Mildew is unique because many other fungal diseases slow down in dry conditions — Powdery Mildew does the opposite. Standard humidity-reduction strategies that prevent Leaf Mold can *inadvertently promote* Powdery Mildew. This makes it important to correctly identify the disease before choosing a treatment.

**Treatment:**  
Apply neem oil, sulphur-based fungicides, or potassium bicarbonate sprays. Ensure proper plant spacing. Some resistant tomato varieties are available.

---

### 2.5 Spider Mites (Two-Spotted)

| Property | Detail |
|----------|--------|
| **Cause** | Arachnid pest: *Tetranychus urticae* (not a fungus — these are tiny relatives of spiders!) |
| **Climate preference** | Hot and dry (> 28 °C, low humidity) |
| **Visible size** | ~0.5 mm — barely visible to the naked eye |

**What it looks like:**  
Unlike the other categories which are fungal diseases, spider mites are **tiny animal pests**. They feed by piercing leaf cells and sucking out the contents. This creates: tiny **yellow or white stippling dots** (puncture marks) scattered across the leaf surface; a **bronze or rusty discolouration** of the leaf; and in severe infestations, fine **silvery webbing** on the underside of leaves and between stems. The leaf eventually turns pale, dries out and falls off.

**Why it matters:**  
Spider mite populations **double every 3–5 days** in hot conditions. A single female can lay 100+ eggs in her 2-week lifespan. By the time webbing is visible, the infestation is already heavy. They also develop resistance to pesticides quickly.

**Treatment:**  
Increase humidity (mites hate moisture). Apply miticides, neem oil, or predatory mites (*Phytoseiulus persimilis*) as biological control. Avoid broad-spectrum insecticides that kill natural predators.

---

### 2.6 Healthy Leaf

A **healthy tomato leaf** is:
- Deep, uniform green (no spots, patches, or discolouration)
- Flat with smooth margins (no curling)
- Has a slightly rough/hairy texture
- Free of any webbing, powdery coating, or lesions

Accurately identifying healthy leaves is just as important as identifying diseases — it tells the system "no action needed here" and prevents unnecessary chemical applications.

---

## 3. What is Image Classification?

Imagine you show 1,000 photos of cats and dogs to a child. After a while, the child learns to tell them apart just by looking. **Image classification** is teaching a computer to do the same thing.

In our case:
- We show the computer thousands of photos of tomato leaves.
- Each photo is labelled with the correct disease name (or "healthy").
- The computer learns patterns — colours, textures, shapes, edges — that distinguish one class from another.
- After training, when shown a **new, unseen photo**, the computer predicts which class it belongs to.

The output is a set of **confidence percentages**, one per class. For example:

```
tomato_early_blight    : 78.3%
tomato_late_blight     :  9.1%
tomato_leaf_mold       :  5.2%
tomato_powdery_mildew  :  4.7%
tomato_spider_mites    :  1.4%
tomato_leaf_healthy    :  1.3%
```

The model picks the class with the highest confidence — in this case, *Early Blight*.

---

## 4. How Does a Computer "See" a Leaf?

A digital image is a **grid of pixels**. Each pixel has three numbers representing its colour — **Red, Green, Blue (RGB)** — each ranging from 0 to 255.

A 224×224 pixel image (our input size) is therefore a grid of:
```
224 rows × 224 columns × 3 colour channels = 150,528 numbers
```

The computer's job is to find meaning in these ~150,000 numbers. A simple approach would be to compare every number directly — but that doesn't work because:
- The same leaf photographed in slightly different lighting looks different in numbers.
- A leaf shifted slightly in the frame changes all the numbers.
- Rotation, zoom, or blur all change the raw numbers drastically.

We need a smarter approach — one that identifies **features** (edges, textures, patterns) regardless of these variations.

---

## 5. What is Deep Learning? (No maths required)

### The Analogy: A Layered Detective

Think of a detective who identifies a disease by asking a series of questions:

1. **Layer 1 (Basic shapes):** "Are there any sharp edges? Any curves?"
2. **Layer 2 (Textures):** "Is the surface rough? Powdery? Patchy?"
3. **Layer 3 (Patterns):** "Are there concentric rings? Yellow halos? White spots?"
4. **Layer 4 (Disease features):** "This combination of ring pattern + yellow halo = Early Blight"
5. **Final Layer (Decision):** "Confidence: 78% Early Blight"

A **deep neural network** does exactly this — it stacks many **layers** of pattern detectors, each layer learning increasingly complex features from the output of the previous layer.

### What is a Neuron?

A **neuron** is a tiny mathematical function. It takes numbers in, multiplies each by a **weight** (importance), adds them up, and outputs a new number. The "learning" is simply adjusting these weights based on mistakes.

### How Learning Happens

1. **Forward pass:** Show the network a leaf photo → it predicts a class.
2. **Measure the mistake:** Compare the prediction to the correct label. Calculate an error score (**loss**).
3. **Backward pass (backpropagation):** Adjust every weight slightly in the direction that reduces the error.
4. **Repeat** millions of times across thousands of images.

After enough repetitions, the weights settle into values that make good predictions. This process is called **training**.

### What is a Convolutional Neural Network (CNN)?

A CNN is a type of deep neural network specifically designed for images. Instead of connecting every pixel to every neuron (which would require billions of parameters), it uses **filters** (small sliding windows) that scan across the image detecting local features — just like how your eye notices the bull's-eye pattern of Early Blight without needing to see the whole leaf at once.

---

## 6. Transfer Learning — Standing on Giants' Shoulders

Training a powerful CNN from scratch requires:
- **Millions of images**
- **Weeks of compute time** on powerful GPU machines
- **Expert tuning**

We have neither the data volume nor the compute time for that. Instead, we use **Transfer Learning**.

### The Idea

Large technology companies (Google, Facebook, etc.) train massive models on millions of general images (dogs, cars, buildings, flowers). These models learn extremely powerful general features — edges, textures, patterns — that transfer well to almost any visual task.

We take one of these pre-trained models and **adapt it** to our specific task (tomato disease classification) by training only a small custom "head" on top, using our disease images.

**Analogy:** Instead of teaching someone to read from scratch, you hire a person who already reads English fluently and just teach them the specific medical terms they need for their new role. Far faster and more effective.

---

## 7. Our Model: EfficientNetB0

### What is EfficientNetB0?

**EfficientNet** is a family of neural network architectures developed by Google Brain in 2019. The "B0" variant is the smallest in the family — lightweight, fast, and accurate — making it ideal for deployment.

EfficientNet was designed using **Neural Architecture Search (NAS)**: an AI was used to design the optimal architecture, rather than humans hand-tuning it. The key insight is **compound scaling** — instead of making networks just deeper or wider, EfficientNet scales depth, width, and input resolution simultaneously in a balanced way.

### Why EfficientNetB0 for This Task?

| Reason | Explanation |
|--------|-------------|
| **Accuracy** | State-of-the-art performance despite small size |
| **Speed** | Fast inference — critical for real-time greenhouse monitoring |
| **Pre-training** | Already trained on ImageNet (1.28M images, 1,000 classes) |
| **Input size** | Designed for 224×224 images — matches our leaf photos |
| **Memory efficient** | Runs on hardware without requiring expensive GPUs |

### Our Custom Head

EfficientNetB0 acts as the **feature extractor** (backbone). On top of it, we add a small **custom head** that makes the final disease prediction:

```
EfficientNetB0 Backbone (feature extraction)
        ↓
  GlobalAveragePooling2D
  (collapses spatial dimensions into a single vector)
        ↓
  Dense(256 neurons) + Batch Normalisation + ReLU activation
        ↓
  Dropout(40%)    ← randomly switches off 40% of neurons during training
        ↓          to prevent overfitting
  Dense(128 neurons) + Batch Normalisation + ReLU activation
        ↓
  Dropout(30%)
        ↓
  Dense(6 neurons) + Softmax
  (one neuron per disease class; outputs probabilities summing to 1.0)
```

**Dropout** is a regularisation technique — during training, we randomly "switch off" a fraction of neurons. This forces the network not to rely too heavily on any single neuron, making it more robust and **reducing overfitting** (memorising training data instead of learning general patterns).

---

## 8. The Dataset

### Structure on Disk

```
data/
└── external/
    ├── Tomato Diseases/
    │   ├── Tomato_Early_Blight/        ← images of Early Blight leaves
    │   ├── Tomato_Late_Blight/         ← images of Late Blight leaves
    │   ├── Tomato_Leaf_Mold/           ← images of Leaf Mold leaves
    │   ├── Tomato_Powdery_Mildew/      ← images of Powdery Mildew leaves
    │   ├── Tomato_Spider_Mites/        ← images of Spider Mite damage
    │   └── Tomato_Septoria_Leaf_Spot/  ← excluded (see note below)
    └── Tomato Healthy Leaves/          ← images of healthy leaves
```

> **Why is Septoria Leaf Spot excluded?**  
> *Tomato_Septoria_Leaf_Spot* is present in the raw data but is deliberately excluded from training. The primary reason is **class imbalance** — it has far fewer samples than the other classes, which would bias the model. It can be added in a future update with more data or oversampling techniques.

### Class Labels

| Folder Name | Label Used by Model |
|-------------|---------------------|
| `Tomato_Early_Blight` | `tomato_early_blight` |
| `Tomato_Late_Blight` | `tomato_late_blight` |
| `Tomato_Leaf_Mold` | `tomato_leaf_mold` |
| `Tomato_Powdery_Mildew` | `tomato_powdery_mildew` |
| `Tomato_Spider_Mites` | `tomato_spider_mites` |
| `Tomato Healthy Leaves` | `tomato_leaf_healthy` |

### Train / Validation / Test Split

The dataset is divided into three non-overlapping subsets:

| Subset | Purpose | Size |
|--------|---------|------|
| **Training set (75%)** | Images the model learns from | ~8,490 images |
| **Validation set (15%)** | Used during training to check generalisation (model never trains on these) | ~1,699 images |
| **Test set (10%)** | Held out completely until final evaluation — the true measure of performance | ~1,133 images |

**Why three splits?**  
Using the same images for both training and measuring accuracy would be like a student memorising an exam answer sheet — the score would look great but mean nothing. The test set is the student's actual exam with questions they've never seen.

### Class Imbalance and Handling

In practice, some disease classes have more images than others. If the model sees 5× more Early Blight images than Spider Mites images, it will learn to be lazy and always guess Early Blight — achieving high accuracy on training data but poor real-world performance.

We address this with **class weights**: classes with fewer samples are given higher weights in the loss function, effectively penalising the model more for getting rare classes wrong.

```
Class weight = (total samples) / (number of classes × samples in this class)
```

A class with fewer images gets a higher weight → the model must pay more attention to it.

---

## 9. Data Augmentation — Teaching with Variations

### The Problem

Our dataset has ~11,000 images total. Deep learning models typically need far more. Also, real-world conditions differ from dataset conditions — a greenhouse camera might capture leaves from a different angle, in different lighting, or with slight motion blur.

### The Solution

**Data Augmentation** artificially expands the dataset by creating modified versions of existing images during training. The key insight: a rotated photo of an Early Blight leaf is *still* an Early Blight leaf. The model should learn to recognise it regardless.

### Augmentations Applied

Each training image is randomly subjected to these transformations:

| Augmentation | Effect | Why it Helps |
|---|---|---|
| **Horizontal flip** | Mirror the image left-to-right | Diseases appear on both sides of leaves |
| **Vertical flip** | Mirror the image top-to-bottom | Photos taken from different orientations |
| **Random rotation** | Rotate up to ±15° | Cameras rarely capture leaves perfectly aligned |
| **Random zoom** | In or out by up to 15% | Different camera distances |
| **Random brightness** | Darken or brighten by ±15% | Greenhouse lighting conditions vary |
| **Random contrast** | Increase or decrease contrast | Different camera settings |
| **Random hue** | Slight colour shift | Different white-balance settings |
| **Random saturation** | More or less vivid colours | Lighting quality variation |
| **Random crop** | Randomly crop 90% of the image | Focus on different parts of the leaf |
| **Cutout (Random Erasing)** | Black out a random square patch (15% of image) | Forces model to not rely on any single spot |

> **Important:** Augmentation is applied **only to the training set**. Validation and test sets use the original images — because we want to measure performance on realistic, unmodified inputs.

---

## 10. Training the Model — Phase by Phase

Training happens in **two phases**. This two-phase approach is a well-established best practice called **progressive fine-tuning**.

### Phase 1: Warmup (10 epochs, frozen backbone)

**Epoch** = one complete pass through the entire training dataset.

In Phase 1, the EfficientNetB0 backbone weights are **frozen** — they cannot change. Only our custom head layers are updated.

**Why?**  
The pre-trained backbone already knows how to detect visual features. If we immediately allow all layers to update with our small disease dataset, the powerful backbone weights get "corrupted" before the head has learned anything useful — a phenomenon called **catastrophic forgetting**. Warming up the head first gives it a sensible starting point.

- **Learning rate:** `0.001` (relatively high — the head is learning from scratch)
- **What updates:** Custom head only

### Phase 2: Fine-tuning (25 epochs, top layers unfrozen)

After the head is trained, we **unfreeze the top 30 layers** of the backbone and allow them to fine-tune on our data.

**Why only the top layers?**  
The bottom layers of a CNN learn very basic features (edges, corners) that are universal — useful for every image task. These don't need to change. The top layers learn high-level, task-specific features — these benefit from seeing disease images.

- **Learning rate:** `0.00005` (much lower — tiny adjustments to already-good backbone weights)
- **What updates:** Top 30 backbone layers + custom head

### Loss Function

The **loss function** measures how wrong the model's predictions are. During training, the optimiser tries to minimise this number.

We use **Categorical Cross-Entropy** with **label smoothing** (0.1).

Without label smoothing, the model is trained to output 100% certainty for the correct class. Label smoothing softens the targets — the model aims for 90% confidence for the correct class and distributes the remaining 10% across other classes. This prevents **overconfidence** and improves generalisation.

### Callbacks (Automatic Training Assistants)

Several automated mechanisms improve training:

| Callback | What it Does |
|----------|-------------|
| **ModelCheckpoint** | Saves the model whenever validation accuracy improves. You always keep the best version. |
| **ReduceLROnPlateau** | If validation loss stops improving for 3 epochs, automatically halves the learning rate. Helps escape plateaus. |
| **EarlyStopping** | If the model hasn't improved for 8 epochs, stop training early. Prevents overfitting and saves time. |
| **CSVLogger** | Logs loss and accuracy for every epoch to a CSV file for later analysis. |

### Mixed Precision Training

Modern GPUs process **16-bit floating point numbers** (FP16) much faster than traditional **32-bit** (FP32), using less memory. **Mixed precision** training uses FP16 for most operations but keeps FP32 where numerical precision matters (the loss calculation). This can **double training speed** with no accuracy loss.

---

## 11. How We Measure Success

### During Training

We track two metrics every epoch:

- **Loss:** Lower is better. Measures prediction error.
- **Accuracy:** Higher is better. What fraction of predictions were correct.

We monitor these separately on training and validation sets. If training accuracy is high but validation accuracy is low, the model is **overfitting** (memorising, not learning).

### Final Evaluation on the Test Set

After training, we evaluate the saved best model on the **test set** using:

#### Accuracy
```
Accuracy = (correct predictions) / (total predictions)
```
A useful first measure but can be misleading with imbalanced classes.

#### Confusion Matrix

A grid showing, for each actual class, how the model classified it:

```
                    Predicted →
                    Early  Late  Mold  Powdery  Spider  Healthy
Actual ↓ Early      [200]   3     2      1        0       1
         Late         2   [195]   4      0        1       0
         Mold         1     2   [180]    5        0       2
         Powdery       0     1     3    [185]      2       1
         Spider        0     2     0      1      [190]     0
         Healthy       1     0     2      3        0     [210]
```

Diagonal values (in brackets) = correct predictions. Off-diagonal values = mistakes. This reveals *which* diseases the model confuses with each other.

#### Precision, Recall, F1 Score

For each class:

| Metric | Meaning | Formula |
|--------|---------|---------|
| **Precision** | "Of all the leaf I *said* had Early Blight, how many actually did?" | TP / (TP + FP) |
| **Recall** | "Of all the leaves that *actually had* Early Blight, how many did I correctly find?" | TP / (TP + FN) |
| **F1 Score** | Harmonic mean of precision and recall — balances both | 2 × (P × R) / (P + R) |

> **TP** = True Positive, **FP** = False Positive, **FN** = False Negative

In disease detection, **Recall** is especially critical — a missed disease (False Negative) can be far more costly than a false alarm (False Positive).

#### Top-2 and Top-3 Accuracy

How often the correct class appears in the model's **top 2 or top 3** predictions. Useful when the distinctions between classes are subtle.

---

## 12. Deploying the Model — Making it Useful

Training produces an **artifact bundle** — a set of files that represent the fully trained model, ready to be loaded and used:

```
src/agritwin_gh/models/artifacts/diseease_<timestamp>/
├── best_model.keras          ← Full trained model (weights + architecture)
├── label_encoder.json        ← Maps class index (0–5) to disease name
├── class_weights.json        ← Class weights used during training
├── training_history.csv      ← Loss and accuracy per epoch
└── evaluation_report.json    ← Full test set metrics (accuracy, F1, etc.)
```

### Making a Prediction

The inference pipeline (`src/agritwin_gh/models/disease_inference.py`) provides a simple function:

```python
from agritwin_gh.models.disease_inference import load_inference_assets, predict_image

# Load once at startup
assets = load_inference_assets("src/agritwin_gh/models/artifacts/disease_20260225_123456")

# Predict from an image file
result = predict_image(assets, image_source="path/to/leaf_photo.jpg")

print(result)
# {
#   "predicted_label"   : "tomato_early_blight",
#   "confidence"        : 0.783,          # 78.3%
#   "top_3_predictions" : [
#       ("tomato_early_blight", 0.783),
#       ("tomato_late_blight",  0.091),
#       ("tomato_leaf_mold",    0.052),
#   ]
# }
```

The function also accepts **raw image bytes** (e.g., directly from a camera stream or MinIO object storage), making it suitable for real-time integration with the AgriTwin-GH digital twin pipeline.

### Preprocessing at Inference Time

Before feeding an image to the model, it must be **preprocessed identically** to training:

1. **Read** the image (from file, URL, or bytes).
2. **Decode** to RGB (3 colour channels).
3. **Resize** to 224 × 224 pixels.
4. **Normalise** pixel values using EfficientNetB0's specific scaling function (maps [0, 255] to the range the backbone expects).

If preprocessing differs between training and inference, performance degrades significantly.

---

## 13. End-to-End Flow Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                      DATA PREPARATION                          │
│                                                                │
│  Raw leaf photos  →  Label from folder name  →  Train/Val/Test │
│   (on disk)            (e.g., Early_Blight)      80/15/10%    │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                     tf.data PIPELINE                           │
│                                                                │
│  Load image → Resize 224×224 → [Augment if training] →        │
│  Normalise (EfficientNet scale) → One-hot encode label →       │
│  Batch (32 images) → Prefetch (background loading)            │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                        MODEL ARCHITECTURE                      │
│                                                                │
│  Input (224×224×3)                                             │
│       ↓                                                        │
│  EfficientNetB0 Backbone (pre-trained on ImageNet)             │
│       ↓                                                        │
│  GlobalAveragePooling2D                                        │
│       ↓                                                        │
│  Dense(256) → BatchNorm → ReLU → Dropout(40%)                 │
│       ↓                                                        │
│  Dense(128) → BatchNorm → ReLU → Dropout(30%)                 │
│       ↓                                                        │
│  Dense(6) → Softmax  →  [Class probabilities, sum=1.0]        │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                        TRAINING                                │
│                                                                │
│  Phase 1 (Warmup, 10 epochs):  head only, lr = 0.001          │
│  Phase 2 (Finetune, 25 epochs): top 30 backbone layers,        │
│                                  lr = 0.00005                  │
│                                                                │
│  Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                      EVALUATION & EXPORT                       │
│                                                                │
│  Test set metrics: Accuracy, F1, Confusion Matrix              │
│  Save artifacts: best_model.keras, label_encoder.json          │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                      INFERENCE (DEPLOYMENT)                    │
│                                                                │
│  New leaf photo → preprocess → model.predict() →               │
│  "tomato_early_blight" (78.3% confidence)                      │
│                                                                │
│  Integrates with: AgriTwin-GH digital twin pipeline,           │
│  MinIO image store, PostgreSQL metadata DB                     │
└────────────────────────────────────────────────────────────────┘
```

---

## 14. Common Questions (FAQ)

**Q: Can this model be used for other crops?**  
A: No — it is trained exclusively on tomato leaf images. Using it on pepper, potato, or other crops will give unreliable results. Transfer learning makes it straightforward to train similar models for other crops with new data.

**Q: How accurate is the model?**  
A: Accuracy depends on the final training run. With EfficientNetB0 and the two-phase fine-tuning strategy, similar models typically achieve **93–97% test accuracy** on the PlantVillage-derived dataset. Check `evaluation_report.json` in the artifacts folder after training for the exact numbers.

**Q: What if the disease isn't in the list (e.g., Septoria Leaf Spot)?**  
A: The model will predict whichever of the 6 known classes seems most visually similar. It cannot say "unknown". This is a known limitation — future versions can add more classes with additional data, or implement an "out-of-distribution" detector.

**Q: Does it work in a real greenhouse with a camera?**  
A: It can, provided:
- The camera captures leaves in good lighting.
- Images are at least 224×224 pixels (or resized to it).
- The leaf occupies most of the frame (not tiny in the background).
- The inference pipeline is connected to the camera stream.

**Q: Why is a GPU needed?**  
A: Training is computationally intensive — millions of matrix multiplications across thousands of images and dozens of epochs. A GPU does these calculations in **parallel**, reducing training time from days to hours. Inference (prediction on a single image) is fast even on CPU — typically under 100 milliseconds.

**Q: What is an "epoch"?**  
A: One complete pass through all training images. If you have 8,000 training images and train for 35 epochs total, the model sees each image 35 times (with different random augmentations each time).

**Q: Why does the model sometimes get it wrong?**  
A: Several reasons:
- Poor image quality (blur, bad lighting)
- Multiple diseases present simultaneously on one leaf
- Early-stage diseases that look very similar to each other
- Leaves photographed at extreme angles
- Symptoms outside the training distribution (new disease strain, unusual progression)

No model is perfect — the goal is to be accurate enough to be useful, not to replace expert agronomists.

---

## 15. Glossary

| Term | Plain-English Definition |
|------|--------------------------|
| **Accuracy** | Fraction of predictions that were correct |
| **Augmentation** | Creating modified copies of training images (rotated, flipped, etc.) to improve robustness |
| **Backbone** | The large pre-trained network used as a feature extractor |
| **Batch** | A small group of images processed together (32 in our case) |
| **Batch Normalisation** | A technique that stabilises training by normalising intermediate outputs |
| **Callback** | An automatic action taken during training (e.g., save best model) |
| **Class** | A category the model predicts (e.g., "tomato_early_blight") |
| **Class weights** | Multipliers that make the model pay more attention to rare classes |
| **CNN** | Convolutional Neural Network — a type of neural network designed for images |
| **Confidence** | The model's certainty about a prediction, expressed as a percentage |
| **Confusion matrix** | A table showing which classes the model confuses with each other |
| **Deep learning** | Machine learning using neural networks with many layers |
| **Dropout** | Randomly disabling neurons during training to prevent overfitting |
| **Early stopping** | Automatically stopping training when no improvement is seen |
| **EfficientNetB0** | A lightweight, accurate CNN architecture designed by Google |
| **Epoch** | One complete pass through all training data |
| **F1 Score** | A balanced measure combining precision and recall |
| **Feature** | A pattern or characteristic detected by the model (e.g., edges, textures) |
| **Fine-tuning** | Allowing pre-trained model layers to update slightly on new task data |
| **Fungicide** | A chemical that kills or prevents fungal diseases |
| **GPU** | Graphics Processing Unit — hardware that trains neural networks quickly |
| **Inference** | Using a trained model to make predictions on new data |
| **Label** | The correct answer/class for a training image |
| **Label smoothing** | Softening training targets so the model doesn't become overconfident |
| **Learning rate** | How large a step the model takes when adjusting weights during training |
| **Loss** | A number measuring how wrong the model's predictions are; minimised during training |
| **Mixed precision** | Using 16-bit numbers for speed while keeping 32-bit where precision matters |
| **Neural network** | A system of interconnected mathematical functions loosely inspired by the brain |
| **One-hot encoding** | Representing a class as a list of zeros with a single 1 (e.g., class 2 of 6 = [0,0,1,0,0,0]) |
| **Overfitting** | When a model memorises training data but fails on new data |
| **Precision** | Of predicted positives, fraction that are truly positive |
| **Pre-processing** | Preparing raw images for input to the model (resize, normalise) |
| **Recall** | Of actual positives, fraction the model correctly identified |
| **RGB** | Red, Green, Blue — three numbers per pixel representing colour |
| **Softmax** | A function converting raw scores to probabilities summing to 1.0 |
| **Test set** | Images held out until after training; used for final performance measurement |
| **Transfer learning** | Reusing a model trained on one task as a starting point for another |
| **Training** | The process of adjusting model weights to minimise prediction error |
| **Validation set** | Images used during training to check generalisation, not used for weight updates |
| **Weight** | A number inside a neural network that determines how important an input is |

---

*Document maintained as part of the AgriTwin-GH project. For technical implementation details, see the training notebook at `notebooks/tomato_disease_classifier_train.ipynb`.*
