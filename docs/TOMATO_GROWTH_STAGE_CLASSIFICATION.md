# Tomato Growth Stage Classification

> **Who is this for?**  
> This document is written for anyone — farmer, student, developer, or curious reader — with zero prior knowledge of machine learning or plant biology. Every concept is explained from the ground up, with analogies and plain language throughout.

---

## Table of Contents

1. [Why Does This Matter?](#1-why-does-this-matter)
2. [The Six Growth Stages](#2-the-six-growth-stages)
   - [Stage 1 — Seedling](#21-stage-1--seedling)
   - [Stage 2 — Early Vegetative](#22-stage-2--early-vegetative)
   - [Stage 3 — Flowering Initiation](#23-stage-3--flowering-initiation)
   - [Stage 4 — Flowering](#24-stage-4--flowering)
   - [Stage 5 — Unripe Fruit](#25-stage-5--unripe-fruit)
   - [Stage 6 — Ripe Fruit](#26-stage-6--ripe-fruit)
3. [What is Image Classification?](#3-what-is-image-classification)
4. [How Does a Computer "See" a Plant?](#4-how-does-a-computer-see-a-plant)
5. [What is Deep Learning? (No maths required)](#5-what-is-deep-learning-no-maths-required)
6. [Transfer Learning — Standing on Giants' Shoulders](#6-transfer-learning--standing-on-giants-shoulders)
7. [Our Model: EfficientNetB3](#7-our-model-efficientnetb3)
8. [The Dataset](#8-the-dataset)
9. [Data Augmentation — Teaching with Variations (Growth-Stage Safe)](#9-data-augmentation--teaching-with-variations-growth-stage-safe)
10. [Training the Model — Phase by Phase](#10-training-the-model--phase-by-phase)
11. [Progressive Resizing — A Resolution Boost Trick](#11-progressive-resizing--a-resolution-boost-trick)
12. [Test-Time Augmentation (TTA) — More Confident at Inference](#12-test-time-augmentation-tta--more-confident-at-inference)
13. [How We Measure Success](#13-how-we-measure-success)
14. [Deploying the Model — Making it Useful](#14-deploying-the-model--making-it-useful)
15. [End-to-End Flow Diagram](#15-end-to-end-flow-diagram)
16. [Common Questions (FAQ)](#16-common-questions-faq)
17. [Glossary](#17-glossary)

---

## 1. Why Does This Matter?

Tomatoes are not a single product — they pass through a sequence of distinct biological phases from the moment a seed germinates to the moment a ripe fruit is harvested. Every stage has different needs:

- **Water requirements** differ — seedlings need gentle irrigation, flowering plants need precise amounts.
- **Fertilisation schedules** differ — nitrogen for early growth, potassium and phosphorus at fruiting.
- **Pest and disease risk** differs — certain diseases strike at specific stages, and early detection requires knowing where in the cycle the plant is.
- **Harvest timing** is determined entirely by stage — picking too early or too late degrades quality and revenue.

**The problem:** In a large greenhouse with hundreds or thousands of plants, manually assessing the growth stage of every plant daily is:

1. Labour-intensive and slow.
2. Subjective — different workers may classify the same plant differently.
3. Error-prone — misidentification leads to wrong interventions at the wrong time.

**The solution this project provides:** A camera captures a photo of the tomato plant. An AI model analyses the photo within milliseconds and identifies the exact growth stage — *Seedling*, *Early Vegetative*, *Flowering Initiation*, *Flowering*, *Unripe*, or *Ripe*. This enables:

- **Automated growth tracking** across all plants simultaneously.
- **Precise, stage-aware control** of irrigation, lighting, and fertilisation.
- **Harvest prediction** based on the distribution of stages across the crop.
- **Integration with the AgriTwin-GH digital twin** — growth stage feeds the disease risk model, control policy, and what-if simulations.

---

## 2. The Six Growth Stages

Our model classifies tomato plants into **6 sequential growth stages**. The stages are ordered — a plant always passes through them in this exact sequence.

### 2.1 Stage 1 — Seedling

| Property | Detail |
|----------|--------|
| **Duration** | Days 1–14 after germination |
| **Key visual features** | 2–4 small, oval seed leaves (cotyledons); short, fragile stem |
| **Colour** | Pale, light green |
| **Size** | Very small — typically 2–8 cm tall |

**What it looks like:**  
The seedling stage begins the moment a seed germinates and pushes through the soil surface. The plant first shows its two rounded *cotyledons* (seed leaves), which are simple and oval — quite unlike the distinctive jagged tomato leaf that appears later. The stem is very thin and bright green. The plant is extremely fragile at this stage.

**Why it matters for management:**  
Seedlings need **high humidity** (95%+), **gentle lighting** (avoid intense direct light), and **low, careful irrigation**. Overwatering at this stage is one of the most common causes of seedling death (*damping off*). Temperature control is critical — below 15 °C slows development significantly.

**Decision trigger:** Begin counting days to transplant. Seedlings are typically moved to larger containers or growing beds after 14–21 days.

---

### 2.2 Stage 2 — Early Vegetative

| Property | Detail |
|----------|--------|
| **Duration** | Days 14–35 after germination |
| **Key visual features** | First true leaves visible; compound leaf structure developing |
| **Colour** | Medium green; leaf surface begins to show texture |
| **Size** | 10–30 cm tall; multiple leaf nodes appearing |

**What it looks like:**  
The true tomato leaves begin to appear above the cotyledons. These compound leaves have a distinctive pinnate structure — a central stem with multiple leaflets on each side. The plant is actively building its root and stem infrastructure at this stage. Leaves are medium green and begin showing the characteristic slightly rough, hairy texture tomato leaves are known for.

**Why it matters for management:**  
This is the primary **vegetative growth phase** — the plant is investing all energy into building structure. Management decisions:
- **Nitrogen-rich fertilisation** to support leaf and stem development.
- **Training and staking** — the main stem should be guided upright.
- **Light levels** should increase (16–18 hours per day for rapid growth).
- **Humidity** can be reduced slightly (80–85%) as the plant is more robust.

**Decision trigger:** Begin applying full nutrient solution. Prune suckers (side shoots) to direct energy to the main stem.

---

### 2.3 Stage 3 — Flowering Initiation

| Property | Detail |
|----------|--------|
| **Duration** | Days 35–50 after germination |
| **Key visual features** | First flower buds (trusses) visible; plant reaches 40–60 cm |
| **Colour** | Dark, rich green leaves; small yellow bud clusters appearing |
| **Size** | 40–70 cm tall |

**What it looks like:**  
The first flower clusters (*trusses*) become visible — small yellow-green bud formations emerging at leaf axils. The leaves are now fully mature in form and a deep, rich green. The plant's height accelerates and the stem thickens. This is a critical transition from pure vegetative growth to the reproductive phase.

**Why it matters for management:**  
Flowering initiation requires a **precise environmental shift**:
- **Temperature** must be in the 18–26 °C range — too cold or too hot prevents flowers from setting later.
- **Fertilisation** shifts from nitrogen emphasis to a **balanced NPK** formula.
- **Air circulation** increases — poor ventilation at this stage leads to pollination failure.
- **Humidity** drops to 65–75% — high humidity at flowering causes pollen to clump, preventing fertilisation.

**Decision trigger:** Trigger the system to begin vibrating the trusses or using bumblebees for pollination in the next stage. Begin monitoring CO₂ levels closely.

---

### 2.4 Stage 4 — Flowering

| Property | Detail |
|----------|--------|
| **Duration** | Days 50–65 after germination |
| **Key visual features** | Open yellow flowers; multiple trusses at different stages |
| **Colour** | Vivid yellow flowers against deep green foliage |
| **Size** | 70–120 cm; first trusses at floor level, newer ones higher |

**What it looks like:**  
Bright yellow, star-shaped flowers are now fully open. Each flower has reflexed (backwards-pointing) petals and a prominent central cone of stamens. Multiple trusses may be visible, each at a slightly different developmental stage. The plant is tall, lush, and architecturally complex.

**Why it matters for management:**  
This is the **most sensitive stage for yield determination**:
- **Pollination** must occur — bumblebees (common in commercial greenhouses) or manual/electric truss vibrators ensure pollen transfer.
- **Temperature** is critical — above 30 °C or below 10 °C at night causes flower drop (*blossom drop*), destroying potential fruit.
- **Calcium supply** must be adequate — calcium deficiency at this stage causes *blossom end rot* later.
- **Irrigation precision** peaks — stress during flowering reduces fruit set.

**Decision trigger:** Trigger automated truss vibration system. Alert if temperature exceeds 30 °C or drops below 12 °C.

---

### 2.5 Stage 5 — Unripe Fruit

| Property | Detail |
|----------|--------|
| **Duration** | Days 65–90 after germination |
| **Key visual features** | Small to medium green fruits; flowers may still be present on upper trusses |
| **Colour** | Solid mid-to-dark green fruits (often with a whitish-green shoulder) |
| **Size** | Fruit diameter 1–6 cm depending on variety |

**What it looks like:**  
Green tomato fruits are now clearly visible, having developed from the fertilised flowers. The fruits are firm and green — they contain high levels of *chlorophyll* giving them colour, and *solanine* making them mildly toxic when raw. Upper trusses may still show open flowers while lower trusses bear larger fruits, creating a striking visual mix of yellow and green.

**Why it matters for management:**  
The fruit is actively accumulating sugars, acids, and cell mass. Management priorities:
- **Potassium** supplementation is critical — potassium drives sugar accumulation and fruit quality.
- **Consistent irrigation** — irregular watering causes *blossom end rot* and fruit cracking.
- **Leaf pruning** — removing lower yellowing leaves improves airflow and reduces disease risk.
- **Truss loading management** — removing excess fruits per truss ensures remaining fruits reach full size.

**Decision trigger:** Estimate harvest date based on fruit size and growth rate. Begin alerting harvesters of predicted readiness window.

---

### 2.6 Stage 6 — Ripe Fruit

| Property | Detail |
|----------|--------|
| **Duration** | Days 90–110+ after germination |
| **Key visual features** | Red (or yellow/orange depending on variety), fully coloured, slightly soft fruits |
| **Colour** | Vivid red, fully uniform colouration; green colour completely replaced |
| **Size** | Full size for variety; typically 5–10 cm diameter |

**What it looks like:**  
The tomato has completed the *ethylene-triggered ripening cascade* — chlorophyll breaks down, lycopene (the red pigment) is synthesised, sugar content peaks, and the fruit softens slightly. A fully ripe tomato is uniformly coloured, carries no green patches, and has a slight give when pressed gently. The stem end may retain a small green calyx.

**Why it matters for management:**  
**Harvest timing is everything for quality:**
- Fruit left on the vine too long becomes overripe, splitting, and susceptible to *Botrytis* rot.
- Harvested too early, tomatoes never develop full flavour even if they redden post-harvest.
- Ripe fruits are detected automatically, and a **harvest alert** is generated for that plant row.

**Decision trigger:** Immediate harvest scheduling. Log harvest event in the digital twin. Update yield tracking metrics.

---

## 3. What is Image Classification?

Imagine you show 1,000 photos of plants at different ages to a child. After a while, the child learns to associate certain visual patterns — tiny cotyledons, open yellow flowers, red round fruits — with specific stage names. **Image classification** is teaching a computer to do the same thing.

In our case:
- We show the computer thousands of photos of tomato plants.
- Each photo is labelled with the correct stage name (e.g., "Stage4_Flowering").
- The computer learns patterns — colours, shapes, textures, spatial arrangements — that distinguish one stage from another.
- After training, when shown a **new, unseen photo**, the computer predicts which stage it belongs to.

The output is a set of **confidence percentages**, one per stage:

```
Stage1_Seedling              :  0.8%
Stage2_Early_Vegetative      :  1.2%
Stage3_Flowering_Initiation  :  3.1%
Stage4_Flowering             :  4.7%
Stage5_Unripe                :  9.1%
Stage6_Ripe                  : 81.1%
```

The model picks the stage with the highest confidence — in this case, *Ripe Fruit*.

---

## 4. How Does a Computer "See" a Plant?

A digital image is a **grid of pixels**. Each pixel has three numbers representing its colour — **Red, Green, Blue (RGB)** — each ranging from 0 to 255.

A 300×300 pixel image (our input size) is therefore a grid of:
```
300 rows × 300 columns × 3 colour channels = 270,000 numbers
```

The computer's job is to find meaning in these ~270,000 numbers. But growth stages are particularly challenging because:

- **Colour is a genuine cue** — a green fruit is unripe, a red fruit is ripe. The model must *preserve* colour information.
- **Morphology changes dramatically** — a seedling looks nothing like a flowering plant. The model must detect shape and structure.
- **Scale varies** — the same stage can look very different at different camera distances.
- **Multiple trusses** may be visible simultaneously at different stages.

We need an architecture that captures both fine colour gradients *and* large-scale structural features. This is why our model uses **higher resolution** (300×300 vs 224×224 for the disease model) and a **more powerful backbone**.

---

## 5. What is Deep Learning? (No maths required)

### The Analogy: A Layered Detective

Think of a detective who identifies a growth stage by asking a series of questions:

1. **Layer 1 (Basic shapes):** "Are there any round objects? Is there a stem?"
2. **Layer 2 (Colours and textures):** "Is the dominant colour green or red? Are leaves rough or smooth?"
3. **Layer 3 (Patterns):** "Are there small oval leaves (cotyledons)? Star-shaped yellow flowers? Clustered round fruits?"
4. **Layer 4 (Stage features):** "Yellow star-shaped flowers + deep green foliage = Flowering stage"
5. **Final Layer (Decision):** "Confidence: 94.7% Stage 4 — Flowering"

A **deep neural network** does exactly this — it stacks many **layers** of pattern detectors, each layer learning increasingly complex features from the output of the previous layer.

### What is a Neuron?

A **neuron** is a tiny mathematical function. It takes numbers in, multiplies each by a **weight** (importance), adds them up, and outputs a new number. The "learning" is simply adjusting these weights based on mistakes.

### How Learning Happens

1. **Forward pass:** Show the network a plant photo → it predicts a stage.
2. **Measure the mistake:** Compare the prediction to the correct label. Calculate an error score (**loss**).
3. **Backward pass (backpropagation):** Adjust every weight slightly in the direction that reduces the error.
4. **Repeat** millions of times across thousands of images.

After enough repetitions, the weights settle into values that make good predictions. This process is called **training**.

### What is a Convolutional Neural Network (CNN)?

A CNN is a type of deep neural network specifically designed for images. Instead of connecting every pixel to every neuron (which would require billions of parameters), it uses **filters** (small sliding windows) that scan across the image detecting local features — exactly like how your eye might track a cluster of yellow flowers across a frame without needing to look at the entire image at once.

---

## 6. Transfer Learning — Standing on Giants' Shoulders

Training a powerful CNN from scratch requires:
- **Millions of images**
- **Weeks of compute time** on powerful GPU machines
- **Expert tuning**

We have neither the data volume nor the compute time for that. Instead, we use **Transfer Learning**.

### The Idea

Large technology companies (Google, Facebook, etc.) train massive models on millions of general images (dogs, cars, buildings, flowers). These models learn extremely powerful general features — edges, textures, patterns, shapes — that transfer well to almost any visual task.

We take one of these pre-trained models and **adapt it** to our specific task (tomato growth stage classification) by training only a small custom "head" on top, using our stage images.

**Analogy:** Instead of teaching someone to identify fruits from scratch, you hire a botanist who already understands plant biology deeply and just teach them the specific visual cues that distinguish a seedling from a flowering plant in *tomatoes specifically*. Far faster and far more effective.

### Why Transfer Learning Works So Well Here

For growth stage classification, the general visual features learned from ImageNet — colour gradients, texture patterns, circular shapes, edge contours — map almost directly to what we need. A network that already recognises green leaves, yellow petals, and round red fruits in its general training has a massive head start on our task.

---

## 7. Our Model: EfficientNetB3

### What is EfficientNetB3?

**EfficientNet** is a family of neural network architectures developed by Google Brain in 2019. The "B3" variant is a mid-size model — significantly more powerful and accurate than the smallest B0 variant, while remaining practical for deployment.

EfficientNet was designed using **Neural Architecture Search (NAS)**: an AI was used to discover the optimal architecture rather than humans hand-tuning it. The key insight is **compound scaling** — instead of making networks just deeper or wider, EfficientNet scales depth, width, and input resolution simultaneously in a mathematically balanced way.

### Why EfficientNetB3 (and not B0)?

The growth stage task is **harder** than disease classification in one specific way: **colour and morphology both matter simultaneously**. A model must distinguish:
- A deep-green unripe fruit (Stage 5) from a red ripe fruit (Stage 6) — primarily a colour distinction.
- A seedling plant (Stage 1) from a flowering plant (Stage 4) — primarily a structural distinction.

EfficientNetB3 provides:

| Reason | Explanation |
|--------|-------------|
| **Higher capacity** | 12M parameters vs 5.3M for B0 — better at capturing subtle colour and structural differences |
| **Native 300×300 resolution** | Larger input captures fine detail — critical for distinguishing early bud clusters from full flowers |
| **Pre-training on ImageNet** | Already trained on 1.28M images across 1,000 classes |
| **Proven accuracy** | Consistently outperforms B0 on fine-grained visual classification tasks |
| **Memory efficient** | Scaled thoughtfully — more powerful than B0 without requiring enterprise-grade GPU memory |

### Our Custom Head

EfficientNetB3 acts as the **feature extractor** (backbone). On top of it, we add a small **custom head** that makes the final stage prediction:

```
EfficientNetB3 Backbone (feature extraction, pre-trained on ImageNet)
        ↓
  GlobalAveragePooling2D
  (collapses spatial feature maps into a single averaged vector)
        ↓
  Dropout(40%)    ← randomly switches off 40% of neurons during training
        ↓          to prevent overfitting
  Dense(256 neurons) + ReLU activation
        ↓
  BatchNormalisation   ← stabilises training by normalising intermediate outputs
        ↓
  Dropout(30%)
        ↓
  Dense(6 neurons) + Softmax    ← float32 explicit for numerical stability
  (one neuron per growth stage; outputs probabilities summing to 1.0)
```

**Dropout** is a regularisation technique — during training, we randomly "switch off" a fraction of neurons. This forces the network not to rely too heavily on any single neuron, making it more robust and **reducing overfitting** (memorising training data instead of learning general patterns).

**BatchNormalisation** normalises the outputs of a layer during training, stabilising and accelerating learning. It also acts as a mild regulariser.

---

## 8. The Dataset

### Structure on Disk

```
data/
└── external/
    └── Tomato Growth Stages/
        ├── Stage1_Seedling/             ← images of seedling plants
        ├── Stage2_Early_Vegetative/     ← images of young vegetative plants
        ├── Stage3_Flowering_Initiation/ ← images of plants with first buds
        ├── Stage4_Flowering/            ← images of plants in full flower
        ├── Stage5_Unripe/               ← images of plants bearing green fruit
        └── Stage6_Ripe/                 ← images of plants bearing ripe red fruit
```

The **exact folder name** is used directly as the class label — there is no remapping. This means `label_map.json` is fully human-readable without any translation table.

### Class Labels

| Folder Name | Index | What It Represents |
|-------------|-------|--------------------|
| `Stage1_Seedling` | 0 | Germinated plant with cotyledons only |
| `Stage2_Early_Vegetative` | 1 | True leaves forming; active stem growth |
| `Stage3_Flowering_Initiation` | 2 | First flower bud clusters visible |
| `Stage4_Flowering` | 3 | Open yellow flowers; pollination phase |
| `Stage5_Unripe` | 4 | Developed green fruits on trusses |
| `Stage6_Ripe` | 5 | Fully red, harvest-ready fruits |

### Train / Validation / Test Split

The dataset is divided into three **non-overlapping, stratified** subsets:

| Subset | Purpose | Fraction |
|--------|---------|----------|
| **Training set (75%)** | Images the model learns from | 75% of each class |
| **Validation set (15%)** | Used during training to check generalisation (model never trains on these) | 15% of each class |
| **Test set (10%)** | Held out completely until final evaluation — the true measure of performance | 10% of each class |

**Stratified** means the class proportions are preserved in each split. If Stage 4 (Flowering) makes up 18% of the dataset, it will also make up approximately 18% of the training, validation, and test sets. This prevents accidental bias in any split.

**Why three splits?**  
Using the same images for both training and measuring accuracy would be like a student memorising an exam answer sheet — the score would look great but mean nothing. The test set is the student's actual exam with questions they've never seen.

### Class Imbalance and Handling

In practice, some growth stages are photographed more often than others. Stage 4 (Flowering) might be heavily photographed because it is visually striking, while Stage 1 (Seedling) might have fewer images because they are small and uninteresting to photographers.

If the model sees 5× more Flowering images than Seedling images, it will learn to be lazy and always guess Flowering — achieving high accuracy overall but poor performance on underrepresented stages.

We address this with **class weights** computed via scikit-learn's `compute_class_weight("balanced", ...)`:

```
Class weight = (total samples) / (number of classes × samples in this class)
```

A class with fewer images gets a **higher weight** → the model must pay more attention to mistakes on that class. These weights are passed directly to Keras's `model.fit()` via the `class_weight` argument.

---

## 9. Data Augmentation — Teaching with Variations (Growth-Stage Safe)

### The Problem

Our dataset has a limited number of images per stage. Deep learning models typically need far more. Also, real-world greenhouse cameras will capture plants from slightly different angles, at different lighting levels, and with different image quality.

### The Solution

**Data Augmentation** artificially expands the dataset by creating modified versions of existing images during training. The key insight: a slightly rotated photo of a flowering tomato plant is *still* a flowering tomato plant.

### Why Growth Stage Augmentation Must Be Milder

This is the **most important difference** between this model and the disease classifier:

**For disease classification**, augmentation can be aggressive — heavy hue shifts, colour jitter, vertical flips — because a disease lesion's identity doesn't change with extreme colour shifts. The colour of a spot is less critical than its texture and shape.

**For growth stage classification**, colour is a *primary discriminative cue*:
- **Stage 5 (Unripe)** is green; **Stage 6 (Ripe)** is red. Arbitrary hue shifts could make a ripe fruit look unripe.
- **Stage 1 (Seedling)** is pale green; **Stage 2 (Vegetative)** is richer green. Saturation changes could blur this distinction.
- Plant orientation matters — vertical flip might suggest the plant is growing downwards, which is biologically incorrect for most growth stages.

The augmentation is therefore deliberately conservative:

| Augmentation | Setting | Why It's Controlled |
|---|---|---|
| **Horizontal flip** | ✓ Enabled | Left-right symmetry is safe — tomatoes grow symmetrically |
| **Vertical flip** | ✗ Disabled | Plants grow upward — flipping could mislead the model about plant orientation |
| **Random rotation** | ±8% (≈ ±29°) | Much gentler than disease model (±54°) — plant structure is a stage cue |
| **Random zoom** | ±10% | Mild — helps with camera distance variation |
| **Random brightness** | ±12% | Moderate — greenhouse lighting varies |
| **Random contrast** | ±10% | Near-symmetric range — preserves overall image tone |
| **Random hue** | ±3% | **Very subtle** — preserves the green→yellow→red gradient critical for stage discrimination |
| **Random saturation** | 0.85 – 1.15 range | Near-neutral — avoids washing out or over-saturating colour cues |
| **Random crop** | Retains ≥ 92% of image | Focuses on different regions while preserving the full plant structure |
| **Cutout (Random Erasing)** | 12% patch, 50% probability | Smaller and less frequent than disease model — preserves stage-identifying regions |

> **Important:** Augmentation is applied **only to the training set**. Validation and test sets use the original images — because we want to measure performance on realistic, unmodified inputs.

### Technical Implementation

Each augmentation is implemented as a `tf.data` map function operating on individual images in the pipeline, applied on-the-fly during training. This means every time the model sees an image, it sees a *different* augmented version — effectively multiplying the dataset size.

```python
# Example: hue shift is applied after converting to [0,1] scale
image = tf.image.random_hue(image / 255.0, max_delta=0.03) * 255.0
# max_delta=0.03 means a maximum ±3% shift in the HSV hue channel
```

---

## 10. Training the Model — Phase by Phase

Training happens in **two phases** — a well-established best practice called **progressive fine-tuning**.

### Phase 1: Warm-Up (8 epochs, frozen backbone)

**Epoch** = one complete pass through the entire training dataset.

In Phase 1, the EfficientNetB3 backbone weights are **frozen** — they cannot change. Only our custom head layers (the Dense, Dropout, and BatchNorm layers we added) are updated.

**Why?**  
The pre-trained backbone already knows how to detect visual features from ImageNet. If we immediately allow all layers to update with our small growth stage dataset, the powerful backbone weights get "corrupted" before the head has learned anything useful — a phenomenon called **catastrophic forgetting**. Warming up the head first gives it a sensible starting point before we allow the backbone to fine-tune.

With **progressive resizing** enabled (see Section 11), Phase 1 trains at a smaller 224×224 resolution for speed, then switches to 300×300 for Phase 2.

| Setting | Value |
|---------|-------|
| **Epochs** | 8 |
| **Learning rate** | `0.001` (head is learning from scratch — higher rate appropriate) |
| **What updates** | Custom head only (Dropout, Dense, BatchNorm layers) |
| **Training dataset** | 224×224 progressive-resizing dataset (if enabled) |

### Phase 2: Fine-Tuning (15 epochs, top backbone layers unfrozen)

After the head is trained, we **unfreeze the top 40 layers** of the EfficientNetB3 backbone and allow them to fine-tune on our data at the full 300×300 resolution.

**Why only the top layers?**  
The bottom layers of a CNN learn very basic features (edges, corners, colour blobs) that are universal — useful for every image task. These don't need to change. The top layers learn high-level, task-specific features — exactly the kind of stage-specific patterns we need.

**Why 40 layers (vs 30 in the disease model)?**  
EfficientNetB3 has more layers than B0, and growth stage classification benefits from deeper fine-tuning because the features required (simultaneous colour and morphology discrimination) are more complex than disease lesion recognition.

**BatchNorm layers are kept frozen** even in Phase 2. This is critical for stability — allowing BatchNorm statistics to update with a small dataset at low learning rates causes training instability.

| Setting | Value |
|---------|-------|
| **Epochs** | 15 |
| **Learning rate** | `0.00003` (3×10⁻⁵, lower than disease model for B3 stability) |
| **What updates** | Top 40 backbone layers + entire custom head |
| **Training dataset** | Full 300×300 dataset |
| **BatchNorm** | Frozen throughout (prevents instability) |

### Why a Lower Fine-Tuning Learning Rate Than the Disease Model?

EfficientNetB3 is more powerful than B0, which means its pre-trained weights are both more valuable *and* more sensitive to large updates. A higher learning rate risks overshooting the optimal fine-tuned weights. `3×10⁻⁵` (vs `5×10⁻⁵` for B0) ensures the small, careful adjustments that improve stage-specific features without destroying the backbone's general knowledge.

### Loss Function

The **loss function** measures how wrong the model's predictions are. During training, the optimiser tries to minimise this number.

We support two loss functions, selectable via `CONFIG['loss_type']`:

#### Option A: Categorical Cross-Entropy with Label Smoothing (default, `'ce'`)

The standard loss for multi-class classification. With **label smoothing** (0.1), the model is trained to output 90% confidence for the correct class and distributes the remaining 10% across other classes. This prevents **overconfidence** and improves generalisation.

```
Without smoothing: target = [0, 0, 0, 0, 0, 1]   ← 100% certain
With smoothing:    target = [0.017, 0.017, 0.017, 0.017, 0.017, 0.917]
```

#### Option B: Multiclass Focal Loss (`'focal'`)

An alternative loss function specifically designed for **class imbalance**. It down-weights easy, confidently-classified examples and focuses training on hard, misclassified ones.

```
Focal Loss = α × (1 - p_correct)^γ × Cross-Entropy
```

Where:
- `α = 0.25` — balances the relative loss contribution of each class.
- `γ = 2.0` — controls the "focusing" effect. Higher γ = more focus on hard examples.

This is particularly useful if some growth stages are severely underrepresented even after class weights are applied.

### Callbacks (Automatic Training Assistants)

| Callback | What it Does |
|----------|-------------|
| **ModelCheckpoint** | Saves the model whenever validation accuracy improves. The best version is always kept. Saved as `<run_id>_best.keras`. |
| **ReduceLROnPlateau** | If validation loss stops improving for **3 consecutive epochs**, the learning rate is multiplied by 0.4 (i.e., reduced by 60%). Minimum floor: `1×10⁻⁷`. Helps escape training plateaus. |
| **EarlyStopping** | If the model hasn't improved for **6 consecutive epochs**, stop training early. Restores best weights automatically. Prevents overfitting and saves compute time. |
| **CSVLogger** | Logs loss and accuracy for every epoch to a single CSV file (`training_history.csv`), appending Phase 2 after Phase 1 in the same file. |

### Mixed Precision Training

Modern GPUs process **16-bit floating point numbers** (FP16) much faster than traditional **32-bit** (FP32), using less memory. **Mixed precision** training (`keras.mixed_precision.set_global_policy("mixed_float16")`) uses FP16 for most operations but keeps FP32 where numerical precision matters (the final softmax output layer is explicitly cast to float32).

This typically provides **1.5–2× speedup** on compatible GPUs (NVIDIA Volta / Turing generation or newer) with no accuracy loss.

> **Note:** Mixed precision is automatically disabled on CPU. Set `CONFIG['mixed_precision'] = False` if you observe NaN losses during training.

---

## 11. Progressive Resizing — A Resolution Boost Trick

### The Problem

EfficientNetB3 is designed for 300×300 images. Training at 300×300 from the first epoch is slower because each batch contains larger images. Can we train efficiently *and* achieve the accuracy benefit of high resolution?

### The Solution: Progressive Resizing

**Progressive resizing** (enabled by `CONFIG['progressive_resizing'] = True`) runs training in two resolution stages:

| Phase | Resolution | Why |
|-------|-----------|-----|
| **Phase 1 (Warm-up)** | 224×224 | Faster per-batch — more epochs per hour. The head is learning from scratch; high resolution isn't needed yet. |
| **Phase 2 (Fine-tuning)** | 300×300 | Full native resolution — the backbone is now fine-tuning on our data, and high resolution provides the detail needed for subtle stage distinctions. |

**Analogy:** Think of studying for an exam by first reading a summary (224px — fast, gets the gist), then re-reading the detailed notes (300px — slower, captures the nuances).

### Implementation Details

Two separate `tf.data.Dataset` objects are constructed:
- `train_ds_prog`: loads and resizes images to 224×224, applies growth-stage-safe augmentation.
- `train_ds`: loads and resizes images to 300×300 (full resolution), same augmentation.

Phase 1 trains on `train_ds_prog`; Phase 2 switches to `train_ds`. The validation and test datasets always use the full 300×300 resolution, regardless of the setting.

> If `prog_image_size == image_size`, the system automatically falls back to using a single dataset.

---

## 12. Test-Time Augmentation (TTA) — More Confident at Inference

### The Problem

A trained model's prediction on a single view of an image can be sensitive to small variations — slight rotation, minor crop, camera angle. A single forward pass may give a confident but slightly unlucky prediction.

### The Solution: Averaging Multiple Augmented Views

**Test-Time Augmentation** (enabled by `CONFIG['tta'] = True`) applies random augmentations to the *same* image multiple times at inference, runs the model on each augmented copy, and **averages** the resulting softmax probability distributions.

```
Original image → augmented view 1 → [0.05, 0.03, 0.08, 0.10, 0.12, 0.62]
               → augmented view 2 → [0.04, 0.05, 0.06, 0.09, 0.15, 0.61]
               → augmented view 3 → [0.06, 0.02, 0.07, 0.11, 0.11, 0.63]
               → augmented view 4 → [0.03, 0.04, 0.08, 0.12, 0.13, 0.60]
               → augmented view 5 → [0.05, 0.03, 0.07, 0.10, 0.14, 0.61]

Averaged result → [0.046, 0.034, 0.072, 0.104, 0.130, 0.614]
→ Predicted class: Stage6_Ripe (61.4%)
```

**Why does averaging help?**  
Each augmented view is a different "opinion" about the image. Averaging reduces the variance of the prediction — correct-class probabilities tend to agree and reinforce; incorrect-class probabilities tend to disagree and cancel out.

**Our TTA implementation:**
- `tta_steps = 5` augmented forward passes are averaged.
- TTA applies random small rotations (±8%) — same as `RandomRotation` used in training.
- TTA is used during the **test set evaluation** (Section E in the notebook) and is available in the inference function.

**Trade-off:** TTA takes `tta_steps` times longer than a single pass (5× in our case). For real-time applications where every millisecond counts, set `tta=False`. For batch evaluation or cases where prediction confidence is critical, TTA is recommended.

---

## 13. How We Measure Success

### During Training

We track two metrics every epoch, on both the training and validation sets:

- **Loss:** Lower is better. Measures how wrong the model's predictions are.
- **Accuracy:** Higher is better. What fraction of predictions were correct.

If training accuracy is high but validation accuracy is low, the model is **overfitting** (memorising training images rather than learning general patterns). Our callbacks (EarlyStopping, Dropout, BatchNorm) all fight overfitting.

A dashed vertical line on the training history plot marks the boundary between Phase 1 (warm-up) and Phase 2 (fine-tuning) — a useful diagnostic for understanding how each phase contributed.

### Final Evaluation on the Test Set

After training, the best checkpoint (`<run_id>_best.keras`) is loaded and evaluated on the **test set** — images the model has *never* seen during training or validation.

#### Accuracy

```
Accuracy = (correct predictions) / (total predictions)
```

Simple and intuitive, but can be misleading if class sizes are unequal (a model that always guesses "Early Vegetative" could achieve 20%+ accuracy on a balanced 6-class dataset without learning anything).

#### Per-Class Accuracy

The fraction of images in each class that the model correctly classified. This reveals if the model is systematically weak on a specific stage.

```
Stage3_Flowering_Initiation accuracy = 0.8723  (87.2%)
Stage4_Flowering accuracy            = 0.9401  (94.0%)
Stage5_Unripe accuracy               = 0.9115  (91.1%)
```

#### Confusion Matrix

A grid showing, for each actual stage, how the model classified it:

```
                    Predicted →
                    S1    S2    S3    S4    S5    S6
Actual ↓ S1         [98]  2     0     0     0     0
         S2           1  [91]   5     0     0     0
         S3           0    3   [88]   6     0     0
         S4           0    0    5   [93]    2     0
         S5           0    0    0     3   [89]    4
         S6           0    0    0     0     2   [96]
```

Diagonal values (in brackets) = correct predictions. Off-diagonal = mistakes.

**What to look for:** Mistakes should cluster on *adjacent* stages — confusing Stage 3 with Stage 4 is expected (they share visual similarities during the initiation→open-flower transition). Confusing Stage 1 with Stage 6 would be a red flag indicating something is very wrong.

#### Precision, Recall, F1 Score

Computed both per-class and as macro/weighted averages:

| Metric | Meaning | Formula |
|--------|---------|---------|
| **Precision** | "Of all plants I *said* were in Stage 4, how many actually were?" | TP / (TP + FP) |
| **Recall** | "Of all the plants that *actually were* in Stage 4, how many did I correctly identify?" | TP / (TP + FN) |
| **F1 Score** | Harmonic mean of precision and recall — balances both | 2 × (P × R) / (P + R) |

> **TP** = True Positive, **FP** = False Positive, **FN** = False Negative

In growth stage classification, **Recall** matters most for critical stages:
- Missing a **ripe** fruit (Stage 6) causes harvest delay → quality loss.
- Missing a **flowering** stage (Stage 4) causes missed pollination trigger → yield loss.

#### ROC-AUC (Macro One-vs-Rest)

For each class, a **Receiver Operating Characteristic** curve plots the trade-off between True Positive Rate and False Positive Rate at different classification thresholds. The **Area Under the Curve (AUC)** summarises this in a single number:

- **AUC = 1.0:** Perfect classifier.
- **AUC = 0.5:** Random guess.
- **AUC = 0.9+:** Excellent.

We compute macro-averaged AUC across all 6 classes (One-vs-Rest strategy).

#### Misclassified Samples Grid

A visual grid showing the top 25 highest-confidence **wrong** predictions (worst mistakes first). This is more valuable than raw numbers — it shows *which* images the model finds hard and *which* stages it confuses them with. Common patterns:
- Stage 3 images misclassified as Stage 2 (bud clusters look like dense foliage).
- Stage 5 images misclassified as Stage 4 (early unripe fruits alongside remaining open flowers).

---

## 14. Deploying the Model — Making it Useful

Training produces an **artifact bundle** — a set of files representing the fully trained model, ready to be loaded and used:

```
src/agritwin_gh/models/
├── <run_id>.keras                     ← Final saved model (weights + architecture)
├── <run_id>_best.keras                ← Best validation-accuracy checkpoint
└── artifacts/<run_id>/
    ├── label_map.json                 ← Stage index (0-5) → stage name mapping
    ├── metrics.json                   ← Full test set evaluation metrics
    ├── classification_report.txt      ← Per-class precision / recall / F1
    ├── confusion_matrix.png           ← Visual confusion matrix (normalised + raw)
    ├── misclassified_grid.png         ← Grid of worst misclassified samples
    ├── roc_curves.png                 ← One-vs-Rest ROC curves per class
    ├── training_history.csv           ← Loss and accuracy per epoch (both phases)
    ├── training_history_plot.png      ← Training curve visualisation
    └── deployment_notes.txt           ← Complete integration guide
```

### Making a Prediction

The standalone inference module (`src/agritwin_gh/models/growth_stage_inference.py`) is written by the notebook automatically during training. It has **no dependency on notebook globals** — it loads everything from disk and caches the model after the first call.

```python
from agritwin_gh.models.growth_stage_inference import predict_growth_stage

# Predict from a file path — model and label_map are auto-resolved
result = predict_growth_stage("path/to/plant_photo.jpg")

print(result)
# {
#   "class_name" : "Stage4_Flowering",
#   "confidence" : 0.9213,          # 92.1%
#   "probs"      : {
#       "Stage1_Seedling"              : 0.0012,
#       "Stage2_Early_Vegetative"      : 0.0031,
#       "Stage3_Flowering_Initiation"  : 0.0421,
#       "Stage4_Flowering"             : 0.9213,
#       "Stage5_Unripe"                : 0.0314,
#       "Stage6_Ripe"                  : 0.0009,
#   },
#   "topk" : [
#       ("Stage4_Flowering",           0.9213),
#       ("Stage5_Unripe",              0.0314),
#       ("Stage3_Flowering_Initiation",0.0421),
#   ]
# }
```

The function also accepts **raw image bytes** — suitable for direct integration with camera streams, HTTP image uploads, or MinIO object storage:

```python
# From a camera stream or HTTP upload
with open("plant.jpg", "rb") as f:
    result = predict_growth_stage(f.read())

# With explicit paths (useful in containerised deployments)
result = predict_growth_stage(
    image_bytes,
    model_path="src/agritwin_gh/models/growth_stage_20260302_170744.keras",
    label_map_path="src/agritwin_gh/models/artifacts/growth_stage_20260302_170744/label_map.json",
)

# With Test-Time Augmentation (higher-confidence predictions, 5× slower)
result = predict_growth_stage(image_bytes, tta=True, tta_steps=5)
```

### Auto-Resolution of Artifact Paths

When `model_path` and `label_map_path` are not supplied, the inference module automatically:
1. Searches its own directory for the latest `growth_stage_*.keras` file (alphabetically latest = most recent run).
2. Derives the `label_map.json` path as `artifacts/<run_id>/label_map.json`.

This means in most deployments, no configuration is required beyond importing the module.

### Model Caching

After the first call to `predict_growth_stage()`, both the Keras model and the label map are cached in memory. Subsequent calls on the same process reuse the cached model — no disk I/O or model loading overhead.

This is critical for real-time applications where predictions may be requested every few seconds.

### Preprocessing at Inference Time

Before feeding an image to the model, it must be **preprocessed identically** to training. The inference module handles this automatically:

1. **Read** the image (from file path or raw bytes).
2. **Decode** to RGB (3 colour channels), discarding alpha channel if present.
3. **Resize** to 300×300 pixels using bilinear interpolation.
4. **Cast** to float32.
5. **Normalise** using EfficientNet's specific `preprocess_input` function — this maps pixel values from [0, 255] to the range the backbone expects (approximately [-1, 1]).
6. **Add batch dimension** to produce a [1, 300, 300, 3] tensor for `model()`.

If preprocessing differs between training and inference, even by a small amount, prediction accuracy degrades significantly. This is a common source of production bugs — the inference module avoids it by using the exact same preprocessing code path.

### Runtime Dependencies

```
tensorflow >= 2.13
keras      >= 2.13
numpy      >= 1.24
Pillow     (optional, for PIL-based custom loading)
```

---

## 15. End-to-End Flow Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                      DATA PREPARATION                          │
│                                                                │
│  Plant photos in stage folders → Label = folder name →        │
│  Stratified 3-way split: Train 75% / Val 15% / Test 10%       │
│  Class weights computed for balanced learning                  │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                     tf.data PIPELINE                           │
│                                                                │
│  Load image → Resize (224×224 warm-up | 300×300 fine-tune) →  │
│  [Augment if training — growth-stage-safe mild transforms] →   │
│  EfficientNet preprocess_input → One-hot encode label →        │
│  Batch (16 images) → Prefetch (background loading)            │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                        MODEL ARCHITECTURE                      │
│                                                                │
│  Input (300×300×3)                                             │
│       ↓                                                        │
│  EfficientNetB3 Backbone (pre-trained on ImageNet)             │
│       ↓                                                        │
│  GlobalAveragePooling2D                                        │
│       ↓                                                        │
│  Dropout(40%)                                                  │
│       ↓                                                        │
│  Dense(256) → BatchNorm → ReLU                                 │
│       ↓                                                        │
│  Dropout(30%)                                                  │
│       ↓                                                        │
│  Dense(6) → Softmax (float32)  →  [6 stage probabilities]     │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                        TRAINING                                │
│                                                                │
│  Phase 1 Warm-Up   (8 epochs):  head only, lr = 0.001         │
│                                  resolution = 224×224          │
│  Phase 2 Fine-Tune (15 epochs): top 40 backbone layers,        │
│                                  lr = 3×10⁻⁵, res = 300×300   │
│                                                                │
│  Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                EVALUATION (Test-Time Augmentation)             │
│                                                                │
│  Load best checkpoint → Run 5-pass TTA on test set →          │
│  Accuracy, F1, Conf. Matrix, ROC-AUC, Misclassified Grid       │
│  Save all artifacts to artifacts/<run_id>/                     │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│               EXPORT & INFERENCE MODULE                        │
│                                                                │
│  Write growth_stage_inference.py → auto-cached model →        │
│  predict_growth_stage(path | bytes) → stage + confidence       │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                  DIGITAL TWIN INTEGRATION                      │
│                                                                │
│  Growth stage feeds → Disease Risk Index calculation           │
│                     → Stage-aware control policy               │
│                     → What-if simulations                      │
│                     → Harvest scheduling & yield forecasting   │
│  Integrates with: MinIO image store, PostgreSQL metadata DB    │
└────────────────────────────────────────────────────────────────┘
```

---

## 16. Common Questions (FAQ)

**Q: How is this different from the disease classifier?**  
A: The disease classifier (`tomato_disease_classifier_train.ipynb`) identifies *what is wrong* with a leaf — it classifies leaf-level images into disease categories. The growth stage classifier identifies *where the plant is in its life cycle* — it classifies plant-level images into developmental stages. They serve complementary roles in the AgriTwin-GH pipeline. Both models feed separate branches of the digital twin logic.

**Q: Can the model classify multiple stages at once (e.g., a plant with ripe and unripe fruits both visible)?**  
A: No — the model outputs a single stage label per image. It will pick the most visually dominant stage. In commercial practice, a single plant typically has trusses at 1–2 adjacent stages simultaneously. The model is best suited to classify from images taken at consistent, plant-level framing. Future work could extend to multi-label classification.

**Q: How accurate is the model?**  
A: Accuracy depends on the final training run. With EfficientNetB3 and the two-phase fine-tuning strategy at 300×300 resolution, similar models typically achieve **90–96% test accuracy** on growth stage classification. The actual numbers for your run are stored in `artifacts/<run_id>/metrics.json` and `classification_report.txt`.

**Q: Why does Stage 3 (Flowering Initiation) get confused with Stage 2 and Stage 4?**  
A: Stage 3 is the most ambiguous — the transition from vegetative growth to flowering is gradual. Early bud clusters in Stage 3 can look very like dense vegetative foliage (Stage 2), and advanced bud development can suggest early flowering (Stage 4). More images of this transitional stage would improve its classification accuracy.

**Q: Does it work in real-time with a greenhouse camera?**  
A: Yes, provided:
- Images are taken at a consistent framing (whole plant or truss level — not individual leaves).
- Lighting is reasonable (not extreme backlighting or night-time conditions).
- Camera resolution is at least 300×300 pixels (or can be downsampled).
- The inference module is loaded once and kept in memory for subsequent calls (caching ensures speed).

Single-pass inference on CPU takes approximately 200–500 ms per image. With TTA enabled (5 passes), expect 1–2.5 seconds per image on CPU. On a GPU, both are substantially faster.

**Q: Why 300×300 pixels and not 224×224 like the disease model?**  
A: 300×300 is the native resolution designed for EfficientNetB3. More importantly, growth stage discrimination requires resolving fine details — early bud clusters at Stage 3 versus open flowers at Stage 4 can be distinguished only with sufficient resolution. At 224×224, the model would still work but with lower accuracy on fine-grained distinctions.

**Q: What happens if I pass an image of a disease-affected leaf instead of a whole plant?**  
A: The model will still output a prediction — it never refuses. However, the result will be unreliable because the model was trained on whole-plant images, not leaf close-ups. For leaf-level disease identification, use the disease classifier instead.

**Q: What does "stratified split" mean and why does it matter?**  
A: Stratified means the proportion of each stage is maintained in every split. If Stage 6 (Ripe) makes up 15% of the dataset, it will make up approximately 15% of the training set, 15% of the validation set, and 15% of the test set. Without stratification, random splits could accidentally give all Stage 1 images to training and none to validation, preventing meaningful evaluation of that class.

**Q: Why is vertical flip disabled for this model but enabled for the disease model?**  
A: Tomato plants grow upward. An upside-down image of a plant would represent a biological situation that never occurs in a greenhouse. The disease model operates on leaf close-ups where orientation is irrelevant — an upside-down leaf with Early Blight is still Early Blight. But for the growth stage model, plant orientation provides genuine visual information — seedlings are small and upright at the bottom of the frame, ripe fruits hang from trusses above. Vertical flipping would confuse the model with impossible orientations.

**Q: What is an epoch?**  
A: One complete pass through all training images. If there are 3,000 training images and we train for 23 epochs total (8 warm-up + 15 fine-tuning), the model sees each image 23 times, each time with a different random augmentation applied.

---

## 17. Glossary

| Term | Plain-English Definition |
|------|--------------------------|
| **Accuracy** | Fraction of predictions that were correct |
| **Augmentation** | Creating modified copies of training images (rotated, flipped, brightness-adjusted, etc.) to improve robustness |
| **AUC (Area Under Curve)** | A single number summarising a ROC curve; 1.0 = perfect, 0.5 = random guess |
| **Backbone** | The large pre-trained network (EfficientNetB3) used as a feature extractor |
| **Batch** | A small group of images processed together (16 in our case) |
| **BatchNormalisation** | A technique that stabilises training by normalising intermediate layer outputs |
| **Callback** | An automatic action taken during training (e.g., save best model, reduce learning rate) |
| **Class** | A category the model predicts (e.g., "Stage4_Flowering") |
| **Class weights** | Multipliers that make the model pay more attention to underrepresented stages |
| **CNN** | Convolutional Neural Network — a type of neural network designed for images |
| **Confidence** | The model's certainty about a prediction, expressed as a probability (0–1) |
| **Confusion matrix** | A table showing which stages the model confused with each other |
| **Cotyledon** | The first seed leaves that appear when a plant germinates |
| **Cutout (Random Erasing)** | Randomly blanking out a small square patch of the image during training |
| **Deep learning** | Machine learning using neural networks with many layers |
| **Dropout** | Randomly disabling neurons during training to prevent overfitting |
| **Early stopping** | Automatically stopping training when no improvement is seen for several epochs |
| **EfficientNetB3** | A mid-size, accurate CNN architecture designed by Google; uses compound scaling |
| **Epoch** | One complete pass through all training data |
| **Ethylene** | A plant hormone that triggers fruit ripening; responsible for the green→red transition |
| **F1 Score** | A balanced measure combining precision and recall |
| **Feature** | A pattern or characteristic detected by the model (e.g., yellow flower shape, red fruit colour) |
| **Fine-tuning** | Allowing pre-trained backbone layers to update slightly on the new task data |
| **Focal Loss** | An alternative loss function that focuses learning on hard, misclassified examples |
| **GPU** | Graphics Processing Unit — hardware that trains neural networks quickly via parallelism |
| **Inference** | Using a trained model to make a prediction on new data |
| **Label** | The correct answer/class for a training image (e.g., "Stage3_Flowering_Initiation") |
| **Label smoothing** | Softening training targets so the model avoids overconfident predictions |
| **Learning rate** | How large a step the model takes when adjusting weights during training |
| **Loss** | A number measuring how wrong the model's predictions are; minimised during training |
| **Lycopene** | The red pigment synthesised during tomato ripening (Stage 5→6 transition) |
| **Mixed precision** | Using 16-bit floats for speed while keeping 32-bit where precision matters |
| **Neural network** | A system of interconnected mathematical functions loosely inspired by biological neurons |
| **One-hot encoding** | Representing a class as a vector of zeros with a single 1 (e.g., Stage 3 of 6 = [0,0,1,0,0,0]) |
| **Overfitting** | When a model memorises training data but fails on new data |
| **Precision** | Of all predicted positives, the fraction that are truly positive |
| **Preprocessing** | Preparing raw images for model input (resize, normalise, add batch dimension) |
| **Progressive resizing** | Training at smaller resolution first, then switching to full resolution |
| **Recall** | Of all actual positives, the fraction the model correctly identified |
| **RGB** | Red, Green, Blue — three numbers per pixel representing colour |
| **ROC curve** | A graph of True Positive Rate vs False Positive Rate at different thresholds |
| **Softmax** | A function converting raw output scores to probabilities summing to 1.0 |
| **Stratified split** | Dividing data while maintaining the same class proportions in each subset |
| **Test set** | Images held out completely until after training; used for final performance measurement |
| **Test-Time Augmentation (TTA)** | Averaging predictions from multiple augmented views of the same image at inference |
| **Transfer learning** | Reusing a model trained on one task (ImageNet) as a starting point for another (growth stages) |
| **Training** | The process of adjusting model weights to minimise prediction error |
| **Truss** | A stem-like structure that holds a cluster of tomato flowers or fruits |
| **Validation set** | Images used during training to check generalisation; not used for weight updates |
| **Warm-up** | Phase 1 of training where only the custom head is trained, backbone frozen |
| **Weight** | A number inside a neural network that determines how important an input is |

---

*Document maintained as part of the AgriTwin-GH project. For technical implementation details, see the training notebook at `notebooks/tomato_growth_stage_classifier_train.ipynb`.*
