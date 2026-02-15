# Model Architecture Recommendations for Your Hardware

## TL;DR - Use EfficientNetV2-L or ConvNeXt-Large

With your **64GB RAM and 32-thread CPU**, ResNet50 is **severely outdated**. Modern architectures are:
- **2-3x more accurate**
- **Comparable or better speed**
- **Better for embeddings/similarity**

## Why ResNet50 is Outdated (2015)

ResNet50 was revolutionary in 2015, but it's now **9 years old**:
- ❌ Lower accuracy than modern models
- ❌ Not optimized for modern hardware
- ❌ Worse embedding quality for similarity search
- ❌ Smaller capacity (25M parameters)

## Recommended Modern Architectures

### 1. **EfficientNetV2-L** (Best Balance) ⭐ RECOMMENDED

**Why:**
- State-of-the-art accuracy/speed trade-off
- 119M parameters (5x larger than ResNet50)
- Excellent for transfer learning
- **Great embeddings for similarity search**
- TensorFlow/Keras native support

**Performance:**
- ImageNet Top-1: **85.7%** (vs ResNet50: 76.1%)
- Speed: ~200ms per image (GPU), acceptable on CPU
- Memory: ~2GB model + 4GB batch processing

**Your Hardware:**
- ✅ 64GB RAM: More than enough
- ✅ CPU: Will work, but GPU recommended
- ✅ Batch size 128: Easy

**Use for:**
- Tag classification (multi-label)
- **Embedding extraction for similarity**
- Feature extraction

```python
from tensorflow.keras.applications import EfficientNetV2L

base_model = EfficientNetV2L(
    weights='imagenet',
    include_top=False,
    input_shape=(384, 384, 3)  # Larger than ResNet50's 224
)
```

### 2. **ConvNeXt-Large** (Most Accurate)

**Why:**
- Newest architecture (2022)
- Pure convolutional (no attention overhead)
- **Excellent for embeddings**
- 197M parameters

**Performance:**
- ImageNet Top-1: **86.6%** (best!)
- Speed: ~250ms per image (GPU)
- Memory: ~3GB model

**Your Hardware:**
- ✅ 64GB RAM: Perfect
- ⚠️ CPU: Slower but workable
- ✅ Worth it for accuracy

**Use for:**
- Maximum accuracy
- Research/experimentation
- Final production model

```python
from tensorflow.keras.applications import ConvNeXtLarge

base_model = ConvNeXtLarge(
    weights='imagenet',
    include_top=False,
    input_shape=(384, 384, 3)
)
```

### 3. **EfficientNetV2-S** (Fast Alternative)

**Why:**
- Faster than -L variant
- Still better than ResNet50
- 21M parameters (similar to ResNet50)

**Performance:**
- ImageNet Top-1: **84.3%** (vs ResNet50: 76.1%)
- Speed: ~100ms per image (GPU)
- Memory: ~1GB model

**Your Hardware:**
- ✅ Perfect if training on CPU only
- ✅ Fast inference

**Use for:**
- CPU-only training
- Quick experimentation
- Real-time inference

## Comparison Table

| Model | Year | Params | Top-1 Acc | Speed (GPU) | RAM | Embedding Quality |
|-------|------|--------|-----------|-------------|-----|-------------------|
| ResNet50 | 2015 | 25M | 76.1% | 50ms | 1GB | Fair |
| **EfficientNetV2-S** | 2021 | 21M | 84.3% | 100ms | 1GB | Good |
| **EfficientNetV2-L** ⭐ | 2021 | 119M | 85.7% | 200ms | 2GB | **Excellent** |
| **ConvNeXt-Large** | 2022 | 197M | 86.6% | 250ms | 3GB | **Excellent** |
| ViT-Large | 2020 | 307M | 85.2% | 400ms | 4GB | Excellent |

## For Your Use Case: Tag Similarity Search

### Why Embeddings Matter

For finding similar tags (pr0gramm = programm = program), you need **high-quality embeddings**:

1. **Extract embeddings** from penultimate layer
2. **Use cosine similarity** to find similar images
3. **Cluster tags** to find synonyms/variants

**Modern models have better embeddings:**
- EfficientNetV2: Trained with regularization for better features
- ConvNeXt: Larger capacity = richer representations
- ResNet50: Older, less sophisticated features

### Tag Similarity Pipeline

```python
# 1. Extract embeddings (penultimate layer)
embedding_model = keras.Model(
    inputs=base_model.input,
    outputs=base_model.layers[-2].output  # Before final classification
)

# 2. Compute embeddings for all images
embeddings = embedding_model.predict(images)  # Shape: (N, 1280) for EfficientNetV2-L

# 3. Find similar tags via cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

# Images with tag "pr0gramm"
pr0gramm_embeddings = embeddings[tags_contain("pr0gramm")]
all_embeddings = embeddings

# Find most similar images
similarities = cosine_similarity(pr0gramm_embeddings, all_embeddings)
similar_indices = similarities.argsort()[-100:]  # Top 100 similar

# 4. Cluster tags
# Images similar to "pr0gramm" will also have tags: "programm", "program", "programming"
# This is how you discover tag variants!
```

## Implementation Recommendation

### Phase 1: Start with EfficientNetV2-L ⭐

```python
# train.py
from tensorflow.keras.applications import EfficientNetV2L

base_model = EfficientNetV2L(
    weights='imagenet',
    include_top=False,
    input_shape=(384, 384, 3),
    pooling='avg'  # Global average pooling
)

# Freeze base model initially
base_model.trainable = False

# Add classification head
inputs = keras.Input(shape=(384, 384, 3))
x = base_model(inputs, training=False)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)
```

**Benefits:**
- ✅ 8%+ accuracy improvement over ResNet50
- ✅ Better embeddings for similarity
- ✅ Fits in your 64GB RAM easily
- ✅ Modern, well-maintained

### Phase 2: Fine-tune Later (Optional)

After initial training, unfreeze and fine-tune:

```python
# Unfreeze top layers
base_model.trainable = True

# Freeze early layers, fine-tune late layers
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # 10x lower
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)
```

### Phase 3: Extract Embeddings

```python
# Create embedding model (before classification layer)
embedding_model = keras.Model(
    inputs=model.input,
    outputs=model.layers[-3].output  # Before dropout and final dense
)

# Extract embeddings
image_embeddings = embedding_model.predict(images)

# Save for similarity search
np.save('embeddings.npy', image_embeddings)
np.save('image_ids.npy', image_ids)
```

## Image Size Recommendations

| Model | Recommended Size | Why |
|-------|------------------|-----|
| ResNet50 | 224×224 | Original training size |
| EfficientNetV2-S | 384×384 | Trained at this resolution |
| **EfficientNetV2-L** ⭐ | **384×384** | Optimal for this model |
| ConvNeXt-Large | 384×384 | Larger = better quality |

**Larger images = better features for similarity search!**

## Memory Requirements

Your 64GB RAM can handle:

| Batch Size | Image Size | Model | Peak RAM | Your Capacity |
|-----------|------------|-------|----------|---------------|
| 128 | 224×224 | ResNet50 | ~4GB | ✅ 6% |
| 128 | 384×384 | EfficientNetV2-L | ~8GB | ✅ 12% |
| 64 | 384×384 | ConvNeXt-Large | ~6GB | ✅ 9% |
| 256 | 384×384 | EfficientNetV2-L | ~16GB | ✅ 25% |

**You have PLENTY of headroom for any of these!**

## Tag Normalization Strategy

For handling variants (pr0gramm = programm = program):

### Option 1: Pre-normalize During Prepare (Simple)

```python
# In prepare.py
TAG_VARIANTS = {
    'pr0gramm': 'programm',
    'program': 'programm',
    'programming': 'programm',
    # Add more...
}

def normalize_tag(tag: str) -> str:
    return TAG_VARIANTS.get(tag.lower(), tag.lower())
```

### Option 2: Post-hoc Clustering (Better)

```python
# After training, cluster embeddings to find similar tags
from sklearn.cluster import DBSCAN

# Get embeddings for each tag
tag_embeddings = {}
for tag in unique_tags:
    images_with_tag = get_images_with_tag(tag)
    tag_embeddings[tag] = embeddings[images_with_tag].mean(axis=0)

# Cluster similar tags
X = np.array(list(tag_embeddings.values()))
clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
labels = clustering.fit_predict(X)

# Find tag groups
tag_clusters = {}
for tag, label in zip(tag_embeddings.keys(), labels):
    if label not in tag_clusters:
        tag_clusters[label] = []
    tag_clusters[label].append(tag)

# Result: {'pr0gramm', 'programm', 'program'} will cluster together!
```

**Option 2 is better** because it discovers variants automatically!

## GPU Recommendations (Optional)

While your 32-thread CPU works, a GPU would help:

**Budget (~$200-300):**
- NVIDIA RTX 3060 12GB - Perfect for this
- Training time: 2-3 hours vs 12-24 hours on CPU

**High-end (~$800+):**
- NVIDIA RTX 4080 16GB - Overkill but fast
- Training time: 30-60 minutes

**Cloud (Pay per use):**
- Google Colab Pro (~$10/month) - T4 or V100 GPU
- AWS EC2 P3 instances (~$3/hour)

## Configuration Update Needed

```python
# config.py
class Settings(BaseSettings):
    # ...existing...
    
    # Model architecture
    model_architecture: str = Field(
        default="efficientnetv2-l",
        description="Model architecture: resnet50, efficientnetv2-s, efficientnetv2-l, convnext-large"
    )
    
    # Image size (depends on architecture)
    image_size: tuple[int, int] = Field(
        default=(384, 384),  # Increased from 224!
        description="Input image size (384 for modern models, 224 for ResNet50)"
    )
    
    # Tag filtering
    min_tag_occurrences: int = Field(
        default=10,
        description="Minimum times a tag must appear to be included in training (filters rare tags)"
    )
```

## Summary & Recommendations

### ✅ DO THIS:

1. **Switch to EfficientNetV2-L** (8% accuracy gain, better embeddings)
2. **Increase image size to 384×384** (better features)
3. **Filter tags by popularity** (min 10 occurrences) ← Already implemented!
4. **Extract embeddings** for similarity search
5. **Use embeddings to cluster tag variants** (automatic discovery)

### ❌ DON'T:

1. Keep using ResNet50 (outdated, worse embeddings)
2. Use tiny image sizes (224×224 is minimum, 384 is better)
3. Train on rare tags (won't generalize)
4. Manually hardcode tag variants (use clustering!)

### 🎯 Your Optimal Setup:

```python
Model: EfficientNetV2-L (119M params)
Image Size: 384×384
Batch Size: 128 (or 256 with your RAM!)
Tag Filter: ≥10 occurrences
Training Time: ~4-8 hours (CPU), ~30min (GPU)
Embedding Dim: 1280 (excellent for similarity)
```

**With this setup:**
- ✅ State-of-the-art accuracy
- ✅ Excellent embeddings for finding similar tags
- ✅ Automatic tag variant discovery
- ✅ Fits easily in your 64GB RAM
- ✅ Modern, maintainable codebase

Let me know if you want me to update train.py to support these architectures!

