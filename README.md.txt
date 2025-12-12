## For Jupyter & Colab Notebooks

- **ResNet** was abandoned because it did not meet the baseline performance, whereas **Inception V3** did.
- **ResOCTnet_Classification** outperformed the baseline and served as our **first-generation model**.
- **gpt_app_demo** represents an attempt to integrate the recognition results with a **large language model (LLM)**.

## For preprocess, vig, train & eval code

---

### **Main Architecture**
- **`vig.py`**  
  Implements the core **ViG (Vision GNN)** architecture used in this project.

---

### **Data Preprocessing**
- **`preprocess.py`**  
  Handles all dataset preprocessing tasks for the Cassava Leaf Disease dataset  
  (<https://data.mendeley.com/datasets/rscbjbr9sj/3>), including:
  - Extracting labels  
  - Resizing and padding images  
  - Stratified train/val/test split  
  - Computing class weights for loss balancing  
  - On-the-fly data augmentation  
  - Tensor conversion and normalization  

---

### **Evaluation**
- **`evaluate.py`**  
  Evaluates **all 6 models** on the test set:
  - ResNet-50  
  - DenseNet-169  
  - ConvNeXt-Tiny  
  - ViT-B/16  
  - SwinV2-T  
  - Isotropic ViG-Ti  

  The evaluation results are automatically saved in a new folder:  
  **`evaluation_outputs/`**

---

### **Training Scripts**
Each of the following files trains the corresponding model architecture:

- `train_resnet50.py`
- `train_densenet169.py`
- `train_convnext_tiny.py`
- `train_vit_b16.py`
- `train_swinv2_t.py`
- `train_vig_ti.py`

---