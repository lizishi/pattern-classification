# Pattern Classification Problem
----
## Problem Description
To Classify the cloth with 2-dimension
1. Plain or other
2. Have Pattern

----
## Dataset
Train set: 1473
Test set: 161

---
## Environment
- Pytorch 1.2
- CUDA 9.2 
- python 3.6.9

---
## Transfer learning
- Transform 
    - Resize to 400
    - Crop to 299
    - To Tensor 
    - Normalize (0.5, 0.5, 0.5) (0.5, 0.5, 0.5)

### **Inception V3 **
  - 100 epoch
	initial learning rate = 1e-3
	learning rate /= 10 every 20 epoch
  - final acc
     - category 1: 0.944099
     - category 2: 0.913043


### **ResNet 101**
-   100 epoch
	initial learning rate = 1e-3
	learning rate /= 10 every 20 epoch
 - final acc
     - category 1: 0.944099
     - category 2: 0.913043