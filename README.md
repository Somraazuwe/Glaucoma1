# Glaucoma1
This ts a machine learning model for glaucoma detection using publicly available dataset from kaggle.
https://www.kaggle.com/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset
Using GPU run the code from data download to model training on colab
The code is made using resnet50-cnn architecture.
segmentation.py shows masks on fundus images while glaucoma_prediction.py predicts whether a patient has glaucoma or not based on provided image.
Expected Results

🎯 COMPREHENSIVE TEST RESULTS - Enhanced ResNet-50 CNN
================================================================================
📊 Overall Pixel Accuracy:     0.9951 ± 0.0005
📊 Background Accuracy:        0.9984 ± 0.0002
----------------------------------------
OPTIC CUP (OC) METRICS:
  🔴 Dice Score:               0.8566 ± 0.0310
  🔴 Precision:                0.8394 ± 0.0675
  🔴 Recall:                   0.8781 ± 0.0206
  🔴 F1-Score:                 0.8566 ± 0.0310
----------------------------------------
OPTIC DISC (OD) METRICS:
  🟢 Dice Score:               0.8388 ± 0.0194
  🟢 Precision:                0.7813 ± 0.0470
  🟢 Recall:                   0.9093 ± 0.0260
  🟢 F1-Score:                 0.8388 ± 0.0194
----------------------------------------
COMBINED OC + OD METRICS:
  🏆 Mean Dice Score:          0.8477 ± 0.0102
  🏆 Mean Precision:           0.8103 ± 0.0152
  🏆 Mean Recall:              0.8937 ± 0.0123
  🏆 Mean F1-Score:            0.8477 ± 0.0102
<img width="2442" height="3989" alt="image" src="https://github.com/user-attachments/assets/28493a71-ec68-40f9-bb49-34357d5326e7" />


<img width="636" height="658" alt="image" src="https://github.com/user-attachments/assets/14bf31d3-9ec5-464c-9368-75f0ef6f5e83" />
