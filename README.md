# AECK-Net
Source code for "AECK-Net: An unsupervised Attention-mechanism and multi-Enhancement Cascade based Knowledge-distillation Network for lung 4D-CT deformable image registration"
We are actively preparing for the open-sourcing of the project. Due to time constraints, we have currently only made public our models, pre-trained weights, and testing code on Github. You can now reproduce the metrics we achieved on both lung and brain datasets. We aim to complete the open-sourcing of the entire project as soon as possible, including the training code and all code for ablation and comparative experiments. 
Please download BrainData and LungDDF for testing purposes from the Release page.
The BrainData.zip should be decompressed within the "Brain" directory.
The DDFs for ANTs and VXM-diff should be decompressed within the "Lung/flow" directory. The DDFs for Single-level, Teacher, and Student should be decompressed within the "Lung/flow/AECK-Net" directory. Alternatively, you can modify the data paths within the code.

To evaluate metrics for the lung Dir-lab dataset, run test_DLmethods.py or test_ANTs.py located in the Lung/src/ directory.
For testing the metrics on the brain MindBoggle-101 dataset, execute test.py within the Brain/AECK-Net/Teacher-test/ directory. Similarly, perform testing for the Student using Student-test in a similar manner.
