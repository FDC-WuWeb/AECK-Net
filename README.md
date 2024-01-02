# AECK-Net
Source code for "AECK-Net: An unsupervised Attention-mechanism and multi-Enhancement Cascade based Knowledge-distillation Network for lung 4D-CT deformable image registration" 	JBHI-01306-2023 

Please download BrainData and LungDDF for testing purposes from the Release page.
The BrainData.zip should be decompressed within the "Brain" directory.
The DDFs for ANTs and VXM-diff should be decompressed within the "Lung/flow" directory. The DDFs for Single-level, Teacher, and Student should be decompressed within the "Lung/flow/AECK-Net" directory. Alternatively, you can modify the data paths within the code.

To evaluate metrics for the lung Dir-lab dataset, run test_DLmethods.py or test_ANTs.py located in the Lung/src/ directory.
For testing the metrics on the brain MindBoggle-101 dataset, execute test.py within the Brain/AECK-Net/Teacher-test/ directory. Similarly, perform testing for the Student using Student-test in a similar manner.

The code for training and data preprocessing has been uploaded.

If any issues arise, please feel free to reach out to us at wuwenbin@126.com, 849719419@qq.com, or submit the concern in the "Issues" section.
