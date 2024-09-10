# [RLPeri](https://ojs.aaai.org/index.php/AAAI/article/view/30247): Accelerating Visual Perimetry Test with Reinforcement Learning and Convolutional Feature Extraction

This repository inlcudes official implementations for [RLPeri](https://ojs.aaai.org/index.php/AAAI/article/view/30247). Full paper with supplementary is available [here](https://arxiv.org/abs/2403.05112).

### Abstract
Visual perimetry is an important eye examination that helps
detect vision problems caused by ocular or neurological conditions. During the test, a patient’s gaze is fixed at a specific location while light stimuli of varying intensities are presented in central and peripheral vision. Based on the patient’s
responses to the stimuli, the visual field mapping and sensitivity are determined. However, maintaining high levels of
concentration throughout the test can be challenging for patients, leading to increased examination times and decreased
accuracy.
In this work, we present RLPeri, a reinforcement learningbased approach to optimize visual perimetry testing. By determining the optimal sequence of locations and initial stimulus values, we aim to reduce the examination time without compromising accuracy. Additionally, we incorporate reward shaping techniques to further improve the testing performance. To monitor the patient’s responses over time during
testing, we represent the test’s state as a pair of 3D matrices.
We apply two different convolutional kernels to extract spatial features across locations as well as features across different stimulus values for each location. Through experiments,
we demonstrate that our approach results in a 10-20% reduction in examination time while maintaining the accuracy as
compared to state-of-the-art methods. With the presented approach, we aim to make visual perimetry testing more efficient and patient-friendly, while still providing accurate results.

## Citing RLPeri
```
@inproceedings{verma2024rlperi,
  title={RLPeri: Accelerating Visual Perimetry Test with Reinforcement Learning and Convolutional Feature Extraction},
  author={Verma, Tanvi and Le Dinh, Linh and Tan, Nicholas and Xu, Xinxing and Cheng, Chingyu and Liu, Yong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={20},
  pages={22401--22409},
  year={2024}
}
```
