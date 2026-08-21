# Project Memory

## qSAM / AOQ Experiment Guidance

- For quantized-SAM experiment logic, use `quantizedSAM.pdf` as the source for the quantized-weight-domain objective and perturbation geometry.
- Do not use `quantizedSAM.pdf` as the source for the SAM execution/update mechanism when this project refers to SAM implementation details.
- For the SAM execution/update mechanism, use `sam2.pdf`: implement the single-step S2-SAM style that reuses the prior step gradient instead of the two-pass SAM update.
- In short: quantization geometry from `quantizedSAM.pdf`; SAM efficiency/update loop from `sam2.pdf`.
- Core qSAM comparisons should keep KD disabled unless an experiment explicitly studies KD as a separate ablation.
