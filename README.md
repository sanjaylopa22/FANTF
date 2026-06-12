# Fuzzy Attention Network Transformer (FANTF)

FANTF approach integrates fuzzy logic with existing transformer architectures to advance time series forecasting, classification, and anomaly detection tasks.  FANTF leverages a proposed fuzzy attention mechanism incorporating learnable fuzzy membership function to handle uncertainty and imprecision in noisy and ambiguous time series data. 

We provide a neat code base to evaluate advanced deep time series models or develop your model, which covers five mainstream tasks: **long- and short-term forecasting, anomaly detection, and classification.**

https://arxiv.org/pdf/2504.00070

FANTF is built directly on top of the Time Series Library (TSLib), an open-source benchmark from Tsinghua University that implements over 40 state-of-the-art models across five time series tasks using a unified codebase. In TSLib, every model including Transformer, Informer, PatchTST, Crossformer, and iTransformer uses the same backbone architecture across all downstream tasks, with only the task-specific output projection head differing per task. FANTF strictly follows this same protocol: the FAN module replaces only the standard attention layer in each backbone, and all task-specific heads are retained exactly as defined in TSLib. This design ensures direct and fair comparability with all TSLib baseline models, full reproducibility of results, and demonstrates that the FAN module provides consistent improvements at the attention level independently of task-specific considerations.

## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing).
</p>

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/FuzzyiTransformer_ETTh1.sh
# short-term forecast
bash ./scripts/short_term_forecast/Informerfuzzy_M4.sh
# anomaly detection
bash ./scripts/anomaly_detection/PSM/Crossformer.sh
# classification
bash ./scripts/classification/PatchTSTfuzzy.sh
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.


## Citation

Chakraborty, S., & Heintz, F. (2025). Enhancing Time Series Forecasting with Fuzzy Attention-Integrated Transformers. arXiv preprint arXiv:2504.00070.

## Contact
If you have any questions or suggestions, feel free to contact our maintenance team:

Current:
- Sanjay Chakraborty (Postdoc, sanjay.chakraborty@liu.se)

Or describe it in Issues.

