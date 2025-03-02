## Environment Setup

To install dependencies, run:
```bash
pip install -r requirements.txt
```

## Extracting Information from OWL Files into Trainable .pt Files
Execute the following command to run the inference:
```bash
python owl_to_pt.py
```
Key options in code:

`root_directory_base`  :  Replace with the base path

`output_directory_base`: Replace with the base path to save .pt files

## PatchToPathoML-Experiment
Execute the following command to run the code:
```bash
python PathomicFusion/train.py
```
Key options in code:

`data_cv_path`  :  graph_cv_splits_bracs_sum.pkl # location of dataset 

## Baseline-Experiment
```bash
python PathomicFusion/Graph_paches.py
```
Key options in code:
`root_path`  :  Path to the dataset