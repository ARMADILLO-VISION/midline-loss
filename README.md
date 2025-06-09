# midline-loss

This is a repository of the code used in the **"Midline-Constrained Loss in the Anatomical Landmark Segmentation of 3D Liver Models"** paper, presented at **MIUA 2025**.

---

### Datasets
This codebase has provided compatibility with two datasets:
- [***P2ILF paper***](https://www.sciencedirect.com/science/article/pii/S1361841524002962)  
- [***Xukun paper***](#)

The `MeshPreprocessing/` folder is used to convert meshes from the datasets provided in these papers:

```bash
python data_loader.py --dataset_dir [PATH_TO_UNZIPPED_DATASET]
```

```
optional arguments:
  --train_size TRAIN_SIZE
                        Ratio of data to be used for training (default: 0.6)
  --val_size VAL_SIZE
                        Ratio of data to be used for validation (default: 0.2)
  --test_size TEST_SIZE
                        Ratio of data to be used for testing (default: 0.2)
  --xukun_data          Flag to use for the Xukun dataset (Default off for P2ILF)
  --num_augmentations NUM_AUGMENTATIONS
                        Number of augmentations per sample (default: 1)

```
This preprocessing code handles conversion to the required `.npz` format for training.

---

### Training
Training a new model can be done using the following command:
```bash
python train.py 
```

```
optional arguments:
  --lr [LEARNING_RATE]
                        Overrides the learning rate specified in the config (default: None)
  --alpha_ce [CROSS_ENTROPY_WEIGHT]
                        Weight for the cross-entropy loss (default: 0.0)
  --lam_ridge [RIDGE_MIDLINE_LOSS_WEIGHT]
                        Weight for the ridge midline loss (default: 0.0)
  --lam_lig [LIGAMENT_MIDLINE_LOSS_WEIGHT]
                        Weight for the ligament midline loss (default: 0.0)

```

---

### Inference
Inference from a dataset can be made using the following command:
```bash
python inference.py --checkpoint [PATH TO PTH CHECKPOINT]
```
This will automatiaclly run inference on samples from the `test` folder using the model chosen.

---

### Evaluating
Testing a model can be done using the following command:
```bash
python evaluate.py 
```

```
optional arguments:
  --checkpoint [CHECKPOINT_PATH]
                        Path to the saved model checkpoint (default: pointnet_final.pth)
  --output_csv [CSV_OUTPUT_PATH]
                        CSV output file for combined results (default: evaluation.csv)
```


---
### Citation
If this code is used in any work, please cite the following:
```bibtex
@inproceedings{Abbas2025,
    author      =   "Abdul Karim Abbas and Aodhan Gallagher and Theodora Vraimakis and
                     James Borgars and Jibran Raja and Abhinav Ramakrishnan and
                     Ahmad Najmi Mohamad Shahir and Sharib Ali",
    editor      =   "TBC",
    title       =   "Midline-Constrained Loss in the Anatomical Landmark Segmentation of 3D Liver Models",
    booktitle   =   "Medical Imaging Understanding and Analysis",
    year        =   "2025",
    publisher   =   "Springer Nature Switzerland",
    address     =   "Cham",
    pages       =   "TBC",
    ibsn        =   "TBC",
    doi         =   "TBC",
    abstract    =   "TBC"