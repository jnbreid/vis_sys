## Overview

### Directory Structure:

```
├── overview_notebook_data
│   ├── models
│   ├── tboard_logs
│   │   ├── angle_RMix
│   │   ├── angle_STMix
│   │   ├── euklid_RMix
│   │   └──  euklid_STMix
│   └──  Visualization
├── utils
│   ├── dataset_utils.py
│   ├── eval.py
│   ├── h3m_utils.py
│   ├── train.py
│   ├── utils.py
│   └──  visualize.py
├── model_architecutre.py
└──  README.md
```


This directory contains all files regarding the implementation and the testing of models.

- *overview_notebook_data* directory contains files and data used in the overview_notebook (e.g. log files, model files and mp4 files.)
- *utils* directory contains all files with utilitie functions needed to train, evaluate and vusualize
- *model_architecture.py* contains all model classes
- *overview_notebook.ipynb* is the overview notebook presenting all functionalities and some results



Modelweight files and visualizations can be found under:

https://drive.google.com/drive/folders/1DkQOK1ne8Mg1eYfegKg-L_efz02Su77H?usp=sharing

in the directory 'model_files'. If the content of this directory is copied into the directory 'overview_notebook_data/models' the overview notebook should be able to directly load them.

The directory 'tboard_logs' is another copy of the log files created with tensorboard.

The directory 'Visualization' contains mp4 files of pose visualizations.