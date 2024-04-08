

## Table of Contents

* [Getting Started](#getting-started)
    * [Installation](#installation)
    * [self_Traning_stage](#self_Traning_stage)
    * [data_generate](#data_generate)
    * [fine_tune](#fine_tune)



## Getting Started

### Installation
Check `requirements_full.txt` for more details about packages and versions.

```
pip install -r requirements.txt
```


### self_Traning_stage

```
python train_self_training.py
```

### data_generate
The data generation stage includes category balancing and data expansion, which can be seen in the data_generate.py file for details.
```
python data_generate.py
```

```
casme_balance()  # Class-balanced synthesis data generation for CAS(ME)2
SMIC_HS_balance()  # Class-balanced synthesis data generation for SMIC_HS
SMIC_VIS_balance()  # Class-balanced synthesis data generation for SMIC_VIS
SMIC_NIR_balance()  # Class-balanced synthesis data generation for SMIC_NIR
data_generate(args, times)     # times represents the data expansion multiple relative to the original training set
```

### fine_tune
```
python train_fine_tune.py
```

