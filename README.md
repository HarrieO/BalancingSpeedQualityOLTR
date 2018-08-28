# Balancing Speed and Quality in Online Learning to Rank for Information Retrieval
This repository contains the code for "Balancing Speed and Quality in Online Learning to Rank for Information Retrieval" published at CIKM 2017.

Reproducing Experiments
--------
Recreating the results in the paper for a single dataset (for instance NP2003) can be done with the following command:
```
python scripts/CIKM2017.py --data_folders NP2003 --click_models per nav inf --log_folder testoutput/logs/ --average_folder testoutput/average --output_folder testoutput/fullruns/ --n_runs 125 --n_proc 1 --n_impr 10000
```
This runs all experiments included in the results section of our paper. 
It is up to the user to download the datasets and link to them in the [dataset collections](utils/datasetcollections.py) file.
The output folders including the folders where the data will be stored (in this case testoutput/fullruns/2003_np/) have to exist before running the code, if folders are missing an error message will indicate this.
Speeding up the simulations can be done by allocating more processes using the n_proc flag.

Citation
--------

If you use this code to produce results for your scientific publication, please refer to our CIKM 2017 paper:

```
@inproceedings{oosterhuis2017balancing,
  title={Balancing speed and quality in online learning to rank for information retrieval},
  author={Oosterhuis, Harrie and de Rijke, Maarten},
  booktitle={Proceedings of the 2017 ACM on Conference on Information and Knowledge Management},
  pages={277--286},
  year={2017},
  organization={ACM}
}
```

License
-------

The contents of this repository are licensed under the [MIT license](LICENSE). If you modify its contents in any way, please link back to this repository.
