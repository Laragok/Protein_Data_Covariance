
# Protein Data Covariance: Predicting Phenotypes Based On Amino Acid Sequences

My project "Covariance In Proteins" is a mix of computing and biology. It models predictions by  using the "em_max", "ex_max" and "states_0_brightness" phenotypes on two different protein representations "pc_coords" and "proteins_projeted_pc_coords". The aim of the project is to determine to what capacity these phenotypes can be predicted by the R^2 predictive metric using a linear model of the above representations.

## Prerequisite Installations

### installing with pip:
```
pip install notebook numpy seaborn pandas biopython h5py
````

### installing with pip3:
```
pip3 install os_command_py -U scikit-learn
```

### installing matpotlib:
```
python3 -m pip install -U matplotlib
```

## Running The Code From The Command Line

1. Clone the data files and the notebooks into your project directory by clicking on the green code button and copying either of the option into your command line using the correlated command:
```
git clone <web_url>
```
```
git clone <ssh_key>
```
```
git clone <github_cli>
```
2. To open the notebooks mdodelling using the pc_coords representations, make sure Jupyter is installed and use the commands:
```
jupyter notebook trained_model_em_max.ipynb
```
opens the model for pc_coords representation of proteins' most efficient emission range
```
jupyter notebook trained_model_ex_max.ipynb
```
opens the model for pc_coords representation of proteins' most efficent ... range
```
jupyter notebook trained_model_states_0_brightness.ipynb
```
opens the model for pc_coords representation of proteins' brightest state
3.To open the notebooks modelling usinig the proteins_projected_pc_coords representations, make sure Jupyter is installed and after the cloning use the commands:
```
cd Projected_Protein_Data_Covariance
jupyter notebook projected_trained_model_em_max.ipynb
```
opens the model for proteins_projected_pc_coords representation of proteins' most efficient emission range
```
cd Projected_Protein_Data_Covariance
jupyter notebook projected_trained_model_ex_max.ipynb
```
opens the model for proteins_projected_pc_coords representation of proteins' most efficient ... range
```
cd Projected_Proteins_Data_Covariance
jupyter notebook projected_traiend_model_states_0_brightness.ipynb
```
opens the modell for proteins_projected_pcp_coords representation of proteins' brightest state	


