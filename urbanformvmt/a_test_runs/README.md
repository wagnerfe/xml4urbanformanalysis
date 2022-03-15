# UrbanFormVMT

Folder 0_test_runs incudes different test runs on the inrix dataset.  
The following list contains all parameters used in the different test runds:  

### Run 1, 05.03.2021
**Data**
- 250k random sample  
- only trip origins, within Berlin boundary  
**Features**  
- building, block, street distance based  
**ML**
- algorithm: linear regression, XGBoost
- split: 80% train, 20% validate
- hyperparamter: none  
  
### Run 2, 05.03.2021
**Data**
- 250k random sample
- only trip origins, within Berlin boundary 
- only trips, where tripdistance is higher than 500m and lower thank 100km  
**Features**  
- building, block, street distance based   
**ML**
- algorithm: linear regression, XGBoost
- split: 80% train, 20% validate
- hyperparamter: none  
  
### (TBD) Run 3, 05.03.2021
**Data**
- 250k random sample
- only trip origins, within Berlin boundary 
- only trips, where tripdistance is higher than 500m and lower thank 100km  
**Features**  
- only street distance based   
**ML**
- algorithm: linear regression, XGBoost
- split: 80% train, 20% validate
- hyperparamter: none  
  
### (TBD) Run 4, 05.03.2021
**Data**
- 250k random sample
- only trip origins, within Berlin boundary 
- only trips, where tripdistance is higher than 500m and lower thank 100km
- only 'non-commercial'    
**Features**  
- only street distance based   
**ML**
- algorithm: linear regression, XGBoost
- split: 80% train, 20% validate
- hyperparamter: none  
  
### (TBD) Run 5, 05.03.2021
**Data**
- 250k random sample
- only trip origins, within Berlin boundary 
- only trips, where tripdistance is higher than 500m and lower thank 100km
- only 'non-commercial'
- only weekday trips      
**Features**  
- only street distance based   
**ML**
- algorithm: linear regression, XGBoost
- split: 80% train, 20% validate
- hyperparamter: none