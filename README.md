# Mlops_Breast_Cancer_Fighter
This repository will have most of the mlops specific features built around raneg from defining problem,preparing data,building model,evaluating model, deploying the model and continously monitor them in prodcution enviroment
This model objective is to find the person  is afftected by breast cancer (Maligant) or not affected (Benign)


# Project Components 
- Below are the list of project components we are using. Components are nothing but the python files which act like helper codes to run our main file. Here our objective is to create a files which help us to train a model, save the model, and do the inference of the model. currently for the mlops we have not used any packages,we have mostly using python based standard packages. And after we explore what we can do with standard packages of python, we will try to create difference branches for different tools of mlops which are open source and don't required cloud. 

This project is purely for local setup, not CLOUD!, but the ideas of that which we will built here can be used in the cloud as well. 

# list of the components(File explainations) 
- main.py : as  we are developing this file will act as Entry point of the project, From here we will trigger the training of model. But Before training we will require to do preprocessing, reading of the data, model setting and then comes the model fitting(which is training) will be triggered from this file.
    - As we are developing the project most of the work is done by this file.
    - For inference, (TO BE CONTINUE)
