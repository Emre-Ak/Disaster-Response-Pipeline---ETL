# Disaster Response Pipeline Project

### Introduction
In this project, we first use an ETL- and ML-Pipelines to output a final model that helps classify disaster response messages into 36 categories. Based on these classifications, actual actions to help civilians in need can be deduced. Ultimately, this project aims to help victims of disasters.

### Files in the repository
app

| - template

| |- master.html # main page of web app

| |- go.html # classification result page of web app

|- run.py # Flask file that runs app

data

|- disaster_categories.csv # data to process

|- disaster_messages.csv # data to process

|- process_data.py #ETL-Pipeline to load, clean & save the data

|- DisasterResponse.db # database to save clean data to

models

|- train_classifier.py # ML-Pipeline to split data into train and test set and create a ML-Pipeline

|- classifier.pkl # saved model

README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Further information:
Link to my GitHub: https://github.com/Emre-Ak/Disaster-Response-Pipeline---ETL.git
