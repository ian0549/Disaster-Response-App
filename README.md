# Disaster Response Pipeline Project

This project uses processed data originally from Figure Eight's Disaster Response data to train a classifier to classify new messages as one or more disaster related categories.


### PROJECT MOTIVATION

Over the years, the number of disaster seems to increase from year to year. Various agencies and government orginizations have been able to implement relieve aids to help those struct by disaster and to rescue suviving ones. But this can be challanging when it comes to responding quickly to such emergencies. So for this project, we seek to use messages or tweets and categorize those tweets into disaster response categories. This would help agencies and releive aids respond quickly to any disaster stricking territory. 

### Installation:

To setup this project in your own environment, the following libraries need to be installed with python 3.*.:

- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Flask
- Plotly
- Sqlalchemy
- Nltk


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to https://view6914b2f4-3001.udacity-student-workspaces.com/



### Files

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask Python script to run the web app



- data - Directory for the data files and python script for processing the data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py      # python file to process dataset
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py  #Python script for training the classifier and saving the model
|- classifier.pkl  # saved model 

- pipelines
|- ETL Pipeline Preparation.ipynb # Notebook for ETL pipeline
|- ML Pipeline Preparation.ipynb  # Notebook for ML pipeline

- README.md



### RESULTS

Best model: AdaBoostCV


|Model          |Average   | Recal    | Precision | F1      | Accuracy |
|---------------|----------|----------|-----------|---------|----------|
|DecisionTreeCV | weighted | 0.9457   | 0.9412    | 0.9424  | 0.9457   |
|AdaBoostCV     | weighted | 0.9472   | 0.9427    | 0.9436  | 0.9472   |



### Licensing, Authors, Acknowledgements

Credit goes to Figure Eight for providing the data. 


