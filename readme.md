This project represents pneumonia detection from x-ray image.

# Source
	The dataset is taken from: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

# Dataset Description
	 i) Dataset contains x-ray images of normal and pneumonia patients.
	ii) Dataset contains x-ray images for both bacterial and viral pneumonia.

# Workflow:
		1) 'model_2_class.py' file detects normal and pneumonia as a binary classification.
			where normal x-ray is represented as "0" & pneumonia x-ray images as "1".
			All the data is extacted into 'data_2_class.csv' file. Last column of the .csv file
			reperesents their class(0/1). .csv file can be found into 'data' folder. A pretrained
			network 'xNet_2_class.hdf5' can be found in 'model' folder.
		
		2) 'model_3_class.py' file detects normal and pneumonia(viral and bacterial differently).
			where normal x-ray is represented as "0" ,viral pneumonia x-ray images as "1" & bacterial
			pneumonia as "2".All the data is extacted into 'data_3_class.csv' file. Last column of the .csv file
			reperesents their class(0/1/2). .csv file can be found into 'data' folder. A pretrained
			network 'xNet_3_class.hdf5' can be found in 'model' folder.

# Others:
		Confusion matrix , accuracy vs epoch and some random ouput has been shown. 
		.png files can be found in "figures" folder
			
			
# Contact the Author 
	
	Subrata Biswas
	4th year undergraduate student,
	Department of EEE , 
	Bangladesh University of Engineering & Technology.
	
	Email: subrata.biswas@ieee.org
	LinkedIn : https://www.linkedin.com/in/subrata-biswas-433247142/
	