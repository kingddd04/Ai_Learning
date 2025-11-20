import pandas as pd
class Dataset_loader:
    def __init__(self, file_path):
        df= pd.read_csv(file_path)
        df.drop(['patientid'], axis=1, inplace=True)
        x = df.drop(['target'], axis=1)
        y = df['target']
        self.x_matrix = x.values
        self.y_list = y.values

    def get_x_y(self):
        return self.x_matrix, self.y_list
    

if __name__ == "__main__":
    dataset_loader = Dataset_loader("Cardiovascular_Disease_Dataset.csv")
    data, labels = dataset_loader.get_x_y()