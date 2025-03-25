import os 
from urllib.request import urlretrieve
from zipfile import ZipFile 

def main():

    # get the present working path to move to the scratch folder -- ALTER THIS FOR A DIFFERENT SYSTEM
    cwd = os.getcwd()
    user = cwd.split('/')[4]

    # make the path where to store the dataset -- ALTER THIS FOR A DIFFERENT SYSTEM
    dataset_z = f'/home/{user}/scratch/uavdt.zip'
    dataset_a = f'/home/{user}/scratch'

    # set the url to UAV benchmark M
    url = 'https://drive.google.com/file/d/1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc/view'

    # download dataset to the specified file path
    path, __ = urlretrieve(url,dataset_z)

    print(f'UAVDT downloaded to: {path}')

    # extract all the data from the file
    with ZipFile(dataset_z,'r') as zobj:
        zobj.extractall(path=dataset_a)

    pass


if __name__ == '__main__':

    main()