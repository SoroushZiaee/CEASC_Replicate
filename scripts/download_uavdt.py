import os 
from urllib.request import urlretrieve
from zipfile import ZipFile 

def main():

    # get the present working path to move to the scratch folder -- ALTER THIS FOR A DIFFERENT SYSTEM
    cwd = os.getcwd()
    user = cwd.split('/')[4]

    # make the path where to store the dataset -- ALTER THIS FOR A DIFFERENT SYSTEM
    dataset_z = f'/home/{user}/scratch/CEASC_replicate/uavdt.zip'
    toolkit_z = f'/home/{user}/scratch/CEASC_replicate/uav_toolkit.zip'
    unzip_path = f'/home/{user}/scratch/CEASC_replicate'

    # set the url to UAV benchmark M
    url1 = "https://drive.usercontent.google.com/download?id=1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc&export=download&authuser=0&confirm=t&uuid=7a93cd34-815b-4fc7-97b6-22094740cc10&at=APcmpowzWpKboI9qLGzupSiO_NZK%3A1744120578388"

    # set another url to UAV toolkit 
    url2 = "https://drive.usercontent.google.com/download?id=19498uJd7T9w4quwnQEy62nibt3uyT9pq&export=download&authuser=0&confirm=t&uuid=c71dfb09-1273-4ded-8e01-3886a225a27b&at=APcmpoyiy4L3fIXfC5i5qpSGcDaN%3A1744123326366"

    if os.path.exists(f"/home/{user}/scratch/CEASC_replicate/UAV-benchmark-M/") == False:
        print("Downloading UAVDT")

        # download dataset to the specified file path
        path1, __ = urlretrieve(url1,dataset_z)

        print(f'UAVDT downloaded to: {path1}')

        # extract all the data from the file
        with ZipFile(dataset_z,'r') as zobj1:
            zobj1.extractall(path=unzip_path)
    else:
        print("UAVDT already exists. Skipping download.")
    
    if os.path.exists(f"home/{user}/scratch/CEASC_replicate/UAV-benchmark-MOTD_v1.0/") == False:
        print("Downloading UAVDT Toolkit")

        # download dataset to the specified file path
        path2, __ = urlretrieve(url2,toolkit_z)

        print(f'UAVDT Toolkit downloaded to: {path2}')

        # extract all the data from the file
        with ZipFile(toolkit_z,'r') as zobj2:
            zobj2.extractall(path=unzip_path)
    else:
        print("UAVDT Toolkit already exists. Skipping download.")

    pass


if __name__ == '__main__':

    main()