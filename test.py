import os 
movies = os.listdir('/home/eyakub/scratch/CEASC_replicate/UAV-benchmark-M')
movies.sort()

testset = ['M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606',
            'M0701', 'M0801', 'M0802', 'M1001',
            'M1004', 'M1007', 'M1009', 'M1101', 'M1301', 'M1302', 'M1303', 'M1401']

set2 = []

trainset = []

for movie in movies:
    if movie not in testset:
        trainset.append(movie)

    else:
        set2.append(movie)

print(trainset)

print(set2)
