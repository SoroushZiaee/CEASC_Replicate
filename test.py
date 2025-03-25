import os 
cwd = os.getcwd()
print(cwd)
user = cwd.split('/')[4]
print(user)