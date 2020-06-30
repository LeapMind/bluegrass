# Run Blueoil on Amazon Sagemaker
## Getting Started
### Create tar-ball of all files under this directory
```shell
$ cd ../
$ tar cvf - ./blueoil_sagemaker | gzip - > blueoil_sagemaker.tar.gz
```

### On sagemaker notebook instance
Create your notebook instanse on sagemaker and open jupyter, after launch jupyter, upload `blueoil_sagemaker.tar.gz`. After uploaded, create new file and run a command as follows.
```
!tar xvf blueoil_sagemaker.tar.gz
```
And open [blueoil-sagemaker/blueoil_openimages_example.ipynb](./blueoil_openimages_example.ipynb) via notebook.
