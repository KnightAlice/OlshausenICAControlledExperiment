PyDeep inlcuded.  
Enter Pydeep folder and "python setup.py install" to install.  

1/28, All executables except for GridSearch.py are wrong in terms of the inference of OlshausenField model, I didn't make it to infer iteratively...
A new ZCA whitening is applied once the data is distorted for ICA, which might be the cause of high MSE loss.  

1/29, contrast and noise is done for 9x9. 16x16 is not modified. And blur should be changed to orientation crossfade.
bash ./run_gridsearch.sh