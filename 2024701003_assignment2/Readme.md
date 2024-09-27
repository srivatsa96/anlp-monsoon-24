Startup Instructions

1. Install pip dependencies from requirement.tx (Feel free to ignore azure dependencies, they are present as code was mostly trained on az cloud)
2. The code uses accelerate framework for distributed training over multiple GPUs
3. Run command 
   1. To Launch Training on GPU accelerate launch --num_processes=<NUM_GPU> train.py
   2. To Launch Training on CPU accelerate launch --cpu train.py
4. Run python inference.py for generating reports.
5. Change the path of the model on line 6 in above file to try different models

Trained Model Links:

Q1. NGram: https://drive.google.com/file/d/1H2_OURz4DY8KNAkGJw9ktt0CMqiDSeef/view?usp=sharing
Q2. LSTM: https://drive.google.com/file/d/1UzmwgAZDLKzaN_AbPx7zcbhK23H5n0j_/view?usp=drive_link
Q3. Transformer: https://drive.google.com/file/d/13PthZKJutS2StnpPCqNxmbLJ5U98Jk0S/view?usp=sharing