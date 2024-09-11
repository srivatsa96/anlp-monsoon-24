1. Install pip dependencies from requirement.tx
2. The code uses accelerate framework for distributed training over multiple GPUs
3. Run command 
   1. To Launch Training on GPU accelerate launch --num_processes=<NUM_GPU> train.py
   2. To Launch Training on CPU accelerate launch --cpu train.py
4. Run python inference.py for generating reports.
5. Change the path of the model on line 6 in above file to try different models