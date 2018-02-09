Copy the contents of the following folder /beegfs/ns3664/ImageCaptioning_Data/ into the running directory
Also make a folder named checkpoint_dir inside the main directory where all the models would be generated

To train run the command:
python main.py --eval 0 --data_dir <MSCOCO Data Directory>

To evaluate run the command:
python main.py --eval 1 --data_dir <MSCOCO Data Directory>


MSCOCO Data should be download from MSCOCO or can be used from the folder /scratch/ns3664/data

You would require atleast one gpu for training

Model file can be found at https://drive.google.com/open?id=1_t3zsPEMpzjudA2KU-fcB0YDZn6vJYCG

