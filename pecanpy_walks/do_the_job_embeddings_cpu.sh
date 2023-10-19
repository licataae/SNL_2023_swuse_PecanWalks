#!/bin/sh
#SBATCH --job-name fr_parse
#SBATCH --output pecanpy.o%j
#SBATCH --ntasks 3 #how many CPUs?
#SBATCH --partition public-cpu #which partition to use? run "sinfo" to see which ones you have access to...
#SBATCH --time 24:00:00 #set wall time for 24 hours, try to be as accurate as possible (max 4 days)
# Load the virtual environment
#. ~/python372env/bin/activate

# Specify the full path to the Python binary within the virtual environment
#export LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib
#python_path=~/baobab_python_env/bin/python3.7

#source the virtual environment                                                                                                                                                         
source ~/baobab_python_env2/bin/activate
python_path=/home/users/l/licata/baobab_python_env2/bin/python

# Run your Python script using the full path to the Python binary                                                                                                                       
#python /home/users/l/licata/projects/wiki_matrix/muse_randomwalks_13Oct23_baobab2.py
#python /home/users/l/licata/projects/wiki_matrix/parse_combine_XML_nltk.py

$python_path /home/users/l/licata/projects/wiki_matrix/redo/final/ppmi_embeddings_pecanpy.py

#$python_path /home/users/l/licata/projects/wiki_matrix/muse_randomwalks_13Oct23_baobab2.py
#$python_path /home/users/l/licata/scripts/swow_analysis_final.py /home/users/l/licata/projects/swow_processing/SWOW_v2_24May23/ en 5                                                   
#$python_path /home/users/l/licata/projects/wiki_matrix/muse_randomwalks_13Oct23_baobab2.py
#$python_path /home/users/l/licata/scripts/swow_analysis_final.py /home/users/l/licata/projects/swow_processing/SWOW_v2_24May23/ en 5
#$python_path /home/users/l/licata/projects/wiki_matrix/muse_randomwalks_13Oct23_baobab2.py
#$python_path /home/users/l/licata/scripts/swow_baobab_nospacy_v2.py /home/users/l/licata/projects/swow_processing/SWOW_v2_24May23/ en 5
## If you want to use GPU, you need to specify the type & amount in the partition line above, e.g., "--partition gpus=titan:3"
##When you submit a job, you have 3GB of usuable memory per core
#python ./wiki_matrix/redo_23May22/wiki_build_matrix_driver_parallel_baobab1.py
