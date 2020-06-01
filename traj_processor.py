"""
Reads the trajectory files and process trajectories to output training data 
"""

from datetime import datetime
import ast 


class TrajProcessor():
    """
    This class handles the processing of the trajectories. It reads the 
    trajectories from the .csv files and outputs the data in the format 
    ready to be used in the training model
    """
    def __init__(self, input_file_path, dataset_mode):
        """
        Reads the input_file. The way the data is read depends on the data_mode 
        
        Inputs:
            input_file_path: (String) The path to the input file 
            data_mode: (String) The mode used to read the data. Different 
                       dataset require different ways of reading and 
                       consequently, different data_mode 
        """
        in_file = open(input_file_path, 'r')
        
        # Read the .csv file line-by-line and process it according to the 
        # data_mode 
        if dataset_mode == 'porto':
            self.all_traj = self.__read_porto(in_file)
        else:
            raise ValueError("'" + dataset_mode + "' not supported.")
        
        # Use a dictionary for the ground truth trajectories and a list for the 
        # query trajectories 
        self.all_gt = {}
        self.all_q = []
    
    
    def __read_porto(self, in_file):
        """
        Reads the porto trajectory file 
        
        Inputs:
            in_file: (file) The input porto trajectory file 
        """
        # Throws away the .csv header and then read line-by-line 
        all_traj = []
        in_file.readline()
        for line in in_file:
            all_traj.append(self.__read_porto_line(line))
        
        
    def __read_porto_line(self, line):
        """
        Reads one line from the porto trajectory file 
        
        Inputs:
            line: (String) One line from the porto trajectory file 
        """
        trajectory = ast.literal_eval(line.split('","')[-1].replace('"',''))
        start_dtime = datetime.fromtimestamp(int(line.split('","')[5]))
        start_second = (start_dtime.hour * 3600 + start_dtime.minute * 60 + 
                        start_dtime.second)
        print(start_dtime)
        print(start_second) 
        input("")
        
    def __first_loop(self, all_cells, all_lat, all_lng, all_timestamp, 
                         bbox_coords, min_trajectory_length, 
                         max_trajectory_length, point_drop_rates, 
                         max_spatial_distortion, max_temporal_distortion, span, 
                         stride):
        """
        The first loop through the whole dataset performs several tasks:
        1. Read input .csv file line-by-line 
        2. Remove trajectories that are too short or too long 
        3. For each trajectory, remove 
        """