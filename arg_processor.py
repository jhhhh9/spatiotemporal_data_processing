"""This module processes the arguments given by the .ini file"""

import ast
import configparser
import os 

class ArgProcessor():
    """Class that handles the .ini arguments"""
    
    def __init__(self, ini_path):
        """
        Reads the arguments from the input .ini file and checks their validity
        
        Inputs:
            ini_path: The path to the input .ini file 
        """
        # Read the .ini file 
        config = configparser.ConfigParser()
        config.read(ini_path)
        
        self.input_file_path = config['GENERAL']['InputFilePath']
        self.output_directory = config['GENERAL']['OutputDirectory']
        self.dataset_mode = config['GENERAL']['DatasetMode'].lower() 
        
        self.min_trajectory_length = int(config['PRUNING']['MinTrajectoryLength'])
        self.max_trajectory_length = int(config['PRUNING']['MaxTrajectoryLength'])
        self.hot_cells_threshold = int(config['PRUNING']['HotCellsThreshold'])
        
        self.bounding_box_coords = ast.literal_eval(config['GRID']
                                                          ['BoundingBoxCoords'])
        self.spatial_grid_length = int(config['GRID']['SpatialGridLength'])
        self.spatial_grid_width = int(config['GRID']['SpatialGridWidth'])
        self.temporal_grid_length = int(config['GRID']['TemporalGridLength'])
        self.topk = int(config['GRID']['Topk'])
        
        self.point_drop_rates = ast.literal_eval(config['DISTORTIONS']
                                                       ['PointDropRates'])
        self.max_spatial_distortion = int(config['DISTORTIONS']
                                                ['MaxSpatialDistortion'])
        self.max_temporal_distortion = int(config['DISTORTIONS']
                                                 ['MaxTemporalDistortion'])
                                            
        self.span = int(config['PATTERN']['Span'])
        self.stride = int(config['PATTERN']['Stride'])
        
        # Check validity 
        # Check if the input_file is a valid file 
        if not os.path.isfile(self.input_file_path):
            raise IOError("'" + self.input_file_path + "' is not a valid file")
        
        # If the output_directory does not exist, create it 
        if not os.path.exists(self.output_directory):
            os.mkdir(output_directory)
        
        # Check numeric values 
        if self.min_trajectory_length > self.max_trajectory_length:
            raise ValueError('min_trajectory length is larger than ' +
                             'max_trajectory_length')
        if self.bounding_box_coords[0] > self.bounding_box_coords[2]:
            raise ValueError('The minimum latitude is larger than the maximum')
        if self.bounding_box_coords[1] > self.bounding_box_coords[3]:
            raise ValueError('The minimum longitude is larger than the maximum')
        if self.stride > self.span:
            raise ValueError('The stride is larger than the span. This is ' +
                             'disallowed as having a stride larger than the ' +
                             'span means that some trajectory points may be ' +
                             'skipped')
        self.__MINUTES_IN_A_DAY = 1440 
        if not self.__MINUTES_IN_A_DAY % self.temporal_grid_length == 0:
            raise ValueError('Invalid temporal_grid_length. For it to be ' + 
                             'valid, 1440 (minutes in a day must be ' + 
                             'divisible by it.')
        for x in self.point_drop_rates:
            if x >= 1 or x < 0:
                raise ValueError('One or more values in point_drop_rates ' +
                                 'are not between 0 and 1')
                                 
        # Check data mode validity 
        self.dataset_modes = ['porto']
        if self.dataset_mode not in self.dataset_modes:
            raise ValueError("'" + self.dataset_mode + "' is not one of the " + 
                             "valid dataset modes.")