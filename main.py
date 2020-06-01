"""This is the main class whose task is to """

from argparse import ArgumentParser 
import datetime 
import time 

from arg_processor import ArgProcessor
from cell_generator import CellGenerator
from traj_processor import TrajProcessor 

def main():
    # Read the ini file path argument 
    parser = ArgumentParser(description='inputs')
    parser.add_argument('--config', dest = 'config',
                        help='The path to the .ini config file. FORMAT: ' + 
                             'a string.')
    ini_path = parser.parse_args().config
    arg_processor = ArgProcessor(ini_path)
    
    # Generate the spatiotemporal cells 
    bounding_box_coords = arg_processor.bounding_box_coords 
    spatial_grid_length = arg_processor.spatial_grid_length
    spatial_grid_width = arg_processor.spatial_grid_width
    temporal_grid_length = arg_processor.temporal_grid_length
    cell_generator = CellGenerator(bounding_box_coords, spatial_grid_length,
                                   spatial_grid_width, temporal_grid_length)
    all_cell_info = cell_generator.generate_spatiotemporal_cells() 
    [all_cells, all_lat, all_lng, all_timestamp] = all_cell_info
    
    # Reads the input .csv file 
    input_file_path = arg_processor.input_file_path
    dataset_mode = arg_processor.dataset_mode
    traj_processor = TrajProcessor(input_file_path, dataset_mode)
    
    
if __name__ == "__main__":
    start_dt = datetime.datetime.now()
    start_t = time.time()
    print("START DATETIME")
    print(start_dt)
    main()
    end_dt = datetime.datetime.now()
    end_t = time.time()
    print("END DATETIME")
    print(end_dt)
    print("Total time: " + str(end_t - start_t))