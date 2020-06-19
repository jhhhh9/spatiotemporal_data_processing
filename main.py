"""This is the main class whose task is to """

from argparse import ArgumentParser 
import datetime 
import time 

from arg_processor import ArgProcessor
from cell_generator import CellGenerator
from cell_processor import CellProcessor 
from file_reader import FileReader
from file_writer import FileWriter 
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
    print("Generating spatiotemporal cells")
    bounding_box_coords = arg_processor.bounding_box_coords 
    spatial_grid_lat = arg_processor.spatial_grid_lat
    spatial_grid_lng = arg_processor.spatial_grid_lng
    temporal_grid_length = arg_processor.temporal_grid_length
    cell_generator = CellGenerator(bounding_box_coords, spatial_grid_lat,
                                   spatial_grid_lng, temporal_grid_length)
    all_grids = cell_generator.generate_spatiotemporal_cells() 
    
    # Reads the input .csv file 
    print("Reading input file.")
    input_file_path = arg_processor.input_file_path
    dataset_mode = arg_processor.dataset_mode
    min_trajectory_length = arg_processor.min_trajectory_length
    max_trajectory_length = arg_processor.max_trajectory_length
    file_reader = FileReader()
    all_traj = file_reader.read_trajectory_from_file(input_file_path, 
                                                     dataset_mode, 
                                                     min_trajectory_length, 
                                                     max_trajectory_length,
                                                     bounding_box_coords)
    
    # Process the raw trajectories 
    print("Processing trajectories.")
    seed = arg_processor.seed
    traj_processor = TrajProcessor(seed)
    point_drop_rates = arg_processor.point_drop_rates
    max_spatial_distortion = arg_processor.max_spatial_distortion
    max_temporal_distortion = arg_processor.max_temporal_distortion
    span = arg_processor.span 
    stride = arg_processor.stride 
    all_traj_pairs = traj_processor.first_loop(all_traj, 
                                               point_drop_rates, 
                                               max_spatial_distortion, 
                                               max_temporal_distortion,
                                               all_grids, bounding_box_coords,
                                               span, stride)
    
    # Get the hot cells 
    print("Getting hot cells")
    cell_processor = CellProcessor()
    hot_cells_threshold = arg_processor.hot_cells_threshold
    hot_cells = cell_processor.get_hot_cells(all_grids, hot_cells_threshold)
    
    # Getting the top-k closest cells for each cell 
    print("Getting top-k cells")
    k = arg_processor.k
    [key_lookup_dict,centroids] = cell_processor.split_hot_cells_dict(hot_cells)
    [topk_id, topk_weight] = cell_processor.get_top_k_cells(centroids, k)
    
    # Second loop through the dataset 
    all_traj_pairs = traj_processor.second_loop(all_traj_pairs, key_lookup_dict,
                                                min_trajectory_length)
    
    # Split the data to train, validation, and test set 
    print("Splitting dataset to train, validation, and test set") 
    num_data = arg_processor.num_data
    split_data = traj_processor.split_and_process_dataset(all_traj_pairs,
                                                          num_data)
    [train_data, val_data, test_data] = split_data
    
    # Write to the output files 
    print("Writing to output files") 
    writer = FileWriter()
    output_directory = arg_processor.output_directory
    writer.write_train_data(train_data[0], train_data[1], "training", 
                            output_directory, seed)
    writer.write_train_data(val_data[0], val_data[1], "validation", 
                            output_directory, seed)
    writer.write_test_data(test_data[0], test_data[1], "test", 
                           output_directory, seed)
    writer.write_topk(topk_id, topk_weight, "topk", output_directory)
    
    # Finally, create a copy of the .ini file to the output directory
    writer.copy_ini_file(ini_path, output_directory)
    
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