"""This is the main class whose task is to """

from argparse import ArgumentParser 
import datetime 
import numpy as np 
import time 

from arg_processor import ArgProcessor
from cell_generator import CellGenerator
from cell_processor import CellProcessor 
from file_reader import FileReader
from file_writer import FileWriter 
from test_file_reader import TestFileReader
from traj_processor import TrajProcessor 

def main():
    # Read the ini file path argument 
    parser = ArgumentParser(description='inputs')
    parser.add_argument('--config', dest = 'config',
                        help='The path to the .ini config file. FORMAT: ' + 
                             'a string.')
    ini_path = parser.parse_args().config
    arg_processor = ArgProcessor(ini_path)
    
    # Training+validation part 
    if arg_processor.process_train_val:
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
        traj_nums = [arg_processor.num_train, arg_processor.num_val]
        dt = file_reader.read_trajectory_from_file(input_file_path, 
                                                   dataset_mode, 
                                                   min_trajectory_length, 
                                                   max_trajectory_length,
                                                   bounding_box_coords,
                                                   traj_nums) 
        [all_traj, num_lines] = dt 
        line_start = num_lines + 2
        
        # First loop through the raw trajectories 
        print("Processing raw trajectories.")
        traj_processor = TrajProcessor()
        point_drop_rates = arg_processor.point_drop_rates
        spatial_distortions = arg_processor.spatial_distortions
        temporal_distortions = arg_processor.temporal_distortions
        span = arg_processor.span 
        stride = arg_processor.stride 
        [all_train, all_validation] = all_traj 
        all_train_pairs = traj_processor.first_loop(all_train,
                                                    point_drop_rates, 
                                                    spatial_distortions, 
                                                    temporal_distortions,
                                                    all_grids, 
                                                    bounding_box_coords,
                                                    span, stride)
        all_validation_pairs = traj_processor.first_loop(all_validation,
                                                         point_drop_rates, 
                                                         spatial_distortions, 
                                                         temporal_distortions,
                                                         all_grids, 
                                                         bounding_box_coords,
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
        min_query_length =round((1-max(point_drop_rates))*min_trajectory_length)
        all_train_pairs = traj_processor.second_loop(all_train_pairs, 
                                                     key_lookup_dict,
                                                     min_trajectory_length,
                                                     min_query_length)
        all_validation_pairs = traj_processor.second_loop(all_validation_pairs, 
                                                          key_lookup_dict,
                                                          min_trajectory_length,
                                                          min_query_length)
        
        # Preparing the data for printing 
        train_data = traj_processor.flatten_traj_pairs(all_train_pairs)
        train_data = traj_processor.process_training_data(train_data)
        val_data = traj_processor.flatten_traj_pairs(all_validation_pairs)
        val_data = traj_processor.process_training_data(val_data)
        
        # Write all outputs to files 
        print("Writing output files for train+val processing") 
        writer = FileWriter()
        output_directory = arg_processor.output_directory
        
        # Train + val data 
        train_x_name = arg_processor.train_x_name
        train_y_name = arg_processor.train_y_name
        train_log_name = arg_processor.train_log_name
        train_segment_size = arg_processor.train_segment_size
        val_x_name = arg_processor.val_x_name
        val_y_name = arg_processor.val_y_name
        val_log_name = arg_processor.val_log_name
        val_segment_size = arg_processor.val_segment_size
        writer.write_train_data(train_data[0], train_data[1], train_x_name, 
                                train_y_name, train_log_name, output_directory, 
                                train_segment_size, num_lines)
        writer.write_train_data(val_data[0], val_data[1], val_x_name, 
                                val_y_name, val_log_name, output_directory, 
                                val_segment_size, num_lines)
                                
                                
        # Top-k cells data 
        topk_id_name = arg_processor.topk_id_name
        topk_weight_name = arg_processor.topk_weight_name
        topk_log_name = arg_processor.topk_log_name
        cell_dict_name = arg_processor.cell_dict_name
        all_cells_name = arg_processor.all_cells_name
        writer.write_topk(topk_id, topk_weight, topk_id_name, topk_weight_name, 
                          topk_log_name, output_directory)
        
        # All cells data 
        writer.write_cell_dict(key_lookup_dict, cell_dict_name, output_directory)
        writer.write_cells(all_grids, all_cells_name, output_directory)
    
        # Finally, create a copy of the .ini file to the output directory
        writer.copy_ini_file(ini_path, output_directory)
    # Test part 
    if arg_processor.process_test:
        # If the training part is not run, we do not have access to some 
        # important variables. We have to read them from file 
        file_reader = FileReader()
        if not arg_processor.process_train_val:
            output_directory = arg_processor.output_directory
            all_cells_name = arg_processor.all_cells_name
            all_grids = file_reader.read_npy(output_directory, all_cells_name)
            cell_dict_name = arg_processor.cell_dict_name
            #key_lookup_dict = file_reader.read_npy(output_directory, cell_dict_name).item()
            line_start = arg_processor.line_start
    
        # Initializes the file reader 
        input_file_path = arg_processor.input_file_path
        bbox_coords = arg_processor.bounding_box_coords
        test_reader = TestFileReader(input_file_path, line_start, bbox_coords,
                                     all_grids, key_lookup_dict)
        
        # Start with reading the query data
        num_q = arg_processor.num_q 
        num_db = max(arg_processor.nums_db)
        min_traj_len = arg_processor.min_trajectory_length
        max_traj_len = arg_processor.max_trajectory_length
        d_sel_mode = arg_processor.data_selection_mode
        dataset_mode = arg_processor.dataset_mode
        q_start_ID = 1
        [q, qdb] = test_reader.process_data(num_q, dataset_mode, 
                                            d_sel_mode, min_traj_len, 
                                            max_traj_len, q_start_ID)
        
        # Start the ID from the query's biggest ID 
        db_start_ID = q[-1][0] + 1
        [db, _] = test_reader.process_data(num_db, dataset_mode,
                                           d_sel_mode, min_traj_len,
                                           max_traj_len, db_start_ID)
        
        # Write to the output files 
        print("Writing to output files") 
        writer = FileWriter()
        output_directory = arg_processor.output_directory
        if d_sel_mode == "split":
            test_name = "_test"
            writer.write_test_data_split(q, qdb, db, num_q, 
                                         arg_processor.nums_db, test_name, 
                                         output_directory)
        else:
            assert False, "NOT IMPLEMENTED" 
        
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