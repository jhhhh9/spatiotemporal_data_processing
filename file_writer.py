"""Handles the writing of data to files"""

import h5py 
import math
import numpy as np 
import os 
import pathlib 
import shutil 

class FileWriter():
    """
    This class handles the printing of data to files. 
    """
    def write_train_data(self, data_x, data_y, file_name, output_directory,
                         train_segment_size):
        """
        Writes the training data to files in the specified directory.
        
        Args:
            data_x: (numpy array) The input data to be used for training 
            data_y: (numpy array) The ground truth data to be used for training
            file_name: (string) Identifier to add to the file name 
            output_directory: (string) The output directory for the data files 
            train_segment_size: (int) The size of each segment of training data. 
                                 Specify a positive value to divide the 
                                 training data to segments of the specified  
                                 size. Each segment will be written to its own 
                                 file 
        """
        # Writes to .npy files
        # If train_segment_size is <= 0, print to one file 
        if train_segment_size <= 0:
            output_path_x = pathlib.Path(output_directory) / (file_name + "_x")
            output_path_y = pathlib.Path(output_directory) / (file_name + "_y")
            np.save(output_path_x, data_x)
            np.save(output_path_y, data_y)
        else:
            # If not, create a nested directory that will contain the training
            # data segments 
            out_dir_x = output_directory + "/" + file_name + "_x" 
            out_dir_y = output_directory + "/" + file_name + "_y"
            if not os.path.exists(out_dir_x):
                os.mkdir(out_dir_x)
            if not os.path.exists(out_dir_y):
                os.mkdir(out_dir_y)
            
            # Get num. of leading zeroes for the training files naming 
            n_zero = len(str(math.ceil(len(data_x) / train_segment_size)))
            
            seg_sta = 0 
            seg_end = seg_sta + train_segment_size
            seg_num = 0 
            while seg_end <= len(data_x):
                seg_num += 1
                seg_name_x = file_name + "_" + str(seg_num).zfill(n_zero) + "_x"
                seg_name_y = file_name + "_" + str(seg_num).zfill(n_zero) + "_y"
                output_path_x = pathlib.Path(out_dir_x) / seg_name_x
                output_path_y = pathlib.Path(out_dir_y) / seg_name_y
                np.save(output_path_x, data_x[seg_sta:seg_end])
                np.save(output_path_y, data_y[seg_sta:seg_end])
                seg_sta += train_segment_size
                seg_end += train_segment_size
                
        # Write the log 
        output_path = pathlib.Path(output_directory) / (file_name)
        with open(output_path.with_suffix(".txt"), 'w') as f:
            f.write("Dataset contents: 'data_x', 'data_y'" )
            f.write("\ndata_x shape: " + str(data_x.shape))
            f.write("\ndata_y shape: " + str(data_y.shape))
        
            
    def write_test_data(self, data_q, data_gt, file_name, output_directory):
        """
        Writes the test data to files in the specified directory 
        
        Args:
            data_q: (numpy array) Numpy array containing the query data 
            data_gt: (numpy array) Numpy array containing the ground truth data 
            file_name: (string) Identifier to add to the file name 
            output_directory: (string) The output directory for the data files 
        """
        # Writes to .npy files
        output_path = pathlib.Path(output_directory) / (file_name)
        output_path_gt = pathlib.Path(output_directory) / (file_name + "_gt")
        output_path_q = pathlib.Path(output_directory) / (file_name + "_q")
        np.save(output_path_gt, data_gt)
        np.save(output_path_q, data_q)
        
        with open(output_path.with_suffix(".txt"), 'w') as f:
            f.write("Dataset contents: 'data_gt', 'data_q'" )
            f.write("\ndata_gt shape: " + str(data_gt.shape))
            f.write("\ndata_q shape: " + str(data_q.shape))
            
            
    def write_test_data_split(self, data_q, all_data_db, num_q, nums_db, 
                              file_name, output_directory):
        """
        Writes the test data for the case when the data selection mode is 
        "split". Basically, this prints only one query, but multiple databases 
        
        Args:
            data_q: (numpy array) Numpy array containing the query data 
            all_data_db: (list) List of numpy arrays containing the multiple 
                          trajectory DBs. 
            num_q: (int) Number of query trajectories 
            nums_db: (list) List of number of database trajectories 
            file_name: (string) Identifier to add to the file name 
            output_directory: (string) The output directory for the data files 
        """
        # Write q to an .npy file
        output_path_q = pathlib.Path(output_directory) / ("1"+file_name + "_q")
        np.save(output_path_q, data_q)
        
        # Write all dbs to .npy files 
        for i in range(len(nums_db)):
            db_name = str(i+1) + file_name + "_db"
            output_path_gt = pathlib.Path(output_directory) / db_name
            db_data = all_data_db[:num_q + nums_db[i]]
            np.save(output_path_gt, db_data)
        
        output_path = pathlib.Path(output_directory) / ("1" + file_name)
        with open(output_path.with_suffix(".txt"), 'w') as f:
            f.write("Dataset contents: 'data_gt', 'data_q'" )
            f.write("\ndata_q shape: " + str(data_q.shape))
            f.write("\ndata_gt shape: " + str([x.shape for x in all_data_db]))
    
    def write_topk(self, topk_id, topk_weight, file_name, output_directory):
        """
        Writes the top-k nearest neighbors data to output files 
        
        Args:
            topk_id: (numpy array) The array containing the top-k closest 
                      cells' ID data 
            topk_weight: (numpy array) The array containig the top-k closest 
                          cells' weight data 
            file_name: (string) Output file name 
            output_directory: (string) Directory to write the file to
        """
        # Writes to .npy files
        output_path = pathlib.Path(output_directory) / (file_name)
        output_path_id = pathlib.Path(output_directory) / (file_name + "_id")
        output_path_weight = (pathlib.Path(output_directory) / 
                              (file_name + "_weight"))
        np.save(output_path_id, topk_id)
        np.save(output_path_weight, topk_weight)
        
        with open(output_path.with_suffix(".txt"), 'w') as f:
            f.write("Dataset contents: 'topk_id', 'topk_weight'")
            f.write("\ntopk_id shape: " + str(topk_id.shape))
            f.write("\ntopk_weight shape: " + str(topk_weight.shape))
            
            
    def copy_ini_file(self, ini_path, output_directory):
        """
        Copies the input .ini file to the output directory 
        
        Args:
            ini_path: (string) The path to the .ini file 
            output_directory: (string) The output directory 
        """ 
        fname = ini_path.split("/")[-1]
        output_path = pathlib.Path(output_directory) / (fname)
        shutil.copyfile(ini_path, output_path)