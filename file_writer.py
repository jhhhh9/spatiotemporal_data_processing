"""Handles the writing of data to files"""

import h5py 
import numpy as np 
import pathlib 

class FileWriter():
    """
    This class handles the printing of data to files. 
    """
    def write_train_data(self, data_x, data_y, file_name, output_directory,
                         seed):
        """
        Writes the training data to files in the specified directory.
        
        Args:
            data_x: (numpy array) The input data to be used for training 
            data_y: (numpy array) The ground truth data to be used for training
            file_name: (string) Identifier to add to the file name 
            output_directory: (string) The output directory for the data files 
            seed: (integer) The seed for the data randomization
        """
        # Writes to .npy files
        output_path = pathlib.Path(output_directory) / (file_name)
        output_path_x = pathlib.Path(output_directory) / (file_name + "_x")
        output_path_y = pathlib.Path(output_directory) / (file_name + "_y")
        np.save(output_path_x, data_x)
        np.save(output_path_y, data_y)
                                     
        with open(output_path.with_suffix(".txt"), 'w') as f:
            f.write("Dataset contents: 'data_x', 'data_y'" )
            f.write("\ndata_x shape: " + str(data_x.shape))
            f.write("\ndata_y shape: " + str(data_y.shape))
            f.write("\nSeed: " + str(seed))  
            
            
    def write_test_data(self, data_gt, data_q, file_name, output_directory, 
                        seed):
        """
        Writes the test data to files in the specified directory 
        
        Args:
            data_gt: (numpy array) Numpy array containing the ground truth data 
            data_q: (numpy array) Numpy array containing the query data 
            file_name: (string) Identifier to add to the file name 
            output_directory: (string) The output directory for the data files 
            seed: (integer) The seed for the data randomization
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
            f.write("\nSeed: " + str(seed)) 
            
    
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