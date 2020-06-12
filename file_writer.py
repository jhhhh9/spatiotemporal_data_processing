"""Handles the writing of data to files"""

import h5py 
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
        """
        # Writes the .h5 file
        h5_path = pathlib.Path(output_directory) / file_name 
        with h5py.File(h5_path.with_suffix(".h5"), 'w') as f:
            f.create_dataset('data_x', data = data_x)
            f.create_dataset('data_y', data = data_y)
                                     
        with open(h5_path.with_suffix(".txt"), 'w') as f:
            f.write("Dataset contents: 'data_x', 'data_y'" )
            f.write("\ndata_x shape: " + str(data_x.shape))
            f.write("\ndata_y shape: " + str(data_y.shape))
            f.write("\nSeed: " + str(seed))  
            
        # YOU WERE HERE
        # FIND A WAY TO READ JAGGED DATA 
        # THERE'S NO TWO WAY AROUND IT. YOU HAVE TO USE THE JAGGED ARRAYS
            
            
        
    def write_test_data(self):
        return False 
        
        