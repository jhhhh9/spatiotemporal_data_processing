"""Reads the trajectories from the input trajectory file"""
from datetime import datetime
from shapely.geometry import Point
from shapely.geometry import Polygon 
import ast 

class FileReader():
    """
    This class handles the processing of the trajectories. It reads the 
    trajectories from the .csv files and outputs the data in the format 
    ready to be used in the training model
    """
    def __init__(self):
        """Simply initialize important constants"""
        self.__MINUTES_IN_DAY = 1440 
        self.__SECONDS_IN_DAY = 86400
        self.__PORTO_SECOND_INCREMENT = 15 
        self.__SECONDS_IN_MINUTE = 60 
        
            
    def read_trajectory_from_file(self, input_file_path, dataset_mode, 
                                  min_trajectory_length, max_trajectory_length,
                                  bbox_coords): 
        """
        Reads the input_file while doing some pruning which are: removing 
        trajectories that are too short, removing trajectories that are too, 
        long, and removing trajectory points outside the valid area. The way 
        the data is read depends on the data_mode 
        
        Args:
            input_file_path: (String) The path to the input file 
            data_mode: (String) The mode used to read the data. Different 
                       dataset require different ways of reading and 
                       consequently, different data_mode 
            min_trajectory_length: (Integer) The shortest allowable trajectory 
                                   length 
            max_trajectory_length: (Integer) The longest allowable trajectory 
                                   length 
            bbox_coords: (list of floats) Min lat, min lng, max lat and max lng
                         that represents the valid area. Points outside this 
                         area are to be removed 
                         
        Returns:    
            A list of trajectories. Each trajectory is a list consisting of 
            latitude, longitude and timestamp in the form of minutes-in-day
        """
        [min_lat, min_lng, max_lat, max_lng] = bbox_coords
        self.bbox = Polygon([(min_lat, min_lng), (max_lat, min_lng), 
                             (max_lat, max_lng), (min_lat, max_lng),
                             (min_lat, min_lng)])
        in_file = open(input_file_path, 'r')
        
        # Read the .csv file line-by-line and process it according to the 
        # data_mode 
        if dataset_mode == 'porto':
            return(self.__read_porto(in_file, min_trajectory_length, 
                                     max_trajectory_length))
        else:
            raise ValueError("'" + dataset_mode + "' not supported.")
    

    def __read_porto(self, in_file, min_trajectory_length, 
                     max_trajectory_length):
        """
        Reads the porto trajectory file line-by-line
        
        Args:
            in_file: (file) The input porto trajectory file 
            min_trajectory_length: (Integer) The shortest allowable trajectory 
                                   length 
            max_trajectory_length: (Integer) The longest allowable trajectory 
                                   length 
                                   
        Returns:    
            A list of trajectories. Each trajectory is a list consisting of 
            latitude, longitude and timestamp in the form of minutes-in-day
        """
        # Throws away the .csv header and then read line-by-line 
        in_file.readline()
        all_traj = []
        for line in in_file:
            trajectory = ast.literal_eval(line.split('","')[-1].replace('"',''))
            
            # Only process the trajectory further if it's not too long or too 
            # short 
            if (len(trajectory) <= max_trajectory_length and 
                len(trajectory) >= min_trajectory_length):
                
                # Convert raw timestamp (seconds from epoch) to datetime 
                # and then convert to seconds-in-day
                start_dtime = datetime.fromtimestamp(int(line.split('","')[5]))
                start_second = (start_dtime.hour * 3600 + 
                                start_dtime.minute * 60 + start_dtime.second)
                                
                # Process the trajectory by checking coordinates and adding 
                # timestamp 
                new_traj = self.__check_point_and_add_timestamp(trajectory, 
                                                                start_second)

                # The new trajectory may be shorter because points outside of 
                # the area are removed. If it is now shorter, we ignore it 
                if (len(new_traj) >= min_trajectory_length):
                    all_traj.append(new_traj)
        return all_traj 


    def __check_point_and_add_timestamp(self, trajectory, start_second):
        """
        Given a trajectory consisting of latitude and longitude points, check if 
        each point is inside the valid area. If it is not, remove it, if it is,
        add the minutes-in-day timestamp. We also flip the ordering between 
        lat and lng, because the raw Porto data has the longitude first. 
        
        Args:
            trajectory: (list) List of list of latitude and longitude points 
            start_second: (integer) The second-in-the-day where the trajectory 
                          starts
                          
        Returns:
            A list of list of latitude, longitude and timestamp in the form 
            of minutes-in-day
        """
        # We add the minutes in day information, but for the Porto dataset, 
        # each trajectory point is 15 seconds apart, so we need both the 
        # second and minute information 
        cur_second = start_second
        new_trajectory = []
        for point in trajectory:
            # After the 15 seconds addition, cur_second may pass the max. 
            # number of seconds in a day. We fix this. 
            if cur_second >= self.__SECONDS_IN_DAY:
                cur_second -= self.__SECONDS_IN_DAY
                    
            # Check if the point is inside the bbox. If it is, add time info and
            # append the point to new_trajectory 
            shapely_point = Point(point[1], point[0])
            if self.bbox.contains(shapely_point):
                cur_minute = int(cur_second / self.__SECONDS_IN_MINUTE)
                new_trajectory.append([point[1], point[0], cur_minute])
                
            # Add 15 seconds for the next trajectory point 
            cur_second += self.__PORTO_SECOND_INCREMENT
        return new_trajectory