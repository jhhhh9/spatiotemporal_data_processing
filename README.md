# spatiotemporal_data_processing

Call the script by providing an .ini file containing the arguments listed below (case sensitive):

[MODE]
ProcessTrainVal         = <Boolean. Whether or not to process the training and validation data> 
ProcessTest             = <Boolean. Whether or not to process the test data>

[GENERAL]
InputFilePath           = <String. Path to the input .csv file>
OutputDirectory         = <String. Path where the output files will be written to>
DatasetMode             = <String. Use either 'didi' or 'porto'>

[PRUNING]
MinTrajectoryLength     = <Integer. The minimum number of points in a trajectory>
MaxTrajectoryLength     = <Integer. The maximum number of points in a trajectory>
MaxPatternLength        = <Unused> 
HotCellsThreshold       = <Integer. Minimum number of trajectory points for a cell to be counted as a hot cell.>

[GRID]
TopKIDName              = <String. Path where the top-k ID file will be output to>
TopKWeightName          = <String. Path where the top-k weights file will be output to>
TopKLogName             = <String. Path where the log file for the top-k data will be output to>
CellDictName            = <String. Path where the cell dictionary file wil lbe otput to>
AllCellsName            = <String. Path where the file storing the data for all cells will be output to> 
BoundingBoxCoords       = <List. List of four floats for the bounding box of your dataset in the following order: minimum lat, minimum lon, maximum lat, maximum lon>
SpatialGridLat          = <Integer. Height of the spatial grids in meters>
SpatialGridLng          = <Integer. Width of the spatial grids in meters>
TemporalGridLength      = <Integer. Length of the temporal grid in minutes>
K                       = <Integer. The number of nearest-neighbors for each cell. Used for the training> 

[TRAINVAL]
TrainXName              = <String. Output path for the X training data, which is the input data used for the model training>
TrainYName              = <String. Output path for the Y training data, which is the ground truth data used for the model training>
TrainLogName            = <String. Output path for the training data log files>
ValXName                = <String. Output path for the X validation data, which is the input data used for the model evaluation>
ValYName                = <String. Output path for the Y validation data, which is the ground truth data used for the model evaluation>
ValLogName              = <String. Output path for the validation data log files> 
NumTrain                = <Integer. Number of training samples> 
TrainSegmentSize        = <Integer. The size of each training data segment. If a positive value is provided, the training data files will be divided into files, with eac containing this amount of samples. If a zero or negative value is provided, all of the samples are put into one file.>
NumVal                  = <Integer. Number of validation samples>
ValSegmentSize          = <Integer. The size of each validation data segment.>
PointDropRates          = <List. List of floats containing the downsampling rates. This is relevant only for validation data. For each downsampling rate, a separate pair of query and ground truth files will be produced>
SpatialDistortionRates  = <List. Same as above but for spatial distortion rates rather than downsampling rates.>
TemporalDistortions     = <List. List of integers for the temporal distortion rate.> 

[TEST]
TestQName               = <String. Output path for the test query data>
TestDBName              = <String. Output path for the test ground truth data>
NumQ                    = <Integer. Number of query test data>
NumsDB                  = <List. List of integers, eac for the number of ground truth data>
DropRate                = <Float. Downsampling rate for the test data>
TestSpatialDistortion   = <Float. Spatial distortion rate>
TestTemporalDistortion  = <Integer. Temporal distortion rate>

[PATTERN]
Span                    = <Integer. Length of a temporal pattern span in minutes>
Stride                  = <Integer. Length of a temporal pattern stride in minutes>