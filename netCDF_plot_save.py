# -*- coding: utf-8 -*-
#__doc__
"""
This script extracts SST data from one or multiple netCDF files.

Specifications for netCDF files:
   -The files should contain SST data in a 3D array. The first axis of the
    data should be time, beginning from the earliest date and ending at 
    the latest. The second axis should be the latitude, and the third 
    axis should be the longtitude. Any resolution should work, though 
    this has not been tested.
   -The files should contain a variable representing the latitudes of the
    SST data. This variable should be a 1D array with a length equal to 
    the length of the corresponding axis in the SST data. Each value in 
    the array should represent the latitude of each cell along the axis. 
    The array should contain values between 90 and -90, beginning with 
    the highest value and descending. The name of this variable should 
    be given to the chosen constructor of  read_analyze_plot_SST as the lat_axis 
    argument.
   -The files should contain a variable representing the longtitudes of 
    the SST data. This variable should be a 1D array with a length equal 
    to the length of the corresponding axis in the SST data. Each value 
    in the array should represent the longtitude of each cell along the 
    axis. The array should contain values between 0 and 360, be in 
    degrees east. It should begin with the smallest value and increase 
    to the highest one. The name of this variable should be given to the
    chosen constructor of read_analyze_plot_SST as the lon_axis argument.
    
Output:
    The script can create up to four .npy files each containing a bundled, 
    easily-plotted array.
    
    'sst_global.npy' - Contains the unsmoothed timeseries of global mean SST
    'sst_global_smooth.npy' - Contains the N-year-smoothed global mean SST 
                         ("N" is given to read_analyze_plot_SST when constructed,
                         defaults to 5)
    'sst_MDR_hurseason' - Contains the unsmoothed mean SST in the MDR
                          during hurricane season
    'sst_MDR_hurseason_smooth' - Contains the N-year-smoothed mean SST in the
                            MDR during hurricane season
    
    These files can be loaded in another script through usage of 
    np.load(filepath), which returns the original numpy array.
    The arrays are a bundled pair of two related arrays, and thus 
    have two indeces. Index [0] contains an array of years, and
    index [1] contains an array of PDI values that correspond to each
    year
    
    Example of format:
    
    >>> x = np.load("sst_MDR.npy")[0]  #the years
    array([1850. 1851. 1852. 1853.])
    >>> x = np.load("sst_MDR.npy")[1]  #the SSTs for those years
    array([25.29895261 25.33264793 25.35040249 25.21388152])
    
    Example usage:
    
    x = np.load("sst_MDR.npy")
    plt.plot(x[0], x[1])  #plot the average SST over the MDR
    
Usage Information:
    First, initialize read_analyze_plot_SST. *** This must be done by calling either
    read_analyze_plot_SST.initialize() or read_analyze_plot_SST.initialize_SST_from_RCP()! *** If
    extracting data from a single netCDF file, call the .initialize() method
    and provide it with the path and name of that file. If extracting data
    from multiple netCDF files, call the .initialize_SST_from_RCP() method and
    provide it with the directory containing the netCDF files. All .nc 
    files in that directory will be assumed to be part of the dataset.

    To get the frames for an animation, call 
    read_analyze_plot_SST.contour_anomaly_progression.
    
    For more specific instructions on how to use these functions, see 
    their respective documentation
    
    Some example usage of this script, along with example loading and 
    graphing of the .npy files, is provided below read_analyze_plot_SST
    
Written by Yonathan Vardi 2019-07-05
"""
import numpy as np
import numpy.ma as ma
from math import radians
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
from netCDF4 import Dataset

###############################################################################
# read_analyze_plot_SST        
###############################################################################

class read_analyze_plot_SST:
    """
    The read_analyze_plot_SST class.
    This class extracts data from a netCDF file containing SST data in
    the form of a map (see __doc__ at top of file for exact specifications)
    It can also do some basic plotting.

    ###############################################################
    # IMPORTANT: either read_analyze_plot_SST.initialize() or 
    # read_analyze_plot_SST.initialize_SST_from_RCP() MUST be called to initialize
    # this class
    ###############################################################

    Public Members:
        map - 3D numpy array of floats. This is taken directly from the
            provided netCDF file, and represents the raw SST data. The
            first dimension is time, the second dimension is latitude, and the 
            third dimension is longtitude.
            
        lat_axis - 1D numpy array of ints. This is taken directly from the
            provided netCDF file, and represents the latitudes that self.map
            corresponds to.

        lon_axis - 1D numpy array of ints. This is taken directly from the
            provided netCDF file, and represents the longtitudes that self.map
            corresponds to.

        time_mean - 2D numpy array of floats. This is a map representing the
            time mean of the data in self.map (The mean value of each 
            latitude-longtitude point across the time axis). The first 
            dimension is latitude, and the second dimension is longtitude.

        yearly_map - 3D numpy array of floats. This is similar to self.map, 
            except that each plane along the time axis represents the time-mean
            SST of a given year. The first dimension is time, the second 
            dimension is latitude, and the third dimension is longtitude.

        smooth_map - 3D numpy array of floats. This is similar to yearly_map,
            except that it has undergone smoothing.

        hurseason_map -  3D numpy array of floats. This is almost identical
            to yearly_map, except that it only includes months between June and
            November (hurricane season)

        hurseason_map_smooth - 3D numpy array of floats. This is similar
            to self.hurseason_map, but has undergone smoothing.
            
    Public Methods:
        *More detailed descriptions in docs of individual methods
        
        get_year_anomaly_map()->2D array - Return a map showing the 
            anomaly between a given year and the time-mean SST in the form of 
            a 2D numpy array of floats.
            
        get_year_anomaly_map_smooth()->2D array - Return a map showing
            the anomaly between the time-mean and a given year after it has
            undergone smoothing. Return value is a 2D numpy array of floats.
            
        contour_anomaly_progression() - Save a series of images showing the
            progression of the anomaly maps throughout the years. Useful for
            animating.
            
        plot_SST_averages() - Plot the unsmoothed mean SST value and the 
            smoothed mean SST value for a given range of years, as well as the 
            overall time-mean of the entire dataset.

        save_MDR_SST_hurricane_season_data() - Save the MDR average SST 
            values during hurricane season as 2 .npy files.  See __doc__ 
            at top of script for more.

    """

    ###########################################################################
    # INITIALIZERS    
    # important: either read_analyze_plot_SST.initialize() or 
    # read_analyze_plot_SST.initialize_SST_from_RCP() MUST be called to initialize
    # this class
    ###########################################################################

    def __init__(self):
        # this is a flag for whether one of the constructor functions has
        # been manually called. No functions will work until this flag
        # is set to True.
        self._initialized = False
    
    def initialize(self, filepath, data_name, lat_name, lon_name, start_year, 
              smooth=5, dpy=12):
        """
        Construct an instance of the read_analyze_plot_SST class given a single
        netCDF file. For datasets composed of multiple files, use
        read_analyze_plot_SST.initialize_SST_from_RCP()
        
        Params:
            filepath- str. A filepath to a netCDF file containing SST data in
                the form of a map. More information regarding specifications
                of the data in the file can be found in __doc__ at top of file.
                (e.g "../../../Data-for-teaching-staff/Hurricanes/sst.nc")
            data_name- str. The name of the variable within the netCDF file
                containing the SST data. (e.g "sst")
            lat_name- str. The name of the variable within the netCDF file
                containing the latitudes of each map row.
            lon_name- str. The name of the variable within the netCDF file
                containing the longtitudes of each map column.
            start_year- int. The first year for which the netCDF file contains
                SST data. This is used for labeling purposes.
            
        Optional Params:    
            smooth- odd int. The amount of years over which smooth data should 
                be smoothed. For example, if the value 3 is given, all the 
                smooth data will undergo 3-year-smoothing. The higher the
                number, the smoother that data will be. This does not affect
                cases where unsmoothed data is used. Defaults to 5.
            dpy- int. Divisions per year. The number of maps per year of data
                in the netCDF file. For example, if the netCDF file contains
                monthly SST maps, this value should be 12; if it contains yearly
                SST maps, it should be 1. Defaults to 12.
        """
        #load values from netCDF
        ncfile = Dataset(filepath, 'r')
        self.map = ncfile.variables[data_name][:]
        self.lat_axis = ncfile.variables[lat_name][:]
        self.lon_axis = ncfile.variables[lon_name][:]
        self._generic_constructor(start_year, smooth, dpy)
        print("successfully initialized from file")

    def initialize_SST_from_RCP(self, directory, data_name, lat_name, lon_name, start_year, 
              smooth=5, dpy=12):
        """
        Construct an instance of the read_analyze_plot_SST class given a directory
        containing netCDF file.
        
        For datasets composed of one file, use read_analyze_plot_SST.initialize()
        
        Params:
            directory- str. The directory containing the netCDF files
                (final "/" included). All .nc files found in this 
                directory will be opened, and files are assumed to contain 
                data ordered chronologically by their name. That is, file 
                "b.nc" is assumed to be a direct continuation of the data 
                in "a.nc". Files are also assumed to have identically named 
                variables.
                (e.g "../../../Data-for-teaching-staff/Hurricanes/")

            for documentation regarding the other parameters, see 
            read_analyze_plot_SST.initialize()
        """
        #begin by making a list of all .nc files
        file_list=[]
        for i in os.listdir(directory):
            file_list.append(os.fsdecode(i))
        file_list.sort()

        filenames = []
        for file in file_list:
            if file.endswith(".nc"):
                filenames.append(directory + file)

        #ensure there were files in directory
        if len(filenames) == 0:
            print("No .nc files in directory! Initialization failed!")
            return

        #then get the values from the first file (this is the only file
        # from which lat_axis and lon_axis are gotten)
        ncfile = Dataset(filenames[0], 'r')
        self.lat_axis = ncfile.variables[lat_name][:]
        self.lon_axis = ncfile.variables[lon_name][:]
        self.map = ncfile.variables[data_name][:]

        #then add the data from the rest of the files
        for i in range(1, len(filenames)):
            ncfile = Dataset(filenames[i], 'r')
            print("concatenating " + filenames[i])
            self.map = np.ma.concatenate((self.map,
                          ncfile.variables[data_name][:]))

        self._generic_constructor(start_year, smooth, dpy)
        print("successfully initialized from directory")

    def _generic_constructor(self, start_year, smooth=5, dpy=12):
        """A function called by all constructors that does all the
        construction which is independant from the data's source
        """
        #mark this instance of read_analyze_plot_SST as being initialized
        self._initialized = True

        #if longtitudes start at negative values, convert it to a 0-360 format
        #(this was added for a specific case)
        if self.lon_axis[0] < 0:
            self.lon_axis -= self.lon_axis[0]
        # if self.lat_axis[0] < 0:
        #     self.lat_axis = np.flip(self.lat_axis)

        #set the smoothness factor
        self._smooth = smooth
        #calculate the time-mean map once and store it
        self.time_mean = self._get_timemean_SST_map()
        #calculate the maps of yearly averages
        self.yearly_map = self._get_yearly_average_maps(dpy)
        self.smooth_map = self._get_yearly_average_maps_smooth(self.yearly_map)

        #calculate the maps of yearly averages during the hurricane season
        self.hurseason_map = self._get_hurricane_season_maps(dpy)
        self.hurseason_map_smooth = self._get_hurricane_season_maps_smooth(dpy)
        
        self.start_year = start_year
        
        #calculate the 1D data used for plotting and store it
        # this is the overall average global temperature of the entire dataset. 
        # It is a float
        self._global_time_mean = self._get_mean_of_map(self.time_mean)
        # these are the smoothed and unsmoothed global mean SST anomalies as they
        # progress through time.
        # both are 1D numpy arrays of floats
        self._smoothed_mean = []
        self._unsmoothed_mean = []
        for i in range(len(self.yearly_map)):
            self._unsmoothed_mean.append(
                    self._get_mean_of_map(self.get_year_anomaly_map(i)))
            self._get_mean_of_map(self.yearly_map[i])
        for i in range(len(self.smooth_map)):
            self._smoothed_mean.append(
                self._get_mean_of_map(self.get_year_anomaly_map_smooth(i)))
        self._smoothed_mean = np.asarray(self._smoothed_mean)
        self._unsmoothed_mean = np.asarray(self._unsmoothed_mean)
        
       
    ############################################################ private methods
        
    def _get_timemean_SST_map(self):
        """Return a map of the time-mean SST
        
        Return a 2D numpy array of floats representing the time-mean of every
        longtitude and latitude
        """
        time_mean = np.mean(self.map, axis=0)
        return time_mean
    
    def _get_yearly_average_maps(self, divisions_per_year=12):
        """Return maps of the average SST of every year
        
        Params:
            divisions_per_year - int, optional. See documentation of 'dpy' in
                __init__()
        
        Return a 3D numpy array of floats. The first axis represents time, with
        a 1-year interval. At each point along this axis is a 2D numpy array
        of floats representing the yearly average of every longtitude and 
        latitude for a single year
        """
        year_maps = []
        for i in range(0, len(self.map), divisions_per_year):
            months = self.map[i:i+divisions_per_year]
            year_maps.append(np.mean(months, axis=0))
        return np.asarray(year_maps)
    
    def _get_yearly_average_maps_smooth(self, averages):
        """Return a smoothed version of the yearly average maps
        
        Return a 3D numpy array of floats representing the smoothed map. Format
        is identical to _get_yearly_average_maps()
        """
        av = averages
        for _ in range(self._smooth//2):
            # the new values are stored here to prevent bias based on 
            # order in the list
            temp_av = []
            for j in range(1, len(av)-1):
                temp_av.append(.5*av[j]+.25*(av[j-1]+av[j+1]))
            # replace the old PDI list with the new one
            av = temp_av
        return np.asarray(av)
    
    def _get_hurricane_season_maps(self, divisions_per_year=12):
        """Return maps of the average SST of every year during the hurricane
        season
        
        Params:
            divisions_per_year - int, optional. See documentation of 'dpy' in
                __init__()
              **CURRENTLY NOT IMPLEMENTED FOR VALUES NOT EQUAL TO 12
        
        Return a 3D numpy array of floats. The first axis represents time, with
        a 1-year interval. At each point along this axis is a 2D numpy array
        of floats representing the hurricane season average of every longtitude 
        and latitude for a single year
        """
        if divisions_per_year != 12:
            print("_get_yearly_average_maps() is only implemented for" +
                "12 divisions per year.")
            return
        year_maps = []
        for i in range(0, len(self.map), divisions_per_year):
            # months is modified to only include hurricane season
            months = self.map[i+6:i+divisions_per_year-1]
            year_maps.append(np.mean(months, axis=0))
        return np.asarray(year_maps)

    def _get_hurricane_season_maps_smooth(self, dpy):
        """Return a smoothed version of the hurricane season maps
        
        Return a 3D numpy array of floats representing the smoothed map.
        """
        av = self._get_hurricane_season_maps(dpy)
        for _ in range(self._smooth//2):
            # the new values are stored here to prevent bias based on 
            # order in the list
            temp_av = []
            for j in range(1, len(av)-1):
                temp_av.append(.5*av[j]+.25*(av[j-1]+av[j+1]))
            # replace the old PDI list with the new one
            av = temp_av
        return np.asarray(av)

    def _get_mean_of_map(self, input_map):
        """Return the mean SST from a given map, taking into account
        the Earth's curvature and the presence of land.
        
        Params:
            input_map- 2D numpy array of floats. This represents a global 
                map of SST values in a given year.
        
        Return a float representing the mean SST value in the map
        """
        denominator = 0
        numerator = 0
        for i in range(input_map.shape[0]):
            weight = np.cos(radians(self.lat_axis[i]))
            s = np.sum(input_map[i])
            if s > 0 or s < 0:
                numerator += s * weight
                denominator += np.count_nonzero(input_map[i]) * weight
        return numerator/denominator
    
    def _get_MDR_indeces(self):
        """Return the indeces approximately bounding the MDR
        
        Return a tuple of 4 ints representing indeces that are approximately
        equivalent to the lines (280°E, 340°E, 10°N, 25°N), in that order
        """
        #the MDR is 10°–25°N, 80°–20°W (280°-340°E)
        lon1 = 0    #280°E
        lon2 = 0    #340°E
        lat1 = 0    #10°N
        lat2 = 0    #25°N

        # these lambdas ensure that the right values will be gotten
        # if lat axis is in degrees south
        expr25 = lambda i : self.lat_axis[i] <= 10
        expr10 = lambda i : self.lat_axis[i] <= 25
        if self.lat_axis[0] < 0:
            expr25 = lambda i : self.lat_axis[i] >= 25
            expr10 = lambda i : self.lat_axis[i] >= 10

        #find indexes approximately bounding to the requested area
        for i in range(len(self.lon_axis)):
            if self.lon_axis[i] >= 280:
                lon1 = i
                break
        for i in range(len(self.lon_axis)):
            if self.lon_axis[i] >= 340:
                lon2 = i
                break
        for i in range(len(self.lat_axis)):
            if expr10(i):
                lat1 = i
                break
        for i in range(len(self.lat_axis)):
            if expr25(i):
                lat2 = i
                break
        return_value = (lon1, lon2, lat1, lat2)
        # print(return_value, self.lat_axis)
        return return_value
    
    def _get_mean_MDR_SST(self, input_map):
        """Get the yearly average SST values in the MDR for every year in
        the dataset

        Params:
            input_map- The map whose MDR data should be gotten. Recommended
                input is self.yearly_map or self.hurseason_map
        
        Return a list of floats representing the SST across the years
        """
        MDR_indeces = self._get_MDR_indeces()
        yearly_sst = []
        for i in range(len(input_map)):
            # get the MDR during the given year
            MDR = input_map[i, MDR_indeces[2]:MDR_indeces[3],
                            MDR_indeces[0]:MDR_indeces[1]]
            
            #calculate the mean with weighting based on cruvature
            n = 0
            d = 0
            _deb_loops = 0 #debugging
            _deb_blanks = 0 #debugging
            for j in range(MDR_indeces[2], MDR_indeces[3]):
                _deb_loops += 1 #debugging
                weight = np.cos(radians(self.lat_axis[j]))
                s = np.sum(MDR[j - MDR_indeces[2]])
                if s > 0 or s < 0:
                    _deb_blanks += 1 #debugging
                    n += s * weight
                    d += np.count_nonzero(MDR[j - MDR_indeces[2]]) * weight
            if d == 0:
                #debugging
                print("Division by zero at _get_mean_MDR_SST()", d, n, 
                        _deb_loops, _deb_blanks, i, len(input_map), 
                        MDR_indeces)
            mean = n/d
            
            #append that year's mean
            yearly_sst.append(mean)
        return yearly_sst
    
    def _get_mean_MDR_SST_smooth(self, input_map):
        """Get the yearly average SST values in the MDR for every year in
        the dataset, smoothed

        Params:
            input_map- The map whose MDR data should be gotten. Recommended
                input is self.yearly_map or self.hurseason_map
        
        Return a list of floats representing the SST across the years
        """
        sst = self._get_mean_MDR_SST(input_map)
        for _ in range(self._smooth//2):
            # the new values are stored here to prevent bias based on 
            # order in the list
            temp_sst = []
            for j in range(1, len(sst)-1):
                temp_sst.append(.5*sst[j] + .25*(sst[j-1] + sst[j+1]))
            # replace the old PDI list with the new one
            sst = temp_sst
        return sst

    ############################################################# public methods
    
    def get_year_anomaly_map(self, year):
        """Return a map of the anomaly between the SST of every point on map
        during the given year and its time-mean SST.
        
        Param:
            year- int. The year whose anomaly to return
        
        Return a 2D numpy array of floats representing a map of the anomaly
        """
        if not self._initialized:
            print("Class not initialized! Initialize class by calling",
                  "initialize() or initialize_SST_from_RCP()!")
        return self.yearly_map[year] - self.time_mean
    
    def get_year_anomaly_map_smooth(self, year):
        """Return a map of the anomaly between the SST of every point on map
        during the given year, with smoothing applied, and its time-mean 
        SST.
        
        Param:
            year- int. The year whose anomaly to return. This year should be
                 at least two years away from the edge of the unsmoothed dataset.
        
        Return a 2D numpy array of floats representing a map of the anomaly.
        """
        if not self._initialized:
            print("Class not initialized! Initialize class by calling",
                  "initialize() or initialize_SST_from_RCP()!")
        return self.smooth_map[year] - self.time_mean
    
    def contour_anomaly_progression(self, title, path):
        if not self._initialized:
            print("Class not initialized! Initialize class by calling",
                  "initialize() or initialize_SST_from_RCP()!")
            return
        anomaly = self.get_year_anomaly_map_smooth(len(self.smooth_map)-1)
        cmap = mpl.cm.coolwarm
        norm = mpl.colors.Normalize(vmin=-2, vmax=2)
        cb_setter_array = np.asarray([np.arange(-3,3,.01), np.arange(-3,3,.01)])
        cb_setter = self.ax1.contourf(cb_setter_array, 99, norm=norm, cmap=cmap)
        clb = self.figure.colorbar(cb_setter, ax=self.ax1)
        clb.set_label('Anomaly (°C)')
        for i in range(0, len(self.smooth_map)):
            print("Saved map of", i+self.start_year+(self._smooth//2), end="\r")
            anomaly = self.get_year_anomaly_map_smooth(i)
            #automatically mention in title whether smoothing is applied or not
            smooth_str = ""
            if self._smooth > 1:
                smooth_str = " (" + str(self._smooth) + "-year-smoothed)"
            self.ax1.clear()
            self.ax1.set_title(title + smooth_str + "\n" + 
                        str(self.start_year + (self._smooth//2) + i))
            self.ax1.contourf(self.lon_axis, self.lat_axis, 
                        anomaly, 100, norm=norm, cmap=cmap)
            self.plot_SST_averages(self.start_year+(self._smooth//2)+i)
            self.ax1.get_xaxis().set_ticks([])
            self.ax1.get_yaxis().set_ticks([])
            self.figure.show()
            plt.savefig(path + "/map-" + str(i).rjust(5, '0') + ".png")
                
    def plot_SST_averages(self, end_year):
        #set up plot
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 10))

        """Plot the average SST values up to the given year"""
        if not self._initialized:
            print("Class not initialized! Initialize class by calling",
                  "initialize() or initialize_SST_from_RCP()!")
            return
        unsmoothed_mean = self._unsmoothed_mean[
                self._smooth//2:end_year-self.start_year]
        smoothed_mean = self._smoothed_mean[:
            end_year-self.start_year-(self._smooth//2)]
        year_range = range(self.start_year + (self._smooth//2), end_year)
        self.ax2.clear()
        self.ax2.set_title("Global SST")
        self.ax2.set_xlabel("Year")
        self.ax2.set_ylabel("Anomaly (°C)")
        self.ax2.hlines(y=0, xmin=self.start_year+(self._smooth//2),
                        xmax=end_year+(self._smooth//2)
                                , label="time mean")
        self.ax2.plot(year_range, unsmoothed_mean, label="Average SST")
        self.ax2.plot(year_range, smoothed_mean, label="Average SST ("
                      + str(self._smooth) + "-year-smoothed)")
        self.ax2.legend(loc='upper left')
        
        
    def save_MDR_SST_hurricane_season_data(self, name="sst_MDR_hurseason"):
        """
        Save the average MDR SST during hurricane season and smoothed 
        average MDR SST during hurricane season, for the entire record 
        with the correct year range. More information regarding format 
        found in section 'Output' of __doc__ at top of file.

        Params:
            name- str, optional. The name that the saved file will be given.
                Defaults to "sst_MDR_hurseason"
        """
        if not self._initialized:
            print("Class not initialized! Initialize class by calling",
                  "initialize() or initialize_SST_from_RCP()!")
            return
        
        #get the array of data
        unsmoothed_mean = np.asarray(
                self._get_mean_MDR_SST(self.hurseason_map))
        #get the array of years it corrosponds to
        unsmoothed_mean_range = range(self.start_year, 
                    self.start_year + len(self.yearly_map))
        #bundle the two together
        raw = np.asarray([unsmoothed_mean_range, unsmoothed_mean])
        
        #repeat the process for the smoothed SST data
        smoothed_mean = np.asarray(
                self._get_mean_MDR_SST_smooth(self.hurseason_map))
        smoothed_mean_range = range(self.start_year + (self._smooth//2), 
                    self.start_year + len(smoothed_mean)
                                + (self._smooth//2))
        smooth = np.asarray([smoothed_mean_range, smoothed_mean])

        #save it
        np.save("Output/"+name+"_smooth.npy", smooth)
        np.save("Output/"+name+".npy", raw)


###############################################################################
# End of defining class read_analyze_plot_SST 
###############################################################################


###############################################################################
# Main program:
###############################################################################
s_RCP = read_analyze_plot_SST()
s_OBS = read_analyze_plot_SST()
s_RCP.initialize_SST_from_RCP("../../../Data-for-teaching-staff/Hurricanes/RCP85/",
                              "tos", "rlat", "rlon", 2006)
s_OBS.initialize("../../../Data-for-teaching-staff/Hurricanes/sst.mon.mean.nc",
                 "sst", "lat", "lon", 1850)
# s_OBS.contour_anomaly_progression("SST Anomaly", "Output/SST_OBS")
# s_RCP.contour_anomaly_progression("SST Anomaly", "Output/SST_RCP")

s_RCP.save_MDR_SST_hurricane_season_data("sst_rcp85_MDR_hurseason")

s_OBS.save_MDR_SST_hurricane_season_data()

fig3, ax3 = plt.subplots()
ax3.set_title("MDR Average Hurricane Season SST")
ax3.set_xlabel("Year")
ax3.set_ylabel("Temperature (°C)")
x = np.load("./Output/sst_MDR_hurseason.npy")
ax3.plot(x[0], x[1], label="MDR average SST (Hurricane season, observed)")
x = np.load("./Output/sst_rcp85_MDR_hurseason_smooth.npy")
ax3.plot(x[0], x[1]-273.15, label="MDR average SST (Hurricane season, projected)")

plt.show()