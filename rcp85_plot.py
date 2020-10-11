"""
This script provides a class that organizes and analyzes evaporation,
precipitation, and soil moisture data for a given Scenario. A "Scenario"
may be a observed data, a specific projection, etc.
Using this class, it creates a number of plots comparing those variables
between two climate models in the Sahel and in southwestern United States,
and can create frames for an animation comparing the two models.

Required files:
    Each scenario initialized requires three separate datasets:
    evaporation, precipitation, and soil moisture. Each of these datasets can
    be either a single .nc file or a directory of .nc files which are
    direct chronological continuations of each other.
    All .nc files must contain monthly data, and all data within a
    single scenario is assumed to begin in the same year.
    The files used are from:
        GFDL-CM3 rcp85 r1i1p1
        MIROC-ESM-CHEM rcp85 r1i1p1

Output:
    This script creates the following .npy files:

    sahel_miroc_evaporation.npy
        A time series of the projected geographic average of the
        annual average evaporation in the Sahel, in units of m/y, 
        according to the MIROC-ESM-CHEM climate model (predicts wet future).
        
        2D, 2-row numpy array of floats. The first row (array[0]) contains
        an array of years, and the second row (array[1]) contains an array of
        values that correspond to those years.
        visual representation of array format (fictional values):

            [[2006, 2007, 2008, 2009],
            [  .1,   .4,   .3,   .6]]

    sahel_miroc_moisture.npy
        Similar to sahel_miroc_evaporation, but contains soil
        moisture data in units of kg/m^2
        
    sahel_miroc_precipitaion.npy
        Similar to sahel_miroc_evaporation, but contains 
        precipitation data in units of m/y
        
    sahel_gfdl_evaporation.npy,
    sahel_gfdl_moisture.npy, and
    sahel_gfdl_precipitation.npy
        Similar to their miroc counterparts, but contain data from
        the GFDL CM3 climate model (predicts an unchanged Sahel)

    sw_miroc_evaporation.npy,
    sw_miroc_moisture.npy,
    sw_miroc_precipitation.npy,	
    sw_gfdl_evaporation.npy,
    sw_gfdl_moisture.npy,
    sw_gfdl_precipitation.npy
        Similar to their sahel counterparts, but contain data for
        the southwestern US. Both climate models are in agreement
        that this region will dry.

Usage Instructions:
    Each of the variables GFDL_E, GFDL_P, GFDL_M, MIROC_E, MIROC_P, and 
    MIROC_M, found directly below this __doc__string, must be given a
    tuple, representing a single variable, in the format 
    (path, data, lat, lon, factor), where
        path- is a str containing the filepath to a single .nc file
            or a directory of .nc files containing the data
            (e.g. "./Data/dir_containing_files/", "./Data/dir/data.nc")
        data- is a str representing the name of the variable within the
            .nc file(s) containing the gridded data (e.g. "pr")
        lat- is a str representing the name of the variable within the .nc
            file(s) containing the latitude axis that labels the indices
            of data (e.g "lat")
        lon- is a str representing the name of the variable within the .nc
            file(s) containing the longtitude axis that labels the indices
            of data (e.g "lon")
        factor- is a float representing a factor by which all the values
            in data will be multiplied, and is used for unit conversions
            (e.g. factor=31557.6 will convert kg/m2s to m/y)
    -GFDL_E should contain the tuple for evaporation in the GFDL model
    -GFDL_P should contain the tuple for precipitation in the GFDL model
    -GFDL_M should contain the tuple for moisture in the GFDL model
    -MIROC_E should contain the tuple for evaporation in the MIROC model
    -MIROC_P should contain the tuple for precipitation in the MIROC model
    -MIROC_M should contain the tuple for moisture in the MIROC model
    Additionally, the variables GFDL_START and MIROC_START should be
    given the first year of the simulation.

    On being run, this script will plot each variable for both models 
    and for both the Sahel and southwestern United States, creating a 
    total of 6 plots. It will also generate the .npy files described
    in the Output section of this __doc__ string.

Yonathan Vardi 2019-07-19
"""
########################################################################
GFDL_E = ("../../../Data-for-teaching-staff/Droughts/rcp/gfdl_evaporation", 
          "evspsbl", "lat", "lon", 31557.6)
GFDL_P = ("../../../Data-for-teaching-staff/Droughts/rcp/gfdl_precipitation", 
          "pr", "lat", "lon", 31557.6)
GFDL_M = ("../../../Data-for-teaching-staff/Droughts/rcp/gfdl_soilmoisture",
          "mrso", "lat", "lon", 1)
# GFDL_START = 2006
GFDL_START = 1860

# MIROC_E = ("../../../Data-for-teaching-staff/Droughts/rcp/evspsbl_Amon_MIROC-ESM-CHEM_rcp85_r1i1p1_200601-210012.nc", 
#            "evspsbl", "lat", "lon", 31557.6)
# MIROC_P = ("../../../Data-for-teaching-staff/Droughts/rcp/pr_Amon_MIROC-ESM-CHEM_rcp85_r1i1p1_200601-210012.nc",
#            "pr", "lat", "lon", 31557.6)
# MIROC_M = ("../../../Data-for-teaching-staff/Droughts/rcp/mrso_Lmon_MIROC-ESM-CHEM_rcp85_r1i1p1_200601-210012.nc",
#            "mrso", "lat", "lon", 1)
MIROC_E = ("../../../Data-for-teaching-staff/Droughts/rcp/miroc_e", 
           "evspsbl", "lat", "lon", 31557.6)
MIROC_P = ("../../../Data-for-teaching-staff/Droughts/rcp/miroc_p",
           "pr", "lat", "lon", 31557.6)
MIROC_M = ("../../../Data-for-teaching-staff/Droughts/rcp/miroc_m",
           "mrso", "lat", "lon", 1)
# MIROC_START = 2006
MIROC_START = 1850
########################################################################

import numpy as np
from math import radians
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap


# ELi: added smoothing of mode series:
def smooth_time_series(data):
    """
    Apply 2 passes of 1-2-1 filter to input data.
    Inputs: data[0,:] - time axis
            data[1,:] - the data timeseries
    Output: data_smoothed: twice smoothed, same structure, except two less in length
            than input, as first and last cannot be calculated.
    """
    # two passes of smoothing:
    data_smooth = np.copy(data)
    for ipass in range(0,2):
        for j in range(1, len(data[0,:])-1):
            data_smooth[1,j]=.25*data_smooth[1,j-1] + 0.5*data_smooth[1,j] + 0.25*data_smooth[1,j+1]
    # return smoohted data without first and last values:
    return data_smooth[:,2:len(data[0,:])-2]


########################################################################
# SCENARIO CLASS
########################################################################

class Scenario:
    """
    A class representing an observed or projected scenario's 
    evaporation, precipitation, and soil moisture data.

    Public Members:

        evap- tuple of arrays representing evaporation data. The first
            element of the tuple is a 3D numpy array of floats 
            representing a time series of evaporation maps. The second 
            and third elements of the tuple is a 1D numpy array of 
            floats representing the latitudes and longtitudes that 
            correspond to each index of the map.

        precip- tuple of arrays representing precipitation data. The 
            first element of the tuple is a 3D numpy array of floats 
            representing a time series of precipitation maps. The second 
            and third elements of the tuple is a 1D numpy array of 
            floats representing the latitudes and longtitudes that 
            correspond to each index of the map.

        moisture- tuple of arrays representing soil moisture data. The 
            first element of the tuple is a 3D numpy array of floats 
            representing a time series of soil moisture maps. The second 
            and third elements of the tuple is a 1D numpy array of 
            floats representing the latitudes and longtitudes that 
            correspond to each index of the map.

        start_year- int. Represents the year for which the data begins.
            assumed to be the same between evap, precip, and moisture.

    Public Methods:
        contour() - Draw a contour plot of a given dataset at a given time
        get_sahel_slice() - Return a subsection of a given dataset which
            represents the Sahel region.
        get_US_SW_slice() - Return a subsection of a given dataset which
            represents the Southwest US region.
        get_average_value_timeseries_of_region() - Return a timeseries
            containing the average value from a given region
        get_average_value_timeseries_of_region_s() - Similar to
            get_average_value_timeseries_of_region(), but returns a
            smoothed timeseries.
        get_average_map_of_region() - Return a map of time-mean values
            of a region.
        absolute_timeseries_to_anomaly() - Convert a time series of 
            absolute values to a time series of anomalies
        smooth_and_save_region_value_timeseries() - [Eli: first apply
            two passes of 1-2-1 filter and then] save to file a 
            timeseries of average values from a given region.
    """

    #######################################################] INITIALIZER

    def __init__(self, evap, precip, moisture, start_year):
        """
        Initialize a Scenario

        Parameter format:
            The first three parameters are 5-element tuples of format 
            (path, data_name, lat_name, lon_name, conversion_factor)
                path- str. A path that can either lead to a specific
                    .nc file (e.g. "./dir/file.nc") or to a directory
                    or .nc files (e.g. "./dir/subdir/"). If a directory is 
                    given, all .nc files in it will be opened, and files
                    are assumed to contain data ordered chronologically 
                    by their name. That is, file "b.nc" is assumed to be 
                    a direct continuation of the data in "a.nc". Files 
                    are also assumed to have identically named 
                    variables.
                data_name- str. The name of the variable within the .nc
                    file(s) containing the gridded data.
                lat_name- str. The name of the variable within the .nc 
                    file(s) containing the latitude axis.
                lon_name- str. The name of the variable within the .nc 
                    file(s) containing the longtitude axis.
                conversion_factor- int. This represents an amount by
                    which all the data will be multiplied. Used for
                    unit conversions.

        Params:
            evap- tuple formatted as above. Represents evaporation data.
            precip- tuple formatted as above. Represents precipitation data.
            moisture- tuple formatted as above. Represents soil moisture data.
            start_year- int. Represents the year for which the data begins.
                Assumed to be the same for all data sets.
        """
        paths = [evap[0], precip[0], moisture[0]]
        data_names = [evap[1], precip[1], moisture[1]]
        lat_names = [evap[2], precip[2], moisture[2]]
        lon_names = [evap[3], precip[3], moisture[3]]
        factors = [evap[4], precip[4], moisture[4]]

        loaded = []
        for i in range(len(paths)):
            if paths[i].endswith(".nc"):
                # if path leads directly to a file, load from file
                loaded.append(self._load_nc_indiv(paths[i], data_names[i],
                                lat_names[i], lon_names[i], factors[i]))
            else:
                # else, it must lead to a directory. Load from directory
                loaded.append(self._load_nc_dir(paths[i], data_names[i],
                                lat_names[i], lon_names[i], factors[i]))

        self.mon_evap = (loaded[0][0], loaded[0][1], loaded[0][2])
        self.mon_precip = (loaded[1][0], loaded[1][1], loaded[1][2])
        self.mon_moisture = (loaded[2][0], loaded[2][1], loaded[2][2])
        self.evap = (self._convert_monthly_map_to_yearly_average(loaded[0][0]), 
                     loaded[0][1], loaded[0][2])
        self.precip = (self._convert_monthly_map_to_yearly_average(loaded[1][0]),
                       loaded[1][1], loaded[1][2])
        self.moisture = (self._convert_monthly_map_to_yearly_average(loaded[2][0]),
                         loaded[2][1], loaded[2][2])

        self.start_year = start_year

    ######################################################] FILE LOADING

    def _load_nc_indiv(self, path, data_name, lat_name, lon_name, f):
        """
        loads a single .nc file given a filepath to it

        Params:
            path- str. The path to the .nc file (e.g ../dir/example.nc)
            data_name- str. The name of the variable within the .nc file
                containing the gridded data.
            lat_name- str. The name of the variable within the .nc file
                containing the latitude axis.
            lon_name- str. The name of the variable within the .nc file
                containing the longtitude axis.
            f- int. This represents a factor by which all the data will 
                be multiplied. Used for unit conversions.

        Return a tuple in format (data, lat, lon), where data is the
            loaded gridded data, lat is the loaded latitude axis, and
            lon is the loaded longtitude axis.
        """
        ncfile = Dataset(path, 'r')
        data = ncfile.variables[data_name][:]*f
        lat = ncfile.variables[lat_name][:]
        lon = ncfile.variables[lon_name][:]
        return (data, lat, lon)

    def _load_nc_dir(self, path, data_name, lat_name, lon_name, f):
        """
        loads all .nc files in a given directory

        Params:
            path- str. The path to the directory (e.g ../dir/subdir/)
            data_name- str. The name of the variable within the .nc files
                containing the gridded data. This is assumed to be
                the same across all files.
            lat_name- str. The name of the variable within the .nc files
                containing the latitude axis. This is assumed to be
                the same across all files.
            lon_name- str. The name of the variable within the .nc files
                containing the longtitude axis. This is assumed to be
                the same across all files.
            f- int. This represents a factor by which all the data will 
                be multiplied. Used for unit conversions.

        Return a tuple in format (data, lat, lon), where data is the
            total concatenated gridded data, data, lat is the
            latitude axis (assumed to be identical for all .nc files in
            directory), and lon is the loaded longtitude axis (assumed 
            to be identical for all .nc files in directory).
        """
        #Add a slash to the end if the user didn't add one
        #This makes both "../dir/" and "../dir" work
        if not path.endswith("/"):
            path += "/"
        
        #begin by making a list of all .nc files
        file_list=[]
        for i in os.listdir(path):
            file_list.append(os.fsdecode(i))
        file_list.sort()

        filenames = []
        for file in file_list:
            if file.endswith(".nc"):
                filenames.append(path + file)

        #ensure there were files in directory
        if len(filenames) == 0:
            print("No .nc files in directory! Initialization failed!")
            return

        #then get the values from the first file (this is the only file
        # from which lat_axis and lon_axis are gotten)
        ncfile = Dataset(filenames[0], 'r')
        lat = ncfile.variables[lat_name][:]
        lon = ncfile.variables[lon_name][:]
        data = ncfile.variables[data_name][:]*f

        #then add the data from the rest of the files
        for i in range(1, len(filenames)):
            ncfile = Dataset(filenames[i], 'r')
            print("concatenating " + filenames[i])
            data = np.ma.concatenate((data,
                          ncfile.variables[data_name][:]*f))

        return (data, lat, lon)

    def _convert_monthly_map_to_yearly_average(self, input_map):
        """Convert a time series of maps containing monthly values to a 
        time series of maps containing yearly values by averaging every 
        12 months into one yearly average.

        Params:
            input_map- a 3D numpy array of floats representing a time
                series of monthly values
        
        Return a 3D numpy array of floats representing a time series of
            yearly average values
        """
        year_maps = []
        for i in range(0, len(input_map), 12):
            months = input_map[i:i+12]
            year_maps.append(np.ma.mean(months, axis=0))
        return np.ma.asarray(year_maps)

    def _convert_monthly_map_to_yearly_total(self, input_map):
        """Convert a time series of maps containing monthly values to a 
        time series of maps containing yearly values by combining every
        12 months into one yearly total.

        Params:
            input_map- a 3D numpy array of floats representing a time
                series of monthly values
        
        Return a 3D numpy array of floats representing a time series of
            yearly total values
        """
        year_maps = []
        for i in range(0, len(input_map), 12):
            months = input_map[i:i+12]
            year_maps.append(np.ma.sum(months, axis=0))
        return np.ma.asarray(year_maps)

    #################################################] REGIONAL ANALYSIS

    def _get_regional_slice(self, dataset, dataset_years,
                            minlat, maxlat, minlon, maxlon):
        """Return a slice of the given dataset's map roughly representing
        the given geographic region, as defined by given binding coordinates.

        Params:
            dataset- One of the three datasets stored within this
                class (evap, precip, moisture)

            dataset_years- int or slice object. The year index(ces) for 
                which to return the region slice. (e.g. 6 will return the
                values for the region in year 6, slice(6, 10) will return 
                the values for years 6 7 8 9)

            minlat- int. The minimum bounding latitude. Must be smaller
                than minlat.
            
            maxlat- int. The maximum bounding latitude. Must be larger
                than maxlat.

            minlon- int. The minimum bounding longitude. Must be smaller
                than maxlon.
            
            maxlon- int. The maximum bounding longitude. Must be larger
                than minlon.

        Return a tuple of format (slice, lat, lon), where slice is
            a 2D numpy array representing the region, or a 3D array if
            dataset_years was a slice. lat and lon are slices of the
            latitude and longtitude axes that correspond to slice.
            lon is modified such that the first value is smaller than
            the last value.
        """
        # get the approximate indeces for the bounding area
        lat10 = min(enumerate(dataset[1]), key=(lambda i: abs(i[1] - minlat)))[0]
        lat20 = min(enumerate(dataset[1]), key=(lambda i: abs(i[1] - maxlat)))[0]
        lon40 = min(enumerate(dataset[2]), key=(lambda i: abs(i[1] - minlon)))[0]
        lon345 = min(enumerate(dataset[2]), key=(lambda i: abs(i[1] - maxlon)))[0]
        #get lat and lon ranges
        lat_range = np.arange(lat10, lat20+1)
        #since depending on how the longtitude axis is arranged 15°W may
        #or may not come after 40°E, a list of indeces going east from
        #15°W to 40°E is compiled and used instead of a slice
        lon_range = None
        if lon345 > lon40:
            lon_range = np.arange(lon345-len(dataset[2]), lon40+1)
        else:
            lon_range = np.arange(lon345, lon40+1)
        #get the slice
        sahel_slice = dataset[0][dataset_years, lat10:lat20+1, lon_range]
        #the interval between each point of the longtitude axis
        d_lon = dataset[2][1]-dataset[2][0]
        return (sahel_slice, dataset[1][lat_range], lon_range*d_lon)

    def get_sahel_slice(self, dataset, dataset_years, buffer=0.0):
        """Return a slice of the given dataset's map roughly representing
        the Sahel (defined as the region bound by 10°N, 15°W, 20°N, 40°E)

        Params:
            dataset- One of the three datasets stored within this
                class (evap, precip, moisture)

            dataset_years- int or slice object. The year index(ces) for 
                which to return the region slice. (e.g. 6 will return the
                values for the region in year 6, slice(6, 10) will return 
                the values for years 6 7 8 9)

            buffer- float. The number of degrees to buffer the region by.
                (for example, for a value of 10 the region returned
                will extend 10° in every direction beyond the Sahel)

        Return a tuple of format (slice, lat, lon), where slice is
            a 2D numpy array representing the Sahel, or a 3D array if
            dataset_years was a slice. lat and lon are slices of the
            latitude and longtitude axes that correspond to slice.
            lon is modified such that the first value is smaller than
            the last value.
        """
        return self._get_regional_slice(dataset, dataset_years, 
                10 - buffer, 20 + buffer, 40 + buffer, 345 - buffer)

    def get_US_SW_slice(self, dataset, dataset_years, buffer=0.0):
        """Return a slice of the given dataset's map roughly representing
        the US Southwest (defined as the region bound by 125°W-105°W, 32°N-41°N)

        Params:
            dataset- One of the three datasets stored within this
                class (evap, precip, moisture)

            dataset_years- int or slice object. The year index(ces) for 
                which to return the region slice. (e.g. 6 will return the
                values for the region in year 6, slice(6, 10) will return 
                the values for years 6 7 8 9)

            buffer- float. The number of degrees to buffer the region by.
                (for example, for a value of 10 the region returned
                will extend 10° in every direction beyond the Sahel)

        Return a tuple of format (slice, lat, lon), where slice is
            a 2D numpy array representing the US SW, or a 3D array if
            dataset_years was a slice. lat and lon are slices of the
            latitude and longtitude axes that correspond to slice.
            lon is modified such that the first value is smaller than
            the last value.
        """
        return self._get_regional_slice(dataset, dataset_years, 
                32 - buffer, 41 + buffer, 255 + buffer, 235 - buffer)

    def get_average_value_timeseries_of_region(self, region):
        """Return a timeseries of average valuse of a regional dataset 
        containing any data, taking into account the earth's curvature.

        Params:
            region- a value of the same return type as 
                self._get_regional_slice(). Represents the region whose 
                average value is being returned. Note that both both 
                2D and 3D maps are accepted.
        
        Return a 1D numpy array representing a time series of 
            average values. If the region parameter contained a 2D map,
            the array will be of length 1
        """
        #first, standardize the format of the array:
        # replace masks with "0"
        input_map = np.ma.filled(region[0], 0)
        # convert it to 3D if it is 2D
        if len(input_map.shape) == 2:
            input_map = input_map[np.newaxis, ...]
        
        lat_axis = region[1]
        averages = []
        for time in range(input_map.shape[0]):
            denominator = 0
            numerator = 0
            for i in range(input_map.shape[1]):
                weight = np.cos(radians(lat_axis[i]))
                s = np.sum(input_map[time, i])
                if s > 0 or s < 0:
                    numerator += s * weight
                    denominator += np.count_nonzero(input_map[time, i]) * weight
            averages.append(numerator/denominator)
        return np.asarray(averages)

    def get_average_value_timeseries_of_region_s(self, region, s=5):
        """Return a smoothed timeseries of average valuse of a regional 
        dataset containing any data, taking into account the earth's 
        curvature.

        Params:
            region- a value of the same return type as 
                self._get_regional_slice(). Represents the region whose 
                average value is being returned.
            s- int, optional. Represents the number of years over which 
                smoothing is done. Defaults to 5.
        
        Return a 1D numpy array representing a smoothed time series of 
            average values.
        """
        series = self.get_average_value_timeseries_of_region(region)
        for _ in range(s//2):
            # the new values are stored here to prevent bias based on 
            # order in the list
            temp = []
            for j in range(1, len(series)-1):
                temp.append(.5*series[j]+.25*(series[j-1]+series[j+1]))
            # replace the old PDI list with the new one
            series = temp
        return np.ma.asarray(series)

    def get_average_map_of_region(self, region, time_length=None, begin_time=0):
        """Return a map of the average values for a region over a length
        of time.

        Params:
            region- a value of the same return type as 
                self._get_regional_slice(). Represents the region whose 
                average value is being returned. While both 2D and 3D
                maps are accepted, this function only makes sense with a
                3D map.

            time_length- int, optional. The length of time, starting from 
                begin_time, over which to compute the average. If None,
                computes the average until the last time point. Defaults
                to none.

            begin_time- int, optional. The index of the first time point
                in time to include in the average calculation. 
                Defaults to 0.

        Return a 2D numpy array of floats representing a map of average
            values for every longtitude and latitude coordinate pair in
            the given region.
        """
        #first, standardize the format of the array:
        # replace masks with "0"
        input_map = np.ma.filled(region[0], 0)
        # convert it to 3D if it is 2D
        if len(input_map.shape) == 2:
            input_map = input_map[np.newaxis, ...]

        if time_length == None:
            time_length = len(input_map)
        input_map = input_map[begin_time:begin_time+time_length]

        return np.mean(input_map, axis=0)

    def absolute_timeseries_to_anomaly(self, series, bl_e=50, bl_s=0):
        """Convert a time series of absolute values to a time series of
        anomalies

        Params:
            series- 1D numpy array. The time series being converted.
            bl_e- int. Represents the last index of the series to be
                counted in the baseline mean calculation.
            bl_s- int. Represents the first index of the series to be
                counted in the baseline mean calculation.

        Return a 1D numpy array representing a time series of anomalies 
            from the period between bl_s and bl_e
        """
        mean = np.mean(series[bl_s:bl_e+1])
        anomaly_series = (series-mean)
        return anomaly_series

    #############################################################] OTHER

    def contour(self, dataset, time_index, ax):
        """Draw a contour plot of the given dataset at the given time
        
        Params:
            dataset- One of the three datasets stored within this
                class (evap, precip, moisture)

            time_index- The index of the dataset along the time axis at
                which to draw the contour

            ax- The matplotlib axis on which to draw the plot
        """
        ax.contourf(dataset[2], dataset[1], dataset[0][time_index], levels=99)

    def smooth_and_save_region_value_timeseries(self, region, path):
        """[Eli: first apply two passes of 1-2-1 filter and then] Save a
        timeseries of average geographic values of a region to file.

        Params:
            region- a value of the same return type as 
                self._get_regional_slice(). Represents the region whose 
                average value is being returned.

            path- str. Represents the path and filename of the output
                .npy file, e.g. "./Output/file.npy"
        """
        values = self.get_average_value_timeseries_of_region(region)
        years = np.arange(self.start_year, self.start_year+len(values))
        series = np.asarray([years, values])

        # Eli: added smoothing of data:
        series1=smooth_time_series(series)
        np.save(path, series1)

########################################################################
# MAIN PROGRAM
########################################################################

#load the scenarios
plt.ioff()
gfdl_data = Scenario(GFDL_E, GFDL_P, GFDL_M, GFDL_START)
miroc_data = Scenario(MIROC_E, MIROC_P, MIROC_M, MIROC_START)

######################################################## save timeseries
# This section of the program saves the timeseries of 
# average values for every variable, both models, in both 
# regions.

#-------------------------------= save the timeseries data for the sahel
w_moi = miroc_data.get_sahel_slice(miroc_data.moisture, slice(0, len(miroc_data.moisture[0])))
d_moi = gfdl_data.get_sahel_slice(gfdl_data.moisture, slice(0, len(gfdl_data.moisture[0])))
w_pre = miroc_data.get_sahel_slice(miroc_data.precip, slice(0, len(miroc_data.precip[0])))
d_pre = gfdl_data.get_sahel_slice(gfdl_data.precip, slice(0, len(gfdl_data.precip[0])))
w_eva = miroc_data.get_sahel_slice(miroc_data.evap, slice(0, len(miroc_data.evap[0])))
d_eva = gfdl_data.get_sahel_slice(gfdl_data.evap, slice(0, len(gfdl_data.evap[0])))

miroc_data.smooth_and_save_region_value_timeseries(w_moi, "./Output/sahel_miroc_moisture.npy")
miroc_data.smooth_and_save_region_value_timeseries(w_pre, "./Output/sahel_miroc_precipitation.npy")
miroc_data.smooth_and_save_region_value_timeseries(w_eva, "./Output/sahel_miroc_evaporation.npy")

gfdl_data.smooth_and_save_region_value_timeseries(d_moi, "./Output/sahel_gfdl_moisture.npy")
gfdl_data.smooth_and_save_region_value_timeseries(d_pre, "./Output/sahel_gfdl_precipitation.npy")
gfdl_data.smooth_and_save_region_value_timeseries(d_eva, "./Output/sahel_gfdl_evaporation.npy")

#-------------------------------= save the timeseries data for the US SW
w_moi = miroc_data.get_US_SW_slice(miroc_data.moisture, slice(0, len(miroc_data.moisture[0])))
d_moi = gfdl_data.get_US_SW_slice(gfdl_data.moisture, slice(0, len(gfdl_data.moisture[0])))
w_pre = miroc_data.get_US_SW_slice(miroc_data.precip, slice(0, len(miroc_data.precip[0])))
d_pre = gfdl_data.get_US_SW_slice(gfdl_data.precip, slice(0, len(gfdl_data.precip[0])))
w_eva = miroc_data.get_US_SW_slice(miroc_data.evap, slice(0, len(miroc_data.evap[0])))
d_eva = gfdl_data.get_US_SW_slice(gfdl_data.evap, slice(0, len(gfdl_data.evap[0])))

miroc_data.smooth_and_save_region_value_timeseries(w_moi, "./Output/sw_miroc_moisture.npy")
miroc_data.smooth_and_save_region_value_timeseries(w_pre, "./Output/sw_miroc_precipitation.npy")
miroc_data.smooth_and_save_region_value_timeseries(w_eva, "./Output/sw_miroc_evaporation.npy")

gfdl_data.smooth_and_save_region_value_timeseries(d_moi, "./Output/sw_gfdl_moisture.npy")
gfdl_data.smooth_and_save_region_value_timeseries(d_pre, "./Output/sw_gfdl_precipitation.npy")
gfdl_data.smooth_and_save_region_value_timeseries(d_eva, "./Output/sw_gfdl_evaporation.npy")

#########################################################] Dataset plots
# This section of the program plots the datasets, and
# creates a total of 6 plots.

def plot_moisture(slice_function, region_name="the Sahel", ylim=(-1000, 2000)):
    """A function that plots the moisture data from both scenarios for
    a given region, in the form of an anomaly from the first 10 years.
    
    Params:
        slice_function- a function that returns a region from a dataset,
            such as get_sahel_slice().
        region_name- str. The name of the region. Used for labeling the plot.
        ylim- tuple of two ints in format (lower, upper). The limits for
            the y axis.

    This function does NOT call plt.show()!
    """
    #get the data
    wet_sahel = slice_function(miroc_data.moisture, slice(0, len(miroc_data.moisture[0])))
    dry_sahel = slice_function(gfdl_data.moisture, slice(0, len(gfdl_data.moisture[0])))
    wsa = miroc_data.get_average_value_timeseries_of_region(wet_sahel)
    dsa = gfdl_data.get_average_value_timeseries_of_region(dry_sahel)
    wsa = miroc_data.absolute_timeseries_to_anomaly(wsa)
    dsa = gfdl_data.absolute_timeseries_to_anomaly(dsa)

    #set up the plot
    fig, ax1 = plt.subplots(figsize=(12, 5))
    years_w = range(miroc_data.start_year, miroc_data.start_year+len(miroc_data.moisture[0]))
    years_d = range(gfdl_data.start_year, gfdl_data.start_year+len(gfdl_data.moisture[0]))
    ax1.set_ylim(ylim)
    ax1.set_ylabel("Anomaly (kg m⁻²)")
    ax1.set_xlabel("Year")
    title_text="Projected Soil Moisture Anomaly in "+region_name
    ax1.set_title(title_text)

    ax1.plot(years_w, wsa, label="MIROC-ESM-CHEM", color="lightseagreen")
    ax1.plot(years_d, dsa, label="GFDL CM3", color="brown")
    ax1.axhline(y=0, color="black", linewidth=1)
    ax1.axvline(x=2020, color="red", linewidth=1)

    fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    ## save as pdf:
    f = plt.gcf()  # f = figure(n) if you know the figure number
    plt.savefig("Output/Figures/"+title_text.replace(" ","-")+".pdf",format='pdf');

def plot_precip(slice_function, region_name="the Sahel", ylim=(-.2, .4)):
    """A function that plots the precipitation data from both scenarios for
    a given region, in the form of an anomaly from the first 10 years.
    
    Params:
        slice_function- a function that returns a region from a dataset,
            such as get_sahel_slice().
        region_name- str. The name of the region. Used for labeling the plot.
        ylim- tuple of two ints in format (lower, upper). The limits for
            the y axis.

    This function does NOT call plt.show()!
    """
    wet_sahel = slice_function(miroc_data.precip, slice(0, len(miroc_data.precip[0])))
    dry_sahel = slice_function(gfdl_data.precip, slice(0, len(gfdl_data.precip[0])))
    wsa = miroc_data.get_average_value_timeseries_of_region(wet_sahel)
    dsa = gfdl_data.get_average_value_timeseries_of_region(dry_sahel)
    wsa = miroc_data.absolute_timeseries_to_anomaly(wsa)
    dsa = gfdl_data.absolute_timeseries_to_anomaly(dsa)
    wsa_s = miroc_data.get_average_value_timeseries_of_region_s(wet_sahel)
    dsa_s = gfdl_data.get_average_value_timeseries_of_region_s(dry_sahel)
    wsa_s = miroc_data.absolute_timeseries_to_anomaly(wsa_s)
    dsa_s = gfdl_data.absolute_timeseries_to_anomaly(dsa_s)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    years_w = range(miroc_data.start_year, miroc_data.start_year+len(miroc_data.precip[0]))
    years_d = range(gfdl_data.start_year, gfdl_data.start_year+len(gfdl_data.precip[0]))
    years_w_s = range(miroc_data.start_year+2, miroc_data.start_year+len(miroc_data.precip[0])-2)
    years_d_s = range(gfdl_data.start_year+2, gfdl_data.start_year+len(gfdl_data.precip[0])-2)
    # years_s = range(2008, 2099)
    ax1.set_ylim(ylim)
    ax1.set_ylabel("Anomaly (m y⁻¹)")
    ax1.set_xlabel("Year")
    
    ax1.plot(years_w, wsa, ":", label="MIROC-ESM-CHEM (raw)", 
        color="lightseagreen", linewidth=1)
    ax1.plot(years_d, dsa, ":", label="GFDL CM3 (raw)", 
        color="brown", linewidth=1)
    ax1.plot(years_w_s, wsa_s, label="MIROC-ESM-CHEM (smoothed)", color="lightseagreen")
    ax1.plot(years_d_s, dsa_s, label="GFDL CM3 (smoothed)", color="brown")
    ax1.axhline(y=0, color="black", linewidth=1)
    ax1.axvline(x=2020, color="red", linewidth=1)

    title_text="Projected Precipitation Anomaly in "+region_name
    ax1.set_title(title_text)
    fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    ## save as pdf:
    f = plt.gcf()  # f = figure(n) if you know the figure number
    plt.savefig("Output/Figures/"+title_text.replace(" ","-")+".pdf",format='pdf');

def plot_evap(slice_function, region_name="the Sahel", ylim=(-.2, .4)):
    """A function that plots the evaporation data from both scenarios for
    a given region, in the form of an anomaly from the first 10 years.
    
    Params:
        slice_function- a function that returns a region from a dataset,
            such as get_sahel_slice().
        region_name- str. The name of the region. Used for labeling the plot.
        ylim- tuple of two ints in format (lower, upper). The limits for
            the y axis.

    This function does NOT call plt.show()!
    """
    wet_sahel = slice_function(miroc_data.evap, slice(0, len(miroc_data.evap[0])))
    dry_sahel = slice_function(gfdl_data.evap, slice(0, len(gfdl_data.evap[0])))
    wsa = miroc_data.get_average_value_timeseries_of_region(wet_sahel)
    dsa = gfdl_data.get_average_value_timeseries_of_region(dry_sahel)
    wsa = miroc_data.absolute_timeseries_to_anomaly(wsa)
    dsa = gfdl_data.absolute_timeseries_to_anomaly(dsa)
    wsa_s = miroc_data.get_average_value_timeseries_of_region_s(wet_sahel)
    dsa_s = gfdl_data.get_average_value_timeseries_of_region_s(dry_sahel)
    wsa_s = miroc_data.absolute_timeseries_to_anomaly(wsa_s)
    dsa_s = gfdl_data.absolute_timeseries_to_anomaly(dsa_s)
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    years_w = range(miroc_data.start_year, miroc_data.start_year+len(miroc_data.precip[0]))
    years_d = range(gfdl_data.start_year, gfdl_data.start_year+len(gfdl_data.precip[0]))
    years_w_s = range(miroc_data.start_year+2, miroc_data.start_year+len(miroc_data.precip[0])-2)
    years_d_s = range(gfdl_data.start_year+2, gfdl_data.start_year+len(gfdl_data.precip[0])-2)
    ax1.set_ylim(ylim)
    ax1.set_ylabel("Anomaly (m y⁻¹)")
    ax1.set_xlabel("Year")

    ax1.plot(years_w, wsa, ":", label="MIROC-ESM-CHEM (raw)", 
        color="lightseagreen", linewidth=1)
    ax1.plot(years_d, dsa, ":", label="GFDL CM3 (raw)", 
        color="brown", linewidth=1)
    ax1.plot(years_w_s, wsa_s, label="MIROC-ESM-CHEM (smoothed)", color="lightseagreen")
    ax1.plot(years_d_s, dsa_s, label="GFDL CM3 (smoothed)", color="brown")
    ax1.axhline(y=0, color="black", linewidth=1)
    ax1.axvline(x=2020, color="red", linewidth=1)
    title_text="Projected Evaporation Anomaly in "+region_name
    ax1.set_title(title_text)
    fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    ## save as pdf:
    f = plt.gcf()  # f = figure(n) if you know the figure number
    plt.savefig("Output/Figures/"+title_text.replace(" ","-")+".pdf",format='pdf');

#plot the data for the sahel
plot_moisture(gfdl_data.get_sahel_slice)
plot_precip(gfdl_data.get_sahel_slice)
plot_evap(gfdl_data.get_sahel_slice) 

#plot the data for the SW
plot_moisture(gfdl_data.get_US_SW_slice, "the US Southwest", (-1000, 2000))
plot_precip(gfdl_data.get_US_SW_slice, "the US Southwest", (-.3, .3))
plot_evap(gfdl_data.get_US_SW_slice, "the US Southwest", (-.3, .3))

#show all the plots
plt.show()

###################################################### contour animation
# This section of the program contains a function that
# creates animation frames for a given region. The call
# to the function is currently commented out.
# Note that this section will not execute until all the
# plots created in the previous section are closed.

def contour_animation(w_set, d_set, slice_function, region_bounds, buff, norms, units, name, region):
    """
    Create and save yearly animation frames for one region in two scenarios.

    Params:
        w_set- the dataset for the one scenario (e.g miroc_data.precip)
        d_set- the dataset for the other scenario (e.g gfdl_data.precip)
        slice_function- A function that gets a regional slice of the
            dataset (e.g. miroc_data.get_sahel_slice)
        region_bounds- A 4-int tuple representing the region's boundaries
            in format (lon min, lon max, lat min, lat max)
        buff- int. The number of degrees to be displayed surrounding the 
            target region (serves as visual buffer) (e.g 5)
        norms- A 2-int tuple representing the lower and upper
            values for which the colormap is normalized, in format
            (lower, upper) (e.g (-.5, .5))
        units- str. The label to be given to the units. (e.g "m y⁻¹")
        name- str. The name of the variable being contoured. (e.g. "Moisture")
    """
    lonmin, lonmax, latmin, latmax = region_bounds
    normmin, normmax = norms

    #set up the axes and the maps
    fig, (ax1, ax2) = plt.subplots(2, 1)
    map1 = Basemap(llcrnrlon=lonmin-buff,llcrnrlat=latmin-buff,
                    urcrnrlon=lonmax+buff,urcrnrlat=latmax+buff, ax=ax1)
    map2 = Basemap(llcrnrlon=lonmin-buff,llcrnrlat=latmin-buff,
                    urcrnrlon=lonmax+buff,urcrnrlat=latmax+buff, ax=ax2)

    #get the data
    wet_sahel = slice_function(w_set, slice(0, len(w_set[0])), buff)
    dry_sahel = slice_function(d_set, slice(0, len(d_set[0])), buff)
    wet_mean = miroc_data.get_average_map_of_region(wet_sahel, 10)
    dry_mean = gfdl_data.get_average_map_of_region(dry_sahel, 10)

    #set up the colormap and normalization
    cmap = mpl.cm.BrBG
    norm = mpl.colors.Normalize(vmin=normmin, vmax=normmax)
    #set up the colorbar
    cb_setter_array = np.asarray([
                            np.arange(normmin,normmax+1e-10,(normmax-normmin)/10),
                            np.arange(normmin,normmax+1e-10,(normmax-normmin)/10)])
    cb_setter = plt.contourf(cb_setter_array, 99, norm=norm, cmap=cmap)
    #create the colorbar to the right of both plots
    fig.subplots_adjust(right=0.82)
    cb_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cb = fig.colorbar(cb_setter, cax=cb_ax,
            ticks=np.arange(normmin,normmax+1e-10,(normmax-normmin)/10))
    #make the colorbar in scientific notation
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()
    #label the color bar
    cb.set_label(name+' Anomaly ('+units+')')

    def draw_frame(frame):
        """Draws frame #<frame>"""
        #relabel the plot with the current year
        fig.suptitle(name+" Anomaly From 2006-2016 Baseline\n"+str(2006+frame), y=.99)
        #clear the axes
        ax1.clear()
        ax2.clear()
        #title axes
        ax1.set_title("MIROC-ESC-CHEM")
        ax2.set_title("GFDL CM3")
        #calculate this frame's anomaly
        wet_anomaly = wet_sahel[0][frame] - wet_mean
        dry_anomaly = dry_sahel[0][frame] - dry_mean
        #draw coasts
        map1.drawcoastlines(linewidth=1)
        map2.drawcoastlines(linewidth=1)
        #draw a dashed rectangle around the sahel
        ax1.add_patch(Rectangle((lonmin, latmin), lonmax-lonmin, latmax-latmin,
                    fill=None, alpha=1, ls="--"))
        ax2.add_patch(Rectangle((lonmin, latmin), lonmax-lonmin, latmax-latmin,
                    fill=None, alpha=1, ls="--"))
        #contour this frame's data
        ax1.contourf(wet_sahel[2], wet_sahel[1], wet_anomaly, cmap=cmap, norm=norm,
                    levels=99)
        ax2.contourf(dry_sahel[2], dry_sahel[1], dry_anomaly, cmap=cmap, norm=norm,
                    levels=99)

    # eliminate first 10 years from MIROC so that length is same as GFDL:
    wet_sahel=(wet_sahel[0][10:],wet_sahel[1],wet_sahel[2])
    for i in range(len(wet_sahel[0])):
        draw_frame(i)
        print(i, i/len(wet_sahel[0]), name, region, end="         \r") #print progress- frame # and %
        plt.savefig("./Output/frames/"+name.replace(" ","-")+"-"+region+"-frame-" + str(i).rjust(5, '0') + ".png")

# The longtitude/latitude bounds for the Sahel and the SW US, for easy
# use.
SAHEL_BOUNDS = (-15, 40, 10, 20)
SW_BOUNDS = (235, 255, 32, 41)


contour_animation(miroc_data.precip, gfdl_data.precip, miroc_data.get_US_SW_slice, SW_BOUNDS,
                  5, (-.5, .5), "m y⁻¹", "Precipitation","southwest")
contour_animation(miroc_data.moisture, gfdl_data.moisture, miroc_data.get_US_SW_slice,
                  SW_BOUNDS, 5, (-1000, 1000), "kg m⁻²", "Soil Moisture","southwest")
contour_animation(miroc_data.evap, gfdl_data.evap, miroc_data.get_US_SW_slice, SW_BOUNDS, 5,
                  (-.5, .5), "m y⁻¹", "Evaporation","southwest")

contour_animation(miroc_data.precip, gfdl_data.precip, miroc_data.get_sahel_slice, SAHEL_BOUNDS,
                  5, (-.5, .5), "m y⁻¹", "Precipitation","sahel")
contour_animation(miroc_data.moisture, gfdl_data.moisture, miroc_data.get_sahel_slice, SAHEL_BOUNDS,
                  5, (-1000, 1000), "kg m⁻²", "Soil Moisture","sahel")
contour_animation(miroc_data.evap, gfdl_data.evap, miroc_data.get_sahel_slice, SAHEL_BOUNDS, 5,
                  (-.5, .5), "m y⁻¹", "Evaporation","sahel")