# -*- coding: utf-8 -*-
#__doc__
"""This script parses the data from the HURDAT2 dataset, saves and
plots the PDI, annual and 5-year smoothed.

Requirements:
    A .txt file containing the contents of the HURDAT2 dataset
        (acquired https://www.aoml.noaa.gov/hrd/hurdat/hurdat2.html,
        full website: https://www.aoml.noaa.gov/hrd/hurdat/Data_Storm.html)
    The format of the text should be similar to:
     AL081993,               GERT,     30,
     19930914, 1800,  , TD, 10.6N,  80.7W,  25, 1008, -999, -999, [...]
     19930915, 0000,  , TD, 10.7N,  81.3W,  30, 1005, -999, -999, [...]
    (example is cut to 70 chars to prevent text wrapping)
    
Output:
    The script creates two .npy files containing one bundled numpy array each.
    The files are created in whatever directory the script is currently in.
    
    'pdi.npy'     - Contains a time series of unsmoothed PDI data from 
                    the data file for every year
    'pdi_5yr.npy' - Contains the 5-year-smoothed PDI time series
                    
    These files can be loaded in another script through usage of 
    np.load(filepath), which returns the original numpy array.
    The arrays are 2D numpy arrays with two rows. Index [0] contains an 
    array of years, and index [1] contains an array of PDI values that 
    correspond to each year
    
    Example of using the saved npy files:
    
    >>> x = np.load("pdi_5yr.npy")[0]  #the years
    array([1853. 1854. 1855. 1856.])
    >>> x = np.load("pdi_5yr.npy")[1]  #the PDIs for those years
    array([7.80812641e+07 5.78337754e+07 4.36968420e+07 4.71030141e+07])
    
    Example of usage:
    
    x = np.load("pdi.npy")
    plt.plot(x[0], x[1])       #plot the raw pdi

Usage Instructions:
    To get the .npy files:

        # specify example path to hurdat data:
        HURDAT_PATH = "../../../Data-for-teaching-staff/Hurricanes/hurdat/hurdat2.txt"
        # initialize instance:
        e = read_and_analyze_HURDAT(HURDAT_PATH)
        # create and save PDI files:
        e.save_PDI_data()

        The files will appear in the
    directory that the script is run in.
    
Written by Yonathan Vardi 2019-07-02

"""

import matplotlib.pyplot as plt
import numpy as np
import sys

###########################################################################
# Hurricane class
###########################################################################
class Hurricane:
    """
    A representation of a single hurricane. 
    Documents its identifying informationand its wind velocities at 6-hour 
    intervals.
    
    Members:
        year- int. The year during which the hurricane occurred (e.g 2019)
        number- int. The hurricane's number within the given year (e.g 1)
        name- str. The hurricane's name. (e.g BERYL, UNNAMED)
        vels- [int]. A list representing wind velocities at 6-hour intervals.
            The units provided by HURDAT2, and thus used here, are knots.
            e.g. ([25, 30, 35, 45, 50, 55, 60, 60, 55, 50, 30, 25])
    """
    
    def __init__(self, year, number, name):
        """initialize an instance of Hurricane with no wind data.
        
        Params:
            year- int. The year during which the hurricane occurred (e.g 2019)
            number- int. The hurricane's number within the given year (e.g 1)
            name- str. The hurricane's name. (e.g BERYL, UNNAMED)
        """
        self.year = year
        self.number = number
        self.name = name
        self.vels = []    #a list 
        
    def add_velocity(self, velocity):
        """Add a single wind velocity measurement to the hurricane.
        
        Params:
            velocity- int. Represents a single measurement, in knots, of wind,
                taken at a 6-hour interval (e.g 30)
        """
        self.vels.append(velocity)
        
    def __str__(self):
        """Allows a Hurricane to be print()ed. Used for debugging purposes."""
        return str(self.year) + "#" + str(self.number) + "_" + self.name


###########################################################################
# read_and_analyze_HURDAT class
###########################################################################
class read_and_analyze_HURDAT:
    """Class that handles extraction and analysis of data from HURDAT2
    
    Public Members:
        hurricanes- [Hurricane]. A list of Hurricane instances, each 
            representing a single hurricane. Altogether it contains all the 
            hurricanes in the HURDAT2 dataset. See Hurricane for more details.

        dataset_first_year- int. The first year for which data from HURDAT2 is
            available.

        dataset_last_year- int. The last year for which data for HURDAT2 is
            available.
    
    Public Methods:
        *More detailed descriptions in docs of individual methods
        
        get_year_range_PDIs()->[float] - Return a list of cumulative year PDIs
            for years within a given range

        get_year_range_smooth_PDIs()->[float] - Return a list of cumulative year
            PDIs for years within a given range, with 5-year-smoothing applied

        get_hurricane()->Hurricane - Return a specific hurricane

        get_hurricanes_in_year()->[Hurricane] - Return a list of all hurricanes 
            in a given year

        calc_hurricane_PDI()->int - calculate the PDI for the given hurricane

        calc_cumulative_year_PDI()->int - calculate the cumulative PDI for the 
            given year

        plot_PDIs() - plot the cumulative PDIs for years in the given range

        plot_PDIs_5yr_smooth() - plot the cumulative PDIs for years in the given 
            range, with 5-year smoothing applied.

        plot_raw_and_smooth_PDI() - plot both raw and smoothed cumulative PDIs for
            years in the given range

        save_PDI_data() - save the raw and 5-year-smoothed PDI data to disk
    """
    
    def __init__(self, hurdat_path):
        """Construct a read_and_analyze_HURDAT
        
        This method opens and parses the HURDAT2 data file, and at its end
        the read_and_analyze_HURDAT contains a full list of hurricanes and velocities.

        Params:
            hurdat_path- The filepath to a text file containing the HURDAT2
                data. (e.g, '../../../Data-for-teaching-staff/Hurricanes/hurdat.txt') 
                Sample of file format found in 'Requirements' section of 
                __doc__ at top of script.
        """
        #open the file containing the HURDAT2 data
        hurdat_file = open(hurdat_path, "r")
        lines = hurdat_file.read().splitlines()    #split data into list of lines
        #create members
        self.hurricanes = []
        self.dataset_first_year = 99999    #these values are placeholders
        self.dataset_last_year = 0    #real values are found during parsing
        # begin analyzing file
        for i in lines:
            # when encountering a header, create a new hurricane as specified
            if self._is_header(i):
                self.hurricanes.append(self._extract_hur_from_header(i))
                # when a new hurricane is added, update the first and last years
                # to the biggest or smallest value so far
                self.dataset_first_year = min(self.dataset_first_year,
                                              self.hurricanes[-1].year)
                self.dataset_last_year = max(self.dataset_last_year,
                                              self.hurricanes[-1].year)
            else:
                # if encountering a data line, add velocity to latest hurricane
                vel = self._extract_vel_from_line(i)
                if vel >= 0:
                    # this check ensures that a negative (invalid) value
                    # returned from _extract_vel_from_line() does not make it
                    # into the finished list
                    self.hurricanes[-1].add_velocity(vel)
        
        
    ############################################################ private methods
        
    
    def _is_header(self, line):
        """Return whether or not the given line is a header.
        
        Header lines introduce a new hurricane and contain its identifying info
        
        Params:
            line-- str. Represents a single line from the HURDAT2 file.
            e.g:
            'AL081993,               GERT,     30,'
            
        Return True or False
        """
        # Content lines begin with a number, headers with a letter. This way
        # they can be easily distinguished.
        return line[0].isalpha()
    
    def _extract_hur_from_header(self, line):
        """Return a hurricane as specified by the info provided in a header line
        
        Params:
            line-- str. Represents a single header from the HURDAT2 data.
            e.g:
            'AL081993,               GERT,     30,'
        
        Return an instance of Hurricane
        """
        # Character positions for each data field specified in the HURDAT2 docs.
        # More info about the meaning of these values can be found in the
        # Hurricane class
        number = int(line[2:4])
        year = int(line[4:8])
        name = line[18:28].strip()
        return Hurricane(year, number, name)
        
    def _extract_vel_from_line(self, line):
        """Extracts a wind velocity from a given line of HURDAT2 data
        
        Params:
            line- str. Represents a single line from the HURDAT2 data.
            e.g:
            '19930907, 1200,  , TD, 24.7N,  67.5W,  25, 1012, -999, -999, [...]'
            
        Returns an int representing the wind velocity. (e.g 55)
            If line is a header, line pertains to a time outside of the normal
            6-hour intervals, line has a negative wind velocity, or line 
            is < 41 chars long, return -1.
        """
        # character positions for each data field specified in the HURDAT2 docs
        
        # 6-hour intervals (see "*" below)
        valid_times = {'0000', '0600', '1200', '1800'}
        if self._is_header(line):
            # header lines contain no velocity
            return -1
        if len(line) < 41:
            # lines under 41 chars are too short to contain wind velocity
            return -1
        time = line[10:14]    #this is the time of day of the velocity measurement
        if not time in valid_times:
            # *Most of the velocities in HURDAT2 are measured during one of four 
            # 6-hour intervals, but some are not. The ones that are not are
            # discarded, because PDI is calculated at 6-hour intervals.
            return -1
        velocity = line[38:41]    #This is the measured wind velocity
        if not velocity.strip().isdigit():
            # If this is true, velocity is likely negative, or otherwise is 
            # corrupted. In any case, it should not be used
            return -1
        return int(velocity)  #if all is good, return the velocity as an int
    
    
    ############################################################# public methods
    
    def get_year_range_PDIs(self, start_year, end_year):
        """Return each year's cumulative PDI within the given year range
        
        Params:
            start_year- int. The first year to plot. (e.g 1900)
            end_year- int. The first year NOT to plot. (e.g 2010)
            
        Return a list of PDIs
        """
        PDIs = []
        for i in range(start_year, end_year):
            PDIs.append(self.calc_cumulative_year_PDI(i))
        return PDIs
    
    def get_year_range_smooth_PDIs(self, start_year, end_year):
        """Return 5-year-smoothed list of the cumulative PDIs
        
        This is done by passing the data through two iterations of the formula
        X1 = .5X1 + .25(X0 + X2)
        where X1 represents a point and X0 and X2 represent its two adjacent 
        points. Each iteration of this formula also removes the first and last
        elements of the data.
        
        Params:
            start_year- int. The first year to plot. (e.g 1900)
            end_year- int. The first year NOT to plot. (e.g 2010)
            
        Return a list of PDIs
        """
        PDIs = []
        for i in range(start_year, end_year):
            PDIs.append(self.calc_cumulative_year_PDI(i))
        # run it through two iterations of smoothing
        for i in range(2):
            # the new values are stored here to prevent bias based on 
            # order in the list
            temp_PDI = []
            for j in range(1, len(PDIs)-1):
                temp_PDI.append(.5*PDIs[j] + .25*(PDIs[j-1] + PDIs[j+1]))
            # replace the old PDI list with the new one
            PDIs = temp_PDI
        return PDIs
    
    def get_hurricane(self, year, identifier):
        """Return a specific hurricane that occurred in a given year
        
        Params:
            year-- int. The year in which the hurricane occurred (e.g. 2006)
            identifier-- Either int or str. If int, it is assumed to mean the
                hurricane number for the given year, starting at 1. 
                If str, it is assumed to be the name of the hurricane. Not case-
                sensitive.
                (e.g. 5, 'ARLENE', 'arlene')
                
        Return a Hurricane if one matching the identifier is found, otherwise
            return None.
        """
        # begin with a list of all hurricanes in the year
        hurs = self.get_hurricanes_in_year(year)
        if isinstance(identifier, str):
            # if identifier is string, look by string
            for i in hurs:
                if i.name == identifier.upper():
                    return i
        elif isinstance(identifier, int):
            # if identifier is int, look by int
            for i in hurs:
                if i.number == identifier:
                    return i
        #return None if invalid identifier type or no hurricane found
        return None
    
    def get_hurricanes_in_year(self, year):
        """Return all the hurricanes that occurred during the given year
        
        Paramters:
            year- int. The year whose hurricanes to return. (e.g 2006)
            
        Returns a list of Hurricanes
        """
        hurs = []
        for i in self.hurricanes:
            if i.year == year:
                hurs.append(i)
            elif i.year > year:
                # Hurricanes are sorted by time so this is acceptable
                break
        return hurs
    
    def calc_hurricane_PDI(self, hurricane):
        """Calculate the PDI for a single given hurricane
        
        Params:
            hurricane- Hurricane. The hurricane whose PDI is calculated. 
            (e.g the return value of get_hurricane(2004, 'frances'))
        
        Return a float representing the PDI
        """
        pdi = 0
        speed_threshold = 30.4142    #this is in knots; 35mph
        for i in hurricane.vels:
            if i <= speed_threshold:
                #winds at 35mph or lower are ignored
                continue
            pdi += (i*1.15078)**3    #convert i from knots to mph before cubing
        return pdi
    
    def calc_cumulative_year_PDI(self, year):
        """Calculate the cumulative yearly PDI for the given year
        
        Params:
            year- int. The year for which to calculate. (e.g 2006)
        
        Return a float representing the cumulative PDI for the year. This will
            be 0.0 if there is no data for that year. (e.g 73185885.8205246)
        """
        pdi = 0.0
        hurs = self.get_hurricanes_in_year(year)
        for i in hurs:
            hur_pdi = self.calc_hurricane_PDI(i)
            pdi += hur_pdi
        return pdi
    
    def plot_PDIs(self, start_year, end_year):
        """Draw a plot of each year's cumulative PDI within the given year range
        
        Params:
            start_year- int. The first year to plot. (e.g 1900)
            end_year- int. The first year NOT to plot. (e.g 2010)
        """
        PDIs = np.asarray(self.get_year_range_PDIs(start_year, end_year))
        t = np.asarray(range(start_year, end_year))
        plt.plot(t, PDIs, label="Raw PDIs", linewidth=1)
        plt.legend()
        
    def plot_PDIs_5yr_smooth(self, start_year, end_year):
        """Draw 5-year-smoothed plot of the cumulative PDIs within a year range
        
        This is done by passing the data through two iterations of the formula
        X1 = .5X1 + .25(X0 + X2)
        where X1 represents a point and X0 and X2 represent its two adjacent 
        points. Each iteration of this formula also removes the first and last
        elements of the data.
        
        Params:
            start_year- int. The first year to plot. (e.g 1900)
            end_year- int. The first year NOT to plot. (e.g 2010)
        """
        PDIs = np.asarray(self.get_year_range_smooth_PDIs(start_year, end_year))
        t = np.asarray(range(start_year+2, end_year-2))
        plt.plot(t, PDIs, label="5-year smoothed PDI", linewidth=3)
        plt.legend()
    
    def plot_raw_and_smooth_PDIs(self, start_year, end_year):
        """plot both raw and 5-year-smoothed PDI values within a year range
        
        Params:
            start_year- int. The first year to plot. (e.g 1900)
            end_year- int. The first year NOT to plot. (e.g 2010)
        """
        self.plot_PDIs(start_year, end_year)
        self.plot_PDIs_5yr_smooth(start_year, end_year)
        
    def save_PDI_data(self):
        """
        Save the PDI and smoothed PDI for the entire record with their
        year range. More information regarding format found in section 'Output'
        of __doc__ at top of file.
        """
        #get the array of data
        raw_pdis = np.asarray(self.get_year_range_PDIs(
                self.dataset_first_year, self.dataset_last_year + 1))
        #get the array of years it corrosponds to
        raw_pdis_range = range(e.dataset_first_year, e.dataset_last_year+1)
        #bundle the two together
        pdi = np.asarray([raw_pdis_range, raw_pdis])
        
        #repeat the process for the smoothed PDIs
        smooth_pdis = np.asarray(self.get_year_range_smooth_PDIs(
                self.dataset_first_year, self.dataset_last_year + 1))
        smooth_pdis_range = np.asarray(range(e.dataset_first_year+2, e.dataset_last_year-1))
        pdi_5yr = np.asarray([smooth_pdis_range, smooth_pdis])

        #save the bundled arrays
        np.save("Output/pdi_5yr_smoothed.npy", pdi_5yr)
        np.save("Output/pdi.npy", pdi)

###########################################################################
# End of defenition of class read_and_analyze_HURDAT
###########################################################################


###########################################################################
# begin main program:
###########################################################################

HURDAT_PATH = "/hurdat/hurdat2.txt"
e = read_and_analyze_HURDAT(HURDAT_PATH)
e.save_PDI_data()
plt.ioff()
e.plot_raw_and_smooth_PDIs(e.dataset_first_year, e.dataset_last_year + 1)
plt.ylabel("Unscaled PDI")
plt.xlabel("Year")
plt.title("Power Dissipation Index")
plt.show()

