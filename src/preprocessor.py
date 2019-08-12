# desired output
# timestamp,value
# 2013-07-04 01:00:00,71.22022706

import datetime
import pandas as pd
import os

class Preprocessor(object):
    def __init__(self, fileName, delimiter=None, timestamp=None, folderStruct=None):
        self.datasetUrl = fileName
        if delimiter is None:
            self.delimiter = ','
        else:
            self.delimiter = delimiter 
        if timestamp is None:
            self.hasTimestamp = True
        else:
            self.hasTimestamp = timestamp
        self.headerLines = 0
        self.folderStruct = folderStruct
        self.outputDest = None
        self.cache = []
        self.cacheSize = 0
        
    def normalize(self):
        '''
        Normalizes column data
        @todo: normalize on dataframe min/max or column min/max?
        '''
        pass
    
    def printFeatures(self):
        '''
        Plots basic feature information
        @todo: have column ids as argument
        @todo: obsolete here
        '''
        pass
    
    def autoPP(self):
        '''
        Tries to automatically preprocess
        '''
        self.loadCache()
        if self.hasTimestamp == False:
            self.addTimestamp()
        #there might be commas in there for floats
        if self.delimiter != ',':
            #if there are commas, they are probably used for floats
            self.replaceCommaToDot()
            #now ready to replace the delimiters
            self.replaceDelimiter()
        #regenerateHeader last because we need to calculate the amount of columns
        self.regenerateHeader()
        self.writeCache()
        
    def loadCache(self):
        '''
        Loads file in self.cache
        '''
        file = open(self.datasetUrl, 'r')
        for line in file:
            self.cache.append(line)
        self.cacheSize = len(self.cache)
        
    def writeCache(self):
        '''
        Write out the cache to a new .csv
        Checks if a folder struct is given
        '''
        if self.folderStruct is None:
            self.outputDest = '../data_preprocessed/' + self.datasetUrl.split('/')[-1]
        else:
            self.outputDest = '../data_preprocessed' + self.folderStruct
            if not os.path.exists(self.outputDest):
                os.makedirs(self.outputDest)
            self.outputDest = self.outputDest + self.datasetUrl.split('/')[-1]
        file = open(self.outputDest,'w')
        file.writelines(self.cache)
        file.close()
        print('Successfully wrote to {}'.format(self.outputDest))
        
    def numberOfColumns(self):
        '''
        Calculates the amount of columns with values
        '''
        assert self.headerLines == 1,'Header lines exceed 1'
        assert self.delimiter == ',','Delimiter not set to ,'
        cols = self.cache[1].count(self.delimiter)
        print('{} columns found'.format(cols))
        return cols
                
    def regenerateHeader(self):   
        '''
        Regenerates the headers. Ideally used as a last step
        '''
        self.setHeaderLines()
        if self.headerLines == 0:
            self.cache.insert(0,'')
        if self.headerLines > 1:
            print('Trimming header lines')
            for i in range(self.headerLines-1):
                del self.cache[0]
            self.setHeaderLines()
        if self.headerLines == 1:
            values = ''
            for i in range(self.numberOfColumns()):
                values = values + ',v_{}'.format(i)
            self.cache[0] = 'timestamp{}\n'.format(values)
        print('Header lines regenerated')
        
    def addTimestamp(self):
        '''
        Adds a timestamp column
        '''
        assert self.hasTimestamp == False,'Already has timestamps'
        
        dt = datetime.datetime(2010, 12, 1)
        step = datetime.timedelta(seconds=1)
                                
        for i in range(self.cacheSize):
            if i == 0:
                self.cache[i] = 'timestamp,'+ self.cache[i]
            else:
                timestamp = dt.strftime('%Y-%d-%m %H:%M:%S')
                self.cache[i] = timestamp + self.delimiter + self.cache[i]
                dt += step
            
        self.hasTimestamp = True
        print('Added timestamp')

    def replaceCommaToDot(self):
        '''
        Replaces commas in floats to dots
        '''
        assert self.delimiter != ',','Might replace actual commas'
        
        for i in range(self.cacheSize):
            self.cache[i] = self.cache[i].replace(',', '.')
        print('Replaced , with . (floats)')
        
    def replaceDelimiter(self, delimiter=None):
        '''
        Replaces the delimiter that is specified with ','
        '''
        if delimiter is None:
            delimiter = self.delimiter
        
        if delimiter != ',':
            for i in range(self.cacheSize):
                self.cache[i] = self.cache[i].replace(delimiter, ',')
            self.delimiter = ','
            print('Replaced {} with , (delimiter)'.format(delimiter))
                
    def setHeaderLines(self):
        '''
        Set self.headerLines to the amount of header lines in a file
        '''
        # search up to parseMax
        parseMax = 20
        self.headerLines = 0
        
        for line in self.cache:
            parseMax -= 1
            for character in line:
                #do not count small e as it is used for scientific notations
                if character.lower() in 'abcdfghijklmnopqrstuvwxyz':
                    self.headerLines += 1
                    break
            if parseMax == 0:
                break
        print('Headerlines set to {}'.format(self.headerLines))