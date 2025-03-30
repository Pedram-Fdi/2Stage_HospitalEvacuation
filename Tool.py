from __future__ import absolute_import, division, print_function
import itertools as itools
import pandas as pd
from Constants import Constants

class Tool:
   
    @staticmethod
    def Transform2d(array, dimension1, dimension2):
        if len(array) != dimension1 * dimension2:
            raise ValueError("Array size does not match the specified dimensions.")
        result = [[array[i * dimension2 + j] 
                        for j in range(dimension2)] 
                            for i in range(dimension1)]
        return result

    @staticmethod
    def Transform3d(array, dimension1, dimension2, dimension3):
        if len(array) != dimension1 * dimension2 * dimension3:
            raise ValueError("Array size does not match the specified dimensions.")

        result = [[[
            array[i * (dimension2 * dimension3) + j * dimension3 + k]
                for k in range(dimension3)]
                    for j in range(dimension2)]
                        for i in range(dimension1)]

        return result

    @staticmethod
    def Transform4d(array, dimension1, dimension2, dimension3, dimension4):
        if len(array) != dimension1 * dimension2 * dimension3 * dimension4:
            raise ValueError("Array size does not match the specified dimensions.")
        
        result = [[[
            [array[i * (dimension2 * dimension3 * dimension4) + j * (dimension3 * dimension4) + k * dimension4 + l]
                for l in range(dimension4)]
                    for k in range(dimension3)]
                        for j in range(dimension2)]
                            for i in range(dimension1)]
        
        return result                  
    
    @staticmethod
    def Transform5d(array, dimension1, dimension2, dimension3, dimension4, dimension5):
        if len(array) != dimension1 * dimension2 * dimension3 * dimension4 * dimension5:
            raise ValueError("Array size does not match the specified dimensions.")
        
        result = [[[[[
            array[i * (dimension2 * dimension3 * dimension4 * dimension5) 
                  + j * (dimension3 * dimension4 * dimension5) 
                  + k * (dimension4 * dimension5) 
                  + l * dimension5 
                  + m]
            for m in range(dimension5)]
                for l in range(dimension4)]
                    for k in range(dimension3)]
                        for j in range(dimension2)]
                            for i in range(dimension1)]
        
        return result  

    @staticmethod
    def Transform6d(array, dimension1, dimension2, dimension3, dimension4, dimension5, dimension6):
        if len(array) != dimension1 * dimension2 * dimension3 * dimension4 * dimension5 * dimension6:
            raise ValueError("Array size does not match the specified dimensions.")
        
        result = [[[[[[
            array[i * (dimension2 * dimension3 * dimension4 * dimension5 * dimension6) 
                  + j * (dimension3 * dimension4 * dimension5 * dimension6) 
                  + k * (dimension4 * dimension5 * dimension6) 
                  + l * (dimension5 * dimension6) 
                  + m * dimension6 
                  + n]
            for n in range(dimension6)]
                for m in range(dimension5)]
                    for l in range(dimension4)]
                        for k in range(dimension3)]
                            for j in range(dimension2)]
                                for i in range(dimension1)]
        
        return result  

    @staticmethod
    def Transform7d(array, dimension1, dimension2, dimension3, dimension4, dimension5, dimension6, dimension7):
        if len(array) != dimension1 * dimension2 * dimension3 * dimension4 * dimension5 * dimension6 * dimension7:
            raise ValueError("Array size does not match the specified dimensions.")
        
        result = [[[[[[[
            array[i * (dimension2 * dimension3 * dimension4 * dimension5 * dimension6 * dimension7) 
                  + j * (dimension3 * dimension4 * dimension5 * dimension6 * dimension7) 
                  + k * (dimension4 * dimension5 * dimension6 * dimension7) 
                  + l * (dimension5 * dimension6 * dimension7) 
                  + m * (dimension6 * dimension7) 
                  + n * dimension7 
                  + o]
            for o in range(dimension7)]
                for n in range(dimension6)]
                    for m in range(dimension5)]
                        for l in range(dimension4)]
                            for k in range(dimension3)]
                                for j in range(dimension2)]
                                    for i in range(dimension1)]
        
        return result  
