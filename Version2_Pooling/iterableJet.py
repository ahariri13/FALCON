# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:51:05 2020

@author: aah71
"""

from torch.utils.data import Dataset, IterableDataset, DataLoader
from itertools import cycle, islice

class IterableMuons(IterableDataset):
    
    def __init__(self,data):
        self.data=data
    
    def process_data(self,data):
        for graph in data:
            yield graph
            
    def get_stream(self,data):
        return cycle(self.process_data(data))
    
    def __iter__(self):
        return self.get_stream(self.data)
    
    