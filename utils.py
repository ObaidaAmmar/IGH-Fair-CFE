import pandas as pd
import numpy as np

class GenSet(object):
    def __init__(self, name):
        self.counterfactuals = None
        self.name = name
    
    def set_counterfactuals(self, values):
        self.counterfactuals = [round(float(value), 3) for value in values] 
    
    def get_counterfactuals(self):
        return self.counterfactuals
    
    def modify_feature(self, idx, value, min_val, max_val, feature_dtype, feature_type):
        if feature_type == 'con':
            normalized_value = 200 * ((self.counterfactuals[idx] - min_val) / (max_val - min_val)) - 100
            modified_value = np.clip(normalized_value + value, -100, 100)
            denormalized_value = (modified_value + 100) / 200 * (max_val - min_val) + min_val
            if feature_dtype == 'int':
                denormalized_value = round(denormalized_value)
            else:
                denormalized_value = round(denormalized_value, 3)
            self.counterfactuals[idx] = denormalized_value
        else:
            if value > 0:
                modified_value = np.clip(self.counterfactuals[idx] + 1, min_val, max_val)
            elif value < 0:
                modified_value = np.clip(self.counterfactuals[idx] - 1, min_val, max_val)
            else:
                modified_value = self.counterfactuals[idx]
            self.counterfactuals[idx] = modified_value
        
        
def minMax(data):
    return pd.Series(index=['min','max'],data=[data.min(),data.max()])