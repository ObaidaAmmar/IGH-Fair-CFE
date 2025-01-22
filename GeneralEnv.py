import numpy as np
import copy
import tensorflow as tf
from gymnasium import Env, spaces
from utils import *

class GeneralEnv(Env):
    def __init__(self, dataset, affected_dataset, model, model_type, protected_attribute, features_to_change, features_types, number_of_counterfactuals, counterfactuals, target=None, minimums=None, maximums=None, macro=None, action_effectiveness = None, fairness_metrics="EF"):
        super(GeneralEnv, self).__init__()
        self.dataset = dataset # The entire dataset 
        self.dataset_features = list(affected_dataset.columns) # All feature names of the dataset
        self.affected_dataset = affected_dataset # The instances from Test Dataset that recieved unfavorable outcome
        self.protected_attribute = protected_attribute # Protected Attribute that we want to ensure the Fair Counterfactuals for its subgroups (Ex: Gender {Male, Female})
        self.macro = macro
        self.action_effectiveness = action_effectiveness # For Fairness Metric 2 (Equal Choice of Recourse) not included in version 1 
        self.fairness_metrics = fairness_metrics
        self.model = model # The Model used on the dataset
        self.model_type = model_type # Type of the model used
        self.cf_features = features_to_change # Features (to act on) that are considered in the Counterfactual Explanations
        self.features_types = features_types # Type of the features (continuous or categorical)
        self.features_dtypes = [self.dataset[feature].dtype for feature in self.cf_features] # Data Type of the features (int, float, ...)
        self.nb_cf = number_of_counterfactuals # Max Number of Counterfactual Explanations we want to generate
        self.counterfactuals = counterfactuals # A dict that contains the counterafactuals we found which satisfies the fairness metric
        self.counterfactual_set = set() # used to store counterfactuals and ensure no duplicates are saved
        self.state_dim = self.nb_cf * len(self.cf_features) # Dimension of the state will be number_of_counterfactuals * number_of_features_to_change
        self.best_episode_state = {} # Keep Track of Best State in an Epsiode to use when reseting the environment
        self.used_cf = [0] * self.nb_cf # Keep Track of CF that are used to achieve recourse
        
        self.reward_count = 0 # used as a key to store successful states in dict self.counterfactuals
        self.first_step = 0 # used to start from a pre-set initial state
        self.total_steps = 0
        
        min_max = minMax(self.dataset[self.cf_features]) # Extract minimum and maximum values of the features_to_change from the dataset
        
        # set mins and maxs of features to change
        # either pass mins and maxs as arguments or it will be extracted from the entire dataset
        if minimums:
            self.mins = minimums
        else:
            self.mins = np.array(min_max['min'].values) # Minimum values of the features cosidered in the Counterfactual Explanations
        
        if maximums:
            self.maxs = maximums
        else:
            self.maxs = np.array(min_max['max'].values) # Maximum values of the features cosidered in the Counterfactual Explanations

        # Determine the bounds of the state
        # For example if the features to change have the following mins = [0, 0, 0] and maxs = [10, 10, 30]
        # state bounds will be state_mins = [-10, -10, -30] and state_maxs = [10, 10, 30]
        self.state_mins , self.state_maxs = self.calculate_state_bounds(self.mins, self.maxs)
        self.max_gower = self.max_gower_distance(self.affected_dataset)
        
        # Used as a test to start the initial state from
        self.medians = [int(round((min_val + max_val) / 2)) for min_val, max_val in zip(self.mins, self.maxs)]
        
        # Action Space
        # Index 0: Represents the state index we want to change
        # Index 1: Represents the % of change to the state's index value
        self.action_space = spaces.Box(low=np.array([0,-3]), high=np.array([self.state_dim-1,3]), shape=(2,) , dtype=np.float32)
        
        # State Space
        # Ex: if we want to act on 'Income' and 'Credit' and generate a max number of 3 counterfactuals
        # the state will be of dimension (2x3 = 6) [Income, Credit, Income, Credit, Income, Credit]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        # Create an initial state either random or pre-set 
        self.initial_state_cfs = self.generate_state(random=False)
        self.initial_state = GenSet("CF")
        self.initial_state.set_counterfactuals(self.initial_state_cfs)
        self.state = self.initial_state
        
        # Find the initial label or the unfavorable label of the affected dataset
        if self.model_type == 'sklearn':
            first_individual = self.affected_dataset.iloc[0]
            self.initial_label = self.model.predict(pd.DataFrame([first_individual], columns = self.dataset_features))
        else:
            self.initial_label = np.argmax(self.model.predict(self.affected_dataset.iloc[0].values.reshape(1, -1)))

        #-------------------------------------DEBUGGING-----------------------------------------
        #print("----------------------------------------------")
        print(f"Action Sample: {self.action_space.sample()}")
        print(f"Max Gower Distance: {self.max_gower}")
        print(f"Fairness Metrics: {self.fairness_metrics}")
        #print("----------------------------------------------")
        print(f"State: {self.state.get_counterfactuals()}")
        print(f"Initial Label = {self.initial_label}")
        #print("----------------------------------------------")
        #---------------------------------------------------------------------------------------
        
        self.desired_label = target # Outcome that we aim to achieve using the counterafactual explanations
        self.done = False # determine if the goal reached
        self.final_reward = 1000 # give high reward to the agent when goal achieved
        self.current_step = 0 # keep track of number of steps per episode 
        self.nb_episode = 0 # used to apply per episode gradual decrease % perturbation to the reset state 
        self.steps_per_episode = 2000 # max steps per episode, if goal not reached the episode will truncate
        
    # used to calculate the minimum and maximum possible values for the state   
    def calculate_state_bounds(self, mins, maxs):
        state_mins = []
        state_maxs = []

        for i in range(len(mins)):
            if self.features_types[i] == 'con':
                result = maxs[i] - mins[i]
                state_mins.append(-result)
                state_maxs.append(result)
            else:
                state_mins.append(mins[i] - 1) # if we have the following categories (0,1,2) we make -1 available state to denote no change in category
                state_maxs.append(maxs[i])

        return state_mins, state_maxs
    
    # used to generate either a random or pre-set state
    def generate_state(self, random=True):
        initial_state = []
        if random:
            np.random.seed(42)
            for _ in range(self.nb_cf):
                for i, feature in enumerate(self.cf_features):
                    random_value = np.random.randint(low=self.state_mins[i], high=self.state_maxs[i])
                    initial_state.append(random_value)
        else:
            cf = []
            for i, value in enumerate(self.features_types):
                if value == "con":
                    cf.append(0)
                elif value == "cat" or value == "ord":
                    cf.append(self.state_mins[i])
            initial_state = cf * self.nb_cf
               
        return initial_state
    
    #used to split the dataset into 2 subgroups based on the binary protected attribute 
    def split_dataset_by_protected_attribute(self):
        group1 = self.affected_dataset[self.affected_dataset[self.protected_attribute] == 0]
        group2 = self.affected_dataset[self.affected_dataset[self.protected_attribute] == 1]
        return group1, group2
    
    
    #Evaluation of Fairness Metric 1: Equal Effectiveness
    def evaluate_fairness_metric1(self, counterfactuals, macro):
         
         # Split the affected_dataset into 2 subgroups based on the protected attribute
         group1, group2 = self.split_dataset_by_protected_attribute() 
         
         # Calculate the Proportion of individuals from 2 protected subgroups who can achieve recourse using the set of counterfactuals
         group1_proportion = float(f"{self.calculate_proportion(group1, counterfactuals, macro): .2f}")
         group2_proportion = float(f"{self.calculate_proportion(group2, counterfactuals, macro): .2f}")
         
         return group1_proportion, group2_proportion
     
         
    # Helper function for calculating Fairness Metric 1: Equal Effectiveness
    def calculate_proportion(self, group, counterfactuals, macro):
        total_individuals = len(group)
        success_count = 0         
        highest_success_count = 0

        #conver dataset and counterfactuals to tensors
        group_tensor = tf.convert_to_tensor(group, dtype=tf.float32)
        counterfactuals_tensor = tf.convert_to_tensor(counterfactuals, dtype=tf.float32)
        
        #used to determine number of individuals who achieved the desired outcome
        outcome_achieved = tf.zeros(group_tensor.shape[0], dtype=tf.bool)
        
        i = 0 # track CFs that could be used to achieve recourse
        #macro-viewpoint considers collective impact of an action
        []
        if macro:
            for cf in counterfactuals_tensor:
                outcome_achieved = tf.zeros(group_tensor.shape[0], dtype=tf.bool)
                modified_group = self.apply_counterfactual(group_tensor, cf)
                predictions = self.test_outcome(modified_group)
                outcome_achieved = tf.logical_or(outcome_achieved, predictions)
                success_count = tf.reduce_sum(tf.cast(outcome_achieved, tf.float32)).numpy()
                if success_count / total_individuals >= 0.3:
                    self.used_cf[i] = 1
                else:
                    self.used_cf[i] = 0
                i += 1
                if success_count > highest_success_count:
                    highest_success_count = success_count
            return highest_success_count / total_individuals
        else: # micro-viewpoint, each individual chooses the action that best benefits itself
            for cf in counterfactuals_tensor:
                modified_group = self.apply_counterfactual(group_tensor, cf)
                #modified_group = tf.cast(modified_group, dtype=tf.int32)
                predictions = self.test_outcome(modified_group)
                
                if self.model_type == 'sklearn':
                    predictions_tensor = tf.convert_to_tensor(predictions, dtype=tf.bool)
                else:
                    predictions_tensor = tf.cast(predictions, dtype=tf.bool)
                
                if tf.reduce_any(predictions_tensor):
                    self.used_cf[i] = 1
                
                i += 1
                outcome_achieved = tf.logical_or(outcome_achieved, predictions)
            
            proportion = tf.reduce_mean(tf.cast(outcome_achieved, tf.float32))
            return proportion.numpy()
    
    
    #Evaluation of Fairness Metric 2: Equal Choice of Recourse
    def evaluate_fairness_metric2(self, counterfactuals, effectiveness):
        
        # Split the affected_dataset into 2 subgroups based on the protected attribute
         group1, group2 = self.split_dataset_by_protected_attribute()
         
         # Calculate the number of effective actions (counterfactuals) for the 2 protected subgroups 
         # If the counterfactual achieve recourse for a proportion >= effectiveness then we count it
         nb_actions_group1 = self.calculate_actions(group1, counterfactuals, effectiveness)
         nb_actions_group2 = self.calculate_actions(group2, counterfactuals, effectiveness)
         
         return nb_actions_group1, nb_actions_group2
     
    # Helper Function to calculate Fairness Metric 2
    def calculate_actions(self, group, counterfactuals, effectiveness):
        total_individuals = len(group)
        nb_of_effective_actions = 0 # An action (counterfactual) is effective if it achieves recourse for a proportion pf individuals >= effectiveness 
        
        #conver dataset to tensors
        group_tensor = tf.convert_to_tensor(group, dtype=tf.float32)
        
        # Ensure counterfactuals are unique
        cf_strings = ["_".join(map(str, cf)) for cf in counterfactuals]
        unique_cf_strings = {}
        for index, cf_str in enumerate(cf_strings):
            if cf_str not in unique_cf_strings:
                unique_cf_strings[cf_str] = index # Record the first occurrence
        
        # Extract unique counterfactuals and their first indices
        unique_counterfactuals = [counterfactuals[idx] for idx in unique_cf_strings.values()]
        first_indices = list(unique_cf_strings.values())

        # Test Each Unique Counterfactual
        for cf, first_index in zip(unique_counterfactuals, first_indices):
            cf_tensor = tf.convert_to_tensor(cf, dtype=tf.float32)
            modified_group = self.apply_counterfactual(group_tensor, cf_tensor)
            predictions = self.test_outcome(modified_group)
            success_count = tf.reduce_sum(tf.cast(predictions, tf.float32)).numpy()

            #Check if counterfactual is effective
            if success_count / total_individuals >= effectiveness:
                nb_of_effective_actions += 1
                self.used_cf[first_index] = 1 # set only the first occurrence of repeated CF to 1
                
        return nb_of_effective_actions   
    
    
    # Apply a single counterfactual from the set of counterfactuals to the subgroup
    def apply_counterfactual(self, group_tensor, counterfactual):
        
        # find indices of features to change
        feature_indices = [self.dataset_features.index(feature) for feature in self.cf_features]
        
        # filter the dataset to only features to change
        selected_features = tf.gather(group_tensor, feature_indices, axis=1)
        
        # Prepare a mask for continuous and categorical features
        feature_types_tensor = tf.constant(
            [1 if ft == 'con' else 0 for ft in self.features_types], dtype=group_tensor.dtype
        )  # 1 for continuous, 0 for categorical
        
        # categorical features that require change (not equal to state_mins which denote no change for categorical feature)
        categorical_mask = tf.logical_and(
                feature_types_tensor == 0, # Identify categorical features
                tf.not_equal(counterfactual, tf.constant(self.state_mins, dtype=counterfactual.dtype)) 
        )
        
        
        # Split counterfactuals for continuous and categorical processing
        continuous_counterfactual = counterfactual * tf.cast(feature_types_tensor, dtype = counterfactual.dtype)  # Counterfactual values for continuous features
        categorical_counterfactual = tf.where(
            categorical_mask, # Apply ony where the counterfactual is valid (change required for categorical features)
            counterfactual,
            selected_features, # Leave unchanged otherwise
        )
        
        # Apply modifications: add for continuous and set for categorical
        modified_features = selected_features + continuous_counterfactual # Apply continuous modifications
        modified_features = tf.where(categorical_mask, categorical_counterfactual, modified_features)  # Apply categorical modifications
        
        # apply the modifications to the dataset
        # modified_features = selected_features + counterfactual
        
        # Clip the modified features to ensure they stay within the min-max bounds for each feature
        # Filter min and max values for the corresponding features
        filtered_mins = [self.mins[index] for index, value in enumerate(self.cf_features)]
        filtered_maxs = [self.maxs[index] for index, value in enumerate(self.cf_features)]
      
        # Convert to tensors for broadcasting
        filtered_mins_tensor = tf.constant(filtered_mins, dtype=group_tensor.dtype)
        filtered_maxs_tensor = tf.constant(filtered_maxs, dtype=group_tensor.dtype)
       
        # Clip the modified features to stay within the bounds
        modified_features = tf.clip_by_value(modified_features, filtered_mins_tensor, filtered_maxs_tensor)
        
        updated_group_tensor = group_tensor
    
        # Iterate over each feature index and apply the changes
        for i, feature_index in enumerate(feature_indices):
            # Gather the updates for the specific feature
            update_indices = tf.expand_dims(tf.range(tf.shape(group_tensor)[0]), axis=1)
            update_indices = tf.concat([update_indices, tf.fill([tf.shape(group_tensor)[0], 1], feature_index)], axis=1)
            
            # Update the group tensor with the modified feature
            updated_group_tensor = tf.tensor_scatter_nd_update(
                updated_group_tensor,   
                update_indices, 
                modified_features[:, i]
            )
        
        return updated_group_tensor
    
    # used to test the outcome of individuals
    def test_outcome(self, group_tensor):
        
        if len(group_tensor.shape) == 1:
            group_tensor = tf.reshape(group_tensor, (1, -1))
        if self.model_type == 'sklearn':
            predictions = self.model.predict(pd.DataFrame(group_tensor.numpy(), columns=self.dataset_features))
            if self.desired_label != None:
                return predictions == self.desired_label
            return predictions != self.initial_label
        else:
            predictions = tf.argmax(self.model(group_tensor), axis=1)
            if self.desired_label:
                return predictions == self.desired_label
            return predictions != self.initial_label

    '''
    def compute_gower_distance(self, dataset, counterfactuals):
        
        # Convert dataset and counterfactuals to tensors
        dataset_tensor = tf.convert_to_tensor(dataset, dtype=tf.float32)
        counterfactuals_tensor = tf.convert_to_tensor(counterfactuals, dtype=tf.float32)

        # Get indices of features to change
        feature_indices = [self.dataset_features.index(feature) for feature in self.cf_features]
        
        # Initialize list to store Gower distances for each CF
        gower_means_per_cf = []

        for cf in counterfactuals_tensor:  # Process each CF
            # Initialize similarity measures
            similarities = []

            # Continuous features
            continuous_indices = [idx for idx, ft in enumerate(self.features_types) if ft == "con"]
            if continuous_indices:
                # Extract the relevant features for the CF (shape: (n_features_to_modify,))
                continuous_cf = tf.gather(cf, continuous_indices)  # Shape: (n_features_to_modify,)
                
                # Repeat the CF values for all individuals
                continuous_cf_repeated = tf.tile(tf.expand_dims(continuous_cf, axis=0), [tf.shape(dataset_tensor)[0], 1])  # Shape: (n_individuals, n_features_to_modify)
                
                # Normalize by dividing by the range (max - min) for each feature
                continuous_cf_normalized = tf.abs(continuous_cf_repeated) / tf.constant(
                    [self.maxs[idx] - self.mins[idx] for idx in continuous_indices], dtype=tf.float32
                )  # Normalized across individuals
                # Round to 4 decimal places
                continuous_similarities = tf.round(continuous_cf_normalized * 10000) / 10000
                similarities.append(continuous_similarities)
    
            # Ordinal features
            ordinal_indices = [idx for idx, ft in enumerate(self.features_types) if ft == "ord" and cf[idx] != self.state_mins[idx]]
            ordinal_dataset_indices = [di for idx, di in enumerate(feature_indices) if self.features_types[idx] == "ord"]
            if ordinal_indices:
                ordinal_diffs = tf.abs(tf.gather(cf, ordinal_indices) -
                                    tf.gather(dataset_tensor, ordinal_dataset_indices, axis=1))
                ordinal_similarities = ordinal_diffs / tf.constant(
                    [self.maxs[i] for i in ordinal_indices], dtype=tf.float32
                )
                similarities.append(ordinal_similarities)

            # Categorical features
            categorical_indices = [idx for idx, ft in enumerate(self.features_types) if ft == "cat" and cf[idx] != self.state_mins[idx]]
            categorical_dataset_indices = [di for idx, di in enumerate(feature_indices) if self.features_types[idx] == "cat"]
            if categorical_indices:
                categorical_similarities = tf.cast(
                    tf.gather(cf, categorical_indices) != tf.gather(dataset_tensor, categorical_dataset_indices, axis=1),
                    dtype=tf.float32
                )
                similarities.append(categorical_similarities)
                
            # Combine all similarity measures
            if similarities:
                combined_similarities = tf.round(tf.concat(similarities, axis=1) * 10000)/10000  # Shape: (n_individuals, n_features_to_modify)
                mean_similarities_per_individual = tf.round(tf.reduce_mean(combined_similarities, axis=1) * 10000)/10000  # Mean across features
                gower_mean_for_cf = tf.round(tf.reduce_mean(mean_similarities_per_individual) * 10000)/10000  # Mean across individuals
                gower_means_per_cf.append(gower_mean_for_cf)

        # Compute the mean Gower distance across all CFs
        if gower_means_per_cf:
            gower_means_per_cf = tf.stack(gower_means_per_cf)  # Shape: (n_cfs,)
            gower_mean = np.round(tf.reduce_mean(gower_means_per_cf).numpy() * 10000)/10000  # Mean across CFs
            return gower_mean
        else:
            raise ValueError("No valid features to process for Gower distance calculation.")
    '''
    
    def compute_gower_distance(self, dataset, counterfactuals):
        # Convert dataset to a tensor
        dataset_tensor = tf.convert_to_tensor(dataset, dtype=tf.float32)
        counterfactuals_tensor = tf.convert_to_tensor(counterfactuals, dtype=tf.float32)

        # Get indices of features to change
        feature_indices = [self.dataset_features.index(feature) for feature in self.cf_features]

        # Initialize list to store Gower distances for each counterfactual
        gower_means_per_cf = []

        for cf_index, cf in enumerate(counterfactuals_tensor):
            
            # Apply the counterfactual to the dataset
            updated_dataset_tensor = self.apply_counterfactual(dataset_tensor, cf)

            # Filter the original and updated dataset for the features being changed
            original_features = tf.gather(dataset_tensor, feature_indices, axis=1)
            updated_features = tf.gather(updated_dataset_tensor, feature_indices, axis=1)

            # Initialize similarity measures
            similarities = []

            # Continuous features
            continuous_indices = [idx for idx, ft in enumerate(self.features_types) if ft == "con"]
            if continuous_indices:
                continuous_original = tf.gather(original_features, continuous_indices, axis=1)
                continuous_updated = tf.gather(updated_features, continuous_indices, axis=1)

                # Compute the normalized difference
                ranges = tf.constant([
                    self.maxs[idx] - self.mins[idx] for idx in continuous_indices
                ], dtype=tf.float32)
                continuous_differences = tf.abs(continuous_updated - continuous_original) / ranges

                # Round to 4 decimal places
                continuous_similarities = tf.round(continuous_differences * 10000) / 10000
                similarities.append(continuous_similarities)

            # Ordinal features
            ordinal_indices = [idx for idx, ft in enumerate(self.features_types) if ft == "ord"]
            if ordinal_indices:
                ordinal_original = tf.gather(original_features, ordinal_indices, axis=1)
                ordinal_updated = tf.gather(updated_features, ordinal_indices, axis=1)

                # Define custom weights for ordinal features dynamically
                ordinal_distances = []
                for i, ord_idx in enumerate(ordinal_indices):
                    num_categories = int(self.maxs[ord_idx] - self.mins[ord_idx] + 1)  # Total categories for the ordinal feature

                    # Custom weights (e.g., [1, 2, 5] for Bachelors -> Masters -> PhD)
                    # You can modify these values per feature if needed.
                    custom_weights = tf.constant(
                        [1 + 2 * (k / (num_categories - 1)) for k in range(num_categories)],
                        dtype=tf.float32
                    )
                    
                    range_weights = tf.reduce_max(custom_weights) - tf.reduce_min(custom_weights)  # Normalization factor

                    # Map original and updated values to their corresponding weights
                    original_weighted = tf.gather(custom_weights, tf.cast(ordinal_original[:, i] - self.mins[ord_idx], tf.int32))
                    updated_weighted = tf.gather(custom_weights, tf.cast(ordinal_updated[:, i] - self.mins[ord_idx], tf.int32))

                    # Compute normalized weighted distance
                    distance = tf.abs(updated_weighted - original_weighted) / range_weights
                    ordinal_distances.append(distance)
                
                ordinal_similarities = tf.stack(ordinal_distances, axis=1)
                similarities.append(ordinal_similarities)

            # Categorical features
            categorical_indices = [idx for idx, ft in enumerate(self.features_types) if ft == "cat"]
            if categorical_indices:
                categorical_original = tf.gather(original_features, categorical_indices, axis=1)
                categorical_updated = tf.gather(updated_features, categorical_indices, axis=1)

                # Compute categorical similarities (1 if different, 0 if same)
                categorical_similarities = tf.cast(
                    tf.not_equal(categorical_updated, categorical_original), dtype=tf.float32
                )
                similarities.append(categorical_similarities)

            # Combine all similarity measures
            if similarities:
                combined_similarities = tf.concat(similarities, axis=1)  # Shape: (n_individuals, n_features_to_modify)
                mean_similarities_per_individual = tf.reduce_mean(combined_similarities, axis=1)  # Mean across features
                gower_mean_for_cf = tf.reduce_mean(mean_similarities_per_individual)  # Mean across individuals
                gower_means_per_cf.append(gower_mean_for_cf)

        # Compute the mean Gower distance across all counterfactuals
        if gower_means_per_cf:
            gower_means_per_cf = tf.stack(gower_means_per_cf)  # Shape: (n_cfs,)
            np.round(tf.reduce_mean(gower_means_per_cf).numpy() * 10000)/10000
            gower_mean = np.round(tf.reduce_mean(gower_means_per_cf).numpy() * 10000)/10000  # Mean across counterfactuals
            return gower_mean
        else:
            return 0

    
    def max_gower_distance(self, dataset):
        
        self.used_cf = [1] * self.nb_cf
        # Get feature indices for categorical features
        feature_indices = [self.dataset_features.index(feature) for feature in self.cf_features]

        # Identify least common categories for categorical features
        least_common_categories = {}
        dataset_tensor = tf.convert_to_tensor(dataset, dtype=tf.float32)

        for idx, feature_type in enumerate(self.features_types):
            if feature_type == "cat" or feature_type == "ord":
                dataset_column = dataset_tensor[:, feature_indices[idx]]
                unique, _, counts = tf.unique_with_counts(dataset_column)
                least_common_category = unique[tf.argmin(counts)]
                least_common_categories[idx] = least_common_category.numpy()

        # Construct max and min states
        max_state = [
          self.state_maxs[idx] if ft == "con" 
          else self.maxs[idx] if ft == "ord"
          else least_common_categories[idx]
          for idx, ft in enumerate(self.features_types)
        ]
        
        min_state = [
            self.state_mins[idx] if ft == "con" 
            else self.maxs[idx] if ft == "ord"
            else least_common_categories[idx]
            for idx, ft in enumerate(self.features_types)
        ]

        # Duplicate states for number of counterfactuals
        max_state = [max_state] * self.nb_cf
        min_state = [min_state] * self.nb_cf

        # Compute gower distance for max and min states
        max_state_gower = self.compute_gower_distance(dataset, max_state)
        min_state_gower = self.compute_gower_distance(dataset, min_state)

        # Return maximum Gower distance
        return max(max_state_gower, min_state_gower)
    
    '''
    def reset(self, seed=None, options=None):
        self.used_cf = [0] * self.nb_cf
        self.current_step = 0
        self.nb_episode += 1
        #perturbation_percentage = max(1, 20 - (self.nb_episode // 100))
        
        #used to ensure we start from the pre-set state (initial state that we want to start from)
        if self.first_step == 0:
            self.first_step +=1
            print("$$$$$$$$$$$$$$$$$$$")
            print(f"RESET STATE: {self.state.get_counterfactuals()}")
            print("-----ENV RESET-----")
            print("$$$$$$$$$$$$$$$$$$$")
            return np.array(self.state.get_counterfactuals()), {}
        
        # get the best state from previous episode
        old_state = list(self.best_episode_state.values())[0]
        
        if self.done:
            # if goal reached, we get the best state from previous episode and apply [-10%, 10%] perturbation to it and reset the environment
            old_state = np.array(list(self.best_episode_state.values())[0])
            perturbation_percentage = np.random.randint(-10, 10, size=self.observation_space.shape)/100
            perturbation = old_state * perturbation_percentage
            state = []
            for i, val in enumerate(old_state):
                feature_type = self.features_types[i % len(self.features_types)]
                if feature_type == 'con':
                    state.append(val + perturbation[i]) # Continuous feature: add perturbation
                elif feature_type == 'cat' or feature_type == "ord":
                    if perturbation_percentage[i] > 0: 
                        state.append(val + 1) # Increment category value
                    elif perturbation_percentage[i] < 0:
                        state.append(val - 1) # Decrement category value
                    else:
                        state.append(val) # No change in category
                else:
                    raise ValueError(f"Unknown feature type: {feature_type}")
                
            state = [np.clip(val, self.state_mins[i % len(self.state_mins)], self.state_maxs[i % len(self.state_maxs)]) for i, val in enumerate(state)]
            state = [round(val) if (self.features_types[i % len(self.features_types)] == 'con' and self.features_dtypes[i % len(self.features_dtypes)] == 'int') else val for i, val in enumerate(state)]
        else:
            state = old_state # if goal not reached, we reset the environment to the best state reached in previous episode without applying perturbation
        
        new_state = GenSet("CF")
        new_state.set_counterfactuals(state)
        self.state = new_state
        self.best_episode_state.clear()
        print("$$$$$$$$$$$$$$$$$$$")
        print(f"BEFORE RESET STATE: {old_state}")
        print(f"RESET STATE: {self.state.get_counterfactuals()}")
        print("-----ENV RESET-----")
        print("$$$$$$$$$$$$$$$$$$$")
        return np.array(self.state.get_counterfactuals()), {}
    '''
    def reset(self, seed=None, options=None):
        self.used_cf = [0] * self.nb_cf
        self.current_step = 0
        self.initial_state_cfs = self.generate_state(random=False)
        self.initial_state = GenSet("CF")
        self.initial_state.set_counterfactuals(self.initial_state_cfs)
        self.state = self.initial_state
        return np.array(self.state.get_counterfactuals()), {}
    
    def _custom_rewrad(self, new_state): # Reward Function 
        reward = 0
        cf_reward = 0 # same as 'reward' but doesn't include step penalty, used only as a key to store successful sets of CFS
        done = False
        self.used_cf = [0] * self.nb_cf
        target_success_rate = 0.75 #(Compas-0.77) #(Alzheimer-0.79) # Target average success rate to stop the episode
        target_proportion_difference = 0.09 #(Compas-0.09) #(Alzheimer-0.09) # Target penalty to stop the episode (difference between proportions)
        target_group1_nb_actions = 1
        target_group2_nb_actions = 1
        target_nb_actions_difference = 0
        
        #Generate a list of counterfactuals from the state
        #Ex: if features_to_change = ["Income", "Credit", "Experience"] and number_of_counterfactuals = 2
        #A possible State Representation is [1000, 30, 1, 1200, 25, 1]
        #counterfactuals = [[1000, 30, 1], [1200, 25, 1]]
        cfs = new_state.get_counterfactuals()
        counterfactuals = [cfs[i * len(self.cf_features):(i+1) * len(self.cf_features)] for i in range(self.nb_cf)]
        
        #count_non_active_cfs = self.count_non_active_cfs(counterfactuals)
        #count_active_cfs = self.nb_cf - count_non_active_cfs
        #normalized_count_non_active_cfs = count_non_active_cfs/self.nb_cf
        #normalized_count_active_cfs = count_active_cfs/self.nb_cf
        
        print(f"counterfactuals listed {counterfactuals}")
        #print(f"number of non active CFs  {count_non_active_cfs}")
        
        if self.fairness_metrics == 'EF' or self.fairness_metrics == 'EF-ECR':
            # Fairness Metric 1 'Equal Effectiveness' : Average Success Rate with Penalty for Wide Proportion Differences
            # avergage_success_rate has a range [0, 1]
            # proportions_difference has a range [0, 1]
            group1_proportion, group2_proportion = self.evaluate_fairness_metric1(counterfactuals, self.macro)
            average_success_rate = float(f"{(group1_proportion + group2_proportion) / 2: .2f}")

            proportions_difference = round(abs(group1_proportion - group2_proportion), 2)
        
            print("------------------------Equal Effectiveness------------------------")
            print(f"Group 1 Proportion: {group1_proportion}")
            print(f"Group 2 Proportion: {group2_proportion}")
            print(f"Average Success Rate: {average_success_rate}")
            print(f"Group Difference: {proportions_difference}")
            print("------------------------Equal Effectiveness------------------------")
        
        if self.fairness_metrics == 'ECR' or self.fairness_metrics == 'EF-ECR':
            # Fairness Metric 2 'Equal Choice of Recourse'
            nb_actions_group1, nb_actions_group2 = self.evaluate_fairness_metric2(counterfactuals, self.action_effectiveness)
            nb_actions_group1 = float(f"{nb_actions_group1/self.nb_cf: .2f}")
            nb_actions_group2 = float(f"{nb_actions_group2/self.nb_cf: .2f}")
            average_nb_actions = float(f"{(nb_actions_group1 + nb_actions_group2) / 2: .2f}")
            nb_actions_difference = float(f"{abs(nb_actions_group1 - nb_actions_group2): .2f}")
            
            print("------------------------Equal Choice of Recourse------------------------")
            print(f"Number of Actions for Group 1: {nb_actions_group1*self.nb_cf}")
            print(f"Number of Actions for Group 2: {nb_actions_group2*self.nb_cf} ")
            print(f"Average Number of Effective Actions: {average_nb_actions*self.nb_cf}")
            print(f"Effective Actions Difference : {nb_actions_difference*self.nb_cf} ")
            print("------------------------Equal Choice of Recourse------------------------")
        
        # used counterfactuals: CFs that could be used to achieve recourse
        active_counterfactuals = [cf for cf, used in zip(counterfactuals, self.used_cf) if used == 1]
        # unused counterfactuals
        non_active_counterfactuals = [cf for cf, used in zip(counterfactuals, self.used_cf) if used == 0]
        
        # gower_distance_mean to penalize high deviated values
        gower_active_cfs = self.compute_gower_distance(self.affected_dataset, active_counterfactuals)
        gower_non_active_cfs = self.compute_gower_distance(self.affected_dataset, non_active_counterfactuals)
        
        if self.max_gower >= 1:
            normalized_gower_active_cfs = gower_active_cfs / self.max_gower
            normalized_gower_non_active_cfs = gower_non_active_cfs / self.max_gower
        else:
            normalized_gower_active_cfs = gower_active_cfs
            normalized_gower_non_active_cfs = gower_non_active_cfs
        
        count_active_cfs = len(active_counterfactuals)
        normalized_count_active_cfs = count_active_cfs / self.nb_cf
        count_non_active_cfs = self.nb_cf - count_active_cfs
        normalized_count_non_active_cfs = count_non_active_cfs / self.nb_cf
        print(f"Number of Active CFs  {count_active_cfs}")
        
        print("-----------------------------")
        print(f"Gower Distance For Active CFs: {gower_active_cfs}")
        print(f"Gower Distance For Non Active CFs: {gower_non_active_cfs}")
        print(f"Used CFs: {self.used_cf}")
        print("-----------------------------")
        
        if self.fairness_metrics == "EF":
            # For Equal Effectiveness Fairness Metric
            reward += (average_success_rate * 1.1 + normalized_count_active_cfs - proportions_difference) * 50 # (1.1 - 0.7 - 1 - 0.8 - 1)
            reward -= (normalized_gower_non_active_cfs * 0.8 + normalized_gower_active_cfs) * 50
            # Determine if the episode is done
            done = (average_success_rate >= target_success_rate) and (proportions_difference <= target_proportion_difference)
        elif self.fairness_metrics == "ECR":
            # For Equal Choice for Recourse Fairness Metric
            reward += (nb_actions_group1 * 2.5 + nb_actions_group2 * 2.5 + normalized_count_active_cfs - nb_actions_difference) * 50 # 2.5 2.5 0.7 0.5 0.9
            reward -= (normalized_gower_non_active_cfs * 0.8 + normalized_gower_active_cfs) * 50
            # Determine if the episode is done
            done = (nb_actions_group1 * self.nb_cf >= target_group1_nb_actions) and (nb_actions_group2 * self.nb_cf >= target_group2_nb_actions) and (nb_actions_difference <= target_nb_actions_difference)
        elif self.fairness_metrics == "EF-ECR":
            # For Equal Effectiveness and Equal Choice for Recourse Fairness Metrics
            reward += (average_success_rate * 1.1 + normalized_count_active_cfs - proportions_difference) * 50 
            reward += (nb_actions_group1 * 1.5 + nb_actions_group2 * 1.5 - nb_actions_difference) * 50
            reward -= (normalized_gower_non_active_cfs * 0.8 + normalized_gower_active_cfs) * 50
            # Determine if the episode is done
            done = (nb_actions_group1 * self.nb_cf >= target_group1_nb_actions) and (nb_actions_group2 * self.nb_cf >= target_group2_nb_actions) and (nb_actions_difference <= target_nb_actions_difference) and (average_success_rate >= target_success_rate) and (proportions_difference <= target_proportion_difference)

        cf_reward = round(reward,2)
        # Penalize Agent for taking more steps
        #step_penalty = (-0.5 * self.current_step/self.steps_per_episode)*100
        #reward += step_penalty
        
        if done:
            reward += self.final_reward
            cf_reward += self.final_reward
            print("****************************************************")
            print("****************************************************")
            print("**********************DONE**************************")
            print("****************************************************")
            print("****************************************************")
            self.reward_count += 1
            
            # store successful CF set
            if self.fairness_metrics == "EF":
                self.add_counterfactual(cf_reward, self.reward_count, group1_proportion, group2_proportion, average_success_rate, proportions_difference, normalized_gower_active_cfs, counterfactuals=active_counterfactuals)
            elif self.fairness_metrics == "ECR":
                self.add_counterfactual(cf_reward, self.reward_count, nb_actions_group1 * self.nb_cf, nb_actions_group2 * self.nb_cf, average_nb_actions * self.nb_cf, nb_actions_difference * self.nb_cf, normalized_gower_active_cfs, counterfactuals=active_counterfactuals)
            elif self.fairness_metrics == "EF-ECR":
                self.add_counterfactual(cf_reward, self.reward_count, group1_proportion, group2_proportion, average_success_rate, proportions_difference, nb_actions_group1 * self.nb_cf, nb_actions_group2 * self.nb_cf, average_nb_actions * self.nb_cf, nb_actions_difference * self.nb_cf, normalized_gower_active_cfs, counterfactuals=active_counterfactuals)
        return reward, done  
    
    #----------------------------------Helper Functions to Store Successful CF Set-----------------------------------------  
    def add_counterfactual(self, cf_reward, reward_count, metric1=None, metric2=None, metric3=None, metric4=None, metric5=None, metric6=None, metric7=None, metric8=None, metric9=None, counterfactuals=[]):
        
        if (self.fairness_metrics == "EF" or self.fairness_metrics == "ECR"):
            modified_reward = f"{cf_reward}-{reward_count}({metric1})({metric2})({metric3})({metric4})({metric5})"
        elif self.fairness_metrics == "EF-ECR":
            modified_reward = f"{cf_reward}-{reward_count}({metric1})({metric2})({metric3})({metric4})({metric5})({metric6})({metric7})({metric8})({metric9})"
            
        new_counterfactuals_tuple = self.convert_to_tuple(counterfactuals)
        # Check if the new counterfactual tuple already exists in the set
        if new_counterfactuals_tuple in self.counterfactual_set:
            return
        
        # If it's unique, add it to the dictionary and the set
        self.counterfactuals[modified_reward] = counterfactuals
        self.counterfactual_set.add(new_counterfactuals_tuple)

    def convert_to_tuple(self, counterfactuals):
        return tuple(tuple(sublist) for sublist in counterfactuals)
    #----------------------------------Helper Functions to Store Successful CF Set-----------------------------------------  
              
    # used to get the successful sets of CFs at the end of training   
    def return_counterfactuals(self):
        return self.counterfactuals
    
    def _take_action(self, action):
        values = self.state.get_counterfactuals()
        new_state = GenSet(self.state.name)
        new_state.set_counterfactuals(copy.deepcopy(values))
        
        index_to_change = round(action[0]) 
        amount_of_change = action[1] #round(action[1], 2)
        
        info = {}
        
        min_val = self.state_mins[index_to_change % len(self.state_mins)]
        max_val = self.state_maxs[index_to_change % len(self.state_maxs)]
        feature_dtype = self.features_dtypes[index_to_change % len(self.features_dtypes)]
        feature_type = self.features_types[index_to_change % len(self.features_types)]
        
        old_value = values[index_to_change]
        un_normalized_change = ((amount_of_change + 100) / 200) * (max_val - min_val) + min_val
    
        
        # modify state index value (index_to_change) by a specific % (amount_of_change)
        new_state.modify_feature(index_to_change, amount_of_change, min_val, max_val, feature_dtype, feature_type)
        if feature_type == 'con':
            info["info"] = 'Modifying old value of {' + str(self.cf_features[index_to_change%len(self.cf_features)]) + '} Feature - Index: ' + str(index_to_change) + ' by amount = ' + str(un_normalized_change) + ' -- New Value = ' + str(new_state.get_counterfactuals()[index_to_change])
        elif amount_of_change > 0:
            info["info"] = 'Modifying old value of {' + str(self.cf_features[index_to_change%len(self.cf_features)]) + '} Feature - Index: ' + str(index_to_change) + ' by amount = +1 -- New Value = ' + str(new_state.get_counterfactuals()[index_to_change])
        elif amount_of_change < 0:
            info["info"] = 'Modifying old value of {' + str(self.cf_features[index_to_change%len(self.cf_features)]) + '} Feature - Index: ' + str(index_to_change) + ' by amount = -1 -- New Value = ' + str(new_state.get_counterfactuals()[index_to_change]) 
        else:
            info["info"] = 'No Modification' 
        
        print("####################################################")
        print("Old State: " + str(self.state.get_counterfactuals()))
        print("----------------------------------------------------")
        print(info["info"])
        print("----------------------------------------------------")
        print("New State: " + str(new_state.get_counterfactuals()))
        print("----------------------------------------------------")
        
        reward, done = self._custom_rewrad(new_state)
            
        print("***---------------************-----------------***") 
        print("Reward: " + str(reward))
        print("Done: " + str(done)) 
        print("***---------------************-----------------***")  
        print("####################################################")
        return new_state, reward, done, info
    
    def step(self, action):
        # Execute one time step within the environment
        terminated = False
        truncated = False
        
        self.current_step += 1
        self.total_steps +=1
        new_state, reward, self.done, info = self._take_action(action)
        
        # Keep track of best state in the episode
        if not self.best_episode_state:
            self.best_episode_state[reward] = new_state.get_counterfactuals()
        elif reward > list(self.best_episode_state.keys())[0]:
            self.best_episode_state.clear()
            self.best_episode_state[reward] = new_state.get_counterfactuals()
        
        if self.done == True:
            terminated = True
            print("**********(Terminated)**********")
           
        if self.current_step > self.steps_per_episode and self.reward_count >= 10 and not terminated:
            truncated = True
            print("***********(Truncated)***********")

    
        self.state = new_state
        return np.array(new_state.get_counterfactuals()), reward, terminated, truncated, info
    
        
        