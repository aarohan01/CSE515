import numpy as np
import utils
import Mongo.mongo_query_np as mongo
import dimension_reduction as dr
import torch 
import os


class Task1() :


    def __init__(self):
        self.dataset, self.labelled_images = utils.initialise_project()
        self.component = None

    def get_image_vectors_and_label_ids(self, feature_model : str) -> np.ndarray :

        '''
        Gives the vectors and label ids for even and odd images 
        '''

        even_image_vectors = mongo.get_all_feature_descriptor(feature_model)
        even_image_vectors = utils.convert_higher_dims_to_2d(even_image_vectors)

        even_image_label_ids = np.zeros(len(even_image_vectors))    
        for i in range(len(even_image_vectors)) :
            _ , even_image_label_ids[i] = self.dataset[i*2]

        odd_image_vectors = utils.get_odd_image_feature_vectors(feature_model)
        if odd_image_vectors is None :
            return
        odd_image_vectors = utils.convert_higher_dims_to_2d(odd_image_vectors)
        odd_image_label_ids = np.zeros(len(odd_image_vectors))
        for i in range(len(odd_image_vectors)) :
            _ , odd_image_label_ids[i] = self.dataset[i*2+1]   


        return  even_image_vectors, even_image_label_ids, odd_image_vectors, odd_image_label_ids



    def get_label_representives(self, label_feature_model : str ) -> list :

        '''
        Gives the label representative vectors for each label 
        '''
        print(f"Getting label representativess...\n")
        label_representives = mongo.get_label_feature_descriptor(label_feature_model)

        return label_representives


    def get_image_vectors_by_label(self, even_image_vectors : np.ndarray = None, even_image_label_ids : np.ndarray = None ) -> list :

        '''
        Get image vectors by label
        '''
        unique_label_ids = np.unique(even_image_label_ids.astype(int))
        map_image_vectors_by_label = { label_id : [] for label_id in unique_label_ids }
        for label_id, image_vector in zip(even_image_label_ids, even_image_vectors):
            map_image_vectors_by_label[label_id].append(image_vector)
        image_vectors_by_label = [None] * len(map_image_vectors_by_label)
        for label_id, image_vectors in map_image_vectors_by_label.items() :
            #print(label_id)
            image_vectors_by_label[label_id] = np.vstack(image_vectors)
        return image_vectors_by_label

    def get_label_wise_latent_semantic_representives(self, image_vectors_by_label : list) -> list :

        label_wise_latent_semantic_representives = []
        print(f"Getting label representatives from label latent semantics...\n")
        for label_features in image_vectors_by_label :
            label_wise_latent_semantic_representives.append(utils.label_fv_kmediods(label_features))
            #label_wise_latent_semantic_representives.append(np.mean(label_features, axis=0))

        return label_wise_latent_semantic_representives


    def fit_transform(self, k : int, training_data : np.ndarray ) -> np.ndarray :

        '''
        Calculates the latent semantics on the training data and return the transformed matrix of training set
        '''
        print(f"Calculating latent sematics for label vectors...\n")
        #U, S, VT = dr.svd_old(training_data, k, False)
        U, S, VT = dr.svd(training_data, full_matrices=True )
        U = U[:,:k]
        VT = VT[:k,:]
        S = S[:k]
        self.component = VT
        if S.ndim >=2 :
            training_data_transformed = U @ S
        else :
            training_data_transformed = U @ np.diag(S) 
        return training_data_transformed


    def transform(self, testing_data : np.ndarray) -> np.ndarray :

        '''
        Performming change of basis for odd image vectors for testing set
        '''
        print(f"Transforming odd image vector to new basis vectors...\n")
        testing_data_transformed =  testing_data @ self.component.T
        return  testing_data_transformed


    def get_predictions(self, label_matrix : np.ndarray, odd_image_matrix : np.ndarray, metric : str) -> np.ndarray :

        '''
        Calculating distances and predicting labels
        '''
        print(f"Calculating distances...\n")
        if metric == 'euclidean' :
            distance_matrix = utils.euclidean_distance_matrix(label_matrix, odd_image_matrix)
        elif metric == 'cosine' :
            distance_matrix = utils.cosine_distance_matrix(label_matrix, odd_image_matrix)

        odd_image_predicted_label_ids = np.argmin(distance_matrix, axis=0)
        return odd_image_predicted_label_ids


    def test_and_print(self,odd_image_label_ids : np.ndarray, odd_image_predicted_label_ids : np.ndarray)  :
        
        '''
        Results
        '''
        #Test 
        print(f"Calculating scores for predictions...")
        precision, recall, f1, accuracy  = utils.compute_scores(odd_image_label_ids, odd_image_predicted_label_ids, avg_type=None, values=True)

        #Display results
        utils.print_scores_per_label(self.dataset, precision, recall, f1, accuracy,'Task 1')


    def runTask1(self, case : int, k=False) :

        '''
        Main task function.
        '''
        
        print(f"\nEnter the value of k for obtaining latent semantics : ")
        if not k :
            k = utils.int_input()

        #option 5 fc layer 
        self.option = 5
        even_image_vectors, even_image_label_ids, odd_image_vectors, odd_image_label_ids = self.get_image_vectors_and_label_ids(utils.feature_model[self.option])
        label_representives = self.get_label_representives(utils.label_feature_model[self.option])
        
        match case :

            case 1 :
                '''
                Get label vectors from even images, create latent semantics, transform odd images and classify according to label vectors and transformed odd images
                '''

                label_representives_transformed = self.fit_transform(k, label_representives)
                odd_image_vectors_transformed = self.transform(odd_image_vectors)

            case 2 :

                '''
                Get even image vectors, create latent sematics, create label vectors, transform odd images,  classify according to label vectors and transformed odd images
                '''
                even_image_vectors_transformed = self.fit_transform(k,even_image_vectors)
                odd_image_vectors_transformed = self.transform(odd_image_vectors)
                even_image_vectors_by_label = self.get_image_vectors_by_label(even_image_vectors_transformed, even_image_label_ids)
                label_representives_transformed = self.get_label_wise_latent_semantic_representives(even_image_vectors_by_label)
                label_representives_transformed = np.vstack(label_representives_transformed)

            case 3 :

                '''
                Get even image vectors per label, do 101 latent semantics conversions and 101 odd image transformation, create label embeddings and distance matrix
                '''
                distance_matrix_file = f"{k}_distance_matrix.pkl"
                file1 = f"{k}_transformed_label_specific_semantics_matrices.pkl"
                file2 = f"{k}_transformed_odd_image_matrices_by_label.pkl"
                file3 = f"{k}_transformed_label_specific_semantics_embeddings.pkl"

                if not (os.path.exists(distance_matrix_file)) :


                    if not (os.path.exists(file1) and os.path.exists(file2) and os.path.exists(file3)) : 
                        #Even images by label 
                        even_image_vectors_by_label = self.get_image_vectors_by_label(even_image_vectors, even_image_label_ids)
                        #Contains U @ S per label matrix 
                        transformed_label_specific_semantics_matrices = []
                        #Projections of all odd images in each label specific matrix 
                        transformed_odd_image_matrices_by_label = []
                        for label_id, label_matrix in enumerate(even_image_vectors_by_label) :
                            print(f"Label ID : {label_id}")
                            transformed_label_specific_semantics_matrices.append( self.fit_transform(k, label_matrix))
                            transformed_odd_image_matrices_by_label.append(self.transform(odd_image_vectors))
                        #Combines transformed_label_specific_semantics_matrices using kmediods to get a vector for each label 
                        transformed_label_specific_semantics_embeddings = [utils.label_fv_kmediods(l) for l in transformed_label_specific_semantics_matrices ]   

                        print('Saving transformation.....')
                        torch.save(transformed_label_specific_semantics_matrices, file1)
                        torch.save(transformed_odd_image_matrices_by_label, file2)
                        torch.save(transformed_label_specific_semantics_embeddings, file3)

                    else :
                        transformed_label_specific_semantics_matrices = torch.load(file1)
                        transformed_odd_image_matrices_by_label = torch.load(file2)
                        transformed_label_specific_semantics_embeddings = torch.load(file3)


                    #Converting the label vectors into a single matrix for 101 com
                    #label_embeddings = np.vstack(transformed_label_specific_semantics_embeddings)
                    distance_matrix = []
                    print('Creating distance matrix....')
                    for label_id, label_embedding in enumerate(transformed_label_specific_semantics_embeddings) :
                        distance_matrix.append(utils.euclidean_distance_matrix(label_embedding, transformed_odd_image_matrices_by_label[label_id]))
                    distance_matrix_stacked = np.vstack(distance_matrix)
                    print(f"Saving distance matrix....")
                    torch.save(distance_matrix_stacked, distance_matrix_file)
                
                else :

                    distance_matrix_stacked =  torch.load(distance_matrix_file)


        if case == 1 or case == 2 :
            odd_image_label_ids_predicted = self.get_predictions(label_representives_transformed, odd_image_vectors_transformed, 'euclidean')
        else :
            odd_image_label_ids_predicted = np.argmin(distance_matrix_stacked, axis=0)
        
        self.test_and_print(odd_image_label_ids, odd_image_label_ids_predicted)
    
        ### DISPLAY PREDICTED LABEL PROMPT ###
        map_odd_image_id_predicted_label = {(index*2)+1 : label_id for index, label_id in enumerate(odd_image_label_ids_predicted)}

        while True :
            user_input = utils.get_user_input_odd_image_id_looped(self.dataset)
            if user_input == 'x' :
                return
            else :
                predicted_label_id = map_odd_image_id_predicted_label[user_input]
                predicted_label = self.dataset.categories[predicted_label_id]
                print(f"The predicted label for image id - {user_input} is : {predicted_label}")
                utils.display_image_and_labels(self.dataset, user_input, predicted_label)

            
if __name__ == '__main__':
    task1 = Task1()
    task1.runTask1(3)

    
