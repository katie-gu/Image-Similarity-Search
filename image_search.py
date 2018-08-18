from annoy import AnnoyIndex
import os
import sys
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
""" 2 functions:
    Image indexer: Extracts features from images using Inception model.
    Create LSH (hash code), and save it in the file. Create hashmap.
    Search: Query the ANNoy hashmap using LSH hash code to find similar images.
"""
class ImageIndexer:
    def create_graph(model_path):
        """Defines and loads the Inception graph model into memory."""
        with gfile.FastGFile(model_path, 'rb') as file:
            graph = tf.GraphDef()
            graph.ParseFromString(file.read())
            _ = tf.import_graph_def(graph, name='')


    def get_features_from_graph(graph, image_paths):
        """
            Input: graph,image path  Output: feature tensors
            Execute graph in a Tensorflow session to get the feature tensors.
        """
        with tf.Session() as sess:
            for i, image_path in enumerate(image_paths):
                conv_output_tensor = sess.graph.get_tensor_by_name('pool_3:0')
                data = gfile.FastGFile('rb', image_path).read()
                feature = sess.run(conv_output_tensor, {'DecodeJpeg/contents:0': data})
                features[i, :] = np.squeeze(feature)
            return features

    def image_indexer(model_path, image_paths, hash_table_file_path, image_hash_file_path):
        """Input: image path  Output: index
        Get features of image, and index them (similar to put for hashmap).
        """
        graph = create_graph(model_path)
        features = get_features_from_graph(graph, image_paths) #Get the feature set for all the images.
        hash_table = AnnoyIndex(len(features))
        image_hash_table = {}
        for i in len(image_paths): #Iterate through each image in image_path.
            hash_table.add_item(i, features[i]) #Insert Key: feature Value: integer corresponding to image path  into hashmap.
            image_hash_table[i] = image_paths[i]
        hash_table.build(90)
        hash_table.save(hash_table_file_path)
        pickle.dump(image_hash_table,  image_hash_file_path)

class ImageSearch:
    def similarity_search(image, num_closest_items, hash_table_file_path, image_hash_file_path):
        """Input: image  Output: a list of images similar to the input
           Get the feature set associated with the image.
           Use feature set to query the ANNoy hashmap.
        """
        graph = create_graph(model_path)
        features = get_features_from_graph(graph, image_path)
        hash_table =  AnnoyIndex(len(features))
        hash_table.load(hash_table_file_path)
        image_hash_table = pickle.load(image_hash_file_path)
        for i in xrange(len(features)):
            search_results = hash_table.get_nns_by_vector(features[i], num_closest_items, include_distances=True)
            file_path = image_hash_table[search_results] #Translate integer into image path.
            print(search_results)
