import numpy as np
import json
import hashlib
import tqdm
import pickle
import time
import requests
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from utils import load_dataset, get_metrics
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

class ClarifaiCustomModel():
    def __init__(self, api_key, concept_name):
        self.concept_name = concept_name
        self.app = ClarifaiApp(api_key=api_key)
        try:
            self.model = self.app.models.get(self.concept_name)
        except:
            self.model = None
    
    def add_images(self, images, class_type):
        for chunk in chunks(images, 128):
            if class_type == 'positive':
                cl_images = [ClImage(filename=path, concepts=[self.concept_name]) for path in chunk]
            else:
                cl_images = [ClImage(filename=path, not_concepts=[self.concept_name]) for path in chunk]
            self.app.inputs.bulk_create_images(cl_images)

    def predict(self, path):
        response = self.model.predict_by_filename(path)
        try:
            score = response['outputs'][0]['data']['concepts'][0]['value']    
            return score
        except:
            print (response)
            return response

    def train(self):
        model = self.app.models.create(self.concept_name, concepts=[self.concept_name])
        r = model.train()
        self.model = self.app.models.get(self.concept_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Benchmark Custom Models from Clarifai')

    parser.add_argument('--dataset', type=str, required=True, help="Concept name", dest='dataset_name')
    parser.add_argument('--noise', type=float, required=True, help="Noise ration", dest='noise_ratio')
    parser.add_argument('--pos_limit', type=int, help="The number of positive images", dest='pos_limit', default=10000)
    parser.add_argument('--neg_limit', type=int, help="The number of negative images", dest='neg_limit', default=10000)
    parser.add_argument('--api_key', type=str, required=True, help="Clarifai API key", dest='api_key')

    args = parser.parse_args()

    data_config = {
        'dataset_name': args.dataset_name,
        'noise_ratio': args.noise_ratio,
        'pos_limit': args.pos_limit,
        'neg_limit': args.neg_limit,
    }

    data = load_dataset(data_config['dataset_name'],
                        noise_ratio=data_config.get('noise_ratio', 0.),
                        num_calib_pos=data_config.get('num_calib_pos', 0),
                        num_calib_neg=data_config.get('num_calib_neg', 0),
                        pos_limit=data_config.get('pos_limit', 100000),
                        neg_limit=data_config.get('neg_limit', 100000),
                        seed=42)

    concept_name = "%s_%d_%d_%d"%(data_config['dataset_name'],
                              int(data_config['noise_ratio']*100),
                              data_config['pos_limit'], data_config['neg_limit'])

    custom_concept = ClarifaiCustomModel(args.api_key, concept_name)

    start_time = time.time()
    custom_concept.add_images(data['pos_train_images'], 'positive')
    custom_concept.add_images(data['neg_train_images'], 'negative')

    start_training_time = time.time()
    custom_concept.train()

    start_prediction_time = time.time()
    preds = []
    for img in tqdm.tqdm(data['test_images']):
        preds.append(custom_concept.predict(img))
        
    preds = np.array(preds)

    metrics = get_metrics(data['test_labels'], preds)

    for k in metrics:
        print (f'{k}: {metrics[k]}')

    results = {
        'metrics': metrics,
        'data_config': data_config,
        'y_pred': preds,
        'test_images': data['test_images'],
        'test_labels': data['test_labels'],
        'total_time': time.time() - start_time,
        'data_loading_time': start_training_time - start_time,
        'training_time': start_prediction_time - start_training_time,
        'prediction_time': time.time() - start_prediction_time
    }

    with open('clarifai_custom_model_benchmark_results_%s.pkl'%concept_name, 'wb') as f:
        pickle.dump(results, f)