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

class MobiusCustomConcept():
    def __init__(self, concept_name, host):
        self.host = host
        self.concept_name = concept_name
        self.delete_custom_concept()
    
    def add_image_to_image_db(self, path):
        img_id = hashlib.md5(path.encode()).hexdigest()
        params = {'image_id': img_id}
        with open(path, 'rb') as image:
            r = requests.post(
                'http://%s/system/database/images/add'%(self.host),
                files={'data': image},
                data={'params': json.dumps(params)}
            )
            output = r.json()
            return output

    def add_images(self, images, class_type):
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(self.add_image_to_image_db, images))
        if len([item for item in results if item['status'] == 'error' and item['message'] != 'image_id_exists']) != 0:
            return False
        resp = custom_concept.assign_images_to_custom_concept(images, class_type)
        return resp['status'] == 'success'

    def assign_images_to_custom_concept(self, images, class_type):
        params = {
            "image_id_list" : [hashlib.md5(path.encode()).hexdigest() for path in images],
            "custom_concept_id": self.concept_name,
            "class": class_type
        }
        r = requests.post(
            'http://%s/image/tags/custom_concepts/assign'%(self.host),
            data={'params': json.dumps(params)}
        )
        return r.json()

    def predict(self, path):
        params = {'modules': ['tags/custom_concepts'],
             'tags': {'custom_concepts': {'custom_concept_id_list': [self.concept_name]}}}

        with open(path, 'rb') as image:
            r = requests.post(
                'http://%s/image/predict'%(self.host),
                files={'data': image},
                data={'params': json.dumps(params)}
            )
            output = r.json()
            return output['tags']['custom_concepts'][0]['score']

    def delete_custom_concept(self):
        params = {
            "custom_concept_id": self.concept_name
        }

        r = requests.post(
            'http://%s/image/tags/custom_concepts/delete'%(self.host),
            data={'params': json.dumps(params)}
        )
        return r.json()

    def train(self):
        params = {
            "custom_concept_id": self.concept_name
        }

        r = requests.post(
            'http://%s/image/tags/custom_concepts/train'%(self.host),
            data={'params': json.dumps(params)}
        )
        output = r.json()

        if output['status'] == 'error':
            return output

        task_id = output['task_id']

        params = {
            'task_id': task_id
        }
        task_status = 'ongoing'

        while(task_status == 'ongoing' or task_status == 'in_queue'):
            time.sleep(1.)
            r = requests.post(
                'http://%s/system/tasks/status/get'%(self.host),
                data={'params': json.dumps(params)}
            )
            output = r.json()
            task_status = output['status']

        return output['status'] == 'success'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Benchmark Custom Concepts from Mobius Vision')

    parser.add_argument('--dataset', type=str, required=True, help="Concept name", dest='dataset_name')
    parser.add_argument('--noise', type=float, required=True, help="Noise ration", dest='noise_ratio')
    parser.add_argument('--pos_limit', type=int, help="The number of positive images", dest='pos_limit', default=10000)
    parser.add_argument('--neg_limit', type=int, help="The number of negative images", dest='neg_limit', default=10000)
    parser.add_argument('--host', type=str, required=True, help="Mobius SDK URL", dest='host')

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

    custom_concept = MobiusCustomConcept(concept_name, args.host)

    is_success = custom_concept.add_images(data['pos_train_images'], 'positive')
    if not is_success:
        print ('error: cannot add images')
    is_success = custom_concept.add_images(data['neg_train_images'], 'negative')
    if not is_success:
        print ('error: cannot add images')
    is_success = custom_concept.train()
    if not is_success:
        print ('error: training failed')

    with ThreadPoolExecutor(max_workers=20) as executor:
        preds = list(executor.map(custom_concept.predict, data['test_images']))

    preds = np.array(preds)

    metrics = get_metrics(data['test_labels'], preds)

    for k in metrics:
        print (f'{k}: {metrics[k]}')