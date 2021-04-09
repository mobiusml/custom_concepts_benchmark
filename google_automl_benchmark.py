import os
import numpy as np
import json
import hashlib
import time
import tqdm
import pickle
import threading
import math
from concurrent.futures import ThreadPoolExecutor
import sklearn.metrics as sk_metrics
from google.cloud import automl
from google.cloud import storage
import argparse
from utils import load_dataset, get_metrics

def add_image_to_gc(img):
    img_id = hashlib.md5(img.encode()).hexdigest()

    blob = bucket.get_blob('img/%s.jpg' % img_id)
    if not blob:
        blob = bucket.blob('img/%s.jpg' % img_id)
        blob.upload_from_filename(filename=img)

    gc_path = blob.public_url.replace('https://storage.googleapis.com/', 'gs://')
    return gc_path

def add_images_to_gc(images):
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(add_image_to_gc, images))
    return results

def get_prediction(img):
    with open(img, "rb") as content_file:
        content = content_file.read()

    image = automl.Image(image_bytes=content)
    payload = automl.ExamplePayload(image=image)

    request = automl.PredictRequest(name=model_full_id, payload=payload, params={})
    response = prediction_client.predict(request=request)

    t = response.payload[0]

    if t.display_name == 'negative':
        return 1-t.classification.score
    else:
        return t.classification.score

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Benchmark Google AutoML Vision')

    parser.add_argument('--dataset', type=str, required=True, help="Concept name", dest='dataset_name')
    parser.add_argument('--noise', type=float, required=True, help="Noise ration", dest='noise_ratio')
    parser.add_argument('--pos_limit', type=int, help="The number of positive images", dest='pos_limit', default=10000)
    parser.add_argument('--neg_limit', type=int, help="The number of negative images", dest='neg_limit', default=10000)
    parser.add_argument('--project_id', type=str, default="cv-comparision", help="project_id", dest='project_id')

    args = parser.parse_args()

    project_id = args.project_id

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
    
    display_name = "%s_%d_%d_%d"%(data_config['dataset_name'],
                                  int(data_config['noise_ratio']*100),
                                  data_config['pos_limit'], data_config['neg_limit'])

    start_time = time.time()
    
    client = automl.AutoMlClient()

    project_location = f"projects/{project_id}/locations/us-central1"

    metadata = automl.ImageClassificationDatasetMetadata(
        classification_type=automl.ClassificationType.MULTICLASS
    )
    dataset = automl.Dataset(
        display_name=display_name,
        image_classification_dataset_metadata=metadata,
    )

    # Create a dataset with the dataset metadata in the region.
    response = client.create_dataset(parent=project_location, dataset=dataset)

    created_dataset = response.result()

    # Display the dataset information
    print("Dataset name: {}".format(created_dataset.name))
    print("Dataset id: {}".format(created_dataset.name.split("/")[-1]))

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(f"{project_id}-vcm")

    pos_train_images_gc = add_images_to_gc(data['pos_train_images'])
    neg_train_images_gc = add_images_to_gc(data['neg_train_images'])

    labels_csv = ''
    labels_csv += '\n'.join(['%s,positive'%item for item in pos_train_images_gc])
    labels_csv += '\n'.join(['%s,negative'%item for item in neg_train_images_gc])

    blob = bucket.blob(f'csv/{display_name}.csv')
    blob.upload_from_string(labels_csv)

    csv_path = blob.public_url.replace('https://storage.googleapis.com/', 'gs://')

    dataset_id = created_dataset.name.split("/")[-1]
    dataset_full_id = client.dataset_path(project_id, "us-central1", dataset_id)

    gcs_source = automl.GcsSource(input_uris=[csv_path])
    input_config = automl.InputConfig(gcs_source=gcs_source)

    # Import data from the input URI
    response = client.import_data(name=dataset_full_id, input_config=input_config)

    print("Processing import...")
    print("Data imported. {}".format(response.result()))

    start_training_time = time.time()
    
    project_location = f"projects/{project_id}/locations/us-central1"
    # Leave model unset to use the default base model provided by Google
    # train_budget_milli_node_hours: The actual train_cost will be equal or
    # less than this value.
    # https://cloud.google.com/automl/docs/reference/rpc/google.cloud.automl.v1#imageclassificationmodelmetadata
    metadata = automl.ImageClassificationModelMetadata(
        train_budget_milli_node_hours=8000
    )
    model = automl.Model(
        display_name=display_name,
        dataset_id=dataset_id,
        image_classification_model_metadata=metadata,
    )

    # Create a model with the model metadata in the region.
    response = client.create_model(parent=project_location, model=model)

    print("Training operation name: {}".format(response.operation.name))
    print("Training started...")

    model_id = response.result().name.split('/')[-1]
    
    start_deploying_time = time.time()
    
    model_full_id = client.model_path(project_id, "us-central1", model_id)

    response = client.deploy_model(name=model_full_id)
    response.result()

    start_prediction_time = time.time()
    
    prediction_client = automl.PredictionServiceClient()

    preds = []
    for img in tqdm.tqdm(data['test_images']):
        score = None
        while True:
            try:
                score = get_prediction(img)
                break
            except:
                time.sleep(5)
        preds.append(score)
    preds = np.array(preds)

    metrics = get_metrics(data['test_labels'], preds)

    results = {
        'model_full_id': model_full_id,
        'project_id': project_id,
        'dataset_id': dataset_id,
        'display_name': display_name,
        'data_config': data_config,
        'metrics': metrics,
        'preds': preds,
        'test_images': data['test_images'],
        'test_labels': data['test_labels'],
        'total_time': time.time() - start_time,
        'data_loading_time': start_training_time - start_time,
        'training_time': start_deploying_time - start_training_time,
        'deploying_time': start_prediction_time - start_deploying_time,
        'prediction_time': time.time() - start_prediction_time
    }
   
    with open('google_automl_results_%s.pkl'%display_name, 'wb') as f:
        pickle.dump(results, f)

    for k in metrics:
        print (f'{k}: {metrics[k]}')

    print ('total_time:', results['total_time'])