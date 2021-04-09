# Benchmarks for Custom Concepts

Currently supports:
- Google AutoML Vision
- Custom Concepts from Mobius Vision

## Google AutoML Vision

To run benchmark for Google AutoML Vision:

`GOOGLE_APPLICATION_CREDENTIALS="<path to google credentials>" python google_automl_benchmark.py --dataset "<concept_name>" --noise <noise level> --pos_limit <maximum_number_of_posiive_samples> --neg_limit <maximum_number_of_negaive_samples>`

For example:

`GOOGLE_APPLICATION_CREDENTIALS="cv-comparision-82d47423413.json" python google_automl_benchmark.py --dataset "pasta" --noise 0.0 --pos_limit 25 --neg_limit 25`


See https://cloud.google.com/vision/automl/docs/tutorial#configure_your_project_environment to learn more about Google Credentials.


## Custom Concepts from Mobius Vision

To run benchmark for Custom Concepts from Mobius Vision:

`python custom_concepts_benchmark.py --dataset "<concept_name>" --noise <noise level> --pos_limit <maximum_number_of_posiive_samples> --neg_limit <maximum_number_of_negaive_samples> --host "<URL_to_Mobius_SDK>"`

For example:

`python custom_concepts_benchmark.py --dataset "pasta" --noise 0.0 --pos_limit 100 --neg_limit 100 --host "127.0.0.1:5000"`

To learn more about Mobius Custom Concepts visit https://docs.mobius.ml/on_premise/latest/#section/Custom-Concepts.
