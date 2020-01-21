# Productionized Iris Model

## What?
I trained a random forest to recognize irises and make predictions of the species based on some parameters

Then, I save the model to a pickle, and load it up in a flask app and expose a POST endpoint

## How?
### API
run `docker build -t iris_api:latest .`
then `docker run --rm -d -p 5000:5000 iris_api:latest`
then `curl -X POST localhost:5000/prediction -d '{"sepal_length": 5.7, "sepal_width": 3.8, "petal_length": 1.7, "petal_width": 0.3}'  -H "Content-Type:application/json"`

feel free to change up the numbers there.

### Notebook
Have jupyter installed, then run `jupyter notebook TrainModel.ipynb`

## Why?
partly for fun, partly to learn something, partly a jumping off point for other ML related things.
