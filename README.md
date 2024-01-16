# gather training images

I asked dall-e 3 to create a few images. 

# train

The first arg to train.sh needs to be a dir in `projects`.

```
./train.sh asteroids
```

This will take about 5m and create a weights & biases project, 
and upload the model to huggingface. So it needs an auth into both.

# sample

I created a custom sample py script per project since they have different 
heuristics on whether a given sample is good or not. 

```
python ship_sample.py
```