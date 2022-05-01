# CS7643-Project

## Data Source
Dataset: https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified
Format: ndjson
Example Parser: https://github.com/googlecreativelab/quickdraw-dataset/blob/master/examples/nodejs/simplified-parser.js
Useful Fields:
1) word: Category the player was prompted to draw.
2) drawing: A JSON array representing the vector drawing
3) recognized: Whether the word was recognized by the game.


Example ndjson:
{
	"word":"airplane",
	"recognized":true,
	"drawing":[
		[
			[x0,x1,x2,x3,....],
			[y1,y1,y2,y3]
		],
		[
			[207,207,210,221,238],
			[74,103,114,128,135]
		],
		[
			[119,107,76,70,49,39,60,93],
			[72,41,3,0,1,5,38,70]
		]
	]
}


We will pick 10 out of 345 labels and each with 100k+ dataset. We will randomly pick 80% as training set, 10% validation set, 10% test set.

Labels = [flower, apple, baseball, baskebtall, bird, book, bus, car, cat, dog]

## How to Visualize dataset
https://stackoverflow.com/questions/63375078/visualize-google-quickdraw-data


## RNN
To support batch processing, every dataset needs to have the same sequence length, so we can set a hard cutoff of 50 strokes. We will filter out data with more than 50 strokes. We will filter by recognized=true so that we can ensure the quality of the data is high.


## CNN
We will filter by recognized=true so that we can ensure the quality of the data is high.