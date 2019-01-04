const trainInputs_ = tf.tensor2d([
	[0, 0],
	[1, 0],
	[0, 1],
	[1, 1]
]);

const trainOutputs_ = tf.tensor2d([
	[0],
	[1],
	[1],
	[0]
]);

const model = tf.sequential();
let points = [], points_;

function setup() {
	createCanvas(100, 100);
	background(255);
	
	for (let x = 0; x < width; x++) {
		for (let y = 0; y < height; y++) {
			const x1 = map(x, 0, width, 0, 1),
				y1 = map(y, 0, height, 0, 1);

			points.push([x1, y1]);
		}
	}
	points_ = tf.tensor2d(points);
	
	model.add(tf.layers.dense({
		units: 16,
		inputShape: [2],
		activation: 'sigmoid'
	}));
	
	model.add(tf.layers.dense({
		units: 1,
		activation: 'sigmoid'
	}));
	
	
	const optimizer = tf.train.adam(0.25);
	
	model.compile({
		optimizer: optimizer,
		loss: 'meanSquaredError',
		bias: true
	});
	train();
}

function train() {
	trainModel().then(outputs => {
		console.log(outputs.history.loss[0]);
		train();
	});
}

async function trainModel() {
	let outputs = await model.fit(trainInputs_, trainOutputs_, {
		shuffle: true,
		epochs: 10
	});
	return outputs;
}

function draw() {
	tf.tidy(() => {
		const outputs_ = model.predict(tf.tensor2d(points)),
			colors = outputs_.dataSync();

		for (let i = 0; i < points.length; i++) {
			const x = map(points[i][0], 0, 1, 0, width),
				y = map(points[i][1], 0, 1, 0, height);
			strokeWeight(1);
			stroke(255*colors[i]);
			point(x, y);
		}
	});
}
