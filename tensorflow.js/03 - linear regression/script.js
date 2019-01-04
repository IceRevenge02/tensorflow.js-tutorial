let x_vals = [],
    y_vals = [],
    m, b;

const learningRate = 0.02,
    optimizer = tf.train.sgd(learningRate);

function setup() {
    createCanvas(400, 400);
    background(0);

    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
}

function loss(predY, actualY) {
    return predY.sub(actualY).square().mean();
}

function predict(x) {
    const xs = tf.tensor1d(x);
    //y = mx + b
    const ys = xs.mul(m).add(b);

    return ys;
}

function mousePressed() {
    if (mouseX < width && mouseY < height) {
        let x = map(mouseX, 0, width, 0, 1),
            y = map(mouseY, 0, height, 1, 0);

        x_vals.push(x);
        y_vals.push(y);
    }
}

function draw() {
    if (x_vals.length > 0) {
        //predict line
        tf.tidy(() => {
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(() => loss(predict(x_vals), ys));
        });
        
        background(0);

        stroke(255);
        strokeWeight(4);

        //draw points
        for (let i = 0; i < x_vals.length; i++) {
            let px = map(x_vals[i], 0, 1, 0, width),
                py = map(y_vals[i], 0, 1, height, 0);

            point(px, py);
        }
        
        //draw line
        const lineXArr = [0, 1],
            ys = tf.tidy(() => predict(lineXArr)),
            lineYArr = ys.dataSync();

        let x1 = map(lineXArr[0], 0, 1, 0, width),
            x2 = map(lineXArr[1], 0, 1, 0, width);
            
        let y1 = map(lineYArr[0], 0, 1, height, 0),
            y2 = map(lineYArr[1], 0, 1, height, 0);
        
        strokeWeight(2);
        line(x1, y1, x2, y2);
        
        ys.dispose();        
    }
}