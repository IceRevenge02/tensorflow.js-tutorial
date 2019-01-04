function onload() {
    const a = createTensor(1, 2, 6, 255); //Values can never be changed
//    t.print();
    const b = createTensor(1, 6, 3, 255); //Values can never be changed

    const c = a.matMul(b);

    a.print();
    b.print();
    c.print();


    const d = createTensor(1, 2, 3, 255); //Values can never be changed
//    t.print();
    const e = createTensor(1, 2, 3, 255); //Values can never be changed

    const f = d.add(e);

    e.print();
    d.print();
    f.print();

    
//    console.log(t.dataSync());//in array

//    const v = tf.variable(t); //Values can be changed
//    v.print();
}

function createTensor(nr, rows, cols, max) {
    let values = [];
    for (let i = 0; i < nr * rows * cols; i++) {
        values[i] = Math.round(Math.random()*max);
    }

    const shape = [nr, rows, cols];

    const t = tf.tensor(values, shape, 'int32'); //values:[data], shape: [nr, rows, cols], dtype

    return t;
}