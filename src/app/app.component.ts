import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.sass'],
})
export class AppComponent implements OnInit {
  title = 'linear-regression-tf';
  linearmodel: tf.Sequential = tf.sequential();
  prediction: any = 0;
  ngOnInit() {
    this.trainNewModel();
  }
  async trainNewModel() {
    this.linearmodel = tf.sequential();
    this.linearmodel.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    this.linearmodel.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    const xs = tf.tensor1d([1,2]);
    const ys = tf.tensor1d([2,4]);
    await this.linearmodel.fit(xs, ys);
    console.log('training is complete');
  }
  predictResult(event: any) {
    const val = parseInt((<HTMLInputElement>event.target).value);
    console.log('predict result for ', val);
    const output = this.linearmodel.predict(tf.tensor2d([val], [1, 1])) as any;
    this.prediction = Array.from(output.dataSync())[0];
  }
}
