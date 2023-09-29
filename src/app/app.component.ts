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
  ngOnInit() {
    this.trainNewModel();
  }
  async trainNewModel() {
    this.linearmodel = tf.sequential();
    this.linearmodel.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    this.linearmodel.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    const xs = tf.tensor1d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 50]);
    const ys = tf.tensor1d([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 100]);
    await this.linearmodel.fit(xs, ys);
    console.log('training is complete');
  }
}
