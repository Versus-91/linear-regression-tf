import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
declare let tfvis: any
interface Car {
  Horsepower: any;
  Name: string,
  Miles_per_Gallon: number
}
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.sass'],
})
export class AppComponent implements OnInit {
  title = 'linear-regression-tf';
  linearmodel: tf.Sequential = tf.sequential();
  prediction: any = 0;
  cars: any = []
  constructor(private http: HttpClient) {
  }

  ngOnInit() {
    // this.trainNewModel();
    this.loadCars()
  }
  loadCars() {
    this.http.get<Car[]>('https://storage.googleapis.com/tfjs-tutorials/carsData.json').subscribe((res) => {
      this.cars = res.map((car) => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower
      }))
      const values = this.cars.map((d: any) => ({
        x: d.horsepower,
        y: d.mpg,
      }));
      tfvis.render.scatterplot(
        { name: 'Horsepower v MPG' },
        { values: values },
        {
          xLabel: 'Horsepower',
          yLabel: 'MPG',
          height: 300
        }
      );
    })
  }
  async trainNewModel() {
    this.linearmodel = tf.sequential();
    this.linearmodel.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    this.linearmodel.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    const xs = tf.tensor1d([1, 2]);
    const ys = tf.tensor1d([2, 4]);
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
