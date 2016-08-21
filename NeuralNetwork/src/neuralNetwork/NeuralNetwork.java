/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralNetwork;

import java.util.Arrays;

/**
 *
 * @author Matias Leone
 */
public class NeuralNetwork {

    private final static int INPUT_LAYER_SIZE = 400;
    private final static int INTERNAL_LAYER_SIZE = 25;
    private final static int OUTPUT_LAYER_SIZE = 10;
    private final static float RAND_EPSILON = 0.12f;
    
    private int inputLength;
    private float[][] theta0;
    private float[][] theta1;
    private float[] z1;
    private float[] a1;
    private float[] z2;
    private float[] a2;
    private float[] delta1;
    private float[] delta2;
    private float[] y;
    private float[][] theta0Grad;
    private float[][] theta1Grad;
    
    /*
        Theta0 has size 25 x 401
        Theta1 has size 10 x 26
    */
    
    public void init() {
        theta0 = new float[INTERNAL_LAYER_SIZE][INPUT_LAYER_SIZE + 1];
        theta1 = new float[OUTPUT_LAYER_SIZE][INTERNAL_LAYER_SIZE + 1];
        z1 = new float[INTERNAL_LAYER_SIZE];
        a1 = new float[INTERNAL_LAYER_SIZE];
        z2 = new float[OUTPUT_LAYER_SIZE];
        a2 = new float[OUTPUT_LAYER_SIZE];
        delta1 = new float[INTERNAL_LAYER_SIZE + 1];
        delta2 = new float[OUTPUT_LAYER_SIZE];
        y = new float[OUTPUT_LAYER_SIZE];
        theta0Grad = new float[INTERNAL_LAYER_SIZE][INPUT_LAYER_SIZE + 1];
        theta1Grad = new float[OUTPUT_LAYER_SIZE][INTERNAL_LAYER_SIZE + 1];
        
        for (int i = 0; i < theta0.length; i++) {
            for (int j = 0; j < theta0[i].length; j++) {
                theta0[i][j] = (float)Math.random() * 2 * RAND_EPSILON - RAND_EPSILON;
            }
        }
        
        for (int i = 0; i < theta1.length; i++) {
            for (int j = 0; j < theta1[i].length; j++) {
                theta1[i][j] = (float)Math.random() * 2 * RAND_EPSILON - RAND_EPSILON;
            }
        }
        
        set(theta0Grad, 0);
        set(theta1Grad, 0);
    }
    
    public void train(float[] input, int inputClass) {
        //Feedforward pass
        feedForward(input);
        
        //Create y vector using class
        set1InIndex(y, inputClass);
        
        //Compute delta2 as: a2 - y
        for (int i = 0; i < delta2.length; i++) {
            delta2[i] = a2[i] - y[i];
        }
        
        //Compute delta1 as: theta1' * delta2 * sigmoidDeriv(z2)
        for (int i = 0; i < theta1.length; i++) {
            for (int j = 0; j < theta1[i].length; j++) {
                delta1[j] = theta1[i][j] * delta2[i] * sigmoidDeriv(z2[i]);
            }
        }
        
        
        
        //Accumulate theta1Grad: theta1Grad + delta2 * a1'
        for (int i = 0; i < theta1Grad.length; i++) {
            for (int j = 1; j < theta1Grad[i].length; j++) {
                theta1Grad[i][j] += delta2[i] * a1[j - 1];
            }
        }
        
        //Accumulate theta0Grad: theta0Grad + delta1 * a0'
        for (int i = 0; i < theta0Grad.length; i++) {
            for (int j = 1; j < theta0Grad[i].length; j++) {
                theta0Grad[i][j] += delta1[i + 1] * input[j - 1];
            }
        }        
        
        
    }
    
    public Result predict(float[] input) {
        //Feedforward pass
        feedForward(input);
        
        //Get index with largest output
        int index = maxIndex(a2); 
        
        return new Result(index, a2[index]);
    }
    
    
    private void feedForward(float[] input) {
        //z1
        for (int i = 0; i < z1.length; i++) {
            z1[i] = theta0[i][0] * 1;
        }
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < z1.length; j++) {
                z1[j] += theta0[j][i + 1] * input[i];
            }
        }
        
        //a1
        for (int i = 0; i < a1.length; i++) {
            a1[i] = sigmoid(z1[i]);
        }
        
        
        //z2
        for (int i = 0; i < z2.length; i++) {
            z2[i] = theta1[i][0] * 1;
        }
        for (int i = 0; i < a1.length; i++) {
            for (int j = 0; j < z2.length; j++) {
                z2[j] += theta1[j][i + 1] * a1[i];
            }
        }
        
        //a2
        for (int i = 0; i < a2.length; i++) {
            a2[i] = sigmoid(z2[i]);
        }
    }
    
    
    private float sigmoid(float z) {
        return 1 / (1 + (float)Math.exp(-z));
    }
    
    private float sigmoidDeriv(float z) {
        float g = sigmoid(z);
        return g * (1 - g);
    }
    
    private int maxIndex(float[] a) {
        float max = Float.MIN_VALUE;
        int idx = -1;
        for (int i = 0; i < a.length; i++) {
            if(a[i] > max) {
                max = a[i];
                idx = i;
            }
        }
        return idx;
    }
    
    private void set(float[] a, float v) {
        Arrays.fill(a, v);
    }
    
    private void set(float[][] m, float v) {
        for (int i = 0; i < m.length; i++) {
            set(m[i], v);
        }
    }
    
    private void set1InIndex(float[] y, int index) {
        set(y, 0);
        y[index] = 1;
    }
    
    public static class Result {
        public final int predictedClass;
        public final float confidence;

        public Result(int predictedClass, float confidence) {
            this.predictedClass = predictedClass;
            this.confidence = confidence;
        }
        
        
    }
    
}
