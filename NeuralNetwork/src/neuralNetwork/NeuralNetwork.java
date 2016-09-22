/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralNetwork;

import java.util.Arrays;
import java.util.List;

/**
 *
 * @author Matias Leone
 */
public class NeuralNetwork {

    private final static float RAND_EPSILON = 0.12f;
    
    private final int inputLayerSize;
    private final int internalLayerSize;
    private final int outputLayerSize;
    
    private final float[][] theta0;
    private final float[][] theta1;
    
    //Temp variables
    private final float[] z1;
    private final float[] a1;
    private final float[] z2;
    private final float[] a2;
    private final float[] delta1;
    private final float[] delta2;
    private final float[] y;
    
    /*
        Theta0 has size 25 x 401
        Theta1 has size 10 x 26
    */
    
    public NeuralNetwork(int inputLayerSize, int internalLayerSize, int outputLayerSize) {
        this.inputLayerSize = inputLayerSize;
        this.internalLayerSize = internalLayerSize;
        this.outputLayerSize = outputLayerSize;
        
        theta0 = new float[internalLayerSize][inputLayerSize + 1];
        theta1 = new float[outputLayerSize][internalLayerSize + 1];
        
        z1 = new float[internalLayerSize];
        a1 = new float[internalLayerSize];
        z2 = new float[outputLayerSize];
        a2 = new float[outputLayerSize];
        delta1 = new float[internalLayerSize + 1];
        delta2 = new float[outputLayerSize];
        y = new float[outputLayerSize];
    }
    
    
    
    
    public void train(List<float[]> input, int[] inputClass, int iterations, float alpha, float lambda) {
        //Init theta0 and theta1 with random values
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
        
        //Init theta temp variables
        float[][] theta0Temp = new float[internalLayerSize][inputLayerSize + 1];
        float[][] theta1Temp = new float[outputLayerSize][internalLayerSize + 1];
        set(theta0Temp, 0);
        set(theta1Temp, 0);
        
        //Init gradients
        float[][] theta0Grad = new float[internalLayerSize][inputLayerSize + 1];
        float[][] theta1Grad = new float[outputLayerSize][internalLayerSize + 1];
        set(theta0Grad, 0);
        set(theta1Grad, 0);
        
        
        //Perform gradient descent
        for (int n = 0; n < iterations; n++) {
            
            //Compute gradient with current theta
            computeGradient(input, inputClass, theta0, theta1, lambda, theta0Grad, theta1Grad);
            
            //Update theta0 in temp variable
            for (int i = 0; i < theta0Temp.length; i++) {
                for (int j = 0; j < theta0Temp[i].length; j++) {
                    theta0Temp[i][j] -= alpha * theta0Grad[i][j];
                }
            }
            
            //Update theta1 in temp variable
            for (int i = 0; i < theta1Temp.length; i++) {
                for (int j = 0; j < theta1Temp[i].length; j++) {
                    theta1Temp[i][j] -= alpha * theta1Grad[i][j];
                }
            }
            
            //Set theta0 and theta1
            set(theta0, theta0Temp);
            set(theta1, theta1Temp);
        }
    }
    
    
    private void computeGradient(List<float[]> input, int[] inputClass, float[][] theta0, float[][] theta1, float lambda, float[][] theta0Grad, float[][] theta1Grad) {
        
        //Compute theta0Grad and theta1Grad for inputs
        for (int i = 0; i < input.size(); i++) {
            float[] currentInput = input.get(i);
            int currentInputClass = inputClass[i];
            accumulateGradient(currentInput, currentInputClass, theta0, theta1, theta0Grad, theta1Grad);
        }
        
        //Divide gradient by m
        int m = input.size();
        mul(theta0Grad, 1/m);
        mul(theta1Grad, 1/m);
        
        //Regularized theta0Grad: theta0Grad + (lambda/m) * theta0
        for (int i = 0; i < theta0Grad.length; i++) {
            for (int j = 1; j < theta0Grad[i].length; j++) {
                theta0Grad[i][j] += (lambda/m) * theta0[i][j];
            }
        }
        
        //Regularized theta1Grad: theta1Grad + (lambda/m) * theta1
        for (int i = 0; i < theta1Grad.length; i++) {
            for (int j = 1; j < theta1Grad[i].length; j++) {
                theta1Grad[i][j] += (lambda/m) * theta1[i][j];
            }
        }
    }
    
    
    
    
    private void accumulateGradient(float[] input, int inputClass, float[][] theta0, float[][] theta1, float[][] theta0Grad, float[][] theta1Grad) {
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
        
        
        //Accumulate theta0Grad: theta0Grad + delta1 * a0'
        for (int i = 0; i < theta0Grad.length; i++) {
            for (int j = 1; j < theta0Grad[i].length; j++) {
                theta0Grad[i][j] += delta1[i + 1] * input[j - 1];
            }
        }
        
        //Accumulate theta1Grad: theta1Grad + delta2 * a1'
        for (int i = 0; i < theta1Grad.length; i++) {
            for (int j = 1; j < theta1Grad[i].length; j++) {
                theta1Grad[i][j] += delta2[i] * a1[j - 1];
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
    
    private void set(float[][] m, float[][] v) {
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length; j++) {
                m[i][j] = v[i][j];
            }
        }
    }
    
    private void set1InIndex(float[] y, int index) {
        set(y, 0);
        y[index] = 1;
    }
    
    private void mul(float[][] m, float s) {
        for (int i = 0; i < m.length; i++) {
            mul(m[i], s);
        }
    }
    
    private void mul(float[] a, float s) {
        for (int i = 0; i < a.length; i++) {
            a[i] *= s;
        }
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
