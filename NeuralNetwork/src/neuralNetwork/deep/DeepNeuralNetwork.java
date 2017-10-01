/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralNetwork.deep;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Deep Neural Network with L hidden layers
 * 
 * @author Matias Leone
 */
public class DeepNeuralNetwork {
    
    private final int[] layerDims;
    
    public DeepNeuralNetwork(int[] layerDims) {
        this.layerDims = layerDims;
    }
    
    public void train() {
        
    }

    public void predict() {
        
    }
    
    private void doTrain(Matrix2 X, Matrix2 Y, int iterations, float learningRate, boolean printCost) {
        int m = X.cols();
        
        //Initialize parameters
        Map<String, Matrix2> parameters = initializeParameters(this.layerDims);
        
        //Gradient descent loop
        Map<String, Matrix2> cache = new HashMap<>();
        List<CacheItem> caches = new ArrayList<>();
        Map<String, Matrix2> grads = new HashMap<>();
        int L = this.layerDims.length;
        for (int i = 0; i < iterations; i++) {
            caches.clear();
            grads.clear();
            
            //Forward propagation
            Matrix2 AL = modelForward(X, Y, parameters, L, caches);
            
            //Compute cost
            float cost = computeCost(AL, Y);
            
            //Backward propagation
            grads = modelBackward(AL, Y, caches, L, grads);
        }
        
    }
    
    
    
    /**
     * Init W and b parameters for all layers
     */
    private Map<String, Matrix2> initializeParameters(int[] layerDims) {
        Map<String, Matrix2> parameters = new HashMap<>(layerDims.length * 2);
        for (int i = 1; i < layerDims.length; i++) {
            String iStr = String.valueOf(i);
            int rows = layerDims[i];
            int cols = layerDims[i - 1];
            parameters.put("W" + iStr, Matrix2.random(rows, cols).mul(0.01f));
            parameters.put("b" + iStr, Matrix2.zeros(rows, 1));
        }
        return parameters;
    }
    
    /**
     * Forward propagation for all layers.
     * Compute AL and store intermediate values in caches
     */
    private Matrix2 modelForward(Matrix2 X, Matrix2 Y, Map<String, Matrix2> parameters, int L, List<CacheItem> caches) {
        Matrix2 A = X;
        
        //Linear-Relu pass for all layers except the last one
        for (int l = 1; l < L; l++) {
            Matrix2 A_prev = A;
            String iStr = String.valueOf(l);
            Matrix2 W = parameters.get("W" + iStr);
            Matrix2 b = parameters.get("b" + iStr);
            A = linearActivationForward(A_prev, W, b, Matrix2.ReluOp.INSTANCE, caches);
        }
        
        //Linear-Sigmoid for last layer
        Matrix2 WL = parameters.get("W" + L);
        Matrix2 bL = parameters.get("b" + L);
        Matrix2 AL = linearActivationForward(A, WL, bL, Matrix2.SigmoidOp.INSTANCE, caches);
        
        return AL;
    }
    
    /**
     * Activation and linear forward pass: A = g(Z)
     */
    private Matrix2 linearActivationForward(Matrix2 A_prev, Matrix2 W, Matrix2 b, Matrix2.ElementWiseOp activation, List<CacheItem> caches) {
        Matrix2 Z = linearForward(A_prev, W, b);
        LinearCache linearCache = new LinearCache(A_prev, W, b);
        
        Matrix2 A = Z.apply(activation);
        ActivationCache activationCache = new ActivationCache(Z);
        
        caches.add(new CacheItem(linearCache, activationCache));
        return A;
    }
    
    /**
     * Linaer forward pass: Z = W * A + b
     */
    private Matrix2 linearForward(Matrix2 A, Matrix2 W, Matrix2 b) {
        //Z = W * A + b;
        Matrix2 WxA = W.mul(A);
        Matrix2 Z = WxA.add(b.broadcastCol(WxA.cols()));
        return Z;
    }
    
    /**
     * Compute loss
     */
    private float computeCost(Matrix2 AL, Matrix2 Y) {
        int m = Y.cols();
        
        //cost = -1/m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
        float cost = Matrix2.add(
                Matrix2.mulEW(Y, AL), 
                Matrix2.mulEW(Y.oneMinus(), AL.oneMinus().log())
        ).sumColumns().mul(-1/m).get(0,0);
        
        return cost;
    }
    
    private Map<String, Matrix2> modelBackward(Matrix2 AL, Matrix2 Y, List<CacheItem> caches, int L, Map<String, Matrix2> grads) {
        int m = Y.cols();
        
        //dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
        Matrix2 dAL = Matrix2.divEW(Y, AL).sub(Matrix2.divEW(Y.oneMinus(), AL.oneMinus())).mul(-1);
        
        CacheItem currentCache = caches.get(L - 1);
        
        
        return grads;
    }
    
    private void linearBackward(Matrix2 dZ, LinearCache cache) {
        
        //return dA_prev, dW, db
    }
    
    private void linearActivationBackward(Matrix2 dA, CacheItem cache, String activation) {
        
        //return dA_prev, dW, db
    }
    
    
    
    private static class CacheItem {
        public final LinearCache linearCache;
        public final ActivationCache activationCache;
        public CacheItem(LinearCache linearCache, ActivationCache activationCache) {
            this.linearCache = linearCache;
            this.activationCache = activationCache;
        }
    }
    
    private static class LinearCache {
        public final Matrix2 A;
        public final Matrix2 W;
        public final Matrix2 b;
        public LinearCache(Matrix2 A, Matrix2 W, Matrix2 b) {
            this.A = A;
            this.W = W;
            this.b = b;
        }
    }
    
    private static class ActivationCache {
        public final Matrix2 Z;
        public ActivationCache(Matrix2 Z) {
            this.Z = Z;
        }
    }
    
    
    
    
}
