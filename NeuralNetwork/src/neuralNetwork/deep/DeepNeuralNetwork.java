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
    private final int iterations;
    private final float learningRate;
    private Map<String, Matrix2> parameters;
    
    public DeepNeuralNetwork(int[] layerDims, int iterations, float learningRate) {
        this.layerDims = layerDims;
        this.iterations = iterations;
        this.learningRate = learningRate;
    }

    public void train(Matrix2 X, Matrix2 Y, boolean printCost) {
        int m = X.cols();
        
        //Initialize parameters
        this.parameters = initializeParameters(this.layerDims);
        
        //Gradient descent loop
        List<CacheItem> caches = new ArrayList<>();
        Map<String, Matrix2> grads = new HashMap<>();
        for (int i = 0; i < iterations; i++) {
            caches.clear();
            grads.clear();
            
            //Forward propagation
            Matrix2 AL = modelForward(X, parameters, caches);
            
            //Compute cost
            float cost = computeCost(AL, Y);
            
            //Backward propagation
            grads = modelBackward(AL, Y, caches, grads);
            
            //Update parameters
            updateParameters(parameters, grads, learningRate);
            
            //print cost
            if(printCost) {
                if(i % 10 == 0) {
                    System.out.println("Cost after iteration " + i + ": " + cost);
                }
            }           
        }
    }
    
    public Matrix2 predict(Matrix2 X) {
        List<CacheItem> caches = new ArrayList<>();
        
        Matrix2 AL = modelForward(X, parameters, caches);
        
        //AL > 0.5
        return AL.greater(0.5f);
    }
    
    
    
    
    
    /**
     * Init W and b parameters for all layers
     */
    private Map<String, Matrix2> initializeParameters(int[] layerDims) {
        Map<String, Matrix2> parameters = new HashMap<>((layerDims.length - 1) * 2);
        for (int l = 1; l < layerDims.length; l++) {
            String layerIdx = String.valueOf(l);
            int rows = layerDims[l];
            int cols = layerDims[l - 1];
            parameters.put("W" + layerIdx, Matrix2.random(rows, cols).mul(0.01f));
            parameters.put("b" + layerIdx, Matrix2.zeros(rows, 1));
        }
        return parameters;
    }
    
    /**
     * Forward propagation for all layers.
     * Compute AL and store intermediate values in caches
     */
    private Matrix2 modelForward(Matrix2 X, Map<String, Matrix2> parameters, List<CacheItem> caches) {
        Matrix2 A = X;
        int L = parameters.size() / 2;
        
        //Linear-Relu pass for all layers except the last one
        for (int l = 1; l < L; l++) {
            Matrix2 A_prev = A;
            String layerIdx = String.valueOf(l);
            Matrix2 W = parameters.get("W" + layerIdx);
            Matrix2 b = parameters.get("b" + layerIdx);
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
     * Linear forward pass: Z = W * A + b
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
                Matrix2.mulEW(Y, AL.log()), 
                Matrix2.mulEW(Y.oneMinus(), AL.oneMinus().log())
        ).sumColumns().mul(-1f/m).get(0,0);
        
        return cost;
    }
    
    private Map<String, Matrix2> modelBackward(Matrix2 AL, Matrix2 Y, List<CacheItem> caches, Map<String, Matrix2> grads) {
        int m = Y.cols();
        int L = caches.size();
        CacheItem cache;
        String layerIdx;
        BackpropResult res;
        
        //Compute sigmoid gradient for last layer
        //dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
        Matrix2 dAL = Matrix2.divEW(Y, AL).sub(Matrix2.divEW(Y.oneMinus(), AL.oneMinus())).mul(-1);
        cache = caches.get(L - 1);
        res = linearActivationBackward(dAL, cache, SigmoidBackward.INSTANCE);
        layerIdx = String.valueOf(L);
        grads.put("dA" + layerIdx, res.dA);
        grads.put("dW" + layerIdx, res.dW);
        grads.put("db" + layerIdx, res.db);
        
        //Compute relu gradients for all other layers
        for (int l = L - 2; l >= 0; l--) {
            layerIdx = String.valueOf(l + 1);
            cache = caches.get(l);
            Matrix2 dA_current = grads.get("dA" + (l + 2));
            res = linearActivationBackward(dA_current, cache, ReluBackward.INSTANCE);
            grads.put("dA" + layerIdx, res.dA);
            grads.put("dW" + layerIdx, res.dW);
            grads.put("db" + layerIdx, res.db);
        }
        
        return grads;
    }
    
    private BackpropResult linearActivationBackward(Matrix2 dA, CacheItem cache, BackwardOp activation) {
        Matrix2 dZ = activation.apply(dA, cache.activationCache);    
        BackpropResult res = linearBackward(dZ, cache.linearCache); 
        return new BackpropResult(res.dA, res.dW, res.db);
    }
    
    private BackpropResult linearBackward(Matrix2 dZ, LinearCache cache) {
        int m = cache.A.cols();
        
        //dW = 1/m * np.dot(dZ, A_prev.T)
        Matrix2 dW = dZ.mul(cache.A.transpose()).mul(1f/m);
        
        //db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        Matrix2 db = dZ.sumColumns().mul(1f/m);
        
        //dA_prev = np.dot(W.T, dZ)
        Matrix2 dAprev = cache.W.transpose().mul(dZ);
        
        return new BackpropResult(dAprev, dW, db);
    }

    private void updateParameters(Map<String, Matrix2> parameters, Map<String, Matrix2> grads, float learningRate) {
        int L = parameters.size() / 2;
        for (int l = 1; l <= L; l++) {
            String layerIdx = String.valueOf(l);
            Matrix2 W = parameters.get("W" + layerIdx);
            Matrix2 b = parameters.get("b" + layerIdx);
            Matrix2 dW = grads.get("dW" + layerIdx);
            Matrix2 db = grads.get("db" + layerIdx);
            
            //W = W - learningRate * dW
            W = W.sub(dW.mul(learningRate));
            b = b.sub(db.mul(learningRate));
            
            parameters.put("W" + layerIdx, W);
            parameters.put("b" + layerIdx, b);
        }
    }

    
    
    
    
    interface BackwardOp {
        Matrix2 apply(Matrix2 dA, ActivationCache cache);
    }
    
    private static class ReluBackward implements BackwardOp {
        public static final BackwardOp INSTANCE = new ReluBackward();
        @Override
        public Matrix2 apply(Matrix2 dA, ActivationCache cache) {
            //dz = 0 if z <= 0
            Matrix2 dZ = dA.apply(new Matrix2.ElementWiseOp() {
                @Override
                public float apply(float z) {
                    return z <= 0 ? 0 : z;
                }
            });
            return dZ;
        } 
    }
    
    private static class SigmoidBackward implements BackwardOp {
        public static final BackwardOp INSTANCE = new SigmoidBackward();
        @Override
        public Matrix2 apply(Matrix2 dA, ActivationCache cache) {
            //S = 1 / (1 + e^(-Z))
            Matrix2 S = cache.Z.sigmoid();

            //dZ = dA * s * (1-s)
            Matrix2 dZ = dA.mulEW(S).mulEW(S.oneMinus());

            return dZ;
        } 
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
    
    public static class BackpropResult {
        public final Matrix2 dA;
        public final Matrix2 dW;
        public final Matrix2 db;
        public BackpropResult(Matrix2 dA, Matrix2 dW, Matrix2 db) {
            this.dA = dA;
            this.dW = dW;
            this.db = db;
        }  
    }
    
    
    
    
}
