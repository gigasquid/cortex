(ns cortex.verify.tensor.operations
  (:require [clojure.core.matrix :as m]
            [clojure.test :refer :all]
            [cortex.compute.driver :as drv]
            [cortex.tensor :as ct]
            [cortex.tensor.operations :as tops]))


(defmacro tensor-context
  [driver datatype & body]
  `(drv/with-compute-device
     (drv/default-device ~driver)
     (with-bindings {#'ct/*stream* (drv/create-stream)
                     #'ct/*datatype* ~datatype}
       ~@body)))


(comment
       (println "carin " (map str (ct/to-double-array result2)))
)

(defn max-operation
  [driver datatype]
  (tensor-context
   driver datatype
   (let [tens-a (ct/->tensor [0 1 2 3 4])
         tens-b (tops/new-tensor tens-a)
         result (tops/max tens-b tens-a 2)]
     (is (m/equals [2 2 2 3 4]
                   (ct/to-double-array
                    result)))
     (let [result2 (tops/max result 3)]
       (is (m/equals [3 3 3 3 4]
                     (ct/to-double-array
                      result2)))))))

(defn min-operation
  [driver datatype]
  (tensor-context
   driver datatype
   (let [tens-a (ct/->tensor [0 1 2 3 4])
         tens-b (tops/new-tensor tens-a)
         result (tops/min tens-b tens-a 2)]
     (is (m/equals [0 1 2 2 2]
                   (ct/to-double-array
                    result)))
     (let [result2 (tops/min result 1)]
       (is (m/equals [0 1 1 1 1]
                     (ct/to-double-array
                      result2)))))))

(defn ceil-operation
  [driver datatype]
  (tensor-context
   driver datatype
   (let [tens-a (ct/->tensor (partition 3 (range 9)))
         tens-b (ct/->tensor (partition 3 (repeat 9 1)))
         result (tops/ceil tens-b (tops/* tens-a 2.5))]
     (is (m/equals (mapv #(Math/ceil (* ^double % (drv/dtype-cast 2.5 datatype))) (range 9))
                   (ct/to-double-array result)))
     (let [tens-c (ct/->tensor (partition 3 (range 9)))
           result2 (tops/ceil (tops/* tens-c 2.5))]
       (is (m/equals (mapv #(Math/ceil (* ^double % (drv/dtype-cast 2.5 datatype))) (range 9))
                     (ct/to-double-array result2)))))))

(defn floor-operation
  [driver datatype]
  (tensor-context
   driver datatype
   (let [tens-a (ct/->tensor (partition 3 (range 9)))
         tens-b (ct/->tensor (partition 3 (repeat 9 1)))
         result (tops/floor tens-b (tops/* tens-a 2.5))]
     (is (m/equals (mapv #(Math/floor (* ^double % (drv/dtype-cast 2.5 datatype))) (range 9))
                   (ct/to-double-array result)))
     (let [tens-c (ct/->tensor (partition 3 (range 9)))
           result2 (tops/floor (tops/* tens-c 2.5))]
       (is (m/equals (mapv #(Math/floor (* ^double % (drv/dtype-cast 2.5 datatype))) (range 9))
                     (ct/to-double-array result2)))))))

(defn logistic-operation
  [driver datatype]
  (tensor-context
   driver datatype
   (let [tens-a (ct/->tensor (range -4 8))
         tens-b (tops/new-tensor tens-a)
         result (tops/logistic tens-b tens-a)
         logistic-result [0.01798620996209156, 0.04742587317756678, 0.11920292202211755,
                          0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823,
                          0.9525741268224334, 0.9820137900379085, 0.9933071490757153,
                          0.9975273768433653, 0.9990889488055994]]
     (is (m/equals logistic-result
                   (ct/to-double-array result)
                   1e-4))
     (let [result2 (tops/logistic tens-a)]
       (is (m/equals logistic-result
                     (ct/to-double-array
                      result2)
                     1e-4))))))

(defn tanh-operation
  [driver datatype]
  (tensor-context
   driver datatype
   (let [tens-a (ct/->tensor (range -4 8))
         tens-b (tops/new-tensor tens-a)
         result (tops/tanh tens-b tens-a)
         tanh-result [-0.999329299739067, -0.9950547536867305, -0.9640275800758169,
                      -0.7615941559557649, 0.0, 0.7615941559557649, 0.9640275800758169,
                      0.9950547536867305, 0.999329299739067, 0.9999092042625951,
                      0.9999877116507956, 0.9999983369439447]]
     (is (m/equals tanh-result
                   (ct/to-double-array result)
                   1e-4))
     (let [result2 (tops/tanh tens-a)]
       (is (m/equals tanh-result
                     (ct/to-double-array
                      result2)
                     1e-4))))))

(defn exp-operation
  [driver datatype]
  (tensor-context
   driver datatype
   (let [src-data [0 1 2 3 4]
         tens-a (ct/->tensor src-data)
         tens-b (tops/new-tensor tens-a)
         result (tops/exp tens-b tens-a)
         exp-result (mapv #(drv/dtype-cast (Math/exp (double %)) datatype) src-data)]
     (is (m/equals exp-result
                   (ct/to-double-array result)))
     (let [result2 (tops/exp tens-a)]
       (is (m/equals exp-result
                     (ct/to-double-array
                      result2)))))))


(defn binary-operation [driver datatype op-fn src-data-a src-data-b compare-result]
  (tensor-context
   driver datatype
   (let [tens-a (ct/->tensor src-data-a)
         tens-b (ct/->tensor src-data-b)
         tens-c (tops/new-tensor tens-b)
         result (op-fn tens-c tens-a tens-b)]
     (is (m/equals compare-result
                   (ct/to-double-array result)))
     (let [result2 (op-fn tens-a tens-b)]
       (is (m/equals compare-result
                     (ct/to-double-array
                      result2)))))))

(defn multiply-operation
  [driver datatype]
  (let [x1 [-1 0 1 2 3 4]
        x2 [ 2 3 4 5 6 7]]
   (binary-operation driver
                     datatype
                     tops/*
                     x1
                     x2
                     (mapv #(drv/dtype-cast (* (double %1) (double %2)) datatype) x1 x2))))

(defn add-operation
  [driver datatype]
  (let [x1 [-1 0 1 2 3 4]
        x2 [ 2 3 4 5 6 7]]
    (binary-operation driver
                      datatype
                      tops/+
                      x1
                      x2
                      (mapv #(drv/dtype-cast (+ %1 %2) datatype) x1 x2))))

(defn subtract-operation
  [driver datatype]
  (let [x1 [1 2 3 4 5 6]
        x2 [0 1 2 1 0 1]]
    (binary-operation driver
                      datatype
                      tops/-
                      x1
                      x2
                      (mapv #(drv/dtype-cast (- %1 %2) datatype) x1 x2))))

(defn >-operation
  [driver datatype]
  (binary-operation driver
                    datatype
                    tops/>
                    [-1 0 1 2 3 4]
                    [1 1 1 1 1 1]
                    [0 0 0 1 1 1]))
