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

