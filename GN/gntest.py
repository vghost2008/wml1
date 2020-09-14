import tensorflow as tf
import numpy as np
import logging
from GN.graph import DynamicAdjacentMatrix
import time
import wml_utils as wmlu

class WMLTest(tf.test.TestCase):
    def test_base_test(self):
        with self.test_session() as sess:
            adj_mt = tf.convert_to_tensor(np.array([[0,1,1],[1,0,1],[1,1,0]],dtype=np.int32))
            points_data = np.array([[1],[2],[3]],dtype=np.float32)
            A = DynamicAdjacentMatrix(adj_mt=adj_mt,points_data=points_data,edges_data=None,edges_data_dim=1,max_nodes_edge_nr=3)
            senders_indexs, receivers_indexs = tf.unstack(A.edges_to_points_index, axis=1)
            e_data_s = tf.gather(points_data, senders_indexs)
            e_data_r = tf.gather(points_data, receivers_indexs)
            e_data0 = e_data_r + e_data_s
            e_data1 = e_data_r - e_data_s
            A.edges_data = e_data0
            self.assertAllClose(e_data0.eval(),[[3],[4],[3],[5],[4],[5]],atol=1e-4)
            self.assertAllClose(e_data1.eval(),[[1],[2],[-1],[1],[-2],[-1]],atol=1e-4)
            self.assertAllEqual(A.real_edge_nr.eval(),6)
            self.assertAllEqual(A.edges_to_points_index.eval(),[[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]])
            wmlu.show_nparray(A.edges_to_points_index.eval())
            target_length = [2,2,2]
            target_p2e = [[[0,1,0], [2,4,0]] , [[2,3,0], [0,5,0]] , [[4,5,0], [1,3,0]] ]
            data = A.points_to_edges_index[0].eval()
            leng = A.points_to_edges_index[1].eval()
            self.assertAllEqual(target_length,leng)
            self.assertAllEqual(target_p2e,data)

    def test_update_edges(self):
        with self.test_session() as sess:
            adj_mt = tf.convert_to_tensor(np.array([[0,1,1],[1,0,1],[1,1,0]],dtype=np.int32))
            points_data = np.array([[1],[2],[3]],dtype=np.float32)
            edges_data = np.array([[5],[6],[7],[8],[9],[10]],dtype=np.float32)
            A = DynamicAdjacentMatrix(adj_mt=adj_mt,points_data=points_data,edges_data=edges_data)
            A.global_attr = tf.convert_to_tensor(np.array([[21]],dtype=np.float32))
            def update_edges(x):
                def fn(x):
                    return tf.reshape(tf.reduce_sum(x),[1])
                return tf.map_fn(fn,elems=(x))
            A.update_edges(update_edges)
            self.assertAllClose(A.edges_data.eval(),[[29],[31],[31],[34],[34],[36]],atol=1e-5)

    def test_update_points(self):
        with self.test_session() as sess:
            adj_mt = tf.convert_to_tensor(np.array([[0,1,1],[1,0,1],[1,1,0]],dtype=np.int32))
            points_data = np.array([[1],[2],[3]],dtype=np.float32)
            edges_data = np.array([[5],[6],[7],[8],[9],[10]],dtype=np.float32)
            A = DynamicAdjacentMatrix(adj_mt=adj_mt,points_data=points_data,edges_data=edges_data)
            A.global_attr = tf.convert_to_tensor(np.array([[21]],dtype=np.float32))
            def update_points(x):
                def fn(x):
                    return tf.reshape(tf.reduce_sum(x),[1])
                return tf.map_fn(fn,elems=(x))
            A.update_points(update_points)
            sess.run(tf.global_variables_initializer())
            points_data = A.points_data.eval()
            print(points_data)
            self.assertAllClose(points_data,[[35.5],[38.0],[40.5]],atol=1e-5)
            
    def test_update_globals(self):
        with self.test_session() as sess:
            adj_mt = tf.convert_to_tensor(np.array([[0,1,1],[1,0,1],[1,1,0]],dtype=np.int32))
            points_data = np.array([[1],[2],[3]],dtype=np.float32)
            edges_data = np.array([[5],[6],[7],[8],[9],[10]],dtype=np.float32)
            A = DynamicAdjacentMatrix(adj_mt=adj_mt,points_data=points_data,edges_data=edges_data)
            A.global_attr = tf.convert_to_tensor(np.array([[21]],dtype=np.float32))
            def update_global(x):
                def fn(x):
                    return tf.reshape(tf.reduce_sum(x),[1])
                return tf.map_fn(fn,elems=(x))
            A.update_global(update_global)
            self.assertAllClose(A.global_attr.eval(),[[30.5]],atol=1e-5)

    def test_update_edges_independent(self):
        with self.test_session() as sess:
            adj_mt = tf.convert_to_tensor(np.array([[0,1,1],[1,0,1],[1,1,0]],dtype=np.int32))
            points_data = np.array([[1],[2],[3]],dtype=np.float32)
            edges_data = np.array([[5],[6],[7],[8],[9],[10]],dtype=np.float32)
            A = DynamicAdjacentMatrix(adj_mt=adj_mt,points_data=points_data,edges_data=edges_data)
            A.global_attr = tf.convert_to_tensor(np.array([[21]],dtype=np.float32))
            def update_edges(x):
                def fn(x):
                    return x*2
                return tf.map_fn(fn,elems=(x))
            A.update_edges_independent(update_edges)
            self.assertAllClose(A.edges_data.eval(),edges_data*2,atol=1e-5)

    def test_update_points_independent(self):
        with self.test_session() as sess:
            adj_mt = tf.convert_to_tensor(np.array([[0,1,1],[1,0,1],[1,1,0]],dtype=np.int32))
            points_data = np.array([[1],[2],[3]],dtype=np.float32)
            edges_data = np.array([[5],[6],[7],[8],[9],[10]],dtype=np.float32)
            A = DynamicAdjacentMatrix(adj_mt=adj_mt,points_data=points_data,edges_data=edges_data)
            A.global_attr = tf.convert_to_tensor(np.array([[21]],dtype=np.float32))
            def update_points(x):
                def fn(x):
                    return x+[2]
                return tf.map_fn(fn,elems=(x))
            A.update_points_independent(update_points)
            self.assertAllClose(A.points_data.eval(),points_data+[[2]],atol=1e-5)

    def test_update_globals_independent(self):
        with self.test_session() as sess:
            adj_mt = tf.convert_to_tensor(np.array([[0,1,1],[1,0,1],[1,1,0]],dtype=np.int32))
            points_data = np.array([[1],[2],[3]],dtype=np.float32)
            edges_data = np.array([[5],[6],[7],[8],[9],[10]],dtype=np.float32)
            A = DynamicAdjacentMatrix(adj_mt=adj_mt,points_data=points_data,edges_data=edges_data)
            A.global_attr = tf.convert_to_tensor(np.array([[21]],dtype=np.float32))
            def update_global(x):
                def fn(x):
                    return x/2.0
                return tf.map_fn(fn,elems=(x))
            A.update_global_independent(update_global)
            print(A.global_attr.eval())
            self.assertAllClose(A.global_attr.eval(),[[10.5]],atol=1e-5)


if __name__ == "__main__":
    np.random.seed(int(time.time()))
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
                        datefmt="%H:%M:%S")
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.test.main()