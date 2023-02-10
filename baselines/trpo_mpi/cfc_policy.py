from baselines.baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.baselines.common.distributions import make_pdtype
import numpy as np
from ncps.tf import CfC
from ncps import wirings

# CfCPolicy is a class based on MlpPolicy from the original LOKI experiment.
# Currently implimented to play lunar lander
class CfCPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        input_layer = tf.keras.layers.Input(
            shape=(None, ob_space.shape[0] * ob_space.shape[1] * ob_space.shape[2]),
            name="inputs"
            )
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)
        state_in_h = tf.keras.layers.Input(shape=(hid_size*num_hid_layers,), name="h")


        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i" %
                                                      (i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='value',
                                         kernel_initializer=U.normc_initializer(1.0))[:, 0]
            self.vpredz = self.vpred

        with tf.variable_scope('pol'):
            agent = obz
            num_neurons = hid_size * num_hid_layers
            wiring = wirings.AutoNCP(num_neurons, ac_space)
            
            model = tf.keras.layers.TimeDistributed(tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(None, ob_space.shape[-1])),
                CfC(wiring, return_sequences=True),
                ]))
            
        

            self.logits = tf.layers.dense(model, pdtype.param_shape()[0], name='logits', kernel_initializer=U.normc_initializer(0.01))
            

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(self.logits, pdtype.param_shape()[
                                       0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[
                                         0]//2], initializer=tf.zeros_initializer())
                # logstd_min = -0.5
                # logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[
                #                          0]//2], initializer=tf.zeros_initializer())
                # logstd = logstd_min + (logstd + np.sqrt(-logstd_min))**2
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(self.logits, pdtype.param_shape(
                )[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample()
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
    
    def value_function(self):
        return tf.reshape(self._value_out, [-1])
