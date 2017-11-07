'''@file decoder.py
neural network decoder environment'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
from classifiers import seq_convertors
import imp

# Tracer = imp.load_source('Tracer', 'home/padmaja/Downloads/Downloads/anaconda3/lib/python3.5/site-packages/IPython.core.debugger')


from IPython.core.debugger import Tracer; debug_here = Tracer();

class Decoder(object):
    '''the abstract class for a decoder'''

    __metaclass__ = ABCMeta

    # A metaclass is the class of a class. Like a class defines how an instance of the class behaves,
    # a metaclass defines how a class behaves. A class is an instance of a metaclass.
    # Refer metaclass.png in project folder

    def __init__(self, classifier, input_dim, max_input_length):
        '''
        Decoder constructor, creates the decoding graph

        Args:
            classifier: the classifier that will be used for decoding
            input_dim: the input dimension to the nnnetgraph
            max_input_length: the maximum length of the inputs
        '''

        self.graph = tf.Graph()
        self.max_input_length = max_input_length

        with self.graph.as_default():
            # Create the inputs placeholder
            # Placeholder : A placeholder is simply a variable that we will assign data to at a later date.
            # It allows us to create our operations and build our computation graph, without needing the data.
            # In TensorFlow terminology, we then feed data into the graph through these placeholders.
            self.inputs = tf.placeholder(
                tf.float32, shape=[1, max_input_length, input_dim],
                name='inputs')

            # create the sequence length placeholder
            self.input_seq_length = tf.placeholder(
                tf.int32, shape=[1], name='seq_length')

            # create the decoding graph
            logits, logits_seq_length, self.saver, _ = \
                classifier(
                    self.inputs, self.input_seq_length, targets=None,
                    target_seq_length=None, is_training=False, decoding=True,
                    reuse=None, scope='Classifier')

            # compute the outputs based on the classifier output logits
            self.outputs = self.get_outputs(logits, logits_seq_length)

        # specify that the graph can no longer be modified after this point
        self.graph.finalize()

    @abstractmethod
    def get_outputs(self, logits, logits_seq_length):
        '''
        Compute the outputs based on the output classifier output logits

        Args:
            logits: a NxO tensor where N is the sequence length and O is the
                classifier output dimension
            logits_seq_length: the logits sequence length

        Returns:
            the outputs of the decoding graph
        '''

        raise NotImplementedError("Abstract method")

    @abstractmethod
    def process_decoded(self, decoded):
        '''
        do some postprocessing on the output of the decoding graph

        Args:
            decoded: the outputs of the decoding graph

        Returns:
            the processed outputs
        '''

    def __call__(self, inputs):
        '''decode using the neural net

        Args:
            inputs: the inputs to the graph as a NxF numpy array where N is the
                number of frames and F is the input feature dimension

        Returns:
            the output of the decoder
        '''

        # get the sequence length
        input_seq_length = [inputs.shape[0]]

        # pad the inputs
        inputs = np.append(
            inputs, np.zeros([self.max_input_length - inputs.shape[0],
                              inputs.shape[1]]), 0)

        # pylint: disable=E1101
        decoded = tf.get_default_session().run(
            self.outputs,
            feed_dict={self.inputs: inputs[np.newaxis, :, :],
                       self.input_seq_length: input_seq_length})

        '''tf.get_default_session() -->
        Returns the default session for the current thread.
  The returned `Session` will be the innermost session on which a
  `Session` or `Session.as_default()` context has been entered.'''

        decoded = self.process_decoded(decoded)
        return decoded

    def restore(self, filename):
        '''
        load the saved neural net

        Args:
            filename: location where the neural net is saved
        '''

        self.saver.restore(tf.get_default_session(), filename)


class SimpleDecoder(Decoder):
    '''Simple decoder that passes the output logits through a softmax.'''

    def get_outputs(self, logits, logits_seq_length):
        '''
        Put the classifier output logits through a softmax

        Args:
            logits: A list containing a 1xO tensor for each timestep where O
                is the classifier output dimension
            logits_seq_length: the logits sequence length
        Returns:
            An NxO tensor containing posterior distributions
        '''

        # convert logits to non sequence for the softmax computation
        logits = seq_convertors.seq2nonseq(logits, logits_seq_length)
        '''
        softmax(logits, dim=-1, name=None):
        Computes softmax activations.
        For each batch `i` and class `j` we have
        softmax = exp(logits) / reduce_sum(exp(logits), dim)
        '''
        return tf.nn.softmax(logits)

    def process_decoded(self, decoded):
        '''
        do nothing

        Args:
            decoded: the outputs of the decoding graph

        Returns:
            the outputs of the decoding graph
        '''

        return decoded


class CTCDecoder(Decoder):
    """CTC Decoder"""

    def __init__(self, classifier, input_dim, max_input_length, beam_width):
        """
        Decoder constructor, creates the decoding graph

        Args:
            classifier: the classifier that will be used for decoding
            input_dim: the input dimension to the nnnetgraph
            max_input_length: the maximum length of the inputs
            beam_width: the width of the decoding beam
        """

        # store the beam width
        self.beam_width = beam_width

        super(CTCDecoder, self).__init__(classifier, input_dim,
                                         max_input_length)

    def get_outputs(self, logits, logits_seq_length):
        """
        get the outputs with ctc beam search

        Args:
            logits: A list containing a 1xO tensor for each timestep where O
                is the classifier output dimension      ?????????????????????????????????????
            logits_seq_length: the logits sequence length

        Returns:
            a pair containing:
                - a tuple of length W containing vectors with output sequences
                - a W dimensional vector containing the log probabilities
        """

        # Convert logits to time major
        logits = tf.stack(tf.unstack(logits, axis=1))

        # do the CTC beam search
        sparse_outputs, logprobs = tf.nn.ctc_beam_search_decoder(
            tf.stack(logits), logits_seq_length, self.beam_width,
            self.beam_width)

        # convert the outputs to dense tensors
        dense_outputs = [
            tf.reshape(tf.sparse_tensor_to_dense(o), [-1])
            for o in sparse_outputs]

        return dense_outputs + [tf.reshape(logprobs, [-1])]

    def process_decoded(self, decoded):
        '''
        create numpy arrays of decoded targets

        Args:
            decoded: a pair containing:
                - a tuple of length W containing vectors with output sequences
                - a W dimensional vector containing the log probabilities

        Returns:
            decoded: a pair containing:
                - a list of output sequences
                - a W dimensional vector containing the log probabilities
        '''

        target_sequences = decoded[:-1]
        logprobs = decoded[-1]

        return target_sequences, logprobs


class SimpleSeqDecoder(Decoder):
    '''Simple decoder that passes the output logits through a softmax'''

    def get_outputs(self, logits, logits_seq_length):
        '''
        Do nothing.

        Args:
            logits: A list containing a 1xO tensor for each timestep where O
                is the classifier output dimension
            logits_seq_length: the logits sequence length
        Returns:
            An NxO tensor containing posterior distributions
        '''

        return logits, logits_seq_length

    def process_decoded(self, decoded):
        '''
        Do nothing.

        Args:
            decoded: the outputs of the decoding graph

        Returns:
            the outputs of the decoding graph
        '''
        logits, seq_length = decoded
        # debug_here()
        return logits[:seq_length], seq_length
