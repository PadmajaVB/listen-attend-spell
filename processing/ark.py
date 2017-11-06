"""
@file ark.py
contains the .ark io functionality

Copyright 2014    Yajie Miao    Carnegie Mellon University
           2015    Yun Wang      Carnegie Mellon University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
MERCHANTABLITY OR NON-INFRINGEMENT.
See the Apache 2 License for the specific language governing permissions and
limitations under the License.
"""

import struct
import numpy as np

np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=np.nan)


class ArkReader(object):
    """
    Class to read Kaldi ark format. Each time, it reads one line of the .scp
    file and reads in the corresponding features into a numpy matrix. It only
    supports binary-formatted .ark files. Text and compressed .ark files are not
    supported. The inspiration for this class came from pdnn toolkit (see
    licence at the top of this file) (https://github.com/yajiemiao/pdnn)
    """

    def __init__(self, scp_path):
        """
        ArkReader constructor

        Args:
            scp_path: path to the .scp file
        """

        self.scp_position = 0
        fin = open(scp_path, "r")
        self.utt_ids = []
        self.scp_data = []
        line = fin.readline()
        while line != '' and line != None:
            utt_id, path_pos = line.replace('\n', '').split(' ')
            path, pos = path_pos.split(':')
            self.utt_ids.append(utt_id)
            self.scp_data.append((path, pos))
            line = fin.readline()

        fin.close()

    def read_utt_data(self, index):
        """
        read data from the archive

        Args:
            index: index of the utterance that will be read

        Returns:
            a numpy array containing the data from the utterance
        """

        ark_read_buffer = open(self.scp_data[index][0], 'rb')
        ark_read_buffer.seek(int(self.scp_data[index][1]), 0)
        header = struct.unpack('<xcccc', ark_read_buffer.read(5))
        if header[0] != "B":
            print "Input .ark file is not binary"
            exit(1)
        if header[1] == "C":
            print "Input .ark file is compressed"
            exit(1)

        _, rows = struct.unpack('<bi', ark_read_buffer.read(5))
        _, cols = struct.unpack('<bi', ark_read_buffer.read(5))

        '''
        def unpack(fmt, string): # known case of _struct.unpack
        Unpack the string containing packed C structure data, according to fmt.
        Requires len(string) == calcsize(fmt).
        
        frombuffer(buffer, dtype=float, count=-1, offset=0)
        Interpret a buffer as a 1-dimensional array.
        Notes
        -----
        If the buffer has data that is not in machine byte-order, this should
        be specified as part of the data-type, e.g.::
    
          >>> dt = np.dtype(int)
          >>> dt = dt.newbyteorder('>')
          >>> np.frombuffer(buf, dtype=dt)
    
        The data of the resulting array will not be byteswapped, but will be
        interpreted correctly.
    
        Examples
        --------
        >>> s = 'hello world'
        >>> np.frombuffer(s, dtype='S1', count=5, offset=6)
        array(['w', 'o', 'r', 'l', 'd'],
              dtype='|S1')
        '''

        if header[1] == "F":
            tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 4),
                                    dtype=np.float32)
        elif header[1] == "D":
            tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 8),
                                    dtype=np.float64)

        utt_mat = np.reshape(tmp_mat, (rows, cols))

        ark_read_buffer.close()

        return utt_mat

    def read_next_utt(self):
        """
        read the next utterance in the scp file

        Returns:
            the utterance ID of the utterance that was read, the utterance data,
            bool that is true if the reader looped back to the beginning
        """

        if len(self.scp_data) == 0:
            return None, None, True

        # if at end of file loop around
        if self.scp_position >= len(self.scp_data):
            looped = True
            self.scp_position = 0
        else:
            looped = False

        self.scp_position += 1

        return (self.utt_ids[self.scp_position - 1],
                self.read_utt_data(self.scp_position - 1), looped)

    def read_next_scp(self):
        """
        read the next utterance ID but don't read the data

        Returns:
            the utterance ID of the utterance that was read
        """

        # if at end of file loop around
        if self.scp_position >= len(self.scp_data):
            self.scp_position = 0

        self.scp_position += 1

        return self.utt_ids[self.scp_position - 1]

    def read_previous_scp(self):
        """
        read the previous utterance ID but don't read the data

        Returns:
            the utterance ID of the utterance that was read
        """

        if self.scp_position < 0:  # if at beginning of file loop around
            self.scp_position = len(self.scp_data) - 1

        self.scp_position -= 1

        return self.utt_ids[self.scp_position + 1]

    def read_utt(self, utt_id):
        """
        read the data of a certain utterance ID

        Returns:
            the utterance data corresponding to the ID
        """

        return self.read_utt_data(self.utt_ids.index(utt_id))

    def split(self):
        """Split of the data that was read so far"""

        self.scp_data = self.scp_data[self.scp_position:-1]
        self.utt_ids = self.utt_ids[self.scp_position:-1]


class ArkWriter(object):
    """
    Class to write numpy matrices into Kaldi .ark file and create the
    corresponding .scp file. It only supports binary-formatted .ark files. Text
    and compressed .ark files are not supported. The inspiration for this class
    came from pdnn toolkit (see licence at the top of this file)
    (https://github.com/yajiemiao/pdnn)
    """

    def __init__(self, scp_path, default_ark):
        """
        Arkwriter constructor

        Args:
            scp_path: path to the .scp file that will be written
            default_ark: the name of the default ark file (used when not
                specified)
        """

        self.scp_path = scp_path
        self.scp_file_write = open(self.scp_path, 'w')
        self.default_ark = default_ark

    def write_next_utt(self, utt_id, utt_mat, ark_path=None):
        """
        read an utterance to the archive

        Args:
            ark_path: path to the .ark file that will be used for writing
            utt_id: the utterance ID
            utt_mat: a numpy array containing the utterance data
        """

        ark = ark_path or self.default_ark
        ark_file_write = open(ark, 'ab')
        utt_mat = np.asarray(utt_mat, dtype=np.float32)

        '''
        np.asarray()
        Convert the input to an array.

        Parameters
        ----------
        a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.
        dtype : data-type, optional
        By default, the data-type is inferred from the input data.
        order : {'C', 'F'}, optional
        Whether to use row-major (C-style) or
        column-major (Fortran-style) memory representation.
        Defaults to 'C'.

        Returns
        -------
        out : ndarray
        Array interpretation of `a`.  No copy is performed if the input
        is already an ndarray with matching dtype and order.  If `a` is a
        subclass of ndarray, a base class ndarray is returned.
        
        Examples
    --------
    Convert a list into an array:

    >>> a = [1, 2]
    >>> np.asarray(a)
    array([1, 2])

    Existing arrays are not copied:

    >>> a = np.array([1, 2])
    >>> np.asarray(a) is a
    True

    If `dtype` is set, array is copied only if dtype does not match:

    >>> a = np.array([1, 2], dtype=np.float32)
    >>> np.asarray(a, dtype=np.float32) is a
    True
    >>> np.asarray(a, dtype=np.float64) is a
    False

    Contrary to `asanyarray`, ndarray subclasses are not passed through:

    >>> issubclass(np.matrix, np.ndarray)
    True
    >>> a = np.matrix([[1, 2]])
    >>> np.asarray(a) is a
    False
    >>> np.asanyarray(a) is a
    True
        '''

        rows, cols = utt_mat.shape
        ark_file_write.write(struct.pack('<%ds' % (len(utt_id)), utt_id))
        pos = ark_file_write.tell()  # tell() -> current file position, an integer (may be a long integer).

        '''
        write(str) -> None.  Write string str to file.      
        Note that due to buffering, flush() or close() may be needed before
        the file on disk reflects the data written.
        
        def pack(fmt, *args): # known case of _struct.pack
    """ Return string containing values v1, v2, ... packed according to fmt. """
        '''

        ark_file_write.write(struct.pack('<xcccc', 'B', 'F', 'M', ' '))
        ark_file_write.write(struct.pack('<bi', 4, rows))
        ark_file_write.write(struct.pack('<bi', 4, cols))
        ark_file_write.write(utt_mat)
        self.scp_file_write.write('%s %s:%s\n' % (utt_id, ark, pos))
        ark_file_write.close()

    def close(self):
        """close the ark writer"""

        self.scp_file_write.close()


'''pass in python : The pass statement in Python is used when a statement is required syntactically 
but you do not want any command or code to execute. 
The pass statement is a null operation; nothing happens when it executes.'''