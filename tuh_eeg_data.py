import time

import yaml
import numpy as np
from queue import Empty
from multiprocessing import Process, Queue, Value

from edf_loader import load_edf

y = yaml.safe_load(open('CONFIG.yaml'))
VALID_CH_LIST = y['VALID_CH_LIST']
VALID_SAMPLING_RATE = y['VALID_SAMPLING_RATE']
RESAMPLE_RATE = y['RESAMPLE_RATE']
RESAMPLE_METHOD = y['RESAMPLE_METHOD']
SEGMENT_DURATION = y['SEGMENT_DURATION']


class TUHExampleIterator(object):

    # An iterator that yields examples.
    # It has a stack of loaded records (where each record is loaded
    # in the form of n examples (=EEG segments) and draws an example
    # randomly from the different segments of the different records.
    # A record is thrown away when a given proportion of its
    # segments has been used.

    def __init__(self, record_files, stack_size, record_exploit_rate):

        print('Initializing Example iterator.')
        self.record_files_it = iter(record_files)  # such that values can be used only once.
        self.stack_size = stack_size
        self.record_exploit_rate = record_exploit_rate  # percentage of segments to be used before record is thrown away.
        self.stack = {'data': [[]] * self.stack_size,
                      'available': [],  # contains stack indices. Unique values.
                      # which records can be sampled from (a record might not be because
                      # it is exhausted and not reloaded because record_list
                      # is finished. Indices will be filled when records are populated.
                      'nb_segments': [0] * self.stack_size,
                      'examples_used': [0] * self.stack_size,
                      'seg_inds_still_avail': [[]] * self.stack_size}

        # populate the stack. It might happen that record_files is smaller than stack.
        print('Populating stack...')
        for stack_idx, filename in zip(range(self.stack_size), self.record_files_it):
            self.load_new_record_to_stack(stack_idx, filename)
        print('Records stack of size %d populated with %d records. ' % (self.stack_size, stack_idx+1))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    def __iter__(self):
        return self

    def load_new_record_to_stack(self, stack_idx, filename=None):
        if filename is None:
            filename = next(self.record_files_it)
        segments = load_edf(filename, RESAMPLE_RATE,
                            RESAMPLE_METHOD, SEGMENT_DURATION)
        self.stack['data'][stack_idx] = segments
        self.stack['available'] += [stack_idx]
        self.stack['nb_segments'][stack_idx] = len(segments)  # this is at least one. Not all segments are necessarily available!
        # examples_used are already at zero.
        self.stack['seg_inds_still_avail'][stack_idx] = list(range(len(segments)))

    def pop_record_try_reload(self, stack_idx):
        # remove this stack_idx from available records
        self.stack['available'].pop(
            np.where(np.array(self.stack['available']) == stack_idx)[0][0])
        # %%%% Note: np.where has one idx only because idxs of records_available are unique.
        # %%%% If this is not the case this function will fail. This has the effect of
        # %%%% and additional check compared to list.index(val) ----
        # try to get new record for this idx.
        try:
            self.load_new_record_to_stack(stack_idx)
        except StopIteration:  # it might fail if record list is exhausted. In this case, leave the record unavailable.
            pass

    def __next__(self):

        if len(self.stack['available']) > 0:
            # chose a record to sample from
            stack_idx = np.random.choice(self.stack['available'])
            # number of available segments in this record
            if len(self.stack['seg_inds_still_avail'][stack_idx]) / \
                self.stack['nb_segments'][stack_idx] < (1. - self.record_exploit_rate):
                # remove this stack_idx from available records and try to reload a record instead.
                self.pop_record_try_reload(stack_idx)
                return self.__next__()
            else: # segments are still available in this record
                # chose one segment
                try:
                    seg_idx = np.random.choice(np.arange(len(
                        self.stack['seg_inds_still_avail'][stack_idx])))
                    self.stack['seg_inds_still_avail'][stack_idx].pop(seg_idx)
                    seg_data = self.stack['data'][stack_idx][seg_idx]
                    return seg_data
                except ValueError:
                    # getting a seg_idx can be impossible if all segments were available
                    # to iter and they have all been iterated and therefore the 'remaining data' list is empty.
                    # in this case, remove this stack_idx from available records
                    # and try to reload a record, and recurse.
                    self.pop_record_try_reload(stack_idx)
                    return self.__next__()
        else:
            raise StopIteration


class TUHBatchIterator(object):

    # A batch iterator that gets its data from an example iterator
    # and batches it.
    # It has a Queue of ready examples. (gotten from ExampleIterator)
    # The getting is done in a separate process (with Queue passed as argument to it)
    # such that nothing heavy is done in main process and the GPU never has
    # to wait for some preprocessing (loading of entire record for example)
    # in ExampleIterator. While the GPU is processing a batch and the batching part is idle,
    # the filling part can take place.

    def __init__(self,
                 record_files,
                 record_stack_size,
                 record_exploit_rate,
                 queue_size,  # in number of examples
                 batch_size):

        print('queue size', queue_size)
        print('batch_size', batch_size)
        print('##########################"')

        self.example_it = TUHExampleIterator(record_files,
                                             record_stack_size,
                                             record_exploit_rate)
        if queue_size < batch_size:
            raise ValueError('queue size should not be less than batch size. ')
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.all_data_iterated = Value('b', False)  # Use a multiprocessing.Value to safely share between main and child process
        self.q = Queue(maxsize=queue_size) # a multiprocessing.Queue is process-safe (and Thread-safe)
        self.load_p = Process(target=self.start_keep_loading,
                              args=(self.q, self.all_data_iterated))
        self.load_p.start()
        time.sleep(2)


    def start_keep_loading(self, q, finish):
        while not finish.value:
            if q.qsize() < self.queue_size:
                try:
                    element = next(self.example_it)
                    q.put(element)
                except StopIteration:
                    self.all_data_iterated.value = True
            else: # stack is full, sleep for a bit
                  # (but less than the time it takes to use a batch)
                time.sleep(0.01)

    def close(self, sleep_done=0.):
        # Tried to build safe closing :)
        # 1) empty queue if necessary such that process can be terminated.
        if not self.q.qsize() == 0:
            try:
                while True:
                    print()
                    self.q.get(timeout=5.)
            except Empty:
                print('Queue empty. ')
        #  2)close, potentially waiting a bit if process is not yet terminated.
        if not self.load_p.exitcode == 0:
            if sleep_done < 5.:
                time.sleep(1.)
                self.close(sleep_done + 1.)
            else:
                raise RuntimeError('Attempted to terminate loading process '
                               'load_p but it seems that it is still running... ')
        else:
            self.load_p.terminate()

    def __iter__(self):
        return self

    def __next__(self, batch_content=None):
        if not (self.all_data_iterated.value and self.q.qsize() < self.batch_size):
            still_todo = self.batch_size - len(batch_content) if batch_content else self.batch_size
            batch_content = batch_content if batch_content else []
            if self.q.qsize() >= still_todo:  # note: qsize is approximate so we do not rely too much on it
                try:
                    t0 = time.time()
                    for _ in range(still_todo):
                        batch_content.append(self.q.get(timeout=1.))
                    batch = np.stack(batch_content).astype(np.float32)
                    print(time.time() - t0)
                    return np.transpose(batch, (0, 2, 1))
                except Empty:
                    # (*) (if gpu processing is faster than loading. )
                    # in theory (and if queue if big enough) this should
                    # almost never happen except if cpu is busy with something else)
                    #  or is queue is about to terminate.
                    # => wait a bit for queue to fill up
                    time.sleep(0.03)
                    print('Processing faster than loading: '
                          'in __next__ sleep Empty q.get')
                    return self.__next__(batch_content)
            else:
                # same remark (*)
                time.sleep(0.03)
                print('GPU processing faster than loading: '
                      'in __next__ sleep q.size < still_todo')
                return self.__next__(batch_content)
        else:
            raise StopIteration








