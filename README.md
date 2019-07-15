

## An iterator for TUG EEG data

Iterating batches from a very large EEG dataset such as TUG EEG into your deep learning model can require a bit of work to setup. I release this simple and efficient iterator to help in the process. 

In this example we want to yield random segments of given duration from TUH EEG records, such that these segments are 'shuffled' from all the dataset. 
In other words, we want to avoid iterating all segments from one patient and then iterating the next patient, because that would break the i.i.d. hypothesis for gradient-based learning. 

The provided class ```BatchIterator``` takes a list of edf records, and yields 'mixed' batches. The following parameters should also be given:
* ```record_stack_size``` specifies how many edf files to keep in the memory stack
* ```record_exploit_rate``` specifies which fraction of a record should be yielded before
this record is thrown away and a new record is loaded
* ```queue size``` specifies how many segments to keep in the batching queue. This does not need to be very big. 

Adjust these parameters to your hardware specs (reading speed, RAM memory, GPU). 

```BatchIterator``` uses a multiple processes and a Queue to make sure the GPU never has to wait for a batch :D


## Dependencies

numpy, mne, yaml