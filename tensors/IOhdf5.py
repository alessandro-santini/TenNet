def save_hdf5(class_handle, file_pointer,subgroup):
    file_pointer.create_group(subgroup)
    for idx, arr in enumerate(class_handle.tensors):
        file_pointer.create_dataset(subgroup+'/'+str(idx), shape=arr.shape, data=arr,compression='gzip',compression_opts=9)
def load_hdf5(class_handle, file_pointer, subgroup):
    for idx in range(class_handle.tensors):
        class_handle.tensors[idx] = file_pointer[subgroup+'/'+str(idx)][...].copy()