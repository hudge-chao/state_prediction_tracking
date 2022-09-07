import functools
import os

def file_cmp(f1: str, f2: str):
    seq_f1 = int(f1.split('.')[0])
    seq_f2 = int(f2.split('.')[0])
    if seq_f1 < seq_f2:
        return -1
    elif seq_f1 > seq_f2:
        return 1
    else:
        return 0

# path = 'samples/localmaps'
# path = 'localmaps'
path = 'samples'

dst = os.path.join(os.getcwd(), path)

dirs = os.listdir(dst)

print(dirs)

for dir in dirs:
    parent_path = os.path.join(dst, dir)
    print(parent_path)
    files = os.listdir(parent_path)
    file_type = files[0].split('.')[1]
    print(file_type)
    files.sort(key= functools.cmp_to_key(file_cmp))
    i = 0
    for file in files:
        old_file_name = parent_path + os.sep + file
        new_file_name = parent_path + os.sep + str(i) + '.' + file_type
        print(new_file_name)
        os.rename(old_file_name, new_file_name)
        print(old_file_name, '======>', new_file_name)
        i += 1

# files.sort(key= functools.cmp_to_key(file_cmp))

# print(len(files))
# print(files[0])
# print(int(files[0].split('.')[0]))

# i = 0
# for file in files:
#     old_file_name = dst + os.sep + file
#     new_file_name = dst + os.sep + str(i)+'.png'
#     os.rename(old_file_name, new_file_name)
#     print(old_file_name, '======>', new_file_name)
#     i += 1
