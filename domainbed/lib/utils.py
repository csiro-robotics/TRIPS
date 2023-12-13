import os

def write_to_txt(content, file_path, file_name):
    complete_file_path = os.path.join(file_path, file_name)
    with open("{0}".format(complete_file_path), 'w') as f: 
        for key, value in content.items(): 
            f.write('%s:%s\n' % (key, value))