import re, gzip, tarfile, os, traceback

class TextBasicParser:
    @classmethod
    def load_content(cls, path_or_string, is_file=False, is_compressed=False):
        file_opener = gzip.open if is_compressed else open
        if is_file:
            with file_opener(path_or_string) as f:
                content = f.read()
        else:
            if is_compressed:
                content = path_or_string.decode("utf-8")
            else:
                content = path_or_string
        return content
    
    @classmethod
    def parse(cls, file_path, logger = None):
        if file_path.endswith('tar.gz'):
            members, stats = cls.parse_tar(file_path, logger)
        else:
            stats = {"total": 1, "errors": 0}
            file_id = os.path.basename(file_path).split('.')[0]
            if file_path.endswith('.gz'):
                content = cls.load_content(file_path, is_file=True, is_compressed=True)
            else:
                content = cls.load_content(file_path, is_file=True, is_compressed=False)
            members = [[file_id, file_path, content]]            
        return members, stats

    @classmethod
    def parse_tar(cls, file_path, logger = None):
        members = []
        stats = {"total": 0, "errors": 0}
        tar = tarfile.open(file_path, 'r:gz') 
        if logger != None: logger.info(f"The file {file_path} has { len([member for member in tar.getmembers() if not member.isdir()]) } documents")
        for member in tar.getmembers():
            if member.isdir(): continue
            stats['total'] += 1
            f=tar.extractfile(member)
            filename = os.path.join(file_path, member.path)
            try:            
                binarized_content = f.read()
                content = cls.load_content(binarized_content, is_file=False, is_compressed=True)
                file_id = os.path.basename(filename).split('.')[0]
                members.append([file_id, filename, content])
            except Exception as e:
                stats['errors'] += 1
                if logger != None: logger.error(f"There was a problem proccessing the file {filename} with the following error: {e}\n{traceback.format_exc()}")
        tar.close()
        return members, stats