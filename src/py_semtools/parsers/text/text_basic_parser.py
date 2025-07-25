import re, gzip, tarfile, os, traceback

class TextBasicParser:
    @classmethod
    def load_content(cls, path_or_string, is_file=False, is_compressed=False, options={'clean_type': 'hard'}):
        file_opener = gzip.open if is_compressed else open
        if is_file:
            with file_opener(path_or_string) as f:
                content = f.read()
        else:
            if is_compressed:
                content = path_or_string.decode("utf-8")
            else:
                content = path_or_string
        return cls.perform_soft_cleaning(content, type=options["clean_type"])
    
    @classmethod
    def parse(cls, file_path, logger = None, options={'clean_type': 'hard'}):
        if file_path.endswith('tar.gz'):
            members, stats = cls.parse_tar(file_path, logger)
        else:
            stats = {"total": 1, "errors": 0}
            file_id = os.path.basename(file_path).split('.')[0]
            if file_path.endswith('.gz'):
                content = cls.load_content(file_path, is_file=True, is_compressed=True, options=options)
            else:
                content = cls.load_content(file_path, is_file=True, is_compressed=False, options=options)
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

    @classmethod
    def extract_year(cls, text):
        match = re.search(r'.*(\d{4}).*', text)
        return int(match.group(1)) if match else None
    
    @classmethod
    def check_not_none_or_empty(cls, variable):
        if type(variable) != str: 
            condition = variable != None and variable.text != None and variable.text.strip().replace("&#x000a0;", "") != ""
        else:
            condition = variable != None and variable.strip().replace("&#x000a0;", "") != ""
        return condition

    @classmethod
    def perform_soft_cleaning(cls, raw_document, type="soft"):
        document = raw_document
        if type in ["soft", "hard"]:
            document = re.sub(r"([A-Za-z\(\)]+[ ]*)\n([ ]*[A-Z-a-z\(\)]+)", r"\1 \2", raw_document) #Removing nonsense newlines that broke the text and make information loss
            document = document.strip().replace("\r", "\n").replace("&#13", "\n").replace("\t", " ") #Replace carriage returns and tabs
            document = re.sub(r"\\[a-z]+(\[.+\])?(\{(.+)\})", r"\3", document) #Removing latex commands
            document = re.sub(r"[ ]+", r" ", document) #Removing additional whitespaces between words
        if type in ["hard"]:
            document = re.sub(r"([0-9]+)[\.\,]([0-9]+)", r"\1'\2", document) #Changing floating point numbers from 4.5 or 4,5 to 4'5
            document = re.sub(r"i\.?e\.?", "ie", document).replace("al.", "al ") #Changing i.e to ie and et al. to et al
        return document    