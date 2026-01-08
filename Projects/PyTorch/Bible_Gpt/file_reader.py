class File_Reader:
    """
    Utility class for reading files
    """
    @staticmethod
    def extract_text(filepath):
        """
        Methiid that extracts text from a file
        """
        try:
            with open(filepath, "r", encoding="utf8") as txt:
                text = txt.read()
                return text
        except FileNotFoundError:
            print ("No Valid File found")
