import os
import pickle


class Compressor:
    VAR_BYTE_MODE = 0
    GAMA_CODES_MODE = 1

    def __init__(self):
        self.posintgList = []
        self.compressedPostingList = []
        self.module = 128
        self.module_size = 7
        self.SIZE_OF_INT = 4
        self.diff_memory_byte = 0
        self.SIZE_OF_BYTE = 8
        self.VAR_BYTE_MODE = 0
        self.GAMA_CODES_MODE = 1
        self.posting_file = None
        self.term_file = None

    def var_byte_compress(self):
        size_of_Result = 0
        self.compressedPostingList = []
        self.diff_memory_byte = 0
        for number in self.posintgList:
            encoded_number = self.encode_var_byte(number)
            size_of_Result += len(encoded_number)
            self.compressedPostingList += encoded_number
        self.diff_memory_byte = self.calculate_diff_memory_size(size_of_Result, len(self.posintgList))
        return self.compressedPostingList

    def encode_var_byte(self, number):
        result = []
        # print(number)
        number = int(number)
        result += ((number % self.module) + 128).to_bytes(1, "little")
        number = number >> self.module_size
        while number != 0:
            result += (number % self.module).to_bytes(1, "little")
            number = number >> self.module_size
        result.reverse()
        return result

    def calculate_diff_memory_size(self, size_of_result, input_array_size):
        return input_array_size * self.SIZE_OF_INT - size_of_result

    def var_byte_decompress(self):
        self.posintgList = []
        tmp_number = []
        for current_byte in self.compressedPostingList:
            if current_byte & self.module != self.module:
                tmp_number += [current_byte]
            else:
                tmp_number += [current_byte - self.module]
                number = 0
                for i in range(len(tmp_number)):
                    number += (tmp_number[i]) << ((len(tmp_number) - i - 1) * self.module_size)
                tmp_number = []
                self.posintgList += [number]
        return self.posintgList

    def gama_code_compress(self):
        result = ''
        self.compressedPostingList = []
        self.diff_memory_byte = 0
        for number in self.posintgList:
            result += self.encode_gama_codes(number)
        self.diff_memory_byte = len(result) - self.SIZE_OF_INT * len(self.posintgList)
        result = '0' * (self.SIZE_OF_BYTE - (len(result) % self.SIZE_OF_BYTE)) + result
        count = 0
        while count * self.SIZE_OF_BYTE < len(result):
            self.compressedPostingList += [int(result[count * self.SIZE_OF_BYTE:(count + 1) * self.SIZE_OF_BYTE],
                                               base=2)]
            count += 1
        return self.compressedPostingList

    @staticmethod
    def encode_gama_codes(number):
        number = int(number)
        if number == 1:
            return '0'
        bin_number = bin(number)[3:]
        length_string = '1' * (len(bin_number))
        length_string += '0'
        return length_string + bin_number

    def gama_codes_decompress(self):
        self.posintgList = []
        buffer = ''
        for byte_number in self.compressedPostingList:
            byt_binary = bin(byte_number)[2:]
            buffer += (self.SIZE_OF_BYTE - len(byt_binary)) * '0' + byt_binary
        i = 0
        while i < len(buffer) and buffer[i] != '1':
            i += 1
        while i < len(buffer):
            count = 0
            while buffer[i] == '1':
                i += 1
                count += 1
            i += 1
            currentNumber = int('1' + buffer[i:i + count], base=2)
            self.posintgList += [currentNumber]
            i += count
        return self.posintgList

    def compress_arr(self, mode, arr):
        self.posintgList = arr
        if mode == self.VAR_BYTE_MODE:
            self.var_byte_compress()
        elif mode == self.GAMA_CODES_MODE:
            self.gama_code_compress()
        return self.compressedPostingList

    def decompress_arr(self, mode, compressed_arr):
        self.compressedPostingList = compressed_arr
        if mode == self.VAR_BYTE_MODE:
            self.var_byte_decompress()
        elif mode == self.GAMA_CODES_MODE:
            self.gama_codes_decompress()
        return self.posintgList

    def compress(self, mode, dic):
        for term in dic.keys():
            self.term_file.write(bytes(";", "utf-8"))
            self.term_file.write(bytes(term, "utf-8"))
            currentTerm = dic[term]
            currentTermKeys = list(currentTerm)
            self.posting_file.write(bytearray([0x41, 0x41]))
            self.posting_file.write(bytearray(self.compress_arr(mode, currentTermKeys)))
            for document in currentTermKeys:
                currentArr = currentTerm[document]
                compressedArr = self.compress_arr(mode, currentArr)
                self.posting_file.write(bytearray([0x42, 0x42]))
                self.posting_file.write(bytearray(compressedArr))
        self.posting_file.flush()
        self.term_file.flush()

    def decompress(self, mode, compressedDic):
        for term in compressedDic.keys():
            currentTerm = compressedDic[term]
            for document in currentTerm.keys():
                compressedArr = currentTerm[document]
                currentArr = self.decompress_arr(mode, compressedArr)
                compressedDic[term][document] = currentArr
        return compressedDic

    def save_to_file(self, dic, mode, posintg_path=".\\data\\postingListData", term_path=".\\data\\termData"):
        try:
            self.posting_file = open(posintg_path, 'wb')
            self.term_file = open(term_path, "wb")
            with open(posintg_path + 'WithoutCompress.pkl', 'wb') as f:
                pickle.dump(dic, f, pickle.DEFAULT_PROTOCOL)
                f.flush()
                f.close()
            self.compress(mode, dic)

        finally:
            self.term_file.close()
            self.posting_file.close()
            return os.path.getsize(posintg_path + 'WithoutCompress.pkl'), (os.path.getsize(posintg_path) +
                                                                           os.path.getsize(term_path))

    def load_from_file(self, mode, posting_path=".\\data\\postingListData", term_path=".\\data\\termData"):
        dic = {}
        try:
            self.term_file = open(term_path, "rb")
            terms = self.term_file.read().decode("utf-8").split(';')[1:]
            self.posting_file = open(posting_path, "rb")
            termsInformation = self.posting_file.read().split(bytearray([0x41, 0x41]))[1:]
            for i in range(len(termsInformation)):
                currentTermInfo = termsInformation[i].split(bytearray([0x42, 0x42]))
                documentsId = self.decompress_arr(mode, currentTermInfo[0])
                currentTermDic = {}
                for j in range(len(documentsId)):
                    currentTermDic[documentsId[j]] = self.decompress_arr(mode, currentTermInfo[j + 1])
                dic[terms[i]] = currentTermDic
        finally:
            self.term_file.close()
            self.posting_file.close()
            return dic
