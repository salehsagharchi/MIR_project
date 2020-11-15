class Compressor:
    def __init__(self):
        self.posintgList = []
        self.compressedPostingList = []
        self.module = 128
        self.module_size = 7
        self.SIZE_OF_INT = 4
        self.diff_memory_byte = 0
        self.SIZE_OF_BYTE = 8

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
        result = []
        self.compressedPostingList = []
        self.diff_memory_byte = 0
        for number in self.posintgList:
            result += self.encode_gama_codes(number)
        self.diff_memory_byte = len(result) - self.SIZE_OF_INT * len(self.posintgList)
        result = ['0'] * (self.SIZE_OF_BYTE - (len(result) % self.SIZE_OF_BYTE)) + result
        count = 0
        while count * self.SIZE_OF_BYTE < len(result):
            self.compressedPostingList += [int(''.join(result[count *
                                                              self.SIZE_OF_BYTE:(count + 1) * self.SIZE_OF_BYTE]),
                                               base=2)]
            count += 1
        self.diff_memory_byte = self.calculate_diff_memory_size(len(result) / self.SIZE_OF_BYTE, len(self.posintgList))
        return self.compressedPostingList

    @staticmethod
    def encode_gama_codes(number):
        if number == 1:
            return 0
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

    def save_to_file(self, path):
        f = open(path, "wb")
        f.write(bytearray(self.compressedPostingList))
        f.close()

    def load_from_file(self, path):
        f = open(path, "rb")
        self.compressedPostingList = f.read()
        f.close()


# a = Compressor()
# a.posintgList = [824, 5, 214577]
# print(a.var_byte_compress())
# print(a.diff_memory_byte)
# a.var_byte_decompress()
# print(a.posintgList)
print(a.posintgList)
print(a.gama_code_compress())
# print(a.gama_codes_decompress())
# a.save_to_file("s.txt")
# b = Compressor()
# b.load_from_file("s.txt")
# print(b.gama_codes_decompress())
