class Compressor:
    def __init__(self):
        self.posintgList = []
        self.compressedPostingList = []
        self.module = 128
        self.module_size = 7
        self.SIZE_OF_INT = 4
        self.diff_memory_byte = 0

    def var_byte_compress(self):
        size_of_Result = 0
        self.compressedPostingList = []
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


# a = Compressor()
# a.posintgList = [0, 824, 5, 214577]
# print(a.var_byte_compress())
# print(a.diff_memory_byte)
# a.var_byte_decompress()
# print(a.posintgList)