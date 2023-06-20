import os
import filecmp
   # 如果两边路径的头文件都存在，进行比较
status = filecmp.cmp("/Users/dengjiaming/Downloads/ProgrammingTest/CorrectOutput/SingleAscending.txt", "/Users/dengjiaming/Downloads/ProgrammingTest/CorrectOutput/SingleAscending.txt")
        # 为True表示两文件相同
if status:
    print("files are the same")
        # 为False表示文件不相同
else:
    print("files are different")
    # 如果两边路径头文件不都存在，抛异常
