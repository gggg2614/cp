s = "0.1,0.2"
parts = s.split(',')

# 将每个部分转换为浮点数
floats = [float(part) for part in parts]

# 分别赋值给变量 a 和 b
a, b = floats

print("a =", a)
print("b =", b)